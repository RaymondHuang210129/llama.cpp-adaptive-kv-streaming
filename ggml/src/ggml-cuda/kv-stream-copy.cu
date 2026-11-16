#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include "kv-stream-partial.cuh"
#include "common.cuh"
#include "../ggml-backend-impl.h"
#include "../ggml-kv-stream-registered-copy.h"
#include <memory>
#include <chrono>
#include <unordered_set>

const ggml_kv_stream_registered_copy_ops * ggml_cuda_kv_stream_registered_copy_ops();

namespace {
// Preserve the first initialization error before rollback releases any owned resources.
static bool initialization_ok(cudaError_t status, const char * operation) {
    if (status == cudaSuccess) return true;
    GGML_LOG_ERROR("CUDA KV copy resource initialization failed: %s: %s (code %d)\n",
        operation, cudaGetErrorString(status), int(status));
    (void) cudaGetLastError();
    return false;
}

// Failed CUDA creation may write a non-null invalid handle; publish only successful outputs.
template<class Handle, class Create, class... Args>
static bool create_resource(Handle & owned, const char * operation, Create create, Args... args) {
    Handle candidate = nullptr;
    if (!initialization_ok(create(&candidate, args...), operation)) return false;
    owned = candidate;
    return true;
}

struct feedback_storage {
    ggml_backend_buffer_t buffer = nullptr, host = nullptr;
    cudaEvent_t start = nullptr, end = nullptr;
    cudaStream_t readback = nullptr;
    std::array<cudaEvent_t,ggml_kv_stream_feedback_slots::capacity> ready{};
    // No queued work exists until a measurement view borrows these resources.
    ~feedback_storage() {
        for (auto event : ready) if (event) CUDA_CHECK(cudaEventDestroy(event));
        if (readback) CUDA_CHECK(cudaStreamDestroy(readback));
        if (start) CUDA_CHECK(cudaEventDestroy(start));
        if (end) CUDA_CHECK(cudaEventDestroy(end));
        ggml_backend_buffer_free(host); ggml_backend_buffer_free(buffer);
    }
};

struct measurement {
    std::shared_ptr<feedback_storage> storage;
    ggml_backend_buffer_t buffer, host;
    cudaEvent_t start, end;
    cudaStream_t readback;
    bool used = false;
    struct snapshot {
        cudaEvent_t ready = nullptr;
        ggml_kv_stream_copy_feedback value;
        bool immediate = false;
    };
    std::array<snapshot,ggml_kv_stream_feedback_slots::capacity> snapshots{};
    ggml_kv_stream_feedback_slots slots;
    ggml_kv_stream_copy_snapshot last;
    struct upload {
        uint64_t ticket = 0;
        size_t first = 0, pages = 0;
        bool eligible = false, sampled = false;
    };
    std::vector<upload> uploads;
    uint64_t ticket = 0;
    size_t occupied = 0;
    bool enabled = false;
    ggml_kv_stream_copy_feedback result;
    std::chrono::steady_clock::time_point begin;
    measurement(size_t slots, std::shared_ptr<feedback_storage> storage) : storage(std::move(storage)),
        buffer(this->storage->buffer), host(this->storage->host), start(this->storage->start),
        end(this->storage->end), readback(this->storage->readback), uploads(slots) {
        for (size_t i = 0; i < snapshots.size(); ++i) snapshots[i].ready = this->storage->ready[i];
    }
    static constexpr size_t counters = 5, readback_counters = 4;
    // The queue drains and selects its device before these diagnostic resources are destroyed.
    ~measurement() {
        if (used) CUDA_CHECK(cudaStreamSynchronize(readback));
    }
    uint64_t * data() const { return static_cast<uint64_t *>(ggml_backend_buffer_get_base(buffer)); }
    uint64_t * active_data() const { return data()+slots.current()*(uploads.size()+counters); }
    uint64_t * host_data(size_t bank) const {
        return static_cast<uint64_t *>(ggml_backend_buffer_get_base(host))+readback_counters*bank;
    }
    bool active() const { return enabled && slots.current() != slots.none; }
    // The owner has drained both streams. An empty run has no recorded timing events.
    void collect() {
        if (!result.timed_bytes) return;
        float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms,start,end));
        result.copy_ms = ms;
    }
};

// Tickets avoid resetting a reused flag after the next consumer has already checked it.
static __global__ void publish_ready(uint64_t * flags, size_t first, size_t count, uint64_t ticket) {
    for (size_t i = 0; i < count; ++i)
        atomicExch(reinterpret_cast<unsigned long long *>(flags+first+i),static_cast<unsigned long long>(ticket));
}

// Sample on the consuming stream, not when the host submits the future dependency.
static __global__ void sample_deadline(uint64_t * flags, size_t counters, size_t first, size_t count, uint64_t ticket) {
    bool missing = false;
    for (size_t i = 0; i < count; ++i)
        missing |= atomicAdd(reinterpret_cast<unsigned long long *>(flags+first+i),0ULL) != ticket;
    ++flags[counters]; flags[counters+1] += missing;
}

// All range probes run before any consumer wait; their OR is one complete-layer deadline.
static __global__ void sample_layer_range(
        uint64_t * flags, size_t counters, size_t first, size_t count, uint64_t ticket) {
    bool missing = false;
    for (size_t i = 0; i < count; ++i)
        missing |= atomicAdd(reinterpret_cast<unsigned long long *>(flags+first+i),0ULL) != ticket;
    if (missing) atomicExch(reinterpret_cast<unsigned long long *>(flags+counters+4),1ULL);
}

static __global__ void finish_layer_sample(uint64_t * flags, size_t counters) {
    ++flags[counters+2];
    flags[counters+3] += atomicExch(reinterpret_cast<unsigned long long *>(flags+counters+4),0ULL) != 0;
}


struct copy_resources {
    ggml_backend_cuda_context * context;
    cudaStream_t stream = nullptr;
    cudaEvent_t producer = nullptr;
    std::vector<cudaEvent_t> ready, consumed;
    std::shared_ptr<feedback_storage> feedback;
    void * owner = nullptr;
    bool prepared;
    copy_resources(ggml_backend_cuda_context * context, size_t slots, bool prepared) :
        context(context), ready(slots), consumed(slots), prepared(prepared) {}
    // Queues retire all pointer users before releasing their final shared bank reference.
    ~copy_resources() {
        ggml_cuda_set_device(context->device);
        feedback.reset();
        for (auto event : ready) if (event) CUDA_CHECK(cudaEventDestroy(event));
        for (auto event : consumed) if (event) CUDA_CHECK(cudaEventDestroy(event));
        if (producer) CUDA_CHECK(cudaEventDestroy(producer));
        if (stream) CUDA_CHECK(cudaStreamDestroy(stream));
    }
};

struct copy_queue {
    std::shared_ptr<copy_resources> resources;
    ggml_backend_cuda_context * context;
    ggml_backend_buffer_t device = nullptr, host = nullptr;
    ggml_kv_stream_registered_copy_source source;
    std::vector<ggml_kv_stream_registered_copy_source> additional_sources;
    ggml_backend_buffer_type_t source_type = nullptr;
    std::unordered_set<ggml_backend_buffer_t> registrations;
    ggml_kv_stream_layout page, ring;
    ggml_kv_stream_copy_state state;
    ggml_kv_stream_copy_stats statistics;
    cudaStream_t stream = nullptr;
    cudaEvent_t producer = nullptr;
    std::vector<cudaEvent_t> ready, consumed;
    std::vector<uint8_t> completed;
    static constexpr size_t no_fence = SIZE_MAX;
    std::vector<size_t> ready_owner,consumed_owner,ready_refs,consumed_refs;
    std::unique_ptr<measurement> timing;
    bool initialized = false;
    copy_queue(ggml_backend_cuda_context * context, size_t slots) : context(context),state(slots),
        ready(slots),consumed(slots),completed(slots,false),ready_owner(slots,no_fence),
        consumed_owner(slots,no_fence),ready_refs(slots),consumed_refs(slots) {}
    // Retire all pointer users before destroying events or releasing pinned/device backing.
    ~copy_queue() {
        ggml_cuda_set_device(context->device);
        if (initialized) {
            if (!resources->prepared || resources->owner == this) CUDA_CHECK(cudaStreamSynchronize(stream));
            // Backing may also have direct tensor users that never entered this copy queue.
            CUDA_CHECK(cudaStreamSynchronize(context->stream()));
        }
        timing.reset();
        if (resources && resources->owner == this) resources->owner = nullptr;
        for (auto * buffer : registrations) ggml_backend_buffer_free(buffer);
        ggml_backend_buffer_free(host); ggml_backend_buffer_free(device);
    }
};

// Allocate O(ring slots) diagnostics through the explicit device-local buffer type, never the KV region.
static std::shared_ptr<feedback_storage> make_feedback(ggml_backend_cuda_context * context, size_t slots) {
    try {
        if (slots > SIZE_MAX/(sizeof(uint64_t)*ggml_kv_stream_feedback_slots::capacity)-
                measurement::counters) return {};
        auto m = std::make_shared<feedback_storage>();
        // Lazy kernel loading can synchronize the context. Resolve kernels before any producer/consumer gate.
        cudaFuncAttributes attributes;
        if (!initialization_ok(cudaFuncGetAttributes(&attributes,publish_ready),"cudaFuncGetAttributes(publish_ready)") ||
                !initialization_ok(cudaFuncGetAttributes(&attributes,sample_deadline),"cudaFuncGetAttributes(sample_deadline)") ||
                !initialization_ok(cudaFuncGetAttributes(&attributes,sample_layer_range),"cudaFuncGetAttributes(sample_layer_range)") ||
                !initialization_ok(cudaFuncGetAttributes(&attributes,finish_layer_sample),"cudaFuncGetAttributes(finish_layer_sample)")) return {};
        m->buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_device_buffer_type(context->device),
            (slots+measurement::counters)*ggml_kv_stream_feedback_slots::capacity*sizeof(uint64_t));
        m->host = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_host_buffer_type(),
            measurement::readback_counters*ggml_kv_stream_feedback_slots::capacity*sizeof(uint64_t));
        if (!m->buffer || !m->host) {
            GGML_LOG_ERROR("CUDA KV copy feedback %s allocation failed\n", m->buffer ? "host" : "device");
            return {};
        }
        cudaPointerAttributes host_attributes{};
        if (!initialization_ok(cudaPointerGetAttributes(&host_attributes,ggml_backend_buffer_get_base(m->host)),
                "cudaPointerGetAttributes(feedback host)")) return {};
        if (host_attributes.type != cudaMemoryTypeHost) {
            GGML_LOG_ERROR("CUDA KV copy feedback requires pinned host storage\n"); return {};
        }
        if (!create_resource(m->readback,"cudaStreamCreateWithFlags(feedback)",cudaStreamCreateWithFlags,cudaStreamNonBlocking)) return {};
        for (auto & event : m->ready) if (!create_resource(event,
                "cudaEventCreateWithFlags(feedback ready)",cudaEventCreateWithFlags,cudaEventDisableTiming)) return {};
        if (!create_resource(m->start,"cudaEventCreate(feedback start)",[](cudaEvent_t * event) { return cudaEventCreate(event); }) ||
                !create_resource(m->end,"cudaEventCreate(feedback end)",[](cudaEvent_t * event) { return cudaEventCreate(event); })) return {};
        return m;
    } catch (const std::bad_alloc &) {
        GGML_LOG_ERROR("CUDA KV copy feedback host metadata allocation failed\n"); return {};
    }
}

// Prepared banks never allocate feedback storage or native handles after admission.
static bool measure(void * handle, bool enable) {
    if (!handle) return false;
    auto & q = *static_cast<copy_queue *>(handle);
    if (q.state.running()) return false;
    ggml_cuda_set_device(q.context->device);
    q.timing.reset();
    if (!enable) {
        if (!q.resources->prepared) q.resources->feedback.reset();
        return true;
    }
    auto storage = q.resources->feedback;
    if (!storage) {
        if (q.resources->prepared) return false;
        storage = make_feedback(q.context, q.ready.size());
        if (!storage) return false;
    }
    try { q.timing = std::make_unique<measurement>(q.ready.size(), std::move(storage)); }
    catch (const std::bad_alloc &) { return false; }
    if (q.timing) { q.timing->enabled = enable; q.timing->result = {}; }
    return true;
}

// Reserve the maximum native capacity once; the bank contains no KV pointers or arena leases.
static std::shared_ptr<copy_resources> make_resources(ggml_backend_cuda_context * context, size_t slots, bool feedback, bool prepared) {
    if (!slots || slots > SIZE_MAX/(8*sizeof(size_t))) return {};
    ggml_cuda_set_device(context->device);
    try {
        if (prepared && !context->streams[context->device][context->curr_stream_no] &&
                !create_resource(context->streams[context->device][context->curr_stream_no],
                    "cudaStreamCreateWithFlags(backend producer)",cudaStreamCreateWithFlags,cudaStreamNonBlocking)) return {};
        auto resources = std::make_shared<copy_resources>(context, slots, prepared);
        if (!create_resource(resources->stream,"cudaStreamCreateWithFlags(copy)",cudaStreamCreateWithFlags,cudaStreamNonBlocking) ||
                !create_resource(resources->producer,"cudaEventCreateWithFlags(producer)",cudaEventCreateWithFlags,cudaEventDisableTiming)) return {};
        for (size_t i = 0; i < slots; ++i) {
            if (!create_resource(resources->ready[i],"cudaEventCreateWithFlags(ready)",cudaEventCreateWithFlags,cudaEventDisableTiming) ||
                    !create_resource(resources->consumed[i],"cudaEventCreateWithFlags(consumed)",cudaEventCreateWithFlags,cudaEventDisableTiming)) return {};
        }
        if (feedback) {
            resources->feedback = make_feedback(context, slots);
            if (!resources->feedback) return {};
        }
        return resources;
    } catch (const std::bad_alloc &) { GGML_LOG_ERROR("CUDA KV copy resource host allocation failed\n"); return {}; }
}

// Return an independent owner; borrowing queues keep its native bank alive after owner release.
static void * prepare(ggml_backend_t backend, size_t slots, bool feedback) {
    if (!backend || !ggml_backend_is_cuda(backend)) return nullptr;
    auto resources = make_resources(static_cast<ggml_backend_cuda_context *>(backend->context), slots, feedback, true);
    if (!resources) return nullptr;
    try { return new std::shared_ptr<copy_resources>(std::move(resources)); }
    catch (const std::bad_alloc &) { return nullptr; }
}

// Drop only this owner's reference; do not retire live borrowing queues.
static void free_prepared(void * prepared) { delete static_cast<std::shared_ptr<copy_resources> *>(prepared); }

// Source spans must remain inside the retained pinned allocation; zero or wrapping ranges are invalid.
static bool source_range(const copy_queue & q, const void * pointer, size_t bytes) {
    const auto p=uintptr_t(pointer);
    if (!pointer || !bytes || bytes>UINTPTR_MAX-p) return false;
    const auto contains=[&](uintptr_t base,size_t capacity) {
        return p>=base && p-base<=capacity && bytes<=capacity-(p-base);
    };
    const auto base=uintptr_t(q.host ? ggml_backend_buffer_get_base(q.host) : q.source.base);
    const size_t capacity=q.host ? ggml_backend_buffer_get_size(q.host) : q.source.bytes;
    if (contains(base,capacity)) return true;
    for (const auto & source:q.additional_sources) if (contains(uintptr_t(source.base),source.bytes)) return true;
    return false;
}

// Allocate only stream/event bookkeeping. KV bytes already belong to the caller's device and host buffers.
static void * create_queue(ggml_backend_t backend, ggml_backend_buffer_t device, ggml_backend_buffer_t host,
        const ggml_kv_stream_shape & shape, size_t slots, std::shared_ptr<copy_resources> resources,
        const ggml_kv_stream_registered_copy_source * source = nullptr) {
    if (!backend || !ggml_backend_is_cuda(backend) || !device || !slots || shape.page_tokens <= 0 ||
            uint64_t(shape.page_tokens) > SIZE_MAX || slots > SIZE_MAX/size_t(shape.page_tokens) ||
            (source ? host != nullptr : !host || !ggml_backend_buffer_is_host(host))) return nullptr;
    auto * ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    if (resources && (resources->context != ctx || slots > resources->ready.size())) return nullptr;
    if (ggml_backend_buffer_get_type(device) != ggml_backend_cuda_device_buffer_type(ctx->device)) return nullptr;
    ggml_kv_stream_layout page, ring;
    if (ggml_kv_stream_layout_make(shape,size_t(shape.page_tokens),page).status != ggml_kv_stream_status::success ||
            ggml_kv_stream_layout_make(shape,slots*size_t(shape.page_tokens),ring).status != ggml_kv_stream_status::success ||
            ring.bytes > ggml_backend_buffer_get_size(device) || !page.bytes || page.v_offset != page.k_bytes ||
            page.k_bytes%shape.alignment || page.v_bytes%shape.alignment) return nullptr;
    const auto base = uintptr_t(ggml_backend_buffer_get_base(device));
    if (!base || base%shape.alignment || base > UINTPTR_MAX-ring.bytes) return nullptr;
    ggml_cuda_set_device(ctx->device);
    ggml_backend_buffer_type_t source_type = nullptr;
    if (source) {
        const auto address = reinterpret_cast<uintptr_t>(source->base);
        if (!address || !source->bytes || address%shape.alignment || source->bytes > UINTPTR_MAX-address) return nullptr;
        using factory = ggml_backend_buffer_type_t (*)(int);
        auto select = reinterpret_cast<factory>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(ggml_backend_get_device(backend)), "ggml_backend_cuda_kv_host_buffer_type"));
        source_type = select ? select(ctx->device) : nullptr;
        if (!source_type) return nullptr;
    } else {
        cudaPointerAttributes attributes = {};
        if (!initialization_ok(cudaPointerGetAttributes(&attributes,ggml_backend_buffer_get_base(host)),
                "cudaPointerGetAttributes(copy host)")) return nullptr;
        if (attributes.type != cudaMemoryTypeHost) {
            GGML_LOG_ERROR("CUDA KV copy requires pinned host storage\n"); return nullptr;
        }
    }
    try {
        if (!resources) resources = make_resources(ctx, slots, false, false);
        if (!resources) return nullptr;
        auto q = std::make_unique<copy_queue>(ctx,slots);
        q->resources = std::move(resources);
        q->page = page; q->ring = ring;
        q->device = ggml_backend_buffer_retain(device); q->host = ggml_backend_buffer_retain(host);
        q->source = source ? *source : ggml_kv_stream_registered_copy_source{};
        q->source_type = source_type;
        q->stream = q->resources->stream; q->producer = q->resources->producer;
        for (size_t i = 0; i < slots; ++i) {
            q->ready[i] = q->resources->ready[i]; q->consumed[i] = q->resources->consumed[i];
        }
        q->initialized = true;
        return q.release();
    } catch (const std::bad_alloc &) {
        GGML_LOG_ERROR("CUDA KV copy host metadata allocation failed\n"); return nullptr;
    }
}

// Preserve direct callers that have not supplied a prepared bank.
static void * create(ggml_backend_t backend, ggml_backend_buffer_t device, ggml_backend_buffer_t host,
        const ggml_kv_stream_shape & shape, size_t slots) {
    return create_queue(backend, device, host, shape, slots, {});
}

// Validate new KV views against retained native capacity without allocating driver resources.
static void * create_prepared(ggml_backend_t backend, void * prepared, ggml_backend_buffer_t device,
        ggml_backend_buffer_t host, const ggml_kv_stream_shape & shape, size_t slots) {
    if (!prepared) return nullptr;
    return create_queue(backend, device, host, shape, slots, *static_cast<std::shared_ptr<copy_resources> *>(prepared));
}

static void * create_registered(ggml_backend_t backend, void * prepared, ggml_backend_buffer_t device,
        const ggml_kv_stream_registered_copy_source & source, const ggml_kv_stream_shape & shape, size_t slots) {
    return create_queue(backend, device, nullptr, shape, slots, prepared ?
        *static_cast<std::shared_ptr<copy_resources> *>(prepared) : std::shared_ptr<copy_resources>{}, &source);
}

static void * create_registered_multi(ggml_backend_t backend,void * prepared,ggml_backend_buffer_t device,
        const ggml_kv_stream_registered_copy_source * sources,size_t count,
        const ggml_kv_stream_shape & shape,size_t slots) {
    if (!sources || !count || count>ggml_kv_stream_registered_copy_max_sources || !shape.alignment) return nullptr;
    for (size_t i=0;i<count;++i) {
        const auto base=uintptr_t(sources[i].base);const auto bytes=sources[i].bytes;
        if (!base || !bytes || base%shape.alignment || bytes>UINTPTR_MAX-base) return nullptr;
        for (size_t j=0;j<i;++j) {
            const auto other=uintptr_t(sources[j].base);
            if (base<other+sources[j].bytes && other<base+bytes) return nullptr;
        }
    }
    auto * queue=static_cast<copy_queue *>(create_registered(backend,prepared,device,sources[0],shape,slots));
    if (!queue) return nullptr;
    try {queue->additional_sources.assign(sources+1,sources+count);return queue;}
    catch (const std::bad_alloc &) {delete queue;return nullptr;}
}

static bool fragments_valid(const copy_queue & q, const ggml_kv_stream_registered_copy_fragment * parts,
        size_t count, size_t expected) {
    if (!parts || !count || count > expected || !q.source_type) return false;
    size_t total = 0;
    uintptr_t previous = 0;
    for (size_t i = 0; i < count; ++i) {
        const auto & part = parts[i];
        if (!part.owner || ggml_backend_buffer_get_type(part.owner) != q.source_type || !part.bytes || part.bytes > expected-total) return false;
        const auto base = uintptr_t(ggml_backend_buffer_get_base(part.owner));
        const size_t capacity = ggml_backend_buffer_get_size(part.owner);
        if (part.offset > capacity || part.bytes > capacity-part.offset || part.offset > UINTPTR_MAX-base ||
                part.bytes > UINTPTR_MAX-base-part.offset ||
                !source_range(q,reinterpret_cast<const void *>(base+part.offset),part.bytes) ||
                (i && (part.offset || base != previous)) ||
                (i+1 < count && part.bytes != capacity-part.offset)) return false;
        previous = base+part.offset+part.bytes;
        total += part.bytes;
    }
    return total == expected;
}

static bool retain_fragments(copy_queue & q, const ggml_kv_stream_registered_copy_fragment * k, size_t nk,
        const ggml_kv_stream_registered_copy_fragment * v, size_t nv) {
    try {
        std::vector<ggml_backend_buffer_t> added;
        for (bool value : {false,true}) {
            const auto * parts = value ? v : k;
            const size_t count = value ? nv : nk;
            for (size_t i = 0; i < count; ++i) {
                auto * owner = parts[i].owner;
                if (!q.registrations.count(owner) && std::find(added.begin(),added.end(),owner) == added.end()) added.push_back(owner);
            }
        }
        if (added.empty()) return true;
        if (added.size() > q.registrations.max_size()-q.registrations.size()) return false;
        q.registrations.reserve(q.registrations.size()+added.size());
        for (auto * owner : added) if (q.registrations.insert(owner).second) ggml_backend_buffer_retain(owner);
        return true;
    } catch (const std::bad_alloc &) { return false; }
}

// Extend a running copy window with work published after its original begin fence.
static bool fence_producer(void * handle) {
    if (!handle) return false;
    auto & q=*static_cast<copy_queue *>(handle);
    if (!q.state.running()) return false;
    ggml_cuda_set_device(q.context->device);
    cudaStreamCaptureStatus status;
    CUDA_CHECK(cudaStreamIsCapturing(q.context->stream(),&status));
    if (status != cudaStreamCaptureStatusNone) return false;
    CUDA_CHECK(cudaEventRecord(q.producer,q.context->stream()));
    CUDA_CHECK(cudaStreamWaitEvent(q.stream,q.producer,0));
    return true;
}

// Fence earlier compute producers before any ring writes; active CUDA capture is unsupported.
static bool begin_with_feedback(void * handle, bool eligible) {
    if (!handle) return false;
    auto & q = *static_cast<copy_queue *>(handle);
    ggml_cuda_set_device(q.context->device);
    cudaStreamCaptureStatus status;
    CUDA_CHECK(cudaStreamIsCapturing(q.context->stream(),&status));
    if (status != cudaStreamCaptureStatusNone || (q.resources->owner && q.resources->owner != &q) || !q.state.begin()) return false;
    q.resources->owner = &q;
    q.statistics = {};
    if (q.timing && q.timing->enabled) {
        auto & m = *q.timing;
        m.result = {}; m.ticket = 0; m.occupied = 0;
        GGML_ASSERT(m.slots.begin(eligible));
        m.begin = std::chrono::steady_clock::now();
        m.result.instrumentation_bytes = ggml_backend_buffer_get_size(m.buffer);
        if (m.active()) {
            m.used = true;
            // One clear per window, without touching a previous bank whose readback may still be pending.
            CUDA_CHECK(cudaMemsetAsync(m.active_data(),0,(q.ready.size()+measurement::counters)*sizeof(uint64_t),q.context->stream()));
        }
    }
    return fence_producer(handle);
}

// Legacy callers declare a fully eligible run.
static bool begin(void * handle) { return begin_with_feedback(handle,true); }

// Queue independent K/V copies and finite tail padding after the previous final consumer retires.
static bool enqueue_span_impl(void * handle, size_t slot, const void * k, const void * v, size_t live, size_t padded, bool eligible,
        const ggml_kv_stream_registered_copy_fragment * kp, size_t nk,
        const ggml_kv_stream_registered_copy_fragment * vp, size_t nv) {
    if (!handle) return false;
    auto & q = *static_cast<copy_queue *>(handle);
    if (!live || live > padded || padded > q.ring.tokens) return false;
    const size_t count = padded/q.page.tokens+(padded%q.page.tokens != 0);
    if (!q.state.can_queue_span(slot,count)) return false;
    if (q.source_type) {
        if (k || v || !fragments_valid(q,kp,nk,live*q.page.k_token_bytes) ||
                !fragments_valid(q,vp,nv,live*q.page.v_token_bytes)) return false;
    } else if (kp || vp || nk || nv || !source_range(q,k,live*q.page.k_token_bytes) || !source_range(q,v,live*q.page.v_token_bytes)) return false;
    auto * m = q.timing && q.timing->active() ? q.timing.get() : nullptr;
    const auto ready_it=std::find(q.ready_refs.begin(),q.ready_refs.end(),size_t(0));
    if (ready_it == q.ready_refs.end()) return false;
    const size_t ready_fence=size_t(ready_it-q.ready_refs.begin());
    for (size_t i=0;i<count;++i)
        if (q.ready_owner[slot+i] != copy_queue::no_fence) return false;
    if (m && m->ticket == UINT64_MAX) return false;
    if (q.source_type && !retain_fragments(q,kp,nk,vp,nv)) return false;
    ggml_cuda_set_device(q.context->device);
    for (size_t i=0;i<count;++i) if (q.state.recycled(slot+i) && !q.completed[slot+i]) {
        const size_t fence=q.consumed_owner[slot+i];
        if (fence == copy_queue::no_fence || !q.consumed_refs[fence]) return false;
        bool first=true;
        for (size_t j=0;j<i;++j) first &= q.consumed_owner[slot+j] != fence;
        if (first) {
            CUDA_CHECK(cudaStreamWaitEvent(q.stream,q.consumed[fence],0));
            ++q.statistics.consumed_waits;
        }
    }
    for (size_t i=0;i<count;++i) if (q.state.recycled(slot+i)) {
        if (!q.completed[slot+i]) {
            const size_t fence=q.consumed_owner[slot+i];
            GGML_ASSERT(fence != copy_queue::no_fence && q.consumed_refs[fence]);
            --q.consumed_refs[fence];
            q.consumed_owner[slot+i]=copy_queue::no_fence;
        }
        q.completed[slot+i]=false;
    }
    auto * base = static_cast<char *>(ggml_backend_buffer_get_base(q.device));
    // One first-upload sample per run; keep every deadline probe, but avoid per-upload timing bookkeeping.
    const bool timed = m && eligible && !m->result.timed_bytes;
    if (timed) CUDA_CHECK(cudaEventRecord(m->start,q.stream));
    for (int value = 0; value < 2; ++value) {
        const size_t stride = value ? q.page.v_token_bytes : q.page.k_token_bytes;
        const size_t offset = value ? q.ring.v_offset+slot*q.page.v_bytes : slot*q.page.k_bytes;
        if (q.source_type) {
            const auto * parts = value ? vp : kp;
            const size_t count = value ? nv : nk;
            size_t copied = 0;
            for (size_t i = 0; i < count; ++i) {
                const auto & part = parts[i];
                const auto * source = static_cast<const char *>(ggml_backend_buffer_get_base(part.owner))+part.offset;
                CUDA_CHECK(cudaMemcpyAsync(base+offset+copied,source,part.bytes,cudaMemcpyHostToDevice,q.stream));
                copied += part.bytes;
                q.statistics.bytes += part.bytes; ++q.statistics.calls;
            }
        } else {
            CUDA_CHECK(cudaMemcpyAsync(base+offset,value ? v : k,live*stride,cudaMemcpyHostToDevice,q.stream));
            q.statistics.bytes += live*stride; ++q.statistics.calls;
        }
        if (padded > live) CUDA_CHECK(cudaMemsetAsync(base+offset+live*stride,0,(padded-live)*stride,q.stream));
    }
    if (m) {
        if (timed) {
            CUDA_CHECK(cudaEventRecord(m->end,q.stream));
            m->result.timed_bytes = live*(q.page.k_token_bytes+q.page.v_token_bytes);
        }
        const uint64_t ticket = ++m->ticket;
        for (size_t i = 0; i < count; ++i) m->uploads[slot+i] = {ticket,slot,count,eligible,false};
        publish_ready<<<1,1,0,q.stream>>>(m->active_data(),slot,count,ticket);
        CUDA_CHECK(cudaGetLastError());
        m->occupied += count;
        m->result.peak_slots = std::max(m->result.peak_slots,m->occupied);
    }
    CUDA_CHECK(cudaEventRecord(q.ready[ready_fence],q.stream));
    ++q.statistics.ready_fences;
    q.ready_refs[ready_fence]=count;
    for (size_t i=0;i<count;++i) q.ready_owner[slot+i]=ready_fence;
    return q.state.queue_span(slot,count);
}

static bool enqueue_span_with_feedback(void * handle, size_t slot, const void * k, const void * v, size_t live, size_t padded, bool eligible) {
    return enqueue_span_impl(handle,slot,k,v,live,padded,eligible,nullptr,0,nullptr,0);
}

static bool enqueue_registered(void * handle, size_t slot,
        const ggml_kv_stream_registered_copy_fragment * k, size_t nk,
        const ggml_kv_stream_registered_copy_fragment * v, size_t nv, size_t live, size_t padded, bool eligible) {
    if (!handle || !static_cast<copy_queue *>(handle)->source_type) return false;
    return enqueue_span_impl(handle,slot,nullptr,nullptr,live,padded,eligible,k,nk,v,nv);
}

// Legacy uploads are immutable snapshots; preserve their full deadline coverage.
static bool enqueue_span(void * handle, size_t slot, const void * k, const void * v, size_t live, size_t padded) {
    return enqueue_span_with_feedback(handle,slot,k,v,live,padded,true);
}

// Keep v1's one-slot admission boundary for existing callers.
static bool enqueue(void * handle, size_t slot, const void * k, const void * v, size_t live, size_t padded) {
    if (!handle || padded > static_cast<copy_queue *>(handle)->page.tokens) return false;
    return enqueue_span(handle,slot,k,v,live,padded);
}

// Report actual DMA submissions, not an inferred count from the consumer's plan.
static ggml_kv_stream_copy_stats stats(void * handle) {
    return handle ? static_cast<copy_queue *>(handle)->statistics : ggml_kv_stream_copy_stats{};
}

// Observation does not transfer ownership or replace the compute stream's mandatory event wait.
static bool ready(void * handle, size_t slot) {
    if (!handle) return false;
    auto & q = *static_cast<copy_queue *>(handle);
    if (q.state.held(slot)) return true;
    if (!q.state.waiting(slot)) return false;
    const size_t fence=q.ready_owner[slot];
    if (fence == copy_queue::no_fence || !q.ready_refs[fence]) return false;
    ggml_cuda_set_device(q.context->device);
    const auto result = cudaEventQuery(q.ready[fence]);
    if (result == cudaErrorNotReady) return false;
    CUDA_CHECK(result); return true;
}

// Snapshot one layer before any ready-event waits can hide its original deadline.
static bool probe_layer(void * handle, const ggml_kv_stream_copy_range * ranges, size_t count) {
    if (!handle || !ranges || !count) return false;
    auto & q = *static_cast<copy_queue *>(handle);
    if (!q.state.running()) return false;
    auto * m = q.timing && q.timing->active() ? q.timing.get() : nullptr;
    for (size_t r = 0; r < count; ++r) {
        const auto range = ranges[r];
        if (!range.count || range.first >= q.ready.size() || range.count > q.ready.size()-range.first) return false;
        for (size_t i = 0; i < range.count; ++i) if (!q.state.waiting(range.first+i)) return false;
        if (!m) continue;
        const auto & upload = m->uploads[range.first];
        if (!upload.ticket || upload.first != range.first || upload.pages != range.count) return false;
        for (size_t i = 1; i < range.count; ++i) {
            if (m->uploads[range.first+i].ticket != upload.ticket) return false;
        }
    }
    if (!m) return true;
    ggml_cuda_set_device(q.context->device);
    for (size_t r = 0; r < count; ++r) {
        const auto & upload = m->uploads[ranges[r].first];
        sample_layer_range<<<1,1,0,q.context->stream()>>>(
            m->active_data(),q.ready.size(),ranges[r].first,ranges[r].count,upload.ticket);
        CUDA_CHECK(cudaGetLastError());
    }
    finish_layer_sample<<<1,1,0,q.context->stream()>>>(m->active_data(),q.ready.size());
    CUDA_CHECK(cudaGetLastError());
    return true;
}

// Queue the consumer dependency without blocking the host on transfer completion.
static bool acquire_span(void * handle, size_t slot, size_t count) {
    if (!handle) return false;
    auto & q = *static_cast<copy_queue *>(handle);
    if (!count || slot >= q.ready.size() || count > q.ready.size()-slot) return false;
    for (size_t i = 0; i < count; ++i) if (!q.state.waiting(slot+i)) return false;
    const size_t fence=q.ready_owner[slot];
    if (fence == copy_queue::no_fence || !q.ready_refs[fence]) return false;
    for (size_t i=1;i<count;++i)
        if (q.ready_owner[slot+i] != fence) return false;
    auto * m = q.timing && q.timing->active() ? q.timing.get() : nullptr;
    // One consumed span must come from a single enqueue_span call.
    if (m) for (size_t i = 1; i < count; ++i) if (m->uploads[slot+i].ticket != m->uploads[slot].ticket) return false;
    ggml_cuda_set_device(q.context->device);
    if (m && m->uploads[slot].eligible && !m->uploads[slot].sampled) {
        auto & upload = m->uploads[slot];
        sample_deadline<<<1,1,0,q.context->stream()>>>(m->active_data(),q.ready.size(),upload.first,1,upload.ticket);
        CUDA_CHECK(cudaGetLastError());
        if (slot == upload.first && count == upload.pages) {
            upload.sampled = true; // All members become acquired below; none can be acquired again without a new upload.
        } else {
            // Remember the first probe on every remaining member, even if the first slot is reused meanwhile.
            for (size_t i = 0; i < upload.pages; ++i) m->uploads[upload.first+i].sampled = true;
        }
    }
    CUDA_CHECK(cudaStreamWaitEvent(q.context->stream(),q.ready[fence],0));
    ++q.statistics.ready_waits;
    for (size_t i=0;i<count;++i) {
        GGML_ASSERT(q.state.acquire(slot+i));
        GGML_ASSERT(q.ready_owner[slot+i] == fence && q.ready_refs[fence]);
        q.ready_owner[slot+i]=copy_queue::no_fence;
        --q.ready_refs[fence];
    }
    return true;
}

// Preserve the one-page interface while using the same deadline sampling boundary.
static bool acquire(void * handle, size_t slot) { return acquire_span(handle,slot,1); }

// Every encoded-slot read must be submitted before recording this final-consumer event.
static bool release_span(void * handle, size_t slot, size_t count) {
    if (!handle) return false;
    auto & q = *static_cast<copy_queue *>(handle);
    if (!count || slot >= q.consumed.size() || count > q.consumed.size()-slot) return false;
    for (size_t i=0;i<count;++i)
        if (!q.state.held(slot+i) || q.consumed_owner[slot+i] != copy_queue::no_fence) return false;
    const auto found=std::find(q.consumed_refs.begin(),q.consumed_refs.end(),size_t(0));
    if (found == q.consumed_refs.end()) return false;
    const size_t fence=size_t(found-q.consumed_refs.begin());
    ggml_cuda_set_device(q.context->device);
    ++q.statistics.consumed_fences;
    CUDA_CHECK(cudaEventRecord(q.consumed[fence],q.context->stream()));
    q.consumed_refs[fence]=count;
    for (size_t i=0;i<count;++i) {
        q.consumed_owner[slot+i]=fence;
        q.completed[slot+i]=false;
        if (q.timing && q.timing->active()) --q.timing->occupied;
        GGML_ASSERT(q.state.release(slot+i));
    }
    return true;
}

static bool release(void * handle, size_t slot) { return release_span(handle,slot,1); }

// The caller already completed every reader; retain the queued-fence path for asynchronous consumers.
static bool release_completed(void * handle, size_t slot) {
    if (!handle) return false;
    auto & q = *static_cast<copy_queue *>(handle);
    if (!q.state.held(slot)) return false;
    if (q.consumed_owner[slot] != copy_queue::no_fence) return false;
    q.completed[slot] = true;
    if (q.timing && q.timing->active()) --q.timing->occupied;
    return q.state.release(slot);
}

// Cancellation drops logical ownership only after both streams complete outstanding work.
static void drain(void * handle) {
    if (!handle) return;
    auto & q = *static_cast<copy_queue *>(handle);
    ggml_cuda_set_device(q.context->device);
    CUDA_CHECK(cudaStreamSynchronize(q.stream)); CUDA_CHECK(cudaStreamSynchronize(q.context->stream()));
    if (q.state.running() && q.timing && q.timing->enabled) {
        auto & m = *q.timing;
        const size_t bank = m.slots.current();
        if (bank != m.slots.none) {
            m.collect();
            auto & snapshot = m.snapshots[bank];
            snapshot.value = m.result;
            snapshot.value.bytes = q.statistics.bytes;
            snapshot.value.elapsed_ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-m.begin).count();
            snapshot.immediate = !q.statistics.bytes;
            if (!snapshot.immediate) {
                CUDA_CHECK(cudaMemcpyAsync(m.host_data(bank),
                    m.data()+bank*(q.ready.size()+measurement::counters)+q.ready.size(),
                    measurement::readback_counters*sizeof(uint64_t),cudaMemcpyDeviceToHost,m.readback));
                CUDA_CHECK(cudaEventRecord(snapshot.ready,m.readback));
            }
        }
        m.slots.seal();
    }
    std::fill(q.ready_owner.begin(),q.ready_owner.end(),copy_queue::no_fence);
    std::fill(q.consumed_owner.begin(),q.consumed_owner.end(),copy_queue::no_fence);
    std::fill(q.ready_refs.begin(),q.ready_refs.end(),size_t(0));
    std::fill(q.consumed_refs.begin(),q.consumed_refs.end(),size_t(0));
    std::fill(q.completed.begin(),q.completed.end(),false);
    q.state.drained();
}

// Inspect only the oldest snapshot; pending telemetry never blocks the compute or copy streams.
static bool poll_feedback(void * handle, ggml_kv_stream_copy_snapshot * output) {
    if (!handle || !output) return false;
    auto & q = *static_cast<copy_queue *>(handle);
    if (!q.timing || !q.timing->enabled) return false;
    auto & m = *q.timing;
    const size_t bank = m.slots.oldest();
    if (bank == m.slots.none) return false;
    ggml_cuda_set_device(q.context->device);
    auto result = m.snapshots[bank].value;
    if (!m.snapshots[bank].immediate) {
        const auto status = cudaEventQuery(m.snapshots[bank].ready);
        if (status == cudaErrorNotReady) return false;
        CUDA_CHECK(status);
        result.samples = m.host_data(bank)[0]; result.misses = m.host_data(bank)[1];
        result.layer_samples = m.host_data(bank)[2];
        result.layer_misses = m.host_data(bank)[3];
    }
    result.available = true;
    m.last = {m.slots.id(bank),result};
    GGML_ASSERT(m.slots.retire(bank));
    *output = m.last; return true;
}

// Identify the current/latest measured window; zero means measurement was disabled or skipped.
static uint64_t feedback_id(void * handle) {
    if (!handle) return 0;
    const auto & q = *static_cast<copy_queue *>(handle);
    return q.timing && q.timing->enabled ? q.timing->slots.latest_id() : 0;
}

// Preserve the v4 completed getter for old callers. The runtime uses the nonblocking v5 poll instead.
static ggml_kv_stream_copy_feedback feedback(void * handle) {
    if (!handle) return {};
    auto & q = *static_cast<copy_queue *>(handle);
    if (!q.timing || q.state.running() || !feedback_id(handle)) return {};
    auto & m = *q.timing;
    ggml_cuda_set_device(q.context->device);
    while (m.slots.oldest() != m.slots.none) {
        const size_t bank = m.slots.oldest();
        if (!m.snapshots[bank].immediate) CUDA_CHECK(cudaEventSynchronize(m.snapshots[bank].ready));
        ggml_kv_stream_copy_snapshot snapshot;
        GGML_ASSERT(poll_feedback(handle,&snapshot));
    }
    return m.last.id == feedback_id(handle) ? m.last.value : ggml_kv_stream_copy_feedback{};
}

// Destruction also drains, so no retained allocation can outlive its final GPU use.
static void destroy(void * handle) { delete static_cast<copy_queue *>(handle); }

// Reorder encoded rows in borrowed attention scratch without changing native KV pages.
struct gather_rectangle {
    const char * source;
    char * destination;
    uint32_t source_stride,destination_stride,row_bytes,rows;
};

static constexpr size_t gather_batch_size=96;
struct gather_batch {
    gather_rectangle rectangles[gather_batch_size];
};
static_assert(sizeof(gather_batch)<=3072,"gather parameters fit the legacy CUDA parameter limit");

// Small scattered rows share a launch; their validated destination ranges do not overlap.
static __global__ void gather_rectangles(const gather_batch batch) {
    const gather_rectangle rectangle=batch.rectangles[blockIdx.x];
    if (!((uintptr_t(rectangle.source)|uintptr_t(rectangle.destination)|rectangle.source_stride|
            rectangle.destination_stride|rectangle.row_bytes)&15)) {
        const size_t width=rectangle.row_bytes/sizeof(uint4),words=width*rectangle.rows;
        for (size_t index=threadIdx.x;index<words;index+=blockDim.x) {
            const size_t row=index/width,column=index%width;
            reinterpret_cast<uint4 *>(rectangle.destination+row*rectangle.destination_stride)[column]=
                reinterpret_cast<const uint4 *>(rectangle.source+row*rectangle.source_stride)[column];
        }
        return;
    }
    const size_t bytes=rectangle.row_bytes*rectangle.rows;
    for (size_t index=threadIdx.x;index<bytes;index+=blockDim.x) {
        const size_t row=index/rectangle.row_bytes,column=index%rectangle.row_bytes;
        rectangle.destination[row*rectangle.destination_stride+column]=
            rectangle.source[row*rectangle.source_stride+column];
    }
}

static bool gather(ggml_backend_t backend,ggml_backend_buffer_t destination,
        const ggml_kv_stream_copy_rectangle * rectangles,size_t count,bool initialize) {
    if (!backend || !ggml_backend_is_cuda(backend) ||
            !ggml_kv_stream_gather_validate(destination,rectangles,count)) return false;
    auto & ctx=*static_cast<ggml_backend_cuda_context *>(backend->context);
    auto * type=ggml_backend_cuda_device_buffer_type(ctx.device);
    if (ggml_backend_buffer_get_type(destination)!=type) return false;
    for (size_t i=0;i<count;++i) if (ggml_backend_buffer_get_type(rectangles[i].source)!=type) return false;
    ggml_cuda_set_device(ctx.device);
    cudaStreamCaptureStatus capture;
    CUDA_CHECK(cudaStreamIsCapturing(ctx.stream(),&capture));
    if (capture!=cudaStreamCaptureStatusNone) return false;
    auto * target=static_cast<char *>(ggml_backend_buffer_get_base(destination));
    if (initialize) CUDA_CHECK(cudaMemsetAsync(target,0,ggml_backend_buffer_get_size(destination),ctx.stream()));
    gather_batch batch{};
    size_t pending=0;
    const auto flush=[&] {
        if (pending) {
            gather_rectangles<<<pending,256,0,ctx.stream()>>>(batch);
            CUDA_CHECK(cudaGetLastError());pending=0;
        }
    };
    for (size_t i=0;i<count;++i) {
        const auto & r=rectangles[i];
        const auto * source=static_cast<const char *>(ggml_backend_buffer_get_base(r.source))+r.source_offset;
        const size_t bytes=r.row_bytes*r.rows;
        if (r.rows<=16 && bytes<=65536) {
            batch.rectangles[pending++]={source,target+r.destination_offset,uint32_t(r.rows==1 ? r.row_bytes : r.source_stride),
                uint32_t(r.rows==1 ? r.row_bytes : r.destination_stride),uint32_t(r.row_bytes),uint32_t(r.rows)};
            if (pending==gather_batch_size) flush();
            continue;
        }
        flush();
        if (r.rows==1 || (r.source_stride==r.row_bytes && r.destination_stride==r.row_bytes)) {
            CUDA_CHECK(cudaMemcpyAsync(target+r.destination_offset,source,bytes,cudaMemcpyDeviceToDevice,ctx.stream()));
        } else {
            CUDA_CHECK(cudaMemcpy2DAsync(target+r.destination_offset,r.destination_stride,source,r.source_stride,
                r.row_bytes,r.rows,cudaMemcpyDeviceToDevice,ctx.stream()));
        }
    }
    flush();
    return true;
}
} // namespace

// Expose the CUDA adapter through an opaque, backend-neutral ownership contract.
const ggml_kv_stream_copy_ops * ggml_cuda_kv_stream_copy_ops() {
    static const ggml_kv_stream_copy_ops ops{12,create,begin,enqueue,ready,acquire,release,drain,destroy,enqueue_span,stats,release_completed,measure,acquire_span,feedback,feedback_id,poll_feedback,begin_with_feedback,enqueue_span_with_feedback,fence_producer,probe_layer,release_span,prepare,free_prepared,create_prepared,gather};
    return &ops;
}

const ggml_kv_stream_registered_copy_ops * ggml_cuda_kv_stream_registered_copy_ops() {
    static const ggml_kv_stream_registered_copy_ops ops{2,create_registered,enqueue_registered,create_registered_multi};
    return &ops;
}
#endif
