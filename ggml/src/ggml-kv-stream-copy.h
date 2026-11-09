#pragma once
#include "ggml-backend.h"
#include "ggml-kv-stream.h"
#include <vector>
#include <algorithm>
#include <array>

struct ggml_kv_stream_copy_rectangle {
    ggml_backend_buffer_t source = nullptr;
    size_t source_offset = 0, source_stride = 0;
    size_t destination_offset = 0, destination_stride = 0;
    size_t row_bytes = 0, rows = 0;
};

struct ggml_kv_stream_gather_row {
    uint32_t source = UINT32_MAX;
    size_t token = 0;
};

// Native columns keep their order, including empty cells. The caller masks empty columns.
// Views may omit unloaded intervals; every referenced row must be available in its private source.
// Destination is one exact 256-row encoded tile. Failure preserves rectangles and all payloads.
GGML_API bool ggml_kv_stream_gather_make(const ggml_kv_stream_span_plan_view * sources, size_t count,
        const std::vector<ggml_kv_stream_gather_row> & columns, size_t first,
        ggml_backend_buffer_t destination, std::vector<ggml_kv_stream_copy_rectangle> & rectangles);

// One bounded 256-row K/V tile. Validate every byte range before clearing or copying scratch.
inline bool ggml_kv_stream_gather_validate(ggml_backend_buffer_t destination,
        const ggml_kv_stream_copy_rectangle * rectangles, size_t count) {
    if (!destination || count > 512 || (count && !rectangles)) return false;
    const size_t capacity=ggml_backend_buffer_get_size(destination);
    const auto base=uintptr_t(ggml_backend_buffer_get_base(destination));
    if (!base || !capacity || capacity > UINTPTR_MAX-base) return false;
    struct range { size_t begin, end; };
    std::array<range,512> writes;
    size_t used=0;
    for (size_t i=0;i<count;++i) {
        const auto & r=rectangles[i];
        if (!r.source || !r.row_bytes || !r.rows || r.rows > 256 || r.rows > writes.size()-used ||
                r.source_stride < r.row_bytes || r.destination_stride < r.row_bytes ||
                (r.rows > 1 && (r.source_stride > INT32_MAX || r.destination_stride > INT32_MAX)) ||
                r.destination_offset > capacity || r.row_bytes > capacity-r.destination_offset ||
                r.rows-1 > (capacity-r.destination_offset-r.row_bytes)/r.destination_stride) return false;
        const size_t source_size=ggml_backend_buffer_get_size(r.source);
        const auto source_base=uintptr_t(ggml_backend_buffer_get_base(r.source));
        if (!source_base || source_size > UINTPTR_MAX-source_base || r.source_offset > source_size ||
                r.row_bytes > source_size-r.source_offset ||
                r.rows-1 > (source_size-r.source_offset-r.row_bytes)/r.source_stride) return false;
        const auto source_begin=source_base+r.source_offset;
        const auto source_end=source_begin+(r.rows-1)*r.source_stride+r.row_bytes;
        if (source_begin < base+capacity && base < source_end) return false;
        for (size_t row=0;row<r.rows;++row) {
            const size_t first=r.destination_offset+row*r.destination_stride;
            writes[used++]={first,first+r.row_bytes};
        }
    }
    std::sort(writes.begin(),writes.begin()+used,[](const range & a,const range & b) {return a.begin<b.begin;});
    for (size_t i=1;i<used;++i) if (writes[i].begin < writes[i-1].end) return false;
    return true;
}

// Clamp a logical run to both the physical ring boundary and the caller's span ceiling.
inline size_t ggml_kv_stream_contiguous_pages(size_t block, size_t blocks, size_t slots, size_t limit) {
    if (block >= blocks || !slots || !limit) return 0;
    return std::min({blocks-block,slots-block%slots,limit});
}

// Owner-thread bookkeeping only; GPU completion is established by backend events, not these states.
class ggml_kv_stream_copy_state {
public:
    enum class phase { empty, queued, acquired, released };
    explicit ggml_kv_stream_copy_state(size_t slots) : slots(slots,phase::empty) {}
    // A new run starts only after the previous run has drained.
    bool begin() { if (active || slots.empty()) return false; active = true; return true; }
    bool running() const { return active; }
    bool can_queue(size_t i) const { return active && i < slots.size() && (slots[i] == phase::empty || slots[i] == phase::released); }
    // Validate the entire physical run before changing any slot's ownership.
    bool can_queue_span(size_t first, size_t count) const {
        if (!active || !count || first >= slots.size() || count > slots.size()-first) return false;
        for (size_t i = 0; i < count; ++i) if (!can_queue(first+i)) return false;
        return true;
    }
    bool queue_span(size_t first, size_t count) {
        if (!can_queue_span(first,count)) return false;
        for (size_t i = 0; i < count; ++i) slots[first+i] = phase::queued;
        return true;
    }
    bool recycled(size_t i) const { return active && i < slots.size() && slots[i] == phase::released; }
    bool waiting(size_t i) const { return active && i < slots.size() && slots[i] == phase::queued; }
    bool held(size_t i) const { return active && i < slots.size() && slots[i] == phase::acquired; }
    // Publish transitions only after the corresponding event/copy commands have been submitted.
    bool queue(size_t i) { if (!can_queue(i)) return false; slots[i] = phase::queued; return true; }
    bool acquire(size_t i) { if (!waiting(i)) return false; slots[i] = phase::acquired; return true; }
    bool release(size_t i) { if (!held(i)) return false; slots[i] = phase::released; return true; }
    // Call only after both streams drain, including on cancellation or failure.
    void drained() { active = false; for (auto & slot : slots) slot = phase::empty; }
private:
    std::vector<phase> slots;
    bool active = false;
};

struct ggml_kv_stream_copy_stats {
    size_t bytes = 0, calls = 0;
    size_t ready_fences = 0, ready_waits = 0, consumed_fences = 0, consumed_waits = 0;
};
struct ggml_kv_stream_copy_range { size_t first = 0, count = 0; };


struct ggml_kv_stream_copy_feedback {
    bool available = false;
    uint64_t samples = 0, misses = 0;
    // Version 9: one sample per streamed layer, missed when any immutable page was late at layer entry.
    uint64_t layer_samples = 0, layer_misses = 0;
    size_t bytes = 0, timed_bytes = 0, peak_slots = 0, instrumentation_bytes = 0;
    // The first eligible upload's copy interval excludes dependency waits; timed_bytes counts live payload, not padding.
    // elapsed_ms is the whole host run window. Version 7 samples each eligible upload batch at its first consumption.
    double copy_ms = 0, elapsed_ms = 0;
};

struct ggml_kv_stream_copy_snapshot {
    uint64_t id = 0;
    ggml_kv_stream_copy_feedback value;
};

// Owner-thread snapshot admission, not a GPU fence. Uncollected slots are never overwritten.
class ggml_kv_stream_feedback_slots {
public:
    static constexpr size_t capacity = 2, none = SIZE_MAX;
    explicit ggml_kv_stream_feedback_slots(uint64_t counter = 0) : counter(counter) {}
    // Exhaustion skips measurement, not inference; zero identifies an unmeasured run.
    bool begin(bool eligible = true) {
        if (active) return false;
        active = true; selected = none; latest = 0;
        if (eligible && counter != UINT64_MAX) for (size_t i = 0; i < capacity; ++i) if (!slots[i].id) {
            selected = i; latest = ++counter; slots[i].id = latest; break;
        }
        return true;
    }
    size_t current() const { return active ? selected : none; }
    uint64_t latest_id() const { return latest; }
    uint64_t id(size_t i) const { return i < capacity ? slots[i].id : 0; }
    size_t seal() {
        if (!active) return none;
        active = false;
        if (selected != none) slots[selected].pending = true;
        return selected;
    }
    size_t oldest() const {
        size_t result = none;
        for (size_t i = 0; i < capacity; ++i) if (slots[i].pending &&
                (result == none || slots[i].id < slots[result].id)) result = i;
        return result;
    }
    // Retire only after the backend established completion and copied the result to its caller.
    bool retire(size_t i) {
        if (i == none || i != oldest()) return false;
        slots[i] = {}; return true;
    }
private:
    struct slot { uint64_t id = 0; bool pending = false; };
    std::array<slot,capacity> slots{};
    uint64_t counter = 0, latest = 0;
    size_t selected = none;
    bool active = false;
};

// Optional registry "ggml_backend_kv_stream_copy_ops". Copy calls enqueue work; drain/free and idle reconfiguration may wait.
// Caller holds the device lease/pin and immutable host content until drain; backend outlives the handle.
// Owner-thread only and outside active CUDA capture; ready() is observation, not a host-content lifetime fence.
struct ggml_kv_stream_copy_ops {
    uint32_t version;
    // Retain buffers and create event resources, without allocating KV storage or submitting work.
    void * (*create)(ggml_backend_t, ggml_backend_buffer_t device, ggml_backend_buffer_t host,
                     const ggml_kv_stream_shape &, size_t slots);
    bool (*begin)(void *);
    bool (*enqueue)(void *, size_t slot, const void * k, const void * v, size_t live_tokens, size_t padded_tokens);
    bool (*ready)(void *, size_t slot);
    bool (*acquire)(void *, size_t slot);
    bool (*release)(void *, size_t slot);
    // Cancellation abandons unconsumed slots only after already-submitted work completes.
    void (*drain)(void *);
    void (*free)(void *);
    // Version 2: one contiguous K transfer and one V transfer; never crosses the physical ring boundary.
    bool (*enqueue_span)(void *, size_t first_slot, const void * k, const void * v, size_t live_tokens, size_t padded_tokens);
    // Actual submitted payload bytes and memcpy calls, excluding padding fills; reset by begin().
    ggml_kv_stream_copy_stats (*stats)(void *);
    // Version 3: caller has already synchronized every encoded-slot reader; no queued consumer fence is needed.
    bool (*release_completed)(void *, size_t slot);
    // Version 4: opt-in measurement may allocate bounded diagnostic storage; change only while idle.
    bool (*measure)(void *, bool enable);
    // Acquire a consumed span before its mandatory ready-event waits. Version 7 probes its upload batch only once.
    bool (*acquire_span)(void *, size_t first_slot, size_t count);
    // Legacy completed getter; may wait for counter readback after drain. New consumers use poll_feedback.
    ggml_kv_stream_copy_feedback (*feedback)(void *);
    // Version 5: current/latest run identity, or zero when measurement was skipped due to backpressure.
    uint64_t (*feedback_id)(void *);
    // Single-consumer FIFO: at most feedback_slots::capacity uncollected snapshots; each poll retires one.
    // Nonblocking, valid also during a later run. False leaves output unchanged.
    bool (*poll_feedback)(void *, ggml_kv_stream_copy_snapshot *);
    // Version 6: owner declares whether this run and each upload are useful prefetch-quality samples.
    bool (*begin_with_feedback)(void *, bool eligible);
    bool (*enqueue_span_with_feedback)(void *, size_t first_slot, const void * k, const void * v,
                                      size_t live_tokens, size_t padded_tokens, bool eligible);
    // Version 8: make later copy-stream submissions wait for current producer-stream work.
    bool (*fence_producer)(void *);
    // Version 9: snapshot all immutable requests for one layer before acquire_span inserts any waits.
    // Each range must exactly name one queued upload batch; an empty list is not a sample.
    bool (*probe_layer)(void *, const ggml_kv_stream_copy_range * ranges, size_t count);
    // Version 10: one final-consumer fence covers a complete transfer span.
    bool (*release_span)(void *, size_t first_slot, size_t count);
    // Version 11: reserve native transport/feedback resources without retaining KV views or submitting work.
    // Backend outlives prepared resources and queues. A bank admits serial replacements, not concurrent owners.
    void * (*prepare)(ggml_backend_t, size_t max_slots, bool feedback);
    void (*free_prepared)(void *);
    // Borrow retained resources; never silently allocate missing capacity or unprepared feedback storage.
    void * (*create_prepared)(ggml_backend_t, void * prepared, ggml_backend_buffer_t device,
                             ggml_backend_buffer_t host, const ggml_kv_stream_shape &, size_t slots);
    // Version 12: gather encoded rows on the producer stream. Initialize clears the tile before its first packet.
    // Later packets preserve earlier rows; the caller proves complete native-wave coverage before attention.
    // No allocation or host wait. Caller retains every source/destination lease until completion.
    bool (*gather)(ggml_backend_t, ggml_backend_buffer_t destination,
                   const ggml_kv_stream_copy_rectangle *, size_t count, bool initialize) = nullptr;
};
using ggml_kv_stream_copy_ops_get = const ggml_kv_stream_copy_ops * (*)();
