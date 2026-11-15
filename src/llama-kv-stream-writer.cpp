#include "llama-kv-stream-writer.h"
#include "llama-memory-executor-cuda.h"
#include "ggml-cpp.h"
#include "../ggml/src/ggml-backend-execution.h"

#include <algorithm>
#include <limits>

struct write_graph {
    ggml_backend_buffer_ptr source_owner;
    ggml_context_ptr context;
    const void * data = nullptr;
    ggml_type type;
    size_t width = 0, rows = 0;
    ggml_tensor * output = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * root = nullptr;
    // Captures retire before metadata and source-buffer ownership disappear.
    std::unique_ptr<llama_memory_cuda_executor> execution;
};

struct llama_kv_stream_writer::implementation {
    ggml_backend_t backend;
    ggml_backend_buffer_ptr buffer;
    ggml_kv_stream_shape shape;
    bool cpu;
    size_t maximum = 0, capacity = 0, scratch_bytes = 0;
    ggml_context_ptr context;
    ggml_tensor * roots[2] = {};
    ggml_tensor * routed_roots[2] = {};
    size_t routed_capacity = 0;
    std::shared_ptr<ggml_backend> host_backend;
    ggml_backend_event_t host_ready[2] = {}, host_consumed[2] = {};
    bool host_busy[2] = {};
    ggml_tensor * indices = nullptr;
    // Plans drain before these destination registrations are released during destruction.
    std::vector<ggml_backend_buffer_ptr> destinations;
    // Cached executions drain before this upload storage is released.
    std::vector<int64_t> index_data;
    std::array<std::vector<std::unique_ptr<write_graph>>,2> plans;
    std::array<std::vector<std::unique_ptr<write_graph>>,2> cached;
    llama_kv_stream_write_stats last;

    ~implementation() {
        if (host_backend) ggml_backend_synchronize(host_backend.get());
        for (auto * event : host_ready) if (event) ggml_backend_event_free(event);
        for (auto * event : host_consumed) if (event) ggml_backend_event_free(event);
    }

    bool prepare_host_backend() {
        if (host_backend) return true;
        auto * device=ggml_backend_get_device(backend);
        std::shared_ptr<ggml_backend> next(ggml_backend_dev_init(device,nullptr),ggml_backend_free);
        if (!next) return false;
        for (size_t side=0;side<2;++side) {
            host_ready[side]=ggml_backend_event_new(device);host_consumed[side]=ggml_backend_event_new(device);
            if (!host_ready[side] || !host_consumed[side]) {
                for (auto & event : host_ready) if (event) {ggml_backend_event_free(event);event=nullptr;}
                for (auto & event : host_consumed) if (event) {ggml_backend_event_free(event);event=nullptr;}
                return false;
            }
            ggml_backend_event_record(host_ready[side],next.get());ggml_backend_event_record(host_consumed[side],next.get());
        }
        ggml_backend_synchronize(next.get());host_backend=std::move(next);return true;
    }

    size_t width(bool value) const { return size_t(value ? shape.head_dim_v : shape.head_dim_k)*size_t(shape.heads); }
    ggml_type type(bool value) const { return ggml_type(value ? shape.type_v : shape.type_k); }
    size_t row_bytes(bool value) const { return ggml_row_size(type(value), int64_t(width(value))); }

    // Immutable per-tile metadata remains valid until every queued launch reaches its completion event.
    write_graph * prepare(const ggml_tensor * source, bool value, size_t tile, size_t offset, size_t count,bool routed=false,bool cache_source=false) {
        auto * root=routed ? routed_roots[value] : roots[value];
        const size_t root_rows=routed ? routed_capacity : capacity;
        auto * owner=source->view_src ? source->view_src->buffer : source->buffer;
        auto * data=static_cast<char *>(source->data)+offset*source->nb[1];
        const auto storage=[](ggml_backend_buffer_t buffer) {
            while (buffer->parent) buffer=buffer->parent;
            return buffer;
        };
        // Extra cached sources must retain the writer's own parent, not another GPU allocation.
        const bool shared=cache_source && !cpu && !ggml_backend_buft_is_execution(ggml_backend_buffer_get_type(owner)) &&
            storage(owner)==storage(buffer.get());
        auto & cache=shared ? cached[value] : plans[value];
        const auto matches=[&](const auto & plan) {
            return plan && (shared ? ggml_backend_buffer_get_type(plan->source_owner.get())==ggml_backend_buffer_get_type(owner) :
                plan->source_owner.get()==owner) && plan->data==data && plan->type==type(value) &&
                plan->rows==count && plan->width==width(value) && plan->root==root;
        };
        if (shared) {
            const auto found=std::find_if(cache.begin(),cache.end(),matches);
            if (found!=cache.end()) {
                auto * result=found->get();std::rotate(found,found+1,cache.end());return result;
            }
            if (cache.size()==64) cache.erase(cache.begin());
            cache.emplace_back();
        } else {
            if (cache.size()<=tile) cache.resize(tile+1);
            if (matches(cache[tile])) return cache[tile].get();
        }
        auto & plan=shared ? cache.back() : cache[tile];
        plan.reset();
        auto next=std::make_unique<write_graph>();
        next->source_owner.reset(ggml_backend_buffer_retain(owner));
        next->context.reset(ggml_init({65536,nullptr,true}));
        if (!next->context) return nullptr;
        next->data=data; next->type=type(value); next->rows=count; next->width=width(value);next->root=root;
        auto * input=ggml_new_tensor_2d(next->context.get(),GGML_TYPE_F32,int64_t(width(value)),int64_t(count));
        if (ggml_backend_tensor_alloc(owner,input,data) != GGML_STATUS_SUCCESS) return nullptr;
        auto * dst=ggml_view_2d(next->context.get(),root,int64_t(width(value)),int64_t(root_rows),row_bytes(value),0);
        auto * idx=ggml_view_1d(next->context.get(),indices,int64_t(count),0);
        if (ggml_backend_view_init(dst) != GGML_STATUS_SUCCESS || ggml_backend_view_init(idx) != GGML_STATUS_SUCCESS) return nullptr;
        next->output=ggml_set_rows(next->context.get(),dst,input,idx);
        if (!ggml_backend_supports_op(backend,next->output) || ggml_backend_view_init(next->output) != GGML_STATUS_SUCCESS) return nullptr;
        next->graph=ggml_new_graph_custom(next->context.get(),32,false);
        ggml_build_forward_expand(next->graph,next->output);
        if (!cpu) {
            next->execution=std::make_unique<llama_memory_cuda_executor>(backend);
            if (!next->execution->bind(next->graph,{},1)) return nullptr;
        }
        plan=std::move(next);
        return plan.get();
    }
};

// The cached executor drains and retires before roots in the shared scratch context are released.
llama_kv_stream_writer::~llama_kv_stream_writer() {
    // Downloads follow the encoding graph's own completion marker.
    if (impl && !impl->destinations.empty()) ggml_backend_synchronize(impl->backend);
}

bool llama_kv_stream_writer::requirements(ggml_backend_buffer_type_t type, const ggml_kv_stream_shape & shape,
        size_t max_rows, size_t & bytes) {
    if (!type || !max_rows || max_rows > INT32_MAX || shape.page_tokens <= 0) return false;
    ggml_kv_stream_layout row;
    if (ggml_kv_stream_layout_make(shape,1,row).status != ggml_kv_stream_status::success) return false;
    const uint64_t dim=uint64_t(std::max(shape.head_dim_k,shape.head_dim_v));
    const uint64_t maximum=std::min(uint64_t(INT32_MAX),uint64_t(SIZE_MAX/sizeof(float)));
    if (dim > maximum/max_rows || uint64_t(shape.heads) > maximum/max_rows/dim) return false;
    const size_t rows=std::min(max_rows,size_t(shape.page_tokens));
    ggml_kv_stream_layout layout;
    if (ggml_kv_stream_layout_make(shape,rows,layout).status != ggml_kv_stream_status::success) return false;
    size_t encoded=0;
    for (bool value : {false,true}) {
        ggml_tensor tensor{};
        tensor.type=ggml_type(value ? shape.type_v : shape.type_k);
        tensor.ne[0]=(value ? shape.head_dim_v : shape.head_dim_k)*shape.heads*int64_t(rows);
        tensor.ne[1]=tensor.ne[2]=tensor.ne[3]=1;
        tensor.nb[0]=ggml_type_size(tensor.type);
        tensor.nb[1]=tensor.nb[2]=tensor.nb[3]=value ? layout.v_bytes : layout.k_bytes;
        const size_t allocated=ggml_backend_buft_get_alloc_size(type,&tensor);
        if (allocated < tensor.nb[1]) return false;
        encoded=std::max(encoded,allocated);
    }
    if (encoded > SIZE_MAX-127) return false;
    const size_t index_offset=(encoded+127)/128*128;
    ggml_tensor index{};
    index.type=GGML_TYPE_I64; index.ne[0]=int64_t(rows); index.ne[1]=index.ne[2]=index.ne[3]=1;
    index.nb[0]=sizeof(int64_t); index.nb[1]=index.nb[2]=index.nb[3]=rows*sizeof(int64_t);
    const size_t indices=ggml_backend_buft_get_alloc_size(type,&index);
    if (indices < index.nb[1] || indices > SIZE_MAX-index_offset) return false;
    bytes=index_offset+indices;
    return true;
}

// Use actual backend padding costs, then reserve aligned indices after both possible encoded layouts.
std::unique_ptr<llama_kv_stream_writer> llama_kv_stream_writer::create(ggml_backend_t backend, ggml_backend_buffer_t buffer,
        size_t scratch_bytes, const ggml_kv_stream_shape & shape, size_t max_rows) {
    if (!backend || !buffer || !max_rows || max_rows > INT32_MAX || scratch_bytes > ggml_backend_buffer_get_size(buffer)) return {};
    ggml_kv_stream_layout row;
    if (ggml_kv_stream_layout_make(shape, 1, row).status != ggml_kv_stream_status::success) return {};
    const uint64_t max_elements = std::min(uint64_t(INT32_MAX), uint64_t(SIZE_MAX/sizeof(float)));
    const uint64_t dim = uint64_t(std::max(shape.head_dim_k, shape.head_dim_v));
    if (dim > max_elements/max_rows || uint64_t(shape.heads) > max_elements/max_rows/dim) return {};
    try {
        std::unique_ptr<llama_kv_stream_writer> writer(new llama_kv_stream_writer);
        writer->impl = std::make_unique<implementation>();
        auto & s = *writer->impl;
        s.backend = backend; s.buffer.reset(ggml_backend_buffer_retain(buffer)); s.shape = shape; s.maximum = max_rows;
        auto * device = ggml_backend_get_device(backend);
        s.cpu = device && ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU;
        if (!s.cpu) { llama_memory_cuda_executor probe(backend); if (!probe.supported()) return {}; }
        const size_t max_row = std::max(row.k_token_bytes, row.v_token_bytes);
        if (max_row > SIZE_MAX - sizeof(int64_t)) return {};
        s.capacity = std::min(max_rows, scratch_bytes/(max_row + sizeof(int64_t)));
        size_t index_offset = 0;
        while (s.capacity) {
            size_t encoded = 0;
            for (bool value : {false, true}) {
                ggml_tensor tensor = {};
                tensor.type = s.type(value);
                tensor.ne[0] = int64_t(s.width(value)*s.capacity);
                tensor.ne[1] = tensor.ne[2] = tensor.ne[3] = 1;
                tensor.nb[0] = ggml_type_size(tensor.type);
                tensor.nb[1] = tensor.nb[2] = tensor.nb[3] = s.row_bytes(value)*s.capacity;
                const size_t bytes = ggml_backend_buffer_get_alloc_size(buffer, &tensor);
                if (bytes < tensor.nb[1]) return {};
                encoded = std::max(encoded, bytes);
            }
            if (encoded <= SIZE_MAX - 127) {
                index_offset = (encoded + 127)/128*128;
                if (index_offset <= scratch_bytes && s.capacity <= (scratch_bytes - index_offset)/sizeof(int64_t)) break;
            }
            --s.capacity;
        }
        if (!s.capacity) return {};
        s.scratch_bytes = index_offset + s.capacity*sizeof(int64_t);
        s.context.reset(ggml_init({16384, nullptr, true}));
        if (!s.context) return {};
        auto * base = static_cast<char *>(ggml_backend_buffer_get_base(buffer));
        for (bool value : {false, true}) {
            s.roots[value] = ggml_new_tensor_1d(s.context.get(), s.type(value), int64_t(s.width(value)*s.capacity));
            if (ggml_backend_tensor_alloc(buffer, s.roots[value], base) != GGML_STATUS_SUCCESS) return {};
        }
        if (!s.cpu && s.capacity>=2) {
            const size_t count=s.capacity/2;
            auto * key=ggml_new_tensor_1d(s.context.get(),s.type(false),int64_t(s.width(false)*count));
            auto * value=ggml_new_tensor_1d(s.context.get(),s.type(true),int64_t(s.width(true)*count));
            const size_t key_bytes=ggml_backend_buffer_get_alloc_size(buffer,key),value_bytes=ggml_backend_buffer_get_alloc_size(buffer,value);
            if (key_bytes<=SIZE_MAX-127) {
                const size_t offset=(key_bytes+127)/128*128;
                if (offset<=index_offset && value_bytes<=index_offset-offset &&
                        ggml_backend_tensor_alloc(buffer,key,base)==GGML_STATUS_SUCCESS &&
                        ggml_backend_tensor_alloc(buffer,value,base+offset)==GGML_STATUS_SUCCESS) {
                    s.routed_roots[0]=key;s.routed_roots[1]=value;s.routed_capacity=count;
                }
            }
        }
        s.indices = ggml_new_tensor_1d(s.context.get(), GGML_TYPE_I64, int64_t(s.capacity));
        if (ggml_backend_tensor_alloc(buffer, s.indices, base + index_offset) != GGML_STATUS_SUCCESS) return {};
        s.index_data.resize(s.capacity);
        for (size_t i = 0; i < s.capacity; ++i) s.index_data[i] = int64_t(i);
        return writer;
    } catch (const std::bad_alloc &) { return {}; }
}

// Admit dense F32 source rows; ordered backend work may still be pending when submission begins.
bool llama_kv_stream_writer::matches_shape(const ggml_kv_stream_shape & shape) const noexcept {
    if (!impl) return false;
    const auto & actual = impl->shape;
    return actual.type_k == shape.type_k && actual.type_v == shape.type_v &&
        actual.head_dim_k == shape.head_dim_k && actual.head_dim_v == shape.head_dim_v &&
        actual.heads == shape.heads && actual.page_tokens == shape.page_tokens &&
        actual.alignment == shape.alignment;
}

bool llama_kv_stream_writer::matches_backend(ggml_backend_t backend) const noexcept {
    return impl && backend && impl->backend==backend;
}


size_t llama_kv_stream_writer::maximum_rows() const noexcept {return impl ? impl->maximum : 0;}

bool llama_kv_stream_writer::accepts(const ggml_tensor * source, bool value) const {
    const auto & s = *impl;
    if (!source || source->type != GGML_TYPE_F32 || !source->data || source->ne[0] != int64_t(s.width(value)) ||
            source->ne[1] <= 0 || uint64_t(source->ne[1]) > s.maximum || source->ne[2] != 1 || source->ne[3] != 1 ||
            source->nb[0] != sizeof(float) || source->nb[1] != s.width(value)*sizeof(float)) return false;
    auto * buffer = source->view_src ? source->view_src->buffer : source->buffer;
    if (!buffer || !ggml_backend_supports_buft(s.backend, ggml_backend_buffer_get_type(buffer))) return false;
    const auto base = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(buffer));
    const auto data = reinterpret_cast<uintptr_t>(source->data);
    const size_t size = ggml_backend_buffer_get_size(buffer);
    const size_t bytes = size_t(source->ne[1])*source->nb[1];
    const auto scratch = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(s.buffer.get()));
    const bool overlaps = data >= scratch ? data-scratch < s.scratch_bytes : bytes > scratch-data;
    return !overlaps && data >= base && data - base <= size && bytes <= size - size_t(data - base);
}

bool llama_kv_stream_writer::generate_async(const ggml_tensor * source, bool value, void * host,
        const std::function<bool(const ggml_tensor *, size_t, size_t)> & publish,
        llama_kv_stream_writer_completion & completion) {
    return generate_async_impl(source,value,host,nullptr,publish,completion);
}

bool llama_kv_stream_writer::generate_async_registered(const ggml_tensor * source, bool value,
        const std::vector<llama_kv_stream_host_range> & destinations,
        const std::function<bool(const ggml_tensor *, size_t, size_t)> & publish,
        llama_kv_stream_writer_completion & completion) {
    return generate_async_impl(source,value,nullptr,&destinations,publish,completion);
}

bool llama_kv_stream_writer::generate_async_routed(const ggml_tensor * source,bool value,
        const std::vector<llama_kv_stream_host_range> & destinations,
        const std::function<bool(const ggml_tensor *,size_t,size_t)> & publish,
        llama_kv_stream_writer_completion & completion) {
    return generate_async_impl(source,value,nullptr,&destinations,publish,completion,true);
}

bool llama_kv_stream_writer::generate_async_impl(const ggml_tensor * source, bool value, void * host,
        const std::vector<llama_kv_stream_host_range> * destinations,
        const std::function<bool(const ggml_tensor *, size_t, size_t)> & publish,
        llama_kv_stream_writer_completion & completion,bool routed) {
    auto & s=*impl;
    s.last={}; s.last.device_scratch_bytes=s.scratch_bytes; s.last.tile_rows=s.capacity;
    if ((!host && !destinations) || !accepts(source,value) || completion.device || completion.host) return false;
    const size_t rows=size_t(source->ne[1]),stride=s.row_bytes(value);
    s.last.host_payload_bytes=rows*stride;
    const auto fail=[&] {
        ggml_backend_synchronize(s.backend);
        if (s.host_backend) ggml_backend_synchronize(s.host_backend.get());
        release_completed();
        return false;
    };
    try {
        if (destinations) {
            if (destinations->empty() || destinations->size() > rows*stride) return false;
            auto * required_type = s.cpu ? nullptr : llama_kv_stream_host_buffer_type(ggml_backend_get_device(s.backend));
            if (!s.cpu && !required_type) return false;
            size_t total = 0;
            uintptr_t previous = 0;
            std::vector<std::pair<uintptr_t,uintptr_t>> intervals;
            if (routed) intervals.reserve(destinations->size());
            for (const auto & range : *destinations) {
                if (!range.buffer || !range.bytes || range.bytes > rows*stride-total ||
                        !ggml_backend_buffer_is_host(range.buffer.get()) ||
                        (!s.cpu && ggml_backend_buffer_get_type(range.buffer.get()) != required_type)) return false;
                const size_t capacity = ggml_backend_buffer_get_size(range.buffer.get());
                const auto address = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(range.buffer.get()));
                if (!address || range.offset > capacity || range.bytes > capacity-range.offset ||
                        range.offset > UINTPTR_MAX-address || range.bytes > UINTPTR_MAX-address-range.offset ||
                        (!routed && total && address+range.offset != previous)) return false;
                previous = address+range.offset+range.bytes;
                if (routed) {
                    const auto begin=address+range.offset,end=previous;
                    const auto source_base=reinterpret_cast<uintptr_t>(source->data);
                    const auto scratch_base=reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(s.buffer.get()));
                    const auto source_bytes=ggml_nbytes(source);
                    if (source_bytes>UINTPTR_MAX-source_base || s.scratch_bytes>UINTPTR_MAX-scratch_base ||
                            (begin<source_base+source_bytes && source_base<end) ||
                            (begin<scratch_base+s.scratch_bytes && scratch_base<end)) return false;
                    intervals.emplace_back(begin,end);
                }
                total += range.bytes;
            }
            if (routed) {
                std::sort(intervals.begin(),intervals.end());
                for (size_t i=1;i<intervals.size();++i) if (intervals[i].first<intervals[i-1].second) return false;
            }
            if (total != rows*stride || destinations->size() > s.destinations.max_size()-s.destinations.size()) return false;
            s.destinations.reserve(s.destinations.size()+destinations->size());
            for (const auto & range : *destinations) {
                const auto found = std::find_if(s.destinations.begin(),s.destinations.end(),
                    [&](const ggml_backend_buffer_ptr & owner) { return owner.get() == range.buffer.get(); });
                if (found == s.destinations.end()) s.destinations.emplace_back(ggml_backend_buffer_retain(range.buffer.get()));
            }
        }
        const bool asynchronous=routed && publish && !s.cpu && rows<=s.routed_capacity && s.prepare_host_backend();
        if (s.host_backend) for (size_t side=0;side<2;++side)
            if (s.host_busy[side] && (!asynchronous || side==size_t(value))) ggml_backend_event_wait(s.backend,s.host_consumed[side]);
        // Another execution phase may have overwritten the shared scratch.
        ggml_backend_tensor_set_async(s.backend,s.indices,s.index_data.data(),0,std::min(rows,s.capacity)*sizeof(int64_t));
        size_t tile=0;
        size_t destination=0, destination_offset=0;
        for (size_t first=0; first < rows; first += std::min(s.capacity,rows-first)) {
            const size_t count=std::min(s.capacity,rows-first);
            auto * plan=s.prepare(source,value,tile++,first,count,asynchronous,routed);
            if (!plan) return fail();
            ++s.last.graph_submissions;
            if (s.cpu) {
                if (ggml_backend_graph_compute(s.backend,plan->graph) != GGML_STATUS_SUCCESS) return fail();
            } else if (plan->execution->compute_async({},1) != GGML_STATUS_SUCCESS) return fail();
            auto * download=s.backend;
            if (asynchronous) {
                if (!publish(s.routed_roots[value],first,count)) return fail();
                s.last.d2d_bytes+=count*stride;++s.last.d2d_calls;
                ggml_backend_event_record(s.host_ready[value],s.backend);
                ggml_backend_event_wait(s.host_backend.get(),s.host_ready[value]);download=s.host_backend.get();
            }
            if (destinations) {
                size_t copied=0;
                while (copied < count*stride) {
                    const auto & range=(*destinations)[destination];
                    const size_t bytes=std::min(range.bytes-destination_offset,count*stride-copied);
                    auto * target=static_cast<char *>(ggml_backend_buffer_get_base(range.buffer.get()))+range.offset+destination_offset;
                    ggml_backend_tensor_get_async(download,plan->output,target,copied,bytes);
                    s.last.d2h_bytes += bytes; ++s.last.d2h_calls;
                    copied += bytes; destination_offset += bytes;
                    if (destination_offset == range.bytes) { ++destination; destination_offset=0; }
                }
            } else {
                ggml_backend_tensor_get_async(s.backend,plan->output,static_cast<char *>(host)+first*stride,0,count*stride);
                s.last.d2h_bytes += count*stride; ++s.last.d2h_calls;
            }
            const bool published=asynchronous || !publish || publish(s.roots[value],first,count);
            if (!published) return fail();
            if (publish && !asynchronous) { s.last.d2d_bytes += count*stride; ++s.last.d2d_calls; }
        }
        if (asynchronous) {
            ggml_backend_event_record(s.host_consumed[value],s.host_backend.get());s.host_busy[value]=true;
        }
        llama_kv_stream_writer_completion next;
        if (s.cpu) {
            next.device=llama_memory_completion::completed(s.backend);
            next.host=llama_memory_completion::completed(s.backend);
        } else {
            next.device=llama_memory_completion::create(s.backend);
            next.host=asynchronous ? llama_memory_completion::create_retained(s.host_backend.get(),s.host_backend) : llama_memory_completion::create(s.backend);
            if (!next.device || !next.host || !next.device->record() || !next.host->record()) return fail();
        }
        if (!next.device || !next.host) return fail();
        completion=std::move(next);
        return true;
    } catch (...) {
        ggml_backend_synchronize(s.backend);
        if (s.host_backend) ggml_backend_synchronize(s.host_backend.get());
        release_completed();
        throw;
    }
}

bool llama_kv_stream_writer::generate(const ggml_tensor * source, bool value, void * host,
        const std::function<bool(const ggml_tensor *, size_t, size_t)> & publish) {
    llama_kv_stream_writer_completion completion;
    if (!generate_async(source,value,host,publish,completion)) return false;
    const bool device=completion.device->synchronize();
    const bool host_ready=completion.host->synchronize();
    release_completed();
    return device && host_ready;
}

void llama_kv_stream_writer::release_completed() noexcept {
    if (!impl) return;
    const auto release=[](auto & caches) {
        for (auto & cache : caches) for (auto & plan : cache)
            if (plan && plan->execution) plan->execution->release_completed();
    };
    if (!impl->cpu) {release(impl->plans);release(impl->cached);}
    impl->destinations.clear();
    impl->host_busy[0]=impl->host_busy[1]=false;
}

llama_kv_stream_write_stats llama_kv_stream_writer::stats() const { return impl->last; }
