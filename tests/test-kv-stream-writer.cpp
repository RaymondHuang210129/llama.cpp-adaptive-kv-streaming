#include "../src/llama-kv-stream-resident.h"
#include "../src/llama-kv-stream-writer.h"
#include "../ggml/src/ggml-kv-stream-device.h"
#include "../src/llama-memory-executor-cuda.h"
#include "ggml-alloc.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "testing.h"
#include "../ggml/src/ggml-backend-impl.h"
#include <stdexcept>

#include <chrono>
#include <array>
#include <cmath>
#include <cstring>

using arena_ptr = std::unique_ptr<ggml_backend_memory_arena, decltype(&ggml_backend_memory_arena_free)>;
using lease_ptr = std::unique_ptr<ggml_backend_memory_lease, decltype(&ggml_backend_memory_lease_free)>;

struct writer_fixture {
    ggml_backend_t backend;
    bool cuda;
    std::shared_ptr<llama_kv_stream_host> host;
    std::shared_ptr<llama_kv_stream_content> content;
    llama_kv_stream_policy_config policy;
    arena_ptr arena{nullptr, ggml_backend_memory_arena_free};
    lease_ptr lease{nullptr, ggml_backend_memory_lease_free};
    std::unique_ptr<llama_kv_stream_binding> binding;
    llama_memory_execution pin;
    llama_kv_stream_resident * resident = nullptr;

    // Fixed Qwen-like geometry and budget, independent of the writer implementation being compared.
    writer_fixture(ggml_backend_t backend, bool cuda, ggml_type k = GGML_TYPE_Q8_0, ggml_type v = GGML_TYPE_Q4_0, bool tiny = false, uint32_t width = 256) : backend(backend), cuda(cuda) {
        llama_kv_stream_host_config c{91, {k, v, width, width, 4, 256, 128},
            {{k, true, true, true, true}, {v, true, true, true, true}, true, true}, 1024, 1};
        auto * device = ggml_backend_get_device(backend);
        if (cuda) {
            auto * reg = ggml_backend_dev_backend_reg(device);
            auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_kv_stream_partial_ops"));
            if (get && get() && get()->version >= 3) c.capabilities = get()->capabilities(backend,k,v);
        }
        auto * ht = cuda ? llama_kv_stream_host_buffer_type(device) : ggml_backend_cpu_buffer_type();
        auto * dt = cuda ? llama_kv_stream_device_buffer_type(device) : ggml_backend_cpu_buffer_type();
        GGML_ASSERT(ht && dt);
        host = llama_kv_stream_host::create(c, ht); GGML_ASSERT(host);
        content = std::make_shared<llama_kv_stream_content>(host);
        ggml_kv_stream_layout page; GGML_ASSERT(ggml_kv_stream_layout_make(c.shape, 256, page).status == ggml_kv_stream_status::success);
        policy.shape = c.shape; policy.capabilities = c.capabilities; policy.layers = 1;
        ggml_kv_stream_execution execution;
        GGML_ASSERT(ggml_kv_stream_resolve(c.shape,c.capabilities,256,execution).status == ggml_kv_stream_status::success);
        policy.pool_bytes = page.bytes*(tiny ? 8 : 16)+execution.conversion.bytes; policy.initial_ring_slots = tiny ? 1 : 0;
        arena.reset(ggml_backend_memory_arena_new(dt, policy.pool_bytes + 256)); GGML_ASSERT(arena);
        const auto base = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(ggml_backend_memory_arena_parent(arena.get())));
        GGML_ASSERT(ggml_backend_memory_arena_begin(arena.get(), 0));
        GGML_ASSERT(ggml_backend_memory_arena_reserve_at(arena.get(), 3, 128 + (128 - base%128)%128, policy.pool_bytes, 64, 0, nullptr));
        GGML_ASSERT(ggml_backend_memory_arena_commit(arena.get()));
        lease.reset(ggml_backend_memory_arena_acquire(arena.get(), 3));
        binding = std::make_unique<llama_kv_stream_binding>(91, dt);
        GGML_ASSERT(binding->bind(lease.get(), policy, [&](const auto & view) {
            auto r = llama_kv_stream_resident::create(view, content, backend); resident = r.get(); return r;
        }));
        pin = binding->acquire(); GGML_ASSERT(pin);
        GGML_ASSERT(resident->synchronize(1024));
    }
};

struct encoder {
    ggml_backend_t backend;
    bool cuda;
    size_t rows, row_bytes;
    ggml_context_ptr context;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * source = nullptr, * output = nullptr;
    ggml_cgraph * graph = nullptr;
    std::unique_ptr<llama_memory_cuda_executor> execution;

    // Ordinary SET_ROWS is the independent GPU quantization control; its source buffer is already on the device.
    encoder(writer_fixture & f, size_t rows, ggml_type type) : backend(f.backend), cuda(f.cuda), rows(rows), row_bytes(ggml_row_size(type, int64_t(f.host->config().shape.head_dim_k)*f.host->config().shape.heads)) {
        const size_t width=size_t(f.host->config().shape.head_dim_k)*f.host->config().shape.heads;
        context.reset(ggml_init({65536, nullptr, true}));
        source = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, int64_t(width), int64_t(rows));
        auto * indices = ggml_new_tensor_1d(context.get(), GGML_TYPE_I64, int64_t(rows));
        auto * dst = ggml_new_tensor_2d(context.get(), type, int64_t(width), int64_t(rows));
        output = ggml_set_rows(context.get(), dst, source, indices);
        GGML_ASSERT(ggml_backend_supports_op(backend, output));
        graph = ggml_new_graph_custom(context.get(), 32, false);
        ggml_build_forward_expand(graph, output);
        buffer.reset(ggml_backend_alloc_ctx_tensors(context.get(), backend)); GGML_ASSERT(buffer);
        std::vector<float> input(rows*width);
        for (size_t i = 0; i < input.size(); ++i) input[i] = std::sin(float(i%1021)*.07f);
        std::vector<int64_t> idx(rows); for (size_t i = 0; i < rows; ++i) idx[i] = int64_t(i);
        ggml_backend_tensor_set(source, input.data(), 0, input.size()*sizeof(float));
        ggml_backend_tensor_set(indices, idx.data(), 0, idx.size()*sizeof(int64_t));
        if (cuda) { execution = std::make_unique<llama_memory_cuda_executor>(backend); GGML_ASSERT(execution->bind(graph, {}, 1)); }
    }
    void run() {
        if (cuda) { GGML_ASSERT(execution->compute_async({}, 1) == GGML_STATUS_SUCCESS); GGML_ASSERT(execution->drain()); }
        else GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    }
};

// Existing coalesced/row-wise callers of the committed content and resident APIs, with identical GPU quantization.
static void baseline(writer_fixture & f, encoder & e, bool rowwise, std::vector<uint8_t> & bytes) {
    e.run();
    if (rowwise) for (size_t r = 0; r < e.rows; ++r) ggml_backend_tensor_get(e.output, bytes.data() + r*e.row_bytes, r*e.row_bytes, e.row_bytes);
    else ggml_backend_tensor_get(e.output, bytes.data(), 0, bytes.size());
    llama_kv_stream_write write;
    GGML_ASSERT(f.content->prepare({{0, ggml_kv_stream_operand::k, 0, bytes.data(), bytes.size()}}, write));
    GGML_ASSERT(f.content->commit(write));
    GGML_ASSERT(f.resident->synchronize(1024));
}

// Compare canonical bytes and the independently addressed resident plane.
static void check_bytes(testing & t, writer_fixture & f, bool value, size_t first, const std::vector<uint8_t> & expected) {
    llama_kv_stream_host_layer host; f.host->layer(0, host);
    const size_t stride = value ? f.host->layout().v_token_bytes : f.host->layout().k_token_bytes;
    t.assert_true(std::memcmp(static_cast<char *>(value ? host.v : host.k) + first*stride, expected.data(), expected.size()) == 0);
    llama_kv_stream_policy_layout layout;
    GGML_ASSERT(llama_kv_stream_policy_layout_make(f.policy, f.binding->view()->initial_policy, 0, layout).status == llama_kv_stream_policy_status::success);
    auto * buffer = ggml_backend_memory_lease_buffer(f.lease.get());
    ggml_context_ptr context(ggml_init({4096, nullptr, true}));
    auto * tensor = ggml_new_tensor_1d(context.get(), GGML_TYPE_I8, int64_t(expected.size()));
    const size_t offset = layout.layers[0].offset + (value ? layout.layers[0].planes.v_offset : 0) + first*stride;
    GGML_ASSERT(ggml_backend_tensor_alloc(buffer, tensor, static_cast<char *>(f.binding->view()->base) + offset) == GGML_STATUS_SUCCESS);
    std::vector<uint8_t> actual(expected.size());
    ggml_backend_tensor_get(tensor, actual.data(), 0, actual.size());
    t.assert_true(actual == expected);
}

// Fail after a real D2D copy, so rollback must repair a partially changed mirror.
struct copy_fault {
    ggml_backend_buffer_t buffer;
    ggml_backend_t backend;
    decltype(ggml_backend_buffer_i::set_tensor) original_set;
    decltype(ggml_backend_i::cpy_tensor_async) original_async;
    int calls = 0;
    inline static copy_fault * active = nullptr;
    copy_fault(ggml_backend_t backend, ggml_backend_buffer_t buffer) : buffer(buffer), backend(backend),
        original_set(buffer->iface.set_tensor), original_async(backend->iface.cpy_tensor_async) {
        GGML_ASSERT(!active); active = this;
        if (original_async) backend->iface.cpy_tensor_async = [](ggml_backend_t a, ggml_backend_t b, const ggml_tensor * src, ggml_tensor * dst) {
            const bool result = active->original_async(a, b, src, dst);
            if (++active->calls == 2) throw std::runtime_error("injected queued second tile failure");
            return result;
        };
        else buffer->iface.set_tensor = [](ggml_backend_buffer_t b, ggml_tensor * dst, const void * data, size_t offset, size_t size) {
            active->original_set(b, dst, data, offset, size);
            if (++active->calls == 2) throw std::runtime_error("injected second tile failure");
        };
    }
    ~copy_fault() { buffer->iface.set_tensor = original_set; backend->iface.cpy_tensor_async = original_async; active = nullptr; }
};

// Observe final input-buffer release without changing its native free behavior.
struct release_probe {
    ggml_backend_buffer_t buffer;
    decltype(ggml_backend_buffer_i::free_buffer) original;
    int frees = 0;
    inline static release_probe * active = nullptr;
    explicit release_probe(ggml_backend_buffer_t buffer) : buffer(buffer), original(buffer->iface.free_buffer) {
        GGML_ASSERT(!active); active = this;
        buffer->iface.free_buffer = [](ggml_backend_buffer_t b) { ++active->frees; if (active->original) active->original(b); };
    }
    ~release_probe() { if (!frees) buffer->iface.free_buffer = original; active = nullptr; }
};

#ifdef KV_STREAM_WRITER_EVENT_TRACE
void kv_stream_writer_capture_events(bool enabled);
bool kv_stream_writer_observed_host_backend(ggml_backend_t producer);
bool kv_stream_writer_observed_reuse_wait(ggml_backend_t producer);
#endif

#ifdef KV_STREAM_WRITER_CACHE_TRACE
void kv_stream_writer_capture_cache(bool enabled);
size_t kv_stream_writer_cache_syncs();
#endif

static void check_routed_publication(testing & t,ggml_backend_t backend,bool cache_sources=false,size_t patterns=3) {
    auto * device=ggml_backend_get_device(backend);
    auto * reg=ggml_backend_dev_backend_reg(device);
    const auto get=reinterpret_cast<ggml_kv_stream_copy_ops_get>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_kv_stream_copy_ops"));
    const auto * copies=get ? get() : nullptr;
    if (!t.assert_true(copies && copies->gather)) return;
    for (uint32_t dimension : {256u,512u}) for (auto formats : {
            std::pair{GGML_TYPE_Q8_0,GGML_TYPE_Q4_0},std::pair{GGML_TYPE_Q8_0,GGML_TYPE_Q8_0},
            std::pair{GGML_TYPE_Q5_1,GGML_TYPE_Q5_1}}) for (size_t rows : {1u,8u,16u}) {
        if (cache_sources && (formats.first!=GGML_TYPE_Q8_0 || formats.second!=GGML_TYPE_Q4_0 ||
                rows!=(patterns>3 ? 1u : 8u) || (patterns>3 && dimension!=256))) continue;
        const size_t repeats=cache_sources ? (patterns>3 ? patterns+2 : 9) : 3;
        writer_fixture f(backend,true,formats.first,formats.second,false,dimension);
        encoder key(f,rows,formats.first),value(f,rows,formats.second);
        encoder * controls[2]={&key,&value};
        const size_t strides[2]={key.row_bytes,value.row_bytes};
        size_t scratch_bytes=0;
        auto * type=llama_kv_stream_device_buffer_type(device);
        if (!t.assert_true(llama_kv_stream_writer::requirements(type,f.host->config().shape,256,scratch_bytes))) return;
        const size_t alignment=ggml_backend_buft_get_alignment(type);
        const size_t first=(scratch_bytes+alignment-1)/alignment*alignment;
        const size_t second=(first+repeats*rows*strides[0]+alignment-1)/alignment*alignment;
        const size_t source_offset=(second+repeats*rows*strides[1]+alignment-1)/alignment*alignment;
        const size_t source_bytes=2*patterns*rows*dimension*4*sizeof(float);
        arena_ptr arena(ggml_backend_memory_arena_new(type,cache_sources ? source_offset+source_bytes : second+repeats*rows*strides[1]),ggml_backend_memory_arena_free);
        if (!t.assert_true(arena && ggml_backend_memory_arena_begin(arena.get(),0) &&
                ggml_backend_memory_arena_reserve_at(arena.get(),1,0,scratch_bytes,alignment,0,nullptr) &&
                ggml_backend_memory_arena_reserve_at(arena.get(),2,first,repeats*rows*strides[0],alignment,0,nullptr) &&
                ggml_backend_memory_arena_reserve_at(arena.get(),3,second,repeats*rows*strides[1],alignment,0,nullptr) &&
                (!cache_sources || ggml_backend_memory_arena_reserve_at(arena.get(),4,source_offset,source_bytes,alignment,0,nullptr)) &&
                ggml_backend_memory_arena_commit(arena.get()))) return;
        auto * scratch=ggml_backend_memory_arena_get_buffer(arena.get(),1);
        auto writer=llama_kv_stream_writer::create(backend,scratch,scratch_bytes,f.host->config().shape,256);
        if (!t.assert_true(writer!=nullptr)) return;
        ggml_context_ptr source_metadata;
        std::vector<ggml_tensor *> inputs(2*patterns);
        if (cache_sources) {
            source_metadata.reset(ggml_init({4096+(2*patterns+8)*ggml_tensor_overhead(),nullptr,true}));
            if (!t.assert_true(bool(source_metadata))) return;
            auto * storage=ggml_backend_memory_arena_get_buffer(arena.get(),4);
            for (size_t i=0;i<2*patterns;++i) {
                inputs[i]=ggml_new_tensor_2d(source_metadata.get(),GGML_TYPE_F32,dimension*4,int64_t(rows));
                if (!t.assert_true(ggml_backend_tensor_alloc(storage,inputs[i],static_cast<char *>(ggml_backend_buffer_get_base(storage))+
                        i*rows*dimension*4*sizeof(float))==GGML_STATUS_SUCCESS)) return;
            }
        }
        std::unique_ptr<llama_kv_stream_host_reservation> hosts[2];
        for (uint32_t slot=0;slot<2;++slot) {
            auto config=f.host->config();config.cache_id=810+slot;
            hosts[slot]=llama_kv_stream_host_reservation::create(config,device,65536);
            if (!t.assert_true(hosts[slot] && hosts[slot]->prepare(cache_sources ? 600 : 300))) return;
        }
        std::vector<std::array<std::vector<uint8_t>,2>> expected(repeats);
        std::vector<std::array<std::vector<std::pair<const uint8_t *,size_t>>,2>> addresses(repeats);
        std::vector<std::array<llama_kv_stream_writer_completion,2>> completions(repeats);
        std::vector<std::array<uintptr_t,2>> encoded(repeats);
#ifdef KV_STREAM_WRITER_EVENT_TRACE
        kv_stream_writer_capture_events(true);
#endif
        for (size_t repeat=0;repeat<repeats;++repeat) for (size_t side=0;side<2;++side) {
            auto & control=*controls[side];
            std::vector<float> input(size_t(ggml_nelements(control.source)));
            for (size_t i=0;i<input.size();++i) input[i]=std::sin(float((i+repeat*17)%1021)*.07f);
            auto * active_source=cache_sources ? inputs[2*(repeat%patterns)+side] : control.source;
            ggml_backend_buffer_ptr transient_view;
            ggml_tensor transient_source{};
            if (cache_sources) {
                auto * storage=ggml_backend_memory_arena_get_buffer(arena.get(),4);
                transient_view.reset(ggml_backend_buffer_view(storage,0,source_bytes));
                if (!t.assert_true(bool(transient_view))) return;
                transient_source=*active_source;transient_source.buffer=nullptr;transient_source.data=nullptr;
                if (!t.assert_true(ggml_backend_tensor_alloc(transient_view.get(),&transient_source,active_source->data)==GGML_STATUS_SUCCESS)) return;
                active_source=&transient_source;
            }
            if (cache_sources) ggml_backend_tensor_set_async(backend,active_source,input.data(),0,input.size()*sizeof(float));
            ggml_backend_tensor_set(control.source,input.data(),0,input.size()*sizeof(float));
            control.run();expected[repeat][side].resize(rows*strides[side]);
            ggml_backend_tensor_get(control.output,expected[repeat][side].data(),0,expected[repeat][side].size());
            std::vector<llama_kv_stream_host_range> routes;
            size_t cursors[2]={7+repeat*(patterns>3 ? 4 : 64),13+repeat*(patterns>3 ? 4 : 64)};
            for (size_t row=0;row<rows;++row) {
                const size_t slot=row%2;
                std::vector<llama_kv_stream_host_range> ranges;
                if (!t.assert_true(hosts[slot]->ranges(0,side ? ggml_kv_stream_operand::v : ggml_kv_stream_operand::k,
                        cursors[slot]++*strides[side],strides[side],ranges))) return;
                for (auto & range : ranges) {
                    addresses[repeat][side].push_back({static_cast<const uint8_t *>(ggml_backend_buffer_get_base(range.buffer.get()))+range.offset,range.bytes});
                    routes.push_back(std::move(range));
                }
            }
            auto * mirror=ggml_backend_memory_arena_get_buffer(arena.get(),side+2);
            const auto publish=[&](const ggml_tensor * source,size_t first_row,size_t count) {
                encoded[repeat][side]=uintptr_t(source->data);
                const ggml_kv_stream_copy_rectangle rectangle{source->buffer,
                    size_t(static_cast<const char *>(source->data)-static_cast<const char *>(ggml_backend_buffer_get_base(source->buffer))),
                    strides[side],(repeat*rows+first_row)*strides[side],strides[side],strides[side],count};
                return copies->gather(backend,mirror,&rectangle,1,false);
            };
#ifdef KV_STREAM_WRITER_CACHE_TRACE
            if (cache_sources && repeat>=patterns) kv_stream_writer_capture_cache(true);
#endif
            const bool generated=writer->generate_async_routed(active_source,side,routes,publish,completions[repeat][side]);
#ifdef KV_STREAM_WRITER_CACHE_TRACE
            if (cache_sources && repeat>=patterns) {
                kv_stream_writer_capture_cache(false);
                if (patterns>3) t.assert_true("an evicted source is safely rebound after the bounded cache fills",kv_stream_writer_cache_syncs()>0);
                else t.assert_true("switching among warmed same-parent sources does not rebuild or drain encoding graphs",kv_stream_writer_cache_syncs()==0);
            }
#else
            if (cache_sources) {t.assert_true("native graph-cache tracing is required",false);return;}
#endif
            if (!t.assert_true(generated)) return;
        }
#ifdef KV_STREAM_WRITER_EVENT_TRACE
        const bool reuse_wait=kv_stream_writer_observed_reuse_wait(backend);
        bool host_backend=true;
        for (auto & pair : completions) for (auto & completion : pair) {
            if (!t.assert_true(completion.host && completion.host->wait(backend))) return;
            host_backend &= kv_stream_writer_observed_host_backend(backend);
        }
        kv_stream_writer_capture_events(false);
        ggml_backend_synchronize(backend);
#endif
        hosts[0].reset();hosts[1].reset();
        for (size_t repeat=0;repeat<repeats;++repeat) for (size_t side=0;side<2;++side) {
            auto & completion=completions[repeat][side];
            if (!t.assert_true(completion.device && completion.host && completion.device->synchronize() && completion.host->synchronize())) return;
#ifdef KV_STREAM_WRITER_EVENT_TRACE
            if (!t.assert_true(completion.host->release_waiter(backend))) return;
#endif
            size_t offset=0;
            for (const auto & range : addresses[repeat][side]) {
                if (!t.assert_true("deferred canonical writes retain every native byte and registration owner",
                        !std::memcmp(expected[repeat][side].data()+offset,range.first,range.second))) return;
                offset+=range.second;
            }
            ggml_context_ptr metadata(ggml_init({4096,nullptr,true}));
            auto * output=ggml_new_tensor_1d(metadata.get(),GGML_TYPE_I8,int64_t(rows*strides[side]));
            auto * mirror=ggml_backend_memory_arena_get_buffer(arena.get(),side+2);
            if (!t.assert_true(ggml_backend_tensor_alloc(mirror,output,static_cast<char *>(ggml_backend_buffer_get_base(mirror))+repeat*rows*strides[side])==GGML_STATUS_SUCCESS)) return;
            std::vector<uint8_t> actual(rows*strides[side]);ggml_backend_tensor_get(output,actual.data(),0,actual.size());
            if (!t.assert_true("device publication preserves native encoding before scratch reuse",actual==expected[repeat][side])) return;
        }
        bool disjoint=true;
        for (const auto & roots : encoded) disjoint &= roots[0]+rows*strides[0]<=roots[1] || roots[1]+rows*strides[1]<=roots[0];
        const bool separate=t.assert_true("routed K and V encodings occupy disjoint existing scratch",disjoint);
#ifdef KV_STREAM_WRITER_EVENT_TRACE
        const bool host=t.assert_true("canonical download completion belongs to a separate same-device backend",host_backend);
        const bool reuse=t.assert_true("scratch reuse waits on the previous host download",reuse_wait);
        if (!separate || !host || !reuse) return;
#else
        if (!separate) return;
#endif
        writer->release_completed();
    }
}

int main(int argc, char ** argv) {
    const bool baseline_only = argc > 1 && std::strcmp(argv[1], "--bench-baseline") == 0;
    const bool bench = baseline_only || (argc > 1 && std::strcmp(argv[1], "--bench") == 0);
    const bool cache_bound=argc>1 && !std::strcmp(argv[1],"--cuda-routed-source-bound");
    const bool cache_sources=cache_bound || (argc>1 && !std::strcmp(argv[1],"--cuda-routed-source-cache"));
    const bool routed_publication=cache_sources || (argc>1 && !std::strcmp(argv[1],"--cuda-routed-publication"));
    const bool cuda = bench || routed_publication || (argc > 1 && std::strcmp(argv[1], "--cuda") == 0);
    ggml_backend_ptr backend;
    if (cuda) { ggml_backend_load_all(); auto * dev = ggml_backend_dev_by_name("CUDA0"); if (!dev) return 1; backend.reset(ggml_backend_dev_init(dev, nullptr)); }
    else backend.reset(ggml_backend_cpu_init());
    if (routed_publication) {
        testing t;t.test("routed_device_publication_precedes_independent_canonical_download",[&](testing & t) {check_routed_publication(t,backend.get(),cache_sources,cache_bound ? 65 : 3);});
        return t.summary();
    }
    if (bench) {
        std::cout << "policy,batch,median_us,p95_us,d2h_bytes,d2h_calls,h2d_bytes,h2d_calls,d2d_bytes,d2d_calls\n";
        for (size_t rows : {size_t(32), size_t(256), size_t(512)}) {
            writer_fixture f(backend.get(), true); encoder e(f, rows, GGML_TYPE_Q8_0);
            std::vector<uint8_t> bytes(rows*e.row_bytes);
            if (!baseline_only) GGML_ASSERT(f.resident->configure_writes(512));
            for (int mode = 0; mode < (baseline_only ? 2 : 4); ++mode) {
                const bool rowwise = mode == 1, staged = mode >= 2;
                arena_ptr page_arena{nullptr,ggml_backend_memory_arena_free};
                lease_ptr page_lease{nullptr,ggml_backend_memory_lease_free};
                if (staged) {
                    auto * type=llama_kv_stream_device_buffer_type(ggml_backend_get_device(backend.get()));
                    size_t scratch=0;
                    GGML_ASSERT(llama_kv_stream_writer::requirements(type,f.host->config().shape,512,scratch));
                    if (mode == 2) scratch=32768;
                    page_arena.reset(ggml_backend_memory_arena_new(type,scratch));
                    GGML_ASSERT(page_arena && ggml_backend_memory_arena_begin(page_arena.get(),0) &&
                        ggml_backend_memory_arena_reserve_at(page_arena.get(),93,0,scratch,128,0,nullptr) &&
                        ggml_backend_memory_arena_commit(page_arena.get()));
                    page_lease.reset(ggml_backend_memory_arena_acquire(page_arena.get(),93));
                    GGML_ASSERT(f.resident->configure_writes(512,page_lease.get()));
                }
                std::vector<double> times;
                for (int repeat = 0; repeat < 110; ++repeat) {
                    const auto start = std::chrono::steady_clock::now();
                    if (staged) {
                        GGML_ASSERT(f.resident->write_rows(0, ggml_kv_stream_operand::k, 0, e.source));
                        GGML_ASSERT(f.resident->synchronize(1024));
                    } else baseline(f, e, rowwise, bytes);
                    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
                    if (repeat >= 10) times.push_back(us);
                }
                std::sort(times.begin(), times.end());
                const auto stats = f.resident->last_write_stats();
                std::cout << (mode == 3 ? "staged_page" : staged ? "staged" : rowwise ? "rowwise" : "coalesced") << ',' << rows << ',' << times[50] << ',' << times[95] << ','
                    << bytes.size() << ',' << (staged ? stats.d2h_calls : rowwise ? rows : 1) << ',' << f.resident->last_upload_bytes() << ',' << f.resident->last_upload_calls()
                    << ',' << (staged ? stats.d2d_bytes : 0) << ',' << (staged ? stats.d2d_calls : 0) << '\n';
            }
        }
        return 0;
    }
    testing t;
    t.test("all_cache_pairs_publish_partial_and_cross_page_rows_like_stock", [&](testing & t) {
        const ggml_type types[] = {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q8_0,
            GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_IQ4_NL};
        for (auto key : types) for (auto value : types) {
            writer_fixture f(backend.get(), cuda, key, value);
            if (!t.assert_true(f.resident->configure_writes(512))) return;
            for (bool values : {false, true}) for (size_t count : {1u, 17u, 257u}) {
                encoder control(f, count, values ? value : key);
                control.run();
                std::vector<uint8_t> expected(count*control.row_bytes);
                ggml_backend_tensor_get(control.output, expected.data(), 0, expected.size());
                if (!t.assert_true(f.resident->write_rows(0, values ? ggml_kv_stream_operand::v : ggml_kv_stream_operand::k,
                        255, control.source))) return;
                if (!t.assert_true(f.resident->synchronize(1024))) return;
                llama_kv_stream_host_layer layer;
                if (!t.assert_true(f.host->layer(0, layer))) return;
                const auto * plane = static_cast<const uint8_t *>(values ? layer.v : layer.k);
                t.assert_true(std::memcmp(plane+255*control.row_bytes, expected.data(), expected.size()) == 0);
                t.assert_true(std::all_of(plane, plane+255*control.row_bytes, [](uint8_t byte) { return byte == 0; }));
                t.assert_true(std::all_of(plane+512*control.row_bytes, plane+513*control.row_bytes,
                    [](uint8_t byte) { return byte == 0; }));
            }
        }
    });
    t.test("joint_writer_routes_native_encoded_rows_to_independent_registered_histories", [&](testing & t) {
        const ggml_type types[]={GGML_TYPE_F32,GGML_TYPE_F16,GGML_TYPE_BF16,GGML_TYPE_Q8_0,GGML_TYPE_Q5_0,GGML_TYPE_Q5_1,GGML_TYPE_Q4_0,GGML_TYPE_Q4_1};
        size_t profiles=0;
        for (uint32_t dimension : {256u,512u}) for (auto key : types) for (auto val : types) for (bool value : {false,true}) {
            writer_fixture f(backend.get(),cuda,key,val,false,dimension);encoder e(f,257,value ? val : key);
            e.run();std::vector<uint8_t> expected(e.rows*e.row_bytes);
            ggml_backend_tensor_get(e.output,expected.data(),0,expected.size());
            std::unique_ptr<llama_kv_stream_host_reservation> hosts[2];
            for (uint32_t slot=0;slot<2;++slot) {
                auto config=f.host->config();config.cache_id=610+slot;
                if (cuda) hosts[slot]=llama_kv_stream_host_reservation::create(config,ggml_backend_get_device(backend.get()),65536);
                else hosts[slot]=llama_kv_stream_host_reservation::create(config,ggml_backend_cpu_buffer_type(),
                    [](ggml_backend_buffer_t owner) {owner->buft=ggml_backend_cpu_buffer_type();return ggml_backend_buffer_retain(owner);},65536);
                if (!t.assert_true(hosts[slot] && hosts[slot]->prepare(300))) return;
            }
            size_t cursors[2]={7,13};
            std::vector<llama_kv_stream_host_range> routes;
            for (size_t row=0;row<e.rows;++row) {
                const size_t slot=row%2;
                std::vector<llama_kv_stream_host_range> ranges;
                if (!t.assert_true(hosts[slot]->ranges(0,value ? ggml_kv_stream_operand::v : ggml_kv_stream_operand::k,
                        cursors[slot]++*e.row_bytes,e.row_bytes,ranges))) return;
                for (auto & range : ranges) routes.push_back(std::move(range));
            }
            for (const auto & range : routes) ggml_backend_buffer_clear(range.buffer.get(),0xa5);
            auto * type=ggml_backend_get_default_buffer_type(backend.get());size_t bytes=0;
            if (!t.assert_true(llama_kv_stream_writer::requirements(type,f.host->config().shape,32,bytes))) return;
            ggml_backend_buffer_ptr scratch(ggml_backend_buft_alloc_buffer(type,bytes));
            auto writer=llama_kv_stream_writer::create(backend.get(),scratch.get(),bytes,f.host->config().shape,512);
            if (!t.assert_true(writer!=nullptr)) return;
            llama_kv_stream_writer_completion rejected;
            t.assert_true(!writer->generate_async_registered(e.source,value,routes,{},rejected));
            t.assert_equal(size_t(0),writer->stats().graph_submissions);
            for (int fault=0;fault<4;++fault) {
                std::vector<llama_kv_stream_host_range> invalid;
                for (const auto & range : routes) invalid.push_back({ggml_backend_buffer_ptr(ggml_backend_buffer_retain(range.buffer.get())),range.offset,range.bytes});
                if (fault==0) {
                    size_t duplicate=1;
                    while (duplicate<invalid.size() && invalid[duplicate].bytes!=invalid[0].bytes) ++duplicate;
                    if (!t.assert_true(duplicate<invalid.size())) return;
                    invalid[duplicate].buffer.reset(ggml_backend_buffer_retain(invalid[0].buffer.get()));invalid[duplicate].offset=invalid[0].offset;
                } else if (fault==1) invalid.back().offset=ggml_backend_buffer_get_size(invalid.back().buffer.get())+1;
                else if (fault==2) invalid.back().offset=SIZE_MAX;
                else invalid.pop_back();
                llama_kv_stream_writer_completion refused;
                t.assert_true(!writer->generate_async_routed(e.source,value,invalid,{},refused));
                t.assert_equal(size_t(0),writer->stats().graph_submissions);
                t.assert_true(!refused.device && !refused.host);
            }
            std::vector<std::pair<const uint8_t *,size_t>> addresses;
            for (const auto & range : routes) {
                const auto * address=static_cast<const uint8_t *>(ggml_backend_buffer_get_base(range.buffer.get()))+range.offset;
                if (!t.assert_true(std::all_of(address,address+range.bytes,[](uint8_t byte) {return byte==0xa5;}))) return;
                addresses.push_back({address,range.bytes});
            }
            llama_kv_stream_writer_completion completion;
            if (!t.assert_true(writer->generate_async_routed(e.source,value,routes,{},completion))) return;
            routes.clear();hosts[0].reset();hosts[1].reset();
            if (!t.assert_true(completion.device && completion.host && completion.device->synchronize() && completion.host->synchronize())) return;
            size_t offset=0;
            for (const auto & range : addresses) {
                if (!t.assert_true("joint publication keeps native encoding and query order",!std::memcmp(expected.data()+offset,range.first,range.second))) return;
                offset+=range.second;
            }
            t.assert_equal(expected.size(),offset);
            t.assert_equal(expected.size(),writer->stats().d2h_bytes);
            t.assert_true(writer->stats().graph_submissions>1);
            writer->release_completed();
            ++profiles;
            t.out << "routed writer D=" << dimension << " K=" << ggml_type_name(key) << " V=" << ggml_type_name(val) << " value=" << value << std::endl;
        }
        t.assert_equal(size_t(256),profiles);
    });
    t.test("registered_writer_tiles_keep_native_bits_across_download_boundaries", [&](testing & t) {
        for (uint32_t dimension : {256u,512u}) for (bool value : {false,true}) {
            writer_fixture f(backend.get(), cuda);
            encoder e(f,257,value ? GGML_TYPE_Q4_0 : GGML_TYPE_Q8_0);
            e.run();
            std::vector<uint8_t> expected(e.rows*e.row_bytes), actual(expected.size());
            ggml_backend_tensor_get(e.output, expected.data(), 0, expected.size());
            auto c = f.host->config(); c.cache_id = 93;
            c.shape.head_dim_k=c.shape.head_dim_v=dimension;
            c.shape.heads=1024/dimension;
            std::unique_ptr<llama_kv_stream_host_reservation> sparse;
            if (cuda) sparse = llama_kv_stream_host_reservation::create(c, ggml_backend_get_device(backend.get()), 65536);
            else sparse = llama_kv_stream_host_reservation::create(c, ggml_backend_cpu_buffer_type(),
                [](ggml_backend_buffer_t owner) {
                    owner->buft = ggml_backend_cpu_buffer_type();
                    return ggml_backend_buffer_retain(owner);
                }, 65536);
            if (!t.assert_true(sparse && sparse->prepare(e.rows))) return;
            std::vector<llama_kv_stream_host_range> ranges;
            if (!t.assert_true(sparse->ranges(0,value ? ggml_kv_stream_operand::v : ggml_kv_stream_operand::k,0,expected.size(),ranges) && ranges.size() > 1)) return;
            std::vector<std::pair<const void *, size_t>> addresses;
            for (const auto & range : ranges) addresses.push_back({
                static_cast<const char *>(ggml_backend_buffer_get_base(range.buffer.get())) + range.offset, range.bytes});
            auto * type = ggml_backend_get_default_buffer_type(backend.get());
            size_t bytes = 0;
            if (!t.assert_true(llama_kv_stream_writer::requirements(type, c.shape, 512, bytes))) return;
            ggml_backend_buffer_ptr scratch(ggml_backend_buft_alloc_buffer(type, bytes));
            auto writer = llama_kv_stream_writer::create(backend.get(), scratch.get(), bytes, c.shape, 512);
            if (!t.assert_true(writer != nullptr)) return;
            for (const auto & range : ranges) ggml_backend_buffer_clear(range.buffer.get(),0xa5);
            std::vector<llama_kv_stream_host_range> duplicated;
            for (const auto & range : ranges) duplicated.push_back({
                ggml_backend_buffer_ptr(ggml_backend_buffer_retain(range.buffer.get())),range.offset,range.bytes});
            duplicated[1].buffer.reset(ggml_backend_buffer_retain(duplicated[0].buffer.get()));
            llama_kv_stream_writer_completion invalid;
            const bool accepted=writer->generate_async_registered(e.source,value,duplicated,{},invalid);
            if (accepted) {
                if (invalid.device) invalid.device->synchronize();
                if (invalid.host) invalid.host->synchronize();
            }
            t.assert_true("duplicate registrations reject before publication",!accepted);
            if (!accepted) {
                t.assert_equal(size_t(0),writer->stats().graph_submissions);
                bool untouched=true;
                for (const auto & range : addresses) {
                    const auto * first=static_cast<const uint8_t *>(range.first);
                    untouched &= std::all_of(first,first+range.second,[](uint8_t value) { return value == 0xa5; });
                }
                t.assert_true(untouched);
            }
            writer->release_completed();
            duplicated.clear();
            release_probe freed(ranges.front().buffer.get());
            llama_kv_stream_writer_completion completion;
            if (!t.assert_true(writer->generate_async_registered(e.source,value,ranges, {}, completion))) return;
            const auto stats = writer->stats();
            t.assert_equal(size_t(2), stats.graph_submissions);
            t.assert_equal(expected.size(), stats.d2h_bytes);
            t.assert_true(stats.d2h_calls >= ranges.size());
            t.assert_equal(bytes, stats.device_scratch_bytes);
            ranges.clear(); sparse.reset();
            t.assert_equal(0, freed.frees);
            t.assert_true(completion.device && completion.host && completion.device->synchronize() && completion.host->synchronize());
            size_t copied = 0;
            for (const auto & range : addresses) {
                std::memcpy(actual.data() + copied, range.first, range.second);
                copied += range.second;
            }
            t.assert_true(actual == expected);
            writer->release_completed();
            t.assert_equal(1, freed.frees);
        }
    });
    t.test("writer_quote_uses_independent_row_traits_and_bounded_tiles", [&](testing & t) {
        const ggml_type types[]={GGML_TYPE_F32,GGML_TYPE_F16,GGML_TYPE_BF16,GGML_TYPE_Q8_0,
            GGML_TYPE_Q4_0,GGML_TYPE_Q4_1,GGML_TYPE_IQ4_NL,GGML_TYPE_Q5_0,GGML_TYPE_Q5_1};
        auto * type=ggml_backend_get_default_buffer_type(backend.get());
        for (auto key : types) for (auto value : types) for (size_t maximum : {64u,256u,512u,1024u}) {
            const ggml_kv_stream_shape shape{key,value,256,256,4,256,128};
            size_t bytes=17;
            if (!t.assert_true(llama_kv_stream_writer::requirements(type,shape,maximum,bytes))) return;
            const size_t rows=std::min(maximum,size_t(256));
            const size_t encoded=std::max(ggml_row_size(key,1024),ggml_row_size(value,1024))*rows;
            t.assert_equal((encoded+127)/128*128+rows*sizeof(int64_t),bytes);
            if (maximum <= 256 && key == GGML_TYPE_Q8_0 && value == GGML_TYPE_Q4_0) {
                ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer(type,bytes));
                auto writer=llama_kv_stream_writer::create(backend.get(),buffer.get(),bytes,shape,maximum);
                if (!t.assert_true(bool(writer))) return;
            }
        }
        size_t bytes=17;
        ggml_kv_stream_shape shape{GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,256,256,4,256,128};
        t.assert_true(!llama_kv_stream_writer::requirements(type,shape,0,bytes));
        t.assert_true(!llama_kv_stream_writer::requirements(type,shape,SIZE_MAX,bytes));
        shape.heads=INT64_MAX;
        t.assert_true(!llama_kv_stream_writer::requirements(type,shape,256,bytes));
        t.assert_equal(size_t(17),bytes);
    });
    t.test("writer_restores_indices_after_scratch_contents_are_reused", [&](testing & t) {
        writer_fixture f(backend.get(), cuda);
        encoder e(f, 17, GGML_TYPE_Q8_0);
        e.run();
        std::vector<uint8_t> expected(e.rows*e.row_bytes), actual(expected.size());
        ggml_backend_tensor_get(e.output, expected.data(), 0, expected.size());
        auto * type = ggml_backend_get_default_buffer_type(backend.get());
        size_t bytes = 0;
        if (!t.assert_true(llama_kv_stream_writer::requirements(type, f.host->config().shape, 256, bytes))) return;
        ggml_backend_buffer_ptr scratch(ggml_backend_buft_alloc_buffer(type, bytes));
        auto writer = llama_kv_stream_writer::create(backend.get(), scratch.get(), bytes, f.host->config().shape, 256);
        if (!t.assert_true(bool(writer))) return;
        for (int repeat = 0; repeat < 2; ++repeat) {
            ggml_backend_buffer_clear(scratch.get(), 0);
            if (!t.assert_true(writer->generate(e.source, false, actual.data(), {}))) return;
            t.assert_true(actual == expected);
        }
    });
    t.test("ordinary_set_rows_baseline_populates_authoritative_cache", [&](testing & t) {
        writer_fixture f(backend.get(), cuda); encoder e(f, 33, GGML_TYPE_Q8_0);
        std::vector<uint8_t> bytes(e.rows*e.row_bytes);
        baseline(f, e, false, bytes);
        llama_kv_stream_host_layer host; f.host->layer(0, host);
        t.assert_true(std::memcmp(host.k, bytes.data(), bytes.size()) == 0);
        t.assert_equal(bytes.size(), f.resident->last_upload_bytes());
    });
    t.test("generated_ticket_is_atomic_and_cancellable", [&](testing & t) {
        writer_fixture f(backend.get(), cuda);
        uint8_t value = 7; llama_kv_stream_write ticket;
        const auto gen = f.content->generation();
        t.assert_true(!f.content->prepare_generated({{0, ggml_kv_stream_operand::k, 0, nullptr, 1}}, [](const auto &, void * out) {
            *static_cast<uint8_t *>(out) = 7; return false;
        }, ticket));
        t.assert_equal(gen, f.content->generation());
        t.assert_true(f.content->prepare_generated({{0, ggml_kv_stream_operand::k, 0, nullptr, 1}}, [&](const auto &, void * out) {
            *static_cast<uint8_t *>(out) = value; return true;
        }, ticket));
        ticket.cancel();
        llama_kv_stream_host_layer host; f.host->layer(0, host);
        t.assert_equal(uint8_t(0), *static_cast<uint8_t *>(host.k));
    });
    t.test("staged_rows_match_stock_quantization_and_avoid_h2d", [&](testing & t) {
        writer_fixture f(backend.get(), cuda); encoder e(f, 33, GGML_TYPE_Q8_0);
        e.run();
        std::vector<uint8_t> expected(e.rows*e.row_bytes);
        ggml_backend_tensor_get(e.output, expected.data(), 0, expected.size());
        if (!t.assert_true(f.resident->configure_writes(512))) return;
        t.assert_true(f.resident->write_rows(0, ggml_kv_stream_operand::k, 17, e.source));
        llama_kv_stream_host_layer host; f.host->layer(0, host);
        t.assert_true(std::memcmp(static_cast<uint8_t *>(host.k) + 17*e.row_bytes, expected.data(), expected.size()) == 0);
        t.assert_true(f.resident->synchronize(1024));
        t.assert_equal(size_t(0), f.resident->last_upload_bytes());
        auto stats = f.resident->last_write_stats();
        t.assert_equal(expected.size(), stats.d2h_bytes);
        t.assert_equal(expected.size(), stats.d2d_bytes);
        t.assert_equal(size_t(1), stats.graph_submissions);
    });
    t.test("quant_pairs_partial_batches_and_scratch_reuse", [&](testing & t) {
        for (auto pair : {std::pair{GGML_TYPE_F16, GGML_TYPE_F16}, std::pair{GGML_TYPE_Q8_0, GGML_TYPE_Q4_0},
                std::pair{GGML_TYPE_Q4_0, GGML_TYPE_Q8_0}, std::pair{GGML_TYPE_Q5_1, GGML_TYPE_Q4_1},
                std::pair{GGML_TYPE_BF16, GGML_TYPE_Q5_0}, std::pair{GGML_TYPE_IQ4_NL, GGML_TYPE_F32}}) {
            writer_fixture f(backend.get(), cuda, pair.first, pair.second);
            if (!t.assert_true(f.resident->configure_writes(512))) continue;
            for (bool value : {false, true}) for (size_t rows : {size_t(1), size_t(7), size_t(256), size_t(257), size_t(512)}) {
                encoder e(f, rows, value ? pair.second : pair.first); e.run();
                std::vector<uint8_t> expected(rows*e.row_bytes);
                ggml_backend_tensor_get(e.output, expected.data(), 0, expected.size());
                if (!t.assert_true(f.resident->write_rows(0, value ? ggml_kv_stream_operand::v : ggml_kv_stream_operand::k, 3, e.source))) continue;
                check_bytes(t, f, value, 3, expected);
                t.assert_true(f.resident->synchronize(1024));
                t.assert_equal(size_t(0), f.resident->last_upload_bytes());
                const auto stats = f.resident->last_write_stats();
                t.assert_equal(expected.size(), stats.host_payload_bytes);
                t.assert_equal(expected.size(), stats.d2h_bytes);
                t.assert_equal(expected.size(), stats.d2d_bytes);
                t.assert_true(stats.device_scratch_bytes < f.policy.pool_bytes);
            }
        }
    });
    t.test("tiled_failure_preserves_host_and_restores_mirror", [&](testing & t) {
        writer_fixture f(backend.get(), cuda, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, true);
        encoder e(f, 512, GGML_TYPE_Q8_0); e.run();
        std::vector<uint8_t> expected(e.rows*e.row_bytes), zero(expected.size(), 0);
        ggml_backend_tensor_get(e.output, expected.data(), 0, expected.size());
        if (!t.assert_true(f.resident->configure_writes(512))) return;
        const auto generation = f.content->generation();
        bool caught = false;
        {
            copy_fault fault(backend.get(), ggml_backend_memory_lease_buffer(f.lease.get()));
            try { f.resident->write_rows(0, ggml_kv_stream_operand::k, 5, e.source); }
            catch (const std::runtime_error &) { caught = true; }
            t.assert_equal(2, fault.calls);
        }
        t.assert_true(caught && !f.resident->ready(1024));
        t.assert_equal(generation, f.content->generation());
        t.assert_true(f.resident->synchronize(1024));
        check_bytes(t, f, false, 5, zero);
        t.assert_true(f.resident->write_rows(0, ggml_kv_stream_operand::k, 5, e.source));
        check_bytes(t, f, false, 5, expected);
        auto stats = f.resident->last_write_stats();
        t.assert_true(stats.tile_rows < 512 && stats.graph_submissions > 1);
        t.assert_true(stats.device_scratch_bytes <= f.policy.pool_bytes/8);
    });
    t.test("queued_reader_completes_before_resident_overwrite", [&](testing & t) {
        writer_fixture f(backend.get(), cuda, GGML_TYPE_F32, GGML_TYPE_F32);
        encoder e(f, 32, GGML_TYPE_F32);
        if (!t.assert_true(f.resident->configure_writes(512))) return;
        t.assert_true(f.resident->write_rows(0, ggml_kv_stream_operand::k, 0, e.source));
        llama_kv_stream_host_layer host; f.host->layer(0, host);
        std::vector<float> old(32*1024), next(old.size(), 7.0f), observed(old.size());
        std::memcpy(old.data(), host.k, old.size()*sizeof(float));
        ggml_backend_tensor_set(e.source, next.data(), 0, next.size()*sizeof(float));
        llama_kv_stream_policy_layout layout;
        GGML_ASSERT(llama_kv_stream_policy_layout_make(f.policy, f.binding->view()->initial_policy, 0, layout).status == llama_kv_stream_policy_status::success);
        ggml_context_ptr context(ggml_init({65536, nullptr, true}));
        auto * input = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, int64_t(old.size()));
        GGML_ASSERT(ggml_backend_tensor_alloc(ggml_backend_memory_lease_buffer(f.lease.get()), input,
            static_cast<char *>(f.binding->view()->base) + layout.layers[0].offset) == GGML_STATUS_SUCCESS);
        auto * output = ggml_scale(context.get(), input, 1.0f);
        auto * graph = ggml_new_graph_custom(context.get(), 16, false);
        ggml_build_forward_expand(graph, output);
        ggml_backend_buffer_ptr output_buffer(ggml_backend_alloc_ctx_tensors(context.get(), backend.get()));
        if (cuda) {
            llama_memory_cuda_executor reader(backend.get());
            t.assert_true(reader.bind(graph, {f.lease.get()}, 1));
            t.assert_true(reader.compute_async({f.lease.get()}, 1) == GGML_STATUS_SUCCESS);
            t.assert_true(f.resident->write_rows(0, ggml_kv_stream_operand::k, 0, e.source));
            t.assert_true(reader.drain());
        } else {
            t.assert_true(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS);
            t.assert_true(f.resident->write_rows(0, ggml_kv_stream_operand::k, 0, e.source));
        }
        ggml_backend_tensor_get(output, observed.data(), 0, observed.size()*sizeof(float));
        t.assert_true(observed == old);
        t.assert_true(std::memcmp(host.k, next.data(), next.size()*sizeof(float)) == 0);
    });
    t.test("invalid_source_and_bounds_do_not_mutate_content", [&](testing & t) {
        writer_fixture f(backend.get(), cuda); encoder e(f, 33, GGML_TYPE_Q8_0);
        const auto generation = f.content->generation();
        t.assert_true(!f.resident->write_rows(0, ggml_kv_stream_operand::k, 0, e.source));
        t.assert_true(!f.resident->configure_writes(0));
        t.assert_true(f.resident->configure_writes(32));
        t.assert_true(!f.resident->write_rows(0, ggml_kv_stream_operand::k, 0, e.source));
        t.assert_true(f.resident->configure_writes(512));
        t.assert_true(!f.resident->write_rows(0, ggml_kv_stream_operand::k, 1000, e.source));
        t.assert_true(!f.resident->write_rows(1, ggml_kv_stream_operand::k, 0, e.source));
        t.assert_true(!f.resident->write_rows(0, ggml_kv_stream_operand::none, 0, e.source));
        auto invalid = *e.source; --invalid.ne[0];
        t.assert_true(!f.resident->write_rows(0, ggml_kv_stream_operand::k, 0, &invalid));
        t.assert_equal(generation, f.content->generation());
        f.resident->release_write_workspace();
        t.assert_true(!f.resident->write_rows(0, ggml_kv_stream_operand::k, 0, e.source));
    });
    t.test("completed_source_graph_is_not_recomputed", [&](testing & t) {
        writer_fixture f(backend.get(), cuda); encoder e(f, 32, GGML_TYPE_Q8_0); e.run();
        std::vector<uint8_t> expected(e.rows*e.row_bytes);
        ggml_backend_tensor_get(e.output, expected.data(), 0, expected.size());
        if (!t.assert_true(f.resident->configure_writes(512))) return;
        const auto saved = *e.source;
        auto input = saved;
        e.source->op = GGML_OP_SCALE; e.source->src[0] = &input;
        const float scale = 3.0f; std::memcpy(e.source->op_params, &scale, sizeof(scale));
        const bool ok = f.resident->write_rows(0, ggml_kv_stream_operand::k, 0, e.source);
        *e.source = saved;
        t.assert_true(ok);
        check_bytes(t, f, false, 0, expected);
    });
    t.test("generated_failure_preserves_existing_ticket_and_admission", [&](testing & t) {
        writer_fixture f(backend.get(), cuda);
        uint8_t value = 4; llama_kv_stream_write ticket;
        t.assert_true(f.content->prepare({{0, ggml_kv_stream_operand::k, 0, &value, 1}}, ticket));
        int calls = 0;
        t.assert_true(!f.content->prepare_generated({{0, ggml_kv_stream_operand::v, 0, &value, 1}}, [&](const auto &, void *) { ++calls; return true; }, ticket));
        t.assert_equal(0, calls);
        t.assert_true(!f.content->prepare_generated({{0, ggml_kv_stream_operand::v, 0, nullptr, 1}}, [&](const auto &, void *) -> bool {
            t.assert_true(!f.content->invalidate());
            throw std::bad_alloc();
        }, ticket));
        t.assert_true(f.content->commit(ticket));
        llama_kv_stream_host_layer host; f.host->layer(0, host);
        t.assert_equal(value, *static_cast<uint8_t *>(host.k));
        t.assert_equal(uint8_t(0), *static_cast<uint8_t *>(host.v));
    });
    t.test("cached_writer_retains_destination_scope_through_final_completion", [&](testing & t) {
        for (size_t maximum : {3u,256u}) for (bool value : {false,true}) {
            writer_fixture f(backend.get(),cuda);
            encoder control(f,3,value ? GGML_TYPE_Q4_0 : GGML_TYPE_Q8_0);
            control.run();
            std::vector<uint8_t> expected(control.rows*control.row_bytes),actual(expected.size());
            ggml_backend_tensor_get(control.output,expected.data(),0,expected.size());
            auto * type=ggml_backend_get_default_buffer_type(backend.get());
            size_t bytes=0;
            if (!t.assert_true(llama_kv_stream_writer::requirements(type,f.host->config().shape,maximum,bytes))) return;
            ggml_backend_buffer_ptr parent(ggml_backend_buft_alloc_buffer(type,bytes+256));
            if (!t.assert_true(bool(parent))) return;
            ggml_backend_buffer_ptr scratch(ggml_backend_buffer_view(parent.get(),128,bytes));
            if (!t.assert_true(bool(scratch))) return;
            auto * identity=scratch.get();
            release_probe probe(identity);
            auto writer=llama_kv_stream_writer::create(backend.get(),identity,bytes,f.host->config().shape,maximum);
            if (!t.assert_true(bool(writer)) || !t.assert_true(writer->generate(control.source,value,actual.data(),{}))) return;
            if (!t.assert_true(actual==expected)) return;
            scratch.reset();
            if (!t.assert_equal("cached writer keeps destination metadata alive after the caller releases it",0,probe.frees)) return;
            ggml_backend_buffer_ptr replacement(ggml_backend_buffer_view(parent.get(),128,bytes));
            if (!t.assert_true(replacement && replacement.get()!=identity)) return;
            std::fill(actual.begin(),actual.end(),uint8_t(0xcd));
            llama_kv_stream_writer_completion completion;
            if (!t.assert_true(writer->generate_async(control.source,value,actual.data(),{},completion))) return;
            writer.reset();
            t.assert_equal(1,probe.frees);
            if (!t.assert_true(completion.host && completion.host->synchronize())) return;
            t.assert_true(actual==expected);
        }
    });
    t.test("release_writer_drops_cached_source_before_phase_reclaim", [&](testing & t) {
        writer_fixture f(backend.get(), cuda);
        auto e = std::make_unique<encoder>(f, 32, GGML_TYPE_Q8_0);
        release_probe probe(e->buffer.get());
        if (!t.assert_true(f.resident->configure_writes(512))) return;
        t.assert_true(f.resident->write_rows(0, ggml_kv_stream_operand::k, 0, e->source));
        e.reset();
        t.assert_equal(0, probe.frees);
        f.resident->release_write_workspace();
        t.assert_equal(1, probe.frees);
        t.assert_true(ggml_backend_memory_arena_lease_count(f.arena.get()) != 0);
        t.assert_true(f.resident->synchronize(1024));
    });
    return t.summary();
}
