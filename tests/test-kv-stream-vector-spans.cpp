#include "kv-stream-block-test.h"
#include "../ggml/src/ggml-cuda/kv-stream-attention-dispatch.h"
#include "../ggml/src/ggml-cuda/kv-stream-attention-plan.h"
#include "../ggml/src/ggml-cuda/kv-stream-span.h"
#include "../ggml/src/ggml-backend-execution.h"
#include "../src/llama-kv-stream-model.h"
#include "../src/llama-kv-stream-selection.h"
#include "../src/llama-kv-stream-resident.h"
#include "../ggml/src/ggml-kv-stream-registered-copy.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <utility>
#include <map>
#include <thread>

#ifdef KV_STREAM_PASCAL_CUDA_TEST
bool kv_stream_test_is_sm61_only();
bool kv_stream_test_is_arch_only(int cc);
size_t kv_stream_test_shared_limit(ggml_backend_t backend, size_t limit);
int kv_stream_test_override_cc(ggml_backend_t backend, int cc);
int kv_stream_test_compiled_cc(ggml_backend_t backend);
bool kv_stream_test_dimension_instance(ggml_backend_t backend,ggml_type key,ggml_type value,
        uint32_t width,uint32_t columns,uint32_t groups,bool mma,bool resume,bool dense,bool report=false);
bool kv_stream_test_runtime_shared_instances();
bool kv_stream_test_dimension_resources(ggml_backend_t backend, ggml_type key, ggml_type value,
        uint32_t width, uint32_t columns, uint32_t groups, bool resume, bool dense,
        size_t & shared_bytes, size_t & local_bytes, int & registers);
bool kv_stream_test_native_tile(ggml_backend_t backend,ggml_tensor * op);
bool kv_stream_test_tile_plan(ggml_backend_t backend,const ggml_tensor * op,size_t tokens,
        ggml_kv_stream_resume_plan & plan,size_t & bound);
bool kv_stream_test_tile_run(ggml_backend_t backend,const ggml_tensor * op,const ggml_kv_stream_span_plan_view & view,
        ggml_backend_buffer_t workspace,const ggml_kv_stream_resume_plan & plan,size_t first,size_t end);

struct pascal_test_guard {
    ggml_backend_t backend;
    int previous = 0;
    bool forced;
    // Only the SM61-only test mode changes host metadata; ordinary modern tests retain their real device.
    pascal_test_guard(ggml_backend_t backend, bool forced) : backend(backend), forced(forced) {
        if (forced) previous = kv_stream_test_override_cc(backend,610);
    }
    ~pascal_test_guard() { if (forced) kv_stream_test_override_cc(backend,previous); }
};

struct tg1_only_test_ops {
    inline static tg1_only_test_ops * active = nullptr;
    ggml_backend_reg_t reg;
    decltype(ggml_backend_reg_i::get_proc_address) original;
    ggml_kv_stream_partial_ops limited;
    decltype(ggml_kv_stream_partial_ops::resume_plan) original_plan;

    // Retain the TG1-only handoff regression even after the real backend gains TG2 support.
    tg1_only_test_ops(ggml_backend_reg_t reg, const ggml_kv_stream_partial_ops & ops) :
        reg(reg), original(reg->iface.get_proc_address), limited(ops), original_plan(ops.resume_plan) {
        GGML_ASSERT(!active); active = this;
        // Exercise the older vector-only provider, not the new four-width workspace capability.
        limited.version=9; limited.decode_workspace=nullptr;
        limited.resume_plan = [](ggml_backend_t backend, int32_t k, int32_t v,
                uint32_t heads, uint32_t kv_heads, uint32_t queries, size_t tokens, ggml_kv_stream_resume_plan & plan) {
            return queries == 1 && active->original_plan(backend,k,v,heads,kv_heads,queries,tokens,plan);
        };
        reg->iface.get_proc_address = [](ggml_backend_reg_t reg, const char * name) -> void * {
            if (!std::strcmp(name,"ggml_backend_kv_stream_partial_ops"))
                return reinterpret_cast<void *>(+[]() -> const ggml_kv_stream_partial_ops * { return &active->limited; });
            return active->original(reg,name);
        };
    }
    ~tg1_only_test_ops() { reg->iface.get_proc_address = original; active = nullptr; }
    tg1_only_test_ops(const tg1_only_test_ops &) = delete;
    tg1_only_test_ops & operator=(const tg1_only_test_ops &) = delete;
};
#endif

struct span_storage {
    std::unique_ptr<block_workspace> storage;
    lease_ptr borrowed_lease{nullptr,ggml_backend_memory_lease_free};
    ggml_kv_stream_span_plan_t plan = nullptr;

    ~span_storage() {
        ggml_kv_stream_span_plan_free(plan);
    }

    static std::unique_ptr<span_storage> create(
            fixture & f, uint32_t layer, size_t active, uint32_t queries,
            const std::vector<size_t> & cuts, bool wrapped = false, ggml_backend_memory_lease_t borrowed = nullptr,
            bool keep_prefix = true) {
        if (cuts.size() < 2 || cuts.front() != 0 || cuts.back() != active) return {};
        auto result = std::make_unique<span_storage>();
        std::vector<ggml_kv_stream_layout> layouts;
        std::vector<size_t> offsets;
        size_t bytes = 0;
        for (size_t i = 0; i + 1 < cuts.size(); ++i) {
            const size_t tokens = cuts[i + 1] - cuts[i];
            if (!tokens) return {};
            ggml_kv_stream_layout layout;
            if (ggml_kv_stream_layout_make(f.policy.shape, tokens, layout).status !=
                    ggml_kv_stream_status::success) return {};
            bytes = (bytes + 127)/128*128;
            offsets.push_back(bytes);
            layouts.push_back(layout);
            if (layout.bytes > SIZE_MAX - bytes) return {};
            bytes += layout.bytes;
        }
        if (wrapped) {
            bytes = 0;
            for (size_t i = 0; i < layouts.size(); ++i) {
                const size_t logical = keep_prefix ? (i == 0 ? 0 : layouts.size()-i) : layouts.size()-1-i;
                bytes = (bytes+127)/128*128;
                offsets[logical] = bytes;
                bytes += layouts[logical].bytes;
            }
        }
        if (borrowed) {
            auto * buffer=ggml_backend_memory_lease_buffer(borrowed);
            auto * type=f.cuda ? llama_kv_stream_device_buffer_type(ggml_backend_get_device(f.backend)) : ggml_backend_cpu_buffer_type();
            if (!buffer || ggml_backend_buffer_get_type(buffer)!=type || bytes>ggml_backend_buffer_get_size(buffer)) return {};
            result->borrowed_lease.reset(ggml_backend_memory_lease_retain(borrowed));
        } else result->storage = std::make_unique<block_workspace>(f, bytes, 71);
        auto * lease=borrowed ? result->borrowed_lease.get() : result->storage->lease.get();
        auto * buffer = ggml_backend_memory_lease_buffer(lease);
        auto * base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(buffer));
        llama_kv_stream_host_layer host;
        if (!f.host->layer(layer, host)) return {};
        const auto & full = f.host->layout();
        std::vector<ggml_kv_stream_span_source> sources;
        for (size_t i = 0; i < layouts.size(); ++i) {
            const size_t first = cuts[i], tokens = cuts[i + 1] - cuts[i];
            const auto & layout = layouts[i];
            const size_t k_offset = offsets[i], v_offset = offsets[i] + layout.v_offset;
            ggml_tensor k = {}, v = {};
            for (auto * tensor : {&k, &v}) {
                tensor->ne[0] = tensor == &k ? f.policy.shape.head_dim_k : f.policy.shape.head_dim_v;
                tensor->ne[1] = int64_t(tokens);
                tensor->ne[2] = f.policy.shape.heads;
                tensor->ne[3] = 1;
                tensor->buffer = buffer;
            }
            k.type = ggml_type(f.policy.shape.type_k);
            k.nb[0] = ggml_type_size(k.type);
            k.nb[1] = layout.k_token_bytes;
            k.nb[2] = layout.k_row_bytes;
            k.nb[3] = layout.k_bytes;
            k.data = base + k_offset;
            v.type = ggml_type(f.policy.shape.type_v);
            v.nb[0] = ggml_type_size(v.type);
            v.nb[1] = layout.v_token_bytes;
            v.nb[2] = layout.v_row_bytes;
            v.nb[3] = layout.v_bytes;
            v.data = base + v_offset;
            ggml_backend_tensor_set(&k,
                static_cast<const uint8_t *>(host.k) + first*full.k_token_bytes,
                0, layout.k_bytes);
            ggml_backend_tensor_set(&v,
                static_cast<const uint8_t *>(host.v) + first*full.v_token_bytes,
                0, layout.v_bytes);
            sources.push_back({
                lease, lease,
                first, tokens, k_offset, v_offset});
        }
        if (ggml_kv_stream_span_plan_make(
                f.policy.shape, sources.data(), sources.size(), active, queries, result->plan).status !=
                ggml_kv_stream_status::success) return {};
        return result;
    }
};

static std::vector<float> ordinary(fixture & f, block_inputs & input, uint32_t layer) {
    ggml_context_ptr ctx(ggml_init({65536, nullptr, true}));
    auto * k = ggml_new_tensor_2d(ctx.get(), ggml_type(f.policy.shape.type_k), 512, input.padded);
    auto * v = ggml_new_tensor_2d(ctx.get(), ggml_type(f.policy.shape.type_v), 512, input.padded);
    auto * key = ggml_view_3d(ctx.get(), k, 256, input.padded, 2,
        ggml_row_size(k->type, 512), ggml_row_size(k->type, 256), 0);
    auto * value = ggml_view_3d(ctx.get(), v, 256, input.padded, 2,
        ggml_row_size(v->type, 512), ggml_row_size(v->type, 256), 0);
    auto * out = ggml_flash_attn_ext(ctx.get(), input.q, key, value, input.mask, 1.0f/16, 0, 0);
    ggml_prec_set_acc(out, GGML_PREC_F32);
    auto * graph = ggml_new_graph_custom(ctx.get(), 64, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), f.backend));
    GGML_ASSERT(buffer);
    llama_kv_stream_host_layer host;
    GGML_ASSERT(f.host->layer(layer, host));
    ggml_backend_tensor_set(k, host.k, 0, ggml_nbytes(k));
    ggml_backend_tensor_set(v, host.v, 0, ggml_nbytes(v));
    GGML_ASSERT(ggml_backend_graph_compute(f.backend, graph) == GGML_STATUS_SUCCESS);
    std::vector<float> result(ggml_nelements(out));
    ggml_backend_tensor_get(out, result.data(), 0, ggml_nbytes(out));
    return result;
}

static std::vector<float> evaluate(
        fixture & f, const ggml_kv_stream_partial_ops * ops,
        span_storage & storage, block_inputs & input, uint32_t layer, bool & accepted,
        size_t * used_workspace = nullptr, float scale = 1.0f/16) {
    ggml_context_ptr ctx(ggml_init({65536, nullptr, true}));
    auto * node = f.resident->attention(ctx.get(), layer, input.q, input.mask, input.active, scale);
    if (!node) {
        accepted = false;
        return {};
    }
    ggml_tensor op = *node;
    op.buffer = input.output->buffer;
    op.data = input.output->data;
    size_t workspace_bytes = 0;
    if (!ops->spans_workspace(f.backend, &op, storage.plan, workspace_bytes)) {
        accepted = false;
        return {};
    }
    if (used_workspace) *used_workspace = workspace_bytes;
    block_workspace workspace(f, workspace_bytes, 79);
    accepted = ops->spans(
        f.backend, &op, storage.plan,
        ggml_backend_memory_lease_buffer(workspace.lease.get()));
    ggml_backend_synchronize(f.backend);
    return accepted ? input.read() : std::vector<float>{};
}


struct stock_attention_case {
    fixture & f;
    ggml_context_ptr context;
    ggml_backend_buffer_ptr buffer;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * k_storage = nullptr;
    ggml_tensor * v_storage = nullptr;
    ggml_tensor * k = nullptr;
    ggml_tensor * v = nullptr;
    ggml_tensor * output = nullptr;

    stock_attention_case(fixture & f, block_inputs & input) : f(f), context(ggml_init({65536, nullptr, true})) {
        const int64_t heads=f.policy.shape.heads;
        const int64_t dk=f.policy.shape.head_dim_k, dv=f.policy.shape.head_dim_v;
        k_storage = ggml_new_tensor_2d(context.get(), ggml_type(f.policy.shape.type_k), dk*heads, input.padded);
        v_storage = ggml_new_tensor_2d(context.get(), ggml_type(f.policy.shape.type_v), dv*heads, input.padded);
        k = ggml_view_3d(context.get(), k_storage, dk, input.padded, heads,
            ggml_row_size(k_storage->type, dk*heads), ggml_row_size(k_storage->type, dk), 0);
        v = ggml_view_3d(context.get(), v_storage, dv, input.padded, heads,
            ggml_row_size(v_storage->type, dv*heads), ggml_row_size(v_storage->type, dv), 0);
        output = ggml_flash_attn_ext(context.get(), input.q, k, v, input.mask, 1.0f/16, 0, 0);
        ggml_prec_set_acc(output, GGML_PREC_F32);
        graph = ggml_new_graph_custom(context.get(), 64, false);
        ggml_build_forward_expand(graph, output);
        buffer.reset(ggml_backend_alloc_ctx_tensors(context.get(), f.backend));
        GGML_ASSERT(buffer);
        llama_kv_stream_host_layer host;
        GGML_ASSERT(f.host->layer(0, host));
        ggml_backend_tensor_set(k_storage, host.k, 0, ggml_nbytes(k_storage));
        ggml_backend_tensor_set(v_storage, host.v, 0, ggml_nbytes(v_storage));
    }

    ~stock_attention_case() {
        if (!f.cuda || !graph || !ggml_graph_n_nodes(graph)) return;
        auto * reg=ggml_backend_dev_backend_reg(ggml_backend_get_device(f.backend));
        const auto release=reinterpret_cast<void (*)(ggml_backend_t,const void *)>(
            ggml_backend_reg_get_proc_address(reg,"ggml_backend_cuda_graph_release"));
        if (release) {ggml_backend_synchronize(f.backend);release(f.backend,ggml_graph_node(graph,0));}
    }

    void run() {
        GGML_ASSERT(ggml_backend_graph_compute(f.backend, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(f.backend);
    }
};

struct span_attention_case {
    fixture & f;
    const ggml_kv_stream_partial_ops * ops;
    span_storage & storage;
    ggml_context_ptr context;
    ggml_tensor op = {};
    std::unique_ptr<block_workspace> workspace;
    ggml_kv_stream_resume_plan continuation{};
    bool resumable=false;

    span_attention_case(
            fixture & f, const ggml_kv_stream_partial_ops * ops, stock_attention_case & stock,
            span_storage & storage, block_inputs & input, ggml_tensor * compact = nullptr) :
        f(f), ops(ops), storage(storage), context(ggml_init({65536, nullptr, true})) {
        auto * node = ggml_flash_attn_ext(
            context.get(),input.q,stock.k,stock.v,input.mask,1.0f/16,0,0);
        GGML_ASSERT(node);
        ggml_prec_set_acc(node, GGML_PREC_F32);
        op = *node;
        op.buffer = input.output->buffer;
        op.data = input.output->data;
        size_t bytes = 0;
        if (compact) {
            op.src[3]=nullptr;GGML_ASSERT(ggml_backend_execution_set_mask_bounds(&op,compact));
            const ggml_kv_stream_attention_request request{f.policy.shape,uint32_t(input.query_heads),uint32_t(input.queries),input.padded};
            GGML_ASSERT(ggml_kv_stream_resume_plan_query(ops,f.backend,request,continuation) &&
                ggml_kv_stream_workspace_query(ops,f.backend,request,ggml_kv_stream_workspace_kind::decode,bytes));
            resumable=true;
        } else GGML_ASSERT(ops->spans_workspace(f.backend, &op, storage.plan, bytes));
        workspace = std::make_unique<block_workspace>(f, bytes, 97);
    }

    void run() {
        auto * scratch=ggml_backend_memory_lease_buffer(workspace->lease.get());
        if (!resumable) {
            GGML_ASSERT(ops->spans(f.backend,&op,storage.plan,scratch));
        } else {
            ggml_kv_stream_span_plan_view view;
            GGML_ASSERT(ggml_kv_stream_span_plan_get_view(storage.plan,view) && view.active_tokens==continuation.tokens);
            if (continuation.span_resume) {
                GGML_ASSERT(ops->resume_spans(f.backend,&op,scratch,continuation,view,0,continuation.tokens));
            } else for (size_t i=0;i<view.count;++i) {
                const auto & span=view.spans[i];
                ggml_kv_stream_layout layout;
                GGML_ASSERT(ggml_kv_stream_layout_make(view.shape,span.tokens,layout).status==ggml_kv_stream_status::success);
                ggml_tensor key{},value{};
                for (auto * tensor : {&key,&value}) {
                    const bool v=tensor==&value;
                    tensor->type=ggml_type(v ? view.shape.type_v : view.shape.type_k);
                    tensor->ne[0]=v ? view.shape.head_dim_v : view.shape.head_dim_k;
                    tensor->ne[1]=int64_t(span.tokens);tensor->ne[2]=view.shape.heads;tensor->ne[3]=1;
                    tensor->nb[0]=ggml_type_size(tensor->type);tensor->nb[1]=v ? layout.v_token_bytes : layout.k_token_bytes;
                    tensor->nb[2]=v ? layout.v_row_bytes : layout.k_row_bytes;tensor->nb[3]=v ? layout.v_bytes : layout.k_bytes;
                    tensor->buffer=v ? span.v_buffer : span.k_buffer;
                    tensor->data=static_cast<uint8_t *>(ggml_backend_buffer_get_base(tensor->buffer))+(v ? span.v_offset : span.k_offset);
                }
                auto step=op;step.src[1]=&key;step.src[2]=&value;
                GGML_ASSERT(ops->resume(f.backend,&step,scratch,continuation,continuation.tokens,span.token_begin,i+1==view.count));
            }
        }
        ggml_backend_synchronize(f.backend);
    }
};

template<typename F>
static double average_us(F && run, int warmup = 16, int iterations = 64) {
    for (int i = 0; i < warmup; ++i) run();
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) run();
    return std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - begin).count()/iterations;
}

// Synthetic one-layer timing used to compare dispatch shapes, not full-model tokens per second.
static int benchmark_spans(bool wide = false, ggml_type type_k = GGML_TYPE_Q8_0, ggml_type type_v = GGML_TYPE_Q4_0,
        size_t context_tokens = 0, uint32_t query_tokens = 0) {
    ggml_backend_load_all();
    auto * dev = ggml_backend_dev_by_name("CUDA0");
    GGML_ASSERT(dev);
    ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
    auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(
        ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev), "ggml_backend_kv_stream_partial_ops"));
    const auto * ops = get ? get() : nullptr;
    GGML_ASSERT(ops && ops->version >= 7 && ops->spans && ops->spans_workspace);
    const auto contexts=context_tokens ? std::vector<size_t>{context_tokens} :
        wide ? std::vector<size_t>{2048,8192,16384,65536} : std::vector<size_t>{16384,65536,131072,131073};
    const auto query_counts=query_tokens ? std::vector<uint32_t>{query_tokens} :
        wide ? std::vector<uint32_t>{1,2,3,4,8,256} : std::vector<uint32_t>{1,2,3,4,5,6,7,8};
    for (size_t active : contexts) {
        for (uint32_t queries : query_counts) {
            const uint32_t width=wide ? 512 : 256;
            fixture f(backend.get(), true, type_k, type_v, (active+255)/256*256,false,2,4,0,width,width);
            GGML_ASSERT(f.attach());
            auto pin = f.binding->acquire();
            GGML_ASSERT(pin);
            block_inputs input(f, active, queries,false,wide ? 32 : 24);
            stock_attention_case stock(f, input);
            const double stock_us = average_us([&] { stock.run(); });
            const std::vector<std::vector<size_t>> layouts{
                {0, active},
                {0, active/2, active},
                {0, active/4, 3*active/4, active},
            };
            for (const auto & cuts : layouts) {
                auto storage = span_storage::create(f, 0, active, queries, cuts);
                GGML_ASSERT(storage && storage->plan);
                span_attention_case spans(f,ops,stock,*storage,input);
                stock.run();
                std::vector<float> expected(ggml_nelements(stock.output));
                ggml_backend_tensor_get(stock.output,expected.data(),0,ggml_nbytes(stock.output));
                spans.run();
                if (!same_float_bits(expected,input.read())) {
                    std::fprintf(stderr,"span benchmark numerical mismatch: active=%zu TG%u spans=%zu\n",active,queries,cuts.size()-1);
                    return 1;
                }
                const double span_us = average_us([&] { spans.run(); });
                if (wide) std::printf("wide-span-format,k=%s,v=%s,width=%u\n",ggml_type_name(type_k),ggml_type_name(type_v),width);
                std::printf(
                    "span-bench,active=%zu,tg=%u,spans=%zu,stock_us=%.3f,span_us=%.3f,overhead_pct=%.2f\n",
                    active, queries, cuts.size()-1, stock_us, span_us, 100.0*(span_us/stock_us-1.0));
            }
        }
    }
    return 0;
}

// Compare the tile decoder with the native whole-plane converter at the requested head width.
static void check_mma_tile_conversion(testing & t, int width) {
    ggml_backend_load_all();
    auto * dev = ggml_backend_dev_by_name("CUDA0");
    ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
    auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(
        ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev),"ggml_backend_kv_stream_partial_ops"));
    const auto * ops = get ? get() : nullptr;
    if (!t.assert_true(ops && ops->version >= 8 && ops->convert && ops->convert_mma_rows)) return;
    for (ggml_type type : {GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,GGML_TYPE_Q4_1,GGML_TYPE_Q5_0,
            GGML_TYPE_Q5_1,GGML_TYPE_F16,GGML_TYPE_BF16,GGML_TYPE_F32}) {
        ggml_context_ptr ctx(ggml_init({65536,nullptr,true}));
        auto * source = ggml_new_tensor_2d(ctx.get(),type,width,7);
        auto * stock = ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F16,width,7);
        auto * tile = ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F16,width,7);
        for (auto * tensor : {source,stock,tile}) tensor->nb[2] = tensor->nb[1];
        ggml_set_input(source);
        ggml_set_output(stock);
        ggml_set_output(tile);
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()));
        if (!t.assert_true(buffer != nullptr)) continue;
        std::vector<float> values(7*width);
        for (size_t i = 0; i < values.size(); ++i) values[i] = .25f*std::sin(float(i%509)*.031f);
        std::vector<uint8_t> packed(7*ggml_row_size(type,width));
        ggml_quantize_chunk(type,values.data(),packed.data(),0,7,width,nullptr);
        ggml_backend_tensor_set(source,packed.data(),0,packed.size());
        if (!t.assert_true(ops->supports_conversion(backend.get(),source,stock) &&
                ops->convert(backend.get(),source,stock))) continue;
        if (!t.assert_true(ops->convert_mma_rows(backend.get(),source,tile))) continue;
        ggml_backend_synchronize(backend.get());
        std::vector<ggml_fp16_t> expected(7*width),actual(7*width);
        ggml_backend_tensor_get(stock,expected.data(),0,expected.size()*sizeof(ggml_fp16_t));
        ggml_backend_tensor_get(tile,actual.data(),0,actual.size()*sizeof(ggml_fp16_t));
        size_t mismatches = 0;
        for (size_t i = 0; i < expected.size(); ++i) mismatches += expected[i] != actual[i];
        t.out << ggml_type_name(type) << " tile mismatches = " << mismatches << " / " << expected.size() << "\n";
        t.assert_equal(size_t(0),mismatches);
    }
}

// Operator-only experiment: one decoded chunk, existing converters, unchanged resume order.
struct prepared_prefill_case {
    fixture & f;
    block_inputs & input;
    stock_attention_case & stock;
    const ggml_kv_stream_partial_ops * ops;
    size_t chunk;
    ggml_kv_stream_shape decoded_shape;
    ggml_kv_stream_layout decoded_layout;
    ggml_kv_stream_resume_plan plans[2];
    ggml_tensor prototypes[2];
    std::unique_ptr<block_workspace> stage, scratch;
    ggml_context_ptr bounds_context;
    ggml_backend_buffer_ptr bounds_buffer;
    ggml_tensor * bounds = nullptr;

    prepared_prefill_case(fixture & f, block_inputs & input, stock_attention_case & stock,
            const ggml_kv_stream_partial_ops * ops, size_t chunk) :
        f(f), input(input), stock(stock), ops(ops), chunk(chunk), decoded_shape(f.policy.shape) {}

    bool initialize() {
        if (!ops || !ops->convert_mma_rows || !ops->resume_spans || !chunk || chunk%256) return false;
        decoded_shape.type_k = decoded_shape.type_v = GGML_TYPE_F16;
        if (ggml_kv_stream_layout_make(decoded_shape,chunk,decoded_layout).status != ggml_kv_stream_status::success) return false;
        for (int prepared = 0; prepared < 2; ++prepared) {
            const auto & shape = prepared ? decoded_shape : f.policy.shape;
            const ggml_kv_stream_attention_request request{shape,uint32_t(input.query_heads),uint32_t(input.queries),input.padded};
            if (!ggml_kv_stream_resume_plan_query(ops,f.backend,request,plans[prepared]) || !plans[prepared].span_resume) return false;
        }
        stage = std::make_unique<block_workspace>(f,decoded_layout.bytes,123);
        scratch = std::make_unique<block_workspace>(f,std::max(plans[0].bytes,plans[1].bytes),127);
        for (int plane = 0; plane < 2; ++plane) {
            auto & tensor = prototypes[plane];
            tensor = *(plane ? stock.v : stock.k);
            tensor.type = GGML_TYPE_F16;
            tensor.nb[0] = sizeof(ggml_fp16_t);
            tensor.nb[1] = ggml_row_size(tensor.type,tensor.ne[0]*tensor.ne[2]);
            tensor.nb[2] = ggml_row_size(tensor.type,tensor.ne[0]);
            tensor.nb[3] = tensor.nb[1]*size_t(tensor.ne[1]);
            tensor.data = nullptr;
            tensor.buffer = nullptr;
        }
        return true;
    }

    bool same_plan() const {
        const auto & left = plans[0];
        const auto & right = plans[1];
        return left.tokens == right.tokens && left.splits == right.splits && left.heads == right.heads &&
            left.queries == right.queries && left.state_bytes == right.state_bytes && left.bytes == right.bytes &&
            left.partial_offset == right.partial_offset && left.meta_offset == right.meta_offset &&
            left.values_per_thread == right.values_per_thread &&
            std::equal(std::begin(left.kernel_config),std::end(left.kernel_config),std::begin(right.kernel_config));
    }

    bool run(bool prepared, ggml_backend_buffer_t external = nullptr, bool compact = false, bool contiguous = false) {
        auto op = *stock.output;
        op.buffer = input.output->buffer;
        op.data = input.output->data;
        if (prepared) {
            op.src[1] = &prototypes[0];
            op.src[2] = &prototypes[1];
        }
        if (compact) {
            if (!bounds) {
                bounds_context.reset(ggml_init({65536,nullptr,true}));
                if (!bounds_context) return false;
                bounds = ggml_new_tensor_2d(bounds_context.get(),GGML_TYPE_I32,4,int64_t(input.queries));
                bounds_buffer.reset(ggml_backend_alloc_ctx_tensors(bounds_context.get(),f.backend));
                if (!bounds_buffer) return false;
                std::vector<int32_t> intervals(4*input.queries);
                for (size_t row = 0; row < input.queries; ++row) {
                    intervals[4*row] = 0;
                    intervals[4*row+1] = int32_t(input.active-input.queries+row+1);
                    intervals[4*row+2] = int32_t(input.padded);
                    intervals[4*row+3] = 1;
                }
                ggml_backend_tensor_set(bounds,intervals.data(),0,ggml_nbytes(bounds));
            }
            op.src[3] = nullptr;
            if (!ggml_backend_execution_set_mask_bounds(&op,bounds)) return false;
        }
        ggml_backend_execution_set_stock_conversion(&op,true);
        if (contiguous) {
            if (!external || prepared || !ops->resume(f.backend,&op,external,plans[0],input.padded,0,true)) return false;
            ggml_backend_synchronize(f.backend);
            return true;
        }
        auto * stage_buffer = ggml_backend_memory_lease_buffer(stage->lease.get());
        for (size_t first = 0; first < input.padded; first += chunk) {
            const size_t count = std::min(chunk,input.padded-first);
            ggml_kv_stream_span span;
            span.token_begin = first;
            span.tokens = count;
            for (int plane = 0; plane < 2; ++plane) {
                const auto * storage = plane ? stock.v_storage : stock.k_storage;
                const size_t row = ggml_row_size(storage->type,f.policy.shape.head_dim_k);
                const size_t source_offset = first*size_t(f.policy.shape.heads)*row;
                const size_t destination_offset = plane ? decoded_layout.v_offset : 0;
                if (prepared) {
                    auto source = *storage;
                    source.ne[0] = f.policy.shape.head_dim_k;
                    source.ne[1] = int64_t(count)*f.policy.shape.heads;
                    source.ne[2] = source.ne[3] = 1;
                    source.nb[1] = row;
                    source.nb[2] = source.nb[3] = row*size_t(source.ne[1]);
                    source.data = static_cast<char *>(storage->data)+source_offset;
                    auto destination = source;
                    destination.type = GGML_TYPE_F16;
                    destination.nb[0] = sizeof(ggml_fp16_t);
                    destination.nb[1] = size_t(destination.ne[0])*sizeof(ggml_fp16_t);
                    destination.nb[2] = destination.nb[3] = destination.nb[1]*size_t(destination.ne[1]);
                    destination.buffer = stage_buffer;
                    destination.data = static_cast<char *>(ggml_backend_buffer_get_base(stage_buffer))+destination_offset;
                    if (!ops->convert_mma_rows(f.backend,&source,&destination)) return false;
                }
                const auto buffer = prepared ? stage_buffer : storage->buffer;
                const size_t offset = prepared ? destination_offset : source_offset+
                    size_t(static_cast<char *>(storage->data)-static_cast<char *>(ggml_backend_buffer_get_base(buffer)));
                if (plane) { span.v_buffer = buffer; span.v_offset = offset; }
                else { span.k_buffer = buffer; span.k_offset = offset; }
            }
            const ggml_kv_stream_span_plan_view view{prepared ? decoded_shape : f.policy.shape,&span,1,input.padded,input.queries};
            if (!ops->resume_spans(f.backend,&op,external ? external : ggml_backend_memory_lease_buffer(scratch->lease.get()),
                    plans[prepared],view,first,first+count)) return false;
        }
        ggml_backend_synchronize(f.backend);
        return true;
    }
};

static void check_prepared_prefill(testing & t, bool benchmark, bool automatic = false) {
    ggml_backend_load_all();
    auto * dev = ggml_backend_dev_by_name("CUDA0");
    if (!t.assert_true(dev != nullptr)) return;
    ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
    auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(dev),"ggml_backend_kv_stream_partial_ops"));
    const auto * ops = get ? get() : nullptr;
    const ggml_type formats[] = {GGML_TYPE_F32,GGML_TYPE_F16,GGML_TYPE_BF16,GGML_TYPE_Q8_0,
        GGML_TYPE_Q5_0,GGML_TYPE_Q5_1,GGML_TYPE_Q4_0,GGML_TYPE_Q4_1};
    for (uint32_t width : {256u,512u}) for (auto key : formats) for (auto value : formats) {
        if (benchmark && !((key == GGML_TYPE_Q8_0 && (value == GGML_TYPE_Q4_0 || value == GGML_TYPE_Q8_0)) ||
                (key == GGML_TYPE_Q5_1 && value == GGML_TYPE_Q5_1) || (key == GGML_TYPE_F16 && value == GGML_TYPE_F16))) continue;
        for (size_t active : benchmark ? std::vector<size_t>{8192,65536,131072} : std::vector<size_t>{2049})
                for (uint32_t queries : automatic ? std::vector<uint32_t>{64,128,129,241,255,256} : std::vector<uint32_t>{256}) {
            fixture f(backend.get(),true,key,value,(active+255)/256*256,false,2,4,0,width,width);
            if (!t.assert_true(f.attach())) return;
            block_inputs input(f,active,queries,false,width == 512 ? 32 : 24);
            stock_attention_case stock(f,input);
            stock.run();
            std::vector<float> expected(ggml_nelements(stock.output));
            ggml_backend_tensor_get(stock.output,expected.data(),0,ggml_nbytes(stock.output));
            prepared_prefill_case candidate(f,input,stock,ops,benchmark ? 2048 : 1024);
            if (!t.assert_true(candidate.initialize())) return;
            t.out << "prepared-prefill width=" << width << " k=" << ggml_type_name(key) << " v=" << ggml_type_name(value)
                << " active=" << active << " stage_bytes=" << candidate.decoded_layout.bytes << " queries=" << queries << '\n';
            if (!t.assert_true("same native split and checkpoint plan",candidate.same_plan())) return;
            if (!t.assert_true(candidate.run(false))) return;
            if (!t.assert_true("encoded waves preserve stock bits",same_float_bits(expected,input.read()))) return;
            if (!t.assert_true(candidate.run(true))) return;
            if (!t.assert_true("prepared chunk preserves stock bits",same_float_bits(expected,input.read()))) return;
            if (!t.assert_true(candidate.run(true))) return;
            if (!t.assert_true("reused chunk preserves stock bits",same_float_bits(expected,input.read()))) return;
            if (automatic) {
                const ggml_kv_stream_attention_request request{f.policy.shape,uint32_t(input.query_heads),queries,input.padded};
                size_t bytes = 0;
                if (!t.assert_true(ggml_kv_stream_workspace_query(ops,backend.get(),request,
                        ggml_kv_stream_workspace_kind::prefill,bytes))) return;
                block_workspace workspace(f,bytes,131);
                auto * parent = ggml_backend_memory_arena_parent(workspace.arena.get());
                const size_t parent_bytes = ggml_backend_buffer_get_size(parent);
                ggml_backend_memory_region region;
                if (!t.assert_true(ggml_backend_memory_lease_get_region(workspace.lease.get(),&region))) return;
                const size_t canary_bytes = std::min(size_t(128),std::min(region.offset,parent_bytes-region.offset-region.size));
                if (!t.assert_true(canary_bytes > 0)) return;
                const std::vector<uint8_t> canary(canary_bytes,0xa5);
                const auto check_canary = [&](bool write) {
                    ggml_tensor marker = {};
                    marker.type = GGML_TYPE_I8;
                    marker.ne[0] = int64_t(canary_bytes);
                    marker.ne[1] = marker.ne[2] = marker.ne[3] = 1;
                    marker.nb[0] = 1;
                    marker.nb[1] = marker.nb[2] = marker.nb[3] = canary_bytes;
                    marker.buffer = parent;
                    for (size_t offset : {size_t(0),parent_bytes-canary_bytes}) {
                        marker.data = static_cast<char *>(ggml_backend_buffer_get_base(parent))+offset;
                        if (write) ggml_backend_tensor_set(&marker,canary.data(),0,canary_bytes);
                        else {
                            std::vector<uint8_t> actual(canary_bytes);
                            ggml_backend_tensor_get(&marker,actual.data(),0,canary_bytes);
                            if (actual != canary) return false;
                        }
                    }
                    return true;
                };
                if (!t.assert_true(check_canary(true))) return;
                if (!t.assert_true(candidate.run(false,ggml_backend_memory_lease_buffer(workspace.lease.get())))) return;
                if (!t.assert_true("automatic prepared prefill preserves stock bits",same_float_bits(expected,input.read()))) return;
                if (!t.assert_true("prepared prefill preserves workspace canaries",check_canary(false))) return;
                if (!t.assert_true(candidate.run(false,ggml_backend_memory_lease_buffer(workspace.lease.get()),true))) return;
                if (!t.assert_true("compact prepared prefill preserves stock bits",same_float_bits(expected,input.read()))) return;
                if (!t.assert_true("compact prefill preserves workspace canaries",check_canary(false))) return;
                if (!t.assert_true(candidate.run(false,ggml_backend_memory_lease_buffer(workspace.lease.get()),false,true))) return;
                if (!t.assert_true("resident prefix preparation preserves stock bits",same_float_bits(expected,input.read()))) return;
                if (!t.assert_true(candidate.run(false,ggml_backend_memory_lease_buffer(workspace.lease.get()),true,true))) return;
                if (!t.assert_true("compact resident prefix preserves stock bits",same_float_bits(expected,input.read()))) return;
                if (!t.assert_true("resident prefix preserves workspace canaries",check_canary(false))) return;
                block_workspace short_workspace(f,candidate.plans[0].bytes-1,137);
                t.assert_true("short native state remains rejected",!candidate.run(false,
                    ggml_backend_memory_lease_buffer(short_workspace.lease.get())));
                if (!t.assert_true("rejected short state preserves public output",same_float_bits(expected,input.read()))) return;
            }
            if (benchmark) {
                std::vector<double> encoded,prepared;
                for (int trial = 0; trial < 5; ++trial) {
                    const auto measure = [&](bool converted) {
                        return average_us([&] { GGML_ASSERT(candidate.run(converted)); },2,8);
                    };
                    if (trial%2) { prepared.push_back(measure(true)); encoded.push_back(measure(false)); }
                    else { encoded.push_back(measure(false)); prepared.push_back(measure(true)); }
                }
                std::sort(encoded.begin(),encoded.end()); std::sort(prepared.begin(),prepared.end());
                std::printf("prepared-prefill-bench,width=%u,k=%s,v=%s,active=%zu,queries=256,chunk=2048,stage_bytes=%zu,encoded_us=%.3f,prepared_us=%.3f,speedup=%.5f\n",
                    width,ggml_type_name(key),ggml_type_name(value),active,candidate.decoded_layout.bytes,
                    encoded[2],prepared[2],encoded[2]/prepared[2]);
                if (!t.assert_true(same_float_bits(expected,input.read()))) return;
            }
        }
    }
}

static void check_d512_pipeline(testing & t, bool hazards) {
    ggml_backend_load_all();
    auto * dev=ggml_backend_dev_by_name("CUDA0");
    if (!t.assert_true(dev != nullptr)) return;
    ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
    auto get=reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(dev),"ggml_backend_kv_stream_partial_ops"));
    const auto * ops=get ? get() : nullptr;
    if (!t.assert_true(ops && ops->resume_spans)) return;
    const std::vector<std::pair<ggml_type,ggml_type>> pairs=hazards ?
        std::vector<std::pair<ggml_type,ggml_type>>{{GGML_TYPE_F16,GGML_TYPE_F16}} :
        std::vector<std::pair<ggml_type,ggml_type>>{{GGML_TYPE_F32,GGML_TYPE_BF16},
            {GGML_TYPE_F16,GGML_TYPE_F16},{GGML_TYPE_Q8_0,GGML_TYPE_Q4_0},
            {GGML_TYPE_Q4_1,GGML_TYPE_Q5_0},{GGML_TYPE_Q5_1,GGML_TYPE_Q8_0},{GGML_TYPE_BF16,GGML_TYPE_F32}};
    for (uint32_t heads : hazards ? std::vector<uint32_t>{32} : std::vector<uint32_t>{24,32})
        for (size_t active : hazards ? std::vector<size_t>{1057} : std::vector<size_t>{257,4097})
        for (uint32_t queries : hazards ? std::vector<uint32_t>{129} : std::vector<uint32_t>{5,8,64,129,256})
        for (const auto & pair : pairs) {
        fixture f(backend.get(),true,pair.first,pair.second,(active+255)/256*256,false,2,4,0,512,512);
        if (!t.assert_true(f.attach())) return;
        block_inputs input(f,active,queries,false,heads);
        stock_attention_case stock(f,input);
        stock.run();
        std::vector<float> expected(ggml_nelements(stock.output));
        ggml_backend_tensor_get(stock.output,expected.data(),0,ggml_nbytes(stock.output));
        prepared_prefill_case candidate(f,input,stock,ops,1024);
        if (!t.assert_true(candidate.initialize() && candidate.same_plan())) return;
        for (bool compact : {false,true}) for (bool prepared : {false,true}) {
            if (!t.assert_true(candidate.run(prepared,nullptr,compact))) return;
            if (!t.assert_true("native operand and accumulation bits",same_float_bits(expected,input.read()))) return;
        }
        if (pair.first == GGML_TYPE_F16 && pair.second == GGML_TYPE_F16) {
            const std::vector<size_t> cuts=input.padded > 1025 ?
                std::vector<size_t>{0,255,1025,input.padded} : std::vector<size_t>{0,255,257,input.padded};
            auto storage=span_storage::create(f,0,input.padded,queries,cuts,true);
            ggml_kv_stream_span_plan_view original;
            if (!t.assert_true(storage && ggml_kv_stream_span_plan_get_view(storage->plan,original))) return;
            auto * scratch=ggml_backend_memory_lease_buffer(candidate.scratch->lease.get());
            const std::vector<float> sentinel(expected.size(),-77);
            for (bool compact : {false,true}) {
                auto op=*stock.output;
                op.buffer=input.output->buffer; op.data=input.output->data;
                ggml_backend_execution_set_stock_conversion(&op,true);
                if (compact) {
                    op.src[3]=nullptr;
                    if (!t.assert_true(candidate.bounds && ggml_backend_execution_set_mask_bounds(&op,candidate.bounds))) return;
                }
                ggml_backend_tensor_set(input.output,sentinel.data(),0,ggml_nbytes(input.output));
                auto bad=original;
                bad.count=4;
                t.assert_true(!ops->resume_spans(f.backend,&op,scratch,candidate.plans[0],bad,0,input.padded));
                auto wrong=candidate.plans[0];
                ++wrong.state_bytes;
                t.assert_true(!ops->resume_spans(f.backend,&op,scratch,wrong,original,0,input.padded));
                block_workspace short_workspace(f,candidate.plans[0].bytes-1,139);
                t.assert_true(!ops->resume_spans(f.backend,&op,ggml_backend_memory_lease_buffer(short_workspace.lease.get()),
                    candidate.plans[0],original,0,input.padded));
                t.assert_true("rejected waves preserve public output",same_float_bits(sentinel,input.read()));
                for (size_t first=0;first<input.padded;first+=2048) {
                    const size_t end=std::min(input.padded,first+2048);
                    ggml_kv_stream_span clipped[3];
                    size_t count=0;
                    for (size_t i=0;i<original.count;++i) {
                        const auto & source=original.spans[i];
                        const size_t begin=std::max(first,source.token_begin);
                        const size_t stop=std::min(end,source.token_begin+source.tokens);
                        if (begin >= stop) continue;
                        auto & span=clipped[count++];
                        span=source;
                        span.token_begin=begin; span.tokens=stop-begin;
                        span.k_offset+=(begin-source.token_begin)*f.host->layout().k_token_bytes;
                        span.v_offset+=(begin-source.token_begin)*f.host->layout().v_token_bytes;
                    }
                    const ggml_kv_stream_span_plan_view wave{f.policy.shape,clipped,count,input.padded,queries};
                    if (!t.assert_true(ops->resume_spans(f.backend,&op,scratch,candidate.plans[0],wave,first,end))) return;
                    ggml_backend_synchronize(f.backend);
                    if (end < input.padded) {
                        if (!t.assert_true("partial wave does not publish output",same_float_bits(sentinel,input.read()))) return;
                        auto invalid=wave;
                        invalid.count=4;
                        t.assert_true(!ops->resume_spans(f.backend,&op,scratch,candidate.plans[0],invalid,first,end));
                        if (!t.assert_true("rejected continuation preserves output",same_float_bits(sentinel,input.read()))) return;
                    }
                }
                if (!t.assert_true("wrapped waves and retry preserve native bits",same_float_bits(expected,input.read()))) return;
            }
        }
        t.out << "d512-pipeline heads=" << heads << " active=" << active << " queries=" << queries
            << " k=" << ggml_type_name(pair.first) << " v=" << ggml_type_name(pair.second) << '\n';
    }
}

// Transport/ownership only: shared native multi-sequence launches are qualified separately.
static void check_slot_page_partitions(testing & t) {
    ggml_backend_load_all();
    auto * device=ggml_backend_dev_by_name("CUDA0");
    if (!t.assert_true(device!=nullptr)) return;
    ggml_backend_ptr backend(ggml_backend_dev_init(device,nullptr));
    auto * reg=ggml_backend_dev_backend_reg(device);
    auto get=reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_kv_stream_partial_ops"));
    const auto * ops=get ? get() : nullptr;
    const auto retire=reinterpret_cast<void (*)(ggml_backend_t)>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_cuda_graph_release_all"));
    if (!t.assert_true(backend && ops && ops->capabilities && ops->spans && ops->spans_workspace && retire)) return;
    auto * type=llama_kv_stream_device_buffer_type(device);
    const ggml_type types[]={GGML_TYPE_F32,GGML_TYPE_F16,GGML_TYPE_BF16,GGML_TYPE_Q8_0,
        GGML_TYPE_Q5_0,GGML_TYPE_Q5_1,GGML_TYPE_Q4_0,GGML_TYPE_Q4_1};
    size_t profiles=0;
    const auto read_bytes=[](ggml_backend_buffer_t buffer,size_t offset=0,size_t count=0) {
        const size_t bytes=ggml_backend_buffer_get_size(buffer);
        ggml_tensor raw{};raw.type=GGML_TYPE_I8;raw.ne[0]=int64_t(bytes);raw.ne[1]=raw.ne[2]=raw.ne[3]=1;
        raw.nb[0]=1;raw.nb[1]=raw.nb[2]=raw.nb[3]=bytes;raw.buffer=buffer;raw.data=ggml_backend_buffer_get_base(buffer);
        std::vector<uint8_t> result(count ? count : bytes);
        ggml_backend_tensor_get(&raw,result.data(),offset,result.size());return result;
    };
    for (uint32_t dimension : {256u,512u}) for (auto key : types) for (auto value : types) {
        const ggml_kv_stream_shape shape{key,value,dimension,dimension,4,256,128};
        ggml_kv_stream_layout page;
        if (!t.assert_true(ggml_kv_stream_layout_make(shape,256,page).status==ggml_kv_stream_status::success)) return;
        llama_kv_stream_policy_config native;
        native.shape=shape;native.capabilities=ops->capabilities(backend.get(),key,value);native.layers=2;
        if (!t.assert_true(native.capabilities.direct_pair)) return;
        size_t minimum=0;
        if (!t.assert_true(llama_kv_stream_policy_minimum_pool_bytes(native,minimum).status==llama_kv_stream_policy_status::success && minimum%page.bytes==0)) return;
        const auto min_pages=uint32_t(minimum/page.bytes);
        llama_kv_stream_selection selected;
        if (!t.assert_true(llama_kv_stream_select({4096,2},{{0,2049},{1,769}},selected)==llama_kv_stream_selection_status::success)) return;
        llama_kv_stream_page_partition partition;
        if (!t.assert_true(llama_kv_stream_partition_pages(selected,17,{{0,min_pages},{1,min_pages}},partition).status==
                llama_kv_stream_page_partition_status::success)) return;
        const size_t alignment=ggml_backend_buft_get_alignment(type),guard=(alignment+127)/128*128;
        arena_ptr parent(ggml_backend_memory_arena_new(type,17*page.bytes+2*guard),ggml_backend_memory_arena_free);
        if (!t.assert_true(parent && ggml_backend_memory_arena_begin(parent.get(),0))) return;
        for (const auto & slot : partition.slots) if (!t.assert_true(ggml_backend_memory_arena_reserve_at(parent.get(),101+slot.slot,
                guard+size_t(slot.first_page)*page.bytes,size_t(slot.pages)*page.bytes,alignment,GGML_BACKEND_MEMORY_REGION_PERSISTENT,nullptr))) return;
        if (!t.assert_true(ggml_backend_memory_arena_commit(parent.get()))) return;
        lease_ptr left(ggml_backend_memory_arena_acquire(parent.get(),101),ggml_backend_memory_lease_free),
            right(ggml_backend_memory_arena_acquire(parent.get(),102),ggml_backend_memory_lease_free);
        if (!t.assert_true(left && right)) return;
        ggml_backend_buffer_clear(ggml_backend_memory_arena_parent(parent.get()),0xa5);
        fixture a(backend.get(),true,key,value,4096,false,2,4,partition.slots[0].pages,dimension,dimension,left.get(),37,0);
        fixture b(backend.get(),true,key,value,4096,false,2,4,partition.slots[1].pages,dimension,dimension,right.get(),38,101);
        if (!t.assert_true(!a.arena && !b.arena && a.host->cache_id()!=b.host->cache_id() && a.attach() && b.attach())) return;
        auto pin_a=a.binding->acquire(),pin_b=b.binding->acquire();
        if (!t.assert_true(pin_a && pin_b)) return;
        for (uint32_t queries : {1u,4u,8u}) {
            block_inputs input_a(a,2049,queries,false,dimension==256 ? 24 : 32),input_b(b,769,queries,false,dimension==256 ? 24 : 32);
            stock_attention_case stock_a(a,input_a),stock_b(b,input_b);
            ggml_context_ptr masks(ggml_init({4096+2*ggml_tensor_overhead(),nullptr,true}));
            auto * bounds_a=ggml_new_tensor_2d(masks.get(),GGML_TYPE_I32,4,queries);
            auto * bounds_b=ggml_new_tensor_2d(masks.get(),GGML_TYPE_I32,4,queries);
            ggml_backend_buffer_ptr mask_buffer(ggml_backend_alloc_ctx_tensors(masks.get(),backend.get()));
            if (!t.assert_true(bool(mask_buffer))) return;
            for (auto side : {false,true}) {
                const auto & input=side ? input_b : input_a;
                std::vector<int32_t> bounds(4*queries);
                for (uint32_t q=0;q<queries;++q) {
                    bounds[4*q]=0;bounds[4*q+1]=int32_t(input.active-queries+q+1);
                    bounds[4*q+2]=int32_t(input.padded);bounds[4*q+3]=1;
                }
                ggml_backend_tensor_set(side ? bounds_b : bounds_a,bounds.data(),0,bounds.size()*sizeof(int32_t));
            }
            struct capture_retirement {
                ggml_backend_t backend;void (*release)(ggml_backend_t);
                ~capture_retirement() { ggml_backend_synchronize(backend);release(backend); }
            } retirement{backend.get(),retire};
            struct geometry { std::vector<size_t> a,b;bool wrap,prefix; };
            const std::vector<geometry> layouts={
                {{0,2304},{0,1024},false,true},
                {{0,256,2304},{0,256,1024},false,true},
                {{0,256,1024,2304},{0,256,512,1024},true,true},
                {{0,1024,2304},{0,512,1024},true,false}};
            for (bool compact : {false,true}) for (const auto & layout : layouts) {
                auto peer=read_bytes(ggml_backend_memory_lease_buffer(right.get()));
                auto spans_a=span_storage::create(a,0,input_a.padded,queries,layout.a,layout.wrap,left.get(),layout.prefix);
                if (!t.assert_true(spans_a!=nullptr)) return;
                t.assert_true("first-slot uploads stay inside its pages",peer==read_bytes(ggml_backend_memory_lease_buffer(right.get())));
                auto previous=read_bytes(ggml_backend_memory_lease_buffer(left.get()));
                auto spans_b=span_storage::create(b,0,input_b.padded,queries,layout.b,layout.wrap,right.get(),layout.prefix);
                if (!t.assert_true(spans_b!=nullptr)) return;
                t.assert_true("second-slot uploads stay inside its pages",previous==read_bytes(ggml_backend_memory_lease_buffer(left.get())));
                span_attention_case streamed_a(a,ops,stock_a,*spans_a,input_a,compact ? bounds_a : nullptr),
                    streamed_b(b,ops,stock_b,*spans_b,input_b,compact ? bounds_b : nullptr);
                for (size_t replay=0;replay<3;++replay) {
                    stock_a.run();stock_b.run();
                    std::vector<float> expected_a(ggml_nelements(stock_a.output)),expected_b(ggml_nelements(stock_b.output));
                    ggml_backend_tensor_get(stock_a.output,expected_a.data(),0,ggml_nbytes(stock_a.output));
                    ggml_backend_tensor_get(stock_b.output,expected_b.data(),0,ggml_nbytes(stock_b.output));
                    streamed_a.run();const auto before=input_a.read();
                    t.assert_true("first slot preserves native bits",same_float_bits(expected_a,before));
                    streamed_b.run();
                    t.assert_true("second slot preserves its own native bits",same_float_bits(expected_b,input_b.read()));
                    t.assert_true("peer execution cannot rewrite a finished output",same_float_bits(before,input_a.read()));
                }
                ++profiles;
            }
        }
        auto first_guard=read_bytes(ggml_backend_memory_arena_parent(parent.get()),0,guard);
        auto last_guard=read_bytes(ggml_backend_memory_arena_parent(parent.get()),guard+17*page.bytes,guard);
        t.assert_true(std::all_of(first_guard.begin(),first_guard.end(),[](uint8_t v) {return v==0xa5;}));
        t.assert_true(std::all_of(last_guard.begin(),last_guard.end(),[](uint8_t v) {return v==0xa5;}));
        t.out << "slot pages D=" << dimension << " K=" << ggml_type_name(key) << " V=" << ggml_type_name(value) << " quota=" << partition.slots[0].pages << '+' << partition.slots[1].pages
              << " minimum=" << min_pages << " shared-parent-bytes=" << 17*page.bytes+2*guard << std::endl;
    }
    t.assert_equal(size_t(3072),profiles);
}

// Return an error only after a real successful operation on one owner's queue.
struct slot_queue_fault {
    inline static slot_queue_fault * active=nullptr;
    ggml_backend_reg_t reg;
    decltype(ggml_backend_reg_i::get_proc_address) original;
    const ggml_kv_stream_copy_ops * base;
    ggml_kv_stream_copy_ops replacement;
    std::map<void *,uintptr_t> owners;
    std::map<uintptr_t,size_t> drains;
    std::map<void *,uint64_t> submitted,collected;
    std::map<void *,ggml_kv_stream_copy_feedback> reports;
    uintptr_t watched=0;
    int kind=-1;
    size_t seen=0;
    bool fired=false;
    bool fail(void * queue,int operation) {
        if (kind!=operation || owners.at(queue)!=watched) return false;
        if (++seen!=(kind==0 ? 3u : 1u)) return false;
        fired=true;return true;
    }
    explicit slot_queue_fault(ggml_backend_t backend) : reg(ggml_backend_dev_backend_reg(ggml_backend_get_device(backend))),
        original(reg->iface.get_proc_address) {
        GGML_ASSERT(!active);active=this;
        const auto get=reinterpret_cast<ggml_kv_stream_copy_ops_get>(original(reg,"ggml_backend_kv_stream_copy_ops"));
        base=get ? get() : nullptr;
        GGML_ASSERT(base && base->version>=10 && base->create && base->enqueue_span &&
            base->enqueue_span_with_feedback && base->acquire_span && base->release_span);
        replacement=*base;
        replacement.create=[](ggml_backend_t b,ggml_backend_buffer_t device,ggml_backend_buffer_t host,
                const ggml_kv_stream_shape & shape,size_t slots) {
            auto * queue=active->base->create(b,device,host,shape,slots);
            if (queue) { active->owners[queue]=uintptr_t(ggml_backend_buffer_get_base(device)); }
            return queue;
        };
        if (base->create_prepared) replacement.create_prepared=[](ggml_backend_t b,void * prepared,
                ggml_backend_buffer_t device,ggml_backend_buffer_t host,const ggml_kv_stream_shape & shape,size_t slots) {
            auto * queue=active->base->create_prepared(b,prepared,device,host,shape,slots);
            if (queue) { active->owners[queue]=uintptr_t(ggml_backend_buffer_get_base(device)); }
            return queue;
        };
        replacement.free=[](void * queue) {
            active->owners.erase(queue);active->submitted.erase(queue);active->collected.erase(queue);active->reports.erase(queue);
            active->base->free(queue);
        };
        replacement.begin_with_feedback=[](void * queue,bool eligible) {
            const bool ok=active->base->begin_with_feedback(queue,eligible);
            if (ok) {const auto id=active->base->feedback_id(queue);if (id) active->submitted[queue]=id;}
            return ok;
        };
        replacement.poll_feedback=[](void * queue,ggml_kv_stream_copy_snapshot * snapshot) {
            const bool ok=active->base->poll_feedback(queue,snapshot);
            if (ok) {active->collected[queue]=snapshot->id;active->reports[queue]=snapshot->value;}
            return ok;
        };
        replacement.drain=[](void * queue) {++active->drains[active->owners.at(queue)];active->base->drain(queue);};
        replacement.enqueue_span=[](void * q,size_t slot,const void * k,const void * v,size_t live,size_t padded) {
            const bool ok=active->base->enqueue_span(q,slot,k,v,live,padded);return ok && !active->fail(q,0);
        };
        replacement.enqueue_span_with_feedback=[](void * q,size_t slot,const void * k,const void * v,size_t live,size_t padded,bool eligible) {
            const bool ok=active->base->enqueue_span_with_feedback(q,slot,k,v,live,padded,eligible);return ok && !active->fail(q,0);
        };
        replacement.acquire_span=[](void * q,size_t slot,size_t count) {
            const bool ok=active->base->acquire_span(q,slot,count);return ok && !active->fail(q,1);
        };
        replacement.release_span=[](void * q,size_t slot,size_t count) {
            const bool ok=active->base->release_span(q,slot,count);return ok && !active->fail(q,2);
        };
        reg->iface.get_proc_address=[](ggml_backend_reg_t r,const char * name) -> void * {
            if (!std::strcmp(name,"ggml_backend_kv_stream_copy_ops"))
                return reinterpret_cast<void *>(+[]() -> const ggml_kv_stream_copy_ops * {return &active->replacement;});
            return active->original(r,name);
        };
    }
    ~slot_queue_fault() {reg->iface.get_proc_address=original;active=nullptr;}
};

static bool slot_feedback_equal(const llama_kv_stream_feedback & a,const llama_kv_stream_feedback & b) {
    return a.available==b.available && a.epoch==b.epoch && a.samples==b.samples && a.misses==b.misses &&
        a.copy_busy_ratio==b.copy_busy_ratio && a.peak_slots==b.peak_slots && a.layer_samples==b.layer_samples && a.layer_misses==b.layer_misses;
}

static void check_slot_page_queues(testing & t) {
    ggml_backend_load_all();auto * device=ggml_backend_dev_by_name("CUDA0");
    if (!t.assert_true(device!=nullptr)) return;
    ggml_backend_ptr backend(ggml_backend_dev_init(device,nullptr));
    auto * reg=ggml_backend_dev_backend_reg(device);
    const auto get=reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_kv_stream_partial_ops"));
    const auto * ops=get ? get() : nullptr;
    const auto retire=reinterpret_cast<void (*)(ggml_backend_t)>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_cuda_graph_release_all"));
    if (!t.assert_true(ops && retire)) return;
    const std::vector<std::pair<ggml_type,ggml_type>> pairs={{GGML_TYPE_Q8_0,GGML_TYPE_Q4_0},
        {GGML_TYPE_Q8_0,GGML_TYPE_Q8_0},{GGML_TYPE_Q5_1,GGML_TYPE_Q5_1},{GGML_TYPE_Q5_0,GGML_TYPE_Q4_1},
        {GGML_TYPE_F16,GGML_TYPE_BF16},{GGML_TYPE_F16,GGML_TYPE_F16},{GGML_TYPE_F32,GGML_TYPE_F32}};
    size_t profiles=0;
    for (uint32_t dimension : {256u,512u}) for (auto pair : pairs) for (uint32_t pages : {6u,17u,96u}) {
        slot_queue_fault fault(backend.get());
        const ggml_kv_stream_shape shape{pair.first,pair.second,dimension,dimension,4,256,128};
        const uint32_t heads=dimension==256 ? 24 : 32;
        const ggml_kv_stream_attention_request requests[]={{shape,heads,1,2304},{shape,heads,4,1024}};
        size_t scratch=0;
        for (const auto & request : requests) {size_t bytes=0;
            if (!t.assert_true(ggml_kv_stream_workspace_query(ops,backend.get(),request,ggml_kv_stream_workspace_kind::decode,bytes))) return;
            scratch=std::max(scratch,bytes);
        }
        ggml_kv_stream_layout page;
        if (!t.assert_true(ggml_kv_stream_layout_make(shape,256,page).status==ggml_kv_stream_status::success)) return;
        llama_kv_stream_policy_config config;config.shape=shape;config.capabilities=ops->capabilities(backend.get(),pair.first,pair.second);config.layers=2;
        size_t minimum=0;
        if (!t.assert_true(llama_kv_stream_policy_minimum_pool_bytes(config,minimum).status==llama_kv_stream_policy_status::success && minimum%page.bytes==0)) return;
        const auto minimum_pages=uint32_t(minimum/page.bytes);
        llama_kv_stream_selection selected;llama_kv_stream_page_partition partition;
        if (!t.assert_true(llama_kv_stream_select({4096,2},{{0,2049},{1,769}},selected)==llama_kv_stream_selection_status::success &&
                llama_kv_stream_partition_pages(selected,pages,{{0,minimum_pages},{1,minimum_pages}},partition).status==llama_kv_stream_page_partition_status::success)) return;
        auto * type=llama_kv_stream_device_buffer_type(device);const size_t alignment=ggml_backend_buft_get_alignment(type);
        const size_t guard=(alignment+127)/128*128,pool_end=guard+size_t(pages)*page.bytes;
        const size_t scratch_offset=(pool_end+guard+alignment-1)/alignment*alignment;
        arena_ptr parent(ggml_backend_memory_arena_new(type,scratch_offset+scratch+guard),ggml_backend_memory_arena_free);
        if (!t.assert_true(parent && ggml_backend_memory_arena_begin(parent.get(),0))) return;
        for (const auto & slot : partition.slots) if (!t.assert_true(ggml_backend_memory_arena_reserve_at(parent.get(),101+slot.slot,
                guard+size_t(slot.first_page)*page.bytes,size_t(slot.pages)*page.bytes,alignment,GGML_BACKEND_MEMORY_REGION_PERSISTENT,nullptr))) return;
        if (!t.assert_true(ggml_backend_memory_arena_reserve_at(parent.get(),201,scratch_offset,scratch,alignment,0,nullptr) && ggml_backend_memory_arena_commit(parent.get()))) return;
        lease_ptr a_lease(ggml_backend_memory_arena_acquire(parent.get(),101),ggml_backend_memory_lease_free),
            b_lease(ggml_backend_memory_arena_acquire(parent.get(),102),ggml_backend_memory_lease_free),
            workspace(ggml_backend_memory_arena_acquire(parent.get(),201),ggml_backend_memory_lease_free);
        if (!t.assert_true(a_lease && b_lease && workspace)) return;
        ggml_backend_buffer_clear(ggml_backend_memory_arena_parent(parent.get()),0xa5);
        fixture a(backend.get(),true,pair.first,pair.second,4096,false,2,4,partition.slots[0].pages,dimension,dimension,a_lease.get(),37,0),
            b(backend.get(),true,pair.first,pair.second,4096,false,2,4,partition.slots[1].pages,dimension,dimension,b_lease.get(),38,101);
        llama_kv_stream_policy_state placement_a,placement_b;
        if (!t.assert_true(llama_kv_stream_policy_initialize(a.policy,placement_a).status==llama_kv_stream_policy_status::success &&
                llama_kv_stream_policy_initialize(b.policy,placement_b).status==llama_kv_stream_policy_status::success)) return;
        if (pages==6) {placement_a.resident_pages_per_layer=placement_b.resident_pages_per_layer=0;
            placement_a.ring_slots=placement_a.budget.pages;placement_b.ring_slots=placement_b.budget.pages;}
        if (!t.assert_true(a.attach(&placement_a) && b.attach(&placement_b) &&
                a.resident->configure_native_graph_attention(true) && b.resident->configure_native_graph_attention(true) &&
                a.resident->configure_resumed_decode(true) && b.resident->configure_resumed_decode(true) &&
                a.resident->configure_feedback(true) && b.resident->configure_feedback(true))) return;
        auto a_pin=a.binding->acquire(),b_pin=b.binding->acquire();
        if (!t.assert_true(a_pin && b_pin)) return;
        const auto gen_a=a.content->generation(),gen_b=b.content->generation();
        std::vector<uint8_t> host_a(a.host->bytes()),host_b(b.host->bytes());
        std::memcpy(host_a.data(),ggml_backend_buffer_get_base(a.host->buffer()),host_a.size());
        std::memcpy(host_b.data(),ggml_backend_buffer_get_base(b.host->buffer()),host_b.size());
        block_inputs input_a(a,2049,1,false,heads),input_b(b,769,4,false,heads);
        struct layer_control : stock_attention_case {
            layer_control(fixture & f,block_inputs & input,uint32_t layer) : stock_attention_case(f,input) {
                if (layer) {llama_kv_stream_host_layer source;GGML_ASSERT(f.host->layer(layer,source));
                    ggml_backend_tensor_set(k_storage,source.k,0,ggml_nbytes(k_storage));
                    ggml_backend_tensor_set(v_storage,source.v,0,ggml_nbytes(v_storage));}
            }
        };
        std::vector<std::unique_ptr<layer_control>> native_a,native_b;
        for (uint32_t layer : {0u,1u}) {native_a.emplace_back(new layer_control(a,input_a,layer));native_b.emplace_back(new layer_control(b,input_b,layer));}
        struct capture_retirement {ggml_backend_t backend;void (*release)(ggml_backend_t);
            ~capture_retirement() {ggml_backend_synchronize(backend);release(backend);}} retirement{backend.get(),retire};
        const auto run_layer=[&](fixture & f,block_inputs & input,layer_control & reference,uint32_t layer) {
            reference.run();std::vector<float> expected(ggml_nelements(reference.output));
            ggml_backend_tensor_get(reference.output,expected.data(),0,ggml_nbytes(reference.output));
            if (!t.assert_true(f.resident->compute_streamed(layer,input.q,input.mask,input.output,input.active,1.0f/16,workspace.get(),true,1))) return false;
            ggml_backend_synchronize(backend.get());
            return t.assert_true("queued slot attention retains native bits",same_float_bits(expected,input.read()));
        };
        const auto begin=[&](fixture & f,const block_inputs & input) {
            return f.resident->begin_sequence({0,1},input.active,1,SIZE_MAX,{uint32_t(input.queries),true,false});
        };
        for (size_t replay=0;replay<3;++replay) {
            if (!t.assert_true(begin(a,input_a) && begin(b,input_b))) return;
            for (uint32_t layer : {0u,1u}) {
                if (!run_layer(a,input_a,*native_a[layer],layer)) return;
                const auto previous=input_a.read();
                if (!run_layer(b,input_b,*native_b[layer],layer)) return;
                t.assert_true("peer queues cannot rewrite the previous output",same_float_bits(previous,input_a.read()));
            }
            t.assert_true(!a.resident->sequence_active() && !b.resident->sequence_active());
            t.assert_true(a.resident->sequence_stats().peak_pages<=placement_a.ring_slots && b.resident->sequence_stats().peak_pages<=placement_b.ring_slots);
        }
        a.resident->cancel_sequence();b.resident->cancel_sequence();
        auto frozen=a.resident->feedback();const auto a_result=input_a.read();
        if (pages!=96) {
            const auto base=uintptr_t(ggml_backend_buffer_get_base(ggml_backend_memory_lease_buffer(a_lease.get())));
            void * queue=nullptr;
            for (const auto & item : fault.owners) if (item.second==base) queue=item.first;
            const auto ticket=queue ? fault.submitted[queue] : 0;
            const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
            while (queue && ticket && fault.collected[queue]<ticket && std::chrono::steady_clock::now()<deadline) {
                frozen=a.resident->feedback();std::this_thread::yield();
            }
            frozen=a.resident->feedback();
            if (!t.assert_true("the TG1 owner has completed real feedback before checking isolation",
                    ticket && fault.collected[queue]>=ticket && frozen.available && frozen.samples>0)) {
                const auto report=fault.reports[queue];
                t.out << "feedback ticket=" << ticket << " collected=" << fault.collected[queue]
                      << " available=" << frozen.available << " samples=" << frozen.samples << " layers=" << frozen.layer_samples
                      << " raw-samples=" << report.samples << " raw-layers=" << report.layer_samples
                      << " bytes=" << report.bytes << " timed=" << report.timed_bytes << " ms=" << report.elapsed_ms << std::endl;
                return;
            }
        }
        if (!t.assert_true(begin(b,input_b))) return;
        for (uint32_t layer : {0u,1u}) if (!run_layer(b,input_b,*native_b[layer],layer)) return;
        b.resident->cancel_sequence();
        t.assert_true("peer geometry cannot reset or train another slot",slot_feedback_equal(frozen,a.resident->feedback()));
        t.assert_true(same_float_bits(a_result,input_a.read()));
        if (dimension==512 && pair.first==GGML_TYPE_Q8_0 && pair.second==GGML_TYPE_Q4_0 && pages==17) {
            const auto b_base=uintptr_t(ggml_backend_buffer_get_base(ggml_backend_memory_lease_buffer(b_lease.get())));
            if (!t.assert_true(begin(b,input_b) && begin(a,input_a))) return;
            const auto drains=fault.drains[b_base];a.resident->cancel_sequence();
            t.assert_true(b.resident->sequence_active() && fault.drains[b_base]==drains);
            for (uint32_t layer : {0u,1u}) if (!run_layer(b,input_b,*native_b[layer],layer)) return;
            for (int kind : {0,1,2}) {
                if (!t.assert_true(begin(b,input_b))) return;
                fault.watched=uintptr_t(ggml_backend_buffer_get_base(ggml_backend_memory_lease_buffer(a_lease.get())));
                fault.kind=kind;fault.seen=0;fault.fired=false;
                bool ok=begin(a,input_a);
                if (ok) ok=a.resident->compute_streamed(0,input_a.q,input_a.mask,input_a.output,input_a.active,1.0f/16,workspace.get(),true,1);
                if (!t.assert_true(!ok && fault.fired && fault.seen>0)) return;
                fault.kind=-1;const auto before=fault.drains[b_base];a.resident->cancel_sequence();ggml_backend_synchronize(backend.get());
                t.assert_true(b.resident->sequence_active() && fault.drains[b_base]==before);
                for (uint32_t layer : {0u,1u}) if (!run_layer(b,input_b,*native_b[layer],layer)) return;
                if (!t.assert_true(begin(a,input_a))) return;
                for (uint32_t layer : {0u,1u}) if (!run_layer(a,input_a,*native_a[layer],layer)) return;
            }
        }
        t.assert_equal(gen_a,a.content->generation());t.assert_equal(gen_b,b.content->generation());
        t.assert_true(!std::memcmp(host_a.data(),ggml_backend_buffer_get_base(a.host->buffer()),host_a.size()) &&
            !std::memcmp(host_b.data(),ggml_backend_buffer_get_base(b.host->buffer()),host_b.size()));
        ggml_tensor raw{};auto * buffer=ggml_backend_memory_arena_parent(parent.get());
        raw.type=GGML_TYPE_I8;raw.ne[0]=int64_t(ggml_backend_buffer_get_size(buffer));raw.ne[1]=raw.ne[2]=raw.ne[3]=1;
        raw.nb[0]=1;raw.nb[1]=raw.nb[2]=raw.nb[3]=size_t(raw.ne[0]);raw.buffer=buffer;raw.data=ggml_backend_buffer_get_base(buffer);
        for (auto offset : {size_t(0),pool_end,scratch_offset+scratch}) {std::vector<uint8_t> bytes(guard);
            ggml_backend_tensor_get(&raw,bytes.data(),offset,bytes.size());
            t.assert_true(std::all_of(bytes.begin(),bytes.end(),[](uint8_t value) {return value==0xa5;}));}
        ++profiles;
        t.out << "slot queues D=" << dimension << " K=" << ggml_type_name(pair.first) << " V=" << ggml_type_name(pair.second)
              << " pages=" << pages << " split=" << partition.slots[0].pages << '+' << partition.slots[1].pages << std::endl;
    }
    t.assert_equal(size_t(42),profiles);
}

struct group_pipeline_probe {
    inline static group_pipeline_probe * active=nullptr;
    ggml_backend_reg_t reg;
    decltype(ggml_backend_reg_i::get_proc_address) original;
    const ggml_kv_stream_copy_ops * copy;
    const ggml_kv_stream_registered_copy_ops * registered;
    const ggml_kv_stream_partial_ops * partial;
    ggml_kv_stream_copy_ops copy_probe;
    ggml_kv_stream_registered_copy_ops registered_probe;
    ggml_kv_stream_partial_ops partial_probe;
    std::vector<char> events;
    std::vector<uintptr_t> before_attention_k;
    std::vector<uintptr_t> before_gather_k;
    struct upload {uintptr_t address;size_t slot,pages;};
    std::vector<upload> uploads;
    std::map<void *,size_t> held;
    size_t maximum_held=0,fail_seen=0,maximum_attention_span=0,maximum_stream_span=0,small_native_copies=0;
    int fail_kind=-1;
    bool fail_fired=false;
    bool force_deadlines=false;
    bool inject(int kind) {
        if (fail_kind!=kind || ++fail_seen!=(kind ? 1u : 3u)) return false;
        fail_fired=true;return true;
    }
    explicit group_pipeline_probe(ggml_backend_t backend) : reg(ggml_backend_dev_backend_reg(ggml_backend_get_device(backend))),
        original(reg->iface.get_proc_address) {
        GGML_ASSERT(!active);active=this;
        copy=reinterpret_cast<ggml_kv_stream_copy_ops_get>(original(reg,"ggml_backend_kv_stream_copy_ops"))();
        registered=reinterpret_cast<ggml_kv_stream_registered_copy_ops_get>(original(reg,"ggml_backend_kv_stream_registered_copy_ops"))();
        partial=reinterpret_cast<ggml_kv_stream_partial_ops_get>(original(reg,"ggml_backend_kv_stream_partial_ops"))();
        copy_probe=*copy;registered_probe=*registered;partial_probe=*partial;
        copy_probe.gather=[](ggml_backend_t backend,ggml_backend_buffer_t output,
                const ggml_kv_stream_copy_rectangle * rectangles,size_t count,bool initialize) {
#ifdef KV_STREAM_COPY_ENTRY_TRACE
            kv_stream_copy_capture_entries(true);
#endif
            const bool okay=active->copy->gather(backend,output,rectangles,count,initialize);
#ifdef KV_STREAM_COPY_ENTRY_TRACE
            active->small_native_copies+=kv_stream_copy_small_native_entries();
            kv_stream_copy_capture_entries(false);
#endif
            if (okay) active->events.push_back('G');
            return okay;
        };
        copy_probe.acquire_span=[](void * queue,size_t slot,size_t count) {
            const bool okay=active->copy->acquire_span(queue,slot,count);
            if (okay) {active->held[queue]+=count;active->maximum_held=std::max(active->maximum_held,active->held[queue]);}
            return okay && !active->inject(1);
        };
        copy_probe.release_span=[](void * queue,size_t slot,size_t count) {
            const bool okay=active->copy->release_span(queue,slot,count);
            if (okay) {GGML_ASSERT(active->held[queue]>=count);active->held[queue]-=count;}
            return okay && !active->inject(2);
        };
        copy_probe.drain=[](void * queue) {active->copy->drain(queue);active->held[queue]=0;};
        copy_probe.poll_feedback=[](void * queue,ggml_kv_stream_copy_snapshot * result) {
            const bool okay=active->copy->poll_feedback(queue,result);
            if (okay && active->force_deadlines && result->value.layer_samples) {
                result->value.layer_misses=result->value.layer_samples;result->value.copy_ms=0;
            }
            return okay;
        };
        copy_probe.probe_layer=[](void * queue,const ggml_kv_stream_copy_range * ranges,size_t count) {
            const bool okay=active->copy->probe_layer(queue,ranges,count);
            if (okay) active->events.push_back('P');
            return okay;
        };
        registered_probe.enqueue=[](void * queue,size_t slot,const ggml_kv_stream_registered_copy_fragment * k,size_t nk,
                const ggml_kv_stream_registered_copy_fragment * v,size_t nv,size_t live,size_t padded,bool feedback) {
            const bool okay=active->registered->enqueue(queue,slot,k,nk,v,nv,live,padded,feedback);
            if (okay) {
                if (nk) active->uploads.push_back({uintptr_t(ggml_backend_buffer_get_base(k[0].owner))+k[0].offset,slot,padded/256});
                if (nk && std::find(active->events.begin(),active->events.end(),'A')==active->events.end())
                    active->before_attention_k.push_back(uintptr_t(ggml_backend_buffer_get_base(k[0].owner))+k[0].offset);
                if (nk && std::find(active->events.begin(),active->events.end(),'G')==active->events.end())
                    active->before_gather_k.push_back(uintptr_t(ggml_backend_buffer_get_base(k[0].owner))+k[0].offset);
                active->events.push_back('C');
            }
            return okay && !active->inject(0);
        };
        partial_probe.resume=[](ggml_backend_t backend,const ggml_tensor * op,ggml_backend_buffer_t workspace,
                const ggml_kv_stream_resume_plan & plan,size_t tokens,size_t first,bool last) {
            active->maximum_attention_span=std::max(active->maximum_attention_span,size_t(op->src[1]->ne[1]));
            if (std::any_of(active->held.begin(),active->held.end(),[](const auto & item) {return item.second!=0;}))
                active->maximum_stream_span=std::max(active->maximum_stream_span,size_t(op->src[1]->ne[1]));
            active->events.push_back('A');return active->partial->resume(backend,op,workspace,plan,tokens,first,last);
        };
        partial_probe.resume_spans=[](ggml_backend_t backend,const ggml_tensor * op,ggml_backend_buffer_t workspace,
                const ggml_kv_stream_resume_plan & plan,const ggml_kv_stream_span_plan_view & view,size_t first,size_t end) {
            active->maximum_attention_span=std::max(active->maximum_attention_span,end-first);
            if (std::any_of(active->held.begin(),active->held.end(),[](const auto & item) {return item.second!=0;}))
                active->maximum_stream_span=std::max(active->maximum_stream_span,end-first);
            active->events.push_back('A');return active->partial->resume_spans(backend,op,workspace,plan,view,first,end);
        };
        reg->iface.get_proc_address=[](ggml_backend_reg_t reg,const char * name) -> void * {
            if (!std::strcmp(name,"ggml_backend_kv_stream_copy_ops"))
                return reinterpret_cast<void *>(+[]() -> const ggml_kv_stream_copy_ops * {return &active->copy_probe;});
            if (!std::strcmp(name,"ggml_backend_kv_stream_registered_copy_ops"))
                return reinterpret_cast<void *>(+[]() -> const ggml_kv_stream_registered_copy_ops * {return &active->registered_probe;});
            if (!std::strcmp(name,"ggml_backend_kv_stream_partial_ops"))
                return reinterpret_cast<void *>(+[]() -> const ggml_kv_stream_partial_ops * {return &active->partial_probe;});
            return active->original(reg,name);
        };
    }
    bool copy_between_gather_and_attention() const {
        bool copied=false;
        for (auto event : events) {
            if (event=='G') copied=false;
            if (event=='C') copied=true;
            if (event=='A' && copied) return true;
        }
        return false;
    }
    ~group_pipeline_probe() {reg->iface.get_proc_address=original;active=nullptr;}
};

static void check_owned_cache_groups(testing & t,bool complete=false,bool double_buffer=false,bool resident_role=false,uint32_t safety_mode=0) {
    ggml_backend_load_all();auto * device=ggml_backend_dev_by_name("CUDA0");
    if (!t.assert_true(device!=nullptr)) return;
    ggml_backend_ptr backend(ggml_backend_dev_init(device,nullptr));
    auto instrumentation=resident_role ? std::make_unique<group_pipeline_probe>(backend.get()) : nullptr;
    auto * registry=ggml_backend_dev_backend_reg(device);
    const auto get=reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(registry,"ggml_backend_kv_stream_partial_ops"));
    const auto copy_get=reinterpret_cast<ggml_kv_stream_copy_ops_get>(ggml_backend_reg_get_proc_address(registry,"ggml_backend_kv_stream_copy_ops"));
    const auto retire=reinterpret_cast<void (*)(ggml_backend_t)>(ggml_backend_reg_get_proc_address(registry,"ggml_backend_cuda_graph_release_all"));
    const auto * partial=get ? get() : nullptr;const auto * copies=copy_get ? copy_get() : nullptr;
    if (!t.assert_true(backend && partial && copies && copies->prepare && copies->free_prepared && retire)) return;
    size_t profiles=0;
    const ggml_type types[]={GGML_TYPE_F32,GGML_TYPE_F16,GGML_TYPE_BF16,GGML_TYPE_Q8_0,GGML_TYPE_Q5_0,GGML_TYPE_Q5_1,GGML_TYPE_Q4_0,GGML_TYPE_Q4_1};
    for (uint32_t width : {256u,512u}) for (auto key : types) for (auto value : types) for (uint32_t queries : {1u,4u}) {
        if (resident_role && (width!=256 || key!=GGML_TYPE_Q8_0 || value!=GGML_TYPE_Q4_0 || queries!=1)) continue;
        if (!complete && !((key==GGML_TYPE_Q8_0 && value==GGML_TYPE_Q4_0) || (key==GGML_TYPE_F16 && value==GGML_TYPE_F16))) continue;
        const ggml_kv_stream_shape shape{key,value,int32_t(width),int32_t(width),4,256,128};
        const uint32_t heads=width==256 ? 24 : 32;const size_t pages=resident_role ? 180 : 24;
        auto * type=llama_kv_stream_device_buffer_type(device);const auto alignment=ggml_backend_buft_get_alignment(type);
        const auto aligned=[&](size_t n) {return (n+alignment-1)/alignment*alignment;};
        ggml_kv_stream_layout tile;ggml_kv_stream_execution execution;size_t attention=0;
        if (!t.assert_true(ggml_kv_stream_layout_make(shape,256,tile).status==ggml_kv_stream_status::success &&
                ggml_kv_stream_resolve(shape,partial->capabilities(backend.get(),key,value),256,execution).status==ggml_kv_stream_status::success &&
                ggml_kv_stream_workspace_query(partial,backend.get(),{shape,heads,queries,resident_role ? 8192u : double_buffer ? 4096u : 2048u},ggml_kv_stream_workspace_kind::decode,attention))) return;
        const auto pool_bytes=pages*tile.bytes+execution.conversion.bytes,stage_offset=aligned(pool_bytes),scratch_offset=aligned(stage_offset+tile.bytes);
        arena_ptr arena(ggml_backend_memory_arena_new(type,scratch_offset+attention),ggml_backend_memory_arena_free);
        if (!t.assert_true(arena && ggml_backend_memory_arena_begin(arena.get(),0) &&
                ggml_backend_memory_arena_reserve_at(arena.get(),101,0,pool_bytes,alignment,GGML_BACKEND_MEMORY_REGION_PERSISTENT,nullptr) &&
                ggml_backend_memory_arena_reserve_at(arena.get(),201,stage_offset,tile.bytes,alignment,0,nullptr) &&
                ggml_backend_memory_arena_reserve_at(arena.get(),202,scratch_offset,attention,alignment,0,nullptr) &&
                ggml_backend_memory_arena_commit(arena.get()))) return;
        lease_ptr pool(ggml_backend_memory_arena_acquire(arena.get(),101),ggml_backend_memory_lease_free),
            staging(ggml_backend_memory_arena_acquire(arena.get(),201),ggml_backend_memory_lease_free),
            scratch(ggml_backend_memory_arena_acquire(arena.get(),202),ggml_backend_memory_lease_free);
        fixture target(backend.get(),true,key,value,resident_role ? 8192 : double_buffer ? 4096 : 2048,false,resident_role ? 16 : 2,4,pages,width,width,pool.get(),37,3);
        fixture auxiliary(backend.get(),true,key,value,resident_role ? 8192 : double_buffer ? 4096 : 2048,false,1,4,pages,width,width,pool.get(),137,93);
        std::vector<std::shared_ptr<void>> prepared;
        for (size_t owner=0;owner<2;++owner) {
            prepared.emplace_back(copies->prepare(backend.get(),pages,double_buffer),copies->free_prepared);
            if (!t.assert_true(prepared.back()!=nullptr)) return;
        }
        block_inputs target_input(target,resident_role ? 4353 : double_buffer ? 2049 : 1537,queries,false,heads),auxiliary_input(auxiliary,resident_role ? 4353 : 1025,queries,false,heads);
        stock_attention_case target_control(target,target_input),auxiliary_control(auxiliary,auxiliary_input);
        uint32_t role=0;
        const std::vector<std::shared_ptr<llama_kv_stream_host>> owners{target.host,auxiliary.host};
        std::unique_ptr<llama_kv_stream_group_reader> reader,draft;
        bool can_quiesce=true;size_t quiesces=0;
        const auto quiesce=[&] {
            ++quiesces;
            return can_quiesce && (!reader || reader->finish()) && (!draft || draft->finish());
        };
        auto buffering=resident_role ? llama_kv_stream_group_buffering::create_aggregate({{target.content,auxiliary.content}},quiesce,{1}) :
            double_buffer ? llama_kv_stream_group_buffering::create({target.content,auxiliary.content},quiesce) : nullptr;
        const auto make=[&](fixture & source,size_t tokens,uint32_t expected) {
            llama_kv_stream_group_reader_source grant{source.host,pool.get(),pages,tokens,prepared[expected],source.content};
            grant.owning_caches=owners;grant.buffering=buffering;
            return llama_kv_stream_group_reader::create(backend.get(),{grant},staging.get(),[&,expected](uint32_t) {return role==expected;});
        };
        reader=make(target,target_input.active,0);draft=make(auxiliary,auxiliary_input.active,1);
        if (!t.assert_true(reader && draft)) return;
        struct retirement {ggml_backend_t backend;void (*retire)(ggml_backend_t);
            ~retirement() {ggml_backend_synchronize(backend);retire(backend);}} captured{backend.get(),retire};
        const auto target_generation=target.content->generation(),auxiliary_generation=auxiliary.content->generation();
        std::vector<uint8_t> target_bytes(target.host->bytes()),auxiliary_bytes(auxiliary.host->bytes());
        std::memcpy(target_bytes.data(),ggml_backend_buffer_get_base(target.host->buffer()),target_bytes.size());
        std::memcpy(auxiliary_bytes.data(),ggml_backend_buffer_get_base(auxiliary.host->buffer()),auxiliary_bytes.size());
        if (resident_role) {
            bool spread=false,stale_checked=false;
            const auto reject=[&](uint32_t selected_role,block_inputs & input,stock_attention_case & control) {
                role=selected_role;auto & selected=role ? draft : reader;
                std::vector<float> canary(size_t(ggml_nelements(input.output)),0.375f);
                ggml_backend_tensor_set(input.output,canary.data(),0,ggml_nbytes(input.output));
                auto op=*control.output;op.buffer=input.output->buffer;op.data=input.output->data;
                auto k=*control.k,v=*control.v;k.buffer=v.buffer=nullptr;k.data=v.data=nullptr;op.src[1]=&k;op.src[2]=&v;
                std::vector<ggml_kv_stream_gather_row> columns(input.padded);
                for (size_t token=0;token<input.active;++token) columns[token]={0,token};
                if (!t.assert_true("unprepared aggregate reads fail before changing output",!selected->enqueue_attention(0,columns,&op,scratch.get(),true))) return false;
                ggml_backend_synchronize(backend.get());
                return t.assert_true(same_float_bits(canary,input.read()));
            };
            const auto verify=[&](uint32_t selected_role,block_inputs * alternate=nullptr,stock_attention_case * alternate_control=nullptr) {
                role=selected_role;auto & source=role ? auxiliary : target;
                auto & input=alternate ? *alternate : (role ? auxiliary_input : target_input);
                auto & control=alternate_control ? *alternate_control : (role ? auxiliary_control : target_control);
                auto & selected=role ? draft : reader;const auto before=buffering->snapshot();
                if (!t.assert_true(selected->prepare_frame({{input.active,input.active}},{queries,true,false}))) return false;
                const auto after=buffering->snapshot();
                if (after.spread_streaming) {spread=true;instrumentation->force_deadlines=false;}
                std::vector<ggml_kv_stream_gather_row> columns(input.padded);
                for (size_t token=0;token<input.active;++token) columns[token]={0,token};
                for (uint32_t layer=0;layer<source.host->config().layers;++layer) {
                    llama_kv_stream_host_layer host;if (!t.assert_true(source.host->layer(layer,host))) return false;
                    ggml_backend_tensor_set(control.k_storage,host.k,0,ggml_nbytes(control.k_storage));
                    ggml_backend_tensor_set(control.v_storage,host.v,0,ggml_nbytes(control.v_storage));
                    control.run();std::vector<float> native(ggml_nelements(control.output));
                    ggml_backend_tensor_get(control.output,native.data(),0,ggml_nbytes(control.output));
                    auto op=*control.output;op.buffer=input.output->buffer;op.data=input.output->data;
                    auto k=*control.k,v=*control.v;k.buffer=v.buffer=nullptr;k.data=v.data=nullptr;op.src[1]=&k;op.src[2]=&v;
                    if (!t.assert_true(selected->enqueue_attention(layer,columns,&op,scratch.get(),true))) return false;
                    ggml_backend_synchronize(backend.get());
                    if (!t.assert_true("role preferences preserve every native output bit",same_float_bits(native,input.read()))) return false;
                }
                if (!t.assert_true(selected->finish())) return false;
                if (safety_mode==1 && !stale_checked && (before.ring_slots!=after.ring_slots ||
                        before.decode_active_pages!=after.decode_active_pages || before.spread_streaming!=after.spread_streaming)) {
                    if (!reject(1-selected_role,selected_role ? target_input : auxiliary_input,
                            selected_role ? target_control : auxiliary_control)) return false;
                    stale_checked=true;role=selected_role;
                }
                return true;
            };
            for (size_t frame=0;frame<(safety_mode==1 ? 2u : 6u);++frame) {
                instrumentation->force_deadlines=safety_mode!=1 && !spread;
                if (!verify(0) || !verify(1)) return;
                const auto state=buffering->snapshot();
                if (spread) {
                    if (!t.assert_equal(size_t(0),draft->sequence_stats().copy_bytes) ||
                            !t.assert_equal(size_t(0),draft->resident_upload_bytes())) return;
                }
                t.assert_equal(17u,state.budget.layers);t.assert_true(!state.budget.double_buffer);
                t.assert_equal(pages,size_t(state.ring_slots)+size_t(state.resident_pages_per_layer)*17);
            }
            if (safety_mode==1) t.assert_true("the stale reader was rejected before re-preparing it",stale_checked);
            else t.assert_true("forced layer deadlines exercised a real rebalance",spread && quiesces);
            if (safety_mode==2) {
                role=0;block_inputs grown(target,4609,queries,false,heads);stock_attention_case grown_control(target,grown);
                if (!t.assert_true(target.host->prepare(grown.active) && reader->update_frontiers({{grown.active,target_input.active}}))) return;
                if (!reject(0,grown,grown_control) || !verify(0,&grown,&grown_control)) return;
                t.assert_equal(pages,size_t(buffering->snapshot().budget.pages));
            }
            t.assert_equal(target_generation,target.content->generation());t.assert_equal(auxiliary_generation,auxiliary.content->generation());
            t.assert_true(!std::memcmp(target_bytes.data(),ggml_backend_buffer_get_base(target.host->buffer()),target_bytes.size()));
            t.assert_true(!std::memcmp(auxiliary_bytes.data(),ggml_backend_buffer_get_base(auxiliary.host->buffer()),auxiliary_bytes.size()));
            return;
        }
        const std::vector<std::pair<uint32_t,uint32_t>> steps=double_buffer ?
            std::vector<std::pair<uint32_t,uint32_t>>{{0,0},{0,1},{1,0},{0,0},{0,1},{1,0}} :
            std::vector<std::pair<uint32_t,uint32_t>>{{0,0},{0,1},{1,0},{0,0},{1,0},{0,1}};
        for (const auto & step : steps) {
            role=step.first;auto & source=role ? auxiliary : target;auto & input=role ? auxiliary_input : target_input;
            auto & control=role ? auxiliary_control : target_control;auto & selected=role ? draft : reader;
            llama_kv_stream_host_layer host;if (!t.assert_true(source.host->layer(step.second,host))) return;
            ggml_backend_tensor_set(control.k_storage,host.k,0,ggml_nbytes(control.k_storage));
            ggml_backend_tensor_set(control.v_storage,host.v,0,ggml_nbytes(control.v_storage));
            control.run();std::vector<float> native(ggml_nelements(control.output));
            ggml_backend_tensor_get(control.output,native.data(),0,ggml_nbytes(control.output));
            auto op=*control.output;op.buffer=input.output->buffer;op.data=input.output->data;
            auto k=*control.k,v=*control.v;k.buffer=v.buffer=nullptr;k.data=v.data=nullptr;op.src[1]=&k;op.src[2]=&v;
            std::vector<ggml_kv_stream_gather_row> columns(input.padded);
            for (size_t token=0;token<input.active;++token) columns[token]={0,token};
            if (double_buffer && !step.second && !t.assert_true(selected->prepare_frame({{input.active,input.active}}))) return;
            if (!t.assert_true(double_buffer ? selected->enqueue_attention(step.second,columns,&op,scratch.get(),true) :
                    selected->attention(step.second,columns,&op,scratch.get(),true))) return;
            if (double_buffer && step.second+1==source.host->config().layers && !t.assert_true(selected->finish())) return;
            ggml_backend_synchronize(backend.get());
            t.out<<"owned-cache role="<<role<<" layer="<<step.second<<" D="<<width<<" queries="<<queries<<" k="<<int(key)<<" v="<<int(value)<<std::endl;
            if (!t.assert_true("alternating owning caches preserve every native attention bit",same_float_bits(native,input.read()))) return;
            if (double_buffer && step.second+1<source.host->config().layers) continue;
            const auto previous=input.read();role=1-role;
            t.assert_true(!selected->attention(step.second,columns,&op,scratch.get(),true));
            t.assert_true(same_float_bits(previous,input.read()));
        }
        if (double_buffer) {
            role=0;
            const auto old=buffering->snapshot();const auto mirror=target.content->mirror_epoch();
            if (!t.assert_true("completed grouped frames train the shared adaptive policy",old.feedback_initialized &&
                    old.samples>0 && old.layer_samples>0)) return;
            can_quiesce=false;
            if (!t.assert_true("a failed quiesce keeps the shared physical layout",!reader->prepare_frame({{2305,2049}}) &&
                    buffering->snapshot().ring_slots==old.ring_slots && target.content->mirror_epoch()==mirror)) return;
            can_quiesce=true;
            if (!t.assert_true(reader->prepare_frame({{2305,2049}}))) return;
            const auto next=buffering->snapshot();
            if (!t.assert_true("context growth keeps two tails in the same owning pool",quiesces &&
                    next.ring_slots>=2*(10-next.resident_pages_per_layer))) return;
            block_inputs grown(target,2305,queries,false,heads);
            stock_attention_case grown_control(target,grown);
            for (uint32_t selected_role : {0u,1u}) {
                role=selected_role;auto & selected=role ? draft : reader;
                auto & source=role ? auxiliary : target;auto & input=role ? auxiliary_input : grown;
                auto & control=role ? auxiliary_control : grown_control;
                if (!t.assert_true(selected->prepare_frame({{input.active,input.active}}))) return;
                std::vector<ggml_kv_stream_gather_row> columns(input.padded);
                for (size_t token=0;token<input.active;++token) columns[token]={0,token};
                for (uint32_t layer=0;layer<source.host->config().layers;++layer) {
                    llama_kv_stream_host_layer host;if (!t.assert_true(source.host->layer(layer,host))) return;
                    ggml_backend_tensor_set(control.k_storage,host.k,0,ggml_nbytes(control.k_storage));
                    ggml_backend_tensor_set(control.v_storage,host.v,0,ggml_nbytes(control.v_storage));
                    control.run();std::vector<float> native(ggml_nelements(control.output));
                    ggml_backend_tensor_get(control.output,native.data(),0,ggml_nbytes(control.output));
                    auto op=*control.output;op.buffer=input.output->buffer;op.data=input.output->data;
                    auto k=*control.k,value=*control.v;k.buffer=value.buffer=nullptr;k.data=value.data=nullptr;
                    op.src[1]=&k;op.src[2]=&value;
                    if (!t.assert_true(selected->enqueue_attention(layer,columns,&op,scratch.get(),true))) return;
                    ggml_backend_synchronize(backend.get());
                    if (!t.assert_true("both owners adopt the resized native-exact layout",same_float_bits(native,input.read()))) return;
                }
                if (!t.assert_true(selected->finish())) return;
            }
            role=0;
            if (!t.assert_true(target.content->reset_mirror() && !reader->prepare_frame({{2305,2305}}))) return;
        }
        for (int invalid=0;invalid<4;++invalid) {
            llama_kv_stream_group_reader_source bad{target.host,pool.get(),pages,1537,prepared[0],target.content};
            if (invalid==0) bad.owning_caches={auxiliary.host};
            if (invalid==1) bad.owning_caches={target.host,target.host};
            if (invalid==2) bad.owning_caches={target.host,nullptr};
            if (invalid==3) {bad.owning_caches=owners;bad.content.reset();}
            t.assert_true(!llama_kv_stream_group_reader::create(backend.get(),{bad},staging.get(),[](uint32_t) {return true;}));
        }
        t.assert_equal(target_generation,target.content->generation());t.assert_equal(auxiliary_generation,auxiliary.content->generation());
        t.assert_true(!std::memcmp(target_bytes.data(),ggml_backend_buffer_get_base(target.host->buffer()),target_bytes.size()));
        t.assert_true(!std::memcmp(auxiliary_bytes.data(),ggml_backend_buffer_get_base(auxiliary.host->buffer()),auxiliary_bytes.size()));
        ++profiles;
    }
    t.assert_equal(size_t(complete ? 256 : 8),profiles);t.out<<"owned-cache profiles="<<profiles<<std::endl;
}

// Queue real GPU work before publishing host rows, so a missing producer fence reads stale KV.
struct group_pending_host_producer {
    ggml_backend_t backend;
    ggml_context_ptr metadata;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * rows[2][2]{};
    ggml_cgraph * delay=nullptr;
    group_pending_host_producer(ggml_backend_t backend,const ggml_kv_stream_layout & page) : backend(backend) {
        metadata.reset(ggml_init({1<<20,nullptr,true}));
        auto * a=ggml_new_tensor_2d(metadata.get(),GGML_TYPE_F32,2048,2048);
        auto * b=ggml_new_tensor_2d(metadata.get(),GGML_TYPE_F32,2048,2048);
        auto * product=ggml_mul_mat(metadata.get(),a,b);
        delay=ggml_new_graph_custom(metadata.get(),64,false);ggml_build_forward_expand(delay,product);
        for (size_t slot=0;slot<2;++slot) for (size_t side=0;side<2;++side)
            rows[slot][side]=ggml_new_tensor_1d(metadata.get(),GGML_TYPE_I8,int64_t(side ? page.v_token_bytes : page.k_token_bytes));
        buffer.reset(ggml_backend_alloc_ctx_tensors(metadata.get(),backend));GGML_ASSERT(buffer);
        ggml_backend_buffer_clear(buffer.get(),0);
    }
    bool publish(const llama_kv_stream_host_layer * targets,const size_t * tokens,const ggml_kv_stream_layout & page) {
        for (size_t i=0;i<16;++i) if (ggml_backend_graph_compute_async(backend,delay)!=GGML_STATUS_SUCCESS) return false;
        for (size_t slot=0;slot<2;++slot) for (size_t side=0;side<2;++side) {
            const size_t stride=side ? page.v_token_bytes : page.k_token_bytes;
            auto * destination=static_cast<uint8_t *>(side ? targets[slot].v : targets[slot].k)+(tokens[slot]-1)*stride;
            ggml_backend_tensor_get_async(backend,rows[slot][side],destination,0,stride);
        }
        return true;
    }
};

static void check_slot_joint_attention(testing & t,bool small,bool wide_only=false,bool managed=false,bool reader_errors=false,bool reader_grid=false,bool attention_adapter=false,size_t resident_pool_pages=0,bool pending_append=false,bool pipeline_probe=false,uint32_t prefill_queries=0,uint32_t pipeline_dimension=0,bool resident_direct=false,bool require_resident_wave=true,bool resident_rewind=false,bool stream_direct=false,bool large_stream_wave=false,bool device_publication=false,bool require_batched_gather=false,bool enqueue=false,bool double_buffer=false,uint32_t owning_layers=2,bool delayed_producer=false,bool aggregate=false,size_t history0=2049,size_t history1=0,uint32_t selected_maximum=2) {
    ggml_backend_load_all();auto * device=ggml_backend_dev_by_name("CUDA0");
    if (!t.assert_true(device!=nullptr)) return;
    ggml_backend_ptr backend(ggml_backend_dev_init(device,nullptr));
    std::unique_ptr<group_pipeline_probe> pipeline;
    if (pipeline_probe || resident_direct) pipeline=std::make_unique<group_pipeline_probe>(backend.get());
    auto * reg=ggml_backend_dev_backend_reg(device);
    const auto partial_get=reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_kv_stream_partial_ops"));
    const auto copy_get=reinterpret_cast<ggml_kv_stream_copy_ops_get>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_kv_stream_copy_ops"));
    const auto * partial=partial_get ? partial_get() : nullptr;const auto * copy=copy_get ? copy_get() : nullptr;
    const auto retire=reinterpret_cast<void (*)(ggml_backend_t)>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_cuda_graph_release_all"));
    if (!t.assert_true(backend && partial && partial->resume && partial->resume_spans && copy && copy->version>=12 && copy->gather && retire)) return;
    const ggml_type types[]={GGML_TYPE_F32,GGML_TYPE_F16,GGML_TYPE_BF16,GGML_TYPE_Q8_0,GGML_TYPE_Q5_0,GGML_TYPE_Q5_1,GGML_TYPE_Q4_0,GGML_TYPE_Q4_1};
    size_t profiles=0;
    for (uint32_t width : {256u,512u}) for (auto key : types) for (auto value : types) {
        const bool extended=(key==GGML_TYPE_Q8_0 && (value==GGML_TYPE_Q4_0 || value==GGML_TYPE_Q8_0)) ||
            (key==GGML_TYPE_Q5_1 && value==GGML_TYPE_Q5_1);
        if (small && !extended && !(selected_maximum>2 && key==GGML_TYPE_F16 && value==GGML_TYPE_F16)) continue;
        if (selected_maximum>2 && resident_direct && (width!=256 || key!=GGML_TYPE_Q8_0 || value!=GGML_TYPE_Q4_0)) continue;
        if (!double_buffer && (reader_errors || pipeline_probe || resident_direct) && (width!=(pipeline_dimension ? pipeline_dimension : 256) ||
                key!=GGML_TYPE_Q8_0 || value!=GGML_TYPE_Q4_0)) continue;
        if (owning_layers>2 && (width!=256 || key!=GGML_TYPE_Q8_0 || value!=GGML_TYPE_Q4_0)) continue;
        if (double_buffer && (reader_errors || pending_append) && (width!=256 || key!=GGML_TYPE_Q8_0 || value!=GGML_TYPE_Q4_0)) continue;
        if (double_buffer && !((key==GGML_TYPE_Q8_0 && value==GGML_TYPE_Q4_0) ||
                (key==GGML_TYPE_F16 && value==GGML_TYPE_F16))) continue;
        const ggml_kv_stream_shape shape{key,value,width,width,4,256,128};
        ggml_kv_stream_layout tile;
        if (!t.assert_true(ggml_kv_stream_layout_make(shape,256,tile).status==ggml_kv_stream_status::success)) return;
        for (size_t selected : (selected_maximum>2 ? std::vector<size_t>{3,4} : std::vector<size_t>{1,2})) {
            if ((reader_errors || reader_grid || pipeline_probe || resident_direct) && selected<2) continue;
            if (wide_only && selected<2) continue;
            std::vector<std::pair<uint32_t,uint32_t>> widths={{1,1}};
            if (extended) {
                widths.push_back({4,4});widths.push_back({8,8});
                if (selected>1) {widths.push_back({1,3});widths.push_back({2,4});}
            }
            if (owning_layers>2) widths=history0>32768 ? std::vector<std::pair<uint32_t,uint32_t>>{{1,1},{4,4},{8,8}} : std::vector<std::pair<uint32_t,uint32_t>>{{1,1}};
            if (wide_only) widths={{8,8}};
            if ((double_buffer && (reader_errors || pending_append)) || (!double_buffer && (reader_errors || reader_grid || pipeline_probe))) widths={{1,3}};
            if (resident_direct) widths={{1,1},{4,4},{8,8},{1,3}};
            if (selected_maximum>2 && resident_direct) widths={{8,8}};
            if (prefill_queries) widths={{prefill_queries/2,prefill_queries/2}};
            for (auto query_counts : widths) for (size_t layout=0;layout<(selected==1 ? 1u : 5u);++layout)
            for (bool compact : {false,true}) {
                if (owning_layers>2 && layout!=1) continue;
                if (compact && layout!=0) continue;
                if (wide_only && layout!=4) continue;
                if (((double_buffer && (reader_errors || pending_append)) || (!double_buffer && (reader_errors || reader_grid || pipeline_probe))) && layout!=4) continue;
                std::vector<uint32_t> own_queries(std::max(selected,size_t(2)),query_counts.second);own_queries.front()=query_counts.first;
                uint32_t queries=0;for (size_t slot=0;slot<selected;++slot) queries+=own_queries[slot];
                const uint32_t heads=width==256 ? 24 : 32;
                std::vector<size_t> lengths(std::max(selected,size_t(2)),history1 ? history1 : (double_buffer ? 2049u : 769u));lengths.front()=history0;
                const size_t gap=layout==3 ? 256 : 0;
                size_t active=gap;for (size_t slot=0;slot<selected;++slot) active+=lengths[slot];
                const size_t padded=(active+255)/256*256;
                ggml_kv_stream_resume_plan plan;
                if (!t.assert_true(ggml_kv_stream_resume_plan_query(partial,backend.get(),{shape,heads,queries,padded},plan))) return;
                size_t scratch_bytes=plan.bytes;
                if ((wide_only || attention_adapter) && !t.assert_true(ggml_kv_stream_workspace_query(partial,backend.get(),
                        {shape,heads,queries,padded},prefill_queries || queries>16 ? ggml_kv_stream_workspace_kind::prefill : ggml_kv_stream_workspace_kind::decode,scratch_bytes))) return;
                if ((wide_only || attention_adapter) && !t.assert_true(scratch_bytes>=plan.bytes)) return;
                const size_t alignment=ggml_backend_buft_get_alignment(llama_kv_stream_device_buffer_type(device));
                ggml_kv_stream_execution page;
                if (!t.assert_true(ggml_kv_stream_resolve(shape,partial->capabilities(backend.get(),key,value),256,page).status==ggml_kv_stream_status::success)) return;
                const size_t pool_pages=(resident_pool_pages ? resident_pool_pages : 3)*(aggregate ? selected : 1);
                const size_t pool_bytes=pool_pages*tile.bytes+(resident_pool_pages ? page.conversion.bytes : 0);
                const size_t guard=(alignment+127)/128*128,kv_end=guard+(aggregate ? 1 : selected)*pool_bytes;
                const size_t stage_offset=(kv_end+guard+alignment-1)/alignment*alignment;
                const size_t partial_offset=(stage_offset+tile.bytes+guard+alignment-1)/alignment*alignment;
                const bool attention_errors=attention_adapter && reader_errors;
                const size_t short_offset=(partial_offset+scratch_bytes+guard+alignment-1)/alignment*alignment;
                const size_t parent_bytes=attention_errors ? short_offset+scratch_bytes-1+guard : partial_offset+scratch_bytes+guard;
                std::vector<std::shared_ptr<void>> prepared;
                if (managed) for (size_t slot=0;slot<(aggregate ? 1 : selected);++slot) {
                    prepared.emplace_back(copy->prepare(backend.get(),pool_pages,double_buffer || aggregate),copy->free_prepared);
                    if (!t.assert_true(prepared.back()!=nullptr)) return;
                }
                arena_ptr parent(ggml_backend_memory_arena_new(llama_kv_stream_device_buffer_type(device),parent_bytes),ggml_backend_memory_arena_free);
                if (!t.assert_true(parent && ggml_backend_memory_arena_begin(parent.get(),0))) return;
                for (size_t slot=0;slot<(aggregate ? 1 : selected);++slot) if (!t.assert_true(ggml_backend_memory_arena_reserve_at(parent.get(),101+slot,
                        guard+slot*pool_bytes,pool_bytes,alignment,GGML_BACKEND_MEMORY_REGION_PERSISTENT,nullptr))) return;
                if (!t.assert_true(ggml_backend_memory_arena_reserve_at(parent.get(),201,stage_offset,tile.bytes,alignment,0,nullptr) &&
                        ggml_backend_memory_arena_reserve_at(parent.get(),202,partial_offset,scratch_bytes,alignment,0,nullptr) &&
                        (!attention_errors || ggml_backend_memory_arena_reserve_at(parent.get(),203,short_offset,scratch_bytes-1,alignment,0,nullptr)) &&
                        ggml_backend_memory_arena_commit(parent.get()))) return;
                ggml_backend_buffer_clear(ggml_backend_memory_arena_parent(parent.get()),0xa5);
                std::vector<lease_ptr> source_lease;
                for (size_t slot=0;slot<selected;++slot) source_lease.emplace_back(nullptr,ggml_backend_memory_lease_free);
                std::vector<std::unique_ptr<fixture>> sources(selected);
                std::vector<std::vector<uint8_t>> canonical(selected);
                std::vector<std::unique_ptr<void,void (*)(void *)>> queues;
                for (size_t slot=0;slot<selected;++slot) {
                    source_lease[slot].reset(ggml_backend_memory_arena_acquire(parent.get(),101+(aggregate ? 0 : slot)));
                    sources[slot]=std::make_unique<fixture>(backend.get(),true,key,value,std::max(size_t(4096),(lengths[slot]+511)/256*256),false,owning_layers,4,pool_pages,width,width,source_lease[slot].get(),37+slot,uint32_t(slot*101));
                    const auto & host=sources[slot]->host;
                    canonical[slot].resize(host->bytes());
                    std::memcpy(canonical[slot].data(),ggml_backend_buffer_get_base(host->buffer()),host->bytes());
                    queues.emplace_back(copy->create(backend.get(),ggml_backend_memory_lease_buffer(source_lease[slot].get()),sources[slot]->host->buffer(),shape,3),copy->free);
                    if (!t.assert_true(source_lease[slot] && queues.back())) return;
                }
                lease_ptr staging(ggml_backend_memory_arena_acquire(parent.get(),201),ggml_backend_memory_lease_free),
                    scratch(ggml_backend_memory_arena_acquire(parent.get(),202),ggml_backend_memory_lease_free);
                lease_ptr short_scratch(attention_errors ? ggml_backend_memory_arena_acquire(parent.get(),203) : nullptr,ggml_backend_memory_lease_free);
                std::unique_ptr<group_pending_host_producer> pending_producer;
                if (delayed_producer) pending_producer=std::make_unique<group_pending_host_producer>(backend.get(),tile);
                std::unique_ptr<llama_kv_stream_group_reader> reader;
                std::vector<std::shared_ptr<llama_kv_stream_group_buffering>> buffer_states;
                bool source_current=true,source_throws=false;size_t source_checks=0;
                if (managed) {
                    std::vector<llama_kv_stream_group_reader_source> grants;
                    for (size_t slot=0;slot<selected;++slot)
                        grants.push_back({sources[slot]->host,source_lease[slot].get(),pool_pages,lengths[slot],prepared[aggregate ? 0 : slot],
                            resident_pool_pages ? sources[slot]->content : nullptr});
                    std::shared_ptr<llama_kv_stream_group_buffering> joint;
                    if (aggregate) {
                        std::vector<std::vector<std::shared_ptr<llama_kv_stream_content>>> owners;
                        for (size_t slot=0;slot<selected;++slot) owners.push_back({sources[slot]->content});
                        joint=llama_kv_stream_group_buffering::create_aggregate(owners);
                        if (!t.assert_true(bool(joint))) return;
                    }
                    if (double_buffer || aggregate) for (size_t slot=0;slot<selected;++slot) {
                        grants[slot].buffering=aggregate ? joint : llama_kv_stream_group_buffering::create({sources[slot]->content});
                        if (!t.assert_true(bool(grants[slot].buffering))) return;
                        if (owning_layers>2) grants[slot].stable_tokens=lengths[slot]-own_queries[slot];
                        buffer_states.push_back(grants[slot].buffering);
                    }
                    reader=llama_kv_stream_group_reader::create(backend.get(),grants,staging.get(),
                        [&](uint32_t) {++source_checks;if (source_throws) throw std::runtime_error("source proof fault");return source_current;});
                    if (!t.assert_true("native group reader constructed with content and prepared feedback events",reader!=nullptr)) return;
                    if (aggregate) {
                        std::vector<llama_kv_stream_group_frontier> frontiers;
                        for (size_t slot=0;slot<selected;++slot) frontiers.push_back({lengths[slot],owning_layers>2 ? lengths[slot]-own_queries[slot] : lengths[slot]});
                        if (!t.assert_true(reader->prepare_frame(frontiers,{queries,!prefill_queries,false}))) return;
                        t.assert_equal(owning_layers,buffer_states.front()->snapshot().budget.layers);
                        t.assert_true(!buffer_states.front()->snapshot().budget.double_buffer);
                    }
                }
                fixture reference(backend.get(),true,key,value,std::max(size_t(double_buffer ? 8192 : 4096),padded+256),false,owning_layers,4,3,width,width,nullptr,49,0);
                std::vector<ggml_kv_stream_gather_row> columns(padded),packet(256);
                size_t cursor=gap;
                const size_t rounds=layout>=2 ? 8 : 0;
                for (size_t slot=0;slot<selected;++slot) {
                    const size_t prefix=lengths[slot]-(layout ? own_queries[slot]+rounds : 0);
                    for (size_t token=0;token<prefix;++token) columns[cursor++]={uint32_t(slot),token};
                }
                for (size_t round=0;round<rounds;++round) for (size_t slot=0;slot<selected;++slot)
                    columns[cursor++]={uint32_t(slot),lengths[slot]-own_queries[slot]-rounds+round};
                if (layout) for (size_t slot=0;slot<selected;++slot) for (size_t q=0;q<own_queries[slot];++q)
                    columns[cursor++]={uint32_t(slot),lengths[slot]-own_queries[slot]+q};
                if (layout==4) {
                    cursor=0;
                    for (size_t row=0;row<lengths[0];++row) for (size_t slot=0;slot<selected;++slot)
                        if (row<lengths[slot]) columns[cursor++]={uint32_t(slot),(row*37)%lengths[slot]};
                }
                if (!t.assert_equal(active,cursor)) return;
                block_inputs input(reference,active,queries,false,heads);
                std::vector<ggml_fp16_t> mask(padded*queries,ggml_fp32_to_fp16(-INFINITY));
                std::vector<int32_t> intervals(4*queries);
                for (size_t q=0;q<queries;++q) {
                    size_t owner=0,local=q;
                    while (local>=own_queries[owner]) {local-=own_queries[owner];++owner;}
                    for (size_t col=0;col<active;++col) if (columns[col].source==owner && columns[col].token<=lengths[owner]-own_queries[owner]+local)
                        mask[q*padded+col]=ggml_fp32_to_fp16(0);
                    ggml_backend_execution_mask_interval interval;
                    const bool representable=ggml_backend_execution_mask_interval_encode(mask.data()+q*padded,padded,interval);
                    if (layout==0) {
                        if (!t.assert_true(representable)) return;
                        std::memcpy(intervals.data()+4*q,&interval,sizeof(interval));
                    } else if (!t.assert_true("native interleaving requires its exact dense mask",!representable)) return;
                }
                ggml_backend_tensor_set(input.mask,mask.data(),0,mask.size()*sizeof(mask[0]));
                ggml_context_ptr bounds_meta(ggml_init({4096,nullptr,true}));
                auto * bounds=ggml_new_tensor_2d(bounds_meta.get(),GGML_TYPE_I32,4,queries);
                ggml_backend_buffer_ptr bounds_buffer(ggml_backend_alloc_ctx_tensors(bounds_meta.get(),backend.get()));
                if (!t.assert_true(bool(bounds_buffer))) return;
                if (compact) ggml_backend_tensor_set(bounds,intervals.data(),0,intervals.size()*sizeof(int32_t));
                struct retirement {ggml_backend_t backend;void (*release)(ggml_backend_t);
                    ~retirement() {ggml_backend_synchronize(backend);release(backend);}} captures{backend.get(),retire};
                const auto generation_a=sources[0]->content->generation(),generation_b=selected==2 ? sources[1]->content->generation() : 0;
                size_t max_sources=0;bool appended=false,compact_active=compact,rewound=false;
                if (pipeline) pipeline->uploads.clear();
                for (uint32_t layer=0;layer<owning_layers;++layer) {
                    std::vector<llama_kv_stream_host_layer> origin(selected);llama_kv_stream_host_layer native;
                    for (size_t slot=0;slot<selected;++slot) if (!t.assert_true(sources[slot]->host->layer(layer,origin[slot]))) return;
                    if (!t.assert_true(reference.host->layer(layer,native))) return;
                    for (size_t col=0;col<active+size_t(appended);++col) for (bool side : {false,true}) {
                        const auto & row=columns[col];const size_t bytes=side ? tile.v_token_bytes : tile.k_token_bytes;
                        if (row.source==UINT32_MAX) continue;
                        std::memcpy(static_cast<uint8_t *>(side ? native.v : native.k)+col*bytes,
                            static_cast<const uint8_t *>(side ? origin[row.source].v : origin[row.source].k)+row.token*bytes,bytes);
                    }
                    if (pending_producer) for (size_t slot=0;slot<selected;++slot) for (size_t side=0;side<2;++side) {
                        const auto type=side ? value : key;const auto stride=side ? tile.v_token_bytes : tile.k_token_bytes;
                        std::vector<float> values(width*4,side ? 7.0f : 0.0f);std::vector<uint8_t> encoded(stride);
                        ggml_quantize_chunk(type,values.data(),encoded.data(),0,1,width*4,nullptr);
                        ggml_backend_tensor_set(pending_producer->rows[slot][side],encoded.data(),0,stride);
                        const auto * base=static_cast<const uint8_t *>(ggml_backend_buffer_get_base(sources[slot]->host->buffer()));
                        const auto * plane=static_cast<const uint8_t *>(side ? origin[slot].v : origin[slot].k);
                        std::memcpy(canonical[slot].data()+size_t(plane-base)+(lengths[slot]-1)*stride,encoded.data(),stride);
                        for (size_t col=0;col<active;++col) if (columns[col].source==slot && columns[col].token+1==lengths[slot])
                            std::memcpy(static_cast<uint8_t *>(side ? native.v : native.k)+col*stride,encoded.data(),stride);
                    }
                    stock_attention_case control(reference,input);
                    if (layer) {
                        ggml_backend_tensor_set(control.k_storage,native.k,0,ggml_nbytes(control.k_storage));
                        ggml_backend_tensor_set(control.v_storage,native.v,0,ggml_nbytes(control.v_storage));
                    }
                    for (size_t replay=0;replay<(double_buffer ? 1u : resident_rewind ? 4u : 3u);++replay) {
                        llama_kv_stream_write pending;
                        const bool publishing=pending_append && !layer && replay==(double_buffer ? 0u : 1u);
                        if (resident_rewind && !layer && replay==3) {
                            const size_t keep=lengths[0]-17;
                            const std::vector<llama_kv_stream_group_frontier> shorter={{keep,keep},{lengths[1],lengths[1]}};
                            t.assert_true("an unproven backward frontier is rejected",!reader->update_frontiers(shorter));
                            if (!t.assert_true(sources[0]->content->invalidate_suffix(keep))) return;
                            if (!t.assert_true("completed suffix invalidation preserves the reader",reader->update_frontiers(shorter))) return;
                            for (size_t col=0;col<columns.size();++col) if (columns[col].source==0 && columns[col].token>=keep) {
                                columns[col]={UINT32_MAX,0};
                                for (size_t q=0;q<queries;++q) mask[q*padded+col]=ggml_fp32_to_fp16(-INFINITY);
                            }
                            ggml_backend_tensor_set(input.mask,mask.data(),0,mask.size()*sizeof(mask[0]));
                            compact_active=false;rewound=true;
                        }
                        if (publishing) {
                            std::vector<std::vector<uint8_t>> encoded(2);
                            std::vector<llama_kv_stream_write_span> spans;
                            for (bool side : {false,true}) {
                                const auto dtype=side ? value : key;
                                const size_t bytes=side ? tile.v_token_bytes : tile.k_token_bytes;
                                encoded[side].resize(bytes);std::vector<float> values(width*4,side ? 2.0f : 0.0f);
                                ggml_quantize_chunk(dtype,values.data(),encoded[side].data(),0,1,width*4,nullptr);
                                spans.push_back({layer,side ? ggml_kv_stream_operand::v : ggml_kv_stream_operand::k,lengths[0]*bytes,nullptr,bytes});
                            }
                            if (!t.assert_true(sources[0]->content->prepare_direct_generated(spans,[&](const auto & span,void * destination) {
                                const bool side=span.operand==ggml_kv_stream_operand::v;
                                std::memcpy(destination,encoded[side].data(),encoded[side].size());return true;
                            },pending))) return;
                            for (bool side : {false,true}) {
                                const size_t bytes=side ? tile.v_token_bytes : tile.k_token_bytes;
                                const auto * plane=static_cast<const uint8_t *>(side ? origin[0].v : origin[0].k);
                                const auto * base=static_cast<const uint8_t *>(ggml_backend_buffer_get_base(sources[0]->host->buffer()));
                                std::memcpy(canonical[0].data()+size_t(plane-base)+lengths[0]*bytes,encoded[side].data(),bytes);
                                std::memcpy(static_cast<uint8_t *>(side ? native.v : native.k)+active*bytes,encoded[side].data(),bytes);
                            }
                            columns[active]={0,lengths[0]};mask[active]=ggml_fp32_to_fp16(0);
                            ggml_backend_tensor_set(input.mask,mask.data(),0,mask.size()*sizeof(mask[0]));
                            if (compact_active) {
                                for (size_t q=0;q<queries;++q) {
                                    ggml_backend_execution_mask_interval interval;
                                    if (!ggml_backend_execution_mask_interval_encode(mask.data()+q*padded,padded,interval)) {
                                        compact_active=false;break;
                                    }
                                    std::memcpy(intervals.data()+4*q,&interval,sizeof(interval));
                                }
                                t.assert_true("a separated appended token requires the exact dense mask",!compact_active);
                                if (compact_active) ggml_backend_tensor_set(bounds,intervals.data(),0,intervals.size()*sizeof(intervals[0]));
                            }
                            ggml_backend_tensor_set(control.k_storage,native.k,0,ggml_nbytes(control.k_storage));
                            ggml_backend_tensor_set(control.v_storage,native.v,0,ggml_nbytes(control.v_storage));
                            if (!t.assert_true(reader->update_frontiers({{lengths[0]+1,lengths[0]},{lengths[1],lengths[1]}}))) return;
                            if (device_publication) {
                                const std::vector<ggml_kv_stream_gather_row> rows={{0,lengths[0]}};
                                auto wrong=rows;wrong[0].source=2;t.assert_true(!reader->can_publish(staging.get(),layer,wrong));
                                wrong=rows;--wrong[0].token;t.assert_true(!reader->can_publish(staging.get(),layer,wrong));
                                if (!t.assert_true("a complete resident append admits native device publication",reader->can_publish(staging.get(),layer,rows))) return;
                                t.assert_true(!reader->complete_publication(layer,rows,false));
                                ggml_context_ptr encoded_metadata(ggml_init({4096,nullptr,true}));
                                for (bool side : {false,true}) {
                                    auto * tensor=ggml_new_tensor_1d(encoded_metadata.get(),side ? value : key,int64_t(width*4));
                                    auto * buffer=ggml_backend_memory_lease_buffer(staging.get());
                                    if (!t.assert_true(ggml_backend_tensor_alloc(buffer,tensor,static_cast<char *>(ggml_backend_buffer_get_base(buffer))+(side ? tile.v_offset : 0))==GGML_STATUS_SUCCESS)) return;
                                    ggml_backend_tensor_set(tensor,encoded[side].data(),0,encoded[side].size());
                                    t.assert_true(!reader->publish_encoded(source_lease[0].get(),layer,side,tensor,0,1,rows));
                                    t.assert_true(!reader->publish_encoded(staging.get(),layer,side,tensor,1,1,rows));
                                    if (!t.assert_true(reader->publish_encoded(staging.get(),layer,side,tensor,0,1,rows))) return;
                                    if (!side) t.assert_true(!reader->complete_publication(layer,rows,false));
                                }
                                ggml_backend_synchronize(backend.get());
                                if (!t.assert_true(reader->complete_publication(layer,rows,false))) return;
                                t.assert_true("device readiness is not a host-commit proof",!reader->complete_publication(layer,rows,true));
                            }
                            t.assert_equal(generation_a,sources[0]->content->generation());
                            appended=true;
                        }
                        control.run();std::vector<float> expected(ggml_nelements(control.output));
                        ggml_backend_tensor_get(control.output,expected.data(),0,ggml_nbytes(control.output));
                        if (managed && !attention_adapter) {
                            if (!t.assert_true(reader->begin(layer,columns))) return;
                        } else if (!managed) for (size_t slot=0;slot<selected;++slot) if (!t.assert_true(copy->begin(queues[slot].get()))) return;
                        auto op=*control.output;op.buffer=input.output->buffer;op.data=input.output->data;
                        auto logical_k=*control.k,logical_v=*control.v;
                        logical_k.buffer=logical_v.buffer=nullptr;logical_k.data=logical_v.data=nullptr;
                        op.src[1]=&logical_k;op.src[2]=&logical_v;
                        if (compact_active) {op.src[3]=nullptr;if (!t.assert_true(ggml_backend_execution_set_mask_bounds(&op,bounds))) return;}
                        std::vector<ggml_kv_stream_copy_rectangle> rectangles;
                        std::vector<size_t> next_physical(selected);
                        for (size_t slot=0;slot<selected;++slot) next_physical[slot]=slot%3;
                        if (attention_adapter) {
                            if (attention_errors && !layer && !replay) {
                                const auto snapshot=[&] {
                                    ggml_tensor raw{};auto * buffer=ggml_backend_memory_arena_parent(parent.get());
                                    raw.type=GGML_TYPE_I8;raw.ne[0]=int64_t(ggml_backend_buffer_get_size(buffer));raw.ne[1]=raw.ne[2]=raw.ne[3]=1;
                                    raw.nb[0]=1;raw.nb[1]=raw.nb[2]=raw.nb[3]=size_t(raw.ne[0]);raw.buffer=buffer;raw.data=ggml_backend_buffer_get_base(buffer);
                                    std::vector<uint8_t> bytes(size_t(raw.ne[0]));ggml_backend_tensor_get(&raw,bytes.data(),0,bytes.size());return bytes;
                                };
                                const auto before=snapshot();const auto checks=source_checks;
                                std::vector<float> canary(size_t(ggml_nelements(input.output)),0.375f);
                                ggml_backend_tensor_set(input.output,canary.data(),0,canary.size()*sizeof(float));
                                auto bad=op;bad.data=ggml_backend_buffer_get_base(ggml_backend_memory_lease_buffer(staging.get()));
                                bad.buffer=ggml_backend_memory_lease_buffer(staging.get());
                                t.assert_true(!reader->attention(layer,columns,&bad,scratch.get(),!prefill_queries));
                                bad=op;auto query=*op.src[0];query.data=ggml_backend_buffer_get_base(ggml_backend_memory_lease_buffer(source_lease[0].get()));
                                query.buffer=ggml_backend_memory_lease_buffer(source_lease[0].get());bad.src[0]=&query;
                                t.assert_true(!reader->attention(layer,columns,&bad,scratch.get(),!prefill_queries));
                                bad=op;bad.data=op.src[0]->data;bad.buffer=op.src[0]->buffer;
                                t.assert_true(!reader->attention(layer,columns,&bad,scratch.get(),!prefill_queries));
                                t.assert_true(!reader->attention(layer,columns,&op,staging.get(),!prefill_queries));
                                if (!t.assert_true(bool(short_scratch))) return;
                                t.assert_true(!reader->attention(layer,columns,&op,short_scratch.get(),!prefill_queries));
                                t.assert_equal(checks,source_checks);
                                source_current=false;t.assert_true(!reader->attention(layer,columns,&op,scratch.get(),!prefill_queries));source_current=true;
                                source_throws=true;t.assert_true(!reader->attention(layer,columns,&op,scratch.get(),!prefill_queries));source_throws=false;
                                ggml_backend_synchronize(backend.get());
                                t.assert_true(before==snapshot());
                                t.assert_true(same_float_bits(canary,input.read()));
                            }
                            if (pipeline && reader_errors && !layer && !replay) {
                                for (int kind : {0,1,2}) {
                                    const std::vector<float> before(size_t(ggml_nelements(input.output)),.375f);
                                    ggml_backend_tensor_set(input.output,before.data(),0,before.size()*sizeof(float));
                                    pipeline->fail_kind=kind;pipeline->fail_seen=0;pipeline->fail_fired=false;
                                    t.assert_true("queued transfer/wait failure closes the read",
                                        !(double_buffer ? reader->enqueue_attention(layer,columns,&op,scratch.get(),!prefill_queries) :
                                            reader->attention(layer,columns,&op,scratch.get(),!prefill_queries)));
                                    t.assert_true(pipeline->fail_fired && pipeline->fail_seen>0);
                                    t.assert_true("failure before attention preserves the public output",same_float_bits(before,input.read()));
                                    pipeline->fail_kind=-1;
                                }
                            }
                            if (pipeline) {pipeline->events.clear();pipeline->before_attention_k.clear();pipeline->before_gather_k.clear();pipeline->maximum_held=0;pipeline->maximum_attention_span=0;pipeline->maximum_stream_span=0;pipeline->small_native_copies=0;}
#ifdef KV_STREAM_READER_SYNC_TRACE
                            if (enqueue) kv_stream_reader_capture_syncs(true);
#endif
                            if (owning_layers>2) t.out<<"multi-layer ring read layer="<<layer<<std::endl;
                            if (pending_producer && !t.assert_true(pending_producer->publish(origin.data(),lengths.data(),tile))) return;
                            const bool read=(enqueue || double_buffer) ? reader->enqueue_attention(layer,columns,&op,scratch.get(),!prefill_queries) :
                                reader->attention(layer,columns,&op,scratch.get(),!prefill_queries);
#ifdef KV_STREAM_READER_SYNC_TRACE
                            if (enqueue) {
                                kv_stream_reader_capture_syncs(false);
                                if (replay==2 && (!t.assert_true("the repeated resident read has no host refresh",reader->resident_upload_bytes()==0) ||
                                        !t.assert_true("a clean resident attention enqueues without a CPU backend wait",kv_stream_reader_syncs()==0))) return;
                            }
#else
                            if (enqueue) {t.assert_true("native backend synchronization tracing is required",false);return;}
#endif
                            if (!t.assert_true(read)) return;
                            if (aggregate && history0>32768 && plan.span_resume) {
                                if (!t.assert_true("a native wave combines the resident prefix and bounded streamed span",
                                        pipeline->maximum_attention_span>96*256)) return;
                            }
                            if (aggregate && pipeline) for (size_t slot=0;slot<selected;++slot) {
                                llama_kv_stream_host_layer current;
                                if (!t.assert_true(sources[slot]->host->layer(layer,current))) return;
                                const auto base=uintptr_t(current.k);
                                std::map<size_t,std::pair<size_t,size_t>> physical;
                                for (const auto & upload:pipeline->uploads) if (upload.address>=base && upload.address-base<sources[slot]->host->layout().k_bytes)
                                    physical[(upload.address-base)/tile.k_token_bytes]={upload.slot,upload.pages};
                                size_t expected=SIZE_MAX;
                                for (const auto & entry:physical) {
                                    const auto & span=entry.second;
                                    if (expected!=SIZE_MAX && expected!=span.first)
                                        t.out<<"fragmented ring layer="<<layer<<" owner="<<slot<<" token="<<entry.first<<" expected_slot="<<expected<<" actual_slot="<<span.first<<'\n';
                                    if (!t.assert_true("an owner's streamed tail remains contiguous across cyclic ring wrap",expected==SIZE_MAX || expected==span.first)) return;
                                    expected=(span.first+span.second)%buffer_states.front()->snapshot().ring_slots;
                                }
                                if (!layer && !slot && lengths[0]>32768) {
                                    if (!t.assert_true(!physical.empty())) return;
                                    const size_t tail=(lengths[0]-own_queries[0])/256*256-physical.begin()->first;
                                    if (!t.assert_true("the long-tail fixture requires more than 96 full pages",tail>96*256)) return;
                                }
                            }
                            if (aggregate && !layer && resident_pool_pages==90 && lengths[0]==2049 && lengths[1]==2049) {
                                const bool future=std::any_of(pipeline->before_gather_k.begin(),pipeline->before_gather_k.end(),[&](uintptr_t address) {
                                    for (size_t slot=0;slot<selected;++slot) {
                                        llama_kv_stream_host_layer first;if (!sources[slot]->host->layer(0,first)) return false;
                                        const auto base=uintptr_t(first.k);
                                        if (address>=base && address-base<sources[slot]->host->layout().k_bytes) return false;
                                    }
                                    return true;
                                });
                                if (!t.assert_true("initial spare capacity is uploaded before the first gather without waiting for a refill batch",future)) return;
                            }
                            if (aggregate && layer && pipeline && pipeline->maximum_held) {
                                const auto probe=std::find(pipeline->events.begin(),pipeline->events.end(),'P');
                                const auto attention=std::find(pipeline->events.begin(),pipeline->events.end(),'A');
                                if (!t.assert_true("joint KV readiness is sampled before any resident attention",
                                        probe!=pipeline->events.end() && attention!=pipeline->events.end() && probe<attention)) return;
                                if (!t.assert_equal(size_t(1),size_t(std::count(pipeline->events.begin(),pipeline->events.end(),'P')))) return;
                            }
                            if (double_buffer && !layer && !t.assert_true("both-slot rings queue next-layer history before returning",reader->sequence_stats().max_layer_distance>=1)) return;
                            if (double_buffer && !aggregate && !layer) for (size_t slot=0;slot<selected;++slot) {
                                llama_kv_stream_host_layer future;
                                if (!t.assert_true(sources[slot]->host->layer(1,future))) return;
                                const auto first=uintptr_t(future.k)+size_t(buffer_states[slot]->snapshot().resident_pages_per_layer)*256*tile.k_token_bytes;
                                if (!t.assert_true("each slot submits its next-layer tail before attention",std::find(pipeline->before_attention_k.begin(),
                                        pipeline->before_attention_k.end(),first)!=pipeline->before_attention_k.end())) return;
                            }

                        } else for (size_t first=0;first<padded;first+=256) {
                            if (managed) {
                                if (!t.assert_true(reader->gather(first))) return;
                            } else {
                            bool initialize=true;size_t covered=0;
                            for (size_t slot=0;slot<selected;++slot) {
                                std::vector<size_t> pages;
                                for (size_t col=first;col<first+256;++col) if (columns[col].source==slot) pages.push_back(columns[col].token/256);
                                std::sort(pages.begin(),pages.end());pages.erase(std::unique(pages.begin(),pages.end()),pages.end());
                                ggml_kv_stream_layout ring;
                                if (!t.assert_true(ggml_kv_stream_layout_make(shape,768,ring).status==ggml_kv_stream_status::success)) return;
                                for (size_t batch=0;batch<pages.size();batch+=3) {
                                    const size_t count=std::min(size_t(3),pages.size()-batch);
                                    ggml_kv_stream_span spans[3];size_t physical[3];
                                    for (size_t part=0;part<count;++part) {
                                        const size_t page=pages[batch+part],token=page*256,live=std::min(size_t(256),lengths[slot]-token);
                                        physical[part]=(next_physical[slot]+part)%3;
                                        if (!t.assert_true(copy->enqueue_span(queues[slot].get(),physical[part],
                                                static_cast<const uint8_t *>(origin[slot].k)+token*tile.k_token_bytes,
                                                static_cast<const uint8_t *>(origin[slot].v)+token*tile.v_token_bytes,live,256) &&
                                                copy->acquire_span(queues[slot].get(),physical[part],1))) return;
                                        auto * buffer=ggml_backend_memory_lease_buffer(source_lease[slot].get());
                                        spans[part]={buffer,buffer,token,256,physical[part]*tile.k_bytes,ring.v_offset+physical[part]*tile.v_bytes};
                                    }
                                    max_sources=std::max(max_sources,count);
                                    std::vector<ggml_kv_stream_span_plan_view> views(selected);
                                    for (size_t owner=0;owner<selected;++owner) views[owner]={shape,nullptr,0,(lengths[owner]+255)/256*256,own_queries[owner]};
                                    views[slot].spans=spans;views[slot].count=count;
                                    std::fill(packet.begin(),packet.end(),ggml_kv_stream_gather_row{});
                                    for (size_t col=first;col<first+256;++col) if (columns[col].source==slot &&
                                            std::binary_search(pages.begin()+batch,pages.begin()+batch+count,columns[col].token/256)) {
                                        packet[col-first]=columns[col];++covered;
                                    }
                                    if (!t.assert_true(ggml_kv_stream_gather_make(views.data(),selected,packet,0,
                                            ggml_backend_memory_lease_buffer(staging.get()),rectangles) &&
                                            copy->gather(backend.get(),ggml_backend_memory_lease_buffer(staging.get()),rectangles.data(),rectangles.size(),initialize))) return;
                                    for (size_t part=0;part<count;++part)
                                        if (!t.assert_true(copy->release_span(queues[slot].get(),physical[part],1))) return;
                                    next_physical[slot]=(next_physical[slot]+count)%3;initialize=false;
                                }
                            }
                            if (initialize && !t.assert_true(copy->gather(backend.get(),ggml_backend_memory_lease_buffer(staging.get()),nullptr,0,true))) return;
                            size_t expected_coverage=0;
                            for (size_t col=first;col<first+256;++col) expected_coverage+=columns[col].source!=UINT32_MAX;
                            if (!t.assert_equal(expected_coverage,covered)) return;
                            }
                            if (plan.span_resume) {
                                auto * buffer=ggml_backend_memory_lease_buffer(staging.get());
                                const ggml_kv_stream_span span{buffer,buffer,first,256,0,tile.v_offset};
                                const ggml_kv_stream_span_plan_view wave{shape,&span,1,padded,queries};
                                if (!t.assert_true(partial->resume_spans(backend.get(),&op,
                                        ggml_backend_memory_lease_buffer(scratch.get()),plan,wave,first,first+256))) return;
                                continue;
                            }
                            ggml_tensor k{},v{};
                            for (auto * tensor : {&k,&v}) {
                                const bool side=tensor==&v;tensor->type=ggml_type(side ? value : key);
                                tensor->ne[0]=width;tensor->ne[1]=256;tensor->ne[2]=4;tensor->ne[3]=1;
                                tensor->nb[0]=ggml_type_size(tensor->type);tensor->nb[1]=side ? tile.v_token_bytes : tile.k_token_bytes;
                                tensor->nb[2]=side ? tile.v_row_bytes : tile.k_row_bytes;tensor->nb[3]=side ? tile.v_bytes : tile.k_bytes;
                                tensor->buffer=ggml_backend_memory_lease_buffer(staging.get());
                                tensor->data=static_cast<char *>(ggml_backend_buffer_get_base(tensor->buffer))+(side ? tile.v_offset : 0);
                            }
                            auto step=op;step.src[1]=&k;step.src[2]=&v;ggml_tensor slice;
                            if (!compact_active) {slice=*input.mask;slice.ne[0]=256;slice.data=static_cast<uint8_t *>(input.mask->data)+first*sizeof(ggml_fp16_t);step.src[3]=&slice;}
                            if (!t.assert_true(partial->resume(backend.get(),&step,ggml_backend_memory_lease_buffer(scratch.get()),plan,padded,first,first+256==padded))) return;
                        }
                        if (managed) {
#ifdef KV_STREAM_READER_SYNC_TRACE
                            if (enqueue) kv_stream_reader_capture_syncs(true);
#endif
                            if ((!double_buffer || layer+1==owning_layers) && !t.assert_true(reader->finish())) return;
#ifdef KV_STREAM_READER_SYNC_TRACE
                            if (enqueue) {
                                kv_stream_reader_capture_syncs(false);
                                if (replay==2 && !t.assert_true("finish fences a pending resident read even after logical close",kv_stream_reader_syncs()>0)) return;
                            }
#endif
                        } else for (size_t slot=0;slot<selected;++slot) copy->drain(queues[slot].get());
                        ggml_backend_synchronize(backend.get());
                        const auto actual=input.read();
                        if (!same_float_bits(expected,actual)) {
                            size_t different=0;float worst=0;
                            for (size_t i=0;i<expected.size();++i) if (std::memcmp(&expected[i],&actual[i],sizeof(float))) {
                                ++different;worst=std::max(worst,std::abs(expected[i]-actual[i]));
                            }
                            t.out<<"joint mismatch layer="<<layer<<" replay="<<replay<<" layout="<<layout<<" compact="<<compact_active
                                <<" different="<<different<<" max_abs="<<worst<<std::endl;
                        }
                        if (!t.assert_true("shared native attention preserves every output bit",same_float_bits(expected,actual))) return;
                        if (device_publication && publishing && !t.assert_true("device-published append needs no host refresh",reader->resident_upload_bytes()==0)) return;
                        if (pipeline_probe && !double_buffer && !t.assert_true("next-wave copies are queued before the current attention dispatch",
                                pipeline->copy_between_gather_and_attention())) return;
                        if (resident_direct && require_resident_wave && layout!=4 && !t.assert_true(stream_direct ? large_stream_wave ?
                                "a native wave retains more than three ring pages across physical wrap" :
                                "several ring pages stay protected through one native continuation wave" :
                                "contiguous resident rows share one native continuation wave",
                                stream_direct && large_stream_wave ? pipeline->maximum_held>3 && (!plan.span_resume || pipeline->maximum_stream_span>768) :
                                stream_direct && resident_pool_pages==3 ? pipeline->maximum_stream_span==256 && pipeline->maximum_held<=1 :
                                (stream_direct ? pipeline->maximum_stream_span : pipeline->maximum_attention_span)>(large_stream_wave ? 768u : 256u))) return;
#ifdef KV_STREAM_COPY_ENTRY_TRACE
                        if (require_batched_gather && !t.assert_true("small gather rectangles do not issue separate native DMA calls",pipeline->small_native_copies==0)) return;
#else
                        if (require_batched_gather) {t.assert_true("copy entry tracing is required",false);return;}
#endif
                        if (resident_rewind && rewound && replay==3 && !t.assert_true("rollback uploads at most one discarded padding page, retaining the unchanged prefix",
                                reader->resident_upload_bytes()<=tile.bytes)) return;
                        if (device_publication && resident_rewind && rewound && replay==3 && !t.assert_true("masked discarded padding does not require an upload",
                                reader->resident_upload_bytes()==0)) return;
                        if (pipeline && prefill_queries>=128) t.assert_true("wide prefill ready waves use at most half the ring",
                            pipeline->maximum_held<=(resident_pool_pages-2)/2+(resident_pool_pages-2)%2);
                        if (resident_pool_pages==26 && !t.assert_true("resident prefix uploads only initially or after a pending append",
                                (!replay || publishing) ? reader->resident_upload_bytes()>0 : reader->resident_upload_bytes()==0)) return;
                        if (publishing) {
                            if (!t.assert_true(pending.pending() && sources[0]->content->commit(pending))) return;
                            if (device_publication) {
                                if (!t.assert_true(reader->complete_publication(layer,{{0,lengths[0]}},true))) return;
                                for (auto operand : {ggml_kv_stream_operand::k,ggml_kv_stream_operand::v}) {
                                    bool dirty=true;
                                    t.assert_true(sources[0]->content->dirty({layer,operand,lengths[0],1},dirty) && !dirty);
                                }
                            }
                            t.assert_true(sources[0]->content->flush({{layer,ggml_kv_stream_operand::k,lengths[0],1},
                                {layer,ggml_kv_stream_operand::v,lengths[0],1}},[](const auto &) {return true;}));
                            if (!t.assert_true(reader->update_frontiers({{lengths[0]+1,lengths[0]+1},{lengths[1],lengths[1]}}))) return;
                        }
                    }
                }
                if (reader_errors) {
                    std::vector<llama_kv_stream_group_reader_source> grants;
                    for (size_t slot=0;slot<selected;++slot)
                        grants.push_back({sources[slot]->host,source_lease[slot].get(),3,lengths[slot],prepared[slot]});
                    auto duplicate=grants;duplicate[1].pool=duplicate[0].pool;
                    t.assert_true(!llama_kv_stream_group_reader::create(backend.get(),duplicate,staging.get(),[](uint32_t) {return true;}));
                    t.assert_true(!llama_kv_stream_group_reader::create(backend.get(),grants,source_lease[0].get(),[](uint32_t) {return true;}));
                    auto unprepared=grants;unprepared[1].prepared_copies.reset();
                    t.assert_true(!llama_kv_stream_group_reader::create(backend.get(),unprepared,staging.get(),[](uint32_t) {return true;}));
                    auto bad=columns;bad.back()={0,lengths[0]};
                    t.assert_true(!reader->begin(0,bad));
                    source_current=false;t.assert_true(!reader->begin(0,columns));source_current=true;
                    source_throws=true;
                    t.assert_true("source proof exceptions close admission",!reader->begin(0,columns));
                    source_throws=false;
                    if (!t.assert_true(reader->begin(0,columns))) return;
                    t.assert_true(!reader->begin(0,columns) && !reader->gather(256));
                    if (!t.assert_true(reader->gather(0))) return;
                    source_current=false;
                    t.assert_true(!reader->gather(256));
                    t.assert_true(reader->finish() && reader->finish());source_current=true;
                    if (!t.assert_true(reader->begin(1,columns))) return;
                    for (size_t col=0;col<padded;col+=256) if (!t.assert_true(reader->gather(col))) return;
                    t.assert_true(!reader->gather(padded) && reader->finish());
                }
                if (layout==4 && !managed) t.assert_equal(size_t(3),max_sources);
                t.assert_equal(generation_a+uint64_t(appended)+uint64_t(rewound),sources[0]->content->generation());
                if (selected==2) t.assert_equal(generation_b,sources[1]->content->generation());
                for (size_t slot=0;slot<selected;++slot) t.assert_true("joint reads preserve all canonical host bytes",
                    !std::memcmp(canonical[slot].data(),ggml_backend_buffer_get_base(sources[slot]->host->buffer()),canonical[slot].size()));
                ggml_tensor raw{};auto * buffer=ggml_backend_memory_arena_parent(parent.get());
                raw.type=GGML_TYPE_I8;raw.ne[0]=int64_t(ggml_backend_buffer_get_size(buffer));raw.ne[1]=raw.ne[2]=raw.ne[3]=1;
                raw.nb[0]=1;raw.nb[1]=raw.nb[2]=raw.nb[3]=size_t(raw.ne[0]);raw.buffer=buffer;raw.data=ggml_backend_buffer_get_base(buffer);
                for (auto offset : {size_t(0),kv_end,stage_offset+tile.bytes,partial_offset+scratch_bytes}) {
                    std::vector<uint8_t> bytes(guard);ggml_backend_tensor_get(&raw,bytes.data(),offset,bytes.size());
                    t.assert_true(std::all_of(bytes.begin(),bytes.end(),[](uint8_t x) {return x==0xa5;}));
                }
                if (resident_rewind) {
                    t.assert_true(sources[0]->content->reset_mirror());
                    t.assert_true("a different physical mirror epoch cannot reuse this reader",!reader->update_frontiers(
                        {{lengths[0]-17,lengths[0]-17},{lengths[1],lengths[1]}}));
                }
                ++profiles;
                t.out << "slot joint D="<<width<<" K="<<ggml_type_name(key)<<" V="<<ggml_type_name(value)<<" selected="<<selected
                      <<" queries="<<own_queries[0]<<'+'<<(selected==2 ? own_queries[1] : 0)<<" layout="<<layout<<" compact="<<compact
                      <<" total_queries="<<queries<<" stage="<<tile.bytes<<" sources="<<max_sources<<" native-plan="<<plan.kernel_config[0]<<" splits="<<plan.splits<<std::endl;
            }
        }
    }
    t.assert_equal(size_t(selected_maximum>2 ? (resident_direct ? 12 : 384) : owning_layers>2 ? (history0>32768 ? 3 : 1) : double_buffer ? ((reader_errors || pending_append) ? 1 : 72) : resident_direct ? 24 : (reader_errors || pipeline_probe) ? 1 : reader_grid ? 128 : wide_only ? 128 : small ? 216 : 1192),profiles);
}

int main(int argc, char ** argv) {
    if (argc>1 && (!std::strcmp(argv[1],"--cuda-slot-group-resident-direct") || !std::strcmp(argv[1],"--cuda-slot-group-stream-direct") ||
            !std::strcmp(argv[1],"--cuda-slot-group-stream-wide") || !std::strcmp(argv[1],"--cuda-slot-group-stream-single") ||
            !std::strcmp(argv[1],"--cuda-slot-group-device-publication") || !std::strcmp(argv[1],"--cuda-slot-group-resident-gather-batch") || !std::strcmp(argv[1],"--cuda-slot-group-resident-enqueue"))) {
        const bool streaming=std::strcmp(argv[1],"--cuda-slot-group-resident-direct") && std::strcmp(argv[1],"--cuda-slot-group-device-publication") && std::strcmp(argv[1],"--cuda-slot-group-resident-gather-batch") && std::strcmp(argv[1],"--cuda-slot-group-resident-enqueue");
        testing t;t.test(streaming ? "ring_ranges_preserve_native_columns_masks_and_output_bits" :
                "resident_ranges_preserve_native_columns_masks_and_output_bits",[&](testing & t) {
            for (auto dimension : {256u,512u})
                check_slot_joint_attention(t,false,false,true,false,false,true,streaming ?
                    !std::strcmp(argv[1],"--cuda-slot-group-stream-single") ? 3 : 10 : 128,true,false,0,dimension,true,
                    !(argc>2 && !std::strcmp(argv[2],"--reference")),argc>2 && !std::strcmp(argv[2],"--rewind"),streaming,
                    !std::strcmp(argv[1],"--cuda-slot-group-stream-wide"),!std::strcmp(argv[1],"--cuda-slot-group-device-publication"),
                    !std::strcmp(argv[1],"--cuda-slot-group-resident-gather-batch"),!std::strcmp(argv[1],"--cuda-slot-group-resident-enqueue"));
        });
        return t.summary();
    }
    if (argc>1 && !std::strcmp(argv[1],"--cuda-selected-workspace")) {
        testing t;
        t.test("aggregate_K7_verification_workspace_covers_all_native_widths_without_enabling_K15",[&](testing & t) {
            ggml_backend_load_all();auto * device=ggml_backend_dev_by_name("CUDA0");
            if (!t.assert_true(device != nullptr)) return;
            ggml_backend_ptr backend(ggml_backend_dev_init(device,nullptr));
            const auto get=reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(device),"ggml_backend_kv_stream_partial_ops"));
            const auto * ops=get ? get() : nullptr;
            if (!t.assert_true(backend && ops && ops->version>=15)) return;
            const ggml_type formats[]={GGML_TYPE_F32,GGML_TYPE_F16,GGML_TYPE_BF16,GGML_TYPE_Q8_0,
                GGML_TYPE_Q5_0,GGML_TYPE_Q5_1,GGML_TYPE_Q4_0,GGML_TYPE_Q4_1};
            size_t profiles=0;
            for (uint32_t dimension : {256u,512u}) for (auto key : formats) for (auto value : formats)
            for (size_t tokens : {size_t(2304),size_t(131072)}) {
                size_t previous=0;
                for (uint32_t queries=1;queries<=16;++queries) {
                    const ggml_kv_stream_attention_request request{{key,value,dimension,dimension,4,256,128},
                        dimension==256 ? 24u : 32u,queries,tokens,3};
                    ggml_kv_stream_resume_plan native;
                    if (!t.assert_true("the native aggregate continuation plan is available",
                            ggml_kv_stream_resume_plan_query(ops,backend.get(),request,native))) return;
                    size_t bytes=73;
                    if (!t.assert_true("aggregate decode quote covers two eight-row verification members",
                            ggml_kv_stream_workspace_query(ops,backend.get(),request,ggml_kv_stream_workspace_kind::decode,bytes))) return;
                    t.assert_true(bytes>=native.bytes && bytes>=previous);
                    previous=bytes; ++profiles;
                }
                size_t untouched=73;
                const ggml_kv_stream_attention_request invalid{{key,value,dimension,dimension,4,256,128},
                    dimension==256 ? 24u : 32u,17,tokens,3};
                t.assert_true(!ggml_kv_stream_workspace_query(ops,backend.get(),invalid,
                    ggml_kv_stream_workspace_kind::decode,untouched));
                t.assert_equal(size_t(73),untouched);
                t.assert_true(!ops->decode_workspace(backend.get(),key,value,24,4,16,tokens,untouched));
                t.assert_equal(size_t(73),untouched);
            }
            t.assert_equal(size_t(4096),profiles);
        });
        return t.summary();
    }
    if (argc>1 && !std::strcmp(argv[1],"--cuda-aggregate-safety")) {
        testing t;t.test("stale_aggregate_placement_is_rejected",[&](testing & t) {check_owned_cache_groups(t,false,true,true,1);});
        t.test("new_page_requires_a_prepared_aggregate_extent",[&](testing & t) {check_owned_cache_groups(t,false,true,true,2);});
        return t.summary();
    }
    if (argc>1 && !std::strcmp(argv[1],"--cuda-aggregate-resident-role")) {
        testing t;t.test("rebalancing_keeps_the_single_layer_role_resident",[&](testing & t) {check_owned_cache_groups(t,false,true,true);});
        return t.summary();
    }
    if (argc>1 && !std::strcmp(argv[1],"--cuda-aggregate-large-tail")) {
        testing t;t.test("aggregate_long_tails_preserve_native_query_shapes",[&](testing & t) {
            check_slot_joint_attention(t,false,false,true,false,false,true,200,false,true,0,256,
                false,true,false,false,false,false,false,false,true,3,false,true,33793,33793);
        });
        return t.summary();
    }
    if (argc>1 && !std::strcmp(argv[1],"--cuda-aggregate-shared-ring")) {
        testing t;t.test("aggregate_layer_uses_one_serial_policy_ring",[&](testing & t) {
            for (auto histories : {std::pair<size_t,size_t>{2049,2049},{2050,2307}}) for (size_t pool : {89u,90u})
                check_slot_joint_attention(t,false,false,true,false,false,true,pool,false,true,0,256,
                    false,true,false,false,false,false,false,false,true,16,false,true,histories.first,histories.second);
        });
        t.test("aggregate_layers_wait_for_pending_host_producers",[&](testing & t) {
            check_slot_joint_attention(t,false,false,true,false,false,true,25,false,true,0,256,
                false,true,false,false,false,false,false,false,true,4,true,true,2050,2307);
        });
        return t.summary();
    }
    if (argc>1 && !std::strcmp(argv[1],"--cuda-double-buffer-producer-fences")) {
        testing t;t.test("every_layer_waits_for_its_pending_host_producer",[&](testing & t) {
            check_slot_joint_attention(t,false,false,true,false,false,true,25,false,true,0,256,
                false,true,false,false,false,false,false,false,true,4,true);
        });
        return t.summary();
    }
    if (argc>1 && !std::strcmp(argv[1],"--cuda-double-buffer-upload-fences")) {
        testing t;t.test("wrapped_multi_layer_reads_wait_each_upload_fence",[&](testing & t) {
            for (size_t pages : {89u,88u})
                check_slot_joint_attention(t,false,false,true,false,false,true,pages,false,true,0,256,
                    false,true,false,false,false,false,false,false,true,16);
        });
        return t.summary();
    }
    if (argc>1 && !std::strcmp(argv[1],"--cuda-owned-double-buffer")) {
        testing t;t.test("owned_double_buffer_repartitions_both_roles_after_quiesce",[&](testing & t) {
            check_owned_cache_groups(t,false,true);
        });
        return t.summary();
    }
    if (argc>1 && !std::strcmp(argv[1],"--cuda-slot-group-double-buffer-pending")) {
        testing t;t.test("double_buffer_pending_rows_cancel_old_future_copies",[&](testing & t) {
            check_slot_joint_attention(t,false,false,true,false,false,true,18,true,true,0,256,
                false,true,false,false,false,false,false,false,true);
        });
        return t.summary();
    }
    if (argc>1 && !std::strcmp(argv[1],"--cuda-slot-group-double-buffer-errors")) {
        testing t;t.test("double_buffer_failure_drains_future_transfers",[&](testing & t) {
            check_slot_joint_attention(t,false,false,true,true,false,true,18,false,true,0,256,
                false,true,false,false,false,false,false,false,true);
        });
        return t.summary();
    }
    if (argc>1 && !std::strcmp(argv[1],"--cuda-slot-group-double-buffer")) {
        testing t;t.test("selected_group_double_buffers_next_layer_for_both_slots",[&](testing & t) {
            check_slot_joint_attention(t,false,false,true,false,false,true,18,false,true,0,256,
                false,true,false,false,false,false,false,false,true);
        });
        return t.summary();
    }
    if (argc>1 && !std::strcmp(argv[1],"--cuda-slot-group-pipeline-wide")) {
        testing t;t.test("wide_selected_groups_keep_native_order_and_bounded_transfer_waves",[&](testing & t) {
            for (auto dimension : {256u,512u}) for (auto queries : {64u,128u,256u}) for (size_t ring : {1u,3u})
                check_slot_joint_attention(t,false,false,true,false,false,true,2+ring,false,true,queries,dimension);
        });
        return t.summary();
    }
    if (argc>1 && !std::strcmp(argv[1],"--cuda-slot-group-pipeline")) {
        testing t;t.test("selected_group_queues_a_future_wave_before_attention",[&](testing & t) {
            check_slot_joint_attention(t,false,false,true,true,false,true,3,false,true);
        });
        return t.summary();
    }
    if (argc>1 && (!std::strcmp(argv[1],"--cuda-slot-group-resident-append") || !std::strcmp(argv[1],"--cuda-slot-group-resident-ring-append"))) {
        testing t;t.test("pending_native_append_refreshes_the_cached_mirror_before_publication",[&](testing & t) {
            check_slot_joint_attention(t,false,false,true,false,true,true,
                !std::strcmp(argv[1],"--cuda-slot-group-resident-append") ? 26 : 3,true);
        });
        return t.summary();
    }
    if (argc>1 && (!std::strcmp(argv[1],"--cuda-slot-group-resident-grid") || !std::strcmp(argv[1],"--cuda-slot-group-resident-ring-grid"))) {
        testing t;t.test("selected_group_resident_cache_reuses_the_existing_page_policy",[&](testing & t) {
            check_slot_joint_attention(t,false,false,true,false,true,true,
                !std::strcmp(argv[1],"--cuda-slot-group-resident-grid") ? 26 : 3);
        });
        return t.summary();
    }
    if (argc>1 && !std::strcmp(argv[1],"--cuda-N-slot-direct-attention")) {
        testing t;t.test("Wide_N_source_verification_retains_native_resident_spans",[&](testing & t) {
            check_slot_joint_attention(t,true,false,true,false,false,true,26,false,false,0,0,true,true,false,false,false,false,false,false,false,2,false,true,2049,0,4);
        });
        return t.summary();
    }
    if (argc>1 && !std::strcmp(argv[1],"--cuda-N-slot-attention")) {
        testing t;t.test("N_source_aggregate_attention_is_native_exact_with_guarded_workspace",[&](testing & t) {
            check_slot_joint_attention(t,true,false,true,false,false,true,3,false,false,0,0,false,true,false,false,false,false,false,false,false,2,false,true,2049,0,4);
        });
        return t.summary();
    }
    if (argc>1 && (!std::strcmp(argv[1],"--cuda-slot-group-attention") || !std::strcmp(argv[1],"--cuda-slot-group-attention-grid") ||
            !std::strcmp(argv[1],"--cuda-slot-group-attention-errors"))) {
        testing t;t.test("selected_group_attention_adapter_preserves_exact_native_outputs",[&](testing & t) {
            check_slot_joint_attention(t,argc>2 && !std::strcmp(argv[2],"--small"),false,true,
                !std::strcmp(argv[1],"--cuda-slot-group-attention-errors"),!std::strcmp(argv[1],"--cuda-slot-group-attention-grid"),true);
        });
        return t.summary();
    }
    if (argc>1 && (!std::strcmp(argv[1],"--cuda-slot-group-reader-errors") || !std::strcmp(argv[1],"--cuda-slot-group-reader-grid"))) {
        testing t;t.test("selected_group_reader_guards_and_all_native_encodings",[&](testing & t) {
            check_slot_joint_attention(t,false,false,true,!std::strcmp(argv[1],"--cuda-slot-group-reader-errors"),
                !std::strcmp(argv[1],"--cuda-slot-group-reader-grid"));
        });
        return t.summary();
    }
    if (argc>1 && !std::strcmp(argv[1],"--cuda-slot-owned-cache-groups")) {
        testing t;t.test("owning_target_and_auxiliary_caches_share_one_pool_without_aliasing",[&](testing & t) {check_owned_cache_groups(t,argc>2 && !std::strcmp(argv[2],"--all-formats"));});
        return t.summary();
    }
    if (argc>1 && !std::strcmp(argv[1],"--cuda-slot-group-reader")) {
        testing t;
        t.test("selected_group_reader_preserves_joint_native_attention",[&](testing & t) {
            check_slot_joint_attention(t,argc>2 && !std::strcmp(argv[2],"--small"),false,true);
        });
        return t.summary();
    }
    if (argc>1 && !std::strcmp(argv[1],"--cuda-slot-joint-wide")) {
        testing t;t.test("all_formats_use_quoted_aggregate_K7_workspace_with_native_query_order",[&](testing & t) {
            check_slot_joint_attention(t,false,true);
        });
        return t.summary();
    }
    if (argc>1 && !std::strcmp(argv[1],"--cuda-slot-joint")) {
        testing t;t.test("projected_native_batches_share_attention_with_exclusive_slot_pages",[&](testing & t) {
            check_slot_joint_attention(t,argc>2 && !std::strcmp(argv[2],"--small"));
        });
        return t.summary();
    }

    if (argc>1 && !std::strcmp(argv[1],"--cuda-slot-queues")) {
        testing t;t.test("shared_parent_slots_keep_queue_progress_and_feedback_independent",[&](testing & t) {check_slot_page_queues(t);});
        return t.summary();
    }

    if (argc>1 && !std::strcmp(argv[1],"--cuda-slot-pages")) {
        testing t;t.test("independent_page_grants_preserve_native_span_bits",[&](testing & t) { check_slot_page_partitions(t); });
        return t.summary();
    }

    if (argc > 1 && (!std::strcmp(argv[1],"--cuda-d512-pipeline") ||
            !std::strcmp(argv[1],"--cuda-d512-pipeline-hazards"))) {
        testing t;
        t.test("d512_feature_pipeline_preserves_order_across_masks_tails_waves_and_rejected_calls", [&](testing & t) {
            check_d512_pipeline(t,!std::strcmp(argv[1],"--cuda-d512-pipeline-hazards"));
        });
        return t.summary();
    }
    if (argc > 1 && !std::strcmp(argv[1],"--cuda-d512-pipeline-admission")) {
#ifdef KV_STREAM_PASCAL_CUDA_TEST
        testing t;
        t.test("feature_pipeline_reports_total_shared_memory_and_rejects_one_byte_short", [](testing & t) {
            ggml_backend_load_all();
            auto * dev=ggml_backend_dev_by_name("CUDA0");
            if (!t.assert_true(dev != nullptr)) return;
            ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
            auto * reg=ggml_backend_dev_backend_reg(dev);
            auto get=reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_kv_stream_partial_ops"));
            using status=ggml_cuda_kv_stream_plan_status;
            using style=ggml_cuda_kv_stream_execution_style;
            using query_t=status (*)(ggml_backend_t,const ggml_tensor *,const ggml_kv_stream_span_plan_view *,style,ggml_cuda_kv_stream_attention_plan &);
            auto query=reinterpret_cast<query_t>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_cuda_kv_stream_attention_plan"));
            fixture f(backend.get(),true,GGML_TYPE_F16,GGML_TYPE_F16,2304,false,2,4,0,512,512);
            if (!t.assert_true(get && get() && query && f.attach())) return;
            block_inputs input(f,2049,129,false,32);
            stock_attention_case stock(f,input);
            stock.run();
            std::vector<float> expected(ggml_nelements(stock.output));
            ggml_backend_tensor_get(stock.output,expected.data(),0,ggml_nbytes(stock.output));
            prepared_prefill_case candidate(f,input,stock,get(),1024);
            if (!t.assert_true(candidate.initialize() && candidate.run(false,nullptr,true))) return;
            auto storage=span_storage::create(f,0,input.padded,input.queries,{0,255,1025,input.padded},true);
            ggml_kv_stream_span_plan_view view;
            if (!t.assert_true(storage && ggml_kv_stream_span_plan_get_view(storage->plan,view))) return;
            auto op=*stock.output;
            op.src[3]=nullptr; op.buffer=input.output->buffer; op.data=input.output->data;
            if (!t.assert_true(ggml_backend_execution_set_mask_bounds(&op,candidate.bounds))) return;
            ggml_backend_execution_set_stock_conversion(&op,true);
            ggml_cuda_kv_stream_attention_plan plain, resumed;
            auto dense=op;
            dense.src[3]=input.mask; dense.src[5]=nullptr;
            const auto plain_status=query(backend.get(),&dense,&view,style::spanned,plain);
            const auto resume_status=query(backend.get(),&op,&view,style::resumable,resumed);
            t.out << "pipeline-plan-status spanned=" << int(plain_status) << " resumed=" << int(resume_status) << '\n';
            if (!t.assert_true(plain_status == status::success && resume_status == status::success)) return;
            size_t shared=0, local=0; int registers=0;
            if (!t.assert_true(kv_stream_test_dimension_resources(backend.get(),GGML_TYPE_F16,GGML_TYPE_F16,
                    512,8,8,true,false,shared,local,registers))) return;
            const size_t total=plain.requirements.shared_bytes+shared;
            t.assert_equal(total,resumed.requirements.shared_bytes);
            t.out << "pipeline-shared-quota dynamic=" << plain.requirements.shared_bytes << " static=" << shared
                << " required=" << total << " reported=" << resumed.requirements.shared_bytes << '\n';
            const auto saved=resumed;
            const size_t previous=kv_stream_test_shared_limit(backend.get(),total-1);
            struct restore {ggml_backend_t backend; size_t limit; ~restore(){kv_stream_test_shared_limit(backend,limit);}} restore_limit{backend.get(),previous};
            const std::vector<float> sentinel(expected.size(),-77);
            ggml_backend_tensor_set(input.output,sentinel.data(),0,ggml_nbytes(input.output));
            t.assert_true(query(backend.get(),&op,&view,style::resumable,resumed) == status::unsupported_launch_resources);
            t.assert_equal(saved.requirements.scratch_bytes,resumed.requirements.scratch_bytes);
            t.assert_equal(saved.requirements.shared_bytes,resumed.requirements.shared_bytes);
            auto * scratch=ggml_backend_memory_lease_buffer(candidate.scratch->lease.get());
            t.assert_true(!get()->resume_spans(f.backend,&op,scratch,candidate.plans[0],view,0,input.padded));
            t.assert_true("short shared quota preserves output",same_float_bits(sentinel,input.read()));
            kv_stream_test_shared_limit(backend.get(),total);
            if (!t.assert_true(query(backend.get(),&op,&view,style::resumable,resumed) == status::success)) return;
            if (!t.assert_true(get()->resume_spans(f.backend,&op,scratch,candidate.plans[0],view,0,input.padded))) return;
            ggml_backend_synchronize(f.backend);
            t.assert_true("exact shared quota preserves stock bits",same_float_bits(expected,input.read()));
        });
        return t.summary();
#else
        return 77;
#endif
    }
    if (argc > 1 && !std::strcmp(argv[1],"--cuda-d512-pipeline-resources")) {
#ifdef KV_STREAM_PASCAL_CUDA_TEST
        testing t;
        t.test("feature_pipeline_reserves_shared_memory_only_for_its_compiled_geometry", [](testing & t) {
            ggml_backend_load_all();
            auto * dev=ggml_backend_dev_by_name("CUDA0");
            if (!t.assert_true(dev != nullptr)) return;
            ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
            const bool sm120=kv_stream_test_compiled_cc(backend.get()) == 1200;
            for (uint32_t width : {256u,512u})
                for (const auto & pair : {std::make_pair(GGML_TYPE_F16,GGML_TYPE_F16),
                        std::make_pair(GGML_TYPE_Q8_0,GGML_TYPE_Q4_0)})
                for (const auto & geometry : {std::make_pair(1u,8u),std::make_pair(2u,8u),
                        std::make_pair(4u,8u),std::make_pair(8u,8u),std::make_pair(4u,2u),std::make_pair(8u,2u)})
                for (bool resume : {false,true}) for (bool dense : {false,true}) {
                size_t shared=SIZE_MAX, local=SIZE_MAX;
                int registers=0;
                if (!t.assert_true(kv_stream_test_dimension_resources(backend.get(),pair.first,pair.second,
                        width,geometry.first,geometry.second,resume,dense,shared,local,registers))) return;
                const bool pipeline=sm120 && width == 512 && pair.first == GGML_TYPE_F16 &&
                    geometry.first == 8 && geometry.second == 8 && resume && !dense;
                t.out << "pipeline-resource width=" << width << " k=" << ggml_type_name(pair.first)
                    << " columns=" << geometry.first << " groups=" << geometry.second << " resume=" << resume
                    << " dense=" << dense << " shared=" << shared << " local=" << local << " registers=" << registers << '\n';
                t.assert_equal(pipeline ? size_t(16384) : size_t(0),shared);
                t.assert_true(registers > 0);
            }
        });
        return t.summary();
#else
        return 77;
#endif
    }
    if (argc > 1 && !std::strcmp(argv[1], "--cuda-prefill-preparation-quota")) {
        testing t;
        t.test("prefill_preparation_is_quoted_before_arena_allocation", [](testing & t) {
            ggml_backend_load_all();
            auto * dev = ggml_backend_dev_by_name("CUDA0");
            if (!t.assert_true(dev != nullptr)) return;
            ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
            auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(dev),"ggml_backend_kv_stream_partial_ops"));
            const auto * ops = get ? get() : nullptr;
            for (uint32_t width : {256u,512u}) for (size_t tokens : {8192u,131072u}) for (uint32_t queries : {64u,128u,256u}) {
                ggml_kv_stream_shape shape{GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,width,width,4,256,128};
                const ggml_kv_stream_attention_request request{shape,width == 512 ? 32u : 24u,queries,tokens};
                auto decoded = request;
                decoded.shape.type_k = decoded.shape.type_v = GGML_TYPE_F16;
                size_t encoded_bytes = 0, native_bytes = 0;
                if (!t.assert_true(ggml_kv_stream_workspace_query(ops,backend.get(),request,
                        ggml_kv_stream_workspace_kind::prefill,encoded_bytes))) return;
                if (!t.assert_true(ggml_kv_stream_workspace_query(ops,backend.get(),decoded,
                        ggml_kv_stream_workspace_kind::prefill,native_bytes))) return;
                ggml_kv_stream_layout chunk;
                if (!t.assert_true(ggml_kv_stream_layout_make(decoded.shape,2048,chunk).status == ggml_kv_stream_status::success)) return;
                const size_t expected = native_bytes+(queries >= 128 ? chunk.bytes : 0);
                t.out << "prefill-preparation-quote width=" << width << " queries=" << queries << " context=" << tokens
                    << " native=" << native_bytes << " actual=" << encoded_bytes << " expected=" << expected << '\n';
                t.assert_true("quoted prefill includes bounded F16 chunk",encoded_bytes == expected);
            }
        });
        return t.summary();
    }
    if (argc > 1 && (!std::strcmp(argv[1], "--cuda-prefill-conversion-reuse") ||
            !std::strcmp(argv[1], "--cuda-prefill-prepared-live") ||
            !std::strcmp(argv[1], "--bench-prefill-conversion-reuse"))) {
        testing t;
        const bool benchmark = !std::strcmp(argv[1], "--bench-prefill-conversion-reuse");
        t.test("bounded_conversion_reuses_exact_native_values_without_changing_attention_order", [&](testing & t) {
            check_prepared_prefill(t, benchmark, !std::strcmp(argv[1], "--cuda-prefill-prepared-live"));
        });
        return t.summary();
    }
    if (argc > 1 && !std::strcmp(argv[1],"--cuda-gemma-resources")) {
#ifdef KV_STREAM_PASCAL_CUDA_TEST
        testing t;
        t.test("wide_native_plans_load_repeatedly_without_changing_resource_identity", [](testing & t) {
            ggml_backend_load_all();
            auto * dev=ggml_backend_dev_by_name("CUDA0");
            if (!t.assert_true(dev != nullptr)) return;
            ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
            for (const auto & pair : {std::make_pair(GGML_TYPE_Q8_0,GGML_TYPE_Q4_0),std::make_pair(GGML_TYPE_Q8_0,GGML_TYPE_Q8_0),
                    std::make_pair(GGML_TYPE_Q5_1,GGML_TYPE_Q5_1),std::make_pair(GGML_TYPE_F16,GGML_TYPE_F16)})
                for (const auto & geometry : {std::make_pair(1u,8u),std::make_pair(2u,8u),std::make_pair(4u,8u),
                        std::make_pair(8u,8u),std::make_pair(4u,2u),std::make_pair(8u,2u)})
                    for (bool resume : {false,true}) for (bool dense : {false,true}) {
                t.assert_true(kv_stream_test_dimension_instance(backend.get(),pair.first,pair.second,512,
                    geometry.first,geometry.second,true,resume,dense,true));
                t.assert_true(kv_stream_test_dimension_instance(backend.get(),pair.first,pair.second,512,
                    geometry.first,geometry.second,true,resume,dense));
            }
            t.assert_true(kv_stream_test_runtime_shared_instances());
        });
        return t.summary();
#else
        return 77;
#endif
    }
    if (argc > 1 && !std::strcmp(argv[1],"--cuda-gemma-tile")) {
#ifdef KV_STREAM_PASCAL_CUDA_TEST
        testing t;
        t.test("head512_tile_single_and_progressive_waves_match_native", [](testing & t) {
            ggml_backend_load_all();
            auto * dev=ggml_backend_dev_by_name("CUDA0");
            if (!t.assert_true(dev != nullptr)) return;
            ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
            const ggml_type formats[]={GGML_TYPE_F32,GGML_TYPE_F16,GGML_TYPE_BF16,GGML_TYPE_Q8_0,
                GGML_TYPE_Q5_0,GGML_TYPE_Q5_1,GGML_TYPE_Q4_0,GGML_TYPE_Q4_1};
            for (auto key : formats) for (auto value : formats) {
                constexpr size_t active=2049;
                fixture f(backend.get(),true,key,value,active,false,2,4,32,512,512);
                if (!t.assert_true(f.attach())) return;
                for (uint32_t queries : {1u,2u,3u,4u,5u,6u,7u,8u,9u,16u,17u,129u,256u}) {
                    block_inputs input(f,active,queries,false,32);
                    const auto expected=stock_attention(f,input,0,1.f/16,kv_stream_test_native_tile);
                    auto storage=span_storage::create(f,0,active,queries,{0,256,768,active},true);
                    ggml_kv_stream_span_plan_view view;
                    if (!t.assert_true(storage && ggml_kv_stream_span_plan_get_view(storage->plan,view))) return;
                    ggml_context_ptr ctx(ggml_init({65536,nullptr,true}));
                    auto * node=f.resident->attention(ctx.get(),0,input.q,input.mask,active,1.f/16);
                    if (!t.assert_true(node != nullptr)) return;
                    auto op=*node; op.data=input.output->data; op.buffer=input.output->buffer;
                    ggml_kv_stream_resume_plan plan;
                    size_t bound=0;
                    if (!t.assert_true(kv_stream_test_tile_plan(backend.get(),&op,input.padded,plan,bound))) return;
                    t.assert_true(plan.head_dim_k == 512 && plan.head_dim_v == 512 && plan.span_resume);
                    t.assert_true(plan.kernel_config[2]*plan.kernel_config[3] <= 16);
                    t.assert_true(plan.bytes <= bound);
                    for (size_t history : {size_t(4096),size_t(65536),size_t(262144)}) {
                        ggml_kv_stream_resume_plan other;
                        size_t other_bound=0;
                        t.assert_true(kv_stream_test_tile_plan(backend.get(),&op,history,other,other_bound));
                        t.assert_equal(bound,other_bound);
                        t.assert_true(other.bytes <= bound);
                    }
                    block_workspace workspace(f,bound,103);
                    auto * scratch=ggml_backend_memory_lease_buffer(workspace.lease.get());
                    auto * bounds=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_I32,4,queries);
                    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()));
                    if (!t.assert_true(bool(buffer))) return;
                    std::vector<int32_t> intervals(4*queries);
                    for (uint32_t row=0;row<queries;++row) {
                        intervals[4*row]=0; intervals[4*row+1]=int32_t(active-queries+row+1);
                        intervals[4*row+2]=int32_t(input.padded); intervals[4*row+3]=1;
                    }
                    ggml_backend_tensor_set(bounds,intervals.data(),0,ggml_nbytes(bounds));
                    for (bool compact : {false,true}) {
                        auto current=op;
                        if (compact) {
                            current.src[3]=nullptr;
                            if (!t.assert_true(ggml_backend_execution_set_mask_bounds(&current,bounds))) return;
                        }
                        const auto label=std::string(ggml_type_name(key))+"/"+ggml_type_name(value)+" TG"+std::to_string(queries)+
                            (compact ? " compact" : " dense");
                        if (!t.assert_true(kv_stream_test_tile_run(backend.get(),&current,view,scratch,plan,0,plan.tokens))) return;
                        ggml_backend_synchronize(backend.get());
                        t.assert_true(label+" complete stock bits",same_float_bits(expected,input.read()));
                        std::vector<float> sentinel(expected.size(),-77);
                        ggml_backend_tensor_set(input.output,sentinel.data(),0,ggml_nbytes(input.output));
                        auto first=view; first.count=2;
                        if (!t.assert_true(kv_stream_test_tile_run(backend.get(),&current,first,scratch,plan,0,768))) return;
                        ggml_backend_synchronize(backend.get());
                        t.assert_true(label+" no early publication",same_float_bits(sentinel,input.read()));
                        auto second=view; second.spans+=2; second.count=1;
                        if (!t.assert_true(kv_stream_test_tile_run(backend.get(),&current,second,scratch,plan,768,plan.tokens))) return;
                        ggml_backend_synchronize(backend.get());
                        t.assert_true(label+" resumed stock bits",same_float_bits(expected,input.read()));
                        block_workspace short_workspace(f,plan.bytes-1,107);
                        t.assert_true(!kv_stream_test_tile_run(backend.get(),&current,view,
                            ggml_backend_memory_lease_buffer(short_workspace.lease.get()),plan,0,plan.tokens));
                        t.assert_true(same_float_bits(expected,input.read()));
                    }
                }
            }
        });
        return t.summary();
#else
        return 77;
#endif
    }
    if (argc > 1 && !std::strcmp(argv[1],"--cuda-gemma-convert")) {
        testing t;
        t.test("head512_tile_conversion_matches_native", [](testing & t) { check_mma_tile_conversion(t,512); });
        return t.summary();
    }
    if (argc > 1 && !std::strcmp(argv[1],"--cuda-gemma-spans")) {
        testing t;
        t.test("head512_spans_preserve_stock_order_for_target_verification", [](testing & t) {
            ggml_backend_load_all();
            auto * dev=ggml_backend_dev_by_name("CUDA0");
            if (!t.assert_true(dev != nullptr)) return;
            ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
            auto get=reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(dev),"ggml_backend_kv_stream_partial_ops"));
            const auto * ops=get ? get() : nullptr;
            if (!t.assert_true(ops && ops->spans && ops->spans_workspace)) return;
            const ggml_type formats[]={GGML_TYPE_F32,GGML_TYPE_F16,GGML_TYPE_BF16,GGML_TYPE_Q8_0,
                GGML_TYPE_Q5_0,GGML_TYPE_Q5_1,GGML_TYPE_Q4_0,GGML_TYPE_Q4_1};
            for (auto key : formats) for (auto value : formats) {
                const auto pair=std::make_pair(key,value);
                constexpr size_t active=8193;
                fixture f(backend.get(),true,pair.first,pair.second,active,false,2,4,82,512,512);
                if (!t.assert_true(f.attach())) return;
                for (uint32_t queries : {1u,2u,3u,4u}) for (bool masked : {false,true}) {
                    block_inputs input(f,active,queries,masked,32);
                    const auto expected=stock_attention(f,input,0,1.0f);
                    for (const auto & cuts : {std::vector<size_t>{0,active},std::vector<size_t>{0,3328,active},
                            std::vector<size_t>{0,3328,6400,active},std::vector<size_t>{0,127,257,active}}) {
                        for (bool wrapped : {false,true}) {
                            if (wrapped && cuts.size() != 4) continue;
                            auto spans=span_storage::create(f,0,active,queries,cuts,wrapped);
                            if (!t.assert_true(bool(spans))) return;
                            bool accepted=false;
                            const auto actual=evaluate(f,ops,*spans,input,0,accepted,nullptr,1.0f);
                            if (!t.assert_true(accepted)) return;
                            t.assert_true(same_float_bits(expected,actual));
                        }
                    }
                }
            }
        });
        t.test("head512_rejects_short_workspace_and_invalid_metadata_without_writes", [](testing & t) {
            ggml_backend_load_all();
            auto * dev=ggml_backend_dev_by_name("CUDA0");
            if (!t.assert_true(dev != nullptr)) return;
            ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
            auto get=reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(dev),"ggml_backend_kv_stream_partial_ops"));
            const auto * ops=get ? get() : nullptr;
            fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,1025,false,2,4,32,512,512);
            if (!t.assert_true(ops && f.attach())) return;
            block_inputs input(f,1025,3,false,32);
            auto storage=span_storage::create(f,0,1025,3,{0,256,768,1025},true);
            if (!t.assert_true(bool(storage))) return;
            ggml_context_ptr ctx(ggml_init({65536,nullptr,true}));
            auto * node=f.resident->attention(ctx.get(),0,input.q,input.mask,input.active,1.0f);
            if (!t.assert_true(node != nullptr)) return;
            auto op=*node; op.buffer=input.output->buffer; op.data=input.output->data;
            size_t required=0;
            if (!t.assert_true(ops->spans_workspace(backend.get(),&op,storage->plan,required) && required > 1)) return;
            block_workspace short_workspace(f,required-1,79);
            auto * short_buffer=ggml_backend_memory_lease_buffer(short_workspace.lease.get());
            t.assert_equal(required-1,ggml_backend_buffer_get_size(short_buffer));
            const auto before=input.read();
            t.assert_true(ops->supports(backend.get(),&op));
            t.assert_true(!ops->partial(backend.get(),&op,short_buffer,false));
            t.assert_true(!ops->spans(backend.get(),&op,storage->plan,short_buffer));
            t.assert_true(same_float_bits(before,input.read()));
            auto invalid=op;
            invalid.ne[0]=256;
            size_t unchanged=77;
            t.assert_true(!ops->spans_workspace(backend.get(),&invalid,storage->plan,unchanged));
            t.assert_equal(size_t(77),unchanged);
            t.assert_true(same_float_bits(before,input.read()));
        });
        return t.summary();
    }
    if (argc > 1 && !std::strcmp(argv[1],"--cuda-gemma-compiled")) {
#ifdef KV_STREAM_PASCAL_CUDA_TEST
        testing t;
        t.test("compiled_wide_stock_family_instances_load_on_the_real_device", [](testing & t) {
            ggml_backend_load_all();
            auto * dev=ggml_backend_dev_by_name("CUDA0");
            if (!t.assert_true(dev != nullptr)) return;
            ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
            t.assert_true(kv_stream_test_runtime_shared_instances());
            const ggml_type formats[]={GGML_TYPE_F32,GGML_TYPE_F16,GGML_TYPE_BF16,GGML_TYPE_Q8_0,
                GGML_TYPE_Q5_0,GGML_TYPE_Q5_1,GGML_TYPE_Q4_0,GGML_TYPE_Q4_1};
            for (const auto key : formats) for (const auto value : formats)
                for (const auto geometry : {std::make_pair(1u,8u),std::make_pair(2u,8u),std::make_pair(4u,8u),
                        std::make_pair(8u,8u),std::make_pair(4u,2u),std::make_pair(8u,2u)})
                    for (bool resume : {false,true}) for (bool dense : {false,true})
                        t.assert_true(kv_stream_test_dimension_instance(backend.get(),key,value,512,
                            geometry.first,geometry.second,true,resume,dense));
            for (const auto pair : {std::make_pair(GGML_TYPE_Q8_0,GGML_TYPE_Q4_0),
                    std::make_pair(GGML_TYPE_Q8_0,GGML_TYPE_Q8_0),
                    std::make_pair(GGML_TYPE_Q5_1,GGML_TYPE_Q5_1),
                    std::make_pair(GGML_TYPE_F16,GGML_TYPE_F16)}) {
                for (uint32_t columns : {1u,2u,4u,8u}) for (bool resume : {false,true}) for (bool dense : {false,true})
                    t.assert_true(kv_stream_test_dimension_instance(backend.get(),pair.first,pair.second,512,columns,8,true,resume,dense));
                for (uint32_t columns : {2u,4u,8u})
                    t.assert_true(kv_stream_test_dimension_instance(backend.get(),pair.first,pair.second,512,columns,1,false,false,false));
            }
            t.assert_true(!kv_stream_test_dimension_instance(backend.get(),GGML_TYPE_Q8_0,GGML_TYPE_IQ4_NL,512,1,8,true,true,false));
            t.assert_true(!kv_stream_test_dimension_instance(backend.get(),GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,384,1,8,true,true,false));
        });
        return t.summary();
#else
        return 77;
#endif
    }
    if (argc > 1 && !std::strcmp(argv[1],"--cuda-gemma-geometry")) {
        testing t;
        t.test("dimension_aware_backend_prepares_full_attention_queries", [](testing & t) {
            ggml_backend_load_all();
            auto * dev=ggml_backend_dev_by_name("CUDA0");
            if (!t.assert_true(dev != nullptr)) return;
            ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
            auto get=reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(dev),"ggml_backend_kv_stream_partial_ops"));
            const auto * ops=get ? get() : nullptr;
            if (!t.assert_true(ops && ops->version >= 15 && ops->geometry)) return;
            const ggml_type formats[]={GGML_TYPE_F32,GGML_TYPE_F16,GGML_TYPE_BF16,GGML_TYPE_Q8_0,
                GGML_TYPE_Q5_0,GGML_TYPE_Q5_1,GGML_TYPE_Q4_0,GGML_TYPE_Q4_1};
            for (auto key : formats) for (auto value : formats) {
                const auto pair=std::make_pair(key,value);
                for (uint32_t width : {1u,2u,3u,4u,256u}) {
                    const ggml_kv_stream_attention_request request{{pair.first,pair.second,512,512,4,256,128},32,width,262144};
                    ggml_kv_stream_resume_plan plan;
                    t.assert_true(ggml_kv_stream_resume_plan_query(ops,backend.get(),request,plan));
                    t.assert_equal(uint32_t(512),plan.head_dim_k);
                    t.assert_equal(uint32_t(512),plan.head_dim_v);
                    size_t bytes=0;
                    t.assert_true(ggml_kv_stream_workspace_query(ops,backend.get(),request,
                        width <= 4 ? ggml_kv_stream_workspace_kind::decode : ggml_kv_stream_workspace_kind::prefill,bytes));
                    t.assert_true(bytes > 0);
                }
            }
            const ggml_kv_stream_attention_request valid{{GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,512,512,4,256,128},32,4,8192};
            for (int defect=0;defect<6;++defect) {
                auto request=valid;
                if (defect == 0) request.shape.head_dim_k=384;
                if (defect == 1) request.shape.head_dim_v=256;
                if (defect == 2) request.shape.type_v=GGML_TYPE_IQ4_NL;
                if (defect == 3) request.query_heads=31;
                if (defect == 4) request.shape.page_tokens=128;
                if (defect == 5) request.query_tokens=0;
                ggml_kv_stream_resume_plan untouched;
                untouched.bytes=77; untouched.head_dim_k=123;
                t.assert_true(!ggml_kv_stream_resume_plan_query(ops,backend.get(),request,untouched));
                t.assert_equal(size_t(77),untouched.bytes);
                t.assert_equal(uint32_t(123),untouched.head_dim_k);
                size_t bytes=77;
                t.assert_true(!ggml_kv_stream_workspace_query(ops,backend.get(),request,
                    ggml_kv_stream_workspace_kind::decode,bytes));
                t.assert_equal(size_t(77),bytes);
            }
        });
        return t.summary();
    }
    const int consumer=argc > 1 && !std::strcmp(argv[1],"--cuda-sm75") ? 750 :
        argc > 1 && !std::strcmp(argv[1],"--cuda-sm86") ? 860 :
        argc > 1 && !std::strcmp(argv[1],"--cuda-sm89") ? 890 :
        argc > 1 && !std::strcmp(argv[1],"--cuda-sm120") ? 1200 : 0;
    if (argc > 1 && !std::strncmp(argv[1],"--cuda-sm",9) && !consumer) return 2;
    ggml_backend_ptr consumer_backend;
#ifdef KV_STREAM_PASCAL_CUDA_TEST
    if (consumer) {
        if (!kv_stream_test_is_arch_only(consumer)) return 77;
        ggml_backend_load_all(); auto * dev=ggml_backend_dev_by_name("CUDA0");
        if (!dev) return 77;
        consumer_backend.reset(ggml_backend_dev_init(dev,nullptr));
        if (!consumer_backend || kv_stream_test_compiled_cc(consumer_backend.get()) != consumer) return 77;
    }
    const int previous=consumer ? kv_stream_test_override_cc(consumer_backend.get(),consumer) : 0;
    struct restore_arch {ggml_backend_t backend; int cc; ~restore_arch(){if (backend) kv_stream_test_override_cc(backend,cc);}} restore_consumer{consumer_backend.get(),previous};
#else
    if (consumer) return 77;
#endif
    const bool pascal = argc > 1 && (!std::strcmp(argv[1],"--cuda-pascal-tg1") || !std::strcmp(argv[1],"--cuda-pascal-tg2"));
    if (pascal) {
#ifdef KV_STREAM_PASCAL_CUDA_TEST
        if (!kv_stream_test_is_sm61_only()) {
            std::puts("SKIP: simulated Pascal vector attention requires an SM61-only build, not a mixed-target binary");
            return 77;
        }
#else
        std::puts("SKIP: CUDA test adapter is not compiled");
        return 77;
#endif
    }
    if (argc > 1 && !std::strcmp(argv[1], "--bench")) return benchmark_spans();
    if (argc > 1 && !std::strcmp(argv[1], "--bench-gemma")) {
        std::setvbuf(stdout,nullptr,_IOLBF,0);
        for (const auto & pair : {std::make_pair(GGML_TYPE_Q8_0,GGML_TYPE_Q4_0),std::make_pair(GGML_TYPE_Q8_0,GGML_TYPE_Q8_0),
                std::make_pair(GGML_TYPE_Q5_1,GGML_TYPE_Q5_1),std::make_pair(GGML_TYPE_F16,GGML_TYPE_F16)}) {
            if (benchmark_spans(true,pair.first,pair.second)) return 1;
        }
        return 0;
    }
    if (argc == 5 && !std::strcmp(argv[1], "--bench-gemma-focus")) {
        char * end=nullptr;
        const auto tokens=std::strtoul(argv[2],&end,10);
        if (*end || tokens < 256 || tokens > 262144 || tokens%256) return 2;
        const auto queries=std::strtoul(argv[3],&end,10);
        if (*end || !queries || queries > 256 || queries > tokens) return 2;
        const bool f16=!std::strcmp(argv[4],"f16-f16");
        if (!f16 && std::strcmp(argv[4],"q8-q4")) return 2;
        std::setvbuf(stdout,nullptr,_IOLBF,0);
        return benchmark_spans(true,f16 ? GGML_TYPE_F16 : GGML_TYPE_Q8_0,
            f16 ? GGML_TYPE_F16 : GGML_TYPE_Q4_0,size_t(tokens),uint32_t(queries));
    }
    testing t;
    t.test("dimensioned_inputs_preserve_independent_key_and_value_rows", [](testing & t) {
        ggml_backend_ptr backend(ggml_backend_cpu_init());
        fixture f(backend.get(),false,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,513,false,2,4,16,512,256);
        block_inputs input(f,513,4,false,32);
        t.assert_equal(int64_t(512),input.q->ne[0]);
        t.assert_equal(int64_t(256),input.output->ne[0]);
        t.assert_equal(size_t(4*32*512),input.qdata.size());
        t.assert_equal(size_t(4*32*256),input.read().size());
        t.assert_equal(size_t(544),f.host->layout().k_row_bytes);
        t.assert_equal(size_t(144),f.host->layout().v_row_bytes);
    });
    const bool quant_matrix = argc > 1 && !std::strcmp(argv[1], "--cuda-quant-matrix");
    const bool float32_only = argc > 1 && !std::strcmp(argv[1], "--cuda-f32");
    if (quant_matrix) t.set_filter("compiled_cache_pairs_execute_stock_order_spanned_attention");
    if (float32_only) t.set_filter("float32_storage_uses_bounded_stock_equivalent_attention|float32_ring_waves_preserve_native_conversion_and_resume_order");
    if (quant_matrix || (argc > 1 && !std::strcmp(argv[1], "--cuda"))) t.test(
            "compiled_cache_pairs_execute_stock_order_spanned_attention", [&](testing & t) {
        ggml_backend_load_all();
        auto * dev = ggml_backend_dev_by_name("CUDA0");
        if (!t.assert_true(dev != nullptr)) return;
        ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
        auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev), "ggml_backend_kv_stream_partial_ops"));
        const auto * ops = get ? get() : nullptr;
        if (!t.assert_true(ops && ops->capabilities && ops->spans_workspace && ops->spans)) return;
        const ggml_type types[] = {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q8_0,
            GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1};
        size_t pairs = 0;
        for (auto key : types) for (auto value : types) {
            if (!ops->capabilities(backend.get(), key, value).direct_pair) continue;
            fixture f(backend.get(), true, key, value, 1025, false, 2, 4, 16);
            if (!t.assert_true(f.attach())) return;
            auto pin = f.binding->acquire();
            if (!t.assert_true(bool(pin))) return;
            for (uint32_t rows = 1; rows <= 8; ++rows) {
                block_inputs input(f, 513, rows, false, 24);
                const auto expected = stock_attention(f, input, 0);
                auto spans = span_storage::create(f, 0, 513, rows, {0, 256, 512, 513}, true);
                ggml_context_ptr context(ggml_init({65536, nullptr, true}));
                auto * node = f.resident->attention(context.get(), 0, input.q, input.mask, input.active, 1.0f/16);
                if (!t.assert_true(spans && node)) return;
                ggml_tensor op = *node;
                op.buffer = input.output->buffer; op.data = input.output->data;
                size_t bytes = 0;
                if (!t.assert_true(ops->spans_workspace(backend.get(), &op, spans->plan, bytes))) return;
                block_workspace workspace(f, bytes, 371);
                if (!t.assert_true(ops->spans(backend.get(), &op, spans->plan,
                        ggml_backend_memory_lease_buffer(workspace.lease.get())))) return;
                ggml_backend_synchronize(backend.get());
                const auto actual = input.read();
                close_values(t, expected, actual, 1e-6f);
                if (rows >= 3) t.assert_true(same_float_bits(expected, actual));
            }
            ++pairs;
        }
        t.out << "qualified compiled cache pairs=" << pairs << '\n';
        if (quant_matrix) t.assert_equal(size_t(64), pairs);
        else t.assert_true(pairs > 0);
    });
    if (argc > 1 && (!std::strcmp(argv[1],"--cuda-prepared-decode") || !std::strcmp(argv[1],"--cuda")))
        t.test("decode_preflight_covers_shortened_token_major_mma_batches", [](testing & t) {
            ggml_backend_load_all();
            auto * device=ggml_backend_dev_by_name("CUDA0");
            if (!t.assert_true(device != nullptr)) return;
            ggml_backend_ptr backend(ggml_backend_dev_init(device,nullptr));
            if (!t.assert_true(bool(backend))) return;
            auto get=reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(device),"ggml_backend_kv_stream_partial_ops"));
            const auto * ops=get ? get() : nullptr;
            if (!t.assert_true(ops && ops->decode_workspace && ops->spans_workspace && ops->spans)) return;
            for (auto pair : {std::make_pair(GGML_TYPE_Q8_0,GGML_TYPE_Q4_0),std::make_pair(GGML_TYPE_F16,GGML_TYPE_F16)}) {
                size_t bytes=0;
                if (!t.assert_true(ops->decode_workspace(backend.get(),pair.first,pair.second,24,4,8,8192,bytes))) return;
                fixture f(backend.get(),true,pair.first,pair.second,8192,false,2,4,80);
                if (!t.assert_true(f.attach())) return;
                auto pin=f.binding->acquire();
                if (!t.assert_true(bool(pin))) return;
                block_workspace workspace(f,bytes,605);
                for (uint32_t queries : {3u,4u,5u,6u,7u,8u}) {
                    block_inputs input(f,8191,queries,false,24);
                    const auto expected=stock_attention(f,input,0);
                    for (bool wrapped : {false,true}) {
                        auto storage=span_storage::create(f,0,input.padded,queries,
                            wrapped ? std::vector<size_t>{0,256,768,input.padded} : std::vector<size_t>{0,input.padded},wrapped);
                        if (!t.assert_true(storage && storage->plan)) return;
                        ggml_context_ptr context(ggml_init({65536,nullptr,true}));
                        auto * node=f.resident->attention(context.get(),0,input.q,input.mask,input.active,1.0f/16);
                        if (!t.assert_true(node != nullptr)) return;
                        ggml_tensor op=*node;
                        op.buffer=input.output->buffer; op.data=input.output->data;
                        for (bool stock_conversion : {false,true}) {
                            ggml_backend_execution_set_stock_conversion(&op,stock_conversion);
                            size_t needed=0;
                            if (!t.assert_true(ops->spans_workspace(backend.get(),&op,storage->plan,needed))) return;
                            t.assert_true(needed <= bytes);
                            if (!t.assert_true(ops->spans(backend.get(),&op,storage->plan,
                                    ggml_backend_memory_lease_buffer(workspace.lease.get())))) return;
                            ggml_backend_synchronize(backend.get());
                            t.assert_true(same_float_bits(expected,input.read()));
                        }
                    }
                }
            }
        });
    if (argc > 1 && (!std::strcmp(argv[1], "--cuda-wide-spans") || !std::strcmp(argv[1], "--cuda")))
        t.test("wide_mma_dispatch_matches_stock_for_resident_and_wrapped_spans", [](testing & t) {
            ggml_backend_load_all();
            auto * dev = ggml_backend_dev_by_name("CUDA0");
            if (!t.assert_true(dev != nullptr)) return;
            ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
            auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(dev), "ggml_backend_kv_stream_partial_ops"));
            const auto * ops = get ? get() : nullptr;
            if (!t.assert_true(ops && ops->spans_workspace && ops->spans)) return;
            fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,8192,false,2,4,80);
            if (!t.assert_true(f.attach())) return;
            auto pin = f.binding->acquire();
            for (size_t active : {size_t(513),size_t(8191)}) for (uint32_t queries : {5u,6u,7u,8u})
                    for (uint32_t ratio : {2u,6u,8u}) {
                block_inputs input(f, active, queries, false, ratio*4);
                const auto expected = stock_attention(f, input, 0);
                for (bool wrapped : {false,true}) {
                    auto storage = span_storage::create(f,0,active,queries,
                        wrapped ? std::vector<size_t>{0,256,512,active} : std::vector<size_t>{0,active},wrapped);
                    if (!t.assert_true(storage && storage->plan)) return;
                    ggml_context_ptr context(ggml_init({65536,nullptr,true}));
                    auto * ks = ggml_new_tensor_2d(context.get(),GGML_TYPE_Q8_0,256*4,input.padded);
                    auto * vs = ggml_new_tensor_2d(context.get(),GGML_TYPE_Q4_0,256*4,input.padded);
                    auto * k = ggml_view_3d(context.get(),ks,256,input.padded,4,
                        ggml_row_size(ks->type,256*4),ggml_row_size(ks->type,256),0);
                    auto * v = ggml_view_3d(context.get(),vs,256,input.padded,4,
                        ggml_row_size(vs->type,256*4),ggml_row_size(vs->type,256),0);
                    auto * op = ggml_flash_attn_ext(context.get(),input.q,k,v,input.mask,1.0f/16,0,0);
                    op->buffer = input.output->buffer;
                    op->data = input.output->data;
                    size_t bytes = 0;
                    if (!t.assert_true(ops->spans_workspace(backend.get(),op,storage->plan,bytes))) return;
                    block_workspace exact(f,bytes,601), short_grant(f,bytes-1,602);
                    const auto sentinel = input.read();
                    t.assert_true(!ops->spans(backend.get(),op,storage->plan,ggml_backend_memory_lease_buffer(short_grant.lease.get())));
                    t.assert_true(same_float_bits(sentinel,input.read()));
                    if (!t.assert_true(ops->spans(backend.get(),op,storage->plan,ggml_backend_memory_lease_buffer(exact.lease.get())))) return;
                    ggml_backend_synchronize(backend.get());
                    t.assert_true(same_float_bits(expected,input.read()));
                }
            }
        });
    if (argc > 1 && (!std::strcmp(argv[1], "--cuda-wide-plan") || !std::strcmp(argv[1], "--cuda")))
        t.test("wide_query_plans_match_compiled_pairs_and_preserve_rejected_output", [](testing & t) {
            using status = ggml_cuda_kv_stream_plan_status;
            using family = ggml_cuda_kv_stream_kernel_family;
            using style = ggml_cuda_kv_stream_execution_style;
            using query_t = status (*)(ggml_backend_t, const ggml_tensor *, const ggml_kv_stream_span_plan_view *, style, ggml_cuda_kv_stream_attention_plan &);
            ggml_backend_load_all();
            auto * dev = ggml_backend_dev_by_name("CUDA0");
            if (!t.assert_true(dev != nullptr)) return;
            auto * reg = ggml_backend_dev_backend_reg(dev);
            auto query = reinterpret_cast<query_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_kv_stream_attention_plan"));
            auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_kv_stream_partial_ops"));
            ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
            if (!t.assert_true(backend && query && get && get() && get()->decode_workspace)) return;
            for (auto pair : {std::pair{GGML_TYPE_Q8_0, GGML_TYPE_Q4_0}, std::pair{GGML_TYPE_Q5_0, GGML_TYPE_Q4_0}}) for (uint32_t width = 5; width <= 9; ++width) {
                ggml_context_ptr ctx(ggml_init({65536, nullptr, true}));
                auto * q = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 256, width, 24);
                auto * ks = ggml_new_tensor_2d(ctx.get(), pair.first, 256*4, 768);
                auto * vs = ggml_new_tensor_2d(ctx.get(), pair.second, 256*4, 768);
                auto * k = ggml_view_3d(ctx.get(), ks, 256, 768, 4,
                    ggml_row_size(pair.first, 256*4), ggml_row_size(pair.first, 256), 0);
                auto * v = ggml_view_3d(ctx.get(), vs, 256, 768, 4,
                    ggml_row_size(pair.second, 256*4), ggml_row_size(pair.second, 256), 0);
                auto * mask = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, 768, width);
                auto * op = ggml_flash_attn_ext(ctx.get(), q, k, v, mask, 1.0f/16, 0, 0);
                ggml_kv_stream_span span{};
                span.tokens = 513;
                ggml_kv_stream_span_plan_view view{{pair.first, pair.second, 256, 256, 4, 256, 128}, &span, 1, 513, width};
                ggml_cuda_kv_stream_attention_plan native;
                t.assert_true(query(backend.get(), op, nullptr, style::native, native) == status::success);
                t.assert_true(native.family == family::mma);
                t.assert_equal(int64_t(width), native.metadata.queries);
                auto unchanged = native;
                const bool available=get()->capabilities(backend.get(),pair.first,pair.second).direct_pair;
                const auto expected=available ? status::success : status::missing_streamed_implementation;
                t.assert_true(query(backend.get(), op, &view, style::spanned, unchanged) == expected);
                if (expected != status::success) t.assert_equal(native.output_allocation_bytes, unchanged.output_allocation_bytes);
                const auto spanned = unchanged;
                t.assert_true(query(backend.get(), op, &view, style::resumable, unchanged) == expected);
                if (expected == status::success)
                    t.assert_true(unchanged.requirements.scratch_bytes >= spanned.requirements.scratch_bytes);
                const auto saved=unchanged;
                size_t bytes = 77;
                const bool decode_available=available && width <= 8;
                t.assert_true(get()->decode_workspace(backend.get(), pair.first, pair.second, 24, 4, width, 768, bytes) == decode_available);
                if (!decode_available) t.assert_equal(size_t(77), bytes);
                ++q->nb[1];
                t.assert_true(query(backend.get(), op, &view, style::spanned, unchanged) == status::unsupported_geometry);
                --q->nb[1];
                auto stale = view;
                ++stale.query_tokens;
                t.assert_true(query(backend.get(), op, &stale, style::spanned, unchanged) == status::invalid_metadata);
                t.assert_equal(saved.output_allocation_bytes, unchanged.output_allocation_bytes);
            }
        });
#ifdef KV_STREAM_PASCAL_CUDA_TEST
    if (consumer || (argc > 1 && !std::strcmp(argv[1],"--cuda-resource-probe"))) t.test(
            "mma_span_resources_are_rejected_before_launch_and_requirements_are_preserved",[&](testing & t) {
        ggml_backend_load_all();
        auto * dev=ggml_backend_dev_by_name("CUDA0");
        if (!t.assert_true(dev != nullptr)) return;
        ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
        using status=ggml_cuda_kv_stream_plan_status;
        using style=ggml_cuda_kv_stream_execution_style;
        using query_t=status (*)(ggml_backend_t,const ggml_tensor *,const ggml_kv_stream_span_plan_view *,style,ggml_cuda_kv_stream_attention_plan &);
        auto * reg=ggml_backend_dev_backend_reg(dev);
        auto query=reinterpret_cast<query_t>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_cuda_kv_stream_attention_plan"));
        auto get=reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_kv_stream_partial_ops"));
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,768,false,2,4,16);
        if (!t.assert_true(query && get && get() && f.attach())) return;
        auto pin=f.binding->acquire(); block_inputs input(f,513,4,false,24);
        auto storage=span_storage::create(f,0,513,4,{0,256,512,513},true);
        ggml_context_ptr context(ggml_init({65536,nullptr,true}));
        auto * node=f.resident->attention(context.get(),0,input.q,input.mask,input.active,1.0f/16);
        if (!t.assert_true(storage && node)) return;
        auto op=*node; op.buffer=input.output->buffer; op.data=input.output->data;
        ggml_kv_stream_span_plan_view view;
        if (!t.assert_true(ggml_kv_stream_span_plan_get_view(storage->plan,view))) return;
        ggml_cuda_kv_stream_attention_plan plan;
        if (!t.assert_true(query(backend.get(),&op,&view,style::spanned,plan) == status::success)) return;
        if (!t.assert_true(plan.family == ggml_cuda_kv_stream_kernel_family::mma &&
                plan.requirements.shared_bytes > 3*sizeof(ggml_cuda_kv_span))) return;
        const auto saved=plan;
        block_workspace workspace(f,plan.requirements.scratch_bytes);
        const size_t previous=kv_stream_test_shared_limit(backend.get(),plan.requirements.shared_bytes-3*sizeof(ggml_cuda_kv_span));
        struct restore {ggml_backend_t backend; size_t limit; ~restore(){kv_stream_test_shared_limit(backend,limit);}} restore_limit{backend.get(),previous};
        t.assert_true(query(backend.get(),&op,&view,style::spanned,plan) == status::unsupported_launch_resources);
        t.assert_equal(saved.requirements.scratch_bytes,plan.requirements.scratch_bytes);
        size_t untouched=77;
        t.assert_true(!get()->spans_workspace(backend.get(),&op,storage->plan,untouched));
        t.assert_equal(size_t(77),untouched);
        t.assert_true(!get()->spans(backend.get(),&op,storage->plan,ggml_backend_memory_lease_buffer(workspace.lease.get())));
        const auto output=input.read();
        t.assert_true(std::all_of(output.begin(),output.end(),[](float v){return v == -77;}));
        ggml_cuda_kv_stream_attention_plan native;
        t.assert_true(query(backend.get(),&op,nullptr,style::native,native) == status::success);
        t.assert_true(native.requirements.output_extra_bytes > 0);
        ggml_backend_execution_set_external_workspace(&op,true);
        t.assert_true(!get()->direct(backend.get(),&op));
    });
    if (consumer) t.test("consumer_rejects_unsupported_cache_formats_and_invalid_workspace_geometry",[&](testing & t) {
        auto * dev=ggml_backend_dev_by_name("CUDA0");
        ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
        auto get=reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev),"ggml_backend_kv_stream_partial_ops"));
        if (!t.assert_true(get && get() && get()->mma_workspace)) return;
        size_t untouched=77;
        for (auto pair : {std::pair{GGML_TYPE_IQ4_NL,GGML_TYPE_Q4_0},std::pair{GGML_TYPE_Q8_0,GGML_TYPE_IQ4_NL}}) {
            t.assert_true(!get()->mma_workspace(backend.get(),pair.first,pair.second,12,2,768,3,untouched));
            t.assert_equal(size_t(77),untouched);
        }
        t.assert_true(!get()->mma_workspace(backend.get(),GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,7,2,768,3,untouched));
        t.assert_true(!get()->mma_workspace(backend.get(),GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,12,2,0,3,untouched));
        t.assert_equal(size_t(77),untouched);
    });
    if (consumer) t.test("consumer_tg1_tg4_stock_family_scratch_and_span_equivalence",[&](testing & t) {
        auto * dev=ggml_backend_dev_by_name("CUDA0");
        ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
        using status=ggml_cuda_kv_stream_plan_status;
        using style=ggml_cuda_kv_stream_execution_style;
        using family=ggml_cuda_kv_stream_kernel_family;
        using query_t=status (*)(ggml_backend_t,const ggml_tensor *,const ggml_kv_stream_span_plan_view *,style,ggml_cuda_kv_stream_attention_plan &);
        auto * reg=ggml_backend_dev_backend_reg(dev);
        auto query=reinterpret_cast<query_t>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_cuda_kv_stream_attention_plan"));
        auto get=reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_kv_stream_partial_ops"));
        if (!t.assert_true(query && get && get())) return;
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,8192,false,2,2,80);
        if (!t.assert_true(f.attach())) return;
        auto pin=f.binding->acquire();
        for (size_t active : {size_t(513),size_t(768),size_t(8191),size_t(8192)})
            for (uint32_t ratio : {2u,6u,8u}) for (uint32_t queries : {1u,2u,3u,4u}) {
                size_t maximum=0;
                if (!t.assert_true(get()->decode_workspace(backend.get(),GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,ratio*2,2,4,8192,maximum))) return;
                block_inputs input(f,active,queries,false,ratio*2);
                const auto expected=stock_attention(f,input,0);
                for (bool wrapped : {false,true}) {
                    const std::vector<size_t> cuts=wrapped ? std::vector<size_t>{0,256,512,active} : std::vector<size_t>{0,active};
                    auto storage=span_storage::create(f,0,active,queries,cuts,wrapped);
                    if (!t.assert_true(storage != nullptr)) return;
                    ggml_context_ptr context(ggml_init({65536,nullptr,true}));
                    auto * node=f.resident->attention(context.get(),0,input.q,input.mask,active,1.0f/16);
                    if (!t.assert_true(node != nullptr)) return;
                    auto op=*node; op.buffer=input.output->buffer; op.data=input.output->data;
                    ggml_kv_stream_span_plan_view view;
                    if (!t.assert_true(ggml_kv_stream_span_plan_get_view(storage->plan,view))) return;
                    ggml_cuda_kv_stream_attention_plan plan;
                    if (!t.assert_true(query(backend.get(),&op,&view,style::spanned,plan) == status::success)) return;
                    const bool vector=queries == 1 || (consumer >= 890 && queries == 2);
                    t.assert_true(plan.family == (vector ? family::vector : family::mma));
                    size_t bytes=0;
                    if (!t.assert_true(get()->spans_workspace(backend.get(),&op,storage->plan,bytes))) return;
                    t.assert_equal(plan.requirements.scratch_bytes,bytes);
                    t.assert_true(bytes <= maximum);
                    block_workspace exact(f,bytes,91), tiny(f,bytes-1,92);
                    const auto before=input.read();
                    t.assert_true(!get()->spans(backend.get(),&op,storage->plan,ggml_backend_memory_lease_buffer(tiny.lease.get())));
                    t.assert_true(same_float_bits(before,input.read()));
                    if (!t.assert_true(get()->spans(backend.get(),&op,storage->plan,ggml_backend_memory_lease_buffer(exact.lease.get())))) return;
                    ggml_backend_synchronize(backend.get());
                    const auto actual=input.read();
                    t.out << "CC" << consumer << " GQA" << ratio << " TG" << queries << " active=" << active << " wrapped=" << wrapped << " exact=" << same_float_bits(expected,actual) << '\n';
                    // Regional vector reductions can differ on unaligned tails; keep the observed-error regression guard.
                    if (!vector || (queries == 2 && active%256 == 0) || (!wrapped && active%256 == 0))
                        t.assert_true(same_float_bits(expected,actual));
                    else close_values(t,expected,actual,1e-8f);
                }
            }
    });
    if (consumer) t.test("consumer_preserves_f16_mma_staging_and_masked_tails",[&](testing & t) {
        auto * dev=ggml_backend_dev_by_name("CUDA0");
        ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
        auto get=reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev),"ggml_backend_kv_stream_partial_ops"));
        fixture f(backend.get(),true,GGML_TYPE_F16,GGML_TYPE_F16,8192,false,2,2,80);
        if (!t.assert_true(get && get() && f.attach())) return;
        auto pin=f.binding->acquire();
        for (size_t active : {size_t(768),size_t(8191)}) for (uint32_t queries : {3u,4u}) {
            block_inputs input(f,active,queries,false,12);
            auto storage=span_storage::create(f,0,active,queries,{0,256,512,active},true);
            if (!t.assert_true(storage != nullptr)) return;
            bool accepted=false;
            const auto actual=evaluate(f,get(),*storage,input,0,accepted);
            if (!t.assert_true(accepted)) return;
            t.assert_true(same_float_bits(stock_attention(f,input,0),actual));
        }
    });
#endif
    t.test("ampere_decode_dispatch_matches_stock", [](testing & t) {
        using path = ggml_cuda_kv_stream_attention_path;
        t.assert_true(ggml_cuda_kv_stream_attention_select(860,1,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0) == path::vector);
        t.assert_true(ggml_cuda_kv_stream_attention_select(860,2,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0) == path::mma);
        t.assert_true(ggml_cuda_kv_stream_attention_select(890,1,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0) == path::vector);
        t.assert_true(ggml_cuda_kv_stream_attention_select(890,2,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0) == path::vector);
        t.assert_true(ggml_cuda_kv_stream_attention_select(860,2,GGML_TYPE_Q5_0,GGML_TYPE_Q4_0) == path::mma);
        t.assert_true(ggml_cuda_kv_stream_attention_select(860,2,GGML_TYPE_F16,GGML_TYPE_F16) == path::mma);
        t.assert_equal(4,ggml_cuda_kv_stream_mma_ncols1(2,2));
        t.assert_equal(2,ggml_cuda_kv_stream_mma_ncols1(2,6));
        t.assert_equal(2,ggml_cuda_kv_stream_mma_ncols1(2,8));
        t.assert_equal(4,ggml_cuda_kv_stream_mma_ncols1(3,8));
    });
    if (argc > 1 && !std::strcmp(argv[1], "--cuda-gqa6")) t.set_filter("qwen_ratio_six_tg3_tg4_matches_stock|qwen_24_4_tg2_aligned_and_wrapped_spans_are_exact");
    t.test("resume_layout_accounts_for_tg1_and_tg2", [](testing & t) {
        ggml_kv_stream_resume_plan one, two;
        if (!t.assert_true(ggml_kv_stream_resume_layout_make(24, 1, 3, 8, one))) return;
        if (!t.assert_true(ggml_kv_stream_resume_layout_make(24, 2, 3, 8, two))) return;
        t.assert_equal(size_t(1), size_t(one.queries));
        t.assert_equal(size_t(2), size_t(two.queries));
        t.assert_equal(2*one.state_bytes, two.state_bytes);
        t.assert_equal(2*(one.meta_offset - one.partial_offset), two.meta_offset - two.partial_offset);
        t.assert_equal(2*one.bytes, two.bytes);
    });

#ifdef KV_STREAM_PASCAL_CUDA_TEST
    const bool tg2 = argc > 1 && (!std::strcmp(argv[1],"--cuda-pascal-tg2") || !std::strcmp(argv[1],"--cuda-tg2"));
    if (pascal || tg2 || (argc > 1 && !std::strcmp(argv[1],"--cuda"))) t.test("quantized_vector_stock_equivalence_across_spans_and_ring_waves", [&](testing & t) {
        ggml_backend_load_all();
        auto * dev = ggml_backend_dev_by_name("CUDA0");
        if (!t.assert_true(dev != nullptr)) return;
        ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
        pascal_test_guard guard(backend.get(),pascal);
        if (pascal) t.out << "SM61 compiled code and simulated host CC610; physical device: " << ggml_backend_dev_description(dev) << '\n';
        auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev),"ggml_backend_kv_stream_partial_ops"));
        const auto * ops = get ? get() : nullptr;
        if (!t.assert_true(ops && ops->resume_plan && ops->spans_workspace && ops->spans)) return;
        const uint32_t queries = tg2 ? 2 : 1;
        for (int type = 0; type < GGML_TYPE_COUNT; ++type) {
            if (ggml_blck_size(ggml_type(type)) > 0) continue;
            ggml_kv_stream_resume_plan untouched;
            untouched.bytes = 77;
            t.assert_true(!ops->resume_plan(backend.get(),type,GGML_TYPE_Q4_0,24,4,1,256,untouched));
            t.assert_true(!ops->resume_plan(backend.get(),GGML_TYPE_Q8_0,type,24,4,1,256,untouched));
            t.assert_equal(size_t(77),untouched.bytes);
        }
        for (auto pair : {std::pair{GGML_TYPE_Q8_0,GGML_TYPE_Q4_0}, std::pair{GGML_TYPE_Q5_1,GGML_TYPE_Q4_1},
                std::pair{GGML_TYPE_Q4_0,GGML_TYPE_F16}, std::pair{GGML_TYPE_Q4_0,GGML_TYPE_BF16}}) {
            fixture f(backend.get(),true,pair.first,pair.second,4097,false,2,4,3);
            const auto capabilities = ops->capabilities(backend.get(),pair.first,pair.second);
            if (!capabilities.direct_pair) {
                ggml_kv_stream_resume_plan unavailable;
                unavailable.bytes = 77;
                t.assert_true(!ops->resume_plan(backend.get(),pair.first,pair.second,24,4,queries,256,unavailable));
                t.assert_equal(size_t(77),unavailable.bytes);
                t.out << "pair not compiled: " << ggml_type_name(pair.first) << '/' << ggml_type_name(pair.second) << '\n';
                continue;
            }
            t.assert_true(capabilities.direct_pair && capabilities.k.direct_attention && capabilities.v.direct_attention);
            f.policy.initial_ring_slots = 1;
            if (!t.assert_true(f.attach() && f.resident->configure_native_graph_attention(true) &&
                    f.resident->configure_resumed_decode(true))) return;
            auto pin = f.binding->acquire();
            fixture control(backend.get(),true,pair.first,pair.second,4097,false,2,4,35);
            control.policy.initial_ring_slots = 1;
            if (!t.assert_true(control.attach())) return;
            auto control_pin = control.binding->acquire();
            for (size_t active : {size_t(33),size_t(255),size_t(256),size_t(257),size_t(513),size_t(1024),size_t(1025),size_t(4097)}) {
                t.out << ggml_type_name(pair.first) << '/' << ggml_type_name(pair.second) << " active=" << active << '\n';
                block_inputs input(f,active,queries,false,24);
                std::vector<ggml_fp16_t> mask(input.padded*queries);
                for (size_t q = 0; q < queries; ++q) for (size_t i = 0; i < input.padded; ++i)
                    mask[q*input.padded+i] = ggml_fp32_to_fp16(i <= active-queries+q && i%7 != 3 ? 0 : -INFINITY);
                ggml_backend_tensor_set(input.mask,mask.data(),0,mask.size()*sizeof(ggml_fp16_t));
                ggml_kv_stream_resume_plan plan;
                if (!t.assert_true("vector resume plan",ops->resume_plan(backend.get(),pair.first,pair.second,24,4,queries,input.padded,plan))) return;
                if (queries == 2) {
                    ggml_kv_stream_resume_plan one;
                    t.assert_true(ggml_kv_stream_resume_layout_make(24,1,plan.splits,plan.values_per_thread,one));
                    t.assert_equal(2*one.bytes,plan.bytes);
                    t.assert_equal(2*one.state_bytes,plan.state_bytes);
                }
                const bool half_value = pair.second == GGML_TYPE_F16 || pair.second == GGML_TYPE_BF16;
                t.assert_equal(uint32_t(half_value ? (kv_stream_test_compiled_cc(backend.get()) < 700 ? 16 : 32) : 8),plan.values_per_thread);
                ggml_kv_stream_resume_plan checked;
                t.assert_true(ggml_kv_stream_resume_layout_make(24,queries,plan.splits,plan.values_per_thread,checked));
                t.assert_equal(checked.bytes,plan.bytes);
                block_workspace workspace(f,plan.bytes), tiny(f,plan.bytes-1,89);
                auto * scratch_parent = ggml_backend_memory_arena_parent(workspace.arena.get());
                ggml_backend_buffer_clear(scratch_parent,0x5a);
                if (!t.assert_true("resident prefix synchronization",f.resident->synchronize(std::min(active,size_t(256))))) return;
                if (!t.assert_true("initial sequence admission",f.resident->begin_sequence({0,1},active,1,SIZE_MAX,{queries,true}))) return;
                if (active > 256) {
                    t.assert_true("one-byte-short scratch rejected",!f.resident->compute_streamed(0,input.q,input.mask,input.output,active,1.0f/16,tiny.lease.get(),true,1));
                    const auto untouched = input.read();
                    t.assert_true(std::all_of(untouched.begin(),untouched.end(),[](float x) { return x == -77; }));
                    if (!t.assert_true("retry sequence after scratch rejection",f.resident->begin_sequence({0,1},active,1,SIZE_MAX,{queries,true}))) return;
                }
                for (uint32_t layer = 0; layer < 2; ++layer) {
                    const auto expected = stock_attention(f,input,layer);
                    if (!t.assert_true("exact-sized resumed layer",f.resident->compute_streamed(layer,input.q,input.mask,input.output,active,1.0f/16,workspace.lease.get(),true,1))) return;
                    ggml_backend_synchronize(backend.get());
                    const auto actual = input.read();
                    close_values(t,expected,actual,1e-6f);
                    if (queries == 2) t.assert_true(!std::equal(actual.begin(),actual.begin()+actual.size()/2,actual.begin()+actual.size()/2));
                    if (active%256 == 0) t.assert_true(same_float_bits(expected,actual));
                    if (active > 512) t.assert_true(f.resident->last_attention_calls() > 1);
                }
                t.assert_true(!f.resident->sequence_active());
                ggml_backend_memory_region scratch_region;
                t.assert_true(ggml_backend_memory_lease_get_region(workspace.lease.get(),&scratch_region));
                ggml_tensor guard_bytes = {};
                guard_bytes.type = GGML_TYPE_I8;
                guard_bytes.buffer = scratch_parent;
                guard_bytes.data = ggml_backend_buffer_get_base(scratch_parent);
                guard_bytes.ne[0] = ggml_backend_buffer_get_size(scratch_parent);
                guard_bytes.nb[0] = 1;
                for (int i = 1; i < 4; ++i) { guard_bytes.ne[i] = 1; guard_bytes.nb[i] = guard_bytes.ne[0]; }
                std::vector<uint8_t> guards(size_t(guard_bytes.ne[0]));
                ggml_backend_tensor_get(&guard_bytes,guards.data(),0,guards.size());
                t.assert_true(std::all_of(guards.begin(),guards.begin()+scratch_region.offset,[](uint8_t x) { return x == 0x5a; }));
                t.assert_true(std::all_of(guards.begin()+scratch_region.offset+scratch_region.size,guards.end(),[](uint8_t x) { return x == 0x5a; }));
                const std::vector<size_t> cuts = active > 512 ? std::vector<size_t>{0,256,512,active} : std::vector<size_t>{0,active};
                if (!t.assert_true(control.resident->synchronize(active))) return;
                auto storage = span_storage::create(control,0,active,queries,cuts,true);
                if (!t.assert_true(storage != nullptr)) return;
                if (queries == 2) {
                    auto stale = span_storage::create(control,0,active,1,cuts,true);
                    ggml_context_ptr ctx(ggml_init({65536,nullptr,true}));
                    auto * node = control.resident->attention(ctx.get(),0,input.q,input.mask,input.active,1.0f/16);
                    if (!t.assert_true(stale && node)) return;
                    auto op = *node; op.buffer = input.output->buffer; op.data = input.output->data;
                    size_t untouched = 77;
                    t.assert_true(!ops->spans_workspace(backend.get(),&op,stale->plan,untouched));
                    t.assert_equal(size_t(77),untouched);
                    const auto before = input.read();
                    t.assert_true(!ops->spans(backend.get(),&op,stale->plan,ggml_backend_memory_lease_buffer(workspace.lease.get())));
                    t.assert_true(same_float_bits(before,input.read()));
                    ggml_kv_stream_resume_plan wrong_width;
                    t.assert_true(ggml_kv_stream_resume_layout_make(24,1,plan.splits,plan.values_per_thread,wrong_width));
                    wrong_width.tokens = plan.tokens;
                    t.assert_true(!ops->resume(backend.get(),&op,ggml_backend_memory_lease_buffer(workspace.lease.get()),wrong_width,plan.tokens,0,true));
                    t.assert_true(same_float_bits(before,input.read()));
                }
                bool accepted = false;
                const auto expected = stock_attention(f,input,0);
                const auto actual = evaluate(control,ops,*storage,input,0,accepted);
                if (!t.assert_true(accepted)) return;
                close_values(t,expected,actual,1e-6f);
            }
        }
    });
    if (tg2) t.test("two_query_tail_visibility_and_rejection_replay_preserve_prefix", [&](testing & t) {
        auto * dev = ggml_backend_dev_by_name("CUDA0");
        if (!t.assert_true(dev != nullptr)) return;
        ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
        pascal_test_guard guard(backend.get(),pascal);
        auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev),"ggml_backend_kv_stream_partial_ops"));
        const auto * ops = get ? get() : nullptr;
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,513,false,2,4,3);
        f.policy.initial_ring_slots = 1;
        if (!t.assert_true(ops && f.attach() && f.resident->configure_native_graph_attention(true) && f.resident->configure_resumed_decode(true))) return;
        auto pin = f.binding->acquire();
        ggml_kv_stream_resume_plan plan;
        if (!t.assert_true(ops->resume_plan(backend.get(),GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,24,4,2,768,plan))) return;
        block_workspace workspace(f,plan.bytes);
        const auto run = [&](size_t active, uint32_t queries) {
            block_inputs input(f,active,queries,false,24);
            std::vector<float> first;
            if (!t.assert_true(f.resident->begin_sequence({0,1},active,1,SIZE_MAX,{queries,true}))) return first;
            for (uint32_t layer = 0; layer < 2; ++layer) {
                const auto expected = stock_attention(f,input,layer);
                if (!t.assert_true(f.resident->compute_streamed(layer,input.q,input.mask,input.output,active,1.0f/16,workspace.lease.get(),true,1))) return std::vector<float>{};
                ggml_backend_synchronize(backend.get());
                const auto actual = input.read();
                close_values(t,expected,actual,1e-6f);
                if (!layer) first = actual;
            }
            t.assert_true(!f.resident->sequence_active());
            return first;
        };
        const auto baseline = run(257,2);
        if (!t.assert_equal(size_t(2*24*256),baseline.size())) return;
        llama_kv_stream_host_layer host;
        if (!t.assert_true(f.host->layer(0,host))) return;
        const auto & layout = f.host->layout();
        auto * k = static_cast<uint8_t *>(host.k)+256*layout.k_token_bytes;
        auto * v = static_cast<uint8_t *>(host.v)+256*layout.v_token_bytes;
        std::vector<uint8_t> saved_k(k,k+layout.k_token_bytes), saved_v(v,v+layout.v_token_bytes);
        std::vector<float> changed(4*256,1.0f);
        ggml_quantize_chunk(GGML_TYPE_Q8_0,changed.data(),k,0,4,256,nullptr);
        ggml_quantize_chunk(GGML_TYPE_Q4_0,changed.data(),v,0,4,256,nullptr);
        if (!t.assert_true(f.content->invalidate_suffix(256))) return;
        const auto modified = run(257,2);
        if (!t.assert_equal(baseline.size(),modified.size())) return;
        t.assert_true(std::equal(baseline.begin(),baseline.begin()+baseline.size()/2,modified.begin()));
        t.assert_true(!std::equal(baseline.begin()+baseline.size()/2,baseline.end(),modified.begin()+modified.size()/2));
        std::memcpy(k,saved_k.data(),saved_k.size()); std::memcpy(v,saved_v.data(),saved_v.size());
        if (!t.assert_true(f.content->invalidate_suffix(256))) return;
        t.assert_equal(size_t(24*256),run(256,1).size());
        t.assert_true(same_float_bits(baseline,run(257,2)));
    });
    if (pascal) t.test("tg1_only_session_preserves_decode_workspace_during_phase_handoff", [&](testing & t) {
        auto * dev = ggml_backend_dev_by_name("CUDA0");
        if (!t.assert_true(dev != nullptr)) return;
        ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
        pascal_test_guard guard(backend.get(),true);
        auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev),"ggml_backend_kv_stream_partial_ops"));
        const auto * ops = get ? get() : nullptr;
        if (!t.assert_true(ops != nullptr)) return;
        tg1_only_test_ops limited(ggml_backend_dev_backend_reg(dev),*ops);
        ops = &limited.limited;
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,513,false,2,4,3);
        ggml_kv_stream_resume_plan plan, untouched;
        if (!t.assert_true(ops && ops->resume_plan(backend.get(),GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,24,4,1,f.host->layout().tokens,plan))) return;
        untouched.bytes = 77;
        t.assert_true(!ops->resume_plan(backend.get(),GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,24,4,2,f.host->layout().tokens,untouched));
        t.assert_equal(size_t(77),untouched.bytes);
        auto model = llama_kv_stream_model::create({backend.get(),f.host->config(),f.policy.pool_bytes,256,24});
        if (!t.assert_true(model != nullptr)) return;
        llama_kv_stream_memory_requirements requirements;
        if (!t.assert_true(model->memory_requirements(requirements))) return;
        t.assert_equal(plan.bytes,requirements.attention_decode_bytes);
        t.assert_true(requirements.attention_decode_bytes < requirements.attention_prefill_bytes);
        if (!t.assert_true(model->begin(1,1,true))) return;
        t.assert_equal(plan.bytes,model->attention_grant_bytes());
        model->abort();
        t.assert_true(model->reset(false));
        t.assert_true(model->begin(1,1,false));
        model->abort();
        t.assert_true(model->reset(false));
        t.assert_true(model->begin(1,1,true));
        model->abort();
    });
    if (pascal && tg2) t.test("two_query_session_handoff_reserves_the_complete_decode_plan", [&](testing & t) {
        auto * dev = ggml_backend_dev_by_name("CUDA0");
        if (!t.assert_true(dev != nullptr)) return;
        ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
        pascal_test_guard guard(backend.get(),true);
        auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev),"ggml_backend_kv_stream_partial_ops"));
        const auto * ops = get ? get() : nullptr;
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,513,false,2,4,3);
        ggml_kv_stream_resume_plan plan;
        if (!t.assert_true(ops && ops->resume_plan(backend.get(),GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,24,4,2,f.host->layout().tokens,plan))) return;
        size_t required=plan.bytes;
        if (ops->version >= 10) {
            if (!t.assert_true(ops->decode_workspace && ops->decode_workspace(backend.get(),GGML_TYPE_Q8_0,
                    GGML_TYPE_Q4_0,24,4,4,f.host->layout().tokens,required))) return;
            t.assert_true(required >= plan.bytes);
        }
        auto model = llama_kv_stream_model::create({backend.get(),f.host->config(),f.policy.pool_bytes,256,24});
        if (!t.assert_true(model != nullptr)) return;
        llama_kv_stream_memory_requirements requirements;
        if (!t.assert_true(model->memory_requirements(requirements))) return;
        t.assert_equal(required,requirements.attention_decode_bytes);
        if (!t.assert_true(model->begin(2,2,true))) return;
        t.assert_equal(required,model->attention_grant_bytes());
        model->abort();
        t.assert_true(model->reset(false));
        t.assert_true(model->begin(1,1,true));
        t.assert_equal(required,model->attention_grant_bytes());
        model->abort();
        t.assert_true(model->reset(false));
        t.assert_true(model->begin(2,2,false));
        t.assert_equal(requirements.attention_prefill_bytes,model->attention_grant_bytes());
        model->abort();
    });
#endif

    if (argc > 1 && !std::strcmp(argv[1], "--cuda")) t.test(
            "native_span_vector_contract_is_available", [](testing & t) {
        ggml_backend_load_all();
        auto * dev = ggml_backend_dev_by_name("CUDA0");
        if (!t.assert_true(dev != nullptr)) return;
        auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(
            ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(dev), "ggml_backend_kv_stream_partial_ops"));
        t.assert_true(get && get() && get()->version >= 8 && get()->spans && get()->spans_workspace && get()->convert_mma_rows);
    });

    if (float32_only || (argc > 1 && !std::strcmp(argv[1], "--cuda"))) t.test("float32_storage_uses_bounded_stock_equivalent_attention", [](testing & t) {
        auto * dev = ggml_backend_dev_by_name("CUDA0");
        if (!t.assert_true(dev != nullptr)) return;
        ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
        auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev), "ggml_backend_kv_stream_partial_ops"));
        const auto * ops = get ? get() : nullptr;
        if (!t.assert_true(ops && ops->capabilities && ops->decode_workspace)) return;
        for (auto pair : {std::pair{GGML_TYPE_F32, GGML_TYPE_F32}, std::pair{GGML_TYPE_F32, GGML_TYPE_Q8_0},
                std::pair{GGML_TYPE_Q5_1, GGML_TYPE_F32}}) for (uint32_t query_heads : {8u, 24u}) {
            const auto caps = ops->capabilities(backend.get(), pair.first, pair.second);
            t.assert_true(caps.direct_pair && caps.k.online_write && caps.v.online_write);
            for (uint32_t rows = 1; rows <= 8; ++rows) {
                size_t bytes = 0;
                t.assert_true(ops->decode_workspace(backend.get(), pair.first, pair.second, query_heads, 4, rows, 262144, bytes));
                t.assert_true(bytes > 0 && bytes < 32*1024*1024);
            }
            fixture f(backend.get(), true, pair.first, pair.second, 1025, false, 2, 4, 16);
            if (!t.assert_true(f.attach())) return;
            auto pin = f.binding->acquire();
            for (uint32_t rows = 1; rows <= 8; ++rows) {
                block_inputs input(f, 513, rows, false, query_heads);
                auto storage = span_storage::create(f, 0, 513, rows, {0, 256, 512, 513}, true);
                ggml_context_ptr context(ggml_init({65536, nullptr, true}));
                auto * node = f.resident->attention(context.get(), 0, input.q, input.mask, input.active, 1.0f/16);
                if (!t.assert_true(pin && storage && node)) return;
                ggml_tensor op = *node;
                op.buffer = input.output->buffer;
                op.data = input.output->data;
                size_t bytes = 0;
                if (!t.assert_true(ops->spans_workspace(backend.get(), &op, storage->plan, bytes))) return;
                block_workspace workspace(f, bytes, 351);
                if (!t.assert_true(ops->spans(backend.get(), &op, storage->plan,
                        ggml_backend_memory_lease_buffer(workspace.lease.get())))) return;
                ggml_backend_synchronize(backend.get());
                const auto expected = stock_attention(f, input, 0);
                const auto actual = input.read();
                t.assert_true(expected.size() == actual.size() &&
                    std::memcmp(expected.data(), actual.data(), expected.size()*sizeof(float)) == 0);
            }
        }
    });

    if (float32_only || (argc > 1 && !std::strcmp(argv[1], "--cuda"))) t.test("float32_ring_waves_preserve_native_conversion_and_resume_order", [](testing & t) {
        auto * dev = ggml_backend_dev_by_name("CUDA0");
        if (!t.assert_true(dev != nullptr)) return;
        ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
        auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev), "ggml_backend_kv_stream_partial_ops"));
        const auto * ops = get ? get() : nullptr;
        if (!t.assert_true(ops && ops->decode_workspace)) return;
        for (auto pair : {std::pair{GGML_TYPE_F32, GGML_TYPE_F32}, std::pair{GGML_TYPE_F32, GGML_TYPE_Q8_0},
                std::pair{GGML_TYPE_Q5_1, GGML_TYPE_F32}}) for (uint32_t query_heads : {8u, 24u}) {
            fixture f(backend.get(), true, pair.first, pair.second, 2049, false, 2, 4, 7);
            if (!t.assert_true(f.attach() && f.resident->configure_native_graph_attention(true) &&
                    f.resident->configure_resumed_decode(true))) return;
            auto pin = f.binding->acquire();
            if (!t.assert_true(bool(pin))) return;
            for (uint32_t rows = 1; rows <= 8; ++rows) {
                block_inputs input(f, 2049, rows, false, query_heads);
                size_t bytes = 0;
                if (!t.assert_true(ops->decode_workspace(backend.get(), pair.first, pair.second, query_heads, 4, rows, input.padded, bytes))) return;
                block_workspace workspace(f, bytes, 361);
                if (!t.assert_true(f.resident->synchronize(256) &&
                        f.resident->begin_sequence({0,1}, input.active, 1, SIZE_MAX, {rows, true}))) return;
                for (uint32_t layer = 0; layer < 2; ++layer) {
                    const auto expected = stock_attention(f, input, layer);
                    if (!t.assert_true(f.resident->compute_streamed(layer, input.q, input.mask, input.output,
                            input.active, 1.0f/16, workspace.lease.get(), true, 1))) return;
                    ggml_backend_synchronize(backend.get());
                    const auto actual = input.read();
                    close_values(t, expected, actual, 1e-6f);
                    if (rows >= 3 || pair.first == pair.second) t.assert_true(same_float_bits(expected, actual));
                    t.assert_true(f.resident->last_attention_calls() > 1);
                }
                t.assert_true(!f.resident->sequence_active());
            }
        }
    });

    if (argc > 1 && !std::strcmp(argv[1], "--cuda")) t.test("stock_selected_plan_matches_workspace_and_launch", [](testing & t) {
        using status = ggml_cuda_kv_stream_plan_status;
        using family = ggml_cuda_kv_stream_kernel_family;
        using style = ggml_cuda_kv_stream_execution_style;
        using query_t = status (*)(ggml_backend_t, const ggml_tensor *, const ggml_kv_stream_span_plan_view *, style, ggml_cuda_kv_stream_attention_plan &);
        auto * dev = ggml_backend_dev_by_name("CUDA0");
        auto * reg = ggml_backend_dev_backend_reg(dev);
        auto query = reinterpret_cast<query_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_kv_stream_attention_plan"));
        if (!t.assert_true(query != nullptr)) return;
        ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
        auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_kv_stream_partial_ops"));
        const auto * ops = get ? get() : nullptr;
        fixture f(backend.get(), true, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 1025, false, 2, 4, 16);
        if (!t.assert_true(ops && f.attach())) return;
        auto pin = f.binding->acquire();
        for (uint32_t queries : {1u, 2u, 3u, 4u}) {
            block_inputs input(f, 513, queries, false, 24);
            auto storage = span_storage::create(f, 0, 513, queries, {0, 256, 512, 513}, true);
            if (!t.assert_true(pin && storage && storage->plan)) return;
            ggml_context_ptr ctx(ggml_init({65536, nullptr, true}));
            auto * node = f.resident->attention(ctx.get(), 0, input.q, input.mask, input.active, 1.0f/16);
            if (!t.assert_true(node != nullptr)) return;
            ggml_tensor op = *node;
            op.buffer = input.output->buffer; op.data = input.output->data;
            ggml_kv_stream_span_plan_view view;
            if (!t.assert_true(ggml_kv_stream_span_plan_get_view(storage->plan, view))) return;
            ggml_cuda_kv_stream_attention_plan plan;
            if (!t.assert_true(query(backend.get(), &op, &view, style::spanned, plan) == status::success)) return;
            t.assert_true(plan.family == (queries <= 2 ? family::vector : family::mma));
            size_t bytes = 0;
            t.assert_true(ops->spans_workspace(backend.get(), &op, storage->plan, bytes));
            t.assert_equal(bytes, plan.requirements.scratch_bytes);
            t.assert_equal(ggml_nbytes(&op), plan.output_allocation_bytes);
            t.assert_true(plan.requirements.backend_scratch_known && plan.requirements.shared_bytes_known);
            ggml_cuda_kv_stream_attention_plan native;
            t.assert_true(query(backend.get(), &op, nullptr, style::native, native) == status::success);
            t.assert_true(native.family == plan.family);
            t.assert_true(!native.requirements.backend_scratch_known && !native.requirements.shared_bytes_known);
            t.assert_equal(ggml_backend_buffer_get_alloc_size(op.buffer, &op), native.output_allocation_bytes);
            if (queries >= 3) t.assert_true(native.output_allocation_bytes > ggml_nbytes(&op));
            ggml_tensor metadata_only = op, q = *op.src[0], k = *op.src[1], v = *op.src[2], mask = *op.src[3];
            metadata_only.buffer = q.buffer = k.buffer = v.buffer = mask.buffer = nullptr;
            metadata_only.data = q.data = k.data = v.data = mask.data = nullptr;
            metadata_only.src[0] = &q; metadata_only.src[1] = &k; metadata_only.src[2] = &v; metadata_only.src[3] = &mask;
            ggml_cuda_kv_stream_attention_plan described;
            t.assert_true(query(backend.get(), &metadata_only, &view, style::spanned, described) == status::success);
            t.assert_equal(bytes, described.requirements.scratch_bytes);
            ++k.nb[1];
            t.assert_true(query(backend.get(), &metadata_only, &view, style::spanned, described) == status::unsupported_geometry);
            --k.nb[1];
            t.assert_equal(bytes, described.requirements.scratch_bytes);
            auto stale = view; ++stale.query_tokens;
            t.assert_true(query(backend.get(), &op, &stale, style::spanned, described) == status::invalid_metadata);
            t.assert_equal(bytes, described.requirements.scratch_bytes);
            block_workspace exact(f, bytes, 301), short_grant(f, bytes-1, 302);
            t.assert_true(!ops->spans(backend.get(), &op, storage->plan, ggml_backend_memory_lease_buffer(short_grant.lease.get())));
            const auto sentinel = input.read();
            t.assert_true(std::all_of(sentinel.begin(), sentinel.end(), [](float value) { return value == -77; }));
            t.assert_true(ops->spans(backend.get(), &op, storage->plan, ggml_backend_memory_lease_buffer(exact.lease.get())));
            ggml_backend_synchronize(backend.get());
            const auto expected = stock_attention(f, input, 0);
            close_values(t, expected, input.read(), 1e-5f);
        }
    });

    if (argc > 1 && !std::strcmp(argv[1], "--cuda")) t.test("native_fallback_cannot_use_a_bounded_output_declaration", [](testing & t) {
        auto * dev = ggml_backend_dev_by_name("CUDA0");
        ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
        auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev), "ggml_backend_kv_stream_partial_ops"));
        const auto * ops = get ? get() : nullptr;
        fixture f(backend.get(), true, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 1024);
        block_inputs input(f, 513, 4);
        stock_attention_case stock(f, input);
        ggml_tensor op = *ggml_graph_node(stock.graph, ggml_graph_n_nodes(stock.graph)-1);
        ggml_backend_buffer_ptr parent(ggml_backend_alloc_buffer(backend.get(), 4*1024*1024));
        if (!t.assert_true(ops && parent)) return;
        ggml_backend_buffer_clear(parent.get(), 0x5a);
        op.buffer = parent.get(); op.data = ggml_backend_buffer_get_base(parent.get());
        ggml_backend_execution_set_external_workspace(&op, true);
        t.assert_true(!ops->direct(backend.get(), &op));
        ggml_tensor canary = op;
        canary.type = GGML_TYPE_I8; canary.op = GGML_OP_NONE;
        canary.data = static_cast<char *>(op.data) + ggml_nbytes(&op) + 128;
        canary.ne[0] = 128; canary.nb[0] = 1;
        for (int i = 1; i < 4; ++i) { canary.ne[i] = 1; canary.nb[i] = 128; }
        uint8_t bytes[128]; ggml_backend_tensor_get(&canary, bytes, 0, sizeof(bytes));
        for (uint8_t value : bytes) t.assert_equal(uint8_t(0x5a), value);
    });

    if (argc > 1 && !std::strcmp(argv[1], "--cuda")) t.test(
            "tg1_tg2_partition_boundaries_match_contiguous_and_stock", [](testing & t) {
        auto * dev = ggml_backend_dev_by_name("CUDA0");
        ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
        auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(
            ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(dev), "ggml_backend_kv_stream_partial_ops"));
        const auto * ops = get ? get() : nullptr;
        if (!t.assert_true(ops && ops->version >= 7 && ops->spans && ops->spans_workspace)) return;
        fixture f(backend.get(), true, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 1280);
        f.policy.initial_ring_slots = 4;
        if (!t.assert_true(f.attach())) return;
        auto pin = f.binding->acquire();
        if (!t.assert_true(bool(pin))) return;
        for (size_t active : {size_t(257), size_t(513), size_t(1017), size_t(1280)}) {
            for (uint32_t queries : {1u, 2u}) {
                std::vector<size_t> cuts{0, 1, 127, 128, 255, 256, 257, active - 1, active};
                cuts.erase(std::remove_if(cuts.begin(), cuts.end(),
                    [&](size_t value) { return value > active; }), cuts.end());
                std::sort(cuts.begin(), cuts.end());
                cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
                auto contiguous = span_storage::create(f, 0, active, queries, {0, active});
                auto segmented = span_storage::create(f, 0, active, queries, cuts);
                if (!t.assert_true(contiguous && segmented && contiguous->plan && segmented->plan)) continue;
                block_inputs input(f, active, queries);
                const auto expected = ordinary(f, input, 0);
                bool accepted = false;
                const auto a = evaluate(f, ops, *contiguous, input, 0, accepted);
                if (!t.assert_true(accepted)) continue;
                if (queries == 1) {
                    const std::vector<std::vector<size_t>> production_cuts{
                        {0,active/2,active},{0,active/3,2*active/3,active}};
                    for (const auto & physical_cuts : production_cuts) {
                        auto fixed = span_storage::create(f,0,active,queries,physical_cuts);
                        if (!t.assert_true(fixed && fixed->plan)) continue;
                        const auto value = evaluate(f,ops,*fixed,input,0,accepted);
                        if (!t.assert_true(accepted)) continue;
                        close_values(t,a,value,1e-7f);
                        close_values(t,expected,value,1e-5f);
                    }
                }
                if (active == 1280 && queries == 2) {
                    auto production = span_storage::create(f,0,active,queries,{0,256,active});
                    if (t.assert_true(production && production->plan)) {
                        const auto value=evaluate(f,ops,*production,input,0,accepted);
                        if (t.assert_true(accepted)) close_values(t,expected,value,1e-5f);
                    }
                }
                const auto b = evaluate(f, ops, *segmented, input, 0, accepted);
                if (!t.assert_true(accepted)) continue;
                close_values(t, a, b, 1e-7f);
                close_values(t, expected, b, 1e-5f);
            }
        }
    });

    if (argc > 1 && !std::strcmp(argv[1], "--cuda")) t.test(
            "tg1_page_aligned_fixed_spans_match_stock", [](testing & t) {
        auto * dev = ggml_backend_dev_by_name("CUDA0");
        ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
        auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(
            ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(dev), "ggml_backend_kv_stream_partial_ops"));
        const auto * ops = get ? get() : nullptr;
        fixture f(backend.get(), true, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 1024);
        if (!t.assert_true(ops && f.attach())) return;
        auto pin = f.binding->acquire();
        if (!t.assert_true(bool(pin))) return;
        block_inputs input(f,1024,1);
        const auto expected = ordinary(f,input,0);
        for (const auto & cuts : {
                std::vector<size_t>{0,1024},
                std::vector<size_t>{0,512,1024},
                std::vector<size_t>{0,256,768,1024}}) {
            auto storage = span_storage::create(f,0,1024,1,cuts);
            if (!t.assert_true(storage && storage->plan)) continue;
            bool accepted = false;
            const auto actual = evaluate(f,ops,*storage,input,0,accepted);
            if (!t.assert_true(accepted)) continue;
            close_values(t,expected,actual,1e-7f);
        }
    });

    if (argc > 1 && !std::strcmp(argv[1], "--cuda")) t.test(
            "tg1_fixed_spans_accept_exact_final_tail", [](testing & t) {
        auto * dev = ggml_backend_dev_by_name("CUDA0");
        ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
        auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(
            ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(dev), "ggml_backend_kv_stream_partial_ops"));
        const auto * ops = get ? get() : nullptr;
        fixture f(backend.get(), true, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 1025);
        if (!t.assert_true(ops && f.attach())) return;
        auto pin = f.binding->acquire();
        if (!t.assert_true(bool(pin))) return;
        block_inputs input(f,1023,1);
        const auto expected = ordinary(f,input,0);
        auto storage = span_storage::create(f,0,1023,1,{0,512,896,1023});
        if (!t.assert_true(storage && storage->plan)) return;
        bool accepted = false;
        const auto actual = evaluate(f,ops,*storage,input,0,accepted);
        if (!t.assert_true(accepted)) return;
        close_values(t,expected,actual,1e-7f);
    });

    if (argc > 1 && !std::strcmp(argv[1], "--cuda")) t.test(
            "tg3_tg4_f16_mma_spans_match_contiguous_and_stock", [](testing & t) {
        auto * dev = ggml_backend_dev_by_name("CUDA0");
        ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
        auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(
            ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(dev), "ggml_backend_kv_stream_partial_ops"));
        const auto * ops = get ? get() : nullptr;
        if (!t.assert_true(ops && ops->version >= 7 && ops->spans && ops->spans_workspace)) return;
        fixture f(backend.get(), true, GGML_TYPE_F16, GGML_TYPE_F16, 1025);
        if (!t.assert_true(f.attach())) return;
        auto pin = f.binding->acquire();
        if (!t.assert_true(bool(pin))) return;
        for (size_t active : {size_t(257), size_t(513), size_t(1017)}) {
            for (uint32_t queries : {3u, 4u}) {
                std::vector<size_t> cuts{0, 1, 63, 64, 65, 127, 128, 255, 256, 257, active - 1, active};
                cuts.erase(std::remove_if(cuts.begin(), cuts.end(),
                    [&](size_t value) { return value > active; }), cuts.end());
                std::sort(cuts.begin(), cuts.end());
                cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
                auto contiguous = span_storage::create(f, 0, active, queries, {0, active});
                auto segmented = span_storage::create(f, 0, active, queries, cuts);
                if (!t.assert_true(contiguous && segmented && contiguous->plan && segmented->plan)) continue;
                block_inputs input(f, active, queries);
                const auto expected = ordinary(f, input, 0);
                bool accepted = false;
                size_t contiguous_workspace = 0, segmented_workspace = 0;
                const auto a = evaluate(f, ops, *contiguous, input, 0, accepted, &contiguous_workspace);
                if (!t.assert_true(accepted)) continue;
                const std::vector<std::vector<size_t>> production_cuts{
                    {0,active/2,active},{0,active/3,2*active/3,active}};
                for (const auto & physical_cuts : production_cuts) {
                    auto fixed = span_storage::create(f,0,active,queries,physical_cuts);
                    if (!t.assert_true(fixed && fixed->plan)) continue;
                    const auto value = evaluate(f,ops,*fixed,input,0,accepted);
                    if (!t.assert_true(accepted)) continue;
                    close_values(t,a,value,1e-5f);
                    close_values(t,expected,value,1e-5f);
                }
                const auto before = input.read();
                const auto b = evaluate(f, ops, *segmented, input, 0, accepted, &segmented_workspace);
                if (cuts.size() > 4) {
                    t.assert_true(!accepted && b.empty());
                    t.assert_true(same_float_bits(before,input.read()));
                    continue;
                }
                if (!t.assert_true(accepted)) continue;
                ggml_kv_stream_layout gathered;
                if (t.assert_true(ggml_kv_stream_layout_make(f.policy.shape, active, gathered).status ==
                        ggml_kv_stream_status::success)) {
                    t.assert_true(contiguous_workspace < gathered.bytes);
                    t.assert_true(segmented_workspace < gathered.bytes);
                }
                close_values(t, a, b, 1e-5f);
                close_values(t, expected, b, 1e-5f);
            }
        }
    });

    if (argc > 1 && !std::strcmp(argv[1], "--cuda")) t.test(
            "encoded_mma_tile_bytes_equal_stock_conversion", [](testing & t) { check_mma_tile_conversion(t,256); });

    if (argc > 1 && !std::strcmp(argv[1], "--cuda")) t.test(
            "tg3_tg4_q8_q4_tile_conversion_matches_stock", [](testing & t) {
        auto * dev = ggml_backend_dev_by_name("CUDA0");
        ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
        auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(
            ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(dev), "ggml_backend_kv_stream_partial_ops"));
        const auto * ops = get ? get() : nullptr;
        fixture f(backend.get(), true, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 1025);
        if (!t.assert_true(ops && ops->version >= 7 && ops->spans && ops->spans_workspace && f.attach())) return;
        auto pin = f.binding->acquire();
        if (!t.assert_true(bool(pin))) return;
        size_t maximum_workspace = 0;
        for (size_t active : {size_t(257),size_t(513),size_t(1017)}) {
            for (uint32_t queries : {3u,4u}) {
                block_inputs input(f,active,queries);
                const auto expected = ordinary(f,input,0);
                const std::vector<std::vector<size_t>> layouts{
                    {0,active},
                    {0,active/2,active},
                    {0,active/4,3*active/4,active},
                    {0,1,127,128,255,256,active},
                };
                for (auto cuts : layouts) {
                    cuts.erase(std::remove_if(cuts.begin(),cuts.end(),
                        [&](size_t value) { return value > active; }),cuts.end());
                    std::sort(cuts.begin(),cuts.end());
                    cuts.erase(std::unique(cuts.begin(),cuts.end()),cuts.end());
                    auto storage = span_storage::create(f,0,active,queries,cuts);
                    if (!t.assert_true(storage && storage->plan)) continue;
                    bool accepted = false;
                    size_t workspace = 0;
                    const auto before = input.read();
                    const auto actual = evaluate(f,ops,*storage,input,0,accepted,&workspace);
                    if (cuts.size() > 4) {
                        t.assert_true(!accepted && actual.empty());
                        t.assert_true(same_float_bits(before,input.read()));
                        continue;
                    }
                    if (!t.assert_true(accepted)) continue;
                    maximum_workspace = std::max(maximum_workspace,workspace);
                    t.assert_true(workspace <= 1024*1024);
                    ggml_kv_stream_layout gathered;
                    auto f16 = f.policy.shape;
                    f16.type_k = f16.type_v = GGML_TYPE_F16;
                    if (t.assert_true(ggml_kv_stream_layout_make(f16,active,gathered).status ==
                            ggml_kv_stream_status::success)) {
                        t.assert_true(workspace < gathered.bytes);
                    }
                    close_values(t,expected,actual,2e-5f);
                }
            }
        }
        t.out << "maximum Q8/Q4 MMA workspace = " << maximum_workspace << " bytes\n";
        t.assert_true(maximum_workspace <= 1024*1024);
    });

    if (argc > 1 && !std::strcmp(argv[1], "--cuda")) t.test(
            "quantized_mma_workspace_ceiling_is_context_independent", [](testing & t) {
        auto * dev = ggml_backend_dev_by_name("CUDA0");
        ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
        auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(
            ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(dev), "ggml_backend_kv_stream_partial_ops"));
        const auto * ops = get ? get() : nullptr;
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,8192);
        if (!t.assert_true(ops != nullptr)) return;
        block_inputs input(f,8192,4);
        stock_attention_case stock(f,input);
        auto storage = span_storage::create(f,0,8192,4,{0,2048,6144,8192});
        if (!t.assert_true(storage && storage->plan)) return;
        ggml_context_ptr ctx(ggml_init({65536,nullptr,true}));
        auto * node = ggml_flash_attn_ext(ctx.get(),input.q,stock.k,stock.v,input.mask,1.0f/16,0,0);
        if (!t.assert_true(node != nullptr)) return;
        ggml_prec_set_acc(node,GGML_PREC_F32);
        ggml_tensor op = *node;
        op.buffer = input.output->buffer;
        op.data = input.output->data;
        size_t bytes = 0;
        if (!t.assert_true(ops->spans_workspace(backend.get(),&op,storage->plan,bytes))) return;
        t.out << "8K Q8/Q4 MMA workspace = " << bytes << " bytes\n";
        t.assert_true(bytes <= 1024*1024);
        ggml_kv_stream_layout gathered;
        auto f16 = f.policy.shape;
        f16.type_k = f16.type_v = GGML_TYPE_F16;
        if (t.assert_true(ggml_kv_stream_layout_make(f16,8192,gathered).status ==
                ggml_kv_stream_status::success)) t.assert_true(bytes < gathered.bytes);
    });

    if (argc > 1 && !std::strcmp(argv[1], "--cuda")) t.test(
            "quantized_mma_undersized_workspace_preserves_output", [](testing & t) {
        auto * dev = ggml_backend_dev_by_name("CUDA0");
        ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
        auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(
            ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(dev), "ggml_backend_kv_stream_partial_ops"));
        const auto * ops = get ? get() : nullptr;
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,769);
        if (!t.assert_true(ops && f.attach())) return;
        auto pin = f.binding->acquire();
        block_inputs input(f,513,4);
        auto storage = span_storage::create(f,0,513,4,{0,256,513});
        if (!t.assert_true(pin && storage && storage->plan)) return;
        ggml_context_ptr ctx(ggml_init({65536,nullptr,true}));
        auto * node = f.resident->attention(ctx.get(),0,input.q,input.mask,input.active,1.0f/16);
        if (!t.assert_true(node != nullptr)) return;
        ggml_tensor op = *node;
        op.buffer = input.output->buffer;
        op.data = input.output->data;
        size_t bytes = 0;
        if (!t.assert_true(ops->spans_workspace(backend.get(),&op,storage->plan,bytes) && bytes > 1)) return;
        block_workspace tiny(f,bytes-1,101);
        t.assert_true(!ops->spans(
            backend.get(),&op,storage->plan,ggml_backend_memory_lease_buffer(tiny.lease.get())));
        const auto actual = input.read();
        t.assert_true(std::all_of(actual.begin(),actual.end(),[](float value) { return value == -77; }));
    });

    if (argc > 1 && !std::strcmp(argv[1], "--cuda")) t.test(
            "invalid_span_plan_and_workspace_preserve_output", [](testing & t) {
        auto * dev = ggml_backend_dev_by_name("CUDA0");
        ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
        auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(
            ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(dev), "ggml_backend_kv_stream_partial_ops"));
        const auto * ops = get ? get() : nullptr;
        fixture f(backend.get(), true, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 769);
        if (!t.assert_true(ops && f.attach())) return;
        auto pin = f.binding->acquire();
        block_inputs input(f, 513, 2);
        auto wrong_width = span_storage::create(f, 0, 513, 1, {0, 256, 513});
        auto valid = span_storage::create(f, 0, 513, 2, {0, 256, 513});
        if (!t.assert_true(wrong_width && valid)) return;
        ggml_context_ptr ctx(ggml_init({65536, nullptr, true}));
        auto * node = f.resident->attention(ctx.get(), 0, input.q, input.mask, input.active, 1.0f/16);
        if (!t.assert_true(node != nullptr)) return;
        ggml_tensor op = *node;
        op.buffer = input.output->buffer;
        op.data = input.output->data;
        ggml_kv_stream_resume_plan plan;
        if (!t.assert_true(ops->resume_plan(
                backend.get(), GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 4, 2, 2, 513, plan))) return;
        block_workspace workspace(f, plan.bytes, 83), tiny(f, plan.bytes - 1, 89);
        t.assert_true(!ops->spans(backend.get(), &op, nullptr,
            ggml_backend_memory_lease_buffer(workspace.lease.get())));
        t.assert_true(!ops->spans(backend.get(), &op, wrong_width->plan,
            ggml_backend_memory_lease_buffer(workspace.lease.get())));
        t.assert_true(!ops->spans(backend.get(), &op, valid->plan,
            ggml_backend_memory_lease_buffer(tiny.lease.get())));
        std::vector<float> actual = input.read();
        t.assert_true(std::all_of(actual.begin(), actual.end(), [](float value) { return value == -77; }));
    });

    if (argc > 1 && (!std::strcmp(argv[1], "--cuda") ||
            !std::strcmp(argv[1], "--cuda-gqa6"))) t.test(
            "qwen_ratio_six_tg3_tg4_matches_stock", [](testing & t) {
        ggml_backend_load_all();
        auto * dev = ggml_backend_dev_by_name("CUDA0");
        if (!t.assert_true(dev != nullptr)) return;
        ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
        auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(
            ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(dev), "ggml_backend_kv_stream_partial_ops"));
        const auto * ops = get ? get() : nullptr;
        if (!t.assert_true(ops && ops->version >= 9 && ops->spans &&
                ops->spans_workspace && ops->mma_workspace)) return;
        fixture f(backend.get(), true, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0,
            768, false, 2, 2, 16);
        if (!t.assert_true(f.attach())) return;
        auto pin = f.binding->acquire();
        if (!t.assert_true(bool(pin))) return;
        size_t maximum = 0;
        if (!t.assert_true(ops->mma_workspace(backend.get(), GGML_TYPE_Q8_0,
                GGML_TYPE_Q4_0, 12, 2, 768, 2, maximum))) return;
        for (size_t active : {size_t(257), size_t(513)}) {
            for (uint32_t queries : {3u, 4u}) {
                block_inputs input(f, active, queries, false, 12);
                const auto expected = ordinary(f, input, 0);
                auto storage = span_storage::create(f, 0, active, queries,
                    {0, 256, active});
                if (!t.assert_true(bool(storage) && storage->plan)) return;
                bool accepted = false;
                size_t workspace = 0;
                const auto actual = evaluate(f, ops, *storage, input, 0,
                    accepted, &workspace);
                if (!t.assert_true(accepted)) return;
                t.assert_true(workspace <= maximum && maximum < 2*1048576);
                close_values(t, expected, actual, 2e-5f);
            }
        }
    });
    if (argc > 1 && (!std::strcmp(argv[1], "--cuda") ||
            !std::strcmp(argv[1], "--cuda-gqa6"))) t.test(
            "qwen_24_4_tg2_aligned_and_wrapped_spans_are_exact", [](testing & t) {
        ggml_backend_load_all();
        auto * dev = ggml_backend_dev_by_name("CUDA0");
        if (!t.assert_true(dev != nullptr)) return;
        ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
        auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(
            ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev),
                "ggml_backend_kv_stream_partial_ops"));
        const auto * ops = get ? get() : nullptr;
        if (!t.assert_true(ops && ops->spans && ops->spans_workspace)) return;
        fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,8192,false,2,4,80);
        if (!t.assert_true(f.attach())) return;
        auto pin = f.binding->acquire();
        if (!t.assert_true(bool(pin))) return;
        for (size_t active : {size_t(8192),size_t(8191)}) {
            block_inputs input(f,active,2,false,24);
            const auto expected = stock_attention(f,input,0);
            const std::vector<std::vector<size_t>> cuts{
                {0,active},{0,1024,active},{0,1024,4096,active},
                {0,127,257,active}};
            for (size_t index = 0; index < cuts.size(); ++index) {
                for (bool wrapped : {false,true}) {
                    if (wrapped && cuts[index].size() != 4) continue;
                    auto storage = span_storage::create(f,0,active,2,cuts[index],wrapped);
                    if (!t.assert_true(storage && storage->plan)) return;
                    bool accepted = false;
                    const auto actual = evaluate(f,ops,*storage,input,0,accepted);
                    if (!t.assert_true(accepted)) return;
                    if (active == 8192 && index < 3)
                        t.assert_true(same_float_bits(expected,actual));
                    else close_values(t,expected,actual,1e-5f);
                }
            }
        }
    });
    return t.summary();
}
