#pragma once
#include "kv-stream-resident-test.h"
#include "../ggml/src/ggml-kv-stream-device.h"

struct block_workspace {
    arena_ptr arena{nullptr, ggml_backend_memory_arena_free};
    lease_ptr lease{nullptr, ggml_backend_memory_lease_free};
    // Exercise an exact-sized lease at a nonzero parent offset.
    block_workspace(fixture & f, size_t bytes, uint64_t id = 19) {
        auto * type = f.cuda ? llama_kv_stream_device_buffer_type(ggml_backend_get_device(f.backend)) : ggml_backend_cpu_buffer_type();
        arena.reset(ggml_backend_memory_arena_new(type, bytes + 256));
        GGML_ASSERT(arena && ggml_backend_memory_arena_begin(arena.get(), 0));
        const auto base = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(ggml_backend_memory_arena_parent(arena.get())));
        GGML_ASSERT(ggml_backend_memory_arena_reserve_at(arena.get(), id, 128 + (128-base%128)%128, bytes, 64, 0, nullptr));
        GGML_ASSERT(ggml_backend_memory_arena_commit(arena.get()));
        lease.reset(ggml_backend_memory_arena_acquire(arena.get(), id));
    }
};

struct block_inputs {
    ggml_context_ptr context;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * q, * mask, * output;
    std::vector<float> qdata;
    size_t active, queries, padded, query_heads, value_width;
    // Keep the full causal-mask row pitch while the two attention calls use separate key slices.
    block_inputs(fixture & f, size_t active, size_t queries, bool masked = false, size_t query_heads = 4,
            ggml_backend_buffer_t workspace = nullptr) : active(active), queries(queries), padded((active+255)/256*256), query_heads(query_heads), value_width(size_t(f.policy.shape.head_dim_v)) {
        context.reset(ggml_init({65536, nullptr, true}));
        q = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, f.policy.shape.head_dim_k, int64_t(queries), int64_t(query_heads));
        mask = ggml_new_tensor_2d(context.get(), GGML_TYPE_F16, int64_t(padded), int64_t(queries));
        output = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, int64_t(value_width), int64_t(query_heads), int64_t(queries));
        if (workspace) {
            buffer.reset(ggml_backend_buffer_retain(workspace));
            ggml_tallocr allocator;
            GGML_ASSERT(ggml_tallocr_new_range(&allocator,workspace,0,ggml_backend_buffer_get_size(workspace)));
            for (auto * tensor : {q,mask,output}) GGML_ASSERT(ggml_tallocr_alloc(&allocator,tensor) == GGML_STATUS_SUCCESS);
        } else buffer.reset(ggml_backend_alloc_ctx_tensors(context.get(), f.backend));
        GGML_ASSERT(buffer);
        qdata.resize(queries*query_heads*size_t(f.policy.shape.head_dim_k));
        for (size_t i = 0; i < qdata.size(); ++i) qdata[i] = .3f*std::cos(float(i%211)*.07f);
        std::vector<ggml_fp16_t> masks(padded*queries);
        for (size_t query = 0; query < queries; ++query) for (size_t token = 0; token < padded; ++token)
            masks[query*padded+token] = ggml_fp32_to_fp16(!masked && token <= active-queries+query ? 0 : -INFINITY);
        ggml_backend_tensor_set(q, qdata.data(), 0, qdata.size()*sizeof(float));
        ggml_backend_tensor_set(mask, masks.data(), 0, masks.size()*sizeof(ggml_fp16_t));
        std::vector<float> sentinel(queries*query_heads*value_width, -77);
        ggml_backend_tensor_set(output, sentinel.data(), 0, sentinel.size()*sizeof(float));
    }
    // Read only the public output, not intermediate or padded scratch bytes.
    std::vector<float> read() const {
        std::vector<float> result(queries*query_heads*value_width);
        ggml_backend_tensor_get(output, result.data(), 0, result.size()*sizeof(float));
        return result;
    }
};

inline std::vector<float> stock_attention(fixture & f, block_inputs & input, uint32_t layer, float scale = 1.0f/16,
        bool (*native_override)(ggml_backend_t,ggml_tensor *) = nullptr) {
    ggml_context_ptr ctx(ggml_init({65536,nullptr,true}));
    const int64_t kv_heads=f.policy.shape.heads;
    const int64_t dk=f.policy.shape.head_dim_k, dv=f.policy.shape.head_dim_v;
    auto * k=ggml_new_tensor_2d(ctx.get(),ggml_type(f.policy.shape.type_k),dk*kv_heads,input.padded);
    auto * v=ggml_new_tensor_2d(ctx.get(),ggml_type(f.policy.shape.type_v),dv*kv_heads,input.padded);
    auto * key=ggml_view_3d(ctx.get(),k,dk,input.padded,kv_heads,ggml_row_size(k->type,dk*kv_heads),ggml_row_size(k->type,dk),0);
    auto * value=ggml_view_3d(ctx.get(),v,dv,input.padded,kv_heads,ggml_row_size(v->type,dv*kv_heads),ggml_row_size(v->type,dv),0);
    auto * out=ggml_flash_attn_ext(ctx.get(),input.q,key,value,input.mask,scale,0,0);
    ggml_prec_set_acc(out,GGML_PREC_F32);
    auto * graph=ggml_new_graph_custom(ctx.get(),64,false); ggml_build_forward_expand(graph,out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),f.backend)); GGML_ASSERT(buffer);
    using release_fn=void (*)(ggml_backend_t,const void *);
    release_fn release=nullptr;
    if (f.cuda) {
        auto * reg=ggml_backend_dev_backend_reg(ggml_backend_get_device(f.backend));
        release=reinterpret_cast<release_fn>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_cuda_graph_release"));
    }
    // This transient reference must retire captures before its metadata and buffers are freed.
    struct retirement {
        ggml_backend_t backend;const void * key;release_fn release;
        ~retirement() {if (release) {ggml_backend_synchronize(backend);release(backend,key);}}
    } captured{f.backend,ggml_graph_node(graph,0),release};
    llama_kv_stream_host_layer host; GGML_ASSERT(f.host->layer(layer,host));
    std::vector<uint8_t> key_bytes,value_bytes;
    const void * key_source=host.k, * value_source=host.v;
    if (f.host->demand_backed()) {
        std::vector<llama_kv_stream_host_range> ranges;
        GGML_ASSERT(f.host->address_ranges(host.k,ggml_nbytes(k),ranges) && f.host->address_ranges(host.v,ggml_nbytes(v),ranges));
        key_bytes.resize(ggml_nbytes(k)); value_bytes.resize(ggml_nbytes(v));
        std::memcpy(key_bytes.data(),host.k,key_bytes.size());
        std::memcpy(value_bytes.data(),host.v,value_bytes.size());
        key_source=key_bytes.data(); value_source=value_bytes.data();
    }
    ggml_backend_tensor_set(k,key_source,0,ggml_nbytes(k)); ggml_backend_tensor_set(v,value_source,0,ggml_nbytes(v));
    GGML_ASSERT(ggml_backend_graph_compute(f.backend,graph) == GGML_STATUS_SUCCESS);
    if (native_override) GGML_ASSERT(native_override(f.backend,out));
    std::vector<float> result(ggml_nelements(out)); ggml_backend_tensor_get(out,result.data(),0,ggml_nbytes(out)); return result;
}

inline bool same_float_bits(const std::vector<float> & expected, const std::vector<float> & actual) {
    return expected.size() == actual.size() &&
        std::memcmp(expected.data(),actual.data(),expected.size()*sizeof(float)) == 0;
}

// Check finiteness before measuring the worst absolute error.
static void close_values(testing & t, const std::vector<float> & expected, const std::vector<float> & actual, float tolerance) {
    if (!t.assert_equal(expected.size(), actual.size())) return;
    float error = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (!std::isfinite(actual[i])) { t.assert_true(false); return; }
        error = std::max(error, std::abs(actual[i]-expected[i]));
    }
    t.out << "max absolute error = " << error << '\n';
    t.assert_true(error < tolerance);
}
