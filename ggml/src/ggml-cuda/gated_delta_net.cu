#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"

constexpr int gdn_cols_per_warp = 4;

bool ggml_cuda_gdn_fp32_replay_exact(int device) {
#ifdef GGML_CUDA_GDN_EXACT_SM120
    return ggml_cuda_info().devices[device].cc==1200;
#else
    GGML_UNUSED(device);
    return false;
#endif
}

template<int width, int values_per_thread=8>
__global__ void gated_delta_net_replay_cuda(const float * checkpoint, const float * decay, const float * keys,
        const float * corrections, const int32_t * prefixes, float * output,
        int64_t heads, int64_t key_heads, int64_t rows, uint3 key_magic, int32_t scalar_prefix) {
    constexpr int threads=256;
    static_assert(width*width % (threads*values_per_thread)==0,"whole replay tiles");
    const int head=blockIdx.x,sequence=blockIdx.y;
    const int first=blockIdx.z*threads*values_per_thread+threadIdx.x;
    const int row=first%width,col=first/width;
    const int64_t base=(int64_t(sequence)*heads+head)*width*width+first;
    const uint32_t key_head=fastmodulo(head,key_magic);
    ggml_cuda_pdl_sync();
    const int32_t prefix = prefixes ? prefixes[sequence] : scalar_prefix;
    if (prefix < 0 || prefix > rows) __trap();
    float values[values_per_thread];
#pragma unroll
    for (int j=0;j<values_per_thread;++j) values[j]=checkpoint[base+j*threads];
    for (int32_t t = 0; t < prefix; ++t) {
        const int64_t token = sequence * rows + t;
        const float alpha = decay[token * heads + head];
        const float key = keys[(token * key_heads + key_head) * width + row];
#pragma unroll
        for (int j=0;j<values_per_thread;++j) {
            const float correction=corrections[(token*heads+head)*width+col+j*(threads/width)];
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
            values[j]=alpha*values[j]+key*correction;
#else
            // The qualified native kernel rounds k*u before the alpha*S FMA.
            values[j]=__fmaf_rn(alpha,values[j],__fmul_rn(key,correction));
#endif
        }
    }
#pragma unroll
    for (int j=0;j<values_per_thread;++j) output[base+j*threads]=values[j];
}

template<int width,int values_per_thread>
static void launch_gdn_replay(ggml_backend_cuda_context & ctx,ggml_tensor * dst) {
    const int64_t heads=dst->ne[2],sequences=dst->ne[3],keys=dst->src[2]->ne[1];
    const ggml_cuda_kernel_launch_params launch(dim3(heads,sequences,width*width/(256*values_per_thread)),dim3(256),0,ctx.stream());
    const uint3 key_magic=init_fastdiv_values(keys);
    ggml_cuda_kernel_launch(gated_delta_net_replay_cuda<width,values_per_thread>, launch,
        (const float *)dst->src[0]->data, (const float *)dst->src[1]->data,
        (const float *)dst->src[2]->data, (const float *)dst->src[3]->data,
        dst->src[4] ? (const int32_t *)dst->src[4]->data : nullptr, (float *)dst->data,
        heads,keys,dst->src[1]->ne[2],key_magic,ggml_get_op_params_i32(dst,0));
}

template<int width>
static void select_gdn_replay(ggml_backend_cuda_context & ctx,ggml_tensor * dst) {
    const auto blocks=dst->ne[2]*dst->ne[3]*(width*width/256);
    const auto target=2*ggml_cuda_info().devices[ctx.device].nsm;
    if (blocks/8>=target) launch_gdn_replay<width,8>(ctx,dst);
    else if (blocks/4>=target) launch_gdn_replay<width,4>(ctx,dst);
    else if (blocks/2>=target) launch_gdn_replay<width,2>(ctx,dst);
    else launch_gdn_replay<width,1>(ctx,dst);
}

void ggml_cuda_op_gated_delta_net_replay(ggml_backend_cuda_context & ctx,ggml_tensor * dst) {
    if (dst->ne[0]==64) select_gdn_replay<64>(ctx,dst);
    else {GGML_ASSERT(dst->ne[0]==128);select_gdn_replay<128>(ctx,dst);}
}

struct gdn_direct_cache {
    const int32_t * destinations=nullptr;
    const int32_t * slots=nullptr;
    int64_t record_stride=0;
    int64_t capacity=0;
    int64_t slot_count=0;
    float * checkpoint=nullptr;
};

template <int S_v, bool KDA, bool keep_rs_t, bool capture = false, bool direct = false, int cols_per_warp = gdn_cols_per_warp>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K,
                                     float *       records,
                                     int64_t       key_heads,
                                     int64_t       retained,
                                     gdn_direct_cache cache) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    // the warp is split into cols_per_warp segments of lanes_per_col lanes; each segment owns
    // one state column and reduces within itself
    constexpr int lanes_per_col = warp_size / cols_per_warp;
    constexpr int rows_per_lane = S_v / lanes_per_col;
    static_assert(S_v % lanes_per_col == 0, "S_v must be a multiple of lanes_per_col");

    const int lane        = threadIdx.x;
    const int col_in_warp = lane / lanes_per_col;              // column slot within the warp
    const int lane_in_col = lane - col_in_warp * lanes_per_col;  // lane within the column's reduction segment
    const int col = (blockIdx.z * blockDim.y + threadIdx.y) * cols_per_warp + col_in_warp;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    int64_t output_sequence=sequence;
    if constexpr (direct) {
        ggml_cuda_pdl_sync();
        output_sequence=cache.destinations[sequence];
        const int32_t slot=cache.slots[sequence];
        if (output_sequence<0 || output_sequence>=2*cache.slot_count || slot<0 || slot>=cache.slot_count) __trap();
        records+=int64_t(slot)*cache.record_stride;
    }
    const int64_t state_out_offset     = (output_sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    float * checkpoint=nullptr;
    if constexpr (capture && !direct) checkpoint=state+H*n_seqs*S_v*S_v;
    if constexpr (direct) if (cache.checkpoint) checkpoint=cache.checkpoint+state_in_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    if constexpr (!direct) ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * lanes_per_col + lane_in_col;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        if constexpr (capture) {
            if (checkpoint && t == n_tokens - retained) {
#pragma unroll
                for (int r = 0; r < rows_per_lane; ++r) {
                    const int i = r * lanes_per_col + lane_in_col;
                    checkpoint[col*S_v+i] = s_shard[r];
                }
            }
        }
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * lanes_per_col + lane_in_col;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            if constexpr (capture) {
              if (t >= n_tokens - retained) {
                const int64_t row = (direct ? 0 : sequence * retained) + t - (n_tokens - retained);
                const int64_t record_rows=direct ? cache.capacity : retained*n_seqs;
                float * keys = records + H * record_rows;
                float * corrections = keys + key_heads * S_v * record_rows;
                if (lane_in_col == 0) {
                    corrections[(row * H + h_idx) * S_v + col] = delta_col;
                    if (col == 0) records[row * H + h_idx] = g_val;
                }
                if (h_idx < key_heads && col == 0) {
#pragma unroll
                    for (int r = 0; r < rows_per_lane; ++r) {
                        const int i = r * lanes_per_col + lane_in_col;
                        keys[(row * key_heads + h_idx) * S_v + i] = k_reg[r];
                    }
                }
              }
            }

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                if constexpr (direct) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
                    s_shard[r]=g_val*s_shard[r]+k_reg[r]*delta_col;
#else
                    s_shard[r]=__fmaf_rn(g_val,s_shard[r],__fmul_rn(k_reg[r],delta_col));
#endif
                } else s_shard[r] = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

            if (lane_in_col == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * lanes_per_col + lane_in_col;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * lanes_per_col + lane_in_col;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

            if (lane_in_col == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * lanes_per_col + lane_in_col;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * lanes_per_col + lane_in_col;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

template <bool KDA, bool keep_rs_t, bool capture = false, bool direct = false>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream,
        float * records = nullptr, int64_t key_heads = 0, int64_t retained = 0,gdn_direct_cache cache={}) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    // four columns per warp (see the kernel); shrink the CTA when the wider CTA would leave
    // SMs without a CTA, so small head counts keep the device filled
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int cols_per_warp = gdn_cols_per_warp;
    int num_warps = 4;
    while (num_warps > 1 && H*n_seqs*(S_v / (cols_per_warp * num_warps)) < nsm) {
        num_warps /= 2;
    }
    // one CTA covers cols_per_warp*num_warps columns (see the kernel)
    dim3      grid_dims(H, n_seqs, (S_v + cols_per_warp * num_warps - 1) / (cols_per_warp * num_warps));
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t, capture, direct>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, records, key_heads, retained, cache);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t, capture, direct>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, records, key_heads, retained, cache);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t, capture, direct>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, records, key_heads, retained, cache);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t, capture, direct>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, records, key_heads, retained, cache);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    if (dst->op == GGML_OP_GATED_DELTA_NET_CAPTURE_DIRECT) {
        GGML_ASSERT(!kda && !cache && (S_v==64 || S_v==128));
        gdn_direct_cache target;
        target.destinations=(const int32_t *)dst->src[8]->data;
        target.slots=(const int32_t *)dst->src[9]->data;
        target.record_stride=dst->src[7]->ne[0];target.slot_count=dst->src[7]->ne[1];
        target.capacity=ggml_get_op_params_i32(dst,2);
        target.checkpoint=n_tokens>target.capacity ? state_d : nullptr;
        state_d=(float *)dst->src[6]->data;
        launch_gated_delta_net<false,false,true,true>(q_d,k_d,v_d,g_d,b_d,s_d,dst_d,state_d,
            S_v,H,n_tokens,n_seqs,sq1,sq2,sq3,sv1,sv2,sv3,sb1,sb2,sb3,neqk1,rq3,scale,state_slot_stride,1,stream,
            (float *)dst->src[7]->data,neqk1,ggml_get_op_params_i32(dst,1),target);
    } else if (dst->op == GGML_OP_GATED_DELTA_NET_CAPTURE) {
        GGML_ASSERT(!kda && !cache && (S_v == 64 || S_v == 128));
        float * records = state_d + 2 * state_slot_stride;
        launch_gated_delta_net<false, false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
            S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, 1, stream, records, neqk1, ggml_get_op_params_i32(dst, 1));
    } else if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}
