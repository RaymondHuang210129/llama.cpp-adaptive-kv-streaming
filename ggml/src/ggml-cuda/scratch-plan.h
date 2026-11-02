#pragma once

#include "../ggml-backend-operator-workspace.h"
#include <cmath>
#include <cstring>
#include <climits>

struct ggml_cuda_operator_tile_traits {
    int occupancy = 0, nbatch = 0, threads = 0;
};
struct ggml_cuda_operator_workspace_caps {
    bool ready = false;
    int cc = 0, nsm = 0, warp_size = 0;
    size_t shared_bytes_per_block = 0, library_bytes = 0;
    std::array<std::array<ggml_cuda_operator_tile_traits,5>,2> tile{};
};
struct ggml_cuda_operator_kernel_metadata {
    const void * kernel = nullptr;
    int threads = 0, nbatch = 0;
};
// Implemented beside the existing native tile instantiation; returns no device payload.
bool ggml_cuda_operator_tile_kernel_metadata(int columns,bool softcap,int cc,
    ggml_cuda_operator_kernel_metadata & output);

static inline bool ggml_cuda_operator_product(size_t & value,size_t factor) {
    if (factor && value>SIZE_MAX/factor) return false;
    value*=factor;return true;
}
// Native non-Stream-K tile policy for unmasked F32/F16 attention with equal heads.
static inline bool ggml_cuda_attention_workspace_plan(const ggml_cuda_operator_workspace_caps & caps,
        const ggml_tensor * op,ggml_backend_operator_workspace & output) {
    if (!caps.ready || caps.nsm<=0 || caps.nsm>1024 || !op || op->op!=GGML_OP_FLASH_ATTN_EXT ||
            op->type!=GGML_TYPE_F32 || !op->src[0] || !op->src[1] || !op->src[2] || op->src[3] || op->src[4]) return false;
    const auto * q=op->src[0];const auto * k=op->src[1];const auto * v=op->src[2];
    if (q->type!=GGML_TYPE_F32 || k->type!=GGML_TYPE_F16 || v->type!=GGML_TYPE_F16 ||
            q->ne[0]!=72 || k->ne[0]!=72 || v->ne[0]!=72 || q->nb[0]!=4 || k->nb[0]!=2 || v->nb[0]!=2 ||
            q->ne[2]!=k->ne[2] || k->ne[2]!=v->ne[2] || k->ne[1]!=v->ne[1] ||
            q->ne[3]!=k->ne[3] || k->ne[3]!=v->ne[3] ||
            op->ne[0]!=72 || op->ne[1]!=q->ne[2] || op->ne[2]!=q->ne[1] || op->ne[3]!=q->ne[3]) return false;
    for (const auto * tensor : {q,k,v,op}) for (int i=0;i<GGML_MAX_DIMS;++i)
        if (tensor->ne[i]<=0 || tensor->ne[i]>INT32_MAX) return false;
    float bias,softcap;
    std::memcpy(&bias,reinterpret_cast<const char *>(op->op_params)+sizeof(float),sizeof(float));
    std::memcpy(&softcap,reinterpret_cast<const char *>(op->op_params)+2*sizeof(float),sizeof(float));
    if (bias!=0.0f || !std::isfinite(softcap)) return false;
    int columns=2,index=0;
    while (columns<32 && q->ne[1]>columns) { columns*=2;++index; }
    const auto & traits=caps.tile[softcap!=0.0f][index];
    if (traits.occupancy<=0 || traits.occupancy>32 || traits.nbatch<=0) return false;
    const int64_t tiles_kv=(k->ne[1]+traits.nbatch-1)/traits.nbatch;
    size_t tiles=(size_t(q->ne[1])+columns-1)/columns;
    if (!ggml_cuda_operator_product(tiles,size_t(q->ne[2])) ||
            !ggml_cuda_operator_product(tiles,size_t(q->ne[3])) || tiles>size_t(INT32_MAX)/size_t(tiles_kv)) return false;
    const int64_t blocks_per_wave=int64_t(caps.nsm)*traits.occupancy;
    int64_t parallel=std::min<int64_t>(traits.occupancy,tiles_kv),best_waves=0,best_efficiency=0;
    for (int64_t candidate=parallel;candidate<=tiles_kv;++candidate) {
        const int64_t blocks=int64_t(tiles)*candidate;
        const int64_t waves=(blocks+blocks_per_wave-1)/blocks_per_wave;
        const int64_t efficiency=100*blocks/(waves*blocks_per_wave);
        if (best_efficiency>=95 && waves>best_waves) break;
        if (efficiency>best_efficiency) { best_waves=waves;best_efficiency=efficiency;parallel=candidate; }
    }
    ggml_backend_operator_workspace next;
    if (parallel>1) {
        size_t rows=size_t(parallel);
        for (int i=1;i<GGML_MAX_DIMS;++i) if (!ggml_cuda_operator_product(rows,size_t(op->ne[i]))) return false;
        size_t payload=rows,metadata=rows;
        if (!ggml_cuda_operator_product(payload,size_t(op->ne[0])) || !ggml_cuda_operator_product(payload,sizeof(float)) ||
                !ggml_cuda_operator_product(metadata,2*sizeof(float)) ||
                !ggml_backend_operator_workspace_add(next,payload) || !ggml_backend_operator_workspace_add(next,metadata)) return false;
    }
    output=next;return true;
}

static constexpr size_t ggml_cuda_operator_conversion_chunk_bytes=512*1024*1024;
static inline bool ggml_cuda_operator_elements(const ggml_tensor * tensor,size_t & output) {
    if (!tensor || tensor->type<0 || tensor->type>=GGML_TYPE_COUNT) return false;
    const int64_t block=ggml_blck_size(tensor->type);
    const size_t element=ggml_type_size(tensor->type);
    if (block<=0 || !element || tensor->ne[0]<=0 || tensor->ne[0]%block) return false;
    size_t count=1,span=element;
    for (int i=0;i<GGML_MAX_DIMS;++i) {
        if (tensor->ne[i]<=0 || tensor->ne[i]>INT32_MAX ||
                !ggml_cuda_operator_product(count,size_t(tensor->ne[i]))) return false;
        size_t term=size_t(tensor->ne[i]/(i==0 ? block : 1))-1;
        if (!ggml_cuda_operator_product(term,tensor->nb[i]) || term>SIZE_MAX-span) return false;
        span+=term;
    }
    output=count;return true;
}
// Match the native cuBLAS adapter's conversion, output and batch-pointer lifetimes.
static inline bool ggml_cuda_cublas_workspace_plan(const ggml_tensor * op,ggml_type compute_type,
        bool prefer_f32_output,size_t library_bytes,ggml_backend_operator_workspace & output) {
    if (!op || op->op!=GGML_OP_MUL_MAT || op->type!=GGML_TYPE_F32 || !op->src[0] || !op->src[1] ||
            (compute_type!=GGML_TYPE_F16 && compute_type!=GGML_TYPE_BF16 && compute_type!=GGML_TYPE_F32)) return false;
    ggml_tensor a=*op->src[0],b=*op->src[1],dst=*op;
    size_t na,nb,nd,checked;
    if (!ggml_cuda_operator_elements(&a,na) || !ggml_cuda_operator_elements(&b,nb) ||
            !ggml_cuda_operator_elements(&dst,nd) || a.nb[0]!=ggml_type_size(a.type) ||
            b.nb[0]!=ggml_type_size(b.type) || a.ne[0]!=b.ne[0] || dst.ne[0]!=a.ne[1] ||
            dst.ne[1]!=b.ne[1] || dst.ne[2]!=b.ne[2] || dst.ne[3]!=b.ne[3] ||
            b.ne[2]%a.ne[2] || b.ne[3]%a.ne[3] ||
            !ggml_backend_operator_workspace_align(library_bytes,256,checked)) return false;
    if (compute_type==GGML_TYPE_F32 && (a.type==GGML_TYPE_F16 || a.type==GGML_TYPE_BF16)) {
        size_t converted=na;
        if (!ggml_cuda_operator_product(converted,sizeof(float))) return false;
        if (converted>ggml_cuda_operator_conversion_chunk_bytes) {
            const size_t row=converted/size_t(a.ne[1]);
            const size_t rows=std::max(size_t(1),ggml_cuda_operator_conversion_chunk_bytes/row);
            if (rows<size_t(a.ne[1])) { a.ne[1]=int64_t(rows);dst.ne[0]=int64_t(rows); }
        }
    }
    if (!ggml_cuda_operator_elements(&a,na) || !ggml_cuda_operator_elements(&dst,nd)) return false;
    ggml_backend_operator_workspace next;next.library_bytes=library_bytes;
    const size_t element=ggml_type_size(compute_type);
    bool cont_a=ggml_is_contiguous_2(&a),cont_b=ggml_is_contiguous_2(&b);
    if (a.type!=compute_type) {
        if (!ggml_cuda_operator_product(na,element) || !ggml_backend_operator_workspace_add(next,na)) return false;
        if (!ggml_is_contiguously_allocated(&a)) cont_a=true;
    }
    if (b.type!=compute_type) {
        if (!ggml_cuda_operator_product(nb,element) || !ggml_backend_operator_workspace_add(next,nb)) return false;
        if (!ggml_is_contiguously_allocated(&b)) cont_b=true;
    }
    if (compute_type!=GGML_TYPE_F32 && !prefer_f32_output &&
            (!ggml_cuda_operator_product(nd,element) || !ggml_backend_operator_workspace_add(next,nd))) return false;
    if (!(b.ne[2]==1 && b.ne[3]==1) &&
            !(b.ne[2]==a.ne[2] && b.ne[3]==a.ne[3] && cont_a && cont_b)) {
        size_t pointers=size_t(b.ne[2]);
        if (!ggml_cuda_operator_product(pointers,size_t(b.ne[3])) || pointers>INT32_MAX ||
                !ggml_cuda_operator_product(pointers,sizeof(void *)) || pointers>SIZE_MAX/2 ||
                !ggml_backend_operator_workspace_add(next,2*pointers) || !ggml_backend_operator_workspace_add(next,pointers)) return false;
    }
    output=next;return true;
}

bool ggml_cuda_mmq_workspace_plan(const ggml_tensor * a,const ggml_tensor * b,const ggml_tensor * dst,
    int cc,size_t shared_bytes,int nsm,ggml_backend_operator_workspace & output,int * selected_columns=nullptr,int * selected_threads=nullptr);

static inline bool ggml_cuda_mmq_fixup_workspace(int64_t rows,int64_t columns,int64_t channels,int64_t samples,
        int tile_rows,int tile_columns,int nsm,bool stream_k,bool nvidia,size_t & output) {
    if (rows<=0 || rows>INT32_MAX || columns<=0 || columns>INT32_MAX || channels<=0 || channels>INT32_MAX ||
            samples<=0 || samples>INT32_MAX || tile_rows<=0 || tile_columns<=0 || nsm<=0 || nsm>1024) return false;
    size_t tiles=(size_t(rows)+size_t(tile_rows)-1)/size_t(tile_rows);
    if (!ggml_cuda_operator_product(tiles,(size_t(columns)+size_t(tile_columns)-1)/size_t(tile_columns)) ||
            !ggml_cuda_operator_product(tiles,size_t(channels)) || !ggml_cuda_operator_product(tiles,size_t(samples)) ||
            tiles>INT32_MAX) return false;
    size_t bytes=0;
    if (stream_k) {
        const size_t waves=(tiles+size_t(nsm)-1)/size_t(nsm);
        const size_t efficiency=100*tiles/(size_t(nsm)*waves);
        const size_t blocks=nvidia && efficiency>=90 ? tiles : size_t(nsm);
        if (tiles%blocks) {
            bytes=blocks;
            if (!ggml_cuda_operator_product(bytes,size_t(tile_columns)) ||
                    !ggml_cuda_operator_product(bytes,size_t(tile_rows)) || !ggml_cuda_operator_product(bytes,sizeof(float))) return false;
        }
    }
    output=bytes;return true;
}
