#include "kv-stream-runtime.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
// Reuse native compiled-pair admission even though the arithmetic reader supports more formats.
ggml_kv_mma_runtime_kernel_t ggml_cuda_kv_stream_mma_runtime_kernel(
        ggml_type key, ggml_type value, uint32_t columns, uint32_t groups, bool resume, bool dense) {
    if (!ggml_cuda_kv_stream_kernel(key,value)) return nullptr;
    const bool encoded=key != GGML_TYPE_F16 || value != GGML_TYPE_F16;
#define RUNTIME_SELECT(C,G) if (columns == C && groups == G) return ggml_cuda_kv_stream_mma_runtime_kernel_##C##_##G(encoded,resume,dense);
    RUNTIME_SELECT(1,8)
    RUNTIME_SELECT(2,8)
    RUNTIME_SELECT(4,8)
    RUNTIME_SELECT(8,8)
    RUNTIME_SELECT(4,2)
    RUNTIME_SELECT(8,2)
#undef RUNTIME_SELECT
    return nullptr;
}
#endif
