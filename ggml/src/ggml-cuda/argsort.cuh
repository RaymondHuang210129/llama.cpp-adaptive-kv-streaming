#include "common.cuh"

#define CUDA_ARGSORT_BLOCK_SIZE 1024

void ggml_cuda_op_argsort(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
struct ggml_backend_operator_workspace;
bool ggml_cuda_argsort_cub_workspace_plan(int columns,int rows,ggml_sort_order order,
        ggml_backend_operator_workspace & output);
bool ggml_cuda_argsort_workspace_plan(ggml_backend_cuda_context & ctx,const ggml_tensor * op,
        ggml_backend_operator_workspace & output);

int ggml_cuda_chunk_nrows(const size_t row_bytes, const int64_t nrows);

#ifdef GGML_CUDA_USE_CUB
void argsort_f32_i32_cuda_cub(ggml_cuda_pool & pool,
                              const float *    x,
                              int *            dst,
                              const int        ncols,
                              const int        nrows,
                              ggml_sort_order  order,
                              cudaStream_t     stream);
#endif  // GGML_CUDA_USE_CUB
void argsort_f32_i32_cuda_bitonic(const float *   x,
                                  int *           dst,
                                  const int       ncols,
                                  const int       nrows,
                                  ggml_sort_order order,
                                  cudaStream_t    stream);
