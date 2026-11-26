#include "common.cuh"

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
struct ggml_backend_operator_workspace;
bool ggml_cuda_top_k_workspace_plan(ggml_backend_cuda_context & ctx,const ggml_tensor * op,
        ggml_backend_operator_workspace & output);
