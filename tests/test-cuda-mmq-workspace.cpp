#include "../ggml/src/ggml-cuda/scratch-plan.h"

#undef NDEBUG
#include <cassert>

int main() {
    ggml_tensor activations{};
    activations.ne[0] = 256;
    activations.ne[1] = 3;
    activations.ne[2] = 2;
    activations.ne[3] = 5;

    ggml_backend_operator_workspace plan;
    // Thirty rows, padded to 512 values, plus 1536 bytes of tile padding.
    assert(ggml_cuda_mmq_activation_workspace(activations, 512, 8, 128, GGML_TYPE_Q5_K, GGML_PREC_Q8, plan));
    assert(plan.count == 1 && plan.requests[0] == 18816);
    assert(ggml_cuda_mmq_activation_workspace(activations, 512, 8, 128, GGML_TYPE_NVFP4, GGML_PREC_Q8, plan));
    assert(plan.count == 1 && plan.requests[0] == 18816);
    assert(ggml_cuda_mmq_activation_workspace(activations, 512, 8, 160, GGML_TYPE_Q5_K, GGML_PREC_Q8, plan));
    assert(plan.count == 1 && plan.requests[0] == 18560);

    assert(ggml_cuda_mmq_activation_workspace(activations, 512, 8, 128, GGML_TYPE_MXFP4, GGML_PREC_Q4, plan));
    assert(plan.count == 1 && plan.requests[0] == 10176);
    assert(ggml_cuda_mmq_activation_workspace(activations, 512, 8, 128, GGML_TYPE_NVFP4, GGML_PREC_Q4, plan));
    assert(plan.count == 2 && plan.requests[0] == 10176 && plan.requests[1] == 120);
    assert(plan.offsets[1] == 10240 && plan.bytes == 10368);

    size_t fixup_bytes = 0;
    assert(ggml_cuda_mmq_fixup_workspace(1152, 576, 1, 1, 128, 128, 70, true, true, fixup_bytes));
    assert(fixup_bytes == 4587520);
    assert(ggml_backend_operator_workspace_add(plan, fixup_bytes));
    assert(plan.count == 3 && plan.offsets[2] == 10368 && plan.requests[2] == fixup_bytes);
    assert(ggml_backend_operator_workspace_valid(plan));

    const auto saved = plan;
    assert(!ggml_cuda_mmq_activation_workspace(activations, 384, 8, 128, GGML_TYPE_NVFP4, GGML_PREC_Q4, plan));
    assert(!ggml_cuda_mmq_activation_workspace(activations, 512, 0, 128, GGML_TYPE_NVFP4, GGML_PREC_Q4, plan));
    assert(!ggml_cuda_mmq_activation_workspace(activations, 512, 8, 0, GGML_TYPE_NVFP4, GGML_PREC_Q4, plan));
    assert(!ggml_cuda_mmq_activation_workspace(activations, 512, 8, 128, GGML_TYPE_NVFP4, GGML_PREC_UNDEFINED, plan));
    activations.ne[3] = 0;
    assert(!ggml_cuda_mmq_activation_workspace(activations, 512, 8, 128, GGML_TYPE_NVFP4, GGML_PREC_Q4, plan));
    activations.ne[3] = INT64_MAX;
    assert(!ggml_cuda_mmq_activation_workspace(activations, 512, 8, 128, GGML_TYPE_NVFP4, GGML_PREC_Q4, plan));
    assert(plan.count == saved.count && plan.requests == saved.requests && plan.offsets == saved.offsets && plan.bytes == saved.bytes);
}
