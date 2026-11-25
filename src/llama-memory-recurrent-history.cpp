#include "llama-memory-recurrent-history.h"

#include <limits>

ggml_tensor * llama_recurrent_history_capture(ggml_context * ctx, ggml_tensor * input,
        uint32_t history, uint32_t rows) {
    if (!ctx || !input || input->type != GGML_TYPE_F32 || !history || !rows ||
            input->ne[3] != 1 || input->ne[0] < int64_t(history) + rows ||
            input->nb[0] != sizeof(float)) return nullptr;
    const size_t offset = size_t(input->ne[0] - history - rows) * sizeof(float);
    return ggml_cont(ctx, ggml_view_3d(ctx, input, int64_t(history) + rows, input->ne[1], input->ne[2],
                input->nb[1], input->nb[2], offset));
}

bool llama_recurrent_history_indices(uint32_t history, uint32_t rows,
        const std::vector<uint32_t> & prefixes, int32_t * output, size_t capacity) noexcept {
    if (!history || !rows || !output || prefixes.empty() ||
            uint64_t(history) + rows > uint64_t(std::numeric_limits<int32_t>::max()) ||
            prefixes.size() > capacity / history) return false;
    for (auto prefix : prefixes) if (prefix > rows) return false;
    for (size_t s = 0; s < prefixes.size(); ++s) for (uint32_t i = 0; i < history; ++i)
        output[s * history + i] = int32_t(prefixes[s] + i);
    return true;
}

ggml_tensor * llama_recurrent_history_restore(ggml_context * ctx, ggml_tensor * timeline, ggml_tensor * indices) {
    if (!ctx || !timeline || !indices || timeline->type != GGML_TYPE_F32 ||
            timeline->ne[3] != 1 || indices->type != GGML_TYPE_I32 ||
            indices->ne[1] != timeline->ne[2] || indices->ne[2] != 1 || indices->ne[3] != 1 ||
            indices->ne[0] > timeline->ne[0]) return nullptr;
    auto * frames = ggml_cont(ctx, ggml_permute(ctx, timeline, 1, 0, 2, 3));
    auto * windows = ggml_get_rows(ctx, frames, indices);
    return ggml_cont(ctx, ggml_permute(ctx, windows, 1, 0, 2, 3));
}
