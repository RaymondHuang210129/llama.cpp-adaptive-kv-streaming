#include "../src/llama-memory-recurrent-history.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cpp.h"
#include "testing.h"

#include <cstring>

static void windows(testing & t, ggml_backend_t backend) {
    for (int slots = 1; slots <= 4; ++slots) for (int rows = 1; rows <= 8; ++rows) {
        const int history = 3, channels = 32, tail = std::min(rows, 4);
        ggml_context_ptr ctx(ggml_init({128 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true}));
        auto * input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, history + rows, channels, slots);
        auto * captured = llama_recurrent_history_capture(ctx.get(), input, history, tail);
        auto * indices = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, history, slots);
        auto * restored = llama_recurrent_history_restore(ctx.get(), captured, indices);
        auto * graph = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(graph, restored);
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
        std::vector<uint32_t> data(ggml_nelements(input));
        for (size_t i = 0; i < data.size(); ++i) data[i] = uint32_t(i * 2654435761u);
        ggml_backend_tensor_set(input, data.data(), 0, ggml_nbytes(input));
        for (int repeat = 0; repeat < 3; ++repeat) for (int prefix = 0; prefix <= tail; ++prefix) {
            std::vector<uint32_t> prefixes(slots);
            for (int s = 0; s < slots; ++s) prefixes[s] = (prefix + s) % (tail + 1);
            std::vector<int32_t> positions(history * slots);
            t.assert_true(llama_recurrent_history_indices(history, tail, prefixes, positions.data(), positions.size()));
            ggml_backend_tensor_set(indices, positions.data(), 0, ggml_nbytes(indices));
            t.assert_true(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
            std::vector<uint32_t> observed(ggml_nelements(restored));
            ggml_backend_tensor_get(restored, observed.data(), 0, ggml_nbytes(restored));
            for (int s = 0; s < slots; ++s) for (int c = 0; c < channels; ++c) for (int h = 0; h < history; ++h) {
                const size_t src = (s * channels + c) * (history + rows) + rows - tail + prefixes[s] + h;
                const size_t dst = (s * channels + c) * history + h;
                t.assert_equal(data[src], observed[dst]);
            }
        }
        t.assert_equal(size_t(history + tail) * channels * slots * sizeof(float), ggml_nbytes(captured));
    }
}

int main(int argc, char ** argv) {
    testing t;
    ggml_backend_load_all();
    if (argc > 2) ggml_backend_load(argv[2]);
    ggml_backend_t backend = argc > 1 ? ggml_backend_init_by_name(argv[1], nullptr) : ggml_backend_cpu_init();
    if (!backend) return 77;
    t.test("bitwise_every_window_unequal_prefixes_repeated_materialization", [&](testing & t) { windows(t, backend); });
    t.test("invalid_prefixes_do_not_modify_descriptors", [](testing & t) {
        int32_t out[6] = {-1,-1,-1,-1,-1,-1};
        t.assert_true(!llama_recurrent_history_indices(3, 4, {1, 5}, out, 6));
        t.assert_true(!llama_recurrent_history_indices(3, 4, {1, 2}, out, 5));
        t.assert_true(!llama_recurrent_history_indices(0, 4, {1, 2}, out, 6));
        for (auto value : out) t.assert_equal(-1, value);
    });
    ggml_backend_free(backend);
    return t.summary();
}
