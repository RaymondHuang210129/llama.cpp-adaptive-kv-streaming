#pragma once

#include "llama-memory-recurrent.h"

struct llama_recurrent_compact_graph_storage {
    ggml_tensor * history;
    ggml_tensor * matrix;
    ggml_tensor * log;
    ggml_tensor * timeline;
    ggml_tensor * destinations;
    ggml_tensor * slots;
};

void llama_recurrent_compact_capture_history(ggml_context * ctx,ggml_cgraph * graph,ggml_tensor * input,
        uint32_t history,uint32_t depth,const llama_recurrent_compact_graph_storage & storage);
std::pair<ggml_tensor *,ggml_tensor *> llama_recurrent_compact_capture_matrix(ggml_context * ctx,ggml_cgraph * graph,
        ggml_tensor * q,ggml_tensor * k,ggml_tensor * v,ggml_tensor * g,ggml_tensor * beta,ggml_tensor * state,
        uint32_t depth,const llama_recurrent_compact_graph_storage & storage);

// Payloads remain in the recurrent owner's buffers. This object owns metadata only.
struct llama_memory_recurrent::compact_storage_state {
    struct sequence {
        llama_seq_id logical = -1;
        int32_t alias = -1;
        uint32_t role = 0;
        uint32_t working = 1;
        uint32_t rows = 0;
        uint64_t generation = 0;
        bool fresh = true;
        bool valid = false;
        bool pending = false;
        bool rolled = false;
    };
    std::vector<sequence> sequences;
    std::vector<int32_t> slots;
    std::vector<ggml_tensor *> logs;
    std::vector<ggml_tensor *> timelines;
    ggml_backend_t backend = nullptr; // borrowed from the live context
    uint64_t generation = 0;
    std::vector<uint32_t> active;
    bool capturing = false;
    bool prefill = false;
    bool failed = false;
    bool poisoned = false;
};
