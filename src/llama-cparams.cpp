#include "llama-cparams.h"

size_t llama_max_parallel_sequences(void) {
    return LLAMA_MAX_SEQ;
}

bool llama_cparams::set_stream_limits(uint32_t logical_tokens) noexcept {
    if (!kv_streaming()) {
        stream_limits = {};
        return true;
    }
    if (!n_seq_max || n_seq_max > LLAMA_MAX_SEQ || (n_seq_max > 1 && (!shared_device_memory_bytes || !kv_unified)) ||
            !logical_tokens || logical_tokens > n_ctx) return false;
    llama_kv_stream_context_limits limits;
    // Keep serial padding semantics. Multi-slot padding does not grant logical growth.
    const uint32_t logical = n_seq_max == 1 ? n_ctx : logical_tokens;
    if (!llama_kv_stream_context_limits_make(logical, n_ctx, n_seq_max, limits)) return false;
    stream_limits = limits;
    n_ctx_seq = logical;
    return true;
}
