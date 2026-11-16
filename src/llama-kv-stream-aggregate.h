#pragma once
#include "llama-kv-stream-policy.h"

struct llama_kv_stream_aggregate_requirements {
    size_t active_tokens=0;
    std::vector<size_t> role_tokens;
    std::vector<size_t> owner_first;
    std::vector<size_t> owner_pages;
};
struct llama_kv_stream_aggregate_layout {
    llama_kv_stream_aggregate_requirements requirements;
    llama_kv_stream_policy_layout storage;
    // Request-major owners, then cache-local layers within each owner.
    std::vector<std::vector<llama_kv_stream_policy_layer>> owners;
};

// Count storage padding at fragment boundaries; the scalar policy sees only the combined extent.
bool llama_kv_stream_aggregate_requirements_make(const std::vector<uint32_t> & role_layers,
        const std::vector<size_t> & owner_tokens,llama_kv_stream_aggregate_requirements & output);
// Slice the scalar policy's resident planes without moving authoritative host histories.
bool llama_kv_stream_aggregate_layout_make(const llama_kv_stream_policy_config & config,
        const llama_kv_stream_policy_state & state,const std::vector<uint32_t> & role_layers,
        const std::vector<size_t> & owner_tokens,llama_kv_stream_aggregate_layout & output);
// Prefer complete residency for named roles when the existing quota and ring permit it.
bool llama_kv_stream_aggregate_layout_make(const llama_kv_stream_policy_config & config,
        const llama_kv_stream_policy_state & state,const std::vector<uint32_t> & role_layers,
        const std::vector<size_t> & owner_tokens,llama_kv_stream_aggregate_layout & output,
        const std::vector<uint32_t> & resident_roles);
