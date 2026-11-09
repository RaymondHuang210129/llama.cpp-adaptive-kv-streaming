#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

struct llama_kv_stream_selection_config {
    size_t context_tokens = 0;
    size_t maximum_slots = 0;
};

struct llama_kv_stream_selection_slot {
    uint32_t id = 0;
    size_t history_tokens = 0;
    size_t minimum_growth = 1;
    uint64_t compatibility = 0;
    bool eligible = true;
};

struct llama_kv_stream_selection_entry {
    uint32_t id = 0;
    size_t history_tokens = 0;
    size_t growth_tokens = 0;
    size_t offset_tokens = 0;
};

struct llama_kv_stream_selection {
    size_t context_tokens = 0;
    size_t history_tokens = 0;
    std::vector<llama_kv_stream_selection_entry> slots;
};

enum class llama_kv_stream_selection_status {
    success, idle, no_progress, invalid_config, allocation_failed,
};

// Pure admission of a complete candidate; it must not mutate resources or request state.
using llama_kv_stream_selection_admit = bool (*)(const llama_kv_stream_selection &, void *);

// Plan only the admitted group, not the pending request queue. Prefer cardinality, then
// smallest history, then slot order. Offsets and growth are logical, not device placement.
// All non-success results leave output unchanged. No request/output counters are modified.
llama_kv_stream_selection_status llama_kv_stream_select(
        const llama_kv_stream_selection_config & config,
        const std::vector<llama_kv_stream_selection_slot> & admitted,
        llama_kv_stream_selection & output,
        llama_kv_stream_selection_admit admit = nullptr, void * user_data = nullptr);
