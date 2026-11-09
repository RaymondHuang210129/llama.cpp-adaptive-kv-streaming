#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

struct llama_kv_stream_context_limits {
    uint32_t slot_limit = 0, group_limit = 0;
    uint32_t slot_storage = 0, history_storage = 0, slots = 0;
};

// Native padding is storage only. Failure preserves output; ordinary context semantics are not changed here.
bool llama_kv_stream_context_limits_make(size_t logical_tokens, size_t padded_tokens, size_t slots,
        llama_kv_stream_context_limits & output) noexcept;

struct llama_kv_stream_generation_budget {
    uint32_t query_rows = 0, proposal_tokens = 0;
};

// The old sampled anchor already occupies logical history. Acceptance can emit proposals plus one bonus/correction.
bool llama_kv_stream_generation_budget_make(size_t headroom, uint32_t maximum_proposals, size_t remaining_outputs,
        llama_kv_stream_generation_budget & output) noexcept;

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

struct llama_kv_stream_page_minimum {
    uint32_t slot = 0, pages = 0;
};

struct llama_kv_stream_page_partition_slot {
    uint32_t slot = 0, first_page = 0, pages = 0;
    size_t logical_offset = 0, projected_tokens = 0;
};

struct llama_kv_stream_page_partition {
    uint32_t pages = 0;
    std::vector<llama_kv_stream_page_partition_slot> slots;
};

enum class llama_kv_stream_page_partition_status {
    success, invalid_selection, invalid_minimum, insufficient_pages, allocation_failed,
};

struct llama_kv_stream_page_partition_result {
    llama_kv_stream_page_partition_status status = llama_kv_stream_page_partition_status::success;
    uint64_t minimum_pages = 0;
};

// One homogeneous owning-KV page unit; aliases share their owner. Caller supplies measured native minima.
// Whole pages are exclusive until the next selection.
// Input is a fresh selection snapshot; live consumption cursors stay with their slot owners.
// Weight by history plus logical growth, clamp minima, then round remainders by slot ID. No GPU allocation.
// Logical offsets are not page indices. Failure preserves output; this does not prove kernel admission.
llama_kv_stream_page_partition_result llama_kv_stream_partition_pages(
        const llama_kv_stream_selection & selected, uint32_t pages,
        const std::vector<llama_kv_stream_page_minimum> & minima, llama_kv_stream_page_partition & output);
