#pragma once

#include "../ggml/src/ggml-backend-memory.h"
#include "llama-memory-phase-owner.h"
#include "llama-memory-bounded-kv.h"

#include <memory>
#include <vector>

struct llama_compute_arena_deleter {
    // Release the arena reference owned by one context binding.
    void operator()(ggml_backend_memory_arena_t arena) const noexcept {
        ggml_backend_memory_arena_free(arena);
    }
};

using llama_compute_arena_ptr = std::unique_ptr<ggml_backend_memory_arena, llama_compute_arena_deleter>;

class llama_compute_reserve_state {
public:
    // Return true until a complete scheduler reservation succeeds.
    bool begin() const noexcept {
        return pending;
    }

    // Mark the current scheduler reservation as complete.
    void complete() noexcept {
        pending = false;
    }

    // Require a new scheduler reservation after graph requirements change.
    void invalidate() noexcept {
        pending = true;
    }

private:
    bool pending = true;
};

struct llama_compute_arena_binding {
    ggml_backend_buffer_type_t buft = nullptr;
    size_t capacity = 0;
    size_t first_slot = 0;
    llama_compute_arena_ptr arena;
};


struct llama_compute_workspace_plan {
    std::vector<ggml_backend_memory_workspace_group> groups;
    // Phase-major sizes, with one entry per canonical buffer-type group.
    std::vector<std::vector<size_t>> phase_sizes;
};

// Preserve each measured phase while retaining the maximum-sized canonical group metadata.
// Failure leaves output unchanged.
bool llama_compute_workspace_plan_make(
        const std::vector<ggml_backend_buffer_type_t> & bufts,
        const std::vector<size_t> & measurements,
        size_t n_phases,
        llama_compute_workspace_plan & output);
// Attach arena-backed groups atomically and leave unsupported groups on scheduler allocation.
bool llama_prepare_compute_arena_bindings(
        ggml_backend_sched_t sched,
        const std::vector<ggml_backend_t> & backends,
        const std::vector<ggml_backend_memory_workspace_group> & groups,
        std::vector<llama_compute_arena_binding> & bindings);

// Target graph/transport envelope only. Native current/SWA and auxiliary cache grants are separate.
struct llama_context_group_workload_quote {
    llama_memory_group_workload_measurement measurement;
    ggml_backend_buffer_type_t buffer_type = nullptr;
    size_t writer_bytes = 0, attention_bytes = 0, encoded_stage_bytes = 0;
    size_t publication_bytes = 0, pool_minimum_bytes = 0, alignment_slack_bytes = 0;
    size_t host_compute_bytes = 0, host_descriptor_bytes = 0;
};

// Metadata only. A successful byte quote does not activate an execution context.
struct llama_context_group_plan {
    llama_context_group_workload_quote target;
    std::vector<llama_context_group_workload_quote> catalog;
    std::vector<llama_memory_cache_requirement> caches;
    std::vector<llama_memory_cache_allocation_measurement> cache_allocations;
    llama_memory_bounded_kv_layout bounded;
    size_t current_state_bytes = 0;
    llama_memory_workload_budget budget;
};
