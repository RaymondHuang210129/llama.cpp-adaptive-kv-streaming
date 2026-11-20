#pragma once

#include "llama-memory-executor.h"
#include "llama-memory-phase.h"
#include "llama-memory-plan.h"
#include "llama-context-workspace.h"
#include <algorithm>
#include <functional>
#include <utility>

// Native recurrent graphs have equal rows per sequence; unified attention graphs use flat rows.
struct llama_context_group_graph_profile {
    std::vector<uint32_t> rows;
    std::vector<uint32_t> outputs;
};

inline bool llama_context_group_graph_profiles(uint32_t selected, uint32_t width, uint32_t maximum,
        uint32_t output_maximum, uint32_t outputs_per_sequence, bool recurrent,
        const std::function<bool(const llama_context_group_graph_profile &)> & visit) {
    if (!selected || selected > uint32_t(llama_max_parallel_sequences()) || !width || !maximum || maximum > INT32_MAX || !outputs_per_sequence || !visit) return false;
    const auto emit = [&](std::vector<uint32_t> rows) {
        uint32_t available = 0;
        for (auto count : rows) available += std::min(count, outputs_per_sequence);
        for (uint32_t count = 0; count <= std::min(available, output_maximum); ++count) {
            llama_context_group_graph_profile profile{rows, std::vector<uint32_t>(selected)};
            auto remaining = count;
            for (uint32_t i = 0; i < selected; ++i) {
                profile.outputs[i] = std::min(remaining, std::min(rows[i], outputs_per_sequence));
                remaining -= profile.outputs[i];
            }
            if (!visit(profile)) return false;
        }
        return true;
    };
    if (recurrent) {
        for (uint32_t rows = 1; rows <= std::min(width, maximum); ++rows) {
            for (uint32_t active = 1; active <= std::min(selected, maximum / rows); ++active) {
                std::vector<uint32_t> counts(selected);
                std::fill_n(counts.begin(), active, rows);
                if (!emit(std::move(counts))) return false;
            }
        }
    } else {
        const auto limit = uint32_t(std::min(uint64_t(maximum), uint64_t(selected) * width));
        for (uint32_t total = 1; total <= limit; ++total) {
            std::vector<uint32_t> counts(selected);
            auto remaining = total;
            for (auto & count : counts) { count = std::min(remaining, width); remaining -= count; }
            if (!emit(std::move(counts))) return false;
        }
    }
    return true;
}

// Keep bounded storage distinct from every owning cache in this operation; retain legacy IDs when available.
inline bool llama_context_group_bounded_identity(const std::vector<llama_memory_phase_cache_identity> & sources,
        const std::vector<llama_memory_phase_cache_identity> & auxiliary,llama_memory_phase_cache_identity & output) {
    const auto maximum=size_t(llama_max_parallel_sequences());
    if (sources.empty() || sources.size()>maximum || auxiliary.size()>maximum) return false;
    for (const auto * list : {&sources,&auxiliary})
        for (const auto & cache : *list) if (!cache.id || !cache.generation) return false;
    uint64_t candidate=sources.front().id<=(UINT64_MAX-1)/4 ? sources.front().id*4+1 : 1;
    for (size_t attempt=0;attempt<=sources.size()+auxiliary.size();++attempt) {
        bool used=false;
        for (const auto * list : {&sources,&auxiliary})
            for (const auto & cache : *list) used=used || cache.id==candidate;
        if (!used) {output={candidate,1};return true;}
        candidate=candidate==UINT64_MAX ? 1 : candidate+1;
    }
    return false;
}

// Native state banks grow monotonically with selected count. This bound does not admit graph workspaces or attention.
inline bool llama_context_group_state_limit(uint32_t maximum,size_t capacity,
        const std::function<bool(uint32_t,size_t &)> & measure,uint32_t & output) {
    if (!maximum || maximum>uint32_t(llama_max_parallel_sequences()) || !capacity || !measure) return false;
    uint32_t low=0,high=maximum+1;
    while (high-low>1) {
        const auto count=low+(high-low)/2;size_t bytes=0;
        if (!measure(count,bytes)) return false;
        if (bytes<=capacity) low=count;
        else high=count;
    }
    output=low;return true;
}

class llama_kv_stream_model;
struct llama_memory_bounded_kv_layout;
class llama_context_memory;
class llama_memory_workspace;
struct llama_memory_workspace_group;

// Return a persistent cache region after its stock tensors and pending work have been released.
class llama_context_memory_loan {
public:
    LLAMA_API ~llama_context_memory_loan();
    llama_context_memory_loan(const llama_context_memory_loan &) = delete;
    llama_context_memory_loan & operator=(const llama_context_memory_loan &) = delete;
    LLAMA_API ggml_backend_memory_lease_t lease() const noexcept;
private:
    friend class llama_context_memory;
    explicit llama_context_memory_loan(std::weak_ptr<llama_context_memory *> owner) : owner(std::move(owner)) {}
    std::weak_ptr<llama_context_memory *> owner;
    ggml_backend_memory_lease_t storage = nullptr;
    uint64_t resource = 0;
};

struct llama_context_memory_diagnostics {
    llama_memory_text_phase phase = llama_memory_text_phase::unspecified;
    size_t parent_bytes = 0, workspace_bytes = 0, kv_pool_bytes = 0;
    size_t kv_writer_bytes = 0, kv_attention_bytes = 0, unused_bytes = 0;
    size_t reclaimed_workspace_bytes = 0;
    uint64_t arena_generation = 0, transition_count = 0, layout_revision = 0;
    uint64_t last_transition_us = 0;
    uint32_t resident_pages_per_layer = 0, ring_slots = 0, active_pages = 0;
    size_t last_copy_bytes = 0, last_copy_calls = 0;
    double last_copy_ms = 0, last_elapsed_ms = 0;
    bool streaming_active = false;
    bool executable_storage_external = true;
    bool kv_device_suspended = false;
    size_t borrowed_phase_bytes = 0;
    size_t persistent_cache_bytes = 0;
    size_t publication_stage_bytes = 0;
};

enum class llama_context_memory_create_status {
    success, invalid_configuration, insufficient_budget, allocation_failed, capture_failed, transition_failed,
};

struct llama_context_memory_create_result {
    llama_context_memory_create_status status = llama_context_memory_create_status::invalid_configuration;
    size_t available_bytes = 0, required_bytes = 0, additional_bytes = 0;
    llama_memory_stage_id stage = 0;
    llama_memory_resource_id resource = 0;
};

const char * llama_context_memory_create_status_name(llama_context_memory_create_status status) noexcept;

// One serial scheduler lifetime; native caches and arena leases retire before scheduler destruction.
// The caller owns the scheduler/backends and must not submit or mutate their graph caches concurrently.
// Callbacks may retry admission, but must not destroy owners or their schedulers.
class llama_context_memory {
public:
    // Unknown native-cache lifetimes keep the existing milestone-3 allocation path.
    static bool supported(const std::vector<ggml_backend_t> & backends);

    // Allocate exactly the existing measured group maxima; leave non-view groups scheduler-owned.
    static std::unique_ptr<llama_context_memory> create(ggml_backend_sched_t sched,
            const std::vector<ggml_backend_t> & backends,
            const std::vector<ggml_backend_memory_workspace_group> & groups);
    static std::unique_ptr<llama_context_memory> create(ggml_backend_sched_t sched,
            const std::vector<ggml_backend_t> & backends,
            const llama_compute_workspace_plan & plan);
    // Optional capture hook runs before a new binding snapshot; false or an exception rejects publication without device work.
    static std::unique_ptr<llama_context_memory> create(ggml_backend_sched_t sched,
            const std::vector<ggml_backend_t> & backends,
            const llama_compute_workspace_plan & plan, llama_kv_stream_model * stream,
            llama_context_memory * serial_parent = nullptr, bool suspended_workspace = false,
            const std::function<bool()> & before_capture = {}, size_t serial_parent_bytes = 0,
            llama_context_memory_create_result * result = nullptr);
    // The common group coordinator supplies grants and gates the complete model submission.
    // This factory creates no parent or backend workspace allocation.
    static std::unique_ptr<llama_context_memory> create_external_workspace(ggml_backend_sched_t sched,
        const std::vector<ggml_backend_t> & backends, std::vector<llama_memory_workspace_group> groups,
        const std::function<bool()> & submission_allowed, const std::function<bool()> & before_capture = {});
    llama_memory_workspace * external_workspace() noexcept;
    bool capture_external_workspace() noexcept;
    // Reserve a fixed cache suffix inside the declared parent, not a second backend allocation.
    LLAMA_API std::unique_ptr<llama_context_memory_loan> reserve_bounded_cache(const llama_memory_bounded_kv_layout & layout);
    // Quote all phase minima before mutation; required_bytes describes a restart with movable fixed suffixes.
    LLAMA_API std::unique_ptr<llama_context_memory_loan> reserve_bounded_cache(const llama_memory_bounded_kv_layout & layout,
        llama_context_memory_create_result * admission);
    // Stable publication staging stays disjoint from both graph scratch and elastic KV.
    LLAMA_API std::unique_ptr<llama_context_memory_loan> reserve_staging(size_t bytes);
    // External leases prevent cleanup; the parent remains usable with the fixed region until they return.
    LLAMA_API bool release_bounded_cache() noexcept;
    LLAMA_API bool has_bounded_cache() const noexcept;
    // A graph rebuild can keep ownership only when its measured catalog fits every existing phase grant.
    LLAMA_API bool covers_workspace(const llama_compute_workspace_plan & plan) const noexcept;
    // Grow the elastic target prefix before attaching serial children; fixed caches and the parent allocation stay in place.
    LLAMA_API bool grow_workspace(const llama_compute_workspace_plan & plan,
        llama_context_memory_create_result * result = nullptr) noexcept;
    // Borrow measured scratch, or the full parent after explicit KV suspension. Return grants before resume.
    LLAMA_API static std::unique_ptr<llama_context_memory> borrow_workspace(ggml_backend_sched_t sched,
            const std::vector<ggml_backend_t> & backends,
            const std::vector<ggml_backend_memory_workspace_group> & groups,
            llama_context_memory & parent);
    LLAMA_API ~llama_context_memory();
    llama_context_memory(const llama_context_memory &) = delete;
    llama_context_memory & operator=(const llama_context_memory &) = delete;

    // Reuse one conservative pin for this immutable scheduler lifetime; retirement drains before releasing it.
    LLAMA_API ggml_status compute_async(ggml_cgraph * graph);
    // Observe completion without rebuilding the immutable lease-validation state on the next token.
    LLAMA_API void synchronize();
    // Drain the other serial scheduler before either context rewrites shared scratch.
    bool prepare_serial_target() noexcept;
    bool prepare_serial_draft(llama_memory_text_phase phase) noexcept;
    // Make the decode KV grant available before a serial consumer acquires its ring lease.
    bool prepare_serial_decode() noexcept;
    LLAMA_API bool prepare_serial_consumer(llama_memory_text_phase phase) noexcept;
    // Retire native graph addresses before the caller replaces graph metadata.
    LLAMA_API bool retire_graph() noexcept;
    LLAMA_API bool serial_ready() const noexcept;
    // Return KV and graph grants while retaining the shared parent and authoritative host state.
    LLAMA_API bool suspend_kv(llama_memory_executor_backend * auxiliary_completion = nullptr) noexcept;
    LLAMA_API bool kv_device_suspended() const noexcept;
    // A failed reverse transition closes this owner permanently; recreate the context to recover.
    LLAMA_API bool valid() const noexcept;
    // Metadata-only preflight preserves current fixed regions; the restart minimum repacks their suffix.
    LLAMA_API bool quote_suspended_loans(const std::vector<size_t> & bytes,
        llama_context_memory_create_result & result) const noexcept;
    // Loan disjoint regions of a suspended parent. Output owns lease handles; all must return before resume.
    LLAMA_API bool lend_suspended(const std::vector<size_t> & bytes,
        std::vector<ggml_backend_memory_lease_t> & output) noexcept;
    // Acquire a fresh measured layout for a suspended target; ordinary execution never resumes implicitly.
    // Optional rebuild runs under the submission gate and must not execute graphs or change persistent data.
    LLAMA_API bool resume_kv(llama_memory_text_phase phase, const std::function<bool()> & rebuild = {}) noexcept;

    // Record text intent and activate the matching shared-parent layout only when the phase changes.
    llama_memory_text_phase_result signal_text_phase(const llama_memory_text_phase_signal & signal) noexcept;
    llama_memory_text_phase_snapshot text_phase() const noexcept;
    bool uses_arenas() const noexcept;
    // Borrowed handles; consumers retain them before capturing addresses from this workspace.
    LLAMA_API const std::vector<ggml_backend_memory_lease_t> & workspace_leases() const noexcept;
    LLAMA_API bool shares_kv_memory() const noexcept;
    LLAMA_API bool has_speculative_consumer() const noexcept;
    // Declare an existing serial child that only reads its parent's committed cache.
    LLAMA_API bool register_shared_read_consumer() noexcept;
    // Declare a serial writer of the parent's existing persistent bounded-cache region.
    LLAMA_API bool register_bounded_write_consumer(ggml_backend_memory_lease_t storage) noexcept;
    // Suspend only registered auxiliary, bounded-writer or shared-reader arrangements.
    LLAMA_API bool can_suspend_for_vision() const noexcept;
    LLAMA_API bool borrows_serial_parent() const noexcept;
    LLAMA_API ggml_backend_buffer_t shared_parent() const noexcept;
    size_t shared_parent_capacity() const noexcept;
    uint64_t shared_arena_generation() const noexcept;
    uint64_t phase_transition_count() const noexcept;
    // Changes when graph bindings are replaced, including recovery after a failed transition.
    uint64_t graph_binding_revision() const noexcept;
    LLAMA_API bool diagnostics(llama_context_memory_diagnostics & output) const noexcept;

private:
    friend class llama_context_memory_loan;
    std::unique_ptr<llama_context_memory_loan> reserve_region(size_t bytes,const llama_memory_bounded_kv_layout * layout,
        llama_context_memory_create_result * admission = nullptr);
    bool release_staging() noexcept;
    bool release_region(bool cache) noexcept;
    struct implementation;
    explicit llama_context_memory(std::unique_ptr<implementation> impl);
    std::unique_ptr<implementation> impl;
};

// Internal serial adapter seam; the context retains ownership and may rebuild this coordinator.
LLAMA_API llama_context_memory * llama_context_compute_memory(llama_context * ctx) noexcept;
LLAMA_API bool llama_context_suspend_kv_device(llama_context * ctx) noexcept;
LLAMA_API bool llama_context_resume_kv_device(llama_context * ctx, llama_memory_text_phase phase) noexcept;
