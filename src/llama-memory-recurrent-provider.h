#pragma once

#include <cstddef>
#include <cstdint>
#include <array>
#include <limits>
#include <vector>

enum class llama_recurrent_representation { dense_device, dense_host, gdn_fp32 };

// FP32 scalar-decay GDN geometry, supplied by the operator, not a model name.
struct llama_recurrent_provider_layer {
    size_t width = 0;
    size_t value_heads = 0;
    size_t key_heads = 0;
    size_t conv_channels = 0;
    size_t history = 0;
};

struct llama_recurrent_provider_capabilities {
    bool shared_history = false;
    bool fp32_capture = false;
    bool exact_replay = false;
};

struct llama_recurrent_provider_request {
    std::vector<llama_recurrent_provider_layer> layers;
    uint32_t depth = 0;
    uint32_t selected = 0;
    uint32_t configured = 0;
    size_t alignment = 1;
    // Measured by the graph/phase owner; includes graph outputs and replay scratch.
    size_t graph_workspace = 0;
    size_t device_budget = std::numeric_limits<size_t>::max();
};

struct llama_recurrent_provider_requirements {
    size_t device_persistent = 0;
    size_t host_persistent = 0;
    size_t device_staging = 0;
    size_t device_peak = 0;
};

// Describes payloads and their simultaneous peak, without allocation or policy changes.
// The owner must pass its real graph quote and charge other live phase resources too.
// A false return leaves output unchanged; unsupported providers retain dense fallback.
bool llama_recurrent_provider_describe(const llama_recurrent_provider_request & request,
        llama_recurrent_representation representation, llama_recurrent_provider_capabilities capabilities,
        llama_recurrent_provider_requirements & output) noexcept;

// The owner supplies a complete simultaneous quote, including other live resources.
struct llama_recurrent_placement_plan {
    llama_recurrent_representation representation = llama_recurrent_representation::dense_device;
    size_t peak_bytes = 0;
    size_t h2d_bytes = 0;
    size_t kv_stream_bytes = 0;
    size_t d2h_bytes = 0;
    double compute_ms = 0;
    double restore_ms = 0; // Includes the expected partial-acceptance frequency.
};

struct llama_recurrent_placement_request {
    std::vector<llama_recurrent_placement_plan> plans;
    size_t budget = 0, reserve = 0;
    double h2d_bytes_per_ms = 20.0 * 1024 * 1024;
    double d2h_bytes_per_ms = 20.0 * 1024 * 1024;
    double hysteresis = .05;
    uint64_t generation = 0, workload = 0;
    uint64_t cooldown = 4;
    bool safe_boundary = false;
    // Only qualified overlap may hide transfer time. Zero is the conservative seed.
    double transfer_overlap_fraction = 0;
};

struct llama_recurrent_placement_state {
    bool chosen = false;
    llama_recurrent_representation current = llama_recurrent_representation::dense_device;
    uint64_t last_generation = 0, last_switch_generation = 0, workload = 0;
    std::array<uint32_t,3> samples{}, next_sample{};
    std::array<uint64_t,3> sample_workloads{};
    std::array<std::array<double,8>,3> ratios{};
};

struct llama_recurrent_placement_choice {
    llama_recurrent_representation representation = llama_recurrent_representation::dense_device;
    double predicted_ms = 0, adjusted_ms = 0;
    size_t peak_bytes = 0;
};

// Failure changes neither the policy state nor output. Selection never allocates storage.
bool llama_recurrent_placement_choose(const llama_recurrent_placement_request & request,
        llama_recurrent_placement_state & state, llama_recurrent_placement_choice & output) noexcept;
bool llama_recurrent_placement_observe(llama_recurrent_placement_state & state,
        llama_recurrent_representation representation, uint64_t workload,
        double predicted_ms, double observed_ms) noexcept;
