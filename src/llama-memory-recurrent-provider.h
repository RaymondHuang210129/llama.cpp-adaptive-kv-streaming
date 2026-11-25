#pragma once

#include <cstddef>
#include <cstdint>
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
