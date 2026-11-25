#pragma once

#include "ggml.h"
#include <cstddef>
#include <cstdint>
#include <vector>

// Graph-owned tensors only: allocation/liveness remains with the existing graph owner.
// Input is [previous history + new rows, channels, selected sequences].
// Retain the last rows and their immediately preceding history once, without K copied windows.
ggml_tensor * llama_recurrent_history_capture(ggml_context * ctx, ggml_tensor * input,
        uint32_t history, uint32_t rows);

// Prefix zero means abort to the checkpoint; zero accepted proposals normally uses prefix one.
// Descriptors are [history, selected sequences], validated in host RAM before upload.
bool llama_recurrent_history_indices(uint32_t history, uint32_t rows,
        const std::vector<uint32_t> & prefixes, int32_t * output, size_t capacity) noexcept;

// Indices must come from the validated descriptor builder above. Output is [history, channels, sequences].
ggml_tensor * llama_recurrent_history_restore(ggml_context * ctx, ggml_tensor * timeline, ggml_tensor * indices);
