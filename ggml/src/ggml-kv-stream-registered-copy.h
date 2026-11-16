#pragma once

#include "ggml-backend.h"
#include "ggml-kv-stream.h"

struct ggml_kv_stream_registered_copy_source {
    const void * base = nullptr;
    size_t bytes = 0;
};

// One DMA must remain inside this retained registration, even if the next host address is contiguous.
struct ggml_kv_stream_registered_copy_fragment {
    ggml_backend_buffer_t owner = nullptr;
    size_t offset = 0, bytes = 0;
};

// Optional registry "ggml_backend_kv_stream_registered_copy_ops". Queues use the existing copy_ops lifecycle.
// Caller retains the source address owner while the queue is alive; queues also retain submitted registration buffers.
constexpr size_t ggml_kv_stream_registered_copy_max_sources=256;

struct ggml_kv_stream_registered_copy_ops {
    uint32_t version;
    // A null prepared bank uses the existing explicit native-resource creation path.
    void * (*create)(ggml_backend_t backend, void * prepared, ggml_backend_buffer_t device,
            const ggml_kv_stream_registered_copy_source & source, const ggml_kv_stream_shape & shape, size_t slots);
    // Validate both complete fragment lists before any submission. Actual bytes/calls feed copy_ops::stats.
    bool (*enqueue)(void * queue, size_t first_slot,
            const ggml_kv_stream_registered_copy_fragment * k, size_t nk,
            const ggml_kv_stream_registered_copy_fragment * v, size_t nv,
            size_t live_tokens, size_t padded_tokens, bool feedback_eligible);
    // Version 2: one ring over an explicit set of disjoint authoritative host ranges.
    void * (*create_multi)(ggml_backend_t backend,void * prepared,ggml_backend_buffer_t device,
            const ggml_kv_stream_registered_copy_source * sources,size_t count,
            const ggml_kv_stream_shape & shape,size_t slots);
};

using ggml_kv_stream_registered_copy_ops_get = const ggml_kv_stream_registered_copy_ops * (*)();
