#pragma once

#include "../ggml/src/ggml-kv-stream.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <functional>
#include <memory>
#include <vector>

struct llama_kv_stream_host_config {
    uint64_t cache_id = 0;
    ggml_kv_stream_shape shape;
    ggml_kv_stream_capabilities capabilities;
    size_t context_tokens = 0;
    uint32_t layers = 0;
};

struct llama_kv_stream_host_layer {
    void * k = nullptr;
    void * v = nullptr;
};

// Optional CUDA adapters. Registration retains the caller's buffer until after unregistration.
// Neither adapter falls back to pageable storage or changes an existing registration's ownership.
// The owner must keep its bytes alive; do not externally unregister a successful wrapper.
ggml_backend_buffer_type_t llama_kv_stream_host_buffer_type(ggml_backend_dev_t device);
ggml_backend_buffer_t llama_kv_stream_host_register(ggml_backend_dev_t device, ggml_backend_buffer_t owner);

// Authoritative host bytes, independent of device leases. Queued users must retain this owner until completion.
// Raw access has no dirty tracking yet; callers must serialize writes against copies/device work.
class llama_kv_stream_host {
public:
    ~llama_kv_stream_host();
    llama_kv_stream_host(const llama_kv_stream_host &) = delete;
    llama_kv_stream_host & operator=(const llama_kv_stream_host &) = delete;

    // Allocate and zero the canonical planes. expected_type must promise the required host allocation class.
    static std::shared_ptr<llama_kv_stream_host> create(
            const llama_kv_stream_host_config & config, ggml_backend_buffer_type_t expected_type);
    // Retain existing storage without clearing its contents; reject a different buffer type or insufficient space.
    static std::shared_ptr<llama_kv_stream_host> from_buffer(
            const llama_kv_stream_host_config & config, ggml_backend_buffer_type_t expected_type, ggml_backend_buffer_t buffer);

    uint64_t cache_id() const noexcept;
    const llama_kv_stream_host_config & config() const noexcept;
    const ggml_kv_stream_layout & layout() const noexcept;
    size_t stride() const noexcept;
    size_t bytes() const noexcept;
    ggml_backend_buffer_t buffer() const noexcept;
    // Layer ordinals follow full-attention execution order. Failure leaves output unchanged.
    bool layer(uint32_t index, llama_kv_stream_host_layer & output) const noexcept;

private:
    llama_kv_stream_host() = default;
    llama_kv_stream_host_config cfg;
    ggml_kv_stream_layout planes;
    size_t layer_stride = 0, storage_bytes = 0;
    void * base = nullptr;
    ggml_backend_buffer_t storage = nullptr;
};

struct llama_kv_stream_host_range {
    ggml_backend_buffer_ptr buffer;
    size_t offset = 0, bytes = 0;
};

// Demand-backed alternative for multi-slot histories. The legacy eager allocator is unchanged.
// This object has no whole-reservation buffer: only committed, registered ranges can be borrowed.
class llama_kv_stream_host_reservation {
public:
    using register_range = std::function<ggml_backend_buffer_t(ggml_backend_buffer_t)>;
    ~llama_kv_stream_host_reservation();
    llama_kv_stream_host_reservation(const llama_kv_stream_host_reservation &) = delete;
    llama_kv_stream_host_reservation & operator=(const llama_kv_stream_host_reservation &) = delete;

    static std::unique_ptr<llama_kv_stream_host_reservation> create(
            const llama_kv_stream_host_config & config, ggml_backend_dev_t device, size_t chunk_bytes = 1024*1024);
    // The backend registrar must retain its input until unregistration. Never adopt a pageable fallback.
    static std::unique_ptr<llama_kv_stream_host_reservation> create(
            const llama_kv_stream_host_config & config, ggml_backend_buffer_type_t expected_type,
            register_range registrar, size_t chunk_bytes);

    const ggml_kv_stream_layout & layout() const noexcept;
    size_t bytes() const noexcept;
    size_t reserved_bytes() const noexcept;
    size_t committed_bytes() const noexcept;
    size_t registered_bytes() const noexcept;
    // Address metadata only, including uncommitted planes; these are CPU addresses, not CUDA aliases.
    bool layer(uint32_t index, llama_kv_stream_host_layer & output) const noexcept;
    // Physical row coverage, including native padding; does not publish a logical frontier/generation.
    // Serialize mutation against all readers/writers. Failed growth preserves prior coverage and bytes.
    bool prepare(size_t rows);
    bool truncate(size_t rows);
    void clear();
    // Split at registration boundaries. Each retained buffer keeps its reservation alive for queued users.
    // Failure leaves output unchanged. Device aliases must be resolved per range, never across registrations.
    bool ranges(uint32_t layer, ggml_kv_stream_operand operand, size_t offset, size_t bytes,
            std::vector<llama_kv_stream_host_range> & output) const;

private:
    llama_kv_stream_host_reservation();
    struct impl;
    std::unique_ptr<impl> state;
};
