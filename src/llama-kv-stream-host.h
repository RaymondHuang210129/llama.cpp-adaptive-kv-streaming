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

class llama_kv_stream_host_reservation;
struct llama_kv_stream_host_range;
struct ggml_backend_execution_ops;

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
    static std::shared_ptr<llama_kv_stream_host> from_reservation(
            std::shared_ptr<llama_kv_stream_host_reservation> reservation);

    uint64_t cache_id() const noexcept;
    const llama_kv_stream_host_config & config() const noexcept;
    const ggml_kv_stream_layout & layout() const noexcept;
    size_t stride() const noexcept;
    size_t bytes() const noexcept;
    size_t committed_bytes() const noexcept;
    // Explicit demand-backed registrations; eager buffers are owned by their backend allocator.
    size_t registered_bytes() const noexcept;
    ggml_backend_buffer_type_t buffer_type() const noexcept;
    ggml_backend_buffer_t buffer() const noexcept;
    bool clear(uint8_t value = 0);
    bool demand_backed() const noexcept;
    bool valid() const noexcept;
    bool prepare(size_t rows);
    bool ranges(uint32_t layer, ggml_kv_stream_operand operand, size_t offset, size_t bytes,
            std::vector<llama_kv_stream_host_range> & output) const;
    bool address_ranges(const void * data, size_t bytes, std::vector<llama_kv_stream_host_range> & output) const;
    // Retain this owner and immutable source bytes until backend completion, as for native async uploads.
    bool upload(ggml_backend_t backend, ggml_tensor * target, size_t offset, const void * data, size_t bytes, size_t & calls) const;
    // The execution context must retain this owner until its destruction callback runs.
    ggml_backend_buffer_t execution_buffer(ggml_backend_dev_t device,
            const ggml_backend_execution_ops & ops, void * context);
    // Layer ordinals follow full-attention execution order. Failure leaves output unchanged.
    bool layer(uint32_t index, llama_kv_stream_host_layer & output) const noexcept;

private:
    llama_kv_stream_host() = default;
    llama_kv_stream_host_config cfg;
    ggml_kv_stream_layout planes;
    size_t layer_stride = 0, storage_bytes = 0;
    void * base = nullptr;
    ggml_backend_buffer_t storage = nullptr;
    std::shared_ptr<llama_kv_stream_host_reservation> demand;
    bool access(const ggml_tensor * tensor, size_t offset, size_t bytes, bool write,
            std::vector<llama_kv_stream_host_range> & output);
    bool locate(const void * data, size_t offset, size_t bytes, uint32_t & layer,
            ggml_kv_stream_operand & operand, size_t & in_plane) const noexcept;
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
    const llama_kv_stream_host_config & config() const noexcept;
    ggml_backend_buffer_type_t buffer_type() const noexcept;
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
    // Clear materialized bytes without releasing physical coverage or publishing logical state.
    void clear(uint8_t value = 0);
    // Split at registration boundaries. Each retained buffer keeps its reservation alive for queued users.
    // Failure leaves output unchanged. Device aliases must be resolved per range, never across registrations.
    bool ranges(uint32_t layer, ggml_kv_stream_operand operand, size_t offset, size_t bytes,
            std::vector<llama_kv_stream_host_range> & output) const;

private:
    llama_kv_stream_host_reservation();
    struct impl;
    std::unique_ptr<impl> state;
};
