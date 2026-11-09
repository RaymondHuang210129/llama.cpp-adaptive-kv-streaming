#pragma once

#include "llama-kv-stream-content.h"
#include "llama-kv-stream-writer.h"
#include "llama-kv-stream-publication.h"
#include "llama-kv-stream-selection.h"

#include <functional>

class llama_kv_cells;

struct llama_kv_stream_logical_identity {
    uint64_t id = 0;
    uint64_t generation = 0;
};

struct llama_kv_stream_logical_checkpoint {
    uint64_t id = 0;
    uint64_t generation = 0;
    size_t tokens = 0;
};

// Owns one authoritative cache and its ordered K/V publication frontier.
class llama_kv_stream_logical_cache {
public:
    static std::unique_ptr<llama_kv_stream_logical_cache> create(std::shared_ptr<llama_kv_stream_host> host);
    ~llama_kv_stream_logical_cache();

    std::shared_ptr<llama_kv_stream_host> host() const noexcept;
    std::shared_ptr<llama_kv_stream_content> content() const noexcept;
    llama_kv_stream_logical_identity identity() const noexcept;
    llama_kv_stream_logical_checkpoint checkpoint() const noexcept;
    llama_kv_stream_publication_frontiers frontiers() const noexcept;
    size_t tokens() const noexcept;
    // No pending publication or unacknowledged host replacement.
    bool ready() const noexcept;

    // One append is active at a time. Host publication does not claim a device mirror.
    bool begin(size_t count);
    bool publish_host(llama_kv_stream_write & write);
    using generated_stage = std::function<bool(bool value, const ggml_tensor * encoded, size_t row, size_t count)>;
    bool begin_generated(std::shared_ptr<llama_kv_stream_writer> writer, const ggml_tensor * k, const ggml_tensor * v,
            const generated_stage & stage = {});
    bool complete_generated();
    bool finish();
    // Cancel only before host publication and after queued backend work is drained.
    bool cancel();
    bool truncate(size_t tokens);
    // Caller restores saved host bytes before invoking this method.
    bool restore(const llama_kv_stream_logical_checkpoint & checkpoint);

private:
    explicit llama_kv_stream_logical_cache(std::shared_ptr<llama_kv_stream_host> host);
    bool complete(llama_kv_stream_publication_domain domain);
    bool covers(const llama_kv_stream_write & write) const noexcept;
    bool drain_generated();
    bool reset_frontier(size_t tokens, bool suffix = false);

    std::shared_ptr<llama_kv_stream_host> backing;
    std::shared_ptr<llama_kv_stream_content> authoritative;
    std::unique_ptr<llama_kv_stream_publications> publications;
    llama_kv_stream_publication_ticket pending;
    size_t committed = 0;
    uint64_t expected_generation = 0;
    bool host_ready = false;
    bool failed = false;
    std::shared_ptr<llama_kv_stream_writer> active_writer;
    llama_kv_stream_writer_completion generated[2];
    llama_kv_stream_write generated_write;
};

struct llama_kv_stream_slot_handle {
    uint32_t slot = 0;
    uint64_t generation = 0;
};

// Owner-thread metadata for stable canonical histories. Read aliases resolve through their source.
// The common memory owner must retire GPU users before advancing a request namespace.
class llama_kv_stream_slot_table {
public:
    static std::unique_ptr<llama_kv_stream_slot_table> create(const llama_kv_stream_context_limits & limits);
    // Bind once at startup. A request change never replaces these stable host addresses.
    bool bind(uint32_t slot, std::vector<std::shared_ptr<llama_kv_stream_logical_cache>> caches,
            llama_kv_stream_slot_handle & output);
    bool get(const llama_kv_stream_slot_handle & handle, uint32_t role,
            std::shared_ptr<llama_kv_stream_logical_cache> & output) const noexcept;
    // Owner-thread native cells keep their original positions/sequence IDs; revalidate handles before execution.
    bool get_cells(const llama_kv_stream_slot_handle & handle, uint32_t role,
            std::shared_ptr<llama_kv_cells> & output) const noexcept;
    // Pending is an accepted next-input anchor, not unaccepted draft rows or network buffering.
    bool logical_used(const llama_kv_stream_slot_handle & handle, size_t pending, size_t & output) const noexcept;
    // Retain prefix bytes/frontiers while invalidating handles from the previous request.
    bool advance(const llama_kv_stream_slot_handle & handle, llama_kv_stream_slot_handle & output) noexcept;
    const llama_kv_stream_context_limits & limits() const noexcept;

private:
    struct slot {
        uint64_t generation = 0;
        std::vector<std::shared_ptr<llama_kv_stream_logical_cache>> caches;
        std::vector<std::shared_ptr<llama_kv_stream_host>> backing;
        std::vector<std::shared_ptr<llama_kv_cells>> cells;
    };
    llama_kv_stream_slot_table() = default;
    llama_kv_stream_context_limits capacity;
    std::vector<slot> slots;
};
