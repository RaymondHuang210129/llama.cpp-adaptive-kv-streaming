#pragma once

#include "llama-kv-stream-content.h"
#include "llama-kv-stream-writer.h"
#include "llama-kv-stream-publication.h"
#include "llama-kv-stream-selection.h"
#include "../ggml/src/ggml-kv-stream-copy.h"

#include <functional>

class llama_kv_cells;
struct llama_ubatch;
enum class llama_memory_work_phase;

struct llama_kv_stream_logical_identity {
    uint64_t id = 0;
    uint64_t generation = 0;
};

struct llama_kv_stream_logical_checkpoint {
    uint64_t id = 0;
    uint64_t generation = 0;
    size_t tokens = 0;
};

// A retained prefix survives appends, but not truncation into its rows or unknown replacement.
class llama_kv_stream_logical_prefix {
    friend class llama_kv_stream_logical_cache;
    uint64_t cache_id=0;
    size_t tokens=0;
    bool valid=true;
};

// Exact committed planes, independent of process-local cache IDs and host addresses.
struct llama_kv_stream_logical_snapshot {
    ggml_kv_stream_shape shape;
    uint32_t layers = 0;
    size_t tokens = 0;
    std::vector<uint8_t> bytes;
    std::shared_ptr<const llama_kv_stream_logical_prefix> prefix;
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

    // Caller retires all device users first. Failure preserves the output snapshot.
    bool save(llama_kv_stream_logical_snapshot & output,bool copy_bytes=true) const;
    // Validate complete geometry and payload before replacing committed bytes at stable addresses.
    bool save_prefix(size_t tokens,llama_kv_stream_logical_snapshot & output) const;
    bool accepts(const llama_kv_stream_logical_snapshot & snapshot) const noexcept;
    bool load(const llama_kv_stream_logical_snapshot & snapshot);

    // One append is active at a time. Host publication does not claim a device mirror.
    bool begin(size_t count);
    bool publish_host(llama_kv_stream_write & write);
    // Publish separately completed native layers under one authoritative generation/frontier.
    bool publish_host(const std::vector<llama_kv_stream_write *> & writes);
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
    bool covers(const std::vector<llama_kv_stream_write *> & writes) const noexcept;
    bool drain_generated();
    bool reset_frontier(size_t tokens, bool suffix = false);
    void invalidate_prefixes(size_t tokens,bool all) noexcept;
    mutable std::vector<std::weak_ptr<llama_kv_stream_logical_prefix>> prefixes;

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

// Native columns pack committed rows; logical growth allowances are not attention columns.
// Canonical host addresses never move.
struct llama_kv_stream_slot_projection {
    llama_kv_stream_selection selected;
    std::vector<llama_kv_stream_slot_handle> handles;
    std::vector<llama_kv_stream_logical_checkpoint> sources;
    std::shared_ptr<llama_kv_cells> cells;
    std::vector<ggml_kv_stream_gather_row> columns;
    uint32_t role = 0;
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
    bool project(const llama_kv_stream_selection & selected, const std::vector<llama_kv_stream_slot_handle> & handles,
        uint32_t role, llama_kv_stream_slot_projection & output) const;
    // Semantic snapshot only; the common owner must still retain source/storage completion pins.
    bool projection_current(const llama_kv_stream_slot_projection & projection) const noexcept;
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

// One native producer batch over independently owned canonical histories. The caller owns
// native cell placement and model completion; this object owns only K/V publication.
class llama_kv_stream_group_producer {
public:
    using publication_rows = std::vector<ggml_kv_stream_gather_row>;
    using publication_admit = std::function<bool(uint32_t,const publication_rows &)>;
    using publication_copy = std::function<bool(uint32_t,bool,const ggml_tensor *,size_t,size_t,const publication_rows &)>;
    using publication_complete = std::function<bool(uint32_t,const publication_rows &,bool)>;
    static std::unique_ptr<llama_kv_stream_group_producer> create(
        std::shared_ptr<llama_kv_stream_slot_table> table, std::shared_ptr<llama_kv_stream_writer> writer,
        uint32_t maximum_rows, uint32_t maximum_verify_rows, uint32_t maximum_selected);
    ~llama_kv_stream_group_producer();
    bool bind_publication(publication_admit admit,publication_copy copy,publication_complete complete);
    // The GPU reader must use this exact ordered backend and fence it before ring uploads.
    bool bind_consumer(ggml_backend_t backend);
    bool begin(std::shared_ptr<llama_kv_stream_slot_projection> projection, const llama_ubatch & batch,
        const std::vector<uint32_t> & native_indices, llama_memory_work_phase phase);
    bool produce(uint32_t layer, const ggml_tensor * key, const ggml_tensor * value);
    bool layer_ready(uint32_t layer) const noexcept;
    bool finish();
    bool abort() noexcept;
private:
    llama_kv_stream_group_producer() = default;
    struct implementation;
    std::unique_ptr<implementation> impl;
};
