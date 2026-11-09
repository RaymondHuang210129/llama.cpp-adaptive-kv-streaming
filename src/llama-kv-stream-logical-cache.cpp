#include "llama-kv-stream-logical-cache.h"
#include "llama-kv-cells.h"

#include <new>
#include <utility>

llama_kv_stream_logical_cache::llama_kv_stream_logical_cache(std::shared_ptr<llama_kv_stream_host> host) :
    backing(std::move(host)) {}

llama_kv_stream_logical_cache::~llama_kv_stream_logical_cache() {
    if (active_writer) {
        try { drain_generated(); } catch (...) {}
    }
}

std::unique_ptr<llama_kv_stream_logical_cache> llama_kv_stream_logical_cache::create(
        std::shared_ptr<llama_kv_stream_host> host) {
    if (!host || !host->valid() || !host->cache_id()) return {};
    try {
        std::unique_ptr<llama_kv_stream_logical_cache> result(new llama_kv_stream_logical_cache(std::move(host)));
        result->authoritative = std::make_shared<llama_kv_stream_content>(result->backing);
        result->expected_generation = result->authoritative->generation();
        result->publications = llama_kv_stream_publications::create({
            0, result->backing->config().context_tokens, result->expected_generation, 1, true});
        return result->publications ? std::move(result) : nullptr;
    } catch (const std::bad_alloc &) {
        return {};
    }
}

std::shared_ptr<llama_kv_stream_host> llama_kv_stream_logical_cache::host() const noexcept { return backing; }
std::shared_ptr<llama_kv_stream_content> llama_kv_stream_logical_cache::content() const noexcept { return authoritative; }
llama_kv_stream_logical_identity llama_kv_stream_logical_cache::identity() const noexcept {
    return {backing->cache_id(), expected_generation};
}
llama_kv_stream_logical_checkpoint llama_kv_stream_logical_cache::checkpoint() const noexcept {
    return {backing->cache_id(), expected_generation, committed};
}
llama_kv_stream_publication_frontiers llama_kv_stream_logical_cache::frontiers() const noexcept {
    return publications->frontiers();
}
size_t llama_kv_stream_logical_cache::tokens() const noexcept { return committed; }
bool llama_kv_stream_logical_cache::ready() const noexcept {
    return !failed && backing->valid() && !pending.pending() && !active_writer &&
        authoritative->generation() == expected_generation && !publications->failed() && !publications->pending() &&
        publications->frontiers().committed == committed;
}

bool llama_kv_stream_logical_cache::begin(size_t count) {
    if (failed || !backing->valid() || pending.pending() || authoritative->generation() != expected_generation ||
            publications->pending() || publications->frontiers().committed != committed ||
            !count || count > backing->config().context_tokens - committed) return false;
    return publications->reserve(committed, count, 1, {backing}, {}, pending);
}

bool llama_kv_stream_logical_cache::begin_generated(
        std::shared_ptr<llama_kv_stream_writer> writer, const ggml_tensor * k, const ggml_tensor * v,
        const generated_stage & stage) {
    if (!writer || active_writer || backing->config().layers != 1 ||
            !writer->matches_shape(backing->config().shape) || !writer->accepts(k, false) ||
            !writer->accepts(v, true) || k->ne[1] != v->ne[1]) return false;
    const size_t rows = size_t(k->ne[1]);
    if (!begin(rows)) return false;
    llama_kv_stream_writer_completion completions[2];
    llama_kv_stream_write write;
    const size_t first = committed;
    const auto & layout = backing->layout();
    const auto fail = [&] {
        for (auto & completion : completions) {
            if (completion.host) completion.host->synchronize();
            if (completion.device) completion.device->synchronize();
        }
        writer->release_completed();
        cancel();
        return false;
    };
    try {
        const std::vector<llama_kv_stream_write_span> spans{
            {0, ggml_kv_stream_operand::k, first*layout.k_token_bytes, nullptr, rows*layout.k_token_bytes},
            {0, ggml_kv_stream_operand::v, first*layout.v_token_bytes, nullptr, rows*layout.v_token_bytes},
        };
        if (!authoritative->prepare_direct_generated(spans, [&](const auto & span, void * destination) {
                const bool value = span.operand == ggml_kv_stream_operand::v;
                const std::function<bool(const ggml_tensor *, size_t, size_t)> publish = stage ?
                    [&](const ggml_tensor * encoded, size_t row, size_t count) {
                        return stage(value, encoded, row, count);
                    } : std::function<bool(const ggml_tensor *, size_t, size_t)>{};
                if (backing->demand_backed()) {
                    std::vector<llama_kv_stream_host_range> ranges;
                    if (!backing->ranges(span.layer,span.operand,span.offset,span.bytes,ranges)) return false;
                    return writer->generate_async_registered(value ? v : k,value,ranges,publish,completions[value]);
                }
                return writer->generate_async(value ? v : k, value, destination, publish, completions[value]);
            }, write)) return fail();
    } catch (...) {
        return fail();
    }
    active_writer = std::move(writer);
    generated[0] = std::move(completions[0]);
    generated[1] = std::move(completions[1]);
    generated_write = std::move(write);
    return true;
}

bool llama_kv_stream_logical_cache::drain_generated() {
    bool ready = true;
    for (auto & completion : generated) {
        if (completion.host) ready = completion.host->synchronize() && ready;
        if (completion.device) ready = completion.device->synchronize() && ready;
    }
    if (active_writer) active_writer->release_completed();
    return ready;
}

bool llama_kv_stream_logical_cache::complete_generated() {
    if (!active_writer || !generated_write.pending()) return false;
    if (!drain_generated()) {
        failed = true;
        return false;
    }
    active_writer.reset();
    for (auto & completion : generated) completion = {};
    if (!publish_host(generated_write)) {
        failed = true;
        return false;
    }
    return finish();
}

bool llama_kv_stream_logical_cache::complete(llama_kv_stream_publication_domain domain) {
    llama_kv_stream_publication_completion k, v;
    if (!pending.submit(0, llama_kv_stream_publication_plane::k, domain, k) ||
            !pending.submit(0, llama_kv_stream_publication_plane::v, domain, v)) return false;
    return k.finish() && v.finish();
}

bool llama_kv_stream_logical_cache::covers(const llama_kv_stream_write & write) const noexcept {
    if (!write.pending()) return false;
    const size_t first = pending.first();
    const size_t count = pending.count();
    const auto & layout = backing->layout();
    for (const auto & part : write.parts) {
        if (part.layer >= backing->config().layers ||
                (part.operand != ggml_kv_stream_operand::k && part.operand != ggml_kv_stream_operand::v)) return false;
        const size_t stride = part.operand == ggml_kv_stream_operand::k ? layout.k_token_bytes : layout.v_token_bytes;
        const size_t begin = first*stride;
        const size_t end = (first + count)*stride;
        if (part.offset < begin || part.offset > end || part.bytes > end - part.offset) return false;
    }
    for (uint32_t layer = 0; layer < backing->config().layers; ++layer) {
        for (const auto operand : {ggml_kv_stream_operand::k, ggml_kv_stream_operand::v}) {
            const size_t stride = operand == ggml_kv_stream_operand::k ? layout.k_token_bytes : layout.v_token_bytes;
            const size_t end = (first + count)*stride;
            size_t covered = first*stride;
            while (covered < end) {
                size_t next = covered;
                for (const auto & part : write.parts) {
                    if (part.layer == layer && part.operand == operand &&
                            part.offset <= covered && part.offset + part.bytes > next) next = part.offset + part.bytes;
                }
                if (next == covered) return false;
                covered = next;
            }
        }
    }
    return true;
}

bool llama_kv_stream_logical_cache::publish_host(llama_kv_stream_write & write) {
    if (failed || active_writer || !pending.pending() || host_ready ||
            authoritative->generation() != expected_generation || !covers(write) ||
            !authoritative->commit(write)) return false;
    expected_generation = authoritative->generation();
    if (!complete(llama_kv_stream_publication_domain::host)) {
        failed = true;
        return false;
    }
    host_ready = true;
    return true;
}

bool llama_kv_stream_logical_cache::finish() {
    if (failed || !backing->valid() || !pending.pending() || !host_ready ||
            !pending.committed() || authoritative->generation() != expected_generation) return false;
    const size_t next = publications->frontiers().committed;
    if (next != committed + pending.count() || !pending.retire()) return false;
    committed = next;
    host_ready = false;
    return true;
}

bool llama_kv_stream_logical_cache::reset_frontier(size_t tokens, bool suffix) {
    if (!backing->valid() || pending.pending() || authoritative->generation() == UINT64_MAX) return false;
    auto replacement = llama_kv_stream_publications::create({
        tokens, backing->config().context_tokens, authoritative->generation() + 1, 1, true});
    if (!replacement || !(suffix ? authoritative->invalidate_suffix(tokens) : authoritative->invalidate())) return false;
    expected_generation = authoritative->generation();
    publications = std::move(replacement);
    committed = tokens;
    host_ready = failed = false;
    return true;
}

bool llama_kv_stream_logical_cache::cancel() {
    if (!pending.pending() || host_ready || authoritative->generation() != expected_generation ||
            expected_generation == UINT64_MAX) return false;
    if (active_writer) {
        if (!drain_generated()) {
            failed = true;
            return false;
        }
        active_writer.reset();
        for (auto & completion : generated) completion = {};
        generated_write.cancel();
    }
    auto replacement = llama_kv_stream_publications::create({
        committed, backing->config().context_tokens, expected_generation + 1, 1, true});
    if (!replacement || !pending.cancel() || !pending.retire() || !authoritative->invalidate()) return false;
    expected_generation = authoritative->generation();
    publications = std::move(replacement);
    failed = !backing->valid();
    return true;
}

bool llama_kv_stream_logical_cache::truncate(size_t tokens) {
    if (failed || !backing->valid() || pending.pending() || tokens > committed ||
            authoritative->generation() != expected_generation) return false;
    return tokens == committed || reset_frontier(tokens,true);
}

bool llama_kv_stream_logical_cache::restore(const llama_kv_stream_logical_checkpoint & checkpoint) {
    if (!checkpoint.id || checkpoint.id != backing->cache_id() || !checkpoint.generation ||
            checkpoint.tokens > backing->config().context_tokens || pending.pending()) return false;
    return reset_frontier(checkpoint.tokens);
}

std::unique_ptr<llama_kv_stream_slot_table> llama_kv_stream_slot_table::create(const llama_kv_stream_context_limits & limits) {
    llama_kv_stream_context_limits checked;
    if (!llama_kv_stream_context_limits_make(limits.slot_limit, limits.slot_storage, limits.slots, checked) ||
            checked.group_limit != limits.group_limit || checked.history_storage != limits.history_storage) return {};
    try {
        std::unique_ptr<llama_kv_stream_slot_table> table(new llama_kv_stream_slot_table);
        table->capacity = checked;
        table->slots.resize(checked.slots);
        return table;
    } catch (const std::bad_alloc &) {
        return {};
    }
}

static bool same_slot_geometry(const llama_kv_stream_host_config & a, const llama_kv_stream_host_config & b) {
    return a.layers == b.layers && a.shape.type_k == b.shape.type_k && a.shape.type_v == b.shape.type_v &&
        a.shape.head_dim_k == b.shape.head_dim_k && a.shape.head_dim_v == b.shape.head_dim_v &&
        a.shape.heads == b.shape.heads && a.shape.page_tokens == b.shape.page_tokens && a.shape.alignment == b.shape.alignment;
}

static bool host_ranges_overlap(const llama_kv_stream_host & a, const llama_kv_stream_host & b) {
    llama_kv_stream_host_layer first, second;
    if (!a.layer(0, first) || !b.layer(0, second)) return true;
    const auto x = reinterpret_cast<uintptr_t>(first.k), y = reinterpret_cast<uintptr_t>(second.k);
    return x >= y ? x - y < b.bytes() : y - x < a.bytes();
}

bool llama_kv_stream_slot_table::bind(uint32_t index,
        std::vector<std::shared_ptr<llama_kv_stream_logical_cache>> caches, llama_kv_stream_slot_handle & output) {
    if (index >= slots.size() || slots[index].generation || caches.empty()) return false;
    try {
        std::vector<std::shared_ptr<llama_kv_stream_host>> backing;
        for (size_t role = 0; role < caches.size(); ++role) {
            const auto & cache = caches[role];
            if (!cache || !cache->ready()) return false;
            const auto host = cache->host();
            if (!host->demand_backed() || cache->content()->host() != host ||
                    host->config().context_tokens != capacity.slot_limit || host->layout().tokens != capacity.slot_storage) return false;
            for (size_t prior = 0; prior < role; ++prior) {
                if (caches[prior]->identity().id == cache->identity().id || host_ranges_overlap(*backing[prior], *host)) return false;
            }
            for (const auto & slot : slots) if (slot.generation) {
                if (slot.caches.size() != caches.size() || !same_slot_geometry(host->config(), slot.backing[role]->config())) return false;
                for (size_t other = 0; other < slot.caches.size(); ++other) {
                    if (slot.caches[other]->identity().id == cache->identity().id || host_ranges_overlap(*slot.backing[other], *host)) return false;
                }
            }
            backing.push_back(host);
        }
        std::vector<std::shared_ptr<llama_kv_cells>> cells;
        for (size_t role = 0; role < caches.size(); ++role) {
            auto metadata = std::make_shared<llama_kv_cells>();
            metadata->resize(capacity.slot_storage);
            cells.push_back(std::move(metadata));
        }
        auto & slot = slots[index];
        slot.cells = std::move(cells);
        slot.caches = std::move(caches);
        slot.backing = std::move(backing);
        slot.generation = 1;
        output = {index, 1};
        return true;
    } catch (const std::bad_alloc &) {
        return false;
    }
}

bool llama_kv_stream_slot_table::get(const llama_kv_stream_slot_handle & handle, uint32_t role,
        std::shared_ptr<llama_kv_stream_logical_cache> & output) const noexcept {
    if (!handle.generation || handle.slot >= slots.size()) return false;
    const auto & slot = slots[handle.slot];
    if (slot.generation != handle.generation || role >= slot.caches.size() || role >= slot.cells.size() ||
            slot.cells[role]->size() != capacity.slot_storage) return false;
    const auto & cache = slot.caches[role];
    if (cache->host() != slot.backing[role] || cache->content()->host() != slot.backing[role] || !slot.backing[role]->valid()) return false;
    output = cache;
    return true;
}

bool llama_kv_stream_slot_table::get_cells(const llama_kv_stream_slot_handle & handle, uint32_t role,
        std::shared_ptr<llama_kv_cells> & output) const noexcept {
    std::shared_ptr<llama_kv_stream_logical_cache> cache;
    if (!get(handle,role,cache)) return false;
    const auto & cells = slots[handle.slot].cells[role];
    if (cells->size() != capacity.slot_storage) return false;
    output = cells;
    return true;
}

bool llama_kv_stream_slot_table::advance(const llama_kv_stream_slot_handle & handle, llama_kv_stream_slot_handle & output) noexcept {
    if (handle.slot >= slots.size() || !handle.generation || handle.generation == UINT64_MAX) return false;
    auto & slot = slots[handle.slot];
    if (slot.generation != handle.generation) return false;
    for (size_t role = 0; role < slot.caches.size(); ++role) {
        std::shared_ptr<llama_kv_stream_logical_cache> cache;
        std::shared_ptr<llama_kv_cells> cells;
        if (!get(handle,uint32_t(role),cache) || !cache->ready() || !get_cells(handle,uint32_t(role),cells)) return false;
    }
    ++slot.generation;
    output = {handle.slot, slot.generation};
    return true;
}

bool llama_kv_stream_slot_table::logical_used(const llama_kv_stream_slot_handle & handle, size_t pending, size_t & output) const noexcept {
    std::shared_ptr<llama_kv_stream_logical_cache> cache;
    if (pending > 1 || !get(handle,0,cache) || !cache->ready() || cache->tokens() > capacity.slot_limit ||
            pending > capacity.slot_limit-cache->tokens()) return false;
    output=cache->tokens()+pending;
    return true;
}

const llama_kv_stream_context_limits & llama_kv_stream_slot_table::limits() const noexcept { return capacity; }
