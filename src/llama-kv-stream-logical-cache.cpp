#include "llama-kv-stream-logical-cache.h"
#include "llama-kv-cells.h"
#include "llama-batch.h"
#include "llama-memory-phase.h"

#include <new>
#include <cstring>
#include <algorithm>
#include <utility>

llama_kv_stream_logical_cache::llama_kv_stream_logical_cache(std::shared_ptr<llama_kv_stream_host> host) :
    backing(std::move(host)) {}

llama_kv_stream_logical_cache::~llama_kv_stream_logical_cache() {
    invalidate_prefixes(0,true);
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

static bool logical_snapshot_bytes(const llama_kv_stream_host & host, size_t tokens, size_t & bytes) {
    const auto & layout=host.layout();
    if (tokens>host.config().context_tokens || layout.k_token_bytes>SIZE_MAX-layout.v_token_bytes) return false;
    const size_t stride=layout.k_token_bytes+layout.v_token_bytes;
    if (stride && tokens>SIZE_MAX/stride) return false;
    const size_t layer_bytes=tokens*stride;
    if (layer_bytes && host.config().layers>SIZE_MAX/layer_bytes) return false;
    bytes=layer_bytes*host.config().layers;return true;
}

bool llama_kv_stream_logical_cache::save(llama_kv_stream_logical_snapshot & output,bool copy_bytes) const {
    if (!ready()) return false;
    try {
        llama_kv_stream_logical_snapshot saved;
        saved.shape=backing->config().shape;saved.layers=backing->config().layers;saved.tokens=committed;
        if (!copy_bytes) return save_prefix(committed,output);
        size_t bytes;
        if (!logical_snapshot_bytes(*backing,committed,bytes)) return false;
        saved.bytes.resize(bytes);
        size_t offset=0;
        for (uint32_t layer=0;layer<saved.layers;++layer) {
            llama_kv_stream_host_layer planes;
            if (!backing->layer(layer,planes)) return false;
            for (bool value : {false,true}) {
                const auto operand=value ? ggml_kv_stream_operand::v : ggml_kv_stream_operand::k;
                const size_t count=committed*(value ? backing->layout().v_token_bytes : backing->layout().k_token_bytes);
                if (!count) continue;
                std::vector<llama_kv_stream_host_range> ranges;
                if (!backing->ranges(layer,operand,0,count,ranges)) return false;
                std::memcpy(saved.bytes.data()+offset,value ? planes.v : planes.k,count);offset+=count;
            }
        }
        output=std::move(saved);return true;
    } catch (const std::bad_alloc &) {return false;}
}

bool llama_kv_stream_logical_cache::save_prefix(size_t tokens,llama_kv_stream_logical_snapshot & output) const {
    if (!ready() || tokens>committed) return false;
    try {
        llama_kv_stream_logical_snapshot saved;saved.shape=backing->config().shape;saved.layers=backing->config().layers;saved.tokens=tokens;
        auto prefix=std::make_shared<llama_kv_stream_logical_prefix>();prefix->cache_id=backing->cache_id();prefix->tokens=tokens;
        prefixes.erase(std::remove_if(prefixes.begin(),prefixes.end(),[](const auto & prefix) {return prefix.expired();}),prefixes.end());
        prefixes.push_back(prefix);saved.prefix=std::move(prefix);output=std::move(saved);return true;
    } catch (const std::bad_alloc &) {return false;}
}

bool llama_kv_stream_logical_cache::accepts(const llama_kv_stream_logical_snapshot & saved) const noexcept {
    const auto & shape=backing->config().shape;
    if (saved.layers!=backing->config().layers || saved.shape.type_k!=shape.type_k || saved.shape.type_v!=shape.type_v ||
            saved.shape.head_dim_k!=shape.head_dim_k || saved.shape.head_dim_v!=shape.head_dim_v || saved.shape.heads!=shape.heads ||
            saved.shape.page_tokens!=shape.page_tokens || saved.shape.alignment!=shape.alignment) return false;
    size_t bytes;
    if (!logical_snapshot_bytes(*backing,saved.tokens,bytes)) return false;
    if (saved.prefix) return saved.bytes.empty() && saved.prefix->valid && saved.prefix->cache_id==backing->cache_id() &&
        saved.prefix->tokens==saved.tokens && saved.tokens<=committed;
    return saved.bytes.size()==bytes;
}

bool llama_kv_stream_logical_cache::load(const llama_kv_stream_logical_snapshot & saved) {
    if (!ready() || expected_generation==UINT64_MAX || !accepts(saved)) return false;
    if (saved.prefix) return truncate(saved.tokens);
    try {
        auto replacement=llama_kv_stream_publications::create({saved.tokens,backing->config().context_tokens,expected_generation+1,1,true});
        if (!replacement || !backing->prepare(saved.tokens)) return false;
        std::vector<llama_kv_stream_host_layer> planes(saved.layers);
        for (uint32_t layer=0;layer<saved.layers;++layer) if (!backing->layer(layer,planes[layer])) return false;
        if (!authoritative->invalidate()) return false;
        invalidate_prefixes(0,true);
        size_t offset=0;
        for (const auto & plane : planes) for (bool value : {false,true}) {
            const size_t count=saved.tokens*(value ? backing->layout().v_token_bytes : backing->layout().k_token_bytes);
            if (!count) continue;
            std::memcpy(value ? plane.v : plane.k,saved.bytes.data()+offset,count);offset+=count;
        }
        expected_generation=authoritative->generation();publications=std::move(replacement);committed=saved.tokens;
        host_ready=failed=false;return true;
    } catch (const std::bad_alloc &) {return false;}
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

bool llama_kv_stream_logical_cache::covers(const std::vector<llama_kv_stream_write *> & writes) const noexcept {
    if (writes.empty()) return false;
    const size_t first=pending.first(),count=pending.count();
    const auto & layout=backing->layout();
    for (size_t i=0;i<writes.size();++i) {
        const auto * write=writes[i];
        if (!write || !write->pending()) return false;
        for (size_t j=0;j<i;++j) if (writes[j]==write) return false;
        for (const auto & part : write->parts) {
            if (part.layer>=backing->config().layers || (part.operand!=ggml_kv_stream_operand::k && part.operand!=ggml_kv_stream_operand::v)) return false;
            const size_t stride=part.operand==ggml_kv_stream_operand::k ? layout.k_token_bytes : layout.v_token_bytes;
            const size_t begin=first*stride,end=(first+count)*stride;
            if (part.offset<begin || part.offset>end || part.bytes>end-part.offset) return false;
        }
    }
    for (uint32_t layer=0;layer<backing->config().layers;++layer) for (auto operand : {ggml_kv_stream_operand::k,ggml_kv_stream_operand::v}) {
        const size_t stride=operand==ggml_kv_stream_operand::k ? layout.k_token_bytes : layout.v_token_bytes;
        size_t covered=first*stride;
        const size_t end=(first+count)*stride;
        while (covered<end) {
            size_t next=covered;
            for (const auto * write : writes) for (const auto & part : write->parts)
                if (part.layer==layer && part.operand==operand && part.offset<=covered && part.offset+part.bytes>next) next=part.offset+part.bytes;
            if (next==covered) return false;
            covered=next;
        }
    }
    return true;
}

bool llama_kv_stream_logical_cache::publish_host(const std::vector<llama_kv_stream_write *> & writes) {
    if (failed || active_writer || !pending.pending() || host_ready ||
            authoritative->generation()!=expected_generation || !covers(writes) || !authoritative->commit(writes)) return false;
    expected_generation=authoritative->generation();
    if (!complete(llama_kv_stream_publication_domain::host)) {failed=true; return false;}
    host_ready=true;
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

void llama_kv_stream_logical_cache::invalidate_prefixes(size_t tokens,bool all) noexcept {
    for (const auto & entry : prefixes) if (auto prefix=entry.lock())
        if (all || prefix->tokens>tokens) prefix->valid=false;
}

bool llama_kv_stream_logical_cache::reset_frontier(size_t tokens, bool suffix) {
    if (!backing->valid() || pending.pending() || authoritative->generation() == UINT64_MAX) return false;
    auto replacement = llama_kv_stream_publications::create({
        tokens, backing->config().context_tokens, authoritative->generation() + 1, 1, true});
    if (!replacement || !(suffix ? authoritative->invalidate_suffix(tokens) : authoritative->invalidate())) return false;
    invalidate_prefixes(tokens,!suffix);
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

bool llama_kv_stream_slot_table::project(const llama_kv_stream_selection & selected,
        const std::vector<llama_kv_stream_slot_handle> & handles,uint32_t role,llama_kv_stream_slot_projection & output) const {
    if (!llama_kv_stream_selection_valid(selected) || selected.context_tokens!=capacity.group_limit ||
            selected.slots.size()!=handles.size() || handles.size()>capacity.slots) return false;
    try {
        llama_kv_stream_slot_projection next;
        next.selected=selected;next.handles=handles;next.role=role;
        next.cells=std::make_shared<llama_kv_cells>();next.cells->resize(capacity.slot_storage);
        next.columns.resize(capacity.slot_storage);
        uint32_t native_rows=0;
        for (size_t source=0;source<selected.slots.size();++source) {
            const auto & slot=selected.slots[source];
            std::shared_ptr<llama_kv_stream_logical_cache> cache;
            std::shared_ptr<llama_kv_cells> cells;
            if (handles[source].slot!=slot.id || !get(handles[source],role,cache) || !cache->ready() ||
                    !get_cells(handles[source],role,cells) || cells->get_has_shift() ||
                    (!role && (slot.history_tokens<cache->tokens() || slot.history_tokens-cache->tokens()>1)) ||
                    (role && cache->tokens()>slot.history_tokens+slot.growth_tokens) || cells->get_used()!=cache->tokens()) return false;
            const auto tokens=uint32_t(cache->tokens());
            for (uint32_t row=0;row<tokens;++row) {
                if (cells->is_empty(row) || cells->seq_count(row)!=1 || !cells->seq_has(row,llama_seq_id(slot.id))) return false;
                next.columns[native_rows+row]={uint32_t(source),row};
            }
            next.cells->set(native_rows,cells->cp(0,tokens));
            native_rows+=tokens;
            next.sources.push_back(cache->checkpoint());
        }
        output=std::move(next);
        return true;
    } catch (const std::bad_alloc &) {return false;}
}

bool llama_kv_stream_slot_table::projection_current(const llama_kv_stream_slot_projection & projection) const noexcept {
    if (!projection.cells || !llama_kv_stream_selection_valid(projection.selected) || projection.selected.context_tokens!=capacity.group_limit ||
            projection.handles.size()!=projection.selected.slots.size() || projection.sources.size()!=projection.handles.size() ||
            projection.cells->size()!=capacity.slot_storage || projection.columns.size()!=capacity.slot_storage) return false;
    for (size_t source=0;source<projection.handles.size();++source) {
        std::shared_ptr<llama_kv_stream_logical_cache> cache;
        std::shared_ptr<llama_kv_cells> cells;
        if (projection.handles[source].slot!=projection.selected.slots[source].id ||
                !get(projection.handles[source],projection.role,cache) || !cache->ready() ||
                !get_cells(projection.handles[source],projection.role,cells)) return false;
        const auto current=cache->checkpoint(),saved=projection.sources[source];
        if (current.id!=saved.id || current.generation!=saved.generation || current.tokens!=saved.tokens ||
                cells->get_used()!=current.tokens || cells->get_has_shift()) return false;
    }
    return !projection.handles.empty();
}

struct llama_kv_stream_group_producer::implementation {
    struct source {
        std::shared_ptr<llama_kv_stream_logical_cache> cache;
        std::shared_ptr<llama_kv_cells> cells;
        llama_kv_stream_logical_checkpoint before;
        std::vector<uint32_t> indices;
        std::vector<llama_kv_stream_write> writes;
        bool begun=false,published=false;
    };
    struct row {uint32_t source,column;size_t token;llama_pos position;};
    struct frame {
        std::shared_ptr<llama_kv_stream_slot_projection> projection;
        std::vector<source> sources;
        std::vector<row> rows;
        std::vector<uint32_t> indices;
        std::vector<ggml_kv_stream_gather_row> previous_columns;
        llama_kv_cells previous_cells;
        std::vector<bool> ready;
        std::vector<bool> device_published;
        std::vector<std::array<llama_kv_stream_writer_completion,2>> transfers;
        bool admitted=false,columns_changed=false;
    };
    std::shared_ptr<llama_kv_stream_slot_table> table;
    std::shared_ptr<llama_kv_stream_writer> writer;
    uint32_t maximum_rows=0,maximum_verify_rows=0,maximum_selected=0;
    std::unique_ptr<frame> active;
    llama_kv_stream_group_producer::publication_admit publication_admit;
    llama_kv_stream_group_producer::publication_copy publication_copy;
    llama_kv_stream_group_producer::publication_complete publication_complete;
    ggml_backend_t consumer=nullptr;
    bool failed=false;
};

std::unique_ptr<llama_kv_stream_group_producer> llama_kv_stream_group_producer::create(
        std::shared_ptr<llama_kv_stream_slot_table> table,std::shared_ptr<llama_kv_stream_writer> writer,
        uint32_t maximum_rows,uint32_t maximum_verify_rows,uint32_t maximum_selected) {
    if (!table || !writer || !maximum_rows || maximum_rows>writer->maximum_rows() ||
            !maximum_verify_rows || maximum_verify_rows>8 || !maximum_selected || maximum_selected>table->limits().slots ||
            maximum_verify_rows>maximum_rows) return {};
    try {
        auto result=std::unique_ptr<llama_kv_stream_group_producer>(new llama_kv_stream_group_producer);
        result->impl=std::make_unique<implementation>();
        result->impl->table=std::move(table);result->impl->writer=std::move(writer);
        result->impl->maximum_rows=maximum_rows;result->impl->maximum_verify_rows=maximum_verify_rows;
        result->impl->maximum_selected=maximum_selected;
        return result;
    } catch (const std::bad_alloc &) {return {};}
}

llama_kv_stream_group_producer::~llama_kv_stream_group_producer() {if (impl) abort();}

bool llama_kv_stream_group_producer::bind_publication(publication_admit admit,publication_copy copy,publication_complete complete) {
    auto & s=*impl;
    if (s.active || s.failed || (bool(admit)!=bool(copy)) || (bool(copy)!=bool(complete))) return false;
    s.publication_admit=std::move(admit);s.publication_copy=std::move(copy);s.publication_complete=std::move(complete);return true;
}

bool llama_kv_stream_group_producer::bind_consumer(ggml_backend_t backend) {
    auto & s=*impl;
    if (s.active || s.failed || (backend && !s.writer->matches_backend(backend))) return false;
    s.consumer=backend;return true;
}

bool llama_kv_stream_group_producer::begin(std::shared_ptr<llama_kv_stream_slot_projection> projection,
        const llama_ubatch & batch,const std::vector<uint32_t> & indices,llama_memory_work_phase phase) {
    auto & s=*impl;
    const bool target=phase==llama_memory_work_phase::target_prefill || phase==llama_memory_work_phase::target_verify;
    const bool auxiliary=phase==llama_memory_work_phase::mtp_catchup || phase==llama_memory_work_phase::mtp_draft;
    if (s.failed || s.active || !projection || !s.table->projection_current(*projection) || !batch.n_tokens ||
            batch.n_tokens>s.maximum_rows || indices.size()!=batch.n_tokens || !batch.pos ||
            !batch.n_seq_id || !batch.seq_id || (batch.n_pos!=1 && batch.n_pos!=4) ||
            projection->handles.size()>s.maximum_selected ||
            (projection->role==0 ? !target : projection->role!=1 || !auxiliary) || (auxiliary && !batch.output)) return false;
    try {
        auto next=std::make_unique<implementation::frame>();next->projection=std::move(projection);next->indices=indices;
        next->sources.resize(next->projection->handles.size());
        uint32_t layers=0;
        for (size_t i=0;i<next->sources.size();++i) {
            auto & source=next->sources[i];
            if (!s.table->get(next->projection->handles[i],next->projection->role,source.cache) || !source.cache->ready() ||
                    !s.table->get_cells(next->projection->handles[i],next->projection->role,source.cells) ||
                    !s.writer->matches_shape(source.cache->host()->config().shape)) return false;
            const auto count=source.cache->host()->config().layers;
            if (!count || (layers && layers!=count)) return false;
            layers=count;source.writes.resize(layers);source.before=source.cache->checkpoint();
        }
        next->ready.resize(layers,false);next->device_published.resize(layers,false);next->transfers.resize(layers);
        for (uint32_t row=0;row<batch.n_tokens;++row) {
            if (batch.n_seq_id[row]!=1 || !batch.seq_id[row] || batch.seq_id[row][0]<0 ||
                    indices[row]>=next->projection->columns.size() || !next->projection->cells->is_empty(indices[row])) return false;
            for (uint32_t previous=0;previous<row;++previous) if (indices[previous]==indices[row]) return false;
            size_t owner=next->sources.size();
            for (size_t i=0;i<next->sources.size();++i)
                if (next->projection->handles[i].slot==uint32_t(batch.seq_id[row][0])) owner=i;
            if (owner==next->sources.size()) return false;
            auto & source=next->sources[owner];
            const size_t token=source.before.tokens+source.indices.size();
            const auto & allowance=next->projection->selected.slots[owner];
            if (token>=s.table->limits().slot_limit || token>=allowance.history_tokens+allowance.growth_tokens ||
                    (phase==llama_memory_work_phase::target_verify && source.indices.size()>=s.maximum_verify_rows) ||
                    (auxiliary && (phase==llama_memory_work_phase::mtp_draft ?
                        source.indices.size()!=0 || !batch.output[row] : bool(batch.output[row])))) return false;
            source.indices.push_back(indices[row]);next->rows.push_back({uint32_t(owner),indices[row],token,batch.pos[row]});
            next->previous_columns.push_back(next->projection->columns[indices[row]]);
        }
        next->previous_cells=next->projection->cells->cp(indices);
        for (auto & source : next->sources) if (!source.indices.empty() &&
                !source.cache->host()->prepare(source.before.tokens+source.indices.size())) return false;
        s.active=std::move(next);
        for (auto & source : s.active->sources) if (!source.indices.empty()) {
            if (!source.cache->begin(source.indices.size())) {abort();return false;}
            source.begun=true;
        }
        for (const auto & row : s.active->rows) s.active->projection->columns[row.column]={row.source,row.token};
        s.active->columns_changed=true;s.active->admitted=true;
        return true;
    } catch (...) {abort();return false;}
}

bool llama_kv_stream_group_producer::produce(uint32_t layer,const ggml_tensor * key,const ggml_tensor * value) {
    auto & s=*impl;
    if (!s.active || !s.active->admitted || layer>=s.active->ready.size() || s.active->ready[layer] ||
            !s.writer->accepts(key,false) || !s.writer->accepts(value,true) ||
            key->ne[1]!=int64_t(s.active->rows.size()) || value->ne[1]!=key->ne[1]) return false;
    for (size_t i=0;i<s.active->sources.size();++i) {
        const auto & source=s.active->sources[i];
        std::shared_ptr<llama_kv_stream_logical_cache> current;
        if (!s.table->get(s.active->projection->handles[i],s.active->projection->role,current) || current!=source.cache ||
                current->tokens()!=source.before.tokens || current->identity().generation!=source.before.generation ||
                current->content()->generation()!=source.before.generation || source.cells->get_used()!=source.before.tokens ||
                (source.begun ? current->frontiers().reserved!=source.before.tokens+source.indices.size() : !current->ready())) return false;
    }
    auto & completions=s.active->transfers[layer];
    const auto drain=[&] {
        bool okay=true;
        for (auto & completion : completions) {
            if (completion.host) okay=completion.host->synchronize() && okay;
            if (completion.device) okay=completion.device->synchronize() && okay;
        }
        return okay;
    };
    try {
        publication_rows publication;
        bool device_publication=false;
        if (s.publication_admit) {
            for (const auto & row : s.active->rows) publication.push_back({row.source,row.token});
            device_publication=s.publication_admit(layer,publication);
        }
        std::function<bool(const ggml_tensor *,size_t,size_t)> publish[2];
        if (device_publication) for (size_t side=0;side<2;++side)
            publish[side]=[&,side](const ggml_tensor * encoded,size_t first,size_t count) {
                return s.publication_copy(layer,side,encoded,first,count,publication);
            };
        std::vector<llama_kv_stream_host_range> routes[2];
        for (bool side : {false,true}) for (const auto & row : s.active->rows) {
            const auto host=s.active->sources[row.source].cache->host();
            const size_t stride=side ? host->layout().v_token_bytes : host->layout().k_token_bytes;
            std::vector<llama_kv_stream_host_range> ranges;
            if (!host->ranges(layer,side ? ggml_kv_stream_operand::v : ggml_kv_stream_operand::k,row.token*stride,stride,ranges)) return false;
            for (auto & range : ranges) routes[side].push_back(std::move(range));
        }
        if (!s.writer->generate_async_routed(key,false,routes[0],publish[0],completions[0]) ||
                !s.writer->generate_async_routed(value,true,routes[1],publish[1],completions[1])) {
            drain();abort();return false;
        }
        // The bound reader uses this producer stream, including the ring queue's begin fence.
        if (s.consumer) {
            if (!s.writer->matches_backend(s.consumer)) {abort();return false;}
        } else if (device_publication) {
            for (auto & completion : completions) if (!completion.device || !completion.device->synchronize()) {abort();return false;}
        } else {
            if (!drain()) {abort();return false;}
            completions={};
        }
        for (auto & source : s.active->sources) if (source.begun) {
            const auto & layout=source.cache->host()->layout();
            const size_t first=source.before.tokens,count=source.indices.size();
            // Retained transfer completions gate these native bytes before canonical publication.
            if (!source.cache->content()->prepare_direct_generated({
                    {layer,ggml_kv_stream_operand::k,first*layout.k_token_bytes,nullptr,count*layout.k_token_bytes},
                    {layer,ggml_kv_stream_operand::v,first*layout.v_token_bytes,nullptr,count*layout.v_token_bytes}},
                    [](const auto &,void *) {return true;},source.writes[layer])) {abort();return false;}
        }
        if (device_publication && !s.publication_complete(layer,publication,false)) {abort();return false;}
        s.active->device_published[layer]=device_publication;
        s.active->ready[layer]=true;return true;
    } catch (...) {drain();abort();return false;}
}

bool llama_kv_stream_group_producer::layer_ready(uint32_t layer) const noexcept {
    return impl->active && impl->active->admitted && layer<impl->active->ready.size() && impl->active->ready[layer];
}

bool llama_kv_stream_group_producer::finish() {
    auto & s=*impl;
    if (!s.active || !s.active->admitted || !std::all_of(s.active->ready.begin(),s.active->ready.end(),[](bool ready) {return ready;})) return false;
    try {
        for (auto & pair : s.active->transfers) for (auto & completion : pair) {
            if ((completion.host && !completion.host->synchronize()) || (completion.device && !completion.device->synchronize())) {abort();return false;}
        }
        s.writer->release_completed();
        std::vector<llama_kv_cells> metadata;
        for (size_t i=0;i<s.active->sources.size();++i) {
            const auto & source=s.active->sources[i];
            std::shared_ptr<llama_kv_stream_logical_cache> current;
            if (!s.table->get(s.active->projection->handles[i],s.active->projection->role,current) || current!=source.cache ||
                    current->tokens()!=source.before.tokens || current->identity().generation!=source.before.generation ||
                    current->content()->generation()!=source.before.generation || source.cells->get_used()!=source.before.tokens) return false;
            for (auto column : source.indices) if (s.active->projection->cells->is_empty(column) ||
                    s.active->projection->cells->seq_count(column)!=1 ||
                    !s.active->projection->cells->seq_has(column,llama_seq_id(s.active->projection->handles[i].slot))) return false;
            metadata.push_back(s.active->projection->cells->cp(source.indices));
        }
        for (const auto & row : s.active->rows)
            if (s.active->projection->cells->pos_get(row.column)!=row.position) return false;
        for (auto & source : s.active->sources) if (source.begun) {
            std::vector<llama_kv_stream_write *> writes;
            for (auto & write : source.writes) writes.push_back(&write);
            if (!source.cache->publish_host(writes) || !source.cache->finish()) {abort();return false;}
            source.published=true;
        }
        for (size_t i=0;i<s.active->sources.size();++i) {
            auto & source=s.active->sources[i];
            if (source.begun) source.cells->set(uint32_t(source.before.tokens),metadata[i]);
            s.active->projection->sources[i]=source.cache->checkpoint();
        }
        if (s.publication_complete) {
            publication_rows publication;
            for (const auto & row : s.active->rows) publication.push_back({row.source,row.token});
            for (uint32_t layer=0;layer<s.active->device_published.size();++layer)
                if (s.active->device_published[layer] && !s.publication_complete(layer,publication,true)) {s.failed=true;abort();return false;}
        }
        s.active.reset();return true;
    } catch (...) {s.failed=true;abort();return false;}
}

bool llama_kv_stream_group_producer::abort() noexcept {
    auto & s=*impl;
    if (!s.active) return !s.failed;
    bool okay=true;
    try {
        for (auto & pair : s.active->transfers) for (auto & completion : pair) {
            if (completion.host) okay=completion.host->synchronize() && okay;
            if (completion.device) okay=completion.device->synchronize() && okay;
        }
        s.writer->release_completed();
        for (auto & source : s.active->sources) if (source.begun) {
            okay=(source.published ? source.cache->truncate(source.before.tokens) : source.cache->cancel()) && okay;
            for (auto & write : source.writes) write.cancel();
        }
        if (s.active->columns_changed) {
            s.active->projection->cells->set(s.active->indices,s.active->previous_cells);
            for (size_t i=0;i<s.active->indices.size();++i)
                s.active->projection->columns[s.active->indices[i]]=s.active->previous_columns[i];
        }
    } catch (...) {okay=false;}
    s.active.reset();s.failed=s.failed || !okay;
    return okay;
}
