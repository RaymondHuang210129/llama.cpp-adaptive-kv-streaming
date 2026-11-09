#include "../src/llama-kv-stream-logical-cache.h"
#include "../src/llama-kv-cells.h"
#include "testing.h"
#include "../src/llama-kv-stream-writer.h"
#include "../src/llama-kv-stream-model.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-backend-impl.h"

#include <cstring>
#include <vector>
#include <thread>

using operand = ggml_kv_stream_operand;

static std::shared_ptr<llama_kv_stream_logical_cache> reserved_cache(uint64_t id) {
    llama_kv_stream_host_config c{id, {GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 32, 32, 1, 8, 128},
        {{GGML_TYPE_Q8_0, true, true, true, true}, {GGML_TYPE_Q4_0, true, true, true, true}, true, true}, 33, 1};
    auto sparse = llama_kv_stream_host_reservation::create(c, ggml_backend_cpu_buffer_type(),
        [](ggml_backend_buffer_t owner) {
            owner->buft = ggml_backend_cpu_buffer_type();
            return ggml_backend_buffer_retain(owner);
        }, 65536);
    return llama_kv_stream_logical_cache::create(llama_kv_stream_host::from_reservation(std::move(sparse)));
}

static std::shared_ptr<llama_kv_stream_host> host(uint64_t id, ggml_type k = GGML_TYPE_Q8_0, ggml_type v = GGML_TYPE_Q4_0) {
    llama_kv_stream_host_config config;
    config.cache_id = id;
    config.shape = {k, v, 32, 32, 1, 8, 128};
    config.capabilities = {
        {k, true, true, true, true},
        {v, true, true, true, true},
        true, true,
    };
    config.context_tokens = 17;
    config.layers = 1;
    return llama_kv_stream_host::create(config, ggml_backend_cpu_buffer_type());
}

static bool append(testing & t, llama_kv_stream_logical_cache & cache, size_t count, uint8_t value) {
    const auto backing = cache.host();
    const size_t k_bytes = backing->layout().k_token_bytes*count;
    const size_t v_bytes = backing->layout().v_token_bytes*count;
    std::vector<uint8_t> k(k_bytes, value), v(v_bytes, value + 1);
    llama_kv_stream_write write;
    const size_t first = cache.tokens();
    if (!t.assert_true(cache.begin(count))) return false;
    if (!t.assert_true(cache.content()->prepare({
            {0, operand::k, first*backing->layout().k_token_bytes, k.data(), k.size()},
            {0, operand::v, first*backing->layout().v_token_bytes, v.data(), v.size()}}, write))) return false;
    if (!t.assert_true(cache.publish_host(write))) return false;
    t.assert_equal(first + count, cache.frontiers().host);
    t.assert_equal(size_t(0), cache.frontiers().device);
    t.assert_equal(first, cache.tokens());
    if (!t.assert_true(cache.finish())) return false;
    t.assert_equal(first + count, cache.tokens());
    return true;
}

struct generated_source {
    ggml_context_ptr context;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * k = nullptr;
    ggml_tensor * v = nullptr;

    generated_source(ggml_backend_t backend, size_t rows) {
        context.reset(ggml_init({65536, nullptr, true}));
        GGML_ASSERT(context);
        k = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 32, int64_t(rows));
        v = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 32, int64_t(rows));
        buffer.reset(ggml_backend_alloc_ctx_tensors(context.get(), backend));
        GGML_ASSERT(buffer);
        std::vector<float> keys(rows*32, 1.25f), values(rows*32, 2.5f);
        ggml_backend_tensor_set(k, keys.data(), 0, keys.size()*sizeof(float));
        ggml_backend_tensor_set(v, values.data(), 0, values.size()*sizeof(float));
    }
};

int main() {
    testing t;
    t.test("cache_namespace_ranges_do_not_wrap_or_overlap_under_concurrent_creation", [](testing & t) {
        const uint64_t limit = (UINT64_MAX - 3)/4;
        std::atomic<uint64_t> clock{1};
        t.assert_equal(uint64_t(0), llama_kv_stream_cache_ids_reserve(clock, 0));
        t.assert_equal(uint64_t(1), clock.load());
        clock = limit;
        t.assert_equal(uint64_t(0), llama_kv_stream_cache_ids_reserve(clock, 2));
        t.assert_equal(limit, clock.load());
        t.assert_equal(limit, llama_kv_stream_cache_ids_reserve(clock, 1));
        t.assert_equal(uint64_t(0), llama_kv_stream_cache_ids_reserve(clock, 1));
        t.assert_equal(limit + 1, clock.load());
        clock = UINT64_MAX;
        t.assert_equal(uint64_t(0), llama_kv_stream_cache_ids_reserve(clock, 2));
        t.assert_equal(uint64_t(UINT64_MAX), clock.load());
        clock = 1;
        std::vector<uint64_t> ids(4*64*2);
        std::vector<std::thread> threads;
        for (size_t thread = 0; thread < 4; ++thread) threads.emplace_back([&, thread] {
            const uint32_t count = uint32_t(thread%2 + 1);
            for (size_t i = 0; i < 64; ++i) {
                const uint64_t first = llama_kv_stream_cache_ids_reserve(clock, count);
                for (size_t j = 0; j < count; ++j) ids[(thread*64 + i)*2 + j] = first + j;
            }
        });
        for (auto & thread : threads) thread.join();
        ids.erase(std::remove(ids.begin(), ids.end(), uint64_t(0)), ids.end());
        std::sort(ids.begin(), ids.end());
        t.assert_equal(size_t(384), ids.size());
        for (size_t i = 0; i < ids.size(); ++i) t.assert_equal(uint64_t(i + 1), ids[i]);
        t.assert_equal(uint64_t(385), clock.load());
    });
    t.test("slot_reuse_epoch_preserves_addresses_and_rejects_old_request_handles", [](testing & t) {
        llama_kv_stream_context_limits limits;
        t.assert_true(llama_kv_stream_context_limits_make(33, 40, 2, limits));
        auto table = llama_kv_stream_slot_table::create(limits);
        auto a = reserved_cache(501), b = reserved_cache(502), auxiliary = reserved_cache(503);
        llama_kv_stream_slot_handle handle_a, handle_b;
        if (!t.assert_true(table && a && b && auxiliary)) return;
        t.assert_true(table->bind(0, {a, auxiliary}, handle_a));
        t.assert_true(table->bind(1, {b, reserved_cache(504)}, handle_b));
        t.assert_true(!table->bind(0, {reserved_cache(505)}, handle_a));
        t.assert_true(append(t, *a, 17, 0x31) && append(t, *b, 9, 0x41));
        llama_kv_stream_host_layer before, after;
        t.assert_true(a->host()->layer(0, before));
        const auto checkpoint_a = a->checkpoint(), checkpoint_b = b->checkpoint();
        auto old_a = handle_a;
        t.assert_true(table->advance(handle_a, handle_a));
        t.assert_equal(old_a.generation + 1, handle_a.generation);
        std::shared_ptr<llama_kv_stream_logical_cache> resolved;
        t.assert_true(table->get(handle_a, 0, resolved) && resolved == a);
        t.assert_true(!table->get(old_a, 0, resolved) && resolved == a);
        t.assert_true(!table->get(handle_a, 2, resolved) && resolved == a);
        t.assert_true(a->host()->layer(0, after) && before.k == after.k && before.v == after.v);
        t.assert_equal(checkpoint_a.generation, a->identity().generation);
        t.assert_equal(checkpoint_a.tokens, a->tokens());
        t.assert_equal(checkpoint_b.generation, b->identity().generation);
        t.assert_true(a->begin(1));
        const auto busy = handle_a;
        t.assert_true(!table->advance(handle_a, handle_a));
        t.assert_equal(busy.generation, handle_a.generation);
        t.assert_true(a->cancel());
        t.assert_true(auxiliary->begin(1));
        t.assert_true(!table->advance(handle_a, handle_a));
        t.assert_true(auxiliary->cancel());
        t.assert_true(table->get(handle_b, 0, resolved) && resolved == b);
    });
    t.test("slot_native_cells_keep_positions_and_image_metadata_outside_group_projection", [](testing & t) {
        llama_kv_stream_context_limits limits;
        t.assert_true(llama_kv_stream_context_limits_make(33,40,2,limits));
        auto table=llama_kv_stream_slot_table::create(limits);
        auto a=reserved_cache(511),b=reserved_cache(512),aux=reserved_cache(513);
        llama_kv_stream_slot_handle ha,hb;
        if (!t.assert_true(table && table->bind(0,{a,aux},ha) && table->bind(1,{b,reserved_cache(514)},hb))) return;
        std::shared_ptr<llama_kv_cells> ca,cb,extra;
        if (!t.assert_true(table->get_cells(ha,0,ca) && table->get_cells(hb,0,cb) && table->get_cells(ha,1,extra))) return;
        t.assert_true(ca != cb && ca != extra && cb != extra);
        t.assert_equal(uint32_t(40),ca->size());
        t.assert_equal(uint32_t(40),cb->size());
        if (!t.assert_true(append(t,*a,32,0x31) && append(t,*b,33,0x51))) return;
        for (uint32_t i=0;i<33;++i) {
            cb->pos_set(i,llama_pos(1024+i));
            cb->ext_set(i,{llama_pos(i),3,i%2 ? llama_token(42) : LLAMA_TOKEN_NULL});
            cb->seq_add(i,1);
            if (i<32) {
                ca->pos_set(i,llama_pos(i+7));
                ca->ext_set(i,{2,llama_pos(i),llama_token(100+i)});
                ca->seq_add(i,0);
            }
        }
        t.assert_equal(uint32_t(32),ca->get_used());
        t.assert_equal(uint32_t(33),cb->get_used());
        t.assert_equal(uint32_t(0),extra->get_used());
        const auto old=ha;
        t.assert_true(table->advance(ha,ha));
        auto resolved=cb;
        t.assert_true(!table->get_cells(old,0,resolved) && resolved == cb);
        t.assert_true(!table->get_cells(ha,2,resolved) && resolved == cb);
        t.assert_true(table->get_cells(ha,0,resolved) && resolved == ca);
        if (!t.assert_true(a->truncate(7))) return;
        for (uint32_t i=7;i<32;++i) ca->rm(i);
        t.assert_equal(uint32_t(7),ca->get_used());
        t.assert_equal(uint32_t(33),cb->get_used());
        for (uint32_t i=0;i<33;++i) {
            t.assert_equal(llama_pos(1024+i),cb->pos_get(i));
            t.assert_true(cb->seq_has(i,1) && !cb->seq_has(i,0));
            t.assert_equal(llama_pos(i),cb->ext_get(i).x);
            t.assert_equal(llama_pos(3),cb->ext_get(i).y);
            t.assert_equal(i%2 ? llama_token(42) : LLAMA_TOKEN_NULL,cb->ext_get(i).tok);
        }
        ca->resize(8);
        resolved=cb;
        t.assert_true(!table->get_cells(ha,0,resolved) && resolved == cb);
        t.assert_true(!table->advance(ha,ha));
        auto cache=b;
        size_t logical=77;
        t.assert_true(!table->get(ha,0,cache) && cache == b);
        t.assert_true(!table->logical_used(ha,0,logical) && logical == 77);
        t.assert_true(table->get_cells(hb,0,resolved) && resolved == cb);
    });
    t.test("sampled_pending_anchor_counts_once_in_logical_context", [](testing & t) {
        llama_kv_stream_context_limits limits;
        t.assert_true(llama_kv_stream_context_limits_make(33,40,1,limits));
        auto table=llama_kv_stream_slot_table::create(limits);
        auto cache=reserved_cache(509);
        llama_kv_stream_slot_handle handle;
        if (!t.assert_true(table && cache && table->bind(0,{cache},handle) && append(t,*cache,30,0x31))) return;
        size_t logical=0;
        t.assert_true(table->logical_used(handle,1,logical));
        t.assert_equal(size_t(31),logical);
        llama_kv_stream_generation_budget budget;
        t.assert_true(llama_kv_stream_generation_budget_make(limits.slot_limit-logical,7,SIZE_MAX,budget));
        t.assert_equal(uint32_t(2),budget.query_rows);
        t.assert_equal(uint32_t(1),budget.proposal_tokens);
        // Forward the old anchor plus one accepted proposal; the new bonus remains pending.
        t.assert_true(append(t,*cache,budget.query_rows,0x41));
        t.assert_true(table->logical_used(handle,1,logical));
        t.assert_equal(size_t(33),logical);
        t.assert_true(!table->logical_used(handle,2,logical));
        t.assert_equal(size_t(33),logical);
        // Explicit forwarding without sampling does not add a second logical token.
        t.assert_true(append(t,*cache,1,0x51));
        t.assert_true(table->logical_used(handle,0,logical));
        t.assert_equal(size_t(33),logical);
        t.assert_true(!table->logical_used(handle,1,logical));
        t.assert_equal(size_t(33),logical);
    });
    t.test("slot_binding_rejects_duplicate_storage_and_inconsistent_capacity", [](testing & t) {
        llama_kv_stream_context_limits limits;
        t.assert_true(llama_kv_stream_context_limits_make(33, 40, 3, limits));
        auto table = llama_kv_stream_slot_table::create(limits);
        auto cache = reserved_cache(601);
        llama_kv_stream_slot_handle handle;
        t.assert_true(table->bind(0, {cache}, handle));
        handle = {99, 77};
        t.assert_true(!table->bind(1, {cache}, handle));
        std::shared_ptr<llama_kv_stream_logical_cache> alias = llama_kv_stream_logical_cache::create(cache->host());
        t.assert_true(!table->bind(1, {alias}, handle));
        t.assert_true(!table->bind(1, {reserved_cache(601)}, handle));
        t.assert_true(!table->bind(3, {reserved_cache(602)}, handle));
        t.assert_true(!table->bind(1, {}, handle));
        t.assert_equal(uint64_t(77), handle.generation);
        t.assert_equal(uint32_t(99), handle.slot);
        auto invalid = limits; ++invalid.group_limit;
        t.assert_true(!llama_kv_stream_slot_table::create(invalid));
        t.assert_true(!table->bind(1, {llama_kv_stream_logical_cache::create(host(603))}, handle));
        t.assert_true(table->bind(1, {reserved_cache(604)}, handle));
    });
    t.test("demand_backed_slot_histories_publish_and_truncate_independently", [](testing & t) {
        auto first = host(401)->config();
        first.context_tokens = 33;
        auto second = first; second.cache_id = 402;
        const auto reserve = [](const llama_kv_stream_host_config & c) {
            auto sparse = llama_kv_stream_host_reservation::create(c, ggml_backend_cpu_buffer_type(),
                [](ggml_backend_buffer_t owner) {
                    auto * buffer = ggml_backend_buffer_retain(owner);
                    buffer->buft = ggml_backend_cpu_buffer_type();
                    return buffer;
                }, 65536);
            return llama_kv_stream_host::from_reservation(std::move(sparse));
        };
        auto a = llama_kv_stream_logical_cache::create(reserve(first));
        auto b = llama_kv_stream_logical_cache::create(reserve(second));
        if (!t.assert_true(a && b)) return;
        t.assert_true(a->host()->demand_backed() && b->host()->demand_backed());
        t.assert_true(a->host()->buffer() == nullptr && b->host()->buffer() == nullptr);
        t.assert_true(append(t, *a, 17, 0x11));
        t.assert_true(append(t, *b, 9, 0x41));
        t.assert_equal(size_t(17), a->tokens());
        t.assert_equal(size_t(9), b->tokens());
        const auto checkpoint_a = a->checkpoint(), checkpoint_b = b->checkpoint();
        llama_kv_stream_host_layer plane_a, plane_b;
        t.assert_true(a->host()->layer(0, plane_a) && b->host()->layer(0, plane_b));
        t.assert_true(plane_a.k != plane_b.k && plane_a.v != plane_b.v);
        t.assert_equal(uint8_t(0x11), *static_cast<uint8_t *>(plane_a.k));
        t.assert_equal(uint8_t(0x41), *static_cast<uint8_t *>(plane_b.k));
        t.assert_true(a->truncate(4));
        t.assert_equal(checkpoint_b.generation, b->identity().generation);
        t.assert_equal(checkpoint_b.tokens, b->tokens());
        t.assert_true(!a->restore(checkpoint_b) && !b->restore(checkpoint_a));
        t.assert_true(append(t, *a, 3, 0x21));
        t.assert_equal(uint8_t(0x41), *static_cast<uint8_t *>(plane_b.k));
        t.assert_true(append(t, *b, 24, 0x51));
        t.assert_equal(size_t(33), b->tokens());
        t.assert_true(!b->begin(1));
        t.assert_true(a->begin(26));
        t.assert_true(a->cancel());
        t.assert_equal(size_t(7), a->tokens());
    });
    t.test("unretired_failed_growth_cannot_reopen_cache_admission", [](testing & t) {
        auto c = host(409)->config();
        c.shape = {GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 256, 256, 4, 256, 128};
        c.context_tokens = 4097;
        ggml_backend_buffer_ptr held;
        bool fail = false;
        auto sparse = llama_kv_stream_host_reservation::create(c, ggml_backend_cpu_buffer_type(),
            [&](ggml_backend_buffer_t owner) -> ggml_backend_buffer_t {
                if (fail) {
                    held.reset(ggml_backend_buffer_retain(owner));
                    return nullptr;
                }
                owner->buft = ggml_backend_cpu_buffer_type();
                return ggml_backend_buffer_retain(owner);
            }, 65536);
        auto cache = llama_kv_stream_logical_cache::create(llama_kv_stream_host::from_reservation(std::move(sparse)));
        if (!t.assert_true(cache && append(t, *cache, 1, 0x31))) return;
        llama_kv_stream_host_layer prefix;
        t.assert_true(cache->host()->layer(0, prefix));
        const auto checkpoint = cache->checkpoint();
        const auto & layout = cache->host()->layout();
        std::vector<uint8_t> k(256*layout.k_token_bytes, 0x51), v(256*layout.v_token_bytes, 0x61);
        llama_kv_stream_write write;
        t.assert_true(cache->begin(256));
        t.assert_true(cache->content()->prepare({
            {0, operand::k, layout.k_token_bytes, k.data(), k.size()},
            {0, operand::v, layout.v_token_bytes, v.data(), v.size()}}, write));
        fail = true;
        t.assert_true(!cache->publish_host(write));
        t.assert_equal(uint8_t(0x31), *static_cast<uint8_t *>(prefix.k));
        t.assert_true(cache->cancel());
        t.assert_true(!cache->ready());
        const bool reopened = cache->begin(1);
        t.assert_true(!reopened);
        if (reopened) t.assert_true(cache->cancel());
        t.assert_true(!cache->restore(checkpoint));
        t.assert_equal(size_t(1), cache->tokens());
        held.reset();
    });
    t.test("suspension_readiness_requires_acknowledged_host_publication", [](testing & t) {
        auto cache = llama_kv_stream_logical_cache::create(host(909));
        if (!t.assert_true(bool(cache) && cache->ready())) return;
        if (!t.assert_true(cache->begin(1))) return;
        t.assert_true(!cache->ready());
        t.assert_true(cache->cancel() && cache->ready());
        const auto backing = cache->host();
        std::vector<uint8_t> k(backing->layout().k_token_bytes,1), v(backing->layout().v_token_bytes,2);
        llama_kv_stream_write write;
        if (!t.assert_true(cache->begin(1) && cache->content()->prepare({
                {0,operand::k,0,k.data(),k.size()},{0,operand::v,0,v.data(),v.size()}},write) &&
                cache->publish_host(write))) return;
        t.assert_true(!cache->ready());
        t.assert_true(cache->finish() && cache->ready());
        t.assert_true(cache->content()->invalidate());
        t.assert_true(!cache->ready());
    });
    t.test("suffix_truncation_preserves_clean_resident_rows_but_restore_invalidates_them", [](testing & t) {
        auto cache = llama_kv_stream_logical_cache::create(host(202));
        if (!t.assert_true(cache && append(t,*cache,4,0x20))) return;
        const auto content=cache->content();
        t.assert_true(content->flush({{0,operand::k,0,4},{0,operand::v,0,4}},[](const auto &) { return true; }));
        t.assert_true(cache->truncate(2));
        bool dirty=true;
        t.assert_true(content->dirty({0,operand::k,0,2},dirty));
        t.assert_true(!dirty);
        t.assert_true(content->dirty({0,operand::v,2,2},dirty));
        t.assert_true(dirty);
        t.assert_true(cache->restore(cache->checkpoint()));
        t.assert_true(content->dirty({0,operand::k,0,2},dirty));
        t.assert_true(dirty);
    });
    t.test("target_and_mtp_have_distinct_host_bytes_and_publication_revisions", [](testing & t) {
        auto target = llama_kv_stream_logical_cache::create(host(101));
        auto mtp = llama_kv_stream_logical_cache::create(host(202));
        if (!t.assert_true(target && mtp)) return;
        t.assert_true(target->host() != mtp->host());
        t.assert_true(target->content() != mtp->content());
        t.assert_equal(uint64_t(101), target->identity().id);
        t.assert_equal(uint64_t(202), mtp->identity().id);
        t.assert_true(append(t, *mtp, 3, 0x31));
        t.assert_equal(size_t(0), target->tokens());
        t.assert_equal(size_t(3), mtp->tokens());
        llama_kv_stream_host_layer target_layer, mtp_layer;
        t.assert_true(target->host()->layer(0, target_layer) && mtp->host()->layer(0, mtp_layer));
        t.assert_equal(uint8_t(0), static_cast<uint8_t *>(target_layer.k)[0]);
        t.assert_equal(uint8_t(0x31), static_cast<uint8_t *>(mtp_layer.k)[0]);
        t.assert_equal(uint8_t(0x32), static_cast<uint8_t *>(mtp_layer.v)[0]);
        t.assert_true(mtp->identity().generation > target->identity().generation);
    });

    t.test("cancel_and_truncate_invalidate_stale_writes_without_touching_other_cache", [](testing & t) {
        auto target = llama_kv_stream_logical_cache::create(host(101));
        auto mtp = llama_kv_stream_logical_cache::create(host(202));
        if (!t.assert_true(target && mtp && append(t, *mtp, 4, 0x20))) return;
        const auto before = mtp->identity();
        if (!t.assert_true(mtp->begin(2))) return;
        uint8_t value = 9;
        llama_kv_stream_write stale;
        if (!t.assert_true(mtp->content()->prepare({{0, operand::k, 0, &value, 1}}, stale))) return;
        t.assert_true(mtp->cancel());
        t.assert_equal(size_t(4), mtp->tokens());
        t.assert_true(!mtp->content()->commit(stale));
        t.assert_true(mtp->identity().generation > before.generation);
        t.assert_true(mtp->truncate(1));
        t.assert_equal(size_t(1), mtp->tokens());
        t.assert_true(!mtp->truncate(2));
        t.assert_equal(size_t(0), target->tokens());
        t.assert_true(append(t, *mtp, 2, 0x40));
        t.assert_equal(size_t(3), mtp->tokens());
    });

    t.test("external_save_restore_checks_cache_identity_and_rebuilds_frontiers", [](testing & t) {
        auto mtp = llama_kv_stream_logical_cache::create(host(202));
        if (!t.assert_true(mtp && append(t, *mtp, 2, 0x50))) return;
        const auto saved = mtp->checkpoint();
        llama_kv_stream_host_layer layer;
        if (!t.assert_true(mtp->host()->layer(0, layer))) return;
        std::vector<uint8_t> saved_bytes(mtp->host()->bytes());
        std::memcpy(saved_bytes.data(), layer.k, saved_bytes.size());
        if (!t.assert_true(append(t, *mtp, 2, 0x70))) return;
        std::memcpy(layer.k, saved_bytes.data(), saved_bytes.size());
        const auto foreign = llama_kv_stream_logical_cache::create(host(303));
        if (!t.assert_true(bool(foreign))) return;
        t.assert_true(!foreign->restore(saved));
        t.assert_true(mtp->restore(saved));
        t.assert_equal(size_t(2), mtp->tokens());
        t.assert_equal(size_t(2), mtp->frontiers().reserved);
        t.assert_equal(size_t(0), mtp->frontiers().device);
        t.assert_equal(size_t(2), mtp->frontiers().host);
        t.assert_equal(size_t(2), mtp->frontiers().committed);
        t.assert_true(std::memcmp(layer.k, saved_bytes.data(), saved_bytes.size()) == 0);
        t.assert_true(mtp->identity().generation > saved.generation);
        t.assert_true(append(t, *mtp, 1, 0x60));
    });
    t.test("partial_or_out_of_range_kv_cannot_publish_a_token_frontier", [](testing & t) {
        auto mtp = llama_kv_stream_logical_cache::create(host(404));
        if (!t.assert_true(bool(mtp) && mtp->begin(2))) return;
        const auto backing = mtp->host();
        const size_t k_stride = backing->layout().k_token_bytes;
        const size_t v_stride = backing->layout().v_token_bytes;
        std::vector<uint8_t> k(k_stride*2, 1), v(v_stride*2, 2);
        llama_kv_stream_write partial, extra, complete;
        t.assert_true(mtp->content()->prepare({{0, operand::k, 0, k.data(), k.size()}}, partial));
        t.assert_true(!mtp->publish_host(partial));
        t.assert_equal(size_t(0), mtp->frontiers().host);
        t.assert_true(mtp->content()->prepare({
            {0, operand::k, 0, k.data(), k.size()},
            {0, operand::v, v_stride, v.data(), v_stride}}, extra));
        t.assert_true(!mtp->publish_host(extra));
        t.assert_true(mtp->content()->prepare({
            {0, operand::k, 0, k.data(), k.size()},
            {0, operand::v, 0, v.data(), v.size()}}, complete));
        t.assert_true(mtp->publish_host(complete));
        t.assert_true(mtp->finish());
        t.assert_equal(size_t(2), mtp->tokens());
    });
    t.test("generated_prompt_chunks_publish_only_after_writer_completion", [](testing & t) {
        ggml_backend_ptr backend(ggml_backend_cpu_init());
        ggml_backend_buffer_ptr scratch(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), 4096));
        auto backing = host(505, GGML_TYPE_F16, GGML_TYPE_F16);
        if (!t.assert_true(bool(backend) && bool(scratch) && bool(backing))) return;
        auto cache = llama_kv_stream_logical_cache::create(backing);
        auto unique = llama_kv_stream_writer::create(backend.get(), scratch.get(), 4096, backing->config().shape, 9);
        if (!t.assert_true(bool(cache) && bool(unique))) return;
        std::shared_ptr<llama_kv_stream_writer> writer(std::move(unique));
        generated_source first(backend.get(), 9);
        if (!t.assert_true(cache->begin_generated(writer, first.k, first.v))) return;
        t.assert_equal(size_t(9), cache->frontiers().reserved);
        t.assert_equal(size_t(0), cache->frontiers().host);
        t.assert_equal(size_t(0), cache->tokens());
        if (!t.assert_true(cache->complete_generated())) return;
        t.assert_equal(size_t(9), cache->tokens());
        t.assert_equal(size_t(0), cache->frontiers().device);
        generated_source second(backend.get(), 8);
        if (!t.assert_true(cache->begin_generated(writer, second.k, second.v) &&
                cache->complete_generated())) return;
        t.assert_equal(size_t(17), cache->tokens());
        llama_kv_stream_host_layer planes;
        if (!t.assert_true(backing->layer(0, planes))) return;
        const auto * keys = static_cast<const ggml_fp16_t *>(planes.k);
        const auto * values = static_cast<const ggml_fp16_t *>(planes.v);
        for (size_t row : {size_t(0), size_t(8), size_t(9), size_t(16)}) {
            t.assert_equal(ggml_fp32_to_fp16(1.25f), keys[row*32]);
            t.assert_equal(ggml_fp32_to_fp16(2.5f), values[row*32]);
        }
    });
    t.test("cancel_generated_append_drains_before_reusing_host_rows", [](testing & t) {
        ggml_backend_ptr backend(ggml_backend_cpu_init());
        ggml_backend_buffer_ptr scratch(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), 4096));
        auto backing = host(606, GGML_TYPE_F16, GGML_TYPE_F16);
        if (!t.assert_true(bool(backend) && bool(scratch) && bool(backing))) return;
        auto cache = llama_kv_stream_logical_cache::create(backing);
        auto unique = llama_kv_stream_writer::create(backend.get(), scratch.get(), 4096, backing->config().shape, 9);
        if (!t.assert_true(bool(cache) && bool(unique))) return;
        std::shared_ptr<llama_kv_stream_writer> writer(std::move(unique));
        generated_source source(backend.get(), 3);
        if (!t.assert_true(cache->begin_generated(writer, source.k, source.v))) return;
        t.assert_equal(size_t(0), cache->frontiers().host);
        t.assert_true(cache->cancel());
        t.assert_equal(size_t(0), cache->tokens());
        t.assert_true(cache->begin_generated(writer, source.k, source.v));
        t.assert_true(cache->complete_generated());
        t.assert_equal(size_t(3), cache->tokens());
    });
    t.test("wide_generated_prefill_crosses_page_and_writer_tile_boundaries", [](testing & t) {
        ggml_backend_ptr backend(ggml_backend_cpu_init());
        ggml_backend_buffer_ptr scratch(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), 4096));
        auto config = host(707, GGML_TYPE_F16, GGML_TYPE_F16)->config();
        config.context_tokens = 513;
        config.shape.page_tokens = 256;
        auto backing = llama_kv_stream_host::create(config, ggml_backend_cpu_buffer_type());
        if (!t.assert_true(bool(backend) && bool(scratch) && bool(backing))) return;
        auto cache = llama_kv_stream_logical_cache::create(backing);
        auto unique = llama_kv_stream_writer::create(backend.get(), scratch.get(), 4096, config.shape, 257);
        if (!t.assert_true(bool(cache) && bool(unique))) return;
        std::shared_ptr<llama_kv_stream_writer> writer(std::move(unique));
        generated_source first(backend.get(), 257);
        auto wrong_shape = config.shape;
        wrong_shape.type_k = GGML_TYPE_Q8_0;
        wrong_shape.type_v = GGML_TYPE_Q4_0;
        ggml_backend_buffer_ptr wrong_scratch(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), 4096));
        if (!t.assert_true(bool(wrong_scratch))) return;
        auto wrong_unique = llama_kv_stream_writer::create(backend.get(), wrong_scratch.get(), 4096, wrong_shape, 257);
        if (!t.assert_true(bool(wrong_unique))) return;
        std::shared_ptr<llama_kv_stream_writer> wrong_writer(std::move(wrong_unique));
        t.assert_true(!cache->begin_generated(wrong_writer, first.k, first.v));
        t.assert_equal(size_t(0), cache->frontiers().reserved);
        generated_source wrong(backend.get(), 3);
        t.assert_true(!cache->begin_generated(writer, first.k, wrong.v));
        t.assert_equal(size_t(0), cache->tokens());
        if (!t.assert_true(cache->begin_generated(writer, first.k, first.v))) return;
        t.assert_equal(size_t(0), cache->frontiers().host);
        if (!t.assert_true(cache->complete_generated())) return;
        t.assert_true(writer->stats().graph_submissions > 2);
        generated_source second(backend.get(), 256);
        if (!t.assert_true(cache->begin_generated(writer, second.k, second.v) &&
                cache->complete_generated())) return;
        t.assert_equal(size_t(513), cache->tokens());
        t.assert_equal(size_t(0), cache->frontiers().device);
        llama_kv_stream_host_layer planes;
        if (!t.assert_true(backing->layer(0, planes))) return;
        const auto * keys = static_cast<const ggml_fp16_t *>(planes.k);
        const auto * values = static_cast<const ggml_fp16_t *>(planes.v);
        for (size_t row : {size_t(0), size_t(255), size_t(256), size_t(257), size_t(512)}) {
            t.assert_equal(ggml_fp32_to_fp16(1.25f), keys[row*32]);
            t.assert_equal(ggml_fp32_to_fp16(2.5f), values[row*32]);
        }
    });
    return t.summary();
}
