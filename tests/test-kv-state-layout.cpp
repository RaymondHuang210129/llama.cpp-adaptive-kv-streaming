#include "../src/llama-kv-cache.h"
#include "../src/llama-model.h"
#include "../src/llama-io.h"
#include "ggml-cpu.h"
#include "ggml-cpp.h"
#include "testing.h"

#include <cstring>

struct state_writer : llama_io_write_i {
    std::vector<uint8_t> bytes;
    void write(const void * data, size_t size) override {
        const auto first = bytes.size(); bytes.resize(first + size);
        if (size) std::memcpy(bytes.data() + first, data, size);
    }
    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        const auto first = bytes.size(); bytes.resize(first + size);
        ggml_backend_tensor_get(tensor, bytes.data() + first, offset, size);
    }
    size_t n_bytes() override { return bytes.size(); }
};

struct state_reader : llama_io_read_i {
    const std::vector<uint8_t> & bytes;
    size_t offset = 0;
    explicit state_reader(const std::vector<uint8_t> & bytes) : bytes(bytes) {}
    void read(void * data, size_t size) override {
        if (size > bytes.size() - offset) throw std::runtime_error("truncated state");
        if (size) std::memcpy(data, bytes.data() + offset, size);
        offset += size;
    }
    void read_tensor(ggml_tensor * tensor, size_t start, size_t size) override {
        if (size > bytes.size() - offset) throw std::runtime_error("truncated tensor");
        ggml_backend_tensor_set(tensor, bytes.data() + offset, start, size); offset += size;
    }
    size_t n_bytes() override { return offset; }
};

struct fixture {
    std::unique_ptr<llama_model> model{llama_model_create(LLM_ARCH_LLAMA, llama_model_default_params())};
    std::unique_ptr<llama_kv_cache> cache;
    explicit fixture(float bias = 0, uint32_t capacity = 32, bool transposed = false) {
        auto & hp = model->hparams;
        hp.n_layer_all = 1; hp.n_embd = 8; hp.n_head_arr[0] = hp.n_head_kv_arr[0] = 1;
        hp.n_embd_head_k_full = hp.n_embd_head_v_full = 8;
        hp.n_rot_full = 8;
        cache = std::make_unique<llama_kv_cache>(*model, hp, GGML_TYPE_F32, GGML_TYPE_F32,
            transposed, false, true, capacity, 4, 1, 0, LLAMA_SWA_TYPE_NONE, nullptr, nullptr, nullptr, nullptr);
        std::vector<float> values(size_t(capacity)*8);
        for (size_t i = 0; i < values.size(); ++i) values[i] = float(i + 1) + bias;
        for (auto * tensor : {cache->get_k_storage(0), cache->get_v_storage(0)})
            ggml_backend_tensor_set(tensor, values.data(), 0, values.size()*sizeof(float));
    }
    void place(llama_seq_id slot, const std::vector<uint32_t> & rows) {
        llama_batch_allocr allocator(1); auto batch = allocator.ubatch_reserve(uint32_t(rows.size()), 1);
        batch.seq_id_unq[0] = slot;
        for (size_t i = 0; i < rows.size(); ++i) {
            batch.token[i] = 1; batch.pos[i] = llama_pos(i); batch.n_seq_id[i] = 1; batch.seq_id[i] = &slot;
        }
        llama_kv_cache::slot_info info{0, 0, {0}, {rows}};
        cache->apply_ubatch(info, batch);
    }
    std::vector<uint8_t> save(llama_seq_id slot) {
        state_writer writer; cache->state_write(writer, slot, 0); return writer.bytes;
    }
    void restore(const std::vector<uint8_t> & bytes, llama_seq_id slot) {
        state_reader reader(bytes); cache->state_read(reader, slot, 0);
        if (reader.n_bytes() != bytes.size()) throw std::runtime_error("unconsumed state");
    }
};

int main() {
    testing t;
    ggml_backend_ptr backend(ggml_backend_cpu_init());
    const std::vector<uint32_t> first{0, 4, 8, 12}, second{1, 5, 9, 13};
    t.test("sequence_and_whole_restore_preserve_global_holes_head_and_payload", [&](testing & t) {
        fixture source; source.place(0, first); source.place(1, second);
        const auto a = source.save(0), b = source.save(1), whole = source.save(-1);
        for (bool full : {false, true}) {
            fixture destination(-1000);
            if (full) destination.restore(whole, -1);
            else { destination.restore(b, 1); destination.restore(a, 0); }
            t.assert_equal(source.cache->get_head(0), destination.cache->get_head(0));
            for (uint32_t row = 0; row < 32; ++row) {
                const auto & x = source.cache->get_cells(0); const auto & y = destination.cache->get_cells(0);
                t.assert_equal(x.is_empty(row), y.is_empty(row));
                if (x.is_empty(row)) continue;
                t.assert_equal(x.pos_get(row), y.pos_get(row));
                t.assert_equal(x.seq_get(row), y.seq_get(row));
                for (auto * tensor : {destination.cache->get_k_storage(0), destination.cache->get_v_storage(0)}) {
                    float values[8]; ggml_backend_tensor_get(tensor, values, size_t(row)*32, sizeof(values));
                    for (int i = 0; i < 8; ++i) t.assert_equal(float(row*8 + i + 1), values[i]);
                }
            }
        }
    });
    t.test("occupied_saved_rows_preserve_peer_and_legacy_states_remain_readable", [&](testing & t) {
        fixture source; source.place(0, first); const auto bytes = source.save(0);
        fixture destination(-1000); destination.place(2, {0, 8}); destination.restore(bytes, 0);
        t.assert_true(destination.cache->get_cells(0).seq_has(0, 2) && destination.cache->get_cells(0).seq_has(8, 2));
        t.assert_equal(size_t(4), destination.cache->get_cells(0).seq_pos_get(0).size());
        std::vector<uint8_t> legacy(bytes.begin() + 4, bytes.begin() + 12);
        legacy.insert(legacy.end(), bytes.begin() + 20 + first.size()*4, bytes.end());
        fixture old(-1000); old.restore(legacy, 0);
        t.assert_equal(size_t(4), old.cache->get_cells(0).seq_pos_get(0).size());
        for (auto row : {0u, 8u}) {
            float value = 0; ggml_backend_tensor_get(destination.cache->get_k_storage(0), &value, size_t(row)*32, sizeof(value));
            t.assert_equal(float(row*8 + 1) - 1000, value);
        }
        fixture smaller(-1000, 8); smaller.restore(bytes, 0);
        t.assert_equal(size_t(4), smaller.cache->get_cells(0).seq_pos_get(0).size());
        for (const auto & point : destination.cache->get_cells(0).seq_pos_get(0)) {
            float value = 0; ggml_backend_tensor_get(destination.cache->get_k_storage(0), &value, size_t(point.second)*32, sizeof(value));
            t.assert_equal(float(first[size_t(point.first)]*8 + 1), value);
        }
    });
    t.test("transposed_values_restore_into_original_and_smaller_layouts", [&](testing & t) {
        fixture source(0, 32, true); source.place(0, first);
        const auto bytes = source.save(0);
        for (uint32_t capacity : {32u, 8u}) {
            fixture destination(-1000, capacity, true); destination.restore(bytes, 0);
            for (const auto & position : destination.cache->get_cells(0).seq_pos_get(0)) {
                for (uint32_t feature = 0; feature < 8; ++feature) {
                    float value = 0;
                    ggml_backend_tensor_get(destination.cache->get_v_storage(0), &value,
                        (size_t(feature)*capacity + position.second)*sizeof(float), sizeof(value));
                    t.assert_equal(float(feature*32 + first[size_t(position.first)] + 1), value);
                }
            }
        }
    });
    t.test("mirrored_whole_restore_uses_the_primary_row_layout", [&](testing & t) {
        fixture primary; primary.place(0, first); primary.place(1, second);
        fixture secondary; secondary.place(0, {2, 6, 10, 14}); secondary.place(1, {3, 7, 11, 15});
        fixture x(-1000), y(-1000);
        const auto a = primary.save(-1), b = secondary.save(-1);
        state_reader read_a(a), read_b(b);
        llama_kv_cache::slot_info_vec_t layout;
        x.cache->state_read_sinfo(read_a, -1, 0, &layout, nullptr);
        y.cache->state_read_sinfo(read_b, -1, 0, nullptr, &layout);
        t.assert_equal("mirrored allocation cursors remain identical",x.cache->get_head(0),y.cache->get_head(0));
        for (uint32_t row = 0; row < 32; ++row) {
            const auto & expected = x.cache->get_cells(0); const auto & actual = y.cache->get_cells(0);
            t.assert_equal(expected.is_empty(row), actual.is_empty(row));
            if (!expected.is_empty(row)) {
                t.assert_equal(expected.seq_get(row), actual.seq_get(row));
                t.assert_equal(expected.pos_get(row), actual.pos_get(row));
                float value = 0; ggml_backend_tensor_get(y.cache->get_k_storage(0), &value, size_t(row)*32, sizeof(value));
                t.assert_equal(float((row + 2)*8 + 1), value);
            }
        }
    });
    t.test("invalid_saved_row_metadata_rejects_before_import", [&](testing & t) {
        fixture source; source.place(0, first); const auto bytes = source.save(0);
        for (auto corrupt : {std::pair<size_t, uint32_t>{16, 33}, {24, 0}, {20, 32}, {8, UINT32_MAX}, {12, 0}}) {
            auto invalid = bytes; std::memcpy(invalid.data() + corrupt.first, &corrupt.second, 4);
            fixture destination; bool rejected = false;
            try { destination.restore(invalid, 0); } catch (const std::exception &) { rejected = true; }
            t.assert_true(rejected && !destination.cache->get_cells(0).get_used());
        }
        for (size_t length : {0u, 4u, 8u, 12u, 16u, 20u, 24u}) {
            auto invalid = bytes; invalid.resize(length);
            fixture destination; bool rejected = false;
            try { destination.restore(invalid, 0); } catch (const std::exception &) { rejected = true; }
            t.assert_true(rejected && !destination.cache->get_cells(0).get_used());
        }
    });
    return t.summary();
}
