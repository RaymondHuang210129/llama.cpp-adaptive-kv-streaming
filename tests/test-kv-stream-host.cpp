#include "../src/llama-kv-stream-host.h"
#include "../src/llama-kv-stream-binding.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "../ggml/src/ggml-backend-execution.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "testing.h"

#ifdef KV_HOST_CUDA_TEST
#include "ggml-cuda.h"
#endif

#include <cstring>
#include <limits>

static llama_kv_stream_host_config config(int k = GGML_TYPE_Q8_0, int v = GGML_TYPE_Q4_0) {
    return {71, {k, v, 256, 256, 4, 256, 128},
        {{k, true, true, true, true}, {v, true, true, true, true}, true, true}, 257, 3};
}

struct allocation_probe {
    ggml_backend_buffer_type type = *ggml_backend_cpu_buffer_type();
    size_t calls = 0;
    bool fail = false, fallback = false, throws = false;
    allocation_probe() {
        type.context = this;
        type.iface.alloc_buffer = [](ggml_backend_buffer_type_t type, size_t size) {
            auto & p = *static_cast<allocation_probe *>(type->context);
            ++p.calls;
            if (p.throws) throw std::bad_alloc();
            if (p.fail) return ggml_backend_buffer_t(nullptr);
            auto * result = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
            if (result && !p.fallback) result->buft = type;
            return result;
        };
    }
};

struct registration_probe {
    ggml_backend_buffer_type type = *ggml_backend_cpu_buffer_type();
    size_t attempts = 0, live = 0, fail_at = SIZE_MAX;
    bool fallback = false, throws = false;
    struct context {
        registration_probe * probe;
        ggml_backend_buffer_t owner;
    };
    ggml_backend_buffer_t pin(ggml_backend_buffer_t owner) {
        if (++attempts == fail_at) {
            if (throws) throw std::bad_alloc();
            return nullptr;
        }
        auto * buffer = ggml_backend_cpu_buffer_from_ptr(ggml_backend_buffer_get_base(owner), ggml_backend_buffer_get_size(owner));
        auto * state = new context{this, ggml_backend_buffer_retain(owner)};
        buffer->buft = fallback ? ggml_backend_cpu_buffer_type() : &type;
        buffer->context = state;
        buffer->iface.get_base = [](ggml_backend_buffer_t b) {
            return ggml_backend_buffer_get_base(static_cast<context *>(b->context)->owner);
        };
        buffer->iface.free_buffer = [](ggml_backend_buffer_t b) {
            auto * c = static_cast<context *>(b->context);
            --c->probe->live;
            ggml_backend_buffer_free(c->owner);
            delete c;
        };
        buffer->iface.clear = [](ggml_backend_buffer_t b, uint8_t value) {
            ggml_backend_buffer_clear(static_cast<context *>(b->context)->owner, value);
        };
        buffer->view_buffer = nullptr;
        ++live;
        return buffer;
    }
};

int main(int argc, char ** argv) {
    testing t;
    t.test("reserved_history_uses_checked_io_without_a_whole_host_buffer", [](testing & t) {
        registration_probe probe;
        auto c = config(); c.context_tokens = 8192;
        auto sparse = llama_kv_stream_host_reservation::create(c, &probe.type,
            [&](ggml_backend_buffer_t b) { return probe.pin(b); }, 65536);
        auto host = llama_kv_stream_host::from_reservation(std::move(sparse));
        if (!t.assert_true(host && host->buffer() == nullptr)) return;
        struct owner { std::shared_ptr<llama_kv_stream_host> host; size_t modified = 0, destroyed = 0; } state{host};
        ggml_backend_execution_ops ops{
            [](void *, const ggml_tensor *) { return false; },
            [](void *, ggml_backend_t, ggml_tensor *) { return GGML_STATUS_FAILED; },
            [](void *) { return true; },
            [](void * p) { ++static_cast<owner *>(p)->modified; },
            [](void * p) { ++static_cast<owner *>(p)->destroyed; },
        };
        ggml_backend_ptr backend(ggml_backend_cpu_init());
        ggml_backend_buffer_ptr managed(host->execution_buffer(ggml_backend_get_device(backend.get()), ops, &state));
        if (!t.assert_true(managed && !ggml_backend_buffer_is_host(managed.get()))) return;
        llama_kv_stream_host_layer plane;
        t.assert_true(host->layer(1, plane));
        ggml_context_ptr metadata(ggml_init({ggml_tensor_overhead(), nullptr, true}));
        auto * tensor = ggml_new_tensor_1d(metadata.get(), GGML_TYPE_I8, int64_t(host->layout().v_bytes));
        t.assert_equal(GGML_STATUS_SUCCESS, ggml_backend_tensor_alloc(managed.get(), tensor, plane.v));
        const size_t offset = 200*host->layout().v_token_bytes + 3;
        std::vector<uint8_t> input(67, 0x51), output(input.size());
        ggml_backend_tensor_set(tensor, input.data(), offset, input.size());
        ggml_backend_tensor_get(tensor, output.data(), offset, output.size());
        t.assert_true(input == output);
        t.assert_equal(size_t(1), state.modified);
        ggml_backend_tensor_memset(tensor, 0x19, offset + 5, 11);
        ggml_backend_tensor_get(tensor, output.data(), offset, output.size());
        std::fill(input.begin() + 5, input.begin() + 16, 0x19);
        t.assert_true(input == output);
        std::vector<llama_kv_stream_host_range> pieces;
        const auto * source = static_cast<const char *>(plane.v) + offset;
        t.assert_true(host->address_ranges(source, input.size(), pieces));
        const size_t piece_count = pieces.size();
        t.assert_true(!host->address_ranges(static_cast<const char *>(plane.v) + host->layout().v_bytes - 1,
            input.size(), pieces));
        t.assert_equal(piece_count, pieces.size());
        ggml_context_ptr copy_metadata(ggml_init({ggml_tensor_overhead(), nullptr, true}));
        auto * copied = ggml_new_tensor_1d(copy_metadata.get(), GGML_TYPE_I8, 128);
        ggml_backend_buffer_ptr copy_buffer(ggml_backend_alloc_ctx_tensors(copy_metadata.get(), backend.get()));
        size_t calls = 0;
        t.assert_true(host->upload(backend.get(), copied, 7, source, input.size(), calls));
        t.assert_equal(size_t(1), calls);
        ggml_backend_synchronize(backend.get());
        ggml_backend_tensor_get(copied, output.data(), 7, output.size());
        t.assert_true(input == output);
        t.assert_true(!host->upload(backend.get(), copied, 100, source, input.size(), calls));
        t.assert_equal(size_t(1), calls);
        t.assert_equal(size_t(2), state.modified);
        ggml_backend_buffer_clear(managed.get(), 0);
        ggml_backend_tensor_get(tensor, output.data(), offset, output.size());
        t.assert_true(std::all_of(output.begin(), output.end(), [](uint8_t b) { return b == 0; }));
        t.assert_equal(size_t(3), state.modified);
        managed.reset();
        t.assert_equal(size_t(1), state.destroyed);
    });
    t.test("demand_backing_reserves_all_planes_without_committing_them", [](testing & t) {
        registration_probe probe;
        auto c = config(); c.context_tokens = 65536;
        auto host = llama_kv_stream_host_reservation::create(c, &probe.type,
            [&](ggml_backend_buffer_t b) { return probe.pin(b); }, 65536);
        if (!t.assert_true(host != nullptr)) return;
        t.assert_equal(size_t(0), host->committed_bytes());
        t.assert_equal(size_t(0), host->registered_bytes());
        t.assert_equal(size_t(0), probe.attempts);
        llama_kv_stream_host_layer before, after;
        t.assert_true(host->layer(2, before));
        t.assert_true(host->prepare(257));
        t.assert_true(host->layer(2, after));
        t.assert_true(before.k == after.k && before.v == after.v);
        t.assert_true(host->committed_bytes() < host->bytes()/16);
        t.assert_equal(host->committed_bytes(), host->registered_bytes());
        const size_t committed = host->committed_bytes(), calls = probe.attempts;
        t.assert_true(host->prepare(257));
        t.assert_equal(committed, host->committed_bytes());
        t.assert_equal(calls, probe.attempts);
        std::vector<llama_kv_stream_host_range> ranges;
        t.assert_true(host->ranges(2, ggml_kv_stream_operand::k, 0, 257*host->layout().k_token_bytes, ranges));
        size_t total = 0;
        for (const auto & range : ranges) {
            auto * bytes = static_cast<uint8_t *>(ggml_backend_buffer_get_base(range.buffer.get())) + range.offset;
            t.assert_true(std::all_of(bytes, bytes + range.bytes, [](uint8_t b) { return b == 0; }));
            std::memset(bytes, 0x38, range.bytes);
            total += range.bytes;
        }
        t.assert_equal(257*host->layout().k_token_bytes, total);
        t.assert_true(host->prepare(513));
        t.assert_equal(uint8_t(0x38), *static_cast<uint8_t *>(before.k));
        t.assert_true(host->truncate(128));
        t.assert_equal(uint8_t(0x38), *(static_cast<uint8_t *>(before.k) + 127*host->layout().k_token_bytes));
        t.assert_equal(uint8_t(0), *(static_cast<uint8_t *>(before.k) + 128*host->layout().k_token_bytes));
        host->clear();
        t.assert_equal(uint8_t(0), *static_cast<uint8_t *>(before.k));
        t.assert_true(host->prepare(257));
        const size_t retained = ranges.size();
        host.reset();
        t.assert_equal(retained, probe.live);
        t.assert_true(!ranges.empty() && ranges.front().buffer != nullptr);
        ranges.clear();
        t.assert_equal(size_t(0), probe.live);
    });
    t.test("failed_demand_growth_rolls_back_only_new_chunks", [](testing & t) {
        registration_probe probe;
        auto c = config(); c.context_tokens = 8192;
        auto host = llama_kv_stream_host_reservation::create(c, &probe.type,
            [&](ggml_backend_buffer_t b) { return probe.pin(b); }, 65536);
        if (!t.assert_true(host && host->prepare(1))) return;
        llama_kv_stream_host_layer layer;
        t.assert_true(host->layer(0, layer));
        *static_cast<uint8_t *>(layer.k) = 0x51;
        const size_t committed = host->committed_bytes(), registered = host->registered_bytes(), live = probe.live;
        probe.fail_at = probe.attempts + 3;
        t.assert_true(!host->prepare(513));
        t.assert_equal(committed, host->committed_bytes());
        t.assert_equal(registered, host->registered_bytes());
        t.assert_equal(live, probe.live);
        t.assert_equal(uint8_t(0x51), *static_cast<uint8_t *>(layer.k));
        probe.fail_at = SIZE_MAX;
        t.assert_true(host->prepare(513));
        t.assert_equal(uint8_t(0x51), *static_cast<uint8_t *>(layer.k));
        std::vector<llama_kv_stream_host_range> ranges;
        t.assert_true(host->ranges(0, ggml_kv_stream_operand::k, 0, 32, ranges));
        const size_t existing = ranges.size();
        t.assert_true(!host->ranges(c.layers, ggml_kv_stream_operand::k, 0, 32, ranges));
        t.assert_true(!host->ranges(0, ggml_kv_stream_operand::k, 8191*host->layout().k_token_bytes, 32, ranges));
        t.assert_equal(existing, ranges.size());
        t.assert_true(!host->prepare(SIZE_MAX));
        probe.fallback = true;
        t.assert_true(!host->prepare(1025));
        t.assert_equal(uint8_t(0x51), *static_cast<uint8_t *>(layer.k));
    });
    t.test("all_registration_failure_positions_preserve_previous_plane_bytes", [](testing & t) {
        auto c = config(); c.context_tokens = 1025; c.layers = 2;
        size_t growth_calls = 0;
        {
            registration_probe probe;
            auto host = llama_kv_stream_host_reservation::create(c, &probe.type,
                [&](ggml_backend_buffer_t b) { return probe.pin(b); }, 65536);
            if (!t.assert_true(host && host->prepare(1))) return;
            const size_t initial = probe.attempts;
            if (!t.assert_true(host->prepare(1025))) return;
            growth_calls = probe.attempts - initial;
        }
        t.assert_true(growth_calls > 3);
        for (bool throws : {false, true}) for (size_t fail = 1; fail <= growth_calls; ++fail) {
            registration_probe probe;
            auto host = llama_kv_stream_host_reservation::create(c, &probe.type,
                [&](ggml_backend_buffer_t b) { return probe.pin(b); }, 65536);
            if (!t.assert_true(host && host->prepare(1))) return;
            llama_kv_stream_host_layer layer;
            t.assert_true(host->layer(1, layer));
            *static_cast<uint8_t *>(layer.v) = 0x5a;
            const size_t committed = host->committed_bytes(), live = probe.live;
            probe.fail_at = probe.attempts + fail;
            probe.throws = throws;
            t.assert_true(!host->prepare(1025));
            t.assert_equal(committed, host->committed_bytes());
            t.assert_equal(committed, host->registered_bytes());
            t.assert_equal(live, probe.live);
            t.assert_equal(uint8_t(0x5a), *static_cast<uint8_t *>(layer.v));
            probe.fail_at = SIZE_MAX;
            t.assert_true(host->prepare(1025));
            t.assert_equal(uint8_t(0x5a), *static_cast<uint8_t *>(layer.v));
        }
    });
    t.test("demand_planes_preserve_all64_formats_at_both_attention_widths", [](testing & t) {
        const int types[] = {GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0,
            GGML_TYPE_Q5_1, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_IQ4_NL};
        for (int dim : {256, 512}) for (int k : types) for (int v : types) {
            registration_probe probe;
            auto c = config(k, v); c.context_tokens = 513; c.layers = 2;
            c.shape.head_dim_k = c.shape.head_dim_v = dim;
            auto host = llama_kv_stream_host_reservation::create(c, &probe.type,
                [&](ggml_backend_buffer_t b) { return probe.pin(b); }, 65536);
            auto native = llama_kv_stream_host::create(c, ggml_backend_cpu_buffer_type());
            if (!t.assert_true(host && native && host->prepare(257))) return;
            for (uint32_t layer = 0; layer < c.layers; ++layer) for (bool value : {false, true}) {
                llama_kv_stream_host_layer reference, actual;
                t.assert_true(native->layer(layer, reference) && host->layer(layer, actual));
                auto * expected = static_cast<uint8_t *>(value ? reference.v : reference.k);
                auto * observed = static_cast<uint8_t *>(value ? actual.v : actual.k);
                const size_t token_bytes = value ? host->layout().v_token_bytes : host->layout().k_token_bytes;
                for (size_t i = 0; i < 257*token_bytes; ++i) expected[i] = uint8_t(i*37 + layer*71 + value*109);
                std::vector<llama_kv_stream_host_range> ranges;
                if (!t.assert_true(host->ranges(layer, value ? ggml_kv_stream_operand::v : ggml_kv_stream_operand::k,
                        0, 257*token_bytes, ranges))) return;
                size_t copied = 0;
                for (const auto & range : ranges) {
                    std::memcpy(static_cast<char *>(ggml_backend_buffer_get_base(range.buffer.get())) + range.offset,
                        expected + copied, range.bytes);
                    copied += range.bytes;
                }
                t.assert_equal(257*token_bytes, copied);
                t.assert_true(std::memcmp(expected, observed, copied) == 0);
            }
            t.assert_true(host->prepare(513));
            for (uint32_t layer = 0; layer < c.layers; ++layer) for (bool value : {false, true}) {
                llama_kv_stream_host_layer reference, actual;
                t.assert_true(native->layer(layer, reference) && host->layer(layer, actual));
                const auto * expected = static_cast<const uint8_t *>(value ? reference.v : reference.k);
                const auto * observed = static_cast<const uint8_t *>(value ? actual.v : actual.k);
                const size_t token_bytes = value ? host->layout().v_token_bytes : host->layout().k_token_bytes;
                t.assert_true(std::memcmp(expected, observed, 513*token_bytes) == 0);
            }
        }
    });
    ggml_backend_dev_t device = nullptr;
    auto * storage_type = ggml_backend_cpu_buffer_type();
    if (argc > 1 && std::strcmp(argv[1], "--cuda") == 0) {
        ggml_backend_load_all();
#ifdef KV_HOST_CUDA_TEST
        if (!ggml_backend_reg_by_name("CUDA")) ggml_backend_register(ggml_backend_cuda_reg());
#endif
        auto * reg = ggml_backend_reg_by_name("CUDA");
        if (!t.assert_true(reg && ggml_backend_reg_dev_count(reg))) return t.summary();
        device = ggml_backend_reg_dev_get(reg, 0);
        storage_type = llama_kv_stream_host_buffer_type(device);
        if (!t.assert_true(storage_type != nullptr)) return t.summary();
    }
    t.test("separate_planes_padded_context_and_layer_bounds", [&](testing & t) {
        auto c = config();
        auto host = llama_kv_stream_host::create(c, storage_type);
        if (!t.assert_true(host != nullptr)) return;
        t.assert_equal(uint64_t(71), host->cache_id());
        t.assert_equal(size_t(512), host->layout().tokens);
        t.assert_equal(size_t(557056), host->layout().k_bytes);
        t.assert_equal(size_t(294912), host->layout().v_bytes);
        t.assert_equal(size_t(851968), host->stride());
        t.assert_equal(size_t(2555904), host->bytes());
        for (uint32_t i = 0; i < c.layers; ++i) {
            llama_kv_stream_host_layer layer;
            if (!t.assert_true(host->layer(i, layer))) return;
            t.assert_equal(size_t(0), reinterpret_cast<uintptr_t>(layer.k) % c.shape.alignment);
            t.assert_equal(size_t(0), reinterpret_cast<uintptr_t>(layer.v) % c.shape.alignment);
            t.assert_equal(host->layout().v_offset, size_t(static_cast<char *>(layer.v) - static_cast<char *>(layer.k)));
            auto * bytes = static_cast<uint8_t *>(layer.k);
            t.assert_true(std::all_of(bytes, bytes + host->stride(), [](uint8_t b) { return b == 0; }));
        }
        llama_kv_stream_host_layer invalid{reinterpret_cast<void *>(1), reinterpret_cast<void *>(2)};
        t.assert_true(!host->layer(c.layers, invalid));
        t.assert_true(invalid.k == reinterpret_cast<void *>(1));
    });
    t.test("all_online_quant_pairs_use_generic_row_sizes", [](testing & t) {
        for (int k : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_IQ4_NL}) {
            for (int v : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_IQ4_NL}) {
                auto c = config(k, v); c.context_tokens = 1; c.layers = 2;
                auto host = llama_kv_stream_host::create(c, ggml_backend_cpu_buffer_type());
                if (!t.assert_true(host != nullptr)) continue;
                t.assert_equal(ggml_row_size(ggml_type(k), 256)*4*256, host->layout().k_bytes);
                t.assert_equal(ggml_row_size(ggml_type(v), 256)*4*256, host->layout().v_bytes);
                t.assert_equal(host->stride()*2, host->bytes());
            }
        }
    });
    t.test("invalid_or_overflowing_metadata_never_allocates", [](testing & t) {
        allocation_probe p;
        for (int bad = 0; bad < 8; ++bad) {
            auto c = config();
            if (bad == 0) c.cache_id = 0;
            if (bad == 1) c.layers = 0;
            if (bad == 2) c.context_tokens = 0;
            if (bad == 3) c.context_tokens = SIZE_MAX;
            if (bad == 4) c.capabilities.k.online_write = false;
            if (bad == 5) c.shape.type_v = -1;
            if (bad == 6) c.shape.alignment = 3;
            if (bad == 7) { c.layers = UINT32_MAX; c.context_tokens = size_t(1) << 30; }
            t.assert_true(!llama_kv_stream_host::create(c, &p.type));
        }
        t.assert_equal(size_t(0), p.calls);
    });
    t.test("allocation_failure_and_pageable_fallback_are_rejected", [](testing & t) {
        allocation_probe p;
        p.fail = true;
        t.assert_true(!llama_kv_stream_host::create(config(), &p.type));
        p.fail = false; p.fallback = true;
        t.assert_true(!llama_kv_stream_host::create(config(), &p.type));
        p.fallback = false;
        t.assert_true(llama_kv_stream_host::create(config(), &p.type) != nullptr);
        t.assert_equal(size_t(3), p.calls);
        p.throws = true;
        t.assert_true(!llama_kv_stream_host::create(config(), &p.type));
    });
    t.test("import_preserves_bytes_and_retains_buffer", [](testing & t) {
        auto c = config(); c.context_tokens = 1; c.layers = 1;
        auto original = llama_kv_stream_host::create(c, ggml_backend_cpu_buffer_type());
        if (!t.assert_true(original != nullptr)) return;
        ggml_backend_buffer_clear(original->buffer(), 0x31);
        auto host = llama_kv_stream_host::from_buffer(c, ggml_backend_cpu_buffer_type(), original->buffer());
        if (!t.assert_true(host != nullptr)) return;
        original.reset();
        llama_kv_stream_host_layer layer;
        t.assert_true(host->layer(0, layer));
        t.assert_equal(uint8_t(0x31), *static_cast<uint8_t *>(layer.k));
        t.assert_equal(uint8_t(0x31), *static_cast<uint8_t *>(layer.v));
        ggml_backend_buffer_ptr small(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), 128));
        t.assert_true(!llama_kv_stream_host::from_buffer(c, ggml_backend_cpu_buffer_type(), small.get()));
        allocation_probe other;
        t.assert_true(!llama_kv_stream_host::from_buffer(c, &other.type, host->buffer()));
        t.assert_true(!llama_kv_stream_host::from_buffer(c, nullptr, host->buffer()));
    });
    t.test("host_lifetime_is_independent_of_device_binding", [&](testing & t) {
        auto host = llama_kv_stream_host::create(config(), storage_type);
        if (!t.assert_true(host != nullptr)) return;
        std::weak_ptr<llama_kv_stream_host> weak = host;
        struct resources : llama_memory_executable { std::shared_ptr<llama_kv_stream_host> host; };
        struct completion : llama_memory_executor_backend { bool drain() override { return true; } } done;
        auto * arena = ggml_backend_memory_arena_new(ggml_backend_cpu_buffer_type(), 1024*1024);
        GGML_ASSERT(arena && ggml_backend_memory_arena_begin(arena, 0));
        auto base = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(ggml_backend_memory_arena_parent(arena)));
        GGML_ASSERT(ggml_backend_memory_arena_reserve_at(arena, 1, (128 - base % 128) % 128, 851968, 64, 0, nullptr));
        GGML_ASSERT(ggml_backend_memory_arena_commit(arena));
        auto * lease = ggml_backend_memory_arena_acquire(arena, 1);
        llama_kv_stream_binding binding(host->cache_id(), ggml_backend_cpu_buffer_type());
        llama_kv_stream_policy_config policy;
        policy.shape = host->config().shape; policy.capabilities = host->config().capabilities;
        policy.pool_bytes = 851968; policy.layers = 1;
        if (!t.assert_true(binding.bind(lease, policy, [&](const auto &) {
            auto result = std::make_unique<resources>(); result->host = host; return result;
        }))) { ggml_backend_memory_lease_free(lease); ggml_backend_memory_arena_free(arena); return; }
        ggml_backend_memory_lease_free(lease);
        ggml_backend_memory_arena_free(arena);
        auto pin = binding.acquire();
        host.reset();
        t.assert_true(!weak.expired());
        t.assert_true(binding.detach(done).status == llama_memory_executor_status::pending);
        pin.reset();
        t.assert_true(binding.detach(done).status == llama_memory_executor_status::retired);
        t.assert_true(weak.expired());
    });
    t.test("unaligned_plane_sizes_have_aligned_layer_stride", [](testing & t) {
        auto c = config(GGML_TYPE_Q4_1, GGML_TYPE_Q5_1);
        c.shape.head_dim_k = c.shape.head_dim_v = 64;
        c.shape.heads = c.shape.page_tokens = 1;
        c.context_tokens = 3;
        auto host = llama_kv_stream_host::create(c, ggml_backend_cpu_buffer_type());
        if (!t.assert_true(host != nullptr)) return;
        t.assert_equal(size_t(120), host->layout().k_bytes);
        t.assert_equal(size_t(128), host->layout().v_offset);
        t.assert_equal(size_t(144), host->layout().v_bytes);
        t.assert_equal(size_t(384), host->stride());
        llama_kv_stream_host_layer first, second;
        t.assert_true(host->layer(0, first) && host->layer(1, second));
        t.assert_equal(host->stride(), size_t(static_cast<char *>(second.k) - static_cast<char *>(first.k)));
        t.assert_equal(size_t(0), reinterpret_cast<uintptr_t>(second.v) % 128);
    });
    t.test("no_host_adapter_fallback_for_cpu", [](testing & t) {
        ggml_backend_ptr cpu(ggml_backend_cpu_init());
        t.assert_true(llama_kv_stream_host_buffer_type(ggml_backend_get_device(cpu.get())) == nullptr);
        t.assert_true(llama_kv_stream_host_register(nullptr, nullptr) == nullptr);
    });
    if (device) t.test("registered_adapter_preserves_imported_planes", [&](testing & t) {
        auto c = config(); c.layers = 1;
        ggml_backend_buffer_ptr owner(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), 1024*1024 + 4096));
        if (!t.assert_true(owner != nullptr)) return;
        ggml_backend_buffer_clear(owner.get(), 0x41);
        auto base = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(owner.get()));
        ggml_backend_buffer_ptr page(ggml_backend_buffer_view(owner.get(), (4096 - base % 4096) % 4096, 1024*1024));
        ggml_backend_buffer_ptr pinned(llama_kv_stream_host_register(device, page.get()));
        if (!t.assert_true(pinned != nullptr)) return;
        auto host = llama_kv_stream_host::from_buffer(c, storage_type, pinned.get());
        if (!t.assert_true(host != nullptr)) return;
        owner.reset(); page.reset(); pinned.reset();
        llama_kv_stream_host_layer layer;
        t.assert_true(host->layer(0, layer));
        t.assert_equal(uint8_t(0x41), *static_cast<uint8_t *>(layer.k));
        t.assert_equal(uint8_t(0x41), *static_cast<uint8_t *>(layer.v));
    });
    if (device) t.test("demand_registered_ranges_survive_async_upload_and_owner_release", [&](testing & t) {
        ggml_backend_ptr backend(ggml_backend_dev_init(device, nullptr));
        if (!t.assert_true(backend != nullptr)) return;
        for (int dim : {256, 512}) for (const auto & pair : std::vector<std::pair<int, int>>{
                {GGML_TYPE_Q8_0, GGML_TYPE_Q4_0}, {GGML_TYPE_Q8_0, GGML_TYPE_Q8_0},
                {GGML_TYPE_Q5_1, GGML_TYPE_Q5_1}, {GGML_TYPE_F16, GGML_TYPE_F16}}) {
            auto c = config(pair.first, pair.second); c.context_tokens = 1025; c.layers = 2;
            c.shape.head_dim_k = c.shape.head_dim_v = dim;
            auto host = llama_kv_stream_host_reservation::create(c, device, 65536);
            if (!t.assert_true(host && host->prepare(513))) return;
            const size_t bytes = 513*host->layout().k_token_bytes;
            std::vector<uint8_t> expected(bytes), observed(bytes);
            for (size_t i = 0; i < bytes; ++i) expected[i] = uint8_t(i*37 + 19);
            std::vector<llama_kv_stream_host_range> ranges;
            if (!t.assert_true(host->ranges(1, ggml_kv_stream_operand::k, 0, bytes, ranges))) return;
            ggml_context_ptr ctx(ggml_init({ggml_tensor_overhead(), nullptr, true}));
            ggml_backend_buffer_ptr output(ggml_backend_buft_alloc_buffer(
                ggml_backend_get_default_buffer_type(backend.get()), bytes));
            if (!t.assert_true(ctx && output)) return;
            auto * tensor = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, int64_t(bytes));
            if (!t.assert_true(ggml_backend_tensor_alloc(output.get(), tensor, ggml_backend_buffer_get_base(output.get())) == GGML_STATUS_SUCCESS)) return;
            size_t offset = 0;
            for (const auto & range : ranges) {
                auto * source = static_cast<char *>(ggml_backend_buffer_get_base(range.buffer.get())) + range.offset;
                std::memcpy(source, expected.data() + offset, range.bytes);
                ggml_backend_tensor_set_async(backend.get(), tensor, source, offset, range.bytes);
                offset += range.bytes;
            }
            ggml_backend_synchronize(backend.get());
            size_t downloaded = 0;
            for (const auto & range : ranges) {
                auto * destination = static_cast<char *>(ggml_backend_buffer_get_base(range.buffer.get())) + range.offset;
                std::memset(destination, 0xcc, range.bytes);
                ggml_backend_tensor_get_async(backend.get(), tensor, destination, downloaded, range.bytes);
                downloaded += range.bytes;
            }
            host.reset();
            ggml_backend_synchronize(backend.get());
            size_t checked = 0;
            for (const auto & range : ranges) {
                const auto * data = static_cast<const char *>(ggml_backend_buffer_get_base(range.buffer.get())) + range.offset;
                t.assert_true(std::memcmp(data, expected.data() + checked, range.bytes) == 0);
                checked += range.bytes;
            }
            ranges.clear();
            ggml_backend_tensor_get(tensor, observed.data(), 0, bytes);
            t.assert_equal(bytes, offset);
            t.assert_equal(bytes, downloaded);
            t.assert_equal(bytes, checked);
            t.assert_true(expected == observed);
        }
    });
    return t.summary();
}
