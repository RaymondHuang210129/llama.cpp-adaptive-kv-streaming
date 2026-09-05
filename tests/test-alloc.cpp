#include "ggml-alloc.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-impl.h"
#include "ggml.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <memory>
#include <vector>

//
// dummy backend with configurable max_buffer_size, tracks allocations

uint8_t * const alloc_base = (uint8_t *) 16;

struct dummy_backend_context {
    size_t max_buffer_size = 64;
    size_t alignment       = 8;
    size_t reset_count     = 0;

    ggml_backend_buffer_i              buffer_interface;
    std::vector<ggml_backend_buffer_t> buffers;

    size_t allocated_total() const {
        size_t n = 0;
        for (ggml_backend_buffer_t buf : buffers) {
            n += ggml_backend_buffer_get_size(buf);
        }
        return n;
    }
};

// ggml_backend_buffer_type interface

static const char * dummy_backend_buffer_type_get_name(ggml_backend_buffer_type_t) {
    return "dummy_buffer_type";
}

static ggml_backend_buffer_t dummy_backend_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    dummy_backend_context * ctx    = (dummy_backend_context *) buft->context;
    ggml_backend_buffer_t & buffer = ctx->buffers.emplace_back();
    buffer                         = ggml_backend_buffer_init(buft, ctx->buffer_interface, ctx, size);
    return buffer;
}

static size_t dummy_backend_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    dummy_backend_context * ctx = (dummy_backend_context *) buft->context;
    return ctx->alignment;
}

static size_t dummy_backend_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    dummy_backend_context * ctx = (dummy_backend_context *) buft->context;
    return ctx->max_buffer_size;
}

static bool dummy_backend_buffer_type_is_host(ggml_backend_buffer_type_t) {
    return true;
}

// ggml_backend_buffer interface

static void dummy_backend_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    dummy_backend_context * ctx = (dummy_backend_context *) buffer->context;

    auto i = std::find(ctx->buffers.begin(), ctx->buffers.end(), buffer);
    GGML_ASSERT(i != ctx->buffers.end());
    ctx->buffers.erase(i);
}

static void * dummy_backend_buffer_get_base(ggml_backend_buffer_t) {
    return alloc_base;
}

static ggml_status dummy_backend_buffer_init_tensor(ggml_backend_buffer_t, ggml_tensor *) {
    return GGML_STATUS_SUCCESS;
}

static void dummy_backend_buffer_memset_tensor(ggml_backend_buffer_t, ggml_tensor *, uint8_t, size_t, size_t) {}

static void dummy_backend_buffer_set_tensor(ggml_backend_buffer_t, ggml_tensor *, const void *, size_t, size_t) {}

static void dummy_backend_buffer_get_tensor(ggml_backend_buffer_t, const ggml_tensor *, void *, size_t, size_t) {}

static void dummy_backend_buffer_clear(ggml_backend_buffer_t, uint8_t) {}

// Count reset calls so tests can detect changes to the parent buffer's metadata.
static void dummy_backend_buffer_reset(ggml_backend_buffer_t buffer) {
    dummy_backend_context * ctx = (dummy_backend_context *) buffer->context;
    ctx->reset_count++;
}

// dummy_backend (not really a full backend, just provides what gallocr needs)

struct dummy_backend {
    std::unique_ptr<dummy_backend_context> context;
    ggml_backend_buffer_type               buffer_type;
};

// Create a test buffer type with configurable alignment, chunk size, and optional reset tracking.
static dummy_backend dummy_backend_init(size_t max_buffer_size, size_t alignment = 8, bool needs_reset = false) {
    dummy_backend b{};
    b.context                  = std::make_unique<dummy_backend_context>();
    b.context->alignment       = alignment;
    b.context->max_buffer_size = max_buffer_size;

    b.context->buffer_interface.free_buffer   = dummy_backend_buffer_free_buffer;
    b.context->buffer_interface.get_base      = dummy_backend_buffer_get_base;
    b.context->buffer_interface.init_tensor   = dummy_backend_buffer_init_tensor;
    b.context->buffer_interface.memset_tensor = dummy_backend_buffer_memset_tensor;
    b.context->buffer_interface.set_tensor    = dummy_backend_buffer_set_tensor;
    b.context->buffer_interface.get_tensor    = dummy_backend_buffer_get_tensor;
    b.context->buffer_interface.clear         = dummy_backend_buffer_clear;
    b.context->buffer_interface.reset         = needs_reset ? dummy_backend_buffer_reset : nullptr;

    b.buffer_type.context             = b.context.get();
    b.buffer_type.iface.get_name      = dummy_backend_buffer_type_get_name;
    b.buffer_type.iface.alloc_buffer  = dummy_backend_buffer_type_alloc_buffer;
    b.buffer_type.iface.get_alignment = dummy_backend_buffer_type_get_alignment;
    b.buffer_type.iface.get_max_size  = dummy_backend_buffer_type_get_max_size;
    b.buffer_type.iface.is_host       = dummy_backend_buffer_type_is_host;
    return b;
}

//
// test utilities

struct test_context_with_graph {
    ggml_context *   ctx;
    ggml_cgraph *    graph;
    ggml_context_ptr ctx_ptr;
};

static test_context_with_graph make_context() {
    ggml_init_params params{};
    params.mem_size = 48 * ggml_tensor_overhead() + ggml_graph_overhead();
    params.no_alloc = true;

    ggml_context *   ctx     = ggml_init(params);
    ggml_context_ptr ctx_ptr = ggml_context_ptr(ctx);
    ggml_cgraph *    graph   = ggml_new_graph(ctx);
    return { ctx, graph, std::move(ctx_ptr) };
}

static ggml_tensor * make_input_1d(ggml_context * ctx, int64_t n_elements) {
    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
    ggml_set_input(t);
    return t;
}

static ggml_tensor * make_input_with_size(ggml_context * ctx, size_t size_bytes) {
    GGML_ASSERT(size_bytes % 4 == 0);
    return make_input_1d(ctx, size_bytes / 4);
}

static void assign_names(ggml_context * ctx, const char * prefix = "x") {
    int i = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        ggml_format_name(t, "%s%d", prefix, i++);
    }
}

static int get_leaf_id(ggml_cgraph * graph, const char * tensor_name) {
    for (int i = 0; i < graph->n_leafs; ++i) {
        if (strncmp(graph->leafs[i]->name, tensor_name, GGML_MAX_NAME) == 0) {
            return i;
        }
    }
    fprintf(stderr, "leaf not found: %s\n", tensor_name);
    return -1;
}

static int get_node_id(ggml_cgraph * graph, const char * tensor_name) {
    for (int i = 0; i < graph->n_nodes; ++i) {
        if (strncmp(graph->nodes[i]->name, tensor_name, GGML_MAX_NAME) == 0) {
            return i;
        }
    }
    fprintf(stderr, "node not found: %s", tensor_name);
    return -1;
}

static ggml_gallocr_ptr allocate_graph(ggml_cgraph * graph, ggml_tensor * out, ggml_backend_buffer_type_t buft) {
    ggml_set_output(out);
    ggml_build_forward_expand(graph, out);

    ggml_gallocr_ptr galloc = ggml_gallocr_ptr(ggml_gallocr_new(buft));
    bool             result = ggml_gallocr_alloc_graph(galloc.get(), graph);
    GGML_ASSERT(result);
    return galloc;
}

//
// correctness checks for result allocations

static void check_all_allocated(ggml_cgraph * graph) {
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        ggml_tensor * t = ggml_graph_node(graph, i);
        GGML_ASSERT(t->buffer != nullptr);
        GGML_ASSERT(t->data != nullptr);
    }
}

static void check_max_size(ggml_context * ctx) {
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        auto   buft     = ggml_backend_buffer_get_type(t->buffer);
        size_t max_size = ggml_backend_buft_get_max_size(buft);
        size_t offset   = (char *) t->data - (char *) ggml_backend_buffer_get_base(t->buffer);
        GGML_ASSERT(t->data >= ggml_backend_buffer_get_base(t->buffer));
        GGML_ASSERT((size_t) offset + ggml_nbytes(t) <= max_size);
    }
}

static bool can_reuse_memory(ggml_cgraph * graph, int current_i, ggml_tensor * current, ggml_tensor * other) {
    if (other->flags & GGML_TENSOR_FLAG_OUTPUT) {
        return false;
    }
    // Check if `other` is still "alive", ie. an input to any node after the `current` op
    for (int i = current_i; i < ggml_graph_n_nodes(graph); ++i) {
        ggml_tensor * t = ggml_graph_node(graph, i);
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            if (t == current && ggml_op_can_inplace(t->op)) {
                continue;
            }
            if (t->src[s] == other) {
                return false;
            }
            if (t->src[s] && t->src[s]->view_src == other) {
                return false;
            }
        }
    }
    return true;
}

static bool memory_overlap(ggml_tensor * a, ggml_tensor * b) {
    if (a->buffer != b->buffer) {
        return false;
    }
    int64_t a0 = (int64_t) a->data;
    int64_t a1 = a0 + ggml_nbytes(a);
    int64_t b0 = (int64_t) b->data;
    int64_t b1 = b0 + ggml_nbytes(b);
    return a1 > b0 && b1 > a0;
}

static ggml_tensor * get_view_source(ggml_tensor * t) {
    while (t->view_src) {
        t = t->view_src;
    }
    return t;
}

static void check_no_overlap(ggml_cgraph * graph) {
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        for (int j = 0; j < i; ++j) {
            ggml_tensor * t = ggml_graph_node(graph, i);
            ggml_tensor * o = ggml_graph_node(graph, j);
            GGML_ASSERT(t != o);

            if (get_view_source(t) == get_view_source(o)) {
                continue;
            }
            if (memory_overlap(t, o)) {
                GGML_ASSERT(can_reuse_memory(graph, i, t, o));
            }
        }
    }
}

// Check that every graph tensor uses the parent buffer and fits in the borrowed range, including padding.
static void check_graph_in_buffer_range(
        ggml_cgraph * graph, ggml_backend_buffer_t buffer, size_t offset, size_t size) {
    const uintptr_t begin = (uintptr_t) ggml_backend_buffer_get_base(buffer) + offset;
    const uintptr_t end   = begin + size;

    auto check = [&](ggml_tensor * tensor) {
        GGML_ASSERT(tensor->buffer == buffer);
        GGML_ASSERT((uintptr_t) tensor->data >= begin);
        GGML_ASSERT((uintptr_t) tensor->data + ggml_backend_buffer_get_alloc_size(buffer, tensor) <= end);
    };

    for (int i = 0; i < graph->n_leafs; ++i) {
        check(graph->leafs[i]);
    }
    for (int i = 0; i < graph->n_nodes; ++i) {
        check(graph->nodes[i]);
    }
}

//
// test cases

// Scenario where the first backend buffer is completely exhausted and there are further
// tensors which require a second buffer
static void test_max_size_too_many_tensors() {
    dummy_backend backend      = dummy_backend_init(16);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[7];
    x[0] = make_input_with_size(ctx, 8);
    x[1] = make_input_with_size(ctx, 8);
    x[2] = make_input_with_size(ctx, 8);
    x[3] = ggml_mul(ctx, x[0], x[1]);
    x[4] = ggml_add(ctx, x[1], x[2]);
    x[5] = ggml_add(ctx, x[3], x[0]);
    x[6] = ggml_add(ctx, x[4], x[5]);
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[6], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend.context->allocated_total() <= 16 + 16);
}

// Scenario where there is some space left in the first buffer, but not enough to accommodate
// a larger tensor, so a second buffer is required
static void test_max_size_tensor_too_large() {
    dummy_backend backend      = dummy_backend_init(32);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[3];
    x[0] = make_input_with_size(ctx, 16);    // chunk 0, [0 , 16)
    x[1] = make_input_with_size(ctx, 8);     // chunk 0, [16, 24)
    x[2] = ggml_concat(ctx, x[0], x[1], 0);  // chunk 1, [0 , 24)
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[2], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend.context->allocated_total() <= 32 + 24);
}

// Scenario where a single tensor exceeds the max buffer size - in this case the allocator
// should try to create a bigger buffer anyway, and wait for the backend to throw an error.
// Backends may report an artificially lower max size in some cases for compatibility reasons.
static void test_tensor_larger_than_max_size() {
    dummy_backend backend      = dummy_backend_init(16);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[2];
    x[0] = make_input_with_size(ctx, 24);
    x[1] = ggml_scale(ctx, x[0], 2.0f);
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[1], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    GGML_ASSERT(backend.context->allocated_total() == 24);
}

// This test assumes a max of 16 buffer chunks, and tries to allocate tensors that would
// require more. Expectation is that the last buffer should grow to fit everything,
// leaving it to the backend to error out if it can't allocate that much.
static void test_not_enough_chunks() {
    const int max_chunks = 16;
    const int max_size   = 8;

    dummy_backend backend      = dummy_backend_init(max_size);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[max_chunks + 1];
    for (int i = 0; i < max_chunks + 1; ++i) {
        x[i] = make_input_with_size(ctx, max_size);
    }
    ggml_tensor * acc = x[0];
    for (int i = 0; i < max_chunks; ++i) {
        acc = ggml_add(ctx, acc, x[i + 1]);
    }
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, acc, &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    GGML_ASSERT(backend.context->allocated_total() > max_chunks * max_size);
}

// Fill up leftover unallocated space of a chunk after allocating a large tensor that
// requires a new chunk.
static void test_fill_leftover_space() {
    dummy_backend backend      = dummy_backend_init(16);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[4];
    x[0] = make_input_with_size(ctx, 8);
    x[1] = ggml_pad(ctx, x[0], 2, 0, 0, 0);
    x[3] = ggml_mean(ctx, x[1]);
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[3], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend.context->allocated_total() <= 12 + 16);
}

// Check that views don't require any extra memory
static void test_view_inplace() {
    dummy_backend backend      = dummy_backend_init(32);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[6];
    x[0] = make_input_1d(ctx, 4);                // chunk 0, [0, 16)
    x[1] = ggml_reshape_2d(ctx, x[0], 2, 2);     // view of x0
    x[2] = ggml_permute(ctx, x[1], 1, 0, 2, 3);  // view of x0
    x[3] = ggml_view_1d(ctx, x[2], 2, 4);        // view of x0
    x[4] = make_input_1d(ctx, 2);                // chunk 0, [16, 24)
    x[5] = ggml_add(ctx, x[3], x[4]);            // reuse (inplace add)
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[5], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend.context->allocated_total() <= 24);
}

static void test_reuse_and_free() {
    dummy_backend backend      = dummy_backend_init(40);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[9];
    x[0] = make_input_with_size(ctx, 24);
    x[1] = make_input_with_size(ctx, 8);
    x[2] = make_input_with_size(ctx, 8);
    x[3] = ggml_add(ctx, x[1], x[2]);        // reuse, free x2
    x[4] = ggml_pad(ctx, x[0], 2, 0, 0, 0);  // alloc new buffer, free x0
    x[5] = ggml_scale(ctx, x[4], 2.0f);      // alloc from free block
    x[6] = ggml_add(ctx, x[4], x[5]);        // reuse, free x5
    x[7] = ggml_view_1d(ctx, x[6], 2, 8);    // view
    x[8] = ggml_add(ctx, x[3], x[7]);        // reuse
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[8], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend.context->allocated_total() <= 40 + 32 + 32);
}

static void test_merge_free_block(size_t max_buffer_size) {
    dummy_backend backend      = dummy_backend_init(max_buffer_size);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[9];
    x[0] = make_input_with_size(ctx, 16);
    x[1] = make_input_with_size(ctx, 16);
    x[2] = make_input_with_size(ctx, 16);
    x[3] = ggml_mean(ctx, x[0]);
    x[4] = ggml_mean(ctx, x[1]);
    x[5] = ggml_pad(ctx, x[2], 2, 0, 0, 0);
    x[6] = ggml_add(ctx, x[3], x[4]);
    x[7] = ggml_pad(ctx, x[6], 5, 0, 0, 0);
    x[8] = ggml_add(ctx, x[5], x[7]);
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[8], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend.context->allocated_total() <= 32 + 32 + 24);
}

// Check that previously allocated but freed memory is preferred over allocating
// additional memory, even if the remaining space in a chunk would match tensor size better
static void test_prefer_already_allocated_memory() {
    dummy_backend backend      = dummy_backend_init(32, /*align*/ 4);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[3];
    x[0] = make_input_with_size(ctx, 24);  // [24b][8b unused]
    x[1] = ggml_mean(ctx, x[0]);           // [24b free][4b][4b unused]
    x[2] = ggml_mean(ctx, x[1]);           // should be allocated in the 24b block
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[2], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    GGML_ASSERT(backend.context->allocated_total() <= 28);
}

// test for allocating on multiple devices with some tensors in the graph
// allocated externally (not by gallocr).
static void test_multiple_buffer_types() {
    dummy_backend backend_a = dummy_backend_init(32);
    dummy_backend backend_b = dummy_backend_init(SIZE_MAX);

    auto [ctx_a, _a, ctx_a_ptr] = make_context();
    auto [ctx_b, _b, ctx_b_ptr] = make_context();
    auto [ctx, graph, ctx_ptr]  = make_context();

    ggml_tensor * a[2];
    a[0] = make_input_with_size(ctx_a, 16);
    a[1] = make_input_with_size(ctx_a, 16);
    assign_names(ctx_a, "a");

    ggml_tensor * b[2];
    b[0] = make_input_with_size(ctx_b, 24);
    b[1] = make_input_with_size(ctx_b, 4);
    assign_names(ctx_b, "b");

    ggml_tensor * x[9];
    x[0] = make_input_with_size(ctx, 16);
    x[1] = ggml_mul(ctx, x[0], a[0]);
    x[2] = ggml_pad(ctx, x[1], 2, 0, 0, 0);
    x[3] = ggml_mul(ctx, x[2], b[0]);
    x[4] = ggml_mean(ctx, x[3]);
    x[5] = ggml_add(ctx, x[4], b[1]);
    x[6] = ggml_pad(ctx, x[5], 3, 0, 0, 0);
    x[7] = ggml_add(ctx, x[6], a[1]);
    x[8] = ggml_scale(ctx, x[7], 2.0f);
    assign_names(ctx, "x");

    ggml_backend_buffer_ptr    buf_a(ggml_backend_alloc_ctx_tensors_from_buft(ctx_a, &backend_a.buffer_type));
    ggml_backend_buffer_ptr    buf_b(ggml_backend_alloc_ctx_tensors_from_buft(ctx_b, &backend_b.buffer_type));
    ggml_backend_buffer_type_t bufts[2] = { &backend_a.buffer_type, &backend_b.buffer_type };

    // assign buffer types manually to avoid extra complexity from backend scheduler
    ggml_set_output(x[8]);
    ggml_build_forward_expand(graph, x[8]);

    GGML_ASSERT(graph->n_leafs == 5);
    int leaf_buffer_ids[5];
    leaf_buffer_ids[get_leaf_id(graph, "a0")] = 0;
    leaf_buffer_ids[get_leaf_id(graph, "a1")] = 0;
    leaf_buffer_ids[get_leaf_id(graph, "b0")] = 1;
    leaf_buffer_ids[get_leaf_id(graph, "b1")] = 1;
    leaf_buffer_ids[get_leaf_id(graph, "x0")] = 0;

    GGML_ASSERT(graph->n_nodes == 8);
    int node_buffer_ids[8];
    node_buffer_ids[get_node_id(graph, "x1")] = 0;
    node_buffer_ids[get_node_id(graph, "x2")] = 0;
    node_buffer_ids[get_node_id(graph, "x3")] = 1;
    node_buffer_ids[get_node_id(graph, "x4")] = 1;
    node_buffer_ids[get_node_id(graph, "x5")] = 1;
    node_buffer_ids[get_node_id(graph, "x6")] = 1;
    node_buffer_ids[get_node_id(graph, "x7")] = 0;
    node_buffer_ids[get_node_id(graph, "x8")] = 0;

    ggml_gallocr_ptr galloc(ggml_gallocr_new_n(bufts, 2));
    ggml_gallocr_reserve_n(galloc.get(), graph, node_buffer_ids, leaf_buffer_ids);
    ggml_gallocr_alloc_graph(galloc.get(), graph);

    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend_a.context->allocated_total() <= 32 + 32 + 24);
    GGML_ASSERT(backend_b.context->allocated_total() <= 32 + 24);
}

static void test_buffer_size_zero() {
    dummy_backend backend_a    = dummy_backend_init(SIZE_MAX);
    dummy_backend backend_b    = dummy_backend_init(SIZE_MAX);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[2];
    x[0] = make_input_with_size(ctx, 16);
    x[1] = ggml_scale(ctx, x[0], 2.0f);

    ggml_set_output(x[1]);
    ggml_build_forward_expand(graph, x[1]);

    int leaf_buffer_ids[1] = { 0 };
    int node_buffer_ids[1] = { 0 };

    ggml_backend_buffer_type_t bufts[2] = { &backend_a.buffer_type, &backend_b.buffer_type };
    ggml_gallocr_ptr           galloc   = ggml_gallocr_ptr(ggml_gallocr_new_n(bufts, 2));
    bool                       res1     = ggml_gallocr_reserve_n(galloc.get(), graph, node_buffer_ids, leaf_buffer_ids);
    bool                       res2     = ggml_gallocr_alloc_graph(galloc.get(), graph);
    GGML_ASSERT(res1 && res2);

    check_all_allocated(graph);
    GGML_ASSERT(backend_a.context->allocated_total() == 16);
    GGML_ASSERT(backend_b.context->allocated_total() == 0);
}

// Test re-using gallocr for a different graph. The new graph has the same
// total size, but one of the chunks is larger, so reallocation is required.
static void test_reallocation() {
    dummy_backend    backend = dummy_backend_init(32, /*align*/ 4);
    ggml_gallocr_ptr galloc;
    {
        auto [ctx, graph, ctx_ptr] = make_context();
        ggml_tensor * x[4];
        x[0] = make_input_with_size(ctx, 24);
        x[1] = make_input_with_size(ctx, 16);
        x[2] = ggml_view_1d(ctx, x[0], 4, 0);
        x[3] = ggml_add(ctx, x[2], x[1]);
        assign_names(ctx);

        galloc = allocate_graph(graph, x[3], &backend.buffer_type);
        check_all_allocated(graph);
        GGML_ASSERT(backend.context->allocated_total() == 40);
    }
    {
        auto [ctx, graph, ctx_ptr] = make_context();
        ggml_tensor * x[3];
        x[0] = make_input_with_size(ctx, 20);
        x[1] = make_input_with_size(ctx, 20);
        x[2] = ggml_add(ctx, x[0], x[1]);
        assign_names(ctx);
        ggml_set_output(x[2]);
        ggml_build_forward_expand(graph, x[2]);

        bool result = ggml_gallocr_alloc_graph(galloc.get(), graph);
        GGML_ASSERT(result);
        check_all_allocated(graph);
        GGML_ASSERT(backend.context->allocated_total() == 40);
    }
}

// Verify offset placement, capacity reporting, and parent lifetime after the graph allocator is freed.
static void test_borrowed_buffer_range() {
    dummy_backend backend = dummy_backend_init(SIZE_MAX);
    ggml_backend_buffer_ptr workspace(ggml_backend_buft_alloc_buffer(&backend.buffer_type, 96));
    GGML_ASSERT(workspace);

    auto [ctx, graph, ctx_ptr] = make_context();
    ggml_tensor * x[3];
    x[0] = make_input_with_size(ctx, 16);
    x[1] = make_input_with_size(ctx, 16);
    x[2] = ggml_add(ctx, x[0], x[1]);
    assign_names(ctx);
    ggml_set_output(x[2]);
    ggml_build_forward_expand(graph, x[2]);

    ggml_gallocr_ptr galloc(ggml_gallocr_new(&backend.buffer_type));
    GGML_ASSERT(ggml_gallocr_set_buffer_range(galloc.get(), 0, workspace.get(), 24, 48));
    GGML_ASSERT(ggml_gallocr_reserve(galloc.get(), graph));
    GGML_ASSERT(ggml_gallocr_alloc_graph(galloc.get(), graph));

    check_all_allocated(graph);
    check_no_overlap(graph);
    check_graph_in_buffer_range(graph, workspace.get(), 24, 48);
    GGML_ASSERT(ggml_gallocr_get_buffer_size(galloc.get(), 0) == 48);
    GGML_ASSERT(backend.context->allocated_total() == 96);

    galloc.reset();
    GGML_ASSERT(backend.context->allocated_total() == 96);
}

// Verify that insufficient workspace fails without allocating extra buffers or assigning tensor addresses.
static void test_borrowed_buffer_range_too_small() {
    dummy_backend backend = dummy_backend_init(SIZE_MAX);
    ggml_backend_buffer_ptr workspace(ggml_backend_buft_alloc_buffer(&backend.buffer_type, 64));
    GGML_ASSERT(workspace);

    auto [ctx, graph, ctx_ptr] = make_context();
    ggml_tensor * x[3];
    x[0] = make_input_with_size(ctx, 16);
    x[1] = make_input_with_size(ctx, 16);
    x[2] = ggml_add(ctx, x[0], x[1]);
    assign_names(ctx);
    ggml_set_output(x[2]);
    ggml_build_forward_expand(graph, x[2]);

    ggml_gallocr_ptr galloc(ggml_gallocr_new(&backend.buffer_type));
    GGML_ASSERT(ggml_gallocr_set_buffer_range(galloc.get(), 0, workspace.get(), 8, 24));
    GGML_ASSERT(!ggml_gallocr_reserve(galloc.get(), graph));
    GGML_ASSERT(backend.context->allocated_total() == 64);

    for (int i = 0; i < graph->n_leafs; ++i) {
        GGML_ASSERT(graph->leafs[i]->data == nullptr);
    }
    for (int i = 0; i < graph->n_nodes; ++i) {
        GGML_ASSERT(graph->nodes[i]->data == nullptr);
    }
}

// Verify rejection of mismatched types, invalid ranges, repeated attachment, and stateful-reset buffers.
static void test_borrowed_buffer_range_validation() {
    dummy_backend backend       = dummy_backend_init(SIZE_MAX);
    dummy_backend other_backend = dummy_backend_init(SIZE_MAX);
    dummy_backend reset_backend = dummy_backend_init(SIZE_MAX, 8, true);

    ggml_backend_buffer_ptr workspace(ggml_backend_buft_alloc_buffer(&backend.buffer_type, 96));
    ggml_backend_buffer_ptr other_workspace(ggml_backend_buft_alloc_buffer(&other_backend.buffer_type, 96));
    ggml_backend_buffer_ptr reset_workspace(ggml_backend_buft_alloc_buffer(&reset_backend.buffer_type, 96));
    GGML_ASSERT(workspace && other_workspace && reset_workspace);

    ggml_gallocr_ptr galloc(ggml_gallocr_new(&backend.buffer_type));
    GGML_ASSERT(!ggml_gallocr_set_buffer_range(galloc.get(), 0, other_workspace.get(), 0, 32));
    GGML_ASSERT(!ggml_gallocr_set_buffer_range(galloc.get(), 0, workspace.get(), 1, 32));
    GGML_ASSERT(!ggml_gallocr_set_buffer_range(galloc.get(), 0, workspace.get(), 80, 32));
    GGML_ASSERT(!ggml_gallocr_set_buffer_range(galloc.get(), 0, workspace.get(), 0, 0));
    GGML_ASSERT(ggml_gallocr_set_buffer_range(galloc.get(), 0, workspace.get(), 16, 32));
    GGML_ASSERT(!ggml_gallocr_set_buffer_range(galloc.get(), 0, workspace.get(), 48, 32));

    ggml_gallocr_ptr reset_galloc(ggml_gallocr_new(&reset_backend.buffer_type));
    GGML_ASSERT(!ggml_gallocr_set_buffer_range(reset_galloc.get(), 0, reset_workspace.get(), 0, 32));
    GGML_ASSERT(reset_backend.context->reset_count == 0);
}

// Verify that aliased allocator slots share one borrowed range and count its capacity only once.
static void test_borrowed_buffer_range_shared_buffer_type() {
    dummy_backend backend = dummy_backend_init(SIZE_MAX);
    ggml_backend_buffer_ptr workspace(ggml_backend_buft_alloc_buffer(&backend.buffer_type, 96));
    GGML_ASSERT(workspace);

    auto [ctx, graph, ctx_ptr] = make_context();
    ggml_tensor * x[3];
    x[0] = make_input_with_size(ctx, 16);
    x[1] = make_input_with_size(ctx, 16);
    x[2] = ggml_add(ctx, x[0], x[1]);
    assign_names(ctx);
    ggml_set_output(x[2]);
    ggml_build_forward_expand(graph, x[2]);

    int leaf_buffer_ids[2];
    leaf_buffer_ids[get_leaf_id(graph, "x0")] = 0;
    leaf_buffer_ids[get_leaf_id(graph, "x1")] = 1;
    int node_buffer_ids[1];
    node_buffer_ids[get_node_id(graph, "x2")] = 1;

    ggml_backend_buffer_type_t bufts[2] = { &backend.buffer_type, &backend.buffer_type };
    ggml_gallocr_ptr galloc(ggml_gallocr_new_n(bufts, 2));
    GGML_ASSERT(ggml_gallocr_set_buffer_range(galloc.get(), 1, workspace.get(), 16, 48));
    GGML_ASSERT(!ggml_gallocr_set_buffer_range(galloc.get(), 0, workspace.get(), 16, 48));
    GGML_ASSERT(ggml_gallocr_reserve_n(galloc.get(), graph, node_buffer_ids, leaf_buffer_ids));
    GGML_ASSERT(ggml_gallocr_alloc_graph(galloc.get(), graph));

    check_graph_in_buffer_range(graph, workspace.get(), 16, 48);
    GGML_ASSERT(ggml_gallocr_get_buffer_size(galloc.get(), 0) == 48);
    GGML_ASSERT(ggml_gallocr_get_buffer_size(galloc.get(), 1) == 0);
    GGML_ASSERT(backend.context->allocated_total() == 96);
}

// Verify that attaching a range after measurement replaces cached zero-based tensor placements.
static void test_borrowed_buffer_range_after_measure() {
    dummy_backend backend = dummy_backend_init(SIZE_MAX);

    auto [ctx, graph, ctx_ptr] = make_context();
    ggml_tensor * x[3];
    x[0] = make_input_with_size(ctx, 16);
    x[1] = make_input_with_size(ctx, 16);
    x[2] = ggml_add(ctx, x[0], x[1]);
    assign_names(ctx);
    ggml_set_output(x[2]);
    ggml_build_forward_expand(graph, x[2]);

    ggml_gallocr_ptr galloc(ggml_gallocr_new(&backend.buffer_type));
    size_t required[1];
    ggml_gallocr_reserve_n_size(galloc.get(), graph, nullptr, nullptr, required);
    GGML_ASSERT(required[0] > 0);

    const size_t offset = 16;
    ggml_backend_buffer_ptr workspace(
        ggml_backend_buft_alloc_buffer(&backend.buffer_type, offset + required[0]));
    GGML_ASSERT(workspace);
    GGML_ASSERT(ggml_gallocr_set_buffer_range(
        galloc.get(), 0, workspace.get(), offset, required[0]));

    GGML_ASSERT(ggml_gallocr_alloc_graph(galloc.get(), graph));
    check_graph_in_buffer_range(graph, workspace.get(), offset, required[0]);
    GGML_ASSERT(backend.context->allocated_total() == offset + required[0]);
}

// Execute an addition graph in borrowed GPU storage and check the result; skip when no GPU is available.
static void test_gpu_borrowed_buffer_range() {
    ggml_backend_load_all();
    ggml_backend_ptr backend(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr));
    if (!backend) {
        return;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend.get());
    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    const size_t offset = 2*alignment;
    const size_t size = 8*alignment;
    ggml_backend_buffer_ptr workspace(ggml_backend_buft_alloc_buffer(buft, offset + size + alignment));
    GGML_ASSERT(workspace);

    auto [ctx, graph, ctx_ptr] = make_context();
    ggml_tensor * x[3];
    x[0] = make_input_with_size(ctx, 16);
    x[1] = make_input_with_size(ctx, 16);
    x[2] = ggml_add(ctx, x[0], x[1]);
    assign_names(ctx);
    ggml_set_output(x[2]);
    ggml_build_forward_expand(graph, x[2]);

    ggml_gallocr_ptr galloc(ggml_gallocr_new(buft));
    GGML_ASSERT(ggml_gallocr_set_buffer_range(
        galloc.get(), 0, workspace.get(), offset, size));
    GGML_ASSERT(ggml_gallocr_reserve(galloc.get(), graph));
    GGML_ASSERT(ggml_gallocr_alloc_graph(galloc.get(), graph));
    check_graph_in_buffer_range(graph, workspace.get(), offset, size);

    const float a[] = { 1.0f, 2.0f, 3.0f, 4.0f };
    const float b[] = { 5.0f, 6.0f, 7.0f, 8.0f };
    float result[4] = {};
    ggml_backend_tensor_set(x[0], a, 0, sizeof(a));
    ggml_backend_tensor_set(x[1], b, 0, sizeof(b));
    GGML_ASSERT(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend.get());
    ggml_backend_tensor_get(x[2], result, 0, sizeof(result));
    for (int i = 0; i < 4; ++i) {
        GGML_ASSERT(result[i] == a[i] + b[i]);
    }
}

// Execute through the CPU scheduler and verify both results and guard bytes outside the borrowed range.
static void test_scheduler_borrowed_buffer_range() {
    ggml_backend_ptr backend(ggml_backend_cpu_init());
    GGML_ASSERT(backend);

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend.get());
    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    const size_t offset = 2*alignment;
    const size_t size = 8*alignment;
    const size_t buffer_size = offset + size + alignment;
    ggml_backend_buffer_ptr workspace(ggml_backend_buft_alloc_buffer(buft, buffer_size));
    GGML_ASSERT(workspace);
    uint8_t * workspace_data = (uint8_t *) ggml_backend_buffer_get_base(workspace.get());
    memset(workspace_data, 0xa5, buffer_size);

    ggml_backend_t backends[] = { backend.get() };
    ggml_backend_buffer_type_t bufts[] = { buft };
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(
        backends, bufts, 1, GGML_DEFAULT_GRAPH_SIZE, false, true));
    GGML_ASSERT(sched);
    GGML_ASSERT(ggml_backend_sched_set_buffer_range(
        sched.get(), backend.get(), workspace.get(), offset, size));

    auto [ctx, graph, ctx_ptr] = make_context();
    ggml_tensor * x[3];
    x[0] = make_input_with_size(ctx, 16);
    x[1] = make_input_with_size(ctx, 16);
    x[2] = ggml_add(ctx, x[0], x[1]);
    assign_names(ctx);
    ggml_set_output(x[2]);
    ggml_build_forward_expand(graph, x[2]);

    GGML_ASSERT(ggml_backend_sched_reserve(sched.get(), graph));
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), graph));
    check_graph_in_buffer_range(graph, workspace.get(), offset, size);

    const float a[] = { 1.0f, 2.0f, 3.0f, 4.0f };
    const float b[] = { 5.0f, 6.0f, 7.0f, 8.0f };
    float result[4] = {};
    ggml_backend_tensor_set(x[0], a, 0, sizeof(a));
    ggml_backend_tensor_set(x[1], b, 0, sizeof(b));
    GGML_ASSERT(ggml_backend_sched_graph_compute(sched.get(), graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(x[2], result, 0, sizeof(result));
    for (int i = 0; i < 4; ++i) {
        GGML_ASSERT(result[i] == a[i] + b[i]);
    }

    for (size_t i = 0; i < offset; ++i) {
        GGML_ASSERT(workspace_data[i] == 0xa5);
    }
    for (size_t i = offset + size; i < buffer_size; ++i) {
        GGML_ASSERT(workspace_data[i] == 0xa5);
    }

    GGML_ASSERT(ggml_backend_sched_get_buffer_size(sched.get(), backend.get()) == size);
}

// Check alignment gaps, exact fits, and recovery after a range runs out of space.
static void test_tallocr_range_capacity() {
    dummy_backend backend = dummy_backend_init(SIZE_MAX);
    ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer(&backend.buffer_type, 128));
    auto [ctx, graph, ctx_ptr] = make_context();
    ggml_tallocr alloc{};
    GGML_ASSERT(ggml_tallocr_new_range(&alloc, buffer.get(), 3, 37));
    GGML_ASSERT(alloc.offset == 8);
    auto * a = make_input_with_size(ctx, 12);
    auto * b = make_input_with_size(ctx, 20);
    auto * c = make_input_with_size(ctx, 16);
    GGML_ASSERT(ggml_tallocr_alloc(&alloc, a) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(a->data == alloc_base + 8);
    GGML_ASSERT(alloc.offset == 24);
    GGML_ASSERT(ggml_tallocr_alloc(&alloc, b) == GGML_STATUS_ALLOC_FAILED);
    GGML_ASSERT(alloc.offset == 24);
    GGML_ASSERT(b->buffer == nullptr && b->data == nullptr);
    GGML_ASSERT(ggml_tallocr_alloc(&alloc, c) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(c->data == alloc_base + 24 && alloc.offset == 40);
    GGML_ASSERT(ggml_tallocr_alloc(&alloc, b) == GGML_STATUS_ALLOC_FAILED);
    GGML_ASSERT(alloc.offset == 40);
    GGML_ASSERT(backend.context->allocated_total() == 128);
}

// Check that invalid constructor arguments preserve an existing allocator.
static void test_tallocr_range_validation() {
    dummy_backend backend = dummy_backend_init(SIZE_MAX);
    ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer(&backend.buffer_type, 64));
    ggml_tallocr alloc = ggml_tallocr_new(buffer.get());
    const auto original = alloc;
    for (const auto & range : std::vector<std::pair<size_t, size_t>>{
            {0, 0}, {65, 1}, {64, 1}, {60, 8}, {SIZE_MAX, 1}, {8, SIZE_MAX}, {1, 1}}) {
        GGML_ASSERT(!ggml_tallocr_new_range(&alloc, buffer.get(), range.first, range.second));
        GGML_ASSERT(alloc.buffer == original.buffer && alloc.base == original.base);
        GGML_ASSERT(alloc.offset == original.offset && alloc.alignment == original.alignment);
        GGML_ASSERT(alloc.limit == original.limit && alloc.bounded == original.bounded);
    }
    GGML_ASSERT(!ggml_tallocr_new_range(nullptr, buffer.get(), 0, 8));
    GGML_ASSERT(!ggml_tallocr_new_range(&alloc, nullptr, 0, 8));
    GGML_ASSERT(ggml_tallocr_new_range(&alloc, buffer.get(), 56, 8));
    ggml_backend_buffer_ptr empty(ggml_backend_buft_alloc_buffer(&backend.buffer_type, 0));
    GGML_ASSERT(!ggml_tallocr_new_range(&alloc, empty.get(), 0, 8));
    backend.context->alignment = 3;
    GGML_ASSERT(!ggml_tallocr_new_range(&alloc, buffer.get(), 0, 8));
    backend.context->alignment = 0;
    GGML_ASSERT(!ggml_tallocr_new_range(&alloc, buffer.get(), 0, 8));
    backend.context->alignment = 8;

    buffer->iface.get_base = [](ggml_backend_buffer_t) -> void * { return (void *) (UINTPTR_MAX - 7); };
    GGML_ASSERT(!ggml_tallocr_new_range(&alloc, buffer.get(), 8, 8));
    GGML_ASSERT(!ggml_tallocr_new_range(&alloc, buffer.get(), 0, 16));
    buffer->iface.get_base = dummy_backend_buffer_get_base;
    GGML_ASSERT(alloc.offset == 56 && alloc.limit == 64);
}

// Simulate a backend that needs eight bytes of padding after every tensor.
static size_t padded_tensor_alloc_size(ggml_backend_buffer_type_t, const ggml_tensor * tensor) {
    return ggml_nbytes(tensor) + 8;
}

// Ensure backend padding is included in the range limit.
static void test_tallocr_range_padding() {
    dummy_backend backend = dummy_backend_init(SIZE_MAX);
    backend.buffer_type.iface.get_alloc_size = padded_tensor_alloc_size;
    ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer(&backend.buffer_type, 128));
    auto [ctx, graph, ctx_ptr] = make_context();
    ggml_tallocr alloc{};
    GGML_ASSERT(ggml_tallocr_new_range(&alloc, buffer.get(), 16, 40));
    auto * a = make_input_with_size(ctx, 16);
    auto * b = make_input_with_size(ctx, 16);
    auto * c = make_input_with_size(ctx, 8);
    GGML_ASSERT(ggml_tallocr_alloc(&alloc, a) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(alloc.offset == 40);
    GGML_ASSERT(ggml_tallocr_alloc(&alloc, b) == GGML_STATUS_ALLOC_FAILED);
    GGML_ASSERT(alloc.offset == 40 && b->data == nullptr);
    GGML_ASSERT(ggml_tallocr_alloc(&alloc, c) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(alloc.offset == 56);
}

// Supply a huge allocation requirement to exercise padding overflow handling.
static size_t oversized_tensor_alloc_size(ggml_backend_buffer_type_t, const ggml_tensor *) {
    return SIZE_MAX;
}

// Reject an overflowing allocation size before changing the tensor or cursor.
static void test_tallocr_range_overflow() {
    dummy_backend backend = dummy_backend_init(SIZE_MAX);
    backend.buffer_type.iface.get_alloc_size = oversized_tensor_alloc_size;
    ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer(&backend.buffer_type, 128));
    auto [ctx, graph, ctx_ptr] = make_context();
    auto * tensor = make_input_with_size(ctx, 16);
    ggml_tallocr alloc{};
    GGML_ASSERT(ggml_tallocr_new_range(&alloc, buffer.get(), 16, 64));
    GGML_ASSERT(ggml_tallocr_alloc(&alloc, tensor) == GGML_STATUS_ALLOC_FAILED);
    GGML_ASSERT(alloc.offset == 16 && tensor->buffer == nullptr && tensor->data == nullptr);
}

// Sweep range boundaries against an independent alignment and capacity calculation.
static void test_tallocr_range_boundaries() {
    for (size_t alignment : {size_t(1), size_t(4), size_t(8), size_t(16), size_t(64)}) {
        dummy_backend backend = dummy_backend_init(SIZE_MAX, alignment);
        ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer(&backend.buffer_type, 256));
        for (size_t start = 0; start < 24; ++start) {
            for (size_t size = 1; size <= 48; ++size) {
                ggml_tallocr alloc{};
                size_t cursor = start;
                while (((uintptr_t) alloc_base + cursor) % alignment != 0) {
                    ++cursor;
                }
                const size_t end = start + size;
                const bool valid = cursor < end;
                GGML_ASSERT(ggml_tallocr_new_range(&alloc, buffer.get(), start, size) == valid);
                if (!valid) {
                    continue;
                }
                auto [ctx, graph, ctx_ptr] = make_context();
                for (size_t bytes : {size_t(20), size_t(12), size_t(8), size_t(4)}) {
                    auto * tensor = make_input_with_size(ctx, bytes);
                    const size_t padded = ((bytes + alignment - 1) / alignment)*alignment;
                    const bool fits = padded <= end - cursor;
                    GGML_ASSERT(ggml_tallocr_alloc(&alloc, tensor) ==
                        (fits ? GGML_STATUS_SUCCESS : GGML_STATUS_ALLOC_FAILED));
                    if (fits) {
                        GGML_ASSERT(tensor->buffer == buffer.get() && tensor->data == alloc_base + cursor);
                        cursor += padded;
                    } else {
                        GGML_ASSERT(tensor->buffer == nullptr && tensor->data == nullptr);
                    }
                    GGML_ASSERT(alloc.offset == cursor && alloc.limit == end);
                }
            }
        }
    }
}

// Verify that aliases into persistent tensors need no allocation and preserve their parent buffer.
static void test_tallocr_range_views() {
    dummy_backend backend = dummy_backend_init(SIZE_MAX);
    ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer(&backend.buffer_type, 64));
    auto [ctx, graph, ctx_ptr] = make_context();
    auto * source = make_input_with_size(ctx, 16);
    ggml_tallocr alloc{};
    GGML_ASSERT(ggml_tallocr_new_range(&alloc, buffer.get(), 16, 16));
    GGML_ASSERT(ggml_tallocr_alloc(&alloc, source) == GGML_STATUS_SUCCESS);
    auto * view = ggml_view_1d(ctx, source, 2, 4);
    if (view->buffer == nullptr) {
        GGML_ASSERT(ggml_backend_view_init(view) == GGML_STATUS_SUCCESS);
    }
    GGML_ASSERT(view->buffer == buffer.get() && view->data == alloc_base + 20);
    GGML_ASSERT(alloc.offset == 32);
}

// Persistent tensor allocation must not reset stateful backend metadata.
static void test_tallocr_range_stateful_buffer() {
    dummy_backend backend = dummy_backend_init(SIZE_MAX, 8, true);
    ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer(&backend.buffer_type, 64));
    auto [ctx, graph, ctx_ptr] = make_context();
    ggml_tallocr alloc{};
    GGML_ASSERT(ggml_tallocr_new_range(&alloc, buffer.get(), 8, 32));
    GGML_ASSERT(ggml_tallocr_alloc(&alloc, make_input_with_size(ctx, 16)) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(backend.context->reset_count == 0);
}

// Keep persistent tensors intact while fresh CPU or GPU graphs reuse another range of the same buffer.
static void test_tallocr_range_shared_workspace(ggml_backend_t target) {
    ggml_backend_ptr backend_cpu(ggml_backend_cpu_init());
    ggml_backend_t backend = target ? target : backend_cpu.get();
    GGML_ASSERT(backend);
    auto * buft = ggml_backend_get_default_buffer_type(backend);
    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    const size_t persistent_offset = alignment;
    const size_t workspace_offset = 4*alignment;
    const size_t workspace_size = 8*alignment;
    const size_t total_size = workspace_offset + workspace_size + alignment;
    ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer(buft, total_size));
    GGML_ASSERT(buffer);
    auto * bytes = static_cast<uint8_t *>(ggml_backend_buffer_get_base(buffer.get()));
    ggml_backend_buffer_clear(buffer.get(), 0xa5);
    auto [ctx, unused_graph, ctx_ptr] = make_context();
    auto * raw = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, total_size);
    GGML_ASSERT(ggml_backend_tensor_alloc(buffer.get(), raw, bytes) == GGML_STATUS_SUCCESS);
    auto * input = make_input_with_size(ctx, 16);
    auto * bias = make_input_with_size(ctx, 16);
    ggml_tallocr alloc{};
    GGML_ASSERT(ggml_tallocr_new_range(&alloc, buffer.get(), persistent_offset, 2*alignment));
    GGML_ASSERT(ggml_tallocr_alloc(&alloc, input) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(ggml_tallocr_alloc(&alloc, bias) == GGML_STATUS_SUCCESS);
    const float a[] = {1, 2, 3, 4};
    const float b[] = {5, 6, 7, 8};
    ggml_backend_tensor_set(input, a, 0, sizeof(a));
    ggml_backend_tensor_set(bias, b, 0, sizeof(b));
    std::vector<uint8_t> snapshot(total_size);
    ggml_backend_tensor_get(raw, snapshot.data(), 0, total_size);
    const auto persistent = snapshot;
    ggml_backend_t backends[] = {backend, backend_cpu.get()};
    const int n_backends = target ? 2 : 1;
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(
        backends, nullptr, n_backends, GGML_DEFAULT_GRAPH_SIZE, false, true));
    GGML_ASSERT(ggml_backend_sched_set_buffer_range(
        sched.get(), backend, buffer.get(), workspace_offset, workspace_size));
    for (int step = 1; step <= 12; ++step) {
        ggml_backend_sched_reset(sched.get());
        auto [gctx, graph, gctx_ptr] = make_context();
        auto * sum = ggml_add(gctx, input, bias);
        auto * output = ggml_scale(gctx, sum, float(step));
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
        GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), graph));
        GGML_ASSERT(ggml_backend_sched_graph_compute(sched.get(), graph) == GGML_STATUS_SUCCESS);
        float result[4];
        ggml_backend_tensor_get(output, result, 0, sizeof(result));
        for (int i = 0; i < 4; ++i) {
            GGML_ASSERT(result[i] == (a[i] + b[i])*step);
        }
        for (int i = 0; i < graph->n_nodes; ++i) {
            auto * tensor = graph->nodes[i];
            const size_t offset = static_cast<uint8_t *>(tensor->data) - bytes;
            GGML_ASSERT(tensor->buffer == buffer.get() && offset >= workspace_offset);
            GGML_ASSERT(offset + ggml_nbytes(tensor) <= workspace_offset + workspace_size);
        }
        ggml_backend_tensor_get(raw, snapshot.data(), 0, total_size);
        GGML_ASSERT(memcmp(snapshot.data(), persistent.data(), workspace_offset) == 0);
        for (size_t i = workspace_offset + workspace_size; i < total_size; ++i) {
            GGML_ASSERT(snapshot[i] == 0xa5);
        }
    }
    sched.reset();
    ggml_backend_tensor_get(raw, snapshot.data(), 0, total_size);
    GGML_ASSERT(memcmp(snapshot.data(), persistent.data(), workspace_offset) == 0);
    GGML_ASSERT(ggml_backend_buffer_get_usage(buffer.get()) == GGML_BACKEND_BUFFER_USAGE_ANY);
}

static void run(const char * name, void (*f)()) {
    printf("%s ", name);
    fflush(stdout);
    f();
    printf("PASSED\n");
}

int main() {
    run("test_max_size_too_many_tensors", test_max_size_too_many_tensors);
    run("test_max_size_tensor_too_large", test_max_size_tensor_too_large);
    run("test_tensor_larger_than_max_size", test_tensor_larger_than_max_size);
    run("test_not_enough_chunks", test_not_enough_chunks);
    run("test_fill_leftover_space", test_fill_leftover_space);
    run("test_view_inplace", test_view_inplace);
    run("test_reuse_and_free", test_reuse_and_free);
    run("test_merge_free_block(32)", []() { test_merge_free_block(32); });
    run("test_merge_free_block(SIZE_MAX)", []() { test_merge_free_block(SIZE_MAX); });
    run("test_prefer_already_allocated_memory", test_prefer_already_allocated_memory);
    run("test_multiple_buffer_types", test_multiple_buffer_types);
    run("test_buffer_size_zero", test_buffer_size_zero);
    run("test_reallocation", test_reallocation);
    run("test_borrowed_buffer_range", test_borrowed_buffer_range);
    run("test_borrowed_buffer_range_too_small", test_borrowed_buffer_range_too_small);
    run("test_borrowed_buffer_range_validation", test_borrowed_buffer_range_validation);
    run("test_borrowed_buffer_range_shared_buffer_type", test_borrowed_buffer_range_shared_buffer_type);
    run("test_borrowed_buffer_range_after_measure", test_borrowed_buffer_range_after_measure);
    run("test_gpu_borrowed_buffer_range", test_gpu_borrowed_buffer_range);
    run("test_scheduler_borrowed_buffer_range", test_scheduler_borrowed_buffer_range);
    run("test_tallocr_range_capacity", test_tallocr_range_capacity);
    run("test_tallocr_range_validation", test_tallocr_range_validation);
    run("test_tallocr_range_padding", test_tallocr_range_padding);
    run("test_tallocr_range_overflow", test_tallocr_range_overflow);
    run("test_tallocr_range_boundaries", test_tallocr_range_boundaries);
    run("test_tallocr_range_views", test_tallocr_range_views);
    run("test_tallocr_range_stateful_buffer", test_tallocr_range_stateful_buffer);
    run("test_tallocr_range_shared_workspace_cpu", [] { test_tallocr_range_shared_workspace(nullptr); });
    ggml_backend_ptr gpu(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr));
    if (gpu) {
        test_tallocr_range_shared_workspace(gpu.get());
        printf("test_tallocr_range_shared_workspace_gpu PASSED\n");
    } else {
        printf("test_tallocr_range_shared_workspace_gpu SKIPPED (no GPU)\n");
    }
    return 0;
}
