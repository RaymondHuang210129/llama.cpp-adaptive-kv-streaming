#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cpp.h"
#include "testing.h"

#include <cstring>
#include <random>

static void check(testing & t, ggml_backend_t backend, int width, int rows, int sequences, bool edge=false) {
    const int heads = 6, key_heads = 2, retained = std::min(rows, 8);
    ggml_context_ptr ctx(ggml_init({128 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true}));
    auto make = [&](int a, int b, int c, int d) { return ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, a, b, c, d); };
    auto * q = make(width, key_heads, rows, sequences);
    auto * k = make(width, key_heads, rows, sequences);
    auto * v = make(width, heads, rows, sequences);
    auto * g = make(1, heads, rows, sequences);
    auto * beta = make(1, heads, rows, sequences);
    auto * state = make(width, width, heads, sequences);
    auto * native = ggml_gated_delta_net(ctx.get(), q, k, v, g, beta, state, rows);
    auto * capture = ggml_gated_delta_net_capture(ctx.get(), q, k, v, g, beta, state, 8);
    auto * prefixes = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, sequences);
    auto part = [&](ggml_gdn_capture_part p) { return ggml_gated_delta_net_capture_view(ctx.get(), capture, p); };
    auto * replay = ggml_gated_delta_net_replay(ctx.get(), part(GGML_GDN_CAPTURE_CHECKPOINT), part(GGML_GDN_CAPTURE_DECAY),
            part(GGML_GDN_CAPTURE_KEY), part(GGML_GDN_CAPTURE_CORRECTION), prefixes);
    const int scalar_prefix = retained / 2;
    auto * scalar = ggml_gated_delta_net_replay_prefix(ctx.get(), part(GGML_GDN_CAPTURE_CHECKPOINT), part(GGML_GDN_CAPTURE_DECAY),
            part(GGML_GDN_CAPTURE_KEY), part(GGML_GDN_CAPTURE_CORRECTION), scalar_prefix);
    const size_t image=size_t(width)*width*heads;
    const size_t pitch=size_t(heads+width*key_heads+width*heads)*8;
    auto * cache=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,image,2*sequences);
    auto * log=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,pitch,sequences);
    auto * destinations=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_I32,sequences);
    auto * slots=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_I32,sequences);
    auto * direct=ggml_gated_delta_net_capture_direct(ctx.get(),q,k,v,g,beta,state,cache,log,destinations,slots,8);
    const size_t attention=size_t(width)*heads*rows*sequences;
    t.assert_equal(int64_t(attention+(rows>8 ? image*sequences : 0)),ggml_nelements(direct));
    auto * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, native);
    ggml_build_forward_expand(graph, replay);
    ggml_build_forward_expand(graph, scalar);
    ggml_build_forward_expand(graph, direct);
    if (!t.assert_true(ggml_backend_supports_op(backend, capture)) ||
            !t.assert_true(ggml_backend_supports_op(backend, replay))) return;
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    std::mt19937 random(117 + rows);
    std::uniform_real_distribution<float> uniform(-0.25f, 0.25f);
    std::vector<float> checkpoint;
    for (auto * input : {q, k, v, g, beta, state}) {
        std::vector<float> data(ggml_nelements(input));
        for (auto & f : data) f = input == g ? -1.0f + uniform(random) : uniform(random);
        if (edge) {
            const float values[]={0.0f,-0.0f,1e-40f,-1e-40f,1e-20f,-1e-20f,0.25f,-0.25f};
            const float gates[]={-0.0f,-80.0f,-104.0f,-1000.0f};
            for (size_t i=0;i<data.size();++i) data[i]=input==g ? gates[i%4] : values[i%8];
        }
        if (input == state) checkpoint = data;
        ggml_backend_tensor_set(input, data.data(), 0, ggml_nbytes(input));
    }
    std::vector<int32_t> physical(sequences),working(sequences);
    std::vector<float> initial_cache(2*image*sequences,0),initial_log(pitch*sequences,-123.0f);
    for (int s=0;s<sequences;++s) {
        physical[s]=sequences-1-s;working[s]=sequences+physical[s];
        std::memcpy(initial_cache.data()+physical[s]*image,checkpoint.data()+s*image,image*sizeof(float));
    }
    ggml_backend_tensor_set(cache,initial_cache.data(),0,ggml_nbytes(cache));
    ggml_backend_tensor_set(log,initial_log.data(),0,ggml_nbytes(log));
    ggml_backend_tensor_set(destinations,working.data(),0,ggml_nbytes(destinations));
    ggml_backend_tensor_set(slots,physical.data(),0,ggml_nbytes(slots));
    for (int p = 0; p <= retained; ++p) {
        std::vector<int32_t> lengths(sequences);
        for (int s = 0; s < sequences; ++s) lengths[s] = (p + s) % (retained + 1);
        ggml_backend_tensor_set(prefixes, lengths.data(), 0, ggml_nbytes(prefixes));
        if (!t.assert_true(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS)) return;
        std::vector<float> dense(ggml_nelements(native)), compact(ggml_nelements(capture)), restored(ggml_nelements(replay));
        ggml_backend_tensor_get(native, dense.data(), 0, ggml_nbytes(native));
        ggml_backend_tensor_get(capture, compact.data(), 0, ggml_nbytes(capture));
        ggml_backend_tensor_get(replay, restored.data(), 0, ggml_nbytes(replay));
        t.assert_equal(0, std::memcmp(dense.data(), compact.data(), attention * sizeof(float)));
        t.assert_equal(0, std::memcmp(dense.data() + attention, compact.data() + attention, image * sequences * sizeof(float)));
        for (int s = 0; s < sequences; ++s) {
            auto * expected = lengths[s] == 0 && rows == retained ? checkpoint.data() + image * s :
                dense.data() + attention + image * (sequences * (retained - lengths[s]) + s);
            if (!t.assert_equal(0, std::memcmp(expected, restored.data() + image * s, image * sizeof(float)))) return;
        }
        ggml_backend_tensor_get(scalar, restored.data(), 0, ggml_nbytes(scalar));
        for (int s = 0; s < sequences; ++s) {
            const auto * expected = scalar_prefix == 0 && rows == retained ? checkpoint.data() + image * s :
                dense.data() + attention + image * (sequences * (retained - scalar_prefix) + s);
            if (!t.assert_equal(0, std::memcmp(expected, restored.data() + image * s, image * sizeof(float)))) return;
        }
        std::vector<float> direct_output(ggml_nelements(direct)),actual_cache(ggml_nelements(cache)),actual_log(ggml_nelements(log));
        ggml_backend_tensor_get(direct,direct_output.data(),0,ggml_nbytes(direct));
        ggml_backend_tensor_get(cache,actual_cache.data(),0,ggml_nbytes(cache));
        ggml_backend_tensor_get(log,actual_log.data(),0,ggml_nbytes(log));
        t.assert_equal(0,std::memcmp(dense.data(),direct_output.data(),attention*sizeof(float)));
        t.assert_equal(0,std::memcmp(initial_cache.data(),actual_cache.data(),image*sequences*sizeof(float)));
        for (int s=0;s<sequences;++s) {
            const auto slot=physical[s];
            t.assert_equal(0,std::memcmp(dense.data()+attention+s*image,actual_cache.data()+working[s]*image,image*sizeof(float)));
            if (rows>8) t.assert_equal(0,std::memcmp(compact.data()+attention+image*(sequences+s),
                        direct_output.data()+attention+s*image,image*sizeof(float)));
            const auto * record=compact.data()+attention+2*image*sequences;
            const size_t widths[]={size_t(heads),size_t(width)*key_heads,size_t(width)*heads};
            size_t source=0,destination=0;
            for (auto field : widths) {
                t.assert_equal(0,std::memcmp(record+source+s*field*retained,actual_log.data()+slot*pitch+destination,
                            field*retained*sizeof(float)));
                for (size_t i=field*retained;i<field*8;++i)
                    if (!t.assert_equal(-123.0f,actual_log[slot*pitch+destination+i])) return;
                source+=field*retained*sequences;destination+=field*8;
            }
        }
    }
}

int main(int argc, char ** argv) {
    testing t;
    ggml_backend_load_all();
    if (argc > 2) ggml_backend_load(argv[2]);
    ggml_backend_t backend = argc > 1 ? ggml_backend_init_by_name(argv[1], nullptr) : ggml_backend_cpu_init();
    if (!backend) return 77;
    t.test("FP32_every_prefix_anchor_abort_grouped_keys_and_unequal_acceptance", [&](testing & t) {
        for (int width : {64, 128}) for (int slots = 1; slots <= 4; ++slots)
            for (int rows : {1, 2, 3, 4, 5, 6, 7, 8, 9, 32}) check(t, backend, width, rows, slots);
    });
    t.test("FP32_signed_zero_subnormal_inputs_and_underflowed_decay", [&](testing & t) {
        for (int width : {64,128}) for (int slots : {1,4}) for (int rows : {1,4,8})
            check(t,backend,width,rows,slots,true);
    });
    ggml_backend_free(backend);
    return t.summary();
}
