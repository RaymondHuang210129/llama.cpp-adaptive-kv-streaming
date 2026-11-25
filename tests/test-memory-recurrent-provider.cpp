#include "../src/llama-memory-recurrent-provider.h"
#include "testing.h"

#include <limits>

using representation = llama_recurrent_representation;

static llama_recurrent_provider_request qwen(uint32_t depth, uint32_t selected) {
    llama_recurrent_provider_request r;
    r.depth = depth;
    r.selected = selected;
    r.configured = 256;
    r.alignment = 256;
    r.layers.assign(48, {128, 48, 16, 10240, 3});
    return r;
}

int main() {
    testing t;
    t.test("selected_not_configured_capacity", [](testing & t) {
        auto r = qwen(3, 4);
        llama_recurrent_provider_requirements out;
        t.assert_true(llama_recurrent_provider_describe(r, representation::dense_device, {}, out));
        t.assert_equal(size_t(2394) * 1024 * 1024, out.device_persistent);
        r.configured = 4;
        llama_recurrent_provider_requirements same;
        t.assert_true(llama_recurrent_provider_describe(r, representation::dense_device, {}, same));
        t.assert_equal(out.device_persistent, same.device_persistent);
    });
    t.test("dense_host_staging_and_overlap_are_charged", [](testing & t) {
        auto r = qwen(7, 2);
        r.graph_workspace = 123456;
        llama_recurrent_provider_requirements out;
        t.assert_true(llama_recurrent_provider_describe(r, representation::dense_host, {}, out));
        t.assert_equal(size_t(299.25 * 1024 * 1024), out.device_persistent);
        t.assert_equal(size_t(2094.75 * 1024 * 1024), out.host_persistent);
        t.assert_equal(size_t(87.28125 * 1024 * 1024), out.device_staging);
        t.assert_equal(out.device_persistent + out.device_staging + r.graph_workspace, out.device_peak);
        r.device_budget = out.device_peak - 1;
        t.assert_true(!llama_recurrent_provider_describe(r, representation::dense_host, {}, out));
    });
    t.test("compact_requires_verified_operator_capabilities", [](testing & t) {
        auto r = qwen(7, 1);
        llama_recurrent_provider_requirements out;
        t.assert_true(!llama_recurrent_provider_describe(r, representation::gdn_fp32, {}, out));
        t.assert_true(llama_recurrent_provider_describe(r, representation::gdn_fp32, {true, true, true}, out));
        t.assert_equal(size_t(320.6953125 * 1024 * 1024), out.device_persistent);
        r.layers[0].key_heads = 17;
        t.assert_true(!llama_recurrent_provider_describe(r, representation::gdn_fp32, {true, true, true}, out));
    });
    t.test("invalid_geometry_overflow_and_failed_output_are_atomic", [](testing & t) {
        for (int bad = 0; bad < 7; ++bad) {
            auto r = qwen(3, 1);
            if (bad == 0) r.selected = 0;
            if (bad == 1) r.selected = 257;
            if (bad == 2) r.depth = 8;
            if (bad == 3) r.alignment = 3;
            if (bad == 4) r.layers[0].width = std::numeric_limits<size_t>::max();
            if (bad == 5) r.graph_workspace = std::numeric_limits<size_t>::max();
            if (bad == 6) r.layers[0].history = 0;
            llama_recurrent_provider_requirements out;
            out.device_peak = 42;
            t.assert_true(!llama_recurrent_provider_describe(r, representation::dense_device, {}, out));
            t.assert_equal(size_t(42), out.device_peak);
        }
    });
    t.test("empty_and_target_only_do_not_acquire_compact_resources", [](testing & t) {
        auto r = qwen(0, 1);
        llama_recurrent_provider_requirements out;
        t.assert_true(!llama_recurrent_provider_describe(r, representation::gdn_fp32, {true, true, true}, out));
        r.layers.clear();
        t.assert_true(llama_recurrent_provider_describe(r, representation::dense_device, {}, out));
        t.assert_equal(size_t(0), out.device_peak);
    });
    return t.summary();
}
