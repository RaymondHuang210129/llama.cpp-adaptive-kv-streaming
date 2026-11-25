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
        t.assert_equal(size_t(331.9453125 * 1024 * 1024), out.device_persistent);
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
    t.test("placement_uses_complete_budget_and_overlapped_transfers", [](testing & t) {
        llama_recurrent_placement_request r;
        r.budget=1000;r.reserve=100;r.safe_boundary=true;r.generation=1;
        r.h2d_bytes_per_ms=r.d2h_bytes_per_ms=100;
        r.plans={{representation::dense_device,950,0,0,0,1,0},
                 {representation::dense_host,600,200,100,400,1,0},
                 {representation::gdn_fp32,800,0,350,0,1,0}};
        llama_recurrent_placement_state state;
        llama_recurrent_placement_choice out;
        t.assert_true(llama_recurrent_placement_choose(r,state,out));
        t.assert_true(out.representation==representation::gdn_fp32);
        t.assert_equal(4.5,out.predicted_ms);
        // RAM uses max(H2D + KV, D2H), not their sum: 3 versus 4 ms.
        r.plans[1].d2h_bytes=200;r.generation=2;
        t.assert_true(llama_recurrent_placement_choose(r,state,out));
        t.assert_true(out.representation==representation::gdn_fp32); // cooldown
        r.generation=5;
        t.assert_true(llama_recurrent_placement_choose(r,state,out));
        t.assert_true(out.representation==representation::dense_host);
    });
    t.test("placement_hysteresis_capacity_escape_and_atomic_failure", [](testing & t) {
        llama_recurrent_placement_request r;
        r.budget=1000;r.safe_boundary=true;r.generation=1;
        r.plans={{representation::dense_device,900,0,0,0,1,0},
                 {representation::gdn_fp32,800,0,0,0,1.02,0}};
        llama_recurrent_placement_state state;llama_recurrent_placement_choice out;
        t.assert_true(llama_recurrent_placement_choose(r,state,out));
        r.plans[1].compute_ms=.98;r.generation=20;
        t.assert_true(llama_recurrent_placement_choose(r,state,out));
        t.assert_true(out.representation==representation::dense_device);
        r.reserve=101;r.generation=21;
        t.assert_true(llama_recurrent_placement_choose(r,state,out));
        t.assert_true(out.representation==representation::gdn_fp32);
        const auto prior=state.last_switch_generation;
        r.safe_boundary=false;r.generation=22;
        t.assert_true(!llama_recurrent_placement_choose(r,state,out));
        t.assert_equal(prior,state.last_switch_generation);
        r.safe_boundary=true;r.reserve=1001;
        t.assert_true(!llama_recurrent_placement_choose(r,state,out));
        r.reserve=0;r.plans[0].compute_ms=std::numeric_limits<double>::quiet_NaN();
        t.assert_true(!llama_recurrent_placement_choose(r,state,out));
    });
    t.test("placement_feedback_is_bounded_and_workload_specific", [](testing & t) {
        llama_recurrent_placement_state state;
        for (int i=0;i<20;++i) t.assert_true(llama_recurrent_placement_observe(state,representation::gdn_fp32,42,2,4));
        t.assert_true(llama_recurrent_placement_observe(state,representation::dense_device,42,1.5,1.5));
        t.assert_equal(8u,state.samples[2]);
        llama_recurrent_placement_request r;r.budget=1000;r.safe_boundary=true;r.generation=1;r.workload=42;
        r.plans={{representation::dense_device,900,0,0,0,1.5,0},
                 {representation::gdn_fp32,800,0,0,0,1,0}};
        llama_recurrent_placement_choice out;
        t.assert_true(llama_recurrent_placement_choose(r,state,out));
        t.assert_true(out.representation==representation::dense_device);
        r.workload=43;r.generation=5;
        t.assert_true(llama_recurrent_placement_choose(r,state,out));
        t.assert_true(out.representation==representation::gdn_fp32);
        t.assert_true(!llama_recurrent_placement_observe(state,representation::gdn_fp32,43,0,1));
    });
    t.test("placement_can_trade_device_state_for_less_KV_streaming", [](testing & t) {
        llama_recurrent_placement_request r;r.safe_boundary=true;r.budget=1000;r.generation=1;
        r.h2d_bytes_per_ms=r.d2h_bytes_per_ms=100;
        r.plans={{representation::dense_device,950,0,0,0,1,0},
                 {representation::gdn_fp32,800,0,0,0,1.1,0},
                 {representation::dense_host,600,5,0,30,1.2,0}};
        llama_recurrent_placement_state state;llama_recurrent_placement_choice out;
        t.assert_true(llama_recurrent_placement_choose(r,state,out));
        t.assert_true(out.representation==representation::dense_device);
        r.workload=1;r.generation=2;
        r.plans[0].kv_stream_bytes=80;r.plans[1].kv_stream_bytes=50;r.plans[2].kv_stream_bytes=20;
        t.assert_true(llama_recurrent_placement_choose(r,state,out));
        t.assert_true(out.representation==representation::dense_host);
        r.generation=1;
        t.assert_true(!llama_recurrent_placement_choose(r,state,out));
        r.generation=3;r.plans.push_back(r.plans[0]);
        t.assert_true(!llama_recurrent_placement_choose(r,state,out));
    });
    t.test("placement_keeps_hysteresis_when_the_workload_changes", [](testing & t) {
        llama_recurrent_placement_request r;r.budget=1000;r.safe_boundary=true;r.generation=1;
        r.plans={{representation::dense_device,900,0,0,0,1,0},
                 {representation::gdn_fp32,800,0,0,0,1.02,0}};
        llama_recurrent_placement_state state;llama_recurrent_placement_choice out;
        t.assert_true(llama_recurrent_placement_choose(r,state,out));
        r.workload=1;r.generation=2;r.plans[1].compute_ms=.98;
        t.assert_true(llama_recurrent_placement_choose(r,state,out));
        t.assert_true(out.representation==representation::dense_device);
        r.workload=2;r.generation=3;r.plans[1].compute_ms=.8;
        t.assert_true(llama_recurrent_placement_choose(r,state,out));
        t.assert_true(out.representation==representation::gdn_fp32);
    });
    return t.summary();
}
