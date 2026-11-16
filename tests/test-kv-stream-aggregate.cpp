#include "../src/llama-kv-stream-aggregate.h"
#include "testing.h"
#include <algorithm>
#include <array>
#include <climits>

static llama_kv_stream_policy_config config(uint32_t pages,uint32_t layers) {
    llama_kv_stream_policy_config c;
    c.shape={GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,256,256,4,256,128};
    c.capabilities={{GGML_TYPE_Q8_0,true,true,true,true},{GGML_TYPE_Q4_0,true,true,true,true},true,true};
    c.pool_bytes=size_t(pages)*425984;c.layers=layers;return c;
}
int main() {
    testing t;
    t.test("resident_role_survives_deadline_spreading_without_more_memory",[](testing & t) {
        auto c=config(4870,17);llama_kv_stream_policy_state state;
        if (!t.assert_true(llama_kv_stream_policy_initialize(c,state).status==llama_kv_stream_policy_status::success)) return;
        state.resident_pages_per_layer=269;state.ring_slots=297;state.decode_active_pages=514;state.spread_streaming=true;
        llama_kv_stream_aggregate_layout first;
        for (auto tokens : {std::vector<size_t>{131584,131584},{65792,65792,65792,65792}}) {
            llama_kv_stream_aggregate_layout layout;
            if (!t.assert_true(llama_kv_stream_aggregate_layout_make(c,state,{16,1},tokens,layout,{1}))) return;
            t.assert_equal(514u,layout.storage.layers.back().capacity_pages);
            size_t pages=state.ring_slots,offset=layout.storage.ring.bytes;
            for (size_t layer=0;layer<layout.storage.layers.size();++layer) {
                const auto & part=layout.storage.layers[layer];pages+=part.capacity_pages;
                t.assert_equal(offset,part.offset);offset+=part.planes.bytes;
                if (layer<16) t.assert_true(state.ring_slots>=part.streamed_pages);
            }
            t.assert_equal(size_t(state.budget.pages),pages);t.assert_equal(layout.storage.conversion_offset,offset);
            for (size_t owner=1;owner<tokens.size();owner+=2) t.assert_equal(0u,layout.owners[owner][0].streamed_pages);
            if (!first.storage.layers.empty()) for (size_t layer=0;layer<17;++layer) {
                t.assert_equal(first.storage.layers[layer].capacity_pages,layout.storage.layers[layer].capacity_pages);
                t.assert_equal(first.storage.layers[layer].offset,layout.storage.layers[layer].offset);
            }
            first=std::move(layout);
        }
        t.assert_equal(269u,state.resident_pages_per_layer);t.assert_equal(297u,state.ring_slots);
    });
    t.test("infeasible_role_preference_preserves_the_generic_layout",[](testing & t) {
        auto c=config(64,17);llama_kv_stream_policy_state state;
        if (!t.assert_true(llama_kv_stream_policy_initialize(c,state).status==llama_kv_stream_policy_status::success)) return;
        state.decode_active_pages=514;state.spread_streaming=true;
        llama_kv_stream_aggregate_layout plain,preferred;
        if (!t.assert_true(llama_kv_stream_aggregate_layout_make(c,state,{16,1},{131584,131584},plain) &&
                llama_kv_stream_aggregate_layout_make(c,state,{16,1},{131584,131584},preferred,{1}))) return;
        for (size_t layer=0;layer<17;++layer) t.assert_equal(plain.storage.layers[layer].capacity_pages,preferred.storage.layers[layer].capacity_pages);
        preferred.requirements.active_tokens=73;
        t.assert_true(!llama_kv_stream_aggregate_layout_make(c,state,{16,1},{131584,131584},preferred,{2}));
        t.assert_true(!llama_kv_stream_aggregate_layout_make(c,state,{16,1},{131584,131584},preferred,{1,1}));
        t.assert_equal(size_t(73),preferred.requirements.active_tokens);
    });
    t.test("role_preferences_can_spend_the_remaining_quota_on_one_cache",[](testing & t) {
        auto c=config(39,3);llama_kv_stream_policy_state state;
        if (!t.assert_true(llama_kv_stream_policy_initialize(c,state).status==llama_kv_stream_policy_status::success)) return;
        state.resident_pages_per_layer=6;state.ring_slots=21;state.decode_active_pages=18;state.spread_streaming=true;
        llama_kv_stream_aggregate_layout layout;
        if (!t.assert_true(llama_kv_stream_aggregate_layout_make(c,state,{2,1},{4608,4608},layout,{1}))) return;
        t.assert_equal(0u,layout.storage.layers[0].capacity_pages);t.assert_equal(0u,layout.storage.layers[1].capacity_pages);
        t.assert_equal(18u,layout.storage.layers[2].capacity_pages);t.assert_equal(size_t(0),layout.owners[0][0].planes.bytes);
        t.assert_equal(layout.storage.conversion_offset,layout.storage.ring.bytes+layout.storage.layers[2].planes.bytes);
    });
    t.test("request_decomposition_is_absent_from_scalar_policy",[](testing & t) {
        auto c=config(4381,16);llama_kv_stream_policy_state initial;
        if (!t.assert_true(llama_kv_stream_policy_initialize(c,initial).status==llama_kv_stream_policy_status::success)) return;
        for (auto tokens : {std::vector<size_t>{65536,65536},{131072},{256,130816},{0,131072}}) {
            llama_kv_stream_aggregate_requirements demand;
            if (!t.assert_true(llama_kv_stream_aggregate_requirements_make({16},tokens,demand))) return;
            t.assert_equal(size_t(131072),demand.active_tokens);
            llama_kv_stream_policy_decision decision;
            if (!t.assert_true(llama_kv_stream_policy_step(c,initial,{demand.active_tokens,8,{},false},decision).status==llama_kv_stream_policy_status::success)) return;
            t.assert_equal(256u,decision.next.resident_pages_per_layer);t.assert_equal(285u,decision.next.ring_slots);
            t.assert_equal(16u,decision.next.budget.layers);
        }
    });
    t.test("role_extents_share_layer_count_and_account_for_partial_pages",[](testing & t) {
        llama_kv_stream_aggregate_requirements demand;
        if (!t.assert_true(llama_kv_stream_aggregate_requirements_make({16,1},{65537,65536,65537,65539},demand))) return;
        t.assert_equal(size_t(514*256),demand.active_tokens);
        t.assert_true(demand.owner_pages==std::vector<size_t>({257,256,257,257}));
        t.assert_true(demand.owner_first==std::vector<size_t>({0,0,257*256,256*256}));
        t.assert_true(demand.role_tokens==std::vector<size_t>({514*256,513*256}));
    });
    t.test("resident_slices_cover_scalar_planes_without_overlap",[](testing & t) {
        for (auto roles : {std::vector<uint32_t>{16},{16,1}}) for (auto tokens : {
                std::vector<size_t>{65537,65537}, {1024,98304}}) {
            std::vector<size_t> owners;
            for (auto token : tokens) for (size_t role=0;role<roles.size();++role) owners.push_back(token-role);
            const uint32_t layers=roles.size()==1 ? 16 : 17;
            auto c=config(4381,layers);llama_kv_stream_policy_state initial;llama_kv_stream_policy_decision decision;
            llama_kv_stream_aggregate_requirements demand;llama_kv_stream_aggregate_layout layout;
            if (!t.assert_true(llama_kv_stream_aggregate_requirements_make(roles,owners,demand) &&
                    llama_kv_stream_policy_initialize(c,initial).status==llama_kv_stream_policy_status::success &&
                    llama_kv_stream_policy_step(c,initial,{demand.active_tokens,8,{},false},decision).status==llama_kv_stream_policy_status::success &&
                    llama_kv_stream_aggregate_layout_make(c,decision.next,roles,owners,layout))) return;
            t.assert_equal(size_t(layers),layout.storage.layers.size());
            size_t global=0;
            for (size_t role=0;role<roles.size();++role) for (size_t layer=0;layer<roles[role];++layer,++global) {
                const auto & plane=layout.storage.layers[global];size_t pages=0;
                for (size_t request=0;request<tokens.size();++request) {
                    const auto & part=layout.owners[request*roles.size()+role][layer];
                    t.assert_equal(plane.offset+pages*plane.planes.k_token_bytes*256,part.offset);
                    t.assert_equal(plane.offset+plane.planes.v_offset+pages*plane.planes.v_token_bytes*256,part.offset+part.planes.v_offset);
                    t.assert_true(part.offset>=layout.storage.ring.bytes && part.offset+part.planes.k_bytes<=plane.offset+plane.planes.k_bytes);
                    t.assert_true(part.offset+part.planes.v_offset+part.planes.v_bytes<=plane.offset+plane.planes.v_offset+plane.planes.v_bytes);
                    pages+=part.capacity_pages;
                }
                t.assert_equal(size_t(plane.capacity_pages),pages);
            }
        }
    });
    t.test("single_owner_layout_is_the_original_serial_layout",[](testing & t) {
        auto c=config(4914,16);llama_kv_stream_policy_state initial;llama_kv_stream_policy_decision decision;
        if (!t.assert_true(llama_kv_stream_policy_initialize(c,initial).status==llama_kv_stream_policy_status::success &&
                llama_kv_stream_policy_step(c,initial,{130568,8,{},false},decision).status==llama_kv_stream_policy_status::success)) return;
        llama_kv_stream_aggregate_layout layout;
        if (!t.assert_true(llama_kv_stream_aggregate_layout_make(c,decision.next,{16},{130568},layout))) return;
        for (size_t layer=0;layer<16;++layer) {
            const auto & a=layout.storage.layers[layer];const auto & b=layout.owners[0][layer];
            t.assert_equal(a.offset,b.offset);t.assert_equal(a.capacity_pages,b.capacity_pages);
            t.assert_equal(a.planes.v_offset,b.planes.v_offset);t.assert_equal(a.planes.bytes,b.planes.bytes);
        }
    });
    t.test("malformed_extents_preserve_output",[](testing & t) {
        llama_kv_stream_aggregate_requirements demand;demand.active_tokens=73;
        t.assert_true(!llama_kv_stream_aggregate_requirements_make({}, {1},demand));
        t.assert_true(!llama_kv_stream_aggregate_requirements_make({16,0},{1,1},demand));
        t.assert_true(!llama_kv_stream_aggregate_requirements_make({16,1},{1},demand));
        t.assert_true(!llama_kv_stream_aggregate_requirements_make({16},{SIZE_MAX},demand));
        t.assert_equal(size_t(73),demand.active_tokens);
    });
    return t.summary();
}
