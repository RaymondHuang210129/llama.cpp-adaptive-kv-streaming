#include "../src/llama-kv-stream-selection.h"
#include "testing.h"

#include <algorithm>
#include <limits>

using selection_status = llama_kv_stream_selection_status;
using selection_slot = llama_kv_stream_selection_slot;
using selection = llama_kv_stream_selection;

static std::vector<uint32_t> ids(const selection & plan) {
    std::vector<uint32_t> result;
    for (const auto & slot : plan.slots) result.push_back(slot.id);
    return result;
}

static bool admit(const selection & plan, void * data) {
    const auto mask = *static_cast<const uint32_t *>(data);
    for (const auto & slot : plan.slots) {
        if (slot.id >= 32 || (mask & (uint32_t(1) << slot.id))) return false;
    }
    return true;
}

// Exhaust all subsets independently of the production search and its pruning.
static std::vector<uint32_t> oracle(size_t capacity, size_t maximum,
        const std::vector<selection_slot> & slots, uint32_t forbidden) {
    std::vector<uint32_t> best;
    size_t best_history = 0;
    for (uint32_t mask = 1; mask < (uint32_t(1) << slots.size()); ++mask) {
        std::vector<const selection_slot *> chosen;
        size_t history = 0;
        bool valid = true;
        for (size_t i = 0; i < slots.size(); ++i) if (mask & (uint32_t(1) << i)) {
            const auto & slot = slots[i];
            if (!slot.eligible || (forbidden & (uint32_t(1) << slot.id)) ||
                    (!chosen.empty() && chosen.front()->compatibility != slot.compatibility)) valid = false;
            chosen.push_back(&slot);
            history += slot.history_tokens;
        }
        if (!valid || chosen.size() > maximum || history > capacity) continue;
        std::sort(chosen.begin(), chosen.end(), [](const selection_slot * a, const selection_slot * b) {
            return a->id < b->id;
        });
        std::vector<uint32_t> candidate;
        for (size_t i = 0; i < chosen.size(); ++i) {
            const size_t growth = (capacity - history)/chosen.size() + (i < (capacity - history)%chosen.size());
            if (growth < chosen[i]->minimum_growth) valid = false;
            candidate.push_back(chosen[i]->id);
        }
        if (valid && (candidate.size() > best.size() || (candidate.size() == best.size() &&
                (history < best_history || (history == best_history && candidate < best))))) {
            best = candidate;
            best_history = history;
        }
    }
    return best;
}

int main() {
    testing t;
    t.test("ninety_thirty_equal_growth_and_packed_offsets", [](testing & t) {
        selection plan;
        t.assert_true(llama_kv_stream_select({128*1024, 2},
            {{1, 30*1024}, {0, 90*1024}}, plan) == selection_status::success);
        if (!t.assert_equal(size_t(2), plan.slots.size())) return;
        t.assert_true(ids(plan) == std::vector<uint32_t>({0, 1}));
        t.assert_equal(size_t(120*1024), plan.history_tokens);
        t.assert_equal(size_t(4*1024), plan.slots[0].growth_tokens);
        t.assert_equal(size_t(4*1024), plan.slots[1].growth_tokens);
        t.assert_equal(size_t(0), plan.slots[0].offset_tokens);
        t.assert_equal(size_t(94*1024), plan.slots[1].offset_tokens);
        t.assert_equal(size_t(128*1024), plan.context_tokens);
    });
    t.test("maximum_cardinality_then_history_then_slot_order", [](testing & t) {
        selection plan;
        t.assert_true(llama_kv_stream_select({100, 2}, {{0, 90}, {2, 20}, {1, 20}}, plan) == selection_status::success);
        t.assert_true(ids(plan) == std::vector<uint32_t>({1, 2}));
        t.assert_true(llama_kv_stream_select({100, 2}, {{8, 40}, {3, 40}, {4, 40}}, plan) == selection_status::success);
        t.assert_true(ids(plan) == std::vector<uint32_t>({3, 4}));
        t.assert_true(llama_kv_stream_select({100, 2}, {{0, 99}, {1, 0}}, plan) == selection_status::success);
        t.assert_true(ids(plan) == std::vector<uint32_t>({1}));
    });
    t.test("compatibility_and_measured_admission_are_not_bypassed", [](testing & t) {
        selection plan;
        uint32_t forbidden = 1;
        const std::vector<selection_slot> slots = {{0, 0, 1, 1}, {1, 1, 1, 2}, {2, 2, 1, 2}};
        t.assert_true(llama_kv_stream_select({20, 3}, slots, plan, admit, &forbidden) == selection_status::success);
        t.assert_true(ids(plan) == std::vector<uint32_t>({1, 2}));
        forbidden = 7;
        t.assert_true(llama_kv_stream_select({20, 3}, slots, plan, admit, &forbidden) == selection_status::no_progress);
        const auto reject_pair = [](const selection & candidate, void *) {
            return ids(candidate) != std::vector<uint32_t>({0, 1});
        };
        t.assert_true(llama_kv_stream_select({20, 2}, {{0, 1}, {1, 2}, {2, 3}}, plan, reject_pair) == selection_status::success);
        t.assert_true(ids(plan) == std::vector<uint32_t>({0, 2}));
    });
    t.test("remainder_and_page_tails_are_logical_not_physical", [](testing & t) {
        selection plan;
        t.assert_true(llama_kv_stream_select({1025, 2}, {{1, 256}, {0, 256}}, plan) == selection_status::success);
        t.assert_equal(size_t(257), plan.slots[0].growth_tokens);
        t.assert_equal(size_t(256), plan.slots[1].growth_tokens);
        t.assert_equal(size_t(513), plan.slots[1].offset_tokens);
        t.assert_true(llama_kv_stream_select({3, 3}, {{2, 0}, {0, 0}, {1, 0}}, plan) == selection_status::success);
        t.assert_equal(size_t(3), plan.slots.size());
        for (const auto & slot : plan.slots) t.assert_equal(size_t(1), slot.growth_tokens);
    });
    t.test("reselection_preserves_histories_and_ineligible_completed_members", [](testing & t) {
        std::vector<selection_slot> slots = {{0, 90}, {1, 30}};
        selection plan;
        t.assert_true(llama_kv_stream_select({128, 2}, slots, plan) == selection_status::success);
        slots[0].history_tokens += plan.slots[0].growth_tokens;
        slots[1].history_tokens += 1;
        t.assert_true(llama_kv_stream_select({128, 2}, slots, plan) == selection_status::success);
        t.assert_equal(size_t(94), slots[0].history_tokens);
        t.assert_equal(size_t(31), slots[1].history_tokens);
        t.assert_equal(size_t(2), plan.slots[0].growth_tokens);
        t.assert_equal(size_t(1), plan.slots[1].growth_tokens);
        slots[0].history_tokens = 96;
        slots[1].history_tokens = 32;
        t.assert_true(llama_kv_stream_select({128, 2}, slots, plan) == selection_status::success);
        t.assert_true(ids(plan) == std::vector<uint32_t>({1}));
        slots[1].eligible = false;
        t.assert_true(llama_kv_stream_select({128, 2}, slots, plan) == selection_status::success);
        t.assert_true(ids(plan) == std::vector<uint32_t>({0}));
        slots[0].history_tokens = 128;
        t.assert_true(llama_kv_stream_select({128, 2}, slots, plan) == selection_status::no_progress);
        slots[0].eligible = false;
        t.assert_true(llama_kv_stream_select({128, 2}, slots, plan) == selection_status::idle);
    });
    t.test("minimum_operation_cannot_be_replaced_by_a_zero_growth_round", [](testing & t) {
        selection plan;
        t.assert_true(llama_kv_stream_select({10, 2}, {{0, 3, 5}, {1, 3, 1}, {2, 4, 1}}, plan) == selection_status::success);
        t.assert_true(ids(plan) == std::vector<uint32_t>({1, 2}));
        t.assert_true(llama_kv_stream_select({10, 2}, {{0, 8, 3}, {1, 9, 2}}, plan) == selection_status::no_progress);
    });
    t.test("invalid_and_overflowing_inputs_leave_output_unchanged", [](testing & t) {
        selection plan;
        plan.context_tokens = 77;
        plan.history_tokens = 88;
        for (const auto & slots : std::vector<std::vector<selection_slot>>{
                {{0, 1}, {0, 2}}, {{0, 11}}, {{0, 0, 0}}, {{0, 0, 11}}}) {
            t.assert_true(llama_kv_stream_select({10, 2}, slots, plan) == selection_status::invalid_config);
            t.assert_equal(size_t(77), plan.context_tokens);
            t.assert_equal(size_t(88), plan.history_tokens);
        }
        t.assert_true(llama_kv_stream_select({0, 1}, {{0, 0}}, plan) == selection_status::invalid_config);
        t.assert_true(llama_kv_stream_select({10, 0}, {{0, 0}}, plan) == selection_status::invalid_config);
        t.assert_true(llama_kv_stream_select({10, 2}, {}, plan) == selection_status::idle);
        t.assert_true(llama_kv_stream_select({10, 2}, {{0, 10}}, plan) == selection_status::no_progress);
        t.assert_equal(size_t(77), plan.context_tokens);
        t.assert_equal(size_t(88), plan.history_tokens);
        t.assert_true(plan.slots.empty());
        const size_t maximum = std::numeric_limits<size_t>::max();
        t.assert_true(llama_kv_stream_select({maximum, 2}, {{0, maximum-1}, {1, maximum-1}}, plan) == selection_status::success);
        t.assert_true(ids(plan) == std::vector<uint32_t>({0}));
        t.assert_equal(size_t(1), plan.slots[0].growth_tokens);
        t.assert_true(llama_kv_stream_select({maximum, 2}, {{0, maximum}}, plan) == selection_status::no_progress);
    });
    t.test("future_groups_match_exhaustive_oracle_without_rotation", [](testing & t) {
        uint32_t random = 8128;
        for (size_t iteration = 0; iteration < 2048; ++iteration) {
            std::vector<selection_slot> slots;
            for (uint32_t id = 0; id < 6; ++id) {
                random = random*1664525u + 1013904223u;
                slots.push_back({id, (random >> 8)%25, 1+(random >> 16)%4,
                    (random >> 24)%2, (random & 3) != 0});
            }
            const size_t capacity = 25 + iteration%40;
            const size_t maximum = 1 + iteration%6;
            uint32_t forbidden = iteration%64;
            const auto expected = oracle(capacity, maximum, slots, forbidden);
            selection plan;
            const auto result = llama_kv_stream_select({capacity, maximum}, slots, plan, admit, &forbidden);
            if (!t.assert_true(expected.empty() ? result == selection_status::idle || result == selection_status::no_progress :
                    result == selection_status::success)) return;
            if (expected.empty()) continue;
            if (!t.assert_true(expected == ids(plan))) return;
            size_t offset = 0, history = 0;
            for (const auto & chosen : plan.slots) {
                t.assert_equal(offset, chosen.offset_tokens);
                t.assert_true(chosen.growth_tokens > 0);
                history += chosen.history_tokens;
                offset += chosen.history_tokens + chosen.growth_tokens;
            }
            t.assert_equal(capacity, offset);
            t.assert_equal(history, plan.history_tokens);
            std::reverse(slots.begin(), slots.end());
            selection reversed;
            t.assert_true(llama_kv_stream_select({capacity, maximum}, slots, reversed, admit, &forbidden) == selection_status::success);
            t.assert_true(ids(plan) == ids(reversed));
        }
    });
    return t.summary();
}
