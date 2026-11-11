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

// Enumerate active lower bounds, then all integer partitions; minimize exact distance to the continuous target.
static std::vector<uint32_t> page_oracle(const std::vector<uint32_t> & weights,
        const std::vector<uint32_t> & minima,uint32_t pages) {
    uint64_t denominator=0;
    std::vector<uint64_t> target(weights.size());
    for (uint32_t mask=0;mask<(uint32_t(1)<<weights.size());++mask) {
        uint32_t reserved=0;uint64_t weight=0;
        for (size_t i=0;i<weights.size();++i) {
            if (mask&(uint32_t(1)<<i)) reserved+=minima[i];else weight+=weights[i];
        }
        if (reserved>pages) continue;
        if (!weight) {
            if (reserved!=pages) continue;
            denominator=1;
            for (size_t i=0;i<weights.size();++i) target[i]=minima[i];
            break;
        }
        bool valid=true;
        for (size_t i=0;i<weights.size();++i) {
            const uint64_t bound=uint64_t(minima[i])*weight,ideal=uint64_t(pages-reserved)*weights[i];
            if (mask&(uint32_t(1)<<i)) { valid&=bound>=ideal;target[i]=bound; }
            else { valid&=bound<=ideal;target[i]=ideal; }
        }
        if (valid) { denominator=weight;break; }
    }
    if (!denominator) return {};
    uint64_t best=UINT64_MAX;
    std::vector<uint32_t> answer,candidate(weights.size());
    const auto search=[&](const auto & self,size_t index,uint32_t used) -> void {
        if (index==weights.size()) {
            if (used!=pages) return;
            uint64_t score=0;
            for (size_t i=0;i<weights.size();++i) {
                const uint64_t scaled=uint64_t(candidate[i])*denominator;
                const uint64_t delta=scaled>target[i] ? scaled-target[i] : target[i]-scaled;
                score+=delta*delta;
            }
            if (score<best || (score==best && candidate>answer)) { best=score;answer=candidate; }
            return;
        }
        for (uint32_t count=minima[index];count<=pages-used;++count) {
            candidate[index]=count;self(self,index+1,used+count);
        }
    };
    search(search,0,0);
    return answer;
}

static selection projected_selection(const std::vector<uint32_t> & weights) {
    selection result;
    for (size_t i=0;i<weights.size();++i) {
        result.slots.push_back({uint32_t(i*3+1),size_t(weights[i]-1),1,result.context_tokens});
        result.context_tokens+=weights[i];result.history_tokens+=weights[i]-1;
    }
    return result;
}

int main() {
    testing t;
    t.test("request_groups_freeze_membership_and_finish_exclusive_preparation_before_decode", [](testing & t) {
        using phase=llama_kv_stream_group_phase;
        llama_kv_stream_request_group group({128,2});
        t.assert_true(group.phase()==phase::idle);
        t.assert_true(group.admit({{1,12,0,5,true},{0,11,0,5,false}}));
        t.assert_true(!group.admit({{2,13,0,5,false}}));
        t.assert_true(group.busy(0,11) && group.busy(1,12));
        t.assert_true(group.phase()==phase::vision);
        llama_kv_stream_group_plan plan;
        t.assert_true(group.plan(plan)==selection_status::no_progress);
        t.assert_true(!group.prepared(0,11,phase::prefill,30));
        t.assert_true(!group.prepared(1,13,phase::vision,0));
        t.assert_true(group.prepared(1,12,phase::vision,0));
        t.assert_true(group.phase()==phase::prefill);
        t.assert_true(!group.prepared(1,12,phase::prefill,90));
        t.assert_true(group.prepared(0,11,phase::prefill,30));
        t.assert_true(group.prepared(1,12,phase::prefill,90));
        t.assert_true(group.phase()==phase::decode);
        if (!t.assert_true(group.plan(plan)==selection_status::success)) return;
        if (!t.assert_equal(size_t(2),plan.selected.slots.size())) return;
        t.assert_equal(size_t(4),plan.selected.slots[0].growth_tokens);
        t.assert_equal(size_t(4),plan.selected.slots[1].growth_tokens);
        t.assert_equal(size_t(34),plan.selected.slots[1].offset_tokens);
        t.assert_true(group.activate(std::move(plan)));
        t.assert_true(group.selected(0,11) && group.selected(1,12));
        llama_kv_stream_generation_budget budget;
        t.assert_true(group.submission(0,11,7,100,budget));
        t.assert_equal(uint32_t(4),budget.query_rows);
        t.assert_equal(uint32_t(3),budget.proposal_tokens);
        t.assert_true(!group.advance({{0,11,35},{1,12,94}}));
        t.assert_true(group.submission(0,11,7,100,budget));
        t.assert_equal(uint32_t(4),budget.query_rows);
        t.assert_true(group.advance({{0,11,34},{1,12,91}}));
        t.assert_true(group.needs_reselection());
        t.assert_true(!group.submission(0,11,0,100,budget));
        t.assert_true(!group.submission(1,12,0,100,budget));
        if (!t.assert_true(group.plan(plan)==selection_status::success)) return;
        t.assert_equal(size_t(2),plan.selected.slots.size());
        t.assert_equal(size_t(2),plan.selected.slots[0].growth_tokens);
        t.assert_equal(size_t(1),plan.selected.slots[1].growth_tokens);
        t.assert_true(group.activate(std::move(plan)));
        t.assert_true(group.advance({{0,11,36},{1,12,92}}));
        if (!t.assert_true(group.plan(plan)==selection_status::success)) return;
        t.assert_equal(size_t(1),plan.selected.slots.size());
        t.assert_equal(uint32_t(0),plan.selected.slots[0].id);
        t.assert_true(group.activate(std::move(plan)));
        t.assert_true(group.selected(0,11) && !group.selected(1,12));
        t.assert_true(group.busy(1,12));
        t.assert_true(!group.finish(1,13));
        t.assert_true(group.finish(0,11));
        if (!t.assert_true(group.plan(plan)==selection_status::success)) return;
        t.assert_equal(uint32_t(1),plan.selected.slots[0].id);
        t.assert_true(group.activate(std::move(plan)));
        t.assert_true(group.finish(1,12));
        t.assert_true(group.phase()==phase::retiring);
        t.assert_true(!group.admit({{0,14,0,5,false}}));
        t.assert_true(group.release());
        t.assert_true(group.phase()==phase::idle);
        t.assert_true(group.admit({{0,14,0,5,false}}));
        t.assert_true(!group.busy(0,11));
    });
    t.test("request_epochs_are_per_slot_and_cannot_reuse_a_retired_namespace", [](testing & t) {
        llama_kv_stream_request_group group({32,2});
        if (!t.assert_true("different slots may start with the same request epoch",group.admit({{0,1,0,0,false},{1,1,0,0,false}}))) return;
        t.assert_true(group.finish(0,1) && group.finish(1,1) && group.release());
        t.assert_true(!group.admit({{0,1,0,0,false},{1,2,0,0,false}}));
        if (!t.assert_true(group.admit({{0,2,0,0,false},{1,2,0,0,false}}))) return;
        t.assert_true(!group.finish(0,1) && !group.finish(1,1));
        t.assert_true(group.finish(0,2) && group.finish(1,2) && group.release());
        if (!t.assert_true(group.admit({{0,UINT64_MAX,0,0,false}}))) return;
        t.assert_true(group.finish(0,UINT64_MAX) && group.release());
        t.assert_true(!group.admit({{0,UINT64_MAX,0,0,false}}));
        t.assert_true(group.admit({{1,3,0,0,false}}));
    });
    t.test("request_group_stale_plans_and_partial_progress_cannot_change_peers", [](testing & t) {
        using phase=llama_kv_stream_group_phase;
        llama_kv_stream_request_group group({32,2});
        t.assert_true(group.admit({{0,71,0,9,false},{1,72,0,9,false}}));
        t.assert_true(group.prepared(0,71,phase::prefill,8));
        t.assert_true(group.prepared(1,72,phase::prefill,8));
        llama_kv_stream_group_plan proposal;
        if (!t.assert_true(group.plan(proposal)==selection_status::success)) return;
        auto stale=proposal;
        t.assert_true(group.activate(std::move(proposal)));
        t.assert_true(!group.activate(std::move(stale)));
        t.assert_true(group.plan(proposal)==selection_status::success);
        t.assert_true(group.advance({{0,71,9},{1,72,10}}));
        t.assert_true(!group.activate(std::move(proposal)));
        llama_kv_stream_generation_budget left,right;
        t.assert_true(group.submission(0,71,7,100,left));
        t.assert_true(group.submission(1,72,7,100,right));
        t.assert_equal(uint32_t(7),left.query_rows);
        t.assert_equal(uint32_t(6),right.query_rows);
        t.assert_true(!group.advance({{0,71,10},{1,73,11}}));
        t.assert_true(!group.advance({{0,71,10},{0,71,11}}));
        t.assert_true(!group.advance({{0,71,8}}));
        t.assert_true(group.submission(0,71,7,100,left));
        t.assert_equal(uint32_t(7),left.query_rows);
        t.assert_true(group.submission(1,72,7,100,right));
        t.assert_equal(uint32_t(6),right.query_rows);
        t.assert_true(!group.release());
        t.assert_true(group.finish(0,71));
        t.assert_true(group.needs_reselection());
        t.assert_true(!group.submission(1,72,0,100,right));
        t.assert_true(group.plan(proposal)==selection_status::success);
        auto old_epoch=proposal;
        t.assert_true(group.finish(1,72));
        t.assert_true(group.release());
        t.assert_true(group.admit({{0,81,0,9,false},{1,82,0,9,false}}));
        t.assert_true(group.prepared(0,81,phase::prefill,9));
        t.assert_true(group.prepared(1,82,phase::prefill,10));
        t.assert_true(!group.activate(std::move(old_epoch)));
        t.assert_true(!group.finish(1,72));
    });
    t.test("request_group_cancellation_keeps_phases_exclusive_and_retires_without_refill", [](testing & t) {
        using phase=llama_kv_stream_group_phase;
        for (int cancel=0;cancel<3;++cancel) {
            llama_kv_stream_request_group group({32,2});
            t.assert_true(group.admit({{0,1,0,0,true},{1,2,0,0,true}}));
            llama_kv_stream_group_request next;
            t.assert_true(group.next_preparation(next) && next.slot==0);
            if (cancel==0) t.assert_true(group.finish(0,1));
            else t.assert_true(group.prepared(0,1,phase::vision,0));
            t.assert_true(group.phase()==phase::vision);
            t.assert_true(group.next_preparation(next) && next.slot==1);
            t.assert_true(group.prepared(1,2,phase::vision,0));
            t.assert_true(group.phase()==phase::prefill);
            if (cancel==1) t.assert_true(group.finish(0,1));
            else if (cancel==2) t.assert_true(group.prepared(0,1,phase::prefill,8));
            t.assert_true(group.prepared(1,2,phase::prefill,8));
            t.assert_true(group.phase()==phase::decode);
            if (cancel==2) t.assert_true(group.finish(0,1));
            t.assert_true(group.busy(1,2) && !group.busy(0,1));
            t.assert_true(!group.admit({{0,3,0,0,false}}));
            t.assert_true(group.finish(1,2));
            t.assert_true(group.phase()==phase::retiring);
            t.assert_true(!group.next_preparation(next));
            t.assert_true(group.release());
            t.assert_true(!group.release());
        }
    });
    t.test("request_group_dynamic_allowances_match_the_independent_selection_oracle", [](testing & t) {
        using phase=llama_kv_stream_group_phase;
        for (size_t capacity=4;capacity<=14;++capacity) for (uint32_t count=1;count<=3;++count)
                for (size_t first=0;first<capacity;++first) {
            llama_kv_stream_request_group group({capacity,count});
            std::vector<llama_kv_stream_group_request> requests;
            std::vector<selection_slot> live;
            for (uint32_t slot=0;slot<count;++slot) {
                const size_t history=(first+slot*3)%capacity;
                requests.push_back({slot,uint64_t(slot)+100,0,0,false});
                live.push_back({slot,history,1,0,true});
            }
            if (!t.assert_true(group.admit(requests))) return;
            for (const auto & slot : live)
                if (!t.assert_true(group.prepared(slot.id,uint64_t(slot.id)+100,phase::prefill,slot.history_tokens))) return;
            size_t rounds=0;
            while (group.phase()==phase::decode) {
                if (!t.assert_true(++rounds<=capacity*count)) return;
                const auto expected=oracle(capacity,count,live,0);
                llama_kv_stream_group_plan proposal;
                if (!t.assert_true(group.plan(proposal)==selection_status::success)) return;
                t.assert_true(expected==ids(proposal.selected));
                auto chosen=proposal.selected;
                if (!t.assert_true(group.activate(std::move(proposal)))) return;
                while (group.phase()==phase::decode && !group.needs_reselection()) {
                    std::vector<llama_kv_stream_group_progress> progress;
                    for (const auto & slot : chosen.slots) {
                        llama_kv_stream_generation_budget work;
                        if (!t.assert_true(group.submission(slot.id,uint64_t(slot.id)+100,7,SIZE_MAX,work))) return;
                        const size_t remaining=slot.history_tokens+slot.growth_tokens-live[slot.id].history_tokens;
                        t.assert_equal(uint32_t(std::min(remaining,size_t(8))),work.query_rows);
                        // Each independently accepted prefix consumes its own allowance.
                        const size_t accepted=std::min(size_t(work.query_rows),size_t(1+slot.id%3));
                        live[slot.id].history_tokens+=accepted;
                        progress.push_back({slot.id,uint64_t(slot.id)+100,live[slot.id].history_tokens});
                    }
                    if (!t.assert_true(group.advance(progress))) return;
                    for (auto & slot : live) if (slot.eligible && slot.history_tokens==capacity) {
                        t.assert_true(group.finish(slot.id,uint64_t(slot.id)+100));slot.eligible=false;
                    }
                }
            }
            t.assert_true(group.phase()==phase::retiring && group.release());
        }
    });
    t.test("selected_page_quotas_follow_projected_extent_not_logical_offsets", [](testing & t) {
        selection chosen;
        if (!t.assert_true(llama_kv_stream_select({128*1024,2},{{1,30*1024},{0,90*1024}},chosen)==selection_status::success)) return;
        llama_kv_stream_page_partition physical;
        const auto status=llama_kv_stream_partition_pages(chosen,64,{{1,2},{0,2}},physical);
        if (!t.assert_true(status.status==llama_kv_stream_page_partition_status::success)) return;
        if (!t.assert_equal(size_t(2),physical.slots.size())) return;
        t.assert_equal(uint32_t(47),physical.slots[0].pages);
        t.assert_equal(uint32_t(17),physical.slots[1].pages);
        t.assert_equal(uint32_t(0),physical.slots[0].first_page);
        t.assert_equal(uint32_t(47),physical.slots[1].first_page);
        t.assert_equal(size_t(94*1024),physical.slots[1].logical_offset);
        t.assert_equal(size_t(34*1024),physical.slots[1].projected_tokens);
        t.assert_equal(uint32_t(64),physical.pages);
        t.assert_equal(uint64_t(4),status.minimum_pages);
    });
    t.test("native_page_minima_are_enforced_without_cross_slot_borrowing", [](testing & t) {
        selection chosen;
        if (!t.assert_true(llama_kv_stream_select({128,2},{{0,90},{1,30}},chosen)==selection_status::success)) return;
        llama_kv_stream_page_partition physical;
        auto result=llama_kv_stream_partition_pages(chosen,13,{{0,2},{1,5}},physical);
        if (!t.assert_true(result.status==llama_kv_stream_page_partition_status::success)) return;
        t.assert_equal(uint32_t(8),physical.slots[0].pages);
        t.assert_equal(uint32_t(5),physical.slots[1].pages);
        result=llama_kv_stream_partition_pages(chosen,6,{{0,2},{1,5}},physical);
        t.assert_true(result.status==llama_kv_stream_page_partition_status::insufficient_pages);
        t.assert_equal(uint64_t(7),result.minimum_pages);
        t.assert_equal(uint32_t(13),physical.pages);
        t.assert_equal(uint32_t(5),physical.slots[1].pages);
    });
    t.test("page_partition_matches_independent_small_integer_oracle", [](testing & t) {
        using ps=llama_kv_stream_page_partition_status;
        for (uint32_t w0=1;w0<=4;++w0) for (uint32_t w1=1;w1<=4;++w1) for (uint32_t w2=1;w2<=4;++w2)
        for (uint32_t m0=1;m0<=3;++m0) for (uint32_t m1=1;m1<=3;++m1) for (uint32_t m2=1;m2<=3;++m2)
        for (uint32_t pages=1;pages<=12;++pages) {
            const std::vector<uint32_t> weights{w0,w1,w2},minimum{m0,m1,m2};
            const auto chosen=projected_selection(weights);
            llama_kv_stream_page_partition output;output.pages=99;
            const auto status=llama_kv_stream_partition_pages(chosen,pages,{{7,m2},{1,m0},{4,m1}},output);
            if (m0+m1+m2>pages) {
                if (!t.assert_true(status.status==ps::insufficient_pages && status.minimum_pages==m0+m1+m2 && output.pages==99)) return;
                continue;
            }
            const auto expected=page_oracle(weights,minimum,pages);
            if (!t.assert_true(status.status==ps::success && output.slots.size()==3 && expected.size()==3)) return;
            uint32_t end=0;
            for (size_t i=0;i<3;++i) {
                if (!t.assert_equal(expected[i],output.slots[i].pages)) return;
                if (!t.assert_equal(end,output.slots[i].first_page)) return;
                end+=output.slots[i].pages;
            }
            if (!t.assert_equal(pages,end)) return;
        }
    });
    t.test("page_partition_large_products_and_future_slot_counts_do_not_wrap", [](testing & t) {
        using ps=llama_kv_stream_page_partition_status;
        auto chosen=projected_selection({uint32_t(INT32_MAX-1),1});
        llama_kv_stream_page_partition output;
        if (!t.assert_true(llama_kv_stream_partition_pages(chosen,UINT32_MAX,{{1,1},{4,1}},output).status==ps::success)) return;
        t.assert_equal(UINT32_MAX-2,output.slots[0].pages);t.assert_equal(uint32_t(2),output.slots[1].pages);
        t.assert_equal(UINT32_MAX-2,output.slots[1].first_page);
        chosen=projected_selection({1,1,1});
        const auto failed=llama_kv_stream_partition_pages(chosen,UINT32_MAX,{{1,UINT32_MAX},{4,UINT32_MAX},{7,UINT32_MAX}},output);
        t.assert_true(failed.status==ps::insufficient_pages);
        t.assert_equal(uint64_t(UINT32_MAX)*3,failed.minimum_pages);
        t.assert_equal(uint32_t(UINT32_MAX),output.pages);
        chosen=projected_selection(std::vector<uint32_t>(256,1));
        std::vector<llama_kv_stream_page_minimum> minima;
        for (const auto & slot : chosen.slots) minima.push_back({slot.id,1});
        if (!t.assert_true(llama_kv_stream_partition_pages(chosen,UINT32_MAX,minima,output).status==ps::success)) return;
        uint64_t end=0;
        for (size_t i=0;i<output.slots.size();++i) {
            t.assert_equal(i==255 ? uint32_t(16777215) : uint32_t(16777216),output.slots[i].pages);
            t.assert_equal(end,uint64_t(output.slots[i].first_page));end+=output.slots[i].pages;
        }
        t.assert_equal(uint64_t(UINT32_MAX),end);
    });
    t.test("malformed_page_inputs_leave_the_previous_partition_unchanged", [](testing & t) {
        using ps=llama_kv_stream_page_partition_status;
        const auto good=projected_selection({3,4});
        llama_kv_stream_page_partition saved; saved.pages=77;saved.slots={{99,4,5,6,7}};
        for (int mutation=0;mutation<10;++mutation) {
            auto invalid=good;
            if (mutation==0) invalid.context_tokens=0;
            if (mutation==1) invalid.context_tokens=SIZE_MAX;
            if (mutation==2) invalid.slots.clear();
            if (mutation==3) invalid.history_tokens++;
            if (mutation==4) invalid.slots[0].growth_tokens=0;
            if (mutation==5) invalid.slots[1].offset_tokens++;
            if (mutation==6) invalid.slots[1].id=invalid.slots[0].id;
            if (mutation==7) invalid.slots[0].history_tokens=SIZE_MAX;
            if (mutation==8) invalid.slots[0].growth_tokens=SIZE_MAX;
            if (mutation==9) std::swap(invalid.slots[0],invalid.slots[1]);
            const auto result=llama_kv_stream_partition_pages(invalid,10,{{1,1},{4,1}},saved);
            t.assert_true(result.status==ps::invalid_selection);
            t.assert_true(saved.pages==77 && saved.slots.size()==1 && saved.slots[0].slot==99 && saved.slots[0].projected_tokens==7);
        }
        for (const auto & minima : std::vector<std::vector<llama_kv_stream_page_minimum>>{
                {},{{1,1}},{{1,1},{4,1},{7,1}},{{1,1},{1,1}},{{1,0},{4,1}},{{1,1},{7,1}}}) {
            t.assert_true(llama_kv_stream_partition_pages(good,10,minima,saved).status==ps::invalid_minimum);
            t.assert_equal(uint32_t(77),saved.pages);
        }
    });
    t.test("page_partition_rejects_edited_equal_growth_snapshot", [](testing & t) {
        selection chosen;
        if (!t.assert_true(llama_kv_stream_select({10,2},{{0,2},{1,2}},chosen)==selection_status::success)) return;
        ++chosen.slots[0].growth_tokens;--chosen.slots[1].growth_tokens;++chosen.slots[1].offset_tokens;
        llama_kv_stream_page_partition saved;saved.pages=77;
        const auto status=llama_kv_stream_partition_pages(chosen,10,{{0,1},{1,1}},saved);
        t.assert_true("complete extents cannot bypass the equal allowance contract",
            status.status==llama_kv_stream_page_partition_status::invalid_selection && saved.pages==77);
    });
    t.test("slot_logical_group_and_total_storage_limits_are_distinct", [](testing & t) {
        llama_kv_stream_context_limits limits;
        t.assert_true(llama_kv_stream_context_limits_make(1000, 1024, 2, limits));
        t.assert_equal(uint32_t(1000), limits.slot_limit);
        t.assert_equal(uint32_t(1000), limits.group_limit);
        t.assert_equal(uint32_t(1024), limits.slot_storage);
        t.assert_equal(uint32_t(2048), limits.history_storage);
        t.assert_equal(uint32_t(2), limits.slots);
        for (const auto & invalid : std::vector<std::vector<size_t>>{
                {0, 1024, 2}, {1000, 999, 2}, {1000, 1024, 0}, {1000, 1024, 257},
                {SIZE_MAX, SIZE_MAX, 2}, {1, size_t(INT32_MAX), 3}}) {
            t.assert_true(!llama_kv_stream_context_limits_make(invalid[0], invalid[1], invalid[2], limits));
            t.assert_equal(uint32_t(2048), limits.history_storage);
        }
        t.assert_true(llama_kv_stream_context_limits_make(8192, 8192, 6, limits));
        t.assert_equal(uint32_t(49152), limits.history_storage);
        t.assert_equal(uint32_t(8192), limits.group_limit);
    });
    t.test("generation_queries_and_bonus_fit_equal_logical_allowance", [](testing & t) {
        llama_kv_stream_generation_budget budget;
        for (size_t headroom : {size_t(1),size_t(2),size_t(4),size_t(8),size_t(9),SIZE_MAX}) {
            for (uint32_t proposals : {0u,1u,3u,7u}) for (size_t remaining : {size_t(1),size_t(3),SIZE_MAX}) {
                t.assert_true(llama_kv_stream_generation_budget_make(headroom,proposals,remaining,budget));
                t.assert_true(budget.query_rows >= 1 && budget.query_rows <= 8);
                t.assert_equal(budget.query_rows-1,budget.proposal_tokens);
                t.assert_true(budget.query_rows <= headroom && budget.query_rows <= remaining);
                t.assert_true(budget.proposal_tokens <= proposals);
            }
        }
        budget={77,66};
        t.assert_true(!llama_kv_stream_generation_budget_make(0,7,SIZE_MAX,budget));
        t.assert_true(!llama_kv_stream_generation_budget_make(8,7,0,budget));
        t.assert_true(!llama_kv_stream_generation_budget_make(16,15,SIZE_MAX,budget));
        t.assert_equal(uint32_t(77),budget.query_rows);
        t.assert_equal(uint32_t(66),budget.proposal_tokens);
    });
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
