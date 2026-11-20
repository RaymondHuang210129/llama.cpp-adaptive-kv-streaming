#include "llama-kv-stream-selection.h"
#include "llama-cparams.h"

#include <algorithm>
#include <new>
#include <utility>

bool llama_kv_stream_context_limits_make(size_t logical_tokens, size_t padded_tokens, size_t slots,
        llama_kv_stream_context_limits & output) noexcept {
    if (!logical_tokens || padded_tokens < logical_tokens || padded_tokens > size_t(INT32_MAX) ||
            !slots || slots > LLAMA_MAX_SEQ || padded_tokens > UINT32_MAX/slots) return false;
    output = {uint32_t(logical_tokens), uint32_t(logical_tokens), uint32_t(padded_tokens),
        uint32_t(padded_tokens*slots), uint32_t(slots)};
    return true;
}

llama_kv_stream_selection_status llama_kv_stream_prefill_plan_make(
        const llama_kv_stream_selection_config & config,
        const std::vector<llama_kv_stream_prefill_slot> & slots, size_t maximum_tokens,
        llama_kv_stream_prefill_plan & output,
        llama_kv_stream_selection_admit admit, void * user_data) {
    using status = llama_kv_stream_selection_status;
    if (!maximum_tokens || maximum_tokens > size_t(INT32_MAX)) return status::invalid_config;
    try {
        std::vector<llama_kv_stream_selection_slot> pending;
        for (const auto & slot : slots) {
            pending.push_back({slot.id, slot.history_tokens, 1, slot.compatibility, slot.remaining_tokens != 0});
        }
        auto limits = config;
        limits.maximum_slots = std::min(limits.maximum_slots, maximum_tokens);
        llama_kv_stream_prefill_plan next;
        const auto result = llama_kv_stream_select(limits, pending, next.selected, admit, user_data);
        if (result != status::success) return result;
        next.tokens_per_slot = maximum_tokens / next.selected.slots.size();
        for (const auto & selected : next.selected.slots) {
            const auto slot = std::find_if(slots.begin(), slots.end(), [&](const auto & value) { return value.id == selected.id; });
            if (slot == slots.end()) return status::invalid_config;
            next.tokens_per_slot = std::min({next.tokens_per_slot, selected.growth_tokens, slot->remaining_tokens});
        }
        if (!next.tokens_per_slot) return status::no_progress;
        output = std::move(next);
        return status::success;
    } catch (const std::bad_alloc &) {
        return status::allocation_failed;
    }
}

bool llama_kv_stream_generation_budget_make(size_t headroom, uint32_t maximum_proposals, size_t remaining_outputs,
        llama_kv_stream_generation_budget & output) noexcept {
    if (!headroom || !remaining_outputs || maximum_proposals > 7) return false;
    const size_t rows=std::min({headroom,remaining_outputs,size_t(maximum_proposals)+1});
    output={uint32_t(rows),uint32_t(rows-1)};
    return true;
}

namespace {
struct selection_search {
    const llama_kv_stream_selection_config & config;
    const std::vector<llama_kv_stream_selection_slot> & slots;
    llama_kv_stream_selection_admit admit;
    void * user_data;
    size_t count = 0;
    std::vector<size_t> chosen;
    llama_kv_stream_selection best;
    bool interrupted = false;

    void visit(size_t first, size_t history) {
        if (interrupted) return;
        if (config.continue_search && !config.continue_search(config.control_data)) {
            interrupted=true;return;
        }
        if (!best.slots.empty() && (history > best.history_tokens ||
                (slots.size()>2 && history==best.history_tokens))) return;
        if (chosen.size() == count) {
            llama_kv_stream_selection candidate;
            candidate.context_tokens = config.context_tokens;
            candidate.history_tokens = history;
            const size_t remaining = config.context_tokens - history;
            const size_t growth = remaining/count;
            const size_t remainder = remaining%count;
            size_t offset = 0;
            for (size_t i = 0; i < count; ++i) {
                const auto & slot = slots[chosen[i]];
                const size_t allowance = growth + (i < remainder);
                if (allowance < slot.minimum_growth) return;
                candidate.slots.push_back({slot.id, slot.history_tokens, allowance, offset});
                offset += slot.history_tokens + allowance;
            }
            if (admit && !admit(candidate, user_data)) return;
            // Traversal follows ascending IDs, so the first equal-history result wins.
            if (best.slots.empty() || history < best.history_tokens) best = std::move(candidate);
            return;
        }
        const size_t needed = count - chosen.size();
        if (slots.size() - first < needed) return;
        if (slots.size()>2) {
            std::vector<size_t> histories;
            for (size_t i=first;i<slots.size();++i)
                if (chosen.empty() || slots[i].compatibility==slots[chosen.front()].compatibility)
                    histories.push_back(slots[i].history_tokens);
            if (histories.size()<needed || count>config.context_tokens-history) return;
            std::partial_sort(histories.begin(),histories.begin()+needed,histories.end());
            size_t lower=history;
            for (size_t i=0;i<needed;++i) {
                if (histories[i]>config.context_tokens-count-lower) return;
                lower+=histories[i];
            }
            // Later equal-history candidates have larger IDs in this traversal.
            if (!best.slots.empty() && lower>=best.history_tokens) return;
        }
        for (size_t i = first; i <= slots.size() - needed; ++i) {
            if (interrupted) return;
            const auto & slot = slots[i];
            if (!chosen.empty() && slot.compatibility != slots[chosen.front()].compatibility) continue;
            if (slot.history_tokens > config.context_tokens - history) continue;
            const size_t next = history + slot.history_tokens;
            if (config.context_tokens - next < count) continue;
            chosen.push_back(i);
            visit(i + 1, next);
            chosen.pop_back();
        }
    }
};
}

llama_kv_stream_selection_status llama_kv_stream_select(
        const llama_kv_stream_selection_config & config,
        const std::vector<llama_kv_stream_selection_slot> & admitted,
        llama_kv_stream_selection & output,
        llama_kv_stream_selection_admit admit, void * user_data) {
    using status = llama_kv_stream_selection_status;
    if (!config.context_tokens || !config.maximum_slots) return status::invalid_config;
    try {
        auto sorted = admitted;
        std::sort(sorted.begin(), sorted.end(), [](const llama_kv_stream_selection_slot & a,
                const llama_kv_stream_selection_slot & b) { return a.id < b.id; });
        bool eligible = false;
        for (size_t i = 0; i < sorted.size(); ++i) {
            const auto & slot = sorted[i];
            if (!slot.minimum_growth || slot.minimum_growth > config.context_tokens ||
                    slot.history_tokens > config.context_tokens || (i && slot.id == sorted[i-1].id)) {
                return status::invalid_config;
            }
            eligible = eligible || slot.eligible;
        }
        sorted.erase(std::remove_if(sorted.begin(), sorted.end(), [&](const llama_kv_stream_selection_slot & slot) {
            return !slot.eligible || slot.minimum_growth > config.context_tokens - slot.history_tokens;
        }), sorted.end());
        selection_search search{config, sorted, admit, user_data, 0, {}, {}};
        const auto maximum=config.maximum_selected_slots ? std::min(config.maximum_slots,config.maximum_selected_slots) : config.maximum_slots;
        for (size_t count = std::min(maximum, sorted.size()); count > 0; --count) {
            search.count = count;
            search.visit(0, 0);
            if (search.interrupted) return status::interrupted;
            if (config.continue_search && !config.continue_search(config.control_data)) return status::interrupted;
            if (!search.best.slots.empty()) {
                output = std::move(search.best);
                return status::success;
            }
        }
        return eligible ? status::no_progress : status::idle;
    } catch (const std::bad_alloc &) {
        return status::allocation_failed;
    }
}

bool llama_kv_stream_selection_valid(const llama_kv_stream_selection & selected) noexcept {
    if (!selected.context_tokens || selected.context_tokens>size_t(INT32_MAX) || selected.slots.empty() ||
            selected.slots.size()>LLAMA_MAX_SEQ || selected.history_tokens>selected.context_tokens) return false;
    size_t extent=0,history=0;
    for (size_t i=0;i<selected.slots.size();++i) {
        const auto & slot=selected.slots[i];
        if ((i && selected.slots[i-1].id>=slot.id) || slot.offset_tokens!=extent || !slot.growth_tokens ||
                slot.history_tokens>selected.context_tokens-history || slot.growth_tokens>selected.context_tokens-extent ||
                slot.history_tokens>selected.context_tokens-extent-slot.growth_tokens) return false;
        history+=slot.history_tokens;extent+=slot.history_tokens+slot.growth_tokens;
    }
    if (extent!=selected.context_tokens || history!=selected.history_tokens) return false;
    const size_t spare=selected.context_tokens-history,count=selected.slots.size();
    for (size_t i=0;i<count;++i)
        if (selected.slots[i].growth_tokens!=spare/count+(i<spare%count)) return false;
    return true;
}

llama_kv_stream_page_partition_result llama_kv_stream_partition_pages(
        const llama_kv_stream_selection & selected, uint32_t pages,
        const std::vector<llama_kv_stream_page_minimum> & minima, llama_kv_stream_page_partition & output) {
    using status=llama_kv_stream_page_partition_status;
    if (!selected.context_tokens || selected.context_tokens>size_t(INT32_MAX) || selected.slots.empty() ||
            selected.slots.size()>LLAMA_MAX_SEQ || selected.history_tokens>selected.context_tokens) return {status::invalid_selection};
    if (minima.size()!=selected.slots.size()) return {status::invalid_minimum};
    if (!llama_kv_stream_selection_valid(selected)) return {status::invalid_selection};
    try {
        auto sorted=minima;
        std::sort(sorted.begin(),sorted.end(),[](const auto & a,const auto & b) { return a.slot<b.slot; });
        uint64_t required=0;
        for (size_t i=0;i<sorted.size();++i) {
            if (sorted[i].slot!=selected.slots[i].id || !sorted[i].pages) return {status::invalid_minimum};
            required+=sorted[i].pages;
        }
        if (required>pages) return {status::insufficient_pages,required};
        llama_kv_stream_page_partition next;
        next.pages=pages;next.slots.reserve(sorted.size());
        for (const auto & slot : selected.slots)
            next.slots.push_back({slot.id,0,0,slot.offset_tokens,slot.history_tokens+slot.growth_tokens});
        std::vector<bool> fixed(sorted.size(),false);
        uint64_t remaining=pages,weight=selected.context_tokens;
        for (;;) {
            bool clamped=false;
            for (size_t i=0;i<sorted.size();++i) {
                if (fixed[i] || uint64_t(sorted[i].pages)*weight<=remaining*next.slots[i].projected_tokens) continue;
                fixed[i]=true;next.slots[i].pages=sorted[i].pages;
                remaining-=sorted[i].pages;weight-=next.slots[i].projected_tokens;
                clamped=true;break;
            }
            if (!clamped) break;
        }
        struct remainder { uint64_t value;size_t slot; };
        std::vector<remainder> remainders;
        uint64_t assigned=0;
        for (size_t i=0;i<sorted.size();++i) if (!fixed[i]) {
            const uint64_t numerator=remaining*next.slots[i].projected_tokens;
            next.slots[i].pages=uint32_t(numerator/weight);
            assigned+=next.slots[i].pages;
            remainders.push_back({numerator%weight,i});
        }
        std::sort(remainders.begin(),remainders.end(),[](const remainder & a,const remainder & b) {
            return a.value!=b.value ? a.value>b.value : a.slot<b.slot;
        });
        for (size_t i=0;i<size_t(remaining-assigned);++i) ++next.slots[remainders[i].slot].pages;
        uint64_t offset=0;
        for (auto & slot : next.slots) { slot.first_page=uint32_t(offset);offset+=slot.pages; }
        output=std::move(next);
        return {status::success,required};
    } catch (const std::bad_alloc &) { return {status::allocation_failed}; }
}

bool llama_kv_stream_request_group::admit(const std::vector<llama_kv_stream_group_request> & requests) {
    const bool joining = !members.empty();
    if ((joining && phase() != llama_kv_stream_group_phase::prefill) || !config.context_tokens || config.context_tokens>size_t(INT32_MAX) ||
            !config.maximum_slots || config.maximum_slots>LLAMA_MAX_SEQ || requests.empty() ||
            requests.size()>config.maximum_slots || epoch==UINT64_MAX || revision==UINT64_MAX) return false;
    try {
        std::vector<member> next;
        auto epochs=request_epochs;
        next.reserve(config.maximum_slots);
        for (const auto & member : members) if (!member.finished) next.push_back(member);
        if (requests.size() > config.maximum_slots - next.size()) return false;
        for (const auto & request : requests) {
            if (request.slot>=LLAMA_MAX_SEQ || !request.request || request.history_tokens>config.context_tokens ||
                    (request.slot<epochs.size() && request.request<=epochs[request.slot]) ||
                    (joining && request.needs_vision)) return false;
            for (const auto & prior : next) if (prior.request.slot==request.slot) return false;
            if (epochs.size()<=request.slot) epochs.resize(size_t(request.slot)+1,0);
            epochs[request.slot]=request.request;
            next.push_back({request,request.needs_vision ? llama_kv_stream_group_phase::vision : llama_kv_stream_group_phase::prefill,false});
        }
        std::sort(next.begin(),next.end(),[](const member & a,const member & b) {return a.request.slot<b.request.slot;});
        members=std::move(next); request_epochs=std::move(epochs); ++epoch;
        return true;
    } catch (const std::bad_alloc &) {return false;}
}

llama_kv_stream_group_phase llama_kv_stream_request_group::phase() const noexcept {
    using phase=llama_kv_stream_group_phase;
    if (members.empty()) return phase::idle;
    bool live=false,prefill=false;
    for (const auto & member : members) if (!member.finished) {
        if (member.pending==phase::vision) return phase::vision;
        prefill=prefill || member.pending==phase::prefill;
        live=true;
    }
    return prefill ? phase::prefill : live ? phase::decode : phase::retiring;
}

const llama_kv_stream_request_group::member * llama_kv_stream_request_group::find(uint32_t slot,uint64_t request) const noexcept {
    for (const auto & member : members)
        if (!member.finished && member.request.slot==slot && member.request.request==request) return &member;
    return nullptr;
}

bool llama_kv_stream_request_group::busy(uint32_t slot,uint64_t request) const noexcept {return find(slot,request)!=nullptr;}

bool llama_kv_stream_request_group::preparing(uint32_t slot, uint64_t request) const noexcept {
    const auto * member = find(slot, request);
    return member && phase() == llama_kv_stream_group_phase::prefill &&
        member->pending == llama_kv_stream_group_phase::prefill;
}

bool llama_kv_stream_request_group::next_preparation(llama_kv_stream_group_request & output) const noexcept {
    const auto pending=phase();
    if (pending!=llama_kv_stream_group_phase::vision && pending!=llama_kv_stream_group_phase::prefill) return false;
    for (const auto & member : members) if (!member.finished && member.pending==pending) {
        output=member.request; return true;
    }
    return false;
}

bool llama_kv_stream_request_group::prepared(uint32_t slot,uint64_t request,llama_kv_stream_group_phase completed,
        size_t history_tokens) noexcept {
    llama_kv_stream_group_request next;
    if (completed == llama_kv_stream_group_phase::prefill) {
        const auto * member = find(slot, request);
        if (!preparing(slot, request) || !member) return false;
        next = member->request;
    } else if (!next_preparation(next)) return false;
    if (next.slot!=slot || next.request!=request || phase()!=completed ||
            history_tokens<next.history_tokens || history_tokens>config.context_tokens ||
            (completed==llama_kv_stream_group_phase::vision && history_tokens!=next.history_tokens)) return false;
    for (auto & member : members) if (member.request.slot==slot) {
        member.request.history_tokens=history_tokens;
        member.pending=completed==llama_kv_stream_group_phase::vision ?
            llama_kv_stream_group_phase::prefill : llama_kv_stream_group_phase::decode;
        return true;
    }
    return false;
}

llama_kv_stream_selection_status llama_kv_stream_request_group::plan(llama_kv_stream_group_plan & output,
        llama_kv_stream_selection_admit admit,void * user_data) const {
    using status=llama_kv_stream_selection_status;
    if (phase()!=llama_kv_stream_group_phase::decode || revision==UINT64_MAX) return status::no_progress;
    try {
        std::vector<llama_kv_stream_selection_slot> eligible;
        for (const auto & member : members) if (!member.finished)
            eligible.push_back({member.request.slot,member.request.history_tokens,1,member.request.compatibility,true});
        llama_kv_stream_group_plan next;
        const auto result=llama_kv_stream_select(config,eligible,next.selected,admit,user_data);
        if (result!=status::success) return result;
        next.epoch=epoch; next.revision=revision+1; output=std::move(next);
        return status::success;
    } catch (const std::bad_alloc &) {return status::allocation_failed;}
}

bool llama_kv_stream_request_group::activate(llama_kv_stream_group_plan && proposal) noexcept {
    const auto & chosen=proposal.selected;
    if (phase()!=llama_kv_stream_group_phase::decode || proposal.epoch!=epoch || revision==UINT64_MAX ||
            proposal.revision!=revision+1 || chosen.context_tokens!=config.context_tokens ||
            chosen.slots.empty() || chosen.slots.size()>members.size() || chosen.history_tokens>config.context_tokens) return false;
    const size_t count=chosen.slots.size(),spare=config.context_tokens-chosen.history_tokens;
    size_t history=0,extent=0;
    uint64_t compatibility=0;
    for (size_t i=0;i<count;++i) {
        const auto & slot=chosen.slots[i];
        const member * found=nullptr;
        for (const auto & member : members) if (!member.finished && member.request.slot==slot.id) found=&member;
        if (!found || (i && chosen.slots[i-1].id>=slot.id) ||
                slot.history_tokens!=found->request.history_tokens || slot.history_tokens>config.context_tokens-history ||
                !slot.growth_tokens || slot.growth_tokens!=spare/count+(i<spare%count) || slot.offset_tokens!=extent ||
                slot.growth_tokens>config.context_tokens-extent || slot.history_tokens>config.context_tokens-extent-slot.growth_tokens ||
                (i && compatibility!=found->request.compatibility)) return false;
        compatibility=found->request.compatibility;
        history+=slot.history_tokens; extent+=slot.history_tokens+slot.growth_tokens;
    }
    if (history!=chosen.history_tokens || extent!=config.context_tokens) return false;
    active=std::move(proposal.selected); revision=proposal.revision;
    return true;
}

bool llama_kv_stream_request_group::selected(uint32_t slot,uint64_t request) const noexcept {
    if (!busy(slot,request) || phase()!=llama_kv_stream_group_phase::decode) return false;
    for (const auto & chosen : active.slots) if (chosen.id==slot) return true;
    return false;
}

bool llama_kv_stream_request_group::needs_reselection() const noexcept {
    if (phase()!=llama_kv_stream_group_phase::decode || active.slots.empty()) return true;
    for (const auto & chosen : active.slots) {
        const member * found=nullptr;
        for (const auto & member : members) if (!member.finished && member.request.slot==chosen.id) found=&member;
        if (!found || found->request.history_tokens==chosen.history_tokens+chosen.growth_tokens) return true;
    }
    return false;
}

bool llama_kv_stream_request_group::submission(uint32_t slot,uint64_t request,uint32_t maximum_proposals,
        size_t remaining_outputs,llama_kv_stream_generation_budget & output) const noexcept {
    if (needs_reselection() || !selected(slot,request)) return false;
    const auto * member=find(slot,request);
    for (const auto & chosen : active.slots) if (chosen.id==slot)
        return llama_kv_stream_generation_budget_make(chosen.history_tokens+chosen.growth_tokens-member->request.history_tokens,
            maximum_proposals,remaining_outputs,output);
    return false;
}

bool llama_kv_stream_request_group::advance(const std::vector<llama_kv_stream_group_progress> & progress) noexcept {
    if (phase()!=llama_kv_stream_group_phase::decode || progress.empty()) return false;
    for (size_t i=0;i<progress.size();++i) {
        const auto & update=progress[i];
        const auto * member=find(update.slot,update.request);
        if (!member || !selected(update.slot,update.request) || update.history_tokens<member->request.history_tokens) return false;
        for (size_t j=0;j<i;++j) if (progress[j].slot==update.slot) return false;
        for (const auto & chosen : active.slots)
            if (chosen.id==update.slot && update.history_tokens>chosen.history_tokens+chosen.growth_tokens) return false;
    }
    for (const auto & update : progress) for (auto & member : members)
        if (member.request.slot==update.slot) member.request.history_tokens=update.history_tokens;
    return true;
}

bool llama_kv_stream_request_group::finish(uint32_t slot,uint64_t request) noexcept {
    for (auto & member : members) if (!member.finished && member.request.slot==slot && member.request.request==request) {
        member.finished=true; return true;
    }
    return false;
}

bool llama_kv_stream_request_group::release() noexcept {
    if (phase()!=llama_kv_stream_group_phase::retiring) return false;
    active.slots.clear(); members.clear();
    return true;
}
