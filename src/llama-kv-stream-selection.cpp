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

    void visit(size_t first, size_t history) {
        if (!best.slots.empty() && history > best.history_tokens) return;
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
        for (size_t i = first; i <= slots.size() - needed; ++i) {
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
        for (size_t count = std::min(config.maximum_slots, sorted.size()); count > 0; --count) {
            search.count = count;
            search.visit(0, 0);
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

llama_kv_stream_page_partition_result llama_kv_stream_partition_pages(
        const llama_kv_stream_selection & selected, uint32_t pages,
        const std::vector<llama_kv_stream_page_minimum> & minima, llama_kv_stream_page_partition & output) {
    using status=llama_kv_stream_page_partition_status;
    if (!selected.context_tokens || selected.context_tokens>size_t(INT32_MAX) || selected.slots.empty() ||
            selected.slots.size()>LLAMA_MAX_SEQ || selected.history_tokens>selected.context_tokens) return {status::invalid_selection};
    if (minima.size()!=selected.slots.size()) return {status::invalid_minimum};
    try {
        size_t extent=0,history=0;
        for (size_t i=0;i<selected.slots.size();++i) {
            const auto & slot=selected.slots[i];
            if ((i && selected.slots[i-1].id>=slot.id) || slot.offset_tokens!=extent || !slot.growth_tokens ||
                    slot.history_tokens>selected.context_tokens-history || slot.growth_tokens>selected.context_tokens-extent ||
                    slot.history_tokens>selected.context_tokens-extent-slot.growth_tokens) return {status::invalid_selection};
            history+=slot.history_tokens;extent+=slot.history_tokens+slot.growth_tokens;
        }
        if (extent!=selected.context_tokens || history!=selected.history_tokens) return {status::invalid_selection};
        const size_t spare=selected.context_tokens-history,count=selected.slots.size();
        for (size_t i=0;i<count;++i) {
            if (selected.slots[i].growth_tokens!=spare/count+(i<spare%count)) return {status::invalid_selection};
        }
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
