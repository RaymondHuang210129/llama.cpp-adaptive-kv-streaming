#include "llama-kv-stream-selection.h"

#include <algorithm>
#include <new>
#include <utility>

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
