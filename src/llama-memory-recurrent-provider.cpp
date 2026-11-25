#include "llama-memory-recurrent-provider.h"

#include <algorithm>
#include <cmath>

namespace {
bool add(size_t a, size_t b, size_t & out) {
    if (b > std::numeric_limits<size_t>::max() - a) return false;
    out = a + b;
    return true;
}

bool mul(size_t a, size_t b, size_t & out) {
    if (b && a > std::numeric_limits<size_t>::max() / b) return false;
    out = a * b;
    return true;
}

bool aligned(size_t value, size_t alignment, size_t & out) {
    if (!add(value, alignment - 1, out)) return false;
    out &= ~(alignment - 1);
    return true;
}
}

bool llama_recurrent_provider_describe(const llama_recurrent_provider_request & r,
        llama_recurrent_representation representation, llama_recurrent_provider_capabilities caps,
        llama_recurrent_provider_requirements & output) noexcept {
    if (!r.selected || r.selected > r.configured || r.depth > 7 ||
            !r.alignment || (r.alignment & (r.alignment - 1))) return false;
    const bool compact = representation == llama_recurrent_representation::gdn_fp32;
    const bool host = representation == llama_recurrent_representation::dense_host;
    if (!compact && !host && representation != llama_recurrent_representation::dense_device) return false;
    if (compact && (!r.depth || !caps.shared_history || !caps.fp32_capture || !caps.exact_replay)) return false;
    const size_t rows = size_t(r.depth) + 1;
    llama_recurrent_provider_requirements next;
    for (const auto & layer : r.layers) {
        if (!layer.width || !layer.value_heads || !layer.key_heads || !layer.history || !layer.conv_channels ||
                layer.value_heads % layer.key_heads) return false;
        size_t matrix, history, frame, bytes;
        if (!mul(layer.width, layer.width, matrix) || !mul(matrix, layer.value_heads, matrix) ||
                !mul(matrix, sizeof(float), matrix) || !mul(matrix, r.selected, matrix) ||
                !mul(layer.conv_channels, sizeof(float), frame) || !mul(frame, r.selected, frame) ||
                !mul(frame, layer.history, history)) return false;
        if (compact) {
            size_t key, correction, record, timeline, images;
            if (!mul(layer.width, layer.key_heads, key) || !mul(layer.width, layer.value_heads, correction) ||
                    !add(key, correction, record) || !add(record, layer.value_heads, record) ||
                    !mul(record, sizeof(float), record) || !mul(record, r.selected, record) ||
                    !mul(record, rows, record) || !add(layer.history, rows, timeline) ||
                    !mul(timeline, frame, timeline) || !mul(matrix, 2, images) ||
                    !aligned(images, r.alignment, images) || !aligned(record, r.alignment, record) ||
                    !aligned(timeline, r.alignment, timeline) || !add(images, record, bytes) ||
                    !add(bytes, timeline, bytes) || !mul(history, 2, history) ||
                    !aligned(history, r.alignment, history) || !add(bytes, history, bytes)) return false;
        } else {
            if (!aligned(matrix, r.alignment, matrix) || !aligned(history, r.alignment, history) ||
                    !add(matrix, history, bytes)) return false;
            if (host) {
                size_t snapshots, staging;
                if (!mul(bytes, r.depth, snapshots) || !add(next.host_persistent, snapshots, next.host_persistent) ||
                        !mul(snapshots, 2, staging)) return false;
                // Publication stages are shared across layers, with two generations live.
                next.device_staging = std::max(next.device_staging, staging);
            } else if (!mul(bytes, rows, bytes)) return false;
        }
        if (!add(next.device_persistent, bytes, next.device_persistent)) return false;
    }
    if (!add(next.device_persistent, next.device_staging, next.device_peak) ||
            !add(next.device_peak, r.graph_workspace, next.device_peak) || next.device_peak > r.device_budget) return false;
    output = next;
    return true;
}

bool llama_recurrent_placement_choose(const llama_recurrent_placement_request & r,
        llama_recurrent_placement_state & state, llama_recurrent_placement_choice & output) noexcept {
    if (!r.safe_boundary || !r.generation || r.generation<state.last_generation || r.plans.empty() ||
            r.reserve>r.budget || !std::isfinite(r.h2d_bytes_per_ms) || r.h2d_bytes_per_ms<=0 ||
            !std::isfinite(r.d2h_bytes_per_ms) || r.d2h_bytes_per_ms<=0 ||
            !std::isfinite(r.hysteresis) || r.hysteresis<0 || r.hysteresis>=1) return false;
    std::array<bool,3> seen{};
    bool comparable=true;
    double common_factor=0;size_t observed=0;
    for (const auto & plan : r.plans) {
        const auto index=size_t(plan.representation);
        if (index>=seen.size()) return false;
        if (plan.peak_bytes>r.budget-r.reserve) continue;
        if (!state.samples[index] || state.sample_workloads[index]!=r.workload) comparable=false;
        else {
            double sum=0;
            for (uint32_t i=0;i<state.samples[index];++i) sum+=state.ratios[index][i];
            common_factor+=sum/state.samples[index];++observed;
        }
    }
    common_factor=observed ? common_factor/observed : 1;
    llama_recurrent_placement_choice best, current;
    best.adjusted_ms=std::numeric_limits<double>::infinity();current.adjusted_ms=best.adjusted_ms;
    for (const auto & plan : r.plans) {
        const auto index=size_t(plan.representation);
        if (index>=seen.size() || seen[index] || !std::isfinite(plan.compute_ms) || plan.compute_ms<0 ||
                !std::isfinite(plan.restore_ms) || plan.restore_ms<0) return false;
        seen[index]=true;
        if (plan.peak_bytes>r.budget-r.reserve) continue;
        const double copies=std::max((double(plan.h2d_bytes)+double(plan.kv_stream_bytes))/r.h2d_bytes_per_ms,
                                     double(plan.d2h_bytes)/r.d2h_bytes_per_ms);
        const double predicted=plan.compute_ms+plan.restore_ms+copies;
        if (!std::isfinite(predicted)) return false;
        double factor=common_factor;
        if (comparable && state.samples[index] && state.sample_workloads[index]==r.workload) {
            factor=0;
            for (uint32_t i=0;i<state.samples[index];++i) factor+=state.ratios[index][i];
            factor/=state.samples[index];
        }
        const llama_recurrent_placement_choice candidate{plan.representation,predicted,predicted*factor,plan.peak_bytes};
        if (candidate.adjusted_ms<best.adjusted_ms) best=candidate;
        if (state.chosen && plan.representation==state.current) current=candidate;
    }
    if (!std::isfinite(best.adjusted_ms)) return false;
    if (state.chosen && std::isfinite(current.adjusted_ms) &&
            ((state.workload==r.workload && r.generation-state.last_switch_generation<r.cooldown) ||
             best.adjusted_ms>=current.adjusted_ms*(1-r.hysteresis))) best=current;
    auto next=state;
    if (!next.chosen || next.current!=best.representation) next.last_switch_generation=r.generation;
    next.chosen=true;next.current=best.representation;next.workload=r.workload;next.last_generation=r.generation;
    state=next;output=best;return true;
}

bool llama_recurrent_placement_observe(llama_recurrent_placement_state & state,
        llama_recurrent_representation representation, uint64_t workload,
        double predicted_ms,double observed_ms) noexcept {
    const auto index=size_t(representation);
    if (index>=state.samples.size() || !std::isfinite(predicted_ms) || predicted_ms<=0 ||
            !std::isfinite(observed_ms) || observed_ms<=0) return false;
    if (state.sample_workloads[index]!=workload) {
        state.samples[index]=state.next_sample[index]=0;state.sample_workloads[index]=workload;
    }
    state.ratios[index][state.next_sample[index]]=std::max(.25,std::min(4.0,observed_ms/predicted_ms));
    state.samples[index]=std::min(8u,state.samples[index]+1);
    state.next_sample[index]=(state.next_sample[index]+1)%8;
    return true;
}
