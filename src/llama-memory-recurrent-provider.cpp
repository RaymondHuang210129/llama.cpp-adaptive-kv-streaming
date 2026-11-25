#include "llama-memory-recurrent-provider.h"

#include <algorithm>

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
                    !add(bytes, timeline, bytes)) return false;
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
