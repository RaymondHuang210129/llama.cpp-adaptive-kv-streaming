#pragma once

#include "ggml-backend.h"
#include <array>
#include <algorithm>
#include <cstdint>
#include <cstddef>

// Prepare under exclusive backend ownership; it may initialize native kernel metadata.
// Subsequent queries describe native scratch without reading payloads or allocating device storage.
enum class ggml_backend_operator_workspace_status { success, unsupported, unprepared, invalid, failed };
struct ggml_backend_operator_workspace {
    std::array<size_t,8> requests{}, offsets{};
    size_t count = 0, bytes = 0, alignment = 128, library_bytes = 0;
};
struct ggml_backend_operator_stream_workspace {
    size_t scratch_offset = 0, scratch_bytes = 0, library_offset = 0, library_bytes = 0;
};
struct ggml_backend_operator_phase_workspace {
    std::array<ggml_backend_operator_stream_workspace,16> streams{};
    size_t count = 0, bytes = 0, alignment = 256;
};
using ggml_backend_operator_workspace_prepare_t = ggml_backend_operator_workspace_status (*)(ggml_backend_t);
using ggml_backend_operator_workspace_query_t = ggml_backend_operator_workspace_status (*)(
    ggml_backend_t,const ggml_tensor *,ggml_backend_operator_workspace *);

static inline bool ggml_backend_operator_workspace_align(size_t bytes,size_t alignment,size_t & output) {
    if (!alignment || (alignment&(alignment-1)) || bytes>SIZE_MAX-(alignment-1)) return false;
    output=(bytes+alignment-1)&~(alignment-1);
    return true;
}
static inline bool ggml_backend_operator_workspace_add(ggml_backend_operator_workspace & plan,size_t bytes) {
    if (plan.count>plan.requests.size()) return false;
    size_t offset,rounded;
    if (!ggml_backend_operator_workspace_align(plan.bytes,plan.alignment,offset) ||
            !ggml_backend_operator_workspace_align(bytes,plan.alignment,rounded) || rounded>SIZE_MAX-offset) return false;
    if (!bytes) return true;
    if (plan.count>=plan.requests.size()) return false;
    plan.requests[plan.count]=bytes;plan.offsets[plan.count++]=offset;plan.bytes=offset+rounded;
    return true;
}
static inline bool ggml_backend_operator_workspace_valid(const ggml_backend_operator_workspace & plan) {
    if (plan.count>plan.requests.size()) return false;
    ggml_backend_operator_workspace rebuilt;rebuilt.alignment=plan.alignment;
    for (size_t i=0;i<plan.count;++i) {
        if (!plan.requests[i] || !ggml_backend_operator_workspace_add(rebuilt,plan.requests[i]) ||
                rebuilt.offsets[i]!=plan.offsets[i]) return false;
    }
    size_t checked;
    return ggml_backend_operator_workspace_align(0,plan.alignment,checked) && rebuilt.bytes==plan.bytes;
}
// Concurrent streams add their peaks; sequential operations reuse a stream's scratch.
static inline bool ggml_backend_operator_workspace_group(const ggml_backend_operator_workspace * ops,
        const uint32_t * streams,size_t count,size_t n_streams,ggml_backend_operator_phase_workspace & output) {
    ggml_backend_operator_phase_workspace next;
    if (!n_streams || n_streams>next.streams.size() || (count && (!ops || !streams))) return false;
    next.count=n_streams;
    for (size_t i=0;i<count;++i) {
        if (streams[i]>=n_streams || !ggml_backend_operator_workspace_valid(ops[i]) || ops[i].alignment>next.alignment) return false;
        auto & stream=next.streams[streams[i]];
        stream.scratch_bytes=std::max(stream.scratch_bytes,ops[i].bytes);
        stream.library_bytes=std::max(stream.library_bytes,ops[i].library_bytes);
    }
    for (size_t i=0;i<n_streams;++i) {
        auto & stream=next.streams[i];
        if (!ggml_backend_operator_workspace_align(next.bytes,next.alignment,stream.library_offset) ||
                stream.library_bytes>SIZE_MAX-stream.library_offset) return false;
        const size_t after_library=stream.library_offset+stream.library_bytes;
        if (!ggml_backend_operator_workspace_align(after_library,next.alignment,stream.scratch_offset) ||
                stream.scratch_bytes>SIZE_MAX-stream.scratch_offset) return false;
        next.bytes=stream.scratch_offset+stream.scratch_bytes;
    }
    if (!ggml_backend_operator_workspace_align(next.bytes,next.alignment,next.bytes)) return false;
    output=next;return true;
}
