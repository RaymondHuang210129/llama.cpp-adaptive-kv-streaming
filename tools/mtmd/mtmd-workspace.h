#pragma once

#include "mtmd.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "../../ggml/src/ggml-backend-memory.h"

#include <memory>
#include <string>
#include <vector>

class llama_context_memory;

struct mtmd_workspace_details {
    size_t graph_bytes = 0, operator_bytes = 0, library_bytes = 0, padding_bytes = 0, total_bytes = 0;
};

// Internal serial scheduler adapter. The scheduler and backends outlive this owner.
class MTMD_API mtmd_compute_workspace {
public:
    mtmd_compute_workspace(ggml_backend_sched_t sched, std::vector<ggml_backend_t> backends, size_t graph_capacity);
    ~mtmd_compute_workspace();
    mtmd_compute_workspace(const mtmd_compute_workspace &) = delete;
    mtmd_compute_workspace & operator=(const mtmd_compute_workspace &) = delete;

    // Measure a detached graph without allocating its compute buffers.
    bool measure(ggml_cgraph * graph, std::vector<ggml_backend_memory_workspace_group> & output);
    // Borrow one committed lease per measured canonical buffer-type group.
    bool attach(const std::vector<ggml_backend_memory_lease_t> & leases);
    // Borrow measured scratch, including reclaimed KV storage while the target is explicitly suspended.
    bool borrow(llama_context_memory & parent);
    // Check the actual graph against its grants before allocating tensor addresses.
    bool alloc_graph(ggml_cgraph * graph);
    // Retain execution dependencies until drain or release completes scheduler work.
    ggml_status compute_async(ggml_cgraph * graph);
    bool drain();
    // Retire captures before the caller overwrites graph metadata; keep workspace leases attached.
    bool retire_graph();
    // Drain, retire native captures, and detach leases before storage can be reused.
    bool release();
    bool details(ggml_backend_buffer_type_t type,mtmd_workspace_details & output) const noexcept;
    bool supported() const noexcept;
    bool ready() const noexcept;

private:
    struct implementation;
    std::unique_ptr<implementation> impl;
};

// Internal vision-only seam. The optional compute type selects the caller's shared arena type.
// Measurement retires previous workspace grants; image embeddings keep their host ownership.
MTMD_API bool mtmd_batch_measure_compute_workspace(mtmd_batch * batch,
    std::vector<ggml_backend_memory_workspace_group> & output,
    ggml_backend_buffer_type_t compute_type = nullptr);
MTMD_API bool mtmd_attach_compute_workspace(mtmd_context * ctx,
    const std::vector<ggml_backend_memory_lease_t> & leases);
MTMD_API bool mtmd_borrow_compute_workspace(mtmd_context * ctx, llama_context_memory & parent);
MTMD_API bool mtmd_release_compute_workspace(mtmd_context * ctx);

struct mtmd_vision_phase_requirements {
    size_t weight_bytes = 0, device_compute_bytes = 0, host_compute_bytes = 0;
    std::vector<ggml_backend_memory_workspace_group> groups;
    mtmd_workspace_details workspace;
};
// Plan from unbound weight descriptors without uploading or reading the parent's device bytes.
MTMD_API bool mtmd_batch_measure_vision_phase(mtmd_batch * batch,ggml_backend_buffer_t parent,
    mtmd_vision_phase_requirements & output);
// Describe the current batch before testing the complete shared-parent quota.
MTMD_API bool mtmd_batch_quote_vision_phase(mtmd_batch * batch,ggml_backend_buffer_t parent,
    mtmd_vision_phase_requirements & output);
MTMD_API mtmd_context * mtmd_batch_context(mtmd_batch * batch) noexcept;

struct mtmd_vision_geometry {
    std::string family;
    int patch_size = 0, merge = 0, min_tokens = 0, max_tokens = 0;
};
struct mtmd_vision_shape {
    int width = 0, height = 0, input_width = 0, input_height = 0, budget = 0;
    size_t tokens = 0;
};
// Planning uses native preprocessing and shape-only graphs, without reading parent bytes.
MTMD_API bool mtmd_get_vision_geometry(mtmd_context * ctx, mtmd_vision_geometry & output);
MTMD_API bool mtmd_vision_shape_for_budget(const mtmd_vision_geometry & geometry, int budget,
    int ratio_width, int ratio_height, mtmd_vision_shape & output);
MTMD_API size_t mtmd_vision_weight_bytes(mtmd_context * ctx, ggml_backend_buffer_type_t type);
MTMD_API bool mtmd_measure_vision_shape(mtmd_context * ctx, ggml_backend_buffer_t parent,
    const mtmd_vision_shape & shape, mtmd_vision_phase_requirements & output);
MTMD_API bool mtmd_batch_vision_shape(const mtmd_batch * batch, mtmd_vision_shape & output);
// Borrowed primary backend; the vision context retains its ownership.
MTMD_API ggml_backend_t mtmd_vision_backend(mtmd_context * ctx);
