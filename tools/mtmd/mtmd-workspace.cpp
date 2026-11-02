#include "mtmd-workspace.h"
#include "../../src/llama-memory-workspace.h"
#include "../../src/llama-memory-executor.h"
#include "../../src/llama-context-memory.h"
#include "../../ggml/src/ggml-cuda-graph.h"
#include "../../ggml/src/ggml-impl.h"
#include "../../ggml/src/ggml-backend-operator-workspace.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>
#include <unordered_set>

namespace {
struct workspace_cache {
    ggml_backend_t backend;
    ggml_backend_cuda_graph_release_all_t release = nullptr;
};

using operator_status=ggml_backend_operator_workspace_status;
using workspace_planner=std::unique_ptr<ggml_backend_memory_planner,decltype(&ggml_backend_memory_planner_free)>;
using workspace_arena=std::unique_ptr<ggml_backend_memory_arena,decltype(&ggml_backend_memory_arena_free)>;
using workspace_lease=std::unique_ptr<ggml_backend_memory_lease,decltype(&ggml_backend_memory_lease_free)>;
struct workspace_operator {
    ggml_backend_operator_workspace_graph_query_t query=nullptr;
    ggml_backend_operator_workspace_bind_t bind=nullptr;
    ggml_backend_operator_workspace_release_t release=nullptr;
};
struct workspace_bank {
    workspace_lease scratch{nullptr,ggml_backend_memory_lease_free};
    ggml_backend_operator_phase_workspace plan;
    bool bound=false;
};
struct workspace_requirements {
    std::vector<ggml_backend_memory_workspace_group> graphs,groups;
    std::vector<ggml_backend_operator_phase_workspace> operators;
    std::vector<mtmd_workspace_details> details;
};
static bool workspace_add(size_t & target,size_t bytes) {
    if (bytes>SIZE_MAX-target) return false;
    target+=bytes;return true;
}

struct workspace_executable : llama_memory_executable {
    std::vector<workspace_cache> caches;
    explicit workspace_executable(std::vector<workspace_cache> caches) : caches(std::move(caches)) {}
    ~workspace_executable() override {
        for (const auto & cache : caches) if (cache.release) cache.release(cache.backend);
    }
};

struct workspace_gate {
    bool & busy;
    explicit workspace_gate(bool & busy) : busy(busy) { busy = true; }
    ~workspace_gate() { busy = false; }
};

// Scheduler measurement rewrites split inputs and may optimize tensor metadata.
struct workspace_graph_restore {
    ggml_cgraph * graph;
    std::vector<ggml_tensor *> nodes,leafs;
    std::vector<std::pair<ggml_tensor *,ggml_tensor>> tensors;
    explicit workspace_graph_restore(ggml_cgraph * graph) : graph(graph),
        nodes(graph->nodes,graph->nodes+graph->n_nodes),leafs(graph->leafs,graph->leafs+graph->n_leafs) {
        std::unordered_set<ggml_tensor *> seen;
        auto pending=nodes;
        pending.insert(pending.end(),leafs.begin(),leafs.end());
        while (!pending.empty()) {
            auto * tensor=pending.back(); pending.pop_back();
            if (!tensor || !seen.insert(tensor).second) continue;
            tensors.emplace_back(tensor,*tensor);
            for (auto * src : tensor->src) pending.push_back(src);
            pending.push_back(tensor->view_src);
        }
    }
    ~workspace_graph_restore() {
        for (const auto & tensor : tensors) *tensor.first=tensor.second;
        std::copy(nodes.begin(),nodes.end(),graph->nodes);
        std::copy(leafs.begin(),leafs.end(),graph->leafs);
        graph->n_nodes=int(nodes.size());
        graph->n_leafs=int(leafs.size());
    }
};
}

struct mtmd_compute_workspace::implementation : llama_memory_executor_backend {
    ggml_backend_sched_t sched;
    std::vector<ggml_backend_t> backends;
    size_t graph_capacity;
    ggml_backend_sched_ptr measurement;
    std::vector<workspace_cache> caches;
    std::vector<workspace_operator> operators;
    std::vector<workspace_bank> banks;
    std::vector<workspace_arena> bank_arenas;
    std::vector<ggml_backend_buffer_type_t> types;
    std::vector<mtmd_workspace_details> components;
    std::vector<ggml_backend_memory_workspace_group> measured,granted;
    std::unique_ptr<llama_memory_workspace> consumer;
    std::unique_ptr<llama_context_memory> borrowed;
    llama_memory_executor executor;
    llama_memory_execution pending;
    ggml_cgraph * graph = nullptr;
    uint64_t revision = 0;
    bool valid = false, busy = false, accepting = true;

    implementation(ggml_backend_sched_t sched,std::vector<ggml_backend_t> backends,size_t graph_capacity) :
        sched(sched),backends(std::move(backends)),graph_capacity(graph_capacity) {
        if (!sched || !graph_capacity || this->backends.empty() ||
                this->backends.size() != size_t(ggml_backend_sched_get_n_backends(sched))) return;
        for (size_t i=0;i<this->backends.size();++i) {
            auto * backend=this->backends[i];
            if (!backend || ggml_backend_sched_get_backend(sched,int(i)) != backend) return;
            auto * dev=ggml_backend_get_device(backend);
            auto * reg=dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
            if (!reg) return;
            workspace_cache cache{backend};
            cache.release=reinterpret_cast<ggml_backend_cuda_graph_release_all_t>(
                ggml_backend_reg_get_proc_address(reg,"ggml_backend_cuda_graph_release_all"));
            if (!cache.release && std::strcmp(ggml_backend_reg_name(reg),"CPU")) return;
            caches.push_back(cache);
            auto prepare=reinterpret_cast<ggml_backend_operator_workspace_prepare_t>(
                ggml_backend_reg_get_proc_address(reg,"ggml_backend_operator_workspace_prepare"));
            workspace_operator op;
            op.query=reinterpret_cast<ggml_backend_operator_workspace_graph_query_t>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_operator_workspace_graph_query"));
            op.bind=reinterpret_cast<ggml_backend_operator_workspace_bind_t>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_operator_workspace_bind"));
            op.release=reinterpret_cast<ggml_backend_operator_workspace_release_t>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_operator_workspace_release"));
            if ((prepare || op.query || op.bind || op.release) &&
                    (!prepare || !op.query || !op.bind || !op.release || prepare(backend)!=operator_status::success)) return;
            operators.push_back(op);
        }
        for (auto * backend : this->backends) types.push_back(ggml_backend_sched_get_buffer_type(sched,backend));
        measurement.reset(ggml_backend_sched_new(this->backends.data(),types.data(),int(types.size()),graph_capacity,false,true));
        valid=bool(measurement);
    }

    bool drain() override {
        if (borrowed) borrowed->synchronize();
        ggml_backend_sched_synchronize(sched);
        pending.reset();
        return true;
    }

    bool invalidate() {
        accepting=false;
        const bool bound=std::any_of(banks.begin(),banks.end(),[](const workspace_bank & bank) { return bank.bound; });
        if (bound && !drain()) return false;
        for (size_t i=0;i<banks.size();++i) if (banks[i].bound) {
            if (operators[i].release(backends[i])!=operator_status::success) return false;
            banks[i].bound=false;
        }
        banks.clear();bank_arenas.clear();
        if (borrowed) { graph=nullptr;accepting=borrowed->retire_graph();return accepting; }
        const auto result=executor.retire(*this);
        if (result.status != llama_memory_executor_status::retired &&
                result.status != llama_memory_executor_status::unchanged) return false;
        if (result.status == llama_memory_executor_status::unchanged) {
            drain();
            for (const auto & cache : caches) if (cache.release) cache.release(cache.backend);
        }
        graph=nullptr;accepting=true;
        return true;
    }

    bool close() {
        if (!invalidate()) return false;
        borrowed.reset();
        if (consumer && !consumer->close()) return false;
        consumer.reset();
        granted.clear();
        return true;
    }

    bool capture() {
        if (!consumer || !consumer->ready() || revision == UINT64_MAX) return false;
        std::unique_ptr<llama_memory_executable> executable=std::make_unique<workspace_executable>(caches);
        if (!executor.capture(executable,consumer->leases(),revision+1)) return false;
        ++revision;
        return true;
    }

    bool requirements(ggml_cgraph * candidate,workspace_requirements & output) {
        if (!candidate || ggml_graph_n_nodes(candidate)<=0 ||
                size_t(ggml_graph_n_nodes(candidate))>graph_capacity ||
                size_t(candidate->n_leafs)>graph_capacity-size_t(ggml_graph_n_nodes(candidate))) return false;
        workspace_requirements next;
        next.operators.resize(backends.size());
        std::vector<size_t> sizes(backends.size());
        workspace_graph_restore restore(candidate);
        ggml_backend_sched_reset(measurement.get());
        ggml_backend_sched_reserve_size(measurement.get(),candidate,sizes.data());
        next.graphs.resize(backends.size());size_t count=next.graphs.size();
        if (!ggml_backend_memory_plan_workspace_groups(types.data(),sizes.data(),1,types.size(),next.graphs.data(),&count)) return false;
        next.graphs.resize(count);next.groups=next.graphs;
        for (size_t i=0;i<backends.size();++i) if (operators[i].query) {
            std::vector<ggml_tensor *> nodes;
            for (int n=0;n<ggml_graph_n_nodes(candidate);++n) {
                auto * node=ggml_graph_node(candidate,n);
                if (ggml_backend_sched_get_tensor_backend(measurement.get(),node)==backends[i]) nodes.push_back(node);
            }
            if (nodes.empty()) continue;
            auto selected=*candidate;selected.nodes=nodes.data();selected.n_nodes=int(nodes.size());
            if (operators[i].query(backends[i],&selected,&next.operators[i])!=operator_status::success ||
                    !ggml_backend_operator_workspace_phase_valid(next.operators[i])) return false;
        }
        for (size_t g=0;g<next.groups.size();++g) {
            auto & group=next.groups[g];mtmd_workspace_details detail;detail.graph_bytes=group.size;
            size_t maximum_alignment=group.alignment;
            for (size_t i=0;i<operators.size();++i) if (types[i]==group.buft && next.operators[i].count) {
                const auto & plan=next.operators[i];
                if (plan.bytes) maximum_alignment=std::max(maximum_alignment,plan.alignment);
                for (size_t s=0;s<plan.count;++s) {
                    if (!workspace_add(detail.operator_bytes,plan.streams[s].scratch_bytes) ||
                            !workspace_add(detail.library_bytes,plan.streams[s].library_bytes)) return false;
                }
            }
            if (!group.alignment || maximum_alignment%group.alignment) return false;
            size_t maximum=0;
            for (size_t bias=0;bias<maximum_alignment;bias+=group.alignment) {
                workspace_planner planner(ggml_backend_memory_planner_new(SIZE_MAX,group.alignment),ggml_backend_memory_planner_free);
                if (!planner || !ggml_backend_memory_planner_begin(planner.get(),0) ||
                        (bias && !ggml_backend_memory_planner_reserve(planner.get(),1,bias,group.alignment,0,nullptr)) ||
                        !ggml_backend_memory_planner_reserve(planner.get(),2,group.size,group.alignment,0,nullptr)) return false;
                for (size_t i=0;i<operators.size();++i) if (types[i]==group.buft && next.operators[i].bytes) {
                    const auto & plan=next.operators[i];
                    if (!ggml_backend_memory_planner_reserve(planner.get(),i+3,plan.bytes,plan.alignment,0,nullptr)) return false;
                }
                if (!ggml_backend_memory_planner_commit(planner.get())) return false;
                size_t needed=ggml_backend_memory_planner_high_water(planner.get())-bias;
                if (!ggml_backend_operator_workspace_align(needed,group.alignment,needed)) return false;
                maximum=std::max(maximum,needed);
            }
            detail.total_bytes=maximum;size_t payload=detail.graph_bytes;
            if (!workspace_add(payload,detail.operator_bytes) || !workspace_add(payload,detail.library_bytes) || payload>maximum) return false;
            detail.padding_bytes=maximum-payload;group.size=maximum;next.details.push_back(detail);
        }
        output=std::move(next);return true;
    }

    const std::vector<ggml_backend_memory_lease_t> & root_leases() const {
        return borrowed ? borrowed->workspace_leases() : consumer->leases();
    }
    ggml_backend_memory_lease_t root_for(ggml_backend_buffer_type_t type) const {
        ggml_backend_memory_lease_t result=nullptr;
        for (auto * lease : root_leases()) if (ggml_backend_buffer_get_type(ggml_backend_memory_lease_buffer(lease))==type) {
            if (result) return nullptr;
            result=lease;
        }
        return result;
    }
    bool limit_graphs(const workspace_requirements & layout) {
        for (const auto & group : layout.graphs) {
            bool managed=false;
            for (size_t i=0;i<operators.size();++i) managed|=types[i]==group.buft && layout.operators[i].count;
            if (managed && !ggml_backend_sched_set_memory_lease_limit(sched,backends[group.first_slot],group.size)) return false;
        }
        return true;
    }
    bool bind_operators(const workspace_requirements & layout) {
        accepting=false;banks.resize(backends.size());
        for (const auto & group : layout.graphs) {
            bool managed=false;
            for (size_t i=0;i<operators.size();++i) managed|=types[i]==group.buft && layout.operators[i].count;
            if (!managed) continue;
            auto * root=root_for(group.buft);auto * buffer=ggml_backend_memory_lease_buffer(root);
            ggml_backend_memory_region region{};
            if (!buffer || !ggml_backend_memory_lease_get_region(root,&region) || region.flags) return false;
            const auto base=reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(buffer));
            if (!base || region.size>UINTPTR_MAX-base || group.size>region.size) return false;
            workspace_arena arena(nullptr,ggml_backend_memory_arena_free);
            size_t cursor=group.size;
            for (size_t i=0;i<operators.size();++i) if (types[i]==group.buft && layout.operators[i].bytes) {
                const auto & plan=layout.operators[i];size_t aligned;
                if (!ggml_backend_operator_workspace_align(base+cursor,std::max(group.alignment,plan.alignment),aligned) ||
                        aligned<base || aligned-base>region.size || plan.bytes>region.size-(aligned-base)) return false;
                cursor=aligned-base;
                if (!arena) {
                    arena.reset(ggml_backend_memory_arena_new_from_buffer(buffer));
                    if (!arena || !ggml_backend_memory_arena_begin(arena.get(),0)) return false;
                }
                if (!ggml_backend_memory_arena_reserve_at(arena.get(),i+1,cursor,plan.bytes,group.alignment,0,nullptr)) return false;
                cursor+=plan.bytes;
            }
            if (arena && !ggml_backend_memory_arena_commit(arena.get())) return false;
            for (size_t i=0;i<operators.size();++i) if (types[i]==group.buft && layout.operators[i].count) {
                auto & bank=banks[i];bank.plan=layout.operators[i];
                if (bank.plan.bytes) {
                    bank.scratch.reset(ggml_backend_memory_arena_acquire(arena.get(),i+1));
                    if (!bank.scratch) return false;
                }
                ggml_backend_operator_workspace_binding binding;
                binding.parent=ggml_backend_memory_lease_parent(root);binding.root=root;binding.root_id=region.id;
                binding.root_generation=ggml_backend_memory_lease_generation(root);binding.scratch=bank.scratch.get();binding.layout=bank.plan;
                if (binding.scratch) {
                    binding.scratch_id=i+1;binding.scratch_generation=ggml_backend_memory_lease_generation(binding.scratch);
                }
                const auto result=operators[i].bind(backends[i],&binding);
                bank.bound=result==operator_status::success || result==operator_status::failed;
                if (result!=operator_status::success) return false;
            }
            bank_arenas.push_back(std::move(arena));
        }
        accepting=true;return true;
    }
    bool pre_compute(ggml_backend_t backend,ggml_cgraph * split) {
        auto found=std::find(backends.begin(),backends.end(),backend);
        if (found==backends.end()) return false;
        const size_t i=size_t(found-backends.begin());
        if (!operators[i].query) return true;
        if (i>=banks.size() || !banks[i].bound) return false;
        ggml_backend_operator_phase_workspace need;
        if (operators[i].query(backend,split,&need)!=operator_status::success || need.count>banks[i].plan.count) return false;
        for (size_t s=0;s<need.count;++s) {
            if (need.streams[s].scratch_bytes>banks[i].plan.streams[s].scratch_bytes ||
                    need.streams[s].library_bytes>banks[i].plan.streams[s].library_bytes) return false;
        }
        return true;
    }
};

mtmd_compute_workspace::mtmd_compute_workspace(ggml_backend_sched_t sched,std::vector<ggml_backend_t> backends,size_t graph_capacity) :
    impl(new implementation(sched,std::move(backends),graph_capacity)) {}

mtmd_compute_workspace::~mtmd_compute_workspace() { GGML_ASSERT(release()); }

bool mtmd_compute_workspace::measure(ggml_cgraph * graph,std::vector<ggml_backend_memory_workspace_group> & output) {
    auto & s=*impl;
    if (!s.valid || s.busy || s.consumer || s.borrowed || !graph) return false;
    workspace_gate gate(s.busy);
    try {
        if (!s.invalidate()) return false;
        workspace_requirements next;
        if (!s.requirements(graph,next) || next.groups.empty()) return false;
        auto result=next.groups;
        s.measured=std::move(next.groups);s.components=std::move(next.details);
        output=std::move(result);
        return true;
    } catch (...) { return false; }
}

bool mtmd_compute_workspace::attach(const std::vector<ggml_backend_memory_lease_t> & leases) {
    auto & s=*impl;
    if (!s.valid || s.busy || s.measured.empty() || leases.size() != s.measured.size()) return false;
    workspace_gate gate(s.busy);
    try {
        llama_memory_transition_target target;
        target.stage=1;
        target.plan.stages={{1,{},{}}};
        llama_memory_layout layout;
        std::vector<llama_memory_workspace_group> groups;
        std::vector<llama_memory_region_binding> bindings;
        std::vector<ggml_backend_memory_workspace_group> grants;
        for (size_t i=0;i<leases.size();++i) {
            ggml_backend_memory_region region{};
            auto * buffer=leases[i] ? ggml_backend_memory_lease_buffer(leases[i]) : nullptr;
            const auto & need=s.measured[i];
            if (!buffer || !ggml_backend_memory_lease_get_region(leases[i],&region) || !region.id ||
                    ggml_backend_buffer_get_type(buffer) != need.buft || region.size != ggml_backend_buffer_get_size(buffer) ||
                    region.flags || region.size < need.size || region.alignment != need.alignment || region.size%need.alignment ||
                    region.offset > SIZE_MAX-region.size) return false;
            for (const auto & group : groups) if (group.resource.id == region.id) return false;
            auto grant=need; grant.size=region.size;
            const auto allocation=ggml_backend_buft_is_host(need.buft) ? LLAMA_MEMORY_ALLOCATION_HOST : LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL;
            groups.push_back({grant,{region.id,region.id,allocation,llama_memory_content::discardable},{}});
            grants.push_back(grant);
            target.plan.domains.push_back({region.id,allocation,LLAMA_MEMORY_CAPABILITY_BUFFER_VIEWS});
            const llama_memory_arena_budget budget{region.id,allocation,region.offset+region.size,region.alignment};
            target.budgets.push_back(budget);
            layout.arenas.push_back({budget,{region},region.size,region.offset+region.size,region.offset});
            bindings.push_back({i,region,leases[i]});
        }
        auto candidate=std::make_unique<llama_memory_workspace>(s.sched,std::move(groups),llama_memory_workspace_hooks{
            [&s] { s.executor.quiesce(); return true; }, [&s] { return s.invalidate(); }});
        if (!candidate->register_resources(target.plan,{1})) return false;
        std::unique_ptr<llama_memory_preparation> proposal;
        if (!candidate->prepare(target,layout,proposal) || !proposal) return false;
        if (!s.close()) return false;
        if (!proposal->quiesce({}) || !proposal->drain() || !proposal->invalidate() || !proposal->release() ||
                !proposal->bind(bindings) || !proposal->activate()) return false;
        proposal.reset();
        s.consumer=std::move(candidate);
        s.granted=std::move(grants);
        if (s.capture()) return true;
        s.close();
        return false;
    } catch (...) { return false; }
}

bool mtmd_compute_workspace::alloc_graph(ggml_cgraph * graph) {
    auto & s=*impl;
    if (!s.valid || s.busy || !graph || (!s.borrowed && (!s.consumer || !s.consumer->ready()))) return false;
    workspace_gate gate(s.busy);
    try {
        if (!s.invalidate() || (s.borrowed && !s.borrowed->prepare_serial_consumer(llama_memory_text_phase::prefill))) return false;
        workspace_requirements actual;
        if (!s.requirements(graph,actual)) return false;
        for (const auto & need : actual.groups) {
            const auto grant=std::find_if(s.granted.begin(),s.granted.end(),[&](const auto & group) { return group.buft == need.buft; });
            if (grant == s.granted.end() || need.size > grant->size) return false;
        }
        ggml_backend_sched_reset(s.sched);
        if (!s.limit_graphs(actual) || !ggml_backend_sched_alloc_graph(s.sched,graph) ||
                (!s.borrowed && !s.capture()) || !s.bind_operators(actual)) return false;
        s.graph=graph;
        return true;
    } catch (...) { return false; }
}

ggml_status mtmd_compute_workspace::compute_async(ggml_cgraph * graph) {
    auto & s=*impl;
    if (!s.valid || s.busy || !s.accepting || !graph || graph!=s.graph) return GGML_STATUS_FAILED;
    workspace_gate gate(s.busy);
    struct submission {
        implementation & state;
        bool installed;
        explicit submission(implementation & state) : state(state),installed(std::any_of(state.operators.begin(),state.operators.end(),
                [](const workspace_operator & op) { return op.query!=nullptr; })) {
            if (installed) ggml_backend_sched_set_pre_compute_callback(state.sched,[](ggml_backend_t backend,ggml_cgraph * graph,void * opaque) {
                return static_cast<implementation *>(opaque)->pre_compute(backend,graph);
            },&state);
        }
        ~submission() { if (installed) ggml_backend_sched_set_pre_compute_callback(state.sched,nullptr,nullptr); }
    } submitted(s);
    try {
        if (s.borrowed) return s.borrowed->compute_async(graph);
        if (!s.consumer || !s.executor.ready()) return GGML_STATUS_FAILED;
        if (!s.pending) s.pending=s.executor.acquire(s.consumer->leases(),s.revision);
        if (!s.pending) return GGML_STATUS_FAILED;
        const auto result=ggml_backend_sched_graph_compute_async(s.sched,graph);
        if (result!=GGML_STATUS_SUCCESS) { s.accepting=false;s.executor.quiesce(); }
        return result;
    } catch (...) { s.accepting=false;s.executor.quiesce();return GGML_STATUS_FAILED; }
}

bool mtmd_compute_workspace::drain() {
    if (!impl->valid || impl->busy) return false;
    workspace_gate gate(impl->busy);
    try { return impl->drain(); } catch (...) { return false; }
}

bool mtmd_compute_workspace::release() {
    if (impl->busy) return false;
    if (!impl->valid) return true;
    workspace_gate gate(impl->busy);
    try { return impl->close(); } catch (...) { return false; }
}

bool mtmd_compute_workspace::retire_graph() {
    if (!impl->valid || impl->busy) return false;
    workspace_gate gate(impl->busy);
    try { return impl->invalidate(); } catch (...) { return false; }
}

bool mtmd_compute_workspace::details(ggml_backend_buffer_type_t type,mtmd_workspace_details & output) const noexcept {
    for (size_t i=0;i<impl->measured.size();++i) if (impl->measured[i].buft==type && i<impl->components.size()) {
        output=impl->components[i];return true;
    }
    return false;
}

bool mtmd_compute_workspace::supported() const noexcept { return impl->valid; }
bool mtmd_compute_workspace::ready() const noexcept {
    return impl->accepting && (impl->borrowed ? impl->borrowed->serial_ready() : impl->consumer && impl->consumer->ready());
}

bool mtmd_compute_workspace::borrow(llama_context_memory & parent) {
    auto & s=*impl;
    if (!s.valid || s.busy || s.measured.empty() || s.consumer || s.borrowed) return false;
    workspace_gate gate(s.busy);
    try {
        auto child=llama_context_memory::borrow_workspace(s.sched,s.backends,s.measured,parent);
        if (!child || !child->prepare_serial_consumer(llama_memory_text_phase::prefill)) return false;
        std::vector<ggml_backend_memory_workspace_group> grants=s.measured;
        s.borrowed=std::move(child);
        s.granted=std::move(grants);
        return true;
    } catch (...) { return false; }
}
