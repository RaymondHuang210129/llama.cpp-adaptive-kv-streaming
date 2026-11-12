#include "llama-memory-phase-owner.h"
#include "llama-cparams.h"

#include <algorithm>
#include <limits>
#include <new>
#include <utility>

bool llama_memory_phase_workspace_budget_make(const llama_memory_phase_workspace_measurement & m,
        llama_memory_phase_workspace_budget & output) noexcept {
    if (!m.alignment || (m.alignment & (m.alignment-1)) ||
            !m.target_prefill_graph_bytes || !m.mtp_draft_graph_bytes) return false;
    const auto aligned_sum = [&](size_t graph, size_t stage, size_t & bytes) {
        if (!graph || stage > SIZE_MAX-graph) return false;
        const size_t total=graph+stage;
        if (total > SIZE_MAX-(m.alignment-1)) return false;
        bytes=(total+m.alignment-1)&~(m.alignment-1);
        return true;
    };
    llama_memory_phase_workspace_budget next;
    if (!aligned_sum(m.target_prefill_graph_bytes,0,next.target_prefill_bytes) ||
            !aligned_sum(m.mtp_draft_graph_bytes,0,next.mtp_draft_bytes)) return false;
    for (size_t width=0;width<4;++width) {
        size_t verify=0,catchup=0;
        if (!aligned_sum(m.target_verify_graph_bytes[width],m.rollback_stage_bytes,verify) ||
                !aligned_sum(m.mtp_catchup_graph_bytes[width],m.mtp_stage_bytes,catchup)) return false;
        next.target_verify_bytes=std::max(next.target_verify_bytes,verify);
        next.mtp_catchup_bytes=std::max(next.mtp_catchup_bytes,catchup);
    }
    next.shared_parent_bytes=std::max({
        next.target_prefill_bytes,next.target_verify_bytes,
        next.mtp_catchup_bytes,next.mtp_draft_bytes});
    next.reclaimed_during_mtp_catchup_bytes=next.shared_parent_bytes-next.mtp_catchup_bytes;
    next.reclaimed_during_mtp_draft_bytes=next.shared_parent_bytes-next.mtp_draft_bytes;
    output=next;
    return true;
}

static bool same_identity(llama_memory_phase_cache_identity a, llama_memory_phase_cache_identity b) {
    return a.id == b.id && a.generation == b.generation;
}

static bool same_workload(const llama_memory_workload_requirement & a, const llama_memory_workload_requirement & b) {
    return a.phase == b.phase && same_identity(a.cache, b.cache) && a.input_rows == b.input_rows &&
        a.query_rows == b.query_rows && a.output_rows == b.output_rows && a.rollback_depth == b.rollback_depth &&
        a.microbatch_rows == b.microbatch_rows && a.serial == b.serial && a.writes_cache == b.writes_cache && a.causal == b.causal &&
        same_identity(a.destination_cache,b.destination_cache) && a.hidden_output_rows==b.hidden_output_rows;
}

static llama_memory_workload_budget_result cache_storage_measure(
        const std::vector<llama_memory_cache_requirement> & caches,
        const std::vector<llama_memory_cache_allocation_measurement> & cache_allocations,
        size_t & persistent) {
    using status=llama_memory_workload_budget_status;
    if (caches.empty()) return {status::invalid_cache};
    if (cache_allocations.size()!=caches.size()) return {status::missing_measurement};
    const auto find_cache=[&](uint64_t id) -> const llama_memory_cache_requirement * {
        for (const auto & cache : caches) if (cache.identity.id == id) return &cache;
        return nullptr;
    };
    for (size_t i = 0; i < caches.size(); ++i) {
        const auto & cache = caches[i];
        const auto & allocation = cache_allocations[i];
        for (size_t j = 0; j < i; ++j) if (caches[j].identity.id == cache.identity.id) return {status::invalid_cache};
        const auto * source = cache.role == llama_memory_cache_role::read_only_source ? find_cache(cache.source.id) : nullptr;
        llama_memory_cache_bound bound;
        if (!llama_memory_cache_bound_make(cache, source, bound)) return {status::invalid_cache};
        if (!same_identity(cache.identity, allocation.cache) || allocation.capacity_rows != bound.capacity_rows ||
                !llama_memory_cache_format_equal(cache.format, allocation.format)) return {status::missing_measurement};
        if (!allocation.alignment || (allocation.alignment & (allocation.alignment-1)) ||
                (cache.role == llama_memory_cache_role::bounded_resident && allocation.bytes < bound.storage_bytes) ||
                (cache.role == llama_memory_cache_role::read_only_source && allocation.bytes)) return {status::invalid_measurement};
        if (!allocation.bytes) continue;
        if (persistent > SIZE_MAX-(allocation.alignment-1)) return {status::invalid_measurement};
        const size_t aligned = (persistent+allocation.alignment-1)&~(allocation.alignment-1);
        if (allocation.bytes > SIZE_MAX-aligned) return {status::invalid_measurement};
        persistent = aligned+allocation.bytes;
    }
    return {};
}

llama_memory_workload_budget_result llama_memory_workload_budget_make(
        const std::vector<llama_memory_cache_requirement> & caches,
        const std::vector<llama_memory_cache_allocation_measurement> & cache_allocations,
        const std::vector<llama_memory_workload_requirement> & workloads,
        const std::vector<llama_memory_workload_measurement> & measurements,
        size_t persistent_extra_bytes, size_t capacity, llama_memory_workload_budget & output) noexcept {
    using status = llama_memory_workload_budget_status;
    if (caches.empty()) return {status::invalid_cache};
    if (workloads.empty()) return {status::invalid_workload};
    if (measurements.size() != workloads.size() || cache_allocations.size() != caches.size()) return {status::missing_measurement};
    try {
        llama_memory_workload_budget next;
        size_t persistent = persistent_extra_bytes;
        const auto find_cache = [&](uint64_t id) -> const llama_memory_cache_requirement * {
            for (const auto & cache : caches) if (cache.identity.id == id) return &cache;
            return nullptr;
        };
        const auto storage=cache_storage_measure(caches,cache_allocations,persistent);
        if (storage.status != status::success) return storage;
        next.persistent_cache_bytes=persistent-persistent_extra_bytes;
        next.workload_bytes.reserve(workloads.size());
        for (size_t i = 0; i < workloads.size(); ++i) {
            const auto & workload = workloads[i];
            const auto & measurement = measurements[i];
            const auto * cache = find_cache(workload.cache.id);
            if (!cache) return {status::missing_cache, i};
            if (!same_identity(cache->identity, workload.cache)) return {status::stale_cache, i};
            const auto * destination=workload.destination_cache.id ? find_cache(workload.destination_cache.id) : nullptr;
            if (workload.destination_cache.id && !destination) return {status::missing_cache,i};
            if (destination && !same_identity(destination->identity,workload.destination_cache)) return {status::stale_cache,i};
            if (!llama_memory_workload_requirement_valid(workload,*cache,destination)) return {status::invalid_workload,i};
            if (!same_workload(workload, measurement.workload)) return {status::missing_measurement, i};
            if (!measurement.execution_available) return {status::missing_kernel, i};
            if (measurement.fallback_selected && !measurement.fallback_measured) return {status::unbudgeted_fallback, i};
            const auto alignment = measurement.alignment;
            if (!measurement.graph_bytes || !alignment || (alignment & (alignment-1)) ||
                    measurement.live_device_bytes > SIZE_MAX-measurement.graph_bytes) return {status::invalid_measurement, i};
            const size_t bytes = measurement.graph_bytes+measurement.live_device_bytes;
            if (bytes > SIZE_MAX-(alignment-1)) return {status::invalid_measurement, i};
            const size_t aligned = (bytes+alignment-1)&~(alignment-1);
            next.workload_bytes.push_back(aligned);
            next.workspace_peak_bytes = std::max(next.workspace_peak_bytes, aligned);
            next.host_feature_peak_bytes = std::max(next.host_feature_peak_bytes, measurement.host_feature_bytes);
        }
        for (size_t i = 0; i < measurements.size(); ++i) {
            const size_t alignment = measurements[i].alignment;
            if (persistent > SIZE_MAX-(alignment-1)) return {status::invalid_measurement, i};
            const size_t aligned = (persistent+alignment-1)&~(alignment-1);
            if (next.workload_bytes[i] > SIZE_MAX-aligned) return {status::invalid_measurement, i};
            next.parent_minimum_bytes = std::max(next.parent_minimum_bytes, aligned+next.workload_bytes[i]);
        }
        if (next.parent_minimum_bytes > capacity) return {status::insufficient_budget, 0, next.parent_minimum_bytes-capacity};
        output = std::move(next);
        return {};
    } catch (const std::bad_alloc &) {
        return {status::allocation_failed};
    }
}

static bool same_group(const llama_memory_group_workload & a,const llama_memory_group_workload & b) {
    if (a.id!=b.id || a.selected_count!=b.selected_count || a.microbatch_rows!=b.microbatch_rows ||
            a.native_batches!=b.native_batches || a.members.size()!=b.members.size()) return false;
    for (size_t i=0;i<a.members.size();++i) if (!same_workload(a.members[i],b.members[i])) return false;
    return true;
}

llama_memory_workload_budget_result llama_memory_group_workload_budget_make(
        const std::vector<llama_memory_cache_requirement> & caches,
        const std::vector<llama_memory_cache_allocation_measurement> & cache_allocations,
        const std::vector<llama_memory_group_workload> & workloads,
        const std::vector<llama_memory_group_workload_measurement> & measurements,
        size_t persistent_extra_bytes,size_t capacity,llama_memory_workload_budget & output) noexcept {
    using status=llama_memory_workload_budget_status;
    if (workloads.empty()) return {status::invalid_workload};
    if (measurements.size()!=workloads.size()) return {status::missing_measurement};
    try {
        size_t persistent=persistent_extra_bytes;
        const auto storage=cache_storage_measure(caches,cache_allocations,persistent);
        if (storage.status!=status::success) return storage;
        const auto find_cache=[&](uint64_t id) -> const llama_memory_cache_requirement * {
            for (const auto & cache : caches) if (cache.identity.id==id) return &cache;
            return nullptr;
        };
        for (size_t i=0;i<workloads.size();++i) {
            const auto & group=workloads[i];
            if (!group.id || !group.selected_count || group.selected_count>LLAMA_MAX_SEQ ||
                    group.selected_count!=workloads.front().selected_count ||
                    !group.microbatch_rows || group.members.empty() || group.members.size()>group.selected_count ||
                    group.native_batches.empty()) return {status::invalid_workload,i};
            for (size_t j=0;j<i;++j) if (workloads[j].id==group.id) return {status::invalid_workload,i};
            for (size_t m=0;m<group.members.size();++m) {
                const auto & member=group.members[m];
                const auto * cache=find_cache(member.cache.id);
                if (!cache) return {status::missing_cache,i};
                if (!same_identity(member.cache,cache->identity)) return {status::stale_cache,i};
                const auto * destination=member.destination_cache.id ? find_cache(member.destination_cache.id) : nullptr;
                if (member.destination_cache.id && !destination) return {status::missing_cache,i};
                if (destination && !same_identity(member.destination_cache,destination->identity)) return {status::stale_cache,i};
                if (member.microbatch_rows!=group.microbatch_rows || member.phase!=group.members[0].phase ||
                        bool(member.destination_cache.id)!=bool(group.members[0].destination_cache.id) ||
                        !llama_memory_workload_group_member_valid(member,*cache,destination)) return {status::invalid_workload,i};
                for (size_t prior=0;prior<m;++prior)
                    if (group.members[prior].cache.id==member.cache.id) return {status::invalid_workload,i};
            }
            std::vector<uint32_t> consumed(group.members.size(),0),outputs(group.members.size(),0),hidden(group.members.size(),0);
            for (const auto & batch : group.native_batches) {
                if (batch.empty()) return {status::invalid_workload,i};
                uint64_t rows=0;
                const llama_memory_cache_requirement * first=nullptr;
                const llama_memory_cache_requirement * first_destination=nullptr;
                for (const auto & range : batch) {
                    if (range.member>=group.members.size() || !range.rows) return {status::invalid_workload,i};
                    const auto member=range.member;
                    const auto & intent=group.members[member];
                    if (range.first_row!=consumed[member] || range.rows>intent.input_rows-consumed[member] ||
                            range.output_rows>range.rows || range.output_rows>intent.output_rows-outputs[member] ||
                            range.hidden_output_rows>range.rows || range.hidden_output_rows>intent.hidden_output_rows-hidden[member])
                        return {status::invalid_workload,i};
                    consumed[member]+=range.rows; outputs[member]+=range.output_rows;hidden[member]+=range.hidden_output_rows;
                    const auto * cache=find_cache(intent.cache.id);
                    if (first && !llama_memory_cache_format_equal(first->format,cache->format)) return {status::invalid_workload,i};
                    const auto * destination=intent.destination_cache.id ? find_cache(intent.destination_cache.id) : cache;
                    if (first_destination && !llama_memory_cache_format_equal(first_destination->format,destination->format))
                        return {status::invalid_workload,i};
                    first_destination=destination;
                    first=cache; rows+=range.rows;
                }
                if (rows>group.microbatch_rows) return {status::invalid_workload,i};
            }
            for (size_t member=0;member<group.members.size();++member)
                if (consumed[member]!=group.members[member].input_rows || outputs[member]!=group.members[member].output_rows ||
                        hidden[member]!=group.members[member].hidden_output_rows)
                    return {status::invalid_workload,i};
        }
        llama_memory_workload_budget next;
        next.persistent_cache_bytes=persistent-persistent_extra_bytes;
        for (size_t i=0;i<workloads.size();++i) {
            const auto & measurement=measurements[i];
            const auto batches=workloads[i].native_batches.size();
            if (!same_group(workloads[i],measurement.workload) || measurement.graph_bytes.size()!=batches ||
                    measurement.execution_available.size()!=batches) return {status::missing_measurement,i};
            size_t graph=0;
            for (size_t batch=0;batch<batches;++batch) {
                if (!measurement.execution_available[batch]) return {status::missing_kernel,i};
                if (!measurement.graph_bytes[batch]) return {status::invalid_measurement,i};
                graph=std::max(graph,measurement.graph_bytes[batch]);
            }
            const auto alignment=measurement.alignment;
            if (!alignment || (alignment&(alignment-1)) || measurement.live_device_bytes>SIZE_MAX-graph)
                return {status::invalid_measurement,i};
            const size_t bytes=graph+measurement.live_device_bytes;
            if (bytes>SIZE_MAX-(alignment-1) || persistent>SIZE_MAX-(alignment-1)) return {status::invalid_measurement,i};
            const size_t workspace=(bytes+alignment-1)&~(alignment-1);
            const size_t prefix=(persistent+alignment-1)&~(alignment-1);
            if (workspace>SIZE_MAX-prefix) return {status::invalid_measurement,i};
            next.workload_bytes.push_back(workspace);
            next.workspace_peak_bytes=std::max(next.workspace_peak_bytes,workspace);
            next.host_feature_peak_bytes=std::max(next.host_feature_peak_bytes,measurement.host_feature_bytes);
            next.parent_minimum_bytes=std::max(next.parent_minimum_bytes,prefix+workspace);
        }
        if (next.parent_minimum_bytes>capacity) return {status::insufficient_budget,0,next.parent_minimum_bytes-capacity};
        output=std::move(next);
        return {};
    } catch (const std::bad_alloc &) {return {status::allocation_failed};}
}

struct llama_memory_phase_owner::implementation {
    std::vector<llama_memory_phase_target> targets;
    std::vector<llama_memory_transition_arena> arenas;
    llama_memory_transition transition;
    llama_memory_work_phase_tracker tracker;
    llama_memory_phase_cache_identity target;
    llama_memory_phase_cache_identity mtp;
    uint32_t profile = 0;
    uint64_t selection_revision = 0;
    bool group_sources = false;
    bool closed = false;
    std::vector<llama_memory_phase_cache_identity> sources;

    implementation(std::vector<llama_memory_phase_target> targets,
            std::vector<llama_memory_transition_arena> arenas,
            std::vector<llama_memory_consumer *> consumers,
            llama_memory_phase_cache_identity target, llama_memory_phase_cache_identity mtp) :
        targets(std::move(targets)), arenas(std::move(arenas)),
        transition(std::move(consumers)), target(target), mtp(mtp) {}
};

llama_memory_phase_owner::llama_memory_phase_owner(std::unique_ptr<implementation> impl) : impl(std::move(impl)) {}
llama_memory_phase_owner::~llama_memory_phase_owner() = default;

static bool phase_catalog_valid(const std::vector<llama_memory_phase_target> & targets,
        const std::vector<llama_memory_transition_arena> & arenas,
        const std::vector<llama_memory_consumer *> & consumers) {
    if (targets.empty() || arenas.empty() || consumers.empty()) return false;
    for (size_t i=0;i<targets.size();++i) {
        if (targets[i].phase == llama_memory_work_phase::none || !targets[i].target.stage || !targets[i].max_tokens ||
                (targets[i].target.selection_revision || !targets[i].target.source_identities.empty())) return false;
        for (size_t j=0;j<i;++j) if (targets[i].phase == targets[j].phase && targets[i].profile == targets[j].profile) return false;
    }
    for (size_t i=0;i<consumers.size();++i) {
        if (!consumers[i]) return false;
        for (size_t j=0;j<i;++j) if (consumers[i] == consumers[j]) return false;
    }
    return true;
}

std::unique_ptr<llama_memory_phase_owner> llama_memory_phase_owner::create(
        std::vector<llama_memory_phase_target> targets,
        std::vector<llama_memory_transition_arena> arenas,
        std::vector<llama_memory_consumer *> consumers,
        llama_memory_phase_cache_identity target, llama_memory_phase_cache_identity mtp) {
    if (!target.id || !target.generation || !mtp.id || !mtp.generation ||
            target.id == mtp.id || targets.empty() || arenas.empty() || consumers.empty()) return {};
    if (!phase_catalog_valid(targets,arenas,consumers)) return {};
    try {
        auto state=std::make_unique<implementation>(
            std::move(targets),std::move(arenas),std::move(consumers),target,mtp);
        return std::unique_ptr<llama_memory_phase_owner>(new llama_memory_phase_owner(std::move(state)));
    } catch (const std::bad_alloc &) {
        return {};
    }
}

std::unique_ptr<llama_memory_phase_owner> llama_memory_phase_owner::create(
        std::vector<llama_memory_phase_target> targets,
        std::vector<llama_memory_transition_arena> arenas,
        std::vector<llama_memory_consumer *> consumers,
        std::vector<llama_memory_phase_cache_identity> sources) {
    if (sources.empty() || !phase_catalog_valid(targets,arenas,consumers)) return {};
    for (size_t i=0;i<sources.size();++i) {
        if (!sources[i].id || !sources[i].generation) return {};
        for (size_t j=0;j<i;++j) if (sources[i].id==sources[j].id) return {};
    }
    try {
        auto state=std::make_unique<implementation>(std::move(targets),std::move(arenas),
            std::move(consumers),llama_memory_phase_cache_identity{},llama_memory_phase_cache_identity{});
        state->group_sources=true;
        state->sources=std::move(sources);
        return std::unique_ptr<llama_memory_phase_owner>(new llama_memory_phase_owner(std::move(state)));
    } catch (const std::bad_alloc &) {return {};}
}

llama_memory_phase_owner_result llama_memory_phase_owner::enter(llama_memory_work_phase phase,
        uint32_t tokens, llama_memory_phase_cache_identity target, llama_memory_phase_cache_identity mtp) {
    return enter(phase,tokens,target,mtp,0,0);
}

llama_memory_phase_owner_result llama_memory_phase_owner::enter(llama_memory_work_phase phase,
        uint32_t tokens, llama_memory_phase_cache_identity target, llama_memory_phase_cache_identity mtp,
        uint32_t profile, uint64_t selection_revision) {
    using status=llama_memory_phase_owner_status;
    auto & s=*impl;
    if (s.group_sources) return {status::invalid_signal,{}};
    if (!same_identity(target,s.target) || !same_identity(mtp,s.mtp)) return {status::stale_identity,{}};
    return enter_profile(phase,tokens,profile,selection_revision);
}

llama_memory_phase_owner_result llama_memory_phase_owner::enter(llama_memory_work_phase phase,
        uint32_t tokens,const std::vector<llama_memory_phase_cache_identity> & sources,
        uint32_t profile,uint64_t selection_revision) {
    using status=llama_memory_phase_owner_status;
    const auto & s=*impl;
    if (!s.group_sources || !profile || !selection_revision) return {status::invalid_signal,{}};
    if (sources.size()!=s.sources.size()) return {status::stale_identity,{}};
    for (size_t i=0;i<sources.size();++i)
        if (!same_identity(sources[i],s.sources[i])) return {status::stale_identity,{}};
    return enter_profile(phase,tokens,profile,selection_revision);
}

llama_memory_phase_owner_result llama_memory_phase_owner::enter_profile(llama_memory_work_phase phase,
        uint32_t tokens,uint32_t profile,uint64_t selection_revision) {
    using status=llama_memory_phase_owner_status;
    auto & s=*impl;
    if (s.closed) return {status::session_invalid,{}};
    if (!tokens || phase == llama_memory_work_phase::none || bool(profile) != bool(selection_revision))
        return {status::invalid_signal,{}};
    if (selection_revision < s.selection_revision) return {status::stale_identity,{}};
    if (s.transition.state() == llama_memory_transition_state::invalid) return {status::session_invalid,{}};
    if (s.transition.state() != llama_memory_transition_state::idle) return {status::busy,{}};
    const auto found=std::find_if(s.targets.begin(),s.targets.end(),
        [&](const auto & entry) { return entry.phase == phase && entry.profile == profile; });
    if (found == s.targets.end()) return {status::invalid_phase,{}};
    if (tokens > found->max_tokens) return {status::invalid_signal,{}};
    const auto current=s.tracker.snapshot();
    if (current.notifications == std::numeric_limits<uint64_t>::max() ||
            (current.phase != phase && current.revision == std::numeric_limits<uint64_t>::max())) {
        return {status::invalid_signal,{}};
    }
    if (current.phase == phase && s.profile == profile && s.selection_revision == selection_revision) {
        s.tracker.notify({phase,tokens,true});
        return {status::unchanged,{}};
    }
    const auto * target_resources=&found->target;
    llama_memory_transition_target selection;
    if (selection_revision) {
        try {
            selection=found->target;
            if (s.group_sources) selection.source_identities=s.sources;
        }
        catch (const std::bad_alloc &) {
            llama_memory_transition_result failure;
            failure.status=llama_memory_transition_status::allocation_failed;
            return {status::transition_failed,failure};
        }
        selection.selection_revision=selection_revision;
        target_resources=&selection;
    }
    auto prepared=s.transition.prepare(*target_resources);
    if (prepared.status != llama_memory_transition_status::prepared &&
            prepared.status != llama_memory_transition_status::no_change) return {status::transition_failed,prepared};
    if (prepared.status == llama_memory_transition_status::prepared) {
        auto activated=s.transition.activate(s.arenas);
        if (activated.status != llama_memory_transition_status::activated) {
            if (s.transition.state() == llama_memory_transition_state::failed &&
                    s.transition.recover().status != llama_memory_transition_status::recovered) {
                return {status::session_invalid,activated};
            }
            return {status::transition_failed,activated};
        }
    }
    s.tracker.notify({phase,tokens,true});
    s.profile=profile;
    s.selection_revision=selection_revision;
    return {status::changed,{}};
}

uint32_t llama_memory_phase_owner::active_profile() const noexcept {return impl->profile;}
uint64_t llama_memory_phase_owner::selection_revision() const noexcept {return impl->selection_revision;}

uint64_t llama_memory_phase_owner::admit() noexcept {
    return impl->closed || impl->tracker.snapshot().phase == llama_memory_work_phase::none ? 0 : impl->transition.admit();
}

bool llama_memory_phase_owner::close() noexcept {
    if (impl->closed) return true;
    if (impl->transition.state()!=llama_memory_transition_state::idle) return false;
    impl->closed=true;return true;
}

bool llama_memory_phase_owner::finish(uint64_t admission,
        llama_memory_phase_cache_identity target, llama_memory_phase_cache_identity mtp) noexcept {
    auto & s=*impl;
    if (s.group_sources || !admission || target.id != s.target.id || mtp.id != s.mtp.id ||
            target.generation < s.target.generation || mtp.generation < s.mtp.generation ||
            !s.transition.finish(admission)) return false;
    s.target=target;
    s.mtp=mtp;
    return true;
}

bool llama_memory_phase_owner::finish(uint64_t admission,
        const std::vector<llama_memory_phase_cache_identity> & sources) noexcept {
    auto & s=*impl;
    if (!s.group_sources || !admission || sources.size()!=s.sources.size()) return false;
    for (size_t i=0;i<sources.size();++i)
        if (sources[i].id!=s.sources[i].id || sources[i].generation<s.sources[i].generation) return false;
    if (!s.transition.finish(admission)) return false;
    for (size_t i=0;i<sources.size();++i) s.sources[i].generation=sources[i].generation;
    return true;
}

const std::vector<llama_memory_phase_cache_identity> & llama_memory_phase_owner::source_identities() const noexcept {
    return impl->sources;
}

llama_memory_work_phase_snapshot llama_memory_phase_owner::snapshot() const noexcept {
    return impl->tracker.snapshot();
}
llama_memory_phase_cache_identity llama_memory_phase_owner::target_identity() const noexcept {
    return impl->target;
}
llama_memory_phase_cache_identity llama_memory_phase_owner::mtp_identity() const noexcept {
    return impl->mtp;
}
