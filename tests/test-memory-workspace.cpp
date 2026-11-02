#include "../src/llama-memory-workspace.h"
#include "../src/llama-context-workspace.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "../ggml/src/ggml-backend-operator-workspace.h"
#include "../ggml/src/ggml-cuda/scratch-plan.h"
#include "testing.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"

#include <stdexcept>
#include <cstring>

using arena_ptr = std::unique_ptr<ggml_backend_memory_arena, decltype(&ggml_backend_memory_arena_free)>;
using status = llama_memory_transition_status;

// A real CPU allocation with view support withheld for scheduler-fallback tests.
static ggml_backend_buffer_t no_views_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    auto * buffer = ggml_backend_cpu_buffer_type()->iface.alloc_buffer(buft, size);
    if (buffer) buffer->view_buffer = nullptr;
    return buffer;
}

struct late_failure : llama_memory_consumer {
    bool fail = false;
    struct token : llama_memory_preparation {
        late_failure & owner;
        explicit token(late_failure & owner) : owner(owner) {}
        bool quiesce(const std::vector<llama_memory_resource_id> &) override { return true; }
        bool drain() override { return true; }
        bool invalidate() override { return true; }
        bool release() override { return true; }
        bool bind(const std::vector<llama_memory_region_binding> &) override { return true; }
        bool activate() override { return !owner.fail; }
        bool prepare_recovery(std::unique_ptr<llama_memory_preparation> &) override { return true; }
    };
    bool prepare(const llama_memory_transition_target &, const llama_memory_layout &,
            std::unique_ptr<llama_memory_preparation> & output) override {
        output = std::make_unique<token>(*this);
        return true;
    }
};

struct fixture {
    ggml_backend_ptr first{ggml_backend_cpu_init()}, second{ggml_backend_cpu_init()};
    ggml_backend_buffer_type separate = *ggml_backend_cpu_buffer_type();
    std::vector<ggml_backend_t> backends{first.get(), second.get()};
    std::vector<ggml_backend_buffer_type_t> bufts;
    ggml_backend_sched_ptr sched;
    std::vector<arena_ptr> parents;
    std::vector<llama_memory_transition_arena> arenas;
    std::vector<llama_memory_workspace_group> groups;
    llama_memory_transition_target target;
    int quiesces = 0, invalidations = 0;
    bool fail_invalidate = false, throw_invalidate = false;
    std::unique_ptr<llama_memory_workspace> workspace;
    late_failure later;
    std::unique_ptr<llama_memory_transition> transition;

    // Use measured phase maxima and the real scheduler's shared-buffer-type grouping.
    fixture(bool alias = true, bool manage_second = true, bool second_views = true) {
        if (!second_views) separate.iface.alloc_buffer = no_views_alloc;
        bufts = {ggml_backend_cpu_buffer_type(), alias ? ggml_backend_cpu_buffer_type() : &separate};
        sched.reset(ggml_backend_sched_new(backends.data(), bufts.data(), 2, 256, false, true));
        GGML_ASSERT(sched);
        const size_t sizes[] = {4096, 8192, 2048, 4096};
        ggml_backend_memory_workspace_group measured[2];
        size_t count = 2;
        GGML_ASSERT(ggml_backend_memory_plan_workspace_groups(bufts.data(), sizes, 2, 2, measured, &count));
        target.plan.stages = {{10, {}, {}}, {20, {10}, {}}};
        target.stage = 10;
        for (size_t i = 0; i < count; ++i) {
            if (i == 1 && !manage_second) continue;
            const auto & group = measured[i];
            const uint64_t domain = i + 1, id = i + 11;
            groups.push_back({group, {id, domain, LLAMA_MEMORY_ALLOCATION_HOST, llama_memory_content::discardable}, {}});
            parents.emplace_back(ggml_backend_memory_arena_new(group.buft, 2*group.size), ggml_backend_memory_arena_free);
            GGML_ASSERT(parents.back());
            arenas.push_back({domain, LLAMA_MEMORY_ALLOCATION_HOST, parents.back().get()});
            target.plan.domains.push_back({domain, LLAMA_MEMORY_ALLOCATION_HOST, LLAMA_MEMORY_CAPABILITY_BUFFER_VIEWS});
            target.budgets.push_back({domain, LLAMA_MEMORY_ALLOCATION_HOST, 2*group.size, group.alignment});
        }
        workspace = std::make_unique<llama_memory_workspace>(sched.get(), groups, llama_memory_workspace_hooks{
            [&] { ++quiesces; return true; },
            [&] {
                ++invalidations;
                if (throw_invalidate) throw std::runtime_error("workspace invalidation");
                return !fail_invalidate;
            },
        });
        transition = std::make_unique<llama_memory_transition>(std::vector<llama_memory_consumer *>{workspace.get(), &later});
    }

    // Build the consumer declarations once, then prepare and activate against supplied parent arenas.
    bool start(testing & t) {
        if (!t.assert_true(workspace->register_resources(target.plan, {10, 20}))) return false;
        if (!t.assert_true(transition->prepare(target).status == status::prepared)) return false;
        return t.assert_true(transition->activate(arenas).status == status::activated);
    }

    // Force a new offset without changing the maximum workspace grant.
    void move_workspace() {
        target.plan.resources.push_back({99, 1, LLAMA_MEMORY_ALLOCATION_HOST, llama_memory_content::discardable});
        auto & requirements = target.plan.stages[0].requirements;
        requirements.insert(requirements.begin(), {99, groups[0].workspace.alignment, groups[0].workspace.alignment,
            groups[0].workspace.alignment, LLAMA_MEMORY_ACCESS_WRITE, 0});
    }

    // Reserve and evaluate with the scheduler to prove attachment is usable, not just retained.
    void compute(testing & t, size_t slot = 0) {
        ggml_context_ptr ctx(ggml_init({1024*1024, nullptr, true}));
        auto * input = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
        auto * output = ggml_scale(ctx.get(), input, 2.0f);
        auto * graph = ggml_new_graph_custom(ctx.get(), 16, false);
        ggml_build_forward_expand(graph, output);
        ggml_backend_sched_set_tensor_backend(sched.get(), input, backends[slot]);
        ggml_backend_sched_set_tensor_backend(sched.get(), output, backends[slot]);
        if (!t.assert_true(ggml_backend_sched_reserve(sched.get(), graph))) return;
        if (!t.assert_true(ggml_backend_sched_alloc_graph(sched.get(), graph))) return;
        const float data[4] = {1, 2, 3, 4};
        float result[4] = {};
        ggml_backend_tensor_set(input, data, 0, sizeof(data));
        t.assert_true(ggml_backend_sched_graph_compute(sched.get(), graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(output, result, 0, sizeof(result));
        for (int i = 0; i < 4; ++i) t.assert_equal(2*data[i], result[i]);
        ggml_backend_sched_reset(sched.get());
    }
};

int main(int argc, char ** argv) {
    testing t;
    t.test("operator_scratch_layout_checks_alignment_overflow_and_request_limits",[](testing & t) {
        ggml_backend_operator_workspace plan;
        t.assert_true(ggml_backend_operator_workspace_add(plan,129));
        t.assert_equal(size_t(0),plan.offsets[0]);t.assert_equal(size_t(256),plan.bytes);
        t.assert_true(ggml_backend_operator_workspace_add(plan,128));
        t.assert_equal(size_t(256),plan.offsets[1]);t.assert_equal(size_t(384),plan.bytes);
        const auto saved=plan;
        t.assert_true(!ggml_backend_operator_workspace_add(plan,SIZE_MAX));
        t.assert_equal(saved.bytes,plan.bytes);t.assert_equal(saved.count,plan.count);
        t.assert_true(ggml_backend_operator_workspace_add(plan,0));t.assert_equal(saved.count,plan.count);
        auto invalid=plan;invalid.alignment=3;
        t.assert_true(!ggml_backend_operator_workspace_add(invalid,1));t.assert_equal(plan.bytes,invalid.bytes);
        while (plan.count<plan.requests.size()) t.assert_true(ggml_backend_operator_workspace_add(plan,1));
        const auto full=plan;
        t.assert_true(!ggml_backend_operator_workspace_add(plan,1));t.assert_equal(full.bytes,plan.bytes);
    });
    t.test("operator_phase_adds_concurrent_stream_peaks_and_one_library_workspace_per_stream",[](testing & t) {
        ggml_backend_operator_workspace ops[3];
        t.assert_true(ggml_backend_operator_workspace_add(ops[0],384));
        t.assert_true(ggml_backend_operator_workspace_add(ops[1],128));
        t.assert_true(ggml_backend_operator_workspace_add(ops[2],128));
        for (auto & op : ops) op.library_bytes=32*1024*1024;
        const uint32_t streams[]={0,0,1};ggml_backend_operator_phase_workspace phase;
        t.assert_true(ggml_backend_operator_workspace_group(ops,streams,3,2,phase));
        t.assert_equal(size_t(384),phase.streams[0].scratch_bytes);
        t.assert_equal(size_t(128),phase.streams[1].scratch_bytes);
        t.assert_equal(size_t(64*1024*1024+768),phase.bytes);
        t.assert_equal(size_t(0),phase.streams[0].library_offset%256);
        t.assert_equal(size_t(0),phase.streams[1].library_offset%256);
        t.assert_true(phase.streams[0].scratch_offset+384<=phase.streams[1].library_offset);
        const auto before=phase;const uint32_t bad[]={0,0,2};
        t.assert_true(!ggml_backend_operator_workspace_group(ops,bad,3,2,phase));
        t.assert_equal(before.bytes,phase.bytes);t.assert_equal(before.count,phase.count);
        ops[2].library_bytes=SIZE_MAX;
        t.assert_true(!ggml_backend_operator_workspace_group(ops,streams,3,2,phase));
        t.assert_equal(before.bytes,phase.bytes);
    });
    t.test("native_plain_tile_attention_quote_matches_observed_split_storage_without_device_payloads",[](testing & t) {
        ggml_context_ptr ctx(ggml_init({16384,nullptr,true}));
        ggml_cuda_operator_workspace_caps caps;caps.ready=true;caps.nsm=70;
        for (auto & row : caps.tile) for (auto & config : row) { config.occupancy=2;config.nbatch=64; }
        for (const auto pair : {std::pair<int64_t,size_t>{256,2424832},{576,19095552},{16,0}}) {
            auto * q=ggml_new_tensor_3d(ctx.get(),GGML_TYPE_F32,72,pair.first,16);
            auto * k=ggml_new_tensor_3d(ctx.get(),GGML_TYPE_F16,72,pair.first,16);
            auto * v=ggml_new_tensor_3d(ctx.get(),GGML_TYPE_F16,72,pair.first,16);
            auto * op=ggml_flash_attn_ext(ctx.get(),q,k,v,nullptr,0.125f,0.0f,0.0f);
            ggml_backend_operator_workspace plan;
            t.assert_true(ggml_cuda_attention_workspace_plan(caps,op,plan));
            t.assert_equal(pair.second,plan.bytes);
            t.assert_true(!q->data && !k->data && !v->data && !op->data);
            const auto saved=plan;caps.ready=false;
            t.assert_true(!ggml_cuda_attention_workspace_plan(caps,op,plan));
            t.assert_equal(saved.bytes,plan.bytes);caps.ready=true;
            const auto width=k->ne[0];k->ne[0]=80;
            t.assert_true(!ggml_cuda_attention_workspace_plan(caps,op,plan));k->ne[0]=width;
            t.assert_equal(saved.bytes,plan.bytes);
        }
    });
    t.test("native_cublas_quote_includes_conversions_outputs_batch_pointers_and_bounded_weight_chunks",[](testing & t) {
        ggml_context_ptr ctx(ggml_init({16384,nullptr,true}));
        auto * a=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F16,1152,4304);
        auto * b=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,1152,256);
        auto * op=ggml_mul_mat(ctx.get(),a,b);
        ggml_backend_operator_workspace plan;
        t.assert_true(ggml_cuda_cublas_workspace_plan(op,GGML_TYPE_F16,false,32*1024*1024,plan));
        t.assert_equal(size_t(2),plan.count);t.assert_equal(size_t(589824),plan.requests[0]);
        t.assert_equal(size_t(2203648),plan.requests[1]);t.assert_equal(size_t(2793472),plan.bytes);
        t.assert_equal(size_t(32*1024*1024),plan.library_bytes);
        auto * large_a=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F16,8192,65536);
        auto * large_b=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,8192,32);
        auto * large=ggml_mul_mat(ctx.get(),large_a,large_b);
        t.assert_true(ggml_cuda_cublas_workspace_plan(large,GGML_TYPE_F32,true,32*1024*1024,plan));
        t.assert_equal(size_t(512*1024*1024),plan.bytes);t.assert_equal(size_t(1),plan.count);
        auto * batch_a=ggml_new_tensor_3d(ctx.get(),GGML_TYPE_F16,16,32,1);
        auto * batch_b=ggml_new_tensor_3d(ctx.get(),GGML_TYPE_F16,16,24,4);
        auto * batched=ggml_mul_mat(ctx.get(),batch_a,batch_b);
        t.assert_true(ggml_cuda_cublas_workspace_plan(batched,GGML_TYPE_F16,false,32*1024*1024,plan));
        t.assert_equal(size_t(3),plan.count);t.assert_equal(size_t(6400),plan.bytes);
        t.assert_equal(size_t(64),plan.requests[1]);t.assert_equal(size_t(32),plan.requests[2]);
        const auto saved=plan;batch_b->ne[3]=INT64_MAX;
        t.assert_true(!ggml_cuda_cublas_workspace_plan(batched,GGML_TYPE_F16,false,32*1024*1024,plan));
        t.assert_equal(saved.bytes,plan.bytes);t.assert_true(!large_a->data && !large_b->data);
    });
    t.test("native_mmq_quote_includes_stream_k_fixup_only_for_fractional_tiles",[](testing & t) {
        size_t bytes=123;
        t.assert_true(ggml_cuda_mmq_fixup_workspace(1152,576,1,1,128,128,70,true,true,bytes));
        t.assert_equal(size_t(4587520),bytes);
        t.assert_true(ggml_cuda_mmq_fixup_workspace(1280,896,1,1,128,128,70,true,true,bytes));
        t.assert_equal(size_t(0),bytes);
        t.assert_true(ggml_cuda_mmq_fixup_workspace(1152,576,1,1,128,128,70,false,true,bytes));
        t.assert_equal(size_t(0),bytes);
        bytes=123;
        t.assert_true(!ggml_cuda_mmq_fixup_workspace(INT64_MAX,576,1,1,128,128,70,true,true,bytes));
        t.assert_equal(size_t(123),bytes);
    });
    if (argc>1 && !std::strcmp(argv[1],"--cuda")) t.test("cuda_operator_queries_require_preparation_and_never_read_tensor_payloads",[](testing & t) {
        ggml_backend_load_all();auto * dev=ggml_backend_dev_by_name("CUDA0");
        if (!t.assert_true(dev!=nullptr)) return;
        ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
        auto * reg=ggml_backend_dev_backend_reg(dev);
        auto prepare=reinterpret_cast<ggml_backend_operator_workspace_prepare_t>(
            ggml_backend_reg_get_proc_address(reg,"ggml_backend_operator_workspace_prepare"));
        auto query=reinterpret_cast<ggml_backend_operator_workspace_query_t>(
            ggml_backend_reg_get_proc_address(reg,"ggml_backend_operator_workspace_query"));
        if (!t.assert_true(prepare && query && backend)) return;
        ggml_context_ptr ctx(ggml_init({16384,nullptr,true}));
        auto * a=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F16,1152,4304);
        auto * b=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,1152,256);auto * op=ggml_mul_mat(ctx.get(),a,b);
        ggml_backend_operator_workspace plan;plan.bytes=123;
        t.assert_true(query(backend.get(),op,&plan)==ggml_backend_operator_workspace_status::unprepared);
        t.assert_equal(size_t(123),plan.bytes);
        if (!t.assert_true(prepare(backend.get())==ggml_backend_operator_workspace_status::success)) return;
        const auto begin=ggml_time_us();
        t.assert_true(query(backend.get(),op,&plan)==ggml_backend_operator_workspace_status::success);
        t.assert_true(ggml_backend_operator_workspace_valid(plan) && plan.library_bytes>0);
        auto * quantized=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_Q8_0,1152,4304);
        auto * quant_op=ggml_mul_mat(ctx.get(),quantized,b);
        t.assert_true(query(backend.get(),quant_op,&plan)==ggml_backend_operator_workspace_status::success);
        t.assert_true(ggml_backend_operator_workspace_valid(plan) && plan.bytes>0);
        auto * q=ggml_new_tensor_3d(ctx.get(),GGML_TYPE_F32,72,576,16);
        auto * k=ggml_new_tensor_3d(ctx.get(),GGML_TYPE_F16,72,576,16);auto * v=ggml_dup_tensor(ctx.get(),k);
        auto * attention=ggml_flash_attn_ext(ctx.get(),q,k,v,nullptr,0.125f,0,0);
        t.assert_true(query(backend.get(),attention,&plan)==ggml_backend_operator_workspace_status::success);
        t.assert_true(ggml_backend_operator_workspace_valid(plan) && plan.library_bytes==0);
        const auto before=plan;auto * unsupported=ggml_soft_max(ctx.get(),q);
        t.assert_true(query(backend.get(),unsupported,&plan)==ggml_backend_operator_workspace_status::unsupported);
        t.assert_equal(before.bytes,plan.bytes);
        q->ne[1]=INT64_MAX;
        t.assert_true(query(backend.get(),attention,&plan)!=ggml_backend_operator_workspace_status::success);
        t.assert_equal(before.bytes,plan.bytes);
        t.assert_true(!a->data && !b->data && !q->data && !k->data && !v->data);
        t.out << "operator-query-window: begin_us=" << begin << " end_us=" << ggml_time_us() << '\n';
    });
    t.test("scheduler_pre_compute_runs_after_input_copies_and_can_reject_without_output", [&](testing & t) {
        const bool gpu=argc>1 && !std::strcmp(argv[1],"--cuda");
        ggml_backend_ptr cpu(ggml_backend_cpu_init()),device;
        if (gpu) {
            ggml_backend_load_all();
            auto * dev=ggml_backend_dev_by_name("CUDA0");
            if (!t.assert_true(dev != nullptr)) return;
            device.reset(ggml_backend_dev_init(dev,nullptr));
        }
        auto * compute=gpu ? device.get() : cpu.get();
        std::vector<ggml_backend_t> backends=gpu ? std::vector<ggml_backend_t>{compute,cpu.get()} : std::vector<ggml_backend_t>{compute};
        ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends.data(),nullptr,int(backends.size()),128,false,true));
        ggml_context_ptr source(ggml_init({16384,nullptr,true})),ctx(ggml_init({16384,nullptr,true}));
        auto * input=ggml_new_tensor_1d(source.get(),GGML_TYPE_F32,4);
        auto * weights=ggml_new_tensor_2d(source.get(),GGML_TYPE_F32,4,4);
        ggml_set_input(input);
        ggml_set_input(weights);
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(source.get(),cpu.get()));
        auto * output=ggml_mul_mat(ctx.get(),weights,input);
        auto * graph=ggml_new_graph_custom(ctx.get(),32,false);
        ggml_build_forward_expand(graph,output);
        ggml_backend_sched_set_tensor_backend(sched.get(),input,cpu.get());
        ggml_backend_sched_set_tensor_backend(sched.get(),output,compute);
        if (!t.assert_true(sched && buffer && ggml_backend_sched_alloc_graph(sched.get(),graph))) return;
        const float data[4]={1,2,3,4},sentinel[4]={-9,-9,-9,-9};
        float matrix[16]={};
        for (size_t i=0;i<4;++i) matrix[5*i]=2.0f;
        ggml_backend_tensor_set(weights,matrix,0,sizeof(matrix));
        ggml_backend_tensor_set(input,data,0,sizeof(data));
        ggml_backend_tensor_set(output,sentinel,0,sizeof(sentinel));
        struct state {
            testing * test;
            ggml_backend_t backend;
            ggml_tensor * input;
            ggml_tensor * output;
            size_t calls=0;
            bool allow=false;
            bool gpu=false;
        } state{&t,compute,input,output,0,false,gpu};
        const auto callback=[](ggml_backend_t backend,ggml_cgraph * split,void * opaque) {
            auto & s=*static_cast<struct state *>(opaque);
            ++s.calls;
            s.test->assert_true(backend==s.backend && ggml_graph_n_nodes(split)==1);
            auto * copied=ggml_graph_node(split,0)->src[1];
            s.test->assert_true(!s.gpu || copied!=s.input);
            s.test->assert_true(s.output->data!=copied->data);
            float values[4];
            ggml_backend_tensor_get(copied,values,0,sizeof(values));
            for (size_t i=0;i<4;++i) s.test->assert_equal(float(i+1),values[i]);
            ggml_backend_tensor_get(s.output,values,0,sizeof(values));
            for (float value : values) s.test->assert_equal(-9.0f,value);
            return s.allow;
        };
        ggml_backend_sched_set_pre_compute_callback(sched.get(),callback,&state);
        t.assert_true(ggml_backend_sched_graph_compute_async(sched.get(),graph)==GGML_STATUS_FAILED);
        t.assert_equal(size_t(1),state.calls);
        state.allow=true;
        t.assert_true(ggml_backend_sched_graph_compute(sched.get(),graph)==GGML_STATUS_SUCCESS);
        t.assert_equal(size_t(2),state.calls);
        ggml_backend_sched_set_pre_compute_callback(sched.get(),nullptr,nullptr);
        t.assert_true(ggml_backend_sched_graph_compute(sched.get(),graph)==GGML_STATUS_SUCCESS);
        t.assert_equal(size_t(2),state.calls);
        float result[4];ggml_backend_tensor_get(output,result,0,sizeof(result));
        for (size_t i=0;i<4;++i) t.assert_equal(2.0f*data[i],result[i]);
    });
    t.test("registers_phase_maximum_once_for_aliased_slots", [](testing & t) {
        fixture f;
        t.assert_true(f.workspace->register_resources(f.target.plan, {10, 20}));
        t.assert_equal(size_t(1), f.target.plan.resources.size());
        if (!t.assert_true(!f.target.plan.stages[0].requirements.empty())) return;
        for (const auto & stage : f.target.plan.stages) {
            t.assert_equal(size_t(1), stage.requirements.size());
            t.assert_equal(size_t(8192), stage.requirements[0].size_min);
            t.assert_equal(size_t(8192), stage.requirements[0].size_preferred);
            t.assert_true(stage.requirements[0].access == LLAMA_MEMORY_ACCESS_WRITE);
        }
        t.assert_equal(size_t(0), ggml_backend_sched_get_buffer_size(f.sched.get(), f.first.get()));
        t.assert_equal(size_t(0), ggml_backend_memory_arena_region_count(f.parents[0].get()));
    });

    t.test("registration_rejects_collisions_without_partial_output", [](testing & t) {
        fixture f;
        t.assert_true(!f.workspace->register_resources(f.target.plan, {10, 10}));
        t.assert_true(!f.workspace->register_resources(f.target.plan, {99}));
        t.assert_true(f.target.plan.resources.empty());
        if (!t.assert_true(f.workspace->register_resources(f.target.plan, {10, 20}))) return;
        t.assert_true(!f.workspace->register_resources(f.target.plan, {10}));
        t.assert_equal(size_t(1), f.target.plan.resources.size());
        t.assert_equal(size_t(1), f.target.plan.stages[0].requirements.size());
    });

    t.test("attach_reserve_compute_and_detach_shared_slots", [](testing & t) {
        fixture f;
        if (!f.start(t)) return;
        t.assert_true(f.workspace->ready());
        t.assert_equal(size_t(8192), ggml_backend_sched_get_buffer_size(f.sched.get(), f.first.get()));
        // Shared storage is reported once, on the first slot; both slots still borrow its lease.
        t.assert_equal(size_t(0), ggml_backend_sched_get_buffer_size(f.sched.get(), f.second.get()));
        t.assert_equal(size_t(1), ggml_backend_memory_arena_lease_count(f.parents[0].get()));
        f.compute(t, 1);
        t.assert_true(f.workspace->close());
        t.assert_true(!f.workspace->ready());
        t.assert_equal(size_t(0), ggml_backend_sched_get_buffer_size(f.sched.get(), f.first.get()));
        t.assert_equal(size_t(0), ggml_backend_sched_get_buffer_size(f.sched.get(), f.second.get()));
        t.assert_equal(size_t(0), ggml_backend_memory_arena_lease_count(f.parents[0].get()));
        t.assert_true(f.workspace->close());
    });

    t.test("unchanged_maximum_avoids_invalidation_in_decode", [](testing & t) {
        fixture f;
        if (!f.start(t)) return;
        const int invalidations = f.invalidations, quiesces = f.quiesces;
        const auto generation = ggml_backend_memory_arena_generation(f.parents[0].get());
        f.target.stage = 20;
        t.assert_true(f.transition->prepare(f.target).status == status::prepared);
        t.assert_true(f.transition->activate(f.arenas).status == status::activated);
        t.assert_equal(invalidations, f.invalidations);
        t.assert_equal(quiesces, f.quiesces);
        t.assert_equal(generation, ggml_backend_memory_arena_generation(f.parents[0].get()));
        f.compute(t);
    });

    t.test("cancelled_preparation_does_not_touch_active_scheduler", [](testing & t) {
        fixture f;
        if (!f.start(t)) return;
        const int invalidations = f.invalidations;
        t.assert_true(f.transition->prepare(f.target).status == status::prepared);
        t.assert_true(!f.workspace->close());
        t.assert_true(f.transition->cancel());
        t.assert_equal(invalidations, f.invalidations);
        t.assert_true(f.workspace->ready());
        f.compute(t);
    });

    t.test("partial_attachment_failure_leaves_foreign_range_alone", [](testing & t) {
        fixture f(false);
        ggml_backend_buffer_ptr external(ggml_backend_buft_alloc_buffer(f.bufts[1], 8192));
        GGML_ASSERT(ggml_backend_sched_set_buffer_range(f.sched.get(), f.second.get(), external.get(), 0, 8192));
        if (!t.assert_true(f.workspace->register_resources(f.target.plan, {10, 20}))) return;
        t.assert_true(f.transition->prepare(f.target).status == status::prepared);
        t.assert_true(f.transition->activate(f.arenas).status == status::activation_failed);
        t.assert_true(!f.workspace->ready());
        t.assert_equal(size_t(0), ggml_backend_sched_get_buffer_size(f.sched.get(), f.first.get()));
        t.assert_equal(size_t(8192), ggml_backend_sched_get_buffer_size(f.sched.get(), f.second.get()));
        for (auto & arena : f.parents) t.assert_equal(size_t(1), ggml_backend_memory_arena_lease_count(arena.get()));
        t.assert_true(f.transition->recover().status == status::recovered);
        t.assert_true(!f.workspace->ready());
        GGML_ASSERT(ggml_backend_sched_clear_buffer_range(f.sched.get(), f.second.get()));
    });

    t.test("moved_workspace_recovers_after_later_consumer_failure", [](testing & t) {
        fixture f;
        if (!f.start(t)) return;
        f.move_workspace();
        f.later.fail = true;
        t.assert_true(f.transition->prepare(f.target).status == status::prepared);
        t.assert_true(f.transition->activate(f.arenas).status == status::activation_failed);
        f.later.fail = false;
        t.assert_true(f.transition->recover().status == status::recovered);
        ggml_backend_memory_region region{};
        t.assert_true(ggml_backend_memory_arena_get_region(f.parents[0].get(), 11, &region));
        t.assert_equal(size_t(0), region.offset);
        t.assert_true(f.workspace->ready());
        f.compute(t);
    });

    t.test("failed_invalidation_does_not_release_old_lease", [](testing & t) {
        fixture f;
        if (!f.start(t)) return;
        f.move_workspace();
        f.fail_invalidate = true;
        t.assert_true(f.transition->prepare(f.target).status == status::prepared);
        t.assert_true(f.transition->activate(f.arenas).status == status::activation_failed);
        t.assert_true(!f.workspace->ready());
        t.assert_equal(size_t(1), ggml_backend_memory_arena_lease_count(f.parents[0].get()));
        f.fail_invalidate = false;
        t.assert_true(f.transition->recover().status == status::recovered);
        t.assert_true(f.workspace->ready());
        f.compute(t);
    });

    t.test("omitted_nonview_group_uses_scheduler_fallback", [](testing & t) {
        fixture f(false, false, false);
        if (!f.start(t)) return;
        t.assert_equal(size_t(1), f.groups.size());
        t.assert_equal(size_t(0), ggml_backend_sched_get_buffer_size(f.sched.get(), f.second.get()));
        f.compute(t, 1);
        const auto fallback_size = ggml_backend_sched_get_buffer_size(f.sched.get(), f.second.get());
        t.assert_true(fallback_size > 0);
        t.assert_true(f.workspace->close());
        t.assert_equal(fallback_size, ggml_backend_sched_get_buffer_size(f.sched.get(), f.second.get()));
    });

    t.test("destruction_detaches_before_parent_owner_disappears", [](testing & t) {
        fixture f;
        if (!f.start(t)) return;
        f.transition.reset();
        f.workspace.reset();
        t.assert_equal(size_t(0), ggml_backend_memory_arena_lease_count(f.parents[0].get()));
        t.assert_equal(size_t(0), ggml_backend_sched_get_buffer_size(f.sched.get(), f.first.get()));
    });

    t.test("rejects_mislabeled_and_noncanonical_scheduler_groups", [](testing & t) {
        fixture f;
        for (int bad : {0, 1, 2, 3}) {
            auto groups = f.groups;
            if (bad == 0) groups[0].workspace.buft = &f.separate;
            if (bad == 1) groups[0].workspace.first_slot = 1;
            if (bad == 2) groups[0].workspace.first_slot = 9;
            if (bad == 3) groups.push_back(groups[0]);
            llama_memory_workspace consumer(f.sched.get(), groups, {[] { return true; }, [] { return true; }});
            auto plan = f.target.plan;
            t.assert_true(!consumer.register_resources(plan, {10}));
            t.assert_true(plan.resources.empty());
        }
    });

    t.test("missing_execution_hooks_are_rejected", [](testing & t) {
        fixture f;
        llama_memory_workspace consumer(f.sched.get(), f.groups, {});
        t.assert_true(!consumer.register_resources(f.target.plan, {10}));
        std::unique_ptr<llama_memory_preparation> proposal;
        t.assert_true(!consumer.prepare(f.target, {}, proposal));
        t.assert_true(!proposal);
    });

    t.test("incorrect_workspace_grant_is_rejected_before_preparation", [](testing & t) {
        fixture f;
        if (!t.assert_true(f.workspace->register_resources(f.target.plan, {10, 20}))) return;
        f.target.plan.stages[0].requirements[0].size_min /= 2;
        f.target.plan.stages[0].requirements[0].size_preferred /= 2;
        t.assert_true(f.transition->prepare(f.target).status == status::consumer_failed);
        t.assert_equal(0, f.quiesces);
        t.assert_equal(size_t(0), ggml_backend_sched_get_buffer_size(f.sched.get(), f.first.get()));
    });

    t.test("invalidation_exception_remains_closed_and_recoverable", [](testing & t) {
        fixture f;
        if (!f.start(t)) return;
        f.move_workspace();
        f.throw_invalidate = true;
        t.assert_true(f.transition->prepare(f.target).status == status::prepared);
        const auto result = f.transition->activate(f.arenas);
        t.assert_true(result.status == status::consumer_exception && result.exception != nullptr);
        t.assert_true(!f.workspace->ready());
        f.throw_invalidate = false;
        t.assert_true(f.transition->recover().status == status::recovered);
        f.compute(t);
    });

    t.test("repeated_create_reserve_detach_releases_all_leases", [](testing & t) {
        for (int i = 0; i < 16; ++i) {
            fixture f(false);
            if (!f.start(t)) return;
            f.compute(t, size_t(i % 2));
            t.assert_true(f.workspace->close());
            for (auto & parent : f.parents) t.assert_equal(size_t(0), ggml_backend_memory_arena_lease_count(parent.get()));
        }
    });


    t.test("recovery_remaps_workspace_indices_when_budget_order_changes", [](testing & t) {
        fixture f(false);
        if (!f.start(t)) return;
        f.move_workspace();
        std::swap(f.target.budgets[0], f.target.budgets[1]);
        std::swap(f.arenas[0], f.arenas[1]);
        f.later.fail = true;
        t.assert_true(f.transition->prepare(f.target).status == status::prepared);
        t.assert_true(f.transition->activate(f.arenas).status == status::activation_failed);
        f.later.fail = false;
        if (!t.assert_true(f.transition->recover().status == status::recovered)) return;
        t.assert_true(f.workspace->ready());
        f.compute(t);
        f.compute(t, 1);
    });

    t.test("plans_aligned_workspace_bytes_per_phase_and_buffer_group", [](testing & t) {
        ggml_backend_buffer_type separate = *ggml_backend_cpu_buffer_type();
        std::vector<ggml_backend_buffer_type_t> bufts = {
            ggml_backend_cpu_buffer_type(),
            ggml_backend_cpu_buffer_type(),
            &separate,
        };
        const size_t measurements[] = {
            4097, 8192, 0,
            1024, 2049, 4096,
            0,    0,    1025,
        };
        llama_compute_workspace_plan plan;
        if (!t.assert_true(llama_compute_workspace_plan_make(
                bufts, {measurements, measurements + 9}, 3, plan))) {
            return;
        }
        if (!t.assert_equal(size_t(2), plan.groups.size()) ||
                !t.assert_equal(size_t(3), plan.phase_sizes.size())) {
            return;
        }
        t.assert_equal(size_t(8192), plan.groups[0].size);
        t.assert_equal(size_t(4096), plan.groups[1].size);
        t.assert_equal(size_t(8192), plan.phase_sizes[0][0]);
        t.assert_equal(size_t(0), plan.phase_sizes[0][1]);
        const auto aligned = [](size_t size, size_t alignment) {
            return (size + alignment - 1) / alignment * alignment;
        };
        t.assert_equal(aligned(2049, plan.groups[0].alignment), plan.phase_sizes[1][0]);
        t.assert_equal(size_t(4096), plan.phase_sizes[1][1]);
        t.assert_equal(size_t(0), plan.phase_sizes[2][0]);
        t.assert_equal(aligned(1025, plan.groups[1].alignment), plan.phase_sizes[2][1]);

        const auto before = plan.phase_sizes;
        t.assert_true(!llama_compute_workspace_plan_make(bufts, {measurements, measurements + 8}, 3, plan));
        t.assert_equal(before.size(), plan.phase_sizes.size());
        for (size_t phase = 0; phase < before.size(); ++phase) {
            t.assert_equal(before[phase].size(), plan.phase_sizes[phase].size());
            for (size_t group = 0; group < before[phase].size(); ++group) {
                t.assert_equal(before[phase][group], plan.phase_sizes[phase][group]);
            }
        }
    });

    t.test("registers_distinct_phase_requirements_and_budget_fit", [](testing & t) {
        fixture f;
        auto groups = f.groups;
        if (!t.assert_equal(size_t(1), groups.size())) {
            return;
        }
        groups[0].stages = {{10, 8192}, {20, 4096}};
        llama_memory_workspace workspace(
            f.sched.get(), groups, {[] { return true; }, [] { return true; }});
        auto plan = f.target.plan;
        if (!t.assert_true(workspace.register_resources(plan, {10, 20}))) {
            return;
        }
        t.assert_equal(size_t(8192), plan.stages[0].requirements[0].size_min);
        t.assert_equal(size_t(4096), plan.stages[1].requirements[0].size_min);

        const std::vector<llama_memory_arena_budget> budget = {{
            groups[0].resource.domain,
            groups[0].resource.allocation_class,
            4096,
            groups[0].workspace.alignment,
        }};
        llama_memory_layout layout;
        t.assert_true(llama_memory_layout_minimum(plan, 10, budget, {}, layout).status ==
            llama_memory_layout_status::placement_failed);
        t.assert_true(llama_memory_layout_minimum(plan, 20, budget, {}, layout).status ==
            llama_memory_layout_status::success);
        if (t.assert_equal(size_t(1), layout.arenas.size()) &&
                t.assert_equal(size_t(1), layout.arenas[0].regions.size())) {
            t.assert_equal(size_t(4096), layout.arenas[0].regions[0].size);
        }
    });

    t.test("larger_serial_grants_require_opt_in_and_recover_transactionally", [](testing & t) {
        fixture f;
        auto groups = f.groups;
        groups[0].allow_larger_grants = true;
        llama_memory_workspace workspace(f.sched.get(),groups,{[] { return true; },[] { return true; }});
        llama_memory_transition transition({&workspace,&f.later});
        auto target = f.target;
        if (!t.assert_true(workspace.register_resources(target.plan,{10,20}))) return;
        if (!t.assert_true(transition.prepare(target).status == status::prepared)) return;
        if (!t.assert_true(transition.activate(f.arenas).status == status::activated)) return;
        const auto original = groups[0].workspace.size;
        auto & requirement = target.plan.stages[0].requirements[0];
        requirement.size_min = requirement.size_preferred = original+1;
        t.assert_true(transition.prepare(target).status == status::consumer_failed);
        t.assert_equal(original,ggml_backend_sched_get_buffer_size(f.sched.get(),f.first.get()));
        requirement.size_min = requirement.size_preferred = original+groups[0].workspace.alignment;
        if (!t.assert_true(transition.prepare(target).status == status::prepared)) return;
        f.later.fail = true;
        t.assert_true(transition.activate(f.arenas).status == status::activation_failed);
        t.assert_true(transition.recover().status == status::recovered);
        t.assert_equal(original,ggml_backend_sched_get_buffer_size(f.sched.get(),f.first.get()));
        f.later.fail = false;
        if (!t.assert_true(transition.prepare(target).status == status::prepared)) return;
        t.assert_true(transition.activate(f.arenas).status == status::activated);
        t.assert_equal(requirement.size_min,ggml_backend_sched_get_buffer_size(f.sched.get(),f.first.get()));
        requirement.size_min = requirement.size_preferred = original-groups[0].workspace.alignment;
        t.assert_true(transition.prepare(target).status == status::consumer_failed);
    });

    t.test("zero_workspace_phase_is_valid_but_misaligned_phase_size_is_not", [](testing & t) {
        fixture f;
        auto groups = f.groups;
        groups[0].stages = {{10, groups[0].workspace.size}, {20, 0}};
        llama_memory_workspace valid(
            f.sched.get(), groups, {[] { return true; }, [] { return true; }});
        auto plan = f.target.plan;
        t.assert_true(valid.register_resources(plan, {10, 20}));
        t.assert_equal(size_t(0), plan.stages[1].requirements[0].size_min);

        groups[0].stages[1].size = groups[0].workspace.alignment + 1;
        llama_memory_workspace invalid(
            f.sched.get(), groups, {[] { return true; }, [] { return true; }});
        plan = f.target.plan;
        t.assert_true(!invalid.register_resources(plan, {10, 20}));
        t.assert_true(plan.resources.empty());
    });

    t.test("zero_workspace_phase_detaches_and_can_return", [](testing & t) {
        fixture f;
        auto groups = f.groups;
        groups[0].stages = {{10, groups[0].workspace.size}, {20, 0}};
        llama_memory_workspace workspace(
            f.sched.get(), groups, {[] { return true; }, [] { return true; }});
        auto target = f.target;
        if (!t.assert_true(workspace.register_resources(target.plan, {10, 20}))) {
            return;
        }
        llama_memory_transition transition({&workspace});
        if (!t.assert_true(transition.prepare(target).status == status::prepared) ||
                !t.assert_true(transition.activate(f.arenas).status == status::activated)) {
            return;
        }
        t.assert_equal(groups[0].workspace.size,
            ggml_backend_sched_get_buffer_size(f.sched.get(), f.first.get()));

        target.stage = 20;
        if (!t.assert_true(transition.prepare(target).status == status::prepared) ||
                !t.assert_true(transition.activate(f.arenas).status == status::activated)) {
            return;
        }
        t.assert_true(workspace.ready());
        t.assert_equal(size_t(0), ggml_backend_sched_get_buffer_size(f.sched.get(), f.first.get()));
        t.assert_equal(size_t(0), ggml_backend_memory_arena_lease_count(f.parents[0].get()));

        target.stage = 10;
        if (!t.assert_true(transition.prepare(target).status == status::prepared) ||
                !t.assert_true(transition.activate(f.arenas).status == status::activated)) {
            return;
        }
        f.compute(t);
    });

    t.test("late_attachment_failure_clears_partial_bindings", [](testing & t) {
        fixture f(false);
        arena_ptr foreign(ggml_backend_memory_arena_new(f.groups[1].workspace.buft,8192),ggml_backend_memory_arena_free);
        if (!t.assert_true(bool(foreign) && ggml_backend_memory_arena_begin(foreign.get(),0)) ||
                !t.assert_true(ggml_backend_memory_arena_reserve_at(foreign.get(),99,0,8192,
                    f.groups[1].workspace.alignment,0,nullptr)) ||
                !t.assert_true(ggml_backend_memory_arena_commit(foreign.get()))) return;
        auto * lease=ggml_backend_memory_arena_acquire(foreign.get(),99);
        if (!t.assert_true(lease != nullptr)) return;
        const bool attached=ggml_backend_sched_attach_memory_lease(f.sched.get(),f.second.get(),lease);
        ggml_backend_memory_lease_free(lease);
        if (!t.assert_true(attached)) return;
        if (!t.assert_true(f.workspace->register_resources(f.target.plan,{10,20})) ||
                !t.assert_true(f.transition->prepare(f.target).status == status::prepared)) return;
        t.assert_true(f.transition->activate(f.arenas).status == status::activation_failed);
        t.assert_true(f.workspace->leases().empty());
        t.assert_equal(size_t(0),ggml_backend_sched_get_buffer_size(f.sched.get(),f.first.get()));
        t.assert_true(ggml_backend_sched_detach_memory_lease(f.sched.get(),f.second.get()));
    });
    return t.summary();
}
