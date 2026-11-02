#include "../tools/mtmd/mtmd-workspace.h"
#include "../tools/mtmd/mtmd-embeddings.h"
#include "../tools/mtmd/mtmd-projector-storage.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "../ggml/src/ggml-backend-operator-workspace.h"
#include "../src/llama-context-memory.h"
#include "testing.h"
#include "ggml-cpu.h"
#include <cmath>
#include <stdexcept>
#include <cstring>
#include <tuple>
#if defined(__linux__)
#include <dlfcn.h>
#endif

using arena_ptr = std::unique_ptr<ggml_backend_memory_arena, decltype(&ggml_backend_memory_arena_free)>;
using lease_ptr = std::unique_ptr<ggml_backend_memory_lease, decltype(&ggml_backend_memory_lease_free)>;

struct workspace_graph {
    ggml_context_ptr ctx{ggml_init({1024*1024,nullptr,true})};
    ggml_tensor * input;
    ggml_tensor * output;
    ggml_cgraph * graph;
    explicit workspace_graph(size_t rows) {
        input=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,64,rows);
        ggml_set_input(input);
        output=ggml_scale(ctx.get(),input,2);
        ggml_set_output(output);
        graph=ggml_new_graph_custom(ctx.get(),64,false);
        ggml_build_forward_expand(graph,output);
    }
};

struct workspace_grant {
    arena_ptr arena{nullptr,ggml_backend_memory_arena_free};
    lease_ptr lease{nullptr,ggml_backend_memory_lease_free};
    workspace_grant(ggml_backend_buffer_type_t type,size_t bytes,uint64_t id=11) {
        const size_t alignment=ggml_backend_buft_get_alignment(type);
        arena.reset(ggml_backend_memory_arena_new(type,bytes+2*alignment));
        GGML_ASSERT(arena && ggml_backend_memory_arena_begin(arena.get(),0));
        GGML_ASSERT(ggml_backend_memory_arena_reserve_at(arena.get(),id,alignment,bytes,alignment,0,nullptr));
        GGML_ASSERT(ggml_backend_memory_arena_commit(arena.get()));
        lease.reset(ggml_backend_memory_arena_acquire(arena.get(),id));
        GGML_ASSERT(lease);
    }
};

struct workspace_fixture {
    ggml_backend_ptr first{ggml_backend_cpu_init()}, second{ggml_backend_cpu_init()};
    std::vector<ggml_backend_t> backends{first.get(),second.get()};
    std::vector<ggml_backend_buffer_type_t> types{ggml_backend_cpu_buffer_type(),ggml_backend_cpu_buffer_type()};
    ggml_backend_sched_ptr sched{ggml_backend_sched_new(backends.data(),types.data(),2,256,false,true)};
    mtmd_compute_workspace workspace{sched.get(),backends,256};
};

struct synchronization_probe {
    inline static synchronization_probe * active=nullptr;
    ggml_backend_t backend;
    decltype(ggml_backend_i::synchronize) original;
    ggml_backend_memory_arena_t arena;
    mtmd_compute_workspace & owner;
    std::vector<size_t> leases;
    bool reentrant_release=false;
    synchronization_probe(ggml_backend_t backend,ggml_backend_memory_arena_t arena,mtmd_compute_workspace & owner) :
        backend(backend),original(backend->iface.synchronize),arena(arena),owner(owner) {
        GGML_ASSERT(!active); active=this;
        backend->iface.synchronize=[](ggml_backend_t backend) {
            active->leases.push_back(ggml_backend_memory_arena_lease_count(active->arena));
            active->reentrant_release |= active->owner.release();
            if (active->original) active->original(backend);
        };
    }
    ~synchronization_probe() { backend->iface.synchronize=original; active=nullptr; }
};

struct queued_workspace_failure {
    inline static queued_workspace_failure * active=nullptr;
    ggml_backend_t backend;
    decltype(ggml_backend_i::graph_compute) original;
    size_t calls=0;
    explicit queued_workspace_failure(ggml_backend_t backend) : backend(backend),original(backend->iface.graph_compute) {
        GGML_ASSERT(!active);active=this;
        backend->iface.graph_compute=[](ggml_backend_t backend,ggml_cgraph * graph) {
            const auto result=active->original(backend,graph);
            if (result!=GGML_STATUS_SUCCESS) return result;
            ++active->calls;return GGML_STATUS_FAILED;
        };
    }
    ~queued_workspace_failure() { backend->iface.graph_compute=original;active=nullptr; }
};
struct workspace_sync_failure {
    inline static workspace_sync_failure * active=nullptr;
    ggml_backend_t backend;
    decltype(ggml_backend_i::synchronize) original;
    size_t calls=0;
    explicit workspace_sync_failure(ggml_backend_t backend) : backend(backend),original(backend->iface.synchronize) {
        GGML_ASSERT(!active);active=this;
        backend->iface.synchronize=[](ggml_backend_t) { ++active->calls;throw std::runtime_error("vision completion failure"); };
    }
    ~workspace_sync_failure() { backend->iface.synchronize=original;active=nullptr; }
};

static void verify_graph(testing & t, mtmd_compute_workspace & owner, workspace_graph & g) {
    const size_t elements=ggml_nelements(g.input);
    std::vector<float> input(elements,3),output(elements,0);
    ggml_backend_tensor_set(g.input,input.data(),0,elements*sizeof(float));
    if (!t.assert_equal(GGML_STATUS_SUCCESS,owner.compute_async(g.graph)) || !t.assert_true(owner.drain())) return;
    ggml_backend_tensor_get(g.output,output.data(),0,elements*sizeof(float));
    t.assert_true(std::all_of(output.begin(),output.end(),[](float value) { return value == 6; }));
}

int main(int argc,char ** argv) {
    const char * model_path=nullptr;
    const char * projector_path=nullptr;
    bool cuda=false,operator_failures=false;
    for (int i=1;i<argc;++i) {
        if (!std::strcmp(argv[i],"--cuda")) cuda=true;
        else if (!std::strcmp(argv[i],"--cuda-operator-failures")) { cuda=true;operator_failures=true; }
        else if (!std::strcmp(argv[i],"--model") && i+1<argc) model_path=argv[++i];
        else if (!std::strcmp(argv[i],"--mmproj") && i+1<argc) projector_path=argv[++i];
        else return 2;
    }
#if !defined(__linux__)
    if (operator_failures) return 2;
#endif
    testing t;
    t.test("compute_root_lease_exposes_its_retained_physical_parent",[](testing & t) {
        workspace_grant grant(ggml_backend_cpu_buffer_type(),1024);
        auto * expected=ggml_backend_memory_arena_parent(grant.arena.get());
        auto * parent=ggml_backend_memory_lease_parent(grant.lease.get());
        t.assert_true(parent==expected && ggml_backend_memory_lease_parent(nullptr)==nullptr);
        grant.arena.reset();
        t.assert_true(ggml_backend_memory_lease_parent(grant.lease.get())==expected);
        t.assert_true(ggml_backend_buffer_get_base(parent)!=nullptr && ggml_backend_buffer_get_size(parent)>=1024);
    });
    if (cuda) t.test("cuda_vision_graph_without_operator_scratch_needs_no_extra_bank",[](testing & t) {
        ggml_backend_load_all();auto * dev=ggml_backend_dev_by_name("CUDA0");
        if (!t.assert_true(dev!=nullptr)) return;
        ggml_backend_ptr gpu(ggml_backend_dev_init(dev,nullptr)),cpu(ggml_backend_cpu_init());auto * reg=ggml_backend_dev_backend_reg(dev);
        using type_t=ggml_backend_buffer_type_t (*)(int);
        auto type=reinterpret_cast<type_t>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_cuda_device_buffer_type"));
        auto release=reinterpret_cast<ggml_backend_operator_workspace_release_t>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_operator_workspace_release"));
        if (!t.assert_true(gpu && type && release)) return;
        std::vector<ggml_backend_t> backends{gpu.get(),cpu.get()};auto * buft=type(0);
        ggml_backend_buffer_type_t types[]={buft,ggml_backend_cpu_buffer_type()};
        ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends.data(),types,2,128,false,true));
        mtmd_compute_workspace owner(sched.get(),backends,128);
        ggml_context_ptr inputs(ggml_init({16384,nullptr,true})),ctx(ggml_init({16384,nullptr,true}));
        auto * source=ggml_new_tensor_2d(inputs.get(),GGML_TYPE_F32,64,8);ggml_set_input(source);
        ggml_backend_buffer_ptr source_buffer(ggml_backend_alloc_ctx_tensors_from_buft(inputs.get(),buft));
        if (!t.assert_true(bool(source_buffer))) return;
        auto * result=ggml_scale(ctx.get(),source,2);ggml_set_output(result);
        auto * graph=ggml_new_graph_custom(ctx.get(),32,false);ggml_build_forward_expand(graph,result);
        std::vector<ggml_backend_memory_workspace_group> groups;mtmd_workspace_details detail;
        if (!t.assert_true(owner.supported() && owner.measure(graph,groups) && groups.size()==1 && owner.details(buft,detail))) return;
        t.assert_true(detail.total_bytes==detail.graph_bytes && !detail.operator_bytes && !detail.library_bytes && !detail.padding_bytes);
        workspace_grant grant(buft,detail.total_bytes);
        if (!t.assert_true(owner.attach({grant.lease.get()}) && owner.alloc_graph(graph))) return;
        grant.lease.reset();std::vector<float> input(512,3),output(512);
        ggml_backend_tensor_set(source,input.data(),0,input.size()*sizeof(float));
        t.assert_equal(GGML_STATUS_SUCCESS,owner.compute_async(graph));
        t.assert_true(release(gpu.get())==ggml_backend_operator_workspace_status::busy);
        t.assert_equal(size_t(1),ggml_backend_memory_arena_lease_count(grant.arena.get()));
        t.assert_true(owner.drain());ggml_backend_tensor_get(result,output.data(),0,output.size()*sizeof(float));
        t.assert_true(std::all_of(output.begin(),output.end(),[](float value) { return value==6; }));
        t.assert_true(owner.release());t.assert_equal(size_t(0),ggml_backend_memory_arena_lease_count(grant.arena.get()));
        t.out << "vision-zero-operators: graph-only bytes, root completion pin, no extra bank\n";
    });
    if (cuda) t.test("cuda_vision_operator_bank_uses_the_same_root_and_preserves_native_results",[&](testing & t) {
        ggml_backend_load_all();auto * dev=ggml_backend_dev_by_name("CUDA0");
        if (!t.assert_true(dev!=nullptr)) return;
        ggml_backend_ptr gpu(ggml_backend_dev_init(dev,nullptr)),cpu(ggml_backend_cpu_init());
        auto * reg=ggml_backend_dev_backend_reg(dev);
        using type_t=ggml_backend_buffer_type_t (*)(int);
        auto type=reinterpret_cast<type_t>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_cuda_device_buffer_type"));
        if (!t.assert_true(gpu && type)) return;
        ggml_context_ptr weights_ctx(ggml_init({16384,nullptr,true}));
        auto * weights=ggml_new_tensor_2d(weights_ctx.get(),GGML_TYPE_F16,64,64);
        ggml_backend_buffer_ptr weights_buffer(ggml_backend_alloc_ctx_tensors_from_buft(weights_ctx.get(),type(0)));
        if (!t.assert_true(bool(weights_buffer))) return;
        ggml_backend_buffer_set_usage(weights_buffer.get(),GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        std::vector<ggml_fp16_t> a(64*64);std::vector<float> input(64*128),native(64*128),actual(native.size());
        for (size_t i=0;i<a.size();++i) a[i]=ggml_fp32_to_fp16(float(int(i%17)-8)/16);
        for (size_t i=0;i<input.size();++i) input[i]=float(int(i%19)-9)/32;
        ggml_backend_tensor_set(weights,a.data(),0,ggml_nbytes(weights));
        const auto build=[&](ggml_context * ctx) {
            auto * source=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,64,128);ggml_set_input(source);
            auto * output=ggml_mul_mat(ctx,weights,source);ggml_set_output(output);
            auto * graph=ggml_new_graph_custom(ctx,32,false);ggml_build_forward_expand(graph,output);
            return std::make_tuple(graph,source,output);
        };
        ggml_context_ptr native_ctx(ggml_init({65536,nullptr,true}));auto reference=build(native_ctx.get());
        ggml_backend_buffer_ptr native_buffer(ggml_backend_alloc_ctx_tensors(native_ctx.get(),gpu.get()));
        if (!t.assert_true(bool(native_buffer))) return;
        ggml_backend_tensor_set(std::get<1>(reference),input.data(),0,input.size()*sizeof(float));
        if (!t.assert_true(ggml_backend_graph_compute(gpu.get(),std::get<0>(reference))==GGML_STATUS_SUCCESS)) return;
        ggml_backend_synchronize(gpu.get());ggml_backend_tensor_get(std::get<2>(reference),native.data(),0,native.size()*sizeof(float));
        if (!t.assert_true(std::all_of(native.begin(),native.end(),[](float x) { return std::isfinite(x); }) &&
                std::any_of(native.begin(),native.end(),[](float x) { return x!=0; }))) return;
        t.out << "vision-native-control: finite nonzero CUDA reference ready\n";
        using faults_t=void (*)(int,bool,bool);using hits_t=int (*)();
        faults_t faults=nullptr;hits_t hits=nullptr;
#if defined(__linux__)
        if (operator_failures) {
            faults=reinterpret_cast<faults_t>(dlsym(RTLD_DEFAULT,"ggml_test_cublas_workspace_faults"));
            hits=reinterpret_cast<hits_t>(dlsym(RTLD_DEFAULT,"ggml_test_cublas_workspace_hits"));
            if (!t.assert_true(faults && hits)) return;
        }
#endif
        for (int mode=0;mode<(operator_failures ? 7 : 1);++mode) {
            if (faults) faults(0,false,false);
            std::vector<ggml_backend_t> backends{gpu.get(),cpu.get()};
            ggml_backend_buffer_type_t types[]={type(0),ggml_backend_cpu_buffer_type()};
            ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends.data(),types,2,128,false,true));
            mtmd_compute_workspace owner(sched.get(),backends,128);
            ggml_context_ptr ctx(ggml_init({65536,nullptr,true}));auto graph=build(ctx.get());
            std::vector<ggml_backend_memory_workspace_group> groups;
            if (!t.assert_true(owner.supported() && owner.measure(std::get<0>(graph),groups))) return;
            mtmd_workspace_details detail;
            auto device_group=std::find_if(groups.begin(),groups.end(),[&](const auto & group) { return group.buft==type(0); });
            if (!t.assert_true(device_group!=groups.end() && owner.details(type(0),detail) && detail.total_bytes==device_group->size &&
                    detail.graph_bytes>0 && detail.operator_bytes>0 && detail.library_bytes>0 &&
                    detail.total_bytes==detail.graph_bytes+detail.operator_bytes+detail.library_bytes+detail.padding_bytes)) return;
            t.assert_equal(size_t(128),detail.padding_bytes);
            workspace_grant grant(type(0),detail.total_bytes,91);
            std::vector<std::unique_ptr<workspace_grant>> host;
            const auto attach=[&](const std::vector<ggml_backend_memory_workspace_group> & plan,ggml_backend_memory_lease_t device_lease) {
                host.clear();std::vector<ggml_backend_memory_lease_t> leases;
                for (size_t i=0;i<plan.size();++i) {
                    if (plan[i].buft==type(0)) leases.push_back(device_lease);
                    else {
                        host.push_back(std::make_unique<workspace_grant>(plan[i].buft,plan[i].size,101+i));
                        leases.push_back(host.back()->lease.get());
                    }
                }
                const bool result=owner.attach(leases);
                for (auto & entry : host) entry->lease.reset();
                return result;
            };
            if (mode==0) {
                workspace_grant short_grant(type(0),detail.total_bytes-1,92);
                t.assert_true(!attach(groups,short_grant.lease.get()));
            }
            t.out << "vision-workspace-quote: graph=" << detail.graph_bytes << " operator=" << detail.operator_bytes
                << " library=" << detail.library_bytes << " padding=" << detail.padding_bytes << " root_residue="
                << reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(ggml_backend_memory_lease_buffer(grant.lease.get())))%256 << '\n';
            if (!t.assert_true(attach(groups,grant.lease.get()))) return;
            grant.lease.reset();
            if (mode==1 || mode==2) faults(1,true,mode==2);
            const bool allocated=owner.alloc_graph(std::get<0>(graph));
            if (mode==1 || mode==2) {
                t.assert_true(!allocated && !owner.ready());t.assert_equal(1,hits());
                t.assert_equal(size_t(1),ggml_backend_memory_arena_lease_count(grant.arena.get()));
                t.assert_equal(GGML_STATUS_FAILED,owner.compute_async(std::get<0>(graph)));
            } else {
                if (!t.assert_true(allocated && owner.ready())) return;
                t.assert_equal(detail.graph_bytes,ggml_backend_sched_get_buffer_size(sched.get(),gpu.get()));
                ggml_backend_tensor_set(std::get<1>(graph),input.data(),0,input.size()*sizeof(float));
                if (mode==5) {
                    queued_workspace_failure failure(gpu.get());
                    t.assert_equal(GGML_STATUS_FAILED,owner.compute_async(std::get<0>(graph)));t.assert_equal(size_t(1),failure.calls);
                    t.assert_true(!owner.ready());
                } else t.assert_equal(GGML_STATUS_SUCCESS,owner.compute_async(std::get<0>(graph)));
                if (mode==6) {
                    workspace_sync_failure failure(gpu.get());
                    t.assert_true(!owner.release() && !owner.ready());t.assert_equal(size_t(1),failure.calls);
                    t.assert_equal(size_t(1),ggml_backend_memory_arena_lease_count(grant.arena.get()));
                }
                if (!t.assert_true(owner.drain())) return;
                ggml_backend_tensor_get(std::get<2>(graph),actual.data(),0,actual.size()*sizeof(float));
                t.assert_true(std::memcmp(native.data(),actual.data(),native.size()*sizeof(float))==0);
                if (mode==3 || mode==4) {
                    faults(1,false,mode==4);
                    t.assert_true(!owner.release() && !owner.ready());t.assert_equal(1,hits());
                    t.assert_equal(size_t(1),ggml_backend_memory_arena_lease_count(grant.arena.get()));
                    t.assert_equal(GGML_STATUS_FAILED,owner.compute_async(std::get<0>(graph)));
                }
            }
            if (faults) faults(0,false,false);
            if (!t.assert_true(owner.release())) return;
            t.assert_equal(size_t(0),ggml_backend_memory_arena_lease_count(grant.arena.get()));
            ggml_context_ptr retry_ctx(ggml_init({65536,nullptr,true}));auto retry=build(retry_ctx.get());
            if (!t.assert_true(owner.measure(std::get<0>(retry),groups))) return;
            grant.lease.reset(ggml_backend_memory_arena_acquire(grant.arena.get(),91));
            if (!t.assert_true(attach(groups,grant.lease.get()) && owner.alloc_graph(std::get<0>(retry)))) return;
            grant.lease.reset();ggml_backend_tensor_set(std::get<1>(retry),input.data(),0,input.size()*sizeof(float));
            t.assert_equal(GGML_STATUS_SUCCESS,owner.compute_async(std::get<0>(retry)));t.assert_true(owner.drain());
            ggml_backend_tensor_get(std::get<2>(retry),actual.data(),0,actual.size()*sizeof(float));
            t.assert_true(std::memcmp(native.data(),actual.data(),native.size()*sizeof(float))==0);
            t.assert_true(owner.release());t.assert_equal(size_t(0),ggml_backend_memory_arena_lease_count(grant.arena.get()));
        }
        t.out << "vision-operator-bridge: native-exact root limit, failure pins and recovery\n";
    });
    t.test("vision_capacity_shapes_follow_native_alignment_and_reject_bad_bounds",[](testing & t) {
        for (bool gemma : {false,true}) {
            mtmd_vision_geometry geometry;
            geometry.family=gemma ? "gemma4" : "qwen3vl";
            geometry.patch_size=16; geometry.merge=gemma ? 3 : 2;
            geometry.min_tokens=64; geometry.max_tokens=gemma ? 1120 : 4096;
            mtmd_vision_shape shape;
            if (!t.assert_true(mtmd_vision_shape_for_budget(geometry,1120,16,9,shape))) return;
            t.assert_equal(gemma ? 2112 : 1408,shape.width);
            t.assert_equal(gemma ? 1200 : 800,shape.height);
            t.assert_equal(size_t(1100),shape.tokens);
            t.assert_true(shape.input_width*9==shape.input_height*16);
            mtmd_vision_shape portrait;
            t.assert_true(mtmd_vision_shape_for_budget(geometry,1120,9,16,portrait));
            t.assert_equal(shape.width,portrait.height);t.assert_equal(shape.height,portrait.width);
            for (int change=0;change<6;++change) {
                auto bad=geometry;int budget=1120,x=1,y=1;
                switch (change) {
                    case 0:budget=0;break;
                    case 1:x=0;break;
                    case 2:bad.merge=0;break;
                    case 3:bad.max_tokens=INT32_MAX;break;
                    case 4:bad.family="unknown";break;
                    case 5:budget=geometry.max_tokens+1;break;
                }
                mtmd_vision_shape unchanged;unchanged.width=123;
                t.assert_true(!mtmd_vision_shape_for_budget(bad,budget,x,y,unchanged));
                t.assert_equal(123,unchanged.width);
            }
        }
    });

#if !defined(_WIN32) || !defined(LLAMA_SHARED)
    t.test("vision_borrows_serial_target_scratch_and_returns_it_on_release", [](testing & t) {
        workspace_fixture f;
        workspace_graph graph(8);
        std::vector<ggml_backend_memory_workspace_group> groups;
        if (!t.assert_true(f.workspace.measure(graph.graph, groups))) return;
        ggml_backend_ptr target_backend(ggml_backend_cpu_init());
        ggml_backend_t backends[] = {target_backend.get()};
        ggml_backend_buffer_type_t types[] = {ggml_backend_cpu_buffer_type()};
        ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, types, 1, 256, false, true));
        {
            auto small = llama_context_memory::create(sched.get(), {target_backend.get()},
                {{types[0], groups[0].alignment, groups[0].alignment, 0}});
            if (!t.assert_true(bool(small))) return;
            t.assert_true(!f.workspace.borrow(*small));
            t.assert_true(small->serial_ready());
            t.assert_true(!f.workspace.ready());
        }
        auto parent = llama_context_memory::create(sched.get(), {target_backend.get()},
            {{types[0], groups[0].size, groups[0].alignment, 0}});
        if (!t.assert_true(bool(parent) && f.workspace.borrow(*parent))) return;
        if (!t.assert_true(f.workspace.alloc_graph(graph.graph))) return;
        const auto first = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(parent->shared_parent()));
        const auto input = reinterpret_cast<uintptr_t>(graph.input->data);
        t.assert_true(input >= first && input + ggml_nbytes(graph.input) <= first + groups[0].size);
        verify_graph(t, f.workspace, graph);
        t.assert_true(f.workspace.release());
        t.assert_true(parent->prepare_serial_target());
        t.assert_true(!f.workspace.ready());
        t.assert_true(f.workspace.compute_async(graph.graph) == GGML_STATUS_FAILED);
        workspace_graph next(8);
        t.assert_true(f.workspace.measure(next.graph, groups));
        t.assert_true(f.workspace.borrow(*parent));
        t.assert_true(f.workspace.retire_graph());
        t.assert_true(f.workspace.alloc_graph(next.graph));
        verify_graph(t, f.workspace, next);
        parent.reset();
        t.assert_true(!f.workspace.ready());
        t.assert_true(f.workspace.compute_async(next.graph) == GGML_STATUS_FAILED);
        t.assert_true(f.workspace.release());
    });
#endif
    t.test("measures_actual_batch_without_allocating_and_normalizes_aliases", [](testing & t) {
        workspace_fixture f;
        if (!t.assert_true(f.workspace.supported())) return;
        std::vector<ggml_backend_memory_workspace_group> small,large;
        workspace_graph a(8),b(64);
        if (!t.assert_true(f.workspace.measure(a.graph,small)) || !t.assert_true(f.workspace.measure(b.graph,large))) return;
        t.assert_equal(size_t(1),small.size());
        t.assert_equal(size_t(1),large.size());
        t.assert_true(large[0].size > small[0].size);
        t.assert_equal(size_t(0),ggml_backend_sched_get_buffer_size(f.sched.get(),f.first.get()));
        t.assert_equal(size_t(0),ggml_backend_sched_get_buffer_size(f.sched.get(),f.second.get()));
    });
    t.test("borrowed_graph_executes_and_release_returns_the_lease", [](testing & t) {
        workspace_fixture f;
        workspace_graph g(16);
        std::vector<ggml_backend_memory_workspace_group> groups;
        if (!t.assert_true(f.workspace.measure(g.graph,groups))) return;
        workspace_grant grant(groups[0].buft,groups[0].size);
        if (!t.assert_true(f.workspace.attach({grant.lease.get()}))) return;
        grant.lease.reset();
        t.assert_equal(size_t(1),ggml_backend_memory_arena_lease_count(grant.arena.get()));
        if (!t.assert_true(f.workspace.alloc_graph(g.graph))) return;
        verify_graph(t,f.workspace,g);
        t.assert_true(f.workspace.release());
        t.assert_true(!f.workspace.ready());
        t.assert_equal(size_t(0),ggml_backend_memory_arena_lease_count(grant.arena.get()));
        t.assert_equal(size_t(0),ggml_backend_sched_get_buffer_size(f.sched.get(),f.first.get()));
        t.assert_true(f.workspace.release());
    });
    t.test("measurement_restores_graph_metadata_after_backend_optimization", [](testing & t) {
        workspace_fixture f;
        workspace_graph g(16);
        const auto before=*g.output;
        f.first->iface.graph_optimize=[](ggml_backend_t,ggml_cgraph * graph,ggml_backend_graph_optimize_params *) {
            ggml_graph_node(graph,ggml_graph_n_nodes(graph)-1)->op_params[0]=0;
        };
        std::vector<ggml_backend_memory_workspace_group> groups;
        if (!t.assert_true(f.workspace.measure(g.graph,groups))) return;
        t.assert_equal(0,std::memcmp(&before,g.output,sizeof(before)));
        f.first->iface.graph_optimize=nullptr;
        workspace_grant grant(groups[0].buft,groups[0].size);
        if (!t.assert_true(f.workspace.attach({grant.lease.get()}) && f.workspace.alloc_graph(g.graph))) return;
        verify_graph(t,f.workspace,g);
    });
    t.test("invalid_grant_preserves_attachment_and_larger_graph_is_bounded", [](testing & t) {
        workspace_fixture f;
        workspace_graph small(8),large(1024);
        std::vector<ggml_backend_memory_workspace_group> groups;
        if (!t.assert_true(f.workspace.measure(small.graph,groups))) return;
        workspace_grant grant(groups[0].buft,groups[0].size),tiny(groups[0].buft,groups[0].alignment);
        if (!t.assert_true(f.workspace.attach({grant.lease.get()}))) return;
        t.assert_true(!f.workspace.attach({tiny.lease.get()}));
        t.assert_true(!f.workspace.attach({nullptr}));
        t.assert_true(f.workspace.ready());
        t.assert_true(!f.workspace.alloc_graph(large.graph));
        t.assert_equal(groups[0].size,ggml_backend_sched_get_buffer_size(f.sched.get(),f.first.get()));
        if (!t.assert_true(f.workspace.alloc_graph(small.graph))) return;
        verify_graph(t,f.workspace,small);
    });
    t.test("rejects_invalid_graphs_and_requires_explicit_release_before_measure", [](testing & t) {
        workspace_fixture f;
        workspace_graph g(8);
        std::vector<ggml_backend_memory_workspace_group> groups;
        t.assert_true(!f.workspace.measure(nullptr,groups));
        t.assert_true(groups.empty());
        t.assert_true(!f.workspace.attach({}));
        if (!t.assert_true(f.workspace.measure(g.graph,groups))) return;
        workspace_grant grant(groups[0].buft,groups[0].size);
        if (!t.assert_true(f.workspace.attach({grant.lease.get()}))) return;
        const auto before=groups[0].size;
        t.assert_true(!f.workspace.measure(g.graph,groups));
        t.assert_equal(before,groups[0].size);
        t.assert_true(f.workspace.release());
        t.assert_true(f.workspace.measure(g.graph,groups));
    });
    t.test("attachment_failure_returns_our_lease_and_preserves_foreign_owner", [](testing & t) {
        workspace_fixture f;
        workspace_graph g(16);
        std::vector<ggml_backend_memory_workspace_group> groups;
        if (!t.assert_true(f.workspace.measure(g.graph,groups))) return;
        workspace_grant a(groups[0].buft,groups[0].size,11),foreign(groups[0].buft,groups[0].size,99);
        if (!t.assert_true(ggml_backend_sched_attach_memory_lease(f.sched.get(),f.first.get(),foreign.lease.get()))) return;
        t.assert_true(!f.workspace.attach({a.lease.get()}));
        t.assert_true(!f.workspace.ready());
        a.lease.reset(); foreign.lease.reset();
        t.assert_equal(size_t(0),ggml_backend_memory_arena_lease_count(a.arena.get()));
        t.assert_equal(size_t(1),ggml_backend_memory_arena_lease_count(foreign.arena.get()));
        t.assert_true(ggml_backend_sched_detach_memory_lease(f.sched.get(),f.first.get()));
    });
    t.test("release_drains_execution_before_returning_storage_and_rejects_reentry", [](testing & t) {
        workspace_fixture f;
        workspace_graph g(32);
        std::vector<ggml_backend_memory_workspace_group> groups;
        if (!t.assert_true(f.workspace.measure(g.graph,groups))) return;
        workspace_grant grant(groups[0].buft,groups[0].size);
        if (!t.assert_true(f.workspace.attach({grant.lease.get()}) && f.workspace.alloc_graph(g.graph))) return;
        grant.lease.reset();
        const std::vector<float> input(ggml_nelements(g.input),1);
        ggml_backend_tensor_set(g.input,input.data(),0,ggml_nbytes(g.input));
        if (!t.assert_equal(GGML_STATUS_SUCCESS,f.workspace.compute_async(g.graph))) return;
        synchronization_probe probe(f.first.get(),grant.arena.get(),f.workspace);
        t.assert_true(f.workspace.release());
        if (!t.assert_true(!probe.leases.empty())) return;
        t.assert_equal(size_t(1),probe.leases.front());
        t.assert_true(!probe.reentrant_release);
        t.assert_equal(size_t(0),ggml_backend_memory_arena_lease_count(grant.arena.get()));
        t.assert_equal(GGML_STATUS_FAILED,f.workspace.compute_async(g.graph));
    });
    t.test("rejects_wrong_type_and_count_before_changing_current_grants", [](testing & t) {
        workspace_fixture f;
        workspace_graph g(16);
        std::vector<ggml_backend_memory_workspace_group> groups;
        if (!t.assert_true(f.workspace.measure(g.graph,groups))) return;
        workspace_grant grant(groups[0].buft,groups[0].size);
        auto distinct=*ggml_backend_cpu_buffer_type();
        workspace_grant wrong(&distinct,groups[0].size);
        if (!t.assert_true(f.workspace.attach({grant.lease.get()}))) return;
        t.assert_true(!f.workspace.attach({wrong.lease.get()}));
        t.assert_true(!f.workspace.attach({grant.lease.get(),grant.lease.get()}));
        t.assert_true(f.workspace.ready());
        t.assert_true(f.workspace.alloc_graph(g.graph));
        verify_graph(t,f.workspace,g);
    });
    t.test("larger_grants_survive_graph_retirement_and_support_a_new_batch", [](testing & t) {
        workspace_fixture f;
        workspace_graph a(16),b(24);
        std::vector<ggml_backend_memory_workspace_group> groups;
        if (!t.assert_true(f.workspace.measure(a.graph,groups))) return;
        workspace_grant grant(groups[0].buft,2*groups[0].size);
        if (!t.assert_true(f.workspace.attach({grant.lease.get()}) && f.workspace.alloc_graph(a.graph))) return;
        verify_graph(t,f.workspace,a);
        t.assert_true(f.workspace.retire_graph());
        t.assert_true(f.workspace.ready());
        t.assert_equal(GGML_STATUS_FAILED,f.workspace.compute_async(a.graph));
        if (!t.assert_true(f.workspace.alloc_graph(b.graph))) return;
        verify_graph(t,f.workspace,b);
    });
    t.test("invalid_scheduler_and_excess_graph_metadata_are_rejected", [](testing & t) {
        mtmd_compute_workspace invalid(nullptr,{},0);
        std::vector<ggml_backend_memory_workspace_group> output;
        workspace_graph g(8);
        t.assert_true(!invalid.supported());
        t.assert_true(!invalid.measure(g.graph,output));
        t.assert_true(invalid.release());
        workspace_fixture f;
        for (int i=0;i<260;++i) g.output=ggml_scale(g.ctx.get(),g.output,1.1f);
        g.graph=ggml_new_graph_custom(g.ctx.get(),512,false);
        ggml_build_forward_expand(g.graph,g.output);
        t.assert_true(!f.workspace.measure(g.graph,output));
        t.assert_true(output.empty());
    });
    if (model_path && projector_path) t.test("real_vision_batches_match_legacy_with_borrowed_workspace", [&](testing & t) {
        auto log=[](ggml_log_level level,const char * text,void *) {
            if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) std::fputs(text,stderr);
        };
        llama_log_set(log,nullptr);
        mtmd_log_set(log,nullptr);
        ggml_backend_load_all();
        auto model_params=llama_model_default_params();
        model_params.no_alloc=true;
        model_params.n_gpu_layers=0;
        model_params.load_mode=LLAMA_LOAD_MODE_NONE;
        model_params.use_extra_bufts=false;
        llama_model_ptr model(llama_model_load_from_file(model_path,model_params));
        if (!t.assert_true(bool(model))) return;
        if (!t.assert_true(llama_model_n_embd_inp(model.get()) > 0)) return;
        auto params=mtmd_context_params_default();
        params.use_gpu=cuda; params.warmup=false; params.n_threads=2;
        params.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
        params.image_min_tokens=64; params.image_max_tokens=4096;
        mtmd::context_ptr stock(mtmd_init_from_file(projector_path,model.get(),params));
        mtmd::context_ptr borrowed(mtmd_init_from_file(projector_path,model.get(),params));
        if (!t.assert_true(bool(stock) && bool(borrowed))) return;
        auto * device=cuda ? ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) : nullptr;
        auto * compute_type=ggml_backend_cpu_buffer_type();
        if (cuda) {
            auto * reg=device ? ggml_backend_dev_backend_reg(device) : nullptr;
            using factory_t=ggml_backend_buffer_type_t (*)(int);
            auto factory=reg ? reinterpret_cast<factory_t>(ggml_backend_reg_get_proc_address(reg,
                "ggml_backend_cuda_device_buffer_type")) : nullptr;
            compute_type=nullptr;
            if (factory) for (size_t i=0;i<ggml_backend_reg_dev_count(reg);++i) {
                if (ggml_backend_reg_dev_get(reg,i) == device) { compute_type=factory(int(i)); break; }
            }
        }
        if (!t.assert_true(compute_type != nullptr)) return;
        std::vector<std::pair<mtmd_embedding_view, std::vector<float>>> retained_outputs;
        size_t previous_bytes=0;
        for (auto shape : {std::pair{320u,320u},std::pair{640u,384u},std::pair{1024u,768u},std::pair{1280u,1024u}}) {
            std::vector<uint8_t> pixels(size_t(shape.first)*shape.second*3);
            for (size_t i=0;i<pixels.size();++i) pixels[i]=uint8_t((i*13+i/101)%256);
            mtmd::bitmap_ptr bitmap(mtmd_bitmap_init(shape.first,shape.second,pixels.data()));
            mtmd::input_chunks_ptr chunks(mtmd_input_chunks_init());
            const auto * image=bitmap.get();
            const std::string prompt=mtmd_get_marker(stock.get());
            const mtmd_input_text input{prompt.c_str(),prompt.size(),true,true};
            if (!t.assert_equal(0,mtmd_tokenize(stock.get(),chunks.get(),&input,&image,1))) return;
            const mtmd_input_chunk * chunk=nullptr;
            for (size_t i=0;i<mtmd_input_chunks_size(chunks.get());++i) {
                auto * next=mtmd_input_chunks_get(chunks.get(),i);
                if (mtmd_input_chunk_get_type(next) == MTMD_INPUT_CHUNK_TYPE_IMAGE) { chunk=next; break; }
            }
            if (!t.assert_true(chunk != nullptr)) return;
            mtmd::batch_ptr reference(mtmd_batch_init(stock.get())),batch(mtmd_batch_init(borrowed.get()));
            if (!t.assert_equal(0,mtmd_batch_add_chunk(reference.get(),chunk)) ||
                    !t.assert_equal(0,mtmd_batch_add_chunk(batch.get(),chunk)) ||
                    !t.assert_equal(0,mtmd_batch_encode(reference.get()))) return;
            const size_t elements=mtmd_input_chunk_get_n_tokens(chunk)*llama_model_n_embd_inp(model.get());
            auto * expected_ptr=mtmd_batch_get_output_embd(reference.get(),chunk);
            if (!t.assert_true(expected_ptr != nullptr)) return;
            std::vector<float> expected(expected_ptr,expected_ptr+elements);
            std::vector<ggml_backend_memory_workspace_group> groups;
            if (!t.assert_true(mtmd_batch_measure_compute_workspace(batch.get(),groups,compute_type)) || !t.assert_true(!groups.empty())) return;
            if (!t.assert_true(groups[0].buft == compute_type)) return;
            size_t bytes=0;
            std::vector<std::unique_ptr<workspace_grant>> grants;
            std::vector<ggml_backend_memory_lease_t> leases;
            for (size_t i=0;i<groups.size();++i) {
                bytes+=groups[i].size;
                grants.push_back(std::make_unique<workspace_grant>(groups[i].buft,groups[i].size,11+i));
                leases.push_back(grants.back()->lease.get());
            }
            t.assert_true(bytes > previous_bytes);
            previous_bytes=bytes;
            auto sentinel=groups;
            t.assert_true(!mtmd_batch_measure_compute_workspace(nullptr,sentinel,compute_type));
            t.assert_equal(groups[0].size,sentinel[0].size);
            workspace_grant tiny(groups[0].buft,groups[0].alignment,91);
            auto invalid=leases; invalid[0]=tiny.lease.get();
            t.assert_true(!mtmd_attach_compute_workspace(borrowed.get(),invalid));
            if (!t.assert_true(mtmd_attach_compute_workspace(borrowed.get(),leases)) ||
                    !t.assert_equal(0,mtmd_batch_encode(batch.get()))) return;
            float * actual=mtmd_batch_get_output_embd(batch.get(),chunk);
            if (!t.assert_true(actual != nullptr)) return;
            mtmd_embedding_view retained;
            if (!t.assert_true(mtmd_batch_acquire_output_embd(batch.get(), chunk, retained))) return;
            t.assert_true(retained.data() == actual);
            t.assert_equal(mtmd_input_chunk_get_n_tokens(chunk), retained.n_tokens());
            t.assert_equal(size_t(llama_model_n_embd_inp(model.get())), retained.n_embd());
            mtmd::input_chunk_ptr duplicate(mtmd_input_chunk_copy(chunk));
            t.assert_equal(2, mtmd_batch_add_chunk(batch.get(), duplicate.get()));
            t.assert_true(mtmd_batch_get_output_embd(batch.get(), duplicate.get()) == nullptr);
            if (!t.assert_true(std::all_of(actual,actual+elements,[](float value) { return std::isfinite(value); }))) return;
            t.assert_true(std::memcmp(actual,expected.data(),elements*sizeof(float))==0);
            float error=0;
            for (size_t i=0;i<elements;++i) {
                error=std::max(error,std::abs(actual[i]-expected[i]));
            }
            t.out << "vision " << shape.first << 'x' << shape.second << ": workspace=" << bytes/1048576.0
                  << " MiB, max_error=" << error << '\n';
            t.assert_true(error <= 1e-5f);
            if (shape.first == 320) {
                std::vector<uint8_t> large_pixels(1280*1024*3,127);
                mtmd::bitmap_ptr large_bitmap(mtmd_bitmap_init(1280,1024,large_pixels.data()));
                const auto * large_image=large_bitmap.get();
                mtmd::input_chunks_ptr large_chunks(mtmd_input_chunks_init());
                if (!t.assert_equal(0,mtmd_tokenize(borrowed.get(),large_chunks.get(),&input,&large_image,1))) return;
                mtmd::batch_ptr too_large(mtmd_batch_init(borrowed.get()));
                for (size_t i=0;i<mtmd_input_chunks_size(large_chunks.get());++i) {
                    auto * next=mtmd_input_chunks_get(large_chunks.get(),i);
                    if (mtmd_input_chunk_get_type(next) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                        if (!t.assert_equal(0,mtmd_batch_add_chunk(too_large.get(),next))) return;
                    }
                }
                t.assert_true(mtmd_batch_encode(too_large.get()) != 0);
                if (!t.assert_equal(0,mtmd_batch_encode(batch.get()))) return;
                actual=mtmd_batch_get_output_embd(batch.get(),chunk);
                t.assert_true(retained.data() != actual);
            }
            if (shape.first == 1280) t.assert_true(mtmd_unload_projector_weights(borrowed.get()));
            else t.assert_true(mtmd_release_compute_workspace(borrowed.get()));
            for (auto & grant : grants) {
                grant->lease.reset();
                t.assert_equal(size_t(0),ggml_backend_memory_arena_lease_count(grant->arena.get()));
            }
            t.assert_true(std::equal(expected.begin(),expected.end(),actual,
                [](float a,float b) { return std::abs(a-b) <= 1e-5f; }));
            t.assert_true(mtmd_batch_encode(batch.get()) != 0);
            t.assert_true(mtmd_batch_get_output_embd(batch.get(), chunk) == actual);
            if (shape.first == 1280) {
                if (!t.assert_true(mtmd_reload_projector_weights(borrowed.get())) ||
                        !t.assert_equal(0,mtmd_batch_encode(batch.get()))) return;
                const auto * restored = mtmd_batch_get_output_embd(batch.get(),chunk);
                t.assert_true(std::equal(expected.begin(),expected.end(),restored,
                    [](float a,float b) { return std::abs(a-b) <= 1e-5f; }));
            }
            mtmd_batch_clear_output_embd(batch.get());
            t.assert_true(mtmd_batch_get_output_embd(batch.get(), chunk) == nullptr);
            batch.reset();
            chunks.reset();
            t.assert_true(std::equal(expected.begin(), expected.end(), retained.data()));
            retained_outputs.emplace_back(std::move(retained), std::move(expected));
        }
        borrowed.reset();
        stock.reset();
        model.reset();
        for (const auto & result : retained_outputs) {
            const auto & view = result.first;
            if (!t.assert_equal(result.second.size(), view.n_tokens() * view.n_embd())) return;
            for (size_t first = 0; first < view.n_tokens(); first += 13) {
                mtmd_embedding_view part;
                const size_t count = std::min(size_t(13), view.n_tokens() - first);
                if (!t.assert_true(view.slice(first, count, part))) return;
                t.assert_true(std::equal(part.data(), part.data() + count * part.n_embd(),
                    result.second.data() + first * part.n_embd()));
            }
        }
    });
    return t.summary();
}
