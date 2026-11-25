#include "../src/llama-memory-recurrent-spill.h"
#include "../src/llama-memory-recurrent.h"
#include "../src/llama-model.h"
#include "../src/llama-io.h"
#include "../ggml/src/ggml-backend-memory.h"
#include "testing.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "../ggml/src/ggml-backend-execution.h"
#include <cstring>
#include <array>
#include <algorithm>
#include <stdexcept>

// Fail after a real transfer, so cancellation must treat its destination as potentially modified.
struct recurrent_transfer_fault {
    inline static recurrent_transfer_fault * active = nullptr;
    ggml_backend_t backend;
    decltype(ggml_backend_i::get_tensor_async) get;
    decltype(ggml_backend_i::set_tensor_async) set;
    recurrent_transfer_fault(ggml_backend_t backend, bool upload) : backend(backend),
        get(backend->iface.get_tensor_async), set(backend->iface.set_tensor_async) {
        GGML_ASSERT(!active); active=this;
        if (upload) backend->iface.set_tensor_async=[](ggml_backend_t b,ggml_tensor * tensor,const void * data,size_t offset,size_t bytes) {
            if (active->set) active->set(b,tensor,data,offset,bytes);
            else ggml_backend_tensor_set(tensor,data,offset,bytes);
            throw std::runtime_error("after recurrent H2D submission");
        };
        else backend->iface.get_tensor_async=[](ggml_backend_t b,const ggml_tensor * tensor,void * data,size_t offset,size_t bytes) {
            if (active->get) active->get(b,tensor,data,offset,bytes);
            else ggml_backend_tensor_get(tensor,data,offset,bytes);
            throw std::runtime_error("after recurrent D2H submission");
        };
    }
    ~recurrent_transfer_fault() {
        backend->iface.get_tensor_async=get; backend->iface.set_tensor_async=set; active=nullptr;
    }
};

// Observe the bank's separately owned copy backend without adding a production test hook.
struct recurrent_copy_backend_probe {
    inline static recurrent_copy_backend_probe * active = nullptr;
    ggml_backend_dev_t device;
    decltype(ggml_backend_device_i::init_backend) original;
    ggml_backend_t copy = nullptr;
    decltype(ggml_backend_i::get_tensor_async) get = nullptr;
    decltype(ggml_backend_i::set_tensor_async) set = nullptr;
    decltype(ggml_backend_i::synchronize) sync = nullptr;
    decltype(ggml_backend_i::free) release = nullptr;
    struct read { const ggml_tensor * tensor; void * data; size_t offset,bytes; };
    std::vector<read> pending;
    struct write { ggml_tensor * tensor; const void * data; size_t offset,bytes; };
    std::vector<write> pending_uploads;
    bool deferred = false;
    bool fail_prepare = false;
    size_t preparations = 0;
    size_t downloads = 0;
    size_t uploads = 0;
    size_t fail_upload = 0;
    size_t backend_frees = 0;
    size_t fail_download = 0;
    bool observed = false;
    bool observed_uploads = false;
    bool deferred_uploads = false;
    bool watching_free = false;
    bool undrained_on_free = false;
    explicit recurrent_copy_backend_probe(ggml_backend_dev_t device) : device(device),original(device->iface.init_backend) {
        GGML_ASSERT(!active); active=this;
        device->iface.init_backend=[](ggml_backend_dev_t dev,const char * params) {
            auto * backend=active->copy=active->original(dev,params);
            if (backend) {
                active->sync=backend->iface.synchronize;
                backend->iface.synchronize=[](ggml_backend_t b) {
                    ++active->preparations;
                    if (active->fail_prepare) { active->fail_prepare=false; throw std::runtime_error("publication resource preparation failed"); }
                    if (active->sync) active->sync(b);
                };
            }
            return backend;
        };
    }
    void flush() {
        for (const auto & read : pending) ggml_backend_tensor_get(read.tensor,read.data,read.offset,read.bytes);
        pending.clear();
        for (const auto & write : pending_uploads) ggml_backend_tensor_set(write.tensor,write.data,write.offset,write.bytes);
        pending_uploads.clear();
    }
    // CPU-only deterministic queue: stage bytes are read only when its consumer is drained.
    void defer_downloads() {
        GGML_ASSERT(copy && !deferred);
        get=copy->iface.get_tensor_async; deferred=true;
        copy->iface.get_tensor_async=[](ggml_backend_t,const ggml_tensor * tensor,void * data,size_t offset,size_t bytes) {
            ++active->downloads;
            active->pending.push_back({tensor,data,offset,bytes});
            if (active->fail_download == active->downloads) throw std::runtime_error("after packed recurrent submission");
        };
        copy->iface.synchronize=[](ggml_backend_t) { active->flush(); };
        device->iface.init_backend=original;
    }
    void observe_downloads() {
        GGML_ASSERT(copy && !deferred && !observed);
        get=copy->iface.get_tensor_async; observed=true;
        copy->iface.get_tensor_async=[](ggml_backend_t backend,const ggml_tensor * tensor,void * data,size_t offset,size_t bytes) {
            ++active->downloads;
            if (active->get) active->get(backend,tensor,data,offset,bytes);
            else ggml_backend_tensor_get(tensor,data,offset,bytes);
        };
        device->iface.init_backend=original;
    }
    void observe_restores() {
        GGML_ASSERT(copy && !observed_uploads);
        set=copy->iface.set_tensor_async; observed_uploads=true;
        copy->iface.set_tensor_async=[](ggml_backend_t backend,ggml_tensor * tensor,const void * data,size_t offset,size_t bytes) {
            ++active->uploads;
            if (active->deferred_uploads) {
                active->pending_uploads.push_back({tensor,data,offset,bytes});
                if (active->uploads==active->fail_upload) throw std::runtime_error("after recurrent restore submission");
                return;
            }
            if (active->set) active->set(backend,tensor,data,offset,bytes);
            else ggml_backend_tensor_set(tensor,data,offset,bytes);
            if (active->uploads==active->fail_upload) throw std::runtime_error("after recurrent restore submission");
        };
        device->iface.init_backend=original;
    }
    void defer_restores() {
        observe_restores(); deferred_uploads=true;
        copy->iface.synchronize=[](ggml_backend_t) { active->flush(); };
    }
    void watch_teardown() {
        GGML_ASSERT(copy && !watching_free);
        release=copy->iface.free; watching_free=true;
        copy->iface.free=[](ggml_backend_t backend) {
            auto & p=*active;
            ++p.backend_frees;
            p.undrained_on_free=!p.pending.empty() || !p.pending_uploads.empty();
            p.release(backend);
            p.copy=nullptr;
            p.deferred=p.observed=p.observed_uploads=p.deferred_uploads=p.watching_free=false;
        };
    }
    ~recurrent_copy_backend_probe() {
        if (deferred_uploads) flush();
        if (deferred) { flush(); copy->iface.get_tensor_async=get; }
        if (observed) copy->iface.get_tensor_async=get;
        if (observed_uploads) copy->iface.set_tensor_async=set;
        if (watching_free) copy->iface.free=release;
        if (copy) copy->iface.synchronize=sync;
        device->iface.init_backend=original; active=nullptr;
    }
};

// Inject after a completed native byte copy, including partially restored candidate banks.
struct recurrent_buffer_transfer_fault {
    inline static recurrent_buffer_transfer_fault * active=nullptr;
    ggml_backend_buffer_ptr buffer;
    decltype(ggml_backend_buffer_i::get_tensor) get;
    decltype(ggml_backend_buffer_i::set_tensor) set;
    size_t calls=0,fail_at;
    recurrent_buffer_transfer_fault(ggml_backend_buffer_t source,bool upload,size_t fail_at) :
        buffer(ggml_backend_buffer_retain(source)),get(source->iface.get_tensor),set(source->iface.set_tensor),fail_at(fail_at) {
        GGML_ASSERT(!active);active=this;
        if (upload) source->iface.set_tensor=[](ggml_backend_buffer_t b,ggml_tensor * tensor,const void * data,size_t offset,size_t bytes) {
            active->set(b,tensor,data,offset,bytes);
            if (++active->calls==active->fail_at) throw std::runtime_error("after current-state upload");
        };
        else source->iface.get_tensor=[](ggml_backend_buffer_t b,const ggml_tensor * tensor,void * data,size_t offset,size_t bytes) {
            active->get(b,tensor,data,offset,bytes);
            if (++active->calls==active->fail_at) throw std::runtime_error("after current-state download");
        };
    }
    ~recurrent_buffer_transfer_fault() {
        buffer->iface.get_tensor=get;buffer->iface.set_tensor=set;active=nullptr;
    }
};

struct recurrent_stage_bind_fault {
    inline static recurrent_stage_bind_fault * active=nullptr;
    llama_memory_recurrent & memory;
    ggml_backend_buffer_type_t type;
    decltype(ggml_backend_buffer_type_i::get_alloc_size) allocate;
    bool fired=false;
    recurrent_stage_bind_fault(llama_memory_recurrent & memory,ggml_backend_buffer_type_t type) :
        memory(memory),type(type),allocate(type->iface.get_alloc_size) {
        GGML_ASSERT(!active);active=this;
        type->iface.get_alloc_size=[](ggml_backend_buffer_type_t type,const ggml_tensor * tensor) -> size_t {
            const auto * first=active->memory.spill_bank(0),* second=active->memory.spill_bank(1);
            if (!active->fired && first && second && first->staged_publication_enabled() && second->publication_stage_parked()) {
                active->fired=true;throw std::runtime_error("after the first selected stage rebound");
            }
            return active->allocate ? active->allocate(type,tensor) : ggml_nbytes(tensor);
        };
    }
    ~recurrent_stage_bind_fault() {type->iface.get_alloc_size=allocate;active=nullptr;}
};

struct recurrent_bytes_writer : llama_io_write_i {
    std::vector<uint8_t> bytes;
    void write(const void * source,size_t size) override {
        const size_t first=bytes.size(); bytes.resize(first+size);
        if (size) std::memcpy(bytes.data()+first,source,size);
    }
    void write_tensor(ggml_tensor * tensor,size_t offset,size_t size) override {
        const size_t first=bytes.size(); bytes.resize(first+size);
        ggml_backend_tensor_get(tensor,bytes.data()+first,offset,size);
    }
    size_t n_bytes() override { return bytes.size(); }
};

struct recurrent_bytes_reader : llama_io_read_i {
    const std::vector<uint8_t> & bytes;
    size_t cursor=0;
    explicit recurrent_bytes_reader(const std::vector<uint8_t> & bytes) : bytes(bytes) {}
    void read(void * destination,size_t size) override {
        if (cursor>bytes.size() || size>bytes.size()-cursor) throw std::runtime_error("short recurrent state");
        if (size) std::memcpy(destination,bytes.data()+cursor,size);
        cursor+=size;
    }
    void read_tensor(ggml_tensor * tensor,size_t offset,size_t size) override {
        if (cursor>bytes.size() || size>bytes.size()-cursor) throw std::runtime_error("short recurrent tensor");
        ggml_backend_tensor_set(tensor,bytes.data()+cursor,offset,size); cursor+=size;
    }
    size_t n_bytes() override { return cursor; }
};

int main(int argc, char ** argv) {
    testing t;
    t.test("owned_restore_queue_drains_partial_submission_and_requires_explicit_recovery", [&](testing & t) {
        const bool gpu=argc>1 && !std::strcmp(argv[1],"--cuda");
        ggml_backend_load_all();
        auto * device=ggml_backend_dev_by_name(gpu ? "CUDA0" : "CPU");
        if (!t.assert_true(device != nullptr)) return;
        for (size_t failure : {1u,2u}) {
            ggml_backend_ptr producer(ggml_backend_dev_init(device,nullptr));
            auto * host=ggml_backend_dev_host_buffer_type(device);
            if (!host) host=ggml_backend_cpu_buffer_type();
            llama_recurrent_spill_layout layout;
            if (!t.assert_true(llama_recurrent_spill_layout_make(
                    {{GGML_TYPE_F32,GGML_TYPE_F32,32,64,true}},1,3,layout))) return;
            auto bank=llama_recurrent_spill_bank::create(layout,host,119,0);
            recurrent_copy_backend_probe probe(device);
            if (!t.assert_true(bank && bank->enable_publication(device,ggml_backend_dev_buffer_type(device),2))) return;
            probe.observe_restores();
            const size_t host_bytes=bank->host_bytes(),stage_bytes=bank->staged_device_bytes();
            ggml_context_ptr ctx(ggml_init({32768,nullptr,true}));
            auto * r=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,32);
            auto * s=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,64);
            auto * dr=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,32);
            auto * ds=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,64);
            ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),producer.get()));
            if (!t.assert_true(bool(buffer))) return;
            std::vector<float> expected_r(32,1.25f),expected_s(64,-2.5f),actual_r(32),actual_s(64);
            expected_r[0]=expected_s[0]=-0.0f;
            ggml_backend_tensor_set(r,expected_r.data(),0,ggml_nbytes(r));
            ggml_backend_tensor_set(s,expected_s.data(),0,ggml_nbytes(s));
            const auto capture=[&](uint64_t epoch) {
                return bank->begin_capture({119,0,epoch}) && bank->capture(producer.get(),0,false,2,r,0) &&
                    bank->capture(producer.get(),0,true,2,s,0) && bank->publish_capture() && bank->complete_capture();
            };
            if (!t.assert_true(capture(1))) return;
            const auto * original=bank->snapshot_data(0,false,2,1);
            if (!t.assert_true(original && bank->begin_restore({119,0,1},2))) return;
            probe.fail_upload=failure;
            const bool first=bank->restore(producer.get(),0,false,dr,0);
            const bool second=first && bank->restore(producer.get(),0,true,ds,0);
            t.assert_true(!second && !bank->publish_restore());
            t.assert_equal(failure,probe.uploads);
            bank->cancel();
            t.assert_true(!bank->begin_restore({119,0,1},2) && !bank->begin_capture({119,0,2}));
            t.assert_true(bank->snapshot_data(0,false,2,1)==original &&
                !std::memcmp(original,expected_r.data(),ggml_nbytes(r)));
            t.assert_equal(uint64_t(1),bank->snapshot_generation(0,true,2));
            bank->reset(true);
            probe.fail_upload=0;
            if (!t.assert_true(capture(2) && !bank->snapshot_data(0,false,2,1) &&
                    bank->begin_restore({119,0,2},2) && bank->restore(producer.get(),0,false,dr,0) &&
                    bank->restore(producer.get(),0,true,ds,0) && bank->publish_restore() && bank->complete_restore())) return;
            ggml_backend_tensor_get(dr,actual_r.data(),0,ggml_nbytes(dr));
            ggml_backend_tensor_get(ds,actual_s.data(),0,ggml_nbytes(ds));
            t.assert_true(!std::memcmp(expected_r.data(),actual_r.data(),ggml_nbytes(r)) &&
                !std::memcmp(expected_s.data(),actual_s.data(),ggml_nbytes(s)));
            t.assert_equal(failure+2,probe.uploads);
            t.assert_equal(host_bytes,bank->host_bytes());
            t.assert_equal(stage_bytes,bank->staged_device_bytes());
        }
    });
    t.test("selected_restore_can_start_before_target_scheduler_and_materialization_joins_it", [](testing & t) {
        const char * actions[]={"materialize","export","import","clear_data","clear_metadata","full_remove",
            "next_capture","repeat_completion","destroy","no_successor","capture_abort"};
        for (size_t action=0;action<sizeof(actions)/sizeof(actions[0]);++action) t.test(actions[action],[&](testing & t) {
        ggml_backend_ptr producer(ggml_backend_cpu_init());
        auto * device=ggml_backend_get_device(producer.get());
        std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_QWEN35,llama_model_default_params()));
        model->hparams.n_layer_all=1;
        model->hparams.ssm_d_conv=2; model->hparams.ssm_d_inner=8;
        model->hparams.ssm_d_state=8; model->hparams.ssm_n_group=1;
        std::unique_ptr<llama_memory_recurrent> memory;
        recurrent_copy_backend_probe probe(device);
        memory.reset(new llama_memory_recurrent(*model,GGML_TYPE_F32,GGML_TYPE_F32,false,1,1,3,{},true));
        ggml_context_ptr ctx(ggml_init({65536,nullptr,true}));
        auto * r=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,model->hparams.n_embd_r());
        auto * s=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,model->hparams.n_embd_s());
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),producer.get()));
        if (!t.assert_true(bool(buffer))) return;
        std::vector<float> expected_r(ggml_nelements(r),1.5f),expected_s(ggml_nelements(s),-3.5f);
        expected_r[0]=expected_s[0]=-0.0f;
        ggml_backend_tensor_set(r,expected_r.data(),0,ggml_nbytes(r));
        ggml_backend_tensor_set(s,expected_s.data(),0,ggml_nbytes(s));
        auto * bank=memory->spill_bank();
        auto * graph=ggml_new_graph_custom(ctx.get(),128,false);
        std::vector<llm_graph_recurrent_snapshot> outputs;
        for (bool value : {false,true}) {
            std::vector<ggml_tensor *> copies;
            for (uint32_t slot=1;slot<=3;++slot) {
                auto * input=value ? s : r;
                copies.push_back(ggml_cpy(ctx.get(),input,bank->staging_tensor(0,value,slot)));
                outputs.push_back({input,0,slot,value,0});
            }
            if (!t.assert_true(bank->publish_staged(ctx.get(),graph,0,value,copies))) return;
        }
        auto * backend=producer.get();
        ggml_backend_sched_ptr sched(ggml_backend_sched_new(&backend,nullptr,1,128,false,true));
        if (!t.assert_true(memory->begin_spill_capture(4) && ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS &&
                memory->capture_spill(sched.get(),outputs) && memory->complete_spill())) return;
        memory->used=1; memory->cells[0].tail=0; memory->cells[0].pos=7; memory->cells[0].seq_id.insert(0);
        std::vector<float> incoming_r(expected_r.size(),11.0f),incoming_s(expected_s.size(),-17.0f);
        ggml_backend_tensor_set(memory->r_l[0],incoming_r.data(),0,ggml_nbytes(r));
        ggml_backend_tensor_set(memory->s_l[0],incoming_s.data(),0,ggml_nbytes(s));
        recurrent_bytes_writer incoming;
        memory->state_write(incoming,0,0);
        if (!t.assert_true(memory->seq_rm(0,6,-1))) return;
        t.assert_equal(uint32_t(2),memory->rs_idx[0]);
        recurrent_bytes_writer selected;
        memory->state_write(selected,0,0);
        probe.defer_restores();
        if (action==9) {
            recurrent_bytes_writer saved;
            memory->state_write(saved,0,0);
            t.assert_true(saved.bytes==selected.bytes && probe.uploads==0 && memory->rs_idx[0]==2);
            return;
        }
        if (!t.assert_true("restore submission does not need the next target scheduler",memory->restore_spill_async(nullptr))) return;
        t.assert_equal(size_t(2),probe.pending_uploads.size());
        t.assert_equal(uint32_t(0),memory->rs_idx[0]);
        t.assert_true(!memory->seq_rm(0,5,-1));
        if (action==8) {
            probe.watch_teardown(); memory.reset();
            t.assert_true(probe.pending_uploads.empty() && !probe.undrained_on_free && probe.backend_frees==1);
            return;
        }
        bool retained=true;
        llama_pos position=5;
        uint64_t generation=1;
        if (action==0) t.assert_true("state materialization joins a pending restore even when rs_idx is zero",memory->materialize_spill_for_device_state());
        if (action==1) {
            recurrent_bytes_writer saved;
            memory->state_write(saved,0,0);
            t.assert_true(saved.bytes==selected.bytes);
        }
        if (action==2) {
            recurrent_bytes_reader saved(incoming.bytes);
            memory->state_read(saved,0,0);
            expected_r=incoming_r; expected_s=incoming_s; retained=false;
            t.assert_equal(llama_pos(7),memory->seq_pos_max(0));
        }
        if (action==3 || action==4 || action==5) {
            if (action==5) t.assert_true(memory->seq_rm(0,0,-1));
            else memory->clear(action==3);
            if (action==3) {
                std::fill(expected_r.begin(),expected_r.end(),0.0f);
                std::fill(expected_s.begin(),expected_s.end(),0.0f);
            }
            retained=false;
            t.assert_equal(llama_pos(-1),memory->seq_pos_max(0));
            t.assert_true(!bank->snapshot_data(0,false,2,1));
        }
        if (action==6 || action==10) {
            if (!t.assert_true(memory->begin_spill_capture(4))) return;
            if (action==6) {
                memory->cells[0].pos=9;
                if (!t.assert_true(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS &&
                        memory->capture_spill(sched.get(),outputs) && memory->seq_rm(0,8,-1) &&
                        memory->restore_spill_async(nullptr) && memory->complete_restore_spill())) return;
                position=7; generation=2;
            } else memory->abort_spill_capture();
        }
        if (action==7) t.assert_true(memory->complete_restore_spill() && memory->complete_restore_spill());
        t.assert_true(probe.pending_uploads.empty());
        if (retained) {
            t.assert_true("a completed restore remains a single-use rollback until new target execution",!memory->seq_rm(0,position,-1));
            t.assert_equal(position,memory->seq_pos_max(0));
        }
        std::vector<float> actual_r(expected_r.size()),actual_s(expected_s.size());
        ggml_backend_tensor_get(memory->r_l[0],actual_r.data(),0,ggml_nbytes(r));
        ggml_backend_tensor_get(memory->s_l[0],actual_s.data(),0,ggml_nbytes(s));
        t.assert_true(!std::memcmp(expected_r.data(),actual_r.data(),ggml_nbytes(r)) &&
            !std::memcmp(expected_s.data(),actual_s.data(),ggml_nbytes(s)));
        t.assert_equal(generation,memory->spill_generation());
        });
    });
    t.test("staged_restore_uses_owned_copy_queue_not_the_next_compute_queue", [](testing & t) {
        ggml_backend_ptr producer(ggml_backend_cpu_init());
        auto * device=ggml_backend_get_device(producer.get());
        llama_recurrent_spill_layout layout;
        if (!t.assert_true(llama_recurrent_spill_layout_make(
                {{GGML_TYPE_F32,GGML_TYPE_F32,32,64,true}},1,3,layout))) return;
        auto bank=llama_recurrent_spill_bank::create(layout,ggml_backend_cpu_buffer_type(),117,0);
        recurrent_copy_backend_probe probe(device);
        if (!t.assert_true(bank && bank->enable_publication(device,ggml_backend_cpu_buffer_type(),2))) return;
        probe.observe_restores();
        ggml_context_ptr ctx(ggml_init({32768,nullptr,true}));
        auto * r=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,32);
        auto * s=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,64);
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),producer.get()));
        if (!t.assert_true(bool(buffer))) return;
        std::vector<float> original_r(32,1.25f),original_s(64,-2.5f),actual_r(32),actual_s(64);
        ggml_backend_tensor_set(r,original_r.data(),0,ggml_nbytes(r));
        ggml_backend_tensor_set(s,original_s.data(),0,ggml_nbytes(s));
        if (!t.assert_true(bank->begin_capture({117,0,1}) &&
                bank->capture(producer.get(),0,false,2,r,0) && bank->capture(producer.get(),0,true,2,s,0) &&
                bank->publish_capture() && bank->complete_capture() && bank->begin_restore({117,0,1},2))) return;
        if (!t.assert_true(bank->restore(producer.get(),0,false,r,0) && bank->restore(producer.get(),0,true,s,0) &&
                bank->publish_restore() && bank->complete_restore())) return;
        ggml_backend_tensor_get(r,actual_r.data(),0,ggml_nbytes(r));
        ggml_backend_tensor_get(s,actual_s.data(),0,ggml_nbytes(s));
        t.assert_true(original_r==actual_r && original_s==actual_s);
        t.assert_equal(uint64_t(1),bank->snapshot_generation(0,false,2));
        t.assert_equal(size_t(2),probe.uploads);
        const auto transfers=bank->transfer_stats();
        t.assert_equal(size_t(384),transfers.d2h_bytes);
        t.assert_equal(size_t(384),transfers.h2d_bytes);
        t.assert_equal(size_t(2),transfers.d2h_calls);
        t.assert_equal(size_t(2),transfers.h2d_calls);
    });
    t.test("grouped_publication_failure_drains_and_invalidates_the_whole_submitted_plane", [](testing & t) {
        for (size_t failure : {1u,2u}) {
            ggml_backend_ptr producer(ggml_backend_cpu_init());
            auto * device=ggml_backend_get_device(producer.get());
            llama_recurrent_spill_layout layout;
            if (!t.assert_true(llama_recurrent_spill_layout_make(
                    {{GGML_TYPE_F32,GGML_TYPE_F32,32,64,true}},1,3,layout))) return;
            auto bank=llama_recurrent_spill_bank::create(layout,ggml_backend_cpu_buffer_type(),109,0);
            recurrent_copy_backend_probe probe(device);
            if (!t.assert_true(bank && bank->enable_publication(device,ggml_backend_cpu_buffer_type(),2))) return;
            probe.defer_downloads(); probe.fail_download=failure;
            ggml_context_ptr ctx(ggml_init({65536,nullptr,true}));
            auto * r=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,32,1);
            auto * s=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,64,1);
            ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),producer.get()));
            if (!t.assert_true(bool(buffer))) return;
            auto * graph=ggml_new_graph_custom(ctx.get(),128,false);
            for (bool value : {false,true}) {
                std::vector<ggml_tensor *> copies;
                for (uint32_t snapshot=1;snapshot<=3;++snapshot)
                    copies.push_back(ggml_cpy(ctx.get(),value ? s : r,bank->staging_tensor(0,value,snapshot)));
                if (!t.assert_true(bank->publish_staged(ctx.get(),graph,0,value,copies))) return;
            }
            if (!t.assert_true(bank->begin_staged_capture({109,0,1},4))) return;
            t.assert_true(ggml_backend_graph_compute(producer.get(),graph)==GGML_STATUS_FAILED);
            t.assert_true(!bank->publish_capture());
            bank->cancel();
            t.assert_true(probe.pending.empty());
            for (uint32_t snapshot=1;snapshot<=3;++snapshot) {
                t.assert_true(!bank->snapshot_pristine(0,false,snapshot));
                t.assert_equal(failure==1,bank->snapshot_pristine(0,true,snapshot));
            }
        }
    });
    t.test("grouped_publication_short_batches_padding_and_invalid_groups_are_exact", [&](testing & t) {
        const bool gpu=argc > 1 && !std::strcmp(argv[1],"--cuda");
        ggml_backend_load_all();
        auto * device=ggml_backend_dev_by_name(gpu ? "CUDA0" : "CPU");
        if (!t.assert_true(device != nullptr)) return;
        ggml_backend_ptr producer(ggml_backend_dev_init(device,nullptr));
        auto * host=ggml_backend_dev_host_buffer_type(device);
        if (!host) host=ggml_backend_cpu_buffer_type();
        for (uint32_t width : {1u,2u,4u,8u}) for (uint32_t padding : {0u,1u}) {
            llama_recurrent_spill_layout layout;
            if (!t.assert_true(llama_recurrent_spill_layout_make(
                    {{GGML_TYPE_F32,GGML_TYPE_F32,32-padding,64-padding,true}},1,7,layout))) return;
            auto bank=llama_recurrent_spill_bank::create(layout,host,107,0);
            if (!t.assert_true(bank && bank->enable_publication(device,ggml_backend_dev_buffer_type(device),2))) return;
            ggml_context_ptr ctx(ggml_init({1024*1024,nullptr,true}));
            std::vector<ggml_tensor *> inputs(14);
            for (size_t i=0;i<inputs.size();++i) inputs[i]=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,(i%2 ? 64 : 32)-padding,1);
            ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),producer.get()));
            if (!t.assert_true(bool(buffer))) return;
            auto * graph=ggml_new_graph_custom(ctx.get(),256,false);
            struct retire_graphs {
                ggml_backend_t backend;
                void (*release)(ggml_backend_t);
                ~retire_graphs() {
                    if (release) { ggml_backend_synchronize(backend);release(backend); }
                }
            };
            auto * reg=ggml_backend_dev_backend_reg(device);
            const auto retire=reinterpret_cast<void (*)(ggml_backend_t)>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_cuda_graph_release_all"));
            const auto captured=reinterpret_cast<bool (*)(ggml_backend_t,const void *)>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_cuda_graph_is_captured"));
            if (gpu && !t.assert_true(retire && captured)) return;
            // Native captures cannot outlive their tensor metadata or backing buffers.
            retire_graphs retirement{producer.get(),retire};
            for (bool value : {false,true}) {
                const uint32_t count=value ? width-1 : 7;
                std::vector<ggml_tensor *> copies;
                for (uint32_t snapshot=count;snapshot>0;--snapshot) {
                    const size_t i=(snapshot-1)*2+size_t(value);
                    ggml_backend_tensor_memset(inputs[i],uint8_t(i+1),0,ggml_nbytes(inputs[i]));
                    copies.push_back(ggml_cpy(ctx.get(),inputs[i],bank->staging_tensor(0,value,snapshot)));
                }
                if (!copies.empty()) {
                    const auto before=ggml_graph_n_nodes(graph);
                    auto invalid=copies; invalid.push_back(copies.front());
                    t.assert_true(!bank->publish_staged(ctx.get(),graph,0,value,invalid));
                    t.assert_equal(before,ggml_graph_n_nodes(graph));
                }
                if (!t.assert_true(bank->publish_staged(ctx.get(),graph,0,value,copies))) return;
            }
            if (!t.assert_true(bank->begin_staged_capture({107,0,1},width) &&
                    ggml_backend_graph_compute(producer.get(),graph)==GGML_STATUS_SUCCESS &&
                    bank->publish_capture() && bank->complete_capture())) return;
            for (bool value : {false,true}) for (uint32_t snapshot=1;snapshot<=7;++snapshot) {
                const bool touched=!value || snapshot < width;
                t.assert_equal(touched ? uint64_t(1) : uint64_t(0),bank->snapshot_generation(0,value,snapshot));
                if (!touched) { t.assert_true(bank->snapshot_pristine(0,value,snapshot)); continue; }
                const auto i=(snapshot-1)*2+size_t(value);
                const auto * bytes=static_cast<const uint8_t *>(bank->snapshot_data(0,value,snapshot,1));
                t.assert_true(bytes && std::all_of(bytes,bytes+ggml_nbytes(inputs[i]),[&](uint8_t b) { return b==i+1; }));
            }
            for (uint64_t epoch : {2u,3u}) {
                if (!t.assert_true(bank->begin_staged_capture({107,0,epoch},width) &&
                        ggml_backend_graph_compute(producer.get(),graph)==GGML_STATUS_SUCCESS &&
                        bank->publish_capture() && bank->complete_capture())) return;
                for (bool value : {false,true}) for (uint32_t snapshot=1;snapshot<=7;++snapshot) {
                    const bool touched=!value || snapshot<width;
                    t.assert_equal(touched ? epoch : uint64_t(0),bank->snapshot_generation(0,value,snapshot));
                    if (!touched) { t.assert_true(bank->snapshot_pristine(0,value,snapshot));continue; }
                    const auto i=(snapshot-1)*2+size_t(value);
                    const auto * bytes=static_cast<const uint8_t *>(bank->snapshot_data(0,value,snapshot,epoch));
                    t.assert_true(bytes && std::all_of(bytes,bytes+ggml_nbytes(inputs[i]),[&](uint8_t byte) {return byte==i+1;}));
                }
            }
            if (gpu) {
                bool replayed=false;
                for (int node=0;node<ggml_graph_n_nodes(graph);++node)
                    replayed|=captured(producer.get(),ggml_graph_node(graph,node));
                t.assert_true("the live graph reached native CUDA capture/replay",replayed);
            }
        }
    });
    t.test("grouped_publication_has_one_boundary_per_produced_plane", [&](testing & t) {
        const bool gpu=argc > 1 && !std::strcmp(argv[1],"--cuda");
        ggml_backend_load_all();
        auto * device=ggml_backend_dev_by_name(gpu ? "CUDA0" : "CPU");
        if (!t.assert_true(device != nullptr)) return;
        ggml_backend_ptr producer(ggml_backend_dev_init(device,nullptr));
        auto * host=ggml_backend_dev_host_buffer_type(device);
        if (!host) host=ggml_backend_cpu_buffer_type();
        llama_recurrent_spill_layout layout;
        if (!t.assert_true(llama_recurrent_spill_layout_make(
                std::vector<llama_recurrent_spill_layer>(3,{GGML_TYPE_F32,GGML_TYPE_F32,32,64,true}),1,7,layout))) return;
        auto bank=llama_recurrent_spill_bank::create(layout,host,105,0);
        if (!t.assert_true(bank && bank->enable_publication(device,ggml_backend_dev_buffer_type(device),2))) return;
        ggml_context_ptr ctx(ggml_init({1024*1024,nullptr,true}));
        std::vector<ggml_tensor *> input(42);
        for (size_t i=0;i<input.size();++i) input[i]=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,i%2 ? 64 : 32,1);
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),producer.get()));
        if (!t.assert_true(bool(buffer))) return;
        auto * graph=ggml_new_graph_custom(ctx.get(),512,false);
        for (size_t layer=0;layer<3;++layer) for (bool value : {false,true}) {
            std::vector<ggml_tensor *> copies;
            for (uint32_t snapshot=7;snapshot>0;--snapshot) {
                const auto i=(layer*7+snapshot-1)*2+size_t(value);
                ggml_backend_tensor_memset(input[i],uint8_t(i+1),0,ggml_nbytes(input[i]));
                copies.push_back(ggml_cpy(ctx.get(),input[i],bank->staging_tensor(layer,value,snapshot)));
            }
            if (!t.assert_true(bank->publish_staged(ctx.get(),graph,layer,value,copies))) return;
        }
        size_t managed=0;
        for (int i=0;i<ggml_graph_n_nodes(graph);++i) {
            auto * node=ggml_graph_node(graph,i);
            ggml_backend_buffer_t owner=nullptr;
            if (!t.assert_true(ggml_backend_execution_owner(node,owner))) return;
            if (owner && node->op==GGML_OP_CPY) ++managed;
        }
        t.assert_equal(size_t(6),managed);
        if (!t.assert_true(bank->begin_staged_capture({105,0,1},8) &&
                ggml_backend_graph_compute(producer.get(),graph)==GGML_STATUS_SUCCESS &&
                bank->publish_capture() && bank->complete_capture())) return;
        for (size_t layer=0;layer<3;++layer) for (bool value : {false,true}) for (uint32_t snapshot=1;snapshot<=7;++snapshot) {
            const auto i=(layer*7+snapshot-1)*2+size_t(value);
            const auto * bytes=static_cast<const uint8_t *>(bank->snapshot_data(layer,value,snapshot,1));
            t.assert_true(bytes && std::all_of(bytes,bytes+ggml_nbytes(input[i]),[&](uint8_t b) { return b==i+1; }));
        }
    });
    t.test("contiguous_snapshot_planes_batch_without_copying_padding_or_untouched_rows", [&](testing & t) {
        const bool gpu=argc > 1 && !std::strcmp(argv[1],"--cuda");
        ggml_backend_load_all();
        auto * device=ggml_backend_dev_by_name(gpu ? "CUDA0" : "CPU");
        if (!t.assert_true(device != nullptr)) return;
        for (uint32_t r_width : {31u,32u}) for (uint32_t s_width : {63u,64u}) {
            ggml_backend_ptr producer(ggml_backend_dev_init(device,nullptr));
            auto * host_type=ggml_backend_dev_host_buffer_type(device);
            if (!host_type) host_type=ggml_backend_cpu_buffer_type();
            llama_recurrent_spill_layout layout;
            if (!t.assert_true(llama_recurrent_spill_layout_make(
                    std::vector<llama_recurrent_spill_layer>(3,{GGML_TYPE_F32,GGML_TYPE_F32,r_width,s_width,true}),1,7,layout))) return;
            auto bank=llama_recurrent_spill_bank::create(layout,host_type,97,0);
            recurrent_copy_backend_probe probe(device);
            if (!t.assert_true(bank && bank->enable_publication(device,ggml_backend_dev_buffer_type(device),2))) return;
            if (gpu) probe.observe_downloads(); else probe.defer_downloads();
            const auto original_stage=bank->staged_device_bytes(),original_host=bank->host_bytes();
            ggml_context_ptr ctx(ggml_init({1024*1024,nullptr,true}));
            std::vector<ggml_tensor *> input(42);
            for (size_t layer=0;layer<3;++layer) for (uint32_t snapshot=1;snapshot<=7;++snapshot)
                for (bool value : {false,true}) input[(layer*7+snapshot-1)*2+size_t(value)]=
                    ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,value ? s_width : r_width);
            ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),producer.get()));
            if (!t.assert_true(bool(buffer))) return;
            std::vector<uint64_t> generations(42);
            std::vector<std::vector<float>> expected(42);
            for (uint32_t width : {1u,2u,4u,8u}) {
                auto * graph=ggml_new_graph_custom(ctx.get(),256,false);
                for (size_t layer=0;layer<3;++layer) for (bool value : {false,true}) {
                    const uint32_t snapshots=value ? width-1 : 7;
                    for (uint32_t j=0;j<snapshots;++j) {
                        const uint32_t snapshot=snapshots-j;
                        const size_t index=(layer*7+snapshot-1)*2+size_t(value);
                        expected[index].assign(value ? s_width : r_width,float(1000*width+100*layer+10*snapshot+value));
                        generations[index]=width;
                        ggml_backend_tensor_set(input[index],expected[index].data(),0,ggml_nbytes(input[index]));
                        auto * staged=ggml_cpy(ctx.get(),input[index],bank->staging_tensor(layer,value,snapshot));
                        ggml_build_forward_expand(graph,ggml_cpy(ctx.get(),staged,bank->publication_tensor(layer,value,snapshot)));
                    }
                }
                const auto before=probe.downloads;
                if (!t.assert_true(bank->begin_staged_capture({97,0,width},width) &&
                        ggml_backend_graph_compute(producer.get(),graph) == GGML_STATUS_SUCCESS &&
                        bank->publish_capture() && bank->complete_capture())) return;
                const size_t r_calls=r_width == 32 ? 1 : 7;
                const size_t s_calls=width == 1 ? 0 : s_width == 64 ? 1 : width-1;
                t.assert_equal(3*(r_calls+s_calls),probe.downloads-before);
                t.assert_equal(original_stage,bank->staged_device_bytes());
                t.assert_equal(original_host,bank->host_bytes());
                for (size_t layer=0;layer<3;++layer) for (uint32_t snapshot=1;snapshot<=7;++snapshot)
                    for (bool value : {false,true}) {
                        const size_t index=(layer*7+snapshot-1)*2+size_t(value);
                        t.assert_equal(generations[index],bank->snapshot_generation(layer,value,snapshot));
                        const auto * bytes=bank->snapshot_data(layer,value,snapshot,generations[index]);
                        if (generations[index]) t.assert_true(bytes && !std::memcmp(bytes,expected[index].data(),ggml_nbytes(input[index])));
                        else t.assert_true(!bytes && bank->snapshot_pristine(layer,value,snapshot));
                    }
            }
        }
    });
    t.test("packed_transfer_failure_invalidates_every_touched_snapshot_and_retries", [](testing & t) {
        for (size_t failure : {1u,2u}) {
            ggml_backend_ptr producer(ggml_backend_cpu_init());
            auto * device=ggml_backend_get_device(producer.get());
            llama_recurrent_spill_layout layout;
            if (!t.assert_true(llama_recurrent_spill_layout_make(
                    {{GGML_TYPE_F32,GGML_TYPE_F32,32,64,true}},1,3,layout))) return;
            auto bank=llama_recurrent_spill_bank::create(layout,ggml_backend_cpu_buffer_type(),99,0);
            recurrent_copy_backend_probe probe(device);
            if (!t.assert_true(bank && bank->enable_publication(device,ggml_backend_cpu_buffer_type(),2))) return;
            probe.defer_downloads();
            ggml_context_ptr ctx(ggml_init({65536,nullptr,true}));
            auto * r=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,32);
            auto * s=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,64);
            ggml_backend_buffer_ptr source(ggml_backend_alloc_ctx_tensors(ctx.get(),producer.get()));
            if (!t.assert_true(bool(source))) return;
            ggml_backend_tensor_memset(r,0x11,0,ggml_nbytes(r));
            ggml_backend_tensor_memset(s,0x22,0,ggml_nbytes(s));
            auto * graph=ggml_new_graph_custom(ctx.get(),64,false);
            for (bool value : {false,true}) for (uint32_t snapshot=1;snapshot<=3;++snapshot) {
                auto * copy=ggml_cpy(ctx.get(),value ? s : r,bank->staging_tensor(0,value,snapshot));
                ggml_build_forward_expand(graph,ggml_cpy(ctx.get(),copy,bank->publication_tensor(0,value,snapshot)));
            }
            probe.fail_download=failure;
            if (!t.assert_true(bank->begin_staged_capture({99,0,1},4))) return;
            t.assert_true(ggml_backend_graph_compute(producer.get(),graph) == GGML_STATUS_FAILED);
            t.assert_true(!bank->publish_capture());
            bank->cancel();
            t.assert_true(probe.pending.empty());
            for (uint32_t snapshot=1;snapshot<=3;++snapshot) {
                t.assert_true(!bank->snapshot_data(0,false,snapshot,1) && !bank->snapshot_pristine(0,false,snapshot));
                t.assert_equal(failure == 1,bank->snapshot_pristine(0,true,snapshot));
            }
            probe.fail_download=0;
            t.assert_true(bank->begin_staged_capture({99,0,2},4) &&
                ggml_backend_graph_compute(producer.get(),graph) == GGML_STATUS_SUCCESS &&
                bank->publish_capture() && bank->complete_capture());
            t.assert_true(bank->snapshot_data(0,true,3,2) != nullptr);
        }
    });
    t.test("publication_resources_are_prepared_transactionally_before_admission", [](testing & t) {
        ggml_backend_ptr producer(ggml_backend_cpu_init());
        auto * device=ggml_backend_get_device(producer.get());
        for (bool fail : {false,true}) {
            llama_recurrent_spill_layout layout;
            if (!t.assert_true(llama_recurrent_spill_layout_make(
                    {{GGML_TYPE_F32,GGML_TYPE_F32,4,8,true}},1,7,layout))) return;
            auto bank=llama_recurrent_spill_bank::create(layout,ggml_backend_cpu_buffer_type(),89,0);
            recurrent_copy_backend_probe probe(device);
            probe.fail_prepare=fail;
            const bool enabled=bank->enable_publication(device,ggml_backend_cpu_buffer_type(),2);
            t.assert_equal(!fail,enabled);
            if (fail && enabled) return;
            if (fail) {
                // Failed admission frees its candidate backend; the bank must permit a clean retry.
                probe.copy=nullptr;
                t.assert_true(!bank->staged_publication_enabled());
                t.assert_true(bank->enable_publication(device,ggml_backend_cpu_buffer_type(),2));
            }
            t.assert_true(probe.preparations > 0);
            t.assert_true(bank->begin_staged_capture({89,0,1},8));
            bank->cancel();
        }
    });
    t.test("streamed_rollback_plan_follows_verification_not_auxiliary_kv", [](testing & t) {
        for (uint32_t depth=0;depth<=7;++depth) {
            llama_recurrent_rollback_plan plan;
            if (!t.assert_true(llama_recurrent_rollback_plan_make(depth,false,256,plan))) return;
            t.assert_equal(depth,plan.depth);
            t.assert_equal(depth+1,plan.verification_rows);
            t.assert_equal(std::max(4u,depth+1),plan.workspace_rows);
            t.assert_equal(false,plan.host_spill);
            t.assert_equal(0u,plan.stage_slots);
            llama_recurrent_rollback_plan ram;
            if (!t.assert_true(llama_recurrent_rollback_plan_make(depth,false,256,ram,true))) return;
            t.assert_equal(plan.depth,ram.depth);
            t.assert_equal(plan.verification_rows,ram.verification_rows);
            t.assert_equal(plan.workspace_rows,ram.workspace_rows);
            t.assert_equal(depth!=0,ram.host_spill);
            t.assert_equal(depth ? 2u : 0u,ram.stage_slots);
            for (bool mtp : {false,true}) {
                auto saved=plan;
                const bool valid=llama_recurrent_rollback_plan_make(depth,mtp,depth+1,plan);
                t.assert_equal(!mtp || depth <= 3,valid);
                if (!valid) {
                    t.assert_equal(saved.depth,plan.depth);
                    t.assert_equal(saved.verification_rows,plan.verification_rows);
                }
            }
        }
        llama_recurrent_rollback_plan sentinel;
        sentinel.depth=123; sentinel.verification_rows=456; sentinel.workspace_rows=789;
        for (auto bad : {std::pair{8u,256u},std::pair{UINT32_MAX,256u},std::pair{7u,7u},std::pair{0u,0u}}) {
            t.assert_true(!llama_recurrent_rollback_plan_make(bad.first,false,bad.second,sentinel));
            t.assert_equal(123u,sentinel.depth);
            t.assert_equal(456u,sentinel.verification_rows);
            t.assert_equal(789u,sentinel.workspace_rows);
        }
    });
    t.test("selected_resident_rollback_accounts_for_history_and_restores_each_prefix", [](testing & t) {
        std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_QWEN35,llama_model_default_params()));
        model->hparams.n_layer_all=1;model->hparams.ssm_d_conv=2;model->hparams.ssm_d_inner=8;
        model->hparams.ssm_d_state=8;model->hparams.ssm_n_group=1;
        for (auto selected : std::vector<std::vector<llama_seq_id>>{{2},{3,1},{3,1,0},{3,1,0,2}})
        for (uint32_t depth=1;depth<=7;++depth) for (uint32_t rollback=0;rollback<=depth;++rollback) {
            const auto count=uint32_t(selected.size());
            llama_memory_recurrent memory(*model,GGML_TYPE_F32,GGML_TYPE_F32,false,count,4,depth,nullptr,false,true,true);
            if (!t.assert_true("native rollback admits a compact selected bank",memory.configure_selected_sequences(selected))) return;
            t.assert_true(!memory.spill_enabled());
            t.assert_equal(int64_t(count*(depth+1)),memory.s_l[0]->ne[1]);
            ggml_backend_buffer_type_t type=nullptr;size_t bytes=0;
            if (!t.assert_true(memory.current_storage_requirements(type,bytes))) return;
            llama_memory_recurrent native(*model,GGML_TYPE_F32,GGML_TYPE_F32,false,count,count,depth,nullptr);
            ggml_backend_buffer_type_t native_type=nullptr;size_t native_bytes=0;
            if (!t.assert_true(native.current_storage_requirements(native_type,native_bytes))) return;
            t.assert_equal(native_bytes,bytes);
            std::unique_ptr<ggml_backend_memory_arena,decltype(&ggml_backend_memory_arena_free)> arena(
                ggml_backend_memory_arena_new(type,bytes),ggml_backend_memory_arena_free);
            const auto alignment=ggml_backend_buft_get_alignment(type);
            if (!t.assert_true(arena && ggml_backend_memory_arena_begin(arena.get(),0) &&
                    ggml_backend_memory_arena_reserve_at(arena.get(),91,0,bytes,alignment,GGML_BACKEND_MEMORY_REGION_PERSISTENT,nullptr) &&
                    ggml_backend_memory_arena_commit(arena.get()))) return;
            std::unique_ptr<ggml_backend_memory_lease,decltype(&ggml_backend_memory_lease_free)> lease(
                ggml_backend_memory_arena_acquire(arena.get(),91),ggml_backend_memory_lease_free);
            if (!t.assert_true(lease && memory.bind_current_storage(lease.get()))) return;
            llama_batch_allocr allocator(1);auto batch=allocator.ubatch_reserve(depth+1,count);
            std::fill(batch.data->seq_idx.begin(),batch.data->seq_idx.end(),-1);
            for (uint32_t seq=0;seq<count;++seq) {
                batch.seq_id_unq[seq]=selected[seq];batch.seq_idx[selected[seq]]=int32_t(seq);
            }
            for (uint32_t seq=0;seq<count;++seq) for (uint32_t i=0;i<=depth;++i) {
                const auto row=seq*(depth+1)+i;
                batch.n_seq_id[row]=1;batch.seq_id[row]=&selected[seq];batch.pos[row]=llama_pos(10+i);
            }
            if (!t.assert_true(memory.prepare({batch}) && memory.find_slot(batch))) return;
            for (auto * tensor : {memory.r_l[0],memory.s_l[0]}) {
                std::vector<float> values(ggml_nelements(tensor));
                const auto width=size_t(tensor->ne[0]);
                for (uint32_t slot=0;slot<=depth;++slot) for (uint32_t cell=0;cell<count;++cell) {
                    const auto sequence=*memory.cells[cell].seq_id.begin();
                    std::fill_n(values.data()+(slot*count+cell)*width,width,float(slot*10+sequence+1));
                }
                ggml_backend_tensor_set(tensor,values.data(),0,ggml_nbytes(tensor));
            }
            for (uint32_t seq=0;seq<count;++seq) {
                const auto sequence=selected[seq];
                const auto rejected=(rollback+seq)%(depth+1);
                if (rejected && !t.assert_true(memory.seq_rm(sequence,11+depth-rejected,-1))) return;
                recurrent_bytes_writer saved;memory.state_write(saved,sequence,0);
                llama_memory_recurrent restored(*model,GGML_TYPE_F32,GGML_TYPE_F32,false,1,1,0,nullptr);
                recurrent_bytes_reader reader(saved.bytes);restored.state_read(reader,0,0);
                t.assert_equal(saved.bytes.size(),reader.cursor);
                for (auto * tensor : {restored.r_l[0],restored.s_l[0]}) {
                    std::vector<float> values(ggml_nelements(tensor));
                    ggml_backend_tensor_get(tensor,values.data(),0,ggml_nbytes(tensor));
                    t.assert_true(std::all_of(values.begin(),values.end(),[&](float value) {return value==float(rejected*10+sequence+1);}));
                }
                t.assert_equal(llama_pos(10+depth-rejected),restored.seq_pos_max(0));
            }
        }
    });
    t.test("selected_current_bank_accepts_logical_id_without_allocating_unchosen_rows", [](testing & t) {
        std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_QWEN35,llama_model_default_params()));
        model->hparams.n_layer_all=1;model->hparams.ssm_d_conv=2;model->hparams.ssm_d_inner=8;
        model->hparams.ssm_d_state=8;model->hparams.ssm_n_group=1;
        llama_memory_recurrent memory(*model,GGML_TYPE_F32,GGML_TYPE_F32,false,1,2,0,nullptr,false,true);
        if (!t.assert_true(memory.configure_selected_sequences({1}))) return;
        ggml_backend_buffer_type_t type=nullptr;size_t bytes=0;
        if (!t.assert_true(memory.current_storage_requirements(type,bytes))) return;
        t.assert_equal(int64_t(1),memory.r_l[0]->ne[1]);
        t.assert_equal(int64_t(1),memory.s_l[0]->ne[1]);
        std::unique_ptr<ggml_backend_memory_arena,decltype(&ggml_backend_memory_arena_free)> arena(
            ggml_backend_memory_arena_new(type,bytes),ggml_backend_memory_arena_free);
        const auto alignment=ggml_backend_buft_get_alignment(type);
        if (!t.assert_true(arena && ggml_backend_memory_arena_begin(arena.get(),0) &&
                ggml_backend_memory_arena_reserve_at(arena.get(),91,0,bytes,alignment,GGML_BACKEND_MEMORY_REGION_PERSISTENT,nullptr) &&
                ggml_backend_memory_arena_commit(arena.get()))) return;
        std::unique_ptr<ggml_backend_memory_lease,decltype(&ggml_backend_memory_lease_free)> lease(
            ggml_backend_memory_arena_acquire(arena.get(),91),ggml_backend_memory_lease_free);
        if (!t.assert_true(lease && memory.bind_current_storage(lease.get()))) return;
        llama_batch_allocr allocator(1);
        auto batch=allocator.ubatch_reserve(2,1);
        llama_seq_id id=1;
        batch.n_seq_id[0]=batch.n_seq_id[1]=1;
        batch.seq_id[0]=batch.seq_id[1]=&id;
        batch.pos[0]=0;batch.pos[1]=1;
        if (!t.assert_true("selected logical sequence uses the sole physical current row",memory.prepare({batch}) && memory.find_slot(batch))) return;
        t.assert_equal(uint32_t(1),memory.used);
        t.assert_equal(llama_pos(-1),memory.seq_pos_max(0));
        t.assert_equal(llama_pos(1),memory.seq_pos_max(1));
        t.assert_true(memory.cells[0].has_seq_id(1));
        std::vector<float> values(ggml_nelements(memory.s_l[0]),-0.125f);values[0]=-0.0f;
        ggml_backend_tensor_set(memory.s_l[0],values.data(),0,ggml_nbytes(memory.s_l[0]));
        recurrent_bytes_writer saved;memory.state_write(saved,1,0);
        if (!t.assert_true(!saved.bytes.empty())) return;
        id=0;batch.pos[0]=2;batch.pos[1]=3;
        t.assert_true(!memory.prepare({batch}) && !memory.find_slot(batch));
        t.assert_equal(llama_pos(1),memory.seq_pos_max(1));
        t.assert_true(memory.seq_rm(1,0,-1));
        t.assert_equal(llama_pos(-1),memory.seq_pos_max(1));
        recurrent_bytes_reader restore(saved.bytes);memory.state_read(restore,1,0);
        t.assert_equal(saved.bytes.size(),restore.cursor);
        recurrent_bytes_writer again;memory.state_write(again,1,0);
        t.assert_true(saved.bytes==again.bytes);
        t.assert_true(memory.cells[0].has_seq_id(1));
        t.assert_true(!memory.configure_selected_sequences({0}));
        memory.clear(false);
        t.assert_equal(llama_pos(-1),memory.seq_pos_max(1));
        id=1;batch.pos[0]=0;batch.pos[1]=1;
        t.assert_true(memory.prepare({batch}) && memory.find_slot(batch));
    });
    t.test("compact_selected_current_rows_restore_native_sequence_images_independently", [](testing & t) {
        std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_QWEN35,llama_model_default_params()));
        model->hparams.n_layer_all=1;model->hparams.ssm_d_conv=2;model->hparams.ssm_d_inner=8;
        model->hparams.ssm_d_state=8;model->hparams.ssm_n_group=1;
        std::vector<std::vector<uint8_t>> references;
        for (uint32_t slot=0;slot<4;++slot) {
            llama_memory_recurrent native(*model,GGML_TYPE_F32,GGML_TYPE_F32,false,1,1,0,nullptr);
            llama_batch_allocr allocator(1);auto batch=allocator.ubatch_reserve(2,1);
            llama_seq_id sequence=0;
            batch.n_seq_id[0]=batch.n_seq_id[1]=1;batch.seq_id[0]=batch.seq_id[1]=&sequence;
            batch.pos[0]=llama_pos(10+slot*5);batch.pos[1]=batch.pos[0]+1;
            if (!t.assert_true(native.find_slot(batch))) return;
            for (auto * tensor : {native.r_l[0],native.s_l[0]}) {
                std::vector<float> values(ggml_nelements(tensor),float(slot+1)*-0.125f);values[0]=-0.0f;
                ggml_backend_tensor_set(tensor,values.data(),0,ggml_nbytes(tensor));
            }
            recurrent_bytes_writer saved;native.state_write(saved,0,0);references.push_back(std::move(saved.bytes));
        }
        for (const auto & ids : std::vector<std::vector<llama_seq_id>>{{0},{1},{3},{0,3},{3,1},{2,0}}) {
            llama_memory_recurrent memory(*model,GGML_TYPE_F32,GGML_TYPE_F32,false,uint32_t(ids.size()),4,0,nullptr,false,true);
            if (!t.assert_true(memory.configure_selected_sequences(ids))) return;
            t.assert_true(!memory.configure_selected_sequences({-1}));
            t.assert_true(!memory.configure_selected_sequences({4}));
            t.assert_true(!memory.configure_selected_sequences({1,1}));
            ggml_backend_buffer_type_t type=nullptr;size_t bytes=0;
            if (!t.assert_true(memory.current_storage_requirements(type,bytes))) return;
            const size_t alignment=ggml_backend_buft_get_alignment(type);
            std::unique_ptr<ggml_backend_memory_arena,decltype(&ggml_backend_memory_arena_free)> arena(
                ggml_backend_memory_arena_new(type,bytes),ggml_backend_memory_arena_free);
            if (!t.assert_true(arena && ggml_backend_memory_arena_begin(arena.get(),0) &&
                    ggml_backend_memory_arena_reserve_at(arena.get(),92,0,bytes,alignment,GGML_BACKEND_MEMORY_REGION_PERSISTENT,nullptr) &&
                    ggml_backend_memory_arena_commit(arena.get()))) return;
            std::unique_ptr<ggml_backend_memory_lease,decltype(&ggml_backend_memory_lease_free)> lease(
                ggml_backend_memory_arena_acquire(arena.get(),92),ggml_backend_memory_lease_free);
            if (!t.assert_true(lease && memory.bind_current_storage(lease.get()))) return;
            auto load_order=ids;std::sort(load_order.begin(),load_order.end());
            for (auto id : load_order) {
                recurrent_bytes_reader source(references[id]);memory.state_read(source,id,0);
                t.assert_equal(references[id].size(),source.cursor);
            }
            const auto exact=[&] {
                for (auto id : ids) {
                    recurrent_bytes_writer saved;memory.state_write(saved,id,0);
                    if (!t.assert_true("selected source image matches its native single-sequence control",saved.bytes==references[id])) return false;
                    t.assert_equal(llama_pos(11+id*5),memory.seq_pos_max(id));
                }
                return true;
            };
            if (!exact()) return;
            for (llama_seq_id id=0;id<4;++id)
                if (std::find(ids.begin(),ids.end(),id)==ids.end()) t.assert_equal(llama_pos(-1),memory.seq_pos_max(id));
            llama_batch_allocr allocator(1);auto batch=allocator.ubatch_reserve(2,uint32_t(ids.size()));
            auto logical_ids=ids;
            std::fill(batch.data->seq_idx.begin(),batch.data->seq_idx.end(),-1);
            for (size_t seq=0;seq<ids.size();++seq) {
                batch.seq_id_unq[seq]=ids[seq];batch.seq_idx[ids[seq]]=int32_t(seq);
            }
            for (size_t seq=0;seq<ids.size();++seq) for (size_t row=0;row<2;++row) {
                const size_t token=seq*2+row;
                batch.n_seq_id[token]=1;batch.seq_id[token]=&logical_ids[seq];
                batch.pos[token]=llama_pos(12+ids[seq]*5+llama_pos(row));
            }
            if (!t.assert_true(memory.prepare({batch}) && memory.prepare({batch}) && exact())) return;
            if (!t.assert_true(memory.park_current_storage())) return;
            ggml_backend_buffer_clear(ggml_backend_memory_arena_parent(arena.get()),0xa5);
            if (!t.assert_true(memory.restore_current_storage(lease.get()) && exact())) return;
            if (!t.assert_true(memory.find_slot(batch))) return;
            t.assert_equal(uint32_t(ids.size()),memory.n);
            for (auto id : ids) t.assert_equal(llama_pos(13+id*5),memory.seq_pos_max(id));
        }
    });
    t.test("deferred_current_geometry_rejects_invalid_rows_before_allocation", [](testing & t) {
        std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_QWEN35,llama_model_default_params()));
        model->hparams.n_layer_all=1;model->hparams.ssm_d_conv=2;model->hparams.ssm_d_inner=8;
        model->hparams.ssm_d_state=8;model->hparams.ssm_n_group=1;
        for (const auto & shape : std::vector<std::array<uint32_t,3>>{{0,1,0},{1,0,0},{3,1,0},{2,2,UINT32_MAX},{1,LLAMA_MAX_SEQ+1,0}}) {
            bool rejected=false;
            try {
                llama_memory_recurrent memory(*model,GGML_TYPE_F32,GGML_TYPE_F32,false,
                    shape[0],shape[1],shape[2],nullptr,false,true);
            } catch (const std::runtime_error &) { rejected=true; }
            t.assert_true("invalid selected-state geometry fails before tensor allocation",rejected);
        }
    });
    t.test("deferred_current_state_rebind_preserves_raw_rows_and_logical_cells", [](testing & t) {
        std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_QWEN35,llama_model_default_params()));
        model->hparams.n_layer_all=3;model->hparams.ssm_d_conv=2;model->hparams.ssm_d_inner=8;
        model->hparams.ssm_d_state=8;model->hparams.ssm_n_group=1;
        llama_memory_recurrent memory(*model,GGML_TYPE_F32,GGML_TYPE_F32,false,2,2,0,
            [](int32_t layer) { return layer!=1; },false,true);
        auto * type=ggml_backend_cpu_buffer_type();
        ggml_backend_buffer_type_t measured=nullptr;size_t bytes=0;
        if (!t.assert_true(memory.current_storage_pending() && memory.current_storage_requirements(measured,bytes))) return;
        t.assert_true(measured==type && bytes>0 && !memory.r_l[1] && !memory.s_l[1]);
        t.assert_true(!memory.r_l[0]->data && !memory.s_l[2]->data && !memory.prepare({}));
        recurrent_bytes_writer unavailable;
        bool rejected=false;
        try { memory.state_write(unavailable,0,0); } catch (const std::runtime_error &) { rejected=true; }
        t.assert_true("unbound current storage rejects state export before writing metadata",rejected && unavailable.bytes.empty());
        recurrent_bytes_reader unavailable_read(unavailable.bytes);
        rejected=false;
        try { memory.state_read(unavailable_read,0,0); } catch (const std::runtime_error &) { rejected=true; }
        t.assert_true("unbound current storage rejects state import before reading metadata",rejected && unavailable_read.cursor==0);
        const size_t a=ggml_backend_buft_get_alignment(type);
        std::unique_ptr<ggml_backend_memory_arena,decltype(&ggml_backend_memory_arena_free)> arena(
            ggml_backend_memory_arena_new(type,bytes+2*a),ggml_backend_memory_arena_free);
        if (!t.assert_true(arena && ggml_backend_memory_arena_begin(arena.get(),0) &&
                ggml_backend_memory_arena_reserve_at(arena.get(),51,a,bytes,a,GGML_BACKEND_MEMORY_REGION_PERSISTENT,nullptr) &&
                ggml_backend_memory_arena_commit(arena.get()))) return;
        std::unique_ptr<ggml_backend_memory_lease,decltype(&ggml_backend_memory_lease_free)> lease(
            ggml_backend_memory_arena_acquire(arena.get(),51),ggml_backend_memory_lease_free);
        if (!t.assert_true("initial current binding",lease && memory.bind_current_storage(lease.get()))) return;
        t.assert_true(!memory.current_storage_pending());
        std::vector<std::vector<float>> rows;
        for (size_t layer : {0u,2u}) for (bool value : {false,true}) {
            auto * tensor=value ? memory.s_l[layer] : memory.r_l[layer];
            std::vector<float> values(ggml_nelements(tensor),float(3+layer+size_t(value)));values[0]=-0.0f;
            ggml_backend_tensor_set(tensor,values.data(),0,ggml_nbytes(tensor));rows.push_back(std::move(values));
        }
        memory.used=2;
        for (uint32_t seq=0;seq<2;++seq) { memory.cells[seq].pos=llama_pos(17+seq);memory.cells[seq].tail=seq;memory.cells[seq].seq_id.insert(seq); }
        if (!t.assert_true("release current roots",memory.release_current_storage())) return;
        t.assert_true(memory.current_storage_pending() && !memory.r_l[0]->data && !memory.s_l[2]->data);
        t.assert_true("release current roots",memory.release_current_storage());
        t.assert_equal(uint32_t(2),memory.used);
        t.assert_equal(llama_pos(18),memory.cells[1].pos);
        lease.reset();
        if (!t.assert_true(ggml_backend_memory_arena_begin(arena.get(),0) &&
                ggml_backend_memory_arena_reserve_at(arena.get(),52,a,bytes,a,GGML_BACKEND_MEMORY_REGION_PERSISTENT,nullptr) &&
                ggml_backend_memory_arena_commit(arena.get()))) return;
        lease.reset(ggml_backend_memory_arena_acquire(arena.get(),52));
        if (!t.assert_true(bool(lease))) return;
        ggml_backend_buffer_clear(ggml_backend_memory_lease_buffer(lease.get()),0xa5);
        if (!t.assert_true("rebind current roots",memory.bind_current_storage(lease.get()))) return;
        size_t index=0;
        for (size_t layer : {0u,2u}) for (bool value : {false,true}) {
            auto * tensor=value ? memory.s_l[layer] : memory.r_l[layer];
            ggml_backend_tensor_set(tensor,rows[index].data(),0,ggml_nbytes(tensor));
            std::vector<float> actual(rows[index].size());ggml_backend_tensor_get(tensor,actual.data(),0,ggml_nbytes(tensor));
            t.assert_true(!std::memcmp(actual.data(),rows[index].data(),ggml_nbytes(tensor)));++index;
        }
        t.assert_equal(llama_pos(17),memory.cells[0].pos);
        t.assert_equal(int32_t(1),memory.cells[1].tail);
        t.assert_true(memory.cells[0].has_seq_id(0) && memory.cells[1].has_seq_id(1));
    });
    t.test("owned_current_parking_preserves_provisional_rows_and_rollback_indices", [](testing & t) {
        std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_QWEN35,llama_model_default_params()));
        model->hparams.n_layer_all=1;model->hparams.ssm_d_conv=2;model->hparams.ssm_d_inner=8;
        model->hparams.ssm_d_state=8;model->hparams.ssm_n_group=1;
        llama_memory_recurrent memory(*model,GGML_TYPE_F32,GGML_TYPE_F32,false,2,2,3,nullptr,false,true);
        auto * type=ggml_backend_cpu_buffer_type();
        ggml_backend_buffer_type_t measured=nullptr;size_t bytes=0;
        if (!t.assert_true(memory.current_storage_requirements(measured,bytes))) return;
        std::unique_ptr<ggml_backend_memory_arena,decltype(&ggml_backend_memory_arena_free)> arena(
            ggml_backend_memory_arena_new(type,bytes),ggml_backend_memory_arena_free);
        if (!t.assert_true(arena && ggml_backend_memory_arena_begin(arena.get(),0) &&
                ggml_backend_memory_arena_reserve_at(arena.get(),61,0,bytes,ggml_backend_buft_get_alignment(type),
                    GGML_BACKEND_MEMORY_REGION_PERSISTENT,nullptr) && ggml_backend_memory_arena_commit(arena.get()))) return;
        std::unique_ptr<ggml_backend_memory_lease,decltype(&ggml_backend_memory_lease_free)> lease(
            ggml_backend_memory_arena_acquire(arena.get(),61),ggml_backend_memory_lease_free);
        if (!t.assert_true(lease && memory.bind_current_storage(lease.get()))) return;
        std::vector<float> r(ggml_nelements(memory.r_l[0])),s(ggml_nelements(memory.s_l[0]));
        for (size_t i=0;i<r.size();++i) r[i]=float(i+1);
        for (size_t i=0;i<s.size();++i) s[i]=-float(i+1);
        r[0]=-0.0f;
        ggml_backend_tensor_set(memory.r_l[0],r.data(),0,ggml_nbytes(memory.r_l[0]));
        ggml_backend_tensor_set(memory.s_l[0],s.data(),0,ggml_nbytes(memory.s_l[0]));
        memory.rs_idx={2,3};memory.used=2;memory.head=1;memory.n=2;memory.rs_z=1;
        for (uint32_t seq=0;seq<2;++seq) {
            memory.cells[seq].pos=llama_pos(29+seq);memory.cells[seq].tail=seq;
            memory.cells[seq].src=int32_t(1-seq);memory.cells[seq].src0=int32_t(seq);memory.cells[seq].seq_id.insert(seq);
        }
        {
            recurrent_buffer_transfer_fault fault(memory.r_l[0]->buffer,false,2);
            t.assert_true(!memory.park_current_storage());
            t.assert_equal(size_t(2),fault.calls);
            t.assert_true(!memory.current_storage_parked() && !memory.current_storage_pending());
        }
        if (!t.assert_true(memory.park_current_storage())) return;
        t.assert_true(memory.current_storage_pending() && memory.current_storage_parked());
        t.assert_true("parked current rows cannot start a restore",!memory.restore_spill_async(nullptr));
        t.assert_true("parked current rows cannot materialize device state",!memory.materialize_spill_for_device_state());
        t.assert_true("parked current rows cannot capture new snapshots",!memory.capture_spill(nullptr,{}));
        lease.reset();
        if (!t.assert_true(ggml_backend_memory_arena_begin(arena.get(),0) &&
                ggml_backend_memory_arena_reserve_at(arena.get(),62,0,bytes,ggml_backend_buft_get_alignment(type),
                    GGML_BACKEND_MEMORY_REGION_PERSISTENT,nullptr) && ggml_backend_memory_arena_commit(arena.get()))) return;
        lease.reset(ggml_backend_memory_arena_acquire(arena.get(),62));
        if (!t.assert_true(bool(lease))) return;
        ggml_backend_buffer_clear(ggml_backend_memory_lease_buffer(lease.get()),0xa5);
        t.assert_true(!memory.bind_current_storage(lease.get()));
        {
            recurrent_buffer_transfer_fault fault(ggml_backend_memory_lease_buffer(lease.get()),true,2);
            t.assert_true(!memory.restore_current_storage(lease.get()));
            t.assert_equal(size_t(2),fault.calls);
            t.assert_true(memory.current_storage_parked() && memory.current_storage_pending() && !memory.r_l[0]->data);
            t.assert_true(!memory.prepare({}));
        }
        if (!t.assert_true(memory.restore_current_storage(lease.get()))) return;
        t.assert_true(!memory.current_storage_parked() && !memory.current_storage_pending());
        std::vector<float> actual_r(r.size()),actual_s(s.size());
        ggml_backend_tensor_get(memory.r_l[0],actual_r.data(),0,ggml_nbytes(memory.r_l[0]));
        ggml_backend_tensor_get(memory.s_l[0],actual_s.data(),0,ggml_nbytes(memory.s_l[0]));
        t.assert_true(!std::memcmp(r.data(),actual_r.data(),r.size()*sizeof(float)));
        t.assert_true(!std::memcmp(s.data(),actual_s.data(),s.size()*sizeof(float)));
        t.assert_true(memory.rs_idx==std::vector<uint32_t>({2,3}));
        t.assert_equal(2u,memory.used);t.assert_equal(1u,memory.head);t.assert_equal(2u,memory.n);t.assert_equal(1,memory.rs_z);
        t.assert_equal(llama_pos(30),memory.cells[1].pos);
        t.assert_equal(1,memory.cells[0].src);t.assert_equal(1,memory.cells[1].src0);
        t.assert_true(memory.cells[0].has_seq_id(0) && memory.cells[1].has_seq_id(1));
        if (!t.assert_true(memory.park_current_storage())) return;
        memory.clear(true);
        if (!t.assert_true(memory.restore_current_storage(lease.get()))) return;
        ggml_backend_tensor_get(memory.r_l[0],actual_r.data(),0,ggml_nbytes(memory.r_l[0]));
        ggml_backend_tensor_get(memory.s_l[0],actual_s.data(),0,ggml_nbytes(memory.s_l[0]));
        const auto zero=[](const std::vector<float> & values) {
            return std::all_of(values.begin(),values.end(),[](float value) { uint32_t bits;std::memcpy(&bits,&value,sizeof(bits));return bits==0; });
        };
        t.assert_true("clearing a parked bank cannot resurrect its previous rows",zero(actual_r) && zero(actual_s));
        t.assert_true(memory.used==0 && memory.cells[0].is_empty() && memory.cells[1].is_empty() &&
            memory.rs_idx==std::vector<uint32_t>({0,0}));
    });
    t.test("selected_recurrent_rollback_describes_only_the_chosen_logical_sequences", [](testing & t) {
        std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_QWEN35,llama_model_default_params()));
        model->hparams.n_layer_all=1;model->hparams.ssm_d_conv=2;model->hparams.ssm_d_inner=8;
        model->hparams.ssm_d_state=8;model->hparams.ssm_n_group=1;
        for (const auto & selected : {std::vector<llama_seq_id>{0,1},std::vector<llama_seq_id>{1}}) {
            llama_memory_recurrent memory(*model,GGML_TYPE_F32,GGML_TYPE_F32,false,uint32_t(selected.size()),2,3,nullptr,true,true,true);
            if (!t.assert_true(memory.configure_selected_sequences(selected))) return;
            t.assert_equal(uint32_t(3),memory.n_rs_seq);
            t.assert_true(memory.current_storage_pending() && !memory.prepare({}));
            t.assert_true(!memory.spill_bank() && !memory.spill_bank(-1) && !memory.spill_bank(2));
            const llama_recurrent_spill_bank * previous=nullptr;
            for (llama_seq_id sequence=0;sequence<2;++sequence) {
                const auto * bank=memory.spill_bank(sequence);
                const bool chosen=std::find(selected.begin(),selected.end(),sequence)!=selected.end();
                if (!t.assert_equal(chosen,bank!=nullptr)) return;
                if (!chosen) continue;
                t.assert_true(bank!=previous && bank->host_bytes()>0 && !bank->allocated_host_bytes());
                previous=bank;
                const auto * r=bank->staging_tensor(0,false,1),* state=bank->staging_tensor(0,true,1);
                if (!t.assert_true(r && state)) return;
                t.assert_equal(int64_t(model->hparams.n_embd_r()),ggml_nelements(r));
                t.assert_equal(int64_t(model->hparams.n_embd_s()),ggml_nelements(state));
            }
            t.assert_true(!memory.configure_selected_sequences(selected));
        }
    });
    t.test("selected_publication_banks_borrow_disjoint_slices_of_one_stage_grant", [](testing & t) {
        std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_QWEN35,llama_model_default_params()));
        model->hparams.n_layer_all=1;model->hparams.ssm_d_conv=2;model->hparams.ssm_d_inner=8;
        model->hparams.ssm_d_state=8;model->hparams.ssm_n_group=1;
        auto * type=ggml_backend_cpu_buffer_type();const size_t alignment=ggml_backend_buft_get_alignment(type);
        using arena_ptr=std::unique_ptr<ggml_backend_memory_arena,decltype(&ggml_backend_memory_arena_free)>;
        using lease_ptr=std::unique_ptr<ggml_backend_memory_lease,decltype(&ggml_backend_memory_lease_free)>;
        ggml_backend_ptr backend(ggml_backend_cpu_init());auto * raw=backend.get();
        ggml_backend_sched_ptr scheduler(ggml_backend_sched_new(&raw,nullptr,1,128,false,true));
        for (uint32_t depth=3;depth<=7;++depth) for (const auto & selected : {std::vector<llama_seq_id>{0,1},std::vector<llama_seq_id>{1}}) {
            llama_memory_recurrent memory(*model,GGML_TYPE_F32,GGML_TYPE_F32,false,uint32_t(selected.size()),2,depth,nullptr,true,true,true);
            if (!t.assert_true(memory.configure_selected_sequences(selected))) return;
            ggml_backend_buffer_type_t measured=nullptr;size_t current_bytes=0,stage_bytes=0;
            if (!t.assert_true(memory.current_storage_requirements(measured,current_bytes) && measured==type &&
                    memory.spill_stage_requirements(measured,stage_bytes) && measured==type)) return;
            size_t expected=0;
            for (auto sequence : selected) expected+=memory.spill_bank(sequence)->staged_device_bytes();
            if (!t.assert_equal(expected,stage_bytes)) return;
            const auto root=[&](size_t bytes) {
                arena_ptr arena(ggml_backend_memory_arena_new(type,bytes),ggml_backend_memory_arena_free);
                if (!arena || !ggml_backend_memory_arena_begin(arena.get(),0) ||
                        !ggml_backend_memory_arena_reserve_at(arena.get(),82,0,bytes,alignment,GGML_BACKEND_MEMORY_REGION_PERSISTENT,nullptr) ||
                        !ggml_backend_memory_arena_commit(arena.get())) return arena_ptr(nullptr,ggml_backend_memory_arena_free);
                return arena;
            };
            auto short_arena=root(stage_bytes-1);if (!t.assert_true(short_arena!=nullptr)) return;
            lease_ptr short_stage(ggml_backend_memory_arena_acquire(short_arena.get(),82),ggml_backend_memory_lease_free);
            const auto * unchanged=memory.spill_bank(selected.front());
            t.assert_true(!memory.bind_spill_stage(short_stage.get()) && memory.spill_bank(selected.front())==unchanged &&
                !unchanged->allocated_host_bytes());
            arena_ptr arena(ggml_backend_memory_arena_new(type,current_bytes+stage_bytes),ggml_backend_memory_arena_free);
            if (!t.assert_true(arena && ggml_backend_memory_arena_begin(arena.get(),0) &&
                    ggml_backend_memory_arena_reserve_at(arena.get(),81,0,current_bytes,alignment,GGML_BACKEND_MEMORY_REGION_PERSISTENT,nullptr) &&
                    ggml_backend_memory_arena_reserve_at(arena.get(),82,current_bytes,stage_bytes,alignment,GGML_BACKEND_MEMORY_REGION_PERSISTENT,nullptr) &&
                    ggml_backend_memory_arena_commit(arena.get()))) return;
            lease_ptr current(ggml_backend_memory_arena_acquire(arena.get(),81),ggml_backend_memory_lease_free),
                stage(ggml_backend_memory_arena_acquire(arena.get(),82),ggml_backend_memory_lease_free);
            if (!t.assert_true(current && stage && memory.bind_current_storage(current.get()))) return;
            t.assert_true(memory.current_storage_pending() && !memory.prepare({}));
            if (!t.assert_true(memory.bind_spill_stage(stage.get()))) return;
            for (auto * tensor : {memory.r_l[0],memory.s_l[0]}) ggml_backend_sched_set_tensor_backend(scheduler.get(),tensor,backend.get());
            auto * buffer=ggml_backend_memory_lease_buffer(stage.get());const auto base=uintptr_t(ggml_backend_buffer_get_base(buffer));
            size_t offset=0;
            for (auto sequence : selected) {
                const auto * bank=memory.spill_bank(sequence);const auto * tensor=bank ? bank->staging_tensor(0,false,1) : nullptr;
                if (!t.assert_true(bank && tensor && bank->staged_publication_enabled() && bank->allocated_host_bytes()==bank->host_bytes())) return;
                t.assert_equal(base+offset,uintptr_t(ggml_backend_buffer_get_base(tensor->buffer)));
                offset+=bank->staged_device_bytes();
            }
            t.assert_equal(stage_bytes,offset);
            t.assert_true(!memory.bind_spill_stage(stage.get()));
            current.reset();stage.reset();
            if (!t.assert_equal("both outer grants stay leased after caller handles retire",size_t(2),ggml_backend_memory_arena_lease_count(arena.get()))) return;
            const auto capture=[&](std::vector<llama_seq_id> sequences,bool cancel=false) {
                llama_batch_allocr allocator(1);auto input=allocator.ubatch_reserve(depth+1,uint32_t(sequences.size()));
                for (size_t i=0;i<sequences.size();++i) for (size_t row=0;row<=depth;++row) {
                    input.n_seq_id[i*(depth+1)+row]=1;input.seq_id[i*(depth+1)+row]=&sequences[i];
                    input.pos[i*(depth+1)+row]=memory.seq_pos_max(sequences[i])+1+llama_pos(row);
                }
                if (!t.assert_true("complete current/publication grants admit native selected rows",memory.find_slot(input))) return false;
                llm_graph_result result(256);result.reset();auto * context=result.get_ctx();auto * graph=result.get_gf();
                llama_memory_recurrent_context native(&memory);
                struct row {llama_seq_id sequence;uint32_t slot;bool value;ggml_tensor * source;size_t offset;std::vector<float> data;};
                std::vector<row> expected;
                const std::array<uint64_t,2> before{memory.spill_generation(0),memory.spill_generation(1)};
                for (bool value : {false,true}) {
                    std::vector<ggml_tensor *> copies;
                    for (uint32_t slot=1;slot<=depth;++slot) {
                        const auto * stage=memory.spill_bank(sequences.front())->staging_tensor(0,value,slot);
                        const auto width=ggml_nelements(stage);
                        auto * source=value ? ggml_new_tensor_2d(context,GGML_TYPE_F32,width,int64_t(sequences.size())) :
                            ggml_new_tensor_3d(context,GGML_TYPE_F32,1,width,int64_t(sequences.size()));
                        for (size_t index=0;index<sequences.size();++index) {
                            const auto sequence=sequences[index];std::vector<float> data(size_t(width),0);
                            for (size_t i=0;i<data.size();++i) data[i]=float(sequence*1000+value*100+slot*10+i);
                            expected.push_back({sequence,slot,value,source,index*size_t(width)*sizeof(float),std::move(data)});
                        }
                        if (!t.assert_true(native.stage_sequence_snapshot(context,result,source,0,value,slot,value ? 1 : 2,input,copies))) return false;
                    }
                    if (!t.assert_true(native.publish_staged(context,graph,0,value,copies))) return false;
                }
                const auto & outputs=result.get_recurrent_snapshots();
                ggml_backend_buffer_ptr inputs(ggml_backend_alloc_ctx_tensors(context,backend.get()));
                if (!t.assert_true(inputs && scheduler)) return false;
                for (const auto & row : expected) ggml_backend_tensor_set(row.source,row.data.data(),row.offset,row.data.size()*sizeof(float));
                if (!t.assert_true(memory.begin_spill_capture(input))) return false;
                t.assert_true(!memory.begin_spill_capture(input));
                if (!t.assert_equal(GGML_STATUS_SUCCESS,ggml_backend_graph_compute(backend.get(),graph))) return false;
                if (cancel) {
                    memory.abort_spill_capture();
                    for (auto sequence : sequences) {
                        t.assert_equal(before[sequence],memory.spill_generation(sequence));
                        t.assert_true(!memory.spill_bank(sequence)->snapshot_data(0,true,1,before[sequence]));
                    }
                    if (!t.assert_true(memory.begin_spill_capture(input) &&
                            ggml_backend_graph_compute(backend.get(),graph)==GGML_STATUS_SUCCESS)) return false;
                }
                if (sequences.size()==2) {
                    auto missing=outputs;
                    missing.erase(std::remove_if(missing.begin(),missing.end(),[&](const auto & output) {return output.sequence==sequences.back();}),missing.end());
                    if (!t.assert_true("a neighboring sequence cannot supply missing snapshot proof",!memory.capture_spill(scheduler.get(),missing))) return false;
                }
                if (!t.assert_true(memory.capture_spill(scheduler.get(),outputs) && memory.complete_spill())) return false;
                for (const auto & row : expected) {
                    const auto generation=memory.spill_generation(row.sequence);
                    const auto * actual=memory.spill_bank(row.sequence)->snapshot_data(0,row.value,row.slot,generation);
                    if (!t.assert_true(generation==before[row.sequence]+1 && actual && !std::memcmp(actual,row.data.data(),row.data.size()*sizeof(float)))) return false;
                }
                llama_seq_id invalid=2;input.seq_id[0]=&invalid;
                t.assert_true(!memory.begin_spill_capture(input));
                for (auto sequence : sequences) t.assert_equal(before[sequence]+1,memory.spill_generation(sequence));
                return true;
            };
            if (!capture(selected)) return;
            if (selected.size()==2) {
                const auto generation=memory.spill_generation(0);
                const auto * bank=memory.spill_bank(0);const auto * tensor=bank->staging_tensor(0,true,1);
                const auto * data=static_cast<const uint8_t *>(bank->snapshot_data(0,true,1,generation));
                std::vector<uint8_t> unchanged(data,data+ggml_nbytes(tensor));
                if (!capture({1},true)) return;
                const auto * retained=bank->snapshot_data(0,true,1,generation);
                t.assert_true(memory.spill_generation(0)==generation && retained && !std::memcmp(retained,unchanged.data(),unchanged.size()));
                if (!capture(selected)) return;
                t.assert_equal(uint64_t(2),memory.spill_generation(0));t.assert_equal(uint64_t(3),memory.spill_generation(1));
            }
            const auto rolled=selected.back();
            const auto physical=[&](llama_seq_id sequence) {
                for (uint32_t row=0;row<memory.size;++row) if (memory.cells[row].has_seq_id(sequence)) return row;
                return memory.size;
            };
            const auto mark=[&] {
                for (auto * tensor : {memory.r_l[0],memory.s_l[0]}) {
                    std::vector<float> values(size_t(ggml_nelements(tensor)));
                    for (size_t i=0;i<values.size();++i) values[i]=float(7000+i);
                    ggml_backend_tensor_set(tensor,values.data(),0,values.size()*sizeof(float));
                }
            };
            const auto verify_restore=[&](uint32_t destination,uint32_t snapshot) {
                const auto generation=memory.spill_generation(rolled);const auto * bank=memory.spill_bank(rolled);
                for (bool value : {false,true}) {
                    auto * tensor=value ? memory.s_l[0] : memory.r_l[0];const auto count=size_t(tensor->ne[0]);
                    std::vector<float> actual(size_t(ggml_nelements(tensor)));
                    ggml_backend_tensor_get(tensor,actual.data(),0,actual.size()*sizeof(float));
                    const auto * expected=bank->snapshot_data(0,value,snapshot,generation);
                    if (!t.assert_true(expected && !std::memcmp(actual.data()+destination*count,expected,count*sizeof(float)))) return false;
                    for (uint32_t row=0;row<memory.size;++row) if (row!=destination)
                        for (size_t i=0;i<count;++i) if (!t.assert_equal(float(7000+row*count+i),actual[row*count+i])) return false;
                }
                return true;
            };
            mark();const auto row=physical(rolled);const auto last=memory.seq_pos_max(rolled);
            if (!t.assert_true("selected rollback records one native snapshot",row<memory.size && memory.seq_rm(rolled,last-1,-1) && memory.rs_idx[rolled]==2)) return;
            if (!t.assert_true("state export materializes the selected physical row",memory.materialize_spill_for_device_state())) return;
            if (!verify_restore(row,2)) return;
            t.assert_equal(uint32_t(0),memory.rs_idx[rolled]);
            t.assert_true(!memory.seq_rm(rolled,memory.seq_pos_max(rolled),-1));
            if (!capture(selected)) return;
            mark();const auto source=physical(rolled);
            if (!t.assert_true(memory.seq_rm(rolled,memory.seq_pos_max(rolled)-1,-1))) return;
            auto reversed=selected;std::reverse(reversed.begin(),reversed.end());
            llama_batch_allocr allocator(1);auto next=allocator.ubatch_reserve(1,uint32_t(reversed.size()));
            for (size_t i=0;i<reversed.size();++i) {
                next.n_seq_id[i]=1;next.seq_id[i]=&reversed[i];next.pos[i]=memory.seq_pos_max(reversed[i])+1;
            }
            if (!t.assert_true("native input rows can reorder the selected physical banks",memory.find_slot(next))) return;
            if (!t.assert_true(memory.restore_spill_async(scheduler.get()))) return;
            if (!t.assert_true("the next capture completes its pending selected rollback restore",memory.begin_spill_capture(next))) return;
            memory.abort_spill_capture();
            if (!t.assert_true("rollback restore follows the pre-input source row",memory.complete_restore_spill())) return;
            if (!verify_restore(source,2)) return;
            llama_memory_recurrent_context reordered(&memory);
            t.assert_equal(int32_t(source),reordered.s_copy(0));
            t.assert_equal(uint32_t(0),memory.rs_idx[rolled]);
            std::array<std::vector<uint8_t>,2> parked;
            for (bool value : {false,true}) {
                auto * tensor=value ? memory.s_l[0] : memory.r_l[0];auto & bytes=parked[value];bytes.resize(ggml_nbytes(tensor));
                ggml_backend_tensor_get(tensor,bytes.data(),0,bytes.size());
            }
            const std::array<uint64_t,2> epochs{memory.spill_generation(0),memory.spill_generation(1)};
            ggml_backend_sched_reset(scheduler.get());
            if (!t.assert_true(memory.park_current_storage() && memory.current_storage_parked())) return;
            if (!t.assert_equal("park releases both current and publication grants",size_t(0),ggml_backend_memory_arena_lease_count(arena.get()))) return;
            size_t quote=0;
            if (!t.assert_true(memory.spill_stage_requirements(measured,quote))) return;
            t.assert_equal(stage_bytes,quote);
            if (!t.assert_true(ggml_backend_memory_arena_begin(arena.get(),0) && ggml_backend_memory_arena_commit(arena.get()))) return;
            ggml_backend_buffer_clear(ggml_backend_memory_arena_parent(arena.get()),0x5a);
            if (!t.assert_true(ggml_backend_memory_arena_begin(arena.get(),0) &&
                    ggml_backend_memory_arena_reserve_at(arena.get(),81,0,current_bytes,alignment,GGML_BACKEND_MEMORY_REGION_PERSISTENT,nullptr) &&
                    ggml_backend_memory_arena_reserve_at(arena.get(),82,current_bytes,stage_bytes,alignment,GGML_BACKEND_MEMORY_REGION_PERSISTENT,nullptr) &&
                    ggml_backend_memory_arena_commit(arena.get()))) return;
            current.reset(ggml_backend_memory_arena_acquire(arena.get(),81));stage.reset(ggml_backend_memory_arena_acquire(arena.get(),82));
            if (!t.assert_true(current && stage && !memory.restore_current_storage(current.get()) && memory.current_storage_pending())) return;
            if (selected.size()==2) {
                {
                    recurrent_stage_bind_fault fault(memory,type);
                    t.assert_true(!memory.restore_current_storage(current.get(),stage.get()) && fault.fired);
                }
                current.reset();stage.reset();
                const bool live=memory.spill_bank(0)->staged_publication_enabled() || memory.spill_bank(1)->staged_publication_enabled();
                if (!t.assert_true("a failed partial rebind cannot leave unleased live stage views",
                        !live || ggml_backend_memory_arena_lease_count(arena.get())>0)) return;
                current.reset(ggml_backend_memory_arena_acquire(arena.get(),81));stage.reset(ggml_backend_memory_arena_acquire(arena.get(),82));
            }
            if (!t.assert_true(current && stage && memory.restore_current_storage(current.get(),stage.get()) && !memory.current_storage_pending())) return;
            for (bool value : {false,true}) {
                auto * tensor=value ? memory.s_l[0] : memory.r_l[0];std::vector<uint8_t> actual(ggml_nbytes(tensor));
                ggml_backend_tensor_get(tensor,actual.data(),0,actual.size());t.assert_true(actual==parked[value]);
            }
            for (auto sequence : selected) {
                t.assert_equal(epochs[sequence],memory.spill_generation(sequence));
                t.assert_true(memory.spill_bank(sequence)->snapshot_data(0,true,1,epochs[sequence])!=nullptr);
            }
            current.reset();stage.reset();t.assert_equal(size_t(2),ggml_backend_memory_arena_lease_count(arena.get()));
            recurrent_bytes_writer checkpoint;memory.state_write(checkpoint,rolled,0);
            recurrent_bytes_reader imported(checkpoint.bytes);memory.state_read(imported,rolled,0);
            recurrent_bytes_writer exact;memory.state_write(exact,rolled,0);
            if (!t.assert_true(checkpoint.bytes==exact.bytes)) return;
            if (!t.assert_true("replacing one logical state invalidates only its old snapshot proof",
                    !memory.spill_bank(rolled)->snapshot_data(0,true,1,epochs[rolled]))) return;
            if (selected.size()==2) t.assert_true(memory.spill_bank(0)->snapshot_data(0,true,1,epochs[0])!=nullptr);
            t.assert_true(!memory.seq_rm(rolled,memory.seq_pos_max(rolled),-1));
            memory.clear(false);
            for (auto sequence : selected) t.assert_true(!memory.spill_bank(sequence)->snapshot_data(0,true,1,memory.spill_generation(sequence)));
        }
    });
    t.test("current_and_publication_bindings_admit_execution_together", [](testing & t) {
        std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_QWEN35,llama_model_default_params()));
        model->hparams.n_layer_all=1;model->hparams.ssm_d_conv=2;model->hparams.ssm_d_inner=8;
        model->hparams.ssm_d_state=8;model->hparams.ssm_n_group=1;
        llama_memory_recurrent memory(*model,GGML_TYPE_F32,GGML_TYPE_F32,false,1,1,3,nullptr,true,true);
        auto * type=ggml_backend_cpu_buffer_type();
        ggml_backend_buffer_type_t measured=nullptr;size_t current_bytes=0;
        if (!t.assert_true(memory.current_storage_requirements(measured,current_bytes))) return;
        llama_recurrent_spill_layout rows;
        llama_recurrent_stage_layout stage;
        const size_t alignment=ggml_backend_buft_get_alignment(type);
        if (!t.assert_true(llama_recurrent_spill_layout_make({{GGML_TYPE_F32,GGML_TYPE_F32,
                    uint64_t(model->hparams.n_embd_r()),uint64_t(model->hparams.n_embd_s()),true}},1,3,rows) &&
                llama_recurrent_stage_layout_make(rows,2,alignment,stage))) return;
        std::unique_ptr<ggml_backend_memory_arena,decltype(&ggml_backend_memory_arena_free)> arena(
            ggml_backend_memory_arena_new(type,current_bytes+stage.total_bytes),ggml_backend_memory_arena_free);
        if (!t.assert_true(arena && ggml_backend_memory_arena_begin(arena.get(),0) &&
                ggml_backend_memory_arena_reserve_at(arena.get(),81,0,current_bytes,alignment,GGML_BACKEND_MEMORY_REGION_PERSISTENT,nullptr) &&
                ggml_backend_memory_arena_reserve_at(arena.get(),82,current_bytes,stage.total_bytes,alignment,GGML_BACKEND_MEMORY_REGION_PERSISTENT,nullptr) &&
                ggml_backend_memory_arena_commit(arena.get()))) return;
        std::unique_ptr<ggml_backend_memory_lease,decltype(&ggml_backend_memory_lease_free)> current(
            ggml_backend_memory_arena_acquire(arena.get(),81),ggml_backend_memory_lease_free), publication(
            ggml_backend_memory_arena_acquire(arena.get(),82),ggml_backend_memory_lease_free);
        if (!t.assert_true(current && publication && memory.bind_current_storage(current.get()))) return;
        ggml_backend_buffer_clear(ggml_backend_memory_lease_buffer(current.get()),0);
        t.assert_true("current rows alone cannot admit execution without publication storage",!memory.prepare({}));
        t.assert_true("missing publication storage cannot silently fall back to native capture",!memory.begin_spill_capture(4));
        if (!t.assert_true(memory.bind_spill_stage(publication.get()) && memory.prepare({}))) return;
        if (!t.assert_true(memory.park_current_storage())) return;
        t.assert_true(memory.spill_bank()->publication_stage_parked() && memory.current_storage_parked());
        t.assert_true(!memory.restore_current_storage(current.get()));
        t.assert_true(!memory.restore_current_storage(current.get(),current.get()));
        t.assert_true(memory.current_storage_pending());
        if (!t.assert_true(memory.restore_current_storage(current.get(),publication.get()))) return;
        t.assert_true(memory.prepare({}) && memory.staged_publication_enabled() && !memory.current_storage_parked());
    });
    t.test("deferred_publication_prepares_resources_without_private_stage_allocation", [&](testing & t) {
        const bool gpu=argc>1 && !std::strcmp(argv[1],"--cuda");
        ggml_backend_load_all();
        auto * device=ggml_backend_dev_by_name(gpu ? "CUDA0" : "CPU");
        if (!t.assert_true(device!=nullptr)) return;
        auto * type=ggml_backend_dev_buffer_type(device);
        auto * host=ggml_backend_dev_host_buffer_type(device);
        if (!host) host=ggml_backend_cpu_buffer_type();
        llama_recurrent_spill_layout layout;
        if (!t.assert_true(llama_recurrent_spill_layout_make({{GGML_TYPE_F32,GGML_TYPE_F32,32,64,true}},1,3,layout))) return;
        auto bank=llama_recurrent_spill_bank::create(layout,host,131,0);
        if (!t.assert_true(bool(bank))) return;
        size_t allocations=0;
        const auto allocate=type->iface.alloc_buffer;
        static size_t * observed=nullptr;observed=&allocations;
        type->iface.alloc_buffer=[](ggml_backend_buffer_type_t,size_t) -> ggml_backend_buffer_t { ++*observed;return nullptr; };
        bool prepared=false;
        try { prepared=bank->enable_publication(device,type,2,true); } catch (...) {}
        type->iface.alloc_buffer=allocate;observed=nullptr;
        if (!t.assert_true("deferred publication setup does not allocate a device stage",prepared && allocations==0)) return;
        t.assert_true(bank->publication_stage_parked() && !bank->staged_publication_enabled() && bank->staged_device_bytes()==0);
        t.assert_true(!bank->begin_capture({131,0,1}) && !bank->begin_staged_capture({131,0,1},4));
        llama_recurrent_stage_layout stage;
        if (!t.assert_true(llama_recurrent_stage_layout_make(layout,2,ggml_backend_buft_get_alignment(type),stage))) return;
        std::unique_ptr<ggml_backend_memory_arena,decltype(&ggml_backend_memory_arena_free)> arena(
            ggml_backend_memory_arena_new(type,stage.total_bytes),ggml_backend_memory_arena_free);
        if (!t.assert_true(arena && ggml_backend_memory_arena_begin(arena.get(),0) &&
                ggml_backend_memory_arena_reserve_at(arena.get(),71,0,stage.total_bytes,stage.alignment,
                    GGML_BACKEND_MEMORY_REGION_PERSISTENT,nullptr) && ggml_backend_memory_arena_commit(arena.get()))) return;
        std::unique_ptr<ggml_backend_memory_lease,decltype(&ggml_backend_memory_lease_free)> lease(
            ggml_backend_memory_arena_acquire(arena.get(),71),ggml_backend_memory_lease_free);
        if (!t.assert_true(lease && bank->bind_publication_stage(lease.get()))) return;
        t.assert_true(bank->staged_publication_enabled() && !bank->publication_stage_parked() && bank->staged_device_bytes()==stage.total_bytes);
    });
    t.test("parking_publication_releases_device_stage_without_erasing_host_or_peer", [&](testing & t) {
        const bool gpu=argc>1 && !std::strcmp(argv[1],"--cuda");
        ggml_backend_load_all();
        auto * device=ggml_backend_dev_by_name(gpu ? "CUDA0" : "CPU");
        if (!t.assert_true(device != nullptr)) return;
        auto * type=ggml_backend_dev_buffer_type(device);
        auto * host=ggml_backend_dev_host_buffer_type(device);
        if (!host) host=ggml_backend_cpu_buffer_type();
        ggml_backend_ptr backend(ggml_backend_dev_init(device,nullptr));
        llama_recurrent_spill_layout layout;
        if (!t.assert_true(llama_recurrent_spill_layout_make({{GGML_TYPE_F32,GGML_TYPE_F32,4,8,true}},1,3,layout))) return;
        auto bank=llama_recurrent_spill_bank::create(layout,host,127,0);
        auto peer=llama_recurrent_spill_bank::create(layout,host,127,1);
        if (!t.assert_true(bank && peer)) return;
        recurrent_copy_backend_probe probe(device);
        if (!t.assert_true(bank->enable_publication(device,type,2))) return;
        device->iface.init_backend=probe.original;
        if (!t.assert_true(peer->enable_publication(device,type,2))) return;
        const auto * peer_stage=peer->staging_tensor(0,true,3);
        const size_t bytes=bank->staged_device_bytes(),alignment=ggml_backend_buft_get_alignment(type);
        std::unique_ptr<ggml_backend_memory_arena,decltype(&ggml_backend_memory_arena_free)> arena(
            ggml_backend_memory_arena_new(type,bytes+2*alignment),ggml_backend_memory_arena_free);
        if (!t.assert_true(arena && ggml_backend_memory_arena_begin(arena.get(),0) &&
                ggml_backend_memory_arena_reserve_at(arena.get(),9,alignment,bytes,alignment,GGML_BACKEND_MEMORY_REGION_PERSISTENT,nullptr) &&
                ggml_backend_memory_arena_commit(arena.get()))) return;
        auto * lease=ggml_backend_memory_arena_acquire(arena.get(),9);
        if (!t.assert_true(lease && bank->bind_publication_stage(lease))) return;
        ggml_backend_memory_lease_free(lease);
        ggml_context_ptr ctx(ggml_init({32768,nullptr,true}));
        auto * r=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,4);
        auto * state=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,8);
        ggml_backend_buffer_ptr input(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()));
        if (!t.assert_true(bool(input))) return;
        const float expected_r[]={-0.0f,1.25f,-2.5f,3.75f};
        const float expected_s[]={-0.0f,2,3,4,5,6,7,8};
        ggml_backend_tensor_set(r,expected_r,0,sizeof(expected_r));
        ggml_backend_tensor_set(state,expected_s,0,sizeof(expected_s));
        for (auto * item : {bank.get(),peer.get()}) {
            const uint64_t sequence=item==bank.get() ? 0 : 1;
            if (!t.assert_true(item->begin_capture({127,sequence,1}) && item->capture(backend.get(),0,false,2,r,0) &&
                    item->capture(backend.get(),0,true,2,state,0) && item->publish_capture() && item->complete_capture())) return;
        }
        const auto * saved=bank->snapshot_data(0,false,2,1);
        if (!t.assert_true(saved && bank->begin_staged_capture({127,0,2},4))) return;
        t.assert_true(!bank->park_publication_stage());
        bank->cancel();
        const auto * before_park=bank->staging_tensor(0,false,1);
        probe.fail_prepare=true;
        t.assert_true(!bank->park_publication_stage());
        t.assert_true(!bank->publication_stage_parked() && bank->staging_tensor(0,false,1)==before_park);
        t.assert_equal(bytes,bank->staged_device_bytes());
        if (!t.assert_true(bank->park_publication_stage() && bank->publication_stage_parked())) return;
        t.assert_equal(size_t(0),bank->staged_device_bytes());
        t.assert_true(!bank->staging_tensor(0,true,3) && !bank->publication_tensor(0,false,2));
        t.assert_true(bank->park_publication_stage());
        t.assert_true(!bank->enable_publication(device,type,2));
        t.assert_true(!bank->begin_staged_capture({127,0,2},4));
        const bool direct=bank->begin_capture({127,0,2});
        if (direct) bank->cancel();
        t.assert_true("parked publication cannot fall back to a direct capture",!direct);
        t.assert_true(peer->staging_tensor(0,true,3)==peer_stage);
        t.assert_true(bank->snapshot_data(0,false,2,1)==saved && !std::memcmp(saved,expected_r,sizeof(expected_r)));
        t.assert_equal(uint64_t(1),peer->snapshot_generation(0,false,2));
        if (!t.assert_true(ggml_backend_memory_arena_begin(arena.get(),0) &&
                ggml_backend_memory_arena_reserve_at(arena.get(),10,alignment,bytes,alignment,GGML_BACKEND_MEMORY_REGION_PERSISTENT,nullptr) &&
                ggml_backend_memory_arena_commit(arena.get()))) return;
        auto * replacement=ggml_backend_memory_arena_acquire(arena.get(),10);
        if (!t.assert_true(replacement != nullptr)) return;
        ggml_backend_buffer_clear(ggml_backend_memory_lease_buffer(replacement),0xa5);
        t.assert_true(!std::memcmp(saved,expected_r,sizeof(expected_r)));
        probe.fail_prepare=true;
        t.assert_true(!bank->bind_publication_stage(replacement));
        t.assert_true(bank->publication_stage_parked() && !bank->staging_tensor(0,false,1));
        t.assert_true(bank->snapshot_data(0,false,2,1)==saved);
        t.assert_true(bank->bind_publication_stage(replacement));
        ggml_backend_memory_lease_free(replacement);
        t.assert_true(!bank->publication_stage_parked() && bank->staged_device_bytes()==bytes);
        t.assert_true(bank->staging_tensor(0,true,3) != nullptr);
        t.assert_true(bank->begin_restore({127,0,1},2) && bank->restore(backend.get(),0,false,r,0) &&
            bank->restore(backend.get(),0,true,state,0) && bank->publish_restore() && bank->complete_restore());
        float actual_r[4],actual_s[8];
        ggml_backend_tensor_get(r,actual_r,0,sizeof(actual_r));ggml_backend_tensor_get(state,actual_s,0,sizeof(actual_s));
        t.assert_true(!std::memcmp(actual_r,expected_r,sizeof(actual_r)) && !std::memcmp(actual_s,expected_s,sizeof(actual_s)));
        t.assert_true(!bank->begin_restore({127,1,1},2));
    });
    t.test("publication_stage_can_borrow_exact_persistent_storage_but_not_during_capture", [](testing & t) {
        auto * type=ggml_backend_cpu_buffer_type();
        ggml_backend_ptr backend(ggml_backend_cpu_init());
        auto * device=ggml_backend_get_device(backend.get());
        llama_recurrent_spill_layout snapshots;
        if (!t.assert_true(llama_recurrent_spill_layout_make({{GGML_TYPE_F32,GGML_TYPE_F32,4,8,true}},1,3,snapshots))) return;
        auto bank=llama_recurrent_spill_bank::create(snapshots,type,77,0);
        if (!t.assert_true(bank && bank->enable_publication(device,type,2))) return;
        const size_t bytes=bank->staged_device_bytes(), a=ggml_backend_buft_get_alignment(type);
        auto * arena=ggml_backend_memory_arena_new(type,bytes+2*a);
        if (!t.assert_true(arena && ggml_backend_memory_arena_begin(arena,0) &&
                ggml_backend_memory_arena_reserve_at(arena,9,a,bytes,a,GGML_BACKEND_MEMORY_REGION_PERSISTENT,nullptr) &&
                ggml_backend_memory_arena_commit(arena))) return;
        auto * lease=ggml_backend_memory_arena_acquire(arena,9);
        if (!t.assert_true(lease && bank->bind_publication_stage(lease))) return;
        const auto base=uintptr_t(ggml_backend_buffer_get_base(ggml_backend_memory_lease_buffer(lease)));
        auto * tensor=bank->staging_tensor(0,true,3);
        t.assert_true(uintptr_t(tensor->data) >= base && uintptr_t(tensor->data)+ggml_nbytes(tensor) <= base+bytes);
        for (bool persistent : {false,true}) {
            const size_t size=persistent ? bytes-1 : bytes;
            std::unique_ptr<ggml_backend_memory_arena,decltype(&ggml_backend_memory_arena_free)> bad(
                ggml_backend_memory_arena_new(type,bytes+2*a),ggml_backend_memory_arena_free);
            if (!t.assert_true(bad && ggml_backend_memory_arena_begin(bad.get(),0) &&
                    ggml_backend_memory_arena_reserve_at(bad.get(),9,a,size,a,persistent ? GGML_BACKEND_MEMORY_REGION_PERSISTENT : 0,nullptr) &&
                    ggml_backend_memory_arena_commit(bad.get()))) return;
            auto * short_grant=ggml_backend_memory_arena_acquire(bad.get(),9);
            t.assert_true(!bank->bind_publication_stage(short_grant) && bank->staging_tensor(0,true,3) == tensor);
            ggml_backend_memory_lease_free(short_grant);
        }
        t.assert_true(bank->begin_staged_capture({77,0,1},4));
        t.assert_true(!bank->bind_publication_stage(lease));
        bank->cancel();
        t.assert_true(bank->staged_publication_enabled());
        ggml_backend_memory_lease_free(lease); ggml_backend_memory_arena_free(arena);
        t.assert_true(bank->staging_tensor(0,true,3) == tensor);
    });
    t.test("queued_recurrent_transfer_failure_invalidates_or_poison_restores", [](testing & t) {
        ggml_backend_ptr backend(ggml_backend_cpu_init());
        llama_recurrent_spill_layout layout;
        if (!t.assert_true(llama_recurrent_spill_layout_make(
                {{GGML_TYPE_F32,GGML_TYPE_F32,4,8,true}},1,7,layout))) return;
        auto bank=llama_recurrent_spill_bank::create(layout,ggml_backend_cpu_buffer_type(),81,0);
        ggml_context_ptr ctx(ggml_init({32768,nullptr,true}));
        auto * r=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,4);
        auto * s=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,8);
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()));
        if (!t.assert_true(bank && buffer && bank->begin_capture({81,0,1}))) return;
        ggml_backend_tensor_memset(r,0x11,0,ggml_nbytes(r));
        ggml_backend_tensor_memset(s,0x22,0,ggml_nbytes(s));
        for (uint32_t slot=1;slot<=7;++slot)
            if (!t.assert_true(bank->capture(backend.get(),0,false,slot,r,0) &&
                    bank->capture(backend.get(),0,true,slot,s,0))) return;
        if (!t.assert_true(bank->publish_capture() && bank->complete_capture() && bank->begin_capture({81,0,2}))) return;
        ggml_backend_tensor_memset(r,0x33,0,ggml_nbytes(r));
        {
            recurrent_transfer_fault fault(backend.get(),false);
            t.assert_true(!bank->capture(backend.get(),0,false,1,r,0));
        }
        t.assert_true(!bank->publish_capture());
        bank->cancel();
        t.assert_equal(uint64_t(0),bank->snapshot_generation(0,false,1));
        t.assert_true(!bank->snapshot_pristine(0,false,1) && !bank->begin_restore({81,0,1},1));
        t.assert_equal(uint64_t(1),bank->snapshot_generation(0,false,2));
        if (!t.assert_true(bank->begin_restore({81,0,1},2))) return;
        {
            recurrent_transfer_fault fault(backend.get(),true);
            t.assert_true(!bank->restore(backend.get(),0,false,r,0));
        }
        bank->cancel();
        t.assert_true(!bank->begin_capture({81,0,2}) && !bank->begin_restore({81,0,1},2));
        bank->reset(true);
        if (!t.assert_true(bank->begin_capture({81,0,2}) &&
                bank->capture(backend.get(),0,false,2,r,0) && bank->capture(backend.get(),0,true,2,s,0) &&
                bank->publish_capture() && bank->complete_capture() && bank->begin_restore({81,0,2},2) &&
                bank->restore(backend.get(),0,false,r,0))) return;
        {
            recurrent_transfer_fault fault(backend.get(),true);
            t.assert_true(!bank->restore(backend.get(),0,true,s,0));
        }
        t.assert_true(!bank->publish_restore());
        bank->cancel();
        t.assert_true(!bank->begin_capture({81,0,3}));
    });
    t.test("layout_matches_stock_recurrent_rows_and_counts_exact_bytes", [](testing & t) {
        const std::vector<llama_recurrent_spill_layer> layers{
            {GGML_TYPE_F32,GGML_TYPE_F32,24576,786432,true},
            {GGML_TYPE_F32,GGML_TYPE_F32,24576,786432,true},
            {GGML_TYPE_F32,GGML_TYPE_F32,24576,786432,false},
        };
        llama_recurrent_spill_layout layout;
        if (!t.assert_true(llama_recurrent_spill_layout_make(layers,1,3,layout))) return;
        t.assert_equal(size_t(98304),layout.layers[0].r_row_bytes);
        t.assert_equal(size_t(3145728),layout.layers[0].s_row_bytes);
        t.assert_equal(size_t(6488064),layout.device_current_bytes);
        t.assert_equal(size_t(19464192),layout.host_snapshot_bytes);
        t.assert_equal(size_t(25952256),layout.stock_device_bytes);
        t.assert_equal(layout.device_current_bytes+layout.host_snapshot_bytes,layout.stock_device_bytes);
        t.assert_equal(size_t(3244032),layout.device_stage_bytes);
        t.assert_true(!layout.layers[2].enabled);
        t.assert_equal(layout.layers[0].s_offset+3*layout.layers[0].s_row_bytes,layout.layers[1].r_offset);
        ggml_context * ctx=ggml_init({32768,nullptr,true});
        auto * r=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,24576,4);
        auto * s=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,786432,4);
        t.assert_equal(2*(ggml_nbytes(r)+ggml_nbytes(s)),layout.stock_device_bytes);
        ggml_free(ctx);
    });
    t.test("depth_seven_two_slot_footprint_is_bounded_by_one_layer", [](testing & t) {
        std::vector<llama_recurrent_spill_layer> layers(48,{GGML_TYPE_F32,GGML_TYPE_F32,30720,786432,true});
        llama_recurrent_spill_layout snapshots;
        llama_recurrent_stage_layout stage;
        if (!t.assert_true(llama_recurrent_spill_layout_make(layers,1,7,snapshots) &&
                llama_recurrent_stage_layout_make(snapshots,2,256,stage))) return;
        t.assert_equal(size_t(149.625*1048576),snapshots.device_current_bytes);
        t.assert_equal(size_t(1047.375*1048576),snapshots.host_snapshot_bytes);
        t.assert_equal(size_t(43.640625*1048576),stage.total_bytes);
        for (uint32_t slot=0;slot<2;++slot) for (uint32_t snapshot=1;snapshot<=7;++snapshot)
            for (bool value : {false,true}) {
                size_t offset=0,bytes=0;
                t.assert_true(llama_recurrent_stage_region(stage,47,value,snapshot,slot,offset,bytes));
                t.assert_true(offset <= stage.total_bytes && bytes <= stage.total_bytes-offset);
            }
    });
    t.test("wide_publication_reuses_two_slots_and_restores_every_candidate", [&](testing & t) {
        ggml_backend_load_all();
        const bool gpu=argc > 1 && !std::strcmp(argv[1],"--cuda");
        auto * device=ggml_backend_dev_by_name(gpu ? "CUDA0" : "CPU");
        if (!t.assert_true(device != nullptr)) return;
        ggml_backend_ptr producer(ggml_backend_dev_init(device,nullptr));
        auto * host_type=ggml_backend_dev_host_buffer_type(device);
        if (!host_type) host_type=ggml_backend_cpu_buffer_type();
        std::vector<llama_recurrent_spill_layer> layers(4,{GGML_TYPE_F32,GGML_TYPE_F32,4,8,true});
        layers[1].enabled=false;
        llama_recurrent_spill_layout layout;
        if (!t.assert_true(llama_recurrent_spill_layout_make(layers,2,7,layout))) return;
        auto bank=llama_recurrent_spill_bank::create(layout,host_type,83,0);
        recurrent_copy_backend_probe probe(device);
        if (!t.assert_true(bank && bank->enable_publication(device,ggml_backend_dev_buffer_type(device),2))) return;
        if (!gpu) probe.defer_downloads();
        const size_t original_bytes=bank->host_bytes(),original_stage=bank->staged_device_bytes();
        ggml_context_ptr ctx(ggml_init({1024*1024,nullptr,true}));
        std::vector<ggml_tensor *> inputs(4*7*2,nullptr);
        std::vector<std::vector<float>> expected(inputs.size());
        std::vector<uint64_t> generations(inputs.size(),0);
        const auto index=[](size_t layer,uint32_t snapshot,bool value) { return (layer*7+snapshot-1)*2+size_t(value); };
        for (size_t layer : {0u,2u,3u}) for (uint32_t snapshot=1;snapshot<=7;++snapshot)
            for (bool value : {false,true}) {
                const size_t i=index(layer,snapshot,value);
                inputs[i]=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,value ? 8 : 4,2);
                expected[i].resize(value ? 16 : 8,0.0f);
            }
        ggml_backend_buffer_ptr source(ggml_backend_alloc_ctx_tensors(ctx.get(),producer.get()));
        if (!t.assert_true(bool(source))) return;
        for (uint32_t width=1;width<=8;++width) {
            auto * graph=ggml_new_graph_custom(ctx.get(),256,false);
            for (size_t layer : {0u,2u,3u}) {
                for (bool value : {false,true}) for (uint32_t j=1;j<= (value ? width-1 : 7);++j) {
                    const uint32_t snapshot=value ? j : 8-j;
                    const size_t i=index(layer,snapshot,value);
                    for (size_t k=0;k<expected[i].size();++k) expected[i][k]=float(10000*width+1000*layer+10*snapshot+value+k);
                    ggml_backend_tensor_set(inputs[i],expected[i].data(),0,ggml_nbytes(inputs[i]));
                    auto * staged=ggml_cpy(ctx.get(),inputs[i],bank->staging_tensor(layer,value,snapshot));
                    ggml_build_forward_expand(graph,ggml_cpy(ctx.get(),staged,bank->publication_tensor(layer,value,snapshot)));
                }
            }
            if (!t.assert_true(bank->begin_staged_capture({83,0,width},width) &&
                    ggml_backend_graph_compute(producer.get(),graph) == GGML_STATUS_SUCCESS && bank->publish_capture())) return;
            t.assert_equal(uint64_t(width-1),bank->snapshot_generation(0,false,1));
            t.assert_true(!bank->begin_restore({83,0,width},1) && !bank->snapshot_data(0,false,1,width));
            if (!gpu) t.assert_true(!probe.pending.empty());
            if (!t.assert_true(bank->stage_ready())) return;
            if (!gpu) t.assert_true(!probe.pending.empty());
            if (!t.assert_true(bank->complete_capture())) return;
            if (!gpu) t.assert_true(probe.pending.empty());
            t.assert_true(!bank->begin_staged_capture({83,0,width},width));
            for (size_t layer : {0u,2u,3u}) for (uint32_t snapshot=1;snapshot<=7;++snapshot)
                for (bool value : {false,true}) {
                    const size_t i=index(layer,snapshot,value);
                    if (!value || snapshot < width) generations[i]=width;
                    t.assert_equal(generations[i],bank->snapshot_generation(layer,value,snapshot));
                    const auto * actual=bank->snapshot_data(layer,value,snapshot,generations[i]);
                    if (generations[i]) t.assert_true(actual && !std::memcmp(actual,expected[i].data(),ggml_nbytes(inputs[i])));
                    else t.assert_true(!actual && bank->snapshot_pristine(layer,value,snapshot));
                }
            if (width > 1) t.assert_true(!bank->begin_restore({83,0,width-1},1));
            for (uint32_t selected=1;selected<=7;++selected) {
                if (!t.assert_true(bank->begin_restore({83,0,width},selected))) return;
                for (size_t layer : {0u,2u,3u}) for (bool value : {false,true})
                    if (!t.assert_true(bank->restore(producer.get(),layer,value,inputs[index(layer,1,value)],0))) return;
                if (!t.assert_true(bank->publish_restore() && bank->complete_restore())) return;
                for (size_t layer : {0u,2u,3u}) for (bool value : {false,true}) {
                    const size_t i=index(layer,selected,value);
                    std::vector<float> actual(expected[i].size());
                    ggml_backend_tensor_get(inputs[index(layer,1,value)],actual.data(),0,actual.size()*sizeof(float));
                    t.assert_true(actual == expected[i]);
                }
            }
            t.assert_equal(original_bytes,bank->host_bytes());
            t.assert_equal(original_stage,bank->staged_device_bytes());
        }
    });
    t.test("failed_staged_copy_cannot_republish_old_host_bytes", [](testing & t) {
        ggml_backend_ptr producer(ggml_backend_cpu_init());
        auto * device=ggml_backend_get_device(producer.get());
        llama_recurrent_spill_layout layout;
        if (!t.assert_true(llama_recurrent_spill_layout_make(
                {{GGML_TYPE_F32,GGML_TYPE_F32,4,8,true}},1,7,layout))) return;
        auto bank=llama_recurrent_spill_bank::create(layout,ggml_backend_cpu_buffer_type(),85,0);
        recurrent_copy_backend_probe probe(device);
        if (!t.assert_true(bank && bank->enable_publication(device,ggml_backend_cpu_buffer_type(),2) && probe.copy)) return;
        ggml_context_ptr ctx(ggml_init({32768,nullptr,true}));
        auto * source=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,4);
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),producer.get()));
        if (!t.assert_true(bool(buffer))) return;
        ggml_backend_tensor_memset(source,0x11,0,ggml_nbytes(source));
        auto * graph=ggml_new_graph_custom(ctx.get(),16,false);
        auto * stage=ggml_cpy(ctx.get(),source,bank->staging_tensor(0,false,7));
        ggml_build_forward_expand(graph,ggml_cpy(ctx.get(),stage,bank->publication_tensor(0,false,7)));
        if (!t.assert_true(bank->begin_staged_capture({85,0,1},8))) return;
        {
            recurrent_transfer_fault fault(probe.copy,false);
            t.assert_equal(GGML_STATUS_FAILED,ggml_backend_graph_compute(producer.get(),graph));
        }
        t.assert_true(!bank->publish_capture());
        bank->cancel();
        t.assert_true(!bank->snapshot_pristine(0,false,7) && !bank->begin_restore({85,0,1},7));
        t.assert_true(bank->snapshot_pristine(0,false,6));
        bank->reset(true);
        t.assert_true(bank->snapshot_pristine(0,false,7));
    });
    t.test("two_slot_stage_is_bounded_by_one_layer_not_all_recurrent_layers", [](testing & t) {
        std::vector<llama_recurrent_spill_layer> layers(
            48,{GGML_TYPE_F32,GGML_TYPE_F32,30720,786432,true});
        llama_recurrent_spill_layout snapshots;
        if (!t.assert_true(llama_recurrent_spill_layout_make(layers,1,3,snapshots))) return;
        t.assert_equal(size_t(149.625*1024*1024),snapshots.device_current_bytes);
        t.assert_equal(size_t(448.875*1024*1024),snapshots.host_snapshot_bytes);
        llama_recurrent_stage_layout stage;
        if (!t.assert_true(llama_recurrent_stage_layout_make(snapshots,2,256,stage))) return;
        t.assert_equal(size_t(9805824),stage.bytes_per_slot);
        t.assert_equal(size_t(19611648),stage.total_bytes);
        size_t offset=0,bytes=0;
        t.assert_true(llama_recurrent_stage_region(stage,0,false,1,0,offset,bytes));
        t.assert_equal(size_t(0),offset);
        t.assert_equal(size_t(122880),bytes);
        t.assert_true(llama_recurrent_stage_region(stage,47,true,3,1,offset,bytes));
        t.assert_equal(size_t(9805824+368640+2*3145728),offset);
        t.assert_equal(size_t(3145728),bytes);
        auto inconsistent=stage;
        inconsistent.bytes_per_slot=1;
        offset=123; bytes=456;
        t.assert_true(!llama_recurrent_stage_region(inconsistent,0,false,1,0,offset,bytes));
        t.assert_equal(size_t(123),offset);
        t.assert_equal(size_t(456),bytes);
        t.assert_true(!llama_recurrent_stage_region(stage,48,true,1,0,offset,bytes));
        t.assert_true(!llama_recurrent_stage_region(stage,0,true,4,0,offset,bytes));
        t.assert_true(!llama_recurrent_stage_region(stage,0,true,1,2,offset,bytes));
        const auto original=stage.total_bytes;
        t.assert_true(!llama_recurrent_stage_layout_make(snapshots,2,3,stage));
        t.assert_equal(original,stage.total_bytes);
        t.assert_true(!llama_recurrent_stage_layout_make(
            snapshots,3,size_t(1) << (sizeof(size_t)*8-1),stage));
        t.assert_equal(original,stage.total_bytes);
        auto malformed=snapshots;
        malformed.layers[0].r_row_bytes=0;
        t.assert_true(!llama_recurrent_stage_layout_make(malformed,2,256,stage));
        t.assert_equal(original,stage.total_bytes);
    });

    t.test("stage_metadata_matches_real_storage_without_allocating_payload", [](testing & t) {
        const std::vector<llama_recurrent_spill_layer> layers={
            {GGML_TYPE_F32,GGML_TYPE_F32,7,13,true},
            {GGML_TYPE_F32,GGML_TYPE_F32,0,0,false},
            {GGML_TYPE_F32,GGML_TYPE_F32,32,64,true}};
        auto * type=ggml_backend_cpu_buffer_type();
        for (uint32_t cells : {1u,2u}) for (uint32_t depth : {3u,7u}) {
            llama_recurrent_spill_layout layout;
            if (!t.assert_true(llama_recurrent_spill_layout_make(layers,cells,depth,layout))) return;
            auto real=llama_recurrent_stage_buffer::create(layout,2,type);
            auto metadata=llama_recurrent_stage_buffer::create_measurement(layout,2,type);
            if (!t.assert_true(real && metadata)) return;
            t.assert_equal(real->bytes(),metadata->bytes());
            t.assert_true(ggml_backend_buffer_get_size(real->buffer())>=real->bytes());
            t.assert_equal(size_t(0),ggml_backend_buffer_get_size(metadata->buffer()));
            for (size_t layer=0;layer<layers.size();++layer) for (uint32_t slot=0;slot<2;++slot)
            for (uint32_t snapshot=1;snapshot<=depth;++snapshot) for (bool value : {false,true}) {
                const auto * expected=real->tensor(layer,value,snapshot,slot);
                const auto * actual=metadata->tensor(layer,value,snapshot,slot);
                t.assert_equal(expected != nullptr,actual != nullptr);
                if (!expected || !actual) continue;
                t.assert_equal(expected->type,actual->type);
                t.assert_true(!std::memcmp(expected->ne,actual->ne,sizeof(expected->ne)));
                t.assert_true(!std::memcmp(expected->nb,actual->nb,sizeof(expected->nb)));
                t.assert_true(actual->buffer==metadata->buffer() && actual->data==nullptr);
                auto * plane=metadata->plane_span(layer,value,slot);
                t.assert_equal(real->plane_span(layer,value,slot) != nullptr,plane != nullptr);
                if (plane) t.assert_true(plane->buffer==metadata->buffer() && plane->data==nullptr);
            }
        }
    });
    t.test("publication_metadata_has_native_shapes_but_cannot_capture_or_restore", [](testing & t) {
        ggml_backend_ptr producer(ggml_backend_cpu_init());
        auto * device=ggml_backend_get_device(producer.get());
        auto * type=ggml_backend_get_default_buffer_type(producer.get());
        const std::vector<llama_recurrent_spill_layer> layers={
            {GGML_TYPE_F32,GGML_TYPE_F32,7,13,true},
            {GGML_TYPE_F32,GGML_TYPE_F32,32,64,true}};
        for (uint32_t cells : {1u,2u}) for (uint32_t depth : {3u,7u}) {
            llama_recurrent_spill_layout layout;
            if (!t.assert_true(llama_recurrent_spill_layout_make(layers,cells,depth,layout))) return;
            auto real=llama_recurrent_spill_bank::create(layout,type,981,0);
            if (!t.assert_true(real && real->enable_publication(device,type,2))) return;
            auto metadata=llama_recurrent_spill_bank::create_measurement(layout,device,type,2,982,0);
            if (!t.assert_true(metadata != nullptr)) return;
            t.assert_equal(real->host_bytes(),metadata->host_bytes());
            t.assert_true(real->allocated_host_bytes()>=real->host_bytes());
            t.assert_equal(size_t(0),metadata->allocated_host_bytes());
            t.assert_equal(real->staged_device_bytes(),metadata->staged_device_bytes());
            for (size_t layer=0;layer<layers.size();++layer) for (uint32_t snapshot=1;snapshot<=depth;++snapshot)
            for (bool value : {false,true}) {
                auto * expected=real->publication_tensor(layer,value,snapshot);
                auto * actual=metadata->publication_tensor(layer,value,snapshot);
                if (!t.assert_true(expected && actual)) return;
                t.assert_true(!std::memcmp(expected->ne,actual->ne,sizeof(expected->ne)));
                t.assert_true(!std::memcmp(expected->nb,actual->nb,sizeof(expected->nb)));
                auto * stage=metadata->staging_tensor(layer,value,snapshot);
                if (!t.assert_true(stage != nullptr)) return;
                t.assert_equal(size_t(0),ggml_backend_buffer_get_size(stage->buffer));
                t.assert_true(stage->data==nullptr);
            }
            t.assert_true(!metadata->begin_capture({982,0,1}));
            t.assert_true(!metadata->begin_staged_capture({982,0,1},depth+1));
            t.assert_true(!metadata->begin_restore({982,0,1},1));
            t.assert_true(!metadata->snapshot_pristine(0,false,1));
            metadata->reset(true);
            t.assert_true(!metadata->snapshot_pristine(0,false,1));
            metadata->cancel();
            t.assert_true(!metadata->stage_ready());
            ggml_context_ptr ctx(ggml_init({4096+ggml_graph_overhead_custom(16,false)+8*ggml_tensor_overhead(),nullptr,true}));
            auto * destination=metadata->publication_tensor(0,false,1);
            auto * source=ggml_dup_tensor(ctx.get(),destination);
            auto * copy=ggml_cpy(ctx.get(),source,destination);
            auto * graph=ggml_new_graph_custom(ctx.get(),16,false);
            ggml_build_forward_expand(graph,copy);
            ggml_backend_buffer_ptr backing(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(),type));
            if (!t.assert_true(backing != nullptr)) return;
            const std::vector<float> original(size_t(ggml_nelements(source)),3.25f);
            ggml_backend_tensor_set(source,original.data(),0,original.size()*sizeof(float));
            t.assert_equal(GGML_STATUS_FAILED,ggml_backend_graph_compute(producer.get(),graph));
            std::vector<float> after(original.size());
            ggml_backend_tensor_get(source,after.data(),0,after.size()*sizeof(float));
            t.assert_true(original==after);
            t.assert_equal(size_t(0),metadata->allocated_host_bytes());
        }
    });
    t.test("typed_stage_tensors_alias_only_their_planned_ring_regions", [](testing & t) {
        llama_recurrent_spill_layout snapshots;
        if (!t.assert_true(llama_recurrent_spill_layout_make(
                {{GGML_TYPE_F32,GGML_TYPE_F32,16,32,true}},1,3,snapshots))) return;
        auto stage=llama_recurrent_stage_buffer::create(snapshots,2,ggml_backend_cpu_buffer_type());
        if (!t.assert_true(bool(stage))) return;
        t.assert_equal(stage->layout().total_bytes,stage->bytes());
        auto * base=static_cast<uint8_t *>(ggml_backend_buffer_get_base(stage->buffer()));
        size_t offset=0,bytes=0;
        if (!t.assert_true(llama_recurrent_stage_region(
                stage->layout(),0,false,2,1,offset,bytes))) return;
        auto * r=stage->tensor(0,false,2,1);
        t.assert_true(r && r->buffer == stage->buffer() && r->data == base+offset);
        t.assert_equal(bytes,ggml_nbytes(r));
        if (!t.assert_true(llama_recurrent_stage_region(
                stage->layout(),0,true,3,1,offset,bytes))) return;
        auto * s=stage->tensor(0,true,3,1);
        t.assert_true(s && s->buffer == stage->buffer() && s->data == base+offset);
        t.assert_equal(bytes,ggml_nbytes(s));
        t.assert_true(stage->tensor(1,false,1,0) == nullptr);
        t.assert_true(stage->tensor(0,true,4,0) == nullptr);
        t.assert_true(stage->tensor(0,true,1,2) == nullptr);
    });

    t.test("invalid_geometry_and_overflow_are_transactional", [](testing & t) {
        llama_recurrent_spill_layout layout;
        const std::vector<llama_recurrent_spill_layer> good{{GGML_TYPE_F32,GGML_TYPE_F32,4,8,true}};
        if (!t.assert_true(llama_recurrent_spill_layout_make(good,1,2,layout))) return;
        const auto bytes=layout.host_snapshot_bytes;
        t.assert_true(!llama_recurrent_spill_layout_make(good,UINT32_MAX,UINT32_MAX,layout));
        t.assert_equal(bytes,layout.host_snapshot_bytes);
        const std::vector<llama_recurrent_spill_layer> bad{{GGML_TYPE_Q8_0,GGML_TYPE_F32,1,8,true}};
        t.assert_true(!llama_recurrent_spill_layout_make(bad,1,2,layout));
        t.assert_equal(bytes,layout.host_snapshot_bytes);
    });
    t.test("host_bank_publishes_and_restores_exact_selected_state", [](testing & t) {
        llama_recurrent_spill_layout layout;
        if (!t.assert_true(llama_recurrent_spill_layout_make(
                {{GGML_TYPE_F32,GGML_TYPE_F32,4,8,true}},1,2,layout))) return;
        ggml_backend_ptr backend(ggml_backend_cpu_init());
        auto bank=llama_recurrent_spill_bank::create(layout,ggml_backend_cpu_buffer_type(),7,0);
        if (!t.assert_true(bool(bank))) return;
        t.assert_equal(layout.host_snapshot_bytes,bank->host_bytes());
        ggml_context_ptr ctx(ggml_init({32768,nullptr,true}));
        auto * r=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,4);
        auto * s=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,8);
        auto * dr=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,4);
        auto * ds=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,8);
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()));
        if (!t.assert_true(bool(buffer))) return;
        const float expected_r[4]={1,2,3,4};
        t.assert_true(bank->snapshot_pristine(0,false,1));
        t.assert_true(bank->snapshot_pristine(0,true,2));
        const float expected_s[8]={9,8,7,6,5,4,3,2};
        ggml_backend_tensor_set(r,expected_r,0,sizeof(expected_r));
        ggml_backend_tensor_set(s,expected_s,0,sizeof(expected_s));
        if (!t.assert_true(bank->begin_capture({7,0,1}) &&
                bank->capture(backend.get(),0,false,1,r,0) &&
                bank->capture(backend.get(),0,true,1,s,0) && bank->publish_capture())) return;
        t.assert_true(!bank->begin_restore({7,0,1},1));
        if (!t.assert_true(bank->complete_capture() && bank->begin_restore({7,0,1},1))) return;
        t.assert_true(bank->restore(backend.get(),0,false,dr,0));
        t.assert_true(bank->restore(backend.get(),0,true,ds,0));
        if (!t.assert_true(bank->publish_restore() && bank->complete_restore())) return;
        float actual_r[4]={},actual_s[8]={};
        ggml_backend_tensor_get(dr,actual_r,0,sizeof(actual_r));
        ggml_backend_tensor_get(ds,actual_s,0,sizeof(actual_s));
        t.assert_true(std::memcmp(actual_r,expected_r,sizeof(expected_r)) == 0);
        t.assert_true(std::memcmp(actual_s,expected_s,sizeof(expected_s)) == 0);
        t.assert_equal(uint64_t(1),bank->snapshot_generation(0,false,1));
        t.assert_true(!bank->snapshot_pristine(0,false,1));
        t.assert_true(!bank->begin_capture({7,1,2}));
        t.assert_true(bank->begin_capture({7,0,2}));
        t.assert_true(bank->capture(backend.get(),0,false,1,r,0));
        bank->cancel();
        t.assert_equal(uint64_t(0),bank->snapshot_generation(0,false,1));
        t.assert_true(!bank->snapshot_pristine(0,false,1));
        bank->reset(false);
        t.assert_true(!bank->snapshot_pristine(0,false,1));
        bank->reset(true);
        t.assert_true(bank->snapshot_pristine(0,false,1));
        t.assert_equal(uint64_t(0),bank->snapshot_generation(0,false,1));
        t.assert_true(bank->snapshot_pristine(0,false,2));
        if (!t.assert_true(bank->begin_restore({7,0,1},1) &&
                bank->restore(backend.get(),0,false,dr,0) &&
                bank->restore(backend.get(),0,true,ds,0) &&
                bank->publish_restore() && bank->complete_restore())) return;
        ggml_backend_tensor_get(dr,actual_r,0,sizeof(actual_r));
        ggml_backend_tensor_get(ds,actual_s,0,sizeof(actual_s));
        const float zero_r[4]={},zero_s[8]={};
        t.assert_true(std::memcmp(actual_r,zero_r,sizeof(actual_r)) == 0);
        t.assert_true(std::memcmp(actual_s,zero_s,sizeof(actual_s)) == 0);

    });

    t.test("mixed_generation_restore_keeps_unchanged_recurrent_plane", [](testing & t) {
        llama_recurrent_spill_layout layout;
        if (!t.assert_true(llama_recurrent_spill_layout_make(
                {{GGML_TYPE_F32,GGML_TYPE_F32,4,4,true}},1,2,layout))) return;
        ggml_backend_ptr backend(ggml_backend_cpu_init());
        auto bank=llama_recurrent_spill_bank::create(layout,ggml_backend_cpu_buffer_type(),19,0);
        if (!t.assert_true(bool(bank))) return;
        ggml_context_ptr ctx(ggml_init({32768,nullptr,true}));
        auto * r=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,4);
        auto * s=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,4);
        auto * dr=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,4);
        auto * ds=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,4);
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()));
        if (!t.assert_true(bool(buffer))) return;
        const float first_r[4]={1,2,3,4}, first_s[4]={5,6,7,8};
        const float second_r[4]={9,10,11,12};
        ggml_backend_tensor_set(r,first_r,0,sizeof(first_r));
        ggml_backend_tensor_set(s,first_s,0,sizeof(first_s));
        if (!t.assert_true(bank->begin_capture({19,0,1}) &&
                bank->capture(backend.get(),0,false,1,r,0) &&
                bank->capture(backend.get(),0,true,1,s,0) &&
                bank->publish_capture() && bank->complete_capture())) return;
        ggml_backend_tensor_set(r,second_r,0,sizeof(second_r));
        if (!t.assert_true(bank->begin_capture({19,0,2}) &&
                bank->capture(backend.get(),0,false,1,r,0) &&
                bank->publish_capture() && bank->complete_capture())) return;
        if (!t.assert_true(bank->begin_restore({19,0,2},1))) return;
        if (!t.assert_true(bank->restore(backend.get(),0,false,dr,0) &&
                bank->restore(backend.get(),0,true,ds,0) &&
                bank->publish_restore() && bank->complete_restore())) return;
        float actual_r[4]={},actual_s[4]={};
        ggml_backend_tensor_get(dr,actual_r,0,sizeof(actual_r));
        ggml_backend_tensor_get(ds,actual_s,0,sizeof(actual_s));
        t.assert_true(std::memcmp(actual_r,second_r,sizeof(actual_r)) == 0);
        t.assert_true(std::memcmp(actual_s,first_s,sizeof(actual_s)) == 0);
    });


    t.test("cpu_graph_publishes_recurrent_snapshots_through_two_slot_stage", [](testing & t) {
        ggml_backend_ptr backend(ggml_backend_cpu_init());
        llama_recurrent_spill_layout layout;
        if (!t.assert_true(llama_recurrent_spill_layout_make(
                {{GGML_TYPE_F32,GGML_TYPE_F32,4,8,true}},1,1,layout))) return;
        auto bank=llama_recurrent_spill_bank::create(layout,ggml_backend_cpu_buffer_type(),31,0);
        if (!t.assert_true(bool(bank))) return;
        if (!t.assert_true(bank->enable_publication(ggml_backend_get_device(backend.get()),
                    ggml_backend_cpu_buffer_type(),2))) return;
        auto * dst_r=bank->publication_tensor(0,false,1);
        auto * dst_s=bank->publication_tensor(0,true,1);
        auto * stage_r=bank->staging_tensor(0,false,1);
        auto * stage_s=bank->staging_tensor(0,true,1);
        if (!t.assert_true(dst_r && dst_s && stage_r && stage_s &&
                bank->staged_publication_enabled())) return;
        ggml_context_ptr ctx(ggml_init({32768,nullptr,true}));
        auto * src_r=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,4);
        auto * src_s=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,8);
        ggml_backend_buffer_ptr source(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()));
        if (!t.assert_true(bool(source))) return;
        const float expected_r[4]={1,2,3,4};
        const float expected_s[8]={8,7,6,5,4,3,2,1};
        ggml_backend_tensor_set(src_r,expected_r,0,sizeof(expected_r));
        ggml_backend_tensor_set(src_s,expected_s,0,sizeof(expected_s));
        auto * graph=ggml_new_graph_custom(ctx.get(),16,false);
        auto * copied_r=ggml_cpy(ctx.get(),src_r,stage_r);
        auto * copied_s=ggml_cpy(ctx.get(),src_s,stage_s);
        ggml_build_forward_expand(graph,ggml_cpy(ctx.get(),copied_r,dst_r));
        ggml_build_forward_expand(graph,ggml_cpy(ctx.get(),copied_s,dst_s));
        if (!t.assert_true(bank->begin_staged_capture({31,0,1},2))) return;
        if (!t.assert_true(ggml_backend_graph_compute(backend.get(),graph) == GGML_STATUS_SUCCESS &&
                bank->publish_capture() && bank->stage_ready() && bank->complete_capture())) return;
        const auto * actual_r=bank->snapshot_data(0,false,1,1);
        const auto * actual_s=bank->snapshot_data(0,true,1,1);
        t.assert_true(actual_r && !std::memcmp(actual_r,expected_r,sizeof(expected_r)));
        t.assert_true(actual_s && !std::memcmp(actual_s,expected_s,sizeof(expected_s)));
        auto * partial=ggml_new_graph_custom(ctx.get(),16,false);
        ggml_build_forward_expand(partial,ggml_cpy(ctx.get(),copied_r,dst_r));
        if (!t.assert_true(bank->begin_staged_capture({31,0,2},2) &&
                ggml_backend_graph_compute(backend.get(),partial) == GGML_STATUS_SUCCESS)) return;
        t.assert_true(!bank->publish_capture());
        bank->cancel();
        t.assert_equal(uint64_t(0),bank->snapshot_generation(0,false,1));
        t.assert_true(!bank->snapshot_pristine(0,false,1));
        if (!t.assert_true(bank->begin_staged_capture({31,0,2},2) &&
                ggml_backend_graph_compute(backend.get(),graph) == GGML_STATUS_SUCCESS &&
                bank->publish_capture() && bank->stage_ready() && bank->complete_capture())) return;
        t.assert_true(bank->snapshot_data(0,false,1,2) != nullptr);
        t.assert_true(bank->snapshot_data(0,true,1,2) != nullptr);
    });

    t.test("three_layers_reuse_two_slots_only_after_d2h_consumption", [&](testing & t) {
        ggml_backend_load_all();
        const bool gpu=argc > 1 && !std::strcmp(argv[1],"--cuda");
        auto * device=ggml_backend_dev_by_name(gpu ? "CUDA0" : "CPU");
        if (!t.assert_true(device != nullptr)) return;
        ggml_backend_ptr producer(ggml_backend_dev_init(device,nullptr));
        auto * host_type=ggml_backend_dev_host_buffer_type(device);
        if (!host_type) host_type=ggml_backend_cpu_buffer_type();
        llama_recurrent_spill_layout layout;
        std::vector<llama_recurrent_spill_layer> layers(
            3,{GGML_TYPE_F32,GGML_TYPE_F32,4,8,true});
        if (!t.assert_true(llama_recurrent_spill_layout_make(layers,1,1,layout))) return;
        auto bank=llama_recurrent_spill_bank::create(layout,host_type,33,0);
        if (!t.assert_true(bool(bank) &&
                bank->enable_publication(device,ggml_backend_dev_buffer_type(device),2))) return;
        ggml_context_ptr ctx(ggml_init({65536,nullptr,true}));
        std::array<ggml_tensor *,3> r{},s{};
        for (size_t layer=0;layer<3;++layer) {
            r[layer]=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,4,1);
            s[layer]=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,8,1);
        }
        ggml_backend_buffer_ptr source(ggml_backend_alloc_ctx_tensors(ctx.get(),producer.get()));
        if (!t.assert_true(bool(source))) return;
        std::array<std::array<float,4>,3> expected_r{};
        std::array<std::array<float,8>,3> expected_s{};
        auto * graph=ggml_new_graph_custom(ctx.get(),64,false);
        for (size_t layer=0;layer<3;++layer) {
            for (size_t i=0;i<4;++i) expected_r[layer][i]=float(100*layer+i);
            for (size_t i=0;i<8;++i) expected_s[layer][i]=float(-100*layer-int(i));
            ggml_backend_tensor_set(r[layer],expected_r[layer].data(),0,ggml_nbytes(r[layer]));
            ggml_backend_tensor_set(s[layer],expected_s[layer].data(),0,ggml_nbytes(s[layer]));
            auto * staged_r=ggml_cpy(ctx.get(),r[layer],bank->staging_tensor(layer,false,1));
            auto * staged_s=ggml_cpy(ctx.get(),s[layer],bank->staging_tensor(layer,true,1));
            ggml_build_forward_expand(graph,ggml_cpy(ctx.get(),staged_r,
                bank->publication_tensor(layer,false,1)));
            ggml_build_forward_expand(graph,ggml_cpy(ctx.get(),staged_s,
                bank->publication_tensor(layer,true,1)));
        }
        if (!t.assert_true(bank->begin_staged_capture({33,0,1},2) &&
                ggml_backend_graph_compute(producer.get(),graph) == GGML_STATUS_SUCCESS &&
                bank->publish_capture() && bank->stage_ready() && bank->complete_capture())) return;
        for (size_t layer=0;layer<3;++layer) {
            const auto * actual_r=bank->snapshot_data(layer,false,1,1);
            const auto * actual_s=bank->snapshot_data(layer,true,1,1);
            t.assert_true(actual_r && !std::memcmp(actual_r,expected_r[layer].data(),ggml_nbytes(r[layer])));
            t.assert_true(actual_s && !std::memcmp(actual_s,expected_s[layer].data(),ggml_nbytes(s[layer])));
        }
    });

    if (argc > 1 && !std::strcmp(argv[1],"--cuda")) t.test(
        "cuda_graph_publishes_staged_snapshots_without_host_graph_sync", [](testing & t) {
        ggml_backend_load_all();
        auto * device=ggml_backend_dev_by_name("CUDA0");
        if (!t.assert_true(device != nullptr)) return;
        ggml_backend_ptr producer(ggml_backend_dev_init(device,nullptr));
        llama_recurrent_spill_layout layout;
        if (!t.assert_true(llama_recurrent_spill_layout_make(
                {{GGML_TYPE_F32,GGML_TYPE_F32,1024,2048,true}},1,1,layout))) return;
        auto bank=llama_recurrent_spill_bank::create(
            layout,ggml_backend_dev_host_buffer_type(device),32,0);
        if (!t.assert_true(bool(bank) &&
                bank->enable_publication(device,ggml_backend_dev_buffer_type(device),2))) return;
        ggml_context_ptr ctx(ggml_init({32768,nullptr,true}));
        auto * src_r=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,1024,1);
        auto * src_s=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,2048,1);
        ggml_backend_buffer_ptr source(ggml_backend_alloc_ctx_tensors(ctx.get(),producer.get()));
        if (!t.assert_true(bool(source))) return;
        std::vector<float> expected_r(1024),expected_s(2048);
        for (size_t i=0;i<expected_r.size();++i) expected_r[i]=float(i)*.125f;
        for (size_t i=0;i<expected_s.size();++i) expected_s[i]=float(i)*-.0625f;
        ggml_backend_tensor_set(src_r,expected_r.data(),0,ggml_nbytes(src_r));
        ggml_backend_tensor_set(src_s,expected_s.data(),0,ggml_nbytes(src_s));
        auto * write_r=ggml_cpy(ctx.get(),src_r,bank->staging_tensor(0,false,1));
        auto * write_s=ggml_cpy(ctx.get(),src_s,bank->staging_tensor(0,true,1));
        auto * graph=ggml_new_graph_custom(ctx.get(),16,false);
        ggml_build_forward_expand(graph,ggml_cpy(ctx.get(),write_r,bank->publication_tensor(0,false,1)));
        ggml_build_forward_expand(graph,ggml_cpy(ctx.get(),write_s,bank->publication_tensor(0,true,1)));
        if (!t.assert_true(bank->begin_staged_capture({32,0,1},2))) return;
        if (!t.assert_true(ggml_backend_graph_compute(producer.get(),graph) == GGML_STATUS_SUCCESS &&
                bank->publish_capture() && bank->stage_ready())) return;
        t.assert_equal(uint64_t(0),bank->snapshot_generation(0,false,1));
        if (!t.assert_true(bank->complete_capture())) return;
        const auto * actual_r=bank->snapshot_data(0,false,1,1);
        const auto * actual_s=bank->snapshot_data(0,true,1,1);
        t.assert_true(actual_r && !std::memcmp(actual_r,expected_r.data(),ggml_nbytes(src_r)));
        t.assert_true(actual_s && !std::memcmp(actual_s,expected_s.data(),ggml_nbytes(src_s)));
    });

    if (argc > 1 && !std::strcmp(argv[1],"--cuda")) t.test("cuda_pinned_host_capture_and_restore_keep_publication_order", [](testing & t) {
        ggml_backend_load_all();
        auto * dev=ggml_backend_dev_by_name("CUDA0");
        if (!t.assert_true(dev != nullptr)) return;
        auto * host_type=ggml_backend_dev_host_buffer_type(dev);
        if (!t.assert_true(host_type && ggml_backend_buft_is_host(host_type))) return;
        ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
        llama_recurrent_spill_layout layout;
        if (!t.assert_true(llama_recurrent_spill_layout_make(
                {{GGML_TYPE_F32,GGML_TYPE_F32,1024,2048,true}},1,2,layout))) return;
        auto bank=llama_recurrent_spill_bank::create(layout,host_type,8,0);
        if (!t.assert_true(bool(bank))) return;
        auto stage=llama_recurrent_stage_buffer::create(layout,2,ggml_backend_dev_buffer_type(dev));
        if (!t.assert_true(bool(stage))) return;
        t.assert_true(!ggml_backend_buffer_is_host(stage->buffer()));
        t.assert_equal(stage->layout().total_bytes,stage->bytes());
        ggml_context_ptr ctx(ggml_init({32768,nullptr,true}));
        auto * r=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,1024);
        auto * s=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,2048);
        auto * dr=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,1024);
        auto * ds=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,2048);
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()));
        if (!t.assert_true(bool(buffer))) return;
        std::vector<float> expected_r(1024),expected_s(2048),actual_r(1024),actual_s(2048);
        for (size_t i=0;i<expected_r.size();++i) expected_r[i]=float(i)*.125f;
        for (size_t i=0;i<expected_s.size();++i) expected_s[i]=float(i)*-.0625f;
        ggml_backend_tensor_set(r,expected_r.data(),0,ggml_nbytes(r));
        ggml_backend_tensor_set(s,expected_s.data(),0,ggml_nbytes(s));
        if (!t.assert_true(bank->begin_capture({8,0,1}) &&
                bank->capture(backend.get(),0,false,2,r,0) &&
                bank->capture(backend.get(),0,true,2,s,0) && bank->publish_capture())) return;
        t.assert_equal(uint64_t(0),bank->snapshot_generation(0,false,2));
        if (!t.assert_true(bank->complete_capture() && bank->begin_restore({8,0,1},2))) return;
        if (!t.assert_true(bank->restore(backend.get(),0,false,dr,0) &&
                bank->restore(backend.get(),0,true,ds,0) &&
                bank->publish_restore() && bank->complete_restore())) return;
        ggml_backend_tensor_get(dr,actual_r.data(),0,ggml_nbytes(dr));
        ggml_backend_tensor_get(ds,actual_s.data(),0,ggml_nbytes(ds));
        t.assert_true(std::memcmp(expected_r.data(),actual_r.data(),ggml_nbytes(r)) == 0);
        t.assert_true(std::memcmp(expected_s.data(),actual_s.data(),ggml_nbytes(s)) == 0);
    });

    return t.summary();
}
