#include "../src/llama-memory-recurrent.h"
#include "../src/llama-memory-recurrent-compact.h"
#include "../src/llama-model.h"
#include "../src/llama-io.h"
#include "../ggml/src/ggml-backend-memory.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "testing.h"

#include <cstring>
#include <random>

struct byte_writer : llama_io_write_i {
    std::vector<uint8_t> bytes;
    void write(const void * source,size_t size) override {
        const auto offset=bytes.size();bytes.resize(offset+size);std::memcpy(bytes.data()+offset,source,size);
    }
    void write_tensor(ggml_tensor * tensor,size_t offset,size_t size) override {
        const auto start=bytes.size();bytes.resize(start+size);ggml_backend_tensor_get(tensor,bytes.data()+start,offset,size);
    }
    size_t n_bytes() override {return bytes.size();}
};
struct byte_reader : llama_io_read_i {
    const std::vector<uint8_t> & bytes;size_t cursor=0;
    explicit byte_reader(const std::vector<uint8_t> & bytes) : bytes(bytes) {}
    void read(void * target,size_t size) override {
        if (size>bytes.size()-cursor) throw std::runtime_error("short state");
        std::memcpy(target,bytes.data()+cursor,size);cursor+=size;
    }
    void read_tensor(ggml_tensor * tensor,size_t offset,size_t size) override {
        if (size>bytes.size()-cursor) throw std::runtime_error("short tensor");
        ggml_backend_tensor_set(tensor,bytes.data()+cursor,offset,size);cursor+=size;
    }
    size_t n_bytes() override {return cursor;}
};

struct fixture {
    std::unique_ptr<llama_model> model{llama_model_create(LLM_ARCH_QWEN35,llama_model_default_params())};
    ggml_backend_ptr backend{ggml_backend_cpu_init()};
    std::unique_ptr<ggml_backend_memory_arena,decltype(&ggml_backend_memory_arena_free)> arena{nullptr,ggml_backend_memory_arena_free};
    std::unique_ptr<ggml_backend_memory_lease,decltype(&ggml_backend_memory_lease_free)> lease{nullptr,ggml_backend_memory_lease_free};
    std::unique_ptr<llama_memory_recurrent> memory;
    std::vector<llama_seq_id> ids;
    uint32_t depth;
    fixture(std::vector<llama_seq_id> ids,uint32_t depth) : ids(std::move(ids)),depth(depth) {
        model->hparams.n_layer_all=1;model->hparams.ssm_d_conv=4;model->hparams.ssm_d_inner=384;
        model->hparams.ssm_d_state=64;model->hparams.ssm_dt_rank=6;model->hparams.ssm_n_group=2;
        memory=std::make_unique<llama_memory_recurrent>(*model,GGML_TYPE_F32,GGML_TYPE_F32,false,
                uint32_t(this->ids.size()),256,depth,nullptr,false,true,true,true);
        GGML_ASSERT(memory->compact_enabled() && memory->configure_selected_sequences(this->ids));
        ggml_backend_buffer_type_t type=nullptr;size_t bytes=0;
        GGML_ASSERT(memory->current_storage_requirements(type,bytes));
        arena.reset(ggml_backend_memory_arena_new(type,bytes));
        GGML_ASSERT(arena && ggml_backend_memory_arena_begin(arena.get(),0) &&
                ggml_backend_memory_arena_reserve_at(arena.get(),91,0,bytes,ggml_backend_buft_get_alignment(type),
                    GGML_BACKEND_MEMORY_REGION_PERSISTENT,nullptr) && ggml_backend_memory_arena_commit(arena.get()));
        lease.reset(ggml_backend_memory_arena_acquire(arena.get(),91));
        GGML_ASSERT(lease && memory->bind_current_storage(lease.get()));
    }

    void step(testing & t,uint32_t rows,uint32_t rejection,bool fail_restore=false,bool cancel=false) {
        const uint32_t count=uint32_t(ids.size()),retained=std::min(rows,depth+1);
        const int w=64,heads=6,keys=2,history=3,channels=640;
        const size_t matrix=w*w*heads;
        llama_batch_allocr allocator(1);auto batch=allocator.ubatch_reserve(rows,count);
        std::fill(batch.data->seq_idx.begin(),batch.data->seq_idx.end(),-1);
        for (uint32_t s=0;s<count;++s) {
            batch.seq_id_unq[s]=ids[s];batch.seq_idx[ids[s]]=int32_t(s);
            const auto begin=std::max<llama_pos>(10,memory->seq_pos_max(ids[s])+1);
            for (uint32_t i=0;i<rows;++i) {
                const auto index=s*rows+i;batch.n_seq_id[index]=1;batch.seq_id[index]=&ids[s];batch.pos[index]=begin+i;
            }
        }
        if (!t.assert_true(memory->prepare({batch}) && memory->find_slot(batch))) return;
        const auto generation=memory->begin_compact_capture(backend.get(),batch,rows>depth+1);
        if (!t.assert_true(generation!=0)) return;
        t.assert_true(!memory->release_current_storage());
        t.assert_true(!memory->park_current_storage());
        bool refused=false;
        try {const std::vector<uint8_t> empty;byte_reader input(empty);memory->state_read(input,ids[0],0);}
        catch (const std::runtime_error &) {refused=true;}
        t.assert_true(refused);
        ggml_context_ptr ctx(ggml_init({512*ggml_tensor_overhead()+ggml_graph_overhead(),nullptr,true}));
        auto * graph=ggml_new_graph(ctx.get());
        auto * sources=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_I32,count);
        auto * destinations=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_I32,count);
        auto * slots=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_I32,count);
        auto make=[&](int a,int b,int c,int d) {return ggml_new_tensor_4d(ctx.get(),GGML_TYPE_F32,a,b,c,d);};
        auto * q=make(w,keys,rows,count),* k=make(w,keys,rows,count),* v=make(w,heads,rows,count);
        auto * g=make(1,heads,rows,count),* beta=make(1,heads,rows,count);
        auto * mixed=make(rows,channels,count,1);
        auto * state=ggml_reshape_4d(ctx.get(),ggml_get_rows(ctx.get(),memory->s_l[0],sources),w,w,heads,count);
        auto * previous=ggml_reshape_3d(ctx.get(),ggml_get_rows(ctx.get(),memory->r_l[0],sources),history,channels,count);
        auto * conv=ggml_concat(ctx.get(),previous,mixed,0);
        auto * native=ggml_gated_delta_net(ctx.get(),q,k,v,g,beta,state,rows);
        ggml_build_forward_expand(graph,native);
        llama_recurrent_compact_graph_storage storage{memory->r_l[0],memory->s_l[0],memory->compact_log(0),
                memory->compact_timeline(0),destinations,slots};
        llama_recurrent_compact_capture_history(ctx.get(),graph,conv,history,depth,storage);
        auto captured=llama_recurrent_compact_capture_matrix(ctx.get(),graph,q,k,v,g,beta,state,depth,storage);
        ggml_build_forward_expand(graph,captured.first);
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()));
        std::vector<int32_t> input(count),output(count),positions(count);
        llama_memory_recurrent_context inputs(memory.get());
        for (uint32_t s=0;s<count;++s) {
            input[s]=inputs.s_copy(int(s));output[s]=memory->compact_working_row(ids[s]);positions[s]=memory->compact_slot(ids[s]);
            t.assert_equal(memory->compact_current_row(ids[s]),input[s]);
        }
        ggml_backend_tensor_set(sources,input.data(),0,ggml_nbytes(sources));
        ggml_backend_tensor_set(destinations,output.data(),0,ggml_nbytes(destinations));
        ggml_backend_tensor_set(slots,positions.data(),0,ggml_nbytes(slots));
        std::mt19937 random(41+rows);std::uniform_real_distribution<float> uniform(-0.2f,0.2f);
        for (auto * tensor : {q,k,v,g,beta,mixed}) {
            std::vector<float> values(ggml_nelements(tensor));
            for (auto & value : values) value=tensor==g ? -1+uniform(random) : uniform(random);
            ggml_backend_tensor_set(tensor,values.data(),0,ggml_nbytes(tensor));
        }
        if (!t.assert_true(ggml_backend_graph_compute(backend.get(),graph)==GGML_STATUS_SUCCESS)) return;
        if (cancel) {
            std::vector<float> checkpoint(matrix*count),observed(matrix);
            ggml_backend_tensor_get(state,checkpoint.data(),0,ggml_nbytes(state));
            memory->cancel_compact_capture();
            for (uint32_t s=0;s<count;++s) {
                ggml_backend_tensor_get(memory->s_l[0],observed.data(),size_t(input[s])*matrix*sizeof(float),matrix*sizeof(float));
                t.assert_equal(0,std::memcmp(checkpoint.data()+s*matrix,observed.data(),matrix*sizeof(float)));
            }
            t.assert_true(!memory->finish_compact_capture(batch,{},generation));
            bool refused=false;
            try {byte_writer output;memory->state_write(output,ids[0],0);} catch (const std::runtime_error &) {refused=true;}
            t.assert_true(refused);
            memory->clear(true);
            return;
        }
        std::vector<std::pair<ggml_tensor *,uint32_t>> checkpoints;
        if (captured.second) checkpoints.emplace_back(captured.second,0);
        t.assert_true(!memory->finish_compact_capture(batch,checkpoints,generation+1));
        if (!t.assert_true(memory->finish_compact_capture(batch,checkpoints,generation))) return;
        std::vector<float> dense(ggml_nelements(native)),timeline(ggml_nelements(conv));
        ggml_backend_tensor_get(native,dense.data(),0,ggml_nbytes(native));
        ggml_backend_tensor_get(conv,timeline.data(),0,ggml_nbytes(conv));
        for (uint32_t s=0;s<count;++s) {
            const auto rejected=(rejection+s)%retained;
            if (rejected && !t.assert_true(memory->seq_rm(ids[s],memory->seq_pos_max(ids[s])+1-rejected,-1))) return;
        }
        if (fail_restore && retained>1) {
            auto original=backend->iface.graph_compute;
            static decltype(original) compute;
            compute=original;
            backend->iface.graph_compute=[](ggml_backend_t b,ggml_cgraph * graph) {
                compute(b,graph);return GGML_STATUS_FAILED;
            };
            t.assert_true(!memory->materialize_compact());
            backend->iface.graph_compute=original;
        }
        if (!t.assert_true(memory->materialize_compact())) return;
        const size_t attention=size_t(w)*heads*rows*count;
        for (uint32_t s=0;s<count;++s) {
            const auto rejected=(rejection+s)%retained;
            const auto row=memory->compact_current_row(ids[s]);
            std::vector<float> observed(matrix),conv_state(history*channels);
            ggml_backend_tensor_get(memory->s_l[0],observed.data(),size_t(row)*matrix*sizeof(float),matrix*sizeof(float));
            t.assert_equal(0,std::memcmp(observed.data(),dense.data()+attention+(rejected*count+s)*matrix,matrix*sizeof(float)));
            ggml_backend_tensor_get(memory->r_l[0],conv_state.data(),size_t(row)*conv_state.size()*sizeof(float),conv_state.size()*sizeof(float));
            for (int c=0;c<channels;++c) t.assert_equal(0,std::memcmp(conv_state.data()+c*history,
                        timeline.data()+(s*channels+c)*(rows+history)+rows-rejected,history*sizeof(float)));
            byte_writer saved;memory->state_write(saved,ids[s],0);
            llama_memory_recurrent restored(*model,GGML_TYPE_F32,GGML_TYPE_F32,false,1,1,0,nullptr);
            byte_reader reader(saved.bytes);restored.state_read(reader,0,0);
            std::vector<float> roundtrip(matrix);
            ggml_backend_tensor_get(restored.s_l[0],roundtrip.data(),0,matrix*sizeof(float));
            t.assert_equal(0,std::memcmp(observed.data(),roundtrip.data(),matrix*sizeof(float)));
        }
    }
};

int main() {
    testing t;
    t.test("native_exact_prefixes_sparse_ids_workspace_checkpoint_and_retry", [](testing & t) {
        const std::vector<llama_seq_id> ids={255,5,17,0};
        for (uint32_t count=1;count<=4;++count) for (uint32_t depth=1;depth<=7;++depth) {
            fixture f({ids.begin(),ids.begin()+count},depth);
            for (uint32_t prefix=0;prefix<=depth;++prefix) f.step(t,depth+1,prefix);
            f.step(t,1,0);
            f.step(t,32,1,true);
        }
    });
    t.test("abort_keeps_committed_image_and_refuses_uncommitted_export", [](testing & t) {
        fixture f({255,17,5,0},7);
        f.step(t,8,1);
        f.step(t,8,0,false,true);
        f.step(t,8,2);
    });
    t.test("parking_reselection_copy_and_slot_reuse_preserve_native_checkpoint_format", [](testing & t) {
        fixture f({255,17,5,0},7);
        f.step(t,8,0);
        std::vector<std::vector<uint8_t>> snapshots;
        for (auto id : f.ids) {byte_writer saved;f.memory->state_write(saved,id,0);snapshots.push_back(std::move(saved.bytes));}
        t.assert_true(f.memory->park_current_storage());
        t.assert_true(f.memory->restore_current_storage(f.lease.get()));
        for (size_t i=0;i<f.ids.size();++i) {
            byte_writer saved;f.memory->state_write(saved,f.ids[i],0);t.assert_true(saved.bytes==snapshots[i]);
        }
        fixture reselected({17,5,0,255},7);
        for (size_t i=f.ids.size();i>0;--i) {
            byte_reader saved(snapshots[i-1]);reselected.memory->state_read(saved,f.ids[i-1],0);
        }
        for (size_t i=0;i<f.ids.size();++i) {
            byte_writer saved;reselected.memory->state_write(saved,f.ids[i],0);t.assert_true(saved.bytes==snapshots[i]);
        }
        reselected.step(t,8,2);
        reselected.memory->seq_cp(17,255,0,-1);
        byte_writer source;reselected.memory->state_write(source,17,0);
        byte_writer copied;reselected.memory->state_write(copied,255,0);
        t.assert_true(source.bytes==copied.bytes);
        t.assert_true(reselected.memory->seq_rm(17,0,-1));
        byte_writer detached;reselected.memory->state_write(detached,255,0);
        t.assert_true(source.bytes==detached.bytes);
        reselected.step(t,4,1);
    });
    t.test("compact_bank_follows_selected_capacity_not_sparse_logical_ids", [](testing & t) {
        std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_QWEN35, llama_model_default_params()));
        model->hparams.n_layer_all = 1;
        model->hparams.ssm_d_conv = 4;
        model->hparams.ssm_d_inner = 384;
        model->hparams.ssm_d_state = 64;
        model->hparams.ssm_dt_rank = 6;
        model->hparams.ssm_n_group = 2;
        for (auto selected : std::vector<std::vector<llama_seq_id>>{{255}, {17, 255}, {255, 5, 17}, {17, 0, 255, 5}}) {
            const auto count = uint32_t(selected.size());
            llama_memory_recurrent memory(*model, GGML_TYPE_F32, GGML_TYPE_F32, false, count, 256, 7, nullptr,
                    false, true, true, true);
            t.assert_true(memory.compact_enabled());
            t.assert_true(memory.configure_selected_sequences(selected));
            t.assert_equal(int64_t(2*count), memory.r_l[0]->ne[1]);
            t.assert_equal(int64_t(2*count), memory.s_l[0]->ne[1]);
            for (uint32_t i = 0; i < count; ++i) {
                t.assert_equal(int32_t(i), memory.compact_current_row(selected[i]));
                t.assert_equal(int32_t(count+i), memory.compact_working_row(selected[i]));
            }
            ggml_backend_buffer_type_t type = nullptr;
            size_t bytes = 0;
            t.assert_true(memory.current_storage_requirements(type, bytes));
            const size_t image = size_t(model->hparams.n_embd_r() + model->hparams.n_embd_s()) * count * sizeof(float);
            t.assert_true(bytes > 2*image && bytes < 8*image);
        }
    });
    t.test("unsupported_geometry_and_target_only_retain_native_storage", [](testing & t) {
        std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_QWEN35, llama_model_default_params()));
        model->hparams.n_layer_all = 1;
        model->hparams.ssm_d_conv = 2;
        model->hparams.ssm_d_inner = 8;
        model->hparams.ssm_d_state = 8;
        model->hparams.ssm_n_group = 1;
        for (uint32_t depth : {0u, 3u}) {
            llama_memory_recurrent memory(*model, GGML_TYPE_F32, GGML_TYPE_F32, false, 1, 1, depth, nullptr,
                    false, false, false, true);
            t.assert_true(!memory.compact_enabled());
            t.assert_equal(int64_t(depth+1), memory.s_l[0]->ne[1]);
        }
    });
    return t.summary();
}
