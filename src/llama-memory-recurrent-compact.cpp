#include "llama-memory-recurrent-compact.h"
#include "llama-model.h"
#include "ggml-cpp.h"
#include "llama-memory-recurrent-history.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace {
// A detached, already produced span. Restoration must not re-execute its producer.
ggml_tensor * span(ggml_context * ctx, ggml_tensor * root, int64_t a, int64_t b, int64_t c,
        size_t offset, size_t stride = 0) {
    auto * view=ggml_view_3d(ctx,root,a,b,c,stride ? stride : a*sizeof(float),
            (stride ? stride : a*sizeof(float))*b,offset);
    view->op=GGML_OP_NONE;view->view_src=nullptr;view->buffer=root->buffer;
    std::fill(std::begin(view->src),std::end(view->src),nullptr);
    return view;
}

bool compute(ggml_backend_t backend, ggml_cgraph * graph) {
    using release_fn=void (*)(ggml_backend_t,const void *);
    auto * registry=ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    auto release=reinterpret_cast<release_fn>(ggml_backend_reg_get_proc_address(registry,"ggml_backend_cuda_graph_release"));
    bool okay=false;
    try {
        okay=ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS;
        ggml_backend_synchronize(backend);
    } catch (...) {
        try {ggml_backend_synchronize(backend);} catch (...) {}
    }
    if (release && ggml_graph_n_nodes(graph)) release(backend,ggml_graph_node(graph,0));
    return okay;
}
}

void llama_recurrent_compact_capture_history(ggml_context * ctx,ggml_cgraph * graph,ggml_tensor * input,
        uint32_t history,uint32_t depth,const llama_recurrent_compact_graph_storage & storage) {
    const int64_t tokens=input->ne[0]-history, rows=std::min<int64_t>(tokens,depth+1);
    const int64_t channels=input->ne[1],sequences=input->ne[2];
    auto * timeline=llama_recurrent_history_capture(ctx,input,history,uint32_t(rows));
    GGML_ASSERT(timeline && storage.slots && storage.destinations);
    if (rows<depth+1) timeline=ggml_pad(ctx,timeline,depth+1-rows,0,0,0);
    auto * packed=ggml_reshape_2d(ctx,timeline,timeline->ne[0]*channels,sequences);
    ggml_build_forward_expand(graph,ggml_set_rows(ctx,storage.timeline,packed,storage.slots));
    auto * last=ggml_view_3d(ctx,input,history,channels,sequences,input->nb[1],input->nb[2],tokens*sizeof(float));
    auto * current=ggml_reshape_2d(ctx,ggml_cont(ctx,last),history*channels,sequences);
    ggml_build_forward_expand(graph,ggml_set_rows(ctx,storage.history,current,storage.destinations));
}

std::pair<ggml_tensor *,ggml_tensor *> llama_recurrent_compact_capture_matrix(ggml_context * ctx,ggml_cgraph * graph,
        ggml_tensor * q,ggml_tensor * k,ggml_tensor * v,ggml_tensor * g,ggml_tensor * beta,ggml_tensor * state,
        uint32_t depth,const llama_recurrent_compact_graph_storage & storage) {
    const int64_t rows=depth+1, sequences=v->ne[3], elements=state->ne[0]*state->ne[1]*state->ne[2];
    auto * capture=ggml_gated_delta_net_capture(ctx,q,k,v,g,beta,state,rows);
    auto part=[&](ggml_gdn_capture_part p) {return ggml_gated_delta_net_capture_view(ctx,capture,p);};
    auto * final=ggml_reshape_2d(ctx,part(GGML_GDN_CAPTURE_FINAL),elements,sequences);
    ggml_build_forward_expand(graph,ggml_set_rows(ctx,storage.matrix,final,storage.destinations));
    ggml_tensor * records=nullptr;
    for (auto p : {GGML_GDN_CAPTURE_DECAY,GGML_GDN_CAPTURE_KEY,GGML_GDN_CAPTURE_CORRECTION}) {
        auto * field=part(p);
        if (field->ne[2]<rows) field=ggml_pad(ctx,field,0,0,rows-field->ne[2],0);
        field=ggml_reshape_2d(ctx,field,field->ne[0]*field->ne[1]*field->ne[2],sequences);
        records=records ? ggml_concat(ctx,records,field,0) : field;
    }
    ggml_build_forward_expand(graph,ggml_set_rows(ctx,storage.log,records,storage.slots));
    ggml_tensor * checkpoint=nullptr;
    if (v->ne[2]>rows) {
        checkpoint=ggml_cont(ctx,part(GGML_GDN_CAPTURE_CHECKPOINT));
        ggml_set_output(checkpoint);ggml_build_forward_expand(graph,checkpoint);
    }
    return {part(GGML_GDN_CAPTURE_ATTENTION),checkpoint};
}

bool llama_memory_recurrent::init_compact(const llama_model & model,ggml_type r,ggml_type s,bool offload,
        const layer_filter_cb & filter) {
    // This adapter is emitted by the Qwen scalar-decay GDN graph builder only.
    // The allocator and requirement comparison remain independent of model names.
    const int64_t w=hparams.ssm_d_state, heads=w ? hparams.ssm_d_inner/w : 0;
    const int64_t keys=hparams.ssm_n_group;
    if (model.arch!=LLM_ARCH_QWEN35 || (w!=64 && w!=128) || !heads || !keys || heads%keys ||
            hparams.ssm_dt_rank!=heads || hparams.ssm_d_inner%w || hparams.ssm_d_conv<2 ||
            hparams.ple_conv_state() || r!=GGML_TYPE_F32 || s!=GGML_TYPE_F32 || !size || size>n_seq_max || n_rs_seq>7) return false;
    ggml_context_ptr ctx(ggml_init({32*ggml_tensor_overhead(),nullptr,true}));
    if (!ctx) return false;
    auto * q=ggml_new_tensor_4d(ctx.get(),GGML_TYPE_F32,w,keys,1,1);
    auto * v=ggml_new_tensor_4d(ctx.get(),GGML_TYPE_F32,w,heads,1,1);
    auto * g=ggml_new_tensor_4d(ctx.get(),GGML_TYPE_F32,1,heads,1,1);
    auto * state=ggml_new_tensor_4d(ctx.get(),GGML_TYPE_F32,w,w,heads,1);
    auto * capture=ggml_gated_delta_net_capture(ctx.get(),q,q,v,g,g,state,n_rs_seq+1);
    auto part=[&](ggml_gdn_capture_part p) {return ggml_gated_delta_net_capture_view(ctx.get(),capture,p);};
    auto * replay=ggml_gated_delta_net_replay_prefix(ctx.get(),part(GGML_GDN_CAPTURE_CHECKPOINT),part(GGML_GDN_CAPTURE_DECAY),
            part(GGML_GDN_CAPTURE_KEY),part(GGML_GDN_CAPTURE_CORRECTION),0);
    bool any=false;
    for (uint32_t layer=0;layer<hparams.n_layer();++layer) if (!filter || filter(layer)) {
        auto * device=offload ? model.dev_layer(layer) : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!device || !ggml_backend_dev_supports_op(device,capture) || !ggml_backend_dev_supports_op(device,replay)) return false;
        if (ggml_backend_dev_type(device)!=GGML_BACKEND_DEVICE_TYPE_CPU) {
            auto * registry=ggml_backend_dev_backend_reg(device);
            using contract_fn=bool (*)(ggml_backend_dev_t);
            auto contract=reinterpret_cast<contract_fn>(ggml_backend_reg_get_proc_address(registry,"ggml_backend_gdn_fp32_replay_exact"));
            if (!contract || !contract(device) || !ggml_backend_reg_get_proc_address(registry,"ggml_backend_cuda_graph_release")) return false;
        }
        any=true;
    }
    if (!any) return false;
    auto next=std::make_unique<compact_storage_state>();
    next->slots.assign(n_seq_max,-1);next->sequences.resize(size);
    next->logs.resize(hparams.n_layer());next->timelines.resize(hparams.n_layer());
    for (uint32_t i=0;i<size;++i) {next->slots[i]=int32_t(i);next->sequences[i].logical=llama_seq_id(i);}
    compact=std::move(next);
    return true;
}

bool llama_memory_recurrent::compact_enabled() const noexcept {return bool(compact);}
int32_t llama_memory_recurrent::compact_slot(llama_seq_id id) const {
    return compact && id>=0 && uint32_t(id)<n_seq_max ? compact->slots[id] : -1;
}
int32_t llama_memory_recurrent::compact_current_row(llama_seq_id id) const {
    auto slot=compact_slot(id);
    for (size_t count=0;slot>=0 && compact->sequences[slot].alias>=0;++count) {
        if (count>=compact->sequences.size()) return -1;
        slot=compact->sequences[slot].alias;
    }
    return slot<0 ? -1 : int32_t(compact->sequences[slot].role*size+slot);
}
int32_t llama_memory_recurrent::compact_working_row(llama_seq_id id) const {
    const auto slot=compact_slot(id);
    return slot<0 ? -1 : int32_t((1-compact->sequences[slot].role)*size+slot);
}
ggml_tensor * llama_memory_recurrent::compact_log(int32_t layer) const {return compact ? compact->logs.at(layer) : nullptr;}
ggml_tensor * llama_memory_recurrent::compact_timeline(int32_t layer) const {return compact ? compact->timelines.at(layer) : nullptr;}

void llama_memory_recurrent::clear_compact(llama_seq_id id) {
    if (!compact) return;
    cancel_compact_capture();
    for (auto & sequence : compact->sequences) if (id<0 || sequence.logical==id) {
        const auto logical=sequence.logical;sequence={};sequence.logical=logical;
    }
    if (id<0 || std::all_of(compact->sequences.begin(),compact->sequences.end(),[](const auto & s) {return s.fresh;})) {
        compact->failed=false;compact->poisoned=false;
    }
}

bool llama_memory_recurrent::materialize_compact_aliases() {
    if (!compact) return true;
    for (size_t destination=0;destination<compact->sequences.size();++destination) {
        auto & target=compact->sequences[destination];
        if (target.alias<0) continue;
        auto source=target.alias;
        for (size_t count=0;compact->sequences[source].alias>=0;++count) {
            if (count>=compact->sequences.size()) return false;
            source=compact->sequences[source].alias;
        }
        const auto saved=compact->sequences[source];
        ggml_context_ptr ctx(ggml_init({16*r_l.size()*ggml_tensor_overhead(),nullptr,true}));
        if (!ctx) return false;
        try {
            for (size_t layer=0;layer<r_l.size();++layer) {
                for (auto * tensor : {r_l[layer],s_l[layer]}) if (tensor) {
                    for (uint32_t role=0;role<2;++role) if (saved.valid || role==saved.role) {
                        auto * src=span(ctx.get(),tensor,tensor->ne[0],1,1,(role*size+source)*tensor->nb[1]);
                        auto * dst=span(ctx.get(),tensor,tensor->ne[0],1,1,(role*size+destination)*tensor->nb[1]);
                        ggml_backend_tensor_copy(src,dst);
                    }
                }
                if (saved.valid) for (auto * tensor : {compact->logs[layer],compact->timelines[layer]}) if (tensor) {
                    auto * src=span(ctx.get(),tensor,tensor->ne[0],1,1,source*tensor->nb[1]);
                    auto * dst=span(ctx.get(),tensor,tensor->ne[0],1,1,destination*tensor->nb[1]);
                    ggml_backend_tensor_copy(src,dst);
                }
            }
        } catch (...) {return false;}
        const auto logical=target.logical;target=saved;target.logical=logical;target.alias=-1;
    }
    return true;
}

bool llama_memory_recurrent::compact_rollback_valid(llama_seq_id id,uint32_t rollback) const {
    const auto slot=compact_slot(id);
    if (slot<0 || compact->capturing || compact->failed) return false;
    const auto & sequence=compact->sequences[slot];
    return sequence.valid && !sequence.rolled && rollback<sequence.rows;
}

bool llama_memory_recurrent::materialize_compact() {
    if (!compact) return true;
    if (compact->capturing || compact->poisoned) return false;
    const int64_t w=hparams.ssm_d_state, heads=hparams.ssm_dt_rank, keys=hparams.ssm_n_group;
    const int64_t rows=n_rs_seq+1, history=hparams.ssm_d_conv-1, channels=hparams.n_embd_r()/history;
    const size_t nodes=std::max<size_t>(2048,32*r_l.size()*compact->sequences.size());
    ggml_context_ptr ctx(ggml_init({nodes*ggml_tensor_overhead()+ggml_graph_overhead_custom(nodes,false),nullptr,true}));
    if (!ctx) return false;
    auto * graph=ggml_new_graph_custom(ctx.get(),nodes,false);
    bool replaying=false;
    for (size_t slot=0;slot<compact->sequences.size();++slot) {
        const auto & sequence=compact->sequences[slot];
        if (sequence.logical<0) continue;
        const uint32_t rollback=rs_idx[sequence.logical];
        if (!rollback) continue;
        if (!sequence.valid || rollback>=sequence.rows || !compact->backend) return false;
        replaying=true;
        const int32_t prefix=int32_t(sequence.rows-rollback);
        for (size_t layer=0;layer<r_l.size();++layer) if (s_l[layer]) {
            auto * s=s_l[layer];auto * r=r_l[layer];auto * log=compact->logs[layer];
            auto * checkpoint=span(ctx.get(),s,w,w,heads,((1-sequence.working)*size+slot)*s->nb[1]);
            auto * decay=span(ctx.get(),log,1,heads,rows,slot*log->nb[1]);
            auto * key=span(ctx.get(),log,w,keys,rows,slot*log->nb[1]+heads*rows*sizeof(float));
            auto * correction=span(ctx.get(),log,w,heads,rows,slot*log->nb[1]+(heads+w*keys)*rows*sizeof(float));
            auto * restored=ggml_gated_delta_net_replay_prefix(ctx.get(),checkpoint,decay,key,correction,prefix);
            restored->buffer=s->buffer;restored->data=(char *)s->data+(sequence.working*size+slot)*s->nb[1];
            ggml_build_forward_expand(graph,restored);
            auto * timeline=compact->timelines[layer];
            auto * window=span(ctx.get(),timeline,history,channels,1,slot*timeline->nb[1]+prefix*sizeof(float),
                    (history+rows)*sizeof(float));
            auto * destination=span(ctx.get(),r,history,channels,1,(sequence.working*size+slot)*r->nb[1]);
            ggml_build_forward_expand(graph,ggml_cpy(ctx.get(),window,destination));
        }
    }
    if (replaying && !compute(compact->backend,graph)) {compact->failed=true;return false;}
    for (auto & sequence : compact->sequences) if (sequence.logical>=0) {
        if (sequence.pending || rs_idx[sequence.logical]) sequence.role=sequence.working;
        if (rs_idx[sequence.logical]) sequence.rolled=true;
        sequence.pending=false;rs_idx[sequence.logical]=0;
    }
    compact->failed=false;
    return materialize_compact_aliases();
}

uint64_t llama_memory_recurrent::begin_compact_capture(ggml_backend_t backend,const llama_ubatch & batch,bool prefill) {
    if (!compact || !backend || compact->capturing || compact->poisoned || !batch.n_seq_tokens || !batch.equal_seqs() ||
            (!prefill && batch.n_seq_tokens>n_rs_seq+1) || compact->generation==UINT64_MAX) return 0;
    if (current_storage_pending()) return 0;
    for (auto * tensor : r_l) if (tensor) {
        auto * device=ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer));
        auto * execution=ggml_backend_get_device(backend);
        if (device ? device!=execution : ggml_backend_dev_type(execution)!=GGML_BACKEND_DEVICE_TYPE_CPU ||
                !ggml_backend_buffer_is_host(tensor->buffer)) return 0;
    }
    compact->backend=backend;
    if (!materialize_compact()) return 0;
    std::vector<uint32_t> active;
    for (uint32_t i=0;i<batch.n_seqs;++i) {
        const auto slot=compact_slot(batch.seq_id_unq[i]);
        if (slot<0 || std::find(active.begin(),active.end(),uint32_t(slot))!=active.end()) return 0;
        active.push_back(uint32_t(slot));
    }
    ggml_backend_synchronize(backend);
    for (auto slot : active) {
        auto & sequence=compact->sequences[slot];
        if (sequence.fresh) for (size_t layer=0;layer<r_l.size();++layer) for (auto * tensor : {r_l[layer],s_l[layer]}) if (tensor)
            ggml_backend_tensor_memset(tensor,0,slot*tensor->nb[1],tensor->nb[1]);
        sequence.fresh=false;sequence.valid=false;sequence.rolled=false;sequence.working=1-sequence.role;
        sequence.generation=compact->generation+1;
    }
    compact->active=std::move(active);compact->capturing=true;compact->prefill=prefill;
    return ++compact->generation;
}

void llama_memory_recurrent::cancel_compact_capture() noexcept {
    if (!compact || !compact->capturing) return;
    for (auto slot : compact->active) {compact->sequences[slot].valid=false;compact->sequences[slot].pending=false;}
    compact->capturing=false;compact->failed=true;
    compact->poisoned=true;
}

bool llama_memory_recurrent::finish_compact_capture(const llama_ubatch & batch,
        const std::vector<std::pair<ggml_tensor *,uint32_t>> & checkpoints,uint64_t generation) {
    if (!compact || !compact->capturing || compact->generation!=generation ||
            compact->active.size()!=batch.n_seqs || !compact->backend) return false;
    for (uint32_t i=0;i<batch.n_seqs;++i) if (compact_slot(batch.seq_id_unq[i])!=int32_t(compact->active[i]) ||
            compact->sequences[compact->active[i]].generation!=generation) return false;
    ggml_backend_synchronize(compact->backend);
    if (batch.n_seq_tokens>n_rs_seq+1) {
        if (!compact->prefill || checkpoints.size()!=size_t(std::count_if(s_l.begin(),s_l.end(),[](auto * s) {return s!=nullptr;}))) return false;
        const size_t nodes=std::max<size_t>(2048,16*r_l.size()*batch.n_seqs);
        ggml_context_ptr ctx(ggml_init({nodes*ggml_tensor_overhead()+ggml_graph_overhead_custom(nodes,false),nullptr,true}));
        if (!ctx) return false;
        auto * graph=ggml_new_graph_custom(ctx.get(),nodes,false);
        std::vector<uint8_t> seen(s_l.size(),0);
        for (const auto & checkpoint : checkpoints) {
            auto * tensor=checkpoint.first;const uint32_t layer=checkpoint.second;
            if (layer>=s_l.size() || seen[layer] || !s_l[layer] || !tensor || !ggml_is_contiguous(tensor) ||
                    ggml_nelements(tensor)!=int64_t(batch.n_seqs)*s_l[layer]->ne[0]) return false;
            seen[layer]=1;
            for (uint32_t i=0;i<batch.n_seqs;++i) {
                const auto slot=compact->active[i];const auto & sequence=compact->sequences[slot];
                auto * source=span(ctx.get(),tensor,s_l[layer]->ne[0],1,1,i*s_l[layer]->nb[1]);
                auto * destination=span(ctx.get(),s_l[layer],s_l[layer]->ne[0],1,1,
                        (sequence.role*size+slot)*s_l[layer]->nb[1]);
                ggml_build_forward_expand(graph,ggml_cpy(ctx.get(),source,destination));
            }
        }
        if (!compute(compact->backend,graph)) {compact->failed=true;return false;}
    }
    for (auto slot : compact->active) {
        auto & sequence=compact->sequences[slot];
        sequence.rows=std::min(batch.n_seq_tokens,n_rs_seq+1);sequence.valid=true;
        sequence.pending=batch.n_seq_tokens<=n_rs_seq+1;
        if (!sequence.pending) sequence.role=sequence.working;
    }
    compact->capturing=false;
    return true;
}

bool llama_memory_recurrent_context::compact_enabled() const {return mem && mem->compact_enabled();}
int32_t llama_memory_recurrent_context::compact_slot(llama_seq_id id) const {return mem ? mem->compact_slot(id) : -1;}
int32_t llama_memory_recurrent_context::compact_working_row(llama_seq_id id) const {return mem ? mem->compact_working_row(id) : -1;}
ggml_tensor * llama_memory_recurrent_context::compact_log(int32_t layer) const {return mem ? mem->compact_log(layer) : nullptr;}
ggml_tensor * llama_memory_recurrent_context::compact_timeline(int32_t layer) const {return mem ? mem->compact_timeline(layer) : nullptr;}
