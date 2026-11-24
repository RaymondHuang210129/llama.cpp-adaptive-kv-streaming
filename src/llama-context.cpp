#include "llama-context.h"
#include "llama-memory-workspace.h"

#include "ggml.h"
#include "llama-arch.h"
#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-memory.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-bounded-kv.h"
#include "llama-kv-cache-iswa.h"
#include "llama-kv-stream-logical-cache.h"
#include "llama-kv-stream-model.h"
#include "llama-kv-stream-positions.h"
#include "llama-mmap.h"
#include "llama-model.h"
#include "llama-moe-cache.h"
#include "llama-ext.h"
#include "llama-sampler.h"
#include "llama.h"
#include "../ggml/src/ggml-backend-execution.h"
#include "../ggml/src/ggml-kv-stream-device.h"
#include "../ggml/src/ggml-kv-stream-copy.h"
#include "llama-kv-stream-writer.h"

#include <cinttypes>
#include <algorithm>
#include <cmath>
#include <atomic>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>

//
// llama_context
//

static void llama_log_memory_phase(const llama_context_memory * memory) {
    llama_context_memory_diagnostics d;
    if (!memory || !memory->diagnostics(d)) return;
    const char * phase = d.kv_device_suspended ? "suspended" : d.phase == llama_memory_text_phase::decode ? "decode" : "prefill";
    LLAMA_LOG_WARN("memory_phase: phase=%s parent=%zu compute=%zu kv_pool=%zu writer=%zu attention=%zu persistent_kv=%zu publication_stage=%zu unused=%zu reclaimed_compute=%zu capture=external transition_us=%" PRIu64 " arena_gen=%" PRIu64 " kv_revision=%" PRIu64 " resident_pages=%u ring_slots=%u active_pages=%u streaming=%d last_h2d_bytes=%zu last_h2d_calls=%zu copy_ms=%.3f elapsed_ms=%.3f\n",
        phase,d.parent_bytes,d.workspace_bytes,d.kv_pool_bytes,d.kv_writer_bytes,
        d.kv_attention_bytes,d.persistent_cache_bytes,d.publication_stage_bytes,d.unused_bytes,d.reclaimed_workspace_bytes,
        d.last_transition_us,d.arena_generation,d.layout_revision,
        d.resident_pages_per_layer,d.ring_slots,d.active_pages,
        int(d.streaming_active),d.last_copy_bytes,d.last_copy_calls,
        d.last_copy_ms,d.last_elapsed_ms);
}

static llm_graph_type ctx_type_to_graph_type(llama_context_type ctx_type) {
    switch (ctx_type) {
        case LLAMA_CONTEXT_TYPE_DEFAULT: return LLM_GRAPH_TYPE_DEFAULT;
        case LLAMA_CONTEXT_TYPE_MTP    : return LLM_GRAPH_TYPE_DECODER_MTP;
    }
    throw std::runtime_error("Unsupported ctx type");
}

static bool group_rollback_supported(const llama_model & model,uint32_t depth) {
    return !depth || (model.arch==LLM_ARCH_QWEN35 && depth<=7);
}

static bool group_auxiliary_supported(const llama_model & model,const llama_context_params & params) {
    return !params.kv_stream_auxiliary_layers || (params.kv_stream_auxiliary_layers==1 && model.arch==LLM_ARCH_QWEN35 &&
        model.hparams.n_layer_nextn==1 && params.n_rs_seq>0 && params.n_rs_seq<=3);
}

static std::vector<llama_seq_id> group_slot_ids(uint32_t count) {
    if (count < 2 || count > LLAMA_MAX_SEQ) throw std::invalid_argument("invalid grouped logical sequence count");
    std::vector<llama_seq_id> result;
    result.reserve(count);
    for (uint32_t slot = 0; slot < count; ++slot) result.push_back(llama_seq_id(slot));
    return result;
}

static uint32_t group_verify_rows(const llama_cparams & params) {
    return params.kv_stream_verify_max ? params.kv_stream_verify_max : params.n_rs_seq ? params.n_rs_seq+1 : 1;
}

static bool group_next_position(const llama_kv_cells & cells,size_t tokens,llama_pos & output) {
    if (!tokens) {output=0;return true;}
    if (tokens>cells.size() || cells.is_empty(uint32_t(tokens-1))) return false;
    auto last=cells.pos_get(uint32_t(tokens-1));
    const auto & ext=cells.ext_get(uint32_t(tokens-1));
    if (ext.tok==LLAMA_TOKEN_NULL) last=std::max({last,ext.x,ext.y});
    if (last<0 || last==INT32_MAX) return false;
    output=last+1;return true;
}

static bool group_dflash_supported(const llama_context & target,const llama_model & model,
        const llama_context_params & params,uint32_t rows) {
    const auto & source=target.get_model();const auto & cp=target.get_cparams();const auto & hp=model.hparams;
    const uint32_t micro=std::min(params.n_batch,params.n_ubatch ? params.n_ubatch : params.n_batch);
    if (source.arch!=LLM_ARCH_QWEN35 || model.arch!=LLM_ARCH_DFLASH || !cp.shared_device_memory_bytes ||
            cp.kv_stream_auxiliary_layers || cp.n_seq_max<2 || cp.n_seq_max>LLAMA_MAX_SEQ || !cp.n_rs_seq || cp.n_rs_seq>7 ||
            params.ctx_type!=LLAMA_CONTEXT_TYPE_DEFAULT || params.n_seq_max!=cp.n_seq_max || !params.kv_unified ||
            params.n_ctx!=target.n_ctx() || params.shared_device_memory_bytes || params.kv_stream_pool_bytes ||
            params.kv_stream_auxiliary_layers || params.kv_stream_verify_max || params.n_rs_seq || params.swa_full ||
            params.embeddings || !params.offload_kqv || params.flash_attn_type!=LLAMA_FLASH_ATTN_TYPE_ENABLED ||
            (params.ctx_other && params.ctx_other!=&target) || !hp.n_swa || !hp.dflash_selector_rank ||
            hp.dflash_block_size<2 || hp.dflash_block_size>8 || rows<2 || rows>hp.dflash_block_size ||
            rows>cp.n_rs_seq+1 || rows>micro || !target.get_kv_resources().full ||
            params.type_k!=target.get_kv_resources().full->type_k() || params.type_v!=target.get_kv_resources().full->type_v() ||
            source.devices.size()!=1 || model.devices.size()!=1 || source.devices.front().dev!=model.devices.front().dev ||
            !source.hparams.n_embd || model.target_layer_ids.empty() ||
            model.target_layer_ids.size()>INT32_MAX/source.hparams.n_embd ||
            hp.n_embd_inp_enc()!=model.target_layer_ids.size()*source.hparams.n_embd) return false;
    for (uint32_t layer=0;layer<hp.n_layer();++layer) if (!hp.is_swa(layer)) return false;
    for (size_t i=0;i<model.target_layer_ids.size();++i) {
        const auto layer=model.target_layer_ids[i];
        if (layer<0 || uint32_t(layer)>source.hparams.n_layer() ||
                std::find(model.target_layer_ids.begin(),model.target_layer_ids.begin()+i,layer)!=model.target_layer_ids.begin()+i) return false;
    }
    return true;
}

struct llm_fused_op_probe {
    llm_fused_op op;
    const char * name;
    uint32_t n_tokens_per_seq;
};

static const llm_fused_op_probe llm_fused_op_flash_attn_probe = {
    /*.op               =*/ LLM_FUSED_OP_FLASH_ATTN,
    /*.name             =*/ "Flash Attention",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_gdn_ar_probe = {
    /*.op               =*/ LLM_FUSED_OP_GDN_AR,
    /*.name             =*/ "fused Gated Delta Net (autoregressive)",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_gdn_ch_probe = {
    /*.op               =*/ LLM_FUSED_OP_GDN_CH,
    /*.name             =*/ "fused Gated Delta Net (chunked)",
    /*.n_tokens_per_seq =*/ 16,
};

static const llm_fused_op_probe llm_fused_op_lid_probe = {
    /*.op               =*/ LLM_FUSED_OP_LIGHTNING_INDEXER,
    /*.name             =*/ "Lightning Indexer",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_pre_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_PRE,
    /*.name             =*/ "fused DeepSeek V4 HC pre",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_comb_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_COMB,
    /*.name             =*/ "fused DeepSeek V4 HC comb",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_post_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_POST,
    /*.name             =*/ "fused DeepSeek V4 HC post",
    /*.n_tokens_per_seq =*/ 1,
};

struct llama_context::group_execution_binding {
    llama_context & context;
    std::shared_ptr<llama_kv_stream_slot_table> table;
    std::shared_ptr<llama_kv_stream_slot_projection> projection;
    std::shared_ptr<void> resources;
    std::shared_ptr<llama_memory_phase_owner> owner;
    llama_memory_work_phase phase;
    uint32_t profile;
    uint64_t revision,admission=0;
    bool failed=false,mutating=false;
    bool shared_roles=false;
    bool shared_readonly=false;
    bool shared_block=false;
    std::vector<llama_kv_stream_logical_checkpoint> draft_sources = {},pending_draft_sources = {};
    std::vector<llama_pos> draft_positions = {},pending_draft_positions = {};
    std::vector<llama_kv_stream_logical_checkpoint> draft_suffixes = {};
    struct source_append {
        size_t source;
        llama_kv_stream_logical_checkpoint before;
        size_t rows;
        bool pinned_image;
    };
    std::vector<source_append> source_appends = {};
    std::function<bool(const llama_ubatch &)> read_prepare={};
    std::function<bool(bool)> read_release={};
    std::function<bool(uint32_t)> reader_prepare={};
    std::function<bool()> reader_finish={};
    group_execution_binding * peer=nullptr;

    ~group_execution_binding() {
        if (!shared_roles || !owner) return;
        if (peer && peer->admission) peer->finish(false);
        if (!owner->close()) {
            failed=true;admission=0;
            if (peer) {peer->failed=true;peer->admission=0;peer->owner.reset();}
            owner.reset();
        }
        if (peer) {peer->failed=true;peer->peer=nullptr;}
    }

    bool identities(std::vector<llama_memory_phase_cache_identity> & output) const {
        if (!shared_roles) {
            for (const auto & source : projection->sources) output.push_back({source.id,source.generation});
            return true;
        }
        for (uint32_t role=0;role<(shared_readonly || shared_block ? 1u : 2u);++role) for (const auto & handle : projection->handles) {
            std::shared_ptr<llama_kv_stream_logical_cache> cache;
            std::shared_ptr<llama_kv_cells> cells;
            if (!table->get(handle,role,cache) || !cache->ready() || !table->get_cells(handle,role,cells) ||
                    cells->get_has_shift() || cells->get_used()!=cache->tokens()) return false;
            output.push_back({cache->identity().id,cache->identity().generation});
        }
        return true;
    }

    bool prepare_dflash(const llama_batch_ext & batch,llama_memory_work_phase & next_phase) {
        auto * target=peer ? &peer->context : nullptr;
        if (!target || context.group_target!=target || peer!=target->group_execution.get() || peer->failed || peer->admission ||
                peer->owner!=owner || !table->projection_current(*peer->projection) || batch.tokens.empty() ||
                batch.tokens.size()>context.cparams.n_ubatch || draft_sources.size()!=projection->handles.size() ||
                batch.n_pos_per_embd!=context.model.hparams.n_pos_per_embd()) return false;
        const bool inject=batch.tokens.front().has_embd;
        if (std::any_of(batch.tokens.begin(),batch.tokens.end(),[&](const auto & token) {return token.has_embd!=inject || token.output;})) return false;
        if (inject) {
            if (batch.n_embd!=context.model.hparams.n_embd_inp_enc() ||
                    target->serial_feature_rows.empty() || target->serial_feature_rows.size()>INT32_MAX) return false;
            if (!target->serial_feature_image_rows.empty()) {
                if (!target->serial_feature_images || target->serial_feature_image_rows.size()!=target->serial_feature_rows.size()) return false;
                llama_batch_ext source(target);
                if (target->serial_feature_rows.size()>source.n_tokens_max) return false;
                source.n_embd=target->serial_feature_image_width;source.embd=target->serial_feature_image_values;
                for (size_t row=0;row<target->serial_feature_rows.size();++row) {
                    const auto & saved=target->serial_feature_rows[row];const auto & image=target->serial_feature_image_rows[row];
                    llama_batch_ext::token token;token.id=saved.token;token.seq_ids.insert(saved.sequence);
                    token.pos=image.position;token.has_embd=image.has_embd;token.embd_off=image.offset;
                    source.tokens.push_back(std::move(token));
                }
                if (!target->serial_features_match(source,context.model.target_layer_ids.data(),uint32_t(context.model.target_layer_ids.size()))) return false;
            } else {
                auto input=llama_batch_init(int32_t(target->serial_feature_rows.size()),0,1);
                struct release_batch {llama_batch & batch;~release_batch() {llama_batch_free(batch);}} release{input};
                input.n_tokens=int32_t(target->serial_feature_rows.size());
                for (int32_t row=0;row<input.n_tokens;++row) {
                    const auto & saved=target->serial_feature_rows[row];input.token[row]=saved.token;input.pos[row]=saved.position;
                    input.n_seq_id[row]=1;input.seq_id[row][0]=saved.sequence;input.logits[row]=false;
                }
                llama_batch_compat source(target,input);
                if (!target->serial_features_match(*source.batch_ext,context.model.target_layer_ids.data(),uint32_t(context.model.target_layer_ids.size()))) return false;
            }
        }
        pending_draft_sources.clear();
        pending_draft_positions.clear();
        if (draft_positions.size()!=draft_sources.size()) return false;
        if (inject) {pending_draft_sources=draft_sources;pending_draft_positions=draft_positions;}
        uint32_t common_width=0;
        size_t active=0;
        for (size_t source=0;source<projection->handles.size();++source) {
            const auto handle=projection->handles[source];const auto sequence=llama_seq_id(handle.slot);
            std::vector<const llama_batch_ext::token *> selected;
            for (const auto & token : batch.tokens) {
                if (token.seq_ids.size()!=1) return false;
                if (token.seq_ids.count(sequence)) selected.push_back(&token);
            }
            if (selected.empty()) continue;
            ++active;
            std::shared_ptr<llama_kv_stream_logical_cache> cache;
            if (!table->get(handle,0,cache) || !cache->ready()) return false;
            const auto current=cache->checkpoint(),saved=draft_sources[source];
            if (saved.id && (saved.id!=current.id || saved.generation!=current.generation || saved.tokens>current.tokens)) return false;
            const size_t first=inject ? saved.tokens : current.tokens;
            llama_pos native_first=0;
            std::shared_ptr<llama_kv_cells> target_cells;
            if (!table->get_cells(handle,0,target_cells)) return false;
            if (!inject && (!group_next_position(*target_cells,current.tokens,native_first) ||
                    draft_positions[source]<-1 || draft_positions[source]>=native_first ||
                    uint64_t(selected.size())>uint64_t(INT32_MAX)-uint32_t(native_first))) return false;
            if (!inject && (!saved.id || saved.tokens!=current.tokens || selected.size()<2 ||
                    selected.size()>context.cparams.serial_block_rows || (common_width && common_width!=selected.size()) ||
                    context.memory->seq_pos_max(sequence)<draft_positions[source])) return false;
            common_width=uint32_t(selected.size());
            const auto & allowance=projection->selected.slots[source];
            const uint64_t end=uint64_t(allowance.history_tokens)+allowance.growth_tokens;
            if (first>end || selected.size()>end-first || first>INT32_MAX || selected.size()>uint64_t(INT32_MAX)-first) return false;
            if (inject && (first>current.tokens || selected.size()>current.tokens-first ||
                    current.tokens>target_cells->size())) return false;
            for (size_t row=0;row<selected.size();++row) {
                const auto & token=*selected[row];
                if (inject) {
                    if (target_cells->is_empty(uint32_t(first+row)) || token.pos[0]!=target_cells->pos_get(uint32_t(first+row))) return false;
                } else if (token.pos[0]!=native_first+llama_pos(row)) return false;
                if (inject) {
                    if (batch.n_pos_per_embd==4 && (token.pos[1]!=token.pos[0] || token.pos[2]!=token.pos[0] || token.pos[3])) return false;
                    if (token.id!=LLAMA_TOKEN_NULL || token.embd_off>batch.embd.size() || batch.n_embd>batch.embd.size()-token.embd_off) return false;
                    const auto found=std::find_if(target->serial_feature_rows.begin(),target->serial_feature_rows.end(),
                        [&](const auto & item) {return item.sequence==sequence && item.position==token.pos[0];});
                    if (found==target->serial_feature_rows.end()) return false;
                    const auto index=size_t(found-target->serial_feature_rows.begin());
                    for (size_t layer=0;layer<context.model.target_layer_ids.size();++layer) {
                        const auto * expected=target->get_embeddings_layer_inp(uint32_t(context.model.target_layer_ids[layer]));
                        const auto width=target->model.hparams.n_embd;
                        if (!expected || std::memcmp(expected+index*width,batch.embd.data()+token.embd_off+layer*width,size_t(width)*sizeof(float))) return false;
                    }
                } else if (token.id<0 || uint32_t(token.id)>=context.model.vocab.n_tokens() ||
                        (row && token.id!=llama_vocab_mask(llama_model_get_vocab(&context.model)))) return false;
            }
            if (inject) {
                pending_draft_sources[source]=current;pending_draft_sources[source].tokens=first+selected.size();
                pending_draft_positions[source]=selected.back()->pos[0];
            }
        }
        if (!active || (inject && active!=1)) return false;
        next_phase=inject ? llama_memory_work_phase::draft_feature_injection : llama_memory_work_phase::draft_block_prediction;
        return true;
    }

    bool begin(const llama_batch_ext & batch) {
        if (failed || admission || !table->projection_current(*projection)) return false;
        source_appends.clear();
        if (shared_readonly && context.cparams.ctx_type==LLAMA_CONTEXT_TYPE_MTP &&
                (!read_prepare || !read_release || !context.group_target || !context.group_target->group_mtp_ready(0))) return false;
        try {
            auto next_phase=phase;
            const bool block_child=shared_block && context.model.arch==LLM_ARCH_DFLASH;
            if (shared_roles) {
                if (block_child) {
                    if (!prepare_dflash(batch,next_phase)) return false;
                } else if (context.cparams.ctx_type==LLAMA_CONTEXT_TYPE_MTP) {
                    const bool predict=std::all_of(batch.tokens.begin(),batch.tokens.end(),[](const auto & token) {return token.output;});
                    if (!predict && std::any_of(batch.tokens.begin(),batch.tokens.end(),[](const auto & token) {return token.output;})) return false;
                    next_phase=predict ? llama_memory_work_phase::mtp_draft : llama_memory_work_phase::mtp_catchup;
                } else next_phase=context.cparams.kv_stream_decode ? llama_memory_work_phase::target_verify : llama_memory_work_phase::target_prefill;
            }
            for (const auto & token : batch.tokens) {
                if (token.seq_ids.size()!=1) return false;
                const auto id=*token.seq_ids.begin();
                if (std::none_of(projection->handles.begin(),projection->handles.end(),
                        [&](const auto & handle) {return llama_seq_id(handle.slot)==id;})) return false;
            }
            for (size_t source=0;source<projection->handles.size();++source) {
                const auto & handle=projection->handles[source];
                if (block_child) break;
                llama_batch_ext selected(batch.n_tokens_max,batch.n_embd_inp,batch.n_embd_inp_enc,batch.n_seq_max,
                    nullptr,batch.n_vocab,batch.n_pos_per_embd);
                selected.n_embd=batch.n_embd;selected.embd=batch.embd;
                for (const auto & token : batch.tokens) if (token.seq_ids.count(llama_seq_id(handle.slot))) selected.tokens.push_back(token);
                if (selected.tokens.empty()) continue;
                if (next_phase==llama_memory_work_phase::mtp_draft && selected.tokens.size()!=1) return false;
                if (next_phase==llama_memory_work_phase::target_verify && selected.tokens.size()>
                        context.kv_resources.full->get_group_model()->maximum_verify_rows()) return false;
                std::shared_ptr<llama_kv_stream_logical_cache> cache;std::shared_ptr<llama_kv_cells> cells;
                if (!table->get(handle,projection->role,cache) || !table->get_cells(handle,projection->role,cells)) return false;
                const auto tokens=cache->tokens();
                llama_kv_stream_append_coordinates coordinates{tokens,table->limits().slot_limit,
                    tokens ? cells->pos_get(uint32_t(tokens-1)) : -1,uint32_t(batch.n_pos_per_embd),
                    next_phase==llama_memory_work_phase::target_verify || next_phase==llama_memory_work_phase::mtp_draft,false};
                coordinates.sequence=llama_seq_id(handle.slot);
                coordinates.token_hidden=shared_roles && context.cparams.ctx_type==LLAMA_CONTEXT_TYPE_MTP &&
                    !context.mtp_visual_hidden;
                if (!llama_kv_stream_validate_append(selected,coordinates)) return false;
                if (shared_block && context.model.arch==LLM_ARCH_QWEN35 && peer) {
                    const bool pinned=context.serial_feature_images && batch.n_pos_per_embd==4 && selected.tokens.size()>1 &&
                        std::all_of(selected.tokens.begin(),selected.tokens.end(),[&](const auto & token) {
                            return token.id==LLAMA_TOKEN_NULL && token.has_embd && token.pos[0]==selected.tokens.front().pos[0];
                        });
                    source_appends.push_back({source,cache->checkpoint(),selected.tokens.size(),pinned});
                }
            }
            std::vector<llama_memory_phase_cache_identity> sources;
            if (!identities(sources)) return false;
            const auto entered=owner->enter(next_phase,uint32_t(batch.tokens.size()),sources,profile,revision);
            if (entered.status!=llama_memory_phase_owner_status::changed && entered.status!=llama_memory_phase_owner_status::unchanged) return false;
            admission=owner->admit();
            if (!admission) return false;
            phase=next_phase;
            if (shared_roles) context.cparams.kv_stream_decode=phase==llama_memory_work_phase::target_verify || phase==llama_memory_work_phase::mtp_draft;
            return true;
        } catch (...) {return false;}
    }
    bool mutate(const std::function<bool()> & mutation) noexcept {
        if (failed || (!mutating && admission) || !table->projection_current(*projection)) return false;
        try {
            if (mutating) return admission && mutation();
            std::vector<llama_memory_phase_cache_identity> sources;
            if (shared_roles && !identities(sources)) return false;
            context.synchronize();
            const auto entered=owner->enter(phase,1,shared_roles ? sources : owner->source_identities(),profile,revision);
            if (entered.status!=llama_memory_phase_owner_status::changed && entered.status!=llama_memory_phase_owner_status::unchanged) return false;
            admission=owner->admit();if (!admission) return false;
            mutating=true;const bool success=mutation();
            const bool closed=finish(success);mutating=false;return closed;
        } catch (...) {mutating=false;failed=true;if (admission) finish(false);return false;}
    }
    bool finish(bool success) noexcept {
        if (!admission) return false;
        try {
            context.synchronize();
            if (read_release && !read_release(success)) {failed=true;return false;}
            if (!success) {
                failed=true;
                if (auto * model=context.kv_resources.full ? context.kv_resources.full->get_group_model() : nullptr) model->abort();
            }
            std::vector<llama_memory_phase_cache_identity> sources;
            if (success) {
                if (!table->projection_current(*projection)) {failed=true;success=false;}
                else if (!identities(sources)) {failed=true;success=false;}
            }
            if (!success) sources=owner->source_identities();
            const bool closed=owner->finish(admission,sources);
            if (closed) admission=0;
            if (!closed) failed=true;
            if (closed && success && !mutating && shared_block && context.model.arch==LLM_ARCH_DFLASH &&
                    phase==llama_memory_work_phase::draft_feature_injection) {
                draft_sources.swap(pending_draft_sources);draft_positions.swap(pending_draft_positions);
            }
            if (closed && success && !mutating && shared_block && context.model.arch==LLM_ARCH_QWEN35 && peer) {
                auto next=peer->draft_sources;
                for (const auto & append : source_appends) {
                    std::shared_ptr<llama_kv_stream_logical_cache> cache;
                    if (append.source>=next.size() || !table->get(projection->handles[append.source],0,cache) || !cache->ready()) {
                        failed=true;peer->failed=true;return false;
                    }
                    const auto after=cache->checkpoint(),saved=next[append.source];
                    if (after.id!=append.before.id || after.tokens!=append.before.tokens+append.rows || after.generation<append.before.generation) {
                        failed=true;peer->failed=true;return false;
                    }
                    if (saved.id==append.before.id && saved.generation==append.before.generation && saved.tokens<=append.before.tokens)
                        next[append.source].generation=after.generation;
                    const bool covered=saved.id==append.before.id && saved.generation==append.before.generation && saved.tokens==append.before.tokens;
                    const bool empty=!saved.id && !saved.tokens && !append.before.tokens &&
                        peer->context.memory->seq_pos_max(llama_seq_id(projection->handles[append.source].slot))<0;
                    // Native DFlash leaves pinned image rows out of its cache.
                    if (append.pinned_image && (covered || empty)) next[append.source]=after;
                }
                peer->draft_sources.swap(next);
            }
            pending_draft_sources.clear();
            pending_draft_positions.clear();
            source_appends.clear();
            return closed && success;
        } catch (...) {failed=true;return false;}
    }

    bool remove_draft(llama_seq_id sequence,llama_pos first,llama_pos end,const std::function<bool()> & remove) {
        if (!shared_block || context.model.arch!=LLM_ARCH_DFLASH || !peer || peer->failed || peer->admission ||
                first<0 || end>=0 || draft_sources.size()!=projection->handles.size() || draft_positions.size()!=draft_sources.size()) return false;
        const auto found=std::find_if(projection->handles.begin(),projection->handles.end(),
            [&](const auto & handle) {return llama_seq_id(handle.slot)==sequence;});
        if (found==projection->handles.end()) return false;
        const auto source=size_t(found-projection->handles.begin());
        std::shared_ptr<llama_kv_stream_logical_cache> cache;
        if (!table->get(*found,0,cache) || !cache->ready()) return false;
        const auto current=cache->checkpoint(),saved=draft_sources[source];
        const bool unchanged=saved.id==current.id && saved.generation==current.generation;
        const auto suffix=source<draft_suffixes.size() ? draft_suffixes[source] : llama_kv_stream_logical_checkpoint{};
        const bool rollback=suffix.id==current.id && suffix.generation==current.generation && suffix.tokens==current.tokens;
        std::shared_ptr<llama_kv_cells> cells;llama_pos next=0;
        if (!table->get_cells(*found,0,cells) || !group_next_position(*cells,current.tokens,next)) return false;
        if (!saved.id || (!unchanged && !rollback) || (rollback ? first!=next : first<=draft_positions[source])) return false;
        const auto & positions=context.kv_resources.bounded->get_cells(0).seq_pos_get(sequence);
        const auto cut=positions.lower_bound({first,0});
        const auto expected=cut==positions.begin() ? llama_pos(-1) : std::prev(cut)->first;
        if (!mutate(remove)) return false;
        const auto actual=context.memory->seq_pos_max(sequence);
        if (actual!=expected) {failed=true;return false;}
        if (rollback) {
            draft_positions[source]=actual;
            draft_sources[source]=current;draft_suffixes[source]={};
        }
        return true;
    }
};

llama_context::llama_context(const llama_model & model, llama_context_params params) :
    llama_context(model, params, 0) {}

llama_context::llama_context(const llama_model & model, llama_context_params params,
        uint32_t serial_prediction_rows) : llama_context(model, params, serial_prediction_rows, false) {}

llama_context::llama_context(const llama_model & model, llama_context_params params, llama_context_mode mode) :
    llama_context(model, params, 0, false, mode) {}

llama_context::llama_context(const llama_model & model, llama_context_params params,
        uint32_t serial_prediction_rows, bool serial_image_inputs) :
    llama_context(model, params, serial_prediction_rows, serial_image_inputs, llama_context_mode::execution) {}

llama_context::llama_context(const llama_model & model,llama_context_params params,
        const std::vector<llama_seq_id> & selected) :
    llama_context(model,params,0,false,llama_context_mode::group_initialization,selected) {}

llama_context::llama_context(const llama_model & model,llama_context_params params,
        const std::vector<llama_seq_id> & selected,uint32_t prediction_rows) :
    llama_context(model,params,prediction_rows,false,llama_context_mode::group_initialization,selected) {}

bool llama_context::group_submission_allowed() const noexcept {
    return group_execution && !group_initializing && !group_execution->failed && group_execution->admission!=0;
}

bool llama_context::group_mtp_ready(uint32_t future_tokens) const noexcept {
    if (group_initializing || !group_execution || !group_execution->shared_roles || group_execution->failed ||
            group_execution->admission || !group_execution->owner || !group_execution->peer) return false;
    const bool readonly=group_execution->shared_readonly;
    if (readonly ? !cparams.kv_stream_verify_max || future_tokens>=cparams.kv_stream_verify_max :
            cparams.kv_stream_auxiliary_layers!=1 || future_tokens>cparams.n_rs_seq) return false;
    try {
        const auto & peer=*group_execution->peer;
        if (readonly && (!peer.read_prepare || !peer.read_release ||
                std::any_of(group_execution->projection->sources.begin(),group_execution->projection->sources.end(),
                    [](const auto & source) {return source.tokens==0;}))) return false;
        if (peer.failed || peer.admission || peer.owner!=group_execution->owner || peer.context.group_target!=this ||
                !group_execution->table->projection_current(*group_execution->projection) ||
                !peer.table->projection_current(*peer.projection)) return false;
        std::vector<llama_memory_phase_cache_identity> sources;
        if (!group_execution->identities(sources)) return false;
        const auto & expected=group_execution->owner->source_identities();
        if (sources.size()!=expected.size()) return false;
        for (size_t i=0;i<sources.size();++i)
            if (sources[i].id!=expected[i].id || sources[i].generation!=expected[i].generation) return false;
        return true;
    } catch (...) {return false;}
}

bool llama_mtp_rebind_compatible(const llama_context * old_target,const llama_context * old_draft,
        const llama_context * target,const llama_context * draft,llama_sampler * const * samplers,size_t count) {
    return draft && draft->cparams.ctx_type==LLAMA_CONTEXT_TYPE_MTP &&
        llama_speculative_rebind_compatible(old_target,old_draft,target,draft,samplers,count);
}

bool llama_speculative_rebind_compatible(const llama_context * old_target,const llama_context * old_draft,
        const llama_context * target,const llama_context * draft,llama_sampler * const * samplers,size_t count) {
    if (!old_target || !old_draft || !target || !draft || !samplers || !count || count>LLAMA_MAX_SEQ ||
            target==draft || old_target==old_draft || target->cparams.ctx_type!=LLAMA_CONTEXT_TYPE_DEFAULT ||
            (draft->model.arch==LLM_ARCH_DFLASH ? draft->cparams.ctx_type!=LLAMA_CONTEXT_TYPE_DEFAULT ||
                target->model.arch!=LLM_ARCH_QWEN35 || !draft->cparams.serial_block_rows :
                draft->cparams.ctx_type!=LLAMA_CONTEXT_TYPE_MTP) ||
            (draft->group_target && draft->group_target!=target) ||
            (draft->mtp_target_ctx && draft->mtp_target_ctx!=target) ||
            (draft->cparams.ctx_other && draft->cparams.ctx_other!=target) ||
            target->group_initializing!=draft->group_initializing || target->group_selected!=draft->group_selected) return false;
    const auto retirement_source=target->group_source_table.lock();
    const bool retiring_group=(draft->model.arch==LLM_ARCH_DFLASH || draft->cparams.ctx_type==LLAMA_CONTEXT_TYPE_MTP) && target->group_storage_pending() &&
        draft->group_storage_pending() && !target->group_execution && !draft->group_execution &&
        retirement_source && old_target->group_execution && old_draft->group_execution &&
        retirement_source==old_target->group_execution->table && retirement_source==old_draft->group_execution->table;
    const std::pair<const llama_context *,const llama_context *> pairs[]={{old_target,target},{old_draft,draft}};
    for (const auto & pair : pairs) {
        const auto & old=*pair.first;const auto & next=*pair.second;const auto & a=old.cparams;const auto & b=next.cparams;
        const auto * old_cache=old.model.arch==LLM_ARCH_DFLASH ? old.kv_resources.bounded : old.kv_resources.full;
        const auto * next_cache=next.model.arch==LLM_ARCH_DFLASH ? next.kv_resources.bounded : next.kv_resources.full;
        if (&old.model!=&next.model || a.ctx_type!=b.ctx_type || a.n_seq_max!=count || b.n_seq_max!=count ||
                a.n_ctx!=b.n_ctx || a.n_batch!=b.n_batch || a.n_ubatch!=b.n_ubatch || a.n_rs_seq!=b.n_rs_seq ||
                a.n_outputs_max!=b.n_outputs_max || a.n_outputs_max_per_seq!=b.n_outputs_max_per_seq ||
                a.kv_unified!=b.kv_unified || a.flash_attn!=b.flash_attn || a.offload_kqv!=b.offload_kqv ||
                a.causal_attn!=b.causal_attn || a.embeddings!=b.embeddings || a.serial_block_rows!=b.serial_block_rows ||
                a.embeddings_nextn!=b.embeddings_nextn || a.embeddings_nextn_masked!=b.embeddings_nextn_masked ||
                !old_cache || !next_cache || old_cache->type_k()!=next_cache->type_k() ||
                old_cache->type_v()!=next_cache->type_v() || old.group_restore_failed || next.group_restore_failed ||
                (old.group_execution && ((old.group_execution->failed && !retiring_group) || old.group_execution->admission)) ||
                (next.group_execution && (next.group_execution->failed || next.group_execution->admission))) return false;
    }
    if (draft->model.arch==LLM_ARCH_DFLASH) {
        if (old_target->group_source_table.lock()!=target->group_source_table.lock()) return false;
    }
    if (bool(target->group_execution)!=bool(draft->group_execution) ||
            (target->group_execution && target->group_execution->owner!=draft->group_execution->owner)) return false;
    for (size_t slot=0;slot<count;++slot) {
        if (!draft->group_selected.empty() && std::find(draft->group_selected.begin(),draft->group_selected.end(),
                llama_seq_id(slot))==draft->group_selected.end()) continue;
        const auto found=draft->sampling.samplers.find(llama_seq_id(slot));
        if ((found==draft->sampling.samplers.end() ? nullptr : found->second)!=samplers[slot]) return false;
    }
    return true;
}

bool llama_context::attach_group_execution(std::unique_ptr<llama_context_memory> & workspace,
        std::shared_ptr<llama_kv_stream_slot_projection> projection,std::unique_ptr<llama_memory_phase_owner> & owner,
        llama_memory_work_phase phase,uint32_t profile,uint64_t selection_revision) {
    return owner && attach_group_execution_impl(workspace,std::move(projection),*owner,phase,profile,selection_revision,false,
        [&owner] {return std::shared_ptr<llama_memory_phase_owner>(std::move(owner));});
}

bool llama_context::attach_group_execution_shared(std::unique_ptr<llama_context_memory> & workspace,
        std::shared_ptr<llama_kv_stream_slot_projection> projection,std::shared_ptr<llama_memory_phase_owner> owner,
        llama_memory_work_phase phase,uint32_t profile,uint64_t selection_revision) {
    return owner && attach_group_execution_impl(workspace,std::move(projection),*owner,phase,profile,selection_revision,true,
        [owner] {return owner;});
}

bool llama_context::attach_group_execution_impl(std::unique_ptr<llama_context_memory> & workspace,
        std::shared_ptr<llama_kv_stream_slot_projection> projection,llama_memory_phase_owner & owner,
        llama_memory_work_phase phase,uint32_t profile,uint64_t selection_revision,bool shared_roles,
        const std::function<std::shared_ptr<llama_memory_phase_owner>()> & acquire_owner) {
    auto * stream=kv_resources.full ? kv_resources.full->get_group_model() : nullptr;
    const bool block=shared_roles && (model.arch==LLM_ARCH_DFLASH ||
        (model.arch==LLM_ARCH_QWEN35 && cparams.ctx_type==LLAMA_CONTEXT_TYPE_DEFAULT && !cparams.kv_stream_auxiliary_layers));
    const bool block_child=block && model.arch==LLM_ARCH_DFLASH && group_target;
    const bool auxiliary=shared_roles && (cparams.ctx_type==LLAMA_CONTEXT_TYPE_MTP || block_child);
    const bool readonly=shared_roles && (model.arch==LLM_ARCH_GEMMA4 || model.arch==LLM_ARCH_GEMMA4_ASSISTANT);
    if (!stream && ((readonly && auxiliary) || block_child) && group_target && group_target->kv_resources.full)
        stream=group_target->kv_resources.full->get_group_model();
    const bool valid_phase=block_child ? phase==llama_memory_work_phase::draft_feature_injection || phase==llama_memory_work_phase::draft_block_prediction :
        auxiliary ? phase==llama_memory_work_phase::mtp_catchup || phase==llama_memory_work_phase::mtp_draft :
        phase==llama_memory_work_phase::target_prefill || phase==llama_memory_work_phase::target_verify;
    if (!group_initializing || group_restore_failed || measurement_only || group_execution || compute_memory || !workspace ||
            !projection || !profile || !selection_revision || !stream || !stream->bound() || kv_resources.full->full_storage_pending() ||
            (kv_resources.recurrent && kv_resources.recurrent->current_storage_pending()) ||
            (kv_resources.bounded && kv_resources.bounded->bounded_storage_pending()) ||
            !valid_phase || (shared_roles && (!readonly && !block_child && model.arch!=LLM_ARCH_QWEN35)) ||
            (shared_roles && projection->role!=(auxiliary && !readonly && !block_child ? 1u : 0u)) ||
            !workspace->external_workspace() || !workspace->external_workspace()->ready()) return false;
    try {
        auto table=stream->source_table();
        if (!table || !table->projection_current(*projection) || projection->handles.size()!=group_selected.size()) return false;
        for (auto id : group_selected) if (std::none_of(projection->handles.begin(),projection->handles.end(),
                [&](const auto & handle) {return llama_seq_id(handle.slot)==id;})) return false;
        std::vector<llama_memory_phase_cache_identity> identities;
        if (shared_roles) {
            for (uint32_t role=0;role<(readonly || block ? 1u : 2u);++role) for (const auto & handle : projection->handles) {
                std::shared_ptr<llama_kv_stream_logical_cache> cache;
                if (!table->get(handle,role,cache) || !cache->ready()) return false;
                identities.push_back({cache->identity().id,cache->identity().generation});
            }
        } else for (const auto & source : projection->sources) identities.push_back({source.id,source.generation});
        const auto & expected=owner.source_identities();
        if (expected.size()!=identities.size()) return false;
        for (size_t i=0;i<expected.size();++i) if (expected[i].id!=identities[i].id || expected[i].generation!=identities[i].generation) return false;
        if (!workspace->capture_external_workspace()) return false;
        auto binding=std::make_unique<group_execution_binding>(group_execution_binding{*this,table,std::move(projection),
            {},{},phase,profile,selection_revision});
        binding->shared_roles=shared_roles;
        binding->shared_readonly=readonly;
        binding->shared_block=block;
        if (block_child) {
            binding->draft_sources.resize(binding->projection->handles.size());
            binding->draft_positions.assign(binding->projection->handles.size(),-1);
        }
        for (auto & graph : gf_res_prev) if (graph) graph->reset();
        gf_res_prev_active=nullptr;
        std::vector<size_t> expected_bytes(backend_ptrs.size());
        for (size_t i=0;i<backend_ptrs.size();++i) for (auto * lease : workspace->workspace_leases()) {
            auto * buffer=ggml_backend_memory_lease_buffer(lease);
            if (buffer && ggml_backend_buffer_get_type(buffer)==ggml_backend_sched_get_buffer_type(sched.get(),backend_ptrs[i])) {
                const auto bytes=ggml_backend_buffer_get_size(buffer);
                if (bytes>SIZE_MAX-expected_bytes[i]) return false;
                expected_bytes[i]+=bytes;
            }
        }
        binding->owner=acquire_owner();
        group_source_table=table;
        compute_memory=std::move(workspace);group_execution=std::move(binding);
        cparams.kv_stream_prefill=true;cparams.kv_stream_mask=false;
        cparams.kv_stream_decode=phase==llama_memory_work_phase::target_verify || phase==llama_memory_work_phase::mtp_draft;
        compute_memory_graph_revision=compute_memory->graph_binding_revision();
        backend_buf_exp_size=std::move(expected_bytes);
        sched_reserve_state.complete();group_initializing=false;return true;
    } catch (...) {return false;}
}

bool llama_context::bind_group_samplers_from(const llama_context & prepared) {
    if (&prepared==this || !group_initializing || group_execution || compute_memory || group_selected.empty() ||
            !prepared.group_initializing || prepared.group_execution || &model!=&prepared.model ||
            cparams.n_outputs_max_per_seq!=prepared.cparams.n_outputs_max_per_seq) return false;
    try {
        decltype(sampling.samplers) selected;
        for (auto sequence : group_selected) {
            const auto found=prepared.sampling.samplers.find(sequence);
            if (found!=prepared.sampling.samplers.end()) selected.emplace(sequence,found->second);
        }
        sampling.samplers=std::move(selected);sched_reserve_state.invalidate();return true;
    } catch (...) {return false;}
}

bool llama_context::detach_group_execution() noexcept {
    if (!group_execution || group_execution->admission || !compute_memory || !kv_resources.full) return false;
    const bool readonly=group_execution->shared_readonly && cparams.ctx_type==LLAMA_CONTEXT_TYPE_MTP;
    const bool block=group_execution->shared_block && model.arch==LLM_ARCH_DFLASH;
    group_initializing=true;group_execution->failed=true;
    try {
        synchronize();
        if (readonly && group_execution->read_release && !group_execution->read_release(false)) return false;
        if (!group_execution->owner->close()) return false;
        if (!compute_memory->retire_graph()) return false;
        auto * model=kv_resources.full->get_group_model();
        if (!readonly && !block && (!model || !model->unbind())) return false;
        if (auto * hybrid=dynamic_cast<llama_memory_hybrid *>(memory.get())) hybrid->bind_group_mutation({});
        if (block) if (auto * iswa=dynamic_cast<llama_kv_cache_iswa *>(memory.get())) iswa->bind_group_remove({});
        group_execution->owner.reset();
        if (!compute_memory->external_workspace() || !compute_memory->external_workspace()->close()) return false;
        ggml_backend_sched_reset(sched.get());
        for (auto & graph : gf_res_prev) if (graph) graph->reset();
        if (gf_res_reserve) gf_res_reserve->reset();
        gf_res_prev_active=nullptr;
        if (kv_resources.recurrent && (!kv_resources.recurrent->park_spill_stage() || !kv_resources.recurrent->release_current_storage())) return false;
        if (!readonly && kv_resources.bounded && !kv_resources.bounded->release_bounded_storage()) return false;
        compute_memory.reset();group_execution.reset();compute_memory_graph_revision=0;
        std::fill(backend_buf_exp_size.begin(),backend_buf_exp_size.end(),0);
        sched_reserve_state.complete();return true;
    } catch (...) {return false;}
}

bool llama_context::abort_group_execution() noexcept {
    if (!group_execution) return false;
    group_execution->failed=true;
    if (group_execution->admission) group_execution->finish(false);
    return detach_group_execution();
}

bool llama_context::no_alloc() const { return measurement_only || group_initializing || model.hparams.no_alloc; }

llama_context::llama_context(
        const llama_model & model,
              llama_context_params params,
              uint32_t serial_prediction_rows, bool serial_image_inputs, llama_context_mode mode,
        const std::vector<llama_seq_id> & selected) :
    model(model),
    cvec(std::make_unique<llama_adapter_cvec>()),
    loras(std::make_unique<llama_adapter_loras>()),
    // MTP uses the embd input for the hidden state
    balloc(std::make_unique<llama_batch_allocr>(model.hparams.n_pos_per_embd(),
                llm_arch_supports_mixed_batch(model.arch) && params.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT)) {
    // TODO warning when creating llama_context with awkward ctx size that is not a power of 2,
    //     may need to be backend-dependent
    LLAMA_LOG_INFO("%s: constructing llama_context\n", __func__);

    t_start_us = model.t_start_us;
    t_load_us  = model.t_load_us;

    const auto & hparams = model.hparams;
    measurement_only=mode == llama_context_mode::measurement;
    group_initializing=mode == llama_context_mode::group_initialization;
    if (group_initializing) {
        llama_memory_params declaration{};declaration.kv_stream_selected_sequences=selected;
        const bool block=model.arch==LLM_ARCH_DFLASH && params.ctx_other;
        if ((model.arch!=LLM_ARCH_QWEN35 && model.arch!=LLM_ARCH_GEMMA4 && model.arch!=LLM_ARCH_GEMMA4_ASSISTANT && !block) || hparams.vocab_only || hparams.no_alloc ||
                params.kv_stream_pool_bytes || params.n_seq_max<2 || params.n_seq_max>LLAMA_MAX_SEQ || !params.kv_unified || (serial_prediction_rows && !block) || serial_image_inputs ||
                selected.empty() || selected.size()>params.n_seq_max || !llama_memory_selected_sequences_valid(declaration,params.n_seq_max))
            throw std::runtime_error("group initialization requires supported deferred storage and valid logical slots");
        if (block) {
            const auto & target=*params.ctx_other;
            if ((!target.group_initializing && !target.group_parent()) ||
                    !group_dflash_supported(target,model,params,serial_prediction_rows) ||
                    !std::all_of(selected.begin(),selected.end(),[&](llama_seq_id id) {
                        return std::find(target.group_selected.begin(),target.group_selected.end(),id)!=target.group_selected.end();
                    })) throw std::runtime_error("group DFlash declaration requires matched target storage and inherited draft geometry");
        } else if (params.ctx_type==LLAMA_CONTEXT_TYPE_MTP) {
            const auto * parent=params.ctx_other;
            const bool owned=parent && model.arch==LLM_ARCH_QWEN35 && &parent->get_model()==&model &&
                hparams.n_layer_nextn==1 && parent->cparams.kv_stream_auxiliary_layers==1;
            const bool readonly=parent && model.arch==LLM_ARCH_GEMMA4_ASSISTANT && parent->model.arch==LLM_ARCH_GEMMA4 &&
                !parent->cparams.kv_stream_auxiliary_layers && parent->cparams.kv_stream_verify_max &&
                hparams.n_layer_nextn && parent->model.hparams.n_layer()>=2 &&
                hparams.n_embd_inp()==parent->model.hparams.n_embd_out() && hparams.n_embd_out()==parent->model.hparams.n_embd_out() &&
                params.n_batch==std::min({parent->cparams.n_seq_max,parent->cparams.n_batch,parent->cparams.n_ubatch}) &&
                params.n_ubatch==params.n_batch && !params.swa_full && !params.embeddings &&
                (params.attention_type==LLAMA_ATTENTION_TYPE_CAUSAL ||
                    (params.attention_type==LLAMA_ATTENTION_TYPE_UNSPECIFIED && hparams.causal_attn)) &&
                parent->kv_resources.bounded && params.type_k==parent->kv_resources.bounded->type_k() &&
                params.type_v==parent->kv_resources.bounded->type_v();
            if (!parent || (!owned && !readonly) ||
                    (!parent->group_initializing && (!parent->group_execution || parent->group_execution->failed)) ||
                    parent->cparams.ctx_type!=LLAMA_CONTEXT_TYPE_DEFAULT || parent->cparams.n_seq_max!=params.n_seq_max ||
                    !parent->cparams.shared_device_memory_bytes ||
                    params.n_ctx!=parent->n_ctx() || params.shared_device_memory_bytes || params.n_rs_seq || params.kv_stream_auxiliary_layers ||
                    !parent->kv_resources.full || params.type_k!=parent->kv_resources.full->type_k() ||
                    params.type_v!=parent->kv_resources.full->type_v() || !params.offload_kqv ||
                    params.flash_attn_type!=LLAMA_FLASH_ATTN_TYPE_ENABLED || model.devices.size()!=1 ||
                    model.dev_layer(hparams.n_layer())!=model.devices.front().dev ||
                    !std::all_of(selected.begin(),selected.end(),[&](llama_seq_id id) {
                        return std::find(parent->group_selected.begin(),parent->group_selected.end(),id)!=parent->group_selected.end();
                    })) throw std::runtime_error("group MTP declaration requires the matching selected target and inherited cache geometry");
            if (readonly) for (uint32_t il=0;il<hparams.n_layer_all;++il) {
                const auto id=parent->model.hparams.n_layer()-(hparams.is_swa(il) ? 2 : 1);
                const auto & source=parent->model.hparams;
                if (model.dev_layer(il)!=parent->model.devices.front().dev || hparams.is_swa(il)!=source.is_swa(id) ||
                        hparams.n_embd_head_k(il)!=source.n_embd_head_k(id) || hparams.n_embd_head_v(il)!=source.n_embd_head_v(id) ||
                        hparams.n_head_kv(il)!=source.n_head_kv(id) || hparams.n_head(il)!=source.n_head(id))
                    throw std::runtime_error("incompatible Gemma assistant shared-KV geometry");
            }
        } else if (model.arch==LLM_ARCH_GEMMA4_ASSISTANT || params.ctx_type!=LLAMA_CONTEXT_TYPE_DEFAULT || params.ctx_other || !params.shared_device_memory_bytes ||
                !group_rollback_supported(model,params.n_rs_seq) || !group_auxiliary_supported(model,params)) {
            throw std::runtime_error("group target declaration has unsupported ownership or verification bounds");
        }
        group_selected=selected;
        if (params.ctx_type==LLAMA_CONTEXT_TYPE_MTP || block) group_target=params.ctx_other;
    } else if (!selected.empty()) throw std::runtime_error("selected IDs require group initialization");
    if (measurement_only && ((model.arch != LLM_ARCH_QWEN35 && model.arch != LLM_ARCH_GEMMA4) ||
            hparams.vocab_only || params.ctx_type != LLAMA_CONTEXT_TYPE_DEFAULT || params.ctx_other ||
            params.kv_stream_pool_bytes || params.shared_device_memory_bytes ||
            (model.arch==LLM_ARCH_QWEN35 && params.n_rs_seq>7) ||
            params.moe_cache_size || serial_prediction_rows || serial_image_inputs ||
            (std::getenv("LLAMA_RS_HOST_SPILL") && std::strcmp(std::getenv("LLAMA_RS_HOST_SPILL"),"1") == 0))) {
        throw std::runtime_error("graph measurement requires an unattached target with supported rollback geometry and no execution grants");
    }


    if (serial_image_inputs && !serial_prediction_rows) {
        throw std::runtime_error("image feature preparation requires an attached serial block drafter");
    }

    if (serial_prediction_rows && (model.arch != LLM_ARCH_DFLASH || !params.ctx_other ||
            !params.ctx_other->get_cparams().kv_streaming())) {
        throw std::runtime_error("explicit serial prediction rows require an attached streamed block drafter");
    }

    cparams.n_seq_max = std::max(1u, params.n_seq_max);
    if (cparams.n_seq_max > LLAMA_MAX_SEQ) {
        throw std::runtime_error("n_seq_max must be <= " + std::to_string(LLAMA_MAX_SEQ));
    }

    cparams.n_rs_seq = params.n_rs_seq;
    if (cparams.n_rs_seq > 0 && !llm_arch_supports_rs_rollback(model.arch)) {
        LLAMA_LOG_DEBUG("%s: n_rs_seq=%u requested but model does not support recurrent partial rollback; clamping to 0\n",
                        __func__, cparams.n_rs_seq);
        cparams.n_rs_seq = 0;
    }

    cparams.n_threads               = params.n_threads;
    cparams.n_threads_batch         = params.n_threads_batch;
    cparams.yarn_ext_factor         = params.yarn_ext_factor  >= 0.0f ? params.yarn_ext_factor  : hparams.yarn_ext_factor;
    cparams.yarn_attn_factor        = params.yarn_attn_factor >= 0.0f ? params.yarn_attn_factor : hparams.yarn_attn_factor;
    cparams.yarn_beta_fast          = params.yarn_beta_fast   >= 0.0f ? params.yarn_beta_fast   : hparams.yarn_beta_fast;
    cparams.yarn_beta_slow          = params.yarn_beta_slow   >= 0.0f ? params.yarn_beta_slow   : hparams.yarn_beta_slow;
    cparams.embeddings              = params.embeddings;
    cparams.embeddings_nextn        = false;
    cparams.embeddings_nextn_masked = false;
    cparams.offload_kqv             = params.offload_kqv;
    cparams.no_perf                 = params.no_perf;
    cparams.warmup                  = false;

    // +1: id n_layer() taps the output of the last layer ("input" of the head)
    cparams.embeddings_layer_inp.resize(hparams.n_layer() + 1, false);
    embd_layer_inp.resize(hparams.n_layer() + 1);

    cparams.ctx_type          = params.ctx_type;
    cparams.rope_scaling_type = params.rope_scaling_type;
    cparams.pooling_type      = params.pooling_type;

    cparams.n_ctx            = params.n_ctx           == 0    ? hparams.n_ctx_train           : params.n_ctx;
    cparams.rope_freq_base   = params.rope_freq_base  == 0.0f ? hparams.rope_freq_base_train  : params.rope_freq_base;
    cparams.rope_freq_scale  = params.rope_freq_scale == 0.0f ? hparams.rope_freq_scale_train : params.rope_freq_scale;

    cparams.n_ctx_orig_yarn  = params.yarn_orig_ctx    != 0 ? params.yarn_orig_ctx    :
                               hparams.n_ctx_orig_yarn != 0 ? hparams.n_ctx_orig_yarn :
                                                              hparams.n_ctx_train;

    cparams.cb_eval           = params.cb_eval;
    cparams.cb_eval_user_data = params.cb_eval_user_data;

    cparams.ctx_other = nullptr;

    // TODO: more generic
    if (model.arch == LLM_ARCH_GEMMA4_ASSISTANT) {
        if (params.ctx_other == nullptr) {
            // TODO: change from runtime_error to llama_exception to avoid printing error message
            throw std::runtime_error("Gemma4Assistant requires ctx_other to be set (this warning is normal during memory fitting)");
        }

        cparams.ctx_other = params.ctx_other;
    }

    if (model.arch == LLM_ARCH_EAGLE3 || model.arch == LLM_ARCH_DFLASH) {
        if (model.tok_embd == nullptr || model.output == nullptr) {
            if (params.ctx_other == nullptr) {
                throw std::runtime_error(model.arch_name() + " requires ctx_other to be set (this warning is normal during memory fitting)");
            }
            cparams.ctx_other = params.ctx_other;
        }
    }

    if (cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED) {
        cparams.rope_scaling_type = hparams.rope_scaling_type_train;
    }

    if (cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_NONE) {
        cparams.rope_freq_scale = 1.0f; // never scale if scaling type is none
    }

    if (cparams.yarn_ext_factor < 0.0f) { // negative indicates 'not set'
        cparams.yarn_ext_factor = cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN ? 1.0f : 0.0f;
    }

    if (cparams.yarn_ext_factor != 0) {
        static auto get_mscale = [](float scale, float mscale) {
            return scale <= 1.0f ? 1.0f : (0.1f * mscale * logf(scale) + 1.0f);
        };

        const float factor = 1.0f / cparams.rope_freq_scale;

        // ref: https://github.com/huggingface/transformers/blob/6d00f6b0a5679c36510f203e4226e36f517c3032/src/transformers/modeling_rope_utils.py#L336-L348
        if (hparams.rope_yarn_log_mul != 0.0f) {
            // note: here we assume `mscale == 1.0f`
            // TODO: start reading the actual value of mscale and handle the case where it is not 1.0f
                  float mscale          = 1.0f;
            const float mscale_all_dims = hparams.rope_yarn_log_mul;

            // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
            // special-case DEEPSEEK v2:
            // https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/blob/main/config.json#L42-L43
            if (model.arch == LLM_ARCH_DEEPSEEK2 && mscale_all_dims != 1.0f) {
                mscale = mscale_all_dims;
            }

            cparams.yarn_attn_factor = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dims);

            LLAMA_LOG_WARN("%s: setting new yarn_attn_factor = %.4f (mscale == %.1f, mscale_all_dim = %.1f)\n",
                    __func__, cparams.yarn_attn_factor, mscale, mscale_all_dims);
        } else {
            cparams.yarn_attn_factor = get_mscale(factor, 1.0f);
        }

        // when YARN is applied with yarn_ext_factor != 0.0f, we need to cancel this factor:
        // https://github.com/ggml-org/llama.cpp/blob/a81a569577cc38b32558958b048228150be63eae/ggml/src/ggml-cpu/ops.cpp#L5541-L5544
        //
        // ref: https://github.com/ggml-org/llama.cpp/discussions/7416
        //      https://github.com/ggml-org/llama.cpp/pull/17945
        cparams.yarn_attn_factor *= 1.0f / (1.0f + 0.1f * logf(factor));
    }

    cparams.yarn_attn_factor *= hparams.rope_attn_factor;

    if (cparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
        if (hparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
            cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
        } else {
            cparams.pooling_type = hparams.pooling_type;
        }
    }

    if (params.attention_type == LLAMA_ATTENTION_TYPE_UNSPECIFIED) {
        cparams.causal_attn = hparams.causal_attn;
    } else {
        cparams.causal_attn = params.attention_type == LLAMA_ATTENTION_TYPE_CAUSAL;
    }

    cparams.flash_attn = params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.auto_fa    = params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO;

    cparams.fused_gdn_ar = true;
    cparams.fused_gdn_ch = true;
    cparams.auto_fgdn    = false;

    cparams.fused_lid = true;
    cparams.auto_flid = false;

    cparams.fused_dsv4_hc_pre  = true;
    cparams.fused_dsv4_hc_comb = true;
    cparams.fused_dsv4_hc_post = true;
    cparams.auto_fhc           = true;

    // with causal attention, the batch size is limited by the context size
    cparams.n_batch = cparams.causal_attn ? std::min(cparams.n_ctx, params.n_batch) : params.n_batch;

    cparams.n_ubatch = std::min(cparams.n_batch, params.n_ubatch == 0 ? params.n_batch : params.n_ubatch);
    cparams.kv_stream_pool_bytes = params.kv_stream_pool_bytes;
    cparams.shared_device_memory_bytes = params.shared_device_memory_bytes;
    cparams.kv_stream_auxiliary_layers = params.kv_stream_auxiliary_layers;
    cparams.kv_stream_verify_max = params.kv_stream_verify_max;
    cparams.recurrent_rollback_ram = params.recurrent_rollback_ram;
    if (cparams.kv_stream_verify_max && (!cparams.kv_streaming() || model.arch != LLM_ARCH_GEMMA4 ||
            params.ctx_type != LLAMA_CONTEXT_TYPE_DEFAULT || params.ctx_other || (!group_initializing && cparams.n_seq_max != 1) ||
            cparams.kv_stream_verify_max > 4 || cparams.kv_stream_verify_max > cparams.n_ubatch)) {
        throw std::runtime_error("explicit streamed verification requires a serial Gemma4 target and 1-4 queries within n_ubatch");
    }
    if (cparams.kv_stream_auxiliary_layers > 1) {
        throw std::runtime_error("model requests " + std::to_string(cparams.kv_stream_auxiliary_layers) +
            " MTP KV layers; adaptive KV currently supports only a single Qwen MTP layer");
    }
    if (cparams.kv_stream_auxiliary_layers && !cparams.kv_streaming()) {
        throw std::runtime_error("auxiliary KV cache requires an enabled KV streaming pool or arena");
    }
    llama_recurrent_rollback_plan rollback_plan;
    if (cparams.kv_streaming() && !llama_recurrent_rollback_plan_make(
            cparams.n_rs_seq,cparams.kv_stream_auxiliary_layers != 0,cparams.n_ubatch,rollback_plan,cparams.recurrent_rollback_ram)) {
        throw std::runtime_error("KV-stream target rollback requires depth 0-7 (embedded MTP: 0-3) and ubatch >= depth+1");
    }
    const bool streamed_rs_rollback=cparams.kv_streaming() && rollback_plan.host_spill;
    if (cparams.kv_stream_pool_bytes && cparams.shared_device_memory_bytes) {
        throw std::runtime_error("kv_stream_pool_bytes and shared_device_memory_bytes are mutually exclusive");
    }
    if (params.kv_stream_pool_bytes || params.shared_device_memory_bytes) {
        if ((model.arch != LLM_ARCH_QWEN35 && model.arch != LLM_ARCH_GEMMA4) || params.ctx_type != LLAMA_CONTEXT_TYPE_DEFAULT || params.ctx_other ||
                hparams.no_alloc || hparams.vocab_only || model.devices.size() != 1 || (!group_initializing && cparams.n_seq_max != 1) ||
                !cparams.offload_kqv || params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED ||
                !cparams.causal_attn || params.embeddings || model.devices.front().is_meta || !llama_kv_stream_device_buffer_type(model.devices.front().dev)) {
            throw std::runtime_error("KV streaming requires one CUDA device, serial Qwen35 or Gemma4 target context, -fa on, supported rollback mode, no embedding mode and allocated weights (--fit off)");
        }
        for (uint32_t il = 0; il < hparams.n_layer(); ++il) if (model.dev_layer(il) != model.devices.front().dev) {
            throw std::runtime_error("initial KV streaming integration requires all model layers on the single CUDA device");
        }
    }
    // The target owns the physical pool. The attached draft writes directly to its
    // separate authoritative host cache and borrows target-owned execution workspace.
    if (!group_initializing && model.arch == LLM_ARCH_GEMMA4_ASSISTANT && params.ctx_other && params.ctx_other->get_cparams().kv_streaming()) {
        auto & target=*params.ctx_other;
        const auto & source=target.get_model();
        if (source.arch != LLM_ARCH_GEMMA4 || cparams.ctx_type != LLAMA_CONTEXT_TYPE_MTP ||
                cparams.n_ctx != target.n_ctx() || cparams.n_seq_max != 1 || cparams.n_rs_seq ||
                cparams.n_batch != 1 || cparams.n_ubatch != 1 || !cparams.causal_attn || params.embeddings ||
                !cparams.offload_kqv || params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED || params.swa_full ||
                model.devices.size() != 1 || source.devices.size() != 1 ||
                model.devices.front().dev != source.devices.front().dev || hparams.no_alloc || !hparams.n_layer_nextn ||
                source.hparams.n_layer() < 2 || hparams.n_embd_inp() != source.hparams.n_embd_out() ||
                hparams.n_embd_out() != source.hparams.n_embd_out() || !target.compute_memory ||
                !target.compute_memory->shares_kv_memory() || !target.kv_resources.full || !target.kv_resources.bounded ||
                !target.kv_resources.full->get_kv_stream()) {
            throw std::runtime_error("shared Gemma KV reads require a matching serial TG1 MTP context and target phase arena");
        }
        for (uint32_t il=0;il<hparams.n_layer_all;++il) {
            const auto id=source.hparams.n_layer()-(hparams.is_swa(il) ? 2 : 1);
            if (model.dev_layer(il) != source.devices.front().dev || hparams.is_swa(il) != source.hparams.is_swa(id) ||
                    hparams.n_embd_head_k(il) != source.hparams.n_embd_head_k(id) ||
                    hparams.n_embd_head_v(il) != source.hparams.n_embd_head_v(id) ||
                    hparams.n_head_kv(il) != source.hparams.n_head_kv(id) || hparams.n_head(il) != source.hparams.n_head(id)) {
                throw std::runtime_error("incompatible Gemma assistant shared-KV geometry");
            }
        }
        if (params.type_k != target.kv_resources.full->type_k() || params.type_v != target.kv_resources.full->type_v() ||
                params.type_k != target.kv_resources.bounded->type_k() || params.type_v != target.kv_resources.bounded->type_v()) {
            throw std::runtime_error("shared KV source has incompatible cache geometry or format");
        }
        // Reserve retained target outputs before a serial child can capture arena addresses.
        target.set_embeddings_nextn(true,false);
        target.sched_reserve();
        serial_target_ctx=&target;
        cparams.kv_shared_read=true;
        cparams.embeddings_nextn=true;
        cparams.embeddings_nextn_masked=true;
    }
    std::shared_ptr<llama_kv_stream_logical_cache> mtp_attached;
    llama_kv_stream_model * mtp_target_stream = nullptr;
    if (!group_initializing && params.ctx_type == LLAMA_CONTEXT_TYPE_MTP && params.ctx_other &&
            params.ctx_other->get_cparams().kv_stream_auxiliary_layers) {
        const auto & target_ctx = *params.ctx_other;
        const auto & target_params = target_ctx.get_cparams();
        if (model.arch != LLM_ARCH_QWEN35 || target_ctx.get_model().arch != LLM_ARCH_QWEN35 ||
                target_params.ctx_type != LLAMA_CONTEXT_TYPE_DEFAULT || &model != &target_ctx.get_model() ||
                hparams.n_layer_nextn != 1 || cparams.n_ctx != target_ctx.n_ctx() ||
                cparams.n_seq_max != 1 || cparams.n_rs_seq || !cparams.offload_kqv ||
                params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED ||
                cparams.kv_streaming() || cparams.kv_stream_auxiliary_layers ||
                model.devices.size() != 1 || target_ctx.get_model().devices.size() != 1 ||
                model.devices.front().dev != target_ctx.get_model().devices.front().dev) {
            throw std::runtime_error("incompatible MTP context for target's auxiliary KV cache");
        }
        auto * target_hybrid = dynamic_cast<llama_memory_hybrid *>(llama_get_memory(params.ctx_other));
        auto * target_stream = target_hybrid ? target_hybrid->get_mem_attn()->get_kv_stream() : nullptr;
        mtp_attached = target_stream ? target_stream->auxiliary_cache() : nullptr;
        mtp_target_stream = target_stream;
        if (!mtp_attached || mtp_attached->tokens() != 0 ||
                mtp_attached->host()->config().shape.type_k != params.type_k ||
                mtp_attached->host()->config().shape.type_v != params.type_v) {
            throw std::runtime_error("missing or incompatible target MTP logical cache");
        }
        mtp_target_ctx = params.ctx_other;
    }
    cparams.mtp_publish_host = bool(mtp_attached);

    if (group_initializing && model.arch==LLM_ARCH_DFLASH && group_target) {
        cparams.serial_block_rows=serial_prediction_rows;
        cparams.embeddings_nextn=true;cparams.embeddings_nextn_masked=false;
    }
    if (!group_initializing && model.arch == LLM_ARCH_DFLASH && params.ctx_other && params.ctx_other->get_cparams().kv_streaming()) {
        auto * target=params.ctx_other;
        const uint32_t target_rows=target->get_cparams().kv_stream_decode_rows();
        const uint32_t block_rows=serial_prediction_rows ? serial_prediction_rows : target_rows;
        if (target->get_model().arch != LLM_ARCH_QWEN35 || target->get_cparams().kv_stream_auxiliary_layers ||
                !target->get_model().hparams.n_embd ||
                !target->compute_memory || params.swa_full || !hparams.n_swa || !hparams.dflash_selector_rank ||
                hparams.dflash_block_size < 2 || hparams.dflash_block_size > 8 ||
                target_rows < 2 || target_rows > 8 || block_rows < 2 || block_rows > target_rows ||
                block_rows > hparams.dflash_block_size || block_rows > cparams.n_ubatch ||
                cparams.n_seq_max != 1 || cparams.n_rs_seq ||
                cparams.n_ctx != target->n_ctx() || !cparams.offload_kqv || !cparams.flash_attn ||
                model.devices.size() != 1 || model.devices.front().dev != target->get_model().devices.front().dev ||
                model.target_layer_ids.empty() || model.target_layer_ids.size() > INT32_MAX/target->get_model().hparams.n_embd ||
                hparams.n_embd_inp_enc() != model.target_layer_ids.size()*target->get_model().hparams.n_embd) {
            throw std::runtime_error("incompatible serial DFlash2 context for adaptive target arena");
        }
        for (size_t i=0;i<model.target_layer_ids.size();++i) {
            const auto id=model.target_layer_ids[i];
            if (id < 0 || uint32_t(id) > target->get_model().hparams.n_layer() ||
                    std::find(model.target_layer_ids.begin(),model.target_layer_ids.begin()+i,id) != model.target_layer_ids.begin()+i) {
                throw std::runtime_error("invalid DFlash2 target feature mapping");
            }
        }
        // Configure every retained feature before a child can capture the target's arena addresses.
        if (target->serial_feature_images!=serial_image_inputs) {
            target->serial_feature_images=serial_image_inputs;
            target->sched_reserve_state.invalidate();
        }
        target->serial_feature_producer=true;
        target->serial_feature_layers=model.target_layer_ids;
        for (auto id : model.target_layer_ids) target->set_embeddings_layer_inp(uint32_t(id),true);
        target->sched_reserve();
        serial_target_ctx=target;
        cparams.serial_block_rows=block_rows;
        cparams.embeddings_nextn=true;
        cparams.embeddings_nextn_masked=false;
    }

    if (cparams.kv_stream_auxiliary_layers || cparams.mtp_publish_host) {
        // MTP consumes these outputs; measure their graph shape before borrowing the shared arena.
        cparams.embeddings_nextn = true;
        cparams.embeddings_nextn_masked = cparams.mtp_publish_host;
    }


    cparams.n_outputs_max = params.n_outputs_max == 0 || llama_model_has_encoder(&model) ? cparams.n_batch : params.n_outputs_max;
    cparams.n_outputs_max_per_seq = params.n_outputs_max_per_seq == 0 ?
            cparams.n_outputs_max : std::min(params.n_outputs_max_per_seq, cparams.n_outputs_max);

    // Initialize backend samplers here so they are part of the sampling graph
    // before the reserve passes run later in this function. This avoids a later
    // re-reserve when graph nodes change.
    if (params.samplers != nullptr && params.n_samplers > 0) {
        for (size_t i = 0; i < params.n_samplers; ++i) {
            const auto & config = params.samplers[i];

            if (llama_sampler_chain_get(config.sampler, -1) == nullptr) {
                throw std::runtime_error("the backend samplers must be of type llama_sampler_chain");
            }

            if (set_sampler(config.seq_id, config.sampler)) {
                const int n_samplers = llama_sampler_chain_n(config.sampler);

                LLAMA_LOG_INFO("%s: setting backend sampler for seq_id %d (n = %d)\n", __func__, config.seq_id, n_samplers);
            }
        }
    }

    cparams.op_offload     = params.op_offload;
    cparams.kv_unified     = params.kv_unified;
    cparams.moe_cache_size = params.moe_cache_size;

    // initialized later
    cparams.pipeline_parallel = false;
    cparams.training = false;

    {
        const char * LLAMA_GRAPH_REUSE_DISABLE = getenv("LLAMA_GRAPH_REUSE_DISABLE");
        graph_reuse_disable = LLAMA_GRAPH_REUSE_DISABLE ? (atoi(LLAMA_GRAPH_REUSE_DISABLE) != 0) : graph_reuse_disable;

        if (graph_reuse_disable) {
            LLAMA_LOG_WARN("%s: graph reuse disabled\n", __func__);
        }
    }

    const uint32_t logical_context = cparams.n_ctx;
    if (cparams.kv_streaming() && logical_context > uint32_t(INT32_MAX) - 255) {
        throw std::runtime_error("context capacity exceeds the padded token index range");
    }
    // ref: https://github.com/ggml-org/llama.cpp/pull/17046#discussion_r2503085732
    cparams.n_ctx = GGML_PAD(cparams.n_ctx, 256);

    if (cparams.kv_unified) {
        cparams.n_ctx_seq = cparams.n_ctx;
    } else {
        cparams.n_ctx_seq = cparams.n_ctx / cparams.n_seq_max;
        cparams.n_ctx_seq = GGML_PAD(cparams.n_ctx_seq, 256);

        if (cparams.n_ctx_seq == 0) {
            throw std::runtime_error("n_ctx_seq == 0");
        }

        if (cparams.n_ctx != cparams.n_ctx_seq * cparams.n_seq_max) {
            cparams.n_ctx =  cparams.n_ctx_seq * cparams.n_seq_max;
            LLAMA_LOG_WARN("%s: n_ctx is not divisible by n_seq_max - rounding down to %u\n", __func__, cparams.n_ctx);
        }
    }

    if (!cparams.set_stream_limits(logical_context)) {
        throw std::runtime_error("streamed context limits require indexed capacity, a native sequence count, and a shared unified arena for multiple slots");
    }

    LLAMA_LOG_INFO("%s: n_seq_max             = %u\n",   __func__, cparams.n_seq_max);
    LLAMA_LOG_INFO("%s: n_ctx                 = %u\n",   __func__, cparams.n_ctx);
    LLAMA_LOG_INFO("%s: n_ctx_seq             = %u\n",   __func__, cparams.n_ctx_seq);
    LLAMA_LOG_INFO("%s: n_batch               = %u\n",   __func__, cparams.n_batch);
    LLAMA_LOG_INFO("%s: n_ubatch              = %u\n",   __func__, cparams.n_ubatch);
    LLAMA_LOG_INFO("%s: causal_attn           = %d\n",   __func__, cparams.causal_attn);
    LLAMA_LOG_INFO("%s: flash_attn            = %s\n",   __func__, llama_flash_attn_type_name(params.flash_attn_type));
    LLAMA_LOG_INFO("%s: kv_unified            = %s\n",   __func__, cparams.kv_unified ? "true" : "false");
    LLAMA_LOG_INFO("%s: freq_base             = %.1f\n", __func__, cparams.rope_freq_base);
    LLAMA_LOG_INFO("%s: freq_scale            = %g\n",   __func__, cparams.rope_freq_scale);
    LLAMA_LOG_INFO("%s: n_rs_seq              = %u\n",   __func__, cparams.n_rs_seq);
    LLAMA_LOG_INFO("%s: n_outputs_max         = %u\n",   __func__, cparams.n_outputs_max);
    LLAMA_LOG_INFO("%s: n_outputs_max_per_seq = %u\n",   __func__, cparams.n_outputs_max_per_seq);

    if (cparams.n_ctx_seq < hparams.n_ctx_train) {
        LLAMA_LOG_INFO("%s: n_ctx_seq (%u) < n_ctx_train (%u) -- the full capacity of the model will not be utilized\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (cparams.n_ctx_seq > hparams.n_ctx_train) {
        LLAMA_LOG_WARN("%s: n_ctx_seq (%u) > n_ctx_train (%u) -- possible training context overflow\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (!hparams.vocab_only) {
        // GPU backends
        for (const auto & dev : model.devices) {
            ggml_backend_t backend = ggml_backend_dev_init(dev.dev, nullptr);
            if (backend == nullptr) {
                throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev.dev)));
            }
            backends.emplace_back(backend);
        }

        // add ACCEL backends (such as BLAS)
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
                ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
                if (backend == nullptr) {
                    throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev)));
                }
                backends.emplace_back(backend);
            }
        }

        // add CPU backend
        backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (backend_cpu == nullptr) {
            throw std::runtime_error("failed to initialize CPU backend");
        }
        backends.emplace_back(backend_cpu);

        // create a list of the set_n_threads functions in the backends
        for (auto & backend : backends) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
            ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
            if (reg) {
                auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
                if (ggml_backend_set_n_threads_fn) {
                    set_n_threads_fns.emplace_back(backend.get(), ggml_backend_set_n_threads_fn);
                }
            }
        }

        llama_set_abort_callback(this, params.abort_callback, params.abort_callback_data);

        // graph outputs buffer
        if (!measurement_only && !group_initializing) {
            const auto initial_outputs=std::min(cparams.n_seq_max,cparams.n_outputs_max);
            if (output_reserve(initial_outputs) < initial_outputs) {
                throw std::runtime_error("failed to reserve initial output buffer");
            }

            LLAMA_LOG_INFO("%s: %10s  output buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buffer_name    (buf_output.get()),
                    ggml_backend_buffer_get_size(buf_output.get()) / 1024.0 / 1024.0);
        }
    }

    // init the memory module
    if (!hparams.vocab_only) {
        llama_memory_params params_mem = {
            /*.type_k    =*/ params.type_k,
            /*.type_v    =*/ params.type_v,
            /*.swa_full  =*/ params.swa_full,
            /*.ctx_type  =*/ cparams.ctx_type,
            /*.mem_other =*/ llama_get_memory(cparams.ctx_other),
        };
        params_mem.measure_only=measurement_only;
        params_mem.defer_full_cache=group_initializing;
        if (group_initializing && group_target &&
                ((params.ctx_type==LLAMA_CONTEXT_TYPE_MTP && model.arch==LLM_ARCH_QWEN35) || model.arch==LLM_ARCH_DFLASH)) {
            params_mem.mem_other=nullptr;
            params_mem.shared_device_memory_bytes=params.ctx_other->get_cparams().shared_device_memory_bytes;
            if (model.arch==LLM_ARCH_DFLASH) params_mem.defer_bounded_cache=true;
        }
        params_mem.kv_stream_selected_sequences=group_selected;
        if (group_initializing && model.arch==LLM_ARCH_GEMMA4_ASSISTANT) {
            params_mem.defer_full_cache=false;
            params_mem.kv_stream_selected_sequences.clear();
        }
        params_mem.kv_stream_rs_rollback=cparams.recurrent_rollback_ram && cparams.n_rs_seq!=0;
        if (model.arch == LLM_ARCH_GEMMA4 && cparams.shared_device_memory_bytes) {
            if (params.swa_full) throw std::runtime_error("Gemma arena streaming requires bounded SWA storage (swa_full=false)");
            params_mem.defer_bounded_cache=true;
        }
        if (serial_target_ctx && cparams.serial_block_rows) {
            auto * parent=serial_target_ctx->compute_memory.get();
            auto * buft=ggml_backend_buffer_get_type(parent->shared_parent());
            llama_memory_bounded_kv_layout layout;
            if (!llama_kv_cache_iswa::bounded_storage_requirements(hparams,params_mem,cparams,buft,{2,1},layout)) {
                throw std::runtime_error("invalid bounded draft allocation quote");
            }
            if (layout.layers.size() != hparams.n_layer())
                throw std::runtime_error("serial draft arena requires one homogeneous resident SWA group");
            for (const auto & layer : layout.layers) if (model.dev_layer(layer.id) != model.devices.front().dev)
                throw std::runtime_error("serial draft arena requires one homogeneous resident SWA group");
            auto * hybrid=dynamic_cast<llama_memory_hybrid *>(serial_target_ctx->memory.get());
            auto * recurrent=hybrid ? hybrid->get_mem_recr() : nullptr;
            const auto * bank=recurrent ? recurrent->spill_bank() : nullptr;
            if (bank && bank->staged_device_bytes() && !serial_target_ctx->publication_stage) {
                auto stage=parent->reserve_staging(bank->staged_device_bytes());
                if (!stage || !recurrent->bind_spill_stage(stage->lease()))
                    throw std::runtime_error("rollback publication staging does not fit the declared serial arena");
                serial_target_ctx->publication_stage=std::move(stage);
                serial_target_ctx->sched_reserve_state.invalidate();
                serial_target_ctx->sched_reserve();
            }
            llama_context_memory_create_result cache_admission;
            if (!(serial_cache=parent->reserve_bounded_cache(layout,&cache_admission))) {
                throw std::runtime_error(std::string("bounded draft KV does not fit the declared serial arena: ")+
                    llama_context_memory_create_status_name(cache_admission.status)+"; required="+std::to_string(cache_admission.required_bytes)+
                    ", available="+std::to_string(cache_admission.available_bytes)+", additional="+std::to_string(cache_admission.additional_bytes)+
                    " bytes; phase="+std::to_string(cache_admission.stage)+", resource="+std::to_string(cache_admission.resource));
            }
            params_mem.bounded_cache_lease=serial_cache->lease();
        }
        params_mem.mtp_host_cache = mtp_attached.get();
        if (params.kv_stream_pool_bytes || params.shared_device_memory_bytes) {
            params_mem.kv_stream_pool_bytes = params.kv_stream_pool_bytes;
            params_mem.shared_device_memory_bytes = params.shared_device_memory_bytes;
            params_mem.kv_stream_auxiliary_layers = cparams.kv_stream_auxiliary_layers;
            params_mem.kv_stream_rs_rollback = streamed_rs_rollback;
            params_mem.kv_stream_max_rows = cparams.n_ubatch;
            params_mem.kv_stream_max_decode_rows = rollback_plan.workspace_rows;
            for (auto & backend : backends) if (ggml_backend_get_device(backend.get()) == model.devices.front().dev) {
                params_mem.kv_stream_backend = backend.get(); break;
            }
            if (!params_mem.kv_stream_backend) throw std::runtime_error("missing KV streaming CUDA backend");
        }

        memory.reset(model.create_memory(params_mem, cparams));
        kv_resources = llama_memory_kv_resources_get(memory.get());
        if (cparams.kv_shared_read) {
            auto * target=cparams.ctx_other;
            auto * stream=target->kv_resources.full->get_kv_stream();
            auto local=target->kv_resources.bounded->retain_bounded_storage(stream->host()->cache_id()+1,1,
                target->get_cparams().n_ctx_seq,target->get_cparams().n_ubatch);
            const auto pin=local ? local->read(local->identity()) : llama_memory_bounded_kv_read{};
            if (!kv_resources.full || !kv_resources.bounded || !pin ||
                    !kv_resources.full->attach_shared_stream_reader(*target->compute_memory,1) ||
                    !kv_resources.bounded->attach_shared_bounded_reader(pin)) {
                throw std::runtime_error("failed to bind leased Gemma assistant KV sources");
            }
        }
        if (cparams.kv_streaming()) {
            auto * stream=kv_resources.full ? kv_resources.full->get_kv_stream() : nullptr;
            cparams.kv_stream_prefill=stream && stream->bounded_prefill();
            cparams.kv_stream_mask=stream && stream->compact_prefill_mask() && kv_resources.full->supports_mask_bounds();
        }
        if (mtp_attached) {
            auto * draft_kv = dynamic_cast<llama_kv_cache *>(memory.get());
            if (!draft_kv || !draft_kv->attach_mtp_auxiliary_cache(mtp_attached, mtp_target_stream)) {
                throw std::runtime_error("failed to attach the MTP logical KV cache");
            }
        }
    }

    // init backends
    if (!hparams.vocab_only) {
        LLAMA_LOG_DEBUG("%s: enumerating backends\n", __func__);

        backend_buft.clear();
        backend_ptrs.clear();
        backend_buf_exp_size.clear();

        for (auto & backend : backends) {
            auto * buft = ggml_backend_get_default_buffer_type(backend.get());
            auto backend_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));

            if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU && !model.devices.empty()) {
                // use the host buffer of the first device CPU for faster transfer of the intermediate state
                const auto & dev = model.devices[0];
                auto * host_buft = ggml_backend_dev_host_buffer_type(dev.dev);
                if (host_buft) {
                    buft = host_buft;
                }
            }

            if ((measurement_only || group_initializing || cparams.kv_streaming() || cparams.mtp_publish_host || cparams.serial_block_rows || cparams.kv_shared_read) &&
                    !model.devices.empty() && ggml_backend_get_device(backend.get()) == model.devices.front().dev) {
                buft = llama_kv_stream_device_buffer_type(model.devices.front().dev);
                if (!buft) throw std::runtime_error("missing device-local KV/compute buffer type");
            }

            backend_buft.push_back(buft);
            backend_ptrs.push_back(backend.get());
            backend_buf_exp_size.push_back(0);
        }

        LLAMA_LOG_DEBUG("%s: backend_ptrs.size() = %zu\n", __func__, backend_ptrs.size());

        // TODO: move these checks to ggml_backend_sched
        // enabling pipeline parallelism in the scheduler increases memory usage, so it is only done when necessary
        bool pipeline_parallel =
            model.n_devices() > 1 &&
            model.n_gpu_layers() > model.hparams.n_layer_all &&
            model.split_mode() == LLAMA_SPLIT_MODE_LAYER &&
            cparams.offload_kqv &&
            !model.has_tensor_overrides() &&
            cparams.moe_cache_size == 0; // not supported by the MoE cache

        // pipeline parallelism requires support for async compute and events in all devices
        if (pipeline_parallel) {
            for (auto & backend : backends) {
                auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));
                if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
                    // ignore CPU backend
                    // TODO: should we ignore ACCEL types too?
                    continue;
                }
                auto * dev = ggml_backend_get_device(backend.get());
                ggml_backend_dev_props props;
                ggml_backend_dev_get_props(dev, &props);
                if (!props.caps.async || !props.caps.events) {
                    // device does not support async compute or events
                    pipeline_parallel = false;
                    break;
                }
            }
        }

        cparams.pipeline_parallel = pipeline_parallel;

        if (cparams.pipeline_parallel) {
            LLAMA_LOG_INFO("%s: pipeline parallelism enabled\n", __func__);
        }

        if (cparams.moe_cache_size > 0) {
            moe_cache = std::make_unique<llama_moe_cache>(model, backend_ptrs, backend_buft, cparams.moe_cache_size);
        }

        sched_reserve();

        if (!cparams.flash_attn) {
            if (ggml_is_quantized(params.type_v)) {
                throw std::runtime_error("quantized V cache was requested, but this requires Flash Attention");
            }
        }
    }

    // Initialize the full vocabulary token ids for backend samplers.
    {
        const int n_vocab = model.vocab.n_tokens();

        sampling.token_ids_full_vocab.resize(n_vocab);
        for (int i = 0; i < n_vocab; ++i) {
            sampling.token_ids_full_vocab[i] = i;
        }
    }
}

llama_context::~llama_context() {
    // wait for any pending asynchronous copies into the output buffers before they are freed
    synchronize();
    if (cparams.kv_shared_read) finish_shared_kv_read();

    // when training, ggml_opt allocates extra buffers through the scheduler, so the sizes no longer match the expectation
    if (!no_alloc() && !opt_ctx) {
        llama_context_memory_diagnostics shared;
        const bool shared_grant = compute_memory && compute_memory->diagnostics(shared);

        for (size_t i = 0; i < backend_ptrs.size(); ++i) {
            ggml_backend_t             backend = backend_ptrs[i];
            ggml_backend_buffer_type_t buft    = backend_buft[i];

            size_t size_exp = compute_memory && compute_memory->kv_device_suspended() ? 0 : backend_buf_exp_size[i];
            if (shared_grant && buft == ggml_backend_buffer_get_type(compute_memory->shared_parent())) {
                size_exp = shared.workspace_bytes;
            }
            const size_t size_act = ggml_backend_sched_get_buffer_size(sched.get(), backend);
            if (size_exp == size_act) {
                LLAMA_LOG_DEBUG("%s: %10s compute buffer size is %8.4f MiB, matches expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            } else {
                LLAMA_LOG_WARN("%s: %10s compute buffer size of %8.4f MiB, does not match expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            }
        }
    }
    ggml_opt_free(opt_ctx);

    // Release scheduler leases while the backend objects are still alive.
    release_kv_workspaces(true);
    if (group_execution && group_execution->admission) group_execution->finish(false);
    if (kv_resources.full && kv_resources.full->get_group_model()) kv_resources.full->get_group_model()->unbind();
    group_execution.reset();
    compute_memory.reset();
    if (serial_cache || publication_stage) { memory.reset(); serial_cache.reset(); publication_stage.reset(); }
    sched.reset();
    compute_arenas.clear();
}

void llama_context::resolve_fused_ops(const llama_memory_context_i * mctx, uint32_t n_seqs) {
    const char * func = __func__;
    auto resolve = [&](const llm_fused_op_probe & probe, bool & enabled) {
        if (!enabled) {
            return;
        }

        const uint32_t n_tokens_probe = probe.n_tokens_per_seq*n_seqs;

        auto * gf = graph_reserve(n_tokens_probe, n_seqs, n_tokens_probe, mctx, true);
        if (!gf) {
            throw std::runtime_error(std::string("failed to reserve graph for ") + probe.name + " check");
        }

        bool device_mismatch = false;
        for (const auto & node : get_gf_res_reserve()->get_fused_nodes()) {
            if (node.op != probe.op) {
                continue;
            }

            GGML_ASSERT(node.il >= 0);

            ggml_backend_t backend_fused = ggml_backend_sched_get_tensor_backend(sched.get(), node.tensor);
            ggml_backend_dev_t device_fused = backend_fused ? ggml_backend_get_device(backend_fused) : nullptr;

            // TODO: make this descriptor-specific; model.dev_layer() preserves the current behavior,
            // but is still wrong for cases like --no-kv-offload.
            ggml_backend_dev_t device_layer = model.dev_layer(node.il);

            if (device_fused != device_layer) {
                LLAMA_LOG_WARN("%s: layer %d is assigned to device %s but %s "
                        "is assigned to device %s (usually due to missing support)\n",
                        func, node.il,
                        device_layer ? ggml_backend_dev_name(device_layer) : "none",
                        probe.name,
                        device_fused ? ggml_backend_dev_name(device_fused) : "none");
                device_mismatch = true;
                break;
            }
        }

        if (device_mismatch) {
            enabled = false;
            LLAMA_LOG_WARN("%s: %s not supported, set to disabled\n", func, probe.name);
        } else {
            enabled = true;
            LLAMA_LOG_INFO("%s: %s enabled\n", func, probe.name);
        }
    };

    if (cparams.auto_fa) {
        resolve(llm_fused_op_flash_attn_probe, cparams.flash_attn);
        cparams.auto_fa = false;
    }

    if (cparams.auto_fgdn) {
        LLAMA_LOG_INFO("%s: resolving fused Gated Delta Net support:\n", func);
        resolve(llm_fused_op_gdn_ar_probe, cparams.fused_gdn_ar);
        resolve(llm_fused_op_gdn_ch_probe, cparams.fused_gdn_ch);
        cparams.auto_fgdn = false;
    }

    if (cparams.auto_flid) {
        LLAMA_LOG_INFO("%s: resolving fused Lightning Indexer support:\n", func);
        resolve(llm_fused_op_lid_probe, cparams.fused_lid);
        cparams.auto_flid = false;
    }

    if (cparams.auto_fhc) {
        LLAMA_LOG_INFO("%s: resolving fused DeepSeek V4 HC support:\n", func);
        resolve(llm_fused_op_dsv4_hc_pre_probe,  cparams.fused_dsv4_hc_pre);
        resolve(llm_fused_op_dsv4_hc_comb_probe, cparams.fused_dsv4_hc_comb);
        resolve(llm_fused_op_dsv4_hc_post_probe, cparams.fused_dsv4_hc_post);
        cparams.auto_fhc = false;
    }
}

static int llama_graph_n_input_tensors(ggml_cgraph * gf) {
    std::unordered_map<const ggml_tensor *, std::vector<ggml_tensor *>> users;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        if (node->flags & GGML_TENSOR_FLAG_INPUT) {
            users[node].push_back(node);
        }
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            ggml_tensor * src = node->src[j];
            if (!src) {
                break;
            }
            if (src->flags & GGML_TENSOR_FLAG_INPUT) {
                users[src].push_back(node);
            }
        }
    }

    for (const auto & [tensor, nodes] : users) {
        GGML_ASSERT(tensor->op == GGML_OP_NONE);
        for (const ggml_tensor * node : nodes) {
            LLAMA_LOG_DEBUG("%s: input tensor '%32s' [%s, ne = { %5" PRId64 ", %5" PRId64 ", %5" PRId64 ", %5" PRId64 " }] is used by node '%s' (%s)\n",
                    __func__, tensor->name, ggml_type_name(tensor->type),
                    tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
                    node->name, ggml_op_name(node->op));
        }
    }

    return (int) users.size();
}

bool llama_context::prepare_compute_arenas(
        const std::vector<size_t> & measurements, size_t n_phases) {
    compute_memory_graph_revision = 0;
    llama_compute_workspace_plan plan;
    if (!llama_compute_workspace_plan_make(backend_buft, measurements, n_phases, plan)) {
        return false;
    }
    // A sampler can change the child's graph footprint after construction. Its graphs are retired before measurement.
    if (compute_memory && cparams.kv_shared_read && compute_memory->borrows_serial_parent() &&
            !compute_memory->covers_workspace(plan)) {
        compute_memory.reset();
    }
    if (compute_memory && (serial_cache || publication_stage || compute_memory->has_bounded_cache() ||
            (cparams.kv_shared_read && compute_memory->borrows_serial_parent()))) {
        if (!compute_memory->covers_workspace(plan) && !compute_memory->grow_workspace(plan)) return false;
        compute_memory_graph_revision=compute_memory->graph_binding_revision();
        return true;
    }

    llama_kv_stream_model * stream = nullptr;
    if (cparams.kv_streaming() && memory) {
        stream = kv_resources.full ? kv_resources.full->get_kv_stream() : nullptr;
        if (!stream) return false;
    }
    if (!cparams.pipeline_parallel && cparams.n_seq_max == 1 &&
            llama_context_memory::supported(backend_ptrs)) {
        auto * source = serial_target_ctx ? serial_target_ctx : cparams.mtp_publish_host ? mtp_target_ctx : nullptr;
        auto * serial_parent = source ? source->compute_memory.get() : nullptr;
        if ((cparams.mtp_publish_host || cparams.serial_block_rows || cparams.kv_shared_read) && !serial_parent) return false;
        llama_context_memory_create_result admission;
        compute_memory = llama_context_memory::create(
            sched.get(),backend_ptrs,plan,stream,serial_parent,false,{},0,&admission);
        if (!compute_memory && serial_parent && !cparams.serial_block_rows && !cparams.kv_shared_read &&
                admission.status == llama_context_memory_create_status::insufficient_budget) {
            // Keep separate scratch if the shared budget cannot hold the draft graph and live KV scratch.
            LLAMA_LOG_WARN("%s: MTP graph exceeds shared arena quota: available=%zu required=%zu additional=%zu bytes; trying a separate workspace\n",
                __func__,admission.available_bytes,admission.required_bytes,admission.additional_bytes);
            llama_context_memory_create_result fallback;
            compute_memory = llama_context_memory::create(
                sched.get(),backend_ptrs,plan,stream,nullptr,false,{},0,&fallback);
            if (!compute_memory) {
                throw std::runtime_error("shared MTP arena quota requires " + std::to_string(admission.additional_bytes) +
                    " additional bytes (required=" + std::to_string(admission.required_bytes) +
                    ", available=" + std::to_string(admission.available_bytes) +
                    "); separate-workspace fallback: " + llama_context_memory_create_status_name(fallback.status));
            }
        } else if (!compute_memory && admission.status != llama_context_memory_create_status::invalid_configuration) {
            throw std::runtime_error(std::string("workspace admission: ") +
                llama_context_memory_create_status_name(admission.status) + "; required=" +
                std::to_string(admission.required_bytes) + ", available=" + std::to_string(admission.available_bytes) +
                ", additional=" + std::to_string(admission.additional_bytes) + " bytes");
        }
        if (compute_memory && kv_resources.bounded && kv_resources.bounded->bounded_storage_pending()) {
            auto * local=kv_resources.bounded;
            if (!stream || serial_cache || !compute_memory->shares_kv_memory())
                throw std::runtime_error("deferred local cache requires the target shared arena");
            llama_memory_params params{};
            params.type_k=local->type_k(); params.type_v=local->type_v();
            llama_memory_bounded_kv_layout layout;
            if (!llama_kv_cache_iswa::bounded_storage_requirements(model.hparams,params,cparams,
                    ggml_backend_buffer_get_type(compute_memory->shared_parent()),{stream->host()->cache_id()+1,1},layout))
                throw std::runtime_error("invalid deferred local cache allocation quote");
            llama_context_memory_create_result local_admission;
            auto loan=compute_memory->reserve_bounded_cache(layout,&local_admission);
            if (!loan) throw std::runtime_error(std::string("persistent local cache does not fit the declared shared arena: ")+
                llama_context_memory_create_status_name(local_admission.status)+"; required="+std::to_string(local_admission.required_bytes)+
                ", available="+std::to_string(local_admission.available_bytes)+", additional="+std::to_string(local_admission.additional_bytes)+
                " bytes; phase="+std::to_string(local_admission.stage)+", resource="+std::to_string(local_admission.resource));
            if (!compute_memory->retire_graph()) throw std::runtime_error("failed to retire local-cache reservation graphs");
            ggml_backend_sched_reset(sched.get());
            if (gf_res_reserve) gf_res_reserve->reset();
            if (!local->bind_bounded_storage(loan->lease()))
                throw std::runtime_error("failed to bind deferred local cache to its persistent arena region");
            serial_cache=std::move(loan);
            if (!compute_memory->prepare_serial_target())
                throw std::runtime_error("failed to recapture the bound local-cache workspace");
        }
        if (compute_memory && cparams.kv_shared_read && !compute_memory->register_shared_read_consumer())
            throw std::runtime_error("failed to register shared read-only serial consumer");
        if (compute_memory && cparams.serial_block_rows &&
                (!serial_cache || !compute_memory->register_bounded_write_consumer(serial_cache->lease())))
            throw std::runtime_error("failed to register parent-owned bounded draft writer");
        if (compute_memory) compute_memory_graph_revision = compute_memory->graph_binding_revision();
        return compute_memory != nullptr;
    }
    if (stream) return false;

    return llama_prepare_compute_arena_bindings(
        sched.get(),backend_ptrs,plan.groups,compute_arenas);
}

void llama_context::release_kv_workspaces(bool retiring) {
    if (!cparams.kv_streaming() || !memory) return;
    auto * stream = kv_resources.full ? kv_resources.full->get_kv_stream() : nullptr;
    if (!stream) return;
    if (retiring) stream->abort();
    const bool released = stream->set_workspaces({});
    if (retiring) { GGML_ASSERT(released); }
    else if (!released) throw std::runtime_error("cannot replace workspace during active KV execution");
}

void llama_context::sched_reserve() {
    if (group_execution) {
        if (sched_reserve_state.begin()) throw std::runtime_error("live group requires a new measured workspace before reconfiguration");
        return;
    }
    if (!sched_reserve_state.begin()) {
        return;
    }

    LLAMA_LOG_INFO("%s: reserving ...\n", __func__);

    synchronize();
    release_kv_workspaces(false);
    const bool keep_parent=compute_memory && (serial_cache || publication_stage || compute_memory->has_bounded_cache() ||
        (cparams.kv_shared_read && compute_memory->borrows_serial_parent()));
    if (keep_parent) {
        if (!compute_memory->retire_graph()) throw std::runtime_error("failed to retire serial arena graph");
        if ((serial_cache || cparams.kv_shared_read) && compute_memory->borrows_serial_parent()) {
            const auto phase=cparams.kv_shared_read ? llama_memory_text_phase::decode : llama_memory_text_phase::prefill;
            if (!compute_memory->prepare_serial_consumer(phase))
                throw std::runtime_error("failed to retain serial draft workspace for remeasurement");
        } else {
            if (!compute_memory->prepare_serial_target()) throw std::runtime_error("failed to drain serial arena consumer");
            const auto signal=compute_memory->signal_text_phase({llama_memory_text_phase::prefill,1,true,true,false});
            if (signal.status != llama_memory_text_phase_status::changed && signal.status != llama_memory_text_phase_status::unchanged)
                throw std::runtime_error("failed to retain serial arena for graph remeasurement");
        }
        ggml_backend_sched_reset(sched.get());
    } else {
        compute_memory.reset(); sched.reset(); compute_arenas.clear();
    }
    struct reserve_intent_restore {
        llama_cparams & params;
        bool decode=params.kv_stream_decode, span=params.mtp_span_attention;
        ~reserve_intent_restore() { params.kv_stream_decode=decode; params.mtp_span_attention=span; }
    } intent{cparams};
    cparams.kv_stream_decode=false; cparams.mtp_span_attention=false;

    const int64_t t_start_us = ggml_time_us();

    const uint32_t n_seqs = group_selected.empty() ? cparams.n_seq_max : uint32_t(group_selected.size());
    const uint32_t n_injection = std::min(cparams.n_ctx, cparams.n_ubatch);
    const uint32_t n_tokens = cparams.serial_graph_reserve_rows(false);

    const size_t max_nodes = this->graph_max_nodes(n_injection);

    LLAMA_LOG_DEBUG("%s: max_nodes = %zu\n", __func__, max_nodes);

    for (auto & res : gf_res_prev) {
        res.reset();
    }
    gf_res_reserve.reset(new llm_graph_result(max_nodes));
    gf_res_prev_active = nullptr;

    if (!keep_parent) sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, cparams.pipeline_parallel, cparams.op_offload));
    ggml_backend_sched_set_copy_callback(sched.get(), sched_copy_experts, this);

    llama_memory_context_ptr mctx;
    if (memory) {
        LLAMA_LOG_DEBUG("%s: reserving full memory module\n", __func__);
        mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory module");
        }
    }

    // avoid reserving graphs with zero outputs - assume one output per sequence
    const int n_outputs = n_seqs;

    LLAMA_LOG_DEBUG("%s: worst-case: n_tokens = %d, n_seqs = %d, n_outputs = %d\n", __func__, n_tokens, n_seqs, n_outputs);

    resolve_fused_ops(mctx.get(), n_seqs);

    // reserve worst-case graph
    int n_splits_pp        = -1;
    int n_nodes_pp         = -1;
    int n_inputs_pp        = -1;
    int n_input_tensors_pp = -1;

    int n_splits_tg        = -1;
    int n_nodes_tg         = -1;
    int n_inputs_tg        = -1;
    int n_input_tensors_tg = -1;

    const uint32_t n_outputs_pp = std::min(n_tokens, cparams.n_outputs_max);
    // [TAG_RESERVE_DIAG_DECAY] These intermediates scale with tokens per sequence squared.
    const uint32_t n_seqs_pp = model.arch == LLM_ARCH_KIMI_LINEAR || model.arch == LLM_ARCH_MINIMAX_01 ? 1 : n_seqs;

    auto prepare_arenas = [&]() {
        struct attention_intent_restore {
            llama_cparams & params;
            bool decode, mtp;
            ~attention_intent_restore() {
                params.kv_stream_decode = decode;
                params.mtp_span_attention = mtp;
            }
        } restore{cparams, cparams.kv_stream_decode, cparams.mtp_span_attention};
        std::vector<size_t> measurements(2*backend_ptrs.size());
        cparams.kv_stream_decode = false;
        cparams.mtp_span_attention = false;
        auto * gf_pp = graph_reserve(
            n_tokens, n_seqs, n_outputs_pp, mctx.get(), true, measurements.data());
        cparams.kv_stream_decode = cparams.kv_streaming();
        cparams.mtp_span_attention = cparams.mtp_publish_host;
        auto * gf_tg = graph_reserve(
            n_seqs, n_seqs, n_seqs, mctx.get(), true,
            measurements.data() + backend_ptrs.size());
        if (!gf_pp || !gf_tg) return false;
        if (n_seqs_pp != n_seqs) {
            std::vector<size_t> single(backend_ptrs.size());
            cparams.kv_stream_decode = false;
            cparams.mtp_span_attention = false;
            if (!graph_reserve(n_tokens, n_seqs_pp, n_outputs_pp, mctx.get(), true, single.data())) return false;
            for (size_t i = 0; i < backend_ptrs.size(); ++i) {
                measurements[i] = std::max(measurements[i], single[i]);
            }
            cparams.kv_stream_decode = cparams.kv_streaming();
            cparams.mtp_span_attention = cparams.mtp_publish_host;
        }
        if (serial_feature_images) {
            const bool decode_intent=cparams.kv_stream_decode;
            std::vector<uint32_t> measured;
            for (uint32_t width : {1u,64u,128u,129u,n_injection ? n_injection-1 : 0,n_injection}) {
                if (!width || width>n_injection || std::find(measured.begin(),measured.end(),width)!=measured.end()) continue;
                measured.push_back(width);
                for (uint32_t outputs : {1u,std::max(1u,std::min(width,cparams.n_outputs_max))}) {
                    std::vector<size_t> image(backend_ptrs.size());
                    cparams.kv_stream_decode=false;
                    if (!graph_reserve(width,1,outputs,mctx.get(),true,image.data(),true)) return false;
                    for (size_t i=0;i<backend_ptrs.size();++i) measurements[i]=std::max(measurements[i],image[i]);
                }
            }
            cparams.kv_stream_decode=decode_intent;
            LLAMA_LOG_INFO("%s: measured %zu image input widths before serial draft registration\n",__func__,measured.size());
        }
        if (model.arch == LLM_ARCH_DFLASH && model.hparams.n_embd_inp_enc() > 0) {
            // Injection and block prediction use different graph inputs but the same workspace.
            std::vector<size_t> injection(backend_ptrs.size());
            if (!graph_reserve(n_injection, n_seqs, std::min(n_injection,cparams.n_outputs_max), mctx.get(), true, injection.data(), true)) return false;
            for (size_t i = 0; i < backend_ptrs.size(); ++i) {
                measurements[i] = std::max(measurements[i], injection[i]);
                measurements[backend_ptrs.size()+i] = std::max(measurements[backend_ptrs.size()+i], injection[i]);
            }
        }
        if (cparams.serial_block_rows) {
            for (uint32_t width=1;width<=cparams.serial_block_rows;++width) {
                std::vector<size_t> block(backend_ptrs.size());
                auto * graph=graph_reserve(width,1,std::min(width,cparams.n_outputs_max),mctx.get(),true,block.data());
                if (!graph) return false;
                for (int node=0;node<ggml_graph_n_nodes(graph);++node) {
                    auto * op=ggml_graph_node(graph,node);
                    if (op->op != GGML_OP_FLASH_ATTN_EXT) continue;
                    bool supported=false;
                    for (auto * backend : backend_ptrs) if (ggml_backend_get_device(backend) == model.devices.front().dev)
                        supported=ggml_backend_supports_op(backend,op);
                    if (!supported) throw std::runtime_error("no compiled device-valid native attention for serial draft block");
                }
                for (size_t i=0;i<backend_ptrs.size();++i) {
                    measurements[i]=std::max(measurements[i],block[i]);
                    measurements[backend_ptrs.size()+i]=std::max(measurements[backend_ptrs.size()+i],block[i]);
                }
            }
        }
        const char * spill_flag=std::getenv("LLAMA_RS_HOST_SPILL");
        if ((cparams.kv_stream_verify_max || (cparams.n_rs_seq &&
                (cparams.kv_streaming() || (spill_flag && std::strcmp(spill_flag,"1") == 0)))) && n_seqs == 1) {
            const uint32_t maximum=std::min(n_tokens,cparams.kv_stream_verify_max ?
                cparams.kv_stream_verify_max : cparams.n_rs_seq+1);
            for (uint32_t width=2;width<=maximum;++width) {
                std::vector<size_t> verify(backend_ptrs.size());
                std::vector<size_t> verify_all(backend_ptrs.size());
                if (!graph_reserve(width,1,1,mctx.get(),true,verify.data()) ||
                        !graph_reserve(width,1,std::min(width,cparams.n_outputs_max),
                            mctx.get(),true,verify_all.data())) return false;
                for (size_t i=0;i<backend_ptrs.size();++i) {
                    const size_t required=std::max(verify[i],verify_all[i]);
                    measurements[i]=std::max(measurements[i],required);
                    measurements[backend_ptrs.size()+i]=std::max(measurements[backend_ptrs.size()+i],required);
                }
            }
        }
        // Target verification and attached MTP catch-up both execute TG1-TG4.
        // Include these graphs in their respective decode-phase grants.
        if ((cparams.kv_stream_auxiliary_layers || cparams.mtp_publish_host) && n_seqs == 1) {
            const uint32_t width = std::min(n_tokens, 4u);
            if (width > 1) {
                std::vector<size_t> verify_one(backend_ptrs.size());
                std::vector<size_t> verify_all(backend_ptrs.size());
                if (!graph_reserve(width, 1, 1, mctx.get(), true, verify_one.data()) ||
                        !graph_reserve(width, 1, std::min(width, cparams.n_outputs_max),
                            mctx.get(), true, verify_all.data())) return false;
                for (size_t i = 0; i < backend_ptrs.size(); ++i)
                    measurements[backend_ptrs.size() + i] = std::max(
                        measurements[backend_ptrs.size() + i],
                        std::max(verify_one[i], verify_all[i]));
            }
        }
        return prepare_compute_arenas(measurements, 2);
    };

    if (!no_alloc() && !prepare_arenas()) {
        throw std::runtime_error("failed to prepare compute arenas");
    }
    llama_log_memory_phase(compute_memory.get());

    // reserve pp (prompt processing) graph first so that buffers are only allocated once
    {
        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(),
                no_alloc(), no_alloc() ? backend_buf_exp_size.data() : nullptr);
        if (!gf) {
            if (cparams.pipeline_parallel) {
                LLAMA_LOG_WARN("%s: compute buffer allocation failed, retrying without pipeline parallelism\n", __func__);
                cparams.pipeline_parallel = false;
                compute_memory.reset();
                sched.reset();
                compute_arenas.clear();
                sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, false, cparams.op_offload));
                ggml_backend_sched_set_copy_callback(sched.get(), sched_copy_experts, this);
                if (no_alloc() || prepare_arenas()) {
                    gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get());
                }
            }
            if (!gf) {
                throw std::runtime_error("failed to allocate compute pp buffers");
            }
        }

        n_splits_pp        = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_pp         = ggml_graph_n_nodes(gf);
        n_inputs_pp        = get_gf_res_reserve()->inputs.size();
        n_input_tensors_pp = this->n_input_tensors;
    }

    // reserve with tg (token generation) graph to get the number of splits and nodes
    if (cparams.training) {
        // no tg graph for training
        n_splits_tg = n_splits_pp;
        n_nodes_tg  = n_nodes_pp;
    } else {
        auto * gf = graph_reserve(n_seqs, n_seqs, n_seqs, mctx.get(), no_alloc());
        if (!gf) {
            throw std::runtime_error("failed to allocate compute tg buffers");
        }

        n_splits_tg        = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_tg         = ggml_graph_n_nodes(gf);
        n_inputs_tg        = get_gf_res_reserve()->inputs.size();
        n_input_tensors_tg = this->n_input_tensors;
    }

    // reserve again with pp graph to avoid ggml-alloc reallocations during inference
    {
        // TODO: the worst case graph is not always reached for `n_seqs > 1`
        //       need to implement a more robust mechanism that tries a few different inputs and analyzes the results
        ggml_cgraph * gf = graph_reserve(n_tokens, n_seqs_pp, n_outputs_pp, mctx.get(), no_alloc());

        if (!gf) {
            throw std::runtime_error("failed to allocate compute pp buffers");
        }
    }

    for (size_t i = 0; i < backend_ptrs.size(); ++i) {
        ggml_backend_t             backend = backend_ptrs[i];
        ggml_backend_buffer_type_t buft    = backend_buft[i];
        if (!no_alloc()) {
            backend_buf_exp_size[i] = ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
        if (backend_buf_exp_size[i] > 1) {
            LLAMA_LOG_INFO("%s: %10s compute buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buft_name(buft),
                    backend_buf_exp_size[i] / 1024.0 / 1024.0);
        }
    }

    {
        const bool diff = n_nodes_pp != n_nodes_tg || n_splits_pp != n_splits_tg ||
                          n_inputs_pp != n_inputs_tg || n_input_tensors_pp != n_input_tensors_tg;

        const auto val = [diff](int v_pp, int v_tg) -> std::string {
            return diff ? format("%d / %d", v_pp, v_tg) : format("%d", v_pp);
        };

        LLAMA_LOG_INFO("%s: graph%s: nodes = %s, splits = %s, input objects = %s, input tensors = %s\n",
                __func__,
                diff ? format(" (pp bs=%d, tg bs=%d)", n_tokens, n_seqs).c_str() : "",
                val(n_nodes_pp, n_nodes_tg).c_str(),
                val(n_splits_pp, n_splits_tg).c_str(),
                val(n_inputs_pp, n_inputs_tg).c_str(),
                val(n_input_tensors_pp, n_input_tensors_tg).c_str());
    }

    const int64_t t_end_us = ggml_time_us();

    LLAMA_LOG_INFO("%s: reserve took %.2f ms, sched copies = %d\n",
            __func__, (t_end_us - t_start_us)/1000.0, ggml_backend_sched_get_n_copies(sched.get()));

    sched_reserve_state.complete();
}

void llama_context::synchronize() {
    if (!sched) {
        return;
    }

    if (compute_memory) {
        compute_memory->synchronize();
    } else {
        ggml_backend_sched_synchronize(sched.get());
    }

    // FIXME: if multiple single tokens are evaluated without a synchronization,
    // the stats will be added to the prompt evaluation stats
    // this should only happen when using batch size 1 to evaluate a batch

    // add the evaluation to the stats
    if (n_queued_tokens == 1) {
        if (!cparams.no_perf) {
            t_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_eval++;
    } else if (n_queued_tokens > 1) {
        if (!cparams.no_perf) {
            t_p_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_p_eval += n_queued_tokens;
    }

    // get a more accurate load time, upon first eval
    if (n_queued_tokens > 0 && !has_evaluated_once) {
        t_load_us = ggml_time_us() - t_start_us;
        has_evaluated_once = true;
    }

    n_queued_tokens = 0;
    t_compute_start_us = 0;
}

const llama_model & llama_context::get_model() const {
    return model;
}

const llama_cparams & llama_context::get_cparams() const {
    return cparams;
}

ggml_backend_sched_t llama_context::get_sched() const {
    return sched.get();
}

// This reports the coordinated path separately from the legacy arena fallback.
bool llama_context::uses_memory_coordinator() const {
    return compute_memory != nullptr;
}
const llama_context_memory * llama_context::get_compute_memory() const noexcept {
    return compute_memory.get();
}

llama_context_memory * llama_context_compute_memory(llama_context * ctx) noexcept {
    return ctx ? const_cast<llama_context_memory *>(ctx->get_compute_memory()) : nullptr;
}

void llama_context::clear_serial_features() {
    serial_feature_rows.clear();
    serial_feature_sources.clear();
    serial_feature_image_rows.clear();
    serial_feature_image_values.clear();
    serial_feature_image_width=0;
}

bool llama_context::serial_features_match(const llama_batch_ext & batch,const int32_t * ids,uint32_t count) const {
    if (!serial_feature_producer || serial_feature_rows.empty() || serial_feature_rows.size() != batch.tokens.size() ||
            !memory) return false;
    if (group_execution) {
        const auto & binding=*group_execution;
        if (binding.failed || binding.admission || !binding.table->projection_current(*binding.projection) ||
                serial_feature_sources.size()!=binding.projection->handles.size()) return false;
        for (size_t i=0;i<serial_feature_sources.size();++i) {
            const auto & saved=serial_feature_sources[i];const auto handle=binding.projection->handles[i];
            std::shared_ptr<llama_kv_stream_logical_cache> cache;
            if (saved.slot!=handle.slot || saved.request_epoch!=handle.generation ||
                    !binding.table->get(handle,binding.projection->role,cache) || !cache->ready() ||
                    cache->identity().id!=saved.cache_id || cache->identity().generation!=saved.generation ||
                    cache->tokens()!=saved.tokens) return false;
        }
    } else if (!serial_feature_sources.empty()) return false;
    if ((ids || count) && (!ids || count != serial_feature_layers.size() ||
            !std::equal(serial_feature_layers.begin(),serial_feature_layers.end(),ids))) return false;
    for (auto id : serial_feature_layers) if (!cparams.embeddings_layer_inp[uint32_t(id)]) return false;
    const bool image_frame=!serial_feature_image_rows.empty();
    if (image_frame) {
        if (!serial_feature_images || serial_feature_image_rows.size()!=batch.tokens.size() ||
                serial_feature_image_width!=batch.n_embd || serial_feature_image_values.empty() ||
                serial_feature_image_values.size()!=batch.embd.size() ||
                std::memcmp(serial_feature_image_values.data(),batch.embd.data(),batch.embd.size()*sizeof(float))) return false;
        for (size_t i=0;i<batch.tokens.size();++i) {
            const auto & token=batch.tokens[i];
            const auto & saved=serial_feature_image_rows[i];
            if (token.has_embd!=saved.has_embd || token.pos!=saved.position ||
                    (token.has_embd && token.embd_off!=saved.offset)) return false;
        }
    }
    std::array<llama_pos,LLAMA_MAX_SEQ> last;last.fill(-1);
    for (size_t i=0;i<batch.tokens.size();++i) {
        const auto & token=batch.tokens[i];
        const auto & saved=serial_feature_rows[i];
        if ((!image_frame && token.has_embd) || token.seq_ids.size()!=1 || saved.sequence<0 ||
                uint32_t(saved.sequence)>=cparams.n_seq_max || (!group_execution && saved.sequence!=0) ||
                !token.seq_ids.count(saved.sequence) || token.id!=saved.token || token.pos[0]!=saved.position) return false;
        last[uint32_t(saved.sequence)]=saved.position;
    }
    for (uint32_t sequence=0;sequence<cparams.n_seq_max;++sequence)
        if (last[sequence]>=0 && memory->seq_pos_max(llama_seq_id(sequence))!=last[sequence]) return false;
    return true;
}

bool llama_serial_features_match(llama_context * ctx,const llama_batch_ext * batch,const int32_t * ids,uint32_t count) {
    return ctx && batch && ctx->serial_features_match(*batch,ids,count);
}
bool llama_context_suspend_kv_device(llama_context * ctx) noexcept {
    if (!ctx || !ctx->get_cparams().kv_streaming()) return false;
    auto * owner = llama_context_compute_memory(ctx);
    const auto resources = ctx->get_kv_resources();
    auto * stream = resources.full ? resources.full->get_kv_stream() : nullptr;
    if (!owner || !stream || (!stream->complete() && !stream->device_suspended())) return false;
    struct context_completion : llama_memory_executor_backend {
        llama_context & ctx;
        llama_memory_recurrent * recurrent;
        context_completion(llama_context & ctx,llama_memory_recurrent * recurrent) : ctx(ctx),recurrent(recurrent) {}
        bool drain() override {
            ctx.synchronize();
            return !recurrent || (recurrent->complete_spill() && recurrent->complete_restore_spill());
        }
    };
    try {
        auto * recurrent = resources.recurrent;
        if (!recurrent && !resources.bounded) return false;
        context_completion completion(*ctx,recurrent);
        return owner->suspend_kv(&completion);
    } catch (...) { return false; }
}

bool llama_context::resume_kv_device(llama_memory_text_phase phase) {
    if (!cparams.kv_streaming() || !compute_memory || !compute_memory->kv_device_suspended()) return false;
    const bool previous_decode = cparams.kv_stream_decode;
    const bool resumed = compute_memory->resume_kv(phase,[&] {
        for (auto & previous : gf_res_prev) if (previous) previous->reset();
        gf_res_prev_active = nullptr;
        gf_res_reserve->reset();
        cparams.kv_stream_decode = phase == llama_memory_text_phase::decode;
        auto reserved = memory->init_full();
        const uint32_t decode_width = std::min(cparams.kv_stream_decode_rows(),std::min(cparams.n_ctx,cparams.n_ubatch));
        const uint32_t width = phase == llama_memory_text_phase::decode ?
            decode_width : std::min(cparams.n_ctx,cparams.n_ubatch);
        return reserved && graph_reserve(width,cparams.n_seq_max,std::min(width,cparams.n_outputs_max),reserved.get());
    });
    if (!resumed) cparams.kv_stream_decode = previous_decode;
    return resumed;
}

bool llama_context_resume_kv_device(llama_context * ctx,llama_memory_text_phase phase) noexcept {
    try { return ctx && ctx->resume_kv_device(phase); } catch (...) { return false; }
}


bool llama_context::uses_compute_arenas() const {
    return !compute_arenas.empty() || (compute_memory && compute_memory->uses_arenas());
}

uint32_t llama_context::n_ctx() const {
    return cparams.n_ctx;
}

uint32_t llama_context::n_ctx_seq() const {
    return cparams.n_ctx_seq;
}

uint32_t llama_context::n_batch() const {
    return cparams.n_batch;
}

uint32_t llama_context::n_ubatch() const {
    return cparams.n_ubatch;
}

uint32_t llama_context::n_seq_max() const {
    return cparams.n_seq_max;
}

uint32_t llama_context::n_threads() const {
    return cparams.n_threads;
}

uint32_t llama_context::n_threads_batch() const {
    return cparams.n_threads_batch;
}

llama_memory_t llama_context::get_memory() const {
    return memory.get();
}

bool llama_context::memory_update(bool optimize) {
    if (!memory) {
        return false;
    }

    {
        const auto mctx = memory->init_update(this, optimize);
        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                    // noop
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    // no updates need to be performed
                    return false;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: failed to prepare memory update\n", __func__);
                    return false;
                }
        }

        // reset the previous graph results to make sure that they won't be reused
        // TODO: make mctx->apply() report if a graph reserve is needed, then reset graph results only if the memory module reset the scheduler
        for (auto & res : gf_res_prev) {
            if (res) {
                res->reset();
            }
        }
        gf_res_prev_active = nullptr;

        if (!mctx->apply()) {
            LLAMA_LOG_ERROR("%s: failed to apply memory update\n", __func__);
        }
    }

    // if the memory module did any computation, we have to reserve a new worst-case graph
    {
        const auto mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory context");
        }

        const uint32_t n_seqs = group_selected.empty() ? cparams.n_seq_max : uint32_t(group_selected.size());
        const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

        const uint32_t n_outputs_max = std::min(n_tokens, cparams.n_outputs_max);

        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_max, mctx.get());
        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to reserve graph after the memory update\n", __func__);
        }
    }

    return true;
}

enum llama_pooling_type llama_context::pooling_type() const {
    return cparams.pooling_type;
}

float * llama_context::get_logits() {
    output_reorder();

    return logits.data;
}

int64_t llama_context::output_resolve_row(int32_t i) const {
    int64_t j = -1;

    // support negative indices (last output row)
    if (i < 0) {
        j = n_outputs + i;
        if (j < 0) {
            throw std::runtime_error(format("negative index out of range [0, %d)", n_outputs));
        }
    } else if ((size_t) i >= output_ids.size()) {
        throw std::runtime_error(format("out of range [0, %zu)", output_ids.size()));
    } else {
        // use output_ids to translate the batch token index into a row number
        // that holds this token's data.
        j = output_ids[i];
    }

    if (j < 0) {
        // the batch token was not configured to output anything
        throw std::runtime_error(format("batch.logits[%d] != true", i));
    }

    if (j >= n_outputs) {
        throw std::runtime_error(format("corrupt output buffer (j=%" PRId64 ", n_outputs=%d)", j, n_outputs));
    }

    return j;
}

float * llama_context::get_logits_ith(int32_t i) {
    output_reorder();

    try {
        if (logits.data == nullptr) {
            throw std::runtime_error("no logits");
        }

        const int64_t j = output_resolve_row(i);
        return logits.data + j*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid logits id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings() {
    output_reorder();

    return embd.data;
}

llama_token * llama_context::get_sampled_tokens()  const{
    return sampling.sampled.data;
}

float * llama_context::get_embeddings_ith(int32_t i) {
    output_reorder();

    try {
        if (embd.data == nullptr) {
            throw std::runtime_error("no embeddings");
        }

        const int64_t j = output_resolve_row(i);
        const uint32_t n_embd_out = model.hparams.n_embd_out();
        return embd.data + j*n_embd_out;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings_seq(llama_seq_id seq_id) {
    auto it = embd_seq.find(seq_id);
    if (it == embd_seq.end()) {
        return nullptr;
    }

    return it->second.data();
}

float * llama_context::get_embeddings_nextn() {
    output_reorder();

    return embd_nextn.data;
}

float * llama_context::get_embeddings_nextn_ith(int32_t i) {
    output_reorder();

    try {
        if (embd_nextn.data == nullptr) {
            throw std::runtime_error("no nextn embeddings");
        }

        const uint32_t n_embd = model.hparams.n_embd_out();

        if (!cparams.embeddings_nextn_masked) {
            // unmasked: nextn rows are stored densely, indexed by raw token position.
            if (i < 0 || (size_t)(i + 1) * n_embd > embd_nextn.size) {
                throw std::runtime_error(format("out of range [0, %zu)", embd_nextn.size / n_embd));
            }
            return embd_nextn.data + (size_t) i * n_embd;
        }

        const int64_t j = output_resolve_row(i);
        return embd_nextn.data + j*n_embd;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid nextn embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings_layer_inp(uint32_t lid) {
    output_reorder();

    GGML_ASSERT(lid < embd_layer_inp.size() && embd_layer_inp[lid].has_data());

    return embd_layer_inp[lid].data;
}

llama_token llama_context::get_sampled_token_ith(int32_t idx) {
    output_reorder();

    if (!sampling.sampled.has_data()) {
        return LLAMA_TOKEN_NULL;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        GGML_ASSERT(row < (int64_t) sampling.sampled.size);
        return sampling.sampled.data[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled token id %d, reason: %s\n", __func__, idx, err.what());
        return LLAMA_TOKEN_NULL;
    }
}

float * llama_context::get_sampled_probs_ith(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size() || sampling.probs_count[row] == 0) {
            return nullptr;
        }
        return sampling.probs.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

float * llama_context::get_sampled_logits_ith(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size() || sampling.logits_count[row] == 0) {
            return nullptr;
        }
        return sampling.logits.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

const llama_token * llama_context::get_sampled_candidates_ith(int32_t idx) {
    output_reorder();

    try {
        const int64_t row = output_resolve_row(idx);
        if (sampling.candidates.has_data() &&
            (size_t) row < sampling.candidates_count.size() &&
            sampling.candidates_count[row] > 0) {
            return sampling.candidates.data + row*model.vocab.n_tokens();
        }
    } catch (const std::exception & err) {
        // fallback to full vocab list
        GGML_UNUSED(err);
    }

    return sampling.token_ids_full_vocab.data();
}

size_t llama_context::get_sampled_candidates_count(int32_t idx) {
    output_reorder();

    if (!sampling.candidates.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.candidates_count.size()) {
            return 0;
        }
        return sampling.candidates_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled candidates count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_logits_count(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return model.vocab.n_tokens();
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size()) {
            return 0;
        }
        return sampling.logits_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_probs_count(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size()) {
            return 0;
        }
        return sampling.probs_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}


void llama_context::attach_threadpool(
           ggml_threadpool_t threadpool,
           ggml_threadpool_t threadpool_batch) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = threadpool;
    this->threadpool_batch = threadpool_batch ? threadpool_batch : threadpool;
}

void llama_context::detach_threadpool() {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = nullptr;
    this->threadpool_batch = nullptr;
}

void llama_context::set_n_threads(int32_t n_threads, int32_t n_threads_batch) {
    LLAMA_LOG_DEBUG("%s: n_threads = %d, n_threads_batch = %d\n", __func__, n_threads, n_threads_batch);

    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads_batch;
}

void llama_context::set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->abort_callback      = abort_callback;
    this->abort_callback_data = abort_callback_data;

    for (auto & backend : backends) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
        if (reg) {
            auto * set_abort_callback_fn = (ggml_backend_set_abort_callback_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_abort_callback");
            if (set_abort_callback_fn) {
                set_abort_callback_fn(backend.get(), this->abort_callback, this->abort_callback_data);
            }
        }
    }
}

void llama_context::set_embeddings(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    cparams.embeddings = value;

    // TODO: not sure yet if we want to reserve here
    //sched_reserve_state.invalidate();
}

void llama_context::set_embeddings_nextn(bool value, bool masked) {
    LLAMA_LOG_DEBUG("%s: value = %d, masked = %d\n", __func__, value, masked);

    if (cparams.embeddings_nextn != value || cparams.embeddings_nextn_masked != masked) {
        // these flags change the graph shape
        sched_reserve_state.invalidate();
    }

    cparams.embeddings_nextn        = value;
    cparams.embeddings_nextn_masked = masked;
}

void llama_context::set_embeddings_layer_inp(uint32_t lid, bool enable) {
    LLAMA_LOG_DEBUG("%s: lid = %d, enable = %d\n", __func__, lid, enable);

    GGML_ASSERT(lid <= model.hparams.n_layer());

    if (cparams.embeddings_layer_inp[lid] == enable) return;

    cparams.embeddings_layer_inp[lid] = enable;

    // note: without this reserve, the draft acceptance drops to zero. not sure why - this is unexpected
    sched_reserve_state.invalidate();
}

void llama_context::set_nextn_layer_offset(int32_t offset) {
    cparams.nextn_layer_offset = offset;
}

void llama_context::set_causal_attn(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.causal_attn == value) {
        return;
    }

    cparams.causal_attn = value;

    sched_reserve_state.invalidate();
}

bool llama_context::get_causal_attn() const {
    return cparams.causal_attn;
}

void llama_context::set_warmup(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.warmup == value) {
        return;
    }

    cparams.warmup = value;

    // warmups are usually with small batches, so no need to reserve
    //sched_reserve_state.invalidate();
}

bool llama_context::set_sampler(llama_seq_id seq_id, llama_sampler * sampler) {
    if (!sampler && sampling.samplers.count(seq_id) == 0) {
        return true;
    }

    LLAMA_LOG_DEBUG("%s: seq_id = %d, sampler = %p\n", __func__, (int) seq_id, (void *) sampler);

    if (sampler && model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        static bool warned = false;
        if (!warned) {
            LLAMA_LOG_WARN("%s: backend sampling not supported with SPLIT_MODE_TENSOR; using CPU\n", __func__);
            warned = true;
        }
        if (sampling.samplers.count(seq_id) > 0) {
            sched_reserve_state.invalidate();
        }
        sampling.samplers.erase(seq_id);
        return false;
    }

    const bool can_offload =
        sampler &&
        sampler->iface->backend_init &&
        sampler->iface->backend_apply &&
        llama_sampler_chain_n(sampler) > 0;

    if (sampler && can_offload) {
        auto * buft = ggml_backend_dev_buffer_type(model.dev_output());

        sampler->iface->backend_init(sampler, buft, cparams.n_outputs_max_per_seq);

        sampling.samplers[seq_id] = sampler;

        sched_reserve_state.invalidate();

        return true;
    }

    if (sampler && !can_offload) {
        LLAMA_LOG_WARN("%s: sampler '%s' for seq_id = %d, cannot be offloaded to the backend\n", __func__, llama_sampler_name(sampler), seq_id);

        if (sampling.samplers.count(seq_id) > 0) {
            sched_reserve_state.invalidate();
        }

        sampling.samplers.erase(seq_id);

        return false;
    }

    sampling.samplers.erase(seq_id);

    sched_reserve_state.invalidate();

    return true;
}

void llama_context::set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    if (adapters_lora_are_same(adapters, n_adapters, scales)) {
        return;
    }

    loras.reset(new llama_adapter_loras());

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] != 0.0f) {
            loras->insert({adapters[i], scales[i]});
        }
    }

    sched_reserve_state.invalidate();
}

bool llama_context::adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    // Adapters with a zero scale are never added to `loras`, so also ignore them for the comparison.
    size_t n_non_zero = 0;

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] == 0.0f) {
            continue;
        }
        n_non_zero++;

        auto it = loras->find(adapters[i]);

        if (it == loras->end() || it->second != scales[i]) {
            return false;
        }
    }

    if (n_non_zero != loras->size()) {
        return false;
    }

    return true;
}

bool llama_context::set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end) {
    LLAMA_LOG_DEBUG("%s: il_start = %d, il_end = %d\n", __func__, il_start, il_end);

    bool res = cvec->apply(model, data, len, n_embd, il_start, il_end);

    sched_reserve_state.invalidate();

    return res;
}

// Complete the managed K/V publication before inspecting its host frontier.
bool llama_context::prepare_shared_kv_read() {
    if (!cparams.kv_shared_read || !cparams.ctx_other ||
            (kv_read_frame_active && !finish_shared_kv_read())) return false;
    auto * target=cparams.ctx_other;
    auto * stream=target->kv_resources.full ? target->kv_resources.full->get_kv_stream() : nullptr;
    if (!stream || !target->kv_resources.bounded || !target->compute_memory || !kv_resources.full || !kv_resources.bounded) return false;
    try {
        target->synchronize();
        stream->release_graphs();
        if (!stream->set_workspaces({}) || !target->compute_memory->prepare_serial_decode()) return false;
        llama_memory_cache_requirement source;
        if (!stream->read_source(source)) return false;
        auto local=target->kv_resources.bounded->retain_bounded_storage(source.identity.id+1,source.identity.generation,
            target->get_cparams().n_ctx_seq,target->get_cparams().n_ubatch);
        const auto pin=local ? local->read(local->identity()) : llama_memory_bounded_kv_read{};
        if (!pin || !kv_resources.full->arm_shared_stream_reader() || !kv_resources.bounded->arm_shared_bounded_reader(pin)) {
            kv_resources.full->release_shared_stream_reader();
            kv_resources.bounded->release_shared_bounded_reader();
            return false;
        }
        kv_read_frame_active=true;
        return true;
    } catch (const std::exception & error) {
        LLAMA_LOG_ERROR("%s: failed to acquire target KV sources: %s\n",__func__,error.what());
        return false;
    }
}

bool llama_context::finish_shared_kv_read() noexcept {
    if (!kv_read_frame_active) return true;
    try {
        if (compute_memory) compute_memory->synchronize();
        else if (sched) ggml_backend_sched_synchronize(sched.get());
        if (!kv_resources.full->release_shared_stream_reader()) return false;
        kv_resources.bounded->release_shared_bounded_reader();
        kv_read_frame_active=false;
        return true;
    } catch (...) { return false; }
}

bool llama_context::publish_mtp_kv(const llama_ubatch & ubatch, const llm_graph_result & graph) {
    auto * draft = dynamic_cast<llama_kv_cache *>(memory.get());
    if (!draft || !draft->complete_mtp_publication()) return false;
    auto cache = draft ? draft->mtp_auxiliary_cache() : nullptr;
    if (!cache || !graph.t_mtp_k || !graph.t_mtp_v || !sched || !ubatch.n_tokens ||
            ubatch.n_seqs_unq != 1 || !ubatch.seq_id_unq || ubatch.seq_id_unq[0] != 0 ||
            !ubatch.pos || ubatch.pos[0] < 0) return false;
    if (cache->tokens() < ubatch.n_tokens) return false;
    const size_t first = cache->tokens()-ubatch.n_tokens;
    if (first > cache->tokens() || first > cache->host()->config().context_tokens ||
            ubatch.n_tokens > cache->host()->config().context_tokens - first) return false;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (ubatch.pos[i] < 0) return false;
    }
    auto * target_hybrid = mtp_target_ctx ?
        dynamic_cast<llama_memory_hybrid *>(llama_get_memory(mtp_target_ctx)) : nullptr;
    auto * target_stream = target_hybrid ? target_hybrid->get_mem_attn()->get_kv_stream() : nullptr;
    if (!target_stream || target_stream->auxiliary_cache() != cache) return false;

    // The proxy completed host publication and retained-span adoption here.
    // Decode attention may have consumed provisional device spans while D2H
    // was pending; the serial phase owner drains later downstream graph work.
    const auto frontiers = cache->frontiers();
    if (cache->tokens() != first + ubatch.n_tokens ||
            frontiers.host != cache->tokens() || frontiers.committed != cache->tokens() ||
            frontiers.device != 0) return false;
    if (!cparams.mtp_span_attention) return !target_stream->has_mtp_layer();
    ggml_kv_stream_span_plan_view view;
    auto * plan = target_stream->mtp_layer_plan(ubatch.n_tokens);
    return plan && ggml_kv_stream_span_plan_get_view(plan, view) &&
        view.active_tokens == cache->tokens() && draft->retire_serial_access();
}

llm_graph_result * llama_context::process_ubatch(
        const llama_ubatch & ubatch, llm_graph_type gtype, llama_memory_context_i * mctx,
        llama_memory_text_phase phase, ggml_status & ret) {
    llama_memory_recurrent * rs_spill = nullptr;
    if (model.arch == LLM_ARCH_QWEN35 && cparams.n_rs_seq && gtype == LLM_GRAPH_TYPE_DEFAULT && memory) {
        rs_spill = kv_resources.recurrent;
        if (rs_spill && rs_spill->spill_enabled() && !rs_spill->complete_spill()) {
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }
    }
    llama_kv_stream_model * stream = nullptr;
    auto * grouped=group_execution && kv_resources.full ? kv_resources.full->get_group_model() : nullptr;
    const bool readonly=group_execution && group_execution->shared_readonly && cparams.ctx_type==LLAMA_CONTEXT_TYPE_MTP;
    if (readonly && (gtype!=LLM_GRAPH_TYPE_DECODER_MTP || !group_execution->read_prepare || !group_execution->read_prepare(ubatch))) {
        LLAMA_LOG_ERROR("%s: grouped shared source frame rejected\n",__func__);
        ret=GGML_STATUS_FAILED;return nullptr;
    }
    const auto * stream_context = grouped || cparams.kv_streaming() ? llama_memory_kv_context_get(mctx) : nullptr;
    if (grouped) {
        std::vector<uint32_t> indices;
        const bool target=gtype==LLM_GRAPH_TYPE_DEFAULT && cparams.ctx_type==LLAMA_CONTEXT_TYPE_DEFAULT;
        const bool auxiliary=gtype==LLM_GRAPH_TYPE_DECODER_MTP && cparams.ctx_type==LLAMA_CONTEXT_TYPE_MTP &&
            group_execution->shared_roles && group_target;
        if (!stream_context || (!target && !auxiliary) || !stream_context->selected_indices(indices) ||
                !grouped->begin(group_execution->projection,ubatch,indices,group_execution->phase)) {
            LLAMA_LOG_ERROR("%s: grouped graph handoff rejected (type=%d, rows=%u)\n",__func__,int(gtype),ubatch.n_tokens);
            ret=GGML_STATUS_FAILED;return nullptr;
        }
    } else if (cparams.kv_streaming()) {
        static const std::vector<ggml_backend_memory_lease_t> empty;
        stream = kv_resources.full ? kv_resources.full->get_kv_stream() : nullptr;
        if (!stream || !stream->set_workspaces(compute_memory ? compute_memory->workspace_leases() : empty)) {
            ret = GGML_STATUS_FAILED; return nullptr;
        }
        if (!stream_context || gtype != LLM_GRAPH_TYPE_DEFAULT ||
                (phase != llama_memory_text_phase::prefill && phase != llama_memory_text_phase::decode) ||
                !stream_context->kv_stream_begin(
                    ubatch, phase == llama_memory_text_phase::decode)) {
            LLAMA_LOG_ERROR("%s: KV streaming rejected non-serial/non-append execution\n",__func__);
            ret = GGML_STATUS_FAILED; return nullptr;
        }
    }
    struct stream_guard {
        llama_kv_stream_model * stream;
        llama_kv_stream_group_model * grouped;
        bool completed = false;
        std::function<bool()> finish_readers;
        ~stream_guard() {
            if (!completed) {
                try {if (finish_readers) finish_readers();} catch (...) {}
                if (stream) stream->abort();
                if (grouped) grouped->abort();
            }
        }
    } streaming{stream,grouped,false,{}};
    if (grouped && group_execution) {
        streaming.finish_readers=group_execution->reader_finish;
        if (group_execution->reader_prepare && !group_execution->reader_prepare(ubatch.n_tokens)) {
            LLAMA_LOG_ERROR("%s: grouped reader frame preparation failed\n",__func__);
            ret=GGML_STATUS_FAILED;return nullptr;
        }
    }
    if (mctx && !mctx->apply()) {
        LLAMA_LOG_ERROR("%s: failed to apply memory context\n", __func__);
        ret = GGML_STATUS_FAILED;
        return nullptr;
    }
    if (grouped && !grouped->apply_cells(kv_resources.full->get_cells(0))) {
        LLAMA_LOG_ERROR("%s: grouped append cell mapping rejected\n",__func__);
        ret=GGML_STATUS_FAILED;return nullptr;
    }
    const bool group_restore_phase=group_execution && group_execution->shared_roles && group_target &&
        (group_execution->phase==llama_memory_work_phase::mtp_draft ||
         group_execution->phase==llama_memory_work_phase::draft_block_prediction);
    auto * group_recurrent=group_restore_phase ? group_target->kv_resources.recurrent : nullptr;
    const bool group_restore=group_recurrent &&
        std::any_of(group_recurrent->rs_idx.begin(),group_recurrent->rs_idx.end(),[](uint32_t index) {return index!=0;}) &&
        group_recurrent_restore_isolated();
    auto * restore_target=group_restore ? &group_execution->peer->context :
        serial_target_ctx ? serial_target_ctx : cparams.mtp_publish_host ? mtp_target_ctx : nullptr;
    auto * restore=restore_target && (group_restore || restore_target->get_cparams().kv_streaming()) ? restore_target->kv_resources.recurrent : nullptr;
    llama_memory_recurrent * deferred_restore=nullptr;
    const bool restore_pending=restore && (group_restore ?
        std::any_of(restore->rs_idx.begin(),restore->rs_idx.end(),[](uint32_t index) {return index!=0;}) :
        !restore->rs_idx.empty() && restore->rs_idx[0]);
    if (restore_pending && restore->staged_publication_enabled()) {
        const auto disjoint=[&](ggml_backend_buffer_t scratch) {
            if (!scratch) return false;
            const size_t bytes=ggml_backend_buffer_get_size(scratch);
            if (!bytes) return true;
            const auto start=reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(scratch));
            if (!start || bytes>UINTPTR_MAX-start) return false;
            for (size_t layer=0;layer<restore->r_l.size();++layer) {
                if (!restore->r_l[layer]) continue;
                for (auto * state : {restore->r_l[layer],restore->s_l[layer]}) {
                    if (!state || !state->buffer) return false;
                    const auto data=reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(state->buffer));
                    const size_t size=ggml_backend_buffer_get_size(state->buffer);
                    if (!data || size>UINTPTR_MAX-data || (start<data+size && data<start+bytes)) return false;
                }
            }
            return true;
        };
        bool isolated=group_restore || (compute_memory && restore_target->compute_memory &&
            disjoint(restore_target->compute_memory->shared_parent()));
        if (!group_restore && compute_memory) for (auto * lease : compute_memory->workspace_leases())
            isolated=isolated && lease && disjoint(ggml_backend_memory_lease_buffer(lease));
        // Group state has its own persistent lease; serial state is outside the draft arena.
        if (isolated) restore_target->synchronize();
        if (!isolated) {
            LLAMA_LOG_ERROR("%s: target recurrent restore overlaps draft storage\n",__func__);
            ret=GGML_STATUS_FAILED;
            return nullptr;
        }
        // MTP loads its own KV before bulk target restores use the copy engine.
        if (!group_restore || group_execution->phase!=llama_memory_work_phase::mtp_draft) deferred_restore=restore;
    }
    if (cparams.kv_shared_read && !prepare_shared_kv_read()) {
        LLAMA_LOG_ERROR("%s: failed to acquire shared target KV sources\n",__func__);
        ret=GGML_STATUS_FAILED; return nullptr;
    }
    struct shared_read_guard {
        llama_context * owner;
        ~shared_read_guard() { if (owner) owner->finish_shared_kv_read(); }
    } reading{cparams.kv_shared_read ? this : nullptr};
    if (stream && cparams.kv_stream_mask && phase == llama_memory_text_phase::prefill &&
            !stream_context->prepare_mask_bounds(ubatch,cparams.causal_attn)) {
        LLAMA_LOG_ERROR("%s: managed prefill mask cannot be represented by an exact visibility interval\n",__func__);
        ret=GGML_STATUS_FAILED;
        return nullptr;
    }
    if (cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP && cparams.mtp_publish_host) {
        auto * draft = dynamic_cast<llama_kv_cache *>(memory.get());
        auto * target_hybrid = mtp_target_ctx ?
            dynamic_cast<llama_memory_hybrid *>(llama_get_memory(mtp_target_ctx)) : nullptr;
        auto * target = target_hybrid ? target_hybrid->get_mem_attn()->get_kv_stream() : nullptr;
        if (!draft || !target || !mctx) { ret = GGML_STATUS_FAILED; return nullptr; }
        const bool eligible = target->has_mtp_layer() &&
            static_cast<llama_kv_cache_context *>(mctx)->mtp_span_append(ubatch) &&
            target->mtp_layer_plan(1) != nullptr;
        cparams.mtp_span_attention = eligible &&
            draft->set_mtp_span_mode(true, draft->mtp_auxiliary_cache()->tokens(), ubatch.n_tokens);
        if (!cparams.mtp_span_attention) draft->set_mtp_span_mode(false);
    }

    auto * res = get_gf_res_prev();
    auto * gf  = res->get_gf();

    // the new graph parameters
    // in order to correctly reuse a graph, it's full topology has to be uniquely determined by these parameters
    const auto gparams = graph_params(res, ubatch, mctx, gtype);

    // A phase handoff can retire bindings even when graph dimensions still match.
    const bool binding_current=(!group_execution && !readonly) || (compute_memory && compute_memory->serial_ready());
    if (binding_current && !graph_reuse_disable && gf_res_prev_active == res && res->can_reuse(gparams)) {
        //LLAMA_LOG_DEBUG("%s: reusing previous graph\n", __func__);

        // with pipeline parallelism, the previous graph_compute_async may still be running
        // on the GPU. we must synchronize before set_inputs to avoid overwriting input tensors
        // that the previous compute is still reading.
        if (cparams.pipeline_parallel) {
            ggml_backend_sched_synchronize(sched.get());
        }

        n_reused++;
    } else {
        gf_res_prev_active = nullptr;
        if ((group_execution || readonly) && compute_memory && !compute_memory->retire_graph()) {ret=GGML_STATUS_FAILED;return nullptr;}
        if (stream) stream->release_graphs();
        res->reset();

        ggml_backend_sched_reset(sched.get());
        ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);

        //const auto t_start_us = ggml_time_us();

        gf = model.build_graph(gparams);

        //LLAMA_LOG_INFO("graph build time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);

        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to initialize graph\n", __func__);
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }

        if (!ggml_backend_sched_alloc_graph(sched.get(), gf)) {
            LLAMA_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            ret = GGML_STATUS_ALLOC_FAILED;
            return nullptr;
        }

        gf_res_prev_active = res;
    }

    if (group_execution && (!compute_memory || !compute_memory->capture_external_workspace())) {
        LLAMA_LOG_ERROR("%s: grouped graph workspace capture rejected\n",__func__);
        ret=GGML_STATUS_FAILED;return nullptr;
    }
    if (rs_spill && !rs_spill->restore_spill_async(sched.get())) {
        LLAMA_LOG_ERROR("%s: failed to enqueue recurrent restore\n",__func__);
        ret = GGML_STATUS_FAILED;
        return nullptr;
    }

    // set the input data for the input tensors
    {
        //const auto t_start_us = ggml_time_us();

        // FIXME this call causes a crash if any model inputs were not used in the graph and were therefore not allocated
        res->set_inputs(&ubatch);

        //LLAMA_LOG_INFO("graph set inputs time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);
    }

    struct staged_spill_guard {
        llama_memory_recurrent * owner;
        bool active;
        ~staged_spill_guard() { if (active) owner->abort_spill_capture(); }
    } spill_capture{rs_spill,rs_spill && rs_spill->staged_publication_enabled()};
    if (spill_capture.active && !rs_spill->begin_spill_capture(ubatch)) {
        LLAMA_LOG_ERROR("%s: recurrent publication capture rejected\n",__func__);
        ret = GGML_STATUS_FAILED;
        return nullptr;
    }

    struct restore_submission_guard {
        ggml_backend_sched_t scheduler;
        ~restore_submission_guard() { if (scheduler) ggml_backend_sched_set_pre_compute_callback(scheduler,nullptr,nullptr); }
    } restore_submission{deferred_restore ? sched.get() : nullptr};
    if (deferred_restore) {
        ggml_backend_sched_set_pre_compute_callback(sched.get(),[](ggml_backend_t backend,ggml_cgraph *,void * opaque) {
            auto * memory=static_cast<llama_memory_recurrent *>(opaque);
            if (std::none_of(memory->rs_idx.begin(),memory->rs_idx.end(),[](uint32_t index) {return index!=0;})) return true;
            for (auto * state : memory->r_l) if (state) {
                if (ggml_backend_get_device(backend) != ggml_backend_buft_get_device(ggml_backend_buffer_get_type(state->buffer))) return true;
                return memory->restore_spill_async(nullptr,false);
            }
            return false;
        },deferred_restore);
    }
    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);
    if (rs_spill && !rs_spill->complete_restore_spill()) {
        LLAMA_LOG_ERROR("%s: failed to complete recurrent restore\n",__func__);
        ret = GGML_STATUS_FAILED;
        return nullptr;
    }
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: failed to compute graph, compute status: %d\n", __func__, status);
        ret = status;
        return nullptr;
    }

    if (cparams.mtp_publish_host && !publish_mtp_kv(ubatch, *res)) {
        LLAMA_LOG_ERROR("%s: failed to publish authoritative MTP host KV\n", __func__);
        ret = GGML_STATUS_FAILED;
        return nullptr;
    }
    if (rs_spill && rs_spill->spill_enabled() &&
            !rs_spill->capture_spill(sched.get(), res->get_recurrent_snapshots())) {
        LLAMA_LOG_ERROR("%s: failed to capture recurrent snapshots\n", __func__);
        ret = GGML_STATUS_FAILED;
        return nullptr;
    }

    if (grouped && ((streaming.finish_readers && !streaming.finish_readers()) || !grouped->finish())) {
        LLAMA_LOG_ERROR("%s: grouped graph publication did not complete\n",__func__);
        ret=GGML_STATUS_FAILED;return nullptr;
    }
    if (stream && !stream->complete()) {
        LLAMA_LOG_ERROR("%s: KV streaming graph did not complete every attention layer\n",__func__);
        ret = GGML_STATUS_FAILED; return nullptr;
    }
    if (cparams.kv_shared_read && !finish_shared_kv_read()) {
        ret=GGML_STATUS_FAILED; return nullptr;
    }
    spill_capture.active=false;
    streaming.completed = true;
    ret = GGML_STATUS_SUCCESS;

    return res;
}

int llama_context::encode(const llama_batch_ext & batch_inp) {
    if (measurement_only || group_initializing) return -1;
    if (batch_inp.tokens.empty()) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    const auto & hparams = model.hparams;

    if (batch_inp.n_embd > 0 && batch_inp.n_embd != hparams.n_embd_inp_enc()) {
        LLAMA_LOG_ERROR("%s: embd row width %zu does not match the encoder input %u\n",
                __func__, batch_inp.n_embd, hparams.n_embd_inp_enc());
        return -1;
    }

    // eagle3/DFlash: features as encoder input, and non-draft paths fall back to model's input dim
    const int64_t n_vocab = model.vocab.n_tokens();

    // note: during encode, we always output all tokens and skip position continuity checks (output_all=true)
    if (!balloc->init(batch_inp, model.vocab, true)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens = balloc->get_n_tokens();

    // [TAG_NO_CACHE_PAD]
    // TODO: add new split mode where we pad the input sequences so that ubatch.equal_seqs == true
    const llama_ubatch ubatch = balloc->split_simple(n_tokens);

    // micro-batching is not possible for non-causal encoding, so we process the batch in a single shot
    GGML_ASSERT(cparams.n_ubatch >= n_tokens && "encoder requires n_ubatch >= n_tokens");

    // TODO: this clear of the buffer can easily be forgotten - need something better
    // sync first so any in-flight async copies into embd_seq complete before it is freed
    if (!embd_seq.empty()) {
        synchronize();
    }
    embd_seq.clear();

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }

    sched_reserve();

    n_queued_tokens += n_tokens;

    // reserve output buffer
    if (output_reserve(n_tokens) < n_tokens) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %u outputs\n", __func__, n_tokens);
        return -2;
    };

    for (uint32_t i = 0; i < n_tokens; ++i) {
        output_ids[i] = i;
    }

    n_outputs = n_tokens;

    const auto causal_attn_org = cparams.causal_attn;

    // always use non-causal attention for encoder graphs
    // TODO: this is a tmp solution until we have a proper way to support enc-dec models
    //       ref: https://github.com/ggml-org/llama.cpp/pull/12181#issuecomment-2730451223
    cparams.causal_attn = false;

    ggml_status status;
    const auto * res = process_ubatch(
        ubatch, LLM_GRAPH_TYPE_ENCODER, nullptr, llama_memory_text_phase::unspecified, status);

    cparams.causal_attn = causal_attn_org;

    if (!res) {
        switch (status) {
            case GGML_STATUS_ABORTED:      return  2;
            case GGML_STATUS_ALLOC_FAILED: return -2;
            case GGML_STATUS_FAILED:       return -3;
            case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
        }
    }

    auto * t_logits  = res->get_logits();
    auto * t_embd    = res->get_embd_pooled() ? res->get_embd_pooled() : res->get_embd();
    auto * t_h_nextn = cparams.embeddings_nextn ? res->get_h_nextn() : nullptr;

    // extract logits
    if (logits.data && t_logits) {
        ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
        GGML_ASSERT(backend_res != nullptr);
        GGML_ASSERT(logits.data != nullptr);

        ggml_backend_tensor_get_async(backend_res, t_logits, logits.data, 0, n_tokens*n_vocab*sizeof(float));
    }

    // extract embeddings
    if (embd.data && t_embd) {
        ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
        GGML_ASSERT(backend_embd != nullptr);

        switch (cparams.pooling_type) {
            case LLAMA_POOLING_TYPE_NONE:
                {
                    // extract token embeddings
                    GGML_ASSERT(embd.data != nullptr);
                    const uint32_t n_embd_out = hparams.n_embd_out();

                    GGML_ASSERT(n_tokens*n_embd_out <= (int64_t) embd.size);
                    ggml_backend_tensor_get_async(backend_embd, t_embd, embd.data, 0, n_tokens*n_embd_out*sizeof(float));
                } break;
            case LLAMA_POOLING_TYPE_MEAN:
            case LLAMA_POOLING_TYPE_CLS:
            case LLAMA_POOLING_TYPE_LAST:
                {
                    // extract sequence embeddings
                    auto & embd_seq_out = embd_seq;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        embd_seq_out[seq_id].resize(n_embd_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd_out*seq_idx)*sizeof(float), n_embd_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_RANK:
                {
                    // extract the rerank score - n_cls_out floats per sequence
                    auto & embd_seq_out = embd_seq;

                    const uint32_t n_cls_out = hparams.n_cls_out;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        embd_seq_out[seq_id].resize(n_cls_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_UNSPECIFIED:
                {
                    GGML_ABORT("unknown pooling type");
                }
        }
    }

    // extract nextn embeddings (hidden state before the final output norm)
    if (embd_nextn.data && t_h_nextn && cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
        ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched.get(), t_h_nextn);
        GGML_ASSERT(backend_h != nullptr);

        const uint32_t n_embd = hparams.n_embd_out();
        GGML_ASSERT(n_tokens*n_embd <= (int64_t) embd_nextn.size);
        ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn.data, 0, n_tokens*n_embd*sizeof(float));
    }

    // TODO: hacky solution
    if (model.arch == LLM_ARCH_T5 && t_embd) {
        //cross.t_embd = t_embd;

        synchronize();

        cross.n_embd = t_embd->ne[0];
        cross.n_enc  = t_embd->ne[1];
        cross.v_embd.resize(cross.n_embd*cross.n_enc);
        memcpy(cross.v_embd.data(), embd.data, ggml_nbytes(t_embd));

        const auto & batch = balloc->get_batch();

        // remember the sequence ids used during the encoding - needed for cross attention later
        cross.seq_ids_enc.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; i++) {
            cross.seq_ids_enc[i].clear();

            for (int s = 0; s < batch.n_seq_id[i]; s++) {
                const llama_seq_id seq_id = batch.seq_id[i][s];

                cross.seq_ids_enc[i].insert(seq_id);
            }
        }
    }

    return 0;
}

template<typename T>
static void copy_tensor_async_rows(
    const std::vector<ggml_tensor *> & tensors,
    const buffer_view<T> & dst,
    size_t stride,
    uint32_t row_offset,
    ggml_backend_sched_t sched,
    std::vector<uint32_t> * counts = nullptr) {
    if (!dst.has_data()) {
        return;
    }

    for (size_t i = 0; i < tensors.size(); ++i) {
        auto * tensor = tensors[i];
        if (tensor == nullptr) {
            continue;
        }

        const uint32_t row = row_offset + i;
        const size_t n_elements = ggml_nelements(tensor);
        GGML_ASSERT(ggml_is_contiguous(tensor) && "sampling tensor must be contiguous for async copy");
        GGML_ASSERT(n_elements <= stride);
        GGML_ASSERT((size_t) row * stride + n_elements <= dst.size);

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        T * row_ptr = dst.data + (size_t) row * stride;
        ggml_backend_tensor_get_async(backend, tensor, row_ptr, 0, ggml_nbytes(tensor));

        if (counts) {
            GGML_ASSERT(row < counts->size());
            (*counts)[row] = n_elements;
        }
    }
}

static bool needs_raw_logits(const llama_ubatch & ubatch, const std::map<llama_seq_id, llama_sampler *> & samplers) {
    for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
        if (!ubatch.output[i]) {
            continue;
        }

        // Check if the output token has at least one sequence without a backend sampler.
        for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) {
            llama_seq_id seq_id = ubatch.seq_id[i][j];
            if (samplers.find(seq_id) == samplers.end()) {
                return true;
            }
        }
    }
    return false; // all sequences use backend sampling
}

int llama_context::decode_mtp_embeddings(const llama_batch & batch, const float * hidden, size_t elements) {
    const auto width = model.hparams.n_embd_out();
    if (!hidden || batch.n_tokens <= 0 || batch.token || !batch.embd || !batch.pos ||
            width <= 0 || model.hparams.n_embd_inp() != width || size_t(batch.n_tokens) > SIZE_MAX/size_t(width) ||
            elements != size_t(batch.n_tokens)*size_t(width)) return -1;
    llama_batch_compat compat(this,batch);
    return decode_mtp_embeddings(*compat.batch_ext,hidden,elements);
}

int llama_context::decode_mtp_embeddings(const llama_batch_ext & batch, const float * hidden, size_t elements) {
    const auto width = model.hparams.n_embd_out();
    if (cparams.ctx_type != LLAMA_CONTEXT_TYPE_MTP || model.arch != LLM_ARCH_QWEN35 ||
            !hidden || mtp_visual_hidden || batch.tokens.empty() || batch.tokens.size() > cparams.n_batch ||
            width <= 0 || model.hparams.n_embd_inp() != width || batch.n_embd != size_t(width) ||
            batch.tokens.size() > SIZE_MAX/size_t(width) || elements != batch.tokens.size()*size_t(width)) return -1;
    for (const auto & token : batch.tokens) {
        if (token.id != LLAMA_TOKEN_NULL || !token.has_embd || token.pos[0] < 0) return -1;
    }
    struct hidden_scope {
        const float * & pointer;
        ~hidden_scope() { pointer = nullptr; }
    } scope{mtp_visual_hidden};
    mtp_visual_hidden = hidden;
    return decode(batch);
}

int llama_decode_mtp_embeddings(llama_context * ctx, const llama_batch & batch, const float * hidden, size_t elements) {
    try { return ctx ? ctx->decode_mtp_embeddings(batch,hidden,elements) : -1; } catch (...) { return -1; }
}

int llama_decode_mtp_embeddings(llama_context * ctx, const llama_batch_ext & batch, const float * hidden, size_t elements) {
    try { return ctx ? ctx->decode_mtp_embeddings(batch,hidden,elements) : -1; } catch (...) { return -1; }
}

int llama_context::decode(const llama_batch_ext & batch_inp) {
    if (measurement_only || group_initializing) return -1;
    if (serial_feature_producer) clear_serial_features();
    if (cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP && model.arch == LLM_ARCH_QWEN35 && !mtp_visual_hidden &&
            std::any_of(batch_inp.tokens.begin(), batch_inp.tokens.end(), [](const auto & token) {
                return token.id == LLAMA_TOKEN_NULL && token.has_embd;
            })) return -1;
    if (!memory) {
        LLAMA_LOG_DEBUG("%s: cannot decode batches with this context (calling encode() instead)\n", __func__);
        return encode(batch_inp);
    }

    if (batch_inp.tokens.empty()) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    if (cparams.kv_streaming() && cparams.kv_stream_decode && cparams.kv_stream_verify_max) {
        if (!group_execution) {
            if (batch_inp.tokens.size() > cparams.kv_stream_verify_max) return -1;
        } else {
            uint32_t widths[LLAMA_MAX_SEQ]{};
            for (const auto & token : batch_inp.tokens) {
                if (token.seq_ids.size()!=1) return -1;
                const auto sequence=*token.seq_ids.begin();
                if (sequence<0 || uint32_t(sequence)>=cparams.n_seq_max ||
                        std::find(group_selected.begin(),group_selected.end(),sequence)==group_selected.end() ||
                        ++widths[sequence]>cparams.kv_stream_verify_max) return -1;
            }
        }
    }

    if (batch_inp.n_embd > 0 && batch_inp.n_embd != batch_inp.n_embd_inp) {
        LLAMA_LOG_ERROR("%s: embd row width %zu does not match the decoder input %zu\n",
                __func__, batch_inp.n_embd, batch_inp.n_embd_inp);
        return -1;
    }

    if (cparams.kv_shared_read) {
        const auto * target=cparams.ctx_other;
        const auto * source=target && target->kv_resources.full ? target->kv_resources.full->get_kv_stream() : nullptr;
        if (!source || !source->complete() || source->tokens() >= target->get_cparams().n_ctx_seq ||
                batch_inp.tokens.size() != 1 || batch_inp.n_pos_per_embd != 1 ||
                batch_inp.n_embd != model.hparams.n_embd_inp()) return -1;
        const auto & token=batch_inp.tokens.front();
        if (token.id < 0 || uint32_t(token.id) >= model.vocab.n_tokens() || !token.has_embd ||
                token.embd_off > batch_inp.embd.size() || batch_inp.n_embd > batch_inp.embd.size()-token.embd_off ||
                token.seq_ids.size() != 1 || !token.seq_ids.count(0) ||
                token.pos[0] < 0 || size_t(token.pos[0]) != source->tokens()) return -1;
    }

    if (cparams.kv_streaming() && !group_execution) {
        auto * stream = kv_resources.full ? kv_resources.full->get_kv_stream() : nullptr;
        if (!stream || !stream->complete()) {
            LLAMA_LOG_ERROR("%s: KV streaming requires an idle target append\n",__func__); return -1;
        }
        const size_t first = stream->tokens();
        if (first > cparams.n_ctx_seq || batch_inp.tokens.size() > cparams.n_ctx_seq-first) {
            LLAMA_LOG_ERROR("%s: KV streaming context capacity exceeded\n",__func__); return 1;
        }
        const auto previous_position = kv_resources.full->seq_pos_max(0);
        if (!llama_kv_stream_validate_append(batch_inp, {first, cparams.n_ctx_seq, previous_position,
                model.hparams.n_pos_per_embd(), cparams.kv_stream_decode, false})) {
            LLAMA_LOG_ERROR("%s: KV streaming rejected target payload, positions or sequence\n",__func__); return -1;
        }
    }

    const auto & vocab   = model.vocab;
    const auto & hparams = model.hparams;

    const int64_t n_vocab = vocab.n_tokens();

    // when computing embeddings, all tokens are output
    const bool output_all   = cparams.embeddings;
    const bool has_samplers = !sampling.samplers.empty();

    const uint32_t n_seq_max = cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max;

    // TODO: avoid this workaround in the future
    // embedding contexts output every token even when no token is explicitly marked as output
    if (has_samplers) {
        std::vector<int32_t> seq_output_count(n_seq_max, 0);

        for (const auto & tok : batch_inp.tokens) {
            if (!output_all && !tok.output) {
                continue;
            }

            for (auto seq_id : tok.seq_ids) {
                if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
                    continue;
                }

                seq_output_count[seq_id]++;
                auto sampler = sampling.samplers.find(seq_id);
                if (sampler != sampling.samplers.end() &&
                        seq_output_count[seq_id] > (int32_t) cparams.n_outputs_max_per_seq) {
                    LLAMA_LOG_ERROR("%s: backend sampling supports at most %u outputs per sequence "
                            "(seq_id %d had %d)\n", __func__, cparams.n_outputs_max_per_seq,
                            seq_id, seq_output_count[seq_id]);
                    return -1;
                }
            }
        }
    }

    if (!balloc->init(batch_inp, vocab, output_all, mtp_visual_hidden)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    if (group_execution && !group_execution->begin(batch_inp)) return -1;
    struct group_decode_guard {
        group_execution_binding * binding;
        ~group_decode_guard() {if (binding && binding->admission) binding->finish(false);}
    } group_decoding{group_execution.get()};
    const uint32_t n_tokens_all  = balloc->get_n_tokens();
    const uint32_t n_outputs_all = balloc->get_n_outputs();
    if (!group_execution && cparams.serial_block_rows && balloc->get_batch().token && n_tokens_all > cparams.serial_block_rows) {
        LLAMA_LOG_ERROR("%s: serial draft prediction exceeds the measured intact block width\n",__func__);
        return -1;
    }

    const auto * serial_source=serial_target_ctx ? serial_target_ctx : group_target;
    const auto text_phase = cparams.kv_streaming() ?
        (cparams.kv_stream_decode ? llama_memory_text_phase::decode : llama_memory_text_phase::prefill) :
        (cparams.kv_shared_read ? llama_memory_text_phase::decode : cparams.serial_block_rows ?
            (balloc->get_batch().token || (serial_source && serial_source->get_cparams().kv_stream_decode) ?
                llama_memory_text_phase::decode : llama_memory_text_phase::prefill) : cparams.mtp_publish_host ?
            (n_tokens_all <= 4 ? llama_memory_text_phase::decode : llama_memory_text_phase::prefill) :
            llama_memory_text_phase::unspecified);

    if (output_all) {
        // require that all tokens are output
        if (n_outputs_all != n_tokens_all) {
            LLAMA_LOG_ERROR("%s: pooled embedding requires that all tokens are output (n_outputs_all = %d, n_tokens_all = %d)\n",
                    __func__, n_outputs_all, n_tokens_all);
            return -1;
        }
    }

    GGML_ASSERT(n_tokens_all <= cparams.n_batch);

    GGML_ASSERT((cparams.causal_attn || cparams.n_ubatch >= n_tokens_all) && "non-causal attention requires n_ubatch >= n_tokens");

    // TODO: this clear of the buffer can easily be forgotten - need something better
    // sync first so any in-flight async copies into embd_seq complete before it is freed
    if (!embd_seq.empty()) {
        synchronize();
    }
    embd_seq.clear();

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }
    n_queued_tokens += n_tokens_all;

    output_swaps.clear();
    embd_batch_idxs.clear();

    if (kv_resources.recurrent && !kv_resources.recurrent->complete_restore_spill()) {
        LLAMA_LOG_ERROR("%s: failed to complete recurrent restore before target handoff\n",__func__);
        return -2;
    }
    const bool reservation_was_pending = sched_reserve_state.begin();
    const uint64_t serial_transitions = compute_memory ? compute_memory->phase_transition_count() : 0;
    // Drain the other scheduler before reservation can discard or rewrite shared scratch.
    if (!group_execution && compute_memory && cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT &&
            !compute_memory->borrows_serial_parent() &&
            !compute_memory->prepare_serial_target()) {
        LLAMA_LOG_ERROR("%s: failed to hand off shared graph scratch to target\n", __func__);
        return -2;
    }
    if (compute_memory && (cparams.mtp_publish_host || cparams.serial_block_rows || cparams.kv_shared_read) && compute_memory->borrows_serial_parent() &&
            !compute_memory->prepare_serial_draft(text_phase)) {
        LLAMA_LOG_ERROR("%s: failed to hand off shared graph scratch to MTP\n", __func__);
        return -2;
    }

    sched_reserve();
    const bool shared_workspace_replanned = compute_memory &&
        compute_memory->graph_binding_revision() != compute_memory_graph_revision;
    bool draft_graph_rebuild = !reservation_was_pending && (cparams.mtp_publish_host || cparams.serial_block_rows || cparams.kv_shared_read) && compute_memory &&
        (compute_memory->phase_transition_count() != serial_transitions || shared_workspace_replanned);
    if (!group_execution && compute_memory && reservation_was_pending) {
        // A dirty scheduler reservation replaced the coordinator. The new
        // owner must receive the handoff as well, before graph inputs are set.
        const auto after_reserve = compute_memory->phase_transition_count();
        if (cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT && !compute_memory->borrows_serial_parent() && !compute_memory->prepare_serial_target()) return -2;
        if ((cparams.mtp_publish_host || cparams.serial_block_rows || cparams.kv_shared_read) && compute_memory->borrows_serial_parent() &&
                !compute_memory->prepare_serial_draft(text_phase)) return -2;
        draft_graph_rebuild = (cparams.mtp_publish_host || cparams.serial_block_rows || cparams.kv_shared_read) &&
            compute_memory->phase_transition_count() != after_reserve;
    }
    if (draft_graph_rebuild) {
        auto reserve_context = memory->init_full();
        const uint32_t reserve_tokens = cparams.serial_graph_reserve_rows(text_phase == llama_memory_text_phase::decode);
        const uint32_t reserve_outputs = std::min(reserve_tokens, cparams.n_outputs_max);
        ggml_cgraph * reserved;
        {
            struct span_intent_restore {
                bool & value;
                bool saved;
                ~span_intent_restore() { value = saved; }
            } restore{cparams.mtp_span_attention, cparams.mtp_span_attention};
            cparams.mtp_span_attention = text_phase == llama_memory_text_phase::decode;
            reserved = reserve_context ? graph_reserve(reserve_tokens, 1, reserve_outputs,
                    reserve_context.get()) : nullptr;
        }
        if (!reserved) {
            LLAMA_LOG_ERROR("%s: failed to rebuild MTP graph after shared scratch handoff\n", __func__);
            return -2;
        }
    }


    if (!group_execution && cparams.kv_streaming() && compute_memory) {
        if (cparams.kv_stream_auxiliary_layers &&
                compute_memory->text_phase().phase != text_phase) {
            auto * hybrid = dynamic_cast<llama_memory_hybrid *>(memory.get());
            auto * target = hybrid ? hybrid->get_mem_attn()->get_kv_stream() : nullptr;
            if (!target || (target->has_mtp_layer() && !target->release_mtp_layer())) {
                LLAMA_LOG_ERROR("%s: failed to retire MTP ring lease before text phase transition\n", __func__);
                return -2;
            }
        }
        const bool supported_verify = (cparams.kv_stream_auxiliary_layers == 1 || cparams.n_rs_seq != 0 || cparams.kv_stream_verify_max) &&
            text_phase == llama_memory_text_phase::decode &&
            n_tokens_all >= 2 && n_tokens_all <= cparams.kv_stream_decode_rows();
        const auto transitions = compute_memory->phase_transition_count();
        const auto phase_result = compute_memory->signal_text_phase({
            text_phase,
            n_tokens_all,
            cparams.n_seq_max == 1,
            cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT,
            cparams.ctx_other != nullptr ||
                (text_phase == llama_memory_text_phase::decode && n_tokens_all != 1 && !supported_verify),
        });
        if (phase_result.status != llama_memory_text_phase_status::changed &&
                phase_result.status != llama_memory_text_phase_status::unchanged) {
            LLAMA_LOG_ERROR("%s: rejected text phase transition\n", __func__);
            return -2;
        }
        if (compute_memory->phase_transition_count() != transitions || shared_workspace_replanned) {
            auto reserve_context = memory->init_full();
            const uint32_t decode_width = std::min(cparams.kv_stream_decode_rows(),std::min(cparams.n_ctx,cparams.n_ubatch));
            const uint32_t reserve_tokens = text_phase == llama_memory_text_phase::decode ?
                decode_width : std::min(cparams.n_ctx,cparams.n_ubatch);
            const uint32_t reserve_outputs = text_phase == llama_memory_text_phase::decode ?
                std::min(decode_width, cparams.n_outputs_max) : std::min(reserve_tokens,cparams.n_outputs_max);
            if (!reserve_context || !graph_reserve(
                    reserve_tokens,cparams.n_seq_max,reserve_outputs,
                    reserve_context.get())) {
                LLAMA_LOG_ERROR("%s: failed to reserve graph after text phase transition\n",__func__);
                return -2;
            }
            llama_log_memory_phase(compute_memory.get());
        }
    }
    if (compute_memory) compute_memory_graph_revision = compute_memory->graph_binding_revision();

    bool did_optimize = false;

    // handle any pending shifts/copies
    memory_update(false);

    llama_memory_context_ptr mctx;

    while (true) {
        mctx = memory->init_batch(*balloc, cparams.n_ubatch, output_all);
        if (!mctx) {
            return -2;
        }

        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    LLAMA_LOG_ERROR("%s: unexpected memory context status: %d\n", __func__, mctx->get_status());

                    return -2;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
                {
                    if (!did_optimize) {
                        did_optimize = true;

                        if (memory_update(true)) {
                            LLAMA_LOG_DEBUG("%s: retrying batch size %d after cache optimization\n", __func__, balloc->get_n_tokens());

                            continue;
                        }
                    }

                    LLAMA_LOG_WARN("%s: failed to find a memory slot for batch of size %d\n", __func__, balloc->get_n_tokens());

                    return 1;
                }
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: compute failed while preparing batch of size %d\n", __func__, balloc->get_n_tokens());

                    return -2;
                }
        }

        break;
    }

    // reserve output buffer
    if (output_reserve(n_outputs_all) < n_outputs_all) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
        return -2;
    };

    // start a new sampling transaction for this logical batch
    for (const auto & entry : sampling.samplers) {
        llama_sampler_backend_begin(entry.second);
    }

    int64_t n_outputs_prev = 0;
    int64_t n_tokens_prev  = 0;

    do {
        const auto & ubatch = mctx->get_ubatch();

        // count the outputs in this ubatch
        {
            int32_t n_outputs_new = 0;

            if (n_outputs_all == n_tokens_all) {
                n_outputs_new = ubatch.n_tokens;
            } else {
                for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
                    n_outputs_new += (int32_t) (ubatch.output[i] != 0);
                }
            }

            // needs to happen before the graph is built
            n_outputs = n_outputs_new;
        }

        ggml_status status;

        const auto * res = process_ubatch(
            ubatch, ctx_type_to_graph_type(cparams.ctx_type), mctx.get(), text_phase, status);

        if (!res) {
            if (group_execution) group_execution->finish(false);
            // the last ubatch failed or was aborted -> remove all positions of that ubatch from the memory module
            llama_pos pos_min[LLAMA_MAX_SEQ];
            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                pos_min[s] = std::numeric_limits<llama_pos>::max();
            }

            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                const auto & seq_id = ubatch.seq_id[i][0];

                pos_min[seq_id] = std::min(pos_min[seq_id], ubatch.pos[i]);
            }

            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                if (pos_min[s] == std::numeric_limits<llama_pos>::max()) {
                    continue;
                }

                LLAMA_LOG_WARN("%s: removing memory module entries for seq_id = %d, pos = [%d, +inf)\n", __func__, s, pos_min[s]);

                memory->seq_rm(s, pos_min[s], -1);
            }

            switch (status) {
                case GGML_STATUS_ABORTED:      return  2;
                case GGML_STATUS_ALLOC_FAILED: return -2;
                case GGML_STATUS_FAILED:       return -3;
                case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
            }
        }

        // plot the computation graph in dot format (for debugging purposes)
        //if (n_past%100 == 0) {
        //    ggml_graph_dump_dot(gf, NULL, "llama.dot");
        //}

        auto * t_logits  = res->get_logits();
        auto * t_embd    = cparams.embeddings       ? res->get_embd()     : nullptr;
        auto * t_h_nextn = cparams.embeddings_nextn ? res->get_h_nextn()  : nullptr;

        if (t_embd && res->get_embd_pooled()) {
            t_embd = res->get_embd_pooled();
        }

        // extract logits
        if (logits.data && t_logits && n_outputs > 0 && needs_raw_logits(ubatch, sampling.samplers)) {
            ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
            GGML_ASSERT(backend_res != nullptr);
            GGML_ASSERT(logits.data != nullptr);

            float * logits_out = logits.data + n_outputs_prev*n_vocab;

            if (n_outputs) {
                GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                GGML_ASSERT((n_outputs_prev + n_outputs)*n_vocab <= (int64_t) logits.size);
                ggml_backend_tensor_get_async(backend_res, t_logits, logits_out, 0, n_outputs*n_vocab*sizeof(float));
            }
        }

        // extract embeddings
        if (embd.data && t_embd && n_outputs > 0) {
            ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
            GGML_ASSERT(backend_embd != nullptr);

            switch (cparams.pooling_type) {
                case LLAMA_POOLING_TYPE_NONE:
                    {
                        // extract token embeddings
                        GGML_ASSERT(embd.data != nullptr);
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        float * embd_out = embd.data + n_outputs_prev*n_embd_out;

                        if (n_outputs) {
                            GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                            GGML_ASSERT((n_outputs_prev + n_outputs)*n_embd_out <= (int64_t) embd.size);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_out, 0, n_outputs*n_embd_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_MEAN:
                case LLAMA_POOLING_TYPE_CLS:
                case LLAMA_POOLING_TYPE_LAST:
                    {
                        // extract sequence embeddings (cleared before processing each batch)
                        auto & embd_seq_out = embd_seq;

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out();

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_embd_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd_out*seq_idx)*sizeof(float), n_embd_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_RANK:
                    {
                        // extract the rerank score - n_cls_out floats per sequence
                        auto & embd_seq_out = embd_seq;

                        const uint32_t n_cls_out = hparams.n_cls_out;

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_cls_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_UNSPECIFIED:
                    {
                        GGML_ABORT("unknown pooling type");
                    }
            }
        }

        // [TAG_EXTRACT_TARGET_EMBEDDINGS]
        bool extract_all_idxs = extract_layer_inputs(res, n_tokens_prev, ubatch.n_tokens);

        // extract nextn embeddings before
        // only meaningful in LLAMA_POOLING_TYPE_NONE (per-token); other pooling modes are ignored.
        {
            const bool masked    = cparams.embeddings_nextn_masked;
            const int64_t n_rows = masked ? n_outputs       : (int64_t) ubatch.n_tokens;
            const int64_t offset = masked ? n_outputs_prev  : n_tokens_prev;

            if (embd_nextn.data && t_h_nextn && n_rows > 0 && cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
                ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched.get(), t_h_nextn);
                GGML_ASSERT(backend_h != nullptr);

                const uint32_t n_embd  = hparams.n_embd_out();
                float * embd_nextn_out = embd_nextn.data + offset*n_embd;

                GGML_ASSERT((offset + n_rows)*n_embd <= (int64_t) embd_nextn.size);
                ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn_out, 0, n_rows*n_embd*sizeof(float));
                extract_all_idxs = extract_all_idxs || !masked;
            }
        }

        if (extract_all_idxs) {
            GGML_ASSERT(ubatch.data && ubatch.data->batch_idxs.size() == ubatch.n_tokens);
            GGML_ASSERT(embd_batch_idxs.size() == (size_t) n_tokens_prev);
            const auto & batch_idxs = ubatch.data->batch_idxs;
            embd_batch_idxs.insert(embd_batch_idxs.end(), batch_idxs.begin(), batch_idxs.end());
        }

        if (has_samplers) {
            const auto stride = n_vocab;

            // async copy the sampling data from the backend to the host
            copy_tensor_async_rows(res->t_sampled,        sampling.sampled,    1,      n_outputs_prev, sched.get());
            copy_tensor_async_rows(res->t_sampled_logits, sampling.logits,     stride, n_outputs_prev, sched.get(), &sampling.logits_count);
            copy_tensor_async_rows(res->t_sampled_probs,  sampling.probs,      stride, n_outputs_prev, sched.get(), &sampling.probs_count);
            copy_tensor_async_rows(res->t_candidates,     sampling.candidates, stride, n_outputs_prev, sched.get(), &sampling.candidates_count);
        }

        n_outputs_prev += n_outputs;
        n_tokens_prev  += ubatch.n_tokens;
    } while (mctx->next());

    // set to total number of outputs in the batch, for use in llama_get_logits_ith
    n_outputs = n_outputs_all;

    // set output mappings
    if (n_outputs > 0) {
        bool sorted_output = true;

        auto & out_ids = balloc->get_out_ids();

        GGML_ASSERT(out_ids.size() == (size_t) n_outputs);

        for (int64_t i = 0; i < n_outputs; ++i) {
            int64_t out_id = out_ids[i];
            output_ids[out_id] = i;
            if (out_id != i) {
                sorted_output = false;
            }
        }

        // make the outputs have the same order they had in the user-provided batch
        // note: this is mostly relevant for recurrent models atm
        if (!sorted_output && n_outputs > 1) {
            GGML_ASSERT((size_t) n_outputs == out_ids.size());

            // TODO: is there something more efficient which also minimizes swaps?
            // selection sort, to minimize swaps (from https://en.wikipedia.org/wiki/Selection_sort)
            for (uint32_t i = 0; i < n_outputs - 1; ++i) {
                uint32_t j_min = i;
                for (uint32_t j = i + 1; j < n_outputs; ++j) {
                    if (out_ids[j] < out_ids[j_min]) {
                        j_min = j;
                    }
                }
                if (j_min == i) {
                    continue;
                }
                std::swap(out_ids[i], out_ids[j_min]);

                // remember the swaps and apply them lazily upon logits/embeddings access
                output_swaps.push_back({ i, j_min });
            }

            std::fill(output_ids.begin(), output_ids.end(), -1);

            for (uint32_t i = 0; i < n_outputs; ++i) {
                output_ids[out_ids[i]] = i;
            }
        }
    }

    // wait for the computation to finish (automatically done when obtaining the model output)
    //synchronize();

    if (serial_feature_producer) {
        serial_feature_rows.reserve(batch_inp.tokens.size());
        for (const auto & token : batch_inp.tokens)
            serial_feature_rows.push_back({token.id,token.pos[0],token.seq_ids.size()==1 ? *token.seq_ids.begin() : -1});
        if (group_execution) for (const auto & handle : group_execution->projection->handles) {
            std::shared_ptr<llama_kv_stream_logical_cache> cache;
            if (!group_execution->table->get(handle,group_execution->projection->role,cache) || !cache->ready()) {
                clear_serial_features();return -2;
            }
            serial_feature_sources.push_back({handle.slot,handle.generation,cache->identity().id,cache->identity().generation,cache->tokens()});
        }
        if (serial_feature_images && !batch_inp.embd.empty()) {
            serial_feature_image_rows.reserve(batch_inp.tokens.size());
            for (const auto & token : batch_inp.tokens)
                serial_feature_image_rows.push_back({token.has_embd,token.embd_off,token.pos});
            serial_feature_image_values=batch_inp.embd;
            serial_feature_image_width=batch_inp.n_embd;
        }
    }

    if (group_execution && !group_execution->finish(true)) return -2;
    return 0;
}

//
// output
//

uint32_t llama_context::output_reserve(int32_t n_outputs) {
    const auto & hparams = model.hparams;
    const auto & vocab   = model.vocab;

    // Configured slots can exceed the number of output rows permitted in one batch.
    const int64_t n_outputs_max = std::max<int64_t>(n_outputs, std::min(n_seq_max(), cparams.n_outputs_max));

    const auto n_batch    = cparams.n_batch;
    const auto n_vocab    = vocab.n_tokens();
    const auto n_embd     = hparams.n_embd;
    const auto n_embd_out = hparams.n_embd_out();

    bool has_logits     = true;
    bool has_embd       = cparams.embeddings;
    bool has_embd_nextn = cparams.embeddings_nextn;

    // TODO: hacky enc-dec support
    if (model.arch == LLM_ARCH_T5) {
        has_logits = true;
        has_embd   = true;
    }

    size_t backend_float_count = 0;
    size_t backend_token_count = 0;
    size_t embd_layer_inp_float_count = 0;

    logits.size     = has_logits     ? n_vocab*n_outputs_max     : 0;
    embd.size       = has_embd       ? n_embd_out*n_outputs_max  : 0;
    embd_nextn.size = has_embd_nextn ? n_embd_out*n_outputs_max  : 0;

    if (has_embd_nextn && !cparams.embeddings_nextn_masked) {
        // unmasked: nextn row exists for every token in the batch, not just
        // those flagged via batch.logits[i] -> size by token count instead.
        embd_nextn.size = (size_t) n_embd_out * n_batch;
    }

    for (bool enabled : cparams.embeddings_layer_inp) {
        if (enabled) {
            embd_layer_inp_float_count += (size_t) n_embd * n_batch;
        }
    }

    // Allocate backend sampling output buffers if there are backend samplers configured.
    const bool has_sampling = !sampling.samplers.empty();
    if (has_sampling) {
        backend_float_count = 2 * n_vocab * n_outputs_max;      // logits + probs
        backend_token_count = (1 + n_vocab) * n_outputs_max;    // sampled + candidates
    }

    if (output_ids.empty()) {
        // init, never resized afterwards
        output_ids.resize(n_batch);
    }

    const size_t prev_size = buf_output ? ggml_backend_buffer_get_size(buf_output.get()) : 0;
    const size_t new_size  =
        (logits.size + embd.size + embd_nextn.size + embd_layer_inp_float_count + backend_float_count) * sizeof(float) +
        (                                                                         backend_token_count) * sizeof(llama_token);

    // alloc only when more than the current capacity is required
    // TODO: also consider shrinking the buffer
    if (!buf_output || prev_size < new_size) {
        if (buf_output) {
#ifndef NDEBUG
            // This doesn't happen often, but may be annoying in some cases (like the HellaSwag benchmark)
            LLAMA_LOG_DEBUG("%s: reallocating output buffer from size %.02f MiB to %.02f MiB\n", __func__, prev_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
#endif
            synchronize();

            // TODO: not needed?
            buf_output = nullptr;
            logits.data = nullptr;
            embd.data = nullptr;
            embd_nextn.data = nullptr;
            for (auto & layer_inp : embd_layer_inp) {
                layer_inp = {nullptr, 0};
            }
        }

        auto * buft = ggml_backend_cpu_buffer_type();
        // try to use the host buffer of the device where the output tensor is allocated for faster transfer to system memory
        auto * output_dev = model.dev_output();
        auto * output_dev_host_buft = output_dev ? ggml_backend_dev_host_buffer_type(output_dev) : nullptr;
        if (output_dev_host_buft) {
            buft = output_dev_host_buft;
        }
        buf_output.reset(ggml_backend_buft_alloc_buffer(buft, new_size));
        if (buf_output == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to allocate output buffer of size %.2f MiB\n", __func__, new_size / (1024.0 * 1024.0));
            return 0;
        }
        ggml_backend_buffer_clear(buf_output.get(), 0);
    }

    float * output_base = (float *) ggml_backend_buffer_get_base(buf_output.get());

    size_t offset = 0;
    uint8_t * base = (uint8_t *) output_base;

    logits = has_logits ? buffer_view<float>{output_base, logits.size} : buffer_view<float>{nullptr, 0};
    offset += logits.size * sizeof(float);

    embd = has_embd ? buffer_view<float>{(float *) (base + offset), embd.size} : buffer_view<float>{nullptr, 0};
    offset += embd.size * sizeof(float);

    embd_nextn = has_embd_nextn ? buffer_view<float>{(float *) (base + offset), embd_nextn.size} : buffer_view<float>{nullptr, 0};
    offset += embd_nextn.size * sizeof(float);

    for (uint32_t il = 0; il < embd_layer_inp.size(); ++il) {
        if (cparams.embeddings_layer_inp[il]) {
            embd_layer_inp[il] = buffer_view<float>{(float *) (base + offset), (size_t) n_embd * n_batch};
            offset += embd_layer_inp[il].size * sizeof(float);
        } else {
            embd_layer_inp[il] = buffer_view<float>{nullptr, 0};
        }
    }

    if (has_sampling) {
        sampling.logits = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.logits.size * sizeof(float);

        sampling.probs = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.probs.size * sizeof(float);

        sampling.sampled = {(llama_token *) (base + offset), (size_t)n_outputs_max};
        offset += sampling.sampled.size * sizeof(llama_token);

        sampling.candidates = {(llama_token *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.candidates.size * sizeof(llama_token);

        // The count vectors keep track of the actual number of logits/probs/candidates
        // copied from the backend for each output row.

        sampling.logits_count.resize(n_outputs_max);
        sampling.probs_count.resize(n_outputs_max);
        sampling.candidates_count.resize(n_outputs_max);

        std::fill(sampling.logits_count.begin(),     sampling.logits_count.end(),     0);
        std::fill(sampling.probs_count.begin(),      sampling.probs_count.end(),      0);
        std::fill(sampling.candidates_count.begin(), sampling.candidates_count.end(), 0);

        std::fill_n(sampling.sampled.data, sampling.sampled.size, LLAMA_TOKEN_NULL);
    } else {
        sampling.logits     = {nullptr, 0};
        sampling.probs      = {nullptr, 0};
        sampling.sampled    = {nullptr, 0};
        sampling.candidates = {nullptr, 0};

        sampling.logits_count.clear();
        sampling.probs_count.clear();
        sampling.candidates_count.clear();
    }

    // set all ids as invalid (negative)
    std::fill(output_ids.begin(), output_ids.end(), -1);

    this->n_outputs = 0;

    GGML_ASSERT(n_outputs_max <= cparams.n_outputs_max);

    return n_outputs_max;
}

bool llama_context::extract_layer_inputs(const llm_graph_result * res, size_t token_offset, size_t n_tokens) {
    bool extracted = false;
    for (uint32_t il = 0; il < cparams.embeddings_layer_inp.size(); ++il) {
        if (!cparams.embeddings_layer_inp[il]) {
            continue;
        }
        if (!embd_layer_inp[il].has_data()) {
            GGML_ABORT("output layer input buffer not allocated");
        }
        ggml_tensor * t = res->get_layer_inp((int) il);
        if (!t) {
            GGML_ABORT("layer input tensor not found");
        }

        const size_t nbytes = ggml_nbytes(t);
        const size_t nfloats = nbytes / sizeof(float);
        GGML_ASSERT(n_tokens > 0);
        GGML_ASSERT(nfloats % n_tokens == 0);

        const size_t row_floats = nfloats / n_tokens;
        GGML_ASSERT(row_floats == model.hparams.n_embd);
        const size_t dst_offset = token_offset * row_floats;
        GGML_ASSERT(dst_offset + nfloats <= embd_layer_inp[il].size);

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched.get(), t);
        GGML_ASSERT(backend != nullptr);
        // Tensor-split backends require a zero source offset.
        ggml_backend_tensor_get_async(backend, t, embd_layer_inp[il].data + dst_offset, 0, nbytes);
        extracted = true;
    }
    return extracted;
}

void llama_context::output_reorder() {
    const uint64_t n_vocab     = model.vocab.n_tokens();
    const uint64_t n_embd      = model.hparams.n_embd;
    const uint64_t n_embd_out  = model.hparams.n_embd_out();

    for (size_t s = 0; s < output_swaps.size(); ++s) {
        const uint64_t i0 = output_swaps[s].i0;
        const uint64_t i1 = output_swaps[s].i1;

        if (logits.size > 0) {
            for (uint64_t k = 0; k < n_vocab; k++) {
                std::swap(logits.data[i0*n_vocab + k], logits.data[i1*n_vocab + k]);
            }
        }

        if (embd.size > 0) {
            for (uint64_t k = 0; k < n_embd_out; k++) {
                std::swap(embd.data[i0*n_embd_out + k], embd.data[i1*n_embd_out + k]);
            }
        }

        if (embd_nextn.size > 0 && cparams.embeddings_nextn_masked) {
            for (uint64_t k = 0; k < n_embd_out; k++) {
                std::swap(embd_nextn.data[i0*n_embd_out + k], embd_nextn.data[i1*n_embd_out + k]);
            }
        }

        if (!sampling.samplers.empty()) {
            assert(sampling.logits.size > 0);
            assert(sampling.probs.size > 0);
            assert(sampling.candidates.size > 0);
            assert(sampling.sampled.size > 0);
            assert(sampling.logits_count.size() > 0);
            assert(sampling.probs_count.size() > 0);
            assert(sampling.candidates_count.size() > 0);

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.logits.data[i0*n_vocab + k], sampling.logits.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.probs.data[i0*n_vocab + k], sampling.probs.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.candidates.data[i0*n_vocab + k], sampling.candidates.data[i1*n_vocab + k]);
            }

            std::swap(sampling.sampled.data[i0],     sampling.sampled.data[i1]);
            std::swap(sampling.logits_count[i0],     sampling.logits_count[i1]);
            std::swap(sampling.probs_count[i0],      sampling.probs_count[i1]);
            std::swap(sampling.candidates_count[i0], sampling.candidates_count[i1]);
        }
    }

    output_swaps.clear();

    // [TAG_EXTRACT_TARGET_EMBEDDINGS]
    // Layer inputs and unmasked NextN embeddings contain all token rows, independent of logits selection.
    for (size_t i = 0; i < embd_batch_idxs.size(); ++i) {
        while (embd_batch_idxs[i] != (int32_t) i) {
            const int32_t j = embd_batch_idxs[i];
            GGML_ASSERT(j >= 0 && (size_t) j < embd_batch_idxs.size());
            if (embd_nextn.has_data() && !cparams.embeddings_nextn_masked) {
                for (size_t k = 0; k < n_embd_out; ++k) {
                    std::swap(embd_nextn.data[i*n_embd_out + k], embd_nextn.data[j*n_embd_out + k]);
                }
            }
            for (auto & layer : embd_layer_inp) {
                if (layer.has_data()) {
                    for (size_t k = 0; k < n_embd; ++k) {
                        std::swap(layer.data[i*n_embd + k], layer.data[j*n_embd + k]);
                    }
                }
            }
            std::swap(embd_batch_idxs[i], embd_batch_idxs[j]);
        }
    }
    embd_batch_idxs.clear();
}

//
// graph
//

uint32_t llama_context::graph_max_nodes(uint32_t n_tokens) const {
    uint32_t res;
    if (model.arch == LLM_ARCH_KIMI_K3 || model.arch == LLM_ARCH_GLM5_NEXT) {
        // the n_tokens*40 budget below is exhausted at ubatch 3840
        res = std::max<uint32_t>(n_tokens * 160, 64u * model.n_tensors());
    } else if (model.arch == LLM_ARCH_HRM_TEXT) {
        // the 128-slot looped graph needs roughly one stack per token budget
        res = std::max<uint32_t>(n_tokens * 80, 64u * model.n_tensors());
    } else if (model.arch == LLM_ARCH_QWEN3NEXT ||
        model.arch == LLM_ARCH_KIMI_LINEAR ||
        model.arch == LLM_ARCH_BAILINGMOE3 ||
        model.arch == LLM_ARCH_QWEN35 ||
        model.arch == LLM_ARCH_QWEN35MOE ||
        model.arch == LLM_ARCH_CLEF ||
        model.arch == LLM_ARCH_QWEN4EXP ||
        model.arch == LLM_ARCH_DEEPSEEK4 ||
        (model.arch == LLM_ARCH_DFLASH && model.hparams.dsv4_hc_mult > 0) ||
        model.arch == LLM_ARCH_NANBEIGE ||
        model.arch == LLM_ARCH_MINIMAX_01 ||
        model.arch == LLM_ARCH_MINIMAX_M3 ||
        model.arch == LLM_ARCH_HY_V4) {
        res = std::max<uint32_t>(n_tokens * 40, 32u * model.n_tensors());
    } else if (model.arch == LLM_ARCH_DFLASH && model.hparams.dflash_selector_rank > 0) {
        // DFlash2's convolutions and selector are shape work rather than matmuls,
        // so they cost ~8.6 nodes per tensor against ~5.9 for a plain DFlash draft
        res = std::max<uint32_t>(1024u, 12u*model.n_tensors());
    } else {
        res = std::max<uint32_t>(1024u, 8u*model.n_tensors());
        for (const auto & lora : model.loras) {
            res += lora->get_n_nodes();
        }
    }

    // every sampler builds n_outputs_max_per_seq chains, see llm_graph_context::build_sampling
    for (const auto & [seq_id, sampler] : sampling.samplers) {
        res += llama_sampler_backend_n_nodes(sampler) * cparams.n_outputs_max_per_seq;
    }

    if (cparams.training) {
        res *= 4;
    }

    return res;
}

llm_graph_result * llama_context::get_gf_res_reserve() const {
    return static_cast<llm_graph_result *>(gf_res_reserve.get());
}

llm_graph_result * llama_context::get_gf_res_prev() {
    auto & res = gf_res_prev[n_outputs > 0];
    if (!res) {
        res.reset(new llm_graph_result(gf_res_reserve->get_max_nodes()));
    }
    return res.get();
}

// pack sampler outputs into as few sequences as possible before using sequences without samplers
static void ubatch_prepare_reserve(
              llama_ubatch                            & ubatch,
              uint32_t                                  n_outputs,
        const std::map<llama_seq_id, llama_sampler *> & samplers,
              uint32_t                                  n_outputs_max_per_seq) {
    const uint32_t n_seqs       = ubatch.n_seqs;
    const uint32_t n_seq_tokens = ubatch.n_seq_tokens;

    for (uint32_t s = 0; s < n_seqs; ++s) {
        for (uint32_t t = 0; t < n_seq_tokens; ++t) {
            const uint32_t i = s * n_seq_tokens + t;
            ubatch.n_seq_id[i] = 1;
            ubatch.seq_id[i] = &ubatch.seq_id_unq[s];
        }
    }

    // sequences with a sampler that fit in this ubatch
    std::vector<uint32_t> sampler_seqs;
    std::vector<bool> has_sampler(n_seqs, false);
    for (const auto & entry : samplers) {
        const auto found=std::find(ubatch.seq_id_unq,ubatch.seq_id_unq+n_seqs,entry.first);
        if (found==ubatch.seq_id_unq+n_seqs) continue;
        const auto index=uint32_t(found-ubatch.seq_id_unq);
        sampler_seqs.push_back(index);
        has_sampler[index] = true;
    }

    uint32_t n_outputs_set = 0;

    const uint32_t n_outputs_per_seq = std::min(n_seq_tokens, n_outputs_max_per_seq);
    for (uint32_t s : sampler_seqs) {
        if (n_outputs_set >= n_outputs) {
            break;
        }

        for (uint32_t t = 0; t < n_outputs_per_seq && n_outputs_set < n_outputs; ++t) {
            ubatch.output[s * n_seq_tokens + t] = true;
            ++n_outputs_set;
        }
    }

    // use sequences without samplers for any remaining outputs
    for (uint32_t t = 0; t < n_seq_tokens && n_outputs_set < n_outputs; ++t) {
        for (uint32_t s = 0; s < n_seqs && n_outputs_set < n_outputs; ++s) {
            if (has_sampler[s]) {
                continue;
            }

            ubatch.output[s * n_seq_tokens + t] = true;
            ++n_outputs_set;
        }
    }
}

llama_context_memory_create_result llama_context::quote_target_group(const llama_batch & batch,
        const std::vector<llama_memory_phase_cache_identity> & sources,
        llama_memory_phase_cache_identity bounded_identity,llama_memory_work_phase phase,
        uint64_t id,size_t capacity,llama_context_group_plan & output,llama_context * auxiliary,
        const std::vector<llama_memory_phase_cache_identity> & auxiliary_sources) {
    if (auxiliary && auxiliary->model.arch==LLM_ARCH_DFLASH) {
        if (!auxiliary_sources.empty()) {llama_context_memory_create_result result;result.available_bytes=capacity;return result;}
        return quote_dflash_group(batch,sources,bounded_identity,phase,id,capacity,output,*auxiliary);
    }
    if (auxiliary) return quote_mtp_group(batch,sources,bounded_identity,phase,id,capacity,output,*auxiliary,auxiliary_sources);
    llama_context_memory_create_result result;
    result.available_bytes=capacity;
    if (!auxiliary_sources.empty()) return result;
    try {
        llama_context_group_plan next;
        if (!measure_target_workload(batch,sources,phase,id,next.target)) {
            LLAMA_LOG_ERROR("%s: native group workload could not be measured\n",__func__);return result;
        }
        std::vector<llama_context_group_workload_quote> catalog;
        const uint32_t verify_rows=group_verify_rows(cparams);
        if (group_initializing && phase==llama_memory_work_phase::target_verify && verify_rows>1) {
            if (sources.size()>2) {
                const auto maximum=std::min(cparams.n_batch,cparams.n_ubatch);
                auto input=llama_batch_init(int32_t(maximum),0,1);
                struct release_batch {llama_batch & batch;~release_batch() {llama_batch_free(batch);}} release{input};
                uint64_t profile_id=1;
                if (!llama_context_group_graph_profiles(uint32_t(sources.size()),verify_rows,maximum,
                        cparams.n_outputs_max,cparams.n_outputs_max_per_seq,kv_resources.recurrent!=nullptr,
                        [&](const llama_context_group_graph_profile & profile) {
                    input.n_tokens=0;
                    for (size_t i=0;i<profile.rows.size();++i) for (uint32_t row=0;row<profile.rows[i];++row) {
                        const auto token=input.n_tokens++;
                        input.token[token]=0;input.pos[token]=llama_pos(row);input.n_seq_id[token]=1;
                        input.seq_id[token][0]=group_selected[i];
                        input.logits[token]=row+profile.outputs[i]>=profile.rows[i];
                    }
                    llama_context_group_workload_quote measured;
                    if (!measure_target_workload(input,sources,phase,profile_id++,measured)) return false;
                    catalog.push_back(std::move(measured));return true;
                })) return result;
            } else {
            struct option {uint32_t rows,outputs;};
            std::vector<option> options{{0,0}};
            for (uint32_t rows=1;rows<=verify_rows;++rows) {
                options.push_back({rows,1});
                const auto outputs=std::min(rows,cparams.n_outputs_max_per_seq);
                if (outputs>1) options.push_back({rows,outputs});
            }
            size_t combinations=1;
            for (size_t i=0;i<sources.size();++i) {
                if (combinations>SIZE_MAX/options.size()) return result;
                combinations*=options.size();
            }
            const auto maximum=uint32_t(std::min(uint64_t(cparams.n_batch),uint64_t(sources.size())*verify_rows));
            auto input=llama_batch_init(int32_t(maximum),0,1);
            struct release_batch {llama_batch & batch;~release_batch() {llama_batch_free(batch);}} release{input};
            for (size_t profile=1;profile<combinations;++profile) {
                size_t value=profile;
                std::vector<option> selected;
                uint64_t rows=0,outputs=0;
                for (size_t i=0;i<sources.size();++i) {
                    const auto shape=options[value%options.size()];value/=options.size();
                    selected.push_back(shape);rows+=shape.rows;outputs+=shape.outputs;
                }
                if (!rows || rows>maximum || outputs>cparams.n_outputs_max) continue;
                input.n_tokens=0;
                for (size_t i=0;i<selected.size();++i) for (uint32_t row=0;row<selected[i].rows;++row) {
                    const auto token=input.n_tokens++;
                    input.token[token]=0;input.pos[token]=llama_pos(row);input.n_seq_id[token]=1;
                    input.seq_id[token][0]=group_selected.empty() ? llama_seq_id(i) : group_selected[i];
                    input.logits[token]=row+selected[i].outputs>=selected[i].rows;
                }
                llama_context_group_workload_quote measured;
                if (!measure_target_workload(input,sources,phase,profile,measured)) return result;
                catalog.push_back(std::move(measured));
            }
            }
            if (catalog.empty()) return result;
            const auto valid_verify=[](const llama_context_group_workload_quote & quote) {
                return std::all_of(quote.measurement.workload.members.begin(),quote.measurement.workload.members.end(),
                    [](const auto & member) {return member.output_rows>0;});
            };
            const auto total_rows=[&](const llama_context_group_workload_quote & quote) {
                if (!valid_verify(quote)) return uint64_t(0);
                uint64_t total=0;
                for (const auto & member : quote.measurement.workload.members) total+=member.input_rows;
                return total;
            };
            next.target=*std::max_element(catalog.begin(),catalog.end(),[&](const auto & a,const auto & b) {
                return total_rows(a)<total_rows(b);
            });
            next.target.measurement.workload.id=id;
            size_t graph=0;
            for (const auto & measured : catalog) {
                if (measured.buffer_type!=next.target.buffer_type || measured.writer_bytes!=next.target.writer_bytes ||
                        measured.encoded_stage_bytes!=next.target.encoded_stage_bytes ||
                        measured.pool_minimum_bytes!=next.target.pool_minimum_bytes ||
                        measured.measurement.alignment!=next.target.measurement.alignment) return result;
                for (auto bytes : measured.measurement.graph_bytes) graph=std::max(graph,bytes);
                next.target.attention_bytes=std::max(next.target.attention_bytes,measured.attention_bytes);
                next.target.publication_bytes=std::max(next.target.publication_bytes,measured.publication_bytes);
                next.target.host_compute_bytes=std::max(next.target.host_compute_bytes,measured.host_compute_bytes);
            }
            // Fixed views must cover every declared profile before a child can borrow them.
            std::fill(next.target.measurement.graph_bytes.begin(),next.target.measurement.graph_bytes.end(),graph);
            next.target.measurement.live_device_bytes=0;next.target.alignment_slack_bytes=0;
            const auto alignment=next.target.measurement.alignment;
            const auto reserve=[&](size_t bytes) {
                if (bytes>SIZE_MAX-(alignment-1)) return false;
                const auto padded=(bytes+alignment-1)&~(alignment-1);
                if (padded>SIZE_MAX-next.target.measurement.live_device_bytes) return false;
                next.target.measurement.live_device_bytes+=padded;next.target.alignment_slack_bytes+=padded-bytes;return true;
            };
            for (auto bytes : {next.target.writer_bytes,next.target.attention_bytes,next.target.encoded_stage_bytes,next.target.publication_bytes})
                if (!reserve(bytes)) return result;
            for (size_t i=0;i<sources.size();++i)
                if (!reserve(next.target.pool_minimum_bytes/sources.size())) return result;
            if (sources.size()>2) for (auto & item : catalog)
                std::fill(item.measurement.graph_bytes.begin(),item.measurement.graph_bytes.end(),graph);
            // Zero-output native fragments contribute to the envelope, not to whole verification intents.
            catalog.erase(std::remove_if(catalog.begin(),catalog.end(),[&](const auto & item) {return !valid_verify(item);}),catalog.end());
            if (catalog.empty()) return result;
        } else catalog.push_back(next.target);
        const auto layers=kv_resources.full->get_layer_ids();
        if (layers.empty()) return result;
        const auto first=layers.front();
        for (auto identity : sources) {
            llama_memory_cache_requirement cache;
            cache.identity=identity; cache.layers=uint32_t(layers.size()); cache.history_rows=cparams.n_ctx;
            cache.row_padding=256;
            cache.format={kv_resources.full->get_k_storage(first)->type,kv_resources.full->get_v_storage(first)->type,
                model.hparams.n_embd_head_k(first),model.hparams.n_embd_head_v(first),model.hparams.n_head_kv(first),false};
            const auto key=ggml_row_size(cache.format.key_type,int64_t(cache.format.key_head_dim)*cache.format.kv_heads);
            const auto value=ggml_row_size(cache.format.value_type,int64_t(cache.format.value_head_dim)*cache.format.kv_heads);
            if (key>SIZE_MAX-value) return result;
            cache.row_bytes=key+value;
            llama_memory_cache_bound bound;
            if (!llama_memory_cache_bound_make(cache,nullptr,bound)) return result;
            next.caches.push_back(cache);
            next.cache_allocations.push_back({identity,0,next.target.measurement.alignment,bound.capacity_rows,cache.format});
        }
        if (kv_resources.bounded) {
            llama_memory_params params{};
            params.type_k=kv_resources.bounded->type_k(); params.type_v=kv_resources.bounded->type_v(); params.swa_full=false;
            params.kv_stream_selected_sequences=group_selected;
            if (!llama_kv_cache_iswa::bounded_storage_requirements(model.hparams,params,cparams,
                    next.target.buffer_type,bounded_identity,next.bounded)) return result;
            next.caches.push_back(next.bounded.cache);
            next.cache_allocations.push_back(next.bounded.allocation);
        }
        if (kv_resources.recurrent) {
            ggml_backend_buffer_type_t type=nullptr;
            if (!kv_resources.recurrent->current_storage_requirements(type,next.current_state_bytes) ||
                    type!=next.target.buffer_type) return result;
        }
        std::vector<llama_memory_group_workload> workloads;
        std::vector<llama_memory_group_workload_measurement> measurements;
        for (const auto & measured : catalog) {
            workloads.push_back(measured.measurement.workload);measurements.push_back(measured.measurement);
            measurements.back().live_device_bytes=next.target.measurement.live_device_bytes;
        }
        const auto budget=llama_memory_group_workload_budget_make(next.caches,next.cache_allocations,
            workloads,measurements,next.current_state_bytes,capacity,next.budget);
        if (budget.status==llama_memory_workload_budget_status::insufficient_budget) {
            result.status=llama_context_memory_create_status::insufficient_budget;
            result.required_bytes=capacity+budget.additional_bytes; result.additional_bytes=budget.additional_bytes;
            result.stage=id;
            return result;
        }
        if (budget.status!=llama_memory_workload_budget_status::success) {
            LLAMA_LOG_ERROR("%s: invalid group workload budget (%d)\n",__func__,int(budget.status));return result;
        }
        result.status=llama_context_memory_create_status::success;
        result.required_bytes=next.budget.parent_minimum_bytes;
        next.catalog=std::move(catalog);
        output=std::move(next);
        return result;
    } catch (const std::bad_alloc &) {
        result.status=llama_context_memory_create_status::allocation_failed;
        return result;
    }
}

llama_context_memory_create_result llama_context::quote_mtp_group(const llama_batch & batch,
        const std::vector<llama_memory_phase_cache_identity> & sources,
        llama_memory_phase_cache_identity bounded_identity,llama_memory_work_phase phase,
        uint64_t id,size_t capacity,llama_context_group_plan & output,llama_context & auxiliary,
        const std::vector<llama_memory_phase_cache_identity> & auxiliary_sources) {
    llama_context_memory_create_result result;result.available_bytes=capacity;
    const bool readonly=model.arch==LLM_ARCH_GEMMA4 && auxiliary.model.arch==LLM_ARCH_GEMMA4_ASSISTANT &&
        !cparams.kv_stream_auxiliary_layers && cparams.kv_stream_verify_max && auxiliary.cparams.ctx_other==this;
    const bool owned=model.arch==LLM_ARCH_QWEN35 && &auxiliary.model==&model && cparams.kv_stream_auxiliary_layers==1;
    if (!group_initializing || !auxiliary.group_initializing || (!readonly && !owned) || auxiliary.group_target!=this ||
            auxiliary.cparams.ctx_type!=LLAMA_CONTEXT_TYPE_MTP || auxiliary.cparams.shared_device_memory_bytes ||
            auxiliary.group_selected!=group_selected || sources.size()!=group_selected.size() ||
            (readonly ? !auxiliary_sources.empty() : auxiliary_sources.size()!=sources.size()) || !id || !batch.seq_id || !batch.n_seq_id ||
            !kv_resources.full || !auxiliary.kv_resources.full ||
            kv_resources.full->type_k()!=auxiliary.kv_resources.full->type_k() ||
            kv_resources.full->type_v()!=auxiliary.kv_resources.full->type_v()) return result;
    try {
        llama_context_group_plan next;
        const auto original=quote_target_group(batch,sources,bounded_identity,phase,id,SIZE_MAX,next);
        if (original.status!=llama_context_memory_create_status::success) return result;
        const auto maximum=std::min(cparams.n_batch,cparams.n_ctx);
        if (!maximum || maximum>INT32_MAX) return result;
        auto input=llama_batch_init(int32_t(maximum),0,1);
        struct release_batch {llama_batch & batch;~release_batch() {llama_batch_free(batch);}} release{input};
        const auto fill=[&](const std::vector<uint32_t> & widths,bool predict) {
            input.n_tokens=0;
            for (size_t i=0;i<widths.size();++i) for (uint32_t row=0;row<widths[i];++row) {
                if (uint32_t(input.n_tokens)>=maximum) return false;
                const auto index=input.n_tokens++;
                input.token[index]=0;input.pos[index]=llama_pos(row);input.n_seq_id[index]=1;
                input.seq_id[index][0]=group_selected[i];input.logits[index]=predict;
            }
            return input.n_tokens>0;
        };
        if (phase!=llama_memory_work_phase::target_verify) {
            if (!fill(std::vector<uint32_t>(sources.size(),group_verify_rows(cparams)),true)) return result;
            llama_context_group_plan verify;
            if (quote_target_group(input,sources,bounded_identity,llama_memory_work_phase::target_verify,id,SIZE_MAX,verify).status!=
                    llama_context_memory_create_status::success) return result;
            next.catalog.insert(next.catalog.end(),verify.catalog.begin(),verify.catalog.end());
        }
        const auto measure=[&](const std::vector<uint32_t> & widths,bool predict) {
            if (!fill(widths,predict)) return false;
            llama_context_group_workload_quote measured;
            if (!auxiliary.measure_target_workload(input,readonly ? sources : auxiliary_sources,predict ? llama_memory_work_phase::mtp_draft :
                    llama_memory_work_phase::mtp_catchup,1,measured)) return false;
            next.catalog.push_back(std::move(measured));return true;
        };
        std::vector<uint32_t> actual(sources.size(),0),full(sources.size(),maximum/uint32_t(sources.size()));
        for (int32_t row=0;row<batch.n_tokens;++row) {
            if (batch.n_seq_id[row]!=1 || !batch.seq_id[row]) return result;
            const auto found=std::find(group_selected.begin(),group_selected.end(),batch.seq_id[row][0]);
            if (found==group_selected.end()) return result;
            ++actual[size_t(found-group_selected.begin())];
        }
        for (size_t i=0;i<maximum%sources.size();++i) ++full[i];
        if (!readonly && (!measure(actual,false) || !measure(full,false))) return result;
        if (sources.size()>2) {
            if (!readonly) for (uint32_t rows=1;rows<=std::min(maximum,auxiliary.cparams.n_ubatch);++rows) {
                std::vector<uint32_t> widths(sources.size());widths.front()=rows;
                if (!measure(widths,false)) return result;
            }
            for (uint32_t active=1;active<=std::min(uint32_t(sources.size()),auxiliary.cparams.n_ubatch);++active) {
                std::vector<uint32_t> widths(sources.size());std::fill_n(widths.begin(),active,1);
                if (!measure(widths,true)) return result;
            }
        } else {
        size_t combinations=1;
        for (size_t i=0;i<sources.size();++i) combinations*=5;
        for (size_t profile=1;!readonly && profile<combinations;++profile) {
            size_t value=profile;std::vector<uint32_t> widths(sources.size());
            for (auto & width : widths) {width=uint32_t(value%5);value/=5;}
            if (!measure(widths,false)) return result;
        }
        for (size_t active=1;active<(size_t(1)<<sources.size());++active) {
            std::vector<uint32_t> widths(sources.size());
            for (size_t i=0;i<widths.size();++i) widths[i]=uint32_t((active>>i)&1);
            if (!measure(widths,true)) return result;
        }
        }
        const auto child_layers=auxiliary.kv_resources.full->get_layer_ids();
        if (readonly ? child_layers.empty() || next.caches.size()!=sources.size()+1 :
                child_layers.size()!=1 || next.caches.size()!=sources.size()) return result;
        for (size_t i=0;!readonly && i<sources.size();++i) {
            auto cache=next.caches[i];cache.identity=auxiliary_sources[i];cache.layers=uint32_t(child_layers.size());
            llama_memory_cache_bound bound;
            if (!llama_memory_cache_bound_make(cache,nullptr,bound)) return result;
            next.caches.push_back(cache);
            next.cache_allocations.push_back({cache.identity,0,next.target.measurement.alignment,bound.capacity_rows,cache.format});
        }
        auto * device=model.devices.front().dev;ggml_backend_t backend=nullptr;
        for (auto * candidate : backend_ptrs) if (ggml_backend_get_device(candidate)==device) backend=candidate;
        const auto get=reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(device),"ggml_backend_kv_stream_partial_ops"));
        const auto * ops=get ? get() : nullptr;
        if (!backend || !ops || !ops->capabilities) return result;
        const auto & format=next.caches.front().format;
        llama_kv_stream_policy_config policy;
        policy.shape={format.key_type,format.value_type,int32_t(format.key_head_dim),int32_t(format.value_head_dim),
            int32_t(format.kv_heads),256,128};
        policy.capabilities=ops->capabilities(backend,format.key_type,format.value_type);
        policy.layers=next.caches.front().layers+(readonly ? 0 : uint32_t(child_layers.size()));
        size_t one_pool=0;
        if (llama_kv_stream_policy_minimum_pool_bytes(policy,one_pool).status!=llama_kv_stream_policy_status::success ||
                one_pool>SIZE_MAX/sources.size()) return result;
        next.target.pool_minimum_bytes=one_pool*sources.size();
        size_t graph=0;
        for (const auto & measured : next.catalog) {
            if (measured.buffer_type!=next.target.buffer_type || measured.measurement.alignment!=next.target.measurement.alignment)
                return result;
            for (auto bytes : measured.measurement.graph_bytes) graph=std::max(graph,bytes);
            next.target.writer_bytes=std::max(next.target.writer_bytes,measured.writer_bytes);
            next.target.attention_bytes=std::max(next.target.attention_bytes,measured.attention_bytes);
            next.target.encoded_stage_bytes=std::max(next.target.encoded_stage_bytes,measured.encoded_stage_bytes);
            next.target.publication_bytes=std::max(next.target.publication_bytes,measured.publication_bytes);
            next.target.host_compute_bytes=std::max(next.target.host_compute_bytes,measured.host_compute_bytes);
        }
        std::fill(next.target.measurement.graph_bytes.begin(),next.target.measurement.graph_bytes.end(),graph);
        next.target.measurement.live_device_bytes=0;next.target.alignment_slack_bytes=0;
        const auto alignment=next.target.measurement.alignment;
        const auto reserve=[&](size_t bytes) {
            if (bytes>SIZE_MAX-(alignment-1)) return false;
            const auto padded=(bytes+alignment-1)&~(alignment-1);
            if (padded>SIZE_MAX-next.target.measurement.live_device_bytes) return false;
            next.target.measurement.live_device_bytes+=padded;next.target.alignment_slack_bytes+=padded-bytes;return true;
        };
        for (auto bytes : {next.target.writer_bytes,next.target.attention_bytes,next.target.encoded_stage_bytes,next.target.publication_bytes})
            if (!reserve(bytes)) return result;
        for (size_t i=0;i<sources.size();++i) if (!reserve(one_pool)) return result;
        std::vector<llama_memory_group_workload> workloads;
        std::vector<llama_memory_group_workload_measurement> measurements;
        if (next.catalog.empty() || next.catalog.size()-1>UINT64_MAX-id) return result;
        for (size_t i=0;i<next.catalog.size();++i) {
            auto & measured=next.catalog[i].measurement;measured.workload.id=id+i;
            measured.live_device_bytes=next.target.measurement.live_device_bytes;
            workloads.push_back(measured.workload);measurements.push_back(measured);
        }
        const auto budget=llama_memory_group_workload_budget_make(next.caches,next.cache_allocations,workloads,measurements,
            next.current_state_bytes,capacity,next.budget);
        if (budget.status==llama_memory_workload_budget_status::insufficient_budget) {
            result.status=llama_context_memory_create_status::insufficient_budget;
            result.required_bytes=capacity+budget.additional_bytes;result.additional_bytes=budget.additional_bytes;result.stage=id;
            return result;
        }
        if (budget.status!=llama_memory_workload_budget_status::success) return result;
        result.status=llama_context_memory_create_status::success;result.required_bytes=next.budget.parent_minimum_bytes;
        output=std::move(next);return result;
    } catch (const std::bad_alloc &) {result.status=llama_context_memory_create_status::allocation_failed;return result;}
}

bool llama_context::measure_dflash_workload(const llama_batch & input,
        const std::vector<llama_memory_phase_cache_identity> & sources,
        llama_memory_phase_cache_identity destination,llama_memory_work_phase phase,
        uint64_t id,llama_context_group_workload_quote & output) {
    const bool predict=phase==llama_memory_work_phase::draft_block_prediction;
    if (!group_initializing || model.arch!=LLM_ARCH_DFLASH || !group_target || !kv_resources.bounded || !id ||
            !destination.id || !destination.generation || sources.size()!=group_selected.size() || input.n_tokens<=0 ||
            uint32_t(input.n_tokens)>cparams.n_batch || (!predict && phase!=llama_memory_work_phase::draft_feature_injection)) return false;
    try {
        llama_context_group_workload_quote next;
        next.buffer_type=llama_kv_stream_device_buffer_type(model.devices.front().dev);
        size_t domain=backend_ptrs.size();
        for (size_t i=0;i<backend_ptrs.size();++i)
            if (ggml_backend_get_device(backend_ptrs[i])==model.devices.front().dev) domain=i;
        if (!next.buffer_type || domain==backend_ptrs.size()) return false;
        llama_batch_compat compat(this,input);llama_batch_allocr allocator(model.hparams.n_pos_per_embd());
        if (!allocator.init(*compat.batch_ext,model.vocab,false)) return false;
        const auto & batch=allocator.get_batch();
        std::array<uint32_t,LLAMA_MAX_SEQ> rows{},member_index{},consumed{};member_index.fill(UINT32_MAX);
        for (int32_t row=0;row<batch.n_tokens;++row) {
            if (batch.n_seq_id[row]!=1 || !batch.seq_id[row] || batch.logits[row]) return false;
            const auto found=std::find(group_selected.begin(),group_selected.end(),batch.seq_id[row][0]);
            if (found==group_selected.end()) return false;
            ++rows[uint32_t(*found)];
        }
        auto & measured=next.measurement;auto & group=measured.workload;
        group.id=id;group.selected_count=uint32_t(sources.size());group.microbatch_rows=cparams.n_ubatch;
        uint32_t block_rows=0;
        for (size_t i=0;i<group_selected.size();++i) {
            const auto slot=uint32_t(group_selected[i]);
            if (!sources[i].id || !sources[i].generation || sources[i].id==destination.id) return false;
            for (size_t j=0;j<i;++j) if (sources[i].id==sources[j].id) return false;
            if (!rows[slot]) continue;
            if (predict && (rows[slot]<2 || rows[slot]>cparams.serial_block_rows ||
                    (block_rows && block_rows!=rows[slot]))) return false;
            block_rows=rows[slot];
            llama_memory_workload_requirement member;
            member.phase=phase;member.cache=sources[i];member.destination_cache=destination;
            member.input_rows=rows[slot];member.query_rows=member.hidden_output_rows=predict ? rows[slot] : 0;
            member.microbatch_rows=cparams.n_ubatch;member.writes_cache=true;member.causal=cparams.causal_attn;
            member_index[slot]=uint32_t(group.members.size());group.members.push_back(member);
        }
        if (group.members.empty() || (predict && uint32_t(input.n_tokens)>cparams.n_ubatch)) return false;
        std::vector<size_t> host_peaks(backend_ptrs.size());
        while (true) {
            auto native=allocator.split_simple(cparams.n_ubatch);if (!native.n_tokens) break;
            std::vector<llama_memory_group_member_range> ranges;
            for (uint32_t row=0;row<native.n_tokens;++row) {
                const auto slot=uint32_t(native.seq_id[row][0]),member=member_index[slot];
                if (member==UINT32_MAX) return false;
                if (ranges.empty() || ranges.back().member!=member) ranges.push_back({member,consumed[slot],0,0,0});
                ++ranges.back().rows;ranges.back().hidden_output_rows+=uint32_t(predict);++consumed[slot];
            }
            float dummy=0;
            if (!predict) {native.token=nullptr;native.embd=&dummy;}
            std::vector<size_t> graph;
            if (!measure_workspace(native,false,graph) || graph.size()!=backend_ptrs.size() || !graph[domain]) return false;
            group.native_batches.push_back(std::move(ranges));measured.graph_bytes.push_back(graph[domain]);
            measured.execution_available.push_back(true);
            for (size_t i=0;i<graph.size();++i) if (i!=domain) host_peaks[i]=std::max(host_peaks[i],graph[i]);
        }
        for (uint32_t slot=0;slot<cparams.n_seq_max;++slot) if (consumed[slot]!=rows[slot]) return false;
        for (auto bytes : host_peaks) {
            if (bytes>SIZE_MAX-next.host_compute_bytes) return false;
            next.host_compute_bytes+=bytes;
        }
        const auto width=predict ? model.hparams.n_embd_out() : model.hparams.n_embd_inp_enc();
        if (uint64_t(width)*sizeof(float)>SIZE_MAX/uint32_t(input.n_tokens)) return false;
        measured.host_feature_bytes=size_t(width)*sizeof(float)*uint32_t(input.n_tokens);
        measured.alignment=ggml_backend_buft_get_alignment(next.buffer_type);
        output=std::move(next);return true;
    } catch (const std::bad_alloc &) {return false;}
}

llama_context_memory_create_result llama_context::quote_dflash_group(const llama_batch & batch,
        const std::vector<llama_memory_phase_cache_identity> & sources,
        llama_memory_phase_cache_identity bounded_identity,llama_memory_work_phase phase,
        uint64_t id,size_t capacity,llama_context_group_plan & output,llama_context & draft) {
    llama_context_memory_create_result result;result.available_bytes=capacity;
    if (!group_initializing || !draft.group_initializing || model.arch!=LLM_ARCH_QWEN35 || !cparams.n_rs_seq ||
            cparams.kv_stream_auxiliary_layers || kv_resources.bounded || draft.group_target!=this ||
            draft.group_selected!=group_selected || !draft.kv_resources.bounded || !draft.cparams.serial_block_rows ||
            sources.size()!=group_selected.size() || sources.empty() || !id) return result;
    try {
        llama_context_group_plan next;
        if (quote_target_group(batch,sources,bounded_identity,phase,id,SIZE_MAX,next).status!=
                llama_context_memory_create_status::success) return result;
        const auto maximum=std::max(cparams.n_batch,draft.cparams.n_batch);
        if (!maximum || maximum>INT32_MAX) return result;
        auto input=llama_batch_init(int32_t(maximum),0,1);
        struct release_batch {llama_batch & batch;~release_batch() {llama_batch_free(batch);}} release{input};
        const auto fill=[&](const std::vector<uint32_t> & widths,bool outputs) {
            input.n_tokens=0;
            for (size_t i=0;i<widths.size();++i) for (uint32_t row=0;row<widths[i];++row) {
                if (uint32_t(input.n_tokens)>=maximum) return false;
                const auto index=input.n_tokens++;
                input.token[index]=0;input.pos[index]=llama_pos(row);input.n_seq_id[index]=1;
                input.seq_id[index][0]=group_selected[i];input.logits[index]=outputs;
            }
            return input.n_tokens>0;
        };
        if (phase!=llama_memory_work_phase::target_verify) {
            if (!fill(std::vector<uint32_t>(sources.size(),group_verify_rows(cparams)),true)) return result;
            llama_context_group_plan verify;
            if (quote_target_group(input,sources,bounded_identity,llama_memory_work_phase::target_verify,id,SIZE_MAX,verify).status!=
                    llama_context_memory_create_status::success) return result;
            next.catalog.insert(next.catalog.end(),verify.catalog.begin(),verify.catalog.end());
        }
        llama_memory_params memory_params{};
        memory_params.type_k=draft.kv_resources.bounded->type_k();memory_params.type_v=draft.kv_resources.bounded->type_v();
        memory_params.kv_stream_selected_sequences=group_selected;
        if (!llama_kv_cache_iswa::bounded_storage_requirements(draft.model.hparams,memory_params,draft.cparams,
                next.target.buffer_type,bounded_identity,next.bounded)) return result;
        next.caches.push_back(next.bounded.cache);next.cache_allocations.push_back(next.bounded.allocation);
        const auto measure=[&](const std::vector<uint32_t> & widths,bool predict) {
            if (!fill(widths,false)) return false;
            llama_context_group_workload_quote item;
            if (!draft.measure_dflash_workload(input,sources,bounded_identity,predict ? llama_memory_work_phase::draft_block_prediction :
                    llama_memory_work_phase::draft_feature_injection,1,item)) return false;
            next.catalog.push_back(std::move(item));return true;
        };
        std::vector<uint32_t> actual(sources.size());
        for (int32_t row=0;row<batch.n_tokens;++row) {
            if (batch.n_seq_id[row]!=1 || !batch.seq_id[row]) return result;
            const auto found=std::find(group_selected.begin(),group_selected.end(),batch.seq_id[row][0]);
            if (found==group_selected.end()) return result;
            ++actual[size_t(found-group_selected.begin())];
        }
        for (size_t slot=0;slot<sources.size();++slot) {
            std::vector<uint32_t> widths;
            const auto add=[&](uint32_t width) {if (width && std::find(widths.begin(),widths.end(),width)==widths.end()) widths.push_back(width);};
            for (uint32_t width=1;width<=group_verify_rows(cparams);++width) add(width);
            add(draft.cparams.n_ubatch);add(std::min(actual[slot],draft.cparams.n_ubatch));
            add(actual[slot]%draft.cparams.n_ubatch);
            for (auto width : widths) {
                std::vector<uint32_t> selected(sources.size());selected[slot]=width;
                if (!measure(selected,false)) return result;
            }
        }
        if (sources.size()>2) {
            for (uint32_t width=2;width<=draft.cparams.serial_block_rows;++width)
                for (uint32_t active=1;active<=std::min(uint32_t(sources.size()),draft.cparams.n_ubatch/width);++active) {
                    std::vector<uint32_t> widths(sources.size());std::fill_n(widths.begin(),active,width);
                    if (!measure(widths,true)) return result;
                }
        } else {
        for (uint32_t width=2;width<=draft.cparams.serial_block_rows;++width)
            for (size_t mask=1;mask<(size_t(1)<<sources.size());++mask) {
                std::vector<uint32_t> widths(sources.size());uint64_t total=0;
                for (size_t slot=0;slot<sources.size();++slot) if ((mask>>slot)&1) {widths[slot]=width;total+=width;}
                if (total<=draft.cparams.n_ubatch && !measure(widths,true)) return result;
            }
        }
        size_t graph=0;
        for (const auto & item : next.catalog) {
            if (item.buffer_type!=next.target.buffer_type || item.measurement.alignment!=next.target.measurement.alignment) return result;
            for (auto bytes : item.measurement.graph_bytes) graph=std::max(graph,bytes);
            next.target.attention_bytes=std::max(next.target.attention_bytes,item.attention_bytes);
            next.target.publication_bytes=std::max(next.target.publication_bytes,item.publication_bytes);
            next.target.host_compute_bytes=std::max(next.target.host_compute_bytes,item.host_compute_bytes);
        }
        std::fill(next.target.measurement.graph_bytes.begin(),next.target.measurement.graph_bytes.end(),graph);
        next.target.measurement.live_device_bytes=0;next.target.alignment_slack_bytes=0;
        const auto alignment=next.target.measurement.alignment;
        const auto reserve=[&](size_t bytes) {
            if (!alignment || bytes>SIZE_MAX-(alignment-1)) return false;
            const auto padded=(bytes+alignment-1)&~(alignment-1);
            if (padded>SIZE_MAX-next.target.measurement.live_device_bytes) return false;
            next.target.measurement.live_device_bytes+=padded;next.target.alignment_slack_bytes+=padded-bytes;return true;
        };
        for (auto bytes : {next.target.writer_bytes,next.target.attention_bytes,next.target.encoded_stage_bytes,next.target.publication_bytes})
            if (!reserve(bytes)) return result;
        for (size_t i=0;i<sources.size();++i) if (!reserve(next.target.pool_minimum_bytes/sources.size())) return result;
        std::vector<llama_memory_group_workload> workloads;
        std::vector<llama_memory_group_workload_measurement> measurements;
        if (next.catalog.empty() || next.catalog.size()-1>UINT64_MAX-id) return result;
        for (size_t i=0;i<next.catalog.size();++i) {
            auto & measured=next.catalog[i].measurement;measured.workload.id=id+i;
            measured.live_device_bytes=next.target.measurement.live_device_bytes;
            workloads.push_back(measured.workload);measurements.push_back(measured);
        }
        const auto budget=llama_memory_group_workload_budget_make(next.caches,next.cache_allocations,workloads,measurements,
            next.current_state_bytes,capacity,next.budget);
        if (budget.status==llama_memory_workload_budget_status::insufficient_budget) {
            result.status=llama_context_memory_create_status::insufficient_budget;result.additional_bytes=budget.additional_bytes;
            result.required_bytes=capacity+budget.additional_bytes;return result;
        }
        if (budget.status!=llama_memory_workload_budget_status::success) return result;
        result.status=llama_context_memory_create_status::success;result.required_bytes=next.budget.parent_minimum_bytes;
        output=std::move(next);return result;
    } catch (const std::bad_alloc &) {result.status=llama_context_memory_create_status::allocation_failed;return result;}
}

bool llama_context::measure_target_workload(const llama_batch & input,
        const std::vector<llama_memory_phase_cache_identity> & sources,
        llama_memory_work_phase phase,uint64_t id,llama_context_group_workload_quote & output) {
    const uint32_t selected_count=group_selected.empty() ? cparams.n_seq_max : uint32_t(group_selected.size());
    const bool auxiliary=group_initializing && cparams.ctx_type==LLAMA_CONTEXT_TYPE_MTP;
    const bool readonly=auxiliary && model.arch==LLM_ARCH_GEMMA4_ASSISTANT && group_target && cparams.ctx_other==group_target;
    const bool supported_phase=readonly ? phase==llama_memory_work_phase::mtp_draft : auxiliary ?
        phase==llama_memory_work_phase::mtp_catchup || phase==llama_memory_work_phase::mtp_draft :
        phase==llama_memory_work_phase::target_prefill || phase==llama_memory_work_phase::target_verify;
    if ((!measurement_only && !group_initializing) || !id || !cparams.kv_unified || !cparams.n_seq_max || cparams.n_seq_max>LLAMA_MAX_SEQ ||
            sources.size()!=selected_count || input.n_tokens<=0 || uint32_t(input.n_tokens)>cparams.n_batch ||
            model.devices.size()!=1 || !kv_resources.full || !supported_phase) return false;
    std::array<uint32_t,LLAMA_MAX_SEQ> source_index;
    source_index.fill(UINT32_MAX);
    for (uint32_t i=0;i<selected_count;++i) {
        const auto slot=group_selected.empty() ? i : uint32_t(group_selected[i]);
        if (slot>=cparams.n_seq_max || source_index[slot]!=UINT32_MAX) return false;
        source_index[slot]=i;
    }
    for (size_t i=0;i<sources.size();++i) {
        if (!sources[i].id || !sources[i].generation) return false;
        for (size_t j=0;j<i;++j) if (sources[i].id==sources[j].id) return false;
    }
    try {
        llama_context_group_workload_quote next;
        auto * device=model.devices.front().dev;
        size_t domain=backend_ptrs.size();
        for (size_t i=0;i<backend_ptrs.size();++i)
            if (ggml_backend_get_device(backend_ptrs[i])==device) domain=i;
        if (domain==backend_ptrs.size()) return false;
        auto * backend=backend_ptrs[domain];
        next.buffer_type=llama_kv_stream_device_buffer_type(device);
        if (!next.buffer_type) return false;
        const auto get=reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(device),"ggml_backend_kv_stream_partial_ops"));
        const auto * ops=get ? get() : nullptr;
        if (!ops || !ops->capabilities) return false;
        const auto layers=kv_resources.full->get_layer_ids();
        if (layers.empty()) return false;
        const auto first=layers.front();
        const auto & hp=model.hparams;
        const ggml_kv_stream_shape shape{kv_resources.full->get_k_storage(first)->type,
            kv_resources.full->get_v_storage(first)->type,hp.n_embd_head_k(first),hp.n_embd_head_v(first),
            hp.n_head_kv(first),256,128};
        const auto heads=hp.n_head(first);
        for (auto layer : layers)
            if (hp.n_embd_head_k(layer)!=shape.head_dim_k || hp.n_embd_head_v(layer)!=shape.head_dim_v ||
                    hp.n_head_kv(layer)!=shape.heads || hp.n_head(layer)!=heads ||
                    kv_resources.full->get_k_storage(layer)->type!=shape.type_k ||
                    kv_resources.full->get_v_storage(layer)->type!=shape.type_v) return false;
        if (!llama_kv_stream_writer::requirements(next.buffer_type,shape,cparams.n_ubatch,next.writer_bytes)) return false;
        ggml_kv_stream_layout encoded;
        if (ggml_kv_stream_layout_make(shape,256,encoded).status!=ggml_kv_stream_status::success) return false;
        next.encoded_stage_bytes=encoded.bytes;
        llama_kv_stream_policy_config policy;
        policy.shape=shape; policy.capabilities=ops->capabilities(backend,shape.type_k,shape.type_v);
        policy.layers=uint32_t(layers.size());
        size_t one_pool=0;
        if (llama_kv_stream_policy_minimum_pool_bytes(policy,one_pool).status!=llama_kv_stream_policy_status::success ||
                one_pool>SIZE_MAX/selected_count) return false;
        next.pool_minimum_bytes=one_pool*selected_count;
        auto & measurement=next.measurement;
        auto & group=measurement.workload;
        group.id=id; group.selected_count=selected_count; group.microbatch_rows=cparams.n_ubatch;
        llama_batch_compat compat(this,input);
        llama_batch_allocr allocator(hp.n_pos_per_embd());
        if (!allocator.init(*compat.batch_ext,model.vocab,false)) return false;
        const auto & batch=allocator.get_batch();
        std::array<uint32_t,LLAMA_MAX_SEQ> rows{},outputs{},member_index{},consumed{};
        member_index.fill(UINT32_MAX);
        for (int32_t row=0;row<batch.n_tokens;++row) {
            if (batch.n_seq_id[row]!=1 || !batch.seq_id[row] || batch.seq_id[row][0]<0 ||
                    uint32_t(batch.seq_id[row][0])>=cparams.n_seq_max) return false;
            const auto slot=uint32_t(batch.seq_id[row][0]);
            if (source_index[slot]==UINT32_MAX) return false;
            ++rows[slot]; outputs[slot]+=uint32_t(batch.logits[row]!=0);
        }
        for (uint32_t slot=0;slot<cparams.n_seq_max;++slot) if (rows[slot]) {
            if (group_initializing && phase==llama_memory_work_phase::target_verify && rows[slot]>group_verify_rows(cparams)) return false;
            if (auxiliary && ((phase==llama_memory_work_phase::mtp_draft && (rows[slot]!=1 || outputs[slot]!=1)) ||
                    (phase==llama_memory_work_phase::mtp_catchup && outputs[slot]))) return false;
            llama_memory_workload_requirement member;
            member.phase=phase; member.cache=sources[source_index[slot]]; member.input_rows=member.query_rows=rows[slot];
            member.output_rows=outputs[slot]; member.microbatch_rows=cparams.n_ubatch;
            member.rollback_depth=phase==llama_memory_work_phase::target_verify ? cparams.n_rs_seq : 0;
            member.writes_cache=!readonly; member.causal=cparams.causal_attn;
            member_index[slot]=uint32_t(group.members.size()); group.members.push_back(member);
        }
        if (group.members.empty()) return false;
        std::vector<size_t> host_peaks(backend_ptrs.size());
        while (true) {
            auto native=kv_resources.recurrent ? allocator.split_equal(cparams.n_ubatch,!cparams.kv_unified,
                cparams.n_rs_seq ? cparams.n_rs_seq+1 : 0) : allocator.split_simple(cparams.n_ubatch);
            if (!native.n_tokens) break;
            std::vector<llama_memory_group_member_range> ranges;
            for (uint32_t row=0;row<native.n_tokens;++row) {
                if (native.n_seq_id[row]!=1 || !native.seq_id[row]) return false;
                const auto slot=uint32_t(native.seq_id[row][0]);
                if (slot>=cparams.n_seq_max || member_index[slot]==UINT32_MAX) return false;
                const auto member=member_index[slot];
                if (ranges.empty() || ranges.back().member!=member)
                    ranges.push_back({member,consumed[slot],0,0});
                ++ranges.back().rows; ranges.back().output_rows+=uint32_t(native.output[row]!=0); ++consumed[slot];
            }
            std::vector<size_t> graph;
            if (!measure_workspace(native,true,graph) || graph.size()!=backend_ptrs.size() || !graph[domain]) {
                LLAMA_LOG_ERROR("%s: native group graph is unavailable (%u rows, %u selected)\n",__func__,native.n_tokens,selected_count);return false;
            }
            measurement.graph_bytes.push_back(graph[domain]); measurement.execution_available.push_back(true);
            group.native_batches.push_back(std::move(ranges));
            for (size_t i=0;i<graph.size();++i) if (i!=domain) host_peaks[i]=std::max(host_peaks[i],graph[i]);
            size_t attention=0;
            const auto kind=phase==llama_memory_work_phase::target_verify || phase==llama_memory_work_phase::mtp_draft ?
                ggml_kv_stream_workspace_kind::decode : ggml_kv_stream_workspace_kind::prefill;
            const ggml_kv_stream_attention_request request{shape,heads,native.n_tokens,cparams.n_ctx};
            if (!ggml_kv_stream_workspace_query(ops,backend,request,kind,attention) &&
                    (kind!=ggml_kv_stream_workspace_kind::decode || !ggml_kv_stream_workspace_query(
                        ops,backend,request,ggml_kv_stream_workspace_kind::prefill,attention))) {
                LLAMA_LOG_ERROR("%s: native group attention workspace is unavailable (%u rows)\n",__func__,native.n_tokens);return false;
            }
            next.attention_bytes=std::max(next.attention_bytes,attention);
        }
        for (uint32_t slot=0;slot<cparams.n_seq_max;++slot) if (consumed[slot]!=rows[slot]) return false;
        if (group.native_batches.empty()) return false;
        if (cparams.n_rs_seq && cparams.recurrent_rollback_ram && kv_resources.recurrent) {
            ggml_backend_buffer_type_t type=nullptr;
            if (!kv_resources.recurrent->prepare_measurement_publication(selected_count) ||
                    !kv_resources.recurrent->spill_stage_requirements(type,next.publication_bytes) || type!=next.buffer_type) return false;
        }
        measurement.alignment=ggml_backend_buft_get_alignment(next.buffer_type);
        const auto alignment=measurement.alignment;
        if (!alignment || (alignment&(alignment-1))) return false;
        const auto add_region=[&](size_t bytes) {
            if (bytes>SIZE_MAX-(alignment-1)) return false;
            const size_t padded=(bytes+alignment-1)&~(alignment-1);
            if (padded>SIZE_MAX-measurement.live_device_bytes) return false;
            next.alignment_slack_bytes+=padded-bytes;
            measurement.live_device_bytes+=padded;
            return true;
        };
        for (size_t bytes : {next.writer_bytes,next.attention_bytes,next.encoded_stage_bytes,next.publication_bytes})
            if (!add_region(bytes)) return false;
        for (uint32_t source=0;source<selected_count;++source) if (!add_region(one_pool)) return false;
        for (auto bytes : host_peaks) {
            if (bytes>SIZE_MAX-next.host_compute_bytes) return false;
            next.host_compute_bytes+=bytes;
        }
        const size_t packets=256*sizeof(ggml_kv_stream_gather_row)+512*sizeof(ggml_kv_stream_copy_rectangle);
        if (cparams.n_ctx>(SIZE_MAX-packets)/sizeof(ggml_kv_stream_gather_row)) return false;
        next.host_descriptor_bytes=size_t(cparams.n_ctx)*sizeof(ggml_kv_stream_gather_row)+packets;
        measurement.alignment=ggml_backend_buft_get_alignment(next.buffer_type);
        output=std::move(next);
        return true;
    } catch (const std::bad_alloc &) {return false;}
}

bool llama_context::measure_workspace(uint32_t tokens, uint32_t sequences, uint32_t outputs,
        bool managed_full_attention, std::vector<size_t> & sizes) {
    const auto maximum=group_selected.empty() ? cparams.n_seq_max : uint32_t(group_selected.size());
    if ((!measurement_only && !group_initializing) || !memory || !sequences || sequences>maximum || !tokens ||
            tokens>cparams.n_ubatch || tokens%sequences || !outputs || outputs>tokens) return false;
    return measure_workspace_impl(nullptr,tokens,sequences,outputs,managed_full_attention,sizes);
}

bool llama_context::measure_workspace(const llama_ubatch & batch, bool managed_full_attention,
        std::vector<size_t> & sizes) {
    const auto maximum=group_selected.empty() ? cparams.n_seq_max : uint32_t(group_selected.size());
    if ((!measurement_only && !group_initializing) || !memory || !batch.n_tokens || batch.n_tokens>cparams.n_ubatch ||
            !batch.n_seqs || batch.n_seqs>batch.n_tokens || !batch.n_seqs_unq || batch.n_seqs_unq>maximum ||
            batch.n_pos!=model.hparams.n_pos_per_embd() || !batch.n_seq_id || !batch.seq_id || !batch.output)
        return false;
    std::array<bool,LLAMA_MAX_SEQ> seen{};
    uint32_t outputs=0,unique=0;
    for (uint32_t row=0;row<batch.n_tokens;++row) {
        if (batch.n_seq_id[row]!=1 || !batch.seq_id[row] || batch.seq_id[row][0]<0 ||
                uint32_t(batch.seq_id[row][0])>=cparams.n_seq_max || batch.output[row]<0 || batch.output[row]>1) return false;
        const auto id=uint32_t(batch.seq_id[row][0]);
        if (group_initializing && std::find(group_selected.begin(),group_selected.end(),llama_seq_id(id))==group_selected.end()) return false;
        if (!seen[id]) {seen[id]=true; ++unique;}
        outputs+=uint32_t(batch.output[row]);
    }
    if (unique!=batch.n_seqs_unq) return false;
    return measure_workspace_impl(&batch,batch.n_tokens,batch.n_seqs,outputs,managed_full_attention,sizes);
}

bool llama_context::measure_workspace_impl(const llama_ubatch * native_batch, uint32_t tokens, uint32_t sequences,
        uint32_t outputs, bool managed_full_attention, std::vector<size_t> & sizes) {
    try {
        std::vector<size_t> measured(backend_ptrs.size());
        if (cparams.n_rs_seq && cparams.recurrent_rollback_ram && kv_resources.recurrent) {
            ggml_backend_sched_reset(sched.get());
            for (auto & previous : gf_res_prev) if (previous) previous->reset();
            gf_res_reserve->reset();
            const auto active=native_batch ? native_batch->n_seqs : sequences;
            if (!kv_resources.recurrent->prepare_measurement_publication(active)) return false;
        }
        auto full=memory->init_full();
        if (!full) return false;
        std::vector<size_t> sampler_bytes(backend_ptrs.size());
        const auto build=[&]() -> ggml_cgraph * {
            auto * graph=graph_build_reserve(tokens,sequences,outputs,full.get(),false,native_batch);
            if (!graph || !group_initializing || group_selected.size()<=2 || sampling.samplers.empty()) return graph;
            auto * result=get_gf_res_reserve();
            for (const auto & input : result->inputs) {
                const auto * sampler=dynamic_cast<const llm_graph_input_sampling *>(input.get());
                if (!sampler || sampler->samplers.empty()) continue;
                const auto name="logits_seq_"+std::to_string(sampler->samplers.begin()->first)+"_0";
                auto * first=ggml_get_tensor(result->get_ctx(),name.c_str());
                if (!first || !first->view_src) return nullptr;
                auto * padded=first->view_src;
                // Sampling appends every chain, including inactive ones. Count all owning tensors after its padded logits.
                for (auto * tensor=padded;tensor;tensor=ggml_get_next_tensor(result->get_ctx(),tensor)) {
                    if (!tensor->view_src && !tensor->buffer) {
                        auto allocation=*tensor;
                        if (tensor==padded) {
                            allocation.ne[1]=int64_t(std::min(cparams.n_outputs_max,cparams.n_ubatch))+1;
                            for (int dim=2;dim<GGML_MAX_DIMS;++dim) {
                                if (allocation.ne[dim-1]<0 || (allocation.nb[dim-1] &&
                                        uint64_t(allocation.ne[dim-1])>SIZE_MAX/allocation.nb[dim-1])) return nullptr;
                                allocation.nb[dim]=allocation.nb[dim-1]*size_t(allocation.ne[dim-1]);
                            }
                        }
                        for (size_t i=0;i<backend_ptrs.size();++i) {
                            auto * type=backend_buft[i];
                            const auto alignment=ggml_backend_buft_get_alignment(type);
                            const auto bytes=ggml_backend_buft_get_alloc_size(type,&allocation);
                            if (!alignment || bytes>SIZE_MAX-(alignment-1)) return nullptr;
                            const auto padded=(bytes+alignment-1)/alignment*alignment;
                            if (padded>SIZE_MAX-sampler_bytes[i]) return nullptr;
                            sampler_bytes[i]+=padded;
                        }
                    }
                }
            }
            struct restore_samplers {
                decltype(sampling.samplers) & current;
                decltype(sampling.samplers) saved;
                ~restore_samplers() {current=std::move(saved);}
            } restore{sampling.samplers,std::move(sampling.samplers)};
            sampling.samplers.clear();
            ggml_backend_sched_reset(sched.get());gf_res_reserve->reset();
            return graph_build_reserve(tokens,sequences,outputs,full.get(),false,native_batch);
        };
        const auto reserve=[&](ggml_cgraph * graph) {
            ggml_backend_sched_reserve_size(sched.get(),graph,measured.data());
            for (size_t i=0;i<measured.size();++i) {
                if (sampler_bytes[i]>SIZE_MAX-measured[i]) return false;
                measured[i]+=sampler_bytes[i];
            }
            return true;
        };
        if (!managed_full_attention) {
            auto * graph=build();
            if (!graph) return false;
            if (!reserve(graph)) return false;
            ggml_backend_sched_reset(sched.get());
            gf_res_reserve->reset();
            sizes=std::move(measured);
            return true;
        }
        if (!cparams.kv_unified || model.devices.size()!=1 || !kv_resources.full) return false;
        ggml_backend_t backend=nullptr;
        for (auto * candidate : backend_ptrs)
            if (ggml_backend_get_device(candidate) == model.devices.front().dev) backend=candidate;
        if (!backend) return false;
        auto * device=ggml_backend_get_device(backend);
        auto * registry=ggml_backend_dev_backend_reg(device);
        auto get=reinterpret_cast<ggml_kv_stream_partial_ops_get>(
            ggml_backend_reg_get_proc_address(registry,"ggml_backend_kv_stream_partial_ops"));
        const auto * partial=get ? get() : nullptr;
        if (!partial || !partial->capabilities) return false;

        std::vector<std::pair<ggml_tensor *,ggml_backend_buffer_t>> roots;
        size_t bytes=0;
        for (auto layer : kv_resources.full->get_layer_ids()) {
            for (auto * tensor : {kv_resources.full->get_k_storage(layer),kv_resources.full->get_v_storage(layer)}) {
                if (!tensor || tensor->view_src) return false;
                if (std::any_of(roots.begin(),roots.end(),[&](const auto & item) {return item.first == tensor;})) continue;
                const size_t payload=ggml_nbytes(tensor);
                if (!payload || payload>SIZE_MAX-bytes) return false;
                bytes+=payload;
                roots.emplace_back(tensor,tensor->buffer);
            }
        }
        if (roots.empty()) return false;
        for (auto * cache : {kv_resources.full,kv_resources.bounded}) if (cache) {
            for (auto layer : cache->get_layer_ids()) {
                auto * tensor=cache->get_k_storage(layer);
                if (!tensor || !tensor->buffer ||
                        ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer))!=device) return false;
            }
        }
        std::unique_ptr<ggml_backend_host_reservation,decltype(&ggml_backend_host_reservation_free)> address(
            ggml_backend_host_reservation_new(bytes),ggml_backend_host_reservation_free);
        if (!address) return false;
        struct measurement_owner {
            ggml_backend_t backend;
            const ggml_kv_stream_partial_ops * partial;
        } owner{backend,partial};
        const ggml_backend_execution_ops ops{
            [](void * data,const ggml_tensor * op) {
                const auto & state=*static_cast<measurement_owner *>(data);
                if (op->op == GGML_OP_SET_ROWS)
                    return op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] && op->src[1]->type == GGML_TYPE_I64;
                if (op->op != GGML_OP_FLASH_ATTN_EXT || !op->src[0] || !op->src[1] || !op->src[2]) return false;
                const auto * k=op->src[1],*v=op->src[2];
                ggml_kv_stream_execution execution;
                const auto caps=state.partial->capabilities(state.backend,k->type,v->type);
                return ggml_kv_stream_attention_validate(op,{k->ne[0],v->ne[0],256,256,128},caps,256,execution).status ==
                    ggml_kv_stream_status::success;
            },
            [](void *,ggml_backend_t,ggml_tensor *) {return GGML_STATUS_FAILED;},
            [](void *) {return true;},[](void *) {},[](void *) {}};
        const ggml_backend_execution_io io{&owner,
            [](void *,const ggml_tensor *,void *,size_t,size_t) {return false;},
            [](void *,const ggml_tensor *,const void *,size_t,size_t) {return false;},
            [](void *,const ggml_tensor *,uint8_t,size_t,size_t) {return false;},
            [](void *,uint8_t) {return false;}};
        ggml_backend_buffer_ptr buffer(ggml_backend_execution_buffer_new_address_space(device,
            ggml_backend_host_reservation_base(address.get()),bytes,128,ops,io,&owner));
        if (!buffer) return false;
        struct restore_roots {
            llama_context & context;
            std::vector<std::pair<ggml_tensor *,ggml_backend_buffer_t>> & roots;
            ~restore_roots() {
                ggml_backend_sched_reset(context.sched.get());
                context.gf_res_reserve->reset();
                for (auto & item : roots) item.first->buffer=item.second;
            }
        } restore{*this,roots};
        for (auto & item : roots) item.first->buffer=buffer.get();
        auto * graph=build();
        if (!graph) return false;
        size_t managed=0;
        for (int i=0;i<ggml_graph_n_nodes(graph);++i) {
            auto * op=ggml_graph_node(graph,i);
            if (op->op != GGML_OP_FLASH_ATTN_EXT) continue;
            auto * root=op->src[1];
            while (root->view_src) root=root->view_src;
            if (root->buffer != buffer.get()) continue;
            const auto * q=op->src[0],*k=op->src[1],*v=op->src[2];
            if (q->ne[3]!=1 || k->ne[2]>INT32_MAX || q->ne[2]>UINT32_MAX || q->ne[1]>UINT32_MAX) return false;
            ggml_kv_stream_resume_plan plan;
            const ggml_kv_stream_attention_request request{
                {k->type,v->type,int32_t(k->ne[0]),int32_t(v->ne[0]),int32_t(k->ne[2]),256,128},
                uint32_t(q->ne[2]),uint32_t(q->ne[1]),size_t(k->ne[1]),3};
            if (!ggml_kv_stream_resume_plan_query(partial,backend,request,plan)) {
                LLAMA_LOG_ERROR("%s: no native resume plan for D%lld, %lld queries, %lld keys, %lld heads\n",__func__,
                    (long long)k->ne[0],(long long)q->ne[1],(long long)k->ne[1],(long long)q->ne[2]);return false;
            }
            ggml_backend_execution_set_external_workspace(op,true);
            ggml_backend_execution_set_verified_workspace(op);
            if (!ggml_backend_execution_supports(buffer.get(),device,op)) {
                LLAMA_LOG_ERROR("%s: unsupported native group attention (%lld queries, %lld keys)\n",__func__,
                    (long long)q->ne[1],(long long)k->ne[1]);return false;
            }
            ++managed;
        }
        if (!managed) return false;
        if (!reserve(graph)) return false;
        sizes=std::move(measured);
        return true;
    } catch (const std::bad_alloc &) {return false;}
}

ggml_cgraph * llama_context::graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only, size_t * sizes, bool feature_input) {
    if (measurement_only && !split_only) return nullptr;
    LLAMA_LOG_DEBUG("%s: reserving a graph for ubatch with n_tokens = %4u, n_seqs = %2u, n_outputs = %4u\n", __func__, n_tokens, n_seqs, n_outputs);
    GGML_ASSERT(n_outputs >= 1);

    // Remeasure topology without constraining it to an older borrowed grant.
    struct scheduler_restore {
        ggml_backend_sched_ptr & current;
        ggml_backend_sched_ptr saved;
        ~scheduler_restore() { if (saved) current=std::move(saved); }
    } measuring{sched,{}};
    if (split_only && sizes && compute_memory) {
        ggml_backend_sched_ptr temporary(ggml_backend_sched_new(backend_ptrs.data(),backend_buft.data(),
            backend_ptrs.size(),size_t(gf_res_reserve->get_max_nodes()),cparams.pipeline_parallel,cparams.op_offload));
        if (!temporary) return nullptr;
        ggml_backend_sched_set_copy_callback(temporary.get(),sched_copy_experts,this);
        measuring.saved=std::move(sched);
        sched=std::move(temporary);
    }

    auto * gf=graph_build_reserve(n_tokens,n_seqs,n_outputs,mctx,feature_input);
    if (!gf) return nullptr;

    // initialize scheduler with the specified graph
    if (split_only) {
        if (sizes) {
            ggml_backend_sched_reserve_size(sched.get(), gf, sizes);
        } else {
            ggml_backend_sched_split_graph(sched.get(), gf);
        }
    } else if (!ggml_backend_sched_reserve(sched.get(), gf)) {
        GGML_ASSERT(!sizes);
        LLAMA_LOG_ERROR("%s: failed to allocate compute buffers\n", __func__);
        return nullptr;
    }

    return gf;
}

ggml_cgraph * llama_context::graph_build_reserve(uint32_t n_tokens, uint32_t n_seqs,
        uint32_t n_outputs, const llama_memory_context_i * mctx, bool feature_input, const llama_ubatch * native_batch) {
    if (!native_batch && n_tokens % n_seqs != 0) {
        n_tokens = ((n_tokens + (n_seqs - 1)) / n_seqs) * n_seqs; // round to next multiple of n_seqs
        LLAMA_LOG_DEBUG("%s: making n_tokens a multiple of n_seqs - n_tokens = %u, n_seqs = %u, n_outputs = %u\n", __func__, n_tokens, n_seqs, n_outputs);
    }

    ggml_backend_sched_reset(sched.get());

    // when the scheduler is reset, we cannot reuse old graphs, so we reset the previous graph results
    for (auto & res : gf_res_prev) {
        if (res) {
            res->reset();
        }
    }
    gf_res_prev_active = nullptr;

    // store the n_outputs as it is, and restore it afterwards
    // TODO: not sure if needed, might simplify in the future by removing this
    const auto save_n_outputs = this->n_outputs;

    this->n_outputs = n_outputs;

    llama_batch_allocr balloc(model.hparams.n_pos_per_embd());
    llama_ubatch ubatch = native_batch ? *native_batch : balloc.ubatch_reserve(n_tokens/n_seqs, n_seqs);

    // Reservation builds topology only; it does not read the dummy embedding value.
    float dummy_feature = 0;
    if (feature_input) {
        ubatch.token = nullptr;
        ubatch.embd = &dummy_feature;
    }

    if (!native_batch) {
        if (!group_selected.empty()) {
            if (n_seqs>group_selected.size()) throw std::runtime_error("reservation names more sequences than the selected group");
            std::fill(ubatch.seq_idx,ubatch.seq_idx+LLAMA_MAX_SEQ,-1);
            for (uint32_t index=0;index<n_seqs;++index) {
                const auto id=group_selected[index];
                ubatch.seq_id_unq[index]=id;ubatch.seq_idx[id]=int32_t(index);
            }
        }
        ubatch_prepare_reserve(ubatch, n_outputs, sampling.samplers, cparams.n_outputs_max_per_seq);
    }

    auto * res = gf_res_reserve.get();

    const auto gparams = graph_params(res, ubatch, mctx, ctx_type_to_graph_type(cparams.ctx_type));

    res->reset();

    auto * gf = model.build_graph(gparams);

    this->n_input_tensors = llama_graph_n_input_tensors(gf);
    this->n_outputs = save_n_outputs;

    return gf;
}

llm_graph_params llama_context::graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const {
    return {
        /*.arch        =*/ model.arch,
        /*.hparams     =*/ model.hparams,
        /*.cparams     =*/ cparams,
        /*.ubatch      =*/ ubatch,
        /*.gtype       =*/ gtype,
        /*.sched       =*/ sched.get(),
        /*.backend_cpu =*/ backend_cpu,
        /*.cvec        =*/ cvec.get(),
        /*.loras       =*/ loras.get(),
        /*.mctx        =*/ mctx,
        /*.cross       =*/ &cross,
        /*.moe_cache   =*/ moe_cache.get(),
        /*.prec_policy =*/ &model.prec_policy,
        /*.samplers    =*/ sampling.samplers,
        /*.n_outputs   =*/ n_outputs,
        /*.cb          =*/ graph_get_cb(),
        /*.res         =*/ res,
    };
}

ggml_status llama_context::graph_compute(
            ggml_cgraph * gf,
                   bool   batched) {
    if (measurement_only || group_initializing) return GGML_STATUS_FAILED;
    int n_threads        = batched ? cparams.n_threads_batch : cparams.n_threads;
    ggml_threadpool_t tp = batched ? threadpool_batch        : threadpool;

    if (backend_cpu != nullptr) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
        auto * set_threadpool_fn = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
        if (set_threadpool_fn) {
            set_threadpool_fn(backend_cpu, tp);
        }
    }

    // set the number of threads for all the backends
    for (const auto & set_n_threads_fn : set_n_threads_fns) {
        set_n_threads_fn.second(set_n_threads_fn.first, n_threads);
    }

    copy_experts.reset();

    auto status = compute_memory ? compute_memory->compute_async(gf) :
        ggml_backend_sched_graph_compute_async(sched.get(), gf);
    if (status != GGML_STATUS_SUCCESS) {
        // A partial/aborted dispatch keeps its guard closed; recreate it before another request.
        if (compute_memory) sched_reserve_state.invalidate();
        LLAMA_LOG_ERROR("%s: ggml_backend_sched_graph_compute_async failed with error %d\n", __func__, status);
    }

    // fprintf(stderr, "splits: %d\n", ggml_backend_sched_get_n_splits(sched));

    return status;
}

bool llama_context::sched_copy_experts(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph, void * user_data) {
    auto * lctx = static_cast<llama_context *>(user_data);

    // the slot maps of the MoE cache
    if (lctx->moe_cache && lctx->moe_cache->copy(backend, src, dst, graph)) {
        return true;
    }

    auto & st = lctx->copy_experts;

    // the ids must be computed before the split starts, so only the first node of the split is considered
    if (ggml_graph_n_nodes(graph) == 0) {
        return false;
    }
    const ggml_tensor * node = ggml_graph_node(graph, 0);
    if (node->op != GGML_OP_MUL_MAT_ID || node->src[0] != dst) {
        return false;
    }

    const ggml_tensor * ids = node->src[2];
    if (ggml_nelements(ids) == 0) {
        return true;
    }

    const int64_t n_expert    = src->ne[2];
    const size_t  expert_size = src->nb[2];

    if (ids != st.ids || (int64_t) st.used.size() != n_expert) {
        st.ids_data.resize(ggml_nbytes(ids)/sizeof(int32_t));
        ggml_backend_tensor_get_async(backend, ids, st.ids_data.data(), 0, ggml_nbytes(ids));
        ggml_backend_synchronize(backend);

        st.used.assign(n_expert, false);
        for (int64_t i1 = 0; i1 < ids->ne[1]; i1++) {
            for (int64_t i0 = 0; i0 < ids->ne[0]; i0++) {
                const int32_t id = st.ids_data[i1*ids->nb[1]/sizeof(int32_t) + i0*ids->nb[0]/sizeof(int32_t)];
                GGML_ASSERT(id >= 0 && id < n_expert);
                st.used[id] = true;
            }
        }

        st.ids = ids;
    }

    // group consecutive experts and copy them together
    for (int64_t first = 0; first < n_expert; ) {
        if (!st.used[first]) {
            first++;
            continue;
        }
        int64_t last = first;
        while (last + 1 < n_expert && st.used[last + 1]) {
            last++;
        }

        // the experts in the MoE cache are copied from device memory, the others are uploaded
        int64_t next = first;
        for (int64_t e = first; e <= last && lctx->moe_cache; ) {
            const int64_t n = lctx->moe_cache->copy_experts(backend, src, dst, e, last);
            if (n == 0) {
                e++;
                continue;
            }
            if (next < e) {
                ggml_backend_tensor_set_async(backend, dst, (const uint8_t *) src->data + next*expert_size, next*expert_size, (e - next)*expert_size);
            }
            e   += n;
            next = e;
        }

        // copy a bit extra to ensure there are no NaNs in the padding of the last expert, this is necessary for MMQ in the CUDA backend
        const size_t offset  = next*expert_size;
        const size_t padding = last < n_expert - 1 ? std::min<size_t>(expert_size, 512) : 0;
        const size_t size    = (last + 1 - next)*expert_size + padding;
        if (size > 0) {
            ggml_backend_tensor_set_async(backend, dst, (const uint8_t *) src->data + offset, offset, size);
        }

        first = last + 1;
    }

    return true;
}

llm_graph_cb llama_context::graph_get_cb() const {
    return [&](const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il) {
        if (il >= 0) {
            ggml_format_name(cur, "%s-%d", name, il);
        } else {
            ggml_set_name(cur, name);
        }

        // - norm may be automatically assigned to the backend of the previous layer, increasing data transfer between backends
        // - force the last op of the layer on the specified backend to avoid running it on the backend of the next layer due to scheduling
        // FIXME: fix in ggml_backend_sched
        const bool full_offload = model.n_gpu_layers() > model.hparams.n_layer_all;
        if (ubatch.n_tokens < 32 || full_offload) {
            if (il != -1 && (strcmp(name, "norm") == 0 || strcmp(name, "l_last") == 0)) {
                const auto & dev_layer = model.dev_layer(il);
                for (const auto & backend : backends) {
                    if (ggml_backend_get_device(backend.get()) == dev_layer) {
                        if (ggml_backend_supports_op(backend.get(), cur)) {
                            ggml_backend_sched_set_tensor_backend(sched.get(), cur, backend.get());
                        }
                    }
                }
            }
        }
    };
}

//
// state save/load
//

class llama_io_write_dummy : public llama_io_write_i {
public:
    llama_io_write_dummy(bool skip_tensors) : skip_tensors(skip_tensors) {}

    void write(const void * /* src */, size_t size) override {
        size_written += size;
    }

    void write_tensor(ggml_tensor * /* tensor */, size_t /* offset */, size_t size) override {
        if (skip_tensors) {
            return;
        }

        size_written += size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    const bool skip_tensors;

    size_t size_written = 0;
};

class llama_io_write_host : public llama_io_write_i {
public:
    llama_io_write_host(
            uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    ~llama_io_write_host() {
        // TODO: add backend support to batch tensor_get? or some other way to speed this up
        for (const auto & winfo : winfos) {
            ggml_backend_tensor_get(winfo.tensor, winfo.ptr, winfo.offset, winfo.size);
        }
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save the write for later during destruction
        winfos.push_back({tensor, ptr, size, offset});

        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<write_info> winfos;
};

class llama_io_read_host : public llama_io_read_i {
public:
    llama_io_read_host(const uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    ~llama_io_read_host() {
        flush_tensor_reads();
    }

    void flush_tensor_reads() override {
        for (const auto & rinfo : rinfos) {
            ggml_backend_tensor_set(rinfo.tensor, rinfo.ptr, rinfo.offset, rinfo.size);
        }
        rinfos.clear();
    }

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save for later during destruction
        rinfos.push_back({tensor, ptr, size, offset});

        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void discard() override {
        rinfos.clear();
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;

    struct read_info {
        ggml_tensor * tensor;
        const uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<read_info> rinfos;
};

class llama_io_write_file : public llama_io_write_i {
public:
    llama_io_write_file(llama_file * f) : file(f) {}

    void write(const void * src, size_t size) override {
        file->write_raw(src, size);
        size_written += size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        ggml_backend_tensor_get(tensor, temp_buffer.data(), offset, size);
        write(temp_buffer.data(), temp_buffer.size());
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    llama_file * file;
    size_t size_written = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_read_file : public llama_io_read_i {
public:
    llama_io_read_file(llama_file * f) : file(f) {}

    void read(void * dst, size_t size) override {
        file->read_raw(dst, size);
        size_read += size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        read(temp_buffer.data(), size);
        ggml_backend_tensor_set(tensor, temp_buffer.data(), offset, size);
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    llama_file * file;
    size_t size_read = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_write_device : public llama_io_write_i {
public:
    llama_io_write_device(uint8_t * p, size_t len, llama_memory_buffers & mbufs) : ptr(p), buf_size(len), mbufs(mbufs)  {
    }

    ~llama_io_write_device() {
        llama_memory_buffers mbufs_new;

        for (const auto & winfo : winfos) {
            auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);

            mbufs_new[buft].n_tensors++;
            mbufs_new[buft].total_size += winfo.size;
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            ggml_init_params params = {
                /*.mem_size   =*/ 2*mbuf.n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            mbuf.ctx.reset(ggml_init(params));

            mbuf.org.reserve(mbuf.n_tensors);
            mbuf.cpy.reserve(mbuf.n_tensors);
        }

        for (const auto & winfo : winfos) {
            auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);

            const int64_t n = winfo.size/ggml_element_size(winfo.tensor);

            auto & mbuf = mbufs_new[buft];

            mbuf.org.push_back(ggml_view_1d      (mbuf.ctx.get(), winfo.tensor, n, winfo.offset));
            mbuf.cpy.push_back(ggml_new_tensor_1d(mbuf.ctx.get(), winfo.tensor->type, n));
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            auto & mbuf_cur = mbufs[buft];

            bool need_alloc = false;

            need_alloc = need_alloc || (!mbuf_cur.buf);
            need_alloc = need_alloc || (mbuf_cur.org.size() != mbuf.org.size());
            need_alloc = need_alloc || (mbuf_cur.total_size != mbuf.total_size);

            if (!need_alloc) {
                for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                    auto * org0 = mbuf_cur.org[i];
                    auto * org1 = mbuf.org[i];

                    if (!ggml_are_same_shape(org0, org1)) {
                        need_alloc = true;
                        break;
                    }

                    if (org0->view_src != org1->view_src || org0->view_offs != org1->view_offs) {
                        need_alloc = true;
                        break;
                    }
                }
            }

            if (need_alloc) {
                if (!mbuf_cur.buf || mbuf_cur.total_size != mbuf.total_size) {
                    mbuf_cur = std::move(mbuf);

                    mbuf_cur.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(mbuf_cur.ctx.get(), buft));

                    LLAMA_LOG_INFO("%s: allocated '%s' buffer %.3f MiB\n", __func__, ggml_backend_buft_name(buft), mbuf.total_size/1024.0/1024.0);
                } else {
                    //LLAMA_LOG_INFO("%s: reallocating tensors in '%s' buffer %.3f MiB\n", __func__, ggml_backend_buft_name(buft), mbuf.total_size/1024.0/1024.0);

                    // save the old buffer and allocate the new tensors in it
                    auto buf = std::move(mbuf_cur.buf);

                    mbuf_cur = std::move(mbuf);

                    ggml_tallocr talloc = ggml_tallocr_new(buf.get());

                    for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                        ggml_backend_view_init(mbuf_cur.org[i]);
                        ggml_tallocr_alloc(&talloc, mbuf_cur.cpy[i]);
                    }

                    mbuf_cur.buf = std::move(buf);
                }
            }

            for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                ggml_backend_tensor_copy(mbuf_cur.org[i], mbuf_cur.cpy[i]);
            }
        }
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        // save the write for later during destruction
        winfos.push_back({tensor, ptr, size, offset});
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<write_info> winfos;

    llama_memory_buffers & mbufs;
};

class llama_io_read_device : public llama_io_read_i {
public:
    llama_io_read_device(const uint8_t * p, size_t len, const llama_memory_buffers & mbufs) : ptr(p), buf_size(len), mbufs(mbufs) {
    }

    ~llama_io_read_device() {
        llama_memory_buffers mbufs_new;

        for (const auto & rinfo : rinfos) {
            auto * buft = ggml_backend_buffer_get_type(rinfo.tensor->buffer);

            mbufs_new[buft].n_tensors++;
            mbufs_new[buft].total_size += rinfo.size;
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            ggml_init_params params = {
                /*.mem_size   =*/ mbuf.n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            mbuf.ctx.reset(ggml_init(params));

            mbuf.org.reserve(mbuf.n_tensors);
        }

        for (const auto & rinfo : rinfos) {
            auto * buft = ggml_backend_buffer_get_type(rinfo.tensor->buffer);

            const int64_t n = rinfo.size/ggml_element_size(rinfo.tensor);

            auto & mbuf = mbufs_new[buft];

            mbuf.org.push_back(ggml_view_1d(mbuf.ctx.get(), rinfo.tensor, n, rinfo.offset));

            ggml_backend_view_init(mbuf.org.back());
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            const auto & mbuf_cur = mbufs.at(buft);

            if (!mbuf_cur.buf || mbuf_cur.total_size != mbuf.total_size) {
                GGML_ABORT("%s: memory buffer mismatch\n", __func__);
            }

            if (mbuf_cur.n_tensors == mbuf.n_tensors) {
                // an equal tensor count does not imply the same chunking, e.g. save ranges [2,1] vs restore runs [1,2]
                bool same_chunking = true;
                for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                    if (ggml_nbytes(mbuf_cur.cpy[i]) != ggml_nbytes(mbuf.org[i])) {
                        same_chunking = false;
                        break;
                    }
                }

                if (same_chunking) {
                    // same chunking: copy 1:1 by index
                    for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                        ggml_backend_tensor_copy(mbuf_cur.cpy[i], mbuf.org[i]);
                    }
                    continue;
                }
            }

            // different chunking: copy the write-side data (mbuf_cur.cpy) into the read-side targets (mbuf.org)
            // with a byte cursor. Write and read enumerate the same logical data in the same order but may chunk
            // it differently (even with an equal number of tensors), so copy across tensor boundaries rather than
            // 1:1 by index.
            const size_t total = mbuf_cur.total_size;

            ggml_init_params params_scratch = {
                /*.mem_size   =*/ 2*(mbuf_cur.cpy.size() + mbuf.org.size())*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };
            ggml_context * ctx_scratch = ggml_init(params_scratch);

            size_t src_pos  = 0;
            size_t dst_pos  = 0;
            size_t src_j    = 0;
            size_t dst_i    = 0;
            size_t src_base = 0;
            size_t dst_base = 0;

            while (src_pos < total) {
                const auto & src_t = mbuf_cur.cpy[src_j];
                const auto & dst_t = mbuf.org[dst_i];

                const size_t src_size = ggml_nbytes(src_t);
                const size_t dst_size = ggml_nbytes(dst_t);

                const size_t src_off  = src_pos - src_base;
                const size_t dst_off  = dst_pos - dst_base;

                const size_t n_copy = std::min(src_size - src_off, dst_size - dst_off);

                const size_t   el   = ggml_element_size(src_t);
                const int64_t n_el = (int64_t) (n_copy / el);

                auto * src_v = ggml_view_1d(ctx_scratch, src_t, n_el, src_off);
                ggml_backend_view_init(src_v);
                auto * dst_v = ggml_view_1d(ctx_scratch, dst_t, n_el, dst_off);
                ggml_backend_view_init(dst_v);

                ggml_backend_tensor_copy(src_v, dst_v);

                src_pos += n_copy;
                dst_pos += n_copy;

                if (src_pos - src_base == src_size) {
                    src_base = src_pos;
                    ++src_j;
                }
                if (dst_pos - dst_base == dst_size) {
                    dst_base = dst_pos;
                    ++dst_i;
                }
            }

            GGML_ASSERT(src_pos == total && dst_pos == total);
            // any tensors left unvisited hold no data
            for (size_t i = src_j; i < mbuf_cur.cpy.size(); ++i) {
                GGML_ASSERT(ggml_nbytes(mbuf_cur.cpy[i]) == 0);
            }
            for (size_t i = dst_i; i < mbuf.org.size(); ++i) {
                GGML_ASSERT(ggml_nbytes(mbuf.org[i]) == 0);
            }

            ggml_free(ctx_scratch);
        }

        GGML_ASSERT(buf_size == 0);
    }

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        // save for later during destruction
        rinfos.push_back({tensor, ptr, size, offset});
    }

    void discard() override {
        rinfos.clear();
        buf_size = 0;
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;

    struct read_info {
        ggml_tensor * tensor;
        const uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<read_info> rinfos;

    const llama_memory_buffers & mbufs;
};

size_t llama_context::state_get_size() {
    llama_io_write_dummy io(false);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_get_data(uint8_t * dst, size_t size) {
    llama_io_write_host io(dst, size);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_set_data(const uint8_t * src, size_t size) {
    llama_io_read_host io(src, size);
    try {
        return state_read_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        io.discard();
        return 0;
    }
}

static constexpr uint32_t io_magic = 0xaf143cd8;

size_t llama_context::state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags) {
    if (cparams.kv_streaming() && (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE)) {
        LLAMA_LOG_ERROR("%s: KV streaming does not yet support device-native snapshots\n",__func__); return 0;
    }
    if ((flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) && cparams.n_rs_seq && model.arch == LLM_ARCH_QWEN35 && memory) {
        auto * recurrent=kv_resources.recurrent;
        if (recurrent && !recurrent->materialize_spill_for_device_state()) return 0;
    }

    llama_io_write_dummy io(flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);

    try {
        io.write(&io_magic, sizeof(io_magic));
        io.write(&seq_id, sizeof(seq_id));

        return state_seq_write_data(io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_get_data(llama_seq_id seq_id, uint8_t * dst, size_t size, llama_state_seq_flags flags) {
    if (cparams.kv_streaming() && (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE)) {
        LLAMA_LOG_ERROR("%s: KV streaming does not yet support device-native snapshots\n",__func__); return 0;
    }
    if ((flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) && cparams.n_rs_seq && model.arch == LLM_ARCH_QWEN35 && memory) {
        auto * recurrent=kv_resources.recurrent;
        if (recurrent && !recurrent->materialize_spill_for_device_state()) return 0;
    }

    std::unique_ptr<llama_io_write_i> io;
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        io = std::make_unique<llama_io_write_device>(dst, size, mem_storage[seq_id]);
    } else {
        io = std::make_unique<llama_io_write_host>(dst, size);
    }

    try {
        io->write(&io_magic, sizeof(io_magic));
        io->write(&seq_id, sizeof(seq_id));

        return state_seq_write_data(*io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags) {
    if (cparams.kv_streaming() && (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE)) {
        LLAMA_LOG_ERROR("%s: KV streaming does not support device-native snapshots\n",__func__); return 0;
    }
    std::unique_ptr<llama_io_read_i> io;
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        // create a temporary io to read the magic and the src seq_id
        io = std::make_unique<llama_io_read_host>(src, size);

        uint32_t magic_read;
        io->read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        llama_seq_id seq_id_read;
        io->read(&seq_id_read, sizeof(seq_id_read));

        GGML_ASSERT(mem_storage.find(seq_id_read) != mem_storage.end());

        io = std::make_unique<llama_io_read_device>(src, size, mem_storage[seq_id_read]);
    } else {
        io = std::make_unique<llama_io_read_host>(src, size);
    }

    try {
        uint32_t magic_read;
        io->read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        llama_seq_id seq_id_read;
        io->read(&seq_id_read, sizeof(seq_id_read));

        return state_seq_read_data(*io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        io->discard();
        return 0;
    }
}

bool llama_context::state_load_file(const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // sanity checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_SESSION_MAGIC || version != LLAMA_SESSION_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for session file: %08x, %08x\n", __func__, magic, version);
            return false;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in session file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return false;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t n_state_size_cur = file.size() - file.tell();

        llama_io_read_file io( &file);
        const size_t n_read = state_read_data(io);

        if (n_read != n_state_size_cur) {
            LLAMA_LOG_ERROR("%s: did not read all of the session file data! size %zu, got %zu\n", __func__, n_state_size_cur, n_read);
            return false;
        }
    }

    return true;
}

bool llama_context::state_save_file(const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_SESSION_MAGIC);
    file.write_u32(LLAMA_SESSION_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_write_data(io);

    return true;
}

size_t llama_context::state_seq_load_file(llama_seq_id seq_id, const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // version checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_STATE_SEQ_MAGIC || version != LLAMA_STATE_SEQ_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for sequence state file: %08x, %08x\n", __func__, magic, version);
            return 0;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (tokens_out == nullptr) {
            const size_t n_token_max = (file.size() - file.tell()) / sizeof(llama_token);
            if (n_token_count > n_token_max) {
                LLAMA_LOG_ERROR("%s: token count in sequence state file exceeds the file size! %u > %zu\n", __func__, n_token_count, n_token_max);
                return 0;
            }

            *n_token_count_out = n_token_count;
            return file.tell();
        }

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in sequence state file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return 0;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t state_size = file.size() - file.tell();
        llama_io_read_file io(&file);
        const size_t nread = state_seq_read_data(io, seq_id, 0);
        if (!nread) {
            LLAMA_LOG_ERROR("%s: failed to restore sequence state\n", __func__);
            return 0;
        }
        GGML_ASSERT(nread <= state_size);
        GGML_ASSERT(nread + sizeof(uint32_t) * 3 + sizeof(llama_token) * *n_token_count_out == file.tell());
    }

    return file.tell();
}

size_t llama_context::state_seq_save_file(llama_seq_id seq_id, const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_STATE_SEQ_MAGIC);
    file.write_u32(LLAMA_STATE_SEQ_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_seq_write_data(io, seq_id, 0);

    const size_t res = file.tell();
    GGML_ASSERT(res == sizeof(uint32_t) * 3 + sizeof(llama_token) * n_token_count + io.n_bytes());

    return res;
}

size_t llama_context::state_write_data(llama_io_write_i & io) {
    LLAMA_LOG_DEBUG("%s: writing state\n", __func__);

    // write model info
    {
        LLAMA_LOG_DEBUG("%s: - writing model info\n", __func__);

        const std::string arch_str = llm_arch_name(model.arch);
        io.write_string(arch_str);
        // TODO: add more model-specific info which should prevent loading the session file if not identical
    }

    if (memory != nullptr) {
        LLAMA_LOG_DEBUG("%s: - writing memory module\n", __func__);
        memory->state_write(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_read_data(llama_io_read_i & io) {
    clear_serial_features();
    LLAMA_LOG_DEBUG("%s: reading state\n", __func__);

    // read model info
    {
        LLAMA_LOG_DEBUG("%s: - reading model info\n", __func__);

        const std::string cur_arch_str = llm_arch_name(model.arch);

        std::string arch_str;
        io.read_string(arch_str);
        if (cur_arch_str != arch_str) {
            throw std::runtime_error(format("wrong model arch: '%s' instead of '%s'", arch_str.c_str(), cur_arch_str.c_str()));
        }
        // TODO: add more info which needs to be identical but which is not verified otherwise
    }

    if (memory) {
        LLAMA_LOG_DEBUG("%s: - reading memory module\n", __func__);

        memory->state_read(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if (memory) {
        memory->state_write(io, seq_id, flags);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_read_data(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    clear_serial_features();

    if (memory) {
        memory->state_read(io, seq_id, flags);
    }

    return io.n_bytes();
}

//
// perf
//

llama_perf_context_data llama_context::perf_get_data() const {
    llama_perf_context_data data = {};

    data.t_start_ms  = 1e-3 * t_start_us;
    data.t_load_ms   = 1e-3 * t_load_us;
    data.t_p_eval_ms = 1e-3 * t_p_eval_us;
    data.t_eval_ms   = 1e-3 * t_eval_us;
    data.n_p_eval    = std::max(1, n_p_eval);
    data.n_eval      = std::max(1, n_eval);
    data.n_reused    = std::max(0, n_reused);

    return data;
}

void llama_context::perf_reset() {
    t_start_us  = ggml_time_us();
    t_eval_us   = n_eval = 0;
    t_p_eval_us = n_p_eval = 0;
    n_reused    = 0;
}

llama_memory_breakdown llama_context::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data> ret;
    for (const auto & [buft, size] : model.memory_breakdown()) {
        ret[buft].model += size;
    }
    if (memory) {
        for (const auto & [buft, size] : memory->memory_breakdown()) {
            ret[buft].context += size;
        }
    }
    if (moe_cache) {
        for (const auto & [buft, size] : moe_cache->memory_breakdown()) {
            ret[buft].context += size;
        }
    }
    if (no_alloc()) {
        for (size_t i = 0; i < backends.size(); ++i) {
            ggml_backend_t             backend = backends[i].get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += backend_buf_exp_size[i];
        }
    } else {
        for (const auto & backend_ptr : backends) {
            ggml_backend_t             backend = backend_ptr.get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
    }
    return ret;
}

//
// training
//

static void llama_set_param(struct ggml_tensor * tensor, llama_opt_param_filter param_filter, void * userdata) {
    if (!tensor || tensor->type != GGML_TYPE_F32) {
        return;
    }
    if (!param_filter(tensor, userdata)) {
        return;
    }
    if (strcmp(tensor->name, "token_embd.weight") == 0) {
        return; // FIXME
    }
    if (strcmp(tensor->name, "rope_freqs.weight") == 0) {
        return; // FIXME
    }
    ggml_set_param(tensor);
}

void llama_context::opt_init(struct llama_model * model, struct llama_opt_params lopt_params) {
    GGML_ASSERT(!opt_ctx);
    model->hparams.n_ctx_train = lopt_params.n_ctx_train > 0 ? lopt_params.n_ctx_train : n_ctx();
    const uint32_t n_batch     = std::min(this->n_batch(),  model->hparams.n_ctx_train);
    const uint32_t n_ubatch    = std::min(this->n_ubatch(), n_batch);
    GGML_ASSERT(model->hparams.n_ctx_train % n_batch  == 0);
    GGML_ASSERT(n_batch                    % n_ubatch == 0);

    if (cparams.flash_attn) {
        LLAMA_LOG_INFO("%s: disabling flash attention, FLASH_ATTN_EXT has no backward pass\n", __func__);
        cparams.flash_attn = false;
    }

    // gradients cannot flow through the KV cache, so the attention reads the K and V of the current ubatch directly
    if (n_ubatch == cparams.n_ctx) {
        cparams.training = true;
    } else {
        LLAMA_LOG_WARN("%s: n_ubatch (%u) != n_ctx (%u), the K and V projections will not receive gradients\n", __func__, n_ubatch, cparams.n_ctx);
    }

    // the training graph is different, need to reserve again
    sched_reserve_state.invalidate();
    sched_reserve();

    ggml_opt_params opt_params = ggml_opt_default_params(sched.get(), GGML_OPT_LOSS_TYPE_CROSS_ENTROPY);
    opt_params.opt_period      = n_batch / n_ubatch;
    opt_params.get_opt_pars    = lopt_params.get_opt_pars;
    opt_params.get_opt_pars_ud = lopt_params.get_opt_pars_ud;
    opt_params.optimizer       = lopt_params.optimizer_type;
    opt_ctx = ggml_opt_init(opt_params);

    llama_opt_param_filter param_filter = lopt_params.param_filter;
    void * param_filter_ud              = lopt_params.param_filter_ud;

  //llama_set_param(model->tok_embd,        param_filter, param_filter_ud); // FIXME
    llama_set_param(model->type_embd,       param_filter, param_filter_ud);
    llama_set_param(model->pos_embd,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm_b,      param_filter, param_filter_ud);
    llama_set_param(model->output_norm,     param_filter, param_filter_ud);
    llama_set_param(model->output_norm_b,   param_filter, param_filter_ud);
    llama_set_param(model->output,          param_filter, param_filter_ud);
    llama_set_param(model->output_b,        param_filter, param_filter_ud);
    llama_set_param(model->output_norm_enc, param_filter, param_filter_ud);
    llama_set_param(model->cls,             param_filter, param_filter_ud);
    llama_set_param(model->cls_b,           param_filter, param_filter_ud);
    llama_set_param(model->cls_out,         param_filter, param_filter_ud);
    llama_set_param(model->cls_out_b,       param_filter, param_filter_ud);
    llama_set_param(model->cls_norm,        param_filter, param_filter_ud);

    for (struct llama_layer & layer : model->layers) {
        for (size_t i = 0; i < sizeof(layer)/sizeof(struct ggml_tensor *); ++i) {
            llama_set_param(reinterpret_cast<struct ggml_tensor **>(&layer)[i], param_filter, param_filter_ud);
        }
    }
}

void llama_context::opt_epoch_iter(
        ggml_opt_dataset_t               dataset,
        ggml_opt_result_t                result,
        const std::vector<llama_token> & tokens,
        const std::vector<llama_token> & labels_sparse,
        llama_batch                    & batch,
        ggml_opt_epoch_callback          callback,
        bool                             train,
        int64_t                          idata_in_loop,
        int64_t                          ndata_in_loop,
        int64_t                          t_loop_start) {
    GGML_ASSERT(opt_ctx);
    const uint32_t n_ctx    = llama_model_n_ctx_train(&model);
    const uint32_t n_batch  = std::min(this->n_batch(),  n_ctx);
    const uint32_t n_ubatch = std::min(this->n_ubatch(), n_batch);

    memory->clear(true);

    for (uint32_t pos_ctx = 0; pos_ctx < n_ctx; pos_ctx += n_batch) {
        batch.n_tokens = n_batch;
        for (uint32_t pos_batch = 0; pos_batch < n_batch; ++pos_batch) {
            batch.token   [pos_batch]    = tokens[pos_ctx + pos_batch];
            batch.pos     [pos_batch]    = pos_ctx + pos_batch;
            batch.n_seq_id[pos_batch]    = 1;
            batch.seq_id  [pos_batch][0] = 0;
            batch.logits  [pos_batch]    = true;
        }

        // TODO: use llama_batch_ext here
        {
            llama_batch_compat compat(this, batch);
            if (!balloc->init(*compat.batch_ext, model.vocab, true)) {
                LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
                return;
            }
        }

        const uint32_t n_tokens_all = balloc->get_n_tokens();

        n_queued_tokens += n_tokens_all;

        embd_seq.clear();

        uint32_t n_outputs_all = n_tokens_all;

        auto mctx = memory->init_batch(*balloc, cparams.n_ubatch, true);
        if (!mctx || mctx->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
            LLAMA_LOG_ERROR("%s: could not initialize batch\n", __func__);
            break;
        }

        // reserve output buffer
        if (output_reserve(n_outputs_all) < n_outputs_all) {
            LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
            GGML_ABORT("TODO: handle this error");
        };

        uint32_t pos_batch = 0;
        do {
            const auto & ubatch = mctx->get_ubatch();

            n_outputs = ubatch.n_tokens;

            if (!mctx->apply()) {
                LLAMA_LOG_ERROR("%s: failed to update the memory context\n", __func__);
                break;
            }

            auto * res = get_gf_res_prev();

            const auto gparams = graph_params(res, ubatch, mctx.get(), ctx_type_to_graph_type(cparams.ctx_type));

            // the optimizer graph is allocated outside sched, so the next decode must rebuild
            gf_res_prev_active = nullptr;
            res->reset();

            auto * gf = model.build_graph(gparams);

            struct ggml_context * ctx_compute_opt;
            {
                const size_t size_gf = ggml_graph_size(gf);
                const size_t size_meta = 4*size_gf*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(size_gf, /*grads = */ true);
                struct ggml_init_params params = {
                    /*.mem_size   =*/ size_meta,
                    /*.mem_buffer =*/ nullptr,
                    /*.no_alloc   =*/ true,
                };
                ctx_compute_opt = ggml_init(params);
            }
            ggml_opt_prepare_alloc(opt_ctx, ctx_compute_opt, gf, res->get_inp_tokens(), res->get_logits());
            ggml_opt_alloc(opt_ctx, train);

            res->set_inputs(&ubatch);
            {
                struct ggml_tensor * labels = ggml_opt_labels(opt_ctx);
                GGML_ASSERT(labels->ne[1] == n_ubatch);
                ggml_set_zero(labels);
                const float onef = 1.0f;
                for (uint32_t pos_ubatch = 0; pos_ubatch < n_ubatch; ++pos_ubatch) {
                    const uint32_t ilabel = pos_ctx + pos_batch + pos_ubatch;
                    GGML_ASSERT(labels_sparse[ilabel] < labels->ne[0]);
                    ggml_backend_tensor_set(labels, &onef, (pos_ubatch*labels->ne[0] + labels_sparse[ilabel])*sizeof(float), sizeof(float));
                }
            }
            copy_experts.reset();
            ggml_opt_eval(opt_ctx, result);
            if (callback) {
                callback(train, opt_ctx, dataset, result, idata_in_loop + (pos_ctx + pos_batch)/n_ubatch + 1, ndata_in_loop, t_loop_start);
            }
            ggml_free(ctx_compute_opt);

            pos_batch += ubatch.n_tokens;
        } while (mctx->next());
    }
}

void llama_context::opt_epoch(
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    const uint32_t n_ctx    = this->n_ctx();
    const uint32_t n_batch  = std::min(cparams.n_batch,  n_ctx);
    const uint32_t n_ubatch = std::min(cparams.n_ubatch, n_batch);
    const  int64_t ndata    = ggml_opt_dataset_ndata(dataset);

    GGML_ASSERT(idata_split >= 0);
    GGML_ASSERT(idata_split <= ndata);

    const uint32_t ubatch_per_ctx = n_ctx / n_ubatch;

    struct llama_batch batch = llama_batch_init(n_batch, 0, 1);
    std::vector<llama_token>        tokens(n_ctx);
    std::vector<llama_token> labels_sparse(n_ctx);

    int64_t idata = 0;

    int64_t t_loop_start = ggml_time_us();
    int64_t ndata_in_loop = idata_split*ubatch_per_ctx;
    for (; idata < idata_split; ++idata) {
        constexpr bool train = true;
        const int64_t idata_in_loop = idata*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_train, tokens, labels_sparse, batch,
            callback_train, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    t_loop_start = ggml_time_us();
    ndata_in_loop = (ndata - idata_split)*ubatch_per_ctx;
    for (; idata < ndata; ++idata) {
        constexpr bool train = false;
        const int64_t idata_in_loop = (idata - idata_split)*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_eval, tokens, labels_sparse, batch,
            callback_eval, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    llama_batch_free(batch);
}

//
// interface implementation
//

llama_context_params llama_context_default_params() {
    llama_context_params result = {
        /*.n_ctx                       =*/ 512,
        /*.n_batch                     =*/ 2048,
        /*.n_ubatch                    =*/ 512,
        /*.n_seq_max                   =*/ 1,
        /*.n_rs_seq                    =*/ 0,
        /*.n_outputs_max               =*/ 0,
        /*.n_outputs_max_per_seq       =*/ 1,
        /*.n_threads                   =*/ GGML_DEFAULT_N_THREADS, // TODO: better default
        /*.n_threads_batch             =*/ GGML_DEFAULT_N_THREADS,
        /*.ctx_type                    =*/ LLAMA_CONTEXT_TYPE_DEFAULT,
        /*.rope_scaling_type           =*/ LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
        /*.pooling_type                =*/ LLAMA_POOLING_TYPE_UNSPECIFIED,
        /*.attention_type              =*/ LLAMA_ATTENTION_TYPE_UNSPECIFIED,
        /*.flash_attn_type             =*/ LLAMA_FLASH_ATTN_TYPE_AUTO,
        /*.rope_freq_base              =*/ 0.0f,
        /*.rope_freq_scale             =*/ 0.0f,
        /*.yarn_ext_factor             =*/ -1.0f,
        /*.yarn_attn_factor            =*/ -1.0f,
        /*.yarn_beta_fast              =*/ -1.0f,
        /*.yarn_beta_slow              =*/ -1.0f,
        /*.yarn_orig_ctx               =*/ 0,
        /*.defrag_thold                =*/ -1.0f,
        /*.cb_eval                     =*/ nullptr,
        /*.cb_eval_user_data           =*/ nullptr,
        /*.type_k                      =*/ GGML_TYPE_F16,
        /*.type_v                      =*/ GGML_TYPE_F16,
        /*.moe_cache_size              =*/ 0,
        /*.abort_callback              =*/ nullptr,
        /*.abort_callback_data         =*/ nullptr,
        /*.embeddings                  =*/ false,
        /*.offload_kqv                 =*/ true,
        /*.no_perf                     =*/ true,
        /*.op_offload                  =*/ true,
        /*.swa_full                    =*/ true,
        /*.kv_unified                  =*/ false,
        /*.sampler                     =*/ nullptr,
        /*.n_sampler                   =*/ 0,
        /*.ctx_other                   =*/ nullptr,
        /*.kv_stream_pool_bytes        =*/ 0,
        /*.shared_device_memory_bytes   =*/ 0,
        /*.kv_stream_auxiliary_layers   =*/ 0,
        /*.kv_stream_verify_max         =*/ 0,
        /*.recurrent_rollback_ram        =*/ false,
    };

    return result;
}

llama_context * llama_init_from_model(
                 llama_model * model,
        llama_context_params   params,
                    uint32_t   serial_prediction_rows, bool serial_image_inputs) {
    if (!model) {
        LLAMA_LOG_ERROR("%s: model cannot be NULL\n", __func__);
        return nullptr;
    }

    if (params.n_batch == 0 && params.n_ubatch == 0) {
        LLAMA_LOG_ERROR("%s: n_batch and n_ubatch cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.n_ctx == 0 && model->hparams.n_ctx_train == 0) {
        LLAMA_LOG_ERROR("%s: n_ctx and model->hparams.n_ctx_train cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && model->arch == LLM_ARCH_GROK) {
        LLAMA_LOG_WARN("%s: flash_attn is not compatible with Grok - forcing off\n", __func__);
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }

    if (model->split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
            LLAMA_LOG_INFO("%s: enabling flash_attn since it is required for SPLIT_MODE_TENSOR\n", __func__);
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
            LLAMA_LOG_ERROR("%s: SPLIT_MODE_TENSOR requires flash_attn to be enabled\n", __func__);
            return nullptr;
        }
        if (model->get_split_state_ud.n_devices == 1) {
            LLAMA_LOG_WARN("%s: SPLIT_MODE_TENSOR being used for a single device is not recommended\n", __func__);
        }
    }

    if ((model->hparams.is_mla() || model->arch == LLM_ARCH_DEEPSEEK4) && params.type_k != params.type_v) {
        LLAMA_LOG_ERROR("%s: model does not support different K (%s) and V (%s) cache types\n", __func__, ggml_type_name(params.type_k), ggml_type_name(params.type_v));
        return nullptr;
    }

    if (ggml_is_quantized(params.type_v) && params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
            LLAMA_LOG_INFO("%s: enabling flash_attn since it is required for quantized V cache\n", __func__);
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED) {
            LLAMA_LOG_ERROR("%s: quantized V cache requires flash_attn to be enabled\n", __func__);
            return nullptr;
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_k)) {
        const uint32_t blck_size = ggml_blck_size(params.type_k);
        for (uint32_t il = 0; il < model->hparams.n_layer(); ++il) {
            if (model->hparams.n_embd_head_k(il) % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: K cache type %s with block size %u does not divide n_embd_head_k=%u\n",
                    __func__, ggml_type_name(params.type_k), blck_size, model->hparams.n_embd_head_k(il));
                return nullptr;
            }
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_v)) {
        const uint32_t blck_size = ggml_blck_size(params.type_v);
        for (uint32_t il = 0; il < model->hparams.n_layer(); ++il) {
            if (model->hparams.n_embd_head_v(il) % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: V cache type %s with block size %u does not divide n_embd_head_v=%u\n",
                    __func__, ggml_type_name(params.type_v), blck_size, model->hparams.n_embd_head_v(il));
                return nullptr;
            }
        }
    }

    if (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED &&
        params.pooling_type != model->hparams.pooling_type) {
        //user-specified pooling-type is different from the model default
        LLAMA_LOG_WARN("%s: model default pooling_type is [%d], but [%d] was specified\n", __func__,
                       model->hparams.pooling_type, params.pooling_type);
    }

    // router_layer >= 0 means n_layer_nextn is repurposed for a router layer, not real MTP
    if (params.ctx_type == LLAMA_CONTEXT_TYPE_MTP &&
        (model->hparams.n_layer_nextn == 0 || model->hparams.router_layer >= 0)) {
        LLAMA_LOG_WARN("%s: context type MTP requested but model doesn't contain MTP layers\n", __func__);
        return nullptr;
    }

    try {
        auto * ctx = new llama_context(*model, params, serial_prediction_rows, serial_image_inputs);
        const auto & cparams = ctx->get_cparams();

        if (cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN && cparams.rope_freq_scale != model->hparams.rope_freq_scale_train) {
            LLAMA_LOG_INFO("%s: custom YaRN scaling detected, re-adjusting n_ctx_train(%u)...\n", __func__, model->hparams.n_ctx_train);
            model->hparams.n_ctx_train = cparams.n_ctx_orig_yarn / cparams.rope_freq_scale;
            LLAMA_LOG_INFO("%s: n_ctx_train adjusted to %u\n", __func__, model->hparams.n_ctx_train);
        }

        return ctx;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to initialize the context: %s\n", __func__, err.what());
    }

    return nullptr;
}

llama_context * llama_init_from_model(llama_model * model, llama_context_params params, uint32_t serial_prediction_rows) {
    return llama_init_from_model(model, params, serial_prediction_rows, false);
}

llama_context * llama_init_from_model(llama_model * model, llama_context_params params) {
    return llama_init_from_model(model, params, 0);
}

// deprecated
llama_context * llama_new_context_with_model(
                 llama_model * model,
        llama_context_params   params) {
    return llama_init_from_model(model, params);
}

void llama_free(llama_context * ctx) {
    delete ctx;
}

uint32_t llama_n_ctx(const llama_context * ctx) {
    return ctx->n_ctx();
}

uint32_t llama_n_ctx_seq(const llama_context * ctx) {
    return ctx->n_ctx_seq();
}

uint32_t llama_n_batch(const llama_context * ctx) {
    return ctx->n_batch();
}

uint32_t llama_n_ubatch(const llama_context * ctx) {
    return ctx->n_ubatch();
}

uint32_t llama_n_seq_max(const llama_context * ctx) {
    return ctx->n_seq_max();
}

uint32_t llama_n_rs_seq(const llama_context * ctx) {
    return ctx->get_cparams().n_rs_seq;
}

const llama_model * llama_get_model(const llama_context * ctx) {
    return &ctx->get_model();
}

enum llama_pooling_type llama_pooling_type(const llama_context * ctx) {
    return ctx->pooling_type();
}

void llama_attach_threadpool(
            llama_context * ctx,
        ggml_threadpool_t   threadpool,
        ggml_threadpool_t   threadpool_batch) {
    ctx->attach_threadpool(threadpool, threadpool_batch);
}

void llama_detach_threadpool(llama_context * ctx) {
    ctx->detach_threadpool();
}

void llama_set_n_threads(llama_context * ctx, int32_t n_threads, int32_t n_threads_batch) {
    ctx->set_n_threads(n_threads, n_threads_batch);
}

int32_t llama_n_threads(llama_context * ctx) {
    return ctx->n_threads();
}

int32_t llama_n_threads_batch(llama_context * ctx) {
    return ctx->n_threads_batch();
}

void llama_set_abort_callback(llama_context * ctx, bool (*abort_callback)(void * data), void * abort_callback_data) {
    ctx->set_abort_callback(abort_callback, abort_callback_data);
}

void llama_set_embeddings(llama_context * ctx, bool embeddings) {
    ctx->set_embeddings(embeddings);
}

void llama_set_causal_attn(llama_context * ctx, bool causal_attn) {
    ctx->set_causal_attn(causal_attn);
}

bool llama_get_causal_attn(const llama_context * ctx) {
    return ctx->get_causal_attn();
}

void llama_set_kv_stream_decode(llama_context * ctx, bool decode) {
    if (ctx) ctx->set_kv_stream_decode(decode);
}
static llama_kv_stream_model * llama_mtp_target_stream(llama_context * ctx) {
    if (!ctx) return nullptr;
    auto * full = ctx->get_kv_resources().full;
    return full ? full->get_kv_stream() : nullptr;
}

bool llama_kv_stream_mtp_prepare(llama_context * ctx, uint32_t future_tokens) {
    if (!ctx || future_tokens > 4) return false;
    if (!ctx->get_cparams().kv_stream_auxiliary_layers) return true;
    if (ctx->get_kv_resources().full && ctx->get_kv_resources().full->get_group_model()) return ctx->group_mtp_ready(future_tokens);
    auto * stream = llama_mtp_target_stream(ctx);
    const auto cache = stream ? stream->auxiliary_cache() : nullptr;
    if (!cache) return false;
    LLAMA_LOG_DEBUG("%s: target=%zu mtp=%zu future=%u active=%d reserved=%zu\n",
        __func__, stream->tokens(), cache->tokens(), future_tokens,
        int(stream->has_mtp_layer()), stream->mtp_reserved_tokens());
    if (cache->tokens() < 4) return true; // ordinary short-prefix MTP path
    const size_t frontier = stream->tokens();
    if (frontier > ctx->n_ctx()) return false;
    const size_t reserve = std::min<size_t>(future_tokens, ctx->n_ctx() - frontier);
    // The final short prompt batch still owns the prefill layout. Retire that
    // workspace before admitting MTP against the larger decode KV grant.
    auto * owner = llama_context_compute_memory(ctx);
    const auto transitions = owner ? owner->phase_transition_count() : 0;
    if (owner && owner->shares_kv_memory() && !owner->prepare_serial_decode()) {
        LLAMA_LOG_ERROR("%s: failed to activate shared decode layout before MTP admission\n", __func__);
        return false;
    }
    if (owner && owner->phase_transition_count() != transitions) llama_log_memory_phase(owner);
    if (stream->has_mtp_layer()) {
        if (stream->mtp_reserved_tokens() >= frontier + reserve) return true;
        if (!stream->release_mtp_layer()) return false;
    }
    if (stream->acquire_mtp_layer(reserve)) return true;
    LLAMA_LOG_WARN("%s: complete MTP span admission failed at target=%zu reserved=%zu\n",
        __func__, frontier, frontier + reserve);
    return false;
}

bool llama_kv_stream_mtp_release(llama_context * ctx) {
    if (!ctx) return false;
    if (!ctx->get_cparams().kv_stream_auxiliary_layers) return true;
    if (ctx->get_kv_resources().full && ctx->get_kv_resources().full->get_group_model()) return ctx->group_mtp_ready(0);
    auto * stream = llama_mtp_target_stream(ctx);
    return stream && stream->release_mtp_layer();
}

bool llama_kv_stream_schedule_next(llama_context * producer, llama_context * consumer, uint32_t future_tokens) {
    if (!producer) return false;
    if (producer->get_cparams().kv_shared_read) {
        auto * parent=producer->get_cparams().ctx_other;
        auto * source=llama_mtp_target_stream(parent);
        if (!source || (consumer && consumer != parent)) return false;
        llama_kv_stream_next_consumer next;
        if (consumer) { next.cache_id=source->host()->cache_id(); next.future_tokens=future_tokens; }
        else if (!producer->get_kv_resources().full || !producer->get_kv_resources().full->shared_stream_consumer(1,next)) return false;
        return source->schedule_next_consumer(llama_kv_stream_access_role::read,next);
    }
    auto * target=llama_mtp_target_stream(producer);
    if (target && consumer && consumer->get_cparams().kv_shared_read && consumer->get_cparams().ctx_other == producer) {
        llama_kv_stream_next_consumer next;
        auto * cache=consumer->get_kv_resources().full;
        return cache && cache->shared_stream_consumer(1,next) &&
            target->schedule_next_consumer(llama_kv_stream_access_role::append,next);
    }
    if (target && !consumer) return target->schedule_next_cache(0,0);
    if (target && producer == consumer) return target->schedule_next_cache(target->host()->cache_id(),future_tokens);
    auto * next_cache=consumer ? dynamic_cast<llama_kv_cache *>(llama_get_memory(consumer)) : nullptr;
    if (target && next_cache && next_cache->mtp_auxiliary_cache() == target->auxiliary_cache() && target->auxiliary_cache())
        return target->schedule_next_cache(target->auxiliary_cache()->identity().id,future_tokens);
    auto * current=dynamic_cast<llama_kv_cache *>(llama_get_memory(producer));
    auto * next_target=llama_mtp_target_stream(consumer);
    if (current && current->mtp_auxiliary_cache() && (!consumer ||
            (next_target && current->mtp_auxiliary_cache() == next_target->auxiliary_cache()))) {
        current->set_next_target(consumer != nullptr);
        return true;
    }
    return false;
}

bool llama_kv_stream_finish_access(llama_context * producer) {
    if (producer && producer->get_cparams().kv_shared_read) {
        auto * source=llama_mtp_target_stream(producer->get_cparams().ctx_other);
        return source && source->finish_read_access();
    }
    auto * cache=producer ? dynamic_cast<llama_kv_cache *>(llama_get_memory(producer)) : nullptr;
    return !cache || cache->finish_serial_access();
}


void llama_set_warmup(llama_context * ctx, bool warmup) {
    ctx->set_warmup(warmup);
}

void llama_synchronize(llama_context * ctx) {
    ctx->synchronize();
}

float * llama_get_logits(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_logits();
}

float * llama_get_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    float * res = nullptr;

    res = ctx->get_sampled_logits_ith(i);

    if (!res) {
        res = ctx->get_logits_ith(i);
    }

    return res;
}

float * llama_get_embeddings(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings();
}

float * llama_get_embeddings_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_ith(i);
}

float * llama_get_embeddings_seq(llama_context * ctx, llama_seq_id seq_id) {
    ctx->synchronize();

    return ctx->get_embeddings_seq(seq_id);
}

void llama_set_embeddings_nextn(llama_context * ctx, bool value, bool masked) {
    ctx->set_embeddings_nextn(value, masked);
}

void llama_set_embeddings_layer_inp(llama_context * ctx, uint32_t lid, bool value) {
    ctx->set_embeddings_layer_inp(lid, value);
}

void llama_set_nextn_layer_offset(llama_context * ctx, int32_t offset) {
    ctx->set_nextn_layer_offset(offset);
}

llama_memory_t llama_get_memory(const struct llama_context * ctx) {
    if (!ctx) {
        return nullptr;
    }

    return ctx->get_memory();
}

float * llama_get_embeddings_nextn(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings_nextn();
}

float * llama_get_embeddings_nextn_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_nextn_ith(i);
}

float * llama_get_embeddings_layer_inp(llama_context * ctx, uint32_t lid) {
    ctx->synchronize();

    return ctx->get_embeddings_layer_inp(lid);
}

bool llama_set_sampler(llama_context * ctx, llama_seq_id seq_id, llama_sampler * smpl) {
    return ctx->set_sampler(seq_id, smpl);
}

llama_token llama_get_sampled_token_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_token_ith(i);
}

float * llama_get_sampled_probs_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_probs_ith(i);
}

float * llama_get_sampled_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_logits_ith(i);
}

llama_token * llama_get_sampled_candidates_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return const_cast<llama_token *>(ctx->get_sampled_candidates_ith(i));
}

uint32_t llama_get_sampled_candidates_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_candidates_count(i));
}

uint32_t llama_get_sampled_logits_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_logits_count(i));
}

uint32_t llama_get_sampled_probs_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_probs_count(i));
}

struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs) {
    auto memory = ctx->get_memory();
    llama_memory_context_ptr mctx;
    if (memory) {
        mctx = memory->init_full();
    }
    return ctx->graph_reserve(n_tokens, n_seqs, n_outputs, mctx.get());
}

// llama adapter API

int32_t llama_set_adapters_lora(
            llama_context * ctx,
            llama_adapter_lora ** adapters,
            size_t n_adapters,
            float * scales) {
    if (adapters == nullptr || scales == nullptr) {
        GGML_ASSERT(n_adapters == 0 && "invalid llama_set_adapters_lora call");
    }

    ctx->set_adapters_lora(adapters, n_adapters, scales);

    return 0;
}

int32_t llama_set_adapter_cvec(
        llama_context * ctx,
          const float * data,
               size_t   len,
              int32_t   n_embd,
              int32_t   il_start,
              int32_t   il_end) {
    bool res = ctx->set_adapter_cvec(data, len, n_embd, il_start, il_end);

    return res ? 0 : -1;
}

//
// memory
//

void llama_memory_clear(llama_memory_t mem, bool data) {
    if (!mem) {
        return;
    }

    mem->clear(data);
}

bool llama_memory_seq_rm(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return true;
    }

    return mem->seq_rm(seq_id, p0, p1);
}

void llama_memory_seq_cp(
        llama_memory_t mem,
          llama_seq_id seq_id_src,
          llama_seq_id seq_id_dst,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return;
    }

    mem->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_seq_keep(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return;
    }

    mem->seq_keep(seq_id);
}

void llama_memory_seq_add(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
             llama_pos delta) {
    if (!mem) {
        return;
    }

    mem->seq_add(seq_id, p0, p1, delta);
}

void llama_memory_seq_div(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
                   int d) {
    if (!mem) {
        return;
    }

    mem->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_seq_pos_min(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_min(seq_id);
}

llama_pos llama_memory_seq_pos_max(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_max(seq_id);
}

bool llama_memory_can_shift(llama_memory_t mem) {
    if (!mem) {
        return false;
    }

    return mem->get_can_shift();
}

// llama state API

// deprecated
size_t llama_get_state_size(llama_context * ctx) {
    return llama_state_get_size(ctx);
}

// deprecated
size_t llama_copy_state_data(llama_context * ctx, uint8_t * dst) {
    return llama_state_get_data(ctx, dst, -1);
}

// deprecated
size_t llama_set_state_data(llama_context * ctx, const uint8_t * src) {
    return llama_state_set_data(ctx, src, -1);
}

// deprecated
bool llama_load_session_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    return llama_state_load_file(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out);
}

// deprecated
bool llama_save_session_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    return llama_state_save_file(ctx, path_session, tokens, n_token_count);
}

// Returns the *actual* size of the state.
// Intended to be used when saving to state to a buffer.
size_t llama_state_get_size(llama_context * ctx) {
    return ctx->state_get_size();
}

size_t llama_state_get_data(llama_context * ctx, uint8_t * dst, size_t size) {
    ctx->synchronize();

    return ctx->state_get_data(dst, size);
}

// Sets the state reading from the specified source address
size_t llama_state_set_data(llama_context * ctx, const uint8_t * src, size_t size) {
    ctx->synchronize();

    return ctx->state_set_data(src, size);
}

bool llama_state_load_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_load_file(path_session, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading session file: %s\n", __func__, err.what());
        return false;
    }
}

bool llama_state_save_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_save_file(path_session, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving session file: %s\n", __func__, err.what());
        return false;
    }
}

size_t llama_state_seq_get_size(llama_context * ctx, llama_seq_id seq_id) {
    return llama_state_seq_get_size_ext(ctx, seq_id, 0);
}

size_t llama_state_seq_get_data(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_get_data_ext(ctx, dst, size, seq_id, 0);
}

size_t llama_state_seq_set_data(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_set_data_ext(ctx, src, size, seq_id, 0);
}

size_t llama_state_seq_get_size_ext(llama_context * ctx, llama_seq_id seq_id, llama_state_seq_flags flags) {
    return ctx->state_seq_get_size(seq_id, flags);
}

size_t llama_state_seq_get_data_ext(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_get_data(seq_id, dst, size, flags);
}
size_t llama_state_seq_set_data_ext(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_set_data(seq_id, src, size, flags);
}

size_t llama_state_seq_save_file(llama_context * ctx, const char * filepath, llama_seq_id seq_id, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_seq_save_file(seq_id, filepath, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_state_seq_load_file(llama_context * ctx, const char * filepath, llama_seq_id dest_seq_id, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_seq_load_file(dest_seq_id, filepath, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

// compat: llama_batch -> llama_batch_ext -> encode/decode

int llama_context::encode(const llama_batch & batch_inp) {
    llama_batch_compat compat(this, batch_inp, model.hparams.n_embd_inp_enc());
    return encode(*compat.batch_ext);
}

int llama_context::decode(const llama_batch & batch_inp) {
    llama_batch_compat compat(this, batch_inp);
    return decode(*compat.batch_ext);
}

///

int32_t llama_encode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->encode(batch);
    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: failed to encode, ret = %d\n", __func__, ret);
    }

    return ret;
}

int32_t llama_decode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->decode(batch);
    if (ret != 0 && ret != 1) {
        LLAMA_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, ret);
    }

    return ret;
}

//
// perf
//

llama_perf_context_data llama_perf_context(const llama_context * ctx) {
    llama_perf_context_data data = {};

    if (ctx == nullptr) {
        return data;
    }

    data = ctx->perf_get_data();

    return data;
}

void llama_perf_context_print(const llama_context * ctx) {
    const auto data = llama_perf_context(ctx);

    const double t_end_ms = 1e-3 * ggml_time_us();

    LLAMA_LOG_INFO("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
    LLAMA_LOG_INFO("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
    LLAMA_LOG_INFO("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
    LLAMA_LOG_INFO("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));
    LLAMA_LOG_INFO("%s:    graphs reused = %10d\n", __func__, data.n_reused);
}

void llama_perf_context_reset(llama_context * ctx) {
    ctx->perf_reset();
}

//
// training
//

bool llama_opt_param_filter_all(const struct ggml_tensor * tensor, void * userdata) {
    GGML_UNUSED(tensor);
    GGML_UNUSED(userdata);
    return true;
}

void llama_opt_init(struct llama_context * ctx, struct llama_model * model, struct llama_opt_params lopt_params) {
    ctx->opt_init(model, lopt_params);
}

void llama_opt_epoch(
        struct llama_context    * ctx,
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    ctx->opt_epoch(
        dataset,
        result_train,
        result_eval,
        idata_split,
        callback_train,
        callback_eval);
}

int32_t llama_process(llama_context * ctx, llama_process_type type, llama_batch_ext * batch) {
    switch (type) {
        case LLAMA_PROCESS_TYPE_ENCODE: return ctx->encode(*batch);
        case LLAMA_PROCESS_TYPE_DECODE: return ctx->decode(*batch);
    }
    return -1;
}

//
// ext
//

llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx) {
    return ctx->memory_breakdown();
}

llama_context * llama_get_ctx_other(struct llama_context * ctx) {
    return ctx->get_cparams().ctx_other;
}

bool llama_context::save_group_sequences(std::vector<llama_context_group_sequence_state> & output) {
    if (!group_execution || group_execution->failed || group_execution->admission ||
            !group_execution->table->projection_current(*group_execution->projection) ||
            (kv_resources.recurrent && kv_resources.recurrent->current_storage_pending()) ||
            (kv_resources.bounded && kv_resources.bounded->bounded_storage_pending())) return false;
    try {
        synchronize();
        const auto save=[](llama_memory_i * memory,llama_seq_id sequence,std::vector<uint8_t> & bytes) {
            if (!memory) return;
            llama_io_write_dummy size(false);memory->state_write(size,sequence,0);
            bytes.resize(size.n_bytes());
            llama_io_write_host data(bytes.data(),bytes.size());memory->state_write(data,sequence,0);
            if (data.n_bytes()!=bytes.size()) throw std::runtime_error("native selected-state size changed during export");
        };
        std::vector<llama_context_group_sequence_state> next;
        next.reserve(group_execution->projection->handles.size());
        for (const auto & handle : group_execution->projection->handles) {
            std::shared_ptr<llama_kv_stream_logical_cache> cache;
            if (!group_execution->table->get(handle,group_execution->projection->role,cache) || !cache->ready()) return false;
            const auto checkpoint=cache->checkpoint();
            const bool block=group_execution->shared_block && model.arch==LLM_ARCH_DFLASH;
            if (block && checkpoint.tokens) {
                if (next.size()>=group_execution->draft_sources.size()) return false;
                const auto derived=group_execution->draft_sources[next.size()];
                if (derived.id!=checkpoint.id || derived.generation!=checkpoint.generation || derived.tokens!=checkpoint.tokens) return false;
            }
            llama_context_group_sequence_state state;
            state.model=&model;state.sequence=llama_seq_id(handle.slot);state.request_epoch=handle.generation;
            state.cache_id=checkpoint.id;state.cache_generation=checkpoint.generation;state.tokens=checkpoint.tokens;
            if (block) {
                if (next.size()>=group_execution->draft_positions.size()) return false;
                state.draft_position=group_execution->draft_positions[next.size()];
                if (state.draft_position<-1 || memory->seq_pos_max(state.sequence)<state.draft_position) return false;
            }
            const auto & native=block ? *group_execution->projection->cells : kv_resources.full->get_cells(0);
            std::vector<bool> seen(state.tokens,false);state.full_columns.reserve(state.tokens);
            for (uint32_t row=0;row<native.used_max_p1();++row) if (!native.is_empty(row) && native.seq_has(row,state.sequence)) {
                const auto column=group_execution->projection->columns[row];
                if (native.seq_count(row)!=1 || column.source!=next.size() || column.token>=state.tokens || seen[column.token]) return false;
                seen[column.token]=true;state.full_columns.push_back(uint32_t(column.token));
            }
            if (state.full_columns.size()!=state.tokens) return false;
            save(kv_resources.recurrent,state.sequence,state.recurrent);
            save(block ? memory.get() : kv_resources.bounded,state.sequence,state.bounded);
            next.push_back(std::move(state));
        }
        output=std::move(next);return true;
    } catch (...) {return false;}
}

bool llama_context::restore_group_sequences(const std::vector<llama_context_group_sequence_state> & states,
        const llama_kv_stream_slot_projection & projection) {
    auto * stream=kv_resources.full ? kv_resources.full->get_group_model() : nullptr;
    const bool block=model.arch==LLM_ARCH_DFLASH && group_target;
    if (!stream && block && group_target->kv_resources.full) stream=group_target->kv_resources.full->get_group_model();
    if (!group_initializing || group_restore_failed || measurement_only || group_execution || !stream ||
            !stream->accepts_projection(projection) || states.size()!=group_selected.size() ||
            projection.handles.size()!=group_selected.size() ||
            (kv_resources.recurrent && kv_resources.recurrent->current_storage_pending()) ||
            (kv_resources.bounded && kv_resources.bounded->bounded_storage_pending())) {
        if (block) LLAMA_LOG_ERROR("%s: draft restore precondition (pending=%d failed=%d measured=%d live=%d stream=%d projection=%d states=%zu selected=%zu bounded=%d)\n",__func__,
            int(group_initializing),int(group_restore_failed),int(measurement_only),int(bool(group_execution)),int(bool(stream)),
            int(stream && stream->accepts_projection(projection)),states.size(),group_selected.size(),
            int(kv_resources.bounded && kv_resources.bounded->bounded_storage_pending()));
        return false;
    }
    try {
        const auto table=stream->source_table();
        for (size_t i=0;i<projection.handles.size();++i) {
            const auto & handle=projection.handles[i];
            if (std::find(group_selected.begin(),group_selected.end(),llama_seq_id(handle.slot))==group_selected.end()) return false;
            const auto found=std::find_if(states.begin(),states.end(),[&](const auto & state) {return state.sequence==llama_seq_id(handle.slot);});
            if (found==states.end() || found->model!=&model || found->request_epoch!=handle.generation ||
                    std::count_if(states.begin(),states.end(),[&](const auto & state) {return state.sequence==found->sequence;})!=1) return false;
            const bool empty=!found->tokens && found->recurrent.empty() && found->bounded.empty();
            if (empty ? memory->seq_pos_max(found->sequence)!=-1 :
                    bool(kv_resources.recurrent)!=!found->recurrent.empty() || bool(kv_resources.bounded)!=!found->bounded.empty()) return false;
            std::shared_ptr<llama_kv_stream_logical_cache> cache;
            if (!table->get(handle,projection.role,cache) || !cache->ready()) return false;
            const auto checkpoint=cache->checkpoint();
            if (found->cache_id!=checkpoint.id || found->cache_generation!=checkpoint.generation || found->tokens!=checkpoint.tokens ||
                    found->full_columns.size()!=found->tokens) return false;
            if (block) {
                std::shared_ptr<llama_kv_cells> cells;llama_pos next=0;
                if (found->draft_position<-1 || !table->get_cells(handle,0,cells) ||
                        !group_next_position(*cells,checkpoint.tokens,next) || found->draft_position>=next) return false;
            }
            size_t ordinal=0;std::vector<bool> seen(found->tokens,false);
            for (uint32_t row=0;row<projection.cells->used_max_p1();++row) if (!projection.cells->is_empty(row) &&
                    projection.cells->seq_has(row,found->sequence)) {
                const auto column=projection.columns[row];
                if (column.source!=i || ordinal>=found->full_columns.size() || column.token!=found->full_columns[ordinal++] ||
                        column.token>=seen.size() || seen[column.token]) return false;
                seen[column.token]=true;
            }
            if (ordinal!=found->full_columns.size()) return false;
        }
        synchronize();group_restore_failed=true;
        const bool empty=std::any_of(states.begin(),states.end(),[](const auto & state) {
            return !state.tokens;
        });
        // Native headers can encode zero history. Initialize the replacement before importing peers.
        if (empty) {
            if (block) memory->clear(true);
            else {
                if (kv_resources.recurrent) kv_resources.recurrent->clear(true);
                if (kv_resources.bounded) kv_resources.bounded->clear(true);
            }
        }
        const auto load=[](llama_memory_i * memory,llama_seq_id sequence,const std::vector<uint8_t> & bytes) {
            if (!memory) return;
            llama_io_read_host data(bytes.data(),bytes.size());
            try {
                memory->state_read(data,sequence,0);
                if (data.n_bytes()!=bytes.size()) throw std::runtime_error("native selected-state restore has trailing data");
                data.flush_tensor_reads();
            } catch (...) {data.discard();throw;}
        };
        for (auto sequence : group_selected) {
            const auto found=std::find_if(states.begin(),states.end(),[&](const auto & state) {return state.sequence==sequence;});
            if (!found->tokens && found->recurrent.empty() && found->bounded.empty()) continue;
            load(kv_resources.recurrent,sequence,found->recurrent);
            load(block ? memory.get() : kv_resources.bounded,sequence,found->bounded);
            if (block && memory->seq_pos_max(sequence)<found->draft_position) return false;
            if (block && found->draft_position>=0) {
                const auto & positions=kv_resources.bounded->get_cells(0).seq_pos_get(sequence);
                const auto point=positions.lower_bound({found->draft_position,0});
                if (point==positions.end() || point->first!=found->draft_position) return false;
            }
        }
        synchronize();group_restore_failed=false;return true;
    } catch (const std::exception & error) {
        LLAMA_LOG_ERROR("%s: native selected-state import failed: %s\n",__func__,error.what());return false;
    } catch (...) {return false;}
}

namespace {
using group_lease_ptr=std::unique_ptr<ggml_backend_memory_lease,decltype(&ggml_backend_memory_lease_free)>;
struct group_execution_resources {
    llama_context & context;
    std::shared_ptr<llama_kv_stream_slot_table> table;
    std::shared_ptr<llama_kv_stream_slot_projection> projection;
    ggml_backend_t backend=nullptr;
    llama_compute_arena_ptr parent,graph,host;
    std::vector<group_lease_ptr> leases;
    ggml_backend_memory_lease_t recurrent_state=nullptr;
    std::vector<std::shared_ptr<void>> prepared;
    std::vector<llama_kv_stream_group_reader_source> sources;
    std::unique_ptr<llama_kv_stream_group_reader> reader;
    std::vector<uint32_t> reader_sources;
    std::shared_ptr<llama_kv_stream_writer> writer;
    std::shared_ptr<llama_kv_stream_group_producer> producer;
    std::unique_ptr<llama_context_memory> workspace;
    std::unique_ptr<llama_memory_phase_owner> owner;
    bool decode=false,shared_roles=false;
    std::function<bool()> reader_admission={};

    explicit group_execution_resources(llama_context & context) : context(context) {}
    ~group_execution_resources() {
        context.synchronize();
        owner.reset();workspace.reset();reader.reset();producer.reset();writer.reset();sources.clear();prepared.clear();
    }
    ggml_backend_memory_lease_t acquire(uint64_t id) {
        auto * value=ggml_backend_memory_arena_acquire(parent.get(),id);
        if (!value) throw std::runtime_error("measured group region could not be leased");
        leases.emplace_back(value,ggml_backend_memory_lease_free);return value;
    }
    bool layer_ready(uint32_t layer) const {
        if (reader_admission) {
            const auto * full=context.get_kv_resources().full;
            const auto * source=full ? full->get_group_model() : nullptr;
            return source && source->shared_read_active() && reader_admission();
        }
        return context.group_submission_allowed() && producer && producer->layer_ready(layer);
    }
    void refresh_reader() {
        if (!reader) return;
        std::vector<llama_kv_stream_group_frontier> frontiers;
        frontiers.reserve(reader_sources.size());
        for (const auto index : reader_sources) {
            std::shared_ptr<llama_kv_stream_logical_cache> cache;
            if (index>=sources.size() || index>=projection->handles.size() ||
                    !table->get(projection->handles[index],projection->role,cache) || !cache->ready() ||
                    cache->host()!=sources[index].host || cache->content()!=sources[index].content) {
                reader.reset();return;
            }
            frontiers.push_back({cache->frontiers().reserved,cache->tokens()});
        }
        if (!reader->update_frontiers(frontiers)) reader.reset();
    }
    bool decode_transport(size_t rows) const {
        const auto & params=context.get_cparams();
        // Small MTP catch-up batches keep their native graph phase and use the decode transport.
        return (shared_roles ? params.kv_stream_decode : decode) ||
            (shared_roles && params.ctx_type==LLAMA_CONTEXT_TYPE_MTP && rows && rows<=8);
    }
    bool prepare_publication(ggml_backend_memory_lease_t encoding,uint32_t layer,
            const llama_kv_stream_group_producer::publication_rows & rows) {
        if (!reader || !decode_transport(rows.size()) ||
                !context.group_submission_allowed() || reader_sources.size()!=sources.size()) return false;
        std::vector<llama_kv_stream_group_frontier> frontiers;
        for (size_t index=0;index<sources.size();++index) {
            std::shared_ptr<llama_kv_stream_logical_cache> cache;
            if (reader_sources[index]!=index || !table->get(projection->handles[index],projection->role,cache) ||
                    cache->host()!=sources[index].host || cache->content()!=sources[index].content) {reader.reset();return false;}
            sources[index].tokens=cache->frontiers().reserved;sources[index].stable_tokens=cache->tokens();
            frontiers.push_back({sources[index].tokens,sources[index].stable_tokens});
        }
        if (!reader->update_frontiers(frontiers)) {reader.reset();return false;}
        return reader->can_publish(encoding,layer,rows);
    }
    bool complete_publication(uint32_t layer,const llama_kv_stream_group_producer::publication_rows & rows,bool committed) {
        if (!reader) return false;
        for (size_t index=0;index<sources.size();++index) {
            std::shared_ptr<llama_kv_stream_logical_cache> cache;
            if (!table->get(projection->handles[index],projection->role,cache) || cache->host()!=sources[index].host ||
                    cache->content()!=sources[index].content || cache->tokens()!=projection->sources[index].tokens ||
                    cache->content()->generation()!=projection->sources[index].generation) return false;
        }
        return reader->complete_publication(layer,rows,committed);
    }
    bool ensure_reader(ggml_backend_memory_lease_t reader_staging,bool prepare,uint32_t rows=1) {
        std::vector<llama_kv_stream_group_frontier> frontiers;
        frontiers.reserve(sources.size());
        std::vector<llama_kv_stream_group_reader_source> visible;
        std::vector<uint32_t> selected_sources;
        for (size_t i=0;i<sources.size();++i) {
            std::shared_ptr<llama_kv_stream_logical_cache> cache;
            if (!table->get(projection->handles[i],projection->role,cache)) return false;
            sources[i].content=cache->content();sources[i].tokens=cache->frontiers().reserved;
            sources[i].stable_tokens=cache->tokens();
            if (sources[i].tokens) {
                visible.push_back(sources[i]);selected_sources.push_back(uint32_t(i));
                frontiers.push_back({sources[i].tokens,sources[i].stable_tokens});
            }
        }
        if (visible.empty()) return false;
        if (reader && reader_sources!=selected_sources) {
            if (!reader->finish()) return false;
            reader.reset();
        }
        if (!reader) reader=llama_kv_stream_group_reader::create(backend,visible,reader_staging,[this](uint32_t reading) {
            if (!layer_ready(reading)) return false;
            for (size_t i=0;i<sources.size();++i) {
                std::shared_ptr<llama_kv_stream_logical_cache> cache;
                if (!table->get(projection->handles[i],projection->role,cache) || cache->host()!=sources[i].host ||
                        cache->content()!=sources[i].content || cache->content()->generation()!=projection->sources[i].generation ||
                        cache->tokens()!=projection->sources[i].tokens || cache->frontiers().reserved!=sources[i].tokens) return false;
            }
            return true;
        });
        if (!reader) {LLAMA_LOG_ERROR("%s: grouped reader creation failed for role %u\n",__func__,projection->role);return false;}
        reader_sources=std::move(selected_sources);
        if (!(prepare ? reader->prepare_frame(frontiers,{rows,decode_transport(rows),false}) : reader->update_frontiers(frontiers))) {LLAMA_LOG_ERROR("%s: grouped reader frontier update failed\n",__func__);return false;}
        return true;
    }
    bool attention(uint32_t layer,ggml_tensor * op,ggml_backend_memory_lease_t staging,
            ggml_backend_memory_lease_t workspace) {
        if (!op || !op->src[0] || op->src[0]->ne[1]<=0 || !op->src[1] || op->src[1]->ne[1]<=0 ||
                size_t(op->src[1]->ne[1])>projection->columns.size() ||
                projection->cells->used_max_p1()>uint32_t(op->src[1]->ne[1]) ||
                !layer_ready(layer)) return false;
        if (!ensure_reader(staging,false)) return false;
        std::vector<uint32_t> remap(sources.size(),UINT32_MAX);
        for (size_t i=0;i<reader_sources.size();++i) remap[reader_sources[i]]=uint32_t(i);
        std::vector<ggml_kv_stream_gather_row> columns(projection->columns.begin(),
            projection->columns.begin()+op->src[1]->ne[1]);
        // Empty histories have no native columns. Keep column order and remap only reader ordinals.
        for (auto & column : columns) if (column.source!=UINT32_MAX) {
            if (column.source>=remap.size() || remap[column.source]==UINT32_MAX) return false;
            column.source=remap[column.source];
        }
        const bool decode_mode=reader_admission || decode_transport(size_t(op->src[0]->ne[1]));
        const auto okay=decode_mode && !reader_admission ? reader->enqueue_attention(layer,columns,op,workspace,true) :
            reader->attention(layer,columns,op,workspace,decode_mode);
        if (okay && layer+1==sources[reader_sources.front()].host->config().layers && !context.group_restore_recurrent_after_attention()) {
            LLAMA_LOG_ERROR("%s: grouped recurrent restore after attention failed\n",__func__);return false;
        }
        if (!okay) LLAMA_LOG_ERROR("%s: grouped role %u layer %u attention failed\n",__func__,projection->role,layer);
        return okay;
    }
};
}

bool llama_context::initialize_group_execution(std::shared_ptr<llama_kv_stream_slot_table> table,
        std::shared_ptr<llama_kv_stream_slot_projection> projection,const llama_batch & representative,
        llama_memory_work_phase phase,uint32_t profile,uint64_t selection_revision,ggml_backend_buffer_t existing_parent,
        const std::vector<llama_context_group_sequence_state> & states,llama_context_memory_create_result * result,
        llama_context * auxiliary,std::shared_ptr<llama_kv_stream_slot_projection> auxiliary_projection,
        const std::vector<llama_context_group_sequence_state> & auxiliary_states) {
    if (result) *result={};
    const bool readonly=auxiliary && model.arch==LLM_ARCH_GEMMA4 && auxiliary->model.arch==LLM_ARCH_GEMMA4_ASSISTANT &&
        auxiliary->group_target==this && auxiliary->cparams.ctx_other==this && !cparams.kv_stream_auxiliary_layers;
    const bool block=auxiliary && auxiliary->model.arch==LLM_ARCH_DFLASH && auxiliary->group_target==this &&
        auxiliary->cparams.ctx_other==this && !cparams.kv_stream_auxiliary_layers;
    const bool owned=auxiliary && !readonly && !block;
    if (!group_initializing || bool(cparams.kv_stream_auxiliary_layers)!=owned || (readonly && !auxiliary_states.empty()) ||
            group_restore_failed || group_execution || compute_memory || !table || !projection ||
            !profile || !selection_revision || !table->projection_current(*projection) || !kv_resources.full ||
            !kv_resources.full->full_storage_pending() || !cparams.shared_device_memory_bytes) return false;
    if (auxiliary && (!auxiliary_projection || auxiliary_projection->role!=(readonly || block ? 0u : 1u) || !table->projection_current(*auxiliary_projection) ||
            !auxiliary->group_initializing || auxiliary->group_execution || auxiliary->compute_memory ||
            !auxiliary->kv_resources.full || (owned && !auxiliary->kv_resources.full->full_storage_pending()) ||
            auxiliary_projection->handles.size()!=projection->handles.size() ||
            auxiliary_projection->selected.context_tokens!=projection->selected.context_tokens ||
            auxiliary_projection->selected.history_tokens!=projection->selected.history_tokens)) return false;
    if (auxiliary) for (size_t i=0;i<projection->handles.size();++i) {
        const auto a=projection->selected.slots[i],b=auxiliary_projection->selected.slots[i];
        if (projection->handles[i].slot!=auxiliary_projection->handles[i].slot ||
                projection->handles[i].generation!=auxiliary_projection->handles[i].generation ||
                a.id!=b.id || a.history_tokens!=b.history_tokens || a.growth_tokens!=b.growth_tokens || a.offset_tokens!=b.offset_tokens) return false;
    }
    try {
        auto resources=std::make_shared<group_execution_resources>(*this);
        auto child=auxiliary ? std::make_shared<group_execution_resources>(*auxiliary) : nullptr;
        resources->table=table;resources->projection=projection;resources->decode=phase==llama_memory_work_phase::target_verify;
        resources->shared_roles=bool(auxiliary);
        for (auto * candidate : backend_ptrs) if (ggml_backend_get_device(candidate)==model.devices.front().dev) resources->backend=candidate;
        if (!resources->backend) return false;
        std::vector<llama_memory_phase_cache_identity> identities;
        std::vector<llama_memory_phase_cache_identity> child_identities;
        bool history=false;
        for (const auto & source : projection->sources) {identities.push_back({source.id,source.generation});history=history || source.tokens!=0;}
        if (history && states.empty()) return false;
        if (child) {
            child->table=table;child->projection=auxiliary_projection;child->shared_roles=true;
            for (auto * candidate : auxiliary->backend_ptrs)
                if (ggml_backend_get_device(candidate)==model.devices.front().dev) child->backend=candidate;
            if (!child->backend) return false;
            for (const auto & source : auxiliary_projection->sources) if (owned) {
                if (source.tokens && auxiliary_states.empty()) return false;
                child_identities.push_back({source.id,source.generation});
            }
        }
        llama_context_group_plan quote;llama_memory_phase_cache_identity bounded_identity;
        if (!llama_context_group_bounded_identity(identities,child_identities,bounded_identity)) return false;
        const auto admission=quote_target_group(representative,identities,bounded_identity,phase,
            profile,cparams.shared_device_memory_bytes,quote,auxiliary,child_identities);
        if (result) *result=admission;
        if (admission.status!=llama_context_memory_create_status::success) return false;
        if (result) result->status=llama_context_memory_create_status::invalid_configuration;
        if (existing_parent && (ggml_backend_buffer_get_type(existing_parent)!=quote.target.buffer_type ||
                ggml_backend_buffer_get_size(existing_parent)!=cparams.shared_device_memory_bytes)) return false;
        const auto layers=kv_resources.full->get_layer_ids();const auto first=layers.front();
        const ggml_kv_stream_shape shape{kv_resources.full->type_k(),kv_resources.full->type_v(),model.hparams.n_embd_head_k(first),
            model.hparams.n_embd_head_v(first),model.hparams.n_head_kv(first),256,128};
        auto * registry=ggml_backend_dev_backend_reg(model.devices.front().dev);
        const auto get_partial=reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(registry,"ggml_backend_kv_stream_partial_ops"));
        const auto get_copy=reinterpret_cast<ggml_kv_stream_copy_ops_get>(ggml_backend_reg_get_proc_address(registry,"ggml_backend_kv_stream_copy_ops"));
        const auto * partial=get_partial ? get_partial() : nullptr;const auto * copies=get_copy ? get_copy() : nullptr;
        if (!partial || !copies || !copies->prepare || !copies->free_prepared) return false;
        llama_kv_stream_policy_config policy;
        policy.shape=shape;policy.capabilities=partial->capabilities(resources->backend,shape.type_k,shape.type_v);
        policy.layers=uint32_t(layers.size())+(owned ? uint32_t(auxiliary->kv_resources.full->get_layer_ids().size()) : 0);
        policy.pool_bytes=quote.target.pool_minimum_bytes/projection->handles.size();
        llama_kv_stream_policy_state minimum;
        if (llama_kv_stream_policy_initialize(policy,minimum).status!=llama_kv_stream_policy_status::success) return false;
        ggml_kv_stream_layout page;
        if (ggml_kv_stream_layout_make(shape,256,page).status!=ggml_kv_stream_status::success ||
                minimum.budget.pages>policy.pool_bytes/page.bytes) return false;
        const size_t conversion=policy.pool_bytes-size_t(minimum.budget.pages)*page.bytes;
        const auto alignment=quote.target.measurement.alignment;
        const size_t graph_bytes=*std::max_element(quote.target.measurement.graph_bytes.begin(),quote.target.measurement.graph_bytes.end());
        std::vector<ggml_backend_memory_region> regions;
        size_t cursor=0;
        const auto reserve=[&](uint64_t id,size_t bytes) {
            if (!bytes || cursor>SIZE_MAX-(alignment-1)) return false;
            cursor=(cursor+alignment-1)/alignment*alignment;
            if (cursor>cparams.shared_device_memory_bytes || bytes>cparams.shared_device_memory_bytes-cursor) return false;
            regions.push_back({id,cursor,bytes,alignment,GGML_BACKEND_MEMORY_REGION_PERSISTENT});cursor+=bytes;return true;
        };
        if ((quote.current_state_bytes && !reserve(101,quote.current_state_bytes)) ||
                (quote.bounded.allocation.bytes && !reserve(102,quote.bounded.allocation.bytes)) || !reserve(201,graph_bytes) ||
                !reserve(202,quote.target.writer_bytes) || !reserve(203,quote.target.attention_bytes) || !reserve(204,page.bytes) ||
                (quote.target.publication_bytes && !reserve(205,quote.target.publication_bytes))) return false;
        if (cursor>SIZE_MAX-(alignment-1)) return false;
        const size_t pools_first=(cursor+alignment-1)/alignment*alignment,count=projection->handles.size();
        if (page.bytes%alignment || conversion%alignment || pools_first>cparams.shared_device_memory_bytes ||
                conversion>cparams.shared_device_memory_bytes-pools_first) return false;
        const size_t pages=(cparams.shared_device_memory_bytes-pools_first-conversion)/page.bytes;
        if (pages>UINT32_MAX || pages<=policy.layers || !reserve(301,pages*page.bytes+conversion)) return false;
        resources->prepared.emplace_back(copies->prepare(resources->backend,pages,true),copies->free_prepared);
        if (!resources->prepared.back()) return false;
        if (owned) {
            child->prepared.emplace_back(copies->prepare(child->backend,pages,true),copies->free_prepared);
            if (!child->prepared.back()) return false;
        }
        resources->parent.reset(existing_parent ? ggml_backend_memory_arena_new_from_buffer(existing_parent) :
            ggml_backend_memory_arena_new(quote.target.buffer_type,cparams.shared_device_memory_bytes));
        if (!resources->parent || !ggml_backend_memory_arena_begin(resources->parent.get(),0)) return false;
        for (const auto & region : regions) if (!ggml_backend_memory_arena_reserve_at(resources->parent.get(),region.id,
                region.offset,region.size,region.alignment,region.flags,nullptr)) return false;
        if (!ggml_backend_memory_arena_commit(resources->parent.get())) return false;
        if (child) child->parent.reset(ggml_backend_memory_arena_retain(resources->parent.get()));
        if (kv_resources.recurrent) {
            resources->recurrent_state=resources->acquire(101);
            if (!kv_resources.recurrent->bind_current_storage(resources->recurrent_state) ||
                    (quote.target.publication_bytes && !kv_resources.recurrent->bind_spill_stage(resources->acquire(205)))) return false;
        }
        if (kv_resources.bounded && !kv_resources.bounded->bind_bounded_storage(resources->acquire(102))) return false;
        if (block && (!auxiliary->kv_resources.bounded ||
                !auxiliary->kv_resources.bounded->bind_bounded_storage(child->acquire(102)))) return false;
        auto graph=resources->acquire(201),encoding=resources->acquire(202),attention=resources->acquire(203),staging=resources->acquire(204);
        if (block) child->leases.emplace_back(ggml_backend_memory_lease_retain(graph),ggml_backend_memory_lease_free);
        std::vector<std::vector<std::shared_ptr<llama_kv_stream_content>>> contents;
        for (size_t i=0;i<count;++i) {
            std::shared_ptr<llama_kv_stream_logical_cache> cache;
            if (!table->get(projection->handles[i],projection->role,cache)) return false;
            resources->sources.push_back({cache->host(),resources->acquire(301),pages,0,resources->prepared.front()});
            if (owned) {
                std::shared_ptr<llama_kv_stream_logical_cache> other;
                if (!table->get(auxiliary_projection->handles[i],1,other)) return false;
                child->sources.push_back({other->host(),child->acquire(301),pages,0,child->prepared.front()});
                resources->sources.back().owning_caches={cache->host(),other->host()};
                child->sources.back().owning_caches=resources->sources.back().owning_caches;
            }
            std::vector<std::shared_ptr<llama_kv_stream_content>> owners{cache->content()};
            if (owned) {
                std::shared_ptr<llama_kv_stream_logical_cache> other;
                if (!table->get(auxiliary_projection->handles[i],1,other)) return false;
                owners.push_back(other->content());
            }
            contents.push_back(std::move(owners));
        }
        // Prefer full residency for the prediction cache reused between target passes.
        auto buffering=llama_kv_stream_group_buffering::create_aggregate(contents,
            [target=std::weak_ptr<group_execution_resources>(resources),draft=std::weak_ptr<group_execution_resources>(child)] {
                for (const auto & weak:{target,draft}) if (auto state=weak.lock())
                    if (state->reader && !state->reader->finish()) return false;
                return true;
            },owned ? std::vector<uint32_t>{1} : std::vector<uint32_t>{});
        if (!buffering) return false;
        for (auto & source:resources->sources) source.buffering=buffering;
        if (owned) for (auto & source:child->sources) source.buffering=buffering;
        const uint32_t verify_rows=group_verify_rows(cparams);
        auto source_model=llama_kv_stream_group_model::create({resources->backend,table,shape,uint32_t(layers.size()),model.hparams.n_head(first),
            cparams.n_ubatch,verify_rows,uint32_t(count)});
        if (!source_model || !kv_resources.full->bind_group_model(source_model,*projection)) return false;
        const auto restore_selected=[](llama_context & context,const std::vector<llama_context_group_sequence_state> & saved,
                const llama_kv_stream_slot_projection & selected) {
            if (!saved.empty()) return context.restore_group_sequences(saved,selected);
            // Fresh arena leases may contain NaNs from a previous layout.
            if (context.model.arch==LLM_ARCH_DFLASH) context.memory->clear(true);
            else {
                if (context.kv_resources.recurrent) context.kv_resources.recurrent->clear(true);
                if (context.kv_resources.bounded) context.kv_resources.bounded->clear(true);
            }
            context.synchronize();return true;
        };
        if (!restore_selected(*this,states,*projection)) {
            LLAMA_LOG_ERROR("%s: selected target state restore rejected\n",__func__);return false;
        }
        resources->writer=llama_kv_stream_writer::create(resources->backend,ggml_backend_memory_lease_buffer(encoding),
            quote.target.writer_bytes,shape,cparams.n_ubatch);
        resources->producer=llama_kv_stream_group_producer::create(table,resources->writer,cparams.n_ubatch,verify_rows,uint32_t(count));
        const auto bind_publication=[](const std::shared_ptr<group_execution_resources> & state,ggml_backend_memory_lease_t encoding) {
            const auto weak=std::weak_ptr<group_execution_resources>(state);
            return state->producer && state->producer->bind_consumer(state->backend) && state->producer->bind_publication(
                [weak,encoding](uint32_t layer,const auto & rows) {
                    auto retained=weak.lock();return retained && retained->prepare_publication(encoding,layer,rows);
                },[weak,encoding](uint32_t layer,bool value,const ggml_tensor * source,size_t first,size_t count,const auto & rows) {
                    auto retained=weak.lock();return retained && retained->reader && retained->context.group_submission_allowed() &&
                        retained->reader->publish_encoded(encoding,layer,value,source,first,count,rows);
                },[weak](uint32_t layer,const auto & rows,bool committed) {
                    auto retained=weak.lock();return retained && retained->complete_publication(layer,rows,committed);
                });
        };
        if (!bind_publication(resources,encoding)) return false;
        const auto allowed=[this] {return group_submission_allowed();};
        if (!kv_resources.full->get_group_model()->bind(resources->producer,[resources,attention,staging](uint32_t layer,ggml_tensor * op) {
                return resources->attention(layer,op,staging,attention);
            },allowed)) return false;
        if (block && !restore_selected(*auxiliary,auxiliary_states,*auxiliary_projection)) {
            LLAMA_LOG_ERROR("%s: bounded draft state restore rejected\n",__func__);return false;
        }
        if (child && !states.empty() && !kv_resources.full->get_group_model()->bind_projection(projection)) return false;
        if (owned) {
            const auto child_layers=auxiliary->kv_resources.full->get_layer_ids();
            auto child_encoding=child->acquire(202),child_attention=child->acquire(203),child_staging=child->acquire(204);
            child->leases.emplace_back(ggml_backend_memory_lease_retain(graph),ggml_backend_memory_lease_free);
            auto child_model=llama_kv_stream_group_model::create({child->backend,table,shape,uint32_t(child_layers.size()),
                model.hparams.n_head(child_layers.front()),auxiliary->cparams.n_ubatch,1,uint32_t(count)});
            if (!child_model || !auxiliary->kv_resources.full->bind_group_model(child_model,*auxiliary_projection)) return false;
            if (!restore_selected(*auxiliary,auxiliary_states,*auxiliary_projection)) return false;
            child->writer=llama_kv_stream_writer::create(child->backend,ggml_backend_memory_lease_buffer(child_encoding),
                quote.target.writer_bytes,shape,auxiliary->cparams.n_ubatch);
            child->producer=llama_kv_stream_group_producer::create(table,child->writer,auxiliary->cparams.n_ubatch,1,uint32_t(count));
            if (!bind_publication(child,child_encoding)) return false;
            if (!auxiliary->kv_resources.full->get_group_model()->bind(child->producer,[child,child_attention,child_staging](uint32_t layer,ggml_tensor * op) {
                    return child->attention(layer,op,child_staging,child_attention);
                },[auxiliary] {return auxiliary->group_submission_allowed();})) return false;
            if (!auxiliary_states.empty() && !auxiliary->kv_resources.full->get_group_model()->bind_projection(auxiliary_projection)) return false;
        } else if (readonly) {
            ggml_backend_sched_reset(auxiliary->sched.get());
            for (auto & previous : auxiliary->gf_res_prev) if (previous) previous->reset();
            auxiliary->gf_res_reserve->reset();auxiliary->gf_res_prev_active=nullptr;
            llama_memory_params memory_params{};
            memory_params.type_k=ggml_type(shape.type_k);memory_params.type_v=ggml_type(shape.type_v);
            memory_params.swa_full=false;memory_params.ctx_type=LLAMA_CONTEXT_TYPE_MTP;memory_params.mem_other=memory.get();
            std::unique_ptr<llama_memory_i> replacement(auxiliary->model.create_memory(memory_params,auxiliary->cparams));
            const auto views=replacement ? llama_memory_kv_resources_get(replacement.get()) : llama_memory_kv_resources{};
            auto bounded=kv_resources.bounded->retain_bounded_storage(quote.bounded.cache.identity.id,
                quote.bounded.cache.identity.generation,cparams.n_ctx_seq,cparams.n_ubatch);
            const auto pin=bounded ? bounded->read(quote.bounded.cache.identity) : llama_memory_bounded_kv_read{};
            if (!views.full || !views.bounded || !pin || !views.full->protect_shared_views() ||
                    !views.bounded->attach_shared_bounded_reader(pin)) return false;
            auxiliary->memory=std::move(replacement);auxiliary->kv_resources=views;
            child->leases.emplace_back(ggml_backend_memory_lease_retain(graph),ggml_backend_memory_lease_free);
        }
        resources->graph.reset(ggml_backend_memory_arena_new_from_buffer(ggml_backend_memory_lease_buffer(graph)));
        if (!resources->graph) return false;
        if (child) child->graph.reset(ggml_backend_memory_arena_retain(resources->graph.get()));
        size_t gpu=backend_ptrs.size();
        for (size_t i=0;i<backend_ptrs.size();++i) if (backend_ptrs[i]==resources->backend) gpu=i;
        if (gpu==backend_ptrs.size()) return false;
        std::vector<llama_memory_workspace_group> groups,child_groups;
        std::vector<llama_memory_arena_budget> budgets;
        std::vector<llama_memory_transition_arena> arenas;
        llama_memory_execution_plan plan;plan.stages.push_back({1,{},{}});
        const std::vector<llama_memory_stage_id> stages=child ? std::vector<llama_memory_stage_id>{1,2,3,4} :
            std::vector<llama_memory_stage_id>{1};
        if (child) for (uint64_t stage=2;stage<=4;++stage) plan.stages.push_back({stage,{},{}});
        if (child && auxiliary->backend_ptrs.size()!=backend_ptrs.size()) return false;
        for (size_t i=0;i<backend_ptrs.size();++i) {
            const bool device=i==gpu;const size_t bytes=device ? graph_bytes : quote.target.host_compute_bytes;
            if (!bytes) continue;
            auto * type=ggml_backend_sched_get_buffer_type(sched.get(),backend_ptrs[i]);
            const auto allocation=device ? LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL : LLAMA_MEMORY_ALLOCATION_HOST_PINNED;
            const auto aligned=ggml_backend_buft_get_alignment(type);
            std::vector<llama_memory_workspace_stage> target_stages,auxiliary_stages;
            for (auto stage : stages) {
                target_stages.push_back({stage,stage<=2 ? bytes : 0});
                auxiliary_stages.push_back({stage,stage>=3 ? bytes : 0});
            }
            groups.push_back({{type,bytes,aligned,i},{uint64_t(701+i),uint64_t(i+1),allocation,llama_memory_content::discardable},target_stages});
            if (child) {
                if (ggml_backend_sched_get_buffer_type(auxiliary->sched.get(),auxiliary->backend_ptrs[i])!=type ||
                        ggml_backend_get_device(auxiliary->backend_ptrs[i])!=ggml_backend_get_device(backend_ptrs[i])) return false;
                child_groups.push_back({{type,bytes,aligned,i},{uint64_t(801+i),uint64_t(i+1),allocation,llama_memory_content::discardable},auxiliary_stages});
            }
            plan.domains.push_back({uint64_t(i+1),allocation,LLAMA_MEMORY_CAPABILITY_BUFFER_VIEWS});
            budgets.push_back({uint64_t(i+1),allocation,bytes,aligned});
            if (!device) {
                if (resources->host) return false;
                resources->host.reset(ggml_backend_memory_arena_new(type,bytes));
                if (!resources->host) return false;
                if (child) child->host.reset(ggml_backend_memory_arena_retain(resources->host.get()));
            }
            arenas.push_back({uint64_t(i+1),allocation,device ? resources->graph.get() : resources->host.get()});
        }
        resources->workspace=llama_context_memory::create_external_workspace(sched.get(),backend_ptrs,groups,allowed);
        if (!resources->workspace || !resources->workspace->external_workspace()->register_resources(plan,stages)) return false;
        if (child) {
            child->workspace=llama_context_memory::create_external_workspace(auxiliary->sched.get(),auxiliary->backend_ptrs,child_groups,
                [auxiliary] {return auxiliary->group_submission_allowed();});
            if (!child->workspace || !child->workspace->external_workspace()->register_resources(plan,stages)) return false;
        }
        llama_memory_transition_target target;target.plan=plan;target.stage=1;target.budgets=budgets;
        uint64_t maximum_rows=0;
        for (const auto & member : quote.target.measurement.workload.members) maximum_rows+=member.input_rows;
        if (!maximum_rows || maximum_rows>UINT32_MAX) return false;
        if (child) {
            const auto make_target=[&](llama_memory_work_phase selected,uint64_t stage,uint32_t maximum) {
                auto transition=target;transition.stage=stage;
                return llama_memory_phase_target{selected,std::move(transition),maximum,profile};
            };
            identities.insert(identities.end(),child_identities.begin(),child_identities.end());
            std::vector<llama_memory_phase_target> targets;
            for (const auto & candidate : {
                    make_target(llama_memory_work_phase::target_prefill,1,uint32_t(representative.n_tokens)),
                    make_target(llama_memory_work_phase::target_verify,2,uint32_t(count)*verify_rows),
                    make_target(block ? llama_memory_work_phase::draft_feature_injection : llama_memory_work_phase::mtp_catchup,3,auxiliary->cparams.n_batch),
                    make_target(block ? llama_memory_work_phase::draft_block_prediction : llama_memory_work_phase::mtp_draft,4,
                        block ? auxiliary->cparams.n_ubatch : uint32_t(count))}) {
                const bool measured=std::any_of(quote.catalog.begin(),quote.catalog.end(),[&](const auto & workload) {
                    return std::any_of(workload.measurement.workload.members.begin(),workload.measurement.workload.members.end(),
                        [&](const auto & member) {return member.phase==candidate.phase;});
                });
                if (measured) targets.push_back(candidate);
            }
            std::shared_ptr<llama_memory_phase_owner> owner=llama_memory_phase_owner::create(std::move(targets),arenas,
                {resources->workspace->external_workspace(),child->workspace->external_workspace()},identities);
            if (!owner) return false;
            const auto child_phase=block ? llama_memory_work_phase::draft_feature_injection :
                readonly ? llama_memory_work_phase::mtp_draft : llama_memory_work_phase::mtp_catchup;
            const auto child_entered=owner->enter(child_phase,1,identities,profile,selection_revision);
            if (child_entered.status!=llama_memory_phase_owner_status::changed) return false;
            if (!auxiliary->attach_group_execution_shared(child->workspace,auxiliary_projection,owner,
                    child_phase,profile,selection_revision)) return false;
            auxiliary->group_execution->resources=child;
            if (block && !auxiliary_states.empty()) {
                for (size_t i=0;i<auxiliary_projection->handles.size();++i) {
                    const auto sequence=llama_seq_id(auxiliary_projection->handles[i].slot);
                    const auto found=std::find_if(auxiliary_states.begin(),auxiliary_states.end(),
                        [&](const auto & state) {return state.sequence==sequence;});
                    if (found==auxiliary_states.end()) return false;
                    if (found->tokens) auxiliary->group_execution->draft_sources[i]={found->cache_id,found->cache_generation,found->tokens};
                    auxiliary->group_execution->draft_positions[i]=found->draft_position;
                }
            }
            const auto entered=owner->enter(phase,uint32_t(representative.n_tokens),identities,profile,selection_revision);
            if (entered.status!=llama_memory_phase_owner_status::changed ||
                    !attach_group_execution_shared(resources->workspace,projection,owner,phase,profile,selection_revision)) {
                auxiliary->group_execution->failed=true;
                auxiliary->group_execution->owner.reset();owner.reset();return false;
            }
            group_execution->peer=auxiliary->group_execution.get();auxiliary->group_execution->peer=group_execution.get();
            if (readonly) {
                std::vector<uint32_t> read_layers;
                for (auto layer : auxiliary->kv_resources.full->get_layer_ids()) {
                    const auto * key=auxiliary->kv_resources.full->get_k_storage(layer);
                    while (key->view_src) key=key->view_src;
                    size_t ordinal=0;
                    for (;ordinal<layers.size();++ordinal) {
                        const auto * native=kv_resources.full->get_k_storage(layers[ordinal]);
                        while (native->view_src) native=native->view_src;
                        if (native==key) break;
                    }
                    if (ordinal==layers.size()) return false;
                    read_layers.push_back(uint32_t(ordinal));
                }
                if (read_layers.empty()) return false;
                auxiliary->group_execution->read_prepare=[this,auxiliary,weak=std::weak_ptr<group_execution_resources>(resources),
                        read_layers,identity=quote.bounded.cache.identity,attention,staging,consumer=child->backend](const llama_ubatch & batch) {
                    auto retained=weak.lock();
                    if (!retained || retained->reader_admission || !auxiliary->group_execution || !group_execution ||
                            group_execution->failed || group_execution->admission ||
                            auxiliary->group_execution->peer!=group_execution.get() || !auxiliary->group_submission_allowed()) return false;
                    auto bounded=kv_resources.bounded->retain_bounded_storage(identity.id,identity.generation,cparams.n_ctx_seq,cparams.n_ubatch);
                    const auto pin=bounded ? bounded->read(identity) : llama_memory_bounded_kv_read{};
                    if (!pin || !auxiliary->kv_resources.bounded->arm_shared_bounded_reader(pin)) return false;
                    const auto ticket=auxiliary->group_execution->admission;
                    retained->reader_admission=[auxiliary,ticket] {
                        return auxiliary->group_execution && auxiliary->group_execution->admission==ticket && auxiliary->group_submission_allowed();
                    };
                    if (!kv_resources.full->get_group_model()->begin_shared_read(consumer,batch,read_layers,
                            [weak,attention,staging](uint32_t layer,ggml_tensor * op) {
                                auto source=weak.lock();return source && source->attention(layer,op,staging,attention);
                            },retained->reader_admission)) {
                        retained->reader_admission={};auxiliary->kv_resources.bounded->release_shared_bounded_reader();return false;
                    }
                    return true;
                };
                auxiliary->group_execution->read_release=[this,auxiliary,weak=std::weak_ptr<group_execution_resources>(resources)](bool success) {
                    if (!auxiliary->group_execution || !auxiliary->group_execution->peer ||
                            &auxiliary->group_execution->peer->context!=this) return false;
                    auto retained=weak.lock();if (!retained) return false;
                    auto * source=kv_resources.full->get_group_model();
                    if (source->shared_read_active() && !(success ? source->finish_shared_read() : source->abort())) return false;
                    retained->reader_admission={};auxiliary->kv_resources.bounded->release_shared_bounded_reader();return true;
                };
            }
            const auto child_mutate=[auxiliary,weak=std::weak_ptr<group_execution_resources>(child)](const std::function<bool()> & mutation) {
                auto retained=weak.lock();
                return retained && auxiliary->group_execution && auxiliary->group_execution->mutate([&] {
                    if (retained->reader && !retained->reader->finish()) return false;
                    if (!mutation()) {retained->reader.reset();return false;}
                    retained->refresh_reader();return true;
                });
            };
            if (owned) auxiliary->kv_resources.full->get_group_model()->bind_mutation(child_mutate);
            if (block) {
                auxiliary->group_execution->draft_suffixes.resize(auxiliary_projection->handles.size());
                auto * iswa=dynamic_cast<llama_kv_cache_iswa *>(auxiliary->memory.get());
                if (!iswa) return false;
                iswa->bind_group_remove([auxiliary,child_mutate](llama_seq_id sequence,llama_pos first,llama_pos end,
                        const std::function<bool()> & remove) {
                    return auxiliary->group_execution && auxiliary->group_execution->remove_draft(sequence,first,end,
                        [&] {return child_mutate(remove);});
                });
            }
        } else {
            resources->owner=llama_memory_phase_owner::create({{phase,target,uint32_t(maximum_rows),profile}},arenas,
                {resources->workspace->external_workspace()},identities);
            if (!resources->owner) return false;
            const auto entered=resources->owner->enter(phase,uint32_t(representative.n_tokens),identities,profile,selection_revision);
            if (entered.status!=llama_memory_phase_owner_status::changed) return false;
            if (!attach_group_execution(resources->workspace,projection,resources->owner,phase,profile,selection_revision)) return false;
        }
        const auto mutate=[this,weak=std::weak_ptr<group_execution_resources>(resources)](const std::function<bool()> & mutation) {
            auto retained=weak.lock();
            return retained && group_execution && group_execution->mutate([&] {
                if (retained->reader && !retained->reader->finish()) return false;
                auto * child=group_execution->shared_block ? group_execution->peer : nullptr;
                std::vector<llama_kv_stream_logical_checkpoint> before;
                if (child) for (const auto & handle : group_execution->projection->handles) {
                    std::shared_ptr<llama_kv_stream_logical_cache> cache;
                    if (!group_execution->table->get(handle,0,cache) || !cache->ready()) return false;
                    before.push_back(cache->checkpoint());
                }
                if (!mutation()) {retained->reader.reset();return false;}
                if (child) for (size_t i=0;i<before.size();++i) {
                    std::shared_ptr<llama_kv_stream_logical_cache> cache;
                    if (!group_execution->table->get(group_execution->projection->handles[i],0,cache) || !cache->ready()) return false;
                    const auto after=cache->checkpoint(),saved=child->draft_sources[i];
                    if (saved.id==before[i].id && saved.generation==before[i].generation &&
                            saved.tokens==before[i].tokens && after.id==before[i].id &&
                            before[i].generation!=UINT64_MAX && after.generation==before[i].generation+1 &&
                            after.tokens<before[i].tokens) child->draft_suffixes[i]=after;
                }
                retained->refresh_reader();
                return true;
            });
        };
        kv_resources.full->get_group_model()->bind_mutation(mutate);
        if (auto * hybrid=dynamic_cast<llama_memory_hybrid *>(memory.get())) hybrid->bind_group_mutation(mutate);
        const auto bind_readers=[](llama_context & context,const std::shared_ptr<group_execution_resources> & state,
                ggml_backend_memory_lease_t staging) {
            const auto weak=std::weak_ptr<group_execution_resources>(state);
            context.group_execution->reader_prepare=[weak,staging](uint32_t rows) {
                auto retained=weak.lock();return retained && retained->ensure_reader(staging,true,rows);
            };
            context.group_execution->reader_finish=[weak] {
                auto retained=weak.lock();return retained && (!retained->reader || retained->reader->finish());
            };
        };
        bind_readers(*this,resources,staging);
        if (owned) {
            auto child_staging=ggml_backend_memory_arena_acquire(child->parent.get(),204);
            if (!child_staging) return false;
            child->leases.emplace_back(child_staging,ggml_backend_memory_lease_free);
            bind_readers(*auxiliary,child,child_staging);
        }
        group_execution->resources=std::move(resources);
        if (result) result->status=llama_context_memory_create_status::success;
        return true;
    } catch (const std::bad_alloc &) {
        if (result) result->status=llama_context_memory_create_status::allocation_failed;
        return false;
    } catch (const std::exception & error) {
        LLAMA_LOG_ERROR("%s: group initialization failed: %s\n",__func__,error.what());return false;
    }
}

ggml_backend_buffer_t llama_context::group_parent() const noexcept {
    const auto storage=group_execution ? static_cast<const group_execution_resources *>(group_execution->resources.get()) : nullptr;
    return storage && storage->parent ? ggml_backend_memory_arena_parent(storage->parent.get()) : nullptr;
}

bool llama_context::group_recurrent_restore_isolated() const noexcept {
    const auto * target=group_target;
    if (!target || !group_execution || !target->group_execution || !compute_memory || !target->compute_memory ||
            !group_execution->shared_roles || !target->group_execution->shared_roles ||
            group_execution->failed || target->group_execution->failed || target->group_execution->admission ||
            group_execution->peer!=target->group_execution.get() || target->group_execution->peer!=group_execution.get() ||
            group_execution->owner!=target->group_execution->owner || group_selected!=target->group_selected ||
            group_execution->table!=target->group_execution->table ||
            !target->group_execution->table->projection_current(*target->group_execution->projection) ||
            group_execution->projection->handles.size()!=target->group_execution->projection->handles.size()) return false;
    // The draft append can be active; request epochs and the completed target must still be current.
    for (size_t i=0;i<group_execution->projection->handles.size();++i) {
        const auto a=group_execution->projection->handles[i],b=target->group_execution->projection->handles[i];
        std::shared_ptr<llama_kv_stream_logical_cache> source,destination;
        if (a.slot!=b.slot || a.generation!=b.generation ||
                !group_execution->table->get(a,group_execution->projection->role,source) ||
                !target->group_execution->table->get(b,target->group_execution->projection->role,destination) ||
                !destination->ready()) return false;
    }
    const auto * recurrent=target->kv_resources.recurrent;
    const auto * state=static_cast<const group_execution_resources *>(target->group_execution->resources.get());
    const auto * draft=static_cast<const group_execution_resources *>(group_execution->resources.get());
    if (!recurrent || recurrent->current_storage_pending() || !state || !draft || !state->recurrent_state ||
            !state->parent || !draft->parent ||
            ggml_backend_memory_arena_parent(state->parent.get())!=ggml_backend_memory_arena_parent(draft->parent.get())) return false;
    auto * region=ggml_backend_memory_lease_buffer(state->recurrent_state);
    if (!region) return false;
    const auto first=reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(region));
    const auto bytes=ggml_backend_buffer_get_size(region);
    if (!first || !bytes || bytes>UINTPTR_MAX-first) return false;
    bool seen=false;
    for (size_t layer=0;layer<recurrent->r_l.size();++layer) if (recurrent->r_l[layer]) {
        for (const auto * tensor : {recurrent->r_l[layer],recurrent->s_l[layer]}) {
            if (!tensor || !tensor->buffer) return false;
            const auto data=reinterpret_cast<uintptr_t>(tensor->data);
            if (data<first || data>first+bytes || ggml_nbytes(tensor)>first+bytes-data) return false;
            seen=true;
        }
    }
    const auto disjoint=[&](ggml_backend_memory_lease_t lease) {
        auto * buffer=lease ? ggml_backend_memory_lease_buffer(lease) : nullptr;
        if (!buffer) return false;
        const auto start=reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(buffer));
        const auto size=ggml_backend_buffer_get_size(buffer);
        return !size || (start && size<=UINTPTR_MAX-start && (start>=first+bytes || first>=start+size));
    };
    for (const auto & lease : draft->leases) if (!disjoint(lease.get())) return false;
    for (auto * lease : compute_memory->workspace_leases()) if (!disjoint(lease)) return false;
    return seen;
}

bool llama_context::group_restore_recurrent_after_attention() {
    if (!group_execution || !group_execution->shared_roles || !group_target ||
            group_execution->phase!=llama_memory_work_phase::mtp_draft) return true;
    auto * restore=group_target->kv_resources.recurrent;
    if (!restore || std::none_of(restore->rs_idx.begin(),restore->rs_idx.end(),[](uint32_t index) {return index!=0;})) return true;
    // Resident history is selected by the next target's state-copy inputs.
    if (!restore->spill_enabled()) return true;
    return restore->staged_publication_enabled() && group_recurrent_restore_isolated() && restore->restore_spill_async(nullptr,false);
}

struct llama_context_group::implementation {
    const llama_model * model=nullptr;
    const llama_model * shared_model=nullptr;
    const llama_model * block_model=nullptr;
    uint32_t prediction_rows=0;
    llama_context_params params{};
    llama_context_params shared_params{};
    std::shared_ptr<llama_kv_stream_slot_table> table;
    std::vector<llama_context_group_sequence_state> parked;
    std::vector<llama_context_group_sequence_state> parked_auxiliary;
    std::vector<llama_kv_stream_slot_handle> handles;
    std::unique_ptr<llama_context> metadata;
    std::unique_ptr<llama_context> auxiliary_metadata;
    ggml_backend_buffer_ptr parent;
    llama_compute_arena_ptr vision;
    bool vision_lent=false;
    std::unique_ptr<llama_context> active;
    std::unique_ptr<llama_context> active_auxiliary;
    std::vector<llama_seq_id> selected;
};

llama_context_group::llama_context_group() = default;
llama_context_group::~llama_context_group() = default;

bool llama_context_group::selected_state_limit(uint32_t maximum,uint32_t & output) const {
    if (!impl || !impl->metadata || impl->active || impl->active_auxiliary || impl->vision_lent ||
            !maximum || maximum>impl->params.n_seq_max || impl->model->devices.size()!=1) return false;
    try {
        auto * type=llama_kv_stream_device_buffer_type(impl->model->devices.front().dev);
        if (!type) return false;
        const auto measure=[&](uint32_t count,size_t & bytes) {
            std::vector<llama_seq_id> selected;
            for (uint32_t slot=0;slot<count;++slot) selected.push_back(llama_seq_id(slot));
            size_t total=0;
            const auto bounded=[&](const llama_context & context) {
                const auto * cache=context.get_kv_resources().bounded;
                if (!cache) return true;
                llama_memory_params params{};params.type_k=cache->type_k();params.type_v=cache->type_v();
                params.kv_stream_selected_sequences=selected;
                llama_memory_bounded_kv_layout layout;
                if (!llama_kv_cache_iswa::bounded_storage_requirements(context.get_model().hparams,params,context.get_cparams(),
                        type,{1,1},layout) || layout.allocation.bytes>SIZE_MAX-total) return false;
                total+=layout.allocation.bytes;return true;
            };
            const auto * recurrent=impl->metadata->get_kv_resources().recurrent;
            if (recurrent) {
                auto params=impl->params;params.samplers=nullptr;params.n_samplers=0;
                llama_context pending(*impl->model,params,selected);
                ggml_backend_buffer_type_t current_type=nullptr;
                if (!pending.get_kv_resources().recurrent->current_storage_requirements(current_type,total) || current_type!=type) return false;
            }
            if (!bounded(*impl->metadata) || (impl->block_model && (!impl->auxiliary_metadata || !bounded(*impl->auxiliary_metadata)))) return false;
            // Shared MTP aliases the target bank; its storage must not be charged twice.
            bytes=total;return true;
        };
        return llama_context_group_state_limit(maximum,impl->params.shared_device_memory_bytes,measure,output);
    } catch (...) {return false;}
}

ggml_backend_buffer_t llama_context_group::prepare_vision_parent() {
    if (!impl || impl->active || impl->active_auxiliary || impl->vision_lent) return nullptr;
    try {
        if (!impl->parent) {
            auto * type=llama_kv_stream_device_buffer_type(impl->model->devices.front().dev);
            llama_compute_arena_ptr arena(ggml_backend_memory_arena_new(type,impl->params.shared_device_memory_bytes));
            if (!arena) return nullptr;
            impl->parent.reset(ggml_backend_buffer_retain(ggml_backend_memory_arena_parent(arena.get())));
        }
        return impl->parent.get();
    } catch (...) {return nullptr;}
}

bool llama_context_group::quote_vision(const std::vector<size_t> & bytes,llama_context_memory_create_result & result) const {
    result={};
    if (!impl || impl->active || impl->active_auxiliary || impl->vision_lent || !impl->parent || bytes.empty()) return false;
    const auto capacity=ggml_backend_buffer_get_size(impl->parent.get());
    const auto alignment=ggml_backend_buffer_get_alignment(impl->parent.get());
    if (!alignment) return false;
    size_t cursor=0;
    for (const auto size : bytes) {
        if (!size || cursor>SIZE_MAX-(alignment-1)) return false;
        cursor=(cursor+alignment-1)/alignment*alignment;
        if (size>SIZE_MAX-cursor) return false;
        cursor+=size;
    }
    result.available_bytes=capacity;result.required_bytes=cursor;
    result.additional_bytes=cursor>capacity ? cursor-capacity : 0;
    result.status=cursor>capacity ? llama_context_memory_create_status::insufficient_budget : llama_context_memory_create_status::success;
    return result.status==llama_context_memory_create_status::success;
}

bool llama_context_group::lend_vision(const std::vector<size_t> & bytes,std::vector<ggml_backend_memory_lease_t> & output) {
    llama_context_memory_create_result quote;
    if (!output.empty() || !quote_vision(bytes,quote)) return false;
    const auto alignment=ggml_backend_buffer_get_alignment(impl->parent.get());
    std::vector<ggml_backend_memory_lease_t> candidate;
    bool committed=false;
    const auto fail=[&] {
        for (auto * lease:candidate) ggml_backend_memory_lease_free(lease);
        if (impl->vision) {
            ggml_backend_memory_arena_rollback(impl->vision.get());
            if (committed && ggml_backend_memory_arena_begin(impl->vision.get(),0) &&
                    ggml_backend_memory_arena_commit(impl->vision.get())) impl->vision_lent=false;
        }
        return false;
    };
    try {
        candidate.reserve(bytes.size());
        if (!impl->vision) impl->vision.reset(ggml_backend_memory_arena_new_from_buffer(impl->parent.get()));
        auto * arena=impl->vision.get();
        if (!arena || !ggml_backend_memory_arena_begin(arena,0)) return false;
        for (size_t i=0;i<bytes.size();++i)
            if (!ggml_backend_memory_arena_reserve(arena,UINT64_MAX-i,bytes[i],alignment,0,nullptr)) return fail();
        if (!ggml_backend_memory_arena_commit(arena)) return fail();
        committed=true;impl->vision_lent=true;
        for (size_t i=0;i<bytes.size();++i) {
            auto * lease=ggml_backend_memory_arena_acquire(arena,UINT64_MAX-i);
            if (!lease) return fail();
            candidate.push_back(lease);
        }
        output=std::move(candidate);return true;
    } catch (...) {return fail();}
}

bool llama_context_group::release_vision() {
    if (!impl || !impl->vision_lent || !impl->vision) return false;
    if (!ggml_backend_memory_arena_begin(impl->vision.get(),0)) return false;
    if (!ggml_backend_memory_arena_commit(impl->vision.get())) {
        ggml_backend_memory_arena_rollback(impl->vision.get());return false;
    }
    impl->vision_lent=false;return true;
}

bool llama_context_group::prepare_dflash(const llama_model & model,llama_context_params params,uint32_t prediction_rows) {
    if (!impl || !impl->metadata || impl->active || impl->parent || impl->auxiliary_metadata || impl->shared_model || impl->block_model ||
            !group_dflash_supported(*impl->metadata,model,params,prediction_rows)) return false;
    try {
        params.ctx_other=impl->metadata.get();
        auto child=std::make_unique<llama_context>(model,params,group_slot_ids(params.n_seq_max),prediction_rows);
        auto layers=model.target_layer_ids;
        for (auto layer : model.target_layer_ids) impl->metadata->set_embeddings_layer_inp(uint32_t(layer),true);
        impl->metadata->serial_feature_layers=std::move(layers);impl->metadata->serial_feature_producer=true;
        impl->metadata->serial_feature_images=true;
        impl->block_model=&model;impl->prediction_rows=prediction_rows;impl->shared_params=params;
        impl->shared_params.samplers=nullptr;impl->shared_params.n_samplers=0;impl->shared_params.ctx_other=nullptr;
        impl->auxiliary_metadata=std::move(child);return true;
    } catch (...) {return false;}
}

bool llama_context_group::prepare_shared_mtp(const llama_model & model,llama_context_params params) {
    if (!impl || !impl->metadata || impl->active || impl->parent || impl->auxiliary_metadata || impl->shared_model ||
            impl->model->arch!=LLM_ARCH_GEMMA4 || model.arch!=LLM_ARCH_GEMMA4_ASSISTANT ||
            impl->params.kv_stream_auxiliary_layers || !impl->params.kv_stream_verify_max ||
            (params.ctx_other && params.ctx_other!=impl->metadata.get())) return false;
    try {
        params.ctx_other=impl->metadata.get();
        auto child=std::make_unique<llama_context>(model,params,group_slot_ids(params.n_seq_max));
        child->set_embeddings_nextn(true,true);
        impl->metadata->set_embeddings_nextn(true,false);
        impl->shared_model=&model;impl->shared_params=params;
        impl->shared_params.samplers=nullptr;impl->shared_params.n_samplers=0;impl->shared_params.ctx_other=nullptr;
        impl->auxiliary_metadata=std::move(child);return true;
    } catch (...) {return false;}
}

std::unique_ptr<llama_context_group> llama_context_group::create(const llama_model & model,llama_context_params params) {
    if (params.n_seq_max<2 || params.n_seq_max>LLAMA_MAX_SEQ || !params.kv_unified || !params.shared_device_memory_bytes ||
            !group_rollback_supported(model,params.n_rs_seq) || !group_auxiliary_supported(model,params)) return {};
    try {
        auto metadata=std::make_unique<llama_context>(model,params,group_slot_ids(params.n_seq_max));
        auto * full=metadata->get_kv_resources().full;
        if (!full || model.devices.size()!=1) return {};
        const auto layers=full->get_layer_ids();if (layers.empty()) return {};
        const auto first=layers.front();auto * device=model.devices.front().dev;
        const ggml_kv_stream_shape shape{full->type_k(),full->type_v(),model.hparams.n_embd_head_k(first),
            model.hparams.n_embd_head_v(first),model.hparams.n_head_kv(first),256,128};
        ggml_backend_t backend=nullptr;auto * scheduler=metadata->get_sched();
        for (int i=0;i<ggml_backend_sched_get_n_backends(scheduler);++i) {
            auto * candidate=ggml_backend_sched_get_backend(scheduler,i);
            if (ggml_backend_get_device(candidate)==device) backend=candidate;
        }
        const auto get=reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(device),"ggml_backend_kv_stream_partial_ops"));
        const auto * partial=get ? get() : nullptr;
        if (!backend || !partial || !partial->capabilities) return {};
        const auto limits=metadata->get_cparams().stream_limits;
        std::shared_ptr<llama_kv_stream_slot_table> table=llama_kv_stream_slot_table::create(limits);
        const uint32_t roles=1+params.kv_stream_auxiliary_layers;
        if (!table || limits.slots>UINT32_MAX/roles) return {};
        if (roles>1) {
            const auto layer=model.hparams.n_layer();
            if (model.hparams.n_embd_head_k(layer)!=shape.head_dim_k || model.hparams.n_embd_head_v(layer)!=shape.head_dim_v ||
                    model.hparams.n_head_kv(layer)!=shape.heads || model.hparams.n_head(layer)!=model.hparams.n_head(first)) return {};
        }
        const auto first_id=llama_kv_stream_cache_ids_reserve(limits.slots*roles);
        if (!first_id) return {};
        std::vector<llama_kv_stream_slot_handle> handles(limits.slots);
        for (uint32_t slot=0;slot<limits.slots;++slot) {
            std::vector<std::shared_ptr<llama_kv_stream_logical_cache>> caches;
            for (uint32_t role=0;role<roles;++role) {
                const llama_kv_stream_host_config config{first_id+slot*roles+role,shape,partial->capabilities(backend,shape.type_k,shape.type_v),
                    limits.slot_limit,role ? 1 : uint32_t(layers.size())};
                std::shared_ptr<llama_kv_stream_logical_cache> cache=llama_kv_stream_logical_cache::create(
                    llama_kv_stream_host::from_reservation(llama_kv_stream_host_reservation::create(config,device)));
                if (!cache) return {};
                caches.push_back(std::move(cache));
            }
            if (!table->bind(slot,std::move(caches),handles[slot])) return {};
        }
        params.n_ctx=limits.slot_limit;
        auto owner=create(model,params,table);if (!owner) return {};
        metadata->group_source_table=table;
        owner->impl->metadata=std::move(metadata);owner->impl->handles=std::move(handles);
        if (params.kv_stream_auxiliary_layers) {
            auto child=params;
            child.ctx_type=LLAMA_CONTEXT_TYPE_MTP;child.ctx_other=owner->impl->metadata.get();
            child.shared_device_memory_bytes=0;child.n_rs_seq=0;child.kv_stream_auxiliary_layers=0;
            child.samplers=nullptr;child.n_samplers=0;
            owner->impl->auxiliary_metadata=std::make_unique<llama_context>(model,child,group_slot_ids(params.n_seq_max));
            owner->impl->auxiliary_metadata->set_embeddings_nextn(true,true);
        }
        return owner;
    } catch (...) {return {};}
}

std::unique_ptr<llama_context_group> llama_context_group::create(const llama_model & model,llama_context_params params,
        std::shared_ptr<llama_kv_stream_slot_table> table) {
    if (!table || params.n_seq_max<2 || params.n_seq_max>LLAMA_MAX_SEQ || table->limits().slots!=params.n_seq_max || !params.kv_unified ||
            !params.shared_device_memory_bytes || !group_rollback_supported(model,params.n_rs_seq) || !group_auxiliary_supported(model,params) ||
            (params.n_ctx && params.n_ctx!=table->limits().slot_limit)) return {};
    try {
        auto owner=std::unique_ptr<llama_context_group>(new llama_context_group);
        owner->impl=std::make_unique<implementation>();
        owner->impl->model=&model;owner->impl->params=params;owner->impl->table=std::move(table);
        owner->impl->parked.resize(params.n_seq_max);owner->impl->parked_auxiliary.resize(params.n_seq_max);return owner;
    } catch (...) {return {};}
}

namespace {
bool restore_parked_group_projection(const llama_model & model,const std::shared_ptr<llama_kv_stream_slot_table> & table,
        const std::shared_ptr<llama_kv_stream_slot_projection> & projection,
        const std::vector<llama_context_group_sequence_state> & parked,std::vector<llama_context_group_sequence_state> & output) {
    const bool history=std::any_of(projection->sources.begin(),projection->sources.end(),[](const auto & source) {return source.tokens!=0;});
    if (!history) return true;
    std::vector<llama_context_group_sequence_state> states;
    std::vector<std::shared_ptr<llama_kv_cells>> canonical;
    for (size_t i=0;i<projection->handles.size();++i) {
        const auto & handle=projection->handles[i];const auto & source=projection->sources[i];
        if (handle.slot>=parked.size()) return false;
        auto saved=parked[handle.slot];
        if (!source.tokens && !saved.model && saved.sequence==-1 && !saved.request_epoch && !saved.cache_id &&
                !saved.cache_generation && !saved.tokens && saved.recurrent.empty() && saved.bounded.empty() && saved.full_columns.empty()) {
            std::shared_ptr<llama_kv_cells> cells;
            if (!table->get_cells(handle,projection->role,cells) || cells->get_used()) return false;
            saved.model=&model;saved.sequence=llama_seq_id(handle.slot);saved.request_epoch=handle.generation;
            saved.cache_id=source.id;saved.cache_generation=source.generation;
        }
        if (saved.sequence!=llama_seq_id(handle.slot) || saved.request_epoch!=handle.generation || saved.model!=&model ||
                saved.cache_id!=source.id || saved.cache_generation!=source.generation || saved.tokens!=source.tokens ||
                saved.full_columns.size()!=saved.tokens) return false;
        std::vector<bool> seen(saved.tokens,false);
        for (auto token : saved.full_columns) {
            if (token>=seen.size() || seen[token]) return false;
            seen[token]=true;
        }
        std::shared_ptr<llama_kv_cells> cells;
        if (!table->get_cells(handle,projection->role,cells)) return false;
        canonical.push_back(std::move(cells));states.push_back(std::move(saved));
    }
    // Native sequence checkpoints retain physical column order, including reused rollback holes.
    projection->cells->reset();uint32_t first=0;
    for (size_t i=0;i<states.size();++i) {
        const auto & order=states[i].full_columns;
        projection->cells->set(first,canonical[i]->cp(order));
        for (size_t row=0;row<order.size();++row) projection->columns[first+row]={uint32_t(i),order[row]};
        first+=uint32_t(order.size());
    }
    output=std::move(states);return true;
}
}

bool llama_context_group::activate(const llama_kv_stream_selection & selection,
        const std::vector<llama_kv_stream_slot_handle> & handles,const llama_batch & representative,
        llama_memory_work_phase phase,uint32_t profile,uint64_t selection_revision,llama_context_memory_create_result * result) {
    if (result) *result={};
    if (!impl || impl->active || impl->active_auxiliary || impl->vision_lent || !profile || !selection_revision) return false;
    try {
        auto projection=std::make_shared<llama_kv_stream_slot_projection>();
        if (!impl->table->project(selection,handles,0,*projection)) return false;
        std::vector<llama_seq_id> selected;
        std::vector<llama_context_group_sequence_state> states;
        selected.reserve(handles.size());states.reserve(handles.size());
        for (size_t i=0;i<handles.size();++i) {
            const auto & handle=handles[i];
            if (handle.slot>=impl->parked.size()) return false;
            selected.push_back(llama_seq_id(handle.slot));
        }
        if (!restore_parked_group_projection(*impl->model,impl->table,projection,impl->parked,states)) return false;
        auto candidate=std::make_unique<llama_context>(*impl->model,impl->params,selected);
        if (impl->metadata) {
            const auto & prepared=impl->metadata->get_cparams();
            candidate->set_causal_attn(prepared.causal_attn);
            candidate->set_embeddings_nextn(prepared.embeddings_nextn,prepared.embeddings_nextn_masked);
            if (!candidate->bind_group_samplers_from(*impl->metadata)) return false;
        }
        if (impl->block_model) {
            candidate->serial_feature_layers=impl->block_model->target_layer_ids;candidate->serial_feature_producer=true;
            candidate->serial_feature_images=impl->metadata->serial_feature_images;
            for (auto layer : candidate->serial_feature_layers) candidate->set_embeddings_layer_inp(uint32_t(layer),true);
        }
        std::unique_ptr<llama_context> child;
        std::shared_ptr<llama_kv_stream_slot_projection> child_projection;
        std::vector<llama_context_group_sequence_state> child_states;
        if (impl->params.kv_stream_auxiliary_layers || impl->shared_model || impl->block_model) {
            if (impl->shared_model || impl->block_model) child_projection=projection;
            else {
                child_projection=std::make_shared<llama_kv_stream_slot_projection>();
                if (!impl->table->project(selection,handles,1,*child_projection)) return false;
                if (!restore_parked_group_projection(*impl->model,impl->table,child_projection,impl->parked_auxiliary,child_states)) return false;
            }
            auto child_params=impl->shared_model || impl->block_model ? impl->shared_params : impl->params;
            child_params.ctx_type=impl->block_model ? LLAMA_CONTEXT_TYPE_DEFAULT : LLAMA_CONTEXT_TYPE_MTP;child_params.ctx_other=candidate.get();
            child_params.shared_device_memory_bytes=0;child_params.n_rs_seq=0;child_params.kv_stream_auxiliary_layers=0;
            child_params.samplers=nullptr;child_params.n_samplers=0;
            if (impl->block_model) {
                if (std::any_of(projection->sources.begin(),projection->sources.end(),[](const auto & source) {return source.tokens!=0;})) {
                    for (size_t i=0;i<handles.size();++i) {
                        const auto handle=handles[i];const auto & source=projection->sources[i];
                        auto saved=impl->parked_auxiliary[handle.slot];
                        if (!source.tokens && !saved.model && saved.sequence==-1 && !saved.request_epoch && !saved.cache_id &&
                                !saved.cache_generation && !saved.tokens && saved.recurrent.empty() && saved.bounded.empty() && saved.full_columns.empty()) {
                            saved=states[i];saved.model=impl->block_model;
                        }
                        if (saved.model!=impl->block_model || saved.sequence!=llama_seq_id(handle.slot) ||
                                saved.request_epoch!=handle.generation || saved.cache_id!=source.id ||
                                saved.cache_generation!=source.generation || saved.tokens!=source.tokens ||
                                saved.full_columns!=states[i].full_columns) return false;
                        child_states.push_back(std::move(saved));
                    }
                }
                child=std::make_unique<llama_context>(*impl->block_model,child_params,selected,impl->prediction_rows);
            } else child=std::make_unique<llama_context>(impl->shared_model ? *impl->shared_model : *impl->model,child_params,selected);
            child->set_embeddings_nextn(true,!impl->block_model);
            if (impl->auxiliary_metadata && !child->bind_group_samplers_from(*impl->auxiliary_metadata)) return false;
        }
        if (!candidate->initialize_group_execution(impl->table,projection,representative,phase,profile,selection_revision,
                impl->parent.get(),states,result,child.get(),child_projection,child_states)) return false;
        if (!impl->parent) impl->parent.reset(ggml_backend_buffer_retain(candidate->group_parent()));
        impl->active=std::move(candidate);impl->active_auxiliary=std::move(child);impl->selected=std::move(selected);return true;
    } catch (const std::bad_alloc &) {
        if (result) result->status=llama_context_memory_create_status::allocation_failed;
        return false;
    } catch (...) {return false;}
}

bool llama_context_group::park() {
    if (!impl || !impl->active) return false;
    try {
        std::vector<llama_context_group_sequence_state> states;
        if (!impl->active->save_group_sequences(states)) return false;
        std::vector<llama_context_group_sequence_state> auxiliary_states;
        if (impl->active_auxiliary && !impl->shared_model && !impl->active_auxiliary->save_group_sequences(auxiliary_states)) return false;
        for (const auto & state : states)
            if (state.sequence<0 || size_t(state.sequence)>=impl->parked.size()) return false;
        for (const auto & state : auxiliary_states)
            if (state.sequence<0 || size_t(state.sequence)>=impl->parked_auxiliary.size()) return false;
        if (impl->active_auxiliary && !impl->active_auxiliary->detach_group_execution()) return false;
        if (!impl->active->detach_group_execution()) return false;
        // Publish only exported members after retirement succeeds; keep parked peers in place.
        static_assert(std::is_nothrow_move_assignable<llama_context_group_sequence_state>::value,"state publication must not allocate");
        for (auto & state : states) impl->parked[size_t(state.sequence)]=std::move(state);
        for (auto & state : auxiliary_states) impl->parked_auxiliary[size_t(state.sequence)]=std::move(state);
        impl->active_auxiliary.reset();impl->active.reset();impl->selected.clear();return true;
    } catch (...) {return false;}
}

bool llama_context_group::discard() noexcept {
    if (!impl || impl->vision_lent) return false;
    if (!impl->active) return true;
    if (impl->active_auxiliary && impl->active_auxiliary->get_compute_memory() &&
            !impl->active_auxiliary->abort_group_execution()) return false;
    if (!impl->active->abort_group_execution()) return false;
    impl->active_auxiliary.reset();impl->active.reset();
    for (auto slot : impl->selected) {impl->parked[size_t(slot)]={};impl->parked_auxiliary[size_t(slot)]={};}
    impl->selected.clear();return true;
}

llama_context * llama_context_group::context() const noexcept {return impl ? impl->active.get() : nullptr;}
llama_context * llama_context_group::auxiliary_context() const noexcept {return impl ? impl->active_auxiliary.get() : nullptr;}

llama_context * llama_context_group::metadata_context() const noexcept {return impl ? impl->metadata.get() : nullptr;}
llama_context * llama_context_group::auxiliary_metadata_context() const noexcept {return impl ? impl->auxiliary_metadata.get() : nullptr;}
std::shared_ptr<llama_kv_stream_slot_table> llama_context_group::source_table() const noexcept {return impl ? impl->table : nullptr;}
const std::vector<llama_kv_stream_slot_handle> & llama_context_group::source_handles() const noexcept {return impl->handles;}

bool llama_context_group::save_request(uint32_t slot,llama_context_group_checkpoint & output,bool copy_bytes) const {
    if (!impl || impl->active || impl->vision_lent || slot>=impl->handles.size()) return false;
    try {
        llama_context_group_checkpoint saved;
        saved.model=impl->model;saved.auxiliary_model=impl->block_model ? impl->block_model : impl->shared_model;saved.slot=slot;
        const auto handle=impl->handles[slot];
        for (uint32_t role=0;role<=impl->params.kv_stream_auxiliary_layers;++role) {
            std::shared_ptr<llama_kv_stream_logical_cache> cache;std::shared_ptr<llama_kv_cells> cells;
            llama_kv_stream_logical_snapshot history;
            if (!impl->table->get(handle,role,cache) || !cache->save(history,copy_bytes) ||
                    !impl->table->get_cells(handle,role,cells) || cells->get_used()!=history.tokens) return false;
            saved.histories.push_back(std::move(history));saved.cells.push_back(std::make_shared<llama_kv_cells>(*cells));
        }
        saved.target=impl->parked[slot];saved.auxiliary=impl->parked_auxiliary[slot];
        const auto & source=saved.histories.front();
        std::shared_ptr<llama_kv_stream_logical_cache> cache;
        if (!impl->table->get(handle,0,cache)) return false;
        if (source.tokens && (saved.target.model!=impl->model || saved.target.sequence!=llama_seq_id(slot) ||
                saved.target.request_epoch!=handle.generation || saved.target.tokens!=source.tokens ||
                saved.target.cache_id!=cache->identity().id || saved.target.cache_generation!=cache->identity().generation)) return false;
        output=std::move(saved);return true;
    } catch (const std::bad_alloc &) {return false;}
}

bool llama_context_group::load_request(uint32_t slot,const llama_context_group_checkpoint & saved) {
    if (!impl || impl->active || impl->vision_lent || slot>=impl->handles.size() || saved.slot>=LLAMA_MAX_SEQ ||
            saved.model!=impl->model || saved.auxiliary_model!=(impl->block_model ? impl->block_model : impl->shared_model) ||
            saved.histories.size()!=size_t(1+impl->params.kv_stream_auxiliary_layers) || saved.cells.size()!=saved.histories.size()) return false;
    try {
        auto target=saved.target,auxiliary=saved.auxiliary;
        std::vector<std::shared_ptr<llama_kv_stream_logical_cache>> caches;
        std::vector<std::shared_ptr<llama_kv_cells>> cells,relocated;
        for (uint32_t role=0;role<saved.histories.size();++role) {
            std::shared_ptr<llama_kv_stream_logical_cache> cache;std::shared_ptr<llama_kv_cells> metadata;
            const auto & history=saved.histories[role];
            if (!impl->table->get(impl->handles[slot],role,cache) || !cache->ready() || !saved.cells[role] ||
                    !impl->table->get_cells(impl->handles[slot],role,metadata) || saved.cells[role]->size()!=metadata->size() ||
                    history.tokens>metadata->size() || saved.cells[role]->get_used()!=history.tokens) return false;
            for (uint32_t row=0;row<metadata->size();++row) {
                const auto & source=*saved.cells[role];
                if ((row<history.tokens && (source.is_empty(row) || source.seq_count(row)!=1 || !source.seq_has(row,llama_seq_id(saved.slot)))) ||
                        (row>=history.tokens && !source.is_empty(row))) return false;
            }
            if (!cache->accepts(history)) return false;
            auto restored=std::make_shared<llama_kv_cells>(*saved.cells[role]);
            if (slot!=saved.slot) for (uint32_t row=0;row<history.tokens;++row) {
                restored->seq_add(row,llama_seq_id(slot));restored->seq_rm(row,llama_seq_id(saved.slot));
            }
            relocated.push_back(std::move(restored));caches.push_back(std::move(cache));cells.push_back(std::move(metadata));
        }
        const auto validate=[&](const llama_context_group_sequence_state & state,const llama_model * model,
                const llama_kv_stream_logical_snapshot & history,const llama_context * metadata,bool block) {
            if (state.model!=model || state.sequence!=llama_seq_id(saved.slot) || state.tokens!=history.tokens ||
                    state.full_columns.size()!=history.tokens || !metadata) return false;
            const auto & resources=metadata->get_kv_resources();
            if (history.tokens && (bool(resources.recurrent)!=!state.recurrent.empty() ||
                    bool(block || resources.bounded)!=!state.bounded.empty())) return false;
            std::vector<bool> seen(history.tokens,false);
            for (auto row : state.full_columns) {
                if (row>=seen.size() || seen[row]) return false;
                seen[row]=true;
            }
            return true;
        };
        if (saved.histories.front().tokens && !validate(target,impl->model,saved.histories.front(),impl->metadata.get(),false)) return false;
        if (impl->params.kv_stream_auxiliary_layers || impl->block_model) {
            const auto role=impl->params.kv_stream_auxiliary_layers ? 1 : 0;
            if (saved.histories[role].tokens && !validate(auxiliary,impl->block_model ? impl->block_model : impl->model,
                    saved.histories[role],impl->auxiliary_metadata.get(),impl->block_model!=nullptr)) return false;
            if (impl->block_model && auxiliary.full_columns!=target.full_columns) return false;
        } else if (auxiliary.model || !auxiliary.recurrent.empty() || !auxiliary.bounded.empty() || !auxiliary.full_columns.empty()) return false;
        const bool prefix=saved.histories.front().prefix!=nullptr;
        for (const auto & history : saved.histories) if (bool(history.prefix)!=prefix) return false;
        if (prefix) {
            llama_kv_stream_slot_handle next;
            if (!impl->table->advance(impl->handles[slot],next)) return false;
            impl->handles[slot]=next;
        } else if (!reset_request(slot)) return false;
        for (size_t role=0;role<caches.size();++role) {
            if (!caches[role]->load(saved.histories[role])) return false;
            *cells[role]=std::move(*relocated[role]);
        }
        const auto handle=impl->handles[slot];
        const auto rebind=[&](llama_context_group_sequence_state & state,uint32_t role) {
            if (!state.model) return;
            state.sequence=llama_seq_id(slot);state.request_epoch=handle.generation;
            state.cache_id=caches[role]->identity().id;state.cache_generation=caches[role]->identity().generation;
        };
        rebind(target,0);rebind(auxiliary,impl->params.kv_stream_auxiliary_layers ? 1 : 0);
        impl->parked[slot]=std::move(target);impl->parked_auxiliary[slot]=std::move(auxiliary);return true;
    } catch (const std::bad_alloc &) {return false;}
}

namespace {
constexpr uint32_t group_checkpoint_magic=0x47534350;
constexpr uint32_t group_checkpoint_version=2;

std::array<uint64_t,8> group_checkpoint_model(const llama_model * model) {
    if (!model) return {};
    return {uint64_t(model->arch),uint64_t(model->ftype()),model->n_elements(),model->size(),
        model->hparams.n_ctx_train,model->hparams.n_layer(),model->hparams.n_embd,model->vocab.n_tokens()};
}
}

bool llama_context_group::write_request(uint32_t slot,llama_io_write_i & io) const {
    llama_context_group_checkpoint saved;
    if (!save_request(slot,saved,false)) return false;
    return write_checkpoint(saved,io,true);
}

bool llama_context_group::write_checkpoint(const llama_context_group_checkpoint & saved,llama_io_write_i & io,bool copy_bytes) const {
    if (!impl || impl->active || impl->vision_lent || saved.slot>=impl->handles.size() || saved.model!=impl->model) return false;
    try {
        const std::array<uint32_t,3> header={group_checkpoint_magic,group_checkpoint_version,saved.slot};
        io.write(header.data(),sizeof(header));
        for (const auto * model : {saved.model,saved.auxiliary_model}) {
            const auto signature=group_checkpoint_model(model);io.write(signature.data(),sizeof(signature));
        }
        const std::array<uint32_t,3> modes={impl->params.n_rs_seq,impl->params.kv_stream_auxiliary_layers,impl->prediction_rows};
        io.write(modes.data(),sizeof(modes));
        for (size_t role=0;role<saved.histories.size();++role) {
            const auto & history=saved.histories[role];const auto & shape=history.shape;
            const std::array<uint64_t,8> descriptor={uint64_t(shape.type_k),uint64_t(shape.type_v),uint64_t(shape.head_dim_k),
                uint64_t(shape.head_dim_v),uint64_t(shape.heads),uint64_t(shape.page_tokens),shape.alignment,history.layers};
            io.write(descriptor.data(),sizeof(descriptor));
            const uint64_t tokens=history.tokens;io.write(&tokens,sizeof(tokens));
            const uint32_t copied=copy_bytes;io.write(&copied,sizeof(copied));
            std::shared_ptr<llama_kv_stream_logical_cache> cache;
            if (!impl->table->get(impl->handles[saved.slot],uint32_t(role),cache) || !cache->accepts(history) || (!copy_bytes && !history.prefix)) return false;
            if (copy_bytes) {
                if (history.prefix) {
                    for (uint32_t layer=0;layer<history.layers;++layer) {
                        llama_kv_stream_host_layer planes;if (!cache->host()->layer(layer,planes)) return false;
                        io.write(planes.k,history.tokens*cache->host()->layout().k_token_bytes);
                        io.write(planes.v,history.tokens*cache->host()->layout().v_token_bytes);
                    }
                } else io.write(history.bytes.data(),history.bytes.size());
            }
            for (uint32_t row=0;row<history.tokens;++row) {
                const auto position=saved.cells[role]->pos_get(row);const auto ext=saved.cells[role]->ext_get(row);
                io.write(&position,sizeof(position));io.write(&ext.x,sizeof(ext.x));io.write(&ext.y,sizeof(ext.y));io.write(&ext.tok,sizeof(ext.tok));
            }
        }
        for (const auto * state : {&saved.target,&saved.auxiliary}) {
            const uint32_t present=state->model!=nullptr;io.write(&present,sizeof(present));
            if (!present) continue;
            const uint64_t tokens=state->tokens;io.write(&tokens,sizeof(tokens));io.write(&state->draft_position,sizeof(state->draft_position));
            for (const auto * bytes : {&state->recurrent,&state->bounded}) {
                const uint64_t size=bytes->size();io.write(&size,sizeof(size));io.write(bytes->data(),bytes->size());
            }
            const uint64_t columns=state->full_columns.size();io.write(&columns,sizeof(columns));
            io.write(state->full_columns.data(),state->full_columns.size()*sizeof(uint32_t));
        }
        return true;
    } catch (const std::exception &) {return false;}
}

bool llama_context_group::read_request(uint32_t slot,llama_io_read_i & io,size_t available) {
    llama_context_group_checkpoint saved;
    return read_checkpoint(slot,io,available,saved) && load_request(slot,saved);
}

bool llama_context_group::read_checkpoint(uint32_t slot,llama_io_read_i & io,size_t available,
        llama_context_group_checkpoint & output,bool allow_prefix) {
    if (!impl || impl->active || impl->vision_lent || slot>=impl->handles.size()) return false;
    try {
        const size_t start=io.n_bytes();
        const auto read=[&](void * destination,size_t bytes) {
            const auto consumed=io.n_bytes()-start;
            if (consumed>available || bytes>available-consumed) throw std::runtime_error("truncated group checkpoint");
            if (bytes) io.read(destination,bytes);
        };
        std::array<uint32_t,3> header{};read(header.data(),sizeof(header));
        if (header[0]!=group_checkpoint_magic || header[1]!=group_checkpoint_version || header[2]>=LLAMA_MAX_SEQ) return false;
        llama_context_group_checkpoint saved;saved.slot=header[2];saved.model=impl->model;
        saved.auxiliary_model=impl->block_model ? impl->block_model : impl->shared_model;
        for (const auto * model : {saved.model,saved.auxiliary_model}) {
            std::array<uint64_t,8> signature{};read(signature.data(),sizeof(signature));
            if (signature!=group_checkpoint_model(model)) return false;
        }
        std::array<uint32_t,3> modes{};read(modes.data(),sizeof(modes));
        if (modes!=std::array<uint32_t,3>{impl->params.n_rs_seq,impl->params.kv_stream_auxiliary_layers,impl->prediction_rows}) return false;
        for (uint32_t role=0;role<=impl->params.kv_stream_auxiliary_layers;++role) {
            std::shared_ptr<llama_kv_stream_logical_cache> cache;std::shared_ptr<llama_kv_cells> cells;
            if (!impl->table->get(impl->handles[slot],role,cache) || !impl->table->get_cells(impl->handles[slot],role,cells)) return false;
            const auto & shape=cache->host()->config().shape;
            const std::array<uint64_t,8> expected={uint64_t(shape.type_k),uint64_t(shape.type_v),uint64_t(shape.head_dim_k),
                uint64_t(shape.head_dim_v),uint64_t(shape.heads),uint64_t(shape.page_tokens),shape.alignment,cache->host()->config().layers};
            std::array<uint64_t,8> descriptor{};read(descriptor.data(),sizeof(descriptor));
            uint64_t tokens;read(&tokens,sizeof(tokens));
            if (descriptor!=expected || tokens>cache->host()->config().context_tokens) return false;
            const auto & layout=cache->host()->layout();
            if (layout.k_token_bytes>SIZE_MAX-layout.v_token_bytes) return false;
            const size_t stride=layout.k_token_bytes+layout.v_token_bytes;
            if ((stride && tokens>SIZE_MAX/stride) || (tokens && stride && descriptor[7]>SIZE_MAX/(tokens*stride))) return false;
            const size_t bytes=size_t(tokens)*stride*descriptor[7];
            uint32_t copied;read(&copied,sizeof(copied));
            if (copied>1 || (!copied && !allow_prefix)) return false;
            const auto consumed=io.n_bytes()-start;
            if (consumed>available || (copied && bytes>available-consumed)) return false;
            llama_kv_stream_logical_snapshot history;history.shape=shape;history.layers=uint32_t(descriptor[7]);history.tokens=size_t(tokens);
            if (copied) {history.bytes.resize(bytes);read(history.bytes.data(),bytes);}
            else if (!cache->save_prefix(size_t(tokens),history)) return false;
            auto metadata=std::make_shared<llama_kv_cells>();metadata->resize(cells->size());
            for (uint32_t row=0;row<tokens;++row) {
                llama_pos position;llama_kv_cell_ext ext;
                read(&position,sizeof(position));read(&ext.x,sizeof(ext.x));read(&ext.y,sizeof(ext.y));read(&ext.tok,sizeof(ext.tok));
                if (position<0 || ext.x<0 || ext.y<0 || (ext.tok!=LLAMA_TOKEN_NULL && (ext.tok<0 || uint32_t(ext.tok)>=impl->model->vocab.n_tokens()))) return false;
                if (!copied && (cells->is_empty(row) || cells->pos_get(row)!=position ||
                        cells->ext_get(row).x!=ext.x || cells->ext_get(row).y!=ext.y || cells->ext_get(row).tok!=ext.tok)) return false;
                metadata->pos_set(row,position);metadata->seq_add(row,llama_seq_id(saved.slot));metadata->ext_set(row,ext);
            }
            saved.histories.push_back(std::move(history));saved.cells.push_back(std::move(metadata));
        }
        const uint64_t metadata_limit=64*(uint64_t(impl->table->limits().slot_limit)+impl->model->hparams.n_layer()+1);
        if (metadata_limit>SIZE_MAX-impl->params.shared_device_memory_bytes) return false;
        const size_t native_limit=impl->params.shared_device_memory_bytes+size_t(metadata_limit);
        for (auto * state : {&saved.target,&saved.auxiliary}) {
            uint32_t present;read(&present,sizeof(present));if (present>1) return false;
            if (!present) continue;
            state->model=state==&saved.target || impl->params.kv_stream_auxiliary_layers ? impl->model : saved.auxiliary_model;
            if (!state->model) return false;
            state->sequence=llama_seq_id(saved.slot);
            uint64_t tokens;read(&tokens,sizeof(tokens));
            if (tokens>impl->table->limits().slot_limit) return false;
            state->tokens=size_t(tokens);read(&state->draft_position,sizeof(state->draft_position));
            for (auto * bytes : {&state->recurrent,&state->bounded}) {
                uint64_t size;read(&size,sizeof(size));const auto consumed=io.n_bytes()-start;
                if (size>native_limit || consumed>available || size>available-consumed) return false;
                bytes->resize(size_t(size));read(bytes->data(),bytes->size());
            }
            uint64_t columns;read(&columns,sizeof(columns));const auto consumed=io.n_bytes()-start;
            if (columns!=tokens || columns>SIZE_MAX/sizeof(uint32_t) || consumed>available || columns*sizeof(uint32_t)>available-consumed) return false;
            state->full_columns.resize(size_t(columns));read(state->full_columns.data(),state->full_columns.size()*sizeof(uint32_t));
        }
        if (saved.slot!=slot) {
            for (size_t role=0;role<saved.cells.size();++role) for (uint32_t row=0;row<saved.histories[role].tokens;++row) {
                saved.cells[role]->seq_add(row,llama_seq_id(slot));saved.cells[role]->seq_rm(row,llama_seq_id(saved.slot));
            }
            for (auto * state : {&saved.target,&saved.auxiliary}) if (state->model) state->sequence=llama_seq_id(slot);
            saved.slot=slot;
        }
        output=std::move(saved);return true;
    } catch (const std::exception &) {return false;}
}

size_t llama_context_group_checkpoint::payload_size() const noexcept {
    size_t bytes=0;
    const auto add=[&](size_t count) {bytes=count>SIZE_MAX-bytes ? SIZE_MAX : bytes+count;};
    for (const auto & history : histories) {
        add(history.bytes.size());
        if (history.tokens>SIZE_MAX/(4*sizeof(int32_t))) return SIZE_MAX;
        add(history.tokens*4*sizeof(int32_t));
    }
    for (const auto * state : {&target,&auxiliary}) {
        add(state->recurrent.size());add(state->bounded.size());
        if (state->full_columns.size()>SIZE_MAX/sizeof(uint32_t)) return SIZE_MAX;
        add(state->full_columns.size()*sizeof(uint32_t));
    }
    return bytes;
}

size_t llama_context_group::parked_payload_bytes() const noexcept {
    if (!impl) return 0;
    size_t bytes=0;
    for (const auto * states : {&impl->parked,&impl->parked_auxiliary}) for (const auto & state : *states)
        bytes+=state.recurrent.size()+state.bounded.size()+state.full_columns.size()*sizeof(uint32_t);
    return bytes;
}

bool llama_context_group::advance_request(uint32_t slot) {
    if (!impl || impl->active || impl->vision_lent || slot>=impl->handles.size()) return false;
    const auto handle=impl->handles[slot];
    const auto valid=[&](const llama_context_group_sequence_state & state,const llama_model * model,uint32_t role) {
        std::shared_ptr<llama_kv_stream_logical_cache> cache;
        if (!impl->table->get(handle,role,cache) || !cache->ready()) return false;
        return !cache->tokens() || (state.model==model && state.sequence==llama_seq_id(slot) && state.request_epoch==handle.generation &&
            state.cache_id==cache->identity().id && state.cache_generation==cache->identity().generation && state.tokens==cache->tokens());
    };
    if (!valid(impl->parked[slot],impl->model,0) ||
            (impl->params.kv_stream_auxiliary_layers && !valid(impl->parked_auxiliary[slot],impl->model,1)) ||
            (impl->block_model && !valid(impl->parked_auxiliary[slot],impl->block_model,0))) return false;
    llama_kv_stream_slot_handle next;
    if (!impl->table->advance(impl->handles[slot],next)) return false;
    impl->handles[slot]=next;
    if (impl->parked[slot].model) impl->parked[slot].request_epoch=next.generation;
    if (impl->parked_auxiliary[slot].model) impl->parked_auxiliary[slot].request_epoch=next.generation;
    return true;
}

bool llama_context_group::reset_request(uint32_t slot) {
    if (!impl || impl->active || impl->vision_lent || slot>=impl->handles.size() || impl->handles[slot].generation==UINT64_MAX) return false;
    try {
        const auto handle=impl->handles[slot];
        std::vector<std::shared_ptr<llama_kv_stream_logical_cache>> caches;
        std::vector<std::shared_ptr<llama_kv_cells>> cells;
        for (uint32_t role=0;role<=impl->params.kv_stream_auxiliary_layers;++role) {
            std::shared_ptr<llama_kv_stream_logical_cache> cache;std::shared_ptr<llama_kv_cells> metadata;
            if (!impl->table->get(handle,role,cache) || !cache->ready() || !impl->table->get_cells(handle,role,metadata)) return false;
            caches.push_back(std::move(cache));cells.push_back(std::move(metadata));
        }
        for (const auto & cache : caches) if (!cache->truncate(0)) return false;
        for (const auto & metadata : cells) metadata->reset();
        llama_kv_stream_slot_handle next;
        if (!impl->table->advance(handle,next)) return false;
        impl->parked[slot]={};impl->parked_auxiliary[slot]={};impl->handles[slot]=next;return true;
    } catch (...) {return false;}
}
