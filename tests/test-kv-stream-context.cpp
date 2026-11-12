#include "llama.h"
#include "../src/llama-memory-hybrid.h"
#include "../src/llama-context.h"
#include "../src/llama-memory-workspace.h"
#include "../src/llama-batch.h"
#include "../src/llama-model.h"
#include "../src/llama-kv-stream-model.h"
#include "../src/llama-kv-stream-logical-cache.h"
#include "../src/llama-memory-recurrent-spill.h"
#include "../src/llama-memory-bounded-kv.h"
#include "../src/llama-kv-cache-iswa.h"
#include "../src/llama-io.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "../ggml/src/ggml-backend-execution.h"
#include "../ggml/src/ggml-kv-stream-device.h"
#include "../ggml/src/ggml-kv-stream-copy.h"
#include "testing.h"
#include "kv-stream-bit-exact-test.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>

#ifdef KV_STREAM_TILE_NATIVE_TEST
bool kv_stream_test_is_sm61_only();
bool kv_stream_test_is_arch_only(int cc);
int kv_stream_test_compiled_cc(ggml_backend_t backend);
int kv_stream_test_override_cc(ggml_backend_t backend, int cc);
bool kv_stream_test_native_graphs_enabled();
#endif

using context_ptr = std::unique_ptr<llama_context,decltype(&llama_free)>;
using model_ptr = std::unique_ptr<llama_model,decltype(&llama_model_free)>;
#include "kv-stream-dflash-test.h"
static bool auxiliary_control = false, embedded_mtp_control = false, target_tg3_control = false, target_stream_tg4_control = false;
static bool suspend_control = false;
static bool suspend_resident_only = false;
static bool resume_control = false, resume_resident_only = false;
static bool target_with_stock_draft = false;
static bool f16_control = false, shared_budget_control = false;
static bool resident_control = false, trace_control = false;
static bool mtp_memory_probe = false, mtp_sustained_control = false, mtp_sustained_rs_control = false;
static bool target_drift_control = false, target_drift_rs_control = false;
static bool rollback_probe = false;
static bool wide_rollback_control = false;
static bool vision_mtp_handoff = false;
static bool dflash_vision_readiness = false;
static uint32_t dflash_vision_microbatch = 256;
static const char * dflash_vision_image = nullptr;
static const char * dflash_vision_control = nullptr;
static bool dflash_vision_writing = false, dflash_vision_native_check = false;
static uint32_t dflash_vision_depth = 3, dflash_vision_ngram = 0;
static bool serial_workspace_growth = false;
static bool mtp_bit_exact_control = false;
static const char * dflash_native_model = nullptr;
static bool dflash_shared_workspace_control = false;
static bool dflash_exact_control = false;
static bool dflash_short_control = false;
static bool dflash_independent_rows = false;
static bool dflash_independent_quotas = false;
static bool dflash_prefetch_hint = false;
static bool dflash_stream_control = false, dflash_full_prediction = false, dflash_native_f3 = false;
static bool dflash_independent_small_batch = false;
static bool dflash_mixed_switching = false;
static bool dflash_mixed_all_widths = false;
static bool dflash_mixed_recovery = false;
static bool test_cache_explicit = false;
static ggml_type test_cache_k = GGML_TYPE_Q8_0, test_cache_v = GGML_TYPE_Q4_0;
static size_t gemma_test_arena_size = size_t(1024)*1048576;
static bool gemma_test_arena_explicit = false;

// Keep instrumentation budgets explicit without changing ordinary model qualification.
static size_t gemma_test_arena_bytes(const char * text) {
    if (!text) return size_t(1024)*1048576;
    size_t mib=0;
    if (!*text) throw std::invalid_argument("missing Gemma test arena MiB");
    for (const char * digit=text;*digit;++digit) {
        if (*digit < '0' || *digit > '9' || mib > (4095u-size_t(*digit-'0'))/10)
            throw std::invalid_argument("Gemma test arena must be 1..4095 MiB");
        mib=mib*10+size_t(*digit-'0');
    }
    if (!mib) throw std::invalid_argument("Gemma test arena must be 1..4095 MiB");
    return mib*1048576;
}

static size_t gemma_image_test_arena_bytes(bool explicit_size, size_t bytes) {
    return explicit_size ? bytes : size_t(2048)*1048576;
}

static void gemma_image_test_outputs(llama_context_params & params,bool verify) {
    const auto limits=common_speculative_get_output_limits(int32_t(params.n_batch),1,verify ? 3 : 0);
    params.n_outputs_max=uint32_t(limits.total);
    params.n_outputs_max_per_seq=uint32_t(limits.per_seq);
}

// Qualify ordinary feature injection without a stream pool or a shared phase arena.
static void test_dflash_native(testing & t, llama_model * model) {
    auto cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = cp.n_ubatch = 256;
    cp.n_seq_max = 1;
    cp.n_rs_seq = 1;
    cp.type_k = GGML_TYPE_Q8_0;
    cp.type_v = GGML_TYPE_Q4_0;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    if (dflash_shared_workspace_control) {
        cp.shared_device_memory_bytes=512u*1024u*1024u;
        cp.n_rs_seq=3;
    }
    context_ptr target(llama_init_from_model(model, cp), llama_free);
    if (!t.assert_true(bool(target))) return;
    common_params params;
    params.n_ctx = 4096;
    params.n_batch = params.n_ubatch = 256;
    params.n_parallel = 1;
    params.n_gpu_layers = 999;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    params.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH};
    params.speculative.draft.n_max = 1;
    params.speculative.draft.n_min = 0;
    params.speculative.draft.p_min = 0;
    params.speculative.draft.n_gpu_layers = 999;
    params.speculative.draft.mparams.path = dflash_native_model;
    if (dflash_shared_workspace_control) {
        const auto revision = target->get_compute_memory()->graph_binding_revision();
        const auto features = target->get_cparams().embeddings_layer_inp;
        for (uint16_t length : {7, 48}) {
            auto invalid = params;
            invalid.speculative.types.push_back(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V);
            invalid.speculative.ngram_map_k4v.size_m = length;
            bool rejected = false;
            try {
                auto unexpected = common_speculative_init_from_params(invalid, model, target.get());
            } catch (const std::invalid_argument & error) {
                rejected = std::string(error.what()).find(length == 7 ? "rollback depth" : "size-m") != std::string::npos;
            }
            t.assert_true("invalid shared proposal limits reject before sidecar load and target mutation", rejected);
            t.assert_equal(revision, target->get_compute_memory()->graph_binding_revision());
            t.assert_true(features == target->get_cparams().embeddings_layer_inp);
        }
    }
    auto draft = common_speculative_init_from_params(params, model, target.get());
    if (!t.assert_true(bool(draft) && draft->context())) return;
    if (dflash_shared_workspace_control) {
        auto * target_owner=llama_context_compute_memory(target.get());
        auto * draft_owner=llama_context_compute_memory(draft->context());
        if (!t.assert_true(target_owner && draft_owner && draft_owner->borrows_serial_parent())) return;
        t.assert_true(!draft->context()->get_cparams().mtp_publish_host);
        t.assert_true(target_owner->shared_parent() == draft_owner->shared_parent());
        if (!t.assert_equal(uint32_t(params.speculative.draft.n_max + 1),
                draft->context()->get_cparams().serial_block_rows)) return;
    }
    auto * stock=dynamic_cast<llama_kv_cache_iswa *>(llama_get_memory(draft->context()));
    if (!t.assert_true(stock && stock->get_base()->get_layer_ids().empty())) return;
    if (dflash_shared_workspace_control) {
        auto * k=stock->get_swa()->get_k_storage(0);
        auto * parent=llama_context_compute_memory(target.get())->shared_parent();
        const auto base=uintptr_t(ggml_backend_buffer_get_base(parent));
        const auto bytes=ggml_backend_buffer_get_size(parent);
        if (!t.assert_true(ggml_backend_buffer_is_view(k->buffer) && uintptr_t(k->data) >= base &&
                uintptr_t(k->data)-base < bytes && ggml_nbytes(k) <= bytes-(uintptr_t(k->data)-base))) return;
        auto * hybrid=dynamic_cast<llama_memory_hybrid *>(llama_get_memory(target.get()));
        const auto * bank=hybrid->get_mem_recr()->spill_bank();
        auto * stage=bank ? bank->staging_tensor(0,true,1) : nullptr;
        if (!t.assert_true(stage && ggml_backend_buffer_is_view(stage->buffer) && uintptr_t(stage->data) >= base &&
                uintptr_t(stage->data)-base < bytes && ggml_nbytes(stage) <= bytes-(uintptr_t(stage->data)-base))) return;
    }
    llama_memory_bounded_kv_layout storage;
    if (!t.assert_true(stock->get_swa()->describe_bounded_storage(903,1,4096,256,storage))) return;
    t.assert_equal(uint32_t(2304),storage.allocation.capacity_rows);
    t.assert_equal(size_t(5),storage.layers.size());
    t.assert_equal(uint32_t(128),storage.cache.format.key_head_dim);
    t.assert_equal(uint32_t(8),storage.cache.format.kv_heads);
    t.assert_true(!stock->get_base()->describe_bounded_storage(904,1,4096,256,storage));
    auto cache_owner=stock->get_swa()->retain_bounded_storage(903,1,4096,256);
    if (!t.assert_true(bool(cache_owner))) return;
    auto source=cache_owner->read({903,1});
    if (!t.assert_true(bool(source))) return;
    t.assert_equal(storage.allocation.bytes,ggml_backend_buffer_get_size(source.buffer()));
    t.assert_equal(uint64_t(0),cache_owner->begin_write({903,1}));
    source.reset();
    params.speculative.draft.ctx_tgt = target.get();
    params.speculative.draft.ctx_dft = draft->context();
    common_speculative_ptr spec(common_speculative_init(params.speculative, 1));
    if (!t.assert_true(bool(spec))) return;
    const auto tokens = common_tokenize(target.get(), "An ordinary DFlash2 feature injection control. ", true);
    common_batch batch(target.get());
    for (int i = 0; i < 256; ++i) batch.add(tokens[size_t(i)%tokens.size()], i, 0, i == 255);
    if (dflash_shared_workspace_control) t.assert_true(!llama_serial_features_match(target.get(),batch.get()));
    if (!t.assert_equal(0, llama_process(target.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()))) return;
    if (dflash_shared_workspace_control) {
        t.assert_true(llama_serial_features_match(target.get(),batch.get()));
        common_batch wrong_position(target.get());
        for (int i=0;i<256;++i) wrong_position.add(tokens[size_t(i)%tokens.size()],i+1,0,i == 255);
        t.assert_true(!llama_serial_features_match(target.get(),wrong_position.get()));
    }
    const auto injection=cache_owner->begin_write({903,1});
    if (!t.assert_true(injection != 0)) return;
    const auto injected=common_speculative_process(spec.get(), batch);
    llama_synchronize(draft->context());
    t.assert_true(cache_owner->finish_write(injection,injected));
    if (!t.assert_true(injected)) return;
    t.assert_equal(llama_pos(255), llama_memory_seq_pos_max(llama_get_memory(draft->context()), 0));
    llama_tokens proposals;
    auto & dp = common_speculative_get_draft_params(spec.get(), 0);
    dp.drafting = true;
    dp.n_max = 1;
    dp.pos0 = 256;
    dp.id_last = tokens[0];
    dp.prompt = &tokens;
    dp.result = &proposals;
    const auto prediction=cache_owner->begin_write(cache_owner->identity());
    if (!t.assert_true(prediction != 0)) return;
    auto * draft_workspace=llama_context_compute_memory(draft->context());
    if (dflash_shared_workspace_control) t.assert_equal(int32_t(0),llama_set_adapter_cvec(draft->context(),nullptr,0,0,-1,-1));
    common_speculative_draft(spec.get());
    llama_synchronize(draft->context());
    t.assert_true(cache_owner->finish_write(prediction,proposals.size() == 1));
    t.assert_equal(size_t(1), proposals.size());
    t.assert_equal(COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH, common_speculative_get_last_type(spec.get(), 0));
    const auto prediction_stats = common_speculative_get_stats(spec.get());
    if (!t.assert_equal(size_t(1), prediction_stats.size())) return;
    t.assert_equal(size_t(1), prediction_stats[0].n_call_draft);
    t.assert_equal(size_t(1), prediction_stats[0].n_predict);
    dp.drafting = false;
    common_speculative_draft(spec.get());
    t.assert_equal(size_t(1), common_speculative_get_stats(spec.get())[0].n_predict);
    proposals.clear();
    dp.pos0 = llama_pos(llama_n_ctx_seq(draft->context()) - 1);
    dp.drafting = true;
    common_speculative_draft(spec.get());
    const auto boundary_stats = common_speculative_get_stats(spec.get());
    t.assert_true(proposals.empty());
    t.assert_equal(size_t(2), boundary_stats[0].n_call_draft);
    t.assert_equal(size_t(1), boundary_stats[0].n_predict);
    dp.pos0 = 256;
    source=cache_owner->read(cache_owner->identity());
    t.assert_true(bool(source));
    source.reset();
    const size_t state_bytes=llama_state_seq_get_size(draft->context(),0);
    std::vector<uint8_t> saved(state_bytes);
    if (!t.assert_true(state_bytes && llama_state_seq_get_data(draft->context(),saved.data(),saved.size(),0) == state_bytes)) return;
    if (dflash_shared_workspace_control) {
        t.assert_true(draft_workspace == llama_context_compute_memory(draft->context()));
        auto * parent=llama_context_compute_memory(target.get());
        auto * allocation=parent->shared_parent();
        t.assert_equal(int32_t(0),llama_set_adapter_cvec(target.get(),nullptr,0,0,-1,-1));
        batch.clear(); batch.add(tokens[0],256,0,true);
        llama_set_kv_stream_decode(target.get(),true);
        if (!t.assert_equal(0,llama_process(target.get(),LLAMA_PROCESS_TYPE_DECODE,batch.get()))) return;
        batch.clear(); for (int i=0;i<256;++i) batch.add(tokens[0],257+i,0,i == 255);
        llama_set_kv_stream_decode(target.get(),false);
        if (!t.assert_equal(0,llama_process(target.get(),LLAMA_PROCESS_TYPE_DECODE,batch.get()))) return;
        t.assert_true(parent == llama_context_compute_memory(target.get()) && allocation == parent->shared_parent());
        std::vector<uint8_t> retained(state_bytes);
        t.assert_equal(state_bytes,llama_state_seq_get_data(draft->context(),retained.data(),retained.size(),0));
        t.assert_true(retained == saved);
        common_batch oversized(draft->context());
        for (int i=0;i<9;++i) oversized.add(tokens[0],258+i,0,true);
        t.assert_true(llama_process(draft->context(),LLAMA_PROCESS_TYPE_DECODE,oversized.get()) != 0);
    }
    const auto restore=cache_owner->begin_write(cache_owner->identity());
    if (!t.assert_true(restore != 0)) return;
    const bool restored=llama_state_seq_set_data(draft->context(),saved.data(),saved.size(),0) == state_bytes;
    llama_synchronize(draft->context());
    t.assert_true(cache_owner->finish_write(restore,restored));
    t.assert_true(restored);
    std::vector<uint8_t> readback(state_bytes);
    t.assert_equal(state_bytes,llama_state_seq_get_data(draft->context(),readback.data(),readback.size(),0));
    t.assert_true(saved == readback);
    const auto reset=cache_owner->begin_write(cache_owner->identity());
    if (!t.assert_true(reset != 0)) return;
    llama_memory_clear(llama_get_memory(draft->context()),true);
    llama_synchronize(draft->context());
    t.assert_true(cache_owner->finish_write(reset,true));
    source=cache_owner->read(cache_owner->identity());
    auto full_params=cp;
    full_params.n_rs_seq=0;
    full_params.swa_full=true;
    full_params.ctx_other=target.get();
    full_params.shared_device_memory_bytes=0;
    context_ptr full(llama_init_from_model(draft->model(),full_params),llama_free);
    if (dflash_shared_workspace_control) {
        t.assert_true(!full);
        cache_owner.reset(); spec.reset(); draft.reset();
        t.assert_true(bool(source));
        return;
    }
    if (!t.assert_true(bool(full))) return;
    auto * full_stock=dynamic_cast<llama_kv_cache_iswa *>(llama_get_memory(full.get()));
    if (!t.assert_true(full_stock != nullptr)) return;
    t.assert_true(!full_stock->get_swa()->describe_bounded_storage(905,1,4096,256,storage));
    t.assert_true(!full_stock->get_swa()->retain_bounded_storage(905,1,4096,256));
    full.reset();
    cache_owner.reset(); spec.reset(); draft.reset();
    t.assert_true(bool(source) && source.layout()->allocation.bytes == storage.allocation.bytes);
}

// Fail one candidate view, then permit parent recovery.
struct serial_parent_view_failure {
    using factory = ggml_backend_buffer_t (*)(ggml_backend_buffer_t,size_t,size_t);
    inline static serial_parent_view_failure * active = nullptr;
    ggml_backend_buffer_t parent;
    factory original;
    size_t calls = 0;
    size_t fail_call;
    explicit serial_parent_view_failure(ggml_backend_buffer_t parent,size_t fail_call=2) :
        parent(parent),original(parent->view_buffer),fail_call(fail_call) {
        GGML_ASSERT(original && !active);
        active = this;
        parent->view_buffer = [](ggml_backend_buffer_t buffer,size_t offset,size_t bytes) {
            GGML_ASSERT(active && buffer == active->parent);
            if (++active->calls == active->fail_call) return static_cast<ggml_backend_buffer_t>(nullptr);
            return active->original(buffer,offset,bytes);
        };
    }
    ~serial_parent_view_failure() { parent->view_buffer = original; active = nullptr; }
};

struct joint_attention_probe {
    inline static joint_attention_probe * active=nullptr;
    std::map<ggml_backend_t,decltype(ggml_backend_i::graph_compute)> compute;
    int64_t head_dim;
    size_t joint=0;
    joint_attention_probe(llama_context & context,int64_t head_dim) : head_dim(head_dim) {
        GGML_ASSERT(!active);active=this;
        const auto scheduler=context.get_sched();
        for (int i=0;i<ggml_backend_sched_get_n_backends(scheduler);++i) {
            auto * backend=ggml_backend_sched_get_backend(scheduler,i);
            if (ggml_backend_dev_type(ggml_backend_get_device(backend))!=GGML_BACKEND_DEVICE_TYPE_GPU) continue;
            compute[backend]=backend->iface.graph_compute;
            backend->iface.graph_compute=[](ggml_backend_t backend,ggml_cgraph * graph) {
                for (int node=0;node<ggml_graph_n_nodes(graph);++node) {
                    const auto * op=ggml_graph_node(graph,node);
                    if (op->op==GGML_OP_FLASH_ATTN_EXT && op->src[0] &&
                            op->src[0]->ne[0]==active->head_dim && op->src[0]->ne[1]>1) ++active->joint;
                }
                return active->compute.at(backend)(backend,graph);
            };
        }
        GGML_ASSERT(!compute.empty());
    }
    ~joint_attention_probe() {
        for (const auto & item : compute) item.first->iface.graph_compute=item.second;
        active=nullptr;
    }
};

struct dflash_compute_failure {
    inline static dflash_compute_failure * active=nullptr;
    std::map<ggml_backend_t,decltype(ggml_backend_i::graph_compute)> compute;
    size_t calls=0,queued=0,injected=0,fail_call;
    ggml_status status;
    bool after_queue;
    dflash_compute_failure(llama_context * context,ggml_status status,size_t fail_call=1,bool after_queue=false) :
        fail_call(fail_call),status(status),after_queue(after_queue) {
        GGML_ASSERT(!active);active=this;
        const auto sched=context->get_sched();
        for (int i=0;i<ggml_backend_sched_get_n_backends(sched);++i) {
            auto * backend=ggml_backend_sched_get_backend(sched,i);
            if (ggml_backend_dev_type(ggml_backend_get_device(backend))!=GGML_BACKEND_DEVICE_TYPE_GPU) continue;
            compute[backend]=backend->iface.graph_compute;
            backend->iface.graph_compute=[](ggml_backend_t backend,ggml_cgraph * graph) {
                auto & fault=*active;
                const bool fail=++fault.calls==fault.fail_call;
                if (fail && !fault.after_queue) {++fault.injected;return fault.status;}
                const auto status=fault.compute.at(backend)(backend,graph);
                if (status==GGML_STATUS_SUCCESS) ++fault.queued;
                if (fail && status==GGML_STATUS_SUCCESS) {++fault.injected;return fault.status;}
                return status;
            };
        }
        GGML_ASSERT(!compute.empty());
    }
    ~dflash_compute_failure() {
        for (const auto & item:compute) item.first->iface.graph_compute=item.second;
        active=nullptr;
    }
};

static void test_dflash_failure_recovery(testing & t,llama_model * model,const char * sidecar,
        const char * export_native=nullptr,const char * import_native=nullptr,bool native_import=false) {
    if (!t.assert_true(!(export_native && import_native) && (!native_import || import_native))) return;
    auto mp=llama_model_default_params();mp.n_gpu_layers=999;
    model_ptr draft_model(llama_model_load_from_file(sidecar,mp),llama_model_free);
    if (!t.assert_true(bool(draft_model))) return;
    const size_t vocab=llama_vocab_n_tokens(llama_model_get_vocab(model));
    struct scenario { const char * phase;ggml_status status;size_t call;bool queued; };
    std::vector<scenario> cases;
    for (auto status:{GGML_STATUS_ABORTED,GGML_STATUS_ALLOC_FAILED,GGML_STATUS_FAILED}) {
        cases.push_back({"target",status,1,false});
        cases.push_back({"inject",status,1,false});
        cases.push_back({"inject-late",status,2,true});
        cases.push_back({"predict",status,1,true});
    }
    cases.push_back({"features",GGML_STATUS_SUCCESS,0,false});
    cases.push_back({"handoff",GGML_STATUS_SUCCESS,0,false});
    cases.push_back({"construct",GGML_STATUS_SUCCESS,0,false});
    kv_stream_bit_oracle oracle;
    const std::array<uint64_t,16> profile{1,4096,61,32,8,3,7,2,1,512,vocab,cases.size(),GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,0,0};
    if (import_native) {
        std::unique_ptr<FILE,kv_stream_exact_file_free> input(std::fopen(import_native,"rb"));
        if (!input) throw std::runtime_error("cannot open recovery native control");
        oracle.load_control(input.get(),profile);
    }
    for (bool streamed:{false,true}) {
        if (import_native && streamed==native_import) continue;
        if (streamed || import_native) oracle.compare();
        for (size_t index=0;index<cases.size();++index) {
            const auto & test=cases[index];
            const std::string name=std::to_string(index)+"/"+test.phase;
            std::fprintf(stderr,"DFlash recovery streamed=%d case=%s status=%d\n",int(streamed),name.c_str(),int(test.status));
            auto cp=llama_context_default_params();
            cp.n_ctx=4096;cp.n_batch=32;cp.n_ubatch=8;cp.n_seq_max=1;cp.n_rs_seq=7;
            cp.n_outputs_max=cp.n_outputs_max_per_seq=8;
            cp.type_k=GGML_TYPE_Q8_0;cp.type_v=GGML_TYPE_Q4_0;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
            cp.shared_device_memory_bytes=streamed ? size_t(512)*1048576 : 0;
            context_ptr target(nullptr,llama_free),child(nullptr,llama_free);
            common_speculative_ptr spec;
            const auto create=[&] {
                target.reset(llama_init_from_model(model,cp));
                if (!t.assert_true(bool(target))) return false;
                auto dp=dflash_independent_child_params(cp,target.get());dp.n_batch=32;dp.n_ubatch=8;
                child.reset(streamed ? llama_init_from_model(draft_model.get(),dp,4) : llama_init_from_model(draft_model.get(),dp));
                if (!t.assert_true(bool(child))) return false;
                common_params_speculative params;
                params.types={COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V,COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH};
                params.draft.ctx_tgt=target.get();params.draft.ctx_dft=child.get();
                params.draft.n_max=3;params.draft.n_min=0;params.draft.p_min=0;
                params.ngram_map_k4v.size_n=2;params.ngram_map_k4v.size_m=7;params.ngram_map_k4v.min_hits=1;
                spec.reset(common_speculative_init(params,1));
                return t.assert_true(bool(spec));
            };
            if (!create()) return;
            llama_tokens history;
            const auto make_batch=[&](const llama_tokens & tokens,size_t first,bool all) {
                common_batch batch(target.get());
                for (size_t i=0;i<tokens.size();++i) batch.add(tokens[i],llama_pos(first+i),0,all || i+1==tokens.size());
                return batch;
            };
            const auto run=[&](const llama_tokens & tokens,size_t first,bool decode) {
                auto batch=make_batch(tokens,first,decode);
                llama_set_kv_stream_decode(target.get(),decode);
                if (!t.assert_equal(0,llama_process(target.get(),LLAMA_PROCESS_TYPE_DECODE,batch.get())) ||
                        !t.assert_true(common_speculative_process(spec.get(),batch))) return false;
                llama_synchronize(target.get());llama_synchronize(child.get());
                return true;
            };
            const llama_tokens pattern{99,100,101,102,103,104,105,106,107,108};
            while (history.size()<61) {
                llama_tokens tokens(std::min(size_t(8),61-history.size()));
                for (size_t i=0;i<tokens.size();++i) tokens[i]=pattern[(history.size()+i)%pattern.size()];
                if (!run(tokens,history.size(),false)) return;
                history.insert(history.end(),tokens.begin(),tokens.end());
            }
            const auto save=[&](llama_context * context) {
                std::vector<uint8_t> state(llama_state_seq_get_size(context,0));
                if (!t.assert_equal(state.size(),llama_state_seq_get_data(context,state.data(),state.size(),0)))
                    throw std::runtime_error("incomplete recovery snapshot");
                return state;
            };
            const auto target_state=save(target.get()),draft_state=save(child.get());
            common_speculative_begin(spec.get(),0,history);
            const auto predict=[&](llama_token anchor) {
                llama_tokens result;
                auto & request=common_speculative_get_draft_params(spec.get(),0);
                request.drafting=true;request.n_max=7;request.pos0=llama_pos(history.size());request.id_last=anchor;
                request.prompt=&history;request.result=&result;
                common_speculative_draft(spec.get());llama_synchronize(child.get());
                request.prompt=nullptr;request.result=nullptr;
                return result;
            };
            if (!t.assert_true(predict(100)==llama_tokens({101,102,103,104,105,106,107}))) return;
            llama_tokens extra(16,99);
            auto pending=make_batch(extra,history.size(),false);
            if (std::string(test.phase)=="target") {
                dflash_compute_failure fault(target.get(),test.status,test.call,test.queued);
                const auto rc=llama_process(target.get(),LLAMA_PROCESS_TYPE_DECODE,pending.get());
                llama_synchronize(target.get());
                if (!t.assert_equal(test.status==GGML_STATUS_ABORTED ? 2 : test.status==GGML_STATUS_ALLOC_FAILED ? -2 : -3,rc) ||
                        !t.assert_equal(test.call,fault.calls)) return;
                if (streamed) {
                    if (!t.assert_true("failed target cannot publish features",!common_speculative_process(spec.get(),pending))) return;
                    t.assert_true(save(child.get())==draft_state);
                }
            } else if (std::string(test.phase).find("inject")==0) {
                if (!t.assert_equal(0,llama_process(target.get(),LLAMA_PROCESS_TYPE_DECODE,pending.get()))) return;
                dflash_compute_failure fault(child.get(),test.status,test.call,test.queued);
                if (!t.assert_true(!common_speculative_process(spec.get(),pending))) return;
                llama_synchronize(child.get());
                if (!t.assert_equal(test.call,fault.calls)) return;
                t.assert_equal(test.queued ? test.call : test.call-1,fault.queued);
            } else if (std::string(test.phase)=="predict") {
                dflash_compute_failure fault(child.get(),test.status,test.call,test.queued);
                if (!t.assert_true("failed prediction supplies no draft",predict(5000).empty()) ||
                        !t.assert_equal(test.call,fault.calls) || !t.assert_equal(test.call,fault.queued)) return;
                t.assert_equal(COMMON_SPECULATIVE_TYPE_NONE,common_speculative_get_last_type(spec.get(),0));
            } else if (streamed && std::string(test.phase)=="features") {
                if (!t.assert_equal(0,llama_process(target.get(),LLAMA_PROCESS_TYPE_DECODE,pending.get()))) return;
                const auto layer=llama_model_target_layer_ids(draft_model.get())[0];
                llama_set_embeddings_layer_inp(target.get(),uint32_t(layer),false);
                if (!t.assert_true("missing extraction rejects without child writes",!common_speculative_process(spec.get(),pending))) return;
                t.assert_true(save(child.get())==draft_state);
            } else if (streamed && std::string(test.phase)=="handoff") {
                auto * owner=const_cast<llama_context_memory *>(target->get_compute_memory());
                if (!t.assert_true("target owns the handoff before the injected view failure",owner->prepare_serial_target())) return;
                if (!t.assert_true(owner->retire_graph())) return;
                ggml_backend_sched_reset(target->get_sched());
                const auto before=owner->text_phase();
                serial_parent_view_failure fault(owner->shared_parent(),1);
                if (!t.assert_true(owner->signal_text_phase({llama_memory_text_phase::decode,8,true,true,false}).status==
                        llama_memory_text_phase_status::transition_failed) || !t.assert_true(fault.calls>0)) return;
                t.assert_true(owner->valid());t.assert_equal(before.revision,owner->text_phase().revision);
            } else if (streamed && std::string(test.phase)=="construct") {
                spec.reset();child.reset();
                auto dp=dflash_independent_child_params(cp,target.get());
                serial_parent_view_failure fault(target->get_compute_memory()->shared_parent(),1);
                context_ptr rejected(llama_init_from_model(draft_model.get(),dp,4),llama_free);
                if (!t.assert_true(!rejected) || !t.assert_true(fault.calls>0)) return;
                t.assert_true(target->get_compute_memory()->valid());
            }
            spec.reset();child.reset();target.reset();
            if (!create()) return;
            if (!t.assert_equal(target_state.size(),llama_state_seq_set_data(target.get(),target_state.data(),target_state.size(),0)) ||
                    !t.assert_equal(draft_state.size(),llama_state_seq_set_data(child.get(),draft_state.data(),draft_state.size(),0))) return;
            common_speculative_begin(spec.get(),0,history);
            t.assert_equal(COMMON_SPECULATIVE_TYPE_NONE,common_speculative_get_last_type(spec.get(),0));
            auto proposed=predict(100);
            if (!t.assert_true("reconstruction rebuilds discarded ngram state",proposed==llama_tokens({101,102,103,104,105,106,107}))) return;
            const size_t first=history.size();
            llama_tokens verified{100};verified.insert(verified.end(),proposed.begin(),proposed.end());
            if (!run(verified,first,true)) return;
            for (size_t row=0;row<verified.size();++row) {
                const auto * logits=llama_get_logits_ith(target.get(),int(row));
                if (!t.assert_true(logits && std::all_of(logits,logits+vocab,[](float f){return std::isfinite(f);}))) return;
                oracle.bytes(name+"/logits/"+std::to_string(row),logits,vocab*sizeof(float));
            }
            for (uint32_t layer=0;layer<llama_model_target_layer_ids_n(draft_model.get());++layer)
                oracle.bytes(name+"/features/"+std::to_string(layer),
                    llama_get_embeddings_layer_inp(target.get(),uint32_t(llama_model_target_layer_ids(draft_model.get())[layer])),
                    verified.size()*size_t(llama_model_n_embd(model))*sizeof(float));
            const size_t kept=first+4;
            if (!t.assert_true(llama_memory_seq_rm(llama_get_memory(target.get()),0,llama_pos(kept),-1) &&
                    llama_memory_seq_rm(llama_get_memory(child.get()),0,llama_pos(kept),-1))) return;
            common_speculative_accept(spec.get(),0,3);
            history.insert(history.end(),verified.begin(),verified.begin()+4);
            auto * hybrid=dynamic_cast<llama_memory_hybrid *>(llama_get_memory(target.get()));
            auto * local=dynamic_cast<llama_kv_cache_iswa *>(llama_get_memory(child.get()));
            if (!t.assert_true(hybrid && local)) return;
            kv_stream_exact_kv(oracle,name+"/target",hybrid->get_mem_attn(),0,kept);
            kv_stream_exact_kv(oracle,name+"/draft",local->get_swa(),0,kept);
            kv_stream_exact_state recurrent;hybrid->get_mem_recr()->state_write(recurrent,0,0);
            recurrent.compare_with(oracle,name+"/recurrent");
            proposed=predict(5000);
            if (!t.assert_equal(size_t(3),proposed.size())) return;
            oracle.bytes(name+"/neural",proposed.data(),proposed.size()*sizeof(llama_token));
            const auto lattice=dflash_exact_lattice(llama_get_embeddings_nextn(child.get()),4,
                size_t(llama_model_n_embd(draft_model.get())),size_t(llama_model_dflash_selector_top_k(draft_model.get())),vocab);
            oracle.bytes(name+"/selector",lattice.data(),lattice.size()*sizeof(uint32_t));
            if (streamed) t.assert_true(child->get_compute_memory()->borrows_serial_parent() &&
                child->get_compute_memory()->shared_parent()==target->get_compute_memory()->shared_parent());
            t.assert_equal(llama_pos(kept-1),llama_memory_seq_pos_max(llama_get_memory(target.get()),0));
        }
        if (streamed || import_native) oracle.finish();
        if (!streamed && export_native) {
            std::unique_ptr<FILE,kv_stream_exact_file_free> output(std::fopen(export_native,"wbx"));
            if (!output) throw std::runtime_error("cannot export recovery native control");
            oracle.save_control(output.get(),profile);
        }
        t.out << "DFlash recovery streamed=" << streamed << " cases=" << cases.size() << " records=" << oracle.records()
              << " bytes=" << oracle.byte_count() << '\n';
    }
}

struct image_writer : llama_io_write_i {
    std::vector<uint8_t> bytes;
    void write(const void * data,size_t size) override {
        const auto * begin=static_cast<const uint8_t *>(data);
        bytes.insert(bytes.end(),begin,begin+size);
    }
    void write_tensor(ggml_tensor * tensor,size_t offset,size_t size) override {
        const auto start=bytes.size(); bytes.resize(start+size);
        ggml_backend_tensor_get(tensor,bytes.data()+start,offset,size);
    }
    size_t n_bytes() override { return bytes.size(); }
};
struct image_reader : llama_io_read_i {
    const std::vector<uint8_t> & bytes;
    size_t cursor=0;
    explicit image_reader(const std::vector<uint8_t> & bytes) : bytes(bytes) {}
    void read(void * data,size_t size) override {
        if (size > bytes.size()-cursor) throw std::runtime_error("truncated lifecycle fixture");
        std::memcpy(data,bytes.data()+cursor,size); cursor+=size;
    }
    void read_tensor(ggml_tensor * tensor,size_t offset,size_t size) override {
        if (size > bytes.size()-cursor) throw std::runtime_error("truncated lifecycle tensor");
        ggml_backend_tensor_set(tensor,bytes.data()+cursor,offset,size); cursor+=size;
    }
    size_t n_bytes() override { return cursor; }
};

struct recurrent_snapshot : llama_io_write_i {
    std::vector<uint8_t> metadata;
    std::vector<std::vector<float>> tensors;
    size_t count = 0;
    void write(const void * p,size_t n) override {
        auto * bytes = static_cast<const uint8_t *>(p); metadata.insert(metadata.end(),bytes,bytes+n); count += n;
    }
    void write_tensor(ggml_tensor * tensor,size_t offset,size_t bytes) override {
        GGML_ASSERT(tensor->type == GGML_TYPE_F32 && bytes%sizeof(float) == 0);
        tensors.emplace_back(bytes/sizeof(float));
        ggml_backend_tensor_get(tensor,tensors.back().data(),offset,bytes); count += bytes;
    }
    size_t n_bytes() override { return count; }
};
struct run_result {
    std::vector<std::vector<float>> logits;
    std::vector<llama_token> continuation;
    recurrent_snapshot recurrent;
    std::vector<std::pair<std::string,std::vector<float>>> trace;
};

// Compare matching target forwards, selected rollback state and replay without allocating a draft context.
struct native_state_copy_fault {
    inline static native_state_copy_fault * active=nullptr;
    ggml_backend_buffer_ptr buffer;
    decltype(ggml_backend_buffer_i::get_tensor) get;
    decltype(ggml_backend_buffer_i::set_tensor) set;
    size_t calls=0;
    native_state_copy_fault(ggml_backend_buffer_t source,bool upload) : buffer(ggml_backend_buffer_retain(source)),
        get(source->iface.get_tensor),set(source->iface.set_tensor) {
        GGML_ASSERT(!active);active=this;
        if (upload) source->iface.set_tensor=[](ggml_backend_buffer_t b,ggml_tensor * tensor,const void * data,size_t offset,size_t bytes) {
            active->set(b,tensor,data,offset,bytes);
            if (++active->calls==2) throw std::runtime_error("after native state upload");
        };
        else source->iface.get_tensor=[](ggml_backend_buffer_t b,const ggml_tensor * tensor,void * data,size_t offset,size_t bytes) {
            active->get(b,tensor,data,offset,bytes);
            if (++active->calls==2) throw std::runtime_error("after native state download");
        };
    }
    ~native_state_copy_fault() { buffer->iface.get_tensor=get;buffer->iface.set_tensor=set;active=nullptr; }
};

struct native_state_export_probe : llama_io_write_i {
    size_t calls=0;
    void write(const void *,size_t) override { ++calls;throw std::runtime_error("state export started"); }
    void write_tensor(ggml_tensor *,size_t,size_t) override { ++calls;throw std::runtime_error("state export started"); }
    size_t n_bytes() override { return 0; }
};

static void test_selected_current_parking(testing & t,llama_model * model) {
    if (!t.assert_true(model && (model->arch==LLM_ARCH_QWEN35 || model->arch==LLM_ARCH_GEMMA4))) return;
    const bool gemma=model->arch==LLM_ARCH_GEMMA4;
    const int32_t prompt=gemma ? 1333 : 64;
    kv_stream_bit_oracle oracle;
    for (bool moved : {false,true}) {
        if (moved) oracle.compare();
        auto cp=llama_context_default_params();
        cp.n_ctx=4096;cp.n_batch=cp.n_ubatch=256;cp.n_seq_max=1;
        cp.n_threads=cp.n_threads_batch=4;cp.n_rs_seq=0;
        cp.type_k=GGML_TYPE_Q8_0;cp.type_v=GGML_TYPE_Q4_0;
        cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.shared_device_memory_bytes=size_t(1024)*1048576;
        cp.swa_full=false;cp.pooling_type=LLAMA_POOLING_TYPE_NONE;
        std::unique_ptr<llama_context_memory_loan> current;
        std::unique_ptr<ggml_backend_memory_arena,decltype(&ggml_backend_memory_arena_free)> local_arena(nullptr,ggml_backend_memory_arena_free);
        std::unique_ptr<ggml_backend_memory_lease,decltype(&ggml_backend_memory_lease_free)> local_lease(nullptr,ggml_backend_memory_lease_free);
        context_ptr context(llama_init_from_model(model,cp),llama_free);
        if (!t.assert_true(bool(context))) return;
        auto * hybrid=dynamic_cast<llama_memory_hybrid *>(llama_get_memory(context.get()));
        auto * owner=llama_context_compute_memory(context.get());
        auto * iswa=dynamic_cast<llama_kv_cache_iswa *>(llama_get_memory(context.get()));
        if (!t.assert_true((gemma ? iswa!=nullptr : hybrid!=nullptr) && owner && owner->shares_kv_memory())) return;
        auto * recurrent=hybrid ? hybrid->get_mem_recr() : nullptr;
        auto * local=iswa ? iswa->get_swa() : nullptr;
        auto * stream=(gemma ? iswa->get_base() : hybrid->get_mem_attn())->get_kv_stream();
        llama_set_embeddings(context.get(),true);
        if (!t.assert_true(stream!=nullptr)) return;
        common_batch batch(context.get());
        for (int32_t first=0;first<prompt;) {
            const int32_t rows=std::min<int32_t>(256,prompt-first);
            batch.clear();
            for (int32_t row=0;row<rows;++row) batch.add(llama_token(10+(first+row)%64),first+row,0,row==rows-1);
            if (!t.assert_equal(0,llama_process(context.get(),LLAMA_PROCESS_TYPE_DECODE,batch.get()))) return;
            first+=rows;
        }
        llama_set_kv_stream_decode(context.get(),true);
        for (int32_t step=0;step<8;++step) {
            if (moved && (step==1 || step==4)) {
                llama_synchronize(context.get());
                if (!t.assert_true(owner->prepare_serial_target() && owner->retire_graph())) return;
                const auto tokens=stream->tokens();
                llama_memory_cache_requirement before_source;
                if (!t.assert_true(stream->read_source(before_source))) return;
                if (gemma) {
                    if (!local_arena) {
                        auto * buffer=local->get_k_storage(0)->buffer;
                        const auto bytes=ggml_backend_buffer_get_size(buffer);
                        local_arena.reset(ggml_backend_memory_arena_new_from_buffer(buffer));
                        if (!t.assert_true(local_arena && ggml_backend_memory_arena_begin(local_arena.get(),0) &&
                                ggml_backend_memory_arena_reserve_at(local_arena.get(),91,0,bytes,
                                    ggml_backend_buft_get_alignment(ggml_backend_buffer_get_type(buffer)),
                                    GGML_BACKEND_MEMORY_REGION_PERSISTENT,nullptr) && ggml_backend_memory_arena_commit(local_arena.get()))) return;
                        local_lease.reset(ggml_backend_memory_arena_acquire(local_arena.get(),91));
                    }
                    if (step==1) {
                        native_state_copy_fault fault(local->get_k_storage(0)->buffer,false);
                        t.assert_true(!local->park_bounded_storage());
                        t.assert_equal(size_t(2),fault.calls);
                        t.assert_true(!local->bounded_storage_pending());
                    }
                    if (!t.assert_true(local_lease && local->park_bounded_storage() && local->bounded_storage_pending())) return;
                    native_state_export_probe probe;
                    bool rejected=false;
                    try { local->state_write(probe,0,0); } catch (const std::runtime_error &) { rejected=true; }
                    if (!t.assert_true("parked bounded state export rejects before touching the output",rejected && probe.calls==0)) return;
                    t.assert_true(!local->bind_bounded_storage(local_lease.get()));
                    ggml_backend_buffer_clear(ggml_backend_memory_lease_buffer(local_lease.get()),0xa5);
                    if (step==1) {
                        native_state_copy_fault fault(ggml_backend_memory_lease_buffer(local_lease.get()),true);
                        t.assert_true(!local->restore_bounded_storage(local_lease.get()));
                        t.assert_equal(size_t(2),fault.calls);
                        t.assert_true(local->bounded_storage_pending() && !local->get_k_storage(0)->data);
                    }
                    if (!t.assert_true(local->restore_bounded_storage(local_lease.get()))) return;
                } else {
                recurrent_snapshot before;recurrent->state_write(before,0,0);
                if (step==1 && !t.assert_true(recurrent->enable_current_storage_parking(
                        ggml_backend_buffer_get_type(owner->shared_parent())))) return;
                ggml_backend_buffer_type_t type=nullptr;size_t bytes=0;
                if (!t.assert_true(recurrent->current_storage_requirements(type,bytes) && recurrent->park_current_storage())) return;
                if (!current) current=owner->reserve_staging(bytes);
                if (!t.assert_true(current && recurrent->restore_current_storage(current->lease()))) return;
                recurrent_snapshot after;recurrent->state_write(after,0,0);
                if (!t.assert_true(before.metadata==after.metadata && before.tensors.size()==after.tensors.size())) return;
                for (size_t layer=0;layer<before.tensors.size();++layer)
                    t.assert_true(kv_stream_bytes_equal(before.tensors[layer].data(),before.tensors[layer].size()*sizeof(float),
                        after.tensors[layer].data(),after.tensors[layer].size()*sizeof(float)));
                }
                llama_memory_cache_requirement after_source;
                if (!t.assert_true(stream->read_source(after_source))) return;
                t.assert_equal(before_source.identity.id,after_source.identity.id);
                t.assert_equal(before_source.identity.generation,after_source.identity.generation);
                t.assert_equal(before_source.history_rows,after_source.history_rows);
                t.assert_equal(tokens,stream->tokens());
                t.assert_equal(cp.shared_device_memory_bytes,owner->shared_parent_capacity());
            }
            batch.clear();batch.add(llama_token(80+step),prompt+step,0,true);
            if (!t.assert_equal(0,llama_process(context.get(),LLAMA_PROCESS_TYPE_DECODE,batch.get()))) return;
            llama_synchronize(context.get());
            const auto name="current/"+std::to_string(step);
            oracle.bytes(name+"/logits",llama_get_logits_ith(context.get(),0),
                size_t(llama_vocab_n_tokens(llama_model_get_vocab(model)))*sizeof(float));
            oracle.bytes(name+"/hidden",llama_get_embeddings_ith(context.get(),0),size_t(model->hparams.n_embd_out())*sizeof(float));
        }
        recurrent_snapshot final;
        if (recurrent) recurrent->state_write(final,0,0);
        oracle.bytes("current/final-metadata",final.metadata.data(),final.metadata.size());
        for (size_t layer=0;layer<final.tensors.size();++layer)
            oracle.bytes("current/final/"+std::to_string(layer),final.tensors[layer].data(),final.tensors[layer].size()*sizeof(float));
        if (moved) oracle.finish();
    }
}

// Real model geometry, native device state, and CPU-mapped weights; this fixture performs no neural computation.
static void test_qwen_current_cache_parking(testing & t,llama_model * model) {
    if (!t.assert_true(model && model->arch==LLM_ARCH_QWEN35)) return;
    for (const auto & shape : std::vector<std::array<uint32_t,3>>{{1,1,0},{2,2,0},{1,1,3},{1,2,0}}) {
        const auto selected=shape[0],logical=shape[1],depth=shape[2];
        const bool projected=selected!=logical;
        llama_memory_recurrent memory(*model,GGML_TYPE_F32,GGML_TYPE_F32,true,selected,logical,depth,
            [&](int32_t layer) { return model->hparams.is_recr(layer); },depth!=0,true);
        if (projected && !t.assert_true(memory.configure_selected_sequences({1}))) return;
        ggml_backend_buffer_type_t type=nullptr;size_t bytes=0;
        if (!t.assert_true(memory.current_storage_requirements(type,bytes))) return;
        const auto alignment=ggml_backend_buft_get_alignment(type);
        size_t stage_bytes=0;
        if (depth) {
            std::vector<llama_recurrent_spill_layer> specs(memory.r_l.size());
            for (size_t layer=0;layer<specs.size();++layer) if (memory.r_l[layer]) specs[layer]={
                GGML_TYPE_F32,GGML_TYPE_F32,uint64_t(memory.r_l[layer]->ne[0]),uint64_t(memory.s_l[layer]->ne[0]),true};
            llama_recurrent_spill_layout rows;llama_recurrent_stage_layout stage;
            if (!t.assert_true(llama_recurrent_spill_layout_make(specs,selected,depth,rows) &&
                    llama_recurrent_stage_layout_make(rows,2,alignment,stage))) return;
            stage_bytes=stage.total_bytes;
            t.assert_true(memory.spill_bank()->publication_stage_parked() && memory.spill_bank()->staged_device_bytes()==0);
        }
        constexpr size_t capacity=size_t(1024)*1048576;
        if (!t.assert_true(bytes<=capacity && stage_bytes<=capacity-bytes)) return;
        std::unique_ptr<ggml_backend_memory_arena,decltype(&ggml_backend_memory_arena_free)> arena(
            ggml_backend_memory_arena_new(type,capacity),ggml_backend_memory_arena_free);
        if (!t.assert_true(bool(arena))) return;
        const auto plan=[&](uint64_t id) {
            return ggml_backend_memory_arena_begin(arena.get(),0) &&
                ggml_backend_memory_arena_reserve_at(arena.get(),id,0,bytes,alignment,GGML_BACKEND_MEMORY_REGION_PERSISTENT,nullptr) &&
                (!depth || ggml_backend_memory_arena_reserve_at(arena.get(),id+1,bytes,stage_bytes,alignment,GGML_BACKEND_MEMORY_REGION_PERSISTENT,nullptr)) &&
                ggml_backend_memory_arena_commit(arena.get());
        };
        if (!t.assert_true(plan(101))) return;
        using lease_ptr=std::unique_ptr<ggml_backend_memory_lease,decltype(&ggml_backend_memory_lease_free)>;
        lease_ptr current(ggml_backend_memory_arena_acquire(arena.get(),101),ggml_backend_memory_lease_free),
            publication(depth ? ggml_backend_memory_arena_acquire(arena.get(),102) : nullptr,ggml_backend_memory_lease_free);
        if (!t.assert_true(current && memory.bind_current_storage(current.get()) && (!depth || memory.bind_spill_stage(publication.get())))) return;
        std::vector<std::vector<float>> expected;
        for (size_t layer=0;layer<memory.r_l.size();++layer) for (auto * tensor : {memory.r_l[layer],memory.s_l[layer]}) {
            if (!tensor) continue;
            expected.emplace_back(ggml_nelements(tensor));
            auto & row=expected.back();
            for (size_t i=0;i<row.size();++i) row[i]=float(int(i%97)-48)*0.125f+float(layer);
            row[0]=-0.0f;
            ggml_backend_tensor_set(tensor,row.data(),0,ggml_nbytes(tensor));
        }
        if (projected) {
            llama_batch_allocr allocator(1);auto batch=allocator.ubatch_reserve(1,1);llama_seq_id id=1;
            batch.pos[0]=72;batch.n_seq_id[0]=1;batch.seq_id[0]=&id;
            if (!t.assert_true(memory.find_slot(batch))) return;
            t.assert_equal(llama_pos(-1),memory.seq_pos_max(0));
            t.assert_equal(llama_pos(72),memory.seq_pos_max(1));
        } else {
            memory.used=selected;memory.head=selected-1;memory.n=selected;
            for (uint32_t seq=0;seq<selected;++seq) {
                memory.cells[seq].pos=llama_pos(71+seq);memory.cells[seq].tail=seq;
                memory.cells[seq].src=int32_t(seq);memory.cells[seq].src0=int32_t(seq);memory.cells[seq].seq_id.insert(seq);
            }
        }
        if (!t.assert_true(memory.park_current_storage() && memory.current_storage_pending())) return;
        current.reset();publication.reset();
        if (!t.assert_true(plan(111))) return;
        current.reset(ggml_backend_memory_arena_acquire(arena.get(),111));
        if (depth) publication.reset(ggml_backend_memory_arena_acquire(arena.get(),112));
        if (!t.assert_true(bool(current))) return;
        ggml_backend_buffer_clear(ggml_backend_memory_arena_parent(arena.get()),0xa5);
        {
            native_state_copy_fault fault(ggml_backend_memory_lease_buffer(current.get()),true);
            t.assert_true(!memory.restore_current_storage(current.get(),publication.get()));
            t.assert_equal(size_t(2),fault.calls);
            t.assert_true(memory.current_storage_parked() && memory.current_storage_pending());
        }
        if (!t.assert_true(memory.restore_current_storage(current.get(),publication.get()))) return;
        size_t index=0;
        for (size_t layer=0;layer<memory.r_l.size();++layer) for (auto * tensor : {memory.r_l[layer],memory.s_l[layer]}) {
            if (!tensor) continue;
            std::vector<float> actual(expected[index].size());
            ggml_backend_tensor_get(tensor,actual.data(),0,ggml_nbytes(tensor));
            t.assert_true(kv_stream_bytes_equal(expected[index].data(),expected[index].size()*sizeof(float),actual.data(),ggml_nbytes(tensor)));
            ++index;
        }
        t.assert_equal(selected,memory.used);t.assert_equal(selected-1,memory.head);
        for (uint32_t seq=0;seq<selected;++seq) {
            const auto id=projected ? 1u : seq;
            t.assert_true(memory.cells[seq].has_seq_id(id) && memory.cells[seq].pos==llama_pos(71+id));
            t.assert_equal(llama_pos(71+id),memory.seq_pos_max(llama_seq_id(id)));
            if (!projected) t.assert_equal(int32_t(seq),memory.cells[seq].tail);
        }
        t.assert_true(!memory.current_storage_pending() && memory.prepare({}));
        if (depth) t.assert_true(memory.staged_publication_enabled() && !memory.spill_bank()->publication_stage_parked());
        t.out << "Qwen native current selected=" << selected << " logical=" << logical << " rollback=" << depth << " bytes=" << bytes
              << " publication=" << stage_bytes << " parent=" << capacity << std::endl;
    }
}

static void test_wide_target_rollback(testing & t,llama_model * model) {
    kv_stream_exact_config cfg;
    std::ifstream input(cfg.corpus,std::ios::binary);
    if (!t.assert_true(bool(input))) return;
    const std::string article((std::istreambuf_iterator<char>(input)),std::istreambuf_iterator<char>());
    const auto * vocabulary=llama_model_get_vocab(model);
    const int count=-llama_tokenize(vocabulary,article.data(),int(article.size()),nullptr,0,false,false);
    if (!t.assert_true(count >= 16381)) return;
    std::vector<llama_token> corpus(size_t(count),0);
    if (!t.assert_equal(count,llama_tokenize(vocabulary,article.data(),int(article.size()),corpus.data(),count,false,false))) return;
    const size_t vocab_size=llama_vocab_n_tokens(vocabulary);
    for (uint32_t depth : {1u,3u,7u}) for (size_t prefix : {249u,505u,2041u,16381u}) {
        kv_stream_bit_oracle oracle;
        for (bool streamed : {false,true}) {
            if (streamed) oracle.compare();
            auto params=llama_context_default_params();
            params.n_ctx=uint32_t((prefix+6*(depth+2)+255)/256*256);
            params.n_batch=params.n_ubatch=256; params.n_seq_max=1; params.n_rs_seq=depth;
            params.n_threads=params.n_threads_batch=8;
            params.n_outputs_max=params.n_outputs_max_per_seq=depth+1;
            params.type_k=GGML_TYPE_Q8_0; params.type_v=GGML_TYPE_Q4_0;
            params.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
            // Include a real phase-arena handoff while other streamed cases force a small physical ring.
            if (streamed && depth == 7 && prefix == 505) params.shared_device_memory_bytes=size_t(2240)*1048576;
            else params.kv_stream_pool_bytes=streamed ? 16*1048576 : 0;
            context_ptr ctx(llama_init_from_model(model,params),llama_free);
            if (!t.assert_true(bool(ctx))) return;
            auto * hybrid=dynamic_cast<llama_memory_hybrid *>(llama_get_memory(ctx.get()));
            if (!t.assert_true(hybrid != nullptr)) return;
            auto * recurrent=hybrid->get_mem_recr();
            auto * kv=hybrid->get_mem_attn();
            auto * stream=kv->get_kv_stream();
            t.assert_equal(streamed,recurrent->spill_enabled());
            if (streamed) {
                if (!t.assert_true(stream && !stream->auxiliary_cache())) return;
                t.out << "rollback D=" << depth << " prefix=" << prefix << " stage MiB=" <<
                    double(recurrent->spill_bank()->staged_device_bytes())/1048576 << '\n';
            }
            const auto drain=[&] {
                llama_synchronize(ctx.get());
                if (!recurrent->complete_spill() || !recurrent->complete_restore_spill())
                    throw std::runtime_error("failed to drain rollback test");
            };
            const auto state=[&](const std::string & name) {
                drain();
                kv_stream_exact_state snapshot;
                recurrent->state_write(snapshot,0,0);
                snapshot.compare_with(oracle,name+"/state");
            };
            const auto submit=[&](size_t first,size_t tokens,bool decode,bool proposals=false) {
                common_batch batch(ctx.get());
                for (size_t i=0;i<tokens;++i) {
                    // A rejected proposal differs from its correction, exercising actual KV overwrite.
                    const auto token=proposals && i ? llama_token((size_t(corpus[first+i])+1+i)%vocab_size) : corpus[first+i];
                    batch.add(token,llama_pos(first+i),0,decode || i+1==tokens);
                }
                llama_set_kv_stream_decode(ctx.get(),decode);
                return llama_process(ctx.get(),LLAMA_PROCESS_TYPE_DECODE,batch.get());
            };
            for (size_t first=0;first<prefix;first+=256)
                if (!t.assert_equal(0,submit(first,std::min(size_t(256),prefix-first),false))) return;
            state("prefill");
            kv_stream_exact_kv(oracle,"prefill/KV",kv,0,prefix);
            size_t position=prefix;
            const uint32_t short_widths[6]={8,5,7,6,2,4};
            for (uint32_t round=0;round<6;++round) {
                const auto name="round/"+std::to_string(round);
                const size_t width=depth == 7 && prefix == 2041 ? short_widths[round] : depth+1;
                if (!t.assert_equal(0,submit(position,width,true,true))) return;
                drain();
                for (size_t q=0;q<width;++q) {
                    const auto * logits=llama_get_logits_ith(ctx.get(),int32_t(q));
                    if (!t.assert_true(logits != nullptr)) return;
                    oracle.bytes(name+"/logits/"+std::to_string(q),logits,vocab_size*sizeof(float));
                }
                state(name+"/verify");
                const size_t proposals=width-1;
                const size_t accepted=round == 0 || round == 3 ? 0 : round == 1 ? proposals/2 : round == 4 ? proposals-1 : proposals;
                const size_t kept=accepted+1;
                if (kept < width && !t.assert_true(llama_memory_seq_rm(llama_get_memory(ctx.get()),0,llama_pos(position+kept),-1))) return;
                t.assert_equal(llama_pos(position+kept-1),llama_memory_seq_pos_max(llama_get_memory(ctx.get()),0));
                state(name+"/selected");
                kv_stream_exact_kv(oracle,name+"/retained_KV",kv,position,kept);
                position+=kept;
                if (!t.assert_equal(0,submit(position,1,true))) return;
                drain();
                oracle.bytes(name+"/correction_logits",llama_get_logits_ith(ctx.get(),0),vocab_size*sizeof(float));
                state(name+"/correction");
                kv_stream_exact_kv(oracle,name+"/corrected_KV",kv,position,1);
                ++position;
                if (streamed) t.assert_equal(position,stream->tokens());
            }
            const size_t capacity=llama_n_ctx(ctx.get());
            for (;position<capacity;) {
                const size_t rows=std::min(size_t(256),capacity-position);
                if (!t.assert_equal(0,submit(position,rows,false))) return;
                position+=rows;
            }
            state("context_end");
            if (!t.assert_true(submit(position,depth+1,true) != 0)) return;
            t.assert_equal(llama_pos(capacity-1),llama_memory_seq_pos_max(llama_get_memory(ctx.get()),0));
            state("rejected_context_end");
            if (streamed) oracle.finish();
        }
        t.out << "D=" << depth << " prefix=" << prefix << " native/spilled rollback bits match (" <<
            oracle.records() << " records, " << oracle.byte_count() << " bytes)\n";
    }
}

static run_result evaluate(testing & t,llama_model * model,const std::vector<llama_token> & prompt,
        uint32_t ubatch,size_t pool,const std::vector<llama_token> & forced) {
    run_result result;
    auto params = llama_context_default_params();
    params.n_ctx = 1024; params.n_batch = 512; params.n_ubatch = ubatch;
    params.n_threads = params.n_threads_batch = 8; params.n_seq_max = 1;
    params.type_k = f16_control ? GGML_TYPE_F16 : GGML_TYPE_Q8_0;
    params.type_v = f16_control ? GGML_TYPE_F16 : GGML_TYPE_Q4_0;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    if (shared_budget_control) params.shared_device_memory_bytes = pool;
    else params.kv_stream_pool_bytes = pool;
    if (trace_control) {
        params.cb_eval_user_data = &result;
        params.cb_eval = [](ggml_tensor * tensor,bool ask,void * opaque) {
            auto & trace = static_cast<run_result *>(opaque)->trace;
            const std::string name = tensor->name;
            if (tensor->type != GGML_TYPE_F32 || (name != "Qcur-3" && name != "Kcur-3" && name != "Vcur-3" && name != "attn_pregate-3")) return false;
            if (std::count_if(trace.begin(),trace.end(),[&](const auto & v) { return v.first == name; }) >= 2) return false;
            if (ask) return true;
            trace.emplace_back(name,std::vector<float>(ggml_nbytes(tensor)/sizeof(float)));
            ggml_backend_tensor_get(tensor,trace.back().second.data(),0,ggml_nbytes(tensor)); return true;
        };
    }
    context_ptr context(llama_init_from_model(model,params),llama_free);
    if (!t.assert_true(bool(context))) return result;
    auto * hybrid = static_cast<llama_memory_hybrid *>(llama_get_memory(context.get()));
    t.assert_true(bool(hybrid->get_mem_attn()->get_kv_stream()) == (pool != 0));
    const size_t vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    auto batch = llama_batch_init(512,0,1);
    struct batch_guard { llama_batch & batch; ~batch_guard() { llama_batch_free(batch); } } batch_owner{batch};
    const auto submit = [&](const std::vector<llama_token> & tokens,size_t first,bool decode) {
        batch.n_tokens = int32_t(tokens.size());
        for (size_t i = 0; i < tokens.size(); ++i) {
            batch.token[i] = tokens[i]; batch.pos[i] = llama_pos(first+i);
            batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0; batch.logits[i] = i+1 == tokens.size();
        }
        llama_set_kv_stream_decode(context.get(),decode);
        return llama_decode(context.get(),batch) == 0;
    };
    for (size_t first = 0; first < prompt.size(); first += 512) {
        const size_t count = std::min(size_t(512),prompt.size()-first);
        if (!t.assert_true(submit({prompt.begin()+first,prompt.begin()+first+count},first,false))) return {};
    }
    for (size_t step = 0; step < 32; ++step) {
        auto * logits = llama_get_logits_ith(context.get(),-1);
        result.logits.emplace_back(logits,logits+vocab);
        const auto token = forced.empty() ? llama_token(std::max_element(logits,logits+vocab)-logits) : forced[step];
        result.continuation.push_back(token);
        if (!t.assert_true(submit({token},prompt.size()+step,true))) return {};
    }
    llama_synchronize(context.get()); hybrid->get_mem_recr()->state_write(result.recurrent,0,0);
    if (pool) {
        auto * stream = hybrid->get_mem_attn()->get_kv_stream();
        t.assert_equal(prompt.size()+32,stream->tokens());
        t.out << "resident attention captures=" << stream->captured_layers() << '\n';
        if ((f16_control || resident_control) && !std::getenv("GGML_CUDA_DISABLE_GRAPHS")) {
            t.assert_equal(hybrid->get_mem_attn()->get_layer_ids().size(),stream->captured_layers());
        }
        t.assert_true(!llama_memory_seq_rm(llama_get_memory(context.get()),0,llama_pos(stream->tokens()-1),-1));
        std::vector<llama_token> overflow(1024-stream->tokens()+1,prompt.front());
        t.assert_true(llama_decode(context.get(),llama_batch_get_one(overflow.data(),int32_t(overflow.size()))) != 0);
        llama_memory_seq_cp(llama_get_memory(context.get()),0,1,0,-1);
        llama_memory_seq_keep(llama_get_memory(context.get()),1);
        llama_memory_seq_add(llama_get_memory(context.get()),0,0,-1,1);
        llama_memory_seq_div(llama_get_memory(context.get()),0,0,-1,2);
        recurrent_snapshot unchanged; hybrid->get_mem_recr()->state_write(unchanged,0,0);
        t.assert_true(unchanged.metadata == result.recurrent.metadata && unchanged.tensors == result.recurrent.tensors);
        const size_t state_size = llama_state_seq_get_size(context.get(),0);
        t.assert_equal(size_t(0),llama_state_seq_get_size_ext(context.get(),0,LLAMA_STATE_SEQ_FLAGS_ON_DEVICE));
        std::vector<uint8_t> saved(state_size);
        t.assert_equal(state_size,llama_state_seq_get_data(context.get(),saved.data(),saved.size(),0));
        t.assert_equal(size_t(0),llama_state_seq_set_data_ext(context.get(),saved.data(),saved.size(),0,LLAMA_STATE_SEQ_FLAGS_ON_DEVICE));
        const auto checkpoint_tokens=stream->tokens();
        t.assert_true(submit({prompt.front()},checkpoint_tokens,true));
        std::vector<float> expected_logits(llama_get_logits_ith(context.get(),-1),llama_get_logits_ith(context.get(),-1)+vocab);
        recurrent_snapshot expected_state; hybrid->get_mem_recr()->state_write(expected_state,0,0);
        t.assert_equal(state_size,llama_state_seq_set_data(context.get(),saved.data(),saved.size(),0));
        t.assert_equal(checkpoint_tokens,stream->tokens());
        t.assert_true(submit({prompt.front()},checkpoint_tokens,true));
        auto * restored_logits=llama_get_logits_ith(context.get(),-1);
        t.assert_true(std::equal(expected_logits.begin(),expected_logits.end(),restored_logits));
        recurrent_snapshot restored_state; hybrid->get_mem_recr()->state_write(restored_state,0,0);
        t.assert_true(expected_state.metadata==restored_state.metadata && expected_state.tensors==restored_state.tensors);
        t.assert_equal(size_t(0),llama_state_seq_set_data(context.get(),saved.data(),saved.size()-1,0));
        t.assert_equal(state_size,llama_state_seq_set_data(context.get(),saved.data(),saved.size(),0));
        auto sparse=saved;
        llama_pos wrong_first=-2;
        const size_t first_position=sizeof(uint32_t)+sizeof(llama_seq_id)+2*sizeof(uint32_t);
        std::memcpy(sparse.data()+first_position,&wrong_first,sizeof(wrong_first));
        t.assert_equal(size_t(0),llama_state_seq_set_data(context.get(),sparse.data(),sparse.size(),0));
        t.assert_equal(state_size,llama_state_seq_set_data(context.get(),saved.data(),saved.size(),0));
        t.assert_equal(prompt.size()+32,stream->tokens());
        struct abort_data { int calls=0; } abort;
        llama_set_abort_callback(context.get(),[](void * p) { return ++static_cast<abort_data *>(p)->calls > 0; },&abort);
        auto abort_token=prompt.front();
        t.assert_equal(int32_t(2),llama_decode(context.get(),llama_batch_get_one(&abort_token,1)));
        llama_set_abort_callback(context.get(),nullptr,nullptr);
        t.assert_true(abort.calls>0);
        t.assert_equal(state_size,llama_state_seq_set_data(context.get(),saved.data(),saved.size(),0));
        t.assert_true(submit({prompt.front()},stream->tokens(),true));
        restored_logits=llama_get_logits_ith(context.get(),-1);
        t.assert_true(std::equal(expected_logits.begin(),expected_logits.end(),restored_logits));
        if (ubatch==256) {
            const auto whole_tokens=stream->tokens();
            const size_t whole_size=llama_state_get_size(context.get());
            std::vector<uint8_t> whole(whole_size);
            t.assert_equal(whole_size,llama_state_get_data(context.get(),whole.data(),whole.size()));
            t.assert_true(submit({prompt.front()},whole_tokens,true));
            std::vector<float> whole_logits(llama_get_logits_ith(context.get(),-1),llama_get_logits_ith(context.get(),-1)+vocab);
            recurrent_snapshot whole_state; hybrid->get_mem_recr()->state_write(whole_state,0,0);
            t.assert_equal(whole_size,llama_state_set_data(context.get(),whole.data(),whole.size()));
            t.assert_equal(whole_tokens,stream->tokens());
            t.assert_true(submit({prompt.front()},whole_tokens,true));
            t.assert_true(std::equal(whole_logits.begin(),whole_logits.end(),llama_get_logits_ith(context.get(),-1)));
            recurrent_snapshot whole_restored; hybrid->get_mem_recr()->state_write(whole_restored,0,0);
            t.assert_true(whole_state.metadata==whole_restored.metadata && whole_state.tensors==whole_restored.tensors);
        }
        // A no-op adapter reset still forces scheduler/workspace reconstruction in the real context.
        t.assert_equal(int32_t(0),llama_set_adapter_cvec(context.get(),nullptr,0,0,-1,-1));
        t.assert_true(submit({prompt.front()},stream->tokens(),true));
        llama_memory_clear(llama_get_memory(context.get()),true);
        t.assert_equal(size_t(0),stream->tokens());
        t.assert_true(submit({prompt.front()},0,false));
        t.assert_equal(size_t(1),stream->tokens());
    }
    return result;
}

struct serial_phase_result {
    std::vector<std::vector<float>> logits;
    recurrent_snapshot recurrent;
    std::vector<size_t> pool_grants;
    std::vector<int64_t> prefill_after_decode_us;
};

static serial_phase_result evaluate_serial_phases(testing & t, llama_model * model,
        const std::vector<llama_token> & prompt, size_t pool) {
    serial_phase_result result;
    auto params = llama_context_default_params();
    params.n_ctx = 2048; params.n_batch = 512; params.n_ubatch = 256;
    params.n_threads = params.n_threads_batch = 8; params.n_seq_max = 1;
    params.type_k = GGML_TYPE_Q8_0; params.type_v = GGML_TYPE_Q4_0;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    if (shared_budget_control) params.shared_device_memory_bytes = pool;
    else params.kv_stream_pool_bytes = pool;
    context_ptr context(llama_init_from_model(model,params),llama_free);
    if (!t.assert_true(bool(context))) return result;
    auto * hybrid = static_cast<llama_memory_hybrid *>(llama_get_memory(context.get()));
    auto * stream = hybrid->get_mem_attn()->get_kv_stream();
    if (!t.assert_true(bool(stream) == (pool != 0))) return result;
    const size_t vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    auto batch = llama_batch_init(512,0,1);
    struct batch_guard {
        llama_batch & batch;
        ~batch_guard() { llama_batch_free(batch); }
    } batch_owner{batch};
    size_t position = 0;

    const auto submit = [&](const std::vector<llama_token> & tokens, bool decode) {
        if (tokens.empty() || tokens.size() > 512) return false;
        batch.n_tokens = int32_t(tokens.size());
        for (size_t i = 0; i < tokens.size(); ++i) {
            batch.token[i] = tokens[i];
            batch.pos[i] = llama_pos(position+i);
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = i+1 == tokens.size();
        }
        llama_set_kv_stream_decode(context.get(),decode);
        if (llama_decode(context.get(),batch) != 0) return false;
        position += tokens.size();
        return true;
    };
    const auto capture = [&] {
        auto * logits = llama_get_logits_ith(context.get(),-1);
        result.logits.emplace_back(logits,logits+vocab);
        if (stream) result.pool_grants.push_back(stream->pool_grant_bytes());
    };
    const auto append = [&](size_t first,size_t count,bool decode) {
        if (first > prompt.size() || count > prompt.size()-first) return false;
        return submit({prompt.begin()+first,prompt.begin()+first+count},decode);
    };
    const auto append_prompt = [&](size_t count) {
        for (size_t first = 0; first < count; first += 512) {
            const size_t chunk = std::min(size_t(512),count-first);
            if (!append(first,chunk,false)) return false;
        }
        return true;
    };
    const auto fixed_decode = [&](size_t count,size_t seed) {
        for (size_t i = 0; i < count; ++i) {
            if (!append((seed+i)%prompt.size(),1,true)) return false;
        }
        return true;
    };
    const auto timed_prefill = [&](size_t first,size_t count) {
        const auto begin = std::chrono::steady_clock::now();
        const bool ok = append(first,count,false);
        const auto end = std::chrono::steady_clock::now();
        result.prefill_after_decode_us.push_back(
            std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count());
        return ok;
    };
    const auto clear = [&] {
        llama_memory_clear(llama_get_memory(context.get()),true);
        position = 0;
        return !stream || stream->tokens() == 0;
    };
    const auto cancel_and_restore = [&](size_t first,size_t count,bool decode) {
        if (first > prompt.size() || count > prompt.size()-first || !count || count > 512) return false;
        const size_t state_size = llama_state_get_size(context.get());
        std::vector<uint8_t> state(state_size);
        if (llama_state_get_data(context.get(),state.data(),state.size()) != state_size) return false;
        batch.n_tokens = int32_t(count);
        for (size_t i = 0; i < count; ++i) {
            batch.token[i] = prompt[first+i];
            batch.pos[i] = llama_pos(position+i);
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = i+1 == count;
        }
        struct abort_data { int calls = 0; } abort;
        llama_set_kv_stream_decode(context.get(),decode);
        llama_set_abort_callback(context.get(),[](void * opaque) {
            return ++static_cast<abort_data *>(opaque)->calls > 0;
        },&abort);
        const int32_t status = llama_decode(context.get(),batch);
        llama_set_abort_callback(context.get(),nullptr,nullptr);
        if (status != 2 || abort.calls == 0 ||
                llama_state_set_data(context.get(),state.data(),state.size()) != state_size) return false;
        return !stream || stream->tokens() == position;
    };

    if (!t.assert_true(append_prompt(640))) return {};
    capture();
    if (!t.assert_true(fixed_decode(192,3))) return {};
    capture();
    if (!t.assert_true(timed_prefill(80,7))) return {};
    capture();
    if (!t.assert_true(fixed_decode(16,101))) return {};
    capture();
    if (!t.assert_true(timed_prefill(160,192))) return {};
    capture();
    if (!t.assert_true(fixed_decode(16,211))) return {};
    capture();
    if (!t.assert_true(timed_prefill(17,3))) return {};
    capture();
    if (!t.assert_true(fixed_decode(8,307))) return {};
    capture();

    if (!t.assert_true(clear() && append_prompt(17))) return {};
    capture();
    if (!t.assert_true(fixed_decode(8,401))) return {};
    capture();
    if (!t.assert_true(clear() && append_prompt(640))) return {};
    capture();
    if (!t.assert_true(fixed_decode(8,503))) return {};
    capture();

    if (!t.assert_true(cancel_and_restore(80,7,false))) return {};
    if (!t.assert_true(append(80,7,false))) return {};
    capture();
    if (!t.assert_true(cancel_and_restore(503,1,true))) return {};
    if (!t.assert_true(append(503,1,true))) return {};
    capture();
    llama_synchronize(context.get());
    hybrid->get_mem_recr()->state_write(result.recurrent,0,0);
    if (stream) t.assert_equal(position,stream->tokens());
    return result;
}

// Keep exact model outputs across fresh processes without accepting truncated or mismatched controls.
static void gemma_logits_file(FILE * file,bool writing,const std::array<uint64_t,8> & profile,
        std::vector<std::vector<float>> & values) {
    constexpr uint64_t magic=0x474d344c4f474954ull;
    if (!file || profile[0] != magic || profile[1] != 1 || !profile[2] || profile[2] > INT32_MAX ||
            profile[3] >= profile[2] || !profile[4] || profile[4] > 256 || profile[4] > profile[2]-profile[3] ||
            !profile[5] || profile[5] > INT32_MAX || profile[5] > (uint64_t(32)<<30)/sizeof(float)/profile[4] ||
            profile[6] >= GGML_TYPE_COUNT || profile[7] >= GGML_TYPE_COUNT)
        throw std::runtime_error("invalid Gemma logit control profile");
    if (writing) {
        if (values.size() != profile[4]) throw std::runtime_error("invalid Gemma logit control rows");
        for (const auto & row : values) if (row.size() != profile[5])
            throw std::runtime_error("invalid Gemma logit control columns");
        if (std::fwrite(profile.data(),1,sizeof(profile),file) != sizeof(profile))
            throw std::runtime_error("cannot write Gemma logit control header");
        for (const auto & row : values) if (std::fwrite(row.data(),sizeof(float),row.size(),file) != row.size())
            throw std::runtime_error("cannot write Gemma logit control values");
        const uint64_t footer=~magic;
        if (std::fwrite(&footer,1,sizeof(footer),file) != sizeof(footer) || std::fflush(file))
            throw std::runtime_error("cannot finish Gemma logit control");
        return;
    }
    std::array<uint64_t,8> actual{};
    if (std::fread(actual.data(),1,sizeof(actual),file) != sizeof(actual) || actual != profile)
        throw std::runtime_error("Gemma logit control header differs");
    auto next=std::vector<std::vector<float>>(size_t(profile[4]),std::vector<float>(size_t(profile[5])));
    for (auto & row : next) if (std::fread(row.data(),sizeof(float),row.size(),file) != row.size())
        throw std::runtime_error("truncated Gemma logit control values");
    uint64_t footer=0;
    if (std::fread(&footer,1,sizeof(footer),file) != sizeof(footer) || footer != ~magic || std::fgetc(file) != EOF || std::ferror(file))
        throw std::runtime_error("invalid Gemma logit control footer");
    values=std::move(next);
}

static std::vector<float> gemma_image_reference(FILE * file,size_t width) {
    if (!file || !width || width>SIZE_MAX/sizeof(float) || std::fseek(file,0,SEEK_END))
        throw std::runtime_error("invalid Gemma image reference");
    const long bytes=std::ftell(file);
    const size_t row_bytes=width*sizeof(float);
    if (bytes<=0 || bytes>32L*1048576 || size_t(bytes)%row_bytes || size_t(bytes)/row_bytes>1120 || std::fseek(file,0,SEEK_SET))
        throw std::runtime_error("invalid Gemma image reference extent");
    std::vector<float> values(size_t(bytes)/sizeof(float));
    if (std::fread(values.data(),sizeof(float),values.size(),file)!=values.size() || std::fgetc(file)!=EOF || std::ferror(file) ||
            !std::all_of(values.begin(),values.end(),[](float value){return std::isfinite(value);}))
        throw std::runtime_error("invalid Gemma image reference values");
    return values;
}

static void gemma_assistant_oracle(const std::array<FILE *,3> & files,bool writing,uint64_t capacity,uint64_t prefix,
        const std::vector<float> & image,size_t width,size_t vocab,std::vector<std::vector<float>> & logits,
        std::vector<std::vector<float>> & hidden) {
    if (!width || !vocab || image.empty() || image.size()%width || image.size()/width>1120)
        throw std::runtime_error("invalid image-conditioned assistant oracle profile");
    for (auto * file : files) if (!file || std::fseek(file,0,SEEK_SET))
        throw std::runtime_error("invalid image-conditioned assistant oracle file");
    const auto profile=[&](size_t rows,size_t columns) {
        return std::array<uint64_t,8>{0x474d344c4f474954ull,1,capacity,prefix,rows,columns,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0};
    };
    std::vector<std::vector<float>> input{image};
    gemma_logits_file(files[0],writing,profile(1,image.size()),input);
    if (!writing && !kv_stream_bytes_equal(image.data(),image.size()*sizeof(float),input[0].data(),input[0].size()*sizeof(float)))
        throw std::runtime_error("assistant oracle image bytes differ");
    auto next_logits=writing ? logits : std::vector<std::vector<float>>{};
    auto next_hidden=writing ? hidden : std::vector<std::vector<float>>{};
    gemma_logits_file(files[1],writing,profile(19,vocab),next_logits);
    gemma_logits_file(files[2],writing,profile(18,width),next_hidden);
    for (const auto * rows : {&next_logits,&next_hidden}) for (const auto & row : *rows)
        if (!std::all_of(row.begin(),row.end(),[](float value){return std::isfinite(value);}))
            throw std::runtime_error("assistant oracle contains nonfinite values");
    if (!writing) { logits=std::move(next_logits);hidden=std::move(next_hidden); }
}

// Qualify the model graph, not just individual cache owners or attention operators.
static void test_gemma_target_context(testing & t, llama_model * model,uint32_t capacity=8192,
        uint32_t prompt=1333,uint32_t outputs=8,bool native_control=true,bool expect_streaming=false,
        const char * export_native=nullptr,const char * import_native=nullptr) {
    if (!t.assert_true(model && model->arch == LLM_ARCH_GEMMA4)) return;
    if (!t.assert_true(capacity <= model->hparams.n_ctx_train && prompt > 1024 &&
            prompt < capacity && outputs > 0 && outputs <= capacity-prompt)) return;
    std::vector<std::vector<float>> reference;
    const auto vocab=size_t(llama_vocab_n_tokens(llama_model_get_vocab(model)));
    const std::array<uint64_t,8> profile{0x474d344c4f474954ull,1,capacity,prompt,outputs,vocab,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0};
    if (!t.assert_true(!(export_native && import_native) && (!(export_native || import_native) || native_control))) return;
    if (import_native) {
        std::unique_ptr<FILE,kv_stream_exact_file_free> file(std::fopen(import_native,"rb"));
        gemma_logits_file(file.get(),false,profile,reference);
    }
    for (bool streaming : {false,true}) {
        if (!streaming && !native_control) continue;
        if ((!streaming && import_native) || (streaming && export_native)) continue;
        auto cp=llama_context_default_params();
        cp.n_ctx=capacity; cp.n_batch=cp.n_ubatch=256; cp.n_seq_max=1;
        cp.n_threads=cp.n_threads_batch=8; cp.swa_full=false;
        cp.type_k=GGML_TYPE_Q8_0; cp.type_v=GGML_TYPE_Q4_0;
        cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
        if (streaming) cp.shared_device_memory_bytes=gemma_test_arena_size;
        context_ptr context(llama_init_from_model(model,cp),llama_free);
        if (!t.assert_true(streaming ? "streamed Gemma target context is admitted" : "native Gemma control starts",bool(context))) return;
        auto * groups=dynamic_cast<llama_kv_cache_iswa *>(llama_get_memory(context.get()));
        if (!t.assert_true(groups && !groups->get_swa()->bounded_storage_pending())) return;
        if (streaming) {
            auto * owner=context->get_compute_memory();
            if (!t.assert_true(owner && owner->shares_kv_memory() && owner->has_bounded_cache())) return;
            t.assert_equal(cp.shared_device_memory_bytes,owner->shared_parent_capacity());
            t.assert_true(ggml_backend_buffer_is_view(groups->get_swa()->get_k_storage(0)->buffer));
        }
        common_batch batch(context.get());
        for (int32_t first=0;first<int32_t(prompt);) {
            const auto rows=std::min(256,int32_t(prompt)-first);
            batch.clear();
            for (int32_t row=0;row<rows;++row) batch.add(llama_token(10+(first+row)%64),first+row,0,row == rows-1);
            if (!t.assert_equal(0,llama_process(context.get(),LLAMA_PROCESS_TYPE_DECODE,batch.get()))) return;
            first+=rows;
            if (first%8192 == 0 || first == int32_t(prompt)) t.out << "Gemma target streaming=" << streaming
                << " prefill=" << first << '/' << prompt << std::endl;
        }
        if (streaming) llama_set_kv_stream_decode(context.get(),true);
        for (uint32_t step=0;step<outputs;++step) {
            const auto * logits=llama_get_logits_ith(context.get(),-1);
            if (!t.assert_true(logits != nullptr)) return;
            t.assert_true("Gemma model logits remain finite",
                std::all_of(logits,logits+vocab,[](float value) { return std::isfinite(value); }));
            if (!streaming) reference.emplace_back(logits,logits+vocab);
            else if (native_control) t.assert_true("native and streamed Gemma logits are bit-exact",
                std::memcmp(reference[size_t(step)].data(),logits,vocab*sizeof(float)) == 0);
            else t.assert_true(std::all_of(logits,logits+vocab,[](float value) { return std::isfinite(value); }));
            batch.clear(); batch.add(llama_token(80+step),llama_pos(prompt+step),0,true);
            if (!t.assert_equal(0,llama_process(context.get(),LLAMA_PROCESS_TYPE_DECODE,batch.get()))) return;
        }
        llama_synchronize(context.get());
        if (streaming) {
            auto * stream=groups->get_base()->get_kv_stream();
            t.assert_true(stream && stream->complete());
            t.assert_equal(size_t(prompt)+outputs,stream->tokens());
            llama_context_memory_diagnostics usage;
            if (!t.assert_true(context->get_compute_memory()->diagnostics(usage))) return;
            t.assert_equal(size_t(425984000),usage.persistent_cache_bytes);
            t.assert_equal(cp.shared_device_memory_bytes,usage.parent_bytes);
            t.assert_true(usage.phase == llama_memory_text_phase::decode);
            if (expect_streaming) t.assert_true("longer-than-pool target uses streaming",usage.streaming_active);
            t.out << "Gemma target context=" << capacity << " prompt=" << prompt << " outputs=" << outputs
                << " native_control=" << native_control << " decode_pool=" << usage.kv_pool_bytes
                << " resident/ring=" << usage.resident_pages_per_layer << '/' << usage.ring_slots
                << " active=" << usage.active_pages << '\n';
        }
    }
    if (export_native) {
        std::unique_ptr<FILE,kv_stream_exact_file_free> file(std::fopen(export_native,"wbx"));
        gemma_logits_file(file.get(),true,profile,reference);
    }
}

static void test_gemma_vision_suspension(testing & t, llama_model * model) {
    if (!t.assert_true(model && model->arch == LLM_ARCH_GEMMA4)) return;
    const size_t vocab=llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<std::vector<float>> expected;
    for (bool streaming : {false,true}) {
        auto cp=llama_context_default_params();
        cp.n_ctx=4096; cp.n_batch=cp.n_ubatch=256; cp.n_seq_max=1;
        cp.n_threads=cp.n_threads_batch=8; cp.swa_full=false;
        cp.type_k=GGML_TYPE_Q8_0; cp.type_v=GGML_TYPE_Q4_0;
        cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
        if (streaming) cp.shared_device_memory_bytes=size_t(2048)*1048576;
        context_ptr ctx(llama_init_from_model(model,cp),llama_free);
        if (!t.assert_true(bool(ctx))) return;
        common_batch batch(ctx.get());
        for (int32_t first=0;first<1333;) {
            const int32_t rows=std::min(256,1333-first);
            batch.clear();
            for (int32_t row=0;row<rows;++row) batch.add(llama_token(10+(first+row)%64),first+row,0,row==rows-1);
            if (!t.assert_equal(0,llama_process(ctx.get(),LLAMA_PROCESS_TYPE_DECODE,batch.get()))) return;
            first+=rows;
        }
        const float * logits=llama_get_logits_ith(ctx.get(),-1);
        if (!t.assert_true(logits && std::all_of(logits,logits+vocab,[](float value){return std::isfinite(value);}))) return;
        if (!streaming) expected.emplace_back(logits,logits+vocab);
        else {
            if (!t.assert_true("native control is exact before suspension",std::memcmp(expected[0].data(),logits,vocab*sizeof(float))==0)) return;
            auto * owner=llama_context_compute_memory(ctx.get());
            if (!t.assert_true(owner && owner->has_bounded_cache() && owner->can_suspend_for_vision())) return;
            auto snapshot=[&] {
                std::vector<uint8_t> bytes(llama_state_get_size(ctx.get()));
                const size_t written=llama_state_get_data(ctx.get(),bytes.data(),bytes.size());
                bytes.resize(written);return bytes;
            };
            const auto saved=snapshot();
            if (!t.assert_true(!saved.empty())) return;
            auto * parent=owner->shared_parent();
            if (!t.assert_true("Gemma vision suspension does not require recurrent state",llama_context_suspend_kv_device(ctx.get()))) return;
            llama_context_memory_diagnostics usage;
            if (!t.assert_true(owner->diagnostics(usage))) return;
            t.assert_true(usage.kv_device_suspended && usage.kv_pool_bytes==0 && usage.workspace_bytes==0);
            t.assert_equal(size_t(425984000),usage.persistent_cache_bytes);
            t.assert_equal(size_t(0),usage.borrowed_phase_bytes);
            std::vector<ggml_backend_memory_lease_t> grants;
            struct return_grants {
                std::vector<ggml_backend_memory_lease_t> & values;
                ~return_grants() { for (auto * lease : values) ggml_backend_memory_lease_free(lease); }
            } returned{grants};
            const auto generation=owner->shared_arena_generation();
            t.assert_true(!owner->lend_suspended({0},grants));
            t.assert_equal(generation,owner->shared_arena_generation());
            if (!t.assert_true("vision borrows around persistent SWA",owner->lend_suspended({65536,131072},grants))) return;
            std::vector<ggml_backend_memory_lease_t> duplicate;
            t.assert_true(!owner->lend_suspended({4096},duplicate) && duplicate.empty() && owner->valid());
            t.assert_true(!owner->lend_suspended({4096},grants) && grants.size()==2);
            for (auto * lease : grants) ggml_backend_buffer_clear(ggml_backend_memory_lease_buffer(lease),0x6d);
            t.assert_true("vision writes preserve all serialized target state",snapshot()==saved);
            t.assert_true(!llama_context_resume_kv_device(ctx.get(),llama_memory_text_phase::prefill));
            for (auto * lease : grants) ggml_backend_memory_lease_free(lease);
            grants.clear();
            const size_t available=usage.parent_bytes-usage.persistent_cache_bytes;
            const auto before_reject=owner->shared_arena_generation();
            t.assert_true(!owner->lend_suspended({available+1},grants) && grants.empty() && owner->valid());
            t.assert_equal(before_reject,owner->shared_arena_generation());
            if (!t.assert_true(owner->lend_suspended({available},grants))) return;
            llama_context_memory_diagnostics full_vision;
            if (!t.assert_true("vision diagnostics count persistent bytes once",owner->diagnostics(full_vision) &&
                    full_vision.borrowed_phase_bytes==available && full_vision.unused_bytes==0 &&
                    full_vision.persistent_cache_bytes==usage.persistent_cache_bytes && full_vision.kv_device_suspended)) return;
            ggml_backend_memory_region region;
            t.assert_true(ggml_backend_memory_lease_get_region(grants.front(),&region) && region.offset==0 && region.size==available);
            ggml_backend_buffer_clear(ggml_backend_memory_lease_buffer(grants.front()),0x3a);
            t.assert_true("entire reusable prefix is disjoint from preserved state",snapshot()==saved);
            for (auto * lease : grants) ggml_backend_memory_lease_free(lease);
            grants.clear();
            if (!t.assert_true("text resumes around persistent SWA",llama_context_resume_kv_device(ctx.get(),llama_memory_text_phase::prefill))) return;
            t.assert_true(owner->shared_parent()==parent && snapshot()==saved);
        }
        for (int32_t step=0;step<8;++step) {
            batch.clear();batch.add(llama_token(80+step),1333+step,0,true);
            if (!t.assert_equal(0,llama_process(ctx.get(),LLAMA_PROCESS_TYPE_DECODE,batch.get()))) return;
            logits=llama_get_logits_ith(ctx.get(),-1);
            if (!t.assert_true(logits!=nullptr)) return;
            if (!streaming) expected.emplace_back(logits,logits+vocab);
            else t.assert_true("native continuation is exact after vision suspension",std::memcmp(expected[size_t(step)+1].data(),logits,vocab*sizeof(float))==0);
        }
    }
}

static void test_gemma_image_features(testing & t, llama_model * model,const char * path) {
    const size_t width=llama_model_n_embd_inp(model);
    if (!t.assert_true(model->arch==LLM_ARCH_GEMMA4)) return;
    std::unique_ptr<FILE,kv_stream_exact_file_free> file(std::fopen(path,"rb"));
    const auto embeddings=gemma_image_reference(file.get(),width);
    const size_t rows=embeddings.size()/width;
    const size_t vocab=llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<std::vector<float>> expected;
    for (bool streaming : {false,true}) {
        auto cp=llama_context_default_params();
        cp.n_ctx=4096;cp.n_batch=cp.n_ubatch=uint32_t((rows+255)/256*256);cp.n_seq_max=1;
        gemma_image_test_outputs(cp,false);
        cp.n_threads=cp.n_threads_batch=8;cp.swa_full=false;
        cp.type_k=GGML_TYPE_Q8_0;cp.type_v=GGML_TYPE_Q4_0;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
        if (streaming) cp.shared_device_memory_bytes=gemma_image_test_arena_bytes(gemma_test_arena_explicit,gemma_test_arena_size);
        context_ptr ctx(llama_init_from_model(model,cp),llama_free);
        if (!t.assert_true(bool(ctx))) return;
        common_batch batch(ctx.get());
        for (int32_t first=0;first<1333;) {
            const int32_t count=std::min(256,1333-first);batch.clear();
            for (int32_t row=0;row<count;++row) batch.add(llama_token(10+(first+row)%64),first+row,0,row==count-1);
            if (!t.assert_equal(0,llama_process(ctx.get(),LLAMA_PROCESS_TYPE_DECODE,batch.get()))) return;
            first+=count;
        }
        batch.clear();
        for (size_t row=0;row<rows;++row) {
            llama_pos position[4]={llama_pos(1333+row),0,0,0};
            batch.add_embd({embeddings.data()+row*width,1,width},position,0,row+1==rows);
        }
        llama_set_causal_attn(ctx.get(),false);
        const int image_result=llama_process(ctx.get(),LLAMA_PROCESS_TYPE_DECODE,batch.get());
        llama_set_causal_attn(ctx.get(),true);
        if (!t.assert_equal(0,image_result)) return;
        for (size_t step=0;step<=8;++step) {
            const auto * logits=llama_get_logits_ith(ctx.get(),-1);
            if (!t.assert_true(logits && std::all_of(logits,logits+vocab,[](float value){return std::isfinite(value);}))) return;
            if (!streaming) expected.emplace_back(logits,logits+vocab);
            else if (!t.assert_true("image-conditioned native and streamed logits are bit-exact",
                    std::memcmp(expected[step].data(),logits,vocab*sizeof(float))==0)) return;
            if (step==8) break;
            batch.clear();batch.add(llama_token(80+step),llama_pos(1333+rows+step),0,true);
            if (!t.assert_equal(0,llama_process(ctx.get(),LLAMA_PROCESS_TYPE_DECODE,batch.get()))) return;
        }
    }
}

// Constructor failures must explain the rejected profile and leave a valid context constructible afterward.
static void test_gemma_target_admission(testing & t, llama_model * model,bool include_forward=true) {
    if (!t.assert_true(model && model->arch == LLM_ARCH_GEMMA4)) return;
    for (uint32_t invalid=0;invalid<6;++invalid) {
        auto cp=llama_context_default_params();
        cp.n_ctx=8192; cp.n_batch=cp.n_ubatch=256; cp.n_seq_max=1; cp.swa_full=false;
        cp.type_k=GGML_TYPE_Q8_0; cp.type_v=GGML_TYPE_Q4_0;
        cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.shared_device_memory_bytes=gemma_test_arena_size;
        const char * expected="KV streaming requires one CUDA device, serial Qwen35 or Gemma4 target context";
        if (invalid == 0) {
            cp.swa_full=true;
            expected="Gemma arena streaming requires bounded SWA storage";
        } else if (invalid == 1) cp.n_seq_max=2;
        else if (invalid == 2) cp.embeddings=true;
        else if (invalid == 3) cp.offload_kqv=false;
        else if (invalid == 4) {
            cp.shared_device_memory_bytes=256u*1024u*1024u;
            expected="workspace admission: insufficient arena quota";
        } else {
            cp.shared_device_memory_bytes=512u*1024u*1024u;
            expected="persistent local cache does not fit the declared shared arena";
        }
        bool rejected=false;
        try {
            auto context=std::make_unique<llama_context>(*model,cp);
        } catch (const std::runtime_error & error) {
            t.out << "Gemma admission case=" << invalid << ": " << error.what() << '\n';
            rejected=std::string(error.what()).find(expected) != std::string::npos;
            if (invalid == 4 || invalid == 5) {
                const std::string message=error.what();
                const auto start=message.find("required=");
                size_t required=0,available=0,additional=0;
                const bool parsed=start!=std::string::npos && std::sscanf(message.c_str()+start,
                    "required=%zu, available=%zu, additional=%zu",&required,&available,&additional)==3;
                t.assert_true("quota diagnostic carries exact byte fields",parsed);
                if (parsed) {
                    t.assert_equal(cp.shared_device_memory_bytes,available);
                    t.assert_true(required>available);
                    t.assert_equal(required-available,additional);
                }
                if (invalid==5) t.assert_true(message.find("phase=")!=std::string::npos && message.find("resource=")!=std::string::npos);
            }
        }
        t.assert_true(std::string("startup rejects profile with the expected cause: ")+expected,rejected);
    }
    {
        auto cp=llama_context_default_params();
        cp.n_ctx=8192; cp.n_batch=cp.n_ubatch=256; cp.n_seq_max=1; cp.swa_full=false;
        cp.type_k=GGML_TYPE_Q8_0; cp.type_v=GGML_TYPE_Q4_0;
        cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.shared_device_memory_bytes=gemma_test_arena_size; cp.n_rs_seq=3;
        context_ptr context(llama_init_from_model(model,cp),llama_free);
        if (!t.assert_true("Gemma keeps upstream's ignored recurrent rollback setting",bool(context))) return;
        t.assert_equal(uint32_t(0),context->get_cparams().n_rs_seq);
        t.assert_true(!context->get_compute_memory()->has_speculative_consumer());
    }
    if (include_forward) test_gemma_target_context(t,model);
}

// A rejected streamed suffix operation must not change the independent local group's cells.
static void test_gemma_lifecycle(testing & t, llama_model * model) {
    if (!t.assert_true(model && model->arch == LLM_ARCH_GEMMA4)) return;
    auto * dev=ggml_backend_dev_by_name("CUDA0");
    if (!t.assert_true(dev != nullptr)) return;
    ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
    llama_memory_params params{};
    params.type_k=GGML_TYPE_Q8_0; params.type_v=GGML_TYPE_Q4_0;
    params.kv_stream_pool_bytes=64u*1024u*1024u;
    params.kv_stream_max_rows=256; params.kv_stream_max_decode_rows=4;
    params.kv_stream_backend=backend.get();
    const auto no_reuse=[](int32_t) { return int32_t(-1); };
    llama_kv_cache_iswa cache(*model,params.type_k,params.type_v,false,true,false,true,
        8192,1,256,256,nullptr,nullptr,no_reuse,nullptr,&params);
    llama_batch_allocr allocator(1);
    auto batch=allocator.ubatch_reserve(64,1);
    batch.data->seq_id_data.resize(batch.n_tokens,0);
    for (uint32_t row=0;row<batch.n_tokens;++row) {
        batch.pos[row]=llama_pos(row); batch.n_seq_id[row]=1;
        batch.seq_id[row]=&batch.data->seq_id_data[row];
    }
    for (auto * group : {cache.get_base(),cache.get_swa()}) {
        auto slots=group->prepare({batch});
        if (!t.assert_true(slots.size() == 1)) return;
        group->apply_ubatch(slots[0],batch);
    }
    auto * stream=cache.get_base()->get_kv_stream();
    if (!t.assert_true(stream != nullptr)) return;
    ggml_backend_buffer_clear(stream->buffer(),0);
    if (!t.assert_true("imported host frontier is restored",stream->restore(64))) return;
    const auto positions=[](const llama_kv_cache & group) {
        std::vector<llama_pos> result;
        const auto & cells=group.get_cells(0);
        for (uint32_t row=0;row<group.get_size();++row)
            result.push_back(cells.is_empty(row) ? -1 : cells.pos_get(row));
        return result;
    };
    const auto full=positions(*cache.get_base()), local=positions(*cache.get_swa());
    t.assert_true(!cache.seq_rm(0,10,20));
    t.assert_equal(size_t(64),stream->tokens());
    t.assert_true(full == positions(*cache.get_base()));
    t.assert_true("rejected interior removal preserves local cells",local == positions(*cache.get_swa()));

    const auto save=[&] {
        image_writer image;
        cache.state_write(image,0,0);
        return image.bytes;
    };
    // Exercise stock placement/serialization with encoded rows, without claiming a model-forward numerical test.
    const auto append=[&](uint32_t first,uint32_t count,uint32_t seed) {
        for (uint32_t done=0;done<count;) {
            const uint32_t rows=std::min(256u,count-done);
            auto batch=allocator.ubatch_reserve(rows,1);
            batch.data->seq_id_data.resize(rows,0);
            for (uint32_t row=0;row<rows;++row) {
                batch.pos[row]=llama_pos(first+done+row); batch.token[row]=llama_token(seed);
                batch.n_seq_id[row]=1; batch.seq_id[row]=&batch.data->seq_id_data[row];
            }
            for (auto * group : {cache.get_base(),cache.get_swa()}) {
                auto slots=group->prepare({batch});
                if (!t.assert_true(slots.size() == 1)) return false;
                group->apply_ubatch(slots[0],batch);
                const int32_t id=group == cache.get_base() ? 5 : 0;
                for (auto * tensor : {group->get_k_storage(id),group->get_v_storage(id)}) {
                    const auto row_bytes=ggml_row_size(tensor->type,tensor->ne[0]);
                    std::vector<float> floats(size_t(tensor->ne[0]));
                    std::vector<uint8_t> encoded(row_bytes);
                    for (uint32_t row=0;row<rows;++row) {
                        std::fill(floats.begin(),floats.end(),float(seed+uint32_t(batch.pos[row])%17)*0.0625f);
                        ggml_quantize_chunk(tensor->type,floats.data(),encoded.data(),0,1,tensor->ne[0],nullptr);
                        ggml_backend_tensor_set(tensor,encoded.data(),slots[0].idxs[0][row]*row_bytes,row_bytes);
                    }
                }
            }
            if (!t.assert_true(stream->restore(first+done+rows))) return false;
            done+=rows;
        }
        return true;
    };
    const auto local_window=[&] {
        auto * group=cache.get_swa();
        const auto & cells=group->get_cells(0);
        std::map<llama_pos,std::vector<uint8_t>> result;
        for (auto * tensor : {group->get_k_storage(0),group->get_v_storage(0)}) {
            std::vector<uint8_t> bytes(ggml_nbytes(tensor));
            ggml_backend_tensor_get(tensor,bytes.data(),0,bytes.size());
            const auto row_bytes=ggml_row_size(tensor->type,tensor->ne[0]);
            for (uint32_t row=0;row<group->get_size();++row) {
                if (cells.is_empty(row)) continue;
                const auto pos=cells.pos_get(row);
                if (llama_hparams::is_masked_swa(model->hparams.n_swa,model->hparams.swa_type,pos,cells.seq_pos_max(0))) continue;
                auto & entry=result[pos];
                entry.insert(entry.end(),bytes.data()+row*row_bytes,bytes.data()+(row+1)*row_bytes);
            }
        }
        return result;
    };
    cache.clear(true);
    if (!t.assert_true(append(0,64,1))) return;
    const auto saved=save();
    t.assert_true(!cache.seq_rm(0,0,16) && saved == save());
    t.assert_true(cache.seq_rm(0,16,-1));
    t.assert_equal(size_t(16),stream->tokens());
    if (!t.assert_true(append(16,48,2))) return;
    t.assert_true("equal-length requests with different content are distinct",saved != save());
    cache.clear(true);
    t.assert_equal(size_t(0),stream->tokens());
    t.assert_equal(uint32_t(0),cache.get_swa()->get_cells(0).get_used());
    image_reader reader(saved);
    cache.state_read(reader,0,0);
    t.assert_equal(saved.size(),reader.n_bytes());
    t.assert_equal(size_t(64),stream->tokens());
    t.assert_true("both groups round-trip through saved state",saved == save());
    t.assert_true(cache.seq_rm(0,16,-1));
    if (!t.assert_true(append(16,48,2))) return;
    const auto rewritten=save();
    t.assert_true(rewritten != saved);
    t.assert_true(cache.seq_rm(0,64,64) && rewritten == save());

    cache.clear(true);
    if (!t.assert_true(append(0,2304,3))) return;
    const auto baseline=local_window();
    t.assert_equal(size_t(model->hparams.n_swa),baseline.size());
    for (uint32_t accepted=0;accepted<=4;++accepted) {
        if (!t.assert_true(append(2304,4,7))) return;
        const auto provisional=local_window();
        if (!t.assert_true(cache.seq_rm(0,llama_pos(2304+accepted),-1))) return;
        t.assert_equal(size_t(2304+accepted),stream->tokens());
        const auto retained=local_window();
        t.assert_equal(baseline.size(),retained.size());
        for (const auto & entry : retained) {
            const auto & reference=entry.first < 2304 ? baseline : provisional;
            const auto found=reference.find(entry.first);
            t.assert_true("accepted window bytes survive rejected writes",found != reference.end() && found->second == entry.second);
        }
        t.assert_true(cache.seq_rm(0,2304,-1));
        t.assert_true("zero acceptance and cancellation recover the complete local window",local_window() == baseline);
    }
    const auto full_before_old_prefix=positions(*cache.get_base());
    t.assert_true("a prefix whose local history was evicted cannot be reused",!cache.seq_rm(0,64,-1));
    t.assert_equal(size_t(2304),stream->tokens());
    t.assert_true(full_before_old_prefix == positions(*cache.get_base()));
    t.assert_true(local_window() == baseline);
    cache.clear(true);
    if (!t.assert_true(append(0,17,9))) return;
    t.assert_equal(llama_pos(16),cache.seq_pos_max(0));
    t.assert_equal(size_t(17),stream->tokens());

    cache.clear(true);
    if (!t.assert_true(append(0,1329,3))) return;
    image_writer checkpoint;
    cache.state_write(checkpoint,0,LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    const auto checkpoint_window=local_window();
    if (!t.assert_true(append(1329,1027,3))) return;
    const auto full_before_restore=positions(*cache.get_base());
    image_reader partial(checkpoint.bytes);
    cache.state_read(partial,0,LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    t.assert_equal(checkpoint.bytes.size(),partial.n_bytes());
    t.assert_equal(size_t(2356),stream->tokens());
    t.assert_true(full_before_restore == positions(*cache.get_base()));
    t.assert_true(checkpoint_window == local_window());
    t.assert_equal(llama_pos(305),cache.get_swa()->seq_pos_min(0));
    t.assert_true(llama_hparams::is_masked_swa(model->hparams.n_swa,model->hparams.swa_type,304,1328));
    t.assert_true("a checkpoint without enough history cannot replay two tokens",!cache.seq_rm(0,1327,-1));
    t.assert_equal(size_t(2356),stream->tokens());
    t.assert_true(full_before_restore == positions(*cache.get_base()));
    t.assert_true(checkpoint_window == local_window());
    if (!t.assert_true(cache.get_swa()->seq_rm(0,305,306))) return;
    t.assert_true("a missing row visible to the next query still rejects",!cache.seq_rm(0,1328,-1));
    t.assert_equal(size_t(2356),stream->tokens());
    t.assert_true(full_before_restore == positions(*cache.get_base()));
    image_reader retry(checkpoint.bytes);
    cache.state_read(retry,0,LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    t.assert_true(checkpoint_window == local_window());
    if (!t.assert_true("a native partial checkpoint permits replay of its final token",cache.seq_rm(0,1328,-1))) return;
    t.assert_equal(size_t(1328),stream->tokens());
    if (!t.assert_true(append(1328,1,3))) return;
    t.assert_true("checkpoint replay recovers exact visible local KV",checkpoint_window == local_window());
}

// Measure native local attention before storage exists, then reuse one parent through both text phases.
static void test_gemma_phase_parent(testing & t, llama_model * model, bool shared_reads=false) {
    if (!t.assert_true(model && model->arch == LLM_ARCH_GEMMA4)) return;
    auto * dev=ggml_backend_dev_by_name("CUDA0");
    if (!t.assert_true(dev != nullptr)) return;
    ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr)), cpu(ggml_backend_cpu_init());
    auto * buft=llama_kv_stream_device_buffer_type(dev);
    std::vector<ggml_backend_t> backends{backend.get(),cpu.get()};
    std::vector<ggml_backend_buffer_type_t> types{buft,ggml_backend_cpu_buffer_type()};
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends.data(),types.data(),2,1024,false,true));
    if (!t.assert_true(bool(sched))) return;
    llama_memory_params params{};
    params.type_k=GGML_TYPE_Q8_0; params.type_v=GGML_TYPE_Q4_0;
    params.shared_device_memory_bytes=768u*1024u*1024u;
    params.kv_stream_max_rows=256; params.kv_stream_max_decode_rows=4;
    params.kv_stream_backend=backend.get(); params.defer_bounded_cache=true;
    const auto no_reuse=[](int32_t) { return int32_t(-1); };
    const uint32_t capacity=shared_reads ? 131072 : 32768;
    llama_kv_cache_iswa cache(*model,params.type_k,params.type_v,false,true,false,true,
        capacity,1,256,256,nullptr,nullptr,no_reuse,nullptr,&params);
    auto * local=cache.get_swa(); auto * stream=cache.get_base()->get_kv_stream();
    if (!t.assert_true(local->bounded_storage_pending() && stream)) return;
    auto * placeholder=local->get_k_storage(0);
    struct attention_graph {
        ggml_context_ptr ctx;
        ggml_cgraph * graph=nullptr;
        ggml_tensor * query=nullptr, * mask=nullptr, * output=nullptr;
    };
    const auto build=[&](uint32_t rows) {
        attention_graph graph;
        graph.ctx.reset(ggml_init({1024*1024,nullptr,true}));
        auto * k=local->get_k_storage(0); auto * v=local->get_v_storage(0);
        auto * keys=ggml_view_3d(graph.ctx.get(),k,256,k->ne[1],16,k->nb[1],ggml_row_size(k->type,256),0);
        auto * values=ggml_view_3d(graph.ctx.get(),v,256,v->ne[1],16,v->nb[1],ggml_row_size(v->type,256),0);
        graph.query=ggml_new_tensor_3d(graph.ctx.get(),GGML_TYPE_F32,256,rows,32);
        graph.mask=ggml_new_tensor_2d(graph.ctx.get(),GGML_TYPE_F16,k->ne[1],GGML_PAD(rows,32));
        ggml_set_input(graph.query); ggml_set_input(graph.mask);
        graph.output=ggml_flash_attn_ext(graph.ctx.get(),graph.query,keys,values,graph.mask,1.0f/16,0,0);
        ggml_prec_set_acc(graph.output,GGML_PREC_F32);
        graph.graph=ggml_new_graph_custom(graph.ctx.get(),64,false);
        ggml_build_forward_expand(graph.graph,graph.output);
        ggml_backend_sched_set_tensor_backend(sched.get(),graph.output,backend.get());
        return graph;
    };
    std::vector<size_t> measurements(4);
    for (size_t phase=0;phase<2;++phase) {
        ggml_backend_sched_reset(sched.get());
        auto graph=build(phase ? 1 : 256);
        ggml_backend_sched_reserve_size(sched.get(),graph.graph,measurements.data()+2*phase);
        t.assert_true(local->bounded_storage_pending() && !placeholder->data &&
            ggml_backend_buffer_get_size(placeholder->buffer) == 0);
        ggml_backend_sched_reset(sched.get());
    }
    llama_compute_workspace_plan plan;
    if (!t.assert_true(llama_compute_workspace_plan_make(types,measurements,2,plan))) return;
    auto owner=llama_context_memory::create(sched.get(),backends,plan,stream);
    if (!t.assert_true(bool(owner))) return;
    t.assert_equal(params.shared_device_memory_bytes,owner->shared_parent_capacity());
    llama_cparams cp{}; cp.n_ctx_seq=capacity; cp.n_ubatch=256; cp.n_seq_max=1; cp.flash_attn=true;
    llama_memory_bounded_kv_layout quote;
    if (!t.assert_true(llama_kv_cache_iswa::bounded_storage_requirements(model->hparams,params,cp,buft,{609,1},quote))) return;
    auto too_large=quote; too_large.allocation.bytes=params.shared_device_memory_bytes;
    t.assert_true(!owner->reserve_bounded_cache(too_large) && owner->valid() && !owner->has_bounded_cache());
    auto loan=owner->reserve_bounded_cache(quote);
    if (!t.assert_true(loan && owner->retire_graph())) return;
    ggml_backend_sched_reset(sched.get());
    if (!t.assert_true(local->bind_bounded_storage(loan->lease()))) return;
    auto * parent=owner->shared_parent();
    const auto parent_base=uintptr_t(ggml_backend_buffer_get_base(parent));
    std::vector<uintptr_t> addresses;
    for (const auto & layer : quote.layers) for (bool value : {false,true}) {
        auto * tensor=value ? local->get_v_storage(int32_t(layer.id)) : local->get_k_storage(int32_t(layer.id));
        addresses.push_back(uintptr_t(tensor->data));
        t.assert_true(ggml_backend_buffer_is_view(tensor->buffer) && uintptr_t(tensor->data) >= parent_base &&
            uintptr_t(tensor->data)-parent_base <= params.shared_device_memory_bytes-ggml_nbytes(tensor));
    }
    auto * v=local->get_v_storage(0);
    std::vector<float> values(ggml_nelements(v));
    for (size_t i=0;i<values.size();++i) values[i]=float(int(i%31)-15)*0.0625f;
    std::vector<uint8_t> encoded(ggml_nbytes(v));
    ggml_quantize_chunk(v->type,values.data(),encoded.data(),0,v->ne[1],v->ne[0],nullptr);
    ggml_backend_tensor_set(v,encoded.data(),0,encoded.size());
    const auto verify_local=[&] {
        size_t ordinal=0;
        for (const auto & layer : quote.layers) for (bool value : {false,true}) {
            auto * tensor=value ? local->get_v_storage(int32_t(layer.id)) : local->get_k_storage(int32_t(layer.id));
            t.assert_equal(addresses[ordinal++],uintptr_t(tensor->data));
            std::vector<uint8_t> data(ggml_nbytes(tensor));
            ggml_backend_tensor_get(tensor,data.data(),0,data.size());
            t.assert_true(value && layer.id == 0 ? data == encoded :
                std::all_of(data.begin(),data.end(),[](uint8_t byte) { return byte == 0; }));
        }
        t.assert_equal(uint32_t(0),local->get_cells(0).get_used());
    };
    verify_local();
    ggml_backend_buffer_clear(stream->buffer(),0);
    if (!t.assert_true(stream->restore(32000))) return;
    std::vector<float> prefill_output;
    llama_context_memory_diagnostics before{}, decode{};
    for (int step=0;step<3;++step) {
        const bool decoding=step == 1;
        const uint32_t rows=decoding ? 1 : 256;
        if (!t.assert_true(owner->retire_graph())) return;
        ggml_backend_sched_reset(sched.get());
        const auto phase=decoding ? llama_memory_text_phase::decode : llama_memory_text_phase::prefill;
        if (decoding) {
            const auto previous=owner->text_phase();
            {
                serial_parent_view_failure fault(parent,1);
                t.assert_true(owner->signal_text_phase({phase,rows,true,true,false}).status ==
                    llama_memory_text_phase_status::transition_failed);
                t.assert_true(fault.calls > 0);
            }
            t.assert_true(owner->valid() && stream->complete());
            t.assert_equal(previous.revision,owner->text_phase().revision);
            t.assert_equal(before.kv_pool_bytes,stream->pool_grant_bytes());
            t.assert_equal(size_t(32000),stream->tokens());
            verify_local();
        }
        if (!t.assert_true(owner->signal_text_phase({phase,rows,true,true,false}).status == llama_memory_text_phase_status::changed)) return;
        if (!t.assert_true(owner->prepare_serial_target())) return;
        auto graph=build(rows);
        if (!t.assert_true(ggml_backend_sched_alloc_graph(sched.get(),graph.graph))) return;
        std::vector<float> query(ggml_nelements(graph.query),0.125f);
        std::vector<ggml_fp16_t> mask(ggml_nelements(graph.mask),ggml_fp32_to_fp16(0));
        ggml_backend_tensor_set(graph.query,query.data(),0,query.size()*sizeof(float));
        ggml_backend_tensor_set(graph.mask,mask.data(),0,mask.size()*sizeof(ggml_fp16_t));
        if (!t.assert_true("native local attention executes after graph retirement",owner->compute_async(graph.graph) == GGML_STATUS_SUCCESS)) return;
        owner->synchronize();
        std::vector<float> output(ggml_nelements(graph.output));
        ggml_backend_tensor_get(graph.output,output.data(),0,output.size()*sizeof(float));
        t.assert_true(std::all_of(output.begin(),output.end(),[](float value) { return std::isfinite(value); }));
        if (step == 0) prefill_output=output;
        t.assert_true(std::any_of(output.begin(),output.end(),[](float value) { return value != 0; }));
        if (step == 2) t.assert_true(std::memcmp(output.data(),prefill_output.data(),output.size()*sizeof(float)) == 0);
        verify_local();
        llama_context_memory_diagnostics diagnostics;
        if (!t.assert_true(owner->diagnostics(diagnostics))) return;
        t.assert_equal(quote.allocation.bytes,diagnostics.persistent_cache_bytes);
        t.assert_equal(params.shared_device_memory_bytes,diagnostics.parent_bytes);
        t.assert_equal(diagnostics.parent_bytes,diagnostics.workspace_bytes+diagnostics.kv_pool_bytes+
            diagnostics.kv_writer_bytes+diagnostics.kv_attention_bytes+diagnostics.persistent_cache_bytes+diagnostics.unused_bytes);
        t.assert_true(owner->shared_parent() == parent && owner->covers_workspace(plan));
        if (step == 0) before=diagnostics;
        if (step == 1) decode=diagnostics;
        if (step == 2) t.assert_equal(before.kv_pool_bytes,diagnostics.kv_pool_bytes);
        if (!t.assert_true(owner->retire_graph())) return;
        ggml_backend_sched_reset(sched.get());
    }
    t.assert_true(decode.kv_pool_bytes > before.kv_pool_bytes && decode.workspace_bytes < before.workspace_bytes);
    t.out << "Gemma common parent bytes=" << params.shared_device_memory_bytes << " local=" << quote.allocation.bytes
        << " prefill pool=" << before.kv_pool_bytes << " decode pool=" << decode.kv_pool_bytes
        << " prefill resident/ring=" << before.resident_pages_per_layer << '/' << before.ring_slots
        << " decode resident/ring=" << decode.resident_pages_per_layer << '/' << decode.ring_slots << '\n';
    if (!shared_reads) return;
    const auto ids=cache.get_base()->get_layer_ids();
    if (!t.assert_true(!ids.empty())) return;
    const int32_t selected=int32_t(ids.back());
    const auto filter=[&](int32_t il) { return il == selected; };
    const auto share=[&](int32_t) { return selected; };
    llama_kv_cache reader(*model,model->hparams,params.type_k,params.type_v,false,true,true,
        capacity,1,256,0,LLAMA_SWA_TYPE_NONE,cache.get_base(),filter,nullptr,share);
    auto * original=reader.get_k_storage(selected);
    const size_t grants=stream->device_grant_bytes();
    t.assert_true(!reader.attach_shared_stream_reader(*owner,5));
    t.assert_true(reader.get_k_storage(selected) == original);
    if (!t.assert_true(reader.attach_shared_stream_reader(*owner,4))) return;
    t.assert_equal(grants,stream->device_grant_bytes());
    t.assert_true(!reader.attach_shared_stream_reader(*owner,4));
    t.assert_true(reader.get_k_storage(selected) != original &&
        reader.get_k_storage(selected)->data == original->data &&
        ggml_backend_buft_is_execution(ggml_backend_buffer_get_type(reader.get_k_storage(selected)->buffer)));
    bool mismatch=false;
    try {
        llama_kv_cache invalid(*model,model->hparams,params.type_k,GGML_TYPE_Q8_0,false,true,true,
            capacity,1,256,0,LLAMA_SWA_TYPE_NONE,cache.get_base(),filter,nullptr,share);
    } catch (const std::runtime_error & error) {
        mismatch=std::string(error.what()) == "shared KV source has incompatible cache geometry or format";
    }
    t.assert_true("shared cache cannot silently override requested quantization",mismatch);
    llama_kv_stream_host_layer host;
    if (!t.assert_true(stream->host()->layer(uint32_t(ids.size()-1),host))) return;
    ggml_backend_buffer_clear(stream->buffer(),0);
    for (bool value : {false,true}) {
        auto * root=value ? cache.get_base()->get_v_storage(selected) : original;
        std::vector<float> source(ggml_nelements(root));
        for (size_t i=0;i<source.size();++i) source[i]=float(int(i%37)-18)*0.0625f;
        ggml_quantize_chunk(root->type,source.data(),value ? host.v : host.k,0,root->ne[1],root->ne[0],nullptr);
    }
    const uint32_t active=128000;
    if (!t.assert_true(stream->restore(active) && owner->prepare_serial_decode() && reader.arm_shared_stream_reader())) return;
    llama_kv_stream_runtime_diagnostics runtime;
    if (!t.assert_true(stream->runtime_diagnostics(runtime))) return;
    const size_t layer_bytes=size_t(active)*(stream->host()->layout().k_token_bytes+stream->host()->layout().v_token_bytes);
    t.out << "borrowed source: rows=" << active << " layer bytes=" << layer_bytes << " entire pool=" << runtime.pool_bytes
        << " resident mean=" << runtime.resident_pages_per_layer << " ring slots=" << runtime.ring_slots << '\n';
    // Even the whole pool cannot contain this layer, so no resident/ring partition can acquire it at once.
    if (!t.assert_true("source history exceeds the entire physical KV pool",layer_bytes > runtime.pool_bytes)) return;
    llama_kv_cache::slot_info slots;
    slots.s0=slots.s1=0; slots.strm={0}; slots.idxs={{0}};
    for (bool value : {false,true}) {
        bool rejected=false;
        try {
            if (value) reader.cpy_v(nullptr,nullptr,nullptr,selected,slots);
            else reader.cpy_k(nullptr,nullptr,nullptr,selected,slots);
        } catch (const std::runtime_error & error) {
            rejected=std::string(error.what()) == "cannot write a borrowed KV source";
        }
        t.assert_true("shared reader cannot build a KV writer",rejected);
    }
    std::vector<uint8_t> saved_head(256);
    std::memcpy(saved_head.data(),host.v,saved_head.size());
    reader.clear(true);
    t.assert_true("clearing the reader does not clear its source",std::memcmp(saved_head.data(),host.v,saved_head.size()) == 0);
    t.assert_equal(size_t(active),stream->tokens());
    auto * root=reader.get_k_storage(selected);
    if (!t.assert_true(root->ne[0] == 2048 && root->ne[1] == capacity && reader.get_size() == capacity)) return;
    for (uint32_t rows : {1u,3u,4u}) {
        ggml_context_ptr ctx(ggml_init({65536,nullptr,true}));
        auto * q=ggml_new_tensor_3d(ctx.get(),GGML_TYPE_F32,512,rows,32);
        auto * mask=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F16,active,rows);
        auto * k=ggml_permute(ctx.get(),reader.get_k(ctx.get(),selected,active,slots),0,2,1,3);
        auto * v=ggml_permute(ctx.get(),reader.get_v(ctx.get(),selected,active,slots),0,2,1,3);
        auto * output=ggml_flash_attn_ext(ctx.get(),q,k,v,mask,1.0f/16,0,0);
        ggml_prec_set_acc(output,GGML_PREC_F32);
        ggml_backend_execution_set_external_workspace(output,true);
        auto * graph=ggml_new_graph_custom(ctx.get(),64,false); ggml_build_forward_expand(graph,output);
        auto * nk=ggml_new_tensor_2d(ctx.get(),params.type_k,2048,active);
        auto * nv=ggml_new_tensor_2d(ctx.get(),params.type_v,2048,active);
        auto * native_k=ggml_view_3d(ctx.get(),nk,512,active,4,nk->nb[1],ggml_row_size(nk->type,512),0);
        auto * native_v=ggml_view_3d(ctx.get(),nv,512,active,4,nv->nb[1],ggml_row_size(nv->type,512),0);
        auto * expected=ggml_flash_attn_ext(ctx.get(),q,native_k,native_v,mask,1.0f/16,0,0);
        ggml_prec_set_acc(expected,GGML_PREC_F32);
        auto * native_graph=ggml_new_graph_custom(ctx.get(),64,false); ggml_build_forward_expand(native_graph,expected);
        ggml_backend_buffer_ptr allocation(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()));
        if (!t.assert_true(bool(allocation))) return;
        std::vector<float> queries(ggml_nelements(q),0.125f);
        std::vector<ggml_fp16_t> masks(ggml_nelements(mask),ggml_fp32_to_fp16(0));
        ggml_backend_tensor_set(q,queries.data(),0,ggml_nbytes(q));
        ggml_backend_tensor_set(mask,masks.data(),0,ggml_nbytes(mask));
        ggml_backend_tensor_set(nk,host.k,0,ggml_nbytes(nk));
        ggml_backend_tensor_set(nv,host.v,0,ggml_nbytes(nv));
        if (!t.assert_equal(GGML_STATUS_SUCCESS,ggml_backend_graph_compute(backend.get(),native_graph))) return;
        std::vector<float> want(ggml_nelements(expected)),got(want.size());
        ggml_backend_tensor_get(expected,want.data(),0,ggml_nbytes(expected));
        for (int repeat=0;repeat<2;++repeat) {
            if (!t.assert_equal(GGML_STATUS_SUCCESS,ggml_backend_graph_compute(backend.get(),graph))) return;
            ggml_backend_tensor_get(output,got.data(),0,ggml_nbytes(output));
            t.assert_true("cache-view borrowed attention matches stock bits",std::memcmp(want.data(),got.data(),ggml_nbytes(output)) == 0);
            t.assert_equal(size_t(active),stream->tokens());
        }
        if (!t.assert_true(reader.release_shared_stream_reader())) return;
        t.assert_equal(GGML_STATUS_FAILED,ggml_backend_graph_compute(backend.get(),graph));
        ggml_backend_tensor_get(output,got.data(),0,ggml_nbytes(output));
        t.assert_true("unarmed shared reader preserves outputs",std::memcmp(want.data(),got.data(),ggml_nbytes(output)) == 0);
        if (!t.assert_true(reader.arm_shared_stream_reader())) return;
        t.assert_equal(GGML_STATUS_SUCCESS,ggml_backend_graph_compute(backend.get(),graph));
        ggml_backend_tensor_get(output,got.data(),0,ggml_nbytes(output));
        t.assert_true("shared reader rearm keeps stock bits",std::memcmp(want.data(),got.data(),ggml_nbytes(output)) == 0);
    }
    t.assert_true(reader.release_shared_stream_reader());
    const auto local_ids=local->get_layer_ids();
    if (!t.assert_true(local_ids.size() >= 3)) return;
    const int32_t local_source=int32_t(local_ids.back());
    auto * native_key=local->get_k_storage(local_source);
    auto * native_value=local->get_v_storage(local_source);
    std::vector<float> local_values(ggml_nelements(native_value));
    for (size_t i=0;i<local_values.size();++i) local_values[i]=float(int(i%31)-15)*0.0625f;
    std::vector<uint8_t> encoded_local(ggml_nbytes(native_value));
    ggml_quantize_chunk(native_value->type,local_values.data(),encoded_local.data(),0,
        native_value->ne[1],native_value->ne[0],nullptr);
    ggml_backend_tensor_set(native_value,encoded_local.data(),0,encoded_local.size());
    auto native_owner=local->retain_bounded_storage(610,1,capacity,256);
    if (!t.assert_true(bool(native_owner))) return;
    auto native_pin=native_owner->read(native_owner->identity());
    const auto local_filter=[&](int32_t il) {
        return std::find(local_ids.end()-3,local_ids.end(),uint32_t(il)) != local_ids.end();
    };
    const auto local_share=[&](int32_t) { return local_source; };
    llama_kv_cache local_reader(*model,model->hparams,params.type_k,params.type_v,false,true,true,
        local->get_size(),1,256,model->hparams.n_swa,model->hparams.swa_type,local,local_filter,nullptr,local_share);
    auto * original_local=local_reader.get_k_storage(local_source);
    llama_memory_bounded_kv_read empty;
    t.assert_true(!local_reader.attach_shared_bounded_reader(empty));
    t.assert_true(local_reader.get_k_storage(local_source) == original_local);
    if (!t.assert_true(local_reader.attach_shared_bounded_reader(native_pin))) return;
    t.assert_true(!local_reader.attach_shared_bounded_reader(native_pin));
    if (!t.assert_true(local_reader.arm_shared_bounded_reader(native_pin))) return;
    for (auto id : local_reader.get_layer_ids()) {
        auto * key=local_reader.get_k_storage(int32_t(id)),* value=local_reader.get_v_storage(int32_t(id));
        t.assert_true(key != native_key && value != native_value && key->view_src == native_key && value->view_src == native_value);
        t.assert_true(key->data == native_key->data && value->data == native_value->data);
        ggml_context_ptr ctx(ggml_init({65536,nullptr,true}));
        auto * q=ggml_new_tensor_3d(ctx.get(),GGML_TYPE_F32,256,1,32);
        auto * mask=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F16,local->get_size(),1);
        const auto attention=[&](ggml_tensor * k,ggml_tensor * v) {
            auto * keys=ggml_view_3d(ctx.get(),k,256,local->get_size(),16,k->nb[1],ggml_row_size(k->type,256),0);
            auto * values=ggml_view_3d(ctx.get(),v,256,local->get_size(),16,v->nb[1],ggml_row_size(v->type,256),0);
            auto * out=ggml_flash_attn_ext(ctx.get(),q,keys,values,mask,1.0f/16,0,0);
            ggml_prec_set_acc(out,GGML_PREC_F32);
            return out;
        };
        auto * result=attention(key,value),* reference=attention(native_key,native_value);
        auto * graph=ggml_new_graph_custom(ctx.get(),64,false);
        ggml_build_forward_expand(graph,result); ggml_build_forward_expand(graph,reference);
        ggml_backend_buffer_ptr allocation(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()));
        if (!t.assert_true(bool(allocation))) return;
        std::vector<float> query(ggml_nelements(q),0.125f);
        std::vector<ggml_fp16_t> masks(ggml_nelements(mask),ggml_fp32_to_fp16(0));
        ggml_backend_tensor_set(q,query.data(),0,ggml_nbytes(q));
        ggml_backend_tensor_set(mask,masks.data(),0,ggml_nbytes(mask));
        if (!t.assert_equal(GGML_STATUS_SUCCESS,ggml_backend_graph_compute(backend.get(),graph))) return;
        std::vector<float> got(ggml_nelements(result)),want(got.size());
        ggml_backend_tensor_get(result,got.data(),0,ggml_nbytes(result));
        ggml_backend_tensor_get(reference,want.data(),0,ggml_nbytes(reference));
        t.assert_true("three local readers preserve native attention bits",std::memcmp(want.data(),got.data(),ggml_nbytes(result)) == 0);
    }
    local_reader.clear(true);
    std::vector<uint8_t> after_local(encoded_local.size());
    ggml_backend_tensor_get(native_value,after_local.data(),0,after_local.size());
    t.assert_true(encoded_local == after_local);
    native_pin.reset();
    t.assert_equal(uint64_t(0),native_owner->begin_write(native_owner->identity()));
    {
        auto wrong_owner=local->retain_bounded_storage(611,1,capacity,256);
        if (!t.assert_true(bool(wrong_owner))) return;
        auto wrong=wrong_owner->read(wrong_owner->identity());
        t.assert_true(!local_reader.arm_shared_bounded_reader(wrong));
        t.assert_equal(uint64_t(0),native_owner->begin_write(native_owner->identity()));
    }
    local_reader.release_shared_bounded_reader();
    const auto write=native_owner->begin_write(native_owner->identity());
    if (!t.assert_true(write != 0 && native_owner->finish_write(write,true))) return;
    native_pin=native_owner->read(native_owner->identity());
    t.assert_true(local_reader.arm_shared_bounded_reader(native_pin));
    local_reader.release_shared_bounded_reader();
}

static void test_group_vision_loans(testing & t,llama_model * model) {
    auto cp=llama_context_default_params();cp.n_ctx=512;cp.n_batch=cp.n_ubatch=256;cp.n_seq_max=2;
    cp.kv_unified=true;cp.swa_full=false;cp.shared_device_memory_bytes=size_t(1024)*1024*1024;
    cp.type_k=GGML_TYPE_Q8_0;cp.type_v=GGML_TYPE_Q4_0;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
    auto group=llama_context_group::create(*model,cp);if (!t.assert_true(group!=nullptr)) return;
    if (!t.assert_true(!group->context() && !group->metadata_context()->get_compute_memory())) return;
    auto * parent=group->prepare_vision_parent();
    if (!t.assert_true("vision uses the group's only retained device parent",parent &&
            ggml_backend_buffer_get_size(parent)==cp.shared_device_memory_bytes && !group->context())) return;
    std::vector<ggml_backend_memory_lease_t> grants;
    struct release {std::vector<ggml_backend_memory_lease_t> & grants;~release() {for (auto * lease:grants) ggml_backend_memory_lease_free(lease);}} cleanup{grants};
    if (!t.assert_true(!group->lend_vision({SIZE_MAX,1},grants) && grants.empty() &&
            !group->lend_vision({0,64},grants) && grants.empty())) return;
    {
        serial_parent_view_failure fault(parent,2);
        if (!t.assert_true("a failed second vision view publishes no grants",!group->lend_vision({65536,131072},grants) &&
                grants.empty() && fault.calls==2)) return;
    }
    if (!t.assert_true(group->lend_vision({65536,131072},grants) && grants.size()==2)) return;
    const auto generation=ggml_backend_memory_lease_generation(grants[0]);
    ggml_backend_memory_region first{},second{};
    if (!t.assert_true(ggml_backend_memory_lease_get_region(grants[0],&first) &&
            ggml_backend_memory_lease_get_region(grants[1],&second) && first.offset+first.size<=second.offset &&
            second.offset+second.size<=cp.shared_device_memory_bytes)) return;
    for (auto * lease:grants) {
        if (!t.assert_true(ggml_backend_memory_lease_parent(lease)==parent)) return;
        if (!t.assert_true(ggml_backend_memory_lease_buffer(lease)!=nullptr)) return;
    }
    std::vector<ggml_backend_memory_lease_t> duplicate;release cleanup_duplicate{duplicate};
    if (!t.assert_true("live vision grants reject another phase borrow",!group->lend_vision({64},duplicate) && duplicate.empty())) return;
    auto input=llama_batch_init(2,0,1);
    struct release_batch {llama_batch & batch;~release_batch() {llama_batch_free(batch);}} release_input{input};input.n_tokens=2;
    for (int32_t row=0;row<2;++row) {input.token[row]=10+row;input.pos[row]=0;input.n_seq_id[row]=1;input.seq_id[row][0]=row;input.logits[row]=true;}
    llama_kv_stream_selection selected;
    if (!t.assert_true(llama_kv_stream_select({512,2},{{0,0},{1,0}},selected)==llama_kv_stream_selection_status::success &&
            !group->activate(selected,group->source_handles(),input,llama_memory_work_phase::target_prefill,1,1))) return;
    std::unique_ptr<std::remove_pointer_t<ggml_backend_memory_lease_t>,decltype(&ggml_backend_memory_lease_free)> retained(
        ggml_backend_memory_lease_retain(grants[0]),ggml_backend_memory_lease_free);
    for (auto * lease:grants) ggml_backend_memory_lease_free(lease);
    grants.clear();
    if (!t.assert_true("retained encoder references block text reactivation",!group->release_vision() &&
            !group->activate(selected,group->source_handles(),input,llama_memory_work_phase::target_prefill,1,2))) return;
    if (!t.assert_true(!group->prepare_vision_parent() && !group->reset_request(0) && !group->discard())) return;
    retained.reset();
    if (!t.assert_true(group->release_vision() && group->prepare_vision_parent()==parent &&
            group->lend_vision({65536},grants) && grants.size()==1 && ggml_backend_memory_lease_generation(grants[0])>generation)) return;
    for (auto * lease:grants) ggml_backend_memory_lease_free(lease);
    grants.clear();
    if (!t.assert_true(group->release_vision() && !group->lend_vision({cp.shared_device_memory_bytes,1},grants) && grants.empty() &&
            group->lend_vision({cp.shared_device_memory_bytes},grants) && grants.size()==1)) return;
    ggml_backend_memory_region whole{};
    if (!t.assert_true(ggml_backend_memory_lease_get_region(grants[0],&whole) && whole.offset==0 && whole.size==cp.shared_device_memory_bytes)) return;
    for (auto * lease:grants) ggml_backend_memory_lease_free(lease);
    grants.clear();
    if (!t.assert_true(group->release_vision() && !group->context() && !group->metadata_context()->get_compute_memory())) return;
    for (const auto & handle:group->source_handles()) {
        std::shared_ptr<llama_kv_stream_logical_cache> source;
        if (!t.assert_true(group->source_table()->get(handle,0,source) && source->ready() && source->tokens()==0)) return;
    }
    if (!t.assert_true(group->lend_vision({65536},grants) && grants.size()==1)) return;
    retained.reset(ggml_backend_memory_lease_retain(grants[0]));
    for (auto * lease:grants) ggml_backend_memory_lease_free(lease);
    grants.clear();group.reset();
    if (!t.assert_true("the final encoder reference retains the physical parent after group destruction",
            ggml_backend_memory_lease_parent(retained.get())==parent &&
            ggml_backend_buffer_get_size(ggml_backend_memory_lease_buffer(retained.get()))==65536)) return;
}

static void test_group_initialization(testing & t,llama_model * model) {
    if (!t.assert_true(model && !model->hparams.no_alloc)) return;
    auto cp=llama_context_default_params();
    cp.n_ctx=4096;cp.n_batch=cp.n_ubatch=256;cp.n_seq_max=2;
    cp.kv_unified=true;cp.swa_full=false;cp.n_rs_seq=0;
    cp.type_k=GGML_TYPE_Q8_0;cp.type_v=GGML_TYPE_Q4_0;
    cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.shared_device_memory_bytes=size_t(1024)*1024*1024;
    for (const std::vector<llama_seq_id> & selected : {std::vector<llama_seq_id>{0,1},std::vector<llama_seq_id>{1}}) {
        llama_context context(*model,cp,selected);
        t.assert_true(!context.is_measurement_only() && context.group_storage_pending());
        const auto resources=context.get_kv_resources();
        if (!t.assert_true(resources.full && resources.full->full_storage_pending() && !resources.full->get_kv_stream())) return;
        if (resources.recurrent) t.assert_true(resources.recurrent->current_storage_pending());
        if (resources.bounded) t.assert_true(resources.bounded->bounded_storage_pending());
        for (const auto & entry : context.get_memory()->memory_breakdown()) t.assert_equal(size_t(0),entry.second);
        auto * scheduler=context.get_sched();
        for (int i=0;i<ggml_backend_sched_get_n_backends(scheduler);++i)
            t.assert_equal(size_t(0),ggml_backend_sched_get_buffer_size(scheduler,ggml_backend_sched_get_backend(scheduler,i)));
        llama_batch empty{};
        t.assert_true(context.decode(empty)<0 && context.encode(empty)<0);
        t.assert_true(context.graph_compute(nullptr,false)==GGML_STATUS_FAILED);
        auto allocator=llama_batch_allocr(1);auto batch=allocator.ubatch_reserve(1,1);
        llama_seq_id id=selected.front();batch.pos[0]=0;batch.token[0]=1;batch.n_seq_id[0]=1;batch.seq_id[0]=&id;
        t.assert_true(resources.full->prepare({batch}).empty());
        auto input=llama_batch_init(int32_t(selected.size()),0,1);
        struct release_batch {llama_batch & batch;~release_batch() {llama_batch_free(batch);}} release{input};
        input.n_tokens=int32_t(selected.size());
        std::vector<llama_memory_phase_cache_identity> sources;
        for (size_t row=0;row<selected.size();++row) {
            input.token[row]=1;input.pos[row]=0;input.logits[row]=1;
            input.n_seq_id[row]=1;input.seq_id[row][0]=selected[row];
            sources.push_back({uint64_t(1700+selected[row]),1});
        }
        llama_context_group_workload_quote quote;
        if (!t.assert_true("pending group measures its selected native source identities",
                context.measure_target_workload(input,sources,llama_memory_work_phase::target_verify,71,quote))) return;
        t.assert_equal(uint32_t(selected.size()),quote.measurement.workload.selected_count);
        t.assert_equal(selected.size(),quote.measurement.workload.members.size());
        for (size_t row=0;row<sources.size();++row)
            t.assert_equal(sources[row].id,quote.measurement.workload.members[row].cache.id);
        t.assert_true(context.group_storage_pending() && context.decode(empty)<0);
        llama_context_group_plan plan;
        const auto shortage=context.quote_target_group(input,sources,{1800,1},llama_memory_work_phase::target_verify,73,0,plan);
        if (!t.assert_true(shortage.status==llama_context_memory_create_status::insufficient_budget && shortage.required_bytes>0)) return;
        const auto exact=context.quote_target_group(input,sources,{1800,1},llama_memory_work_phase::target_verify,73,shortage.required_bytes,plan);
        if (!t.assert_true(exact.status==llama_context_memory_create_status::success)) return;
        const auto one_short=context.quote_target_group(input,sources,{1800,1},llama_memory_work_phase::target_verify,
            74,shortage.required_bytes-1,plan);
        t.assert_true(one_short.status==llama_context_memory_create_status::insufficient_budget &&
            one_short.required_bytes==shortage.required_bytes && one_short.additional_bytes==1);
        t.assert_equal(uint64_t(73),plan.target.measurement.workload.id);
        t.out << "pending group selected=" << selected.size() << " current=" << plan.current_state_bytes
              << " SWA=" << plan.bounded.allocation.bytes << " minimum=" << shortage.required_bytes << std::endl;
        if (resources.bounded && !t.assert_true("bounded quota includes selected windows only",
                plan.bounded.cache.sequence_capacity==selected.size())) return;
        if (selected.size()==1) {
            input.seq_id[0][0]=0;
            t.assert_true(!context.measure_target_workload(input,sources,llama_memory_work_phase::target_verify,72,quote));
            t.assert_equal(uint64_t(71),quote.measurement.workload.id);
            std::vector<size_t> sizes{19};
            t.assert_true(!context.measure_workspace(2,2,2,true,sizes));
            t.assert_true(sizes==std::vector<size_t>{19});
        }
    }
    for (const std::vector<llama_seq_id> & selected : {std::vector<llama_seq_id>{},std::vector<llama_seq_id>{0,0},
            std::vector<llama_seq_id>{2},std::vector<llama_seq_id>{-1}}) {
        bool rejected=false;
        try {llama_context context(*model,cp,selected);} catch (const std::runtime_error &) {rejected=true;}
        t.assert_true(rejected);
    }
}

static void test_group_full_binding(testing & t,llama_model * model) {
    if (!t.assert_true(model && model->devices.size()==1)) return;
    auto cp=llama_context_default_params();
    cp.n_ctx=4096;cp.n_batch=cp.n_ubatch=256;cp.n_seq_max=2;cp.kv_unified=true;cp.swa_full=false;
    cp.type_k=GGML_TYPE_Q8_0;cp.type_v=GGML_TYPE_Q4_0;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.shared_device_memory_bytes=size_t(1024)*1024*1024;
    for (const std::vector<llama_seq_id> & selected : {std::vector<llama_seq_id>{0,1},std::vector<llama_seq_id>{1}}) {
        llama_context context(*model,cp,selected);
        auto * full=context.get_kv_resources().full;
        if (!t.assert_true(full && full->full_storage_pending())) return;
        auto * scheduler=context.get_sched();ggml_backend_t backend=nullptr;
        for (int i=0;i<ggml_backend_sched_get_n_backends(scheduler);++i) {
            auto * candidate=ggml_backend_sched_get_backend(scheduler,i);
            if (ggml_backend_get_device(candidate)==model->devices.front().dev) backend=candidate;
        }
        if (!t.assert_true(backend!=nullptr)) return;
        const auto layers=full->get_layer_ids();if (!t.assert_true(!layers.empty())) return;
        const auto first=layers.front();
        const ggml_kv_stream_shape shape{GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,model->hparams.n_embd_head_k(first),
            model->hparams.n_embd_head_v(first),model->hparams.n_head_kv(first),256,128};
        const auto get_partial=reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(model->devices.front().dev),"ggml_backend_kv_stream_partial_ops"));
        const auto * partial=get_partial ? get_partial() : nullptr;
        if (!t.assert_true(partial && partial->capabilities)) return;
        const auto capabilities=partial->capabilities(backend,shape.type_k,shape.type_v);
        const auto limits=context.get_cparams().stream_limits;
        std::shared_ptr<llama_kv_stream_slot_table> table(llama_kv_stream_slot_table::create(limits));
        std::vector<llama_kv_stream_slot_handle> handles(2);
        for (uint32_t slot=0;slot<2;++slot) {
            llama_kv_stream_host_config config{uint64_t(1900+slot),shape,capabilities,limits.slot_limit,uint32_t(layers.size())};
            auto reservation=llama_kv_stream_host_reservation::create(config,model->devices.front().dev);
            auto cache=llama_kv_stream_logical_cache::create(llama_kv_stream_host::from_reservation(std::move(reservation)));
            if (!t.assert_true(table && cache && table->bind(slot,{std::move(cache)},handles[slot]))) return;
        }
        std::vector<llama_kv_stream_selection_slot> candidates;
        std::vector<llama_kv_stream_slot_handle> chosen;
        for (auto slot : selected) {candidates.push_back({uint32_t(slot),0});chosen.push_back(handles[slot]);}
        llama_kv_stream_selection selection;llama_kv_stream_slot_projection projection;
        if (!t.assert_true(llama_kv_stream_select({limits.group_limit,selected.size()},candidates,selection)==
                llama_kv_stream_selection_status::success && table->project(selection,chosen,0,projection))) return;
        llama_kv_stream_group_model_config config{backend,table,shape,uint32_t(layers.size()),model->hparams.n_head(first),
            256,1,uint32_t(selected.size())};
        auto candidate=llama_kv_stream_group_model::create(config);
        if (!t.assert_true(candidate!=nullptr)) return;
        auto invalid=config;invalid.shape.type_v=GGML_TYPE_Q8_0;
        auto wrong=llama_kv_stream_group_model::create(invalid);
        auto * old=full->get_k_storage(first);
        if (!t.assert_true(wrong && !full->bind_group_model(wrong,projection) && wrong && full->get_k_storage(first)==old)) return;
        if (selected.size()==1) {
            llama_kv_stream_selection foreign_selection;llama_kv_stream_slot_projection foreign;
            if (!t.assert_true(llama_kv_stream_select({limits.group_limit,1},{{0,0}},foreign_selection)==
                    llama_kv_stream_selection_status::success && table->project(foreign_selection,{handles[0]},0,foreign))) return;
            auto foreign_model=llama_kv_stream_group_model::create(config);
            if (!t.assert_true("full cache binding rejects an unselected logical source",
                    foreign_model && !full->bind_group_model(foreign_model,foreign) && foreign_model && full->full_storage_pending())) return;
        }
        auto * expected=candidate.get();
        if (!t.assert_true(full->bind_group_model(candidate,projection) && !candidate && full->get_group_model()==expected)) return;
        t.assert_true(!full->full_storage_pending() && context.group_storage_pending());
        for (size_t i=0;i<layers.size();++i) {
            llama_kv_stream_host_layer planes;if (!t.assert_true(expected->layer(uint32_t(i),planes))) return;
            t.assert_true(full->get_k_storage(layers[i])->data==planes.k && full->get_v_storage(layers[i])->data==planes.v);
            t.assert_true(full->get_k_storage(layers[i])->buffer==expected->buffer());
        }
        for (const auto & entry : full->memory_breakdown()) t.assert_equal(size_t(0),entry.second);
        llama_batch empty{};t.assert_true(context.decode(empty)<0 && context.encode(empty)<0);
        auto allocator=llama_batch_allocr(1);auto batch=allocator.ubatch_reserve(1,1);llama_seq_id id=selected.front();
        batch.n_seq_id[0]=1;batch.seq_id[0]=&id;batch.pos[0]=0;batch.token[0]=1;
        t.assert_true(!full->prepare({batch}).empty());
        if (selected.size()==1) {id=0;t.assert_true(full->prepare({batch}).empty());}
        t.assert_true(!full->bind_group_model(wrong,projection) && wrong);
    }
}

static void test_live_group_context(testing & t,llama_model * model) {
    using arena_ptr=std::unique_ptr<ggml_backend_memory_arena,decltype(&ggml_backend_memory_arena_free)>;
    using lease_ptr=std::unique_ptr<ggml_backend_memory_lease,decltype(&ggml_backend_memory_lease_free)>;
    if (!t.assert_true(model && model->devices.size()==1)) return;
    auto cp=llama_context_default_params();
    cp.n_ctx=4096;cp.n_batch=cp.n_ubatch=256;cp.n_seq_max=2;cp.kv_unified=true;cp.swa_full=false;
    cp.type_k=GGML_TYPE_Q8_0;cp.type_v=GGML_TYPE_Q4_0;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.n_threads=cp.n_threads_batch=4;
    const size_t vocabulary=model->vocab.n_tokens();
    for (const std::vector<llama_seq_id> & selected : {std::vector<llama_seq_id>{0,1},std::vector<llama_seq_id>{1}}) {
        auto batch=llama_batch_init(int32_t(selected.size()),0,1);
        struct release_batch {llama_batch & batch;~release_batch() {llama_batch_free(batch);}} release{batch};
        batch.n_tokens=int32_t(selected.size());
        const auto input=[&](int round) {
            for (size_t row=0;row<selected.size();++row) {
                batch.token[row]=llama_token(10+selected[row]*10+round);batch.pos[row]=round;
                batch.n_seq_id[row]=1;batch.seq_id[row][0]=selected[row];batch.logits[row]=1;
            }
        };
        std::vector<std::vector<float>> expected;
        std::vector<std::array<std::vector<uint8_t>,2>> expected_state;
        {
            llama_context native(*model,cp);
            for (int round=0;round<4;++round) {
                input(round);if (!t.assert_true(native.decode(batch)==0)) return;
                for (size_t row=0;row<selected.size();++row) {
                    const auto * values=llama_get_logits_ith(&native,int32_t(row));
                    if (!t.assert_true(values!=nullptr)) return;
                    expected.emplace_back(values,values+vocabulary);
                }
            }
            const auto roles=native.get_kv_resources();
            for (auto slot : selected) {
                image_writer recurrent,bounded;
                if (roles.recurrent) roles.recurrent->state_write(recurrent,slot,0);
                if (roles.bounded) roles.bounded->state_write(bounded,slot,0);
                expected_state.push_back({std::move(recurrent.bytes),std::move(bounded.bytes)});
            }
        }
        t.out << "live group: native reference complete, selected=" << selected.size() << std::endl;
        cp.shared_device_memory_bytes=size_t(1024)*1024*1024;
        std::unique_ptr<llama_context> owned_context;
        std::unique_ptr<llama_context_memory> workspace;
        arena_ptr parent(nullptr,ggml_backend_memory_arena_free),graph_arena(nullptr,ggml_backend_memory_arena_free),
            host_arena(nullptr,ggml_backend_memory_arena_free);
        std::vector<lease_ptr> leases;
        std::vector<std::shared_ptr<void>> prepared;
        std::vector<llama_kv_stream_group_reader_source> sources;
        std::unique_ptr<llama_memory_phase_owner> owner;
        std::shared_ptr<llama_kv_stream_writer> writer;
        std::shared_ptr<llama_kv_stream_group_producer> producer;
        owned_context=std::make_unique<llama_context>(*model,cp,selected);
        auto & context=*owned_context;
        struct cleanup_guard {std::function<void()> cleanup;~cleanup_guard() {cleanup();}} cleanup{[&] {
            t.out << "live group: cleanup, assertions=" << t.assertions << " failures=" << t.failures << std::endl;
            context.synchronize();
            if (context.get_kv_resources().full && context.get_kv_resources().full->get_group_model())
                context.get_kv_resources().full->get_group_model()->abort();
            owner.reset();producer.reset();writer.reset();workspace.reset();sources.clear();prepared.clear();owned_context.reset();
        }};
        auto * scheduler=context.get_sched();std::vector<ggml_backend_t> backends;
        ggml_backend_t backend=nullptr;size_t gpu=0;
        for (int i=0;i<ggml_backend_sched_get_n_backends(scheduler);++i) {
            auto * candidate=ggml_backend_sched_get_backend(scheduler,i);backends.push_back(candidate);
            if (ggml_backend_get_device(candidate)==model->devices.front().dev) {backend=candidate;gpu=size_t(i);}
        }
        if (!t.assert_true(backend!=nullptr)) return;
        auto resources=context.get_kv_resources();auto * full=resources.full;
        const auto layers=full->get_layer_ids();if (!t.assert_true(!layers.empty())) return;
        const auto first=layers.front();
        const ggml_kv_stream_shape shape{GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,model->hparams.n_embd_head_k(first),
            model->hparams.n_embd_head_v(first),model->hparams.n_head_kv(first),256,128};
        auto * registry=ggml_backend_dev_backend_reg(model->devices.front().dev);
        const auto get_partial=reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(registry,"ggml_backend_kv_stream_partial_ops"));
        const auto get_copy=reinterpret_cast<ggml_kv_stream_copy_ops_get>(ggml_backend_reg_get_proc_address(registry,"ggml_backend_kv_stream_copy_ops"));
        const auto * partial=get_partial ? get_partial() : nullptr;const auto * copies=get_copy ? get_copy() : nullptr;
        if (!t.assert_true(partial && copies && copies->prepare && copies->free_prepared)) return;
        const auto limits=context.get_cparams().stream_limits;
        std::shared_ptr<llama_kv_stream_slot_table> table(llama_kv_stream_slot_table::create(limits));
        std::vector<llama_kv_stream_slot_handle> handles(2),chosen;
        for (uint32_t slot=0;slot<2;++slot) {
            llama_kv_stream_host_config config{uint64_t(2900+slot),shape,partial->capabilities(backend,shape.type_k,shape.type_v),
                limits.slot_limit,uint32_t(layers.size())};
            std::shared_ptr<llama_kv_stream_logical_cache> cache(llama_kv_stream_logical_cache::create(llama_kv_stream_host::from_reservation(
                llama_kv_stream_host_reservation::create(config,model->devices.front().dev,65536))));
            if (!t.assert_true(table && cache && table->bind(slot,{cache},handles[slot]))) return;
        }
        std::vector<llama_kv_stream_selection_slot> candidates;
        for (auto slot : selected) {candidates.push_back({uint32_t(slot),0});chosen.push_back(handles[slot]);}
        auto projection=std::make_shared<llama_kv_stream_slot_projection>();llama_kv_stream_selection selection;
        if (!t.assert_true(llama_kv_stream_select({limits.group_limit,selected.size()},candidates,selection)==
                llama_kv_stream_selection_status::success && table->project(selection,chosen,0,*projection))) return;
        std::vector<llama_memory_phase_cache_identity> identities;
        for (const auto & source : projection->sources) identities.push_back({source.id,source.generation});
        t.out << "live group: quote" << std::endl;
        input(0);llama_context_group_plan quote;
        if (!t.assert_true(context.quote_target_group(batch,identities,{3000,1},llama_memory_work_phase::target_verify,
                1,cp.shared_device_memory_bytes,quote).status==llama_context_memory_create_status::success)) return;
        t.out << "live group: prepare native copies" << std::endl;
        for (size_t i=0;i<selected.size();++i) {
            prepared.emplace_back(copies->prepare(backend,3,false),copies->free_prepared);
            if (!t.assert_true(prepared.back()!=nullptr)) return;
        }
        parent.reset(ggml_backend_memory_arena_new(quote.target.buffer_type,cp.shared_device_memory_bytes));
        if (!t.assert_true(parent && ggml_backend_memory_arena_begin(parent.get(),0))) return;
        size_t cursor=0;
        const auto reserve=[&](uint64_t id,size_t bytes) {
            const auto alignment=quote.target.measurement.alignment;cursor=(cursor+alignment-1)/alignment*alignment;
            if (!bytes || !ggml_backend_memory_arena_reserve_at(parent.get(),id,cursor,bytes,alignment,
                    GGML_BACKEND_MEMORY_REGION_PERSISTENT,nullptr)) return false;
            cursor+=bytes;return true;
        };
        if (quote.current_state_bytes && !t.assert_true(reserve(101,quote.current_state_bytes))) return;
        if (quote.bounded.allocation.bytes && !t.assert_true(reserve(102,quote.bounded.allocation.bytes))) return;
        const auto graph_bytes=*std::max_element(quote.target.measurement.graph_bytes.begin(),quote.target.measurement.graph_bytes.end());
        if (!t.assert_true(reserve(201,graph_bytes) && reserve(202,quote.target.writer_bytes) &&
                reserve(203,quote.target.attention_bytes) && reserve(204,quote.target.encoded_stage_bytes))) return;
        for (size_t i=0;i<selected.size();++i) if (!t.assert_true(reserve(301+i,quote.target.pool_minimum_bytes/selected.size()))) return;
        if (!t.assert_true(ggml_backend_memory_arena_commit(parent.get()))) return;
        const auto lease=[&](uint64_t id) {
            leases.emplace_back(ggml_backend_memory_arena_acquire(parent.get(),id),ggml_backend_memory_lease_free);
            return leases.back().get();
        };
        t.out << "live group: bind native state" << std::endl;
        if (resources.recurrent && !t.assert_true(resources.recurrent->bind_current_storage(lease(101)))) return;
        if (resources.bounded && !t.assert_true(resources.bounded->bind_bounded_storage(lease(102)))) return;
        auto graph=lease(201),encoding=lease(202),attention=lease(203),staging=lease(204);
        for (size_t i=0;i<selected.size();++i) {
            std::shared_ptr<llama_kv_stream_logical_cache> cache;if (!t.assert_true(table->get(chosen[i],0,cache))) return;
            sources.push_back({cache->host(),lease(301+i),3,0,prepared[i]});
        }
        t.out << "live group: bind model" << std::endl;
        auto group_model=llama_kv_stream_group_model::create({backend,table,shape,uint32_t(layers.size()),model->hparams.n_head(first),
            256,1,uint32_t(selected.size())});
        if (!t.assert_true(group_model && full->bind_group_model(group_model,*projection))) return;
        writer=llama_kv_stream_writer::create(backend,ggml_backend_memory_lease_buffer(encoding),quote.target.writer_bytes,shape,256);
        producer=llama_kv_stream_group_producer::create(table,writer,256,1,uint32_t(selected.size()));
        const auto admitted=[&] {return context.group_submission_allowed();};
        if (!t.assert_true(full->get_group_model()->bind(producer,[&](uint32_t layer,ggml_tensor * op) {
            auto frame=sources;
            for (size_t i=0;i<frame.size();++i) {
                std::shared_ptr<llama_kv_stream_logical_cache> cache;
                if (!table->get(chosen[i],0,cache)) return false;
                frame[i].tokens=cache->frontiers().reserved;
            }
            auto reader=llama_kv_stream_group_reader::create(backend,frame,staging,[&](uint32_t reading) {
                if (!admitted() || !producer->layer_ready(reading)) return false;
                for (size_t i=0;i<frame.size();++i) {
                    std::shared_ptr<llama_kv_stream_logical_cache> cache;
                    if (!table->get(chosen[i],0,cache) || cache->content()->generation()!=projection->sources[i].generation ||
                            cache->tokens()!=projection->sources[i].tokens || cache->frontiers().reserved!=frame[i].tokens) return false;
                }
                return true;
            });
            if (!reader || op->src[1]->ne[1]<=0 || size_t(op->src[1]->ne[1])>projection->columns.size()) return false;
            const std::vector<ggml_kv_stream_gather_row> columns(projection->columns.begin(),projection->columns.begin()+op->src[1]->ne[1]);
            return reader->attention(layer,columns,op,attention,true);
        },admitted))) return;
        graph_arena.reset(ggml_backend_memory_arena_new_from_buffer(ggml_backend_memory_lease_buffer(graph)));
        if (!t.assert_true(graph_arena!=nullptr)) return;
        std::vector<llama_memory_workspace_group> groups;std::vector<llama_memory_arena_budget> budgets;
        std::vector<llama_memory_transition_arena> arenas;
        llama_memory_execution_plan plan;plan.stages.push_back({1,{},{}});
        for (size_t i=0;i<backends.size();++i) {
            const bool device=i==gpu;const size_t bytes=device ? graph_bytes : quote.target.host_compute_bytes;
            if (!bytes) continue;
            auto * type=ggml_backend_sched_get_buffer_type(scheduler,backends[i]);
            const auto allocation=device ? LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL : LLAMA_MEMORY_ALLOCATION_HOST_PINNED;
            const auto alignment=ggml_backend_buft_get_alignment(type);
            groups.push_back({{type,bytes,alignment,i},{uint64_t(701+i),uint64_t(i+1),allocation,llama_memory_content::discardable},{{1,bytes}}});
            plan.domains.push_back({uint64_t(i+1),allocation,LLAMA_MEMORY_CAPABILITY_BUFFER_VIEWS});
            budgets.push_back({uint64_t(i+1),allocation,bytes,alignment});
            if (!device) {if (!t.assert_true(!host_arena)) return;host_arena.reset(ggml_backend_memory_arena_new(type,bytes));}
            arenas.push_back({uint64_t(i+1),allocation,device ? graph_arena.get() : host_arena.get()});
        }
        t.out << "live group: external workspace" << std::endl;
        workspace=llama_context_memory::create_external_workspace(scheduler,backends,groups,admitted);
        if (!t.assert_true(workspace && workspace->external_workspace()->register_resources(plan,{1}))) return;
        llama_memory_transition_target target;target.plan=plan;target.stage=1;target.budgets=budgets;
        t.out << "live group: phase owner" << std::endl;
        owner=llama_memory_phase_owner::create({{llama_memory_work_phase::target_verify,target,uint32_t(selected.size()),1}},
            arenas,{workspace->external_workspace()},identities);
        if (!t.assert_true(owner!=nullptr)) return;
        t.out << "live group: enter measured profile" << std::endl;
        const auto entered=owner->enter(llama_memory_work_phase::target_verify,uint32_t(selected.size()),identities,1,1);
        if (!t.assert_true(entered.status==llama_memory_phase_owner_status::changed)) return;
        t.out << "live group: activate" << std::endl;
        if (!t.assert_true(context.attach_group_execution(workspace,projection,owner,llama_memory_work_phase::target_verify,1,1))) return;
        if (selected.size()==2) {
            auto invalid=llama_batch_init(2,0,1);release_batch release_invalid{invalid};invalid.n_tokens=2;
            for (int row=0;row<2;++row) {
                invalid.token[row]=llama_token(10+row);invalid.pos[row]=row;
                invalid.n_seq_id[row]=1;invalid.seq_id[row][0]=selected.front();invalid.logits[row]=1;
            }
            t.assert_true("per-source verification overflow rejects before admission",context.decode(invalid)<0);
            t.assert_true(!context.group_submission_allowed());
            for (const auto & handle : chosen) {
                std::shared_ptr<llama_kv_stream_logical_cache> cache;
                if (!t.assert_true(table->get(handle,0,cache))) return;
                t.assert_true(cache->ready() && cache->tokens()==0 && cache->identity().generation==1);
            }
        }
        for (int round=0;round<4;++round) {
            t.out << "live group: decode round=" << round << std::endl;
            input(round);if (!t.assert_true(context.decode(batch)==0)) return;
            for (size_t row=0;row<selected.size();++row) {
                const auto * values=llama_get_logits_ith(&context,int32_t(row));
                if (!t.assert_true(values && std::all_of(values,values+vocabulary,[](float value) {return std::isfinite(value);}))) return;
                if (!t.assert_true("group target preserves every native logit bit",!std::memcmp(values,
                        expected[size_t(round)*selected.size()+row].data(),vocabulary*sizeof(float)))) return;
            }
        }
        std::vector<llama_context_group_sequence_state> saved;
        if (!t.assert_true(context.save_group_sequences(saved))) return;
        if (!t.assert_equal(selected.size(),saved.size())) return;
        for (size_t i=0;i<saved.size();++i) {
            t.assert_true(saved[i].model==model && saved[i].sequence==selected[i] && saved[i].request_epoch==chosen[i].generation);
            t.assert_true(saved[i].cache_id==projection->sources[i].id && saved[i].cache_generation==projection->sources[i].generation && saved[i].tokens==4);
            t.assert_true("selected recurrent export preserves native bytes",saved[i].recurrent==expected_state[i][0]);
            t.assert_true("selected bounded export preserves native bytes",saved[i].bounded==expected_state[i][1]);
        }
        if (!t.assert_true(context.detach_group_execution())) return;
        t.assert_true(context.group_storage_pending() && !context.group_submission_allowed() && context.decode(batch)<0);
        if (resources.recurrent) t.assert_true(resources.recurrent->current_storage_pending());
        if (resources.bounded) t.assert_true(resources.bounded->bounded_storage_pending());
        std::vector<llama_context_group_sequence_state> untouched(1);untouched[0].sequence=-2;
        t.assert_true(!context.save_group_sequences(untouched) && untouched.size()==1 && untouched[0].sequence==-2);
        producer.reset();writer.reset();sources.clear();prepared.clear();leases.clear();graph_arena.reset();host_arena.reset();
        t.assert_true("the physical parent can be replanned after selection retirement",
            ggml_backend_memory_arena_begin(parent.get(),0) && ggml_backend_memory_arena_commit(parent.get()));
        cursor=0;
        if (!t.assert_true(ggml_backend_memory_arena_begin(parent.get(),0))) return;
        if (quote.current_state_bytes && !t.assert_true(reserve(901,quote.current_state_bytes))) return;
        if (quote.bounded.allocation.bytes && !t.assert_true(reserve(902,quote.bounded.allocation.bytes))) return;
        if (!t.assert_true(ggml_backend_memory_arena_commit(parent.get()))) return;
        if (resources.recurrent && !t.assert_true(resources.recurrent->bind_current_storage(lease(901)))) return;
        if (resources.bounded && !t.assert_true(resources.bounded->bind_bounded_storage(lease(902)))) return;
        --saved[0].cache_generation;
        t.assert_true(!context.restore_group_sequences(saved,*projection));
        ++saved[0].cache_generation;
        const auto order=saved[0].full_columns;
        if (!t.assert_equal(size_t(4),order.size())) return;
        for (int bad=0;bad<4;++bad) {
            auto & columns=saved[0].full_columns;
            if (bad==0) columns.pop_back();
            if (bad==1) columns[1]=columns[0];
            if (bad==2) columns[0]=UINT32_MAX;
            if (bad==3) std::swap(columns[0],columns[1]);
            t.assert_true(!context.restore_group_sequences(saved,*projection));
            columns=order;
        }
        if (!t.assert_true(context.restore_group_sequences(saved,*projection))) return;
        for (const auto & state : saved) {
            image_writer recurrent,bounded;
            if (resources.recurrent) resources.recurrent->state_write(recurrent,state.sequence,0);
            if (resources.bounded) resources.bounded->state_write(bounded,state.sequence,0);
            t.assert_true(recurrent.bytes==state.recurrent && bounded.bytes==state.bounded);
        }
        t.assert_true(context.group_storage_pending() && context.decode(batch)<0);
        if (resources.recurrent) t.assert_true(resources.recurrent->release_current_storage());
        if (resources.bounded) t.assert_true(resources.bounded->release_bounded_storage());
        leases.clear();
        t.assert_true(ggml_backend_memory_arena_begin(parent.get(),0) && ggml_backend_memory_arena_commit(parent.get()));
        cp.shared_device_memory_bytes=0;
    }
}

static bool make_group_fixture_sources(testing & t,llama_context & context,llama_model * model,
        std::shared_ptr<llama_kv_stream_slot_table> & table,std::vector<llama_kv_stream_slot_handle> & handles) {
    auto * full=context.get_kv_resources().full;
        const auto layers=full->get_layer_ids();if (!t.assert_true(!layers.empty())) return false;
        const auto first=layers.front();
        const ggml_kv_stream_shape shape{GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,model->hparams.n_embd_head_k(first),
            model->hparams.n_embd_head_v(first),model->hparams.n_head_kv(first),256,128};
        auto * scheduler=context.get_sched();ggml_backend_t backend=nullptr;
        for (int i=0;i<ggml_backend_sched_get_n_backends(scheduler);++i) {
            auto * candidate=ggml_backend_sched_get_backend(scheduler,i);
            if (ggml_backend_get_device(candidate)==model->devices.front().dev) backend=candidate;
        }
        if (!t.assert_true(backend!=nullptr)) return false;
        const auto get=reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(model->devices.front().dev),"ggml_backend_kv_stream_partial_ops"));
        const auto * partial=get ? get() : nullptr;if (!t.assert_true(partial!=nullptr)) return false;
        const auto limits=context.get_cparams().stream_limits;
        table=llama_kv_stream_slot_table::create(limits);
        handles.resize(2);
        for (uint32_t slot=0;slot<2;++slot) {
            llama_kv_stream_host_config config{uint64_t(3900+slot),shape,partial->capabilities(backend,shape.type_k,shape.type_v),
                limits.slot_limit,uint32_t(layers.size())};
            std::shared_ptr<llama_kv_stream_logical_cache> cache(llama_kv_stream_logical_cache::create(llama_kv_stream_host::from_reservation(
                llama_kv_stream_host_reservation::create(config,model->devices.front().dev,65536))));
            if (!t.assert_true(cache && table->bind(slot,{cache},handles[slot]))) return false;
        }
    return true;
}

static void test_owned_group_context(testing & t,llama_model * model,uint32_t queries=1,bool managed=false,bool final_output_only=false) {
    auto cp=llama_context_default_params();
    cp.n_ctx=4096;cp.n_batch=cp.n_ubatch=256;cp.n_seq_max=2;cp.kv_unified=true;cp.swa_full=false;
    cp.type_k=GGML_TYPE_Q8_0;cp.type_v=GGML_TYPE_Q4_0;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.n_threads=cp.n_threads_batch=4;
    const size_t vocabulary=model->vocab.n_tokens();
    const std::vector<std::vector<llama_seq_id>> selections=queries==1 ?
        std::vector<std::vector<llama_seq_id>>{{0,1},{1}} : std::vector<std::vector<llama_seq_id>>{{1}};
    const auto phase=queries==1 ? llama_memory_work_phase::target_verify : llama_memory_work_phase::target_prefill;
    for (const auto & selected : selections) {
        auto batch=llama_batch_init(int32_t(selected.size()*queries),0,1);
        struct release_batch {llama_batch & batch;~release_batch() {llama_batch_free(batch);}} release{batch};
        batch.n_tokens=int32_t(selected.size()*queries);
        const auto input=[&](int step) {for (size_t source=0;source<selected.size();++source) for (uint32_t query=0;query<queries;++query) {
            const size_t row=source*queries+query;
            batch.token[row]=llama_token(10+selected[source]*10+step*queries+query);batch.pos[row]=step*queries+query;
            batch.n_seq_id[row]=1;batch.seq_id[row][0]=selected[source];batch.logits[row]=query+1==queries && (!final_output_only || step%4==3);
        }};
        std::vector<std::vector<float>> reference, restored_reference;
        std::vector<std::vector<uint8_t>> native_states(selected.size());
        {
            auto native_params=cp;native_params.shared_device_memory_bytes=0;
            llama_context native(*model,native_params);
            for (int step=0;step<4;++step) {
                input(step);if (!t.assert_true(native.decode(batch)==0)) return;
                for (size_t row=0;row<selected.size();++row) {
                    if (!batch.logits[row*queries+queries-1]) {reference.emplace_back();continue;}
                    const auto * logits=llama_get_logits_ith(&native,int32_t(row*queries+queries-1));
                    if (!t.assert_true(logits!=nullptr)) return;
                    reference.emplace_back(logits,logits+vocabulary);
                }
            }
            for (size_t row=0;row<selected.size();++row) {
                auto & state=native_states[row];
                state.resize(llama_state_seq_get_size(&native,selected[row]));
                if (!t.assert_true(!state.empty() && llama_state_seq_get_data(&native,state.data(),state.size(),selected[row])==state.size())) return;
            }
        }
        {
            auto native_params=cp;native_params.shared_device_memory_bytes=0;
            llama_context native(*model,native_params);
            for (size_t row=0;row<selected.size();++row) {
                const auto & state=native_states[row];
                if (!t.assert_equal(state.size(),llama_state_seq_set_data(&native,state.data(),state.size(),selected[row]))) return;
            }
            for (int step=4;step<8;++step) {
                input(step);if (!t.assert_true(native.decode(batch)==0)) return;
                for (size_t row=0;row<selected.size();++row) {
                    if (!batch.logits[row*queries+queries-1]) {restored_reference.emplace_back();continue;}
                    const auto * logits=llama_get_logits_ith(&native,int32_t(row*queries+queries-1));
                    if (!t.assert_true(logits!=nullptr)) return;
                    restored_reference.emplace_back(logits,logits+vocabulary);
                }
            }
        }
        cp.shared_device_memory_bytes=size_t(1024)*1024*1024;
        llama_context context(*model,cp,selected);
        std::shared_ptr<llama_kv_stream_slot_table> table;
        std::vector<llama_kv_stream_slot_handle> handles,chosen;
        if (!make_group_fixture_sources(t,context,model,table,handles)) return;
        const auto limits=table->limits();
        std::vector<llama_kv_stream_selection_slot> candidates;
        for (auto slot : selected) {candidates.push_back({uint32_t(slot),0});chosen.push_back(handles[slot]);}
        llama_kv_stream_selection selection;auto projection=std::make_shared<llama_kv_stream_slot_projection>();
        if (!t.assert_true(llama_kv_stream_select({limits.group_limit,selected.size()},candidates,selection)==
                llama_kv_stream_selection_status::success && table->project(selection,chosen,0,*projection))) return;
        input(0);llama_context_memory_create_result admission;
        // Quote the final output row as well as the preceding cache-only microbatches.
        if (final_output_only) batch.logits[queries-1]=1;
        std::vector<llama_memory_phase_cache_identity> identities;
        for (const auto & source : projection->sources) identities.push_back({source.id,source.generation});
        llama_context_group_plan quote;
        const auto quoted=context.quote_target_group(batch,identities,{4900,1},phase,1,cp.shared_device_memory_bytes,quote);
        if (quoted.status!=llama_context_memory_create_status::success) {
            std::fprintf(stderr,"group quote status=%d required=%zu available=%zu additional=%zu queries=%u selected=%zu\n",
                int(quoted.status),quoted.required_bytes,quoted.available_bytes,quoted.additional_bytes,queries,selected.size());
        }
        if (!t.assert_equal(int(llama_context_memory_create_status::success),int(quoted.status))) return;
        const auto alignment=quote.target.measurement.alignment;
        const auto aligned=[&](size_t bytes) {return (bytes+alignment-1)/alignment*alignment;};
        size_t selected_live=aligned(quote.target.writer_bytes)+aligned(quote.target.attention_bytes)+
            aligned(quote.target.encoded_stage_bytes)+aligned(quote.target.publication_bytes)+
            selected.size()*aligned(quote.target.pool_minimum_bytes/selected.size());
        t.assert_equal("pool quota counts only chosen sources",selected_live,quote.target.measurement.live_device_bytes);
        std::unique_ptr<llama_context_group> group;
        if (managed) {
            group=llama_context_group::create(*model,cp,table);
            if (!t.assert_true(group && group->activate(selection,chosen,batch,phase,1,1,&admission))) return;
        } else if (!t.assert_true(context.initialize_group_execution(table,projection,batch,phase,
                1,1,nullptr,{},&admission))) return;
        auto * execution=managed ? group->context() : &context;
        auto * parent=execution->group_parent();if (!t.assert_true(parent!=nullptr)) return;
        ggml_backend_buffer_ptr retained(ggml_backend_buffer_retain(parent));
        t.assert_equal(cp.shared_device_memory_bytes,ggml_backend_buffer_get_size(parent));
        for (int step=0;step<4;++step) {
            input(step);if (!t.assert_true(execution->decode(batch)==0)) return;
            for (size_t row=0;row<selected.size();++row) {
                if (!batch.logits[row*queries+queries-1]) continue;
                const auto * logits=llama_get_logits_ith(execution,int32_t(row*queries+queries-1));
                if (!t.assert_true(logits && !std::memcmp(logits,reference[size_t(step)*selected.size()+row].data(),
                        vocabulary*sizeof(float)))) return;
            }
        }
        std::vector<llama_context_group_sequence_state> states;
        if (!t.assert_true(execution->save_group_sequences(states) && states.size()==selected.size())) return;
        if (managed) {
            if (!t.assert_true(group->park() && !group->context())) return;
        } else if (!t.assert_true(context.detach_group_execution() && !context.group_parent() && context.group_storage_pending())) return;
        std::unique_ptr<ggml_backend_memory_arena,decltype(&ggml_backend_memory_arena_free)> reuse(
            ggml_backend_memory_arena_new_from_buffer(retained.get()),ggml_backend_memory_arena_free);
        if (!t.assert_true(reuse && ggml_backend_memory_arena_begin(reuse.get(),0) && ggml_backend_memory_arena_commit(reuse.get()))) return;
        reuse.reset();
        for (auto & candidate : candidates) candidate.history_tokens=4*queries;
        auto restored_projection=std::make_shared<llama_kv_stream_slot_projection>();
        if (!t.assert_true(llama_kv_stream_select({limits.group_limit,selected.size()},candidates,selection)==
                llama_kv_stream_selection_status::success && table->project(selection,chosen,0,*restored_projection))) return;
        llama_context restored(*model,cp,selected);
        input(4);
        if (final_output_only) batch.logits[queries-1]=1;
        if (!managed) {
            ggml_backend_buffer_ptr short_parent(ggml_backend_buffer_view(retained.get(),0,cp.shared_device_memory_bytes-1));
            if (!t.assert_true(short_parent!=nullptr)) return;
            t.assert_true(!restored.initialize_group_execution(table,restored_projection,batch,phase,
                1,2,short_parent.get(),states,&admission));
            t.assert_equal("a rejected physical parent is not a successful activation",
                int(llama_context_memory_create_status::invalid_configuration),int(admission.status));
            t.assert_true(restored.group_storage_pending() && !restored.group_parent());
        }
        if (managed) {
            if (!t.assert_true(group->activate(selection,chosen,batch,phase,1,2,&admission))) return;
        } else if (!t.assert_true(restored.initialize_group_execution(table,restored_projection,batch,phase,
                1,2,retained.get(),states,&admission))) return;
        auto * resumed=managed ? group->context() : &restored;
        if (!t.assert_true(resumed->group_parent()==retained.get())) return;
        for (int step=4;step<8;++step) {
            input(step);if (!t.assert_true(resumed->decode(batch)==0)) return;
            for (size_t row=0;row<selected.size();++row) {
                if (!batch.logits[row*queries+queries-1]) continue;
                const auto * logits=llama_get_logits_ith(resumed,int32_t(row*queries+queries-1));
                if (!t.assert_true(logits && !std::memcmp(logits,restored_reference[size_t(step-4)*selected.size()+row].data(),
                        vocabulary*sizeof(float)))) return;
            }
        }
        t.assert_true(managed ? group->park() : restored.detach_group_execution());
        cp.shared_device_memory_bytes=0;
    }
}

static void test_group_reselection(testing & t,llama_model * model) {
    auto cp=llama_context_default_params();
    cp.n_ctx=4096;cp.n_batch=cp.n_ubatch=256;cp.n_seq_max=2;cp.kv_unified=true;cp.swa_full=false;
    cp.type_k=GGML_TYPE_Q8_0;cp.type_v=GGML_TYPE_Q4_0;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.n_threads=cp.n_threads_batch=4;
    const std::vector<std::vector<llama_seq_id>> rounds{{0,1},{1},{0,1}};
    auto batch=llama_batch_init(2,0,1);
    struct release_batch {llama_batch & batch;~release_batch() {llama_batch_free(batch);}} release{batch};
    const auto input=[&](const std::vector<llama_seq_id> & selected,const std::array<uint32_t,2> & past) {
        batch.n_tokens=int32_t(selected.size());
        for (size_t row=0;row<selected.size();++row) {
            const auto sequence=selected[row];
            batch.token[row]=llama_token(10+sequence*10+past[sequence]);batch.pos[row]=past[sequence];
            batch.n_seq_id[row]=1;batch.seq_id[row][0]=sequence;batch.logits[row]=1;
        }
    };
    std::vector<std::vector<std::vector<float>>> reference;
    std::array<std::vector<uint8_t>,2> native_states;
    std::array<uint32_t,2> past{};
    const auto vocabulary=model->vocab.n_tokens();
    for (const auto & selected : rounds) {
        llama_context native(*model,cp);
        for (auto sequence : selected) if (!native_states[sequence].empty()) {
            const auto & state=native_states[sequence];
            if (!t.assert_equal(state.size(),llama_state_seq_set_data(&native,state.data(),state.size(),sequence))) return;
        }
        std::vector<std::vector<float>> expected;
        for (uint32_t step=0;step<4;++step) {
            input(selected,past);if (!t.assert_true(native.decode(batch)==0)) return;
            for (size_t row=0;row<selected.size();++row) {
                const auto * logits=llama_get_logits_ith(&native,int32_t(row));
                if (!t.assert_true(logits!=nullptr)) return;
                expected.emplace_back(logits,logits+vocabulary);++past[selected[row]];
            }
        }
        for (auto sequence : selected) {
            auto & state=native_states[sequence];state.resize(llama_state_seq_get_size(&native,sequence));
            if (!t.assert_true(!state.empty() && llama_state_seq_get_data(&native,state.data(),state.size(),sequence)==state.size())) return;
        }
        reference.push_back(std::move(expected));
    }
    std::vector<float> peer_continuation;
    {
        llama_context native(*model,cp);
        const auto & state=native_states[0];
        if (!t.assert_equal(state.size(),llama_state_seq_set_data(&native,state.data(),state.size(),0))) return;
        input({0},past);if (!t.assert_equal(0,native.decode(batch))) return;
        const auto * logits=llama_get_logits_ith(&native,0);
        if (!t.assert_true(logits!=nullptr)) return;
        peer_continuation.assign(logits,logits+vocabulary);
    }
    cp.shared_device_memory_bytes=size_t(1024)*1024*1024;
    auto group=llama_context_group::create(*model,cp);
    if (!t.assert_true(group && !group->context() && group->metadata_context() && group->metadata_context()->group_storage_pending())) return;
    const auto table=group->source_table();const auto handles=group->source_handles();
    if (!t.assert_true(table && handles.size()==2)) return;
    std::array<std::shared_ptr<llama_kv_stream_host>,2> backing;
    for (uint32_t sequence=0;sequence<2;++sequence) {
        std::shared_ptr<llama_kv_stream_logical_cache> cache;
        if (!t.assert_true(table->get(handles[sequence],0,cache) && cache->ready())) return;
        backing[sequence]=cache->host();
        t.assert_true(backing[sequence]->demand_backed() && backing[sequence]->committed_bytes()==0);
    }
    ggml_backend_buffer_ptr retained;
    past={};uint64_t parked_generation=0;
    for (size_t round=0;round<rounds.size();++round) {
        const auto & selected=rounds[round];
        std::vector<llama_kv_stream_selection_slot> candidates;std::vector<llama_kv_stream_slot_handle> chosen;
        for (auto sequence : selected) {candidates.push_back({uint32_t(sequence),past[sequence]});chosen.push_back(handles[sequence]);}
        llama_kv_stream_selection selection;
        if (!t.assert_true(llama_kv_stream_select({table->limits().group_limit,selected.size()},candidates,selection)==
                llama_kv_stream_selection_status::success)) return;
        input(selected,past);
        if (!t.assert_true(group->activate(selection,chosen,batch,llama_memory_work_phase::target_verify,1,round+1))) return;
        auto * context=group->context();if (!t.assert_true(context!=nullptr)) return;
        if (!retained) retained.reset(ggml_backend_buffer_retain(context->group_parent()));
        if (!t.assert_true(context->group_parent()==retained.get())) return;
        for (uint32_t step=0;step<4;++step) {
            input(selected,past);if (!t.assert_true(context->decode(batch)==0)) return;
            for (size_t row=0;row<selected.size();++row) {
                const auto * logits=llama_get_logits_ith(context,int32_t(row));
                if (!t.assert_true(logits && !std::memcmp(logits,reference[round][step*selected.size()+row].data(),vocabulary*sizeof(float)))) return;
                ++past[selected[row]];
            }
        }
        if (!t.assert_true(group->park() && !group->context())) return;
        for (uint32_t sequence=0;sequence<2;++sequence) {
            std::shared_ptr<llama_kv_stream_logical_cache> cache;
            if (!t.assert_true(table->get(handles[sequence],0,cache) && cache->ready())) return;
            t.assert_equal(size_t(past[sequence]),cache->tokens());
            if (!round && !sequence) parked_generation=cache->content()->generation();
            if (round==1 && !sequence) t.assert_equal(parked_generation,cache->content()->generation());
        }
        llama_compute_arena_ptr arena(ggml_backend_memory_arena_new_from_buffer(retained.get()));
        if (!t.assert_true(arena && ggml_backend_memory_arena_begin(arena.get(),0) && ggml_backend_memory_arena_commit(arena.get()))) return;
    }
    std::shared_ptr<llama_kv_stream_logical_cache> peer;
    if (!t.assert_true(table->get(handles[0],0,peer))) return;
    const auto peer_checkpoint=peer->checkpoint();
    llama_kv_stream_selection failed_selection;
    if (!t.assert_true(llama_kv_stream_select({table->limits().group_limit,1},{{1,past[1]}},failed_selection)==
            llama_kv_stream_selection_status::success)) return;
    input({1},past);
    if (!t.assert_true(group->activate(failed_selection,{handles[1]},batch,llama_memory_work_phase::target_verify,1,4))) return;
    {
        dflash_compute_failure fault(group->context(),GGML_STATUS_FAILED);
        if (!t.assert_true(group->context()->decode(batch)!=0 && fault.calls>=1)) return;
    }
    if (!t.assert_true(group->discard() && !group->context() && table->get(handles[0],0,peer))) return;
    t.assert_equal(peer_checkpoint.tokens,peer->tokens());
    t.assert_equal(peer_checkpoint.generation,peer->content()->generation());
    llama_kv_stream_selection peer_selection;
    if (!t.assert_true(llama_kv_stream_select({table->limits().group_limit,1},{{0,past[0]}},peer_selection)==
            llama_kv_stream_selection_status::success)) return;
    input({0},past);
    if (!t.assert_true(group->activate(peer_selection,{handles[0]},batch,llama_memory_work_phase::target_verify,1,5) &&
            group->context()->group_parent()==retained.get() && group->context()->decode(batch)==0)) return;
    const auto * peer_logits=llama_get_logits_ith(group->context(),0);
    if (!t.assert_true(peer_logits && !std::memcmp(peer_logits,peer_continuation.data(),vocabulary*sizeof(float)))) return;
    ++past[0];
    if (!t.assert_true(group->park())) return;
    t.assert_true(group->reset_request(1));
    const auto next_handles=group->source_handles();
    t.assert_equal(handles[1].generation+1,next_handles[1].generation);
    std::shared_ptr<llama_kv_stream_logical_cache> cache;
    t.assert_true(!table->get(handles[1],0,cache));
    for (uint32_t sequence=0;sequence<2;++sequence) {
        if (!t.assert_true(table->get(next_handles[sequence],0,cache) && cache->ready() && cache->host()==backing[sequence])) return;
        t.assert_equal(sequence ? size_t(0) : size_t(past[0]),cache->tokens());
    }
    std::vector<llama_kv_stream_selection_slot> stale_candidates{{1,0}};
    llama_kv_stream_selection stale;
    if (!t.assert_true(llama_kv_stream_select({table->limits().group_limit,1},stale_candidates,stale)==
            llama_kv_stream_selection_status::success)) return;
    input({1},std::array<uint32_t,2>{});
    t.assert_true(!group->activate(stale,{handles[1]},batch,llama_memory_work_phase::target_verify,1,6) && !group->context());
}

static void test_group_verify_widths(testing & t,llama_model * model,bool keep_binding=false,bool split_verification=false) {
    if (!t.assert_equal(LLM_ARCH_GEMMA4,model->arch)) return;
    auto cp=llama_context_default_params();
    cp.n_ctx=4096;cp.n_batch=256;cp.n_ubatch=split_verification ? 4 : 256;cp.n_seq_max=2;cp.kv_unified=true;cp.swa_full=false;
    cp.n_outputs_max=8;cp.n_outputs_max_per_seq=4;
    cp.type_k=GGML_TYPE_Q8_0;cp.type_v=GGML_TYPE_Q4_0;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.n_threads=cp.n_threads_batch=4;
    auto batch=llama_batch_init(10,0,1);
    struct release_batch {llama_batch & batch;~release_batch() {llama_batch_free(batch);}} release{batch};
    const auto input=[&](std::array<uint32_t,2> widths,const std::array<uint32_t,2> & past) {
        batch.n_tokens=0;
        for (llama_seq_id slot=0;slot<2;++slot) for (uint32_t row=0;row<widths[slot];++row) {
            const auto i=batch.n_tokens++;batch.token[i]=llama_token(10+slot*20+past[slot]+row);batch.pos[i]=past[slot]+row;
            batch.n_seq_id[i]=1;batch.seq_id[i][0]=slot;batch.logits[i]=1;
        }
    };
    const auto vocabulary=model->vocab.n_tokens(),hidden=model->hparams.n_embd_out();
    std::array<std::vector<uint8_t>,2> native_states;
    std::vector<std::vector<std::vector<float>>> logits,embeddings;
    const std::array<uint32_t,2> profiles[]={{1,1},{2,2},{3,3},{4,4},{1,4},{4,1}};
    std::array<uint32_t,2> past{};
    std::unique_ptr<llama_context> continuous;
    if (keep_binding) {continuous=std::make_unique<llama_context>(*model,cp);llama_set_embeddings_nextn(continuous.get(),true,false);}
    for (const auto & widths : profiles) {
        std::unique_ptr<llama_context> transient;
        if (!keep_binding) {transient=std::make_unique<llama_context>(*model,cp);llama_set_embeddings_nextn(transient.get(),true,false);}
        auto & native=keep_binding ? *continuous : *transient;
        for (llama_seq_id slot=0;slot<2;++slot) if (!native_states[slot].empty()) {
            const auto & state=native_states[slot];
            if (!t.assert_equal(state.size(),llama_state_seq_set_data(&native,state.data(),state.size(),slot))) return;
        }
        input(widths,past);if (!t.assert_equal(0,native.decode(batch))) return;
        logits.emplace_back();embeddings.emplace_back();
        for (int32_t row=0;row<batch.n_tokens;++row) {
            const auto * values=llama_get_logits_ith(&native,row);const auto * h=llama_get_embeddings_nextn_ith(&native,row);
            if (!t.assert_true(values && h &&
                    std::all_of(values,values+vocabulary,[](float x) {return std::isfinite(x);}) &&
                    std::all_of(h,h+hidden,[](float x) {return std::isfinite(x);}))) return;
            logits.back().emplace_back(values,values+vocabulary);embeddings.back().emplace_back(h,h+hidden);
        }
        if (!keep_binding) for (llama_seq_id slot=0;slot<2;++slot) {
            auto & state=native_states[slot];state.resize(llama_state_seq_get_size(&native,slot));
            if (!t.assert_true(!state.empty() && llama_state_seq_get_data(&native,state.data(),state.size(),slot)==state.size())) return;
        }
        for (size_t slot=0;slot<2;++slot) past[slot]+=widths[slot];
    }
    continuous.reset();
    t.out<<"group verify native rows=30 widths=1-4 mixed=1+4,4+1 RS0 hidden="<<hidden<<" vocabulary="<<vocabulary<<'\n';
    cp.shared_device_memory_bytes=size_t(1024)*1024*1024;cp.kv_stream_verify_max=4;
    for (int invalid=0;invalid<4;++invalid) {
        auto bad=cp;
        if (invalid==0) bad.kv_stream_verify_max=5;
        if (invalid==1) bad.n_ubatch=3;
        if (invalid==2) bad.n_seq_max=3;
        if (invalid==3) bad.kv_unified=false;
        if (!t.assert_true(!llama_context_group::create(*model,bad))) return;
    }
    auto group=llama_context_group::create(*model,cp);
    if (!t.assert_true("declared per-slot verification supports two logical slots without recurrent snapshots",group!=nullptr)) return;
    llama_set_embeddings_nextn(group->metadata_context(),true,false);
    size_t catalog_required=0;
    if (keep_binding) {
        llama_context measured(*model,cp,std::vector<llama_seq_id>{0,1});llama_set_embeddings_nextn(&measured,true,false);
        std::vector<llama_memory_phase_cache_identity> sources;
        for (const auto & handle : group->source_handles()) {
            std::shared_ptr<llama_kv_stream_logical_cache> cache;
            if (!t.assert_true(group->source_table()->get(handle,0,cache))) return;
            sources.push_back({cache->host()->cache_id(),cache->content()->generation()});
        }
        input({1,1},std::array<uint32_t,2>{});
        const llama_memory_phase_cache_identity bounded{sources.front().id*4+1,1};
        llama_context_group_plan quote;
        auto admitted=measured.quote_target_group(batch,sources,bounded,llama_memory_work_phase::target_verify,1,
            cp.shared_device_memory_bytes,quote);
        if (!t.assert_equal(int(llama_context_memory_create_status::success),int(admitted.status))) return;
        t.assert_equal(size_t(63),quote.budget.workload_bytes.size());
        if (split_verification) {
            t.assert_equal(size_t(2),quote.target.measurement.workload.native_batches.size());
            t.assert_equal(uint32_t(4),quote.target.measurement.workload.microbatch_rows);
            for (const auto & member : quote.target.measurement.workload.members) t.assert_equal(uint32_t(4),member.input_rows);
        }
        const auto required=admitted.required_bytes;catalog_required=required;
        if (!t.assert_true(required>0 && required<=cp.shared_device_memory_bytes)) return;
        llama_context_group_plan untouched;untouched.current_state_bytes=73;
        admitted=measured.quote_target_group(batch,sources,bounded,llama_memory_work_phase::target_verify,1,required-1,untouched);
        t.assert_equal(int(llama_context_memory_create_status::insufficient_budget),int(admitted.status));
        t.assert_equal(required,admitted.required_bytes);t.assert_equal(size_t(1),admitted.additional_bytes);
        t.assert_equal(size_t(73),untouched.current_state_bytes);
        admitted=measured.quote_target_group(batch,sources,bounded,llama_memory_work_phase::target_verify,1,required,quote);
        t.assert_equal(int(llama_context_memory_create_status::success),int(admitted.status));
        t.assert_equal(size_t(63),quote.budget.workload_bytes.size());
        t.out<<"group verification catalog profiles=63 required="<<required<<" short_additional=1\n";
    }
    ggml_backend_buffer_ptr parent;
    past={};
    for (size_t profile=0;profile<sizeof(profiles)/sizeof(*profiles);++profile) {
        const auto widths=profiles[profile];
        input(widths,past);
        if (!keep_binding || !profile) {
            llama_kv_stream_selection selection;
            if (!t.assert_true(llama_kv_stream_select({4096,2},{{0,past[0]},{1,past[1]}},selection)==llama_kv_stream_selection_status::success)) return;
            if (!t.assert_true(group->activate(selection,group->source_handles(),batch,llama_memory_work_phase::target_verify,1,profile+1))) return;
        }
        auto * context=group->context();
        if (!parent) parent.reset(ggml_backend_buffer_retain(context->group_parent()));
        if (!t.assert_true(context->group_parent()==parent.get() && context->get_cparams().embeddings_nextn)) return;
        t.assert_equal(uint32_t(0),context->get_cparams().n_rs_seq);
        t.assert_equal(uint32_t(4),context->get_cparams().kv_stream_verify_max);
        if (!t.assert_equal(0,context->decode(batch))) return;
        for (int32_t row=0;row<batch.n_tokens;++row) {
            const auto * values=llama_get_logits_ith(context,row);const auto * h=llama_get_embeddings_nextn_ith(context,row);
            if (!t.assert_true(values && h && !std::memcmp(values,logits[profile][row].data(),vocabulary*sizeof(float)) &&
                    !std::memcmp(h,embeddings[profile][row].data(),hidden*sizeof(float)))) return;
        }
        for (size_t slot=0;slot<2;++slot) past[slot]+=widths[slot];
        if (profile+1==sizeof(profiles)/sizeof(*profiles)) {
            std::vector<llama_context_group_sequence_state> before,after;
            if (!t.assert_true(context->save_group_sequences(before))) return;
            input({5,1},past);t.assert_equal(-1,context->decode(batch));
            if (!t.assert_true(context->save_group_sequences(after) && before.size()==after.size())) return;
            for (int32_t row=0;row<int32_t(widths[0]+widths[1]);++row) {
                const auto * values=llama_get_logits_ith(context,row);const auto * h=llama_get_embeddings_nextn_ith(context,row);
                if (!t.assert_true(values && h && !std::memcmp(values,logits[profile][row].data(),vocabulary*sizeof(float)) &&
                        !std::memcmp(h,embeddings[profile][row].data(),hidden*sizeof(float)))) return;
            }
            for (size_t i=0;i<before.size();++i) {
                t.assert_equal(before[i].tokens,after[i].tokens);t.assert_equal(before[i].cache_generation,after[i].cache_generation);
                t.assert_true(before[i].bounded==after[i].bounded && before[i].recurrent==after[i].recurrent);
            }
        }
        if (!keep_binding && !t.assert_true(group->park())) return;
    }
    if (keep_binding) {
        if (!t.assert_true(group->park())) return;
        group.reset();parent.reset();
        for (bool short_quota : {true,false}) {
            auto minimum_params=cp;minimum_params.shared_device_memory_bytes=catalog_required-size_t(short_quota);
            auto minimum=llama_context_group::create(*model,minimum_params);
            if (!t.assert_true(minimum!=nullptr)) return;
            llama_set_embeddings_nextn(minimum->metadata_context(),true,false);
            llama_kv_stream_selection selection;
            if (!t.assert_true(llama_kv_stream_select({4096,2},{{0,0},{1,0}},selection)==llama_kv_stream_selection_status::success)) return;
            input({1,1},std::array<uint32_t,2>{});
            llama_context_memory_create_result admission;
            const bool active=minimum->activate(selection,minimum->source_handles(),batch,llama_memory_work_phase::target_verify,1,1,&admission);
            if (short_quota) {
                t.assert_true(!active && !minimum->context());
                t.assert_equal(int(llama_context_memory_create_status::insufficient_budget),int(admission.status));
                t.assert_equal(catalog_required,admission.required_bytes);t.assert_equal(size_t(1),admission.additional_bytes);
                for (const auto & handle : minimum->source_handles()) {
                    std::shared_ptr<llama_kv_stream_logical_cache> cache;
                    if (!t.assert_true(minimum->source_table()->get(handle,0,cache) && cache->ready() &&
                            !cache->tokens() && !cache->host()->committed_bytes())) return;
                }
            } else {
                if (!t.assert_true(active && admission.status==llama_context_memory_create_status::success)) return;
                std::array<uint32_t,2> minimum_past{};
                for (size_t profile=0;profile<sizeof(profiles)/sizeof(*profiles);++profile) {
                    input(profiles[profile],minimum_past);
                    if (!t.assert_equal(0,minimum->context()->decode(batch))) return;
                    for (int32_t row=0;row<batch.n_tokens;++row) {
                        const auto * values=llama_get_logits_ith(minimum->context(),row);
                        const auto * h=llama_get_embeddings_nextn_ith(minimum->context(),row);
                        if (!t.assert_true(values && h && !std::memcmp(values,logits[profile][row].data(),vocabulary*sizeof(float)) &&
                                !std::memcmp(h,embeddings[profile][row].data(),hidden*sizeof(float)))) return;
                    }
                    for (size_t slot=0;slot<2;++slot) minimum_past[slot]+=profiles[profile][slot];
                }
                t.assert_true(minimum->park());
            }
        }
    }
}

struct group_ring_probe {
    inline static group_ring_probe * active=nullptr;
    ggml_backend_reg_t registry;
    decltype(ggml_backend_reg_i::get_proc_address) original;
    ggml_kv_stream_copy_ops copy;
    decltype(ggml_kv_stream_copy_ops::acquire_span) acquire;
    const std::string & phase;
    std::array<size_t,3> spans{};
    group_ring_probe(ggml_backend_dev_t device,const std::string & phase) : registry(ggml_backend_dev_backend_reg(device)),
        original(registry->iface.get_proc_address),phase(phase) {
        GGML_ASSERT(!active);
        const auto get=reinterpret_cast<ggml_kv_stream_copy_ops_get>(original(registry,"ggml_backend_kv_stream_copy_ops"));
        GGML_ASSERT(get && get() && get()->acquire_span);
        copy=*get();acquire=copy.acquire_span;active=this;
        copy.acquire_span=[](void * queue,size_t first,size_t count) {
            const bool okay=active->acquire(queue,first,count);
            if (okay) {
                const auto & phase=active->phase;
                if (phase=="prefill") active->spans[0]+=count;
                if (phase=="catchup") active->spans[1]+=count;
                if (phase=="predict") active->spans[2]+=count;
            }
            return okay;
        };
        registry->iface.get_proc_address=[](ggml_backend_reg_t registry,const char * name) -> void * {
            if (!std::strcmp(name,"ggml_backend_kv_stream_copy_ops"))
                return reinterpret_cast<void *>(+[]() -> const ggml_kv_stream_copy_ops * {return &active->copy;});
            return active->original(registry,name);
        };
    }
    ~group_ring_probe() {registry->iface.get_proc_address=original;active=nullptr;}
};

static void test_two_slot_mtp_image_catchup(testing & t,llama_model * model,const char * image_path,bool grouped=false) {
    if (!t.assert_true(model && model->arch==LLM_ARCH_QWEN35 && model->hparams.n_layer_nextn==1)) return;
    const size_t width=model->hparams.n_embd_out(),rows=64;
    std::vector<float> embeddings(rows*width);
    std::ifstream image(image_path,std::ios::binary);
    image.read(reinterpret_cast<char *>(embeddings.data()),std::streamsize(embeddings.size()*sizeof(float)));
    if (!t.assert_true(image && image.peek()==EOF && std::all_of(embeddings.begin(),embeddings.end(),[](float x) {return std::isfinite(x);}))) return;
    auto cp=llama_context_default_params();
    cp.n_ctx=512;cp.n_batch=cp.n_ubatch=256;cp.n_seq_max=2;cp.kv_unified=true;
    cp.n_outputs_max=8;cp.n_outputs_max_per_seq=4;cp.n_threads=cp.n_threads_batch=4;
    cp.type_k=GGML_TYPE_Q8_0;cp.type_v=GGML_TYPE_Q4_0;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.n_rs_seq=grouped ? 3 : 0;
    kv_stream_bit_oracle prefix_vectors,image_vectors,child_states,next_vectors,next_drafts;
    const auto vectors=[&](llama_context & context,kv_stream_bit_oracle & oracle,size_t hidden_rows,size_t logit_rows) {
        context.synchronize();const auto * hidden=llama_get_embeddings_nextn(&context);const auto * logits=llama_get_logits(&context);
        if (!t.assert_true(hidden && logits)) return false;
        oracle.bytes("hidden",hidden,hidden_rows*width*sizeof(float));
        oracle.bytes("logits",logits,logit_rows*size_t(model->vocab.n_tokens())*sizeof(float));return true;
    };
    const auto picture=[&](llama_context & context,llama_seq_id selected) {
        common_batch visual(&context);const llama_pos first=selected ? 7 : 5;
        for (size_t row=0;row<rows;++row) {
            const llama_pos position[4]={first,first+llama_pos(row/8),first+llama_pos(row%8),0};
            visual.add_embd({embeddings.data()+row*width,1,width},position,selected,row+1==rows);
        }
        return visual;
    };
    const auto prompt=[&](llama_context & context) {
        common_batch batch(&context);
        for (llama_seq_id slot=0;slot<2;++slot) for (int row=0;row<(slot ? 7 : 5);++row)
            batch.add(100+slot*20+row,row,slot,row+1==(slot ? 7 : 5));
        return batch;
    };
    const auto suffix=[&](llama_context & context) {
        common_batch batch(&context);
        for (llama_seq_id slot=0;slot<2;++slot) batch.add(500+slot,slot ? 15 : 13,slot,true);
        return batch;
    };
    const auto predict=[&](llama_context & target,llama_context & child,common_speculative * spec) {
        std::array<llama_tokens,2> histories,proposals;
        for (llama_seq_id slot=0;slot<2;++slot) {
            for (int row=0;row<(slot ? 7 : 5);++row) histories[slot].push_back(100+slot*20+row);
            histories[slot].insert(histories[slot].end(),rows,LLAMA_TOKEN_NULL);histories[slot].push_back(500+slot);
            const auto * logits=llama_get_logits_ith(&target,slot);
            if (!t.assert_true(logits!=nullptr)) return false;
            const auto anchor=llama_token(std::max_element(logits,logits+model->vocab.n_tokens())-logits);
            common_speculative_begin(spec,slot,histories[slot]);
            common_speculative_get_draft_params(spec,slot)={true,uint16_t(slot ? 3 : 1),llama_pos(slot ? 16 : 14),anchor,
                &histories[slot],&proposals[slot],nullptr,0.0f,123};
        }
        common_speculative_draft(spec);child.synchronize();
        for (llama_seq_id slot=0;slot<2;++slot) {
            if (!t.assert_equal(size_t(slot ? 3 : 1),proposals[slot].size())) return false;
            std::vector<uint8_t> carry;if (!t.assert_true(common_speculative_get_state(spec,slot,carry))) return false;
            next_drafts.bytes("proposal",proposals[slot].data(),proposals[slot].size()*sizeof(llama_token));
            next_drafts.bytes("carry",carry.data(),carry.size());
        }
        return true;
    };
    {
    llama_context target(*model,cp);llama_set_embeddings_nextn(&target,true,false);
    auto dp=cp;dp.n_rs_seq=0;dp.ctx_type=LLAMA_CONTEXT_TYPE_MTP;dp.ctx_other=&target;
    llama_context child(*model,dp);llama_set_embeddings_nextn(&child,true,true);
    common_params_speculative parameters;parameters.types={COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
    parameters.draft.ctx_tgt=&target;parameters.draft.ctx_dft=&child;parameters.draft.n_max=3;
    common_speculative_ptr spec(common_speculative_init(parameters,2));
    if (!t.assert_true(spec!=nullptr)) return;
    auto prefix=prompt(target);
    if (!t.assert_equal(0,llama_process(&target,LLAMA_PROCESS_TYPE_DECODE,prefix.get())) ||
            !t.assert_true(common_speculative_process(spec.get(),prefix))) return;
    if (grouped && !vectors(target,prefix_vectors,12,2)) return;
    const auto snapshot=[&](llama_seq_id slot,std::vector<uint8_t> & bytes) {
        child.synchronize();bytes.resize(llama_state_seq_get_size(&child,slot));
        return t.assert_true(!bytes.empty()) && t.assert_equal(bytes.size(),llama_state_seq_get_data(&child,bytes.data(),bytes.size(),slot));
    };
    const auto restore=[&](const std::array<std::vector<uint8_t>,2> & states) {
        llama_memory_clear(child.get_memory(),true);
        for (llama_seq_id slot=0;slot<2;++slot)
            if (!t.assert_equal(states[slot].size(),llama_state_seq_set_data(&child,states[slot].data(),states[slot].size(),slot))) return false;
        return true;
    };
    for (llama_seq_id selected : {1,0}) {
        if (grouped) {
            std::array<std::vector<uint8_t>,2> states;
            target.synchronize();
            for (llama_seq_id slot=0;slot<2;++slot) {
                states[slot].resize(llama_state_seq_get_size(&target,slot));
                if (!t.assert_equal(states[slot].size(),llama_state_seq_get_data(&target,states[slot].data(),states[slot].size(),slot))) return;
            }
            llama_memory_clear(target.get_memory(),true);
            for (llama_seq_id slot=0;slot<2;++slot)
                if (!t.assert_equal(states[slot].size(),llama_state_seq_set_data(&target,states[slot].data(),states[slot].size(),slot))) return;
        }
        auto visual=picture(target,selected);
        if (!t.assert_equal(0,llama_process(&target,LLAMA_PROCESS_TYPE_DECODE,visual.get()))) return;
        if (grouped && !vectors(target,image_vectors,rows,1)) return;
        target.synchronize();const auto * hidden=llama_get_embeddings_nextn(&target);
        if (!t.assert_true(hidden!=nullptr)) return;
        std::array<std::vector<uint8_t>,2> carries,states,expected;
        for (llama_seq_id slot=0;slot<2;++slot)
            if (!t.assert_true(common_speculative_get_state(spec.get(),slot,carries[slot])) || !snapshot(slot,states[slot])) return;
        if (!t.assert_equal(sizeof(uint32_t)+width*sizeof(float),carries[selected].size())) return;
        std::vector<float> shifted(rows*width);
        std::memcpy(shifted.data(),carries[selected].data()+sizeof(uint32_t),width*sizeof(float));
        std::memcpy(shifted.data()+width,hidden,(rows-1)*width*sizeof(float));
        auto expected_carry=carries[selected];
        std::memcpy(expected_carry.data()+sizeof(uint32_t),hidden+(rows-1)*width,width*sizeof(float));
        common_batch manual(&child);manual.tokens=visual.tokens;
        if (!t.assert_true(manual.n_pos==visual.n_pos && restore(states)) ||
                !t.assert_equal(0,llama_decode_mtp_embeddings(&child,*manual.get(),shifted.data(),shifted.size()))) return;
        for (llama_seq_id slot=0;slot<2;++slot) if (!snapshot(slot,expected[slot])) return;
        if (!restore(states)) return;
        for (int invalid=0;invalid<7;++invalid) {
            common_batch bad(&target);bad.tokens=visual.tokens;
            switch (invalid) {
                case 0: bad.tokens[1].seq_id=-1;break;
                case 1: bad.tokens[1].seq_id=2;break;
                case 2: bad.tokens[1].seq_ids_extra={1-selected};break;
                case 3: bad.tokens[1].embd.n_embd=width-1;break;
                case 4: bad.tokens[1].embd.n_rows=2;break;
                case 5: bad.tokens[1].embd.data=nullptr;break;
                case 6: bad.tokens[1].seq_id=1-selected;break;
            }
            if (!t.assert_true("invalid image catch-up rejects before changing either slot",!common_speculative_process(spec.get(),bad))) return;
            for (llama_seq_id slot=0;slot<2;++slot) {
                std::vector<uint8_t> actual,carry;
                if (!snapshot(slot,actual) || !t.assert_true(common_speculative_get_state(spec.get(),slot,carry)) ||
                        !t.assert_true(actual==states[slot] && carry==carries[slot])) return;
            }
        }
        if (!t.assert_true("two-slot MTP image catch-up accepts the selected sequence after a valid native control",
                common_speculative_process(spec.get(),visual))) return;
        for (llama_seq_id slot=0;slot<2;++slot) {
            std::vector<uint8_t> actual,carry;
            if (!snapshot(slot,actual) || !t.assert_true(common_speculative_get_state(spec.get(),slot,carry))) return;
            if (!t.assert_true("image catch-up preserves every native KV and recurrent state byte",actual==expected[slot]) ||
                    !t.assert_true("only the selected slot advances its hidden carry",carry==(slot==selected ? expected_carry : carries[slot]))) return;
            if (grouped) {
                child_states.bytes("state",actual.data(),actual.size());child_states.bytes("carry",carry.data(),carry.size());
            }
        }
        t.out<<"native image MTP sequence="<<selected<<" rows="<<rows<<" exact_state_and_carry=1\n";
    }
    if (grouped) {
        std::array<std::vector<uint8_t>,2> target_states,states;
        target.synchronize();
        for (llama_seq_id slot=0;slot<2;++slot) {
            target_states[slot].resize(llama_state_seq_get_size(&target,slot));
            if (!t.assert_equal(target_states[slot].size(),llama_state_seq_get_data(&target,target_states[slot].data(),target_states[slot].size(),slot)) ||
                    !snapshot(slot,states[slot])) return;
        }
        llama_memory_clear(target.get_memory(),true);
        for (llama_seq_id slot=0;slot<2;++slot)
            if (!t.assert_equal(target_states[slot].size(),llama_state_seq_set_data(&target,target_states[slot].data(),target_states[slot].size(),slot))) return;
        if (!restore(states)) return;
        auto next=suffix(target);
        if (!t.assert_equal(0,llama_process(&target,LLAMA_PROCESS_TYPE_DECODE,next.get())) ||
                !vectors(target,next_vectors,2,2) || !t.assert_true(common_speculative_process(spec.get(),next)) ||
                !predict(target,child,spec.get())) return;
    }
    }
    if (!grouped) return;
    cp.shared_device_memory_bytes=size_t(1024)*1048576;cp.kv_stream_auxiliary_layers=1;
    auto group=llama_context_group::create(*model,cp);
    if (!t.assert_true(group && group->auxiliary_metadata_context())) return;
    llama_set_embeddings_nextn(group->metadata_context(),true,false);
    common_params_speculative parameters;parameters.types={COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
    parameters.draft.ctx_tgt=group->metadata_context();parameters.draft.ctx_dft=group->auxiliary_metadata_context();parameters.draft.n_max=3;
    common_speculative_ptr spec(common_speculative_init(parameters,2));
    if (!t.assert_true(spec!=nullptr)) return;
    std::array<size_t,2> history{};uint64_t revision=0;
    const auto activate=[&](common_batch & batch,llama_memory_work_phase phase=llama_memory_work_phase::target_prefill) {
        llama_kv_stream_selection selected;
        if (llama_kv_stream_select({512,2},{{0,history[0]},{1,history[1]}},selected)!=llama_kv_stream_selection_status::success) return false;
        llama_batch_allocr allocator(uint32_t(batch.get()->n_pos_per_embd));
        if (!allocator.init(*batch.get(),model->vocab,false)) return false;
        llama_context_memory_create_result result;
        if (!group->activate(selected,group->source_handles(),allocator.get_batch(),phase,1,++revision,&result)) {
            t.out<<"image MTP admission status="<<llama_context_memory_create_status_name(result.status)<<'\n';return false;
        }
        return common_speculative_rebind(spec.get(),group->context(),group->auxiliary_context());
    };
    const auto park=[&]() {
        if (!common_speculative_rebind(spec.get(),group->metadata_context(),group->auxiliary_metadata_context()) || !group->park()) return false;
        auto * parent=group->prepare_vision_parent();std::vector<ggml_backend_memory_lease_t> loans;
        if (!parent || !group->lend_vision({cp.shared_device_memory_bytes},loans)) return false;
        const bool parked=!group->context() && !group->auxiliary_context() && loans.size()==1 &&
            ggml_backend_buffer_get_size(ggml_backend_memory_lease_buffer(loans.front()))==cp.shared_device_memory_bytes;
        if (parked) ggml_backend_buffer_clear(ggml_backend_memory_lease_buffer(loans.front()),0xa5);
        for (auto * loan : loans) ggml_backend_memory_lease_free(loan);
        return group->release_vision() && parked;
    };
    auto prefix=prompt(*group->metadata_context());
    if (!t.assert_true(activate(prefix)) || !t.assert_equal(0,llama_process(group->context(),LLAMA_PROCESS_TYPE_DECODE,prefix.get())) ||
            !t.assert_true(common_speculative_process(spec.get(),prefix))) return;
    prefix_vectors.compare();if (!vectors(*group->context(),prefix_vectors,12,2)) return;prefix_vectors.finish();
    history={5,7};
    if (!t.assert_true(park())) return;
    image_vectors.compare();child_states.compare();
    ggml_backend_buffer_t parent=nullptr;
    for (llama_seq_id selected : {1,0}) {
        auto visual=picture(*group->metadata_context(),selected);
        if (!t.assert_true("shared target and MTP image graphs are admitted before any image submission",activate(visual))) return;
        auto * target=group->context();auto * child=group->auxiliary_context();
        if (!parent) parent=target->group_parent();
        if (!t.assert_true(parent && parent==target->group_parent() && parent==child->group_parent())) return;
        llama_set_kv_stream_decode(target,false);
        if (!t.assert_equal(0,llama_process(target,LLAMA_PROCESS_TYPE_DECODE,visual.get())) ||
                !vectors(*target,image_vectors,rows,1)) return;
        if (!t.assert_true("the selected MTP image catch-up uses its measured shared workspace",common_speculative_process(spec.get(),visual))) return;
        child->synchronize();
        for (llama_seq_id slot=0;slot<2;++slot) {
            std::vector<uint8_t> actual(llama_state_seq_get_size(child,slot)),carry;
            if (!t.assert_equal(actual.size(),llama_state_seq_get_data(child,actual.data(),actual.size(),slot)) ||
                    !t.assert_true(common_speculative_get_state(spec.get(),slot,carry))) return;
            child_states.bytes("state",actual.data(),actual.size());child_states.bytes("carry",carry.data(),carry.size());
        }
        history[selected]+=rows;
        if (!t.assert_true(park())) return;
        t.out<<"group image MTP sequence="<<selected<<" rows="<<rows<<" same_parent=1\n";
    }
    image_vectors.finish();child_states.finish();
    auto next=suffix(*group->metadata_context());
    if (!t.assert_true(activate(next,llama_memory_work_phase::target_verify))) return;
    llama_set_kv_stream_decode(group->context(),true);
    next_vectors.compare();next_drafts.compare();
    if (!t.assert_equal(0,llama_process(group->context(),LLAMA_PROCESS_TYPE_DECODE,next.get())) ||
            !vectors(*group->context(),next_vectors,2,2) || !t.assert_true(common_speculative_process(spec.get(),next)) ||
            !predict(*group->context(),*group->auxiliary_context(),spec.get())) return;
    next_vectors.finish();next_drafts.finish();
    if (!t.assert_true(park())) return;
    t.out<<"group image MTP continuation native_exact=1 draft_widths=1,3 full_parent_overwrite=1\n";
}

static void test_group_mtp_admission(testing & t,llama_model * model,bool resume=false,bool phase_guard=false,bool ring=false,bool partial=false,
        std::array<uint16_t,2> draft_widths={1,3},std::array<uint16_t,2> accepted_widths={0,2},bool backend_sampling=false,uint32_t cutoff=0) {
    if (!t.assert_true(model && model->arch==LLM_ARCH_QWEN35 && model->hparams.n_layer_nextn==1)) return;
    auto cp=llama_context_default_params();
    const auto maximum_depth=*std::max_element(draft_widths.begin(),draft_widths.end());
    cp.n_ctx=ring ? 2048 : 512;cp.n_batch=ring ? 2048 : 256;cp.n_ubatch=256;cp.n_seq_max=2;cp.n_rs_seq=maximum_depth;cp.kv_unified=true;
    cp.type_k=GGML_TYPE_Q8_0;cp.type_v=GGML_TYPE_Q4_0;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.n_outputs_max=8;cp.n_outputs_max_per_seq=4;cp.n_threads=cp.n_threads_batch=4;
    kv_stream_bit_oracle oracle;
    kv_stream_exact_trace trace(oracle,"target");
    cp.cb_eval=kv_stream_exact_trace::callback;cp.cb_eval_user_data=&trace;
    std::unique_ptr<group_ring_probe> transport;
    if (ring) {
        if (!t.assert_true(model->devices.size()==1)) return;
        transport=std::make_unique<group_ring_probe>(model->devices.front().dev,trace.phase);
    }
    const uint32_t lengths[2]={ring ? 513u : 17u,ring ? 515u : 19u};
    const std::array<uint16_t,2> expected_widths={uint16_t(cutoff==1 ? 0 : draft_widths[0]),
        uint16_t(cutoff==1 ? 0 : draft_widths[1])};
    std::array<llama_tokens,2> histories,proposals;
    std::array<llama_token,2> anchors{};
    std::array<std::vector<uint8_t>,2> carries,target_states,child_states;
    const bool native_restore=resume && !backend_sampling;
    auto safe_params=cp;safe_params.shared_device_memory_bytes=size_t(1024)*1024*1024;safe_params.kv_stream_auxiliary_layers=1;
    auto safe=native_restore ? llama_context_group::create(*model,safe_params) : nullptr;
    if (native_restore && !t.assert_true(safe && safe->auxiliary_metadata_context())) return;
    common_speculative_ptr native_spec,stream_spec;
    struct driver_guard {
        common_speculative_ptr & spec;
        llama_context * target;
        llama_context * child;
        ~driver_guard() {
            if (spec && (!target || !child || !common_speculative_rebind(spec.get(),target,child))) spec.reset();
        }
    };
    const auto retarget=[&](common_speculative_ptr & spec,llama_context * target,llama_context * child) {
        const auto before=common_speculative_get_stats(spec.get());
        const std::array<common_speculative_type,2> providers={common_speculative_get_last_type(spec.get(),0),
            common_speculative_get_last_type(spec.get(),1)};
        t.assert_true(!common_speculative_rebind(nullptr,target,child));
        t.assert_true(!common_speculative_rebind(spec.get(),nullptr,child));
        t.assert_true(!common_speculative_rebind(spec.get(),target,nullptr));
        t.assert_true(!common_speculative_rebind(spec.get(),target,target));
        t.assert_true(!common_speculative_rebind(spec.get(),child,target));
        if (!t.assert_true(common_speculative_rebind(spec.get(),target,child))) return false;
        const auto after=common_speculative_get_stats(spec.get());
        if (!t.assert_equal(before.size(),after.size())) return false;
        for (size_t i=0;i<before.size();++i) {
            const auto & a=before[i];const auto & b=after[i];
            t.assert_true(a.type==b.type && a.n_call_begin==b.n_call_begin && a.n_call_draft==b.n_call_draft &&
                a.n_call_accept==b.n_call_accept && a.n_gen_drafts==b.n_gen_drafts && a.n_acc_drafts==b.n_acc_drafts &&
                a.n_gen_tokens==b.n_gen_tokens && a.n_acc_tokens==b.n_acc_tokens && a.n_predict==b.n_predict);
        }
        for (llama_seq_id slot=0;slot<2;++slot) {
            t.assert_equal(providers[slot],common_speculative_get_last_type(spec.get(),slot));
            std::vector<uint8_t> state;
            t.assert_true(common_speculative_get_state(spec.get(),slot,state) && state==carries[slot]);
        }
        return true;
    };
    for (uint32_t slot=0;slot<2;++slot) for (uint32_t row=0;row<lengths[slot];++row)
        histories[slot].push_back(llama_token(100+slot*31+row%13));
    const auto prompt=[&](common_batch & input) {
        for (llama_seq_id slot=0;slot<2;++slot) for (uint32_t row=0;row<lengths[slot];++row)
            input.add(histories[slot][row],llama_pos(row),slot,row+1==lengths[slot]);
    };
    const auto initialize_driver=[&](llama_context & target,llama_context & child,common_speculative_ptr & spec) {
        common_params_speculative parameters;parameters.types={COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
        parameters.draft.ctx_tgt=&target;parameters.draft.ctx_dft=&child;
        parameters.draft.n_max=maximum_depth;parameters.draft.n_min=cutoff==2 ? 2 : 0;
        parameters.draft.p_min=cutoff==1 ? 1.0f : 0.0f;parameters.draft.backend_sampling=backend_sampling;
        spec.reset(common_speculative_init(parameters,2));
        return t.assert_true(spec!=nullptr);
    };
    const auto execute=[&](llama_context & target,llama_context & child,common_speculative_ptr & spec) {
        if (!t.assert_true("Qwen owns its MTP cache even when its workspace belongs to the target",llama_get_ctx_other(&child)==nullptr)) return false;
        for (auto & proposal : proposals) proposal.clear();
        if (spec) {
            if (!t.assert_true(common_speculative_rebind(spec.get(),&target,&child))) return false;
        } else if (!initialize_driver(target,child,spec)) return false;
        common_batch input(&target);prompt(input);
        trace.phase="prefill";
        const auto decoded=llama_process(&target,LLAMA_PROCESS_TYPE_DECODE,input.get());trace.check();
        if (!t.assert_equal(0,decoded)) return false;
        trace.phase="catchup";
        const auto caught_up=common_speculative_process(spec.get(),input);trace.check();
        if (!t.assert_true(caught_up)) return false;
        for (llama_seq_id slot=0;slot<2;++slot) {
            const auto row=slot ? lengths[0]+lengths[1]-1 : lengths[0]-1;
            const auto * logits=llama_get_logits_ith(&target,int32_t(row));const auto count=model->vocab.n_tokens();
            if (!t.assert_true(logits && std::all_of(logits,logits+count,[](float value) {return std::isfinite(value);}))) return false;
            oracle.bytes("anchor/"+std::to_string(slot),logits,size_t(count)*sizeof(float));
            const auto next=llama_token(std::max_element(logits,logits+count)-logits);
            anchors[slot]=next;
            common_speculative_begin(spec.get(),slot,histories[slot]);
            common_speculative_get_draft_params(spec.get(),slot)={true,draft_widths[slot],llama_pos(lengths[slot]),next,
                &histories[slot],&proposals[slot],nullptr,0.0f,123};
        }
        trace.phase="predict";
        common_speculative_draft(spec.get());llama_synchronize(&child);trace.check();
        if (backend_sampling && !t.assert_true("drafting used real backend candidates rather than CPU fallback",
                llama_get_sampled_candidates_count_ith(&child,0)==10 && llama_get_sampled_candidates_ith(&child,0))) return false;
        if (!t.assert_equal(size_t(expected_widths[0]),proposals[0].size()) || !t.assert_equal(size_t(expected_widths[1]),proposals[1].size())) return false;
        for (llama_seq_id slot=0;slot<2;++slot) {
            std::vector<uint8_t> carry;
            if (!t.assert_true(common_speculative_get_state(spec.get(),slot,carry) && !carry.empty())) return false;
            oracle.bytes("proposal/"+std::to_string(slot),proposals[slot].data(),proposals[slot].size()*sizeof(llama_token));
            oracle.bytes("carry/"+std::to_string(slot),carry.data(),carry.size());
            carries[slot]=std::move(carry);
        }
        trace.phase.clear();return true;
    };
    const auto save=[&](llama_context & context,llama_seq_id sequence,std::vector<uint8_t> & bytes) {
        context.synchronize();bytes.resize(llama_state_seq_get_size(&context,sequence));
        return t.assert_true(!bytes.empty()) &&
            t.assert_equal(bytes.size(),llama_state_seq_get_data(&context,bytes.data(),bytes.size(),sequence));
    };
    const auto load=[&](llama_context & context,llama_seq_id sequence,const std::vector<uint8_t> & bytes) {
        const auto read=llama_state_seq_set_data(&context,bytes.data(),bytes.size(),sequence);
        context.synchronize();return t.assert_equal(bytes.size(),read);
    };
    const auto verify_batch=[&](common_batch & batch) {
        for (llama_seq_id slot=0;slot<2;++slot) {
            batch.add(anchors[slot],llama_pos(lengths[slot]),slot,true);
            for (size_t row=0;row<proposals[slot].size();++row)
                batch.add(proposals[slot][row],llama_pos(lengths[slot]+row+1),slot,true);
        }
    };
    const auto continue_round=[&](llama_context & target,llama_context & child,common_speculative_ptr & spec) {
        if (!t.assert_true(spec!=nullptr)) return false;
        for (llama_seq_id slot=0;slot<2;++slot) {
            std::vector<uint8_t> restored;
            if (!t.assert_true(common_speculative_get_state(spec.get(),slot,restored) && restored==carries[slot])) return false;
            oracle.bytes("resume/imported-carry/"+std::to_string(slot),restored.data(),restored.size());
            if (!t.assert_true(llama_memory_seq_rm(child.get_memory(),slot,llama_pos(lengths[slot]),-1))) return false;
        }
        common_batch batch(&target);verify_batch(batch);
        llama_set_kv_stream_decode(&target,true);trace.phase="resume/verify";
        const auto result=llama_process(&target,LLAMA_PROCESS_TYPE_DECODE,batch.get());trace.check();
        if (!t.assert_equal(0,result)) return false;
        trace.phase="resume/catchup";
        const auto caught_up=common_speculative_process(spec.get(),batch);trace.check();
        if (!t.assert_true(caught_up)) return false;
        std::array<llama_token,2> corrections{};
        std::array<uint32_t,2> next_positions{};
        for (llama_seq_id slot=0;slot<2;++slot) {
            const auto accepted_count=partial ? accepted_widths[slot] : uint16_t(proposals[slot].size());
            common_speculative_accept(spec.get(),slot,accepted_count);
            std::vector<uint8_t> accepted;
            if (!t.assert_true(common_speculative_get_state(spec.get(),slot,accepted))) return false;
            oracle.bytes("resume/accepted-carry/"+std::to_string(slot),accepted.data(),accepted.size());
            const auto row=(slot ? expected_widths[0]+1 : 0)+accepted_count;const auto * logits=llama_get_logits_ith(&target,row);
            if (!t.assert_true(logits!=nullptr)) return false;
            oracle.bytes("resume/accepted-logits/"+std::to_string(slot),logits,size_t(model->vocab.n_tokens())*sizeof(float));
            corrections[slot]=llama_token(std::max_element(logits,logits+model->vocab.n_tokens())-logits);
            next_positions[slot]=lengths[slot]+1+accepted_count;
        }
        if (partial) {
            common_batch correction(&target);
            for (llama_seq_id slot=0;slot<2;++slot) {
                if (!t.assert_true(llama_memory_seq_rm(target.get_memory(),slot,llama_pos(next_positions[slot]),-1)) ||
                        !t.assert_true(llama_memory_seq_rm(child.get_memory(),slot,llama_pos(next_positions[slot]),-1))) return false;
                correction.add(corrections[slot],llama_pos(next_positions[slot]),slot,true);
                histories[slot].push_back(anchors[slot]);
                histories[slot].insert(histories[slot].end(),proposals[slot].begin(),proposals[slot].begin()+accepted_widths[slot]);
                histories[slot].push_back(corrections[slot]);
            }
            trace.phase="resume/correction";
            const auto corrected=llama_process(&target,LLAMA_PROCESS_TYPE_DECODE,correction.get());trace.check();
            if (!t.assert_equal(0,corrected)) return false;
            trace.phase="resume/correction-catchup";
            const auto processed=common_speculative_process(spec.get(),correction);trace.check();
            if (!t.assert_true(processed)) return false;
            for (llama_seq_id slot=0;slot<2;++slot) {
                const auto * logits=llama_get_logits_ith(&target,slot);
                if (!t.assert_true(logits!=nullptr)) return false;
                oracle.bytes("resume/correction-logits/"+std::to_string(slot),logits,size_t(model->vocab.n_tokens())*sizeof(float));
                const auto anchor=llama_token(std::max_element(logits,logits+model->vocab.n_tokens())-logits);
                proposals[slot].clear();
                common_speculative_get_draft_params(spec.get(),slot)={true,draft_widths[slot],llama_pos(next_positions[slot]+1),anchor,
                    &histories[slot],&proposals[slot],nullptr,0.0f,123};
            }
            trace.phase="resume/correction-predict";
            common_speculative_draft(spec.get());llama_synchronize(&child);trace.check();
            for (llama_seq_id slot=0;slot<2;++slot) {
                if (!t.assert_equal(size_t(expected_widths[slot]),proposals[slot].size())) return false;
                oracle.bytes("resume/correction-proposals/"+std::to_string(slot),proposals[slot].data(),proposals[slot].size()*sizeof(llama_token));
                if (!t.assert_true(common_speculative_get_state(spec.get(),slot,carries[slot]))) return false;
                oracle.bytes("resume/correction-carry/"+std::to_string(slot),carries[slot].data(),carries[slot].size());
            }
        }
        trace.phase.clear();return true;
    };
    {
        llama_context target(*model,cp);llama_set_embeddings_nextn(&target,true,false);
        auto dp=cp;dp.n_rs_seq=0;dp.ctx_type=LLAMA_CONTEXT_TYPE_MTP;dp.ctx_other=&target;
        llama_context child(*model,dp);llama_set_embeddings_nextn(&child,true,true);
        driver_guard parked{native_spec,safe ? safe->metadata_context() : nullptr,safe ? safe->auxiliary_metadata_context() : nullptr};
        if (!execute(target,child,native_spec)) return;
        if (resume) for (llama_seq_id slot=0;slot<2;++slot) {
            if (!save(target,slot,target_states[slot]) || !save(child,slot,child_states[slot])) return;
        }
        t.out<<"native two-slot MTP control: histories="<<lengths[0]<<','<<lengths[1]<<" proposals="<<draft_widths[0]<<','<<draft_widths[1]<<" carry=2\n";
        if (native_restore && !retarget(native_spec,safe->metadata_context(),safe->auxiliary_metadata_context())) return;
        if (resume && backend_sampling) {
            llama_memory_clear(target.get_memory(),true);llama_memory_clear(child.get_memory(),true);
            for (llama_seq_id slot=0;slot<2;++slot)
                if (!load(target,slot,target_states[slot]) || !load(child,slot,child_states[slot])) return;
            if (!continue_round(target,child,native_spec)) return;
            t.out<<"native same-context imported backend MTP verification controls=2\n";
        }
    }
    if (native_restore) {
        llama_context target(*model,cp);llama_set_embeddings_nextn(&target,true,false);
        auto dp=cp;dp.n_rs_seq=0;dp.ctx_type=LLAMA_CONTEXT_TYPE_MTP;dp.ctx_other=&target;
        llama_context child(*model,dp);llama_set_embeddings_nextn(&child,true,true);
        driver_guard parked{native_spec,safe->metadata_context(),safe->auxiliary_metadata_context()};
        for (llama_seq_id slot=0;slot<2;++slot)
            if (!load(target,slot,target_states[slot]) || !load(child,slot,child_states[slot])) return;
        if (!retarget(native_spec,&target,&child) || !continue_round(target,child,native_spec)) return;
        t.out<<"native restored MTP verification controls=2\n";
    }
    if (const auto * path=std::getenv("LLAMA_TEST_GROUP_MTP_ORACLE")) {
        const auto name=std::string(path)+(partial ? "."+std::to_string(draft_widths[0])+"-"+std::to_string(draft_widths[1])+"-"+
            std::to_string(accepted_widths[0])+"-"+std::to_string(accepted_widths[1]) : "");
        std::unique_ptr<FILE,kv_stream_exact_file_free> file(std::fopen(name.c_str(),"wbx"));
        oracle.save_control(file.get(),{0x4d53394d5450,cp.n_ctx,cp.n_batch,cp.n_ubatch,cp.n_seq_max,cp.n_rs_seq,
            uint64_t(cp.type_k),uint64_t(cp.type_v),lengths[0],lengths[1],draft_widths[0],draft_widths[1],
            uint64_t(model->hparams.n_embd_out()),uint64_t(model->vocab.n_tokens()),uint64_t(phase_guard)|(uint64_t(ring)<<1)|
            (uint64_t(partial)<<2)|(uint64_t(backend_sampling)<<3)|(uint64_t(cutoff)<<4)|
            (uint64_t(accepted_widths[0])<<8)|(uint64_t(accepted_widths[1])<<16),uint64_t(resume)});
    }
    oracle.compare();
    if (partial) {
        for (llama_seq_id slot=0;slot<2;++slot) {
            histories[slot].resize(lengths[slot]);
        }
    }
    cp.shared_device_memory_bytes=size_t(1024)*1024*1024;cp.kv_stream_auxiliary_layers=1;
    auto group=llama_context_group::create(*model,cp);
    if (!t.assert_true(group && group->metadata_context())) return;
    if (backend_sampling && !initialize_driver(*group->metadata_context(),*group->auxiliary_metadata_context(),stream_spec)) return;
    auto input=llama_batch_init(int32_t(lengths[0]+lengths[1]),0,1);
    struct release_batch {llama_batch & batch;~release_batch() {llama_batch_free(batch);}} release{input};
    for (llama_seq_id slot=0;slot<2;++slot) for (uint32_t row=0;row<lengths[slot];++row) {
        const auto index=input.n_tokens++;
        input.token[index]=histories[slot][row];input.pos[index]=llama_pos(row);
        input.n_seq_id[index]=1;input.seq_id[index][0]=slot;input.logits[index]=row+1==lengths[slot];
    }
    llama_kv_stream_selection selection;
    if (!t.assert_true(llama_kv_stream_select({cp.n_ctx,2},{{0,0},{1,0}},selection)==llama_kv_stream_selection_status::success)) return;
    if (ring) {
        std::vector<llama_memory_phase_cache_identity> sources,auxiliary;
        for (const auto & handle : group->source_handles()) for (uint32_t role=0;role<2;++role) {
            std::shared_ptr<llama_kv_stream_logical_cache> cache;
            if (!t.assert_true(group->source_table()->get(handle,role,cache))) return;
            (role ? auxiliary : sources).push_back({cache->identity().id,cache->content()->generation()});
        }
        llama_context_group_plan plan;
        const auto quoted=group->metadata_context()->quote_target_group(input,sources,{9001,1},llama_memory_work_phase::target_prefill,
            100,SIZE_MAX,plan,group->auxiliary_metadata_context(),auxiliary);
        if (!t.assert_true(quoted.status==llama_context_memory_create_status::success)) return;
        cp.shared_device_memory_bytes=quoted.required_bytes;
        group=llama_context_group::create(*model,cp);
        if (!t.assert_true(group!=nullptr)) return;
        t.out<<"explicit ring profile: context="<<cp.n_ctx<<" batch="<<cp.n_batch<<" ubatch="<<cp.n_ubatch
            <<" exact_arena="<<cp.shared_device_memory_bytes<<'\n';
    }
    driver_guard parked{stream_spec,safe ? safe->metadata_context() : nullptr,safe ? safe->auxiliary_metadata_context() : nullptr};
    if (!t.assert_true("target and trained MTP head must compose on one admitted group workspace",
            group->activate(selection,group->source_handles(),input,llama_memory_work_phase::target_prefill,1,1))) return;
    auto * target=group->context();auto * child=group->auxiliary_context();
    if (!t.assert_true(target && child && target->group_parent() && target->group_parent()==child->group_parent() &&
            !child->get_cparams().shared_device_memory_bytes)) return;
    if (!execute(*target,*child,stream_spec)) return;
    if (ring) {
        t.assert_true("prefill, catchup and drafting each acquired real nonresident KV spans",
            transport->spans[0]>0 && transport->spans[1]>0 && transport->spans[2]>0);
        t.out<<"actual ring spans: prefill="<<transport->spans[0]<<" catchup="<<transport->spans[1]
            <<" predict="<<transport->spans[2]<<'\n';
        t.assert_equal(cp.shared_device_memory_bytes,ggml_backend_buffer_get_size(target->group_parent()));
    }
    auto * parent=target->group_parent();
    std::vector<llama_kv_stream_logical_checkpoint> checkpoints;
    for (const auto & handle : group->source_handles()) for (uint32_t role=0;role<2;++role) {
        std::shared_ptr<llama_kv_stream_logical_cache> cache;
        if (!t.assert_true(group->source_table()->get(handle,role,cache))) return;
        checkpoints.push_back(cache->checkpoint());
    }
    t.assert_true(llama_kv_stream_mtp_prepare(target,maximum_depth) && llama_kv_stream_mtp_release(target));
    t.assert_true(!llama_kv_stream_mtp_prepare(target,maximum_depth+1) && !llama_kv_stream_mtp_prepare(target,maximum_depth+2));
    if (!retarget(stream_spec,native_restore ? safe->metadata_context() : group->metadata_context(),
            native_restore ? safe->auxiliary_metadata_context() : group->auxiliary_metadata_context())) return;
    if (!t.assert_true("both model roles retire before the parent can be selected again",group->park())) return;
    t.assert_true(!group->context() && !group->auxiliary_context());
    size_t ordinal=0;
    for (const auto & handle : group->source_handles()) for (uint32_t role=0;role<2;++role) {
        std::shared_ptr<llama_kv_stream_logical_cache> cache;
        if (!t.assert_true(group->source_table()->get(handle,role,cache))) return;
        const auto before=checkpoints[ordinal++],after=cache->checkpoint();
        t.assert_true(cache->ready() && before.id==after.id && before.generation==after.generation && before.tokens==after.tokens);
    }
    if (resume) {
        if (!t.assert_true(llama_kv_stream_select({512,2},{{0,lengths[0]+1},{1,lengths[1]+1}},selection)==llama_kv_stream_selection_status::success)) return;
        input.n_tokens=0;
        for (llama_seq_id slot=0;slot<2;++slot) {
            const auto first=input.n_tokens++;
            input.token[first]=anchors[slot];input.pos[first]=llama_pos(lengths[slot]);input.n_seq_id[first]=1;
            input.seq_id[first][0]=slot;input.logits[first]=true;
            for (size_t row=0;row<proposals[slot].size();++row) {
                const auto index=input.n_tokens++;
                input.token[index]=proposals[slot][row];input.pos[index]=llama_pos(lengths[slot]+row+1);
                input.n_seq_id[index]=1;input.seq_id[index][0]=slot;input.logits[index]=true;
            }
        }
        if (!t.assert_true("parked target and auxiliary histories resume together on the original parent",
                group->activate(selection,group->source_handles(),input,llama_memory_work_phase::target_verify,1,2))) return;
        target=group->context();child=group->auxiliary_context();
        if (!t.assert_true(target && child && target->group_parent()==parent && child->group_parent()==parent)) return;
        if (phase_guard) {
            std::vector<llama_context_group_sequence_state> before,after;
            if (!t.assert_true(target->save_group_sequences(before))) return;
            llama_set_kv_stream_decode(target,false);
            if (!t.assert_equal("an undeclared prefill cannot mutate a restored verification group",-1,target->decode(input))) return;
            llama_set_kv_stream_decode(target,true);
            if (!t.assert_true(target->save_group_sequences(after) && before.size()==after.size())) return;
            for (size_t i=0;i<before.size();++i) {
                const auto & a=before[i];const auto & b=after[i];
                t.assert_true(a.model==b.model && a.sequence==b.sequence && a.request_epoch==b.request_epoch &&
                    a.cache_id==b.cache_id && a.cache_generation==b.cache_generation && a.tokens==b.tokens &&
                    a.recurrent==b.recurrent && a.bounded==b.bounded && a.full_columns==b.full_columns);
            }
        }
        if (!retarget(stream_spec,target,child) || !continue_round(*target,*child,stream_spec)) return;
    }
    oracle.finish();
    t.out<<"streamed joint MTP exact_records="<<oracle.records()<<" bytes="<<oracle.byte_count()<<'\n';
}
static void test_group_gemma_assistant_metadata(testing & t,llama_model * model,const char * path,bool late_attachment=false,
        bool joint_quote=false,bool joint_factory=false,bool joint_execution=false,bool manager_mode=false,bool ring=false,
        std::array<uint16_t,2> widths={1,3},std::array<int16_t,2> accepted={-1,-1}) {
    if (!t.assert_true(model && model->arch==LLM_ARCH_GEMMA4 && path)) return;
    const bool acceptance=accepted[0]>=0 || accepted[1]>=0;
    if (acceptance && !t.assert_true(accepted[0]>=0 && accepted[1]>=0 && accepted[0]<=widths[0] && accepted[1]<=widths[1])) return;
    const auto maximum=*std::max_element(widths.begin(),widths.end());
    common_params declared;
    declared.n_ctx=ring ? 6400 : 4096;declared.n_batch=ring ? 6400 : 256;declared.n_ubatch=ring ? 32 : 256;declared.n_parallel=2;declared.kv_unified=true;
    declared.shared_device_memory_bytes=size_t(1024)*1024*1024;
    declared.cache_type_k=GGML_TYPE_Q8_0;declared.cache_type_v=GGML_TYPE_Q4_0;
    declared.speculative.types={COMMON_SPECULATIVE_TYPE_DRAFT_MTP};declared.speculative.draft.n_max=3;
    declared.speculative.draft.mparams.path=path;
    const auto resolved=common_context_params_to_llama(declared,model);
    if (!t.assert_true("model-only Gemma startup derives shared verification without rollback snapshots",
            resolved.kv_stream_verify_max==4 && !resolved.kv_stream_auxiliary_layers && !resolved.n_rs_seq &&
            resolved.n_seq_max==2 && resolved.kv_unified && resolved.shared_device_memory_bytes==declared.shared_device_memory_bytes &&
            resolved.type_k==GGML_TYPE_Q8_0 && resolved.type_v==GGML_TYPE_Q4_0 && !declared.kv_stream_auxiliary_layers)) return;
    auto mp=llama_model_default_params();mp.n_gpu_layers=999;
    model_ptr assistant(llama_model_load_from_file(path,mp),llama_model_free);
    if (!t.assert_true(assistant && assistant->arch==LLM_ARCH_GEMMA4_ASSISTANT)) return;
    auto cp=llama_context_default_params();
    cp.n_ctx=ring ? 6400 : 4096;cp.n_batch=ring ? 6400 : 256;cp.n_ubatch=ring ? 32 : 256;cp.n_seq_max=2;cp.kv_unified=true;cp.swa_full=false;
    cp.type_k=GGML_TYPE_Q8_0;cp.type_v=GGML_TYPE_Q4_0;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.n_outputs_max=8;cp.n_outputs_max_per_seq=4;cp.n_threads=cp.n_threads_batch=4;
    const std::array<uint32_t,2> lengths={ring ? 3073u : 5u,ring ? 3075u : 7u};
    const std::array<int32_t,2> last={int32_t(lengths[0]-1),int32_t(lengths[0]+lengths[1]-1)};
    size_t ring_budget=0;
    kv_stream_bit_oracle oracle;
    std::array<llama_tokens,2> native_proposals;
    std::array<std::vector<uint8_t>,2> native_carries;
    std::array<llama_tokens,2> native_resumed_proposals;
    std::array<std::vector<uint8_t>,2> native_resumed_carries;
    kv_stream_exact_trace target_trace(oracle,"target"),child_trace(oracle,"draft");
    std::unique_ptr<group_ring_probe> transport;
    if (ring) transport=std::make_unique<group_ring_probe>(model->devices.front().dev,child_trace.phase);
    const auto bounded_image=[&](llama_context & target) {
        std::array<std::vector<uint8_t>,2> result;
        target.synchronize();
        for (llama_seq_id slot=0;slot<2;++slot) {
            image_writer writer;target.get_kv_resources().bounded->state_write(writer,slot,0);
            result[slot]=std::move(writer.bytes);
        }
        return result;
    };
    const auto continue_target=[&](llama_context & target,common_speculative * spec=nullptr,std::array<llama_tokens,2> * histories=nullptr,
            llama_context * assistant_context=nullptr) {
        if (acceptance) {
            if (!t.assert_true(spec && histories && assistant_context)) return false;
            common_batch verification(&target);std::array<llama_token,2> anchors{},bonus{};
            const std::array<int32_t,2> offsets={0,int32_t(widths[0]+1)};
            for (llama_seq_id slot=0;slot<2;++slot) {
                const auto * logits=llama_get_logits_ith(&target,last[slot]);
                if (!t.assert_true(logits && native_proposals[slot].size()==widths[slot])) return false;
                anchors[slot]=llama_token(std::max_element(logits,logits+model->vocab.n_tokens())-logits);
                for (uint32_t row=0;row<=widths[slot];++row)
                    verification.add(row ? native_proposals[slot][row-1] : anchors[slot],llama_pos(lengths[slot]+row),slot,true);
            }
            llama_set_kv_stream_decode(&target,true);target_trace.phase="accepted-verification";
            if (!t.assert_equal(0,llama_process(&target,LLAMA_PROCESS_TYPE_DECODE,verification.get()))) return false;
            target_trace.check();target_trace.phase.clear();
            if (!t.assert_true(common_speculative_process(spec,verification))) return false;
            for (llama_seq_id slot=0;slot<2;++slot) {
                const auto * logits=llama_get_logits_ith(&target,offsets[slot]+accepted[slot]);
                if (!t.assert_true(logits!=nullptr)) return false;
                bonus[slot]=llama_token(std::max_element(logits,logits+model->vocab.n_tokens())-logits);
                common_speculative_accept(spec,slot,uint16_t(accepted[slot]));
                if (!t.assert_true(llama_memory_seq_rm(target.get_memory(),slot,llama_pos(lengths[slot]+1+accepted[slot]),-1))) return false;
                (*histories)[slot].push_back(anchors[slot]);
                (*histories)[slot].insert((*histories)[slot].end(),native_proposals[slot].begin(),native_proposals[slot].begin()+accepted[slot]);
            }
            const auto before=bounded_image(target);
            std::array<llama_tokens,2> proposals;
            for (llama_seq_id slot=0;slot<2;++slot)
                common_speculative_get_draft_params(spec,slot)={true,widths[slot],llama_pos((*histories)[slot].size()),bonus[slot],
                    &(*histories)[slot],&proposals[slot],nullptr,0.0f,123};
            child_trace.phase="accepted-predict";common_speculative_draft(spec);llama_synchronize(assistant_context);child_trace.check();child_trace.phase.clear();
            if (!t.assert_true(before==bounded_image(target))) return false;
            for (llama_seq_id slot=0;slot<2;++slot) {
                std::vector<uint8_t> carry;
                if (!t.assert_true(proposals[slot].size()==widths[slot] && common_speculative_get_state(spec,slot,carry) &&
                        llama_memory_seq_pos_max(target.get_memory(),slot)==llama_pos((*histories)[slot].size()-1))) return false;
                oracle.bytes("accepted/proposals/"+std::to_string(slot),proposals[slot].data(),proposals[slot].size()*sizeof(llama_token));
                oracle.bytes("accepted/carry/"+std::to_string(slot),carry.data(),carry.size());
            }
            common_batch correction(&target);
            for (llama_seq_id slot=0;slot<2;++slot) {
                correction.add(bonus[slot],llama_pos((*histories)[slot].size()),slot,true);(*histories)[slot].push_back(bonus[slot]);
            }
            target_trace.phase="accepted-correction";
            if (!t.assert_equal(0,llama_process(&target,LLAMA_PROCESS_TYPE_DECODE,correction.get()))) return false;
            target_trace.check();target_trace.phase.clear();
            if (!t.assert_true(common_speculative_process(spec,correction))) return false;
            for (llama_seq_id slot=0;slot<2;++slot) common_speculative_accept(spec,slot,0);
            return true;
        }
        common_batch next(&target);
        for (llama_seq_id slot=0;slot<2;++slot) {
            const auto * logits=llama_get_logits_ith(&target,last[slot]);
            if (!t.assert_true(logits!=nullptr)) return false;
            const auto token=llama_token(std::max_element(logits,logits+model->vocab.n_tokens())-logits);
            next.add(token,lengths[slot],slot,true);
            if (histories) (*histories)[slot].push_back(token);
        }
        llama_set_kv_stream_decode(&target,true);target_trace.phase="target-resume";
        const auto result=llama_process(&target,LLAMA_PROCESS_TYPE_DECODE,next.get());target_trace.check();
        target_trace.phase.clear();
        if (!t.assert_equal("target continues after borrowed assistant reads",0,result)) return false;
        if (spec) {
            if (!t.assert_true(common_speculative_process(spec,next))) return false;
            for (llama_seq_id slot=0;slot<2;++slot) common_speculative_accept(spec,slot,0);
        }
        return true;
    };
    const auto resumed_predict=[&](llama_context & target,llama_context & child,common_speculative * spec,
            std::array<llama_tokens,2> & histories,std::array<llama_tokens,2> & proposals,
            const std::array<llama_token,2> & anchors,bool control) {
        for (llama_seq_id slot=0;slot<2;++slot) {
            proposals[slot].clear();
            common_speculative_get_draft_params(spec,slot)={true,widths[slot],llama_pos(histories[slot].size()),anchors[slot],
                &histories[slot],&proposals[slot],nullptr,0.0f,123};
        }
        child_trace.phase="resumed-predict";common_speculative_draft(spec);llama_synchronize(&child);child_trace.check();
        child_trace.phase.clear();
        for (llama_seq_id slot=0;slot<2;++slot) {
            std::vector<uint8_t> carry;
            if (!t.assert_true(common_speculative_get_state(spec,slot,carry))) return false;
            if (control) {native_resumed_proposals[slot]=proposals[slot];native_resumed_carries[slot]=std::move(carry);}
            else if (!t.assert_true(proposals[slot]==native_resumed_proposals[slot] && carry==native_resumed_carries[slot])) return false;
            if (!t.assert_equal(llama_pos(histories[slot].size()-1),llama_memory_seq_pos_max(target.get_memory(),slot))) return false;
        }
        return true;
    };
    cp.cb_eval=kv_stream_exact_trace::callback;cp.cb_eval_user_data=&target_trace;
    for (bool late : {false,true}) {
        if (late && !late_attachment) break;
        llama_context target(*model,cp);llama_set_embeddings_nextn(&target,true,false);
        auto dp=cp;dp.ctx_type=LLAMA_CONTEXT_TYPE_MTP;dp.ctx_other=&target;
        dp.n_batch=dp.n_ubatch=2;dp.n_outputs_max=2;dp.n_outputs_max_per_seq=1;
        dp.cb_eval_user_data=&child_trace;
        std::unique_ptr<llama_context> child;
        if (!late) child=std::make_unique<llama_context>(*assistant,dp);
        std::array<llama_tokens,2> histories,proposals;
        common_batch prompt(&target);
        for (llama_seq_id slot=0;slot<2;++slot) for (uint32_t row=0;row<uint32_t(lengths[slot]);++row) {
            const auto token=llama_token(100+slot*20+row);histories[slot].push_back(token);
            prompt.add(token,llama_pos(row),slot,row==uint32_t(lengths[slot]-1));
        }
        target_trace.phase="prefill";
        if (!t.assert_equal(0,llama_process(&target,LLAMA_PROCESS_TYPE_DECODE,prompt.get()))) return;
        target_trace.check();target_trace.phase.clear();
        if (late) {
            if (!t.assert_equal(llama_pos(lengths[0]-1),llama_memory_seq_pos_max(target.get_memory(),0)) ||
                    !t.assert_equal(llama_pos(lengths[1]-1),llama_memory_seq_pos_max(target.get_memory(),1))) return;
            const auto image=[&](llama_seq_id slot,std::vector<uint8_t> & bytes) {
                bytes.resize(llama_state_seq_get_size(&target,slot));
                return t.assert_true(!bytes.empty()) &&
                    t.assert_equal(bytes.size(),llama_state_seq_get_data(&target,bytes.data(),bytes.size(),slot));
            };
            std::array<std::vector<uint8_t>,2> before,after;
            for (llama_seq_id slot=0;slot<2;++slot) if (!image(slot,before[slot])) return;
            auto wrong=dp;wrong.kv_unified=false;bool rejected=false;
            try {llama_context incompatible(*assistant,wrong);}
            catch (const std::runtime_error & error) {
                rejected=std::string(error.what())=="shared KV source cell geometry does not match the reader";
            }
            if (!t.assert_true("mismatched shared stream geometry rejects without resetting the source",rejected)) return;
            for (llama_seq_id slot=0;slot<2;++slot) {
                if (!image(slot,after[slot]) || !t.assert_true(before[slot]==after[slot])) return;
            }
            child=std::make_unique<llama_context>(*assistant,dp);
            if (!t.assert_equal("attaching a shared reader preserves already committed source history",llama_pos(lengths[0]-1),
                    llama_memory_seq_pos_max(target.get_memory(),0)) ||
                    !t.assert_equal(llama_pos(lengths[1]-1),llama_memory_seq_pos_max(target.get_memory(),1))) return;
            for (llama_seq_id slot=0;slot<2;++slot) {
                if (!image(slot,after[slot]) || !t.assert_true("all source KV and metadata bytes survive reader attachment",before[slot]==after[slot])) return;
            }
        }
        llama_set_embeddings_nextn(child.get(),true,true);
        if (!t.assert_true(llama_get_ctx_other(child.get())==&target)) return;
        for (const auto & entry : child->get_memory()->memory_breakdown())
            if (!t.assert_equal(size_t(0),entry.second)) return;
        common_params_speculative parameters;parameters.types={COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
        parameters.draft.ctx_tgt=&target;parameters.draft.ctx_dft=child.get();
        parameters.draft.n_max=maximum;parameters.draft.n_min=0;parameters.draft.p_min=0;parameters.draft.backend_sampling=false;
        common_speculative_ptr spec(common_speculative_init(parameters,2));
        if (!t.assert_true(spec!=nullptr)) return;
        if (!t.assert_true(common_speculative_process(spec.get(),prompt))) return;
        for (llama_seq_id slot=0;slot<2;++slot) {
            const auto * logits=llama_get_logits_ith(&target,last[slot]);
            if (!t.assert_true(logits!=nullptr)) return;
            const auto next=llama_token(std::max_element(logits,logits+model->vocab.n_tokens())-logits);
            common_speculative_begin(spec.get(),slot,histories[slot]);
            common_speculative_get_draft_params(spec.get(),slot)={true,widths[slot],llama_pos(histories[slot].size()),next,
                &histories[slot],&proposals[slot],nullptr,0.0f,123};
        }
        const auto before=joint_execution ? bounded_image(target) : std::array<std::vector<uint8_t>,2>{};
        std::unique_ptr<joint_attention_probe> joint;
        if (joint_execution) joint=std::make_unique<joint_attention_probe>(*child,model->hparams.n_embd_head_k(model->hparams.n_layer()-1));
        child_trace.phase="predict";common_speculative_draft(spec.get());llama_synchronize(child.get());child_trace.check();child_trace.phase.clear();
        if (!t.assert_equal(size_t(widths[0]),proposals[0].size()) || !t.assert_equal(size_t(widths[1]),proposals[1].size())) return;
        t.assert_equal(llama_pos(lengths[0]-1),llama_memory_seq_pos_max(target.get_memory(),0));
        t.assert_equal(llama_pos(lengths[1]-1),llama_memory_seq_pos_max(target.get_memory(),1));
        if (joint_execution) {
            if (!t.assert_true("native assistant submits joint full-attention query rows",joint->joint>0)) return;
            joint.reset();
            if (!t.assert_true(!before[0].empty() && before==bounded_image(target))) return;
            native_proposals=proposals;
            for (llama_seq_id slot=0;slot<2;++slot)
                if (!t.assert_true(common_speculative_get_state(spec.get(),slot,native_carries[slot]))) return;
            if (!continue_target(target,manager_mode ? spec.get() : nullptr,manager_mode ? &histories : nullptr,child.get())) return;
            if (manager_mode) {
                std::array<llama_token,2> anchors{};std::array<std::vector<uint8_t>,2> states;
                for (llama_seq_id slot=0;slot<2;++slot) {
                    const auto * logits=llama_get_logits_ith(&target,slot);
                    if (!t.assert_true(logits!=nullptr)) return;
                    anchors[slot]=llama_token(std::max_element(logits,logits+model->vocab.n_tokens())-logits);
                    states[slot].resize(llama_state_seq_get_size(&target,slot));
                    if (!t.assert_true(!states[slot].empty() && llama_state_seq_get_data(&target,states[slot].data(),states[slot].size(),slot)==states[slot].size())) return;
                }
                llama_memory_clear(target.get_memory(),true);
                for (llama_seq_id slot=0;slot<2;++slot) if (!t.assert_equal(states[slot].size(),
                        llama_state_seq_set_data(&target,states[slot].data(),states[slot].size(),slot))) return;
                if (!resumed_predict(target,*child,spec.get(),histories,proposals,anchors,true)) return;
            }
        }
        t.out<<"native Gemma joint assistant: TG1 per source, widths="<<widths[0]<<","<<widths[1]
            <<" accepted="<<accepted[0]<<","<<accepted[1]<<", exact finite records="<<oracle.records()
            <<" bytes="<<oracle.byte_count()<<std::endl;
        if (late_attachment && !late) oracle.compare();
    }
    if (late_attachment) {oracle.finish();return;}
    const auto * control_path=std::getenv("LLAMA_TEST_GEMMA_GROUP_ORACLE");
    if (!control_path && ring) control_path=std::getenv("LLAMA_TEST_GEMMA_GROUP_RING_ORACLE");
    if (control_path) {
        const auto name=std::string(control_path)+(acceptance ? "."+std::to_string(widths[0])+"-"+std::to_string(widths[1])+"-"+
            std::to_string(accepted[0])+"-"+std::to_string(accepted[1]) : "");
        std::unique_ptr<FILE,kv_stream_exact_file_free> file(std::fopen(name.c_str(),"wbx"));
        oracle.save_control(file.get(),{0x4d5339475247,cp.n_ctx,cp.n_batch,cp.n_ubatch,cp.n_seq_max,0,
            uint64_t(cp.type_k),uint64_t(cp.type_v),lengths[0],lengths[1],widths[0],widths[1],
            uint64_t(model->hparams.n_embd_out()),uint64_t(model->vocab.n_tokens()),
            acceptance ? uint64_t(1)|(uint64_t(ring)<<1)|(uint64_t(accepted[0])<<8)|(uint64_t(accepted[1])<<16) : 0,1});
    }
    cp.shared_device_memory_bytes=size_t(1024)*1024*1024;cp.kv_stream_verify_max=4;
    auto group=llama_context_group::create(*model,cp);
    if (!t.assert_true(group && group->metadata_context())) return;
    if (joint_quote && !manager_mode) llama_set_embeddings_nextn(group->metadata_context(),true,false);
    auto dp=cp;dp.shared_device_memory_bytes=0;dp.kv_stream_verify_max=0;
    dp.ctx_type=LLAMA_CONTEXT_TYPE_MTP;dp.ctx_other=group->metadata_context();
    dp.n_batch=dp.n_ubatch=2;dp.n_outputs_max=2;dp.n_outputs_max_per_seq=1;
    dp.cb_eval_user_data=&child_trace;
    if (manager_mode) {
        const auto hidden=group->metadata_context()->get_cparams().embeddings_nextn;
        auto bad=dp;bad.type_v=GGML_TYPE_Q8_0;
        t.assert_true(!group->prepare_shared_mtp(*model,dp) && !group->prepare_shared_mtp(*assistant,bad) &&
            !group->auxiliary_metadata_context() && hidden==group->metadata_context()->get_cparams().embeddings_nextn);
        if (!t.assert_true("manager declares the external shared assistant before parent allocation",
                group->prepare_shared_mtp(*assistant,dp))) return;
        t.assert_true(group->auxiliary_metadata_context() && group->metadata_context()->get_cparams().embeddings_nextn &&
            !group->prepare_shared_mtp(*assistant,dp) && !group->context() && !group->auxiliary_context());
    }
    std::unique_ptr<llama_context> child;
    try {child=std::make_unique<llama_context>(*assistant,dp,std::vector<llama_seq_id>{0,1});}
    catch (const std::runtime_error & error) {t.out<<"selected assistant declaration: "<<error.what()<<'\n';}
    if (!t.assert_true("Gemma readonly assistant declares selected target aliases without owned KV",child!=nullptr)) return;
    t.assert_true(child->group_storage_pending() && !child->group_parent());
    common_batch blocked(child.get());blocked.add(100,0,0,true);
    t.assert_equal("readonly declaration cannot execute before common-parent admission",-1,
        llama_process(child.get(),LLAMA_PROCESS_TYPE_DECODE,blocked.get()));
    for (const auto & entry : child->get_memory()->memory_breakdown()) t.assert_equal(size_t(0),entry.second);
    const auto target_resources=group->metadata_context()->get_kv_resources();
    const auto check_aliases=[&](const llama_context & context) {
        const auto resources=context.get_kv_resources();
        if (!t.assert_true(resources.full && resources.bounded && !resources.recurrent &&
                context.group_target_context()==group->metadata_context() && !context.get_compute_memory())) return false;
        for (uint32_t il=0;il<assistant->hparams.n_layer_all;++il) {
            const bool local=assistant->hparams.is_swa(il);const auto source=model->hparams.n_layer()-(local ? 2 : 1);
            auto * own=local ? resources.bounded : resources.full;auto * parent=local ? target_resources.bounded : target_resources.full;
            if (!t.assert_true(own->get_k_storage(il)==parent->get_k_storage(source) &&
                    own->get_v_storage(il)==parent->get_v_storage(source))) return false;
        }
        return true;
    };
    if (!check_aliases(*child)) return;
    for (const auto & selected : {std::vector<llama_seq_id>{0},std::vector<llama_seq_id>{1}}) {
        llama_context subset(*assistant,dp,selected);
        if (!check_aliases(subset)) return;
        for (const auto & entry : subset.get_memory()->memory_breakdown()) t.assert_equal(size_t(0),entry.second);
    }
    const auto hidden_before=group->metadata_context()->get_cparams().embeddings_nextn;
    for (uint32_t invalid=0;invalid<12;++invalid) {
        auto bad=dp;std::vector<llama_seq_id> selected{0,1};
        if (invalid==0) bad.ctx_other=nullptr;
        if (invalid==1) bad.n_rs_seq=1;
        if (invalid==2) bad.shared_device_memory_bytes=1048576;
        if (invalid==3) bad.kv_stream_auxiliary_layers=1;
        if (invalid==4) bad.type_v=GGML_TYPE_Q8_0;
        if (invalid==5) bad.n_ubatch=1;
        if (invalid==6) bad.n_ctx=2048;
        if (invalid==7) selected={2};
        if (invalid==8) bad.n_seq_max=1;
        if (invalid==9) bad.swa_full=true;
        if (invalid==10) bad.attention_type=LLAMA_ATTENTION_TYPE_NON_CAUSAL;
        if (invalid==11) bad.embeddings=true;
        bool rejected=false;
        try {llama_context wrong(*assistant,bad,selected);}
        catch (const std::runtime_error &) {rejected=true;}
        t.assert_true("unsupported readonly declaration rejects before changing target metadata",rejected);
        t.assert_equal(hidden_before,group->metadata_context()->get_cparams().embeddings_nextn);
    }
    for (const auto & handle : group->source_handles()) {
        std::shared_ptr<llama_kv_stream_logical_cache> cache;std::shared_ptr<llama_kv_cells> cells;
        if (!t.assert_true(group->source_table()->get(handle,0,cache) && group->source_table()->get_cells(handle,0,cells))) return;
        t.assert_true(cache->ready() && !cache->tokens() && !cache->host()->committed_bytes() && !cells->get_used());
        t.assert_true(!group->source_table()->get(handle,1,cache));
    }
    if (joint_quote) {
        llama_set_embeddings_nextn(child.get(),true,true);
        auto input=llama_batch_init(joint_execution ? int32_t(lengths[0]+lengths[1]) : 2,0,1);
        struct release_batch {llama_batch & batch;~release_batch() {llama_batch_free(batch);}} release{input};
        input.n_tokens=2;std::vector<llama_memory_phase_cache_identity> sources;
        for (llama_seq_id slot=0;slot<2;++slot) {
            input.token[slot]=100+slot;input.pos[slot]=0;input.n_seq_id[slot]=1;input.seq_id[slot][0]=slot;input.logits[slot]=true;
            std::shared_ptr<llama_kv_stream_logical_cache> cache;
            if (!t.assert_true(group->source_table()->get(group->source_handles()[slot],0,cache))) return;
            sources.push_back({cache->identity().id,cache->content()->generation()});
        }
        if (ring) {
            input.n_tokens=0;
            for (llama_seq_id slot=0;slot<2;++slot) for (uint32_t row=0;row<lengths[slot];++row) {
                const auto index=input.n_tokens++;
                input.token[index]=llama_token(100+slot*20+row);input.pos[index]=llama_pos(row);
                input.n_seq_id[index]=1;input.seq_id[index][0]=slot;input.logits[index]=row+1==lengths[slot];
            }
        }
        std::vector<std::pair<ggml_tensor *,ggml_backend_buffer_t>> buffers;
        for (auto * cache : {target_resources.full,target_resources.bounded}) for (auto layer : cache->get_layer_ids())
            for (auto * tensor : {cache->get_k_storage(layer),cache->get_v_storage(layer)}) buffers.emplace_back(tensor,tensor->buffer);
        llama_context_group_plan base;
        const auto native_target=group->metadata_context()->quote_target_group(input,sources,{9001,1},
            llama_memory_work_phase::target_prefill,100,SIZE_MAX,base);
        if (!t.assert_true(native_target.status==llama_context_memory_create_status::success)) return;
        llama_context_group_plan plan;
        const auto quote=group->metadata_context()->quote_target_group(input,sources,{9001,1},
            llama_memory_work_phase::target_prefill,100,SIZE_MAX,plan,child.get(),{});
        if (!t.assert_true("common arena quote includes readonly assistant graphs without auxiliary KV histories",
                quote.status==llama_context_memory_create_status::success)) return;
        t.assert_equal(sources.size()+1,plan.caches.size());
        t.assert_true(plan.caches.back().identity.id==9001 &&
            plan.caches.back().role==llama_memory_cache_role::bounded_resident);
        t.assert_equal(base.bounded.allocation.bytes,plan.bounded.allocation.bytes);
        t.assert_equal(base.target.pool_minimum_bytes,plan.target.pool_minimum_bytes);
        size_t reads=0;
        for (const auto & profile : plan.catalog) for (const auto & member : profile.measurement.workload.members)
            if (member.phase==llama_memory_work_phase::mtp_draft) {
                ++reads;t.assert_true(!member.writes_cache && member.input_rows==1 && member.query_rows==1 && member.output_rows==1);
                t.assert_true(std::any_of(sources.begin(),sources.end(),[&](const auto & source) {
                    return source.id==member.cache.id && source.generation==member.cache.generation;
                }));
            }
        t.assert_equal(size_t(4),reads);
        llama_context_group_plan sentinel;sentinel.current_state_bytes=123;
        const auto short_quote=group->metadata_context()->quote_target_group(input,sources,{9001,1},
            llama_memory_work_phase::target_prefill,100,quote.required_bytes-1,sentinel,child.get(),{});
        t.assert_true(short_quote.status==llama_context_memory_create_status::insufficient_budget &&
            short_quote.required_bytes==quote.required_bytes && short_quote.additional_bytes==1 &&
            short_quote.available_bytes==quote.required_bytes-1 && sentinel.current_state_bytes==123 && sentinel.catalog.empty());
        const auto exact=group->metadata_context()->quote_target_group(input,sources,{9001,1},
            llama_memory_work_phase::target_prefill,100,quote.required_bytes,sentinel,child.get(),{});
        t.assert_true(exact.status==llama_context_memory_create_status::success && exact.required_bytes==quote.required_bytes);
        sentinel={};sentinel.current_state_bytes=123;
        const auto foreign=group->metadata_context()->quote_target_group(input,sources,{9001,1},
            llama_memory_work_phase::target_prefill,100,SIZE_MAX,sentinel,child.get(),sources);
        t.assert_true(foreign.status==llama_context_memory_create_status::invalid_configuration &&
            sentinel.current_state_bytes==123 && sentinel.catalog.empty());
        llama_context_group_workload_quote catchup;catchup.writer_bytes=123;
        t.assert_true(!child->measure_target_workload(input,sources,llama_memory_work_phase::mtp_catchup,100,catchup) &&
            catchup.writer_bytes==123 && catchup.measurement.workload.members.empty());
        for (const auto & buffer : buffers) t.assert_true(buffer.first->buffer==buffer.second);
        if (!check_aliases(*child)) return;
        for (const auto & handle : group->source_handles()) {
            std::shared_ptr<llama_kv_stream_logical_cache> cache;
            if (!t.assert_true(group->source_table()->get(handle,0,cache))) return;
            t.assert_true(!cache->tokens() && !cache->host()->committed_bytes());
        }
        t.out<<"Gemma readonly joint quote: required="<<quote.required_bytes<<" target_only="<<native_target.required_bytes
            <<" bounded="<<plan.bounded.allocation.bytes<<" read_members="<<reads<<" exact_and_one_byte_short\n";
        if (ring) {
            if (!t.assert_true(quote.required_bytes<=cp.shared_device_memory_bytes)) return;
            ring_budget=quote.required_bytes;child.reset();group.reset();
            cp.shared_device_memory_bytes=ring_budget;
            group=llama_context_group::create(*model,cp);dp.ctx_other=group ? group->metadata_context() : nullptr;
            if (!t.assert_true(group && group->prepare_shared_mtp(*assistant,dp))) return;
            t.out<<"Gemma additional ring profile: context="<<cp.n_ctx<<" prefixes="<<lengths[0]<<","<<lengths[1]
                <<" batch="<<cp.n_batch<<" ubatch="<<cp.n_ubatch<<" exact_arena="<<ring_budget
                <<" pool_minimum="<<plan.target.pool_minimum_bytes<<std::endl;
        }
        if (joint_factory) {
            if (joint_execution) {
                input.n_tokens=0;
                for (llama_seq_id slot=0;slot<2;++slot) for (uint32_t row=0;row<uint32_t(lengths[slot]);++row) {
                    const auto index=input.n_tokens++;
                    input.token[index]=llama_token(100+slot*20+row);input.pos[index]=llama_pos(row);
                    input.n_seq_id[index]=1;input.seq_id[index][0]=slot;input.logits[index]=row==uint32_t(lengths[slot]-1);
                }
            }
            llama_kv_stream_selection selected;
            auto projection=std::make_shared<llama_kv_stream_slot_projection>();
            if (!t.assert_true(llama_kv_stream_select({cp.n_ctx,2},{{0,0},{1,0}},selected)==llama_kv_stream_selection_status::success &&
                    group->source_table()->project(selected,group->source_handles(),0,*projection))) return;
            llama_context_memory_create_result result;
            auto * target=group->metadata_context();
            auto * execution_child=child.get();
            if (!t.assert_true("readonly assistant binds the measured target parent without a private cache or arena",
                    manager_mode ? group->activate(selected,group->source_handles(),input,llama_memory_work_phase::target_prefill,1,1,&result) :
                    target->initialize_group_execution(group->source_table(),projection,input,llama_memory_work_phase::target_prefill,
                        1,1,nullptr,{},&result,execution_child,projection,{}))) return;
            if (manager_mode) {target=group->context();execution_child=group->auxiliary_context();}
            if (!t.assert_true(target && execution_child)) return;
            t.assert_true(target->group_parent() && target->group_parent()==execution_child->group_parent() &&
                !execution_child->get_cparams().shared_device_memory_bytes && !target->group_storage_pending() && !execution_child->group_storage_pending());
            if (joint_execution) {
                common_params_speculative parameters;parameters.types={COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
                parameters.draft.ctx_tgt=target;parameters.draft.ctx_dft=execution_child;
                parameters.draft.n_max=maximum;parameters.draft.n_min=0;parameters.draft.p_min=0;parameters.draft.backend_sampling=false;
                common_speculative_ptr spec(common_speculative_init(parameters,2));
                if (!t.assert_true(spec!=nullptr)) return;
                common_batch prompt(target);std::array<llama_tokens,2> histories,proposals;
                for (llama_seq_id slot=0;slot<2;++slot) for (uint32_t row=0;row<uint32_t(lengths[slot]);++row) {
                    const auto token=llama_token(100+slot*20+row);histories[slot].push_back(token);
                    prompt.add(token,llama_pos(row),slot,row==uint32_t(lengths[slot]-1));
                }
                oracle.compare();target_trace.phase="prefill";
                if (!t.assert_equal(0,llama_process(target,LLAMA_PROCESS_TYPE_DECODE,prompt.get()))) return;
                target_trace.check();target_trace.phase.clear();
                if (!t.assert_true(common_speculative_process(spec.get(),prompt))) return;
                for (llama_seq_id slot=0;slot<2;++slot) {
                    const auto * logits=llama_get_logits_ith(target,last[slot]);
                    if (!t.assert_true(logits!=nullptr)) return;
                    const auto next=llama_token(std::max_element(logits,logits+model->vocab.n_tokens())-logits);
                    common_speculative_begin(spec.get(),slot,histories[slot]);
                    common_speculative_get_draft_params(spec.get(),slot)={true,widths[slot],llama_pos(histories[slot].size()),next,
                        &histories[slot],&proposals[slot],nullptr,0.0f,123};
                }
                if (!t.assert_true("committed target sources admit readonly group MTP",
                        target->group_mtp_ready(maximum) && llama_kv_stream_mtp_prepare(target,maximum))) return;
                std::vector<llama_context_group_sequence_state> before,after;
                if (!t.assert_true(target->save_group_sequences(before))) return;
                {
                    joint_attention_probe joint(*execution_child,model->hparams.n_embd_head_k(model->hparams.n_layer()-1));
                    child_trace.phase="predict";common_speculative_draft(spec.get());llama_synchronize(execution_child);child_trace.check();
                    child_trace.phase.clear();
                    if (!t.assert_true("streamed assistant submits joint full-attention query rows",joint.joint>0)) return;
                }
                for (llama_seq_id slot=0;slot<2;++slot) {
                    std::vector<uint8_t> carry;
                    t.assert_true(proposals[slot]==native_proposals[slot] &&
                        common_speculative_get_state(spec.get(),slot,carry) && carry==native_carries[slot]);
                    t.assert_equal(llama_pos(lengths[slot]-1),llama_memory_seq_pos_max(target->get_memory(),slot));
                }
                if (!t.assert_true(target->save_group_sequences(after) && before.size()==after.size())) return;
                for (size_t i=0;i<before.size();++i) {
                    const auto & a=before[i];const auto & b=after[i];
                    if (!t.assert_true(a.model==b.model && a.sequence==b.sequence && a.request_epoch==b.request_epoch &&
                            a.cache_id==b.cache_id && a.cache_generation==b.cache_generation && a.tokens==b.tokens &&
                            a.bounded==b.bounded && a.recurrent==b.recurrent && a.full_columns==b.full_columns)) return;
                }
                if (!continue_target(*target,manager_mode ? spec.get() : nullptr,manager_mode ? &histories : nullptr,execution_child)) return;
                if (manager_mode) {
                    std::array<llama_token,2> anchors{};
                    for (llama_seq_id slot=0;slot<2;++slot) {
                        const auto * logits=llama_get_logits_ith(target,slot);
                        if (!t.assert_true(logits!=nullptr)) return;
                        anchors[slot]=llama_token(std::max_element(logits,logits+model->vocab.n_tokens())-logits);
                    }
                    const auto stats=common_speculative_get_stats(spec.get());
                    auto * parent=target->group_parent();
                    if (!t.assert_true(common_speculative_rebind(spec.get(),group->metadata_context(),group->auxiliary_metadata_context()))) return;
                    if (!t.assert_true("readonly assistant retires without saving or releasing target-owned state",group->park())) return;
                    t.assert_true(!group->context() && !group->auxiliary_context());
                    std::vector<llama_kv_stream_selection_slot> demands;
                    for (uint32_t slot=0;slot<2;++slot) demands.push_back({slot,histories[slot].size()+1});
                    if (!t.assert_true(llama_kv_stream_select({cp.n_ctx,2},demands,selected)==llama_kv_stream_selection_status::success)) return;
                    input.n_tokens=2;
                    for (llama_seq_id slot=0;slot<2;++slot) {
                        input.token[slot]=anchors[slot];input.pos[slot]=llama_pos(histories[slot].size());
                        input.n_seq_id[slot]=1;input.seq_id[slot][0]=slot;input.logits[slot]=true;
                    }
                    if (!t.assert_true(group->activate(selected,group->source_handles(),input,llama_memory_work_phase::target_verify,1,2))) return;
                    target=group->context();execution_child=group->auxiliary_context();
                    if (!t.assert_true(target && execution_child && target->group_parent()==parent && execution_child->group_parent()==parent &&
                            common_speculative_rebind(spec.get(),target,execution_child))) return;
                    const auto rebound=common_speculative_get_stats(spec.get());
                    if (!t.assert_equal(stats.size(),rebound.size())) return;
                    for (size_t i=0;i<stats.size();++i) {
                        const auto & a=stats[i];const auto & b=rebound[i];
                        t.assert_true(a.type==b.type && a.n_call_begin==b.n_call_begin && a.n_call_draft==b.n_call_draft &&
                            a.n_call_accept==b.n_call_accept && a.n_gen_drafts==b.n_gen_drafts && a.n_acc_drafts==b.n_acc_drafts &&
                            a.n_gen_tokens==b.n_gen_tokens && a.n_acc_tokens==b.n_acc_tokens && a.n_predict==b.n_predict);
                    }
                    if (!resumed_predict(*target,*execution_child,spec.get(),histories,proposals,anchors,false)) return;
                }
                oracle.finish();
                if (ring) {
                    t.out<<"Gemma actual assistant ring spans="<<transport->spans[2]<<std::endl;
                    if (!t.assert_true("readonly assistant requires real nonresident source transport",
                            transport->spans[2]>0 && ggml_backend_buffer_get_size(target->group_parent())==ring_budget)) return;
                }
                t.out<<"streamed Gemma joint assistant exact_records="<<oracle.records()<<" bytes="<<oracle.byte_count()<<std::endl;
            }
        }
    }
}

static void test_group_auxiliary_metadata(testing & t,llama_model * model) {
    if (!t.assert_true(model && model->arch==LLM_ARCH_QWEN35 && model->hparams.n_layer_nextn==1)) return;
    common_params declared;
    declared.n_ctx=512;declared.n_batch=declared.n_ubatch=256;declared.n_parallel=2;declared.kv_unified=true;
    declared.shared_device_memory_bytes=size_t(1024)*1024*1024;
    declared.speculative.types={COMMON_SPECULATIVE_TYPE_DRAFT_MTP};declared.speculative.draft.n_max=3;
    declared.cache_type_k=GGML_TYPE_Q8_0;declared.cache_type_v=GGML_TYPE_Q4_0;
    const auto resolved=common_context_params_to_llama(declared,model);
    if (!t.assert_true("model-only Qwen startup derives the actual owned MTP layer and rollback depth",
            resolved.kv_stream_auxiliary_layers==1 && resolved.n_rs_seq==3 && !resolved.kv_stream_verify_max &&
            resolved.n_seq_max==2 && resolved.type_k==GGML_TYPE_Q8_0 && resolved.type_v==GGML_TYPE_Q4_0 &&
            !declared.kv_stream_auxiliary_layers)) return;
    auto cp=llama_context_default_params();
    cp.n_ctx=512;cp.n_batch=cp.n_ubatch=256;cp.n_seq_max=2;cp.kv_unified=true;cp.swa_full=false;
    cp.n_outputs_max=8;cp.n_outputs_max_per_seq=4;
    cp.type_k=GGML_TYPE_Q8_0;cp.type_v=GGML_TYPE_Q4_0;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.n_threads=cp.n_threads_batch=4;
    uint32_t full_layers=0,auxiliary_layers=0;
    {
        llama_context target(*model,cp);
        auto draft=cp;draft.ctx_type=LLAMA_CONTEXT_TYPE_MTP;draft.ctx_other=&target;
        llama_context child(*model,draft);
        const auto target_resources=target.get_kv_resources(),child_resources=child.get_kv_resources();
        if (!t.assert_true(target_resources.full && target_resources.recurrent && child_resources.full && !child_resources.recurrent)) return;
        full_layers=uint32_t(target_resources.full->get_layer_ids().size());auxiliary_layers=uint32_t(child_resources.full->get_layer_ids().size());
        if (!t.assert_true(full_layers>1 && auxiliary_layers==1 && child_resources.full->get_layer_ids()[0]==model->hparams.n_layer())) return;
    }
    t.out<<"native auxiliary geometry layers="<<full_layers<<'+'<<auxiliary_layers<<" selected=2\n";
    cp.shared_device_memory_bytes=size_t(1024)*1024*1024;cp.n_rs_seq=3;cp.kv_stream_auxiliary_layers=1;
    auto group=llama_context_group::create(*model,cp);
    if (!t.assert_true("one group reserves the actual target and embedded-head histories before payload allocation",group!=nullptr)) return;
    std::vector<uint64_t> ids;
    for (const auto & handle : group->source_handles()) for (uint32_t role=0;role<2;++role) {
        std::shared_ptr<llama_kv_stream_logical_cache> cache;std::shared_ptr<llama_kv_cells> cells;
        if (!t.assert_true(group->source_table()->get(handle,role,cache) && group->source_table()->get_cells(handle,role,cells))) return;
        t.assert_true(cache->ready() && !cache->tokens() && !cache->host()->committed_bytes() && !cells->get_used());
        t.assert_equal(role ? auxiliary_layers : full_layers,cache->host()->config().layers);
        t.assert_true(std::find(ids.begin(),ids.end(),cache->identity().id)==ids.end());ids.push_back(cache->identity().id);
    }
    auto * metadata=group->metadata_context();
    t.assert_true(metadata && metadata->group_storage_pending() && !metadata->group_parent());
    auto * auxiliary_metadata=group->auxiliary_metadata_context();
    if (!t.assert_true("the group owns its auxiliary graph declaration before allocating the shared parent",auxiliary_metadata!=nullptr)) return;
    t.assert_true(auxiliary_metadata->group_storage_pending() && !auxiliary_metadata->group_parent() &&
        auxiliary_metadata->group_target_context()==metadata && llama_get_ctx_other(auxiliary_metadata)==nullptr &&
        !auxiliary_metadata->get_cparams().shared_device_memory_bytes);
    t.assert_true(auxiliary_metadata->get_cparams().embeddings_nextn && auxiliary_metadata->get_cparams().embeddings_nextn_masked);
    for (const auto & entry : auxiliary_metadata->get_memory()->memory_breakdown()) t.assert_equal(size_t(0),entry.second);
    {
        auto input=llama_batch_init(2,0,1);
        struct release_batch {llama_batch & batch;~release_batch() {llama_batch_free(batch);}} release{input};
        input.n_tokens=2;
        std::vector<llama_memory_phase_cache_identity> target_sources,auxiliary_sources;
        for (llama_seq_id slot=0;slot<2;++slot) {
            input.token[slot]=10+slot;input.pos[slot]=0;input.n_seq_id[slot]=1;input.seq_id[slot][0]=slot;input.logits[slot]=true;
            for (uint32_t role=0;role<2;++role) {
                std::shared_ptr<llama_kv_stream_logical_cache> cache;
                if (!t.assert_true(group->source_table()->get(group->source_handles()[slot],role,cache))) return;
                (role ? auxiliary_sources : target_sources).push_back({cache->identity().id,cache->content()->generation()});
            }
        }
        llama_context_group_plan quote;
        const auto result=metadata->quote_target_group(input,target_sources,{9001,1},llama_memory_work_phase::target_prefill,
            100,cp.shared_device_memory_bytes,quote,auxiliary_metadata,auxiliary_sources);
        if (!t.assert_true("the shared parent covers target and auxiliary native workload catalogs before allocation",
                result.status==llama_context_memory_create_status::success)) return;
        bool prefill=false,verify=false,catchup=false,predict=false,all_verify_outputs=false;
        for (const auto & workload : quote.catalog) for (const auto & member : workload.measurement.workload.members) {
            prefill=prefill || member.phase==llama_memory_work_phase::target_prefill;
            verify=verify || member.phase==llama_memory_work_phase::target_verify;
            catchup=catchup || member.phase==llama_memory_work_phase::mtp_catchup;
            predict=predict || member.phase==llama_memory_work_phase::mtp_draft;
            all_verify_outputs=all_verify_outputs || (member.phase==llama_memory_work_phase::target_verify &&
                member.input_rows==4 && member.output_rows==4);
        }
        t.assert_true(prefill && verify && catchup && predict && all_verify_outputs);
        t.assert_equal(size_t(93),quote.catalog.size());
        t.assert_equal(size_t(4),quote.caches.size());
        const auto minimum=quote.budget.parent_minimum_bytes;
        t.assert_true(minimum && minimum<cp.shared_device_memory_bytes);
        llama_context_group_plan unchanged;unchanged.current_state_bytes=17;
        const auto short_result=metadata->quote_target_group(input,target_sources,{9001,1},llama_memory_work_phase::target_prefill,
            100,minimum-1,unchanged,auxiliary_metadata,auxiliary_sources);
        t.assert_true(short_result.status==llama_context_memory_create_status::insufficient_budget &&
            short_result.required_bytes==minimum && short_result.additional_bytes==1 && unchanged.current_state_bytes==17);
        const auto exact_result=metadata->quote_target_group(input,target_sources,{9001,1},llama_memory_work_phase::target_prefill,
            100,minimum,unchanged,auxiliary_metadata,auxiliary_sources);
        t.assert_true(exact_result.status==llama_context_memory_create_status::success && unchanged.budget.parent_minimum_bytes==minimum);
        auto foreign=auxiliary_sources;foreign[0]=target_sources[0];
        unchanged.current_state_bytes=17;
        t.assert_true(metadata->quote_target_group(input,target_sources,{9001,1},llama_memory_work_phase::target_prefill,
            100,cp.shared_device_memory_bytes,unchanged,auxiliary_metadata,foreign).status==llama_context_memory_create_status::invalid_configuration);
        t.assert_equal(size_t(17),unchanged.current_state_bytes);
        t.assert_true(metadata->quote_target_group(input,target_sources,{9001,1},llama_memory_work_phase::target_prefill,
            UINT64_MAX,cp.shared_device_memory_bytes,unchanged,auxiliary_metadata,auxiliary_sources).status==llama_context_memory_create_status::invalid_configuration);
        t.assert_equal(size_t(17),unchanged.current_state_bytes);
        t.out<<"joint MTP workload catalog="<<quote.catalog.size()<<" exact_minimum="<<minimum<<'\n';
    }
    for (const auto & entry : metadata->get_memory()->memory_breakdown()) t.assert_equal(size_t(0),entry.second);
    const auto previous=group->source_handles()[0];
    std::shared_ptr<llama_kv_stream_logical_cache> auxiliary;std::shared_ptr<llama_kv_cells> cells;
    if (!t.assert_true(group->source_table()->get(previous,1,auxiliary) && group->source_table()->get_cells(previous,1,cells))) return;
    const auto & layout=auxiliary->host()->layout();
    std::vector<uint8_t> k(layout.k_token_bytes,0x11),v(layout.v_token_bytes,0x22);
    llama_kv_stream_write write;
    if (!t.assert_true(auxiliary->begin(1) && auxiliary->content()->prepare({
            {0,ggml_kv_stream_operand::k,0,k.data(),k.size()},{0,ggml_kv_stream_operand::v,0,v.data(),v.size()}},write) &&
            auxiliary->publish_host(write) && auxiliary->finish())) return;
    cells->pos_set(0,0);cells->seq_add(0,0);
    if (!t.assert_true(group->reset_request(0))) return;
    if (!t.assert_true("request reset clears every owned cache before advancing its namespace",!auxiliary->tokens() && !cells->get_used())) return;
    const auto next=group->source_handles()[0];
    t.assert_true(next.generation==previous.generation+1 && !group->source_table()->get(previous,1,auxiliary));
    t.assert_true(group->source_table()->get(next,1,auxiliary) && auxiliary->ready() && !auxiliary->tokens());
    auto child_params=cp;child_params.shared_device_memory_bytes=0;child_params.n_rs_seq=0;
    child_params.kv_stream_auxiliary_layers=0;child_params.ctx_type=LLAMA_CONTEXT_TYPE_MTP;child_params.ctx_other=metadata;
    for (const auto & selected : {std::vector<llama_seq_id>{0,1},std::vector<llama_seq_id>{1}}) {
        std::unique_ptr<llama_context> child;
        try {child=std::make_unique<llama_context>(*model,child_params,selected);} catch (const std::runtime_error & error) {t.out<<"child metadata: "<<error.what()<<'\n';}
        if (!t.assert_true("the trained auxiliary head can be declared without an independent arena or KV payload",child!=nullptr)) return;
        const auto resources=child->get_kv_resources();
        t.assert_true(child->group_storage_pending() && !child->group_parent() && !child->get_cparams().shared_device_memory_bytes);
        if (!t.assert_true(resources.full && resources.full->full_storage_pending() && !resources.recurrent)) return;
        t.assert_equal(size_t(1),resources.full->get_layer_ids().size());
        t.assert_equal(model->hparams.n_layer(),resources.full->get_layer_ids()[0]);
        for (const auto & entry : child->get_memory()->memory_breakdown()) t.assert_equal(size_t(0),entry.second);
        llama_batch empty{};t.assert_equal(-1,child->decode(empty));
        std::vector<llama_memory_phase_cache_identity> sources;
        for (auto sequence : selected) {
            std::shared_ptr<llama_kv_stream_logical_cache> cache;
            if (!t.assert_true(group->source_table()->get(group->source_handles()[sequence],1,cache))) return;
            sources.push_back({cache->identity().id,cache->content()->generation()});
        }
        auto input=llama_batch_init(256,0,1);
        struct release_input {llama_batch & batch;~release_input() {llama_batch_free(batch);}} release{input};
        const auto measure=[&](std::array<uint32_t,2> widths,llama_memory_work_phase phase,uint64_t id) {
            input.n_tokens=0;
            const bool predict=phase==llama_memory_work_phase::mtp_draft;
            for (auto sequence : selected) for (uint32_t row=0;row<widths[sequence];++row) {
                const auto token=input.n_tokens++;
                input.token[token]=10+sequence;input.pos[token]=llama_pos(row);input.n_seq_id[token]=1;
                input.seq_id[token][0]=sequence;input.logits[token]=predict;
            }
            llama_context_group_workload_quote quote;
            if (!t.assert_true("auxiliary phases measure their own native graphs before the common arena is allocated",
                    child->measure_target_workload(input,sources,phase,id,quote))) return false;
            if (!t.assert_equal("dense auxiliary measurement follows the native unified-cache microbatch",size_t(1),
                    quote.measurement.workload.native_batches.size())) return false;
            uint32_t rows=0,outputs=0;
            for (const auto & member : quote.measurement.workload.members) {
                t.assert_equal(int(phase),int(member.phase));
                t.assert_true(member.writes_cache && !member.rollback_depth);
                t.assert_equal(predict ? member.input_rows : 0u,member.output_rows);
                rows+=member.input_rows;outputs+=member.output_rows;
            }
            t.assert_equal(uint32_t(input.n_tokens),rows);
            t.assert_equal(predict ? rows : 0u,outputs);
            t.assert_true(quote.writer_bytes && quote.attention_bytes && quote.pool_minimum_bytes && !quote.publication_bytes);
            t.assert_true(!quote.measurement.graph_bytes.empty() && std::all_of(quote.measurement.graph_bytes.begin(),
                quote.measurement.graph_bytes.end(),[](size_t bytes) {return bytes!=0;}));
            for (const auto & entry : child->get_memory()->memory_breakdown()) t.assert_equal(size_t(0),entry.second);
            return true;
        };
        uint64_t profile=101;
        for (auto widths : {std::array<uint32_t,2>{1,1},{4,4},{1,4},{128,128}})
            if (!measure(widths,llama_memory_work_phase::mtp_catchup,profile++)) return;
        if (!measure({1,1},llama_memory_work_phase::mtp_draft,profile++)) return;
        input.n_tokens=1;input.token[0]=10;input.pos[0]=0;input.n_seq_id[0]=1;input.seq_id[0][0]=selected.front();input.logits[0]=1;
        llama_context_group_workload_quote invalid;invalid.writer_bytes=17;
        t.assert_true(!child->measure_target_workload(input,sources,llama_memory_work_phase::target_verify,profile,invalid));
        t.assert_equal(size_t(17),invalid.writer_bytes);
        t.assert_true(!child->measure_target_workload(input,sources,llama_memory_work_phase::mtp_catchup,profile,invalid));
        input.logits[0]=false;
        t.assert_true(!child->measure_target_workload(input,sources,llama_memory_work_phase::mtp_draft,profile,invalid));
        input.n_tokens=2;input.token[1]=11;input.pos[1]=1;input.n_seq_id[1]=1;
        input.seq_id[1][0]=selected.front();input.logits[0]=input.logits[1]=true;
        t.assert_true(!child->measure_target_workload(input,sources,llama_memory_work_phase::mtp_draft,profile,invalid));
        input.n_tokens=1;input.seq_id[0][0]=2;
        t.assert_true(!child->measure_target_workload(input,sources,llama_memory_work_phase::mtp_draft,profile,invalid));
        t.assert_equal(size_t(17),invalid.writer_bytes);
        t.assert_true(child->group_storage_pending() && !child->group_parent());
    }
    for (int invalid=0;invalid<4;++invalid) {
        auto bad=child_params;
        if (invalid==0) bad.type_v=GGML_TYPE_Q8_0;
        if (invalid==1) bad.shared_device_memory_bytes=cp.shared_device_memory_bytes;
        if (invalid==2) bad.n_rs_seq=1;
        if (invalid==3) bad.n_ctx=1024;
        bool rejected=false;
        try {llama_context invalid_child(*model,bad,std::vector<llama_seq_id>{0,1});} catch (const std::runtime_error &) {rejected=true;}
        t.assert_true(rejected);
    }
    auto batch=llama_batch_init(2,0,1);
    struct release_batch {llama_batch & batch;~release_batch() {llama_batch_free(batch);}} release{batch};
    batch.n_tokens=2;
    for (int32_t slot=0;slot<2;++slot) {batch.token[slot]=10+slot;batch.pos[slot]=0;batch.n_seq_id[slot]=1;batch.seq_id[slot][0]=slot;batch.logits[slot]=1;}
    llama_kv_stream_selection selection;
    if (!t.assert_true(llama_kv_stream_select({512,2},{{0,0},{1,0}},selection)==llama_kv_stream_selection_status::success)) return;
    auto projection=std::make_shared<llama_kv_stream_slot_projection>();
    if (!t.assert_true(group->source_table()->project(selection,group->source_handles(),0,*projection))) return;
    t.assert_true("a child declaration cannot enable execution before its shared workloads are composed",
        !metadata->initialize_group_execution(group->source_table(),projection,batch,llama_memory_work_phase::target_prefill,1,1) &&
        metadata->group_storage_pending() && !metadata->group_parent());
}

static void test_native_two_slot_dflash3(testing & t,llama_model * model,const char * path) {
    if (!t.assert_true(model && model->arch==LLM_ARCH_QWEN35 && path)) return;
    auto mp=llama_model_default_params();mp.n_gpu_layers=999;
    model_ptr draft(llama_model_load_from_file(path,mp),llama_model_free);
    if (!t.assert_true(draft && draft->arch==LLM_ARCH_DFLASH)) return;
    auto cp=llama_context_default_params();
    cp.n_ctx=512;cp.n_batch=cp.n_ubatch=256;cp.n_seq_max=2;cp.kv_unified=true;cp.swa_full=false;
    cp.n_rs_seq=3;cp.n_outputs_max=16;cp.n_outputs_max_per_seq=8;cp.n_threads=cp.n_threads_batch=4;
    cp.type_k=GGML_TYPE_Q8_0;cp.type_v=GGML_TYPE_Q4_0;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
    t.out<<"native mode=dflash3 slots=2 target_rollback=3 draft_max=3 K4V=off C512 Q8Q4\n"<<std::flush;
    llama_context target(*model,cp);
    auto child_params=cp;child_params.n_rs_seq=0;child_params.ctx_other=&target;
    llama_context child(*draft,child_params);
    common_params_speculative params;params.types={COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH};
    params.draft.ctx_tgt=&target;params.draft.ctx_dft=&child;
    params.draft.n_max=3;params.draft.n_min=0;params.draft.p_min=0;
    common_speculative_ptr driver(common_speculative_init(params,2));
    if (!t.assert_true(driver!=nullptr)) return;
    std::array<llama_tokens,2> prompts,proposals;
    std::array<llama_token,2> seeds{};
    common_batch prefill(&target);
    for (llama_seq_id slot=0;slot<2;++slot) for (int32_t row=0;row<(slot ? 7 : 5);++row) {
        prompts[slot].push_back(10+slot*20+row);prefill.add(prompts[slot].back(),row,slot,true);
    }
    if (!t.assert_equal(0,llama_process(&target,LLAMA_PROCESS_TYPE_DECODE,prefill.get())) ||
            !t.assert_true(common_speculative_process(driver.get(),prefill))) return;
    for (llama_seq_id slot=0;slot<2;++slot) {
        const auto row=slot ? 11 : 4;const auto * logits=target.get_logits_ith(row);
        if (!t.assert_true(logits && std::all_of(logits,logits+model->vocab.n_tokens(),[](float v) {return std::isfinite(v);}))) return;
        seeds[slot]=llama_token(std::max_element(logits,logits+model->vocab.n_tokens())-logits);
        common_speculative_begin(driver.get(),slot,prompts[slot]);
        auto & request=common_speculative_get_draft_params(driver.get(),slot);
        request.drafting=true;request.pos0=llama_pos(prompts[slot].size());request.id_last=seeds[slot];
        request.prompt=&prompts[slot];request.result=&proposals[slot];
    }
    common_speculative_draft(driver.get());child.synchronize();
    for (const auto & proposal : proposals) {
        if (!t.assert_equal(size_t(3),proposal.size())) return;
        for (auto token : proposal) t.assert_true(token>=0 && uint32_t(token)<model->vocab.n_tokens());
    }
    const auto * lattice=child.get_embeddings_nextn();
    t.assert_true(lattice && std::all_of(lattice,lattice+size_t(8)*draft->hparams.n_embd_out(),[](float v) {return std::isfinite(v);}));
    common_batch verify(&target);
    for (llama_seq_id slot=0;slot<2;++slot) {
        const auto first=llama_pos(prompts[slot].size());
        if (!t.assert_true("retire the native provisional draft rows before teacher-forced injection",
                llama_memory_seq_rm(child.get_memory(),slot,first,-1))) return;
        verify.add(seeds[slot],first,slot,true);
        for (size_t i=0;i<proposals[slot].size();++i) verify.add(proposals[slot][i],first+1+llama_pos(i),slot,true);
    }
    if (!t.assert_equal(0,llama_process(&target,LLAMA_PROCESS_TYPE_DECODE,verify.get())) ||
            !t.assert_true(common_speculative_process(driver.get(),verify))) return;
    t.out<<"native DFlash3 two-slot prefill, joint draft and four-row target verification completed\n";
}

static void test_group_dflash_rollback_metadata(testing & t,llama_model * model) {
    if (!t.assert_true(model && model->arch==LLM_ARCH_QWEN35)) return;
    auto cp=llama_context_default_params();
    cp.n_ctx=512;cp.n_batch=cp.n_ubatch=256;cp.n_seq_max=2;cp.kv_unified=true;
    cp.n_outputs_max=16;cp.n_outputs_max_per_seq=8;cp.n_rs_seq=3;
    cp.type_k=GGML_TYPE_Q8_0;cp.type_v=GGML_TYPE_Q4_0;
    cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.shared_device_memory_bytes=size_t(1024)*1024*1024;
    if (!t.assert_true("existing grouped K3 target metadata is valid",llama_context_group::create(*model,cp)!=nullptr)) return;
    for (const auto depth : {8u,15u}) {
        cp.n_rs_seq=depth;
        t.assert_true("aggregate rows cannot enable K8 or K15 rollback",llama_context_group::create(*model,cp)==nullptr);
    }
    cp.n_rs_seq=7;cp.kv_stream_auxiliary_layers=1;
    t.assert_true("embedded MTP still rejects seven rollback slots",llama_context_group::create(*model,cp)==nullptr);
    cp.kv_stream_auxiliary_layers=0;
    auto group=llama_context_group::create(*model,cp);
    if (!t.assert_true("DFlash K7 declares per-sequence target rollback without allocating the arena",group!=nullptr)) return;
    auto * metadata=group->metadata_context();
    t.assert_true(metadata && metadata->group_storage_pending() && !metadata->group_parent());
    t.assert_equal(uint32_t(7),llama_n_rs_seq(metadata));
    t.assert_true(group->auxiliary_metadata_context()==nullptr);
    for (const auto & entry : metadata->get_memory()->memory_breakdown()) t.assert_equal(size_t(0),entry.second);
    for (const auto & handle : group->source_handles()) {
        std::shared_ptr<llama_kv_stream_logical_cache> cache;
        if (!t.assert_true(group->source_table()->get(handle,0,cache))) return;
        t.assert_true(!cache->tokens() && !cache->host()->committed_bytes());
    }
}

static std::array<uint64_t,16> native_dflash_stage_profile(const llama_model & target,const llama_model & draft) {
    return {3,512,256,2,7,5,7,uint64_t(GGML_TYPE_Q8_0),uint64_t(GGML_TYPE_Q4_0),target.hparams.n_embd,
        target.vocab.n_tokens(),draft.hparams.n_embd_inp_enc(),draft.hparams.n_embd_out(),draft.hparams.n_layer(),
        draft.target_layer_ids.size(),draft.hparams.dflash_block_size};
}

static std::array<uint64_t,16> native_dflash_draft_profile(const llama_model & target,const llama_model & draft) {
    auto profile=native_dflash_stage_profile(target,draft);profile[0]=4;return profile;
}

static std::array<uint64_t,15> native_dflash_borrowed_profile(const llama_model & target) {
    std::array<uint64_t,15> result{};
    const std::array<const ggml_tensor *,3> tensors{target.tok_embd,target.output,target.output_s};
    for (size_t i=0;i<tensors.size();++i) {
        const auto * tensor=tensors[i];
        if (!tensor) {if (i<2) throw std::runtime_error("missing borrowed target weight in staged native reference");continue;}
        if (!tensor->buffer) throw std::runtime_error("unbound borrowed target weight in staged native reference");
        result[i*5]=1;result[i*5+1]=uint64_t(tensor->type);result[i*5+2]=uint64_t(tensor->ne[0]);
        result[i*5+3]=uint64_t(tensor->ne[1]);result[i*5+4]=uint64_t(ggml_backend_buffer_is_host(tensor->buffer));
    }
    return result;
}

static bool staged_dflash_attention_shape(ggml_tensor * tensor,bool ask,void *) {
    const char * path=std::getenv("LLAMA_TEST_DFLASH_SHAPE_DUMP");
    if (!path || !ask || !tensor || tensor->op!=GGML_OP_FLASH_ATTN_EXT || !tensor->src[0] || tensor->src[0]->ne[0]!=128) return false;
    std::unique_ptr<FILE,kv_stream_exact_file_free> file(std::fopen(path,"a"));
    if (!file) throw std::runtime_error("cannot preserve native attention shape diagnostic");
    std::fprintf(file.get(),"%s",tensor->name);
    for (size_t source=0;source<4;++source) {
        const auto * input=tensor->src[source];
        if (!input) {std::fprintf(file.get()," | null");continue;}
        std::fprintf(file.get()," | %d",int(input->type));
        for (auto size : input->ne) std::fprintf(file.get()," %lld",static_cast<long long>(size));
        for (auto stride : input->nb) std::fprintf(file.get()," %zu",stride);
    }
    std::fprintf(file.get()," | params");
    for (const auto word : tensor->op_params) std::fprintf(file.get()," %d",word);
    std::fprintf(file.get(),"\n");
    return false;
}

static void test_native_dflash_target_verify(testing & t,llama_model * model,const char * draft_path,
        const char * prefix_path,const char * proposals_path,const char * output_path) {
    if (!t.assert_true(model && model->arch==LLM_ARCH_QWEN35 && draft_path && prefix_path && proposals_path && output_path)) return;
    const llama_model_tensor_buft_override weights[]={{".*",ggml_backend_cpu_buffer_type()},{nullptr,nullptr}};
    auto mp=llama_model_default_params();mp.n_gpu_layers=999;mp.tensor_buft_overrides=weights;
    model_ptr draft(llama_model_load_from_file(draft_path,mp),llama_model_free);
    if (!t.assert_true(draft && draft->arch==LLM_ARCH_DFLASH)) return;
    std::unique_ptr<FILE,kv_stream_exact_file_free> proposal_file(std::fopen(proposals_path,"rb"));
    std::unique_ptr<FILE,kv_stream_exact_file_free> prefix_file(std::fopen(prefix_path,"rb"));
    if (!t.assert_true(proposal_file && prefix_file)) return;
    auto proposal_profile=native_dflash_draft_profile(*model,*draft);proposal_profile[0]=10;
    kv_stream_bit_oracle proposal_input;proposal_input.load_control(proposal_file.get(),proposal_profile);proposal_input.compare();
    std::array<llama_token,2> seeds{};std::array<std::array<llama_token,7>,2> proposals{};
    std::array<uint8_t,32> target_parent{},draft_parent{};
    proposal_input.input("seeds",seeds.data(),sizeof(seeds));
    for (size_t slot=0;slot<2;++slot) proposal_input.input("proposals/"+std::to_string(slot),proposals[slot].data(),sizeof(proposals[slot]));
    proposal_input.input("target_parent_sha256",target_parent.data(),target_parent.size());
    proposal_input.input("draft_parent_sha256",draft_parent.data(),draft_parent.size());proposal_input.finish();
    for (size_t slot=0;slot<2;++slot) {
        if (!t.assert_true(seeds[slot]>=0 && uint32_t(seeds[slot])<model->vocab.n_tokens())) return;
        for (auto token : proposals[slot]) if (!t.assert_true(token>=0 && uint32_t(token)<model->vocab.n_tokens())) return;
    }
    auto cp=llama_context_default_params();
    cp.n_ctx=512;cp.n_batch=cp.n_ubatch=256;cp.n_seq_max=2;cp.kv_unified=true;cp.swa_full=false;
    cp.n_outputs_max=16;cp.n_outputs_max_per_seq=8;cp.n_rs_seq=7;
    cp.type_k=GGML_TYPE_Q8_0;cp.type_v=GGML_TYPE_Q4_0;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context target(*model,cp);
    auto dp=cp;dp.n_rs_seq=0;dp.ctx_other=&target;
    llama_context metadata(*draft,dp);
    for (auto layer : draft->target_layer_ids) target.set_embeddings_layer_inp(uint32_t(layer),true);
    auto batch=llama_batch_init(12,0,1);
    struct release_batch {llama_batch & batch;~release_batch() {llama_batch_free(batch);}} release{batch};
    std::array<int32_t,12> tokens{},positions{},sequences{};
    batch.n_tokens=0;
    for (llama_seq_id slot=0;slot<2;++slot) for (int32_t position=0;position<(slot ? 7 : 5);++position) {
        const auto row=batch.n_tokens++;tokens[row]=10+slot*20+position;positions[row]=position;sequences[row]=slot;
        batch.token[row]=tokens[row];batch.pos[row]=position;batch.n_seq_id[row]=1;batch.seq_id[row][0]=slot;batch.logits[row]=true;
    }
    if (!t.assert_equal(0,target.decode(batch))) return;
    target.synchronize();
    kv_stream_bit_oracle prefix;prefix.load_control(prefix_file.get(),native_dflash_stage_profile(*model,*draft));prefix.compare();
    prefix.bytes("tokens",tokens.data(),sizeof(tokens));prefix.bytes("positions",positions.data(),sizeof(positions));
    prefix.bytes("sequences",sequences.data(),sizeof(sequences));prefix.bytes("seeds",seeds.data(),sizeof(seeds));
    const auto borrowed=native_dflash_borrowed_profile(*model);
    prefix.bytes("target_borrowed_weights",borrowed.data(),sizeof(borrowed));
    prefix.bytes("target_layers",draft->target_layer_ids.data(),draft->target_layer_ids.size()*sizeof(int32_t));
    for (auto layer : draft->target_layer_ids) {
        const auto * values=target.get_embeddings_layer_inp(uint32_t(layer));
        if (!t.assert_true(values!=nullptr)) return;
        prefix.bytes("features/"+std::to_string(layer),values,size_t(12)*model->hparams.n_embd*sizeof(float));
    }
    for (int32_t row=0;row<12;++row) {
        const auto * values=target.get_logits_ith(row);if (!t.assert_true(values!=nullptr)) return;
        prefix.bytes("logits/"+std::to_string(row),values,size_t(model->vocab.n_tokens())*sizeof(float));
        if (row==4 || row==11) if (!t.assert_equal(seeds[row==11],llama_token(std::max_element(values,values+model->vocab.n_tokens())-values))) return;
    }
    for (llama_seq_id slot=0;slot<2;++slot) {
        const uint64_t size=llama_state_seq_get_size(&target,slot);
        if (!t.assert_true(size>0 && size<=(uint64_t(2)<<30))) return;
        std::vector<uint8_t> state(size_t(size),0);
        if (!t.assert_equal(state.size(),llama_state_seq_get_data(&target,state.data(),state.size(),slot))) return;
        prefix.bytes("target_state_size/"+std::to_string(slot),&size,sizeof(size));
        prefix.bytes("target_state/"+std::to_string(slot),state.data(),state.size());
    }
    prefix.finish();
    common_batch verify(&target);
    std::array<int32_t,16> verify_tokens{},verify_positions{},verify_sequences{};
    for (llama_seq_id slot=0;slot<2;++slot) for (int32_t offset=0;offset<8;++offset) {
        const auto row=slot*8+offset;const auto token=offset ? proposals[slot][offset-1] : seeds[slot];
        const auto position=(slot ? 7 : 5)+offset;
        verify.add(token,position,slot,true);verify_tokens[row]=token;verify_positions[row]=position;verify_sequences[row]=slot;
    }
    if (!t.assert_equal(0,llama_process(&target,LLAMA_PROCESS_TYPE_DECODE,verify.get()))) return;
    target.synchronize();kv_stream_bit_oracle output;
    output.bytes("target_parent_sha256",target_parent.data(),target_parent.size());
    output.bytes("draft_parent_sha256",draft_parent.data(),draft_parent.size());
    output.bytes("tokens",verify_tokens.data(),sizeof(verify_tokens));output.bytes("positions",verify_positions.data(),sizeof(verify_positions));
    output.bytes("sequences",verify_sequences.data(),sizeof(verify_sequences));
    output.bytes("target_layers",draft->target_layer_ids.data(),draft->target_layer_ids.size()*sizeof(int32_t));
    for (auto layer : draft->target_layer_ids) {
        const auto * values=target.get_embeddings_layer_inp(uint32_t(layer));
        if (!t.assert_true(values && std::all_of(values,values+size_t(16)*model->hparams.n_embd,[](float value) {return std::isfinite(value);}))) return;
        output.bytes("features/"+std::to_string(layer),values,size_t(16)*model->hparams.n_embd*sizeof(float));
    }
    for (int32_t row=0;row<16;++row) {
        const auto * values=target.get_logits_ith(row);
        if (!t.assert_true(values && std::all_of(values,values+model->vocab.n_tokens(),[](float value) {return std::isfinite(value);}))) return;
        output.bytes("logits/"+std::to_string(row),values,size_t(model->vocab.n_tokens())*sizeof(float));
    }
    for (llama_seq_id slot=0;slot<2;++slot) {
        const uint64_t size=llama_state_seq_get_size(&target,slot);
        if (!t.assert_true(size>0 && size<=(uint64_t(2)<<30))) return;
        std::vector<uint8_t> state(size_t(size),0);
        if (!t.assert_equal(state.size(),llama_state_seq_get_data(&target,state.data(),state.size(),slot))) return;
        output.bytes("target_state_size/"+std::to_string(slot),&size,sizeof(size));
        output.bytes("target_state/"+std::to_string(slot),state.data(),state.size());
    }
    std::unique_ptr<FILE,kv_stream_exact_file_free> destination(std::fopen(output_path,"wb"));
    if (!t.assert_true(destination!=nullptr)) return;
    auto output_profile=native_dflash_stage_profile(*model,*draft);output_profile[0]=11;
    output.save_control(destination.get(),output_profile);
    t.out<<"native target verification records="<<output.records()<<" bytes="<<output.byte_count()<<" prefix_records="<<prefix.records()<<" verify_rows=8,8\n";
}

struct dflash_verified_rows {
    std::array<int32_t,16> tokens{},positions{},sequences{};
    std::map<int32_t,std::vector<float>> features;
    std::array<std::array<std::vector<float>,8>,2> logits;
};

static bool load_dflash_verified_rows(testing & t,const llama_model & model,const llama_model & draft,
        const char * path,dflash_verified_rows & rows) {
    std::unique_ptr<FILE,kv_stream_exact_file_free> file(std::fopen(path,"rb"));if (!t.assert_true(file!=nullptr)) return false;
    auto profile=native_dflash_stage_profile(model,draft);profile[0]=11;
    kv_stream_bit_oracle input;input.load_control(file.get(),profile);input.compare();
    std::array<uint8_t,32> parent{};
    input.input("target_parent_sha256",parent.data(),parent.size());input.input("draft_parent_sha256",parent.data(),parent.size());
    input.input("tokens",rows.tokens.data(),sizeof(rows.tokens));input.input("positions",rows.positions.data(),sizeof(rows.positions));
    input.input("sequences",rows.sequences.data(),sizeof(rows.sequences));
    std::vector<int32_t> layers(draft.target_layer_ids.size());input.input("target_layers",layers.data(),layers.size()*sizeof(int32_t));
    if (!t.assert_true(layers==draft.target_layer_ids)) return false;
    for (auto layer : layers) {
        auto & values=rows.features[layer];values.resize(size_t(16)*model.hparams.n_embd);
        input.input("features/"+std::to_string(layer),values.data(),values.size()*sizeof(float));
        if (!t.assert_true(std::all_of(values.begin(),values.end(),[](float x) {return std::isfinite(x);}))) return false;
    }
    for (size_t row=0;row<16;++row) {
        auto & values=rows.logits[row/8][row%8];values.resize(model.vocab.n_tokens());
        input.input("logits/"+std::to_string(row),values.data(),values.size()*sizeof(float));
        if (!t.assert_true(std::all_of(values.begin(),values.end(),[](float x) {return std::isfinite(x);}))) return false;
    }
    for (llama_seq_id slot=0;slot<2;++slot) {
        uint64_t size=0;input.input("target_state_size/"+std::to_string(slot),&size,sizeof(size));
        if (!t.assert_true(size && size<=(uint64_t(2)<<30))) return false;
        std::vector<uint8_t> state(size_t(size),0);input.input("target_state/"+std::to_string(slot),state.data(),state.size());
    }
    input.finish();return true;
}

static bool test_dflash_after_teacher(testing & t,llama_model & model,llama_model & draft,llama_context * target,
        llama_context & child,common_speculative * driver,std::array<llama_tokens,2> & prompts,
        std::array<llama_tokens,2> & proposals,std::array<llama_token,2> seeds,const char * verified_path,
        const char * output_path,bool streamed) {
    dflash_verified_rows verified;if (!load_dflash_verified_rows(t,model,draft,verified_path,verified)) return false;
    std::array<uint32_t,2> accepted{1,6};
    if (const auto * value=std::getenv("LLAMA_TEST_DFLASH_ACCEPTED")) {
        unsigned first=0,second=0;char trailing=0;
        if (!t.assert_true(std::sscanf(value,"%u,%u%c",&first,&second,&trailing)==2 && first<=7 && second<=7)) return false;
        accepted={first,second};
    }
    for (llama_seq_id slot=0;slot<2;++slot) {
        if (!t.assert_equal(size_t(7),proposals[slot].size())) return false;
        for (size_t row=0;row<8;++row) {
            const auto index=size_t(slot)*8+row;
            if (!t.assert_true(verified.tokens[index]==(row ? proposals[slot][row-1] : seeds[slot]) &&
                    verified.positions[index]==llama_pos(prompts[slot].size()+row) && verified.sequences[index]==slot)) return false;
        }
        if (!streamed) {
            if (!t.assert_true(llama_memory_seq_rm(child.get_memory(),slot,llama_pos(prompts[slot].size()),-1))) return false;
            std::vector<float> features(size_t(8)*draft.hparams.n_embd_inp_enc());common_batch injection(&child);
            for (size_t row=0;row<8;++row) {
                for (size_t layer=0;layer<draft.target_layer_ids.size();++layer)
                    std::memcpy(features.data()+row*draft.hparams.n_embd_inp_enc()+layer*model.hparams.n_embd,
                        verified.features.at(draft.target_layer_ids[layer]).data()+(size_t(slot)*8+row)*model.hparams.n_embd,
                        size_t(model.hparams.n_embd)*sizeof(float));
                const llama_pos p=verified.positions[size_t(slot)*8+row],position[4]={p,p,p,0};
                injection.add_embd({features.data()+row*draft.hparams.n_embd_inp_enc(),1,draft.hparams.n_embd_inp_enc()},position,slot,false);
            }
            if (!t.assert_equal(0,llama_process(&child,LLAMA_PROCESS_TYPE_DECODE,injection.get()))) return false;
        }
    }
    child.synchronize();auto profile=native_dflash_draft_profile(model,draft);profile[0]=12;
    kv_stream_bit_oracle output;
    if (streamed) {
        std::unique_ptr<FILE,kv_stream_exact_file_free> file(std::fopen(output_path,"rb"));if (!t.assert_true(file!=nullptr)) return false;
        output.load_control(file.get(),profile);output.compare();
    }
    output.bytes("accepted",accepted.data(),sizeof(accepted));
    for (llama_seq_id slot=0;slot<2;++slot)
        kv_stream_exact_kv(output,"teacher_KV/"+std::to_string(slot),child.get_kv_resources().bounded,0,prompts[slot].size()+8,slot);
    for (llama_seq_id slot=0;slot<2;++slot) {
        common_speculative_accept(driver,slot,uint16_t(accepted[slot]));
        const auto keep=llama_pos(prompts[slot].size()+1+accepted[slot]);
        if (streamed && !t.assert_true(llama_memory_seq_rm(target->get_memory(),slot,keep,-1))) return false;
        if (!t.assert_true(llama_memory_seq_rm(child.get_memory(),slot,keep,-1))) return false;
        const auto & logits=verified.logits[slot][accepted[slot]];
        const auto next=llama_token(std::max_element(logits.begin(),logits.end())-logits.begin());
        prompts[slot].push_back(seeds[slot]);prompts[slot].insert(prompts[slot].end(),proposals[slot].begin(),proposals[slot].begin()+accepted[slot]);
        proposals[slot].clear();seeds[slot]=next;
        auto & request=common_speculative_get_draft_params(driver,slot);request.drafting=true;
        request.pos0=keep;request.id_last=next;
    }
    common_speculative_draft(driver);child.synchronize();output.bytes("seeds",seeds.data(),sizeof(seeds));
    for (llama_seq_id slot=0;slot<2;++slot) {
        if (!t.assert_equal(size_t(7),proposals[slot].size())) return false;
        output.bytes("proposals/"+std::to_string(slot),proposals[slot].data(),proposals[slot].size()*sizeof(llama_token));
        kv_stream_exact_kv(output,"next_KV/"+std::to_string(slot),child.get_kv_resources().bounded,0,prompts[slot].size()+8,slot);
    }
    const auto * lattice=child.get_embeddings_nextn();if (!t.assert_true(lattice!=nullptr)) return false;
    for (size_t slot=0;slot<2;++slot) {
        const auto canonical=dflash_exact_lattice(lattice+slot*8*draft.hparams.n_embd_out(),8,
            draft.hparams.n_embd_out(),draft.hparams.dflash_selector_top_k,model.vocab.n_tokens());
        output.bytes("selector_lattice/"+std::to_string(slot),canonical.data(),canonical.size()*sizeof(uint32_t));
    }
    const auto stats=common_speculative_get_stats(driver);
    if (!t.assert_true(stats.size()==1 && stats[0].n_predict==2 && stats[0].n_gen_tokens==28 &&
            stats[0].n_acc_tokens==size_t(accepted[0]+accepted[1]))) return false;
    if (streamed) output.finish();
    else {
        std::unique_ptr<FILE,kv_stream_exact_file_free> file(std::fopen(output_path,"wb"));if (!t.assert_true(file!=nullptr)) return false;
        output.save_control(file.get(),profile);
    }
    t.out<<"DFlash teacher/rollback/next native-exact records="<<output.records()<<" bytes="<<output.byte_count()
        <<" accepted="<<accepted[0]<<','<<accepted[1]<<" forwards=2\n";
    return true;
}

static void test_native_dflash_draft_stage(testing & t,llama_model * model,const char * draft_path,const char * input_path,
        const char * output_path,bool streamed=false,bool negative=false,bool parking=false,bool suffix=false,bool rebind=false,bool k4v=false,bool repeat=false,bool partition=false) {
    if (!t.assert_true(model && model->arch==LLM_ARCH_QWEN35 && draft_path && input_path && output_path)) return;
    const auto * native_teacher=streamed ? nullptr : std::getenv("LLAMA_TEST_DFLASH_NATIVE_TEACHER_ORACLE");
    auto mp=llama_model_default_params();mp.n_gpu_layers=999;
    model_ptr draft(llama_model_load_from_file(draft_path,mp),llama_model_free);
    if (!t.assert_true(draft && draft->arch==LLM_ARCH_DFLASH)) return;
    kv_stream_bit_oracle input;
    std::unique_ptr<FILE,kv_stream_exact_file_free> source(std::fopen(input_path,"rb"));
    if (!t.assert_true(source!=nullptr)) return;
    input.load_control(source.get(),native_dflash_stage_profile(*model,*draft));input.compare();
    std::array<int32_t,12> tokens{},positions{},sequences{};std::array<llama_token,2> seeds{};
    input.input("tokens",tokens.data(),sizeof(tokens));input.input("positions",positions.data(),sizeof(positions));
    input.input("sequences",sequences.data(),sizeof(sequences));input.input("seeds",seeds.data(),sizeof(seeds));
    std::array<uint64_t,15> borrowed{};input.input("target_borrowed_weights",borrowed.data(),sizeof(borrowed));
    if (!t.assert_true("the staged draft retains the original embedding/output/scale types and backends",borrowed==native_dflash_borrowed_profile(*model))) return;
    std::vector<int32_t> layers(draft->target_layer_ids.size());
    input.input("target_layers",layers.data(),layers.size()*sizeof(int32_t));
    if (!t.assert_true(layers==draft->target_layer_ids)) return;
    std::map<int32_t,std::vector<float>> features;
    for (auto layer : layers) {
        auto & values=features[layer];values.resize(size_t(12)*model->hparams.n_embd);
        input.input("features/"+std::to_string(layer),values.data(),values.size()*sizeof(float));
        if (!t.assert_true(std::all_of(values.begin(),values.end(),[](float value) {return std::isfinite(value);}))) return;
    }
    std::vector<std::vector<float>> reference_logits;
    for (size_t row=0;row<12;++row) {
        const int32_t slot=row<5 ? 0 : 1;const int32_t position=int32_t(row)-(slot ? 5 : 0);
        if (!t.assert_true(sequences[row]==slot && positions[row]==position && tokens[row]>=0 && uint32_t(tokens[row])<model->vocab.n_tokens())) return;
        std::vector<float> logits(model->vocab.n_tokens());input.input("logits/"+std::to_string(row),logits.data(),logits.size()*sizeof(float));
        if (!t.assert_true(std::all_of(logits.begin(),logits.end(),[](float value) {return std::isfinite(value);}))) return;
        if (row==4 || row==11) t.assert_equal(llama_token(std::max_element(logits.begin(),logits.end())-logits.begin()),seeds[slot]);
        reference_logits.push_back(std::move(logits));
    }
    for (llama_seq_id slot=0;slot<2;++slot) {
        uint64_t size=0;input.input("target_state_size/"+std::to_string(slot),&size,sizeof(size));
        if (!t.assert_true(size>0 && size<=(uint64_t(2)<<30))) return;
        std::vector<uint8_t> checkpoint(size_t(size),0);
        input.input("target_state/"+std::to_string(slot),checkpoint.data(),checkpoint.size());
    }
    input.finish();
    auto cp=llama_context_default_params();
    cp.n_ctx=512;cp.n_batch=cp.n_ubatch=256;cp.n_seq_max=2;cp.kv_unified=true;cp.swa_full=false;
    cp.n_outputs_max=16;cp.n_outputs_max_per_seq=8;cp.n_threads=cp.n_threads_batch=4;
    cp.type_k=GGML_TYPE_Q8_0;cp.type_v=GGML_TYPE_Q4_0;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
    if (std::getenv("LLAMA_TEST_DFLASH_SHAPE_DUMP")) cp.cb_eval=staged_dflash_attention_shape;
    std::unique_ptr<llama_context> metadata,native_child;
    std::unique_ptr<llama_context_group> group;
    llama_context * target=nullptr,* child_pointer=nullptr;
    if (streamed) {
        cp.n_rs_seq=7;cp.shared_device_memory_bytes=size_t(1024)*1024*1024;
        group=llama_context_group::create(*model,cp);if (!t.assert_true(group!=nullptr)) return;
        auto child_params=cp;child_params.n_rs_seq=0;child_params.shared_device_memory_bytes=0;
        if (!t.assert_true(group->prepare_dflash(*draft,child_params,8))) return;
        auto batch=llama_batch_init(12,0,1);
        struct release_batch {llama_batch & batch;~release_batch() {llama_batch_free(batch);}} release{batch};
        batch.n_tokens=12;
        for (size_t row=0;row<12;++row) {
            batch.token[row]=tokens[row];batch.pos[row]=positions[row];batch.n_seq_id[row]=1;
            batch.seq_id[row][0]=sequences[row];batch.logits[row]=true;
        }
        llama_kv_stream_selection selected;
        if (!t.assert_true(llama_kv_stream_select({512,2},{{0,0},{1,0}},selected)==llama_kv_stream_selection_status::success &&
                group->activate(selected,group->source_handles(),batch,llama_memory_work_phase::target_prefill,1,1))) return;
        target=group->context();child_pointer=group->auxiliary_context();
        if (!t.assert_equal(0,target->decode(batch))) return;
        target->synchronize();
        for (const auto & item : features) {
            const auto * actual=target->get_embeddings_layer_inp(uint32_t(item.first));
            if (!t.assert_true(actual && !std::memcmp(actual,item.second.data(),item.second.size()*sizeof(float)))) return;
        }
        for (size_t row=0;row<12;++row) {
            const auto * logits=target->get_logits_ith(int32_t(row));
            if (!t.assert_true(logits && !std::memcmp(logits,reference_logits[row].data(),reference_logits[row].size()*sizeof(float)))) return;
        }
    } else {
        metadata=std::make_unique<llama_context>(*model,cp,llama_context_mode::measurement);
        const llama_batch empty{};
        if (!t.assert_true(!metadata->get_compute_memory() && metadata->decode(empty)==-1)) return;
        for (const auto & entry : metadata->get_memory()->memory_breakdown()) t.assert_equal(size_t(0),entry.second);
        auto child_params=cp;child_params.ctx_other=metadata.get();
        native_child=std::make_unique<llama_context>(*draft,child_params);
        target=metadata.get();child_pointer=native_child.get();
    }
    auto & child=*child_pointer;
    common_params_speculative params;params.types={COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH};
    if (k4v) {
        params.types.insert(params.types.begin(),COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V);
        params.ngram_map_k4v.size_n=12;params.ngram_map_k4v.size_m=7;params.ngram_map_k4v.min_hits=1;
    }
    params.draft.ctx_tgt=target;params.draft.ctx_dft=&child;
    params.draft.n_max=7;params.draft.n_min=0;params.draft.p_min=0;
    common_speculative_ptr driver(common_speculative_init(params,2));if (!t.assert_true(driver!=nullptr)) return;
    std::array<uint32_t,2> partition_limits{1,7};
    if (partition) if (const auto * widths=std::getenv("LLAMA_TEST_DFLASH_WIDTHS")) {
        unsigned first=0,second=0;char trailing=0;
        if (!t.assert_true(std::sscanf(widths,"%u,%u%c",&first,&second,&trailing)==2 &&
                first>=1 && first<=7 && second>=1 && second<=7)) return;
        partition_limits={first,second};
    }
    std::array<llama_tokens,2> prompts,proposals;
    for (llama_seq_id slot=0;slot<2;++slot) {
        const size_t first=slot ? 5 : 0,count=slot ? 7 : 5;
        std::vector<float> gathered(count*draft->hparams.n_embd_inp_enc());
        common_batch batch(&child);
        for (size_t row=0;row<count;++row) {
            for (size_t layer=0;layer<layers.size();++layer)
                std::memcpy(gathered.data()+row*draft->hparams.n_embd_inp_enc()+layer*model->hparams.n_embd,
                    features.at(layers[layer]).data()+(first+row)*model->hparams.n_embd,size_t(model->hparams.n_embd)*sizeof(float));
            const llama_pos position[4]={positions[first+row],positions[first+row],positions[first+row],0};
            batch.add_embd({gathered.data()+row*draft->hparams.n_embd_inp_enc(),1,draft->hparams.n_embd_inp_enc()},position,slot,false);
            prompts[slot].push_back(tokens[first+row]);
        }
        auto * wire=batch.get();
        if (negative && slot==0) {
            if (!t.assert_true(streamed && wire->n_pos_per_embd==4)) return;
            const auto original=wire->tokens[0];uint32_t original_bits=0;
            std::memcpy(&original_bits,wire->embd.data(),sizeof(original_bits));
            for (uint32_t defect=0;defect<6;++defect) {
                wire->tokens[0]=original;std::memcpy(wire->embd.data(),&original_bits,sizeof(original_bits));
                switch (defect) {
                    case 0: {const auto changed=original_bits^1;std::memcpy(wire->embd.data(),&changed,sizeof(changed));break;}
                    case 1: wire->tokens[0].seq_ids={1};break;
                    case 2: ++wire->tokens[0].pos[0];break;
                    case 3: wire->tokens[0].embd_off=wire->embd.size();break;
                    case 4: wire->tokens[0].id=10;break;
                    case 5: ++wire->tokens[0].pos[1];break;
                }
                t.out<<"DFlash source negative profile="<<defect<<std::endl;
                if (!t.assert_equal("reject malformed source before draft KV mutation",-1,
                        llama_process(&child,LLAMA_PROCESS_TYPE_DECODE,wire))) return;
                for (llama_seq_id sequence=0;sequence<2;++sequence)
                    t.assert_equal(llama_pos(-1),llama_memory_seq_pos_max(child.get_memory(),sequence));
            }
            wire->tokens[0]=original;std::memcpy(wire->embd.data(),&original_bits,sizeof(original_bits));
        }
        if (!t.assert_equal("admit exact staged target feature injection",0,llama_process(&child,LLAMA_PROCESS_TYPE_DECODE,wire))) return;
        common_speculative_begin(driver.get(),slot,prompts[slot]);
        auto & request=common_speculative_get_draft_params(driver.get(),slot);
        request.drafting=true;request.pos0=llama_pos(count);request.id_last=seeds[slot];request.prompt=&prompts[slot];request.result=&proposals[slot];
        if (partition) request.n_max=int32_t(partition_limits[slot]);
    }
    common_speculative_draft(driver.get());child.synchronize();
    if (partition) {
        const auto stats=common_speculative_get_stats(driver.get());
        const size_t forwards=partition_limits[0]==partition_limits[1] ? 1 : 2;
        if (!t.assert_true(stats.size()==1) || !t.assert_equal("complete draft blocks use exactly the compatible native partitions",forwards,stats[0].n_predict)) return;
        for (llama_seq_id slot=0;slot<2;++slot) {
            const auto limit=partition_limits[slot];
            t.assert_equal(size_t(limit),proposals[slot].size());
            t.assert_equal("the native draft writes only the requested block width",llama_pos(prompts[slot].size()+limit),
                llama_memory_seq_pos_max(child.get_memory(),slot));
        }
        auto profile=native_dflash_draft_profile(*model,*draft);profile[0]=9;
        kv_stream_bit_oracle output;
        if (streamed) {
            std::unique_ptr<FILE,kv_stream_exact_file_free> reference(std::fopen(output_path,"rb"));
            if (!t.assert_true(reference!=nullptr)) return;
            output.load_control(reference.get(),profile);output.compare();
        }
        const auto limits=partition_limits;output.bytes("limits",limits.data(),sizeof(limits));
        for (llama_seq_id slot=0;slot<2;++slot) {
            output.bytes("proposals/"+std::to_string(slot),proposals[slot].data(),proposals[slot].size()*sizeof(llama_token));
            kv_stream_exact_kv(output,"draft_KV/"+std::to_string(slot),child.get_kv_resources().bounded,0,
                prompts[slot].size()+limits[slot]+1,slot);
        }
        const auto * lattice=child.get_embeddings_nextn();if (!t.assert_true(lattice!=nullptr)) return;
        const auto width=limits[1]+1;
        const auto canonical=dflash_exact_lattice(lattice+(forwards==1 ? size_t(width)*draft->hparams.n_embd_out() : 0),width,
            draft->hparams.n_embd_out(),draft->hparams.dflash_selector_top_k,model->vocab.n_tokens());
        output.bytes("selector_lattice/1",canonical.data(),canonical.size()*sizeof(uint32_t));
        if (streamed) output.finish();
        else {
            std::unique_ptr<FILE,kv_stream_exact_file_free> destination(std::fopen(output_path,"wb"));
            if (!t.assert_true(destination!=nullptr)) return;
            output.save_control(destination.get(),profile);
        }
        t.out<<"bounded DFlash native-exact records="<<output.records()<<" bytes="<<output.byte_count()<<" forwards="<<forwards
            <<" blocks="<<limits[0]+1<<','<<limits[1]+1<<'\n';
        return;
    }
    if (repeat) {
        std::array<image_writer,2> states;
        for (llama_seq_id slot=0;slot<2;++slot) child.get_memory()->state_write(states[slot],slot,0);
        child.get_memory()->clear(true);
        for (llama_seq_id slot=0;slot<2;++slot) {
            image_reader restored(states[slot].bytes);child.get_memory()->state_read(restored,slot,0);restored.flush_tensor_reads();
            if (!t.assert_equal(states[slot].bytes.size(),restored.n_bytes())) return;
        }
        child.synchronize();
        for (llama_seq_id slot=0;slot<2;++slot) {
            if (!t.assert_true(llama_memory_seq_rm(child.get_memory(),slot,llama_pos(prompts[slot].size()),-1))) return;
            proposals[slot].clear();common_speculative_get_draft_params(driver.get(),slot).drafting=true;
        }
        common_speculative_draft(driver.get());child.synchronize();
    }
    if (suffix) {
        if (streamed && !t.assert_true("known target suffix rollback succeeds",llama_memory_seq_rm(target->get_memory(),0,4,-1))) return;
        if (streamed) {
            kv_stream_bit_oracle unchanged;
            for (llama_seq_id slot=0;slot<2;++slot)
                kv_stream_exact_kv(unchanged,"rejected_trim/"+std::to_string(slot),child.get_kv_resources().bounded,0,prompts[slot].size()+8,slot);
            if (!t.assert_true("reject draft trim below the proven target suffix",!llama_memory_seq_rm(child.get_memory(),0,3,-1)) ||
                    !t.assert_true("a peer cannot consume another source's rollback proof",!llama_memory_seq_rm(child.get_memory(),1,4,-1))) return;
            unchanged.compare();
            for (llama_seq_id slot=0;slot<2;++slot)
                kv_stream_exact_kv(unchanged,"rejected_trim/"+std::to_string(slot),child.get_kv_resources().bounded,0,prompts[slot].size()+8,slot);
            unchanged.finish();
        }
        if (!t.assert_true(llama_memory_seq_rm(child.get_memory(),0,4,-1) && llama_memory_seq_rm(child.get_memory(),1,7,-1))) return;
        prompts[0].resize(4);
        const auto & logits=reference_logits[3];
        seeds[0]=llama_token(std::max_element(logits.begin(),logits.end())-logits.begin());
        for (llama_seq_id slot=0;slot<2;++slot) {
            proposals[slot].clear();auto & request=common_speculative_get_draft_params(driver.get(),slot);
            request.drafting=true;request.pos0=llama_pos(prompts[slot].size());request.id_last=seeds[slot];
        }
        common_speculative_draft(driver.get());child.synchronize();
    }
    if (std::getenv("LLAMA_TEST_DFLASH_SHAPE_DUMP")) {
        auto * cache=child.get_kv_resources().bounded;
        const auto & cells=cache->get_cells(0);
        for (llama_seq_id sequence=0;sequence<2;++sequence) {
            t.out<<"draft native cell map slot="<<sequence;
            for (const auto & entry : cells.seq_pos_get(sequence)) t.out<<' '<<entry.first<<':'<<entry.second;
            t.out<<std::endl;
        }
    }
    kv_stream_bit_oracle output;
    if (streamed || native_teacher) {
        const auto * initial=native_teacher ? std::getenv("LLAMA_TEST_DFLASH_INITIAL_DRAFT_ORACLE") : output_path;
        if (!t.assert_true(initial!=nullptr)) return;
        std::unique_ptr<FILE,kv_stream_exact_file_free> reference(std::fopen(initial,"rb"));
        if (!t.assert_true(reference!=nullptr)) return;
        auto profile=native_dflash_draft_profile(*model,*draft);if (suffix) {profile[0]=5;profile[5]=4;}
        output.load_control(reference.get(),profile);output.compare();
    }
    output.bytes("seeds",seeds.data(),sizeof(seeds));
    for (llama_seq_id slot=0;slot<2;++slot) {
        if (!t.assert_equal("source-ready proposal count",size_t(7),proposals[slot].size())) return;
        for (auto token : proposals[slot]) t.assert_true(token>=0 && uint32_t(token)<model->vocab.n_tokens());
        output.bytes("proposals/"+std::to_string(slot),proposals[slot].data(),proposals[slot].size()*sizeof(llama_token));
        auto * cache=child.get_kv_resources().bounded;
        if (!t.assert_true(cache!=nullptr)) return;
        kv_stream_exact_kv(output,"draft_KV/"+std::to_string(slot),cache,0,prompts[slot].size()+8,slot);
    }
    const auto * lattice=child.get_embeddings_nextn();const auto values=size_t(16)*draft->hparams.n_embd_out();
    if (!t.assert_true(lattice && std::all_of(lattice,lattice+values,[](float value) {return std::isfinite(value);}))) return;
    if (const char * path=std::getenv("LLAMA_TEST_DFLASH_LATTICE_DUMP")) {
        std::unique_ptr<FILE,kv_stream_exact_file_free> dump(std::fopen(path,"wb"));
        if (!dump || std::fwrite(lattice,sizeof(float),values,dump.get())!=values)
            throw std::runtime_error("cannot preserve exact selector-lattice diagnostic");
    }
    for (size_t slot=0;slot<2;++slot) {
        const auto canonical=dflash_exact_lattice(lattice+slot*8*draft->hparams.n_embd_out(),8,
            draft->hparams.n_embd_out(),draft->hparams.dflash_selector_top_k,model->vocab.n_tokens());
        output.bytes("selector_lattice/"+std::to_string(slot),canonical.data(),canonical.size()*sizeof(uint32_t));
    }
    if (native_teacher) {
        output.finish();test_dflash_after_teacher(t,*model,*draft,target,child,driver.get(),prompts,proposals,seeds,native_teacher,output_path,false);
        return;
    }
    if (streamed) {
        output.finish();
        if (const auto * verify_path=std::getenv("LLAMA_TEST_DFLASH_TARGET_VERIFY_ORACLE")) {
            std::unique_ptr<FILE,kv_stream_exact_file_free> reference(std::fopen(verify_path,"rb"));
            if (!t.assert_true(reference!=nullptr)) return;
            auto profile=native_dflash_stage_profile(*model,*draft);profile[0]=11;
            kv_stream_bit_oracle verified;verified.load_control(reference.get(),profile);verified.compare();
            std::array<uint8_t,32> target_parent{},draft_parent{};
            verified.input("target_parent_sha256",target_parent.data(),target_parent.size());
            verified.input("draft_parent_sha256",draft_parent.data(),draft_parent.size());
            common_batch verify(target);
            std::array<int32_t,16> verify_tokens{},verify_positions{},verify_sequences{};
            std::array<llama_kv_stream_logical_checkpoint,2> before_verify{};
            for (llama_seq_id slot=0;slot<2;++slot) {
                std::shared_ptr<llama_kv_stream_logical_cache> cache;
                if (!t.assert_true(group->source_table()->get(group->source_handles()[slot],0,cache))) return;
                before_verify[slot]=cache->checkpoint();
                if (!t.assert_true(llama_memory_seq_rm(child.get_memory(),slot,llama_pos(prompts[slot].size()),-1))) return;
                for (int32_t offset=0;offset<8;++offset) {
                    const auto row=slot*8+offset;const auto token=offset ? proposals[slot][offset-1] : seeds[slot];
                    const auto position=llama_pos(prompts[slot].size())+offset;
                    verify.add(token,position,slot,true);verify_tokens[row]=token;verify_positions[row]=position;verify_sequences[row]=slot;
                }
            }
            llama_set_kv_stream_decode(target,true);
            if (!t.assert_equal(0,llama_process(target,LLAMA_PROCESS_TYPE_DECODE,verify.get()))) return;
            target->synchronize();
            verified.bytes("tokens",verify_tokens.data(),sizeof(verify_tokens));
            verified.bytes("positions",verify_positions.data(),sizeof(verify_positions));
            verified.bytes("sequences",verify_sequences.data(),sizeof(verify_sequences));
            verified.bytes("target_layers",draft->target_layer_ids.data(),draft->target_layer_ids.size()*sizeof(int32_t));
            for (auto layer : draft->target_layer_ids) {
                const auto * values=target->get_embeddings_layer_inp(uint32_t(layer));
                if (!t.assert_true(values!=nullptr)) return;
                verified.bytes("features/"+std::to_string(layer),values,size_t(16)*model->hparams.n_embd*sizeof(float));
            }
            for (int32_t row=0;row<16;++row) {
                const auto * values=target->get_logits_ith(row);if (!t.assert_true(values!=nullptr)) return;
                verified.bytes("logits/"+std::to_string(row),values,size_t(model->vocab.n_tokens())*sizeof(float));
            }
            // Native exported checkpoints remain inputs for later staged target controls.
            // Public grouped checkpoint serialization is a separate qualification stage.
            for (llama_seq_id slot=0;slot<2;++slot) {
                uint64_t bytes=0;verified.input("target_state_size/"+std::to_string(slot),&bytes,sizeof(bytes));
                if (!t.assert_true(bytes>0 && bytes<=(uint64_t(2)<<30))) return;
                std::vector<uint8_t> checkpoint(size_t(bytes),0);
                verified.input("target_state/"+std::to_string(slot),checkpoint.data(),checkpoint.size());
            }
            verified.finish();
            const auto capture_target=[&](kv_stream_bit_oracle & retained) {
                for (auto layer : draft->target_layer_ids) {
                    const auto * values=target->get_embeddings_layer_inp(uint32_t(layer));if (!t.assert_true(values!=nullptr)) return false;
                    retained.bytes("retained_features/"+std::to_string(layer),values,size_t(16)*model->hparams.n_embd*sizeof(float));
                }
                for (int32_t row=0;row<16;++row) {
                    const auto * values=target->get_logits_ith(row);if (!t.assert_true(values!=nullptr)) return false;
                    retained.bytes("retained_logits/"+std::to_string(row),values,size_t(model->vocab.n_tokens())*sizeof(float));
                }
                return true;
            };
            kv_stream_bit_oracle retained_target;
            if (!capture_target(retained_target)) return;
            const auto before_counters=common_speculative_get_stats(driver.get());
            if (!t.assert_true("verified target frame retains current source provenance",llama_serial_features_match(target,verify.get(),
                    draft->target_layer_ids.data(),uint32_t(draft->target_layer_ids.size())))) return;
            for (llama_seq_id slot=0;slot<2;++slot) {
                std::shared_ptr<llama_kv_stream_logical_cache> cache;
                if (!t.assert_true(group->source_table()->get(group->source_handles()[slot],0,cache))) return;
                const auto after=cache->checkpoint();
                t.out<<"teacher source="<<slot<<" generation="<<before_verify[slot].generation<<"->"<<after.generation
                    <<" tokens="<<before_verify[slot].tokens<<"->"<<after.tokens<<'\n';
            }
            kv_stream_bit_oracle retained;
            for (llama_seq_id slot=0;slot<2;++slot)
                kv_stream_exact_kv(retained,"rejected_target_metadata/"+std::to_string(slot),child.get_kv_resources().bounded,0,prompts[slot].size(),slot);
            if (std::getenv("LLAMA_TEST_DFLASH_COMMON_SOURCE_GUARD")) {
                const auto token=verify.tokens[0].id;verify.tokens[0].id=token ? 0 : 1;
                if (!t.assert_true("common DFlash rejects token metadata from a different target invocation",
                        !common_speculative_process(driver.get(),verify))) return;
                retained.compare();
                for (llama_seq_id slot=0;slot<2;++slot)
                    kv_stream_exact_kv(retained,"rejected_target_metadata/"+std::to_string(slot),child.get_kv_resources().bounded,0,prompts[slot].size(),slot);
                retained.finish();verify.tokens[0].id=token;
            }
            if (!t.assert_true("matching target verification features remain usable after rejected metadata",
                    common_speculative_process(driver.get(),verify))) return;
            child.synchronize();
            retained_target.compare();if (!capture_target(retained_target)) return;retained_target.finish();
            const auto after_counters=common_speculative_get_stats(driver.get());
            if (!t.assert_true(before_counters.size()==1 && after_counters.size()==1 &&
                    before_counters[0].n_predict==after_counters[0].n_predict && after_counters[0].n_predict==1 &&
                    before_counters[0].n_gen_tokens==after_counters[0].n_gen_tokens && after_counters[0].n_gen_tokens==14)) return;
            for (llama_seq_id slot=0;slot<2;++slot)
                if (!t.assert_equal(llama_pos(prompts[slot].size()+7),llama_memory_seq_pos_max(child.get_memory(),slot))) return;
            t.out<<"streamed target verification exact records="<<verified.records()<<" bytes="<<verified.byte_count()
                <<" teacher_rows=8,8 rejected_metadata=1\n";
            if (const auto * next=std::getenv("LLAMA_TEST_DFLASH_NEXT_DRAFT_ORACLE"))
                if (!test_dflash_after_teacher(t,*model,*draft,target,child,driver.get(),prompts,proposals,seeds,verify_path,next,true)) return;
        }
        if (suffix) {
            kv_stream_bit_oracle unchanged;
            for (llama_seq_id slot=0;slot<2;++slot)
                kv_stream_exact_kv(unchanged,"unknown_generation/"+std::to_string(slot),child.get_kv_resources().bounded,0,prompts[slot].size()+8,slot);
            std::shared_ptr<llama_kv_stream_logical_cache> source;
            if (!t.assert_true(group->source_table()->get(group->source_handles()[0],0,source))) return;
            const auto before=source->checkpoint();
            if (!t.assert_true("same-frontier restore is an unknown semantic change",source->restore(before)) ||
                    !t.assert_true("unknown generation cannot authorize a draft trim",!llama_memory_seq_rm(child.get_memory(),0,4,-1))) return;
            for (llama_seq_id slot=0;slot<2;++slot) {
                proposals[slot].clear();common_speculative_get_draft_params(driver.get(),slot).drafting=true;
            }
            common_speculative_draft(driver.get());child.synchronize();
            for (const auto & proposed : proposals) if (!t.assert_true("unknown generation cannot produce a draft",proposed.empty())) return;
            unchanged.compare();
            for (llama_seq_id slot=0;slot<2;++slot)
                kv_stream_exact_kv(unchanged,"unknown_generation/"+std::to_string(slot),child.get_kv_resources().bounded,0,prompts[slot].size()+8,slot);
            unchanged.finish();
        }
        if (parking) {
            const auto counters=[&] {
                std::vector<std::array<size_t,9>> result;
                for (const auto & stats : common_speculative_get_stats(driver.get())) result.push_back({size_t(stats.type),
                    stats.n_call_begin,stats.n_call_draft,stats.n_call_accept,stats.n_gen_drafts,stats.n_acc_drafts,
                    stats.n_gen_tokens,stats.n_acc_tokens,stats.n_predict});
                return result;
            };
            const auto before_counters=rebind ? counters() : std::vector<std::array<size_t,9>>{};
            kv_stream_bit_oracle retained;
            for (llama_seq_id slot=0;slot<2;++slot)
                kv_stream_exact_kv(retained,"parked_KV/"+std::to_string(slot),child.get_kv_resources().bounded,0,prompts[slot].size()+8,slot);
            const bool reset_selected=std::getenv("LLAMA_TEST_DFLASH_RESET_SELECTED")!=nullptr;
            kv_stream_bit_oracle retained_peer;
            if (reset_selected) kv_stream_exact_kv(retained_peer,"unmodified_peer",child.get_kv_resources().bounded,0,prompts[1].size()+8,1);
            std::vector<llama_context_group_sequence_state> before,after;
            if (!t.assert_true(target->save_group_sequences(before))) return;
            auto * parent=target->group_parent();
            if (rebind) {
                auto other=llama_context_group::create(*model,cp);
                auto child_params=cp;child_params.n_rs_seq=0;child_params.shared_device_memory_bytes=0;
                if (!t.assert_true(other && other->prepare_dflash(*draft,child_params,8)) ||
                        !t.assert_true("another group's metadata cannot inherit this driver's source history",
                            !common_speculative_rebind(driver.get(),other->metadata_context(),other->auxiliary_metadata_context()))) return;
                if (!t.assert_true(!common_speculative_rebind(driver.get(),nullptr,&child) &&
                        !common_speculative_rebind(driver.get(),target,nullptr) &&
                        !common_speculative_rebind(driver.get(),target,target) &&
                        !common_speculative_rebind(driver.get(),&child,target))) return;
                if (!t.assert_true("retain the DFlash driver before retiring its execution contexts",
                        common_speculative_rebind(driver.get(),group->metadata_context(),group->auxiliary_metadata_context()))) return;
            } else driver.reset();
            if (!t.assert_true("park populated draft banks while retaining their canonical source proof",group->park())) return;
            if (reset_selected) {
                if (!t.assert_true(rebind && group->reset_request(0))) return;
                llama_kv_stream_selection selection;
                if (!t.assert_true(llama_kv_stream_select({512,2},{{0,0},{1,7}},selection)==llama_kv_stream_selection_status::success)) return;
                auto input=llama_batch_init(3,0,1);
                struct release_batch {llama_batch & batch;~release_batch() {llama_batch_free(batch);}} release{input};
                input.n_tokens=3;
                for (int32_t row=0;row<3;++row) {
                    input.token[row]=70+row;input.pos[row]=row;input.n_seq_id[row]=1;input.seq_id[row][0]=0;input.logits[row]=true;
                }
                llama_context_memory_create_result result;
                const bool active=group->activate(selection,group->source_handles(),input,llama_memory_work_phase::target_prefill,1,2,&result);
                t.out<<"mixed empty/populated activation status="<<llama_context_memory_create_status_name(result.status)<<'\n';
                if (!t.assert_true("a reset source and populated peer can share a new selection",active)) return;
                if (!t.assert_true(common_speculative_rebind(driver.get(),group->context(),group->auxiliary_context()))) return;
                retained_peer.compare();
                kv_stream_exact_kv(retained_peer,"unmodified_peer",group->auxiliary_context()->get_kv_resources().bounded,0,prompts[1].size()+8,1);
                retained_peer.finish();
                if (!t.assert_true(group->context()->save_group_sequences(after))) return;
                const auto saved=std::find_if(before.begin(),before.end(),[](const auto & state) {return state.sequence==1;});
                const auto restored=std::find_if(after.begin(),after.end(),[](const auto & state) {return state.sequence==1;});
                if (!t.assert_true(saved!=before.end() && restored!=after.end() && saved->recurrent==restored->recurrent &&
                        saved->cache_generation==restored->cache_generation && saved->tokens==restored->tokens)) return;
                auto * current=group->context();auto * draft_context=group->auxiliary_context();
                kv_stream_bit_oracle unchanged_peer;
                kv_stream_exact_kv(unchanged_peer,"live_peer",draft_context->get_kv_resources().bounded,0,prompts[1].size()+8,1);
                prompts[0].clear();proposals[0].clear();common_speculative_begin(driver.get(),0,prompts[0]);
                auto & request=common_speculative_get_draft_params(driver.get(),0);
                request.drafting=true;request.pos0=0;request.id_last=70;
                const auto predictions=common_speculative_get_stats(driver.get())[0].n_predict;
                common_speculative_draft(driver.get());
                if (!t.assert_true("an empty reset source cannot fabricate a ready draft prefix",proposals[0].empty() &&
                        common_speculative_get_stats(driver.get())[0].n_predict==predictions &&
                        common_speculative_get_last_type(driver.get(),1)==COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH)) return;
                common_batch fresh(current);
                for (int32_t row=0;row<3;++row) {fresh.add(70+row,row,0,true);prompts[0].push_back(70+row);}
                llama_set_kv_stream_decode(current,false);
                if (!t.assert_equal(0,llama_process(current,LLAMA_PROCESS_TYPE_DECODE,fresh.get())) ||
                        !t.assert_true(common_speculative_process(driver.get(),fresh))) return;
                for (int32_t row=0;row<3;++row) {
                    const auto * logits=current->get_logits_ith(row);
                    if (!t.assert_true(logits && std::all_of(logits,logits+model->vocab.n_tokens(),[](float x) {return std::isfinite(x);}))) return;
                }
                unchanged_peer.compare();
                kv_stream_exact_kv(unchanged_peer,"live_peer",draft_context->get_kv_resources().bounded,0,prompts[1].size()+8,1);
                unchanged_peer.finish();
                std::vector<llama_context_group_sequence_state> completed;
                if (!t.assert_true(current->save_group_sequences(completed))) return;
                const auto peer_state=std::find_if(completed.begin(),completed.end(),[](const auto & state) {return state.sequence==1;});
                if (!t.assert_true(peer_state!=completed.end() && peer_state->recurrent==saved->recurrent &&
                        peer_state->cache_generation==saved->cache_generation && peer_state->tokens==saved->tokens)) return;
                if (!t.assert_true(llama_memory_seq_pos_max(draft_context->get_memory(),0)==2 &&
                        common_speculative_rebind(driver.get(),group->metadata_context(),group->auxiliary_metadata_context()) && group->park())) return;
                t.out<<"mixed empty/populated DFlash peer checkpoint and KV remain exact\n";
                return;
            }
            llama_kv_stream_selection selected;
            if (!t.assert_true(llama_kv_stream_select({512,2},{{0,5},{1,7}},selected)==llama_kv_stream_selection_status::success)) return;
            auto anchor=llama_batch_init(2,0,1);
            struct release_batch {llama_batch & batch;~release_batch() {llama_batch_free(batch);}} release{anchor};
            anchor.n_tokens=2;
            for (llama_seq_id slot=0;slot<2;++slot) {
                anchor.token[slot]=seeds[slot];anchor.pos[slot]=llama_pos(prompts[slot].size());anchor.n_seq_id[slot]=1;
                anchor.seq_id[slot][0]=slot;anchor.logits[slot]=true;
            }
            llama_context_memory_create_result admission;
            const bool activated=group->activate(selected,group->source_handles(),anchor,llama_memory_work_phase::target_verify,1,2,&admission);
            t.out<<"DFlash populated resume status="<<llama_context_memory_create_status_name(admission.status)
                <<" required="<<admission.required_bytes<<" available="<<admission.available_bytes<<std::endl;
            if (!t.assert_true(activated)) return;
            auto * resumed=group->auxiliary_context();
            if (!t.assert_true(resumed && resumed->group_parent()==parent && group->context()->group_parent()==parent)) return;
            if (rebind && !t.assert_true("retarget the retained DFlash driver after activation",
                    common_speculative_rebind(driver.get(),group->context(),resumed))) return;
            retained.compare();
            for (llama_seq_id slot=0;slot<2;++slot)
                kv_stream_exact_kv(retained,"parked_KV/"+std::to_string(slot),resumed->get_kv_resources().bounded,0,prompts[slot].size()+8,slot);
            retained.finish();
            if (!t.assert_true(group->context()->save_group_sequences(after) && before.size()==after.size())) return;
            for (size_t i=0;i<before.size();++i) t.assert_true(before[i].recurrent==after[i].recurrent &&
                before[i].full_columns==after[i].full_columns && before[i].cache_generation==after[i].cache_generation);
            if (rebind) {
                if (!t.assert_true("rebind preserves every speculative counter",counters()==before_counters)) return;
                for (llama_seq_id slot=0;slot<2;++slot) {
                    if (!t.assert_true(common_speculative_get_last_type(driver.get(),slot)==COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH &&
                            llama_memory_seq_rm(resumed->get_memory(),slot,llama_pos(prompts[slot].size()),-1))) return;
                    proposals[slot].clear();common_speculative_get_draft_params(driver.get(),slot).drafting=true;
                }
                common_speculative_draft(driver.get());resumed->synchronize();
                const auto * repeated_path=std::getenv("LLAMA_TEST_DFLASH_REPEAT_ORACLE");
                if (!t.assert_true(repeated_path!=nullptr)) return;
                std::unique_ptr<FILE,kv_stream_exact_file_free> reference(std::fopen(repeated_path,"rb"));
                if (!t.assert_true(reference!=nullptr)) return;
                kv_stream_bit_oracle repeated_output;
                auto repeated_profile=native_dflash_draft_profile(*model,*draft);repeated_profile[0]=7;
                repeated_output.load_control(reference.get(),repeated_profile);repeated_output.compare();
                repeated_output.bytes("seeds",seeds.data(),sizeof(seeds));
                for (llama_seq_id slot=0;slot<2;++slot) {
                    if (!t.assert_equal(size_t(7),proposals[slot].size())) return;
                    repeated_output.bytes("proposals/"+std::to_string(slot),proposals[slot].data(),proposals[slot].size()*sizeof(llama_token));
                    kv_stream_exact_kv(repeated_output,"draft_KV/"+std::to_string(slot),resumed->get_kv_resources().bounded,0,prompts[slot].size()+8,slot);
                }
                const auto * repeated=resumed->get_embeddings_nextn();if (!t.assert_true(repeated!=nullptr)) return;
                for (size_t slot=0;slot<2;++slot) {
                    const auto canonical=dflash_exact_lattice(repeated+slot*8*draft->hparams.n_embd_out(),8,
                        draft->hparams.n_embd_out(),draft->hparams.dflash_selector_top_k,model->vocab.n_tokens());
                    repeated_output.bytes("selector_lattice/"+std::to_string(slot),canonical.data(),canonical.size()*sizeof(uint32_t));
                }
                repeated_output.finish();
                auto expected=before_counters;
                for (auto & stats : expected) {
                    ++stats[2];
                    if (stats[0]==size_t(COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH)) {stats[4]+=2;stats[6]+=14;++stats[8];}
                }
                t.assert_true("one joint prediction advances only the exact per-source draft counters",counters()==expected);
            }
        }
        t.out<<"streamed DFlash staged native exact records="<<output.records()<<" bytes="<<output.byte_count()<<std::endl;
        return;
    }
    std::unique_ptr<FILE,kv_stream_exact_file_free> destination(std::fopen(output_path,"wb"));if (!t.assert_true(destination!=nullptr)) return;
    auto profile=native_dflash_draft_profile(*model,*draft);if (suffix) {profile[0]=5;profile[5]=4;}
    if (repeat) profile[0]=7;
    output.save_control(destination.get(),profile);
    t.out<<"native draft stage records="<<output.records()<<" bytes="<<output.byte_count()<<" target_NN=0 proposal_counts=7,7\n";
}

#include "kv-stream-dflash-group-test.h"

static void test_group_dflash_image_features(testing & t,llama_model * model,const char * draft_path,const char * image_path,bool skip=false,bool teacher=false,bool tail=false,bool direct=false,bool peer=false) {
    if (!t.assert_true(model && model->arch==LLM_ARCH_QWEN35 && draft_path && image_path)) return;
    auto mp=llama_model_default_params();mp.n_gpu_layers=999;
    model_ptr draft(llama_model_load_from_file(draft_path,mp),llama_model_free);
    if (!t.assert_true(draft && draft->arch==LLM_ARCH_DFLASH && !draft->target_layer_ids.empty())) return;
    const size_t width=model->hparams.n_embd,rows=64;
    std::vector<float> embeddings(rows*width);std::ifstream file(image_path,std::ios::binary);
    file.read(reinterpret_cast<char *>(embeddings.data()),std::streamsize(embeddings.size()*sizeof(float)));
    if (!t.assert_true(file && file.peek()==EOF && std::all_of(embeddings.begin(),embeddings.end(),[](float x) {return std::isfinite(x);}))) return;
    auto cp=llama_context_default_params();cp.n_ctx=512;cp.n_batch=cp.n_ubatch=256;cp.n_seq_max=2;cp.kv_unified=true;cp.n_rs_seq=3;cp.swa_full=false;
    cp.n_outputs_max=8;cp.n_outputs_max_per_seq=4;cp.n_threads=cp.n_threads_batch=4;
    cp.type_k=GGML_TYPE_Q8_0;cp.type_v=GGML_TYPE_Q4_0;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
    const auto picture=[&](llama_context & context) {
        common_batch batch(&context);
        for (size_t row=0;row<rows;++row) {
            const llama_pos position[4]={0,llama_pos(row/8),llama_pos(row%8),0};
            batch.add_embd({embeddings.data()+row*width,1,width},position,1,row+1==rows);
        }
        return batch;
    };
    const auto suffix=[&](llama_context & context) {
        common_batch next(&context);
        if (tail) {
            const llama_pos position[4]={8,8,8,0};
            next.add_embd({embeddings.data()+(rows-1)*width,1,width},position,1,true);
        } else next.add(500,8,1,true);
        return next;
    };
    kv_stream_bit_oracle oracle,teacher_vectors,teacher_state,predictions,peer_vectors;
    llama_token peer_anchor=LLAMA_TOKEN_NULL;
    const auto peer_prefix=[&](llama_context & context) {
        common_batch batch(&context);
        for (int row=0;row<3;++row) batch.add(100+row,row,0,row==2);
        return batch;
    };
    const auto capture=[&](llama_context & context,kv_stream_bit_oracle & output,size_t count) {
        context.synchronize();
        for (auto layer : draft->target_layer_ids) {
            const auto * values=llama_get_embeddings_layer_inp(&context,uint32_t(layer));
            if (!t.assert_true(values && std::all_of(values,values+count*width,[](float x) {return std::isfinite(x);}))) return false;
            output.bytes("features/"+std::to_string(layer),values,count*width*sizeof(float));
        }
        const auto * logits=llama_get_logits(&context);
        if (!t.assert_true(logits!=nullptr)) return false;
        output.bytes("logits",logits,size_t(model->vocab.n_tokens())*sizeof(float));return true;
    };
    const auto driver=[&](llama_context * target,llama_context * child) {
        common_params_speculative parameters;parameters.types={COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH};
        parameters.draft.ctx_tgt=target;parameters.draft.ctx_dft=child;parameters.draft.n_max=3;
        return common_speculative_ptr(common_speculative_init(parameters,2));
    };
    const auto predict=[&](llama_context & target,llama_context & child,common_speculative * spec,llama_token & anchor,llama_pos position=9) {
        if (anchor==LLAMA_TOKEN_NULL) {
            const auto * logits=llama_get_logits(&target);
            if (!t.assert_true(logits!=nullptr)) return false;
            anchor=llama_token(std::max_element(logits,logits+model->vocab.n_tokens())-logits);
        }
        llama_tokens history(rows,LLAMA_TOKEN_NULL),proposal;
        llama_tokens other_history={100,101,102},other_proposal;
        if (!direct) history.push_back(tail ? LLAMA_TOKEN_NULL : 500);
        if (!t.assert_true("remove provisional DFlash rows before repeating the native prediction",
                llama_memory_seq_rm(child.get_memory(),1,position,-1))) return false;
        common_speculative_begin(spec,1,history);
        common_speculative_get_draft_params(spec,1)={true,3,position,anchor,&history,&proposal,nullptr,0.0f,123};
        if (peer) {
            if (!t.assert_true(llama_memory_seq_rm(child.get_memory(),0,3,-1))) return false;
            common_speculative_begin(spec,0,other_history);
            common_speculative_get_draft_params(spec,0)={true,3,3,peer_anchor,&other_history,&other_proposal,nullptr,0.0f,123};
        }
        bool completed=true;
        try {common_speculative_draft(spec);child.synchronize();}
        catch (const std::exception & error) {t.out<<"image-conditioned DFlash failure: "<<error.what()<<'\n';completed=false;}
        if (!t.assert_true(target.group_parent() ? "group DFlash prediction uses native position 9 after image teacher injection" :
                "ordinary native image-conditioned DFlash prediction succeeds",completed)) return false;
        if (!t.assert_equal(size_t(3),proposal.size())) return false;
        const auto * values=llama_get_embeddings_nextn(&child);
        const auto lattice=dflash_exact_lattice(values+(peer ? 4*size_t(draft->hparams.n_embd_out()) : 0),4,draft->hparams.n_embd_out(),
            16,model->vocab.n_tokens());
        predictions.bytes("proposal",proposal.data(),proposal.size()*sizeof(llama_token));
        predictions.bytes("selector",lattice.data(),lattice.size()*sizeof(uint32_t));
        if (peer) {
            if (!t.assert_equal(size_t(3),other_proposal.size())) return false;
            const auto other=dflash_exact_lattice(values,4,draft->hparams.n_embd_out(),16,model->vocab.n_tokens());
            predictions.bytes("peer_proposal",other_proposal.data(),other_proposal.size()*sizeof(llama_token));
            predictions.bytes("peer_selector",other.data(),other.size()*sizeof(uint32_t));
        }
        return true;
    };
    {
        llama_context native(*model,cp);
        for (auto layer : draft->target_layer_ids) native.set_embeddings_layer_inp(uint32_t(layer),true);
        std::unique_ptr<llama_context> child;common_speculative_ptr spec;
        if (skip) {
            auto dp=cp;dp.ctx_other=&native;dp.n_rs_seq=0;
            child=std::make_unique<llama_context>(*draft,dp);spec=driver(&native,child.get());
            if (!t.assert_true(spec!=nullptr)) return;
        }
        if (peer) {
            auto prefix=peer_prefix(native);
            if (!t.assert_equal(0,llama_process(&native,LLAMA_PROCESS_TYPE_DECODE,prefix.get())) ||
                    !capture(native,peer_vectors,3) || !t.assert_true(common_speculative_process(spec.get(),prefix))) return;
            const auto * logits=llama_get_logits(&native);peer_anchor=llama_token(std::max_element(logits,logits+model->vocab.n_tokens())-logits);
        }
        auto batch=picture(native);
        if (!t.assert_equal(0,llama_process(&native,LLAMA_PROCESS_TYPE_DECODE,batch.get())) || !capture(native,oracle,rows)) return;
        if (skip) {
            if (!t.assert_true(common_speculative_process(spec.get(),batch))) return;
            for (llama_seq_id slot=0;slot<2;++slot)
                if (!t.assert_equal("ordinary native DFlash skips pinned image rows",llama_pos(peer && !slot ? 2 : -1),llama_memory_seq_pos_max(child->get_memory(),slot))) return;
        }
        if (direct) {
            llama_token anchor=LLAMA_TOKEN_NULL;
            if (!predict(native,*child,spec.get(),anchor,8)) return;
        }
        if (teacher) {
            std::array<std::vector<uint8_t>,2> saved;
            for (llama_seq_id slot=0;slot<2;++slot) {
                saved[slot].resize(llama_state_seq_get_size(&native,slot));
                if (!t.assert_equal(saved[slot].size(),llama_state_seq_get_data(&native,saved[slot].data(),saved[slot].size(),slot))) return;
            }
            llama_memory_clear(native.get_memory(),true);
            for (llama_seq_id slot=0;slot<2;++slot)
                if (!t.assert_equal(saved[slot].size(),llama_state_seq_set_data(&native,saved[slot].data(),saved[slot].size(),slot))) return;
            auto next=suffix(native);
            if (!t.assert_equal(0,llama_process(&native,LLAMA_PROCESS_TYPE_DECODE,next.get())) ||
                    !capture(native,teacher_vectors,1) || !t.assert_true(common_speculative_process(spec.get(),next))) return;
            if (!t.assert_equal(llama_pos(8),llama_memory_seq_pos_max(child->get_memory(),1))) return;
            child->synchronize();
            kv_stream_exact_state state;child->get_kv_resources().bounded->state_write(state,1,0);
            state.compare_with(teacher_state,"teacher");
            llama_token anchor=LLAMA_TOKEN_NULL;
            if (!predict(native,*child,spec.get(),anchor)) return;
            std::array<std::vector<uint8_t>,2> states;
            for (llama_seq_id slot=0;slot<2;++slot) {
                states[slot].resize(llama_state_seq_get_size(child.get(),slot));
                if (!t.assert_equal(states[slot].size(),llama_state_seq_get_data(child.get(),states[slot].data(),states[slot].size(),slot))) return;
            }
            llama_memory_clear(child->get_memory(),true);
            for (llama_seq_id slot=0;slot<2;++slot)
                if (!t.assert_equal(states[slot].size(),llama_state_seq_set_data(child.get(),states[slot].data(),states[slot].size(),slot))) return;
            if (!predict(native,*child,spec.get(),anchor)) return;
        }
    }
    cp.shared_device_memory_bytes=size_t(1024)*1048576;
    auto group=llama_context_group::create(*model,cp);
    auto dp=cp;dp.n_rs_seq=0;dp.shared_device_memory_bytes=0;
    if (!t.assert_true(group && group->prepare_dflash(*draft,dp,4))) return;
    if (peer) {
        auto prefix=peer_prefix(*group->metadata_context());
        llama_batch_allocr allocator(uint32_t(prefix.get()->n_pos_per_embd));llama_kv_stream_selection selected;
        if (!t.assert_true(allocator.init(*prefix.get(),model->vocab,false) &&
                llama_kv_stream_select({512,2},{{0,0},{1,0}},selected)==llama_kv_stream_selection_status::success &&
                group->activate(selected,group->source_handles(),allocator.get_batch(),llama_memory_work_phase::target_prefill,1,10))) return;
        auto spec=driver(group->context(),group->auxiliary_context());
        if (!t.assert_true(spec!=nullptr) || !t.assert_equal(0,llama_process(group->context(),LLAMA_PROCESS_TYPE_DECODE,prefix.get()))) return;
        peer_vectors.compare();if (!capture(*group->context(),peer_vectors,3)) return;peer_vectors.finish();
        if (!t.assert_true(common_speculative_process(spec.get(),prefix) &&
                common_speculative_rebind(spec.get(),group->metadata_context(),group->auxiliary_metadata_context()) && group->park())) return;
    }
    auto batch=picture(*group->metadata_context());
    llama_batch_allocr allocator(uint32_t(batch.get()->n_pos_per_embd));
    llama_kv_stream_selection selection;
    if (!t.assert_true(allocator.init(*batch.get(),model->vocab,false) &&
            llama_kv_stream_select({512,2},{{0,peer ? 3u : 0u},{1,0}},selection)==llama_kv_stream_selection_status::success &&
            group->activate(selection,group->source_handles(),allocator.get_batch(),llama_memory_work_phase::target_prefill,1,1))) return;
    auto * target=group->context();
    if (!t.assert_equal(0,llama_process(target,LLAMA_PROCESS_TYPE_DECODE,batch.get()))) return;
    oracle.compare();if (!capture(*target,oracle,rows)) return;oracle.finish();
    if (!t.assert_true("completed grouped image features retain their exact source embedding and spatial positions",
            llama_serial_features_match(target,batch.get(),draft->target_layer_ids.data(),uint32_t(draft->target_layer_ids.size())))) return;
    batch.tokens[1].pos[1]=7;
    t.assert_true(!llama_serial_features_match(target,batch.get(),draft->target_layer_ids.data(),uint32_t(draft->target_layer_ids.size())));
    batch.tokens[1].pos[1]=0;
    const float saved=embeddings[1];embeddings[1]=std::nextafter(saved,1.0f);
    t.assert_true(!llama_serial_features_match(target,batch.get(),draft->target_layer_ids.data(),uint32_t(draft->target_layer_ids.size())));
    embeddings[1]=saved;
    t.assert_true(llama_serial_features_match(target,batch.get(),draft->target_layer_ids.data(),uint32_t(draft->target_layer_ids.size())));
    if (skip) {
        auto spec=driver(target,group->auxiliary_context());
        if (!t.assert_true(spec && common_speculative_process(spec.get(),batch))) return;
        for (llama_seq_id slot=0;slot<2;++slot)
            if (!t.assert_equal("group DFlash preserves the native image skip",llama_pos(peer && !slot ? 2 : -1),llama_memory_seq_pos_max(group->auxiliary_context()->get_memory(),slot))) return;
        const auto * logits=llama_get_logits(target);
        if (!t.assert_true(logits!=nullptr)) return;
        auto image_anchor=llama_token(std::max_element(logits,logits+model->vocab.n_tokens())-logits);
        if (!t.assert_true(common_speculative_rebind(spec.get(),group->metadata_context(),group->auxiliary_metadata_context()))) return;
        if (!t.assert_true("a validated native image skip covers the image source before parking",group->park())) return;
        t.out<<"group DFlash native image skip park=1\n";
        if (direct) {
            common_batch query(group->metadata_context());query.add(image_anchor,8,1,true);
            llama_batch_allocr allocator(uint32_t(query.get()->n_pos_per_embd));
            if (!t.assert_true(allocator.init(*query.get(),model->vocab,false) &&
                    llama_kv_stream_select({512,2},{{0,0},{1,rows}},selection)==llama_kv_stream_selection_status::success &&
                    group->activate(selection,group->source_handles(),allocator.get_batch(),llama_memory_work_phase::target_verify,1,2))) return;
            target=group->context();auto * child=group->auxiliary_context();
            if (!t.assert_true(common_speculative_rebind(spec.get(),target,child))) return;
            predictions.compare();if (!predict(*target,*child,spec.get(),image_anchor,8)) return;predictions.finish();
            if (!t.assert_true(common_speculative_rebind(spec.get(),group->metadata_context(),group->auxiliary_metadata_context()) && group->park())) return;
            t.out<<"group DFlash direct image prediction native_exact=1 native_position=8 physical_rows=64\n";return;
        }
        if (!teacher) return;
        auto next=suffix(*group->metadata_context());
        llama_batch_allocr allocator(uint32_t(next.get()->n_pos_per_embd));
        if (!t.assert_true(allocator.init(*next.get(),model->vocab,false) &&
                llama_kv_stream_select({512,2},{{0,peer ? 3u : 0u},{1,rows}},selection)==llama_kv_stream_selection_status::success &&
                group->activate(selection,group->source_handles(),allocator.get_batch(),llama_memory_work_phase::target_prefill,1,2))) return;
        target=group->context();auto * child=group->auxiliary_context();
        if (!t.assert_true(common_speculative_rebind(spec.get(),target,child)) ||
                !t.assert_equal(0,llama_process(target,LLAMA_PROCESS_TYPE_DECODE,next.get()))) return;
        teacher_vectors.compare();if (!capture(*target,teacher_vectors,1)) return;teacher_vectors.finish();
        if (!t.assert_true(tail ? "a single image row uses the same exact source proof as native DFlash injection" :
                "DFlash teacher injection uses native position 8 after 64 image rows",common_speculative_process(spec.get(),next))) return;
        if (!t.assert_equal(llama_pos(8),llama_memory_seq_pos_max(child->get_memory(),1))) return;
        child->synchronize();kv_stream_exact_state state;child->get_kv_resources().bounded->state_write(state,1,0);
        teacher_state.compare();state.compare_with(teacher_state,"teacher");teacher_state.finish();
        llama_token anchor=LLAMA_TOKEN_NULL;
        predictions.compare();if (!predict(*target,*child,spec.get(),anchor)) return;
        std::vector<llama_context_group_sequence_state> states;
        if (!t.assert_true(child->save_group_sequences(states) && states.size()==2)) return;
        for (const auto & state : states) if (state.sequence==1) {
            if (!t.assert_equal(llama_pos(8),state.draft_position) || !t.assert_equal(size_t(65),state.tokens)) return;
        }
        if (!t.assert_true(common_speculative_rebind(spec.get(),group->metadata_context(),group->auxiliary_metadata_context()) && group->park())) return;
        auto * parent=group->prepare_vision_parent();std::vector<ggml_backend_memory_lease_t> loans;
        if (!t.assert_true(parent && group->lend_vision({cp.shared_device_memory_bytes},loans) && loans.size()==1)) return;
        ggml_backend_buffer_clear(ggml_backend_memory_lease_buffer(loans.front()),0xa5);
        for (auto * loan : loans) ggml_backend_memory_lease_free(loan);
        if (!t.assert_true(group->release_vision())) return;
        common_batch query(group->metadata_context());query.add(anchor,9,1,true);
        llama_batch_allocr query_allocator(uint32_t(query.get()->n_pos_per_embd));
        if (!t.assert_true(query_allocator.init(*query.get(),model->vocab,false) &&
                llama_kv_stream_select({512,2},{{0,peer ? 3u : 0u},{1,rows+1}},selection)==llama_kv_stream_selection_status::success &&
                group->activate(selection,group->source_handles(),query_allocator.get_batch(),llama_memory_work_phase::target_verify,1,3))) return;
        target=group->context();child=group->auxiliary_context();
        if (!t.assert_true(target->group_parent()==parent && child->group_parent()==parent && common_speculative_rebind(spec.get(),target,child)) ||
                !predict(*target,*child,spec.get(),anchor)) return;
        predictions.finish();
        if (!t.assert_true(common_speculative_rebind(spec.get(),group->metadata_context(),group->auxiliary_metadata_context()) && group->park())) return;
        t.out<<"group DFlash image teacher native_exact=1 position=8 physical_rows=65 restored_provisional=1 peer="<<peer<<'\n';return;
    }
    t.assert_true(group->discard());
    t.out<<"group DFlash image source native_exact=1 rows="<<rows<<" layers="<<draft->target_layer_ids.size()<<'\n';
}

static void test_group_dflash_metadata(testing & t,llama_model * model,const char * path,bool features=false,bool quotation=false,bool gpu=false,
        bool binding=false,const char * native_export=nullptr) {
    if (!t.assert_true(model && model->arch==LLM_ARCH_QWEN35 && path)) return;
    const llama_model_tensor_buft_override weights[]={{".*",ggml_backend_cpu_buffer_type()},{nullptr,nullptr}};
    auto mp=llama_model_default_params();mp.n_gpu_layers=999;if (!gpu) mp.tensor_buft_overrides=weights;
    model_ptr draft(llama_model_load_from_file(path,mp),llama_model_free);
    if (!t.assert_true(draft && draft->arch==LLM_ARCH_DFLASH && draft->hparams.dflash_selector_rank>0)) return;
    if (gpu) {
        const auto placed=[](const llama_model & model) {
            size_t bytes=0;
            for (const auto & item : model.memory_breakdown()) {
                auto * device=ggml_backend_buft_get_device(item.first);
                if (device && ggml_backend_dev_type(device)==GGML_BACKEND_DEVICE_TYPE_GPU) bytes+=item.second;
            }
            return bytes;
        };
        if (!t.assert_true("both native workload catalogs use the actual GPU weights",
                placed(*model)>size_t(1024)*1024*1024 && placed(*draft)>size_t(32)*1024*1024)) return;
    }
    auto cp=llama_context_default_params();
    cp.n_ctx=512;cp.n_batch=cp.n_ubatch=256;cp.n_seq_max=2;cp.kv_unified=true;cp.swa_full=false;
    cp.n_outputs_max=16;cp.n_outputs_max_per_seq=8;cp.n_rs_seq=7;
    cp.type_k=GGML_TYPE_Q8_0;cp.type_v=GGML_TYPE_Q4_0;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.shared_device_memory_bytes=size_t(1024)*1024*1024;
    auto group=llama_context_group::create(*model,cp);
    if (!t.assert_true(group!=nullptr)) return;
    auto batch=llama_batch_init(12,0,1);
    struct release_batch {llama_batch & batch;~release_batch() {llama_batch_free(batch);}} release{batch};
    batch.n_tokens=0;
    for (llama_seq_id slot=0;slot<2;++slot) for (int32_t row=0;row<(slot ? 7 : 5);++row) {
        const auto i=batch.n_tokens++;batch.token[i]=10+slot*20+row;batch.pos[i]=row;
        batch.n_seq_id[i]=1;batch.seq_id[i][0]=slot;batch.logits[i]=true;
    }
    std::map<uint32_t,std::vector<float>> reference_features;
    std::vector<std::vector<float>> reference_logits;
    auto child=cp;child.n_rs_seq=0;child.shared_device_memory_bytes=0;
    {
        auto reference=child;
        reference.n_rs_seq=features ? cp.n_rs_seq : 0;
        if (features && gpu) t.out<<"native two-model feature control: GPU weights, two slots, K7 snapshots, C512\n"<<std::flush;
        llama_context native_target(*model,reference);
        reference.ctx_other=&native_target;reference.n_rs_seq=0;
        llama_context native(*draft,reference);
        t.assert_true(native.get_kv_resources().bounded && native.get_kv_resources().full &&
            native.get_kv_resources().full->get_layer_ids().empty());
        if (features) {
            for (auto layer : draft->target_layer_ids) native_target.set_embeddings_layer_inp(uint32_t(layer),true);
            if (!t.assert_equal(0,native_target.decode(batch))) return;
            native_target.synchronize();
            const auto values=size_t(batch.n_tokens)*model->hparams.n_embd;
            for (auto layer : draft->target_layer_ids) {
                const auto * data=native_target.get_embeddings_layer_inp(uint32_t(layer));
                if (!t.assert_true(data!=nullptr)) return;
                reference_features[uint32_t(layer)]={data,data+values};
            }
            for (int32_t row=0;row<batch.n_tokens;++row) {
                const auto * logits=native_target.get_logits_ith(row);
                if (!t.assert_true(logits!=nullptr)) return;
                reference_logits.push_back({logits,logits+model->vocab.n_tokens()});
            }
            if (native_export) {
                kv_stream_bit_oracle oracle;
                std::array<int32_t,12> tokens{},positions{},sequences{};
                std::array<llama_token,2> seeds{};
                for (int32_t row=0;row<batch.n_tokens;++row) {
                    tokens[row]=batch.token[row];positions[row]=batch.pos[row];sequences[row]=batch.seq_id[row][0];
                }
                for (llama_seq_id slot=0;slot<2;++slot) {
                    const auto & logits=reference_logits[slot ? 11 : 4];
                    if (!t.assert_true(std::all_of(logits.begin(),logits.end(),[](float value) {return std::isfinite(value);}))) return;
                    seeds[slot]=llama_token(std::max_element(logits.begin(),logits.end())-logits.begin());
                }
                oracle.bytes("tokens",tokens.data(),sizeof(tokens));oracle.bytes("positions",positions.data(),sizeof(positions));
                oracle.bytes("sequences",sequences.data(),sizeof(sequences));oracle.bytes("seeds",seeds.data(),sizeof(seeds));
                const auto borrowed=native_dflash_borrowed_profile(*model);
                oracle.bytes("target_borrowed_weights",borrowed.data(),sizeof(borrowed));
                oracle.bytes("target_layers",draft->target_layer_ids.data(),draft->target_layer_ids.size()*sizeof(int32_t));
                for (auto layer : draft->target_layer_ids) {
                    const auto & values=reference_features.at(uint32_t(layer));
                    if (!t.assert_true(std::all_of(values.begin(),values.end(),[](float value) {return std::isfinite(value);}))) return;
                    oracle.bytes("features/"+std::to_string(layer),values.data(),values.size()*sizeof(float));
                }
                for (size_t row=0;row<reference_logits.size();++row)
                    oracle.bytes("logits/"+std::to_string(row),reference_logits[row].data(),reference_logits[row].size()*sizeof(float));
                for (llama_seq_id slot=0;slot<2;++slot) {
                    const uint64_t size=llama_state_seq_get_size(&native_target,slot);
                    if (!t.assert_true(size>0 && size<=(uint64_t(2)<<30))) return;
                    std::vector<uint8_t> state(size_t(size),0);
                    if (!t.assert_equal(size_t(size),llama_state_seq_get_data(&native_target,state.data(),state.size(),slot))) return;
                    oracle.bytes("target_state_size/"+std::to_string(slot),&size,sizeof(size));
                    oracle.bytes("target_state/"+std::to_string(slot),state.data(),state.size());
                }
                std::unique_ptr<FILE,kv_stream_exact_file_free> output(std::fopen(native_export,"wb"));
                if (!t.assert_true(output!=nullptr)) return;
                oracle.save_control(output.get(),native_dflash_stage_profile(*model,*draft));
                t.out<<"native target stage records="<<oracle.records()<<" bytes="<<oracle.byte_count()<<" saved="<<native_export<<std::endl;
                return;
            }
        }
    }
    const auto original=group->metadata_context()->get_cparams().embeddings_layer_inp;
    for (uint32_t invalid=0;invalid<8;++invalid) {
        auto bad=child;uint32_t rows=4;
        switch (invalid) {
            case 0: bad.n_seq_max=1;break;
            case 1: ++bad.n_ctx;break;
            case 2: bad.shared_device_memory_bytes=1048576;break;
            case 3: bad.n_rs_seq=1;break;
            case 4: bad.type_v=GGML_TYPE_Q5_1;break;
            case 5: bad.ctx_type=LLAMA_CONTEXT_TYPE_MTP;break;
            case 6: rows=1;break;
            case 7: rows=9;break;
        }
        t.assert_true(!group->prepare_dflash(*draft,bad,rows));
        t.assert_true(!group->auxiliary_metadata_context() &&
            group->metadata_context()->get_cparams().embeddings_layer_inp==original);
    }
    if (!t.assert_true("declare DFlash3 independently of the target K7 verification budget",
            group->prepare_dflash(*draft,child,4))) return;
    auto * metadata=group->auxiliary_metadata_context();
    t.assert_true(metadata && metadata->group_storage_pending() && !metadata->group_parent() &&
        metadata->group_target_context()==group->metadata_context() && !metadata->get_cparams().shared_device_memory_bytes &&
        !metadata->get_cparams().n_rs_seq && metadata->get_cparams().serial_block_rows==4);
    t.assert_true(metadata->get_kv_resources().bounded && metadata->get_kv_resources().bounded->bounded_storage_pending());
    for (const auto & entry : metadata->get_memory()->memory_breakdown()) t.assert_equal(size_t(0),entry.second);
    for (auto layer : draft->target_layer_ids)
        t.assert_true(group->metadata_context()->get_cparams().embeddings_layer_inp[uint32_t(layer)]);
    t.assert_true(!group->prepare_dflash(*draft,child,4));
    for (const auto & handle : group->source_handles()) {
        std::shared_ptr<llama_kv_stream_logical_cache> cache;
        t.assert_true(group->source_table()->get(handle,0,cache) && !cache->host()->committed_bytes() && !cache->tokens());
    }
    if (quotation) {
        auto * target=group->metadata_context();
        std::vector<llama_memory_phase_cache_identity> sources;
        for (const auto & handle : group->source_handles()) {
            std::shared_ptr<llama_kv_stream_logical_cache> cache;
            if (!t.assert_true(group->source_table()->get(handle,0,cache))) return;
            sources.push_back({cache->identity().id,cache->identity().generation});
        }
        const llama_memory_phase_cache_identity bounded{llama_kv_stream_cache_ids_reserve(1),1};
        llama_context_group_plan target_only;
        if (!t.assert_true("the native target workload fits before adding the bounded draft",
                target->quote_target_group(batch,sources,bounded,llama_memory_work_phase::target_prefill,
                    90,cp.shared_device_memory_bytes,target_only).status==llama_context_memory_create_status::success)) return;
        llama_context_group_plan quoted;
        const auto result=target->quote_target_group(batch,sources,bounded,llama_memory_work_phase::target_prefill,
            91,cp.shared_device_memory_bytes,quoted,metadata);
        if (!t.assert_true("one quote covers the target and the independent bounded DFlash workloads",
                result.status==llama_context_memory_create_status::success)) return;
        t.assert_equal(size_t(3),quoted.caches.size());
        t.assert_true(quoted.bounded.allocation.bytes>0 && quoted.bounded.cache.identity.id==bounded.id);
        bool inject=false,predict=false,joint=false;
        for (const auto & item : quoted.catalog) {
            bool block=false;
            for (const auto & member : item.measurement.workload.members) {
                if (member.phase!=llama_memory_work_phase::draft_feature_injection &&
                        member.phase!=llama_memory_work_phase::draft_block_prediction) continue;
                const bool prediction=member.phase==llama_memory_work_phase::draft_block_prediction;
                inject=inject || !prediction;predict=predict || prediction;block=block || prediction;
                t.assert_true(member.destination_cache.id==bounded.id && !member.output_rows &&
                    member.hidden_output_rows==(prediction ? member.input_rows : 0));
            }
            for (const auto & native : item.measurement.workload.native_batches) joint=joint || (block && native.size()==2);
        }
        t.assert_true(inject && predict && joint);
        const auto minimum=quoted.budget.parent_minimum_bytes;
        t.assert_true(minimum>=target_only.budget.parent_minimum_bytes+quoted.bounded.allocation.bytes);
        t.out<<"group DFlash quote minimum="<<minimum<<" bounded="<<quoted.bounded.allocation.bytes
            <<" target_only="<<target_only.budget.parent_minimum_bytes<<" profiles="<<quoted.catalog.size()<<std::endl;
        const auto short_fit=target->quote_target_group(batch,sources,bounded,llama_memory_work_phase::target_prefill,
            91,minimum-1,quoted,metadata);
        t.assert_true(short_fit.status==llama_context_memory_create_status::insufficient_budget &&
            short_fit.required_bytes==minimum && short_fit.additional_bytes==1 && quoted.budget.parent_minimum_bytes==minimum);
        for (const auto & entry : metadata->get_memory()->memory_breakdown()) t.assert_equal(size_t(0),entry.second);
        for (const auto & handle : group->source_handles()) {
            std::shared_ptr<llama_kv_stream_logical_cache> cache;
            t.assert_true(group->source_table()->get(handle,0,cache) && !cache->tokens() && !cache->host()->committed_bytes());
        }
    }
    if (binding) {
        llama_kv_stream_selection selection;
        if (!t.assert_true(llama_kv_stream_select({cp.n_ctx,2},{{0,0},{1,0}},selection)==llama_kv_stream_selection_status::success)) return;
        if (!t.assert_true("bind the measured DFlash child to the selected target's single parent",
                group->activate(selection,group->source_handles(),batch,llama_memory_work_phase::target_prefill,1,1))) return;
        auto * target=group->context();auto * auxiliary=group->auxiliary_context();
        if (!t.assert_true(target && auxiliary && target->group_parent() && auxiliary->group_parent()==target->group_parent())) return;
        t.assert_equal(cp.shared_device_memory_bytes,ggml_backend_buffer_get_size(target->group_parent()));
        t.assert_true(!auxiliary->get_kv_resources().bounded->bounded_storage_pending());
        auto * workspace=llama_context_compute_memory(auxiliary);
        t.assert_true(workspace && workspace->external_workspace() && workspace->external_workspace()->ready());
        t.assert_true(!auxiliary->get_cparams().shared_device_memory_bytes && !auxiliary->get_cparams().n_rs_seq);
        for (auto layer : draft->target_layer_ids) t.assert_true(target->get_cparams().embeddings_layer_inp[uint32_t(layer)]);
        auto * parent=target->group_parent();
        if (!t.assert_true(group->park() && !group->context() && !group->auxiliary_context())) return;
        if (!t.assert_true(group->activate(selection,group->source_handles(),batch,llama_memory_work_phase::target_prefill,1,2))) return;
        t.assert_true(group->context()->group_parent()==parent && group->auxiliary_context()->group_parent()==parent);
        t.assert_true(group->park());
    }
    if (features) {
        // Isolate target feature provenance before the draft execution catalog is enabled.
        auto * target=group->metadata_context();
        llama_kv_stream_selection selection;
        auto projection=std::make_shared<llama_kv_stream_slot_projection>();
        if (!t.assert_true(llama_kv_stream_select({cp.n_ctx,2},{{0,0},{1,0}},selection)==llama_kv_stream_selection_status::success &&
                group->source_table()->project(selection,group->source_handles(),0,*projection))) return;
        if (!t.assert_true(target->initialize_group_execution(group->source_table(),projection,batch,
                llama_memory_work_phase::target_prefill,1,1))) return;
        if (!t.assert_equal(0,target->decode(batch))) return;
        target->synchronize();
        for (const auto & item : reference_features) {
            const auto * data=target->get_embeddings_layer_inp(item.first);
            if (!t.assert_true(data && !std::memcmp(data,item.second.data(),item.second.size()*sizeof(float)))) return;
        }
        for (int32_t row=0;row<batch.n_tokens;++row) {
            const auto * logits=target->get_logits_ith(row);
            if (!t.assert_true(logits && !std::memcmp(logits,reference_logits[row].data(),reference_logits[row].size()*sizeof(float)))) return;
        }
        llama_batch_compat source(target,batch);
        if (!t.assert_true("completed two-slot native-exact target features have sequence-aware provenance",
                llama_serial_features_match(target,source.batch_ext,draft->target_layer_ids.data(),uint32_t(draft->target_layer_ids.size())))) return;
        const auto old=batch.seq_id[0][0];batch.seq_id[0][0]=1;
        llama_batch_compat wrong_sequence(target,batch);
        t.assert_true(!llama_serial_features_match(target,wrong_sequence.batch_ext));batch.seq_id[0][0]=old;
        llama_batch_compat restored(target,batch);t.assert_true(llama_serial_features_match(target,restored.batch_ext));
        std::shared_ptr<llama_kv_stream_logical_cache> cache;
        if (!t.assert_true(group->source_table()->get(group->source_handles()[1],0,cache))) return;
        const auto before=cache->checkpoint();
        t.assert_true("idle cancellation preserves the completed feature frame",!cache->cancel() &&
            cache->identity().generation==before.generation && llama_serial_features_match(target,restored.batch_ext));
        if (!t.assert_true("restoring unchanged host bytes advances semantic generation",cache->restore(before))) return;
        t.assert_true(cache->tokens()==before.tokens && cache->identity().generation!=before.generation &&
            !llama_serial_features_match(target,restored.batch_ext));
    }
}

static void test_group_rollback(testing & t,llama_model * model,bool recurrent=false,bool owner_failure=false,
        uint32_t maximum=3,bool endpoints=false,bool interleave=false) {
    if (!t.assert_equal(recurrent ? LLM_ARCH_QWEN35 : LLM_ARCH_GEMMA4,model->arch)) return;
    auto cp=llama_context_default_params();
    cp.n_ctx=recurrent ? 512 : 4096;cp.n_batch=cp.n_ubatch=256;cp.n_rs_seq=recurrent ? maximum : 0;cp.n_seq_max=2;cp.kv_unified=true;cp.swa_full=false;
    cp.n_outputs_max=2*(maximum+1);cp.n_outputs_max_per_seq=maximum+1;
    cp.type_k=GGML_TYPE_Q8_0;cp.type_v=GGML_TYPE_Q4_0;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.n_threads=cp.n_threads_batch=4;
    const size_t vocabulary=model->vocab.n_tokens(),hidden=model->hparams.n_embd_out();
    auto batch=llama_batch_init(int32_t(std::max(10u,2*(maximum+1))),0,1);
    struct release_batch {llama_batch & batch;~release_batch() {llama_batch_free(batch);}} release{batch};
    const auto input=[&](std::array<uint32_t,2> widths,std::array<uint32_t,2> past,bool all=true) {
        batch.n_tokens=0;
        for (llama_seq_id slot=0;slot<2;++slot) for (uint32_t row=0;row<widths[slot];++row) {
            const auto i=batch.n_tokens++;batch.token[i]=llama_token(10+slot*20+past[slot]+row);batch.pos[i]=past[slot]+row;
            batch.n_seq_id[i]=1;batch.seq_id[i][0]=slot;batch.logits[i]=all || row+1==widths[slot];
        }
    };
    struct record {std::vector<float> logits,hidden;};
    struct control {
        uint32_t depth;
        std::array<uint32_t,2> accepted;
        std::vector<record> verify,next,resumed;
        std::array<std::vector<uint8_t>,2> bounded,recurrent;
    };
    const auto capture=[&](llama_context & context,std::vector<record> & records) {
        for (int32_t row=0;row<batch.n_tokens;++row) {
            auto * logits=llama_get_logits_ith(&context,row);auto * h=llama_get_embeddings_nextn_ith(&context,row);
            if (!t.assert_true(logits && h && std::all_of(logits,logits+vocabulary,[](float x) {return std::isfinite(x);}) &&
                    std::all_of(h,h+hidden,[](float x) {return std::isfinite(x);}))) return false;
            records.push_back({{logits,logits+vocabulary},{h,h+hidden}});
        }
        return true;
    };
    const auto compare=[&](llama_context & context,const std::vector<record> & records) {
        if (!t.assert_equal(size_t(batch.n_tokens),records.size())) return false;
        for (int32_t row=0;row<batch.n_tokens;++row) {
            auto * logits=llama_get_logits_ith(&context,row);auto * h=llama_get_embeddings_nextn_ith(&context,row);
            if (!t.assert_true(logits && h && !std::memcmp(logits,records[row].logits.data(),vocabulary*sizeof(float)) &&
                    !std::memcmp(h,records[row].hidden.data(),hidden*sizeof(float)))) return false;
        }
        return true;
    };
    const auto native_params=cp;
    const auto check=[&](const control & item) {
        auto cp=native_params;
        cp.shared_device_memory_bytes=size_t(1024)*1024*1024;cp.kv_stream_verify_max=recurrent ? 0 : maximum+1;
        t.out<<"rollback depth="<<item.depth<<" accepted="<<item.accepted[0]<<','<<item.accepted[1]<<std::endl;
        auto group=llama_context_group::create(*model,cp);if (!t.assert_true(group!=nullptr)) return false;
        llama_set_embeddings_nextn(group->metadata_context(),true,false);
        llama_kv_stream_selection selection;
        if (!t.assert_true(llama_kv_stream_select({cp.n_ctx,2},{{0,0},{1,0}},selection)==llama_kv_stream_selection_status::success)) return false;
        input({5,5},{0,0},false);
        if (!t.assert_true(group->activate(selection,group->source_handles(),batch,llama_memory_work_phase::target_prefill,1,1))) return false;
        if (!t.assert_equal(0,group->context()->decode(batch)) || !t.assert_true(group->park())) return false;
        if (!t.assert_true(llama_kv_stream_select({cp.n_ctx,2},{{0,5},{1,5}},selection)==llama_kv_stream_selection_status::success)) return false;
        input({item.depth+1,item.depth+1},{5,5});
        if (!t.assert_true(group->activate(selection,group->source_handles(),batch,llama_memory_work_phase::target_verify,1,2))) return false;
        auto * context=group->context();
        t.out<<"rollback phase=verify"<<std::endl;
        if (!t.assert_equal(0,context->decode(batch)) || !compare(*context,item.verify)) return false;
        if (owner_failure) {
            std::vector<llama_context_group_sequence_state> before,after;
            if (!t.assert_true(context->save_group_sequences(before))) return false;
            context->get_kv_resources().full->get_group_model()->bind_mutation([](const std::function<bool()> &) {return false;});
            if (!t.assert_true(!llama_memory_seq_rm(context->get_memory(),0,6,-1))) return false;
            if (!t.assert_true("failed whole-hybrid rollback closes group admission",!context->save_group_sequences(after))) return false;
            input({1,1},{7,7});t.assert_equal(-1,context->decode(batch));
            input({2,2},{5,5});if (!compare(*context,item.verify)) return false;
            for (size_t i=0;i<before.size();++i) {
                std::shared_ptr<llama_kv_stream_logical_cache> cache;
                if (!t.assert_true(group->source_table()->get(group->source_handles()[i],0,cache))) return false;
                t.assert_equal(before[i].cache_generation,cache->identity().generation);t.assert_equal(before[i].tokens,cache->tokens());
            }
            auto * parent=context->group_parent();
            if (!t.assert_true(group->discard() && group->reset_request(0) && group->reset_request(1))) return false;
            if (!t.assert_true(llama_kv_stream_select({cp.n_ctx,2},{{0,0},{1,0}},selection)==llama_kv_stream_selection_status::success)) return false;
            input({5,5},{0,0},false);
            if (!t.assert_true(group->activate(selection,group->source_handles(),batch,llama_memory_work_phase::target_prefill,1,3))) return false;
            if (!t.assert_equal(0,group->context()->decode(batch)) || !t.assert_true(group->park())) return false;
            if (!t.assert_true(llama_kv_stream_select({cp.n_ctx,2},{{0,5},{1,5}},selection)==llama_kv_stream_selection_status::success)) return false;
            input({2,2},{5,5});
            if (!t.assert_true(group->activate(selection,group->source_handles(),batch,llama_memory_work_phase::target_verify,1,4))) return false;
            context=group->context();t.assert_true(context->group_parent()==parent);
            if (!t.assert_equal(0,context->decode(batch)) || !compare(*context,item.verify)) return false;
            return t.assert_true(group->park());
        }
        if (item.depth==1 && !item.accepted[0] && !item.accepted[1]) {
            std::vector<llama_context_group_sequence_state> before,after;
            if (!t.assert_true(context->save_group_sequences(before))) return false;
            t.assert_true(!llama_memory_seq_rm(context->get_memory(),-2,6,-1));
            t.assert_true(!llama_memory_seq_rm(context->get_memory(),2,6,-1));
            t.assert_true(!llama_memory_seq_rm(context->get_memory(),0,1,2));
            t.assert_true(llama_memory_seq_rm(context->get_memory(),1,99,-1));
            if (!t.assert_true(context->save_group_sequences(after) && before.size()==after.size())) return false;
            for (size_t i=0;i<before.size();++i) {
                t.assert_equal(before[i].tokens,after[i].tokens);t.assert_equal(before[i].cache_generation,after[i].cache_generation);
                t.assert_true(before[i].bounded==after[i].bounded);
            }
            if (!compare(*context,item.verify)) return false;
        }
        for (llama_seq_id slot=0;slot<2;++slot) {
            const auto handle=group->source_handles()[slot],peer=group->source_handles()[1-slot];
            std::shared_ptr<llama_kv_stream_logical_cache> cache,other;
            if (!t.assert_true(group->source_table()->get(handle,0,cache) && group->source_table()->get(peer,0,other))) return false;
            const auto before=cache->checkpoint(),unchanged=other->checkpoint();
            const auto keep=size_t(6+item.accepted[slot]);
            if (!t.assert_true(llama_memory_seq_rm(context->get_memory(),slot,llama_pos(keep),-1))) return false;
            if (!t.assert_equal(keep,cache->tokens())) return false;
            t.assert_equal(before.generation+uint64_t(keep<before.tokens),cache->identity().generation);
            t.assert_true(other->checkpoint().id==unchanged.id && other->checkpoint().generation==unchanged.generation &&
                other->tokens()==unchanged.tokens);
        }
        t.out<<"rollback phase=bounded-state"<<std::endl;
        std::vector<llama_context_group_sequence_state> saved;
        if (!t.assert_true(context->save_group_sequences(saved) && saved.size()==2)) return false;
        for (const auto & state : saved) {
            if (!t.assert_true(state.sequence>=0 && state.sequence<2 && state.bounded==item.bounded[state.sequence] && state.recurrent==item.recurrent[state.sequence])) return false;
        }
        input({1,1},{6+item.accepted[0],6+item.accepted[1]});
        t.out<<"rollback phase=continuous"<<std::endl;
        if (!t.assert_equal(0,context->decode(batch)) || !compare(*context,item.next)) return false;
        if (item.depth==3 && !item.accepted[0] && !item.accepted[1]) {
            if (!t.assert_true(context->save_group_sequences(saved))) return false;
            t.assert_true(saved[1].full_columns==std::vector<uint32_t>({0,1,2,3,4,6,5}));
        }
        auto * parent=context->group_parent();
        if (!t.assert_true(group->park())) return false;
        if (!t.assert_true(llama_kv_stream_select({cp.n_ctx,2},{{0,7+item.accepted[0]},{1,7+item.accepted[1]}},selection)==llama_kv_stream_selection_status::success)) return false;
        input({1,1},{7+item.accepted[0],7+item.accepted[1]});
        if (!t.assert_true(group->activate(selection,group->source_handles(),batch,llama_memory_work_phase::target_verify,1,3))) return false;
        context=group->context();t.assert_true(context->group_parent()==parent);
        t.out<<"rollback phase=resumed"<<std::endl;
        if (!t.assert_equal(0,context->decode(batch)) || !compare(*context,item.resumed)) return false;
        if (!t.assert_true(group->park())) return false;
        return true;
    };
    std::vector<control> controls;
    size_t expected_controls=0,expected_records=0;
    for (uint32_t depth=endpoints || interleave ? maximum : 1;depth<=maximum;++depth) for (uint32_t a=0;a<=depth;++a) for (uint32_t b=0;b<=depth;++b) {
        if (owner_failure && (depth!=1 || a || b)) continue;
        if (endpoints && !((a==0 && b==depth) || (a==depth && b==0))) continue;
        ++expected_controls;expected_records+=2*(depth+1)+4;
        control item{depth,{a,b},{},{},{},{},{}};
        const auto generate=[&] {
            auto native_owner=std::make_unique<llama_context>(*model,cp);auto & native=*native_owner;
            llama_set_embeddings_nextn(&native,true,false);
            input({5,5},{0,0},false);if (!t.assert_equal(0,native.decode(batch))) return false;
            input({depth+1,depth+1},{5,5});if (!t.assert_equal(0,native.decode(batch)) || !capture(native,item.verify)) return false;
            for (llama_seq_id slot=0;slot<2;++slot) {
                if (!t.assert_true(llama_memory_seq_rm(native.get_memory(),slot,llama_pos(6+item.accepted[slot]),-1))) return false;
            }
            native.synchronize();
            auto * local=native.get_kv_resources().bounded;
            for (llama_seq_id slot=0;slot<2;++slot) {
                image_writer bounded,state;
                if (local) local->state_write(bounded,slot,0);
                if (auto * memory=native.get_kv_resources().recurrent) memory->state_write(state,slot,0);
                if (!t.assert_true(!bounded.bytes.empty() || !state.bytes.empty())) return false;
                item.bounded[slot]=std::move(bounded.bytes);item.recurrent[slot]=std::move(state.bytes);
            }
            input({1,1},{6+a,6+b});if (!t.assert_equal(0,native.decode(batch)) || !capture(native,item.next)) return false;
            std::array<std::vector<uint8_t>,2> states;
            for (llama_seq_id slot=0;slot<2;++slot) {
                auto & state=states[slot];state.resize(llama_state_seq_get_size(&native,slot));
                if (!t.assert_true(!state.empty() && llama_state_seq_get_data(&native,state.data(),state.size(),slot)==state.size())) return false;
            }
            native_owner.reset();
            llama_context resumed(*model,cp);llama_set_embeddings_nextn(&resumed,true,false);
            for (llama_seq_id slot=0;slot<2;++slot) {
                const auto & state=states[slot];
                if (!t.assert_equal(state.size(),llama_state_seq_set_data(&resumed,state.data(),state.size(),slot))) return false;
            }
            input({1,1},{7+a,7+b});if (!t.assert_equal(0,resumed.decode(batch)) || !capture(resumed,item.resumed)) return false;
            return true;
        };
        if (!generate()) return;
        if (interleave) {if (!check(item)) return;}
        else controls.push_back(std::move(item));
    }
    if (!interleave && !t.assert_equal(expected_controls,controls.size())) return;
    if (owner_failure) t.out<<"group rollback denial native controls=1 exact_records=8\n";
    else t.out<<"group rollback native controls="<<expected_controls<<" exact_records="<<expected_records
        <<" depths="<<(endpoints || interleave ? maximum : 1)<<'-'<<maximum
        <<(endpoints ? " asymmetric_boundary_pairs" : " all_acceptance_pairs")<<" park_resume\n";
    if (!interleave) for (const auto & item : controls) if (!check(item)) return;
}
static void test_group_compute_failure(testing & t,llama_model * model) {
    auto cp=llama_context_default_params();
    cp.n_ctx=4096;cp.n_batch=cp.n_ubatch=256;cp.n_seq_max=2;cp.kv_unified=true;cp.swa_full=false;
    cp.type_k=GGML_TYPE_Q8_0;cp.type_v=GGML_TYPE_Q4_0;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.n_threads=cp.n_threads_batch=4;
    auto batch=llama_batch_init(2,0,1);
    struct release_batch {llama_batch & batch;~release_batch() {llama_batch_free(batch);}} release{batch};
    const auto input=[&](uint32_t pos) {
        batch.n_tokens=2;
        for (int32_t slot=0;slot<2;++slot) {
            batch.token[slot]=llama_token(10+slot*10+pos);batch.pos[slot]=pos;
            batch.n_seq_id[slot]=1;batch.seq_id[slot][0]=slot;batch.logits[slot]=1;
        }
    };
    const auto vocabulary=model->vocab.n_tokens();
    std::vector<std::vector<float>> reference;
    {
        llama_context native(*model,cp);
        for (uint32_t pos=0;pos<2;++pos) {
            input(pos);if (!t.assert_equal(0,native.decode(batch))) return;
            for (int32_t slot=0;slot<2;++slot) {
                const auto * logits=llama_get_logits_ith(&native,slot);
                if (!t.assert_true(logits!=nullptr)) return;
                reference.emplace_back(logits,logits+vocabulary);
            }
        }
    }
    cp.shared_device_memory_bytes=size_t(1024)*1024*1024;
    auto group=llama_context_group::create(*model,cp);
    if (!t.assert_true(group!=nullptr)) return;
    ggml_backend_buffer_ptr parent;
    uint64_t revision=0;
    for (auto status : {GGML_STATUS_ABORTED,GGML_STATUS_ALLOC_FAILED,GGML_STATUS_FAILED}) for (bool queued : {false,true}) {
        const auto activate=[&] {
            llama_kv_stream_selection selection;
            if (!group->reset_request(0) || !group->reset_request(1) ||
                    llama_kv_stream_select({4096,2},{{0,0},{1,0}},selection)!=llama_kv_stream_selection_status::success) return false;
            input(0);
            return group->activate(selection,group->source_handles(),batch,llama_memory_work_phase::target_verify,1,++revision);
        };
        if (!t.assert_true(activate())) return;
        auto * context=group->context();
        if (!parent) parent.reset(ggml_backend_buffer_retain(context->group_parent()));
        if (!t.assert_true(context->group_parent()==parent.get() && context->decode(batch)==0)) return;
        std::array<uint64_t,2> generations{};
        std::array<std::vector<uint8_t>,2> prefix;
        const auto append_prefix=[](const llama_kv_stream_host & host,std::vector<uint8_t> & bytes) {
            for (uint32_t layer=0;layer<host.config().layers;++layer) {
                llama_kv_stream_host_layer native;
                if (!host.layer(layer,native)) return false;
                for (bool value : {false,true}) {
                    const auto * first=static_cast<const uint8_t *>(value ? native.v : native.k);
                    const size_t count=value ? host.layout().v_token_bytes : host.layout().k_token_bytes;
                    bytes.insert(bytes.end(),first,first+count);
                }
            }
            return true;
        };
        for (uint32_t slot=0;slot<2;++slot) {
            const auto * logits=llama_get_logits_ith(context,int32_t(slot));
            if (!t.assert_true(logits && !std::memcmp(logits,reference[slot].data(),vocabulary*sizeof(float)))) return;
            std::shared_ptr<llama_kv_stream_logical_cache> cache;
            if (!t.assert_true(group->source_table()->get(group->source_handles()[slot],0,cache) && cache->tokens()==1)) return;
            generations[slot]=cache->content()->generation();
            if (!t.assert_true(append_prefix(*cache->host(),prefix[slot]))) return;
        }
        input(1);
        {
            dflash_compute_failure fault(context,status,1,queued);
            if (!t.assert_true(context->decode(batch)!=0 && fault.calls>=1 && (!queued || fault.queued>=1))) return;
        }
        t.assert_true("failed native current state cannot be published as a parked checkpoint",!group->park());
        if (!t.assert_true(group->discard() && !group->context())) return;
        for (uint32_t slot=0;slot<2;++slot) {
            std::shared_ptr<llama_kv_stream_logical_cache> cache;
            if (!t.assert_true(group->source_table()->get(group->source_handles()[slot],0,cache) && cache->ready())) return;
            t.assert_equal(size_t(1),cache->tokens());
            // Cancellation invalidates pending snapshots without changing the retained bytes.
            t.assert_equal(generations[slot]+1,cache->content()->generation());
            t.assert_equal(cache->content()->generation(),cache->identity().generation);
            std::vector<uint8_t> retained;
            if (!t.assert_true(append_prefix(*cache->host(),retained) && retained==prefix[slot])) return;
        }
        if (!t.assert_true(activate() && group->context()->group_parent()==parent.get())) return;
        for (uint32_t pos=0;pos<2;++pos) {
            input(pos);if (!t.assert_equal(0,group->context()->decode(batch))) return;
            for (int32_t slot=0;slot<2;++slot) {
                const auto * logits=llama_get_logits_ith(group->context(),slot);
                if (!t.assert_true(logits && !std::memcmp(logits,reference[pos*2+slot].data(),vocabulary*sizeof(float)))) return;
            }
        }
        if (!t.assert_true(group->park() && group->discard())) return;
    }
}

static void test_measurement_context(testing & t, llama_model * model) {
    if (!t.assert_true(model && !model->hparams.no_alloc)) return;
    const auto params_before=model->hparams.no_alloc;
    for (uint32_t capacity : {4096u,131072u}) for (uint32_t selected : {1u,2u}) {
        auto cp=llama_context_default_params();
        cp.n_ctx=capacity; cp.n_batch=cp.n_ubatch=256; cp.n_seq_max=selected;
        cp.kv_unified=true; cp.swa_full=false; cp.n_rs_seq=0;
        cp.type_k=GGML_TYPE_Q8_0; cp.type_v=GGML_TYPE_Q4_0;
        cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
        llama_context context(*model,cp,llama_context_mode::measurement);
        t.assert_true(context.is_measurement_only());
        t.assert_equal(params_before,model->hparams.no_alloc);
        auto * memory=context.get_memory();
        if (!t.assert_true(memory != nullptr)) return;
        for (const auto & entry : memory->memory_breakdown()) {
            t.assert_equal(size_t(0),entry.second);
            auto * device=ggml_backend_buft_get_device(entry.first);
            if (device && ggml_backend_dev_type(device)==GGML_BACKEND_DEVICE_TYPE_GPU &&
                    !t.assert_true("cache metadata uses the arena's device-local class",
                        entry.first==llama_kv_stream_device_buffer_type(device))) return;
        }
        const auto resources=context.get_kv_resources();
        if (resources.bounded && !t.assert_true("unallocated bounded metadata stays pending",resources.bounded->bounded_storage_pending())) return;
        if (resources.recurrent) t.assert_true(resources.recurrent->current_storage_pending());
        auto full=memory->init_full();
        if (!t.assert_true(full != nullptr)) return;
        auto * sched=context.get_sched();
        for (int i=0;i<ggml_backend_sched_get_n_backends(sched);++i) {
            auto * device=ggml_backend_get_device(ggml_backend_sched_get_backend(sched,i));
            if (ggml_backend_dev_type(device)==GGML_BACKEND_DEVICE_TYPE_GPU &&
                    !t.assert_true("measure the arena's device-local buffer class",
                        ggml_backend_sched_get_buffer_type(sched,ggml_backend_sched_get_backend(sched,i))==llama_kv_stream_device_buffer_type(device))) return;
        }
        std::vector<size_t> sizes(ggml_backend_sched_get_n_backends(sched),SIZE_MAX);
        auto * graph=context.graph_reserve(selected*4,selected,selected*4,full.get(),true,sizes.data());
        if (!t.assert_true(graph != nullptr)) return;
        t.assert_true(std::any_of(sizes.begin(),sizes.end(),[](size_t bytes) {return bytes && bytes != SIZE_MAX;}));
        for (int i=0;i<ggml_backend_sched_get_n_backends(sched);++i)
            t.assert_equal(size_t(0),ggml_backend_sched_get_buffer_size(sched,ggml_backend_sched_get_backend(sched,i)));
        t.assert_true("measurement cannot allocate graph storage",context.graph_reserve(selected,selected,selected,full.get()) == nullptr);
        t.assert_equal(GGML_STATUS_FAILED,context.graph_compute(graph,false));
        llama_token token=1;
        t.assert_equal(-1,context.decode(llama_batch_get_one(&token,1)));
        t.assert_equal(-1,context.encode(llama_batch_get_one(&token,1)));
        t.assert_equal(params_before,model->hparams.no_alloc);
        const auto before=memory->memory_breakdown();
        std::vector<size_t> native_bytes,managed_bytes;
        t.assert_true("native size measurement completes",context.measure_workspace(selected*4,selected,selected*4,false,native_bytes));
        t.assert_true("managed size measurement completes",context.measure_workspace(selected*4,selected,selected*4,true,managed_bytes));
        if (!t.assert_equal(native_bytes.size(),managed_bytes.size())) return;
        bool reduced=false;
        for (size_t i=0;i<managed_bytes.size();++i) {
            t.out << "measured capacity=" << capacity << " selected=" << selected << " backend=" << i << " native=" << native_bytes[i] << " managed=" << managed_bytes[i] << std::endl;
            t.assert_true("managed scratch does not exceed the native quote",managed_bytes[i]<=native_bytes[i]);
            reduced=reduced || managed_bytes[i]<native_bytes[i];
        }
        // At short histories the native local attention can already set the graph peak.
        if (capacity == 131072) t.assert_true("long-history managed attention scratch is not counted twice",reduced);
        t.assert_true(before == memory->memory_breakdown());
        const auto saved=managed_bytes;
        t.assert_true(!context.measure_workspace(3,2,3,true,managed_bytes));
        t.assert_true(saved == managed_bytes);
        t.assert_true(!context.measure_workspace(0,1,1,true,managed_bytes));
        t.assert_true(!context.measure_workspace(1,0,1,true,managed_bytes));
        t.assert_true(!context.measure_workspace(257,1,1,true,managed_bytes));
        t.assert_true(!context.measure_workspace(1,1,2,true,managed_bytes));
        t.assert_true(saved == managed_bytes);
    }
}

static void test_native_measurement_batches(testing & t,llama_model * model) {
    if (!t.assert_true(model != nullptr)) return;
    auto cp=llama_context_default_params();
    cp.n_ctx=4096; cp.n_batch=cp.n_ubatch=256; cp.n_seq_max=2;
    cp.kv_unified=true; cp.swa_full=false; cp.type_k=GGML_TYPE_Q8_0; cp.type_v=GGML_TYPE_Q4_0;
    cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context context(*model,cp,llama_context_mode::measurement);
    for (auto counts : {std::pair<int,int>{1,3},{2,4},{4,4},{8,8}}) for (int outputs : {0,1,2}) {
        auto batch=llama_batch_init(counts.first+counts.second,0,1);
        struct batch_guard {llama_batch & batch;~batch_guard() {llama_batch_free(batch);}} release{batch};
        batch.n_tokens=counts.first+counts.second;
        for (int i=0;i<batch.n_tokens;++i) {
            const bool second=i>=counts.first;
            batch.token[i]=100+i; batch.pos[i]=second ? 17+i-counts.first : 100+i;
            batch.n_seq_id[i]=1; batch.seq_id[i][0]=second ? 1 : 0;
            batch.logits[i]=outputs==2 || (outputs==1 && (i==counts.first-1 || i==batch.n_tokens-1));
        }
        llama_batch_compat compat(&context,batch);
        llama_batch_allocr allocator(model->hparams.n_pos_per_embd());
        if (!t.assert_true(allocator.init(*compat.batch_ext,model->vocab,false))) return;
        size_t tokens=0,batches=0;
        while (true) {
            auto native=model->arch == LLM_ARCH_QWEN35 ? allocator.split_equal(256,false,0) : allocator.split_simple(256);
            if (!native.n_tokens) break;
            const auto positions=native.data->pos,ids=native.data->seq_id_data;
            const auto flags=native.data->output;
            std::vector<size_t> measured;
            if (!t.assert_true("native ragged topology is measured without padded fake queries",
                    context.measure_workspace(native,true,measured))) return;
            t.assert_true(std::any_of(measured.begin(),measured.end(),[](size_t bytes) {return bytes != 0;}));
            t.assert_true(positions==native.data->pos && ids==native.data->seq_id_data && flags==native.data->output);
            t.out << "native measurement members=" << counts.first << '+' << counts.second << " output_mode=" << outputs
                << " rows=" << native.n_tokens << " sequence_sets=" << native.n_seqs << " unique=" << native.n_seqs_unq << std::endl;
            tokens+=native.n_tokens; ++batches;
        }
        t.assert_equal(size_t(batch.n_tokens),tokens);
        t.assert_true(batches != 0);
    }
}

static void test_rollback_measurement_batches(testing & t,llama_model * model) {
    for (uint32_t depth : {3u,7u}) for (uint32_t selected : {1u,2u}) {
        auto cp=llama_context_default_params();
        cp.n_ctx=4096; cp.n_batch=cp.n_ubatch=256; cp.n_seq_max=selected; cp.n_rs_seq=depth;
        cp.kv_unified=true; cp.swa_full=false; cp.type_k=GGML_TYPE_Q8_0; cp.type_v=GGML_TYPE_Q4_0;
        cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
        std::unique_ptr<llama_context> context;
        try {context=std::make_unique<llama_context>(*model,cp,llama_context_mode::measurement);}
        catch (const std::exception & error) {
            t.out << error.what() << std::endl;
            t.assert_true("staged rollback graph can be measured before payload allocation",false); return;
        }
        auto * recurrent=context->get_kv_resources().recurrent;
        const auto * bank=recurrent ? recurrent->spill_bank() : nullptr;
        if (!t.assert_true(bank && bank->staged_publication_enabled())) return;
        t.assert_equal(size_t(156893184)*selected*depth,bank->host_bytes());
        t.assert_equal(size_t(0),bank->allocated_host_bytes());
        t.assert_equal(size_t(3268608)*2*selected*depth,bank->staged_device_bytes());
        const uint32_t maximum=depth+1;
        for (auto counts : {std::pair<uint32_t,uint32_t>{1,maximum},{2,maximum},{maximum,maximum}}) {
            const auto total=counts.first+(selected==2 ? counts.second : 0);
            auto batch=llama_batch_init(int32_t(total),0,1);
            struct batch_guard {llama_batch & batch;~batch_guard() {llama_batch_free(batch);}} release{batch};
            batch.n_tokens=int32_t(total);
            for (uint32_t row=0;row<total;++row) {
                const bool second=selected==2 && row>=counts.first;
                batch.token[row]=100+int32_t(row); batch.pos[row]=int32_t(second ? row-counts.first : row);
                batch.n_seq_id[row]=1; batch.seq_id[row][0]=second ? 1 : 0; batch.logits[row]=1;
            }
            llama_batch_compat compat(context.get(),batch);
            llama_batch_allocr allocator(model->hparams.n_pos_per_embd());
            if (!t.assert_true(allocator.init(*compat.batch_ext,model->vocab,false))) return;
            size_t measured=0;
            while (true) {
                auto native=allocator.split_equal(256,false,depth+1);
                if (!native.n_tokens) break;
                std::vector<size_t> bytes;
                if (!t.assert_true("staged publication uses the native active sequence count",
                        context->measure_workspace(native,true,bytes))) return;
                t.assert_equal(size_t(0),recurrent->spill_bank()->allocated_host_bytes());
                t.out << "rollback measurement depth=" << depth << " selected=" << selected << " rows=" << native.n_tokens
                    << " native_sequences=" << native.n_seqs << " stage=" << recurrent->spill_bank()->staged_device_bytes() << std::endl;
                measured+=native.n_tokens;
            }
            t.assert_equal(size_t(total),measured);
        }
    }
}

static void test_target_workload_quote(testing & t,llama_model * model) {
    auto cp=llama_context_default_params();
    cp.n_ctx=4096; cp.n_batch=cp.n_ubatch=256; cp.n_seq_max=2;
    cp.kv_unified=true; cp.swa_full=false; cp.type_k=GGML_TYPE_Q8_0; cp.type_v=GGML_TYPE_Q4_0;
    cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context context(*model,cp,llama_context_mode::measurement);
    auto batch=llama_batch_init(6,0,1);
    struct batch_guard {llama_batch & batch;~batch_guard() {llama_batch_free(batch);}} release{batch};
    batch.n_tokens=6;
    for (int row=0;row<6;++row) {
        batch.token[row]=100+row; batch.pos[row]=row<2 ? row+10 : row+20;
        batch.n_seq_id[row]=1; batch.seq_id[row][0]=row<2 ? 0 : 1; batch.logits[row]=1;
    }
    const std::vector<llama_memory_phase_cache_identity> sources={{4101,1},{4102,2}};
    llama_context_group_workload_quote quote;
    if (!t.assert_true(context.measure_target_workload(batch,sources,llama_memory_work_phase::target_verify,91,quote))) return;
    const auto & measured=quote.measurement;
    t.assert_equal(uint64_t(91),measured.workload.id);
    t.assert_equal(uint32_t(2),measured.workload.selected_count);
    t.assert_equal(size_t(2),measured.workload.members.size());
    t.assert_equal(uint32_t(2),measured.workload.members[0].input_rows);
    t.assert_equal(uint32_t(4),measured.workload.members[1].input_rows);
    t.assert_equal(measured.workload.native_batches.size(),measured.graph_bytes.size());
    t.assert_true(quote.writer_bytes && quote.attention_bytes && quote.encoded_stage_bytes && quote.pool_minimum_bytes);
    t.assert_equal(quote.writer_bytes+quote.attention_bytes+quote.encoded_stage_bytes+quote.publication_bytes+
        quote.pool_minimum_bytes+quote.alignment_slack_bytes,measured.live_device_bytes);
    size_t rows=0;
    for (const auto & native : measured.workload.native_batches)
        for (const auto & range : native) rows+=range.rows;
    t.assert_equal(size_t(6),rows);
    if (model->arch==LLM_ARCH_QWEN35) {
        t.assert_equal(size_t(2),measured.workload.native_batches.size());
        t.assert_equal(uint32_t(2),measured.workload.native_batches[1][0].first_row);
    } else t.assert_equal(size_t(1),measured.workload.native_batches.size());
    auto bad=sources; bad[1]=bad[0];
    t.assert_true(!context.measure_target_workload(batch,bad,llama_memory_work_phase::target_verify,92,quote));
    t.assert_equal(uint64_t(91),quote.measurement.workload.id);
    t.assert_true(!context.measure_target_workload(batch,sources,llama_memory_work_phase::none,92,quote));
    t.assert_true(!context.measure_target_workload(batch,sources,llama_memory_work_phase::target_verify,0,quote));
    t.assert_equal(uint64_t(91),quote.measurement.workload.id);
    llama_context_group_plan plan;
    const auto planned=context.quote_target_group(batch,sources,{4201,1},llama_memory_work_phase::target_verify,91,SIZE_MAX,plan);
    if (!t.assert_true(planned.status==llama_context_memory_create_status::success)) return;
    t.assert_equal(plan.budget.parent_minimum_bytes,planned.required_bytes);
    const auto minimum=planned.required_bytes;
    t.assert_true(minimum>0);
    const auto rejected=context.quote_target_group(batch,sources,{4201,1},llama_memory_work_phase::target_verify,91,minimum-1,plan);
    t.assert_true(rejected.status==llama_context_memory_create_status::insufficient_budget);
    t.assert_equal(minimum,rejected.required_bytes);
    t.assert_equal(size_t(1),rejected.additional_bytes);
    t.assert_equal(minimum,plan.budget.parent_minimum_bytes);
    const auto resources=context.get_kv_resources();
    if (resources.recurrent) t.assert_true(resources.recurrent->current_storage_pending());
    if (resources.bounded) t.assert_true(resources.bounded->bounded_storage_pending());
    t.out << "group minimum current=" << plan.current_state_bytes << " bounded=" << plan.bounded.allocation.bytes
        << " graph=" << plan.budget.workspace_peak_bytes << " total=" << minimum << std::endl;
    auto * sched=context.get_sched();
    ggml_backend_t backend=nullptr;
    for (int i=0;i<ggml_backend_sched_get_n_backends(sched);++i) {
        auto * candidate=ggml_backend_sched_get_backend(sched,i);
        if (ggml_backend_get_device(candidate)==model->devices.front().dev) backend=candidate;
    }
    if (!t.assert_true(backend != nullptr)) return;
    const auto get_copy=reinterpret_cast<ggml_kv_stream_copy_ops_get>(ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(ggml_backend_get_device(backend)),"ggml_backend_kv_stream_copy_ops"));
    const auto * copy=get_copy ? get_copy() : nullptr;
    if (!t.assert_true(copy && copy->version>=12 && copy->prepare && copy->free_prepared)) return;
    std::vector<std::shared_ptr<void>> prepared;
    for (size_t slot=0;slot<sources.size();++slot) {
        auto * bank=copy->prepare(backend,minimum/plan.target.encoded_stage_bytes,true);
        if (!t.assert_true(bank != nullptr)) return;
        prepared.emplace_back(bank,copy->free_prepared);
    }
    llama_compute_arena_ptr parent(ggml_backend_memory_arena_new(plan.target.buffer_type,minimum));
    if (!t.assert_true(parent && ggml_backend_memory_arena_begin(parent.get(),0))) return;
    size_t offset=plan.current_state_bytes;
    if (offset && !t.assert_true(ggml_backend_memory_arena_reserve_at(parent.get(),101,0,offset,
            plan.target.measurement.alignment,GGML_BACKEND_MEMORY_REGION_PERSISTENT,nullptr))) return;
    if (plan.bounded.allocation.bytes) {
        const auto alignment=plan.bounded.allocation.alignment;
        offset=(offset+alignment-1)&~(alignment-1);
        if (!t.assert_true(ggml_backend_memory_arena_reserve_at(parent.get(),102,offset,plan.bounded.allocation.bytes,
                alignment,GGML_BACKEND_MEMORY_REGION_PERSISTENT,nullptr))) return;
        offset+=plan.bounded.allocation.bytes;
    }
    const auto alignment=plan.target.measurement.alignment;
    offset=(offset+alignment-1)&~(alignment-1);
    if (!t.assert_equal(minimum,offset+plan.budget.workspace_peak_bytes)) return;
    if (!t.assert_true(ggml_backend_memory_arena_reserve_at(parent.get(),103,offset,plan.budget.workspace_peak_bytes,
            alignment,0,nullptr) && ggml_backend_memory_arena_commit(parent.get()))) return;
    using group_lease=std::unique_ptr<ggml_backend_memory_lease,decltype(&ggml_backend_memory_lease_free)>;
    group_lease current(plan.current_state_bytes ? ggml_backend_memory_arena_acquire(parent.get(),101) : nullptr,ggml_backend_memory_lease_free);
    group_lease bounded(plan.bounded.allocation.bytes ? ggml_backend_memory_arena_acquire(parent.get(),102) : nullptr,ggml_backend_memory_lease_free);
    group_lease workspace(ggml_backend_memory_arena_acquire(parent.get(),103),ggml_backend_memory_lease_free);
    llama_compute_arena_ptr transient(ggml_backend_memory_arena_new_from_buffer(ggml_backend_memory_lease_buffer(workspace.get())));
    if (!t.assert_true(transient && ggml_backend_memory_arena_begin(transient.get(),0))) return;
    size_t cursor=0;
    uint64_t region=1;
    std::vector<size_t> live={*std::max_element(plan.target.measurement.graph_bytes.begin(),plan.target.measurement.graph_bytes.end()),
        plan.target.writer_bytes,plan.target.attention_bytes,plan.target.encoded_stage_bytes,plan.target.publication_bytes};
    for (size_t source=0;source<sources.size();++source) live.push_back(plan.target.pool_minimum_bytes/sources.size());
    for (auto bytes : live) {
        if (!bytes) continue;
        cursor=(cursor+alignment-1)&~(alignment-1);
        if (!t.assert_true(ggml_backend_memory_arena_reserve_at(transient.get(),region++,cursor,bytes,alignment,0,nullptr))) return;
        cursor+=bytes;
    }
    t.assert_true(cursor<=plan.budget.workspace_peak_bytes);
    if (!t.assert_true(ggml_backend_memory_arena_commit(transient.get()))) return;
    t.assert_true(ggml_backend_buffer_get_base(ggml_backend_memory_arena_parent(transient.get()))==
        ggml_backend_buffer_get_base(ggml_backend_memory_lease_buffer(workspace.get())));
    ggml_backend_buffer_clear(ggml_backend_memory_arena_parent(parent.get()),0xa5);
    if (resources.recurrent && !t.assert_true(current && resources.recurrent->bind_current_storage(current.get()))) return;
    if (resources.bounded && !t.assert_true(bounded && resources.bounded->bind_bounded_storage(bounded.get()))) return;
    llama_context_group_plan bound_plan;
    if (!t.assert_true(context.quote_target_group(batch,sources,{4201,1},llama_memory_work_phase::target_verify,91,
            minimum,bound_plan).status==llama_context_memory_create_status::success)) return;
    t.assert_equal(plan.budget.parent_minimum_bytes,bound_plan.budget.parent_minimum_bytes);
    t.assert_true(plan.target.measurement.graph_bytes==bound_plan.target.measurement.graph_bytes);
    auto * buffer=ggml_backend_memory_lease_buffer(workspace.get());
    ggml_tensor raw{}; raw.type=GGML_TYPE_I8; raw.ne[0]=int64_t(ggml_backend_buffer_get_size(buffer));
    raw.ne[1]=raw.ne[2]=raw.ne[3]=1; raw.nb[0]=1; raw.nb[1]=raw.nb[2]=raw.nb[3]=size_t(raw.ne[0]);
    raw.buffer=buffer; raw.data=ggml_backend_buffer_get_base(buffer);
    for (size_t position : {size_t(0),size_t(raw.ne[0])-128}) {
        std::array<uint8_t,128> guard{}; ggml_backend_tensor_get(&raw,guard.data(),position,guard.size());
        t.assert_true(std::all_of(guard.begin(),guard.end(),[](uint8_t value) {return value==0xa5;}));
    }
    if (resources.bounded) t.assert_true(resources.bounded->release_bounded_storage());
    if (resources.recurrent) t.assert_true(resources.recurrent->release_current_storage());
    // Compare the catalog to native unified splitting, including reversed sequence order.
    for (bool reversed : {false,true}) for (auto counts : {std::pair<int,int>{1,3},{2,4},{4,4},{8,8}}) {
        auto reordered=llama_batch_init(counts.first+counts.second,0,1);
        batch_guard release_reordered{reordered};
        reordered.n_tokens=counts.first+counts.second;
        for (int row=0;row<reordered.n_tokens;++row) {
            const bool second=row>=counts.first;
            reordered.token[row]=100+row; reordered.pos[row]=second ? row-counts.first : row;
            reordered.n_seq_id[row]=1; reordered.seq_id[row][0]=int(second)!=int(reversed);
            reordered.logits[row]=1;
        }
        llama_batch_compat compat(&context,reordered);
        llama_batch_allocr allocator(model->hparams.n_pos_per_embd());
        if (!t.assert_true(allocator.init(*compat.batch_ext,model->vocab,false))) return;
        std::vector<std::vector<llama_memory_group_member_range>> expected;
        std::array<uint32_t,2> consumed{};
        while (true) {
            auto native=model->arch==LLM_ARCH_QWEN35 ? allocator.split_equal(cp.n_ubatch,false,0) : allocator.split_simple(cp.n_ubatch);
            if (!native.n_tokens) break;
            std::vector<llama_memory_group_member_range> ranges;
            for (uint32_t row=0;row<native.n_tokens;++row) {
                const auto member=uint32_t(native.seq_id[row][0]);
                if (ranges.empty() || ranges.back().member!=member) ranges.push_back({member,consumed[member],0,0});
                ++ranges.back().rows; ++ranges.back().output_rows; ++consumed[member];
            }
            expected.push_back(std::move(ranges));
        }
        llama_context_group_workload_quote ordered_quote;
        if (!t.assert_true(context.measure_target_workload(reordered,sources,llama_memory_work_phase::target_verify,93,ordered_quote))) return;
        t.out << "native catalog reversed=" << reversed << " members=" << counts.first << '+' << counts.second << std::endl;
        if (!t.assert_true("catalog retains native unified query order and microbatch boundaries",
                expected==ordered_quote.measurement.workload.native_batches)) return;
    }

}

static void test_gemma_deferred_local(testing & t, llama_model * model,uint32_t selected=1,uint32_t logical=0) {
    if (!logical) logical=selected;
    const bool projected=logical!=selected;
    const llama_seq_id sequence=projected ? 1 : 0;
    if (!t.assert_true(model && model->arch == LLM_ARCH_GEMMA4)) return;
    auto * dev=ggml_backend_dev_by_name("CUDA0");
    if (!t.assert_true(dev != nullptr)) return;
    ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
    llama_memory_params params{};
    params.type_k=GGML_TYPE_Q8_0; params.type_v=GGML_TYPE_Q4_0;
    params.kv_stream_pool_bytes=selected==1 && !projected ? 64u*1024u*1024u : 0;
    if (projected) {params.measure_only=true;params.kv_stream_selected_sequences={sequence};}
    params.kv_stream_max_rows=256; params.kv_stream_max_decode_rows=4;
    params.kv_stream_backend=backend.get(); params.defer_bounded_cache=true;
    const auto no_reuse=[](int32_t) { return int32_t(-1); };
    llama_kv_cache_iswa cache(*model,params.type_k,params.type_v,false,true,false,true,
        8192,logical,256,256,nullptr,nullptr,no_reuse,nullptr,&params);
    auto * local=cache.get_swa();
    if (!t.assert_true(local->bounded_storage_pending() && bool(cache.get_base()->get_kv_stream())==(selected==1 && !projected))) return;
    auto * placeholder=local->get_k_storage(0);
    t.assert_true(placeholder && !placeholder->data && ggml_backend_buffer_get_size(placeholder->buffer) == 0);
    local->clear(true);
    t.assert_true(local->bounded_storage_pending() && local->prepare({}).empty() &&
        local->init_update(nullptr,false)->get_status() == LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    for (auto id : local->get_layer_ids()) t.assert_true(!local->get_k_storage(int32_t(id))->data &&
        !local->get_v_storage(int32_t(id))->data);
    llama_cparams cp{}; cp.n_ctx=8192; cp.n_ctx_seq=8192/logical; cp.n_ubatch=256; cp.n_seq_max=logical; cp.flash_attn=true; cp.kv_unified=true;
    auto * buft=llama_kv_stream_device_buffer_type(dev);
    llama_memory_bounded_kv_layout quote;
    if (!t.assert_true(llama_kv_cache_iswa::bounded_storage_requirements(model->hparams,params,cp,buft,{608,1},quote))) return;
    using arena_ptr=std::unique_ptr<ggml_backend_memory_arena,decltype(&ggml_backend_memory_arena_free)>;
    using lease_ptr=std::unique_ptr<ggml_backend_memory_lease,decltype(&ggml_backend_memory_lease_free)>;
    const auto bytes=quote.allocation.bytes, alignment=quote.allocation.alignment;
    arena_ptr arena(ggml_backend_memory_arena_new(buft,bytes+2*alignment),ggml_backend_memory_arena_free);
    if (!t.assert_true(arena && ggml_backend_memory_arena_begin(arena.get(),0) &&
            ggml_backend_memory_arena_reserve_at(arena.get(),78,alignment,bytes,alignment,
                GGML_BACKEND_MEMORY_REGION_PERSISTENT,nullptr) && ggml_backend_memory_arena_commit(arena.get()))) return;
    auto * parent=ggml_backend_memory_arena_parent(arena.get());
    ggml_backend_buffer_clear(parent,0x5a);
    ggml_backend_buffer_ptr short_buffer(ggml_backend_buffer_view(parent,alignment,bytes-1));
    arena_ptr short_arena(ggml_backend_memory_arena_new_from_buffer(short_buffer.get()),ggml_backend_memory_arena_free);
    if (!t.assert_true(short_arena && ggml_backend_memory_arena_begin(short_arena.get(),0) &&
            ggml_backend_memory_arena_reserve_at(short_arena.get(),78,0,bytes-1,alignment,
                GGML_BACKEND_MEMORY_REGION_PERSISTENT,nullptr) && ggml_backend_memory_arena_commit(short_arena.get()))) return;
    lease_ptr short_lease(ggml_backend_memory_arena_acquire(short_arena.get(),78),ggml_backend_memory_lease_free);
    t.assert_true(!local->bind_bounded_storage(short_lease.get()));
    t.assert_true(local->bounded_storage_pending() && local->get_k_storage(0) == placeholder && !placeholder->data);
    lease_ptr lease(ggml_backend_memory_arena_acquire(arena.get(),78),ggml_backend_memory_lease_free);
    if (!t.assert_true(local->bind_bounded_storage(lease.get()))) return;
    t.assert_true(!local->bounded_storage_pending() && placeholder->data == nullptr);
    const auto base=uintptr_t(ggml_backend_buffer_get_base(parent))+alignment;
    for (const auto & layer : quote.layers) {
        t.assert_equal(base+layer.k_offset,uintptr_t(local->get_k_storage(int32_t(layer.id))->data));
        t.assert_equal(base+layer.v_offset,uintptr_t(local->get_v_storage(int32_t(layer.id))->data));
    }
    std::vector<uint8_t> marker(256,0x39), after(marker.size());
    ggml_backend_tensor_set(local->get_k_storage(0),marker.data(),0,marker.size());
    t.assert_true(local->bind_bounded_storage(lease.get()));
    ggml_backend_tensor_get(local->get_k_storage(0),after.data(),0,after.size());
    t.assert_true(marker == after);
    ggml_tensor raw{}; raw.type=GGML_TYPE_I8;
    raw.ne[0]=int64_t(bytes+2*alignment); raw.ne[1]=raw.ne[2]=raw.ne[3]=1;
    raw.nb[0]=1; raw.nb[1]=raw.nb[2]=raw.nb[3]=size_t(raw.ne[0]);
    raw.buffer=parent; raw.data=ggml_backend_buffer_get_base(parent);
    std::vector<uint8_t> guard(alignment);
    ggml_backend_tensor_get(&raw,guard.data(),0,guard.size());
    t.assert_true(std::all_of(guard.begin(),guard.end(),[](uint8_t value) { return value == 0x5a; }));
    ggml_backend_tensor_get(&raw,guard.data(),alignment+bytes,guard.size());
    t.assert_true(std::all_of(guard.begin(),guard.end(),[](uint8_t value) { return value == 0x5a; }));
    llama_memory_bounded_kv_layout bound;
    t.assert_true(local->describe_bounded_storage(608,1,8192,256,bound));
    t.assert_equal(bytes,bound.allocation.bytes);
    t.assert_equal(selected,bound.cache.sequence_capacity);
    if (projected) {
        t.assert_equal(size_t(425984000),bytes);
        t.assert_equal(uint32_t(1280),local->get_size());
        llama_batch_allocr allocator(1);auto foreign=allocator.ubatch_reserve(1,1);llama_seq_id id=0;
        foreign.pos[0]=0;foreign.n_seq_id[0]=1;foreign.seq_id[0]=&id;
        if (!t.assert_true("unchosen logical slot cannot prepare a selected SWA bank",local->prepare({foreign}).empty())) return;
        t.assert_equal(uint32_t(0),local->get_cells(0).get_used());
    }
    auto & cells=const_cast<llama_kv_cells &>(local->get_cells(0));
    cells.pos_set(0,17);cells.seq_add(0,sequence);
    image_writer image;
    local->state_write(image,sequence,0);
    const auto full_generation=(cache.get_base()->get_kv_stream() ? cache.get_base()->get_kv_stream()->tokens() : 0);
    if (!t.assert_true(local->release_bounded_storage())) return;
    t.assert_true(local->bounded_storage_pending() && local->get_cells(0).get_used()==0);
    t.assert_true(!local->get_k_storage(0)->data && !local->get_v_storage(0)->data);
    t.assert_true(local->release_bounded_storage());
    lease.reset();short_lease.reset();short_arena.reset();short_buffer.reset();
    if (!t.assert_true(ggml_backend_memory_arena_begin(arena.get(),0) &&
            ggml_backend_memory_arena_reserve_at(arena.get(),79,alignment,bytes,alignment,
                GGML_BACKEND_MEMORY_REGION_PERSISTENT,nullptr) && ggml_backend_memory_arena_commit(arena.get()))) return;
    lease.reset(ggml_backend_memory_arena_acquire(arena.get(),79));
    if (!t.assert_true(bool(lease))) return;
    ggml_backend_buffer_clear(ggml_backend_memory_lease_buffer(lease.get()),0xa5);
    if (!t.assert_true(local->bind_bounded_storage(lease.get()))) return;
    image_reader restored(image.bytes);
    local->state_read(restored,sequence,0);
    t.assert_equal(image.bytes.size(),restored.n_bytes());
    t.assert_equal(llama_pos(17),local->seq_pos_max(sequence));
    t.assert_equal(uint32_t(1),local->get_cells(0).get_used());
    ggml_backend_tensor_get(local->get_k_storage(0),after.data(),0,after.size());
    t.assert_true(marker==after);
    t.assert_equal(full_generation,(cache.get_base()->get_kv_stream() ? cache.get_base()->get_kv_stream()->tokens() : 0));
    cells.pos_set(511,18);cells.seq_add(511,sequence);cells.pos_set(1250,19);cells.seq_add(1250,sequence);
    cells.ext_set(511,{31,12,4097});cells.ext_set(1250,{5,7,LLAMA_TOKEN_NULL});
    if (selected==2) {
        cells.pos_set(1800,513);cells.seq_add(1800,1);cells.ext_set(1800,{23,41,199});
        t.assert_equal(llama_pos(513),local->seq_pos_max(1));
    }
    const auto physical_cells=local->get_cells(0);
    kv_stream_bit_oracle physical;
    for (const auto & layer : quote.layers) for (bool value : {false,true}) {
        auto * tensor=value ? local->get_v_storage(int32_t(layer.id)) : local->get_k_storage(int32_t(layer.id));
        physical.tensor("SWA/"+std::to_string(layer.id)+(value ? "/V" : "/K"),tensor);
    }
    {
        native_state_copy_fault fault(local->get_k_storage(0)->buffer,false);
        t.assert_true(!local->park_bounded_storage());
        t.assert_equal(size_t(2),fault.calls);t.assert_true(!local->bounded_storage_pending());
    }
    if (!t.assert_true(local->park_bounded_storage())) return;
    t.assert_true(local->bounded_storage_pending() && !local->get_k_storage(0)->data);
    t.assert_equal(physical_cells.get_used(),local->get_cells(0).get_used());
    native_state_export_probe probe;
    bool rejected=false;
    try { local->state_write(probe,sequence,0); } catch (const std::runtime_error &) { rejected=true; }
    t.assert_true(rejected && probe.calls==0);
    lease.reset();
    if (!t.assert_true(ggml_backend_memory_arena_begin(arena.get(),0) &&
            ggml_backend_memory_arena_reserve_at(arena.get(),89,alignment,bytes,alignment,
                GGML_BACKEND_MEMORY_REGION_PERSISTENT,nullptr) && ggml_backend_memory_arena_commit(arena.get()))) return;
    lease.reset(ggml_backend_memory_arena_acquire(arena.get(),89));
    if (!t.assert_true(bool(lease))) return;
    ggml_backend_buffer_clear(ggml_backend_memory_lease_buffer(lease.get()),0x6b);
    {
        native_state_copy_fault fault(ggml_backend_memory_lease_buffer(lease.get()),true);
        t.assert_true(!local->restore_bounded_storage(lease.get()));
        t.assert_equal(size_t(2),fault.calls);
        t.assert_true(local->bounded_storage_pending() && !local->get_k_storage(0)->data);
    }
    t.assert_true(!local->bind_bounded_storage(lease.get()));
    if (!t.assert_true(local->restore_bounded_storage(lease.get()))) return;
    physical.compare();
    for (const auto & layer : quote.layers) for (bool value : {false,true}) {
        auto * tensor=value ? local->get_v_storage(int32_t(layer.id)) : local->get_k_storage(int32_t(layer.id));
        physical.tensor("SWA/"+std::to_string(layer.id)+(value ? "/V" : "/K"),tensor);
    }
    physical.finish();
    for (uint32_t row=0;row<quote.allocation.capacity_rows;++row) {
        const auto & restored_cells=local->get_cells(0);
        if (!t.assert_equal(physical_cells.is_empty(row),restored_cells.is_empty(row))) return;
        if (!physical_cells.is_empty(row)) {
            t.assert_equal(physical_cells.pos_get(row),restored_cells.pos_get(row));
            t.assert_true(physical_cells.seq_get_all(row)==restored_cells.seq_get_all(row));
            const auto & expected_ext=physical_cells.ext_get(row);
            const auto & actual_ext=restored_cells.ext_get(row);
            t.assert_equal(expected_ext.x,actual_ext.x);t.assert_equal(expected_ext.y,actual_ext.y);
            t.assert_equal(expected_ext.tok,actual_ext.tok);
        }
    }
    t.assert_equal(physical_cells.get_used(),local->get_cells(0).get_used());
    t.assert_equal(full_generation,(cache.get_base()->get_kv_stream() ? cache.get_base()->get_kv_stream()->tokens() : 0));
    if (!t.assert_true(local->park_bounded_storage())) return;
    local->clear(true);
    if (!t.assert_true(local->restore_bounded_storage(lease.get()))) return;
    t.assert_equal(uint32_t(0),local->get_cells(0).get_used());
    for (const auto & layer : quote.layers) for (bool value : {false,true}) {
        auto * tensor=value ? local->get_v_storage(int32_t(layer.id)) : local->get_k_storage(int32_t(layer.id));
        std::vector<uint8_t> zero(ggml_nbytes(tensor));
        ggml_backend_tensor_get(tensor,zero.data(),0,zero.size());
        t.assert_true(std::all_of(zero.begin(),zero.end(),[](uint8_t byte) {return byte==0;}));
    }
    ggml_backend_tensor_get(&raw,guard.data(),0,guard.size());
    t.assert_true(std::all_of(guard.begin(),guard.end(),[](uint8_t byte) {return byte==0x5a;}));
    ggml_backend_tensor_get(&raw,guard.data(),alignment+bytes,guard.size());
    t.assert_true(std::all_of(guard.begin(),guard.end(),[](uint8_t byte) {return byte==0x5a;}));
}

static void test_gemma_stream_groups(testing & t, llama_model * model) {
    if (!t.assert_true(model && model->arch == LLM_ARCH_GEMMA4)) return;
    auto * dev=ggml_backend_dev_by_name("CUDA0");
    if (!t.assert_true(dev != nullptr)) return;
    ggml_backend_ptr backend(ggml_backend_dev_init(dev,nullptr));
    llama_memory_params params{};
    params.type_k=GGML_TYPE_Q8_0; params.type_v=GGML_TYPE_Q4_0;
    params.kv_stream_pool_bytes=64u*1024u*1024u;
    params.kv_stream_max_rows=256; params.kv_stream_max_decode_rows=4;
    params.kv_stream_backend=backend.get();
    const auto no_reuse=[](int32_t) { return int32_t(-1); };
    try {
        llama_kv_cache_iswa cache(*model,params.type_k,params.type_v,false,true,false,true,
            8192,1,256,256,nullptr,nullptr,no_reuse,nullptr,&params);
        auto * base=cache.get_base(); auto * local=cache.get_swa();
        auto * stream=base->get_kv_stream();
        if (!t.assert_true(stream && !local->get_kv_stream())) return;
        const auto resources=llama_memory_kv_resources_get(&cache);
        t.assert_true(resources.full == base && resources.bounded == local && !resources.recurrent);
        const auto & host=stream->host()->config();
        t.assert_equal(10u,host.layers);
        t.assert_equal(size_t(8192),host.context_tokens);
        t.assert_equal(int64_t(512),host.shape.head_dim_k);
        t.assert_equal(int64_t(512),host.shape.head_dim_v);
        t.assert_equal(int64_t(4),host.shape.heads);
        t.assert_equal(size_t(272629760),stream->host()->bytes());
        const auto full=base->get_layer_ids();
        t.assert_equal(size_t(10),full.size());
        for (size_t ordinal=0;ordinal<full.size();++ordinal) {
            const uint32_t id=uint32_t(5+6*ordinal);
            t.assert_equal(id,full[ordinal]);
            llama_kv_stream_host_layer planes;
            if (!t.assert_true(stream->host()->layer(uint32_t(ordinal),planes))) return;
            t.assert_true(base->get_k_storage(int32_t(id))->data == planes.k);
            t.assert_true(base->get_v_storage(int32_t(id))->data == planes.v);
            t.assert_true(base->get_k_storage(int32_t(id))->buffer == stream->buffer());
        }
        t.assert_equal(size_t(50),local->get_layer_ids().size());
        t.assert_equal(uint32_t(1280),local->get_size());
        for (auto id : local->get_layer_ids()) {
            auto * tensor=local->get_k_storage(int32_t(id));
            t.assert_true(tensor->buffer != stream->buffer() && !ggml_backend_buffer_is_host(tensor->buffer));
        }
        t.assert_true(stream->bounded_prefill() && stream->tokens() == 0 && stream->complete());
        for (int invalid=0;invalid<3;++invalid) {
            auto bad=params;
            if (invalid == 1) bad.kv_stream_auxiliary_layers=1;
            const llama_memory_i::layer_reuse_cb reuse=invalid == 2 ?
                +[](int32_t il) { return il == 5 ? int32_t(11) : int32_t(-1); } : +[](int32_t) { return int32_t(-1); };
            bool rejected=false;
            try {
                llama_kv_cache_iswa invalid_cache(*model,bad.type_k,bad.type_v,false,true,invalid == 0,true,
                    8192,1,256,256,nullptr,nullptr,reuse,nullptr,&bad);
            } catch (const std::runtime_error & error) {
                const char * expected=invalid == 0 ? "KV streaming requires bounded SWA storage (swa_full=false)" :
                    invalid == 1 ? "Gemma shared-KV drafting does not allocate auxiliary KV histories" :
                    "KV streaming cannot alias selected layers";
                rejected=std::string(error.what()) == expected;
            }
            t.assert_true(rejected);
        }
    } catch (const std::exception & error) {
        t.out << "Gemma private streaming: " << error.what() << '\n';
        t.assert_true(false);
    }
}

static void test_gemma_assistant_source(testing & t, llama_model * model, const char * path,
        bool forced=false, bool compact=false, bool verify=false, bool quotas=false,bool vision=false,const char * image_path=nullptr,
        const char * export_native=nullptr,const char * import_native=nullptr) {
    if (!t.assert_true(model && model->arch == LLM_ARCH_GEMMA4 && path)) return;
    auto mp=llama_model_default_params(); mp.n_gpu_layers=999;
    model_ptr assistant(llama_model_load_from_file(path,mp),llama_model_free);
    if (!t.assert_true(assistant && assistant->arch == LLM_ARCH_GEMMA4_ASSISTANT)) return;
    const size_t width=model->hparams.n_embd_out();
    const size_t vocab=llama_vocab_n_tokens(llama_model_get_vocab(assistant.get()));
    if (!t.assert_true(width == assistant->hparams.n_embd_inp() && width == assistant->hparams.n_embd_out())) return;
    std::vector<std::vector<float>> logits[2],hidden_states[2];
    std::vector<llama_token> proposals[2];
    llama_context_params quota_params{};
    size_t exact_budget=0;
    std::vector<float> image;
    if (image_path) {
        if (!t.assert_true(!forced && !compact && !quotas)) return;
        std::unique_ptr<FILE,kv_stream_exact_file_free> file(std::fopen(image_path,"rb"));
        image=gemma_image_reference(file.get(),width);
    }
    const size_t image_rows=image.size()/width;
    const int text_prefix=forced ? 6143 : compact ? 2305 : 1333;
    const int prefix=text_prefix+int(image_rows)+(image_rows ? 1 : 0);
    if ((export_native || import_native) && !t.assert_true(image_rows && verify && vision && !forced && !compact && !quotas &&
            !(export_native && import_native))) return;
    auto oracle_file=[&](const char * base,bool writing) {
        std::array<std::unique_ptr<FILE,kv_stream_exact_file_free>,3> files;
        std::array<FILE *,3> handles{};
        const char * extensions[]={".input.bin",".logits.bin",".hidden.bin"};
        for (size_t i=0;i<files.size();++i) {
            const auto name=std::string(base)+extensions[i];
            files[i].reset(std::fopen(name.c_str(),writing ? "wbx" : "rb"));handles[i]=files[i].get();
        }
        gemma_assistant_oracle(handles,writing,4096,uint64_t(prefix),image,width,vocab,logits[0],hidden_states[0]);
    };
    if (import_native) {
        oracle_file(import_native,false);
        for (size_t i=0;i<3;++i) proposals[0].push_back(llama_token(std::max_element(logits[0][i].begin(),logits[0][i].end())-logits[0][i].begin()));
    }
    for (bool streamed : {false,true}) {
        if ((!streamed && import_native) || (streamed && export_native)) continue;
        auto cp=llama_context_default_params();
        cp.n_ctx=forced ? 6144 : 4096;
        cp.n_batch=cp.n_ubatch=image_rows ? uint32_t((image_rows+255)/256*256) : forced ? 1 : compact ? 32 : 256; cp.n_seq_max=1;
        if (image_rows) gemma_image_test_outputs(cp,verify);
        cp.n_threads=cp.n_threads_batch=8; cp.swa_full=false;
        cp.type_k=GGML_TYPE_Q8_0; cp.type_v=GGML_TYPE_Q4_0;
        cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.shared_device_memory_bytes=streamed ? (image_rows ? gemma_image_test_arena_bytes(gemma_test_arena_explicit,gemma_test_arena_size) : size_t(forced ? 424 : compact ? 480 : 1024)*1048576) : 0;
        cp.kv_stream_verify_max=streamed && verify ? 4 : 0;
        if (streamed && verify) for (int invalid=0;invalid<4;++invalid) {
            auto bad=cp;
            if (invalid == 0) bad.kv_stream_verify_max=5;
            if (invalid == 1) bad.n_ubatch=3;
            if (invalid == 2) bad.shared_device_memory_bytes=0;
            if (invalid == 3) bad.n_seq_max=2;
            bool rejected=false;
            try { llama_context wrong(*model,bad); }
            catch (const std::runtime_error & error) {
                rejected=std::string(error.what()) == "explicit streamed verification requires a serial Gemma4 target and 1-4 queries within n_ubatch";
            }
            if (!t.assert_true("invalid verification declaration rejects before target allocation",rejected)) return;
        }
        context_ptr target(llama_init_from_model(model,cp),llama_free);
        if (!t.assert_true(streamed ? "streamed target for actual assistant" : "native target for actual assistant",bool(target))) return;
        llama_set_embeddings_nextn(target.get(),true,false);
        const auto resources=llama_memory_kv_resources_get(llama_get_memory(target.get()));
        if (!t.assert_true(resources.full && resources.bounded && !resources.recurrent)) return;
        auto * stream=resources.full->get_kv_stream();
        if (streamed && verify) {
            t.assert_equal(uint32_t(0),target->get_cparams().n_rs_seq);
            t.assert_equal(uint32_t(4),target->get_cparams().kv_stream_decode_rows());
            t.assert_equal(uint32_t(4),stream->max_decode_rows());
        }
        auto dp=cp;
        dp.n_batch=dp.n_ubatch=1; dp.shared_device_memory_bytes=0;
        if (image_rows) gemma_image_test_outputs(dp,false);
        dp.kv_stream_verify_max=0;
        dp.ctx_type=LLAMA_CONTEXT_TYPE_MTP; dp.ctx_other=target.get();
        if (streamed && !forced) {
            const auto revision=target->get_compute_memory()->graph_binding_revision();
            for (int profile=0;profile<10;++profile) {
                auto invalid=dp;
                switch (profile) {
                    case 0: invalid.n_batch=2; break;
                    case 1: invalid.n_ctx+=256; break;
                    case 2: invalid.swa_full=true; break;
                    case 3: invalid.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED; break;
                    case 4: invalid.embeddings=true; break;
                    case 5: invalid.offload_kqv=false; break;
                    case 6: invalid.attention_type=LLAMA_ATTENTION_TYPE_NON_CAUSAL; break;
                    case 7: invalid.ctx_type=LLAMA_CONTEXT_TYPE_DEFAULT; break;
                    case 8: invalid.type_k=GGML_TYPE_Q4_0; break;
                    case 9: invalid.type_v=GGML_TYPE_Q8_0; break;
                }
                bool rejected=false;
                try { llama_context wrong(*assistant,invalid); }
                catch (const std::runtime_error & error) {
                    const char * expected=profile < 8 ?
                        "shared Gemma KV reads require a matching serial TG1 MTP context and target phase arena" :
                        "shared KV source has incompatible cache geometry or format";
                    rejected=std::string(error.what()) == expected;
                    if (!rejected) t.out << "unexpected assistant rejection: " << error.what() << '\n';
                }
                if (!t.assert_true("incompatible assistant rejected before changing target",rejected)) return;
                t.assert_true(stream->complete() && stream->tokens() == 0);
                t.assert_equal(uint32_t(0),resources.full->get_cells(0).get_used());
                t.assert_equal(uint32_t(0),resources.bounded->get_cells(0).get_used());
                t.assert_equal(revision,target->get_compute_memory()->graph_binding_revision());
            }
        }
        size_t free=0,total=0;
        ggml_backend_dev_memory(model->devices.front().dev,&free,&total);
        std::fprintf(stderr,"Gemma actual assistant: streamed=%d arena=%zu device_free_before_context=%zu\n",int(streamed),cp.shared_device_memory_bytes,free);
        context_ptr draft(llama_init_from_model(assistant.get(),dp),llama_free);
        if (!t.assert_true(streamed ? "streamed Gemma assistant context admitted" : "native Gemma assistant context admitted",bool(draft))) return;
        if (streamed) {
            const auto * child=draft->get_compute_memory();
            const auto * parent=target->get_compute_memory();
            if (!t.assert_true("Gemma assistant borrows the existing serial target workspace",
                    child && parent && child->borrows_serial_parent() && child->shared_parent() == parent->shared_parent())) return;
            t.assert_true(parent->has_speculative_consumer());
            if (quotas) {
                llama_context_memory_diagnostics usage;
                const auto & config=stream->host()->config();
                llama_kv_stream_policy_config policy;
                policy.shape=config.shape; policy.capabilities=config.capabilities; policy.layers=config.layers;
                size_t minimum=0;
                if (!t.assert_true(parent->diagnostics(usage) && usage.unused_bytes == 0 &&
                        llama_kv_stream_policy_minimum_pool_bytes(policy,minimum).status == llama_kv_stream_policy_status::success &&
                        minimum <= usage.kv_pool_bytes && usage.kv_pool_bytes <= usage.parent_bytes)) return;
                exact_budget=usage.parent_bytes-usage.kv_pool_bytes+minimum;
                quota_params=cp;
                std::fprintf(stderr,"Gemma joint arena quota: required=%zu current=%zu minimum_pool=%zu\n",
                    exact_budget,usage.parent_bytes,minimum);
            }
        }
        llama_set_embeddings_nextn(draft.get(),true,true);
        const auto words=common_tokenize(target.get(),"A woodland stream flows past trees and birds. ",true);
        if (!t.assert_true(!words.empty())) return;
        common_batch prompt(target.get());
        int last_count=0;
        for (int first=0;first<text_prefix;first+=int(cp.n_ubatch)) {
            prompt.clear(); last_count=std::min(int(cp.n_ubatch),text_prefix-first);
            for (int i=0;i<last_count;++i) prompt.add(words[size_t(first+i)%words.size()],first+i,0,i == last_count-1);
            if (first < 4) std::fprintf(stderr,"Gemma assistant prefill submit: streamed=%d first=%d rows=%d\n",int(streamed),first,last_count);
            if (!t.assert_equal(0,llama_process(target.get(),LLAMA_PROCESS_TYPE_DECODE,prompt.get()))) return;
            if (compact) llama_synchronize(target.get());
            if (first < 4 || (compact && (first+last_count)%128 == 0) || (first+last_count)%512 == 0 || first+last_count == text_prefix)
                std::fprintf(stderr,"Gemma assistant control prefill: streamed=%d progress=%d/%d\n",int(streamed),first+last_count,text_prefix);
        }
        if (image_rows) {
            prompt.clear();
            for (size_t row=0;row<image_rows;++row) {
                llama_pos position[4]={llama_pos(text_prefix+row),0,0,0};
                prompt.add_embd({image.data()+row*width,1,width},position,0,row+1==image_rows);
            }
            llama_set_causal_attn(target.get(),false);
            const int result=llama_process(target.get(),LLAMA_PROCESS_TYPE_DECODE,prompt.get());
            llama_set_causal_attn(target.get(),true);
            if (!t.assert_equal("image-conditioned assistant source prefill",0,result)) return;
            prompt.clear();prompt.add(words.front(),prefix-1,0,true);
            if (!t.assert_equal(0,llama_process(target.get(),LLAMA_PROCESS_TYPE_DECODE,prompt.get()))) return;
            last_count=1;
            std::fprintf(stderr,"Gemma assistant image source: streamed=%d image_rows=%zu frontier=%d arena_bytes=%zu\n",int(streamed),image_rows,prefix,cp.shared_device_memory_bytes);
        }
        llama_synchronize(target.get());
        const auto * h=llama_get_embeddings_nextn_ith(target.get(),last_count-1);
        const auto * initial=llama_get_logits_ith(target.get(),-1);
        if (!t.assert_true(h && initial)) return;
        std::vector<float> hidden(h,h+width);
        const auto target_hidden=hidden;
        const std::vector<float> target_logits(initial,initial+vocab);
        const llama_token first_token=llama_token(std::max_element(initial,initial+vocab)-initial);
        auto token=first_token;
        const auto full_ids=resources.full->get_layer_ids(),local_ids=resources.bounded->get_layer_ids();
        if (!t.assert_true(!full_ids.empty() && !local_ids.empty())) return;
        const std::array<ggml_tensor *,4> source_tensors{
            resources.full->get_k_storage(int32_t(full_ids.back())),resources.full->get_v_storage(int32_t(full_ids.back())),
            resources.bounded->get_k_storage(int32_t(local_ids.back())),resources.bounded->get_v_storage(int32_t(local_ids.back()))};
        const auto snapshot=[&] {
            std::vector<std::vector<uint8_t>> bytes;
            for (auto * tensor : source_tensors) {
                bytes.emplace_back(ggml_nbytes(tensor));
                ggml_backend_tensor_get(tensor,bytes.back().data(),0,bytes.back().size());
            }
            return bytes;
        };
        const auto before=snapshot();
        const auto full_used=resources.full->get_cells(0).get_used(),local_used=resources.bounded->get_cells(0).get_used();
        if (streamed && vision) {
            auto * owner=llama_context_compute_memory(target.get());
            if (!t.assert_true("shared read-only assistant permits vision suspension",owner->can_suspend_for_vision())) return;
            if (!t.assert_true(llama_context_suspend_kv_device(target.get()))) return;
            std::vector<ggml_backend_memory_lease_t> grants;
            llama_context_memory_diagnostics usage;
            if (!t.assert_true(owner->diagnostics(usage) && usage.persistent_cache_bytes<=usage.parent_bytes &&
                    usage.publication_stage_bytes<=usage.parent_bytes-usage.persistent_cache_bytes)) return;
            const size_t reusable=usage.parent_bytes-usage.persistent_cache_bytes-usage.publication_stage_bytes;
            if (!t.assert_true(reusable && owner->lend_suspended({reusable},grants))) return;
            ggml_backend_buffer_clear(ggml_backend_memory_lease_buffer(grants.front()),0x4e);
            for (auto * lease : grants) ggml_backend_memory_lease_free(lease);
            if (!t.assert_true(llama_context_resume_kv_device(target.get(),llama_memory_text_phase::prefill))) return;
            t.assert_true("assistant vision handoff preserves both target KV groups",snapshot()==before);
        }
        if (streamed && !forced) {
            common_batch valid(draft.get());
            const auto id=valid.add(token,prefix,0,true);
            valid.set_embd(id,{hidden.data(),1,width});
            const auto revision=target->get_compute_memory()->graph_binding_revision();
            llama_memory_cache_requirement original;
            if (!t.assert_true(stream->read_source(original))) return;
            for (int profile=0;profile<11;++profile) {
                auto invalid=*valid.get();
                switch (profile) {
                    case 0: invalid.tokens.front().has_embd=false; break;
                    case 1: invalid.tokens.front().id=LLAMA_TOKEN_NULL; break;
                    case 2: invalid.tokens.front().id=llama_token(vocab); break;
                    case 3: invalid.tokens.front().pos[0]=-1; break;
                    case 4: --invalid.tokens.front().pos[0]; break;
                    case 5: ++invalid.tokens.front().pos[0]; break;
                    case 6: invalid.tokens.front().seq_ids={1}; break;
                    case 7: invalid.tokens.front().embd_off=SIZE_MAX; break;
                    case 8: invalid.embd.pop_back(); break;
                    case 9: invalid.n_embd=width-1; break;
                    case 10: invalid.tokens.push_back(invalid.tokens.front()); break;
                }
                if (!t.assert_equal("invalid assistant payload rejects before source acquisition",-1,
                        llama_process(draft.get(),LLAMA_PROCESS_TYPE_DECODE,&invalid))) return;
                llama_memory_cache_requirement unchanged;
                t.assert_true(stream->read_source(unchanged) && unchanged.identity.id == original.identity.id &&
                    unchanged.identity.generation == original.identity.generation);
                t.assert_true("invalid payload leaves all shared source bytes unchanged",snapshot() == before);
                t.assert_equal(full_used,resources.full->get_cells(0).get_used());
                t.assert_equal(local_used,resources.bounded->get_cells(0).get_used());
                t.assert_equal(revision,target->get_compute_memory()->graph_binding_revision());
            }
            llama_memory_clear(llama_get_memory(draft.get()),true);
            t.assert_true("assistant clear cannot clear target KV",snapshot() == before);
            t.assert_equal(full_used,resources.full->get_cells(0).get_used());
            t.assert_equal(local_used,resources.bounded->get_cells(0).get_used());
        }
        for (int step=0;step<3;++step) {
            common_batch batch(draft.get());
            const auto id=batch.add(token,prefix,0,true);
            batch.set_embd(id,{hidden.data(),1,width});
            if (!t.assert_equal("actual assistant forward",0,llama_process(draft.get(),LLAMA_PROCESS_TYPE_DECODE,batch.get()))) return;
            llama_synchronize(draft.get());
            const auto * values=llama_get_logits_ith(draft.get(),-1);
            const auto * next=llama_get_embeddings_nextn_ith(draft.get(),-1);
            if (!t.assert_true(values && next)) return;
            if (!t.assert_true(std::all_of(values,values+vocab,[](float value) { return std::isfinite(value); }))) return;
            logits[streamed].emplace_back(values,values+vocab);
            hidden_states[streamed].emplace_back(next,next+width);
            token=llama_token(std::max_element(values,values+vocab)-values);
            proposals[streamed].push_back(token);
            hidden.assign(next,next+width);
            const auto * retained_h=llama_get_embeddings_nextn_ith(target.get(),last_count-1);
            const auto * retained_logits=llama_get_logits_ith(target.get(),-1);
            t.assert_true("assistant preserves retained target hidden output",retained_h &&
                !std::memcmp(retained_h,target_hidden.data(),width*sizeof(float)));
            t.assert_true("assistant preserves retained target logits",retained_logits &&
                !std::memcmp(retained_logits,target_logits.data(),vocab*sizeof(float)));
            t.assert_true("assistant leaves source KV bytes unchanged",snapshot() == before);
            t.assert_equal(full_used,resources.full->get_cells(0).get_used());
            t.assert_equal(local_used,resources.bounded->get_cells(0).get_used());
            if (stream) {
                t.assert_true(stream->complete() && stream->tokens() == size_t(prefix));
                if (compact) {
                    llama_context_memory_diagnostics usage;
                    t.assert_true(target->get_compute_memory()->diagnostics(usage) && usage.streaming_active);
                }
                if (forced) {
                    const auto & layout=stream->host()->layout();
                    const size_t layer_bytes=size_t(prefix)*(layout.k_token_bytes+layout.v_token_bytes);
                    std::fprintf(stderr,"Gemma assistant forced ring: step=%d source_bytes=%zu entire_pool=%zu\n",step,layer_bytes,stream->pool_grant_bytes());
                    if (!t.assert_true("actual assistant source exceeds the entire pool",layer_bytes > stream->pool_grant_bytes())) return;
                }
            }
        }
        if (!forced) {
            llama_memory_cache_requirement old_source;
            if (stream && !t.assert_true(stream->read_source(old_source))) return;
            if (!t.assert_true(llama_memory_seq_rm(llama_get_memory(target.get()),0,prefix-1,-1))) return;
            if (stream) {
                common_batch stale(draft.get());
                const auto id=stale.add(token,prefix,0,true);
                stale.set_embd(id,{hidden.data(),1,width});
                if (!t.assert_equal("old frontier is not readable after truncation",-1,
                        llama_process(draft.get(),LLAMA_PROCESS_TYPE_DECODE,stale.get()))) return;
            }
            prompt.clear(); prompt.add(words[size_t(prefix-1)%words.size()],prefix-1,0,true);
            if (!t.assert_equal("target rewrites while assistant metadata stays attached",0,
                    llama_process(target.get(),LLAMA_PROCESS_TYPE_DECODE,prompt.get()))) return;
            h=llama_get_embeddings_nextn_ith(target.get(),0);
            initial=llama_get_logits_ith(target.get(),-1);
            if (!t.assert_true(h && initial)) return;
            hidden.assign(h,h+width);
            token=llama_token(std::max_element(initial,initial+vocab)-initial);
            if (stream) {
                llama_memory_cache_requirement refreshed;
                if (!t.assert_true(stream->read_source(refreshed) && refreshed.identity.id == old_source.identity.id &&
                        refreshed.identity.generation > old_source.identity.generation && stream->tokens() == size_t(prefix))) return;
            }
            const auto rewritten=snapshot();
            common_batch fresh(draft.get());
            const auto id=fresh.add(token,prefix,0,true);
            fresh.set_embd(id,{hidden.data(),1,width});
            if (!t.assert_equal("assistant rebinds the new target generation",0,
                    llama_process(draft.get(),LLAMA_PROCESS_TYPE_DECODE,fresh.get()))) return;
            const auto * values=llama_get_logits_ith(draft.get(),-1);
            const auto * next=llama_get_embeddings_nextn_ith(draft.get(),-1);
            if (!t.assert_true(values && next)) return;
            logits[streamed].emplace_back(values,values+vocab);
            hidden_states[streamed].emplace_back(next,next+width);
            t.assert_true("refreshed assistant read preserves rewritten sources",snapshot() == rewritten);
        }
        if (verify) {
            llama_set_kv_stream_decode(target.get(),true);
            if (stream) {
                const auto unchanged=snapshot();
                llama_memory_cache_requirement before_source,after_source;
                if (!t.assert_true(stream->read_source(before_source))) return;
                prompt.clear();
                for (int row=0;row<5;++row) prompt.add(words[size_t(row)%words.size()],prefix+row,0,true);
                if (!t.assert_equal("verification wider than the declared profile rejects before execution",-1,
                        llama_process(target.get(),LLAMA_PROCESS_TYPE_DECODE,prompt.get()))) return;
                t.assert_equal(size_t(prefix),stream->tokens());
                t.assert_true(stream->read_source(after_source) && after_source.identity.id == before_source.identity.id &&
                    after_source.identity.generation == before_source.identity.generation);
                t.assert_true(snapshot() == unchanged);
            }
            for (int count=1;count<=4;++count) {
                prompt.clear();
                for (int row=0;row<count;++row) prompt.add(words[size_t(row)%words.size()],prefix+row,0,true);
                if (!t.assert_equal("declared target verification width executes",0,
                        llama_process(target.get(),LLAMA_PROCESS_TYPE_DECODE,prompt.get()))) return;
                for (int row=0;row<count;++row) {
                    const auto * values=llama_get_logits_ith(target.get(),row);
                    const auto * next=llama_get_embeddings_nextn_ith(target.get(),row);
                    if (!t.assert_true(values && next)) return;
                    logits[streamed].emplace_back(values,values+vocab);
                    hidden_states[streamed].emplace_back(next,next+width);
                    if (row+1 == count) {
                        hidden.assign(next,next+width);
                        token=llama_token(std::max_element(values,values+vocab)-values);
                    }
                }
                if (stream && !t.assert_equal(size_t(prefix+count),stream->tokens())) return;
                const auto verified=snapshot();
                common_batch read(draft.get());
                const auto id=read.add(token,prefix+count,0,true);
                read.set_embd(id,{hidden.data(),1,width});
                if (!t.assert_equal("assistant uses the completed verification frontier",0,
                        llama_process(draft.get(),LLAMA_PROCESS_TYPE_DECODE,read.get()))) return;
                const auto * values=llama_get_logits_ith(draft.get(),-1);
                const auto * next=llama_get_embeddings_nextn_ith(draft.get(),-1);
                if (!t.assert_true(values && next)) return;
                logits[streamed].emplace_back(values,values+vocab);
                hidden_states[streamed].emplace_back(next,next+width);
                t.assert_true("post-verification assistant leaves sources unchanged",snapshot() == verified);
                if (!t.assert_true(llama_memory_seq_rm(llama_get_memory(target.get()),0,prefix,-1))) return;
            }
        }
        draft.reset();
        llama_set_kv_stream_decode(target.get(),true);
        prompt.clear(); prompt.add(first_token,prefix,0,true);
        if (!t.assert_equal("target resumes after reader destruction",0,llama_process(target.get(),LLAMA_PROCESS_TYPE_DECODE,prompt.get()))) return;
        const auto * values=llama_get_logits_ith(target.get(),-1);
        if (!t.assert_true(values != nullptr)) return;
        logits[streamed].emplace_back(values,values+vocab);
        std::fprintf(stderr,"Gemma actual assistant control complete: streamed=%d predictions=3 verify_max=%d\n",int(streamed),verify ? 4 : 0);
    }
    if (quotas) {
        if (!t.assert_true(exact_budget > 1)) return;
        for (bool sufficient : {false,true}) {
            auto cp=quota_params; cp.shared_device_memory_bytes=exact_budget-(sufficient ? 0 : 1);
            context_ptr target(nullptr,llama_free), draft(nullptr,llama_free);
            std::string error;
            try {
                target.reset(new llama_context(*model,cp));
                auto dp=cp; dp.n_batch=dp.n_ubatch=1;
                dp.shared_device_memory_bytes=0; dp.kv_stream_verify_max=0;
                dp.ctx_type=LLAMA_CONTEXT_TYPE_MTP; dp.ctx_other=target.get();
                draft.reset(new llama_context(*assistant,dp));
            } catch (const std::runtime_error & failure) { error=failure.what(); }
            if (!sufficient) {
                const auto fields=error.find("required=");
                size_t required=0,available=0,additional=0;
                if (!t.assert_true("one-byte-short joint arena rejects without private workspace fallback",
                        error.find("persistent local cache does not fit the declared shared arena: insufficient arena quota;")==0 &&
                        fields!=std::string::npos && std::sscanf(error.c_str()+fields,
                            "required=%zu, available=%zu, additional=%zu",&required,&available,&additional)==3)) {
                    t.out << "one-byte-short joint arena error: " << error << '\n'; return;
                }
                t.assert_equal(exact_budget,required);
                t.assert_equal(exact_budget-1,available);
                t.assert_equal(size_t(1),additional);
                continue;
            }
            if (!t.assert_true("exact joint arena admits both contexts",error.empty() && target && draft)) {
                t.out << "exact joint arena error: " << error << '\n'; return;
            }
            t.assert_equal(exact_budget,target->get_compute_memory()->shared_parent_capacity());
            t.assert_true(draft->get_compute_memory()->borrows_serial_parent());
            const auto words=common_tokenize(target.get(),"A woodland stream flows past trees. ",true);
            if (!t.assert_true(!words.empty())) return;
            common_batch prompt(target.get());
            for (int row=0;row<16;++row) prompt.add(words[size_t(row)%words.size()],row,0,row == 15);
            if (!t.assert_equal("exact joint arena short prefill",0,llama_process(target.get(),LLAMA_PROCESS_TYPE_DECODE,prompt.get()))) return;
            const auto * h=llama_get_embeddings_nextn_ith(target.get(),15);
            const auto * values=llama_get_logits_ith(target.get(),-1);
            if (!t.assert_true(h && values)) return;
            common_batch read(draft.get());
            const auto id=read.add(llama_token(std::max_element(values,values+vocab)-values),16,0,true);
            read.set_embd(id,{h,1,width});
            t.assert_equal("exact joint arena assistant handoff",0,llama_process(draft.get(),LLAMA_PROCESS_TYPE_DECODE,read.get()));
        }
    }
    if (export_native) { oracle_file(export_native,true);return; }
    t.assert_true(proposals[0] == proposals[1]);
    if (!t.assert_equal(logits[0].size(),logits[1].size())) return;
    for (size_t row=0;row<logits[0].size();++row) t.assert_true("actual assistant/target logits match native bits",
        std::memcmp(logits[0][row].data(),logits[1][row].data(),vocab*sizeof(float)) == 0);
    for (size_t row=0;row<hidden_states[0].size();++row) t.assert_true("actual assistant hidden rows match native bits",
        std::memcmp(hidden_states[0][row].data(),hidden_states[1][row].data(),width*sizeof(float)) == 0);
}

// Observe the real graph inputs and every assistant block without changing the draft loop.
struct gemma_candidate_trace {
    struct record {
        std::string name;
        std::vector<uint8_t> bytes;
        bool operator==(const record & other) const { return name == other.name && bytes == other.bytes; }
    };
    std::vector<record> records;
    bool valid=true;

    void capture(const char * name,ggml_tensor * tensor,ggml_type type) {
        if (!tensor || tensor->type != type || !ggml_is_contiguous(tensor)) {
            std::fprintf(stderr,"Gemma candidate trace cannot read %s: tensor=%s expected_type=%d actual_type=%d\n",
                name,tensor ? tensor->name : "null",int(type),tensor ? int(tensor->type) : -1);
            valid=false; return;
        }
        records.push_back({name,std::vector<uint8_t>(ggml_nbytes(tensor))});
        ggml_backend_tensor_get(tensor,records.back().bytes.data(),0,records.back().bytes.size());
    }
    static bool callback(ggml_tensor * tensor,bool ask,void * opaque) {
        auto & trace=*static_cast<gemma_candidate_trace *>(opaque);
        const std::string name=tensor->name;
        const bool token_input=tensor->op == GGML_OP_GET_ROWS && tensor->src[1] &&
            !std::strncmp(tensor->src[1]->name,"inp_tokens",10);
        const bool wanted=token_input || name == "inp_xh" || name == "Qcur_pos-0" || name == "result_output" ||
            name == "h_nextn" || name == "out_scaled-0" || name == "out_scaled-1" ||
            name == "out_scaled-2" || name == "out_scaled-3";
        if (!wanted) return false;
        if (ask) return true;
        if (token_input) {
            trace.capture("tokens",tensor->src[1],GGML_TYPE_I32);
        } else if (name == "inp_xh") {
            trace.capture("hidden",tensor->src[1],GGML_TYPE_F32);
        } else if (name == "Qcur_pos-0") {
            trace.capture("position",tensor->src[1],GGML_TYPE_I32);
        } else {
            trace.capture(name.c_str(),tensor,GGML_TYPE_F32);
        }
        return true;
    }
};

// Match the common driver to explicit sequential stock predictions and frozen target sources.
static void test_gemma_candidate_loop(testing & t,llama_model * model,const char * path,bool compact) {
    if (!t.assert_true(model && model->arch == LLM_ARCH_GEMMA4 && path)) return;
    auto mp=llama_model_default_params(); mp.n_gpu_layers=999;
    model_ptr assistant(llama_model_load_from_file(path,mp),llama_model_free);
    if (!t.assert_true(assistant && assistant->arch == LLM_ARCH_GEMMA4_ASSISTANT && assistant->hparams.n_layer_nextn == 4)) return;
    const size_t width=model->hparams.n_embd_out();
    const size_t vocab=llama_vocab_n_tokens(llama_model_get_vocab(assistant.get()));
    kv_stream_bit_oracle oracle;
    for (bool streamed : {false,true}) {
        if (streamed) oracle.compare();
        auto cp=llama_context_default_params();
        cp.n_ctx=4096; cp.n_batch=cp.n_ubatch=compact ? 32 : 256; cp.n_seq_max=1;
        cp.n_threads=cp.n_threads_batch=8; cp.swa_full=false;
        cp.type_k=GGML_TYPE_Q8_0; cp.type_v=GGML_TYPE_Q4_0;
        cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.shared_device_memory_bytes=streamed ? size_t(compact ? 480 : 1024)*1048576 : 0;
        cp.kv_stream_verify_max=streamed ? 4 : 0;
        context_ptr target(llama_init_from_model(model,cp),llama_free);
        if (!t.assert_true(bool(target))) return;
        gemma_candidate_trace trace;
        auto dp=cp; dp.n_batch=dp.n_ubatch=1;
        dp.shared_device_memory_bytes=0; dp.kv_stream_verify_max=0;
        dp.ctx_type=LLAMA_CONTEXT_TYPE_MTP; dp.ctx_other=target.get();
        dp.cb_eval=gemma_candidate_trace::callback; dp.cb_eval_user_data=&trace;
        context_ptr draft(llama_init_from_model(assistant.get(),dp),llama_free);
        if (!t.assert_true(bool(draft))) return;
        llama_set_embeddings_nextn(target.get(),true,false);
        llama_set_embeddings_nextn(draft.get(),true,true);
        const auto words=common_tokenize(target.get(),"A woodland stream flows past trees and birds. ",true);
        if (!t.assert_true(!words.empty())) return;
        const int prefix=compact ? 2305 : 1333;
        llama_tokens prompt;
        common_batch batch(target.get());
        for (int first=0;first<prefix;first+=int(cp.n_ubatch)) {
            batch.clear();
            const int count=std::min(int(cp.n_ubatch),prefix-first);
            for (int row=0;row<count;++row) {
                const auto token=words[size_t(first+row)%words.size()];
                prompt.push_back(token); batch.add(token,first+row,0,row+1 == count);
            }
            if (!t.assert_equal(0,llama_process(target.get(),LLAMA_PROCESS_TYPE_DECODE,batch.get()))) return;
        }
        llama_set_kv_stream_decode(target.get(),true);
        const auto * h=llama_get_embeddings_nextn_ith(target.get(),batch.size()-1);
        const auto * logits=llama_get_logits_ith(target.get(),-1);
        if (!t.assert_true(h && logits)) return;
        const std::vector<float> hidden(h,h+width);
        const llama_token anchor=llama_token(std::max_element(logits,logits+vocab)-logits);
        const auto resources=llama_memory_kv_resources_get(llama_get_memory(target.get()));
        if (!t.assert_true(resources.full && resources.bounded && !resources.recurrent)) return;
        const auto snapshot=[&] {
            std::vector<std::vector<uint8_t>> bytes;
            for (auto * cache : {resources.full,resources.bounded}) {
                const auto ids=cache->get_layer_ids();
                for (auto * tensor : {cache->get_k_storage(int32_t(ids.back())),cache->get_v_storage(int32_t(ids.back()))}) {
                    bytes.emplace_back(ggml_nbytes(tensor));
                    ggml_backend_tensor_get(tensor,bytes.back().data(),0,bytes.back().size());
                }
            }
            return bytes;
        };
        const auto source_bytes=snapshot();
        auto * physical_parent=target->get_compute_memory()->shared_parent();
        llama_memory_cache_requirement source;
        auto * stream=resources.full->get_kv_stream();
        if (stream && !t.assert_true(stream->read_source(source))) return;
        trace.records.clear();
        llama_tokens explicit_tokens;
        auto token=anchor;
        auto next_hidden=hidden;
        for (int step=0;step<3;++step) {
            common_batch read(draft.get());
            const auto id=read.add(token,prefix,0,true);
            read.set_embd(id,{next_hidden.data(),1,width});
            if (!t.assert_equal(0,llama_process(draft.get(),LLAMA_PROCESS_TYPE_DECODE,read.get()))) return;
            const auto * values=llama_get_logits_ith(draft.get(),-1);
            const auto * next=llama_get_embeddings_nextn_ith(draft.get(),-1);
            if (!t.assert_true(values && next)) return;
            const size_t offset=size_t(step)*9;
            if (!t.assert_true(trace.valid && trace.records.size() == offset+9)) return;
            const auto & token_record=trace.records[offset];
            const auto & hidden_record=trace.records[offset+1];
            const auto & position_record=trace.records[offset+2];
            const llama_pos position=prefix;
            t.assert_true("observed token is the current sequential prediction",token_record.name == "tokens" &&
                kv_stream_bytes_equal(&token,sizeof(token),token_record.bytes.data(),token_record.bytes.size()));
            t.assert_true("observed hidden state evolves from the previous prediction",hidden_record.name == "hidden" &&
                kv_stream_bytes_equal(next_hidden.data(),width*sizeof(float),hidden_record.bytes.data(),hidden_record.bytes.size()));
            t.assert_true("all predictions use the fixed target frontier",position_record.name == "position" &&
                kv_stream_bytes_equal(&position,sizeof(position),position_record.bytes.data(),position_record.bytes.size()));
            for (size_t layer=0;layer<4;++layer) t.assert_equal("out_scaled-"+std::to_string(layer),trace.records[offset+3+layer].name);
            token=llama_token(std::max_element(values,values+vocab)-values);
            explicit_tokens.push_back(token); next_hidden.assign(next,next+width);
        }
        if (!trace.valid || trace.records.size() != 27) {
            t.out << "Gemma candidate trace valid=" << trace.valid << " records=" << trace.records.size() << '\n';
            for (const auto & record : trace.records) t.out << record.name << " bytes=" << record.bytes.size() << '\n';
        }
        if (!t.assert_true("three sequential passes observe all inputs, four blocks and outputs",trace.valid && trace.records.size() == 27)) return;
        const auto explicit_trace=trace.records;
        for (bool backend : {false,true}) for (int depth=1;depth<=3;++depth) for (int scenario=0;scenario<4;++scenario) {
            common_params_speculative params;
            params.types={COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
            params.draft.ctx_tgt=target.get(); params.draft.ctx_dft=draft.get();
            params.draft.n_max=depth; params.draft.backend_sampling=backend;
            params.draft.p_min=scenario == 1 ? 1.0f : 0.0f;
            params.draft.n_min=scenario == 2 ? depth+1 : 0;
            common_speculative_ptr spec(common_speculative_init(params,1));
            if (!t.assert_true(bool(spec))) return;
            trace.records.clear();
            common_speculative_begin(spec.get(),0,prompt);
            if (!t.assert_true(common_speculative_process(spec.get(),batch))) return;
            if (!t.assert_true("shared KV process performs no assistant catch-up",trace.records.empty())) return;
            std::vector<uint8_t> saved;
            if (!t.assert_true(common_speculative_get_state(spec.get(),0,saved) && saved.size() == sizeof(uint32_t)+width*sizeof(float))) return;
            if (!t.assert_true("draft carry starts at the target's final hidden row",!std::memcmp(saved.data()+sizeof(uint32_t),hidden.data(),width*sizeof(float)))) return;
            auto stale=saved;
            std::fill(stale.begin()+sizeof(uint32_t),stale.end(),0);
            common_speculative_set_state(spec.get(),0,stale);
            if (!t.assert_true(common_speculative_process(spec.get(),batch))) return;
            std::vector<uint8_t> refreshed;
            if (!t.assert_true("processing current target outputs replaces stale draft carry",
                    common_speculative_get_state(spec.get(),0,refreshed) && refreshed == saved && trace.records.empty())) return;
            for (int repeat=0;repeat<2;++repeat) {
                if (repeat) common_speculative_set_state(spec.get(),0,saved);
                trace.records.clear();
                llama_tokens result;
                auto & request=common_speculative_get_draft_params(spec.get(),0);
                request.drafting=true; request.n_max=scenario == 3 ? 1 : -1;
                request.pos0=prefix; request.id_last=anchor; request.prompt=&prompt; request.result=&result;
                common_speculative_draft(spec.get());
                if (streamed && !t.assert_true("sampler graph changes keep the same shared physical parent",
                        target->get_compute_memory()->shared_parent() == physical_parent &&
                        draft->get_compute_memory()->borrows_serial_parent() &&
                        draft->get_compute_memory()->shared_parent() == physical_parent)) return;
                const size_t calls=scenario == 1 ? 1 : size_t(depth);
                const size_t returned=(scenario == 1 || scenario == 2) ? 0 : scenario == 3 ? 1 : size_t(depth);
                if (!t.assert_true("stock driver observes the explicit fixed-position token/hidden chain",trace.valid &&
                        trace.records.size() == calls*9 && std::equal(trace.records.begin(),trace.records.end(),explicit_trace.begin()))) return;
                if (!t.assert_true("configured length, confidence and minimum-length rules match stock",result ==
                        llama_tokens(explicit_tokens.begin(),explicit_tokens.begin()+returned))) return;
                if (!t.assert_true("all draft passes leave target KV unchanged",snapshot() == source_bytes)) return;
                if (stream) {
                    llama_memory_cache_requirement after;
                    t.assert_true(stream->read_source(after) && after.identity.id == source.identity.id &&
                        after.identity.generation == source.identity.generation && stream->tokens() == size_t(prefix));
                    if (compact) {
                        llama_context_memory_diagnostics usage;
                        t.assert_true("candidate memory profile exercises the ring",target->get_compute_memory()->diagnostics(usage) && usage.streaming_active);
                    }
                }
                const std::string label=std::to_string(int(backend))+"/"+std::to_string(depth)+"/"+std::to_string(scenario)+"/"+std::to_string(repeat);
                oracle.bytes(label+"/candidates",result.data(),result.size()*sizeof(llama_token));
                for (size_t row=0;row<trace.records.size();++row) {
                    const auto & record=trace.records[row];
                    oracle.bytes(label+"/"+std::to_string(row)+"/"+record.name,record.bytes.data(),record.bytes.size());
                }
            }
        }
        if (streamed) oracle.finish();
        std::fprintf(stderr,"Gemma candidate loop: streamed=%d matched_records=%zu bytes=%llu\n",int(streamed),oracle.records(),(unsigned long long)oracle.byte_count());
    }
}

// Exercise the server's accepted-prefix contract with real target writes and the stock MTP driver.
static void test_gemma_acceptance(testing & t,llama_model * model,const char * path,bool compact) {
    if (!t.assert_true(model && model->arch == LLM_ARCH_GEMMA4 && path)) return;
    auto mp=llama_model_default_params(); mp.n_gpu_layers=999;
    model_ptr assistant(llama_model_load_from_file(path,mp),llama_model_free);
    if (!t.assert_true(assistant && assistant->arch == LLM_ARCH_GEMMA4_ASSISTANT)) return;
    const size_t width=model->hparams.n_embd_out();
    const size_t vocab=llama_vocab_n_tokens(llama_model_get_vocab(model));
    kv_stream_bit_oracle oracle;
    for (bool streamed : {false,true}) {
        if (streamed) oracle.compare();
        auto cp=llama_context_default_params();
        cp.n_ctx=4096; cp.n_batch=cp.n_ubatch=compact ? 32 : 256; cp.n_seq_max=1;
        cp.n_threads=cp.n_threads_batch=8; cp.swa_full=false;
        cp.type_k=GGML_TYPE_Q8_0; cp.type_v=GGML_TYPE_Q4_0;
        cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.shared_device_memory_bytes=streamed ? size_t(compact ? 480 : 1024)*1048576 : 0;
        cp.kv_stream_verify_max=streamed ? 4 : 0;
        context_ptr target(llama_init_from_model(model,cp),llama_free);
        if (!t.assert_true(bool(target))) return;
        auto dp=cp; dp.n_batch=dp.n_ubatch=1;
        dp.shared_device_memory_bytes=0; dp.kv_stream_verify_max=0;
        dp.ctx_type=LLAMA_CONTEXT_TYPE_MTP; dp.ctx_other=target.get();
        context_ptr draft(llama_init_from_model(assistant.get(),dp),llama_free);
        if (!t.assert_true(bool(draft))) return;
        llama_set_embeddings_nextn(target.get(),true,false);
        llama_set_embeddings_nextn(draft.get(),true,true);
        const auto resources=llama_memory_kv_resources_get(llama_get_memory(target.get()));
        if (!t.assert_true(resources.full && resources.bounded && !resources.recurrent)) return;
        auto * stream=resources.full->get_kv_stream();
        auto * physical_parent=target->get_compute_memory()->shared_parent();
        const auto words=common_tokenize(target.get(),"A woodland stream flows past trees and birds. ",true);
        if (!t.assert_true(!words.empty())) return;
        llama_tokens history;
        common_batch batch(target.get());
        common_speculative_ptr spec;
        const auto carry=[&] {
            std::vector<uint8_t> bytes;
            if (!common_speculative_get_state(spec.get(),0,bytes)) throw std::runtime_error("missing Gemma carry state");
            return bytes;
        };
        const auto canonical=[&](kv_stream_bit_oracle & out,const std::string & label) {
            llama_synchronize(target.get()); llama_synchronize(draft.get());
            const size_t frontier=history.size();
            for (auto * cache : {resources.full,resources.bounded}) {
                const bool local=cache == resources.bounded;
                const auto tag=label+(local ? "/local" : "/full");
                const size_t first=local && frontier > model->hparams.n_swa ? frontier-model->hparams.n_swa : 0;
                const auto & cells=cache->get_cells(0);
                std::vector<int64_t> metadata;
                const bool extended=model->hparams.n_pos_per_embd() > 1 || model->hparams.ple_n_heads > 0;
                for (const auto & item : cells.seq_pos_get(0)) {
                    if (item.first < 0 || size_t(item.first) >= frontier) throw std::runtime_error("Gemma cache retained a rejected suffix");
                    if (size_t(item.first) < first) continue;
                    metadata.push_back(item.first);
                    if (extended) {
                        const auto & ext=cells.ext_get(item.second);
                        metadata.insert(metadata.end(),{ext.x,ext.y,ext.tok});
                    }
                }
                if (metadata.size() != (frontier-first)*(extended ? 4 : 1)) throw std::runtime_error("incomplete Gemma visible cache metadata");
                out.bytes(tag+"/positions",metadata.data(),metadata.size()*sizeof(int64_t));
                kv_stream_exact_kv(out,tag,cache,first,frontier-first);
            }
        };
        const auto submit=[&](const std::string & label,const llama_tokens & tokens,bool decode) {
            batch.clear();
            for (size_t row=0;row<tokens.size();++row) batch.add(tokens[row],llama_pos(history.size()+row),0,decode || row+1 == tokens.size());
            llama_set_kv_stream_decode(target.get(),decode);
            if (!t.assert_equal(label,0,llama_process(target.get(),LLAMA_PROCESS_TYPE_DECODE,batch.get()))) return false;
            if (spec && !t.assert_true(common_speculative_process(spec.get(),batch))) return false;
            history.insert(history.end(),tokens.begin(),tokens.end());
            if (decode) for (size_t row=0;row<tokens.size();++row) {
                const auto * values=llama_get_logits_ith(target.get(),int(row));
                const auto * hidden=llama_get_embeddings_nextn_ith(target.get(),int(row));
                if (!t.assert_true(values && hidden)) return false;
                oracle.bytes(label+"/logits/"+std::to_string(row),values,vocab*sizeof(float));
                oracle.bytes(label+"/hidden/"+std::to_string(row),hidden,width*sizeof(float));
            }
            return true;
        };
        const auto prefill=[&](size_t frontier,size_t salt) {
            while (history.size() < frontier) {
                llama_tokens tokens(std::min(size_t(cp.n_ubatch),frontier-history.size()));
                for (size_t row=0;row<tokens.size();++row) tokens[row]=words[(history.size()+row+salt)%words.size()];
                if (!submit("prefill/"+std::to_string(history.size()),tokens,false)) return false;
            }
            return true;
        };
        if (!prefill(compact ? 2303 : 1279,0)) return;
        llama_token next=0;
        const auto sample=[&](int row) {
            const auto * values=llama_get_logits_ith(target.get(),row);
            if (!values) throw std::runtime_error("missing Gemma correction logits");
            return llama_token(std::max_element(values,values+vocab)-values);
        };
        next=sample(-1);
        const auto predict=[&](const std::string & label,int depth,llama_token anchor) {
            llama_tokens result;
            common_speculative_get_draft_params(spec.get(),0)={true,depth,llama_pos(history.size()),anchor,&history,&result,nullptr,0.0f,123};
            common_speculative_draft(spec.get());
            if (result.size() != size_t(depth)) throw std::runtime_error("incomplete Gemma acceptance draft");
            oracle.bytes(label,result.data(),result.size()*sizeof(llama_token));
            return result;
        };
        for (int depth=1;depth<=3;++depth) {
            spec.reset();
            common_params_speculative params;
            params.types={COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
            params.draft.ctx_tgt=target.get(); params.draft.ctx_dft=draft.get();
            params.draft.n_max=depth; params.draft.backend_sampling=true;
            spec.reset(common_speculative_init(params,1));
            if (!t.assert_true(bool(spec))) return;
            common_speculative_begin(spec.get(),0,history);
            if (!t.assert_true(common_speculative_process(spec.get(),batch))) return;
            for (int accepted=0;accepted<=depth;++accepted) {
                const std::string label="depth/"+std::to_string(depth)+"/accept/"+std::to_string(accepted);
                const size_t first=history.size();
                const auto proposed=predict(label+"/proposals",depth,next);
                llama_tokens verified{next}; verified.insert(verified.end(),proposed.begin(),proposed.end());
                if (!submit(label+"/verify",verified,true)) return;
                const auto correction=sample(accepted);
                const auto * hidden=llama_get_embeddings_nextn_ith(target.get(),accepted);
                if (!t.assert_true(hidden != nullptr)) return;
                const std::vector<float> selected(hidden,hidden+width);
                common_speculative_accept(spec.get(),0,uint16_t(accepted));
                const auto selected_carry=carry();
                if (!t.assert_true("acceptance chooses the matching target hidden row",selected_carry.size() == sizeof(uint32_t)+width*sizeof(float) &&
                        !std::memcmp(selected_carry.data()+sizeof(uint32_t),selected.data(),width*sizeof(float)))) return;
                const size_t frontier=first+size_t(accepted)+1;
                if (!t.assert_true("accepted prefix truncation succeeds",llama_memory_seq_rm(llama_get_memory(target.get()),0,llama_pos(frontier),-1))) return;
                history.resize(frontier);
                canonical(oracle,label+"/accepted-cache");
                oracle.bytes(label+"/accepted-carry",selected_carry.data(),selected_carry.size());
                (void)predict(label+"/after-accept",depth,correction);
                if (!submit(label+"/correction",{correction},true)) return;
                next=sample(-1);
                if (stream) {
                    t.assert_equal(history.size(),stream->tokens());
                    t.assert_true(target->get_compute_memory()->shared_parent() == physical_parent &&
                        draft->get_compute_memory()->shared_parent() == physical_parent);
                    if (compact) {
                        llama_context_memory_diagnostics usage;
                        t.assert_true("acceptance profile exercises streaming",target->get_compute_memory()->diagnostics(usage) && usage.streaming_active);
                    }
                }
            }
            const std::string label="depth/"+std::to_string(depth)+"/cancel";
            const size_t first=history.size();
            const auto saved_carry=carry();
            common_prompt_checkpoint checkpoint;
            checkpoint.update_tgt(target.get(),0,LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            kv_stream_bit_oracle cancelled;
            canonical(cancelled,"entry");
            const auto proposed=predict(label+"/proposals",depth,next);
            llama_tokens verified{next}; verified.insert(verified.end(),proposed.begin(),proposed.end());
            if (!submit(label+"/provisional",verified,true)) return;
            if (!t.assert_true("cancelled verification restores the saved target prefix",
                    checkpoint.load_tgt(target.get(),0,LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) &&
                    llama_memory_seq_rm(llama_get_memory(target.get()),0,llama_pos(first),-1))) return;
            history.resize(first);
            common_speculative_set_state(spec.get(),0,saved_carry);
            cancelled.compare(); canonical(cancelled,"entry"); cancelled.finish();
            if (!t.assert_true("cancellation retry produces the same draft",predict(label+"/retry",depth,next) == proposed)) return;
            canonical(oracle,label+"/restored-cache");
            if (!submit(label+"/resume",{next},true)) return;
            next=sample(-1);
        }
        // Match the server's extra-token headroom before the final speculative round.
        if (!prefill(cp.n_ctx-3,0)) return;
        next=sample(-1);
        kv_stream_bit_oracle boundary;
        canonical(boundary,"end-prefix");
        batch.clear();
        for (int row=0;row<4;++row) batch.add(words[size_t(row)%words.size()],llama_pos(history.size()+row),0,true);
        llama_set_kv_stream_decode(target.get(),true);
        if (!t.assert_true("oversized final verification rejects before mutation",llama_process(target.get(),LLAMA_PROCESS_TYPE_DECODE,batch.get()) != 0)) return;
        boundary.compare(); canonical(boundary,"end-prefix"); boundary.finish();
        const auto proposed=predict("end/proposals",1,next);
        if (!submit("end/verify",{next,proposed.front()},true)) return;
        common_speculative_accept(spec.get(),0,1);
        if (!submit("end/correction",{sample(-1)},true)) return;
        t.assert_equal(size_t(cp.n_ctx),history.size());
        const auto empty_requests=history;
        common_speculative_get_draft_params(spec.get(),0).drafting=false;
        common_speculative_draft(spec.get());
        t.assert_true("no drafting is scheduled once context headroom is exhausted",history == empty_requests);
        for (size_t request=0;request<2;++request) {
            llama_memory_clear(llama_get_memory(target.get()),true);
            llama_memory_clear(llama_get_memory(draft.get()),true);
            history.clear();
            const llama_tokens short_prompt(17,words[(request+1)%words.size()]);
            common_speculative_begin(spec.get(),0,short_prompt);
            const auto empty=carry();
            t.assert_true("new empty request resets hidden carry",empty.size() == sizeof(uint32_t)+width*sizeof(float) &&
                std::all_of(empty.begin()+sizeof(uint32_t),empty.end(),[](uint8_t value) { return value == 0; }));
            if (!submit("request/"+std::to_string(request)+"/prefill",short_prompt,false)) return;
            canonical(oracle,"request/"+std::to_string(request)+"/cache");
            const auto proposed=predict("request/"+std::to_string(request)+"/proposals",3,sample(-1));
            t.assert_equal(size_t(3),proposed.size());
            const auto request_carry=carry();
            oracle.bytes("request/"+std::to_string(request)+"/carry",request_carry.data(),request_carry.size());
        }
        spec.reset();
        if (streamed) oracle.finish();
        std::fprintf(stderr,"Gemma acceptance: streamed=%d records=%zu bytes=%llu\n",int(streamed),oracle.records(),(unsigned long long)oracle.byte_count());
    }
}

struct warmup_compute_failure {
    inline static warmup_compute_failure * active=nullptr;
    ggml_backend_dev_t device;
    decltype(ggml_backend_device_i::init_backend) initialize;
    std::map<ggml_backend_t,decltype(ggml_backend_i::graph_compute)> compute;
    ggml_status status;
    size_t calls=0;
    warmup_compute_failure(ggml_backend_dev_t device,ggml_status status) :
        device(device),initialize(device->iface.init_backend),status(status) {
        GGML_ASSERT(!active); active=this;
        device->iface.init_backend=[](ggml_backend_dev_t dev,const char * params) {
            auto * backend=active->initialize(dev,params);
            if (backend) {
                active->compute[backend]=backend->iface.graph_compute;
                backend->iface.graph_compute=[](ggml_backend_t backend,ggml_cgraph *) {
                    backend->iface.graph_compute=active->compute.at(backend);
                    ++active->calls;
                    return active->status;
                };
            }
            return backend;
        };
    }
    ~warmup_compute_failure() { device->iface.init_backend=initialize; active=nullptr; }
};

static void test_gemma_warmup_failures(testing & t,const char * target_path,int mode=-1,const char * logit_path=nullptr) {
    auto * reg=ggml_backend_reg_by_name("CUDA");
    if (!t.assert_true(reg && ggml_backend_reg_dev_count(reg))) return;
    auto * device=ggml_backend_reg_dev_get(reg,0);
    std::vector<float> controls[2];
    for (bool streamed : {false,true}) {
        if (mode>=0 && int(streamed)!=mode) continue;
        common_params params;
        params.model.path=target_path;
        params.n_ctx=4096; params.n_batch=params.n_ubatch=256; params.n_parallel=1;
        params.n_gpu_layers=999; params.fit_params=false; params.swa_full=false;
        params.shared_device_memory_bytes=streamed ? size_t(1024)*1048576 : 0;
        params.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
        params.cache_type_k=GGML_TYPE_Q8_0; params.cache_type_v=GGML_TYPE_Q4_0;
        params.cpuparams.n_threads=params.cpuparams_batch.n_threads=8;
        params.warmup=true;
        for (auto status : {GGML_STATUS_ABORTED,GGML_STATUS_ALLOC_FAILED,GGML_STATUS_FAILED}) {
            warmup_compute_failure fault(device,status);
            auto result=common_init_from_params(params);
            t.out << "warmup failure streamed=" << streamed << " status=" << status << " calls=" << fault.calls << '\n';
            if (!t.assert_equal("warmup reaches the injected backend failure",size_t(1),fault.calls)) return;
            t.assert_true("warmup failure retains the initialization result and loaded model",result && result->model());
            t.assert_true("failed warmup must not publish a usable context",result && !result->context());
        }
        auto retry=common_init_from_params(params);
        if (!t.assert_true("startup succeeds after failed warmup teardown",retry && retry->context())) return;
        t.assert_equal("successful warmup leaves no request cells",llama_pos(-1),
            llama_memory_seq_pos_max(llama_get_memory(retry->context()),0));
        const auto words=common_tokenize(retry->context(),"A woodland stream flows past trees. ",true);
        if (!t.assert_true(!words.empty())) return;
        common_batch prompt(retry->context());
        for (int row=0;row<2;++row) prompt.add(words[size_t(row)%words.size()],row,0,row==1);
        if (!t.assert_equal("fresh request computes after failed startup cleanup",0,
                llama_process(retry->context(),LLAMA_PROCESS_TYPE_DECODE,prompt.get()))) return;
        const auto * logits=llama_get_logits_ith(retry->context(),-1);
        const size_t vocab=llama_vocab_n_tokens(llama_model_get_vocab(retry->model()));
        if (!t.assert_true(logits && std::all_of(logits,logits+vocab,[](float value) { return std::isfinite(value); }))) return;
        controls[streamed].assign(logits,logits+vocab);
        retry.reset();
        for (bool model_only : {false,true}) {
            params.warmup=model_only;
            warmup_compute_failure fault(device,GGML_STATUS_FAILED);
            auto bypass=common_init_from_params(params,model_only);
            t.assert_true("disabled warmup and model-only loading do not execute a graph",bypass && bypass->model() &&
                bool(bypass->context())!=model_only);
            t.assert_equal(size_t(0),fault.calls);
        }
    }
    if (mode<0) {
        t.assert_true("recovered native and streamed startup produce exact logits",kv_stream_bytes_equal(
            controls[0].data(),controls[0].size()*sizeof(float),controls[1].data(),controls[1].size()*sizeof(float)));
    } else {
        if (!t.assert_true(logit_path && !controls[mode].empty())) return;
        const std::array<uint64_t,8> profile{0x474d344c4f474954ull,1,4096,2,1,controls[mode].size(),GGML_TYPE_Q8_0,GGML_TYPE_Q4_0};
        std::unique_ptr<FILE,kv_stream_exact_file_free> file(std::fopen(logit_path,mode==0 ? "wb" : "rb"));
        if (!t.assert_true(bool(file))) return;
        std::vector<std::vector<float>> values;
        if (mode==0) values.push_back(controls[0]);
        gemma_logits_file(file.get(),mode==0,profile,values);
        if (mode==1) t.assert_true("fresh-process recovered logits match every native bit",values.size()==1 &&
            kv_stream_bytes_equal(values[0].data(),values[0].size()*sizeof(float),controls[1].data(),controls[1].size()*sizeof(float)));
    }
}

static void test_group_common_dflash_startup(testing & t,const char * target_path,const char * draft_path) {
    common_params params;params.model.path=target_path;params.n_gpu_layers=999;params.fit_params=false;params.warmup=true;
    params.n_ctx=512;params.n_batch=params.n_ubatch=256;params.n_parallel=2;params.kv_unified=true;
    params.shared_device_memory_bytes=size_t(1024)*1024*1024;params.n_outputs_max=16;params.n_outputs_max_per_seq=8;
    params.cache_type_k=GGML_TYPE_Q8_0;params.cache_type_v=GGML_TYPE_Q4_0;params.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
    params.cpuparams.n_threads=params.cpuparams_batch.n_threads=4;
    params.speculative.types={COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V,COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH};
    params.speculative.ngram_map_k4v.size_n=2;params.speculative.ngram_map_k4v.size_m=7;
    params.speculative.draft.n_max=3;params.speculative.draft.n_min=0;params.speculative.draft.p_min=0;
    params.speculative.draft.n_gpu_layers=999;params.speculative.draft.mparams.path=draft_path;
    auto target=common_init_from_params(params,true);
    if (!t.assert_true(target && target->model() && !target->context())) return;
    auto group=llama_context_group::create(*target->model(),common_context_params_to_llama(params,target->model()));
    if (!t.assert_true(group && !group->context())) return;
    auto dp=common_base_params_to_speculative(params);common_speculative_init_result_ptr draft;
    for (unsigned invalid=0;invalid<8;++invalid) {
        auto bad=dp;
        switch (invalid) {
            case 0:bad.n_parallel=3;break;
            case 1:bad.kv_unified=false;break;
            case 2:bad.speculative.draft.cache_type_k_explicit=true;break;
            case 3:bad.speculative.draft.cache_type_v_explicit=true;break;
            case 4:bad.shared_device_memory_bytes=1;break;
            case 5:bad.mmproj.path="not-yet-grouped-vision";break;
            case 6:bad.speculative.ngram_map_k4v.size_m=8;break;
            case 7:bad.speculative.draft.n_max=8;break;
        }
        bool rejected=false;
        try {auto unexpected=common_speculative_init_from_params(bad,target->model(),group->metadata_context(),group.get());}
        catch (const std::invalid_argument &) {rejected=true;}
        if (!t.assert_true(rejected && !group->auxiliary_metadata_context() && !group->metadata_context()->group_parent())) return;
    }
    try {draft=common_speculative_init_from_params(dp,target->model(),group->metadata_context(),group.get());}
    catch (const std::invalid_argument &) {}
    if (!t.assert_true("common startup declares the shared DFlash role before arena allocation",draft && draft->context() &&
            draft->context()==group->auxiliary_metadata_context())) return;
    t.assert_true(!group->metadata_context()->group_parent() && !draft->context()->group_parent() &&
        draft->context()->group_storage_pending() && draft->context()->get_cparams().serial_block_rows==4 &&
        llama_n_rs_seq(group->metadata_context())==7);
    for (const auto & entry : draft->context()->get_memory()->memory_breakdown()) t.assert_equal(size_t(0),entry.second);
    params.speculative.draft.ctx_tgt=group->metadata_context();params.speculative.draft.ctx_dft=draft->context();
    common_speculative_ptr spec(common_speculative_init(params.speculative,2));if (!t.assert_true(spec!=nullptr)) return;
    auto input=llama_batch_init(12,0,1);
    struct release_batch {llama_batch & batch;~release_batch() {llama_batch_free(batch);}} release{input};input.n_tokens=12;
    for (int32_t row=0;row<12;++row) {
        const auto slot=llama_seq_id(row/6);input.token[row]=10+slot*20+row%6;input.pos[row]=row%6;
        input.n_seq_id[row]=1;input.seq_id[row][0]=slot;input.logits[row]=true;
    }
    llama_kv_stream_selection selected;
    if (!t.assert_true(llama_kv_stream_select({512,2},{{0,0},{1,0}},selected)==llama_kv_stream_selection_status::success &&
            group->activate(selected,group->source_handles(),input,llama_memory_work_phase::target_prefill,1,1))) return;
    auto * active=group->context();auto * decoder=group->auxiliary_context();
    if (!t.assert_true(common_speculative_rebind(spec.get(),active,decoder))) return;
    common_batch prefill(active);std::array<llama_tokens,2> histories,proposals;
    for (int32_t row=0;row<12;++row) {prefill.add(input.token[row],input.pos[row],input.seq_id[row][0],true);histories[row/6].push_back(input.token[row]);}
    if (!t.assert_equal(0,llama_process(active,LLAMA_PROCESS_TYPE_DECODE,prefill.get())) ||
            !t.assert_true(common_speculative_process(spec.get(),prefill))) return;
    for (llama_seq_id slot=0;slot<2;++slot) {
        const auto * logits=active->get_logits_ith(slot*6+5);if (!t.assert_true(logits!=nullptr)) return;
        common_speculative_begin(spec.get(),slot,histories[slot],{0,1});
        auto & request=common_speculative_get_draft_params(spec.get(),slot);request.drafting=true;request.n_max=7;request.pos0=6;
        request.id_last=llama_token(std::max_element(logits,logits+target->model()->vocab.n_tokens())-logits);
        request.prompt=&histories[slot];request.result=&proposals[slot];
    }
    if (std::getenv("LLAMA_TEST_GROUP_DFLASH_QUEUED_FAILURE")) {
        common_speculative_get_draft_params(spec.get(),0).n_max=1;
        common_speculative_draft(spec.get());decoder->synchronize();
        const auto expected=proposals;
        if (!t.assert_true(expected[0].size()==1 && expected[1].size()==3)) return;
        std::array<common_speculative_draft_params,2> settings;
        for (llama_seq_id slot=0;slot<2;++slot) {
            settings[slot]=common_speculative_get_draft_params(spec.get(),slot);settings[slot].drafting=true;
            proposals[slot].clear();
            if (!t.assert_true(llama_memory_seq_rm(decoder->get_memory(),slot,6,-1))) return;
            common_speculative_get_draft_params(spec.get(),slot)=settings[slot];
        }
        bool rejected=false;
        {
            dflash_compute_failure fault(decoder,GGML_STATUS_FAILED,2,true);
            try {common_speculative_draft(spec.get());}
            catch (const std::runtime_error & error) {rejected=std::string(error.what())=="selected DFlash execution failed";}
            t.out<<"DFlash queued failure rejected="<<rejected<<" partial="<<proposals[0].size()<<","<<proposals[1].size()
                <<" calls="<<fault.calls<<" queued="<<fault.queued<<" injected="<<fault.injected<<'\n';
            if (!t.assert_true("a queued DFlash error rejects the earlier partition's partial proposals",rejected &&
                    proposals[0].empty() && proposals[1].empty() && fault.injected==1 && fault.queued==fault.calls && fault.calls>=2)) return;
        }
        if (!t.assert_true("failed execution contexts cannot be reused",!common_speculative_rebind(spec.get(),active,decoder))) return;
        if (!t.assert_true(common_speculative_rebind(spec.get(),group->metadata_context(),group->auxiliary_metadata_context()) &&
                group->discard() && group->reset_request(0) && group->reset_request(1))) return;
        for (llama_seq_id slot=0;slot<2;++slot) common_speculative_begin(spec.get(),slot,{}, {0,2});
        if (!t.assert_true(group->activate(selected,group->source_handles(),input,llama_memory_work_phase::target_prefill,1,1))) return;
        active=group->context();decoder=group->auxiliary_context();
        if (!t.assert_true(common_speculative_rebind(spec.get(),active,decoder))) return;
        common_batch retry(active);
        for (int32_t row=0;row<12;++row) retry.add(input.token[row],input.pos[row],input.seq_id[row][0],true);
        if (!t.assert_true(llama_process(active,LLAMA_PROCESS_TYPE_DECODE,retry.get())==0 && common_speculative_process(spec.get(),retry))) return;
        for (llama_seq_id slot=0;slot<2;++slot) {
            common_speculative_begin(spec.get(),slot,histories[slot],{0,2});
            settings[slot].ngram_context={0,2};common_speculative_get_draft_params(spec.get(),slot)=settings[slot];
        }
        common_speculative_draft(spec.get());decoder->synchronize();
        if (!t.assert_true("reconstructed DFlash proposals match the complete pre-failure control",proposals==expected)) return;
        if (!t.assert_true(common_speculative_rebind(spec.get(),group->metadata_context(),group->auxiliary_metadata_context()) && group->park())) return;
        t.out<<"common grouped DFlash queued error and exact-proposal recovery completed\n";return;
    }
    common_speculative_draft(spec.get());decoder->synchronize();
    const auto stats=common_speculative_get_stats(spec.get());
    if (!t.assert_true(stats.size()==2 && stats[1].n_predict==1)) return;
    common_batch verify(active);
    for (llama_seq_id slot=0;slot<2;++slot) {
        if (!t.assert_equal(size_t(3),proposals[slot].size()) ||
                !t.assert_true(common_speculative_get_last_type(spec.get(),slot)==COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH &&
                    llama_memory_seq_rm(decoder->get_memory(),slot,6,-1))) return;
        verify.add(common_speculative_get_draft_params(spec.get(),slot).id_last,6,slot,true);
        for (size_t row=0;row<proposals[slot].size();++row) verify.add(proposals[slot][row],7+llama_pos(row),slot,true);
    }
    llama_set_kv_stream_decode(active,true);
    if (!t.assert_equal(0,llama_process(active,LLAMA_PROCESS_TYPE_DECODE,verify.get())) ||
            !t.assert_true(common_speculative_process(spec.get(),verify))) return;
    if (!t.assert_true(common_speculative_rebind(spec.get(),group->metadata_context(),group->auxiliary_metadata_context()) && group->park())) return;
    t.out<<"common grouped DFlash3/K4V7 startup/prefill/draft/verify/teacher/park completed\n";
}

static void test_group_common_mtp_startup(testing & t,const char * target_path,const char * assistant_path,bool failure=false,uint32_t maximum=3) {
    common_params params;
    params.model.path=target_path;params.n_gpu_layers=999;params.fit_params=false;params.warmup=true;
    params.n_ctx=4096;params.n_batch=params.n_ubatch=256;params.n_parallel=2;params.kv_unified=true;
    params.shared_device_memory_bytes=size_t(1024)*1024*1024;
    params.cache_type_k=GGML_TYPE_Q8_0;params.cache_type_v=GGML_TYPE_Q4_0;params.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
    params.n_outputs_max=8;params.n_outputs_max_per_seq=4;
    params.cpuparams.n_threads=params.cpuparams_batch.n_threads=4;
    params.speculative.types={COMMON_SPECULATIVE_TYPE_DRAFT_MTP};params.speculative.draft.n_max=int(maximum);
    params.speculative.draft.n_min=0;params.speculative.draft.p_min=0;params.speculative.draft.backend_sampling=false;
    params.speculative.draft.n_gpu_layers=999;
    if (assistant_path) params.speculative.draft.mparams.path=assistant_path;
    auto target=common_init_from_params(params,true);
    if (!t.assert_true(target && target->model() && !target->context())) return;
    const auto cparams=common_context_params_to_llama(params,target->model());
    common_speculative_init_result_ptr draft;
    auto group=llama_context_group::create(*target->model(),cparams);
    if (!t.assert_true(group && group->metadata_context() && !group->context())) return;
    {
        llama_context lone(*target->model(),cparams,std::vector<llama_seq_id>{1});
        t.assert_true(lone.group_storage_pending() && !lone.group_parent());
        if (auto * recurrent=lone.get_kv_resources().recurrent)
            t.assert_true(recurrent->spill_bank(1) && !recurrent->spill_bank(0));
    }
    auto dparams=common_base_params_to_speculative(params);
    for (const bool key : {true,false}) {
        auto invalid=dparams;
        invalid.speculative.draft.cache_type_k_explicit=key;
        invalid.speculative.draft.cache_type_v_explicit=!key;
        bool rejected=false;
        try {auto unexpected=common_speculative_init_from_params(invalid,target->model(),group->metadata_context(),group.get());}
        catch (const std::invalid_argument &) {rejected=true;}
        if (!t.assert_true(rejected)) return;
        t.assert_true(!group->context() && !group->metadata_context()->group_parent());
    }
    for (unsigned profile=0;profile<7;++profile) {
        auto invalid=dparams;
        switch (profile) {
            case 0:invalid.n_parallel=3;break;
            case 1:invalid.kv_unified=false;break;
            case 2:invalid.embedding=true;break;
            case 3:invalid.mmproj.path="unused-invalid-projector";break;
            case 4:invalid.shared_device_memory_bytes=1;break;
            case 5:invalid.kv_stream_pool_bytes=1;break;
            case 6:invalid.speculative.types.push_back(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V);break;
        }
        bool rejected=false;
        try {auto unexpected=common_speculative_init_from_params(invalid,target->model(),group->metadata_context(),group.get());}
        catch (const std::invalid_argument &) {rejected=true;}
        if (!t.assert_true(rejected)) return;
    }
    struct loading_state {llama_context_group * group;unsigned calls=0;bool unallocated=true;} loading{group.get()};
    dparams.load_progress_callback=[](float,void * data) {
        auto & state=*static_cast<loading_state *>(data);++state.calls;
        state.unallocated &= !state.group->context() && !state.group->metadata_context()->group_parent();
        auto * child=state.group->auxiliary_metadata_context();
        state.unallocated &= !child || !child->group_parent();
        return true;
    };
    dparams.load_progress_callback_user_data=&loading;
    draft=common_speculative_init_from_params(dparams,target->model(),group->metadata_context(),group.get());
    if (!t.assert_true(draft && draft->context()==group->auxiliary_metadata_context() && draft->context())) return;
    t.assert_true(loading.unallocated && (assistant_path ? loading.calls>0 : loading.calls==0));
    t.assert_true(!group->metadata_context()->group_parent() && !draft->context()->group_parent());
    t.assert_true(draft->context()->group_storage_pending() && !draft->context()->get_cparams().shared_device_memory_bytes);
    for (const auto & entry : draft->context()->get_memory()->memory_breakdown()) t.assert_equal(size_t(0),entry.second);
    params.speculative.draft.ctx_tgt=group->metadata_context();params.speculative.draft.ctx_dft=draft->context();
    common_speculative_ptr spec(common_speculative_init(params.speculative,2));
    if (!t.assert_true(spec!=nullptr)) return;
    if (failure) for (llama_seq_id slot=0;slot<2;++slot) {
        std::vector<uint8_t> seeded,observed;
        if (!t.assert_true(common_speculative_get_state(spec.get(),slot,seeded) && seeded.size()>sizeof(uint32_t))) return;
        std::fill(seeded.begin()+sizeof(uint32_t),seeded.end(),0x3f);
        common_speculative_set_state(spec.get(),slot,seeded);
        if (!t.assert_true(common_speculative_get_state(spec.get(),slot,observed) && observed==seeded)) return;
        common_speculative_begin(spec.get(),slot,{});
        if (!t.assert_true("an empty new-request context clears previous MTP hidden carry",
                common_speculative_get_state(spec.get(),slot,observed) &&
                std::all_of(observed.begin()+sizeof(uint32_t),observed.end(),[](uint8_t value) {return value==0;}))) return;
    }
    auto input=llama_batch_init(2,0,1);
    struct release_batch {llama_batch & batch;~release_batch() {llama_batch_free(batch);}} release{input};
    input.n_tokens=2;
    for (llama_seq_id slot=0;slot<2;++slot) {
        input.token[slot]=100+slot;input.pos[slot]=0;input.n_seq_id[slot]=1;input.seq_id[slot][0]=slot;input.logits[slot]=true;
    }
    llama_kv_stream_selection selection;
    if (!t.assert_true(llama_kv_stream_select({cparams.n_ctx,2},{{0,0},{1,0}},selection)==llama_kv_stream_selection_status::success &&
            group->activate(selection,group->source_handles(),input,llama_memory_work_phase::target_prefill,1,1))) return;
    auto * active=group->context();auto * child=group->auxiliary_context();
    if (!t.assert_true(active && child && active->group_parent()==child->group_parent() &&
            common_speculative_rebind(spec.get(),active,child))) return;
    common_batch prompt(active);std::array<llama_tokens,2> histories,proposals;
    for (llama_seq_id slot=0;slot<2;++slot) {histories[slot]={llama_token(100+slot)};prompt.add(100+slot,0,slot,true);}
    if (!t.assert_equal(0,llama_process(active,LLAMA_PROCESS_TYPE_DECODE,prompt.get())) ||
            !t.assert_true(common_speculative_process(spec.get(),prompt))) return;
    for (llama_seq_id slot=0;slot<2;++slot) {
        const auto size=llama_state_seq_get_size_ext(child,slot,LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        std::vector<uint8_t> state(size);
        if (!t.assert_true(size && llama_state_seq_get_data_ext(child,state.data(),state.size(),slot,
                LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)==size)) return;
    }
    for (llama_seq_id slot=0;slot<2;++slot) {
        const auto * logits=llama_get_logits_ith(active,slot);
        if (!t.assert_true(logits && std::all_of(logits,logits+target->model()->vocab.n_tokens(),[](float value) {return std::isfinite(value);}))) return;
        const auto next=llama_token(std::max_element(logits,logits+target->model()->vocab.n_tokens())-logits);
        common_speculative_begin(spec.get(),slot,histories[slot]);
        common_speculative_get_draft_params(spec.get(),slot)={true,1,1,next,&histories[slot],&proposals[slot],nullptr,0.0f,123};
    }
    size_t prediction_calls=0;
    if (failure) {
        dflash_compute_failure observer(child,GGML_STATUS_SUCCESS,1,true);
        common_speculative_draft(spec.get());llama_synchronize(child);prediction_calls=observer.calls;
    } else {common_speculative_draft(spec.get());llama_synchronize(child);}
    t.assert_true(proposals[0].size()==1 && proposals[1].size()==1);
    if (maximum<3) {
        for (llama_seq_id slot=0;slot<2;++slot)
            if (!t.assert_true(llama_memory_seq_rm(child->get_memory(),slot,1,-1))) return;
        if (!t.assert_true(common_speculative_rebind(spec.get(),group->metadata_context(),group->auxiliary_metadata_context()) && group->park())) return;
        for (llama_seq_id slot=0;slot<2;++slot) {
            input.pos[slot]=1;input.token[slot]=common_speculative_get_draft_params(spec.get(),slot).id_last;
        }
        if (!t.assert_true(llama_kv_stream_select({cparams.n_ctx,2},{{0,2},{1,2}},selection)==llama_kv_stream_selection_status::success &&
                group->activate(selection,group->source_handles(),input,llama_memory_work_phase::target_verify,1,2))) return;
        active=group->context();child=group->auxiliary_context();
        if (!t.assert_true(common_speculative_rebind(spec.get(),active,child))) return;
        std::array<llama_kv_stream_logical_checkpoint,2> before;
        for (uint32_t slot=0;slot<2;++slot) {
            std::shared_ptr<llama_kv_stream_logical_cache> cache;
            if (!t.assert_true(group->source_table()->get(group->source_handles()[slot],0,cache))) return;
            before[slot]=cache->checkpoint();
        }
        common_batch invalid(active);
        for (llama_seq_id slot=0;slot<2;++slot) for (uint32_t row=0;row<maximum+2;++row) invalid.add(100+slot,1+row,slot,true);
        if (!t.assert_equal("verification beyond declared per-sequence width rejects before publication",-1,
                llama_process(active,LLAMA_PROCESS_TYPE_DECODE,invalid.get()))) return;
        for (uint32_t slot=0;slot<2;++slot) {
            std::shared_ptr<llama_kv_stream_logical_cache> cache;
            if (!t.assert_true(group->source_table()->get(group->source_handles()[slot],0,cache) && cache->ready() &&
                    cache->tokens()==before[slot].tokens && cache->identity().generation==before[slot].generation &&
                    llama_memory_seq_pos_max(active->get_memory(),llama_seq_id(slot))==0)) return;
        }
        common_batch valid(active);
        for (llama_seq_id slot=0;slot<2;++slot) valid.add(input.token[slot],1,slot,true);
        if (!t.assert_true(llama_process(active,LLAMA_PROCESS_TYPE_DECODE,valid.get())==0 && common_speculative_process(spec.get(),valid))) return;
    }
    if (failure) {
        if (!t.assert_true(prediction_calls>=3 && prediction_calls%3==0)) return;
        const auto expected=proposals;
        std::array<common_speculative_draft_params,2> settings;
        for (llama_seq_id slot=0;slot<2;++slot) {
            settings[slot]=common_speculative_get_draft_params(spec.get(),slot);
            settings[slot].drafting=true;proposals[slot].clear();
            if (!t.assert_true(llama_memory_seq_rm(child->get_memory(),slot,1,-1))) return;
            common_speculative_get_draft_params(spec.get(),slot)=settings[slot];
        }
        bool rejected=false;
        {
            dflash_compute_failure fault(child,GGML_STATUS_FAILED,2*(prediction_calls/3)+1,true);
            try {common_speculative_draft(spec.get());}
            catch (const std::runtime_error & error) {rejected=std::string(error.what())=="selected MTP execution failed";t.out<<"MTP fault status="<<error.what()<<std::endl;}
            t.out<<"MTP queued fault: rejected="<<rejected<<" proposals="<<proposals[0].size()<<","<<proposals[1].size()
                <<" queued="<<fault.queued<<" calls="<<fault.calls<<" injected_at="<<fault.fail_call<<std::endl;
            if (!t.assert_true("a queued MTP error rejects every partial proposal",rejected &&
                    proposals[0].empty() && proposals[1].empty() && fault.queued==fault.calls && fault.calls>=fault.fail_call && fault.injected==1)) return;
        }
        if (!t.assert_true(common_speculative_rebind(spec.get(),group->metadata_context(),group->auxiliary_metadata_context()))) return;
        if (!t.assert_true(group->discard() && group->reset_request(0) && group->reset_request(1))) return;
        for (llama_seq_id slot=0;slot<2;++slot) common_speculative_begin(spec.get(),slot,{});
        if (!t.assert_true(group->activate(selection,group->source_handles(),input,llama_memory_work_phase::target_prefill,1,2))) return;
        active=group->context();child=group->auxiliary_context();
        if (!t.assert_true(common_speculative_rebind(spec.get(),active,child))) return;
        common_batch retry(active);
        for (llama_seq_id slot=0;slot<2;++slot) retry.add(100+slot,0,slot,true);
        if (!t.assert_true(llama_process(active,LLAMA_PROCESS_TYPE_DECODE,retry.get())==0 && common_speculative_process(spec.get(),retry))) return;
        for (llama_seq_id slot=0;slot<2;++slot) {
            const auto * logits=llama_get_logits_ith(active,slot);
            const auto next=llama_token(std::max_element(logits,logits+target->model()->vocab.n_tokens())-logits);
            if (!t.assert_equal(settings[slot].id_last,next)) return;
            common_speculative_begin(spec.get(),slot,histories[slot]);
            common_speculative_get_draft_params(spec.get(),slot)=settings[slot];
        }
        common_speculative_draft(spec.get());llama_synchronize(child);
        if (!t.assert_true(proposals==expected)) return;
    }
    if (!t.assert_true(common_speculative_rebind(spec.get(),group->metadata_context(),group->auxiliary_metadata_context()))) return;
    t.assert_true(group->park() && !group->context() && !group->auxiliary_context());
}

static void test_gemma_common_startup(testing & t,const char * target_path,const char * assistant_path) {
    common_params params;
    params.model.path=target_path;
    params.n_ctx=4096; params.n_batch=params.n_ubatch=256; params.n_parallel=1;
    params.n_gpu_layers=999; params.fit_params=false; params.swa_full=false;
    params.shared_device_memory_bytes=size_t(1024)*1048576;
    params.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
    params.cache_type_k=GGML_TYPE_Q8_0; params.cache_type_v=GGML_TYPE_Q4_0;
    params.cpuparams.n_threads=params.cpuparams_batch.n_threads=8;
    params.speculative.types={COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
    params.speculative.draft.n_max=3;
    params.speculative.draft.n_gpu_layers=999;
    params.speculative.draft.mparams.path=assistant_path;
    auto main=common_init_from_params(params);
    auto * target=main ? main->context() : nullptr;
    if (!t.assert_true("common Gemma target startup succeeds",target != nullptr)) return;
    const auto * parent=target->get_compute_memory();
    if (!t.assert_true(parent && parent->serial_ready())) return;
    t.assert_equal(uint32_t(0),target->get_cparams().n_rs_seq);
    t.assert_equal(uint32_t(4),target->get_cparams().kv_stream_verify_max);
    struct load_probe {
        const llama_context_memory * parent;
        std::atomic<uint32_t> observed{0};
    } probe{parent};
    auto draft_params=common_base_params_to_speculative(params);
    draft_params.load_progress_callback=[](float,void * opaque) {
        auto & p=*static_cast<load_probe *>(opaque);
        p.observed.fetch_or(p.parent->serial_ready() ? 3u : 1u,std::memory_order_relaxed);
        return true;
    };
    draft_params.load_progress_callback_user_data=&probe;
    auto speculative=common_speculative_init_from_params(draft_params,main->model(),target);
    auto * draft=speculative ? speculative->context() : nullptr;
    if (!t.assert_true("common Gemma assistant startup succeeds",draft != nullptr)) return;
    auto * full=llama_memory_kv_resources_get(llama_get_memory(target)).full;
    auto * reader_source=full ? full->get_kv_stream() : nullptr;
    if (!t.assert_true(reader_source && llama_kv_stream_schedule_next(target,draft,3))) return;
    const auto next_read=reader_source->next_consumer(llama_kv_stream_access_role::append);
    t.assert_true("shared MTP schedules an immutable source read, not a new append",
        next_read.access == llama_kv_stream_access_role::read && next_read.cache_id == reader_source->host()->cache_id() &&
        next_read.layer+1 == full->get_layer_ids().size() && next_read.query_tokens == 1 && next_read.future_tokens == 0);
    t.assert_equal("target warmup graphs are retired before assistant weight loading",uint32_t(1),
        probe.observed.load(std::memory_order_relaxed));
    t.assert_true(draft->get_compute_memory()->borrows_serial_parent());
    t.assert_true(draft->get_compute_memory()->shared_parent() == parent->shared_parent());
    t.assert_equal(uint32_t(1),draft->n_batch());
    t.assert_equal(uint32_t(1),draft->n_ubatch());
    const auto words=common_tokenize(target,"A woodland stream flows past trees and birds. ",true);
    if (!t.assert_true(!words.empty())) return;
    common_batch batch(target);
    for (int row=0;row<64;++row) batch.add(words[size_t(row)%words.size()],row,0,row == 63);
    if (!t.assert_equal("short prefill after common warmup",0,llama_process(target,LLAMA_PROCESS_TYPE_DECODE,batch.get()))) return;
    const size_t width=target->get_model().hparams.n_embd_out();
    const size_t vocab=llama_vocab_n_tokens(llama_model_get_vocab(main->model()));
    const auto predict=[&](int position,int hidden_row) {
        const auto * hidden=llama_get_embeddings_nextn_ith(target,hidden_row);
        const auto * logits=llama_get_logits_ith(target,-1);
        if (!t.assert_true(hidden && logits)) return false;
        common_batch read(draft);
        const auto id=read.add(llama_token(std::max_element(logits,logits+vocab)-logits),position,0,true);
        read.set_embd(id,{hidden,1,width});
        if (!t.assert_true(llama_kv_stream_schedule_next(draft,target,0))) return false;
        if (!t.assert_equal("common assistant reads target outputs",0,llama_process(draft,LLAMA_PROCESS_TYPE_DECODE,read.get()))) return false;
        if (!t.assert_true("actual assistant adopts source lookahead",reader_source->sequence_stats(llama_kv_stream_access_role::read).adopted)) return false;
        const auto * result=llama_get_logits_ith(draft,-1);
        return t.assert_true(result && std::all_of(result,result+vocab,[](float value) { return std::isfinite(value); }));
    };
    if (!predict(64,63)) return;
    llama_memory_cache_requirement before_probe,after_probe;
    if (!t.assert_true(reader_source && reader_source->read_source(before_probe))) return;
    const auto revision=parent->graph_binding_revision();
    t.assert_equal("shared assistant rollback classification performs no ordinary decode probe",
        COMMON_CONTEXT_SEQ_RM_TYPE_PART,common_context_can_seq_rm(draft));
    t.assert_true(reader_source->read_source(after_probe) &&
        before_probe.identity.id == after_probe.identity.id && before_probe.identity.generation == after_probe.identity.generation &&
        reader_source->tokens() == 64);
    t.assert_equal(revision,parent->graph_binding_revision());
    llama_set_kv_stream_decode(target,true);
    batch.clear();
    for (int row=0;row<4;++row) batch.add(words[size_t(row)%words.size()],64+row,0,true);
    if (!t.assert_equal("common target verifies declared four-query batch",0,
            llama_process(target,LLAMA_PROCESS_TYPE_DECODE,batch.get()))) return;
    t.assert_true("target verification hands its committed source to MTP",reader_source->prefetch_primed());
    if (!predict(68,3)) return;
    t.assert_true(llama_kv_stream_schedule_next(target,nullptr,0));
    speculative.reset();
    t.assert_true(!parent->has_speculative_consumer());
    batch.clear(); batch.add(words.front(),68,0,true);
    t.assert_equal("common target resumes after assistant destruction",0,
        llama_process(target,LLAMA_PROCESS_TYPE_DECODE,batch.get()));
}

static void test_gemma_bounded_cache(testing & t, llama_model * model) {
    if (!t.assert_true(model && model->arch == LLM_ARCH_GEMMA4)) return;
    std::vector<std::vector<float>> controls[2];
    for (bool borrowed : {false,true}) {
        auto cp=llama_context_default_params();
        cp.n_ctx=2048; cp.n_batch=cp.n_ubatch=256; cp.n_seq_max=1;
        cp.swa_full=false;
        cp.type_k=GGML_TYPE_Q8_0; cp.type_v=GGML_TYPE_Q4_0;
        cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
        context_ptr context(llama_init_from_model(model,cp),llama_free);
        if (!t.assert_true(bool(context))) return;
        auto resources=llama_memory_kv_resources_get(llama_get_memory(context.get()));
        auto * local=resources.bounded;
        if (!t.assert_true(local && resources.full && !resources.recurrent)) return;
        auto * storage=local->get_k_storage(0)->buffer;
        llama_memory_params params{}; params.type_k=cp.type_k; params.type_v=cp.type_v;
        llama_memory_bounded_kv_layout quote, actual;
        if (!t.assert_true(llama_kv_cache_iswa::bounded_storage_requirements(model->hparams,params,
                context->get_cparams(),ggml_backend_buffer_get_type(storage),{607,1},quote) &&
                local->describe_bounded_storage(607,1,2048,256,actual))) return;
        t.assert_equal(size_t(50),quote.layers.size());
        t.assert_equal(uint32_t(1280),quote.allocation.capacity_rows);
        t.assert_equal(size_t(425984000),quote.allocation.bytes);
        t.assert_equal(actual.allocation.bytes,quote.allocation.bytes);
        using arena_ptr=std::unique_ptr<ggml_backend_memory_arena,decltype(&ggml_backend_memory_arena_free)>;
        using lease_ptr=std::unique_ptr<ggml_backend_memory_lease,decltype(&ggml_backend_memory_lease_free)>;
        arena_ptr arena(nullptr,ggml_backend_memory_arena_free);
        lease_ptr lease(nullptr,ggml_backend_memory_lease_free);
        std::unique_ptr<llama_memory_bounded_kv> owner;
        if (borrowed) {
            arena.reset(ggml_backend_memory_arena_new_from_buffer(storage));
            if (!t.assert_true(arena && ggml_backend_memory_arena_begin(arena.get(),0) &&
                    ggml_backend_memory_arena_reserve_at(arena.get(),77,0,quote.allocation.bytes,quote.allocation.alignment,
                        GGML_BACKEND_MEMORY_REGION_PERSISTENT,nullptr) && ggml_backend_memory_arena_commit(arena.get()))) return;
            lease.reset(ggml_backend_memory_arena_acquire(arena.get(),77));
            params.bounded_cache_lease=lease.get();
            const auto filter=[&](int32_t il) { return model->hparams.is_swa(uint32_t(il)); };
            const auto no_reuse=[](int32_t) { return int32_t(-1); };
            bool alias_rejected=false;
            try {
                const auto alias=[](int32_t il) { return il == 0 ? int32_t(1) : int32_t(-1); };
                llama_kv_cache invalid(*model,model->hparams,cp.type_k,cp.type_v,false,true,true,1280,1,256,
                    model->hparams.n_swa,model->hparams.swa_type,nullptr,filter,alias,nullptr,"",&params);
            } catch (const std::runtime_error & error) {
                alias_rejected=std::string(error.what()) == "bounded cache lease cannot alias selected layers";
            }
            if (!t.assert_true(alias_rejected)) return;
            ggml_backend_buffer_ptr short_buffer(ggml_backend_buffer_view(storage,0,quote.allocation.bytes-1));
            arena_ptr short_arena(ggml_backend_memory_arena_new_from_buffer(short_buffer.get()),ggml_backend_memory_arena_free);
            if (!t.assert_true(short_arena && ggml_backend_memory_arena_begin(short_arena.get(),0) &&
                    ggml_backend_memory_arena_reserve_at(short_arena.get(),77,0,quote.allocation.bytes-1,quote.allocation.alignment,
                        GGML_BACKEND_MEMORY_REGION_PERSISTENT,nullptr) && ggml_backend_memory_arena_commit(short_arena.get()))) return;
            lease_ptr short_lease(ggml_backend_memory_arena_acquire(short_arena.get(),77),ggml_backend_memory_lease_free);
            auto short_params=params; short_params.bounded_cache_lease=short_lease.get();
            bool short_rejected=false;
            try {
                llama_kv_cache invalid(*model,model->hparams,cp.type_k,cp.type_v,false,true,true,1280,1,256,
                    model->hparams.n_swa,model->hparams.swa_type,nullptr,filter,no_reuse,nullptr,"",&short_params);
            } catch (const std::runtime_error & error) {
                short_rejected=std::string(error.what()) == "bounded SWA cache exceeds its measured grant";
            }
            if (!t.assert_true(short_rejected)) return;
            try {
                llama_kv_cache cache(*model,model->hparams,cp.type_k,cp.type_v,false,true,true,1280,1,256,
                    model->hparams.n_swa,model->hparams.swa_type,nullptr,filter,no_reuse,nullptr,"",&params);
                t.assert_true(local->get_layer_ids() == cache.get_layer_ids());
                t.assert_true(cache.get_k_storage(0)->data == local->get_k_storage(0)->data);
            } catch (const std::exception & error) {
                t.out << "no-op target reuse: " << error.what() << '\n';
                t.assert_true(false); return;
            }
            owner=std::make_unique<llama_memory_bounded_kv>(quote,
                llama_memory_resource{77,1,LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL,llama_memory_content::preserve});
            if (!t.assert_true(owner->bind(lease.get()))) return;
            llama_memory_execution_plan plan;
            plan.domains.push_back({1,LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL,LLAMA_MEMORY_CAPABILITY_BUFFER_VIEWS});
            plan.stages={{1,{},{}},{2,{1},{}}};
            if (!t.assert_true(owner->register_resources(plan,{1,2}))) return;
            for (const auto & stage : plan.stages) t.assert_equal(quote.allocation.bytes,stage.requirements.front().size_min);
        }
        const auto words=common_tokenize(context.get(),"A forest has trees, birds, and flowing water. ",true);
        if (!t.assert_true(!words.empty())) return;
        uint64_t ticket=owner ? owner->begin_write(owner->identity()) : 0;
        if (owner && !t.assert_true(ticket != 0)) return;
        common_batch batch(context.get());
        for (int first=0;first<1333;first+=256) {
            batch.clear();
            const int count=std::min(256,1333-first);
            for (int i=0;i<count;++i) batch.add(words[size_t(first+i)%words.size()],first+i,0,i == count-1);
            if (!t.assert_equal(0,llama_process(context.get(),LLAMA_PROCESS_TYPE_DECODE,batch.get()))) return;
        }
        llama_synchronize(context.get());
        if (owner) {
            if (!t.assert_true(owner->finish_write(ticket,true))) return;
            std::vector<std::vector<uint8_t>> bytes;
            for (auto id : local->get_layer_ids()) for (auto * tensor : {local->get_k_storage(int32_t(id)),local->get_v_storage(int32_t(id))}) {
                bytes.emplace_back(ggml_nbytes(tensor));
                ggml_backend_tensor_get(tensor,bytes.back().data(),0,bytes.back().size());
            }
            std::vector<llama_pos> positions;
            for (uint32_t cell=0;cell<local->get_size();++cell) positions.push_back(
                local->get_cells(0).is_empty(cell) ? -1 : local->get_cells(0).pos_get(cell));
            if (!t.assert_true(owner->bind(lease.get()))) return;
            size_t index=0;
            for (auto id : local->get_layer_ids()) for (auto * tensor : {local->get_k_storage(int32_t(id)),local->get_v_storage(int32_t(id))}) {
                std::vector<uint8_t> after(ggml_nbytes(tensor));
                ggml_backend_tensor_get(tensor,after.data(),0,after.size());
                t.assert_true(bytes[index++] == after);
            }
            for (uint32_t cell=0;cell<local->get_size();++cell) t.assert_equal(positions[cell],
                local->get_cells(0).is_empty(cell) ? -1 : local->get_cells(0).pos_get(cell));
            t.assert_true(bool(owner->read(owner->identity())));
            {
                auto source=owner->read(owner->identity());
                ggml_context_ptr metadata(ggml_init({4096,nullptr,true}));
                ggml_tensor * k=nullptr,* v=nullptr;
                auto * native_k=local->get_k_storage(58),* native_v=local->get_v_storage(58);
                if (!t.assert_true(source.bind_layer_views(metadata.get(),58,native_k,native_v,k,v))) return;
                t.assert_true(k != native_k && v != native_v && k->view_src == native_k && v->view_src == native_v);
                t.assert_true(k->data == native_k->data && v->data == native_v->data);
                t.assert_equal(int64_t(4096),k->ne[0]); t.assert_equal(int64_t(1280),k->ne[1]);
                t.assert_equal(uint64_t(0),owner->begin_write(owner->identity()));
                auto * saved_k=k; auto * saved_v=v;
                t.assert_true(!source.bind_layer_views(metadata.get(),58,local->get_k_storage(57),native_v,k,v));
                t.assert_true(k == saved_k && v == saved_v);
                for (auto item : {std::pair{k,bytes.size()-2},std::pair{v,bytes.size()-1}}) {
                    std::vector<uint8_t> copy(ggml_nbytes(item.first));
                    ggml_backend_tensor_get(item.first,copy.data(),0,copy.size());
                    t.assert_true(copy == bytes[item.second]);
                }
            }
        }
        const size_t vocab=size_t(llama_vocab_n_tokens(llama_model_get_vocab(model)));
        for (int step=0;step<8;++step) {
            auto * logits=llama_get_logits_ith(context.get(),-1);
            if (!t.assert_true(logits != nullptr)) return;
            controls[borrowed].emplace_back(logits,logits+vocab);
            const auto token=llama_token(std::max_element(logits,logits+vocab)-logits);
            batch.clear(); batch.add(token,1333+step,0,true);
            ticket=owner ? owner->begin_write(owner->identity()) : 0;
            if (owner && !t.assert_true(ticket != 0)) return;
            if (!t.assert_equal(0,llama_process(context.get(),LLAMA_PROCESS_TYPE_DECODE,batch.get()))) return;
            llama_synchronize(context.get());
            if (owner && !t.assert_true(owner->finish_write(ticket,true))) return;
        }
    }
    if (!t.assert_equal(controls[0].size(),controls[1].size())) return;
    for (size_t row=0;row<controls[0].size();++row) t.assert_true(std::memcmp(controls[0][row].data(),
        controls[1][row].data(),controls[0][row].size()*sizeof(float)) == 0);
    auto full=llama_context_default_params();
    full.n_ctx=2048; full.n_batch=full.n_ubatch=256; full.n_seq_max=1;
    full.type_k=GGML_TYPE_Q8_0; full.type_v=GGML_TYPE_Q4_0; full.swa_full=true;
    full.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
    context_ptr context(llama_init_from_model(model,full),llama_free);
    if (!t.assert_true(bool(context))) return;
    auto * local=llama_memory_kv_resources_get(llama_get_memory(context.get())).bounded;
    if (!t.assert_true(local != nullptr)) return;
    t.assert_equal(uint32_t(2048),local->get_size());
    llama_memory_bounded_kv_layout unchanged;
    unchanged.allocation.bytes=17;
    llama_memory_params params{}; params.type_k=full.type_k; params.type_v=full.type_v; params.swa_full=true;
    t.assert_true(!llama_kv_cache_iswa::bounded_storage_requirements(model->hparams,params,context->get_cparams(),
        ggml_backend_buffer_get_type(local->get_k_storage(0)->buffer),{607,1},unchanged));
    t.assert_equal(size_t(17),unchanged.allocation.bytes);
}

int main(int argc,char ** argv) {
    int baseline=0;
    bool exact_options = false, cache_options = false;
    for (int i = 1; i < argc; ++i) {
        mtp_bit_exact_control |= !std::strcmp(argv[i], "--mtp-bit-exact");
        exact_options |= !std::strncmp(argv[i], "--exact-", 8);
        cache_options |= !std::strncmp(argv[i], "--test-cache-", 13);
    }
    if (((mtp_bit_exact_control || cache_options) && (argc < 3 || std::strcmp(argv[1], "--model"))) ||
            (exact_options && !mtp_bit_exact_control)) {
        std::fprintf(stderr, "strict MTP usage: %s --model GGUF --mtp-bit-exact [--test-cache-k TYPE --test-cache-v TYPE --exact-prefill N --exact-rounds N --exact-pool-mib N --exact-batch-size N --exact-length N --exact-prefill-file FILE]\n", argv[0]);
        return 1;
    }
    for (int i=3;i<argc;++i) {
        if (!std::strcmp(argv[i],"--pascal")) baseline=610;
        if (!std::strcmp(argv[i],"--sm75")) baseline=750;
        if (!std::strcmp(argv[i],"--sm86")) baseline=860;
        if (!std::strcmp(argv[i],"--sm89")) baseline=890;
        if (!std::strcmp(argv[i],"--sm120")) baseline=1200;
    }
#ifdef KV_STREAM_TILE_NATIVE_TEST
    if (baseline && !kv_stream_test_is_arch_only(baseline)) return 77;
#else
    if (baseline) return 77;
#endif
    testing t;
    t.test("bit_exact_control_framing_preserves_profile_records_and_transaction",[](testing & t) {
        const std::array<uint64_t,16> profile{1,8192,2047,3,7,2,1,256,8};
        const std::array<float,4> values{1.0f,-0.0f,0.0f,-2.0f};
        kv_stream_bit_oracle original;
        original.bytes("values",values.data(),sizeof(values));
        original.bytes("empty",nullptr,0);
        std::unique_ptr<FILE,kv_stream_exact_file_free> file(std::tmpfile());
        if (!t.assert_true(bool(file))) return;
        original.save_control(file.get(),profile);
        if (!t.assert_equal(0,std::fseek(file.get(),0,SEEK_END))) return;
        const auto size=std::ftell(file.get());
        if (!t.assert_true(size>0) || !t.assert_equal(0,std::fseek(file.get(),0,SEEK_SET))) return;
        std::vector<uint8_t> frame(size);
        if (!t.assert_equal(frame.size(),std::fread(frame.data(),1,frame.size(),file.get()))) return;
        const auto source=[&](const std::vector<uint8_t> & bytes) {
            std::unique_ptr<FILE,kv_stream_exact_file_free> result(std::tmpfile());
            if (!result || (!bytes.empty() && std::fwrite(bytes.data(),1,bytes.size(),result.get())!=bytes.size()) ||
                    std::fseek(result.get(),0,SEEK_SET)) throw std::runtime_error("cannot make control fixture");
            return result;
        };
        {
            auto input=source(frame);
            kv_stream_bit_oracle loaded;
            loaded.load_control(input.get(),profile);
            t.assert_equal(original.records(),loaded.records());
            t.assert_equal(original.byte_count(),loaded.byte_count());
            loaded.compare();
            loaded.bytes("values",values.data(),sizeof(values));
            loaded.bytes("empty",nullptr,0);
            loaded.finish();
        }
        const auto reject=[&](const std::vector<uint8_t> & bytes,const std::array<uint64_t,16> & expected_profile) {
            auto input=source(bytes);
            kv_stream_bit_oracle loaded;
            loaded.bytes("unchanged",values.data(),sizeof(values));
            bool rejected=false;
            try { loaded.load_control(input.get(),expected_profile); }
            catch (const std::runtime_error &) { rejected=true; }
            t.assert_true("malformed control rejects transactionally",rejected);
            loaded.compare();loaded.bytes("unchanged",values.data(),sizeof(values));loaded.finish();
        };
        auto wrong_profile=profile;wrong_profile[1]++;
        reject(frame,wrong_profile);
        for (size_t bytes=0;bytes<frame.size();++bytes)
            reject(std::vector<uint8_t>(frame.begin(),frame.begin()+bytes),profile);
        auto bad=frame;bad.push_back(0);reject(bad,profile);
        for (size_t offset : {size_t(0),size_t(136),size_t(144),size_t(152),size_t(160)}) {
            bad=frame;bad[offset]=255;reject(bad,profile);
        }
        for (bool corrupt_payload : {false,true}) {
            auto input=source(frame);
            kv_stream_bit_oracle loaded;loaded.load_control(input.get(),profile);loaded.compare();
            auto changed=values;changed[1]=0.0f;
            bool rejected=false;
            try { loaded.bytes(corrupt_payload ? "values" : "wrong-label",changed.data(),sizeof(changed)); }
            catch (const std::runtime_error &) { rejected=true; }
            t.assert_true("label and signed-zero mismatches stay exact",rejected);
        }
    });
    t.test("Gemma_assistant_split_oracle_keeps_input_and_all_vectors_exact",[](testing & t) {
        std::array<std::unique_ptr<FILE,kv_stream_exact_file_free>,3> files;
        std::array<FILE *,3> handles{};
        for (size_t i=0;i<files.size();++i) {
            files[i].reset(std::tmpfile());handles[i]=files[i].get();
            if (!t.assert_true(handles[i]!=nullptr)) return;
        }
        std::vector<float> image(8,0.125f);image[1]=-0.0f;
        std::vector<std::vector<float>> logits(19,std::vector<float>(8,0.5f)),hidden(18,std::vector<float>(4,-0.0f));
        gemma_assistant_oracle(handles,true,4096,1336,image,4,8,logits,hidden);
        std::vector<std::vector<float>> actual_logits,actual_hidden;
        gemma_assistant_oracle(handles,false,4096,1336,image,4,8,actual_logits,actual_hidden);
        t.assert_true(actual_logits==logits && actual_hidden.size()==hidden.size() &&
            !std::memcmp(actual_hidden[0].data(),hidden[0].data(),4*sizeof(float)));
        image[0]=0.0f;
        bool rejected=false;
        try { gemma_assistant_oracle(handles,false,4096,1336,image,4,8,actual_logits,actual_hidden); }
        catch (const std::runtime_error &) { rejected=true; }
        t.assert_true("same-size wrong image cannot reuse an oracle",rejected);
        image[0]=0.125f;
        if (!t.assert_equal(0,std::fseek(handles[2],0,SEEK_END)) || !t.assert_true(std::fputc(0,handles[2])!=EOF)) return;
        const auto unchanged_logits=actual_logits,unchanged_hidden=actual_hidden;
        rejected=false;
        try { gemma_assistant_oracle(handles,false,4096,1336,image,4,8,actual_logits,actual_hidden); }
        catch (const std::runtime_error &) { rejected=true; }
        t.assert_true("bad hidden frame cannot publish partially read logits",rejected && actual_logits==unchanged_logits && actual_hidden==unchanged_hidden);
    });
    t.test("Gemma_image_reference_validates_extent_and_preserves_float_bits",[](testing & t) {
        std::unique_ptr<FILE,kv_stream_exact_file_free> file(std::tmpfile());
        if (!t.assert_true(bool(file))) return;
        std::vector<float> expected(64,0.125f);expected[1]=-0.0f;
        if (!t.assert_equal(expected.size(),std::fwrite(expected.data(),sizeof(float),expected.size(),file.get()))) return;
        const auto values=gemma_image_reference(file.get(),32);
        t.assert_true(values.size()==expected.size() && !std::memcmp(values.data(),expected.data(),values.size()*sizeof(float)));
        auto reject=[&](FILE * source,size_t width) {
            bool rejected=false;
            try { gemma_image_reference(source,width); }
            catch (const std::runtime_error &) { rejected=true; }
            t.assert_true("invalid image fixture rejects before allocation",rejected);
        };
        reject(nullptr,32);reject(file.get(),0);reject(file.get(),SIZE_MAX);
        for (long bytes : {0L,1L,127L,129L,1121L*32*4,32L*1048576+1}) {
            std::unique_ptr<FILE,kv_stream_exact_file_free> bad(std::tmpfile());
            if (!t.assert_true(bool(bad))) return;
            if (bytes && (!t.assert_equal(0,std::fseek(bad.get(),bytes-1,SEEK_SET)) || !t.assert_true(std::fputc(0,bad.get())!=EOF))) return;
            reject(bad.get(),32);
        }
        for (float bad_value : {std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
            std::unique_ptr<FILE,kv_stream_exact_file_free> bad(std::tmpfile());
            expected[0]=bad_value;
            if (!t.assert_true(bool(bad)) || !t.assert_equal(expected.size(),std::fwrite(expected.data(),sizeof(float),expected.size(),bad.get()))) return;
            reject(bad.get(),32);
        }
    });
    t.test("memory_resources_resolve_cache_roles_without_hybrid_casts", [](testing & t) {
        std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_GEMMA4, llama_model_default_params()));
        auto & hp = model->hparams;
        hp = {};
        hp.n_layer_all = 2;
        hp.n_embd = 64;
        hp.n_head_arr.fill(2);
        hp.n_head_kv_arr.fill(1);
        hp.n_embd_head_k_full = hp.n_embd_head_v_full = 32;
        hp.n_embd_head_k_swa = hp.n_embd_head_v_swa = 32;
        hp.n_swa = 128;
        hp.swa_type = LLAMA_SWA_TYPE_STANDARD;
        hp.is_swa_impl[0] = true;
        llama_kv_cache_iswa local(*model, GGML_TYPE_F32, GGML_TYPE_F32, false, false, false,
                true, 256, 1, 64, 1, nullptr, nullptr, nullptr, nullptr);
        const auto view = llama_memory_kv_resources_get(&local);
        t.assert_true(view.full == local.get_base());
        t.assert_true(view.bounded == local.get_swa());
        t.assert_true(view.recurrent == nullptr);
        auto ctx = local.init_full();
        const auto * iswa = dynamic_cast<llama_kv_cache_iswa_context *>(ctx.get());
        t.assert_true(iswa && llama_memory_kv_context_get(ctx.get()) == iswa->get_base());
        t.assert_true(llama_memory_kv_resources_get(view.full).full == view.full);
        auto full_ctx = view.full->init_full();
        t.assert_true(llama_memory_kv_context_get(full_ctx.get()) == dynamic_cast<llama_kv_cache_context *>(full_ctx.get()));
        t.assert_true(!llama_memory_kv_resources_get(nullptr).full);
        t.assert_true(!llama_memory_kv_context_get(nullptr));

        const auto skip = [](int32_t) { return false; };
        llama_memory_hybrid hybrid(*model, GGML_TYPE_F32, GGML_TYPE_F32, false, 256, 1, 0,
                LLAMA_SWA_TYPE_NONE, GGML_TYPE_F32, GGML_TYPE_F32, 1, 1, 0,
                false, true, skip, skip);
        const auto hybrid_view = llama_memory_kv_resources_get(&hybrid);
        t.assert_true(hybrid_view.full == hybrid.get_mem_attn());
        t.assert_true(hybrid_view.recurrent == hybrid.get_mem_recr());
        t.assert_true(hybrid_view.bounded == nullptr);
        auto hybrid_ctx = hybrid.init_full();
        const auto * hybrid_cast = dynamic_cast<llama_memory_hybrid_context *>(hybrid_ctx.get());
        t.assert_true(hybrid_cast && llama_memory_kv_context_get(hybrid_ctx.get()) == hybrid_cast->get_attn());
        t.assert_true(!llama_memory_kv_resources_get(hybrid.get_mem_recr()).full);
        auto recurrent_ctx = hybrid.get_mem_recr()->init_full();
        t.assert_true(!llama_memory_kv_context_get(recurrent_ctx.get()));
    });
    test_kv_stream_bit_exact_comparator(t);
    t.test("image_oracle_roundtrip_preserves_bits_and_rejects_wrong_profile",[](testing & t) {
        kv_stream_image_input image;
        image.width=4; image.prefix={1,2}; image.suffix={3}; image.next_position=4;
        image.values={0.0f,-0.0f,0.25f,0.5f,0.75f,1.0f,1.25f,1.5f};
        image.positions={{{2,2,2,0}},{{2,3,2,0}}};
        std::unique_ptr<FILE,kv_stream_exact_file_free> file(std::tmpfile());
        if (!t.assert_true(bool(file))) return;
        image.save(file.get()); std::rewind(file.get());
        const auto actual=kv_stream_image_input::load(file.get(),4);
        t.assert_true(actual.prefix==image.prefix && actual.suffix==image.suffix && actual.positions==image.positions);
        t.assert_true(kv_stream_bytes_equal(actual.values.data(),actual.values.size()*sizeof(float),image.values.data(),image.values.size()*sizeof(float)));
        std::rewind(file.get()); bool rejected=false;
        try { auto invalid=kv_stream_image_input::load(file.get(),8); } catch (const std::runtime_error &) { rejected=true; }
        t.assert_true("an image from another embedding geometry is rejected",rejected);
        image.positions[0][0]=-1; rejected=false;
        try { image.save(file.get()); } catch (const std::runtime_error &) { rejected=true; }
        t.assert_true("negative image positions reject before use",rejected);
    });

    t.test("independent_dflash_quota_fixture_declares_a_bounded_child", [&](testing & t) {
        auto target=llama_context_default_params();
        target.n_ctx=4096; target.n_rs_seq=7;
        target.shared_device_memory_bytes=size_t(512)*1024*1024;
        target.type_k=GGML_TYPE_Q8_0; target.type_v=GGML_TYPE_Q4_0;
        const auto child=dflash_independent_child_params(target,nullptr);
        t.assert_true("independent child must not inherit full SWA from native defaults",!child.swa_full);
        t.assert_equal(uint32_t(4096),child.n_ctx);
        t.assert_equal(uint32_t(0),child.n_rs_seq);
        t.assert_equal(size_t(0),child.shared_device_memory_bytes);
        t.assert_equal(uint32_t(4),child.n_outputs_max);
        t.assert_equal(uint32_t(4),child.n_outputs_max_per_seq);
        t.assert_equal(GGML_TYPE_Q8_0,child.type_k);
        t.assert_equal(GGML_TYPE_Q4_0,child.type_v);
        t.assert_equal(uint32_t(7),target.n_rs_seq);
        t.assert_equal(size_t(512)*1024*1024,target.shared_device_memory_bytes);
    });
    t.test("short_dflash_profiles_keep_native_controls_physically_bounded", [&](testing & t) {
        const auto original = dflash_exact_profiles(false, false, false, false, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);
        t.assert_equal(size_t(16000), original.back()[1]);
        t.assert_equal(size_t(16384), original.back()[2]);
        const ggml_type types[] = {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q8_0,
            GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1};
        for (auto key : types) for (auto value : types) {
            const auto profiles = dflash_exact_profiles(false, false, false, true, key, value);
            t.assert_equal(size_t(4), profiles.size());
            const size_t maximum = ggml_row_size(key,1024)+ggml_row_size(value,1024) > 4096 ? 4096 : 8192;
            for (const auto & profile : profiles) {
                t.assert_true(profile[2] <= maximum && profile[1]+64 <= profile[2]);
                t.assert_true(profile[0] == 1 || profile[0] == 3 || profile[0] == 7);
            }
            t.assert_equal(size_t(2047), profiles.front()[1]);
            t.assert_equal(maximum-192, profiles.back()[1]);
        }
    });
    t.test("selector_lattice_permutations_preserve_scores_but_one_bit_changes_do_not",[](testing & t) {
        std::vector<float> a{0,0,0,0,0,0,4,2,0.2f,0.4f,0.2f,0.4f};
        auto b=a;
        b[6]=2; b[7]=4; b[8]=b[10]=0.4f; b[9]=b[11]=0.2f;
        const auto canonical=dflash_exact_lattice(a.data(),2,6,2,16);
        t.assert_true(canonical == dflash_exact_lattice(b.data(),2,6,2,16));
        uint32_t bits; std::memcpy(&bits,&b[8],sizeof(bits)); bits^=1;
        std::memcpy(&b[8],&bits,sizeof(bits)); b[10]=b[8];
        t.assert_true(canonical != dflash_exact_lattice(b.data(),2,6,2,16));
        a.insert(a.end(),{3,5,0.1f,0.2f,0.3f,0.4f});
        b={0,0,0,0,0,0,2,4,0.4f,0.2f,0.4f,0.2f,5,3,0.4f,0.3f,0.2f,0.1f};
        t.assert_true(dflash_exact_lattice(a.data(),3,6,2,16) == dflash_exact_lattice(b.data(),3,6,2,16));
    });
    t.test("gemma_logit_files_validate_profile_framing_and_exact_payload", [&](testing & t) {
        const std::array<uint64_t,8> profile{0x474d344c4f474954ull,1,16,4,2,3,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0};
        std::vector<std::vector<float>> source{{0.0f,-0.0f,1.5f},{-2.0f,3.0f,4.0f}}, result;
        std::unique_ptr<FILE,kv_stream_exact_file_free> file(std::tmpfile());
        if (!t.assert_true(bool(file))) return;
        gemma_logits_file(file.get(),true,profile,source);
        std::rewind(file.get());
        gemma_logits_file(file.get(),false,profile,result);
        t.assert_equal(source.size(),result.size());
        for (size_t row=0;row<source.size();++row) t.assert_true(kv_stream_bytes_equal(
            source[row].data(),source[row].size()*sizeof(float),result[row].data(),result[row].size()*sizeof(float)));
        const auto unchanged=result;
        const auto rejects=[&](const std::array<uint64_t,8> & expected) {
            std::rewind(file.get());
            try { gemma_logits_file(file.get(),false,expected,result); }
            catch (const std::runtime_error &) { return true; }
            return false;
        };
        auto wrong=profile; wrong[2]+=1;
        t.assert_true(rejects(wrong) && result == unchanged);
        wrong=profile; wrong[5]=UINT64_MAX;
        t.assert_true(rejects(wrong) && result == unchanged);
        const unsigned char extra=7;
        std::fseek(file.get(),0,SEEK_END); std::fwrite(&extra,1,1,file.get()); std::fflush(file.get());
        t.assert_true(rejects(profile) && result == unchanged);
        file.reset(std::tmpfile());
        if (!t.assert_true(bool(file))) return;
        std::fwrite(profile.data(),1,sizeof(profile),file.get()); std::fflush(file.get());
        t.assert_true(rejects(profile) && result == unchanged);
        file.reset(std::tmpfile());
        if (!t.assert_true(bool(file))) return;
        gemma_logits_file(file.get(),true,profile,source);
        std::rewind(file.get()); std::fwrite(&extra,1,1,file.get()); std::fflush(file.get());
        t.assert_true(rejects(profile) && result == unchanged);
    });
    t.test("gemma_test_arena_override_is_explicit_and_bounded", [](testing & t) {
        t.assert_equal(size_t(1024)*1048576,gemma_test_arena_bytes(nullptr));
        t.assert_equal(size_t(768)*1048576,gemma_test_arena_bytes("768"));
        t.assert_equal(size_t(1)*1048576,gemma_test_arena_bytes("1"));
        t.assert_equal(size_t(4095)*1048576,gemma_test_arena_bytes("4095"));
        for (const char * invalid : {"", "0", "-1", "+768", " 768", "768 ", "768x", "4096", "18446744073709551616"}) {
            bool rejected=false;
            try { (void)gemma_test_arena_bytes(invalid); }
            catch (const std::invalid_argument &) { rejected=true; }
            t.assert_true(rejected);
        }
    });
    t.test("gemma_image_arena_override_preserves_the_normal_profile", [](testing & t) {
        t.assert_equal(size_t(2048)*1048576,gemma_image_test_arena_bytes(false,gemma_test_arena_bytes(nullptr)));
        t.assert_equal(size_t(2048)*1048576,gemma_image_test_arena_bytes(false,gemma_test_arena_bytes("1536")));
        t.assert_equal(size_t(1536)*1048576,gemma_image_test_arena_bytes(true,gemma_test_arena_bytes("1536")));
        t.assert_equal(size_t(1024)*1048576,gemma_image_test_arena_bytes(true,gemma_test_arena_bytes("1024")));
    });
    t.test("gemma_image_fixture_declares_every_requested_output_not_every_input", [](testing & t) {
        for (uint32_t rows : {256u,1280u}) for (bool verify : {false,true}) {
            auto cp=llama_context_default_params();cp.n_batch=cp.n_ubatch=rows;
            gemma_image_test_outputs(cp,verify);
            t.assert_equal(verify ? 4u : 1u,cp.n_outputs_max);
            t.assert_equal(verify ? 4u : 1u,cp.n_outputs_max_per_seq);
            t.assert_equal(rows,cp.n_batch);
            t.assert_equal(rows,cp.n_ubatch);
        }
    });
    t.test("streaming_is_disabled_by_default", [&](testing & t) {
        const auto defaults = llama_context_default_params();
        t.assert_equal(size_t(0),defaults.kv_stream_pool_bytes);
        t.assert_equal(size_t(0),defaults.shared_device_memory_bytes);
        t.assert_equal(uint32_t(0), defaults.kv_stream_auxiliary_layers);
    });
    t.test("streamed_verification_width_is_independent_of_recurrent_storage", [&](testing & t) {
        t.assert_equal(uint32_t(0),llama_context_default_params().kv_stream_verify_max);
        llama_cparams params{};
        params.n_seq_max=1;
        t.assert_equal(uint32_t(1),params.kv_stream_decode_rows());
        for (uint32_t rows=1;rows<=4;++rows) {
            params.kv_stream_verify_max=rows;
            params.n_rs_seq=0;
            t.assert_equal(rows,params.kv_stream_decode_rows());
            params.n_rs_seq=3;
            t.assert_equal(rows,params.kv_stream_decode_rows());
        }
        params.kv_stream_verify_max=0;
        t.assert_equal(uint32_t(4),params.kv_stream_decode_rows());
        params.n_rs_seq=0; params.kv_stream_auxiliary_layers=1;
        t.assert_equal(uint32_t(4),params.kv_stream_decode_rows());
    });
    t.test("serial_prediction_catalog_is_independent_of_target_rollback_catalog", [&](testing & t) {
        for (uint32_t predictions=1;predictions<=7;++predictions) {
            for (uint32_t continuation=1;continuation<=7;++continuation) {
                llama_cparams target{},child{};
                target.n_ctx=child.n_ctx=4096;
                target.n_ubatch=child.n_ubatch=256;
                target.n_seq_max=child.n_seq_max=1;
                target.n_rs_seq=std::max(predictions,continuation);
                child.serial_block_rows=predictions+1;
                t.assert_equal(std::max(predictions,continuation)+1,target.kv_stream_decode_rows());
                t.assert_equal(uint32_t(0),child.n_rs_seq);
                t.assert_equal(predictions+1,child.serial_graph_reserve_rows(false));
                t.assert_equal(predictions+1,child.serial_graph_reserve_rows(true));
                t.assert_equal(uint32_t(256),target.serial_graph_reserve_rows(false));
            }
        }
    });
    if (argc < 3 || std::strcmp(argv[1],"--model")) return t.summary();
    for (int i = 3; i < argc; ++i) {
        if (!std::strcmp(argv[i],"--gemma-test-arena-mib")) {
            if (++i == argc) { std::fprintf(stderr,"missing Gemma test arena MiB\n"); return 1; }
            try { gemma_test_arena_size=gemma_test_arena_bytes(argv[i]); }
            catch (const std::invalid_argument & error) { std::fprintf(stderr,"%s\n",error.what()); return 1; }
            gemma_test_arena_explicit=true;
            continue;
        }
        if (!std::strcmp(argv[i], "--test-cache-k") || !std::strcmp(argv[i], "--test-cache-v")) {
            const bool key = !std::strcmp(argv[i], "--test-cache-k");
            if (++i == argc) {
                std::fprintf(stderr, "missing test cache type\n");
                return 1;
            }
            try {
                const auto type = kv_stream_exact_cache_type(argv[i]);
                (key ? test_cache_k : test_cache_v) = type;
                test_cache_explicit = true;
            } catch (const std::exception & error) {
                std::fprintf(stderr, "%s\n", error.what());
                return 1;
            }
            continue;
        }
        if (!std::strcmp(argv[i], "--dflash-native") && i+1 < argc) dflash_native_model = argv[++i];
        if (!std::strcmp(argv[i], "--dflash-shared-workspace")) dflash_shared_workspace_control = true;
        if (!std::strcmp(argv[i], "--dflash-bit-exact")) dflash_exact_control = true;
        if (!std::strcmp(argv[i], "--dflash-short")) dflash_short_control = dflash_exact_control = true;
        if (!std::strcmp(argv[i], "--dflash-independent-rows")) {
            dflash_independent_rows = dflash_short_control = dflash_exact_control = true;
        }
        if (!std::strcmp(argv[i], "--dflash-independent-ubatch8")) {
            dflash_independent_small_batch = dflash_independent_rows = dflash_short_control = dflash_exact_control = true;
        }
        if (!std::strcmp(argv[i], "--dflash-independent-quotas")) dflash_independent_quotas = true;
        if (!std::strcmp(argv[i], "--dflash-prefetch-hint")) dflash_prefetch_hint = true;
        if (!std::strcmp(argv[i],"--dflash-vision-image") && i+1<argc) dflash_vision_image=argv[++i];
        if ((!std::strcmp(argv[i],"--dflash-vision-export") || !std::strcmp(argv[i],"--dflash-vision-import") ||
                !std::strcmp(argv[i],"--dflash-vision-native-check")) && i+1<argc) {
            dflash_vision_writing=!std::strcmp(argv[i],"--dflash-vision-export");
            dflash_vision_native_check=!std::strcmp(argv[i],"--dflash-vision-native-check");
            dflash_vision_control=argv[++i];
        }
        if (!std::strcmp(argv[i],"--dflash-vision-ngram") && i+1<argc) {
            const auto depth=std::stoul(argv[++i]);if (depth<1 || depth>7) return 1;dflash_vision_ngram=uint32_t(depth);
        }
        if (!std::strcmp(argv[i],"--dflash-vision-depth") && i+1<argc) {
            const auto depth=std::stoul(argv[++i]);if (depth<1 || depth>7) return 1;dflash_vision_depth=uint32_t(depth);
        }
        if (!std::strcmp(argv[i], "--dflash-vision-readiness")) dflash_vision_readiness = true;
        if (!std::strcmp(argv[i], "--dflash-vision-microbatch") && i+1<argc) {
            const auto value=std::stoul(argv[++i]);
            if (value<8 || value>256) return 1;
            dflash_vision_microbatch=uint32_t(value);
        }
        if (!std::strcmp(argv[i], "--dflash-stream-control")) dflash_stream_control = dflash_mixed_all_widths = dflash_mixed_switching = true;
        if (!std::strcmp(argv[i], "--dflash-full-prediction")) dflash_full_prediction = true;
        if (!std::strcmp(argv[i], "--dflash-native-f3")) dflash_native_f3 = dflash_short_control = dflash_exact_control = true;
        if (!std::strcmp(argv[i], "--dflash-mixed-switching")) dflash_mixed_switching = true;
        if (!std::strcmp(argv[i], "--dflash-mixed-all-widths")) dflash_mixed_all_widths = dflash_mixed_switching = true;
        if (!std::strcmp(argv[i], "--dflash-mixed-recovery")) dflash_mixed_recovery = true;
        if (!std::strcmp(argv[i], "--dflash-lifecycle")) dflash_exact_control = true;
        if (!std::strcmp(argv[i], "--dflash-context-end")) dflash_exact_control = true;
        if (!std::strcmp(argv[i], "--dflash-rejection-long")) dflash_exact_control = true;
        f16_control = f16_control || !std::strcmp(argv[i],"--f16");
        resident_control = resident_control || !std::strcmp(argv[i],"--resident");
        shared_budget_control = shared_budget_control || !std::strcmp(argv[i],"--shared-budget");
        suspend_control = suspend_control || !std::strcmp(argv[i],"--suspend-only");
        suspend_resident_only = suspend_resident_only || !std::strcmp(argv[i],"--suspend-resident-only");
        suspend_control = suspend_control || suspend_resident_only;
        resume_control = resume_control || !std::strcmp(argv[i],"--resume-only");
        resume_resident_only = resume_resident_only || !std::strcmp(argv[i],"--resume-resident-only");
        resume_control = resume_control || resume_resident_only;
        trace_control = trace_control || !std::strcmp(argv[i],"--trace");
        auxiliary_control = auxiliary_control || !std::strcmp(argv[i],"--auxiliary-only");
        embedded_mtp_control = embedded_mtp_control || !std::strcmp(argv[i],"--embedded-mtp-pair");
        target_tg3_control = target_tg3_control || !std::strcmp(argv[i],"--target-tg3-only");
        target_stream_tg4_control = target_stream_tg4_control || !std::strcmp(argv[i],"--target-stream-tg4");
        target_with_stock_draft = target_with_stock_draft || !std::strcmp(argv[i],"--target-with-stock-draft");
        mtp_memory_probe = mtp_memory_probe || !std::strcmp(argv[i],"--mtp-memory-probe");
        mtp_sustained_control = mtp_sustained_control || !std::strcmp(argv[i],"--embedded-mtp-sustained");
        mtp_sustained_rs_control = mtp_sustained_rs_control || !std::strcmp(argv[i],"--embedded-mtp-sustained-rs");
        mtp_sustained_control = mtp_sustained_control || mtp_sustained_rs_control;
        target_drift_control = target_drift_control || !std::strcmp(argv[i],"--mtp-target-drift");
        target_drift_rs_control = target_drift_rs_control || !std::strcmp(argv[i],"--mtp-target-drift-rs");
        target_drift_control = target_drift_control || target_drift_rs_control;
        rollback_probe = rollback_probe || !std::strcmp(argv[i],"--mtp-rs-rollback-probe");
        wide_rollback_control = wide_rollback_control || !std::strcmp(argv[i],"--target-rs-rollback");
        vision_mtp_handoff = vision_mtp_handoff || !std::strcmp(argv[i],"--vision-mtp-handoff");
        serial_workspace_growth = serial_workspace_growth || !std::strcmp(argv[i],"--serial-workspace-growth");
    }
    embedded_mtp_control |= vision_mtp_handoff;
    if (test_cache_explicit && !mtp_bit_exact_control && !dflash_exact_control && !dflash_mixed_switching && !vision_mtp_handoff && !dflash_vision_control) {
        std::fprintf(stderr, "test cache types require a numerical oracle or vision handoff profile\n");
        return 1;
    }
    ggml_backend_load_all(); llama_backend_init();
    ggml_backend_ptr baseline_backend;
#ifdef KV_STREAM_TILE_NATIVE_TEST
    if (baseline) {
        auto * device=ggml_backend_dev_by_name("CUDA0");
        if (!device) return 77;
        baseline_backend.reset(ggml_backend_dev_init(device,nullptr));
        if (!baseline_backend || kv_stream_test_compiled_cc(baseline_backend.get()) != baseline) return 77;
    }
    const int previous=baseline ? kv_stream_test_override_cc(baseline_backend.get(),baseline) : 0;
    struct restore {
        ggml_backend_t backend; int previous;
        ~restore() {if (backend) kv_stream_test_override_cc(backend,previous);}
    } restore_cc{baseline_backend.get(),previous};
#endif
    for (int i=3;i<argc;++i) if (!std::strcmp(argv[i],"--gemma-warmup-native-logits") || !std::strcmp(argv[i],"--gemma-warmup-streamed-logits")) {
        if (i+1==argc) { std::fprintf(stderr,"missing Gemma warmup logit path\n"); return 1; }
        const int mode=!std::strcmp(argv[i],"--gemma-warmup-native-logits") ? 0 : 1;
        t.test("Gemma_warmup_failure_and_recovery_matches_framed_native_control",
            [&](testing & t) { test_gemma_warmup_failures(t,argv[2],mode,argv[i+1]); });
        return t.summary();
    }
    if (std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--gemma-warmup-failures"); })) {
        t.test("Gemma_common_startup_rejects_failed_warmup_and_can_retry",
            [&](testing & t) { test_gemma_warmup_failures(t,argv[2]); });
        return t.summary();
    }
    for (int i=3;i<argc;++i) if (!std::strcmp(argv[i],"--gemma-common-startup")) {
        if (i+1 == argc) { std::fprintf(stderr,"missing Gemma assistant artifact\n"); return 1; }
        t.test("Gemma_common_startup_and_short_prefill_borrow_workspace",[&](testing & t) {
            test_gemma_common_startup(t,argv[2],argv[i+1]);
        });
        return t.summary();
    }
    auto mparams = llama_model_default_params(); mparams.n_gpu_layers = 999;
    const llama_model_tensor_buft_override cache_only_weights[]={{".*",ggml_backend_cpu_buffer_type()},{nullptr,nullptr}};
    const llama_model_tensor_buft_override staged_draft_weights[]={{"blk\\..*",ggml_backend_cpu_buffer_type()},{nullptr,nullptr}};
    const bool deferred_local_cache_only=std::any_of(argv+3,argv+argc,[](const char * arg) {
        return !std::strcmp(arg,"--gemma-deferred-local-cache-only") || !std::strcmp(arg,"--gemma-selected-local-cache-only");
    });
    const bool measurement_only=std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--measure-graphs-only"); });
    const bool measurement_gpu_weights=std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--measure-gpu-weights"); });
    const bool recurrent_cache_only=std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--qwen-current-cache-only"); });
    const bool group_dflash_rollback_metadata=std::any_of(argv+3,argv+argc,[](const char * arg) {return !std::strcmp(arg,"--group-dflash-rollback-metadata");});
    const bool group_dflash_metadata=std::any_of(argv+3,argv+argc,[](const char * arg) {
        return !std::strcmp(arg,"--group-dflash-metadata") || !std::strcmp(arg,"--group-dflash-quote") ||
            !std::strcmp(arg,"--group-dflash-bind") || !std::strcmp(arg,"--native-dflash-draft-stage") ||
            !std::strcmp(arg,"--native-dflash-draft-suffix") || !std::strcmp(arg,"--native-dflash-draft-repeat") ||
            !std::strcmp(arg,"--native-dflash-draft-widths") || !std::strcmp(arg,"--native-dflash-k4v-routes") || !std::strcmp(arg,"--group-vision-loan-only");
    });
    if ((measurement_only && !measurement_gpu_weights) || recurrent_cache_only || group_dflash_rollback_metadata || group_dflash_metadata || deferred_local_cache_only || std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--gemma-read-source-cache"); })) {
        // Cache-only oracles compute no weights; preserve device memory for their native cache/control graphs.
        mparams.tensor_buft_overrides=cache_only_weights;
    }
    if (std::any_of(argv+3,argv+argc,[](const char * arg) {
        return !std::strcmp(arg,"--native-dflash-draft-stage") || !std::strcmp(arg,"--native-dflash-draft-suffix") ||
            !std::strcmp(arg,"--native-dflash-draft-repeat") || !std::strcmp(arg,"--native-dflash-draft-widths") || !std::strcmp(arg,"--native-dflash-k4v-routes");
    }))
        mparams.tensor_buft_overrides=staged_draft_weights;
    const bool group_mtp_ring=std::any_of(argv+3,argv+argc,[](const char * arg) {return !std::strcmp(arg,"--group-mtp-ring-only");});
    const bool group_mtp_image_workspace=std::any_of(argv+3,argv+argc,[](const char * arg) {return !std::strcmp(arg,"--group-mtp-image-workspace");});
    const bool group_mtp_image=group_mtp_image_workspace || std::any_of(argv+3,argv+argc,[](const char * arg) {return !std::strcmp(arg,"--group-mtp-image-catchup");});
    const bool group_mtp_backend=std::any_of(argv+3,argv+argc,[](const char * arg) {return !std::strcmp(arg,"--group-mtp-backend-only");});
    const bool group_mtp_backend_resume=std::any_of(argv+3,argv+argc,[](const char * arg) {return !std::strcmp(arg,"--group-mtp-backend-resume-only");});
    const bool group_mtp_confidence=std::any_of(argv+3,argv+argc,[](const char * arg) {return !std::strcmp(arg,"--group-mtp-confidence-only");});
    const bool group_mtp_phase_guard=std::any_of(argv+3,argv+argc,[](const char * arg) {return !std::strcmp(arg,"--group-mtp-phase-guard-only");});
    const bool group_mtp_partial=group_mtp_backend_resume || std::any_of(argv+3,argv+argc,[](const char * arg) {return !std::strcmp(arg,"--group-mtp-partial-only");});
    const bool group_mtp_acceptance=std::any_of(argv+3,argv+argc,[](const char * arg) {return !std::strcmp(arg,"--group-mtp-acceptance-only");});
    const bool group_mtp_resume=group_mtp_confidence || group_mtp_acceptance || group_mtp_partial || group_mtp_phase_guard || std::any_of(argv+3,argv+argc,[](const char * arg) {return !std::strcmp(arg,"--group-mtp-resume-only");});
    const bool group_mtp_admission=std::any_of(argv+3,argv+argc,[](const char * arg) {return !std::strcmp(arg,"--group-mtp-admission-only");});
    const bool group_auxiliary_metadata=std::any_of(argv+3,argv+argc,[](const char * arg) {return !std::strcmp(arg,"--group-auxiliary-metadata-only");});
    mparams.load_mtp = group_mtp_image || group_mtp_backend || group_mtp_ring || group_mtp_resume || group_mtp_admission || group_auxiliary_metadata || embedded_mtp_control || mtp_memory_probe || mtp_sustained_control || target_drift_control || rollback_probe || serial_workspace_growth || mtp_bit_exact_control;
    for (int i=3;i<argc;++i) if (!std::strcmp(argv[i],"--group-common-mtp-startup") || !std::strcmp(argv[i],"--group-common-mtp-failure") ||
            !std::strcmp(argv[i],"--group-common-mtp-width1") || !std::strcmp(argv[i],"--group-common-mtp-width2")) {
        const char * assistant=i+1<argc ? argv[i+1] : nullptr;
        const bool failure=!std::strcmp(argv[i],"--group-common-mtp-failure");
        t.test("Common_model_only_startup_declares_both_group_MTP_roles_before_arena",[&](testing & t) {
            const uint32_t maximum=!std::strcmp(argv[i],"--group-common-mtp-width1") ? 1 :
                !std::strcmp(argv[i],"--group-common-mtp-width2") ? 2 : 3;
            test_group_common_mtp_startup(t,argv[2],assistant,failure,maximum);
        });
        return t.summary();
    }
    for (int i=3;i+1<argc;++i) if (!std::strcmp(argv[i],"--group-common-dflash-startup")) {
        t.test("Common_model_only_startup_declares_DFlash_and_K4V_without_private_contexts",[&](testing & t) {
            test_group_common_dflash_startup(t,argv[2],argv[i+1]);
        });return t.summary();
    }
    model_ptr model(llama_model_load_from_file(argv[2],mparams),llama_model_free);
    if (std::any_of(argv+3,argv+argc,[](const char * arg) {return !std::strcmp(arg,"--group-vision-loan-only");})) {
        t.test("Parked_group_vision_loans_share_one_parent_and_retained_grants_block_text",[&](testing & t) {test_group_vision_loans(t,model.get());});
        return t.summary();
    }

    for (int i=3;i+3<argc;++i) if (!std::strcmp(argv[i],"--native-dflash-k4v-target") ||
            !std::strcmp(argv[i],"--native-dflash-k4v-routes") || !std::strcmp(argv[i],"--group-dflash-k4v-routes")) {
        t.test("Per_slot_real_copy_hits_and_neural_fallback_keep_independent_source_histories",[&](testing & t) {
            test_dflash_k4v_routes(t,model.get(),argv[i+1],argv[i+2],argv[i+3],
                !std::strcmp(argv[i],"--native-dflash-k4v-target"),!std::strcmp(argv[i],"--group-dflash-k4v-routes"));
        });return t.summary();
    }
    for (int i=3;i+4<argc;++i) if (!std::strcmp(argv[i],"--native-dflash-target-verify")) {
        t.test("Staged_native_target_verifies_exact_draft_proposals_with_matching_prefix",[&](testing & t) {
            test_native_dflash_target_verify(t,model.get(),argv[i+1],argv[i+2],argv[i+3],argv[i+4]);
        });
        return t.summary();
    }
    for (int i=3;i+3<argc;++i) if (!std::strcmp(argv[i],"--group-dflash-staged-exact") || !std::strcmp(argv[i],"--group-dflash-staged-guards") ||
            !std::strcmp(argv[i],"--group-dflash-staged-park") || !std::strcmp(argv[i],"--group-dflash-staged-suffix") ||
            !std::strcmp(argv[i],"--group-dflash-staged-rebind") || !std::strcmp(argv[i],"--group-dflash-staged-rebind-k4v") ||
            !std::strcmp(argv[i],"--group-dflash-staged-widths")) {
        t.test("Streamed_DFlash_matches_staged_native_target_and_draft_vectors",[&](testing & t) {
            test_native_dflash_draft_stage(t,model.get(),argv[i+1],argv[i+2],argv[i+3],true,
                !std::strcmp(argv[i],"--group-dflash-staged-guards"),
                !std::strcmp(argv[i],"--group-dflash-staged-park") || !std::strcmp(argv[i],"--group-dflash-staged-rebind") ||
                    !std::strcmp(argv[i],"--group-dflash-staged-rebind-k4v"),
                !std::strcmp(argv[i],"--group-dflash-staged-suffix"),
                !std::strcmp(argv[i],"--group-dflash-staged-rebind") || !std::strcmp(argv[i],"--group-dflash-staged-rebind-k4v"),
                !std::strcmp(argv[i],"--group-dflash-staged-rebind-k4v"),false,!std::strcmp(argv[i],"--group-dflash-staged-widths"));
        });
        return t.summary();
    }
    for (int i=3;i+3<argc;++i) if (!std::strcmp(argv[i],"--native-dflash-draft-stage") || !std::strcmp(argv[i],"--native-dflash-draft-suffix") ||
            !std::strcmp(argv[i],"--native-dflash-draft-repeat") || !std::strcmp(argv[i],"--native-dflash-draft-widths")) {
        t.test("Staged_native_draft_consumes_exact_target_features_without_target_neural_execution",[&](testing & t) {
            test_native_dflash_draft_stage(t,model.get(),argv[i+1],argv[i+2],argv[i+3],false,false,false,
                !std::strcmp(argv[i],"--native-dflash-draft-suffix"),false,false,!std::strcmp(argv[i],"--native-dflash-draft-repeat"),
                !std::strcmp(argv[i],"--native-dflash-draft-widths"));
        });
        return t.summary();
    }
    for (int i=3;i+2<argc;++i) if (!std::strcmp(argv[i],"--native-dflash-target-stage")) {
        t.test("Staged_native_target_exports_exact_features_and_checkpoints",[&](testing & t) {
            test_group_dflash_metadata(t,model.get(),argv[i+1],true,false,false,false,argv[i+2]);
        });
        return t.summary();
    }
    for (int i=3;i+1<argc;++i) if (!std::strcmp(argv[i],"--native-two-slot-dflash3")) {
        t.test("Native_DFlash3_pair_uses_three_snapshots_per_slot_and_executes_both_models",[&](testing & t) {
            test_native_two_slot_dflash3(t,model.get(),argv[i+1]);
        });
        return t.summary();
    }
    for (int i=3;i+1<argc;++i) if (!std::strcmp(argv[i],"--group-dflash-feature-proof") ||
            !std::strcmp(argv[i],"--group-dflash-feature-proof-gpu")) {
        t.test("DFlash_group_retains_exact_target_features_and_per_slot_source_proofs",[&](testing & t) {
            test_group_dflash_metadata(t,model.get(),argv[i+1],true,false,!std::strcmp(argv[i],"--group-dflash-feature-proof-gpu"));
        });
        return t.summary();
    }
    for (int i=3;i+2<argc;++i) if (!std::strcmp(argv[i],"--group-dflash-image-source") || !std::strcmp(argv[i],"--group-dflash-image-skip") ||
            !std::strcmp(argv[i],"--group-dflash-image-teacher") || !std::strcmp(argv[i],"--group-dflash-image-tail") || !std::strcmp(argv[i],"--group-dflash-image-direct") ||
            !std::strcmp(argv[i],"--group-dflash-image-peer")) {
        t.test("Group_DFlash_image_features_preserve_the_completed_source",[&](testing & t) {
            test_group_dflash_image_features(t,model.get(),argv[i+1],argv[i+2],std::strcmp(argv[i],"--group-dflash-image-source")!=0,
                !std::strcmp(argv[i],"--group-dflash-image-teacher") || !std::strcmp(argv[i],"--group-dflash-image-tail") || !std::strcmp(argv[i],"--group-dflash-image-peer"),
                !std::strcmp(argv[i],"--group-dflash-image-tail"),!std::strcmp(argv[i],"--group-dflash-image-direct"),!std::strcmp(argv[i],"--group-dflash-image-peer"));
        });return t.summary();
    }
    for (int i=3;i+1<argc;++i) if (!std::strcmp(argv[i],"--group-dflash-metadata") || !std::strcmp(argv[i],"--group-dflash-quote") ||
            !std::strcmp(argv[i],"--group-dflash-quote-gpu") || !std::strcmp(argv[i],"--group-dflash-bind-gpu") ||
            !std::strcmp(argv[i],"--group-dflash-bind")) {
        t.test("DFlash_group_declares_bounded_draft_state_before_allocating_the_parent",[&](testing & t) {
            const bool binding=!std::strcmp(argv[i],"--group-dflash-bind-gpu") || !std::strcmp(argv[i],"--group-dflash-bind");
            const bool gpu=!std::strcmp(argv[i],"--group-dflash-bind-gpu") || !std::strcmp(argv[i],"--group-dflash-quote-gpu");
            test_group_dflash_metadata(t,model.get(),argv[i+1],false,binding || gpu || !std::strcmp(argv[i],"--group-dflash-quote"),gpu,binding);
        });
        return t.summary();
    }
    if (group_dflash_rollback_metadata) {
        t.test("DFlash_group_declares_seven_per_slot_snapshots_without_K15",[&](testing & t) {
            test_group_dflash_rollback_metadata(t,model.get());
        });
        return t.summary();
    }
    if (std::any_of(argv+3,argv+argc,[](const char * arg) {return !std::strcmp(arg,"--group-dflash-target-boundary");})) {
        t.test("Grouped_DFlash_K7_target_matches_native_rollback_and_continuation",[&](testing & t) {
            test_group_rollback(t,model.get(),true,false,7,true);
        });
        return t.summary();
    }
    for (int i=3;i+1<argc;++i) if (!std::strcmp(argv[i],"--group-gemma-assistant-acceptance") ||
            !std::strcmp(argv[i],"--group-gemma-assistant-ring-acceptance") || !std::strcmp(argv[i],"--group-gemma-assistant-prefix-probe")) {
        const bool ring=!std::strcmp(argv[i],"--group-gemma-assistant-ring-acceptance");
        const bool probe=!std::strcmp(argv[i],"--group-gemma-assistant-prefix-probe");
        t.test("Every_Gemma_shared_MTP_prefix_keeps_native_correction_and_next_prediction",[&](testing & t) {
            if (ring || probe) {
                test_group_gemma_assistant_metadata(t,model.get(),argv[i+1],false,true,true,true,true,ring,{3,3},{0,3});return;
            }
            for (uint16_t depth=1;depth<=3;++depth) for (int16_t first=0;first<=depth;++first) for (int16_t second=0;second<=depth;++second) {
                t.out<<"Gemma MTP acceptance widths="<<depth<<","<<depth<<" accepted="<<first<<","<<second<<std::endl;
                test_group_gemma_assistant_metadata(t,model.get(),argv[i+1],false,true,true,true,true,false,{depth,depth},{first,second});
                if (t.failures) return;
            }
            for (int16_t first=0;first<=1;++first) for (int16_t second=0;second<=3;++second) {
                t.out<<"Gemma MTP acceptance widths=1,3 accepted="<<first<<","<<second<<std::endl;
                test_group_gemma_assistant_metadata(t,model.get(),argv[i+1],false,true,true,true,true,false,{1,3},{first,second});
                if (t.failures) return;
            }
        });return t.summary();
    }
    for (int i=3;i+1<argc;++i) if (!std::strcmp(argv[i],"--group-gemma-assistant-ring")) {
        t.test("Gemma_shared_assistant_requires_real_ring_transport_with_exact_native_vectors",[&](testing & t) {
            test_group_gemma_assistant_metadata(t,model.get(),argv[i+1],false,true,true,true,true,true);
        });
        return t.summary();
    }
    for (int i=3;i+1<argc;++i) if (!std::strcmp(argv[i],"--group-gemma-assistant-metadata") ||
            !std::strcmp(argv[i],"--group-gemma-assistant-late") || !std::strcmp(argv[i],"--group-gemma-assistant-quote") ||
            !std::strcmp(argv[i],"--group-gemma-assistant-factory") || !std::strcmp(argv[i],"--group-gemma-assistant-execution") ||
            !std::strcmp(argv[i],"--group-gemma-assistant-manager")) {
        t.test("Selected_Gemma_assistant_declares_only_target_read_aliases",[&](testing & t) {
            test_group_gemma_assistant_metadata(t,model.get(),argv[i+1],!std::strcmp(argv[i],"--group-gemma-assistant-late"),
                !std::strcmp(argv[i],"--group-gemma-assistant-quote") || !std::strcmp(argv[i],"--group-gemma-assistant-factory") ||
                    !std::strcmp(argv[i],"--group-gemma-assistant-execution") || !std::strcmp(argv[i],"--group-gemma-assistant-manager"),
                !std::strcmp(argv[i],"--group-gemma-assistant-factory") || !std::strcmp(argv[i],"--group-gemma-assistant-execution") ||
                    !std::strcmp(argv[i],"--group-gemma-assistant-manager"),
                !std::strcmp(argv[i],"--group-gemma-assistant-execution") || !std::strcmp(argv[i],"--group-gemma-assistant-manager"),
                !std::strcmp(argv[i],"--group-gemma-assistant-manager"));
        });
        return t.summary();
    }
    if (group_mtp_image) {
        for (int i=3;i+1<argc;++i) if (!std::strcmp(argv[i],"--group-mtp-image-catchup") || !std::strcmp(argv[i],"--group-mtp-image-workspace"))
            t.test("Selected_MTP_image_catchup_preserves_the_other_slot",[&](testing & t) {test_two_slot_mtp_image_catchup(t,model.get(),argv[i+1],group_mtp_image_workspace);});
        return t.summary();
    }
    if (group_mtp_confidence) {
        t.test("Selected_MTP_confidence_and_minimum_length_cutoffs_keep_native_boundaries",[&](testing & t) {
            for (uint32_t cutoff : {1u,2u}) {
                t.out<<"MTP cutoff="<<cutoff<<std::endl;
                test_group_mtp_admission(t,model.get(),true,false,false,true,{1,3},{0,uint16_t(cutoff==1 ? 0 : 2)},false,cutoff);
                if (t.failures) return;
            }
        });
        return t.summary();
    }
    if (group_mtp_acceptance) {
        t.test("Every_selected_MTP_prefix_preserves_native_correction_and_next_draft",[&](testing & t) {
            for (uint16_t depth=1;depth<=3;++depth) for (uint16_t a=0;a<=depth;++a) for (uint16_t b=0;b<=depth;++b) {
                t.out<<"MTP acceptance widths="<<depth<<','<<depth<<" accepted="<<a<<','<<b<<std::endl;
                test_group_mtp_admission(t,model.get(),true,false,false,true,{depth,depth},{a,b});
                if (t.failures) return;
            }
            for (uint16_t a=0;a<=1;++a) for (uint16_t b=0;b<=3;++b) {
                t.out<<"MTP acceptance widths=1,3 accepted="<<a<<','<<b<<std::endl;
                test_group_mtp_admission(t,model.get(),true,false,false,true,{1,3},{a,b});
                if (t.failures) return;
            }
        });
        return t.summary();
    }
    if (group_mtp_backend || group_mtp_ring || group_mtp_admission || group_mtp_resume) {
        t.test("Selected_target_and_MTP_head_share_one_admitted_workspace",[&](testing & t) {test_group_mtp_admission(t,model.get(),group_mtp_resume,group_mtp_phase_guard,group_mtp_ring,group_mtp_partial,{1,3},{0,2},group_mtp_backend || group_mtp_backend_resume);});
        return t.summary();
    }
    if (group_auxiliary_metadata) {
        t.test("Selected_group_describes_both_owned_MTP_caches_before_execution",[&](testing & t) {test_group_auxiliary_metadata(t,model.get());});
        return t.summary();
    }
    if (std::any_of(argv+3,argv+argc,[](const char * arg) {return !std::strcmp(arg,"--group-dflash-target-prefixes");})) {
        t.test("Every_two_slot_seven_token_acceptance_prefix_matches_native_correction_and_restore",[&](testing & t) {
            test_group_rollback(t,model.get(),true,false,7,false,true);
        });
        return t.summary();
    }
    if (std::any_of(argv+3,argv+argc,[](const char * arg) {return !std::strcmp(arg,"--group-rollback-owner-failure-only");})) {
        t.test("Failed_hybrid_rollback_closes_admission_before_reconstruction",[&](testing & t) {test_group_rollback(t,model.get(),true,true);});
        return t.summary();
    }
    if (std::any_of(argv+3,argv+argc,[](const char * arg) {return !std::strcmp(arg,"--group-recurrent-rollback-only");})) {
        t.test("Grouped_recurrent_snapshots_preserve_each_slot_and_accepted_prefix",[&](testing & t) {test_group_rollback(t,model.get(),true);});
        return t.summary();
    }
    if (std::any_of(argv+3,argv+argc,[](const char * arg) {return !std::strcmp(arg,"--group-rollback-only");})) {
        t.test("Grouped_MTP_rollback_preserves_each_accepted_prefix_and_the_peer",[&](testing & t) {test_group_rollback(t,model.get());});
        return t.summary();
    }
    if (std::any_of(argv+3,argv+argc,[](const char * arg) {return !std::strcmp(arg,"--group-verify-split-only");})) {
        t.test("Grouped_verification_catalog_covers_multiple_native_microbatches",[&](testing & t) {test_group_verify_widths(t,model.get(),true,true);});
        return t.summary();
    }
    if (std::any_of(argv+3,argv+argc,[](const char * arg) {return !std::strcmp(arg,"--group-verify-catalog-only");})) {
        t.test("Declared_group_verification_catalog_covers_every_width_before_activation",[&](testing & t) {test_group_verify_widths(t,model.get(),true);});
        return t.summary();
    }
    if (std::any_of(argv+3,argv+argc,[](const char * arg) {return !std::strcmp(arg,"--group-verify-widths-only");})) {
        t.test("Grouped_verification_width_is_per_sequence_and_independent_of_RS0",[&](testing & t) {test_group_verify_widths(t,model.get());});
        return t.summary();
    }
    if (std::any_of(argv+3,argv+argc,[](const char * arg) {return !std::strcmp(arg,"--group-compute-failure-only");})) {
        t.test("Failed_group_compute_retires_admission_before_reusing_the_parent",[&](testing & t) {test_group_compute_failure(t,model.get());});
        return t.summary();
    }
    if (std::any_of(argv+3,argv+argc,[](const char * arg) {return !std::strcmp(arg,"--group-reselection-only");})) {
        t.test("Group_reselection_parks_the_peer_and_resumes_both_with_native_exact_results",[&](testing & t) {test_group_reselection(t,model.get());});
        return t.summary();
    }
    if (std::any_of(argv+3,argv+argc,[](const char * arg) {return !std::strcmp(arg,"--group-manager-only");})) {
        t.test("Group_manager_parks_native_state_and_reuses_one_physical_parent",[&](testing & t) {test_owned_group_context(t,model.get(),1,true);});
        return t.summary();
    }
    if (std::any_of(argv+3,argv+argc,[](const char * arg) {return !std::strcmp(arg,"--group-prefill-silent-only");})) {
        t.test("Group_prefill_accepts_cache_only_native_batches_before_the_final_output",[&](testing & t) {test_owned_group_context(t,model.get(),128,false,true);});
        return t.summary();
    }
    if (std::any_of(argv+3,argv+argc,[](const char * arg) {return !std::strcmp(arg,"--group-owned-prefill-only");})) {
        t.test("Owned_group_context_preserves_native_wide_prefill_across_restore",[&](testing & t) {test_owned_group_context(t,model.get(),128);});
        return t.summary();
    }
    if (std::any_of(argv+3,argv+argc,[](const char * arg) {return !std::strcmp(arg,"--group-owned-neural-only");})) {
        t.test("Owned_group_context_uses_one_parent_and_preserves_native_neural_results",[&](testing & t) {test_owned_group_context(t,model.get());});
        return t.summary();
    }
    if (std::any_of(argv+3,argv+argc,[](const char * arg) {return !std::strcmp(arg,"--group-neural-only");})) {
        t.test("Selected_group_context_executes_joint_native_exact_target_steps",[&](testing & t) {test_live_group_context(t,model.get());});
        return t.summary();
    }
    if (std::any_of(argv+3,argv+argc,[](const char * arg) {return !std::strcmp(arg,"--group-full-binding-only");})) {
        t.test("Selected_native_full_cache_binds_group_projection_without_exposing_execution",[&](testing & t) {
            test_group_full_binding(t,model.get());
        });
        return t.summary();
    }
    if (std::any_of(argv+3,argv+argc,[](const char * arg) {return !std::strcmp(arg,"--group-initialization-only");})) {
        t.test("Selected_group_context_stays_unallocated_and_non_executable_until_bound",[&](testing & t) {
            test_group_initialization(t,model.get());
        });
        return t.summary();
    }
    if (measurement_only) {
        if (std::any_of(argv+3,argv+argc,[](const char * arg) {return !std::strcmp(arg,"--measure-gpu-weights");})) {
            bool placed=false;
            if (model) for (const auto & allocation : model->memory_breakdown()) {
                auto * device=ggml_backend_buft_get_device(allocation.first);
                placed=placed || (device && ggml_backend_dev_type(device)==GGML_BACKEND_DEVICE_TYPE_GPU &&
                    allocation.second>size_t(1024)*1024*1024);
            }
            t.test("Measurement_uses_the_actual_GPU_weight_placement",[&](testing & t) {t.assert_true(placed);});
            if (!placed) return t.summary();
        }
        t.test("Graph_measurement_context_cannot_allocate_or_execute_and_does_not_mutate_model",[&](testing & t) {test_measurement_context(t,model.get());});
        t.test("Graph_measurement_preserves_native_unequal_microbatches_and_output_rows",[&](testing & t) {test_native_measurement_batches(t,model.get());});
        if (model) t.test("Target_workload_quote_combines_real_native_graph_and_transport_requirements",
            [&](testing & t) {test_target_workload_quote(t,model.get());});
        if (model && model->arch==LLM_ARCH_QWEN35) t.test("Graph_measurement_includes_staged_K3_K7_publication_without_allocating_payload",
            [&](testing & t) {test_rollback_measurement_batches(t,model.get());});
        return t.summary();
    }
    if (recurrent_cache_only) {
        t.test("Qwen_real_geometry_native_current_banks_park_and_restore",[&](testing & t) { test_qwen_current_cache_parking(t,model.get()); });
        return t.summary();
    }
    if (std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--qwen-current-park") || !std::strcmp(arg,"--gemma-current-park"); })) {
        t.test("Selected_current_state_parking_preserves_native_vectors",[&](testing & t) { test_selected_current_parking(t,model.get()); });
        return t.summary();
    }
    for (int i=3;i<argc;++i) if (!std::strcmp(argv[i],"--gemma-acceptance")) {
        if (i+1 == argc) { std::fprintf(stderr,"missing Gemma assistant artifact\n"); return 1; }
        const bool compact=std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--gemma-assistant-memory"); });
        t.test("Gemma_acceptance_rollback_and_request_reuse_match_native",[&](testing & t) {
            test_gemma_acceptance(t,model.get(),argv[i+1],compact);
        });
        return t.summary();
    }
    for (int i=3;i<argc;++i) if (!std::strcmp(argv[i],"--gemma-candidate-loop")) {
        if (i+1 == argc) { std::fprintf(stderr,"missing Gemma assistant artifact\n"); return 1; }
        t.test("Gemma_common_candidate_loop_matches_native_fixed_position_predictions",[&](testing & t) {
            const bool compact=std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--gemma-assistant-memory"); });
            test_gemma_candidate_loop(t,model.get(),argv[i+1],compact);
        });
        return t.summary();
    }
    for (int i=3;i<argc;++i) if (!std::strcmp(argv[i],"--gemma-assistant-source")) {
        if (i+1 == argc) { std::fprintf(stderr,"missing Gemma assistant artifact\n"); return 1; }
        const char * image_path=nullptr,* export_native=nullptr,* import_native=nullptr;
        for (int j=3;j<argc;++j) {
            const char ** destination=!std::strcmp(argv[j],"--gemma-assistant-image-features") ? &image_path :
                !std::strcmp(argv[j],"--gemma-assistant-export-native") ? &export_native :
                !std::strcmp(argv[j],"--gemma-assistant-import-native") ? &import_native : nullptr;
            if (!destination) continue;
            if (j+1==argc) { std::fprintf(stderr,"missing Gemma assistant reference path\n");return 1; }
            *destination=argv[j+1];
        }
        t.test("Gemma_actual_assistant_reads_target_sources_without_new_KV",[&](testing & t) {
            const bool forced=std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--gemma-assistant-forced"); });
            const bool compact=std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--gemma-assistant-memory"); });
            const bool verify=std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--gemma-assistant-verify"); });
            const bool quotas=std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--gemma-assistant-quotas"); });
            const bool vision=std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--gemma-assistant-vision-suspend"); });
            test_gemma_assistant_source(t,model.get(),argv[i+1],forced,compact,verify,quotas,vision,image_path,export_native,import_native);
        });
        return t.summary();
    }
    if (std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--gemma-admission-only"); })) {
        t.test("Gemma_target_admission_memory_boundaries",
            [&](testing & t) { test_gemma_target_admission(t,model.get(),false); });
        return t.summary();
    }
    for (int i=3;i<argc;++i) if (!std::strcmp(argv[i],"--gemma-native-logits") || !std::strcmp(argv[i],"--gemma-streamed-logits")) {
        const bool native=!std::strcmp(argv[i],"--gemma-native-logits");
        if (i+1 == argc) { std::fprintf(stderr,"missing Gemma logit control path\n"); return 1; }
        const char * path=argv[i+1];
        t.test(native ? "Gemma_native_logit_control_export" : "Gemma_streamed_target_matches_external_native_control",
            [&](testing & t) { test_gemma_target_context(t,model.get(),8192,1333,8,true,false,
                native ? path : nullptr,native ? nullptr : path); });
        return t.summary();
    }
    if (std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--gemma-admission-guards"); })) {
        t.test("Gemma_target_rejects_invalid_profiles_and_recovers_after_quota_failures",
            [&](testing & t) { test_gemma_target_admission(t,model.get()); });
        return t.summary();
    }
    if (std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--gemma-target"); })) {
        t.test("Gemma_streamed_target_matches_native_model_execution",
            [&](testing & t) { test_gemma_target_context(t,model.get()); });
        return t.summary();
    }
    if (std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--gemma-vision-suspend"); })) {
        t.test("Gemma_vision_lending_preserves_local_and_host_KV_then_native_continuation",
            [&](testing & t) { test_gemma_vision_suspension(t,model.get()); });
        return t.summary();
    }
    for (int i=3;i+1<argc;++i) if (!std::strcmp(argv[i],"--gemma-image-features")) {
        t.test("Gemma_image_SWA_and_causal_global_attention_match_native",[&](testing & t) {
            test_gemma_image_features(t,model.get(),argv[i+1]);
        });
        return t.summary();
    }
    if (std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--gemma-target-medium"); })) {
        t.test("Gemma_streamed_target_ring_matches_native_model_execution",
            [&](testing & t) { test_gemma_target_context(t,model.get(),32768,32000,16,true,true); });
        return t.summary();
    }
    if (std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--gemma-target-medium-fit"); })) {
        t.test("Gemma_streamed_target_light_ring_matches_fitting_native_context",
            [&](testing & t) { test_gemma_target_context(t,model.get(),24576,23552,16,true,true); });
        return t.summary();
    }
    if (std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--gemma-target-ring"); })) {
        t.test("Gemma_streamed_target_ring_exceeds_measured_native_context_capacity",
            [&](testing & t) { test_gemma_target_context(t,model.get(),32768,32000,16,false,true); });
        return t.summary();
    }
    if (std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--gemma-target-long"); })) {
        t.test("Gemma_streamed_target_long_history_fits_fixed_physical_budget",
            [&](testing & t) { test_gemma_target_context(t,model.get(),262144,65536,16,false,true); });
        return t.summary();
    }
    if (std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--gemma-lifecycle"); })) {
        t.test("Gemma_serial_lifecycle_preserves_both_cache_groups",
            [&](testing & t) { test_gemma_lifecycle(t,model.get()); });
        return t.summary();
    }
    if (std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--gemma-phase-parent"); })) {
        t.test("Gemma_native_local_cache_and_streamed_full_cache_share_one_phase_parent",
            [&](testing & t) { test_gemma_phase_parent(t,model.get()); });
        return t.summary();
    }
    if (std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--gemma-read-source-cache"); })) {
        t.test("Gemma_shared_full_cache_reads_use_progressive_target_storage",
            [&](testing & t) { test_gemma_phase_parent(t,model.get(),true); });
        return t.summary();
    }
    if (std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--gemma-selected-local-cache-only"); })) {
        t.test("Gemma_selected_native_local_groups_bind_park_and_restore_without_private_storage",[&](testing & t) {
            for (uint32_t selected : {1u,2u}) {
                t.out << "selected native SWA group=" << selected << std::endl;
                try {test_gemma_deferred_local(t,model.get(),selected);}
                catch (const std::exception & error) {t.out << error.what() << std::endl;t.assert_true("selected unified bounded cache admits native geometry",false);return;}
            }
            test_gemma_deferred_local(t,model.get(),1,2);
        });
        return t.summary();
    }
    if (deferred_local_cache_only || std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--gemma-deferred-local"); })) {
        t.test("Gemma_local_cache_metadata_binds_transactionally_without_private_storage",
            [&](testing & t) { test_gemma_deferred_local(t,model.get()); });
        return t.summary();
    }
    if (std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--gemma-stream-groups"); })) {
        t.test("Gemma_streams_only_owned_full_attention_cache_roots",
            [&](testing & t) { test_gemma_stream_groups(t,model.get()); });
        return t.summary();
    }
    if (std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--gemma-bounded"); })) {
        t.test("Gemma_target_local_cache_reuses_bounded_storage_without_changing_causal_execution",
            [&](testing & t) { test_gemma_bounded_cache(t,model.get()); });
        return t.summary();
    }
    if (dflash_vision_control) {
        if (!t.assert_true(bool(model) && dflash_native_model && dflash_vision_image)) return t.summary();
        t.test("DFlash_real_image_native_streamed_complete_values",[&](testing & t) {
            test_dflash_vision_exact(t,model.get(),dflash_native_model,dflash_vision_image,dflash_vision_control,
                dflash_vision_writing,dflash_vision_native_check,dflash_vision_depth,test_cache_k,test_cache_v,dflash_vision_ngram);
        });
        return t.summary();
    }

    if (dflash_vision_readiness) {
        if (!t.assert_true(bool(model) && dflash_native_model)) return t.summary();
        t.test("DFlash_image_provenance_and_vision_owner_readiness",[&](testing & t) {
            test_dflash_vision_readiness(t,model.get(),dflash_native_model,dflash_vision_microbatch);
        });
        return t.summary();
    }

    if (dflash_prefetch_hint) {
        if (!t.assert_true(bool(model) && dflash_native_model)) return t.summary();
        t.test("DFlash_context_owns_flexible_prefetch_and_restores_previous_intents", [&](testing & t) {
            test_dflash_prefetch_intent(t, model.get(), dflash_native_model);
        });
        return t.summary();
    }
    if (dflash_mixed_recovery) {
        if (!t.assert_true(bool(model) && dflash_native_model)) return t.summary();
        const char * export_native=nullptr,* import_native=nullptr;
        bool native_import=false;
        for (int i=3;i<argc;++i) {
            auto ** dest=!std::strcmp(argv[i],"--dflash-recovery-export") ? &export_native :
                (!std::strcmp(argv[i],"--dflash-recovery-import") || !std::strcmp(argv[i],"--dflash-recovery-native-check")) ? &import_native : nullptr;
            if (dest) {
                if (!t.assert_true(i+1<argc)) return t.summary();
                native_import |= !std::strcmp(argv[i],"--dflash-recovery-native-check");
                *dest=argv[++i];
            }
        }
        t.test("DFlash_mixed_failure_cleanup_and_reconstruction_matches_native",[&](testing & t) {
            test_dflash_failure_recovery(t,model.get(),dflash_native_model,export_native,import_native,native_import);
        });
        return t.summary();
    }
    if (dflash_mixed_switching) {
        if (!t.assert_true(bool(model) && dflash_native_model)) return t.summary();
        const char * export_native=nullptr,* import_native=nullptr;
        for (int i=3;i<argc;++i) {
            auto ** dest=!std::strcmp(argv[i],"--dflash-mixed-export") ? &export_native :
                !std::strcmp(argv[i],"--dflash-mixed-import") ? &import_native : nullptr;
            if (dest) {
                if (!t.assert_true(i+1<argc)) return t.summary();
                *dest=argv[++i];
            }
        }
        t.test("mixed_K4V_DFlash_switching_matches_native_and_independent_feature_maintenance",[&](testing & t) {
            test_dflash_mixed_switching(t,model.get(),dflash_native_model,test_cache_k,test_cache_v,export_native,import_native,dflash_mixed_all_widths,dflash_stream_control,dflash_full_prediction ? 7 : 3);
        });
        return t.summary();
    }
    if (dflash_independent_quotas) {
        if (!t.assert_true(bool(model) && dflash_native_model)) return t.summary();
        t.test("independent_DFlash_prediction_and_target_verification_joint_quota",
                [&](testing & t) { test_dflash_independent_quota(t, model.get(), dflash_native_model); });
        return t.summary();
    }
    if (dflash_native_model && !dflash_exact_control) {
        if (!t.assert_true(bool(model))) return t.summary();
        t.test("ordinary_dflash_feature_workspace", [&](testing & t) { test_dflash_native(t, model.get()); });
        return t.summary();
    }
    if (!t.assert_true(bool(model))) return t.summary();
    if (dflash_exact_control) {
        if (!t.assert_true(dflash_native_model != nullptr)) return t.summary();
        const bool lifecycle=std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--dflash-lifecycle"); });
        const bool context_end=std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--dflash-context-end"); });
        const bool sustained=std::any_of(argv+3,argv+argc,[](const char * arg) { return !std::strcmp(arg,"--dflash-rejection-long"); });
        t.test("streamed_DFlash2_matches_native_forced_decisions",[&](testing & t) {
            test_kv_stream_dflash_exact(t,model.get(),dflash_native_model,lifecycle,context_end || sustained,sustained,
                test_cache_k,test_cache_v,test_cache_explicit || dflash_short_control,dflash_short_control,dflash_independent_rows,
                dflash_independent_small_batch,dflash_native_f3 ? 3 : 0);
        });
        return t.summary();
    }
    if (wide_rollback_control) {
        t.test("target_TG1_TG8_rollback_matches_native_without_draft_KV", [&](testing & t) {
            test_wide_target_rollback(t,model.get());
        });
        return t.summary();
    }
    if (mtp_bit_exact_control) {
        t.test("streamed_and_native_mtp_bit_exact", [&](testing & t) {
            test_kv_stream_mtp_bit_exact(t, model.get(), kv_stream_exact_arguments(argc, argv));
        });
        return t.summary();
    }
    if (serial_workspace_growth) {
        t.test("serial_draft_growth_rebuilds_target_and_preserves_long_gather", [&](testing & t) {
            auto p = llama_context_default_params();
            p.n_ctx = 8192; p.n_batch = p.n_ubatch = 256;
            p.n_threads = p.n_threads_batch = 8;
            p.type_k = GGML_TYPE_Q8_0; p.type_v = GGML_TYPE_Q4_0;
            p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            p.shared_device_memory_bytes = size_t(2240)*1048576;
            p.n_outputs_max = p.n_outputs_max_per_seq = 4;
            p.kv_stream_auxiliary_layers = 1;
            context_ptr target(llama_init_from_model(model.get(),p),llama_free);
            if (!t.assert_true(bool(target))) return;
            llama_token token = 1000;
            llama_set_kv_stream_decode(target.get(),false);
            if (!t.assert_equal(0,llama_decode(target.get(),llama_batch_get_one(&token,1)))) return;
            auto checkpoint = std::vector<uint8_t>(llama_state_get_size(target.get()));
            if (!t.assert_equal(checkpoint.size(),llama_state_get_data(target.get(),checkpoint.data(),checkpoint.size()))) return;
            if (!t.assert_equal(0,llama_decode(target.get(),llama_batch_get_one(&token,1)))) return;
            const size_t vocabulary = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
            const auto * logits = llama_get_logits_ith(target.get(),-1);
            std::vector<float> expected(logits,logits+vocabulary);
            auto * hybrid = static_cast<llama_memory_hybrid *>(llama_get_memory(target.get()));
            recurrent_snapshot expected_recurrent;
            hybrid->get_mem_recr()->state_write(expected_recurrent,0,0);
            if (!t.assert_equal(checkpoint.size(),llama_state_set_data(target.get(),checkpoint.data(),checkpoint.size()))) return;
            const auto before = target->get_compute_memory()->phase_transition_count();
            auto * parent = target->get_compute_memory()->shared_parent();
            auto dp = p;
            dp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
            dp.ctx_other = target.get();
            dp.shared_device_memory_bytes = 0;
            dp.kv_stream_auxiliary_layers = 0;
            const auto old_revision = target->get_compute_memory()->graph_binding_revision();
            {
                serial_parent_view_failure fault(parent);
                context_ptr fallback(llama_init_from_model(model.get(),dp),llama_free);
                t.assert_true(fault.calls >= 2);
                // Recovery must remain usable even if separate fallback scratch cannot fit the GPU.
                if (fallback) t.assert_true(!fallback->get_compute_memory()->borrows_serial_parent());
            }
            t.assert_true(target->get_compute_memory()->valid());
            t.assert_equal(before,target->get_compute_memory()->phase_transition_count());
            t.assert_true(target->get_compute_memory()->graph_binding_revision() > old_revision);
            if (!t.assert_equal(0,llama_decode(target.get(),llama_batch_get_one(&token,1)))) return;
            logits = llama_get_logits_ith(target.get(),-1);
            t.assert_true(std::equal(expected.begin(),expected.end(),logits));
            recurrent_snapshot recovered_recurrent;
            hybrid->get_mem_recr()->state_write(recovered_recurrent,0,0);
            t.assert_true(expected_recurrent.metadata == recovered_recurrent.metadata && expected_recurrent.tensors == recovered_recurrent.tensors);
            if (!t.assert_equal(checkpoint.size(),llama_state_set_data(target.get(),checkpoint.data(),checkpoint.size()))) return;
            context_ptr draft(llama_init_from_model(model.get(),dp),llama_free);
            if (!t.assert_true(bool(draft))) return;
            t.assert_true(draft->get_compute_memory()->borrows_serial_parent());
            t.assert_true(target->get_compute_memory()->phase_transition_count() > before);
            t.assert_true(parent == target->get_compute_memory()->shared_parent());
            t.assert_equal(size_t(p.shared_device_memory_bytes),target->get_compute_memory()->shared_parent_capacity());
            if (!t.assert_equal(0,llama_decode(target.get(),llama_batch_get_one(&token,1)))) return;
            logits = llama_get_logits_ith(target.get(),-1);
            t.assert_true(std::equal(expected.begin(),expected.end(),logits));
            auto batch = llama_batch_init(256,llama_model_n_embd(model.get()),1);
            std::vector<llama_token> tokens(256,token);
            batch.token = tokens.data();
            struct cleanup {
                llama_batch & batch;
                ~cleanup() { batch.token = nullptr; llama_batch_free(batch); }
            } guard{batch};
            std::fill(batch.embd,batch.embd+256*llama_model_n_embd(model.get()),0.0f);
            for (int first = 0; first < 7936; first += 256) {
                batch.n_tokens = 256;
                for (int row = 0; row < 256; ++row) {
                    batch.pos[row] = first+row;
                    batch.n_seq_id[row] = 1;
                    batch.seq_id[row][0] = 0;
                    batch.logits[row] = row == 255;
                }
                if (!t.assert_equal(0,llama_decode(draft.get(),batch))) return;
            }
            auto * kv = static_cast<llama_kv_cache *>(llama_get_memory(draft.get()));
            t.assert_equal(size_t(7936),kv->mtp_auxiliary_cache()->tokens());
            t.assert_equal(size_t(p.shared_device_memory_bytes),target->get_compute_memory()->shared_parent_capacity());
        });
        return t.summary();
    }
    if (rollback_probe) {
        t.test("streamed_mtp_uses_bounded_host_spilled_recurrent_rollback", [&](testing & t) {
            auto p=llama_context_default_params();
            p.n_ctx=1024; p.n_batch=p.n_ubatch=256;
            p.n_threads=p.n_threads_batch=8;
            p.type_k=GGML_TYPE_Q8_0; p.type_v=GGML_TYPE_Q4_0;
            p.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
            p.kv_stream_pool_bytes=16*1048576;
            p.kv_stream_auxiliary_layers=1;
            p.n_rs_seq=3;
            auto invalid=p;
            invalid.n_rs_seq=8;
            context_ptr unsafe(llama_init_from_model(model.get(),invalid),llama_free);
            t.assert_true(!unsafe);
            invalid=p;
            invalid.n_rs_seq=4;
            context_ptr too_deep(llama_init_from_model(model.get(),invalid),llama_free);
            t.assert_true(!too_deep);
            context_ptr ctx(llama_init_from_model(model.get(),p),llama_free);
            if (!t.assert_true(bool(ctx))) return;
            t.assert_equal(uint32_t(3),llama_n_rs_seq(ctx.get()));
            auto * hybrid=dynamic_cast<llama_memory_hybrid *>(llama_get_memory(ctx.get()));
            if (!t.assert_true(hybrid != nullptr && hybrid->get_mem_attn()->get_kv_stream())) return;
            auto * spill=hybrid->get_mem_recr()->spill_bank();
            if (!t.assert_true(spill != nullptr)) return;
            t.assert_true(spill->host_bytes() >= 448*1048576 && spill->host_bytes() <= 450*1048576);
            t.assert_true(spill->staged_device_bytes() >= 18*1048576 && spill->staged_device_bytes() <= 20*1048576);
            auto batch=llama_batch_init(256,0,1);
            for (int i=0; i<256; ++i) {
                batch.token[i]=1; batch.pos[i]=i; batch.n_seq_id[i]=1;
                batch.seq_id[i][0]=0; batch.logits[i]=i==255;
            }
            batch.n_tokens=256;
            llama_set_kv_stream_decode(ctx.get(),false);
            t.assert_equal(0,llama_decode(ctx.get(),batch));
            batch.n_tokens=4;
            for (int i=0; i<4; ++i) {
                batch.token[i]=1; batch.pos[i]=256+i; batch.logits[i]=i==3;
            }
            llama_set_kv_stream_decode(ctx.get(),true);
            t.assert_equal(0,llama_decode(ctx.get(),batch));
            llama_batch_free(batch);
        });
        t.test("independent_target_rollback_needs_no_auxiliary_kv_layer", [&](testing & t) {
            auto p=llama_context_default_params();
            p.n_ctx=1024; p.n_batch=p.n_ubatch=256; p.n_seq_max=1;
            p.n_threads=p.n_threads_batch=8;
            p.n_rs_seq=3;
            p.n_outputs_max=p.n_outputs_max_per_seq=4;
            p.type_k=GGML_TYPE_Q8_0; p.type_v=GGML_TYPE_Q4_0;
            p.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
            p.kv_stream_pool_bytes=16*1048576;
            context_ptr ctx(llama_init_from_model(model.get(),p),llama_free);
            if (!t.assert_true(bool(ctx))) return;
            auto * hybrid=dynamic_cast<llama_memory_hybrid *>(llama_get_memory(ctx.get()));
            if (!t.assert_true(hybrid && hybrid->get_mem_recr()->spill_enabled())) return;
            auto * target=hybrid->get_mem_attn()->get_kv_stream();
            if (!t.assert_true(target && !target->auxiliary_cache())) return;
            t.assert_equal(uint32_t(4),ctx->get_cparams().kv_stream_decode_rows());
            auto batch=llama_batch_init(256,0,1);
            struct release { llama_batch & batch; ~release() { llama_batch_free(batch); } } guard{batch};
            const auto submit=[&](uint32_t width,llama_pos first,bool decode) {
                batch.n_tokens=int32_t(width);
                for (uint32_t i=0;i<width;++i) {
                    batch.token[i]=1; batch.pos[i]=first+llama_pos(i);
                    batch.n_seq_id[i]=1; batch.seq_id[i][0]=0; batch.logits[i]=decode || i+1==width;
                }
                llama_set_kv_stream_decode(ctx.get(),decode);
                return llama_decode(ctx.get(),batch);
            };
            if (!t.assert_equal(0,submit(256,0,false))) return;
            llama_pos position=256;
            for (uint32_t width : {1u,2u,3u,4u}) {
                if (!t.assert_equal(0,submit(width,position,true))) return;
                position+=llama_pos(width);
                t.assert_equal(size_t(position),target->tokens());
            }
            t.assert_true(llama_memory_seq_rm(llama_get_memory(ctx.get()),0,position-3,-1));
            t.assert_equal(size_t(position-3),target->tokens());
            t.assert_equal(0,submit(1,position-3,true));
        });
        return t.summary();
    }
    if (target_drift_control) {
        t.test("teacher_forced_target_verification_and_replay", [&](testing & t) {
            if (!t.assert_true(argc >= 5)) return;
            std::ifstream input(argv[4]);
            if (!t.assert_true(bool(input))) return;
            const std::string article((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
            const size_t prefill_tokens = std::getenv("LLAMA_DIAG_PREFILL_TOKENS") ? size_t(std::atoi(std::getenv("LLAMA_DIAG_PREFILL_TOKENS"))) : 7932;
            const size_t decode_tokens = std::getenv("LLAMA_DIAG_DECODE_TOKENS") ? size_t(std::atoi(std::getenv("LLAMA_DIAG_DECODE_TOKENS"))) : 16;
            if (!t.assert_true(prefill_tokens >= 256 && prefill_tokens <= 261888 &&
                    decode_tokens >= 16 && decode_tokens <= 256 && prefill_tokens+decode_tokens <= 262144)) return;
            const auto * vocab = llama_model_get_vocab(model.get());
            const int needed = -llama_tokenize(vocab, article.data(), int(article.size()), nullptr, 0, false, false);
            std::vector<llama_token> tokens(needed);
            if (!t.assert_true(llama_tokenize(vocab, article.data(), int(article.size()),
                    tokens.data(), needed, false, false) >= int32_t(prefill_tokens))) return;
            auto p = llama_context_default_params();
            p.n_ctx = uint32_t((prefill_tokens+decode_tokens+255)/256*256); p.n_batch = p.n_ubatch = 256;
            p.n_rs_seq = target_drift_rs_control ? 3 : 0;
            p.n_threads = p.n_threads_batch = 8;
            p.n_outputs_max = p.n_outputs_max_per_seq = 4;
            p.type_k = GGML_TYPE_Q8_0; p.type_v = GGML_TYPE_Q4_0;
            p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            struct diagnostic_trace {
                bool enabled = false;
                size_t stable = 0;
                std::vector<std::pair<std::string,std::vector<float>>> values;
                struct cache { std::string name; std::vector<uint8_t> bytes; size_t stride; };
                std::vector<cache> caches;
            } trace_stock, trace_streamed;
            p.cb_eval_user_data = &trace_stock;
            p.cb_eval = [](ggml_tensor * tensor, bool ask, void * opaque) {
                auto & trace = *static_cast<diagnostic_trace *>(opaque);
                std::string name = tensor->name;
                bool first_attention = false;
                if (tensor->op == GGML_OP_FLASH_ATTN_EXT) {
                    const ggml_tensor * q = tensor->src[0];
                    for (int depth = 0; q && depth < 8; ++depth, q = q->src[0]) {
                        if (std::strcmp(q->name,"Qcur-3") == 0) { first_attention = true; break; }
                    }
                }
                if (first_attention) name = "FA-3";
                if (!trace.enabled || tensor->type != GGML_TYPE_F32 ||
                        (name != "Qcur-3" && name != "Kcur-3" && name != "Vcur-3" &&
                         !first_attention && name != "attn_pregate-3")) return false;
                if (ask) return true;
                trace.values.emplace_back(name,std::vector<float>(ggml_nbytes(tensor)/sizeof(float)));
                ggml_backend_tensor_get(tensor,trace.values.back().second.data(),0,ggml_nbytes(tensor));
                if (tensor->op == GGML_OP_FLASH_ATTN_EXT) {
                    for (int plane = 1; plane <= 2; ++plane) {
                        const auto * source = tensor->src[plane];
                        trace.caches.push_back({plane == 1 ? "Kcache" : "Vcache",
                            std::vector<uint8_t>(trace.stable*source->nb[1]),source->nb[1]});
                        ggml_backend_tensor_get(source,trace.caches.back().bytes.data(),0,trace.caches.back().bytes.size());
                    }
                }
                return true;
            };
            context_ptr stock(llama_init_from_model(model.get(), p), llama_free);
            const size_t arena_mib = std::getenv("LLAMA_DIAG_ARENA_MIB") ? size_t(std::atoi(std::getenv("LLAMA_DIAG_ARENA_MIB"))) : 512;
            if (!t.assert_true(arena_mib >= 128 && arena_mib <= 4096)) return;
            p.shared_device_memory_bytes = arena_mib*1048576;
            p.kv_stream_auxiliary_layers = 1;
            p.cb_eval_user_data = &trace_streamed;
            context_ptr streamed(llama_init_from_model(model.get(), p), llama_free);
            if (!t.assert_true(stock && streamed)) return;
            llama_set_embeddings_nextn(stock.get(), true, false);
            llama_set_embeddings_nextn(streamed.get(), true, false);
            auto batch = llama_batch_init(256, 0, 1);
            struct cleanup { llama_batch & batch; ~cleanup() { llama_batch_free(batch); } } owner{batch};
            const auto submit = [&](llama_context * ctx, const std::vector<llama_token> & rows, size_t first, bool decode) {
                auto & trace = ctx == stock.get() ? trace_stock : trace_streamed;
                trace.enabled = decode; trace.stable = first;
                trace.values.clear(); trace.caches.clear();
                batch.n_tokens = int32_t(rows.size());
                for (size_t i = 0; i < rows.size(); ++i) {
                    batch.token[i] = rows[i]; batch.pos[i] = llama_pos(first+i);
                    batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0; batch.logits[i] = decode || i+1 == rows.size();
                }
                llama_set_kv_stream_decode(ctx, decode);
                const bool result = t.assert_equal(0, llama_decode(ctx, batch));
                llama_synchronize(ctx);
                return result;
            };
            const size_t vocabulary = llama_vocab_n_tokens(vocab);
            const auto compare = [&](const char * label, size_t first, size_t rows) {
                if (!t.assert_equal(trace_stock.values.size(),trace_streamed.values.size()) ||
                        !t.assert_equal(trace_stock.caches.size(),trace_streamed.caches.size())) return;
                for (size_t i = 0; i < std::min(trace_stock.values.size(),trace_streamed.values.size()); ++i) {
                    const auto & a = trace_stock.values[i]; const auto & b = trace_streamed.values[i];
                    if (!t.assert_equal(a.first,b.first) || !t.assert_equal(a.second.size(),b.second.size())) return;
                    float error = 0;
                    for (size_t j = 0; j < a.second.size(); ++j) error=std::max(error,std::abs(a.second[j]-b.second[j]));
                    t.out << label << " pos=" << first << " tensor=" << a.first << " max=" << error << '\n';
                }
                for (size_t i = 0; i < std::min(trace_stock.caches.size(),trace_streamed.caches.size()); ++i) {
                    const auto & a = trace_stock.caches[i]; const auto & b = trace_streamed.caches[i];
                    if (!t.assert_equal(a.bytes.size(),b.bytes.size())) return;
                    const auto mismatch=std::mismatch(a.bytes.begin(),a.bytes.end(),b.bytes.begin());
                    const size_t byte=size_t(mismatch.first-a.bytes.begin());
                    t.out << label << " pos=" << first << " cache=" << a.name << " first_different_token="
                        << (byte == a.bytes.size() ? -1 : int64_t(byte/a.stride)) << '\n';
                }
                for (size_t row = 0; row < rows; ++row) {
                    const float * a = llama_get_logits_ith(stock.get(), int32_t(row)-int32_t(rows));
                    const float * b = llama_get_logits_ith(streamed.get(), int32_t(row)-int32_t(rows));
                    float error = 0;
                    for (size_t i = 0; i < vocabulary; ++i) {
                        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) { t.assert_true(false); return; }
                        error = std::max(error, std::abs(a[i]-b[i]));
                    }
                    t.assert_true(error == 0);
                    t.out << label << " pos=" << first+row << " logit_max=" << error
                        << " stock_top=" << std::max_element(a,a+vocabulary)-a
                        << " streamed_top=" << std::max_element(b,b+vocabulary)-b << '\n';
                }
                recurrent_snapshot a, b;
                llama_synchronize(stock.get()); llama_synchronize(streamed.get());
                static_cast<llama_memory_hybrid *>(llama_get_memory(stock.get()))->get_mem_recr()->state_write(a,0,0);
                static_cast<llama_memory_hybrid *>(llama_get_memory(streamed.get()))->get_mem_recr()->state_write(b,0,0);
                if (!t.assert_equal(a.tensors.size(), b.tensors.size())) return;
                size_t different = 0; float worst = 0;
                for (size_t part = 0; part < a.tensors.size(); ++part) {
                    if (!t.assert_equal(a.tensors[part].size(), b.tensors[part].size())) return;
                    float error = 0;
                    for (size_t i = 0; i < a.tensors[part].size(); ++i) {
                        if (!std::isfinite(a.tensors[part][i]) || !std::isfinite(b.tensors[part][i])) { t.assert_true(false); return; }
                        error = std::max(error, std::abs(a.tensors[part][i]-b.tensors[part][i]));
                    }
                    if (error) {
                        if (!different) t.out << label << " first_state_part=" << part << " error=" << error << '\n';
                        ++different; worst = std::max(worst,error);
                    }
                }
                t.out << label << " different_state_parts=" << different << " state_max=" << worst << '\n';
                t.assert_equal(size_t(0), different);
            };
            for (size_t first = 0; first < prefill_tokens; first += 256) {
                const size_t n = std::min<size_t>(256,prefill_tokens-first);
                const std::vector<llama_token> rows(tokens.begin()+first,tokens.begin()+first+n);
                if (!submit(stock.get(),rows,first,false) || !submit(streamed.get(),rows,first,false)) return;
            }
            compare("prefill", prefill_tokens-1, 1);
            const std::vector<llama_token> pattern{264,4927,5253,383,2919,4802,11,4927,5253,383,4128,958,11,4927,5253,383};
            std::vector<llama_token> continuation(decode_tokens);
            for (size_t i = 0; i < decode_tokens; ++i) continuation[i] = pattern[i%pattern.size()];
            const size_t width = std::getenv("LLAMA_DIAG_QUERY_WIDTH") ? size_t(std::atoi(std::getenv("LLAMA_DIAG_QUERY_WIDTH"))) : 2;
            if (!t.assert_true(width >= 1 && width <= 4)) return;
            for (size_t first = 0; first < continuation.size(); first += width) {
                const size_t count = std::min(width, continuation.size()-first);
                const auto save = [&](llama_context * ctx) {
                    std::vector<uint8_t> result(llama_state_seq_get_size_ext(ctx,0,LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
                    t.assert_equal(result.size(),llama_state_seq_get_data_ext(ctx,result.data(),result.size(),0,LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
                    return result;
                };
                auto a = save(stock.get()), b = save(streamed.get());
                const std::vector<llama_token> rows(continuation.begin()+first,continuation.begin()+first+count);
                if (!submit(stock.get(),rows,prefill_tokens+first,true) || !submit(streamed.get(),rows,prefill_tokens+first,true)) return;
                compare("verify",prefill_tokens+first,count);
                if (first%32 == 4 || first%32 == 12) {
                    const auto restore = [&](llama_context * ctx, const std::vector<uint8_t> & state) {
                        return t.assert_equal(state.size(),llama_state_seq_set_data_ext(ctx,state.data(),state.size(),0,LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)) &&
                            t.assert_true(llama_memory_seq_rm(llama_get_memory(ctx),0,llama_pos(prefill_tokens+first),-1));
                    };
                    if (!restore(stock.get(),a) || !restore(streamed.get(),b)) return;
                    if (!submit(stock.get(),rows,prefill_tokens+first,true) || !submit(streamed.get(),rows,prefill_tokens+first,true)) return;
                    compare("replay",prefill_tokens+first,count);
                }
            }
        });
        return t.summary();
    }
    if (mtp_memory_probe) {
        t.set_filter("native_context_serial_parent_probe");
        t.test("native_context_serial_parent_probe", [&](testing & t) {
            auto p = llama_context_default_params();
            p.n_ctx = 262144;
            p.n_batch = p.n_ubatch = 256;
            p.n_threads = p.n_threads_batch = 8;
            p.type_k = GGML_TYPE_Q8_0;
            p.type_v = GGML_TYPE_Q4_0;
            p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            p.kv_stream_auxiliary_layers = 1;
            p.shared_device_memory_bytes = size_t(1800)*1048576;
            context_ptr target(llama_init_from_model(model.get(), p), llama_free);
            if (!t.assert_true(bool(target) && target->get_compute_memory())) return;
            t.assert_true(target->get_cparams().embeddings_nextn && !target->get_cparams().embeddings_nextn_masked);
            auto d = p;
            d.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
            d.ctx_other = target.get();
            d.kv_stream_auxiliary_layers = 0;
            d.shared_device_memory_bytes = 0;
            context_ptr draft(llama_init_from_model(model.get(), d), llama_free);
            if (!t.assert_true(bool(draft) && draft->get_compute_memory())) return;
            t.assert_true(draft->get_cparams().embeddings_nextn && draft->get_cparams().embeddings_nextn_masked);
            auto * draft_kv = static_cast<llama_kv_cache *>(llama_get_memory(draft.get()));
            const int mtp_layer = llama_model_n_layer(model.get());
            t.assert_true(ggml_backend_buffer_is_host(draft_kv->get_k_storage(mtp_layer)->buffer));
            t.assert_true(ggml_backend_buffer_is_host(draft_kv->get_v_storage(mtp_layer)->buffer));
            auto * physical = target->get_compute_memory()->shared_parent();
            if (!t.assert_true(physical != nullptr)) return;
            bool found = false;
            for (auto * lease : draft->get_compute_memory()->workspace_leases()) {
                auto * buffer = lease ? ggml_backend_memory_lease_buffer(lease) : nullptr;
                if (!buffer || ggml_backend_buffer_get_type(buffer) != ggml_backend_buffer_get_type(physical)) continue;
                ggml_backend_memory_region region{};
                if (!t.assert_true(ggml_backend_memory_lease_get_region(lease, &region))) return;
                found = true;
                t.out << "native MTP draft workspace MiB=" << double(region.size)/1048576
                    << ", target parent MiB=" << double(ggml_backend_buffer_get_size(physical))/1048576 << '\n';
                t.assert_true(ggml_backend_buffer_get_base(buffer) == ggml_backend_buffer_get_base(physical));
            }
            t.assert_true(found);
            t.assert_true(draft->get_compute_memory()->borrows_serial_parent());
            auto target_batch = llama_batch_init(256, 0, 1);
            auto mtp_batch = llama_batch_init(256, llama_model_n_embd(model.get()), 1);
            std::vector<llama_token> mtp_tokens(256, 1);
            mtp_batch.token = mtp_tokens.data();
            struct batch_cleanup {
                llama_batch & target, & mtp;
                ~batch_cleanup() { mtp.token = nullptr; llama_batch_free(target); llama_batch_free(mtp); }
            } cleanup{target_batch, mtp_batch};
            std::fill(mtp_batch.embd, mtp_batch.embd + 256*llama_model_n_embd(model.get()), 0.0f);
            target_batch.n_tokens = mtp_batch.n_tokens = 256;
            for (int i = 0; i < 256; ++i) {
                target_batch.token[i] = mtp_batch.token[i] = 1;
                target_batch.pos[i] = mtp_batch.pos[i] = i;
                target_batch.n_seq_id[i] = mtp_batch.n_seq_id[i] = 1;
                target_batch.seq_id[i][0] = mtp_batch.seq_id[i][0] = 0;
                target_batch.logits[i] = mtp_batch.logits[i] = i == 255;
            }
            llama_set_kv_stream_decode(target.get(), false);
            if (!t.assert_equal(0, llama_decode(target.get(), target_batch))) return;
            llama_context_memory_diagnostics before{};
            if (!t.assert_true(target->get_compute_memory()->diagnostics(before))) return;
            if (!t.assert_equal(0, llama_decode(draft.get(), mtp_batch))) return;
            target_batch.n_tokens = 1;
            target_batch.token[0] = 1;
            target_batch.pos[0] = 256;
            target_batch.logits[0] = 1;
            llama_set_kv_stream_decode(target.get(), true);
            if (!t.assert_equal(0, llama_decode(target.get(), target_batch))) return;
            llama_context_memory_diagnostics after{};
            if (!t.assert_true(target->get_compute_memory()->diagnostics(after))) return;
            t.assert_true(after.kv_pool_bytes > before.kv_pool_bytes);
            if (!t.assert_true(llama_kv_stream_mtp_prepare(target.get(), 0))) return;
            mtp_batch.n_tokens = 1;
            mtp_batch.token[0] = 1;
            mtp_batch.pos[0] = 256;
            mtp_batch.logits[0] = 1;
            if (!t.assert_equal(0, llama_decode(draft.get(), mtp_batch))) return;
            auto * hybrid = static_cast<llama_memory_hybrid *>(llama_get_memory(target.get()));
            t.assert_equal(size_t(257), hybrid->get_mem_attn()->get_kv_stream()->tokens());
            t.assert_equal(size_t(257), hybrid->get_mem_attn()->get_kv_stream()->auxiliary_cache()->tokens());
            auto * stream=hybrid->get_mem_attn()->get_kv_stream();
            t.assert_true(stream->has_mtp_layer());
            target_batch.pos[0]=257;
            if (!t.assert_equal(0,llama_decode(target.get(),target_batch))) return;
            t.assert_true(!stream->has_mtp_layer());
            const bool target_removed = llama_memory_seq_rm(llama_get_memory(target.get()), 0, 0, -1);
            t.out << "native target full remove=" << target_removed << '\n';
            t.assert_true(target_removed);
            const bool draft_removed = llama_memory_seq_rm(llama_get_memory(draft.get()), 0, 0, -1);
            t.out << "native draft full remove=" << draft_removed << '\n';
            t.assert_true(draft_removed);
        });
        return t.summary();
    }
    const char * auxiliary_test_name = embedded_mtp_control ?
        "target_and_mtp_contexts_share_auxiliary_identity" :
        "target_context_owns_distinct_auxiliary_mtp_host_cache";
    if (auxiliary_control || embedded_mtp_control || target_tg3_control || target_stream_tg4_control) t.set_filter(auxiliary_test_name);
    const auto * vocab = llama_model_get_vocab(model.get());
    std::string text;
    for (int i = 0; i < 200; ++i) text += "The capital of France is Paris. We are testing a serial language model with a bounded key and value cache. ";
    const int size = -llama_tokenize(vocab,text.data(),int(text.size()),nullptr,0,true,false);
    std::vector<llama_token> prompt(size);
    const int tokens = llama_tokenize(vocab,text.data(),int(text.size()),prompt.data(),size,true,false);
    if (!t.assert_true(tokens >= (target_stream_tg4_control ? 3005 : 640))) return t.summary();
    prompt.resize(target_stream_tg4_control ? 3005 : 640);
    if (resume_control) {
        for (bool streamed : {false,true}) {
            if (streamed && resume_resident_only) continue;
            for (auto phase : {llama_memory_text_phase::decode,llama_memory_text_phase::prefill}) {
                const auto name = std::string(streamed ? "streamed" : "resident") +
                    "_hybrid_resume_matches_uninterrupted_" + (phase == llama_memory_text_phase::decode ? "decode" : "prefill");
                t.test(name, [&](testing & t) {
                    auto p = llama_context_default_params();
                    p.n_ctx = streamed ? 8192 : 1024;
                    p.n_batch = p.n_ubatch = streamed ? 64 : 256;
                    p.n_seq_max = 1; p.n_rs_seq = 0;
                    p.type_k = GGML_TYPE_Q8_0; p.type_v = GGML_TYPE_Q4_0;
                    p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
                    p.kv_stream_pool_bytes = 16*1048576;
                    context_ptr ctx(llama_init_from_model(model.get(),p),llama_free);
                    if (!t.assert_true(bool(ctx))) return;
                    std::vector<llama_token> history(streamed ? 6144 : 256);
                    for (size_t i = 0; i < history.size(); ++i) history[i] = prompt[i%prompt.size()];
                    llama_set_kv_stream_decode(ctx.get(),false);
                    for (size_t first = 0; first < history.size(); first += p.n_ubatch)
                        if (!t.assert_equal(0,llama_decode(ctx.get(),llama_batch_get_one(history.data()+first,p.n_ubatch)))) return;
                    const auto checkpoint = [&] {
                        std::vector<uint8_t> data(llama_state_get_size(ctx.get()));
                        data.resize(llama_state_get_data(ctx.get(),data.data(),data.size())); return data;
                    };
                    const auto prefill_state = checkpoint();
                    llama_set_kv_stream_decode(ctx.get(),true);
                    for (size_t i = 0; i < 4; ++i)
                        if (!t.assert_equal(0,llama_decode(ctx.get(),llama_batch_get_one(&prompt[i],1)))) return;
                    auto * hybrid = static_cast<llama_memory_hybrid *>(llama_get_memory(ctx.get()));
                    auto * stream = hybrid->get_mem_attn()->get_kv_stream();
                    auto snapshot = [&] {
                        std::vector<uint8_t> data(llama_state_get_size(ctx.get()));
                        data.resize(llama_state_get_data(ctx.get(),data.data(),data.size())); return data;
                    };
                    const auto saved = snapshot();
                    const auto initial_frontier = stream->tokens();
                    const auto host = stream->host();
                    const auto consumer = stream->memory_consumer();
                    const auto next = [&]() {
                        llama_set_kv_stream_decode(ctx.get(),phase == llama_memory_text_phase::decode);
                        const int32_t rows = phase == llama_memory_text_phase::decode ? 1 : 3;
                        return llama_decode(ctx.get(),llama_batch_get_one(prompt.data()+4,rows)) == 0;
                    };
                    const auto generate = [&]() {
                        std::vector<llama_token> generated;
                        llama_set_kv_stream_decode(ctx.get(),true);
                        for (size_t i = 0; i < 16; ++i) {
                            const auto * logits = llama_get_logits_ith(ctx.get(),-1);
                            auto token = llama_token(std::max_element(logits,logits+llama_vocab_n_tokens(vocab))-logits);
                            generated.push_back(token);
                            if (llama_decode(ctx.get(),llama_batch_get_one(&token,1)) != 0) { generated.clear(); break; }
                        }
                        return generated;
                    };
                    if (!t.assert_true(next())) return;
                    std::vector<float> expected_next(llama_get_logits_ith(ctx.get(),-1),
                        llama_get_logits_ith(ctx.get(),-1)+llama_vocab_n_tokens(vocab));
                    const auto expected_tokens = generate();
                    if (!t.assert_equal(size_t(16),expected_tokens.size())) return;
                    recurrent_snapshot expected_recurrent;
                    hybrid->get_mem_recr()->state_write(expected_recurrent,0,0);
                    std::vector<float> expected_logits(llama_get_logits_ith(ctx.get(),-1),
                        llama_get_logits_ith(ctx.get(),-1)+llama_vocab_n_tokens(vocab));
                    if (!t.assert_equal(saved.size(),llama_state_set_data(ctx.get(),saved.data(),saved.size()))) return;
                    if (!t.assert_true(next())) return;
                    const auto control_tokens = generate();
                    t.out << "checkpoint-only matching tokens=" << (control_tokens == expected_tokens) << '\n';
                    if (!t.assert_true(control_tokens == expected_tokens) ||
                            !t.assert_equal(prefill_state.size(),llama_state_set_data(ctx.get(),prefill_state.data(),prefill_state.size()))) return;
                    llama_set_kv_stream_decode(ctx.get(),true);
                    for (size_t i = 0; i < 4; ++i)
                        if (!t.assert_equal(0,llama_decode(ctx.get(),llama_batch_get_one(&prompt[i],1)))) return;
                    t.assert_true(snapshot() == saved);
                    if (!streamed) {
#ifdef KV_STREAM_TILE_NATIVE_TEST
                        if (kv_stream_test_native_graphs_enabled()) t.assert_true(stream->captured_layers() > 0);
                        else t.assert_equal(size_t(0),stream->captured_layers());
#endif
                    }
                    const auto revision = stream->binding_view().revision;
                    if (!t.assert_true(llama_context_suspend_kv_device(ctx.get()))) return;
                    t.assert_equal(size_t(0),stream->captured_layers());
                    if (!t.assert_true(llama_context_resume_kv_device(ctx.get(),phase))) return;
                    t.assert_true(stream->host() == host && stream->memory_consumer() == consumer);
                    t.assert_equal(initial_frontier,stream->tokens());
                    t.assert_true(stream->binding_view().revision > revision);
                    t.assert_equal(size_t(0),stream->captured_layers());
                    t.assert_true(snapshot() == saved);
                    if (!t.assert_true(next())) return;
                    float next_error = 0;
                    const auto * next_logits = llama_get_logits_ith(ctx.get(),-1);
                    for (size_t i = 0; i < expected_next.size(); ++i) next_error = std::max(next_error,std::abs(next_logits[i]-expected_next[i]));
                    t.out << "resume first-batch error=" << next_error << '\n';
                    t.assert_equal(0.0f,next_error);
                    const auto actual_tokens = generate();
                    t.assert_true(actual_tokens == expected_tokens);
                    const auto * actual_logits = llama_get_logits_ith(ctx.get(),-1);
                    float maximum = 0;
                    for (size_t i = 0; i < expected_logits.size(); ++i) maximum = std::max(maximum,std::abs(actual_logits[i]-expected_logits[i]));
                    t.out << "resume max_logit_error=" << maximum << ", matching tokens=" << (actual_tokens == expected_tokens) << '\n';
                    t.assert_equal(0.0f,maximum);
                    recurrent_snapshot actual_recurrent;
                    hybrid->get_mem_recr()->state_write(actual_recurrent,0,0);
                    t.assert_true(expected_recurrent.metadata == actual_recurrent.metadata && expected_recurrent.tensors == actual_recurrent.tensors);
                });
            }
        }
        return t.summary();
    }
    if (suspend_control) {
        t.test("populated_hybrid_state_survives_device_suspension", [&](testing & t) {
            auto p = llama_context_default_params();
            p.n_ctx = 1024; p.n_batch = p.n_ubatch = 256;
            p.n_seq_max = 1; p.n_rs_seq = 0;
            p.type_k = GGML_TYPE_Q8_0; p.type_v = GGML_TYPE_Q4_0;
            p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            p.kv_stream_pool_bytes = 16*1048576;
            context_ptr ctx(llama_init_from_model(model.get(),p),llama_free);
            if (!t.assert_true(bool(ctx))) return;
            llama_set_kv_stream_decode(ctx.get(),false);
            if (!t.assert_equal(0,llama_decode(ctx.get(),llama_batch_get_one(prompt.data(),256)))) return;
            llama_set_kv_stream_decode(ctx.get(),true);
            for (size_t i = 256; i < 260; ++i)
                if (!t.assert_equal(0,llama_decode(ctx.get(),llama_batch_get_one(&prompt[i],1)))) return;
            auto * hybrid = static_cast<llama_memory_hybrid *>(llama_get_memory(ctx.get()));
            auto * stream = hybrid->get_mem_attn()->get_kv_stream();
            if (!t.assert_true(stream && stream->complete())) return;
            auto snapshot = [&] {
                std::vector<uint8_t> data(llama_state_get_size(ctx.get()));
                const size_t written = llama_state_get_data(ctx.get(),data.data(),data.size());
                data.resize(written); return data;
            };
            const auto before = snapshot();
            recurrent_snapshot recurrent;
            hybrid->get_mem_recr()->state_write(recurrent,0,0);
            const auto host = stream->host();
            const auto frontier = llama_memory_seq_pos_max(llama_get_memory(ctx.get()),0);
            if (!t.assert_true(llama_context_suspend_kv_device(ctx.get()))) return;
            t.assert_true(stream->host() == host && stream->device_suspended());
            t.assert_equal(size_t(260),stream->tokens());
            t.assert_equal(size_t(0),stream->device_grant_bytes());
            t.assert_true(snapshot() == before);
            recurrent_snapshot after;
            hybrid->get_mem_recr()->state_write(after,0,0);
            t.assert_true(recurrent.metadata == after.metadata && recurrent.tensors == after.tensors);
            t.assert_equal(frontier,llama_memory_seq_pos_max(llama_get_memory(ctx.get()),0));
            t.assert_true(llama_decode(ctx.get(),llama_batch_get_one(&prompt[260],1)) != 0);
            t.assert_true(snapshot() == before);
            t.assert_true(llama_context_suspend_kv_device(ctx.get()));
        });
        if (!suspend_resident_only) t.test("streamed_hybrid_suspension_retires_cross_token_prefetch", [&](testing & t) {
            auto p = llama_context_default_params();
            p.n_ctx = 8192; p.n_batch = p.n_ubatch = 64;
            p.n_seq_max = 1; p.n_rs_seq = 0;
            p.type_k = GGML_TYPE_Q8_0; p.type_v = GGML_TYPE_Q4_0;
            p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            p.kv_stream_pool_bytes = 16*1048576;
            context_ptr ctx(llama_init_from_model(model.get(),p),llama_free);
            if (!t.assert_true(bool(ctx))) return;
            std::vector<llama_token> history(6144);
            for (size_t i = 0; i < history.size(); ++i) history[i] = prompt[i%prompt.size()];
            llama_set_kv_stream_decode(ctx.get(),false);
            for (size_t first = 0; first < history.size(); first += 64)
                if (!t.assert_equal(0,llama_decode(ctx.get(),llama_batch_get_one(history.data()+first,64)))) return;
            llama_set_kv_stream_decode(ctx.get(),true);
            for (size_t i = 0; i < 4; ++i)
                if (!t.assert_equal(0,llama_decode(ctx.get(),llama_batch_get_one(&prompt[i],1)))) return;
            auto * hybrid = static_cast<llama_memory_hybrid *>(llama_get_memory(ctx.get()));
            auto * stream = hybrid->get_mem_attn()->get_kv_stream();
            llama_kv_stream_runtime_diagnostics before;
            if (!t.assert_true(stream && stream->runtime_diagnostics(before) && before.streaming_active)) return;
            t.assert_true(stream->prefetch_primed());
            const auto host = stream->host();
            const auto consumer = stream->memory_consumer();
            std::vector<uint8_t> bytes(host->bytes());
            std::memcpy(bytes.data(),ggml_backend_buffer_get_base(host->buffer()),bytes.size());
            recurrent_snapshot recurrent;
            hybrid->get_mem_recr()->state_write(recurrent,0,0);
            if (!t.assert_true(llama_context_suspend_kv_device(ctx.get()))) return;
            t.assert_true(stream->device_suspended() && !stream->prefetch_primed());
            t.assert_equal(size_t(6148),stream->tokens());
            t.assert_true(stream->memory_consumer() == consumer && stream->host() == host);
            t.assert_equal(size_t(0),stream->device_grant_bytes());
            t.assert_true(std::memcmp(bytes.data(),ggml_backend_buffer_get_base(host->buffer()),bytes.size()) == 0);
            recurrent_snapshot after;
            hybrid->get_mem_recr()->state_write(after,0,0);
            t.assert_true(recurrent.metadata == after.metadata && recurrent.tensors == after.tensors);
            t.assert_true(llama_decode(ctx.get(),llama_batch_get_one(&prompt[4],1)) != 0);
        });
        return t.summary();
    }
    if (mtp_sustained_control) {
        t.set_filter("mtp_serial_requests_change_prefill_and_reset_prompt_cache");
        t.test("mtp_serial_requests_change_prefill_and_reset_prompt_cache", [&](testing & t) {
            auto p = llama_context_default_params();
            p.n_ctx = 1024;
            p.n_batch = p.n_ubatch = 256;
            p.n_threads = p.n_threads_batch = 8;
            p.type_k = GGML_TYPE_Q8_0;
            p.type_v = GGML_TYPE_Q4_0;
            p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            p.kv_stream_pool_bytes = 16*1048576;
            p.kv_stream_auxiliary_layers = 1;
            p.n_rs_seq = mtp_sustained_rs_control ? 3 : 0;
            context_ptr target_ctx(llama_init_from_model(model.get(), p), llama_free);
            if (!t.assert_true(bool(target_ctx))) return;
            auto * hybrid = static_cast<llama_memory_hybrid *>(llama_get_memory(target_ctx.get()));
            auto * stream = hybrid->get_mem_attn()->get_kv_stream();
            if (!t.assert_true(stream != nullptr)) return;
            auto cache = stream->auxiliary_cache();
            if (!t.assert_true(bool(cache))) return;
            auto draft_params = p;
            draft_params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
            draft_params.ctx_other = target_ctx.get();
            draft_params.kv_stream_pool_bytes = 0;
            draft_params.kv_stream_auxiliary_layers = 0;
            draft_params.n_rs_seq = 0;
            context_ptr draft_ctx(llama_init_from_model(model.get(), draft_params), llama_free);
            if (!t.assert_true(bool(draft_ctx))) return;
            draft_params.ctx_other = nullptr;
            context_ptr stock_ctx(llama_init_from_model(model.get(), draft_params), llama_free);
            if (!t.assert_true(bool(stock_ctx))) return;
            auto target_batch = llama_batch_init(256, 0, 1);
            auto mtp_batch = llama_batch_init(256, llama_model_n_embd(model.get()), 1);
            std::vector<llama_token> mtp_tokens(256, prompt.front());
            mtp_batch.token = mtp_tokens.data();
            struct batch_cleanup {
                llama_batch & target, & mtp;
                ~batch_cleanup() { mtp.token = nullptr; llama_batch_free(target); llama_batch_free(mtp); }
            } cleanup{target_batch, mtp_batch};
            std::fill(mtp_batch.embd, mtp_batch.embd + 256*llama_model_n_embd(model.get()), 0.0f);
            const size_t vocab_size = llama_vocab_n_tokens(vocab);
            size_t settled_free = 0;
            float worst = 0;
            for (size_t cycle = 0; cycle < 24; ++cycle) {
                const size_t prefill = std::array<size_t, 8>{32, 128, 256, 64, 192, 32, 256, 96}[cycle % 8];
                if (cycle) {
                    llama_synchronize(stock_ctx.get());
                    llama_synchronize(draft_ctx.get());
                    llama_synchronize(target_ctx.get());
                    llama_memory_clear(llama_get_memory(stock_ctx.get()), true);
                    llama_memory_clear(llama_get_memory(draft_ctx.get()), true);
                    llama_memory_clear(llama_get_memory(target_ctx.get()), true);
                    if (!t.assert_equal(size_t(0), stream->tokens()) ||
                            !t.assert_equal(size_t(0), cache->tokens()) ||
                            !t.assert_true(!stream->has_mtp_layer())) return;
                }
                target_batch.n_tokens = mtp_batch.n_tokens = int32_t(prefill);
                for (size_t i = 0; i < prefill; ++i) {
                    const auto token = prompt[(cycle*37 + i)%prompt.size()];
                    target_batch.token[i] = mtp_tokens[i] = token;
                    target_batch.pos[i] = mtp_batch.pos[i] = llama_pos(i);
                    target_batch.n_seq_id[i] = mtp_batch.n_seq_id[i] = 1;
                    target_batch.seq_id[i][0] = mtp_batch.seq_id[i][0] = 0;
                    target_batch.logits[i] = mtp_batch.logits[i] = i + 1 == prefill;
                }
                llama_set_kv_stream_decode(target_ctx.get(), false);
                if (!t.assert_equal(0, llama_decode(target_ctx.get(), target_batch)) ||
                        !t.assert_equal(0, llama_decode(draft_ctx.get(), mtp_batch)) ||
                        !t.assert_equal(0, llama_decode(stock_ctx.get(), mtp_batch))) return;
                if (!t.assert_equal(prefill, stream->tokens()) ||
                        !t.assert_equal(prefill, cache->tokens())) return;
                target_batch.n_tokens = mtp_batch.n_tokens = 1;
                target_batch.token[0] = mtp_tokens[0] = prompt[(cycle*53 + prefill)%prompt.size()];
                target_batch.pos[0] = mtp_batch.pos[0] = llama_pos(prefill);
                target_batch.logits[0] = mtp_batch.logits[0] = 1;
                llama_set_kv_stream_decode(target_ctx.get(), true);
                if (!t.assert_equal(0, llama_decode(target_ctx.get(), target_batch)) ||
                        !t.assert_true(llama_kv_stream_mtp_prepare(target_ctx.get(), 1))) return;
                for (int attempt = 0; attempt < 2; ++attempt) {
                    if (attempt) {
                        if (!t.assert_true(llama_memory_seq_rm(
                                llama_get_memory(draft_ctx.get()), 0, llama_pos(prefill), -1) &&
                                llama_memory_seq_rm(
                                llama_get_memory(stock_ctx.get()), 0, llama_pos(prefill), -1))) return;
                        if (!t.assert_equal(prefill, cache->tokens())) return;
                        mtp_tokens[0] = prompt[(cycle*53 + prefill + 1)%prompt.size()];
                    }
                    if (!t.assert_equal(0, llama_decode(draft_ctx.get(), mtp_batch)) ||
                            !t.assert_equal(0, llama_decode(stock_ctx.get(), mtp_batch))) return;
                    llama_synchronize(draft_ctx.get());
                    llama_synchronize(stock_ctx.get());
                    const float * actual = llama_get_logits_ith(draft_ctx.get(), -1);
                    const float * expected = llama_get_logits_ith(stock_ctx.get(), -1);
                    if (!t.assert_true(actual && expected)) return;
                    float error = 0;
                    for (size_t i = 0; i < vocab_size; ++i) {
                        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {
                            t.assert_true(false); return;
                        }
                        error = std::max(error, std::abs(actual[i] - expected[i]));
                    }
                    worst = std::max(worst, error);
                    t.assert_true(error < .05f);
                    t.assert_equal(
                        std::max_element(actual, actual + vocab_size) - actual,
                        std::max_element(expected, expected + vocab_size) - expected);
                    t.assert_equal(prefill + 1, cache->tokens());
                    t.assert_true(stream->has_mtp_layer());
                }
                auto * device = ggml_backend_dev_by_name("CUDA0");
                size_t free_bytes = 0, total_bytes = 0;
                if (device) ggml_backend_dev_memory(device, &free_bytes, &total_bytes);
                if (cycle == 7) settled_free = free_bytes;
                if (cycle > 7 && device) t.assert_true(free_bytes + 16*1048576 >= settled_free);
                t.out << "MTP serial cycle " << cycle << " prefill=" << prefill
                    << " free VRAM MiB=" << free_bytes/1048576 << '\n';
            }
            t.out << "MTP serial max stock-logit error=" << worst << '\n';
        });
        return t.summary();
    }
    t.test("invalid_streaming_context_modes_fail_before_execution", [&](testing & t) {
        for (int mode = 0; mode < 6; ++mode) {
            auto p = llama_context_default_params(); p.kv_stream_pool_bytes = 16*1048576; p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            if (mode == 0) p.n_seq_max = 2;
            if (mode == 1) p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
            if (mode == 2) p.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
            if (mode == 3) p.offload_kqv = false;
            if (mode == 4) p.n_rs_seq = 1;
            if (mode == 5) p.shared_device_memory_bytes = 640*1048576;
            context_ptr context(llama_init_from_model(model.get(),p),llama_free); t.assert_true(!context);
        }
    });
    if (auxiliary_control || embedded_mtp_control || target_tg3_control || target_stream_tg4_control) t.test(auxiliary_test_name, [&](testing & t) {
        auto p = llama_context_default_params();
        p.n_ctx = target_tg3_control ? 8192 : target_stream_tg4_control ? 4096 : 1024; p.n_batch = p.n_ubatch = 256;
        if (target_stream_tg4_control) p.n_outputs_max = p.n_outputs_max_per_seq = 4;
        p.n_threads = p.n_threads_batch = 8;
        p.type_k = GGML_TYPE_Q8_0; p.type_v = GGML_TYPE_Q4_0;
        p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        p.kv_stream_pool_bytes = 16*1048576;
        if (vision_mtp_handoff) {
            p.type_k = test_cache_k;
            p.type_v = test_cache_v;
            p.n_ctx = 8192;
            p.kv_stream_pool_bytes = 0;
            p.shared_device_memory_bytes = 640*1048576;
        }
        if (target_tg3_control) {
            p.kv_stream_pool_bytes = 0;
            p.shared_device_memory_bytes = 1800*1048576;
        }
        p.kv_stream_auxiliary_layers = 1;
        context_ptr context(llama_init_from_model(model.get(), p), llama_free);
        if (!t.assert_true(bool(context))) return;
        auto * hybrid = static_cast<llama_memory_hybrid *>(llama_get_memory(context.get()));
        auto * target = hybrid->get_mem_attn()->get_kv_stream();
        if (!t.assert_true(target != nullptr)) return;
        auto mtp = target->auxiliary_cache();
        if (!t.assert_true(bool(mtp))) return;
        t.assert_true(target->host()->cache_id() != mtp->host()->cache_id());
        t.assert_equal(uint32_t(1), mtp->host()->config().layers);
        t.assert_equal(size_t(0), mtp->tokens());
        const auto view = target->binding_view();
        t.assert_equal(target->host()->config().layers + 1, view.config.layers);
        if (!t.assert_equal(size_t(2), view.config.caches.size())) return;
        t.assert_equal(target->host()->cache_id(), view.config.caches[0].id);
        t.assert_equal(mtp->host()->cache_id(), view.config.caches[1].id);
        const auto breakdown = hybrid->get_mem_attn()->memory_breakdown();
        const auto host_type = ggml_backend_buffer_get_type(target->host()->buffer());
        t.assert_equal(target->host()->bytes() + mtp->host()->bytes(), breakdown.at(host_type));
        llama_token token = prompt.front();
        llama_set_kv_stream_decode(context.get(), false);
        t.assert_equal(0, llama_decode(context.get(), llama_batch_get_one(&token, 1)));
        t.assert_equal(size_t(1), target->tokens());
        t.assert_equal(size_t(0), mtp->tokens());
        if (!embedded_mtp_control && !target_tg3_control && !target_stream_tg4_control) return;
        auto target_batch = llama_batch_init(256, 0, 1);
        struct target_batch_guard {
            llama_batch & batch;
            ~target_batch_guard() { llama_batch_free(batch); }
        } target_batch_owner{target_batch};
        target_batch.n_tokens = 256;
        for (int i = 0; i < 256; ++i) {
            target_batch.token[i] = prompt[size_t(i + 1)];
            target_batch.pos[i] = i + 1;
            target_batch.n_seq_id[i] = 1;
            target_batch.seq_id[i][0] = 0;
            target_batch.logits[i] = i == 255;
        }
        if (!t.assert_equal(0, llama_decode(context.get(), target_batch))) return;
        if (target_stream_tg4_control) {
            for (size_t first = 257; first < 3001; first += 256) {
                const size_t count = std::min<size_t>(256, 3001 - first);
                target_batch.n_tokens = int(count);
                for (size_t i = 0; i < count; ++i) {
                    target_batch.token[i] = prompt[first + i];
                    target_batch.pos[i] = llama_pos(first + i);
                    target_batch.n_seq_id[i] = 1;
                    target_batch.seq_id[i][0] = 0;
                    target_batch.logits[i] = i + 1 == count;
                }
                llama_set_kv_stream_decode(context.get(), false);
                if (!t.assert_equal(0, llama_decode(context.get(), target_batch))) return;
            }
            t.assert_equal(size_t(3001), target->tokens());
            target_batch.n_tokens = 4;
            for (int i = 0; i < 4; ++i) {
                target_batch.token[i] = prompt[size_t(3001 + i)];
                target_batch.pos[i] = 3001 + i;
                target_batch.n_seq_id[i] = 1;
                target_batch.seq_id[i][0] = 0;
                target_batch.logits[i] = i == 3;
            }
            llama_set_kv_stream_decode(context.get(), true);
            if (!t.assert_equal(0, llama_decode(context.get(), target_batch))) return;
            t.assert_equal(size_t(3005), target->tokens());
            llama_kv_stream_runtime_diagnostics diagnostics;
            if (t.assert_true(target->runtime_diagnostics(diagnostics)))
                t.assert_true(diagnostics.streaming_active);
            llama_synchronize(context.get());
            const float * streamed_logits = llama_get_logits_ith(context.get(), -1);
            const size_t vocab_size = llama_vocab_n_tokens(vocab);
            if (!t.assert_true(streamed_logits != nullptr)) return;
            const std::vector<float> streamed(streamed_logits, streamed_logits + vocab_size);
            auto stock_params = p;
            stock_params.kv_stream_pool_bytes = 0;
            stock_params.kv_stream_auxiliary_layers = 0;
            context_ptr stock(llama_init_from_model(model.get(), stock_params), llama_free);
            if (!t.assert_true(bool(stock))) return;
            target_batch.n_tokens = 1;
            target_batch.token[0] = prompt[0];
            target_batch.pos[0] = 0;
            target_batch.n_seq_id[0] = 1;
            target_batch.seq_id[0][0] = 0;
            target_batch.logits[0] = 1;
            if (!t.assert_equal(0, llama_decode(stock.get(), target_batch))) return;
            for (size_t first = 1; first < 3001; first += 256) {
                const size_t count = std::min<size_t>(256, 3001 - first);
                target_batch.n_tokens = int(count);
                for (size_t i = 0; i < count; ++i) {
                    target_batch.token[i] = prompt[first + i];
                    target_batch.pos[i] = llama_pos(first + i);
                    target_batch.n_seq_id[i] = 1;
                    target_batch.seq_id[i][0] = 0;
                    target_batch.logits[i] = i + 1 == count;
                }
                if (!t.assert_equal(0, llama_decode(stock.get(), target_batch))) return;
            }
            target_batch.n_tokens = 4;
            for (int i = 0; i < 4; ++i) {
                target_batch.token[i] = prompt[size_t(3001 + i)];
                target_batch.pos[i] = 3001 + i;
                target_batch.n_seq_id[i] = 1;
                target_batch.seq_id[i][0] = 0;
                target_batch.logits[i] = i == 3;
            }
            if (!t.assert_equal(0, llama_decode(stock.get(), target_batch))) return;
            llama_synchronize(stock.get());
            const float * expected = llama_get_logits_ith(stock.get(), -1);
            if (!t.assert_true(expected != nullptr)) return;
            float worst = 0;
            for (size_t i = 0; i < vocab_size; ++i)
                worst = std::max(worst, std::abs(streamed[i] - expected[i]));
            t.out << "streamed TG4 at 3005 tokens max logit error=" << worst << '\n';
            t.assert_true(worst < .05f);
            t.assert_equal(
                std::max_element(streamed.begin(), streamed.end()) - streamed.begin(),
                std::max_element(expected, expected + vocab_size) - expected);
            return;
        }
            context_ptr stock_draft(nullptr, llama_free);
            if (target_with_stock_draft) {
                auto draft_only = p;
                draft_only.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
                draft_only.ctx_other = nullptr;
                draft_only.kv_stream_pool_bytes = 0;
                draft_only.kv_stream_auxiliary_layers = 0;
                stock_draft.reset(llama_init_from_model(model.get(), draft_only));
                if (!t.assert_true(bool(stock_draft))) return;
            }
        if (target_tg3_control) {
            for (size_t first = 257; first < 7933; first += 256) {
                const size_t count = std::min<size_t>(256, 7933 - first);
                target_batch.n_tokens = int(count);
                for (size_t i = 0; i < count; ++i) {
                    target_batch.token[i] = prompt[(first + i) % prompt.size()];
                    target_batch.pos[i] = llama_pos(first + i);
                    target_batch.n_seq_id[i] = 1;
                    target_batch.seq_id[i][0] = 0;
                    target_batch.logits[i] = i + 1 == count;
                }
                llama_set_kv_stream_decode(context.get(), false);
                if (!t.assert_equal(0, llama_decode(context.get(), target_batch))) return;
            }
            target_batch.n_tokens = 3;
            for (int i = 0; i < 3; ++i) {
                target_batch.token[i] = prompt[size_t(7933 + i) % prompt.size()];
                target_batch.pos[i] = 7933 + i;
                target_batch.n_seq_id[i] = 1;
                target_batch.seq_id[i][0] = 0;
                target_batch.logits[i] = i == 2;
            }
            llama_set_kv_stream_decode(context.get(), true);
            t.assert_equal(0, llama_decode(context.get(), target_batch));
            return;
        }
        t.assert_equal(size_t(257), target->tokens());
        if (vision_mtp_handoff) t.assert_true(!context->get_compute_memory()->can_suspend_for_vision());
        auto draft_params = p;
        draft_params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        draft_params.ctx_other = context.get();
        draft_params.kv_stream_pool_bytes = 0;
        draft_params.shared_device_memory_bytes = 0;
        draft_params.kv_stream_auxiliary_layers = 0;
        context_ptr draft(llama_init_from_model(model.get(), draft_params), llama_free);
        if (!t.assert_true(bool(draft))) return;
        if (vision_mtp_handoff) {
            t.assert_true("attached draft permits vision", context->get_compute_memory()->can_suspend_for_vision());
            t.assert_true("draft cannot lend target parent", !draft->get_compute_memory()->can_suspend_for_vision());
        }
        auto ordinary_params = draft_params;
        ordinary_params.ctx_other = nullptr;
        context_ptr ordinary_draft(llama_init_from_model(model.get(), ordinary_params), llama_free);
        if (!t.assert_true(bool(ordinary_draft))) return;
        auto * ordinary_kv = static_cast<llama_kv_cache *>(llama_get_memory(ordinary_draft.get()));
        t.assert_true(ordinary_kv->mtp_auxiliary_cache() == nullptr);
        auto * draft_kv = static_cast<llama_kv_cache *>(llama_get_memory(draft.get()));
        t.assert_true(draft_kv->get_kv_stream() == nullptr);
        t.assert_true(draft_kv->mtp_auxiliary_cache() == mtp);
        t.assert_equal(size_t(0), draft_kv->mtp_auxiliary_cache()->tokens());
        t.assert_equal(mtp->host()->cache_id(), draft_kv->mtp_auxiliary_cache()->host()->cache_id());
        auto mtp_batch = llama_batch_init(257, llama_model_n_embd(model.get()), 1);
        std::vector<llama_token> mtp_tokens(257, prompt.front());
        mtp_batch.token = mtp_tokens.data();
        struct mtp_batch_guard {
            llama_batch & batch;
            ~mtp_batch_guard() { batch.token = nullptr; llama_batch_free(batch); }
        } mtp_batch_owner{mtp_batch};
        std::fill(mtp_batch.embd, mtp_batch.embd + 257*llama_model_n_embd(model.get()), 0.0f);
        const auto append_mtp = [&](int first, int count) {
            mtp_batch.n_tokens = count;
            for (int i = 0; i < count; ++i) {
                mtp_batch.pos[i] = first + i;
                mtp_batch.n_seq_id[i] = 1;
                mtp_batch.seq_id[i][0] = 0;
                mtp_batch.logits[i] = i + 1 == count;
            }
            const int attached_status = llama_decode(draft.get(), mtp_batch);
            const int stock_status = llama_decode(ordinary_draft.get(), mtp_batch);
            return attached_status == 0 && stock_status == 0;
        };
        if (!t.assert_true(append_mtp(0, 256))) return;
        t.assert_equal(size_t(256), mtp->tokens());
        t.assert_equal(size_t(0), mtp->frontiers().device);
        if (vision_mtp_handoff) {
            auto * owner = llama_context_compute_memory(context.get());
            if (!t.assert_true(owner && owner->text_phase().phase == llama_memory_text_phase::prefill)) return;
            if (!t.assert_true(llama_kv_stream_mtp_prepare(context.get(), 0))) return;
            if (!t.assert_true(owner->text_phase().phase == llama_memory_text_phase::decode)) return;
        } else if (!t.assert_true(target->acquire_mtp_layer())) return;
        const auto initial_upload = target->mtp_layer_population();
        t.assert_true(initial_upload.bytes > 0);
        if (!t.assert_true(append_mtp(256, 1))) return;
        t.assert_equal(size_t(257), mtp->tokens());
        t.assert_equal(size_t(1), draft_kv->mtp_span_attention_calls());
        t.assert_equal(size_t(0), mtp->frontiers().device);
        const auto & tail_layout = mtp->host()->layout();
        t.assert_true(target->has_mtp_layer());
        t.assert_equal(initial_upload.bytes + tail_layout.k_token_bytes +
            tail_layout.v_token_bytes, target->mtp_layer_population().bytes);
        ggml_kv_stream_span_plan_view advanced;
        if (!t.assert_true(ggml_kv_stream_span_plan_get_view(
                target->mtp_layer_plan(1), advanced))) return;
        t.assert_equal(size_t(257), advanced.active_tokens);
        llama_synchronize(draft.get());
        llama_synchronize(ordinary_draft.get());
        const float * attached_logits = llama_get_logits_ith(draft.get(), -1);
        const float * stock_logits = llama_get_logits_ith(ordinary_draft.get(), -1);
        if (!t.assert_true(attached_logits != nullptr && stock_logits != nullptr)) return;
        const size_t vocab_size = llama_vocab_n_tokens(vocab);
        float maximum_logit_error = 0;
        for (size_t i = 0; i < vocab_size; ++i) {
            if (!std::isfinite(attached_logits[i]) || !std::isfinite(stock_logits[i])) {
                t.assert_true(false);
                return;
            }
            maximum_logit_error = std::max(maximum_logit_error,
                std::abs(attached_logits[i] - stock_logits[i]));
        }
        t.out << "real MTP TG1 max logit error=" << maximum_logit_error << '\n';
        t.assert_true(maximum_logit_error < .05f);
        t.assert_equal(
            std::max_element(attached_logits, attached_logits + vocab_size) - attached_logits,
            std::max_element(stock_logits, stock_logits + vocab_size) - stock_logits);
        llama_kv_stream_host_layer host_rows;
        if (!t.assert_true(mtp->host()->layer(0, host_rows))) return;
        const auto & kv_layout = mtp->host()->layout();
        const int mtp_layer = llama_model_n_layer(model.get());
        std::vector<uint8_t> stock_k(257*kv_layout.k_token_bytes);
        std::vector<uint8_t> stock_v(257*kv_layout.v_token_bytes);
        ggml_backend_tensor_get(draft_kv->get_k_storage(mtp_layer), stock_k.data(), 0, stock_k.size());
        ggml_backend_tensor_get(draft_kv->get_v_storage(mtp_layer), stock_v.data(), 0, stock_v.size());
        t.assert_true(std::memcmp(stock_k.data(), host_rows.k, stock_k.size()) == 0);
        t.assert_true(std::memcmp(stock_v.data(), host_rows.v, stock_v.size()) == 0);
        const auto prior_generation = mtp->identity().generation;
        const std::vector<uint8_t> prior_k(static_cast<const uint8_t *>(host_rows.k),
                static_cast<const uint8_t *>(host_rows.k) + stock_k.size());
        const std::vector<uint8_t> prior_v(static_cast<const uint8_t *>(host_rows.v),
                static_cast<const uint8_t *>(host_rows.v) + stock_v.size());
        const auto before_rewrite = target->mtp_layer_population();
        if (!t.assert_true(llama_memory_seq_rm(llama_get_memory(ordinary_draft.get()), 0, 256, -1))) return;
        if (!t.assert_true(llama_memory_seq_rm(llama_get_memory(draft.get()), 0, 256, -1))) return;
        t.assert_true(target->has_mtp_layer());
        t.assert_equal(before_rewrite.bytes, target->mtp_layer_population().bytes);
        ggml_kv_stream_span_plan_view rewritten_prefix;
        if (!t.assert_true(ggml_kv_stream_span_plan_get_view(
                target->mtp_layer_plan(1), rewritten_prefix))) return;
        t.assert_equal(size_t(256), rewritten_prefix.active_tokens);

        mtp_tokens[0] = prompt[1];
        if (!t.assert_true(append_mtp(256, 1))) return;
        t.assert_equal(size_t(257), mtp->tokens());
        t.assert_true(mtp->identity().generation > prior_generation);
        t.assert_equal(size_t(2), draft_kv->mtp_span_attention_calls());
        t.assert_true(std::memcmp(prior_k.data(), host_rows.k, 256*kv_layout.k_token_bytes) == 0);
        t.assert_true(std::memcmp(prior_v.data(), host_rows.v, 256*kv_layout.v_token_bytes) == 0);
        t.assert_true(std::memcmp(prior_k.data() + 256*kv_layout.k_token_bytes,
                static_cast<const uint8_t *>(host_rows.k) + 256*kv_layout.k_token_bytes,
                kv_layout.k_token_bytes) != 0);
        t.assert_true(target->has_mtp_layer());
        t.assert_equal(before_rewrite.bytes + kv_layout.k_token_bytes +
            kv_layout.v_token_bytes, target->mtp_layer_population().bytes);
        size_t verified = 257;
        for (uint32_t width : {2u, 3u, 4u}) {
            target_batch.n_tokens = int(width);
            for (uint32_t i = 0; i < width; ++i) {
                target_batch.token[i] = prompt[verified + i];
                target_batch.pos[i] = llama_pos(verified + i);
                target_batch.n_seq_id[i] = 1;
                target_batch.seq_id[i][0] = 0;
                target_batch.logits[i] = i + 1 == width;
            }
            llama_set_kv_stream_decode(context.get(), true);
            t.out << "real target TG" << width << " verify begin\n";
            if (!t.assert_equal(0, llama_decode(context.get(), target_batch))) return;
            if (width == 2) t.assert_true(!target->has_mtp_layer());
            t.assert_equal(verified + width, target->tokens());
            if (!t.assert_true(llama_kv_stream_mtp_prepare(context.get(), width == 4 ? 4 : 0))) return;
            const auto before_tail = target->mtp_layer_population();
            mtp_tokens[0] = prompt.front();
            t.out << "real MTP TG" << width << " catch-up begin\n";
            if (!t.assert_true(append_mtp(int(verified), int(width)))) return;
            verified += width;
            t.assert_equal(verified, mtp->tokens());
            t.assert_equal(size_t(width + 1), draft_kv->mtp_span_attention_calls());
            t.assert_equal(before_tail.bytes + width*(kv_layout.k_token_bytes +
                kv_layout.v_token_bytes), target->mtp_layer_population().bytes);
            llama_synchronize(draft.get());
            llama_synchronize(ordinary_draft.get());
            const float * actual = llama_get_logits_ith(draft.get(), -1);
            const float * expected = llama_get_logits_ith(ordinary_draft.get(), -1);
            if (!t.assert_true(actual && expected)) return;
            float worst = 0;
            for (size_t i = 0; i < vocab_size; ++i) {
                if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {
                    t.assert_true(false);
                    return;
                }
                worst = std::max(worst, std::abs(actual[i] - expected[i]));
            }
            t.out << "real MTP TG" << width << " max logit error=" << worst << '\n';
            t.assert_true(worst < .05f);
            t.assert_equal(
                std::max_element(actual, actual + vocab_size) - actual,
                std::max_element(expected, expected + vocab_size) - expected);
            if (width == 4) {
                t.assert_equal(verified + 4, target->mtp_reserved_tokens());
                const auto after_catchup = target->mtp_layer_population();
                const size_t calls = draft_kv->mtp_span_attention_calls();
                for (size_t step = 0; step < 2; ++step) {
                    if (!t.assert_true(append_mtp(int(verified + step), 1))) return;
                    t.assert_equal(verified, target->tokens());
                    t.assert_equal(verified + step + 1, mtp->tokens());
                    t.assert_equal(calls + step + 1, draft_kv->mtp_span_attention_calls());
                    t.assert_true(target->has_mtp_layer());
                    const auto population = target->mtp_layer_population();
                    t.assert_equal(after_catchup.bytes + (step + 1)*
                        (kv_layout.k_token_bytes + kv_layout.v_token_bytes), population.bytes);
                    ggml_kv_stream_span_plan_view draft_plan;
                    if (!t.assert_true(ggml_kv_stream_span_plan_get_view(
                            target->mtp_layer_plan(1), draft_plan))) return;
                    t.assert_equal(verified + step + 1, draft_plan.active_tokens);
                    llama_synchronize(draft.get());
                    llama_synchronize(ordinary_draft.get());
                    const float * live = llama_get_logits_ith(draft.get(), -1);
                    const float * stock = llama_get_logits_ith(ordinary_draft.get(), -1);
                    if (!t.assert_true(live && stock)) return;
                    float error = 0;
                    for (size_t i = 0; i < vocab_size; ++i)
                        error = std::max(error, std::abs(live[i] - stock[i]));
                    t.out << "real MTP draft step " << step << " max logit error=" << error << '\n';
                    t.assert_true(error < .05f);
                }
            }
            if (width == 4) {
                const auto before_partial = target->mtp_layer_population();
                if (!t.assert_true(llama_memory_seq_rm(
                        llama_get_memory(ordinary_draft.get()), 0, llama_pos(verified + 1), -1))) return;
                if (!t.assert_true(llama_memory_seq_rm(
                        llama_get_memory(draft.get()), 0, llama_pos(verified + 1), -1))) return;
                t.assert_equal(verified + 1, mtp->tokens());
                t.assert_true(target->has_mtp_layer());
                t.assert_equal(verified + 4, target->mtp_reserved_tokens());
                t.assert_equal(before_partial.bytes, target->mtp_layer_population().bytes);
                ggml_kv_stream_span_plan_view partial_plan;
                if (!t.assert_true(ggml_kv_stream_span_plan_get_view(
                        target->mtp_layer_plan(1), partial_plan))) return;
                t.assert_equal(verified + 1, partial_plan.active_tokens);

                for (size_t accepted = 0; accepted <= 4; ++accepted) {
                    if (mtp->tokens() != verified) {
                        if (!t.assert_true(llama_memory_seq_rm(
                                llama_get_memory(ordinary_draft.get()), 0, llama_pos(verified), -1) &&
                                llama_memory_seq_rm(
                                llama_get_memory(draft.get()), 0, llama_pos(verified), -1))) return;
                    }
                    t.assert_equal(verified, mtp->tokens());
                    if (target->has_mtp_layer() && !t.assert_true(target->release_mtp_layer())) return;
                    if (!t.assert_true(target->acquire_mtp_layer(4))) return;
                    const size_t calls_before = draft_kv->mtp_span_attention_calls();
                    for (size_t i = 0; i < 4; ++i) {
                        if (!t.assert_true(append_mtp(int(verified + i), 1))) return;
                        t.assert_equal(calls_before + i + 1,
                            draft_kv->mtp_span_attention_calls());
                    }
                    t.assert_equal(verified + 4, mtp->tokens());
                    t.assert_equal(verified, target->tokens());
                    if (accepted == 4) {
                        // Interior holes cannot be represented by the logical prefix frontier.
                        t.assert_true(!llama_memory_seq_rm(
                            llama_get_memory(draft.get()), 0, llama_pos(verified + 1),
                            llama_pos(verified + 2)));
                        t.assert_equal(verified + 4, mtp->tokens());
                        t.assert_true(target->has_mtp_layer());
                    }
                    const uint64_t generation = mtp->identity().generation;
                    const auto before_acceptance = target->mtp_layer_population();
                    if (!t.assert_true(llama_memory_seq_rm(
                            llama_get_memory(ordinary_draft.get()), 0,
                            llama_pos(verified + accepted), -1) &&
                            llama_memory_seq_rm(
                            llama_get_memory(draft.get()), 0,
                            llama_pos(verified + accepted), -1))) return;
                    t.assert_equal(verified + accepted, mtp->tokens());
                    t.assert_true(target->has_mtp_layer());
                    t.assert_equal(verified + 4, target->mtp_reserved_tokens());
                    t.assert_equal(before_acceptance.bytes, target->mtp_layer_population().bytes);
                    ggml_kv_stream_span_plan_view accepted_plan;
                    if (!t.assert_true(ggml_kv_stream_span_plan_get_view(
                            target->mtp_layer_plan(1), accepted_plan))) return;
                    t.assert_equal(verified + accepted, accepted_plan.active_tokens);

                    if (accepted == 4) t.assert_equal(generation, mtp->identity().generation);
                    else t.assert_true(mtp->identity().generation > generation);
                }
                llama_synchronize(draft.get());
                const size_t state_bytes = llama_state_seq_get_size(draft.get(), 0);
                if (!t.assert_true(state_bytes > 0)) return;
                std::vector<uint8_t> saved_state(state_bytes);
                if (!t.assert_equal(state_bytes, llama_state_seq_get_data(
                        draft.get(), saved_state.data(), saved_state.size(), 0))) return;
                if (!t.assert_true(mtp->host()->layer(0, host_rows))) return;
                const size_t prefix_k_bytes = mtp->tokens()*kv_layout.k_token_bytes;
                const size_t prefix_v_bytes = mtp->tokens()*kv_layout.v_token_bytes;
                const std::vector<uint8_t> saved_k(static_cast<const uint8_t *>(host_rows.k),
                    static_cast<const uint8_t *>(host_rows.k) + prefix_k_bytes);
                const std::vector<uint8_t> saved_v(static_cast<const uint8_t *>(host_rows.v),
                    static_cast<const uint8_t *>(host_rows.v) + prefix_v_bytes);
                llama_memory_clear(llama_get_memory(draft.get()), true);
                t.assert_equal(size_t(0), mtp->tokens());
                t.assert_true(!target->has_mtp_layer());
                t.assert_equal(size_t(0), target->mtp_reserved_tokens());
                ggml_backend_buffer_clear(mtp->host()->buffer(), 0);
                if (!t.assert_equal(state_bytes, llama_state_seq_set_data(
                        draft.get(), saved_state.data(), saved_state.size(), 0))) return;
                t.assert_equal(verified + 4, mtp->tokens());
            } else if (!t.assert_true(llama_kv_stream_mtp_release(context.get()))) return;
        }
        if (vision_mtp_handoff) {
            auto * owner = llama_context_compute_memory(context.get());
            const auto snapshot = [](llama_context * ctx) {
                std::vector<uint8_t> bytes(llama_state_seq_get_size(ctx, 0));
                bytes.resize(llama_state_seq_get_data(ctx, bytes.data(), bytes.size(), 0));
                return bytes;
            };
            const auto target_state = snapshot(context.get()), draft_state = snapshot(draft.get());
            const auto tokens = mtp->tokens();
            const auto parent = owner->shared_parent();
            if (!t.assert_true("populated draft permits suspension", owner->can_suspend_for_vision() && llama_context_suspend_kv_device(context.get()))) return;
            t.assert_true("suspension retires MTP ring", owner->kv_device_suspended() && !target->has_mtp_layer());
            t.assert_equal(size_t(0), target->device_grant_bytes());
            t.assert_true("suspension preserves both states", snapshot(context.get()) == target_state && snapshot(draft.get()) == draft_state);
            std::vector<ggml_backend_memory_lease_t> loans;
            if (!t.assert_true(owner->lend_suspended({1024}, loans))) return;
            ggml_backend_buffer_clear(ggml_backend_memory_lease_buffer(loans.front()), 0xa5);
            t.assert_true(!llama_context_resume_kv_device(context.get(), llama_memory_text_phase::prefill));
            for (auto * loan : loans) ggml_backend_memory_lease_free(loan);
            if (!t.assert_true(llama_context_resume_kv_device(context.get(), llama_memory_text_phase::prefill))) return;
            t.assert_true("resume preserves parent and admission", owner->shared_parent() == parent && owner->can_suspend_for_vision());
            t.assert_equal(tokens, mtp->tokens());
            t.assert_true("resume preserves both states", snapshot(context.get()) == target_state && snapshot(draft.get()) == draft_state);
            t.assert_true(llama_memory_seq_rm(llama_get_memory(draft.get()), 0, llama_pos(verified), -1));
            t.assert_true("resume permits new MTP lease", llama_kv_stream_mtp_prepare(context.get(), 4));
            t.assert_true(llama_kv_stream_mtp_release(context.get()));
        }
        auto wrong_ctx = draft_params;
        wrong_ctx.n_ctx = 2048;
        context_ptr mismatched_context(llama_init_from_model(model.get(), wrong_ctx), llama_free);
        t.assert_true(!mismatched_context);
        auto wrong_quant = draft_params;
        wrong_quant.type_k = GGML_TYPE_F16;
        context_ptr mismatched_quant(llama_init_from_model(model.get(), wrong_quant), llama_free);
        t.assert_true(!mismatched_quant);
    });
    t.test("serial_phase_alternation_and_boundary_cancellation_match_stock", [&](testing & t) {
        const auto baseline = evaluate_serial_phases(t,model.get(),prompt,0);
        const auto streamed = evaluate_serial_phases(t,model.get(),prompt,(shared_budget_control ? 640 : 16)*1048576);
        if (!t.assert_equal(baseline.logits.size(),streamed.logits.size()) ||
                !t.assert_equal(size_t(14),streamed.logits.size()) ||
                !t.assert_equal(streamed.logits.size(),streamed.pool_grants.size()) ||
                !t.assert_equal(size_t(3),streamed.prefill_after_decode_us.size()) ||
                !t.assert_equal(baseline.prefill_after_decode_us.size(),
                    streamed.prefill_after_decode_us.size())) return;
        float maximum = 0;
        size_t matching = 0;
        for (size_t phase = 0; phase < baseline.logits.size(); ++phase) {
            if (!t.assert_equal(baseline.logits[phase].size(),streamed.logits[phase].size())) return;
            for (size_t token = 0; token < baseline.logits[phase].size(); ++token) {
                const float actual = streamed.logits[phase][token];
                if (!std::isfinite(actual)) { t.assert_true(false); return; }
                maximum = std::max(maximum,std::abs(baseline.logits[phase][token]-actual));
            }
            matching += std::max_element(baseline.logits[phase].begin(),baseline.logits[phase].end())-
                baseline.logits[phase].begin() ==
                std::max_element(streamed.logits[phase].begin(),streamed.logits[phase].end())-
                streamed.logits[phase].begin();
        }
        t.out << "serial phase max logit error=" << maximum
              << ", matching boundaries=" << matching << '/' << baseline.logits.size() << '\n';
        t.assert_true(maximum < .05f);
        t.assert_equal(baseline.logits.size(),matching);
        t.assert_true(baseline.recurrent.metadata == streamed.recurrent.metadata);
        t.assert_true(baseline.recurrent.tensors == streamed.recurrent.tensors);
        const size_t prefill_pool = streamed.pool_grants.front();
        const size_t decode_pool = streamed.pool_grants[1];
        t.assert_true(decode_pool > prefill_pool);
        for (size_t i = 0; i < 8; ++i) {
            t.assert_equal(i%2 ? decode_pool : prefill_pool,streamed.pool_grants[i]);
        }
        t.assert_equal(prefill_pool,streamed.pool_grants[8]);
        t.assert_equal(decode_pool,streamed.pool_grants[9]);
        t.assert_equal(prefill_pool,streamed.pool_grants[10]);
        t.assert_equal(decode_pool,streamed.pool_grants[11]);
        t.assert_equal(prefill_pool,streamed.pool_grants[12]);
        t.assert_equal(decode_pool,streamed.pool_grants[13]);
        t.out << "decode-to-prefill baseline -> streamed us:";
        for (size_t i = 0; i < streamed.prefill_after_decode_us.size(); ++i) {
            t.out << ' ' << baseline.prefill_after_decode_us[i] << "->"
                  << streamed.prefill_after_decode_us[i] << " (delta "
                  << streamed.prefill_after_decode_us[i]-baseline.prefill_after_decode_us[i] << ')';
        }
        t.out << '\n';
    });
    for (uint32_t ubatch : {256u,512u}) t.test("hybrid_prefill_decode_and_recurrent_state_match_ub_"+std::to_string(ubatch), [&](testing & t) {
        const auto baseline = evaluate(t,model.get(),prompt,ubatch,0,{});
        if (!t.assert_equal(size_t(32),baseline.continuation.size())) return;
        const auto streamed = evaluate(t,model.get(),prompt,ubatch,(shared_budget_control ? 640 : (f16_control || resident_control ? 64 : 16))*1048576,baseline.continuation);
        if (trace_control && baseline.trace.size() == streamed.trace.size()) for (size_t n = 0; n < baseline.trace.size(); ++n) {
            const auto & a = baseline.trace[n]; const auto & b = streamed.trace[n];
            if (a.first != b.first || a.second.size() != b.second.size()) continue;
            double delta = 0, energy = 0; float maximum = 0;
            for (size_t i = 0; i < a.second.size(); ++i) { const auto d = a.second[i]-b.second[i]; delta += double(d)*d; energy += double(a.second[i])*a.second[i]; maximum = std::max(maximum,std::abs(d)); }
            t.out << "trace " << a.first << " max=" << maximum << " relative=" << std::sqrt(delta/std::max(energy,1e-30)) << '\n';
        }
        if (!t.assert_equal(baseline.logits.size(),streamed.logits.size())) return;
        float maximum = 0; size_t same_top = 0;
        for (size_t step = 0; step < baseline.logits.size(); ++step) {
            const auto & a = baseline.logits[step]; const auto & b = streamed.logits[step];
            if (!t.assert_equal(a.size(),b.size())) return;
            for (size_t i = 0; i < a.size(); ++i) { if (!std::isfinite(b[i])) { t.assert_true(false); return; } maximum = std::max(maximum,std::abs(a[i]-b[i])); }
            same_top += std::max_element(a.begin(),a.end())-a.begin() == std::max_element(b.begin(),b.end())-b.begin();
        }
        t.out << "logit max error=" << maximum << ", matching top tokens=" << same_top << "/32\n";
        t.assert_true(maximum < .05f);
        t.assert_true(baseline.recurrent.metadata == streamed.recurrent.metadata);
        if (!t.assert_equal(baseline.recurrent.tensors.size(),streamed.recurrent.tensors.size())) return;
        double delta = 0, energy = 0;
        for (size_t n = 0; n < baseline.recurrent.tensors.size(); ++n) {
            const auto & a = baseline.recurrent.tensors[n]; const auto & b = streamed.recurrent.tensors[n];
            if (!t.assert_equal(a.size(),b.size())) return;
            for (size_t i = 0; i < a.size(); ++i) { if (!std::isfinite(b[i])) { t.assert_true(false); return; } delta += double(a[i]-b[i])*(a[i]-b[i]); energy += double(a[i])*a[i]; }
        }
        t.out << "recurrent relative L2=" << std::sqrt(delta/std::max(energy,1e-30)) << '\n';
        t.assert_true(delta < std::max(energy,1e-30)*1e-4);
    });
    return t.summary();
}
