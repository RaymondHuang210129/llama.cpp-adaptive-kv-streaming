#include "../src/llama-kv-stream-logical-cache.h"
#include "../src/llama-kv-cells.h"
#include "../src/llama-batch.h"
#include "../src/llama-memory-phase.h"
#include "../src/llama-memory-executor-cuda.h"
#include "testing.h"
#include "../src/llama-kv-stream-writer.h"
#include "../src/llama-kv-stream-model.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "../ggml/src/ggml-backend-execution.h"

#include <cstring>
#include <vector>
#include <thread>
#include <stdexcept>

using operand = ggml_kv_stream_operand;

struct group_read_sync_probe {
    inline static group_read_sync_probe * active=nullptr;
    ggml_backend_t source,consumer;
    decltype(ggml_backend_i::synchronize) source_sync,consumer_sync;
    uint32_t source_calls=0,consumer_calls=0;
    bool fail_source=false;
    group_read_sync_probe(ggml_backend_t source,ggml_backend_t consumer) : source(source),consumer(consumer),
        source_sync(source->iface.synchronize),consumer_sync(consumer->iface.synchronize) {
        GGML_ASSERT(!active && source!=consumer && source_sync && consumer_sync);active=this;
        source->iface.synchronize=[](ggml_backend_t backend) {
            ++active->source_calls;active->source_sync(backend);
            if (active->fail_source) throw std::runtime_error("after source drain");
        };
        consumer->iface.synchronize=[](ggml_backend_t backend) {++active->consumer_calls;active->consumer_sync(backend);};
    }
    ~group_read_sync_probe() {
        source->iface.synchronize=source_sync;consumer->iface.synchronize=consumer_sync;active=nullptr;
    }
};

static std::shared_ptr<llama_kv_stream_logical_cache> reserved_cache(uint64_t id,uint32_t layers=1,ggml_backend_dev_t device=nullptr,uint32_t width=32,uint32_t page=8) {
    llama_kv_stream_host_config c{id, {GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, width, width, 1, page, 128},
        {{GGML_TYPE_Q8_0, true, true, true, true}, {GGML_TYPE_Q4_0, true, true, true, true}, true, true}, 33, layers};
    auto sparse = device ? llama_kv_stream_host_reservation::create(c,device,65536) :
        llama_kv_stream_host_reservation::create(c, ggml_backend_cpu_buffer_type(),
        [](ggml_backend_buffer_t owner) {
            owner->buft = ggml_backend_cpu_buffer_type();
            return ggml_backend_buffer_retain(owner);
        }, 65536);
    return llama_kv_stream_logical_cache::create(llama_kv_stream_host::from_reservation(std::move(sparse)));
}

static std::shared_ptr<llama_kv_stream_host> host(uint64_t id, ggml_type k = GGML_TYPE_Q8_0, ggml_type v = GGML_TYPE_Q4_0) {
    llama_kv_stream_host_config config;
    config.cache_id = id;
    config.shape = {k, v, 32, 32, 1, 8, 128};
    config.capabilities = {
        {k, true, true, true, true},
        {v, true, true, true, true},
        true, true,
    };
    config.context_tokens = 17;
    config.layers = 1;
    return llama_kv_stream_host::create(config, ggml_backend_cpu_buffer_type());
}

static bool append(testing & t, llama_kv_stream_logical_cache & cache, size_t count, uint8_t value) {
    const auto backing = cache.host();
    const size_t k_bytes = backing->layout().k_token_bytes*count;
    const size_t v_bytes = backing->layout().v_token_bytes*count;
    std::vector<uint8_t> k(k_bytes, value), v(v_bytes, value + 1);
    llama_kv_stream_write write;
    const size_t first = cache.tokens();
    if (!t.assert_true(cache.begin(count))) return false;
    if (!t.assert_true(cache.content()->prepare({
            {0, operand::k, first*backing->layout().k_token_bytes, k.data(), k.size()},
            {0, operand::v, first*backing->layout().v_token_bytes, v.data(), v.size()}}, write))) return false;
    if (!t.assert_true(cache.publish_host(write))) return false;
    t.assert_equal(first + count, cache.frontiers().host);
    t.assert_equal(size_t(0), cache.frontiers().device);
    t.assert_equal(first, cache.tokens());
    if (!t.assert_true(cache.finish())) return false;
    t.assert_equal(first + count, cache.tokens());
    return true;
}

struct generated_source {
    ggml_context_ptr context;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * k = nullptr;
    ggml_tensor * v = nullptr;

    generated_source(ggml_backend_t backend, size_t rows, size_t width=32) {
        context.reset(ggml_init({65536, nullptr, true}));
        GGML_ASSERT(context);
        k = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, int64_t(width), int64_t(rows));
        v = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, int64_t(width), int64_t(rows));
        buffer.reset(ggml_backend_alloc_ctx_tensors(context.get(), backend));
        GGML_ASSERT(buffer);
        std::vector<float> keys(rows*width, 1.25f), values(rows*width, 2.5f);
        ggml_backend_tensor_set(k, keys.data(), 0, keys.size()*sizeof(float));
        ggml_backend_tensor_set(v, values.data(), 0, values.size()*sizeof(float));
    }
};

static std::vector<uint8_t> native_rows(ggml_backend_t backend, bool cuda, ggml_type type,
        const std::vector<float> & values,size_t width=32) {
    ggml_context_ptr context(ggml_init({65536,nullptr,true}));
    GGML_ASSERT(context && values.size()%width==0);
    const size_t rows=values.size()/width;
    auto * input=ggml_new_tensor_2d(context.get(),GGML_TYPE_F32,int64_t(width),int64_t(rows));
    auto * indices=ggml_new_tensor_1d(context.get(),GGML_TYPE_I64,int64_t(rows));
    auto * destination=ggml_new_tensor_2d(context.get(),type,int64_t(width),int64_t(rows));
    auto * output=ggml_set_rows(context.get(),destination,input,indices);
    auto * graph=ggml_new_graph_custom(context.get(),32,false);
    ggml_build_forward_expand(graph,output);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(context.get(),backend));
    GGML_ASSERT(buffer);
    std::vector<int64_t> offsets(rows);
    for (size_t row=0;row<rows;++row) offsets[row]=int64_t(row);
    ggml_backend_tensor_set(input,values.data(),0,values.size()*sizeof(float));
    ggml_backend_tensor_set(indices,offsets.data(),0,offsets.size()*sizeof(int64_t));
    if (cuda) {
        llama_memory_cuda_executor execution(backend);
        GGML_ASSERT(execution.bind(graph,{},1));
        GGML_ASSERT(execution.compute_async({},1)==GGML_STATUS_SUCCESS && execution.drain());
    } else {
        GGML_ASSERT(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS);
    }
    std::vector<uint8_t> encoded(ggml_nbytes(output));
    ggml_backend_tensor_get(output,encoded.data(),0,encoded.size());
    return encoded;
}

int main(int argc,char ** argv) {
    const bool consumer_frame=argc==2 && (!std::strcmp(argv[1],"--cuda-group-consumer-stream") || !std::strcmp(argv[1],"--cuda-group-publication-stream"));
    const bool publication_frame=argc==2 && (!std::strcmp(argv[1],"--cuda-group-publication") || !std::strcmp(argv[1],"--cuda-group-publication-stream"));
    const bool auxiliary=argc==2 && std::strcmp(argv[1],"--cuda-auxiliary-producer")==0;
    const bool proxy_read=argc==2 && std::strcmp(argv[1],"--cuda-group-proxy-read")==0;
    const bool proxy_frame=consumer_frame || publication_frame || proxy_read || (argc==2 && std::strcmp(argv[1],"--cuda-group-proxy-frame")==0);
    const bool proxy=argc==2 && std::strcmp(argv[1],"--cuda-group-proxy")==0;
    const bool cuda=auxiliary || proxy_frame || proxy || (argc==2 && std::strcmp(argv[1],"--cuda-group-producer")==0);
    if (argc>1 && !cuda) return 1;
    ggml_backend_ptr joint_backend;
    if (cuda) {
        ggml_backend_load_all();
        auto * device=ggml_backend_dev_by_name("CUDA0");
        if (!device) return 1;
        joint_backend.reset(ggml_backend_dev_init(device,nullptr));
    } else joint_backend.reset(ggml_backend_cpu_init());
    if (!joint_backend) return 1;
    testing t;
    if (cuda) t.set_filter(auxiliary ? "auxiliary_producer_uses_explicit_catchup_and_prediction_phases" :
        proxy ? "group_proxy_reserves_native_views_but_cannot_execute_before_binding" :
        "joint_producer_routes_rows_and_commits_all_layers_under_one_source_frontier");
    t.test("logical_snapshot_relocates_history_without_reusing_source_identity", [](testing & t) {
        auto source=reserved_cache(501), destination=reserved_cache(502), peer=reserved_cache(503);
        if (!t.assert_true(source && destination && peer && append(t,*source,17,0x31) && append(t,*destination,3,0x61) && append(t,*peer,9,0x71))) return;
        llama_kv_stream_logical_snapshot saved;
        if (!t.assert_true(source->save(saved))) return;
        const auto peer_before=peer->checkpoint();
        const auto destination_before=destination->checkpoint();
        llama_kv_stream_host_layer address;
        if (!t.assert_true(destination->host()->layer(0,address))) return;
        if (!t.assert_true(destination->load(saved))) return;
        t.assert_equal(size_t(17),destination->tokens());
        t.assert_equal(destination_before.id,destination->identity().id);
        t.assert_true(destination_before.generation<destination->identity().generation);
        llama_kv_stream_host_layer restored;
        t.assert_true(destination->host()->layer(0,restored));
        t.assert_true(address.k==restored.k && address.v==restored.v);
        t.assert_equal(uint8_t(0x31),static_cast<uint8_t *>(restored.k)[0]);
        t.assert_equal(uint8_t(0x32),static_cast<uint8_t *>(restored.v)[0]);
        llama_kv_stream_logical_snapshot copy;
        t.assert_true(destination->save(copy));
        t.assert_true(saved.bytes==copy.bytes);
        t.assert_equal(peer_before.generation,peer->identity().generation);
        t.assert_equal(peer_before.tokens,peer->tokens());
        const auto unchanged=destination->checkpoint();
        auto invalid=saved;invalid.bytes.pop_back();
        t.assert_true(!destination->load(invalid));
        invalid=saved;invalid.shape.type_v=GGML_TYPE_Q8_0;
        t.assert_true(!destination->load(invalid));
        invalid=saved;invalid.tokens=SIZE_MAX;
        t.assert_true(!destination->load(invalid));
        t.assert_equal(unchanged.generation,destination->identity().generation);
        t.assert_true(destination->save(copy) && saved.bytes==copy.bytes);
        t.assert_true(source->begin(1));
        auto previous=copy.bytes;
        t.assert_true(!source->save(copy) && copy.bytes==previous);
        t.assert_true(!source->load(saved));
        t.assert_true(source->cancel());
        t.assert_true(destination->truncate(0) && destination->load(saved));
        t.assert_true(destination->save(copy) && copy.bytes==saved.bytes);
        t.assert_true(append(t,*destination,1,0x41));
        t.assert_equal(size_t(18),destination->tokens());
    });
    t.test("compact_checkpoints_pin_only_an_unchanged_authoritative_prefix", [](testing & t) {
        auto cache=llama_kv_stream_logical_cache::create(host(507));
        auto peer=llama_kv_stream_logical_cache::create(host(508));
        if (!t.assert_true(cache && peer && append(t,*cache,4,0x31))) return;
        t.assert_equal(size_t(0),peer->host()->registered_bytes());
        llama_kv_stream_logical_snapshot prefix;
        if (!t.assert_true(cache->save(prefix,false))) return;
        t.assert_true(prefix.bytes.empty());
        if (!t.assert_true(append(t,*cache,3,0x41))) return;
        t.assert_true(cache->accepts(prefix));
        t.assert_true(!peer->accepts(prefix) && !peer->load(prefix));
        t.assert_true(cache->load(prefix));
        t.assert_equal(size_t(4),cache->tokens());
        t.assert_true(cache->accepts(prefix));
        llama_kv_stream_logical_snapshot longer;
        if (!t.assert_true(append(t,*cache,3,0x51) && cache->save(longer,false))) return;
        if (!t.assert_true(cache->truncate(4))) return;
        t.assert_true(cache->accepts(prefix));
        t.assert_true(!cache->accepts(longer));
        if (!t.assert_true(append(t,*cache,3,0x61))) return;
        t.assert_true(!cache->load(longer));
        llama_kv_stream_logical_snapshot full;
        if (!t.assert_true(cache->save(full) && cache->load(full))) return;
        t.assert_true(!cache->load(prefix));
        auto ready=cache->checkpoint();
        t.assert_true(cache->save(prefix,false));
        t.assert_true(cache->restore(ready));
        t.assert_true(!cache->accepts(prefix));
    });
    t.test("cache_namespace_ranges_do_not_wrap_or_overlap_under_concurrent_creation", [](testing & t) {
        const uint64_t limit = (UINT64_MAX - 3)/4;
        std::atomic<uint64_t> clock{1};
        t.assert_equal(uint64_t(0), llama_kv_stream_cache_ids_reserve(clock, 0));
        t.assert_equal(uint64_t(1), clock.load());
        clock = limit;
        t.assert_equal(uint64_t(0), llama_kv_stream_cache_ids_reserve(clock, 2));
        t.assert_equal(limit, clock.load());
        t.assert_equal(limit, llama_kv_stream_cache_ids_reserve(clock, 1));
        t.assert_equal(uint64_t(0), llama_kv_stream_cache_ids_reserve(clock, 1));
        t.assert_equal(limit + 1, clock.load());
        clock = UINT64_MAX;
        t.assert_equal(uint64_t(0), llama_kv_stream_cache_ids_reserve(clock, 2));
        t.assert_equal(uint64_t(UINT64_MAX), clock.load());
        clock = 1;
        std::vector<uint64_t> ids(4*64*2);
        std::vector<std::thread> threads;
        for (size_t thread = 0; thread < 4; ++thread) threads.emplace_back([&, thread] {
            const uint32_t count = uint32_t(thread%2 + 1);
            for (size_t i = 0; i < 64; ++i) {
                const uint64_t first = llama_kv_stream_cache_ids_reserve(clock, count);
                for (size_t j = 0; j < count; ++j) ids[(thread*64 + i)*2 + j] = first + j;
            }
        });
        for (auto & thread : threads) thread.join();
        ids.erase(std::remove(ids.begin(), ids.end(), uint64_t(0)), ids.end());
        std::sort(ids.begin(), ids.end());
        t.assert_equal(size_t(384), ids.size());
        for (size_t i = 0; i < ids.size(); ++i) t.assert_equal(uint64_t(i + 1), ids[i]);
        t.assert_equal(uint64_t(385), clock.load());
    });
    t.test("slot_reuse_epoch_preserves_addresses_and_rejects_old_request_handles", [](testing & t) {
        llama_kv_stream_context_limits limits;
        t.assert_true(llama_kv_stream_context_limits_make(33, 40, 2, limits));
        auto table = llama_kv_stream_slot_table::create(limits);
        auto a = reserved_cache(501), b = reserved_cache(502), auxiliary = reserved_cache(503);
        llama_kv_stream_slot_handle handle_a, handle_b;
        if (!t.assert_true(table && a && b && auxiliary)) return;
        t.assert_true(table->bind(0, {a, auxiliary}, handle_a));
        t.assert_true(table->bind(1, {b, reserved_cache(504)}, handle_b));
        t.assert_true(!table->bind(0, {reserved_cache(505)}, handle_a));
        t.assert_true(append(t, *a, 17, 0x31) && append(t, *b, 9, 0x41));
        llama_kv_stream_host_layer before, after;
        t.assert_true(a->host()->layer(0, before));
        const auto checkpoint_a = a->checkpoint(), checkpoint_b = b->checkpoint();
        auto old_a = handle_a;
        t.assert_true(table->advance(handle_a, handle_a));
        t.assert_equal(old_a.generation + 1, handle_a.generation);
        std::shared_ptr<llama_kv_stream_logical_cache> resolved;
        t.assert_true(table->get(handle_a, 0, resolved) && resolved == a);
        t.assert_true(!table->get(old_a, 0, resolved) && resolved == a);
        t.assert_true(!table->get(handle_a, 2, resolved) && resolved == a);
        t.assert_true(a->host()->layer(0, after) && before.k == after.k && before.v == after.v);
        t.assert_equal(checkpoint_a.generation, a->identity().generation);
        t.assert_equal(checkpoint_a.tokens, a->tokens());
        t.assert_equal(checkpoint_b.generation, b->identity().generation);
        t.assert_true(a->begin(1));
        const auto busy = handle_a;
        t.assert_true(!table->advance(handle_a, handle_a));
        t.assert_equal(busy.generation, handle_a.generation);
        t.assert_true(a->cancel());
        t.assert_true(auxiliary->begin(1));
        t.assert_true(!table->advance(handle_a, handle_a));
        t.assert_true(auxiliary->cancel());
        t.assert_true(table->get(handle_b, 0, resolved) && resolved == b);
    });
    t.test("slot_native_cells_keep_positions_and_image_metadata_outside_group_projection", [](testing & t) {
        llama_kv_stream_context_limits limits;
        t.assert_true(llama_kv_stream_context_limits_make(33,40,2,limits));
        auto table=llama_kv_stream_slot_table::create(limits);
        auto a=reserved_cache(511),b=reserved_cache(512),aux=reserved_cache(513);
        llama_kv_stream_slot_handle ha,hb;
        if (!t.assert_true(table && table->bind(0,{a,aux},ha) && table->bind(1,{b,reserved_cache(514)},hb))) return;
        std::shared_ptr<llama_kv_cells> ca,cb,extra;
        if (!t.assert_true(table->get_cells(ha,0,ca) && table->get_cells(hb,0,cb) && table->get_cells(ha,1,extra))) return;
        t.assert_true(ca != cb && ca != extra && cb != extra);
        t.assert_equal(uint32_t(40),ca->size());
        t.assert_equal(uint32_t(40),cb->size());
        if (!t.assert_true(append(t,*a,32,0x31) && append(t,*b,33,0x51))) return;
        for (uint32_t i=0;i<33;++i) {
            cb->pos_set(i,llama_pos(1024+i));
            cb->ext_set(i,{llama_pos(i),3,i%2 ? llama_token(42) : LLAMA_TOKEN_NULL});
            cb->seq_add(i,1);
            if (i<32) {
                ca->pos_set(i,llama_pos(i+7));
                ca->ext_set(i,{2,llama_pos(i),llama_token(100+i)});
                ca->seq_add(i,0);
            }
        }
        t.assert_equal(uint32_t(32),ca->get_used());
        t.assert_equal(uint32_t(33),cb->get_used());
        t.assert_equal(uint32_t(0),extra->get_used());
        const auto old=ha;
        t.assert_true(table->advance(ha,ha));
        auto resolved=cb;
        t.assert_true(!table->get_cells(old,0,resolved) && resolved == cb);
        t.assert_true(!table->get_cells(ha,2,resolved) && resolved == cb);
        t.assert_true(table->get_cells(ha,0,resolved) && resolved == ca);
        if (!t.assert_true(a->truncate(7))) return;
        for (uint32_t i=7;i<32;++i) ca->rm(i);
        t.assert_equal(uint32_t(7),ca->get_used());
        t.assert_equal(uint32_t(33),cb->get_used());
        for (uint32_t i=0;i<33;++i) {
            t.assert_equal(llama_pos(1024+i),cb->pos_get(i));
            t.assert_true(cb->seq_has(i,1) && !cb->seq_has(i,0));
            t.assert_equal(llama_pos(i),cb->ext_get(i).x);
            t.assert_equal(llama_pos(3),cb->ext_get(i).y);
            t.assert_equal(i%2 ? llama_token(42) : LLAMA_TOKEN_NULL,cb->ext_get(i).tok);
        }
        ca->resize(8);
        resolved=cb;
        t.assert_true(!table->get_cells(ha,0,resolved) && resolved == cb);
        t.assert_true(!table->advance(ha,ha));
        auto cache=b;
        size_t logical=77;
        t.assert_true(!table->get(ha,0,cache) && cache == b);
        t.assert_true(!table->logical_used(ha,0,logical) && logical == 77);
        t.assert_true(table->get_cells(hb,0,resolved) && resolved == cb);
    });
    t.test("group_proxy_reserves_native_views_but_cannot_execute_before_binding",[&](testing & t) {
        llama_kv_stream_context_limits limits;
        if (!t.assert_true(llama_kv_stream_context_limits_make(33,256,2,limits))) return;
        std::shared_ptr<llama_kv_stream_slot_table> table(llama_kv_stream_slot_table::create(limits));
        llama_kv_stream_group_model_config config;
        config.backend=joint_backend.get();config.table=table;
        config.shape={GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,256,256,4,256,128};
        config.layers=2;config.query_heads=24;config.maximum_rows=256;config.maximum_verify_rows=1;config.maximum_selected=2;
        auto model=llama_kv_stream_group_model::create(config);
        if (!cuda) {t.assert_true(!model);return;}
        if (!t.assert_true(model!=nullptr)) return;
        auto * buffer=model->buffer();
        t.assert_true(buffer && ggml_backend_buft_is_execution(ggml_backend_buffer_get_type(buffer)));
        ggml_kv_stream_layout layout;
        if (!t.assert_true(ggml_kv_stream_layout_make(config.shape,256,layout).status==ggml_kv_stream_status::success)) return;
        t.assert_equal(layout.bytes*2,ggml_backend_buffer_get_size(buffer));
        llama_kv_stream_host_layer first,second,untouched{reinterpret_cast<void *>(1),reinterpret_cast<void *>(2)};
        t.assert_true(model->layer(0,first) && model->layer(1,second));
        t.assert_equal(layout.bytes,uintptr_t(second.k)-uintptr_t(first.k));
        t.assert_true(!model->layer(2,untouched) && untouched.k==reinterpret_cast<void *>(1));
        ggml_backend_buffer_clear(buffer,0);
        auto allocator=llama_batch_allocr(1);auto batch=allocator.ubatch_reserve(1,1);
        t.assert_true(!model->begin({},batch,{0},llama_memory_work_phase::target_verify));
        t.assert_true(!model->begin_shared_read(joint_backend.get(),batch,{0},[](uint32_t,ggml_tensor *) {return true;},
            [] {return true;}) && !model->shared_read_active());
        t.assert_true(!model->finish() && model->abort());
        t.assert_true(!model->bind({},[](uint32_t,ggml_tensor *) {return true;},[] {return true;}));
        auto split=config;split.maximum_rows=split.maximum_verify_rows=4;
        t.assert_true(llama_kv_stream_group_model::create(split)!=nullptr);
        auto invalid=split;invalid.maximum_rows=3;
        t.assert_true(!llama_kv_stream_group_model::create(invalid));
        invalid=config;invalid.maximum_verify_rows=9;
        t.assert_true(!llama_kv_stream_group_model::create(invalid));
        ggml_backend_buffer_ptr retained(ggml_backend_buffer_retain(buffer));
        model.reset();
        ggml_backend_buffer_clear(retained.get(),0);
        ggml_tensor metadata{};metadata.op=GGML_OP_NONE;
        t.assert_true(ggml_backend_execution_compute(retained.get(),joint_backend.get(),&metadata)==GGML_STATUS_SUCCESS);
    });
    t.test("joint_producer_routes_rows_and_commits_all_layers_under_one_source_frontier", [&](testing & t) {
        llama_kv_stream_context_limits limits;
        if (!t.assert_true(llama_kv_stream_context_limits_make(33,proxy_frame ? 256 : 40,2,limits))) return;
        std::shared_ptr<llama_kv_stream_slot_table> table(llama_kv_stream_slot_table::create(limits));
        auto * device=cuda ? ggml_backend_get_device(joint_backend.get()) : nullptr;
        const size_t width=proxy_frame ? 256 : 32;
        auto a=reserved_cache(721,2,device,uint32_t(width),proxy_frame ? 256 : 8),
             b=reserved_cache(722,2,device,uint32_t(width),proxy_frame ? 256 : 8);
        llama_kv_stream_slot_handle ha,hb;
        if (!t.assert_true(table && table->bind(0,{a},ha) && table->bind(1,{b},hb))) return;
        auto projection=std::make_shared<llama_kv_stream_slot_projection>();
        llama_kv_stream_selection selected;
        if (!t.assert_true(llama_kv_stream_select({33,2},{{0,1},{1,1}},selected)==llama_kv_stream_selection_status::success &&
                table->project(selected,{ha,hb},0,*projection))) return;
        auto backend=joint_backend.get();
        auto * buffer_type=cuda ? llama_kv_stream_device_buffer_type(device) : ggml_backend_cpu_buffer_type();
        size_t scratch_bytes=4096;
        if (proxy_frame && !t.assert_true(llama_kv_stream_writer::requirements(buffer_type,a->host()->config().shape,9,scratch_bytes))) return;
        ggml_backend_buffer_ptr scratch(ggml_backend_buft_alloc_buffer(buffer_type,scratch_bytes));
        std::shared_ptr<llama_kv_stream_writer> writer(llama_kv_stream_writer::create(backend,scratch.get(),scratch_bytes,a->host()->config().shape,9));
        std::shared_ptr<llama_kv_stream_group_producer> producer(llama_kv_stream_group_producer::create(table,writer,9,2,2));
        std::unique_ptr<llama_kv_stream_group_model> model;
        bool admitted=false;uint32_t attention_calls=0;
        if (proxy_frame) {
            llama_kv_stream_group_model_config config{backend,table,a->host()->config().shape,2,4,9,2,2};
            model=llama_kv_stream_group_model::create(config);
            if (!t.assert_true(model && model->bind(producer,[&](uint32_t layer,ggml_tensor *) {
                    ++attention_calls;return producer->layer_ready(layer);},[&] {return admitted;}))) return;
        }
        if (!t.assert_true(producer!=nullptr)) return;
        if (consumer_frame) {
            ggml_backend_ptr other(ggml_backend_dev_init(ggml_backend_get_device(backend),nullptr));
            if (!t.assert_true(other && !producer->bind_consumer(other.get()) && producer->bind_consumer(backend))) return;
        }
        size_t publication_copies=0,publication_ready=0,publication_committed=0;
        std::vector<std::vector<uint8_t>> published[2];published[0].resize(2);published[1].resize(2);
        if (publication_frame && !t.assert_true(producer->bind_publication(
                [&](uint32_t,const auto & rows) {return rows.size()==3;},
                [&](uint32_t layer,bool value,const ggml_tensor * encoded,size_t first,size_t count,const auto & rows) {
                    if (!t.assert_true(layer<2 && rows.size()==3 && rows[0].source==0 && rows[1].source==1 && rows[2].source==0 &&
                            rows[0].token==0 && rows[1].token==0 && rows[2].token==1)) return false;
                    const size_t stride=ggml_row_size(encoded->type,int64_t(width));
                    published[value][layer].resize(3*stride);
                    ggml_backend_synchronize(backend);
                    ggml_backend_tensor_get(encoded,published[value][layer].data()+first*stride,0,count*stride);
                    ++publication_copies;return true;
                },[&](uint32_t,const auto & rows,bool committed) {
                    t.out<<"publication completion committed="<<committed<<" tokens="<<a->tokens()<<','<<b->tokens()
                        <<" ready="<<a->ready()<<','<<b->ready()<<" projection="<<table->projection_current(*projection)<<std::endl;
                    if (!t.assert_true(rows.size()==3)) return false;
                    if (committed) {
                        ++publication_committed;
                        return t.assert_true(a->ready() && b->ready() && a->tokens()==2 && b->tokens()==1 && table->projection_current(*projection));
                    }
                    ++publication_ready;
                    return t.assert_true(!a->ready() && !b->ready() && !a->tokens() && !b->tokens());
                }))) return;
        auto allocator=llama_batch_allocr(1);auto batch=allocator.ubatch_reserve(3,1);
        llama_seq_id ids[3]={0,1,0};
        for (size_t row=0;row<3;++row) {
            batch.n_seq_id[row]=1;batch.seq_id[row]=&ids[row];batch.pos[row]=row==2 ? 1 : 0;batch.token[row]=llama_token(100+row);
        }
        const auto begin=[&](const auto & input,const std::vector<uint32_t> & rows) {
            return model ? model->begin(projection,input,rows,llama_memory_work_phase::target_verify) :
                producer->begin(projection,input,rows,llama_memory_work_phase::target_verify);
        };
        const auto finish=[&] {return model ? model->finish() : producer->finish();};
        const auto abort=[&] {return model ? model->abort() : producer->abort();};
        if (model) {
            t.assert_true(!begin(batch,{0,1,2}) && a->ready() && b->ready());admitted=true;
        }
        if (!t.assert_true(begin(batch,{0,1,2}))) return;
        if (publication_frame) t.assert_true("active publication bindings cannot be replaced",!producer->bind_publication({}, {}, {}));
        if (consumer_frame) t.assert_true("an active consumer dependency cannot be replaced",!producer->bind_consumer(nullptr));
        t.assert_true(!a->ready() && !b->ready());
        generated_source source(backend,3,width);
        std::vector<float> keys(3*width),values(3*width);
        for (size_t row=0;row<3;++row) for (size_t col=0;col<width;++col) {
            keys[row*width+col]=float(row*8+col)*0.125f;values[row*width+col]=-float(row*16+col)*0.0625f;
        }
        std::vector<std::vector<uint8_t>> expected_keys(2),expected_values(2);

        for (uint32_t row=0;row<3;++row) {
            projection->cells->pos_set(row,batch.pos[row]);projection->cells->seq_add(row,ids[row]);
            projection->cells->ext_set(row,{0,0,batch.token[row]});
        }
        ggml_context_ptr coordinate_metadata(ggml_init({4096,nullptr,true}));
        auto * coordinate=ggml_new_tensor_1d(coordinate_metadata.get(),GGML_TYPE_I64,3);
        ggml_backend_buffer_ptr coordinate_buffer(ggml_backend_alloc_ctx_tensors(coordinate_metadata.get(),backend));
        const auto publish=[&](uint32_t layer,const generated_source & input,const std::vector<int64_t> & rows) {
            if (!model) return producer->produce(layer,input.k,input.v);
            ggml_backend_tensor_set(coordinate,rows.data(),0,rows.size()*sizeof(int64_t));
            auto index=*coordinate;index.ne[0]=int64_t(rows.size());
            index.nb[1]=index.nb[2]=index.nb[3]=rows.size()*sizeof(int64_t);
            llama_kv_stream_host_layer planes;
            if (!model->layer(layer,planes)) return false;
            ggml_tensor key{},value{},op{};
            key.type=GGML_TYPE_Q8_0;value.type=GGML_TYPE_Q4_0;
            key.data=planes.k;value.data=planes.v;key.buffer=value.buffer=model->buffer();
            op.op=GGML_OP_SET_ROWS;op.src[0]=input.k;op.src[1]=&index;op.src[2]=&key;
            if (ggml_backend_execution_compute(model->buffer(),backend,&op)!=GGML_STATUS_SUCCESS) return false;
            op.src[0]=input.v;op.src[2]=&value;op.src[3]=input.k;
            return ggml_backend_execution_compute(model->buffer(),backend,&op)==GGML_STATUS_SUCCESS;
        };
        if (model && !t.assert_true(model->apply_cells(*projection->cells))) return;
        for (uint32_t layer=0;layer<2;++layer) {
            for (auto & value : keys) value+=0.5f;
            for (auto & value : values) value-=0.25f;
            if (consumer_frame) {
                ggml_backend_tensor_set_async(backend,source.k,keys.data(),0,keys.size()*sizeof(float));
                ggml_backend_tensor_set_async(backend,source.v,values.data(),0,values.size()*sizeof(float));
            } else {
                ggml_backend_tensor_set(source.k,keys.data(),0,keys.size()*sizeof(float));
                ggml_backend_tensor_set(source.v,values.data(),0,values.size()*sizeof(float));
            }
            expected_keys[layer]=native_rows(backend,cuda,GGML_TYPE_Q8_0,keys,width);
            expected_values[layer]=native_rows(backend,cuda,GGML_TYPE_Q4_0,values,width);
#ifdef KV_STREAM_CONSUMER_TRACE
            if (consumer_frame) kv_stream_consumer_capture(true);
#endif
            const bool produced=publish(layer,source,{0,1,2}) && producer->layer_ready(layer);
#ifdef KV_STREAM_CONSUMER_TRACE
            if (consumer_frame) {
                kv_stream_consumer_capture(false);
                t.assert_true("an ordered GPU consumer does not wait on the CPU at publication",kv_stream_consumer_syncs()==0);
            }
#else
            if (consumer_frame) t.assert_true("native completion tracing is required",false);
#endif
            if (!t.assert_true(produced)) return;
            if (model) {
                llama_kv_stream_host_layer planes;if (!t.assert_true(model->layer(layer,planes))) return;
                ggml_tensor query{},key{},value{},mask{},op{};
                query.type=op.type=GGML_TYPE_F32;key.type=GGML_TYPE_Q8_0;value.type=GGML_TYPE_Q4_0;
                query.ne[0]=int64_t(width);query.ne[1]=3;query.ne[2]=4;query.ne[3]=1;
                key.ne[0]=value.ne[0]=int64_t(width);key.ne[1]=value.ne[1]=256;
                key.ne[2]=value.ne[2]=key.ne[3]=value.ne[3]=1;
                key.data=planes.k;value.data=planes.v;key.buffer=value.buffer=model->buffer();
                op.op=GGML_OP_FLASH_ATTN_EXT;op.src[0]=&query;op.src[1]=&key;op.src[2]=&value;op.src[3]=&mask;
                const float params[3]={0.125f,0,0};std::memcpy(op.op_params,params,sizeof(params));
                if (!t.assert_true(ggml_backend_execution_compute(model->buffer(),backend,&op)==GGML_STATUS_SUCCESS)) return;
            }
            if (!layer) {
                t.assert_true(!finish() && !a->ready() && !b->ready());
                t.assert_equal(size_t(0),a->tokens());t.assert_equal(size_t(0),b->tokens());
            }
        }
        if (!t.assert_true(finish())) return;
        t.assert_equal(size_t(2),a->tokens());t.assert_equal(size_t(1),b->tokens());
        t.assert_true(a->ready() && b->ready() && table->projection_current(*projection));
        for (uint32_t layer_index=0;layer_index<2;++layer_index) for (bool value : {false,true}) {
            const auto type=value ? GGML_TYPE_Q4_0 : GGML_TYPE_Q8_0;
            const size_t stride=ggml_row_size(type,int64_t(width));
            const auto & encoded=value ? expected_values[layer_index] : expected_keys[layer_index];
            for (uint32_t row=0;row<3;++row) {
                const auto cache=row==1 ? b : a;
                const size_t token=row==2 ? 1 : 0;
                llama_kv_stream_host_layer layer;
                if (!t.assert_true(cache->host()->layer(layer_index,layer))) return;
                const auto * address=static_cast<const uint8_t *>(value ? layer.v : layer.k)+token*stride;
                t.assert_true("canonical producer bytes match the native row encoder",!std::memcmp(encoded.data()+row*stride,address,stride));
                t.assert_equal(uint32_t(row==1 ? 1 : 0),projection->columns[row].source);
                t.assert_equal(token,projection->columns[row].token);
            }
        }
        if (publication_frame) {
            const bool copies=t.assert_equal("every encoded plane is published on device",size_t(4),publication_copies);
            const bool ready=t.assert_equal("device readiness precedes canonical commit",size_t(2),publication_ready);
            const bool committed=t.assert_equal("host completion closes every published layer",size_t(2),publication_committed);
            if (!copies || !ready || !committed) return;
            for (size_t layer=0;layer<2;++layer) {
                t.assert_true("published key bytes match native encoding",published[0][layer]==expected_keys[layer]);
                t.assert_true("published value bytes match native encoding",published[1][layer]==expected_values[layer]);
            }
            if (!t.assert_true(producer->bind_publication({}, {}, {}))) return;
        }
        if (consumer_frame && !t.assert_true(producer->bind_consumer(nullptr))) return;
        const auto before_a=a->checkpoint(),before_b=b->checkpoint();
        if (model) {
            admitted=false;
            ggml_context_ptr metadata(ggml_init({8192,nullptr,true}));
            for (uint32_t layer_index=0;layer_index<2;++layer_index) for (const bool value : {false,true}) {
                llama_kv_stream_host_layer planes;
                if (!t.assert_true(model->layer(layer_index,planes))) return;
                const auto type=value ? GGML_TYPE_Q4_0 : GGML_TYPE_Q8_0;
                const auto & expected=value ? expected_values[layer_index] : expected_keys[layer_index];
                const auto stride=ggml_row_size(type,int64_t(width));
                auto * tensor=ggml_new_tensor_2d(metadata.get(),type,int64_t(width),limits.slot_storage);
                tensor->buffer=model->buffer();tensor->data=value ? planes.v : planes.k;
                std::vector<uint8_t> exported(expected.size()+2,0x5a);
                ggml_backend_tensor_get(tensor,exported.data()+1,0,expected.size());
                t.assert_true(exported.front()==0x5a && exported.back()==0x5a &&
                    !std::memcmp(exported.data()+1,expected.data(),expected.size()));
                std::fill(exported.begin(),exported.end(),0x5a);
                const auto offset=stride-3,count=stride+7;
                ggml_backend_tensor_get(tensor,exported.data()+1,offset,count);
                t.assert_true(exported.front()==0x5a && exported[count+1]==0x5a &&
                    !std::memcmp(exported.data()+1,expected.data()+offset,count));
            }
            t.assert_true(a->identity().generation==before_a.generation && b->identity().generation==before_b.generation &&
                table->projection_current(*projection));
            admitted=true;
        }
        if (proxy_read) {
            auto borrowed=allocator.ubatch_reserve(1,2);llama_seq_id read_ids[2]={0,1};
            for (size_t row=0;row<2;++row) {
                borrowed.n_seq_id[row]=1;borrowed.seq_id[row]=&read_ids[row];borrowed.pos[row]=row ? 1 : 2;
                borrowed.token[row]=llama_token(200+row);borrowed.output[row]=true;
            }
            bool reader_admitted=false;uint32_t reads=0;
            const auto allowed=[&] {return reader_admitted;};
            // This callback tests frame ownership only; native attention math has separate controls.
            const auto attention=[&](uint32_t layer,ggml_tensor *) {
                ++reads;return layer==1 && a->ready() && b->ready();
            };
            t.assert_true(!model->begin_shared_read(backend,borrowed,{1,1},attention,allowed));
            reader_admitted=true;
            t.assert_true(!model->begin_shared_read(backend,borrowed,{},attention,allowed));
            t.assert_true(!model->begin_shared_read(backend,borrowed,{2},attention,allowed));
            read_ids[1]=0;t.assert_true(!model->begin_shared_read(backend,borrowed,{1,1},attention,allowed));read_ids[1]=1;
            borrowed.pos[0]=3;t.assert_true(!model->begin_shared_read(backend,borrowed,{1,1},attention,allowed));borrowed.pos[0]=2;
            if (!t.assert_true(model->begin_shared_read(backend,borrowed,{1,1},attention,allowed))) return;
            t.assert_true(model->shared_read_active() && !model->finish_shared_read() && !begin(batch,{0,1,2}));
            llama_kv_stream_host_layer planes;if (!t.assert_true(model->layer(1,planes))) return;
            ggml_tensor query{},key{},value{},mask{},op{};
            query.type=op.type=GGML_TYPE_F32;key.type=GGML_TYPE_Q8_0;value.type=GGML_TYPE_Q4_0;
            query.ne[0]=int64_t(width);query.ne[1]=2;query.ne[2]=4;query.ne[3]=1;
            key.ne[0]=value.ne[0]=int64_t(width);key.ne[1]=value.ne[1]=256;
            key.ne[2]=value.ne[2]=key.ne[3]=value.ne[3]=1;
            key.data=planes.k;value.data=planes.v;key.buffer=value.buffer=model->buffer();
            op.op=GGML_OP_FLASH_ATTN_EXT;op.src[0]=&query;op.src[1]=&key;op.src[2]=&value;op.src[3]=&mask;
            const float params[3]={0.125f,0,0};std::memcpy(op.op_params,params,sizeof(params));
            for (uint32_t read=0;read<2;++read) {
                if (!t.assert_true(ggml_backend_execution_compute(model->buffer(),backend,&op)==GGML_STATUS_SUCCESS)) return;
                if (!read) t.assert_true(!model->finish_shared_read());
            }
            t.assert_true(model->finish_shared_read() && !model->shared_read_active());
            t.assert_equal(uint32_t(2),reads);
            t.assert_true(a->checkpoint().id==before_a.id && a->identity().generation==before_a.generation && a->tokens()==before_a.tokens &&
                b->checkpoint().id==before_b.id && b->identity().generation==before_b.generation && b->tokens()==before_b.tokens);
            if (!t.assert_true(model->begin_shared_read(backend,borrowed,{0},attention,allowed))) return;
            t.assert_true(ggml_backend_execution_compute(model->buffer(),backend,&op)==GGML_STATUS_FAILED);
            t.assert_true(!model->finish_shared_read() && model->shared_read_active() && model->abort() && !model->shared_read_active());
            t.assert_true(a->ready() && b->ready() && table->projection_current(*projection));
            if (!t.assert_true(model->begin_shared_read(backend,borrowed,{1},attention,allowed))) return;
            t.assert_true(!publish(0,source,{0,1,2}) && model->shared_read_active() && !model->finish_shared_read());
            t.assert_true(model->abort() && a->ready() && b->ready() && table->projection_current(*projection));
            t.assert_true(!model->begin_shared_read(backend,borrowed,{1},attention,[]()->bool {
                throw std::runtime_error("read admission fault");
            }) && !model->shared_read_active());
            if (!t.assert_true(model->begin_shared_read(backend,borrowed,{1},attention,allowed))) return;
            if (!t.assert_true(ggml_backend_execution_compute(model->buffer(),backend,&op)==GGML_STATUS_SUCCESS)) return;
            reader_admitted=false;
            t.assert_true(!model->finish_shared_read() && model->shared_read_active() && model->abort());
            reader_admitted=true;
            ggml_backend_ptr consumer(ggml_backend_dev_init(device,nullptr));
            if (!t.assert_true(consumer && consumer.get()!=backend)) return;
            {
                group_read_sync_probe sync(backend,consumer.get());
                struct read_guard {
                    llama_kv_stream_group_model * model;
                    ~read_guard() {if (model->shared_read_active()) model->abort();}
                } guard{model.get()};
                bool consumed=false;
                const auto bridged=[&](uint32_t layer,ggml_tensor *) {
                    consumed=layer==1 && sync.consumer_calls>0 && sync.source_calls==0;
                    return consumed;
                };
                if (!t.assert_true(model->begin_shared_read(consumer.get(),borrowed,{1},bridged,allowed))) return;
                if (!t.assert_true(ggml_backend_execution_compute(model->buffer(),consumer.get(),&op)==GGML_STATUS_SUCCESS)) return;
                t.assert_true(consumed && sync.source_calls>0 && model->finish_shared_read());
            }
            t.assert_true(!model->shared_read_active() && a->ready() && b->ready() &&
                a->identity().generation==before_a.generation && b->identity().generation==before_b.generation);
            {
                auto fault=llama_kv_stream_group_model::create({backend,table,a->host()->config().shape,2,4,9,2,2});
                if (!t.assert_true(fault && fault->bind(producer,attention,[&] {return admitted;}) && fault->bind_projection(projection))) return;
                llama_kv_stream_host_layer fault_planes;if (!t.assert_true(fault->layer(1,fault_planes))) return;
                key.data=fault_planes.k;value.data=fault_planes.v;key.buffer=value.buffer=fault->buffer();
                group_read_sync_probe sync(backend,consumer.get());
                struct cleanup_guard {
                    llama_kv_stream_group_model * model;group_read_sync_probe & sync;
                    ~cleanup_guard() {sync.fail_source=false;if (model->shared_read_active()) model->abort();}
                } guard{fault.get(),sync};
                uint32_t checks=0;
                if (!t.assert_true(fault->begin_shared_read(consumer.get(),borrowed,{1},attention,[&] {++checks;return true;}) &&
                        ggml_backend_execution_compute(fault->buffer(),consumer.get(),&op)==GGML_STATUS_SUCCESS)) return;
                sync.fail_source=true;
                t.assert_true(!fault->abort() && fault->shared_read_active());
                const auto before=checks;
                t.assert_true(ggml_backend_execution_compute(fault->buffer(),consumer.get(),&op)==GGML_STATUS_FAILED);
                t.assert_equal("failed source cleanup closes admission before callback access",before,checks);
                t.assert_true(!fault->finish_shared_read() && fault->shared_read_active());
                sync.fail_source=false;
                t.assert_true(fault->abort() && !fault->shared_read_active() && !fault->bound());
                t.assert_true(a->ready() && b->ready() && table->projection_current(*projection));
            }
        }
        ids[0]=2;
        t.assert_true(!begin(batch,{3,4,5}));
        t.assert_equal(before_a.generation,a->identity().generation);t.assert_equal(before_b.generation,b->identity().generation);
        auto one=allocator.ubatch_reserve(1,1);llama_seq_id id=0;
        one.pos[0]=2;one.token[0]=200;one.n_seq_id[0]=1;one.seq_id[0]=&id;
        if (!t.assert_true(begin(one,{3}))) return;
        projection->cells->pos_set(3,2);projection->cells->seq_add(3,0);
        generated_source extra(backend,1,width);
        if (model && !t.assert_true(model->apply_cells(*projection->cells))) return;
        if (!t.assert_true(publish(0,extra,{3}) && !finish())) return;
        t.assert_true(abort());
        t.assert_equal(size_t(2),a->tokens());t.assert_equal(size_t(1),b->tokens());
        t.assert_true(a->ready() && b->ready());
        if (!t.assert_true(llama_kv_stream_select({33,2},{{0,2},{1,1}},selected)==llama_kv_stream_selection_status::success &&
                table->project(selected,{ha,hb},0,*projection))) return;
        if (!t.assert_true(begin(one,{3}))) return;
        projection->cells->pos_set(3,2);projection->cells->seq_add(3,0);
        if (model && !t.assert_true(model->apply_cells(*projection->cells))) return;
        if (!t.assert_true(a->content()->invalidate())) return;
        if (!t.assert_true("stale source generation rejects before joint producer submission",!publish(0,extra,{3}))) {
            abort();return;
        }
        t.assert_true("unknown external content mutation cannot reopen cache admission",!abort());
        t.assert_equal(size_t(2),a->tokens());t.assert_equal(size_t(1),b->tokens());
        t.assert_true(!a->ready() && b->ready());
        t.assert_true(!begin(one,{3}));
        if (model) t.assert_equal(uint32_t(2),attention_calls);
    });
    t.test("selected_native_projection_keeps_canonical_rows_positions_and_logical_allowances", [](testing & t) {
        llama_kv_stream_context_limits limits;
        if (!t.assert_true(llama_kv_stream_context_limits_make(33,40,2,limits))) return;
        auto table=llama_kv_stream_slot_table::create(limits);
        auto a=reserved_cache(601),b=reserved_cache(602);
        llama_kv_stream_slot_handle ha,hb;
        if (!t.assert_true(table && table->bind(0,{a},ha) && table->bind(1,{b},hb))) return;
        if (!t.assert_true(append(t,*a,7,0x31) && append(t,*b,13,0x51))) return;
        std::shared_ptr<llama_kv_cells> ca,cb;
        if (!t.assert_true(table->get_cells(ha,0,ca) && table->get_cells(hb,0,cb))) return;
        for (uint32_t row=0;row<13;++row) {
            cb->pos_set(row,llama_pos(91+row));cb->seq_add(row,1);cb->ext_set(row,{llama_pos(row),3,LLAMA_TOKEN_NULL});
            if (row<7) {ca->pos_set(row,llama_pos(17+row));ca->seq_add(row,0);ca->ext_set(row,{2,llama_pos(row),llama_token(100+row)});}
        }
        llama_kv_stream_selection selected;
        if (!t.assert_true(llama_kv_stream_select({33,2},{{0,8},{1,14}},selected)==llama_kv_stream_selection_status::success)) return;
        llama_kv_stream_slot_projection projection;
        if (!t.assert_true(table->project(selected,{ha,hb},0,projection))) return;
        if (!t.assert_true(projection.cells && projection.columns.size()==40)) return;
        t.assert_true(table->projection_current(projection));
        t.assert_equal(uint32_t(20),projection.cells->get_used());
        if (!t.assert_true("logical growth allowances do not increase native attention key extent",projection.cells->used_max_p1()==20)) return;
        t.assert_equal(size_t(14),selected.slots[1].offset_tokens);
        for (size_t col=0;col<40;++col) {
            const bool first=col<7,second=col>=7 && col<20;
            if (!first && !second) {
                t.assert_true(projection.cells->is_empty(uint32_t(col)) && projection.columns[col].source==UINT32_MAX);
                continue;
            }
            const uint32_t row=uint32_t(first ? col : col-7),slot=first ? 0 : 1;
            const auto & source=first ? *ca : *cb;
            t.assert_equal(slot,projection.columns[col].source);
            t.assert_equal(size_t(row),projection.columns[col].token);
            t.assert_equal(source.pos_get(row),projection.cells->pos_get(uint32_t(col)));
            t.assert_true(projection.cells->seq_get_all(uint32_t(col))==source.seq_get_all(row));
            const auto & want=source.ext_get(row);const auto & got=projection.cells->ext_get(uint32_t(col));
            t.assert_true(want.x==got.x && want.y==got.y && want.tok==got.tok);
        }
        auto before=projection.cells;
        auto bad=selected;bad.slots[1].offset_tokens=13;
        t.assert_true(!table->project(bad,{ha,hb},0,projection) && projection.cells==before);
        t.assert_true(!table->project(selected,{hb,ha},0,projection) && projection.cells==before);
        t.assert_true(!table->project(selected,{ha,hb},1,projection) && projection.cells==before);
        t.assert_true(table->advance(ha,ha));
        t.assert_true(!table->projection_current(projection));
        t.assert_true(table->project(selected,{ha,hb},0,projection));
        if (!t.assert_true(b->begin(1))) return;
        t.assert_true(!table->projection_current(projection));
        t.assert_true(!table->project(selected,{ha,hb},0,projection));
        t.assert_true(b->cancel());
        t.assert_true(!table->projection_current(projection));
        t.assert_true(table->project(selected,{ha,hb},0,projection));
        t.assert_true(ca->get_used()==7 && cb->get_used()==13);
    });
    t.test("owned_auxiliary_projection_uses_its_frontier_inside_target_windows", [](testing & t) {
        llama_kv_stream_context_limits limits;
        if (!t.assert_true(llama_kv_stream_context_limits_make(33,40,2,limits))) return;
        auto table=llama_kv_stream_slot_table::create(limits);
        auto a=reserved_cache(2431),aa=reserved_cache(2432),b=reserved_cache(2433),ba=reserved_cache(2434);
        llama_kv_stream_slot_handle ha,hb;
        if (!t.assert_true(table && a && aa && b && ba && table->bind(0,{a,aa},ha) && table->bind(1,{b,ba},hb))) return;
        if (!append(t,*a,7,11) || !append(t,*b,13,21) || !append(t,*aa,2,31) || !append(t,*ba,9,41)) return;
        const auto cells=[&](llama_kv_stream_slot_handle handle,uint32_t role,size_t tokens) {
            std::shared_ptr<llama_kv_cells> value;
            if (!table->get_cells(handle,role,value)) return false;
            for (uint32_t row=value->get_used();row<tokens;++row) {value->pos_set(row,llama_pos(row));value->seq_add(row,llama_seq_id(handle.slot));}
            return true;
        };
        if (!t.assert_true(cells(ha,0,7) && cells(hb,0,13) && cells(ha,1,2) && cells(hb,1,9))) return;
        llama_kv_stream_selection selected;
        if (!t.assert_true(llama_kv_stream_select({33,2},{{0,7},{1,13}},selected)==llama_kv_stream_selection_status::success)) return;
        llama_kv_stream_slot_projection target,auxiliary;
        if (!t.assert_true(table->project(selected,{ha,hb},0,target))) return;
        if (!t.assert_true("owned auxiliary cache may lag more than one target token",table->project(selected,{ha,hb},1,auxiliary))) return;
        t.assert_equal(size_t(2),auxiliary.sources[0].tokens);t.assert_equal(size_t(9),auxiliary.sources[1].tokens);
        t.assert_equal(aa->identity().id,auxiliary.sources[0].id);t.assert_equal(ba->identity().id,auxiliary.sources[1].id);
        t.assert_equal(uint32_t(11),auxiliary.cells->used_max_p1());
        t.assert_equal(uint32_t(1),auxiliary.columns[2].source);t.assert_equal(size_t(0),auxiliary.columns[2].token);
        if (!append(t,*aa,7,51) || !t.assert_true(cells(ha,1,9))) return;
        t.assert_true(table->projection_current(target) && !table->projection_current(auxiliary));
        if (!t.assert_true("draft history may advance inside the target's unchanged logical allowance",table->project(selected,{ha,hb},1,auxiliary))) return;
        t.assert_equal(uint32_t(18),auxiliary.cells->used_max_p1());
        const auto before=auxiliary.cells;
        const auto window=selected.slots[0].history_tokens+selected.slots[0].growth_tokens;
        if (!append(t,*aa,window+1-aa->tokens(),61) || !t.assert_true(cells(ha,1,window+1))) return;
        t.assert_true(!table->project(selected,{ha,hb},1,auxiliary) && auxiliary.cells==before);
        t.assert_true(table->projection_current(target));
        if (!append(t,*a,2,71) || !t.assert_true(cells(ha,0,9))) return;
        const auto original=target.cells;
        t.assert_true(!table->project(selected,{ha,hb},0,target) && target.cells==original);
    });
    t.test("auxiliary_producer_uses_explicit_catchup_and_prediction_phases", [&](testing & t) {
        llama_kv_stream_context_limits limits;
        if (!t.assert_true(llama_kv_stream_context_limits_make(33,40,2,limits))) return;
        std::shared_ptr<llama_kv_stream_slot_table> table(llama_kv_stream_slot_table::create(limits));
        auto * device=cuda ? ggml_backend_get_device(joint_backend.get()) : nullptr;
        auto a=reserved_cache(2441,1,device),aa=reserved_cache(2442,1,device),
             b=reserved_cache(2443,1,device),ba=reserved_cache(2444,1,device);
        llama_kv_stream_slot_handle ha,hb;
        if (!t.assert_true(table && a && aa && b && ba && table->bind(0,{a,aa},ha) && table->bind(1,{b,ba},hb))) return;
        if (!append(t,*a,3,11) || !append(t,*b,4,21)) return;
        for (auto handle : {ha,hb}) {
            std::shared_ptr<llama_kv_cells> cells;
            if (!t.assert_true(table->get_cells(handle,0,cells))) return;
            for (uint32_t row=0;row<(handle.slot ? 4u : 3u);++row) {cells->pos_set(row,llama_pos(row));cells->seq_add(row,llama_seq_id(handle.slot));}
        }
        const auto before_a=a->checkpoint(),before_b=b->checkpoint();
        llama_kv_stream_selection selected;
        auto projection=std::make_shared<llama_kv_stream_slot_projection>();
        if (!t.assert_true(llama_kv_stream_select({33,2},{{0,3},{1,4}},selected)==llama_kv_stream_selection_status::success &&
                table->project(selected,{ha,hb},1,*projection))) return;
        auto * backend=joint_backend.get();
        const auto shape=aa->host()->config().shape;
        auto * type=cuda ? llama_kv_stream_device_buffer_type(ggml_backend_get_device(backend)) : ggml_backend_cpu_buffer_type();
        size_t bytes=0;
        if (!t.assert_true(llama_kv_stream_writer::requirements(type,shape,4,bytes))) return;
        ggml_backend_buffer_ptr scratch(ggml_backend_buft_alloc_buffer(type,bytes));
        std::shared_ptr<llama_kv_stream_writer> writer(llama_kv_stream_writer::create(backend,scratch.get(),bytes,shape,4));
        auto producer=llama_kv_stream_group_producer::create(table,writer,4,1,2);
        if (!t.assert_true(producer!=nullptr)) return;
        auto allocator=llama_batch_allocr(1);auto batch=allocator.ubatch_reserve(4,1);
        llama_seq_id ids[4]={0,0,0,1};
        for (uint32_t row=0;row<4;++row) {
            batch.n_seq_id[row]=1;batch.seq_id[row]=&ids[row];batch.pos[row]=row==3 ? 0 : llama_pos(row);
            batch.token[row]=llama_token(10+row);batch.output[row]=false;
        }
        const std::vector<uint32_t> columns{0,1,2,3};
        t.assert_true(!producer->begin(projection,batch,columns,llama_memory_work_phase::target_prefill));
        t.assert_true(!producer->begin(projection,batch,columns,llama_memory_work_phase::mtp_draft));
        if (!t.assert_true("auxiliary catchup may exceed prediction width without changing target histories",
                producer->begin(projection,batch,columns,llama_memory_work_phase::mtp_catchup))) return;
        for (uint32_t row=0;row<4;++row) {
            projection->cells->pos_set(row,batch.pos[row]);projection->cells->seq_add(row,ids[row]);
            projection->cells->ext_set(row,{0,0,batch.token[row]});
        }
        generated_source source(backend,4,32);
        std::vector<float> keys(128),values(128);
        for (size_t i=0;i<keys.size();++i) {keys[i]=float(i)*0.125f;values[i]=-float(i)*0.0625f;}
        ggml_backend_tensor_set(source.k,keys.data(),0,keys.size()*sizeof(float));
        ggml_backend_tensor_set(source.v,values.data(),0,values.size()*sizeof(float));
        const auto encoded_k=native_rows(backend,cuda,GGML_TYPE_Q8_0,keys,32),encoded_v=native_rows(backend,cuda,GGML_TYPE_Q4_0,values,32);
        if (!t.assert_true(producer->produce(0,source.k,source.v) && producer->finish())) return;
        t.assert_equal(size_t(3),aa->tokens());t.assert_equal(size_t(1),ba->tokens());
        for (uint32_t row=0;row<4;++row) for (bool value : {false,true}) {
            const auto cache=row==3 ? ba : aa;const size_t token=row==3 ? 0 : row;
            llama_kv_stream_host_layer layer;if (!t.assert_true(cache->host()->layer(0,layer))) return;
            const auto stride=ggml_row_size(value ? GGML_TYPE_Q4_0 : GGML_TYPE_Q8_0,32);
            const auto * payload=static_cast<const uint8_t *>(value ? layer.v : layer.k)+token*stride;
            t.assert_true(!std::memcmp(payload,(value ? encoded_v : encoded_k).data()+row*stride,stride));
        }
        t.assert_true(a->ready() && b->ready() && a->tokens()==before_a.tokens && b->tokens()==before_b.tokens &&
            a->identity().generation==before_a.generation && b->identity().generation==before_b.generation);
        batch=allocator.ubatch_reserve(1,1);llama_seq_id id=1;
        batch.n_seq_id[0]=1;batch.seq_id[0]=&id;batch.pos[0]=1;batch.token[0]=20;batch.output[0]=true;
        t.assert_true(!producer->begin(projection,batch,{4},llama_memory_work_phase::mtp_catchup));
        if (!t.assert_true(producer->begin(projection,batch,{4},llama_memory_work_phase::mtp_draft))) return;
        t.assert_true(producer->abort());
        t.assert_true(aa->ready() && ba->ready() && aa->tokens()==3 && ba->tokens()==1);
        if (!t.assert_true(table->project(selected,{ha,hb},0,*projection))) return;
        t.assert_true(!producer->begin(projection,batch,{7},llama_memory_work_phase::mtp_draft));
    });
    t.test("sampled_pending_anchor_counts_once_in_logical_context", [](testing & t) {
        llama_kv_stream_context_limits limits;
        t.assert_true(llama_kv_stream_context_limits_make(33,40,1,limits));
        auto table=llama_kv_stream_slot_table::create(limits);
        auto cache=reserved_cache(509);
        llama_kv_stream_slot_handle handle;
        if (!t.assert_true(table && cache && table->bind(0,{cache},handle) && append(t,*cache,30,0x31))) return;
        size_t logical=0;
        t.assert_true(table->logical_used(handle,1,logical));
        t.assert_equal(size_t(31),logical);
        llama_kv_stream_generation_budget budget;
        t.assert_true(llama_kv_stream_generation_budget_make(limits.slot_limit-logical,7,SIZE_MAX,budget));
        t.assert_equal(uint32_t(2),budget.query_rows);
        t.assert_equal(uint32_t(1),budget.proposal_tokens);
        // Forward the old anchor plus one accepted proposal; the new bonus remains pending.
        t.assert_true(append(t,*cache,budget.query_rows,0x41));
        t.assert_true(table->logical_used(handle,1,logical));
        t.assert_equal(size_t(33),logical);
        t.assert_true(!table->logical_used(handle,2,logical));
        t.assert_equal(size_t(33),logical);
        // Explicit forwarding without sampling does not add a second logical token.
        t.assert_true(append(t,*cache,1,0x51));
        t.assert_true(table->logical_used(handle,0,logical));
        t.assert_equal(size_t(33),logical);
        t.assert_true(!table->logical_used(handle,1,logical));
        t.assert_equal(size_t(33),logical);
    });
    t.test("slot_binding_rejects_duplicate_storage_and_inconsistent_capacity", [](testing & t) {
        llama_kv_stream_context_limits limits;
        t.assert_true(llama_kv_stream_context_limits_make(33, 40, 3, limits));
        auto table = llama_kv_stream_slot_table::create(limits);
        auto cache = reserved_cache(601);
        llama_kv_stream_slot_handle handle;
        t.assert_true(table->bind(0, {cache}, handle));
        handle = {99, 77};
        t.assert_true(!table->bind(1, {cache}, handle));
        std::shared_ptr<llama_kv_stream_logical_cache> alias = llama_kv_stream_logical_cache::create(cache->host());
        t.assert_true(!table->bind(1, {alias}, handle));
        t.assert_true(!table->bind(1, {reserved_cache(601)}, handle));
        t.assert_true(!table->bind(3, {reserved_cache(602)}, handle));
        t.assert_true(!table->bind(1, {}, handle));
        t.assert_equal(uint64_t(77), handle.generation);
        t.assert_equal(uint32_t(99), handle.slot);
        auto invalid = limits; ++invalid.group_limit;
        t.assert_true(!llama_kv_stream_slot_table::create(invalid));
        t.assert_true(!table->bind(1, {llama_kv_stream_logical_cache::create(host(603))}, handle));
        t.assert_true(table->bind(1, {reserved_cache(604)}, handle));
    });
    t.test("demand_backed_slot_histories_publish_and_truncate_independently", [](testing & t) {
        auto first = host(401)->config();
        first.context_tokens = 33;
        auto second = first; second.cache_id = 402;
        const auto reserve = [](const llama_kv_stream_host_config & c) {
            auto sparse = llama_kv_stream_host_reservation::create(c, ggml_backend_cpu_buffer_type(),
                [](ggml_backend_buffer_t owner) {
                    auto * buffer = ggml_backend_buffer_retain(owner);
                    buffer->buft = ggml_backend_cpu_buffer_type();
                    return buffer;
                }, 65536);
            return llama_kv_stream_host::from_reservation(std::move(sparse));
        };
        auto a = llama_kv_stream_logical_cache::create(reserve(first));
        auto b = llama_kv_stream_logical_cache::create(reserve(second));
        if (!t.assert_true(a && b)) return;
        t.assert_true(a->host()->demand_backed() && b->host()->demand_backed());
        t.assert_true(a->host()->buffer() == nullptr && b->host()->buffer() == nullptr);
        t.assert_equal(size_t(0),a->host()->registered_bytes());
        t.assert_true(append(t, *a, 17, 0x11));
        t.assert_true(a->host()->registered_bytes()>0 && a->host()->registered_bytes()==a->host()->committed_bytes());
        t.assert_true(append(t, *b, 9, 0x41));
        t.assert_equal(size_t(17), a->tokens());
        t.assert_equal(size_t(9), b->tokens());
        const auto checkpoint_a = a->checkpoint(), checkpoint_b = b->checkpoint();
        llama_kv_stream_host_layer plane_a, plane_b;
        t.assert_true(a->host()->layer(0, plane_a) && b->host()->layer(0, plane_b));
        t.assert_true(plane_a.k != plane_b.k && plane_a.v != plane_b.v);
        t.assert_equal(uint8_t(0x11), *static_cast<uint8_t *>(plane_a.k));
        t.assert_equal(uint8_t(0x41), *static_cast<uint8_t *>(plane_b.k));
        t.assert_true(a->truncate(4));
        t.assert_equal(checkpoint_b.generation, b->identity().generation);
        t.assert_equal(checkpoint_b.tokens, b->tokens());
        t.assert_true(!a->restore(checkpoint_b) && !b->restore(checkpoint_a));
        t.assert_true(append(t, *a, 3, 0x21));
        t.assert_equal(uint8_t(0x41), *static_cast<uint8_t *>(plane_b.k));
        t.assert_true(append(t, *b, 24, 0x51));
        t.assert_equal(size_t(33), b->tokens());
        t.assert_true(!b->begin(1));
        t.assert_true(a->begin(26));
        t.assert_true(a->cancel());
        t.assert_equal(size_t(7), a->tokens());
    });
    t.test("unretired_failed_growth_cannot_reopen_cache_admission", [](testing & t) {
        auto c = host(409)->config();
        c.shape = {GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 256, 256, 4, 256, 128};
        c.context_tokens = 4097;
        ggml_backend_buffer_ptr held;
        bool fail = false;
        auto sparse = llama_kv_stream_host_reservation::create(c, ggml_backend_cpu_buffer_type(),
            [&](ggml_backend_buffer_t owner) -> ggml_backend_buffer_t {
                if (fail) {
                    held.reset(ggml_backend_buffer_retain(owner));
                    return nullptr;
                }
                owner->buft = ggml_backend_cpu_buffer_type();
                return ggml_backend_buffer_retain(owner);
            }, 65536);
        auto cache = llama_kv_stream_logical_cache::create(llama_kv_stream_host::from_reservation(std::move(sparse)));
        if (!t.assert_true(cache && append(t, *cache, 1, 0x31))) return;
        llama_kv_stream_host_layer prefix;
        t.assert_true(cache->host()->layer(0, prefix));
        const auto checkpoint = cache->checkpoint();
        const auto & layout = cache->host()->layout();
        std::vector<uint8_t> k(256*layout.k_token_bytes, 0x51), v(256*layout.v_token_bytes, 0x61);
        llama_kv_stream_write write;
        t.assert_true(cache->begin(256));
        t.assert_true(cache->content()->prepare({
            {0, operand::k, layout.k_token_bytes, k.data(), k.size()},
            {0, operand::v, layout.v_token_bytes, v.data(), v.size()}}, write));
        fail = true;
        t.assert_true(!cache->publish_host(write));
        t.assert_equal(uint8_t(0x31), *static_cast<uint8_t *>(prefix.k));
        t.assert_true(cache->cancel());
        t.assert_true(!cache->ready());
        const bool reopened = cache->begin(1);
        t.assert_true(!reopened);
        if (reopened) t.assert_true(cache->cancel());
        t.assert_true(!cache->restore(checkpoint));
        t.assert_equal(size_t(1), cache->tokens());
        held.reset();
    });
    t.test("suspension_readiness_requires_acknowledged_host_publication", [](testing & t) {
        auto cache = llama_kv_stream_logical_cache::create(host(909));
        if (!t.assert_true(bool(cache) && cache->ready())) return;
        if (!t.assert_true(cache->begin(1))) return;
        t.assert_true(!cache->ready());
        t.assert_true(cache->cancel() && cache->ready());
        const auto backing = cache->host();
        std::vector<uint8_t> k(backing->layout().k_token_bytes,1), v(backing->layout().v_token_bytes,2);
        llama_kv_stream_write write;
        if (!t.assert_true(cache->begin(1) && cache->content()->prepare({
                {0,operand::k,0,k.data(),k.size()},{0,operand::v,0,v.data(),v.size()}},write) &&
                cache->publish_host(write))) return;
        t.assert_true(!cache->ready());
        t.assert_true(cache->finish() && cache->ready());
        t.assert_true(cache->content()->invalidate());
        t.assert_true(!cache->ready());
    });
    t.test("suffix_truncation_preserves_clean_resident_rows_but_restore_invalidates_them", [](testing & t) {
        auto cache = llama_kv_stream_logical_cache::create(host(202));
        if (!t.assert_true(cache && append(t,*cache,4,0x20))) return;
        const auto content=cache->content();
        t.assert_true(content->flush({{0,operand::k,0,4},{0,operand::v,0,4}},[](const auto &) { return true; }));
        t.assert_true(cache->truncate(2));
        bool dirty=true;
        t.assert_true(content->dirty({0,operand::k,0,2},dirty));
        t.assert_true(!dirty);
        t.assert_true(content->dirty({0,operand::v,2,2},dirty));
        t.assert_true(dirty);
        t.assert_true(cache->restore(cache->checkpoint()));
        t.assert_true(content->dirty({0,operand::k,0,2},dirty));
        t.assert_true(dirty);
    });
    t.test("target_and_mtp_have_distinct_host_bytes_and_publication_revisions", [](testing & t) {
        auto target = llama_kv_stream_logical_cache::create(host(101));
        auto mtp = llama_kv_stream_logical_cache::create(host(202));
        if (!t.assert_true(target && mtp)) return;
        t.assert_true(target->host() != mtp->host());
        t.assert_true(target->content() != mtp->content());
        t.assert_equal(uint64_t(101), target->identity().id);
        t.assert_equal(uint64_t(202), mtp->identity().id);
        t.assert_true(append(t, *mtp, 3, 0x31));
        t.assert_equal(size_t(0), target->tokens());
        t.assert_equal(size_t(3), mtp->tokens());
        llama_kv_stream_host_layer target_layer, mtp_layer;
        t.assert_true(target->host()->layer(0, target_layer) && mtp->host()->layer(0, mtp_layer));
        t.assert_equal(uint8_t(0), static_cast<uint8_t *>(target_layer.k)[0]);
        t.assert_equal(uint8_t(0x31), static_cast<uint8_t *>(mtp_layer.k)[0]);
        t.assert_equal(uint8_t(0x32), static_cast<uint8_t *>(mtp_layer.v)[0]);
        t.assert_true(mtp->identity().generation > target->identity().generation);
    });

    t.test("cancel_and_truncate_invalidate_stale_writes_without_touching_other_cache", [](testing & t) {
        auto target = llama_kv_stream_logical_cache::create(host(101));
        auto mtp = llama_kv_stream_logical_cache::create(host(202));
        if (!t.assert_true(target && mtp && append(t, *mtp, 4, 0x20))) return;
        const auto before = mtp->identity();
        if (!t.assert_true(mtp->begin(2))) return;
        uint8_t value = 9;
        llama_kv_stream_write stale;
        if (!t.assert_true(mtp->content()->prepare({{0, operand::k, 0, &value, 1}}, stale))) return;
        t.assert_true(mtp->cancel());
        t.assert_equal(size_t(4), mtp->tokens());
        t.assert_true(!mtp->content()->commit(stale));
        t.assert_true(mtp->identity().generation > before.generation);
        t.assert_true(mtp->truncate(1));
        t.assert_equal(size_t(1), mtp->tokens());
        t.assert_true(!mtp->truncate(2));
        t.assert_equal(size_t(0), target->tokens());
        t.assert_true(append(t, *mtp, 2, 0x40));
        t.assert_equal(size_t(3), mtp->tokens());
    });

    t.test("external_save_restore_checks_cache_identity_and_rebuilds_frontiers", [](testing & t) {
        auto mtp = llama_kv_stream_logical_cache::create(host(202));
        if (!t.assert_true(mtp && append(t, *mtp, 2, 0x50))) return;
        const auto saved = mtp->checkpoint();
        llama_kv_stream_host_layer layer;
        if (!t.assert_true(mtp->host()->layer(0, layer))) return;
        std::vector<uint8_t> saved_bytes(mtp->host()->bytes());
        std::memcpy(saved_bytes.data(), layer.k, saved_bytes.size());
        if (!t.assert_true(append(t, *mtp, 2, 0x70))) return;
        std::memcpy(layer.k, saved_bytes.data(), saved_bytes.size());
        const auto foreign = llama_kv_stream_logical_cache::create(host(303));
        if (!t.assert_true(bool(foreign))) return;
        t.assert_true(!foreign->restore(saved));
        t.assert_true(mtp->restore(saved));
        t.assert_equal(size_t(2), mtp->tokens());
        t.assert_equal(size_t(2), mtp->frontiers().reserved);
        t.assert_equal(size_t(0), mtp->frontiers().device);
        t.assert_equal(size_t(2), mtp->frontiers().host);
        t.assert_equal(size_t(2), mtp->frontiers().committed);
        t.assert_true(std::memcmp(layer.k, saved_bytes.data(), saved_bytes.size()) == 0);
        t.assert_true(mtp->identity().generation > saved.generation);
        t.assert_true(append(t, *mtp, 1, 0x60));
    });
    t.test("layered_publication_commits_one_frontier_only_after_all_native_layers_complete", [](testing & t) {
        auto config=host(711)->config();config.layers=2;
        auto backing=llama_kv_stream_host::create(config,ggml_backend_cpu_buffer_type());
        auto cache=llama_kv_stream_logical_cache::create(backing);
        if (!t.assert_true(cache && cache->begin(3))) return;
        const auto generation=cache->identity().generation;
        llama_kv_stream_write writes[2];
        for (uint32_t layer=0;layer<2;++layer) {
            const size_t kb=3*backing->layout().k_token_bytes,vb=3*backing->layout().v_token_bytes;
            std::vector<uint8_t> keys(kb,uint8_t(10+layer)),values(vb,uint8_t(20+layer));
            if (!t.assert_true(cache->content()->prepare({{layer,operand::k,0,keys.data(),kb},
                    {layer,operand::v,0,values.data(),vb}},writes[layer]))) return;
        }
        std::vector<llama_kv_stream_write *> partial{&writes[0]};
        t.assert_true(!cache->publish_host(partial));
        t.assert_equal(size_t(0),cache->tokens());
        t.assert_equal(generation,cache->identity().generation);
        t.assert_true(writes[0].pending() && writes[1].pending());
        std::vector<llama_kv_stream_write *> invalid{&writes[0],nullptr,&writes[1]};
        t.assert_true(!cache->publish_host(invalid));
        std::vector<llama_kv_stream_write *> complete{&writes[0],&writes[1]};
        if (!t.assert_true(cache->publish_host(complete) && cache->finish())) return;
        t.assert_equal(size_t(3),cache->tokens());
        t.assert_equal(generation+1,cache->identity().generation);
        for (uint32_t layer=0;layer<2;++layer) {
            llama_kv_stream_host_layer planes;
            if (!t.assert_true(backing->layer(layer,planes))) return;
            const auto * key=static_cast<const uint8_t *>(planes.k),*value=static_cast<const uint8_t *>(planes.v);
            t.assert_true(std::all_of(key,key+3*backing->layout().k_token_bytes,[&](uint8_t byte) {return byte==10+layer;}));
            t.assert_true(std::all_of(value,value+3*backing->layout().v_token_bytes,[&](uint8_t byte) {return byte==20+layer;}));
        }
    });
    t.test("partial_or_out_of_range_kv_cannot_publish_a_token_frontier", [](testing & t) {
        auto mtp = llama_kv_stream_logical_cache::create(host(404));
        if (!t.assert_true(bool(mtp) && mtp->begin(2))) return;
        const auto backing = mtp->host();
        const size_t k_stride = backing->layout().k_token_bytes;
        const size_t v_stride = backing->layout().v_token_bytes;
        std::vector<uint8_t> k(k_stride*2, 1), v(v_stride*2, 2);
        llama_kv_stream_write partial, extra, complete;
        t.assert_true(mtp->content()->prepare({{0, operand::k, 0, k.data(), k.size()}}, partial));
        t.assert_true(!mtp->publish_host(partial));
        t.assert_equal(size_t(0), mtp->frontiers().host);
        t.assert_true(mtp->content()->prepare({
            {0, operand::k, 0, k.data(), k.size()},
            {0, operand::v, v_stride, v.data(), v_stride}}, extra));
        t.assert_true(!mtp->publish_host(extra));
        t.assert_true(mtp->content()->prepare({
            {0, operand::k, 0, k.data(), k.size()},
            {0, operand::v, 0, v.data(), v.size()}}, complete));
        t.assert_true(mtp->publish_host(complete));
        t.assert_true(mtp->finish());
        t.assert_equal(size_t(2), mtp->tokens());
    });
    t.test("generated_prompt_chunks_publish_only_after_writer_completion", [](testing & t) {
        ggml_backend_ptr backend(ggml_backend_cpu_init());
        ggml_backend_buffer_ptr scratch(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), 4096));
        auto backing = host(505, GGML_TYPE_F16, GGML_TYPE_F16);
        if (!t.assert_true(bool(backend) && bool(scratch) && bool(backing))) return;
        auto cache = llama_kv_stream_logical_cache::create(backing);
        auto unique = llama_kv_stream_writer::create(backend.get(), scratch.get(), 4096, backing->config().shape, 9);
        if (!t.assert_true(bool(cache) && bool(unique))) return;
        std::shared_ptr<llama_kv_stream_writer> writer(std::move(unique));
        generated_source first(backend.get(), 9);
        if (!t.assert_true(cache->begin_generated(writer, first.k, first.v))) return;
        t.assert_equal(size_t(9), cache->frontiers().reserved);
        t.assert_equal(size_t(0), cache->frontiers().host);
        t.assert_equal(size_t(0), cache->tokens());
        if (!t.assert_true(cache->complete_generated())) return;
        t.assert_equal(size_t(9), cache->tokens());
        t.assert_equal(size_t(0), cache->frontiers().device);
        generated_source second(backend.get(), 8);
        if (!t.assert_true(cache->begin_generated(writer, second.k, second.v) &&
                cache->complete_generated())) return;
        t.assert_equal(size_t(17), cache->tokens());
        llama_kv_stream_host_layer planes;
        if (!t.assert_true(backing->layer(0, planes))) return;
        const auto * keys = static_cast<const ggml_fp16_t *>(planes.k);
        const auto * values = static_cast<const ggml_fp16_t *>(planes.v);
        for (size_t row : {size_t(0), size_t(8), size_t(9), size_t(16)}) {
            t.assert_equal(ggml_fp32_to_fp16(1.25f), keys[row*32]);
            t.assert_equal(ggml_fp32_to_fp16(2.5f), values[row*32]);
        }
    });
    t.test("cancel_generated_append_drains_before_reusing_host_rows", [](testing & t) {
        ggml_backend_ptr backend(ggml_backend_cpu_init());
        ggml_backend_buffer_ptr scratch(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), 4096));
        auto backing = host(606, GGML_TYPE_F16, GGML_TYPE_F16);
        if (!t.assert_true(bool(backend) && bool(scratch) && bool(backing))) return;
        auto cache = llama_kv_stream_logical_cache::create(backing);
        auto unique = llama_kv_stream_writer::create(backend.get(), scratch.get(), 4096, backing->config().shape, 9);
        if (!t.assert_true(bool(cache) && bool(unique))) return;
        std::shared_ptr<llama_kv_stream_writer> writer(std::move(unique));
        generated_source source(backend.get(), 3);
        if (!t.assert_true(cache->begin_generated(writer, source.k, source.v))) return;
        t.assert_equal(size_t(0), cache->frontiers().host);
        t.assert_true(cache->cancel());
        t.assert_equal(size_t(0), cache->tokens());
        t.assert_true(cache->begin_generated(writer, source.k, source.v));
        t.assert_true(cache->complete_generated());
        t.assert_equal(size_t(3), cache->tokens());
    });
    t.test("wide_generated_prefill_crosses_page_and_writer_tile_boundaries", [](testing & t) {
        ggml_backend_ptr backend(ggml_backend_cpu_init());
        ggml_backend_buffer_ptr scratch(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), 4096));
        auto config = host(707, GGML_TYPE_F16, GGML_TYPE_F16)->config();
        config.context_tokens = 513;
        config.shape.page_tokens = 256;
        auto backing = llama_kv_stream_host::create(config, ggml_backend_cpu_buffer_type());
        if (!t.assert_true(bool(backend) && bool(scratch) && bool(backing))) return;
        auto cache = llama_kv_stream_logical_cache::create(backing);
        auto unique = llama_kv_stream_writer::create(backend.get(), scratch.get(), 4096, config.shape, 257);
        if (!t.assert_true(bool(cache) && bool(unique))) return;
        std::shared_ptr<llama_kv_stream_writer> writer(std::move(unique));
        generated_source first(backend.get(), 257);
        auto wrong_shape = config.shape;
        wrong_shape.type_k = GGML_TYPE_Q8_0;
        wrong_shape.type_v = GGML_TYPE_Q4_0;
        ggml_backend_buffer_ptr wrong_scratch(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), 4096));
        if (!t.assert_true(bool(wrong_scratch))) return;
        auto wrong_unique = llama_kv_stream_writer::create(backend.get(), wrong_scratch.get(), 4096, wrong_shape, 257);
        if (!t.assert_true(bool(wrong_unique))) return;
        std::shared_ptr<llama_kv_stream_writer> wrong_writer(std::move(wrong_unique));
        t.assert_true(!cache->begin_generated(wrong_writer, first.k, first.v));
        t.assert_equal(size_t(0), cache->frontiers().reserved);
        generated_source wrong(backend.get(), 3);
        t.assert_true(!cache->begin_generated(writer, first.k, wrong.v));
        t.assert_equal(size_t(0), cache->tokens());
        if (!t.assert_true(cache->begin_generated(writer, first.k, first.v))) return;
        t.assert_equal(size_t(0), cache->frontiers().host);
        if (!t.assert_true(cache->complete_generated())) return;
        t.assert_true(writer->stats().graph_submissions > 2);
        generated_source second(backend.get(), 256);
        if (!t.assert_true(cache->begin_generated(writer, second.k, second.v) &&
                cache->complete_generated())) return;
        t.assert_equal(size_t(513), cache->tokens());
        t.assert_equal(size_t(0), cache->frontiers().device);
        llama_kv_stream_host_layer planes;
        if (!t.assert_true(backing->layer(0, planes))) return;
        const auto * keys = static_cast<const ggml_fp16_t *>(planes.k);
        const auto * values = static_cast<const ggml_fp16_t *>(planes.v);
        for (size_t row : {size_t(0), size_t(255), size_t(256), size_t(257), size_t(512)}) {
            t.assert_equal(ggml_fp32_to_fp16(1.25f), keys[row*32]);
            t.assert_equal(ggml_fp32_to_fp16(2.5f), values[row*32]);
        }
    });
    return t.summary();
}
