#include "kv-stream-block-test.h"
#include "../ggml/src/ggml-kv-stream-copy.h"
#include "../ggml/src/ggml-kv-stream-registered-copy.h"
#include "ggml-cuda.h"
#include <chrono>
#include <iomanip>
#ifdef KV_COPY_CUDA_TEST
#include <cuda_runtime_api.h>
#include <atomic>
#include <thread>

// Hold an auxiliary CUDA stream without issuing CUDA calls from its callback.
struct delayed_event {
    ggml_backend_event_t event;
    cudaStream_t stream = nullptr;
    std::atomic<bool> opened{false};
    explicit delayed_event(ggml_backend_t backend) : event(ggml_backend_event_new(ggml_backend_get_device(backend))) {
        GGML_ASSERT(event && cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking) == cudaSuccess);
    }
    void arm(ggml_backend_t backend) {
        GGML_ASSERT(cudaLaunchHostFunc(stream,[](void * p) {
            auto & gate = *static_cast<delayed_event *>(p);
            while (!gate.opened.load(std::memory_order_acquire)) std::this_thread::yield();
        },this) == cudaSuccess);
        GGML_ASSERT(cudaEventRecord(static_cast<cudaEvent_t>(event->context),stream) == cudaSuccess);
        ggml_backend_event_wait(backend,event);
    }
    void open() { opened.store(true,std::memory_order_release); }
    // Always release callbacks before joining GPU work, including failed test branches.
    ~delayed_event() { open(); cudaStreamSynchronize(stream); ggml_backend_event_free(event); cudaStreamDestroy(stream); }
};
#endif

static int registered_multi_contract(ggml_backend_t backend) {
    testing t;auto * device=ggml_backend_get_device(backend);auto * reg=ggml_backend_dev_backend_reg(device);
    auto get=reinterpret_cast<ggml_kv_stream_registered_copy_ops_get>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_kv_stream_registered_copy_ops"));
    auto copy_get=reinterpret_cast<ggml_kv_stream_copy_ops_get>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_kv_stream_copy_ops"));
    if (!t.assert_true(get && get() && get()->version>=2 && get()->create_multi && copy_get && copy_get())) return t.summary();
    const auto * copies=copy_get();const auto * registered=get();
    const ggml_kv_stream_shape shape{GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,256,256,4,256,128};ggml_kv_stream_layout page,ring;
    if (!t.assert_true(ggml_kv_stream_layout_make(shape,256,page).status==ggml_kv_stream_status::success &&
            ggml_kv_stream_layout_make(shape,512,ring).status==ggml_kv_stream_status::success)) return t.summary();
    ggml_backend_buffer_ptr gpu(ggml_backend_buft_alloc_buffer(llama_kv_stream_device_buffer_type(device),ring.bytes));
    std::unique_ptr<void,void (*)(void *)> prepared(copies->prepare(backend,2,false),copies->free_prepared);
    if (!t.assert_true(gpu && prepared)) return t.summary();
    llama_kv_stream_host_config config{99,shape,{{GGML_TYPE_Q8_0,true,true,true,true},{GGML_TYPE_Q4_0,true,true,true,true},true,true},256,1};
    std::vector<decltype(llama_kv_stream_host_reservation::create(config,device,65536))> hosts;
    std::vector<ggml_kv_stream_registered_copy_source> scopes;
    std::vector<ggml_kv_stream_registered_copy_fragment> parts[3][2];
    for (size_t source=0;source<3;++source) {
        hosts.push_back(llama_kv_stream_host_reservation::create(config,device,65536));llama_kv_stream_host_layer layer;
        if (!t.assert_true(hosts.back() && hosts.back()->prepare(256) && hosts.back()->layer(0,layer))) return t.summary();
        scopes.push_back({layer.k,hosts.back()->bytes()});
        for (size_t side=0;side<2;++side) {
            std::vector<llama_kv_stream_host_range> ranges;
            if (!t.assert_true(hosts.back()->ranges(0,side ? ggml_kv_stream_operand::v : ggml_kv_stream_operand::k,
                    0,side ? page.v_bytes : page.k_bytes,ranges))) return t.summary();
            for (const auto & range:ranges) {
                std::memset(static_cast<char *>(ggml_backend_buffer_get_base(range.buffer.get()))+range.offset,int(0x50+source+16*side),range.bytes);
                parts[source][side].push_back({range.buffer.get(),range.offset,range.bytes});
            }
        }
    }
    std::vector<size_t> order={0,1,2};std::sort(order.begin(),order.end(),[&](size_t a,size_t b) {return uintptr_t(scopes[a].base)<uintptr_t(scopes[b].base);});
    const ggml_kv_stream_registered_copy_source allowed[]={scopes[order[0]],scopes[order[2]]};
    t.assert_true(!registered->create_multi(backend,prepared.get(),gpu.get(),nullptr,2,shape,2));
    t.assert_true(!registered->create_multi(backend,prepared.get(),gpu.get(),allowed,0,shape,2));
    auto duplicate=std::vector<ggml_kv_stream_registered_copy_source>{allowed[0],allowed[0]};
    t.assert_true(!registered->create_multi(backend,prepared.get(),gpu.get(),duplicate.data(),2,shape,2));
    std::unique_ptr<void,void (*)(void *)> queue(registered->create_multi(backend,prepared.get(),gpu.get(),allowed,2,shape,2),copies->free);
    if (!t.assert_true("one ring accepts multiple disjoint registered sources",queue!=nullptr)) return t.summary();
    ggml_backend_buffer_clear(gpu.get(),0xa5);
    if (!t.assert_true(copies->begin(queue.get()))) return t.summary();
    const auto & foreign_k=parts[order[1]][0];const auto & foreign_v=parts[order[1]][1];
    t.assert_true("the gap between authorized sources does not authorize another cache",!registered->enqueue(queue.get(),0,
        foreign_k.data(),foreign_k.size(),foreign_v.data(),foreign_v.size(),256,256,false));
    t.assert_equal(size_t(0),copies->stats(queue.get()).bytes);
    for (size_t slot=0;slot<2;++slot) {
        const auto owner=order[slot*2];const auto & k=parts[owner][0];const auto & v=parts[owner][1];
        if (!t.assert_true(registered->enqueue(queue.get(),slot,k.data(),k.size(),v.data(),v.size(),256,256,false) &&
                copies->acquire_span(queue.get(),slot,1))) return t.summary();
    }
    ggml_backend_synchronize(backend);
    ggml_context_ptr metadata(ggml_init({ggml_tensor_overhead(),nullptr,true}));auto * tensor=ggml_new_tensor_1d(metadata.get(),GGML_TYPE_I8,int64_t(ring.bytes));
    if (!t.assert_true(ggml_backend_tensor_alloc(gpu.get(),tensor,ggml_backend_buffer_get_base(gpu.get()))==GGML_STATUS_SUCCESS)) return t.summary();
    std::vector<uint8_t> observed(ring.bytes);ggml_backend_tensor_get(tensor,observed.data(),0,observed.size());
    for (size_t slot=0;slot<2;++slot) for (size_t side=0;side<2;++side) {
        const size_t first=(side ? ring.v_offset : 0)+slot*(side ? page.v_bytes : page.k_bytes);
        const size_t bytes=side ? page.v_bytes : page.k_bytes;const auto pattern=uint8_t(0x50+order[slot*2]+16*side);
        t.assert_true(std::all_of(observed.begin()+first,observed.begin()+first+bytes,[&](uint8_t value) {return value==pattern;}));
    }
    t.assert_true(copies->release_span(queue.get(),0,2));copies->drain(queue.get());return t.summary();
}

static int registered_copy_contract(ggml_backend_t backend) {
    testing t;
    auto * device = ggml_backend_get_device(backend);
    auto * reg = ggml_backend_dev_backend_reg(device);
    auto old_get = reinterpret_cast<ggml_kv_stream_copy_ops_get>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_kv_stream_copy_ops"));
    const auto * old = old_get ? old_get() : nullptr;
    if (!t.assert_true(old && old->version >= 11)) return t.summary();
    const ggml_kv_stream_shape shape{GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 256, 256, 4, 256, 128};
    ggml_kv_stream_layout page, ring;
    if (!t.assert_true(ggml_kv_stream_layout_make(shape, 256, page).status == ggml_kv_stream_status::success &&
            ggml_kv_stream_layout_make(shape, 512, ring).status == ggml_kv_stream_status::success)) return t.summary();
    ggml_backend_buffer_ptr gpu(ggml_backend_buft_alloc_buffer(llama_kv_stream_device_buffer_type(device), ring.bytes));
    ggml_backend_buffer_ptr host(ggml_backend_buft_alloc_buffer(llama_kv_stream_host_buffer_type(device), page.bytes));
    if (!t.assert_true(gpu && host)) return t.summary();
    ggml_context_ptr metadata(ggml_init({ggml_tensor_overhead(), nullptr, true}));
    auto * tensor = ggml_new_tensor_1d(metadata.get(), GGML_TYPE_I8, int64_t(ring.bytes));
    if (!t.assert_equal(GGML_STATUS_SUCCESS, ggml_backend_tensor_alloc(gpu.get(), tensor, ggml_backend_buffer_get_base(gpu.get())))) return t.summary();
    auto * bytes = static_cast<uint8_t *>(ggml_backend_buffer_get_base(host.get()));
    std::memset(bytes, 0x51, page.k_bytes);
    std::memset(bytes + page.v_offset, 0x61, page.v_bytes);
    ggml_backend_buffer_clear(gpu.get(), 0xa5);
    std::unique_ptr<void, void (*)(void *)> queue(old->create(backend, gpu.get(), host.get(), shape, 2), old->free);
    if (!t.assert_true(queue && old->begin(queue.get()) && old->enqueue_span(queue.get(), 0,
            bytes, bytes + page.v_offset, 129, 256) && old->acquire_span(queue.get(), 0, 1))) return t.summary();
    ggml_backend_synchronize(backend);
    std::vector<uint8_t> observed(ring.bytes);
    ggml_backend_tensor_get(tensor, observed.data(), 0, observed.size());
    for (bool value : {false, true}) {
        const size_t first = value ? ring.v_offset : 0;
        const size_t stride = value ? page.v_token_bytes : page.k_token_bytes;
        const size_t capacity = value ? page.v_bytes : page.k_bytes;
        t.assert_true(std::all_of(observed.begin() + first, observed.begin() + first + 129*stride,
            [value](uint8_t b) { return b == (value ? 0x61 : 0x51); }));
        t.assert_true(std::all_of(observed.begin() + first + 129*stride, observed.begin() + first + capacity,
            [](uint8_t b) { return b == 0; }));
        t.assert_true(std::all_of(observed.begin() + first + capacity, observed.begin() + first + 2*capacity,
            [](uint8_t b) { return b == 0xa5; }));
    }
    t.assert_true(old->release_completed(queue.get(), 0));
    old->drain(queue.get());
    queue.reset();
    if (t.failures) return t.summary();
    const auto get = reinterpret_cast<ggml_kv_stream_registered_copy_ops_get>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_kv_stream_registered_copy_ops"));
    if (!t.assert_true(get && get() && get()->version >= 1)) return t.summary();
    const auto * registered = get();
    llama_kv_stream_host_config config{99,shape,
        {{GGML_TYPE_Q8_0,true,true,true,true},{GGML_TYPE_Q4_0,true,true,true,true},true,true},8192,1};
    auto sparse = llama_kv_stream_host_reservation::create(config,device,65536);
    if (!t.assert_true(sparse && sparse->prepare(129))) return t.summary();
    llama_kv_stream_host_layer source;
    if (!t.assert_true(sparse->layer(0,source))) return t.summary();
    const ggml_kv_stream_registered_copy_source scope{source.k,sparse->bytes()};
    std::vector<llama_kv_stream_host_range> keys,values;
    if (!t.assert_true(sparse->ranges(0,ggml_kv_stream_operand::k,0,129*page.k_token_bytes,keys) &&
            sparse->ranges(0,ggml_kv_stream_operand::v,0,129*page.v_token_bytes,values))) return t.summary();
    std::vector<ggml_kv_stream_registered_copy_fragment> kp,vp;
    for (const auto & part : keys) {
        std::memset(static_cast<char *>(ggml_backend_buffer_get_base(part.buffer.get()))+part.offset,0x51,part.bytes);
        kp.push_back({part.buffer.get(),part.offset,part.bytes});
    }
    for (const auto & part : values) {
        std::memset(static_cast<char *>(ggml_backend_buffer_get_base(part.buffer.get()))+part.offset,0x61,part.bytes);
        vp.push_back({part.buffer.get(),part.offset,part.bytes});
    }
    t.assert_true(kp.size() > 1 && vp.size() > 1);
    std::unique_ptr<void,void (*)(void *)> prepared(old->prepare(backend,2,false),old->free_prepared);
    if (!t.assert_true(prepared != nullptr)) return t.summary();
    t.assert_true(!registered->create(backend,prepared.get(),gpu.get(),{nullptr,scope.bytes},shape,2));
    queue.reset(registered->create(backend,prepared.get(),gpu.get(),scope,shape,2));
    if (!t.assert_true(queue != nullptr)) return t.summary();
    {
        std::unique_ptr<void,void (*)(void *)> direct(registered->create(backend,nullptr,gpu.get(),scope,shape,2),old->free);
        t.assert_true(direct != nullptr);
    }
    ggml_backend_buffer_clear(gpu.get(),0xa5);
    if (!t.assert_true(old->begin(queue.get()))) return t.summary();
    auto bad=vp; --bad.back().bytes;
    t.assert_true(!registered->enqueue(queue.get(),0,kp.data(),kp.size(),bad.data(),bad.size(),129,256,true));
    bad=vp; bad.front().owner=nullptr;
    t.assert_true(!registered->enqueue(queue.get(),0,kp.data(),kp.size(),bad.data(),bad.size(),129,256,true));
    auto bad_key=kp; bad_key.front().offset=SIZE_MAX;
    t.assert_true(!registered->enqueue(queue.get(),0,bad_key.data(),bad_key.size(),vp.data(),vp.size(),129,256,true));
    bad_key=kp; bad_key[1]=bad_key[0];
    t.assert_true(!registered->enqueue(queue.get(),0,bad_key.data(),bad_key.size(),vp.data(),vp.size(),129,256,true));
    const ggml_kv_stream_registered_copy_fragment foreign{host.get(),0,129*page.k_token_bytes};
    t.assert_true(!registered->enqueue(queue.get(),0,&foreign,1,vp.data(),vp.size(),129,256,true));
    t.assert_true(!old->enqueue_span(queue.get(),0,source.k,source.v,129,256));
    t.assert_equal(size_t(0),old->stats(queue.get()).bytes);
    t.assert_equal(size_t(0),old->stats(queue.get()).calls);
    ggml_backend_synchronize(backend);
    ggml_backend_tensor_get(tensor,observed.data(),0,observed.size());
    t.assert_true(std::all_of(observed.begin(),observed.end(),[](uint8_t b) { return b == 0xa5; }));
    if (!t.assert_true(registered->enqueue(queue.get(),0,kp.data(),kp.size(),vp.data(),vp.size(),129,256,true))) return t.summary();
    const size_t expected_calls=kp.size()+vp.size();
    keys.clear(); values.clear(); sparse.reset();
    if (!t.assert_true(old->acquire_span(queue.get(),0,1))) return t.summary();
    ggml_backend_synchronize(backend);
    ggml_backend_tensor_get(tensor,observed.data(),0,observed.size());
    for (bool value : {false,true}) {
        const size_t first=value ? ring.v_offset : 0;
        const size_t stride=value ? page.v_token_bytes : page.k_token_bytes;
        const size_t capacity=value ? page.v_bytes : page.k_bytes;
        t.assert_true(std::all_of(observed.begin()+first,observed.begin()+first+129*stride,
            [value](uint8_t b) { return b == (value ? 0x61 : 0x51); }));
        t.assert_true(std::all_of(observed.begin()+first+129*stride,observed.begin()+first+capacity,
            [](uint8_t b) { return b == 0; }));
        t.assert_true(std::all_of(observed.begin()+first+capacity,observed.begin()+first+2*capacity,
            [](uint8_t b) { return b == 0xa5; }));
    }
    t.assert_equal(expected_calls,old->stats(queue.get()).calls);
    t.assert_equal(129*(page.k_token_bytes+page.v_token_bytes),old->stats(queue.get()).bytes);
    t.assert_true(old->release_completed(queue.get(),0));
    t.assert_true(registered->enqueue(queue.get(),1,kp.data(),kp.size(),vp.data(),vp.size(),129,256,true));
    t.assert_true(old->acquire_span(queue.get(),1,1));
    ggml_backend_synchronize(backend);
    t.assert_true(old->release_completed(queue.get(),1));
    for (bool value : {false,true}) {
        const size_t first=value ? ring.v_offset : 0;
        const size_t stride=value ? page.v_token_bytes : page.k_token_bytes;
        const size_t capacity=value ? page.v_bytes : page.k_bytes;
        ggml_backend_tensor_memset(tensor,0xa5,first+capacity+129*stride,127*stride);
    }
    ggml_backend_synchronize(backend);
    const ggml_kv_stream_registered_copy_fragment one_k{kp.front().owner,kp.front().offset,page.k_token_bytes};
    const ggml_kv_stream_registered_copy_fragment one_v{vp.front().owner,vp.front().offset,page.v_token_bytes};
    t.assert_true(registered->enqueue(queue.get(),0,&one_k,1,&one_v,1,1,256,true));
    t.assert_true(old->acquire_span(queue.get(),0,1));
    ggml_backend_synchronize(backend);
    ggml_backend_tensor_get(tensor,observed.data(),0,observed.size());
    for (bool value : {false,true}) {
        const size_t first=value ? ring.v_offset : 0;
        const size_t stride=value ? page.v_token_bytes : page.k_token_bytes;
        const size_t capacity=value ? page.v_bytes : page.k_bytes;
        const uint8_t pattern=value ? 0x61 : 0x51;
        t.assert_true(std::all_of(observed.begin()+first,observed.begin()+first+stride,[pattern](uint8_t b) { return b == pattern; }));
        t.assert_true(std::all_of(observed.begin()+first+stride,observed.begin()+first+capacity,[](uint8_t b) { return b == 0; }));
        t.assert_true(std::all_of(observed.begin()+first+capacity,observed.begin()+first+capacity+129*stride,[pattern](uint8_t b) { return b == pattern; }));
        t.assert_true(std::all_of(observed.begin()+first+capacity+129*stride,observed.begin()+first+2*capacity,[](uint8_t b) { return b == 0xa5; }));
    }
    t.assert_equal(2*expected_calls+2,old->stats(queue.get()).calls);
    t.assert_equal(259*(page.k_token_bytes+page.v_token_bytes),old->stats(queue.get()).bytes);
    t.assert_true(old->release_completed(queue.get(),0));
    old->drain(queue.get()); queue.reset();
    return t.summary();
}

// Fixed synthetic attention cases, not model/server token throughput.
static int benchmark(ggml_backend_t backend, bool overlap, size_t span_pages, size_t slots, bool fallback, bool wide) {
    const std::vector<size_t> query_counts = wide ? std::vector<size_t>{256,512,1025} : std::vector<size_t>{1,33};
    for (size_t active : {size_t(1025),size_t(8193),size_t(32769)}) for (size_t queries : query_counts) {
        fixture f(backend,true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,active,fallback);
        ggml_kv_stream_execution page;
        GGML_ASSERT(ggml_kv_stream_resolve(f.policy.shape,f.policy.capabilities,256,page).status == ggml_kv_stream_status::success);
        f.policy.pool_bytes = page.storage.bytes*(slots+2)+page.conversion.bytes; f.policy.initial_ring_slots = slots;
        GGML_ASSERT(f.attach()); auto pin = f.binding->acquire();
        block_inputs input(f,active,queries);
        ggml_kv_stream_block_layout layout; ggml_kv_stream_block_layout_make(queries*4,256,layout);
        block_workspace workspace(f,layout.bytes);
        std::vector<double> times;
        for (size_t i = 0; i < 23; ++i) {
            const auto begin = std::chrono::steady_clock::now();
            GGML_ASSERT(f.resident->compute_streamed(0,input.q,input.mask,input.output,active,1.0f/16,workspace.lease.get(),overlap,span_pages));
            const double ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
            if (i >= 3) times.push_back(ms);
        }
        std::sort(times.begin(),times.end());
        double sum = 0; for (float value : input.read()) sum += value;
        std::cout << std::setprecision(10) << "BENCH," << active << ',' << queries << ',' << (times[9]+times[10])/2
                  << ',' << f.resident->last_upload_bytes() << ',' << f.resident->last_upload_calls() << ',' << sum
                  << ',' << f.resident->last_attention_calls() << '\n';
    }
    return 0;
}

#ifdef KV_COPY_BATCH_TRACE
void kv_copy_capture_batch(bool enabled);
size_t kv_copy_batch_launches();
#endif

static int gather_copy_contract(ggml_backend_t backend,bool batch=false) {
    testing t;
    t.test("producer_stream_gather_preserves_private_sources_and_native_byte_order", [&](testing & t) {
        auto * device=ggml_backend_get_device(backend);
        const auto get=reinterpret_cast<ggml_kv_stream_copy_ops_get>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(device),"ggml_backend_kv_stream_copy_ops"));
        const auto * ops=get ? get() : nullptr;
        if (!t.assert_true(ops && ops->version>=11)) return;
        ggml_backend_buffer_ptr parent(ggml_backend_buft_alloc_buffer(llama_kv_stream_device_buffer_type(device),12800));
        if (!t.assert_true(bool(parent))) return;
        ggml_backend_buffer_ptr a(ggml_backend_buffer_view(parent.get(),128,4096)),
            b(ggml_backend_buffer_view(parent.get(),4224,4096)),
            scratch(ggml_backend_buffer_view(parent.get(),8320,4096));
        if (!t.assert_true(a && b && scratch)) return;
        ggml_backend_buffer_clear(parent.get(),0xa5);
        const auto raw=[](ggml_backend_buffer_t buffer) {
            ggml_tensor tensor{};tensor.type=GGML_TYPE_I8;tensor.ne[0]=int64_t(ggml_backend_buffer_get_size(buffer));
            tensor.ne[1]=tensor.ne[2]=tensor.ne[3]=1;tensor.nb[0]=1;
            tensor.nb[1]=tensor.nb[2]=tensor.nb[3]=size_t(tensor.ne[0]);
            tensor.buffer=buffer;tensor.data=ggml_backend_buffer_get_base(buffer);return tensor;
        };
        std::vector<uint8_t> left(4096),right(4096),observed(4096);
        for (size_t i=0;i<left.size();++i) {left[i]=uint8_t(i*7+1);right[i]=uint8_t(i*11+37);}
        auto ta=raw(a.get()),tb=raw(b.get());
        ggml_backend_tensor_set(&ta,left.data(),0,left.size());ggml_backend_tensor_set(&tb,right.data(),0,right.size());
        ggml_backend_tensor_get(&ta,observed.data(),0,observed.size());
        if (!t.assert_true("ordinary CUDA transport control",observed==left)) return;
        if (!t.assert_true("bounded producer-stream gather extension",ops->version>=12 && ops->gather)) return;
        const ggml_kv_stream_copy_rectangle rectangles[]={
            {a.get(),0,32,0,64,32,16},{b.get(),0,32,32,64,32,16}};
        auto output=raw(scratch.get());
        std::vector<uint8_t> expected(4096,0);
        for (size_t row=0;row<16;++row) {
            std::memcpy(expected.data()+64*row,left.data()+32*row,32);
            std::memcpy(expected.data()+64*row+32,right.data()+32*row,32);
        }
        for (size_t replay=0;replay<16;++replay) {
            if (!t.assert_true(ops->gather(backend,scratch.get(),rectangles,2,true))) return;
            ggml_backend_synchronize(backend);
            ggml_backend_tensor_get(&output,observed.data(),0,observed.size());
            if (!t.assert_true("interleaved rows and zero padding are exact",observed==expected)) return;
        }
#ifdef KV_COPY_CUDA_TEST
        {
            ggml_backend_buffer_ptr host(ggml_backend_buft_alloc_buffer(llama_kv_stream_host_buffer_type(device),12288));
            if (!t.assert_true(bool(host))) return;
            auto * bytes=static_cast<uint8_t *>(ggml_backend_buffer_get_base(host.get()));
            for (size_t i=0;i<4096;++i) {left[i]=uint8_t(i*3+97);right[i]=uint8_t(i*13+71);}
            std::memcpy(bytes,left.data(),4096);std::memcpy(bytes+4096,right.data(),4096);
            for (size_t row=0;row<16;++row) {
                std::memcpy(expected.data()+64*row,left.data()+32*row,32);
                std::memcpy(expected.data()+64*row+32,right.data()+32*row,32);
            }
            delayed_event gate(backend);
            std::atomic<bool> submitted{false},watchdog_used{false};
            std::thread watchdog([&]() {
                const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
                while (!submitted.load() && std::chrono::steady_clock::now()<deadline)
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                if (!submitted.load()) {watchdog_used=true;gate.open();}
            });
            gate.arm(backend);
            ggml_backend_tensor_set_async(backend,&ta,bytes,0,4096);
            ggml_backend_tensor_set_async(backend,&tb,bytes+4096,0,4096);
            const bool queued=ops->gather(backend,scratch.get(),rectangles,2,true);
            ggml_backend_tensor_get_async(backend,&output,bytes+8192,0,4096);
            submitted=true;gate.open();watchdog.join();
            ggml_backend_synchronize(backend);
            t.assert_true("gather enqueues without waiting for a blocked producer",queued && !watchdog_used.load());
            t.assert_true("gather and consumer follow the actual producer stream",!std::memcmp(bytes+8192,expected.data(),4096));
        }
#endif
        for (int change=0;change<7;++change) {
            auto bad=rectangles[1];
            if (change==0) bad.source=nullptr;
            if (change==1) bad.source_offset=4090;
            if (change==2) bad.destination_offset=4090;
            if (change==3) bad.destination_offset=0;
            if (change==4) bad.source=scratch.get();
            if (change==5) bad.rows=257;
            if (change==6) bad.source_stride=SIZE_MAX;
            const ggml_kv_stream_copy_rectangle invalid[]={rectangles[0],bad};
            ggml_backend_tensor_memset(&output,0xa5,0,4096);
            t.assert_true(!ops->gather(backend,scratch.get(),invalid,2,true));
            ggml_backend_synchronize(backend);
            ggml_backend_tensor_get(&output,observed.data(),0,observed.size());
            t.assert_true("invalid later rectangle cannot clear or modify scratch",
                std::all_of(observed.begin(),observed.end(),[](uint8_t x) {return x==0xa5;}));
        }
        ggml_backend_tensor_get(&ta,observed.data(),0,observed.size());t.assert_true(observed==left);
        ggml_backend_tensor_get(&tb,observed.data(),0,observed.size());t.assert_true(observed==right);
        ggml_backend_buffer_ptr foreign(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(),4096));
        auto invalid=rectangles[0];invalid.source=foreign.get();
        t.assert_true(!ops->gather(backend,scratch.get(),&invalid,1,true));
        t.assert_true(!ops->gather(nullptr,scratch.get(),rectangles,2,true));
        t.assert_true(!ops->gather(backend,nullptr,rectangles,2,true));
        t.assert_true(ops->gather(backend,scratch.get(),nullptr,0,true));
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(&output,observed.data(),0,observed.size());
        t.assert_true(std::all_of(observed.begin(),observed.end(),[](uint8_t x) {return x==0;}));
        auto root=raw(parent.get());std::vector<uint8_t> all(12800);
        ggml_backend_tensor_get(&root,all.data(),0,all.size());
        t.assert_true(std::all_of(all.begin(),all.begin()+128,[](uint8_t x) {return x==0xa5;}) &&
            std::all_of(all.begin()+12416,all.end(),[](uint8_t x) {return x==0xa5;}));
        t.assert_true(ops->gather(backend,scratch.get(),rectangles,1,true));
        t.assert_true(ops->gather(backend,scratch.get(),rectangles+1,1,false));
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(&output,observed.data(),0,observed.size());
        t.assert_true("a later private-page packet preserves the earlier packet",observed==expected);
        if (batch) {
            for (size_t count : {33u,65u,127u,256u,512u}) for (size_t width : {1u,3u,8u,16u}) {
                if (count*width>4096) continue;
                std::vector<ggml_kv_stream_copy_rectangle> packets;
                std::vector<uint8_t> reference(4096,0);
                for (size_t i=0;i<count;++i) {
                    const size_t offset=(i*19%(4096/width))*width;
                    const size_t stride=i%2 ? SIZE_MAX : width;
                    packets.push_back({i%2 ? b.get() : a.get(),offset,stride,i*width,stride,width,1});
                    std::memcpy(reference.data()+i*width,(i%2 ? right : left).data()+offset,width);
                }
#ifdef KV_COPY_BATCH_TRACE
                kv_copy_capture_batch(true);
#endif
                const bool copied=ops->gather(backend,scratch.get(),packets.data(),packets.size(),true);
#ifdef KV_COPY_BATCH_TRACE
                kv_copy_capture_batch(false);
                const size_t launches=kv_copy_batch_launches();
                t.out<<"scatter count="<<count<<" width="<<width<<" launches="<<launches<<std::endl;
                t.assert_true("a scattered tile uses at most one kernel submission per64 rectangles",launches>0 && launches<=(count+63)/64);
#else
                t.assert_true("native kernel entry tracing is required",false);
#endif
                if (!t.assert_true(copied)) return;
                ggml_backend_synchronize(backend);
                ggml_backend_tensor_get(&output,observed.data(),0,observed.size());
                t.assert_true("aligned and unaligned scattered bytes retain their exact native order",observed==reference);
            }
            ggml_backend_tensor_get(&ta,observed.data(),0,observed.size());t.assert_true(observed==left);
            ggml_backend_tensor_get(&tb,observed.data(),0,observed.size());t.assert_true(observed==right);
            ggml_backend_tensor_get(&root,all.data(),0,all.size());
            t.assert_true(std::all_of(all.begin(),all.begin()+128,[](uint8_t x) {return x==0xa5;}) &&
                std::all_of(all.begin()+12416,all.end(),[](uint8_t x) {return x==0xa5;}));
        }
    });
    return t.summary();
}

int main(int argc, char ** argv) {
    const bool multi = argc > 1 && std::strcmp(argv[1],"--cuda-registered-multi") == 0;
    const bool registered = argc > 1 && std::strcmp(argv[1], "--cuda-registered") == 0;
    const bool batch = argc>1 && !std::strcmp(argv[1],"--cuda-gather-batch");
    const bool gather = batch || (argc > 1 && std::strcmp(argv[1], "--cuda-gather") == 0);
    const bool wide = argc > 1 && std::strcmp(argv[1],"--bench-wide") == 0;
    const bool ordered = argc > 1 && std::strcmp(argv[1],"--bench-ordered") == 0;
    const bool spans = argc > 1 && std::strcmp(argv[1],"--bench-span") == 0;
    const bool fallback = argc > 1 && std::strcmp(argv[1],"--bench-fallback") == 0;
    const bool bench = wide || ordered || spans || fallback || (argc > 1 && std::strcmp(argv[1],"--bench") == 0);
    const bool cuda = multi || gather || registered || bench || (argc > 1 && std::strcmp(argv[1],"--cuda") == 0);
    const bool queue_only = cuda && !bench && argc == 3 && std::strcmp(argv[2],"--queue-only") == 0;
    ggml_backend_ptr backend;
    if (cuda) {
        ggml_backend_load_all();
#ifdef KV_COPY_CUDA_TEST
        if (!ggml_backend_reg_by_name("CUDA")) ggml_backend_register(ggml_backend_cuda_reg());
#endif
        auto * dev = ggml_backend_dev_by_name("CUDA0"); if (!dev) return 1;
        backend.reset(ggml_backend_dev_init(dev,nullptr));
    } else backend.reset(ggml_backend_cpu_init());
    if (!backend) return 1;
    if (gather) return gather_copy_contract(backend.get(),batch);
    if (multi) return registered_multi_contract(backend.get());
    if (registered) return registered_copy_contract(backend.get());
    if (bench) {
        const size_t span_pages = argc > 2 ? std::strtoull(argv[2],nullptr,10) : (spans ? 2 : 1);
        const size_t slots = argc > 3 ? std::strtoull(argv[3],nullptr,10) : 2;
        if (!span_pages || !slots || slots > 14 || argc > 4) return 2;
        return benchmark(backend.get(),!ordered,span_pages,slots,fallback,wide);
    }
    testing t;
    t.test("gather_rectangles_validate_all_rows_before_touching_shared_scratch", [](testing & t) {
        ggml_backend_buffer_ptr source(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(),4096));
        ggml_backend_buffer_ptr destination(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(),4096));
        if (!t.assert_true(source && destination)) return;
        const ggml_kv_stream_copy_rectangle rectangles[]={
            {source.get(),0,32,0,64,32,16},
            {source.get(),1024,32,32,64,32,16}};
        t.assert_true(ggml_kv_stream_gather_validate(destination.get(),rectangles,2));
        t.assert_true(ggml_kv_stream_gather_validate(destination.get(),nullptr,0));
        for (int change=0;change<13;++change) {
            auto bad=rectangles[1];
            if (change==0) bad.source=nullptr;
            if (change==1) bad.source_offset=SIZE_MAX;
            if (change==2) bad.destination_offset=SIZE_MAX;
            if (change==3) bad.source_stride=SIZE_MAX;
            if (change==4) bad.destination_stride=SIZE_MAX;
            if (change==5) bad.row_bytes=0;
            if (change==6) bad.rows=0;
            if (change==7) bad.rows=257;
            if (change==8) bad.source_stride=31;
            if (change==9) bad.destination_stride=31;
            if (change==10) bad.destination_offset=0;
            if (change==11) bad.source=destination.get();
            if (change==12) bad.row_bytes=4097;
            const ggml_kv_stream_copy_rectangle invalid[]={rectangles[0],bad};
            t.assert_true(!ggml_kv_stream_gather_validate(destination.get(),invalid,2));
        }
        t.assert_true(!ggml_kv_stream_gather_validate(nullptr,rectangles,2));
        t.assert_true(!ggml_kv_stream_gather_validate(destination.get(),nullptr,1));
        t.assert_true(!ggml_kv_stream_gather_validate(destination.get(),rectangles,513));
    });
    t.test("gather_rectangles_match_an_independent_byte_ownership_oracle", [](testing & t) {
        ggml_backend_buffer_ptr source(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(),4096));
        ggml_backend_buffer_ptr destination(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(),4096));
        if (!t.assert_true(source && destination)) return;
        uint32_t seed=0x713512;
        const auto next=[&]() {seed=1664525*seed+1013904223;return seed;};
        const size_t capacity=ggml_backend_buffer_get_size(destination.get());
        for (size_t trial=0;trial<4096;++trial) {
            ggml_kv_stream_copy_rectangle rectangles[2];
            std::vector<bool> occupied(capacity,false);
            bool expected=true;
            for (auto & r : rectangles) {
                r={source.get(),next()%64,next()%32,next()%64,next()%32,next()%16+1,next()%9};
                if (trial%7==0) r.destination_offset=capacity-16+next()%32;
                if (trial%11==0) r.source_offset=ggml_backend_buffer_get_size(source.get())-16+next()%32;
                expected&=r.rows && r.source_stride>=r.row_bytes && r.destination_stride>=r.row_bytes;
                for (size_t row=0;row<r.rows;++row) for (size_t byte=0;byte<r.row_bytes;++byte) {
                    const size_t src=r.source_offset+row*r.source_stride+byte;
                    const size_t dst=r.destination_offset+row*r.destination_stride+byte;
                    expected&=src<ggml_backend_buffer_get_size(source.get()) && dst<capacity;
                    if (dst<capacity) {expected=expected && !occupied[dst];occupied[dst]=true;}
                }
            }
            if (!t.assert_equal(expected,ggml_kv_stream_gather_validate(destination.get(),rectangles,2))) return;
        }
    });
    t.test("gather_views_use_physical_aliases_and_bound_host_metadata", [](testing & t) {
        ggml_backend_buffer_ptr parent(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(),16384));
        if (!t.assert_true(bool(parent))) return;
        ggml_backend_buffer_ptr source(ggml_backend_buffer_view(parent.get(),0,4096));
        ggml_backend_buffer_ptr destination(ggml_backend_buffer_view(parent.get(),4096,4096));
        if (!t.assert_true(source && destination)) return;
        ggml_kv_stream_copy_rectangle r{source.get(),0,32,0,64,32,16};
        t.assert_true(ggml_kv_stream_gather_validate(destination.get(),&r,1));
        r.source=parent.get();
        t.assert_true(ggml_kv_stream_gather_validate(destination.get(),&r,1));
        r.source_offset=4000;
        t.assert_true(!ggml_kv_stream_gather_validate(destination.get(),&r,1));
        std::vector<ggml_kv_stream_copy_rectangle> copies;
        for (size_t i=0;i<512;++i) copies.push_back({source.get(),0,1,i,1,1,1});
        t.assert_true(ggml_kv_stream_gather_validate(destination.get(),copies.data(),copies.size()));
        copies.push_back({source.get(),0,1,512,1,1,1});
        t.assert_true(!ggml_kv_stream_gather_validate(destination.get(),copies.data(),copies.size()));
        copies={{source.get(),0,1,0,1,1,256},{source.get(),0,1,256,1,1,256}};
        t.assert_true(ggml_kv_stream_gather_validate(destination.get(),copies.data(),copies.size()));
        copies.push_back({source.get(),0,1,512,1,1,1});
        t.assert_true(!ggml_kv_stream_gather_validate(destination.get(),copies.data(),copies.size()));
    });
    t.test("feedback_slots_never_overwrite_uncollected_windows", [](testing & t) {
        ggml_kv_stream_feedback_slots slots;
        t.assert_true(slots.begin());
        const size_t first = slots.current();
        t.assert_true(first < slots.capacity && slots.latest_id() == 1);
        t.assert_true(!slots.begin());
        t.assert_equal(first,slots.seal());
        t.assert_true(slots.begin());
        const size_t second = slots.current();
        t.assert_true(second != first && slots.latest_id() == 2);
        t.assert_equal(second,slots.seal());
        t.assert_true(slots.begin());
        t.assert_equal(slots.none,slots.current());
        t.assert_equal(uint64_t(0),slots.latest_id());
        t.assert_equal(slots.none,slots.seal());
        t.assert_true(!slots.retire(second));
        t.assert_equal(uint64_t(1),slots.id(first));
        t.assert_true(slots.retire(first));
        t.assert_true(slots.begin());
        t.assert_equal(first,slots.current());
        t.assert_equal(uint64_t(3),slots.latest_id());
        t.assert_equal(first,slots.seal());
        t.assert_equal(second,slots.oldest());
        t.assert_true(slots.retire(second) && slots.retire(first));
        t.assert_equal(slots.none,slots.oldest());
        ggml_kv_stream_feedback_slots exhausted(UINT64_MAX);
        t.assert_true(exhausted.begin());
        t.assert_equal(exhausted.none,exhausted.current());
        t.assert_equal(uint64_t(0),exhausted.latest_id());
    });
    t.test("contiguous_spans_stop_at_wrap_and_admission_is_atomic", [](testing & t) {
        t.assert_equal(size_t(3),ggml_kv_stream_contiguous_pages(0,11,5,3));
        t.assert_equal(size_t(2),ggml_kv_stream_contiguous_pages(3,11,5,3));
        t.assert_equal(size_t(1),ggml_kv_stream_contiguous_pages(10,11,5,3));
        t.assert_equal(size_t(0),ggml_kv_stream_contiguous_pages(11,11,5,3));
        t.assert_equal(size_t(0),ggml_kv_stream_contiguous_pages(0,11,0,3));
        t.assert_equal(size_t(0),ggml_kv_stream_contiguous_pages(0,11,5,0));
        t.assert_equal(size_t(2),ggml_kv_stream_contiguous_pages(SIZE_MAX-2,SIZE_MAX,SIZE_MAX,SIZE_MAX));
        ggml_kv_stream_copy_state state(5);
        t.assert_true(state.begin() && state.queue(2));
        t.assert_true(!state.queue_span(0,3));
        t.assert_true(state.can_queue(0) && state.can_queue(1));
        t.assert_true(!state.queue_span(4,2) && !state.queue_span(0,0));
        t.assert_true(state.acquire(2) && state.release(2));
        t.assert_true(state.queue_span(0,3));
        t.assert_true(state.waiting(0) && state.waiting(1) && state.waiting(2));
    });
    t.test("slot_state_rejects_early_reuse_and_resets_after_drain", [](testing & t) {
        ggml_kv_stream_copy_state state(2), zero(0);
        t.assert_true(!zero.begin());
        t.assert_true(!state.queue(0));
        t.assert_true(state.begin());
        t.assert_true(!state.begin());
        for (size_t round = 0; round < 100; ++round) {
            const size_t slot = round%2;
            t.assert_true(state.can_queue(slot));
            t.assert_true(state.recycled(slot) == (round >= 2));
            t.assert_true(state.queue(slot));
            t.assert_true(!state.queue(slot));
            t.assert_true(!state.release(slot));
            t.assert_true(state.acquire(slot));
            t.assert_true(!state.acquire(slot) && !state.can_queue(slot));
            t.assert_true(state.release(slot));
            t.assert_true(!state.release(slot));
        }
        t.assert_true(!state.queue(SIZE_MAX) && !state.acquire(2) && !state.release(2));
        state.drained(); t.assert_true(state.begin());
        t.assert_true(!state.recycled(0) && state.queue(0));
        state.drained(); t.assert_true(state.begin() && state.queue(0));
    });
    if (cuda) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
        auto get = reinterpret_cast<ggml_kv_stream_copy_ops_get>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_kv_stream_copy_ops"));
        t.test("copy_events_are_discoverable", [&](testing & t) { t.assert_true(get && get() && get()->version >= 2 && get()->enqueue_span && get()->stats); });
        if (!get || !get()) return t.summary();
        t.test("copy_feedback_extension_is_available", [&](testing & t) {
            t.assert_true(get()->version >= 10);
            t.assert_true(get()->probe_layer);
            t.assert_true(get()->release_span);
        });
        // Queue-only qualification does not require an attention kernel or Tensor Cores.
        if (!queue_only) t.test("wide_microbatches_share_uploads_and_preserve_final_tile_readers", [&](testing & t) {
            for (bool fallback : {false,true}) for (bool token_major : {false,true}) {
                fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,2049,fallback);
                ggml_kv_stream_execution page; ggml_kv_stream_resolve(f.policy.shape,f.policy.capabilities,256,page);
                f.policy.pool_bytes = page.storage.bytes*5+page.conversion.bytes; f.policy.initial_ring_slots = 3;
                if (!t.assert_true(f.attach())) continue;
                auto pin = f.binding->acquire();
                for (size_t queries : {size_t(255),size_t(256),size_t(257),size_t(511),size_t(512),size_t(1025)}) {
                    block_inputs input(f,2049,queries);
                    if (token_major) {
                        std::vector<float> packed(input.qdata.size());
                        for (size_t q = 0; q < queries; ++q) for (size_t h = 0; h < 4; ++h)
                            std::copy_n(input.qdata.data()+(h*queries+q)*256,256,packed.data()+(q*4+h)*256);
                        ggml_backend_tensor_set(input.q,packed.data(),0,packed.size()*sizeof(float));
                        input.q->nb[1] = 4*256*sizeof(float); input.q->nb[2] = 256*sizeof(float);
                    }
                    ggml_kv_stream_block_layout layout; ggml_kv_stream_block_layout_make(queries*4,256,layout);
                    block_workspace workspace(f,layout.bytes);
                    t.assert_true(f.resident->synchronize(256));
                    if (!t.assert_true(f.resident->compute_streamed(0,input.q,input.mask,input.output,2049,1.0f/16,workspace.lease.get(),true,2))) continue;
                    const auto actual = input.read();
                    if (queries == 1025) for (size_t index : {size_t(255),size_t(256),size_t(1024)}) {
                        std::vector<float> query(4*256), values(4*256);
                        for (size_t head = 0; head < 4; ++head)
                            std::copy_n(input.qdata.data()+(head*queries+index)*256,256,query.data()+head*256);
                        std::copy_n(actual.data()+index*4*256,values.size(),values.data());
                        close_values(t,oracle(f,0,2049-queries+index+1,1,query),values,1e-3f);
                    }
                    const size_t bytes = f.resident->last_upload_bytes(), calls = f.resident->last_upload_calls();
                    t.assert_equal(size_t(1793)*(page.storage.k_token_bytes+page.storage.v_token_bytes),bytes);
                    t.assert_equal(size_t(10),calls);
                    t.assert_equal(fallback ? size_t(9) : size_t(6),f.resident->last_attention_calls());
                    // Independent <=128-query calls exercise different boundaries and one launch each.
                    for (size_t first = 0; first < queries; first += 128) {
                        const size_t count = std::min(size_t(128),queries-first);
                        auto q = *input.q, mask = *input.mask, output = *input.output;
                        q.data = static_cast<char *>(q.data)+first*q.nb[1]; q.ne[1] = int64_t(count);
                        mask.data = static_cast<char *>(mask.data)+first*mask.nb[1]; mask.ne[1] = int64_t(count);
                        mask.nb[2] = mask.nb[3] = count*mask.nb[1];
                        output.data = static_cast<char *>(output.data)+first*output.nb[2]; output.ne[2] = int64_t(count);
                        output.nb[3] = count*output.nb[2];
                        t.assert_true(f.resident->compute_streamed(0,&q,&mask,&output,2049,1.0f/16,workspace.lease.get(),false,2));
                    }
                    close_values(t,actual,input.read(),1e-6f);
                    // Reusing one ring across two layers must not overwrite the last query tile.
                    t.assert_true(f.resident->begin_sequence({0,1},2049,2));
                    t.assert_true(f.resident->compute_streamed(0,input.q,input.mask,input.output,2049,1.0f/16,workspace.lease.get(),true,2));
                    close_values(t,actual,input.read(),1e-6f);
                    t.assert_true(f.resident->compute_streamed(1,input.q,input.mask,input.output,2049,1.0f/16,workspace.lease.get(),true,2));
                    auto second = input.read();
                    t.assert_true(!f.resident->sequence_active());
                    t.assert_true(f.resident->compute_streamed(1,input.q,input.mask,input.output,2049,1.0f/16,workspace.lease.get(),false,2));
                    close_values(t,second,input.read(),1e-6f);
                }
            }
        });
        if (!queue_only) t.test("late_query_failure_preserves_all_output_and_drains_sequence", [&](testing & t) {
            for (bool fallback : {false,true}) {
                fixture f(backend.get(),true,GGML_TYPE_F16,GGML_TYPE_F16,1025,fallback);
                ggml_kv_stream_execution page; ggml_kv_stream_resolve(f.policy.shape,f.policy.capabilities,256,page);
                f.policy.pool_bytes = page.storage.bytes*5+page.conversion.bytes; f.policy.initial_ring_slots = 3;
                if (!t.assert_true(f.attach())) continue;
                auto pin = f.binding->acquire();
                block_inputs input(f,1025,513);
                ggml_kv_stream_block_layout layout; ggml_kv_stream_block_layout_make(513*4,256,layout);
                block_workspace workspace(f,layout.bytes), short_workspace(f,layout.bytes-1);
                t.assert_true(!f.resident->compute_streamed(0,input.q,input.mask,input.output,1025,1.0f/16,short_workspace.lease.get(),true,2));
                const auto before = input.read();
                // Only the final query is invalid, after two valid complete tiles.
                std::vector<float> row(256,NAN);
                for (size_t head = 0; head < 4; ++head)
                    ggml_backend_tensor_set(input.q,row.data(),(head*513+512)*256*sizeof(float),row.size()*sizeof(float));
                t.assert_true(f.resident->begin_sequence({0,1},1025,2));
                t.assert_true(!f.resident->compute_streamed(0,input.q,input.mask,input.output,1025,1.0f/16,workspace.lease.get(),true,2));
                t.assert_true(before == input.read() && !f.resident->sequence_active());
                ggml_backend_tensor_set(input.q,input.qdata.data(),0,input.qdata.size()*sizeof(float));
                t.assert_true(f.resident->compute_streamed(0,input.q,input.mask,input.output,1025,1.0f/16,workspace.lease.get(),true,2));
                std::vector<ggml_fp16_t> masked(input.padded*input.queries,ggml_fp32_to_fp16(-INFINITY));
                ggml_backend_tensor_set(input.mask,masked.data(),0,masked.size()*sizeof(ggml_fp16_t));
                t.assert_true(f.resident->compute_streamed(0,input.q,input.mask,input.output,1025,1.0f/16,workspace.lease.get(),true,2));
                const auto empty = input.read();
                t.assert_true(std::all_of(empty.begin(),empty.end(),[](float x) { return x == 0; }));
            }
        });
        if (!queue_only) t.test("batched_spans_preserve_values_and_reduce_copy_calls", [&](testing & t) {
            for (auto pair : {std::pair{GGML_TYPE_Q8_0,GGML_TYPE_Q4_0},std::pair{GGML_TYPE_IQ4_NL,GGML_TYPE_F32}})
            for (size_t slots : {size_t(3),size_t(5)}) for (bool overlap : {false,true})
            for (size_t limit : {size_t(2),size_t(3),SIZE_MAX}) {
                fixture f(backend.get(),true,pair.first,pair.second,2561);
                ggml_kv_stream_execution page; ggml_kv_stream_resolve(f.policy.shape,f.policy.capabilities,256,page);
                f.policy.pool_bytes = page.storage.bytes*(2+slots)+page.conversion.bytes; f.policy.initial_ring_slots = slots;
                if (!t.assert_true(f.attach())) continue;
                auto pin = f.binding->acquire();
                block_inputs input(f,2561,33);
                ggml_kv_stream_block_layout layout; ggml_kv_stream_block_layout_make(132,256,layout);
                block_workspace workspace(f,layout.bytes);
                t.assert_true(f.resident->synchronize(256));
                t.assert_true(f.resident->compute_streamed(0,input.q,input.mask,input.output,2561,1.0f/16,workspace.lease.get(),overlap,1));
                auto expected = input.read();
                const size_t before = f.resident->last_upload_calls();
                t.assert_true(f.resident->compute_streamed(0,input.q,input.mask,input.output,2561,1.0f/16,workspace.lease.get(),overlap,limit));
                close_values(t,expected,input.read(),1e-6f);
                size_t calls = 0;
                for (size_t block = 0; block < 10;) {
                    const size_t count = std::min({limit,slots-block%slots,size_t(10)-block});
                    block += count; calls += 2;
                }
                t.assert_equal(calls,f.resident->last_upload_calls());
                t.assert_equal(size_t(1)+(f.policy.capabilities.direct_pair ? calls/2 : 10),f.resident->last_attention_calls());
                t.assert_true(f.resident->last_upload_calls() < before);
                t.assert_equal(size_t(2305)*(page.storage.k_token_bytes+page.storage.v_token_bytes),f.resident->last_upload_bytes());
            }
        });
        if (!queue_only) t.test("overlap_matches_ordered_for_native_and_fallback_waves", [&](testing & t) {
            for (auto pair : {std::pair{GGML_TYPE_Q8_0,GGML_TYPE_Q4_0},std::pair{GGML_TYPE_IQ4_NL,GGML_TYPE_F32}})
            for (size_t slots : {size_t(1),size_t(2),size_t(3)}) for (size_t queries : {size_t(1),size_t(33),size_t(257)}) {
                fixture f(backend.get(),true,pair.first,pair.second,2049);
                ggml_kv_stream_execution page; ggml_kv_stream_resolve(f.policy.shape,f.policy.capabilities,256,page);
                f.policy.pool_bytes = page.storage.bytes*(2+slots)+page.conversion.bytes; f.policy.initial_ring_slots = slots;
                if (!t.assert_true(f.attach())) continue;
                auto pin = f.binding->acquire();
                block_inputs input(f,2049,queries);
                ggml_kv_stream_block_layout layout; ggml_kv_stream_block_layout_make(queries*4,256,layout);
                block_workspace workspace(f,layout.bytes);
                for (uint32_t layer : {0u,1u}) {
                    if (!t.assert_true(f.resident->compute_streamed(layer,input.q,input.mask,input.output,2049,1.0f/16,workspace.lease.get(),false))) continue;
                    auto expected = input.read();
                    if (!t.assert_true(f.resident->compute_streamed(layer,input.q,input.mask,input.output,2049,1.0f/16,workspace.lease.get(),true))) continue;
                    t.assert_true(expected == input.read());
                    t.assert_equal(size_t(1793)*(page.storage.k_token_bytes+page.storage.v_token_bytes),f.resident->last_upload_bytes());
                    t.assert_equal(size_t(16),f.resident->last_upload_calls());
                }
            }
        });
        if (!queue_only) t.test("failed_attention_drains_prefetch_and_host_replacement_rebinds", [&](testing & t) {
            fixture f(backend.get(),true,GGML_TYPE_F16,GGML_TYPE_F16,2049);
            f.policy.pool_bytes = f.policy.pool_bytes/16*5; f.policy.initial_ring_slots = 3;
            if (!t.assert_true(f.attach())) return;
            auto pin = f.binding->acquire();
            block_inputs input(f,2049,8);
            ggml_kv_stream_block_layout layout; ggml_kv_stream_block_layout_make(32,256,layout);
            block_workspace workspace(f,layout.bytes);
            std::vector<ggml_fp16_t> row(512,ggml_fp32_to_fp16(NAN));
            llama_kv_stream_write write;
            t.assert_true(f.content->prepare({{0,ggml_kv_stream_operand::v,300*row.size()*2,row.data(),row.size()*2}},write));
            t.assert_true(f.content->commit(write));
            t.assert_true(!f.resident->compute_streamed(0,input.q,input.mask,input.output,2049,1.0f/16,workspace.lease.get(),true));
            auto untouched = input.read();
            t.assert_true(std::all_of(untouched.begin(),untouched.end(),[](float x){return x == -77;}));
            std::fill(row.begin(),row.end(),ggml_fp32_to_fp16(2));
            t.assert_true(f.content->prepare({{0,ggml_kv_stream_operand::v,300*row.size()*2,row.data(),row.size()*2}},write));
            t.assert_true(f.content->commit(write));
            t.assert_true(f.resident->compute_streamed(0,input.q,input.mask,input.output,2049,1.0f/16,workspace.lease.get(),false));
            auto expected = input.read();
            t.assert_true(f.resident->compute_streamed(0,input.q,input.mask,input.output,2049,1.0f/16,workspace.lease.get(),true));
            t.assert_true(expected == input.read());
            auto replacement = llama_kv_stream_host::create(f.host->config(),ggml_backend_buffer_get_type(f.host->buffer()));
            populate(replacement);
            t.assert_true(f.content->replace(replacement)); f.host = replacement;
            t.assert_true(f.resident->compute_streamed(0,input.q,input.mask,input.output,2049,1.0f/16,workspace.lease.get(),false));
            expected = input.read();
            t.assert_true(f.resident->compute_streamed(0,input.q,input.mask,input.output,2049,1.0f/16,workspace.lease.get(),true));
            t.assert_true(expected == input.read());
        });
        if (!queue_only) t.test("failed_subspan_drains_and_retries_without_publishing_partial_output", [&](testing & t) {
            for (auto value : {GGML_TYPE_F16,GGML_TYPE_F32}) {
                fixture f(backend.get(),true,GGML_TYPE_F16,value,2049);
                ggml_kv_stream_execution page; ggml_kv_stream_resolve(f.policy.shape,f.policy.capabilities,256,page);
                f.policy.pool_bytes = page.storage.bytes*5+page.conversion.bytes; f.policy.initial_ring_slots = 3;
                if (!t.assert_true(f.attach())) continue;
                auto pin = f.binding->acquire();
                block_inputs input(f,2049,8);
                ggml_kv_stream_block_layout layout; ggml_kv_stream_block_layout_make(32,256,layout);
                block_workspace workspace(f,layout.bytes);
                t.assert_true(!f.resident->compute_streamed(0,input.q,input.mask,input.output,2049,1.0f/16,workspace.lease.get(),true,0));
                for (size_t token : {size_t(300),size_t(2048)}) {
                    auto before = input.read();
                    std::vector<float> row(512,NAN); std::vector<uint8_t> encoded(page.storage.v_token_bytes);
                    ggml_quantize_chunk(value,row.data(),encoded.data(),0,2,256,nullptr);
                    llama_kv_stream_write write;
                    t.assert_true(f.content->prepare({{0,ggml_kv_stream_operand::v,token*encoded.size(),encoded.data(),encoded.size()}},write));
                    t.assert_true(f.content->commit(write));
                    t.assert_true(!f.resident->compute_streamed(0,input.q,input.mask,input.output,2049,1.0f/16,workspace.lease.get(),true,3));
                    t.assert_true(before == input.read());
                    std::fill(row.begin(),row.end(),2);
                    ggml_quantize_chunk(value,row.data(),encoded.data(),0,2,256,nullptr);
                    t.assert_true(f.content->prepare({{0,ggml_kv_stream_operand::v,token*encoded.size(),encoded.data(),encoded.size()}},write));
                    t.assert_true(f.content->commit(write));
                    t.assert_true(f.resident->compute_streamed(0,input.q,input.mask,input.output,2049,1.0f/16,workspace.lease.get(),false,1));
                    auto expected = input.read();
                    t.assert_true(f.resident->compute_streamed(0,input.q,input.mask,input.output,2049,1.0f/16,workspace.lease.get(),true,3));
                    close_values(t,expected,input.read(),1e-6f);
                }
            }
        });
        if (!queue_only) t.test("all_writable_pairs_match_with_two_slot_prefetch", [&](testing & t) {
            const ggml_type types[] = {GGML_TYPE_F16,GGML_TYPE_BF16,GGML_TYPE_Q4_0,GGML_TYPE_Q4_1,GGML_TYPE_Q5_0,GGML_TYPE_Q5_1,GGML_TYPE_Q8_0,GGML_TYPE_F32,GGML_TYPE_IQ4_NL};
            for (auto key : types) for (auto value : types) {
                fixture f(backend.get(),true,key,value,1025);
                ggml_kv_stream_execution page; ggml_kv_stream_resolve(f.policy.shape,f.policy.capabilities,256,page);
                f.policy.pool_bytes = page.storage.bytes*4+page.conversion.bytes; f.policy.initial_ring_slots = 2;
                if (!t.assert_true(f.attach())) continue;
                auto pin = f.binding->acquire();
                block_inputs input(f,1025,8);
                ggml_kv_stream_block_layout layout; ggml_kv_stream_block_layout_make(32,256,layout);
                block_workspace workspace(f,layout.bytes);
                if (!t.assert_true(f.resident->compute_streamed(0,input.q,input.mask,input.output,1025,1.0f/16,workspace.lease.get(),false))) continue;
                auto expected = input.read();
                if (!t.assert_true(f.resident->compute_streamed(0,input.q,input.mask,input.output,1025,1.0f/16,workspace.lease.get(),true))) continue;
                t.assert_true(expected == input.read());
                t.assert_equal(size_t(769)*(page.storage.k_token_bytes+page.storage.v_token_bytes),f.resident->last_upload_bytes());
                t.assert_true(f.resident->compute_streamed(0,input.q,input.mask,input.output,1025,1.0f/16,workspace.lease.get(),true,2));
                close_values(t,expected,input.read(),1e-6f);
                t.assert_equal(size_t(4),f.resident->last_upload_calls());
                t.assert_equal(size_t(769)*(page.storage.k_token_bytes+page.storage.v_token_bytes),f.resident->last_upload_bytes());
            }
        });
#ifdef KV_COPY_CUDA_TEST
        const auto * ops = get();
        t.test("gpu_deadline_samples_each_span_and_resets_reused_tickets", [&](testing & t) {
            if (!t.assert_true(ops->version >= 4 && ops->measure && ops->acquire_span && ops->feedback)) return;
            fixture f(backend.get(),true);
            auto * device = ggml_backend_memory_lease_buffer(f.lease.get());
            std::unique_ptr<void,void(*)(void*)> queue(ops->create(backend.get(),device,f.host->buffer(),f.policy.shape,3),ops->free);
            if (!t.assert_true(bool(queue))) return;
            llama_kv_stream_host_layer host; f.host->layer(0,host);
            t.assert_true(!ops->feedback(queue.get()).available);
            t.assert_true(ops->measure(queue.get(),true));
            // First-use instrumentation can synchronize a context; execute diagnostics before closing a consumer gate.
            t.assert_true(ops->begin(queue.get()));
            t.assert_true(ops->enqueue_span(queue.get(),0,host.k,host.v,512,512));
            t.assert_true(ops->enqueue(queue.get(),2,host.k,host.v,256,256));
            t.assert_true(ops->acquire_span(queue.get(),0,2));
            t.assert_true(ops->acquire_span(queue.get(),2,1));
            ggml_backend_synchronize(backend.get());
            for (size_t slot = 0; slot < 3; ++slot) t.assert_true(ops->release_completed(queue.get(),slot));
            ops->drain(queue.get());
            t.assert_true(ops->feedback(queue.get()).available);
            for (int epoch = 0; epoch < 2; ++epoch) {
                t.assert_true(ops->begin(queue.get()));
                t.assert_true(!ops->measure(queue.get(),false));
                for (int round = 0; round < 5; ++round) {
                    // Hold consumer execution after the producer fence; copies can become ready meanwhile.
                    delayed_event gate(backend.get()); gate.arm(backend.get());
                    t.assert_true(ops->enqueue_span(queue.get(),0,host.k,host.v,512,512));
                    t.assert_true(ops->enqueue(queue.get(),2,host.k,host.v,256,256));
                    const auto deadline = std::chrono::steady_clock::now()+std::chrono::seconds(5);
                    while ((!ops->ready(queue.get(),1) || !ops->ready(queue.get(),2)) && std::chrono::steady_clock::now() < deadline)
                        std::this_thread::yield();
                    const bool ready = ops->ready(queue.get(),1) && ops->ready(queue.get(),2);
                    t.assert_true("copy readiness reached before the five-second diagnostic deadline",ready);
                    t.assert_true(!ops->acquire_span(queue.get(),0,3)); // separate uploads have separate tickets
                    t.assert_true(!ops->acquire_span(queue.get(),0,0));
                    t.assert_true(ops->acquire_span(queue.get(),0,2));
                    t.assert_true(ops->acquire_span(queue.get(),2,1));
                    t.assert_true(!ops->feedback(queue.get()).available);
                    gate.open(); ggml_backend_synchronize(backend.get());
                    for (size_t slot = 0; slot < 3; ++slot) t.assert_true(ops->release_completed(queue.get(),slot));
                }
                ops->drain(queue.get());
                auto measured = ops->feedback(queue.get());
                t.assert_true(measured.available);
                t.assert_equal(uint64_t(10),measured.samples);
                t.assert_equal(uint64_t(0),measured.misses);
                t.assert_equal(ops->stats(queue.get()).bytes,measured.bytes);
                ggml_kv_stream_layout first_upload;
                t.assert_true(ggml_kv_stream_layout_make(f.policy.shape,512,first_upload).status == ggml_kv_stream_status::success);
                t.assert_equal(first_upload.bytes,measured.timed_bytes);
                t.assert_equal(size_t(3),measured.peak_slots);
                t.assert_true("nonnegative GPU copy interval",measured.copy_ms >= 0);
                t.assert_true("positive host measurement interval",measured.elapsed_ms > 0);
                t.assert_true("diagnostic storage covers all counters",measured.instrumentation_bytes >= 5*sizeof(uint64_t));
                ops->drain(queue.get());
                t.assert_equal(measured.elapsed_ms,ops->feedback(queue.get()).elapsed_ms);
            }
            t.assert_true(ops->measure(queue.get(),false));
            t.assert_true(!ops->feedback(queue.get()).available);
        });
        t.test("deferred_feedback_preserves_two_windows_and_skips_measurement_when_full", [&](testing & t) {
            if (!t.assert_true(ops->version >= 5 && ops->poll_feedback && ops->feedback_id)) return;
            fixture f(backend.get(),true);
            auto * device = ggml_backend_memory_lease_buffer(f.lease.get());
            std::unique_ptr<void,void(*)(void*)> queue(ops->create(backend.get(),device,f.host->buffer(),f.policy.shape,1),ops->free);
            if (!t.assert_true(bool(queue) && ops->measure(queue.get(),true))) return;
            llama_kv_stream_host_layer host; f.host->layer(0,host);
            ggml_kv_stream_layout page; ggml_kv_stream_layout_make(f.policy.shape,256,page);
            auto copy = [&] {
                t.assert_true(ops->enqueue(queue.get(),0,host.k,host.v,256,256));
                t.assert_true(ops->acquire(queue.get(),0));
                ggml_backend_synchronize(backend.get());
                t.assert_true(ops->release_completed(queue.get(),0));
            };
            ggml_kv_stream_copy_snapshot sentinel; sentinel.id = 99; sentinel.value.bytes = 123;
            t.assert_true(!ops->poll_feedback(queue.get(),&sentinel));
            t.assert_true(sentinel.id == 99 && sentinel.value.bytes == 123);
            std::array<double,2> elapsed_upper{};
            for (size_t run = 1; run <= 2; ++run) {
                const auto start = std::chrono::steady_clock::now();
                t.assert_true(ops->begin(queue.get()));
                t.assert_equal(uint64_t(run),ops->feedback_id(queue.get()));
                for (size_t i = 0; i < run; ++i) copy();
                ops->drain(queue.get()); // Do not consume either completed snapshot yet.
                elapsed_upper[run-1] = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            t.assert_true(ops->begin(queue.get()));
            t.assert_equal(uint64_t(0),ops->feedback_id(queue.get()));
            for (size_t run = 1; run <= 2; ++run) {
                ggml_kv_stream_copy_snapshot result;
                const auto deadline = std::chrono::steady_clock::now()+std::chrono::seconds(5);
                while (!ops->poll_feedback(queue.get(),&result) && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
                t.assert_equal(uint64_t(run),result.id);
                t.assert_true(result.value.available && result.value.samples == run && result.value.misses <= run);
                t.assert_equal(run*page.bytes,result.value.bytes);
                t.assert_equal(page.bytes,result.value.timed_bytes);
                t.assert_true(result.value.elapsed_ms <= elapsed_upper[run-1]); // Polling delay is not compute/copy time.
            }
            copy(); ops->drain(queue.get());
            t.assert_equal(page.bytes,ops->stats(queue.get()).bytes); // Inference still copied its data.
            t.assert_true(!ops->poll_feedback(queue.get(),&sentinel) && sentinel.id == 99);
            t.assert_true(ops->begin(queue.get()));
            t.assert_equal(uint64_t(3),ops->feedback_id(queue.get()));
            copy(); ops->drain(queue.get());
            const auto legacy = ops->feedback(queue.get());
            t.assert_true(legacy.available && legacy.samples == 1);
            t.assert_true(!ops->poll_feedback(queue.get(),&sentinel)); // Legacy getter consumed the same snapshot.
            t.assert_true(ops->begin(queue.get())); copy(); ops->drain(queue.get());
            t.assert_true(ops->measure(queue.get(),false)); // Retire pending DMA before freeing host/device diagnostics.
            t.assert_true(!ops->poll_feedback(queue.get(),&sentinel));
        });
        t.test("feedback_eligibility_does_not_change_transfers", [&](testing & t) {
            if (!t.assert_true(ops->version >= 6)) return;
            fixture f(backend.get(),true);
            auto * device = ggml_backend_memory_lease_buffer(f.lease.get());
            std::unique_ptr<void,void(*)(void*)> queue(ops->create(backend.get(),device,f.host->buffer(),f.policy.shape,3),ops->free);
            if (!t.assert_true(bool(queue) && ops->measure(queue.get(),true))) return;
            llama_kv_stream_host_layer host; f.host->layer(0,host);
            const size_t stride = f.host->layout().k_token_bytes+f.host->layout().v_token_bytes;
            for (bool profile : {false,true}) {
                t.assert_true(ops->begin_with_feedback(queue.get(),profile));
                t.assert_true(ops->enqueue_span_with_feedback(queue.get(),0,host.k,host.v,1,256,false));
                t.assert_true(ops->enqueue_span_with_feedback(queue.get(),1,host.k,host.v,512,512,true));
                t.assert_true(ops->acquire_span(queue.get(),0,1) && ops->acquire_span(queue.get(),1,2));
                ggml_backend_synchronize(backend.get());
                for (size_t slot = 0; slot < 3; ++slot) t.assert_true(ops->release_completed(queue.get(),slot));
                ops->drain(queue.get());
                t.assert_equal(size_t(513)*stride,ops->stats(queue.get()).bytes);
                const auto measured = ops->feedback(queue.get());
                t.assert_true(measured.available == profile);
                if (profile) {
                    t.assert_equal(uint64_t(1),measured.samples);
                    t.assert_equal(size_t(512)*stride,measured.timed_bytes);
                } else t.assert_equal(uint64_t(0),ops->feedback_id(queue.get()));
            }
        });
        t.test("one_deadline_per_upload_survives_partial_consumption_and_marker_reuse", [&](testing & t) {
            for (size_t first : {size_t(0),size_t(1)}) {
                fixture f(backend.get(),true);
                auto * device = ggml_backend_memory_lease_buffer(f.lease.get());
                std::unique_ptr<void,void(*)(void*)> queue(ops->create(backend.get(),device,f.host->buffer(),f.policy.shape,2),ops->free);
                if (!t.assert_true(bool(queue) && ops->measure(queue.get(),true) && ops->begin(queue.get()))) continue;
                llama_kv_stream_host_layer host; f.host->layer(0,host);
                ggml_kv_stream_layout page, ring;
                ggml_kv_stream_layout_make(f.policy.shape,256,page); ggml_kv_stream_layout_make(f.policy.shape,512,ring);
                auto wait_ready = [&](size_t slot) {
                    const auto deadline = std::chrono::steady_clock::now()+std::chrono::seconds(5);
                    while (!ops->ready(queue.get(),slot) && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
                    t.assert_true(ops->ready(queue.get(),slot));
                };
                t.assert_true(ops->enqueue_span(queue.get(),0,host.k,host.v,512,512));
                wait_ready(1);
                t.assert_true(ops->acquire(queue.get(),first));
                ggml_backend_synchronize(backend.get()); t.assert_true(ops->release_completed(queue.get(),first));
                t.assert_true(ops->enqueue(queue.get(),first,static_cast<const char *>(host.k)+2*page.k_bytes,
                    static_cast<const char *>(host.v)+2*page.v_bytes,256,256));
                wait_ready(first);
                t.assert_true(ops->acquire(queue.get(),1-first));
                t.assert_true(ops->acquire(queue.get(),first));
                ggml_backend_synchronize(backend.get());
                for (size_t slot = 0; slot < 2; ++slot) t.assert_true(ops->release_completed(queue.get(),slot));
                ops->drain(queue.get());
                const auto measured = ops->feedback(queue.get());
                t.assert_equal(uint64_t(2),measured.samples);
                t.assert_equal(uint64_t(0),measured.misses);
                for (size_t slot = 0; slot < 2; ++slot) for (bool value : {false,true}) {
                    const size_t bytes = value ? page.v_bytes : page.k_bytes;
                    std::vector<uint8_t> actual(bytes);
                    auto * data = static_cast<char *>(ggml_backend_buffer_get_base(device))+(value ? ring.v_offset : 0)+slot*bytes;
                    t.assert_true(cudaMemcpy(actual.data(),data,bytes,cudaMemcpyDeviceToHost) == cudaSuccess);
                    const auto * expected = static_cast<const char *>(value ? host.v : host.k)+(slot == first ? 2 : slot)*bytes;
                    t.assert_true(!std::memcmp(actual.data(),expected,bytes));
                }
            }
        });
        t.test("one_timing_sample_counts_live_bytes_and_resets_after_empty_runs", [&](testing & t) {
            if (!t.assert_true(ops->version >= 4)) return;
            for (auto pair : {std::pair{GGML_TYPE_Q8_0,GGML_TYPE_Q4_0},std::pair{GGML_TYPE_F16,GGML_TYPE_F16},
                              std::pair{GGML_TYPE_IQ4_NL,GGML_TYPE_F32}}) {
                fixture f(backend.get(),true,pair.first,pair.second);
                const auto stride = f.host->layout().k_token_bytes+f.host->layout().v_token_bytes;
                auto * device = ggml_backend_memory_lease_buffer(f.lease.get());
                std::unique_ptr<void,void(*)(void*)> queue(ops->create(backend.get(),device,f.host->buffer(),f.policy.shape,3),ops->free);
                if (!t.assert_true(bool(queue) && ops->measure(queue.get(),true))) continue;
                llama_kv_stream_host_layer host; f.host->layer(0,host);
                for (bool tail_first : {true,false}) {
                    t.assert_true(ops->begin(queue.get()));
                    ops->drain(queue.get());
                    auto empty = ops->feedback(queue.get());
                    t.assert_true(empty.available && empty.samples == 0 && empty.timed_bytes == 0 && empty.copy_ms == 0);
                    t.assert_true(ops->begin(queue.get()));
                    t.assert_true(!ops->enqueue(queue.get(),0,nullptr,host.v,1,256));
                    const size_t first_slot = tail_first ? 2 : 1, second_slot = 0;
                    t.assert_true(ops->enqueue_span(queue.get(),first_slot,host.k,host.v,tail_first ? 1 : 512,tail_first ? 256 : 512));
                    t.assert_true(ops->enqueue_span(queue.get(),second_slot,host.k,host.v,tail_first ? 512 : 1,tail_first ? 512 : 256));
                    t.assert_true(ops->acquire_span(queue.get(),first_slot,tail_first ? 1 : 2));
                    t.assert_true(ops->acquire_span(queue.get(),second_slot,tail_first ? 2 : 1));
                    ggml_backend_synchronize(backend.get());
                    for (size_t slot = 0; slot < 3; ++slot) t.assert_true(ops->release_completed(queue.get(),slot));
                    ops->drain(queue.get());
                    auto measured = ops->feedback(queue.get());
                    t.assert_true(measured.available && measured.samples == 2 && measured.misses <= 2);
                    t.assert_equal(size_t(513)*stride,measured.bytes);
                    t.assert_equal((tail_first ? size_t(1) : size_t(512))*stride,measured.timed_bytes);
                    t.assert_true(measured.copy_ms >= 0 && measured.elapsed_ms > 0);
                }
            }
        });

        t.test("layer_probe_observes_one_pre_wait_deadline", [&](testing & t) {
            if (!t.assert_true(ops->version >= 9 && ops->probe_layer)) return;
            auto * dev=ggml_backend_get_device(backend.get());
            const ggml_kv_stream_shape shape{GGML_TYPE_F16,GGML_TYPE_F16,256,256,2,65536,128};
            ggml_kv_stream_layout plane;
            t.assert_true(ggml_kv_stream_layout_make(shape,131072,plane).status == ggml_kv_stream_status::success);
            ggml_backend_buffer_ptr device(ggml_backend_buft_alloc_buffer(llama_kv_stream_device_buffer_type(dev),plane.bytes));
            ggml_backend_buffer_ptr host(ggml_backend_buft_alloc_buffer(llama_kv_stream_host_buffer_type(dev),plane.bytes));
            if (!t.assert_true(bool(device) && bool(host))) return;
            ggml_backend_buffer_clear(host.get(),0);
            auto * base=static_cast<char *>(ggml_backend_buffer_get_base(host.get()));
            std::unique_ptr<void,void(*)(void*)> queue(
                ops->create(backend.get(),device.get(),host.get(),shape,2),ops->free);
            if (!t.assert_true(bool(queue) && ops->measure(queue.get(),true))) return;
            const ggml_kv_stream_copy_range range{0,2};
            for (bool force_miss : {true,false}) {
                delayed_event gate(backend.get());
                if (force_miss) gate.arm(backend.get());
                t.assert_true(ops->begin(queue.get()));
                t.assert_true(ops->enqueue_span(
                    queue.get(),0,base,base+plane.v_offset,131072,131072));
                if (!force_miss) {
                    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
                    while (!ops->ready(queue.get(),1) && std::chrono::steady_clock::now() < deadline)
                        std::this_thread::yield();
                    t.assert_true(ops->ready(queue.get(),1));
                }
                t.assert_true(ops->probe_layer(queue.get(),&range,1));
                t.assert_true(ops->acquire_span(queue.get(),0,2));
                gate.open(); ggml_backend_synchronize(backend.get());
                t.assert_true(ops->release_completed(queue.get(),0));
                t.assert_true(ops->release_completed(queue.get(),1));
                ops->drain(queue.get());
                const auto measured=ops->feedback(queue.get());
                t.assert_true(measured.available);
                t.assert_equal(uint64_t(1),measured.layer_samples);
                t.assert_equal(uint64_t(force_miss),measured.layer_misses);
            }
            t.assert_true(ops->measure(queue.get(),false));
        });
        t.test("large_transfers_report_actual_gpu_deadlines", [&](testing & t) {
            if (!t.assert_true(ops->version >= 4)) return;
            auto * dev = ggml_backend_get_device(backend.get());
            const ggml_kv_stream_shape shape{GGML_TYPE_F16,GGML_TYPE_F16,256,256,2,131072,128};
            ggml_kv_stream_layout plane;
            t.assert_true(ggml_kv_stream_layout_make(shape,131072,plane).status == ggml_kv_stream_status::success);
            ggml_backend_buffer_ptr device(ggml_backend_buft_alloc_buffer(llama_kv_stream_device_buffer_type(dev),plane.bytes));
            ggml_backend_buffer_ptr host(ggml_backend_buft_alloc_buffer(llama_kv_stream_host_buffer_type(dev),plane.bytes));
            if (!t.assert_true(bool(device) && bool(host))) return;
            ggml_backend_buffer_clear(host.get(),0);
            auto * base = static_cast<char *>(ggml_backend_buffer_get_base(host.get()));
            std::unique_ptr<void,void(*)(void*)> queue(ops->create(backend.get(),device.get(),host.get(),shape,1),ops->free);
            if (!t.assert_true(bool(queue) && ops->measure(queue.get(),true) && ops->begin(queue.get()))) return;
            for (int run = 0; run < 3; ++run) {
                t.assert_true(ops->enqueue(queue.get(),0,base,base+plane.v_offset,131072,131072));
                t.assert_true(ops->acquire_span(queue.get(),0,1));
                ggml_backend_synchronize(backend.get());
                t.assert_true(ops->release_completed(queue.get(),0));
            }
            ops->drain(queue.get());
            const auto result = ops->feedback(queue.get());
            t.assert_true(result.available && result.samples == 3 && result.misses <= 3);
            t.assert_equal(3*plane.bytes,result.bytes);
            t.assert_equal(plane.bytes,result.timed_bytes);
            // Scheduling can make any transfer ready before demand; missing is an observation, not a forced outcome.
            t.out << "Large-transfer GPU deadlines: " << result.misses << '/' << result.samples << " missed\n";
        });
        t.test("completed_consumer_release_is_explicit", [&](testing & t) {
            if (!t.assert_true(ops->version >= 3 && ops->release_completed)) return;
            fixture f(backend.get(),true);
            auto * device = ggml_backend_memory_lease_buffer(f.lease.get());
            std::unique_ptr<void,void(*)(void*)> queue(ops->create(backend.get(),device,f.host->buffer(),f.policy.shape,1),ops->free);
            if (!t.assert_true(bool(queue))) return;
            llama_kv_stream_host_layer host; f.host->layer(0,host);
            delayed_event pending_consumer(backend.get());
            t.assert_true(ops->begin(queue.get()));
            t.assert_true(!ops->release_completed(queue.get(),0));
            t.assert_true(ops->enqueue(queue.get(),0,host.k,host.v,256,256));
            t.assert_true(ops->acquire(queue.get(),0));
            ggml_backend_synchronize(backend.get());
            t.assert_true(ops->release_completed(queue.get(),0));
            t.assert_true(ops->enqueue(queue.get(),0,host.k,host.v,256,256));
            t.assert_true(ops->acquire(queue.get(),0));
            pending_consumer.arm(backend.get());
            t.assert_true(ops->release(queue.get(),0)); // the ordinary queued release still works on reuse
            t.assert_true(ops->enqueue(queue.get(),0,host.k,host.v,256,256));
            const auto deadline = std::chrono::steady_clock::now()+std::chrono::milliseconds(20);
            while (!ops->ready(queue.get(),0) && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
            t.assert_true(!ops->ready(queue.get(),0));
            pending_consumer.open();
            ops->drain(queue.get());
        });
        t.test("multi_page_batches_use_one_ready_and_consumed_fence", [&](testing & t) {
            fixture f(backend.get(),true);
            auto * device=ggml_backend_memory_lease_buffer(f.lease.get());
            std::unique_ptr<void,void(*)(void*)> queue(
                ops->create(backend.get(),device,f.host->buffer(),f.policy.shape,2),ops->free);
            if (!t.assert_true(bool(queue) && ops->begin(queue.get()))) return;
            llama_kv_stream_host_layer host; f.host->layer(0,host);
            for (size_t round=0;round<2;++round) {
                t.assert_true(ops->enqueue_span(queue.get(),0,host.k,host.v,512,512));
                auto stats=ops->stats(queue.get());
                t.assert_equal(round+1,stats.ready_fences);
                t.assert_equal(round,stats.consumed_waits);
                t.assert_true(ops->acquire_span(queue.get(),0,2));
                stats=ops->stats(queue.get());
                t.assert_equal(round+1,stats.ready_waits);
                t.assert_true(ops->release_span(queue.get(),0,2));
                stats=ops->stats(queue.get());
                t.assert_equal(round+1,stats.consumed_fences);
            }
            const auto stats=ops->stats(queue.get());
            t.assert_equal(size_t(2),stats.ready_fences);
            t.assert_equal(size_t(2),stats.ready_waits);
            t.assert_equal(size_t(2),stats.consumed_fences);
            t.assert_equal(size_t(1),stats.consumed_waits);
            ops->drain(queue.get());
        });
        t.test("batched_dma_waits_for_every_consumer_and_counts_actual_transfers", [&](testing & t) {
            fixture f(backend.get(),true);
            ggml_kv_stream_layout page,ring;
            ggml_kv_stream_layout_make(f.policy.shape,256,page); ggml_kv_stream_layout_make(f.policy.shape,512,ring);
            auto * device = ggml_backend_memory_lease_buffer(f.lease.get());
            std::unique_ptr<void,void(*)(void*)> queue(ops->create(backend.get(),device,f.host->buffer(),f.policy.shape,2),ops->free);
            if (!t.assert_true(bool(queue))) return;
            ggml_context_ptr context(ggml_init({65536,nullptr,true}));
            auto * raw = ggml_new_tensor_1d(context.get(),GGML_TYPE_I8,int64_t(ring.bytes));
            GGML_ASSERT(ggml_backend_tensor_alloc(device,raw,ggml_backend_buffer_get_base(device)) == GGML_STATUS_SUCCESS);
            ggml_backend_buffer_ptr sink(ggml_backend_buft_alloc_buffer(llama_kv_stream_host_buffer_type(ggml_backend_get_device(backend.get())),2*ring.k_bytes));
            auto * out = static_cast<uint8_t *>(ggml_backend_buffer_get_base(sink.get()));
            llama_kv_stream_host_layer host; f.host->layer(0,host);
            delayed_event last_consumer(backend.get());
            t.assert_true(ops->begin(queue.get()));
            t.assert_true(!ops->enqueue_span(queue.get(),1,host.k,host.v,512,512));
            t.assert_equal(size_t(0),ops->stats(queue.get()).calls);
            t.assert_true(ops->enqueue_span(queue.get(),0,host.k,host.v,512,512));
            t.assert_equal(size_t(2),ops->stats(queue.get()).calls);
            t.assert_true(!ops->enqueue_span(queue.get(),0,host.k,host.v,512,512));
            t.assert_true(ops->acquire(queue.get(),0));
            ggml_backend_tensor_get_async(backend.get(),raw,out,0,page.k_bytes);
            t.assert_true(ops->release(queue.get(),0));
            t.assert_true(ops->acquire(queue.get(),1));
            last_consumer.arm(backend.get());
            ggml_backend_tensor_get_async(backend.get(),raw,out+page.k_bytes,page.k_bytes,page.k_bytes);
            t.assert_true(ops->release(queue.get(),1));
            auto * next_k = static_cast<char *>(host.k)+ring.k_bytes;
            auto * next_v = static_cast<char *>(host.v)+ring.v_bytes;
            t.assert_true(ops->enqueue_span(queue.get(),0,next_k,next_v,257,512));
            t.assert_true(!ops->ready(queue.get(),0) && !ops->ready(queue.get(),1));
            t.assert_equal(size_t(4),ops->stats(queue.get()).calls);
            t.assert_equal(size_t(769)*(page.k_token_bytes+page.v_token_bytes),ops->stats(queue.get()).bytes);
            last_consumer.open();
            t.assert_true(ops->acquire(queue.get(),0) && ops->acquire(queue.get(),1));
            ggml_backend_tensor_get_async(backend.get(),raw,out+ring.k_bytes,0,ring.k_bytes);
            t.assert_true(ops->release(queue.get(),0) && ops->release(queue.get(),1));
            const auto fence_stats=ops->stats(queue.get());
            t.assert_equal(size_t(2),fence_stats.ready_fences);
            t.assert_equal(size_t(4),fence_stats.ready_waits);
            t.assert_equal(size_t(4),fence_stats.consumed_fences);
            t.assert_equal(size_t(2),fence_stats.consumed_waits);
            ops->drain(queue.get());
            t.assert_true(std::memcmp(out,host.k,ring.k_bytes) == 0);
            t.assert_true(std::memcmp(out+ring.k_bytes,next_k,257*page.k_token_bytes) == 0);
            t.assert_true(std::all_of(out+ring.k_bytes+257*page.k_token_bytes,out+2*ring.k_bytes,[](uint8_t x){return x == 0;}));
            t.assert_true(ops->begin(queue.get()));
            t.assert_equal(size_t(0),ops->stats(queue.get()).bytes);
            ops->drain(queue.get());
        });
        t.test("invalid_storage_and_out_of_range_sources_do_not_queue", [&](testing & t) {
            fixture f(backend.get(),true);
            auto * device = ggml_backend_memory_lease_buffer(f.lease.get());
            ggml_kv_stream_layout page; ggml_kv_stream_layout_make(f.policy.shape,256,page);
            ggml_backend_buffer_ptr pageable(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(),page.bytes));
            t.assert_true(!ops->create(backend.get(),device,pageable.get(),f.policy.shape,1));
            t.assert_true(!ops->create(backend.get(),device,f.host->buffer(),f.policy.shape,0));
            t.assert_true(!ops->create(backend.get(),device,f.host->buffer(),f.policy.shape,SIZE_MAX));
            ggml_backend_buffer_ptr short_device(ggml_backend_buffer_view(device,0,page.bytes-1));
            t.assert_true(!ops->create(backend.get(),short_device.get(),f.host->buffer(),f.policy.shape,1));
            std::unique_ptr<void,void(*)(void*)> q(ops->create(backend.get(),device,f.host->buffer(),f.policy.shape,1),ops->free);
            if (!t.assert_true(bool(q))) return;
            llama_kv_stream_host_layer host; f.host->layer(0,host);
            const auto end = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(f.host->buffer()))+ggml_backend_buffer_get_size(f.host->buffer());
            t.assert_true(!ops->acquire(q.get(),0) && !ops->release(q.get(),0));
            t.assert_true(ops->begin(q.get()) && !ops->begin(q.get()));
            t.assert_true(!ops->enqueue(q.get(),0,reinterpret_cast<void *>(end-1),host.v,1,256));
            t.assert_true(!ops->enqueue(q.get(),0,reinterpret_cast<void *>(UINTPTR_MAX),host.v,1,256));
            t.assert_true(ops->enqueue(q.get(),0,host.k,host.v,256,256));
            t.assert_true(ops->acquire(q.get(),0));
            ops->drain(q.get()); // cancel an acquired slot without issuing a final consumer
            t.assert_true(ops->begin(q.get()) && ops->enqueue(q.get(),0,host.k,host.v,256,256));
            ops->drain(q.get());
            t.assert_true(cudaGetLastError() == cudaSuccess);
        });
        t.test("view_boundaries_and_last_token_copies_preserve_canaries", [&](testing & t) {
            fixture f(backend.get(),true);
            ggml_kv_stream_layout page;
            if (!t.assert_true(ggml_kv_stream_layout_make(f.policy.shape,256,page).status == ggml_kv_stream_status::success)) return;
            const size_t guard = f.policy.shape.alignment, bytes = page.bytes+2*guard;
            ggml_backend_buffer_ptr device(ggml_backend_buft_alloc_buffer(
                llama_kv_stream_device_buffer_type(ggml_backend_get_device(backend.get())),bytes));
            ggml_backend_buffer_ptr host(ggml_backend_buft_alloc_buffer(
                llama_kv_stream_host_buffer_type(ggml_backend_get_device(backend.get())),bytes));
            if (!t.assert_true(device && host)) return;
            auto * base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(device.get()));
            ggml_backend_buffer_clear(device.get(),0xa5);
            ggml_backend_buffer_ptr ring(ggml_backend_buffer_view(device.get(),guard,page.bytes));
            ggml_backend_buffer_ptr pinned(ggml_backend_buffer_view(host.get(),guard,page.bytes));
            if (!t.assert_true(ring && pinned)) return;
            auto * k = static_cast<uint8_t *>(ggml_backend_buffer_get_base(pinned.get()));
            auto * v = k+page.v_offset;
            std::memset(k,0x31,page.k_bytes); std::memset(v,0x52,page.v_bytes);
            std::unique_ptr<void,void(*)(void*)> queue(ops->create(backend.get(),ring.get(),pinned.get(),f.policy.shape,1),ops->free);
            if (!t.assert_true(bool(queue))) return;
            device.reset(); host.reset(); ring.reset(); pinned.reset();
            t.assert_true(ops->begin(queue.get()));
            // The parent has more bytes, but DMA must stay inside the retained host view.
            t.assert_true(!ops->enqueue(queue.get(),0,k,v+1,256,256));
            t.assert_true(ops->enqueue(queue.get(),0,k,v,256,256));
            t.assert_true(ops->acquire(queue.get(),0) && ops->release(queue.get(),0));
            ops->drain(queue.get());
            std::vector<uint8_t> read(bytes);
            if (!t.assert_true(cudaMemcpy(read.data(),base,bytes,cudaMemcpyDeviceToHost) == cudaSuccess)) return;
            std::vector<uint8_t> expected(bytes,0xa5);
            std::fill_n(expected.data()+guard,page.k_bytes,uint8_t(0x31));
            std::fill_n(expected.data()+guard+page.v_offset,page.v_bytes,uint8_t(0x52));
            t.assert_true(expected == read);
            t.assert_true(ops->begin(queue.get()));
            t.assert_true(!ops->enqueue(queue.get(),0,k,v+page.v_bytes,1,256));
            t.assert_true(ops->enqueue(queue.get(),0,k+page.k_bytes-page.k_token_bytes,v+page.v_bytes-page.v_token_bytes,1,256));
            t.assert_true(ops->acquire(queue.get(),0) && ops->release(queue.get(),0));
            ops->drain(queue.get());
            if (!t.assert_true(cudaMemcpy(read.data(),base,bytes,cudaMemcpyDeviceToHost) == cudaSuccess)) return;
            std::fill_n(expected.data()+guard,page.bytes,uint8_t(0));
            std::fill_n(expected.data()+guard,page.k_token_bytes,uint8_t(0x31));
            std::fill_n(expected.data()+guard+page.v_offset,page.v_token_bytes,uint8_t(0x52));
            t.assert_true(expected == read);
            t.assert_true(cudaGetLastError() == cudaSuccess);
        });
        t.test("producer_and_final_consumer_events_prevent_early_copy_and_reuse", [&](testing & t) {
            for (uint32_t width : {256u,512u}) {
            fixture f(backend.get(),true,GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,769,false,2,2,0,width,width);
            ggml_kv_stream_layout page; ggml_kv_stream_layout_make(f.policy.shape,256,page);
            auto * device = ggml_backend_memory_lease_buffer(f.lease.get());
            std::unique_ptr<void,void(*)(void*)> queue(ops->create(backend.get(),device,f.host->buffer(),f.policy.shape,1),ops->free);
            if (!t.assert_true(bool(queue))) return;
            ggml_context_ptr context(ggml_init({65536,nullptr,true}));
            auto * raw = ggml_new_tensor_1d(context.get(),GGML_TYPE_I8,int64_t(page.bytes));
            GGML_ASSERT(ggml_backend_tensor_alloc(device,raw,ggml_backend_buffer_get_base(device)) == GGML_STATUS_SUCCESS);
            ggml_backend_buffer_ptr sink(ggml_backend_buft_alloc_buffer(llama_kv_stream_host_buffer_type(ggml_backend_get_device(backend.get())),2*page.k_bytes));
            auto * output = static_cast<uint8_t *>(ggml_backend_buffer_get_base(sink.get()));
            llama_kv_stream_host_layer host; GGML_ASSERT(f.host->layer(0,host));
            delayed_event producer(backend.get()), consumer(backend.get());
            producer.arm(backend.get());
            t.assert_true(ops->begin(queue.get()));
            t.assert_true(ops->enqueue(queue.get(),0,host.k,host.v,256,256));
            t.assert_true(!ops->ready(queue.get(),0));
            t.assert_true(cudaGetLastError() == cudaSuccess);
            t.assert_true(!ops->enqueue(queue.get(),0,host.k,host.v,256,256));
            producer.open();
            t.assert_true(ops->acquire(queue.get(),0));
            consumer.arm(backend.get());
            ggml_backend_tensor_get_async(backend.get(),raw,output,0,page.k_bytes);
            t.assert_true(ops->release(queue.get(),0));
            auto * next_k = static_cast<char *>(host.k)+page.k_bytes;
            auto * next_v = static_cast<char *>(host.v)+page.v_bytes;
            t.assert_true(ops->enqueue(queue.get(),0,next_k,next_v,1,256));
            t.assert_true(!ops->ready(queue.get(),0));
            t.assert_true(cudaGetLastError() == cudaSuccess);
            consumer.open();
            t.assert_true(ops->acquire(queue.get(),0));
            ggml_backend_tensor_get_async(backend.get(),raw,output+page.k_bytes,0,page.k_bytes);
            t.assert_true(ops->release(queue.get(),0));
            ops->drain(queue.get());
            t.assert_true(std::memcmp(output,host.k,page.k_bytes) == 0);
            t.assert_true(std::memcmp(output+page.k_bytes,next_k,page.k_token_bytes) == 0);
            t.assert_true(std::all_of(output+page.k_bytes+page.k_token_bytes,output+2*page.k_bytes,[](uint8_t x){return x == 0;}));
            t.assert_true(!ops->ready(queue.get(),0) && ops->begin(queue.get()));
            t.assert_true(!ops->enqueue(queue.get(),1,host.k,host.v,256,256));
            t.assert_true(!ops->enqueue(queue.get(),0,host.k,host.v,257,256));
            t.assert_true(!ops->enqueue(queue.get(),0,output,host.v,256,256));
            t.assert_true(ops->enqueue(queue.get(),0,host.k,host.v,256,256));
            ops->drain(queue.get()); // cancel a queued, unacquired block
            t.assert_true(ops->begin(queue.get()));
            ops->drain(queue.get());
            }
        });
        t.test("pending_teardown_drains_and_retains_backing", [&](testing & t) {
            fixture f(backend.get(),true);
            auto * device_type = llama_kv_stream_device_buffer_type(ggml_backend_get_device(backend.get()));
            auto * host_type = llama_kv_stream_host_buffer_type(ggml_backend_get_device(backend.get()));
            ggml_kv_stream_layout page; ggml_kv_stream_layout_make(f.policy.shape,256,page);
            ggml_backend_buffer_ptr device(ggml_backend_buft_alloc_buffer(device_type,page.bytes));
            ggml_backend_buffer_ptr host(ggml_backend_buft_alloc_buffer(host_type,page.bytes));
            std::memset(ggml_backend_buffer_get_base(host.get()),0x35,page.bytes);
            void * q = ops->create(backend.get(),device.get(),host.get(),f.policy.shape,1);
            if (!t.assert_true(q != nullptr)) return;
            auto * k = static_cast<char *>(ggml_backend_buffer_get_base(host.get()));
            delayed_event gate(backend.get());
            gate.arm(backend.get());
            t.assert_true(ops->begin(q));
            t.assert_true(ops->enqueue(q,0,k,k+page.v_offset,256,256));
            // Retention must make these releases non-blocking while the transfer is gated.
            host.reset(); device.reset();
            std::thread release([&] { std::this_thread::sleep_for(std::chrono::milliseconds(20)); gate.open(); });
            ops->free(q);
            t.assert_true(gate.opened.load(std::memory_order_acquire));
            release.join();
        });
#endif
    }
    return t.summary();
}
