#include <cublas_v2.h>
#include <dlfcn.h>
#include <atomic>
#include <cstdlib>
#include <cstdio>

extern "C" __attribute__((visibility("default"))) void ggml_test_cublas_workspace_faults(int,bool,bool);
extern "C" __attribute__((visibility("default"))) int ggml_test_cublas_workspace_hits();
extern "C" __attribute__((visibility("default"))) void ggml_test_cublas_create_faults(int);
extern "C" __attribute__((visibility("default"))) int ggml_test_cublas_create_calls();
extern "C" __attribute__((visibility("default"))) void ggml_test_cublas_workspace_skip(int);

static std::atomic<int> remaining{0}, create_remaining{0}, skip_remaining{0}, hits{0};
static std::atomic<int> create_calls{0};
static std::atomic<bool> match_borrowed{false}, fail_after{false};

extern "C" __attribute__((visibility("default"))) void ggml_test_cublas_workspace_faults(int count,bool borrowed,bool after) {
    match_borrowed=borrowed;fail_after=after;hits=0;skip_remaining=0;remaining=count;
}

extern "C" __attribute__((visibility("default"))) int ggml_test_cublas_workspace_hits() { return hits.load(); }

extern "C" __attribute__((visibility("default"))) void ggml_test_cublas_create_faults(int count) { hits=0;create_calls=0;create_remaining=count; }
extern "C" __attribute__((visibility("default"))) int ggml_test_cublas_create_calls() { return create_calls.load(); }

extern "C" __attribute__((visibility("default"))) void ggml_test_cublas_workspace_skip(int count) { skip_remaining=count; }

extern "C" __attribute__((visibility("default"))) cublasStatus_t cublasCreate_v2(cublasHandle_t * handle) {
    using function_t=decltype(&cublasCreate_v2);
    static auto actual=reinterpret_cast<function_t>(dlsym(RTLD_NEXT,"cublasCreate_v2"));
    if (!actual) std::abort();
    ++create_calls;
    if (const auto * path=std::getenv("GGML_TEST_CUBLAS_CREATE_GUARD_FILE")) {
        if (auto * guard=std::fopen(path,"r")) {
            std::fclose(guard);std::fputs("test_cublas: rejected late handle creation\n",stderr);
            ++hits;*handle=nullptr;return CUBLAS_STATUS_ALLOC_FAILED;
        }
    }
    int count=create_remaining.load();
    while (count>0 && !create_remaining.compare_exchange_weak(count,count-1)) {}
    if (count>0) { ++hits;*handle=nullptr;return CUBLAS_STATUS_ALLOC_FAILED; }
    return actual(handle);
}

extern "C" __attribute__((visibility("default"))) cublasStatus_t cublasSetWorkspace_v2(cublasHandle_t handle,void * workspace,size_t bytes) {
    using function_t=decltype(&cublasSetWorkspace_v2);
    static auto actual=reinterpret_cast<function_t>(dlsym(RTLD_NEXT,"cublasSetWorkspace_v2"));
    if (!actual) std::abort();
    bool inject=false;
    if (bool(bytes)==match_borrowed.load()) {
        int skip=skip_remaining.load();
        while (skip>0 && !skip_remaining.compare_exchange_weak(skip,skip-1)) {}
        if (skip>0) return actual(handle,workspace,bytes);
        int count=remaining.load();
        while (count>0 && !remaining.compare_exchange_weak(count,count-1)) {}
        inject=count>0;
    }
    if (inject && !fail_after.load()) { ++hits;return CUBLAS_STATUS_INTERNAL_ERROR; }
    const auto result=actual(handle,workspace,bytes);
    if (inject && result==CUBLAS_STATUS_SUCCESS) { ++hits;return CUBLAS_STATUS_INTERNAL_ERROR; }
    return result;
}

extern "C" __attribute__((visibility("default"))) void ggml_test_cuda_stream_create_faults(int);
extern "C" __attribute__((visibility("default"))) int ggml_test_cuda_stream_create_calls();
static std::atomic<int> stream_remaining{0},stream_calls{0};
extern "C" __attribute__((visibility("default"))) void ggml_test_cuda_stream_create_faults(int count) {
    stream_calls=0;stream_remaining=count;
}
extern "C" __attribute__((visibility("default"))) int ggml_test_cuda_stream_create_calls() { return stream_calls.load(); }
extern "C" __attribute__((visibility("default"))) cudaError_t cudaStreamCreateWithFlags(cudaStream_t * stream,unsigned int flags) {
    using function_t=decltype(&cudaStreamCreateWithFlags);
    static auto actual=reinterpret_cast<function_t>(dlsym(RTLD_NEXT,"cudaStreamCreateWithFlags"));
    if (!actual) std::abort();
    ++stream_calls;
    int count=stream_remaining.load();
    while (count>0 && !stream_remaining.compare_exchange_weak(count,count-1)) {}
    if (count>0) { *stream=nullptr;return cudaErrorMemoryAllocation; }
    return actual(stream,flags);
}
