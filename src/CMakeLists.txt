llama_add_compile_flags()

#
# libraries
#

# llama

file(GLOB LLAMA_MODELS_SOURCES "models/*.cpp")

add_library(llama
            ../include/llama.h
            llama.cpp
            llama-adapter.cpp
            llama-arch.cpp
            llama-batch.cpp
            llama-chat.cpp
            llama-context.cpp
            llama-context-memory.cpp
            llama-context-memory.h
            llama-context-workspace.cpp
            llama-context-workspace.h
            llama-cparams.cpp
            llama-grammar.cpp
            llama-graph.cpp
            llama-hparams.cpp
            llama-impl.cpp
            llama-io.cpp
            llama-kv-cache.cpp
            llama-kv-cache-iswa.cpp
            llama-kv-cache-dsa.cpp
            llama-kv-cache-msa.cpp
            llama-kv-cache-dsv4.cpp
            llama-memory.cpp
            llama-kv-stream-config.cpp
            llama-kv-stream-config.h
            llama-kv-stream-policy.cpp
            llama-kv-stream-policy.h
            llama-kv-stream-layer-lease.cpp
            llama-kv-stream-layer-lease.h
            llama-kv-stream-binding.cpp
            llama-kv-stream-binding.h
            llama-kv-stream-host.cpp
            llama-kv-stream-host.h
            llama-kv-stream-content.cpp
            llama-kv-stream-content.h
            llama-kv-stream-logical-cache.cpp
            llama-kv-stream-logical-cache.h
            llama-kv-stream-resident.cpp
            llama-kv-stream-mtp-proxy.cpp
            llama-kv-stream-mtp-proxy.h
            llama-kv-stream-publication.cpp
            llama-kv-stream-publication.h
            llama-kv-stream-resident.h
            llama-kv-stream-capture.cpp
            llama-kv-stream-capture.h
            llama-kv-stream-session.cpp
            llama-kv-stream-session.h
            llama-kv-stream-model.cpp
            llama-kv-stream-model.h
            llama-kv-stream-positions.cpp
            llama-kv-stream-positions.h
            llama-kv-stream-prefetch.h
            llama-memory-completion.cpp
            llama-memory-completion.h
            llama-kv-stream-writer.cpp
            llama-kv-stream-writer.h
            llama-memory-executor.cpp
            llama-memory-executor.h
            llama-memory-executor-cuda.cpp
            llama-memory-executor-cuda.h
            llama-memory-workspace.cpp
            llama-memory-workspace.h
            llama-memory-transition.cpp
            llama-memory-transition.h
            llama-memory-layout.cpp
            llama-memory-layout.h
            llama-memory-budget.cpp
            llama-memory-budget.h
            llama-memory-phase.cpp
            llama-memory-phase.h
            llama-memory-phase-owner.cpp
            llama-memory-phase-owner.h
            llama-memory-plan.cpp
            llama-memory-plan.h
            llama-memory-requirements.cpp
            llama-memory-requirements.h
            llama-memory-hybrid.cpp
            llama-memory-hybrid-iswa.cpp
            llama-memory-recurrent.cpp
            llama-mmap.cpp
            llama-model-loader.cpp
            llama-memory-recurrent-spill.cpp
            llama-memory-recurrent-spill.h
            llama-model-saver.cpp
            llama-model.cpp
            llama-quant.cpp
            llama-sampler.cpp
            llama-vocab.cpp
            unicode-data.cpp
            unicode.cpp
            unicode.h
            ${LLAMA_MODELS_SOURCES}
            )

set_target_properties(llama PROPERTIES
    VERSION ${LLAMA_VERSION_BASE}
    SOVERSION ${LLAMA_VERSION_MAJOR}
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
)

target_compile_definitions(llama PRIVATE
    LLAMA_VERSION="${LLAMA_VERSION}"
    LLAMA_COMMIT="${LLAMA_BUILD_COMMIT}"
)

target_include_directories(llama PRIVATE .)
target_include_directories(llama PUBLIC ../include)
target_compile_features   (llama PRIVATE cxx_std_17) # don't bump

target_link_libraries(llama PUBLIC ggml)

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()
