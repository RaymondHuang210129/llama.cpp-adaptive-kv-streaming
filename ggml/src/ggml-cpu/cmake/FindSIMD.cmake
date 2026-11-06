include(CheckCSourceRuns)

# Discard results from probes whose work could be optimized away.
if (NOT GGML_MSVC_SIMD_PROBE_VERSION STREQUAL "1")
    foreach (type AVX AVX2 FMA AVX512)
        unset(${type}_FOUND CACHE)
        unset(${type}_FLAGS CACHE)
        foreach (index 1 2)
            unset(HAS_${type}_${index} CACHE)
            unset(HAS_${type}_${index}_COMPILED CACHE)
            unset(HAS_${type}_${index}_EXITCODE CACHE)
        endforeach()
    endforeach()
    set(GGML_MSVC_SIMD_PROBE_VERSION 1 CACHE INTERNAL "MSVC SIMD probe version" FORCE)
endif()

# Keep the intrinsic results observable when compiler checks use optimization.
set(AVX_CODE "
    #include <immintrin.h>
    int main()
    {
        volatile __m256 input = _mm256_set1_ps(-1.0f);
        __m256 a = _mm256_add_ps(input, _mm256_set1_ps(2.0f));
        return _mm256_movemask_ps(a);
    }
")

set(AVX512_CODE "
    #include <immintrin.h>
    int main()
    {
        volatile __m512i a = _mm512_set1_epi8(0);
        __m512i b = a;
        __mmask64 equality_mask = _mm512_cmp_epi8_mask(a, b, _MM_CMPINT_EQ);
        return equality_mask != ~(__mmask64) 0;
    }
")

set(AVX2_CODE "
    #include <immintrin.h>
    int main()
    {
        volatile __m256i input = _mm256_set1_epi16(-1);
        __m256i a = _mm256_abs_epi16(input);
        return _mm256_movemask_epi8(a) || _mm256_extract_epi64(a, 0) != 0x0001000100010001LL;
    }
")

set(FMA_CODE "
    #include <immintrin.h>
    int main()
    {
        volatile __m256 input = _mm256_set1_ps(1.0f);
        __m256 acc = _mm256_fmadd_ps(input, input, input);
        return _mm256_movemask_ps(_mm256_sub_ps(acc, _mm256_set1_ps(2.0f)));
    }
")

macro(check_sse type flags)
    set(__FLAG_I 1)
    set(CMAKE_REQUIRED_FLAGS_SAVE ${CMAKE_REQUIRED_FLAGS})
    foreach (__FLAG ${flags})
        if (NOT ${type}_FOUND)
            set(CMAKE_REQUIRED_FLAGS ${__FLAG})
            check_c_source_runs("${${type}_CODE}" HAS_${type}_${__FLAG_I})
            if (HAS_${type}_${__FLAG_I})
                set(${type}_FOUND TRUE CACHE BOOL "${type} support")
                set(${type}_FLAGS "${__FLAG}" CACHE STRING "${type} flags")
            endif()
            math(EXPR __FLAG_I "${__FLAG_I}+1")
        endif()
    endforeach()
    set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_SAVE})

    if (NOT ${type}_FOUND)
        set(${type}_FOUND FALSE CACHE BOOL "${type} support")
        set(${type}_FLAGS "" CACHE STRING "${type} flags")
    endif()

    mark_as_advanced(${type}_FOUND ${type}_FLAGS)
endmacro()

# flags are for MSVC only!
check_sse("AVX" " ;/arch:AVX")
if (NOT ${AVX_FOUND})
    set(GGML_AVX OFF)
else()
    set(GGML_AVX ON)
endif()

check_sse("AVX2" " ;/arch:AVX2")
check_sse("FMA" " ;/arch:AVX2")
if ((NOT ${AVX2_FOUND}) OR (NOT ${FMA_FOUND}))
    set(GGML_AVX2 OFF)
else()
    set(GGML_AVX2 ON)
endif()

check_sse("AVX512" " ;/arch:AVX512")
if (NOT ${AVX512_FOUND})
    set(GGML_AVX512 OFF)
else()
    set(GGML_AVX512 ON)
endif()
