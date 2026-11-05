cmake_minimum_required(VERSION 3.14)

set(directory "${CMAKE_CURRENT_BINARY_DIR}/msvc-simd-config-test")
file(MAKE_DIRECTORY "${directory}/source")
file(WRITE "${directory}/source/CMakeLists.txt" [=[
cmake_minimum_required(VERSION 3.14)
project(msvc_simd_config_test C)

# Repeat the checks on every test run, including after a probe source change.
unset(GGML_MSVC_SIMD_PROBE_VERSION CACHE)
foreach(type AVX AVX2 FMA AVX512)
    unset(${type}_FOUND CACHE)
    unset(${type}_FLAGS CACHE)
    foreach(index 1 2)
        unset(HAS_${type}_${index} CACHE)
        unset(HAS_${type}_${index}_COMPILED CACHE)
        unset(HAS_${type}_${index}_EXITCODE CACHE)
    endforeach()
endforeach()
if (TEST_LEGACY_CACHE)
    set(AVX512_FOUND TRUE CACHE BOOL "Old optimized probe result")
    set(HAS_AVX512_1 1 CACHE INTERNAL "Old optimized probe result")
endif()
include("${SIMD_MODULE}")
file(WRITE "${CMAKE_BINARY_DIR}/features.txt" "${AVX_FOUND};${AVX2_FOUND};${FMA_FOUND};${AVX512_FOUND}")
]=])

set(generator_args -G "${TEST_GENERATOR}")
if (TEST_PLATFORM)
    list(APPEND generator_args -A "${TEST_PLATFORM}")
endif()
if (TEST_TOOLSET)
    list(APPEND generator_args -T "${TEST_TOOLSET}")
endif()

foreach(case unoptimized optimized legacy)
    set(optimization /O2)
    if (case STREQUAL "unoptimized")
        set(optimization /Od)
    endif()
    set(legacy OFF)
    if (case STREQUAL "legacy")
        set(legacy ON)
    endif()
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -S "${directory}/source" -B "${directory}/${case}"
            ${generator_args} "-DCMAKE_C_COMPILER=${TEST_C_COMPILER}"
            "-DCMAKE_MAKE_PROGRAM=${TEST_MAKE_PROGRAM}"
            -DCMAKE_BUILD_TYPE=Release -DCMAKE_TRY_COMPILE_CONFIGURATION=Release
            "-DCMAKE_C_FLAGS_RELEASE=${optimization} /DNDEBUG"
            "-DSIMD_MODULE=${SIMD_MODULE}" "-DTEST_LEGACY_CACHE=${legacy}"
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if (NOT result EQUAL 0)
        message(FATAL_ERROR "${case} SIMD configure failed (${result}):\n${output}\n${error}")
    endif()
    file(READ "${directory}/${case}/features.txt" features)
    if (case STREQUAL "unoptimized")
        set(expected "${features}")
    elseif (NOT features STREQUAL expected)
        message(FATAL_ERROR "${case} SIMD results ${features} differ from unoptimized ${expected}")
    endif()
endforeach()
message(STATUS "Optimized and legacy-cache SIMD probes match unoptimized results: ${expected}")
