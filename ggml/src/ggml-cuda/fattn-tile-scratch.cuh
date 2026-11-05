#pragma once
#include "scratch-plan.h"

// Host metadata for kernels already instantiated in this translation unit.
template<int columns,bool softcap>
static ggml_cuda_operator_kernel_metadata ggml_cuda_tile_workspace_metadata_case(int cc) {
    return {reinterpret_cast<const void *>(flash_attn_tile<72,72,columns,1,softcap>),
        ggml_cuda_fattn_tile_get_nthreads(72,72,columns,cc),ggml_cuda_fattn_tile_get_nbatch_fa(72,72,columns,cc)};
}

template<bool softcap>
static bool ggml_cuda_tile_workspace_metadata_columns(int columns,int cc,ggml_cuda_operator_kernel_metadata & output) {
    switch (columns) {
        case 2:output=ggml_cuda_tile_workspace_metadata_case<2,softcap>(cc);return true;
        case 4:output=ggml_cuda_tile_workspace_metadata_case<4,softcap>(cc);return true;
        case 8:output=ggml_cuda_tile_workspace_metadata_case<8,softcap>(cc);return true;
        case 16:output=ggml_cuda_tile_workspace_metadata_case<16,softcap>(cc);return true;
        case 32:output=ggml_cuda_tile_workspace_metadata_case<32,softcap>(cc);return true;
        default:return false;
    }
}

bool ggml_cuda_operator_tile_kernel_metadata(int columns,bool softcap,int cc,
        ggml_cuda_operator_kernel_metadata & output) {
    return softcap ? ggml_cuda_tile_workspace_metadata_columns<true>(columns,cc,output) :
        ggml_cuda_tile_workspace_metadata_columns<false>(columns,cc,output);
}
