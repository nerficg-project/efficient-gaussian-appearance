// note: this header is compiled in two environments: included by the host-side rasterization code,
// and spliced verbatim into the jit-compiled preprocess kernel source (see appearance.cu), which
// nvrtc compiles with c++14; it must therefore avoid host-only includes outside the guard below
// and c++17 syntax (e.g. nested namespace definitions)
#ifndef __CUDACC_RTC__
#pragma once
#include <cuda_fp16.h>
#include <cstdint>
#endif

namespace faster_gs { namespace rasterization {

    // appearance tensor pointers for the jit-compiled preprocess kernels; unused fields are null
    struct AppearanceInputs {
        const float3* precomputed_colors;
        const float3* base_colors;     // view-independent base colors
        const float* residual_params;  // model-specific residual parameters
        const __half* mlp_weights;     // neural only: parameters in tcnn's jit layout
        __half* mlp_outputs;           // neural only: raw mlp outputs for the fused backward (training only)
        uint8_t* fwd_ctx;              // neural only: forward context for the fused backward (training only)
    };

    // appearance gradient pointers for the backward preprocess kernel
    struct AppearanceGradients {
        float3* precomputed_colors;
        float3* base_colors;
        float* residual_params;
        __half* mlp_weights;
    };

}}
