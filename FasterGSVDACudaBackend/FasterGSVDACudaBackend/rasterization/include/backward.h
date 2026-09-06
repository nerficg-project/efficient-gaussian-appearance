#pragma once

#include "appearance_params.h"
#include "helper_math.h"
#include <functional>

namespace tcnn { class CudaRtcKernel; }

namespace faster_gs::rasterization {

    void backward(
        tcnn::CudaRtcKernel* rtc_preprocess_backward,
        const float* grad_image,
        const float3* means,
        const float3* scales,
        const float4* rotations,
        const float* opacities,
        const AppearanceInputs& appearance,
        const float4* w2c,
        const float3* cam_position,
        const float3* bg_color,
        const float* image,
        char* primitive_buffers_blob,
        char* tile_buffers_blob,
        char* instance_buffers_blob,
        char* bucket_buffers_blob,
        float2* grad_mean2d_helper,
        float* grad_conic_helper,
        float* grad_color_helper,
        float3* grad_means,
        float3* grad_scales,
        float4* grad_rotations,
        float* grad_opacities,
        const AppearanceGradients& grad_appearance,
        float* densification_info,
        const int n_primitives,
        const int n_instances,
        const int n_buckets,
        const int instance_primitive_indices_selector,
        const int backward_shmem_bytes,
        const int appearance_degree,
        const int width,
        const int height,
        const float focal_x,
        const float focal_y,
        const float center_x,
        const float center_y,
        const bool proper_antialiasing,
        const bool render_base);

}
