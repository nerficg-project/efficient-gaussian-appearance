#pragma once

#include "appearance_params.h"
#include "buffer_utils.h"
#include "rasterization_config.h"
#include "utils.h"
#include <tiny-cuda-nn/rtc_kernel.h>

namespace faster_gs::rasterization {

    inline void launch_rtc_preprocess(
        tcnn::CudaRtcKernel* rtc_preprocess,
        const float3* means,
        const float3* scales,
        const float4* rotations,
        const float* opacities,
        const AppearanceInputs& appearance,
        const float4* w2c,
        const float3* cam_position,
        PrimitiveBuffers& primitive_buffers,
        const int n_primitives,
        const int appearance_degree,
        const dim3& grid,
        const int width,
        const int height,
        const float focal_x,
        const float focal_y,
        const float center_x,
        const float center_y,
        const float near_plane,
        const float far_plane,
        const bool proper_antialiasing,
        const bool render_base)
    {
        rtc_preprocess->launch(
            div_round_up(n_primitives, config::block_size_preprocess), config::block_size_preprocess, 0, nullptr,
            means,
            scales,
            rotations,
            opacities,
            appearance,
            w2c,
            cam_position,
            primitive_buffers.depth_keys.Current(),
            primitive_buffers.primitive_indices.Current(),
            primitive_buffers.n_touched_tiles,
            primitive_buffers.screen_bounds,
            primitive_buffers.mean2d,
            primitive_buffers.conic_opacity,
            primitive_buffers.color,
            primitive_buffers.n_visible_primitives,
            primitive_buffers.n_instances,
            n_primitives,
            appearance_degree,
            grid.x,
            grid.y,
            static_cast<float>(width),
            static_cast<float>(height),
            focal_x,
            focal_y,
            center_x,
            center_y,
            near_plane,
            far_plane,
            proper_antialiasing,
            render_base
        );
    }

}
