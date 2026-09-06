#pragma once

#include "appearance_params.h"
#include "buffer_utils.h"
#include "helper_math.h"
#include <functional>

namespace tcnn { class CudaRtcKernel; }

namespace faster_gs::rasterization {

    void inference(
        std::function<char* (size_t)> resize_primitive_buffers,
        std::function<char* (size_t)> resize_tile_buffers,
        std::function<char* (size_t)> resize_instance_buffers,
        tcnn::CudaRtcKernel* rtc_preprocess,
        const float3* means,
        const float3* scales,
        const float4* rotations,
        const float* opacities,
        const AppearanceInputs& appearance,
        const float4* w2c,
        const float3* cam_position,
        const float3* bg_color,
        float* image,
        const int n_primitives,
        const int appearance_degree,
        const int width,
        const int height,
        const float focal_x,
        const float focal_y,
        const float center_x,
        const float center_y,
        const float near_plane,
        const float far_plane,
        const bool proper_antialiasing,
        const bool render_base,
        const bool to_chw,
        const bool clamp_output);

    template <typename KeyT>
    void rasterize(
        std::function<char* (size_t)>& resize_instance_buffers,
        PrimitiveBuffers& primitive_buffers,
        TileBuffers& tile_buffers,
        const dim3& grid,
        const dim3& block,
        const float3* bg_color,
        float* image,
        const cudaStream_t memset_stream,
        const int n_visible_primitives,
        const int n_instances,
        const int end_bit,
        const int width,
        const int height,
        const bool to_chw,
        const bool clamp_output);

}
