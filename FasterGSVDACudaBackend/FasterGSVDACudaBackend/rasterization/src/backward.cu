#include "backward.h"
#include "kernels_backward.cuh"
#include "buffer_utils.h"
#include "rasterization_config.h"
#include "utils.h"
#include <tiny-cuda-nn/rtc_kernel.h>

void faster_gs::rasterization::backward(
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
    const bool render_base)
{
    const dim3 grid(div_round_up(width, config::tile_width), div_round_up(height, config::tile_height), 1);
    const int n_tiles = grid.x * grid.y;
    const int end_bit = extract_end_bit(n_tiles - 1);

    PrimitiveBuffers primitive_buffers = PrimitiveBuffers::from_blob(primitive_buffers_blob, n_primitives);
    TileBuffers tile_buffers = TileBuffers::from_blob(tile_buffers_blob, n_tiles);
    BucketBuffers bucket_buffers = BucketBuffers::from_blob(bucket_buffers_blob, n_buckets);

    // note that with c++20 one could use a templated lambda to improve readability here
    auto dispatch_rasterize_backward = [&](const uint* instance_primitive_indices) {
        kernels::backward::blend_backward_cu<<<n_buckets, 32>>>(
            tile_buffers.instance_ranges,
            tile_buffers.buckets_offset,
            instance_primitive_indices,
            primitive_buffers.mean2d,
            primitive_buffers.conic_opacity,
            primitive_buffers.color,
            bg_color,
            grad_image,
            image,
            tile_buffers.final_transmittances,
            tile_buffers.max_n_processed,
            tile_buffers.n_processed,
            bucket_buffers.tile_index,
            bucket_buffers.color_transmittance,
            grad_mean2d_helper,
            grad_conic_helper,
            grad_opacities,
            grad_color_helper,
            n_primitives,
            width,
            height,
            grid.x,
            proper_antialiasing
        );
        CHECK_CUDA(config::debug, "blend_backward")
    };
    if (end_bit <= 16) {
        auto instance_buffers = InstanceBuffers<ushort>::from_blob(instance_buffers_blob, n_instances, end_bit);
        instance_buffers.primitive_indices.selector = instance_primitive_indices_selector;
        dispatch_rasterize_backward(instance_buffers.primitive_indices.Current());
    }
    else {
        auto instance_buffers = InstanceBuffers<uint>::from_blob(instance_buffers_blob, n_instances, end_bit);
        instance_buffers.primitive_indices.selector = instance_primitive_indices_selector;
        dispatch_rasterize_backward(instance_buffers.primitive_indices.Current());
    }

    // jit-compiled preprocess backward (the neural appearance passes its dynamic shared memory requirement)
    rtc_preprocess_backward->launch(
        div_round_up(n_primitives, config::block_size_preprocess), config::block_size_preprocess, backward_shmem_bytes, nullptr,
        means,
        scales,
        rotations,
        opacities,
        appearance,
        w2c,
        cam_position,
        primitive_buffers.n_touched_tiles,
        primitive_buffers.color,
        grad_mean2d_helper,
        grad_conic_helper,
        grad_color_helper,
        grad_means,
        grad_scales,
        grad_rotations,
        grad_opacities,
        grad_appearance,
        densification_info,
        n_primitives,
        appearance_degree,
        static_cast<float>(width),
        static_cast<float>(height),
        focal_x,
        focal_y,
        center_x,
        center_y,
        proper_antialiasing,
        render_base
    );
    CHECK_CUDA(config::debug, "preprocess_backward")
}
