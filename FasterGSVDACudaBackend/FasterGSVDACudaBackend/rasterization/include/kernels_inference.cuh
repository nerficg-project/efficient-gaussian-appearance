#pragma once

#include "kernels_common.cuh"
#include "rasterization_config.h"
#include "helper_math.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace faster_gs::rasterization::kernels::inference {

    __global__ void __launch_bounds__(config::block_size_blend) blend_cu(
        const uint2* __restrict__ tile_instance_ranges,
        const uint* __restrict__ instance_primitive_indices,
        const ushort4* __restrict__ primitive_screen_bounds,
        const float2* __restrict__ primitive_mean2d,
        const float4* __restrict__ primitive_conic_opacity,
        const float3* __restrict__ primitive_color,
        const float3* __restrict__ bg_color,
        float* __restrict__ image,
        const uint width,
        const uint height,
        const uint grid_width,
        const bool output_chw,
        const bool clamp_output)
    {
        auto block = cg::this_thread_block();
        auto warp = cg::tiled_partition<32>(block);
        const dim3 group_index = block.group_index();
        const uint thread_rank = block.thread_rank();
        const uint lane_idx = warp.thread_rank();
        const uint warp_idx = warp.meta_group_rank();
        // setup subtiling
        const uint2 subtile_origin = make_uint2(
            group_index.x * config::tile_width + (warp_idx % config::subgrid_width) * config::subtile_width,
            group_index.y * config::tile_height + (warp_idx / config::subgrid_width) * config::subtile_height
        );
        const ushort4 subtile_bounds = make_ushort4(
            subtile_origin.x, // x_min
            subtile_origin.x + config::subtile_width, // x_max
            subtile_origin.y, // y_min
            subtile_origin.y + config::subtile_height // y_max
        );
        const uint2 pixel_coords = make_uint2(subtile_origin.x + lane_idx % config::subtile_width, subtile_origin.y + lane_idx / config::subtile_width);
        const bool inside = pixel_coords.x < width && pixel_coords.y < height;
        const float2 pixel = make_float2(__uint2float_rn(pixel_coords.x), __uint2float_rn(pixel_coords.y)) + 0.5f;
        // setup shared memory
        __shared__ ushort4 collected_screen_bounds[config::block_size_blend];
        __shared__ float2 collected_mean2d[config::block_size_blend];
        __shared__ float4 collected_conic_opacity[config::block_size_blend];
        __shared__ float3 collected_color[config::block_size_blend];
        // initialize local storage
        float3 color_pixel = make_float3(0.0f);
        float transmittance = 1.0f;
        bool done = !inside;
        // collaborative loading and processing
        const uint2 tile_range = tile_instance_ranges[group_index.y * grid_width + group_index.x];
        for (int n_primitives_remaining = tile_range.y - tile_range.x, current_fetch_idx = tile_range.x + thread_rank; n_primitives_remaining > 0; n_primitives_remaining -= config::block_size_blend, current_fetch_idx += config::block_size_blend) {
            if (__syncthreads_and(done)) break;
            if (current_fetch_idx < tile_range.y) {
                const uint primitive_idx = instance_primitive_indices[current_fetch_idx];
                collected_screen_bounds[thread_rank] = primitive_screen_bounds[primitive_idx];
                collected_mean2d[thread_rank] = primitive_mean2d[primitive_idx];
                collected_conic_opacity[thread_rank] = primitive_conic_opacity[primitive_idx];
                collected_color[thread_rank] = primitive_color[primitive_idx];
            }
            block.sync();
            const int current_batch_size = min(config::block_size_blend, n_primitives_remaining);
            for (int current_shmem_offset = 0; current_shmem_offset < current_batch_size; current_shmem_offset += 32) {
                // skip all Gaussians that do not overlap the subtile of this warp
                bool overlaps_subtile = false;
                if (current_shmem_offset + lane_idx < current_batch_size) {
                    const ushort4 screen_bounds = collected_screen_bounds[current_shmem_offset + lane_idx];
                    overlaps_subtile = screen_bounds.x < subtile_bounds.y && subtile_bounds.x < screen_bounds.y &&
                        screen_bounds.z < subtile_bounds.w && subtile_bounds.z < screen_bounds.w;
                }
                uint pending_primitives = warp.ballot(overlaps_subtile);
                while (!done && pending_primitives != 0u) {
                    const int j = current_shmem_offset + __ffs(static_cast<int>(pending_primitives)) - 1;
                    pending_primitives &= pending_primitives - 1;

                    // evaluate current Gaussian at pixel
                    const float4 conic_opacity = collected_conic_opacity[j];
                    const float3 conic = make_float3(conic_opacity);
                    const float opacity = conic_opacity.w;
                    const float2 delta = collected_mean2d[j] - pixel;
                    const float exponent = -0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) - conic.y * delta.x * delta.y;
                    const float gaussian = expf(fminf(exponent, 0.0f));
                    if (!config::original_opacity_interpretation && gaussian < config::min_alpha_threshold) continue;
                    const float alpha = opacity * gaussian;
                    if (config::original_opacity_interpretation && alpha < config::min_alpha_threshold) continue;

                    // blend fragment into pixel color
                    color_pixel += transmittance * alpha * collected_color[j];

                    // update transmittance
                    transmittance *= 1.0f - alpha;

                    // early stopping
                    if (transmittance < config::transmittance_threshold) done = true;
                }
            }
        }
        if (inside) {
            // apply background color
            color_pixel += transmittance * bg_color[0];
            // store results
            const uint pixel_idx = width * pixel_coords.y + pixel_coords.x;
            if (clamp_output) {
                color_pixel.x = __saturatef(color_pixel.x);
                color_pixel.y = __saturatef(color_pixel.y);
                color_pixel.z = __saturatef(color_pixel.z);
            }
            if (output_chw) {
                const uint n_pixels = width * height;
                image[pixel_idx] = color_pixel.x;
                image[n_pixels + pixel_idx] = color_pixel.y;
                image[2 * n_pixels + pixel_idx] = color_pixel.z;
            }
            else {
                const uint base_idx = 3 * pixel_idx;
                image[base_idx] = color_pixel.x;
                image[base_idx + 1] = color_pixel.y;
                image[base_idx + 2] = color_pixel.z;
            }
        }
    }

}
