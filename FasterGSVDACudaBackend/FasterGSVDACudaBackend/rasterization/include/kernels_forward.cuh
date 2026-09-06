#pragma once

#include "kernels_common.cuh"
#include "rasterization_config.h"
#include "helper_math.h"
#include "utils.h"
#include <cub/cub.cuh>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace faster_gs::rasterization::kernels::forward {

    __global__ void extract_bucket_counts(
        const uint2* __restrict__ tile_instance_ranges,
        uint* __restrict__ tile_n_buckets,
        const uint n_tiles)
    {
        const uint tile_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (tile_idx >= n_tiles) return;
        const uint2 instance_range = tile_instance_ranges[tile_idx];
        const uint n_buckets = div_round_up(instance_range.y - instance_range.x, 32u);
        tile_n_buckets[tile_idx] = n_buckets;
    }

    __global__ void __launch_bounds__(config::block_size_blend) blend_cu(
        const uint2* __restrict__ tile_instance_ranges,
        const uint* __restrict__ tile_buckets_offset,
        const uint* __restrict__ instance_primitive_indices,
        const ushort4* __restrict__ primitive_screen_bounds,
        const float2* __restrict__ primitive_mean2d,
        const float4* __restrict__ primitive_conic_opacity,
        const float3* __restrict__ primitive_color,
        const float3* __restrict__ bg_color,
        float* __restrict__ image,
        float* __restrict__ tile_final_transmittances,
        uint* __restrict__ tile_max_n_processed,
        uint* __restrict__ tile_n_processed,
        uint* __restrict__ bucket_tile_index,
        float4* __restrict__ bucket_color_transmittance,
        const uint width,
        const uint height,
        const uint grid_width)
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
        const uint local_idx = (pixel_coords.y - group_index.y * config::tile_height) * config::tile_width + (pixel_coords.x - group_index.x * config::tile_width);
        // setup tile info
        const uint tile_idx = group_index.y * grid_width + group_index.x;
        const uint2 tile_range = tile_instance_ranges[tile_idx];
        const int n_primitives_total = tile_range.y - tile_range.x;
        // setup bucket to tile mapping
        const int n_buckets = div_round_up(n_primitives_total, 32);
        uint bucket_offset = (tile_idx == 0) ? 0 : tile_buckets_offset[tile_idx - 1];
        for (int n_buckets_remaining = n_buckets, current_bucket_idx = thread_rank; n_buckets_remaining > 0; n_buckets_remaining -= config::block_size_blend, current_bucket_idx += config::block_size_blend) {
            if (current_bucket_idx < n_buckets) bucket_tile_index[bucket_offset + current_bucket_idx] = tile_idx;
        }
        // setup shared memory
        __shared__ ushort4 collected_screen_bounds[config::block_size_blend];
        __shared__ float2 collected_mean2d[config::block_size_blend];
        __shared__ float4 collected_conic_opacity[config::block_size_blend];
        __shared__ float3 collected_color[config::block_size_blend];
        // initialize local storage
        float3 color_pixel = make_float3(0.0f);
        float transmittance = 1.0f;
        uint n_processed_and_used = 0;
        bool done = !inside;
        // collaborative loading and processing
        for (int n_primitives_remaining = n_primitives_total, current_fetch_idx = tile_range.x + thread_rank; n_primitives_remaining > 0; n_primitives_remaining -= config::block_size_blend, current_fetch_idx += config::block_size_blend) {
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
            const uint n_processed_batch_start = n_primitives_total - n_primitives_remaining;
            for (int current_shmem_offset = 0; current_shmem_offset < current_batch_size; current_shmem_offset += 32) {
                // store current color and transmittance every 32 Gaussians
                if (!done) {
                    const float4 current_color_transmittance = make_float4(color_pixel, transmittance);
                    bucket_color_transmittance[bucket_offset * config::block_size_blend + local_idx] = current_color_transmittance;
                    bucket_offset++;
                }

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

                    // update the number of used Gaussians
                    n_processed_and_used = n_processed_batch_start + j + 1;

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
            const uint n_pixels = width * height;
            image[pixel_idx] = color_pixel.x;
            image[n_pixels + pixel_idx] = color_pixel.y;
            image[2 * n_pixels + pixel_idx] = color_pixel.z;
            tile_final_transmittances[pixel_idx] = transmittance;
            tile_n_processed[pixel_idx] = n_processed_and_used;
        }
        // max reduce the number of processed Gaussians per tile
        typedef cub::BlockReduce<uint, config::tile_width, cub::BLOCK_REDUCE_WARP_REDUCTIONS, config::tile_height> BlockReduce;
        __shared__ typename BlockReduce::TempStorage temp_storage;
        n_processed_and_used = BlockReduce(temp_storage).Reduce(n_processed_and_used, cub::Max());
        if (thread_rank == 0) tile_max_n_processed[tile_idx] = n_processed_and_used;
    }

}
