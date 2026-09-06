#pragma once

#include "rasterization_config.h"
#include "tile_culling.cuh"
#include "helper_math.h"
#include "utils.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// instance pipeline shared by the training forward, optimized inference, and pruning score paths
namespace faster_gs::rasterization::kernels {

    // internal linkage: a non-template __global__ definition in a header would collide at link time
    static __global__ void apply_depth_ordering_cu(
        const uint* __restrict__ primitive_indices_sorted,
        const uint* __restrict__ primitive_n_touched_tiles,
        uint* __restrict__ primitive_offset,
        const uint n_visible_primitives)
    {
        const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_visible_primitives) return;
        const uint primitive_idx = primitive_indices_sorted[idx];
        primitive_offset[idx] = primitive_n_touched_tiles[primitive_idx];
    }

    // based on https://github.com/r4dl/StopThePop-Rasterization/blob/d8cad09919ff49b11be3d693d1e71fa792f559bb/cuda_rasterizer/stopthepop/stopthepop_common.cuh#L325
    template <typename KeyT>
    __global__ void create_instances_cu(
        const uint* __restrict__ primitive_indices_sorted,
        const uint* __restrict__ primitive_offsets,
        const ushort4* __restrict__ primitive_screen_bounds,
        const float2* __restrict__ primitive_mean2d,
        const float4* __restrict__ primitive_conic_opacity,
        KeyT* __restrict__ instance_keys,
        uint* __restrict__ instance_primitive_indices,
        const uint grid_width,
        const uint n_visible_primitives)
    {
        constexpr uint warp_size = 32;
        auto block = cg::this_thread_block();
        auto warp = cg::tiled_partition<warp_size>(block);
        const uint thread_idx = cg::this_grid().thread_rank();
        const uint thread_rank = block.thread_rank();
        const uint warp_idx = warp.meta_group_rank();
        const uint warp_start = warp_idx * warp_size;
        const uint lane_idx = warp.thread_rank();
        const uint previous_lanes_mask = (1 << lane_idx) - 1;

        uint original_idx = thread_idx;
        bool active = true;
        if (original_idx >= n_visible_primitives) {
            active = false;
            original_idx = n_visible_primitives - 1;
        }

        if (warp.ballot(active) == 0) return;

        const uint primitive_idx = primitive_indices_sorted[original_idx];

        const uint4 tile_bounds = convert_screen_bounds_to_tile_bounds(primitive_screen_bounds[primitive_idx]);
        const uint tile_bounds_width = tile_bounds.y - tile_bounds.x;
        const uint instance_count = (tile_bounds.w - tile_bounds.z) * tile_bounds_width;
        const float2 mean2d = primitive_mean2d[primitive_idx];
        const float2 mean2d_shifted = mean2d - 0.5f;
        const float4 conic_opacity = primitive_conic_opacity[primitive_idx];
        const float3 conic = make_float3(conic_opacity);
        const float opacity = conic_opacity.w;
        const float power_threshold = config::original_opacity_interpretation ? logf(opacity * config::min_alpha_threshold_rcp) : config::max_power_threshold;

        uint current_write_offset = primitive_offsets[original_idx];

        for (uint instance_idx = 0; active && instance_idx < instance_count && instance_idx < config::n_sequential_threshold; instance_idx++) {
            const uint tile_x = tile_bounds.x + (instance_idx % tile_bounds_width);
            const uint tile_y = tile_bounds.z + (instance_idx / tile_bounds_width);
            if (will_primitive_contribute(mean2d_shifted, conic, tile_x, tile_y, power_threshold)) {
                const uint tile_idx = tile_y * grid_width + tile_x;
                const KeyT instance_key = static_cast<KeyT>(tile_idx);
                instance_keys[current_write_offset] = instance_key;
                instance_primitive_indices[current_write_offset] = primitive_idx;
                current_write_offset++;
            }
        }

        const bool compute_cooperatively = active && instance_count > config::n_sequential_threshold;
        const uint remaining_threads = warp.ballot(compute_cooperatively);
        if (remaining_threads == 0) return;

        __shared__ ushort4 collected_tile_bounds[config::block_size_create_instances];
        __shared__ float2 collected_mean2d_shifted[config::block_size_create_instances];
        __shared__ float4 collected_conic_power_threshold[config::block_size_create_instances];
        collected_tile_bounds[thread_rank] = make_ushort4(tile_bounds.x, tile_bounds.y, tile_bounds.z, tile_bounds.w);
        collected_mean2d_shifted[thread_rank] = mean2d_shifted;
        collected_conic_power_threshold[thread_rank] = make_float4(conic, power_threshold);

        const uint n_remaining_threads = __popc(remaining_threads);
        for (uint n = 0; n < n_remaining_threads && n < warp_size; n++) {
            const uint current_lane = __fns(remaining_threads, 0, n + 1);
            const uint primitive_idx_coop = warp.shfl(primitive_idx, current_lane);
            uint current_write_offset_coop = warp.shfl(current_write_offset, current_lane);

            const uint read_offset_shared = warp_start + current_lane;
            const ushort4 tile_bounds_coop = collected_tile_bounds[read_offset_shared];
            const float2 mean2d_shifted_coop = collected_mean2d_shifted[read_offset_shared];
            const float4 conic_power_threshold_coop = collected_conic_power_threshold[read_offset_shared];

            const uint tile_bounds_width_coop = static_cast<uint>(tile_bounds_coop.y - tile_bounds_coop.x);
            const uint instance_count_coop = tile_bounds_width_coop * static_cast<uint>(tile_bounds_coop.w - tile_bounds_coop.z);
            const float3 conic_coop = make_float3(conic_power_threshold_coop);
            const float power_threshold_coop = conic_power_threshold_coop.w;

            const uint remaining_instance_count = instance_count_coop - config::n_sequential_threshold;
            const uint n_iterations = div_round_up(remaining_instance_count, warp_size);
            for (uint i = 0; i < n_iterations; i++) {
                const uint instance_idx = i * warp_size + lane_idx + config::n_sequential_threshold;
                const uint tile_x = tile_bounds_coop.x + (instance_idx % tile_bounds_width_coop);
                const uint tile_y = tile_bounds_coop.z + (instance_idx / tile_bounds_width_coop);
                const bool write = instance_idx < instance_count_coop && will_primitive_contribute(mean2d_shifted_coop, conic_coop, tile_x, tile_y, power_threshold_coop);
                const uint write_ballot = warp.ballot(write);
                if (write) {
                    const uint write_offset = current_write_offset_coop + __popc(write_ballot & previous_lanes_mask);
                    const uint tile_idx = tile_y * grid_width + tile_x;
                    const KeyT instance_key = static_cast<KeyT>(tile_idx);
                    instance_keys[write_offset] = instance_key;
                    instance_primitive_indices[write_offset] = primitive_idx_coop;
                }
                const uint n_written = __popc(write_ballot);
                current_write_offset_coop += n_written;
            }
            warp.sync();
        }
    }

    template <typename KeyT>
    __global__ void extract_instance_ranges_cu(
        const KeyT* __restrict__ instance_keys,
        uint2* __restrict__ tile_instance_ranges,
        const uint n_instances)
    {
        const uint instance_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (instance_idx >= n_instances) return;
        const KeyT instance_tile_idx = instance_keys[instance_idx];
        if (instance_idx == 0) tile_instance_ranges[instance_tile_idx].x = 0;
        else {
            const KeyT previous_instance_tile_idx = instance_keys[instance_idx - 1];
            if (instance_tile_idx != previous_instance_tile_idx) {
                tile_instance_ranges[previous_instance_tile_idx].y = instance_idx;
                tile_instance_ranges[instance_tile_idx].x = instance_idx;
            }
        }
        if (instance_idx == n_instances - 1) tile_instance_ranges[instance_tile_idx].y = n_instances;
    }

}
