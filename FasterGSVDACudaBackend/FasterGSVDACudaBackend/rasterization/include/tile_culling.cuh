// note: this header is compiled in two environments: included by the aot-compiled kernels, and
// spliced verbatim into the jit-compiled preprocess kernel source (see appearance.cu), which
// nvrtc compiles with c++14; it must therefore avoid host-only includes outside the guard below,
// c++17 syntax, and float2/float3 operator helpers (the two environments define different sets)
#ifndef __CUDACC_RTC__
#pragma once
#include "rasterization_config.h"
#include <cstdint>
#endif

namespace faster_gs { namespace rasterization {

    __device__ inline uint4 convert_screen_bounds_to_tile_bounds(const ushort4& screen_bounds) {
        return make_uint4(
            screen_bounds.x / config::tile_width,
            (screen_bounds.y + config::tile_width - 1) / config::tile_width,
            screen_bounds.z / config::tile_height,
            (screen_bounds.w + config::tile_height - 1) / config::tile_height
        );
    }

    // based on https://github.com/r4dl/StopThePop-Rasterization/blob/d8cad09919ff49b11be3d693d1e71fa792f559bb/cuda_rasterizer/stopthepop/stopthepop_common.cuh#L131
    __device__ inline bool will_primitive_contribute(
        const float2& mean,
        const float3& conic,
        const uint32_t tile_x,
        const uint32_t tile_y,
        const float power_threshold)
    {
        const float2 rect_min = make_float2(static_cast<float>(tile_x * config::tile_width), static_cast<float>(tile_y * config::tile_height));
        const float2 rect_max = make_float2(static_cast<float>((tile_x + 1) * config::tile_width - 1), static_cast<float>((tile_y + 1) * config::tile_height - 1));

        const float x_min_diff = rect_min.x - mean.x;
        const float x_left = static_cast<float>(x_min_diff >= 0.0f);
        const float not_in_x_range = x_left + static_cast<float>(mean.x > rect_max.x);

        const float y_min_diff = rect_min.y - mean.y;
        const float y_above = static_cast<float>(y_min_diff >= 0.0f);
        const float not_in_y_range = y_above + static_cast<float>(mean.y > rect_max.y);

        if (not_in_y_range + not_in_x_range == 0.0f) return true;
        else {
            const float2 closest_corner = make_float2(
                rect_max.x + x_left * (rect_min.x - rect_max.x),
                rect_max.y + y_above * (rect_min.y - rect_max.y)
            );

            const float2 diff = make_float2(mean.x - closest_corner.x, mean.y - closest_corner.y);

            const float2 d = make_float2(
                copysignf(static_cast<float>(config::tile_width - 1), x_min_diff),
                copysignf(static_cast<float>(config::tile_height - 1), y_min_diff)
            );

            const float2 t = make_float2(
                not_in_y_range * __saturatef((d.x * conic.x * diff.x + d.x * conic.y * diff.y) / (d.x * conic.x * d.x)),
                not_in_x_range * __saturatef((d.y * conic.y * diff.x + d.y * conic.z * diff.y) / (d.y * conic.z * d.y))
            );

            const float2 max_contribution_point = make_float2(closest_corner.x + t.x * d.x, closest_corner.y + t.y * d.y);
            const float2 delta = make_float2(mean.x - max_contribution_point.x, mean.y - max_contribution_point.y);
            const float max_power_in_tile = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y;
            return max_power_in_tile <= power_threshold;
        }
    }

}}
