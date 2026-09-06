// jit-compiled forward preprocess kernel, shared by the training forward, optimized inference, and
// pruning score paths for every appearance model (JIT_APPEARANCE_MODEL: 0 = precomputed colors, 1 = sh,
// 2 = spherical voronoi, 3 = nasg, 4 = nasgabor, 5 = neural); the token preamble, preprocess_common.cuh, and the selected
// appearance file are prepended by the loader (plus, for the neural model, the tcnn-generated eval_model)
//
// the appearance tensors are bundled in an AppearanceInputs struct (see appearance_params.h) and
// appearance_degree is the appearance degree used for rendering (sh: active sh degree, neural:
// number of enabled direction degrees, nasgabor: number of active lobes, sv: number of active sites)
// JIT_KERNEL_NAME expands to preprocess_forward_{precomputed|sh|sv|nasgabor|neural}_cu so the compiled
// variants are distinguishable in profiler traces and the rtc ptx cache

__global__ void JIT_KERNEL_NAME(
    const float3* __restrict__ means,
    const float3* __restrict__ scales,
    const float4* __restrict__ rotations,
    const float* __restrict__ opacities,
    const faster_gs::AppearanceInputs appearance,
    const float4* __restrict__ w2c,
    const float3* __restrict__ cam_position,
    uint32_t* __restrict__ primitive_depth_keys,
    uint32_t* __restrict__ primitive_indices,
    uint32_t* __restrict__ primitive_n_touched_tiles,
    ushort4* __restrict__ primitive_screen_bounds,
    float2* __restrict__ primitive_mean2d,
    float4* __restrict__ primitive_conic_opacity,
    float3* __restrict__ primitive_color,
    uint32_t* __restrict__ n_visible_primitives,
    uint32_t* __restrict__ n_instances,
    const uint32_t n_primitives,
    const uint32_t appearance_degree,
    const uint32_t grid_width,
    const uint32_t grid_height,
    const float width,
    const float height,
    const float focal_x,
    const float focal_y,
    const float center_x,
    const float center_y,
    const float near_plane,
    const float far_plane,
    const bool proper_antialiasing,
    const bool render_base)
{
    using namespace faster_gs;
    constexpr uint32_t full_mask = 0xffffffff;
    const uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

#if JIT_APPEARANCE_MODEL == 5
    // advance the forward context pointer to this warp's segment (see tcnn's generate_kernel)
    uint8_t* fwd_ctx = appearance.fwd_ctx;
    if (fwd_ctx != nullptr) fwd_ctx += ((threadIdx.x / 32) * 32 + blockIdx.x * blockDim.x) * JIT_CTX_BYTES;
#endif

    bool active = true;
    uint32_t primitive_idx = thread_idx;
    if (primitive_idx >= n_primitives) {
        active = false;
        primitive_idx = n_primitives - 1;
    }

    if (active) primitive_n_touched_tiles[primitive_idx] = 0;

    // load 3d mean
    const float3 mean3d = means[primitive_idx];

    // z culling
    const float4 w2c_r3 = w2c[2];
    const float depth = w2c_r3.x * mean3d.x + w2c_r3.y * mean3d.y + w2c_r3.z * mean3d.z + w2c_r3.w;
    if (depth < near_plane || depth > far_plane) active = false;

    // early exit if whole warp is inactive
    if (__ballot_sync(full_mask, active) == 0) {
#if JIT_APPEARANCE_MODEL == 5
        if (fwd_ctx != nullptr) zero_warp_fwd_ctx(fwd_ctx);
#endif
        return;
    }

    // load opacity
    const float raw_opacity = opacities[primitive_idx];
    float opacity = sigmoid(raw_opacity);
    if (config::original_opacity_interpretation && opacity < config::min_alpha_threshold) active = false;

    // compute 3d covariance from scale and rotation
    const float3 raw_scale = scales[primitive_idx];
    const float3 variance = expf(2.0f * raw_scale);
    const float4 raw_rotation = rotations[primitive_idx];
    float quaternion_norm_sq = 1.0f;
    // the matrix types must stay qualified: tcnn's rtc preamble defines an ambiguous tcnn::mat3x3
    const faster_gs::mat3x3 R = convert_quaternion_to_rotation_matrix(raw_rotation, quaternion_norm_sq);
    if (quaternion_norm_sq < 1e-8f) active = false;
    const faster_gs::mat3x3 RSS = {
        R.m11 * variance.x, R.m12 * variance.y, R.m13 * variance.z,
        R.m21 * variance.x, R.m22 * variance.y, R.m23 * variance.z,
        R.m31 * variance.x, R.m32 * variance.y, R.m33 * variance.z
    };
    const faster_gs::mat3x3_triu cov3d {
        RSS.m11 * R.m11 + RSS.m12 * R.m12 + RSS.m13 * R.m13,
        RSS.m11 * R.m21 + RSS.m12 * R.m22 + RSS.m13 * R.m23,
        RSS.m11 * R.m31 + RSS.m12 * R.m32 + RSS.m13 * R.m33,
        RSS.m21 * R.m21 + RSS.m22 * R.m22 + RSS.m23 * R.m23,
        RSS.m21 * R.m31 + RSS.m22 * R.m32 + RSS.m23 * R.m33,
        RSS.m31 * R.m31 + RSS.m32 * R.m32 + RSS.m33 * R.m33,
    };

    // compute 2d mean in normalized image coordinates
    const float4 w2c_r1 = w2c[0];
    const float x = (w2c_r1.x * mean3d.x + w2c_r1.y * mean3d.y + w2c_r1.z * mean3d.z + w2c_r1.w) / depth;
    const float4 w2c_r2 = w2c[1];
    const float y = (w2c_r2.x * mean3d.x + w2c_r2.y * mean3d.y + w2c_r2.z * mean3d.z + w2c_r2.w) / depth;

    // ewa splatting
    const float clip_left = (-0.15f * width - center_x) / focal_x;
    const float clip_right = (1.15f * width - center_x) / focal_x;
    const float clip_top = (-0.15f * height - center_y) / focal_y;
    const float clip_bottom = (1.15f * height - center_y) / focal_y;
    const float x_clipped = clamp(x, clip_left, clip_right);
    const float y_clipped = clamp(y, clip_top, clip_bottom);
    const float j11 = focal_x / depth;
    const float j13 = -j11 * x_clipped;
    const float j22 = focal_y / depth;
    const float j23 = -j22 * y_clipped;
    const float3 jw_r1 = ::make_float3(
        j11 * w2c_r1.x + j13 * w2c_r3.x,
        j11 * w2c_r1.y + j13 * w2c_r3.y,
        j11 * w2c_r1.z + j13 * w2c_r3.z
    );
    const float3 jw_r2 = ::make_float3(
        j22 * w2c_r2.x + j23 * w2c_r3.x,
        j22 * w2c_r2.y + j23 * w2c_r3.y,
        j22 * w2c_r2.z + j23 * w2c_r3.z
    );
    const float3 jwc_r1 = ::make_float3(
        jw_r1.x * cov3d.m11 + jw_r1.y * cov3d.m12 + jw_r1.z * cov3d.m13,
        jw_r1.x * cov3d.m12 + jw_r1.y * cov3d.m22 + jw_r1.z * cov3d.m23,
        jw_r1.x * cov3d.m13 + jw_r1.y * cov3d.m23 + jw_r1.z * cov3d.m33
    );
    const float3 jwc_r2 = ::make_float3(
        jw_r2.x * cov3d.m11 + jw_r2.y * cov3d.m12 + jw_r2.z * cov3d.m13,
        jw_r2.x * cov3d.m12 + jw_r2.y * cov3d.m22 + jw_r2.z * cov3d.m23,
        jw_r2.x * cov3d.m13 + jw_r2.y * cov3d.m23 + jw_r2.z * cov3d.m33
    );
    float3 cov2d = ::make_float3(
        dot(jwc_r1, jw_r1),
        dot(jwc_r1, jw_r2),
        dot(jwc_r2, jw_r2)
    );
    const float determinant_raw = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    const float kernel_size = proper_antialiasing ? config::dilation_proper_antialiasing : config::dilation;
    cov2d.x += kernel_size;
    cov2d.z += kernel_size;
    const float determinant = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    if (determinant < config::min_cov2d_determinant) active = false;
    const float3 conic = ::make_float3(
        cov2d.z / determinant,
        -cov2d.y / determinant,
        cov2d.x / determinant
    );
    if (proper_antialiasing) {
        opacity *= sqrtf(fmaxf(determinant_raw / determinant, 0.0f));
        if (config::original_opacity_interpretation && opacity < config::min_alpha_threshold) active = false;
    }

    // 2d mean in screen space
    const float2 mean2d = make_float2(
        x * focal_x + center_x,
        y * focal_y + center_y
    );

    // compute bounds
    const float power_threshold = config::original_opacity_interpretation ? logf(opacity * config::min_alpha_threshold_rcp) : config::max_power_threshold;
    const float cutoff_factor = 2.0f * power_threshold;
    const float extent_x = fmaxf(sqrtf(cov2d.x * cutoff_factor) - 0.5f, 0.0f);
    const float extent_y = fmaxf(sqrtf(cov2d.z * cutoff_factor) - 0.5f, 0.0f);
    // clamp to the tile grid extent, not the image size, so fully off-screen primitives map to zero tiles instead of the last (partially visible) tile row/column
    const int padded_width = static_cast<int>(grid_width * config::tile_width);
    const int padded_height = static_cast<int>(grid_height * config::tile_height);
    const ushort4 screen_bounds = make_ushort4(
        min(padded_width, max(0, __float2int_rd(mean2d.x - extent_x))), // x_min
        min(padded_width, max(0, __float2int_ru(mean2d.x + extent_x))), // x_max
        min(padded_height, max(0, __float2int_rd(mean2d.y - extent_y))), // y_min
        min(padded_height, max(0, __float2int_ru(mean2d.y + extent_y))) // y_max
    );
    const uint4 tile_bounds = convert_screen_bounds_to_tile_bounds(screen_bounds);
    const uint32_t n_touched_tiles_max = (tile_bounds.y - tile_bounds.x) * (tile_bounds.w - tile_bounds.z);
    if (n_touched_tiles_max == 0) active = false;

    // early exit if whole warp is inactive
    if (__ballot_sync(full_mask, active) == 0) {
#if JIT_APPEARANCE_MODEL == 5
        if (fwd_ctx != nullptr) zero_warp_fwd_ctx(fwd_ctx);
#endif
        return;
    }

    // compute exact number of tiles the primitive overlaps
    const uint32_t n_touched_tiles = compute_exact_n_touched_tiles(
        mean2d, conic, tile_bounds,
        power_threshold, n_touched_tiles_max, active
    );

#if JIT_APPEARANCE_MODEL == 5
    // residual mlp evaluation where all lanes of surviving warps must participate (warp-cooperative mma)
    tvec<__half, 3> mlp_output;
    const float3 residual = neural_residual(
        appearance,
        mean3d, cam_position[0],
        fwd_ctx,
        mlp_output,
        primitive_idx,
        appearance_degree
    );
#endif

    // cooperative threads no longer needed
    if (n_touched_tiles == 0 || !active) return;

    // store results
    primitive_n_touched_tiles[primitive_idx] = n_touched_tiles;
    primitive_screen_bounds[primitive_idx] = screen_bounds;
    primitive_mean2d[primitive_idx] = mean2d;
    primitive_conic_opacity[primitive_idx] = make_float4(conic, opacity);

#if JIT_APPEARANCE_MODEL == 0
    primitive_color[primitive_idx] = appearance.precomputed_colors[primitive_idx];
#else
#if JIT_APPEARANCE_MODEL == 1
    const float3 residual = sh_residual(
        reinterpret_cast<const float3*>(appearance.residual_params),
        mean3d, cam_position[0],
        primitive_idx, appearance_degree
    );
#elif JIT_APPEARANCE_MODEL == 2
    const float3 residual = sv_residual(
        appearance.residual_params,
        mean3d, cam_position[0],
        primitive_idx, appearance_degree
    );
#elif JIT_APPEARANCE_MODEL == 3
    const float3 residual = nasg_residual(
        appearance.residual_params,
        mean3d, cam_position[0],
        primitive_idx, appearance_degree
    );
#elif JIT_APPEARANCE_MODEL == 4
    const float3 residual = nasgabor_residual(
        appearance.residual_params,
        mean3d, cam_position[0],
        primitive_idx, appearance_degree
    );
#elif JIT_APPEARANCE_MODEL == 5
#if JIT_RESIDUAL_ACTIVATION != 0
    // store raw mlp outputs for the backward pass
    if (appearance.mlp_outputs != nullptr && appearance_degree > 0) {
        appearance.mlp_outputs[primitive_idx] = mlp_output[0];
        appearance.mlp_outputs[n_primitives + primitive_idx] = mlp_output[1];
        appearance.mlp_outputs[2 * n_primitives + primitive_idx] = mlp_output[2];
    }
#endif
#endif
    const float3 base = base_activation(appearance.base_colors[primitive_idx]);
    float3 color = color_activation(base + residual);
    if (!render_base) {
        const float3 base_color = color_activation(base);
        color = ::make_float3(
            fabsf(color.x - base_color.x),
            fabsf(color.y - base_color.y),
            fabsf(color.z - base_color.z)
        );
    }
    primitive_color[primitive_idx] = color;
#endif

    const uint32_t offset = atomicAdd(n_visible_primitives, 1);
    const uint32_t depth_key = __float_as_uint(depth);
    primitive_depth_keys[offset] = depth_key;
    primitive_indices[offset] = primitive_idx;
    atomicAdd(n_instances, n_touched_tiles);
}
