// jit-compiled backward preprocess kernel for every appearance model (JIT_APPEARANCE_MODEL: 0 = precomputed
// colors, 1 = sh, 2 = spherical voronoi, 3 = nasg, 4 = nasgabor, 5 = neural); the token preamble,
// preprocess_common.cuh, and the selected appearance file are prepended by the loader (plus, for the
// neural model, the tcnn-generated eval_model and backward_eval_model)
//
// the blend backward accumulates the raw gradient w.r.t. the activated per-primitive color into
// grad_color; this kernel recovers the activation derivative from the stored (post-activation)
// primitive color and backpropagates through the appearance model (for precomputed colors the raw
// gradient is returned to python untouched, so no color work happens here)
//
// neural only: backward_eval_model consumes the forward context written by the forward preprocess kernel,
// accumulates loss-scaled parameter gradients via block-wide reduction + atomics (requires the
// dynamic shared memory the launcher passes), and returns input gradients via dL_dx — no early
// exits are allowed before it
//
// JIT_KERNEL_NAME expands to preprocess_backward_{precomputed|sh|sv|nasgabor|neural}_cu so the compiled
// variants are distinguishable in profiler traces and the rtc ptx cache

__global__ void JIT_KERNEL_NAME(
    const float3* __restrict__ means,
    const float3* __restrict__ scales,
    const float4* __restrict__ rotations,
    const float* __restrict__ opacities,
    const faster_gs::AppearanceInputs appearance,
    const float4* __restrict__ w2c,
    const float3* __restrict__ cam_position,
    const uint32_t* __restrict__ primitive_n_touched_tiles,
    const float3* __restrict__ primitive_color,
    const float2* __restrict__ grad_mean2d,
    const float* __restrict__ grad_conic,
    const float* __restrict__ grad_color,
    float3* __restrict__ grad_means,
    float3* __restrict__ grad_scales,
    float4* __restrict__ grad_rotations,
    float* __restrict__ grad_opacities,
    const faster_gs::AppearanceGradients grad_appearance,
    float* __restrict__ densification_info,
    const uint32_t n_primitives,
    const uint32_t appearance_degree,
    const float width,
    const float height,
    const float focal_x,
    const float focal_y,
    const float center_x,
    const float center_y,
    const bool proper_antialiasing,
    const bool render_base)
{
    using namespace faster_gs;
    const uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

#if JIT_APPEARANCE_MODEL == 5
    // advance the forward context pointer to this warp's segment (must match preprocess_forward_cu)
    const uint8_t* fwd_ctx = appearance.fwd_ctx + ((threadIdx.x / 32) * 32 + blockIdx.x * blockDim.x) * JIT_CTX_BYTES;

    // no early exits are allowed before the mlp backward: its parameter gradient reduction is block-wide,
    // so every thread of the block must reach it (non-contributing lanes carry zero gradients, and warps
    // culled in the forward pass read the forward context segment they zeroed themselves before exiting)
    uint32_t primitive_idx = thread_idx;
    bool contributes = primitive_idx < n_primitives;
    if (!contributes) primitive_idx = n_primitives - 1;
    contributes = contributes && primitive_n_touched_tiles[primitive_idx] > 0;
#else
    const uint32_t primitive_idx = thread_idx;
    if (primitive_idx >= n_primitives || primitive_n_touched_tiles[primitive_idx] == 0) return;
    constexpr bool contributes = true;
#endif

    // raw gradient w.r.t. the activated primitive color, as accumulated by the blend backward
    const float3 dL_dcolor = contributes ? ::make_float3(grad_color[primitive_idx], grad_color[n_primitives + primitive_idx], grad_color[2 * n_primitives + primitive_idx]) : ::make_float3(0.0f, 0.0f, 0.0f);
#if JIT_APPEARANCE_MODEL != 0
    // the color activation derivative is recovered from the stored (post-activation) primitive color
    const float3 color = contributes ? primitive_color[primitive_idx] : ::make_float3(0.0f, 0.0f, 0.0f);
    const float3 dL_dpre = color_activation_grad(color) * dL_dcolor;

    // shared base gradient
    if (contributes && render_base) {
        grad_appearance.base_colors[primitive_idx] = base_activation_grad(appearance.base_colors[primitive_idx]) * dL_dpre;
    }
#endif

    // load 3d mean
    const float3 mean3d = means[primitive_idx];

    float3 dL_dmean3d_from_color = ::make_float3(0.0f, 0.0f, 0.0f);
#if JIT_APPEARANCE_MODEL == 0
    grad_appearance.precomputed_colors[primitive_idx] = dL_dcolor;
#elif JIT_APPEARANCE_MODEL == 1
    dL_dmean3d_from_color = sh_residual_backward(
        reinterpret_cast<const float3*>(appearance.residual_params),
        dL_dpre,
        reinterpret_cast<float3*>(grad_appearance.residual_params),
        mean3d, cam_position[0], primitive_idx,
        appearance_degree
    );
#elif JIT_APPEARANCE_MODEL == 2
    dL_dmean3d_from_color = sv_residual_backward(
        appearance.residual_params,
        dL_dpre,
        grad_appearance.residual_params,
        mean3d, cam_position[0], primitive_idx,
        appearance_degree
    );
#elif JIT_APPEARANCE_MODEL == 3
    dL_dmean3d_from_color = nasg_residual_backward(
        appearance.residual_params,
        dL_dpre,
        grad_appearance.residual_params,
        mean3d, cam_position[0], primitive_idx,
        appearance_degree
    );
#elif JIT_APPEARANCE_MODEL == 4
    dL_dmean3d_from_color = nasgabor_residual_backward(
        appearance.residual_params,
        dL_dpre,
        grad_appearance.residual_params,
        mean3d, cam_position[0], primitive_idx,
        appearance_degree
    );
#elif JIT_APPEARANCE_MODEL == 5
    dL_dmean3d_from_color = neural_residual_backward(
        appearance,
        dL_dpre,
        grad_appearance,
        mean3d, cam_position[0], fwd_ctx,
        primitive_idx, n_primitives,
        appearance_degree,
        contributes
    );
    if (!contributes) return;
#endif

    const float4 w2c_r3 = w2c[2];
    const float depth = w2c_r3.x * mean3d.x + w2c_r3.y * mean3d.y + w2c_r3.z * mean3d.z + w2c_r3.w;
    const float4 w2c_r1 = w2c[0];
    const float x = (w2c_r1.x * mean3d.x + w2c_r1.y * mean3d.y + w2c_r1.z * mean3d.z + w2c_r1.w) / depth;
    const float4 w2c_r2 = w2c[1];
    const float y = (w2c_r2.x * mean3d.x + w2c_r2.y * mean3d.y + w2c_r2.z * mean3d.z + w2c_r2.w) / depth;

    // compute 3d covariance from scale and rotation
    const float3 raw_scale = scales[primitive_idx];
    const float3 variance = expf(2.0f * raw_scale);
    const float4 raw_rotation = rotations[primitive_idx];
    float quaternion_norm_sq = 1.0f;
    const faster_gs::mat3x3 R = convert_quaternion_to_rotation_matrix(raw_rotation, quaternion_norm_sq);
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

    // ewa splatting gradient helpers
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

    // 2d covariance gradient
    const float a_raw = dot(jwc_r1, jw_r1), b = dot(jwc_r1, jw_r2), c_raw = dot(jwc_r2, jw_r2);
    const float kernel_size = proper_antialiasing ? config::dilation_proper_antialiasing : config::dilation;
    const float a = a_raw + kernel_size, c = c_raw + kernel_size;
    const float aa = a * a, bb = b * b, cc = c * c;
    const float ac = a * c, ab = a * b, bc = b * c;
    const float determinant = ac - bb;
    const float determinant_sq = determinant * determinant;
    const float determinant_rcp_sq = 1.0f / determinant_sq;
    const float3 dL_dconic = ::make_float3(
        grad_conic[primitive_idx],
        grad_conic[n_primitives + primitive_idx],
        grad_conic[2 * n_primitives + primitive_idx]
    );
    float3 dL_dcov2d = determinant_rcp_sq * ::make_float3(
        2.0f * bc * dL_dconic.y - cc * dL_dconic.x - bb * dL_dconic.z,
        bc * dL_dconic.x - (ac + bb) * dL_dconic.y + ab * dL_dconic.z,
        2.0f * ab * dL_dconic.y - bb * dL_dconic.x - aa * dL_dconic.z
    );

    // account for proper antialiasing
    if (proper_antialiasing) {
        const float opacity = sigmoid(opacities[primitive_idx]);
        const float dL_dopacity_conv_factor = grad_opacities[primitive_idx];
        const float determinant_raw = a_raw * c_raw - bb;
        const float radicand = fmaxf(determinant_raw / determinant, 0.0f);
        const float conv_factor = sqrtf(radicand);
        const float dL_dopacity = dL_dopacity_conv_factor * conv_factor * opacity * (1.0f - opacity);
        grad_opacities[primitive_idx] = dL_dopacity;
        // the remaining part works but causes exploding gradients that lead to lots of degenerate Gaussians
        if (!config::detach_dilation_proper_antialiasing_from_cov2d) {
            // based on https://github.com/nerfstudio-project/gsplat/blob/65042cc501d1cdbefaf1d6f61a9a47575eec8c71/gsplat/cuda/include/Utils.cuh#L390
            const float3 conic = ::make_float3(
                c / determinant,
                -b / determinant,
                a / determinant
            );
            const float determinant_conic = conic.x * conic.z - conic.y * conic.y;
            const float dL_dradicand = dL_dopacity_conv_factor * opacity / fmaxf(2.0f * conv_factor, 1e-6f);
            const float one_minus_radicand = 1.0f - radicand;
            dL_dcov2d.x += dL_dradicand * (one_minus_radicand * conic.x - kernel_size * determinant_conic);
            dL_dcov2d.y += dL_dradicand * one_minus_radicand * conic.y;
            dL_dcov2d.z += dL_dradicand * (one_minus_radicand * conic.z - kernel_size * determinant_conic);
        }
    }

    // 3d covariance gradient
    const faster_gs::mat3x3_triu dL_dcov3d = {
        jw_r1.x * jw_r1.x * dL_dcov2d.x + 2.0f * jw_r1.x * jw_r2.x * dL_dcov2d.y + jw_r2.x * jw_r2.x * dL_dcov2d.z,
        jw_r1.x * jw_r1.y * dL_dcov2d.x + (jw_r1.x * jw_r2.y + jw_r1.y * jw_r2.x) * dL_dcov2d.y + jw_r2.x * jw_r2.y * dL_dcov2d.z,
        jw_r1.x * jw_r1.z * dL_dcov2d.x + (jw_r1.x * jw_r2.z + jw_r1.z * jw_r2.x) * dL_dcov2d.y + jw_r2.x * jw_r2.z * dL_dcov2d.z,
        jw_r1.y * jw_r1.y * dL_dcov2d.x + 2.0f * jw_r1.y * jw_r2.y * dL_dcov2d.y + jw_r2.y * jw_r2.y * dL_dcov2d.z,
        jw_r1.y * jw_r1.z * dL_dcov2d.x + (jw_r1.y * jw_r2.z + jw_r1.z * jw_r2.y) * dL_dcov2d.y + jw_r2.y * jw_r2.z * dL_dcov2d.z,
        jw_r1.z * jw_r1.z * dL_dcov2d.x + 2.0f * jw_r1.z * jw_r2.z * dL_dcov2d.y + jw_r2.z * jw_r2.z * dL_dcov2d.z,
    };

    // gradient of J * W
    const float3 dL_djw_r1 = 2.0f * ::make_float3(
        jwc_r1.x * dL_dcov2d.x + jwc_r2.x * dL_dcov2d.y,
        jwc_r1.y * dL_dcov2d.x + jwc_r2.y * dL_dcov2d.y,
        jwc_r1.z * dL_dcov2d.x + jwc_r2.z * dL_dcov2d.y
    );
    const float3 dL_djw_r2 = 2.0f * ::make_float3(
        jwc_r1.x * dL_dcov2d.y + jwc_r2.x * dL_dcov2d.z,
        jwc_r1.y * dL_dcov2d.y + jwc_r2.y * dL_dcov2d.z,
        jwc_r1.z * dL_dcov2d.y + jwc_r2.z * dL_dcov2d.z
    );

    // gradient of non-zero entries in J
    const float dL_dj11 = w2c_r1.x * dL_djw_r1.x + w2c_r1.y * dL_djw_r1.y + w2c_r1.z * dL_djw_r1.z;
    const float dL_dj22 = w2c_r2.x * dL_djw_r2.x + w2c_r2.y * dL_djw_r2.y + w2c_r2.z * dL_djw_r2.z;
    const float dL_dj13 = w2c_r3.x * dL_djw_r1.x + w2c_r3.y * dL_djw_r1.y + w2c_r3.z * dL_djw_r1.z;
    const float dL_dj23 = w2c_r3.x * dL_djw_r2.x + w2c_r3.y * dL_djw_r2.y + w2c_r3.z * dL_djw_r2.z;

    // load gradient of 2d mean
    const float2 dL_dmean2d = grad_mean2d[primitive_idx];

    // for adaptive density control
    if (densification_info != nullptr) {
        densification_info[primitive_idx] += 1.0f;
        const float2 dL_dmean2d_ndc = 0.5f * make_float2(
            dL_dmean2d.x * width,
            dL_dmean2d.y * height
        );
        densification_info[n_primitives + primitive_idx] += norm(dL_dmean2d_ndc);
    }

    // mean3d camera space gradient from mean2d
    float3 dL_dmean3d_cam = ::make_float3(
        j11 * dL_dmean2d.x,
        j22 * dL_dmean2d.y,
        -j11 * x * dL_dmean2d.x - j22 * y * dL_dmean2d.y
    );

    // add mean3d camera space gradient from J while accounting for clipping
    const bool valid_x = x >= clip_left && x <= clip_right;
    const bool valid_y = y >= clip_top && y <= clip_bottom;
    if (valid_x) dL_dmean3d_cam.x -= j11 * dL_dj13 / depth;
    if (valid_y) dL_dmean3d_cam.y -= j22 * dL_dj23 / depth;
    const float factor_x = 1.0f + static_cast<float>(valid_x);
    const float factor_y = 1.0f + static_cast<float>(valid_y);
    dL_dmean3d_cam.z += (j11 * (factor_x * x_clipped * dL_dj13 - dL_dj11) + j22 * (factor_y * y_clipped * dL_dj23 - dL_dj22)) / depth;

    // 3d mean gradient from splatting
    const float3 dL_dmean3d_from_splatting = ::make_float3(
        w2c_r1.x * dL_dmean3d_cam.x + w2c_r2.x * dL_dmean3d_cam.y + w2c_r3.x * dL_dmean3d_cam.z,
        w2c_r1.y * dL_dmean3d_cam.x + w2c_r2.y * dL_dmean3d_cam.y + w2c_r3.y * dL_dmean3d_cam.z,
        w2c_r1.z * dL_dmean3d_cam.x + w2c_r2.z * dL_dmean3d_cam.y + w2c_r3.z * dL_dmean3d_cam.z
    );

    // write total 3d mean gradient; sole JIT_DIRECTION_GRADIENT gate: with the token disabled, the
    // inlined direction-gradient computation (and loads feeding only it) is dead-code eliminated
    const float3 dL_dmean3d = JIT_DIRECTION_GRADIENT ? dL_dmean3d_from_splatting + dL_dmean3d_from_color : dL_dmean3d_from_splatting;
    grad_means[primitive_idx] = dL_dmean3d;

    // scale gradient
    const float3 dL_dvariance = ::make_float3(
        R.m11 * R.m11 * dL_dcov3d.m11 + R.m21 * R.m21 * dL_dcov3d.m22 + R.m31 * R.m31 * dL_dcov3d.m33 +
            2.0f * (R.m11 * R.m21 * dL_dcov3d.m12 + R.m11 * R.m31 * dL_dcov3d.m13 + R.m21 * R.m31 * dL_dcov3d.m23),
        R.m12 * R.m12 * dL_dcov3d.m11 + R.m22 * R.m22 * dL_dcov3d.m22 + R.m32 * R.m32 * dL_dcov3d.m33 +
            2.0f * (R.m12 * R.m22 * dL_dcov3d.m12 + R.m12 * R.m32 * dL_dcov3d.m13 + R.m22 * R.m32 * dL_dcov3d.m23),
        R.m13 * R.m13 * dL_dcov3d.m11 + R.m23 * R.m23 * dL_dcov3d.m22 + R.m33 * R.m33 * dL_dcov3d.m33 +
            2.0f * (R.m13 * R.m23 * dL_dcov3d.m12 + R.m13 * R.m33 * dL_dcov3d.m13 + R.m23 * R.m33 * dL_dcov3d.m23)
    );
    const float3 dL_dscale = 2.0f * variance * dL_dvariance;
    grad_scales[primitive_idx] = dL_dscale;

    // rotation gradient
    const faster_gs::mat3x3 dL_dR = {
        2.0f * (RSS.m11 * dL_dcov3d.m11 + RSS.m21 * dL_dcov3d.m12 + RSS.m31 * dL_dcov3d.m13),
        2.0f * (RSS.m12 * dL_dcov3d.m11 + RSS.m22 * dL_dcov3d.m12 + RSS.m32 * dL_dcov3d.m13),
        2.0f * (RSS.m13 * dL_dcov3d.m11 + RSS.m23 * dL_dcov3d.m12 + RSS.m33 * dL_dcov3d.m13),
        2.0f * (RSS.m11 * dL_dcov3d.m12 + RSS.m21 * dL_dcov3d.m22 + RSS.m31 * dL_dcov3d.m23),
        2.0f * (RSS.m12 * dL_dcov3d.m12 + RSS.m22 * dL_dcov3d.m22 + RSS.m32 * dL_dcov3d.m23),
        2.0f * (RSS.m13 * dL_dcov3d.m12 + RSS.m23 * dL_dcov3d.m22 + RSS.m33 * dL_dcov3d.m23),
        2.0f * (RSS.m11 * dL_dcov3d.m13 + RSS.m21 * dL_dcov3d.m23 + RSS.m31 * dL_dcov3d.m33),
        2.0f * (RSS.m12 * dL_dcov3d.m13 + RSS.m22 * dL_dcov3d.m23 + RSS.m32 * dL_dcov3d.m33),
        2.0f * (RSS.m13 * dL_dcov3d.m13 + RSS.m23 * dL_dcov3d.m23 + RSS.m33 * dL_dcov3d.m33)
    };
    const float4 dL_drotation = convert_quaternion_to_rotation_matrix_backward(raw_rotation, dL_dR);
    grad_rotations[primitive_idx] = dL_drotation;
}
