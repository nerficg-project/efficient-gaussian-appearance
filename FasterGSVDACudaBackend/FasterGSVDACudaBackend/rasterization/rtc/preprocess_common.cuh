// shared helpers for the jit-compiled preprocess kernels; concatenated into the rtc source by
// appearance.cu (see assemble_rtc_source for the assembly order), so it must stay self-contained:
// no project includes, no cooperative_groups, only nvrtc built-ins and the tcnn rtc preamble
//
// tokens defined by the preamble generated in appearance.cu:
//   JIT_APPEARANCE_MODEL: 0 = precomputed colors, 1 = spherical harmonics, 2 = spherical voronoi, 3 = nasg, 4 = nasgabor, 5 = neural (residual mlp)
//   JIT_BASE_ACTIVATION: 0 = none, 1 = exp (exp(3x))
//   JIT_RESIDUAL_ACTIVATION: 0 = none, 1 = tanh, 2 = softplus (beta=10)
//   JIT_COLOR_ACTIVATION: 0 = none, 1 = relu, 2 = softplus (beta=10), 3 = sigmoid (sigmoid(4x)), 4 = hardsigmoid (clamp(x + 0.5, 0, 1)), 5 = satexp (1 - exp(-x))
//   JIT_DIRECTION_GRADIENT: whether the color backward propagates through the view direction
//   JIT_SH_REST_BASES, the neural tokens JIT_N_INPUT_DIMS, JIT_N_FEATURE_DIMS, JIT_BASE_INPUT,
//   JIT_N_FREQUENCIES, JIT_CTX_BYTES, JIT_LOSS_SCALE, JIT_SH_DEGREE_MASK, JIT_NASG_N_LOBES, JIT_NASGABOR_N_LOBES, and JIT_SV_N_SITES

namespace faster_gs {

    // the faster_gs::config constants are emitted into the preamble from rasterization_config.h

    using rasterization::AppearanceInputs;
    using rasterization::AppearanceGradients;
    using rasterization::convert_screen_bounds_to_tile_bounds;
    using rasterization::will_primitive_contribute;

    struct mat3x3 {
        float m11, m12, m13;
        float m21, m22, m23;
        float m31, m32, m33;
    };

    struct mat3x3_triu {
        float m11, m12, m13;
        float m22, m23;
        float m33;
    };

    __device__ inline float2 operator-(const float2& a, const float b) { return make_float2(a.x - b, a.y - b); }
    __device__ inline float2 operator*(const float a, const float2& b) { return make_float2(a * b.x, a * b.y); }
    __device__ inline float norm(const float2& a) { return sqrtf(a.x * a.x + a.y * a.y); }
    __device__ inline float3 operator*(const float3& a, const float3& b) { return ::make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
    __device__ inline float3 operator+(const float3& a, const float3& b) { return ::make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
    __device__ inline float3 operator-(const float3& a, const float3& b) { return ::make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
    __device__ inline float3 operator*(const float a, const float3& b) { return ::make_float3(a * b.x, a * b.y, a * b.z); }
    __device__ inline float3 operator*(const float3& a, const float b) { return ::make_float3(a.x * b, a.y * b, a.z * b); }
    __device__ inline float4 operator*(const float a, const float4& b) { return ::make_float4(a * b.x, a * b.y, a * b.z, a * b.w); }
    __device__ inline float3 expf(const float3& a) { return ::make_float3(::expf(a.x), ::expf(a.y), ::expf(a.z)); }
    __device__ inline float dot(const float3& a, const float3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
    __device__ inline float4 make_float4(const float3& a, const float w) { return ::make_float4(a.x, a.y, a.z, w); }
    __device__ inline float clamp(const float x, const float lo, const float hi) { return fminf(fmaxf(x, lo), hi); }
    __device__ inline float norm_rcp(const float3& v) { return rsqrtf(fmaxf(dot(v, v), 1e-24f)); }
    __device__ inline float3 normalize(const float3& v) { return norm_rcp(v) * v; }
    __device__ inline float3 normalize(const float3& v, float& v_norm_rcp) { v_norm_rcp = norm_rcp(v); return v_norm_rcp * v; }
    __device__ inline float sigmoid(const float x) { return 1.0f / (1.0f + ::expf(-x)); }
    __device__ inline float sigmoid_grad(const float y) { return y - y * y; }
    __device__ inline float relu(const float x) { return fmaxf(x, 0.0f); }
    __device__ inline float relu_grad(const float y) { return y > 0.0f ? 1.0f : 0.0f; }
    __device__ inline float softplus(const float x, const float beta) { return (relu(beta * x) + log1pf(::expf(-fabsf(beta * x)))) / beta; }
    __device__ inline float softplus_grad(const float y, const float beta) { return -expm1f(-beta * y); }
    __device__ inline float hard_sigmoid(const float x) { return clamp(x / 6.0f + 0.5f, 0.0f, 1.0f); }
    __device__ inline float hard_sigmoid_grad(const float y) { return (y > 0.0f && y < 1.0f) ? (1.0f / 6.0f) : 0.0f; }
    __device__ inline float satexp(const float x) { return -expm1f(-x); }
    __device__ inline float satexp_grad(const float y) { return 1.0f - y; }

    // base term entering the color activation (JIT_BASE_ACTIVATION: 0 = none, the 0-centered
    // pre-activation color, 1 = exp: the non-negative log-parametrized intensity exp(3x))
    __device__ inline float base_activation(const float x) {
        switch (JIT_BASE_ACTIVATION) {
            case 0: return x;
            case 1: return ::expf(3.0f * x);
            default: __trap();
        }
        return x;
    }

    // derivative of the base activation with respect to its input value
    __device__ inline float base_activation_grad(const float x) {
        switch (JIT_BASE_ACTIVATION) {
            case 0: return 1.0f;
            case 1: return 3.0f * ::expf(3.0f * x);
            default: __trap();
        }
        return 1.0f;
    }

    // optional activation on the view-dependent amplitudes (JIT_RESIDUAL_ACTIVATION: 0 = none,
    // 1 = tanh, bounding the pre-activation swing to (-1, 1) per amplitude,
    // 2 = softplus with beta=10, rectifying the amplitudes into non-negative intensities)
    __device__ inline float residual_activation(const float x) {
        switch (JIT_RESIDUAL_ACTIVATION) {
            case 0: return x;
            case 1: return tanhf(x);
            case 2: return softplus(x, 10.0f);
            default: __trap();
        }
        return x;
    }

    // derivative of the residual activation, expressed in terms of its output value
    __device__ inline float residual_activation_grad(const float y) {
        switch (JIT_RESIDUAL_ACTIVATION) {
            case 0: return 1.0f;
            case 1: return 1.0f - y * y;
            case 2: return softplus_grad(y, 10.0f);
            default: __trap();
        }
        return 1.0f;
    }

    // activations scaled to behave more similar to ReLU and shifted so x is 0-centered for activation(x) in [0, 1];
    // shared by all appearance models, which parametrize the 0-centered pre-activation color (satexp maps a
    // non-negative intensity sum and is anchored at black instead, so it takes x unshifted)
    __device__ inline float color_activation(const float x) {
        switch (JIT_COLOR_ACTIVATION) {
            case 0: return x + 0.5f;
            case 1: return relu(x + 0.5f);
            case 2: return softplus(x + 0.5f, 10.0f);
            case 3: return sigmoid(4.0f * x);
            case 4: return hard_sigmoid(6.0f * x);
            case 5: return satexp(x);
            default: __trap();
        }
        return x;
    }

    // derivative of the color activation, expressed in terms of its output value
    __device__ inline float color_activation_grad(const float y) {
        switch (JIT_COLOR_ACTIVATION) {
            case 0: return 1.0f;
            case 1: return relu_grad(y);
            case 2: return softplus_grad(y, 10.0f);
            case 3: return 4.0f * sigmoid_grad(y);
            case 4: return 6.0f * hard_sigmoid_grad(y);
            case 5: return satexp_grad(y);
            default: __trap();
        }
        return 1.0f;
    }

    __device__ inline float3 base_activation(const float3& x) { return ::make_float3(base_activation(x.x), base_activation(x.y), base_activation(x.z)); }
    __device__ inline float3 base_activation_grad(const float3& x) { return ::make_float3(base_activation_grad(x.x), base_activation_grad(x.y), base_activation_grad(x.z)); }
    __device__ inline float3 residual_activation(const float3& x) { return ::make_float3(residual_activation(x.x), residual_activation(x.y), residual_activation(x.z)); }
    __device__ inline float3 residual_activation_grad(const float3& y) { return ::make_float3(residual_activation_grad(y.x), residual_activation_grad(y.y), residual_activation_grad(y.z)); }
    __device__ inline float3 color_activation(const float3& raw_color) { return ::make_float3(color_activation(raw_color.x), color_activation(raw_color.y), color_activation(raw_color.z)); }
    __device__ inline float3 color_activation_grad(const float3& color) { return ::make_float3(color_activation_grad(color.x), color_activation_grad(color.y), color_activation_grad(color.z)); }

    __device__ inline mat3x3 convert_quaternion_to_rotation_matrix(
        const float4& quaternion,
        float& norm_sq)
    {
        const float r = quaternion.x, x = quaternion.y, y = quaternion.z, z = quaternion.w;
        const float xx = x * x, yy = y * y, zz = z * z;
        const float xy = x * y, xz = x * z, yz = y * z;
        const float rx = r * x, ry = r * y, rz = r * z;
        norm_sq = r * r + xx + yy + zz;
        const float norm_sq_rcp = 1.0f / norm_sq;
        return {
            1.0f - 2.0f * (yy + zz) * norm_sq_rcp, 2.0f * (xy - rz) * norm_sq_rcp, 2.0f * (xz + ry) * norm_sq_rcp,
            2.0f * (xy + rz) * norm_sq_rcp, 1.0f - 2.0f * (xx + zz) * norm_sq_rcp, 2.0f * (yz - rx) * norm_sq_rcp,
            2.0f * (xz - ry) * norm_sq_rcp, 2.0f * (yz + rx) * norm_sq_rcp, 1.0f - 2.0f * (xx + yy) * norm_sq_rcp
        };
    }

    __device__ inline float4 convert_quaternion_to_rotation_matrix_backward(
        const float4& quaternion,
        const mat3x3& dL_dR)
    {
        const float r = quaternion.x, x = quaternion.y, y = quaternion.z, z = quaternion.w;
        const float xx = x * x, yy = y * y, zz = z * z;
        const float xy = x * y, xz = x * z, yz = y * z;
        const float rx = r * x, ry = r * y, rz = r * z;
        const float norm_sq = r * r + xx + yy + zz;
        const float norm_sq_rcp = 1.0f / norm_sq;
        const float dL_dxx = dL_dR.m22 + dL_dR.m33;
        const float dL_dyy = dL_dR.m11 + dL_dR.m33;
        const float dL_dzz = dL_dR.m11 + dL_dR.m22;
        const float dL_drz = dL_dR.m21 - dL_dR.m12;
        const float dL_dxy = dL_dR.m21 + dL_dR.m12;
        const float dL_dry = dL_dR.m13 - dL_dR.m31;
        const float dL_dxz = dL_dR.m13 + dL_dR.m31;
        const float dL_drx = dL_dR.m32 - dL_dR.m23;
        const float dL_dyz = dL_dR.m32 + dL_dR.m23;
        const float two_over_norm_sq = 2.0f * norm_sq_rcp;
        const float dL_dnorm_helper = two_over_norm_sq * (xy * dL_dxy + xz * dL_dxz + yz * dL_dyz + rx * dL_drx + ry * dL_dry + rz * dL_drz - xx * dL_dxx - yy * dL_dyy - zz * dL_dzz);
        return two_over_norm_sq * ::make_float4(
            x * dL_drx + y * dL_dry + z * dL_drz - r * dL_dnorm_helper,
            r * dL_drx - 2.0f * x * dL_dxx + y * dL_dxy + z * dL_dxz - x * dL_dnorm_helper,
            r * dL_dry + x * dL_dxy - 2.0f * y * dL_dyy + z * dL_dyz - y * dL_dnorm_helper,
            r * dL_drz + x * dL_dxz + y * dL_dyz - 2.0f * z * dL_dzz - z * dL_dnorm_helper
        );
    }

    __device__ inline uint32_t compute_exact_n_touched_tiles(
        const float2& mean2d,
        const float3& conic,
        const uint4& tile_bounds,
        const float power_threshold,
        const uint32_t tile_count,
        const bool active)
    {
        constexpr uint32_t warp_size = 32;
        constexpr uint32_t full_mask = 0xffffffff;
        const uint32_t lane_idx = threadIdx.x % warp_size;

        const float2 mean2d_shifted = mean2d - 0.5f;

        uint32_t n_touched_tiles = 0;
        const uint32_t tile_bounds_width = tile_bounds.y - tile_bounds.x;
        for (uint32_t instance_idx = 0; active && instance_idx < tile_count && instance_idx < config::n_sequential_threshold; instance_idx++) {
            const uint32_t tile_x = tile_bounds.x + (instance_idx % tile_bounds_width);
            const uint32_t tile_y = tile_bounds.z + (instance_idx / tile_bounds_width);
            if (will_primitive_contribute(mean2d_shifted, conic, tile_x, tile_y, power_threshold)) n_touched_tiles++;
        }

        const bool compute_cooperatively = active && tile_count > config::n_sequential_threshold;
        const uint32_t remaining_threads = __ballot_sync(full_mask, compute_cooperatively);
        if (remaining_threads == 0) return n_touched_tiles;

        const uint32_t n_remaining_threads = __popc(remaining_threads);
        for (uint32_t n = 0; n < n_remaining_threads && n < warp_size; n++) {
            const uint32_t current_lane = __fns(remaining_threads, 0, n + 1);

            const uint2 min_tile_bounds_coop = make_uint2(
                __shfl_sync(full_mask, tile_bounds.x, current_lane),
                __shfl_sync(full_mask, tile_bounds.z, current_lane)
            );
            const uint32_t tile_bounds_width_coop = __shfl_sync(full_mask, tile_bounds_width, current_lane);
            const uint32_t tile_count_coop = __shfl_sync(full_mask, tile_count, current_lane);

            const float2 mean2d_shifted_coop = make_float2(
                __shfl_sync(full_mask, mean2d_shifted.x, current_lane),
                __shfl_sync(full_mask, mean2d_shifted.y, current_lane)
            );
            const float3 conic_coop = ::make_float3(
                __shfl_sync(full_mask, conic.x, current_lane),
                __shfl_sync(full_mask, conic.y, current_lane),
                __shfl_sync(full_mask, conic.z, current_lane)
            );
            const float power_threshold_coop = __shfl_sync(full_mask, power_threshold, current_lane);

            const uint32_t remaining_tile_count = tile_count_coop - config::n_sequential_threshold;
            const uint32_t n_iterations = (remaining_tile_count + warp_size - 1) / warp_size;
            for (uint32_t i = 0; i < n_iterations; i++) {
                const uint32_t instance_idx = i * warp_size + lane_idx + config::n_sequential_threshold;
                const uint32_t tile_x = min_tile_bounds_coop.x + (instance_idx % tile_bounds_width_coop);
                const uint32_t tile_y = min_tile_bounds_coop.y + (instance_idx / tile_bounds_width_coop);
                const bool contributes = instance_idx < tile_count_coop && will_primitive_contribute(mean2d_shifted_coop, conic_coop, tile_x, tile_y, power_threshold_coop);
                const uint32_t contributes_ballot = __ballot_sync(full_mask, contributes);
                const uint32_t n_contributes = __popc(contributes_ballot);
                n_touched_tiles += (current_lane == lane_idx) * n_contributes;
            }
        }

        return n_touched_tiles;
    }

}
