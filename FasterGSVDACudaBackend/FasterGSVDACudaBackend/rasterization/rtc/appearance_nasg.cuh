// nasg appearance model for the jit-compiled preprocess kernels (JIT_APPEARANCE_MODEL == 3)
// the maximum number of lobes is baked in via JIT_NASG_N_LOBES, the residual is the lobe sum with the shared residual activation applied to the lobe weights
// (tanh in the reference configuration), and the backward takes the (activation-corrected) color
// gradient as a value
//
// Normalized Anisotropic Spherical Gaussians (NASG).
// The view-dependent color of a primitive is
//     C(v) = c0 + sum_i pdf_i(v) * w_i
// with the per-lobe response, given the lobe frame vectors x and z,
//     K    = (<v, z> + 1) / 2
//     K_e  = eps + a * <v, x>^2 / (1 - <v, z>^2)
//     E    = K^K_e
//     N    = lambda * sqrt(1 + a) / (2pi * (1 + eps_n - exp(-2 lambda)))
//     pdf  = exp(2 lambda (E * K - 1)) * E * N
// Each lobe stores 8 raw (i.e. pre-activation) parameters:
//     [0..2] frame        -> tanh -> cos(theta), cos(phi), cos(tau)
//     [3]    lambda       -> exp  -> sharpness
//     [4]    a            -> exp  -> anisotropy
//     [5..7] weight (rgb) -> residual activation -> lobe amplitude

namespace faster_gs {

    constexpr float nasg_two_pi = 6.28318530717958647692f;
    constexpr float nasg_eps = 5e-6f;               // stabilizes the exponent K_e
    constexpr float nasg_eps_norm = 1e-8f;          // stabilizes the normalization denominator
    constexpr float nasg_cos_limit = 0.999999f;     // keeps sin(.) = sqrt(1 - cos(.)^2) away from zero
    constexpr float nasg_pole_limit = 0.99999988f;  // |<v, z>| bound of the pole-free interval (~1e-7 tolerance at the two poles)
    constexpr float nasg_shape_limit = 1e4f;        // upper bound for lambda and a

    constexpr uint32_t nasg_params_per_lobe = 8;

    // right-handed lobe frame; only x and z are needed to evaluate a lobe
    struct nasg_frame {
        float3 x, z;
        float cos_theta, cos_phi, cos_tau;
        float sin_theta, sin_phi, sin_tau;
    };

    __device__ inline nasg_frame build_nasg_frame(const float cos_theta, const float cos_phi, const float cos_tau)
    {
        const float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);
        const float sin_phi = sqrtf(1.0f - cos_phi * cos_phi);
        const float sin_tau = sqrtf(1.0f - cos_tau * cos_tau);
        return {
            ::make_float3(
                cos_theta * cos_phi * cos_tau - sin_theta * sin_tau,
                sin_theta * cos_phi * cos_tau + cos_theta * sin_tau,
                -sin_phi * cos_tau
            ),
            ::make_float3(
                cos_theta * sin_phi,
                sin_theta * sin_phi,
                cos_phi
            ),
            cos_theta, cos_phi, cos_tau,
            sin_theta, sin_phi, sin_tau
        };
    }

    __device__ inline float3 nasg_residual(
        const float* __restrict__ nasg_params,
        const float3& position,
        const float3& cam_position,
        const uint32_t primitive_idx,
        const uint32_t active_lobes)
    {
        float3 result = ::make_float3(0.0f, 0.0f, 0.0f);
        if (active_lobes == 0) return result;

        const float3 direction = normalize(position - cam_position);
        const float* lobe_ptr = nasg_params + primitive_idx * JIT_NASG_N_LOBES * nasg_params_per_lobe;
        for (uint32_t lobe_idx = 0; lobe_idx < active_lobes; ++lobe_idx, lobe_ptr += nasg_params_per_lobe) {
            const nasg_frame frame = build_nasg_frame(
                clamp(tanhf(lobe_ptr[0]), -nasg_cos_limit, nasg_cos_limit),
                clamp(tanhf(lobe_ptr[1]), -nasg_cos_limit, nasg_cos_limit),
                clamp(tanhf(lobe_ptr[2]), -nasg_cos_limit, nasg_cos_limit)
            );
            const float lambda = fminf(::expf(lobe_ptr[3]), nasg_shape_limit);
            const float a = fminf(::expf(lobe_ptr[4]), nasg_shape_limit);
            const float v_z = dot(direction, frame.z);
            const float v_x = dot(direction, frame.x);

            const float v_z_clamped = clamp(v_z, -nasg_pole_limit, nasg_pole_limit);
            const float K = 0.5f * (v_z_clamped + 1.0f);
            const float K_e = nasg_eps + a * v_x * v_x / (1.0f - v_z_clamped * v_z_clamped);
            const float E = powf(K, K_e);
            const float N = lambda * sqrtf(1.0f + a) / (nasg_two_pi * (1.0f + nasg_eps_norm - ::expf(-2.0f * lambda)));
            const float p = ::expf(2.0f * lambda * (E * K - 1.0f));
            const float pdf = v_z >= nasg_pole_limit ? 1.0f : (v_z > -nasg_pole_limit ? p * E * N : 0.0f);

            result = result + pdf * residual_activation(::make_float3(lobe_ptr[5], lobe_ptr[6], lobe_ptr[7]));
        }
        return result;
    }

    __device__ inline float3 nasg_residual_backward(
        const float* __restrict__ nasg_params,
        const float3& grad_color,
        float* __restrict__ grad_nasg_params,
        const float3& position,
        const float3& cam_position,
        const uint32_t primitive_idx,
        const uint32_t active_lobes)
    {
        if (active_lobes == 0) return ::make_float3(0.0f, 0.0f, 0.0f);

        float direction_norm_rcp;
        const float3 direction = normalize(position - cam_position, direction_norm_rcp);
        const uint32_t params_base_idx = primitive_idx * JIT_NASG_N_LOBES * nasg_params_per_lobe;
        const float* lobe_ptr = nasg_params + params_base_idx;
        float* grad_lobe_ptr = grad_nasg_params + params_base_idx;
        float3 grad_direction = ::make_float3(0.0f, 0.0f, 0.0f);

        for (uint32_t lobe_idx = 0; lobe_idx < active_lobes; ++lobe_idx, lobe_ptr += nasg_params_per_lobe, grad_lobe_ptr += nasg_params_per_lobe) {
            // recompute the activations, keeping what the chain rule needs
            const float tanh_theta = tanhf(lobe_ptr[0]);
            const float tanh_phi = tanhf(lobe_ptr[1]);
            const float tanh_tau = tanhf(lobe_ptr[2]);
            const nasg_frame frame = build_nasg_frame(
                clamp(tanh_theta, -nasg_cos_limit, nasg_cos_limit),
                clamp(tanh_phi, -nasg_cos_limit, nasg_cos_limit),
                clamp(tanh_tau, -nasg_cos_limit, nasg_cos_limit)
            );
            const float lambda_unclamped = ::expf(lobe_ptr[3]);
            const float a_unclamped = ::expf(lobe_ptr[4]);
            const float lambda = fminf(lambda_unclamped, nasg_shape_limit);
            const float a = fminf(a_unclamped, nasg_shape_limit);
            const float3 w = residual_activation(::make_float3(lobe_ptr[5], lobe_ptr[6], lobe_ptr[7]));

            const float v_z = dot(direction, frame.z);
            const float v_x = dot(direction, frame.x);

            // recompute the forward quantities
            const float v_z_clamped = clamp(v_z, -nasg_pole_limit, nasg_pole_limit);
            const float K = 0.5f * (v_z_clamped + 1.0f);
            const float one_minus_v_z_sq = 1.0f - v_z_clamped * v_z_clamped;
            const float K_e = nasg_eps + a * v_x * v_x / one_minus_v_z_sq;
            const float E = powf(K, K_e);
            const float norm_denominator = nasg_two_pi * (1.0f + nasg_eps_norm - ::expf(-2.0f * lambda));
            const float N = lambda * sqrtf(1.0f + a) / norm_denominator;
            const float p = ::expf(2.0f * lambda * (E * K - 1.0f));
            const float log_K = logf(K);
            const bool valid_mask = fabsf(v_z) < nasg_pole_limit;
            const float pdf = v_z >= nasg_pole_limit ? 1.0f : (valid_mask ? p * E * N : 0.0f);

            // lobe weights, through the shared residual activation
            const float3 grad_w = grad_color * pdf * residual_activation_grad(w);
            grad_lobe_ptr[5] = grad_w.x;
            grad_lobe_ptr[6] = grad_w.y;
            grad_lobe_ptr[7] = grad_w.z;

            const float grad_pdf = valid_mask ? dot(grad_color, w) : 0.0f;

            // lambda enters through p and through the normalization N
            const float dp_dlambda = p * 2.0f * (E * K - 1.0f);
            const float sqrt_1_plus_a = sqrtf(1.0f + a);
            const float dN_dlambda = (sqrt_1_plus_a * norm_denominator
                                      - lambda * sqrt_1_plus_a * 2.0f * nasg_two_pi * ::expf(-2.0f * lambda))
                                   / (norm_denominator * norm_denominator);
            const float dpdf_dlambda = dp_dlambda * E * N + p * E * dN_dlambda;
            // gradients of clamped shape parameters are detached
            grad_lobe_ptr[3] = lambda_unclamped >= nasg_shape_limit ? 0.0f : grad_pdf * dpdf_dlambda * lambda;

            // a enters through K_e (hence E and p) and through N
            const float dE_da = E * log_K * (v_x * v_x / one_minus_v_z_sq);
            const float dp_da = p * 2.0f * lambda * K * dE_da;
            const float dN_da = (lambda / (2.0f * sqrt_1_plus_a)) / norm_denominator;
            const float dpdf_da = dp_da * E * N + p * dE_da * N + p * E * dN_da;
            grad_lobe_ptr[4] = a_unclamped >= nasg_shape_limit ? 0.0f : grad_pdf * dpdf_da * a;

            // projection onto the lobe frame: v_x = <direction, x>
            const float dE_dv_x = E * log_K * (2.0f * a * v_x / one_minus_v_z_sq);
            const float dp_dv_x = p * 2.0f * lambda * K * dE_dv_x;
            const float grad_v_x = grad_pdf * (dp_dv_x * E * N + p * dE_dv_x * N);

            // projection onto the lobe frame: v_z = <direction, z>
            const float dK_e_dv_z = a * v_x * v_x * 2.0f * v_z / (one_minus_v_z_sq * one_minus_v_z_sq);
            const float dE_dv_z = E * (K_e / K) * 0.5f + E * log_K * dK_e_dv_z;
            const float dp_dv_z = p * 2.0f * lambda * (K * dE_dv_z + E * 0.5f);
            const float grad_v_z = grad_pdf * (dp_dv_z * E * N + p * dE_dv_z * N);

            const float3 grad_x = grad_v_x * direction;
            const float3 grad_z = grad_v_z * direction;
            grad_direction = grad_direction + grad_v_x * frame.x + grad_v_z * frame.z;

            // frame parameters, through their tanh activation
            const float dsin_theta_dcos_theta = -frame.cos_theta / frame.sin_theta;
            const float dsin_phi_dcos_phi = -frame.cos_phi / frame.sin_phi;
            const float dsin_tau_dcos_tau = -frame.cos_tau / frame.sin_tau;
            const float grad_cos_theta = grad_x.x * (frame.cos_phi * frame.cos_tau - dsin_theta_dcos_theta * frame.sin_tau)
                                       + grad_x.y * (dsin_theta_dcos_theta * frame.cos_phi * frame.cos_tau + frame.sin_tau)
                                       + grad_z.x * frame.sin_phi
                                       + grad_z.y * dsin_theta_dcos_theta * frame.sin_phi;
            const float grad_cos_phi = grad_x.x * frame.cos_theta * frame.cos_tau
                                     + grad_x.y * frame.sin_theta * frame.cos_tau
                                     - grad_x.z * dsin_phi_dcos_phi * frame.cos_tau
                                     + grad_z.x * frame.cos_theta * dsin_phi_dcos_phi
                                     + grad_z.y * frame.sin_theta * dsin_phi_dcos_phi
                                     + grad_z.z;
            const float grad_cos_tau = grad_x.x * (frame.cos_theta * frame.cos_phi - frame.sin_theta * dsin_tau_dcos_tau)
                                     + grad_x.y * (frame.sin_theta * frame.cos_phi + frame.cos_theta * dsin_tau_dcos_tau)
                                     - grad_x.z * frame.sin_phi;
            grad_lobe_ptr[0] = fabsf(tanh_theta) >= nasg_cos_limit ? 0.0f : grad_cos_theta * (1.0f - tanh_theta * tanh_theta);
            grad_lobe_ptr[1] = fabsf(tanh_phi) >= nasg_cos_limit ? 0.0f : grad_cos_phi * (1.0f - tanh_phi * tanh_phi);
            grad_lobe_ptr[2] = fabsf(tanh_tau) >= nasg_cos_limit ? 0.0f : grad_cos_tau * (1.0f - tanh_tau * tanh_tau);
        }

        return (grad_direction - dot(grad_direction, direction) * direction) * direction_norm_rcp;
    }

}
