// spherical voronoi appearance model for the jit-compiled preprocess kernels (JIT_APPEARANCE_MODEL == 2)
// the number of sites is baked in via JIT_SV_N_SITES, the residual is the softmax-weighted site color sum
// with the shared residual activation applied to the site colors (none in the reference configuration),
// and the backward takes the (activation-corrected) color gradient as a value
//
// Soft spherical Voronoi partition of the direction sphere.
// The view-dependent residual of a primitive is, given the view direction v,
//     dist_k = ||normalize(site_k) - v||
//     w      = softmax_k(-tau_k * dist_k)
//     R(v)   = sum_k w_k * c_k
// The softmax weights sum to one, so the shared base color takes the role of the reference's common
// site color component and the per-site colors parametrize the deviations from it.
// The softmax is evaluated in a single pass using a running maximum, i.e. the site data is read only once.
// Each site stores 7 raw (i.e. pre-activation) parameters:
//     [0..2] site        -> normalize -> Voronoi site direction
//     [3]    tau         -> exp -> softmax temperature
//     [4..6] color (rgb) -> residual activation -> site color

namespace faster_gs {

    constexpr float sv_site_length_eps = 1e-12f;  // matches the eps used by torch.nn.functional.normalize()
    constexpr float sv_distance_eps = 1e-12f;     // avoids division by zero for sites that coincide with the view direction

    constexpr uint32_t sv_params_per_site = 7;

    __device__ inline float3 sv_residual(
        const float* __restrict__ sv_params,
        const float3& position,
        const float3& cam_position,
        const uint32_t primitive_idx,
        const uint32_t active_sites)
    {
        if (active_sites == 0) return ::make_float3(0.0f, 0.0f, 0.0f);

        const float3 direction = normalize(position - cam_position);
        const float* site_ptr = sv_params + primitive_idx * JIT_SV_N_SITES * sv_params_per_site;

        float max_logit = -1e30f;
        float weight_sum = 0.0f;
        float3 weighted_color = ::make_float3(0.0f, 0.0f, 0.0f);
        for (uint32_t site_idx = 0; site_idx < active_sites; ++site_idx, site_ptr += sv_params_per_site) {
            const float3 raw_site = ::make_float3(site_ptr[0], site_ptr[1], site_ptr[2]);
            const float3 site = (1.0f / fmaxf(sqrtf(dot(raw_site, raw_site)), sv_site_length_eps)) * raw_site;
            const float3 site_to_direction = site - direction;
            const float distance = sqrtf(dot(site_to_direction, site_to_direction));
            const float logit = -::expf(site_ptr[3]) * distance;
            // update the running maximum and rescale the accumulators accordingly
            const float new_max_logit = fmaxf(max_logit, logit);
            const float rescale = ::expf(max_logit - new_max_logit);
            // the clamp guards against fma contraction: the unrounded -tau * distance can exceed the
            // rounded new_max_logit by half an ulp, which expf turns into inf at large tau
            const float weight = ::expf(fminf(logit - new_max_logit, 0.0f));
            weight_sum = weight_sum * rescale + weight;
            weighted_color = weighted_color * rescale + weight * residual_activation(::make_float3(site_ptr[4], site_ptr[5], site_ptr[6]));
            max_logit = new_max_logit;
        }
        return (1.0f / weight_sum) * weighted_color;
    }

    __device__ inline float3 sv_residual_backward(
        const float* __restrict__ sv_params,
        const float3& grad_color,
        float* __restrict__ grad_sv_params,
        const float3& position,
        const float3& cam_position,
        const uint32_t primitive_idx,
        const uint32_t active_sites)
    {
        if (active_sites == 0) return ::make_float3(0.0f, 0.0f, 0.0f);

        float direction_norm_rcp;
        const float3 direction = normalize(position - cam_position, direction_norm_rcp);
        const uint32_t params_base_idx = primitive_idx * JIT_SV_N_SITES * sv_params_per_site;

        // recompute the forward softmax (first pass over the site data)
        float max_logit = -1e30f;
        float weight_sum = 0.0f;
        float3 weighted_color = ::make_float3(0.0f, 0.0f, 0.0f);
        const float* site_ptr = sv_params + params_base_idx;
        for (uint32_t site_idx = 0; site_idx < active_sites; ++site_idx, site_ptr += sv_params_per_site) {
            const float3 raw_site = ::make_float3(site_ptr[0], site_ptr[1], site_ptr[2]);
            const float3 site = (1.0f / fmaxf(sqrtf(dot(raw_site, raw_site)), sv_site_length_eps)) * raw_site;
            const float3 site_to_direction = site - direction;
            const float distance = sqrtf(dot(site_to_direction, site_to_direction));
            const float logit = -::expf(site_ptr[3]) * distance;
            const float new_max_logit = fmaxf(max_logit, logit);
            const float rescale = ::expf(max_logit - new_max_logit);
            // the clamp guards against fma contraction: the unrounded -tau * distance can exceed the
            // rounded new_max_logit by half an ulp, which expf turns into inf at large tau
            const float weight = ::expf(fminf(logit - new_max_logit, 0.0f));
            weight_sum = weight_sum * rescale + weight;
            weighted_color = weighted_color * rescale + weight * residual_activation(::make_float3(site_ptr[4], site_ptr[5], site_ptr[6]));
            max_logit = new_max_logit;
        }
        const float weight_sum_rcp = 1.0f / weight_sum;
        // sum_k w_k * <grad_color, c_k> = <grad_color, R(v)>, needed for the softmax pullback
        const float grad_weight_mean = dot(grad_color, weight_sum_rcp * weighted_color);

        // write the site gradients (second pass over the site data)
        float3 grad_direction = ::make_float3(0.0f, 0.0f, 0.0f);
        site_ptr = sv_params + params_base_idx;
        float* grad_site_ptr = grad_sv_params + params_base_idx;
        for (uint32_t site_idx = 0; site_idx < active_sites; ++site_idx, site_ptr += sv_params_per_site, grad_site_ptr += sv_params_per_site) {
            const float3 raw_site = ::make_float3(site_ptr[0], site_ptr[1], site_ptr[2]);
            const float site_length_rcp = 1.0f / fmaxf(sqrtf(dot(raw_site, raw_site)), sv_site_length_eps);
            const float3 site = site_length_rcp * raw_site;
            const float3 site_to_direction = site - direction;
            const float distance = sqrtf(dot(site_to_direction, site_to_direction));
            const float tau = ::expf(site_ptr[3]);
            const float weight = ::expf(fminf(-tau * distance - max_logit, 0.0f)) * weight_sum_rcp;

            // site colors, through the shared residual activation
            const float3 color = residual_activation(::make_float3(site_ptr[4], site_ptr[5], site_ptr[6]));
            const float3 grad_site_color = weight * grad_color * residual_activation_grad(color);
            grad_site_ptr[4] = grad_site_color.x;
            grad_site_ptr[5] = grad_site_color.y;
            grad_site_ptr[6] = grad_site_color.z;

            // softmax pullback: dL/dlogit = w * (<grad_color, color> - grad_weight_mean)
            const float grad_logit = weight * (dot(grad_color, color) - grad_weight_mean);

            // logit = -tau * distance with tau = exp(raw)
            grad_site_ptr[3] = -distance * tau * grad_logit;
            const float grad_distance = -tau * grad_logit;

            // distance = ||site - direction||
            const float3 grad_site_to_direction = (grad_distance / fmaxf(distance, sv_distance_eps)) * site_to_direction;

            // site = raw_site / ||raw_site||
            const float3 grad_site = (grad_site_to_direction - dot(grad_site_to_direction, site) * site) * site_length_rcp;
            grad_site_ptr[0] = grad_site.x;
            grad_site_ptr[1] = grad_site.y;
            grad_site_ptr[2] = grad_site.z;

            grad_direction = grad_direction - grad_site_to_direction;
        }

        return (grad_direction - dot(grad_direction, direction) * direction) * direction_norm_rcp;
    }

}
