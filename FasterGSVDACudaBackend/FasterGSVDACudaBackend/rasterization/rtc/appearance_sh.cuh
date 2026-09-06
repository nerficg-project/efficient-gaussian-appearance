// spherical harmonics appearance model for the jit-compiled preprocess kernels (JIT_APPEARANCE_MODEL == 1)
// the constant-band basis factor is folded into the base color parametrization, the total number of
// residual bases is baked in via JIT_SH_REST_BASES, the residual is the band sum beyond the constant
// band with the shared residual activation applied to it, and the backward takes the
// (activation-corrected) color gradient as a value

namespace faster_gs {

    // degree 1
    constexpr float sh_c1 = 0.48860251190291987f;
    // degree 2
    constexpr float sh_c2a = 1.0925484305920792f;
    constexpr float sh_c2b = 0.94617469575755997f;
    constexpr float sh_c2c = 0.31539156525251999f;
    constexpr float sh_c2d = 0.54627421529603959f;
    constexpr float sh_c2e = 1.8923493915151202f;
    // degree 3
    constexpr float sh_c3a = 0.59004358992664352f;
    constexpr float sh_c3b = 1.7701307697799304f;
    constexpr float sh_c3c = 2.8906114426405538f;
    constexpr float sh_c3d = 0.45704579946446572f;
    constexpr float sh_c3e = 2.2852289973223288f;
    constexpr float sh_c3f = 1.865881662950577f;
    constexpr float sh_c3g = 1.1195289977703462f;
    constexpr float sh_c3h = 1.4453057213202769f;
    constexpr float sh_c3i = 3.5402615395598609f;
    constexpr float sh_c3j = 4.5704579946446566f;
    constexpr float sh_c3k = 5.597644988851731f;

    __device__ inline float3 sh_residual(
        const float3* __restrict__ sh_coefficients_rest,
        const float3& position,
        const float3& cam_position,
        const uint32_t primitive_idx,
        const uint32_t active_sh_degree)
    {
        // computation adapted from https://github.com/NVlabs/tiny-cuda-nn/blob/212104156403bd87616c1a4f73a1c5f2c2e172a9/include/tiny-cuda-nn/common_device.h#L340
        if (active_sh_degree == 0) return ::make_float3(0.0f, 0.0f, 0.0f);

        const float3* coefficients_ptr = sh_coefficients_rest + primitive_idx * JIT_SH_REST_BASES;
        const float3 direction = normalize(position - cam_position);
        const float x = direction.x, y = direction.y, z = direction.z;
        float3 result = -sh_c1 * y * coefficients_ptr[0]
                      + sh_c1 * z * coefficients_ptr[1]
                      - sh_c1 * x * coefficients_ptr[2];
        if (active_sh_degree > 1) {
            const float xx = x * x, yy = y * y, zz = z * z;
            const float xy = x * y, xz = x * z, yz = y * z;
            result = result + sh_c2a * xy * coefficients_ptr[3]
                            - sh_c2a * yz * coefficients_ptr[4]
                            + (sh_c2b * zz - sh_c2c) * coefficients_ptr[5]
                            - sh_c2a * xz * coefficients_ptr[6]
                            + sh_c2d * (xx - yy) * coefficients_ptr[7];
            if (active_sh_degree > 2) {
                result = result + y * (sh_c3a * yy - sh_c3b * xx) * coefficients_ptr[8]
                                + sh_c3c * xy * z * coefficients_ptr[9]
                                + y * (sh_c3d - sh_c3e * zz) * coefficients_ptr[10]
                                + z * (sh_c3f * zz - sh_c3g) * coefficients_ptr[11]
                                + x * (sh_c3d - sh_c3e * zz) * coefficients_ptr[12]
                                + sh_c3h * z * (xx - yy) * coefficients_ptr[13]
                                + x * (sh_c3b * yy - sh_c3a * xx) * coefficients_ptr[14];
            }
        }
        return residual_activation(result);
    }

    __device__ inline float3 sh_residual_backward(
        const float3* __restrict__ sh_coefficients_rest,
        const float3& grad_color,
        float3* __restrict__ grad_sh_coefficients_rest,
        const float3& position,
        const float3& cam_position,
        const uint32_t primitive_idx,
        const uint32_t active_sh_degree)
    {
        // computation adapted from https://github.com/NVlabs/tiny-cuda-nn/blob/212104156403bd87616c1a4f73a1c5f2c2e172a9/include/tiny-cuda-nn/common_device.h#L421
        if (active_sh_degree == 0) return ::make_float3(0.0f, 0.0f, 0.0f);

        const uint32_t coefficients_base_idx = primitive_idx * JIT_SH_REST_BASES;
        const float3* coefficients_ptr = sh_coefficients_rest + coefficients_base_idx;
        float3* grad_coefficients_ptr = grad_sh_coefficients_rest + coefficients_base_idx;
        float direction_norm_rcp;
        const float3 direction = normalize(position - cam_position, direction_norm_rcp);
        const float x = direction.x, y = direction.y, z = direction.z;
        // the residual activation is applied to the band sum, so its derivative is a single factor
        // shared by every coefficient, recovered from the recomputed activation output
        const float3 dL_dsum = grad_color * residual_activation_grad(
            sh_residual(sh_coefficients_rest, position, cam_position, primitive_idx, active_sh_degree));
        const float3 c0 = coefficients_ptr[0];
        const float3 c1 = coefficients_ptr[1];
        const float3 c2 = coefficients_ptr[2];
        grad_coefficients_ptr[0] = -sh_c1 * y * dL_dsum;
        grad_coefficients_ptr[1] = sh_c1 * z * dL_dsum;
        grad_coefficients_ptr[2] = -sh_c1 * x * dL_dsum;
        float3 grad_direction_x = -sh_c1 * c2;
        float3 grad_direction_y = -sh_c1 * c0;
        float3 grad_direction_z = sh_c1 * c1;
        if (active_sh_degree > 1) {
            const float xx = x * x, yy = y * y, zz = z * z;
            const float xy = x * y, xz = x * z, yz = y * z;
            const float3 c3 = coefficients_ptr[3];
            const float3 c4 = coefficients_ptr[4];
            const float3 c5 = coefficients_ptr[5];
            const float3 c6 = coefficients_ptr[6];
            const float3 c7 = coefficients_ptr[7];
            grad_coefficients_ptr[3] = sh_c2a * xy * dL_dsum;
            grad_coefficients_ptr[4] = -sh_c2a * yz * dL_dsum;
            grad_coefficients_ptr[5] = (sh_c2b * zz - sh_c2c) * dL_dsum;
            grad_coefficients_ptr[6] = -sh_c2a * xz * dL_dsum;
            grad_coefficients_ptr[7] = sh_c2d * (xx - yy) * dL_dsum;
            grad_direction_x = grad_direction_x + sh_c2a * y * c3
                                                - sh_c2a * z * c6
                                                + sh_c2a * x * c7;
            grad_direction_y = grad_direction_y + sh_c2a * x * c3
                                                - sh_c2a * z * c4
                                                - sh_c2a * y * c7;
            grad_direction_z = grad_direction_z - sh_c2a * y * c4
                                                + sh_c2e * z * c5
                                                - sh_c2a * x * c6;
            if (active_sh_degree > 2) {
                const float3 c8 = coefficients_ptr[8];
                const float3 c9 = coefficients_ptr[9];
                const float3 c10 = coefficients_ptr[10];
                const float3 c11 = coefficients_ptr[11];
                const float3 c12 = coefficients_ptr[12];
                const float3 c13 = coefficients_ptr[13];
                const float3 c14 = coefficients_ptr[14];
                grad_coefficients_ptr[8] = y * (sh_c3a * yy - sh_c3b * xx) * dL_dsum;
                grad_coefficients_ptr[9] = sh_c3c * xy * z * dL_dsum;
                grad_coefficients_ptr[10] = y * (sh_c3d - sh_c3e * zz) * dL_dsum;
                grad_coefficients_ptr[11] = z * (sh_c3f * zz - sh_c3g) * dL_dsum;
                grad_coefficients_ptr[12] = x * (sh_c3d - sh_c3e * zz) * dL_dsum;
                grad_coefficients_ptr[13] = sh_c3h * z * (xx - yy) * dL_dsum;
                grad_coefficients_ptr[14] = x * (sh_c3b * yy - sh_c3a * xx) * dL_dsum;
                grad_direction_x = grad_direction_x - sh_c3i * xy * c8
                                                    + sh_c3c * yz * c9
                                                    + (sh_c3d - sh_c3e * zz) * c12
                                                    + sh_c3c * xz * c13
                                                    + sh_c3b * (yy - xx) * c14;
                grad_direction_y = grad_direction_y + sh_c3b * (yy - xx) * c8
                                                    + sh_c3c * xz * c9
                                                    + (sh_c3d - sh_c3e * zz) * c10
                                                    - sh_c3c * yz * c13
                                                    + sh_c3i * xy * c14;
                grad_direction_z = grad_direction_z + sh_c3c * xy * c9
                                                    - sh_c3j * yz * c10
                                                    + (sh_c3k * zz - sh_c3g) * c11
                                                    - sh_c3j * xz * c12
                                                    + sh_c3h * (xx - yy) * c13;
            }
        }

        const float3 grad_direction = ::make_float3(
            dot(grad_direction_x, dL_dsum),
            dot(grad_direction_y, dL_dsum),
            dot(grad_direction_z, dL_dsum)
        );
        return (grad_direction - dot(grad_direction, direction) * direction) * direction_norm_rcp;
    }

}
