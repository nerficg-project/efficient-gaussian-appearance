// SH appearance model residual chunk (spliced into capture_common.glsl).
//
// Coefficients are stored quantized in Spark's packed degree-3 layout, one
// texture per degree, rgb fields bit-packed little-endian, signed, at a
// per-level scale:
//   degree 1: paramTex0 (RG32UI),   9 × 7-bit codes
//   degree 2: paramTex1 (RGBA32UI), 15 × 8-bit codes
//   degree 3: paramTex2 (RGBA32UI), 21 × 6-bit codes
// residual = residual_activation(Σ basis_i(dir) · rgb_i); the activation applies
// to the band SUM as in the reference renderer, so the coefficients are raw.
//
// Injected defines: SH_DEGREE_MASK (contiguous from degree 1), RESIDUAL_ACT.
// Uniform: shScales, the per-level dequantization factor.

#if RENDER_RESIDUAL
uniform vec3 shScales;

vec3 residualColor(ivec2 uv, vec3 d) {
    vec3 res = vec3(0.0);
    float x = d.x, y = d.y, z = d.z;
#if (SH_DEGREE_MASK & 2) != 0
    {
        uvec2 q = texelFetch(paramTex0, uv, 0).rg;
        vec3 sh1_0 = vec3(ivec3(
            int(q.x << 25u) >> 25,
            int(q.x << 18u) >> 25,
            int(q.x << 11u) >> 25
        ));
        vec3 sh1_1 = vec3(ivec3(
            int(q.x << 4u) >> 25,
            int((q.x >> 3u) | (q.y << 29u)) >> 25,
            int(q.y << 22u) >> 25
        ));
        vec3 sh1_2 = vec3(ivec3(
            int(q.y << 15u) >> 25,
            int(q.y << 8u) >> 25,
            int(q.y << 1u) >> 25
        ));
        res += (sh1_0 * (-0.48860251190291987 * y)
            + sh1_1 * (0.48860251190291987 * z)
            + sh1_2 * (-0.48860251190291987 * x)) * shScales.x;
    }
#endif
#if (SH_DEGREE_MASK & 4) != 0
    float xy = x * y, xz = x * z, yz = y * z;
    float x2 = x * x, y2 = y * y, z2 = z * z;
    {
        uvec4 q = texelFetch(paramTex1, uv, 0);
        vec3 sh2_0 = vec3(ivec3(
            int(q.x << 24u) >> 24,
            int(q.x << 16u) >> 24,
            int(q.x << 8u) >> 24
        ));
        vec3 sh2_1 = vec3(ivec3(
            int(q.x) >> 24,
            int(q.y << 24u) >> 24,
            int(q.y << 16u) >> 24
        ));
        vec3 sh2_2 = vec3(ivec3(
            int(q.y << 8u) >> 24,
            int(q.y) >> 24,
            int(q.z << 24u) >> 24
        ));
        vec3 sh2_3 = vec3(ivec3(
            int(q.z << 16u) >> 24,
            int(q.z << 8u) >> 24,
            int(q.z) >> 24
        ));
        vec3 sh2_4 = vec3(ivec3(
            int(q.w << 24u) >> 24,
            int(q.w << 16u) >> 24,
            int(q.w << 8u) >> 24
        ));
        res += (sh2_0 * (1.0925484305920792 * xy)
            + sh2_1 * (-1.0925484305920792 * yz)
            + sh2_2 * (0.94617469575755997 * z2 - 0.31539156525251999)
            + sh2_3 * (-1.0925484305920792 * xz)
            + sh2_4 * (0.54627421529603959 * (x2 - y2))) * shScales.y;
    }
#endif
#if (SH_DEGREE_MASK & 8) != 0
    {
        uvec4 q = texelFetch(paramTex2, uv, 0);
        vec3 sh3_0 = vec3(ivec3(
            int(q.x << 26u) >> 26,
            int(q.x << 20u) >> 26,
            int(q.x << 14u) >> 26
        ));
        vec3 sh3_1 = vec3(ivec3(
            int(q.x << 8u) >> 26,
            int(q.x << 2u) >> 26,
            int((q.x >> 4u) | (q.y << 28u)) >> 26
        ));
        vec3 sh3_2 = vec3(ivec3(
            int(q.y << 22u) >> 26,
            int(q.y << 16u) >> 26,
            int(q.y << 10u) >> 26
        ));
        vec3 sh3_3 = vec3(ivec3(
            int(q.y << 4u) >> 26,
            int((q.y >> 2u) | (q.z << 30u)) >> 26,
            int(q.z << 24u) >> 26
        ));
        vec3 sh3_4 = vec3(ivec3(
            int(q.z << 18u) >> 26,
            int(q.z << 12u) >> 26,
            int(q.z << 6u) >> 26
        ));
        vec3 sh3_5 = vec3(ivec3(
            int(q.z) >> 26,
            int(q.w << 26u) >> 26,
            int(q.w << 20u) >> 26
        ));
        vec3 sh3_6 = vec3(ivec3(
            int(q.w << 14u) >> 26,
            int(q.w << 8u) >> 26,
            int(q.w << 2u) >> 26
        ));
        res += (sh3_0 * (0.59004358992664352 * y * (-3.0 * x2 + y2))
            + sh3_1 * (2.8906114426405538 * xy * z)
            + sh3_2 * (0.45704579946446572 * y * (1.0 - 5.0 * z2))
            + sh3_3 * (0.3731763325901154 * z * (5.0 * z2 - 3.0))
            + sh3_4 * (0.45704579946446572 * x * (1.0 - 5.0 * z2))
            + sh3_5 * (1.4453057213202769 * z * (x2 - y2))
            + sh3_6 * (0.59004358992664352 * x * (-x2 + 3.0 * y2))) * shScales.z;
    }
#endif
#if RESIDUAL_ACT == 1
    return tanh(res);
#elif RESIDUAL_ACT == 2
    return softplus10(res);
#else
    return res;
#endif
}
#endif
