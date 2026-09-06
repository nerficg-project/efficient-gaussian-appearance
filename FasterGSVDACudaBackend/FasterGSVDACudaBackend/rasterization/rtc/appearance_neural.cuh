// neural appearance model (residual mlp) for the jit-compiled preprocess kernels (JIT_APPEARANCE_MODEL == 5)
// the residual is computed by the tcnn-generated eval_model from the (optionally frequency-encoded)
// per-Gaussian features and the sh degree values of the view direction, with the shared residual
// activation applied to the rgb outputs

namespace faster_gs {

    constexpr float sh_c0 = 0.28209479177387814f;  // 1/(2*sqrt(pi))

    // NeRF-style frequency encoding of the per-Gaussian features (JIT_N_FREQUENCIES == 0: raw passthrough)
    // output layout per feature: sin(2^0*x), cos(2^0*x), sin(2^1*x), cos(2^1*x), ...
    // with JIT_BASE_INPUT, the raw base values join the learned features as 3 additional
    // encoder inputs (indices JIT_N_FEATURE_DIMS..JIT_N_FEATURE_DIMS + 2), encoded like the rest
    constexpr uint32_t n_raw_feature_dims = JIT_N_FEATURE_DIMS + (JIT_BASE_INPUT ? 3 : 0);
    constexpr uint32_t n_encoded_feature_dims = JIT_N_FREQUENCIES == 0 ? n_raw_feature_dims : n_raw_feature_dims * 2 * JIT_N_FREQUENCIES;

    template <typename IN_T, typename OUT_T>
    __device__ inline void encode_features(IN_T&& in, OUT_T&& out) {
        #pragma unroll
        for (uint32_t i = 0; i < n_raw_feature_dims; i++) {
            const float value = in(i);
            if (JIT_N_FREQUENCIES == 0) {
                out(i) = value;
                continue;
            }
            #pragma unroll
            for (uint32_t f = 0; f < JIT_N_FREQUENCIES; f++) {
                float s, c;
                sincosf(scalbnf(value, f), &s, &c);
                out(i * 2 * JIT_N_FREQUENCIES + 2 * f) = s;
                out(i * 2 * JIT_N_FREQUENCIES + 2 * f + 1) = c;
            }
        }
    }

    // gradient of the frequency encoding for a single feature value
    template <typename GRAD_T>
    __device__ inline float encode_features_grad(const float feature, GRAD_T&& dL_dencoded, const uint32_t i) {
        if (JIT_N_FREQUENCIES == 0) return dL_dencoded(i);
        float dL_dfeature = 0.0f;
        #pragma unroll
        for (uint32_t f = 0; f < JIT_N_FREQUENCIES; f++) {
            float s, c;
            sincosf(scalbnf(feature, f), &s, &c);
            dL_dfeature += __uint2float_rn(1u << f) * (c * dL_dencoded(i * 2 * JIT_N_FREQUENCIES + 2 * f) - s * dL_dencoded(i * 2 * JIT_N_FREQUENCIES + 2 * f + 1));
        }
        return dL_dfeature;
    }

    // evaluates the sh basis of the configured degrees for a unit direction
    // JIT_SH_DEGREE_MASK is a compile-time bitmask of included degrees (bit l set = degree l included, l in 0..6)
    // degrees are laid out in ascending order; degree 0 is the direction-independent input and is never gated,
    // while the i-th included degree >= 1 is zeroed unless i < active_degree
    // polynomials taken from tiny-cuda-nn's sh_enc
    template <typename OUT_T>
    __device__ inline void eval_sh_degrees(const float3& dir, const uint32_t active_degree, OUT_T&& out) {
        const float x = dir.x, y = dir.y, z = dir.z;
        const float xy = x * y, xz = x * z, yz = y * z, x2 = x * x, y2 = y * y, z2 = z * z;
        const float x4 = x2 * x2, y4 = y2 * y2, z4 = z2 * z2;
        const float x6 = x4 * x2, y6 = y4 * y2, z6 = z4 * z2;
        uint32_t offset = 0;
        uint32_t degree_rank = 0;
        if (JIT_SH_DEGREE_MASK & (1 << 0)) {
            out(offset) = sh_c0;
            offset += 1;
        }
        if (JIT_SH_DEGREE_MASK & (1 << 1)) {
            const float m = degree_rank++ < active_degree ? 1.0f : 0.0f;
            out(offset + 0) = m * (-0.48860251190291987f * y);
            out(offset + 1) = m * (0.48860251190291987f * z);
            out(offset + 2) = m * (-0.48860251190291987f * x);
            offset += 3;
        }
        if (JIT_SH_DEGREE_MASK & (1 << 2)) {
            const float m = degree_rank++ < active_degree ? 1.0f : 0.0f;
            out(offset + 0) = m * (1.0925484305920792f * xy);
            out(offset + 1) = m * (-1.0925484305920792f * yz);
            out(offset + 2) = m * (0.94617469575755997f * z2 - 0.31539156525251999f);
            out(offset + 3) = m * (-1.0925484305920792f * xz);
            out(offset + 4) = m * (0.54627421529603959f * x2 - 0.54627421529603959f * y2);
            offset += 5;
        }
        if (JIT_SH_DEGREE_MASK & (1 << 3)) {
            const float m = degree_rank++ < active_degree ? 1.0f : 0.0f;
            out(offset + 0) = m * (0.59004358992664352f * y * (-3.0f * x2 + y2));
            out(offset + 1) = m * (2.8906114426405538f * xy * z);
            out(offset + 2) = m * (0.45704579946446572f * y * (1.0f - 5.0f * z2));
            out(offset + 3) = m * (0.3731763325901154f * z * (5.0f * z2 - 3.0f));
            out(offset + 4) = m * (0.45704579946446572f * x * (1.0f - 5.0f * z2));
            out(offset + 5) = m * (1.4453057213202769f * z * (x2 - y2));
            out(offset + 6) = m * (0.59004358992664352f * x * (-x2 + 3.0f * y2));
            offset += 7;
        }
        if (JIT_SH_DEGREE_MASK & (1 << 4)) {
            const float m = degree_rank++ < active_degree ? 1.0f : 0.0f;
            out(offset + 0) = m * (2.5033429417967046f * xy * (x2 - y2));
            out(offset + 1) = m * (1.7701307697799304f * yz * (-3.0f * x2 + y2));
            out(offset + 2) = m * (0.94617469575756008f * xy * (7.0f * z2 - 1.0f));
            out(offset + 3) = m * (0.66904654355728921f * yz * (3.0f - 7.0f * z2));
            out(offset + 4) = m * (-3.1735664074561294f * z2 + 3.7024941420321507f * z4 + 0.31735664074561293f);
            out(offset + 5) = m * (0.66904654355728921f * xz * (3.0f - 7.0f * z2));
            out(offset + 6) = m * (0.47308734787878004f * (x2 - y2) * (7.0f * z2 - 1.0f));
            out(offset + 7) = m * (1.7701307697799304f * xz * (-x2 + 3.0f * y2));
            out(offset + 8) = m * (-3.7550144126950569f * x2 * y2 + 0.62583573544917614f * x4 + 0.62583573544917614f * y4);
            offset += 9;
        }
        if (JIT_SH_DEGREE_MASK & (1 << 5)) {
            const float m = degree_rank++ < active_degree ? 1.0f : 0.0f;
            out(offset + 0) = m * (0.65638205684017015f * y * (10.0f * x2 * y2 - 5.0f * x4 - y4));
            out(offset + 1) = m * (8.3026492595241645f * xy * z * (x2 - y2));
            out(offset + 2) = m * (-0.48923829943525038f * y * (3.0f * x2 - y2) * (9.0f * z2 - 1.0f));
            out(offset + 3) = m * (4.7935367849733241f * xy * z * (3.0f * z2 - 1.0f));
            out(offset + 4) = m * (0.45294665119569694f * y * (14.0f * z2 - 21.0f * z4 - 1.0f));
            out(offset + 5) = m * (0.1169503224534236f * z * (-70.0f * z2 + 63.0f * z4 + 15.0f));
            out(offset + 6) = m * (0.45294665119569694f * x * (14.0f * z2 - 21.0f * z4 - 1.0f));
            out(offset + 7) = m * (2.3967683924866621f * z * (x2 - y2) * (3.0f * z2 - 1.0f));
            out(offset + 8) = m * (-0.48923829943525038f * x * (x2 - 3.0f * y2) * (9.0f * z2 - 1.0f));
            out(offset + 9) = m * (2.0756623148810411f * z * (-6.0f * x2 * y2 + x4 + y4));
            out(offset + 10) = m * (0.65638205684017015f * x * (10.0f * x2 * y2 - x4 - 5.0f * y4));
            offset += 11;
        }
        if (JIT_SH_DEGREE_MASK & (1 << 6)) {
            const float m = degree_rank++ < active_degree ? 1.0f : 0.0f;
            out(offset + 0) = m * (1.3663682103838286f * xy * (-10.0f * x2 * y2 + 3.0f * x4 + 3.0f * y4));
            out(offset + 1) = m * (2.3666191622317521f * yz * (10.0f * x2 * y2 - 5.0f * x4 - y4));
            out(offset + 2) = m * (2.0182596029148963f * xy * (x2 - y2) * (11.0f * z2 - 1.0f));
            out(offset + 3) = m * (-0.92120525951492349f * yz * (3.0f * x2 - y2) * (11.0f * z2 - 3.0f));
            out(offset + 4) = m * (0.92120525951492349f * xy * (-18.0f * z2 + 33.0f * z4 + 1.0f));
            out(offset + 5) = m * (0.58262136251873131f * yz * (30.0f * z2 - 33.0f * z4 - 5.0f));
            out(offset + 6) = m * (6.6747662381009842f * z2 - 20.024298714302954f * z4 + 14.684485723822165f * z6 - 0.31784601133814211f);
            out(offset + 7) = m * (0.58262136251873131f * xz * (30.0f * z2 - 33.0f * z4 - 5.0f));
            out(offset + 8) = m * (0.46060262975746175f * (x2 - y2) * (11.0f * z2 * (3.0f * z2 - 1.0f) - 7.0f * z2 + 1.0f));
            out(offset + 9) = m * (-0.92120525951492349f * xz * (x2 - 3.0f * y2) * (11.0f * z2 - 3.0f));
            out(offset + 10) = m * (0.50456490072872406f * (11.0f * z2 - 1.0f) * (-6.0f * x2 * y2 + x4 + y4));
            out(offset + 11) = m * (2.3666191622317521f * xz * (10.0f * x2 * y2 - x4 - 5.0f * y4));
            out(offset + 12) = m * (10.247761577878714f * x2 * y4 - 10.247761577878714f * x4 * y2 + 0.6831841051919143f * x6 - 0.6831841051919143f * y6);
            offset += 13;
        }
    }

    // backpropagates gradients from the configured sh degree values to the unit direction
    // derivatives taken from tiny-cuda-nn's sh_enc_grad, degree layout matches eval_sh_degrees
    template <typename GRAD_T>
    __device__ inline float3 eval_sh_degrees_grad(const float3& dir, const uint32_t active_degree, GRAD_T&& dL_dsh_values) {
        const float x = dir.x, y = dir.y, z = dir.z;
        const float xy = x * y, xz = x * z, yz = y * z, x2 = x * x, y2 = y * y, z2 = z * z;
        const float x4 = x2 * x2, y4 = y2 * y2, z4 = z2 * z2;
        float3 d = ::make_float3(0.0f, 0.0f, 0.0f);
        uint32_t offset = 0;
        uint32_t degree_rank = 0;
        if (JIT_SH_DEGREE_MASK & (1 << 0)) offset += 1;
        if (JIT_SH_DEGREE_MASK & (1 << 1)) {
            if (degree_rank++ < active_degree) {
                d.y += dL_dsh_values(offset + 0) * (-0.48860251190291992f);
                d.z += dL_dsh_values(offset + 1) * (0.48860251190291992f);
                d.x += dL_dsh_values(offset + 2) * (-0.48860251190291992f);
            }
            offset += 3;
        }
        if (JIT_SH_DEGREE_MASK & (1 << 2)) {
            if (degree_rank++ < active_degree) {
                d.x += dL_dsh_values(offset + 0) * (1.0925484305920792f * y);
                d.y += dL_dsh_values(offset + 0) * (1.0925484305920792f * x);
                d.y += dL_dsh_values(offset + 1) * (-1.0925484305920792f * z);
                d.z += dL_dsh_values(offset + 1) * (-1.0925484305920792f * y);
                d.z += dL_dsh_values(offset + 2) * (1.8923493915151202f * z);
                d.x += dL_dsh_values(offset + 3) * (-1.0925484305920792f * z);
                d.z += dL_dsh_values(offset + 3) * (-1.0925484305920792f * x);
                d.x += dL_dsh_values(offset + 4) * (1.0925484305920792f * x);
                d.y += dL_dsh_values(offset + 4) * (-1.0925484305920792f * y);
            }
            offset += 5;
        }
        if (JIT_SH_DEGREE_MASK & (1 << 3)) {
            if (degree_rank++ < active_degree) {
                d.x += dL_dsh_values(offset + 0) * (-3.5402615395598609f * xy);
                d.y += dL_dsh_values(offset + 0) * (-1.7701307697799304f * x2 + 1.7701307697799304f * y2);
                d.x += dL_dsh_values(offset + 1) * (2.8906114426405538f * yz);
                d.y += dL_dsh_values(offset + 1) * (2.8906114426405538f * xz);
                d.z += dL_dsh_values(offset + 1) * (2.8906114426405538f * xy);
                d.y += dL_dsh_values(offset + 2) * (0.45704579946446572f - 2.2852289973223288f * z2);
                d.z += dL_dsh_values(offset + 2) * (-4.5704579946446566f * yz);
                d.z += dL_dsh_values(offset + 3) * (5.597644988851731f * z2 - 1.1195289977703462f);
                d.x += dL_dsh_values(offset + 4) * (0.45704579946446572f - 2.2852289973223288f * z2);
                d.z += dL_dsh_values(offset + 4) * (-4.5704579946446566f * xz);
                d.x += dL_dsh_values(offset + 5) * (2.8906114426405538f * xz);
                d.y += dL_dsh_values(offset + 5) * (-2.8906114426405538f * yz);
                d.z += dL_dsh_values(offset + 5) * (1.4453057213202769f * x2 - 1.4453057213202769f * y2);
                d.x += dL_dsh_values(offset + 6) * (-1.7701307697799304f * x2 + 1.7701307697799304f * y2);
                d.y += dL_dsh_values(offset + 6) * (3.5402615395598609f * xy);
            }
            offset += 7;
        }
        if (JIT_SH_DEGREE_MASK & (1 << 4)) {
            if (degree_rank++ < active_degree) {
                d.x += dL_dsh_values(offset + 0) * (2.5033429417967046f * y * (3.0f * x2 - y2));
                d.y += dL_dsh_values(offset + 0) * (2.5033429417967046f * x * (x2 - 3.0f * y2));
                d.x += dL_dsh_values(offset + 1) * (-10.620784618679583f * xy * z);
                d.y += dL_dsh_values(offset + 1) * (5.3103923093397913f * z * (-x2 + y2));
                d.z += dL_dsh_values(offset + 1) * (1.7701307697799304f * y * (-3.0f * x2 + y2));
                d.x += dL_dsh_values(offset + 2) * (0.94617469575756008f * y * (7.0f * z2 - 1.0f));
                d.y += dL_dsh_values(offset + 2) * (0.94617469575756008f * x * (7.0f * z2 - 1.0f));
                d.z += dL_dsh_values(offset + 2) * (13.246445740605839f * xy * z);
                d.y += dL_dsh_values(offset + 3) * (0.66904654355728921f * z * (3.0f - 7.0f * z2));
                d.z += dL_dsh_values(offset + 3) * (2.0071396306718676f * y * (1.0f - 7.0f * z2));
                d.z += dL_dsh_values(offset + 4) * (14.809976568128603f * z * z2 - 6.3471328149122579f * z);
                d.x += dL_dsh_values(offset + 5) * (0.66904654355728921f * z * (3.0f - 7.0f * z2));
                d.z += dL_dsh_values(offset + 5) * (2.0071396306718676f * x * (1.0f - 7.0f * z2));
                d.x += dL_dsh_values(offset + 6) * (0.94617469575756008f * x * (7.0f * z2 - 1.0f));
                d.y += dL_dsh_values(offset + 6) * (0.94617469575756008f * y * (1.0f - 7.0f * z2));
                d.z += dL_dsh_values(offset + 6) * (6.6232228703029197f * z * (x2 - y2));
                d.x += dL_dsh_values(offset + 7) * (5.3103923093397913f * z * (-x2 + y2));
                d.y += dL_dsh_values(offset + 7) * (10.620784618679583f * xy * z);
                d.z += dL_dsh_values(offset + 7) * (1.7701307697799304f * x * (-x2 + 3.0f * y2));
                d.x += dL_dsh_values(offset + 8) * (2.5033429417967046f * x * (x2 - 3.0f * y2));
                d.y += dL_dsh_values(offset + 8) * (2.5033429417967046f * y * (-3.0f * x2 + y2));
            }
            offset += 9;
        }
        if (JIT_SH_DEGREE_MASK & (1 << 5)) {
            if (degree_rank++ < active_degree) {
                d.x += dL_dsh_values(offset + 0) * (13.127641136803401f * xy * (-x2 + y2));
                d.y += dL_dsh_values(offset + 0) * (19.6914617052051f * x2 * y2 - 3.2819102842008503f * x4 - 3.2819102842008503f * y4);
                d.x += dL_dsh_values(offset + 1) * (8.3026492595241645f * yz * (3.0f * x2 - y2));
                d.y += dL_dsh_values(offset + 1) * (8.3026492595241645f * xz * (x2 - 3.0f * y2));
                d.z += dL_dsh_values(offset + 1) * (8.3026492595241645f * xy * (x2 - y2));
                d.x += dL_dsh_values(offset + 2) * (2.9354297966115022f * xy * (1.0f - 9.0f * z2));
                d.y += dL_dsh_values(offset + 2) * (-1.4677148983057511f * (x2 - y2) * (9.0f * z2 - 1.0f));
                d.z += dL_dsh_values(offset + 2) * (8.8062893898345074f * yz * (-3.0f * x2 + y2));
                d.x += dL_dsh_values(offset + 3) * (4.7935367849733241f * yz * (3.0f * z2 - 1.0f));
                d.y += dL_dsh_values(offset + 3) * (4.7935367849733241f * xz * (3.0f * z2 - 1.0f));
                d.z += dL_dsh_values(offset + 3) * (4.7935367849733241f * xy * (9.0f * z2 - 1.0f));
                d.y += dL_dsh_values(offset + 4) * (6.3412531167397574f * z2 - 9.5118796751096362f * z4 - 0.45294665119569694f);
                d.z += dL_dsh_values(offset + 4) * (12.682506233479513f * yz * (1.0f - 3.0f * z2));
                d.z += dL_dsh_values(offset + 5) * (-24.559567715218954f * z2 + 36.839351572828434f * z4 + 1.754254836801354f);
                d.x += dL_dsh_values(offset + 6) * (6.3412531167397574f * z2 - 9.5118796751096362f * z4 - 0.45294665119569694f);
                d.z += dL_dsh_values(offset + 6) * (12.682506233479513f * xz * (1.0f - 3.0f * z2));
                d.x += dL_dsh_values(offset + 7) * (4.7935367849733241f * xz * (3.0f * z2 - 1.0f));
                d.y += dL_dsh_values(offset + 7) * (4.7935367849733241f * yz * (1.0f - 3.0f * z2));
                d.z += dL_dsh_values(offset + 7) * (2.3967683924866621f * (x2 - y2) * (9.0f * z2 - 1.0f));
                d.x += dL_dsh_values(offset + 8) * (-13.209434084751759f * x2 * z2 + 1.4677148983057511f * x2 + 13.209434084751759f * y2 * z2 - 1.4677148983057511f * y2);
                d.y += dL_dsh_values(offset + 8) * (2.9354297966115022f * xy * (9.0f * z2 - 1.0f));
                d.z += dL_dsh_values(offset + 8) * (8.8062893898345074f * xz * (-x2 + 3.0f * y2));
                d.x += dL_dsh_values(offset + 9) * (8.3026492595241645f * xz * (x2 - 3.0f * y2));
                d.y += dL_dsh_values(offset + 9) * (8.3026492595241645f * yz * (-3.0f * x2 + y2));
                d.z += dL_dsh_values(offset + 9) * (-12.453973889286246f * x2 * y2 + 2.0756623148810411f * x4 + 2.0756623148810411f * y4);
                d.x += dL_dsh_values(offset + 10) * (19.6914617052051f * x2 * y2 - 3.2819102842008503f * x4 - 3.2819102842008503f * y4);
                d.y += dL_dsh_values(offset + 10) * (13.127641136803401f * xy * (x2 - y2));
            }
            offset += 11;
        }
        if (JIT_SH_DEGREE_MASK & (1 << 6)) {
            if (degree_rank++ < active_degree) {
                d.x += dL_dsh_values(offset + 0) * (4.0991046311514854f * y * (-10.0f * x2 * y2 + 5.0f * x4 + y4));
                d.y += dL_dsh_values(offset + 0) * (4.0991046311514854f * x * (-10.0f * x2 * y2 + x4 + 5.0f * y4));
                d.x += dL_dsh_values(offset + 1) * (47.332383244635047f * xy * z * (-x2 + y2));
                d.y += dL_dsh_values(offset + 1) * (11.833095811158762f * z * (6.0f * x2 * y2 - x4 - y4));
                d.z += dL_dsh_values(offset + 1) * (2.3666191622317521f * y * (10.0f * x2 * y2 - 5.0f * x4 - y4));
                d.x += dL_dsh_values(offset + 2) * (2.0182596029148963f * y * (3.0f * x2 - y2) * (11.0f * z2 - 1.0f));
                d.y += dL_dsh_values(offset + 2) * (2.0182596029148963f * x * (x2 - 3.0f * y2) * (11.0f * z2 - 1.0f));
                d.z += dL_dsh_values(offset + 2) * (44.401711264127719f * xy * z * (x2 - y2));
                d.x += dL_dsh_values(offset + 3) * (5.5272315570895412f * xy * z * (3.0f - 11.0f * z2));
                d.y += dL_dsh_values(offset + 3) * (-2.7636157785447706f * z * (x2 - y2) * (11.0f * z2 - 3.0f));
                d.z += dL_dsh_values(offset + 3) * (-2.7636157785447706f * y * (3.0f * x2 - y2) * (11.0f * z2 - 1.0f));
                d.x += dL_dsh_values(offset + 4) * (0.92120525951492349f * y * (-18.0f * z2 + 33.0f * z4 + 1.0f));
                d.y += dL_dsh_values(offset + 4) * (0.92120525951492349f * x * (-18.0f * z2 + 33.0f * z4 + 1.0f));
                d.z += dL_dsh_values(offset + 4) * (11.054463114179082f * xy * z * (11.0f * z2 - 3.0f));
                d.y += dL_dsh_values(offset + 5) * (0.58262136251873131f * z * (30.0f * z2 - 33.0f * z4 - 5.0f));
                d.z += dL_dsh_values(offset + 5) * (2.9131068125936568f * y * (18.0f * z2 - 33.0f * z4 - 1.0f));
                d.z += dL_dsh_values(offset + 6) * (2.6699064952403937f * z * (-30.0f * z2 + 33.0f * z4 + 5.0f));
                d.x += dL_dsh_values(offset + 7) * (0.58262136251873131f * z * (30.0f * z2 - 33.0f * z4 - 5.0f));
                d.z += dL_dsh_values(offset + 7) * (2.9131068125936568f * x * (18.0f * z2 - 33.0f * z4 - 1.0f));
                d.x += dL_dsh_values(offset + 8) * (0.92120525951492349f * x * (-18.0f * z2 + 33.0f * z4 + 1.0f));
                d.y += dL_dsh_values(offset + 8) * (0.92120525951492349f * y * (18.0f * z2 - 33.0f * z4 - 1.0f));
                d.z += dL_dsh_values(offset + 8) * (5.5272315570895412f * z * (x2 - y2) * (11.0f * z2 - 3.0f));
                d.x += dL_dsh_values(offset + 9) * (-2.7636157785447706f * z * (x2 - y2) * (11.0f * z2 - 3.0f));
                d.y += dL_dsh_values(offset + 9) * (5.5272315570895412f * xy * z * (11.0f * z2 - 3.0f));
                d.z += dL_dsh_values(offset + 9) * (-2.7636157785447706f * x * (x2 - 3.0f * y2) * (11.0f * z2 - 1.0f));
                d.x += dL_dsh_values(offset + 10) * (2.0182596029148963f * x * (x2 - 3.0f * y2) * (11.0f * z2 - 1.0f));
                d.y += dL_dsh_values(offset + 10) * (-2.0182596029148963f * y * (3.0f * x2 - y2) * (11.0f * z2 - 1.0f));
                d.z += dL_dsh_values(offset + 10) * (11.10042781603193f * z * (-6.0f * x2 * y2 + x4 + y4));
                d.x += dL_dsh_values(offset + 11) * (11.833095811158762f * z * (6.0f * x2 * y2 - x4 - y4));
                d.y += dL_dsh_values(offset + 11) * (47.332383244635047f * xy * z * (x2 - y2));
                d.z += dL_dsh_values(offset + 11) * (2.3666191622317521f * x * (10.0f * x2 * y2 - x4 - 5.0f * y4));
                d.x += dL_dsh_values(offset + 12) * (4.0991046311514854f * x * (-10.0f * x2 * y2 + x4 + 5.0f * y4));
                d.y += dL_dsh_values(offset + 12) * (4.0991046311514854f * y * (10.0f * x2 * y2 - 5.0f * x4 - y4));
            }
            offset += 13;
        }
        return d;
    }

    // zeroes the calling warp's forward context segment; warps culled before eval_model must call this before
    // exiting, because the backward pass unconditionally runs the mlp backward for every thread of the launch
    // grid and relies on zeroed activations to produce zero gradient contributions (0 x garbage may be NaN)
    __device__ inline void zero_warp_fwd_ctx(uint8_t* fwd_ctx) {
        // the segment is 32 * JIT_CTX_BYTES contiguous bytes, always a multiple of 16 and 16-byte aligned
        constexpr uint32_t n_segment_elements = 32 * JIT_CTX_BYTES / sizeof(float4);
        float4* segment = reinterpret_cast<float4*>(fwd_ctx);
        for (uint32_t i = threadIdx.x % 32; i < n_segment_elements; i += 32) {
            segment[i] = ::make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }

    // evaluates the residual mlp and returns the activated rgb residual, writing the raw outputs the
    // backward needs to recover the residual activation derivative; eval_model is warp-cooperative, so
    // the caller must invoke this from every lane of a surviving warp, before any per-thread exit
    __device__ inline float3 neural_residual(
        const AppearanceInputs& appearance,
        const float3& position,
        const float3& cam_position,
        uint8_t* __restrict__ fwd_ctx,
        tvec<__half, 3>& mlp_output,
        const uint32_t primitive_idx,
        const uint32_t appearance_degree)
    {
        if (appearance_degree == 0) {
            mlp_output = tvec<__half, 3>::zero();
            return ::make_float3(0.0f, 0.0f, 0.0f);
        }

        tvec<float, JIT_N_INPUT_DIMS> mlp_input;
        const float* residual_params = appearance.residual_params + primitive_idx * JIT_N_FEATURE_DIMS;
        const float3 base_input = JIT_BASE_INPUT ? appearance.base_colors[primitive_idx] : ::make_float3(0.0f, 0.0f, 0.0f);
        encode_features(
            [&](uint32_t i) -> float {
                if (!JIT_BASE_INPUT || i < JIT_N_FEATURE_DIMS) return residual_params[i];
                return i == JIT_N_FEATURE_DIMS ? base_input.x : (i == JIT_N_FEATURE_DIMS + 1 ? base_input.y : base_input.z);
            },
            [&](uint32_t i) -> float& { return mlp_input[i]; });
        const float3 direction = normalize(position - cam_position);
        eval_sh_degrees(direction, appearance_degree, [&](uint32_t i) -> float& { return mlp_input[n_encoded_feature_dims + i]; });
        mlp_output = eval_model(mlp_input, appearance.mlp_weights, fwd_ctx);
        return residual_activation(::make_float3(
            static_cast<float>(mlp_output[0]),
            static_cast<float>(mlp_output[1]),
            static_cast<float>(mlp_output[2])
        ));
    }

#if JIT_WITH_BACKWARD
    // backpropagates the (activation-corrected) color gradient through the residual mlp, accumulating the
    // parameter gradients and writing the feature (and, with JIT_BASE_INPUT, base color) gradients, and
    // returns the position gradient through the view direction; gradients are loss-scaled to avoid fp16
    // underflow and unscaled again on the way out
    // backward_eval_model reduces parameter gradients block-wide, so like the forward it must be reached by
    // every thread of the block: non-contributing lanes pass zero gradients instead of exiting
    __device__ inline float3 neural_residual_backward(
        const AppearanceInputs& appearance,
        const float3& grad_color,
        const AppearanceGradients& grad_appearance,
        const float3& position,
        const float3& cam_position,
        const uint8_t* __restrict__ fwd_ctx,
        const uint32_t primitive_idx,
        const uint32_t n_primitives,
        const uint32_t appearance_degree,
        const bool contributes)
    {
        if (appearance_degree == 0) return ::make_float3(0.0f, 0.0f, 0.0f);

#if JIT_RESIDUAL_ACTIVATION == 0
        const float3 dL_dmlp_output = grad_color;
#else
        // the residual activation derivative is recovered from the stored (pre-activation) mlp outputs
        const float3 mlp_output = contributes ? ::make_float3(
            __half2float(appearance.mlp_outputs[primitive_idx]),
            __half2float(appearance.mlp_outputs[n_primitives + primitive_idx]),
            __half2float(appearance.mlp_outputs[2 * n_primitives + primitive_idx])
        ) : ::make_float3(0.0f, 0.0f, 0.0f);
        const float3 dL_dmlp_output = grad_color * residual_activation_grad(residual_activation(mlp_output));
#endif
        tvec<__half, 3> dL_dy;
        dL_dy[0] = static_cast<__half>(JIT_LOSS_SCALE * dL_dmlp_output.x);
        dL_dy[1] = static_cast<__half>(JIT_LOSS_SCALE * dL_dmlp_output.y);
        dL_dy[2] = static_cast<__half>(JIT_LOSS_SCALE * dL_dmlp_output.z);
        auto dL_dinput = tvec<float, JIT_N_INPUT_DIMS>::zero();
        backward_eval_model(dL_dy, appearance.mlp_weights, fwd_ctx, grad_appearance.mlp_weights, &dL_dinput);
        if (!contributes) return ::make_float3(0.0f, 0.0f, 0.0f);

        // feature gradients through the optional frequency encoding
        constexpr float loss_scale_rcp = 1.0f / JIT_LOSS_SCALE;
        #pragma unroll
        for (uint32_t i = 0; i < JIT_N_FEATURE_DIMS; i++) {
            const float feature = appearance.residual_params[primitive_idx * JIT_N_FEATURE_DIMS + i];
            grad_appearance.residual_params[primitive_idx * JIT_N_FEATURE_DIMS + i] = loss_scale_rcp * encode_features_grad(
                feature, [&](uint32_t j) -> float { return dL_dinput[j]; }, i);
        }

        if (JIT_BASE_INPUT) {
            // second gradient path into the base values through their (encoded) mlp inputs
            const float3 base_input_values = appearance.base_colors[primitive_idx];
            grad_appearance.base_colors[primitive_idx] = grad_appearance.base_colors[primitive_idx] + loss_scale_rcp * ::make_float3(
                encode_features_grad(base_input_values.x, [&](uint32_t j) -> float { return dL_dinput[j]; }, JIT_N_FEATURE_DIMS),
                encode_features_grad(base_input_values.y, [&](uint32_t j) -> float { return dL_dinput[j]; }, JIT_N_FEATURE_DIMS + 1),
                encode_features_grad(base_input_values.z, [&](uint32_t j) -> float { return dL_dinput[j]; }, JIT_N_FEATURE_DIMS + 2)
            );
        }

        // position gradient through the sh degree values of the view direction
        float direction_norm_rcp;
        const float3 direction = normalize(position - cam_position, direction_norm_rcp);
        const float3 grad_direction = eval_sh_degrees_grad(direction, appearance_degree,
            [&](uint32_t i) -> float { return dL_dinput[n_encoded_feature_dims + i] * loss_scale_rcp; });
        return (grad_direction - dot(grad_direction, direction) * direction) * direction_norm_rcp;
    }
#endif

}
