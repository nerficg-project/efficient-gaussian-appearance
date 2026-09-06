// Neural appearance model residual chunk (spliced into capture_common.glsl).
//
// The bias-free MLP is evaluated as in SMERF: one weight texel = 4 consecutive
// input weights of one output neuron, texel[layerBase + j * outDim + o] =
// W[o, 4j:4j+4]; each layer accumulates input_block * mat4(4 texels) per 4-wide
// output block, the 3-row output layer reads 3 texels per input block. Weights
// come from a vec4 uniform array (WEIGHTS_UNIFORM, when they fit the device
// limit) or a 64-wide RGBA16F texture.
//
// Injected defines: BAKED (1 = layer 0 starts from the per-splat h0_static
// cache and takes only the view-direction inputs; 0 = full layout: encoded
// features in the textures, whole layer-0 matrix), N_L0_IN (stored layer-0
// input width), N_PARAM_TEX, N_FEATURES / SH0_INPUT (full layout only),
// N_NEURONS, N_HIDDEN, SH_DEGREE_MASK, RESIDUAL_ACT (0 none / 1 tanh / 2
// softplus(β10)), D1_OFF..D6_OFF (input-array offsets of the SH degrees).
//
// Color model (the wrapper composes color_activation(base + residual)):
//   h0       = W0 . [features, (SH_C0), sh_degrees(dir)]      (BAKED 0)
//            = h0_static + W0_dyn . sh_degrees(dir)           (BAKED 1)
//   residual = residual_activation(MLP layers 1.. applied to h0)
//   with dir = normalize(mean - cameraPos)

const float SH_C0 = 0.28209479177387814;

// weight texture layout constants (3-row output layer)
const int L0_TEXELS = (N_L0_IN / 4) * N_NEURONS;
const int HID_TEXELS = (N_NEURONS / 4) * N_NEURONS;
const int OUT_BASE = L0_TEXELS + (N_HIDDEN - 1) * HID_TEXELS;

// MLP_FP16: mediump weights and layer math (matches the reference's fp16
// evaluation, ~2x on mobile; desktop treats mediump as fp32). Input assembly
// stays highp.
#ifdef MLP_FP16
#define MLPP mediump
#else
#define MLPP highp
#endif

// weight texel accessor: uniform array or 64-wide texture (texel t at (t % 64, t / 64))
#ifdef WEIGHTS_UNIFORM
uniform MLPP vec4 uWeights[N_WEIGHT_TEXELS];
#define W(t) uWeights[t]
#else
uniform MLPP sampler2D mlpWeights;
#define W(t) texelFetch(mlpWeights, ivec2((t) & 63, (t) >> 6), 0)
#endif

#if RENDER_RESIDUAL
// view-dependent residual for direction d
vec3 residualColor(ivec2 uv, vec3 d) {
    // ---- layer-0 input vector ----
    float inp[N_L0_IN];
#if BAKED
    for (int i = 0; i < N_L0_IN; i++) inp[i] = 0.0; // only the padded tail matters
#else
    {
        uvec4 t0 = texelFetch(paramTex0, uv, 0);
        for (int i = 0; i < 4; i++) {
            vec2 v = unpackHalf2x16(t0[i]);
            if (2 * i < N_FEATURES) inp[2 * i] = v.x;
            if (2 * i + 1 < N_FEATURES) inp[2 * i + 1] = v.y;
        }
#if N_PARAM_TEX > 1
        uvec4 t1 = texelFetch(paramTex1, uv, 0);
        for (int i = 0; i < 4; i++) {
            vec2 v = unpackHalf2x16(t1[i]);
            if (8 + 2 * i < N_FEATURES) inp[8 + 2 * i] = v.x;
            if (8 + 2 * i + 1 < N_FEATURES) inp[8 + 2 * i + 1] = v.y;
        }
#endif
    }
#if SH0_INPUT
    inp[N_FEATURES] = SH_C0;
#endif
#endif

    // view-direction SH degrees (SH basis polynomials, ascending)
    float x = d.x, y = d.y, z = d.z;
    float xy = x * y, xz = x * z, yz = y * z;
    float x2 = x * x, y2 = y * y, z2 = z * z;
#if (SH_DEGREE_MASK & 2) != 0
    inp[D1_OFF + 0] = -0.48860251190291987 * y;
    inp[D1_OFF + 1] = 0.48860251190291987 * z;
    inp[D1_OFF + 2] = -0.48860251190291987 * x;
#endif
#if (SH_DEGREE_MASK & 4) != 0
    inp[D2_OFF + 0] = 1.0925484305920792 * xy;
    inp[D2_OFF + 1] = -1.0925484305920792 * yz;
    inp[D2_OFF + 2] = 0.94617469575755997 * z2 - 0.31539156525251999;
    inp[D2_OFF + 3] = -1.0925484305920792 * xz;
    inp[D2_OFF + 4] = 0.54627421529603959 * x2 - 0.54627421529603959 * y2;
#endif
#if (SH_DEGREE_MASK & 8) != 0
    inp[D3_OFF + 0] = 0.59004358992664352 * y * (-3.0 * x2 + y2);
    inp[D3_OFF + 1] = 2.8906114426405538 * xy * z;
    inp[D3_OFF + 2] = 0.45704579946446572 * y * (1.0 - 5.0 * z2);
    inp[D3_OFF + 3] = 0.3731763325901154 * z * (5.0 * z2 - 3.0);
    inp[D3_OFF + 4] = 0.45704579946446572 * x * (1.0 - 5.0 * z2);
    inp[D3_OFF + 5] = 1.4453057213202769 * z * (x2 - y2);
    inp[D3_OFF + 6] = 0.59004358992664352 * x * (-x2 + 3.0 * y2);
#endif
#if (SH_DEGREE_MASK & 112) != 0
    float x4 = x2 * x2, y4 = y2 * y2, z4 = z2 * z2;
#endif
#if (SH_DEGREE_MASK & 16) != 0
    inp[D4_OFF + 0] = 2.5033429417967046 * xy * (x2 - y2);
    inp[D4_OFF + 1] = 1.7701307697799304 * yz * (-3.0 * x2 + y2);
    inp[D4_OFF + 2] = 0.94617469575756008 * xy * (7.0 * z2 - 1.0);
    inp[D4_OFF + 3] = 0.66904654355728921 * yz * (3.0 - 7.0 * z2);
    inp[D4_OFF + 4] = -3.1735664074561294 * z2 + 3.7024941420321507 * z4 + 0.31735664074561293;
    inp[D4_OFF + 5] = 0.66904654355728921 * xz * (3.0 - 7.0 * z2);
    inp[D4_OFF + 6] = 0.47308734787878004 * (x2 - y2) * (7.0 * z2 - 1.0);
    inp[D4_OFF + 7] = 1.7701307697799304 * xz * (-x2 + 3.0 * y2);
    inp[D4_OFF + 8] = -3.7550144126950569 * x2 * y2 + 0.62583573544917614 * x4 + 0.62583573544917614 * y4;
#endif
#if (SH_DEGREE_MASK & 32) != 0
    inp[D5_OFF + 0] = 0.65638205684017015 * y * (10.0 * x2 * y2 - 5.0 * x4 - y4);
    inp[D5_OFF + 1] = 8.3026492595241645 * xy * z * (x2 - y2);
    inp[D5_OFF + 2] = -0.48923829943525038 * y * (3.0 * x2 - y2) * (9.0 * z2 - 1.0);
    inp[D5_OFF + 3] = 4.7935367849733241 * xy * z * (3.0 * z2 - 1.0);
    inp[D5_OFF + 4] = 0.45294665119569694 * y * (14.0 * z2 - 21.0 * z4 - 1.0);
    inp[D5_OFF + 5] = 0.1169503224534236 * z * (-70.0 * z2 + 63.0 * z4 + 15.0);
    inp[D5_OFF + 6] = 0.45294665119569694 * x * (14.0 * z2 - 21.0 * z4 - 1.0);
    inp[D5_OFF + 7] = 2.3967683924866621 * z * (x2 - y2) * (3.0 * z2 - 1.0);
    inp[D5_OFF + 8] = -0.48923829943525038 * x * (x2 - 3.0 * y2) * (9.0 * z2 - 1.0);
    inp[D5_OFF + 9] = 2.0756623148810411 * z * (-6.0 * x2 * y2 + x4 + y4);
    inp[D5_OFF + 10] = 0.65638205684017015 * x * (10.0 * x2 * y2 - x4 - 5.0 * y4);
#endif
#if (SH_DEGREE_MASK & 64) != 0
    float x6 = x4 * x2, y6 = y4 * y2, z6 = z4 * z2;
    inp[D6_OFF + 0] = 1.3663682103838286 * xy * (-10.0 * x2 * y2 + 3.0 * x4 + 3.0 * y4);
    inp[D6_OFF + 1] = 2.3666191622317521 * yz * (10.0 * x2 * y2 - 5.0 * x4 - y4);
    inp[D6_OFF + 2] = 2.0182596029148963 * xy * (x2 - y2) * (11.0 * z2 - 1.0);
    inp[D6_OFF + 3] = -0.92120525951492349 * yz * (3.0 * x2 - y2) * (11.0 * z2 - 3.0);
    inp[D6_OFF + 4] = 0.92120525951492349 * xy * (-18.0 * z2 + 33.0 * z4 + 1.0);
    inp[D6_OFF + 5] = 0.58262136251873131 * yz * (30.0 * z2 - 33.0 * z4 - 5.0);
    inp[D6_OFF + 6] = 6.6747662381009842 * z2 - 20.024298714302954 * z4 + 14.684485723822165 * z6 - 0.31784601133814211;
    inp[D6_OFF + 7] = 0.58262136251873131 * xz * (30.0 * z2 - 33.0 * z4 - 5.0);
    inp[D6_OFF + 8] = 0.46060262975746175 * (x2 - y2) * (11.0 * z2 * (3.0 * z2 - 1.0) - 7.0 * z2 + 1.0);
    inp[D6_OFF + 9] = -0.92120525951492349 * xz * (x2 - 3.0 * y2) * (11.0 * z2 - 3.0);
    inp[D6_OFF + 10] = 0.50456490072872406 * (11.0 * z2 - 1.0) * (-6.0 * x2 * y2 + x4 + y4);
    inp[D6_OFF + 11] = 2.3666191622317521 * xz * (10.0 * x2 * y2 - x4 - 5.0 * y4);
    inp[D6_OFF + 12] = 10.247761577878714 * x2 * y4 - 10.247761577878714 * x4 * y2 + 0.6831841051919143 * x6 - 0.6831841051919143 * y6;
#endif

    // ---- layer 0: from h0_static (baked) or zero (full) ----
    vec4 h[N_NEURONS / 4];
    vec4 act[N_NEURONS / 4];
#if BAKED
    {
        uvec4 t0 = texelFetch(paramTex0, uv, 0);
        h[0] = vec4(unpackHalf2x16(t0.x), unpackHalf2x16(t0.y));
#if N_NEURONS > 4
        h[1] = vec4(unpackHalf2x16(t0.z), unpackHalf2x16(t0.w));
#endif
#if N_PARAM_TEX > 1
        uvec4 t1 = texelFetch(paramTex1, uv, 0);
        h[2] = vec4(unpackHalf2x16(t1.x), unpackHalf2x16(t1.y));
#if N_NEURONS > 12
        h[3] = vec4(unpackHalf2x16(t1.z), unpackHalf2x16(t1.w));
#endif
#endif
    }
#else
    for (int i = 0; i < N_NEURONS / 4; i++) h[i] = vec4(0.0);
#endif
    for (int j = 0; j < N_L0_IN / 4; j++) {
        vec4 b = vec4(inp[4 * j], inp[4 * j + 1], inp[4 * j + 2], inp[4 * j + 3]);
        for (int i = 0; i < N_NEURONS; i += 4) {
            int t = j * N_NEURONS + i;
            h[i / 4] += b * mat4(W(t), W(t + 1), W(t + 2), W(t + 3));
        }
    }

    // ---- hidden layers 1 .. N_HIDDEN-1 ----
    for (int l = 1; l < N_HIDDEN; l++) {
        int base = L0_TEXELS + (l - 1) * HID_TEXELS;
        for (int i = 0; i < N_NEURONS / 4; i++) {
            act[i] = max(h[i], vec4(0.0)); // ReLU
            h[i] = vec4(0.0);
        }
        for (int j = 0; j < N_NEURONS / 4; j++) {
            for (int i = 0; i < N_NEURONS; i += 4) {
                int t = base + j * N_NEURONS + i;
                h[i / 4] += act[j] * mat4(W(t), W(t + 1), W(t + 2), W(t + 3));
            }
        }
    }

    // ---- output layer (3 dims: rgb residual; 3 texels per input block) ----
    vec3 mlpOut = vec3(0.0);
    for (int j = 0; j < N_NEURONS / 4; j++) {
        int t = OUT_BASE + j * 3;
        vec4 b = max(h[j], vec4(0.0));
        mlpOut += vec3(dot(b, W(t)), dot(b, W(t + 1)), dot(b, W(t + 2)));
    }

#if RESIDUAL_ACT == 2
    return softplus10(mlpOut);
#elif RESIDUAL_ACT == 1
    return tanh(mlpOut);
#else
    return mlpOut;
#endif
}
#endif
