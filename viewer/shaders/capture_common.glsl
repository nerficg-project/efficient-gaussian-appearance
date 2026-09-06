// Shared scaffolding of the capture-pass color evaluation. main.js splices the
// active appearance model's chunk (eval_*.glsl) at the marker below and injects
// the result into capture.frag and capture.vert (the same code must run in
// either stage, see chooseCaptureStage). Each chunk defines
//     vec3 residualColor(ivec2 texel, vec3 dir)
// and this file adds the shared uniforms, helpers, and captureColor(), which
// composes color_activation(base + residual).
//
// Injected defines: N_PARAM_TEX, COLOR_ACT (0 relu / 1 softplus(β10) / 2
// sigmoid(4x) / 3 satexp / 4 hardsigmoid / 5 none), BASE_ACT (0 identity / 1
// exp(3x)), COLOR_SHIFT (0.5 with the exp base for the shifted color
// activations, else 0.0), RENDER_BASE / RENDER_RESIDUAL (debug switches),
// COLOR_MAX (1.0, or the fp16 maximum for PPISP files with an unbounded color
// activation, whose cache keeps the unclamped radiance).

uniform highp usampler2D splatData;
// per-splat residual parameters, layout per appearance model (see the chunks)
#if N_PARAM_TEX > 0
uniform highp usampler2D paramTex0;
#endif
#if N_PARAM_TEX > 1
uniform highp usampler2D paramTex1;
#endif
#if N_PARAM_TEX > 2
uniform highp usampler2D paramTex2;
#endif
#if N_PARAM_TEX > 3
uniform highp usampler2D paramTex3;
#endif
#if N_PARAM_TEX > 4
uniform highp usampler2D paramTex4;
#endif
#if N_PARAM_TEX > 5
uniform highp usampler2D paramTex5;
#endif
#if N_PARAM_TEX > 6
uniform highp usampler2D paramTex6;
#endif
#if N_PARAM_TEX > 7
uniform highp usampler2D paramTex7;
#endif
uniform vec3 cameraPos;
uniform vec3 cameraFwd; // normalized camera forward, for the behind-plane early-out
uniform vec2 uBase; // x = scale, y = offset: base pre-activation d = code8 * scale + offset

// softplus with beta 10, linear above x > 2 (the numerically stable form)
vec3 softplus10(vec3 x) {
    vec3 soft = log(1.0 + exp(10.0 * min(x, vec3(2.0)))) * 0.1;
    return mix(soft, x, greaterThan(x, vec3(2.0)));
}

// the 0.5 shift of the shifted variants is pre-baked into the stored base with
// the identity base (COLOR_SHIFT 0.0) and applied here with the exp base
vec3 colorActivation(vec3 x) {
#if COLOR_ACT == 1
    return softplus10(x + vec3(COLOR_SHIFT));
#elif COLOR_ACT == 2
    return 1.0 / (1.0 + exp(-4.0 * x));
#elif COLOR_ACT == 3
    return 1.0 - exp(-x);
#elif COLOR_ACT == 4
    return clamp(x + vec3(COLOR_SHIFT), 0.0, 1.0);
#elif COLOR_ACT == 5
    return x + vec3(COLOR_SHIFT);
#else
    return max(x + vec3(COLOR_SHIFT), vec3(0.0));
#endif
}

#if N_PARAM_TEX > 0
// unpacks the splat's fp16 parameter stream (8 values per RGBA32UI texel); the
// SH chunk decodes its packed texels itself
void loadParams(ivec2 uv, out float p[N_PARAM_TEX * 8]) {
    uvec4 t;
    vec2 v;
    t = texelFetch(paramTex0, uv, 0);
    v = unpackHalf2x16(t.x); p[0] = v.x; p[1] = v.y;
    v = unpackHalf2x16(t.y); p[2] = v.x; p[3] = v.y;
    v = unpackHalf2x16(t.z); p[4] = v.x; p[5] = v.y;
    v = unpackHalf2x16(t.w); p[6] = v.x; p[7] = v.y;
#if N_PARAM_TEX > 1
    t = texelFetch(paramTex1, uv, 0);
    v = unpackHalf2x16(t.x); p[8] = v.x; p[9] = v.y;
    v = unpackHalf2x16(t.y); p[10] = v.x; p[11] = v.y;
    v = unpackHalf2x16(t.z); p[12] = v.x; p[13] = v.y;
    v = unpackHalf2x16(t.w); p[14] = v.x; p[15] = v.y;
#endif
#if N_PARAM_TEX > 2
    t = texelFetch(paramTex2, uv, 0);
    v = unpackHalf2x16(t.x); p[16] = v.x; p[17] = v.y;
    v = unpackHalf2x16(t.y); p[18] = v.x; p[19] = v.y;
    v = unpackHalf2x16(t.z); p[20] = v.x; p[21] = v.y;
    v = unpackHalf2x16(t.w); p[22] = v.x; p[23] = v.y;
#endif
#if N_PARAM_TEX > 3
    t = texelFetch(paramTex3, uv, 0);
    v = unpackHalf2x16(t.x); p[24] = v.x; p[25] = v.y;
    v = unpackHalf2x16(t.y); p[26] = v.x; p[27] = v.y;
    v = unpackHalf2x16(t.z); p[28] = v.x; p[29] = v.y;
    v = unpackHalf2x16(t.w); p[30] = v.x; p[31] = v.y;
#endif
#if N_PARAM_TEX > 4
    t = texelFetch(paramTex4, uv, 0);
    v = unpackHalf2x16(t.x); p[32] = v.x; p[33] = v.y;
    v = unpackHalf2x16(t.y); p[34] = v.x; p[35] = v.y;
    v = unpackHalf2x16(t.z); p[36] = v.x; p[37] = v.y;
    v = unpackHalf2x16(t.w); p[38] = v.x; p[39] = v.y;
#endif
#if N_PARAM_TEX > 5
    t = texelFetch(paramTex5, uv, 0);
    v = unpackHalf2x16(t.x); p[40] = v.x; p[41] = v.y;
    v = unpackHalf2x16(t.y); p[42] = v.x; p[43] = v.y;
    v = unpackHalf2x16(t.z); p[44] = v.x; p[45] = v.y;
    v = unpackHalf2x16(t.w); p[46] = v.x; p[47] = v.y;
#endif
#if N_PARAM_TEX > 6
    t = texelFetch(paramTex6, uv, 0);
    v = unpackHalf2x16(t.x); p[48] = v.x; p[49] = v.y;
    v = unpackHalf2x16(t.y); p[50] = v.x; p[51] = v.y;
    v = unpackHalf2x16(t.z); p[52] = v.x; p[53] = v.y;
    v = unpackHalf2x16(t.w); p[54] = v.x; p[55] = v.y;
#endif
#if N_PARAM_TEX > 7
    t = texelFetch(paramTex7, uv, 0);
    v = unpackHalf2x16(t.x); p[56] = v.x; p[57] = v.y;
    v = unpackHalf2x16(t.y); p[58] = v.x; p[59] = v.y;
    v = unpackHalf2x16(t.z); p[60] = v.x; p[61] = v.y;
    v = unpackHalf2x16(t.w); p[62] = v.x; p[63] = v.y;
#endif
}
#endif

// @inject residual_eval

// capture-pass output for the splat at `texel`: rgb = view-dependent color, a = opacity
vec4 captureColor(ivec2 texel) {
    uvec4 p = texelFetch(splatData, texel, 0);

    // pre-activation base (range-coded rgb8) + opacity (a8)
    vec3 d = vec3(uvec3(p.x & 0xFFu, (p.x >> 8u) & 0xFFu, (p.x >> 16u) & 0xFFu))
        * uBase.x + uBase.y;
    float opacity = float((p.x >> 24u) & 0xFFu) / 255.0;

#if BASE_ACT == 1
    vec3 baseTerm = exp(3.0 * d); // exp base activation
#else
    vec3 baseTerm = d;            // identity base activation (0.5 shift pre-baked)
#endif
    vec3 baseColor = colorActivation(baseTerm);

#if RENDER_RESIDUAL
    vec3 pos = vec3(unpackHalf2x16(p.y), unpackHalf2x16(p.z).x);
    vec3 rel = pos - cameraPos;
    // behind-plane early-out: the render pass culls center-behind splats anyway
    vec3 full = dot(rel, cameraFwd) <= 0.0
        ? baseColor
        : colorActivation(baseTerm + residualColor(texel, normalize(rel)));
#else
    vec3 full = baseColor; // base color only
#endif

#if RENDER_BASE
    vec3 rgb = full;
#else
    vec3 rgb = abs(full - baseColor); // residual color only
#endif

    return vec4(clamp(rgb, 0.0, COLOR_MAX), opacity);
}
