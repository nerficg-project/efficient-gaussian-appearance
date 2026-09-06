// NASG appearance model residual chunk (spliced into capture_common.glsl).
//
// Normalized Anisotropic Spherical Gaussians, R(dir) = Σ_l pdf_l(dir) * weight_l:
//     K   = (<v, z> + 1) / 2
//     K_e = eps + a * <v, x>² / (1 - <v, z>²)
//     E   = K^K_e
//     pdf = exp(2λ (E K - 1)) * E * N
// The per-splat payload holds N_LOBES lobes of 12 fp16 values (frame vectors x
// and z, 2λ, anisotropy a, normalization constant N = λ√(1+a) / (2π(1+ε−e^{−2λ})),
// activated rgb weights), so only the direction-dependent response runs per
// frame. pdf is 1 at the +z pole and 0 at the −z pole, like the reference renderer.
//
// Injected define: N_LOBES.

const float EPS = 5e-6;              // stabilizes the exponent Ke
const float POLE_LIMIT = 0.99999988; // |<dir, frame z>| bound of the pole-free interval

#if RENDER_RESIDUAL
vec3 residualColor(ivec2 uv, vec3 d) {
    float p[N_PARAM_TEX * 8];
    loadParams(uv, p);
    vec3 res = vec3(0.0);
    for (int l = 0; l < N_LOBES; l++) {
        int b = 12 * l;
        vec3 frameX = vec3(p[b], p[b + 1], p[b + 2]);
        vec3 frameZ = vec3(p[b + 3], p[b + 4], p[b + 5]);
        float vz = dot(d, frameZ);
        float vx = dot(d, frameX);
        float vzc = clamp(vz, -POLE_LIMIT, POLE_LIMIT);
        float K = 0.5 * (vzc + 1.0);
        float Ke = EPS + p[b + 7] * vx * vx / (1.0 - vzc * vzc);
        float E = pow(K, Ke);
        bool valid_mask = abs(vz) < POLE_LIMIT;
        float pdf = exp(p[b + 6] * (E * K - 1.0)) * E * p[b + 8];
        pdf = vz >= POLE_LIMIT ? 1.0 : (valid_mask ? pdf : 0.0);
        res += pdf * vec3(p[b + 9], p[b + 10], p[b + 11]);
    }
    return res;
}
#endif
