// Decode helpers for the packed splat layout (pure functions, spliced into render_ewa.vert).

const float LN_SCALE_MIN = -12.0;
const float LN_SCALE_MAX = 9.0;

// word0: r8 | g8<<8 | b8<<16 | a8<<24  ->  normalized RGBA
vec4 unpackRgba8(uint word0) {
    return vec4(
        float(word0 & 0xFFu),
        float((word0 >> 8u) & 0xFFu),
        float((word0 >> 16u) & 0xFFu),
        float((word0 >> 24u) & 0xFFu)
    ) / 255.0;
}

// word3 low 3 bytes: per-axis log-scale codes (0 = sentinel zero) -> linear scales
vec3 decodeScales(uint word3) {
    uvec3 u = uvec3(word3 & 0xFFu, (word3 >> 8u) & 0xFFu, (word3 >> 16u) & 0xFFu);
    float lnStep = (LN_SCALE_MAX - LN_SCALE_MIN) / 254.0;
    return vec3(
        u.x == 0u ? 0.0 : exp(LN_SCALE_MIN + float(u.x - 1u) * lnStep),
        u.y == 0u ? 0.0 : exp(LN_SCALE_MIN + float(u.y - 1u) * lnStep),
        u.z == 0u ? 0.0 : exp(LN_SCALE_MIN + float(u.z - 1u) * lnStep)
    );
}

// The 24-bit quaternion code is split across word2 (bytes 0,1) and word3 (byte 2).
uint quatCode(uint word2, uint word3) {
    return ((word2 >> 16u) & 0xFFu) | (((word2 >> 24u) & 0xFFu) << 8u) | (((word3 >> 24u) & 0xFFu) << 16u);
}

// Folded-octahedral axis (xy, 8+8 bit) + 8-bit angle  ->  unit quaternion
vec4 decodeQuatOctXy88R8(uint packed) {
    float pu = float(packed & 0xFFu) / 255.0 * 2.0 - 1.0;
    float pv = float((packed >> 8u) & 0xFFu) / 255.0 * 2.0 - 1.0;
    float theta = float((packed >> 16u) & 0xFFu) / 255.0 * 3.14159265358979;

    vec3 axis = vec3(pu, pv, 1.0 - abs(pu) - abs(pv));
    if (axis.z < 0.0)
        axis.xy = (1.0 - abs(axis.yx)) * sign(axis.xy);
    axis = normalize(axis);

    float halfTheta = theta * 0.5;
    return vec4(axis * sin(halfTheta), cos(halfTheta));
}
