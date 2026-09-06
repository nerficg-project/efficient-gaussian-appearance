precision highp float;

// EWA splatting vertex stage: project each Gaussian's covariance to a 2D conic
// (Zwicker et al.) and emit its minimum-area bounding parallelogram (Cholesky
// factor of the conic). In quad coordinates the alpha cutoff is the unit circle,
// so the fragment discard `dot(quadPos, quadPos) > 1` is the alpha-floor test.
// Colors come from the colorCache written by the capture pass.

uniform highp usampler2D splatData;
uniform highp sampler2D colorCache;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;
uniform vec2 halfWH;
uniform float uMinAlpha; // alpha floor = exp(-N^2/2) for the N-sigma cutoff
uniform float uScaleModifier; // global gaussian scale multiplier (GUI slider)

// PROPER_AA (mip-splatting style): 0.1 dilation instead of 0.3, opacity scaled
// by sqrt(det_raw / det_dilated) to compensate
#ifdef PROPER_AA
const float DILATION = 0.1;
#else
const float DILATION = 0.3;
#endif

in vec2 position;
in float splatIndex; // per-instance draw order -> splat id (float is exact below 2^24)

flat out vec4 rgba;      // per-splat constants: flat skips interpolation
out vec2 quadPos;        // quad coords: the alpha cutoff is exactly the unit circle
flat out float cutoffSq; // per-gaussian squared cutoff: rho^2 = 2 ln(alpha0 / uMinAlpha)

// @inject splat_decode

mat3 createRS(vec3 s, vec4 q) {
    float x = q.x, y = q.y, z = q.z, w = q.w;
    float x2 = 2.0 * x, y2 = 2.0 * y, z2 = 2.0 * z;
    float xx = x * x2, yy = y * y2, zz = z * z2;
    float xy = x * y2, xz = x * z2, yz = y * z2;
    float wx = w * x2, wy = w * y2, wz = w * z2;
    return mat3(
        s.x * vec3(1.0 - (yy + zz), xy + wz, xz - wy),
        s.y * vec3(xy - wz, 1.0 - (xx + zz), yz + wx),
        s.z * vec3(xz + wy, yz - wx, 1.0 - (xx + yy))
    );
}

void main () {
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);

    int splatId = int(splatIndex);
    int row = splatId / 2048;
    int col = splatId % 2048;
    uvec4 p = texelFetch(splatData, ivec2(col, row), 0);
    uint word1 = p.y, word2 = p.z, word3 = p.w;

    vec3 worldMean = vec3(unpackHalf2x16(word1), unpackHalf2x16(word2).x);
    vec3 viewMean = (viewMatrix * vec4(worldMean, 1.0)).xyz;
    if (viewMean.z >= 0.0)
        return;
    vec4 clipMean = projectionMatrix * vec4(viewMean, 1.0);
    if (abs(clipMean.z) >= clipMean.w)
        return;

    rgba = texelFetch(colorCache, ivec2(col, row), 0);

    // peak opacity already below the floor (the PROPER_AA compensation only lowers it)
    if (rgba.a < uMinAlpha)
        return;

    quadPos = position;

    vec3 scales = decodeScales(word3) * uScaleModifier;
    vec4 quaternion = decodeQuatOctXy88R8(quatCode(word2, word3));
    mat3 RS = createRS(scales, quaternion);
    mat3 M = mat3(viewMatrix) * RS;
    mat3 cov3D = M * transpose(M);

    vec2 focal = halfWH * vec2(projectionMatrix[0][0], projectionMatrix[1][1]);
    vec2 xyClip = 1.3 * halfWH / focal;
    vec2 J1 = focal / viewMean.z;
    vec2 J2 = -J1 * clamp(viewMean.xy / viewMean.z, -xyClip, xyClip);
    mat3 J = mat3(
        J1.x, 0.0, 0.0,
        0.0, J1.y, 0.0,
        J2.x, J2.y, 0.0
    );
    mat3 cov2D = J * cov3D * transpose(J);
    float a = cov2D[0][0] + DILATION;
    float b = cov2D[0][1];
    float c = cov2D[1][1] + DILATION;
    float det = a * c - b * b;
    if (det < 1e-8)
        return;

#ifdef PROPER_AA
    // opacity compensation: scale by the density lost to the dilation kernel
    float detRaw = cov2D[0][0] * cov2D[1][1] - cov2D[0][1] * cov2D[0][1];
    rgba.a *= sqrt(max(detRaw / det, 0.0));
    if (rgba.a < uMinAlpha)
        return;
#endif

    // radius where this gaussian's alpha decays to uMinAlpha: rho^2 = 2 ln(alpha0 / uMinAlpha)
    cutoffSq = 2.0 * log(max(rgba.a / uMinAlpha, 1.0));
    float stdDevCutoff = sqrt(cutoffSq);

    // minimum-area bounding parallelogram via the Cholesky factor of [[a, b], [b, c]]
    float e1 = sqrt(a);
    float l21 = b / e1;
    float e2 = sqrt(max(c - l21 * l21, 0.0));
    vec2 pixelOffset = stdDevCutoff * (position.x * vec2(e1, l21) + position.y * vec2(0.0, e2));
    vec2 ndcOffset = pixelOffset / halfWH;
    vec2 ndcMean = clipMean.xy / clipMean.w;
    gl_Position = vec4((ndcMean + ndcOffset) * clipMean.w, clipMean.zw);
}
