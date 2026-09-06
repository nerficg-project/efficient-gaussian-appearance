precision highp float;

// PPISP screen-space pass for files of PPISP-trained models: exposure, color
// homography in (r, g, r+g+b) space, and the per-channel response curve of
// ppisp_math.cuh. The vignetting stays out: it is a lens artifact PPISP factors
// out of the radiance.

uniform highp sampler2D tScene;  // linear radiance
uniform float uExposure;         // 2^exposure
uniform mat3 uColorH;
uniform vec3 uCrfShape[3];       // toe, shoulder, gamma
uniform vec3 uCrfCurve[3];       // center, a, b

out vec4 outColor;

float responseCurve(float x, vec3 shape, vec3 curve) {
    x = clamp(x, 0.0, 1.0);
    float y = x <= curve.x
        ? curve.y * pow(x / curve.x, shape.x)
        : 1.0 - curve.z * pow((1.0 - x) / (1.0 - curve.x), shape.y);
    return pow(max(y, 0.0), shape.z);
}

void main() {
    vec3 rgb = texelFetch(tScene, ivec2(gl_FragCoord.xy), 0).rgb * uExposure;
    float intensity = rgb.r + rgb.g + rgb.b;
    vec3 rgi = uColorH * vec3(rgb.r, rgb.g, intensity);
    rgi *= intensity / (rgi.z + 1e-5);
    rgb = vec3(rgi.x, rgi.y, rgi.z - rgi.x - rgi.y);
    outColor = vec4(
        responseCurve(rgb.r, uCrfShape[0], uCrfCurve[0]),
        responseCurve(rgb.g, uCrfShape[1], uCrfCurve[1]),
        responseCurve(rgb.b, uCrfShape[2], uCrfCurve[2]), 1.0);
}
