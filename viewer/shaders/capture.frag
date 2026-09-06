precision highp float;

// Fragment-stage capture: one texel per splat (splatData layout) into an RGBA8
// target, run only when the camera moves. The evaluation is capture_common.glsl
// plus the active residual chunk, shared with capture.vert.

// @inject color_eval

out vec4 outColor;

void main() {
    outColor = captureColor(ivec2(gl_FragCoord.xy));
}
