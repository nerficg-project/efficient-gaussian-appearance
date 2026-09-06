precision highp float;

// Vertex-stage capture: one 1-pixel point per splat with all fetches and the
// color evaluation in the vertex shader, writing the same colorCache texel as
// capture.frag. Fallback for drivers whose fragment-stage integer texelFetch
// returns constants (Samsung Android).

// @inject color_eval

uniform vec2 uCacheSize; // colorCache texture size in texels

flat out vec4 vColor;

void main() {
    int id = gl_VertexID;
    ivec2 texel = ivec2(id % 2048, id / 2048);
    vColor = captureColor(texel);
    // position the point exactly on its texel center
    vec2 ndc = (vec2(texel) + 0.5) / uCacheSize * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
    gl_PointSize = 1.0;
}
