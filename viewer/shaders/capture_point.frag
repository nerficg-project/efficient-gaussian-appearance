precision highp float;

// fragment stage of the vertex-variant capture: the color is already computed

flat in vec4 vColor;
out vec4 outColor;

void main() { outColor = vColor; }
