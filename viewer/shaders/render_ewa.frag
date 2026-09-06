precision highp float;

// EWA splatting fragment stage: the quad is the conic's Cholesky parallelogram
// sized at the alpha cutoff, so the unit-circle test is the alpha-floor test.
// Composited back-to-front with NormalBlending.

flat in vec4 rgba;
in vec2 quadPos;
flat in float cutoffSq; // rho^2 = 2 ln(alpha0 / uMinAlpha)

out vec4 fragColor;

void main () {
    float r2 = dot(quadPos, quadPos); // squared Mahalanobis distance / cutoffSq
    if (r2 > 1.0) // outside the cutoff ellipse: evaluated alpha below the floor
        discard;

    float alpha = rgba.a * exp(-0.5 * cutoffSq * r2);
    fragColor = vec4(rgba.rgb, alpha); // composited back-to-front by the sort order
}
