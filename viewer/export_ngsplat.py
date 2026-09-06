#!/usr/bin/env python3
"""
Export a trained FasterGSVDA checkpoint to a compact .ngsplat binary for the
web viewer in this directory. All five appearance models are supported —
SH, SV (spherical Voronoi), NASG, NASGabor, and Neural (residual mlp) —
selected by the checkpoint's MODEL.APPEARANCE.TYPE and marked in the header
flags (bits 3-4 + bit 6). The file contains the EXACT texture payloads the
viewer uploads (splat geometry, the per-splat residual parameters, and for
Neural the fp16 MLP weights in capture-shader texel order), so the viewer does
zero repacking.

Every model shares the composition color = COLOR_ACTIVATION(BASE_ACTIVATION(
base) + residual), and every activation of the reference implementation is
supported. The 0.5 gray shift of the shifted color activations
(none/relu/softplus/hardsigmoid) is pre-baked into the stored base colors when
the base activation is the identity; with the exp base the viewer applies it
in the shader (the shift sits outside the exp).
The per-splat residual payloads are optimized for the viewer's per-frame eval:

  - SH: raw coefficients quantized to the viewer's packed SH layout (7/8/6-bit
    levels for degrees 1/2/3, shared per-level max|coefficient| scales stored
    in the header). The shared residual activation applies to the band SUM, so
    the shader applies it after accumulation.
  - SV: activated sites — unit direction, exp'd temperature, activated site
    color (7 fp16 values per site); the shader only runs the softmax.
  - NASG: fully baked lobes — frame vectors x and z, 2*lambda, the anisotropy
    a, the precomputed normalization constant N, and the activated rgb weights
    (12 fp16 values per lobe); the shader evaluates just the
    direction-dependent response.
  - NASGabor: the NASG layout plus the Gabor frequency k (13 fp16 values per
    lobe).
  - Neural: fp16 h0_static (the baked view-independent share of mlp layer 0)
    plus the view-direction weight columns — see the layer-0 bake below.

Neural color model (must match the training code: the fused rasterizer kernels
in the CUDA backend's rasterization/rtc/{preprocess_forward.cu,
preprocess_common.cuh, appearance_neural.cuh} and the PyTorch reference in
../Appearance/{Base,Neural,utils}.py; the SV/NASG/NASGabor formulas mirror
appearance_{sv,nasg,nasgabor}.cuh the same way):

    d        = base_colors + 0.5 (the 0.5 shift the color activation applies; not for satexp)
    input    = [encoded features (F), SH_C0 (unless disabled), sh_degrees(dir)]
               (dir = normalize(mean - cam); with NEURAL.BASE_INPUT the raw base
                colors are 3 of the encoder's input dims and are stored encoded)
    out      = residual_mlp(input)                    (no biases, ReLU hidden)
    color    = COLOR_ACTIVATION(BASE_ACTIVATION(d) + RESIDUAL_ACTIVATION(out))

The encoded features and the constant SH_C0 input are view-INdependent, so their
share of the first mlp layer is baked at export time: the file stores, per splat,
the partial pre-activation h0_static = W0[:, features+SH_C0] . [features, SH_C0]
(one fp16 value per neuron) instead of the features, and only the view-direction
columns of W0 (zero-padded to a multiple of 4). The viewer starts layer 0 from
h0_static and evaluates just the direction inputs — for the default 32->16->16->3
architecture that removes 31% of the per-splat MACs at identical file size.
The bake is skipped when the h0_static cache (one value per neuron) would need
more per-splat textures than the features it replaces (fetches are the cost;
ties go to baked) or exceed the viewer's 2-texture cap — such files use the
full layout: stored encoded features + the full layer-0 matrix.

Notes:
  - appearance parameters live under MODEL.APPEARANCE; the checkpoint itself
    carries that config, so the yaml is only needed for
    RENDERER.PROPER_ANTIALIASING
  - the appearance schedule (mlp view-direction SH degrees / SH degrees /
    NASGabor lobes) is baked into the export: a checkpoint's active_degree
    gates the Neural weights (see below) and trims SH degrees / lobes
  - the 0.5 gray shift of the shifted color activations is pre-baked into the
    stored base colors (identity base only, see above), and with
    NEURAL.BASE_INPUT the raw base colors are 3 of the encoder's input dims
    (counting towards FEATURE_DIM)
  - COLOR_ACTIVATION none produces unbounded per-splat colors, which the
    viewer's rgba8 color cache clamps to [0, 1] per splat (the rasterizer
    clamps after blending; the bounded activations make the same
    approximation above 1)

PPISP: models trained with PPISP (MODEL.PPISP.USE) render raw radiance
that PPISP post-processes per image: exposure and an 8-parameter color
correction per frame (predicted by a controller CNN for novel views), then
per-camera vignetting and a per-channel camera response curve. The viewer
reproduces this in a screen-space pass without the vignetting (a lens artifact
PPISP factors out), so the export bakes camera 0's response curve and, per test
view, the controller's prediction on that view's rendering (obtained by
rendering the test views through the Framework). Off the test views the viewer
uses the exposure and color parameters of the training frame with the median
exposure. The numpy port below, vignetting included, is checked against the
CUDA output of the first views.

Binary layout (little-endian). P = texWidth * texHeight padded texel count.
─────────────
  "NGSPLAT\n"                8 bytes   magic
  numSplats        u32
  texWidth         u32                 2048
  texHeight        u32                 ceil(numSplats / texWidth)
  featureDim       u32                 SH: 0; SV: number of sites;
                                       NASG/NASGabor: number of lobes; Neural:
                                       ENCODED mlp feature dims (with the baked
                                       layer 0 informational — the features
                                       only exist inside h0_static)
  nFrequencies     u32                 Neural: k (informational); others: 0
  shBandMask       u32                 Neural: bit l set = view-dir SH degree l
                                       is an mlp input (1..6); SH: bit l set =
                                       coefficient degree l stored (contiguous
                                       1..D); others: 0
  colorActivation  u32                 0 relu | 1 softplus(beta 10) | 2 sigmoid(4x) |
                                       3 satexp | 4 hardsigmoid | 5 none
  residualActivation u32               0 none | 1 tanh | 2 softplus. Functional
                                       for Neural (applied to the mlp output) and
                                       SH (applied to the band sum);
                                       informational for SV/NASG/NASGabor (baked into
                                       the stored site colors / lobe weights)
  nNeurons         u32                 0 for non-Neural models
  nHiddenLayers    u32                 0 for non-Neural models
  flags            u32                 bit 0 = trained with PROPER_ANTIALIASING
                                       bit 1 = constant SH_C0 mlp input DISABLED (Neural)
                                       bit 2 = baked layer-0 layout (Neural;
                                       clear = full layout, see above)
                                       bits 3-4 + bit 6 (the high bit) =
                                       appearance model id: 1 SH | 2 SV |
                                       3 NASG | 4 NASGabor | 0 Neural
                                       bit 5 = BASE_ACTIVATION exp
                                       bit 7 = PPISP block present
  baseScale        f32                 d = baseOffset + code8 * baseScale
  baseOffset       f32
  shLevelScales    f32×3               SH FILES ONLY: max|coefficient| quantization
                                       scale per level (degrees 1/2/3; 0 = level absent)
  camCenter        f32×3               orbit target hint
  camUp            f32×3               world up hint
  camDistance      f32                 initial orbit distance hint
  nTestCameras     u32                 baked test-set viewpoints (0 if unavailable)
  testCameras      f32×18 × nTestCameras   per view: c2w rows (3×4 row-major, COLMAP
                                       convention: columns = right/down/forward, world
                                       space of the checkpoint), fx, fy, cx, cy,
                                       width, height (native dataset intrinsics; the
                                       viewer benchmark keeps them at a fixed viewport,
                                       720p by default, like scripts/inference.py
                                       --benchmark)
  PPISP BLOCK, only with flags bit 7 (models trained with PPISP; see below):
  crf              f32×18              camera 0, per channel: toe, shoulder, gamma,
                                       center (activated) and the curve's a, b
  defaultPpisp       f32×9               exposure and 8 color latents (b, r, g, n
                                       offsets) of the training frame with the
                                       median exposure, used off the test views
  viewPpisp          f32×9 × nTestCameras   per test view: the controller's exposure
                                       and 8 color latents
  nWeightTexels    u32                 total RGBA16F texels over all layers (0 for
                                       non-Neural models)
  weights          u16×4 × nWeightTexels    raw fp16 bits, capture texel order
  splatData        u32×4 × P           RGBA32UI: word0 base-color rgb8 + opacity a8,
                                       words1-3 pos fp16 / quat 24b / scales 3×8b
  paramTex[k]      u32×c × P           per-splat residual payload. SH: one
                                       texture per stored level — RG32UI (c=2)
                                       with 9×7-bit codes for degree 1, RGBA32UI
                                       with 15×8-bit codes for degree 2 and
                                       21×6-bit codes for degree 3 (fields
                                       bit-packed little-endian, signed,
                                       symmetric per-level scale).
                                       SV/NASG/NASGabor: the activated site/lobe
                                       values (SV 7 per site, NASG 12 /
                                       NASGabor 13 per lobe), 8 fp16 values per
                                       RGBA32UI texel, at most 8 textures.
                                       Neural: k = ceil(nNeurons/8) RGBA32UI
                                       textures (c=4), 8 fp16 baked h0_static
                                       values each

Weight texel order per layer (W row-major [out × in], in a multiple of 4):
  for j in range(in // 4): for o in range(out): texel = W[o, 4j:4j+4]
so the shader accumulates `inp_j * mat4(t, t+1, t+2, t+3)` over output blocks
(hidden layers; the 3-row output layer reads 3 texels per input block).
Layers: [n_neurons × dyn_in] (the view-direction columns of layer 0, zero-padded
to a multiple of 4), hidden [n × n]..., output trimmed to [3 × n].

Precision (inherent to the format): positions are fp16, so distant
background/floater Gaussians lose absolute accuracy; scales are 8-bit log codes
clamped to [e^-12, e^9], which rounds the degenerate near-zero axes MCMC
training produces up to e^-12 (still far below a pixel); base colors are rgb8
over their own range, so colors carry up to half a quantization step of error
(~0.01 for a typical scene) while the baked h0_static values stay fp16 (the
same rounding the fused kernels' fp16 mlp evaluation applies).

Appearance schedule: APPEARANCE.NEURAL.SH_DEGREES are enabled one by one during
training, and the rasterizer zeroes the not-yet-enabled degrees. The viewer has
no such gating, so a checkpoint with active_degree < number of degrees >= 1 is
exported with the corresponding input-layer weight columns zeroed, which is
numerically identical. Final checkpoints have all degrees enabled.

Usage
─────
  python src/Methods/FasterGSVDA/viewer/export_ngsplat.py <output_dir | checkpoint.pt>
      [--out FILE] [--dump-keys] [--center X Y Z] [--up X Y Z] [--distance D]
      [--no-test-cameras] [--no-ppisp]

Run it from the NeRFICG root so the relative DATASET.PATH in training_config.yaml
resolves and the test cameras can be baked. <output_dir> is a training run as
written by train.py, e.g. output/FasterGSVDA/<run>; the default output file is
<output_dir>/scene.ngsplat.

The orbit-camera hint in the header (up vector and distance) is derived from
the baked test cameras following mesh-splatting's export_web.py: up is the mean
of the per-camera up axes and distance the median camera-to-center distance.
The center stays the per-axis median of the splat means, which is robust to the
far-flung floaters MCMC scenes contain. --up / --distance / --center override.

Pass a FasterGSVDA training output directory (the one containing checkpoints/
and training_config.yaml): checkpoints/final.pt is loaded, the appearance config
is taken from the checkpoint, and RENDERER.PROPER_ANTIALIASING from the yaml.
A bare checkpoint path works too (no antialiasing flag then). If tiny-cuda-nn +
CUDA are available, the sliced weights are verified against the live tcnn model
on random inputs.

For the directory form, the NeRFICG Framework is used to reload the dataset the
model was trained on (DATASET.* in training_config.yaml, hence the NeRFICG root
as working directory) and the test-subset viewpoints are baked into the file for
the viewer's benchmark mode. Requires the dataset on disk; skipped with a
warning otherwise (or via --no-test-cameras).
"""

import argparse
import math
import struct
import sys

import numpy as np

SH_C0 = 0.28209479177387814
TEX_WIDTH = 2048
MIN_ALPHA = 1.0 / 255.0

# Log-scale encoding for Gaussian scales, matching the viewer's splat_decode.glsl.
LN_SCALE_MIN = -12.0
LN_SCALE_MAX = 9.0
LN_SCALE_ENCODE = 254.0 / (LN_SCALE_MAX - LN_SCALE_MIN)

# every activation of the reference implementation has a viewer counterpart
COLOR_ACTIVATION_CODES = {'relu': 0, 'softplus': 1, 'sigmoid': 2, 'satexp': 3, 'hardsigmoid': 4, 'none': 5}
# color activations that apply the 0.5 gray shift (act(x + 0.5)); the shift is
# baked into the stored base when the base activation is the identity, and
# applied in the shader when the exp base sits between the raw base and it
SHIFTED_COLOR_ACTIVATIONS = {'none', 'relu', 'softplus', 'hardsigmoid'}
# residual activation codes: functional for Neural (applied to the mlp output)
# and SH (applied to the band sum in the shader); informational for SV/NASG/NASGabor
# (baked into the stored site colors / lobe weights)
RESIDUAL_CODES = {'none': 0, 'tanh': 1, 'softplus': 2}

# appearance model id, header flags bits 3-4 plus bit 6 as the high bit; Neural
# is 0, the others follow the backend enum
APPEARANCE_MODEL_IDS = {'sh': 1, 'sv': 2, 'nasg': 3, 'nasgabor': 4, 'neural': 0}

# the viewer reads the per-splat residual parameters from at most 8 RGBA32UI
# textures holding 8 fp16 values each
PARAM_VALUE_CAP = 64

# activation constants of the SV/NASG/NASGabor kernels (values baked at export
# time must match the fused rasterizer's rtc appearance headers; NASG and
# NASGabor share identical constants)
SV_SITE_LENGTH_EPS = 1e-12
NASGABOR_COS_LIMIT = 0.999999
NASGABOR_SHAPE_LIMIT = 1e4
NASGABOR_FREQUENCY_SCALE = 20.0
NASGABOR_EPS = 5e-6
NASGABOR_EPS_NORM = 1e-8
NASGABOR_POLE_EPS = 1e-7
# exp'd SV temperatures are stored as fp16; the softmax is effectively a hard
# assignment far below this, so the clamp is invisible
FP16_SAFE_MAX = 60000.0

# PPISP constants (match ppisp_math.cuh): ZCA blocks mapping the 8 color latents to
# chromaticity offsets of the blue / red / green / neutral control points, and the
# lower bounds of the softplus-activated response curve parameters
PPISP_COLOR_PINV_BLOCKS = np.array([
    [[0.0480542, -0.0043631], [-0.0043631, 0.0481283]],
    [[0.0580570, -0.0179872], [-0.0179872, 0.0431061]],
    [[0.0433336, -0.0180537], [-0.0180537, 0.0580500]],
    [[0.0128369, -0.0034654], [-0.0034654, 0.0128158]],
], np.float32)
PPISP_CRF_MIN = (0.3, 0.3, 0.1)  # toe, shoulder, gamma

# fallback config when neither the checkpoint nor a training_config.yaml
# provides a value (matches the tracked fastergsvda_*.yaml configs); the keys
# after 'sh_degree' apply to the Neural appearance model only
CONFIG_DEFAULTS = {
    'appearance_type': 'neural',
    'base_activation': 'none',
    'residual_activation': 'tanh',
    'color_activation': 'relu',
    'sh_degree': 3,
    'n_sites': 7,
    'nasg_n_lobes': 1,
    'n_lobes': 1,
    'base_input': False,
    'n_frequencies': 1,
    'sh_degrees': [0, 1, 2, 3],
    'n_neurons': 16,
    'n_hidden_layers': 2,
    'hidden_activation': 'relu',
}
# config keys printed per appearance model (the checkpoint carries all
# sub-blocks, so only the selected model's keys are relevant)
CONFIG_PRINT_KEYS = {
    'sh': ['appearance_type', 'base_activation', 'residual_activation', 'color_activation', 'sh_degree'],
    'sv': ['appearance_type', 'base_activation', 'residual_activation', 'color_activation', 'n_sites'],
    'nasg': ['appearance_type', 'base_activation', 'residual_activation', 'color_activation', 'nasg_n_lobes'],
    'nasgabor': ['appearance_type', 'base_activation', 'residual_activation', 'color_activation', 'n_lobes'],
    'neural': [key for key in CONFIG_DEFAULTS if key not in ('sh_degree', 'n_sites', 'nasg_n_lobes', 'n_lobes')],
}


# ────────────────────────────────────────────────────────────────────────────
# Packing helpers (vectorized inverses of the viewer's decode functions)
# ────────────────────────────────────────────────────────────────────────────

def f16_bits(x: np.ndarray) -> np.ndarray:
    """float32 array -> uint32 array of float16 bit patterns."""
    return x.astype(np.float16).view(np.uint16).astype(np.uint32)


def pack_half2x16(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return f16_bits(x) | (f16_bits(y) << 16)


def encode_scales(s: np.ndarray) -> np.ndarray:
    """Linear scales [N, 3] -> 8-bit log codes (0 = sentinel for exactly 0)."""
    code = np.minimum(
        255,
        np.round(np.maximum(0.0, (np.log(np.maximum(s, 1e-30)) - LN_SCALE_MIN) * LN_SCALE_ENCODE)) + 1,
    ).astype(np.uint32)
    return np.where(s == 0.0, np.uint32(0), code)


def encode_quat_oct(q: np.ndarray) -> np.ndarray:
    """Unit quaternions [N, 4] as (x, y, z, w) -> 24-bit folded-octahedral codes."""
    q = np.where(q[:, 3:4] < 0.0, -q, q)
    half_theta = np.arccos(np.clip(q[:, 3], -1.0, 1.0))
    theta = 2.0 * half_theta
    s = np.sin(half_theta)
    degenerate = np.abs(s) < 1e-6
    safe_s = np.where(degenerate, 1.0, s)
    axis = q[:, :3] / safe_s[:, None]
    axis[degenerate] = [1.0, 0.0, 0.0]

    total = np.abs(axis).sum(axis=1)
    pu = axis[:, 0] / total
    pv = axis[:, 1] / total
    fold = axis[:, 2] < 0.0
    pu_folded = (1.0 - np.abs(pv)) * np.where(pu >= 0.0, 1.0, -1.0)
    pv_folded = (1.0 - np.abs(pu)) * np.where(pv >= 0.0, 1.0, -1.0)
    pu, pv = np.where(fold, pu_folded, pu), np.where(fold, pv_folded, pv)

    def q8(v):
        return np.round(np.clip(v, 0.0, 255.0)).astype(np.uint32)

    quant_u = q8((pu * 0.5 + 0.5) * 255.0)
    quant_v = q8((pv * 0.5 + 0.5) * 255.0)
    angle = q8(theta / math.pi * 255.0)
    return (angle << 16) | (quant_v << 8) | quant_u


def pack_splat_texture(means, scales, quats, base_code8, opacity8, n_padded):
    """Build the RGBA32UI splatData payload [P, 4] (uint32)."""
    n = means.shape[0]
    words = np.zeros((n_padded, 4), dtype=np.uint32)
    words[:n, 0] = (
        base_code8[:, 0] | (base_code8[:, 1] << 8) | (base_code8[:, 2] << 16) | (opacity8 << 24)
    )
    words[:n, 1] = pack_half2x16(means[:, 0], means[:, 1])
    quat_code = encode_quat_oct(quats)
    words[:n, 2] = f16_bits(means[:, 2]) | ((quat_code & 0xFF) << 16) | (((quat_code >> 8) & 0xFF) << 24)
    scale_code = encode_scales(scales)
    words[:n, 3] = scale_code[:, 0] | (scale_code[:, 1] << 8) | (scale_code[:, 2] << 16) | ((quat_code >> 16) << 24)
    return words


def pack_half_textures(values, n_padded):
    """Per-splat values [N, F] float32 -> list of ceil(F/8) RGBA32UI payloads [P, 4],
    8 fp16 values per texel (the baked h0_static and the SV/NASG/NASGabor payloads)."""
    features = values
    n, f = features.shape
    n_textures = (f + 7) // 8
    padded = np.zeros((n_padded, n_textures * 8), dtype=np.float32)
    padded[:n, :f] = features
    bits = f16_bits(padded)  # [P, 8 * n_textures] as uint32
    textures = []
    for k in range(n_textures):
        block = bits[:, k * 8:(k + 1) * 8]
        textures.append((block[:, 0::2] | (block[:, 1::2] << 16)).astype(np.uint32))
    return textures


def pack_weight_texels(matrices):
    """Layer matrices (float32, row-major [out × in]) -> fp16 texel stream [T, 4]."""
    texels = []
    for w in matrices:
        out_dim, in_dim = w.shape
        if in_dim % 4:
            raise ValueError(f'layer input dim must be a multiple of 4, got {w.shape}')
        # texel[j * out + o] = W[o, 4j:4j+4]
        blocks = w.reshape(out_dim, in_dim // 4, 4).transpose(1, 0, 2).reshape(-1, 4)
        texels.append(blocks)
    stream = np.concatenate(texels, axis=0).astype(np.float32)
    return stream.astype(np.float16).view(np.uint16)


def frequency_encode(features: np.ndarray, n_frequencies: int) -> np.ndarray:
    """NeRF-style frequency encoding, per feature: sin(2^f*x), cos(2^f*x) for f in [0, k)."""
    if n_frequencies == 0:
        return features
    scaled = features[..., None] * np.exp2(np.arange(n_frequencies, dtype=np.float32))
    return np.stack([np.sin(scaled), np.cos(scaled)], axis=-1).reshape(features.shape[0], -1).astype(np.float32)


# ────────────────────────────────────────────────────────────────────────────
# Reference forward pass (numpy) — used for the tcnn self-check and sanity print
# ────────────────────────────────────────────────────────────────────────────

def eval_sh_degrees(directions: np.ndarray, degrees: 'list[int]') -> np.ndarray:
    """tcnn sh_enc polynomials for the configured degrees >= 1, ascending ([N, sum(2l+1)])."""
    x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
    xy, xz, yz, x2, y2, z2 = x * y, x * z, y * z, x * x, y * y, z * z
    x4, y4, z4 = x2 * x2, y2 * y2, z2 * z2
    x6, y6, z6 = x4 * x2, y4 * y2, z4 * z2
    degree_values = {
        1: lambda: [
            -0.48860251190291987 * y,
            0.48860251190291987 * z,
            -0.48860251190291987 * x,
        ],
        2: lambda: [
            1.0925484305920792 * xy,
            -1.0925484305920792 * yz,
            0.94617469575755997 * z2 - 0.31539156525251999,
            -1.0925484305920792 * xz,
            0.54627421529603959 * x2 - 0.54627421529603959 * y2,
        ],
        3: lambda: [
            0.59004358992664352 * y * (-3.0 * x2 + y2),
            2.8906114426405538 * xy * z,
            0.45704579946446572 * y * (1.0 - 5.0 * z2),
            0.3731763325901154 * z * (5.0 * z2 - 3.0),
            0.45704579946446572 * x * (1.0 - 5.0 * z2),
            1.4453057213202769 * z * (x2 - y2),
            0.59004358992664352 * x * (-x2 + 3.0 * y2),
        ],
        4: lambda: [
            2.5033429417967046 * xy * (x2 - y2),
            1.7701307697799304 * yz * (-3.0 * x2 + y2),
            0.94617469575756008 * xy * (7.0 * z2 - 1.0),
            0.66904654355728921 * yz * (3.0 - 7.0 * z2),
            -3.1735664074561294 * z2 + 3.7024941420321507 * z4 + 0.31735664074561293,
            0.66904654355728921 * xz * (3.0 - 7.0 * z2),
            0.47308734787878004 * (x2 - y2) * (7.0 * z2 - 1.0),
            1.7701307697799304 * xz * (-x2 + 3.0 * y2),
            -3.7550144126950569 * x2 * y2 + 0.62583573544917614 * x4 + 0.62583573544917614 * y4,
        ],
        5: lambda: [
            0.65638205684017015 * y * (10.0 * x2 * y2 - 5.0 * x4 - y4),
            8.3026492595241645 * xy * z * (x2 - y2),
            -0.48923829943525038 * y * (3.0 * x2 - y2) * (9.0 * z2 - 1.0),
            4.7935367849733241 * xy * z * (3.0 * z2 - 1.0),
            0.45294665119569694 * y * (14.0 * z2 - 21.0 * z4 - 1.0),
            0.1169503224534236 * z * (-70.0 * z2 + 63.0 * z4 + 15.0),
            0.45294665119569694 * x * (14.0 * z2 - 21.0 * z4 - 1.0),
            2.3967683924866621 * z * (x2 - y2) * (3.0 * z2 - 1.0),
            -0.48923829943525038 * x * (x2 - 3.0 * y2) * (9.0 * z2 - 1.0),
            2.0756623148810411 * z * (-6.0 * x2 * y2 + x4 + y4),
            0.65638205684017015 * x * (10.0 * x2 * y2 - x4 - 5.0 * y4),
        ],
        6: lambda: [
            1.3663682103838286 * xy * (-10.0 * x2 * y2 + 3.0 * x4 + 3.0 * y4),
            2.3666191622317521 * yz * (10.0 * x2 * y2 - 5.0 * x4 - y4),
            2.0182596029148963 * xy * (x2 - y2) * (11.0 * z2 - 1.0),
            -0.92120525951492349 * yz * (3.0 * x2 - y2) * (11.0 * z2 - 3.0),
            0.92120525951492349 * xy * (-18.0 * z2 + 33.0 * z4 + 1.0),
            0.58262136251873131 * yz * (30.0 * z2 - 33.0 * z4 - 5.0),
            6.6747662381009842 * z2 - 20.024298714302954 * z4 + 14.684485723822165 * z6 - 0.31784601133814211,
            0.58262136251873131 * xz * (30.0 * z2 - 33.0 * z4 - 5.0),
            0.46060262975746175 * (x2 - y2) * (11.0 * z2 * (3.0 * z2 - 1.0) - 7.0 * z2 + 1.0),
            -0.92120525951492349 * xz * (x2 - 3.0 * y2) * (11.0 * z2 - 3.0),
            0.50456490072872406 * (11.0 * z2 - 1.0) * (-6.0 * x2 * y2 + x4 + y4),
            2.3666191622317521 * xz * (10.0 * x2 * y2 - x4 - 5.0 * y4),
            10.247761577878714 * x2 * y4 - 10.247761577878714 * x4 * y2 + 0.6831841051919143 * x6 - 0.6831841051919143 * y6,
        ],
    }
    return np.stack(sum((degree_values[degree]() for degree in degrees), []), axis=-1)


def reference_forward(matrices, features, directions, degrees, sh_degree_0_input=True):
    """Numpy forward pass through the (fp16-rounded) MLP; returns the raw rgb residual outputs."""
    n = features.shape[0]
    parts = [features] + ([np.full((n, 1), SH_C0, dtype=np.float32)] if sh_degree_0_input else []) \
        + [eval_sh_degrees(directions, degrees)]
    inputs = np.concatenate(parts, axis=-1).astype(np.float32)
    h = inputs
    for i, w in enumerate(matrices):
        w16 = w.astype(np.float16).astype(np.float32)
        h = h @ w16.T
        if i < len(matrices) - 1:
            h = np.maximum(h, 0.0)
    return h


def apply_residual_activation(raw, activation):
    """Reference of FasterGSVDA/Appearance/utils.py residual_activation."""
    if activation == 'tanh':
        return np.tanh(raw)
    if activation == 'softplus':  # beta 10, evaluated in the numerically stable form
        return np.where(raw > 2.0, raw, np.log1p(np.exp(10.0 * np.minimum(raw, 2.0))) / 10.0)
    return raw


def apply_color_activation(raw, activation):
    """Reference of FasterGSVDA/Appearance/utils.py color_activation. The 0.5 shift
    of the shifted activations is already applied to raw (baked into the stored
    base, or added by the caller for the exp-base layout)."""
    if activation == 'relu':
        return np.maximum(raw, 0.0)
    if activation == 'softplus':
        return np.where(raw > 2.0, raw, np.log1p(np.exp(10.0 * np.minimum(raw, 2.0))) / 10.0)
    if activation == 'sigmoid':
        return 1.0 / (1.0 + np.exp(-4.0 * raw))
    if activation == 'satexp':
        return -np.expm1(-raw)
    if activation == 'hardsigmoid':
        return np.clip(raw, 0.0, 1.0)
    return raw  # none


# ────────────────────────────────────────────────────────────────────────────
# SH / SV / NASG / NASGabor payloads: the stored per-splat values are the ACTIVATED
# residual parameters (fp16, 8 per RGBA32UI texel), so the viewer evaluates the
# view-dependent residual without re-applying the per-parameter activations.
# The reference evaluators below mirror the viewer shaders exactly and are used
# for the sanity print (fed the fp16-rounded stored values).
# ────────────────────────────────────────────────────────────────────────────

# quantized SH level layout, shared with the viewer's packed-SH evaluator:
# per level, coefficient-major rgb fields of `bits` each, bit-packed
# little-endian into `words` u32 words per splat (level 1: 9x7b in RG,
# level 2: 15x8b in RGBA, level 3: 21x6b in RGBA)
SH_LEVELS = [
    {'degree': 1, 'bits': 7, 'words': 2},
    {'degree': 2, 'bits': 8, 'words': 4},
    {'degree': 3, 'bits': 6, 'words': 4},
]


def build_sh_payload(residual_params, config, active_degree):
    """Raw view-dependent SH coefficients [N, C, 3] in basis order.

    The shared residual activation applies to the summed residual (not the
    coefficients), so the coefficients stay raw and the shader applies it.
    Not-yet-enabled degrees of the appearance schedule are trimmed away.
    """
    n, n_coefficients, _ = residual_params.shape
    max_degree = int(round(math.sqrt(n_coefficients + 1))) - 1
    if (max_degree + 1) ** 2 - 1 != n_coefficients:
        sys.exit(f'error: {n_coefficients} sh coefficients do not form complete degrees 1..D')
    if max_degree != config['sh_degree']:
        print(f'WARNING: the config says SH_DEGREE={config["sh_degree"]} but the checkpoint holds '
              f'degree-{max_degree} coefficients — exporting what the checkpoint holds')
    if max_degree > 3:
        sys.exit(f'error: the viewer\'s quantized sh layout covers degrees 1-3, '
                 f'got a degree-{max_degree} checkpoint')
    if active_degree is None:
        print(f'warning: no appearance schedule state in the checkpoint — assuming all '
              f'{max_degree} view-dependent sh degrees are enabled')
        active_degree = max_degree
    export_degree = min(active_degree, max_degree)
    if export_degree < max_degree:
        print(f'WARNING: the checkpoint has only {export_degree} of {max_degree} sh degrees enabled — '
              f'the not-yet-enabled degrees are trimmed to reproduce the rasterizer; '
              f'export a final checkpoint for the full model')
    coefficients = residual_params[:, :(export_degree + 1) ** 2 - 1, :].astype(np.float32)
    degree_mask = sum(1 << degree for degree in range(1, export_degree + 1))
    return coefficients, degree_mask, export_degree


def pack_sh_quantized(coefficients, n_padded):
    """Quantizes raw SH coefficients [N, C, 3] into the viewer's packed layout.

    Per present level: symmetric quantization to the level's signed bit width at
    a shared max|coefficient| scale. Returns the texture payloads (one per
    level, [P, words] u32), the three level scales (0 for absent levels), and
    the dequantized coefficients for the sanity print.
    """
    n = coefficients.shape[0]
    textures, scales = [], []
    dequantized = np.zeros_like(coefficients)
    offset = 0
    for level in SH_LEVELS:
        width = 2 * level['degree'] + 1
        if offset + width > coefficients.shape[1]:
            scales.append(0.0)
            continue
        values = coefficients[:, offset:offset + width, :].reshape(n, -1)
        scale = max(float(np.abs(values).max(initial=0.0)), 1e-8)
        quant_max = (1 << (level['bits'] - 1)) - 1
        codes = np.clip(np.round(values / scale * quant_max), -quant_max, quant_max).astype(np.int32)
        dequantized[:, offset:offset + width, :] = \
            (codes.astype(np.float32) * (scale / quant_max)).reshape(n, width, 3)
        codes = (codes & ((1 << level['bits']) - 1)).astype(np.uint32)
        words = np.zeros((n_padded, level['words']), np.uint32)
        for field_idx in range(codes.shape[1]):
            bit = level['bits'] * field_idx
            word_idx, shift = bit // 32, bit % 32
            words[:n, word_idx] |= codes[:, field_idx] << shift
            if shift + level['bits'] > 32:
                words[:n, word_idx + 1] |= codes[:, field_idx] >> (32 - shift)
        textures.append(words)
        scales.append(scale)
        offset += width
    return textures, scales, dequantized


def sh_residual_reference(coefficients, directions, residual_act):
    """Viewer-shader reference: residual_activation(sum_i basis_i(dir) * rgb_i)
    for coefficients [N, C, 3] in basis order."""
    n_coefficients = coefficients.shape[1]
    if n_coefficients == 0:
        return np.zeros((coefficients.shape[0], 3), np.float32)
    degree = int(round(math.sqrt(n_coefficients + 1))) - 1
    basis = eval_sh_degrees(directions, list(range(1, degree + 1)))  # [N, C]
    return apply_residual_activation(np.einsum('nc,ncr->nr', basis, coefficients), residual_act)


def build_sv_payload(residual_params, config, residual_act):
    """Activated spherical Voronoi sites [N, K * 7]: unit site xyz, exp'd
    temperature, activated site color rgb (matching the fused kernels)."""
    n, n_sites, _ = residual_params.shape
    if n_sites != config['n_sites']:
        print(f'WARNING: the config says N_SITES={config["n_sites"]} but the checkpoint holds '
              f'{n_sites} sites — exporting what the checkpoint holds')
    raw_sites = residual_params[..., 0:3]
    lengths = np.maximum(np.linalg.norm(raw_sites, axis=-1, keepdims=True), SV_SITE_LENGTH_EPS)
    tau = np.exp(residual_params[..., 3:4])
    if float(tau.max(initial=0.0)) > FP16_SAFE_MAX:
        print(f'note: {int((tau > FP16_SAFE_MAX).sum())} site temperatures exceed the fp16 range '
              f'and are clamped to {FP16_SAFE_MAX:.0f} (the softmax is a hard assignment there)')
    values = np.concatenate([
        raw_sites / lengths,
        np.minimum(tau, FP16_SAFE_MAX),
        apply_residual_activation(residual_params[..., 4:7], residual_act),
    ], axis=-1)
    return values.reshape(n, -1).astype(np.float32), n_sites


def sv_residual_reference(values, directions):
    """Viewer-shader reference: softmax_k(-tau_k * ||site_k - dir||)-weighted site color sum."""
    n = values.shape[0]
    sites = values.reshape(n, -1, 7)
    distances = np.linalg.norm(sites[..., 0:3] - directions[:, None], axis=-1)
    logits = -sites[..., 3] * distances
    weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    return (weights[..., None] * sites[..., 4:7]).sum(axis=1).astype(np.float32)


def build_nasg_payload(residual_params, config, residual_act, active_degree):
    """Fully baked NASG lobes [N, L * 12] — the NASGabor layout without the
    Gabor frequency: frame vectors x and z, 2*lambda, the anisotropy a, the
    normalization constant N, and the activated rgb lobe weights.
    Not-yet-enabled lobes of the appearance schedule are trimmed away."""
    n, n_lobes, _ = residual_params.shape
    if n_lobes != config['nasg_n_lobes']:
        print(f'WARNING: the config says N_LOBES={config["nasg_n_lobes"]} but the checkpoint holds '
              f'{n_lobes} lobes — exporting what the checkpoint holds')
    if active_degree is None:
        print(f'warning: no appearance schedule state in the checkpoint — assuming all '
              f'{n_lobes} lobes are enabled')
        active_degree = n_lobes
    export_lobes = min(active_degree, n_lobes)
    if export_lobes < n_lobes:
        print(f'WARNING: the checkpoint has only {export_lobes} of {n_lobes} lobes enabled — '
              f'the not-yet-enabled lobes are trimmed to reproduce the rasterizer; '
              f'export a final checkpoint for the full model')
    lobes = residual_params[:, :export_lobes, :]
    cos = np.clip(np.tanh(lobes[..., 0:3]), -NASGABOR_COS_LIMIT, NASGABOR_COS_LIMIT)
    sin = np.sqrt(1.0 - cos * cos)
    cos_theta, cos_phi, cos_tau = cos[..., 0], cos[..., 1], cos[..., 2]
    sin_theta, sin_phi, sin_tau = sin[..., 0], sin[..., 1], sin[..., 2]
    frame_x = np.stack([
        cos_theta * cos_phi * cos_tau - sin_theta * sin_tau,
        sin_theta * cos_phi * cos_tau + cos_theta * sin_tau,
        -sin_phi * cos_tau,
    ], axis=-1)
    frame_z = np.stack([cos_theta * sin_phi, sin_theta * sin_phi, cos_phi], axis=-1)
    lam, a = (np.minimum(np.exp(lobes[..., i]), NASGABOR_SHAPE_LIMIT) for i in (3, 4))
    norm = lam * np.sqrt(1.0 + a) / (2.0 * math.pi * (1.0 + NASGABOR_EPS_NORM - np.exp(-2.0 * lam)))
    values = np.concatenate([
        frame_x,
        frame_z,
        (2.0 * lam)[..., None],
        a[..., None],
        norm[..., None],
        apply_residual_activation(lobes[..., 5:8], residual_act),
    ], axis=-1)
    return values.reshape(n, -1).astype(np.float32), export_lobes


def nasg_residual_reference(values, directions):
    """Viewer-shader reference: sum of pdf-weighted lobe colors from the baked
    lobe values (see build_nasg_payload for the layout)."""
    n = values.shape[0]
    lobes = values.reshape(n, -1, 12)
    result = np.zeros((n, 3), np.float32)
    for lobe_idx in range(lobes.shape[1]):
        frame_x = lobes[:, lobe_idx, 0:3]
        frame_z = lobes[:, lobe_idx, 3:6]
        lam2, a, norm = (lobes[:, lobe_idx, i] for i in (6, 7, 8))
        v_z = (directions * frame_z).sum(axis=-1)
        v_x = (directions * frame_x).sum(axis=-1)
        with np.errstate(divide='ignore', invalid='ignore'):
            K = 0.5 * (v_z + 1.0)
            K_e = NASGABOR_EPS + a * v_x ** 2 / (1.0 - v_z ** 2)
            E = K ** K_e
            pdf = np.exp(lam2 * (E * K - 1.0)) * E * norm
        pdf = np.where(v_z >= 1.0 - NASGABOR_POLE_EPS, 1.0,
                       np.where(v_z > -1.0 + NASGABOR_POLE_EPS, pdf, 0.0))
        result += pdf[:, None] * lobes[:, lobe_idx, 9:12]
    return result.astype(np.float32)


def build_nasgabor_payload(residual_params, config, residual_act, active_degree):
    """Fully baked NASGabor lobes [N, L * 13]: everything the per-frame response
    does NOT need the view direction for is precomputed — the lobe frame vectors
    x and z (from the clamped tanh cosines), 2*lambda (the shape of the response
    exponent), the anisotropy a (clamped exp), the Gabor frequency k, the
    normalization constant N = lambda * sqrt(1 + a) / (2pi * (1 + eps - exp(-2 lambda))),
    and the activated rgb lobe weights. Not-yet-enabled lobes of the appearance
    schedule are trimmed away."""
    n, n_lobes, _ = residual_params.shape
    if n_lobes != config['n_lobes']:
        print(f'WARNING: the config says N_LOBES={config["n_lobes"]} but the checkpoint holds '
              f'{n_lobes} lobes — exporting what the checkpoint holds')
    if active_degree is None:
        print(f'warning: no appearance schedule state in the checkpoint — assuming all '
              f'{n_lobes} lobes are enabled')
        active_degree = n_lobes
    export_lobes = min(active_degree, n_lobes)
    if export_lobes < n_lobes:
        print(f'WARNING: the checkpoint has only {export_lobes} of {n_lobes} lobes enabled — '
              f'the not-yet-enabled lobes are trimmed to reproduce the rasterizer; '
              f'export a final checkpoint for the full model')
    lobes = residual_params[:, :export_lobes, :]
    cos = np.clip(np.tanh(lobes[..., 0:3]), -NASGABOR_COS_LIMIT, NASGABOR_COS_LIMIT)
    sin = np.sqrt(1.0 - cos * cos)
    cos_theta, cos_phi, cos_tau = cos[..., 0], cos[..., 1], cos[..., 2]
    sin_theta, sin_phi, sin_tau = sin[..., 0], sin[..., 1], sin[..., 2]
    frame_x = np.stack([
        cos_theta * cos_phi * cos_tau - sin_theta * sin_tau,
        sin_theta * cos_phi * cos_tau + cos_theta * sin_tau,
        -sin_phi * cos_tau,
    ], axis=-1)
    frame_z = np.stack([cos_theta * sin_phi, sin_theta * sin_phi, cos_phi], axis=-1)
    lam, a = (np.minimum(np.exp(lobes[..., i]), NASGABOR_SHAPE_LIMIT) for i in (3, 4))
    norm = lam * np.sqrt(1.0 + a) / (2.0 * math.pi * (1.0 + NASGABOR_EPS_NORM - np.exp(-2.0 * lam)))
    values = np.concatenate([
        frame_x,
        frame_z,
        (2.0 * lam)[..., None],
        a[..., None],
        ((np.tanh(lobes[..., 5]) + 1.0) * NASGABOR_FREQUENCY_SCALE)[..., None],
        norm[..., None],
        apply_residual_activation(lobes[..., 6:9], residual_act),
    ], axis=-1)
    return values.reshape(n, -1).astype(np.float32), export_lobes


def nasgabor_residual_reference(values, directions):
    """Viewer-shader reference: sum of pdf-weighted lobe colors from the baked
    lobe values (see build_nasgabor_payload for the layout)."""
    n = values.shape[0]
    lobes = values.reshape(n, -1, 13)
    result = np.zeros((n, 3), np.float32)
    for lobe_idx in range(lobes.shape[1]):
        frame_x = lobes[:, lobe_idx, 0:3]
        frame_z = lobes[:, lobe_idx, 3:6]
        lam2, a, k, norm = (lobes[:, lobe_idx, i] for i in (6, 7, 8, 9))
        v_z = (directions * frame_z).sum(axis=-1)
        v_x = (directions * frame_x).sum(axis=-1)
        with np.errstate(divide='ignore', invalid='ignore'):
            K = 0.5 * (v_z + 1.0)
            K_e = NASGABOR_EPS + a * v_x ** 2 / (1.0 - v_z ** 2)
            E = K ** K_e
            G = 0.5 * (1.0 + np.cos(k * v_x))
            pdf = np.exp(lam2 * (E * K - 1.0)) * E * G * norm
        pdf = np.where(v_z >= 1.0 - NASGABOR_POLE_EPS, 1.0,
                       np.where(v_z > -1.0 + NASGABOR_POLE_EPS, pdf, 0.0))
        result += pdf[:, None] * lobes[:, lobe_idx, 10:13]
    return result.astype(np.float32)


# ────────────────────────────────────────────────────────────────────────────
# PPISP: numpy port of the forward pass (ppisp_math.cuh), used to bake the
# response curve and to check the export against the CUDA output
# ────────────────────────────────────────────────────────────────────────────

def ppisp_homography(latents):
    """3x3 color homography from the 8 latents (b, r, g, n offsets), as compute_homography."""
    offsets = [PPISP_COLOR_PINV_BLOCKS[i] @ np.asarray(latents[2 * i:2 * i + 2], np.float64) for i in range(4)]
    t_b = np.array([offsets[0][0], offsets[0][1], 1.0])
    t_r = np.array([1.0 + offsets[1][0], offsets[1][1], 1.0])
    t_g = np.array([offsets[2][0], 1.0 + offsets[2][1], 1.0])
    t_n = np.array([1.0 / 3.0 + offsets[3][0], 1.0 / 3.0 + offsets[3][1], 1.0])
    T = np.stack([t_b, t_r, t_g], axis=1)
    skew = np.array([[0.0, -t_n[2], t_n[1]], [t_n[2], 0.0, -t_n[0]], [-t_n[1], t_n[0], 0.0]])
    M = skew @ T
    lam = np.cross(M[0], M[1])
    if lam @ lam < 1e-20:
        lam = np.cross(M[0], M[2])
        if lam @ lam < 1e-20:
            lam = np.cross(M[1], M[2])
    S_inv = np.array([[-1.0, -1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    H = T @ np.diag(lam) @ S_inv
    if abs(H[2, 2]) > 1e-20:
        H = H / H[2, 2]
    return H.astype(np.float32)


def ppisp_crf_curve(crf_raw):
    """Activated response curve parameters per channel [3, 6]: toe, shoulder, gamma,
    center, and the piecewise curve's a, b (apply_crf_ppisp)."""
    out = np.zeros((3, 6), np.float32)
    for ch in range(3):
        toe, shoulder, gamma = (m + np.log1p(np.exp(v)) for v, m in zip(crf_raw[ch, :3], PPISP_CRF_MIN))
        center = 1.0 / (1.0 + np.exp(-crf_raw[ch, 3]))
        a = shoulder * center / (toe + (shoulder - toe) * center)
        out[ch] = (toe, shoulder, gamma, center, a, 1.0 - a)
    return out


def ppisp_reference(rgb, exposure, latents, crf_curve, vignetting=None):
    """Forward PPISP on an [H, W, 3] radiance image: exposure, optionally the vignetting
    (pixel centers, coordinates normalized by the larger image side; only for the check
    against CUDA), color homography in (r, g, r+g+b) space, and the response curve with
    its input clamped to [0, 1]."""
    h, w, _ = rgb.shape
    out = rgb.astype(np.float32) * np.float32(2.0 ** exposure)
    if vignetting is not None:
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32) + 0.5
        max_res = float(max(w, h))
        u, v = (xs - w * 0.5) / max_res, (ys - h * 0.5) / max_res
        for ch in range(3):
            cx, cy, a0, a1, a2 = vignetting[ch]
            r2 = (u - cx) ** 2 + (v - cy) ** 2
            out[..., ch] *= np.clip(1.0 + a0 * r2 + a1 * r2 ** 2 + a2 * r2 ** 3, 0.0, 1.0)
    H = ppisp_homography(latents)
    intensity = out.sum(axis=-1)
    rgi = np.stack([out[..., 0], out[..., 1], intensity], axis=-1) @ H.T
    rgi *= (intensity / (rgi[..., 2] + 1e-5))[..., None]
    out = np.stack([rgi[..., 0], rgi[..., 1], rgi[..., 2] - rgi[..., 0] - rgi[..., 1]], axis=-1)
    x = np.clip(out, 0.0, 1.0)
    for ch in range(3):
        toe, shoulder, gamma, center, a, b = crf_curve[ch]
        xc = x[..., ch]
        y = np.where(xc <= center, a * (xc / center) ** toe, 1.0 - b * ((1.0 - xc) / (1.0 - center)) ** shoulder)
        out[..., ch] = np.maximum(y, 0.0) ** gamma
    return out.astype(np.float32)


# ────────────────────────────────────────────────────────────────────────────
# Model assembly
# ────────────────────────────────────────────────────────────────────────────

# nn.Module plumbing segments, dropped from paths so suffix matching works the
# same whether the checkpoint holds a flat state dict or live module objects
_MODULE_SEGMENTS = {'_parameters', '_buffers', '_modules'}


def _walk_tensors(node, visit, prefix='', seen=None):
    """Depth-first walk over dicts/lists/objects (via __dict__, so live nn.Module
    graphs work too), reporting (path, tensor) pairs through visit."""
    if seen is None:
        seen = set()
    if id(node) in seen:
        return
    seen.add(id(node))
    if hasattr(node, 'shape') and hasattr(node, 'dtype'):
        visit(prefix, node)
        return
    if hasattr(node, 'items'):
        entries = node.items()
    elif isinstance(node, (list, tuple)):
        entries = ((f'[{i}]', v) for i, v in enumerate(node))
    elif isinstance(getattr(node, '__dict__', None), dict):
        entries = node.__dict__.items()
    else:
        return
    for key, value in entries:
        key = str(key)
        if key in _MODULE_SEGMENTS:
            path = prefix
        elif key.startswith('['):
            path = f'{prefix}{key}'
        else:
            path = f'{prefix}.{key}' if prefix else key
        _walk_tensors(value, visit, path, seen)


def find_tensor(state, suffix):
    """Search a (possibly nested) checkpoint for a tensor whose path ends in suffix."""
    matches = []
    _walk_tensors(state, lambda path, t: matches.append((path, t)) if path.endswith(suffix) else None)
    if not matches:
        raise KeyError(f'no tensor ending in "{suffix}" found in checkpoint (use --dump-keys)')
    if len(matches) > 1:
        names = ', '.join(name for name, _ in matches)
        raise KeyError(f'ambiguous suffix "{suffix}": {names}')
    return matches[0][1]


def find_mapping(state, key, seen=None):
    """Depth-first search for the first non-tensor mapping holding `key` (None if absent).

    Used for the config lists and the appearance schedule state, which the tensor
    walk above skips; both are plain dicts (Framework.ConfigParameterList is a
    Munch, i.e. a dict subclass, and the stubbed unpickling produces dicts too).
    """
    if seen is None:
        seen = set()
    node = state
    if id(node) in seen or (hasattr(node, 'shape') and hasattr(node, 'dtype')):
        return None
    seen.add(id(node))
    if hasattr(node, 'items'):
        if key in node:
            return node
        children = list(node.values())
    elif isinstance(node, (list, tuple)):
        children = list(node)
    elif isinstance(getattr(node, '__dict__', None), dict):
        children = list(node.__dict__.values())
    else:
        return None
    for child in children:
        if (found := find_mapping(child, key, seen)) is not None:
            return found
    return None


def dump_keys(state):
    _walk_tensors(state, lambda path, t: print(f'  {path}  {tuple(t.shape)}  {t.dtype}'))


def _die_corrupt_checkpoint(path, err):
    """Explain a torch zip-archive failure (truncated / partially written file)."""
    import os
    import zipfile
    size = os.path.getsize(path)
    detail = ''
    try:
        bad = zipfile.ZipFile(path).testzip()
        detail = f' (zip central directory readable; first bad record: {bad})' if bad else ''
    except Exception as zip_err:
        detail = f' (not a readable zip archive: {zip_err})'
    sys.exit(
        f'error: checkpoint {path} ({size / 1024 / 1024:.1f} MB) is truncated or corrupted{detail}.\n'
        f'torch said: {err}\n'
        f'Common causes: the exporter ran while the checkpoint was still being written, '
        f'an interrupted save (crash / disk full), or an incomplete copy. '
        f'Wait for training to finish writing it, or restore/re-save the checkpoint.'
    )


def add_nerficg_src_to_path():
    """Makes the NeRFICG modules importable (this file lives in src/Methods/FasterGSVDA/viewer/),
    so checkpoints unpickle natively and the Framework can reload the dataset."""
    import os
    src_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def permissive_torch_load(path):
    """torch.load that survives checkpoints pickled with unimportable project classes
    (e.g. when the NeRFICG modules are not importable): unknown classes are unpickled
    as inert dict stubs — only the tensors and config lists inside them are needed."""
    import torch
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except ModuleNotFoundError as e:
        print(f'note: {e} — retrying with stubbed unpickling (tensors only)')
    except RuntimeError as e:
        if 'PytorchStreamReader' in str(e):
            _die_corrupt_checkpoint(path, e)
        raise

    import pickle
    import types

    class _Stub(dict):
        """Absorbs constructor args and pickled state; nested values land in the dict."""
        def __init__(self, *args, **kwargs):
            super().__init__()

        def __setstate__(self, state):
            for part in state if isinstance(state, tuple) else (state,):
                if isinstance(part, dict):
                    self.update(part)

    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            try:
                return super().find_class(module, name)
            except (ModuleNotFoundError, AttributeError):
                return type(name, (_Stub,), {'__module__': module})

    shim = types.ModuleType('ngsplat_stub_pickle')
    shim.Unpickler = _Unpickler
    shim.load = pickle.load
    shim.loads = pickle.loads
    return torch.load(path, map_location='cpu', weights_only=False, pickle_module=shim)


def slice_mlp_params(params, n_in, n_neurons, n_hidden_layers, n_out=3, out_padded=16):
    """tcnn FullyFusedMLP flat params -> row-major [out × in] matrices (output trimmed)."""
    expected = n_in * n_neurons + (n_hidden_layers - 1) * n_neurons * n_neurons + out_padded * n_neurons
    if params.size != expected:
        raise ValueError(
            f'mlp params length {params.size} != expected {expected} for a {n_in}-wide input '
            f'(tcnn pads the input to a multiple of 16; the viewer has no notion of padded inputs, '
            f'so only configurations whose encoded features + sh degree values already fill a '
            f'multiple of 16 can be exported)'
        )
    matrices, offset = [], 0
    dims = [(n_neurons, n_in)] + [(n_neurons, n_neurons)] * (n_hidden_layers - 1) + [(out_padded, n_neurons)]
    for rows, cols in dims:
        matrices.append(params[offset:offset + rows * cols].reshape(rows, cols).astype(np.float32))
        offset += rows * cols
    matrices[-1] = matrices[-1][:n_out]
    return matrices


def read_appearance_config(appearance):
    """Maps a MODEL.APPEARANCE mapping (from the checkpoint or the yaml) to exporter config keys."""
    sh = appearance.get('SH') or {}
    sv = appearance.get('SV') or {}
    nasg = appearance.get('NASG') or {}
    nasgabor = appearance.get('NASGABOR') or {}
    neural = appearance.get('NEURAL') or {}
    config = {}
    if 'TYPE' in appearance: config['appearance_type'] = str(appearance['TYPE']).lower()
    if 'BASE_ACTIVATION' in appearance: config['base_activation'] = str(appearance['BASE_ACTIVATION']).lower()
    if 'RESIDUAL_ACTIVATION' in appearance: config['residual_activation'] = str(appearance['RESIDUAL_ACTIVATION']).lower()
    if 'COLOR_ACTIVATION' in appearance: config['color_activation'] = str(appearance['COLOR_ACTIVATION']).lower()
    if 'SH_DEGREE' in sh: config['sh_degree'] = int(sh['SH_DEGREE'])
    if 'N_SITES' in sv: config['n_sites'] = int(sv['N_SITES'])
    if 'N_LOBES' in nasg: config['nasg_n_lobes'] = int(nasg['N_LOBES'])
    if 'N_LOBES' in nasgabor: config['n_lobes'] = int(nasgabor['N_LOBES'])
    if 'BASE_INPUT' in neural: config['base_input'] = bool(neural['BASE_INPUT'])
    if 'FEATURE_N_FREQUENCIES' in neural: config['n_frequencies'] = int(neural['FEATURE_N_FREQUENCIES'])
    if 'SH_DEGREES' in neural: config['sh_degrees'] = [int(degree) for degree in neural['SH_DEGREES']]
    if 'N_NEURONS' in neural: config['n_neurons'] = int(neural['N_NEURONS'])
    if 'N_HIDDEN_LAYERS' in neural: config['n_hidden_layers'] = int(neural['N_HIDDEN_LAYERS'])
    if 'HIDDEN_ACTIVATION' in neural: config['hidden_activation'] = str(neural['HIDDEN_ACTIVATION']).lower()
    if 'FEATURE_DIM' in neural: config['feature_dim'] = int(neural['FEATURE_DIM'])
    return config


def resolve_model_dir(path):
    """If `path` is a FasterGSVDA output directory, locate its checkpoint and read
    training_config.yaml. Returns the checkpoint path, a default output path, the
    appearance config found in the yaml, and the renderer flags."""
    import glob
    import os
    if not os.path.isdir(path):
        return path, path + '.ngsplat', {}, {}
    ckpt = os.path.join(path, 'checkpoints', 'final.pt')
    if not os.path.isfile(ckpt):
        candidates = sorted(glob.glob(os.path.join(path, 'checkpoints', '*.pt')) +
                            glob.glob(os.path.join(path, '*.pt')), key=os.path.getmtime)
        if not candidates:
            raise FileNotFoundError(f'no checkpoint (.pt) found under {path}')
        ckpt = candidates[-1]

    default_out = os.path.join(path, 'scene.ngsplat')
    cfg_path = os.path.join(path, 'training_config.yaml')
    if not os.path.isfile(cfg_path):
        print(f'warning: no training_config.yaml in {path} — RENDERER.PROPER_ANTIALIASING is assumed to be false')
        return ckpt, default_out, {}, {}
    try:
        import yaml
    except ImportError:
        print(f'warning: pyyaml not installed — cannot read {cfg_path}; RENDERER.PROPER_ANTIALIASING is assumed to be false')
        return ckpt, default_out, {}, {}
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    yaml_config = read_appearance_config(((cfg.get('MODEL') or {}).get('APPEARANCE')) or {})
    renderer = cfg.get('RENDERER') or {}
    print(f'renderer config from {cfg_path}:')
    print(f'  proper_antialiasing = {bool(renderer.get("PROPER_ANTIALIASING"))}')
    if not renderer.get('RENDER_BASE_COLOR', True) or not renderer.get('RENDER_RESIDUAL_COLOR', True):
        print('note: RENDER_BASE_COLOR/RENDER_RESIDUAL_COLOR are debug switches of the renderer; '
              'the exported file always holds the full color model')
    scale_mod = renderer.get('SCALE_MODIFIER', 1.0)
    if scale_mod != 1.0:
        print(f'note: RENDERER.SCALE_MODIFIER is {scale_mod} in the config; the viewer '
              f'defaults its Scale modifier slider to 1.0 — adjust it manually to match')
    return ckpt, default_out, yaml_config, renderer


def load_test_cameras(model_dir, ckpt_path=None, ppisp_static=None):
    """Reload the dataset the model was trained on (via the NeRFICG Framework and the
    training_config.yaml in `model_dir`) and return the test-subset viewpoints as an
    [N, 18] float32 array: c2w rows (3×4 row-major) + fx, fy, cx, cy, width, height.
    With `ppisp_static` (vignetting, response curve of camera 0) the model is rendered
    through the Framework for every test view to bake the PPISP controller's exposure
    and color prediction ([N, 9], second return value; None otherwise).

    The c2w matrices live in the same (possibly PCA-transformed) world space as the
    checkpoint's gaussians, because the dataset loader applies the identical transform.
    Returns (None, None) if the Framework or the dataset is unavailable.
    """
    import os
    cfg_path = os.path.join(model_dir, 'training_config.yaml')
    if not os.path.isfile(cfg_path):
        print('warning: no training_config.yaml — cannot bake test cameras')
        return None, None
    add_nerficg_src_to_path()
    try:
        import Framework
        Framework.setup(require_custom_config=True, config_path=cfg_path)
        from Implementations import Datasets as DI
        dataset = DI.get_dataset(
            dataset_type=Framework.config.GLOBAL.DATASET_TYPE,
            path=Framework.config.DATASET.PATH,
        )
    except Exception as e:
        print(f'warning: could not load the dataset for test cameras ({type(e).__name__}: {e})')
        return None, None
    views = list(dataset.test())
    if not views:
        print('warning: the dataset has no test views — no cameras baked')
        return None, None
    rows = []
    for view in views:
        cam = view.camera
        rows.append(np.concatenate([
            view.c2w_numpy[:3, :4].reshape(-1),
            [cam.focal_x, cam.focal_y, cam.center_x, cam.center_y, cam.width, cam.height],
        ]).astype(np.float32))
    cameras = np.stack(rows)
    print(f'Test cameras: {len(views)} baked ({int(cameras[0, 16])}x{int(cameras[0, 17])} native)')
    view_ppisp = bake_ppisp_views(views, ckpt_path, ppisp_static) if ppisp_static is not None else None
    return cameras, view_ppisp


def bake_ppisp_views(views, ckpt_path, ppisp_static):
    """Render every test view through the Framework and return the PPISP controller's
    prediction per view ([N, 9]: exposure + 8 color latents), checking the numpy port
    against the CUDA output of the first views. None if rendering is unavailable."""
    import torch
    import Framework
    try:
        from Implementations import Methods as MI
        model = MI.get_model(method=Framework.config.GLOBAL.METHOD_TYPE, checkpoint=ckpt_path).eval()
        renderer = MI.get_renderer(method=Framework.config.GLOBAL.METHOD_TYPE, model=model)
    except Exception as e:
        print(f'warning: could not load the model for the PPISP bake ({type(e).__name__}: {e}) — '
              f'the viewer will use identity exposure and color')
        return None
    ppisp = model.ppisp
    captured = {}
    original_forward = ppisp.forward

    def capturing_forward(rgb, view):  # keeps the raw radiance and the PPISP output of each render
        captured['raw'] = rgb.detach()
        captured['out'] = original_forward(rgb, view)
        return captured['out']
    ppisp.forward = capturing_forward

    vignetting, crf_curve = ppisp_static
    rows, max_err = [], 0.0
    with torch.no_grad():
        for i, view in enumerate(views):
            renderer.render_image(view)
            raw = captured['raw']
            camera_idx = ppisp.known_camera_indices.get(view.camera_index, 0)
            if camera_idx != 0:
                print(f'warning: test view {i} uses camera {camera_idx}; the viewer bakes camera 0 only')
            exposure, color = ppisp.model.controllers[camera_idx](raw)
            rows.append(np.concatenate([[float(exposure)], color.float().cpu().numpy().reshape(-1)]))
            if i < 3:
                expected = ppisp_reference(raw.float().cpu().numpy(), rows[-1][0], rows[-1][1:], crf_curve, vignetting)
                max_err = max(max_err, float(np.abs(expected - captured['out'].float().cpu().numpy()).max()))
    ppisp.forward = original_forward
    print(f'PPISP: controller predictions baked for {len(rows)} test views; numpy port vs CUDA '
          f'max |Δ| = {max_err:.5f} {"OK" if max_err < 5e-3 else "MISMATCH — check the PPISP port!"}')
    return np.stack(rows).astype(np.float32)


def camera_hint_from_test_cameras(cameras, center):
    """Orbit-camera hint derived from the training cameras, following
    mesh-splatting's export_web.py (compute_camera_params):

        up       = mean of the per-camera up axes. The c2w columns are
                   right/down/forward, so a camera's up is -column 1.
        distance = median distance from the camera positions to `center`.

    Returns (up, distance); up is None if the averaged direction degenerates
    (cameras with cancelling roll), in which case the caller falls back.
    """
    down = cameras[:, [1, 5, 9]].astype(np.float64)  # c2w column 1 per camera
    ups = -down
    ups /= np.maximum(np.linalg.norm(ups, axis=1, keepdims=True), 1e-12)
    mean_up = ups.sum(axis=0)
    norm = float(np.linalg.norm(mean_up))
    positions = cameras[:, [3, 7, 11]].astype(np.float64)
    distance = float(np.median(np.linalg.norm(positions - center, axis=1)))
    if norm < 1e-6:
        return None, distance
    return (mean_up / norm).astype(np.float32), distance


def load_checkpoint(path, dump_keys_only):
    """Loads the checkpoint and returns it together with the appearance config it carries
    (MODEL.APPEARANCE is stored alongside the state dict) and the appearance schedule position."""
    print(f'Loading checkpoint: {path}')
    state = permissive_torch_load(path)
    if dump_keys_only:
        print('Checkpoint tensors:')
        dump_keys(state)
        sys.exit(0)

    appearance = find_mapping(state, 'BASE_ACTIVATION')
    checkpoint_config = read_appearance_config(appearance) if appearance is not None else {}
    extra_state = find_mapping(state, 'active_degree')
    active_degree = int(extra_state['active_degree']) if extra_state is not None else None
    ppisp = find_mapping(state, 'CONTROLLER_TRAINING_STEPS')
    has_ppisp = ppisp is not None and bool(ppisp.get('USE'))
    if has_ppisp:
        print('note: the model was trained with PPISP — baking it for the viewer\'s screen-space pass')
    return state, checkpoint_config, active_degree, has_ppisp


def extract_tensors(state, with_mlp):
    """Pulls the Gaussian attributes (and, for the Neural appearance model, the
    residual mlp parameters) out of the checkpoint."""
    import torch
    get = lambda suffix: find_tensor(state, suffix).detach().float().numpy()
    quats_wxyz = get('_rotations')
    quats_wxyz /= np.linalg.norm(quats_wxyz, axis=1, keepdims=True)
    return dict(
        means=get('_means'),
        base_colors=get('appearance._base_colors'),
        residual_params=get('appearance._residual_params'),
        opacities=1.0 / (1.0 + np.exp(-get('_opacities').reshape(-1))),
        scales=np.exp(get('_scales')),
        quats=quats_wxyz[:, [1, 2, 3, 0]],  # checkpoint stores (w, x, y, z)
        # the fused kernels evaluate the mlp in fp16, so the export rounds the same way
        params=find_tensor(state, 'residual_mlp.params').detach().to(torch.float16).numpy().reshape(-1)
        if with_mlp else None,
    )


def gate_inactive_degrees(matrices, encoded_dim, sh_degrees, active_degree):
    """Bakes the appearance schedule into the weights: the input columns of the not-yet-enabled
    view-direction degrees are zeroed, exactly as the rasterizer zeroes their values."""
    gated = []
    offset = encoded_dim
    rank = 0
    for degree in sh_degrees:
        width = 2 * degree + 1
        if degree > 0:
            if rank >= active_degree:
                matrices[0][:, offset:offset + width] = 0.0
                gated.append(degree)
            rank += 1
        offset += width
    return gated


def tcnn_self_check(matrices, config, n_in):
    """Compare the sliced matrices against a live tcnn model fed the same params."""
    try:
        import torch
        import tinycudann as tcnn
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA not available')
    except Exception as e:
        print(f'tcnn self-check skipped ({e})')
        return
    model = tcnn.NetworkWithInputEncoding(
        n_input_dims=n_in, n_output_dims=3,
        encoding_config={'otype': 'Identity'},
        network_config={
            'otype': 'FullyFusedMLP', 'activation': 'ReLU', 'output_activation': 'None',
            'n_neurons': config['n_neurons'], 'n_hidden_layers': config['n_hidden_layers'],
        },
    )
    flat = np.concatenate([m.reshape(-1) for m in matrices[:-1]] + [
        np.concatenate([matrices[-1], np.zeros((13, config['n_neurons']), np.float32)]).reshape(-1)
    ])
    with torch.no_grad():
        model.params.copy_(torch.from_numpy(flat).to(model.params.dtype))
        x = torch.rand(256, n_in, device='cuda') * 2.0 - 1.0
        got = model(x).float().cpu().numpy()
    h = x.cpu().numpy()
    for i, w in enumerate(matrices):
        h = h @ w.astype(np.float16).astype(np.float32).T
        if i < len(matrices) - 1:
            h = np.maximum(h, 0.0)
    err = np.abs(got - h).max()
    print(f'tcnn self-check: max |Δ| = {err:.5f} {"OK" if err < 5e-2 else "MISMATCH — check layout!"}')


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Export FasterGSVDA checkpoint to .ngsplat')
    parser.add_argument('checkpoint',
                        help='torch checkpoint path, or a FasterGSVDA output directory '
                             '(loads checkpoints/final.pt and the renderer flags from training_config.yaml)')
    parser.add_argument('--out', default=None, help='output .ngsplat path')
    parser.add_argument('--dump-keys', action='store_true', help='list checkpoint tensors and exit')
    # camera hint overrides; by default the up vector and the orbit distance are
    # derived from the baked test cameras (see camera_hint_from_test_cameras),
    # falling back to the COLMAP convention the training uses (-y is world up)
    parser.add_argument('--center', type=float, nargs=3, default=None)
    parser.add_argument('--up', type=float, nargs=3, default=None)
    parser.add_argument('--distance', type=float, default=None)
    parser.add_argument('--no-test-cameras', action='store_true',
                        help='skip baking the test-set viewpoints (no dataset reload)')
    parser.add_argument('--no-ppisp', action='store_true',
                        help='skip baking PPISP for a PPISP-trained model')
    args = parser.parse_args()

    ckpt_path, default_out, yaml_config, renderer = resolve_model_dir(args.checkpoint)
    add_nerficg_src_to_path()
    state, checkpoint_config, active_degree, has_ppisp = load_checkpoint(ckpt_path, args.dump_keys)
    has_ppisp = has_ppisp and not args.no_ppisp

    # the appearance config is stored in the checkpoint itself, so it wins over the
    # yaml (which may have been edited since training); defaults fill the rest
    config = dict(CONFIG_DEFAULTS)
    config.update(yaml_config)
    config.update(checkpoint_config)
    sources = [name for name, values in
               (('checkpoint', checkpoint_config), ('training_config.yaml', yaml_config)) if values]
    model_type = config['appearance_type']
    if model_type not in APPEARANCE_MODEL_IDS:
        parser.error(f'unknown APPEARANCE.TYPE "{model_type}" '
                     f'(supported: {", ".join(APPEARANCE_MODEL_IDS)})')
    print(f'appearance config from {" over ".join(sources) if sources else "defaults"}:')
    for key in CONFIG_PRINT_KEYS[model_type]:
        print(f'  {key} = {config[key]}')

    # validate the config against what the viewer implements
    if config['color_activation'] not in COLOR_ACTIVATION_CODES:
        parser.error(f'unknown COLOR_ACTIVATION "{config["color_activation"]}" '
                     f'(supported: {", ".join(COLOR_ACTIVATION_CODES)})')
    if config['base_activation'] not in ('none', 'exp'):
        parser.error(f'unknown BASE_ACTIVATION "{config["base_activation"]}" (supported: none, exp)')
    if config['residual_activation'] not in RESIDUAL_CODES:
        parser.error(f'unknown RESIDUAL_ACTIVATION "{config["residual_activation"]}" '
                     f'(supported: {", ".join(RESIDUAL_CODES)})')
    if model_type == 'neural':
        if config['n_hidden_layers'] < 1:
            parser.error(f'N_HIDDEN_LAYERS must be >= 1, got {config["n_hidden_layers"]}')
        if config['hidden_activation'] != 'relu':
            parser.error(f'unsupported HIDDEN_ACTIVATION "{config["hidden_activation"]}" '
                         f'(the viewer mlp implements relu hidden layers only)')
        if config['n_neurons'] % 4 != 0 or config['n_neurons'] < 4:
            parser.error(f'N_NEURONS must be a multiple of 4 >= 4, got {config["n_neurons"]}')
        degrees = config['sh_degrees']
        if degrees != sorted(set(degrees)) or any(degree < 0 or degree > 6 for degree in degrees):
            parser.error(f'SH_DEGREES must be strictly ascending in [0, 6], got {degrees}')
        if not any(degree > 0 for degree in degrees):
            parser.error(f'SH_DEGREES must contain at least one view-direction degree (>= 1), got {degrees}')

    tensors = extract_tensors(state, with_mlp=model_type == 'neural')
    means, base_colors = tensors['means'], tensors['base_colors']
    residual_params, params = tensors['residual_params'], tensors['params']
    opacities, scales, quats = tensors['opacities'], tensors['scales'], tensors['quats']

    if model_type == 'neural':
        # mlp input layout: [encoded features] + [SH_C0] + [SH degree values]; with BASE_INPUT the
        # raw base colors are 3 of the encoder's input dims (they count towards FEATURE_DIM)
        raw_feature_dim = residual_params.shape[1] + (3 if config['base_input'] else 0)
        if 'feature_dim' in config and config['feature_dim'] != raw_feature_dim:
            print(f'WARNING: the config says FEATURE_DIM={config["feature_dim"]} but the checkpoint holds '
                  f'{residual_params.shape[1]} residual feature dims (BASE_INPUT={config["base_input"]}) '
                  f'— is this the right config?')
        k = config['n_frequencies']
        encoded_dim = raw_feature_dim * (2 * k if k else 1)
        sh_degree_0_input = 0 in degrees
        direction_degrees = [degree for degree in degrees if degree > 0]
        n_direction_values = sum(2 * degree + 1 for degree in direction_degrees)
        n_in = encoded_dim + sum(2 * degree + 1 for degree in degrees)
        matrices = slice_mlp_params(
            params.astype(np.float32), n_in, config['n_neurons'], config['n_hidden_layers']
        )

        # bake the appearance schedule (degrees are enabled one by one during training)
        if active_degree is None:
            print(f'warning: no appearance schedule state in the checkpoint — assuming all '
                  f'{len(direction_degrees)} view-direction degrees are enabled')
            active_degree = len(direction_degrees)
        if active_degree == 0:
            # the rasterizer skips the mlp entirely at degree 0, which zeroing its direction inputs does not
            # reproduce (the features and the constant degree 0 value still drive the output), so the whole
            # residual is switched off by zeroing the output layer
            if config['residual_activation'] == 'softplus':
                sys.exit('error: the checkpoint has no active view-direction degree yet and RESIDUAL_ACTIVATION '
                         'softplus maps a zero mlp output to a non-zero residual, so the rasterizer\'s '
                         '"skip the mlp" behavior cannot be baked into the weights — export a later checkpoint')
            matrices[-1][:] = 0.0
            print('WARNING: the checkpoint has no view-direction degree enabled yet (active_degree=0) — the '
                  'rasterizer renders the base color only, so the exported mlp is zeroed to match')
        elif active_degree < len(direction_degrees):
            gated = gate_inactive_degrees(matrices, encoded_dim, degrees, active_degree)
            print(f'WARNING: the checkpoint has only {active_degree} of {len(direction_degrees)} view-direction '
                  f'degrees enabled (active_degree={active_degree}) — the input weights of degrees {gated} are '
                  f'zeroed to reproduce the rasterizer; export a final checkpoint for the full model')

        tcnn_self_check(matrices, config, n_in)

    # PPISP: camera 0's vignetting (self-check only) and response curve, and the
    # parameters of the training frame with the median exposure, come from the checkpoint
    ppisp_static, ppisp_default = None, None
    if has_ppisp:
        get = lambda suffix: find_tensor(state, suffix).detach().float().numpy()
        ppisp_static = (get('ppisp.model.vignetting_params')[0], ppisp_crf_curve(get('ppisp.model.crf_params')[0]))
        exposures = get('ppisp.model.exposure_params')
        median_frame = int(np.argsort(exposures)[len(exposures) // 2])
        ppisp_default = np.concatenate([[exposures[median_frame]], get('ppisp.model.color_params')[median_frame]]).astype(np.float32)
        print(f'PPISP: default parameters from training frame {median_frame} (median exposure {exposures[median_frame]:+.3f} EV)')

    # test-set viewpoints for the viewer's fixed-resolution benchmark mode (and the
    # PPISP controller predictions for them)
    test_cameras, view_ppisp = None, None
    if not args.no_test_cameras:
        import os
        if os.path.isdir(args.checkpoint):
            test_cameras, view_ppisp = load_test_cameras(args.checkpoint, ckpt_path, ppisp_static)
        else:
            print('note: bare checkpoint path — no training_config.yaml, so no test cameras are baked')
    if has_ppisp and view_ppisp is None and test_cameras is not None:
        view_ppisp = np.zeros((test_cameras.shape[0], 9), np.float32)

    # cull splats whose opacity can never contribute
    keep = opacities >= MIN_ALPHA
    n_culled = int((~keep).sum())
    means, base_colors, residual_params = means[keep], base_colors[keep], residual_params[keep]
    opacities, scales, quats = opacities[keep], scales[keep], quats[keep]
    n = means.shape[0]
    print(f'Splats: {n} ({n_culled} culled below alpha 1/255)')

    # pre-activation base color, range-quantized to rgb8 (scale/offset in header). The 0.5 gray
    # shift of the shifted color activations is folded in here when nothing sits between the raw
    # base and the shift; with the exp base activation the viewer adds it in the shader instead
    # (sigmoid and satexp take their input unshifted)
    color_shift = 0.5 if config['color_activation'] in SHIFTED_COLOR_ACTIVATIONS \
        and config['base_activation'] == 'none' else 0.0
    base = (color_shift + base_colors).astype(np.float32)
    d_min, d_max = float(base.min()), float(base.max())
    d_scale = max(d_max - d_min, 1e-6) / 255.0
    base_code = np.round((base - d_min) / d_scale).clip(0, 255).astype(np.uint32)
    print(f'Base pre-activation range: [{d_min:.4f}, {d_max:.4f}]')

    opacity8 = np.round(opacities * 255.0).clip(0, 255).astype(np.uint32)

    tex_height = (n + TEX_WIDTH - 1) // TEX_WIDTH
    n_padded = TEX_WIDTH * tex_height
    splat_words = pack_splat_texture(means, scales, quats, base_code, opacity8, n_padded)

    # per-splat residual payload + the per-model header slots (see the docstring)
    weight_texels = np.zeros((0, 4), '<u2')
    degree_mask = 0
    if model_type == 'neural':
        features_for_mlp = np.concatenate([residual_params, base_colors], axis=-1) if config['base_input'] else residual_params
        features_encoded = frequency_encode(features_for_mlp, k)
        # bake layer 0 unless the h0_static cache (one value per neuron) needs
        # more per-splat textures than the features it replaces (fetches are
        # the cost; ties go to baked) or exceeds the viewer's 2-texture cap
        use_baked = config['n_neurons'] <= 16 and \
            -(-config['n_neurons'] // 8) <= -(-encoded_dim // 8)
        if use_baked:
            # the encoded features and the constant SH_C0 input are view-independent,
            # so their share of layer 0 is baked here (see the module docstring): the
            # file stores h0_static per splat and only the view-direction columns of W0
            w0 = matrices[0]
            h0_static = features_encoded @ w0[:, :encoded_dim].T
            if sh_degree_0_input:
                h0_static = h0_static + SH_C0 * w0[:, encoded_dim]
            h0_static = h0_static.astype(np.float16).astype(np.float32)
            n_static_cols = encoded_dim + (1 if sh_degree_0_input else 0)
            dyn_in = (n_direction_values + 3) // 4 * 4
            w0_dyn = np.zeros((config['n_neurons'], dyn_in), np.float32)
            w0_dyn[:, :n_direction_values] = w0[:, n_static_cols:]
            param_textures = pack_half_textures(h0_static, n_padded)
            weight_texels = pack_weight_texels([w0_dyn] + matrices[1:])
        else:
            if encoded_dim > 16:
                sys.exit(f'error: {encoded_dim} encoded feature dims exceed the viewer\'s cap of '
                         f'16 (2 feature textures) and {config["n_neurons"]} neurons exceed the '
                         f'baked h0_static cache — this configuration is unrepresentable')
            print(f'note: {config["n_neurons"]} neurons need a larger per-splat cache than the '
                  f'{encoded_dim} encoded features — writing the full layer-0 layout '
                  f'(stored features + full layer-0 matrix)')
            param_textures = pack_half_textures(features_encoded, n_padded)
            weight_texels = pack_weight_texels(matrices)
        for degree in direction_degrees:
            degree_mask |= 1 << degree
        header_model_dims, header_frequencies = encoded_dim, k
        header_neurons, header_hidden = config['n_neurons'], config['n_hidden_layers']
    else:
        sh_scales = None
        if model_type == 'sh':
            coefficients, degree_mask, export_degree = build_sh_payload(residual_params, config, active_degree)
            param_textures, sh_scales, sh_dequantized = pack_sh_quantized(coefficients, n_padded)
            header_model_dims = 0
        else:
            if model_type == 'sv':
                param_values, export_count = build_sv_payload(
                    residual_params, config, config['residual_activation'])
            elif model_type == 'nasg':
                param_values, export_count = build_nasg_payload(
                    residual_params, config, config['residual_activation'], active_degree)
            else:  # NASGabor
                param_values, export_count = build_nasgabor_payload(
                    residual_params, config, config['residual_activation'], active_degree)
            header_model_dims = export_count
            if param_values.shape[1] > PARAM_VALUE_CAP:
                sys.exit(f'error: {param_values.shape[1]} residual values per splat exceed the viewer\'s '
                         f'cap of {PARAM_VALUE_CAP} (8 fp16 textures) — reduce the model size')
            param_textures = pack_half_textures(param_values, n_padded)
        header_frequencies = 0
        header_neurons = header_hidden = 0

    # camera hint — robust to the far-flung background/floater gaussians that
    # MCMC-trained mipnerf360 scenes contain (max distance can be 1000x the scene
    # core): center on the per-axis median and orbit at 1.5x the median distance
    center = np.array(args.center, np.float32) if args.center else np.median(means, axis=0).astype(np.float32)
    median_radius = float(np.median(np.linalg.norm(means - center, axis=1))) if n else 1.0
    # up + orbit distance come from the training cameras when they are available
    up, distance, hint_source = None, None, 'test cameras'
    if test_cameras is not None:
        up, distance = camera_hint_from_test_cameras(test_cameras, center)
        if up is None:
            print('warning: the averaged camera up direction degenerates — falling back to -y')
    if args.up is not None:
        up, hint_source = np.array(args.up, np.float32), '--up'
    elif up is None:
        up, hint_source = np.array([0.0, -1.0, 0.0], np.float32), 'COLMAP convention'
    up = up / np.linalg.norm(up)
    if args.distance is not None:
        distance = args.distance
    elif distance is None:
        distance = 1.5 * median_radius
    tilt = math.degrees(math.acos(min(1.0, abs(float(np.dot(up, [0.0, -1.0, 0.0]))))))
    print(f'Camera hint: center {np.round(center, 3).tolist()}, distance {distance:.2f}, '
          f'up {np.round(up, 4).tolist()} (from {hint_source}, {tilt:.1f}° off -y)')

    # sanity: reference colors for a camera on the +x axis of the scene,
    # evaluated from the (quantized) stored payload values
    cam = center + distance * np.array([1.0, 0.0, 0.0], np.float32)
    dirs = means[:4] - cam
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    if model_type == 'neural':
        mlp_out = reference_forward(matrices, features_encoded[:4], dirs, direction_degrees, sh_degree_0_input)
        if use_baked:
            # the baked path (stored fp16 h0_static + sliced dynamic columns) must match
            # the full-input evaluation up to the single fp16 rounding of h0_static
            h = h0_static[:4] + eval_sh_degrees(dirs, direction_degrees) \
                @ w0_dyn[:, :n_direction_values].astype(np.float16).astype(np.float32).T
            for w_layer in matrices[1:]:
                h = np.maximum(h, 0.0) @ w_layer.astype(np.float16).astype(np.float32).T
            baked_err = float(np.abs(h - mlp_out).max())
            print(f'static-bake self-check: max |Δ| vs full evaluation = {baked_err:.5f} '
                  f'{"OK" if baked_err < 5e-2 else "MISMATCH — check the layer-0 split!"}')
        residual = apply_residual_activation(mlp_out, config['residual_activation'])
    elif model_type == 'sh':
        residual = sh_residual_reference(sh_dequantized[:4], dirs, config['residual_activation'])
    else:
        stored = param_values[:4].astype(np.float16).astype(np.float32)
        residual = {'sv': sv_residual_reference, 'nasg': nasg_residual_reference,
                    'nasgabor': nasgabor_residual_reference}[model_type](stored, dirs)
    base_term = np.exp(3.0 * base[:4]) if config['base_activation'] == 'exp' else base[:4]
    shader_shift = 0.5 if config['color_activation'] in SHIFTED_COLOR_ACTIVATIONS \
        and config['base_activation'] == 'exp' else 0.0
    colors = apply_color_activation(base_term + residual + shader_shift, config['color_activation'])
    if config['color_activation'] == 'none':
        print('note: COLOR_ACTIVATION none produces unbounded per-splat colors; the viewer\'s '
              'rgba8 color cache clamps them to [0, 1] per splat (the rasterizer instead clamps '
              'after blending — the same approximation the bounded activations make above 1)')
    print('Sample colors (first 4 splats, +x camera):')
    for row in colors:
        print(f'  [{row[0]: .4f} {row[1]: .4f} {row[2]: .4f}]')

    out_path = args.out or default_out
    print(f'Writing {out_path} ...')
    with open(out_path, 'wb') as f:
        f.write(b'NGSPLAT\n')
        # flags bits 3-4 + bit 6 carry the appearance model id (0 = Neural); bit 1
        # (constant SH_C0 input disabled) and bit 2 (baked layer 0) are Neural-only
        model_id = APPEARANCE_MODEL_IDS[model_type]
        flags = (1 if renderer.get('PROPER_ANTIALIASING') else 0) \
            | ((model_id & 3) << 3) | ((model_id >> 2) << 6)
        if config['base_activation'] == 'exp':  # bit 5 = exp base activation
            flags |= 32
        if ppisp_static is not None:  # bit 7 = PPISP block
            flags |= 128
        if model_type == 'neural':
            flags |= (0 if sh_degree_0_input else 2) | (4 if use_baked else 0)
        residual_code = RESIDUAL_CODES[config['residual_activation']]
        f.write(struct.pack(
            '<11I', n, TEX_WIDTH, tex_height,
            header_model_dims, header_frequencies, degree_mask,
            COLOR_ACTIVATION_CODES[config['color_activation']], residual_code,
            header_neurons, header_hidden,
            flags,
        ))
        f.write(struct.pack('<2f', d_scale, d_min))
        if model_type == 'sh':
            # per-level quantization scales of the packed SH coefficients
            f.write(struct.pack('<3f', *sh_scales))
        f.write(struct.pack('<3f', *center.tolist()))
        f.write(struct.pack('<3f', *up.tolist()))
        f.write(struct.pack('<f', float(distance)))
        if test_cameras is None:
            f.write(struct.pack('<I', 0))
        else:
            f.write(struct.pack('<I', test_cameras.shape[0]))
            f.write(test_cameras.astype('<f4').tobytes())
        if ppisp_static is not None:
            f.write(ppisp_static[1].astype('<f4').tobytes())
            f.write(ppisp_default.astype('<f4').tobytes())
            if test_cameras is not None:
                f.write(view_ppisp.astype('<f4').tobytes())
        f.write(struct.pack('<I', weight_texels.shape[0]))
        f.write(weight_texels.tobytes())
        f.write(splat_words.tobytes())
        for tex in param_textures:
            f.write(tex.tobytes())

    import os
    bytes_per_splat = 16 + 4 * sum(tex.shape[1] for tex in param_textures)
    print(f'Done. {os.path.getsize(out_path) / 1024 / 1024:.2f} MB '
          f'({bytes_per_splat} B/splat + {weight_texels.shape[0]} weight texels)')


if __name__ == '__main__':
    main()
