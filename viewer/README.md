# FasterGSVDA WebGL Viewer

The web viewer of [Compact Neural Appearance Models for Efficient Gaussian
Splatting](https://fhahlbohm.github.io/efficient-gaussian-appearance/), hosted at
https://fhahlbohm.github.io/efficient-gaussian-appearance/viewer/.

A standalone Three.js viewer for `.ngsplat` exports (written by
`export_ngsplat.py`): standard 3D Gaussian Splatting (EWA projection,
CPU-sorted back-to-front alpha blending) where the per-splat view-dependent
color comes from one of five appearance models, selected by the file header:

- **SH** — per-splat spherical harmonics up to degree 3, quantized to the
  packed 7/8/6-bit-per-level layout Spark uses;
- **SV** — a soft spherical Voronoi partition of the direction sphere
  (per-splat sites with temperatures and colors, softmax-blended);
- **NASG** — normalized anisotropic spherical Gaussian lobes;
- **NASGabor** — NASG lobes with an additional Gabor term;
- **Neural** — a small shared residual MLP over per-splat features (default: 2
  hidden layers × 16 neurons, ReLU, **no biases**, fp16 weights —
  tiny-cuda-nn `FullyFusedMLP`).

## Color model

Per splat, mirroring the training code in this repository (`Renderer.py` and
the `FasterGSVDACudaBackend` rasterizer, called the reference implementation
below):

```
residual = the appearance model's view-dependent residual(dir)   # dir = normalize(mean − camera)
color    = color_activation(base_activation(base) + residual)
```

Every activation of the reference implementation is supported: color
activations relu / softplus(β10) / sigmoid(4x) / satexp / hardsigmoid / none
(codes 0–5), the identity and exp(3x) base activations (header flags bit 5),
and residual activations none / tanh / softplus(β10) for every model. The 0.5
gray shift of the shifted color activations (none/relu/softplus/hardsigmoid) is
pre-baked into the stored base `d` with the identity base; with the exp base
it is applied in the shader, since the exp sits between the raw base and the
shift. The `none` color activation produces unbounded per-splat colors, which
the RGBA8 color cache of files without PPISP clamps to [0, 1] per splat — the
rasterizer instead clamps after blending, the same approximation the other
activations make above 1 (files with PPISP keep the radiance, see below).

Everything view-independent is precomputed at export time, so the per-frame
shader work is only the direction-dependent part of each model:

- **SH**: `residual = residual_activation(Σ basisᵢ(dir) · rgbᵢ)` — the shared
  residual activation applies to the band **sum** (as in the reference
  renderer), so the coefficients are stored raw, quantized to the packed
  per-level layout with shared max|coefficient| scales from the header.
- **SV**: `residual = Σ softmaxₛ(−temperatureₛ ·
  ‖siteₛ − dir‖) · colorₛ`, evaluated with a running maximum in a single pass
  over the site data. Sites are stored unit-normalized, temperatures exp'd,
  and colors activated, so the shader only runs the softmax.
- **NASG / NASGabor**: `residual = Σ pdfₗ(dir) · weightₗ` with the
  anisotropic spherical Gaussian lobe response. The lobe frame vectors,
  activated shape parameters, the normalization constant, and the activated
  weights are all baked at export time, so the shader evaluates just the
  direction-dependent response (two dot products, one `pow`, one `exp` per
  lobe — plus one `cos` for NASGabor's Gabor term).
- **Neural**: `residual = residual_activation(MLP([features, (SH_C0),
  sh_degrees(dir)]))` (3 outputs, no biases). The encoded features and the
  constant SH_C0 input are view-independent, so the exporter pre-multiplies
  their share of the first MLP layer into a per-splat `h0_static` (one fp16
  value per neuron, stored in place of the features) and keeps only the
  view-direction columns of layer 0 in the weight stream — the viewer starts
  layer 0 from the cache and evaluates just the direction inputs (for the
  default 32→16→16→3 architecture that removes 31% of the per-splat MACs at
  identical file size). When the bake would not pay off (the h0_static cache
  needing more textures than the features it replaces, or more than two), the
  exporter writes the full layout instead: the encoded features and the whole
  layer-0 matrix. Both layouts load (header flags bit 2) and produce identical
  output.

The **Shading** control mirrors the reference renderer's debug switches:

- **Full** — `RENDER_BASE_COLOR` + `RENDER_RESIDUAL_COLOR` on
- **Base color only** — `RENDER_RESIDUAL_COLOR` off: the rasterizer treats the
  appearance degree as 0 and skips the MLP entirely, so the viewer renders
  `color_activation(base)`
- **Residual only** — `RENDER_BASE_COLOR` off: `|full − base-only|`, exactly the
  `if (!render_base)` branch of the CUDA preprocess kernel

URL equivalents: `?residual=0` / `?base=0`.

## Color evaluation in the shader

The residual runs in a **capture pass** (`shaders/capture.frag`): a fragment
shader over one texel per splat that writes the finished color + opacity into
the `colorCache` (RGBA8, or RGBA16F for PPISP files with radiance above 1),
re-run **only when the camera moves** (colors depend on the camera position,
not the pixel). The per-frame splat pass just reads the
cache. The shared scaffolding (`shaders/capture_common.glsl` — uniforms, color
activation, behind-plane early-out, base/residual shading toggles) is spliced
together with the active appearance model's chunk (`eval_sh` / `eval_sv` /
`eval_nasg` / `eval_nasgabor` / `eval_neural.glsl`), each defining
`residualColor(texel, dir)`.
Model dims / activations are `#define`d into the shader from the file header
at load time (`utils.js` `captureDefines`).

For the Neural model, layer 0 starts from the per-splat `h0_static` cache and
accumulates only the view-direction SH inputs (zero-padded to a multiple of
4); for full-layout files the same shader (`BAKED 0`) assembles the full input
vector and starts layer 0 at zero. Weights live in a 64-wide RGBA16F texture
(raw fp16 bits); one texel = 4 consecutive input weights of one output neuron;
layers accumulate `input_block * mat4(4 texels)` per 4-wide output block, and
the 3-row output layer reads 3 texels per input block. No biases → pure matrix
products.

Robustness / tuning features:

- a **vertex-stage** capture fallback (`capture.vert`) for drivers whose
  fragment-stage integer `texelFetch` returns constants (Samsung Android), with
  a startup probe that picks the stage by comparing readbacks against the CPU
  reference in `utils.js` (see Verification below);
- for Neural files, an **MLP precision** toggle (fp16 default, mobile-only
  effect) and an **MLP weights** uniform-array (default, within the device
  uniform budget) vs texture backing;
- fatal errors / shader logs / context loss in an on-page overlay;
- memory-conscious loading: the download streams into one preallocated buffer
  (no Blob, which fails above a few hundred MB), and every texture payload's
  host copy is dropped right after its upload.

## PPISP

The paper's models were trained with [PPISP](https://research.nvidia.com/labs/sil/projects/ppisp/),
which post-processes the rendered radiance per image: exposure and an
8-parameter color correction per frame (predicted for novel views by a small
controller network from the rendering), then per-camera vignetting and a
per-channel camera response curve. The viewer reproduces this in a
**screen-space pass** (`shaders/ppisp.frag`) without the vignetting, a lens
artifact PPISP factors out of the radiance: the splats render into an
offscreen target and a fullscreen pass applies exposure, color correction, and
camera 0's response curve. The exporter bakes the response curve, the
controller's prediction for every test view (it renders the test views through
the framework and checks its numpy port of PPISP against the CUDA output), and
the parameters of the training frame with the median exposure as the default.
A pinned test view uses its own prediction; navigating uses the default.

Models with an unbounded color activation (relu, softplus, none) represent
overexposed regions as radiance far above 1, which the response curve maps to
white. For such files the color cache is RGBA16F, so per-splat radiance above 1
survives as in the CUDA rasterizer, and the **Blend target** is selectable:
RGBA16F (the default) is exact, RGBA8 halves the blending bandwidth but clips
the accumulated radiance at 1 mid-blend, which is visible in bright regions;
the GUI's frame time reflects the selected target. Files with a bounded
activation (sigmoid, hardsigmoid, satexp) keep the radiance in [0, 1] and use
RGBA8 throughout, which is exact. Devices without half-float render targets
fall back to RGBA8 throughout.

The **PPISP** toggle in the GUI disables the pass, and the residual-only
shading skips it like the reference renderer. The benchmark excludes PPISP
entirely: no pass, RGBA8 cache and target like files without it (the CUDA
benchmark of the reference implementation includes PPISP). Files of models
trained without PPISP carry no PPISP block and render as before.

## Baked test cameras + benchmark

The exporter reloads the training dataset and bakes the **test-subset
viewpoints** (c2w in the checkpoint's world space + native intrinsics) into
the file. The viewer uses them for:

- the **initial camera pose** (test view 0) and a **Test view** GUI slider
  (`?view=N`) to jump to any test viewpoint (native vertical FOV; dragging
  returns to the free camera);
- the **Benchmark** button (at the bottom of the collapsed **Benchmark** GUI
  folder), timing the file's appearance model: every test viewpoint is applied
  with its exact pose and native intrinsics at a **fixed viewport** (default
  1280×720; intrinsics are kept, exactly like `scripts/inference.py
  --benchmark` in NeRFICG, so the 720p numbers are directly comparable to it).
  First an **untimed warmup
  pass over the whole test set** (sort + one render per view — shader
  compilation and driver warm-up), then per view a full re-sort (awaiting the
  worker), one untimed warmup render absorbing the fresh sort-attribute
  upload, and the configured **timed renders** (default 10); every render
  includes the capture pass and ends in a 1×1 readback (timings include GPU
  completion, not vsync-quantized). By default sorting and its attribute
  upload stay outside the timing, matching the viewer's amortized async-sorter
  design — the numbers are the steady-state frame cost at a novel camera
  position. The run reports avg/med/min/max ms over per-view means and FPS;
  all other settings (shading, and for Neural files precision and weight
  backing) are used as configured. The button itself shows run progress.
- the **Benchmark** folder's settings above the button: **Resolution**
  (360p/480p/720p/1080p/1440p/2160p, all 16:9; the native intrinsics stay
  fixed, so other presets crop/extend the field of view exactly like the 720p
  reference does), **Timed renders / view**, and **Time sorting** — the latter
  folds a forced re-sort (worker roundtrip + order upload) into every timed
  render for an end-to-end novel-view cost; the camera is nudged by ±0.002
  scene units per sample to defeat the sort worker's same-camera dedup, which
  is visually and cost-wise negligible. Settings are snapshotted when a run
  starts.

Files without baked cameras (bare-checkpoint exports) fall back to a seeded
random-orbit sequence of 16 views.

## .ngsplat format

Produced by `export_ngsplat.py` (see its docstring for the exact binary
layout). Ready-to-upload texture payloads; the only derived data is the float
splat centers for the CPU sorter:

- `splatData` RGBA32UI, 1 texel/splat: word0 = pre-activation base rgb8
  (range-coded; scale/offset in the header) + opacity a8; words 1–3 = xyz fp16
  + quat 24-bit folded-octahedral + log-scales 3×8b (`shaders/splat_decode.glsl`).
- per-splat residual parameter textures, one appearance model per file (header
  flags bits 3–4 + bit 6): SH — one packed texture per degree level (RG32UI
  9×7-bit / RGBA32UI 15×8-bit / 21×6-bit signed codes, per-level scales in the
  header); SV / NASG / NASGabor — fp16 value streams of 7 per site / 12 or 13
  per lobe (activations and lobe frames pre-baked), up to 8 textures; Neural —
  1–2 RGBA32UI textures of 8 fp16 values each (`h0_static`, the baked
  view-independent share of MLP layer 0, or the encoded features in the full
  layout) plus the MLP weights as fp16 bits in capture-shader texel order
  (layer 0 = the view-direction columns only, zero-padded to a multiple of 4 —
  or the full trained matrix in the full layout; output trimmed to 3×N; tcnn's
  padded output rows dropped).
- test cameras: 18 f32 each (c2w 3×4 row-major, COLMAP right/down/forward
  convention + fx, fy, cx, cy, width, height).
- PPISP block (header flags bit 7): camera 0's activated response curve
  (3 × 6 f32), the default exposure and 8 color latents (9 f32, from the
  training frame with the median exposure), then per test view the
  controller's exposure and 8 color latents (9 f32).

## Export

```bash
# run from the NeRFICG root so the relative DATASET.PATH in training_config.yaml
# resolves and the test cameras can be baked; writes <output_dir>/scene.ngsplat
python src/Methods/FasterGSVDA/viewer/export_ngsplat.py output/FasterGSVDA/<run>
```

## Running

The hosted viewer at https://fhahlbohm.github.io/efficient-gaussian-appearance/viewer/
serves the gallery from `models.json` (scenes on Hugging Face). Locally:

```bash
python3 -m http.server 8000   # from this directory
# http://localhost:8000/                        gallery from models.json
# http://localhost:8000/?model=<path or url>    any .ngsplat, e.g. models/garden.ngsplat
```

Gallery entries come from `models.json`. Each scene carries one file per
appearance model: an entry with an `id` resolves to
`<baseUrl>/<model>/<id>.ngsplat` for every model in the manifest's
`appearances` list (or the entry's own `appearances`); an entry may instead
give an explicit `variants` map (model → url) or a single `url` with its
`appearance`. URLs may be absolute or relative to the served page; the
gitignored `models/` folder is a convenient place for local exports. To host
the viewer elsewhere, serve `index.html`, `main.js`, `utils.js`, `sorter.js`,
`styles.css`, `models.json`, `dummy_params.json`, and `shaders/`.

Each dataset section has a display filter next to its title (default Neural,
or the manifest's `defaultAppearance`): it selects which model's file of every
scene is listed, and **All** lists one card per file.

In the **Paper Benchmark** section of single checkpoints the selector has no
**All** and instead decides which model the picked file runs under
(`?appearance=` presets it; the checkpoints are Neural, the default). When
the file does not carry that model, its residual parameters are replaced by
dummy values sized by `dummy_params.json` (defaults mirror the training
configs: SH degree 3, 7 Voronoi sites, 1 NASG / NASGabor lobe, Neural 16×2
with degrees 0–3; edit the json to benchmark other sizes), while the
Gaussians, cameras, and activation codes are kept. Every appearance model can
thus be benchmarked on the *same* set of Gaussians, one run per model (fair on
devices that throttle under sustained load). The residual colors are
meaningless then, but every model's evaluation cost is data-independent, so
the timings are representative; the model info panel and the benchmark
results mark such runs with "dummy params".

## URL parameters

| Parameter | Values | Effect |
|---|---|---|
| `model` | `.ngsplat` path or URL | load that file directly, skipping the gallery |
| `appearance` | `sh`, `sv`, `nasg`, `nasgabor`, `neural` | force the appearance model (dummy parameters if the file lacks it) |
| `view` | test view index | start on that baked test viewpoint |
| `residual` | `0` | start in "Base color only" shading |
| `base` | `0` | start in "Residual only" shading |

## Verification

Some mobile drivers return wrong values from fragment-stage integer texture
fetches (seen on Samsung Android GPUs) without any GL error, so only a
functional check can catch it: at startup, the first 64 splats are evaluated on
the CPU (`utils.js`, which implements all five appearance models) and compared
against the capture pass. A mismatch > 3/255 falls back to vertex-stage
capture and logs a warning.
