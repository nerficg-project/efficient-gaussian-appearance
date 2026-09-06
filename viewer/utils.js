// ============================================================================
// Fetch / file helpers
// ============================================================================

export async function fetchFile(path, type = "text") {
    const response = await fetch(path);
    if (!response.ok) throw new Error(`Unable to load ${path} (Error ${response.status})`);
    if (type === "none") return response;
    return response[type]();
}

// Fetch a URL into an ArrayBuffer with progress reporting. Chunks go into one
// preallocated buffer (grown when the size is unknown); a Blob would double the
// peak memory and fails on very large files.
export async function readUrlWithProgress(url, onProgress) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Unable to load ${url} (Error ${res.status})`);
    const total = Number(res.headers.get("content-length")) || 0;
    const reader = res.body.getReader();
    let data = new Uint8Array(total || 1 << 24);
    let offset = 0;
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (offset + value.length > data.length) {
            const grown = new Uint8Array(Math.max(data.length * 2, offset + value.length));
            grown.set(data.subarray(0, offset));
            data = grown;
        }
        data.set(value, offset);
        offset += value.length;
        if (total) onProgress(offset / total);
    }
    if (!total) onProgress(1);
    return offset === data.length ? data.buffer : data.buffer.slice(0, offset);
}

// Files of a manifest entry per appearance model: a `variants` map (model -> url),
// `<baseUrl>/<model>/<id>.ngsplat` per listed model, or a single `url` with its
// `appearance`.
function entryVariants(manifest, entry) {
    if (entry.variants) return entry.variants;
    if (entry.id) {
        const models = entry.appearances ?? manifest.appearances ?? APPEARANCE_MODELS;
        return Object.fromEntries(models.map((m) => [m, `${manifest.baseUrl}/${m}/${entry.id}.ngsplat`]));
    }
    return { [entry.appearance ?? "neural"]: entry.url };
}

// display order of the appearance models in the filters
const MODEL_ORDER = ["sh", "sv", "nasg", "nasgabor", "neural"];

// Segmented control: one button per [value, label] under a sliding highlight.
// layout() positions the highlight once the element is in the DOM; setting
// `value` moves it without firing onChange.
function segmentedControl(options, initial, onChange) {
    const root = document.createElement("div");
    root.className = "segmented";
    const slider = document.createElement("div");
    slider.className = "segment-slider";
    root.append(slider);
    const segments = new Map();
    let value = initial;
    const select = (v, fire) => {
        value = v;
        for (const [k, seg] of segments) seg.classList.toggle("active", k === v);
        slider.style.left = `${segments.get(v).offsetLeft}px`;
        slider.style.width = `${segments.get(v).offsetWidth}px`;
        if (fire) onChange?.(v);
    };
    for (const [v, label] of options) {
        const seg = document.createElement("button");
        seg.type = "button";
        seg.className = "segment";
        seg.textContent = label;
        seg.addEventListener("click", () => select(v, true));
        root.append(seg);
        segments.set(v, seg);
    }
    window.addEventListener("resize", () => select(value, false));
    return {
        element: root,
        layout: () => select(value, false),
        get value() { return value; },
        set value(v) { select(v, false); },
    };
}

// Render the model grid, one section per dataset with a segmented selector next
// to its title: a display filter ("All" = one card per file), or for sections of
// single-file entries (the paper benchmark) the model to force the picked file
// under. Resolves with { name, url, override } of the clicked card.
export function chooseModel(sections, manifest, defaultFilter, defaultForce) {
    return new Promise((resolve) => {
        const el = (tag, className, text) => {
            const node = document.createElement(tag);
            node.className = className;
            if (text != null) node.textContent = text;
            return node;
        };
        const groups = new Map();
        for (const entry of manifest.models ?? []) {
            const key = entry.group ?? "Models";
            if (!groups.has(key)) groups.set(key, []);
            groups.get(key).push(entry);
        }
        const filterModels = MODEL_ORDER.filter((m) => (manifest.appearances ?? APPEARANCE_MODELS).includes(m));

        sections.replaceChildren();
        for (const [title, entries] of groups) {
            const grid = el("div", "gallery-grid");
            let force = null; // the model selector of a single-file section
            const render = (filter) => {
                grid.replaceChildren();
                for (const entry of entries) {
                    const variants = entryVariants(manifest, entry);
                    const name = entry.name ?? entry.id ?? entry.url;
                    const shown = filter === "all" ? Object.keys(variants) : (filter in variants ? [filter] : []);
                    for (const m of shown) {
                        const card = el("button", "card");
                        card.type = "button";
                        card.append(el("div", "card-title", name), el("div", "card-variants", APPEARANCE_NAMES[m]));
                        card.addEventListener("click", () => resolve({
                            name, url: variants[m],
                            override: force ? force.value : null,
                        }));
                        grid.append(card);
                    }
                }
                if (!grid.childElementCount)
                    grid.append(el("div", "grid-empty", `no ${APPEARANCE_NAMES[filter]} files in this set`));
            };
            const header = el("div", "section-header");
            header.append(el("div", "section-title", title));
            const section = el("div", "gallery-section");
            section.append(header, grid);
            sections.append(section);
            const filterable = entries.some((e) => Object.keys(entryVariants(manifest, e)).length > 1);
            if (filterable) {
                const control = segmentedControl(
                    [...filterModels.map((m) => [m, APPEARANCE_NAMES[m]]), ["all", "All"]], defaultFilter, render);
                header.append(control.element);
                control.layout();
            } else {
                force = segmentedControl(filterModels.map((m) => [m, APPEARANCE_NAMES[m]]), defaultForce);
                header.append(force.element);
                header.after(el("div", "section-note",
                    "Note: Non-neural models use the same Gaussians but with random view-dependent appearance parameters."));
                force.layout();
            }
            render(filterable ? defaultFilter : "all");
        }
    });
}

// ============================================================================
// .ngsplat loading
//
// The file holds the exact texture payloads: splatData (RGBA32UI: base rgb8 +
// opacity a8, pos fp16, quat 24b, log-scales 3x8b) plus the per-splat residual
// parameters of one appearance model (header flags bits 3-4 + bit 6):
//   Neural (0): fp16 textures (8 values per RGBA32UI texel) and the MLP weights
//     as raw fp16 bits in capture-shader texel order (3-row output layer). Two
//     layer-0 layouts, flags bit 2: BAKED stores h0_static (the pre-multiplied
//     view-independent share of layer 0, one fp16 per neuron) and only the
//     view-direction weight columns, zero-padded to /4; FULL stores the encoded
//     features and the whole [neurons x (features + SH_C0 + degrees)] matrix.
//   SH (1): quantized coefficients in Spark's packed degree-3 layout, one
//     texture per degree (RG32UI 9x7-bit / RGBA32UI 15x8-bit / 21x6-bit signed
//     codes, per-level scales in the header).
//   SV (2): 7 fp16 per site (unit direction, exp'd temperature, activated color).
//   NASG (3): 12 fp16 per lobe (frame x and z, 2*lambda, anisotropy,
//     normalization constant, activated rgb weights).
//   NASGabor (4): the NASG layout plus the Gabor frequency (13 fp16 per lobe).
// fp16 streams use 8 values per RGBA32UI texel, at most 8 textures. The file
// also carries the test-set viewpoints for the benchmark. Nothing is repacked;
// the only derived data is the float splat centers for the CPU sorter.
// ============================================================================

const MAGIC = "NGSPLAT\n";
const SH_C0 = 0.28209479177387814;
// header flags bits 3-4 + bit 6 (high bit); Neural = 0, the others follow the
// backend enum
export const APPEARANCE_MODELS = ["neural", "sh", "sv", "nasg", "nasgabor"];
export const APPEARANCE_NAMES = {
    sh: "SH", sv: "SV", nasg: "NASG", nasgabor: "NASGabor", neural: "Neural",
};
// header colorActivation codes
export const COLOR_ACTIVATION_NAMES = ["relu", "softplus (β10)", "sigmoid (4x)", "satexp", "hardsigmoid", "none"];
// color codes with the 0.5 gray shift (act(x + 0.5)): baked into the stored
// base with the identity base, applied in the shader with the exp base
const SHIFTED_COLOR_CODES = [0, 1, 4, 5];
export const RESIDUAL_ACTIVATION_NAMES = ["none", "tanh", "softplus (β10)"];
// packed SH level layout (shared with eval_sh.glsl): bits per signed field and
// u32 words per splat, degrees 1..3
const SH_PACK_LEVELS = [
    { bits: 7, words: 2 },
    { bits: 8, words: 4 },
    { bits: 6, words: 4 },
];

// float16 bit pattern -> float32, via a lazily built 64K lookup table.
let halfTable = null;
function buildHalfTable() {
    const table = new Float32Array(65536);
    const f32 = new Float32Array(1);
    const u32 = new Uint32Array(f32.buffer);
    for (let h = 0; h < 65536; h++) {
        const sign = (h & 0x8000) << 16;
        const exp = (h >> 10) & 0x1F;
        const mant = h & 0x3FF;
        if (exp === 0) {
            // subnormal: mant * 2^-24
            table[h] = (sign ? -1 : 1) * mant * 5.960464477539063e-8;
        } else if (exp === 31) {
            u32[0] = sign | 0x7F800000 | (mant << 13);
            table[h] = f32[0];
        } else {
            u32[0] = sign | ((exp - 15 + 127) << 23) | (mant << 13);
            table[h] = f32[0];
        }
    }
    return table;
}
export function halfToFloat(bits) {
    if (!halfTable) halfTable = buildHalfTable();
    return halfTable[bits & 0xFFFF];
}

export function loadNgsplat(buffer) {
    const magic = new TextDecoder().decode(new Uint8Array(buffer, 0, 8));
    if (magic !== MAGIC) throw new Error("Not a .ngsplat file (bad magic)");
    const dv = new DataView(buffer);
    let o = 8;
    const u32 = () => { const v = dv.getUint32(o, true); o += 4; return v; };
    const f32 = () => { const v = dv.getFloat32(o, true); o += 4; return v; };

    const numSplats = u32();
    const textureWidth = u32();
    const textureHeight = u32();
    // Neural: encoded feature dims | SH: 0 | SV: sites | NASG/NASGabor: lobes
    const modelDims = u32();
    const nFrequencies = u32(); // Neural only, informational
    // Neural: bit l = view-direction SH degree l (1..6) is an mlp input;
    // SH: bit l = degree l stored
    const shDegreeMask = u32();
    // 0 relu | 1 softplus(β10) | 2 sigmoid(4x) | 3 satexp | 4 hardsigmoid | 5 none
    const colorActivation = u32();
    // 0 none | 1 tanh | 2 softplus; functional for Neural/SH, baked otherwise
    const residualActivation = u32();
    const nNeurons = u32();
    const nHiddenLayers = u32();
    const flags = u32();
    const properAA = (flags & 1) !== 0;
    const sh0Input = (flags & 2) === 0; // bit 1 = constant SH_C0 mlp input DISABLED (Neural)
    const baked = (flags & 4) !== 0;    // bit 2 = layer 0 is baked (Neural; see the header comment)
    const appearance = APPEARANCE_MODELS[((flags >> 3) & 3) | (((flags >> 6) & 1) << 2)];
    const baseExp = (flags & 32) !== 0;  // bit 5 = exp base activation
    const baseScale = f32(); // base pre-activation d = code8 * scale + offset
    const baseOffset = f32();
    // per-level quantization scales of the packed SH coefficients (0 = absent)
    const shScales = appearance === "sh" ? [f32(), f32(), f32()] : null;
    const camCenter = [f32(), f32(), f32()];
    const camUp = [f32(), f32(), f32()];
    const camDistance = f32();

    // baked test-set viewpoints: c2w rows (3x4, COLMAP right/down/forward) +
    // native intrinsics
    const nTestCameras = u32();
    const testCameras = [];
    for (let i = 0; i < nTestCameras; i++) {
        const c2w = new Float32Array(12);
        for (let j = 0; j < 12; j++) c2w[j] = f32();
        testCameras.push({
            c2w,
            fx: f32(), fy: f32(), cx: f32(), cy: f32(),
            width: f32(), height: f32(),
        });
    }
    // PPISP block (flags bit 7): camera 0's response curve, the default exposure +
    // 8 color latents (the training frame with the median exposure), and the
    // controller's prediction per test view
    let ppisp = null;
    if (flags & 128) {
        const read = (n) => { const a = new Float32Array(n); for (let i = 0; i < n; i++) a[i] = f32(); return a; };
        ppisp = { crf: read(18), defaults: read(9), views: read(9 * nTestCameras) };
    }
    const nWeightTexels = u32();

    if (textureWidth !== 2048) throw new Error(`texWidth must be 2048 (render_ewa.vert hardcodes it), got ${textureWidth}`);
    if (shDegreeMask & ~0b1111110) throw new Error("shDegreeMask has bits outside degrees 1-6");
    if (colorActivation > 5) throw new Error(`unknown colorActivation code ${colorActivation}`);
    if (residualActivation > 2) throw new Error(`unknown residualActivation code ${residualActivation}`);

    const degrees = [];
    for (let l = 1; l <= 6; l++) if (shDegreeMask & (1 << l)) degrees.push(l);
    // per-model payload plan: u32 components per parameter texture
    let l0In = 0, paramComponents, nSites = 0, nLobes = 0;
    if (appearance === "neural") {
        if (nNeurons % 4 !== 0 || nNeurons < 4) throw new Error(`nNeurons must be a multiple of 4 >= 4, got ${nNeurons}`);
        if (nHiddenLayers < 1) throw new Error(`nHiddenLayers must be >= 1, got ${nHiddenLayers}`);
        const nDirectionValues = degrees.reduce((s, l) => s + 2 * l + 1, 0);
        // stored layer-0 input width, and fp16 values per splat
        let nPerSplatValues;
        if (baked) {
            // view-direction SH values only, zero-padded to /4
            if (!nDirectionValues) throw new Error("baked layout needs at least one view-direction SH degree");
            if (nNeurons > 16)
                throw new Error(`nNeurons must be <= 16 in the baked layout, got ${nNeurons} `
                    + `(the h0_static cache is capped at 2 textures = 16 fp16 values)`);
            l0In = Math.ceil(nDirectionValues / 4) * 4;
            nPerSplatValues = nNeurons;
        } else {
            // the full trained input: encoded features + SH_C0 + degrees
            if (modelDims < 1 || modelDims > 16)
                throw new Error(`unsupported featureDim ${modelDims} (max 2 feature textures = 16 encoded dims)`);
            l0In = modelDims + (sh0Input ? 1 : 0) + nDirectionValues;
            if (l0In % 4 !== 0) throw new Error(`MLP input dim ${l0In} is not a multiple of 4`);
            nPerSplatValues = modelDims;
        }
        // 3-row output layer: (nNeurons / 4) input blocks x 3 output texels each
        const expectedTexels =
            (l0In / 4) * nNeurons + (nHiddenLayers - 1) * (nNeurons / 4) * nNeurons + (nNeurons / 4) * 3;
        if (nWeightTexels !== expectedTexels)
            throw new Error(`weight texel count ${nWeightTexels} != expected ${expectedTexels} `
                + `for the ${baked ? "baked" : "full"} layer-0 layout`);
        paramComponents = Array.from({ length: Math.ceil(nPerSplatValues / 8) }, () => 4);
    } else {
        if (nWeightTexels !== 0)
            throw new Error(`the ${appearance} payload carries no weight stream, got ${nWeightTexels} texels`);
        if (appearance === "sh") {
            // one packed texture per stored level, contiguous from degree 1
            if (degrees.some((l, idx) => l !== idx + 1) || degrees.length > 3)
                throw new Error(`sh degrees must be contiguous 1..3, got mask ${shDegreeMask}`);
            paramComponents = degrees.map((l) => SH_PACK_LEVELS[l - 1].words);
        } else {
            const perUnit = { sv: 7, nasg: 12, nasgabor: 13 }[appearance];
            if (appearance === "sv") nSites = modelDims;
            else nLobes = modelDims;
            const nValues = modelDims * perUnit;
            if (nValues > 64)
                throw new Error(`${nValues} residual values per splat exceed the cap of 64 (8 fp16 textures)`);
            paramComponents = Array.from({ length: Math.ceil(nValues / 8) }, () => 4);
        }
    }

    // slice() copies into fresh, aligned ArrayBuffers.
    const take = (Type, count) => {
        const arr = new Type(buffer.slice(o, o + count * Type.BYTES_PER_ELEMENT));
        o += count * Type.BYTES_PER_ELEMENT;
        return arr;
    };
    const texels = textureWidth * textureHeight;
    const weightData = take(Uint16Array, nWeightTexels * 4); // raw fp16 bits (RGBA16F upload)
    const splatData = take(Uint32Array, texels * 4);
    // per-splat residual parameters (see the format comment above)
    const paramTexData = paramComponents.map((components) =>
        ({ data: take(Uint32Array, texels * components), components }));
    if (o !== buffer.byteLength)
        console.warn(`ngsplat: ${buffer.byteLength - o} trailing bytes ignored`);

    // Unpack the fp16 centers for the CPU depth sorter (the one derived buffer).
    const centers = new Float32Array(numSplats * 3);
    for (let i = 0; i < numSplats; i++) {
        const w1 = splatData[i * 4 + 1], w2 = splatData[i * 4 + 2];
        centers[i * 3] = halfToFloat(w1);
        centers[i * 3 + 1] = halfToFloat(w1 >>> 16);
        centers[i * 3 + 2] = halfToFloat(w2);
    }

    return {
        numSplats, textureWidth, textureHeight, appearance,
        featureDim: modelDims, nFrequencies, shDegreeMask, degrees,
        colorActivation, residualActivation, baseExp,
        nNeurons, nHiddenLayers, sh0Input, properAA, baked, l0In,
        nSites, nLobes, shScales,
        baseScale, baseOffset,
        camCenter, camUp, camDistance,
        testCameras, ppisp,
        weightData, nWeightTexels, splatData, paramTexData, centers,
        fileBytes: buffer.byteLength,
    };
}

// PPISP files with an unbounded color activation (relu, softplus, none) carry
// per-splat radiance above 1 that must survive the cache and the blending
export function unboundedRadiance(model) {
    return Boolean(model.ppisp) && [0, 1, 5].includes(model.colorActivation);
}

// ============================================================================
// PPISP: the color homography of the viewer's screen-space pass (exposure and
// response curve live in shaders/ppisp.frag; the vignetting is left out).
// ============================================================================

// ZCA blocks mapping the color latents to chromaticity offsets of the blue,
// red, green, and neutral control points (ppisp_math.cuh)
const PPISP_COLOR_PINV_BLOCKS = [
    [0.0480542, -0.0043631, -0.0043631, 0.0481283],
    [0.0580570, -0.0179872, -0.0179872, 0.0431061],
    [0.0433336, -0.0180537, -0.0180537, 0.0580500],
    [0.0128369, -0.0034654, -0.0034654, 0.0128158],
];

// 3x3 color homography (row-major) from the 8 latents, as compute_homography
export function ppispHomography(latents) {
    const off = PPISP_COLOR_PINV_BLOCKS.map((m, i) => [
        m[0] * latents[2 * i] + m[1] * latents[2 * i + 1],
        m[2] * latents[2 * i] + m[3] * latents[2 * i + 1],
    ]);
    const tb = [off[0][0], off[0][1], 1], tr = [1 + off[1][0], off[1][1], 1];
    const tg = [off[2][0], 1 + off[2][1], 1], tn = [1 / 3 + off[3][0], 1 / 3 + off[3][1], 1];
    const T = [[tb[0], tr[0], tg[0]], [tb[1], tr[1], tg[1]], [tb[2], tr[2], tg[2]]];
    const skew = [[0, -tn[2], tn[1]], [tn[2], 0, -tn[0]], [-tn[1], tn[0], 0]];
    const mul = (A, B) => A.map((row) => B[0].map((_, j) => row.reduce((acc, v, k) => acc + v * B[k][j], 0)));
    const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const M = mul(skew, T);
    let lam = cross(M[0], M[1]);
    if (dot(lam, lam) < 1e-20) {
        lam = cross(M[0], M[2]);
        if (dot(lam, lam) < 1e-20) lam = cross(M[1], M[2]);
    }
    const D = [[lam[0], 0, 0], [0, lam[1], 0], [0, 0, lam[2]]];
    const Sinv = [[-1, -1, 1], [1, 0, 0], [0, 1, 0]];
    const H = mul(mul(T, D), Sinv);
    const s = H[2][2];
    if (Math.abs(s) > 1e-20) for (const row of H) for (let j = 0; j < 3; j++) row[j] /= s;
    return H.flat();
}


// ============================================================================
// Appearance override (benchmark selector, ?appearance=): re-dress a loaded
// model with a different appearance model filled with dummy parameters from
// dummy_params.json, so any model can be benchmarked on the same gaussians.
// Every model's cost is data-independent, so the timings are representative
// while the residual colors are meaningless.
// ============================================================================

const _floatBits = new Float32Array(1);
const _uintBits = new Uint32Array(_floatBits.buffer);
// float32 -> fp16 bits by truncation (the dummy values are fp16 normals)
function floatToHalf(v) {
    _floatBits[0] = v;
    const b = _uintBits[0];
    return ((b >>> 16) & 0x8000) | ((((b >>> 23) & 0xFF) - 112) << 10) | ((b >>> 13) & 0x3FF);
}

// deterministic dummy words: a hash-filled tile, repeated (caches work on
// addresses, so repetition is timing-neutral)
function dummyWords(count, seed, makeWord) {
    const words = new Uint32Array(count);
    let h = seed | 0;
    const rand = () => {
        h = (Math.imul(h ^ (h >>> 15), 2654435761) + 0x9E3779B9) | 0;
        return (h >>> 0) / 4294967296;
    };
    const tile = Math.min(count, 4096);
    for (let i = 0; i < tile; i++) words[i] = makeWord(rand);
    for (let filled = tile; filled < count; ) {
        const n = Math.min(filled, count - filled);
        words.copyWithin(filled, 0, n);
        filled += n;
    }
    return words;
}

// fp16 dummy values in ±[0.1, 0.55]: no denormals, and every model stays in
// its main branch
const dummyHalf = (rand) => floatToHalf((rand() < 0.5 ? -1 : 1) * (0.1 + 0.45 * rand()));
const dummyHalfPair = (rand) => dummyHalf(rand) | (dummyHalf(rand) << 16);

export function applyAppearanceOverride(model, appearance, dummyConfigs) {
    if (appearance === model.appearance) return model;
    const config = dummyConfigs?.[appearance];
    if (!config) throw new Error(`dummy_params.json has no "${appearance}" configuration`);
    const texels = model.textureWidth * model.textureHeight;
    const halfTex = (seed) => ({ data: dummyWords(texels * 4, seed, dummyHalfPair), components: 4 });
    const out = {
        ...model,
        appearance,
        dummyParams: true,
        featureDim: 0, nFrequencies: 0, shDegreeMask: 0, degrees: [],
        sh0Input: false, baked: false, l0In: 0,
        nNeurons: 0, nHiddenLayers: 0, nWeightTexels: 0,
        weightData: new Uint16Array(0),
        nSites: 0, nLobes: 0, shScales: null,
    };
    if (appearance === "sh") {
        // a single maximum degree, like the reference configuration
        if (!(config.shDegree >= 1 && config.shDegree <= 3))
            throw new Error("dummy sh configuration: shDegree must be in 1..3 (the packed layout's cap)");
        out.degrees = Array.from({ length: config.shDegree }, (_, i) => i + 1);
        out.shDegreeMask = out.degrees.reduce((m, l) => m | (1 << l), 0);
        // dummy dequantization scales
        out.shScales = [0.3, 0.15, 0.08];
        // bit-packed coefficient codes: raw hash words decode to valid ints
        out.paramTexData = out.degrees.map((l, i) => ({
            data: dummyWords(texels * SH_PACK_LEVELS[l - 1].words, 17 + i, (rand) => (rand() * 4294967296) >>> 0),
            components: SH_PACK_LEVELS[l - 1].words,
        }));
    } else if (appearance === "sv") {
        out.nSites = config.nSites;
        out.paramTexData = Array.from({ length: Math.ceil(out.nSites * 7 / 8) }, (_, i) => halfTex(23 + i));
    } else if (appearance === "nasg" || appearance === "nasgabor") {
        const perLobe = appearance === "nasg" ? 12 : 13;
        out.nLobes = config.nLobes;
        out.paramTexData = Array.from(
            { length: Math.ceil(out.nLobes * perLobe / 8) }, (_, i) => halfTex(31 + i));
    } else {  // Neural
        // featureDim counts ENCODED inputs; degree 0 = the constant input
        out.featureDim = config.featureDim;
        out.nNeurons = config.nNeurons;
        out.nHiddenLayers = config.nHiddenLayers;
        out.degrees = config.degrees.filter((l) => l > 0);
        out.sh0Input = config.degrees.includes(0);
        out.shDegreeMask = out.degrees.reduce((m, l) => m | (1 << l), 0);
        if (out.featureDim > 16 || out.nNeurons % 4 !== 0)
            throw new Error("dummy neural configuration: featureDim is capped at 16 "
                + "(2 feature textures) and nNeurons must be a multiple of 4");
        const nDirectionValues = out.degrees.reduce((s, l) => s + 2 * l + 1, 0);
        // bake layer 0 unless the h0_static cache needs more textures than the
        // features it replaces or exceeds the 2-texture cap (then the full layout)
        out.baked = out.nNeurons <= 16
            && Math.ceil(out.nNeurons / 8) <= Math.ceil(out.featureDim / 8);
        if (out.baked) {
            out.l0In = Math.ceil(nDirectionValues / 4) * 4;
            out.paramTexData = Array.from(
                { length: Math.ceil(out.nNeurons / 8) }, (_, i) => halfTex(11 + i));
        } else {
            out.l0In = out.featureDim + (out.sh0Input ? 1 : 0) + nDirectionValues;
            if (out.l0In % 4 !== 0)
                throw new Error(`dummy neural configuration: mlp input width ${out.l0In} `
                    + "is not a multiple of 4 — adjust featureDim or degrees");
            out.paramTexData = Array.from(
                { length: Math.ceil(out.featureDim / 8) }, (_, i) => halfTex(11 + i));
        }
        out.nWeightTexels = (out.l0In / 4) * out.nNeurons
            + (out.nHiddenLayers - 1) * (out.nNeurons / 4) * out.nNeurons
            + (out.nNeurons / 4) * 3;
        const weightWords = dummyWords(out.nWeightTexels * 2, 7, dummyHalfPair);
        out.weightData = new Uint16Array(weightWords.buffer);
    }
    return out;
}

// input width of the TRAINED mlp (differs from model.l0In in the baked layout)
export function mlpInputDim(model) {
    return model.featureDim + (model.sh0Input ? 1 : 0)
        + model.degrees.reduce((s, l) => s + 2 * l + 1, 0);
}

// #define block for the capture shader. For the Neural chunk the degree
// offsets index the layer-0 input array: from 0 in the baked layout, after the
// features and SH_C0 in the full layout.
export function captureDefines(model, { renderBase, renderResidual, weightSource, precision }, clampedCache = false) {
    // with the exp base the 0.5 shift could not be baked into the stored base
    const shiftInShader = model.baseExp && SHIFTED_COLOR_CODES.includes(model.colorActivation);
    // a schedule-trimmed export without residual units compiles the residual out
    const hasResidual = model.appearance === "neural" || model.paramTexData.length > 0;
    const common = [
        `#define N_PARAM_TEX ${model.paramTexData.length}`,
        `#define COLOR_ACT ${model.colorActivation}`,
        `#define BASE_ACT ${model.baseExp ? 1 : 0}`,
        `#define COLOR_SHIFT ${shiftInShader ? "0.5" : "0.0"}`,
        `#define RENDER_BASE ${renderBase ? 1 : 0}`,
        `#define RENDER_RESIDUAL ${renderResidual && hasResidual ? 1 : 0}`,
        `#define COLOR_MAX ${unboundedRadiance(model) && !clampedCache ? "65504.0" : "1.0"}`,
    ];
    if (model.appearance === "sh")
        return [...common,
            `#define SH_DEGREE_MASK ${model.shDegreeMask}`,
            `#define RESIDUAL_ACT ${model.residualActivation}`,
            "",
        ].join("\n");
    if (model.appearance === "sv")
        return [...common, `#define N_SITES ${model.nSites}`, ""].join("\n");
    if (model.appearance === "nasg" || model.appearance === "nasgabor")
        return [...common, `#define N_LOBES ${model.nLobes}`, ""].join("\n");
    let off = model.baked ? 0 : model.featureDim + (model.sh0Input ? 1 : 0);
    const degreeOff = { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0 };
    for (const l of model.degrees) {
        degreeOff[l] = off;
        off += 2 * l + 1;
    }
    return [
        ...common,
        `#define BAKED ${model.baked ? 1 : 0}`,
        `#define N_L0_IN ${model.l0In}`,
        `#define N_FEATURES ${model.featureDim}`, // full layout only
        `#define SH0_INPUT ${model.sh0Input ? 1 : 0}`, // full layout only
        `#define N_NEURONS ${model.nNeurons}`,
        `#define N_HIDDEN ${model.nHiddenLayers}`,
        `#define N_WEIGHT_TEXELS ${model.nWeightTexels}`,
        `#define SH_DEGREE_MASK ${model.shDegreeMask}`,
        `#define RESIDUAL_ACT ${model.residualActivation}`,
        weightSource === "uniform" ? "#define WEIGHTS_UNIFORM" : "",
        precision === "fp16" ? "#define MLP_FP16" : "",
        `#define D1_OFF ${degreeOff[1]}`,
        `#define D2_OFF ${degreeOff[2]}`,
        `#define D3_OFF ${degreeOff[3]}`,
        `#define D4_OFF ${degreeOff[4]}`,
        `#define D5_OFF ${degreeOff[5]}`,
        `#define D6_OFF ${degreeOff[6]}`,
        "",
    ].join("\n");
}

// ============================================================================
// CPU reference of the capture-pass color model, the ground truth of the
// startup stage probe. Must match capture_common.glsl plus the active residual
// chunk exactly:
//   color = color_activation(base_activation(d) + residual(dir))
//   residual off -> color_activation(base); base off -> |full - base-only|
// ============================================================================

// MLP weight matrices from the fp16 texel stream (layer 0 is model.l0In wide)
export function buildMlpLayers(model) {
    const N = model.nNeurons;
    const readLayer = (baseTexel, outDim, inDim) => {
        const w = Array.from({ length: outDim }, () => new Float64Array(inDim));
        for (let j = 0; j < inDim / 4; j++)
            for (let o = 0; o < outDim; o++) {
                const t = (baseTexel + j * outDim + o) * 4;
                for (let k = 0; k < 4; k++)
                    w[o][4 * j + k] = halfToFloat(model.weightData[t + k]);
            }
        return w;
    };
    const layers = [readLayer(0, N, model.l0In)];
    let base = (model.l0In / 4) * N;
    for (let l = 1; l < model.nHiddenLayers; l++) {
        layers.push(readLayer(base, N, N));
        base += (N / 4) * N;
    }
    layers.push(readLayer(base, 3, N));
    return layers;
}

// SH basis polynomials, degrees 1-6 (matches eval_neural.glsl; the SH model
// uses the same ordering for its degrees 1..3)
function shDegreeValues(model, d) {
    const [x, y, z] = d;
    const xy = x * y, xz = x * z, yz = y * z, x2 = x * x, y2 = y * y, z2 = z * z;
    const x4 = x2 * x2, y4 = y2 * y2, z4 = z2 * z2;
    const x6 = x4 * x2, y6 = y4 * y2, z6 = z4 * z2;
    const per = {
        1: [-0.48860251190291987 * y, 0.48860251190291987 * z, -0.48860251190291987 * x],
        2: [
            1.0925484305920792 * xy, -1.0925484305920792 * yz,
            0.94617469575755997 * z2 - 0.31539156525251999,
            -1.0925484305920792 * xz,
            0.54627421529603959 * x2 - 0.54627421529603959 * y2,
        ],
        3: [
            0.59004358992664352 * y * (-3.0 * x2 + y2),
            2.8906114426405538 * xy * z,
            0.45704579946446572 * y * (1.0 - 5.0 * z2),
            0.3731763325901154 * z * (5.0 * z2 - 3.0),
            0.45704579946446572 * x * (1.0 - 5.0 * z2),
            1.4453057213202769 * z * (x2 - y2),
            0.59004358992664352 * x * (-x2 + 3.0 * y2),
        ],
        4: [
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
        5: [
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
        6: [
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
    };
    return model.degrees.flatMap((l) => per[l]);
}

// softplus with beta 10, linear above x > 2 (matches the shader)
const softplus10 = (x) => (x > 2 ? x : Math.log1p(Math.exp(10 * Math.min(x, 2))) * 0.1);

// unpacks a splat's fp16 parameter stream (the shader's loadParams)
function splatHalfParams(model, i) {
    const out = [];
    for (const tex of model.paramTexData)
        for (let w = 0; w < tex.components; w++) {
            const word = tex.data[i * tex.components + w];
            out.push(halfToFloat(word), halfToFloat(word >>> 16));
        }
    return out;
}

// sign-extended field of a splat's bit-packed SH level (the shader's shifts)
function packedField(data, base, fieldIdx, bits) {
    const bit = fieldIdx * bits;
    const wordIdx = bit >> 5, off = bit & 31;
    let v = data[base + wordIdx] >>> off;
    if (off + bits > 32) v |= data[base + wordIdx + 1] << (32 - off);
    return (v << (32 - bits)) >> (32 - bits);
}

// SH: residual_activation(sum of dequantized coefficient rgb * basis)
function shResidual(model, i, dir) {
    // no stored degrees: the reference returns zero before the residual activation
    if (!model.degrees.length) return [0, 0, 0];
    const basis = shDegreeValues(model, dir); // flat over the stored degrees, ascending
    const res = [0, 0, 0];
    let basisOffset = 0;
    model.degrees.forEach((degree, levelIdx) => {
        const { bits } = SH_PACK_LEVELS[degree - 1];
        const tex = model.paramTexData[levelIdx];
        const scale = model.shScales[degree - 1] / ((1 << (bits - 1)) - 1);
        const width = 2 * degree + 1;
        for (let c = 0; c < width; c++)
            for (let ch = 0; ch < 3; ch++)
                res[ch] += basis[basisOffset + c] * scale
                    * packedField(tex.data, i * tex.components, c * 3 + ch, bits);
        basisOffset += width;
    });
    return model.residualActivation === 1 ? res.map(Math.tanh)
        : model.residualActivation === 2 ? res.map(softplus10)
        : res;
}

// SV: softmax(-temperature * ||site - dir||)-weighted site color sum
function svResidual(model, i, dir) {
    const p = splatHalfParams(model, i);
    let maxLogit = -Infinity;
    const logits = [];
    for (let s = 0; s < model.nSites; s++) {
        const dist = Math.hypot(p[7 * s] - dir[0], p[7 * s + 1] - dir[1], p[7 * s + 2] - dir[2]);
        logits.push(-p[7 * s + 3] * dist);
        maxLogit = Math.max(maxLogit, logits[s]);
    }
    let weightSum = 0;
    const res = [0, 0, 0];
    for (let s = 0; s < model.nSites; s++) {
        const w = Math.exp(logits[s] - maxLogit);
        weightSum += w;
        for (let ch = 0; ch < 3; ch++) res[ch] += w * p[7 * s + 4 + ch];
    }
    return res.map((v) => v / weightSum);
}

// NASG: the NASGabor response without the Gabor term (12 values per lobe)
function nasgResidual(model, i, dir) {
    const p = splatHalfParams(model, i);
    const res = [0, 0, 0];
    for (let l = 0; l < model.nLobes; l++) {
        const b = 12 * l;
        const vz = dir[0] * p[b + 3] + dir[1] * p[b + 4] + dir[2] * p[b + 5];
        const vx = dir[0] * p[b] + dir[1] * p[b + 1] + dir[2] * p[b + 2];
        let pdf = 0;
        if (vz >= 1 - 1e-7) {
            pdf = 1;
        } else if (vz > -1 + 1e-7) {
            const K = 0.5 * (vz + 1);
            const Ke = 5e-6 + p[b + 7] * vx * vx / (1 - vz * vz);
            const E = Math.pow(K, Ke);
            pdf = Math.exp(p[b + 6] * (E * K - 1)) * E * p[b + 8];
        }
        for (let ch = 0; ch < 3; ch++) res[ch] += pdf * p[b + 9 + ch];
    }
    return res;
}

// NASGabor: the NASG response times the Gabor term (13 values per lobe)
function nasgaborResidual(model, i, dir) {
    const p = splatHalfParams(model, i);
    const res = [0, 0, 0];
    for (let l = 0; l < model.nLobes; l++) {
        const b = 13 * l;
        const vz = dir[0] * p[b + 3] + dir[1] * p[b + 4] + dir[2] * p[b + 5];
        const vx = dir[0] * p[b] + dir[1] * p[b + 1] + dir[2] * p[b + 2];
        let pdf = 0;
        if (vz >= 1 - 1e-7) {
            pdf = 1;
        } else if (vz > -1 + 1e-7) {
            const K = 0.5 * (vz + 1);
            const Ke = 5e-6 + p[b + 7] * vx * vx / (1 - vz * vz);
            const E = Math.pow(K, Ke);
            const G = 0.5 * (1 + Math.cos(p[b + 8] * vx));
            pdf = Math.exp(p[b + 6] * (E * K - 1)) * E * G * p[b + 9];
        }
        for (let ch = 0; ch < 3; ch++) res[ch] += pdf * p[b + 10 + ch];
    }
    return res;
}

// Neural: residual_activation(mlp([features, (SH_C0), sh_degrees(dir)]))
function neuralResidual(model, layers, i, dir) {
    // h0_static (baked) or the encoded features (full)
    const perSplat = splatHalfParams(model, i);
    // layer 0 starts from h0_static with only the direction inputs (baked) or
    // from zero with the full input vector (full)
    const inp = model.baked ? [] : [
        ...perSplat.slice(0, model.featureDim),
        ...(model.sh0Input ? [SH_C0] : []),
    ];
    inp.push(...shDegreeValues(model, dir));
    while (inp.length < model.l0In) inp.push(0);
    let h = layers[0].map((row, o) =>
        (model.baked ? perSplat[o] : 0) + row.reduce((s, w, j) => s + w * inp[j], 0));
    for (let l = 1; l < layers.length; l++) {
        h = h.map((v) => Math.max(v, 0));
        h = layers[l].map((row) => row.reduce((s, w, j) => s + w * h[j], 0));
    }
    return model.residualActivation === 2 ? h.map(softplus10)
        : model.residualActivation === 1 ? h.map(Math.tanh)
        : h;
}

export function referenceCaptureColor(model, layers, i, camPos, camFwd, { renderBase, renderResidual }) {
    const w0 = model.splatData[i * 4];
    const opacity8 = (w0 >>> 24) & 0xFF;
    // with the exp base the 0.5 shift is applied here, like the shader's COLOR_SHIFT
    const shift = model.baseExp && SHIFTED_COLOR_CODES.includes(model.colorActivation) ? 0.5 : 0;
    const act = (x) => {
        switch (model.colorActivation) {
            case 1: return softplus10(x + shift);
            case 2: return 1 / (1 + Math.exp(-4 * x));
            case 3: return -Math.expm1(-x);
            case 4: return Math.min(Math.max(x + shift, 0), 1);
            case 5: return x + shift;
            default: return Math.max(x + shift, 0);
        }
    };

    // pre-activation base d (range-coded rgb8)
    const d = [w0 & 0xFF, (w0 >>> 8) & 0xFF, (w0 >>> 16) & 0xFF]
        .map((c) => c * model.baseScale + model.baseOffset);
    const baseTerm = model.baseExp ? d.map((v) => Math.exp(3.0 * v)) : d;

    // the capture shader skips the residual for splats behind the camera plane
    const c = [model.centers[i * 3], model.centers[i * 3 + 1], model.centers[i * 3 + 2]];
    let rel = [c[0] - camPos[0], c[1] - camPos[1], c[2] - camPos[2]];
    const inFront = rel[0] * camFwd[0] + rel[1] * camFwd[1] + rel[2] * camFwd[2] > 0;

    let residual = [0, 0, 0];
    if (renderResidual && inFront) {
        const len = Math.hypot(...rel);
        rel = rel.map((v) => v / len);
        residual = model.appearance === "sh" ? shResidual(model, i, rel)
            : model.appearance === "sv" ? svResidual(model, i, rel)
            : model.appearance === "nasg" ? nasgResidual(model, i, rel)
            : model.appearance === "nasgabor" ? nasgaborResidual(model, i, rel)
            : neuralResidual(model, layers, i, rel);
    }
    const rgb = [0, 1, 2].map((k) => {
        const baseColor = act(baseTerm[k]);
        const full = renderResidual ? act(baseTerm[k] + residual[k]) : baseColor;
        const v = renderBase ? full : Math.abs(full - baseColor);
        return Math.round(Math.min(Math.max(v, 0), 1) * 255);
    });
    return [...rgb, opacity8];
}
