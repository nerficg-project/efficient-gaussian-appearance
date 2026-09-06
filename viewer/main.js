import {
    fetchFile, loadNgsplat, chooseModel, readUrlWithProgress,
    captureDefines, mlpInputDim, APPEARANCE_MODELS, APPEARANCE_NAMES,
    COLOR_ACTIVATION_NAMES, RESIDUAL_ACTIVATION_NAMES,
    buildMlpLayers, referenceCaptureColor, halfToFloat,
    applyAppearanceOverride, ppispHomography, unboundedRadiance,
} from "./utils.js";
import { createSplatSorter } from "./sorter.js";

// Three.js viewer for .ngsplat exports (Compact Neural Appearance Models for
// Efficient Gaussian Splatting). EWA splatting with CPU-sorted alpha blending;
// the per-splat view-dependent color color_activation(base + residual(dir))
// comes from one of five appearance models (SH, SV, NASG, NASGabor, Neural),
// evaluated in a capture pass that reruns only when the camera moves. The
// Benchmark GUI folder times the file's baked test-set viewpoints at a fixed
// viewport (720p with native intrinsics matches the reference benchmark).
// See README.md.

// Three.js and lil-gui are loaded on demand once a model is picked
let THREE, OrbitControls, GUI;
async function loadRenderingLibraries() {
    [THREE, { OrbitControls }, { default: GUI }] = await Promise.all([
        import("three"), import("three/addons/controls/OrbitControls.js"), import("lil-gui"),
    ]);
}

const NEAR = 0.2, FAR = 1000.0;

// benchmark settings (GUI folder "Benchmark"), snapshotted when a run starts
const BENCH_RESOLUTIONS = {
    "360p": [640, 360], "480p": [854, 480], "720p": [1280, 720],
    "1080p": [1920, 1080], "1440p": [2560, 1440], "2160p": [3840, 2160],
};
const benchSettings = {
    resolution: "720p",
    repeats: 10,         // timed renders per view
    timeSorting: false,  // fold a forced re-sort into every timed render
};

const state = {
    // "full" | "base" | "residual"; drives renderBase / renderResidual, the
    // reference renderer's debug switches
    shading: "full",
    sigma: 3.329,          // cutoff in std-devs; minAlpha = exp(-sigma^2/2). 3.329 -> 1/255
    renderBase: true,      // off = |full - base-only| visualization
    renderResidual: true,  // off = base color only (residual eval skipped)
    scaleModifier: 1.0,    // global gaussian scale multiplier
    ppisp: true,           // screen-space PPISP pass (files of PPISP-trained models)
    // blend target of the pass for files with radiance above 1: "fp16" is exact,
    // "rgba8" halves the blending bandwidth but clips mid-blend (visible in bright regions)
    blendTarget: "fp16",
    highRes: false,        // false: half native DPR on high-DPI; true: native capped at 2
    testView: -1,          // baked test-set viewpoint index (-1 = free camera)
    // capture stage, decided by the startup probe: fragment (default) or vertex
    // (fallback for drivers with broken fragment-stage integer fetches)
    mlpMode: "capture-fragment",
    // MLP weight backing: "uniform" (vec4 array, default when it fits the device
    // limit) or "texture" (always available)
    weightSource: "uniform",
    // MLP precision: "fp16" matches the reference implementation and is ~2x faster
    // on mobile; desktop drivers run mediump as fp32 anyway
    precision: "fp16",
};

// bottom-centered overlay stack for the benchmark panel and the error field
function overlayContainer() {
    let c = document.getElementById("overlays");
    if (!c) {
        c = document.createElement("div");
        c.id = "overlays";
        // full-width click-transparent strip: children center and shrink to fit
        c.style.cssText = "position:absolute;bottom:8px;left:0;right:0;" +
            "display:flex;flex-direction:column;align-items:center;gap:8px;" +
            "z-index:98;pointer-events:none;";
        document.body.append(c);
    }
    return c;
}
const OVERLAY_BASE_CSS = "border-radius:8px;padding:12px 14px;box-sizing:border-box;" +
    "font:12px ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:break-word;" +
    "max-width:calc(100vw - 16px);overflow:auto;pointer-events:auto;";

// visible error reporting (mobile browsers have no reachable console)
function showFatal(msg) {
    let el = document.getElementById("fatal");
    if (!el) {
        el = document.createElement("div");
        el.id = "fatal";
        el.style.cssText = OVERLAY_BASE_CSS +
            "background:#3a0d0d;color:#ffb4b4;border:1px solid #a33;order:2;max-height:45vh;";
        overlayContainer().append(el);
    }
    el.textContent += (el.textContent ? "\n" : "") + msg;
    console.error(msg);
}

async function main() {
    // --- gallery: pick a model (a manifest card, or ?model=URL) ---
    const gallery = document.getElementById("gallery");
    const sections = document.getElementById("gallery-sections");
    const loading = document.getElementById("loading");
    const progressFill = document.getElementById("progress-fill");
    const progressLabel = document.getElementById("progress-label");

    const manifest = await fetchFile("./models.json", "json").catch(() => ({ models: [] }));
    const params = new URLSearchParams(location.search);
    const urlModel = params.get("model");
    // ?appearance= forces a model: presets the benchmark section's selector, or
    // re-dresses a ?model= file (dummy residual parameters if the file lacks it)
    const forced = APPEARANCE_MODELS.includes(params.get("appearance")) ? params.get("appearance") : null;
    let model;
    let sceneName;
    let override = null;
    let librariesReady = null;
    while (true) {
        const source = urlModel
            ? { url: urlModel, override: forced }
            : await chooseModel(sections, manifest, manifest.defaultAppearance ?? "neural", forced ?? "neural");
        librariesReady ??= loadRenderingLibraries(); // in parallel with the model download
        const label = source.name ?? source.url;
        override = source.override;
        // gallery name, or the file's basename for direct ?model= links
        sceneName = source.name ?? source.url.split(/[?#]/)[0].split("/").pop().replace(/\.ngsplat$/, "");
        loading.hidden = false;
        progressFill.style.width = "0%";
        const onProgress = (frac) => {
            const pct = Math.round(frac * 100);
            progressFill.style.width = pct + "%";
            progressLabel.innerText = `${label} — ${pct}%`;
        };
        try {
            const buffer = await readUrlWithProgress(source.url, onProgress);
            model = loadNgsplat(buffer);
            break;
        } catch (err) {
            console.error(err);
            loading.hidden = true;
            if (urlModel) throw err; // no gallery to fall back to
        }
    }
    loading.hidden = true;
    gallery.classList.add("hidden");
    await librariesReady;

    const canvas = document.getElementById("canvas");
    canvas.addEventListener("webglcontextlost", (e) => {
        e.preventDefault();
        // no restore path: the host payloads are gone
        renderer.setAnimationLoop(null);
        showFatal("WebGL context lost — the GPU workload is likely too heavy for this device.");
    });
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, depth: false });
    renderer.autoClear = false;
    renderer.debug.onShaderError = (gl, program, vs, fs) => {
        const log = (s) => (gl.getShaderInfoLog(s) || "").trim();
        showFatal(`Shader compile/link failed:\nprogram: ${(gl.getProgramInfoLog(program) || "").trim()}` +
            `\nvertex: ${log(vs)}\nfragment: ${log(fs)}`);
    };

    if (override)
        model = applyAppearanceOverride(model, override, await fetchFile("./dummy_params.json", "json"));

    const { numSplats, textureWidth, textureHeight } = model;
    // ?residual=0 / ?base=0 preset the capture toggles
    if (params.get("residual") === "0") state.renderResidual = false;
    if (params.get("base") === "0") state.renderBase = false;
    state.shading = !state.renderResidual ? "base" : (!state.renderBase ? "residual" : "full");
    const isNeural = model.appearance === "neural";
    const modelName = APPEARANCE_NAMES[model.appearance]
        + (model.dummyParams ? " (dummy params)" : "");
    const bytesPerSplat = 16 + 4 * model.paramTexData.reduce((s, t) => s + t.components, 0);
    const sceneInfo = {
        appearance: modelName,
        splats: numSplats.toLocaleString(),
        fileSize: `${(model.fileBytes / 1024 / 1024).toFixed(1)} MB (${bytesPerSplat} B/splat)`,
        antialiasing: model.properAA ? "on" : "off",
        ppisp: model.ppisp ? `baked (${model.testCameras.length} views)` : "no",
        baseRange: `[${model.baseOffset.toFixed(2)}, `
            + `${(model.baseOffset + 255 * model.baseScale).toFixed(2)}]`,
        baseActivation: model.baseExp ? "exp" : "none",
        colorActivation: COLOR_ACTIVATION_NAMES[model.colorActivation],
        // for SV/NASG/NASGabor the residual activation is pre-applied at export
        residualActivation: RESIDUAL_ACTIVATION_NAMES[model.residualActivation]
            + (["sv", "nasg", "nasgabor"].includes(model.appearance) ? " (baked)" : ""),
        testViews: String(model.testCameras.length),
    };
    if (model.appearance === "sh") {
        sceneInfo.degrees = model.degrees.join(", ");
        sceneInfo.coefficients = `${3 * model.degrees.reduce((s, l) => s + 2 * l + 1, 0)} `
            + `(7/8/6-bit quantized)`;
    } else if (model.appearance === "sv") {
        sceneInfo.sites = String(model.nSites);
    } else if (model.appearance === "nasg" || model.appearance === "nasgabor") {
        sceneInfo.lobes = String(model.nLobes);
    } else {  // Neural
        const rawFeatures = model.nFrequencies
            ? model.featureDim / (2 * model.nFrequencies) : model.featureDim;
        const nIn = mlpInputDim(model);
        Object.assign(sceneInfo, {
            features: model.nFrequencies
                ? `${rawFeatures} × freq(${model.nFrequencies}) → ${model.featureDim}`
                : `${model.featureDim}`,
            degrees: [...(model.sh0Input ? [0] : []), ...model.degrees].join(", "),
            // baked: the view-independent inputs are pre-multiplied into a per-splat
            // h0_static cache and only the view-direction columns are evaluated
            layer0: model.baked
                ? `baked (${model.featureDim + (model.sh0Input ? 1 : 0)} of ${nIn} inputs pre-multiplied)`
                : "full input",
            inputs: String(nIn),
            hiddenLayers: String(model.nHiddenLayers),
            neurons: String(model.nNeurons),
            // unpadded weight count (the output layer is stored trimmed to 3 rows)
            weights: (
                nIn * model.nNeurons
                + (model.nHiddenLayers - 1) * model.nNeurons * model.nNeurons
                + 3 * model.nNeurons
            ).toLocaleString(),
            mlpActivation: "relu", // structurally fixed by the reference implementation
        });
    }
    const perf = { frameTime: "—", sortTime: "—" };
    let fpsCtrl = null;
    let sortCtrl = null;
    let benchCtrl = null;
    const BENCH_BUTTON_LABEL = "Benchmark";

    // --- camera + controls (world up from the header; initial pose = baked test view 0) ---
    const worldUp = new THREE.Vector3(...model.camUp).normalize();
    const camera = new THREE.PerspectiveCamera(60, 1, NEAR, FAR);
    camera.up.copy(worldUp);
    const controls = new OrbitControls(camera, canvas);

    // Pose the camera on a baked test view: position and forward from the c2w
    // (columns right/down/forward), vertical fov from the intrinsics, orbit target
    // on the view ray near the scene center. World up is kept, which levels any
    // roll exactly as the first controls.update() would.
    const _basisX = new THREE.Vector3(), _basisY = new THREE.Vector3(), _basisZ = new THREE.Vector3();
    const _rot = new THREE.Matrix4();
    function applyTestViewPose(cam) {
        const m = cam.c2w;
        camera.position.set(m[3], m[7], m[11]);
        camera.fov = THREE.MathUtils.radToDeg(2 * Math.atan(cam.height / (2 * cam.fy)));
        camera.updateProjectionMatrix();
        const forward = new THREE.Vector3(m[2], m[6], m[10]).normalize();
        const center = new THREE.Vector3(...model.camCenter);
        const along = Math.max(center.sub(camera.position).dot(forward), 0.25 * model.camDistance);
        controls.target.copy(camera.position).addScaledVector(forward, along);
        camera.up.copy(worldUp);
        camera.lookAt(controls.target);
    }

    if (model.testCameras.length) {
        const startView = Math.min(Math.max(Number(params.get("view")) || 0, 0), model.testCameras.length - 1);
        state.testView = startView;
        applyTestViewPose(model.testCameras[startView]);
    } else {
        // header hint fallback: orbit start on the +x axis of the scene center
        const center = new THREE.Vector3(...model.camCenter);
        camera.position.copy(center).add(new THREE.Vector3(model.camDistance, 0, 0));
        camera.lookAt(center);
        controls.target.copy(center);
    }

    // --- capture-stage probe reference (CPU ground truth), computed before the textures drop the host payloads ---
    const PROBE_SPLATS = 64;
    const nProbe = Math.min(PROBE_SPLATS, numSplats);
    const probeExpected = new Uint8Array(nProbe * 4);
    {
        const layers = isNeural ? buildMlpLayers(model) : null;
        const camPos = camera.position.toArray();
        const camFwd = camera.getWorldDirection(new THREE.Vector3()).toArray();
        const opts = { renderBase: state.renderBase, renderResidual: state.renderResidual };
        for (let i = 0; i < nProbe; i++)
            probeExpected.set(referenceCaptureColor(model, layers, i, camPos, camFwd, opts), i * 4);
    }

    // --- GPU textures (payloads uploaded exactly as stored in the file) ---
    // each host copy is dropped right after its upload (three would otherwise keep it for the
    // texture's lifetime; SV's 112 B/splat can exceed a mobile tab's memory budget)
    const makeIntTex = (data, components = 4) => {
        const tex = new THREE.DataTexture(
            data, textureWidth, textureHeight,
            components === 2 ? THREE.RGIntegerFormat : THREE.RGBAIntegerFormat,
            THREE.UnsignedIntType
        );
        tex.minFilter = THREE.NearestFilter;
        tex.magFilter = THREE.NearestFilter;
        tex.needsUpdate = true;
        renderer.initTexture(tex); // upload now
        tex.image.data = null;
        return tex;
    };
    const splatTexture = makeIntTex(model.splatData);
    model.splatData = null;
    // per-splat residual parameter textures (fp16 streams, or the packed SH
    // coefficient levels — the loader reports each texture's u32 components)
    const paramTextures = model.paramTexData.map((t) => {
        const tex = makeIntTex(t.data, t.components);
        t.data = null; // the entry stays: captureDefines reads paramTexData.length
        return tex;
    });
    // MLP weights (Neural only): raw fp16 bits in a 64-wide RGBA16F texture, texel
    // t at (t % 64, t / 64), mirrored by W() in eval_neural.glsl
    let weightsTexture = null;
    let weightsFloat = null;
    let uniformWeightsFit = false;
    let maxUniformVec = 0;
    if (isNeural) {
        const WEIGHTS_W = 64;
        const weightsH = Math.ceil(model.nWeightTexels / WEIGHTS_W);
        const weightsData = new Uint16Array(WEIGHTS_W * weightsH * 4);
        weightsData.set(model.weightData);
        weightsTexture = new THREE.DataTexture(
            weightsData, WEIGHTS_W, weightsH, THREE.RGBAFormat, THREE.HalfFloatType
        );
        weightsTexture.minFilter = THREE.NearestFilter;
        weightsTexture.magFilter = THREE.NearestFilter;
        weightsTexture.needsUpdate = true;
        // fp16 → fp32 weights for the uniform-array backing (flat vec4 stream)
        weightsFloat = new Float32Array(model.nWeightTexels * 4);
        for (let i = 0; i < weightsFloat.length; i++) weightsFloat[i] = halfToFloat(model.weightData[i]);
        // the uniform-array backing is offered only if it fits the smaller stage limit
        const gl = renderer.getContext();
        maxUniformVec = Math.min(
            gl.getParameter(gl.MAX_VERTEX_UNIFORM_VECTORS),
            gl.getParameter(gl.MAX_FRAGMENT_UNIFORM_VECTORS)
        );
        uniformWeightsFit = model.nWeightTexels + 8 <= maxUniformVec;
    }
    if (!uniformWeightsFit) state.weightSource = "texture"; // the uniform default needs the budget

    // --- capture pass: per-splat color eval -> color cache, one texel per splat
    //     (RGBA8, or half float for PPISP files whose radiance exceeds 1) ---
    const gl = renderer.getContext();
    const halfFloatTargets = Boolean(gl.getExtension("EXT_color_buffer_half_float") || gl.getExtension("EXT_color_buffer_float"));
    const needsFloat = unboundedRadiance(model);
    if (needsFloat && !halfFloatTargets)
        console.warn("ngsplat: no half-float render targets — the PPISP pass runs on RGBA8, bright colors clip");
    const useFloat = needsFloat && halfFloatTargets;
    if (!useFloat) state.blendTarget = "rgba8";
    const cacheOptions = { depthBuffer: false, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter };
    const colorRT = new THREE.WebGLRenderTarget(textureWidth, textureHeight,
        { ...cacheOptions, type: useFloat ? THREE.HalfFloatType : THREE.UnsignedByteType });
    const orthoCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    const fsGeo = new THREE.PlaneGeometry(2, 2);
    const fsVert = `precision highp float;
in vec3 position;
void main() { gl_Position = vec4(position.xy, 0.0, 1.0); }`;

    // the shared scaffolding plus the file's residual chunk are spliced into both
    // capture variants: fragment-stage fullscreen quad (default) and vertex-stage
    // points (fallback for broken fragment-stage integer texelFetch, Samsung Android)
    const captureFragWrap = await fetchFile("./shaders/capture.frag");
    const captureVertWrap = await fetchFile("./shaders/capture.vert");
    const capturePointFragSrc = await fetchFile("./shaders/capture_point.frag");
    const captureCommonGlsl = await fetchFile("./shaders/capture_common.glsl");
    const modelEvalGlsl = await fetchFile(`./shaders/eval_${model.appearance}.glsl`);
    const colorEvalGlsl = captureCommonGlsl.replace("// @inject residual_eval", modelEvalGlsl);
    const inject = (wrap, chunk) => wrap.replace("// @inject color_eval", chunk);
    const captureFragSrc = inject(captureFragWrap, colorEvalGlsl);
    const captureVertSrc = inject(captureVertWrap, colorEvalGlsl);

    const captureUniforms = {
        splatData: { value: splatTexture },
        cameraPos: { value: new THREE.Vector3() },
        cameraFwd: { value: new THREE.Vector3() },
        uBase: { value: new THREE.Vector2(model.baseScale, model.baseOffset) },
        uCacheSize: { value: new THREE.Vector2(textureWidth, textureHeight) },
    };
    for (let k = 0; k < paramTextures.length; k++)
        captureUniforms[`paramTex${k}`] = { value: paramTextures[k] };
    if (isNeural) {
        captureUniforms.mlpWeights = { value: weightsTexture }; // active only in texture mode
        captureUniforms.uWeights = { value: weightsFloat };     // active only in uniform mode
    } else if (model.appearance === "sh") {
        // per-level dequantization factor: level scale / signed quantization max
        captureUniforms.shScales = { value: new THREE.Vector3(
            model.shScales[0] / 63, model.shScales[1] / 127, model.shScales[2] / 31
        ) };
    }

    // fragment variant: fullscreen quad
    const captureScene = new THREE.Scene();
    const captureMesh = new THREE.Mesh(fsGeo, null);
    captureScene.add(captureMesh);
    // vertex variant, built lazily: one 1-pixel point per splat (ids come from gl_VertexID)
    let capturePoints = null;
    let capturePointScene = null;
    const ensureCapturePoints = () => {
        if (capturePoints) return;
        const pointsGeo = new THREE.BufferGeometry();
        pointsGeo.setAttribute("position", new THREE.BufferAttribute(new Uint8Array(numSplats), 1));
        capturePoints = new THREE.Points(pointsGeo, null);
        capturePoints.frustumCulled = false;
        capturePointScene = new THREE.Scene();
        capturePointScene.add(capturePoints);
    };

    let captureMat = null;
    function buildCaptureMaterial(clampedCache = false) {
        const wantFragment = state.mlpMode !== "capture-vertex";
        const defines = captureDefines(model, state, clampedCache);
        const m = new THREE.RawShaderMaterial({
            glslVersion: THREE.GLSL3,
            uniforms: captureUniforms,
            vertexShader: wantFragment ? fsVert : defines + captureVertSrc,
            fragmentShader: wantFragment ? defines + captureFragSrc : capturePointFragSrc,
            depthTest: false, depthWrite: false, blending: THREE.NoBlending,
        });
        if (wantFragment) captureMesh.material = m;
        else { ensureCapturePoints(); capturePoints.material = m; }
        if (captureMat) captureMat.dispose();
        captureMat = m;
    }

    // Pick the capture stage by comparing each variant's output with the CPU
    // reference: some drivers silently corrupt fragment-stage integer fetches
    // (Samsung Android), which only a readback check catches. Fragment wins
    // whenever it is correct (cheaper); the reference tells which variant is wrong.
    function chooseCaptureStage() {
        const TOL = 3; // RGBA8 steps: GPU fp32 vs CPU fp64 + output rounding
        // probe in fp32 with texture weights: the check targets broken fetch paths,
        // and fp16 drift (~1-3 steps on Apple GPUs) would exceed TOL
        const savedPrecision = state.precision;
        state.precision = "fp32";
        const savedWeights = state.weightSource;
        state.weightSource = "texture";
        captureUniforms.cameraPos.value.copy(camera.position);
        camera.getWorldDirection(captureUniforms.cameraFwd.value);

        // the probe reads bytes, so a half-float cache gets a temporary RGBA8 target
        const probeRT = useFloat
            ? new THREE.WebGLRenderTarget(textureWidth, textureHeight, { ...cacheOptions, type: THREE.UnsignedByteType })
            : colorRT;
        const errorFor = (mode) => {
            state.mlpMode = mode;
            buildCaptureMaterial();
            renderer.setRenderTarget(probeRT);
            renderer.setClearColor(0x000000, 0);
            renderer.clear(true, false, false);
            renderer.render(mode === "capture-vertex" ? capturePointScene : captureScene, orthoCamera);
            const out = new Uint8Array(textureWidth * 4);
            renderer.readRenderTargetPixels(probeRT, 0, 0, textureWidth, 1, out);
            let maxErr = 0;
            for (let i = 0; i < nProbe * 4; i++)
                maxErr = Math.max(maxErr, Math.abs(out[i] - probeExpected[i]));
            return maxErr;
        };
        const errFrag = errorFor("capture-fragment");
        let stage;
        if (errFrag <= TOL) {
            stage = "capture-fragment";
        } else {
            const errVert = errorFor("capture-vertex");
            console.warn(`ngsplat: fragment-stage capture mismatches the CPU reference (max err ${errFrag}, vertex ${errVert}) — ` +
                `using ${errVert <= errFrag ? "vertex" : "fragment"}-stage capture`);
            stage = errVert <= errFrag ? "capture-vertex" : "capture-fragment";
        }
        state.precision = savedPrecision;
        state.weightSource = savedWeights;
        if (probeRT !== colorRT) probeRT.dispose();
        return stage;
    }
    state.mlpMode = chooseCaptureStage();
    buildCaptureMaterial();
    // free the point scene if the fragment stage won
    if (state.mlpMode !== "capture-vertex" && capturePoints) {
        capturePoints.geometry.dispose();
        capturePoints = null;
        capturePointScene = null;
    }

    // --- render pass: EWA splats, CPU-sorted back-to-front alpha blending ---
    const splatDecodeGlsl = await fetchFile("./shaders/splat_decode.glsl");
    const renderVert = (await fetchFile("./shaders/render_ewa.vert"))
        .replace("// @inject splat_decode", splatDecodeGlsl);
    const renderFrag = await fetchFile("./shaders/render_ewa.frag");

    const uniforms = {
        splatData: { value: splatTexture },
        colorCache: { value: colorRT.texture },
        viewMatrix: { value: new THREE.Matrix4() },
        projectionMatrix: { value: new THREE.Matrix4() },
        halfWH: { value: new THREE.Vector2() },
        uMinAlpha: { value: Math.exp(-state.sigma * state.sigma / 2) },
        uScaleModifier: { value: state.scaleModifier },
    };

    const geometry = new THREE.PlaneGeometry(2, 2);
    // CPU depth sorter: instance i draws splat splatIndex[i]; the `count` leading
    // entries are in front of the camera, the rest are skipped via the draw count
    const sortOrder = new Float32Array(numSplats);
    for (let i = 0; i < numSplats; i++) sortOrder[i] = i;
    const sortAttr = new THREE.InstancedBufferAttribute(sortOrder, 1);
    sortAttr.setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute("splatIndex", sortAttr);
    let onSortApplied = null; // one-shot hook used by the benchmark to await a sort
    const sorter = createSplatSorter(model.centers, numSplats, (order, count, sortTime) => {
        sortAttr.array.set(order);
        sortAttr.needsUpdate = true;
        mesh.count = count;
        perf.sortTime = `${sortTime.toFixed(1)} ms (${count.toLocaleString()} drawn)`;
        if (sortCtrl) sortCtrl.updateDisplay();
        if (onSortApplied) { const fn = onSortApplied; onSortApplied = null; fn(); }
    });
    const _sortBackward = new THREE.Vector3();

    const material = new THREE.RawShaderMaterial({
        glslVersion: THREE.GLSL3,
        vertexShader: (model.properAA ? "#define PROPER_AA\n" : "") + renderVert,
        fragmentShader: renderFrag,
        uniforms,
        depthTest: false, depthWrite: false,
        blending: THREE.NormalBlending,
        transparent: true,
    });
    const scene = new THREE.Scene();
    const mesh = new THREE.InstancedMesh(geometry, material, numSplats);
    mesh.frustumCulled = false;
    scene.add(mesh);

    // --- PPISP pass (files of PPISP-trained models): the splats render into an
    //     offscreen target and a fullscreen pass applies the baked PPISP with a
    //     pinned test view's exposure and color correction, else the defaults ---
    let ppispPass = null;
    if (model.ppisp) {
        const vec3s = (arr, off) => Array.from({ length: 3 }, (_, i) => new THREE.Vector3(...arr.subarray(6 * i + off, 6 * i + off + 3)));
        const targetOptions = () => ({ type: state.blendTarget === "fp16" ? THREE.HalfFloatType : THREE.UnsignedByteType });
        let sceneRT = new THREE.WebGLRenderTarget(1, 1, { ...cacheOptions, ...targetOptions() });
        let sceneSize = [1, 1];
        const ppispUniforms = {
            tScene: { value: sceneRT.texture },
            uExposure: { value: 1.0 },
            uColorH: { value: new THREE.Matrix3() },
            uCrfShape: { value: vec3s(model.ppisp.crf, 0) },
            uCrfCurve: { value: vec3s(model.ppisp.crf, 3) },
        };
        const ppispMaterial = new THREE.RawShaderMaterial({
            glslVersion: THREE.GLSL3, uniforms: ppispUniforms,
            vertexShader: fsVert, fragmentShader: await fetchFile("./shaders/ppisp.frag"),
            depthTest: false, depthWrite: false, blending: THREE.NoBlending,
        });
        const ppispScene = new THREE.Scene();
        ppispScene.add(new THREE.Mesh(fsGeo, ppispMaterial));
        ppispPass = {
            active: () => state.ppisp && state.renderBase, // no PPISP for the residual-only view, as in the reference
            resize: (w, h) => { sceneSize = [w, h]; sceneRT.setSize(w, h); },
            retarget: () => {
                sceneRT.dispose();
                sceneRT = new THREE.WebGLRenderTarget(...sceneSize, { ...cacheOptions, ...targetOptions() });
                ppispUniforms.tScene.value = sceneRT.texture;
            },
            update: () => {
                const { views, defaults } = model.ppisp;
                const params = state.testView >= 0 ? views.subarray(9 * state.testView, 9 * state.testView + 9) : defaults;
                ppispUniforms.uExposure.value = 2 ** params[0];
                ppispUniforms.uColorH.value.set(...ppispHomography(params.subarray(1)));
            },
            render: () => {
                renderer.setRenderTarget(sceneRT);
                renderer.setClearColor(0x000000, 1);
                renderer.clear(true, false, false);
                renderer.render(scene, camera);
                renderer.setRenderTarget(null);
                renderer.render(ppispScene, orthoCamera);
            },
        };
    }

    let cameraChanged = true;
    // auto: half native DPR on high-DPI (>= 2); High res forces native, capped at 2
    const effectiveDpr = () => {
        const dpr = window.devicePixelRatio || 1;
        return state.highRes ? Math.min(dpr, 2) : (dpr >= 2 ? dpr * 0.5 : dpr);
    };
    const resize = () => {
        const w = window.innerWidth, h = window.innerHeight;
        renderer.setPixelRatio(effectiveDpr());
        renderer.setSize(w, h);
        if (ppispPass) ppispPass.resize(Math.round(w * effectiveDpr()), Math.round(h * effectiveDpr()));
        uniforms.halfWH.value.set(w / 2, h / 2); // CSS px (DPR-invariant projection math)
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        cameraChanged = true;
    };
    window.addEventListener("resize", resize);
    controls.addEventListener("change", () => { cameraChanged = true; });
    let testViewCtrl = null;
    controls.addEventListener("start", () => {
        // manual navigation leaves the pinned test view
        if (state.testView !== -1) {
            state.testView = -1;
            if (testViewCtrl) testViewCtrl.updateDisplay();
        }
    });

    setupGui();
    resize();

    // per-camera uniforms; the benchmark substitutes its own projection matrix
    const updateCameraUniforms = () => {
        camera.updateMatrixWorld(true);
        uniforms.viewMatrix.value.copy(camera.matrixWorldInverse);
        uniforms.projectionMatrix.value.copy(camera.projectionMatrix);
        captureUniforms.cameraPos.value.copy(camera.position);
        camera.getWorldDirection(captureUniforms.cameraFwd.value);
    };

    const runCapture = (target = colorRT) => {
        renderer.setRenderTarget(target);
        if (state.mlpMode === "capture-vertex") {
            renderer.setClearColor(0x000000, 0); // padded tail texels stay defined
            renderer.clear(true, false, false);
            renderer.render(capturePointScene, orthoCamera);
        } else {
            renderer.render(captureScene, orthoCamera); // quad covers every texel
        }
    };

    const clock = new THREE.Clock();
    let avgMs = 0;

    const animationLoop = () => {
        const delta = clock.getDelta();
        controls.update();

        if (cameraChanged) {
            updateCameraUniforms();
            // camera backward axis (world-matrix Z column), as PlayCanvas passes
            _sortBackward.setFromMatrixColumn(camera.matrixWorld, 2).normalize();
            sorter.setCamera(camera.position, _sortBackward);
            runCapture(); // captured colors depend only on the camera position
            if (ppispPass) ppispPass.update();
            cameraChanged = false;
        }

        if (ppispPass?.active()) {
            ppispPass.render();
        } else {
            renderer.setRenderTarget(null);
            renderer.setClearColor(0x000000, 1);
            renderer.clear(true, false, false);
            renderer.render(scene, camera);
        }

        const ms = delta * 1000;
        avgMs = avgMs * 0.9 + ms * 0.1;
        perf.frameTime = avgMs.toFixed(2) + " ms";
        if (fpsCtrl) fpsCtrl.updateDisplay();
    };
    renderer.setAnimationLoop(animationLoop);

    // ---- benchmark: fixed-viewport renders of the baked test-set viewpoints ----
    // Per view: exact pose and native intrinsics at the configured viewport (720p
    // = the reference benchmark). An untimed warmup pass over all views, then per
    // view a re-sort, one warmup render, and `repeats` timed renders, each with
    // the capture pass and a 1x1 readback (GPU completion, not vsync). Sorting is
    // untimed by default (steady-state frame cost); "Time sorting" folds a forced
    // re-sort into every timed render. Without baked cameras a seeded random
    // orbit is used.
    let benchmarkRunning = false;

    // the button doubles as progress bar; updated only between timed sections
    function updateBenchButton(label, frac) {
        if (!benchCtrl) return;
        benchCtrl.name(label);
        const btn = benchCtrl.domElement.querySelector("button");
        if (btn) {
            const pct = frac == null ? null : Math.round(frac * 100);
            btn.style.background = pct == null ? "" :
                `linear-gradient(to right, rgba(78, 161, 255, 0.45) ${pct}%, transparent ${pct}%)`;
        }
    }

    // off-axis projection from the native intrinsics (u = cx + fx * X/Z, v from the top)
    function benchProjection(cam, W, H) {
        const n = NEAR, f = FAR;
        return new THREE.Matrix4().set(
            2 * cam.fx / W, 0, 1 - 2 * cam.cx / W, 0,
            0, 2 * cam.fy / H, 2 * cam.cy / H - 1, 0,
            0, 0, -(f + n) / (f - n), -2 * f * n / (f - n),
            0, 0, -1, 0
        );
    }

    function randomOrbitPoses(count) {
        const mulberry32 = (a) => () => {
            a |= 0; a = (a + 0x6D2B79F5) | 0;
            let t = Math.imul(a ^ (a >>> 15), 1 | a);
            t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
        const rand = mulberry32(0x5EED);
        const target = controls.target.clone();
        const up = camera.up.clone().normalize();
        const baseDist = camera.position.distanceTo(target);
        const e1 = new THREE.Vector3(1, 0, 0);
        if (Math.abs(up.dot(e1)) > 0.9) e1.set(0, 0, 1);
        e1.cross(up).normalize();
        const e2 = new THREE.Vector3().crossVectors(up, e1).normalize();
        return Array.from({ length: count }, () => {
            const azim = rand() * Math.PI * 2;
            const elev = (rand() * 0.9 - 0.2) * (Math.PI / 4); // -9deg .. +31deg
            const r = baseDist * (0.6 + 0.9 * rand());
            return {
                position: new THREE.Vector3()
                    .addScaledVector(e1, Math.cos(azim) * Math.cos(elev) * r)
                    .addScaledVector(e2, Math.sin(azim) * Math.cos(elev) * r)
                    .addScaledVector(up, Math.sin(elev) * r)
                    .add(target),
                target,
            };
        });
    }

    async function runBenchmark() {
        if (benchmarkRunning) return;
        benchmarkRunning = true;
        renderer.setAnimationLoop(null); // pause the app loop
        benchCtrl?.disable();
        updateBenchButton("preparing…", 0);

        // settings snapshot: GUI changes during the run have no effect
        const [benchW, benchH] = BENCH_RESOLUTIONS[benchSettings.resolution];
        const repeats = Math.max(1, Math.round(benchSettings.repeats));
        const timeSorting = benchSettings.timeSorting;

        const useTestCams = model.testCameras.length > 0;
        const views = useTestCams ? model.testCameras : randomOrbitPoses(16);
        const fallbackProjection = new THREE.Matrix4().copy(camera.projectionMatrix);
        if (!useTestCams) {
            // no baked cameras: benchmark viewport aspect with the current fov
            const saved = camera.aspect;
            camera.aspect = benchW / benchH;
            camera.updateProjectionMatrix();
            fallbackProjection.copy(camera.projectionMatrix);
            camera.aspect = saved;
        }
        const applyPose = (i) => {
            if (useTestCams) {
                const cam = views[i];
                const m = cam.c2w;
                _basisX.set(m[0], m[4], m[8]);
                _basisY.set(-m[1], -m[5], -m[9]);
                _basisZ.set(-m[2], -m[6], -m[10]);
                camera.position.set(m[3], m[7], m[11]);
                camera.quaternion.setFromRotationMatrix(_rot.makeBasis(_basisX, _basisY, _basisZ));
                camera.projectionMatrix.copy(benchProjection(cam, benchW, benchH));
            } else {
                camera.position.copy(views[i].position);
                camera.lookAt(views[i].target);
                camera.projectionMatrix.copy(fallbackProjection);
            }
            camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
        };

        // settings snapshot shown with the results
        const kv = (k, v) => `${k.padEnd(9)} ${v}`;
        const captureStage = state.mlpMode === "capture-vertex" ? "vertex shader" : "fragment shader";
        const shadingName = { full: "full", base: "base only", residual: "residual only" }[state.shading];
        const settings = [
            kv("splats", numSplats.toLocaleString()),
            kv("views", useTestCams
                ? `${views.length} test-set viewpoints`
                : `${views.length} random orbits (no baked test cameras)`),
            kv("res", `${benchW}×${benchH} fixed (native intrinsics)`),
            kv("renders", `${repeats} timed + 1 warmup / view`),
            kv("sorting", timeSorting
                ? "timed (forced re-sort per render)" : "untimed (async, pre-applied)"),
            kv("stage", captureStage),
            ...(isNeural ? [
                kv("weights", state.weightSource === "uniform" ? "uniform array" : "texture"),
                kv("precision", state.precision),
            ] : []),
            kv("shading", shadingName),
            ...(model.ppisp ? [kv("ppisp", "excluded (no pass, RGBA8 cache and target)")] : []),
            kv("σ cutoff", `${Math.round(state.sigma * 1000) / 1000}σ`),
            kv("scale", state.scaleModifier.toFixed(2)),
        ].join("\n");

        const savedPos = camera.position.clone();
        const savedQuat = camera.quaternion.clone();
        const savedHalfWH = uniforms.halfWH.value.clone();
        uniforms.halfWH.value.set(benchW / 2, benchH / 2);
        // PPISP is excluded: no pass, RGBA8 cache and target like files without it
        const rt = new THREE.WebGLRenderTarget(benchW, benchH, {
            depthBuffer: false, type: THREE.UnsignedByteType,
        });
        const px = new Uint8Array(4);
        const benchCache = useFloat
            ? new THREE.WebGLRenderTarget(textureWidth, textureHeight, { ...cacheOptions, type: THREE.UnsignedByteType })
            : colorRT;
        uniforms.colorCache.value = benchCache.texture;
        if (useFloat) buildCaptureMaterial(true); // the plain files' capture shader

        // progress units: warmup pass + measured pass
        const V = views.length;
        const totalUnits = 2 * V;

        // apply view v's pose and await its sort order (2 s safety timeout);
        // `jitter` exceeds the worker's 0.001 dedup epsilon so a repeated sort of
        // the same view is not skipped
        const sortView = async (v, jitter = 0) => {
            applyPose(v);
            if (jitter) {
                camera.position.x += jitter;
                camera.position.y += jitter;
                camera.position.z += jitter;
            }
            updateCameraUniforms();
            _sortBackward.setFromMatrixColumn(camera.matrixWorld, 2).normalize();
            await new Promise((resolve) => {
                const timer = setTimeout(resolve, 2000);
                onSortApplied = () => { clearTimeout(timer); resolve(); };
                sorter.setCamera(camera.position, _sortBackward);
            });
            onSortApplied = null;
        };

        const frame = () => {
            runCapture(benchCache);
            renderer.setRenderTarget(rt);
            renderer.setClearColor(0x000000, 1);
            renderer.clear(true, false, false);
            renderer.render(scene, camera);
            renderer.readRenderTargetPixels(rt, 0, 0, 1, 1, px); // GPU sync
        };
        // setTimeout, not rAF: rAF is throttled to zero in hidden/headless tabs
        const yieldLoop = () => new Promise((resolve) => setTimeout(resolve, 0));

        // untimed warmup pass (shader compilation, allocator and driver warm-up),
        // then per view: sort, one untimed render (absorbs the sort-attribute
        // upload), `repeats` timed renders, or with "Time sorting" a forced
        // re-sort + render per sample. Orders are sorted on demand, never cached.
        const viewMeans = [];
        try {
            for (let v = 0; v < V; v++) {
                updateBenchButton(`${modelName} warmup ${v + 1}/${V}`, v / totalUnits);
                await sortView(v);
                frame();
                await yieldLoop();
            }
            for (let v = 0; v < V; v++) {
                updateBenchButton(`${modelName} ${v + 1}/${V}`, (V + v) / totalUnits);
                await sortView(v);
                frame(); // warmup render: absorbs the fresh sort-attribute upload
                if (timeSorting) {
                    let total = 0;
                    for (let r = 0; r < repeats; r++) {
                        const t0 = performance.now();
                        await sortView(v, (r % 2 ? 1 : -1) * 0.002);
                        frame();
                        total += performance.now() - t0;
                    }
                    viewMeans.push(total / repeats);
                } else {
                    const t0 = performance.now();
                    for (let r = 0; r < repeats; r++) frame();
                    viewMeans.push((performance.now() - t0) / repeats);
                }
                await yieldLoop();
            }
        } finally {
            rt.dispose();
            uniforms.colorCache.value = colorRT.texture;
            if (benchCache !== colorRT) { benchCache.dispose(); buildCaptureMaterial(); }
            onSortApplied = null;
            updateBenchButton(BENCH_BUTTON_LABEL, null);
            benchCtrl?.enable();
            uniforms.halfWH.value.copy(savedHalfWH);
            camera.position.copy(savedPos);
            camera.quaternion.copy(savedQuat);
            camera.updateProjectionMatrix(); // drop the benchmark projection
            cameraChanged = true;
            renderer.setAnimationLoop(animationLoop);
            benchmarkRunning = false;
        }

        // stats table: label row + the model's row
        const sorted = [...viewMeans].sort((a, b) => a - b);
        const avg = sorted.reduce((s, v) => s + v, 0) / sorted.length;
        const cells = [
            ["avg", `${avg.toFixed(2)} ms`],
            ["med", `${sorted[sorted.length >> 1].toFixed(2)} ms`],
            ["min", `${sorted[0].toFixed(2)} ms`],
            ["max", `${sorted[sorted.length - 1].toFixed(2)} ms`],
            ["fps", (1000 / avg).toFixed(1)],
        ];
        const colW = cells.map(([k, v]) => Math.max(k.length, v.length));
        const header = "".padEnd(modelName.length) + "   " +
            cells.map(([k], i) => k.padEnd(colW[i])).join("   ").trimEnd();
        const row = modelName + "   " +
            cells.map(([, v], i) => v.padEnd(colW[i])).join("   ").trimEnd();
        const statsText = [header, row].join("\n");

        // results panel (tap to dismiss), above any error field in the overlay stack
        document.getElementById("bench-results")?.remove();
        const el = document.createElement("div");
        el.id = "bench-results";
        el.style.cssText = OVERLAY_BASE_CSS +
            "background:#101318;color:#cfd2da;border:1px solid rgba(255,255,255,0.2);" +
            "cursor:pointer;order:1;";
        el.textContent =
            `Benchmark — ${views.length} views × ${repeats} renders, re-sorted per view, ` +
            `warmup pass\n\n` +
            `${settings}\n\n${statsText}\n\n(tap to dismiss)`;
        el.addEventListener("click", () => el.remove());
        overlayContainer().append(el);
    }

    function setupGui() {
        const guiModel = APPEARANCE_NAMES[model.appearance] + (model.dummyParams ? ", dummy params" : "");
        const gui = new GUI({ title: `${sceneName} (${guiModel})` });
        gui.domElement.style.position = "absolute";
        gui.domElement.style.top = "8px";
        gui.domElement.style.right = "8px";
        gui.domElement.style.left = "auto";

        // lil-gui has no separator widget: a styled div in the children container
        const addSeparator = () => {
            const sep = document.createElement("div");
            sep.style.cssText = "border-top:1px solid rgba(255,255,255,0.18);margin:6px 8px;";
            gui.$children.appendChild(sep);
        };

        gui.add({ gallery: () => location.assign(location.pathname) }, "gallery").name("← Gallery");

        addSeparator();
        fpsCtrl = gui.add(perf, "frameTime").name("Frame time").disable();
        sortCtrl = gui.add(perf, "sortTime").name("Sort time").disable();
        addSeparator();

        // base-color / residual debug switches of the reference renderer
        gui.add(state, "shading", {
            "Full": "full",
            "Base color only": "base",
            "Residual only": "residual",
        }).name("Shading")
            .onChange(() => {
                state.renderResidual = state.shading !== "base";
                state.renderBase = state.shading !== "residual";
                buildCaptureMaterial();
                cameraChanged = true;
            });
        if (model.ppisp) {
            gui.add(state, "ppisp").name("PPISP");
            if (useFloat)
                gui.add(state, "blendTarget", { "RGBA16F (exact)": "fp16", "RGBA8 (fast)": "rgba8" }).name("Blend target")
                    .onChange(() => { ppispPass.retarget(); cameraChanged = true; });
        }

        if (model.testCameras.length) {
            testViewCtrl = gui.add(state, "testView", -1, model.testCameras.length - 1, 1)
                .name("Test view")
                .onChange(() => {
                    if (state.testView >= 0) applyTestViewPose(model.testCameras[state.testView]);
                    cameraChanged = true; // -1 also switches PPISP back to the defaults
                });
        }

        // each option shows the N-sigma radius and the alpha floor it implies
        const cutoffModes = [
            { sigma: 1 }, { sigma: 2 }, { sigma: 3 },
            { sigma: 3.329, label: "3.329σ  (α ≥ 1/255)" },
        ];
        const sigmaOptions = {};
        for (const m of cutoffModes) {
            const alpha = Math.exp(-m.sigma * m.sigma / 2);
            const label = m.label ?? `${Math.round(m.sigma * 100) / 100}σ  (α ≥ ${alpha.toFixed(2)})`;
            sigmaOptions[label] = m.sigma;
        }
        gui.add(state, "sigma", sigmaOptions).name("Sigma cutoff")
            .onChange(() => {
                uniforms.uMinAlpha.value = Math.exp(-state.sigma * state.sigma / 2);
            });

        gui.add(state, "scaleModifier", 0.05, 2.0, 0.05).name("Scale modifier")
            .onChange(() => { uniforms.uScaleModifier.value = state.scaleModifier; });

        // only meaningful on high-DPI displays
        if ((window.devicePixelRatio || 1) >= 2)
            gui.add(state, "highRes").name("High res").onChange(resize);

        if (isNeural) {
            // uniform array (default) vs texture; pinned to texture when the
            // uniform budget is exceeded
            const weightCtrl = gui.add(state, "weightSource", {
                "Uniform array": "uniform",
                "Texture": "texture",
            }).name("MLP weights")
                .onChange(() => { buildCaptureMaterial(); cameraChanged = true; });
            if (!uniformWeightsFit) {
                state.weightSource = "texture";
                weightCtrl.setValue("texture").disable()
                    .name(`MLP weights (uniform > ${maxUniformVec} vec4 limit)`);
            }

            gui.add(state, "precision", { "fp16": "fp16", "fp32": "fp32" }).name("MLP precision")
                .onChange(() => { buildCaptureMaterial(); cameraChanged = true; });
        }

        const modelFolder = gui.addFolder("Model info");
        modelFolder.add(sceneInfo, "appearance").name("Appearance").disable();
        modelFolder.add(sceneInfo, "splats").name("Splats").disable();
        modelFolder.add(sceneInfo, "fileSize").name("File size").disable();
        modelFolder.add(sceneInfo, "antialiasing").name("Mip filter").disable();
        if (needsFloat && !halfFloatTargets) sceneInfo.ppisp += ", RGBA8 fallback";
        modelFolder.add(sceneInfo, "ppisp").name("PPISP").disable();
        modelFolder.add(sceneInfo, "baseRange").name("Base range").disable();
        if (model.appearance === "sh") {
            modelFolder.add(sceneInfo, "degrees").name("SH degrees").disable();
            modelFolder.add(sceneInfo, "coefficients").name("Coefficients").disable();
        } else if (model.appearance === "sv") {
            modelFolder.add(sceneInfo, "sites").name("Voronoi sites").disable();
        } else if (model.appearance === "nasg" || model.appearance === "nasgabor") {
            modelFolder.add(sceneInfo, "lobes")
                .name(model.appearance === "nasg" ? "Lobes" : "Gabor lobes").disable();
        } else {  // Neural
            modelFolder.add(sceneInfo, "features").name("Features").disable();
            modelFolder.add(sceneInfo, "degrees").name("SH degrees").disable();
        }
        modelFolder.add(sceneInfo, "testViews").name("Test views").disable();
        if (isNeural) {
            modelFolder.add(sceneInfo, "layer0").name("Layer 0").disable();
            const mlpFolder = modelFolder.addFolder("MLP");
            mlpFolder.add(sceneInfo, "weights").name("Weights").disable();
            mlpFolder.add(sceneInfo, "inputs").name("Inputs").disable();
            mlpFolder.add(sceneInfo, "hiddenLayers").name("Hidden layers").disable();
            mlpFolder.add(sceneInfo, "neurons").name("Neurons / layer").disable();
            mlpFolder.add(sceneInfo, "mlpActivation").name("Activation").disable();
        }
        modelFolder.add(sceneInfo, "baseActivation").name("Base activation").disable();
        modelFolder.add(sceneInfo, "colorActivation").name("Color activation").disable();
        modelFolder.add(sceneInfo, "residualActivation").name("Residual activation").disable();
        modelFolder.close();

        const benchFolder = gui.addFolder("Benchmark");
        benchFolder.add(benchSettings, "resolution", Object.keys(BENCH_RESOLUTIONS))
            .name("Resolution");
        benchFolder.add(benchSettings, "repeats", 1, 100, 1).name("Timed renders / view");
        // forced re-sort in every timed render (end-to-end novel-view cost)
        benchFolder.add(benchSettings, "timeSorting").name("Time sorting");
        benchCtrl = benchFolder.add({ benchmark: () => runBenchmark() }, "benchmark")
            .name(BENCH_BUTTON_LABEL);
        benchFolder.close();
    }
}

main().catch((err) => showFatal(String(err?.stack ?? err)));
