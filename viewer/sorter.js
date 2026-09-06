// CPU splat depth sorter, a close port of the PlayCanvas engine's gsplat sorter
// (gsplat-sort-worker.js + gsplat-sorter.js). A web worker owns the splat centers
// (transferred once) and counting-sorts splat indices by view depth per camera update:
// - Keys follow the camera's BACKWARD axis, so ascending key order is back-to-front
//   draw order and behind-camera splats land at the end, where `count` cuts them off.
// - 2^clamp(round(log2(n/4)), 10, 20) buckets, distributed by a 32-bin density
//   histogram along the view axis (from per-256-splat chunk bounds). Essential for
//   MCMC scenes whose background stretches the depth range far beyond the core.
// - The camera-changed epsilon check (0.001) lives in the worker; the order buffer is
//   double-buffered and transferred back and forth, so no per-sort allocation.
// Deviations from PlayCanvas: chunk bounds always come from the centers, no
// compressed-format mapping, the order is a Float32Array (exact below 2^24) copied
// straight into the instanced attribute, and bin/key indexing is clamped.

function sortWorker() {
    let order = null; // Float32Array available for the next sort (double-buffered with main)
    let centers = null;
    let chunks = null; // per-256-splat bounding spheres [x, y, z, r], for the density histogram
    let cameraPosition = null;
    let cameraDirection = null;

    let forceUpdate = false;

    const lastCameraPosition = { x: 0, y: 0, z: 0 };
    const lastCameraDirection = { x: 0, y: 0, z: 0 };

    const boundMin = { x: 0, y: 0, z: 0 };
    const boundMax = { x: 0, y: 0, z: 0 };

    let distances = null;
    let countBuffer = null;

    // could be increased, but this seems a good compromise between stability and performance
    const numBins = 32;
    const binCount = new Uint32Array(numBins);
    const binBase = new Float64Array(numBins);
    const binDivider = new Float64Array(numBins);

    const binarySearch = (m, n, compare_fn) => {
        while (m <= n) {
            const k = (n + m) >> 1;
            const cmp = compare_fn(k);
            if (cmp > 0) {
                m = k + 1;
            } else if (cmp < 0) {
                n = k - 1;
            } else {
                return k;
            }
        }
        return ~m;
    };

    const update = () => {
        if (!order || !centers || centers.length === 0 || !cameraPosition || !cameraDirection) return;

        const sortStartTime = performance.now();

        const px = cameraPosition.x;
        const py = cameraPosition.y;
        const pz = cameraPosition.z;
        const dx = cameraDirection.x;
        const dy = cameraDirection.y;
        const dz = cameraDirection.z;

        const epsilon = 0.001;

        if (!forceUpdate &&
            Math.abs(px - lastCameraPosition.x) < epsilon &&
            Math.abs(py - lastCameraPosition.y) < epsilon &&
            Math.abs(pz - lastCameraPosition.z) < epsilon &&
            Math.abs(dx - lastCameraDirection.x) < epsilon &&
            Math.abs(dy - lastCameraDirection.y) < epsilon &&
            Math.abs(dz - lastCameraDirection.z) < epsilon) {
            return;
        }

        forceUpdate = false;

        lastCameraPosition.x = px;
        lastCameraPosition.y = py;
        lastCameraPosition.z = pz;
        lastCameraDirection.x = dx;
        lastCameraDirection.y = dy;
        lastCameraDirection.z = dz;

        // calc min/max distance using bound
        let minDist = Infinity, maxDist = -Infinity;
        for (let i = 0; i < 8; ++i) {
            const d = (i & 1 ? boundMin.x : boundMax.x) * dx
                    + (i & 2 ? boundMin.y : boundMax.y) * dy
                    + (i & 4 ? boundMin.z : boundMax.z) * dz;
            if (d < minDist) minDist = d;
            if (d > maxDist) maxDist = d;
        }

        const numVertices = centers.length / 3;

        // calculate number of bits needed to store sorting result
        const compareBits = Math.max(10, Math.min(20, Math.round(Math.log2(numVertices / 4))));
        const bucketCount = 2 ** compareBits + 1;

        if (distances?.length !== numVertices) distances = new Uint32Array(numVertices);
        if (countBuffer?.length !== bucketCount) countBuffer = new Uint32Array(bucketCount);
        else countBuffer.fill(0);

        const range = maxDist - minDist;

        if (range < 1e-6) {
            // all points are at the same distance
            for (let i = 0; i < numVertices; ++i) {
                distances[i] = 0;
                countBuffer[0]++;
            }
        } else {
            // use chunks to calculate rough histogram of splats per distance
            binCount.fill(0);
            const numChunks = chunks.length / 4;
            for (let i = 0; i < numChunks; ++i) {
                const d = chunks[i * 4] * dx + chunks[i * 4 + 1] * dy + chunks[i * 4 + 2] * dz - minDist;
                const r = chunks[i * 4 + 3];
                const binMin = Math.max(0, Math.floor((d - r) * numBins / range));
                const binMax = Math.min(numBins, Math.ceil((d + r) * numBins / range));
                for (let j = binMin; j < binMax; ++j) binCount[j]++;
            }

            // count total number of histogram bin entries
            let binTotal = 0;
            for (let i = 0; i < numBins; ++i) binTotal += binCount[i];

            // calculate per-bin base and divider
            for (let i = 0; i < numBins; ++i) {
                binDivider[i] = (binCount[i] / binTotal * (bucketCount - 1)) >>> 0;
            }
            for (let i = 0; i < numBins; ++i) {
                binBase[i] = i === 0 ? 0 : binBase[i - 1] + binDivider[i - 1];
            }

            // generate per vertex distance key using histogram to distribute bits
            const maxKey = bucketCount - 1;
            const invBinRange = numBins / range;
            for (let i = 0, j = 0; i < numVertices; ++i, j += 3) {
                const d = (centers[j] * dx + centers[j + 1] * dy + centers[j + 2] * dz - minDist) * invBinRange;
                let bin = d | 0;
                if (bin > numBins - 1) bin = numBins - 1;
                let sortKey = (binBase[bin] + binDivider[bin] * (d - bin)) | 0;
                if (sortKey > maxKey) sortKey = maxKey;
                distances[i] = sortKey;
                countBuffer[sortKey]++;
            }
        }

        // change countBuffer[i] so that it contains actual position of this digit in output
        for (let i = 1; i < bucketCount; i++) {
            countBuffer[i] += countBuffer[i - 1];
        }

        // build the output array
        for (let i = 0; i < numVertices; i++) {
            order[--countBuffer[distances[i]]] = i;
        }

        // behind-camera splats sit at the end of the order; `count` is where they begin
        const cameraDist = px * dx + py * dy + pz * dz;
        const dist = (i) => {
            let o = order[i] * 3;
            return centers[o++] * dx + centers[o++] * dy + centers[o] * dz - cameraDist;
        };
        const findZero = () => {
            const result = binarySearch(0, numVertices - 1, i => -dist(i));
            return Math.min(numVertices, Math.abs(result));
        };

        const count = dist(numVertices - 1) >= 0 ? findZero() : numVertices;

        // send results
        self.postMessage({
            order: order.buffer,
            count,
            sortTime: performance.now() - sortStartTime
        }, [order.buffer]);

        order = null; // until the main thread hands a buffer back
    };

    self.addEventListener("message", (e) => {
        const msg = e.data;
        if (msg.order) order = new Float32Array(msg.order);
        if (msg.centers) {
            centers = new Float32Array(msg.centers);
            forceUpdate = true;

            // chunk bounds from the centers: one sphere per 256 splats plus the global AABB
            const numVertices = centers.length / 3;
            const numChunks = Math.ceil(numVertices / 256);
            chunks = new Float32Array(numChunks * 4);

            boundMin.x = boundMin.y = boundMin.z = Infinity;
            boundMax.x = boundMax.y = boundMax.z = -Infinity;

            for (let c = 0; c < numChunks; ++c) {
                let mx = Infinity, my = Infinity, mz = Infinity;
                let Mx = -Infinity, My = -Infinity, Mz = -Infinity;

                const start = c * 256;
                const end = Math.min(numVertices, start + 256);
                for (let i = start; i < end; ++i) {
                    const x = centers[i * 3];
                    const y = centers[i * 3 + 1];
                    const z = centers[i * 3 + 2];

                    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
                        centers[i * 3] = centers[i * 3 + 1] = centers[i * 3 + 2] = 0; // guard packed NaNs
                        continue;
                    }

                    if (x < mx) mx = x;
                    if (x > Mx) Mx = x;
                    if (y < my) my = y;
                    if (y > My) My = y;
                    if (z < mz) mz = z;
                    if (z > Mz) Mz = z;
                }
                if (mx > Mx) { mx = Mx = my = My = mz = Mz = 0; } // all-NaN chunk

                // calculate chunk center and radius from bound min/max
                chunks[c * 4] = (mx + Mx) * 0.5;
                chunks[c * 4 + 1] = (my + My) * 0.5;
                chunks[c * 4 + 2] = (mz + Mz) * 0.5;
                chunks[c * 4 + 3] = Math.sqrt((Mx - mx) ** 2 + (My - my) ** 2 + (Mz - mz) ** 2) * 0.5;

                if (mx < boundMin.x) boundMin.x = mx;
                if (Mx > boundMax.x) boundMax.x = Mx;
                if (my < boundMin.y) boundMin.y = my;
                if (My > boundMax.y) boundMax.y = My;
                if (mz < boundMin.z) boundMin.z = mz;
                if (Mz > boundMax.z) boundMax.z = Mz;
            }
        }
        if (msg.cameraPosition) cameraPosition = msg.cameraPosition;
        if (msg.cameraDirection) cameraDirection = msg.cameraDirection;

        update();
    });
}

/**
 * Spins up the sort worker. `centers` (Float32Array, 3 per splat) is transferred and
 * must not be used afterwards. `onOrder(order, count, sortTime)` receives the
 * back-to-front splat indices (valid only during the call; the buffer is recycled),
 * the number of leading in-front entries (the instanced draw count), and the sort
 * duration in ms.
 */
export function createSplatSorter(centers, numSplats, onOrder) {
    const source = `(${sortWorker.toString()})()`;
    const worker = new Worker(URL.createObjectURL(new Blob([source], { type: "application/javascript" })));

    const first = new Float32Array(numSplats); // the worker's initial sort target
    worker.postMessage({ centers: centers.buffer, order: first.buffer }, [centers.buffer, first.buffer]);
    let spare = new ArrayBuffer(numSplats * 4); // double-buffer held on this side

    worker.addEventListener("message", (e) => {
        const order = new Float32Array(e.data.order);
        onOrder(order, e.data.count, e.data.sortTime);
        worker.postMessage({ order: spare }, [spare]); // hand the worker its next target
        spare = e.data.order;                          // and keep this one as the new spare
    });

    return {
        // pass -forward as `backward` (PlayCanvas's cameraMat.getZ()); the worker
        // dedupes with an epsilon, so calling every frame is fine
        setCamera(pos, backward) {
            worker.postMessage({
                cameraPosition: { x: pos.x, y: pos.y, z: pos.z },
                cameraDirection: { x: backward.x, y: backward.y, z: backward.z },
            });
        },
        destroy() { worker.terminate(); },
    };
}
