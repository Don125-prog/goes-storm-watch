/**
 * ONNX inference engine for convective cloud detection.
 * Loads ResNet-18 ONNX model, normalizes input, tiles into 64×64 patches,
 * runs batch inference, and assembles a prediction mask.
 */

const NORM = {
    means: [288.66, 245.55, 278.85, 278.19, 275.79],
    stds:  [19.17,  12.98,  23.49,  23.76,  23.10]
};
const PATCH = 64;
const STRIDE = 48;
const THRESHOLD = 0.5;

let session = null;
let modelReady = false;

/**
 * Load the ONNX model.
 * @param {string} modelPath - URL to the .onnx file
 * @param {function} onStatus - callback(message)
 */
async function loadModel(modelPath, onStatus) {
    onStatus(typeof t === 'function' ? t('loadingModel') : 'Loading ONNX model...');
    try {
        // Set WASM paths to CDN
        ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/';
        ort.env.wasm.numThreads = 1; // single thread for compatibility

        // Fetch model as ArrayBuffer first (more reliable for large files)
        onStatus(typeof t === 'function' ? t('loadingModel') : 'Downloading model...');
        const response = await fetch(modelPath);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const buffer = await response.arrayBuffer();

        onStatus(typeof t === 'function' ? t('loadingModel') : 'Initializing model...');
        session = await ort.InferenceSession.create(buffer, {
            executionProviders: ['wasm']
        });
        modelReady = true;
        onStatus(typeof t === 'function' ? t('modelLoaded') : 'Model loaded');
    } catch (e) {
        const msg = (typeof t === 'function' ? t('modelFailed') : 'Model load failed') + ': ' + (e.message || e);
        onStatus(msg);
        throw e;
    }
}

/**
 * Normalize a single patch in-place.
 * @param {Float32Array} patchData - length 5*64*64
 */
function normalizePatch(patchData) {
    const px = PATCH * PATCH;
    for (let c = 0; c < 5; c++) {
        const offset = c * px;
        const mean = NORM.means[c];
        const std = NORM.stds[c];
        for (let i = 0; i < px; i++) {
            patchData[offset + i] = (patchData[offset + i] - mean) / std;
        }
    }
}

/**
 * Extract a 64×64 patch from the full image.
 * @param {Float32Array} data - shape (5, H, W)
 * @param {number} H
 * @param {number} W
 * @param {number} r - top-left row
 * @param {number} c - top-left col
 * @returns {Float32Array} normalized patch data (5*64*64)
 */
function extractPatch(data, H, W, r, c) {
    const patch = new Float32Array(5 * PATCH * PATCH);
    for (let ch = 0; ch < 5; ch++) {
        const srcOff = ch * H * W;
        const dstOff = ch * PATCH * PATCH;
        for (let pr = 0; pr < PATCH; pr++) {
            const srcRow = srcOff + (r + pr) * W + c;
            const dstRow = dstOff + pr * PATCH;
            for (let pc = 0; pc < PATCH; pc++) {
                patch[dstRow + pc] = data[srcRow + pc];
            }
        }
    }
    normalizePatch(patch);
    return patch;
}

/**
 * Softmax over 2 logits.
 */
function softmax2(a, b) {
    const max = Math.max(a, b);
    const ea = Math.exp(a - max);
    const eb = Math.exp(b - max);
    return eb / (ea + eb); // probability of class 1 (convective)
}

/**
 * Run inference on the full image.
 * @param {Float32Array} data - raw image data, shape (5, H, W)
 * @param {number[]} shape - [5, H, W]
 * @param {function} onProgress - callback(done, total)
 * @returns {{ mask: Float32Array, patchPositions: Array, H: number, W: number, time: number }}
 */
async function runInference(data, shape, onProgress) {
    if (!modelReady) throw new Error('Model not loaded');

    const [, H, W] = shape;
    const t0 = performance.now();

    // Compute patch positions
    const positions = [];
    for (let r = 0; r <= H - PATCH; r += STRIDE) {
        for (let c = 0; c <= W - PATCH; c += STRIDE) {
            positions.push([r, c]);
        }
    }

    const total = positions.length;
    const BATCH_SIZE = 16;
    const mask = new Float32Array(H * W); // probability map
    const counts = new Float32Array(H * W); // overlap counter

    for (let bi = 0; bi < total; bi += BATCH_SIZE) {
        const batchEnd = Math.min(bi + BATCH_SIZE, total);
        const batchN = batchEnd - bi;

        // Build batch tensor
        const batchData = new Float32Array(batchN * 5 * PATCH * PATCH);
        for (let k = 0; k < batchN; k++) {
            const [r, c] = positions[bi + k];
            const patch = extractPatch(data, H, W, r, c);
            batchData.set(patch, k * 5 * PATCH * PATCH);
        }

        const tensor = new ort.Tensor('float32', batchData, [batchN, 5, PATCH, PATCH]);
        const results = await session.run({ input: tensor });
        const output = results.output.data; // shape [batchN, 2]

        // Scatter predictions into mask
        for (let k = 0; k < batchN; k++) {
            const prob = softmax2(output[k * 2], output[k * 2 + 1]);
            const [r, c] = positions[bi + k];
            for (let pr = 0; pr < PATCH; pr++) {
                for (let pc = 0; pc < PATCH; pc++) {
                    const idx = (r + pr) * W + (c + pc);
                    mask[idx] += prob;
                    counts[idx] += 1;
                }
            }
        }

        if (onProgress) onProgress(batchEnd, total);

        // Yield to UI
        await new Promise(r => setTimeout(r, 0));
    }

    // Average overlapping predictions
    for (let i = 0; i < mask.length; i++) {
        if (counts[i] > 0) mask[i] /= counts[i];
    }

    const time = performance.now() - t0;

    return {
        mask,
        patchPositions: positions,
        H, W,
        time,
        threshold: THRESHOLD,
        patchSize: PATCH,
        stride: STRIDE
    };
}
