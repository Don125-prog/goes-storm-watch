/**
 * Visualization: render C13 brightness temperature as grayscale,
 * overlay convective predictions in red.
 */

/**
 * Render the C13 channel (index 2) as a grayscale image.
 * Cold = bright (white), warm = dark.
 * @param {Float32Array} data - raw (5, H, W)
 * @param {number} H
 * @param {number} W
 * @param {HTMLCanvasElement} canvas
 */
function renderC13(data, H, W, canvas) {
    canvas.width = W;
    canvas.height = H;
    const ctx = canvas.getContext('2d');
    const img = ctx.createImageData(W, H);

    // C13 is channel index 2
    const c13Offset = 2 * H * W;

    // Find min/max for contrast stretch
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < H * W; i++) {
        const v = data[c13Offset + i];
        if (v < min) min = v;
        if (v > max) max = v;
    }

    const range = max - min || 1;

    for (let i = 0; i < H * W; i++) {
        const v = data[c13Offset + i];
        // Invert: cold temperatures (low K) → bright (high pixel value)
        const norm = 1 - (v - min) / range;
        const px = Math.round(norm * 255);
        const idx = i * 4;
        img.data[idx]     = px;
        img.data[idx + 1] = px;
        img.data[idx + 2] = px;
        img.data[idx + 3] = 255;
    }

    ctx.putImageData(img, 0, 0);
}

/**
 * Render prediction overlay.
 * @param {Float32Array} mask - probability map (H * W)
 * @param {number} H
 * @param {number} W
 * @param {HTMLCanvasElement} canvas
 * @param {number} opacity - 0..1
 * @param {number} threshold
 */
function renderOverlay(mask, H, W, canvas, opacity, threshold) {
    canvas.width = W;
    canvas.height = H;
    const ctx = canvas.getContext('2d');
    const img = ctx.createImageData(W, H);

    for (let i = 0; i < H * W; i++) {
        const p = mask[i];
        const idx = i * 4;
        if (p >= threshold) {
            // Red overlay for convective
            const intensity = Math.min(p * 1.5, 1);
            img.data[idx]     = Math.round(255 * intensity);
            img.data[idx + 1] = Math.round(40 * (1 - intensity));
            img.data[idx + 2] = 40;
            img.data[idx + 3] = Math.round(opacity * 255);
        } else {
            // Transparent for non-convective
            img.data[idx + 3] = 0;
        }
    }

    ctx.putImageData(img, 0, 0);
}

/**
 * Draw patch grid lines on the overlay canvas.
 * @param {HTMLCanvasElement} canvas
 * @param {Array} positions - [[r, c], ...]
 * @param {number} patchSize
 */
function drawPatchGrid(canvas, positions, patchSize) {
    const ctx = canvas.getContext('2d');
    ctx.strokeStyle = 'rgba(79, 195, 247, 0.15)';
    ctx.lineWidth = 0.5;

    const drawn = new Set();
    for (const [r, c] of positions) {
        const key = `${r},${c}`;
        if (drawn.has(key)) continue;
        drawn.add(key);
        ctx.strokeRect(c + 0.5, r + 0.5, patchSize - 1, patchSize - 1);
    }
}

/**
 * Update stats bar.
 */
function updateStats(result, convCount) {
    const { H, W, patchPositions, time } = result;
    document.getElementById('stat-dims').innerHTML =
        `Size: <strong>${W} × ${H}</strong>`;
    document.getElementById('stat-patches').innerHTML =
        `Patches: <strong>${patchPositions.length}</strong>`;
    document.getElementById('stat-conv').innerHTML =
        `Convective: <strong>${convCount} px</strong> (${(convCount / (H * W) * 100).toFixed(1)}%)`;
    document.getElementById('stat-time').innerHTML =
        `Inference: <strong>${(time / 1000).toFixed(1)}s</strong>`;
}
