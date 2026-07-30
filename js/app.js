/**
 * Main application controller.
 * Handles file upload, example loading, inference pipeline, UI state.
 */

const MODEL_PATH = 'model/resnet18_goes.onnx';
const EXAMPLES_MANIFEST = 'examples/manifest.json';

let currentData = null;   // { data: Float32Array, shape: number[] }
let currentResult = null; // inference result

// ── UI Refs ──────────────────────────────────────────────────
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const statusPanel = document.getElementById('status-panel');
const statusText = document.getElementById('status-text');
const progressContainer = document.getElementById('progress-container');
const progressBar = document.getElementById('progress-bar');
const progressText = document.getElementById('progress-text');
const resultsPanel = document.getElementById('results-panel');
const canvasInput = document.getElementById('canvas-input');
const canvasBase = document.getElementById('canvas-base');
const canvasOverlay = document.getElementById('canvas-overlay');
const opacitySlider = document.getElementById('opacity-slider');
const showGridCheckbox = document.getElementById('show-grid');
const examplesList = document.getElementById('examples-list');

// ── Init ─────────────────────────────────────────────────────
(async function init() {
    setupUploadHandlers();
    loadExamples();

    // Pre-load model
    showStatus('Loading model...');
    try {
        await loadModel(MODEL_PATH, msg => setStatus(msg));
        hideStatus();
    } catch (e) {
        setStatus('Failed to load model. Make sure resnet18_goes.onnx is in model/');
        console.error(e);
    }
})();

// ── Upload Handlers ──────────────────────────────────────────
function setupUploadHandlers() {
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', e => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', e => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        const file = e.dataTransfer.files[0];
        if (file) handleFile(file);
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files[0]) handleFile(fileInput.files[0]);
    });

    // Overlay controls
    opacitySlider.addEventListener('input', () => {
        if (currentResult) redrawOverlay();
    });

    showGridCheckbox.addEventListener('change', () => {
        if (currentResult) redrawOverlay();
    });
}

// ── File Handling ────────────────────────────────────────────
async function handleFile(file) {
    if (!file.name.endsWith('.npy')) {
        setStatus('Please upload a .npy file');
        return;
    }

    showStatus(`Reading ${file.name}...`);
    try {
        const buffer = await file.arrayBuffer();
        const { shape, data } = parseNpy(buffer);

        if (shape.length !== 3 || shape[0] !== 5) {
            setStatus(`Invalid shape: [${shape}]. Expected (5, H, W)`);
            return;
        }

        currentData = { data, shape };
        await processImage();
    } catch (e) {
        setStatus('Error reading file: ' + e.message);
        console.error(e);
    }
}

async function handleExampleUrl(url, name) {
    showStatus(`Downloading ${name}...`);
    try {
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const buffer = await resp.arrayBuffer();
        const { shape, data } = parseNpy(buffer);

        currentData = { data, shape };
        await processImage();
    } catch (e) {
        setStatus('Error loading example: ' + e.message);
        console.error(e);
    }
}

// ── Processing Pipeline ─────────────────────────────────────
async function processImage() {
    const { data, shape } = currentData;
    const [, H, W] = shape;

    // Render input
    renderC13(data, H, W, canvasInput);

    // Check model
    if (!modelReady) {
        setStatus('Waiting for model to load...');
        await loadModel(MODEL_PATH, msg => setStatus(msg));
    }

    // Run inference
    showStatus('Running inference...');
    showProgress();

    currentResult = await runInference(data, shape, (done, total) => {
        const pct = Math.round(done / total * 100);
        setProgress(pct, `${done}/${total} patches`);
    });

    hideStatus();

    // Render results
    renderC13(data, H, W, canvasBase);
    redrawOverlay();
    updateResultsStats();

    resultsPanel.hidden = false;
    resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function redrawOverlay() {
    const { mask, H, W, patchPositions, threshold, patchSize } = currentResult;
    const opacity = opacitySlider.value / 100;

    renderOverlay(mask, H, W, canvasOverlay, opacity, threshold);

    if (showGridCheckbox.checked) {
        drawPatchGrid(canvasOverlay, patchPositions, patchSize);
    }
}

function updateResultsStats() {
    const { mask, H, W, threshold } = currentResult;
    let convCount = 0;
    for (let i = 0; i < mask.length; i++) {
        if (mask[i] >= threshold) convCount++;
    }
    updateStats(currentResult, convCount);
}

// ── Examples ─────────────────────────────────────────────────
async function loadExamples() {
    try {
        const resp = await fetch(EXAMPLES_MANIFEST);
        if (!resp.ok) {
            examplesList.innerHTML = '<p class="loading-msg">No examples available yet. Run prepare_examples.py first.</p>';
            return;
        }
        const manifest = await resp.json();

        examplesList.innerHTML = '';
        for (const ex of manifest) {
            const card = document.createElement('div');
            card.className = 'example-card';
            card.innerHTML = `
                <div>
                    <div class="name">${ex.title}</div>
                    <div class="meta">${ex.date} ${ex.hour}Z &middot; ${ex.convective_pct}% convective</div>
                </div>
                <div class="actions">
                    <button class="btn-small" data-action="download" title="Download .npy">&#8595;</button>
                    <button class="btn-small primary" data-action="load" title="Load for inference">Load</button>
                </div>
            `;

            const fileUrl = `examples/${ex.file}`;

            card.querySelector('[data-action="load"]').addEventListener('click', e => {
                e.stopPropagation();
                handleExampleUrl(fileUrl, ex.title);
            });

            card.querySelector('[data-action="download"]').addEventListener('click', e => {
                e.stopPropagation();
                const a = document.createElement('a');
                a.href = fileUrl;
                a.download = ex.file;
                a.click();
            });

            examplesList.appendChild(card);
        }
    } catch (e) {
        examplesList.innerHTML = '<p class="loading-msg">Examples not available</p>';
    }
}

// ── Status Helpers ───────────────────────────────────────────
function showStatus(msg) {
    statusPanel.hidden = false;
    statusText.textContent = msg || '';
}

function setStatus(msg) {
    statusText.textContent = msg;
}

function hideStatus() {
    statusPanel.hidden = true;
    progressContainer.hidden = true;
}

function showProgress() {
    progressContainer.hidden = false;
    setProgress(0, '');
}

function setProgress(pct, text) {
    progressBar.style.setProperty('--pct', pct + '%');
    progressText.textContent = text;
}
