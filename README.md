# GOES Storm Watch

Binary classification of convective (cumulonimbus) clouds from GOES-16 ABI infrared satellite imagery using ResNet-18. Runs entirely in the browser via ONNX Runtime Web.

**Live demo:** [Don125-prog.github.io/goes-storm-watch](https://Don125-prog.github.io/goes-storm-watch)

## How it works

1. Upload a `.npy` file with 5 IR channels (C07, C09, C13, C14, C15), shape `(5, H, W)`, float32
2. The image is tiled into 64×64 patches (stride 48) and each patch is classified
3. Predictions are overlaid on a C13 brightness temperature visualization

## Model

- **Architecture:** ResNet-18 adapted for 5 input channels
- **Training data:** GOES-16 ABI-L2-MCMIPC + ABI-L2-ACTPC, 46 snapshots, 69,552 patches
- **Label:** Cloud Top Phase == Ice AND CMI_C13 < 220 K
- **Performance:** Precision 91%, Recall 82%, F1 86%, AUC 0.993

## Setup

### 1. Export ONNX model (in Google Colab)

```bash
pip install torch torchvision onnx
python scripts/export_onnx.py --weights checkpoints/best_model.pth --output model/resnet18_goes.onnx
```

This creates `model/resnet18_goes.onnx` and `model/norm_stats.json`.

### 2. Prepare example crops (in Google Colab)

```bash
pip install s3fs xarray h5netcdf
python scripts/prepare_examples.py --output examples/
```

This downloads interesting storm region crops from GOES-16 S3 and saves them as `.npy` files with a `manifest.json`.

### 3. Deploy

Push everything to the `main` branch. GitHub Pages serves from root.

## Project structure

```
├── index.html              # Single-page app
├── css/style.css           # Styles
├── js/
│   ├── npy.js              # .npy file parser
│   ├── inference.js        # ONNX inference engine
│   ├── visualization.js    # Canvas rendering
│   └── app.js              # Main controller
├── model/
│   ├── resnet18_goes.onnx  # Exported model
│   └── norm_stats.json     # Normalization statistics
├── examples/
│   ├── manifest.json       # Example metadata
│   └── *.npy               # Example crops
├── checkpoints/
│   └── best_model.pth      # Trained PyTorch weights
└── scripts/
    ├── export_onnx.py      # PyTorch → ONNX export
    └── prepare_examples.py # Generate example crops
```

## Technology

- ONNX Runtime Web (WASM backend)
- Vanilla JavaScript, no frameworks
- GitHub Pages static hosting

---

Practice project — TsNIIMash / Garpix, 2026
