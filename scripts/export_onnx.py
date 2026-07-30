"""
Export ResNet-18 (5-channel) to ONNX for browser inference.
Run in Google Colab or any environment with PyTorch installed.

Usage (Colab):
    %run scripts/export_onnx.py
Or edit WEIGHTS/OUTPUT/STATS paths at the bottom of this file.
"""

import json
import torch
import torch.nn as nn
from torchvision import models
import numpy as np

# ── Channel normalization stats (from training) ──────────────────────
NORM_STATS = {
    "channels": ["CMI_C07", "CMI_C09", "CMI_C13", "CMI_C14", "CMI_C15"],
    "means": [288.66, 245.55, 278.85, 278.19, 275.79],
    "stds":  [19.17,  12.98,  23.49,  23.76,  23.10],
    "patch_size": 64,
    "stride": 48,
    "convective_threshold": 0.5
}


def build_model():
    """Build the same ResNet-18 architecture used in training."""
    model = models.resnet18(weights=None)
    # Adapt first conv for 5 input channels
    old_conv = model.conv1
    model.conv1 = nn.Conv2d(
        5, 64, kernel_size=7, stride=2, padding=3, bias=False
    )
    with torch.no_grad():
        # Copy first 3 channels, average for channels 4-5
        model.conv1.weight[:, :3] = old_conv.weight
        model.conv1.weight[:, 3] = old_conv.weight.mean(dim=1)
        model.conv1.weight[:, 4] = old_conv.weight.mean(dim=1)
    # Binary classification head
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def export(weights_path, output_path, stats_path):
    print(f"Loading weights from {weights_path}...")
    model = build_model()
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # Dummy input: batch=1, channels=5, 64x64
    dummy = torch.randn(1, 5, 64, 64)

    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch"},
            "output": {0: "batch"}
        },
        opset_version=13
    )

    # Save normalization stats as JSON
    print(f"Saving normalization stats to {stats_path}...")
    with open(stats_path, "w") as f:
        json.dump(NORM_STATS, f, indent=2)

    # Verify
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    import os
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"Done! Model size: {size_mb:.1f} MB")
    print(f"Stats saved to {stats_path}")


if __name__ == "__main__":
    import os

    # ── Edit these paths if needed ──
    WEIGHTS = "checkpoints/best_model.pth"
    OUTPUT  = "model/resnet18_goes.onnx"
    STATS   = "model/norm_stats.json"

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    export(WEIGHTS, OUTPUT, STATS)
