"""
Download GOES-16 snapshots and crop storm regions as .npy examples.
Run in Google Colab or any environment with s3fs/xarray.

Usage:
    python prepare_examples.py --output examples/

Each .npy file: shape (5, H, W), float32, RAW values (NOT normalized).
Normalization happens in the browser during inference.
"""

import argparse
import os
import json
import numpy as np
import s3fs
import xarray as xr

INPUT_CHANNELS = ['CMI_C07', 'CMI_C09', 'CMI_C13', 'CMI_C14', 'CMI_C15']

# Interesting storm regions to crop (row_start, col_start, size)
# These coordinates target known convective areas in CONUS imagery
EXAMPLES = [
    {
        "date": "2022-06-12",
        "hour": "21",
        "name": "great_plains_storm",
        "title": "Great Plains Thunderstorms",
        "row": 200, "col": 800, "size": 256
    },
    {
        "date": "2022-07-05",
        "hour": "22",
        "name": "midwest_convection",
        "title": "Midwest Convective System",
        "row": 150, "col": 600, "size": 256
    },
    {
        "date": "2022-08-15",
        "hour": "20",
        "name": "gulf_storms",
        "title": "Gulf Coast Storms",
        "row": 400, "col": 500, "size": 256
    },
    {
        "date": "2019-05-20",
        "hour": "23",
        "name": "tornado_alley",
        "title": "Tornado Alley Supercells",
        "row": 250, "col": 700, "size": 256
    },
    {
        "date": "2021-07-28",
        "hour": "21",
        "name": "southeast_cells",
        "title": "Southeast Isolated Cells",
        "row": 350, "col": 400, "size": 256
    },
]

fs = s3fs.S3FileSystem(anon=True)


def find_snapshot(date, hour):
    """Find MCMIPC and ACTPC files for a given date/hour."""
    year, month, day = date.split("-")
    from datetime import datetime
    doy = datetime(int(year), int(month), int(day)).timetuple().tm_yday

    mcmip_prefix = f"noaa-goes16/ABI-L2-MCMIPC/{year}/{doy:03d}/{hour}/"
    actpc_prefix = f"noaa-goes16/ABI-L2-ACTPC/{year}/{doy:03d}/{hour}/"

    mcmip_files = sorted(fs.ls(mcmip_prefix))
    actpc_files = sorted(fs.ls(actpc_prefix))

    if not mcmip_files or not actpc_files:
        return None, None

    return mcmip_files[0], actpc_files[0]


def open_nc(s3_path, variables=None):
    """Open a NetCDF file from S3."""
    with fs.open(s3_path, 'rb') as f:
        ds = xr.open_dataset(f, engine='h5netcdf')
        if variables:
            ds = ds[variables]
        ds.load()
    return ds


def crop_example(ex, output_dir):
    """Download, crop, and save one example."""
    print(f"\n=== {ex['title']} ({ex['date']} {ex['hour']}Z) ===")

    mcmip_path, actpc_path = find_snapshot(ex["date"], ex["hour"])
    if mcmip_path is None:
        print("  Files not found, skipping")
        return None

    print(f"  Loading MCMIPC...")
    ds_mcmip = open_nc(mcmip_path, INPUT_CHANNELS)

    print(f"  Loading ACTPC...")
    ds_actpc = open_nc(actpc_path, ['Phase'])

    r, c, s = ex["row"], ex["col"], ex["size"]

    # Extract 5 channels
    channels = []
    for ch in INPUT_CHANNELS:
        data = ds_mcmip[ch].values[r:r+s, c:c+s].astype(np.float32)
        # Fill NaN with channel mean
        if np.isnan(data).any():
            data[np.isnan(data)] = np.nanmean(data)
        channels.append(data)

    crop = np.stack(channels, axis=0)  # (5, H, W)

    # Also extract ground truth mask for reference
    phase = ds_actpc['Phase'].values[r:r+s, c:c+s]
    ctt = ds_mcmip['CMI_C13'].values[r:r+s, c:c+s]
    gt_mask = ((phase == 4) & (ctt < 220)).astype(np.float32)
    gt_mask[np.isnan(phase) | np.isnan(ctt)] = 0

    conv_pct = gt_mask.mean() * 100
    print(f"  Crop shape: {crop.shape}, convective: {conv_pct:.1f}%")

    # Save raw (unnormalized) crop
    npy_path = os.path.join(output_dir, f"{ex['name']}.npy")
    np.save(npy_path, crop)
    size_kb = os.path.getsize(npy_path) / 1024
    print(f"  Saved: {npy_path} ({size_kb:.0f} KB)")

    # Save ground truth mask for reference
    gt_path = os.path.join(output_dir, f"{ex['name']}_gt.npy")
    np.save(gt_path, gt_mask)

    return {
        "name": ex["name"],
        "title": ex["title"],
        "date": ex["date"],
        "hour": ex["hour"],
        "file": f"{ex['name']}.npy",
        "gt_file": f"{ex['name']}_gt.npy",
        "shape": list(crop.shape),
        "convective_pct": round(conv_pct, 1)
    }


def main(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    manifest = []

    for ex in EXAMPLES:
        info = crop_example(ex, output_dir)
        if info:
            manifest.append(info)

    # Save manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved: {manifest_path}")
    print(f"Total examples: {len(manifest)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="examples/")
    args = parser.parse_args()
    main(args.output)
