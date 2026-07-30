"""
Download GOES-16 snapshots and AUTO-FIND regions with highest convective activity.
Scans the full CONUS image, scores each 256x256 window, picks the best crop.

Run in Google Colab. Each .npy: shape (5, H, W), float32, RAW values.
"""

import os
import json
import numpy as np
import s3fs
import xarray as xr
from datetime import datetime

INPUT_CHANNELS = ['CMI_C07', 'CMI_C09', 'CMI_C13', 'CMI_C14', 'CMI_C15']
CROP_SIZE = 256
SCAN_STRIDE = 64  # stride for scanning (smaller = more thorough but slower)
MIN_CONV_PCT = 5.0  # minimum % convective to keep

# Dates/hours known to have significant convective activity
SNAPSHOTS = [
    {"date": "2019-05-20", "hour": "23", "title": "Central US Supercells"},
    {"date": "2019-05-28", "hour": "00", "title": "Kansas Tornado Outbreak"},
    {"date": "2020-08-10", "hour": "18", "title": "Midwest Derecho"},
    {"date": "2021-12-11", "hour": "03", "title": "Kentucky Tornado Outbreak"},
    {"date": "2022-04-05", "hour": "22", "title": "Southeast Severe Storms"},
    {"date": "2022-06-12", "hour": "22", "title": "Great Plains Convection"},
    {"date": "2022-07-05", "hour": "22", "title": "Midwest Summer Storms"},
    {"date": "2021-07-28", "hour": "22", "title": "Southeast Thunderstorms"},
]

OUTPUT_DIR = "examples"
os.makedirs(OUTPUT_DIR, exist_ok=True)

fs = s3fs.S3FileSystem(anon=True)


def find_snapshot(date, hour):
    year, month, day = date.split("-")
    doy = datetime(int(year), int(month), int(day)).timetuple().tm_yday
    mcmip_prefix = f"noaa-goes16/ABI-L2-MCMIPC/{year}/{doy:03d}/{hour}/"
    actpc_prefix = f"noaa-goes16/ABI-L2-ACTPC/{year}/{doy:03d}/{hour}/"
    try:
        mcmip_files = sorted(fs.ls(mcmip_prefix))
        actpc_files = sorted(fs.ls(actpc_prefix))
    except Exception:
        return None, None
    if not mcmip_files or not actpc_files:
        return None, None
    return mcmip_files[0], actpc_files[0]


def open_nc(s3_path, variables=None):
    with fs.open(s3_path, 'rb') as f:
        ds = xr.open_dataset(f, engine='h5netcdf')
        if variables:
            ds = ds[variables]
        ds.load()
    return ds


def find_best_crop(ds_mcmip, ds_actpc, crop_size=256, stride=64):
    """Scan image and find the crop with highest convective percentage."""
    phase = ds_actpc['Phase'].values
    ctt = ds_mcmip['CMI_C13'].values
    H, W = phase.shape

    gt_full = ((phase == 4) & (ctt < 220)).astype(np.float32)
    gt_full[np.isnan(phase) | np.isnan(ctt)] = 0

    best_score = 0
    best_r, best_c = 0, 0

    for r in range(0, H - crop_size + 1, stride):
        for c in range(0, W - crop_size + 1, stride):
            patch = gt_full[r:r+crop_size, c:c+crop_size]
            score = patch.mean()
            if score > best_score:
                best_score = score
                best_r, best_c = r, c

    return best_r, best_c, best_score * 100


def extract_crop(ds_mcmip, ds_actpc, r, c, size):
    """Extract 5-channel crop and ground truth mask."""
    channels = []
    for ch in INPUT_CHANNELS:
        data = ds_mcmip[ch].values[r:r+size, c:c+size].astype(np.float32)
        if np.isnan(data).any():
            data[np.isnan(data)] = np.nanmean(data)
        channels.append(data)

    crop = np.stack(channels, axis=0)

    phase = ds_actpc['Phase'].values[r:r+size, c:c+size]
    ctt = ds_mcmip['CMI_C13'].values[r:r+size, c:c+size]
    gt_mask = ((phase == 4) & (ctt < 220)).astype(np.float32)
    gt_mask[np.isnan(phase) | np.isnan(ctt)] = 0

    return crop, gt_mask


manifest = []

for snap in SNAPSHOTS:
    print(f"\n=== {snap['title']} ({snap['date']} {snap['hour']}Z) ===")

    mcmip_path, actpc_path = find_snapshot(snap["date"], snap["hour"])
    if mcmip_path is None:
        print("  Files not found, skipping")
        continue

    print("  Loading MCMIPC...")
    ds_mcmip = open_nc(mcmip_path, INPUT_CHANNELS)
    print("  Loading ACTPC...")
    ds_actpc = open_nc(actpc_path, ['Phase'])

    print(f"  Scanning for best {CROP_SIZE}x{CROP_SIZE} crop...")
    best_r, best_c, conv_pct = find_best_crop(
        ds_mcmip, ds_actpc, CROP_SIZE, SCAN_STRIDE
    )
    print(f"  Best crop at ({best_r}, {best_c}): {conv_pct:.1f}% convective")

    if conv_pct < MIN_CONV_PCT:
        print(f"  Skipping (below {MIN_CONV_PCT}% threshold)")
        continue

    crop, gt_mask = extract_crop(ds_mcmip, ds_actpc, best_r, best_c, CROP_SIZE)

    name = snap["title"].lower().replace(" ", "_").replace("-", "_")
    npy_path = os.path.join(OUTPUT_DIR, f"{name}.npy")
    np.save(npy_path, crop)
    size_kb = os.path.getsize(npy_path) / 1024
    print(f"  Saved: {npy_path} ({size_kb:.0f} KB)")

    gt_path = os.path.join(OUTPUT_DIR, f"{name}_gt.npy")
    np.save(gt_path, gt_mask)

    manifest.append({
        "name": name,
        "title": snap["title"],
        "date": snap["date"],
        "hour": snap["hour"],
        "file": f"{name}.npy",
        "gt_file": f"{name}_gt.npy",
        "shape": list(crop.shape),
        "convective_pct": round(float(conv_pct), 1)
    })

with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2, default=lambda x: float(x))

print(f"\nDone! Examples: {len(manifest)}")
for m in manifest:
    print(f"  {m['title']}: {m['convective_pct']}%")
