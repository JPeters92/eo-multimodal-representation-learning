#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import numpy as np
import xarray as xr

# =====================
# CONFIG
# =====================
cube_nums = [f"{i:03d}" for i in range(500)]


s2_template   = "/net/data/deepfeatures/trainingcubes/000{cube}.zarr"
feat_template = "/net/data_ssd/deepfeatures/trainingcubes_processed/s1_s2_000{cube}_v.zarr"

# crop S2 by 7 px to match feature grid
BORDER = 7

# write a CSV summary?
WRITE_CSV = False
csv_path  = "valid_pixels_feature_vs_s2.csv"

# =====================
# HELPERS
# =====================
def daily(values):
    """Convert datetime64 array to daily resolution (datetime64[D])."""
    return np.asarray(values, dtype="datetime64[D]")

def select_time_by_day(da: xr.DataArray, target_day: np.datetime64) -> int:
    """
    Return the index in da.time whose day equals target_day.
    If multiple matches on the same day exist, take the one with
    the highest completeness (any-channel non-NaN) to be robust.
    """
    # day arrays
    days = daily(da.time.values)
    idxs = np.where(days == target_day)[0]
    if idxs.size == 0:
        # fallback to nearest by absolute difference in time
        return int(np.argmin(np.abs(da.time.values - np.datetime64(target_day))))
    if idxs.size == 1:
        return int(idxs[0])

    # multiple frames same day -> pick the most complete
    sub = da.isel(time=idxs)  # (..., time=k, y, x)
    # completeness = fraction of valid entries across channels and pixels
    valid = sub.notnull()  # (chan, time, y, x)
    valid_any = valid.any(dim=list(set(sub.dims) - {"time", "y", "x"}))  # (time, y, x)
    frac = (valid_any.sum(dim=("y","x")) / valid_any.isel(time=0).size).values  # (time,)
    best = int(idxs[int(np.argmax(frac))])
    return best

def count_valid_feature_pixels(feats_frame: xr.DataArray) -> int:
    """
    feats_frame: DataArray (feature, y, x) for one time.
    Valid if ANY feature is valid for that pixel.
    """
    valid_px = feats_frame.notnull().any(dim="feature")
    return int(valid_px.sum().values), int(valid_px.size)

def count_valid_s2_pixels(s2_frame: xr.DataArray) -> int:
    """
    s2_frame: DataArray (band, y, x) for one time.
    Valid if ANY band is valid for that pixel.
    """
    valid_px = s2_frame.notnull().any(dim="band")
    return int(valid_px.sum().values), int(valid_px.size)

# =====================
# MAIN
# =====================
if WRITE_CSV:
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cube", "date", "valid_feat", "total_feat", "pct_feat",
                         "valid_s2", "total_s2", "pct_s2", "diff_valid", "diff_pct"])

for cube in cube_nums:
    feat_path = feat_template.format(cube=cube)
    s2_path   = s2_template.format(cube=cube)

    if not (os.path.exists(feat_path) and os.path.exists(s2_path)):
        print(f"⚠️  Missing cube {cube}, skipping.")
        continue

    try:
        feat_ds = xr.open_zarr(feat_path, consolidated=True)
        s2_ds   = xr.open_zarr(s2_path,   consolidated=True)
    except Exception as e:
        print(f"❌ open_zarr failed for cube {cube}: {e}")
        continue

    if "features" not in feat_ds or "s2l2a" not in s2_ds or "cloud_mask" not in s2_ds:
        print(f"⚠️  Required variables missing for cube {cube}, skipping.")
        continue

    # feature times (already cropped grid in your pipeline)
    feat_times_D = daily(feat_ds.time.values)

    # S2 cloud-masked and cropped to match feature grid
    s2 = s2_ds.s2l2a.where(s2_ds.cloud_mask == 0)  # (band, time, y, x)
    # Crop by 7 px on each side to match features (y,x dims)
    if s2.sizes["y"] >= 2*BORDER and s2.sizes["x"] >= 2*BORDER:
        s2c = s2.isel(y=slice(BORDER, -BORDER), x=slice(BORDER, -BORDER))
    else:
        s2c = s2  # fallback (should not happen if shapes are standard)

    s2_times_D = daily(s2c.time.values)

    # common dates by day
    common_days = np.intersect1d(feat_times_D, s2_times_D)
    if common_days.size == 0:
        print(f"⚠️  No overlapping dates for cube {cube}")
        continue

    print(f"\n=== Cube {cube} ===")
    print(f"{'Date':<12} | {'Feat valid':>11} | {'%Feat':>6} | {'S2 valid':>10} | {'%S2':>5} | {'Δvalid':>10} | {'Δ%':>6}")
    print("-" * 72)

    for day in common_days:
        # pick indices for this day
        # features: exact by day
        feat_day_idxs = np.where(feat_times_D == day)[0]
        if feat_day_idxs.size == 0:
            continue
        # If multiples exist for features (rare), just take the first:
        t_idx_feat = int(feat_day_idxs[0])

        # sentinel-2: choose best/most complete frame of that same day
        t_idx_s2 = select_time_by_day(s2c, day)

        # slice frames
        feats_frame = feat_ds["features"].isel(time=t_idx_feat)  # (feature, y, x)
        s2_frame    = s2c.isel(time=t_idx_s2)                    # (band, y, x)

        # count valid
        v_feat, tot_feat = count_valid_feature_pixels(feats_frame)
        v_s2,   tot_s2   = count_valid_s2_pixels(s2_frame)

        pct_feat = 100.0 * v_feat / max(tot_feat, 1)
        pct_s2   = 100.0 * v_s2   / max(tot_s2,   1)

        date_str = str(np.datetime_as_string(day, unit="D"))
        print(f"{date_str:<12} | {v_feat:>11,} | {pct_feat:>5.1f}% | {v_s2:>10,} | {pct_s2:>4.1f}% | {v_feat - v_s2:>10,} | {(pct_feat - pct_s2):>5.1f}%")

        if WRITE_CSV:
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([cube, date_str, v_feat, tot_feat, f"{pct_feat:.3f}",
                                 v_s2, tot_s2, f"{pct_s2:.3f}", v_feat - v_s2, f"{(pct_feat - pct_s2):.3f}"])

print("\n✅ Done.")
if WRITE_CSV:
    print(f"📄 CSV written to: {csv_path}")
