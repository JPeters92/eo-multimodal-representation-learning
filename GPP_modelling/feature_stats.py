#!/usr/bin/env python3
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

# ------------------------
# Paths / inputs
# ------------------------
IN_DIR   = Path("/net/data_ssd/deepfeatures/sciencecubes_processed")
OUT_DIR  = Path("/net/data_ssd/deepfeatures/sciencecubes_processed")

CUBE_IDS = [
   "009", "021", "022", "003", "008", "027", "028"
]

VAR_NAME = "feature_mean_ucm"  # produced by your writer

# ------------------------
# Helpers
# ------------------------
def _safe(da: xr.DataArray) -> xr.DataArray:
    """Replace ±inf with NaN to avoid contaminating stats."""
    return xr.where(np.isfinite(da), da, np.nan)

def _read_cube_da(cid: str):
    """Load cube and return DataArray, radiation index, radiation units."""
    z = IN_DIR / f"s1_s2_{cid}_v_mean_ucm_flux.zarr"
    ds = xr.open_zarr(z, consolidated=True)
    if VAR_NAME not in ds:
        raise FileNotFoundError(f"{VAR_NAME} not found in {z}")
    da = ds[VAR_NAME].sortby("time")
    rad_idx   = ds[VAR_NAME].attrs.get("radiation_feature_index", None)
    rad_units = ds[VAR_NAME].attrs.get("radiation_units", "")
    return _safe(da), (int(rad_idx) if rad_idx is not None else None), str(rad_units or "")

def _per_feature_time_stats(da_ft: xr.DataArray):
    """Compute feature-wise time stats."""
    s_min  = da_ft.min(dim="time", skipna=True).compute().values.astype("float64")
    s_max  = da_ft.max(dim="time", skipna=True).compute().values.astype("float64")
    s_mean = da_ft.mean(dim="time", skipna=True).compute().values.astype("float64")
    s_std  = da_ft.std(dim="time", skipna=True, ddof=1).compute().values.astype("float64")
    return {"min": s_min, "max": s_max, "mean": s_mean, "std": s_std}

# ------------------------
# Main loop
# ------------------------
per_cube_rows = []
all_for_overall = []
rad_idx_global = None
rad_units_global = ""

for cid in CUBE_IDS:
    print(f"\n→ Processing cube {cid}")
    try:
        da_ft, rad_idx, rad_units = _read_cube_da(cid)
    except Exception as e:
        print(f"⚠️  Skipping {cid}: {e}")
        continue

    # remember radiation feature index once
    if rad_idx is not None and rad_idx_global is None:
        rad_idx_global = rad_idx
        rad_units_global = rad_units

    # compute stats
    stats = _per_feature_time_stats(da_ft)
    feats = da_ft["feature"].values if "feature" in da_ft.coords else np.arange(da_ft.sizes["feature"])

    dfc = pd.DataFrame({
        "cube_id": cid,
        "feature": feats,
        "min":  stats["min"],
        "max":  stats["max"],
        "mean": stats["mean"],
        "std":  stats["std"],
    })
    dfc["is_radiation"] = (dfc["feature"] == rad_idx) if rad_idx is not None else False
    dfc["units"] = np.where(dfc["is_radiation"], rad_units, "")

    if da_ft.sizes.get("time", 0) > 0:
        t0 = pd.to_datetime(da_ft["time"].values[0]).date()
        t1 = pd.to_datetime(da_ft["time"].values[-1]).date()
        dfc["time_start"] = t0
        dfc["time_end"] = t1

    # print cube summary
    print("   Summary:")
    print(dfc[["feature", "min", "max", "mean", "std", "is_radiation"]].round(4).to_string(index=False))

    # save per-cube CSV
    out_csv = OUT_DIR / f"s1_s2_{cid}_v_mean_ucm_flux_stats.csv"
    dfc.to_csv(out_csv, index=False)
    print(f"✅ Saved {out_csv}")

    per_cube_rows.append(dfc)
    all_for_overall.append(da_ft.expand_dims(cube=[cid]))

# ------------------------
# Combined per-cube table
# ------------------------
if per_cube_rows:
    df_all = pd.concat(per_cube_rows, ignore_index=True)
    out_all_cubes = OUT_DIR / "feature_mean_ucm_stats_all_cubes_per_cube.csv"
    df_all.to_csv(out_all_cubes, index=False)
    print(f"\n📦 Combined per-cube table → {out_all_cubes}")
else:
    print("No per-cube stats produced.")
    df_all = None

# ------------------------
# Overall (pooled across ALL cubes & days)
# ------------------------
if all_for_overall:
    print("\n→ Computing overall pooled stats across all cubes...")
    big = xr.concat(all_for_overall, dim="cube")

    overall_min  = big.min(dim=("cube", "time"), skipna=True).compute().values.astype("float64")
    overall_max  = big.max(dim=("cube", "time"), skipna=True).compute().values.astype("float64")
    overall_mean = big.mean(dim=("cube", "time"), skipna=True).compute().values.astype("float64")
    overall_std  = big.std(dim=("cube", "time"), skipna=True, ddof=1).compute().values.astype("float64")

    feats = big["feature"].values if "feature" in big.coords else np.arange(big.sizes["feature"])
    df_overall = pd.DataFrame({
        "feature": feats,
        "min":  overall_min,
        "max":  overall_max,
        "mean": overall_mean,
        "std":  overall_std,
    })

    if rad_idx_global is not None:
        df_overall["is_radiation"] = (df_overall["feature"] == rad_idx_global)
        df_overall["units"] = np.where(df_overall["is_radiation"], rad_units_global, "")
    else:
        df_overall["is_radiation"] = False
        df_overall["units"] = ""

    # print overall results
    print("\n🌍 Overall pooled per-feature statistics:")
    print(df_overall.round(4).to_string(index=False))

    out_overall = OUT_DIR / "feature_mean_ucm_stats_overall_pooled.csv"
    df_overall.to_csv(out_overall, index=False)
    print(f"✅ Saved overall pooled stats → {out_overall}")
else:
    print("No overall stats produced (nothing loaded).")