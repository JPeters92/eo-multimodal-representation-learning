#!/usr/bin/env python3
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Set
from config import CUBE_IDS
from sites import sites_dict  # unchanged from your code

ROOT_DIR = Path("/net/data/Fluxnet/")
IN_DIR   = Path("/net/data_ssd/deepfeatures/sciencecubes_processed")
OUT_DIR  = Path("/net/data_ssd/deepfeatures/sciencecubes_processed")

# ---------------------------------------------------------------------
# detect FLUX years (unchanged helper)
# ---------------------------------------------------------------------
def detect_flux_years_for_site(site: str, root: Path) -> Set[int]:
    years: Set[int] = set()
    ww_dir = root / "FLUXNET2020-ICOS-WarmWinter"
    if ww_dir.exists():
        for p in ww_dir.glob("FLX_*_FLUXNET2015_FULLSET_DD_*_beta-3.csv"):
            if site in p.name:
                years.update({2017, 2018, 2019, 2020})
                break
    for dpat in ["ICOS_2021_I", "ICOS_2022_I", "ICOS_2023_I", "ICOS_2024_I"]:
        d = root / dpat
        if not d.exists():
            continue
        y = int(dpat.split("_")[1])
        for p in d.glob("ICOSETC_*_FLUXNET_DD_01.csv"):
            if site in p.name:
                years.add(y)
                break
    return {y for y in years if 2017 <= y <= 2024}

# ---------------------------------------------------------------------
# Fill a single feature for a given year – LINEAR ONLY
# ---------------------------------------------------------------------
def linear_fill_one_year(ts: pd.Series, year: int) -> pd.Series:
    """Daily index → linear interpolate only → no climatology or UCM."""
    # restrict to year
    ts_year = ts[ts.index.year == year]
    if len(ts_year) == 0:
        # return empty year
        idx = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        return pd.Series(np.nan, index=idx)

    # aggregate duplicates to 1/day
    ts_day = ts_year.groupby(ts_year.index.normalize()).mean()

    idx = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    daily = ts_day.reindex(idx)

    # linear-only interpolation
    filled = daily.interpolate("linear", limit_direction="both")
    return filled

# ---------------------------------------------------------------------
# Fill for all features for a year
# ---------------------------------------------------------------------
def fill_feature_means_one_year(mean_da: xr.DataArray, year: int) -> xr.DataArray:
    feats = mean_da["feature"].values
    full_idx = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    out = []
    for f in feats:
        ts = mean_da.sel(feature=f).to_series()
        filled = linear_fill_one_year(ts, year)
        out.append(
            xr.DataArray(
                filled.values.astype("float32"),
                coords={"time": full_idx},
                dims=["time"]
            )
        )
    return xr.concat(out, dim="feature").assign_coords(feature=feats)

# ---------------------------------------------------------------------
# Fill std features for all features for a year
# ---------------------------------------------------------------------
def fill_feature_stds_one_year(std_da: xr.DataArray, year: int) -> xr.DataArray:
    feats = std_da["feature"].values
    full_idx = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    out = []
    for f in feats:
        ts = std_da.sel(feature=f).to_series()
        filled = linear_fill_one_year(ts, year)
        out.append(
            xr.DataArray(
                filled.values.astype("float32"),
                coords={"time": full_idx},
                dims=["time"]
            )
        )
    return xr.concat(out, dim="feature").assign_coords(feature=feats)

# ---------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------
for cid in CUBE_IDS:
    print(f"\n→ processing cube {cid}")
    in_path  = IN_DIR / f"s1_s2_{cid}_t.zarr"
    out_path = OUT_DIR / f"s1_s2_{cid}_t_mean_linear.zarr"

    ds = xr.open_zarr(in_path, consolidated=True)
    da = ds["features"]  # (feature, time, y, x)

    # spatial mean and spatial std
    mean_da = da.mean(dim=("y", "x"))
    std_da = da.std(dim=("y", "x"), skipna=True)

    # detect FLUX years
    site_code = sites_dict[cid][0]
    flux_years = detect_flux_years_for_site(site_code, ROOT_DIR)
    years_have = set(int(y) for y in np.unique(mean_da.time.dt.year))
    target_years = sorted(flux_years & years_have)

    if not target_years:
        print("   no matching years — skipping.")
        continue

    # restrict
    mean_da = mean_da.sel(time=mean_da.time.dt.year.isin(target_years))
    print(f"   target years: {target_years}")

    # fill
    yearly_mean = [fill_feature_means_one_year(mean_da, y) for y in target_years]
    yearly_std = [fill_feature_stds_one_year(std_da, y) for y in target_years]
    filled_mean = xr.concat(yearly_mean, dim="time").sortby("time")
    filled_std = xr.concat(yearly_std, dim="time").sortby("time")

    # optionally sanitize
    filled_mean = xr.where(np.isfinite(filled_mean), filled_mean, np.nan)
    filled_std = xr.where(np.isfinite(filled_std), filled_std, np.nan)

    # write
    ds_out = xr.Dataset({
        "feature_mean_linear": filled_mean,
        "feature_std_linear": filled_std,
    })
    radiation_idx = da.attrs.get("radiation_feature_index")
    if radiation_idx is not None:
        ds_out["feature_mean_linear"].attrs["radiation_feature_index"] = int(radiation_idx)
        ds_out["feature_std_linear"].attrs["radiation_feature_index"] = int(radiation_idx)
    enc = {
        "feature_mean_linear": {"chunks": (64, 180)},
        "feature_std_linear": {"chunks": (64, 180)},
    }
    ds_out.to_zarr(out_path, mode="w", consolidated=True, encoding=enc)

    print(f"   saved: {out_path}")
