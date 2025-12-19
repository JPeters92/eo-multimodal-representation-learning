#!/usr/bin/env python3
import random
import re
from pathlib import Path
from typing import Dict, Set, List, Tuple, Optional

import pandas as pd
import xarray as xr
import numpy as np

from sites import sites_dict

# -------------------- CONFIG --------------------
FEATURE_DIR = Path("/net/data_ssd/deepfeatures/sciencecubes_processed")

ROOT_DIR = Path("/net/data/Fluxnet/")
DIR_PATTERNS  = [
    "ICOS_2020_I",
    "ICOS_2021_I",
    "ICOS_2022_I",
    "ICOS_2023_I",
    "ICOS_2024_I",
    "FLUXNET2020-ICOS-WarmWinter"
]
FILE_PATTERNS = [
    "ICOSETC_*_FLUXNET_DD_01.csv",
    "FLX_*_FLUXNET2015_FULLSET_DD_*_beta-3.csv",
]

# Train/val split
RNG_SEED = 42
N_TRAIN  = 28

# Minimum year to consider (flux + cube timestamps)
MIN_YEAR = 2017


# -------------------- HELPERS --------------------
def list_available_feature_cubes(feature_dir: Path) -> set[str]:
    """
    Return set of 3-digit cube IDs for which s1_s2_XXX_t.zarr exists.
    """
    ids: Set[str] = set()
    pattern = re.compile(r"^s1_s2_(\d{3})_t\.zarr$")
    for p in feature_dir.glob("s1_s2_*_t.zarr"):
        m = pattern.match(p.name)
        if m:
            ids.add(m.group(1))
    return ids


def collect_fluxnet_files(root: Path,
                          dir_patterns: List[str],
                          file_patterns: List[str]) -> List[Path]:
    """
    Collect all FLUXNET/ICOS files across the given directory+file patterns.
    """
    files: List[Path] = []
    for dpat in dir_patterns:
        for d in root.glob(dpat):
            for fpat in file_patterns:
                files.extend(d.glob(fpat))
    return files


def extract_site_from_filename(path: Path) -> Optional[str]:
    """
    Extract site code from known filename forms.
      FLX_DE-Hai_FLUXNET2015_FULLSET_DD_2014-2020_beta-3.csv  -> DE-Hai
      ICOSETC_DE-Hai_FLUXNET_DD_01.csv                        -> DE-Hai
    """
    tokens = path.stem.split("_")
    if len(tokens) >= 2:
        return tokens[1]
    site_re = re.compile(r"^[A-Z0-9]{2,3}[-_][A-Za-z0-9]{2,4}$")
    for t in tokens:
        if site_re.match(t):
            return t
    return None


def years_from_filename_hint(name: str) -> Set[int]:
    """
    Best-effort parse of a year range embedded in filenames, e.g. '_2014-2020_'.
    """
    yrs: Set[int] = set()
    m = re.search(r"_(\d{4})-(\d{4})_", name)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if a <= b:
            yrs.update(range(a, b + 1))
    return yrs


def safe_iter_years_from_csv(csv_path: Path, chunksize: int = 200_000) -> Set[int]:
    """
    Efficiently scan the CSV for a timestamp column and collect its years.
    Supports 'TIMESTAMP' or 'TIMESTAMP_START'. Falls back to filename hint if needed.
    """
    years: Set[int] = set()
    try:
        head = pd.read_csv(csv_path, nrows=0)
        cols_upper = head.columns.str.upper().tolist()
        ts_col: Optional[str] = None
        for cand in ("TIMESTAMP", "TIMESTAMP_START"):
            if cand in cols_upper:
                ts_col = head.columns[cols_upper.index(cand)]
                break

        if ts_col is None:
            return years_from_filename_hint(csv_path.name)

        for chunk in pd.read_csv(csv_path, usecols=[ts_col], chunksize=chunksize):
            s = chunk[ts_col]
            yrs = pd.to_numeric(s.astype(str).str.slice(0, 4), errors="coerce")
            years.update(int(y) for y in yrs.dropna().unique())

    except Exception:
        years |= years_from_filename_hint(csv_path.name)

    return years


def compute_site_years(files: List[Path]) -> Dict[str, Set[int]]:
    """
    Map: site_code -> set(years present in any of its files).
    """
    site_years: Dict[str, Set[int]] = {}
    for fp in files:
        site = extract_site_from_filename(fp)
        if not site:
            continue
        yrs = safe_iter_years_from_csv(fp)
        if not yrs:
            continue
        site_years.setdefault(site, set()).update(yrs)
    return site_years


def pick_data_var(ds: xr.Dataset) -> xr.DataArray:
    """
    Pick the main data variable in the feature Zarr. Tries common names,
    else the first variable with a 'time' dim, else the first data var.
    """
    for name in ["features", "bands", "data", "vars"]:
        if name in ds.data_vars:
            return ds[name]

    time_vars = [k for k in ds.data_vars if "time" in ds[k].dims]
    if time_vars:
        time_vars.sort(key=lambda v: ds[v].ndim, reverse=True)
        return ds[time_vars[0]]

    if ds.data_vars:
        k = list(ds.data_vars)[0]
        return ds[k]

    raise ValueError("No data variables found in feature dataset.")


def ensure_datetime_time(ds: xr.Dataset) -> xr.DataArray:
    """
    Ensure ds['time'] is datetime64 for dt.year operations.
    """
    t = ds["time"]
    if np.issubdtype(t.dtype, np.datetime64):
        return t
    # Try CF decode; if that fails, fall back to pandas
    try:
        t2 = xr.decode_cf(ds[["time"]])["time"]
        if np.issubdtype(t2.dtype, np.datetime64):
            return t2
    except Exception:
        pass
    return xr.DataArray(pd.to_datetime(t.values), dims=["time"], name="time")


def count_valid_timestamps_for_years(zarr_path: Path, years: Set[int]) -> Tuple[int, int, int]:
    """
    Returns (valid_count, total_in_years, total_all_time) where:
      - valid_count = # timesteps in selected years NOT entirely NaN across non-time dims
      - total_in_years = # timesteps in selected years
      - total_all_time = total timesteps in the cube
    Avoids `.item()` to stay Dask-friendly.
    """
    ds = xr.open_zarr(zarr_path, consolidated=True)
    da = pick_data_var(ds)

    if "time" not in da.dims:
        raise ValueError(f"'time' dimension not found in {zarr_path}")

    total_all = int(da.sizes.get("time", 0))
    if total_all == 0:
        return (0, 0, 0)

    if not years:
        return (0, 0, total_all)

    # Ensure datetime time for .dt.year
    time = ensure_datetime_time(ds)

    # Filter to selected years using xarray ops (keeps laziness)
    years = {y for y in years if y >= MIN_YEAR}
    if not years:
        return (0, 0, total_all)

    year_mask = time.dt.year.isin(sorted(years))
    da_years = da.sel(time=year_mask)

    total_in_years = int(da_years.sizes.get("time", 0))
    if total_in_years == 0:
        return (0, 0, total_all)

    # Timesteps fully NaN across all non-time dims
    non_time_dims = [d for d in da_years.dims if d != "time"]
    fully_nan = da_years.isnull().all(dim=non_time_dims)

    # Count timesteps that are NOT fully NaN (lazy until compute())
    valid_count_da = (~fully_nan).sum()

    # Avoid .item(): compute if dask-backed, else coerce via np.asarray
    valid_count: int
    if hasattr(valid_count_da, "compute"):
        valid_count = int(valid_count_da.compute())
    else:
        valid_count = int(np.asarray(valid_count_da))

    return (valid_count, total_in_years, total_all)


# -------------------- MAIN --------------------
if __name__ == "__main__":
    # 1) Feature cubes present
    feature_ids = list_available_feature_cubes(FEATURE_DIR)

    # 2) Fluxnet files and sites
    flux_files = collect_fluxnet_files(ROOT_DIR, DIR_PATTERNS, FILE_PATTERNS)
    flux_sites = {extract_site_from_filename(p) for p in flux_files}
    flux_sites.discard(None)

    # 2b) Years per site (then restrict to >= MIN_YEAR)
    site_years = compute_site_years(flux_files)
    site_years = {s: {y for y in yrs if y >= MIN_YEAR} for s, yrs in site_years.items()}

    # 3) Map site -> cube id(s)
    site_to_id: Dict[str, List[str]] = {}
    for cid, (site, _coords) in sites_dict.items():
        site_to_id.setdefault(site, []).append(cid)

    # 4) Identify candidates / missing
    candidate_ids: List[str] = []
    missing_feature: List[str] = []
    missing_flux: List[str] = []
    for cid, (site, _coords) in sites_dict.items():
        has_feature = cid in feature_ids
        has_flux    = site in flux_sites
        if has_feature and has_flux:
            candidate_ids.append(cid)
        else:
            if not has_feature:
                missing_feature.append(cid)
            if not has_flux:
                missing_flux.append(cid)

    # 5) Reproducible split
    random.Random(RNG_SEED).shuffle(candidate_ids)
    n_train = min(N_TRAIN, len(candidate_ids))
    train_ids = sorted(candidate_ids[:n_train])
    val_ids   = sorted(candidate_ids[n_train:])

    # 6) Summary
    print(f"Found feature cubes: {len(feature_ids)}")
    print(f"Found flux sites:    {len(flux_sites)}")
    print(f"Cubes with both:     {len(candidate_ids)}")

    if missing_feature:
        print(f"\n⚠️ Cubes missing feature file: {sorted(missing_feature)}")
    if missing_flux:
        print(f"⚠️ Cubes with no flux files:  {sorted(missing_flux)}")

    print("\n# -------- TRAIN (28 or fewer if limited) --------")
    print(f"train_cubes = {train_ids}")

    print("\n# -------------- VALIDATION --------------------")
    print(f"val_cubes   = {val_ids}")

    # 7) Year coverage (≥ MIN_YEAR) & valid timestamp counts
    print("\n# ============== YEAR COVERAGE ≥ {0} & VALID TIMESTAMPS ==============".format(MIN_YEAR))
    for cid in sorted(candidate_ids):
        site, _coords = sites_dict[cid]
        yrs = sorted(site_years.get(site, set()))
        zarr_path = FEATURE_DIR / f"s1_s2_{cid}_t.zarr"

        if not yrs:
            print(f"CID {cid} | Site {site}: no flux years ≥ {MIN_YEAR} → skipping year-filtered count.")
            # Optional: show all-years valid count (commented out by default)
            # try:
            #     all_valid, all_in, all_total = count_valid_timestamps_for_years(zarr_path, set(range(MIN_YEAR, 3000)))
            #     print(f"  (≥{MIN_YEAR} only) valid_timestamps={all_valid} / {all_in}  (total_all={all_total})")
            # except Exception as e:
            #     print(f"  Error opening {zarr_path}: {e}")
            continue

        try:
            valid_cnt, total_in_years, total_all = count_valid_timestamps_for_years(zarr_path, set(yrs))
            print(f"CID {cid} | Site {site} | years={yrs}: "
                  f"valid_timestamps_in_years={valid_cnt} / {total_in_years}  (total_all={total_all})")
        except Exception as e:
            print(f"CID {cid} | Site {site} | years={yrs}: ERROR → {e}")
