import re
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, List, Set
import torch
import torch.nn as nn

# ------------------------
# Project imports
# ------------------------
from model import GPPTemporalTransformer
from sites import sites_dict

# ------------------------
# User config
# ------------------------
#CKPT_PATH = "/scratch/jpeters/DeepFeatures/use_cases/lightning_logs/version_109/checkpoints/gpp-tx-epoch=33-val_loss=0.5008.ckpt"
CKPT_PATH = "/scratch/jpeters/DeepFeatures/use_cases/grid_logs/bs6_dm128_h4_L3_ff1024_do0p05_last_lr0p0001_wd1e-06_wu200_rp7_rf0p25/checkpoints/best-epoch=53-val_loss=0.4920.ckpt"
CKPT_PATH = "/scratch/jpeters/DeepFeatures/use_cases/grid_logs2/bs6_dm128_h4_L3_ff1024_do0p05_last_lr0p0001_wd1e-06_wu400_rp7_rf0p2/checkpoints/best-epoch=10-val_loss=0.4825.ckpt"
CKPT_PATH = "/scratch/jpeters/DeepFeatures/use_cases/grid_logs2/bs6_dm128_h4_L4_ff1024_do0p05_last_lr0p0001_wd1e-06_wu200_rp7_rf0p2/checkpoints/best-epoch=03-val_loss=0.4828.ckpt"
CKPT_PATH = "/scratch/jpeters/DeepFeatures/use_cases/grid_logs_7_2/bs12_dm96_h4_L3_ff1024_do0p05_last_lr0p0001_wd1e-06_wu400_rp7_rf0p1/checkpoints/best-epoch=34-val_loss=0.5754.ckpt"
#CKPT_PATH = "/scratch/jpeters/DeepFeatures/use_cases/grid_logs8/bs6_dm128_h4_L3_ff1024_do0p05_last_lr0p0001_wd1e-06_wu400_rp7_rf0p2/checkpoints/best-epoch=10-val_loss=0.4825-v1.ckpt"

ROOT_DIR = Path("/net/data/Fluxnet/")
IN_DIR   = Path("/net/data_ssd/deepfeatures/sciencecubes_processed")
OUT_DIR  = Path("./gpp_recon_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CUBE_IDS = ["028", "021", "003"]
VAR_NAME = "feature_mean_ucm"

WINDOW   = 90
OVERLAP  = 89
assert 0 <= OVERLAP < WINDOW, "OVERLAP must be in [0, WINDOW)"
STRIDE = WINDOW - OVERLAP

# --- radiation handling for validation ---
#   "include" -> keep radiation feature and standardize it
#   "exclude" -> drop radiation feature entirely before windowing
RADIATION_MODE = "exclude"   # change to "exclude" for without-radiation validation

# --- constants (must match training) ---
RAD_MEAN = 28.8545
RAD_STD  = 6.8393
GPP_MEAN = 4.042
GPP_STD  = 4.386

QC_THRESH = 70.0  # set 0.0 to accept all QC

# ------------------------
# Helpers
# ------------------------
def _site_in_filename(site: str, name: str) -> bool:
    return site in name

def detect_flux_years_for_site(site: str, root: Path) -> Set[int]:
    years: Set[int] = set()
    ww_dir = root / "FLUXNET2020-ICOS-WarmWinter"  # 2017–2020 coverage
    if ww_dir.exists():
        for p in ww_dir.glob("FLX_*_FLUXNET2015_FULLSET_DD_*_beta-3.csv"):
            if _site_in_filename(site, p.name):
                years.update({2017, 2018, 2019, 2020})
                break
    for dpat in ["ICOS_2021_I", "ICOS_2022_I", "ICOS_2023_I", "ICOS_2024_I"]:
        d = root / dpat
        if not d.exists():
            continue
        try:
            y = int(dpat.split("_")[1])
        except Exception:
            continue
        for p in d.glob("ICOSETC_*_FLUXNET_DD_01.csv"):
            if _site_in_filename(site, p.name):
                years.add(y)
                break
    return {y for y in years if 2017 <= y <= 2024}

def _safe(da: xr.DataArray) -> xr.DataArray:
    return xr.where(np.isfinite(da), da, np.nan)

def _open_cube_da(cid: str) -> Tuple[xr.DataArray, Optional[int]]:
    """
    Open (feature,time) DataArray + radiation index attribute.
    """
    z = IN_DIR / f"s1_s2_{cid}_v_mean_ucm_flux.zarr"
    if not z.exists():
        raise FileNotFoundError(z)
    ds = xr.open_zarr(z, consolidated=True)
    if VAR_NAME not in ds:
        raise KeyError(f"{VAR_NAME} not in {z}")
    da = _safe(ds[VAR_NAME]).sortby("time")
    ridx = ds[VAR_NAME].attrs.get("radiation_feature_index", None)
    return da, int(ridx) if ridx is not None else None

def _parse_date_col(df: pd.DataFrame) -> pd.Series:
    candidates = [c for c in df.columns if c.upper().startswith("TIMESTAMP")]
    if not candidates:
        raise ValueError("No TIMESTAMP column")
    col = candidates[0]
    s = df[col].astype(str)
    m8 = s.str.match(r"^\d{8}$")
    if m8.any():
        s.loc[m8] = pd.to_datetime(s[m8], format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d")
    m12 = s.str.match(r"^\d{12}$")
    if m12.any():
        s.loc[m12] = pd.to_datetime(s[m12], format="%Y%m%d%H%M", errors="coerce").dt.strftime("%Y-%m-%d")
    dt = pd.to_datetime(s, errors="coerce", utc=False)
    return dt.dt.normalize()

def _choose_gpp_column(cols: List[str]) -> Optional[str]:
    pref = [c for c in cols if c.upper() == "GPP_NT_VUT_REF"]
    if pref: return pref[0]
    alt = [c for c in cols if re.match(r"(?i)^GPP($|[_])", c)]
    return alt[0] if alt else None

def _load_fluxnet_daily_gpp(site: str, qc_thresh: float = QC_THRESH) -> pd.Series:
    files: List[Path] = []
    ww_dir = ROOT_DIR / "FLUXNET2020-ICOS-WarmWinter"
    if ww_dir.exists():
        files += list(ww_dir.glob(f"FLX_*{site}*_FLUXNET2015_FULLSET_DD_*_beta-3.csv"))
    for year in range(2021, 2025):
        d = ROOT_DIR / f"ICOS_{year}_I"
        if d.exists():
            files += list(d.glob(f"ICOSETC_*{site}*_FLUXNET_DD_01.csv"))
    if not files:
        raise FileNotFoundError(f"No FLUXNET DD files for site {site}")

    def _priority(p: Path) -> tuple:
        if "ICOSETC_" in p.name:
            m = re.search(r"ICOS_(\d{4})_I", str(p.parent))
            yr = int(m.group(1)) if m else 0
            return (0, -yr)
        return (1, 0)

    files.sort(key=_priority)
    parts: List[pd.Series] = []

    for f in files:
        df = pd.read_csv(f, low_memory=False)
        dt = _parse_date_col(df)
        gcol = _choose_gpp_column(df.columns.tolist())
        if not gcol or gcol not in df.columns:
            continue
        qc_candidates = [c for c in df.columns if "QC" in c.upper()]
        if not qc_candidates:
            continue
        qc_col = qc_candidates[0]

        gpp = pd.Series(pd.to_numeric(df[gcol], errors="coerce").values, index=dt)
        qc  = pd.Series(pd.to_numeric(df[qc_col], errors="coerce").values, index=dt)

        if qc.max(skipna=True) <= 1.1:
            qc = qc * 100.0

        valid_mask = qc >= qc_thresh
        valid_days = gpp[valid_mask & gpp.notna() & np.isfinite(gpp)]
        if not valid_days.empty:
            parts.append(valid_days)

    if not parts:
        raise ValueError(f"No valid GPP data (QC ≥ {qc_thresh}%) for {site}")

    df_stack = pd.concat(parts, axis=1)
    gpp = df_stack.bfill(axis=1).iloc[:, 0]
    gpp = gpp.sort_index()
    gpp.name = "GPP"
    return gpp

def _standardize_radiation(da_ft: xr.DataArray, ridx: Optional[int]) -> xr.DataArray:
    if ridx is None:
        return da_ft
    da = da_ft.copy()
    rad = da.isel(feature=ridx)
    da.loc[dict(feature=ridx)] = (rad - RAD_MEAN) / (RAD_STD or 1.0)
    return da

def _apply_radiation_mode(da_ft: xr.DataArray, ridx: Optional[int], mode: str) -> xr.DataArray:
    mode = mode.lower()
    if mode not in {"include", "exclude"}:
        raise ValueError(f"Invalid RADIATION_MODE: {mode}")
    if ridx is None:
        return da_ft  # nothing to do if we don't know the index

    if mode == "include":
        return _standardize_radiation(da_ft, ridx)

    # exclude: drop radiation feature
    try:
        return da_ft.drop_isel(feature=[ridx])
    except Exception:
        sel = np.ones(da_ft.sizes["feature"], dtype=bool)
        sel[ridx] = False
        return da_ft.isel(feature=np.where(sel)[0])

def _trim_to_flux_years(da_ft: xr.DataArray, years: List[int]) -> xr.DataArray:
    if not years:
        return da_ft.isel(time=slice(0, 0))
    mask = np.isin(pd.to_datetime(da_ft.time.values).year, years)
    if not mask.any():
        return da_ft.isel(time=slice(0, 0))
    return da_ft.isel(time=np.where(mask)[0])

def _make_windows(da_ft: xr.DataArray,
                  gpp: pd.Series,
                  cid: str,
                  site: str) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    times = pd.to_datetime(da_ft["time"].values).normalize()
    F = da_ft.sizes["feature"]
    Xs, ys, ts = [], [], []

    for end_idx in range(WINDOW - 1, len(times), STRIDE):
        start_idx = end_idx - (WINDOW - 1)
        win_times = times[start_idx:end_idx + 1]
        end_day = win_times[-1]

        tgt = gpp.get(end_day, np.nan)
        if pd.isna(tgt):
            continue

        tgt_std = (tgt - GPP_MEAN) / (GPP_STD or 1.0)
        win = da_ft.isel(time=slice(start_idx, end_idx + 1))
        arr = np.asarray(win.values, np.float32)  # (F, T)

        if not np.isfinite(arr).all():
            continue

        arr_tf = np.transpose(arr, (1, 0))  # (T, F)
        Xs.append(arr_tf)
        ys.append(float(tgt_std))
        ts.append(pd.Timestamp(end_day))

    if not Xs:
        return (np.empty((0, WINDOW, F), np.float32),
                np.empty((0,), np.float32),
                [])
    X = np.stack(Xs, axis=0)  # (N, T, F)
    y = np.array(ys, np.float32)
    return X, y, ts

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))

def nrmse(a: np.ndarray, b: np.ndarray) -> float:
    r = rmse(a, b)
    denom = float(np.max(b) - np.min(b)) if len(b) > 0 else np.nan
    return r / denom if denom > 0 else np.nan

# ------------------------
# Model loading + alignment
# ------------------------
def _infer_model_feature_count(model: nn.Module) -> int:
    # Try hparams first
    nf = getattr(getattr(model, "hparams", object()), "num_features", None)
    if isinstance(nf, (int, np.integer)):
        return int(nf)
    # Fallback: infer from first linear layer
    lin = getattr(model, "input_proj", None)
    if isinstance(lin, nn.Linear):
        return int(lin.in_features)
    raise RuntimeError("Cannot infer model's expected num_features; ensure model.hparams.num_features is saved.")

def load_model(ckpt_path: str, device: torch.device) -> Tuple[GPPTemporalTransformer, int]:
    model = GPPTemporalTransformer.load_from_checkpoint(ckpt_path, map_location=device)
    model.eval().to(device)
    f_model = _infer_model_feature_count(model)
    print(f"[model] expects num_features = {f_model}")
    # sanity: time_first
    time_first = getattr(model.hparams, "time_first", True)
    assert time_first, "Model was trained with time_first=True; expected (B,T,F) inputs."
    return model, f_model

def align_X_to_model(X: np.ndarray, f_model: int, ridx: Optional[int], mode: str) -> np.ndarray:
    """
    Ensure X.shape[-1] == f_model.
    - If X has 8 and model wants 7, drop radiation if ridx set (or last feature as fallback).
    - If X has 7 and model wants 8, insert a zero radiation column at ridx (standardized 0).
    """
    f_data = X.shape[-1]
    if f_data == f_model:
        return X

    if f_data == f_model + 1:
        # Need to drop one feature (prefer radiation)
        drop_idx = ridx if ridx is not None else (f_data - 1)
        keep = [i for i in range(f_data) if i != drop_idx]
        X2 = X[..., keep]
        print(f"[align] dropped feature idx {drop_idx} → {f_data}→{X2.shape[-1]}")
        return X2

    if f_data + 1 == f_model:
        # Need to add one feature (insert standardized 0 radiation at ridx or append)
        insert_idx = ridx if ridx is not None else f_data
        zeros = np.zeros((*X.shape[:-1], 1), dtype=X.dtype)  # standardized rad ~0
        X2 = np.concatenate([X[..., :insert_idx], zeros, X[..., insert_idx:]], axis=-1)
        print(f"[align] inserted zero feature at idx {insert_idx} → {f_data}→{X2.shape[-1]}")
        return X2

    raise ValueError(f"Cannot align features: data has {f_data}, model expects {f_model}.")

# ------------------------
# Plotting
# ------------------------
def plot_reconstruction(dates: List[pd.Timestamp],
                        y_true: np.ndarray,
                        y_pred: np.ndarray,
                        cid: str,
                        out_dir: Path):
    rmse_val = rmse(y_pred, y_true)
    nrmse_val = nrmse(y_pred, y_true)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, y_true, label="True GPP (daily, window end)", linewidth=1.5)
    ax.plot(dates, y_pred, label="Predicted GPP", linewidth=1.5)
    ax.set_title(f"Cube {cid} — RMSE={rmse_val:.3f}, NRMSE={nrmse_val:.3f}")
    ax.set_xlabel("Date"); ax.set_ylabel("GPP")
    ax.legend(); ax.grid(True, linewidth=0.5)

    out_png = out_dir / f"gpp_reconstruction_cube_{cid}_mode-{RADIATION_MODE}.png"
    fig.tight_layout(); fig.savefig(out_png, dpi=180); plt.close(fig)
    print(f"   📈 saved: {out_png}")

# ------------------------
# Main
# ------------------------
def main():
    import matplotlib
    matplotlib.use("Agg")

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, f_model = load_model(CKPT_PATH, device)

    for cid in CUBE_IDS:
        site = sites_dict.get(cid, [None])[0]
        if site is None:
            print(f"⚠️  Missing site mapping for cube {cid} — skip"); continue

        flux_years = sorted(detect_flux_years_for_site(site, ROOT_DIR))
        if not flux_years:
            print(f"→ {cid} ({site}): no flux years — skip"); continue

        try:
            da_ft, ridx = _open_cube_da(cid)
        except Exception as e:
            print(f"⚠️  Skip {cid}: {e}"); continue

        da_ft = _trim_to_flux_years(da_ft, flux_years)
        if da_ft.sizes.get("time", 0) < WINDOW:
            print(f"→ {cid} ({site}): too few days after trim — skip"); continue

        # apply radiation mode (include → standardize; exclude → drop)
        da_ft = _apply_radiation_mode(da_ft, ridx, RADIATION_MODE)

        # Load GPP
        try:
            gpp_series = _load_fluxnet_daily_gpp(site, qc_thresh=QC_THRESH)
        except Exception as e:
            print(f"⚠️  GPP load failed for {site}: {e}"); continue
        gpp_series = gpp_series[gpp_series.index.year.isin(flux_years)]

        # Build windows
        X, y_std, ts = _make_windows(da_ft, gpp_series, cid, site)
        if X.shape[0] == 0:
            print(f"→ {cid} ({site}): no valid windows — skip"); continue

        # Align last-dim to what the model expects (handles 7↔8 flexibly)
        X = align_X_to_model(X, f_model=f_model, ridx=ridx, mode=RADIATION_MODE)

        # Predict
        with torch.no_grad():
            Xt = torch.tensor(X, dtype=torch.float32, device=device)  # (N, T, F_model)
            y_hat_std = model(Xt).detach().cpu().numpy()

        # Unstandardize predictions and gather ground truth on same dates
        y_pred = y_hat_std * GPP_STD + GPP_MEAN
        y_true = np.array([gpp_series.get(pd.Timestamp(t).normalize(), np.nan) for t in ts], dtype=float)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        dates = [t for t, m in zip(ts, mask) if m]
        y_true = y_true[mask]; y_pred = y_pred[mask]

        if len(y_true) == 0:
            print(f"→ {cid} ({site}): no comparable points after masking — skip"); continue

        # Plot & export
        plot_reconstruction(dates, y_true, y_pred, cid, OUT_DIR)
        out_csv = OUT_DIR / f"gpp_reconstruction_cube_{cid}_mode-{RADIATION_MODE}.csv"
        pd.DataFrame({"date": pd.to_datetime(dates), "gpp_true": y_true, "gpp_pred": y_pred}).to_csv(out_csv, index=False)
        print(f"   💾 saved: {out_csv}")

if __name__ == "__main__":
    main()