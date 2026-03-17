#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# --- CONFIG ------------------------------------------------------------------
ROOT_DIR = Path("/net/data/Fluxnet/")

DIR_PATTERNS  = ["ICOS_202*_I", "FLUXNET2020-ICOS-WarmWinter"]
FILE_PATTERNS = [
    "ICOSETC_*_FLUXNET_DD_01.csv",
    "FLX_*_FLUXNET2015_FULLSET_DD_*_beta-3.csv"
]

GPP_COL   = "GPP_NT_VUT_REF"
QC_COL    = "NEE_VUT_REF_QC"
QC_THRESH = 70.0  # % threshold
NA_VALUES = ["-9999", "-9999.0", "NA", "NaN", ""]

# --- RESULTS COLLECTOR ------------------------------------------------------
site_stats = []
all_valid_gpp = []

# --- FIND ALL MATCHING FILES ------------------------------------------------
all_csv_files = []

for dir_pattern in DIR_PATTERNS:
    for year_dir in sorted(ROOT_DIR.glob(dir_pattern)):
        for file_pattern in FILE_PATTERNS:
            year_files = list(year_dir.glob(file_pattern))
            all_csv_files.extend(year_files)

print(f"📂 Found {len(all_csv_files)} daily FLUXNET files across:")
for d in sorted(set(p.parent for p in all_csv_files)):
    print(f"   - {d}")

# --- PROCESS EACH FILE ------------------------------------------------------
for file in tqdm(all_csv_files, desc="Processing site files"):
    try:
        df = pd.read_csv(file, na_values=NA_VALUES, low_memory=False)

        if GPP_COL not in df.columns or QC_COL not in df.columns:
            continue

        # Normalize QC to 0–100% if needed
        qc = df[QC_COL].astype(float)
        if qc.max(skipna=True) <= 1.1:
            qc = qc * 100.0
        df["QC_pct"] = qc

        # Filter valid data
        df_valid = df[df["QC_pct"] >= QC_THRESH]
        gpp = df_valid[GPP_COL].astype(float).dropna()

        if gpp.empty:
            continue

        site_id = file.stem.split("_")[1]  # e.g., BE-Maa
        year = file.parent.name.replace("ICOS_", "").replace("_I", "")  # e.g., 2021

        site_stats.append({
            "site": site_id,
            "year": year,
            "n_days": len(gpp),
            "mean_gpp": gpp.mean(),
            "std_gpp": gpp.std(),
            "min_gpp": gpp.min(),
            "max_gpp": gpp.max(),
        })

        all_valid_gpp.append(gpp)

    except Exception as e:
        print(f"⚠️ Error in {file.name}: {e}")
        continue

# --- SITE-WISE SUMMARY ------------------------------------------------------
df_sites = pd.DataFrame(site_stats).sort_values(["site", "year"])

print("\n📊 Site-level GPP summary (after QC ≥ 70%):")
if df_sites.empty:
    print("No valid data found.")
else:
    print(df_sites.round(2).to_string(index=False))

# --- GLOBAL SUMMARY ---------------------------------------------------------
if all_valid_gpp:
    all_valid_gpp = pd.concat(all_valid_gpp, ignore_index=True)
    print("\n🌍 Global GPP Statistics across all valid data points:")
    print(f"  Valid days total: {all_valid_gpp.size:,}")
    print(f"  Global mean GPP:  {all_valid_gpp.mean():.3f} µmol CO₂ m⁻² s⁻¹")
    print(f"  Global std  GPP:  {all_valid_gpp.std():.3f}")
    print(f"  Global min  GPP:  {all_valid_gpp.min():.3f}")
    print(f"  Global max  GPP:  {all_valid_gpp.max():.3f}")
else:
    print("\n❌ No valid GPP values found above the QC threshold.")
