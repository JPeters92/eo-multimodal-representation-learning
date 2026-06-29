import os
import csv
import time
import math
import itertools
import random
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor, TQDMProgressBar
)
from lightning.pytorch.loggers import CSVLogger
from config import FEATURE_SET_TAG, INCLUDE_STD_FEATURES

# ---- Your modules ----
from GPP_loader import make_loaders
from model import GPPTemporalTransformer
SEED = 42
DEVICE_ID = 2                 # cuda device index, e.g. 0/1/2/3
MAX_EPOCHS = 150
MAX_TRIALS = 180
# ------------------------------------------------------------------
# Reproducibility + performance
# ------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
torch.set_float32_matmul_precision('high')  # RTX A6000 Tensor Cores optimization


BASE = "/net/data_ssd/deepfeatures/sciencecubes_processed"

WINDOW = 90
OVERLAP = 80
STRIDE = WINDOW - OVERLAP
QC_THRESH = 70.0
TRAIN_YEARS = "2017-2019"
VAL_YEARS = "2020"
FEATURE_SOURCE = "linear"
RADIATION_MODE = "noRad"

FEATURE_SOURCE_TAGS = {
    "linear": "linear",
    "ucm_flux": "ucm_flux",
}
RADIATION_MODE_TAGS = {
    "include": "rad",
    "exclude": "noRad",
    "rad": "rad",
    "noRad": "noRad",
}


def build_dataset_paths(base: str):
    if FEATURE_SOURCE not in FEATURE_SOURCE_TAGS:
        raise ValueError(f"Unknown FEATURE_SOURCE: {FEATURE_SOURCE}")
    if RADIATION_MODE not in RADIATION_MODE_TAGS:
        raise ValueError(f"Unknown RADIATION_MODE: {RADIATION_MODE}")

    source_tag = FEATURE_SOURCE_TAGS[FEATURE_SOURCE]
    rad_tag = RADIATION_MODE_TAGS[RADIATION_MODE]

    train_npz = (
        f"{base}/gpp_{WINDOW}day_samples_stride{STRIDE}_overlap{OVERLAP}_qc{QC_THRESH}"
        f"_years{TRAIN_YEARS}_{source_tag}_{FEATURE_SET_TAG}_{rad_tag}_gppstd_train.npz"
    )
    train_meta = (
        f"{base}/gpp_{WINDOW}day_samples_meta_stride{STRIDE}_overlap{OVERLAP}_qc{QC_THRESH}"
        f"_years{TRAIN_YEARS}_{source_tag}_{FEATURE_SET_TAG}_{rad_tag}_train.csv"
    )
    val_npz = (
        f"{base}/gpp_{WINDOW}day_samples_stride{STRIDE}_overlap{OVERLAP}_qc{QC_THRESH}"
        f"_years{VAL_YEARS}_{source_tag}_{FEATURE_SET_TAG}_{rad_tag}_gppstd_val.npz"
    )
    val_meta = (
        f"{base}/gpp_{WINDOW}day_samples_meta_stride{STRIDE}_overlap{OVERLAP}_qc{QC_THRESH}"
        f"_years{VAL_YEARS}_{source_tag}_{FEATURE_SET_TAG}_{rad_tag}_val.csv"
    )
    return train_npz, train_meta, val_npz, val_meta


def infer_num_features(npz_path: str) -> int:
    with np.load(npz_path, mmap_mode="r") as data:
        x = data["X"]
        if x.ndim != 3:
            raise ValueError(f"Expected X to have 3 dims in {npz_path}, got shape {x.shape}")
        return int(x.shape[1])


TRAIN_NPZ, TRAIN_META, VAL_NPZ, VAL_META = build_dataset_paths(BASE)
BASE_NUM_FEATURES = infer_num_features(TRAIN_NPZ)
RUN_TAG = f"{FEATURE_SOURCE}_{FEATURE_SET_TAG}_{RADIATION_MODE}_{TRAIN_YEARS}_to_{VAL_YEARS}"
RESULTS_CSV = f"grid_results_{RUN_TAG}.csv"
LOG_DIR = f"grid_logs_2_{RUN_TAG}"


SPACE  = {
        "num_features":  [BASE_NUM_FEATURES],
        #"batch_size":    [6, 12],          # <- you asked for these
        "batch_size":    [12],          # <- you asked for these
        "d_model":       [96],
        "nhead":         [8],          # filtered to divide d_model
        #"nhead":         [4, 8, 16],          # filtered to divide d_model
        "num_layers":    [3, 4],
        "dim_ff":        [1024],
        "dropout":       [0.05],
        "pool":          ["last",],
        "lr":            [1e-4],
        "weight_decay":  [1e-6],
        #"warmup_steps":  [200, 300, 400],
        "warmup_steps":  [300],
        "reduce_pat":    [7],
        "reduce_factor": [0.1],
    }

# → status: ok, best val: 0.494917, ckpt: grid_logs/bs6_dm128_h4_L3_ff1024_do0p05_last_lr0p0001_wd1e-06_wu400_rp7_rf0p1/checkpoints/best-epoch=63-val_loss=0.4949.ckpt


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_search_space():
    keys = list(SPACE.keys())
    all_combos = (dict(zip(keys, vals)) for vals in itertools.product(*(SPACE[k] for k in keys)))
    valid = [c for c in all_combos if (c["d_model"] % c["nhead"] == 0)]
    return valid


def run_once(params, max_epochs=MAX_EPOCHS, log_dir=LOG_DIR):
    """Train one config; return metrics dict."""
    set_seed(SEED)

    loaders = make_loaders(
        TRAIN_NPZ, TRAIN_META, VAL_NPZ, VAL_META,
        batch_size=params["batch_size"],
        time_first=True, num_workers=8, pin_memory=True,
        #feature_slice = list(range(6))
    )

    train_batch = next(iter(loaders["train"]))
    X0, y0 = train_batch[:2]
    print(f"[data] TRAIN_NPZ={TRAIN_NPZ}")
    print(f"[data] VAL_NPZ={VAL_NPZ}")
    print(f"[data] first train batch X={tuple(X0.shape)} y={tuple(y0.shape)}")


    model = GPPTemporalTransformer(
        num_features=params["num_features"],
        seq_len=90,
        d_model=params["d_model"],
        nhead=params["nhead"],
        num_layers=params["num_layers"],
        dim_feedforward=params["dim_ff"],
        dropout=params["dropout"],
        pool=params["pool"],
        learning_rate=params["lr"],
        weight_decay=params["weight_decay"],
        warmup_steps=params["warmup_steps"],
        reduce_patience=params["reduce_pat"],
        reduce_factor=params["reduce_factor"],
    )
    model.criterion = nn.L1Loss()  # MAE

    run_name = (
        f"bs{params['batch_size']}_dm{params['d_model']}_h{params['nhead']}"
        f"_L{params['num_layers']}_ff{params['dim_ff']}_do{params['dropout']}"
        f"_{params['pool']}_lr{params['lr']}_wd{params['weight_decay']}"
        f"_wu{params['warmup_steps']}_rp{params['reduce_pat']}_rf{params['reduce_factor']}"
    ).replace(".", "p")

    logger = CSVLogger(save_dir=str(log_dir), name=run_name, version="")
    ckpt_cb = ModelCheckpoint(
        monitor="val_loss", mode="min", save_top_k=1,
        filename="best-{epoch:02d}-{val_loss:.4f}",
    )
    callbacks = [
        ckpt_cb,
        EarlyStopping(monitor="val_loss", mode="min", patience=16, verbose=True),
        LearningRateMonitor(logging_interval="step"),
        TQDMProgressBar(refresh_rate=10),
    ]

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[DEVICE_ID],
        precision="16-mixed",
        max_epochs=max_epochs,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        logger=logger,
    )

    t0 = time.time()
    status = "ok"
    best_val = math.inf
    best_path = ""
    try:
        trainer.fit(model, train_dataloaders=loaders["train"], val_dataloaders=loaders["val"])
        best_val = float(ckpt_cb.best_model_score.cpu().item()) if ckpt_cb.best_model_score is not None else math.inf
        best_path = ckpt_cb.best_model_path or ""
    except RuntimeError as e:
        traceback.print_exc()
        if "out of memory" in str(e).lower():
            status = "oom"
        else:
            status = f"runtime_error: {e}"
    except Exception as e:
        traceback.print_exc()
        status = f"error: {e}"
    wall = time.time() - t0

    return {
        "status": status,
        "best_val_loss": best_val,
        "best_ckpt": best_path,
        "run_name": run_name,
        "wall_time_sec": round(wall, 2),
        **params,
    }


def main():
    set_seed(SEED)

    # Build + (optionally) subsample search space
    space = build_search_space()
    random.shuffle(space)
    if MAX_TRIALS > 0:
        space = space[:MAX_TRIALS]

    results_path = Path(RESULTS_CSV)
    write_header = not results_path.exists()
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames = [
    "status", "best_val_loss", "best_ckpt", "run_name", "wall_time_sec",
    "num_features",            # <-- add this
    "batch_size", "d_model", "nhead", "num_layers", "dim_ff", "dropout",
    "pool", "lr", "weight_decay", "warmup_steps", "reduce_pat", "reduce_factor",
])
        if write_header:
            writer.writeheader()

        total = len(space)
        for i, params in enumerate(space, 1):
            print(f"\n=== Trial {i}/{total} ===")
            print(params)
            metrics = run_once(params)
            writer.writerow(metrics)
            f.flush()
            print(f"→ status: {metrics['status']}, best val: {metrics['best_val_loss']:.6f}, ckpt: {metrics['best_ckpt']}")

    print(f"\nDone. Results saved to {results_path.resolve()}")


if __name__ == "__main__":
    main()
