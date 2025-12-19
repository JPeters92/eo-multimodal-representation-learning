# gpp_dataloaders.py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Sequence, Dict, Any

class GPPSlidingWindowDataset(Dataset):
    """
    Loads 90-day GPP window dataset from NPZ/CSV files.
    Each sample:
        X: (F, 90) standardized input features
        y: (1,) standardized GPP target
    """
    def __init__(
        self,
        npz_path: str,
        meta_csv: Optional[str] = None,
        time_first: bool = False,     # if True -> (T, F)
        dtype: torch.dtype = torch.float32,
        return_meta: bool = False,
        device: Optional[torch.device] = None,
        feature_slice: Optional[Sequence[int]] = None,  # e.g. [0,1,2,7]
    ):
        data = np.load(npz_path)
        self.X = data["X"]             # (N, F, 90)
        self.y = data["y"]             # (N,)
        self.meta = pd.read_csv(meta_csv) if meta_csv else None
        if feature_slice is not None:
            self.X = self.X[:, feature_slice, :]
        self.time_first = time_first
        self.dtype = dtype
        self.return_meta = return_meta
        self.device = device

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int):
        x = torch.tensor(self.X[idx], dtype=self.dtype)
        y = torch.tensor(self.y[idx], dtype=self.dtype)
        if self.time_first:
            x = x.transpose(0, 1)  # -> (90, F)
        if self.device is not None:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
        if self.return_meta and self.meta is not None:
            row = self.meta.iloc[int(idx)]
            m = {
                "cube_id": row.get("cube_id"),
                "site": row.get("site"),
                "end_date": row.get("end_date"),
                "index": int(idx),
            }
            return x, y, m
        return x, y


def _default_collate(batch):
    """Keeps metadata as list when present."""
    if len(batch[0]) == 3:
        xs, ys, metas = zip(*batch)
        return torch.stack(xs), torch.stack(ys), list(metas)
    else:
        xs, ys = zip(*batch)
        return torch.stack(xs), torch.stack(ys)


def make_loaders(
    train_npz: str,
    train_meta: str,
    val_npz: Optional[str] = None,
    val_meta: Optional[str] = None,
    batch_size: int = 64,
    time_first: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
    return_meta: bool = True,
    device: Optional[str] = None,
    feature_slice: Optional[Sequence[int]] = None,
) -> Dict[str, DataLoader]:
    """Creates PyTorch DataLoaders for train (+optional val)."""
    dev = torch.device(device) if device else None

    train_ds = GPPSlidingWindowDataset(
        train_npz, train_meta,
        time_first=time_first, return_meta=return_meta,
        device=dev, feature_slice=feature_slice,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=_default_collate, drop_last=False,
    )

    loaders = {"train": train_loader}

    if val_npz and val_meta:
        val_ds = GPPSlidingWindowDataset(
            val_npz, val_meta,
            time_first=time_first, return_meta=return_meta,
            device=dev, feature_slice=feature_slice,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
            collate_fn=_default_collate, drop_last=False,
        )
        loaders["val"] = val_loader

    return loaders




paths = "/net/data_ssd/deepfeatures/sciencecubes_processed"

train_npz  = f"{paths}/gpp_90day_samples_stride5_overlap85_radstd_gppstd_train.npz"
train_meta = f"{paths}/gpp_90day_samples_meta_stride5_overlap85_train.csv"
val_npz    = f"{paths}/gpp_90day_samples_stride5_overlap85_radstd_gppstd_val.npz"
val_meta   = f"{paths}/gpp_90day_samples_meta_stride5_overlap85_val.csv"

loaders = make_loaders(
    train_npz, train_meta,
    val_npz, val_meta,
    batch_size=128,
    time_first=True,   # (B, 90, F) for Transformer models
    num_workers=8,
    feature_slice=list(range(7))
)

#for batch in loaders["train"]:
#    X, y, meta = batch   # X: (B, 90, F), y: (B,)
#    print(X.shape, y.shape, len(meta))
#    break
#