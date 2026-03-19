import pickle
import spyndex
import numpy as np
import xarray as xr
from typing import Dict, Union, List


def prepare_spectral_data(da: xr.DataArray, to_ds = True) -> xr.DataArray:
    """
    Prepares an input DataArray with shape (band, time, y, x) by computing selected spectral indices
    and stacking the result into a single DataArray with shape (index, time, y, x).

    Args:
        da (xr.DataArray): Input cube with dimensions (band, time, y, x)
        to_ds (bool): Whether to return an xarray Dataset (with variable names) or a DataArray with 'index' dimension.
    Returns:
        xr.DataArray: Output cube with dimensions (index, time, y, x)
    """
    da = da.clip(min=0, max=1)
    da = da.sel(band=[b for b in da.band.values if b not in ["B01", "B09"]])
    bands = da['band'].values


    # === Step 1: Create a Dataset with bands as variables ===
    all_bands = xr.Dataset(
        {band: da.sel(band=band).drop_vars('band') for band in bands},
        coords={dim: da.coords[dim] for dim in ['time', 'y', 'x']}
    )


    # === Step 2: Merge bands and indices ===
    # index_result has dimensions (index, time, y, x), we now add bands
    bands_da = xr.concat([all_bands[var] for var in all_bands.data_vars], dim="index")
    full_stack = bands_da.assign_coords(index=("index", list(all_bands.data_vars.keys())))
    if to_ds:
        index_values = full_stack.index.values
        data_vars = {str(idx): full_stack.sel(index=idx).drop_vars('index') for idx in index_values}
        full_stack = xr.Dataset(data_vars)  # [['EVI']]
        full_stack = full_stack.map(lambda da: da.clip(0, 1))
    else:
        full_stack = full_stack.clip(0, 1)

    return full_stack  # shape: (index, time, y, x)
