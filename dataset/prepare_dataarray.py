import pickle
import spyndex
import numpy as np
import xarray as xr
from typing import Dict, Union, List


def normalize(ds: Union[xr.Dataset, Dict[str, np.ndarray]], range_dict: Dict[str, List[float]], filter_var: str = None) -> Union[xr.Dataset, Dict[str, np.ndarray]]:
    """
    Normalize all variables in the dataset or dictionary using the provided range dictionary.

    Args:
        ds (Union[xr.Dataset, Dict[str, np.ndarray]]): The xarray dataset  or dictionary to normalize.
        range_dict (Dict[str, List[float]]): Dictionary with min and max values for each variable.
        filter_var (str): Variable name to exclude from normalization, such as a mask variable (e.g., 'land_mask').

    Returns:
        Union[xr.Dataset, Dict[str, np.ndarray]]: The normalized dataset or dictionary.
    """
    normalized_ds = ds.copy()
    data_vars = normalized_ds.data_vars if isinstance(normalized_ds, xr.Dataset) else normalized_ds.keys()
    for var in data_vars:
        if var == 'split' or var == filter_var: continue
        if var in range_dict:
            xmin, xmax = range_dict[var]
            if xmax != xmin:
                normalized_data = (ds[var] - xmin) / (xmax - xmin)
                if isinstance(normalized_ds, xr.Dataset):
                    normalized_ds[var] = normalized_data
                else:
                    normalized_ds[var] = normalized_data
            else:
                normalized_ds[var] = ds[var]  # If xmin == xmax, normalization isn't possible
    return normalized_ds


def normalize_dataarray(da: xr.DataArray, range_dict: Dict[str, List[float]]) -> xr.DataArray:
    """
    Normalize a DataArray with 'index' dimension using per-variable min-max ranges.

    Args:
        da (xr.DataArray): DataArray with shape (index, time, y, x)
        range_dict (Dict[str, List[float]]): Dictionary with min/max values per index label

    Returns:
        xr.DataArray: Normalized DataArray
    """
    index_names = da.coords["index"].values
    normalized_slices = []

    for i, var_name in enumerate(index_names):
        if var_name in range_dict:
            xmin, xmax = range_dict[var_name]
            slice_i = da.isel(index=i)
            if xmax != xmin:
                normalized = (slice_i - xmin) / (xmax - xmin)
            else:
                normalized = slice_i
        else:
            normalized = da.isel(index=i)  # untouched if no range

        normalized_slices.append(normalized.expand_dims(index=[var_name]))

    return xr.concat(normalized_slices, dim="index")




def prepare_spectral_data(da: xr.DataArray, min_max_dict=None, to_ds = True, load_b01b09=False) -> xr.DataArray:
    """
    Prepares an input DataArray with shape (band, time, y, x) by computing selected spectral indices
    and stacking the result into a single DataArray with shape (index, time, y, x).

    Args:
        da (xr.DataArray): Input cube with dimensions (band, time, y, x)
        min_max_dict (dict): Dictionary of min/max values for normalization

    Returns:
        xr.DataArray: Output cube with dimensions (index, time, y, x)
    """
    da = da.clip(min=0, max=1)
    if not load_b01b09:
        da = da.sel(band=[b for b in da.band.values if b not in ["B01", "B09"]])
    bands = da['band'].values

    if min_max_dict is None:
        try:
            with open("../all_ranges_no_clouds.pkl", "rb") as f:
                min_max_dict = pickle.load(f)
        except:
            with open("./all_ranges_no_clouds.pkl", "rb") as f:
                min_max_dict = pickle.load(f)

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
        full_stack = normalize(full_stack, min_max_dict)

        full_stack = full_stack.map(lambda da: da.clip(0, 1))
    else:
        full_stack = normalize_dataarray(full_stack, min_max_dict)
        full_stack = full_stack.clip(0, 1)

    return full_stack  # shape: (index, time, y, x)
