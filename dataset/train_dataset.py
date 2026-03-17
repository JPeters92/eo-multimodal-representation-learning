import os
import h5py
import torch
import random
import numpy as np
import xarray as xr
from typing import Dict
from dataset.utils import compute_time_gaps
from dataset.prepare_dataarray import prepare_spectral_data
from dataset.preprocess_sentinel import match_sentinel1_to_s2_cube, extract_sentinel_patches, nearest_indices

def ensure_band(var: xr.DataArray) -> xr.DataArray:
    if "band" in var.dims:
        return var
    if "index" in var.dims:
        return var.rename({"index": "band"})
    raise ValueError(f"No band-like dim in {var.dims}")

def verify_patches_against_cube(
    da: xr.DataArray,
    patches: torch.Tensor,
    coords_out: Dict[str, np.ndarray],
    n_samples: int = 5
):
    """
    Verifies extracted patches against the original xarray cube.

    Args:
        da: xarray.DataArray with dims (band, time, y, x)
        patches: torch.Tensor of shape (N, bands, select_t, h, w)
        coords_out: dict with keys 'time', 'y', 'x'
        n_samples: number of random samples to verify
    """
    print(f"\n🔍 Verifying {n_samples} random patches...")

    N = patches.shape[0]
    sample_ids = np.random.choice(N, size=n_samples, replace=False)

    for idx in sample_ids:
        t_coords = coords_out["time"][idx]
        y_coords = coords_out["y"][idx]
        x_coords = coords_out["x"][idx]

        # Select original patch using coordinate values
        da_patch = da.sel(
            time=xr.DataArray(t_coords, dims="time"),
            y=xr.DataArray(y_coords, dims="y"),
            x=xr.DataArray(x_coords, dims="x")
        ).transpose("time", "index", "y", "x")

        print(da_patch.shape)

        da_patch_np = da_patch.values
        extracted_np = patches[idx]


        is_equal = np.allclose(da_patch_np, extracted_np, equal_nan=True)

        print(f"🧪 Patch {idx}: {'✅ MATCH' if is_equal else '❌ MISMATCH'}")

    print("🔁 Verification complete.\n")


def process_cube(cube_num, sentinel_set='s2'):

    cube_path = os.path.join(base_path, cube_num + '.zarr')
    print(cube_path)


    ds = xr.open_zarr(cube_path)

    if sentinel_set == 's1':
        s1 = match_sentinel1_to_s2_cube(ds)
        da_s1 = s1.backscatter#.transpose('band', 'time', 'y', 'x')

        s1_coords_in = {
            'time': da_s1.coords['time'].values,
            'y': da_s1.coords['y'].values,
            'x': da_s1.coords['x'].values,
        }

        s1_patches_t, s1_coords, s1_valid_mask_t, rm_unvalid = extract_sentinel_patches(
            da_s1.values, s1_coords_in['time'], s1_coords_in['y'], s1_coords_in['x'], time_win=40, time_stride=40, h_stride = 15, w_stride = 15, layout='TBYX'
        )

        if s1_patches_t.numel() == 0 or s1_patches_t.shape[0] == 0:
            return None, None, None, None

        return s1_patches_t.numpy(), s1_coords, s1_valid_mask_t.numpy(), s1


    da = ds.s2l2a.where((ds.cloud_mask == 0))


    n_total = da.sizes["band"] * da.sizes["y"] * da.sizes["x"]
    threshold = int(n_total * 0.03)

    # Count non-NaN points per time step
    valid_data_count = da.notnull().sum(dim=["band", "y", "x"])

    # Keep only time steps with at least 3.5% valid data
    da = da.sel(time=valid_data_count >= threshold)

    chunks = {"time": da.sizes["time"], "y": 90, "x": 90}
    da = da.chunk(chunks)

    da = prepare_spectral_data(da, to_ds=False)

    #da = da.sel(index="B02").expand_dims(index=1)
    coords = {dim: da.coords[dim].values for dim in da.dims if dim in ["time", "y", "x"]}

    if sentinel_set == 's1_s2':
        # Load S1 aligned to S2 spatial grid
        s1 = match_sentinel1_to_s2_cube(ds)  # dims assumed: backscatter: (time, band, y, x)
        s1_times = s1.time.values  # (T_s1,)
        # Use the exact S2 times that are used to make `patches`
        s2_times_used = xr.DataArray(coords['time'], dims=["time"])
        nearest_idx = nearest_indices(s2_times_used.values, s1_times)  # (T_s2,)

        # Lazily align S1 to S2 times: 1 S1 slice per S2 time (nearest)
        # (optional) add tolerance="3D" or similar if you want a max gap
        s1_matched = s1.backscatter.isel(time=xr.DataArray(nearest_idx, dims=["time"]))

        s1_matched = s1_matched.assign_coords(time=s2_times_used)

        # Reorder both to BTYX lazily (xarray transpose is lazy)
        s2_BTYX = ensure_band(da).reset_coords(drop=True).assign_attrs({})
        s1_BTYX = s1_matched.transpose("band", "time", "y", "x").reset_coords(drop=True).assign_attrs({})

        # Now concat safely along band
        combined_BTYX = xr.concat(
            [s2_BTYX, s1_BTYX],
            dim="band",
            coords="minimal",
            compat="override",
            join="outer",
        )

        # Also pass the matched S1 times so the extractor returns coords['time_add']
        matched_s1_times = s1_times[nearest_idx]  # (T_s2,)

        # Single extraction call for both modalities
        # If your extractor accepts numpy arrays only, it will trigger compute here.
        # To keep memory bounded, ensure your xarray chunks are reasonable for the windows.

        patches_all, coords_all, valid_mask_all, rm_unvalid = extract_sentinel_patches(
            combined_BTYX.values,  # dask array; extractor should handle array-like or call np.asarray internally
            s2_times_used.values,  # small vector -> fine to realize
            coords['y'],  # numpy or small vector
            coords['x'],
            time_coords_2 = matched_s1_times
            # pass your window/stride/layout args as needed; layout='BTYX' here
            # time_win=20, time_stride=17, h_win=15, w_win=15, h_stride=9, w_stride=9, layout='BTYX'
        )

        if patches_all.shape[0] == 0:  # no S2 samples left
            return None, None, None, None


        return patches_all, coords_all, valid_mask_all, s1


    patches, coords, valid_mask = extract_sentinel_patches(da.values, coords['time'], coords['y'], coords['x'])
    patches = patches.numpy()

    if patches.shape[0] == 0:  # no S2 samples left
        return None, None, None, None
    else:
        return patches, coords, valid_mask, da


def divide_mini_cubes(split = 0.75):
    # Set the seed for reproducibility
    random.seed(42)

    # Generate a list of numbers from 0 to 499
    numbers = list(range(500))

    # Calculate 80% of 500
    count_to_select = int(split * 500)

    # Randomly select 80% of the numbers
    selected_numbers = random.sample(numbers, count_to_select)

    # Find the remaining numbers
    remaining_numbers = [num for num in numbers if num not in selected_numbers]

    print(f"Selected Numbers ({int(split * 100)}%):", selected_numbers)
    print(f"Remaining Numbers ({int((1-split) * 100)}%):", remaining_numbers)
    return selected_numbers, remaining_numbers


# Call the method
selected_numbers, remaining_numbers = divide_mini_cubes()

# Randomly split remaining_numbers into 2/3 validation and 1/3 test
random.shuffle(remaining_numbers)  # Shuffle for randomness

val_count = int(2 / 3 * len(remaining_numbers))
val_numbers = remaining_numbers[:val_count]
test_numbers = remaining_numbers[val_count:]

print(f"Validation cubes ({len(val_numbers)}):", val_numbers)
print(f"Test cubes ({len(test_numbers)}):", test_numbers)
print(f"create train set ")

# Convert to 6-digit strings if needed
test_six_digit_strings = [f"{num:06d}" for num in test_numbers]
train_six_digit_strings = [f"{num:06d}" for num in selected_numbers]
val_six_digit_strings = [f"{num:06d}" for num in val_numbers]

# Iterate through the selected numbers and create 6-digit strings
six_digit_strings = val_six_digit_strings
six_digit_strings = [f"{num:06d}" for num in selected_numbers]

base_path = '/net/data_ssd/deepfeatures/trainingcubes'

current_train_size = 0
batch_size = 1
from time import time
sentinel_ds = 's1_s2'
dataset = 'test'


if dataset == 'validate':
    file_name = f"val_{sentinel_ds}.h5"
    six_digit_strings = val_six_digit_strings
elif dataset == 'test':
    file_name = f"test_{sentinel_ds}.h5"
    six_digit_strings = test_six_digit_strings
elif dataset == 'train':
    file_name = f"train_{sentinel_ds}.h5"
    six_digit_strings = train_six_digit_strings
else:
    file_name = 'file.h5'
    six_digit_strings = []

with h5py.File(file_name, "w") as train_file:
    # Initialize expandable datasets for train
    if sentinel_ds =='s1_s2': train_shape = (11, 12, 15, 15)  # Train data shape (e.g., (209, 11, 15, 15))
    elif sentinel_ds == 's1': train_shape = (11, 2, 15, 15)
    else: train_shape = (11, 10, 15, 15)
    mask_shape = train_shape  # Masks have the same shape as the data

    time_gap_shape = (10,)
    time_gap_shape_c = (1,)
    # Train datasets
    train_data_dset = train_file.create_dataset(
        "data", shape=(0, *train_shape), maxshape=(None, *train_shape), dtype='float32', chunks=True
    )
    train_mask_dset = train_file.create_dataset(
        "mask", shape=(0, *mask_shape), maxshape=(None, *mask_shape), dtype='bool', chunks=True
    )
    if sentinel_ds == 's1_s2':
        train_time_gaps_dset_s1 = train_file.create_dataset(
            "time_gaps_s1", shape=(0, *time_gap_shape), maxshape=(None, *time_gap_shape), dtype='int32', chunks=True
        )
        train_time_gaps_dset_s2 = train_file.create_dataset(
            "time_gaps_s2", shape=(0, *time_gap_shape), maxshape=(None, *time_gap_shape), dtype='int32', chunks=True
        )
        train_time_gaps_dset_c = train_file.create_dataset(
            "time_gaps_c", shape=(0, *time_gap_shape_c), maxshape=(None, *time_gap_shape), dtype='int32', chunks=True
        )
    else:
        train_time_gaps_dset = train_file.create_dataset(
            "time_gaps", shape=(0, *time_gap_shape), maxshape=(None, *time_gap_shape), dtype='int32', chunks=True
        )


    current_train_size = 0
    batch_size = 1
    for i in range(0, len(six_digit_strings), batch_size):
        batch = six_digit_strings[i:i + batch_size]
        print(f"Processing batch: {batch}")

        start = time()
        try:
            sentinel_patches, coords, sentinel_mask, da = process_cube(batch[0], sentinel_set=sentinel_ds)
        except: continue
        if sentinel_patches is None: continue
        print(f'time taken: {time() - start} to process cube {batch[0]}')
#
        #'#verify_patches_against_cube(da, sentinel_patches, coords)
#
        #'# Compute the size of the current batch
        batch_chunk_size = sentinel_patches.shape[0]
#
        if batch_chunk_size > 0:
            # Resize datasets to accommodate the new batch
            train_data_dset.resize(current_train_size + batch_chunk_size, axis=0)
            train_mask_dset.resize(current_train_size + batch_chunk_size, axis=0)
            # Write the batch data into the HDF5 datasets
            train_data_dset[current_train_size:current_train_size + batch_chunk_size] = sentinel_patches
            train_mask_dset[current_train_size:current_train_size + batch_chunk_size] = sentinel_mask
            time_gaps = compute_time_gaps(coords['time'])
#
            if sentinel_ds == 's1_s2':
                time_gaps_s1 = compute_time_gaps(coords['time_add'])
                time_gaps_c = np.abs(
                    (coords['time_add'][:, 5] - coords['time'][:, 5])
                    .astype('timedelta64[D]')
                ).astype(int).reshape(-1, 1)
#
                train_time_gaps_dset_s1.resize(current_train_size + batch_chunk_size, axis=0)
                train_time_gaps_dset_s2.resize(current_train_size + batch_chunk_size, axis=0)
                train_time_gaps_dset_c.resize(current_train_size + batch_chunk_size, axis=0)
                train_time_gaps_dset_s1[current_train_size:current_train_size + batch_chunk_size] = time_gaps_s1.numpy()
                train_time_gaps_dset_s2[current_train_size:current_train_size + batch_chunk_size] = time_gaps.numpy()
                train_time_gaps_dset_c[current_train_size:current_train_size + batch_chunk_size] = np.array([time_gaps_c], dtype=np.int32)
            else:
                train_time_gaps_dset.resize(current_train_size + batch_chunk_size, axis=0)
                train_time_gaps_dset[current_train_size:current_train_size + batch_chunk_size] = time_gaps.numpy()
#
            # Update the current train size
            current_train_size += batch_chunk_size
#
        print(f"Train batch of size {batch_chunk_size} written to HDF5 file.")

    print(current_train_size)




