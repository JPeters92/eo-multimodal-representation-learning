import torch
import random
import numpy as np
from typing import Dict, List, Tuple


def drop_if_central_point_nan_or_inf(
        ds: Dict[str, np.ndarray],
        coords: Dict[str, np.ndarray],
        vars: List[str]
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Drop samples entirely if the central point of any variable contains NaN or inf values.

    Args:
        ds (Dict[str, np.ndarray]): The dataset to filter. It should be a dictionary where keys are variable names and values are numpy arrays.
        coords (Dict[str, np.ndarray]): The coordinates associated with the dataset.
        vars (List[str]): The variables to check for NaN or inf values at their central point.

    Returns:
        Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]: The filtered dataset and coordinates.
    """
    valid_mask_lists = []


    for var in vars:
        if var not in ds:
            continue
        value = ds[var]

        # Determine the central point index
        central_index = tuple(dim_size // 2 for dim_size in value.shape[1:])

        if value.ndim == 1:  # For 1D arrays, the central point is the single value
            valid_mask = ~np.isnan(value) & ~np.isinf(value)
        elif value.ndim >= 2:  # For multi-dimensional arrays
            central_values = value[:, 5, 7, 7]  # Extract central point along non-batch axes
            valid_mask = ~np.isnan(central_values) #& ~np.isinf(central_values)
        else:
            raise ValueError("Unsupported number of dimensions for the variable.")

        # num_invalid = np.count_nonzero(~valid_mask)
        #print(f"{var}: {num_invalid} invalid samples at central point")
        valid_mask_lists.append(valid_mask)

    # Combine masks across all variables using logical AND
    valid_mask = np.all(valid_mask_lists, axis=0)

    # Filter the dataset and coordinates
    ds = {key: ds[key][valid_mask] for key in ds.keys()}
    coords = {key: coords[key][valid_mask] for key in coords.keys()}

    return ds, coords


def drop_if_central_point_nan_at_selected_times(
    ds: Dict[str, np.ndarray],
    coords: Dict[str, np.ndarray],
    vars: List[str],
    required_time_indices: List[int]
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Drop samples where the central point at any of the specified time indices contains NaN.

    Args:
        ds (Dict[str, np.ndarray]): The dataset to filter. Each value is a NumPy array of shape
                                    (num_samples, time, height, width).
        coords (Dict[str, np.ndarray]): Associated coordinates (e.g., 'time', 'x', 'y').
        vars (List[str]): Variable names to check.
        required_time_indices (List[int]): Timestamps (indices) to check the central point for NaNs.

    Returns:
        Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]: Filtered dataset and coordinates.
    """
    valid_mask_lists = []

    for var in vars:
        if var not in ds:
            continue

        value = ds[var]  # (N, T, H, W)
        if value.ndim < 4:
            raise ValueError(f"Variable {var} must be at least 4D (batch, time, height, width)")

        # Get central spatial index
        center_y = value.shape[2] // 2
        center_x = value.shape[3] // 2

        # Collect boolean mask across all specified time indices
        combined_valid = np.ones(value.shape[0], dtype=bool)
        for t_idx in required_time_indices:
            if t_idx >= value.shape[1]:
                raise IndexError(f"Requested time index {t_idx} out of bounds for variable '{var}'")

            central_values = value[:, t_idx, center_y, center_x]
            is_valid = ~np.isnan(central_values)  # You can also add ~np.isinf(...) if needed
            combined_valid &= is_valid

        valid_mask_lists.append(combined_valid)

    # Combine across all variables (logical AND)
    valid_mask = np.all(valid_mask_lists, axis=0)

    # Apply mask
    ds_filtered = {key: ds[key][valid_mask] for key in ds}
    coords_filtered = {key: coords[key][valid_mask] for key in coords}

    return ds_filtered, coords_filtered



def concatenate(
    chunks, coords, masks, shuffle: bool = False
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """
    Concatenate the chunks along the time dimension.

    Args:
        chunks: List of data chunks
        coords: List of coordinate dictionaries
        masks: List of mask dictionaries
        shuffle: If True, shuffles the concatenated data along the sample dimension

    Returns:
        Tuple:
            - Stacked data array
            - Concatenated coordinates
            - Stacked masks array
    """
    # Filter out empty entries
    valid_entries = [
        (chunk, coord, mask)
        for chunk, coord, mask in zip(chunks, coords, masks)
        if chunk and coord and mask
    ]

    if len(valid_entries) == 0:
        raise ValueError("All chunks are empty after filtering. Nothing to concatenate.")

    chunks, coords, masks = zip(*valid_entries)

    # Preallocate memory-efficient structures
    chunk_keys = [key for key in chunks[0] if key != 'split']
    coord_keys = list(coords[0].keys())

    # Concatenate data chunks directly into arrays
    stacked_data = np.concatenate(
        [np.stack([chunk[key] for key in chunk_keys], axis=1) for chunk in chunks],
        axis=0
    )

    # Concatenate masks directly into arrays
    stacked_masks = np.concatenate(
        [np.stack([mask[key] for key in chunk_keys], axis=1) for mask in masks],
        axis=0
    )

    # Normalize and concatenate coordinates
    concatenated_coords = {}
    for key in coord_keys:
        filtered_coords = [coord[key] for coord in coords if len(coord[key].shape) > 1]

        if not filtered_coords:
            raise ValueError(f"No valid coordinate arrays for key '{key}' after filtering.")

        max_dim = max(len(arr.shape) for arr in filtered_coords)
        target_shape = (0,) + tuple(
            max(arr.shape[i] for arr in filtered_coords if arr.size > 0) for i in range(1, max_dim)
        )

        normalized_arrays = [
            np.empty(target_shape, dtype=arr.dtype) if arr.size == 0 else arr
            for arr in filtered_coords
        ]

        concatenated_coords[key] = np.concatenate(normalized_arrays, axis=0)

    # Optional shuffling
    if shuffle:
        idx = np.random.permutation(stacked_data.shape[0])
        stacked_data = stacked_data[idx]
        stacked_masks = stacked_masks[idx]
        for key in concatenated_coords:
            concatenated_coords[key] = concatenated_coords[key][idx]

    return stacked_data, concatenated_coords, stacked_masks


def compute_time_gaps(time_coords: np.ndarray) -> torch.Tensor:
    """
    Computes time gaps (in days) between consecutive timestamps for each sample.

    Args:
        time_coords (np.ndarray):
            Either a 1D array with shape (n_timestamps,) for a single sample,
            or a 2D array with shape (batch_size, n_timestamps).

    Returns:
        torch.Tensor: A 2D tensor of shape (batch_size, n_timestamps - 1)
                      containing integer day differences.
    """
    if time_coords.ndim == 1:
        # Single sample → reshape to (1, n_timestamps)
        time_coords = np.expand_dims(time_coords, axis=0)

    elif time_coords.ndim > 2:
        raise ValueError(f"time_coords must be 1D or 2D, got shape {time_coords.shape}")

    # Handle case with not enough time steps
    if time_coords.shape[1] < 2:
        return torch.empty((time_coords.shape[0], 0), dtype=torch.int32)

    # Compute day differences along time axis
    time_deltas = np.diff(time_coords.astype('datetime64[D]'), axis=1).astype(int)

    return torch.tensor(time_deltas, dtype=torch.int32)



def select_random_timestamps(chunk, coords, num_timestamps=11):
    """
    Selects a random subset of timestamps for each sample in a data cube.

    Args:
        chunk (dict of np.ndarray): The data cube, where each variable is a NumPy array of shape
                                  (num_samples, num_time, height, width).
        coords (dict): Coordinates associated with the data cube. Assumes 'time' key exists.
        num_timestamps (int): The number of random timestamps to select for each sample.

    Returns:
        updated_chunk (dict of np.ndarray): The updated data cube with selected timestamps.
        updated_coords (dict): The updated coordinates with adjusted 'time'.
    """
    # Determine the shape from a representative variable
    example_var = next(iter(chunk.values()))  # Take the first variable as an example
    num_samples, total_timestamps, height, width = example_var.shape

    # Generate random indices for selecting timestamps for each sample
    selected_indices = np.array([
        sorted(random.sample(range(total_timestamps), num_timestamps)) for _ in range(num_samples)
    ])

    # Create a new dictionary to store the updated data
    updated_chunk = {}

    # Iterate over each variable in the chunk
    for var_name, data in chunk.items():
        # Prepare an array to store the selected timestamps for this variable
        selected_data = np.empty((num_samples, num_timestamps, height, width), dtype=data.dtype)

        # Apply random selection for each sample
        for i in range(num_samples):
            selected_data[i] = data[i, selected_indices[i], :, :]

        # Store the updated data in the new chunk
        updated_chunk[var_name] = selected_data

    # Update 'time' in coords
    updated_coords = coords.copy()
    updated_coords['time'] = np.array([
        np.array(coords['time'][i]).flatten()[selected_indices[i]] for i in range(num_samples)
    ])

    # Verify that all updated samples have exactly num_timestamps
    for var_name, data in updated_chunk.items():
        assert data.shape[1] == num_timestamps, f"Variable {var_name} does not have {num_timestamps} timestamps."

    return updated_chunk, updated_coords



def select_timestamps_from_sections(chunk, coords, section_lengths, num_select_per_section):
    """
    Selects a specified number of random timestamps from defined sections of a time series.

    Args:
        chunk (dict of np.ndarray): The data cube, where each variable is a NumPy array of shape
                                    (num_samples, num_time, height, width).
        coords (dict): Coordinates associated with the data cube. Must include a 'time' key.
        section_lengths (list of int): Defines how the time axis is split into consecutive sections.
        num_select_per_section (list of int): Number of timestamps to randomly select from each section.

    Returns:
        updated_chunk (dict of np.ndarray): The updated data with selected timestamps.
        updated_coords (dict): The updated coordinates with adjusted 'time'.
    """
    assert len(section_lengths) == len(num_select_per_section), \
        "section_lengths and num_select_per_section must have the same length."

    example_var = next(iter(chunk.values()))
    num_samples, total_timestamps, height, width = example_var.shape

    assert sum(section_lengths) == total_timestamps, \
        f"Section lengths {section_lengths} do not sum to total timestamps {total_timestamps}."

    # Precompute start and end indices of each section
    section_boundaries = np.cumsum([0] + section_lengths)

    # Generate selected indices per sample
    selected_indices = []
    for _ in range(num_samples):
        sample_indices = []
        for i, n_select in enumerate(num_select_per_section):
            start = section_boundaries[i]
            end = section_boundaries[i + 1]
            assert (end - start) >= n_select, \
                f"Section {i} has only {(end - start)} timestamps, but {n_select} were requested."
            section_indices = random.sample(range(start, end), n_select)
            sample_indices.extend(sorted(section_indices))
        selected_indices.append(sample_indices)

    selected_indices = np.array(selected_indices)

    # Create a new dictionary to store the updated data
    updated_chunk = {}
    for var_name, data in chunk.items():
        selected_data = np.empty((num_samples, selected_indices.shape[1], height, width), dtype=data.dtype)
        for i in range(num_samples):
            selected_data[i] = data[i, selected_indices[i], :, :]
        updated_chunk[var_name] = selected_data

    # Update time coordinates
    updated_coords = coords.copy()
    updated_coords['time'] = np.array([
        np.array(coords['time'][i]).flatten()[selected_indices[i]] for i in range(num_samples)
    ])

    return updated_chunk, updated_coords


def extract_center_coordinates(coords):
    # Extract center coordinates as arrays
    center_time = coords['time'][0, coords['time'].shape[1] // 2]  # Single time value
    center_xs = coords['x'][:, coords['x'].shape[1] // 2]  # Flattened
    center_ys = coords['y'][:, coords['y'].shape[1] // 2]  # Flattened

    return center_time, center_xs, center_ys
