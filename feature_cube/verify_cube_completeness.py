import xarray as xr

# Path to Zarr cube
zarr_path = '/net/data_ssd/deepfeatures/processed/000010.zarr'

## Open the dataset
ds_reloaded = xr.open_zarr(zarr_path, chunks=None)
print(ds_reloaded)

#ds_reloaded = ds_reloaded.isel(time=slice(0, 422))
# Assign names to the 'feature' dimension
ds_reloaded = ds_reloaded.assign_coords(feature=[f"feature_{i}" for i in range(ds_reloaded.dims['feature'])])

# Get the actual data array
features_zarr = ds_reloaded["features"]
data = features_zarr.sel(feature='feature_0')

# Total number of pixels per timestep
total_elements_per_timestep = data.sizes['x'] * data.sizes['y']

# Count NaNs per timestep
nan_counts = xr.where(data.isnull(), 1, 0).sum(dim=['x', 'y']).compute()

# Compute % NaNs
percentage_per_timestep = nan_counts / total_elements_per_timestep * 100

# Print both absolute and percentage NaN counts
cnt = 0
perct = 0
for time, abs_nan, pct in zip(data['time'].values, nan_counts.values, percentage_per_timestep.values):
    print(f"{str(time)}: {int(abs_nan):,} NaNs ({pct:.4f}%)")
    perct += pct
    cnt += 1

print(f"\nAverage % NaNs per timestep: {perct / cnt:.4f}%")


# Global NaN stats
total_elements = data.sizes['time'] * data.sizes['x'] * data.sizes['y']
total_nan_count = xr.where(data.isnull(), 1, 0).sum(dim=['x', 'y', 'time']).compute()
available_percentage = 100 - (total_nan_count / total_elements * 100)

print(f"\nOverall available data (non-NaN): {available_percentage:.2f}%")
#print(f"\nOverall available data (non-NaN): {total_elements}")

# Timestamps with any non-NaN values
timestamps_with_data = int((nan_counts < total_elements_per_timestep).sum().item())
total_timestamps = int(data.sizes['time'])
print(f"\nTimestamps with non-NaN values: {timestamps_with_data} / {total_timestamps}")
