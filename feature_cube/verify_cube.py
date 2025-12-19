import numpy as np
import xarray as xr
#cube_nums = ["000", "001", "003", "004", "005", "006", "007", "008", "009", '011', '012', '013', '016', '017', '018', "019", "020", "021", '022', '024', '025', '026', '027',"028", '029', '030', '034', '035', '036', '038', '039', '040', '042', '044', '046']
# Path to Zarr cube
zarr_path = '/net/data_ssd/deepfeatures/sciencecubes_processed/000000_feature_cube.zarr'
zarr_path = '/net/data_ssd/deepfeatures/sciencecubes_processed/s1_s2_000000.zarr'
zarr_path = ('/net/data_ssd/deepfeatures/sciencecubes_processed/s1_s2_039_v.zarr')
#zarr_path = '/net/data_ssd/deepfeatures/sciencecubes_processed/000000.zarr'


## Open the dataset
ds_reloaded = xr.open_zarr(zarr_path, chunks=None)
print(ds_reloaded)
#ds_reloaded = ds_reloaded.isel(time=slice(0, 86))

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

print(f"\nOverall available data (non-NaN): {total_elements} ({available_percentage:.2f}%)")

# Check if entire frame is NaN -> reduce over (x,y)
frame_all_nan = data.isnull().all(dim=["x", "y"])  # boolean per time

# Absolute number of frames entirely NaN
num_frames_all_nan = int(frame_all_nan.sum().item())

# Total number of frames
num_total_frames = data.sizes["time"]

# Percentage
pct_frames_all_nan = 100.0 * num_frames_all_nan / num_total_frames

print(f"\nFrames entirely NaN: {num_frames_all_nan} / {num_total_frames} "
      f"({pct_frames_all_nan:.2f}%)")

# Also: number and percentage of frames that are NOT entirely NaN
num_frames_not_all_nan = num_total_frames - num_frames_all_nan
pct_frames_not_all_nan = 100.0 * num_frames_not_all_nan / num_total_frames

print(f"Frames with at least some valid data: {num_frames_not_all_nan} / {num_total_frames} "
      f"({pct_frames_not_all_nan:.2f}%)")



