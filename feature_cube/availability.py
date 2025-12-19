import numpy as np
import xarray as xr

CUBE = "001"
TS = np.datetime64("2017-06-15T10:50:31.026000000", "ns")

SCI_PATH  = f"/net/data/deepfeatures/sciencecubes/{CUBE}.zarr"
SCI_PATH  = f"/net/data_ssd/deepfeatures/s1_s2_chubes/s1_s2_{CUBE}.zarr"
PROC_PATH = f"/net/data_ssd/deepfeatures/sciencecubes_processed/s1_s2_{CUBE}.zarr"

SCI_VAR   = "s2l2a"
CLOUD_VAR = "cloud_mask"
PROC_VAR  = "features"

VALID_FRACTION_THRESHOLD = 0.15
CENTER_TEST_SIZE = 6
KM_WINDOW = 100

def centered_slice(n, win):
    s = n//2 - win//2
    return slice(s, s+win)

# --- Load SCI and evaluate criteria at TS ---
ds = xr.open_zarr(SCI_PATH, chunks=None)

H = ds.sizes["y"]  # 1000
W = ds.sizes["x"]  # 1000
start_y = (H // 2) - 100
end_y = (H // 2) + 100
start_x = (W // 2) - 100
end_x = (W // 2) + 100
da = ds.isel(y=slice(start_y, end_y), x=slice(start_x, end_x))

# Cloud-masked S2 (dims typically: band,time,y,x)
#da = ds.s2l2a.where(ds.cloud_mask == 0)
print(da)
da = da.sel(time=TS, band='B02')
da = da["bands"]
da.plot()
import matplotlib.pyplot as plt
plt.show()

array = da.values
print(array)

array = da.values          # Convert to NumPy array
num_nans = np.isnan(array).sum()

print("Number of NaNs:", num_nans)

total = np.prod(da.shape)
missing_pct = (num_nans / total) * 100
print(f"Missing percentage: {missing_pct:.2f}%")