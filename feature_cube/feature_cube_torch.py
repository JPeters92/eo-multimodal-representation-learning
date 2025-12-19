import os
import time
import torch
import warnings
import numpy as np
import xarray as xr
from tqdm import tqdm
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from utils.utils import compute_time_gaps, extract_center_coordinates
from model.model_s1_s2 import TransformerAE
from model.model_fusion import FusedS1S2
from dataset.preprocess_sentinel import merge_s1_s2, extract_sentinel_patches



def create_empty_dataset(feature_names, xs, ys, out_path, times=None, dtype=np.float32):
    """
    Create an empty xarray.Dataset with dims (feature, time, y, x).
    If times is None -> time dim length 0 (good for appending).
    """


    if os.path.exists(out_path):
        # Just open existing store
        return xr.open_zarr(out_path)

    times = np.asarray(times).astype("datetime64[ns]")

    shape = (len(feature_names), len(times), len(ys), len(xs))
    data = np.full(shape, np.nan, dtype=dtype)

    da = xr.DataArray(
        data,
        dims=("feature", "time", "y", "x"),
        coords={
            "feature": np.asarray(feature_names, dtype=str),
            "time": times,
            "y": np.asarray(ys),
            "x": np.asarray(xs),
        },
        name="features",
    )

    # chunk along time=1 to match region writes
    encoding = {"features": {"chunks": (len(feature_names), 1, len(ys), len(xs))}}

    ds_out = xr.Dataset({"features": da}).drop_vars("feature")
    ds_out.to_zarr(out_path, mode="w", encoding=encoding)
    return ds_out


def init_output_from_source(da, feature_names, out_path):
    """
    Determine valid timestamps (>= 5% complete pixels) on the cropped grid
    and create the target Zarr with exactly these times.
    - time is sliced as time[5:-5]
    - y, x are sliced as [7:-7]
    """

    if os.path.exists(output_path):
        print(':)')
        ds0 = xr.open_zarr(output_path, consolidated=True)

        feats = ds0["features"]  # (feature, time, y, x)
        reduce_dims = tuple(d for d in feats.dims if d != "time")

        # Build mask and COMPUTE it to a NumPy bool array (1D over time)
        empty_mask_da = feats.isnull().all(dim=reduce_dims)  # (time,) dask-backed
        empty_mask_np = empty_mask_da.compute().values.astype(bool)  # -> (T,)

        # Index times via NumPy (avoid .where(drop=True) with dask)
        times_np = ds0["time"].values  # (T,)
        empty_times = times_np[empty_mask_np]  # timestamps entirely NaN

        xs = ds0["x"].values
        ys = ds0["y"].values

        t_ns = ds0['time'].values.astype('datetime64[ns]')
        time_to_idx = {int(t.astype('int64')): i for i, t in enumerate(t_ns)}
        return ds0, empty_times, xs, ys, time_to_idx

    # Crops: time[5:-5], y[7:-7], x[7:-7]
    da_c = da.isel(time=slice(5, -5), y=slice(7, -7), x=slice(7, -7), band=slice(0, 10))

    # "Complete pixel" per time = all bands valid (no NaNs) -> boolean (time, y, x)
    complete_px = np.isfinite(da_c).all(dim="band")        # (time,y,x) -> bool
    frac_complete = complete_px.mean(dim=("y", "x"))       # (time,)

    # Filter (>= 5%)
    okA = (frac_complete >= 0.035).compute().values  # (time,) bool np.ndarray

    #cy0 = da_c.sizes["y"] // 2 - 6 // 2
    #cx0 = da_c.sizes["x"] // 2 - 6 // 2
    #ys6 = slice(cy0, cy0 + 6)
    #xs6 = slice(cx0, cx0 + 6)
#
    #center = da_c.isel(y=ys6, x=xs6)  # (band,time,6,6)
    #center_any = center.notnull().any(dim="band")  # (time,6,6) -> bool
    #center_valid_counts = center_any.sum(dim=("y", "x"))  # (time,)
    #okB = center_valid_counts >= 1

    # ---------- Keep times that satisfy A OR B ----------
    keep_mask = (okA)# | okB)



    ok_idx = np.flatnonzero(keep_mask)  # integer positions
    times_ok_ns = np.asarray(da_c.time.values)[ok_idx]

    #  Cropped target coordinates
    global_xs = da.x.values[7:-7]
    global_ys = da.y.values[7:-7]

    # Create Zarr with exactly these times
    ds0 = create_empty_dataset(
        feature_names=feature_names,
        xs=global_xs,
        ys=global_ys,
        out_path=out_path,
        times=times_ok_ns,
        dtype=np.float32,
    )
    t_ns = ds0['time'].values.astype('datetime64[ns]')
    time_to_idx = {int(t.astype('int64')): i for i, t in enumerate(t_ns)}
    return ds0, times_ok_ns, global_xs, global_ys, time_to_idx



def reset_frame():
    global current_canvas, filled_once, current_time
    current_canvas[:] = np.nan
    filled_once[:] = False
    current_time = None


def flush_frame(canvas, f_ds, out_path, time, time_to_idx):
    """Add current frame to Zarr."""

    if time is None:
        return False

    t_ns = np.datetime64(time, 'ns')
    key = int(t_ns.astype('int64'))
    if key not in time_to_idx:
        # Zeit nicht vorgesehen -> NICHT schreiben
        print(f"[flush_frame] Skip: {t_ns} not in target times.")
        return False

    idx = time_to_idx[key]

    # shape (C, Y, X) -> expand to (C, 1, Y, X)
    da = xr.DataArray(
        canvas[:, np.newaxis, :, :],   # add time axis in 2nd position
        dims=("feature", "time", "y", "x"),
        coords={
            "feature": f_ds.feature.values,
            "time": f_ds.time.values[idx:idx+1],
            "y": f_ds.y.values,
            "x": f_ds.x.values,
        },
        name="features",
    )

    ds = xr.Dataset({"features": da}).drop_vars("feature")

    ds.to_zarr(out_path, mode="r+", region={
        "time": slice(idx, idx + 1),
        "y": slice(0, len(f_ds.y)),
        "x": slice(0, len(f_ds.x))
    })


    return True


def coord_to_idx(vals, mapping, axis_vals):
    """
    Map 1D array of coords -> indices.
    Uses dict fast path for exact float matches.
    Falls back to nearest index if not found.
    """
    vals = np.asarray(vals)
    idxs = np.empty(vals.shape, dtype=np.int64)

    for j, v in enumerate(vals):
        fv = float(v)
        if fv in mapping:
            idxs[j] = mapping[fv]
        else:
            # fallback: nearest
            idxs[j] = int(np.argmin(np.abs(axis_vals - v)))
    return idxs


class XrFeatureDataset:
    def __init__(
            self,
            data_cube: xr.DataArray,
            matched_s1_times,
            times_ok_ns,
            time_block_size: int = 11,
            space_block_size: int = 90,
            time_overlap: int = 10,
            space_overlap: int = 14
    ):
        self.data_cube = data_cube
        self.matched_s1_times = matched_s1_times
        self.times_ok_ns = times_ok_ns
        self.time_block_size = time_block_size
        self.space_block_size = space_block_size
        self.time_overlap = time_overlap
        self.space_overlap = space_overlap

        # infer sizes (dims: band, time, y, x)
        self.time_len = int(data_cube.sizes["time"])
        self.y_len = int(data_cube.sizes["y"])
        self.x_len = int(data_cube.sizes["x"])

        print(self.y_len, self.x_len, self.time_len)

        self.save_frame = True

        self.chunks_bounds = self.compute_bounds(time_slide = True, time_block=self.time_block_size, space_block=self.space_block_size)  # list of (t0,t1,y0,y1,x0,x1)
        print(f'Chunks to process: {len(self.chunks_bounds)}')
        #self.chunk_idx = 415
        self.chunk_idx = 0

        # fast membership test on ns-int
        self.times_ok_ns = np.asarray(times_ok_ns).astype("datetime64[ns]")
        self._times_ok_set = set(self.times_ok_ns.astype("int64").tolist())

    def __iter__(self):
        return self

    def compute_bounds(self, time_slide, time_block, space_block, split_chunk=False):
        """Return list of (t0, t1, y0, y1, x0, x1) with overlaps.
        Ends are computed from the nominal (non-overlapped) starts, then
        starts are shifted backward by the overlap for i>0."""

        def nominal_ranges(n, block, *, sliding=False):
            """
            Compute ranges as (start, end).
            - If sliding=True: use stride=1 (e.g., 0-11, 1-12, 2-13, ...)
            - Else: use stride=block (non-overlapping blocks).
            """
            if sliding:
                return [(i, i + block) for i in range(0, n - block + 1, 1)]
            else:
                return [(i, i + block) for i in range(0, n, block) if i + block <= n]


        if split_chunk and self.chunk_idx == 0:
            t_len = self.time_block_size
        elif split_chunk and self.chunk_idx > 0:
            t_len = self.time_block_size + 10
        else: t_len = self.time_len

        t_nom = nominal_ranges(t_len, time_block, sliding=time_slide)
        y_nom = nominal_ranges(self.y_len, space_block)
        x_nom = nominal_ranges(self.x_len, space_block)

        chunks = []
        # iterate Y, then X, then T  ⟶ fills same spatial frame first
        for (t0_nom, t1_nom) in t_nom:
            if time_slide:
                t0 = t0_nom
            else:
                t0 = t0_nom - self.time_overlap if t0_nom > 0 else t0_nom

            t1 = t1_nom #- self.time_overlap if t0_nom > 0 else t1_nom
            t0 = max(0, t0);
            t1 = min(t_len, t1)

            for (y0_nom, y1_nom) in y_nom:
                y0 = y0_nom - self.space_overlap if y0_nom > 0 else y0_nom
                y1 = y1_nom
                y0 = max(0, y0); y1 = min(self.y_len, y1)

                for (x0_nom, x1_nom) in x_nom:
                    x0 = x0_nom - self.space_overlap if x0_nom > 0 else x0_nom
                    x1 = x1_nom
                    x0 = max(0, x0); x1 = min(self.x_len, x1)

                    chunks.append((t0, t1, y0, y1, x0, x1))

        return chunks

    def nan_stats(self, da: xr.DataArray) -> tuple[int, int]:
        """
        Returns (nan_count, non_nan_count) for an xarray.DataArray.
        Works for both NumPy- and Dask-backed arrays.
        """
        # count NaNs across all dims
        nan_da = da.isnull().sum()  # reduces over all dims
        nan = nan_da.compute().item() if getattr(da.data, "chunks", None) else nan_da.item()

        # total elements = product of dimension sizes
        total = 1
        for d in da.dims:
            total *= da.sizes[d]

        return int(nan), int(total - nan)

    def reset(self):
        pass  # Nothing to reset in this version


    def __next__(self):
        #if self.chunk is None:
        warnings.filterwarnings('ignore')

        if self.chunk_idx >= len(self.chunks_bounds):
            raise StopIteration

        t0, t1, y0, y1, x0, x1 = self.chunks_bounds[self.chunk_idx]
        print(f"Getting chunk time={t0}-{t1} y={y0}-{y1} x={x0}-{x1}")

        chunk = self.data_cube.isel(
            time=slice(t0, t1),
            y=slice(y0, y1),
            x=slice(x0, x1),
        )

        data = chunk.values

        #nan_count = np.isnan(data[3, 5, 7:-7, 7:-7]).sum()
        #non_nan_count = np.count_nonzero(~np.isnan(data[3, 5, 7:-7, 7:-7]))
        valid_pixel_mask  = np.isnan(data[:10, 5, 7:-7, 7:-7])#.any(axis=0)
        nan_count = valid_pixel_mask.sum()
        non_nan_count = (~valid_pixel_mask).sum()

        print(f"Chunk NaNs: {nan_count:,}, Non-NaNs: {non_nan_count:,}")
        print(f'Chunk shape: {data.shape}')
        coords = {k: chunk.coords[k].values for k in chunk.coords}

        ct_idx = coords["time"].size // 2
        ct = np.datetime64(coords["time"][ct_idx]).astype("datetime64[ns]")
        print(ct)

        if int(ct.astype("int64")) not in self._times_ok_set:
            print(f"⏭️ Skipping chunk {self.chunk_idx}: center time {ct} not in times_ok_ns.")
            #self.chunk_idx = ((self.chunk_idx // 16) + 1) * 16 - 1
            self.save_frame = False
            print(f'Setting flag save frame to {self.save_frame}')
            return None, None, None, None, None, None

        if non_nan_count == 0:

            return None, None, None, None, None, None





        print(f'Splitting chunk {self.chunk_idx}')
        patches_all, coords_all, valid_mask_all, not_val = extract_sentinel_patches(
            data,  # dask array; extractor should handle array-like or call np.asarray internally
            coords['time'],  # small vector -> fine to realize
            coords['y'],  # numpy or small vector
            coords['x'],
            time_coords_2=self.matched_s1_times[t0:t1],
            time_win=11,
            time_stride=1,
            h_stride=1,
            w_stride=1
        )

        if patches_all.shape[0] == 0 and not_val: self.save_frame = False

        if patches_all.shape[0] == 0: return None, None, None, None, None, None


        time_gaps_s2 = compute_time_gaps(coords_all['time'])
        time_gaps_s1 = compute_time_gaps(coords_all['time_add'])
        time_add = coords_all["time_add"][:, 5].astype("datetime64[D]").astype("int64")
        time_ref = coords_all["time"][:, 5].astype("datetime64[D]").astype("int64")

        # difference in days as torch tensor
        time_gaps_c = torch.abs(torch.from_numpy(time_add - time_ref)).view(-1, 1)


        return patches_all, coords_all, valid_mask_all, time_gaps_s1, time_gaps_s2, time_gaps_c


#s1_d2_ckpt = '../checkpoints/003_025_072_test/s1_s2_3/ae-9-epoch=93-val_loss=2.035e-03.ckpt'
s1_d2_ckpt = '../checkpoints/s2/ae-7-epoch=113-val_loss=2.161e-03.ckpt'
device = torch.device("cuda:0")

ae_s1 = TransformerAE(dbottleneck=2, channels=2, num_reduced_tokens=7).eval()
ae_s2 = TransformerAE(dbottleneck=9, channels=10, num_reduced_tokens=6).eval()
model = FusedS1S2(
    enc_s1=ae_s1.encoder, dec_s1=ae_s1.decoder,
    enc_s2=ae_s2.encoder, dec_s2=ae_s2.decoder,
    dbottleneck_s1=2,
    dbottleneck_s2=9,
    freeze_encoders=False,
    dbottleneck=7
)

checkpoint = torch.load(s1_d2_ckpt, map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model.to(device)
model.eval()




batch_size = 64
#
cube_nums = ["005","010","012","017","018","026","029","030","033","034","037","042","049","052","056","059",
    "063","065","069","071","073","074","079","080","083","085","086","089","090","092",
    "095","096",
    "099","107","111","116","119","120","125","127","128","130","133","134","137","138","141","143",
    "145","147","154","159","161","165","168","170","174","178","181","184","186","193","202","205",
    "206","208","210","215","216","221","222","225","229","231","232","233","234","236","240","243",
    "246","248","251","253","257","259","261","268","271","274","278","279","280","281","282","284",
    "288","291","303","306","308","315","319","320","322","325","331","335","336","339","341","343",
    "345","347","348","351","352","355","356","362","363","366","368","370","373","374","377","378",
    "382","387","389","391","393","395","401","402","404","405","411","413","415","417","423","427",
    "428","430","432","434","436","439","442","447","449","450","452","455","456","459","463","464",
    "469","471","473","477","478","480","484","486","495","497"
                 ]


# Convert to set for fast lookup
existing = set(cube_nums)

# Generate all 3-digit strings 000–499
all_nums = [f"{i:03d}" for i in range(500)]

# Get missing ones
cube_nums = [num for num in all_nums if num not in existing]

# Convert to integers for easy comparison
cube_nums = [num for num in all_nums if num not in existing]

# Keep only cubes after 248
cube_nums = [num for num in cube_nums if int(num) > 248]


print(cube_nums)

cube_nums = [f"{int(num):06d}" for num in cube_nums]


for cube_num in cube_nums:


    global_mae_sum = 0.0
    global_mape_sum = 0.0
    global_count = 0
    eps = 1e-6  # avoid div by zero

    print(f'processing cube {cube_num}')

    #ds = xr.open_zarr('/net/data_ssd/deepfeatures/sciencecubes/000000.zarr')

    #ds = merge_s1_s2('004', base_path = '/net/data/deepfeatures/sciencecubes', chunks={"time": 1, "y": 1000, "x": 1000}, save_path='/net/data_ssd/deepfeatures/s1_s2_chubes/s1_s2_004.zarr')
    ds = merge_s1_s2(cube_num, base_path = '/net/data/deepfeatures/trainingcubes', save_path=f'/net/data_ssd/deepfeatures/s1_s2_chubes/s1_s2_{cube_num}_v.zarr')


    #print(ds.time.values)


    matched_s1_times = ds['s1_time']
    feature_names = ['F01', 'F02', 'F03', 'F04', 'F05', 'F06', 'F07']


    output_path = f"/net/data_ssd/deepfeatures/trainingcubes_processed/s1_s2_{cube_num}_v.zarr"
    init_path = f"/net/data_ssd/deepfeatures/trainingcubes_processed/s1_s2_{cube_num}_t.zarr"
    ds0, times_ok_ns, global_xs, global_ys, time_to_idx = init_output_from_source(ds, feature_names, output_path)

    print(f'creating {cube_num}_v.zarr')


    print(ds0)



    x_to_idx = {float(v): i for i, v in enumerate(global_xs)} #['2016-12-27T10:54:42.026000000'
    y_to_idx = {float(v): i for i, v in enumerate(global_ys)}


    C, Y, X = len(feature_names), len(global_ys), len(global_xs)
    current_canvas = np.full((C, Y, X), np.nan, dtype=np.float32)
    filled_once = np.zeros((Y, X), dtype=bool)  # first-write-wins for overlaps
    current_time = None

    dataset = XrFeatureDataset(
        data_cube=ds,
        matched_s1_times = matched_s1_times,
        times_ok_ns = times_ok_ns
    )

    for chunk_idx, chunk in enumerate(dataset):
        mae_sum = 0.0
        mape_sum = 0.0
        count = 0
        start_time = time.time()
        processed_data, coords, valid_mask, time_gaps_s1, time_gaps_s2, time_gaps_c = chunk
        if processed_data is None: N = 0
        else:
            N = processed_data.shape[0]

            center_time, center_xs, center_ys = extract_center_coordinates(coords)
            current_time = center_time  # <--- add this line

        for start in tqdm(range(0, N, batch_size), desc="Reconstructing", unit="batch"):
            end = min(start + batch_size, N)

            # slice + move to device
            batch_processed = processed_data[start:end].to(device, dtype=torch.float32)
            batch_mask = valid_mask[start:end].to(device, dtype=torch.bool)
            batch_s1 = time_gaps_s1[start:end].to(device, dtype=torch.int32)
            batch_s1 = time_gaps_s1[start:end].to(device, dtype=torch.int32)
            batch_s2 = time_gaps_s2[start:end].to(device, dtype=torch.int32)
            batch_c = time_gaps_c[start:end].to(device, dtype=torch.int32)

            #batch_s1 = torch.ones_like(batch_s1, device=batch_s1.device, dtype=batch_s1.dtype)
            #batch_s2 = torch.ones_like(batch_s2, device=batch_s2.device, dtype=batch_s2.dtype)

            x_s2 = batch_processed[:, :, :10, :, :]
            x_s1 = batch_processed[:, :, 10:, :, :]
            #
            mask_s2 = valid_mask[:, :, :10, :, :]
            mask_s1 = valid_mask[:, :, 10:, :, :]


            model_input = (x_s1, x_s2, batch_s1, batch_s2, batch_c)
            y_s1, y_s2, zf = model(model_input)
            y_all = torch.cat([y_s2, y_s1], dim=2)

            # --- central coordinate ---
            B, T, C, H, W = batch_processed.shape
            ct, cx, cy = T // 2, H // 2, W // 2

            central_in = batch_processed[:, ct, :, cx, cy]  # [B, C]
            central_out = y_all[:, ct, :, cx, cy]  # [B, C]
            central_mask = batch_mask[:, ct, :, cx, cy]  # [B, C] (bool)

            # ---- write predictions to ds_pred at correct (band,time,y,x) ----
            # center_xs/center_ys are aligned to patches globally; take the batch slice
            bx = center_xs[start:end]  # length B
            by = center_ys[start:end]  # length B

            x_idx = coord_to_idx(bx, x_to_idx, global_xs)
            y_idx = coord_to_idx(by, y_to_idx, global_ys)

            # then vectorized write
            current_canvas[:, y_idx, x_idx] = zf.detach().cpu().numpy().astype(np.float32).T
            filled_once[y_idx, x_idx] = True

            # move predicted central values to CPU/np
            central_out_np = central_out.detach().cpu().numpy().astype(np.float32)
            central_in_np = central_in.detach().cpu().numpy().astype(np.float32)

            # filter only valid entries
            valid_in = central_in[central_mask]
            valid_out = central_out[central_mask]

            diff = (valid_out - valid_in).abs()
            mae_sum += diff.sum().item()
            mape_sum += (diff / valid_in.abs().clamp_min(eps)).sum().item()
            count += valid_mask[:, ct, :, cx, cy].sum().item()
            global_mae_sum += diff.sum().item()
            global_mape_sum += (diff / valid_in.abs().clamp_min(eps)).sum().item()
            global_count += valid_mask[:, ct, :, cx, cy].sum().item()
        print(f'Chunk {dataset.chunk_idx} ({cube_num}) processed in {time.time() - start_time:.3f} s')

        # global center metrics
        chunk_mae = mae_sum / max(count, 1)
        global_mae_sum += mae_sum
        chunk_mape = 100.0 * mape_sum / max(count, 1)
        global_mape_sum +=  100.0 * mape_sum

        # Your rule: after every 16 chunks, flush the frame
        #if (dataset.chunk_idx + 1) % 16 == 0:
        if dataset.save_frame:
            saved = flush_frame(
                canvas=current_canvas, f_ds=ds0, out_path=output_path,
                time=current_time, time_to_idx=time_to_idx
            )

            print(f"🗂️ Frame {np.datetime_as_string(current_time, unit='D')} "
                  f"{'saved' if saved else 'skipped (<5% coverage)'}.")
            reset_frame()
        else: dataset.save_frame = True

        dataset.chunk_idx += 1
        print(f"\n✅ Central-pixel Chunk MAE:  {chunk_mae:.6f}")
        print(f"✅ Central-pixel Chunk MAPE: {chunk_mape:.4f}%")

    print(f"\n✅ Central-pixel Global MAE:  {global_mae_sum / max(global_count, 1):.6f}")
    print(f"✅ Central-pixel Global MAPE: {global_mape_sum / max(global_count, 1):.4f}%")



#print(da)

