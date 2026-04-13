import numpy as np
import torch
import xarray as xr

import constants
from version import version


def get_s2l2a_single_training_year(
    super_store: dict, idx: int, check_nan: bool = True
) -> xr.Dataset | None:
    data_id = f"cubes/temp/{constants.TRAINING_FOLDER_NAME}/{version}/{idx:04}.zarr"
    if not super_store["store_team"].has_data(data_id):
        constants.LOG.info(
            f"Dataset with data ID {data_id} does not exists. We "
            f"discard the data cube generation for {data_id} for now."
        )
        return None
    ds = super_store["store_team"].open_data(data_id)
    if check_nan:
        constants.LOG.info(f"Check dataset with data ID {data_id} for nan values")
        threshold = 10
        exceeded = assert_dataset_nan(ds, threshold, no_angles=True)
        if exceeded:
            return None
    ds = ds.isel(x=slice(0, 90), y=slice(0, 90))

    # add attributes
    xcube_stac_attrs = {}
    xcube_stac_attrs["source"] = (
        "https://documentation.dataspace.copernicus.eu/APIs/S3.html"
    )
    xcube_stac_attrs["institution"] = "Copernicus Data Space Ecosystem"
    xcube_stac_attrs["standard_name"] = "sentinel2_l2a"
    xcube_stac_attrs["long_name"] = "Sentinel-2 L2A prduct"
    xcube_stac_attrs["stac_catalog_url"] = ds.attrs["stac_catalog_url"]
    xcube_stac_attrs["stac_item_ids"] = ds.attrs["stac_item_ids"]

    ds.attrs = dict()
    ds.attrs["xcube_stac_attrs"] = xcube_stac_attrs
    ds.attrs["affine_transform"] = ds.rio.transform()
    return ds


def reorganize_cube(ds: xr.Dataset) -> xr.Dataset:
    scl = ds.SCL.astype(np.uint8)
    ds = ds.drop_vars(["SCL"])
    s2l2a = ds.to_dataarray(dim="band").astype(np.float32)
    s2l2a = s2l2a.sel(
        band=[
            "B01",
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B09",
            "B11",
            "B12",
        ]
    )
    s2l2a = s2l2a.chunk(
        chunks=dict(
            band=s2l2a.sizes["band"],
            time=constants.CHUNKSIZE_TIME,
            x=constants.CHUNKSIZE_X,
            y=constants.CHUNKSIZE_Y,
        )
    )
    scl = scl.chunk(
        chunks=dict(
            time=constants.CHUNKSIZE_TIME,
            x=constants.CHUNKSIZE_X,
            y=constants.CHUNKSIZE_Y,
        )
    )
    cube = xr.Dataset()
    cube_attrs = ds.attrs
    sen2_attrs = cube_attrs.pop("xcube_stac_attrs")
    cube["s2l2a"] = s2l2a
    cube["s2l2a"].attrs = sen2_attrs
    cube["scl"] = scl
    sen2_attrs.update(
        dict(
            description="Scene classification layer of the Sentinel-2 L2A product.",
            flag_values=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            flag_meanings=(
                "no_data saturated_or_defective_pixel topographic_casted_shadows "
                "cloud_shadows vegetation not_vegetation water "
                "unclassified cloud_medium_probability "
                "cloud_high_probability thin_cirrus snow_or_ice"
            ),
            flag_colors=(
                "#000000 #ff0000 #2f2f2f #643200 #00a000 #ffe65a #0000ff "
                "#808080 #c0c0c0 #ffffff #64c8ff #ff96ff"
            ),
        )
    )
    cube["scl"].attrs = sen2_attrs
    cube.attrs = cube_attrs

    return cube


def add_cloudmask(super_store: dict, cube: xr.Dataset) -> xr.Dataset:
    s2l2a = cube.s2l2a
    s2l2a = s2l2a.sel(band=constants.CLOUDMASK_BANDS)
    s2l2a = s2l2a.transpose(*constants.CLOUDMASK_COORDS)

    dataarrays_to_merge = []
    for time_step in range(0, s2l2a.sizes["time"], constants.CLOUDMASK_BATCHSIZE_TIME):
        s2l2a_sub = s2l2a.isel(
            time=slice(time_step, time_step + constants.CLOUDMASK_BATCHSIZE_TIME)
        )
        res = _compute_earthnet_cloudmask(super_store, s2l2a_sub)
        dataarrays_to_merge.append(res)
    cloud_mask = xr.concat(dataarrays_to_merge, dim="time")
    cloud_mask = cloud_mask.chunk(
        chunks=dict(
            time=constants.CHUNKSIZE_TIME,
            x=constants.CHUNKSIZE_X,
            y=constants.CHUNKSIZE_Y,
        )
    )
    cube["cloud_mask"] = cloud_mask
    cube["cloud_mask"].attrs = dict(
        standard_name="cloudmask",
        long_name=(
            "Cloudmask generated using an AI approach following the implementation "
            "of EarthNet Minicuber, based on CloudSEN12."
        ),
        source="https://github.com/earthnet2021/earthnet-minicuber",
        institution="https://cloudsen12.github.io/",
        flag_values=[0, 1, 2, 3],
        flag_meanings="clear thick_cloud thin_cloud cloud_shadow",
        flag_colors="#000000 #FFFFFF #D3D3D3 #636363",
    )
    return cube


def get_cloudmask(super_store: dict, cube: xr.Dataset) -> xr.Dataset:
    data_id = f"cubes/temp/{constants.SCIENCE_FOLDER_NAME}/{version}/{cube.attrs['id']:03}_cloudmask.zarr"
    if not super_store["store_team"].has_data(data_id):
        constants.LOG.info(
            f"Cloud mask with data ID {data_id} does not exists. We "
            f"discard the data cube generation for now."
        )
        return None
    cube["cloud_mask"] = super_store["store_team"].open_data(data_id)["cloud_mask"]
    return cube


def _compute_earthnet_cloudmask(super_store: dict, da: xr.DataArray):
    x = torch.from_numpy(da.fillna(1.0).values.astype("float32"))
    b, c, h, w = x.shape

    h_big = (h // 32 + 1) * 32
    h_pad_left = (h_big - h) // 2
    h_pad_right = ((h_big - h) + 1) // 2

    w_big = (w // 32 + 1) * 32
    w_pad_left = (w_big - w) // 2
    w_pad_right = ((w_big - w) + 1) // 2

    x = torch.nn.functional.pad(
        x, (w_pad_left, w_pad_right, h_pad_left, h_pad_right), mode="reflect"
    )
    x = torch.nn.functional.interpolate(
        x, scale_factor=constants.CLOUDMASK_SCALE_FACTOR, mode="bilinear"
    )
    with torch.no_grad():
        y_hat = super_store["cloudmask_model"](x)
    y_hat = torch.argmax(y_hat, dim=1).float()
    y_hat = torch.nn.functional.max_pool2d(y_hat[:, None, ...], kernel_size=2)[
        :, 0, ...
    ]
    y_hat = y_hat[:, h_pad_left:-h_pad_right, w_pad_left:-w_pad_right]

    return xr.DataArray(
        y_hat.cpu().numpy().astype("uint8"),
        dims=("time", "y", "x"),
        coords=dict(time=da.coords["time"], y=da.coords["y"], x=da.coords["x"]),
    )


def assert_dataset_nan(
    ds: xr.Dataset, threshold: float | int, no_angles: bool = False
) -> bool:
    exceeded = False
    for key in list(ds.keys()):
        if no_angles and key in ["solar_angle", "viewing_angle"]:
            continue
        array = ds[key].values.ravel()
        null_size = array[np.isnan(array)].size
        perc = (null_size / array.size) * 100
        if perc > threshold:
            constants.LOG.info(f"Data variable {key} has {perc:.3f}% nan values.")
            exceeded = True
            break
    return exceeded
