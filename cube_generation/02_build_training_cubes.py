import json

import pandas as pd
import segmentation_models_pytorch as smp
import torch
import zarr
from xcube.core.chunk import chunk_dataset
from xcube.core.store import new_data_store

import constants
import get_datasets
import utils
from version import version


def setup_cloudmask_model():
    checkpoint = torch.utils.model_zoo.load_url(constants.CLOUDMASK_MODEL_URL)
    model = smp.Unet(
        encoder_name="mobilenet_v2",
        encoder_weights=None,
        classes=4,
        in_channels=len(constants.CLOUDMASK_BANDS),
    )
    model.load_state_dict(checkpoint)
    model.eval()
    return model


if __name__ == "__main__":
    with open("s3-credentials.json") as f:
        s3_credentials = json.load(f)

    super_store = dict(
        store_team=new_data_store(
            "s3",
            root=s3_credentials["bucket"],
            max_depth=10,
            storage_options=dict(
                anon=False,
                key=s3_credentials["key"],
                secret=s3_credentials["secret"],
            ),
        ),
        cloudmask_model=setup_cloudmask_model(),
    )

    # loop over sites
    sites_params = pd.read_csv(constants.PATH_SITES_PARAMETERS_TRAINING)
    # for idx in range(len(sites_params)):
    for idx in range(1):
        constants.LOG.info(f"Generation of cube {idx} started.")

        path = f"cubes/{constants.TRAINING_FOLDER_NAME}/{version}/{idx:04}.zarr"
        if super_store["store_team"].has_data(path):
            constants.LOG.info(f"Cube {path} already generated.")
            continue

        # get Sentinel-2 data
        cube = get_datasets.get_s2l2a_single_training_year(super_store, idx)
        if cube is None:
            continue
        constants.LOG.info("Open Sentinel-2 L2A.")

        # get attributes of cube
        time_range_start = cube.time.values[0].astype(str)[:-3]
        time_range_end = cube.time.values[-1].astype(str)[:-3]
        site_params = sites_params.iloc[idx]
        attrs = utils.readin_sites_parameters(
            site_params,
            constants.TRAINING_FOLDER_NAME,
            size_bbox=site_params["size_bbox"] * 1000,  # km to meter
            time_range_start=time_range_start,
            time_range_end=time_range_end,
        )
        attrs = utils.correct_attrs(cube, attrs)
        cube.attrs.update(attrs)

        # reorgnaize cube
        cube = get_datasets.reorganize_cube(cube)
        constants.LOG.info("Cube reorgnaized.")

        # add cloud mask
        cube = get_datasets.add_cloudmask(super_store, cube)
        constants.LOG.info("Cloud mask added.")

        # add grid_mapping to encoding
        for var in cube.data_vars:
            if "grid_mapping" in cube[var].attrs:
                del cube[var].attrs["grid_mapping"]
            if "grid_mapping" in cube[var].encoding:
                del cube[var].encoding["grid_mapping"]
            if cube[var].dims[-2:] == ("y", "x"):
                cube[var].attrs["grid_mapping"] = "spatial_ref"
        constants.LOG.info("Grid mapping added to attrs.")

        # write final cube
        cube["band"] = cube.band.astype("str")
        cube.coords["angle"] = ["Zenith", "Azimuth"]
        cube = chunk_dataset(
            cube, chunk_sizes=dict(time=20, x=90, y=90), format_name="zarr"
        )
        compressor = zarr.Blosc(cname="zstd", clevel=5, shuffle=1)
        encoding = {"s2l2a": {"compressor": compressor}}
        super_store["store_team"].write_data(
            cube, path, replace=True, encoding=encoding
        )
        constants.LOG.info(f"Final cube written to {path}.")
