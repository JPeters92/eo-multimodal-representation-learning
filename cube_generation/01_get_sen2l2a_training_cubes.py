import json

import pandas as pd
from xcube.core.store import new_data_store

import constants
from version import version


def get_s2l2a(super_store: dict, site_params: pd.Series):
    data_id = f"cubes/temp/{constants.TRAINING_FOLDER_NAME}/{version}/{idx:04}.zarr"

    def _get_s2l2a_year(time_range: list[str], data_id: str):
        if not super_store["store_team"].has_data(data_id):
            constants.LOG.info(f"Open cube {idx} for year {time_range[1][:4]}.")
            ds = super_store["store_stac"].open_data(
                data_id="sentinel-2-l2a",
                point=(site_params["lon"], site_params["lat"]),
                bbox_width=site_params["size_bbox"] * 1000,  # km to meter
                spatial_res=constants.SPATIAL_RES,
                time_range=time_range,
                apply_scaling=True,
                asset_names=[
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
                    "SCL",
                ],
            )
            print(ds)
            constants.LOG.info(f"Writing of cube to {data_id} started.")
            super_store["store_team"].write_data(ds, data_id, replace=True)
            constants.LOG.info(f"Writing of cube to {data_id} finished.")
        else:
            constants.LOG.info(f"Cube {data_id} already retrieved.")

    time_range = [site_params["time_range_start"], site_params["time_range_end"]]
    for attempt in range(1, 4):
        try:
            _get_s2l2a_year(time_range, data_id)
            break
        except Exception as e:
            if super_store["store_team"].has_data(data_id):
                super_store["store_team"].delete_data(data_id)
            constants.LOG.error(f"Attempt {attempt} failed: {e}")
            if attempt == 3:
                constants.LOG.info(
                    f"Cube {data_id} tried to retrieve {attempt} times. " f"We go on..."
                )


if __name__ == "__main__":
    with open("s3-credentials.json") as f:
        s3_credentials = json.load(f)
    with open("cdse-credentials.json") as f:
        cdse_credentails = json.load(f)

    super_store = dict(
        store_stac=new_data_store("stac-cdse-ardc", **cdse_credentails),
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
    )

    sites_params = pd.read_csv(constants.PATH_SITES_PARAMETERS_TRAINING)
    # for idx in range(len(sites_params)):
    for idx in range(1):
        constants.LOG.info(f"Generation of cube {idx} started.")
        site_params = sites_params.loc[idx]
        get_s2l2a(super_store, site_params)
