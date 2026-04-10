import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from xcube.core.maskset import MaskSet
from xcube.core.store import new_data_store

import constants


def generate_time_interval() -> tuple[tuple[str, str], tuple[str, str]]:
    dt_start = datetime.date.fromisoformat(constants.DT_START)
    dt_end = datetime.date.fromisoformat(constants.DT_END)
    nb_days = (dt_end - dt_start).days
    half_year = constants.TRAINING_TEMPORAL_RANGE_DAYS // 2
    delta_days0 = np.random.randint(half_year, nb_days - half_year)
    if delta_days0 < nb_days / 2:
        delta_days1 = np.random.randint(
            delta_days0 + 3 * half_year, nb_days - half_year
        )
    else:
        delta_days1 = np.random.randint(half_year, delta_days0 - 3 * half_year)

    dt0 = dt_start + datetime.timedelta(days=delta_days0)
    dt1 = dt_start + datetime.timedelta(days=delta_days1)
    td = datetime.timedelta(days=half_year)
    return (
        ((dt0 - td).isoformat(), (dt0 + td).isoformat()),
        ((dt1 - td).isoformat(), (dt1 + td).isoformat()),
    )


def get_central_point(
    lc_masked: xr.Dataset,
    center_points: np.ndarray,
    idxs_nonnan: tuple[np.ndarray],
) -> np.ndarray:

    while True:
        isel = np.random.randint(0, len(idxs_nonnan[0]))
        lat_random = lc_masked.lat[idxs_nonnan[0][isel]].values
        lon_random = lc_masked.lon[idxs_nonnan[1][isel]].values

        # test so that the selected training cube site is not close to other cubes
        if center_points.size > 0:
            diff = np.sqrt(
                (center_points[:, 0] - lat_random) ** 2
                + (center_points[:, 1] - lon_random) ** 2
            )
            if np.any(diff < constants.TRAINING_SPACING_SITES):
                continue

        # otherwise break
        break

    if center_points.size > 0:
        return np.concatenate(
            [center_points, np.array([[lat_random, lon_random]])], axis=0
        )
    else:
        return np.array([[lat_random, lon_random]])


if __name__ == "__main__":
    _DIR = Path(__file__).parent.resolve()
    store_lccs = new_data_store("s3", root="deep-esdl-public")
    mlds_lc = store_lccs.open_data(constants.DATA_ID_LAND_COVER_CLASS)
    lc = mlds_lc.base_dataset
    bbox = constants.TRAINING_SPATIAL_EXTENT
    lc = lc.sel(
        time=datetime.datetime(2022, 1, 1),
        lat=slice(bbox[3], bbox[1]),
        lon=slice(bbox[0], bbox[2]),
    )
    lc_mask = MaskSet(lc.lccs_class)
    center_points = np.array([[]], dtype=float)
    lc_dist = constants.TRAINING_LANDCOVER_DISTRIBUTION

    # generate site definition for mini-cubes
    table = pd.DataFrame(
        data=np.full((constants.TRAINING_NB_CUBES, 7), np.nan),
        columns=[
            "lat",
            "lon",
            "time_range_start",
            "time_range_end",
            "size_bbox",
            "land_cover_value",
            "land_cover_label",
        ],
    )
    idx = 0
    for lc_label, fraction in lc_dist.items():
        if lc_label == "random":
            lc_masked = ~lc_mask["water"].astype(bool)
        else:
            keys = [key for key in lc_mask.keys() if lc_label in key]
            if not keys:
                print("error")
            lc_masked = lc_mask[keys[0]].astype(bool)
            for key in keys[1:]:
                lc_masked |= lc_mask[key].astype(bool)
        idxs_nonnan = np.where(lc_masked.values)
        nb_sites = round(fraction * constants.TRAINING_NB_CUBES)
        while nb_sites > 0:
            constants.LOG.info(f"Generation of mini-cube attributes {idx}.")
            time_ranges = generate_time_interval()
            center_points = get_central_point(lc_masked, center_points, idxs_nonnan)
            # add two cubes (2 individual years) to the table
            for time_range in time_ranges:
                table.loc[idx, "lat"] = center_points[-1, 0]
                table.loc[idx, "lon"] = center_points[-1, 1]
                table.loc[idx, "time_range_start"] = time_range[0]
                table.loc[idx, "time_range_end"] = time_range[1]
                table.loc[idx, "size_bbox"] = constants.TRAINING_SIZE_BBOX
                table.loc[idx, "land_cover_value"] = lc.lccs_class.sel(
                    lat=center_points[-1, 0], lon=center_points[-1, 1]
                ).values
                flag_idx = np.where(
                    lc.lccs_class.attrs["flag_values"]
                    == table.loc[idx, "land_cover_value"]
                )[0]
                assert flag_idx.size == 1
                label = lc.lccs_class.attrs["flag_meanings"].split(" ")[flag_idx[0]]
                table.loc[idx, "land_cover_label"] = label
                idx += 1
            nb_sites -= 1

    table.to_csv(os.path.join(_DIR, constants.PATH_SITES_PARAMETERS_TRAINING))
