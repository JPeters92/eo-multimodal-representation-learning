import copy
import datetime

import pandas as pd
import spyndex
import utm
import xarray as xr
from xcube.core.gridmapping import GridMapping

from constants import BANDID_TRANSLATOR, DT_END, DT_START
from version import version


def readin_sites_parameters(
    site_params: pd.Series,
    folder_name: str,
    **kwargs,
) -> dict:
    index = int(site_params["ID"])
    lat = float(site_params["lat"])
    lon = float(site_params["lon"])

    if "size_bbox" in site_params:
        bbox = create_utm_bounding_box(lat, lon, box_size_km=site_params["size_bbox"])
    elif "size_bbox" in kwargs:
        bbox = create_utm_bounding_box(lat, lon, box_size_km=kwargs["size_bbox"])
    else:
        bbox = create_utm_bounding_box(lat, lon)
    if folder_name == "science":
        cube_type = "ScienceCube"
    elif folder_name == "training":
        cube_type = "TrainingCube"
    else:
        raise NameError(
            f"<folder_name> needs to be either 'science' or 'training', but is {folder_name}."
        )
    cube_attrs = dict(
        id=index,
        center_wgs84=bbox["center_wgs84"],
        center_utm=bbox["center_utm"],
        bbox_wgs84=bbox["bbox_wgs84"],
        bbox_utm=bbox["bbox_utm"],
        utm_zone=bbox["utm_zone"],
        version=version,
        creation_datetime=datetime.datetime.now().isoformat(),
        last_modified_datetime=datetime.datetime.now().isoformat(),
        landcover_first=None,
        landcover_first_percentage=None,
        landcover_second=None,
        landcover_second_percentage=None,
        acknowledgment="DeepFeatures project",
        creator_name=["Leipzig University", "Brockmann Consult GmbH"],
        creator_email="info@brockmann-consult.de",
        creator_url=[
            "https://www.uni-leipzig.de/",
            "https://www.brockmann-consult.de/",
        ],
        institution=["Leipzig University", "Brockmann Consult GmbH"],
        project="DeepFeatures",
        cube_type=cube_type,
        Conventions="CF-1.8",
    )
    if "time_range_start" in site_params:
        cube_attrs["time_range_start"] = site_params["time_range_start"]
    elif "time_range_start" in kwargs:
        cube_attrs["time_range_start"] = kwargs["time_range_start"]
    else:
        cube_attrs["time_range_start"] = DT_START
    if "time_range_end" in site_params:
        cube_attrs["time_range_end"] = site_params["time_range_end"]
    elif "time_range_end" in kwargs:
        cube_attrs["time_range_end"] = kwargs["time_range_end"]
    else:
        cube_attrs["time_range_end"] = DT_END

    return cube_attrs


def correct_attrs(cube: xr.Dataset, attrs: dict) -> dict:
    gm = GridMapping.from_dataset(cube)
    zone_number = attrs["utm_zone"][:2]
    zone_letter = attrs["utm_zone"][2]
    cube_zone_number = str(gm.crs.to_epsg())[-2:]
    if zone_number != cube_zone_number:
        attrs["utm_zone"] = cube_zone_number + zone_letter
        attrs["bbox_utm"] = gm.xy_bbox
        attrs["center_utm"] = (
            (gm.xy_bbox[0] + gm.xy_bbox[2]) / 2,
            (gm.xy_bbox[1] + gm.xy_bbox[3]) / 2,
        )
        # transform the bounds to lat lon
        point_west_south = utm.to_latlon(
            attrs["bbox_utm"][0],
            attrs["bbox_utm"][1],
            int(cube_zone_number),
            zone_letter,
        )
        point_east_north = utm.to_latlon(
            attrs["bbox_utm"][2],
            attrs["bbox_utm"][3],
            int(cube_zone_number),
            zone_letter,
        )
        attrs["bbox_wgs84"] = [
            float(point_west_south[1]),
            float(point_west_south[0]),
            float(point_east_north[1]),
            float(point_east_north[0]),
        ]
        attrs["center_wgs84"] = utm.to_latlon(
            attrs["center_utm"][0],
            attrs["center_utm"][1],
            int(cube_zone_number),
            zone_letter,
        )
    return attrs


def create_utm_bounding_box(
    latitude: float, longitude: float, box_size_km: float = 10
) -> dict:
    # Convert WGS84 coordinates to UTM
    easting, northing, zone_number, zone_letter = utm.from_latlon(latitude, longitude)

    # Calculate half the size of the box in meters (5 km in each direction)
    # reduce the half size by half a pixel, because xcube evaluates the values at
    # the center of a pixel. Otherwise we get 1001x1001pixels instead of 1000x1000 pixel.
    box_size_m = box_size_km * 1000
    half_size_m = int(box_size_m / 2)

    # Calculate the coordinates of the bounding box corners, rounded to full meters
    easting_min = int(easting - half_size_m)
    northing_min = int(northing - half_size_m)
    easting_max = easting_min + box_size_m
    northing_max = northing_min + box_size_m

    # transform the bounds to lat lon
    point_west_south = utm.to_latlon(
        easting_min, northing_min, zone_number, zone_letter
    )
    point_east_north = utm.to_latlon(
        easting_max, northing_max, zone_number, zone_letter
    )
    bounding_box = {
        "center_utm": [float(northing), float(easting)],
        "center_wgs84": [latitude, longitude],
        "bbox_utm": [easting_min, northing_min, easting_max, northing_max],
        "bbox_wgs84": [
            float(point_west_south[1]),
            float(point_west_south[0]),
            float(point_east_north[1]),
            float(point_east_north[0]),
        ],
        "utm_zone": f"{zone_number}{zone_letter}",
    }
    return bounding_box


def compute_spectral_indices(cube: xr.Dataset) -> xr.Dataset:
    ds = cube["s2l2a"].to_dataset(dim="band")
    indices = list(spyndex.indices.keys())

    # remove indices which are not provided by Sentinel-2
    for index in indices.copy():
        if "Sentinel-2" not in spyndex.indices.get(index).platforms:
            indices.remove(index)

    # remove index NIRvP, which needs Photosynthetically Available Radiation (PAR)
    # note that PAR is given by Sentinel-3 Level 2
    indices.remove("NIRvP")

    # prepare the parameters for the mapping from expressions to data
    params = {}
    for band in BANDID_TRANSLATOR.keys():
        params[band] = ds[BANDID_TRANSLATOR[band]]

    extra = dict(
        # Kernel parameters
        kNN=1.0,
        kGG=1.0,
        kNR=spyndex.computeKernel(
            kernel="RBF",
            a=ds[BANDID_TRANSLATOR["N"]],
            b=ds[BANDID_TRANSLATOR["R"]],
            sigma=(
                ((ds[BANDID_TRANSLATOR["N"]] + ds[BANDID_TRANSLATOR["R"]]) / 2).median(
                    dim=["y", "x"]
                )
            ),
        ),
        kNB=spyndex.computeKernel(
            kernel="RBF",
            a=ds[BANDID_TRANSLATOR["N"]],
            b=ds[BANDID_TRANSLATOR["B"]],
            sigma=(
                ((ds[BANDID_TRANSLATOR["N"]] + ds[BANDID_TRANSLATOR["B"]]) / 2).median(
                    dim=["y", "x"]
                )
            ),
        ),
        kNL=spyndex.computeKernel(
            kernel="RBF",
            a=ds[BANDID_TRANSLATOR["N"]],
            b=spyndex.constants.L.default,
            sigma=(
                ((ds[BANDID_TRANSLATOR["N"]] + spyndex.constants.L.default) / 2).median(
                    dim=["y", "x"]
                )
            ),
        ),
        kGR=spyndex.computeKernel(
            kernel="RBF",
            a=ds[BANDID_TRANSLATOR["G"]],
            b=ds[BANDID_TRANSLATOR["R"]],
            sigma=(
                ((ds[BANDID_TRANSLATOR["G"]] + ds[BANDID_TRANSLATOR["R"]]) / 2).median(
                    dim=["y", "x"]
                )
            ),
        ),
        kGB=spyndex.computeKernel(
            kernel="RBF",
            a=ds[BANDID_TRANSLATOR["G"]],
            b=ds[BANDID_TRANSLATOR["B"]],
            sigma=(
                ((ds[BANDID_TRANSLATOR["G"]] + ds[BANDID_TRANSLATOR["B"]]) / 2).median(
                    dim=["y", "x"]
                )
            ),
        ),
        # Additional parameters
        L=spyndex.constants.L.default,
        C1=spyndex.constants.C1.default,
        C2=spyndex.constants.C2.default,
        g=spyndex.constants.g.default,
        gamma=spyndex.constants.gamma.default,
        alpha=spyndex.constants.alpha.default,
        sla=spyndex.constants.sla.default,
        slb=spyndex.constants.slb.default,
        nexp=spyndex.constants.nexp.default,
        cexp=spyndex.constants.cexp.default,
        k=spyndex.constants.k.default,
        fdelta=spyndex.constants.fdelta.default,
        epsilon=spyndex.constants.epsilon.default,
        omega=spyndex.constants.omega.default,
        beta=spyndex.constants.beta.default,
        # Wavelength parameters
        lambdaN=spyndex.bands.N.modis.wavelength,
        lambdaG=spyndex.bands.G.modis.wavelength,
        lambdaR=spyndex.bands.R.modis.wavelength,
        lambdaS1=spyndex.bands.S1.modis.wavelength,
    )
    params.update(extra)

    # calculate indices
    cube["spec_indices"] = spyndex.computeIndex(index=indices, params=params)

    return cube


def get_temp_file(attrs: dict) -> str:
    data_id_components = attrs["path"].split("/")
    fname = f"{attrs['id']:06}_s2l2a.zarr"
    return f"{data_id_components[0]}/temp/{'/'.join(data_id_components[1:-1])}/{fname}"


def update_dict(dic: dict, dic_update: dict, inplace: bool = True) -> dict:
    if not inplace:
        dic = copy.deepcopy(dic)
    for key, val in dic_update.items():
        if isinstance(val, dict):
            dic[key] = update_dict(dic.get(key, {}), val)
        else:
            dic[key] = val
    return dic
