import logging
import os
from pathlib import Path

import spyndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(name)s %(levelname)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("botocore.httpchecksum").setLevel(logging.WARNING)
LOG = logging.getLogger("deepfeatures_cubgen")

DIR = Path(__file__).parent.resolve()
PATH_SITES_PARAMETERS_TRAINING = os.path.join(DIR, "sites_training.csv")

DATA_ID_LAND_COVER_CLASS = "LC-1x2025x2025-2.0.0.levels"
DATA_ID_ERA5 = "cubes/aux/era5.zarr"

CLOUDMASK_MODEL_URL = "https://nextcloud.bgc-jena.mpg.de/s/Ti4aYdHe2m3jBHy/download/mobilenetv2_l2a_rgbnir.pth"
CLOUDMASK_BANDS = ["B02", "B03", "B04", "B8A"]
CLOUDMASK_COORDS = ("time", "band", "y", "x")
CLOUDMASK_SCALE_FACTOR = 2
CLOUDMASK_BATCHSIZE_TIME = 10


DT_START = "2016-11-01"
DT_END = "2024-12-31"
SPATIAL_RES = 10
CHUNKSIZE_TIME = 20
CHUNKSIZE_X = 500
CHUNKSIZE_Y = 500

TRAINING_NB_CUBES = 250
TRAINING_SIZE_BBOX = 0.9
TRAINING_SPATIAL_EXTENT = (0.0, 42.0, 30.0, 62.0)
TRAINING_TEMPORAL_RANGE_DAYS = 366
TRAINING_SPACING_SITES = 1
TRAINING_LANDCOVER_DISTRIBUTION = dict(
    needleleaved=0.2,
    broadleaved=0.2,
    grassland=0.2,
    urban=0.05,
    random=0.35,
)
TRAINING_FOLDER_NAME = "training"
SCIENCE_FOLDER_NAME = "science"

BANDID_TRANSLATOR = {}
for band in spyndex.bands:
    if hasattr(spyndex.bands.get(band), "sentinel2a"):
        s2_band = spyndex.bands.get(band).sentinel2a.band
        s2_band = s2_band[0] + s2_band[1:].zfill(2)
        BANDID_TRANSLATOR[band] = s2_band
