# Cube Generation

This module handles the **generation of training data cubes**, including:

- Sampling training sites  
- Retrieving Sentinel-2 data via [CDSE STAC API](https://browser.stac.dataspace.copernicus.eu/collections/sentinel-2-l2a)
- Assembling data cubes with auxiliary information  

---

## 🔐 Credentials Setup

To access data via S3 and CDSE, you need to store your credentials locally.

### S3 Credentials (`s3-credentials.json`)
```json
{
  "bucket": "bucket_name",
  "key": "xxx",
  "secret": "xxx"
}
```

### CDSE Credentials (`cdse-credentials.json`)
```json
key	"xxx"
secret	"xxx"
```

To access Earth Observation data via S3 from the Copernicus Data Space Ecosystem, you
need to generate [Generate S3 credentials](https://documentation.dataspace.copernicus.eu/APIs/S3.html#generate-secrets).  


---

## Workflow Overview

The pipeline consists of three main steps:


### 1. Sample Training Sites

Use [`00_generate_training_sites_table.py`](00_generate_training_sites_table.py)

This script generates a table of randomly sampled training sites.

**Sampling Strategy:**
- Based on block-sampling
- Ensures diversity across:
  - climate zones  
  - environmental ecosystems  
- Bounding box: `[0°, 42°, 30°, 62°]` (west, south, east, north)

- Land cover stratification using ESA CCI Land Cover:

| Land Cover Class | Distribution |
|------------------|-------------|
| Needleleaf       | 20%         |
| Broadleaved      | 20%         |
| Grassland        | 20%         |
| Urban            | 5%          |
| Random (no water)| 35%         |


**Output:**
- `sites_training.csv`

---

### 2. Retrieve Sentinel-2 Data

Use [`01_get_sen2l2a_training_cubes.py`](01_get_sen2l2a_training_cubes.py)

**This script:**
- Reads `sites_traing.csv`  
- Uses `xcube-stac`  
- Accesses the [CDSE STAC API](https://browser.stac.dataspace.copernicus.eu/collections/sentinel-2-l2a)  
- Assembles Sentinel-2 L2A data cubes  

**Output:**

- Sentinel-2 L2A data cubes stored in S3 object storage  

---

### 3.Assemble Training Cubes

Use [`02_build_training_cubes.py`](02_build_training_cubes.py)


**This step:**
- Reads Sentinel-2 Level-2A data cubes from S3 
- Computes cloud masks  
- Merges spectral bands into a single array  
- Rechunks data for performance  

**Final Output:** A processed data cube containing:

- Spectral bands  
- Scene Classification Layer (SCL)  
- Cloud mask  

Stored in S3 object storage.
