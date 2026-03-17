# Dataset Module

This module contains the dataset preparation pipeline for the Sentinel-based training data used by the main model training workflow.

### 1. Spectral preparation

- create the Sentinel-2 `.zarr` cubes before running this module

### 2. Sentinel preprocessing and patch extraction

Use [`preprocess_sentinel.py`](preprocess_sentinel.py).

Purpose:
- accesses the Planetary Computer to retrieve the corresponding Sentinel-1 data for existing Sentinel-2 cubes
- matches Sentinel-1 to the Sentinel-2 cubes in space and time
- extracts training patches from Sentinel-1/2 cubes
- preprocessing utilities used by [`train_dataset.py`](train_dataset.py)

### 3. Spectral preparation

Use [`prepare_dataarray.py`](prepare_dataarray.py).

Purpose:
- prepares Sentinel band inputs
- normalizes variables using configured min/max ranges
- returns band stacks in a model-ready format
- sed inside the preprocessing and dataset-building pipeline

### 4. Training dataset creation

Use [`train_dataset.py`](train_dataset.py).

Purpose:
- loads the Sentinel-2 cubes
- creates aligned Sentinel-1/2 training inputs through the preprocessing pipeline
- builds train, validation, or test datasets
- writes the resulting datasets to HDF5
- Outputs: HDF5 files containing training tensors, masks, and time-gap information

### 5. Model dataloading

Use [`dataloader.py`](dataloader.py).

Purpose:
- loads the generated HDF5 datasets
- exposes them as PyTorch datasets
- returns tensors, masks, and temporal metadata for training

#### Notes

- [`train_dataset.py`](train_dataset.py) is the main entry point for dataset creation.
- [`dataloader.py`](dataloader.py) is the main entry point for consuming the generated HDF5 datasets in training.
- the exact source and output paths are configured inside the scripts and may differ between machines.

### 6. Dataset utilities

Use [`utils.py`](utils.py).

Purpose:
- provides helper functions for filtering samples
- computes time gaps and extracts center coordinates
- supports timestamp selection and concatenation utilities used during dataset creation