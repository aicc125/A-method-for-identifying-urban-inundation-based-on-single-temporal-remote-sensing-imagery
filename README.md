# A-method-for-identifying-urban-inundation-based-on-single-temporal-remote-sensing-imagery
We created this code to store our proposed code for single temporal urban flood identification.

# Urban Flood Detection from Remote Sensing Imagery

A Python-based urban flood detection system that identifies flood-affected areas in Sentinel-2 satellite imagery by combining spectral water indices, dynamic river/lake exclusion, impervious surface extraction, and proximity-based flood determination.

## Overview

This system implements a multi-step pipeline for detecting urban flooding:

1. **Image Reading** — Reads multi-band Sentinel-2 GeoTIFF with automatic band mapping
2. **Cloud Removal** — Uses SCL (Scene Classification Layer) when available, with a spectral fallback
3. **Vegetation Analysis** — Computes NDVI, FVC, and an adaptive L parameter for SAVI
4. **Spectral Index Computation** — Calculates MNDWI, SAVI, NBI, kNDVI, and a modified IBI (IBI_NBI4)
5. **Water Extraction** — MNDWI-based water detection with Ashman's D bimodality test and adaptive Otsu thresholding
6. **River & Lake Exclusion** — Multi-feature river scoring (skeleton ratio, width consistency, spectral stability, boundary touch, sinuosity, aspect ratio) with data-driven threshold selection
7. **Impervious Surface Extraction** — IBI_NBI4-based built-up area detection with texture refinement
8. **Flood Determination** — Proximity-based classification: water bodies near impervious surfaces are flagged as flood
9. **Visualization** — 9-panel diagnostic figure output

## Key Features

- **Bimodality-aware thresholding**: Ashman's D test validates whether Otsu's method is appropriate before applying it, with conservative fallback thresholds for non-bimodal scenes
- **Dynamic river scoring**: A weighted composite score (0–1) combining six morphological and spectral features replaces simple aspect-ratio-based river detection
- **Small sample protection** (v7): When river score samples are fewer than a configurable threshold, the system switches from natural-break/IQR methods to a robust median-based approach
- **Sensitivity analysis**: Automatic river threshold sensitivity report showing the impact of different thresholds on exclusion counts

## Input Data

The script expects a **Sentinel-2 Level-2A GeoTIFF** with at least 12 bands. The default band mapping is:

| Band Name | Band Index | Sentinel-2 Band |
|-----------|-----------|-----------------|
| aerosol   | 1         | B1              |
| blue      | 2         | B2              |
| green     | 3         | B3              |
| red       | 4         | B4              |
| nir       | 8         | B8              |
| swir1     | 11        | B11             |
| swir2     | 12        | B12             |
| scl       | 15        | SCL (optional)  |

Band values should be in DN (divided by 10000 internally to get reflectance). If your band order differs, modify the `BAND_MAP` dictionary at the top of the script.

## Project Structure
```
.
├── README.md
├── flood_detection.py     # Main detection script
├── data/                  # Input imagery (user-provided)
│   └── input.tif
└── output/                # Detection results (auto-created)
    └── flood/
```
The recognition results are as follows:
<img width="10155" height="7855" alt="fig8" src="https://github.com/user-attachments/assets/c47e1f19-b74b-4314-bcf8-d366cb9382e4" />



## License

This project is provided for academic and research purposes.
