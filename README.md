# AI/ML-Driven Glacial Lake Detection & Temporal Change Analysis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![GeoPandas](https://img.shields.io/badge/GeoPandas-0.13+-green.svg)](https://geopandas.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An automated deep learning and geospatial intelligence framework for multi-temporal detection, delineation, and change analysis of high-altitude glacial lakes (e.g., Himalayas, Hindu Kush, Karakoram) to evaluate Glacial Lake Outburst Flood (GLOF) risks.

---

## 🌟 Key Features

- **Multi-Source Geospatial Fusion**: Seamless integration of Sentinel-2 (B2, B3, B4, B8, B11), Sentinel-1 SAR backscatter (VV/VH), and SRTM/CartoDEM topography.
- **Deep Learning Lake Segmentation**: Multi-channel U-Net architecture trained with combined Dice + Focal Loss to accurately segment complex glacial lake water bodies across glacier termini and moraine dams.
- **RGB Fallback Model**: Lightweight inference model capable of processing standard satellite captures (JPEG/PNG) without requiring pre-stacked GeoTIFF channels.
- **Multi-Temporal Change Classification**:
  - 🟡 **Newly Formed Lake**: Detected in Year 2 (e.g. 2022) but absent in Year 1 (e.g. 2016).
  - 🟢 **Survived Lake**: Present across both epochs with quantified surface area delta ($\pm\%$).
  - 🔴 **Drained Lake**: Present in Year 1 but drained or disappeared by Year 2.
- **Interactive Streamlit Web Dashboard**: Real-time file upload, side-by-side epoch comparison, interactive Leaflet/Folium mapping, and instant GIS export (Shapefile, GeoJSON, CSV).
- **Publication-Quality Visualization**: 3-panel layout generator matching scientific literature (Panel a: Satellite Overview, Panel b: Multi-temporal Boundary Overlays, Panel c: Classified Dynamics).

---

## 📁 Repository Structure

```text
├── data/
│   ├── raw/                  # Raw Sentinel-2, Sentinel-1 SAR, DEM rasters
│   ├── reference/            # Ground truth shapefiles (GLIMS, ICIMOD, Hi-MAG)
│   └── processed/            # Derived indices (NDWI, MNDWI), masks & co-registered stacks
├── src/
│   ├── preprocessing/        # Cloud masking, SAR speckle filter, NDWI/MNDWI indices
│   ├── dataset/              # Multi-channel stacking, dynamic 256x256 tiling, normalization
│   ├── segmentation/         # Multi-band & RGB U-Net, Dice+BCE losses, training loop
│   ├── change_detection/     # Spatial polygon matching (IoU + Centroid) and change classifier
│   ├── postprocessing/       # Raster mask to vector polygon conversion & Douglas-Peucker smoothing
│   ├── inference/            # Tiling-and-stitching full-scene inference and boundary overlays
│   └── utils/                # Configuration, CRS transforms, GeoTIFF IO
├── dashboard/
│   ├── app.py                # Main Streamlit web application
│   └── components/           # Upload handler, interactive map viewer, and analytics
├── notebooks/                # Jupyter research notebooks (Preprocessing, Training, Change Detection, Maps)
├── models/                   # Pretrained weights (.pth) & band statistics (.json)
├── outputs/                  # Exported shapefiles, summary tables, and generated maps
├── tests/                    # Unit & integration test suite
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Installation
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the Interactive Dashboard
Launch the Streamlit web application:
```bash
streamlit run dashboard/app.py
```

### 3. Run Pipeline via CLI
- **Generate Sample Data**: `python -m src.dataset.build_input_stack`
- **Run Model Training**: `python -m src.segmentation.train`
- **Run Temporal Change Analysis**: `python -m src.change_detection.classify_change`
- **Run Full Pipeline Inference**: `python -m src.inference.run_inference`

### 4. Run Automated Tests
```bash
python -m unittest discover -s tests
```

---

## 📊 Scientific Methodology

### 1. Spectral Water Indices
- **Normalized Difference Water Index (NDWI)**:
  $$\text{NDWI} = \frac{\text{Green} - \text{NIR}}{\text{Green} + \text{NIR}} = \frac{\text{B03} - \text{B08}}{\text{B03} + \text{B08}}$$
- **Modified Normalized Difference Water Index (MNDWI)**:
  $$\text{MNDWI} = \frac{\text{Green} - \text{SWIR1}}{\text{Green} + \text{SWIR1}} = \frac{\text{B03} - \text{B11}}{\text{B03} + \text{B11}}$$

### 2. Loss Optimization
To counteract extreme class imbalance between glacial lakes and surrounding mountain terrain:
$$\mathcal{L}_{\text{total}} = 0.5 \cdot \mathcal{L}_{\text{Dice}} + 0.5 \cdot \mathcal{L}_{\text{BCE}}$$

---

## 📜 License
This project is licensed under the MIT License.
