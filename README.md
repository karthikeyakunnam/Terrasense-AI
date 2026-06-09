# 🌱 TerraSense AI: AI-Powered Soil Intelligence Platform

TerraSense AI is a production-grade, portfolio-ready agricultural intelligence platform that integrates spatial satellite imagery (Sentinel-2), gridded soil properties (SoilGrids), and localized physical laboratory soil tests to generate sub-meter parcel-level nutrient maps and crop-specific fertilizer advisories. 

The platform leverages **Ensemble Tree Variance** for predictive uncertainty quantification, maps soil health status, evaluates crop suitability (Rice, Wheat, Cotton, Maize), and provides an interactive dashboard for decision support.

---

## 🚀 Key Features

1.  **Geospatial Data Pipeline**: Ingests and warps coordinate reference systems (CRS), aligns multi-spectral satellite imagery and SoilGrids layers, and samples raster values at precise latitude/longitude point coordinates.
2.  **Dynamic Feature Engineering**: Computes key vegetation indices (NDVI, EVI, NDRE) from raw spectral bands and normalizes soil parameters.
3.  **Multi-Target Machine Learning**: Evaluates and compares Random Forest, Gradient Boosting, and HistGradientBoosting regressors for five soil variables: Nitrogen (N), Phosphorus (P), Potassium (K), Soil Organic Carbon (SOC), and Soil pH.
4.  **Uncertainty Quantification**: Calculates prediction standard deviation across individual trees in the Random Forest ensemble, translating variance into a normalized **Confidence Score (0-100%)** and 90% Prediction Intervals.
5.  **Precision Fertilizer Advisory**: Formulates exact chemical fertilizer requirements (Urea, DAP, MOP) based on soil density-to-area stocks ($mg/kg \to kg/ha$) and adjusts calculations for nutrient overlap (e.g. DAP N-content correction).
6.  **Soil Amendments**: Recommends Agricultural Lime or Gypsum based on critical pH limits.
7.  **Interactive Web Dashboard**: A multi-page Streamlit portal displaying interactive Folium parcel maps, Plotly soil diagnostic gauges, active-sampling priority tables, suitability comparisons, and ML performance metrics.

---

## 🛠️ Tech Stack

*   **Language**: Python
*   **Geospatial Processing**: Rasterio, GeoPandas, Shapely
*   **Data Science & Machine Learning**: Pandas, NumPy, Scikit-Learn, Joblib, XGBoost (optional)
*   **Data Visualization**: Plotly, Folium, Streamlit-Folium
*   **Dashboard Framework**: Streamlit
*   **Testing**: Pytest

---

## 📁 Repository Structure

```
/Users/karthikeyaunnam/Documents/TerraSense AI/
├── config/
│   └── config.yaml             # System configurations (agronomy targets, ML model parameters)
├── data/
│   ├── raw/                    # Raw spatial rasters, parcels, and lab test points
│   └── processed/              # Merged tabular dataset ready for ML training
├── models/                     # Serialized best models (.joblib) and metric JSON files
├── logs/                       # Rotating application logs
├── src/                        # Modular package source files
│   ├── __init__.py
│   ├── config.py               # YAML configuration loader
│   ├── logger.py               # console and rotating file logger setup
│   ├── utils.py                # CRS conversion, math and confidence scaling helpers
│   ├── data_generator.py       # Spatial synthetic raster and lab test data generator
│   ├── data_pipeline.py        # Spatial query and ingestion engine
│   ├── features.py             # Sentinel-2 indices engineering and scaling
│   ├── train.py                # CV ML training and metric selection pipeline
│   ├── inference.py            # Predictor loader with ensemble variance calculations
│   └── recommender.py          # Agronomic crop suitability and fertilizer split-dosages
├── dashboard/                  # Streamlit application portal
│   ├── app.py                  # Main entry point and landing page
│   └── pages/                  # Subpages automatically loaded by Streamlit
│       ├── 01_parcel_map.py    # Folium map & coordinate point querying
│       ├── 02_nutrient_heatmaps.py # Plotly contour prediction map overlays
│       ├── 03_soil_health.py   # Diagnostics, organic matter index, macronutrients balance
│       ├── 04_confidence.py    # Uncertainty UQ maps & active sampling guides
│       ├── 05_suitability.py   # Crop suitability rating evaluations
│       ├── 06_advisory.py      # Split-application dosages & what-if simulators
│       └── 07_performance.py   # ML validation metric details & feature importances
├── tests/                      # Pytest unit testing suite
│   ├── __init__.py
│   └── test_pipelines.py
├── docs/                       # Technical documentation
│   └── interview_prep.md       # Complete system design & 30 interview questions
├── requirements.txt            # Python dependencies
└── .env.example                # Template for environment variables
```

---

## ⚡ Quick Start

### 1. Installation
Clone this repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Generate Data & Run Spatial Pipeline
To generate the high-fidelity mock grids (Sentinel-2 and SoilGrids bands), sample them at the lab coordinates, and output the processed dataset, run:
```bash
PYTHONPATH=. python3 src/data_pipeline.py
```
*(Note: If files in `data/raw/` are missing, the pipeline will auto-generate them automatically).*

### 3. Train ML Models
To run cross-validation comparisons (RF, GBDT, HistGBDT) for all five soil targets, select the best models, and serialize the checkpoints, run:
```bash
PYTHONPATH=. python3 src/train.py
```

### 4. Run Unit Tests
To run the automated validation tests confirming config loaders, coordinate round-tripping, vegetation index calculations, and agronomic split calculations are working, run:
```bash
PYTHONPATH=. pytest tests/test_pipelines.py
```

### 5. Launch the Streamlit Dashboard
To launch the interactive soil intelligence web portal, run:
```bash
streamlit run dashboard/app.py
```

---

## 🧠 Core Systems & Methodology

### A. Ensemble Uncertainty Quantification
Model disagreement represents **epistemic uncertainty**. For each target:
1.  We extract predictions from all 100 individual decision trees in the Random Forest ensemble: $f_b(x)$ for $b \in \{1..100\}$.
2.  We compute the standard deviation: $\sigma(x) = \sqrt{\text{Var}(f_b(x))}$.
3.  We map this standard deviation to a 0-100% confidence rating calibrated against validation RMSE:
    $$\text{Confidence}(x) = e^{-\frac{\sigma(x)}{\text{RMSE}_{\text{val}}}} \times 100$$
4.  We construct the 90% Prediction Interval: $[\hat{y} - 1.645 \cdot \sigma(x), \hat{y} + 1.645 \cdot \sigma(x)]$.

### B. Precision Fertilizer Calculations
1.  **Concentration to Area Stock ($mg/kg \to kg/ha$)**:
    $$\text{Available Stock (kg/ha)} = \text{Soil Test (mg/kg)} \times \text{Bulk Density (g/cm}^3) \times \text{Depth (0-15cm)} \times 0.1 \times 10$$
2.  **Deficit Correction**:
    $$\text{Deficit} = \max(0, \text{Target Requirement} - \text{Available Stock})$$
3.  **Overlap Adjustment (Nitrogen & Phosphorus)**:
    $$\text{DAP Rate} = \frac{P_{\text{deficit}}}{0.46}$$
    $$\text{Urea Rate} = \max\left(0, \frac{N_{\text{deficit}} - (0.18 \times \text{DAP Rate})}{0.46}\right)$$
4.  **Split Scheduling**: Nitrogen requirements are split across key growth stages (e.g. 50% basal, 25% tillering, 25% panicle for Rice) to minimize leaching. DAP and Potassium (MOP) are applied 100% basal due to soil immobility.

---

## 🎓 Technical Interview Ready
A comprehensive system design document explaining the commercial business case, tech stack decisions, and **30 highly detailed senior-level interview questions and answers** is available in the [docs/interview_prep.md](file:///Users/karthikeyaunnam/Documents/TerraSense%20AI/docs/interview_prep.md) file.
