import os
import sys
import multiprocessing

# Prevent resource_tracker crashes/warning popups on macOS under Python 3.14+
if sys.platform == "darwin":
    try:
        multiprocessing.set_start_method("fork", force=True)
    except Exception:
        pass

import streamlit as st
import geopandas as gpd
import pandas as pd
from pathlib import Path

# Resolve project root path and append to sys.path to permit 'src' package imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import Config
from src.logger import setup_logger
from src.inference import SoilPredictor
from src.recommender import CropSuitabilityEngine, FertilizerRecommender

logger = setup_logger("dashboard", level="INFO")

# Page config - MUST be called first
st.set_page_config(
    page_title="TerraSense AI - Soil Intelligence Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS styling (applied globally)
st.markdown("""
<style>
    /* Google Font Import */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Metric styling */
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 800;
        color: #2E7D32;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 0.95rem;
        color: #555;
        font-weight: 600;
    }
    
    /* Card design */
    .metric-card {
        background-color: #F5F7FA;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border-left: 5px solid #2E7D32;
        margin-bottom: 20px;
        border-top: 1px solid rgba(255,255,255,0.7);
    }
    
    .glass-card {
        background: rgba(245, 245, 245, 0.95);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 30px;
        border: 1px solid rgba(255, 255, 255, 0.6);
        box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.12);
        color: #2C3E50;
    }
    .glass-card h3 {
        color: #1B5E20;
        font-weight: 800;
        margin-top: 0;
        margin-bottom: 15px;
    }
    .glass-card p {
        color: #34495E;
        line-height: 1.6;
        font-size: 1.05rem;
    }
    .glass-card li {
        color: #34495E;
        margin-bottom: 8px;
        line-height: 1.5;
    }
    .glass-card li b {
        color: #1B5E20;
    }
    
    .title-highlight {
        background: linear-gradient(120deg, #2E7D32, #4CAF50);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- CACHED RESOURCES & DATA -----------------

@st.cache_resource
def get_config() -> Config:
    return Config()

@st.cache_resource
def get_predictor(_config: Config) -> SoilPredictor:
    return SoilPredictor(_config)

@st.cache_resource
def get_recommenders(_config: Config) -> tuple:
    suitability = CropSuitabilityEngine(_config)
    recommender = FertilizerRecommender(_config)
    return suitability, recommender

@st.cache_data
def load_gis_data(_config: Config) -> tuple:
    raw_dir = Path(_config.paths["raw_data_dir"])
    parcels_path = raw_dir / "parcels.geojson"
    lab_tests_path = raw_dir / "lab_soil_tests.csv"
    processed_path = Path(_config.paths["processed_data_dir"]) / "aligned_soil_dataset.csv"
    
    parcels_df = gpd.read_file(parcels_path)
    lab_df = pd.read_csv(lab_tests_path)
    aligned_df = pd.read_csv(processed_path)
    
    return parcels_df, lab_df, aligned_df

# ----------------- LOAD & SET GLOBAL STATE -----------------

config = get_config()
predictor = get_predictor(config)
suitability_eng, recommender_eng = get_recommenders(config)

try:
    parcels, lab_samples, aligned_data = load_gis_data(config)
except Exception as e:
    st.error(f"Error loading GIS data: {e}. Make sure to run the training pipeline first.")
    st.stop()

# Initialize session state variables
if "selected_parcel_id" not in st.session_state:
    st.session_state["selected_parcel_id"] = "UNASSIGNED"

if "selected_point" not in st.session_state:
    # Default focus point: Guntur, Andhra Pradesh (Latitude: 16.3008, Longitude: 80.4428)
    st.session_state["selected_point"] = (16.3008, 80.4428)

if "active_crop" not in st.session_state:
    st.session_state["active_crop"] = "rice"

if "map_zoom" not in st.session_state:
    st.session_state["map_zoom"] = 11

if "location_metadata" not in st.session_state:
    from src.utils import reverse_geocode
    lat, lon = st.session_state["selected_point"]
    st.session_state["location_metadata"] = reverse_geocode(lat, lon)

# ----------------- SIDEBAR BRANDING & SEARCH -----------------

st.sidebar.markdown(
    """
    <div style="text-align: center; margin-bottom: 25px;">
        <h2 style='margin-bottom:0px;'><span class='title-highlight'>TerraSense AI</span></h2>
        <small style='color:#777; font-weight:600;'>India Precision Agriculture Platform</small>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("### 🗺️ India Location Search")

search_query = st.sidebar.text_input(
    "Search Village, Mandal, District, State",
    placeholder="e.g. Vemulapadu, Guntur",
    key="global_location_search_input"
)
search_btn = st.sidebar.button("Search Location", use_container_width=True)

if search_btn and search_query:
    from src.utils import geocode_location, reverse_geocode
    with st.spinner(f"Geocoding '{search_query}'..."):
        coords = geocode_location(search_query)
        if coords:
            st.session_state["selected_point"] = (coords["lat"], coords["lon"])
            st.session_state["location_metadata"] = reverse_geocode(coords["lat"], coords["lon"])
            st.session_state["map_zoom"] = 13
            st.session_state["just_searched"] = True
            # Force reset predictions so they recalculate for the new point
            if "point_predictions" in st.session_state:
                del st.session_state["point_predictions"]
            st.sidebar.success(f"📍 Zoomed to {coords['display_name'][:35]}...")
            st.rerun()
        else:
            st.sidebar.error("Location not found. Try spelling it differently or add State/India.")

# Display active location metadata
loc_meta = st.session_state["location_metadata"]
focus_states = ["Andhra Pradesh", "Telangana", "Karnataka", "Tamil Nadu"]
is_focus = loc_meta.get("state") in focus_states

st.sidebar.markdown(
    f"""
    <div class="metric-card" style="padding: 14px; margin-top: 15px; border-left-color: {'#2E7D32' if is_focus else '#757575'};">
        <span style="font-size: 0.75rem; font-weight: 700; color: #555; text-transform: uppercase;">Active Location</span>
        <div style="font-weight: 800; color: {'#1B5E20' if is_focus else '#333'}; font-size: 1.1rem; margin-top: 3px;">
            📍 {loc_meta['village']}
        </div>
        <div style="font-size: 0.85rem; color: #444; margin-top: 4px; line-height: 1.4;">
            <b>Mandal:</b> {loc_meta['mandal']}<br>
            <b>District:</b> {loc_meta['district']}<br>
            <b>State:</b> {loc_meta['state']}
        </div>
        <div style="font-size: 0.72rem; color: #777; margin-top: 6px; font-family: monospace;">
            {st.session_state['selected_point'][0]:.4f}°N, {st.session_state['selected_point'][1]:.4f}°E
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

if is_focus:
    st.sidebar.markdown(
        """<div style='background-color: #E8F5E9; color: #2E7D32; font-size: 0.75rem; font-weight: 700; padding: 6px 10px; border-radius: 8px; text-align: center; border: 1px solid #C8E6C9; margin-bottom: 15px;'>
        🛡️ Focus Region Calibration Active
        </div>""",
        unsafe_allow_html=True
    )

# Expose assets to session state for subpages
st.session_state["config"] = config
st.session_state["predictor"] = predictor
st.session_state["suitability_eng"] = suitability_eng
st.session_state["recommender_eng"] = recommender_eng
st.session_state["parcels"] = parcels
st.session_state["lab_samples"] = lab_samples
st.session_state["aligned_data"] = aligned_data

# ----------------- NAVIGATION ROUTING -----------------

current_dir = Path(__file__).resolve().parent

pg = st.navigation([
    st.Page(str(current_dir / "pages" / "home.py"), title="Home Overview", icon="🏠"),
    st.Page(str(current_dir / "pages" / "parcel_map.py"), title="1. Parcel Map", icon="🗺️"),
    st.Page(str(current_dir / "pages" / "nutrient_heatmaps.py"), title="2. Nutrient Heatmaps", icon="🔥"),
    st.Page(str(current_dir / "pages" / "soil_health.py"), title="3. Soil Health Summary", icon="🩺"),
    st.Page(str(current_dir / "pages" / "confidence_analysis.py"), title="4. Confidence Analysis", icon="🛡️"),
    st.Page(str(current_dir / "pages" / "crop_recommendation.py"), title="5. Crop Recommendation", icon="🌾"),
    st.Page(str(current_dir / "pages" / "fertilizer_advisory.py"), title="6. Fertilizer Advisory", icon="🧪"),
    st.Page(str(current_dir / "pages" / "model_performance.py"), title="7. Model Performance", icon="📈")
])

pg.run()
