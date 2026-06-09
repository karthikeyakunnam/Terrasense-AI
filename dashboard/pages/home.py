import streamlit as st
import pandas as pd
from pathlib import Path

# Verify credentials / variables are loaded
config = st.session_state.get("config")
lab_samples = st.session_state.get("lab_samples")

if config is None or lab_samples is None:
    st.warning("Please visit the main dashboard page to initialize data.")
    st.stop()

st.markdown("# 🌱 Welcome to **TerraSense AI**")
st.markdown("##### Production-Grade India-Wide Precision Agriculture Decision Support Platform")

st.markdown("---")

# Row 1 of nationwide metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        """
        <div class="metric-card" style="border-left-color: #2E7D32;">
            <small style="color: #666; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Platform Coverage</small>
            <h3 style="margin: 5px 0 0 0; color: #2E7D32; font-size: 1.8rem; font-weight: 800;">India-wide</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        """
        <div class="metric-card" style="border-left-color: #1E88E5;">
            <small style="color: #666; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">States Covered</small>
            <h3 style="margin: 5px 0 0 0; color: #1E88E5; font-size: 1.8rem; font-weight: 800;">28 States (4 Focus)</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
with col3:
    st.markdown(
        """
        <div class="metric-card" style="border-left-color: #F57C00;">
            <small style="color: #666; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Districts Covered</small>
            <h3 style="margin: 5px 0 0 0; color: #F57C00; font-size: 1.8rem; font-weight: 800;">700+ Districts</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

# Row 2 of nationwide metrics
col4, col5, col6 = st.columns(3)
with col4:
    st.markdown(
        """
        <div class="metric-card" style="border-left-color: #9C27B0;">
            <small style="color: #666; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Satellite Resolution</small>
            <h3 style="margin: 5px 0 0 0; color: #9C27B0; font-size: 1.8rem; font-weight: 800;">10m Sentinel-2</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
with col5:
    st.markdown(
        """
        <div class="metric-card" style="border-left-color: #E91E63;">
            <small style="color: #666; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Calibrated Lab Samples</small>
            <h3 style="margin: 5px 0 0 0; color: #E91E63; font-size: 1.8rem; font-weight: 800;">300+ Samples</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
with col6:
    st.markdown(
        """
        <div class="metric-card" style="border-left-color: #009688;">
            <small style="color: #666; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Prediction Models</small>
            <h3 style="margin: 5px 0 0 0; color: #009688; font-size: 1.8rem; font-weight: 800;">5 Ensemble Regressors</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# Main introduction text
st.markdown(
    """
    <div class="glass-card">
        <h3>Platform Abstract</h3>
        <p>TerraSense AI integrates high-resolution spatial satellite inputs (Sentinel-2 multispectral bands) and physical regional grids (SoilGrids, OpenLandMap) with localized, ground-truth laboratory reports to construct a sub-meter parcel nutrient analysis model. 
        It evaluates predictive uncertainty dynamically (ensemble variance) to generate reliable crop-specific fertilizer schedules (N-P-K ratios, split applications, soil conditioning) for sustainable precision agriculture across all of India, with deep calibration for key focus states (<b>Andhra Pradesh, Telangana, Karnataka, Tamil Nadu</b>).</p>
        <p><b>⬅️ Use the Sidebar to navigate through pages:</b></p>
        <ul>
            <li><b>1. Location Analysis & Map</b>: Interactive soil map viewer to search any village, mandal, district or state in India and execute dynamic soil predictions.</li>
            <li><b>2. Nutrient Heatmaps</b>: Spatial raster predictions of Nitrogen, Phosphorus, Potassium, SOC, and pH.</li>
            <li><b>3. Soil Health Summary</b>: Visual diagnostics, nutrient balance sheets, and carbon indexing.</li>
            <li><b>4. Confidence Analysis</b>: Model uncertainty maps highlighting areas requiring further lab validation.</li>
            <li><b>5. Crop Recommendation</b>: Redesigned suitability assessment tool for <b>Chilli, Cotton, Rice, Maize, Groundnut, Tomato</b>.</li>
            <li><b>6. Fertilizer Advisory</b>: Detailed chemical fertilizer recommendations (Urea, DAP, MOP) and growth stage scheduling.</li>
            <li><b>7. Model Performance</b>: ML validation curves, feature importances, and evaluation metrics.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)
