import os
import streamlit as st
import pandas as pd
import rasterio
from pathlib import Path

from src.utils import wgs84_to_utm
from src.features import engineer_features, extract_features_at_coords

st.markdown("# 🧪 Precision Fertilizer Advisory System")
st.markdown("##### Tailored agronomic fertilizer application schedules, soil amendments, and growth-stage splits.")

config = st.session_state.get("config")
predictor = st.session_state.get("predictor")
recommender_eng = st.session_state.get("recommender_eng")
parcels = st.session_state.get("parcels")
suitability_eng = st.session_state.get("suitability_eng")

if config is None or predictor is None or recommender_eng is None or parcels is None or suitability_eng is None:
    st.warning("Please visit the landing page first to initialize assets.")
    st.stop()

# Spatial coordinates feature extraction is imported from src.features

# Initialize predictions if missing
curr_lat, curr_lon = st.session_state["selected_point"]

if "point_predictions" not in st.session_state:
    with st.spinner("Calculating point predictions..."):
        feats = extract_features_at_coords(curr_lat, curr_lon, config)
        preds = predictor.predict_point(feats)
        st.session_state["point_predictions"] = preds
        st.session_state["point_features"] = feats
else:
    preds = st.session_state["point_predictions"]
    feats = st.session_state["point_features"]

# ----------------- WHAT-IF SLIDERS & SIDE PANEL -----------------

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎚️ What-If Soil Tuning")
st.sidebar.write("Override predicted values to test recommendations under alternative conditions:")

scen_n = st.sidebar.slider("Nitrogen (mg/kg)", 10.0, 160.0, float(preds["nitrogen"]["prediction"]), step=1.0)
scen_p = st.sidebar.slider("Phosphorus (mg/kg)", 2.0, 90.0, float(preds["phosphorus"]["prediction"]), step=1.0)
scen_k = st.sidebar.slider("Potassium (mg/kg)", 30.0, 350.0, float(preds["potassium"]["prediction"]), step=1.0)
scen_soc = st.sidebar.slider("Organic Carbon (%)", 0.1, 3.0, float(preds["soc"]["prediction"]), step=0.1)
scen_ph = st.sidebar.slider("pH Level", 4.0, 9.5, float(preds["ph"]["prediction"]), step=0.1)

input_vals = {
    "nitrogen": scen_n,
    "phosphorus": scen_p,
    "potassium": scen_k,
    "soc": scen_soc,
    "ph": scen_ph
}

# Calculate suitability for options based on tuned/what-if inputs
crop_names_map = {key: val["name"] for key, val in config.agronomy["crops"].items()}
from src.utils import fetch_weather_forecast
weather_info = fetch_weather_forecast(curr_lat, curr_lon)
temp_val = weather_info.get("current_temp", 28.0)
suitability_results = suitability_eng.evaluate_suitability(
    input_vals,
    bulk_density=feats.get("bulk_density", 1.3),
    temperature=temp_val
)

selected_crop = st.session_state.get("active_crop", "rice")
if selected_crop not in config.agronomy["crops"]:
    selected_crop = "rice"

active_crop_name = config.agronomy["crops"][selected_crop]["name"]
score = suitability_results.get(selected_crop, {}).get("suitability_score", 0.0)
rating = suitability_results.get(selected_crop, {}).get("rating", "Unsuitable")

st.markdown(
    f"""
    <div style="background-color: #E8F5E9; border: 1.5px solid #2E7D32; border-radius: 8px; padding: 10px 15px; margin-bottom: 20px;">
        <span style="font-size: 0.72rem; font-weight: 700; color: #1B5E20; text-transform: uppercase;">Active Target Crop (Sync'd from starting page)</span>
        <div style="font-size: 1.15rem; font-weight: 800; color: #2E7D32; margin-top: 2px;">
            🌾 {active_crop_name} <span style="font-size: 0.9rem; font-weight: 600; color: #555;">({score:.1f}% Match - {rating})</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------- COMPUTE & RENDER ADVISORY -----------------

bd_val = feats.get("bulk_density", 1.3)
advisory = recommender_eng.calculate_recommendations(input_vals, selected_crop, bulk_density=bd_val)
col_doses, col_reasoning = st.columns([6, 6])

with col_doses:
    st.markdown("### 📦 Total Required Dosages")
    
    ferts = advisory["fertilizers"]
    
    st.markdown(
        f"""
        <div style="display: flex; gap: 10px; margin-bottom: 20px;">
            <div style="flex: 1; background-color: #ECEFF1; border: 1.5px solid #607D8B; border-radius: 8px; padding: 15px; text-align: center;">
                <span style="font-size: 0.8rem; font-weight: 700; color: #555; text-transform: uppercase;">Urea (46-0-0)</span>
                <h3 style="margin: 5px 0; color: #263238;">{ferts['urea_total_kg_ha']}</h3>
                <span style="font-size: 0.75rem; color: #666;">kg / hectare</span>
            </div>
            <div style="flex: 1; background-color: #ECEFF1; border: 1.5px solid #795548; border-radius: 8px; padding: 15px; text-align: center;">
                <span style="font-size: 0.8rem; font-weight: 700; color: #555; text-transform: uppercase;">DAP (18-46-0)</span>
                <h3 style="margin: 5px 0; color: #3E2723;">{ferts['dap_total_kg_ha']}</h3>
                <span style="font-size: 0.75rem; color: #666;">kg / hectare</span>
            </div>
            <div style="flex: 1; background-color: #ECEFF1; border: 1.5px solid #673AB7; border-radius: 8px; padding: 15px; text-align: center;">
                <span style="font-size: 0.8rem; font-weight: 700; color: #555; text-transform: uppercase;">MOP (0-0-60)</span>
                <h3 style="margin: 5px 0; color: #311B92;">{ferts['mop_total_kg_ha']}</h3>
                <span style="font-size: 0.75rem; color: #666;">kg / hectare</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Cost Optimization Panel (Feature 3)
    st.markdown("### 💰 Input Cost Optimization")
    costs = advisory["costs"]
    st.markdown(
        f"""
        <div style="background-color: white; border: 1.5px solid #ddd; border-radius: 8px; padding: 15px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-around; align-items: center; text-align: center;">
                <div>
                    <span style="font-size: 0.75rem; font-weight: 700; color: #666; text-transform: uppercase;">Standard Cost</span>
                    <h3 style="margin: 2px 0; color: #424242;">₹{int(costs['total_cost']):,}</h3>
                </div>
                <div style="font-size: 1.5rem; font-weight: 800; color: #757575;">🆚</div>
                <div>
                    <span style="font-size: 0.75rem; font-weight: 700; color: #2E7D32; text-transform: uppercase;">Eco-Banding Cost</span>
                    <h3 style="margin: 2px 0; color: #2E7D32;">₹{int(costs['alternative_eco_cost']):,}</h3>
                </div>
                <div style="background-color: #E8F5E9; padding: 8px 12px; border-radius: 8px; border: 1.5px dashed #2E7D32;">
                    <span style="font-size: 0.75rem; font-weight: 700; color: #2E7D32; text-transform: uppercase;">Net Savings</span>
                    <h3 style="margin: 2px 0; color: #1B5E20; font-size: 1.5rem; font-weight: 800;">₹{int(costs['savings']):,}</h3>
                </div>
            </div>
            <p style="font-size: 0.75rem; color: #666; margin: 10px 0 0 0; font-style: italic; text-align: center;">
                *Precision Subsurface Eco-Banding deep-places fertilizer, reducing volatilization/leaching loss by 15% and lowering input costs.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    conditioners = advisory["conditioners"]
    if conditioners:
        st.markdown("#### 🩹 Soil Conditioning Required")
        for cond in conditioners:
            st.warning(
                f"**Apply {cond['name']}**: **{cond['rate_kg_ha']} kg/ha**\n\n"
                f"*Reasoning:* {cond['reason']}"
            )
    else:
        st.success("✨ **Soil pH is stable.** No pH-corrective soil conditioning is required.")
        
    st.markdown("### 📅 Growth-Stage Split Schedule")
    df_schedule = pd.DataFrame(advisory["schedule"])
    df_schedule.columns = ["Crop Stage", "Recommended Fertilizer", "Dosage (kg/ha)", "Timing / Application Notes"]
    st.table(df_schedule)

with col_reasoning:
    # Weather Integration Panel (Feature 6)
    st.markdown("### 🌦️ Meteorological Feed Integration")
    st.write("Cross-references soil applications with local Open-Meteo forecasts to prevent nitrogen washing.")
    
    heavy_rain_threshold = config.get("weather", {}).get("heavy_rain_threshold_mm", 20.0)
    
    # Interactive simulation slider
    rain_forecast = st.slider(
        "Simulate 3-Day Forecast Rainfall (mm)", 
        0.0, 50.0, 24.5, step=1.0, 
        key="rain_sim_slider"
    )
    
    if rain_forecast >= heavy_rain_threshold:
        st.error(
            f"**🔴 CRITICAL WARNING: HEAVY PRECIPITATION ALERT**\n\n"
            f"Open-Meteo predicts **{rain_forecast:.1f} mm** of rainfall within the next 72 hours (exceeds risk threshold of {heavy_rain_threshold} mm).\n\n"
            f"⚠️ **Action Directive**: **DELAY UREA APPLICATION**. Soluble Nitrogen will volatilize and leach rapidly under heavy rainfall, "
            f"causing complete input loss and environmental contamination. Postpone application by 48 hours until runoff risk subsides."
        )
    else:
        st.success(
            f"**🌤️ WEATHER CONDITIONS STABLE**\n\n"
            f"Open-Meteo predicts **{rain_forecast:.1f} mm** of rainfall within the next 72 hours. "
            f"Conditions are optimal for immediate fertilizing and split top-dressing."
        )
        
    st.markdown("### 📖 Scientific & Agronomic Reasoning")
    for r in advisory["reasoning"]:
        st.markdown(f"👉 {r}")
        
    st.markdown("---")
    st.markdown("#### 📜 Field Action Summary (Printable)")
    
    summary_text = (
        f"TERRASENSE AI - FIELD ADVISORY SUMMARY\n"
        f"Coordinates: {curr_lat:.6f} N, {curr_lon:.6f} E\n"
        f"Crop Target: {crop_names_map[selected_crop].upper()}\n"
        f"--------------------------------------------------\n"
        f"1. TOTAL DOSAGE REQUIREMENTS:\n"
        f"   - Urea: {ferts['urea_total_kg_ha']} kg/ha\n"
        f"   - DAP:  {ferts['dap_total_kg_ha']} kg/ha\n"
        f"   - MOP:  {ferts['mop_total_kg_ha']} kg/ha\n"
        f"2. ESTIMATED INPUT COST:\n"
        f"   - Standard Plan: ₹{int(costs['total_cost']):,}\n"
        f"   - Eco-Banding:   ₹{int(costs['alternative_eco_cost']):,} (Savings: ₹{int(costs['savings']):,})\n"
    )
    
    if conditioners:
        summary_text += "3. SOIL CONDITIONERS:\n"
        for cond in conditioners:
            summary_text += f"   - {cond['name']}: {cond['rate_kg_ha']} kg/ha\n"
    else:
        summary_text += "3. SOIL CONDITIONERS: None required.\n"
        
    summary_text += "4. SPLIT TIMELINE:\n"
    for item in advisory["schedule"]:
        summary_text += f"   - {item['stage']}: Apply {item['rate_kg_ha']} kg/ha of {item['fertilizer']}\n"
        
    st.text_area("Advisory Clipboard Data:", value=summary_text, height=220)
    st.info("💡 Pro-Tip: Copy the text block above for mobile messaging updates or print summaries for field workers.")
