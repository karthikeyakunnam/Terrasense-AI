import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# Custom CSS for crop scorecard
st.markdown("""
<style>
    .crop-score-card {
        background-color: #F5F7FA;
        border: 1px solid rgba(0,0,0,0.08);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
    }
    .crop-score-header {
        font-size: 1.3rem;
        font-weight: 800;
        color: #2E7D32;
        margin-bottom: 15px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px dashed rgba(0,0,0,0.1);
        padding-bottom: 8px;
    }
    .crop-metric {
        font-size: 0.92rem;
        font-weight: 600;
        color: #555;
        margin-bottom: 10px;
    }
    .crop-value {
        font-weight: 800;
        color: #2C3E50;
        float: right;
    }
    .explain-badge {
        font-weight: 700;
        font-size: 0.75rem;
        padding: 4px 10px;
        border-radius: 20px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🌾 Nationwide Crop Recommendation Engine")
st.markdown("##### Dynamic agricultural suitability assessment and evaluation for key focus crops.")

config = st.session_state.get("config")
predictor = st.session_state.get("predictor")
suitability_eng = st.session_state.get("suitability_eng")

if config is None or predictor is None or suitability_eng is None:
    st.warning("Please visit the landing page first to initialize data.")
    st.stop()

# Coordinate management
curr_lat, curr_lon = st.session_state["selected_point"]
loc_meta = st.session_state["location_metadata"]

from src.features import extract_features_at_coords
from src.utils import fetch_weather_forecast

# Ensure predictions are run
if "point_predictions" not in st.session_state:
    with st.spinner("Calculating point predictions..."):
        feats = extract_features_at_coords(curr_lat, curr_lon, config)
        preds = predictor.predict_point(feats)
        st.session_state["point_predictions"] = preds
        st.session_state["point_features"] = feats
else:
    preds = st.session_state["point_predictions"]
    feats = st.session_state["point_features"]

# Retrieve N-P-K-pH-SOC values
bd_val = feats.get("bulk_density", 1.3)
pred_vals = {key: val["prediction"] for key, val in preds.items()}

# Sidebar What-If Suitability Simulator
st.sidebar.markdown("---")
st.sidebar.markdown("### 🌾 What-If Suitability Simulator")
st.sidebar.write("Simulate soil improvement interventions (e.g. organic composting, cover crops, conservation tillage):")
added_n = st.sidebar.slider("Simulate nitrogen increase (mg/kg)", 0.0, 50.0, 0.0, step=5.0)
added_soc = st.sidebar.slider("Simulate soil organic carbon increase (%)", 0.0, 1.5, 0.0, step=0.1)

# Apply simulation
simulated_vals = pred_vals.copy()
simulated_vals["nitrogen"] += added_n
simulated_vals["soc"] += added_soc

# Fetch weather context for explanations and climate suitability
weather_info = fetch_weather_forecast(curr_lat, curr_lon)
temp_val = weather_info.get("current_temp", 28.0)

# Calculate crop suitability results
suitability_results = suitability_eng.evaluate_suitability(pred_vals, bulk_density=bd_val, temperature=temp_val)
sim_suitability_results = suitability_eng.evaluate_suitability(simulated_vals, bulk_density=bd_val, temperature=temp_val)

# Determine the average temperature range
temp_min = weather_info.get("temp_min_3day", 22.0)
temp_max = weather_info.get("temp_max_3day", 33.0)

st.write(f"Evaluating agricultural suitability for **{loc_meta['village']}, {loc_meta['district']}, {loc_meta['state']}**.")

# ----------------- SECTION A: BEST CROPS FOR THIS LOCATION -----------------
st.markdown("---")
st.markdown("### A. Best Crops For This Location")
st.write("Calculated based on Sentinel-2 indices match, SoilGrids physical clay/sand textures, pH constraints, and N-P-K target deficits.")

# Sort suitability
sorted_suitability = sorted(
    suitability_results.items(),
    key=lambda item: item[1]["suitability_score"],
    reverse=True
)

# Render Best Crops in a Table / Grid
crop_table_data = []
for crop_key, crop_val in sorted_suitability:
    crop_name = crop_val["crop_name"]
    score = crop_val["suitability_score"]
    rating = crop_val["rating"]
    expected_yield = crop_val["expected_yield"]
    risk = crop_val["risk_level"]
    
    # Check if simulated score differs
    sim_score = sim_suitability_results[crop_key]["suitability_score"]
    diff = sim_score - score
    score_display = f"{score:.1f}%"
    if diff > 0.1:
        score_display += f" 📈 (Simulated: {sim_score:.1f}%)"
        
    crop_table_data.append({
        "Crop Name": crop_name,
        "Suitability Score": score_display,
        "Expected Yield": expected_yield,
        "Risk Level": risk
    })

df_crops = pd.DataFrame(crop_table_data)

# Show beautiful list of recommended crops
for row_idx, row in df_crops.iterrows():
    # Set risk color badge
    r_level = row["Risk Level"]
    if r_level == "Low":
        r_color = "#2E7D32" # Green
        r_bg = "#E8F5E9"
    elif r_level == "Moderate":
        r_color = "#EF6C00" # Orange
        r_bg = "#FFF3E0"
    else:
        r_color = "#C62828" # Red
        r_bg = "#FFEBEE"
        
    st.markdown(
        f"""
        <div style="border: 1px solid rgba(0,0,0,0.08); border-radius: 8px; padding: 12px 18px; margin-bottom: 10px; background-color: #F8F9FA; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 2px 4px rgba(0,0,0,0.015);">
            <div style="display: flex; align-items: center; gap: 15px;">
                <span style="font-size: 1.3rem; font-weight: 800; color: #2E7D32; background-color: #E8F5E9; border-radius: 50%; width: 36px; height: 36px; display: flex; align-items: center; justify-content: center;">{row_idx+1}</span>
                <div>
                    <h5 style="margin: 0; color: #2C3E50; font-weight: 800; font-size: 1.05rem;">{row['Crop Name']}</h5>
                    <small style="color: #666; font-weight: 600;">Expected Yield Potential: <b>{row['Expected Yield']}</b></small>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 20px; text-align: right;">
                <div>
                    <span style="font-size: 1.15rem; font-weight: 800; color: #2E7D32;">{row['Suitability Score']}</span>
                    <br><small style="color: #888; font-weight: 500;">Match Score</small>
                </div>
                <div style="background-color: {r_bg}; color: {r_color}; padding: 6px 12px; border-radius: 6px; font-weight: 800; font-size: 0.8rem; text-transform: uppercase; border: 1px solid {r_color}40;">
                    {r_level} Risk
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ----------------- SECTION B: EVALUATE MY CROP -----------------
st.markdown("---")
st.markdown("### B. Evaluate My Crop")
st.write("Detailed multi-criteria scorecard check against current soil nutrients, texture parameters, and weather feeds for the active target crop:")

eval_crop_key = st.session_state.get("active_crop", "rice")
# Ensure the crop key exists in config.agronomy["crops"]
if eval_crop_key not in config.agronomy["crops"]:
    eval_crop_key = "rice"

active_crop_name = config.agronomy["crops"][eval_crop_key]["name"]
st.markdown(
    f"""
    <div style="background-color: #E8F5E9; border: 1.5px solid #2E7D32; border-radius: 8px; padding: 10px 15px; margin-bottom: 15px;">
        <span style="font-size: 0.72rem; font-weight: 700; color: #1B5E20; text-transform: uppercase;">Active Target Crop Under Evaluation (Sync'd from starting page)</span>
        <div style="font-size: 1.15rem; font-weight: 800; color: #2E7D32; margin-top: 2px;">
            🌾 {active_crop_name}
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Fetch data for evaluation
eval_rules = config.agronomy["crops"][eval_crop_key]
eval_res = suitability_results[eval_crop_key]

# Get model confidence score
conf_score = preds["nitrogen"]["confidence_score"] # representative NPK confidence

# Display scorecard
col_card1, col_card2 = st.columns(2)

with col_card1:
    st.markdown(
        f"""
        <div class="crop-score-card">
            <div class="crop-score-header">
                <span>📊 {eval_rules['name']} Scorecard</span>
                <span class="explain-badge" style="background-color: {'#2E7D32' if eval_res['rating'] in ['Highly Suitable','Suitable'] else '#EF6C00'};">
                    {eval_res['rating']}
                </span>
            </div>
            <div class="crop-metric">Suitability Index <span class="crop-value" style="color: #2E7D32; font-size: 1.25rem;">{eval_res['suitability_score']}%</span></div>
            <div class="crop-metric">Expected Yield <span class="crop-value" style="color: #1E88E5;">{eval_res['expected_yield']}</span></div>
            <div class="crop-metric">Water Requirement <span class="crop-value" style="color: #00838F;">{eval_rules.get('water_requirement_mm', 0.0)} mm</span></div>
            <div class="crop-metric">Risk Category <span class="crop-value" style="color: {'#2E7D32' if eval_res['risk_level']=='Low' else '#C62828'};">{eval_res['risk_level']} Risk</span></div>
            <div class="crop-metric">Ensemble Confidence <span class="crop-value" style="color: #7B1FA2;">{conf_score:.1f}%</span></div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_card2:
    # Components breakdown chart
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=[eval_res["breakdown"]["ph_score"], eval_res["breakdown"]["soc_score"], eval_res["breakdown"]["nutrient_score"]],
        theta=["pH Match", "Organic Carbon Match", "NPK Nutrient Match"],
        fill='toself',
        fillcolor='rgba(46, 125, 50, 0.2)',
        line=dict(color='#2E7D32', width=2),
        name="Suitability Parameters"
    ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        ),
        showlegend=False,
        height=220,
        margin=dict(t=25, b=25, l=45, r=40)
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# ----------------- EXPLAIN RECOMMENDATION PANEL -----------------
st.markdown("#### 📖 Explain Recommendation: Crop Parameters Alignment")
st.write(f"Evaluating how the chemical and meteorological properties of **{loc_meta['village']}** align with optimal growth ranges for **{eval_rules['name']}**:")

# Calculate status
ph_val = pred_vals["ph"]
ph_status = "Ideal" if eval_rules["optimal_ph_min"] <= ph_val <= eval_rules["optimal_ph_max"] else "Deviated"

soc_val = pred_vals["soc"]
soc_status = "Optimal" if soc_val >= eval_rules["optimal_soc_min"] else "Deficient"

# Available stock calculation
avail_n = pred_vals["nitrogen"] * bd_val * 1.5
avail_p = pred_vals["phosphorus"] * bd_val * 1.5
avail_k = pred_vals["potassium"] * bd_val * 1.5

k_status_text = "Strong" if avail_k >= eval_rules["target_k"] else "Marginal" if avail_k >= eval_rules["target_k"] * 0.7 else "Deficient"

# Simulated water support
rainfall_support = "Sufficient" if (1000.0 >= eval_rules.get("water_requirement_mm", 600.0)) else "Requires Irrigation"
temp_status = "Favorable" if (20.0 <= temp_max <= 38.0) else "Extreme"

col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    st.markdown(
        f"""
        <div style="background-color: #F8F9FA; border-left: 4px solid #2E7D32; border-radius: 4px; padding: 12px; margin-bottom: 12px;">
            <h6 style="margin: 0; color: #1B5E20; font-weight: 700;">🧪 Soil pH Suitability</h6>
            <p style="margin: 6px 0 0 0; font-size: 0.85rem; color: #444;">
                <b>Ideal pH Bounds:</b> {eval_rules['optimal_ph_min']} - {eval_rules['optimal_ph_max']}<br>
                <b>Measured Soil pH:</b> {ph_val:.2f} ({ph_status})
            </p>
        </div>
        <div style="background-color: #F8F9FA; border-left: 4px solid #8D6E63; border-radius: 4px; padding: 12px; margin-bottom: 12px;">
            <h6 style="margin: 0; color: #5D4037; font-weight: 700;">🪱 Soil Organic Carbon (SOC)</h6>
            <p style="margin: 6px 0 0 0; font-size: 0.85rem; color: #444;">
                <b>Target SOC:</b> >= {eval_rules['optimal_soc_min']}%<br>
                <b>Measured SOC:</b> {soc_val:.2f}% ({soc_status})
            </p>
        </div>
        <div style="background-color: #F8F9FA; border-left: 4px solid #00838F; border-radius: 4px; padding: 12px;">
            <h6 style="margin: 0; color: #006064; font-weight: 700;">🌦️ Rainfall & Irrigation Match</h6>
            <p style="margin: 6px 0 0 0; font-size: 0.85rem; color: #444;">
                <b>Water Requirement:</b> {eval_rules.get('water_requirement_mm', 0.0)} mm<br>
                <b>Regional Compatibility:</b> {rainfall_support}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_exp2:
    st.markdown(
        f"""
        <div style="background-color: #F8F9FA; border-left: 4px solid #FF9800; border-radius: 4px; padding: 12px; margin-bottom: 12px;">
            <h6 style="margin: 0; color: #E65100; font-weight: 700;">🧪 Primary N-P-K Nutrients Mapped</h6>
            <p style="margin: 6px 0 0 0; font-size: 0.85rem; color: #444; line-height: 1.4;">
                <b>Nitrogen (N) Stock:</b> {avail_n:.1f} kg/ha (Target: {eval_rules['target_n']} kg/ha)<br>
                <b>Phosphorus (P) Stock:</b> {avail_p:.1f} kg/ha (Target: {eval_rules['target_p']} kg/ha)<br>
                <b>Potassium (K) Stock:</b> {avail_k:.1f} kg/ha (Target: {eval_rules['target_k']} kg/ha)
            </p>
        </div>
        <div style="background-color: #F8F9FA; border-left: 4px solid #7B1FA2; border-radius: 4px; padding: 12px; margin-bottom: 12px;">
            <h6 style="margin: 0; color: #4A148C; font-weight: 700;">🌟 Potassium (K) Availability</h6>
            <p style="margin: 6px 0 0 0; font-size: 0.85rem; color: #444;">
                <b>Measured K availability status:</b> {k_status_text}<br>
                <i>Potassium is critical for boll density, fruit loading, and pest resistance in focus regions.</i>
            </p>
        </div>
        <div style="background-color: #F8F9FA; border-left: 4px solid #D84315; border-radius: 4px; padding: 12px;">
            <h6 style="margin: 0; color: #BF360C; font-weight: 700;">🌡️ Favorable Thermal Ranges</h6>
            <p style="margin: 6px 0 0 0; font-size: 0.85rem; color: #444;">
                <b>Optimal temperature check:</b> {temp_status}<br>
                <b>Active temperature forecast:</b> {temp_min:.1f}°C to {temp_max:.1f}°C
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
