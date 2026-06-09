import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

st.markdown("# 🩺 Soil Health Diagnostic Summary")
st.markdown("##### Comprehensive soil wellness report, chemical index balance, and site-specific agronomic recommendations.")

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

# Ensure predictions are run
if "point_predictions" not in st.session_state:
    with st.spinner("Calculating soil predictions..."):
        feats = extract_features_at_coords(curr_lat, curr_lon, config)
        preds = predictor.predict_point(feats)
        st.session_state["point_predictions"] = preds
        st.session_state["point_features"] = feats
else:
    preds = st.session_state["point_predictions"]
    feats = st.session_state["point_features"]

# Retrieve N-P-K-pH-SOC values
ph_val = preds["ph"]["prediction"]
soc_val = preds["soc"]["prediction"]
n_val = preds["nitrogen"]["prediction"]
p_val = preds["phosphorus"]["prediction"]
k_val = preds["potassium"]["prediction"]
bd_val = feats.get("bulk_density", 1.3)

# ----------------- SOIL HEALTH SCORE CALCULATION -----------------

# pH score component (ideal is 6.5)
ph_diff = abs(ph_val - 6.5)
ph_score = max(20.0, 100.0 - (ph_diff * 45.0))

# SOC score component (ideal is >= 1.2%)
soc_score = 100.0 if soc_val >= 1.2 else (soc_val / 1.2) * 100.0
soc_score = max(10.0, soc_score)

# Nutrient score component (compared to generic focus crops targets)
n_ratio = min(1.0, n_val / 45.0)
p_ratio = min(1.0, p_val / 18.0)
k_ratio = min(1.0, k_val / 140.0)
nutrient_score = (n_ratio * 0.4 + p_ratio * 0.3 + k_ratio * 0.3) * 100.0

# Overall soil health index
soil_health_score = (ph_score * 0.3) + (soc_score * 0.3) + (nutrient_score * 0.4)
soil_health_score = int(np.clip(soil_health_score, 10, 100))

# ----------------- RUN SUITABILITY FOR CROP CLASSIFICATION -----------------
from src.utils import fetch_weather_forecast
weather_info = fetch_weather_forecast(curr_lat, curr_lon)
temp_val = weather_info.get("current_temp", 28.0)

suitability_results = suitability_eng.evaluate_suitability(
    {
        "nitrogen": n_val,
        "phosphorus": p_val,
        "potassium": k_val,
        "ph": ph_val,
        "soc": soc_val
    },
    bulk_density=bd_val,
    temperature=temp_val
)

best_crops = []
unsuitable_crops = []

# Sort all crop suitability results by score descending
sorted_by_score = sorted(suitability_results.items(), key=lambda item: item[1]["suitability_score"], reverse=True)

for k, val in suitability_results.items():
    if val["suitability_score"] >= 70.0:
        best_crops.append(f"**{val['crop_name']}** ({val['suitability_score']:.1f}%)")
    elif val["suitability_score"] < 50.0:
        reasons_str = "; ".join(val.get("reasons", []))
        reason_suffix = f" — <span style='font-size:0.82rem; color:#666;'><i>{reasons_str}</i></span>" if reasons_str else ""
        unsuitable_crops.append(f"❌ **{val['crop_name']}** ({val['suitability_score']:.1f}% Match){reason_suffix}")

if not best_crops:
    if sorted_by_score:
        top_crop_name = sorted_by_score[0][1]["crop_name"]
        top_crop_score = sorted_by_score[0][1]["suitability_score"]
        best_crops = [f"⚠️ *No crops meet 70% threshold due to heat ({temp_val:.1f}°C).* Best alternative: **{top_crop_name}** ({top_crop_score:.1f}% Match)"]
    else:
        best_crops = ["No crops meet optimal suitability thresholds."]
        
if not unsuitable_crops:
    unsuitable_list_html = "<div style='font-style: italic;'>No crops are classified as unsuitable.</div>"
else:
    unsuitable_list_html = "".join([f"<div style='margin-bottom: 6px;'>{item}</div>" for item in unsuitable_crops])

# ----------------- IDENTIFY LIMITING FACTORS & INSIGHTS -----------------
limiting_factors = []
agronomic_insights = []

if ph_val < 5.8:
    limiting_factors.append(f"🔴 **High Acidity**: Measured pH of {ph_val:.2f} restricts availability of primary nutrients.")
    agronomic_insights.append("👉 **Liming intervention**: Apply Agricultural Lime (CaCO3) to neutralize acid ions and increase Phosphorus availability.")
elif ph_val > 7.8:
    limiting_factors.append(f"🔴 **High Alkalinity**: Measured pH of {ph_val:.2f} can induce metallic nutrient lockout.")
    agronomic_insights.append("👉 **Alkalinity corrective**: Incorporate Gypsum (CaSO4) basal treatment to leach sodium and correct alkaline pH drift.")
else:
    agronomic_insights.append("👉 Soil chemical pH is stable. Continue maintaining crop rotations to protect microbial buffers.")

if soc_val < 1.0:
    limiting_factors.append(f"🟠 **Organic Carbon Deficit**: Soil organic carbon stands at {soc_val:.2f}%, indicating carbon depletion.")
    agronomic_insights.append("👉 **Humus Restoration**: Practice zero-tillage, apply organic manures/vermicompost, or incorporate crop residues to build humus reserves.")
else:
    agronomic_insights.append("👉 Soil Organic Matter is at a healthy baseline. Maintain soil cover crops to prevent carbon oxidation.")

if n_val < 35.0:
    limiting_factors.append(f"🟡 **Nitrogen (N) Deficiency**: Nitrogen levels are low ({n_val:.1f} mg/kg).")
    agronomic_insights.append("👉 **Nitrogen Management**: Apply chemical nitrogen (Urea) in multiple splits to prevent volatilization. Integrate legumes (like Groundnut) to fix atmospheric N.")

if p_val < 12.0:
    limiting_factors.append(f"🟡 **Phosphorus (P) Deficiency**: Phosphorus levels are low ({p_val:.1f} mg/kg).")
    agronomic_insights.append("👉 **Phosphorus Amendment**: Deep-place Diammonium Phosphate (DAP) basal fertilizer during sowing to facilitate immediate root absorption.")

if k_val < 110.0:
    limiting_factors.append(f"🟡 **Potassium (K) Deficiency**: Potassium levels are low ({k_val:.1f} mg/kg).")
    agronomic_insights.append("👉 **Potassium Application**: Apply Muriate of Potash (MOP) to strengthen crop stems, improve drought tolerance, and boost fruit quality.")

if bd_val > 1.42:
    limiting_factors.append(f"🔴 **Compaction Risk**: High bulk density ({bd_val:.2f} g/cm³) indicating compacted subsoil structure.")
    agronomic_insights.append("👉 **Compaction Relief**: Implement deep chiseling/subsoiling once every two seasons to improve root penetration and water drainage.")

if not limiting_factors:
    limiting_factors.append("🟢 No critical limiting factors detected. Soil chemistry balances match optimal agronomical thresholds.")

# ----------------- RENDER LAYOUT -----------------

st.markdown(f"#### Active Location: **{loc_meta['village']}, {loc_meta['district']}, {loc_meta['state']}**")

col_health, col_insights = st.columns([5, 7])

with col_health:
    # Soil Health Gauge Chart
    if soil_health_score >= 80:
        health_color = "#2E7D32" # Green
        health_bg = "#E8F5E9"
    elif soil_health_score >= 60:
        health_color = "#EF6C00" # Orange
        health_bg = "#FFF3E0"
    else:
        health_color = "#C62828" # Red
        health_bg = "#FFEBEE"
        
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = soil_health_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "<b>Soil Health Score</b>", 'font': {'size': 20, 'color': '#2C3E50'}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#444"},
            'bar': {'color': health_color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#ddd",
            'steps': [
                {'range': [0, 50], 'color': '#FFCDD2'},
                {'range': [50, 75], 'color': '#FFE0B2'},
                {'range': [75, 100], 'color': '#C8E6C9'}
            ]
        }
    ))
    fig_gauge.update_layout(height=260, margin=dict(t=40, b=0, l=40, r=40))
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    st.markdown(
        f"""
        <div style="background-color: {health_bg}; border: 1.5px solid {health_color}; border-radius: 12px; padding: 18px; text-align: center; box-shadow: 0 4px 10px rgba(0,0,0,0.04);">
            <span style="font-size: 0.82rem; font-weight: 800; color: #555; text-transform: uppercase; letter-spacing: 0.5px;">Soil Suitability Classification</span>
            <h4 style="margin: 5px 0; color: {health_color}; font-weight: 800;">
                {"Excellent Health Condition" if soil_health_score>=80 else "Moderate / Restrictive" if soil_health_score>=60 else "Degraded Soil Quality"}
            </h4>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_insights:
    st.markdown("### 🏆 Crop Suitability Matching")
    
    st.markdown(
        f"""
        <div style="background-color: #F5F7FA; border: 1px solid rgba(0,0,0,0.06); border-radius: 10px; padding: 15px; margin-bottom: 15px;">
            <span style="font-weight: 700; color: #2E7D32; font-size: 0.95rem;">✅ Best Suited Crops For This Location:</span>
            <div style="margin-top: 6px; font-size: 0.92rem; line-height: 1.5; color: #2C3E50;">
                {", ".join(best_crops)}
            </div>
        </div>
        <div style="background-color: #F5F7FA; border: 1px solid rgba(0,0,0,0.06); border-radius: 10px; padding: 15px;">
            <span style="font-weight: 700; color: #C62828; font-size: 0.95rem;">❌ Unsuitable / High-Risk Crops:</span>
            <div style="margin-top: 6px; font-size: 0.92rem; line-height: 1.5; color: #2C3E50;">
                {unsuitable_list_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
st.markdown("---")
col_factors, col_actions = st.columns(2)

with col_factors:
    st.markdown("### 🔴 Limiting Factors Mapped")
    st.write("Identified constraints and soil properties restricting crop yield potential at this coordinate:")
    for fact in limiting_factors:
        st.markdown(f"<div style='background-color: #FFF5F5; border-left: 4px solid #E53935; padding: 10px; border-radius: 4px; margin-bottom: 8px; font-size: 0.88rem; color: #2C3E50;'>{fact}</div>", unsafe_allow_html=True)

with col_actions:
    st.markdown("### 🩺 Recommended Agronomic Interventions")
    st.write("Actionable pathways to restore chemical balance and organic matter levels for sustainable cropping:")
    for ins in agronomic_insights:
        st.markdown(f"<div style='background-color: #F4FBF7; border-left: 4px solid #2E7D32; padding: 10px; border-radius: 4px; margin-bottom: 8px; font-size: 0.88rem; color: #2C3E50;'>{ins}</div>", unsafe_allow_html=True)
