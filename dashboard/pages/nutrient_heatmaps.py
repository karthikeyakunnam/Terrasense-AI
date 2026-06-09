import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import textwrap
from pathlib import Path

st.markdown("# 🔥 High-Resolution Nutrient Heatmaps")
st.markdown("##### Visualizing spatial soil property predictions and difference layers across the region.")

config = st.session_state.get("config")
predictor = st.session_state.get("predictor")
suitability_eng = st.session_state.get("suitability_eng")

if config is None or predictor is None or suitability_eng is None:
    st.warning("Please visit the landing page first to initialize data.")
    st.stop()

# Coordinate management
curr_lat, curr_lon = st.session_state["selected_point"]
loc_meta = st.session_state["location_metadata"]

# Ensure predictions are run
if "point_predictions" not in st.session_state:
    from src.features import extract_features_at_coords
    with st.spinner("Calculating soil predictions..."):
        feats = extract_features_at_coords(curr_lat, curr_lon, config)
        preds = predictor.predict_point(feats)
        st.session_state["point_predictions"] = preds
        st.session_state["point_features"] = feats
else:
    preds = st.session_state["point_predictions"]
    feats = st.session_state["point_features"]

# ----------------- DYNAMIC GRID GENERATOR -----------------

@st.cache_data
def calculate_grid_predictions_dynamic(lat: float, lon: float, _config, _model_serialized_key) -> dict:
    from src.utils import calculate_grid_predictions_dynamic as _calc
    return _calc(lat, lon, _config, _model_serialized_key)

# Execute predictions
with st.spinner("Generating spatial grids (400 coordinates)..."):
    model_version_token = "".join([f"{t['name']}" for t in config.targets])
    grid_data = calculate_grid_predictions_dynamic(curr_lat, curr_lon, config, model_version_token)

# ----------------- UI CONTROLS & RENDER -----------------

# Calculate crop suitability scores for AI recommendations
bd_val = feats.get("bulk_density", 1.3)
pred_vals = {key: val["prediction"] for key, val in preds.items()}

from src.utils import fetch_weather_forecast
weather_info = fetch_weather_forecast(curr_lat, curr_lon)
temp_val = weather_info.get("current_temp", 28.0)
suitability_results = suitability_eng.evaluate_suitability(pred_vals, bulk_density=bd_val, temperature=temp_val)

crop_list = list(config.agronomy["crops"].keys())
# Sort options by suitability score descending
crop_list_sorted = sorted(crop_list, key=lambda k: suitability_results[k]["suitability_score"], reverse=True)

# Default to the globally selected active crop
target_select_key = st.session_state.get("active_crop", "rice")
if target_select_key in crop_list_sorted:
    default_index = crop_list_sorted.index(target_select_key)
else:
    default_index = 0

def format_crop_option(crop_key):
    crop_name = config.agronomy["crops"][crop_key]["name"]
    score = suitability_results[crop_key]["suitability_score"]
    rating = suitability_results[crop_key]["rating"]
    is_top = (crop_key == crop_list_sorted[0])
    recommend_suffix = " 🌟 AI Recommended" if is_top else ""
    return f"{crop_name} ({score:.1f}% Match - {rating}){recommend_suffix}"

target_options = {t["name"]: t["description"] for t in config.targets}
col_sel1, col_sel2 = st.columns(2)

with col_sel1:
    selected_target = st.selectbox(
        "Choose soil property to map:",
        options=list(target_options.keys()),
        format_func=lambda x: target_options[x],
        key="heatmap_target_selector"
    )

with col_sel2:
    selected_crop = st.session_state.get("active_crop", "rice")
    active_crop_name = config.agronomy["crops"][selected_crop]["name"]
    st.markdown(
        f"""
        <div style="background-color: #E8F5E9; border: 1.5px solid #2E7D32; border-radius: 8px; padding: 10px 15px; margin-top: 5px;">
            <span style="font-size: 0.72rem; font-weight: 700; color: #1B5E20; text-transform: uppercase;">Active Crop (Sync'd from starting page)</span>
            <div style="font-size: 1.05rem; font-weight: 800; color: #2E7D32; margin-top: 2px;">
                🌾 {active_crop_name}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Active crop info banner and recommendations
curr_crop_name = config.agronomy["crops"][selected_crop]["name"]
best_crop_key = crop_list_sorted[0]
best_crop_name = config.agronomy["crops"][best_crop_key]["name"]
best_crop_score = suitability_results[best_crop_key]["suitability_score"]

st.info(
    f"🌾 **Currently Simulating Soil Targets for**: **{curr_crop_name}**  \n"
    f"💡 **AI Recommendation**: The most suitable crop for this location is **{best_crop_name}** with a match score of **{best_crop_score:.1f}%**. "
    f"Other highly suitable options for this field: "
    + ", ".join([f"**{config.agronomy['crops'][c]['name']}** ({suitability_results[c]['suitability_score']:.1f}%)" 
                 for c in crop_list_sorted[1:3]])
)

target_meta = [t for t in config.targets if t["name"] == selected_target][0]
t_name = selected_target

raw_pred = grid_data[f"{t_name}_pred"]
bd_grid = grid_data["bulk_density"]

# NPK conversion variables
if t_name in ["nitrogen", "phosphorus", "potassium"]:
    current_grid = raw_pred * bd_grid * 1.5
    unit = "kg/ha"
    
    crop_rules = config.agronomy["crops"][selected_crop]
    target_val = crop_rules[f"target_{t_name[0]}"] # target_n, target_p, target_k
    
    deficit_grid = np.maximum(0.0, target_val - current_grid)
    post_grid = np.maximum(current_grid, target_val)
    
elif t_name == "soc":
    current_grid = raw_pred
    unit = "%"
    crop_rules = config.agronomy["crops"][selected_crop]
    target_val = crop_rules["optimal_soc_min"]
    
    deficit_grid = np.maximum(0.0, target_val - current_grid)
    post_grid = np.maximum(current_grid, target_val)
    
else: # ph
    current_grid = raw_pred
    unit = "pH"
    crop_rules = config.agronomy["crops"][selected_crop]
    opt_min = crop_rules["optimal_ph_min"]
    opt_max = crop_rules["optimal_ph_max"]
    
    deficit_grid = np.where(current_grid < opt_min, opt_min - current_grid, 
                            np.where(current_grid > opt_max, current_grid - opt_max, 0.0))
    post_grid = np.clip(current_grid, opt_min, opt_max)

colorscales = {
    "nitrogen": "YlGn",
    "phosphorus": "Electric",
    "potassium": "Plasma",
    "soc": "YlOrBr",
    "ph": "Portland"
}

# ----------------- MAPS & EXPLANATION ROW -----------------

col_maps, col_explain = st.columns([7, 5])

def draw_contour(z, map_unit, scale_name):
    fig = go.Figure(data=go.Contour(
        z=z,
        x=grid_data["lons_axis"],
        y=grid_data["lats_axis"],
        colorscale=colorscales.get(selected_target, "Viridis") if scale_name != "deficit" else "Reds",
        colorbar=dict(title=map_unit),
        hoverinfo="x+y+z",
        contours=dict(
            coloring="heatmap",
            showlabels=True,
            labelfont=dict(size=12, color="white")
        )
    ))
    
    # Add active coordinate marker
    fig.add_trace(go.Scatter(
        x=[curr_lon],
        y=[curr_lat],
        mode="markers",
        marker=dict(color="red", size=12, symbol="star"),
        name="Active Query Location",
        showlegend=False
    ))
    
    fig.update_layout(
        xaxis_title="Longitude (°E)",
        yaxis_title="Latitude (°N)",
        height=450,
        margin=dict(l=40, r=40, b=40, t=25),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig

with col_maps:
    tab_curr, tab_diff, tab_post = st.tabs([
        "📂 Current Soil State Map", 
        "🚨 Deficit Layer Map", 
        "✨ Post-Fertilizer Simulated Map"
    ])
    
    with tab_curr:
        st.markdown(f"##### Current Soil Stock ({unit})")
        fig_curr = draw_contour(current_grid, unit, "current")
        st.plotly_chart(fig_curr, use_container_width=True, key="heatmap_current")
    
    with tab_diff:
        st.markdown(f"##### Difference / Deficit Map ({unit})")
        fig_diff = draw_contour(deficit_grid, unit, "deficit")
        st.plotly_chart(fig_diff, use_container_width=True, key="heatmap_deficit")
        
    with tab_post:
        st.markdown(f"##### Post-Fertilizer Simulated Soil Stock ({unit})")
        fig_post = draw_contour(post_grid, unit, "post")
        st.plotly_chart(fig_post, use_container_width=True, key="heatmap_post")

with col_explain:
    st.subheader("📍 Coordinate Click Explanation")
    st.write("Diagnostic analysis for the active query coordinate marker:")
    
    # Run calculation logic for the active coordinate explanations
    bd_val = feats.get("bulk_density", 1.3)
    val_pred = preds[selected_target]["prediction"]
    
    # Determine Status, Deficiency Category, and Reason
    if selected_target == "nitrogen":
        n_stock = val_pred * bd_val * 1.5
        if n_stock < 60.0:
            status = "Deficient"
            category = "High / Critical Deficit"
            reason = "Frequent heavy cropping without nitrogen splits has depleted root-active nitrogen. High sand fractions facilitate rapid nitrogen leaching."
            color = "#C62828" # Red
        elif n_stock < 90.0:
            status = "Marginal"
            category = "Medium Deficiency"
            reason = "Satisfactory for low-yield cycles, but requires additional urea top-dressing to meet precision agriculture targets."
            color = "#EF6C00" # Orange
        else:
            status = "Sufficient"
            category = "Low / No Deficiency"
            reason = "Stable nitrogen reserves supported by recent organic carbon compost inputs or nitrogen-fixing crop residues."
            color = "#2E7D32" # Green
            
    elif selected_target == "phosphorus":
        p_stock = val_pred * bd_val * 1.5
        if p_stock < 25.0:
            status = "Deficient"
            category = "High / Critical Deficit"
            reason = "High Phosphorus fixing in acidic or highly alkaline soil structures. Soluble phosphates are locked and unavailable."
            color = "#C62828"
        elif p_stock < 40.0:
            status = "Marginal"
            category = "Medium Deficiency"
            reason = "Phosphate stocks are restricted. Requires deep placement of Diammonium Phosphate (DAP) during seeding."
            color = "#EF6C00"
        else:
            status = "Sufficient"
            category = "Low / No Deficiency"
            reason = "Optimal phosphate availability. Plant roots can readily extract phosphorus for early growth and leaf structures."
            color = "#2E7D32"
            
    elif selected_target == "potassium":
        k_stock = val_pred * bd_val * 1.5
        if k_stock < 70.0:
            status = "Deficient"
            category = "High / Critical Deficit"
            reason = "Potassium reserves are severely depleted. High risk of weak stems and low drought tolerance in focus regions."
            color = "#C62828"
        elif k_stock < 110.0:
            status = "Marginal"
            category = "Medium Deficiency"
            reason = "Moderate potassium reserves. Crop requires Muriate of Potash (MOP) to improve cell density and disease resistance."
            color = "#EF6C00"
        else:
            status = "Sufficient"
            category = "Low / No Deficiency"
            reason = "Adequate potassium stock. Crops will show high drought resilience and optimal fruit/boll sizing."
            color = "#2E7D32"
            
    elif selected_target == "soc":
        if val_pred < 0.6:
            status = "Deficient"
            category = "High / Critical Deficit"
            reason = "Critical organic carbon depletion due to intensive tillage and lack of organic inputs. High erosion hazard."
            color = "#C62828"
        elif val_pred < 1.0:
            status = "Marginal"
            category = "Medium Deficiency"
            reason = "Satisfactory carbon levels. Transition to conservation tillage and cover cropping is recommended."
            color = "#EF6C00"
        else:
            status = "Sufficient"
            category = "Low / No Deficiency"
            reason = "Organic carbon is at an ideal level. Buffers pH variations well and enhances moisture retention."
            color = "#2E7D32"
            
    else: # ph
        if val_pred < 5.8:
            status = "Highly Acidic"
            category = "Acidic Drift"
            reason = "High hydrogen ion concentration locking up primary nutrients. Lime application is required."
            color = "#C62828"
        elif val_pred > 7.8:
            status = "Highly Alkaline"
            category = "Alkaline Drift"
            reason = "High carbonates lockout metallic micro-nutrients. Gypsum application is required."
            color = "#7B1FA2" # Purple
        else:
            status = "Neutral / Stable"
            category = "Optimal Balance"
            reason = "Chemical balance is in the neutral buffering zone, maximizing nutrient solubility and root uptake."
            color = "#2E7D32"

    st.markdown(
        textwrap.dedent(f"""
        <div class="glass-card" style="padding: 20px; border-left: 5px solid {color}; box-shadow: 0 4px 12px rgba(0,0,0,0.05);">
            <div style="font-size: 0.82rem; font-weight: 700; color: #555; text-transform: uppercase; letter-spacing: 0.5px;">Location Mapped</div>
            <div style="font-weight: 800; font-size: 1.15rem; color: #2C3E50; margin-top: 3px; margin-bottom: 15px;">
                📍 {loc_meta['village']}, {loc_meta['district']}
            </div>
            
            <div style="margin-bottom: 12px;">
                <span style="font-weight: 600; color: #555; font-size: 0.9rem;">Mapped Nutrient Level:</span>
                <span style="font-weight: 800; color: #2E7D32; float: right; font-size: 1.05rem;">{val_pred:.2f} {target_meta['unit']}</span>
            </div>
            <div style="margin-bottom: 12px;">
                <span style="font-weight: 600; color: #555; font-size: 0.9rem;">Status Assessment:</span>
                <span style="font-weight: 800; color: {color}; float: right; font-size: 1.05rem;">{status}</span>
            </div>
            <div style="margin-bottom: 15px;">
                <span style="font-weight: 600; color: #555; font-size: 0.9rem;">Deficiency Category:</span>
                <span style="font-weight: 700; color: #34495E; float: right;">{category}</span>
            </div>
            
            <div style="border-top: 1px dashed rgba(0,0,0,0.1); padding-top: 12px;">
                <span style="font-weight: 700; color: #1B5E20; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.5px;">Agronomic Assessment Reason:</span>
                <p style="margin: 5px 0 0 0; font-size: 0.88rem; line-height: 1.4; color: #444;">{reason}</p>
            </div>
        </div>
        """),
        unsafe_allow_html=True
    )

st.markdown("---")

# ----------------- VISUAL COLOR LEGENDS PANEL -----------------
st.markdown("### 🗺️ Soil Color Band Legend Explanation")
st.write("Explanatory mapping of colors used across the high-resolution soil property and difference heatmaps:")

col_l1, col_l2, col_l3, col_l4 = st.columns(4)

with col_l1:
    st.markdown(
        textwrap.dedent("""
        <div style="background-color: #E8F5E9; border: 1.5px solid #2E7D32; border-radius: 8px; padding: 12px; text-align: center; height: 100px;">
            <span style="color: #1B5E20; font-weight: 800; font-size: 0.9rem; text-transform: uppercase;">🟢 Green Band</span>
            <p style="margin: 6px 0 0 0; font-size: 0.78rem; color: #444; line-height: 1.3;">
                <b>Sufficient / Optimal:</b> Meets or exceeds requirements for target crop yields.
            </p>
        </div>
        """),
        unsafe_allow_html=True
    )

with col_l2:
    st.markdown(
        textwrap.dedent("""
        <div style="background-color: #FFFDE7; border: 1.5px solid #FBC02D; border-radius: 8px; padding: 12px; text-align: center; height: 100px;">
            <span style="color: #F57F17; font-weight: 800; font-size: 0.9rem; text-transform: uppercase;">🟡 Yellow Band</span>
            <p style="margin: 6px 0 0 0; font-size: 0.78rem; color: #444; line-height: 1.3;">
                <b>Marginal Level:</b> Requires maintenance crop scheduling splits.
            </p>
        </div>
        """),
        unsafe_allow_html=True
    )

with col_l3:
    st.markdown(
        textwrap.dedent("""
        <div style="background-color: #FFF3E0; border: 1.5px solid #EF6C00; border-radius: 8px; padding: 12px; text-align: center; height: 100px;">
            <span style="color: #E65100; font-weight: 800; font-size: 0.9rem; text-transform: uppercase;">🟠 Orange Band</span>
            <p style="margin: 6px 0 0 0; font-size: 0.78rem; color: #444; line-height: 1.3;">
                <b>Moderate Deficiency:</b> Restricts crop growth; active fertilizing advised.
            </p>
        </div>
        """),
        unsafe_allow_html=True
    )

with col_l4:
    st.markdown(
        textwrap.dedent("""
        <div style="background-color: #FFEBEE; border: 1.5px solid #C62828; border-radius: 8px; padding: 12px; text-align: center; height: 100px;">
            <span style="color: #B71C1C; font-weight: 800; font-size: 0.9rem; text-transform: uppercase;">🔴 Red Band</span>
            <p style="margin: 6px 0 0 0; font-size: 0.78rem; color: #444; line-height: 1.3;">
                <b>Severe Deficiency:</b> Critical starvation; high risk of harvest yield drop.
            </p>
        </div>
        """),
        unsafe_allow_html=True
    )
