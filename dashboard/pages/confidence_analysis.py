import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

st.markdown("# 🛡️ Predictive Confidence & Active Sampling Analysis")
st.markdown("##### Quantifying model uncertainties, mapping prediction intervals, and identifying active laboratory sampling zones.")

config = st.session_state.get("config")
predictor = st.session_state.get("predictor")

if config is None or predictor is None:
    st.warning("Please visit the landing page first to initialize data.")
    st.stop()

# Coordinate management
curr_lat, curr_lon = st.session_state["selected_point"]

# ----------------- EDUCATIONAL CARD -----------------

st.markdown(
    """
    <div style="background-color: #f5f5f5; border-radius: 10px; padding: 20px; border-left: 6px solid #1565C0; margin-bottom: 25px;">
        <h4 style="margin-top: 0; color: #1565C0;">🔬 The Mathematics of Uncertainty Quantification</h4>
        <p style="font-size: 0.9rem; margin-bottom: 8px;">
            <b>Why does confidence matter?</b> In digital soil mapping, spatial autocorrelation, atmospheric haze in satellite inputs, and micro-terrain anomalies introduce variance. Presenting flat predictions without error margins can lead to sub-optimal fertilizer over-application or crop damage.
        </p>
        <p style="font-size: 0.9rem; margin-bottom: 8px;">
            <b>How is uncertainty calculated?</b> We deploy <b>Ensemble Tree Variance</b>. For a given spatial point, we query the predictions from all individual decision trees within the Random Forest model:
        </p>
        <div style="text-align: center; font-size: 1.1rem; padding: 10px; font-weight: 600; font-family: monospace;">
            σ²(x) = (1/B) * Σ [f_b(x) - ŷ]²
        </div>
        <p style="font-size: 0.9rem; margin-top: 8px;">
            where <i>B</i> is the number of trees (100) and <i>f_b(x)</i> is the prediction of tree <i>b</i>. 
            The standard deviation <i>σ(x)</i> represents model disagreement (epistemic uncertainty). 
            We scale this uncertainty exponentially relative to the model's out-of-bag validation residuals to compute a normalized <b>Confidence Score (0-100%)</b>:
        </p>
        <div style="text-align: center; font-size: 1.1rem; padding: 10px; font-weight: 600; font-family: monospace;">
            Confidence(x) = exp(-λ * σ(x)) * 100
        </div>
        <p style="font-size: 0.9rem; margin-top: 8px; margin-bottom: 0;">
            Where λ is calibrated using validation residuals. Thus, areas with high tree consensus output near 100% confidence, while unfamiliar feature spaces approach 0% confidence.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------- INGEST GRID INFORMATION -----------------

from src.utils import calculate_grid_predictions_dynamic

with st.spinner("Retrieving dynamic grid confidence boundaries..."):
    model_version_token = "".join([f"{t['name']}" for t in config.targets])
    grid_data = calculate_grid_predictions_dynamic(curr_lat, curr_lon, config, model_version_token)

# Selection of target variables
target_options = {t["name"]: t["description"] for t in config.targets}
selected_target = st.selectbox(
    "Choose soil property to analyze confidence:",
    options=list(target_options.keys()),
    format_func=lambda x: target_options[x],
    key="confidence_target_selector"
)

target_meta = [t for t in config.targets if t["name"] == selected_target][0]
conf_grid = grid_data[f"{selected_target}_confidence"]

# Check if lab report is active - if so, active point has 100% confidence
lab_override = st.session_state.get("lab_report_data")
if lab_override and selected_target in lab_override:
    # Set the center cell (approx index 10,10 in 20x20 grid) to 100% confidence
    conf_grid[10, 10] = 100.0

# ----------------- PLOT CONFIDENCE MAP -----------------

st.subheader(f"Model Confidence Map: {target_meta['description']}")

fig = go.Figure(data=go.Contour(
    z=conf_grid,
    x=grid_data["lons_axis"],
    y=grid_data["lats_axis"],
    colorscale="RdYlGn",
    zmin=0.0,
    zmax=100.0,
    colorbar=dict(title="Confidence %"),
    hoverinfo="x+y+z",
    contours=dict(
        coloring="heatmap",
        showlabels=True
    )
))

# Overlay active query coordinate marker
fig.add_trace(go.Scatter(
    x=[curr_lon],
    y=[curr_lat],
    mode="markers",
    marker=dict(color="blue", size=12, symbol="circle-open", line=dict(color="blue", width=3)),
    name="Active Location Center",
    showlegend=False
))

fig.update_layout(
    xaxis_title="Longitude (°E)",
    yaxis_title="Latitude (°N)",
    height=480,
    margin=dict(l=40, r=40, b=40, t=25),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)"
)

st.plotly_chart(fig, use_container_width=True)

# ----------------- ACTIVE SAMPLING RECOMMENDER -----------------

st.markdown("---")
st.subheader("🎯 Active Sampling: Recommended In-Situ Soil Sample Locations")
st.markdown("The platform runs an **Active Learning** loop. Below are the coordinates within the mapped region where the model exhibits the **highest uncertainty** (lowest confidence score). Gathering actual physical laboratory soil cores at these points will maximize model learning rate upon next retraining cycle.")

flat_conf = conf_grid.flatten()
lons_mesh, lats_mesh = np.meshgrid(grid_data["lons_axis"], grid_data["lats_axis"])
lons_flat = lons_mesh.flatten()
lats_flat = lats_mesh.flatten()

# Sort by confidence ascending
points_sorted = sorted(zip(flat_conf, lats_flat, lons_flat), key=lambda x: x[0])

recommended_points = []
min_distance_degrees = 0.005 # approximate spatial separation

for conf, lat, lon in points_sorted:
    too_close = False
    for r_conf, rlat, rlon in recommended_points:
        dist = np.sqrt((lat - rlat)**2 + (lon - rlon)**2)
        if dist < min_distance_degrees:
            too_close = True
            break
            
    if not too_close:
        # Prevent selecting the center if it was forced to 100% due to lab report
        if conf < 99.9 or not lab_override:
            recommended_points.append((conf, lat, lon))
        
    if len(recommended_points) >= 5:
        break

sampling_data = []
for idx, (conf, lat, lon) in enumerate(recommended_points):
    sampling_data.append({
        "Priority": f"🔥 Priority {idx+1}",
        "Model Confidence": f"{conf:.1f}%",
        "Latitude Target": f"{lat:.6f}°N",
        "Longitude Target": f"{lon:.6f}°E",
        "Required Action": "Core Soil Sample Core Extraction"
    })
df_sampling = pd.DataFrame(sampling_data)

st.table(df_sampling)
st.success("📝 Action Plan: Dispatch field sampling technicians with GPS units to collect core soil samples at these exact coordinate points. Run them in the lab, append to raw CSV, and execute train.py to retrain.")
