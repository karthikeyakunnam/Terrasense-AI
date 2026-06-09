import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import urllib.parse
import json
import io
from pathlib import Path

# Custom styling for diagnostics
st.markdown("""
<style>
    .metric-unit {
        font-size: 0.85rem;
        color: #777;
        font-weight: 500;
        margin-left: 3px;
    }
    .status-badge {
        font-size: 0.75rem;
        font-weight: 700;
        padding: 4px 8px;
        border-radius: 4px;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🗺️ Location Soil Analysis & Mapping")
st.markdown("##### Search any location in India, upload a soil lab report, and generate dynamic soil recommendations.")

# Get cached session assets
config = st.session_state.get("config")
predictor = st.session_state.get("predictor")
suitability_eng = st.session_state.get("suitability_eng")
recommender_eng = st.session_state.get("recommender_eng")

if config is None or predictor is None or suitability_eng is None or recommender_eng is None:
    st.warning("Please visit the landing page first to initialize assets.")
    st.stop()

from src.features import extract_features_at_coords
from src.utils import geocode_location, reverse_geocode, fetch_weather_forecast, parse_soil_lab_report

# ----------------- COORDINATE MANAGEMENT -----------------

curr_lat, curr_lon = st.session_state["selected_point"]
zoom = st.session_state.get("map_zoom", 11)

# Local Page search input
search_col1, search_col2 = st.columns([8, 2])
with search_col1:
    search_query = st.text_input(
        "🔍 Search Village, Mandal, District, or State in India (e.g. Vemulapadu, Guntur, Nandyal, Kadapa, Anantapur):",
        placeholder="Type a location name...",
        key="page_location_search_input"
    )
with search_col2:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    search_btn = st.button("Search Location", use_container_width=True, key="page_search_btn")

if search_btn and search_query:
    with st.spinner(f"Geocoding '{search_query}'..."):
        coords = geocode_location(search_query)
        if coords:
            st.session_state["selected_point"] = (coords["lat"], coords["lon"])
            st.session_state["location_metadata"] = reverse_geocode(coords["lat"], coords["lon"])
            st.session_state["map_zoom"] = 13
            # Clear old predictions
            if "point_predictions" in st.session_state:
                del st.session_state["point_predictions"]
            st.success(f"Located: {coords['display_name'][:50]}...")
            st.rerun()
        else:
            st.error("Location not found. Please try a different query.")

# ----------------- MAIN LAYOUT -----------------

col_map, col_details = st.columns([7, 5])

with col_map:
    st.subheader("🗺️ Interactive Satellite Map")
    st.caption("Click anywhere on the map to query dynamic soil predictions for that point.")
    
    # Create Folium Map centered on active coordinates
    m = folium.Map(location=[curr_lat, curr_lon], zoom_start=zoom, control_scale=True)
    
    # Add Google Hybrid basemap
    tiles = "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}"
    folium.TileLayer(
        tiles=tiles,
        attr="Google Map Satellite",
        name="Satellite Imagery",
        overlay=False,
        control=True
    ).add_to(m)
    
    # Add Active query marker
    folium.Marker(
        location=[curr_lat, curr_lon],
        icon=folium.Icon(color="red", icon="info-sign"),
        tooltip="Active Query Coordinate"
    ).add_to(m)
    
    # Render map
    map_data = st_folium(m, width="100%", height=450, key="folium_soil_map")
    
    # Handle map clicks
    if map_data and map_data.get("last_clicked"):
        click_lat = map_data["last_clicked"]["lat"]
        click_lon = map_data["last_clicked"]["lng"]
        
        # Swallow stale click events immediately after a search geocoding request
        if st.session_state.get("just_searched"):
            st.session_state["just_searched"] = False
            st.session_state["last_map_click"] = (click_lat, click_lon)
        else:
            # Prevent duplicate click triggers from overriding search state on page rerun
            last_click = st.session_state.get("last_map_click")
            if last_click is None or abs(click_lat - last_click[0]) > 1e-6 or abs(click_lon - last_click[1]) > 1e-6:
                st.session_state["last_map_click"] = (click_lat, click_lon)
                st.session_state["selected_point"] = (click_lat, click_lon)
                st.session_state["location_metadata"] = reverse_geocode(click_lat, click_lon)
                st.session_state["map_zoom"] = map_data.get("zoom", zoom)
                if "point_predictions" in st.session_state:
                    del st.session_state["point_predictions"]
                st.rerun()

    # ----------------- LAB REPORT UPLOADER -----------------
    st.markdown("---")
    st.markdown("### 🧪 Lab Soil Report Calibration (Override)")
    st.write("Upload a soil laboratory testing report (PDF, CSV, or XLSX). Values found will override satellite-derived prediction values and increase confidence to 100%.")
    
    uploaded_file = st.file_uploader(
        "Upload lab report:",
        type=["pdf", "csv", "xlsx"],
        help="Upload CSV/Excel with columns N, P, K, pH, SOC, or a PDF soil testing report."
    )
    
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        try:
            parsed_labs = parse_soil_lab_report(file_bytes, uploaded_file.name)
            if parsed_labs:
                st.session_state["lab_report_data"] = parsed_labs
                st.success("🔬 **Lab Report Parsed Successfully!**")
                
                # Show parsed values in a nice format
                vals_df = pd.DataFrame([
                    {
                        "Soil Nutrient / Metric": k.upper() if k != "soc" else "Soil Organic Carbon (SOC)",
                        "Parsed Value": f"{v:.2f} mg/kg" if k in ["nitrogen", "phosphorus", "potassium"] else f"{v:.2f}" + ("%" if k=="soc" else "")
                    }
                    for k, v in parsed_labs.items()
                ])
                st.table(vals_df)
            else:
                st.warning("⚠️ No matching soil parameters (N, P, K, pH, SOC) were found in the uploaded file. Check headers/formatting.")
        except Exception as e:
            st.error(f"Failed parsing file: {e}")
            
    if st.session_state.get("lab_report_data"):
        if st.button("🗑️ Clear Lab Report Override", use_container_width=True):
            del st.session_state["lab_report_data"]
            st.rerun()

with col_details:
    st.subheader("📍 Active Point Diagnosis")
    
    loc_meta = st.session_state["location_metadata"]
    
    # Extract features at coordinates dynamically
    with st.spinner("Extracting Sentinel-2 & SoilGrids layers..."):
        try:
            point_features = extract_features_at_coords(curr_lat, curr_lon, config)
        except Exception as e:
            st.error(f"Error querying spatial layers at coordinates: {e}")
            st.stop()
            
    # Run Inference
    with st.spinner("Running ensemble models..."):
        predictions = predictor.predict_point(point_features)
        
    # Check if we should override with lab report data
    lab_override = st.session_state.get("lab_report_data")
    if lab_override:
        for target_name, val in lab_override.items():
            if target_name in predictions:
                predictions[target_name]["prediction"] = round(val, 2)
                predictions[target_name]["confidence_score"] = 100.0
                predictions[target_name]["lower_bound"] = round(val, 2)
                predictions[target_name]["upper_bound"] = round(val, 2)
                predictions[target_name]["std_dev"] = 0.0
                predictions[target_name]["evidence"] = "Calibrated from lab report data upload"
                predictions[target_name]["explanations"] = [
                    {"feature": "Lab Report Upload", "percentage": 100.0, "direction": 1}
                ]
                
    st.session_state["point_features"] = point_features
    st.session_state["point_predictions"] = predictions
    
    # Setup values for recommender
    pred_vals = {key: val["prediction"] for key, val in predictions.items()}
    
    # Recommender target selection (Rice, Maize, Cotton, Tomato, Chilli, Groundnut)
    st.markdown("##### Target Crop Selection")
    
    # Text input to search / add any crop in the world
    custom_crop_search = st.text_input(
        "🔍 Search or type ANY crop in the world (e.g. Chickpeas, Sugarcane, Mustard):",
        key="map_custom_crop_search_input"
    )
    
    default_crop_key = None
    if custom_crop_search.strip():
        from src.recommender import register_custom_crop
        custom_key = register_custom_crop(config, custom_crop_search.strip())
        if custom_key:
            default_crop_key = custom_key
            st.session_state["active_crop"] = custom_key

    # Fetch weather context for temperature-based suitability
    weather_info = fetch_weather_forecast(curr_lat, curr_lon)
    temp_val = weather_info.get("current_temp", 28.0)

    # Calculate crop suitability scores for AI recommendations
    suitability_results = suitability_eng.evaluate_suitability(
        pred_vals, 
        bulk_density=point_features.get("bulk_density", 1.3),
        temperature=temp_val
    )
    
    crop_list = list(config.agronomy["crops"].keys())
    # Sort options by suitability score descending so the best crop is first
    crop_list_sorted = sorted(crop_list, key=lambda k: suitability_results[k]["suitability_score"], reverse=True)
    
    # Find the default index based on search key or current global active crop
    target_select_key = default_crop_key or st.session_state.get("active_crop", "rice")
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
    
    active_crop = st.selectbox(
        "Select target crop for fertilizer schedule:",
        options=crop_list_sorted,
        index=default_index,
        format_func=format_crop_option,
        key="map_crop_selector"
    )
    st.session_state["active_crop"] = active_crop
    
    # Calculate Recommendations
    advisory = recommender_eng.calculate_recommendations(
        pred_vals, 
        active_crop, 
        bulk_density=point_features.get("bulk_density", 1.3)
    )
    
    # Show location card
    st.markdown(
        f"""
        <div style="background-color: #F5F7FA; border: 1px solid rgba(0,0,0,0.06); border-radius: 12px; padding: 15px; margin-bottom: 20px; font-size: 0.95rem; color: #2C3E50; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-weight: 600; color: #555;">Village/Town:</span>
                <span style="font-weight: 700; color: #1E88E5;">{loc_meta['village']}</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-weight: 600; color: #555;">Mandal & District:</span>
                <span style="font-weight: 700; color: #34495E;">{loc_meta['mandal']}, {loc_meta['district']}</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: 600; color: #555;">State:</span>
                <span style="font-weight: 700; color: #2E7D32;">{loc_meta['state']}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Create Tabs for layout
    tab_diag, tab_preds, tab_explain, tab_agronomy = st.tabs([
        "Diagnostics", 
        "Nutrients", 
        "Explainability",
        "AI Agronomist Report"
    ])
    
    with tab_diag:
        h_score = advisory["health_score"]
        if h_score >= 80:
            score_color = "#2E7D32"
            score_bg = "#E8F5E9"
            score_text_color = "#1B5E20"
        elif h_score >= 60:
            score_color = "#EF6C00"
            score_bg = "#FFF3E0"
            score_text_color = "#E65100"
        else:
            score_color = "#C62828"
            score_bg = "#FFEBEE"
            score_text_color = "#B71C1C"
            
        risks_joined = ", ".join(advisory["risks"]) if advisory["risks"] else "No major nutritional risks detected."
        
        st.markdown(
            f"""
            <div style="background-color: {score_bg}; border: 1.5px solid {score_color}; border-radius: 12px; padding: 16px; margin-top: 10px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-size: 0.75rem; font-weight: 700; color: #555; text-transform: uppercase; letter-spacing: 0.5px;">Soil Health Index</span>
                        <h2 style="margin: 2px 0; color: {score_color}; font-size: 2.2rem; font-weight: 800;">{h_score}/100</h2>
                    </div>
                    <div style="text-align: right; max-width: 65%;">
                        <span style="font-size: 0.75rem; font-weight: 700; color: #555; text-transform: uppercase; letter-spacing: 0.5px;">Detected Risks</span>
                        <div style="font-size: 0.88rem; font-weight: 700; color: {score_text_color}; margin-top: 4px; line-height: 1.3;">
                            {risks_joined}
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Display uploader calibration status badge
        if lab_override:
            st.success("🔬 **Lab Report Calibration Active**: Standard deviation variance set to zero. Predictions overridden with lab values.")
        else:
            st.info("🛰️ **Remote Sensing Feed**: Standard deviation variance calculated from XGBoost/Gradient Boosting models.")
            
        # Weather Integration
        weather_info = fetch_weather_forecast(curr_lat, curr_lon)
        st.markdown("##### 🌦️ Weather & Soil Application Safety")
        st.write(f"Current Temp: **{weather_info['current_temp']}°C** | 3-Day Expected Rain: **{weather_info['precip_3day']:.1f} mm**")
        
        heavy_rain_threshold = config.get("weather", {}).get("heavy_rain_threshold_mm", 20.0)
        if weather_info["precip_3day"] >= heavy_rain_threshold:
            st.error(f"⚠️ **Precipitation Warning**: High runoff risk. Postpone urea split-application to prevent nitrogen leaching.")
        else:
            st.success("✅ **Stable Weather**: Soil moisture is optimal. Safe for crop fertilizing.")

    with tab_preds:
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        # Display values
        for key, val in predictions.items():
            t_meta = [t for t in config.targets if t["name"] == key][0]
            desc = t_meta["description"]
            conf = val["confidence_score"]
            
            if conf >= 98.0:
                conf_color = "#2E7D32" # Gold standard
            elif conf >= 80:
                conf_color = "#1565C0"
            elif conf >= 60:
                conf_color = "#8E24AA"
            else:
                conf_color = "#EF6C00"
                
            st.markdown(
                f"""
                <div style="border: 1px solid rgba(0,0,0,0.08); border-radius: 12px; padding: 14px; margin-bottom: 12px; background-color: #F5F7FA; box-shadow: 0 2px 5px rgba(0,0,0,0.02);">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 700; color: #2C3E50; font-size: 0.95rem;">{desc} ({t_meta['name'].upper()})</span>
                        <span style="background-color: {conf_color}; color: white; padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.3px;">
                            {conf:.1f}% Conf.
                        </span>
                    </div>
                    <div style="margin-top: 6px; display: flex; justify-content: space-between; align-items: baseline;">
                        <div>
                            <span style="font-size: 1.7rem; font-weight: 800; color: #2E7D32;">{val['prediction']}</span>
                            <span class="metric-unit">{val['unit']}</span>
                        </div>
                        <div style="font-size: 0.8rem; color: #555; font-weight: 600;">
                            90% PI: [{val['lower_bound']} - {val['upper_bound']}]
                        </div>
                    </div>
                    <div style="font-size: 0.78rem; color: #777; font-style: italic; margin-top: 4px; border-top: 1px dashed rgba(0,0,0,0.06); padding-top: 4px;">
                        {val['evidence']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
    with tab_explain:
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        st.markdown("<h5 style='margin: 0 0 6px 0; color: #2C3E50; font-weight:700;'>Attribution Explanations</h5>", unsafe_allow_html=True)
        
        selected_exp_target = st.selectbox(
            "Target Soil Property:",
            options=list(predictions.keys()),
            format_func=lambda x: [t for t in config.targets if t["name"] == x][0]["description"],
            key="shap_target_selector"
        )
        
        exp_data = predictions[selected_exp_target]["explanations"]
        
        feats_list = [item["feature"] for item in exp_data]
        p_vals = [item["percentage"] for item in exp_data]
        dirs = [item["direction"] for item in exp_data]
        
        labels = []
        colors = []
        for f, p, d in zip(feats_list, p_vals, dirs):
            sign = "+" if d == 1 else "-"
            labels.append(f"{f} ({sign}{p:.1f}%)")
            colors.append("#2E7D32" if d == 1 else "#C62828")
            
        fig_exp = go.Figure(go.Bar(
            x=p_vals[::-1],
            y=labels[::-1],
            orientation='h',
            marker_color=colors[::-1],
            text=[f"{'+' if d == 1 else '-'}{p:.1f}%" for p, d in zip(p_vals[::-1], dirs[::-1])],
            textposition="auto"
        ))
        
        fig_exp.update_layout(
            xaxis_title="Relative Attribution Strength (%)",
            height=280,
            margin=dict(t=5, b=25, l=45, r=40),
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.1)")
        )
        st.plotly_chart(fig_exp, use_container_width=True)

    with tab_agronomy:
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        
        # Build narrative report
        h_score = advisory["health_score"]
        n_status = "deficient" if pred_vals["nitrogen"] < 35 else "sufficient"
        p_status = "deficient" if pred_vals["phosphorus"] < 15 else "sufficient"
        k_status = "deficient" if pred_vals["potassium"] < 120 else "sufficient"
        ph_val = pred_vals["ph"]
        soc_val = pred_vals["soc"]
        
        # Get crop recommendations
        best_crops = suitability_eng.evaluate_suitability(pred_vals)
        sorted_crops = sorted(best_crops.items(), key=lambda x: x[1]["suitability_score"], reverse=True)
        
        weather_info = fetch_weather_forecast(curr_lat, curr_lon)
        heavy_rain_risk = "High" if weather_info["precip_3day"] >= config.get("weather", {}).get("heavy_rain_threshold_mm", 20.0) else "Low"
        
        calibration_badge = "🟢 CALIBRATED WITH LAB REPORT OVERRIDE" if lab_override else "🟡 Remote Sensing Satellite & SoilGrids Inferences"
        
        narrative_report = f"""
        ### 📜 AI Agronomist Narrative Assessment Report
        **TerraSense AI Precision Agriculture Advisory Engine**
        
        ---
        
        #### 📍 Location Context
        * **Village/Town**: {loc_meta['village']}
        * **Mandal/Tehsil**: {loc_meta['mandal']}
        * **District**: {loc_meta['district']}
        * **State**: {loc_meta['state']}
        * **Coordinates**: `{curr_lat:.6f}°N, {curr_lon:.6f}°E`
        
        #### 🩺 Soil Health & Chemistry Profile
        * **Soil Health Index**: **{h_score}/100**
        * **pH Level**: **{ph_val:.2f}** ({'Acidic - requires lime' if ph_val < 5.8 else 'Alkaline - requires gypsum' if ph_val > 7.8 else 'Stable / Ideal'})
        * **Soil Organic Carbon (SOC)**: **{soc_val:.2f}%** ({'Low' if soc_val < 0.8 else 'Sufficient'})
        * **Dominant Textures (SoilGrids)**: Clay: **{point_features.get('clay', 0.0):.1f}%**, Sand: **{point_features.get('sand', 0.0):.1f}%**, Silt: **{point_features.get('silt', 0.0):.1f}%**
        
        #### 🧪 Nutrient Summary
        * **Available Nitrogen (N)**: **{pred_vals['nitrogen']:.1f} mg/kg** ({n_status.upper()})
        * **Available Phosphorus (P)**: **{pred_vals['phosphorus']:.1f} mg/kg** ({p_status.upper()})
        * **Available Potassium (K)**: **{pred_vals['potassium']:.1f} mg/kg** ({k_status.upper()})
        
        #### 🌾 Top Mapped Crop Suitability
        1. **{sorted_crops[0][1]['crop_name']}** (Score: **{sorted_crops[0][1]['suitability_score']}/100**) — *Highly Recommended*
        2. **{sorted_crops[1][1]['crop_name']}** (Score: **{sorted_crops[1][1]['suitability_score']}/100**)
        3. **{sorted_crops[2][1]['crop_name']}** (Score: **{sorted_crops[2][1]['suitability_score']}/100**)
        
        #### 🧪 Precision Fertilizer Schedule (Target Crop: {advisory['crop_name']})
        * **Urea Recommended Dosage**: **{advisory['fertilizers']['urea_total_kg_ha']} kg/ha**
        * **DAP Recommended Dosage**: **{advisory['fertilizers']['dap_total_kg_ha']} kg/ha** (Basal application)
        * **MOP Recommended Dosage**: **{advisory['fertilizers']['mop_total_kg_ha']} kg/ha** (Basal application)
        * **Soil Conditioners**: {', '.join([c['name'] for c in advisory['conditioners']]) if advisory['conditioners'] else 'None required'}
        
        #### 🌦️ Meteorological Feed Risks
        * **3-Day Forecast Precipitation**: **{weather_info['precip_3day']:.1f} mm**
        * **Heavy Rain Risk**: **{heavy_rain_risk.upper()}**
        * **Directive**: {"⚠️ DELAY Urea split-application immediately due to heavy runoff forecasts." if heavy_rain_risk == "High" else "✅ Conditions stable. Safe for immediate top-dressing split-applications."}
        
        #### 🛡️ Platform Trust & Confidence
        * **Uncertainty Calibration Status**: **{calibration_badge}**
        * **Model Ensemble Confidence**: **{predictions['nitrogen']['confidence_score']:.1f}%**
        """
        st.markdown(narrative_report)
