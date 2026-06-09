import os
import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

st.markdown("# 📈 ML Model Performance & Validation Dashboard")
st.markdown("##### Detailed evaluation metrics, cross-validation runs, and feature importances for all predictive models.")

config = st.session_state.get("config")
predictor = st.session_state.get("predictor")

if config is None or predictor is None:
    st.warning("Please visit the landing page first to initialize assets.")
    st.stop()

# ----------------- LOAD METRICS METADATA -----------------

models_dir = Path(config.paths["models_dir"])
metrics_file = models_dir / "training_metrics.json"

if not metrics_file.exists():
    st.error("Training metrics JSON not found. Please run train.py to generate checkpoints and statistics.")
    st.stop()
    
with open(metrics_file, "r") as f:
    metrics_metadata = json.load(f)

# ----------------- UI VIEW CONTROLS -----------------

target_options = {t["name"]: t["description"] for t in config.targets}
selected_target = st.selectbox(
    "Choose soil property to review metrics:",
    options=list(target_options.keys()),
    format_func=lambda x: target_options[x]
)

target_data = metrics_metadata.get(selected_target)

if not target_data:
    st.warning(f"No validation data found for target '{selected_target}'.")
    st.stop()
    
st.subheader(f"Evaluation Details: {target_options[selected_target]}")

st.markdown(
    f"""
    <div style="background-color: #E3F2FD; border-left: 6px solid #1E88E5; border-radius: 8px; padding: 15px; margin-bottom: 20px;">
        <h4 style="margin: 0; color: #0D47A1;">🏆 Best Performing Model: <b>{target_data['selected_model']}</b></h4>
        <p style="margin: 5px 0 0 0; font-size: 0.9rem; color: #1565C0;">
            Test R² Score: <b>{target_data['test_r2']:.4f}</b> | Test RMSE: <b>{target_data['test_rmse']:.4f}</b> | Test MAE: <b>{target_data['test_mae']:.4f}</b>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

col_table, col_importance = st.columns([6, 6])

with col_table:
    st.markdown("### 📊 Model Comparisons (5-Fold CV & Test Set)")
    
    comparison_rows = []
    for model_name, model_metrics in target_data["all_metrics"].items():
        comparison_rows.append({
            "Model": model_name,
            "CV R² (Mean)": f"{model_metrics['cv_r2_mean']:.4f} ± {model_metrics['cv_r2_std']:.4f}",
            "CV RMSE (Mean)": f"{model_metrics['cv_rmse_mean']:.4f}",
            "Test R²": f"{model_metrics['test_r2']:.4f}",
            "Test RMSE": f"{model_metrics['test_rmse']:.4f}",
            "Test MAE": f"{model_metrics['test_mae']:.4f}"
        })
    df_comparison = pd.DataFrame(comparison_rows)
    st.table(df_comparison.set_index("Model"))
    
    st.markdown(
        """
        <div style="background-color: #f9f9f9; padding: 15px; border-radius: 8px; border: 1px solid #eee; margin-top: 10px;">
            <small style="color: #666; font-weight: 600;">SELECTION CRITERIA</small>
            <p style="font-size: 0.8rem; color: #555; margin: 5px 0 0 0;">
                The platform automatically compares Random Forest, Gradient Boosting, and XGBoost (HistGradientBoosting fallback). 
                The model showing the highest R² score on the hold-out test set is serialized as the production-grade predictor. 
                A secondary Random Forest model is trained concurrently for every target to support tree-based variance calculations in the uncertainty estimation engine.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_importance:
    st.markdown("### 🧬 Feature Importance Ratings")
    
    importance_dict = target_data.get("feature_importances", {})
    
    if not importance_dict:
        try:
            predictor_obj = st.session_state.get("predictor")
            if predictor_obj:
                rf_model = predictor_obj.rf_models.get(selected_target)
                if rf_model and hasattr(rf_model, "feature_importances_"):
                    importances = rf_model.feature_importances_
                    importance_dict = dict(zip(predictor_obj.feature_cols, importances))
                    importance_dict = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))
                    st.caption("ℹ️ *Feature importances estimated from consensus Random Forest validation*")
        except Exception:
            pass
            
    if not importance_dict:
        st.write("Feature importances not available for the selected model.")
    else:
        feats_sorted = list(importance_dict.keys())[:10]
        vals_sorted = [importance_dict[f] for f in feats_sorted]
        
        fig_imp = go.Figure(go.Bar(
            x=vals_sorted[::-1],
            y=feats_sorted[::-1],
            orientation='h',
            marker_color="#1E88E5",
            text=[f"{v*100:.1f}%" for v in vals_sorted[::-1]],
            textposition="auto"
        ))
        
        fig_imp.update_layout(
            xaxis_title="Relative Feature Importance Score",
            yaxis_title="Feature Variable Name",
            height=380,
            margin=dict(t=10, b=30, l=40, r=40),
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_imp, use_container_width=True)
        
st.markdown("---")
st.markdown("### 🛰️ Feature Glossary Reference")
st.markdown(
    """
    *   **NDVI, EVI, NDRE**: Canopy greenness indices.
    *   **B02, B03, B04, B08, B11, B12**: Spectral bands.
    *   **sand, clay, silt**: Gridded texture properties.
    *   **bulk_density**: Soil dry density ($g/cm^3$).
    """
)
