import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Union, List, Tuple

from src.config import Config
from src.logger import setup_logger
from src.utils import calculate_confidence_score

logger = setup_logger("inference", level="INFO")

class SoilPredictor:
    """Loads trained estimators and runs predictions with ensemble-based uncertainty bounds."""
    
    def __init__(self, config: Config):
        self.config = config
        self.paths_cfg = config.paths
        self.models_dir = Path(self.paths_cfg["models_dir"])
        self.targets = self.config.targets
        
        self.feature_cols = (
            self.config.features["satellite_bands"] +
            self.config.features["vegetation_indices"] +
            self.config.features["soilgrids"]
        )
        
        # Load baseline statistics for explainability attributions
        processed_file = Path(self.paths_cfg["processed_data_dir"]) / "aligned_soil_dataset.csv"
        if processed_file.exists():
            try:
                df_train = pd.read_csv(processed_file)
                self.feature_means = df_train[self.feature_cols].mean().to_dict()
                self.feature_stds = df_train[self.feature_cols].std().replace(0, 1e-6).to_dict()
            except Exception as e:
                logger.warning(f"Failed to calculate training statistics: {e}. Using fallback defaults.")
                self.feature_means = {feat: 0.5 for feat in self.feature_cols}
                self.feature_stds = {feat: 0.25 for feat in self.feature_cols}
        else:
            self.feature_means = {feat: 0.5 for feat in self.feature_cols}
            self.feature_stds = {feat: 0.25 for feat in self.feature_cols}
            
        # Load models and training metadata
        self.models = {}
        self.rf_models = {} # Stored separately for uncertainty calculations
        self.metrics_metadata = {}
        self._load_models()
        
    def _load_models(self) -> None:
        """Loads serialized models and validation metrics metadata."""
        metrics_file = self.models_dir / "training_metrics.json"
        
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                self.metrics_metadata = json.load(f)
        else:
            logger.warning("Training metrics JSON not found. Calibrating uncertainty with default parameters.")
            
        for target in self.targets:
            target_name = target["name"]
            best_model_path = self.models_dir / f"{target_name}_best_model.joblib"
            rf_model_path = self.models_dir / f"{target_name}_rf_model.joblib"
            
            if not best_model_path.exists() or not rf_model_path.exists():
                logger.error(f"Models for target '{target_name}' are missing. Run train.py first.")
                continue
                
            self.models[target_name] = joblib.load(best_model_path)
            self.rf_models[target_name] = joblib.load(rf_model_path)
            
        logger.info(f"Loaded predictors for targets: {list(self.models.keys())}")
        
    def predict_point(self, features: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """Runs predictions, confidence scoring, and intervals for a single coordinate location.
        
        Args:
            features: Dictionary containing feature names and their values.
            
        Returns:
            Dictionary containing prediction, confidence, and bounds for each target.
        """
        # Convert single sample dict to DataFrame
        df = pd.DataFrame([features])[self.feature_cols]
        
        results = {}
        for target in self.targets:
            target_name = target["name"]
            
            if target_name not in self.models:
                continue
                
            model = self.models[target_name]
            rf_model = self.rf_models[target_name]
            
            # 1. Primary Prediction
            pred_val = float(model.predict(df)[0])
            
            # 2. Uncertainty Quantification via RF Ensemble Variance
            # Extract tree predictions
            tree_preds = np.array([tree.predict(df.values)[0] for tree in rf_model.estimators_])
            std_dev = float(np.std(tree_preds))
            
            # 3. Confidence Score Calibration
            # Use test RMSE as the decay constant scaling factor.
            # If RMSE is not available, default to 10% of typical target value range.
            target_metadata = self.metrics_metadata.get(target_name, {})
            test_rmse = target_metadata.get("test_rmse", None)
            
            if test_rmse is None or test_rmse <= 0:
                # Default calibration factors
                default_calibrations = {
                    "nitrogen": 15.0,
                    "phosphorus": 8.0,
                    "potassium": 30.0,
                    "soc": 0.25,
                    "ph": 0.35
                }
                scale = default_calibrations.get(target_name, 1.0)
            else:
                scale = test_rmse
                
            # Compute confidence score: C = exp(-std_dev / scale) * 100
            # A standard deviation equal to one RMSE gives ~36.8% confidence.
            # A standard deviation of half RMSE gives ~60.6% confidence.
            lambda_factor = 1.0 / scale
            confidence_score = calculate_confidence_score(std_dev, calibration_factor=lambda_factor)
            
            # 4. Calculate 90% Prediction Interval: Mean +/- 1.645 * std_dev
            lower_bound = float(pred_val - 1.645 * std_dev)
            upper_bound = float(pred_val + 1.645 * std_dev)
            
            # Post-processing boundaries (e.g. pH can't exceed 14, nutrients can't be negative)
            if target_name == "ph":
                lower_bound = max(0.0, min(14.0, lower_bound))
                upper_bound = max(0.0, min(14.0, upper_bound))
            else:
                lower_bound = max(0.0, lower_bound)
                upper_bound = max(0.0, upper_bound)
                
            # 5. Calculate Local Feature Attribution for Explanations
            feature_imp = target_metadata.get("feature_importances", {})
            
            raw_attrs = {}
            for feat in self.feature_cols:
                val = features.get(feat, 0.0)
                mean = self.feature_means.get(feat, 0.5)
                std = self.feature_stds.get(feat, 0.25)
                importance = feature_imp.get(feat, 0.0)
                
                # Z-Score deviation from training baseline
                z_score = (val - mean) / std
                raw_attrs[feat] = z_score * importance
                
            # Compute relative percentage attributions
            abs_sum = sum(abs(v) for v in raw_attrs.values())
            if abs_sum == 0:
                abs_sum = 1e-6
                
            explanations = []
            for feat, attr in raw_attrs.items():
                pct = (abs(attr) / abs_sum) * 100.0
                direction = 1 if attr >= 0 else -1
                explanations.append({
                    "feature": feat,
                    "percentage": round(pct, 1),
                    "direction": direction
                })
                
            # Sort by percentage descending, take top 5
            explanations = sorted(explanations, key=lambda x: x["percentage"], reverse=True)[:5]
            
            top_features = list(feature_imp.keys())[:3]
            evidence = ", ".join([f"{feat} ({features.get(feat, 0):.2f})" for feat in top_features])
            
            results[target_name] = {
                "prediction": round(pred_val, 2),
                "confidence_score": round(confidence_score, 1),
                "lower_bound": round(lower_bound, 2),
                "upper_bound": round(upper_bound, 2),
                "std_dev": round(std_dev, 3),
                "unit": target["unit"],
                "evidence": f"Top drivers: {evidence}",
                "explanations": explanations
            }
            
        return results

    def predict_grid(self, df_grid: pd.DataFrame) -> pd.DataFrame:
        """Runs batch predictions and confidence values for a grid. Useful for heatmap generation.
        
        Args:
            df_grid: DataFrame containing features at all coordinate grid cells.
            
        Returns:
            DataFrame with predictions and confidence scores appended for all targets.
        """
        out_df = df_grid.copy()
        X = df_grid[self.feature_cols]
        
        for target in self.targets:
            target_name = target["name"]
            if target_name not in self.models:
                continue
                
            model = self.models[target_name]
            rf_model = self.rf_models[target_name]
            
            # Batch Prediction
            out_df[f"{target_name}_pred"] = model.predict(X)
            
            # Batch Tree Predictions for Standard Deviation
            # shape: (n_samples, n_trees)
            logger.info(f"Computing uncertainty grid for {target_name}...")
            tree_preds = np.column_stack([tree.predict(X.values) for tree in rf_model.estimators_])
            std_devs = np.std(tree_preds, axis=1)
            
            target_metadata = self.metrics_metadata.get(target_name, {})
            test_rmse = target_metadata.get("test_rmse", 1.0)
            lambda_factor = 1.0 / (test_rmse if test_rmse > 0 else 1.0)
            
            # Vectorized confidence score
            conf_scores = np.exp(-lambda_factor * std_devs) * 100.0
            out_df[f"{target_name}_confidence"] = np.clip(conf_scores, 0.0, 100.0)
            
        return out_df

if __name__ == "__main__":
    # Test predictor load
    config = Config()
    try:
        predictor = SoilPredictor(config)
        # Mock features
        mock_feat = {feat: 0.5 for feat in predictor.feature_cols}
        res = predictor.predict_point(mock_feat)
        print("Mock point inference results:")
        print(json.dumps(res, indent=4))
    except Exception as e:
        print(f"Prediction test error: {e}")
