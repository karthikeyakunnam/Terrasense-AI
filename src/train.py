import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, List
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
try:
    from xgboost import XGBRegressor
    # Instantiate to verify C library loads
    _ = XGBRegressor()
    HAS_XGBOOST = True
except (ImportError, Exception) as e:
    HAS_XGBOOST = False

from src.config import Config
from src.logger import setup_logger
from src.data_pipeline import SoilDataPipeline

logger = setup_logger("train", level="INFO")

if not HAS_XGBOOST:
    logger.warning("XGBoost library could not be loaded (e.g., missing libomp on macOS). A fallback to Scikit-Learn's HistGradientBoostingRegressor will be used.")


class SoilModelTrainer:
    """Handles model training, validation, metric comparison, selection, and serialization."""
    
    def __init__(self, config: Config):
        self.config = config
        self.paths_cfg = config.paths
        self.train_cfg = config.training
        
        self.models_dir = Path(self.paths_cfg["models_dir"])
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.targets = self.config.targets
        
        # Assemble feature columns list
        self.feature_cols = (
            self.config.features["satellite_bands"] +
            self.config.features["vegetation_indices"] +
            self.config.features["soilgrids"]
        )
        
    def load_training_data(self) -> pd.DataFrame:
        """Loads or builds aligned preprocessed dataset."""
        processed_file = Path(self.paths_cfg["processed_data_dir"]) / "aligned_soil_dataset.csv"
        
        if not processed_file.exists():
            logger.info("Aligned soil dataset not found. Triggering spatial data pipeline...")
            pipeline = SoilDataPipeline(self.config)
            df = pipeline.run_ingestion_pipeline()
        else:
            logger.info(f"Loading preprocessed aligned soil dataset from {processed_file}...")
            df = pd.read_csv(processed_file)
            
        return df
        
    def evaluate_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Trains and evaluates RF, GBDT, and XGBoost models for each soil target property.
        
        Returns:
            Dictionary containing model selection details and metric statistics.
        """
        logger.info(f"Starting ML model evaluations. Training Features: {self.feature_cols}")
        
        results = {}
        best_models_metadata = {}
        
        X = df[self.feature_cols]
        random_state = self.train_cfg.get("random_state", 42)
        test_size = self.train_cfg.get("test_size", 0.2)
        cv_splits = self.train_cfg.get("cv_splits", 5)
        
        for target in self.targets:
            target_name = target["name"]
            target_col = target["column"]
            y = df[target_col]
            
            logger.info(f"--- Training models for target variable: {target_name} ({target_col}) ---")
            
            # Train/Test Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Define models to compare
            rf_params = self.train_cfg["models"]["random_forest"]
            gb_params = self.train_cfg["models"]["gradient_boosting"]
            xgb_params = self.train_cfg["models"]["xgboost"]
            
            model_candidates = {
                "RandomForest": RandomForestRegressor(
                    n_estimators=rf_params["n_estimators"],
                    max_depth=rf_params["max_depth"],
                    min_samples_split=rf_params["min_samples_split"],
                    random_state=random_state,
                    n_jobs=-1
                ),
                "GradientBoosting": GradientBoostingRegressor(
                    n_estimators=gb_params["n_estimators"],
                    learning_rate=gb_params["learning_rate"],
                    max_depth=gb_params["max_depth"],
                    random_state=random_state
                )
            }
            
            if HAS_XGBOOST:
                model_candidates["XGBoost"] = XGBRegressor(
                    n_estimators=xgb_params["n_estimators"],
                    learning_rate=xgb_params["learning_rate"],
                    max_depth=xgb_params["max_depth"],
                    random_state=random_state,
                    n_jobs=-1
                )
            else:
                from sklearn.ensemble import HistGradientBoostingRegressor
                model_candidates["HistGradientBoosting"] = HistGradientBoostingRegressor(
                    max_iter=xgb_params["n_estimators"],
                    learning_rate=xgb_params["learning_rate"],
                    max_depth=xgb_params["max_depth"],
                    random_state=random_state
                )

            
            target_metrics = {}
            best_model_name = None
            best_r2 = -999.0
            best_trained_model = None
            
            for model_name, model in model_candidates.items():
                # K-Fold Cross Validation
                kf = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
                cv_r2_scores = []
                cv_rmse_scores = []
                cv_mae_scores = []
                
                for train_idx, val_idx in kf.split(X_train):
                    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    
                    # Fit
                    model.fit(X_tr, y_tr)
                    preds = model.predict(X_val)
                    
                    cv_r2_scores.append(r2_score(y_val, preds))
                    cv_rmse_scores.append(np.sqrt(mean_squared_error(y_val, preds)))
                    cv_mae_scores.append(mean_absolute_error(y_val, preds))
                    
                # Full test set evaluation
                model.fit(X_train, y_train)
                test_preds = model.predict(X_test)
                
                test_r2 = r2_score(y_test, test_preds)
                test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
                test_mae = mean_absolute_error(y_test, test_preds)
                
                target_metrics[model_name] = {
                    "cv_r2_mean": float(np.mean(cv_r2_scores)),
                    "cv_r2_std": float(np.std(cv_r2_scores)),
                    "cv_rmse_mean": float(np.mean(cv_rmse_scores)),
                    "cv_mae_mean": float(np.mean(cv_mae_scores)),
                    "test_r2": float(test_r2),
                    "test_rmse": float(test_rmse),
                    "test_mae": float(test_mae)
                }
                
                logger.info(
                    f"Model: {model_name:18} | CV R²: {np.mean(cv_r2_scores):.4f} | "
                    f"Test R²: {test_r2:.4f} | Test RMSE: {test_rmse:.4f}"
                )
                
                # Check if this model is the best (based on Test R2 score)
                if test_r2 > best_r2:
                    best_r2 = test_r2
                    best_model_name = model_name
                    best_trained_model = model
            
            logger.info(f"Winner for {target_name}: {best_model_name} (R² = {best_r2:.4f})")
            
            # Serialize the best model
            # Note: For Random Forest, we will keep it as the default or always save it as a backup
            # because we need the ensemble tree outputs for the confidence scoring variance calculations.
            # If XGBoost/GBDT is selected as best, we save it as the primary inference engine,
            # but we also save the RandomForest regressor for that target so we can derive tree variance.
            # To simplify and ensure high performance, we train and save BOTH the best model AND the RandomForest model.
            
            best_model_filename = f"{target_name}_best_model.joblib"
            joblib.dump(best_trained_model, self.models_dir / best_model_filename)
            
            # Extract and save RF specifically for confidence estimation calculations
            rf_model = model_candidates["RandomForest"]
            rf_model.fit(X_train, y_train)
            joblib.dump(rf_model, self.models_dir / f"{target_name}_rf_model.joblib")
            
            # Calculate feature importances from best model, fallback to RF model if unavailable
            importances = []
            if hasattr(best_trained_model, "feature_importances_"):
                importances = best_trained_model.feature_importances_.tolist()
            elif hasattr(rf_model, "feature_importances_"):
                importances = rf_model.feature_importances_.tolist()
                
            feature_imp_map = dict(zip(self.feature_cols, importances))
            
            # Sort importances
            feature_imp_map = dict(sorted(feature_imp_map.items(), key=lambda item: item[1], reverse=True))
            
            best_models_metadata[target_name] = {
                "selected_model": best_model_name,
                "test_r2": float(best_r2),
                "test_rmse": float(target_metrics[best_model_name]["test_rmse"]),
                "test_mae": float(target_metrics[best_model_name]["test_mae"]),
                "feature_importances": feature_imp_map,
                "all_metrics": target_metrics
            }
            
        # Write metadata JSON
        metrics_file = self.models_dir / "training_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(best_models_metadata, f, indent=4)
        logger.info(f"Model training metadata exported successfully to {metrics_file}")
        
        return best_models_metadata

if __name__ == "__main__":
    config = Config()
    trainer = SoilModelTrainer(config)
    df = trainer.load_training_data()
    trainer.evaluate_models(df)
