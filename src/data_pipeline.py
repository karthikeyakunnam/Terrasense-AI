import os
import pandas as pd
import numpy as np
import rasterio
from pathlib import Path
from typing import Dict, Any, List, Tuple

from src.config import Config
from src.logger import setup_logger
from src.utils import wgs84_to_utm
from src.features import engineer_features
from src.data_generator import SpatialDataGenerator

logger = setup_logger("data_pipeline", level="INFO")

class SoilDataPipeline:
    """Manages raw spatial ingestion, alignment, feature extraction, and dataset production."""
    
    def __init__(self, config: Config):
        self.config = config
        self.paths_cfg = config.paths
        self.spatial_cfg = config.spatial
        
        self.raw_dir = Path(self.paths_cfg["raw_data_dir"])
        self.processed_dir = Path(self.paths_cfg["processed_data_dir"])
        
        self.utm_crs = self.spatial_cfg["crs"]
        self.bands = self.config.features["satellite_bands"]
        self.soilgrids_vars = self.config.features["soilgrids"]
        
    def _verify_raw_data(self) -> None:
        """Verifies if raw data assets are present. If missing, triggers synthetic generator."""
        required_rasters = [f"sentinel2_{b}.tif" for b in self.bands] + [f"soilgrids_{s}.tif" for s in self.soilgrids_vars]
        required_csv = "lab_soil_tests.csv"
        
        missing = []
        for file in required_rasters:
            if not (self.raw_dir / file).exists():
                missing.append(file)
        if not (self.raw_dir / required_csv).exists():
            missing.append(required_csv)
            
        if missing:
            logger.warning(f"Raw spatial data files are missing: {missing}. Automatically running generator...")
            generator = SpatialDataGenerator(self.config)
            generator.generate_all()
        else:
            logger.info("All raw data files verified successfully.")
            
    def run_ingestion_pipeline(self) -> pd.DataFrame:
        """Runs the entire spatial extraction and tabular alignment pipeline.
        
        Returns:
            DataFrame containing aligned features and labels ready for model training.
        """
        self._verify_raw_data()
        
        # Load lab test CSV
        csv_path = self.raw_dir / "lab_soil_tests.csv"
        logger.info(f"Loading laboratory test points from {csv_path}...")
        lab_df = pd.read_csv(csv_path)
        
        # Project Lat/Lon coordinates to UTM Easting/Northing
        logger.info(f"Projecting coordinate system to target CRS: {self.utm_crs}...")
        easting, northing = wgs84_to_utm(lab_df["longitude"], lab_df["latitude"], self.utm_crs)
        lab_df["easting"] = easting
        lab_df["northing"] = northing
        
        # Read values from Sentinel-2 bands at lab coordinate locations
        logger.info("Sampling Sentinel-2 multi-band rasters at test point coordinates...")
        for band in self.bands:
            raster_path = self.raw_dir / f"sentinel2_{band}.tif"
            lab_df[band] = self._sample_raster_at_coords(raster_path, lab_df["easting"], lab_df["northing"])
            
        # Read values from SoilGrids maps at lab coordinate locations
        logger.info("Sampling SoilGrids maps at test point coordinates...")
        for var in self.soilgrids_vars:
            raster_path = self.raw_dir / f"soilgrids_{var}.tif"
            lab_df[var] = self._sample_raster_at_coords(raster_path, lab_df["easting"], lab_df["northing"])
            
        # Extract features (Spectral indices like NDVI, EVI, NDRE)
        logger.info("Calculating vegetation indices and performing feature engineering...")
        processed_df = engineer_features(lab_df)
        
        # Save processed dataset
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        output_file = self.processed_dir / "aligned_soil_dataset.csv"
        processed_df.to_csv(output_file, index=False)
        logger.info(f"Aligned and preprocessed data exported successfully to {output_file} (shape: {processed_df.shape})")
        
        return processed_df
        
    def _sample_raster_at_coords(
        self,
        raster_path: Path,
        eastings: pd.Series,
        northings: pd.Series
    ) -> List[float]:
        """Extracts cell values from a raster at the given easting/northing UTM coordinates.
        
        Args:
            raster_path: Path to the GeoTIFF raster.
            eastings: Series of UTM Easting coordinates.
            northings: Series of UTM Northing coordinates.
            
        Returns:
            List of float values extracted from the raster.
        """
        values = []
        with rasterio.open(raster_path) as src:
            # Check for CRS compatibility
            if src.crs.to_string() != self.utm_crs:
                logger.warning(
                    f"Raster CRS ({src.crs.to_string()}) does not match system target CRS ({self.utm_crs}). "
                    "Automatic warping may be required in production."
                )
            
            # Extract coordinates as pairs
            coords = list(zip(eastings, northings))
            
            # Use rasterio sample generator
            for val_arr in src.sample(coords):
                # val_arr is a numpy array of shape (count,) where count is bands count (1 here)
                values.append(float(val_arr[0]))
                
        return values

if __name__ == "__main__":
    config = Config()
    pipeline = SoilDataPipeline(config)
    pipeline.run_ingestion_pipeline()
