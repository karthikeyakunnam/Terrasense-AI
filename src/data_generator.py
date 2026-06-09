import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon, Point
from pathlib import Path
from typing import Dict, Any

from src.config import Config
from src.logger import setup_logger
from src.utils import wgs84_to_utm, utm_to_wgs84

logger = setup_logger("data_generator", level="INFO")

class SpatialDataGenerator:
    """Generates realistic synthetic geospatial rasters, parcel polygons, and lab tests."""
    
    def __init__(self, config: Config):
        self.config = config
        self.spatial_cfg = config.spatial
        self.paths_cfg = config.paths
        
        self.min_x = self.spatial_cfg["extent"]["min_x"]
        self.max_x = self.spatial_cfg["extent"]["max_x"]
        self.min_y = self.spatial_cfg["extent"]["min_y"]
        self.max_y = self.spatial_cfg["extent"]["max_y"]
        
        self.width_m = self.max_x - self.min_x
        self.height_m = self.max_y - self.min_y
        
        self.resolution = self.spatial_cfg["raster_resolution"]
        self.cols = int(self.width_m / self.resolution)
        self.rows = int(self.height_m / self.resolution)
        
        # Spatial transform for Rasterio (origin is top-left: west, north)
        self.transform = from_origin(self.min_x, self.max_y, self.resolution, self.resolution)
        self.crs = self.spatial_cfg["crs"]
        
        # Output paths
        self.raw_dir = Path(self.paths_cfg["raw_data_dir"])
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_all(self, num_samples: int = 300) -> None:
        """Executes full synthetic dataset generation pipeline."""
        logger.info("Starting synthetic agricultural dataset generation...")
        
        # 1. Generate underlying spatial noise grids (simulating soil and canopy variations)
        soil_pattern = self._generate_spatial_noise(seed=42)
        veg_pattern = self._generate_spatial_noise(seed=84, scale=15.0)
        
        # 2. Write Sentinel-2 Bands
        # Typical values: Red (B04) is low in dense veg; NIR (B08) is high; SWIR (B11, B12) varies
        ndvi = 0.1 + 0.75 * veg_pattern # Normalized greenness (0.1 to 0.85)
        # Reconstruct B04 (Red) and B08 (NIR) from NDVI
        # NDVI = (NIR - Red) / (NIR + Red)
        # Let's model NIR directly as a function of veg, and solve for Red
        nir = 0.15 + 0.6 * veg_pattern
        red = nir * (1.0 - ndvi) / (1.0 + ndvi)
        green = 0.08 + 0.15 * veg_pattern
        blue = 0.05 + 0.08 * veg_pattern
        swir1 = 0.25 - 0.1 * soil_pattern
        swir2 = 0.20 - 0.12 * soil_pattern
        
        bands = {
            "B02": blue,
            "B03": green,
            "B04": red,
            "B08": nir,
            "B11": swir1,
            "B12": swir2
        }
        
        for name, data in bands.items():
            self._write_geotiff(self.raw_dir / f"sentinel2_{name}.tif", data)
            
        # 3. Write SoilGrids Maps
        # sand, clay, silt (must sum to roughly 100%), bulk density
        clay = 15.0 + 35.0 * soil_pattern  # 15% to 50%
        sand = 10.0 + 55.0 * (1.0 - soil_pattern) # 10% to 65%
        silt = 100.0 - clay - sand
        bulk_density = 1.6 - 0.4 * soil_pattern # 1.2 to 1.6 g/cm^3
        
        soilgrids = {
            "sand": sand,
            "clay": clay,
            "silt": silt,
            "bulk_density": bulk_density
        }
        
        for name, data in soilgrids.items():
            self._write_geotiff(self.raw_dir / f"soilgrids_{name}.tif", data)
            
        # 4. Generate Parcels boundaries (GeoJSON)
        parcels_gdf = self._generate_parcels()
        parcels_gdf.to_file(self.raw_dir / "parcels.geojson", driver="GeoJSON")
        
        # 5. Generate Lab Soil Samples (CSV)
        self._generate_lab_samples(
            num_samples=num_samples,
            bands=bands,
            soilgrids=soilgrids,
            parcels=parcels_gdf
        )
        
        logger.info("Data generation complete. All raw files created successfully.")
        
    def _generate_spatial_noise(self, seed: int, scale: float = 20.0) -> np.ndarray:
        """Generates continuous spatially correlated 2D noise (simulating terrain)."""
        np.random.seed(seed)
        # Create a grid of points
        x = np.linspace(0, 10, self.cols)
        y = np.linspace(0, 10, self.rows)
        xx, yy = np.meshgrid(x, y)
        
        # Multi-scale sinusoidal noise
        grid = (
            np.sin(xx / 2.0) * np.cos(yy / 2.0) +
            0.5 * np.sin(xx) * np.cos(yy) +
            0.25 * np.sin(xx * 2.0) * np.sin(yy * 2.0) +
            0.1 * np.random.randn(self.rows, self.cols)
        )
        
        # Normalize between 0 and 1
        grid = (grid - grid.min()) / (grid.max() - grid.min())
        return grid.astype(np.float32)
        
    def _write_geotiff(self, path: Path, data: np.ndarray) -> None:
        """Writes a single band numpy array to a GeoTIFF file."""
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=self.rows,
            width=self.cols,
            count=1,
            dtype=rasterio.float32,
            crs=self.crs,
            transform=self.transform
        ) as dst:
            dst.write(data, 1)
        logger.debug(f"Saved GeoTIFF: {path.name}")
        
    def _generate_parcels(self) -> gpd.GeoDataFrame:
        """Generates 5 distinct farm parcel polygons within the extent."""
        logger.info("Generating synthetic agricultural parcels...")
        
        # Divide our 10km x 10km grid into 5 rectangular/polygonal parcels
        # (UTM coordinates: X between 600,000 and 610,000; Y between 3,000,000 and 3,010,000)
        # Sub-divide the bounding box into quadrants with some buffers
        coords = [
            # Parcel 1: North-West quadrant
            ((600500, 3009500), (604500, 3009500), (604500, 3005500), (600500, 3005500)),
            # Parcel 2: North-East quadrant
            ((605500, 3009500), (609500, 3009500), (609500, 3005500), (605500, 3005500)),
            # Parcel 3: South-West quadrant
            ((600500, 3004500), (604500, 3004500), (604500, 3000500), (600500, 3000500)),
            # Parcel 4: South-East quadrant
            ((605500, 3004500), (609500, 3004500), (609500, 3000500), (605500, 3000500)),
            # Parcel 5: Central Valley parcel (overlapping parts, adjusted size)
            ((603000, 3006000), (607000, 3006000), (607000, 3004000), (603000, 3004000))
        ]
        
        parcel_names = ["Northwest Field A", "Northeast Field B", "Southwest Field C", "Southeast Field D", "Central Valley Field E"]
        crops = ["rice", "wheat", "maize", "cotton", "rice"]
        
        geometries = []
        for poly_coords in coords:
            geometries.append(Polygon(poly_coords))
            
        gdf = gpd.GeoDataFrame(
            {
                "parcel_id": [f"PARCEL_{i+1:03d}" for i in range(len(coords))],
                "parcel_name": parcel_names,
                "crop_type": crops,
                "area_ha": [poly.area / 10000.0 for poly in geometries]
            },
            geometry=geometries,
            crs=self.crs
        )
        # Convert geometries to WGS84 for GeoJSON representation
        gdf_wgs84 = gdf.to_crs("EPSG:4326")
        return gdf_wgs84
        
    def _generate_lab_samples(
        self,
        num_samples: int,
        bands: Dict[str, np.ndarray],
        soilgrids: Dict[str, np.ndarray],
        parcels: gpd.GeoDataFrame
    ) -> None:
        """Generates lab test samples correlated with coordinates and spatial rasters."""
        logger.info(f"Generating {num_samples} correlated soil lab test samples...")
        
        np.random.seed(10)
        
        # 1. Randomly sample coordinate points within the bounding box
        easting = np.random.uniform(self.min_x + 500, self.max_x - 500, num_samples)
        northing = np.random.uniform(self.min_y + 500, self.max_y - 500, num_samples)
        
        # Convert to WGS84 coordinates (lon, lat) for lab dataset
        lon, lat = utm_to_wgs84(easting, northing, self.crs)
        
        samples_data = []
        
        # Project parcels back to UTM for spatial point-in-polygon check
        parcels_utm = parcels.to_crs(self.crs)
        
        for i in range(num_samples):
            e, n = easting[i], northing[i]
            ln, lt = lon[i], lat[i]
            
            # Map UTM coordinates to raster pixel index
            row, col = rasterio.transform.rowcol(self.transform, e, n)
            row = np.clip(row, 0, self.rows - 1)
            col = np.clip(col, 0, self.cols - 1)
            
            # Extract underlying raster feature values for correlation
            b04_val = bands["B04"][row, col]
            b08_val = bands["B08"][row, col]
            ndvi_val = (b08_val - b04_val) / (b08_val + b04_val + 1e-6)
            
            clay_val = soilgrids["clay"][row, col]
            sand_val = soilgrids["sand"][row, col]
            bd_val = soilgrids["bulk_density"][row, col]
            
            # Point in polygon check to assign parcels and crop type
            point = Point(e, n)
            matching_parcel = parcels_utm[parcels_utm.contains(point)]
            
            if not matching_parcel.empty:
                parcel_id = matching_parcel.iloc[0]["parcel_id"]
                crop_type = matching_parcel.iloc[0]["crop_type"]
            else:
                parcel_id = "UNASSIGNED"
                crop_type = np.random.choice(["rice", "wheat", "maize", "cotton"])
                
            # Create correlated label values with physical noise
            # Soil organic carbon (SOC): high clay and high NDVI (organic matter)
            soc = 0.3 + 1.2 * ndvi_val + 1.0 * (clay_val / 100.0) + np.random.normal(0, 0.05)
            soc = np.clip(soc, 0.1, 3.0)
            
            # pH: high sand/leaching can be acidic, clay neutralizes or can be alkaline. Let's model:
            ph = 5.2 + 2.5 * (sand_val / 100.0) + 1.0 * (clay_val / 100.0) - 0.5 * ndvi_val + np.random.normal(0, 0.2)
            ph = np.clip(ph, 4.5, 8.5)
            
            # Nitrogen (N): highly correlated with NDVI (chlorophyll / greenness proxy)
            n_val = 15.0 + 110.0 * ndvi_val + 30.0 * soc + np.random.normal(0, 5.0)
            n_val = np.clip(n_val, 10.0, 160.0)
            
            # Phosphorus (P): correlated with organic carbon and clay properties
            p_val = 5.0 + 40.0 * (1.0 - sand_val / 100.0) + 20.0 * soc + np.random.normal(0, 3.0)
            p_val = np.clip(p_val, 2.0, 90.0)
            
            # Potassium (K): highly correlated with clay content
            k_val = 40.0 + 220.0 * (clay_val / 100.0) - 20.0 * bd_val + np.random.normal(0, 10.0)
            k_val = np.clip(k_val, 30.0, 350.0)
            
            samples_data.append({
                "sample_id": f"SAMP_{i+1:04d}",
                "longitude": ln,
                "latitude": lt,
                "parcel_id": parcel_id,
                "crop_type": crop_type,
                "N_mg_kg": round(float(n_val), 2),
                "P_mg_kg": round(float(p_val), 2),
                "K_mg_kg": round(float(k_val), 2),
                "SOC_percent": round(float(soc), 2),
                "pH": round(float(ph), 2)
            })
            
        df = pd.DataFrame(samples_data)
        df.to_csv(self.raw_dir / "lab_soil_tests.csv", index=False)
        logger.info(f"Saved laboratory test records to {self.raw_dir / 'lab_soil_tests.csv'}")

if __name__ == "__main__":
    config = Config()
    generator = SpatialDataGenerator(config)
    generator.generate_all()
