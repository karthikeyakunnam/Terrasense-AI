import os
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import Config
from src.utils import wgs84_to_utm, utm_to_wgs84, calculate_confidence_score
from src.features import calculate_ndvi, calculate_evi, calculate_ndre, engineer_features
from src.recommender import CropSuitabilityEngine, FertilizerRecommender

def test_config_loader():
    """Tests if config loader successfully parses YAML and generates directories."""
    config = Config()
    assert config.paths is not None
    assert "raw_data_dir" in config.paths
    assert config.spatial["crs"] == "EPSG:32643"
    assert len(config.targets) == 5

def test_coordinate_transforms():
    """Tests if coordinate projection functions are exact round-trip inverses."""
    # Centroid of UTM Zone 43N area: around Lon 73, Lat 27
    test_lon, test_lat = 73.1, 27.2
    
    # Forward: WGS84 -> UTM
    easting, northing = wgs84_to_utm(test_lon, test_lat, "EPSG:32643")
    assert easting > 0
    assert northing > 0
    
    # Reverse: UTM -> WGS84
    lon_back, lat_back = utm_to_wgs84(easting, northing, "EPSG:32643")
    
    # Round-trip verification (close precision threshold)
    assert pytest.approx(test_lon, abs=1e-5) == lon_back
    assert pytest.approx(test_lat, abs=1e-5) == lat_back

def test_feature_indices():
    """Tests calculations of NDVI, EVI, and NDRE indices."""
    red = 0.1
    nir = 0.5
    blue = 0.05
    green = 0.15
    
    ndvi = calculate_ndvi(red, nir)
    assert ndvi == pytest.approx(0.66666, abs=1e-3)
    
    # EVI: 2.5 * (0.5 - 0.1) / (0.5 + 6*0.1 - 7.5*0.05 + 1)
    # = 1.0 / (0.5 + 0.6 - 0.375 + 1) = 1.0 / 1.725 = 0.5797
    evi = calculate_evi(blue, red, nir)
    assert evi == pytest.approx(0.5797, abs=1e-3)
    
    ndre = calculate_ndre(green, nir)
    assert ndre == pytest.approx(0.5384, abs=1e-3)

def test_confidence_math():
    """Tests exponential decay mapping of prediction standard deviation to confidence score."""
    # Std dev 0 should have 100% confidence
    assert calculate_confidence_score(0.0) == 100.0
    
    # Higher std dev should reduce confidence
    c1 = calculate_confidence_score(0.1, calibration_factor=1.0)
    c2 = calculate_confidence_score(0.5, calibration_factor=1.0)
    assert c1 > c2
    assert 0.0 <= c1 <= 100.0
    assert 0.0 <= c2 <= 100.0

def test_recommender_agronomy():
    """Tests logic of fertilizer recommendations, ensuring DAP Nitrogen correction."""
    config = Config()
    recommender = FertilizerRecommender(config)
    
    # Low nutrient values to trigger fertilizer requirements
    predictions = {
        "nitrogen": 20.0,
        "phosphorus": 5.0,
        "potassium": 10.0,
        "ph": 5.0 # Should trigger lime conditioner
    }
    
    rec = recommender.calculate_recommendations(predictions, "rice", bulk_density=1.3)
    
    # Verify conditioners
    conds = rec["conditioners"]
    assert len(conds) == 1
    assert "Lime" in conds[0]["name"]
    
    # Verify fertilizer calculations
    ferts = rec["fertilizers"]
    assert ferts["dap_total_kg_ha"] > 0
    assert ferts["mop_total_kg_ha"] > 0
    assert ferts["urea_total_kg_ha"] >= 0
    
    # Verify split-schedule timeline contains DAP basal and Urea top dressing splits
    stages = [item["stage"] for item in rec["schedule"]]
    assert "Basal (Transplanting)" in stages
    assert "Active Tillering" in stages

def test_advanced_features():
    """Tests Cost Optimization, Health Scores, and Local Attributions."""
    config = Config()
    recommender = FertilizerRecommender(config)
    
    # Low nutrient values to trigger cost and risk outputs
    predictions = {
        "nitrogen": 15.0,
        "phosphorus": 5.0,
        "potassium": 20.0,
        "ph": 5.2,
        "soc": 0.5
    }
    
    rec = recommender.calculate_recommendations(predictions, "rice", bulk_density=1.3)
    
    # 1. Test Cost Optimization
    costs = rec["costs"]
    assert costs["total_cost"] > 0
    assert costs["alternative_eco_cost"] > 0
    assert costs["savings"] > 0
    assert costs["total_cost"] == pytest.approx(costs["alternative_eco_cost"] + costs["savings"], abs=2.0)
    
    # 2. Test Health Score and Risks
    assert 10 <= rec["health_score"] <= 100
    assert len(rec["risks"]) > 0
    assert any("Nitrogen" in r for r in rec["risks"])
    assert any("pH" in r for r in rec["risks"])
    
    # 3. Test AI narrative
    assert len(rec["narrative"]) > 50
    assert "Soil diagnosis" in rec["narrative"]


def test_crops_database_extension():
    """Verifies that Chilli, Groundnut, and Tomato are present with water and yield metadata."""
    config = Config()
    crops = config.agronomy["crops"]
    
    assert "chilli" in crops
    assert "groundnut" in crops
    assert "tomato" in crops
    
    for c_key in ["rice", "chilli", "groundnut", "tomato"]:
        assert "water_requirement_mm" in crops[c_key]
        assert "yield_range_tons_ha" in crops[c_key]
        assert crops[c_key]["water_requirement_mm"] > 0
        assert len(crops[c_key]["yield_range_tons_ha"]) > 0


def test_geocoding_utilities():
    """Verifies the geocoder and reverse geocoder fallbacks."""
    from src.utils import reverse_geocode
    
    # Test focus state AP coordinates fallback
    res_ap = reverse_geocode(16.3008, 80.4428)
    assert res_ap["state"] in ["Andhra Pradesh", "Unknown State"]
    assert res_ap["village"] in ["Vemulapadu", "Guntur", "Guntur Mandal", "Unknown Village"]
    
    # Test focus state Karnataka coordinates fallback
    res_ka = reverse_geocode(13.0, 77.5)
    assert res_ka["state"] in ["Karnataka", "Unknown State"]
    assert res_ka["district"] in [
        "Bengaluru Rural", "Bengaluru", "Bengaluru Urban", 
        "Bengaluru West City Corporation", "Bengaluru West", "Unknown District"
    ]


def test_soil_lab_report_parser():
    """Verifies that soil report CSV parses N, P, K, pH, SOC correctly."""
    from src.utils import parse_soil_lab_report
    
    csv_content = (
        "nitrogen,phosphorus,potassium,ph,soc\n"
        "42.5,18.2,145.0,6.5,1.2\n"
    )
    
    res = parse_soil_lab_report(csv_content.encode("utf-8"), "test_soil.csv")
    
    assert res["nitrogen"] == 42.5
    assert res["phosphorus"] == 18.2
    assert res["potassium"] == 145.0
    assert res["ph"] == 6.5
    assert res["soc"] == 1.2


