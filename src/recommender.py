import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

from src.config import Config
from src.logger import setup_logger

logger = setup_logger("recommender", level="INFO")

DYNAMIC_CROPS = {
    "chickpeas": {
        "name": "Chickpeas",
        "target_n": 30.0,
        "target_p": 50.0,
        "target_k": 50.0,
        "optimal_ph_min": 6.0,
        "optimal_ph_max": 7.5,
        "optimal_soc_min": 0.8,
        "optimal_temp_min": 15.0,
        "optimal_temp_max": 28.0,
        "water_requirement_mm": 400.0,
        "yield_range_tons_ha": "1.5 - 2.5",
        "split_schedule": {"basal": 0.8, "vegetative": 0.2},
        "growth_stages": ["Basal (Sowing)", "Vegetative Stage"]
    },
    "sugarcane": {
        "name": "Sugarcane",
        "target_n": 250.0,
        "target_p": 80.0,
        "target_k": 120.0,
        "optimal_ph_min": 6.0,
        "optimal_ph_max": 7.5,
        "optimal_soc_min": 1.2,
        "optimal_temp_min": 20.0,
        "optimal_temp_max": 38.0,
        "water_requirement_mm": 1500.0,
        "yield_range_tons_ha": "60.0 - 100.0",
        "split_schedule": {"basal": 0.3, "tillering": 0.4, "grand_growth": 0.3},
        "growth_stages": ["Basal (Planting)", "Tillering Stage", "Grand Growth"]
    },
    "potato": {
        "name": "Potato",
        "target_n": 120.0,
        "target_p": 100.0,
        "target_k": 150.0,
        "optimal_ph_min": 5.5,
        "optimal_ph_max": 6.5,
        "optimal_soc_min": 1.2,
        "optimal_temp_min": 15.0,
        "optimal_temp_max": 25.0,
        "water_requirement_mm": 600.0,
        "yield_range_tons_ha": "15.0 - 25.0",
        "split_schedule": {"basal": 0.5, "tuber_initiation": 0.5},
        "growth_stages": ["Basal (Planting)", "Tuber Initiation"]
    },
    "soybean": {
        "name": "Soybean",
        "target_n": 40.0,
        "target_p": 60.0,
        "target_k": 60.0,
        "optimal_ph_min": 6.0,
        "optimal_ph_max": 7.0,
        "optimal_soc_min": 1.0,
        "optimal_temp_min": 20.0,
        "optimal_temp_max": 32.0,
        "water_requirement_mm": 600.0,
        "yield_range_tons_ha": "2.0 - 3.5",
        "split_schedule": {"basal": 0.8, "flowering": 0.2},
        "growth_stages": ["Basal (Sowing)", "Flowering Stage"]
    },
    "mustard": {
        "name": "Mustard",
        "target_n": 90.0,
        "target_p": 40.0,
        "target_k": 40.0,
        "optimal_ph_min": 6.0,
        "optimal_ph_max": 7.5,
        "optimal_soc_min": 0.8,
        "optimal_temp_min": 10.0,
        "optimal_temp_max": 25.0,
        "water_requirement_mm": 400.0,
        "yield_range_tons_ha": "1.5 - 2.5",
        "split_schedule": {"basal": 0.6, "flowering": 0.4},
        "growth_stages": ["Basal (Sowing)", "Rosette / Flowering"]
    },
    "banana": {
        "name": "Banana",
        "target_n": 200.0,
        "target_p": 60.0,
        "target_k": 300.0,
        "optimal_ph_min": 5.5,
        "optimal_ph_max": 7.0,
        "optimal_soc_min": 1.2,
        "optimal_temp_min": 20.0,
        "optimal_temp_max": 35.0,
        "water_requirement_mm": 1800.0,
        "yield_range_tons_ha": "30.0 - 50.0",
        "split_schedule": {"basal": 0.3, "shooting": 0.4, "fruiting": 0.3},
        "growth_stages": ["Basal (Planting)", "Shooting Stage", "Fruiting Stage"]
    },
    "strawberry": {
        "name": "Strawberry",
        "target_n": 80.0,
        "target_p": 40.0,
        "target_k": 90.0,
        "optimal_ph_min": 5.5,
        "optimal_ph_max": 6.5,
        "optimal_soc_min": 1.2,
        "optimal_temp_min": 10.0,
        "optimal_temp_max": 25.0,
        "water_requirement_mm": 600.0,
        "yield_range_tons_ha": "15.0 - 25.0",
        "split_schedule": {"basal": 0.4, "flowering": 0.3, "fruiting": 0.3},
        "growth_stages": ["Basal (Planting)", "Flowering Stage", "Fruiting Stage"]
    }
}

def get_crop_rules_by_name(name: str) -> Dict[str, Any]:
    key = name.strip().lower()
    if key in DYNAMIC_CROPS:
        return DYNAMIC_CROPS[key]
        
    # Heuristics lookup / fallback
    if any(w in key for w in ["strawberry", "berry"]):
        return {
            "name": name.title(),
            "target_n": 80.0,
            "target_p": 40.0,
            "target_k": 90.0,
            "optimal_ph_min": 5.5,
            "optimal_ph_max": 6.5,
            "optimal_soc_min": 1.2,
            "optimal_temp_min": 10.0,
            "optimal_temp_max": 25.0,
            "water_requirement_mm": 600.0,
            "yield_range_tons_ha": "15.0 - 25.0",
            "split_schedule": {"basal": 0.4, "flowering": 0.3, "fruiting": 0.3},
            "growth_stages": ["Basal (Planting)", "Flowering Stage", "Fruiting Stage"]
        }
    elif any(w in key for w in ["bean", "gram", "lentil", "chickpea", "pea", "cowpea", "groundnut", "legume"]):
        return {
            "name": name.title(),
            "target_n": 40.0,
            "target_p": 50.0,
            "target_k": 50.0,
            "optimal_ph_min": 6.0,
            "optimal_ph_max": 7.5,
            "optimal_soc_min": 0.8,
            "optimal_temp_min": 15.0,
            "optimal_temp_max": 30.0,
            "water_requirement_mm": 500.0,
            "yield_range_tons_ha": "2.0 - 3.5",
            "split_schedule": {"basal": 0.7, "flowering": 0.3},
            "growth_stages": ["Basal (Sowing)", "Flowering / Podding"]
        }
    elif any(w in key for w in ["cane", "sugarcane", "banana", "corn", "maize"]):
        return {
            "name": name.title(),
            "target_n": 200.0,
            "target_p": 80.0,
            "target_k": 120.0,
            "optimal_ph_min": 6.0,
            "optimal_ph_max": 7.5,
            "optimal_soc_min": 1.2,
            "optimal_temp_min": 20.0,
            "optimal_temp_max": 38.0,
            "water_requirement_mm": 1200.0,
            "yield_range_tons_ha": "10.0 - 25.0" if "banana" in key else "50.0 - 90.0" if "cane" in key else "5.0 - 8.0",
            "split_schedule": {"basal": 0.4, "vegetative": 0.3, "fruiting": 0.3},
            "growth_stages": ["Basal (Planting)", "Active Vegetative", "Maturity Stage"]
        }
    elif any(w in key for w in ["rice", "wheat", "barley", "oats", "rye", "grain", "cereal", "sorghum", "millet", "ragi", "bajra"]):
        return {
            "name": name.title(),
            "target_n": 130.0,
            "target_p": 60.0,
            "target_k": 50.0,
            "optimal_ph_min": 5.8,
            "optimal_ph_max": 7.0,
            "optimal_soc_min": 1.0,
            "optimal_temp_min": 15.0,
            "optimal_temp_max": 35.0,
            "water_requirement_mm": 600.0,
            "yield_range_tons_ha": "3.0 - 6.0",
            "split_schedule": {"basal": 0.5, "tillering": 0.5},
            "growth_stages": ["Basal (Sowing)", "Active Growth Stage"]
        }
    elif any(w in key for w in ["potato", "onion", "garlic", "carrot", "beet", "radish", "ginger", "turmeric", "yam", "tuber", "root"]):
        return {
            "name": name.title(),
            "target_n": 110.0,
            "target_p": 70.0,
            "target_k": 140.0,
            "optimal_ph_min": 5.5,
            "optimal_ph_max": 6.8,
            "optimal_soc_min": 1.2,
            "optimal_temp_min": 12.0,
            "optimal_temp_max": 28.0,
            "water_requirement_mm": 600.0,
            "yield_range_tons_ha": "12.0 - 22.0",
            "split_schedule": {"basal": 0.5, "tuber_development": 0.5},
            "growth_stages": ["Basal (Planting)", "Bulb/Tuber Growth"]
        }
    elif any(w in key for w in ["chilli", "tomato", "pepper", "brinjal", "eggplant", "okra", "melon", "gourd"]):
        return {
            "name": name.title(),
            "target_n": 120.0,
            "target_p": 60.0,
            "target_k": 100.0,
            "optimal_ph_min": 6.0,
            "optimal_ph_max": 6.8,
            "optimal_soc_min": 1.0,
            "optimal_temp_min": 18.0,
            "optimal_temp_max": 32.0,
            "water_requirement_mm": 650.0,
            "yield_range_tons_ha": "5.0 - 15.0",
            "split_schedule": {"basal": 0.5, "flowering": 0.5},
            "growth_stages": ["Basal (Sowing)", "Flowering / Fruiting"]
        }
    else:
        return {
            "name": name.title(),
            "target_n": 100.0,
            "target_p": 60.0,
            "target_k": 60.0,
            "optimal_ph_min": 6.0,
            "optimal_ph_max": 7.0,
            "optimal_soc_min": 1.0,
            "optimal_temp_min": 15.0,
            "optimal_temp_max": 35.0,
            "water_requirement_mm": 600.0,
            "yield_range_tons_ha": "3.0 - 5.0",
            "split_schedule": {"basal": 0.6, "active_growth": 0.4},
            "growth_stages": ["Basal (Sowing)", "Active Growth Stage"]
        }

def register_custom_crop(config: Config, crop_name: str) -> str:
    key = crop_name.strip().lower()
    if not key:
        return ""
    if "crops" not in config.agronomy:
        config._config_dict["agronomy"]["crops"] = {}
    if key not in config.agronomy["crops"]:
        rules = get_crop_rules_by_name(crop_name)
        config.agronomy["crops"][key] = rules
    return key

class CropSuitabilityEngine:
    """Calculates crop suitability ratings based on predicted chemical and structural soil state."""
    
    def __init__(self, config: Config):
        self.config = config
        self.agronomy_cfg = config.agronomy["crops"]
        # Default temperature bounds for config-based crops
        self.default_temp_bounds = {
            "rice": (20.0, 35.0),
            "wheat": (10.0, 25.0),
            "maize": (18.0, 32.0),
            "cotton": (20.0, 35.0),
            "chilli": (18.0, 35.0),
            "groundnut": (20.0, 32.0),
            "tomato": (18.0, 28.0)
        }
        
    def evaluate_suitability(
        self,
        predictions: Dict[str, float],
        bulk_density: float = 1.3,
        temperature: float = None
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluates the agricultural suitability index for multiple crops.
        
        Args:
            predictions: Dict containing predicted values for 'nitrogen', 'phosphorus', 'potassium', 'soc', 'ph'.
            bulk_density: Bulk density of the soil (g/cm^3) for nutrient calculations.
            temperature: Current or seasonal temperature (°C) of the query coordinate center.
            
        Returns:
            Dict mapping crop key (e.g., 'rice') to its score and rating.
        """
        results = {}
        
        ph = predictions.get("ph", 6.0)
        soc = predictions.get("soc", 1.0)
        n = predictions.get("nitrogen", 50.0)
        p = predictions.get("phosphorus", 25.0)
        k = predictions.get("potassium", 150.0)
        
        # Convert test values in mg/kg to plant-available stock in kg/ha
        # Stock (kg/ha) = Test (mg/kg) * Bulk Density (g/cm^3) * 15cm (depth) * 0.1 * 10
        available_n = n * bulk_density * 1.5
        available_p = p * bulk_density * 1.5
        available_k = k * bulk_density * 1.5
        
        for crop_key, crop_rules in self.agronomy_cfg.items():
            bounds = self.default_temp_bounds.get(crop_key, (15.0, 35.0))
            opt_temp_min = crop_rules.get("optimal_temp_min", bounds[0])
            opt_temp_max = crop_rules.get("optimal_temp_max", bounds[1])
            
            reasons = []
            
            # 1. pH Suitability Score (Weight: 35%)
            opt_ph_min = crop_rules["optimal_ph_min"]
            opt_ph_max = crop_rules["optimal_ph_max"]
            
            if opt_ph_min <= ph <= opt_ph_max:
                ph_score = 100.0
            else:
                # Deduct score based on distance from optimal range
                dist = min(abs(ph - opt_ph_min), abs(ph - opt_ph_max))
                ph_score = max(0.0, 100.0 - (dist * 40.0)) # Severe penalty for extreme pH
                reasons.append(f"Soil pH ({ph:.2f}) is outside optimal growth range ({opt_ph_min}-{opt_ph_max})")
                
            # 2. SOC Suitability Score (Weight: 15%)
            opt_soc_min = crop_rules["optimal_soc_min"]
            if soc >= opt_soc_min:
                soc_score = 100.0
            else:
                soc_score = (soc / opt_soc_min) * 100.0
                reasons.append(f"Soil Organic Carbon ({soc:.2f}%) is below target of {opt_soc_min}%")
                
            # 3. Nutrient (N-P-K) Availability Score (Weight: 50%)
            # Compare available soil nutrients to crop target requirements
            target_n = crop_rules["target_n"]
            target_p = crop_rules["target_p"]
            target_k = crop_rules["target_k"]
            
            n_ratio = min(1.0, available_n / target_n)
            p_ratio = min(1.0, available_p / target_p)
            k_ratio = min(1.0, available_k / target_k)
            
            if available_n < target_n:
                reasons.append(f"Available Nitrogen ({available_n:.1f} kg/ha) is below requirement of {target_n} kg/ha")
            if available_p < target_p:
                reasons.append(f"Available Phosphorus ({available_p:.1f} kg/ha) is below requirement of {target_p} kg/ha")
            if available_k < target_k:
                reasons.append(f"Available Potassium ({available_k:.1f} kg/ha) is below requirement of {target_k} kg/ha")
                
            nutrient_score = (n_ratio * 0.4 + p_ratio * 0.3 + k_ratio * 0.3) * 100.0
            
            # Weighted overall index
            overall_score = (ph_score * 0.35) + (soc_score * 0.15) + (nutrient_score * 0.50)
            
            # Climate Temperature Penalty
            temp_deduction = 0.0
            if temperature is not None:
                if temperature < opt_temp_min or temperature > opt_temp_max:
                    deviation = min(abs(temperature - opt_temp_min), abs(temperature - opt_temp_max))
                    temp_deduction = deviation * 15.0 # 15% reduction per degree of deviation
                    reasons.append(f"Climate Temperature ({temperature:.1f}°C) is outside optimal growth range ({opt_temp_min}-{opt_temp_max}°C)")
            
            overall_score = max(0.0, overall_score - temp_deduction)
            overall_score = round(float(overall_score), 1)
            
            # Categorize suitability
            if overall_score >= 85:
                rating = "Highly Suitable"
                color = "green"
                risk_level = "Low"
            elif overall_score >= 70:
                rating = "Suitable"
                color = "blue"
                risk_level = "Low"
            elif overall_score >= 50:
                rating = "Marginally Suitable"
                color = "orange"
                risk_level = "Moderate"
            else:
                rating = "Unsuitable"
                color = "red"
                risk_level = "High"
                
            yield_str = crop_rules.get("yield_range_tons_ha", "0.0 - 0.0")
            try:
                parts = [float(x.strip()) for x in yield_str.split("-")]
                min_y, max_y = parts[0], parts[1]
                scaled_yield = min_y + (max_y - min_y) * (overall_score / 100.0)
                expected_yield = f"{scaled_yield:.2f} tons/ha"
            except Exception:
                expected_yield = yield_str
                
            results[crop_key] = {
                "crop_name": crop_rules["name"],
                "suitability_score": overall_score,
                "rating": rating,
                "color": color,
                "risk_level": risk_level,
                "expected_yield": expected_yield,
                "water_requirement_mm": crop_rules.get("water_requirement_mm", 0.0),
                "reasons": reasons,
                "breakdown": {
                    "ph_score": round(ph_score, 1),
                    "soc_score": round(soc_score, 1),
                    "nutrient_score": round(nutrient_score, 1)
                }
            }
            
        return results

class FertilizerRecommender:
    """Calculates specific fertilizer application schedules and soil amendments."""
    
    def __init__(self, config: Config):
        self.config = config
        self.agronomy_cfg = config.agronomy["crops"]
        self.fert_cfg = config.fertilizers
        self.conditioners_cfg = config.soil_conditioners
        
    def calculate_recommendations(
        self,
        predictions: Dict[str, float],
        crop_type: str,
        bulk_density: float = 1.3
    ) -> Dict[str, Any]:
        """Generates detailed fertilizing schedule, pricing options, risks, health scores, and summaries.
        
        Args:
            predictions: Dict containing predicted values for 'nitrogen', 'phosphorus', 'potassium', 'ph', 'soc'.
            crop_type: Target crop key ('rice', 'wheat', 'maize', 'cotton').
            bulk_density: Bulk density of the soil.
            
        Returns:
            Dictionary containing fertilizer rates, cost break-downs, risk factors, health score, and narrative.
        """
        crop_rules = self.agronomy_cfg.get(crop_type)
        if not crop_rules:
            raise KeyError(f"Unsupported crop type: {crop_type}")
            
        ph = predictions.get("ph", 6.0)
        n = predictions.get("nitrogen", 50.0)
        p = predictions.get("phosphorus", 25.0)
        k = predictions.get("potassium", 150.0)
        soc = predictions.get("soc", 1.0)
        
        # 1. Convert test values to available stock (kg/ha) in topsoil
        available_n = n * bulk_density * 1.5
        available_p = p * bulk_density * 1.5
        available_k = k * bulk_density * 1.5
        
        # 2. Deficits calculation
        target_n = crop_rules["target_n"]
        target_p = crop_rules["target_p"]
        target_k = crop_rules["target_k"]
        
        n_deficit = max(0.0, target_n - available_n)
        p_deficit = max(0.0, target_p - available_p)
        k_deficit = max(0.0, target_k - available_k)
        
        # 3. Fertilizer Formulation
        # DAP (18-46-0) supplies P. DAP rate = P_deficit / 0.46
        dap_rate = p_deficit / self.fert_cfg["dap"]["p_content"]
        
        # N supplied by DAP = DAP rate * 18%
        n_from_dap = dap_rate * self.fert_cfg["dap"]["n_content"]
        
        # Remaining N deficit to be supplied by Urea (46% N)
        remaining_n = max(0.0, n_deficit - n_from_dap)
        urea_rate = remaining_n / self.fert_cfg["urea"]["n_content"]
        
        # MOP (60% K2O) supplies K. MOP rate = K_deficit / 0.60
        mop_rate = k_deficit / self.fert_cfg["mop"]["k_content"]
        
        # 4. Soil Conditioner Amendments
        conditioners = []
        lime_cfg = self.conditioners_cfg["lime"]
        gypsum_cfg = self.conditioners_cfg["gypsum"]
        
        if ph < lime_cfg["trigger_ph_below"]:
            conditioners.append({
                "name": lime_cfg["name"],
                "rate_kg_ha": lime_cfg["recommended_rate"],
                "reason": f"Predicted pH is {ph:.1f} (highly acidic). {lime_cfg['reason']}"
            })
        elif ph > gypsum_cfg["trigger_ph_above"]:
            conditioners.append({
                "name": gypsum_cfg["name"],
                "rate_kg_ha": gypsum_cfg["recommended_rate"],
                "reason": f"Predicted pH is {ph:.1f} (highly alkaline). {gypsum_cfg['reason']}"
            })
            
        # 5. Split-Application Schedule
        split_schedule = crop_rules["split_schedule"]
        growth_stages = crop_rules["growth_stages"]
        
        schedule = []
        
        # DAP is applied 100% basal due to phosphorus immobility in soil
        schedule.append({
            "stage": growth_stages[0],
            "fertilizer": self.fert_cfg["dap"]["name"],
            "rate_kg_ha": round(dap_rate, 1),
            "timing": "Apply at the time of sowing/transplanting."
        })
        
        # MOP is also applied 100% basal to support early root strength and health
        schedule.append({
            "stage": growth_stages[0],
            "fertilizer": self.fert_cfg["mop"]["name"],
            "rate_kg_ha": round(mop_rate, 1),
            "timing": "Apply at the time of sowing/transplanting."
        })
        
        # Urea is split based on growth stages to minimize leaching and volatilization
        stage_keys = list(split_schedule.keys())
        for idx, stage_name in enumerate(growth_stages):
            if idx < len(stage_keys):
                stage_key = stage_keys[idx]
                pct = split_schedule[stage_key]
                if pct > 0:
                    stage_urea = urea_rate * pct
                    schedule.append({
                        "stage": stage_name,
                        "fertilizer": self.fert_cfg["urea"]["name"],
                        "rate_kg_ha": round(stage_urea, 1),
                        "timing": f"Top dressing of Urea ({pct*100:.0f}%) at this crop stage."
                    })
                    
        # 6. Cost Optimization Calculations (Feature 3)
        prices = self.config.get("prices", {})
        urea_price = prices.get("urea_cost_per_kg", 6.0)
        dap_price = prices.get("dap_cost_per_kg", 27.0)
        mop_price = prices.get("mop_cost_per_kg", 34.0)
        eco_discount = prices.get("precision_eco_discount", 0.15)
        
        cost_urea = urea_rate * urea_price
        cost_dap = dap_rate * dap_price
        cost_mop = mop_rate * mop_price
        total_cost = cost_urea + cost_dap + cost_mop
        
        # Alternative Eco-Banding plan: reduces chemical usage by 15% via targeted placement
        eco_total_cost = total_cost * (1.0 - eco_discount)
        cost_savings = total_cost - eco_total_cost
        
        # 7. Parcel Health & Risk Scoring (Feature 4)
        n_def_ratio = max(0.0, min(1.0, n_deficit / (target_n if target_n > 0 else 1.0)))
        p_def_ratio = max(0.0, min(1.0, p_deficit / (target_p if target_p > 0 else 1.0)))
        k_def_ratio = max(0.0, min(1.0, k_deficit / (target_k if target_k > 0 else 1.0)))
        
        n_penalty = n_def_ratio * 40.0
        p_penalty = p_def_ratio * 30.0
        k_penalty = k_def_ratio * 30.0
        
        opt_ph_min = crop_rules["optimal_ph_min"]
        opt_ph_max = crop_rules["optimal_ph_max"]
        if ph < opt_ph_min:
            ph_penalty = min(20.0, (opt_ph_min - ph) * 20.0)
        elif ph > opt_ph_max:
            ph_penalty = min(20.0, (ph - opt_ph_max) * 20.0)
        else:
            ph_penalty = 0.0
            
        health_score = 100.0 - (n_penalty * 0.4 + p_penalty * 0.3 + k_penalty * 0.3 + ph_penalty)
        health_score = int(np.clip(health_score, 10, 100))
        
        risks = []
        if n_def_ratio > 0.4:
            risks.append("Low Available Nitrogen (N)")
        if p_def_ratio > 0.4:
            risks.append("Low Available Phosphorus (P)")
        if k_def_ratio > 0.4:
            risks.append("Low Available Potassium (K)")
        if ph < opt_ph_min or ph > opt_ph_max:
            risks.append("Soil pH Deviation / Drift")
        if soc < crop_rules["optimal_soc_min"]:
            risks.append("Deficient Soil Organic Carbon (SOC)")
            
        if not risks:
            risks.append("No major nutritional risks detected.")
            
        # 8. AI Agronomist Narrative Summary (Feature 8)
        n_status = "severe deficit" if n_def_ratio > 0.5 else "moderate deficiency" if n_def_ratio > 0.1 else "optimal levels"
        carbon_status = "healthy organic structures" if soc >= crop_rules["optimal_soc_min"] else "depleted carbon levels"
        ph_class = "highly acidic" if ph < 5.5 else "neutral/stable" if ph <= 7.2 else "alkaline"
        
        narrative = (
            f"Soil diagnosis reveals {n_status} for available Nitrogen. Soil Organic Carbon stands at {soc:.2f}%, indicating {carbon_status}. "
            f"The soil pH of {ph:.1f} is classified as {ph_class}, which affects mineral solubility. "
            f"For {crop_rules['name']}, the parcel shows a health index of {health_score}/100. "
            f"Recommended intervention: Apply {round(urea_rate, 1)} kg/ha of Urea split across growth stages, and {round(dap_rate, 1)} kg/ha of DAP basal to build root capacity."
        )
        
        # 9. Agronomic Reasoning Explanation
        reasoning = []
        reasoning.append(f"Predicted soil test results: N={n:.1f} mg/kg, P={p:.1f} mg/kg, K={k:.1f} mg/kg. Target nutrients for {crop_rules['name']}: N={target_n} kg/ha, P={target_p} kg/ha, K={target_k} kg/ha.")
        reasoning.append(f"Converting test results to available nutrient stock: available Nitrogen={available_n:.1f} kg/ha, available Phosphorus={available_p:.1f} kg/ha, available Potassium={available_k:.1f} kg/ha.")
        
        if p_deficit > 0:
            reasoning.append(f"Phosphorus deficit of {p_deficit:.1f} kg/ha identified. Diammonium Phosphate (DAP) is recommended basal. Since DAP contains 18% Nitrogen, it contributes {n_from_dap:.1f} kg/ha of Nitrogen to the crop.")
        else:
            reasoning.append("Phosphorus level is sufficient. No phosphate fertilizer is recommended.")
            
        if remaining_n > 0:
            reasoning.append(f"Net Nitrogen deficit of {remaining_n:.1f} kg/ha (accounting for DAP contribution) corrected with Urea. Urea application is split into multiple stages to prevent nitrogen leaching.")
        else:
            reasoning.append("Nitrogen requirements are fully met by soil stock or DAP contribution.")
            
        if k_deficit > 0:
            reasoning.append(f"Potassium deficit of {k_deficit:.1f} kg/ha is corrected using Muriate of Potash (MOP) applied basal.")
            
        return {
            "crop_name": crop_rules["name"],
            "fertilizers": {
                "urea_total_kg_ha": round(urea_rate, 1),
                "dap_total_kg_ha": round(dap_rate, 1),
                "mop_total_kg_ha": round(mop_rate, 1)
            },
            "costs": {
                "urea_cost": round(cost_urea, 0),
                "dap_cost": round(cost_dap, 0),
                "mop_cost": round(cost_mop, 0),
                "total_cost": round(total_cost, 0),
                "alternative_eco_cost": round(eco_total_cost, 0),
                "savings": round(cost_savings, 0)
            },
            "health_score": health_score,
            "risks": risks,
            "narrative": narrative,
            "conditioners": conditioners,
            "schedule": schedule,
            "reasoning": reasoning
        }

if __name__ == "__main__":
    config = Config()
    recommender = FertilizerRecommender(config)
    suitability = CropSuitabilityEngine(config)
    
    # Test values
    preds = {
        "nitrogen": 45.0,
        "phosphorus": 12.0,
        "potassium": 140.0,
        "soc": 0.9,
        "ph": 5.2
    }
    
    print("Suitabilities:")
    print(suitability.evaluate_suitability(preds))
    print("\nAdvisory for Rice:")
    print(recommender.calculate_recommendations(preds, "rice"))
