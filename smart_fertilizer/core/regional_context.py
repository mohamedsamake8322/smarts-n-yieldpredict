from typing import Dict, List, Optional
import json
from datetime import datetime, timedelta

class RegionalContext:
    """
    Regional context management for African agricultural regions
    """
    
    def __init__(self):
        self.regions_data = self._load_regional_data()
        self.crop_calendars = self._load_crop_calendars()
        self.market_prices = self._load_market_prices()
        self.climate_data = self._load_climate_data()
        
    def _load_regional_data(self) -> Dict:
        """Load regional agricultural data for African countries"""
        return {
            "nigeria": {
                "name": "Nigeria",
                "zones": ["guinea_savanna", "sudan_savanna", "sahel", "forest"],
                "dominant_soils": ["ferrasols", "luvisols", "acrisols"],
                "climate_type": "tropical",
                "rainfall_pattern": "bimodal",
                "major_crops": ["maize", "rice", "yam", "cassava", "sorghum"],
                "fertilizer_subsidies": True,
                "currency": "NGN",
                "price_factor": 1.2,
                "crop_price_per_ton": 250,
                "languages": ["english", "hausa", "yoruba", "igbo"]
            },
            "kenya": {
                "name": "Kenya",
                "zones": ["highlands", "coast", "arid_semi_arid"],
                "dominant_soils": ["andosols", "vertisols", "ferralsols"],
                "climate_type": "tropical_highland",
                "rainfall_pattern": "bimodal",
                "major_crops": ["maize", "wheat", "rice", "tea", "coffee"],
                "fertilizer_subsidies": True,
                "currency": "KES",
                "price_factor": 1.1,
                "crop_price_per_ton": 280,
                "languages": ["english", "swahili"]
            },
            "ghana": {
                "name": "Ghana",
                "zones": ["forest", "transitional", "savanna"],
                "dominant_soils": ["acrisols", "lixisols", "gleysols"],
                "climate_type": "tropical",
                "rainfall_pattern": "bimodal",
                "major_crops": ["maize", "rice", "cocoa", "yam", "cassava"],
                "fertilizer_subsidies": True,
                "currency": "GHS",
                "price_factor": 1.15,
                "crop_price_per_ton": 220,
                "languages": ["english", "twi", "ga"]
            },
            "south_africa": {
                "name": "South Africa",
                "zones": ["highveld", "lowveld", "coastal", "karoo"],
                "dominant_soils": ["vertisols", "cambisols", "luvisols"],
                "climate_type": "semi_arid",
                "rainfall_pattern": "summer",
                "major_crops": ["maize", "wheat", "sorghum", "sugarcane"],
                "fertilizer_subsidies": False,
                "currency": "ZAR",
                "price_factor": 0.9,
                "crop_price_per_ton": 180,
                "languages": ["english", "afrikaans", "zulu", "xhosa"]
            },
            "ethiopia": {
                "name": "Ethiopia",
                "zones": ["highlands", "rift_valley", "lowlands"],
                "dominant_soils": ["vertisols", "cambisols", "luvisols"],
                "climate_type": "tropical_highland",
                "rainfall_pattern": "unimodal",
                "major_crops": ["teff", "maize", "wheat", "barley", "sorghum"],
                "fertilizer_subsidies": True,
                "currency": "ETB",
                "price_factor": 1.3,
                "crop_price_per_ton": 200,
                "languages": ["amharic", "oromo", "tigrinya"]
            },
            "mali": {
                "name": "Mali",
                "zones": ["sahel", "sudan_savanna", "niger_delta"],
                "dominant_soils": ["arenosols", "lixisols", "vertisols"],
                "climate_type": "arid_semi_arid",
                "rainfall_pattern": "unimodal",
                "major_crops": ["millet", "sorghum", "maize", "rice", "cotton"],
                "fertilizer_subsidies": True,
                "currency": "XOF",
                "price_factor": 1.4,
                "crop_price_per_ton": 190,
                "languages": ["french", "bambara", "fulfulde"]
            }
        }
    
    def _load_crop_calendars(self) -> Dict:
        """Load crop calendars for different regions"""
        return {
            "nigeria": {
                "maize": {
                    "first_season": {"planting": "April", "harvest": "August"},
                    "second_season": {"planting": "September", "harvest": "December"}
                },
                "rice": {
                    "wet_season": {"planting": "May", "harvest": "September"},
                    "dry_season": {"planting": "November", "harvest": "March"}
                },
                "yam": {
                    "main_season": {"planting": "March", "harvest": "December"}
                }
            },
            "kenya": {
                "maize": {
                    "long_rains": {"planting": "March", "harvest": "August"},
                    "short_rains": {"planting": "October", "harvest": "February"}
                },
                "wheat": {
                    "main_season": {"planting": "June", "harvest": "October"}
                }
            },
            "ghana": {
                "maize": {
                    "major_season": {"planting": "April", "harvest": "August"},
                    "minor_season": {"planting": "September", "harvest": "December"}
                },
                "rice": {
                    "rainfed": {"planting": "May", "harvest": "September"},
                    "irrigated": {"planting": "November", "harvest": "March"}
                }
            }
        }
    
    def _load_market_prices(self) -> Dict:
        """Load current market prices for fertilizers and crops"""
        return {
            "fertilizer_prices_usd": {
                "urea": {"base_price": 450, "last_updated": "2024-01-15"},
                "dap": {"base_price": 550, "last_updated": "2024-01-15"},
                "npk": {"base_price": 600, "last_updated": "2024-01-15"},
                "kcl": {"base_price": 400, "last_updated": "2024-01-15"},
                "tsp": {"base_price": 500, "last_updated": "2024-01-15"}
            },
            "crop_prices_usd": {
                "maize": {"price_per_ton": 200, "quality": "grade_1"},
                "rice": {"price_per_ton": 400, "quality": "milled"},
                "wheat": {"price_per_ton": 250, "quality": "bread_wheat"},
                "sorghum": {"price_per_ton": 180, "quality": "grain"},
                "millet": {"price_per_ton": 220, "quality": "grain"}
            },
            "regional_multipliers": {
                "transport_cost": {"landlocked": 1.2, "coastal": 1.0},
                "import_duties": {"low": 1.05, "medium": 1.15, "high": 1.25},
                "dealer_margins": {"rural": 1.15, "urban": 1.10}
            }
        }
    
    def _load_climate_data(self) -> Dict:
        """Load climate data for regions"""
        return {
            "west_africa": {
                "temperature_range": {"min": 20, "max": 35},
                "rainfall_range": {"min": 600, "max": 2000},
                "humidity_range": {"min": 60, "max": 90},
                "evapotranspiration": {"average": 1200, "peak": 1800},
                "risk_factors": ["drought", "flooding", "heat_stress"]
            },
            "east_africa": {
                "temperature_range": {"min": 15, "max": 30},
                "rainfall_range": {"min": 400, "max": 1800},
                "humidity_range": {"min": 50, "max": 80},
                "evapotranspiration": {"average": 1000, "peak": 1500},
                "risk_factors": ["drought", "irregular_rainfall", "altitude_stress"]
            },
            "southern_africa": {
                "temperature_range": {"min": 10, "max": 32},
                "rainfall_range": {"min": 200, "max": 1200},
                "humidity_range": {"min": 40, "max": 70},
                "evapotranspiration": {"average": 800, "peak": 1400},
                "risk_factors": ["drought", "frost", "water_scarcity"]
            }
        }
    
    def get_region_data(self, region_name: str) -> Dict:
        """Get comprehensive regional data"""
        region_key = region_name.lower().replace(" ", "_")
        base_data = self.regions_data.get(region_key, {})
        
        if not base_data:
            # Return default data if region not found
            return self._get_default_region_data()
        
        # Enhance with current market data
        enhanced_data = base_data.copy()
        enhanced_data.update({
            "current_fertilizer_prices": self._get_regional_fertilizer_prices(region_key),
            "current_crop_prices": self._get_regional_crop_prices(region_key),
            "seasonal_calendar": self.crop_calendars.get(region_key, {}),
            "climate_info": self._get_regional_climate(region_key)
        })
        
        return enhanced_data
    
    def _get_default_region_data(self) -> Dict:
        """Return default regional data for unknown regions"""
        return {
            "name": "Unknown Region",
            "zones": ["mixed"],
            "dominant_soils": ["mixed"],
            "climate_type": "tropical",
            "rainfall_pattern": "variable",
            "major_crops": ["maize", "rice"],
            "fertilizer_subsidies": False,
            "currency": "USD",
            "price_factor": 1.0,
            "crop_price_per_ton": 200,
            "languages": ["english"]
        }
    
    def _get_regional_fertilizer_prices(self, region_key: str) -> Dict:
        """Get fertilizer prices adjusted for region"""
        base_prices = self.market_prices["fertilizer_prices_usd"]
        region_data = self.regions_data.get(region_key, {})
        price_factor = region_data.get("price_factor", 1.0)
        
        regional_prices = {}
        for fertilizer, data in base_prices.items():
            regional_prices[fertilizer] = {
                "price_per_kg": (data["base_price"] / 1000) * price_factor,
                "currency": region_data.get("currency", "USD"),
                "last_updated": data["last_updated"]
            }
        
        return regional_prices
    
    def _get_regional_crop_prices(self, region_key: str) -> Dict:
        """Get crop prices adjusted for region"""
        base_prices = self.market_prices["crop_prices_usd"]
        region_data = self.regions_data.get(region_key, {})
        price_factor = region_data.get("price_factor", 1.0)
        
        regional_prices = {}
        for crop, data in base_prices.items():
            regional_prices[crop] = {
                "price_per_ton": data["price_per_ton"] * price_factor,
                "currency": region_data.get("currency", "USD"),
                "quality": data["quality"]
            }
        
        return regional_prices
    
    def _get_regional_climate(self, region_key: str) -> Dict:
        """Get climate information for region"""
        region_data = self.regions_data.get(region_key, {})
        climate_type = region_data.get("climate_type", "tropical")
        
        # Map climate types to climate data
        climate_mapping = {
            "tropical": "west_africa",
            "tropical_highland": "east_africa",
            "semi_arid": "southern_africa",
            "arid_semi_arid": "southern_africa"
        }
        
        climate_key = climate_mapping.get(climate_type, "west_africa")
        return self.climate_data.get(climate_key, {})
    
    def get_available_regions(self) -> List[Dict]:
        """Get list of available regions"""
        regions = []
        for key, data in self.regions_data.items():
            regions.append({
                "key": key,
                "name": data["name"],
                "major_crops": data["major_crops"],
                "climate_type": data["climate_type"]
            })
        return regions
    
    def get_seasonal_recommendations(self, region: str, crop: str, current_date: datetime) -> Dict:
        """Get seasonal recommendations based on regional crop calendar"""
        region_key = region.lower().replace(" ", "_")
        crop_calendar = self.crop_calendars.get(region_key, {}).get(crop.lower(), {})
        
        if not crop_calendar:
            return {"message": "No seasonal data available for this crop in this region"}
        
        current_month = current_date.strftime("%B")
        recommendations = {
            "current_month": current_month,
            "seasonal_advice": [],
            "optimal_timing": {}
        }
        
        # Determine current season and provide advice
        for season, timing in crop_calendar.items():
            planting_month = timing.get("planting", "")
            harvest_month = timing.get("harvest", "")
            
            if current_month == planting_month:
                recommendations["seasonal_advice"].append(f"Optimal planting time for {season}")
                recommendations["optimal_timing"]["fertilizer_application"] = "Apply basal fertilizer at planting"
            elif current_month == harvest_month:
                recommendations["seasonal_advice"].append(f"Harvest time for {season}")
                recommendations["optimal_timing"]["fertilizer_application"] = "Prepare for next season planting"
        
        return recommendations
    
    def calculate_logistics_cost(self, region: str, fertilizer_amount_kg: float) -> Dict:
        """Calculate logistics and transportation costs"""
        region_data = self.regions_data.get(region.lower(), {})
        
        # Base logistics parameters
        base_transport_cost_per_km_per_kg = 0.01  # USD
        average_distance_to_farm = 50  # km
        dealer_margin = 0.15  # 15%
        storage_cost_per_kg = 0.02  # USD
        
        # Regional adjustments
        if region_data.get("zones") and "landlocked" in str(region_data.get("zones")):
            transport_multiplier = 1.3
        else:
            transport_multiplier = 1.0
        
        # Calculate costs
        transport_cost = fertilizer_amount_kg * base_transport_cost_per_km_per_kg * average_distance_to_farm * transport_multiplier
        dealer_cost = fertilizer_amount_kg * dealer_margin * 0.5  # Assume $0.5 base cost per kg
        storage_cost = fertilizer_amount_kg * storage_cost_per_kg
        
        total_logistics_cost = transport_cost + dealer_cost + storage_cost
        
        return {
            "transport_cost": round(transport_cost, 2),
            "dealer_margin": round(dealer_cost, 2),
            "storage_cost": round(storage_cost, 2),
            "total_logistics_cost": round(total_logistics_cost, 2),
            "cost_per_kg": round(total_logistics_cost / fertilizer_amount_kg, 3) if fertilizer_amount_kg > 0 else 0
        }
    
    def get_extension_recommendations(self, region: str, crop: str) -> List[str]:
        """Get extension service recommendations for the region"""
        region_data = self.regions_data.get(region.lower(), {})
        
        recommendations = []
        
        # General recommendations
        recommendations.append("Contact local agricultural extension officers for field-specific advice")
        recommendations.append("Join farmer cooperatives for group purchasing of fertilizers")
        
        # Regional specific recommendations
        if region_data.get("fertilizer_subsidies"):
            recommendations.append("Check for government fertilizer subsidy programs in your area")
        
        climate_type = region_data.get("climate_type", "")
        if "arid" in climate_type:
            recommendations.append("Consider drought-resistant varieties and water-efficient practices")
        elif "tropical" in climate_type:
            recommendations.append("Plan fertilizer applications around rainfall patterns")
        
        # Crop specific recommendations
        major_crops = region_data.get("major_crops", [])
        if crop.lower() in major_crops:
            recommendations.append(f"{crop.title()} is well-suited for your region - follow local best practices")
        else:
            recommendations.append(f"Consider if {crop} is optimal for your region - consult local experts")
        
        return recommendations
    
    def get_market_access_info(self, region: str) -> Dict:
        """Get market access information for the region"""
        region_data = self.regions_data.get(region.lower(), {})
        
        return {
            "currency": region_data.get("currency", "USD"),
            "market_access": {
                "fertilizer_dealers": "Available in major towns and agricultural centers",
                "crop_markets": "Local markets and commodity exchanges available",
                "transportation": "Road network connects major agricultural areas"
            },
            "financial_services": {
                "agricultural_loans": "Available through banks and microfinance institutions",
                "crop_insurance": "Limited availability - check with local providers",
                "mobile_money": "Widely available for transactions"
            },
            "support_services": {
                "extension_services": "Government and NGO extension services available",
                "input_suppliers": "Authorized dealers in major agricultural areas",
                "technical_support": "Contact local agricultural research institutes"
            }
        }
