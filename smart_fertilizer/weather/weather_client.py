import requests
import os
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json

class WeatherClient:
    """
    Weather data client for agricultural applications
    """
    
    def __init__(self):
        self.openweather_api_key = os.getenv("OPENWEATHER_API_KEY", "demo_key")
        self.base_url = "https://api.openweathermap.org/data/2.5"
        self.onecall_url = "https://api.openweathermap.org/data/3.0/onecall"
        
    def get_current_weather(self, lat: float, lon: float) -> Optional[Dict]:
        """Get current weather data for location"""
        
        if self.openweather_api_key == "demo_key":
            return self._get_mock_current_weather()
        
        try:
            url = f"{self.base_url}/weather"
            params = {
                "lat": lat,
                "lon": lon,
                "appid": self.openweather_api_key,
                "units": "metric"
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            return self._process_current_weather(data)
            
        except Exception as e:
            print(f"Error fetching current weather: {e}")
            return self._get_mock_current_weather()
    
    def get_weather_forecast(self, lat: float, lon: float, days: int = 7) -> Optional[Dict]:
        """Get weather forecast for location"""
        
        if self.openweather_api_key == "demo_key":
            return self._get_mock_forecast(days)
        
        try:
            url = f"{self.onecall_url}"
            params = {
                "lat": lat,
                "lon": lon,
                "appid": self.openweather_api_key,
                "units": "metric",
                "exclude": "minutely,alerts"
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            return self._process_forecast_data(data, days)
            
        except Exception as e:
            print(f"Error fetching weather forecast: {e}")
            return self._get_mock_forecast(days)
    
    def get_agricultural_indices(self, lat: float, lon: float) -> Dict:
        """Get agricultural weather indices"""
        
        current_weather = self.get_current_weather(lat, lon)
        forecast = self.get_weather_forecast(lat, lon, 14)
        
        if not current_weather or not forecast:
            return self._get_mock_agricultural_indices()
        
        indices = {
            "growing_degree_days": self._calculate_gdd(forecast),
            "precipitation_forecast": self._analyze_precipitation(forecast),
            "evapotranspiration": self._estimate_et(current_weather, forecast),
            "soil_temperature_estimate": self._estimate_soil_temp(current_weather),
            "fertilizer_application_suitability": self._assess_application_conditions(forecast),
            "drought_risk": self._assess_drought_risk(forecast),
            "leaching_risk": self._assess_leaching_risk(forecast)
        }
        
        return indices
    
    def _process_current_weather(self, data: Dict) -> Dict:
        """Process current weather API response"""
        
        return {
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "pressure": data["main"]["pressure"],
            "wind_speed": data["wind"]["speed"],
            "wind_direction": data["wind"].get("deg", 0),
            "precipitation": data.get("rain", {}).get("1h", 0),
            "weather_condition": data["weather"][0]["main"],
            "description": data["weather"][0]["description"],
            "visibility": data.get("visibility", 10000),
            "uv_index": data.get("uvi", 0),
            "timestamp": datetime.fromtimestamp(data["dt"])
        }
    
    def _process_forecast_data(self, data: Dict, days: int) -> Dict:
        """Process forecast API response"""
        
        daily_forecasts = []
        hourly_forecasts = []
        
        # Process daily forecasts
        for day_data in data.get("daily", [])[:days]:
            daily_forecast = {
                "date": datetime.fromtimestamp(day_data["dt"]),
                "temp_min": day_data["temp"]["min"],
                "temp_max": day_data["temp"]["max"],
                "humidity": day_data["humidity"],
                "precipitation": day_data.get("rain", {}).get("1h", 0) + day_data.get("snow", {}).get("1h", 0),
                "wind_speed": day_data["wind_speed"],
                "weather_condition": day_data["weather"][0]["main"],
                "description": day_data["weather"][0]["description"],
                "pressure": day_data["pressure"],
                "uv_index": day_data.get("uvi", 0)
            }
            daily_forecasts.append(daily_forecast)
        
        # Process hourly forecasts (first 48 hours)
        for hour_data in data.get("hourly", [])[:48]:
            hourly_forecast = {
                "datetime": datetime.fromtimestamp(hour_data["dt"]),
                "temperature": hour_data["temp"],
                "humidity": hour_data["humidity"],
                "precipitation": hour_data.get("rain", {}).get("1h", 0),
                "wind_speed": hour_data["wind_speed"],
                "weather_condition": hour_data["weather"][0]["main"]
            }
            hourly_forecasts.append(hourly_forecast)
        
        return {
            "daily_forecast": daily_forecasts,
            "hourly_forecast": hourly_forecasts,
            "location": {
                "lat": data["lat"],
                "lon": data["lon"],
                "timezone": data["timezone"]
            }
        }
    
    def _calculate_gdd(self, forecast: Dict, base_temp: float = 10.0) -> Dict:
        """Calculate Growing Degree Days"""
        
        daily_gdd = []
        cumulative_gdd = 0
        
        for day in forecast.get("daily_forecast", []):
            avg_temp = (day["temp_min"] + day["temp_max"]) / 2
            gdd = max(0, avg_temp - base_temp)
            cumulative_gdd += gdd
            
            daily_gdd.append({
                "date": day["date"],
                "daily_gdd": round(gdd, 1),
                "cumulative_gdd": round(cumulative_gdd, 1)
            })
        
        return {
            "daily_values": daily_gdd,
            "total_accumulated": round(cumulative_gdd, 1),
            "base_temperature": base_temp
        }
    
    def _analyze_precipitation(self, forecast: Dict) -> Dict:
        """Analyze precipitation patterns"""
        
        daily_precip = []
        total_precip = 0
        rainy_days = 0
        
        for day in forecast.get("daily_forecast", []):
            precip = day["precipitation"]
            total_precip += precip
            if precip > 1.0:  # Significant precipitation threshold
                rainy_days += 1
            
            daily_precip.append({
                "date": day["date"],
                "precipitation_mm": round(precip, 1)
            })
        
        return {
            "daily_precipitation": daily_precip,
            "total_precipitation_mm": round(total_precip, 1),
            "rainy_days": rainy_days,
            "average_daily_precip": round(total_precip / len(daily_precip), 1) if daily_precip else 0
        }
    
    def _estimate_et(self, current: Dict, forecast: Dict) -> Dict:
        """Estimate evapotranspiration using simplified Penman-Monteith"""
        
        et_values = []
        
        for day in forecast.get("daily_forecast", []):
            # Simplified ET calculation
            temp_avg = (day["temp_min"] + day["temp_max"]) / 2
            rh_avg = day["humidity"]
            wind_speed = day["wind_speed"]
            
            # Simplified ET formula (mm/day)
            et_daily = (0.0023 * (temp_avg + 17.8) * 
                       ((day["temp_max"] - day["temp_min"]) ** 0.5) * 
                       (100 - rh_avg) / 100 * 
                       (1 + 0.1 * wind_speed))
            
            et_values.append({
                "date": day["date"],
                "et_mm": round(max(0, et_daily), 1)
            })
        
        total_et = sum(day["et_mm"] for day in et_values)
        
        return {
            "daily_et": et_values,
            "total_et_mm": round(total_et, 1),
            "average_daily_et": round(total_et / len(et_values), 1) if et_values else 0
        }
    
    def _estimate_soil_temp(self, current: Dict) -> Dict:
        """Estimate soil temperature based on air temperature"""
        
        air_temp = current["temperature"]
        
        # Simplified soil temperature estimation
        # Soil temperature is generally more stable than air temperature
        soil_temp_5cm = air_temp * 0.9 + 2  # 5cm depth
        soil_temp_10cm = air_temp * 0.85 + 3  # 10cm depth
        soil_temp_20cm = air_temp * 0.8 + 4   # 20cm depth
        
        return {
            "air_temperature": air_temp,
            "soil_5cm": round(soil_temp_5cm, 1),
            "soil_10cm": round(soil_temp_10cm, 1),
            "soil_20cm": round(soil_temp_20cm, 1),
            "note": "Estimated values based on air temperature"
        }
    
    def _assess_application_conditions(self, forecast: Dict) -> Dict:
        """Assess fertilizer application conditions"""
        
        recommendations = []
        suitability_score = 0
        
        # Check next 3 days
        for i, day in enumerate(forecast.get("daily_forecast", [])[:3]):
            day_score = 5  # Base score
            day_notes = []
            
            # Wind conditions
            if day["wind_speed"] > 15:
                day_score -= 2
                day_notes.append("High wind - avoid foliar applications")
            elif day["wind_speed"] < 5:
                day_score += 1
                day_notes.append("Good wind conditions")
            
            # Precipitation
            if day["precipitation"] > 10:
                day_score -= 3
                day_notes.append("Heavy rain expected - delay application")
            elif day["precipitation"] > 2:
                day_score -= 1
                day_notes.append("Light rain expected - consider timing")
            else:
                day_score += 1
                day_notes.append("No precipitation - good for application")
            
            # Temperature
            if day["temp_max"] > 35:
                day_score -= 1
                day_notes.append("High temperature - apply early morning")
            elif 20 <= day["temp_max"] <= 30:
                day_score += 1
                day_notes.append("Optimal temperature range")
            
            recommendations.append({
                "day": i + 1,
                "date": day["date"],
                "suitability_score": max(0, min(5, day_score)),
                "notes": day_notes
            })
            
            suitability_score += day_score
        
        average_suitability = suitability_score / 3 if recommendations else 0
        
        return {
            "overall_suitability": round(average_suitability, 1),
            "daily_recommendations": recommendations,
            "best_application_day": max(recommendations, key=lambda x: x["suitability_score"])["day"] if recommendations else 1
        }
    
    def _assess_drought_risk(self, forecast: Dict) -> Dict:
        """Assess drought risk based on precipitation forecast"""
        
        total_precip = sum(day["precipitation"] for day in forecast.get("daily_forecast", []))
        days_forecast = len(forecast.get("daily_forecast", []))
        
        if days_forecast == 0:
            return {"risk_level": "unknown", "total_precipitation": 0}
        
        avg_daily_precip = total_precip / days_forecast
        
        if avg_daily_precip < 1:
            risk_level = "high"
        elif avg_daily_precip < 3:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return {
            "risk_level": risk_level,
            "total_precipitation_mm": round(total_precip, 1),
            "average_daily_precipitation": round(avg_daily_precip, 1),
            "forecast_days": days_forecast,
            "recommendation": self._get_drought_recommendation(risk_level)
        }
    
    def _assess_leaching_risk(self, forecast: Dict) -> Dict:
        """Assess nutrient leaching risk"""
        
        heavy_rain_days = 0
        total_heavy_rain = 0
        
        for day in forecast.get("daily_forecast", []):
            if day["precipitation"] > 20:  # Heavy rain threshold
                heavy_rain_days += 1
                total_heavy_rain += day["precipitation"]
        
        if heavy_rain_days >= 3:
            risk_level = "high"
        elif heavy_rain_days >= 1:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return {
            "risk_level": risk_level,
            "heavy_rain_days": heavy_rain_days,
            "total_heavy_rain_mm": round(total_heavy_rain, 1),
            "recommendation": self._get_leaching_recommendation(risk_level)
        }
    
    def _get_drought_recommendation(self, risk_level: str) -> str:
        """Get drought management recommendation"""
        
        recommendations = {
            "high": "Consider irrigation if available. Apply fertilizers only if adequate soil moisture is present.",
            "medium": "Monitor soil moisture closely. Consider split applications to reduce risk.",
            "low": "Good conditions for fertilizer application. Follow normal schedule."
        }
        
        return recommendations.get(risk_level, "Monitor weather conditions closely.")
    
    def _get_leaching_recommendation(self, risk_level: str) -> str:
        """Get leaching management recommendation"""
        
        recommendations = {
            "high": "High leaching risk. Consider delaying nitrogen applications until after heavy rains.",
            "medium": "Moderate leaching risk. Use slow-release fertilizers or split applications.",
            "low": "Low leaching risk. Normal fertilizer applications are suitable."
        }
        
        return recommendations.get(risk_level, "Monitor precipitation patterns.")
    
    def _get_mock_current_weather(self) -> Dict:
        """Get mock current weather data when API is not available"""
        
        return {
            "temperature": 28.5,
            "humidity": 65,
            "pressure": 1013,
            "wind_speed": 8.2,
            "wind_direction": 180,
            "precipitation": 0.0,
            "weather_condition": "Clear",
            "description": "clear sky",
            "visibility": 10000,
            "uv_index": 6,
            "timestamp": datetime.now()
        }
    
    def _get_mock_forecast(self, days: int) -> Dict:
        """Get mock forecast data when API is not available"""
        
        daily_forecasts = []
        base_date = datetime.now()
        
        for i in range(days):
            date = base_date + timedelta(days=i)
            daily_forecasts.append({
                "date": date,
                "temp_min": 22 + (i % 3),
                "temp_max": 30 + (i % 4),
                "humidity": 60 + (i % 20),
                "precipitation": 2.5 if i % 3 == 0 else 0.0,
                "wind_speed": 8 + (i % 5),
                "weather_condition": "Partly Cloudy" if i % 2 == 0 else "Clear",
                "description": "partly cloudy" if i % 2 == 0 else "clear sky",
                "pressure": 1013 + (i % 10),
                "uv_index": 6 + (i % 3)
            })
        
        return {
            "daily_forecast": daily_forecasts,
            "hourly_forecast": [],
            "location": {
                "lat": 0.0,
                "lon": 0.0,
                "timezone": "UTC"
            }
        }
    
    def _get_mock_agricultural_indices(self) -> Dict:
        """Get mock agricultural indices when API is not available"""
        
        return {
            "growing_degree_days": {
                "total_accumulated": 125.5,
                "base_temperature": 10.0
            },
            "precipitation_forecast": {
                "total_precipitation_mm": 25.0,
                "rainy_days": 3,
                "average_daily_precip": 3.6
            },
            "evapotranspiration": {
                "total_et_mm": 35.0,
                "average_daily_et": 5.0
            },
            "soil_temperature_estimate": {
                "air_temperature": 28.5,
                "soil_5cm": 27.7,
                "soil_10cm": 27.2,
                "soil_20cm": 26.8
            },
            "fertilizer_application_suitability": {
                "overall_suitability": 4.2,
                "best_application_day": 2
            },
            "drought_risk": {
                "risk_level": "low",
                "recommendation": "Good conditions for fertilizer application."
            },
            "leaching_risk": {
                "risk_level": "medium",
                "recommendation": "Moderate leaching risk. Consider split applications."
            }
        }
