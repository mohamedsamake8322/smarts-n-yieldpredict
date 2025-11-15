import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

class IoTSensorSimulator:
    """
    IoT sensor simulator for agricultural monitoring
    """
    
    def __init__(self):
        self.sensors = {
            "soil_moisture": {"min": 20, "max": 80, "unit": "%", "current": 45},
            "soil_temperature": {"min": 15, "max": 35, "unit": "°C", "current": 25},
            "soil_ph": {"min": 4.5, "max": 8.5, "unit": "pH", "current": 6.2},
            "soil_ec": {"min": 0.1, "max": 3.0, "unit": "dS/m", "current": 1.2},
            "ambient_temperature": {"min": 18, "max": 40, "unit": "°C", "current": 28},
            "humidity": {"min": 30, "max": 95, "unit": "%", "current": 65},
            "light_intensity": {"min": 0, "max": 100000, "unit": "lux", "current": 45000},
            "wind_speed": {"min": 0, "max": 25, "unit": "m/s", "current": 5.2},
            "rainfall": {"min": 0, "max": 50, "unit": "mm/h", "current": 0},
            "atmospheric_pressure": {"min": 980, "max": 1030, "unit": "hPa", "current": 1013}
        }
        
        self.sensor_history = {sensor: [] for sensor in self.sensors.keys()}
        self.alerts = []
        
    def generate_sensor_reading(self, sensor_name: str) -> Dict:
        """Generate realistic sensor reading"""
        
        if sensor_name not in self.sensors:
            return {"error": f"Unknown sensor: {sensor_name}"}
        
        sensor_config = self.sensors[sensor_name]
        current_value = sensor_config["current"]
        
        # Add realistic variation based on sensor type
        if sensor_name == "soil_moisture":
            # Soil moisture changes slowly
            variation = random.uniform(-2, 2)
        elif sensor_name == "soil_temperature":
            # Soil temperature changes slowly
            variation = random.uniform(-1, 1)
        elif sensor_name == "ambient_temperature":
            # Air temperature varies more
            variation = random.uniform(-3, 3)
        elif sensor_name == "humidity":
            # Humidity can change moderately
            variation = random.uniform(-5, 5)
        elif sensor_name == "wind_speed":
            # Wind speed can be quite variable
            variation = random.uniform(-2, 3)
        elif sensor_name == "rainfall":
            # Rainfall is either 0 or significant
            variation = random.choice([0, 0, 0, 0, random.uniform(1, 10)])
        else:
            # Default variation
            variation = random.uniform(-1, 1)
        
        # Apply variation with bounds checking
        new_value = current_value + variation
        new_value = max(sensor_config["min"], min(sensor_config["max"], new_value))
        
        # Update current value
        sensor_config["current"] = new_value
        
        # Create reading
        reading = {
            "sensor_id": sensor_name,
            "value": round(new_value, 2),
            "unit": sensor_config["unit"],
            "timestamp": datetime.now().isoformat(),
            "status": "online",
            "battery_level": random.randint(70, 100),
            "signal_strength": random.randint(60, 100)
        }
        
        # Store in history
        self.sensor_history[sensor_name].append(reading)
        
        # Keep only last 100 readings
        if len(self.sensor_history[sensor_name]) > 100:
            self.sensor_history[sensor_name] = self.sensor_history[sensor_name][-100:]
        
        # Check for alerts
        self._check_alerts(sensor_name, new_value)
        
        return reading
    
    def get_all_current_readings(self) -> Dict:
        """Get current readings from all sensors"""
        
        readings = {}
        for sensor_name in self.sensors.keys():
            readings[sensor_name] = self.generate_sensor_reading(sensor_name)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "farm_id": "demo_farm_001",
            "location": {"lat": -1.2921, "lon": 36.8219},  # Nairobi coordinates
            "sensors": readings,
            "system_status": "operational"
        }
    
    def get_sensor_history(self, sensor_name: str, hours: int = 24) -> List[Dict]:
        """Get sensor history for specified hours"""
        
        if sensor_name not in self.sensor_history:
            return []
        
        # Filter by time if needed
        cutoff_time = datetime.now() - timedelta(hours=hours)
        history = self.sensor_history[sensor_name]
        
        filtered_history = []
        for reading in history:
            try:
                reading_time = datetime.fromisoformat(reading["timestamp"])
                if reading_time >= cutoff_time:
                    filtered_history.append(reading)
            except ValueError:
                continue
        
        return filtered_history
    
    def _check_alerts(self, sensor_name: str, value: float):
        """Check for alert conditions"""
        
        alert_conditions = {
            "soil_moisture": {"low": 25, "high": 75},
            "soil_temperature": {"low": 18, "high": 32},
            "soil_ph": {"low": 5.5, "high": 7.5},
            "ambient_temperature": {"low": 20, "high": 35},
            "humidity": {"low": 40, "high": 85},
            "wind_speed": {"low": 0, "high": 15}
        }
        
        if sensor_name in alert_conditions:
            conditions = alert_conditions[sensor_name]
            
            if value < conditions["low"]:
                self._create_alert(sensor_name, value, f"Low {sensor_name.replace('_', ' ')}", "warning")
            elif value > conditions["high"]:
                self._create_alert(sensor_name, value, f"High {sensor_name.replace('_', ' ')}", "warning")
    
    def _create_alert(self, sensor_name: str, value: float, message: str, severity: str):
        """Create an alert"""
        
        alert = {
            "alert_id": f"alert_{int(time.time())}_{sensor_name}",
            "sensor_name": sensor_name,
            "value": value,
            "message": message,
            "severity": severity,
            "timestamp": datetime.now().isoformat(),
            "acknowledged": False
        }
        
        self.alerts.append(alert)
        
        # Keep only last 50 alerts
        if len(self.alerts) > 50:
            self.alerts = self.alerts[-50:]
    
    def get_active_alerts(self) -> List[Dict]:
        """Get all active (unacknowledged) alerts"""
        
        return [alert for alert in self.alerts if not alert["acknowledged"]]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        
        for alert in self.alerts:
            if alert["alert_id"] == alert_id:
                alert["acknowledged"] = True
                return True
        
        return False
    
    def get_agricultural_summary(self) -> Dict:
        """Get agricultural summary from sensor data"""
        
        current_readings = self.get_all_current_readings()
        sensors = current_readings.get("sensors", {})
        
        # Calculate derived metrics
        soil_moisture = sensors.get("soil_moisture", {}).get("value", 0)
        soil_temp = sensors.get("soil_temperature", {}).get("value", 0)
        air_temp = sensors.get("ambient_temperature", {}).get("value", 0)
        humidity = sensors.get("humidity", {}).get("value", 0)
        
        # Irrigation need assessment
        if soil_moisture < 30:
            irrigation_need = "high"
        elif soil_moisture < 50:
            irrigation_need = "medium"
        else:
            irrigation_need = "low"
        
        # Growing conditions assessment
        if 20 <= soil_temp <= 30 and 40 <= soil_moisture <= 70:
            growing_conditions = "optimal"
        elif 15 <= soil_temp <= 35 and 25 <= soil_moisture <= 80:
            growing_conditions = "good"
        else:
            growing_conditions = "suboptimal"
        
        # Fertilizer application suitability
        wind_speed = sensors.get("wind_speed", {}).get("value", 0)
        rainfall = sensors.get("rainfall", {}).get("value", 0)
        
        if rainfall > 5:
            fertilizer_application = "not_suitable"
        elif wind_speed > 10:
            fertilizer_application = "moderate"
        else:
            fertilizer_application = "suitable"
        
        return {
            "timestamp": datetime.now().isoformat(),
            "soil_conditions": {
                "moisture_level": soil_moisture,
                "temperature": soil_temp,
                "irrigation_need": irrigation_need
            },
            "environmental_conditions": {
                "air_temperature": air_temp,
                "humidity": humidity,
                "wind_speed": wind_speed,
                "rainfall_rate": rainfall
            },
            "agricultural_indices": {
                "growing_conditions": growing_conditions,
                "fertilizer_application_suitability": fertilizer_application,
                "pest_disease_risk": "low" if humidity < 70 else "medium"
            },
            "recommendations": self._generate_recommendations(sensors),
            "active_alerts_count": len(self.get_active_alerts())
        }
    
    def _generate_recommendations(self, sensors: Dict) -> List[str]:
        """Generate recommendations based on sensor data"""
        
        recommendations = []
        
        # Soil moisture recommendations
        soil_moisture = sensors.get("soil_moisture", {}).get("value", 0)
        if soil_moisture < 30:
            recommendations.append("Consider irrigation - soil moisture is low")
        elif soil_moisture > 75:
            recommendations.append("Reduce irrigation - soil moisture is high")
        
        # Temperature recommendations
        soil_temp = sensors.get("soil_temperature", {}).get("value", 0)
        if soil_temp < 18:
            recommendations.append("Soil temperature is low - consider mulching")
        elif soil_temp > 32:
            recommendations.append("Soil temperature is high - provide shade or increase irrigation")
        
        # pH recommendations
        soil_ph = sensors.get("soil_ph", {}).get("value", 0)
        if soil_ph < 5.5:
            recommendations.append("Soil is acidic - consider lime application")
        elif soil_ph > 7.5:
            recommendations.append("Soil is alkaline - monitor nutrient availability")
        
        # Wind and rainfall recommendations
        wind_speed = sensors.get("wind_speed", {}).get("value", 0)
        rainfall = sensors.get("rainfall", {}).get("value", 0)
        
        if wind_speed > 15:
            recommendations.append("High wind conditions - avoid foliar applications")
        
        if rainfall > 10:
            recommendations.append("Heavy rainfall - delay fertilizer applications")
        elif rainfall == 0 and sensors.get("humidity", {}).get("value", 0) < 40:
            recommendations.append("Dry conditions - monitor crop stress")
        
        return recommendations
    
    def simulate_data_collection_period(self, hours: int = 24, interval_minutes: int = 60) -> Dict:
        """Simulate data collection over a period"""
        
        total_readings = (hours * 60) // interval_minutes
        collected_data = {
            "period_start": (datetime.now() - timedelta(hours=hours)).isoformat(),
            "period_end": datetime.now().isoformat(),
            "total_readings": total_readings,
            "sensors": {},
            "summary_statistics": {}
        }
        
        # Generate readings for each sensor
        for sensor_name in self.sensors.keys():
            sensor_readings = []
            
            for i in range(total_readings):
                reading_time = datetime.now() - timedelta(hours=hours) + timedelta(minutes=i * interval_minutes)
                reading = self.generate_sensor_reading(sensor_name)
                reading["timestamp"] = reading_time.isoformat()
                sensor_readings.append(reading)
            
            collected_data["sensors"][sensor_name] = sensor_readings
            
            # Calculate statistics
            values = [r["value"] for r in sensor_readings]
            collected_data["summary_statistics"][sensor_name] = {
                "min": round(min(values), 2),
                "max": round(max(values), 2),
                "average": round(sum(values) / len(values), 2),
                "latest": round(values[-1], 2)
            }
        
        return collected_data
    
    def export_sensor_data(self, format_type: str = "json") -> str:
        """Export sensor data in specified format"""
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "farm_id": "demo_farm_001",
            "current_readings": self.get_all_current_readings(),
            "sensor_history": self.sensor_history,
            "active_alerts": self.get_active_alerts(),
            "agricultural_summary": self.get_agricultural_summary()
        }
        
        if format_type.lower() == "json":
            return json.dumps(export_data, indent=2, default=str)
        else:
            return str(export_data)
