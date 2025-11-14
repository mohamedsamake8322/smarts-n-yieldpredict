
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any, Tuple
import random

class IoTSensor:
    def __init__(self, sensor_id: str, sensor_type: str, location: Dict[str, float]):
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type
        self.location = location  # {'lat': float, 'lon': float}
        self.is_active = True
        self.last_reading = None
        self.calibration_date = datetime.now()
        self.battery_level = 100
        
    def get_reading(self) -> Dict[str, Any]:
        """Simulate sensor reading"""
        timestamp = datetime.now()
        
        # Simulate different sensor types
        if self.sensor_type == 'soil_moisture':
            value = np.random.normal(45, 10)  # Percentage
            value = max(0, min(100, value))
        elif self.sensor_type == 'soil_temperature':
            value = np.random.normal(18, 5)  # Celsius
        elif self.sensor_type == 'soil_ph':
            value = np.random.normal(6.5, 0.5)
            value = max(4.0, min(9.0, value))
        elif self.sensor_type == 'air_temperature':
            value = np.random.normal(22, 8)
        elif self.sensor_type == 'humidity':
            value = np.random.normal(65, 15)
            value = max(0, min(100, value))
        elif self.sensor_type == 'light_intensity':
            value = np.random.normal(50000, 15000)  # Lux
            value = max(0, value)
        elif self.sensor_type == 'wind_speed':
            value = np.random.exponential(5)  # m/s
        elif self.sensor_type == 'rainfall':
            value = np.random.exponential(2) if np.random.random() < 0.3 else 0  # mm
        else:
            value = np.random.normal(50, 10)
        
        # Simulate battery drain
        self.battery_level = max(0, self.battery_level - 0.01)
        
        reading = {
            'sensor_id': self.sensor_id,
            'sensor_type': self.sensor_type,
            'timestamp': timestamp.isoformat(),
            'value': round(value, 2),
            'unit': self._get_unit(),
            'location': self.location,
            'battery_level': round(self.battery_level, 1),
            'status': 'active' if self.is_active and self.battery_level > 10 else 'low_battery'
        }
        
        self.last_reading = reading
        return reading
    
    def _get_unit(self) -> str:
        """Get measurement unit for sensor type"""
        units = {
            'soil_moisture': '%',
            'soil_temperature': '°C',
            'soil_ph': 'pH',
            'air_temperature': '°C',
            'humidity': '%',
            'light_intensity': 'lux',
            'wind_speed': 'm/s',
            'rainfall': 'mm'
        }
        return units.get(self.sensor_type, 'units')

class SmartIrrigationSystem:
    def __init__(self):
        self.zones = {}
        self.is_active = True
        self.schedule = []
        self.water_usage = 0
        self.efficiency_score = 85
        
    def add_zone(self, zone_id: str, area: float, crop_type: str, sensors: List[str]):
        """Add irrigation zone"""
        self.zones[zone_id] = {
            'area': area,  # hectares
            'crop_type': crop_type,
            'sensors': sensors,
            'last_irrigation': None,
            'water_requirement': self._calculate_water_requirement(crop_type),
            'is_irrigating': False,
            'moisture_threshold': 30,  # Percentage below which irrigation starts
            'target_moisture': 70     # Target moisture level
        }
    
    def _calculate_water_requirement(self, crop_type: str) -> float:
        """Calculate daily water requirement for crop type (mm/day)"""
        requirements = {
            'wheat': 4.5,
            'corn': 6.0,
            'rice': 8.0,
            'soybeans': 5.0,
            'cotton': 5.5,
            'tomatoes': 7.0,
            'potatoes': 4.0
        }
        return requirements.get(crop_type.lower(), 5.0)
    
    def evaluate_irrigation_needs(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Determine irrigation needs based on sensor data and weather"""
        recommendations = {}
        
        for zone_id, zone in self.zones.items():
            zone_sensors = [s for s in sensor_data if s['sensor_id'] in zone['sensors']]
            
            if not zone_sensors:
                continue
            
            # Get average soil moisture for zone
            moisture_readings = [s['value'] for s in zone_sensors if s['sensor_type'] == 'soil_moisture']
            avg_moisture = np.mean(moisture_readings) if moisture_readings else 50
            
            # Get weather factors
            temp_readings = [s['value'] for s in zone_sensors if s['sensor_type'] == 'air_temperature']
            avg_temp = np.mean(temp_readings) if temp_readings else 25
            
            humidity_readings = [s['value'] for s in zone_sensors if s['sensor_type'] == 'humidity']
            avg_humidity = np.mean(humidity_readings) if humidity_readings else 65
            
            # Calculate evapotranspiration estimate
            et_rate = self._calculate_et_rate(avg_temp, avg_humidity, zone['crop_type'])
            
            # Determine irrigation need
            needs_irrigation = avg_moisture < zone['moisture_threshold']
            
            # Calculate irrigation amount needed
            if needs_irrigation:
                moisture_deficit = zone['target_moisture'] - avg_moisture
                irrigation_amount = (moisture_deficit / 100) * zone['area'] * 10  # mm to liters/hectare conversion
            else:
                irrigation_amount = 0
            
            recommendations[zone_id] = {
                'needs_irrigation': needs_irrigation,
                'current_moisture': round(avg_moisture, 1),
                'target_moisture': zone['target_moisture'],
                'irrigation_amount': round(irrigation_amount, 1),
                'et_rate': round(et_rate, 2),
                'priority': 'high' if avg_moisture < 20 else 'medium' if avg_moisture < 30 else 'low',
                'estimated_duration': round(irrigation_amount / 50, 1) if irrigation_amount > 0 else 0  # hours
            }
        
        return recommendations
    
    def _calculate_et_rate(self, temperature: float, humidity: float, crop_type: str) -> float:
        """Calculate evapotranspiration rate"""
        # Simplified ET calculation
        base_et = 0.0023 * (temperature + 17.8) * np.sqrt(abs(temperature - humidity)) * 2.45
        
        # Crop coefficient
        crop_coefficients = {
            'wheat': 1.0,
            'corn': 1.2,
            'rice': 1.3,
            'soybeans': 1.1,
            'cotton': 1.1,
            'tomatoes': 1.15,
            'potatoes': 1.05
        }
        
        kc = crop_coefficients.get(crop_type.lower(), 1.0)
        return base_et * kc
    
    def create_irrigation_schedule(self, recommendations: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create optimized irrigation schedule"""
        schedule = []
        
        # Sort zones by priority
        sorted_zones = sorted(recommendations.items(), 
                             key=lambda x: {'high': 3, 'medium': 2, 'low': 1}[x[1]['priority']], 
                             reverse=True)
        
        current_time = datetime.now()
        
        for zone_id, rec in sorted_zones:
            if rec['needs_irrigation']:
                # Schedule for early morning (optimal time)
                schedule_time = current_time.replace(hour=5, minute=0, second=0) + timedelta(days=0 if current_time.hour < 6 else 1)
                
                schedule.append({
                    'zone_id': zone_id,
                    'scheduled_time': schedule_time.isoformat(),
                    'duration_hours': rec['estimated_duration'],
                    'water_amount': rec['irrigation_amount'],
                    'priority': rec['priority'],
                    'status': 'scheduled'
                })
        
        return schedule
    
    def execute_irrigation(self, zone_id: str, duration: float) -> Dict[str, Any]:
        """Execute irrigation for a specific zone"""
        if zone_id not in self.zones:
            return {'success': False, 'error': 'Zone not found'}
        
        zone = self.zones[zone_id]
        water_used = duration * 50 * zone['area']  # Approximate water usage
        
        # Update zone status
        zone['is_irrigating'] = True
        zone['last_irrigation'] = datetime.now().isoformat()
        
        # Update system water usage
        self.water_usage += water_used
        
        return {
            'success': True,
            'zone_id': zone_id,
            'duration': duration,
            'water_used': water_used,
            'start_time': datetime.now().isoformat(),
            'estimated_end_time': (datetime.now() + timedelta(hours=duration)).isoformat()
        }

class PlantStressDetector:
    def __init__(self):
        self.stress_indicators = {
            'water_stress': {
                'soil_moisture_threshold': 25,
                'temperature_threshold': 35,
                'humidity_threshold': 40
            },
            'heat_stress': {
                'temperature_threshold': 40,
                'duration_threshold': 2  # hours
            },
            'nutrient_deficiency': {
                'ph_range': (5.5, 7.5),
                'conductivity_threshold': 0.5
            },
            'disease_risk': {
                'humidity_threshold': 85,
                'temperature_range': (20, 30),
                'duration_threshold': 6  # hours
            }
        }
        
    def analyze_stress_indicators(self, sensor_data: List[Dict[str, Any]], 
                                historical_data: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze sensor data to detect plant stress"""
        stress_analysis = {
            'overall_stress_level': 'low',
            'detected_stresses': [],
            'recommendations': [],
            'alerts': []
        }
        
        # Current conditions
        current_moisture = self._get_sensor_value(sensor_data, 'soil_moisture')
        current_temp = self._get_sensor_value(sensor_data, 'air_temperature')
        current_humidity = self._get_sensor_value(sensor_data, 'humidity')
        current_ph = self._get_sensor_value(sensor_data, 'soil_ph')
        
        # Water stress detection
        if current_moisture and current_moisture < self.stress_indicators['water_stress']['soil_moisture_threshold']:
            stress_analysis['detected_stresses'].append({
                'type': 'water_stress',
                'severity': 'high' if current_moisture < 15 else 'medium',
                'indicator': f'Soil moisture: {current_moisture}%',
                'recommendation': 'Immediate irrigation required'
            })
            stress_analysis['recommendations'].append('Activate irrigation system')
            stress_analysis['alerts'].append('URGENT: Severe water stress detected')
        
        # Heat stress detection
        if current_temp and current_temp > self.stress_indicators['heat_stress']['temperature_threshold']:
            stress_analysis['detected_stresses'].append({
                'type': 'heat_stress',
                'severity': 'high' if current_temp > 45 else 'medium',
                'indicator': f'Air temperature: {current_temp}°C',
                'recommendation': 'Increase irrigation frequency, provide shade if possible'
            })
            stress_analysis['recommendations'].append('Implement heat mitigation strategies')
        
        # Nutrient deficiency indicators
        if current_ph:
            ph_range = self.stress_indicators['nutrient_deficiency']['ph_range']
            if not (ph_range[0] <= current_ph <= ph_range[1]):
                stress_analysis['detected_stresses'].append({
                    'type': 'nutrient_deficiency',
                    'severity': 'medium',
                    'indicator': f'Soil pH: {current_ph}',
                    'recommendation': 'Soil amendment required - lime or sulfur application'
                })
                stress_analysis['recommendations'].append('Adjust soil pH')
        
        # Disease risk assessment
        if (current_humidity and current_humidity > self.stress_indicators['disease_risk']['humidity_threshold'] and
            current_temp and 20 <= current_temp <= 30):
            stress_analysis['detected_stresses'].append({
                'type': 'disease_risk',
                'severity': 'medium',
                'indicator': f'High humidity ({current_humidity}%) + moderate temperature',
                'recommendation': 'Monitor for disease symptoms, improve air circulation'
            })
            stress_analysis['recommendations'].append('Increase disease monitoring')
        
        # Determine overall stress level
        if any(s['severity'] == 'high' for s in stress_analysis['detected_stresses']):
            stress_analysis['overall_stress_level'] = 'high'
        elif stress_analysis['detected_stresses']:
            stress_analysis['overall_stress_level'] = 'medium'
        
        return stress_analysis
    
    def _get_sensor_value(self, sensor_data: List[Dict[str, Any]], sensor_type: str) -> float:
        """Get the latest value for a specific sensor type"""
        matching_sensors = [s for s in sensor_data if s['sensor_type'] == sensor_type]
        if matching_sensors:
            return matching_sensors[-1]['value']  # Get latest reading
        return None
    
    def generate_automated_actions(self, stress_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate automated actions based on stress analysis"""
        actions = []
        
        for stress in stress_analysis['detected_stresses']:
            if stress['type'] == 'water_stress' and stress['severity'] == 'high':
                actions.append({
                    'action_type': 'irrigation',
                    'priority': 'immediate',
                    'parameters': {
                        'duration': 2.0,  # hours
                        'intensity': 'high'
                    },
                    'reason': 'Severe water stress detected'
                })
            
            elif stress['type'] == 'heat_stress':
                actions.append({
                    'action_type': 'cooling',
                    'priority': 'high',
                    'parameters': {
                        'increase_irrigation': True,
                        'timing': 'early_morning_evening'
                    },
                    'reason': 'Heat stress mitigation'
                })
            
            elif stress['type'] == 'disease_risk':
                actions.append({
                    'action_type': 'monitoring',
                    'priority': 'medium',
                    'parameters': {
                        'increase_frequency': True,
                        'check_for_symptoms': True
                    },
                    'reason': 'High disease risk conditions'
                })
        
        return actions

class IoTDataManager:
    def __init__(self):
        self.sensors = {}
        self.irrigation_system = SmartIrrigationSystem()
        self.stress_detector = PlantStressDetector()
        self.data_history = []
        
    def add_sensor(self, sensor_id: str, sensor_type: str, location: Dict[str, float]):
        """Add a new IoT sensor to the system"""
        self.sensors[sensor_id] = IoTSensor(sensor_id, sensor_type, location)
        
    def collect_all_readings(self) -> List[Dict[str, Any]]:
        """Collect readings from all active sensors"""
        readings = []
        for sensor in self.sensors.values():
            if sensor.is_active:
                reading = sensor.get_reading()
                readings.append(reading)
        
        # Store in history
        self.data_history.extend(readings)
        
        # Keep only last 1000 readings for memory management
        if len(self.data_history) > 1000:
            self.data_history = self.data_history[-1000:]
        
        return readings
    
    def get_sensor_dashboard_data(self) -> Dict[str, Any]:
        """Get formatted data for dashboard display"""
        current_readings = self.collect_all_readings()
        
        # Group by sensor type
        sensor_groups = {}
        for reading in current_readings:
            sensor_type = reading['sensor_type']
            if sensor_type not in sensor_groups:
                sensor_groups[sensor_type] = []
            sensor_groups[sensor_type].append(reading)
        
        # Calculate summary statistics
        summary = {}
        for sensor_type, readings in sensor_groups.items():
            values = [r['value'] for r in readings]
            summary[sensor_type] = {
                'count': len(values),
                'average': round(np.mean(values), 2),
                'min': round(min(values), 2),
                'max': round(max(values), 2),
                'latest_reading': readings[-1]['timestamp'],
                'unit': readings[0]['unit']
            }
        
        return {
            'current_readings': current_readings,
            'summary': summary,
            'total_sensors': len(self.sensors),
            'active_sensors': len([s for s in self.sensors.values() if s.is_active]),
            'low_battery_sensors': len([s for s in self.sensors.values() if s.battery_level < 20])
        }
    
    def run_automated_analysis(self) -> Dict[str, Any]:
        """Run automated analysis and generate recommendations"""
        current_readings = self.collect_all_readings()
        
        # Stress analysis
        stress_analysis = self.stress_detector.analyze_stress_indicators(current_readings)
        
        # Irrigation recommendations
        irrigation_needs = self.irrigation_system.evaluate_irrigation_needs(current_readings)
        
        # Generate automated actions
        automated_actions = self.stress_detector.generate_automated_actions(stress_analysis)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'stress_analysis': stress_analysis,
            'irrigation_recommendations': irrigation_needs,
            'automated_actions': automated_actions,
            'sensor_status': self.get_sensor_status()
        }
    
    def get_sensor_status(self) -> Dict[str, Any]:
        """Get status of all sensors"""
        status = {
            'total': len(self.sensors),
            'active': 0,
            'inactive': 0,
            'low_battery': 0,
            'sensors': []
        }
        
        for sensor in self.sensors.values():
            sensor_status = {
                'id': sensor.sensor_id,
                'type': sensor.sensor_type,
                'active': sensor.is_active,
                'battery': sensor.battery_level,
                'last_reading': sensor.last_reading['timestamp'] if sensor.last_reading else None
            }
            status['sensors'].append(sensor_status)
            
            if sensor.is_active:
                status['active'] += 1
            else:
                status['inactive'] += 1
                
            if sensor.battery_level < 20:
                status['low_battery'] += 1
        
        return status

# Global IoT system instance
iot_system = IoTDataManager()

# Initialize with sample sensors
sample_sensors = [
    ('soil_01', 'soil_moisture', {'lat': 45.5017, 'lon': -73.5673}),
    ('soil_02', 'soil_temperature', {'lat': 45.5018, 'lon': -73.5674}),
    ('soil_03', 'soil_ph', {'lat': 45.5019, 'lon': -73.5675}),
    ('air_01', 'air_temperature', {'lat': 45.5020, 'lon': -73.5676}),
    ('air_02', 'humidity', {'lat': 45.5021, 'lon': -73.5677}),
    ('light_01', 'light_intensity', {'lat': 45.5022, 'lon': -73.5678}),
    ('weather_01', 'wind_speed', {'lat': 45.5023, 'lon': -73.5679}),
    ('rain_01', 'rainfall', {'lat': 45.5024, 'lon': -73.5680})
]

for sensor_id, sensor_type, location in sample_sensors:
    iot_system.add_sensor(sensor_id, sensor_type, location)

# Add irrigation zones
iot_system.irrigation_system.add_zone('zone_1', 10.5, 'wheat', ['soil_01', 'soil_02'])
iot_system.irrigation_system.add_zone('zone_2', 8.2, 'corn', ['soil_03', 'air_01'])
