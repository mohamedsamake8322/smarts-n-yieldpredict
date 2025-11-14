import requests
import json
import os
import streamlit as st  # Ajout pour gérer les secrets sur Streamlit Cloud
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from dotenv import load_dotenv

# ✅ Charger les variables d'environnement
load_dotenv()

# 🔎 Vérification immédiate (uniquement en local)
if "STREAMLIT_CLOUD" not in os.environ:
    print("🔎 OPENWEATHER_API_KEY :", os.getenv("OPENWEATHER_API_KEY"))
    print("🔎 WEATHERAPI_KEY :", os.getenv("WEATHERAPI_KEY"))

class WeatherAPI:
    """
    Weather API integration for agricultural applications.
    Supports multiple weather data providers with fallback options.
    """

    def __init__(self):
        # 📌 Charger les variables d'environnement
        load_dotenv()

        # ✅ Vérifier les API Keys localement et sur Streamlit Cloud
        self.openweather_api_key = os.getenv("OPENWEATHER_API_KEY") or st.secrets.get("OPENWEATHER_API_KEY")
        self.weatherapi_key = os.getenv("WEATHERAPI_KEY", st.secrets.get("WEATHERAPI_KEY"))

        # 🚨 Si aucune clé API n'est trouvée, générer une erreur
        if not self.openweather_api_key or not self.weatherapi_key:
            raise ValueError("❌ API keys are missing! Add them in Streamlit Cloud Secrets.")

        # ✅ URLs des services météo
        self.openweather_base_url = "https://api.openweathermap.org/data/2.5"
        self.weatherapi_base_url = "https://api.weatherapi.com/v1"

        # ✅ Cache des requêtes
        self._cache = {}
        self._cache_duration = 600  # 10 minutes

    def _get_cache_key(self, endpoint: str, params: dict) -> str:
        """Generate cache key for API requests."""
        return f"{endpoint}_{hash(str(sorted(params.items())))}"

    def _is_cache_valid(self, cache_entry: dict) -> bool:
        """Check if cached data is still valid."""
        if not cache_entry:
            return False

        cache_time = cache_entry.get('timestamp', 0)
        current_time = datetime.now().timestamp()

        return (current_time - cache_time) < self._cache_duration

    def _make_request(self, url: str, params: dict) -> Optional[dict]:
        """Make HTTP request with error handling."""
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            print(f"❌ HTTP Error: {e}")
        except requests.exceptions.ConnectionError:
            print("❌ Connection error! Check your network.")
        except requests.exceptions.Timeout:
            print("❌ Timeout error! Server took too long to respond.")
        except requests.exceptions.RequestException as e:
            print(f"❌ API request failed: {e}")
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse JSON response: {e}")

        return None
    def get_current_weather(self, location: str) -> Optional[Dict]:
        """
        Get current weather data for a location.

        Args:
            location: Location string (city name, coordinates, etc.)

        Returns:
            Dictionary with current weather data or None if failed
        """
        cache_key = self._get_cache_key("current", {"location": location})

        # Check cache first
        if cache_key in self._cache and self._is_cache_valid(self._cache[cache_key]):
            return self._cache[cache_key]['data']

        # Try OpenWeatherMap first
        weather_data = self._get_openweather_current(location)

        # Fallback to WeatherAPI if OpenWeatherMap fails
        if not weather_data:
            weather_data = self._get_weatherapi_current(location)

        # If still no data, return mock data for demonstration
        if not weather_data:
            weather_data = self._get_mock_current_weather(location)

        # Cache the result
        if weather_data:
            self._cache[cache_key] = {
                'data': weather_data,
                'timestamp': datetime.now().timestamp()
            }

        return weather_data

    def _get_openweather_current(self, location: str) -> Optional[Dict]:
        """Get current weather from OpenWeatherMap API."""
        url = f"{self.openweather_base_url}/weather"
        params = {
            'q': location,
            'appid': self.openweather_api_key,
            'units': 'metric'
        }

        response_data = self._make_request(url, params)
        if not response_data:
            return None

        try:
            return {
                'temperature': response_data['main']['temp'],
                'feels_like': response_data['main']['feels_like'],
                'humidity': response_data['main']['humidity'],
                'pressure': response_data['main']['pressure'],
                'wind_speed': response_data.get('wind', {}).get('speed', 0) * 3.6,  # m/s to km/h
                'wind_direction': response_data.get('wind', {}).get('deg', 0),
                'visibility': response_data.get('visibility', 0) / 1000,  # m to km
                'uv_index': response_data.get('uvi', 0),
                'description': response_data['weather'][0]['description'],
                'location': response_data['name'],
                'country': response_data['sys']['country']
            }
        except KeyError as e:
            print(f"Error parsing OpenWeatherMap response: {e}")
            return None

    def _get_weatherapi_current(self, location: str) -> Optional[Dict]:
        """Get current weather from WeatherAPI."""
        url = f"{self.weatherapi_base_url}/current.json"
        params = {
            'key': self.weatherapi_key,
            'q': location,
            'aqi': 'no'
        }

        response_data = self._make_request(url, params)
        if not response_data:
            return None

        try:
            current = response_data['current']
            location_data = response_data['location']

            return {
                'temperature': current['temp_c'],
                'feels_like': current['feelslike_c'],
                'humidity': current['humidity'],
                'pressure': current['pressure_mb'],
                'wind_speed': current['wind_kph'],
                'wind_direction': current['wind_degree'],
                'visibility': current['vis_km'],
                'uv_index': current['uv'],
                'description': current['condition']['text'],
                'location': location_data['name'],
                'country': location_data['country']
            }
        except KeyError as e:
            print(f"Error parsing WeatherAPI response: {e}")
            return None

    def _get_mock_current_weather(self, location: str) -> Dict:
        """Generate mock weather data for demonstration purposes."""
        import random

        return {
            'temperature': round(random.uniform(15, 30), 1),
            'feels_like': round(random.uniform(15, 30), 1),
            'humidity': random.randint(40, 80),
            'pressure': random.randint(1000, 1020),
            'wind_speed': round(random.uniform(5, 25), 1),
            'wind_direction': random.randint(0, 360),
            'visibility': round(random.uniform(10, 50), 1),
            'uv_index': random.randint(1, 10),
            'description': random.choice(['Clear sky', 'Partly cloudy', 'Overcast', 'Light rain']),
            'location': location.split(',')[0],
            'country': 'Demo'
        }

    def get_forecast(self, location: str, days: int = 5) -> Optional[List[Dict]]:
        """
        Get weather forecast for specified days.

        Args:
            location: Location string
            days: Number of days to forecast (1-10)

        Returns:
            List of forecast data or None if failed
        """
        cache_key = self._get_cache_key("forecast", {"location": location, "days": days})

        # Check cache first
        if cache_key in self._cache and self._is_cache_valid(self._cache[cache_key]):
            return self._cache[cache_key]['data']

        # Try OpenWeatherMap first
        forecast_data = self._get_openweather_forecast(location, days)

        # Fallback to WeatherAPI
        if not forecast_data:
            forecast_data = self._get_weatherapi_forecast(location, days)

        # If still no data, return mock data
        if not forecast_data:
            forecast_data = self._get_mock_forecast(location, days)

        # Cache the result
        if forecast_data:
            self._cache[cache_key] = {
                'data': forecast_data,
                'timestamp': datetime.now().timestamp()
            }

        return forecast_data

    def _get_openweather_forecast(self, location: str, days: int) -> Optional[List[Dict]]:
        """Get forecast from OpenWeatherMap API."""
        url = f"{self.openweather_base_url}/forecast"
        params = {
            'q': location,
            'appid': self.openweather_api_key,
            'units': 'metric',
            'cnt': days * 8  # 8 forecasts per day (3-hour intervals)
        }

        response_data = self._make_request(url, params)
        if not response_data:
            return None

        try:
            forecast_list = []
            daily_data = {}

            for item in response_data['list']:
                date = datetime.fromtimestamp(item['dt']).date()

                if date not in daily_data:
                    daily_data[date] = {
                        'temperatures': [],
                        'humidity': [],
                        'precipitation': 0,
                        'description': item['weather'][0]['description']
                    }

                daily_data[date]['temperatures'].append(item['main']['temp'])
                daily_data[date]['humidity'].append(item['main']['humidity'])

                # Add precipitation if available
                if 'rain' in item:
                    daily_data[date]['precipitation'] += item['rain'].get('3h', 0)
                if 'snow' in item:
                    daily_data[date]['precipitation'] += item['snow'].get('3h', 0)

            for date, data in daily_data.items():
                forecast_list.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'temperature_max': max(data['temperatures']),
                    'temperature_min': min(data['temperatures']),
                    'humidity': sum(data['humidity']) / len(data['humidity']),
                    'precipitation': data['precipitation'],
                    'description': data['description']
                })

            return forecast_list[:days]

        except KeyError as e:
            print(f"Error parsing OpenWeatherMap forecast: {e}")
            return None

    def _get_weatherapi_forecast(self, location: str, days: int) -> Optional[List[Dict]]:
        """Get forecast from WeatherAPI."""
        url = f"{self.weatherapi_base_url}/forecast.json"
        params = {
            'key': self.weatherapi_key,
            'q': location,
            'days': min(days, 10),  # WeatherAPI supports up to 10 days
            'aqi': 'no',
            'alerts': 'no'
        }

        response_data = self._make_request(url, params)
        if not response_data:
            return None

        try:
            forecast_list = []

            for day_data in response_data['forecast']['forecastday']:
                day = day_data['day']
                forecast_list.append({
                    'date': day_data['date'],
                    'temperature_max': day['maxtemp_c'],
                    'temperature_min': day['mintemp_c'],
                    'humidity': day['avghumidity'],
                    'precipitation': day['totalprecip_mm'],
                    'description': day['condition']['text']
                })

            return forecast_list

        except KeyError as e:
            print(f"Error parsing WeatherAPI forecast: {e}")
            return None

    def _get_mock_forecast(self, location: str, days: int) -> List[Dict]:
        """Generate mock forecast data."""
        import random

        forecast_list = []
        base_date = datetime.now().date()

        for i in range(days):
            date = base_date + timedelta(days=i)
            temp_max = round(random.uniform(20, 35), 1)
            temp_min = round(random.uniform(10, temp_max - 5), 1)

            forecast_list.append({
                'date': date.strftime('%Y-%m-%d'),
                'temperature_max': temp_max,
                'temperature_min': temp_min,
                'humidity': random.randint(40, 80),
                'precipitation': round(random.uniform(0, 10), 1),
                'description': random.choice([
                    'Sunny', 'Partly cloudy', 'Cloudy', 'Light rain', 'Showers'
                ])
            })

        return forecast_list

    def get_historical_data(self, location: str, start_date: datetime, end_date: datetime) -> Optional[List[Dict]]:
        """
        Get historical weather data for a date range.

        Args:
            location: Location string
            start_date: Start date for historical data
            end_date: End date for historical data

        Returns:
            List of historical weather data or None if failed
        """
        # For demonstration, return mock historical data
        return self._get_mock_historical_data(location, start_date, end_date)

    def _get_mock_historical_data(self, location: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Generate mock historical weather data."""
        import random

        historical_data = []
        current_date = start_date

        while current_date <= end_date:
            historical_data.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'temperature': round(random.uniform(15, 30), 1),
                'humidity': random.randint(40, 80),
                'precipitation': round(random.uniform(0, 15), 1),
                'wind_speed': round(random.uniform(5, 25), 1),
                'pressure': random.randint(1000, 1020)
            })
            current_date += timedelta(days=1)

        return historical_data

    def get_agricultural_weather_index(self, location: str) -> Optional[Dict]:
        """
        Calculate agricultural weather indices based on current conditions.

        Args:
            location: Location string

        Returns:
            Dictionary with agricultural weather indices
        """
        current_weather = self.get_current_weather(location)
        if not current_weather:
            return None

        temp = current_weather.get('temperature', 20)
        humidity = current_weather.get('humidity', 60)
        wind_speed = current_weather.get('wind_speed', 10)

        # Calculate various agricultural indices

        # Heat Index (simplified)
        heat_index = temp + 0.5 * (humidity - 50)

        # Growing Degree Days (base temperature 10°C)
        gdd = max(0, temp - 10)

        # Evapotranspiration estimate (simplified Penman equation)
        et_estimate = max(0, (temp - 5) * (1 - humidity/100) * 0.1)

        # Wind chill factor
        if temp < 10 and wind_speed > 5:
            wind_chill = 13.12 + 0.6215 * temp - 11.37 * (wind_speed ** 0.16) + 0.3965 * temp * (wind_speed ** 0.16)
        else:
            wind_chill = temp

        return {
            'heat_index': round(heat_index, 1),
            'growing_degree_days': round(gdd, 1),
            'evapotranspiration': round(et_estimate, 2),
            'wind_chill': round(wind_chill, 1),
            'agricultural_suitability': self._calculate_agricultural_suitability(temp, humidity, wind_speed)
        }

    def _calculate_agricultural_suitability(self, temp: float, humidity: float, wind_speed: float) -> str:
        """Calculate overall agricultural suitability rating."""
        score = 0

        # Temperature scoring
        if 15 <= temp <= 25:
            score += 3
        elif 10 <= temp < 15 or 25 < temp <= 30:
            score += 2
        elif 5 <= temp < 10 or 30 < temp <= 35:
            score += 1

        # Humidity scoring
        if 50 <= humidity <= 70:
            score += 3
        elif 40 <= humidity < 50 or 70 < humidity <= 80:
            score += 2
        elif 30 <= humidity < 40 or 80 < humidity <= 90:
            score += 1

        # Wind scoring
        if 5 <= wind_speed <= 15:
            score += 2
        elif wind_speed < 5 or 15 < wind_speed <= 25:
            score += 1

        # Convert score to rating
        if score >= 7:
            return "Excellent"
        elif score >= 5:
            return "Good"
        elif score >= 3:
            return "Fair"
        else:
            return "Poor"
