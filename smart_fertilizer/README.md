# Smart Fertilizer - African Agriculture

🌾 **Intelligent Fertilizer Recommendation System for Sustainable African Agriculture**

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

Smart Fertilizer is a comprehensive, data-driven application that generates automated fertilization programs based on soil analysis and regional conditions across Africa. The system leverages advanced agronomic models, regional agricultural data, and machine learning to provide precise, cost-effective fertilizer recommendations for African farmers.

## 🌟 Key Features

### 🧪 Intelligent Soil Analysis
- Comprehensive soil parameter interpretation
- pH, organic matter, and nutrient availability assessment
- Soil health indexing and recommendations
- Support for laboratory and IoT sensor data

### 🌱 Crop-Specific Recommendations
- Multi-crop support (maize, rice, wheat, sorghum, millet, etc.)
- Regional variety recommendations
- Growth stage-based fertilizer scheduling
- Yield optimization strategies

### 🌍 Regional Adaptation
- Country-specific fertilizer prices and availability
- Local currency and unit conversions
- Climate-adapted timing recommendations
- Cultural and linguistic localization (English, French, Swahili, Hausa, Amharic)

### 📊 Advanced Analytics
- STCR (Soil Test Crop Response) methodology
- Machine learning yield prediction models
- Economic analysis and ROI calculations
- Risk assessment and mitigation strategies

### 🌤️ Weather Integration
- Current weather conditions monitoring
- 7-day agricultural forecasts
- Fertilizer application timing optimization
- Drought and leaching risk assessment

### 📡 IoT Sensor Support
- Real-time soil monitoring
- Agricultural condition dashboards
- Automated data collection
- Smart irrigation recommendations

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- 2GB RAM minimum (4GB recommended)
- Internet connection for weather data

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/smartfertilizer/smart-fertilizer.git
   cd smart-fertilizer
   ```

2. **Install dependencies**
   ```bash
   pip install streamlit fastapi uvicorn pandas numpy plotly
   pip install requests reportlab openpyxl scikit-learn scipy
   ```

3. **Run the application**
   ```bash
   # Web interface (recommended)
   python run_fertilizer.py

   # Or directly with Streamlit
   streamlit run app.py --server.port 5000
   ```

4. **Access the application**
   - Web Interface: http://localhost:5000
   - API Documentation: http://localhost:8000/docs (if running API mode)

### Alternative Launch Methods

```bash
# FastAPI backend only
python run_fertilizer.py --mode api

# Command-line interface
python run_fertilizer.py --mode cli

# Development mode (both web + API)
python run_fertilizer.py --mode dev

# Check dependencies
python run_fertilizer.py --check-deps
