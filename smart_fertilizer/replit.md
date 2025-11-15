# Smart Fertilizer - African Agriculture

## Overview

This is a comprehensive Smart Fertilizer application designed to provide intelligent fertilizer recommendations for sustainable African agriculture. The system combines advanced soil analysis, crop-specific recommendations, regional adaptation, weather integration, and IoT sensor support to optimize agricultural productivity across African farming communities.

The application is built as a hybrid system supporting both web-based (Streamlit) and API-based (FastAPI) interfaces, with comprehensive data analysis capabilities and multi-format export options.

## System Architecture

### Frontend Architecture
- **Primary Interface**: Streamlit web application (`app.py`)
- **Multi-language Support**: English, French, Swahili, Hausa, Amharic through translation system
- **Interactive UI Components**: Region selector, soil analysis forms, crop selection, IoT dashboards
- **Visualization**: Plotly charts for data analysis and recommendations display
- **Export Capabilities**: PDF reports, Excel, CSV, JSON, XML formats

### Backend Architecture
- **Core Engine**: `SmartFertilizerEngine` - Main recommendation system
- **API Layer**: FastAPI REST API for programmatic access
- **Knowledge Base**: `AgronomicKnowledgeBase` with STCR (Soil Test Crop Response) methodology
- **Optimization Engine**: `FertilizerOptimizer` for cost-effective fertilizer selection
- **Regional Context**: Country-specific data and pricing information

### Data Processing Pipeline
1. **Soil Analysis Input** → Nutrient deficiency calculation
2. **Crop Selection** → Growth stage requirements mapping
3. **Regional Context** → Local pricing and availability integration
4. **STCR Methodology** → Scientific fertilizer requirement calculation
5. **Optimization** → Cost-effective fertilizer combination selection
6. **Scheduling** → Growth stage-based application timing

## Key Components

### Core Components
- **Smart Fertilizer Engine** (`core/smart_fertilizer_engine.py`): Main recommendation system
- **Agronomic Knowledge Base** (`core/agronomic_knowledge_base.py`): STCR coefficients and crop nutrition data
- **Fertilizer Optimizer** (`core/fertilizer_optimizer.py`): Cost optimization algorithms
- **Regional Context Manager** (`core/regional_context.py`): Country and region-specific data

### Data Components
- **Crop Profiles** (`data/crop_profiles.json`): Comprehensive crop nutrition requirements
- **Regional Prices** (`data/regional_prices.json`): Fertilizer pricing by country
- **Soil Samples** (`data/soil_samples.csv`): Training data for model validation
- **Yield Training Data** (`data/yield_training_data.csv`): Historical yield performance data

### UI Components
- **Smart UI** (`ui/smart_ui.py`): Main Streamlit interface controller
- **Translations** (`ui/translations.py`): Multi-language support system
- **Region Selector** (`regions/region_selector.py`): Interactive region selection interface

### Integration Components
- **Weather Client** (`weather/weather_client.py`): Weather data integration for application timing
- **IoT Simulator** (`weather/iot_simulator.py`): Agricultural sensor data simulation and monitoring
- **Export Utilities** (`exports/export_utils.py`): Multi-format data export capabilities
- **PDF Generator** (`exports/pdf_generator.py`): Professional report generation

## Data Flow

1. **User Input Collection**:
   - Region selection (country, agro-ecological zone)
   - Soil analysis parameters (pH, nutrients, texture)
   - Crop selection (type, variety, planting season)
   - Farm details (area, target yield)

2. **Data Processing**:
   - Soil nutrient deficiency calculation using critical levels
   - Crop nutrient requirements based on target yield
   - Regional price and availability integration
   - Weather data integration for timing optimization

3. **Recommendation Generation**:
   - STCR methodology application for scientific accuracy
   - Fertilizer optimization for cost-effectiveness
   - Application scheduling based on crop growth stages
   - Risk assessment and mitigation strategies

4. **Output Generation**:
   - Comprehensive fertilizer recommendations
   - Cost-benefit analysis with ROI calculations
   - Application timing schedules with weather considerations
   - Professional PDF reports and data exports

## External Dependencies

### Core Dependencies
- **Streamlit**: Web application framework
- **FastAPI**: REST API framework
- **Pandas/NumPy**: Data processing and analysis
- **Scikit-learn**: Machine learning algorithms
- **SciPy**: Scientific computing and optimization
- **Plotly**: Interactive data visualization
- **ReportLab**: PDF report generation
- **OpenPyXL**: Excel file handling

### Data Sources
- **FAO**: Food and Agriculture Organization datasets
- **ESDAC**: European Soil Data Centre
- **ICAR/ICRISAT**: Indian Council of Agricultural Research / International Crops Research Institute
- **NOAA/CHIRPS**: Climate and precipitation data
- **OpenWeatherMap API**: Weather data integration

### Optional Integrations
- **Weather APIs**: Real-time weather data for application timing
- **IoT Sensors**: Agricultural monitoring equipment
- **SMS/WhatsApp APIs**: Farmer notification systems
- **Payment Gateways**: E-commerce integration for fertilizer purchasing

## Deployment Strategy

### Current Configuration
- **Platform**: Replit with Nix environment
- **Runtime**: Python 3.11
- **Port**: 5000 (Streamlit application)
- **Deployment Target**: Autoscale
- **Dependencies**: Managed via pyproject.toml and uv.lock

### Production Considerations
- **Containerization**: Docker support recommended for production
- **Database**: Currently file-based, can be extended to PostgreSQL with Drizzle ORM
- **Caching**: Redis for API response caching
- **CDN**: Asset delivery optimization
- **Load Balancing**: For high-traffic scenarios
- **Monitoring**: Application performance and error tracking

### Scalability Features
- **Modular Architecture**: Easy component replacement and scaling
- **API-First Design**: Supports multiple frontend implementations
- **Data Pipeline**: Batch processing capabilities for large datasets
- **Regional Deployment**: Country-specific instances for latency optimization

## Changelog

- June 26, 2025. Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.