import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import random

def validate_agricultural_data(data: pd.DataFrame, data_type: str) -> Dict:
    """
    Validate agricultural data based on type and expected structure.
    
    Args:
        data: DataFrame to validate
        data_type: Type of data ('Agricultural Data', 'Weather Data', 'Soil Data', 'Yield Records')
        
    Returns:
        Dictionary with validation results
    """
    validation_result = {
        'is_valid': True,
        'checks': {},
        'errors': [],
        'warnings': []
    }
    
    # Check if data is empty
    if data.empty:
        validation_result['is_valid'] = False
        validation_result['checks']['data_exists'] = {
            'passed': False,
            'message': 'No data found in the uploaded file'
        }
        return validation_result
    
    validation_result['checks']['data_exists'] = {
        'passed': True,
        'message': f'Data loaded successfully with {len(data)} rows'
    }
    
    # Define required columns for each data type
    required_columns = {
        'Agricultural Data': ['crop_type', 'yield', 'area'],
        'Weather Data': ['date', 'temperature', 'humidity', 'rainfall'],
        'Soil Data': ['field_id', 'ph', 'moisture'],
        'Yield Records': ['crop_type', 'yield', 'area', 'harvest_date']
    }
    
    optional_columns = {
        'Agricultural Data': ['date', 'profit', 'cost', 'temperature', 'rainfall', 'soil_ph'],
        'Weather Data': ['wind_speed', 'pressure', 'uv_index', 'visibility'],
        'Soil Data': ['date', 'nitrogen', 'phosphorus', 'potassium', 'organic_matter', 'temperature'],
        'Yield Records': ['planting_date', 'variety', 'treatment', 'quality_grade']
    }
    
    # Check for required columns
    if data_type in required_columns:
        required_cols = required_columns[data_type]
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            validation_result['is_valid'] = False
            validation_result['checks']['required_columns'] = {
                'passed': False,
                'message': f'Missing required columns: {", ".join(missing_cols)}'
            }
        else:
            validation_result['checks']['required_columns'] = {
                'passed': True,
                'message': 'All required columns present'
            }
    
    # Check data types and ranges
    if data_type == 'Agricultural Data':
        # Validate yield values
        if 'yield' in data.columns:
            invalid_yields = data[data['yield'] < 0]
            if not invalid_yields.empty:
                validation_result['warnings'].append(f'Found {len(invalid_yields)} records with negative yield values')
            
            extreme_yields = data[data['yield'] > 20]  # Assuming max reasonable yield is 20 tons/ha
            if not extreme_yields.empty:
                validation_result['warnings'].append(f'Found {len(extreme_yields)} records with unusually high yield values (>20 tons/ha)')
        
        # Validate area values
        if 'area' in data.columns:
            invalid_areas = data[data['area'] <= 0]
            if not invalid_areas.empty:
                validation_result['is_valid'] = False
                validation_result['checks']['area_validation'] = {
                    'passed': False,
                    'message': f'Found {len(invalid_areas)} records with invalid area values (≤0)'
                }
            else:
                validation_result['checks']['area_validation'] = {
                    'passed': True,
                    'message': 'Area values are valid'
                }
    
    elif data_type == 'Weather Data':
        # Validate temperature values
        if 'temperature' in data.columns:
            extreme_temps = data[(data['temperature'] < -50) | (data['temperature'] > 60)]
            if not extreme_temps.empty:
                validation_result['warnings'].append(f'Found {len(extreme_temps)} records with extreme temperature values')
        
        # Validate humidity values
        if 'humidity' in data.columns:
            invalid_humidity = data[(data['humidity'] < 0) | (data['humidity'] > 100)]
            if not invalid_humidity.empty:
                validation_result['is_valid'] = False
                validation_result['checks']['humidity_validation'] = {
                    'passed': False,
                    'message': f'Found {len(invalid_humidity)} records with invalid humidity values (not between 0-100%)'
                }
            else:
                validation_result['checks']['humidity_validation'] = {
                    'passed': True,
                    'message': 'Humidity values are valid'
                }
        
        # Validate rainfall values
        if 'rainfall' in data.columns:
            negative_rainfall = data[data['rainfall'] < 0]
            if not negative_rainfall.empty:
                validation_result['is_valid'] = False
                validation_result['checks']['rainfall_validation'] = {
                    'passed': False,
                    'message': f'Found {len(negative_rainfall)} records with negative rainfall values'
                }
            else:
                validation_result['checks']['rainfall_validation'] = {
                    'passed': True,
                    'message': 'Rainfall values are valid'
                }
    
    elif data_type == 'Soil Data':
        # Validate pH values
        if 'ph' in data.columns:
            invalid_ph = data[(data['ph'] < 0) | (data['ph'] > 14)]
            if not invalid_ph.empty:
                validation_result['is_valid'] = False
                validation_result['checks']['ph_validation'] = {
                    'passed': False,
                    'message': f'Found {len(invalid_ph)} records with invalid pH values (not between 0-14)'
                }
            else:
                validation_result['checks']['ph_validation'] = {
                    'passed': True,
                    'message': 'pH values are valid'
                }
        
        # Validate moisture values
        if 'moisture' in data.columns:
            invalid_moisture = data[(data['moisture'] < 0) | (data['moisture'] > 100)]
            if not invalid_moisture.empty:
                validation_result['is_valid'] = False
                validation_result['checks']['moisture_validation'] = {
                    'passed': False,
                    'message': f'Found {len(invalid_moisture)} records with invalid moisture values (not between 0-100%)'
                }
            else:
                validation_result['checks']['moisture_validation'] = {
                    'passed': True,
                    'message': 'Moisture values are valid'
                }
    
    # Check for date columns and validate format
    date_columns = ['date', 'planting_date', 'harvest_date']
    for col in date_columns:
        if col in data.columns:
            try:
                pd.to_datetime(data[col])
                validation_result['checks'][f'{col}_format'] = {
                    'passed': True,
                    'message': f'{col} format is valid'
                }
            except:
                validation_result['warnings'].append(f'{col} column contains invalid date formats')
    
    # Check for excessive missing values
    missing_percentage = (data.isnull().sum() / len(data)) * 100
    high_missing_cols = missing_percentage[missing_percentage > 50]
    
    if not high_missing_cols.empty:
        validation_result['warnings'].append(f'Columns with >50% missing values: {list(high_missing_cols.index)}')
    
    validation_result['checks']['data_quality'] = {
        'passed': True,
        'message': f'Data quality check completed. {len(validation_result["warnings"])} warnings found.'
    }
    
    return validation_result


def clean_agricultural_data(data: pd.DataFrame, method: str = 'all') -> pd.DataFrame:
    """
    Clean agricultural data by handling missing values, duplicates, and outliers.
    
    Args:
        data: DataFrame to clean
        method: Cleaning method ('missing_values', 'duplicates', 'outliers', 'all')
        
    Returns:
        Cleaned DataFrame
    """
    df = data.copy()
    
    if method in ['missing_values', 'all']:
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        # Fill numeric columns with median
        for col in numeric_columns:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical columns with mode
        for col in categorical_columns:
            if df[col].isnull().sum() > 0:
                mode_value = df[col].mode()
                if not mode_value.empty:
                    df[col].fillna(mode_value.iloc[0], inplace=True)
    
    if method in ['duplicates', 'all']:
        # Remove duplicates
        df = df.drop_duplicates()
    
    if method in ['outliers', 'all']:
        # Handle outliers using IQR method
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers instead of removing them
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    return df


def get_sample_agricultural_data(n_samples: int = 100) -> pd.DataFrame:
    """
    Generate sample agricultural data for demonstration purposes.
    Note: This should only be used when explicitly requested by the user.
    
    Args:
        n_samples: Number of sample records to generate
        
    Returns:
        DataFrame with sample agricultural data
    """
    # This function should only be called when the user explicitly requests sample data
    # In a production environment, this would load real data from authenticated sources
    
    np.random.seed(42)
    
    crop_types = ['Wheat', 'Corn', 'Rice', 'Soybeans', 'Barley', 'Cotton']
    
    # Generate base data
    data = {
        'crop_type': np.random.choice(crop_types, n_samples),
        'area': np.random.uniform(5, 50, n_samples),
        'temperature': np.random.normal(22, 8, n_samples),
        'rainfall': np.random.exponential(500, n_samples),
        'humidity': np.random.normal(65, 15, n_samples),
        'soil_ph': np.random.normal(6.5, 1, n_samples),
        'soil_nitrogen': np.random.normal(40, 15, n_samples),
        'soil_phosphorus': np.random.normal(25, 8, n_samples),
        'soil_potassium': np.random.normal(200, 50, n_samples),
        'fertilizer_used': np.random.uniform(100, 500, n_samples),
    }
    
    # Generate dates
    start_date = datetime.now() - timedelta(days=365)
    dates = [start_date + timedelta(days=x) for x in range(n_samples)]
    data['date'] = dates
    
    # Generate yield based on conditions (simplified model)
    base_yields = {'Wheat': 4.2, 'Corn': 7.5, 'Rice': 5.8, 'Soybeans': 3.1, 'Barley': 3.8, 'Cotton': 2.1}
    
    yields = []
    profits = []
    costs = []
    
    for i in range(n_samples):
        crop = data['crop_type'][i]
        base_yield = base_yields[crop]
        
        # Adjust yield based on conditions
        temp_factor = 1.0 if 15 <= data['temperature'][i] <= 25 else 0.8
        rain_factor = 1.0 if 400 <= data['rainfall'][i] <= 800 else 0.9
        ph_factor = 1.0 if 6.0 <= data['soil_ph'][i] <= 7.5 else 0.85
        
        # Add some randomness
        random_factor = np.random.normal(1.0, 0.2)
        
        yield_value = base_yield * temp_factor * rain_factor * ph_factor * random_factor
        yield_value = max(0.5, yield_value)  # Minimum yield
        
        yields.append(yield_value)
        
        # Calculate profit and cost (simplified)
        cost_per_ha = np.random.uniform(800, 1500)
        price_per_ton = np.random.uniform(200, 400)
        
        area = data['area'][i]
        total_cost = cost_per_ha * area
        total_revenue = yield_value * area * price_per_ton
        profit = total_revenue - total_cost
        
        costs.append(total_cost)
        profits.append(profit)
    
    data['yield'] = yields
    data['cost'] = costs
    data['profit'] = profits
    
    return pd.DataFrame(data)


def generate_soil_sample_data(n_samples: int = 200) -> pd.DataFrame:
    """
    Generate sample soil monitoring data.
    Note: This should only be used when explicitly requested by the user.
    
    Args:
        n_samples: Number of sample records to generate
        
    Returns:
        DataFrame with sample soil data
    """
    np.random.seed(42)
    
    field_ids = ['Field_1', 'Field_2', 'Field_3', 'Field_4', 'Field_5']
    
    # Generate timestamps over the last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    data = []
    
    for _ in range(n_samples):
        # Random field
        field_id = np.random.choice(field_ids)
        
        # Random date within range
        random_days = np.random.randint(0, 31)
        date = start_date + timedelta(days=random_days)
        
        # Generate soil parameters with some field-specific characteristics
        field_base_ph = {
            'Field_1': 6.2, 'Field_2': 6.8, 'Field_3': 6.0, 'Field_4': 7.1, 'Field_5': 6.5
        }
        
        record = {
            'field_id': field_id,
            'date': date,
            'ph': np.random.normal(field_base_ph[field_id], 0.3),
            'moisture': np.random.normal(55, 10),
            'temperature': np.random.normal(18, 5),
            'conductivity': np.random.normal(1.2, 0.3),
            'nitrogen': np.random.normal(45, 12),
            'phosphorus': np.random.normal(28, 8),
            'potassium': np.random.normal(180, 40),
            'organic_matter': np.random.normal(3.5, 1.0)
        }
        
        # Ensure realistic ranges
        record['ph'] = np.clip(record['ph'], 4.0, 9.0)
        record['moisture'] = np.clip(record['moisture'], 20, 90)
        record['temperature'] = np.clip(record['temperature'], 5, 35)
        record['conductivity'] = np.clip(record['conductivity'], 0.5, 3.0)
        record['nitrogen'] = np.clip(record['nitrogen'], 10, 80)
        record['phosphorus'] = np.clip(record['phosphorus'], 5, 50)
        record['potassium'] = np.clip(record['potassium'], 50, 300)
        record['organic_matter'] = np.clip(record['organic_matter'], 1.0, 8.0)
        
        data.append(record)
    
    return pd.DataFrame(data)


def calculate_crop_statistics(data: pd.DataFrame) -> Dict:
    """
    Calculate statistical summaries for crop data.
    
    Args:
        data: Agricultural data DataFrame
        
    Returns:
        Dictionary with statistical summaries
    """
    if data.empty:
        return {}
    
    stats = {}
    
    # Overall statistics
    if 'yield' in data.columns:
        stats['overall'] = {
            'total_records': len(data),
            'avg_yield': data['yield'].mean(),
            'median_yield': data['yield'].median(),
            'min_yield': data['yield'].min(),
            'max_yield': data['yield'].max(),
            'std_yield': data['yield'].std()
        }
    
    # Crop-specific statistics
    if 'crop_type' in data.columns and 'yield' in data.columns:
        crop_stats = data.groupby('crop_type')['yield'].agg([
            'count', 'mean', 'median', 'min', 'max', 'std'
        ]).to_dict('index')
        stats['by_crop'] = crop_stats
    
    # Area statistics
    if 'area' in data.columns:
        stats['area'] = {
            'total_area': data['area'].sum(),
            'avg_area': data['area'].mean(),
            'median_area': data['area'].median()
        }
    
    # Profit statistics
    if 'profit' in data.columns:
        stats['profit'] = {
            'total_profit': data['profit'].sum(),
            'avg_profit': data['profit'].mean(),
            'profit_per_hectare': data['profit'].sum() / data['area'].sum() if 'area' in data.columns else 0
        }
    
    return stats


def detect_data_quality_issues(data: pd.DataFrame) -> Dict:
    """
    Detect various data quality issues in agricultural data.
    
    Args:
        data: DataFrame to analyze
        
    Returns:
        Dictionary with detected issues
    """
    issues = {
        'missing_values': {},
        'duplicates': 0,
        'outliers': {},
        'inconsistencies': []
    }
    
    # Missing values
    missing_counts = data.isnull().sum()
    issues['missing_values'] = missing_counts[missing_counts > 0].to_dict()
    
    # Duplicates
    issues['duplicates'] = data.duplicated().sum()
    
    # Outliers (using IQR method)
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
        if not outliers.empty:
            issues['outliers'][col] = len(outliers)
    
    # Data inconsistencies
    if 'yield' in data.columns and 'area' in data.columns:
        zero_area_nonzero_yield = data[(data['area'] == 0) & (data['yield'] > 0)]
        if not zero_area_nonzero_yield.empty:
            issues['inconsistencies'].append('Records with zero area but non-zero yield')
    
    if 'planting_date' in data.columns and 'harvest_date' in data.columns:
        try:
            planting = pd.to_datetime(data['planting_date'])
            harvest = pd.to_datetime(data['harvest_date'])
            invalid_dates = data[harvest <= planting]
            if not invalid_dates.empty:
                issues['inconsistencies'].append('Harvest date before or equal to planting date')
        except:
            pass
    
    return issues


def prepare_data_for_analysis(data: pd.DataFrame, analysis_type: str = 'yield_prediction') -> pd.DataFrame:
    """
    Prepare data for specific types of analysis.
    
    Args:
        data: Raw data DataFrame
        analysis_type: Type of analysis ('yield_prediction', 'trend_analysis', 'comparison')
        
    Returns:
        Prepared DataFrame
    """
    df = data.copy()
    
    if analysis_type == 'yield_prediction':
        # Ensure required columns for yield prediction
        required_cols = ['crop_type', 'area', 'temperature', 'rainfall']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns for yield prediction: {missing_cols}")
        
        # Fill missing values with reasonable defaults
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown')
    
    elif analysis_type == 'trend_analysis':
        # Ensure date column exists and is properly formatted
        if 'date' not in df.columns:
            raise ValueError("Date column required for trend analysis")
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Add time-based features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['season'] = df['date'].dt.month.map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
    
    elif analysis_type == 'comparison':
        # Ensure we have grouping variables
        if 'crop_type' not in df.columns:
            raise ValueError("Crop type column required for comparison analysis")
        
        # Standardize crop type names
        df['crop_type'] = df['crop_type'].str.title()
    
    return df
