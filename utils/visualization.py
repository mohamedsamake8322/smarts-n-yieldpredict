import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

def create_overview_charts(data: pd.DataFrame) -> Dict[str, go.Figure]:
    """
    Create overview charts for agricultural data dashboard.
    
    Args:
        data: Agricultural data DataFrame
        
    Returns:
        Dictionary of plotly figures
    """
    charts = {}
    
    if data.empty:
        # Return empty charts if no data
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        return {'empty': empty_fig}
    
    # Yield distribution chart
    if 'yield' in data.columns:
        charts['yield_distribution'] = px.histogram(
            data, 
            x='yield',
            nbins=20,
            title="Yield Distribution",
            labels={'yield': 'Yield (tons/ha)', 'count': 'Frequency'}
        )
        charts['yield_distribution'].update_layout(
            showlegend=False,
            height=400
        )
    
    # Crop type distribution
    if 'crop_type' in data.columns:
        crop_counts = data['crop_type'].value_counts()
        charts['crop_distribution'] = px.pie(
            values=crop_counts.values,
            names=crop_counts.index,
            title="Crop Type Distribution"
        )
        charts['crop_distribution'].update_layout(height=400)
    
    # Yield by crop type
    if 'crop_type' in data.columns and 'yield' in data.columns:
        charts['yield_by_crop'] = px.box(
            data,
            x='crop_type',
            y='yield',
            title="Yield Distribution by Crop Type",
            labels={'yield': 'Yield (tons/ha)', 'crop_type': 'Crop Type'}
        )
        charts['yield_by_crop'].update_layout(height=400)
    
    # Area vs Yield scatter plot
    if 'area' in data.columns and 'yield' in data.columns:
        charts['area_yield_scatter'] = px.scatter(
            data,
            x='area',
            y='yield',
            color='crop_type' if 'crop_type' in data.columns else None,
            title="Area vs Yield Relationship",
            labels={'area': 'Area (hectares)', 'yield': 'Yield (tons/ha)'}
        )
        charts['area_yield_scatter'].update_layout(height=400)
    
    return charts


def create_trend_analysis(data: pd.DataFrame, date_column: str = 'date', 
                         value_column: str = 'yield') -> go.Figure:
    """
    Create trend analysis chart for time series data.
    
    Args:
        data: DataFrame with time series data
        date_column: Name of date column
        value_column: Name of value column to analyze
        
    Returns:
        Plotly figure with trend analysis
    """
    if data.empty or date_column not in data.columns or value_column not in data.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for trend analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        return fig
    
    # Ensure date column is datetime
    data = data.copy()
    data[date_column] = pd.to_datetime(data[date_column])
    
    # Group by date and calculate mean
    if len(data) > 1:
        daily_avg = data.groupby(data[date_column].dt.date)[value_column].mean().reset_index()
        daily_avg.columns = [date_column, value_column]
    else:
        daily_avg = data[[date_column, value_column]]
    
    # Create trend line
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=daily_avg[date_column],
        y=daily_avg[value_column],
        mode='lines+markers',
        name=f'{value_column.title()} Trend',
        line=dict(width=2)
    ))
    
    # Add trend line if enough data points
    if len(daily_avg) > 2:
        # Simple linear trend
        x_numeric = np.arange(len(daily_avg))
        z = np.polyfit(x_numeric, daily_avg[value_column], 1)
        p = np.poly1d(z)
        
        fig.add_trace(go.Scatter(
            x=daily_avg[date_column],
            y=p(x_numeric),
            mode='lines',
            name='Trend Line',
            line=dict(dash='dash', color='red')
        ))
    
    fig.update_layout(
        title=f'{value_column.title()} Trend Over Time',
        xaxis_title='Date',
        yaxis_title=value_column.title(),
        height=500,
        hovermode='x unified'
    )
    
    return fig


def create_weather_charts(weather_data: pd.DataFrame) -> Dict[str, go.Figure]:
    """
    Create weather-related charts for agricultural analysis.
    
    Args:
        weather_data: Weather data DataFrame
        
    Returns:
        Dictionary of weather charts
    """
    charts = {}
    
    if weather_data.empty:
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text="No weather data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        return {'empty': empty_fig}
    
    # Temperature and rainfall chart
    if 'date' in weather_data.columns:
        weather_data['date'] = pd.to_datetime(weather_data['date'])
        
        if 'temperature' in weather_data.columns and 'rainfall' in weather_data.columns:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Temperature Trend', 'Rainfall Pattern'),
                vertical_spacing=0.1
            )
            
            # Temperature
            fig.add_trace(
                go.Scatter(
                    x=weather_data['date'],
                    y=weather_data['temperature'],
                    mode='lines',
                    name='Temperature (°C)',
                    line=dict(color='red')
                ),
                row=1, col=1
            )
            
            # Rainfall
            fig.add_trace(
                go.Bar(
                    x=weather_data['date'],
                    y=weather_data['rainfall'],
                    name='Rainfall (mm)',
                    marker_color='blue'
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                title="Weather Conditions Over Time",
                height=600,
                showlegend=True
            )
            
            charts['weather_overview'] = fig
    
    # Humidity vs Temperature scatter
    if 'temperature' in weather_data.columns and 'humidity' in weather_data.columns:
        charts['temp_humidity'] = px.scatter(
            weather_data,
            x='temperature',
            y='humidity',
            title="Temperature vs Humidity Relationship",
            labels={'temperature': 'Temperature (°C)', 'humidity': 'Humidity (%)'}
        )
        charts['temp_humidity'].update_layout(height=400)
    
    return charts


def create_soil_monitoring_charts(soil_data: pd.DataFrame, field_id: str = None) -> Dict[str, go.Figure]:
    """
    Create soil monitoring charts.
    
    Args:
        soil_data: Soil monitoring data DataFrame
        field_id: Specific field ID to filter data
        
    Returns:
        Dictionary of soil monitoring charts
    """
    charts = {}
    
    if soil_data.empty:
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text="No soil data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        return {'empty': empty_fig}
    
    # Filter by field if specified
    if field_id and 'field_id' in soil_data.columns:
        soil_data = soil_data[soil_data['field_id'] == field_id]
    
    # pH trend over time
    if 'date' in soil_data.columns and 'ph' in soil_data.columns:
        soil_data['date'] = pd.to_datetime(soil_data['date'])
        
        charts['ph_trend'] = px.line(
            soil_data.sort_values('date'),
            x='date',
            y='ph',
            title="Soil pH Trend",
            labels={'ph': 'pH Level', 'date': 'Date'}
        )
        
        # Add optimal pH range
        charts['ph_trend'].add_hline(y=6.0, line_dash="dash", line_color="green", 
                                    annotation_text="Optimal Min (6.0)")
        charts['ph_trend'].add_hline(y=7.5, line_dash="dash", line_color="green", 
                                    annotation_text="Optimal Max (7.5)")
        charts['ph_trend'].update_layout(height=400)
    
    # Nutrient levels
    nutrient_cols = ['nitrogen', 'phosphorus', 'potassium']
    available_nutrients = [col for col in nutrient_cols if col in soil_data.columns]
    
    if available_nutrients and 'date' in soil_data.columns:
        fig = go.Figure()
        
        for nutrient in available_nutrients:
            fig.add_trace(go.Scatter(
                x=soil_data['date'],
                y=soil_data[nutrient],
                mode='lines+markers',
                name=nutrient.title()
            ))
        
        fig.update_layout(
            title="Soil Nutrient Levels Over Time",
            xaxis_title="Date",
            yaxis_title="Concentration (ppm)",
            height=400
        )
        
        charts['nutrient_trends'] = fig
    
    # Moisture levels
    if 'date' in soil_data.columns and 'moisture' in soil_data.columns:
        charts['moisture_trend'] = px.area(
            soil_data.sort_values('date'),
            x='date',
            y='moisture',
            title="Soil Moisture Levels",
            labels={'moisture': 'Moisture Content (%)', 'date': 'Date'}
        )
        
        # Add optimal moisture range
        charts['moisture_trend'].add_hline(y=40, line_dash="dash", line_color="green", 
                                         annotation_text="Optimal Min (40%)")
        charts['moisture_trend'].add_hline(y=70, line_dash="dash", line_color="green", 
                                         annotation_text="Optimal Max (70%)")
        charts['moisture_trend'].update_layout(height=400)
    
    return charts


def create_yield_prediction_charts(actual_yields: List[float], predicted_yields: List[float],
                                 feature_importance: Dict[str, float] = None) -> Dict[str, go.Figure]:
    """
    Create charts for yield prediction analysis.
    
    Args:
        actual_yields: List of actual yield values
        predicted_yields: List of predicted yield values
        feature_importance: Dictionary of feature names and importance scores
        
    Returns:
        Dictionary of prediction analysis charts
    """
    charts = {}
    
    if not actual_yields or not predicted_yields:
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text="No prediction data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        return {'empty': empty_fig}
    
    # Actual vs Predicted scatter plot
    min_val = min(min(actual_yields), min(predicted_yields))
    max_val = max(max(actual_yields), max(predicted_yields))
    
    charts['actual_vs_predicted'] = go.Figure()
    
    charts['actual_vs_predicted'].add_trace(go.Scatter(
        x=actual_yields,
        y=predicted_yields,
        mode='markers',
        name='Predictions',
        marker=dict(size=8, opacity=0.7)
    ))
    
    # Add perfect prediction line
    charts['actual_vs_predicted'].add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(dash='dash', color='red')
    ))
    
    charts['actual_vs_predicted'].update_layout(
        title="Actual vs Predicted Yields",
        xaxis_title="Actual Yield (tons/ha)",
        yaxis_title="Predicted Yield (tons/ha)",
        height=400
    )
    
    # Residuals plot
    residuals = np.array(actual_yields) - np.array(predicted_yields)
    
    charts['residuals'] = go.Figure()
    charts['residuals'].add_trace(go.Scatter(
        x=predicted_yields,
        y=residuals,
        mode='markers',
        name='Residuals',
        marker=dict(size=8, opacity=0.7)
    ))
    
    charts['residuals'].add_hline(y=0, line_dash="solid", line_color="red")
    
    charts['residuals'].update_layout(
        title="Residuals Plot",
        xaxis_title="Predicted Yield (tons/ha)",
        yaxis_title="Residuals",
        height=400
    )
    
    # Feature importance chart
    if feature_importance:
        features = list(feature_importance.keys())
        importance = list(feature_importance.values())
        
        # Sort by importance
        sorted_data = sorted(zip(features, importance), key=lambda x: x[1], reverse=True)
        features, importance = zip(*sorted_data)
        
        charts['feature_importance'] = go.Figure(go.Bar(
            x=importance,
            y=features,
            orientation='h'
        ))
        
        charts['feature_importance'].update_layout(
            title="Feature Importance in Yield Prediction",
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=400
        )
    
    return charts


def create_comparative_analysis_charts(data: pd.DataFrame, group_column: str, 
                                     value_column: str) -> Dict[str, go.Figure]:
    """
    Create comparative analysis charts for different groups.
    
    Args:
        data: DataFrame with data to compare
        group_column: Column to group by
        value_column: Column with values to compare
        
    Returns:
        Dictionary of comparative charts
    """
    charts = {}
    
    if data.empty or group_column not in data.columns or value_column not in data.columns:
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text="Insufficient data for comparison",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        return {'empty': empty_fig}
    
    # Box plot comparison
    charts['box_comparison'] = px.box(
        data,
        x=group_column,
        y=value_column,
        title=f"{value_column.title()} Comparison by {group_column.title()}",
        labels={value_column: value_column.title(), group_column: group_column.title()}
    )
    charts['box_comparison'].update_layout(height=400)
    
    # Violin plot comparison
    charts['violin_comparison'] = px.violin(
        data,
        x=group_column,
        y=value_column,
        box=True,
        title=f"{value_column.title()} Distribution by {group_column.title()}",
        labels={value_column: value_column.title(), group_column: group_column.title()}
    )
    charts['violin_comparison'].update_layout(height=400)
    
    # Statistical summary table
    summary_stats = data.groupby(group_column)[value_column].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(2)
    
    # Convert to table format for display
    charts['summary_table'] = go.Figure(data=[go.Table(
        header=dict(
            values=['Group'] + list(summary_stats.columns),
            fill_color='paleturquoise',
            align='left'
        ),
        cells=dict(
            values=[summary_stats.index] + [summary_stats[col] for col in summary_stats.columns],
            fill_color='lavender',
            align='left'
        )
    )])
    
    charts['summary_table'].update_layout(
        title=f"Statistical Summary: {value_column.title()} by {group_column.title()}",
        height=400
    )
    
    return charts


def create_correlation_heatmap(data: pd.DataFrame, title: str = "Correlation Matrix") -> go.Figure:
    """
    Create correlation heatmap for numeric columns.
    
    Args:
        data: DataFrame with numeric data
        title: Title for the heatmap
        
    Returns:
        Plotly heatmap figure
    """
    # Select only numeric columns
    numeric_data = data.select_dtypes(include=[np.number])
    
    if numeric_data.empty or numeric_data.shape[1] < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient numeric data for correlation analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        return fig
    
    # Calculate correlation matrix
    corr_matrix = numeric_data.corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu_r',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=title,
        height=500,
        width=500
    )
    
    return fig


def create_gauge_chart(value: float, title: str, min_val: float = 0, 
                      max_val: float = 100, unit: str = "") -> go.Figure:
    """
    Create a gauge chart for displaying single metric values.
    
    Args:
        value: Current value to display
        title: Title for the gauge
        min_val: Minimum value for the gauge
        max_val: Maximum value for the gauge
        unit: Unit of measurement
        
    Returns:
        Plotly gauge figure
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"{title} ({unit})" if unit else title},
        gauge={
            'axis': {'range': [None, max_val]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [min_val, max_val * 0.5], 'color': "lightgray"},
                {'range': [max_val * 0.5, max_val * 0.8], 'color': "yellow"},
                {'range': [max_val * 0.8, max_val], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_val * 0.9
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=20))
    
    return fig


def create_time_series_forecast_chart(historical_data: pd.DataFrame, 
                                    forecast_data: pd.DataFrame = None,
                                    date_col: str = 'date', 
                                    value_col: str = 'value') -> go.Figure:
    """
    Create a time series chart with optional forecast data.
    
    Args:
        historical_data: Historical time series data
        forecast_data: Future forecast data (optional)
        date_col: Name of date column
        value_col: Name of value column
        
    Returns:
        Plotly time series figure
    """
    fig = go.Figure()
    
    if not historical_data.empty:
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical_data[date_col],
            y=historical_data[value_col],
            mode='lines+markers',
            name='Historical Data',
            line=dict(color='blue')
        ))
    
    if forecast_data is not None and not forecast_data.empty:
        # Forecast data
        fig.add_trace(go.Scatter(
            x=forecast_data[date_col],
            y=forecast_data[value_col],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red', dash='dash')
        ))
    
    fig.update_layout(
        title="Time Series with Forecast",
        xaxis_title="Date",
        yaxis_title=value_col.title(),
        height=500,
        hovermode='x unified'
    )
    
    return fig
