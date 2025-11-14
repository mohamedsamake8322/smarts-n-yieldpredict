/**
 * Interface React ultra-réactive pour l'analyse agricole
 * Dashboard interactif avec visualisations temps réel
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
} from 'chart.js';
import { Line, Bar, Pie, Scatter } from 'react-chartjs-2';
import axios from 'axios';

// Configuration Chart.js
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

// Configuration axios avec intercepteurs pour performance
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 5000,
  headers: {
    'Content-Type': 'application/json',
  }
});

// Hook personnalisé pour les données temps réel
const useRealtimeData = (endpoint, interval = 30000) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchData = useCallback(async () => {
    try {
      const response = await apiClient.get(endpoint);
      setData(response.data);
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [endpoint]);

  useEffect(() => {
    fetchData();
    const intervalId = setInterval(fetchData, interval);
    return () => clearInterval(intervalId);
  }, [fetchData, interval]);

  return { data, loading, error, refetch: fetchData };
};

// Composant MetricCard optimisé
const MetricCard = React.memo(({ title, value, unit, trend, color = 'blue' }) => (
  <div className={`bg-white rounded-lg shadow-md p-6 border-l-4 border-${color}-500`}>
    <div className="flex items-center justify-between">
      <div>
        <p className="text-sm font-medium text-gray-600 uppercase tracking-wide">
          {title}
        </p>
        <p className="text-2xl font-bold text-gray-900">
          {value} {unit && <span className="text-sm text-gray-500">{unit}</span>}
        </p>
      </div>
      {trend && (
        <div className={`text-sm ${trend > 0 ? 'text-green-600' : 'text-red-600'}`}>
          {trend > 0 ? '↗' : '↘'} {Math.abs(trend)}%
        </div>
      )}
    </div>
  </div>
));

// Composant YieldPredictionForm
const YieldPredictionForm = ({ onPredictionResult }) => {
  const [formData, setFormData] = useState({
    crop_type: 'wheat',
    area: 10,
    soil_ph: 6.5,
    soil_nitrogen: 40,
    temperature: 22,
    rainfall: 500,
    humidity: 65,
    sunlight: 10
  });
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    
    try {
      const response = await apiClient.post('/predict/yield', formData);
      onPredictionResult(response.data);
    } catch (error) {
      console.error('Erreur prédiction:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  return (
    <form onSubmit={handleSubmit} className="bg-white rounded-lg shadow-md p-6">
      <h3 className="text-lg font-semibold mb-4 text-gray-800">
        Prédiction de Rendement IA
      </h3>
      
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Type de Culture
          </label>
          <select
            value={formData.crop_type}
            onChange={(e) => handleInputChange('crop_type', e.target.value)}
            className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
          >
            <option value="wheat">Blé</option>
            <option value="corn">Maïs</option>
            <option value="rice">Riz</option>
            <option value="soybeans">Soja</option>
            <option value="barley">Orge</option>
            <option value="cotton">Coton</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Surface (hectares)
          </label>
          <input
            type="number"
            value={formData.area}
            onChange={(e) => handleInputChange('area', parseFloat(e.target.value))}
            className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
            min="0.1"
            step="0.1"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            pH du Sol
          </label>
          <input
            type="number"
            value={formData.soil_ph}
            onChange={(e) => handleInputChange('soil_ph', parseFloat(e.target.value))}
            className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
            min="0"
            max="14"
            step="0.1"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Azote (ppm)
          </label>
          <input
            type="number"
            value={formData.soil_nitrogen}
            onChange={(e) => handleInputChange('soil_nitrogen', parseFloat(e.target.value))}
            className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
            min="0"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Température (°C)
          </label>
          <input
            type="number"
            value={formData.temperature}
            onChange={(e) => handleInputChange('temperature', parseFloat(e.target.value))}
            className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
            step="0.1"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Précipitations (mm)
          </label>
          <input
            type="number"
            value={formData.rainfall}
            onChange={(e) => handleInputChange('rainfall', parseFloat(e.target.value))}
            className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
            min="0"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Humidité (%)
          </label>
          <input
            type="number"
            value={formData.humidity}
            onChange={(e) => handleInputChange('humidity', parseFloat(e.target.value))}
            className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
            min="0"
            max="100"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Heures de Soleil/Jour
          </label>
          <input
            type="number"
            value={formData.sunlight}
            onChange={(e) => handleInputChange('sunlight', parseFloat(e.target.value))}
            className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
            min="0"
            max="24"
          />
        </div>
      </div>

      <button
        type="submit"
        disabled={loading}
        className="w-full mt-6 bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 disabled:opacity-50 transition-colors"
      >
        {loading ? 'Prédiction en cours...' : 'Générer Prédiction IA'}
      </button>
    </form>
  );
};

// Composant WeatherWidget
const WeatherWidget = ({ location = 'Paris,FR' }) => {
  const { data: weather, loading } = useRealtimeData(`/weather/current/${location}`);

  if (loading) return <div className="bg-white rounded-lg shadow-md p-6">Chargement météo...</div>;

  if (!weather?.weather) return <div className="bg-white rounded-lg shadow-md p-6">Données météo indisponibles</div>;

  const weatherData = weather.weather;

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h3 className="text-lg font-semibold mb-4 text-gray-800 flex items-center">
        <span className="mr-2">🌤️</span>
        Météo Temps Réel - {weatherData.location}
      </h3>
      
      <div className="grid grid-cols-2 gap-4">
        <div className="text-center">
          <p className="text-3xl font-bold text-blue-600">{weatherData.temperature}°C</p>
          <p className="text-sm text-gray-600">Température</p>
        </div>
        
        <div className="text-center">
          <p className="text-2xl font-bold text-green-600">{weatherData.humidity}%</p>
          <p className="text-sm text-gray-600">Humidité</p>
        </div>
        
        <div className="text-center">
          <p className="text-xl font-bold text-gray-600">{weatherData.wind_speed} km/h</p>
          <p className="text-sm text-gray-600">Vent</p>
        </div>
        
        <div className="text-center">
          <p className="text-xl font-bold text-purple-600">{weatherData.pressure} hPa</p>
          <p className="text-sm text-gray-600">Pression</p>
        </div>
      </div>
      
      <div className="mt-4 p-3 bg-gray-50 rounded-md">
        <p className="text-sm text-gray-700 capitalize">{weatherData.description}</p>
        {weather.agricultural_indices && (
          <p className="text-xs text-green-600 mt-1">
            Indice agricole: {weather.agricultural_indices.agricultural_suitability}
          </p>
        )}
      </div>
    </div>
  );
};

// Composant principal Dashboard
const AgriculturalDashboard = () => {
  const [predictionResult, setPredictionResult] = useState(null);
  const [activeTab, setActiveTab] = useState('dashboard');
  
  // Données temps réel du dashboard
  const { data: analytics } = useRealtimeData('/analytics/dashboard');
  const { data: modelPerformance } = useRealtimeData('/models/performance');

  const tabs = [
    { id: 'dashboard', label: 'Dashboard', icon: '📊' },
    { id: 'prediction', label: 'Prédiction IA', icon: '🔮' },
    { id: 'weather', label: 'Météo', icon: '🌤️' },
    { id: 'analytics', label: 'Analytics', icon: '📈' }
  ];

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <h1 className="text-2xl font-bold text-gray-900">
              🌾 Agricultural Analytics Platform
            </h1>
            <div className="flex items-center space-x-4">
              <div className="text-sm text-gray-600">
                Dernière mise à jour: {new Date().toLocaleTimeString('fr-FR')}
              </div>
              <div className="h-2 w-2 bg-green-500 rounded-full animate-pulse"></div>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8">
            {tabs.map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <span className="mr-2">{tab.icon}</span>
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Contenu principal */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'dashboard' && (
          <div className="space-y-6">
            {/* Métriques principales */}
            {analytics && (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <MetricCard
                  title="Total Enregistrements"
                  value={analytics.total_records}
                  color="blue"
                />
                <MetricCard
                  title="Rendement Moyen"
                  value={analytics.average_yield}
                  unit="t/ha"
                  color="green"
                />
                <MetricCard
                  title="Prédictions Récentes"
                  value={analytics.recent_predictions}
                  color="purple"
                />
                <MetricCard
                  title="Statut Système"
                  value={analytics.system_status}
                  color="green"
                />
              </div>
            )}

            {/* Widgets */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <WeatherWidget />
              
              {modelPerformance && (
                <div className="bg-white rounded-lg shadow-md p-6">
                  <h3 className="text-lg font-semibold mb-4 text-gray-800">
                    Performance des Modèles IA
                  </h3>
                  <div className="space-y-3">
                    {Object.entries(modelPerformance.models || {}).map(([model, metrics]) => (
                      <div key={model} className="flex justify-between items-center p-3 bg-gray-50 rounded">
                        <span className="font-medium capitalize">{model.replace('_', ' ')}</span>
                        <span className="text-blue-600 font-bold">
                          R² {(metrics.r2_score * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'prediction' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <YieldPredictionForm onPredictionResult={setPredictionResult} />
            
            {predictionResult && (
              <div className="bg-white rounded-lg shadow-md p-6">
                <h3 className="text-lg font-semibold mb-4 text-gray-800">
                  Résultat de la Prédiction
                </h3>
                
                <div className="space-y-4">
                  <div className="p-4 bg-blue-50 rounded-lg">
                    <p className="text-2xl font-bold text-blue-600">
                      {predictionResult.prediction?.yield?.toFixed(2)} tonnes/ha
                    </p>
                    <p className="text-sm text-gray-600">Rendement prédit</p>
                  </div>
                  
                  <div className="p-4 bg-green-50 rounded-lg">
                    <p className="text-xl font-bold text-green-600">
                      {predictionResult.prediction?.total_production?.toFixed(1)} tonnes
                    </p>
                    <p className="text-sm text-gray-600">Production totale</p>
                  </div>
                  
                  <div className="p-4 bg-yellow-50 rounded-lg">
                    <p className="text-lg font-bold text-yellow-600">
                      {predictionResult.model_confidence}% confiance
                    </p>
                    <p className="text-sm text-gray-600">Fiabilité du modèle</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'weather' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <WeatherWidget location="Paris,FR" />
            <WeatherWidget location="Lyon,FR" />
            <WeatherWidget location="Marseille,FR" />
            <WeatherWidget location="Toulouse,FR" />
          </div>
        )}

        {activeTab === 'analytics' && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold mb-4 text-gray-800">
              Analytics Avancées
            </h3>
            <p className="text-gray-600">
              Interface d'analytics en cours de développement...
            </p>
          </div>
        )}
      </main>
    </div>
  );
};

export default AgriculturalDashboard;