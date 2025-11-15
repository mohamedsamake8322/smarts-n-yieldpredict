"""
Multi-language translation support for Smart Fertilizer application
"""

from typing import Dict, Optional

class Translator:
    """
    Translation manager for multiple African languages
    """
    
    def __init__(self):
        self.translations = self._load_translations()
        self.default_language = 'en'
    
    def _load_translations(self) -> Dict:
        """Load translation dictionaries"""
        
        return {
            'en': {
                'app_title': 'Smart Fertilizer - African Agriculture',
                'app_subtitle': 'Intelligent Fertilizer Recommendations for Sustainable Farming',
                'region_selection': 'Region Selection',
                'crop_selection': 'Crop Selection',
                'soil_analysis': 'Soil Analysis',
                'farm_details': 'Farm Details',
                'recommendation': 'Recommendation',
                'cost_analysis': 'Cost Analysis',
                'application_schedule': 'Application Schedule',
                'weather_monitoring': 'Weather Monitoring',
                'iot_dashboard': 'IoT Dashboard',
                'data_analysis': 'Data Analysis',
                'help_support': 'Help & Support',
                
                # Form labels
                'select_region': 'Select your agricultural region',
                'select_crop': 'Select crop type',
                'crop_variety': 'Crop variety',
                'planting_season': 'Planting season',
                'growth_duration': 'Growth duration (days)',
                'farm_area': 'Farm area (hectares)',
                'target_yield': 'Target yield (tons/ha)',
                'soil_ph': 'Soil pH',
                'organic_matter': 'Organic matter (%)',
                'available_nitrogen': 'Available nitrogen (ppm)',
                'available_phosphorus': 'Available phosphorus (ppm)',
                'available_potassium': 'Available potassium (ppm)',
                'cec': 'Cation exchange capacity (cmol/kg)',
                'soil_texture': 'Soil texture',
                
                # Messages
                'generating_recommendation': 'Generating intelligent fertilizer recommendation...',
                'recommendation_complete': 'Fertilizer recommendation generated successfully!',
                'data_validation_error': 'Please check your input data and try again',
                'export_success': 'Report exported successfully!',
                'no_data_available': 'No data available',
                
                # Recommendations
                'high_yield_potential': 'High yield potential with proper fertilization',
                'moderate_yield_potential': 'Moderate yield potential',
                'low_yield_potential': 'Low yield potential - consider soil amendments',
                'cost_effective': 'Cost-effective fertilizer program',
                'high_cost_warning': 'High fertilizer costs - consider alternatives',
                
                # Weather
                'current_weather': 'Current Weather Conditions',
                'weather_forecast': 'Weather Forecast',
                'temperature': 'Temperature',
                'humidity': 'Humidity',
                'precipitation': 'Precipitation',
                'wind_speed': 'Wind Speed',
                'pressure': 'Atmospheric Pressure',
                
                # IoT
                'sensor_readings': 'Sensor Readings',
                'soil_moisture': 'Soil Moisture',
                'soil_temperature': 'Soil Temperature',
                'irrigation_need': 'Irrigation Need',
                'growing_conditions': 'Growing Conditions',
                'application_suitability': 'Application Suitability',
                
                # Risk factors
                'low_ph_warning': 'Low soil pH may limit nutrient availability',
                'high_ph_warning': 'High soil pH may cause micronutrient deficiencies',
                'low_organic_matter': 'Low organic matter - consider organic amendments',
                'drought_risk': 'Drought risk assessment',
                'leaching_risk': 'Nutrient leaching risk',
                
                # Export
                'download_pdf': 'Download PDF Report',
                'download_excel': 'Download Excel Report',
                'download_csv': 'Download CSV Data',
                'export_format': 'Export format',
                
                # Navigation
                'back': 'Back',
                'next': 'Next',
                'generate': 'Generate Recommendation',
                'refresh': 'Refresh',
                'submit': 'Submit',
                'cancel': 'Cancel',
                'save': 'Save',
                'edit': 'Edit',
                'delete': 'Delete',
                
                # Status
                'optimal': 'Optimal',
                'good': 'Good',
                'fair': 'Fair',
                'poor': 'Poor',
                'low': 'Low',
                'medium': 'Medium',
                'high': 'High',
                'very_high': 'Very High'
            },
            
            'fr': {
                'app_title': 'Engrais Intelligent - Agriculture Africaine',
                'app_subtitle': 'Recommandations Intelligentes d\'Engrais pour une Agriculture Durable',
                'region_selection': 'Sélection de Région',
                'crop_selection': 'Sélection de Culture',
                'soil_analysis': 'Analyse du Sol',
                'farm_details': 'Détails de la Ferme',
                'recommendation': 'Recommandation',
                'cost_analysis': 'Analyse des Coûts',
                'application_schedule': 'Calendrier d\'Application',
                'weather_monitoring': 'Surveillance Météorologique',
                'iot_dashboard': 'Tableau de Bord IoT',
                'data_analysis': 'Analyse des Données',
                'help_support': 'Aide et Support',
                
                # Form labels
                'select_region': 'Sélectionnez votre région agricole',
                'select_crop': 'Sélectionnez le type de culture',
                'crop_variety': 'Variété de culture',
                'planting_season': 'Saison de plantation',
                'growth_duration': 'Durée de croissance (jours)',
                'farm_area': 'Superficie de la ferme (hectares)',
                'target_yield': 'Rendement cible (tonnes/ha)',
                'soil_ph': 'pH du sol',
                'organic_matter': 'Matière organique (%)',
                'available_nitrogen': 'Azote disponible (ppm)',
                'available_phosphorus': 'Phosphore disponible (ppm)',
                'available_potassium': 'Potassium disponible (ppm)',
                'cec': 'Capacité d\'échange cationique (cmol/kg)',
                'soil_texture': 'Texture du sol',
                
                # Messages
                'generating_recommendation': 'Génération de recommandation intelligente d\'engrais...',
                'recommendation_complete': 'Recommandation d\'engrais générée avec succès!',
                'data_validation_error': 'Veuillez vérifier vos données d\'entrée et réessayer',
                'export_success': 'Rapport exporté avec succès!',
                'no_data_available': 'Aucune donnée disponible',
                
                # Recommendations
                'high_yield_potential': 'Potentiel de rendement élevé avec une fertilisation appropriée',
                'moderate_yield_potential': 'Potentiel de rendement modéré',
                'low_yield_potential': 'Faible potentiel de rendement - considérer les amendements du sol',
                'cost_effective': 'Programme d\'engrais rentable',
                'high_cost_warning': 'Coûts d\'engrais élevés - considérer des alternatives',
                
                # Weather
                'current_weather': 'Conditions Météorologiques Actuelles',
                'weather_forecast': 'Prévisions Météorologiques',
                'temperature': 'Température',
                'humidity': 'Humidité',
                'precipitation': 'Précipitations',
                'wind_speed': 'Vitesse du Vent',
                'pressure': 'Pression Atmosphérique',
                
                # Status
                'optimal': 'Optimal',
                'good': 'Bon',
                'fair': 'Passable',
                'poor': 'Pauvre',
                'low': 'Bas',
                'medium': 'Moyen',
                'high': 'Élevé',
                'very_high': 'Très Élevé'
            },
            
            'sw': {
                'app_title': 'Mbolea Mahiri - Kilimo cha Afrika',
                'app_subtitle': 'Mapendekezo ya Mbolea kwa Kilimo Endelevu',
                'region_selection': 'Uchaguzi wa Mkoa',
                'crop_selection': 'Uchaguzi wa Mazao',
                'soil_analysis': 'Uchambuzi wa Udongo',
                'farm_details': 'Maelezo ya Shamba',
                'recommendation': 'Pendekezo',
                'cost_analysis': 'Uchambuzi wa Gharama',
                'application_schedule': 'Ratiba ya Matumizi',
                'weather_monitoring': 'Ufuatiliaji wa Hali ya Hewa',
                'iot_dashboard': 'Dashibodi ya IoT',
                'data_analysis': 'Uchambuzi wa Data',
                'help_support': 'Msaada na Usaidizi',
                
                # Form labels
                'select_region': 'Chagua mkoa wako wa kilimo',
                'select_crop': 'Chagua aina ya mazao',
                'crop_variety': 'Aina ya mazao',
                'planting_season': 'Msimu wa kupanda',
                'growth_duration': 'Muda wa ukuaji (siku)',
                'farm_area': 'Eneo la shamba (hekta)',
                'target_yield': 'Mavuno lengwa (tani/ha)',
                'soil_ph': 'pH ya udongo',
                'organic_matter': 'Viumbe hai (%)',
                'available_nitrogen': 'Naitrojeni iliyopo (ppm)',
                'available_phosphorus': 'Fosforasi iliyopo (ppm)',
                'available_potassium': 'Potasiamu iliyopo (ppm)',
                'cec': 'Uwezo wa kubadilishana kationi (cmol/kg)',
                'soil_texture': 'Muundo wa udongo',
                
                # Messages
                'generating_recommendation': 'Kutengeneza pendekezo la mbolea mahiri...',
                'recommendation_complete': 'Pendekezo la mbolea limetengenezwa kwa mafanikio!',
                'data_validation_error': 'Tafadhali kagua data yako na ujaribu tena',
                'export_success': 'Ripoti imehamishwa kwa mafanikio!',
                'no_data_available': 'Hakuna data iliyopo',
                
                # Status
                'optimal': 'Bora Zaidi',
                'good': 'Nzuri',
                'fair': 'Wastani',
                'poor': 'Mbaya',
                'low': 'Chini',
                'medium': 'Wastani',
                'high': 'Juu',
                'very_high': 'Juu Sana'
            },
            
            'ha': {
                'app_title': 'Taki Mai Hankali - Noman Afrika',
                'app_subtitle': 'Shawarwarin Taki don Noma Mai Dorewa',
                'region_selection': 'Zabin Yanki',
                'crop_selection': 'Zabin Amfani',
                'soil_analysis': 'Binciken Kasa',
                'farm_details': 'Bayanan Gona',
                'recommendation': 'Shawara',
                'cost_analysis': 'Binciken Farashi',
                'application_schedule': 'Jadawalin Amfani',
                'weather_monitoring': 'Sa\'a idai Yanayi',
                'iot_dashboard': 'Dashboard IoT',
                'data_analysis': 'Binciken Bayanai',
                'help_support': 'Taimako da Goyon Baya',
                
                # Form labels
                'select_region': 'Zaɓi yankin noman ku',
                'select_crop': 'Zaɓi irin amfani',
                'crop_variety': 'Irin amfani',
                'planting_season': 'Lokacin shuki',
                'growth_duration': 'Tsawon girma (kwanaki)',
                'farm_area': 'Girman gona (hekta)',
                'target_yield': 'Amfanin da ake son samu (tan/ha)',
                'soil_ph': 'pH na kasa',
                'organic_matter': 'Abubuwan halitta (%)',
                'available_nitrogen': 'Nitrogen da ke samuwa (ppm)',
                'available_phosphorus': 'Phosphorus da ke samuwa (ppm)',
                'available_potassium': 'Potassium da ke samuwa (ppm)',
                'cec': 'Ikon musayar cation (cmol/kg)',
                'soil_texture': 'Yanayin kasa',
                
                # Status
                'optimal': 'Mafi Kyau',
                'good': 'Mai Kyau',
                'fair': 'Matsakaici',
                'poor': 'Mara Kyau',
                'low': 'Kasa',
                'medium': 'Matsakaici',
                'high': 'Sama',
                'very_high': 'Sama Sosai'
            },
            
            'am': {
                'app_title': 'ብልህ ዳግማ - የአፍሪካ ግብርና',
                'app_subtitle': 'ለዘላቂ ግብርና ብልህ የዳግማ መመሪያዎች',
                'region_selection': 'የክልል ምርጫ',
                'crop_selection': 'የሰብል ምርጫ',
                'soil_analysis': 'የአፈር ትንተና',
                'farm_details': 'የእርሻ ዝርዝሮች',
                'recommendation': 'ምክር',
                'cost_analysis': 'የወጪ ትንተና',
                'application_schedule': 'የአተገባበር ፕሮግራም',
                'weather_monitoring': 'የአየር ሁኔታ ክትትል',
                'iot_dashboard': 'IoT ዳሽቦርድ',
                'data_analysis': 'የመረጃ ትንተና',
                'help_support': 'እርዳታ እና ድጋፍ',
                
                # Form labels
                'select_region': 'የግብርና ክልልዎን ይምረጡ',
                'select_crop': 'የሰብል አይነት ይምረጡ',
                'crop_variety': 'የሰብል ዝርያ',
                'planting_season': 'የመዝራት ወቅት',
                'growth_duration': 'የእድገት ጊዜ (ቀናት)',
                'farm_area': 'የእርሻ ስፋት (ሄክታር)',
                'target_yield': 'ዒላማ ምርት (ቶን/ሄክታር)',
                'soil_ph': 'የአፈር pH',
                'organic_matter': 'ኦርጋኒክ ነገር (%)',
                'available_nitrogen': 'ያለ ናይትሮጅን (ppm)',
                'available_phosphorus': 'ያለ ፎስፈረስ (ppm)',
                'available_potassium': 'ያለ ፖታሲየም (ppm)',
                'cec': 'የካቲዮን ልውውጥ አቅም (cmol/kg)',
                'soil_texture': 'የአፈር ሸክላ',
                
                # Status
                'optimal': 'ጥሩ',
                'good': 'ጥሩ',
                'fair': 'መካከለኛ',
                'poor': 'ዝቅተኛ',
                'low': 'ዝቅተኛ',
                'medium': 'መካከለኛ',
                'high': 'ከፍተኛ',
                'very_high': 'በጣም ከፍተኛ'
            }
        }
    
    def get_text(self, key: str, language: str = None) -> str:
        """
        Get translated text for given key and language
        
        Args:
            key: Translation key
            language: Language code (en, fr, sw, ha, am)
            
        Returns:
            Translated text or key if translation not found
        """
        
        if language is None:
            language = self.default_language
        
        # Get language dictionary
        lang_dict = self.translations.get(language, self.translations[self.default_language])
        
        # Get translation or fallback to English
        text = lang_dict.get(key)
        if text is None and language != self.default_language:
            text = self.translations[self.default_language].get(key, key)
        
        return text or key
    
    def get_available_languages(self) -> Dict[str, str]:
        """Get available languages with their display names"""
        
        return {
            'en': 'English',
            'fr': 'Français',
            'sw': 'Kiswahili',
            'ha': 'Hausa',
            'am': 'አማርኛ (Amharic)'
        }
    
    def set_default_language(self, language: str):
        """Set default language"""
        
        if language in self.translations:
            self.default_language = language
    
    def translate_dict(self, data: Dict, language: str = None) -> Dict:
        """
        Translate a dictionary of keys
        
        Args:
            data: Dictionary with keys to translate
            language: Target language
            
        Returns:
            Dictionary with translated values
        """
        
        translated = {}
        for key, value in data.items():
            if isinstance(value, str) and key in self.translations.get(language or self.default_language, {}):
                translated[key] = self.get_text(key, language)
            else:
                translated[key] = value
        
        return translated
    
    def get_form_labels(self, language: str = None) -> Dict[str, str]:
        """Get all form labels for a language"""
        
        lang_dict = self.translations.get(language or self.default_language, {})
        
        form_labels = {}
        label_keys = [
            'select_region', 'select_crop', 'crop_variety', 'planting_season',
            'growth_duration', 'farm_area', 'target_yield', 'soil_ph',
            'organic_matter', 'available_nitrogen', 'available_phosphorus',
            'available_potassium', 'cec', 'soil_texture'
        ]
        
        for key in label_keys:
            form_labels[key] = lang_dict.get(key, key)
        
        return form_labels
    
    def get_status_labels(self, language: str = None) -> Dict[str, str]:
        """Get status labels for a language"""
        
        lang_dict = self.translations.get(language or self.default_language, {})
        
        status_labels = {}
        status_keys = ['optimal', 'good', 'fair', 'poor', 'low', 'medium', 'high', 'very_high']
        
        for key in status_keys:
            status_labels[key] = lang_dict.get(key, key)
        
        return status_labels
    
    def format_currency(self, amount: float, currency: str, language: str = None) -> str:
        """Format currency based on language and region"""
        
        if language == 'fr':
            return f"{amount:,.2f} {currency}"
        elif language == 'sw':
            return f"{currency} {amount:,.2f}"
        elif language == 'ha':
            return f"{currency} {amount:,.2f}"
        elif language == 'am':
            return f"{amount:,.2f} {currency}"
        else:  # English default
            return f"{currency} {amount:,.2f}"
    
    def format_number(self, number: float, decimals: int = 1, language: str = None) -> str:
        """Format numbers based on language"""
        
        if language in ['fr', 'sw', 'ha', 'am']:
            # Use comma as thousand separator, period as decimal
            return f"{number:,.{decimals}f}".replace(',', ' ').replace('.', ',').replace(' ', '.')
        else:  # English default
            return f"{number:,.{decimals}f}"
    
    def get_help_text(self, section: str, language: str = None) -> str:
        """Get help text for specific sections"""
        
        help_texts = {
            'en': {
                'soil_ph': 'Soil pH affects nutrient availability. Optimal range is 6.0-7.0 for most crops.',
                'organic_matter': 'Organic matter improves soil structure and nutrient retention. Target >2% for good soil health.',
                'cec': 'Cation Exchange Capacity indicates soil\'s ability to hold nutrients. Higher CEC means better nutrient retention.',
                'target_yield': 'Set realistic yield targets based on your farm\'s historical performance and local conditions.',
                'farm_area': 'Enter the total area to be fertilized in hectares (1 hectare = 10,000 square meters).',
                'application_timing': 'Apply fertilizers at the right growth stages for maximum efficiency and minimal loss.'
            },
            'fr': {
                'soil_ph': 'Le pH du sol affecte la disponibilité des nutriments. La gamme optimale est 6,0-7,0 pour la plupart des cultures.',
                'organic_matter': 'La matière organique améliore la structure du sol et la rétention des nutriments. Objectif >2% pour une bonne santé du sol.',
                'cec': 'La capacité d\'échange cationique indique la capacité du sol à retenir les nutriments.',
                'target_yield': 'Fixez des objectifs de rendement réalistes basés sur les performances historiques de votre ferme.',
                'farm_area': 'Entrez la superficie totale à fertiliser en hectares (1 hectare = 10 000 mètres carrés).',
                'application_timing': 'Appliquez les engrais aux bons stades de croissance pour une efficacité maximale.'
            }
        }
        
        lang_dict = help_texts.get(language or self.default_language, help_texts['en'])
        return lang_dict.get(section, '')


def get_translator() -> Translator:
    """Get global translator instance"""
    return Translator()
