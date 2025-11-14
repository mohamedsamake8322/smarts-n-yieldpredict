import json
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
class DiseaseDatabase:
    """
    Base de données complète des maladies agricoles
    Contient informations, symptômes, traitements et préventions
    """

    def __init__(self):
        self.diseases_data = self._initialize_disease_database()
        self.treatments_data = self._initialize_treatments_database()
        self.prevention_data = self._initialize_prevention_database()

    def _initialize_disease_database(self) -> Dict[str, Dict]:
        """
        Initialise la base de données des maladies
        """
        return {
            "Tomato_Late_blight": {
                "name": "Mildiou de la Tomate",
                "scientific_name": "Phytophthora infestans",
                "category": "Fongiques",
                "cause": "Oomycète pathogène",
                "description": "Maladie destructrice causant des taches brunes sur feuilles, tiges et fruits",
                "severity": "Élevée",
                "season": "Temps humide et frais",
                "affected_crops": ["Tomate", "Pomme de terre"],
                "symptoms": [
                    "Taches brunes irrégulières sur les feuilles",
                    "Flétrissement rapide des feuilles",
                    "Taches sombres sur les tiges",
                    "Pourriture des fruits",
                    "Duvet blanc sous les feuilles par temps humide",
                ],
                "favorable_conditions": [
                    "Température 15-20°C",
                    "Humidité élevée (>85%)",
                    "Temps pluvieux",
                    "Mauvaise circulation d'air",
                ],
            },
            "Tomato_Early_blight": {
                "name": "Alternariose de la Tomate",
                "scientific_name": "Alternaria solani",
                "category": "Fongiques",
                "cause": "Champignon pathogène",
                "description": "Maladie fongique causant des taches concentriques caractéristiques",
                "severity": "Modérée",
                "season": "Été chaud et humide",
                "affected_crops": ["Tomate", "Pomme de terre", "Aubergine"],
                "symptoms": [
                    "Taches circulaires brunes avec anneaux concentriques",
                    "Jaunissement et flétrissement des feuilles inférieures",
                    "Taches sur les tiges et pétioles",
                    "Pourriture des fruits près du pédoncule",
                ],
                "favorable_conditions": [
                    "Température 24-29°C",
                    "Alternance humidité/sécheresse",
                    "Stress hydrique",
                    "Plants affaiblis",
                ],
            },
            "Tomato_Bacterial_spot": {
                "name": "Tache Bactérienne de la Tomate",
                "scientific_name": "Xanthomonas spp.",
                "category": "Bactériennes",
                "cause": "Bactérie pathogène",
                "description": "Infection bactérienne causant des petites taches noires sur feuilles et fruits",
                "severity": "Modérée",
                "season": "Temps chaud et humide",
                "affected_crops": ["Tomate", "Poivron"],
                "symptoms": [
                    "Petites taches noires avec halo jaune",
                    "Taches sur feuilles, tiges et fruits",
                    "Défoliation en cas d'infection sévère",
                    "Fruits craquelés et déformés",
                ],
                "favorable_conditions": [
                    "Température 25-30°C",
                    "Humidité élevée",
                    "Blessures sur les plants",
                    "Propagation par éclaboussures",
                ],
            },
            "Potato_Late_blight": {
                "name": "Mildiou de la Pomme de Terre",
                "scientific_name": "Phytophthora infestans",
                "category": "Fongiques",
                "cause": "Oomycète pathogène",
                "description": "Maladie la plus destructrice de la pomme de terre",
                "severity": "Très Élevée",
                "season": "Temps frais et humide",
                "affected_crops": ["Pomme de terre", "Tomate"],
                "symptoms": [
                    "Taches brunes aqueuses sur feuilles",
                    "Pourriture noire des tubercules",
                    "Flétrissement rapide du feuillage",
                    "Odeur désagréable des tubercules infectés",
                ],
                "favorable_conditions": [
                    "Température 10-20°C",
                    "Humidité >90%",
                    "Temps pluvieux prolongé",
                    "Rosée persistante",
                ],
            },
            "Corn_Common_rust": {
                "name": "Rouille Commune du Maïs",
                "scientific_name": "Puccinia sorghi",
                "category": "Fongiques",
                "cause": "Champignon pathogène",
                "description": "Maladie fongique caractérisée par des pustules orange-brun",
                "severity": "Modérée",
                "season": "Été frais et humide",
                "affected_crops": ["Maïs"],
                "symptoms": [
                    "Pustules orange-brun sur les feuilles",
                    "Pustules ovales à circulaires",
                    "Jaunissement des feuilles",
                    "Réduction du rendement en grains",
                ],
                "favorable_conditions": [
                    "Température 16-23°C",
                    "Humidité élevée",
                    "Rosée matinale",
                    "Variétés sensibles",
                ],
            },
            "Wheat_Leaf_rust": {
                "name": "Rouille Brune du Blé",
                "scientific_name": "Puccinia triticina",
                "category": "Fongiques",
                "cause": "Champignon pathogène",
                "description": "Maladie importante du blé causant des pertes de rendement",
                "severity": "Élevée",
                "season": "Printemps et début été",
                "affected_crops": ["Blé", "Orge"],
                "symptoms": [
                    "Pustules orange-brun sur les feuilles",
                    "Taches circulaires à ovales",
                    "Jaunissement prématuré",
                    "Réduction du poids des grains",
                ],
                "favorable_conditions": [
                    "Température 15-22°C",
                    "Humidité élevée",
                    "Irrigation par aspersion",
                    "Densité de plantation élevée",
                ],
            },
            "Rice_Blast": {
                "name": "Pyriculariose du Riz",
                "scientific_name": "Magnaporthe oryzae",
                "category": "Fongiques",
                "cause": "Champignon pathogène",
                "description": "Maladie la plus destructrice du riz dans le monde",
                "severity": "Très Élevée",
                "season": "Saison des pluies",
                "affected_crops": ["Riz"],
                "symptoms": [
                    "Taches losangiques gris-brun sur feuilles",
                    "Lésions sur le col de la panicule",
                    "Échaudage des grains",
                    "Cassure des tiges",
                ],
                "favorable_conditions": [
                    "Température 25-28°C",
                    "Humidité élevée",
                    "Fertilisation azotée excessive",
                    "Variétés sensibles",
                ],
            },
            "Grape_Powdery_mildew": {
                "name": "Oïdium de la Vigne",
                "scientific_name": "Erysiphe necator",
                "category": "Fongiques",
                "cause": "Champignon pathogène",
                "description": "Maladie fongique formant un duvet blanc sur les organes verts",
                "severity": "Élevée",
                "season": "Printemps et été",
                "affected_crops": ["Vigne", "Raisin"],
                "symptoms": [
                    "Duvet blanc poudreux sur feuilles",
                    "Taches blanches sur les grappes",
                    "Déformation des feuilles",
                    "Éclatement des baies",
                ],
                "favorable_conditions": [
                    "Température 20-27°C",
                    "Humidité modérée",
                    "Temps sec après rosée",
                    "Mauvaise aération",
                ],
            },
            "Pepper_Bacterial_spot": {
                "name": "Tache Bactérienne du Poivron",
                "scientific_name": "Xanthomonas campestris",
                "category": "Bactériennes",
                "cause": "Bactérie pathogène",
                "description": "Infection bactérienne affectant feuilles et fruits du poivron",
                "severity": "Modérée",
                "season": "Temps chaud et humide",
                "affected_crops": ["Poivron", "Piment", "Tomate"],
                "symptoms": [
                    "Petites taches brunes avec halo jaune",
                    "Taches liégeuses sur les fruits",
                    "Défoliation sévère",
                    "Réduction de la qualité des fruits",
                ],
                "favorable_conditions": [
                    "Température 24-30°C",
                    "Humidité élevée",
                    "Blessures mécaniques",
                    "Propagation par eau",
                ],
            },
            "Healthy": {
                "name": "Plant Saine",
                "scientific_name": "N/A",
                "category": "Aucune",
                "cause": "Aucune",
                "description": "Plant en bonne santé sans signes de maladie",
                "severity": "Aucune",
                "season": "N/A",
                "affected_crops": ["Toutes"],
                "symptoms": [
                    "Feuillage vert et vigoureux",
                    "Croissance normale",
                    "Absence de taches ou lésions",
                    "Système racinaire sain",
                ],
                "favorable_conditions": [
                    "Nutrition équilibrée",
                    "Irrigation adéquate",
                    "Bonne aération",
                    "Conditions climatiques favorables",
                ],
            },
        }

    def _initialize_treatments_database(self) -> Dict[str, List[Dict]]:
        """
        Initialise la base de données des traitements
        """
        return {
            "Tomato_Late_blight": [
                {
                    "type": "Fongicide préventif",
                    "description": "Application de fongicides cupriques avant l'apparition des symptômes",
                    "products": [
                        "Bouillie bordelaise",
                        "Oxychlorure de cuivre",
                        "Mancozèbe",
                    ],
                    "application": "Pulvérisation foliaire tous les 7-10 jours",
                    "timing": "Avant et pendant les périodes à risque",
                },
                {
                    "type": "Fongicide curatif",
                    "description": "Traitement systémique dès les premiers symptômes",
                    "products": ["Métalaxyl", "Cymoxanil", "Fluazinam"],
                    "application": "Pulvérisation avec adjuvant",
                    "timing": "Dès détection des premiers symptômes",
                },
                {
                    "type": "Mesures culturales",
                    "description": "Amélioration des conditions de culture",
                    "products": [
                        "Paillis plastique",
                        "Système d'irrigation goutte-à-goutte",
                    ],
                    "application": "Installation en début de culture",
                    "timing": "Avant plantation",
                },
            ],
            "Tomato_Early_blight": [
                {
                    "type": "Fongicide préventif",
                    "description": "Protection avant apparition des symptômes",
                    "products": ["Chlorothalonil", "Mancozèbe", "Azoxystrobine"],
                    "application": "Pulvérisation régulière",
                    "timing": "Dès la formation des premiers fruits",
                },
                {
                    "type": "Biocontrôle",
                    "description": "Utilisation d'agents biologiques",
                    "products": ["Bacillus subtilis", "Trichoderma harzianum"],
                    "application": "Traitement des semences et sol",
                    "timing": "Avant semis et plantation",
                },
            ],
            "Tomato_Bacterial_spot": [
                {
                    "type": "Bactéricide cuivre",
                    "description": "Application de produits cupriques",
                    "products": ["Sulfate de cuivre", "Hydroxyde de cuivre"],
                    "application": "Pulvérisation préventive",
                    "timing": "Conditions favorables prévues",
                },
                {
                    "type": "Résistance variétale",
                    "description": "Utilisation de variétés résistantes",
                    "products": ["Variétés certifiées résistantes"],
                    "application": "Choix variétal",
                    "timing": "Avant plantation",
                },
            ],
            "Corn_Common_rust": [
                {
                    "type": "Fongicide foliaire",
                    "description": "Traitement préventif des feuilles",
                    "products": ["Tébuconazole", "Propiconazole", "Azoxystrobine"],
                    "application": "Pulvérisation aérienne ou terrestre",
                    "timing": "Avant floraison",
                }
            ],
            "Wheat_Leaf_rust": [
                {
                    "type": "Fongicide systémique",
                    "description": "Protection systémique de la plante",
                    "products": ["Tébuconazole", "Propiconazole", "Époxiconazole"],
                    "application": "Pulvérisation foliaire",
                    "timing": "Montaison à épiaison",
                }
            ],
            "Rice_Blast": [
                {
                    "type": "Fongicide systémique",
                    "description": "Traitement préventif et curatif",
                    "products": ["Tricyclazole", "Carbendazime", "Isoprothiolane"],
                    "application": "Pulvérisation ou granulés",
                    "timing": "Stades critiques de développement",
                }
            ],
            "Grape_Powdery_mildew": [
                {
                    "type": "Fongicide préventif",
                    "description": "Protection avant infection",
                    "products": ["Soufre", "Kresoxim-méthyl", "Myclobutanil"],
                    "application": "Poudrage ou pulvérisation",
                    "timing": "Débourrement à véraison",
                }
            ],
            "Pepper_Bacterial_spot": [
                {
                    "type": "Bactéricide préventif",
                    "description": "Protection contre l'infection bactérienne",
                    "products": ["Streptomycine", "Kasugamycine", "Cuivre"],
                    "application": "Pulvérisation préventive",
                    "timing": "Conditions favorables",
                }
            ],
        }

    def _initialize_prevention_database(self) -> Dict[str, List[str]]:
        """
        Initialise la base de données des mesures préventives
        """
        return {
            "Tomato_Late_blight": [
                "Éviter l'irrigation par aspersion",
                "Assurer une bonne aération entre les plants",
                "Éliminer les résidus de culture infectés",
                "Rotation des cultures (3-4 ans)",
                "Utiliser des semences certifiées",
                "Drainage efficace des parcelles",
                "Éviter l'excès d'azote",
                "Surveillance météorologique",
            ],
            "Tomato_Early_blight": [
                "Rotation des cultures",
                "Élimination des débris végétaux",
                "Irrigation au pied des plants",
                "Éviter le stress hydrique",
                "Fertilisation équilibrée",
                "Espacement adéquat des plants",
                "Utilisation de paillis",
                "Variétés résistantes",
            ],
            "Tomato_Bacterial_spot": [
                "Semences traitées et certifiées",
                "Désinfection des outils",
                "Éviter la manipulation par temps humide",
                "Contrôle des insectes vecteurs",
                "Irrigation localisée",
                "Élimination des plants infectés",
                "Rotation avec cultures non-hôtes",
                "Hygiène stricte en serre",
            ],
            "Corn_Common_rust": [
                "Utilisation de variétés résistantes",
                "Rotation des cultures",
                "Élimination des résidus infectés",
                "Éviter les semis tardifs",
                "Fertilisation azotée modérée",
                "Surveillance régulière",
                "Espacement optimal des plants",
            ],
            "Wheat_Leaf_rust": [
                "Variétés résistantes ou tolérantes",
                "Rotation des cultures",
                "Élimination des repousses",
                "Semis à la date optimale",
                "Fertilisation équilibrée",
                "Surveillance des bulletins d'alerte",
                "Éviter les densités excessives",
            ],
            "Rice_Blast": [
                "Variétés résistantes",
                "Gestion de l'eau d'irrigation",
                "Fertilisation azotée raisonnée",
                "Élimination des chaumes infectés",
                "Rotation avec cultures sèches",
                "Semences saines",
                "Éviter l'excès d'humidité",
            ],
            "Grape_Powdery_mildew": [
                "Taille pour aérer la végétation",
                "Élimination des sarments infectés",
                "Palissage pour éviter l'ombrage",
                "Éviter l'excès d'azote",
                "Variétés moins sensibles",
                "Surveillance météorologique",
                "Nettoyage d'hiver rigoureux",
            ],
            "Pepper_Bacterial_spot": [
                "Semences certifiées",
                "Rotation avec cultures non-solanacées",
                "Éviter l'irrigation par aspersion",
                "Désinfection des outils et structures",
                "Élimination des plants infectés",
                "Contrôle de l'humidité en serre",
                "Éviter les blessures mécaniques",
            ],
        }

    def get_disease_info(self, disease_name: str) -> Optional[Dict]:
        """
        Récupère les informations complètes d'une maladie
        """
        return self.diseases_data.get(
            disease_name, None
        )  # Assure un retour explicite si la maladie n'existe pas

    def get_treatment_info(self, disease_name: str) -> List[Dict]:
        """
        Récupère les informations de traitement d'une maladie
        """
        return self.treatments_data.get(disease_name, [])

    def get_prevention_info(self, disease_name: str) -> List[str]:
        """
        Récupère les mesures préventives d'une maladie
        """
        return self.prevention_data.get(disease_name, [])

    def get_all_diseases(self) -> List[Dict]:
        """
        Récupère la liste de toutes les maladies
        """
        diseases = []
        return list(
            self.diseases_data.values()
        )  # Retourne directement toutes les maladies sans manipulation inutile

    def search_diseases(self, query: str, category: str = None) -> List[Dict]:
        """
        Recherche des maladies par nom ou symptôme
        """
        query_lower = query.lower()
        results = []

        for disease_id, disease_data in self.diseases_data.items():
            # Search in name, scientific name, and symptoms
            searchable_text = (
                disease_data.get("name", "").lower()
                + " "
                + disease_data.get("scientific_name", "").lower()
                + " "
                + " ".join(disease_data.get("symptoms", [])).lower()
            )

            if query_lower in searchable_text:
                if category is None or disease_data.get("category") == category:
                    disease_info = disease_data.copy()
                    disease_info["id"] = disease_id
                    results.append(disease_info)

        return results

    def get_diseases_by_crop(self, crop_name: str) -> List[Dict]:
        """
        Récupère les maladies affectant une culture spécifique
        """
        crop_diseases = []

        for disease_id, disease_data in self.diseases_data.items():
            affected_crops = disease_data.get("affected_crops", [])

            for crop in affected_crops:
                if crop_name.lower() in crop.lower():
                    disease_info = disease_data.copy()
                    disease_info["id"] = disease_id
                    crop_diseases.append(disease_info)
                    break

        return crop_diseases

    def get_disease_statistics(self) -> Dict[str, Any]:
        """
        Génère des statistiques sur la base de données des maladies
        """
        total_diseases = len(self.diseases_data)

        # Count by category
        category_counts = {}
        severity_counts = {}
        crop_counts = {}

        for disease_data in self.diseases_data.values():
            # Category stats
            category = disease_data.get("category", "Unknown")
            category_counts[category] = category_counts.get(category, 0) + 1

            # Severity stats
            severity = disease_data.get("severity", "Unknown")
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

            # Crop stats
            for crop in disease_data.get("affected_crops", []):
                crop_counts[crop] = crop_counts.get(crop, 0) + 1

        return {
            "total_diseases": total_diseases,
            "category_distribution": category_counts,
            "severity_distribution": severity_counts,
            "most_affected_crops": dict(
                sorted(crop_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
            "database_version": datetime.now().strftime("%Y-%m-%d"),
            "coverage": {
                "fungal_diseases": category_counts.get("Fongiques", 0),
                "bacterial_diseases": category_counts.get("Bactériennes", 0),
                "viral_diseases": category_counts.get("Virales", 0),
                "parasitic_diseases": category_counts.get("Parasitaires", 0),
            },
        }


def export_database(self, format_type: str = "json") -> str:
    """
    Exporte la base de données au format spécifié.
    """
    if not self.diseases_data:
        raise ValueError("🚨 La base de données est vide, impossible d'exporter.")

    if format_type == "json":
        export_data = {
            "diseases": self.diseases_data,
            "treatments": self.treatments_data,
            "prevention": self.prevention_data,
            "export_date": datetime.now().isoformat(),
            "version": "1.0",
        }
        return json.dumps(export_data, ensure_ascii=False, indent=2)

    elif format_type == "csv":
        # Convertir en DataFrame pour l'export CSV
        diseases_list = []
        for disease_id, disease_data in self.diseases_data.items():
            row = disease_data.copy()
            row["disease_id"] = disease_id
            row["symptoms"] = "; ".join(row.get("symptoms", []))
            row["affected_crops"] = "; ".join(row.get("affected_crops", []))
            row["favorable_conditions"] = "; ".join(row.get("favorable_conditions", []))
            diseases_list.append(row)

        df = pd.DataFrame(diseases_list)
        return df.to_csv(index=False)

    else:
        raise ValueError(f"🚨 Format non supporté: {format_type}")

    def add_disease(self, disease_id: str, disease_data: Dict) -> bool:
        """
        Ajoute une nouvelle maladie à la base de données
        """
        try:
            if disease_id not in self.diseases_data:
                self.diseases_data[disease_id] = disease_data
                return True
            else:
                return False  # Disease already exists
        except Exception as e:
            print(f"Erreur lors de l'ajout de la maladie: {e}")
            return False

    def update_disease(self, disease_id: str, updated_data: Dict) -> bool:
        """
        Met à jour les informations d'une maladie existante
        """
        try:
            if disease_id in self.diseases_data:
                self.diseases_data[disease_id].update(updated_data)
                return True
            else:
                return False  # Disease not found
        except Exception as e:
            print(f"Erreur lors de la mise à jour de la maladie: {e}")
            return False
