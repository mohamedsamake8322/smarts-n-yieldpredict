"""
Script pour analyser et proposer des corrections de noms scientifiques
pour les classes de maladies et ravageurs agricoles
"""

import csv
import re
from pathlib import Path

class ScientificNameCorrector:
    """Correcteur de noms scientifiques pour phytopathologie et entomologie"""

    def __init__(self):
        # Base de données de noms scientifiques corrects
        self.scientific_names = {
            # MALADIES FUNGALES - Apple
            'Apple Apple scab': 'Venturia inaequalis',
            'Apple Black rot': 'Botryosphaeria obtusa',
            'Apple Cedar apple rust': 'Gymnosporangium juniperi-virginianae',
            'apple___apple_scab': 'Venturia inaequalis',
            'apple___black_rot': 'Botryosphaeria obtusa',

            # MALADIES FUNGALES - Tomato
            'Tomato Early blight': 'Alternaria solani',
            'Tomato Late blight': 'Phytophthora infestans',
            'Tomato Leaf Mold': 'Passalora fulva',
            'Tomato Septoria leaf spot': 'Septoria lycopersici',
            'Tomato Target Spot': 'Corynespora cassiicola',
            'tomato___early_blight': 'Alternaria solani',
            'tomato___late_blight': 'Phytophthora infestans',
            'tomato___leaf_mold': 'Passalora fulva',
            'tomato___septoria_leaf_spot': 'Septoria lycopersici',
            'tomato___target_spot': 'Corynespora cassiicola',

            # MALADIES BACTÉRIENNES
            'Tomato Bacterial spot': 'Xanthomonas spp.',
            'tomato___bacterial_spot': 'Xanthomonas spp.',
            'Pepper bell Bacterial spot': 'Xanthomonas spp.',
            'pepper_bell___bacterial_spot': 'Xanthomonas spp.',
            'Orange Haunglongbing Citrus greening': 'Candidatus Liberibacter asiaticus',
            'peach___bacterial_spot': 'Xanthomonas arboricola',

            # MALADIES VIRALES
            'Tomato Tomato mosaic virus': 'Tomato mosaic virus',

            # MALADIES FUNGALES - Potato
            'Potato Early blight': 'Alternaria solani',
            'Potato Late blight': 'Phytophthora infestans',
            'potato___early_blight': 'Alternaria solani',
            'potato___late_blight': 'Phytophthora infestans',

            # MALADIES FUNGALES - Corn/Maize
            'Blight in corn Leaf': 'Exserohilum turcicum',
            'Common Rust in corn Leaf': 'Puccinia sorghi',
            'Gray Leaf Spot in corn Leaf': 'Cercospora zeae-maydis',
            'Corn_Gray_Leaf_Spot': 'Cercospora zeae-maydis',
            'corn_maize___cercospora_leaf_spot_gray_leaf_spot': 'Cercospora zeae-maydis',
            'corn_maize___common_rust': 'Puccinia sorghi',
            'corn_maize___northern_leaf_blight': 'Setosphaeria turcica',

            # MALADIES FUNGALES - Rice
            'Bacterial leaf blight in rice leaf': 'Xanthomonas oryzae',
            'Brown spot in rice leaf': 'Bipolaris oryzae',
            'Leaf smut in rice leaf': 'Entyloma oryzae',

            # MALADIES FUNGALES - Grape
            'Grape Black rot': 'Guignardia bidwellii',
            'Grape Esca Black Measles': 'Phaeomoniella chlamydospora',
            'Grape Leaf blight Isariopsis Leaf Spot': 'Isariopsis clavispora',
            'grape___black_rot': 'Guignardia bidwellii',
            'grape___esca_black_measles': 'Phaeomoniella chlamydospora',
            'grape___leaf_blight_isariopsis_leaf_spot': 'Isariopsis clavispora',

            # MALADIES FUNGALES - Strawberry
            'Strawberry Leaf scorch': 'Diplocarpon earlianum',
            'strawberry___leaf_scorch': 'Diplocarpon earlianum',

            # MALADIES FUNGALES - Squash
            'squash___powdery_mildew': 'Podosphaera xanthii',

            # MALADIES FUNGALES - Tea
            'algal leaf in tea': 'Cephaleuros virescens',
            'anthracnose in tea': 'Colletotrichum camelliae',
            'bird eye spot in tea': 'Cercospora theae',
            'brown blight in tea': 'Glomerella cingulata',
            'red leaf spot in tea': 'Corticium theae',

            # MALADIES FUNGALES - Other
            'Cercospora leaf spot': 'Cercospora spp.',
            'Cercospora_zeae_maydis': 'Cercospora zeae-maydis',
            'Black_Leaf_Spot': 'Alternaria spp.',
            'Cauliflower_Alternaria_Leaf_Spot': 'Alternaria brassicae',
            'Eggplant_Cercospora_Leaf_Spot': 'Cercospora melongenae',
            'Early_Arachis hypogaea_Leaf_Spot': 'Cercospora arachidicola',
            'Late_Arachis hypogaea_Leaf_Spot': 'Cercosporidium personatum',

            # RAVAGEURS - Aphids
            'aphis_citricola_vander_goot': 'Aphis citricola',
            'english_grain_aphid': 'Sitobion avenae',
            'therioaphis_maculata_buckton': 'Therioaphis maculata',
            'toxoptera_aurantii': 'Toxoptera aurantii',
            'toxoptera_citricidus': 'Toxoptera citricidus',

            # RAVAGEURS - Beetles
            'Cicadellidae': 'Cicadellidae',  # Famille
            'colomerus_vitis': 'Colomerus vitis',
            'lytta_polita': 'Lytta polita',
            'oides_decempunctata': 'Oides decempunctata',

            # RAVAGEURS - Moths/Caterpillars
            'cabbage looper': 'Trichoplusia ni',
            'corn_borer': 'Ostrinia nubilalis',
            'limacodidae': 'Limacodidae',  # Famille
            'meadow_moth': 'Loxostege sticticalis',
            'papilio_xuthus': 'Papilio xuthus',
            'pieris_canidia': 'Pieris canidia',
            'prodenia_litura': 'Spodoptera litura',
            'yellow_rice_borer': 'Scirpophaga incertulas',

            # RAVAGEURS - Flies
            'dacus_dorsalishendel': 'Bactrocera dorsalis',
            'rice_gall_midge': 'Orseolia oryzae',
            'wheat_blossom_midge': 'Sitodiplosis mosellana',

            # RAVAGEURS - Thrips
            'grain_spreader_thrips': 'Frankliniella schultzei',
            'odontothrips_loti': 'Odontothrips loti',
            'scirtothrips_dorsalis_hood': 'Scirtothrips dorsalis',

            # RAVAGEURS - Bugs
            'apolugus_lucorum': 'Apolygus lucorum',
            'miridae': 'Miridae',  # Famille
            'stink_bug': 'Nezara viridula',

            # RAVAGEURS - Mites
            'longlegged_spider_mite': 'Neoseiulus cucumeris',
            'phyllocoptes_oleiverus_ashmead': 'Phyllocoptes oleivorus',

            # RAVAGEURS - Weevils
            'rice_water_weevil': 'Lissorhoptrus oryzophilus',

            # RAVAGEURS - Other insects
            'dasineura_sp': 'Dasineura spp.',
            'deporaus_marginatus_pascoe': 'Deporaus marginatus',
            'icerya_purchasi_maskell': 'Icerya purchasi',
            'lawana_imitata_melichar': 'Lawana imitata',
            'locustoidea': 'Locustoidea',  # Superfamille
            'lycorma_delicatula': 'Lycorma delicatula',
            'mole_cricket': 'Gryllotalpa spp.',
            'paddy_stem_maggot': 'Chlorops oryzae',
            'panonchus_citri_mcgregor': 'Panonychus citri',
            'penthaleus_major': 'Penthaleus major',
            'phyllocnistis_citrella_stainton': 'Phyllocnistis citrella',
            'potosiabre_vitarsis': 'Potosia brevitarsis',
            'pseudococcus_comstocki_kuwana': 'Pseudococcus comstocki',
            'rhytidodera_bowrinii_white': 'Rhytidodera bowringii',
            'rice_leaf_caterpillar': 'Naranga aenescens',
            'rice_leaf_roller': 'Cnaphalocrocis medinalis',
            'rice_leafhopper': 'Nephotettix spp.',
            'salurnis_marginella_guerr': 'Salurnis marginella',
            'sericaorient_alismots_chulsky': 'Serica orientalis',
            'small_brown_plant_hopper': 'Laodelphax striatellus',
            'sternochetus_frigidus': 'Sternochetus frigidus',
            'tetradacus_c_bactrocera_minax': 'Bactrocera minax',
            'trialeurodes_vaporariorum': 'Trialeurodes vaporariorum',
            'unaspis_yanonensis': 'Unaspis yanonensis',
            'viteus_vitifoliae': 'Viteus vitifoliae',
            'wheat_phloeothrips': 'Haplothrips tritici',
            'white_backed_plant_hopper': 'Sogatella furcifera',
            'xylotrechus': 'Xylotrechus spp.',

            # ARAIGNÉES ROUGES / ACARIENS
            'Tomato Spider mites Two spotted spider mite': 'Tetranychus urticae',
            'tomato___spider_mites_twospotted_spider_mite': 'Tetranychus urticae',

            # CHENILLES
            'rice_leaf_caterpillar': 'Naranga aenescens',
            'rice_leaf_roller': 'Cnaphalocrocis medinalis',

            # AUTRES
            'Grub': 'Melolontha spp.',
        }

        # Noms sains (healthy)
        self.healthy_names = {
            'Apple healthy': 'Malus domestica (healthy)',
            'Apple_healthy': 'Malus domestica (healthy)',
            'apple___healthy': 'Malus domestica (healthy)',
            'Blueberry healthy': 'Vaccinium corymbosum (healthy)',
            'blueberry___healthy': 'Vaccinium corymbosum (healthy)',
            'Cherry (including_sour) healthy': 'Prunus spp. (healthy)',
            'cherry_including_sour___healthy': 'Prunus spp. (healthy)',
            'Corn (maize) healthy': 'Zea mays (healthy)',
            'corn_maize___healthy': 'Zea mays (healthy)',
            'Grape healthy': 'Vitis vinifera (healthy)',
            'Peach_healthy': 'Prunus persica (healthy)',
            'Pepper bell healthy': 'Capsicum annuum (healthy)',
            'pepper_bell___healthy': 'Capsicum annuum (healthy)',
            'Potato healthy': 'Solanum tuberosum (healthy)',
            'Potato_healthy': 'Solanum tuberosum (healthy)',
            'potato___healthy': 'Solanum tuberosum (healthy)',
            'Raspberry_healthy': 'Rubus idaeus (healthy)',
            'Soybean healthy': 'Glycine max (healthy)',
            'soybean___healthy': 'Glycine max (healthy)',
            'Strawberry healthy': 'Fragaria × ananassa (healthy)',
            'Tomato healthy': 'Solanum lycopersicum (healthy)',
            'tomato___healthy': 'Solanum lycopersicum (healthy)',
            'healthy tea leaf': 'Camellia sinensis (healthy)',
            'healthy_leaf': 'Plantae (healthy)',
        }

    def normalize_name(self, name):
        """Normalise un nom pour la recherche"""
        # Convertir en minuscules, remplacer espaces et tirets par underscores
        normalized = re.sub(r'[-\s]+', '_', name.lower())
        # Supprimer les caractères spéciaux
        normalized = re.sub(r'[^\w_]', '', normalized)
        return normalized

    def get_scientific_name(self, original_name):
        """Retourne le nom scientifique pour un nom original"""
        # Chercher d'abord dans les noms de maladies/ravageurs
        if original_name in self.scientific_names:
            return self.scientific_names[original_name]

        # Chercher dans les noms sains
        if original_name in self.healthy_names:
            return self.healthy_names[original_name]

        # Essayer avec une version normalisée
        normalized = self.normalize_name(original_name)
        for key in self.scientific_names:
            if self.normalize_name(key) == normalized:
                return self.scientific_names[key]

        for key in self.healthy_names:
            if self.normalize_name(key) == normalized:
                return self.healthy_names[key]

        # Si pas trouvé, retourner le nom original avec une note
        return f"{original_name} (nom à vérifier)"

    def generate_csv_report(self, folder_names, output_file='scientific_names_correction.csv'):
        """Génère un CSV avec les correspondances noms originaux -> noms scientifiques"""

        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['original_name', 'scientific_name', 'category', 'confidence']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for name in folder_names:
                scientific = self.get_scientific_name(name)

                # Déterminer la catégorie
                if 'healthy' in name.lower() or scientific.endswith('(healthy)'):
                    category = 'healthy'
                elif any(word in scientific.lower() for word in ['spp.', 'family', 'superfamily']):
                    category = 'pest_family'
                elif any(word in scientific.lower() for word in ['virus', 'viroid']):
                    category = 'viral'
                elif 'xanthomonas' in scientific.lower() or 'liberibacter' in scientific.lower():
                    category = 'bacterial'
                elif any(word in scientific.lower() for word in ['venturia', 'botryosphaeria', 'alternaria', 'phytophthora', 'cercospora', 'colletotrichum', 'diplocarpon', 'guignardia']):
                    category = 'fungal'
                else:
                    category = 'pest'

                # Déterminer le niveau de confiance
                if scientific.endswith('(nom à vérifier)'):
                    confidence = 'low'
                elif scientific in self.scientific_names.values() or scientific in self.healthy_names.values():
                    confidence = 'high'
                else:
                    confidence = 'medium'

                writer.writerow({
                    'original_name': name,
                    'scientific_name': scientific,
                    'category': category,
                    'confidence': confidence
                })

        print(f"CSV généré: {output_file}")
        return output_file

def main():
    """Fonction principale"""
    # Liste des noms de dossiers (extraite de votre sortie)
    folder_names = [
        'Apple Apple scab', 'Apple Black rot', 'Apple Cedar apple rust', 'Apple healthy', 'Apple_healthy',
        'Bacterial leaf blight in rice leaf', 'Black_Leaf_Spot', 'Blight in corn Leaf', 'Blueberry healthy',
        'Brown spot in rice leaf', 'Cauliflower_Alternaria_Leaf_Spot', 'Cercospora leaf spot', 'Cercospora_zeae_maydis',
        'Cherry (including sour) Powdery mildew', 'Cherry (including_sour) healthy', 'Cicadellidae',
        'Common Rust in corn Leaf', 'Corn (maize) healthy', 'Corn_Gray_Leaf_Spot', 'Early_Arachis hypogaea_Leaf_Spot',
        'Eggplant_Cercospora_Leaf_Spot', 'Grape Black rot', 'Grape Esca Black Measles', 'Grape Leaf blight Isariopsis Leaf Spot',
        'Grape healthy', 'Gray Leaf Spot in corn Leaf', 'Grub', 'Late_Arachis hypogaea_Leaf_Spot', 'Leaf smut in rice leaf',
        'Orange Haunglongbing Citrus greening', 'Peach_healthy', 'Pepper bell Bacterial spot', 'Pepper bell healthy',
        'Potato Early blight', 'Potato Late blight', 'Potato healthy', 'Potato_healthy', 'Raspberry_healthy',
        'Soybean healthy', 'Strawberry Leaf scorch', 'Strawberry healthy', 'Tomato Bacterial spot', 'Tomato Early blight',
        'Tomato Late blight', 'Tomato Leaf Mold', 'Tomato Septoria leaf spot', 'Tomato Spider mites Two spotted spider mite',
        'Tomato Target Spot', 'Tomato Tomato mosaic virus', 'Tomato healthy', 'algal leaf in tea', 'anthracnose in tea',
        'aphis_citricola_vander_goot', 'apolugus_lucorum', 'apple___apple_scab', 'apple___black_rot', 'apple___healthy',
        'bird eye spot in tea', 'blueberry___healthy', 'brown blight in tea', 'cabbage looper', 'cherry_including_sour___healthy',
        'cherry_including_sour___powdery_mildew', 'chili_cercospora_leaf_spot', 'chili_white_spot', 'colomerus_vitis',
        'corn_borer', 'corn_maize___cercospora_leaf_spot_gray_leaf_spot', 'corn_maize___common_rust', 'corn_maize___healthy',
        'corn_maize___northern_leaf_blight', 'dacus_dorsalishendel', 'dasineura_sp', 'deporaus_marginatus_pascoe',
        'english_grain_aphid', 'grain_spreader_thrips', 'grape___black_rot', 'grape___esca_black_measles',
        'grape___leaf_blight_isariopsis_leaf_spot', 'healthy tea leaf', 'healthy_leaf', 'icerya_purchasi_maskell',
        'lawana_imitata_melichar', 'lemon canker', 'limacodidae', 'locustoidea', 'longlegged_spider_mite', 'lycorma_delicatula',
        'lytta_polita', 'meadow_moth', 'miridae', 'mole_cricket', 'odontothrips_loti', 'oides_decempunctata',
        'paddy_stem_maggot', 'panonchus_citri_mcgregor', 'papilio_xuthus', 'peach___bacterial_spot', 'peach_borer',
        'penthaleus_major', 'pepper_bell___bacterial_spot', 'pepper_bell___healthy', 'phyllocnistis_citrella_stainton',
        'phyllocoptes_oleiverus_ashmead', 'pieris_canidia', 'potato___early_blight', 'potato___late_blight',
        'potosiabre_vitarsis', 'prodenia_litura', 'pseudococcus_comstocki_kuwana', 'red leaf spot in tea',
        'rhytidodera_bowrinii_white', 'rice_gall_midge', 'rice_leaf_caterpillar', 'rice_leaf_roller', 'rice_leafhopper',
        'rice_water_weevil', 'salurnis_marginella_guerr', 'scirtothrips_dorsalis_hood', 'sericaorient_alismots_chulsky',
        'small_brown_plant_hopper', 'soybean___healthy', 'squash___powdery_mildew', 'sternochetus_frigidus',
        'strawberry___leaf_scorch', 'tetradacus_c_bactrocera_minax', 'therioaphis_maculata_buckton', 'tomato___bacterial_spot',
        'tomato___early_blight', 'tomato___healthy', 'tomato___late_blight', 'tomato___leaf_mold', 'tomato___septoria_leaf_spot',
        'tomato___spider_mites_twospotted_spider_mite', 'tomato___target_spot', 'toxoptera_aurantii', 'toxoptera_citricidus',
        'trialeurodes_vaporariorum', 'unaspis_yanonensis', 'viteus_vitifoliae', 'wheat_blossom_midge', 'wheat_phloeothrips',
        'white_backed_plant_hopper', 'xylotrechus', 'yellow_rice_borer'
    ]

    # Créer le correcteur
    corrector = ScientificNameCorrector()

    # Générer le CSV
    csv_file = corrector.generate_csv_report(folder_names)

    # Afficher un résumé
    print(f"\nAnalyse terminée pour {len(folder_names)} noms de dossiers")
    print(f"CSV généré: {csv_file}")

    # Compter les catégories
    categories = {}
    for name in folder_names:
        scientific = corrector.get_scientific_name(name)
        if 'healthy' in name.lower() or scientific.endswith('(healthy)'):
            cat = 'healthy'
        elif any(word in scientific.lower() for word in ['spp.', 'family', 'superfamily']):
            cat = 'pest_family'
        elif any(word in scientific.lower() for word in ['virus', 'viroid']):
            cat = 'viral'
        elif 'xanthomonas' in scientific.lower() or 'liberibacter' in scientific.lower():
            cat = 'bacterial'
        elif any(word in scientific.lower() for word in ['venturia', 'botryosphaeria', 'alternaria', 'phytophthora', 'cercospora', 'colletotrichum', 'diplocarpon', 'guignardia']):
            cat = 'fungal'
        else:
            cat = 'pest'

        categories[cat] = categories.get(cat, 0) + 1

    print("\nRépartition par catégorie:")
    for cat, count in categories.items():
        print(f"  {cat}: {count} classes")

if __name__ == "__main__":
    main()