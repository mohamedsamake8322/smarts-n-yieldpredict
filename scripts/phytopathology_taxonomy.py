"""
PHYTOPATHOLOGY TAXONOMY MAPPING
Create comprehensive mapping for production-grade dataset organization
"""

import os
import re
from pathlib import Path
from collections import defaultdict

class PhytopathologyTaxonomy:
    """Phytopathology-based taxonomy for plant disease and pest classification"""

    def __init__(self):
        # Level 1: Main categories
        self.level_1_categories = {
            'disease': 'biotic and abiotic plant disorders',
            'pest': 'insects, mites, nematodes, and other pests',
            'growth_stage': 'plant development and growth stages'
        }

        # Level 2: Disease subcategories
        self.disease_categories = {
            'fungal': 'fungal pathogens (mycoses)',
            'bacterial': 'bacterial pathogens (bacterioses)',
            'viral': 'viral pathogens (viroses)',
            'abiotic': 'nutrient deficiencies and environmental stress'
        }

        # Level 2: Pest subcategories
        self.pest_categories = {
            'insect': 'insects (coleoptera, hemiptera, lepidoptera, etc.)',
            'mite': 'mites and ticks',
            'nematode': 'nematodes',
            'mollusk': 'slugs and snails'
        }

        # Level 2: Growth stage subcategories
        self.growth_categories = {
            'vegetative': 'leaf development and growth',
            'reproductive': 'flowering and fruiting',
            'maturity': 'ripening and harvest stages'
        }

        # Initialize taxonomy mapping
        self.taxonomy_map = {}
        self.reverse_map = {}  # For validation
        self._create_taxonomy_mapping()

    def _create_taxonomy_mapping(self):
        """Create comprehensive taxonomy mapping based on phytopathology standards"""

        # DISEASES - Fungal
        fungal_diseases = {
            # Tomato diseases
            'tomato_early_blight': ('disease', 'fungal', 'Alternaria solani'),
            'tomato_late_blight': ('disease', 'fungal', 'Phytophthora infestans'),
            'tomato_septoria_leaf_spot': ('disease', 'fungal', 'Septoria lycopersici'),
            'tomato_leaf_mold': ('disease', 'fungal', 'Passalora fulva'),

            # Chili/Pepper diseases
            'chili_cercospora_leaf_spot': ('disease', 'fungal', 'Cercospora capsici'),
            'chili_white_spot': ('disease', 'fungal', 'Pseudocercospora cubensis'),

            # Other fungal diseases
            'apple_scab': ('disease', 'fungal', 'Venturia inaequalis'),
            'grape_black_rot': ('disease', 'fungal', 'Guignardia bidwellii'),
            'corn_rust': ('disease', 'fungal', 'Puccinia spp.'),
            'corn_gray_leaf_spot': ('disease', 'fungal', 'Cercospora zeae-maydis'),
            'corn_northern_leaf_blight': ('disease', 'fungal', 'Setosphaeria turcica'),
            'squash_powdery_mildew': ('disease', 'fungal', 'Podosphaera xanthii'),
            'strawberry_leaf_scorch': ('disease', 'fungal', 'Diplocarpon earlianum'),
            'potato_early_blight': ('disease', 'fungal', 'Alternaria solani'),
            'potato_late_blight': ('disease', 'fungal', 'Phytophthora infestans'),
        }

        # DISEASES - Bacterial
        bacterial_diseases = {
            'chili_bacterial_spot': ('disease', 'bacterial', 'Xanthomonas spp.'),
            'tomato_bacterial_spot': ('disease', 'bacterial', 'Xanthomonas spp.'),
            'pepper_bacterial_spot': ('disease', 'bacterial', 'Xanthomonas spp.'),
            'orange_huanglongbing': ('disease', 'bacterial', 'Candidatus Liberibacter asiaticus'),
        }

        # DISEASES - Viral
        viral_diseases = {
            'chili_curl_virus': ('disease', 'viral', 'Begomovirus spp.'),
            'tomato_yellow_leaf_curl_virus': ('disease', 'viral', 'Begomovirus spp.'),
        }

        # DISEASES - Abiotic
        abiotic_diseases = {
            'chili_nutrition_deficiency': ('disease', 'abiotic', 'nutrient imbalance'),
        }

        # PESTS - Insects
        insect_pests = {
            'aphids': ('pest', 'insect', 'Aphididae family'),
            'beetles': ('pest', 'insect', 'Coleoptera order'),
            'weevils': ('pest', 'insect', 'Curculionidae family'),
            'borers': ('pest', 'insect', 'various families'),
            'caterpillars': ('pest', 'insect', 'Lepidoptera larvae'),
            'leafhoppers': ('pest', 'insect', 'Cicadellidae family'),
            'planthoppers': ('pest', 'insect', 'Delphacidae family'),
            'thrips': ('pest', 'insect', 'Thripidae family'),
            'whiteflies': ('pest', 'insect', 'Aleyrodidae family'),
            'spider_mites': ('pest', 'insect', 'Tetranychidae family'),
            'mealybugs': ('pest', 'insect', 'Pseudococcidae family'),
            'scales': ('pest', 'insect', 'Coccoidea superfamily'),
            'fruit_flies': ('pest', 'insect', 'Tephritidae family'),
            'sawflies': ('pest', 'insect', 'Symphyta suborder'),
            'cutworms': ('pest', 'insect', 'Noctuidae family'),
            'wireworms': ('pest', 'insect', 'Elateridae larvae'),
            'flea_beetles': ('pest', 'insect', 'Chrysomelidae family'),
            'blister_beetles': ('pest', 'insect', 'Meloidae family'),
            'grasshoppers': ('pest', 'insect', 'Acrididae family'),
            'crickets': ('pest', 'insect', 'Gryllidae family'),
            'mole_crickets': ('pest', 'insect', 'Gryllotalpidae family'),
        }

        # GROWTH STAGES
        growth_stages = {
            'chili_green': ('growth_stage', 'vegetative', 'green chili development'),
            'chili_flower': ('growth_stage', 'reproductive', 'flowering stage'),
            'chili_red': ('growth_stage', 'maturity', 'red ripe chili'),
            'chili_dry': ('growth_stage', 'maturity', 'dried chili'),
            'chili_rotten': ('growth_stage', 'maturity', 'overripe/rotting chili'),
        }

        # Combine all mappings
        all_mappings = {}
        all_mappings.update(fungal_diseases)
        all_mappings.update(bacterial_diseases)
        all_mappings.update(viral_diseases)
        all_mappings.update(abiotic_diseases)
        all_mappings.update(insect_pests)
        all_mappings.update(growth_stages)

        self.taxonomy_map = all_mappings

        # Create reverse mapping for validation
        for class_name, (l1, l2, scientific) in all_mappings.items():
            key = f"{l1}/{l2}/{class_name}"
            self.reverse_map[key] = scientific

    def map_class_name(self, original_path):
        """Map original class path to normalized taxonomy path"""
        # Normalize the original path to extract meaningful class name
        path_str = str(original_path).lower().replace('\\', '/')

        # Extract class name from path
        parts = path_str.split('/')
        class_name = parts[-1] if parts else ''

        # Clean up class name
        class_name = re.sub(r'[^\w\s]', '', class_name)  # Remove special chars
        class_name = re.sub(r'\s+', '_', class_name)     # Spaces to underscores
        class_name = class_name.strip('_')               # Remove leading/trailing underscores

        # Try to find mapping
        if class_name in self.taxonomy_map:
            l1, l2, scientific = self.taxonomy_map[class_name]
            return f"{l1}/{l2}/{class_name}", scientific

        # Try fuzzy matching for common variations
        fuzzy_mappings = {
            # Chili diseases
            'cercospora_leaf_spot': 'chili_cercospora_leaf_spot',
            'white_spot': 'chili_white_spot',
            'bacterial_spot': 'chili_bacterial_spot',
            'curl_virus': 'chili_curl_virus',
            'nutrition_deficiency': 'chili_nutrition_deficiency',

            # Tomato diseases
            'early_blight': 'tomato_early_blight',
            'late_blight': 'tomato_late_blight',
            'septoria_leaf_spot': 'tomato_septoria_leaf_spot',
            'leaf_mold': 'tomato_leaf_mold',
            'bacterial_spot': 'tomato_bacterial_spot',
            'yellow_leaf_curl_virus': 'tomato_yellow_leaf_curl_virus',

            # Other mappings
            'apple_scab': 'apple_scab',
            'black_rot': 'grape_black_rot',
            'common_rust': 'corn_rust',
            'gray_leaf_spot': 'corn_gray_leaf_spot',
            'northern_leaf_blight': 'corn_northern_leaf_blight',
            'powdery_mildew': 'squash_powdery_mildew',
            'leaf_scorch': 'strawberry_leaf_scorch',
        }

        if class_name in fuzzy_mappings:
            mapped_name = fuzzy_mappings[class_name]
            if mapped_name in self.taxonomy_map:
                l1, l2, scientific = self.taxonomy_map[mapped_name]
                return f"{l1}/{l2}/{mapped_name}", scientific

        # Default fallback - try to infer from context
        return self._infer_taxonomy_from_context(path_str, class_name)

    def _infer_taxonomy_from_context(self, path_str, class_name):
        """Infer taxonomy from path context when direct mapping fails"""

        # Check path context for clues
        if 'disease' in path_str:
            if any(word in class_name for word in ['spot', 'blight', 'mold', 'rot', 'scab', 'rust', 'mildew']):
                return f"disease/fungal/{class_name}", "inferred fungal disease"
            elif 'bacterial' in path_str or 'bacteria' in class_name:
                return f"disease/bacterial/{class_name}", "inferred bacterial disease"
            elif 'virus' in class_name or 'viral' in path_str:
                return f"disease/viral/{class_name}", "inferred viral disease"
            else:
                return f"disease/fungal/{class_name}", "inferred disease"

        elif 'pest' in path_str or any(word in path_str for word in ['insect', 'bug', 'beetle', 'worm', 'fly']):
            return f"pest/insect/{class_name}", "inferred insect pest"

        elif any(word in path_str for word in ['growth', 'stage', 'green', 'red', 'flower', 'fruit']):
            if 'green' in class_name or 'vegetative' in path_str:
                return f"growth_stage/vegetative/{class_name}", "inferred vegetative stage"
            elif 'flower' in class_name or 'reproductive' in path_str:
                return f"growth_stage/reproductive/{class_name}", "inferred reproductive stage"
            else:
                return f"growth_stage/maturity/{class_name}", "inferred maturity stage"

        else:
            # Ultimate fallback - assume disease/fungal
            return f"disease/fungal/{class_name}", "fallback classification"

    def validate_taxonomy(self):
        """Validate taxonomy structure and consistency"""
        issues = []

        # Check for duplicate mappings
        seen_paths = set()
        for class_name, (l1, l2, scientific) in self.taxonomy_map.items():
            path = f"{l1}/{l2}/{class_name}"
            if path in seen_paths:
                issues.append(f"Duplicate path: {path}")
            seen_paths.add(path)

        # Check level 1 validity
        for class_name, (l1, l2, scientific) in self.taxonomy_map.items():
            if l1 not in self.level_1_categories:
                issues.append(f"Invalid level 1 category '{l1}' for {class_name}")

        # Check level 2 validity
        for class_name, (l1, l2, scientific) in self.taxonomy_map.items():
            if l1 == 'disease' and l2 not in self.disease_categories:
                issues.append(f"Invalid disease category '{l2}' for {class_name}")
            elif l1 == 'pest' and l2 not in self.pest_categories:
                issues.append(f"Invalid pest category '{l2}' for {class_name}")
            elif l1 == 'growth_stage' and l2 not in self.growth_categories:
                issues.append(f"Invalid growth category '{l2}' for {class_name}")

        return issues

    def get_taxonomy_summary(self):
        """Get summary of taxonomy structure"""
        summary = {
            'level_1': {},
            'level_2': {},
            'total_classes': len(self.taxonomy_map)
        }

        for class_name, (l1, l2, scientific) in self.taxonomy_map.items():
            if l1 not in summary['level_1']:
                summary['level_1'][l1] = 0
            summary['level_1'][l1] += 1

            l2_key = f"{l1}/{l2}"
            if l2_key not in summary['level_2']:
                summary['level_2'][l2_key] = 0
            summary['level_2'][l2_key] += 1

        return summary

# Test the taxonomy
if __name__ == '__main__':
    taxonomy = PhytopathologyTaxonomy()

    print("PHYTOPATHOLOGY TAXONOMY VALIDATION")
    print("="*50)

    issues = taxonomy.validate_taxonomy()
    if issues:
        print(f"Found {len(issues)} issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ Taxonomy validation passed")

    summary = taxonomy.get_taxonomy_summary()
    print(f"\nTotal classes: {summary['total_classes']}")

    print("\nLevel 1 distribution:")
    for l1, count in summary['level_1'].items():
        print(f"  {l1}: {count} classes")

    print("\nLevel 2 distribution:")
    for l2, count in summary['level_2'].items():
        print(f"  {l2}: {count} classes")

    # Test some mappings
    print("\nSAMPLE MAPPINGS:")
    test_classes = [
        'Chili Leaf Disease Augmented Dataset/Bacterial Spot',
        'train/aphids',
        'Chili Growth Stage Augmented Dataset/Green Chili',
        'val/Tomato___Early_blight'
    ]

    for test_class in test_classes:
        mapped_path, scientific = taxonomy.map_class_name(test_class)
        print(f"  {test_class} -> {mapped_path} ({scientific})")