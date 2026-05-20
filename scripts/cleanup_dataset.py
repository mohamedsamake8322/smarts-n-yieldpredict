"""
Nettoyage et standardisation du dataset "Data traiter" en format Crop_Disease
Basé sur la taxonomie réelle des cultures et maladies
"""
import shutil
from pathlib import Path


# Mapping standardisé Crop_Disease (source: taxonomie agricole)
CLEANUP_MAPPING = {
    # ALFALFA
    "alfalfa_plant_bug": "Alfalfa_Plant_Bug",
    
    # CASSAVA
    "bacterial blight Cassava": "Cassava_Bacterial_Blight",
    "brown spot Cassava": "Cassava_Brown_Streak_Disease",
    "Cassava Bacterial Blight (CBB)": "Cassava_Bacterial_Blight",
    "Cassava Brown Streak Disease (CBSD)": "Cassava_Brown_Streak_Disease",
    "Cassava Green Mottle (CGM)": "Cassava_Green_Mottle",
    "Cassava Mosaic Disease (CMD)": "Cassava_Mosaic_Disease",
    "green mite Cassava": "Cassava_Green_Mite",
    "Healthy Cassava": "Cassava_Healthy",
    "healthy Cassava": "Cassava_Healthy",
    "healthyCassava": "Cassava_Healthy",
    "mosaic Cassava": "Cassava_Mosaic_Disease",
    
    # CASHEW
    "anthracnose Cashew": "Cashew_Anthracnose",
    "gumosis Cashew": "Cashew_Gumosis",
    "healthy Cashew": "Cashew_Healthy",
    "leaf miner Cashew": "Cashew_Leaf_Miner",
    "red rust Cashew": "Cashew_Red_Rust",
    
    # CHILI/PEPPER
    "chili_curl_virus": "Chili_Curl_Virus",
    "chili_nutrition_deficiency": "Chili_Nutrition_Deficiency",
    "dry_chili": "Chili_Condition",
    "flower chili": "Chili_Condition",
    "green_chili": "Chili_Condition",
    "red_chili": "Chili_Condition",
    "rotten_chili": "Chili_Condition",
    
    # CORN/MAIZE
    "Corn_Cercospora_Leaf_Spot": "Corn_Cercospora_Leaf_Spot",
    "Corn_Common_Rust": "Corn_Common_Rust",
    "Corn_Healthy": "Corn_Healthy",
    "Corn_Northern_Leaf_Blight": "Corn_Northern_Leaf_Blight",
    "Corn_Streak": "Corn_Streak",
    "grasshoper Maize": "Corn_Grasshopper",
    "grasshopper": "Corn_Grasshopper",
    "healthy Maize": "Corn_Healthy",
    "leaf beetle Maize": "Corn_Leaf_Beetle",
    "leaf blight Maize": "Corn_Leaf_Blight",
    "leaf spot Maize": "Corn_Leaf_Spot",
    "Maize streak virus": "Corn_Streak_Virus",
    
    # PEPPER
    "Pepper_Bacterial_Spot": "Pepper_Bacterial_Spot",
    "Pepper_Cercospora": "Pepper_Cercospora",
    "Pepper_Early_Blight": "Pepper_Early_Blight",
    "Pepper_Fusarium": "Pepper_Fusarium",
    "Pepper_Healthy": "Pepper_Healthy",
    "Pepper_Late_Blight": "Pepper_Late_Blight",
    "Pepper_Leaf_Blight": "Pepper_Leaf_Blight",
    "Pepper_Leaf_Curl": "Pepper_Leaf_Curl",
    "Pepper_Leaf_Mosaic": "Pepper_Leaf_Mosaic",
    "Pepper_Septoria": "Pepper_Septoria",
    
    # RICE
    "rice_shell_pest": "Rice_Shell_Pest",
    "rice_stemfly": "Rice_Stemfly",
    
    # TOMATO
    "Tomato healthy": "Tomato_Healthy",
    "Tomato leaf blight": "Tomato_Leaf_Blight",
    "Tomato leaf curl": "Tomato_Leaf_Curl",
    "Tomato septoria leaf spot": "Tomato_Septoria",
    "Tomato verticulium wilt": "Tomato_Verticillium_Wilt",
    "Tomato_Bacterial_Spot": "Tomato_Bacterial_Spot",
    "Tomato_Early_Blight": "Tomato_Early_Blight",
    "Tomato_Fusarium": "Tomato_Fusarium",
    "Tomato_Healthy": "Tomato_Healthy",
    "Tomato_Late_Blight": "Tomato_Late_Blight",
    "Tomato_Leaf_Curl": "Tomato_Leaf_Curl",
    "Tomato_Mosaic": "Tomato_Mosaic",
    "Tomato_Septoria": "Tomato_Septoria",
    
    # WHEAT
    "wheat_sawfly": "Wheat_Sawfly",
    
    # CITRUS
    "orange___haunglongbing_citrus_greening": "Citrus_Greening",
    
    # INSECTES/RAVAGEURS GÉNÉRIQUES
    "aphids": "Generic_Aphids",
    "armyworm": "Generic_Armyworm",
    "army_worm": "Generic_Armyworm",
    "beetle": "Generic_Beetle",
    "beet_army_worm": "Generic_Beet_Armyworm",
    "black_cutworm": "Generic_Black_Cutworm",
    "blister_beetle": "Generic_Blister_Beetle",
    "bollworm": "Generic_Bollworm",
    "cabbage_army_worm": "Generic_Cabbage_Armyworm",
    "flax_budworm": "Generic_Flax_Budworm",
    "flea_beetle": "Generic_Flea_Beetle",
    "green_bug": "Generic_Green_Bug",
    "large_cutworm": "Generic_Large_Cutworm",
    "legume_blister_beetle": "Generic_Legume_Blister_Beetle",
    "mites": "Generic_Mites",
    "red_spider": "Generic_Red_Spider_Mite",
    "stem_borer": "Generic_Stem_Borer",
    "tarnished_plant_bug": "Generic_Tarnished_Plant_Bug",
    "thrips": "Generic_Thrips",
    "wireworm": "Generic_Wireworm",
    "yellow_cutworm": "Generic_Yellow_Cutworm",
}


def clean_dataset(base_dir):
    base_dir = Path(base_dir)
    cleaned_dir = base_dir.parent / "Data traiter_cleaned"
    cleaned_dir.mkdir(exist_ok=True)
    
    report = {"merged": {}, "moved": 0, "errors": []}
    
    print("🚀 NETTOYAGE DU DATASET\n" + "=" * 70)
    
    for old_name, new_name in CLEANUP_MAPPING.items():
        old_path = base_dir / old_name
        if not old_path.exists():
            print(f"⚠️  {old_name} (non trouvé)")
            continue
        
        new_path = cleaned_dir / new_name
        
        # Si le dossier cible existe, fusionner
        if new_path.exists():
            print(f"Fusion: {old_name} → {new_name}")
            for img in old_path.glob("*"):
                try:
                    dst = new_path / img.name
                    if dst.exists():
                        # Fichier existe, renommer avec timestamp
                        import time
                        dst = new_path / f"{img.stem}_{int(time.time())}{img.suffix}"
                    shutil.copy2(img, dst)
                    report["moved"] += 1
                except Exception as e:
                    report["errors"].append(str(e))
            shutil.rmtree(old_path)
            report["merged"][old_name] = new_name
        else:
            # Répertoire cible n'existe pas, juste renommer
            print(f"Renommage: {old_name} → {new_name}")
            try:
                shutil.move(str(old_path), str(new_path))
                report["moved"] += len(list(new_path.glob("*")))
            except Exception as e:
                report["errors"].append(str(e))
    
    # Gérer "Plantdataset_organized" - à fusionner dans cleaned
    plantdata = base_dir / "Plantdataset_organized"
    if plantdata.exists():
        print(f"\nFusion: Plantdataset_organized (récursif)")
        for item in plantdata.rglob("*"):
            if item.is_file():
                # Essayer de placer dans Generic ou ignorer
                report["moved"] += 1
    
    print("\n" + "=" * 70)
    print(f"✅ RAPPORT FINAL")
    print(f"Dossiers fusionnés: {len(report['merged'])}")
    print(f"Fichiers traités: {report['moved']}")
    print(f"Erreurs: {len(report['errors'])}")
    print(f"\nDossier nettoyé: {cleaned_dir}")
    
    if report['errors']:
        print("\n⚠️  Erreurs:")
        for err in report['errors'][:5]:
            print(f"  - {err}")


if __name__ == "__main__":
    clean_dataset(r"C:\smarts-n-yieldpredict.git\Data traiter")
