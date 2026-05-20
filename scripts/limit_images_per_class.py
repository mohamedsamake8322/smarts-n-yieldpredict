import hashlib
import os
from pathlib import Path
from PIL import Image
import json


def hash_file(path, block_size=65536):
    """Calculer le hash SHA1 d'un fichier"""
    hasher = hashlib.sha1()
    try:
        with path.open('rb') as f:
            while True:
                data = f.read(block_size)
                if not data:
                    break
                hasher.update(data)
        return hasher.hexdigest()
    except Exception:
        return None


def get_image_quality_score(path):
    """Évaluer la qualité d'une image (résolution + taille fichier)"""
    try:
        img = Image.open(path)
        width, height = img.size
        pixels = width * height
        
        file_size = path.stat().st_size / (1024 * 1024)  # En MB
        
        # Score = pixels * densité de fichier
        # Les images de meilleure qualité auront un score plus élevé
        quality_score = pixels * (1 + file_size / 10)
        return quality_score, (width, height), file_size
    except Exception:
        return 0, (0, 0), 0


def remove_duplicates_and_limit(base_dir, max_per_class=1000):
    """
    1. Supprimer les doublons par hash
    2. Limiter chaque classe à max_per_class images (garder les meilleures)
    """
    base_dir = Path(base_dir)
    report = {
        "duplicates_removed": 0,
        "quality_removed": 0,
        "classes_processed": 0,
        "total_removed": 0,
        "details": {}
    }
    
    print("🚀 LIMITATION À 1000 IMAGES PAR CLASSE\n" + "=" * 80)
    
    for class_dir in sorted(base_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        images = list(class_dir.glob("*.*"))
        initial_count = len(images)
        
        if initial_count == 0:
            continue
        
        # ÉTAPE 1: Supprimer les doublons
        hashes = {}
        duplicates_to_remove = []
        
        for img_path in images:
            h = hash_file(img_path)
            if h is None:
                continue
            
            if h not in hashes:
                hashes[h] = img_path
            else:
                duplicates_to_remove.append(img_path)
        
        # Supprimer les doublons
        for dup_path in duplicates_to_remove:
            try:
                dup_path.unlink()
                report["duplicates_removed"] += 1
            except Exception as e:
                print(f"Erreur suppression {dup_path}: {e}")
        
        # Mettre à jour la liste
        images = list(class_dir.glob("*.*"))
        after_dedup = len(images)
        
        # ÉTAPE 2: Si > max_per_class, supprimer les moins bonnes
        if after_dedup > max_per_class:
            print(f"\n[{class_name}]")
            print(f"  Initial: {initial_count} | Après dédup: {after_dedup}")
            print(f"  → Dépassement: {after_dedup - max_per_class} images")
            
            # Évaluer la qualité de chaque image
            quality_scores = []
            for img_path in images:
                score, res, size = get_image_quality_score(img_path)
                quality_scores.append((img_path, score, res, size))
            
            # Trier par score (descending) - garder les meilleures
            quality_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Supprimer les moins bonnes
            to_remove_count = after_dedup - max_per_class
            for img_path, score, res, size in quality_scores[max_per_class:]:
                try:
                    img_path.unlink()
                    report["quality_removed"] += 1
                except Exception as e:
                    print(f"  Erreur: {e}")
            
            final_count = len(list(class_dir.glob("*.*")))
            print(f"  Final: {final_count} | Supprimés: {to_remove_count}")
            
            report["details"][class_name] = {
                "initial": initial_count,
                "duplicates_removed": len(duplicates_to_remove),
                "quality_removed": to_remove_count,
                "final": final_count
            }
        else:
            if len(duplicates_to_remove) > 0:
                print(f"[{class_name}] Doublons supprimés: {len(duplicates_to_remove)}")
            
            report["details"][class_name] = {
                "initial": initial_count,
                "duplicates_removed": len(duplicates_to_remove),
                "quality_removed": 0,
                "final": after_dedup
            }
        
        report["classes_processed"] += 1
    
    # RAPPORT
    print("\n" + "=" * 80)
    print(f"✅ NETTOYAGE TERMINÉ\n")
    print(f"Classes traitées: {report['classes_processed']}")
    print(f"Doublons supprimés: {report['duplicates_removed']}")
    print(f"Images basse qualité supprimées: {report['quality_removed']}")
    print(f"Total supprimé: {report['duplicates_removed'] + report['quality_removed']}")
    
    # Sauvegarder le rapport
    report_file = base_dir.parent / "cleanup_report.json"
    with open(report_file, 'w') as f:
        # Convertir Path en str pour JSON
        json_report = {
            "duplicates_removed": report["duplicates_removed"],
            "quality_removed": report["quality_removed"],
            "classes_processed": report["classes_processed"],
            "details": {k: v for k, v in report["details"].items()}
        }
        json.dump(json_report, f, indent=2)
    
    print(f"\nRapport sauvegardé: {report_file}")


if __name__ == "__main__":
    remove_duplicates_and_limit(r"C:\smarts-n-yieldpredict.git\Data traiter_cleaned")
