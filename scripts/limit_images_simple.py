import os
from pathlib import Path
from PIL import Image


def get_image_quality_score(path):
    """Évaluer la qualité par résolution + taille"""
    try:
        # Essayer d'ouvrir l'image
        img = Image.open(path)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        width, height = img.size
        pixels = width * height
        
        file_size = path.stat().st_size
        
        # Score = pixels 
        return pixels, (width, height), file_size
    except Exception as e:
        # Image corrompue ou non lisible
        return 0, (0, 0), 0


def limit_images(base_dir, max_per_class=1000):
    """Limiter strictement à max_per_class images par classe"""
    base_dir = Path(base_dir)
    
    print("🚀 LIMITATION STRICTE À 1000 IMAGES PAR CLASSE\n" + "=" * 80)
    
    total_removed = 0
    
    for class_dir in sorted(base_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpeg"))
        
        if len(images) <= max_per_class:
            continue
        
        print(f"\n[{class_name}] {len(images)} → {max_per_class}")
        
        # Évaluer chaque image
        quality_list = []
        for img_path in images:
            pixels, res, file_size = get_image_quality_score(img_path)
            quality_list.append((img_path, pixels, file_size))
        
        # Trier par pixels (descending) - garder les meilleures résolutions
        quality_list.sort(key=lambda x: x[1], reverse=True)
        
        # Supprimer les moins bonnes
        to_remove = quality_list[max_per_class:]
        removed_count = 0
        
        for img_path, _, _ in to_remove:
            try:
                img_path.unlink()
                removed_count += 1
            except Exception as e:
                pass
        
        print(f"  → Supprimé: {removed_count}")
        total_removed += removed_count
    
    print("\n" + "=" * 80)
    print(f"✅ Total supprimé: {total_removed}")


if __name__ == "__main__":
    limit_images(r"C:\smarts-n-yieldpredict.git\Data traiter_cleaned")
