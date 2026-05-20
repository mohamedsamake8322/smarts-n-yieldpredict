import os
import csv
from pathlib import Path
from collections import defaultdict

def analyze_dataset_to_csv(root_path, output_csv):
    """Analyse l'arborescence et exporte en CSV"""
    root = Path(root_path)
    
    # Extensions d'image courantes
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    
    results = []
    
    for dirpath, dirnames, filenames in os.walk(root):
        rel_path = Path(dirpath).relative_to(root)
        image_count = sum(1 for f in filenames if Path(f).suffix.lower() in image_extensions)
        
        if image_count > 0:
            # Récupérer chaque niveau du chemin
            parts = str(rel_path).split(os.sep)
            
            results.append({
                'chemin_complet': str(rel_path),
                'niveau1': parts[0] if len(parts) > 0 else '',
                'niveau2': parts[1] if len(parts) > 1 else '',
                'niveau3': parts[2] if len(parts) > 2 else '',
                'dossier': Path(dirpath).name,
                'nombre_images': image_count
            })
    
    # Écrire en CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['chemin_complet', 'niveau1', 'niveau2', 'niveau3', 'dossier', 'nombre_images'])
        writer.writeheader()
        writer.writerows(results)
    
    return len(results), sum(r['nombre_images'] for r in results)

if __name__ == '__main__':
    root = r'C:\smarts-n-yieldpredict.git\Diseasedataset'
    output_csv = r'C:\smarts-n-yieldpredict.git\Diseasedataset_analysis.csv'
    
    folder_count, image_count = analyze_dataset_to_csv(root, output_csv)
    
    print(f"✅ Rapport CSV créé : {output_csv}")
    print(f"   - {folder_count} dossiers avec images")
    print(f"   - {image_count} images au total")
