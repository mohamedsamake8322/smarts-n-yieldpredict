import os
from pathlib import Path
from collections import defaultdict

def analyze_dataset(root_path):
    """Analyse l'arborescence et compte les images par dossier"""
    root = Path(root_path)
    results = defaultdict(int)
    
    # Extensions d'image courantes
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    
    for dirpath, dirnames, filenames in os.walk(root):
        rel_path = Path(dirpath).relative_to(root)
        image_count = sum(1 for f in filenames if Path(f).suffix.lower() in image_extensions)
        
        if image_count > 0:
            results[str(rel_path)] = image_count
    
    return results

def print_tree(root_path, results):
    """Affiche le résultat sous forme d'arborescence"""
    print(f"📁 Analyse de : {root_path}\n")
    print("=" * 80)
    
    root = Path(root_path)
    
    # Pour chaque chemin trouvé
    paths = sorted(results.keys())
    
    # Organiser hiérarchiquement
    tree = {}
    for path, count in results.items():
        parts = path.split(os.sep)
        current = tree
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = count
    
    def print_tree_recursive(node, prefix="", is_last=True):
        if isinstance(node, dict):
            items = list(node.items())
            for i, (key, value) in enumerate(items):
                is_last_item = (i == len(items) - 1)
                current_prefix = "└── " if is_last_item else "├── "
                
                if isinstance(value, int):
                    print(f"{prefix}{current_prefix}📄 {key} ({value} images)")
                else:
                    print(f"{prefix}{current_prefix}📁 {key}/")
                    next_prefix = prefix + ("    " if is_last_item else "│   ")
                    print_tree_recursive(value, next_prefix, is_last_item)
    
    print_tree_recursive(tree)
    
    # Statistiques globales
    print("\n" + "=" * 80)
    total_images = sum(results.values())
    total_folders = len(results)
    print(f"\n📊 Résumé:")
    print(f"  • Nombre total de dossiers avec images: {total_folders}")
    print(f"  • Nombre total d'images: {total_images}")
    print(f"  • Moyenne par dossier: {total_images / total_folders:.1f}")

if __name__ == '__main__':
    root = r'C:\smarts-n-yieldpredict.git\Diseasedataset'
    results = analyze_dataset(root)
    print_tree(root, results)
