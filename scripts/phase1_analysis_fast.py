"""
OPTIMIZED DATASET ANALYSIS - FAST VERSION
Analyse rapide sans chargement PIL (trop lent sur 80k images)
"""

import os
import csv
import json
from pathlib import Path
from collections import defaultdict

def analyze_dataset_fast(root_path):
    """Analyse rapide"""
    root = Path(root_path)
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    
    classes = defaultdict(int)
    class_details = {}
    
    # SCAN
    for dirpath, dirnames, filenames in os.walk(root):
        rel_path = Path(dirpath).relative_to(root)
        image_files = [f for f in filenames if Path(f).suffix.lower() in image_extensions]
        
        if image_files:
            class_name = str(rel_path)
            count = len(image_files)
            classes[class_name] = count
            class_details[class_name] = {
                'path': dirpath,
                'count': count,
                'images': image_files
            }
    
    # STATS
    counts = list(classes.values())
    total_images = sum(counts)
    total_classes = len(counts)
    
    stats = {
        'total_images': total_images,
        'total_classes': total_classes,
        'min_class': min(counts) if counts else 0,
        'max_class': max(counts) if counts else 0,
        'avg_class': total_images / total_classes if total_classes > 0 else 0,
        'imbalance_ratio': (max(counts) / min(counts)) if min(counts) > 0 else 0,
        'problematic': [],  # < 100
        'low_data': [],  # 100-300
        'weak': [],  # 300-500
        'balanced': [],  # 500-3000
        'oversized': [],  # > 3000
    }
    
    # CLASSIFY
    for class_name, count in sorted(classes.items()):
        if count < 100:
            stats['problematic'].append((class_name, count))
        elif count < 300:
            stats['low_data'].append((class_name, count))
        elif count < 500:
            stats['weak'].append((class_name, count))
        elif count <= 3000:
            stats['balanced'].append((class_name, count))
        else:
            stats['oversized'].append((class_name, count))
    
    return classes, class_details, stats

def generate_reports(classes, class_details, stats, output_dir):
    """Générer CSV et JSON"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # CSV DÉTAILLÉ
    csv_file = output_dir / 'dataset_quality_report.csv'
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Classe', 'Nombre_Images', 'Classification', 'Chemin'])
        
        for class_name in sorted(classes.keys()):
            count = classes[class_name]
            path = class_details[class_name]['path']
            
            if count < 100:
                classification = 'UNUSABLE'
            elif count < 300:
                classification = 'LOW_DATA'
            elif count < 500:
                classification = 'WEAK'
            elif count <= 3000:
                classification = 'BALANCED'
            else:
                classification = 'OVERSIZED'
            
            writer.writerow([class_name, count, classification, path])
    
    # JSON SYNTHÈSE
    json_file = output_dir / 'dataset_analysis.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    return str(csv_file), str(json_file)

def print_report(classes, stats):
    """Afficher rapport console"""
    print("\n" + "="*80)
    print("📊 RAPPORT ANALYSE DATASET - FRAMEWORK D'EXCELLENCE")
    print("="*80 + "\n")
    
    print(f"✓ Total images: {stats['total_images']:,}")
    print(f"✓ Total classes: {stats['total_classes']}")
    print(f"✓ Min par classe: {stats['min_class']}")
    print(f"✓ Max par classe: {stats['max_class']}")
    print(f"✓ Moyenne: {stats['avg_class']:.1f} images/classe")
    print(f"✓ Ratio déséquilibre: {stats['imbalance_ratio']:.1f}x (CRITIQUE si > 10)\n")
    
    # Breakdown
    print("📌 CLASSIFICATION DES CLASSES:")
    print(f"   🔴 UNUSABLE (< 100): {len(stats['problematic'])} classes")
    if len(stats['problematic']) > 0 and len(stats['problematic']) <= 10:
        for cls, count in stats['problematic']:
            print(f"      - {cls}: {count}")
    elif len(stats['problematic']) > 10:
        print(f"      (Afficher les {len(stats['problematic'])} dans le CSV)\n")
    
    print(f"   🟠 LOW_DATA (100-300): {len(stats['low_data'])} classes")
    print(f"   🟡 WEAK (300-500): {len(stats['weak'])} classes")
    print(f"   🟢 BALANCED (500-3000): {len(stats['balanced'])} classes")
    print(f"   🔵 OVERSIZED (> 3000): {len(stats['oversized'])} classes")
    
    if len(stats['oversized']) > 0:
        print(f"\n   Oversized classes (à réduire):")
        for cls, count in sorted(stats['oversized'], key=lambda x: -x[1]):
            print(f"      - {cls}: {count}")
    
    print("\n" + "="*80)
    print("🎯 RECOMMANDATIONS D'ACTION:\n")
    
    actions = []
    
    if len(stats['problematic']) > 0:
        actions.append(f"1️⃣  SUPPRIMER {len(stats['problematic'])} classes < 100 images (inutilisables)")
    else:
        actions.append(f"1️⃣  ✓ Pas de classes < 100 images")
    
    if len(stats['low_data']) > 0:
        actions.append(f"2️⃣  AUGMENTER {len(stats['low_data'])} classes 100-300 (données faibles)")
    else:
        actions.append(f"2️⃣  ✓ Pas de classes 100-300")
    
    if stats['imbalance_ratio'] > 10:
        actions.append(f"3️⃣  RÉDUIRE {len(stats['oversized'])} classes > 3000 (déséquilibre {stats['imbalance_ratio']:.1f}x)")
    else:
        actions.append(f"3️⃣  ✓ Déséquilibre acceptable ({stats['imbalance_ratio']:.1f}x)")
    
    if len(stats['weak']) > 10:
        actions.append(f"4️⃣  CONSIDÉRER augmentation pour {len(stats['weak'])} classes 300-500")
    else:
        actions.append(f"4️⃣  ✓ Peu de classes faibles (300-500)")
    
    for action in actions:
        print(action)
    
    print("\n" + "="*80)

if __name__ == '__main__':
    root = r'C:\smarts-n-yieldpredict.git\Diseasedataset'
    output_dir = r'C:\smarts-n-yieldpredict.git\dataset_analysis'
    
    print("\n🔍 Analyse rapide du dataset...\n")
    
    classes, details, stats = analyze_dataset_fast(root)
    csv_path, json_path = generate_reports(classes, details, stats, output_dir)
    print_report(classes, stats)
    
    print(f"\n📄 Rapports générés:")
    print(f"   ✓ CSV: {csv_path}")
    print(f"   ✓ JSON: {json_path}\n")
