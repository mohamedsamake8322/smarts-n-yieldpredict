"""
📊 Dataset Analysis & Balancing Verification Script
====================================================
Use this script to analyze your dataset BEFORE training.
Shows class distribution, imbalance ratio, and balancing strategy effectiveness.

Run this locally or in a Kaggle notebook before the main training.
"""

import os
import json
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_DIR = Path('/path/to/your/dataset')  # Change this to your data directory

# ============================================================================
# DISCOVER CLASSES
# ============================================================================
def discover_classes(root_dir):
    """Find all classes and their images."""
    class_to_images = defaultdict(list)
    root_path = Path(root_dir)
    
    for ext in ['jpg', 'png', 'JPG', 'PNG']:
        for img_path in sorted(root_path.rglob(f'*.{ext}')):
            class_name = img_path.parent.name
            class_to_images[class_name].append(str(img_path))
    
    return dict(class_to_images)

# ============================================================================
# ANALYZE DATASET
# ============================================================================
print(f"📂 Analyzing dataset in: {DATA_DIR}")

class_to_images = discover_classes(DATA_DIR)
class_counts = {k: len(v) for k, v in class_to_images.items()}

if not class_counts:
    print("❌ No images found. Check your DATA_DIR path.")
    exit(1)

num_classes = len(class_counts)
total_images = sum(class_counts.values())
sorted_classes = sorted(class_counts.items(), key=lambda x: x[1])

print(f"\n{'='*70}")
print(f"📊 DATASET STATISTICS")
print(f"{'='*70}")
print(f"Total classes: {num_classes}")
print(f"Total images: {total_images}")
print(f"Avg images/class: {total_images/num_classes:.0f}")

# Statistics
counts = list(class_counts.values())
print(f"\n📈 Class Distribution:")
print(f"   Min: {min(counts)} images ({sorted_classes[0][0]})")
print(f"   Max: {max(counts)} images ({sorted_classes[-1][0]})")
print(f"   Median: {sorted(counts)[num_classes//2]} images")
print(f"   Mean: {np.mean(counts):.0f} images")
print(f"   Std Dev: {np.std(counts):.0f} images")
print(f"   Imbalance Ratio: {max(counts) / min(counts):.1f}x")

# Underrepresented classes
underrep_threshold = 300
underrep_classes = {k: v for k, v in class_counts.items() if v < underrep_threshold}
print(f"\n⚠️  Underrepresented classes (< {underrep_threshold} images): {len(underrep_classes)}")

if underrep_classes:
    sorted_underrep = sorted(underrep_classes.items(), key=lambda x: x[1])
    print(f"   Examples:")
    for class_name, count in sorted_underrep[:10]:
        print(f"     - {class_name}: {count} images")

# ============================================================================
# COMPUTE CLASS WEIGHTS
# ============================================================================
print(f"\n⚖️  CLASS WEIGHTS (for weighted sampler):")
class_names = sorted(class_counts.keys())
class_weights = 1.0 / (np.array([class_counts[c] for c in class_names]) + 1e-8)
class_weights = class_weights / class_weights.sum() * len(class_names)

min_weight_class = class_names[np.argmin(class_weights)]
max_weight_class = class_names[np.argmax(class_weights)]
min_weight = class_weights.min()
max_weight = class_weights.max()

print(f"   Weight range: {min_weight:.4f} - {max_weight:.4f}")
print(f"   Lightest class ({min_weight_class}): {min_weight:.4f}")
print(f"   Heaviest class ({max_weight_class}): {max_weight:.4f}")
print(f"   Weight ratio: {max_weight/min_weight:.1f}x")

# ============================================================================
# AUGMENTATION STRATEGY
# ============================================================================
print(f"\n🎨 AUGMENTATION STRATEGY:")
print(f"   Classes < {underrep_threshold} images → AGGRESSIVE augmentation")
print(f"   Other classes → STANDARD augmentation")

aggr_aug_count = len(underrep_classes)
print(f"   Affected: {aggr_aug_count} classes")

# ============================================================================
# BALANCED BATCH SIMULATION
# ============================================================================
print(f"\n🎲 BALANCED BATCH SIMULATION:")
print(f"   (What your batches will look like with WeightedRandomSampler)")

# Simulate batch composition
batch_size = 16
n_simulations = 10
class_indices = {i: class_names[i] for i in range(len(class_names))}

simulated_batches = []
for _ in range(n_simulations):
    # Sample according to weights
    batch_indices = np.random.choice(
        len(class_names), 
        size=batch_size, 
        p=class_weights / class_weights.sum(),
        replace=True
    )
    batch_classes = [class_names[i] for i in batch_indices]
    simulated_batches.append(batch_classes)

# Count distribution in first simulated batch
first_batch_counts = Counter(simulated_batches[0])
print(f"\n   Example batch (first 16 samples):")
for class_name, count in first_batch_counts.most_common():
    pct = count / batch_size * 100
    print(f"     - {class_name}: {count} samples ({pct:.0f}%)")

# ============================================================================
# VISUALIZATION
# ============================================================================
print(f"\n📊 Generating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Class distribution histogram
ax = axes[0, 0]
counts_sorted = sorted(class_counts.values())
ax.hist(counts_sorted, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax.axvline(np.mean(counts_sorted), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(counts_sorted):.0f}')
ax.axvline(np.median(counts_sorted), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(counts_sorted):.0f}')
ax.set_xlabel('Images per Class')
ax.set_ylabel('Number of Classes')
ax.set_title('Class Distribution Histogram')
ax.legend()
ax.grid(alpha=0.3)

# 2. Top 20 largest classes
ax = axes[0, 1]
top_20 = sorted_classes[-20:]
class_names_top = [c[0][:20] for c in top_20]
counts_top = [c[1] for c in top_20]
bars = ax.barh(range(len(class_names_top)), counts_top, color='coral')
ax.set_yticks(range(len(class_names_top)))
ax.set_yticklabels(class_names_top, fontsize=9)
ax.set_xlabel('Number of Images')
ax.set_title('Top 20 Largest Classes')
ax.invert_yaxis()
for i, bar in enumerate(bars):
    ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2, 
            f'{int(bar.get_width())}', va='center', fontsize=8)
ax.grid(axis='x', alpha=0.3)

# 3. Top 20 smallest classes
ax = axes[1, 0]
bottom_20 = sorted_classes[:20]
class_names_bot = [c[0][:20] for c in bottom_20]
counts_bot = [c[1] for c in bottom_20]
bars = ax.barh(range(len(class_names_bot)), counts_bot, color='lightcoral')
ax.set_yticks(range(len(class_names_bot)))
ax.set_yticklabels(class_names_bot, fontsize=9)
ax.set_xlabel('Number of Images')
ax.set_title('Bottom 20 Smallest Classes')
ax.invert_yaxis()
for i, bar in enumerate(bars):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
            f'{int(bar.get_width())}', va='center', fontsize=8)
ax.grid(axis='x', alpha=0.3)

# 4. Class weights distribution
ax = axes[1, 1]
ax.hist(class_weights, bins=50, color='lightgreen', alpha=0.7, edgecolor='black')
ax.axvline(np.mean(class_weights), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(class_weights):.4f}')
ax.set_xlabel('Class Weight (for weighted sampler)')
ax.set_ylabel('Number of Classes')
ax.set_title('Class Weight Distribution')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('dataset_analysis.png', dpi=100, bbox_inches='tight')
print(f"   ✅ Saved: dataset_analysis.png")

# ============================================================================
# REPORT
# ============================================================================
report = {
    "num_classes": num_classes,
    "total_images": total_images,
    "class_distribution": {
        "min": min(counts),
        "max": max(counts),
        "mean": float(np.mean(counts)),
        "median": float(np.median(counts)),
        "std": float(np.std(counts)),
        "imbalance_ratio": float(max(counts) / min(counts)),
    },
    "underrepresented_classes": {
        "count": len(underrep_classes),
        "threshold": underrep_threshold,
        "examples": dict(sorted_underrep[:10]) if underrep_classes else {},
    },
    "class_weights": {
        "min": float(min_weight),
        "max": float(max_weight),
        "weight_ratio": float(max_weight / min_weight),
    },
    "recommendations": [
        "✅ Use WeightedRandomSampler for balanced batches",
        "✅ Use weighted CrossEntropyLoss to penalize errors on small classes",
        "✅ Apply aggressive augmentation to classes < 300 images",
        f"⚠️  Watch out for classes with < 50 images: {len([c for c in counts if c < 50])} classes",
        f"💡 Consider combining classes with similar diseases (e.g., different rust types)",
    ]
}

# Save report
report_path = 'dataset_analysis_report.json'
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)
print(f"   ✅ Saved: {report_path}")

# ============================================================================
# SUMMARY
# ============================================================================
print(f"\n{'='*70}")
print(f"✅ ANALYSIS COMPLETE")
print(f"{'='*70}")
print(f"\n📌 KEY FINDINGS:")
print(f"   • Severe imbalance detected ({max(counts) / min(counts):.0f}x ratio)")
print(f"   • {len(underrep_classes)} classes need special attention (< {underrep_threshold} images)")
print(f"   • Balanced training strategy STRONGLY RECOMMENDED")
print(f"\n💡 RECOMMENDATIONS:")
for rec in report['recommendations']:
    print(f"   {rec}")
print(f"\n📁 Generated files:")
print(f"   - dataset_analysis.png (visualization)")
print(f"   - dataset_analysis_report.json (detailed report)")
