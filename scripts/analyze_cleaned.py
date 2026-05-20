from pathlib import Path

base_dir = Path(r"C:\smarts-n-yieldpredict.git\Data traiter_cleaned")

print("📊 DISTRIBUTION DES IMAGES NETTOYÉES\n" + "=" * 70)
print(f"{'Classe':<45} {'Images':<10}")
print("-" * 70)

total_images = 0
class_sizes = []

for class_dir in sorted(base_dir.iterdir()):
    if class_dir.is_dir():
        count = len(list(class_dir.glob("*.*")))
        total_images += count
        class_sizes.append((class_dir.name, count))
        print(f"{class_dir.name:<45} {count:<10}")

print("-" * 70)
print(f"\n✅ RÉSUMÉ")
print(f"Total images: {total_images}")
print(f"Nombre de classes: {len(class_sizes)}")
print(f"Moyenne par classe: {total_images / len(class_sizes):.0f}")
print(f"Min: {min(c[1] for c in class_sizes)} | Max: {max(c[1] for c in class_sizes)}")

# Classes faibles
weak = [c for c in class_sizes if c[1] < 100]
if weak:
    print(f"\n⚠️  Classes faibles (< 100 images):")
    for name, count in sorted(weak, key=lambda x: x[1]):
        print(f"  - {name}: {count}")
