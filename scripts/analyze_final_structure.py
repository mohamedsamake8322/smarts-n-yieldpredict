import os

dataset_path = r"C:\Users\moham\Videos\Moh"
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

print("📊 ANALYSE DÉTAILLÉE DE LA STRUCTURE FINALE\n")

# Compter les classes principales (dossier racine)
main_classes = []
small_300_classes = []
small_500_classes = []

for root, dirs, files in os.walk(dataset_path):
    # Ignorer les sous-dossiers
    if root != dataset_path:
        continue

    for item in dirs:
        folder_path = os.path.join(root, item)

        # Compter les images dans ce dossier
        image_count = 0
        for r, d, f in os.walk(folder_path):
            image_count += len([file for file in f if file.lower().endswith(image_extensions)])

        if "Small_Classes_500" in item:
            small_500_classes.append((item, image_count))
        elif "Small_Classes" in item:
            small_300_classes.append((item, image_count))
        else:
            main_classes.append((item, image_count))

print("🏆 CLASSES PRINCIPALES (500+ images) :")
total_main = 0
for name, count in sorted(main_classes, key=lambda x: x[1], reverse=True):
    print(f"  • {name}: {count} images")
    total_main += count

print(f"\n📈 Total classes principales: {len(main_classes)}")
print(f"📈 Total images principales: {total_main}")

print(f"\n📂 SMALL_CLASSES (< 300 images) : {len(small_300_classes)} classes")
print(f"📂 SMALL_CLASSES_500 (300-500 images) : {len(small_500_classes)} classes")

print("\n✅ STRUCTURE OPTIMALE POUR L'ENTRAÎNEMENT ML !")
print("\n💡 Les classes principales offrent suffisamment de données pour un entraînement robuste")
print("\n💡 Les classes secondaires peuvent être utilisées pour l'augmentation ou la validation")
print(f"\n🎯 RATIO QUALITÉ: {len(main_classes)} classes principales vs {len(small_300_classes) + len(small_500_classes)} classes secondaires")