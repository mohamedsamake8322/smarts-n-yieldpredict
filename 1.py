import os
import csv

# ==============================
# CONFIGURATION
# ==============================
TRAIN_DIR = r"C:\smarts-n-yieldpredict.git\dataset_final\train"
MIN_IMAGES = 80   # 🔥 change ici le seuil
EXPORT_CSV = True
EXPORT_TXT = True

# ==============================
# ANALYSE
# ==============================
class_counts = {}

for class_name in os.listdir(TRAIN_DIR):
    class_path = os.path.join(TRAIN_DIR, class_name)

    if os.path.isdir(class_path):
        images = [
            f for f in os.listdir(class_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
        class_counts[class_name] = len(images)

# ==============================
# FILTRAGE
# ==============================
low_classes = {
    k: v for k, v in class_counts.items()
    if v < MIN_IMAGES
}

# ==============================
# AFFICHAGE
# ==============================
print("\n🔎 CLASSES SOUS LE SEUIL :", MIN_IMAGES)
print("-------------------------------------------------")

if not low_classes:
    print("✅ Toutes les classes respectent le seuil.")
else:
    for cls, count in sorted(low_classes.items(), key=lambda x: x[1]):
        print(f"📁 {cls} → {count} images")

print("\n📊 RÉSUMÉ")
print("-------------------------------------------------")
print("Nombre total de classes :", len(class_counts))
print("Classes sous seuil :", len(low_classes))

# ==============================
# EXPORT TXT
# ==============================
if EXPORT_TXT:
    with open("classes_sous_seuil.txt", "w", encoding="utf-8") as f:
        for cls, count in sorted(low_classes.items(), key=lambda x: x[1]):
            f.write(f"{cls},{count}\n")

# ==============================
# EXPORT CSV
# ==============================
if EXPORT_CSV:
    with open("classes_sous_seuil.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Classe", "Nombre_images"])
        for cls, count in sorted(low_classes.items(), key=lambda x: x[1]):
            writer.writerow([cls, count])

print("\n✅ Script terminé.")
