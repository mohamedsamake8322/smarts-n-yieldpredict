  
import os
import shutil

source_folder = r"C:\Downloads\BLIP2_i18n-20260321T080948Z-1-001\BLIP2_i18n\sw"
duplicates_folder = os.path.join(source_folder, "duplicates")

# Créer le dossier duplicates s'il n'existe pas
os.makedirs(duplicates_folder, exist_ok=True)

moved_count = 0

for file in os.listdir(source_folder):
    if "(1)" in file:
        src_path = os.path.join(source_folder, file)
        dst_path = os.path.join(duplicates_folder, file)
        
        shutil.move(src_path, dst_path)
        print(f"➜ Déplacé : {file}")
        moved_count += 1

print(f"\n✅ Total déplacés : {moved_count}")