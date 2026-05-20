import os
import re
from pathlib import Path

dataset_path = r"C:\Users\moham\Videos\Moh"

print("🔧 NETTOYAGE DES NOMS DE DOSSIERS\n")

# Fonction pour nettoyer les noms
def clean_folder_name(name):
    # Supprimer les caractères spéciaux et parenthèses
    cleaned = re.sub(r'[()*]', '', name)
    # Remplacer espaces multiples par underscore
    cleaned = re.sub(r'\s+', '_', cleaned)
    # Supprimer underscores multiples
    cleaned = re.sub(r'_+', '_', cleaned)
    return cleaned.strip('_')

# Lister les dossiers actuels
folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]

print(f"Dossiers trouvés: {len(folders)}\n")

renamed = 0
for folder in folders:
    old_path = os.path.join(dataset_path, folder)
    cleaned_name = clean_folder_name(folder)

    if cleaned_name != folder:
        new_path = os.path.join(dataset_path, cleaned_name)

        # Vérifier si le nom nettoyé existe déjà
        if os.path.exists(new_path):
            print(f"⚠️ Conflit: {folder} → {cleaned_name} (existe déjà)")
            # Fusionner
            for file in os.listdir(old_path):
                src = os.path.join(old_path, file)
                dst = os.path.join(new_path, file)
                if not os.path.exists(dst):
                    try:
                        os.rename(src, dst)
                    except Exception as e:
                        print(f"   Erreur: {e}")
            try:
                os.rmdir(old_path)
                print(f"   ✅ Fusionné et supprimé")
            except:
                print(f"   ⚠️ Impossible de supprimer {old_path}")
        else:
            try:
                os.rename(old_path, new_path)
                print(f"✅ {folder} → {cleaned_name}")
                renamed += 1
            except Exception as e:
                print(f"❌ Erreur renommage {folder}: {e}")
    else:
        print(f"✓ {folder} (déjà propre)")

print(f"\n📊 RÉSUMÉ: {renamed} dossiers renommés")