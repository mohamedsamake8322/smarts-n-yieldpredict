import os
import shutil

# Chemin absolu de votre dossier de classes sous Windows
BASE_DIR = r"C:\Downloads\Classes"

def restructure_plant_classes(base_dir):
    if not os.path.exists(base_dir):
        print(f"❌ Le dossier spécifié n'existe pas : {base_dir}")
        return

    print("🔄 Renommage des classes et création de l'arborescence Health/Healthy...\n")
    
    # Étape 1 : Parcourir et renommer les dossiers principaux (enlever "plant")
    for folder_name in os.listdir(base_dir):
        old_path = os.path.join(base_dir, folder_name)
        
        if os.path.isdir(old_path):
            # Supprime "plant" ou "Plant" et nettoie les espaces restants
            new_folder_name = folder_name.replace("plant", "").replace("Plant", "").strip()
            new_path = os.path.join(base_dir, new_folder_name)
            
            # Renommer le dossier si le nom a changé
            if old_path != new_path:
                # Si le dossier de destination existe déjà, on fusionne
                if os.path.exists(new_path):
                    for item in os.listdir(old_path):
                        shutil.move(os.path.join(old_path, item), os.path.join(new_path, item))
                    os.rmdir(old_path)
                else:
                    os.rename(old_path, new_path)
                print(f"✅ Classe renommée : {folder_name} ➡️ {new_folder_name}")
            else:
                new_path = old_path

            # Étape 2 : Créer la structure interne Health/Healthy et déplacer les images
            target_subfolder = os.path.join(new_path, "Health", "Healthy")
            
            # Lister les images à déplacer AVANT de créer le dossier cible (pour éviter une boucle infinie)
            items_to_move = [
                item for item in os.listdir(new_path) 
                if os.path.isfile(os.path.join(new_path, item))
            ]
            
            if items_to_move:
                # Création des sous-dossiers imbriqués
                os.makedirs(target_subfolder, exist_ok=True)
                
                # Déplacement des fichiers
                for file_name in items_to_move:
                    src_file = os.path.join(new_path, file_name)
                    dst_file = os.path.join(target_subfolder, file_name)
                    
                    # Sécurité doublon
                    if os.path.exists(dst_file):
                        name, ext = os.path.splitext(file_name)
                        dst_file = os.path.join(target_subfolder, f"{name}_copie{ext}")
                        
                    shutil.move(src_file, dst_file)
                
                print(f"   ↳ 📁 {len(items_to_move)} images déplacées dans {new_folder_name}\\Health\\Healthy")

    print("\n✨ Restructuration complète terminée !")

# Lancement du script
restructure_plant_classes(BASE_DIR)
