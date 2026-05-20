from pathlib import Path

def remove_empty_folders(base_dir):
    """Supprimer les dossiers vides"""
    base_dir = Path(base_dir)
    removed = 0
    
    for folder in base_dir.iterdir():
        if folder.is_dir():
            try:
                # Vérifier si le dossier est vide
                if not any(folder.iterdir()):
                    folder.rmdir()
                    removed += 1
                    print(f"Supprimé: {folder.name}")
            except Exception as e:
                print(f"Erreur avec {folder.name}: {e}")
    
    print(f"\nTotal dossiers vides supprimés: {removed}")

if __name__ == "__main__":
    remove_empty_folders(r"C:\smarts-n-yieldpredict.git\Data traiter_cleaned")