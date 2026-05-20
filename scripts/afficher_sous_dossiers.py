"""
Script pour afficher les noms des sous-dossiers
dans le répertoire Image Data base
"""

import os
from pathlib import Path

def afficher_sous_dossiers():
    """Affiche les noms des sous-dossiers dans le répertoire spécifié"""

    # Chemin du répertoire
    chemin = r"C:\Users\moham\Pictures\Image Data base"

    # Vérifier si le répertoire existe
    if not os.path.exists(chemin):
        print(f"Le répertoire '{chemin}' n'existe pas.")
        return

    if not os.path.isdir(chemin):
        print(f"'{chemin}' n'est pas un répertoire.")
        return

    # Lister les sous-dossiers
    try:
        sous_dossiers = [d for d in os.listdir(chemin)
                        if os.path.isdir(os.path.join(chemin, d))]

        if not sous_dossiers:
            print(f"Aucun sous-dossier trouvé dans '{chemin}'.")
            return

        print(f"Sous-dossiers dans '{chemin}':")
        print("-" * 50)

        for dossier in sorted(sous_dossiers):
            print(f"  - {dossier}")

        print("-" * 50)
        print(f"Total: {len(sous_dossiers)} sous-dossier(s)")

    except PermissionError:
        print(f"Permission refusée pour accéder à '{chemin}'.")
    except Exception as e:
        print(f"Erreur lors de l'accès au répertoire: {e}")

def afficher_sous_dossiers_pathlib():
    """Version alternative utilisant pathlib"""

    chemin = Path(r"C:\Users\moham\Pictures\Image Data base")

    if not chemin.exists():
        print(f"Le répertoire '{chemin}' n'existe pas.")
        return

    if not chemin.is_dir():
        print(f"'{chemin}' n'est pas un répertoire.")
        return

    try:
        sous_dossiers = [d.name for d in chemin.iterdir() if d.is_dir()]

        if not sous_dossiers:
            print(f"Aucun sous-dossier trouvé dans '{chemin}'.")
            return

        print(f"Sous-dossiers dans '{chemin}':")
        print("-" * 50)

        for dossier in sorted(sous_dossiers):
            print(f"  - {dossier}")

        print("-" * 50)
        print(f"Total: {len(sous_dossiers)} sous-dossier(s)")

    except PermissionError:
        print(f"Permission refusée pour accéder à '{chemin}'.")
    except Exception as e:
        print(f"Erreur lors de l'accès au répertoire: {e}")

if __name__ == "__main__":
    print("=== AFFICHAGE DES SOUS-DOSSIERS ===")
    print()

    # Utiliser la version pathlib (plus moderne)
    afficher_sous_dossiers_pathlib()