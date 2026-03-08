import os

# Chemin du dossier
dossier = r"C:\Users\moham\Documents\Moh"

# Chemin du fichier de sortie
fichier_sortie = r"C:\Users\moham\Documents\Moh\noms_fichiers_json.txt"

# Lister tous les fichiers dans le dossier
fichiers = os.listdir(dossier)

# Filtrer uniquement les fichiers .json
json_files = [f for f in fichiers if f.endswith('.json')]

# Enregistrer dans le fichier txt
with open(fichier_sortie, 'w', encoding='utf-8') as f:
    for fichier in json_files:
        f.write(fichier + '\n')

print(f"{len(json_files)} fichiers JSON enregistrés dans {fichier_sortie}")