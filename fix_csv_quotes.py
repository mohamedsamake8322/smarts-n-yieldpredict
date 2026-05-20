import csv

# Chemin du fichier CSV problématique
csv_path = r"C:\Users\moham\Videos\mapping.csv"
fixed_csv_path = r"C:\Users\moham\Videos\mapping_fixed.csv"

print("🔧 CORRECTION DU CSV - AJOUT DE GUILLEMETS\n")

with open(csv_path, 'r', encoding='utf-8') as infile, \
     open(fixed_csv_path, 'w', newline='', encoding='utf-8') as outfile:

    reader = csv.reader(infile)
    writer = csv.writer(outfile, quoting=csv.QUOTE_MINIMAL)

    line_num = 0
    for row in reader:
        line_num += 1

        # Vérifier si cette ligne a le bon nombre de colonnes
        if len(row) != 2:
            print(f"⚠️ Ligne {line_num}: {len(row)} colonnes trouvées (attendu: 2)")
            print(f"   Contenu: {row}")

            # Essayer de fusionner les colonnes si nécessaire
            if len(row) > 2:
                # Supposer que la première colonne contient la virgule
                fixed_row = [row[0] + ',' + row[1], row[2]]
                writer.writerow(fixed_row)
                print(f"   ✅ Corrigé: {fixed_row}")
                continue

        # Écrire la ligne normalement
        writer.writerow(row)

print(f"\n✅ CSV corrigé sauvegardé: {fixed_csv_path}")
print("💡 Utilisez maintenant ce fichier dans votre script 0.py")