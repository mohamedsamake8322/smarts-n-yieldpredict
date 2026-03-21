import os

folder_main = r"C:\smarts-n-yieldpredict.git\BLIP2"
folder_ha = r"C:\smarts-n-yieldpredict.git\BLIP2_i18n\am"

# Récupérer les fichiers
files_main = set(os.listdir(folder_main))
files_ha = set(os.listdir(folder_ha))

# Compter
print(f"📁 BLIP2 (source) : {len(files_main)} fichiers")
print(f"📁 ha (traduction) : {len(files_ha)} fichiers")

# Trouver les fichiers manquants dans ha
missing_in_ha = files_main - files_ha

print(f"\n❌ Fichiers manquants dans ha : {len(missing_in_ha)}\n")

for f in sorted(missing_in_ha):
    print(f"➜ {f}")