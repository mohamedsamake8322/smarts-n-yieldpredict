import os
import requests
from googlesearch import search
import time
import re

# Liste complète des 109 pathologies
diseases = [
    "Alfalfa_Leaf_Mosaic_Virus", "Alluim_Leaf_Bacterial_Soft_Rot", "Alternaria black molds stem cankers (Genus Alternaria Nees)",
    "Alternaria_Leaf_Spot", "Angular_Leaf_Spot_Cucumber", "Anthracnose (Colletotrichum orbiculare (Berk. & Mo",
    "Aphids", "Apple_Scab", "Aspergillus ear and kernel rot (Aspergillus flavus Maize)",
    "Banana_Fruit_Anthracnose", "Banana_Leaf_Anthracnose", "Banana_Leaf_Black_Sigatoka",
    "Banana_Leaf_Bunchy_Top_Virus", "Banana_Leaf_Streak_Virus", "Banana_Leaf_Yellow_Sigatoka",
    "Bean_Leaf_Bacterial_Blight", "Bean_Leaf_Rust", "Bell pepper_Fruit_Blossom_End_Rot",
    "Bread mold (Rhizopus stolonifer (Ehrenb.) Vuill.)", "Broad mite (Polyphagotarsonemus latus (Banks))",
    "Brown_Rot_Fruit", "Cabbage_Leaf_Bacterial_Blight", "Cabbage_Leaf_Bacterial_Soft_Rot",
    "Carota_ Alternaria_Fruit_Spot", "Carrot_Root_Fly", "Cereals_Leaf_Bacterial_Spot",
    "Cherry_Leaf_Spot", "Choanephora fruit rot (Choanephora cucurbitarum", "Citrillus_Fruit_Blossom_End_Rot",
    "Citrullus_Bacterial_Fruit_Blotch", "Citrullus_Bacterial_Leaf_Blotch", "Citrus_Canker",
    "Citrus_Leaf_Greening", "Corn smut (Ustilago maydis)", "Corynespora leaf spot (Corynespora cassiicola",
    "Cotton bollworm (Helicoverpa armigera)", "Cotton_Adult_Bollworm", "Cotton_Larve_Bollworm",
    "Crucifers_Xanthomonas_campestris_pv_campestris_Black_rot)", "Cucurbit_Leaf_Downy_Mildew",
    "Curculionoidea (Weevil)", "Damages_Fruit_Bollworm", "fall armyworm (Spodoptera frugiperda) Dammage",
    "fall armyworm (Spodoptera frugiperda) Larve", "fall armyworm (Spodoptera frugiperda) Larve adults",
    "Flower blights (Genus Choanephora Curr.)", "Fusarium_Cereal_Head_Blight", "Geotrichum_Rot_Cucurbitaceae_Fruit",
    "Grape_Leaf_Esca", "Grape_Leaf_Leafroll_Virus", "Hosta_Leaf_Virus_X", "kudzu bug (Megacopta cribraria (Fabricius))",
    "Late_Blight_Fruit", "Late_Blight_Leaf", "Leaf_Bacterial_Wilt", "Leaf_Curl_Virus", "Leaf_Spurge_Flea_Beetle",
    "lesser cornstalk borer (Elasmopalpus lignosella)", "Lettuce Chlorosis Virus (LCV) (Crinivirus Lettuce Chlorosis Virus)",
    "Longhorn_Beetle", "Loose smut of wheat or barley (Ustilago tritici (Pers.) Rostr.)", "Maize_Tar_Spot",
    "Mealybug", "Northern_Maize_Leaf_Blight", "Penicillium fungi (Genus Penicillium Link)",
    "Phytophthora_Blight_Cucurbitaceae", "Potyvirus_Mosaic_Virus", "Pythium diseases (Genus Pythium Pringsh.)",
    "Red_Spider_Mite", "Rhizopus soft rots (Genus Rhizopus Ehrenb.)", "rhododendron stem borer (Oberea myops",
    "Rice_Leaf_Blast", "root rotdamping off (Genus Rhizoctonia DC.)", "root-knot nematode (Genus Meloidogyne)",
    "Sclerotinia sclerotiorum_Fruit_Rot", "Sclerotinia sclerotiorum_Leaf_Rot", "Sclerotinia sclerotiorum_Stem_Rot",
    "Sclerotinia timber rot (Sclerotinia sclerotiorum)", "Solanaceae_Bacterial_Soft_Rot", "Solanaceae_Bacterial_Spot",
    "Southern red mite (Oligonychus ilicis)", "Southern root-knot nematode", "Soybean_Leaf_Halo_Blight",
    "Soybean_Leaf_Mosaic_Virus", "Soybean_Leaf_Rust", "Soybean_Tospovirus_Leaf_Vein_Necrosis_Virus",
    "Sphinx_Moth_Adult", "Sphinx_Moth_Larve_Damages", "Spider_Mite_Leaf_Damages", "Spotted_Wilt_Virus",
    "stem-boring weevil (Mecinus janthinus Germar)", "Stem_Borer_Leaf_Damages", "Sweet_Potato_Fruit_Feathery_Mottle_Virus",
    "Sweet_Potato_Leaf_Feathery_Mottle_Virus", "Target_Spot_Leaf", "Thrips_Leaf_Damages",
    "Tobacco_Leaf_Etch_Virus", "Tobacco_Leaf_Ringspot_Virus", "Tomato_Early_Blight",
    "Tomato_Fruit_Bacterial_Canker", "Tomato_Fruit_Blossom_End_Rot", "Tomato_Leaf_Bacterial_Canker",
    "Tomato_Leaf_Mold", "Tomato_Leaf_Mosaic_Virus", "Tomato_Leaf_Septoria", "Tomato_Yellow_Leaf_Curl_Virus",
    "Twospotted spider mite (Tetranychus urticae Koch)", "Wheat_Leaf_Streak_Mosaic_Virus", "Whiteflies_Leaf"
]

download_folder = "maladies_agricoles_pdf"
if not os.path.exists(download_folder):
    os.makedirs(download_folder)

def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "", name).replace(" ", "_")

def download_pdf(query):
    # Requête ciblée pour des résultats académiques/professionnels
    search_query = f'"{query.replace("_", " ")}" filetype:pdf site:edu OR site:gov OR site:apsnet.org'
    print(f"\n--- Recherche ({diseases.index(query)+1}/{len(diseases)}) : {query}")
    
    try:
        # On parcourt les 3 premiers résultats pour maximiser les chances de trouver un PDF valide
        for url in search(search_query, num_results=3):
            if ".pdf" in url.lower():
                response = requests.get(url, timeout=15, stream=True)
                if response.status_code == 200:
                    filename = sanitize_filename(query) + ".pdf"
                    path = os.path.join(download_folder, filename)
                    with open(path, 'wb') as f:
                        f.write(response.content)
                    print(f"    [OK] Téléchargé depuis : {url}")
                    return True
        print(f"    [!] Aucun PDF trouvé.")
        return False
    except Exception as e:
        print(f"    [ERREUR] {e}")
        return False

# Lancement du processus
for disease in diseases:
    download_pdf(disease)
    # Pause de 5 secondes pour éviter le bannissement IP par Google
    time.sleep(5)

print("\nTraitement terminé. Les fichiers sont dans le dossier 'maladies_agricoles_pdf'.")
