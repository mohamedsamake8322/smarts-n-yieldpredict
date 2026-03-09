import os
import requests
import time
import random
import re
from googlesearch import search

DOWNLOAD_FOLDER = "maladies_agricoles_pdf"
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
LOG_FILE = "downloaded.txt"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
]

def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "", name).replace(" ", "_")

def already_downloaded(name):
    if not os.path.exists(LOG_FILE): return False
    with open(LOG_FILE, "r") as f:
        return name in f.read()

def log_download(name):
    with open(LOG_FILE, "a") as f:
        f.write(name + "\n")

def download_file(url, filename):
    try:
        # On vérifie l'extension avant de télécharger
        if not url.lower().endswith('.pdf'):
            return False
            
        response = requests.get(url, headers={"User-Agent": random.choice(USER_AGENTS)}, timeout=15, stream=True)
        if response.status_code == 200 and "pdf" in response.headers.get("Content-Type", "").lower():
            path = os.path.join(DOWNLOAD_FOLDER, filename)
            with open(path, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return True
    except:
        return False
    return False

def download_pdf(query, index, total):
    if already_downloaded(query):
        print(f"[{index}/{total}] Déjà fait : {query}")
        return

    # REQUÊTE ASSOUPLIE : On enlève les guillemets et on élargit les sites
    search_query = f"{query} disease factsheet filetype:pdf site:edu OR site:gov OR site:org"
    
    print(f"\n[{index}/{total}] Recherche : {query}")
    try:
        # On cherche dans les 8 premiers résultats pour plus de chances
        results = search(search_query, num_results=8)
        for url in results:
            # On tente le téléchargement si l'URL semble être un PDF
            if ".pdf" in url.lower():
                if download_file(url, sanitize_filename(query) + ".pdf"):
                    print(f"    -> Succès : {url[:70]}...")
                    log_download(query)
                    return
        print("    [!] Aucun PDF trouvé.")
    except Exception as e:
        print(f"    [!] Erreur recherche : {e}")
        time.sleep(20) # Pause longue si Google nous bloque

# --- Liste des maladies (Gardée telle quelle) ---
diseases = [
    "Alfalfa Leaf Mosaic Virus",
    "Allium Leaf Bacterial Soft Rot",
    "Alternaria Black Mold Stem Cankers",
    "Alternaria Leaf Spot",
    "Angular Leaf Spot Cucumber",
    "Anthracnose Colletotrichum orbiculare",
    "Aphids",
    "Apple Scab",
    "Aspergillus Ear and Kernel Rot",
    "Banana Fruit Anthracnose",
    "Banana Leaf Anthracnose",
    "Banana Leaf Black Sigatoka",
    "Banana Leaf Bunchy Top Virus",
    "Banana Leaf Streak Virus",
    "Banana Leaf Yellow Sigatoka",
    "Bean Leaf Bacterial Blight",
    "Bean Leaf Rust",
    "Bell Pepper Fruit Blossom End Rot",
    "Bread Mold Rhizopus stolonifer",
    "Broad Mite Polyphagotarsonemus latus",
    "Brown Rot Fruit",
    "Cabbage Leaf Bacterial Blight",
    "Cabbage Leaf Bacterial Soft Rot",
    "Carrot Alternaria Fruit Spot",
    "Carrot Root Fly",
    "Cereals Leaf Bacterial Spot",
    "Cherry Leaf Spot",
    "Choanephora Fruit Rot",
    "Citrullus Fruit Blossom End Rot",
    "Citrullus Bacterial Fruit Blotch",
    "Citrullus Bacterial Leaf Blotch",
    "Citrus Canker",
    "Citrus Leaf Greening",
    "Corn Smut Ustilago maydis",
    "Corynespora Leaf Spot",
    "Cotton Bollworm Helicoverpa armigera",
    "Cotton Adult Bollworm",
    "Cotton Larvae Bollworm",
    "Crucifers Xanthomonas campestris Black Rot",
    "Cucurbit Leaf Downy Mildew",
    "Curculionoidea Weevil",
    "Fruit Bollworm Damages",
    "Fall Armyworm Spodoptera frugiperda",
    "Fall Armyworm Larvae",
    "Fall Armyworm Adults",
    "Flower Blights Choanephora",
    "Fusarium Cereal Head Blight",
    "Geotrichum Rot Cucurbitaceae Fruit",
    "Grape Leaf Esca",
    "Grape Leaf Leafroll Virus",
    "Hosta Leaf Virus X",
    "Kudzu Bug Megacopta cribraria",
    "Late Blight Fruit",
    "Late Blight Leaf",
    "Leaf Bacterial Wilt",
    "Leaf Curl Virus",
    "Leaf Spurge Flea Beetle",
    "Lesser Cornstalk Borer Elasmopalpus lignosella",
    "Lettuce Chlorosis Virus LCV",
    "Longhorn Beetle",
    "Loose Smut Wheat Ustilago tritici",
    "Maize Tar Spot",
    "Mealybug",
    "Northern Maize Leaf Blight",
    "Penicillium Fungi",
    "Phytophthora Blight Cucurbitaceae",
    "Potyvirus Mosaic Virus",
    "Pythium Diseases",
    "Red Spider Mite",
    "Rhizopus Soft Rots",
    "Rhododendron Stem Borer Oberea myops",
    "Rice Leaf Blast",
    "Root Rot Damping Off Rhizoctonia",
    "Root Knot Nematode Meloidogyne",
    "Sclerotinia Sclerotiorum Fruit Rot",
    "Sclerotinia Sclerotiorum Leaf Rot",
    "Sclerotinia Sclerotiorum Stem Rot",
    "Sclerotinia Timber Rot",
    "Solanaceae Bacterial Soft Rot",
    "Solanaceae Bacterial Spot",
    "Southern Red Mite Oligonychus ilicis",
    "Southern Root Knot Nematode",
    "Soybean Leaf Halo Blight",
    "Soybean Leaf Mosaic Virus",
    "Soybean Leaf Rust",
    "Soybean Tospovirus Leaf Vein Necrosis Virus",
    "Sphinx Moth Adult",
    "Sphinx Moth Larvae Damages",
    "Spider Mite Leaf Damages",
    "Spotted Wilt Virus",
    "Stem Boring Weevil Mecinus janthinus",
    "Stem Borer Leaf Damages",
    "Sweet Potato Fruit Feathery Mottle Virus",
    "Sweet Potato Leaf Feathery Mottle Virus",
    "Target Spot Leaf",
    "Thrips Leaf Damages",
    "Tobacco Leaf Etch Virus",
    "Tobacco Leaf Ringspot Virus",
    "Tomato Early Blight",
    "Tomato Fruit Bacterial Canker",
    "Tomato Fruit Blossom End Rot",
    "Tomato Leaf Bacterial Canker",
    "Tomato Leaf Mold",
    "Tomato Leaf Mosaic Virus",
    "Tomato Leaf Septoria",
    "Tomato Yellow Leaf Curl Virus",
    "Twospotted Spider Mite Tetranychus urticae",
    "Wheat Leaf Streak Mosaic Virus",
    "Whiteflies Leaf"
]



for i, disease in enumerate(diseases, start=1):
    download_pdf(disease, i, len(diseases))
    time.sleep(random.uniform(5, 10))


