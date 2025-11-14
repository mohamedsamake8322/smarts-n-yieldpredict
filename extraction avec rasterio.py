import os
import zipfile
import rasterio
from rasterio.windows import Window
import numpy as np

# Fichier à traiter
zip_file = "wc2.1_30s_tavg.zip"
folder_path = r"C:\Users\moham\Music\Data"
zip_path = os.path.join(folder_path, zip_file)
extract_path = os.path.join(folder_path, "temp_tavg")

print(f"📦 Début de l'extraction du fichier : {zip_file}")

# Extraction
os.makedirs(extract_path, exist_ok=True)
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print(f"✅ Extraction terminée dans : {extract_path}")

# Lecture par blocs
for file in os.listdir(extract_path):
    if file.endswith(".tif"):
        tif_path = os.path.join(extract_path, file)
        print(f"\n📂 Lecture du fichier raster : {file}")
        with rasterio.open(tif_path) as src:
            print(f" - Dimensions : {src.width} x {src.height}")
            print(f" - CRS : {src.crs}")
            print(f" - Nombre de bandes : {src.count}")

            block_size = 512
            total_blocks = ((src.width + block_size - 1) // block_size) * ((src.height + block_size - 1) // block_size)
            block_count = 0

            for i in range(0, src.width, block_size):
                for j in range(0, src.height, block_size):
                    block_count += 1
                    print(f"🔄 Lecture du bloc {block_count}/{total_blocks} (coin supérieur : {i},{j})")
                    window = Window(i, j, block_size, block_size)
                    data = src.read(1, window=window)
                    print(f"   ↪ Valeurs min/max : {data.min()} / {data.max()}")
