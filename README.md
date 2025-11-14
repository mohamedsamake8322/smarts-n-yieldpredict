🌱 Indices utilisés dans le pipeline
🌿 NDVI – Normalized Difference Vegetation Index
Objectif : Évaluer la couverture végétale et la vigueur de la végétation.

Formule :

text
NDVI = (NIR - RED) / (NIR + RED)
NIR : Réflectance dans l’infrarouge proche (Near Infrared)

RED : Réflectance dans le rouge

Interprétation :

Valeurs proches de +1 ⇒ végétation dense et en bonne santé

Valeurs proches de 0 ⇒ sol nu ou zones urbaines

Valeurs négatives ⇒ eau ou nuages

💧 NDMI – Normalized Difference Moisture Index
Objectif : Détecter le stress hydrique et mesurer l’humidité de la végétation.

Formule :

text
NDMI = (NIR - SWIR) / (NIR + SWIR)
NIR : Infrarouge proche

SWIR : Infrarouge à ondes courtes

Interprétation :

Valeurs élevées ⇒ végétation bien hydratée

Valeurs basses ⇒ stress hydrique ou sol sec

🏗️ NDBI – Normalized Difference Built-up Index
Objectif : Identifier les zones urbanisées et les constructions.

Formule :

text
NDBI = (SWIR - NIR) / (SWIR + NIR)
SWIR : Infrarouge à ondes courtes

NIR : Infrarouge proche

Interprétation :

Valeurs positives ⇒ zones bâties

Valeurs négatives ⇒ végétation ou surface naturelle

🌾 VHM – Vegetation Health Metric (composite personnalisé)
Objectif : Mesurer la santé globale de la végétation en combinant plusieurs indices.

Formule proposée :

text
VHM = w1 × NDVI + w2 × NDMI
w1, w2 : pondérations personnalisables selon ton objectif (par ex. w1=0.6, w2=0.4)

Tu peux aussi intégrer des anomalies CHIRPS ou un facteur de stress.

Interprétation :

VHM > seuil ⇒ végétation saine

