#!/bin/bash
# Script de déploiement automatique pour HF Spaces
# Utilisation: ./deploy_to_spaces.sh

echo "🚀 Déploiement automatique vers HF Spaces..."

# Configuration
SPACE_NAME="mohamedsamake8322/sene-disease-api"
HF_TOKEN="${HF_TOKEN:-$HUGGINGFACE_TOKEN}"  # Utilise la variable d'environnement

if [ -z "$HF_TOKEN" ]; then
    echo "❌ Erreur: HF_TOKEN non défini. Définissez HUGGINGFACE_TOKEN ou HF_TOKEN"
    exit 1
fi

# Vérifier si huggingface_hub est installé
if ! command -v huggingface-cli &> /dev/null; then
    echo "📦 Installation de huggingface_hub..."
    pip install huggingface_hub
fi

# Se connecter à Hugging Face
echo "🔑 Connexion à Hugging Face..."
huggingface-cli login --token "$HF_TOKEN"

# Créer le dossier temporaire pour les fichiers à uploader
TEMP_DIR="temp_deploy"
mkdir -p "$TEMP_DIR"

# Copier les fichiers essentiels
echo "📁 Préparation des fichiers..."
cp app_simple.py "$TEMP_DIR/app.py"  # Renommer app_simple.py en app.py
cp requirements.txt "$TEMP_DIR/"
cp Dockerfile "$TEMP_DIR/"
cp README.md "$TEMP_DIR/"

# Créer un fichier .gitkeep vide pour éviter les erreurs
touch "$TEMP_DIR/.gitkeep"

# Uploader vers le Space
echo "⬆️ Upload vers $SPACE_NAME..."
huggingface-cli upload-folder \
    --repo-type space \
    --repo-id "$SPACE_NAME" \
    "$TEMP_DIR" \
    . \
    --delete-deleted

# Nettoyer
rm -rf "$TEMP_DIR"

echo "✅ Déploiement terminé!"
echo "🌐 Votre API sera disponible sur: https://$SPACE_NAME.hf.space"
echo "⏳ Attendez quelques minutes que le Space redémarre..."

# Test automatique
echo "🧪 Test de l'API dans 30 secondes..."
sleep 30
python test_api.py "https://$SPACE_NAME.hf.space"