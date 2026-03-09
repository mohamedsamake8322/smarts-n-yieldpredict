# Script de déploiement automatique pour HF Spaces (Windows)
# Utilisation: .\deploy_to_spaces.ps1

param(
    [string]$HuggingFaceToken = $env:HUGGINGFACE_TOKEN
)

Write-Host "🚀 Déploiement automatique vers HF Spaces..." -ForegroundColor Green

# Configuration
$SPACE_NAME = "mohamedsamake8322/sene-disease-api"

if (-not $HuggingFaceToken) {
    Write-Host "❌ Erreur: HUGGINGFACE_TOKEN non défini. Définissez la variable d'environnement HUGGINGFACE_TOKEN" -ForegroundColor Red
    exit 1
}

# Vérifier si huggingface_hub est installé
try {
    $null = Get-Command huggingface-cli -ErrorAction Stop
} catch {
    Write-Host "📦 Installation de huggingface_hub..." -ForegroundColor Yellow
    pip install huggingface_hub
}

# Se connecter à Hugging Face
Write-Host "🔑 Connexion à Hugging Face..." -ForegroundColor Yellow
huggingface-cli login --token $HuggingFaceToken

# Créer le dossier temporaire pour les fichiers à uploader
$TEMP_DIR = "temp_deploy"
if (Test-Path $TEMP_DIR) {
    Remove-Item -Recurse -Force $TEMP_DIR
}
New-Item -ItemType Directory -Path $TEMP_DIR | Out-Null

# Copier les fichiers essentiels
Write-Host "📁 Préparation des fichiers..." -ForegroundColor Yellow
Copy-Item "app_simple.py" "$TEMP_DIR\app.py"  # Renommer app_simple.py en app.py
Copy-Item "requirements.txt" "$TEMP_DIR\"
Copy-Item "Dockerfile" "$TEMP_DIR\"
Copy-Item "README.md" "$TEMP_DIR\"

# Créer un fichier .gitkeep vide pour éviter les erreurs
New-Item -ItemType File "$TEMP_DIR\.gitkeep" | Out-Null

# Uploader vers le Space
Write-Host "⬆️ Upload vers $SPACE_NAME..." -ForegroundColor Yellow
huggingface-cli upload-folder `
    --repo-type space `
    --repo-id $SPACE_NAME `
    $TEMP_DIR `
    . `
    --delete-deleted

# Nettoyer
Remove-Item -Recurse -Force $TEMP_DIR

Write-Host "✅ Déploiement terminé!" -ForegroundColor Green
Write-Host "🌐 Votre API sera disponible sur: https://$SPACE_NAME.hf.space" -ForegroundColor Cyan
Write-Host "⏳ Attendez quelques minutes que le Space redémarre..." -ForegroundColor Yellow

# Test automatique
Write-Host "🧪 Test de l'API dans 30 secondes..." -ForegroundColor Yellow
Start-Sleep -Seconds 30
python test_api.py "https://$SPACE_NAME.hf.space"