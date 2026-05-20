# Répartit les images de master_images en sous-dossiers par classe selon dataset_manifest.csv
# Exécutez depuis PowerShell avec :
#   Set-Location c:\smarts-n-yieldpredict.git
#   .\sort_images_by_manifest.ps1

$sourceDir = "C:\Downloads\archive (10). important\master_images\images"
$csvPath = "C:\Downloads\archive (10). important\metadata\dataset_manifest.csv"
$destRoot = "C:\Downloads\archive (10). important\sorted_images"

if (-not (Test-Path $csvPath)) {
    Write-Error "Fichier CSV introuvable : $csvPath"
    exit 1
}

if (-not (Test-Path $sourceDir)) {
    Write-Error "Dossier source introuvable : $sourceDir"
    exit 1
}

# Créer le dossier de destination si nécessaire
if (-not (Test-Path $destRoot)) {
    New-Item -Path $destRoot -ItemType Directory | Out-Null
    Write-Host "Créé dossier de destination : $destRoot"
}

# Lire le CSV (en ignorant les erreurs de parsing)
$manifest = Import-Csv -Path $csvPath -ErrorAction SilentlyContinue

$processed = 0
$errors = 0

foreach ($row in $manifest) {
    $imageId = $row.image_id
    $imagePath = $row.image_path
    $canonicalClass = $row.canonical_class

    if ([string]::IsNullOrWhiteSpace($imageId) -or [string]::IsNullOrWhiteSpace($canonicalClass)) {
        $errors++
        continue
    }

    # Nettoyer le nom de classe pour le dossier (remplacer espaces et caractères spéciaux)
    $classFolder = $canonicalClass -replace '[^\w\s-]', '' -replace '\s+', '_'

    # Chemin source de l'image
    $sourcePath = Join-Path -Path $sourceDir -ChildPath "$imageId.jpg"

    # Chemin destination
    $destDir = Join-Path -Path $destRoot -ChildPath $classFolder
    $destPath = Join-Path -Path $destDir -ChildPath "$imageId.jpg"

    # Créer le dossier de classe si nécessaire
    if (-not (Test-Path $destDir)) {
        New-Item -Path $destDir -ItemType Directory | Out-Null
    }

    # Déplacer l'image
    if (Test-Path $sourcePath) {
        try {
            Move-Item -Path $sourcePath -Destination $destPath -ErrorAction Stop
            $processed++
            if ($processed % 1000 -eq 0) {
                Write-Host "Traités : $processed images"
            }
        }
        catch {
            Write-Warning "Échec du déplacement de $imageId : $_"
            $errors++
        }
    } else {
        Write-Warning "Image introuvable : $sourcePath"
        $errors++
    }
}

Write-Host "Tri terminé. Images traitées : $processed, Erreurs : $errors"