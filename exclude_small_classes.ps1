# Script pour exclure les classes avec moins de 100 images
# Déplace les classes petites vers un dossier excluded_small_classes

$sourcePath = "C:\Downloads\archive (10). important\sorted_images"
$excludedPath = "C:\Downloads\archive (10). important\excluded_small_classes"

# Créer le dossier d'exclusion s'il n'existe pas
if (!(Test-Path $excludedPath)) {
    New-Item -ItemType Directory -Path $excludedPath
}

# Obtenir toutes les classes (dossiers)
$classes = Get-ChildItem -Path $sourcePath -Directory

foreach ($class in $classes) {
    $imageCount = (Get-ChildItem -Path $class.FullName -File).Count
    Write-Host "Classe: $($class.Name) - Images: $imageCount"

    if ($imageCount -lt 100) {
        Write-Host "Déplacement de $($class.Name) vers excluded_small_classes"
        Move-Item -Path $class.FullName -Destination $excludedPath
    }
}

Write-Host "Exclusion terminée."