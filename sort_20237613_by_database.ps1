# Script de tri des images de C:\Downloads\20237613 à partir de Database.csv

$rootPath = "C:\Downloads\20237613"
$csvPath = Join-Path $rootPath "Database.csv"
$imagesPath = Join-Path $rootPath "leaf_images"
$destPath = Join-Path $rootPath "sorted_images"

# Mapping des colonnes à une classe lisible
$classMap = @{
    GLS = "Grey_Leaf_Spot"
    NCLB = "Northern_Corn_Leaf_Blight"
    PLS = "Phaeosphaeria_Leaf_Spot"
    CR = "Common_Rust"
    SR = "Southern_Rust"
    NoFoliarSymptoms = "Healthy"
    Other = "Other"
    UnidentifiedDisease = "Unknown"
}

if (!(Test-Path $destPath)) {
    New-Item -ItemType Directory -Path $destPath | Out-Null
}

$mappings = Import-Csv -Path $csvPath

# Créer les dossiers de classes
foreach ($className in $classMap.Values) {
    $classDir = Join-Path $destPath $className
    if (!(Test-Path $classDir)) {
        New-Item -ItemType Directory -Path $classDir | Out-Null
    }
}

$missing = @()
$copied = 0
$skipped = 0

foreach ($row in $mappings) {
    $relativeFile = $row.filePath.Trim()
    $sourceFile = Join-Path $imagesPath $relativeFile

    if (!(Test-Path $sourceFile)) {
        # Si le nom de fichier n'existe pas, tenter en ignorant les caractères non valides
        $normalized = $relativeFile -replace '[\\/:*?"<>|]', '_'
        $sourceFile = Join-Path $imagesPath $normalized
    }

    if (!(Test-Path $sourceFile)) {
        $missing += $relativeFile
        continue
    }

    $labels = @()
    foreach ($col in $classMap.Keys) {
        if ($row.$col -eq '1' -or $row.$col -eq 1) {
            $labels += $classMap[$col]
        }
    }

    if ($labels.Count -eq 0) {
        $labels = @("Unmapped")
        $destSub = Join-Path $destPath $labels[0]
        if (!(Test-Path $destSub)) { New-Item -ItemType Directory -Path $destSub | Out-Null }
        Copy-Item -Path $sourceFile -Destination $destSub -Force
        $copied++
    } else {
        foreach ($label in $labels) {
            $destSub = Join-Path $destPath $label
            if (!(Test-Path $destSub)) { New-Item -ItemType Directory -Path $destSub | Out-Null }
            Copy-Item -Path $sourceFile -Destination $destSub -Force
            $copied++
        }
    }
}

Write-Host "Tri terminé. Images copiées : $copied"
if ($missing.Count -gt 0) {
    $missingFile = Join-Path $rootPath "missing_files_from_database.txt"
    $missing | Sort-Object -Unique | Set-Content -Path $missingFile -Encoding UTF8
    Write-Host "Fichiers manquants : $($missing.Count). Liste enregistrée dans $missingFile"
}
