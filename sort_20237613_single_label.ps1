# Script pour trier les images avec une seule classe par image (priorité)

$rootPath = "C:\Downloads\20237613"
$csvPath = Join-Path $rootPath "Database.csv"
$imagesPath = Join-Path $rootPath "leaf_images"
$destPath = Join-Path $rootPath "sorted_images_single_label"

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

# Priorité des classes (plus haut = priorité plus élevée)
$priorityOrder = @("Grey_Leaf_Spot", "Northern_Corn_Leaf_Blight", "Phaeosphaeria_Leaf_Spot", "Common_Rust", "Southern_Rust", "Healthy", "Other", "Unknown")

if (!(Test-Path $destPath)) {
    New-Item -ItemType Directory -Path $destPath | Out-Null
}

foreach ($className in $classMap.Values) {
    $classDir = Join-Path $destPath $className
    if (!(Test-Path $classDir)) {
        New-Item -ItemType Directory -Path $classDir | Out-Null
    }
}

$mappings = Import-Csv -Path $csvPath
$missing = @()
$created = 0
$skipped = 0

foreach ($row in $mappings) {
    $relativeFile = $row.filePath.Trim()
    $sourceFile = Join-Path $imagesPath $relativeFile

    if (!(Test-Path $sourceFile)) {
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
        $selectedLabel = "Unmapped"
    } else {
        # Sélectionner la classe avec la plus haute priorité
        $selectedLabel = $labels | Where-Object { $_ -in $priorityOrder } | Sort-Object { $priorityOrder.IndexOf($_) } | Select-Object -First 1
        if (!$selectedLabel) {
            $selectedLabel = $labels[0]  # Fallback
        }
    }

    $classDir = Join-Path $destPath $selectedLabel
    if (!(Test-Path $classDir)) { New-Item -ItemType Directory -Path $classDir | Out-Null }
    $destFile = Join-Path $classDir (Split-Path $sourceFile -Leaf)

    if (!(Test-Path $destFile)) {
        try {
            New-Item -ItemType HardLink -Path $destFile -Target $sourceFile | Out-Null
            $created++
        } catch {
            Copy-Item -Path $sourceFile -Destination $destFile -Force
            $created++
        }
    } else {
        $skipped++
    }
}

Write-Host "Tri terminé. Hardlinks/créations : $created, déjà existant : $skipped"
if ($missing.Count -gt 0) {
    $missingFile = Join-Path $rootPath "missing_files_from_database.txt"
    $missing | Sort-Object -Unique | Set-Content -Path $missingFile -Encoding UTF8
    Write-Host "Fichiers manquants : $($missing.Count). Liste enregistrée dans $missingFile"
}