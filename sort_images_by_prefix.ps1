# Trie les images d'un dossier en sous-dossiers par classe,
# en se basant sur le préfixe du nom de fichier avant le premier tiret.
# Exemple : ants-1-_jpg.rf... => dossier ants

$sourceDir = "C:\Users\moham\Pictures\train\train\images"

if (-not (Test-Path $sourceDir)) {
    Write-Error "Dossier introuvable : $sourceDir"
    exit 1
}

$files = Get-ChildItem -Path $sourceDir -File | Where-Object { $_.Name -match '^[^-]+' }

if ($files.Count -eq 0) {
    Write-Host "Aucun fichier image trouvé dans : $sourceDir"
    exit 0
}

foreach ($file in $files) {
    $className = ($file.Name -split '-')[0]
    if ([string]::IsNullOrWhiteSpace($className)) {
        Write-Warning "Impossible d'extraire le nom de classe pour : $($file.Name)"
        continue
    }

    $destDir = Join-Path -Path $sourceDir -ChildPath $className

    if (-not (Test-Path $destDir)) {
        New-Item -Path $destDir -ItemType Directory | Out-Null
    }

    $destPath = Join-Path -Path $destDir -ChildPath $file.Name

    try {
        Move-Item -Path $file.FullName -Destination $destPath -ErrorAction Stop
        Write-Host "Déplacé : $($file.Name) -> $className"
    }
    catch {
        Write-Warning "Échec du déplacement de $($file.Name) : $_"
    }
}

Write-Host "Tri terminé."
