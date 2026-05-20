# Renomme les dossiers dans "Data non traité" selon le mapping défini dans non_traite_classes_cleaned.csv
# Exécutez ce script depuis PowerShell avec :
#   Set-Location c:\smarts-n-yieldpredict.git
#   .\rename_non_traite_folders.ps1

$sourceRoot = "C:\Users\moham\Pictures\Data non traité"
$csvPath = "c:\smarts-n-yieldpredict.git\non_traite_classes_cleaned.csv"

if (-not (Test-Path $csvPath)) {
    Write-Error "Fichier CSV introuvable : $csvPath"
    exit 1
}

if (-not (Test-Path $sourceRoot)) {
    Write-Error "Dossier source introuvable : $sourceRoot"
    exit 1
}

$mapping = Import-Csv -Path $csvPath

foreach ($row in $mapping) {
    $oldName = $row.original_name
    $newName = $row.new_name

    if ([string]::IsNullOrWhiteSpace($oldName) -or [string]::IsNullOrWhiteSpace($newName)) {
        Write-Warning "Mapping invalide trouvé (original_name ou new_name vide). Ligne ignorée."
        continue
    }

    $sourcePath = Join-Path -Path $sourceRoot -ChildPath $oldName
    $destPath = Join-Path -Path $sourceRoot -ChildPath $newName

    if (-not (Test-Path $sourcePath)) {
        Write-Warning "Dossier introuvable : $sourcePath"
        continue
    }

    if (Test-Path $destPath) {
        Write-Warning "Destination déjà existante, renommage ignoré : $destPath"
        continue
    }

    try {
        Rename-Item -Path $sourcePath -NewName $newName -ErrorAction Stop
        Write-Host "Renommé : $oldName -> $newName"
    }
    catch {
        Write-Warning "Échec du renommage de '$oldName' en '$newName' : $_"
    }
}

Write-Host "Renommage terminé."
