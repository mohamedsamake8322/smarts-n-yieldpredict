# Regroupe les classes du dataset selon le mapping défini dans dataset_hight_regroupement.csv
# Crée les nouveaux dossiers et déplace les images.
# Exécutez depuis PowerShell avec :
#   Set-Location c:\smarts-n-yieldpredict.git
#   .\regrouper_dataset_hight.ps1

$sourceRoot = "C:\Users\moham\Pictures\Dataset hight"
$csvPath = "c:\smarts-n-yieldpredict.git\dataset_hight_regroupement.csv"

if (-not (Test-Path $csvPath)) {
    Write-Error "Fichier CSV introuvable : $csvPath"
    exit 1
}

if (-not (Test-Path $sourceRoot)) {
    Write-Error "Dossier source introuvable : $sourceRoot"
    exit 1
}

$mapping = Import-Csv -Path $csvPath

# Créer un dictionnaire pour les regroupements
$groupMappings = @{}
foreach ($row in $mapping) {
    $oldName = $row.original_name
    $newName = $row.new_name
    if (-not $groupMappings.ContainsKey($newName)) {
        $groupMappings[$newName] = @()
    }
    $groupMappings[$newName] += $oldName
}

# Pour chaque nouveau groupe
foreach ($newName in $groupMappings.Keys) {
    $destDir = Join-Path -Path $sourceRoot -ChildPath $newName

    if (-not (Test-Path $destDir)) {
        New-Item -Path $destDir -ItemType Directory | Out-Null
        Write-Host "Créé dossier : $newName"
    }

    # Déplacer les images de chaque classe source vers le nouveau dossier
    foreach ($oldName in $groupMappings[$newName]) {
        $sourceDir = Join-Path -Path $sourceRoot -ChildPath $oldName

        if (Test-Path $sourceDir) {
            $files = Get-ChildItem -Path $sourceDir -File
            foreach ($file in $files) {
                $destPath = Join-Path -Path $destDir -ChildPath $file.Name
                try {
                    Move-Item -Path $file.FullName -Destination $destPath -ErrorAction Stop
                    Write-Host "Déplacé : $($file.Name) de $oldName vers $newName"
                }
                catch {
                    Write-Warning "Échec du déplacement de $($file.Name) : $_"
                }
            }

            # Supprimer le dossier source s'il est vide
            if ((Get-ChildItem -Path $sourceDir).Count -eq 0) {
                Remove-Item -Path $sourceDir -Force
                Write-Host "Supprimé dossier vide : $oldName"
            }
        } else {
            Write-Warning "Dossier source introuvable : $sourceDir"
        }
    }
}

Write-Host "Regroupement terminé."