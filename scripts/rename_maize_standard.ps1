# Script pour renommer les dossiers selon les noms standardisés

$rootPath = "C:\Downloads\20237613"
$csvPath = "c:\smarts-n-yieldpredict.git\maize_standard_names.csv"
$sortedPath = Join-Path $rootPath "sorted_images_single_label"

$mappings = Import-Csv -Path $csvPath

foreach ($mapping in $mappings) {
    $oldName = $mapping.old_name
    $newName = $mapping.new_name

    $oldPath = Join-Path $sortedPath $oldName
    $newPath = Join-Path $sortedPath $newName

    if (Test-Path $oldPath) {
        if (!(Test-Path $newPath)) {
            Rename-Item -Path $oldPath -NewName $newName
            Write-Host "Renommé : $oldName -> $newName"
        } else {
            Write-Host "Le dossier $newName existe déjà, fusion en cours..."
            Get-ChildItem -Path $oldPath -File | ForEach-Object {
                $destFile = Join-Path $newPath $_.Name
                if (!(Test-Path $destFile)) {
                    Move-Item -Path $_.FullName -Destination $destFile
                } else {
                    Write-Host "Fichier déjà existant : $($_.Name)"
                }
            }
            Remove-Item -Path $oldPath -Recurse -Force
        }
    } else {
        Write-Host "Dossier $oldName non trouvé"
    }
}

Write-Host "Renommage terminé."