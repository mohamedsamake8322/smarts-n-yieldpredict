# Script pour renommer les classes selon le mapping CSV

$csvPath = "c:\smarts-n-yieldpredict.git\class_rename_mapping.csv"
$sourcePath = "C:\Downloads\archive (10). important\sorted_images"

# Importer le CSV
$mappings = Import-Csv -Path $csvPath

foreach ($mapping in $mappings) {
    $oldName = $mapping.old_name
    $newName = $mapping.new_name

    $oldPath = Join-Path -Path $sourcePath -ChildPath $oldName
    $newPath = Join-Path -Path $sourcePath -ChildPath $newName

    if (Test-Path $oldPath) {
        if (Test-Path $newPath) {
            # Si le nouveau nom existe déjà, déplacer les images dedans
            Write-Host "Fusion de $oldName dans $newName"
            Get-ChildItem -Path $oldPath -File | Move-Item -Destination $newPath
            Remove-Item -Path $oldPath -Recurse
        } else {
            Write-Host "Renommage de $oldName vers $newName"
            Rename-Item -Path $oldPath -NewName $newName
        }
    } else {
        Write-Host "Dossier $oldName non trouvé"
    }
}

Write-Host "Renommage terminé."