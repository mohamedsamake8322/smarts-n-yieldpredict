# Script pour appliquer les noms standardisés selon le CSV proposé

$csvPath = "c:\smarts-n-yieldpredict.git\proposed_standard_names.csv"
$sourcePath = "C:\Downloads\archive (10). important\sorted_images"

# Importer le CSV
$mappings = Import-Csv -Path $csvPath

foreach ($mapping in $mappings) {
    $currentName = $mapping.current_name
    $newName = $mapping.proposed_standard_name

    $currentPath = Join-Path -Path $sourcePath -ChildPath $currentName
    $newPath = Join-Path -Path $sourcePath -ChildPath $newName

    if (Test-Path $currentPath) {
        if (Test-Path $newPath) {
            # Si le nouveau nom existe déjà, déplacer les images dedans
            Write-Host "Fusion de $currentName dans $newName"
            Get-ChildItem -Path $currentPath -File | Move-Item -Destination $newPath
            Remove-Item -Path $currentPath -Recurse
        } else {
            Write-Host "Renommage de $currentName vers $newName"
            Rename-Item -Path $currentPath -NewName $newName
        }
    } else {
        Write-Host "Dossier $currentName non trouvé, ignoré"
    }
}

Write-Host "Application des noms standardisés terminée."