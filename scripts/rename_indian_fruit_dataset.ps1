# Script pour renommer les classes du dataset "6  indian Fruit crops Fruit and Leaf Disease Dataset"

$csvPath = "c:\smarts-n-yieldpredict.git\indian_fruit_dataset_rename.csv"
$sourcePath = "C:\Downloads\6  indian Fruit crops Fruit and Leaf Disease Dataset"

$mappings = Import-Csv -Path $csvPath

foreach ($mapping in $mappings) {
    $currentName = $mapping.current_name
    $newName = $mapping.proposed_standard_name

    $currentPath = Join-Path -Path $sourcePath -ChildPath $currentName
    $newPath = Join-Path -Path $sourcePath -ChildPath $newName

    if (Test-Path $currentPath) {
        if (Test-Path $newPath) {
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

Write-Host "Renommage du dataset terminé."