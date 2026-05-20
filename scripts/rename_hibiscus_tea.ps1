# Script pour renommer les classes du dataset Hibiscus and Tea

$csvPath = "c:\smarts-n-yieldpredict.git\hibiscus_tea_rename.csv"
$sourcePath = "C:\Downloads\Hibiscus and Tea Augmented Dataset"

# Importer le CSV
$mappings = Import-Csv -Path $csvPath

foreach ($mapping in $mappings) {
    $currentName = $mapping.current_name
    $newName = $mapping.proposed_standard_name

    $currentPath = Join-Path -Path $sourcePath -ChildPath $currentName
    $newPath = Join-Path -Path $sourcePath -ChildPath $newName

    if (Test-Path $currentPath) {
        Write-Host "Renommage de $currentName vers $newName"
        Rename-Item -Path $currentPath -NewName $newName
    } else {
        Write-Host "Dossier $currentName non trouvé"
    }
}

Write-Host "Renommage terminé."