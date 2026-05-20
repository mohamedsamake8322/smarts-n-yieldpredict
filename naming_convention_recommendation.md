# Recommandation de nomenclature pour entrainement ML

## Format recommande

Utiliser un format unique:

`CommonCrop_Disease`

Exemples:

- `Tomato_Early_Blight`
- `Cassava_Bacterial_Blight`
- `Rice_Tungro_Disease`

## Pourquoi ce choix

- Plus lisible pour le debug, l'analyse d'erreurs et les dashboards.
- Plus compatible avec la litterature datasets open-source (PlantVillage-like).
- Evite les collisions entre variantes de noms scientifiques partiels.
- N'impacte pas la performance du modele: le modele apprend des images, pas la semantique du texte.

## Regles pratiques

- Camel Snake Case: tokens separes par `_`, chaque token en `TitleCase`.
- Garder `Healthy` comme classe standard pour les feuilles/plantes saines.
- Ajouter `_Virus`, `_Blight`, `_Rust`, etc. quand l'etiologie est connue.
- Eviter de melanger nom commun et nom scientifique dans les labels de classes.

## Ou garder le nom scientifique

Conserver les noms scientifiques dans des metadonnees separees:

- `scientific_crop_name`
- `scientific_pathogen_name`

Ainsi tu gardes l'universalite scientifique sans complexifier les labels d'entrainement.
