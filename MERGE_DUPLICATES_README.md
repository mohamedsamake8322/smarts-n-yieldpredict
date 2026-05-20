# 🔄 FUSION DES DOUBLONS - DATASET PLANTDATASET

Ce dossier contient tous les scripts nécessaires pour analyser, sauvegarder et fusionner les dossiers dupliqués dans votre dataset de maladies des plantes.

## 📋 SCRIPTS DISPONIBLES

### 1. `analyze_duplicates.py`
**Analyse préliminaire des doublons**
- Identifie les dossiers dupliqués basés sur des noms normalisés
- Génère un rapport détaillé des doublons trouvés
- Crée un plan de fusion recommandé

### 2. `backup_dataset.py`
**Système de sauvegarde**
- Crée une copie complète du dataset avant fusion
- Vérifie l'intégrité de la sauvegarde
- Permet la restauration en cas de problème

### 3. `merge_plant_dataset.py`
**Script principal de fusion**
- Orchestre tout le processus: analyse → sauvegarde → fusion
- Déplace les fichiers au lieu de les copier (économie d'espace)
- Archive les dossiers vides dans un dossier séparé

### 4. `verify_merge.py`
**Vérification post-fusion**
- Vérifie que la fusion s'est bien déroulée
- Génère des statistiques détaillées
- Crée un rapport de santé du dataset

## 🚀 UTILISATION RECOMMANDÉE

### Étape 1: Analyse préliminaire
```bash
python analyze_duplicates.py
```
Modifiez le chemin du dataset dans le script avant exécution.

### Étape 2: Fusion complète (recommandé)
```bash
python merge_plant_dataset.py --path "C:\path\to\your\Plantdataset"
```

### Étape 3: Vérification
```bash
python verify_merge.py
```

## ⚙️ OPTIONS AVANCÉES

### Mode simulation (pour tester)
```bash
python merge_plant_dataset.py --path "C:\path\to\your\Plantdataset" --dry-run
```

### Sans sauvegarde (dangereux!)
```bash
python merge_plant_dataset.py --path "C:\path\to\your\Plantdataset" --no-backup
```

## 📁 STRUCTURE APRÈS FUSION

```
Plantdataset/
├── Classe1/           # Dossier fusionné avec toutes les images
├── Classe2/           # Dossier fusionné avec toutes les images
├── ...
├── empty_folders/     # Archive des dossiers vides
│   ├── AncienDoublon1/
│   ├── AncienDoublon2/
│   └── ...
├── duplicate_analysis.csv     # Analyse originale
├── merge_plan.csv            # Plan de fusion
├── merge_verification.csv    # Statistiques finales
└── dataset_health_report.txt # Rapport de santé
```

## 🔧 PERSONNALISATION

### Modifier les règles de normalisation
Éditez la fonction `normalize_name()` dans les scripts pour:
- Ajouter de nouveaux termes de remplacement
- Modifier les règles de normalisation
- Gérer des cas spécifiques de votre dataset

### Ajuster le seuil de détection
Modifiez la logique de groupement dans `analyze_duplicates.py` pour:
- Changer la sensibilité de détection des doublons
- Exclure certains dossiers de la fusion
- Personnaliser les règles de priorité

## ⚠️ RECOMMANDATIONS DE SÉCURITÉ

1. **Toujours créer une sauvegarde** avant la fusion
2. **Testez d'abord en mode simulation** (`--dry-run`)
3. **Vérifiez les résultats** avec `verify_merge.py`
4. **Gardez la sauvegarde** jusqu'à validation complète du dataset

## 🔍 DÉPANNAGE

### Problèmes courants:
- **Chemin incorrect**: Vérifiez que le chemin vers Plantdataset est correct
- **Permissions**: Assurez-vous d'avoir les droits d'écriture
- **Espace disque**: La fusion nécessite de l'espace pour la sauvegarde
- **Noms de fichiers**: Les conflits sont automatiquement résolus par numérotation

### Récupération:
En cas de problème, restaurez depuis la sauvegarde:
```bash
# Copiez le contenu du dossier _backup vers Plantdataset
```

## 📊 RÉSULTATS ATTENDUS

Après fusion réussie:
- ✅ Réduction significative du nombre de dossiers
- ✅ Consolidation des images dans des classes cohérentes
- ✅ Préservation de toutes les images originales
- ✅ Archive propre des dossiers vides
- ✅ Rapports détaillés pour validation

## 🤝 SUPPORT

Si vous rencontrez des problèmes:
1. Vérifiez les logs d'erreur dans la console
2. Consultez les fichiers de rapport générés
3. Testez d'abord avec un petit sous-ensemble du dataset
4. Contactez pour assistance si nécessaire

---

**Note**: Ces scripts sont optimisés pour les datasets de maladies des plantes mais peuvent être adaptés pour d'autres types de datasets avec des doublons de noms similaires.