#!/usr/bin/env python3
"""
🎯 SCRIPT ROBUSTE DE CURATION DU DATASET
========================================

Objectif: Construire un dataset final robuste pour entraîner le modèle principal
avec réduction drastique du bruit, équilibre inter-classes contrôlé, et renforcement
des classes rares sans trahir la réalité biologique.
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import imagehash
from tqdm import tqdm
import shutil
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import random
import glob
import albumentations as A

# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT = r"C:\smarts-n-yieldpredict.git\dataset_images"
OUTPUT = r"C:\smarts-n-yieldpredict.git\dataset_final"
QUARANTINE = r"C:\smarts-n-yieldpredict.git\dataset_quarantine"
REPORT_DIR = r"C:\smarts-n-yieldpredict.git\curation_reports"
CHECKPOINT_FILE = r"C:\smarts-n-yieldpredict.git\curation_reports\checkpoint.json"

# Seuils de qualité
MIN_RES = 224
IDEAL_RES = 384
BLUR_THRESHOLD = 80
MIN_BRIGHTNESS = 30  # 0-255
MAX_BRIGHTNESS = 220  # 0-255
COMPRESSION_THRESHOLD = 0.1  # bytes per pixel minimum

# Doublons
PHASH_THRESHOLD = 6  # Distance Hamming

# Plafonnement
MAX_DOMINANT = 500  # >10k images
MAX_LARGE = 300     # 3k-10k images
MAX_MEDIUM = 200    # 1k-3k images
# <1k images: toutes conservées

# Augmentation
MIN_STANDARD = 80   # Classe standard
MIN_RARE = 40       # Classe rare
MIN_CRITICAL = 20   # Classe critique

# Facteurs d'augmentation selon nombre d'images réelles
AUGMENTATION_FACTORS = {
    20: 10,
    30: 8,
    50: 6,
    80: 4
}

# Mots-clés pour nettoyage sémantique
SEMANTIC_NOISE_KEYWORDS = [
    'book', 'poster', 'diagram', 'scheme', 'schema', 'illustration',
    'microscope', 'microscopy', 'microscopic',
    'trap', 'sticky', 'multiple', 'mixed',
    'drawing', 'sketch'
]

# Créer les dossiers
os.makedirs(OUTPUT, exist_ok=True)
os.makedirs(QUARANTINE, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ============================================================================
# GESTION DES CHECKPOINTS
# ============================================================================

def load_checkpoint():
    """Charge le fichier de checkpoint s'il existe"""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
                print(f"✓ Checkpoint trouvé: {len(checkpoint.get('processed_classes', []))} classes déjà traitées")
                return checkpoint
        except Exception as e:
            # Backup the corrupt checkpoint and try to reconstruct processed classes
            try:
                backup_path = CHECKPOINT_FILE + ".corrupt_" + datetime.now().strftime('%Y%m%d_%H%M%S')
                shutil.copy2(CHECKPOINT_FILE, backup_path)
                print(f"⚠️  Checkpoint corrompu sauvegardé: {backup_path}")
            except Exception:
                pass
            print(f"⚠️  Erreur lecture checkpoint: {e}")

            # Rebuild processed classes from existing OUTPUT folder structure
            processed = []
            try:
                for cat in os.listdir(OUTPUT):
                    cat_path = os.path.join(OUTPUT, cat)
                    if not os.path.isdir(cat_path):
                        continue
                    for sub in os.listdir(cat_path):
                        sub_path = os.path.join(cat_path, sub)
                        if os.path.isdir(sub_path):
                            # consider as processed if it contains at least one image
                            imgs = [f for f in os.listdir(sub_path) if f.lower().endswith(('.jpg','.jpeg','.png'))]
                            if len(imgs) > 0:
                                processed.append(f"{cat}/{sub}")
                print(f"✓ Reconstruit checkpoint depuis OUTPUT: {len(processed)} classes détectées")
            except Exception as e2:
                print(f"⚠️  Impossible de reconstruire checkpoint depuis OUTPUT: {e2}")
                processed = []

            return {'processed_classes': processed, 'start_time': datetime.now().isoformat()}
    # Default empty checkpoint
    return {'processed_classes': [], 'start_time': datetime.now().isoformat()}

def save_checkpoint(checkpoint):
    """Sauvegarde le checkpoint"""
    try:
        with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️  Erreur sauvegarde checkpoint: {e}")

# Créer les dossiers
os.makedirs(OUTPUT, exist_ok=True)
os.makedirs(QUARANTINE, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Statistiques
stats = {
    'total_scanned': 0,
    'quality_rejected': 0,
    'duplicates_removed': 0,
    'semantic_noise_removed': 0,
    'classes_capped': 0,
    'images_augmented': 0,
    'final_images': 0,
    'class_distribution': {},
    'quality_reasons': defaultdict(int),
    'processing_errors': []
}

# ============================================================================
# 1. FILTRAGE TECHNIQUE DE QUALITÉ
# ============================================================================

def check_resolution(img_path):
    """Vérifie la résolution minimale"""
    try:
        img = Image.open(img_path)
        w, h = img.size
        return w >= MIN_RES and h >= MIN_RES, (w, h)
    except:
        return False, (0, 0)

def check_blur(img_path):
    """Vérifie le flou avec variance Laplacien"""
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return False, 0.0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var > BLUR_THRESHOLD, laplacian_var
    except:
        return False, 0.0

def check_brightness(img_path):
    """Vérifie la luminosité moyenne"""
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return False, 0.0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        return MIN_BRIGHTNESS <= brightness <= MAX_BRIGHTNESS, brightness
    except:
        return False, 0.0

def check_compression(img_path):
    """Détecte la compression excessive"""
    try:
        size = os.path.getsize(img_path)
        img = Image.open(img_path)
        w, h = img.size
        pixels = w * h
        if pixels > 0:
            bytes_per_pixel = size / pixels
            return bytes_per_pixel > COMPRESSION_THRESHOLD
        return False
    except:
        return False

def check_artificial_borders(img_path):
    """Détecte les bordures artificielles"""
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return True  # En cas de doute, on garde

        h, w = img.shape[:2]
        border_size = min(10, w // 20, h // 20)

        if border_size < 2:
            return True

        # Vérifier les bords pour uniformité excessive
        top_border = img[0:border_size, :].mean()
        bottom_border = img[-border_size:, :].mean()
        left_border = img[:, 0:border_size].mean()
        right_border = img[:, -border_size:].mean()

        borders = [top_border, bottom_border, left_border, right_border]
        std_dev = np.std(borders)

        return std_dev > 5.0  # Variation minimale attendue
    except:
        return True  # En cas de doute, on garde

def is_valid_image(img_path):
    """Validation complète d'une image"""
    stats['total_scanned'] += 1

    # Résolution
    res_ok, (w, h) = check_resolution(img_path)
    if not res_ok:
        stats['quality_reasons']['resolution'] += 1
        return False, 'resolution'

    # Blur
    blur_ok, blur_score = check_blur(img_path)
    if not blur_ok:
        stats['quality_reasons']['blur'] += 1
        return False, 'blur'

    # Luminosité
    bright_ok, brightness = check_brightness(img_path)
    if not bright_ok:
        stats['quality_reasons']['brightness'] += 1
        return False, 'brightness'

    # Compression
    comp_ok = check_compression(img_path)
    if not comp_ok:
        stats['quality_reasons']['compression'] += 1
        return False, 'compression'

    # Bordures
    borders_ok = check_artificial_borders(img_path)
    if not borders_ok:
        stats['quality_reasons']['borders'] += 1
        return False, 'borders'

    return True, None

# ============================================================================
# 2. SUPPRESSION DES DOUBLONS (PERCEPTUAL HASH)
# ============================================================================

def remove_duplicates(images):
    """Supprime les doublons en gardant la meilleure image"""
    if len(images) == 0:
        return []

    hashes = {}
    clean = []
    duplicates_to_remove = []

    # Calculer les hash et trouver les doublons
    for img_path in images:
        try:
            h = imagehash.phash(Image.open(img_path))
            duplicate = False
            best_path = None

            # Chercher un hash similaire
            for existing_hash, existing_path in hashes.items():
                if h - existing_hash < PHASH_THRESHOLD:
                    # Comparer la qualité (taille du fichier = proxy de qualité)
                    if os.path.getsize(img_path) > os.path.getsize(existing_path):
                        # Nouvelle image est meilleure, remplacer
                        duplicates_to_remove.append(existing_path)
                        best_path = img_path
                        hashes[existing_hash] = img_path
                    else:
                        # Ancienne image est meilleure
                        duplicates_to_remove.append(img_path)
                        best_path = existing_path
                    duplicate = True
                    break

            if not duplicate:
                hashes[h] = img_path
                clean.append(img_path)
            elif best_path == img_path:
                # Remplacer l'ancienne dans clean
                if existing_path in clean:
                    clean.remove(existing_path)
                clean.append(img_path)
        except Exception as e:
            stats['processing_errors'].append(f"Erreur hash {img_path}: {e}")
            continue

    stats['duplicates_removed'] += len(duplicates_to_remove)
    return clean

# ============================================================================
# 3. NETTOYAGE SÉMANTIQUE
# ============================================================================

def is_semantic_noise(img_path):
    """Détecte si une image est du bruit sémantique"""
    path_str = str(img_path).lower()

    # Vérifier les mots-clés dans le chemin
    for keyword in SEMANTIC_NOISE_KEYWORDS:
        if keyword in path_str:
            return True

    # Heuristique: images très petites peuvent être des schémas
    try:
        img = Image.open(img_path)
        w, h = img.size
        if w < 200 or h < 200:
            return True
    except:
        pass

    return False

# ============================================================================
# 4. PLAFONNEMENT DES CLASSES DOMINANTES
# ============================================================================

def cap_class(images, count):
    """Détermine le plafond et sélectionne les images"""
    if count > 10000:
        max_images = MAX_DOMINANT
    elif count > 3000:
        max_images = MAX_LARGE
    elif count > 1000:
        max_images = MAX_MEDIUM
    else:
        max_images = count  # Garder toutes les images

    if count > max_images:
        # Sélection stratifiée aléatoire
        selected = random.sample(images, max_images)
        stats['classes_capped'] += (count - max_images)
        return selected
    else:
        return images

# ============================================================================
# 5. AUGMENTATION BIOLOGIQUEMENT VALIDE
# ============================================================================

# Augmentation avec Albumentations (biologiquement valide uniquement)
augmenter = A.Compose([
    A.Rotate(limit=15, p=0.7),  # Rotation ±15° max
    A.RandomBrightnessContrast(
        brightness_limit=0.2,    # ±20%
        contrast_limit=0.15,     # ±15%
        p=0.7
    ),
    A.ShiftScaleRotate(
        shift_limit=0.05,        # Translation légère
        scale_limit=0.1,         # Zoom 0.9-1.1
        rotate_limit=0,          # Pas de rotation supplémentaire
        p=0.7
    ),
    A.GaussNoise(
        std_limit=(2.0, 5.0),    # Écart-type du bruit gaussien faible
        p=0.3
    ),
    # Interdictions respectées:
    # - Pas de rotation 90°/180° (limit=15)
    # - Pas de modifications de couleur non réalistes (seulement brightness/contrast)
    # - Pas de mirroring vertical (pas de HorizontalFlip)
    # - Pas de style transfer (pas de ColorJitter agressif)
])

def get_augmentation_factor(count):
    """Détermine le facteur d'augmentation selon le nombre d'images"""
    if count >= MIN_STANDARD:
        return 1  # Pas d'augmentation nécessaire
    elif count >= MIN_RARE:
        return AUGMENTATION_FACTORS.get(50, 6)
    elif count >= MIN_CRITICAL:
        if count <= 20:
            return AUGMENTATION_FACTORS.get(20, 10)
        elif count <= 30:
            return AUGMENTATION_FACTORS.get(30, 8)
        else:
            return AUGMENTATION_FACTORS.get(50, 6)
    else:
        # Classe trop rare (< 20), utiliser le facteur max mais avec avertissement
        return AUGMENTATION_FACTORS.get(20, 10)

def augment_class(images, output_dir, class_name):
    """Augmente les images d'une classe si nécessaire"""
    count = len(images)

    if count == 0:
        return

    factor = get_augmentation_factor(count)

    # Copier les images originales
    for i, img_path in enumerate(images):
        try:
            dest_path = os.path.join(output_dir, f"real_{i:05d}.jpg")
            shutil.copy2(img_path, dest_path)
        except Exception as e:
            stats['processing_errors'].append(f"Erreur copie {img_path}: {e}")

    # Augmenter si nécessaire
    if factor > 1:
        target_count = min(300, count * factor)  # Plafonner à 300 max
        num_augmentations_per_image = max(1, (target_count - count) // count)

        # Limiter le nombre d'images à augmenter pour éviter explosion
        images_to_augment = min(len(images), 50)

        aug_count = 0
        for i, img_path in enumerate(images[:images_to_augment]):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                for k in range(num_augmentations_per_image):
                    aug = augmenter(image=img)['image']
                    aug_path = os.path.join(output_dir, f"aug_{i:05d}_{k:03d}.jpg")
                    cv2.imwrite(aug_path, aug)
                    aug_count += 1
            except Exception as e:
                stats['processing_errors'].append(f"Erreur augmentation {img_path}: {e}")
                continue

        stats['images_augmented'] += aug_count

# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

print("=" * 80)
print("🎯 PIPELINE DE CURATION DU DATASET")
print("=" * 80)
print(f"\n📁 Input:  {ROOT}")
print(f"📁 Output: {OUTPUT}")
print(f"📁 Quarantine: {QUARANTINE}")
print(f"📁 Reports: {REPORT_DIR}\n")

# Charger le checkpoint
checkpoint = load_checkpoint()
processed_classes = set(checkpoint.get('processed_classes', []))

# Parcourir toutes les catégories principales
categories = [d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))]

print(f"📊 {len(categories)} catégories principales trouvées\n")

total_subclasses = 0
for category in categories:
    category_path = os.path.join(ROOT, category)
    subclasses = [d for d in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, d))]
    total_subclasses += len(subclasses)

print(f"📊 {total_subclasses} sous-classes au total\n")

if processed_classes:
    print(f"📊 {len(processed_classes)} sous-classes déjà traitées, reprise depuis le dernier checkpoint\n")


# Parcourir chaque catégorie et ses sous-classes
for category in tqdm(categories, desc="Catégories"):
    category_path = os.path.join(ROOT, category)

    # Récupérer les sous-classes
    subclasses = [d for d in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, d))]

    if len(subclasses) == 0:
        continue

    print(f"\n📁 Catégorie: {category} ({len(subclasses)} sous-classes)")

    # Traiter chaque sous-classe
    for subclass in subclasses:
        cls_path = os.path.join(category_path, subclass)

        # Nom complet de la classe pour le checkpoint
        full_class_id = f"{category}/{subclass}"

        # Vérifier si cette classe a déjà été traitée
        if full_class_id in processed_classes:
            print(f"  ⏭️  {subclass} (déjà traitée, ignorée)")
            continue

        # Récupérer toutes les images de cette sous-classe
        all_images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            all_images.extend(glob.glob(os.path.join(cls_path, ext)))

        if len(all_images) == 0:
            continue

        # Nom complet de la classe (catégorie/sous-classe)
        full_class_name = f"{category}/{subclass}"

        print(f"\n  📂 Sous-classe: {subclass} ({len(all_images)} images)")

        # ÉTAPE 1: Filtrage qualité
        print("    🔍 Filtrage qualité...", end=" ")
        valid_images = []
        for img_path in all_images:
            is_valid, reason = is_valid_image(img_path)
            if is_valid:
                valid_images.append(img_path)
            else:
                # Déplacer vers quarantaine
                try:
                    quarantine_path = os.path.join(QUARANTINE, "quality_rejected", category, subclass, os.path.basename(img_path))
                    os.makedirs(os.path.dirname(quarantine_path), exist_ok=True)
                    shutil.move(img_path, quarantine_path)
                    stats['quality_rejected'] += 1
                except Exception as e:
                    stats['processing_errors'].append(f"Erreur déplacement {img_path}: {e}")

        print(f"{len(valid_images)}/{len(all_images)} conservées")

        # ÉTAPE 2: Suppression doublons
        print("    🔄 Suppression doublons...", end=" ")
        unique_images = remove_duplicates(valid_images)

        # Déplacer les doublons vers quarantaine
        duplicates = set(valid_images) - set(unique_images)
        for dup_path in duplicates:
            try:
                quarantine_path = os.path.join(QUARANTINE, "duplicates", category, subclass, os.path.basename(dup_path))
                os.makedirs(os.path.dirname(quarantine_path), exist_ok=True)
                if os.path.exists(dup_path):
                    shutil.move(dup_path, quarantine_path)
            except Exception as e:
                stats['processing_errors'].append(f"Erreur déplacement doublon {dup_path}: {e}")

        print(f"{len(unique_images)}/{len(valid_images)} uniques")

        # ÉTAPE 3: Nettoyage sémantique
        print("    🧹 Nettoyage sémantique...", end=" ")
        clean_images = []
        for img_path in unique_images:
            if not is_semantic_noise(img_path):
                clean_images.append(img_path)
            else:
                # Déplacer vers quarantaine
                try:
                    quarantine_path = os.path.join(QUARANTINE, "semantic_noise", category, subclass, os.path.basename(img_path))
                    os.makedirs(os.path.dirname(quarantine_path), exist_ok=True)
                    shutil.move(img_path, quarantine_path)
                    stats['semantic_noise_removed'] += 1
                except Exception as e:
                    stats['processing_errors'].append(f"Erreur déplacement {img_path}: {e}")

        print(f"{len(clean_images)}/{len(unique_images)} conservées")

        # ÉTAPE 4: Plafonnement
        print("    📉 Plafonnement...", end=" ")
        final_images = cap_class(clean_images, len(clean_images))

        # Déplacer les images excédentaires vers quarantaine
        capped = set(clean_images) - set(final_images)
        for capped_path in capped:
            try:
                quarantine_path = os.path.join(QUARANTINE, "capped", category, subclass, os.path.basename(capped_path))
                os.makedirs(os.path.dirname(quarantine_path), exist_ok=True)
                if os.path.exists(capped_path):
                    shutil.move(capped_path, quarantine_path)
            except Exception as e:
                stats['processing_errors'].append(f"Erreur déplacement {capped_path}: {e}")

        print(f"{len(final_images)}/{len(clean_images)} sélectionnées")

        # ÉTAPE 5: Augmentation et copie vers output
        print("    ✨ Augmentation...", end=" ")
        # Conserver la structure hiérarchique dans le dataset final
        out_cls = os.path.join(OUTPUT, category, subclass)
        os.makedirs(out_cls, exist_ok=True)

        augment_class(final_images, out_cls, full_class_name)

        # Compter les images finales
        final_count = len([f for f in os.listdir(out_cls) if f.endswith(('.jpg', '.jpeg', '.png'))])
        stats['class_distribution'][full_class_name] = final_count
        stats['final_images'] += final_count

        print(f"{final_count} images finales")

        # Mettre à jour le checkpoint pour cette sous-classe
        checkpoint['processed_classes'].append(full_class_id)
        save_checkpoint(checkpoint)

# ============================================================================
# GÉNÉRATION DU RAPPORT
# ============================================================================

print("\n" + "=" * 80)
print("📄 Génération du rapport...")
print("=" * 80)

report = {
    'timestamp': datetime.now().isoformat(),
    'configuration': {
        'min_resolution': MIN_RES,
        'blur_threshold': BLUR_THRESHOLD,
        'phash_threshold': PHASH_THRESHOLD,
        'max_dominant': MAX_DOMINANT,
        'max_large': MAX_LARGE,
        'max_medium': MAX_MEDIUM,
        'augmentation_factors': AUGMENTATION_FACTORS
    },
    'statistics': stats,
    'summary': {
        'total_scanned': stats['total_scanned'],
        'quality_rejected': stats['quality_rejected'],
        'duplicates_removed': stats['duplicates_removed'],
        'semantic_noise_removed': stats['semantic_noise_removed'],
        'classes_capped': stats['classes_capped'],
        'images_augmented': stats['images_augmented'],
        'final_images': stats['final_images'],
        'total_classes': len(stats['class_distribution'])
    },
    'class_distribution': stats['class_distribution'],
    'quality_rejection_reasons': dict(stats['quality_reasons']),
    'errors': stats['processing_errors'][:100]  # Limiter à 100 erreurs
}

# Sauvegarder JSON
report_path = os.path.join(REPORT_DIR, f"curation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

# Sauvegarder rapport texte
txt_report_path = report_path.replace('.json', '.txt')
with open(txt_report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("RAPPORT DE CURATION DU DATASET\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Date: {report['timestamp']}\n\n")
    f.write("RÉSUMÉ\n")
    f.write("-" * 80 + "\n")
    f.write(f"Images scannées: {report['summary']['total_scanned']:,}\n")
    f.write(f"Images rejetées (qualité): {report['summary']['quality_rejected']:,}\n")
    f.write(f"Doublons supprimés: {report['summary']['duplicates_removed']:,}\n")
    f.write(f"Bruit sémantique supprimé: {report['summary']['semantic_noise_removed']:,}\n")
    f.write(f"Images plafonnées: {report['summary']['classes_capped']:,}\n")
    f.write(f"Images augmentées générées: {report['summary']['images_augmented']:,}\n")
    f.write(f"Images finales: {report['summary']['final_images']:,}\n")
    f.write(f"Classes finales: {report['summary']['total_classes']}\n\n")

    f.write("DISTRIBUTION PAR CLASSE\n")
    f.write("-" * 80 + "\n")
    for class_name, count in sorted(report['class_distribution'].items(),
                                   key=lambda x: x[1], reverse=True):
        f.write(f"{class_name}: {count:,}\n")

    if report['quality_rejection_reasons']:
        f.write("\nRAISONS DE REJET (QUALITÉ)\n")
        f.write("-" * 80 + "\n")
        for reason, count in report['quality_rejection_reasons'].items():
            f.write(f"{reason}: {count:,}\n")

    if report['errors']:
        f.write(f"\nERREURS ({len(report['errors'])} premières)\n")
        f.write("-" * 80 + "\n")
        for error in report['errors']:
            f.write(f"{error}\n")

    f.write("\n" + "=" * 80 + "\n")

print(f"✓ Rapport JSON: {report_path}")
print(f"✓ Rapport texte: {txt_report_path}")

# ============================================================================
# RÉSUMÉ FINAL
# ============================================================================

print("\n" + "=" * 80)
print("✅ PIPELINE TERMINÉ AVEC SUCCÈS")
print("=" * 80)
print(f"\n📊 Statistiques:")
print(f"  • Images scannées: {stats['total_scanned']:,}")
print(f"  • Images rejetées (qualité): {stats['quality_rejected']:,}")
print(f"  • Doublons supprimés: {stats['duplicates_removed']:,}")
print(f"  • Bruit sémantique: {stats['semantic_noise_removed']:,}")
print(f"  • Images plafonnées: {stats['classes_capped']:,}")
print(f"  • Images augmentées: {stats['images_augmented']:,}")
print(f"  • Images finales: {stats['final_images']:,}")
print(f"  • Classes finales: {len(stats['class_distribution'])}")
print(f"\n📁 Dataset final: {OUTPUT}")
print(f"📁 Images rejetées: {QUARANTINE}")
print(f"📁 Rapports: {REPORT_DIR}")
print("\n✅ DATASET FINAL PRÊT POUR MODÈLE PRINCIPAL")



