"""
🎯 DINOv2 — VERSION 5.1 (NIVEAU 95%+)
======================================
V5 + ajustements supplémentaires :

  • w_canon dynamique 0 → max (souvent max=0.10) — évite fusion classes fines
  • Stage-2 backbone LR bas (lr_backbone_stage2_mult ~0.08)
  • Logs LR réels + global_epoch ; cudnn.benchmark ; workers seedés
  • Sélection best checkpoint : EMA rapide OU EMA+TTA tous les N epochs (aligné final)
  • Mix rare décroissant 0.7 → 0.4 → 0.2 avant la queue fine-tune
  • Label smoothing défaut 0.05
  • Sauvegarde best_model_s{seed}.pt — ensemble multi-seed (CONFIG['ensemble_checkpoints'])

PRÉ-REQUIS : 01 + 02 + 02b exécutés
"""

# ============================================================================
# CELL 1 : MOUNT
# ============================================================================
from google.colab import drive
drive.mount('/content/drive')

# ============================================================================
# CELL 2 : INSTALL
# ============================================================================
import subprocess
for pkg in ["torch torchvision", "timm", "opencv-python",
            "albumentations", "scikit-learn", "tqdm",
            "tensorboard", "matplotlib", "seaborn"]:
    subprocess.run(f"pip install -q {pkg}", shell=True)
print("✅ Dépendances installées")

# ============================================================================
# CELL 3 : IMPORTS & CONFIG
# ============================================================================
import os, json, random, math
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast

import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"   GPU  : {torch.cuda.get_device_name(0)}")
    print(f"   VRAM : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    torch.backends.cudnn.benchmark = True

META_DIR = Path('/content/drive/MyDrive/Plantdataset_metadata')
OUT_DIR  = Path('/content/drive/MyDrive/models_dinov2_v5')
CKPT_DIR = Path('/content/drive/MyDrive/checkpoints_dinov2_v5')
LOG_DIR  = Path('/content/drive/MyDrive/logs_dinov2_v5')
for d in [OUT_DIR, CKPT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def _seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Sera ré-appliqué après CONFIG['seed']
_seed_all(42)

def load_json(p):
    with open(p, 'r') as f:
        return json.load(f)

# ── Métadonnées ───────────────────────────────────────────────────────────
class_report     = load_json(META_DIR / 'class_report.json')
class_groups     = load_json(META_DIR / 'class_groups.json')
class_mapping    = load_json(META_DIR / 'class_mapping.json')
class_hierarchy  = load_json(META_DIR / 'class_hierarchy.json')
phase_groups     = load_json(META_DIR / 'phase_groups.json')
multitask_cfg    = load_json(META_DIR / 'multitask_config.json')
training_cfg     = load_json(META_DIR / 'training_config.json')
class_weights    = load_json(META_DIR / 'class_weights_log.json')
clean_label_map  = load_json(META_DIR / 'clean_label_map.json')  # AMÉLIORATION 4

class_to_idx = class_mapping['class_to_idx']
idx_to_class = {int(k): v for k, v in class_mapping['idx_to_class'].items()}
NUM_CLASSES  = len(class_to_idx)

crop_to_idx     = multitask_cfg['crop_to_idx']
category_to_idx = multitask_cfg['category_to_idx']
NUM_CROPS       = multitask_cfg['num_crops']
NUM_CATEGORIES  = multitask_cfg['num_categories']
LOSS_WEIGHTS    = multitask_cfg['loss_weights']  # main=1.0 / crop=0.2 / cat=0.15

core_classes     = set(class_groups['CORE'])
extended_classes = set(class_groups['EXTENDED'])
rare_classes     = set(class_groups['RARE'])

# ── Mapping alias → index canonical (pour CleanLabelLoss) ─────────────────
canonical_idx = {}   # {alias_idx: canonical_idx}
for alias, canonical in clean_label_map.items():
    ai = class_to_idx.get(alias)
    ci = class_to_idx.get(canonical)
    if ai is not None and ci is not None:
        canonical_idx[ai] = ci
print(f"✅ {len(canonical_idx)} paires alias/canonical chargées")

CONFIG = {
    # Profils: "cost_stable" (coût réduit) | "balanced" | "ultra_solid"
    'training_profile': training_cfg.get('training_profile', 'cost_stable'),
    # Fine-tuning mode: feature_extract | partial | full
    'fine_tune_mode': training_cfg.get('fine_tune_mode', 'partial'),
    'backbone':       training_cfg.get('backbone',        'dinov2_vits14'),  # Smaller model for cost reduction
    'image_size':     training_cfg.get('input_size',      224),  # Reduced image size
    'embed_dim':      training_cfg.get('embed_dim',       384),  # Adjusted for ViT-S
    'num_classes':    NUM_CLASSES,
    'num_crops':      NUM_CROPS,
    'num_categories': NUM_CATEGORIES,
    'batch_size':     training_cfg.get('batch_size',      16),  # Reduced for cost
    'num_epochs':     training_cfg.get('epochs',          36),
    'warmup_epochs':  training_cfg.get('warmup_epochs',   5),
    'lr_head':        training_cfg.get('lr_head',         1e-4),
    'lr_backbone':    training_cfg.get('lr_backbone',     1e-5),
    'weight_decay':   training_cfg.get('weight_decay',    0.05),
    'label_smoothing':training_cfg.get('label_smoothing', 0.05),
    'focal_gamma':    training_cfg.get('focal_gamma',     2.0),
    'focal_weight':   0.7,
    'smooth_weight':  0.3,
    'mixup_alpha':    0.3,
    'cutmix_alpha':   1.0,
    'patience':       8,
    'num_workers':    4,
    # AMÉLIORATION 1 — progressive unfreezing
    # DINOv2 ViT-B = 12 blocs. On dégèle progressivement du dernier au premier.
    'total_blocks':   12,       # ViT-B (mettre 24 pour ViT-L)
    'unfreeze_stage1_epoch': 5,  # dégeler blocs 8-11 (last 4)
    'unfreeze_stage2_epoch': 15, # dégeler blocs 0-11 (full)
    # AMÉLIORATION 3 — hard mining (V5 : réaction chaque epoch)
    'hard_mining_start_epoch': 5,   # commencer après warm-up
    'hard_mining_update_freq': 1,   # reconstruire le sampler tous les N epochs
    'hard_mining_ema_alpha':   0.4, # EMA pour lisser les taux d'erreur
    'hard_mining_boost':       3.0, # multiplicateur max pour classes dures
    'hard_mining_boost_cap':   3.5, # cap global du boost factor (stabilité sur labels bruités)
    # AMÉLIORATION 4 — clean label (dual batch-wide) : w dynamique 0 → max (évite fusion classes fines)
    'canonical_penalty_max':    0.10,   # plafond w (tester 0.08–0.12 si besoin)
    'canonical_penalty_ramp_end_epoch': None,  # None = utilise unfreeze_stage2_epoch
    # MixUp / CutMix : décroissance 0.7 → 0.4 → 0.2 (0 = tail / finetune)
    'mix_rare_fraction':      0.7,  # valeur initiale (écrasée par schedule si epoch < tail)
    'mix_schedule_epoch_1':   20,    # avant : 0.7, entre e1 et e2 : 0.4, après e2 : 0.2
    'mix_schedule_epoch_2':   40,
    # Curriculum — replay CORE en phase 2 & 3 (fraction du volume « phase »)
    'core_replay_fraction':   0.25,  # ~25 % du jeu = samples CORE additionnels
    # Fine-tuning final (sans mix lourd)
    'finetune_tail_epochs':   5,
    'finetune_lr_mult':       0.1,   # multiplicateur LR une fois en début de tail
    'tail_scheduler_enabled': False, # False recommandé pour calibration finale
    # Stage 2 backbone : conservateur (souvent +2–4 % vs LR trop haut)
    'lr_backbone_stage2_mult': 0.08,  # cible 0.05–0.10 ; était 0.20
    # Sélection du meilleur modèle : TTA tous les N epochs (aligné métrique finale)
    # Recommandé : 3 à 5 selon budget GPU (5 = plus rapide)
    'tta_checkpoint_every':   10,
    # TTA pondérée (somme = 1.0)
    'tta_weights': {
        'orig': 0.40,
        'flip': 0.25,
        'crop': 0.20,
        'scale': 0.15,
    },
    # Calibration température post-training (sur val, EMA+TTA)
    'temperature_scaling': False,
    # Reproductibilité & ensemble multi-seed (2e run avec seed 1337 puis moyenne logits)
    'seed':                     42,
    # ViT-L : si VRAM OK, mettre backbone dinov2_vitl14 + total_blocks 24
    'ensemble_checkpoints':     [],  # ex. [Path(...)/best_s42.pt, Path(...)/best_s1337.pt]
    # Reprise entraînement
    'auto_resume':              True,
    'resume_checkpoint_name':   'resume_state.pt',
    'intra_epoch_resume':       True,   # reprise au batch près
    'resume_save_every_steps':  1000,    # sauvegarde reprise toutes les N itérations train
    # Optimisation stabilité/perf
    'grad_accum_steps':         training_cfg.get('grad_accum_steps', 1),
    'ema_decay':                training_cfg.get('ema_decay', 0.9999),
    # Unfreeze / LR (configurables)
    'stage1_unfreeze_blocks':   training_cfg.get('stage1_unfreeze_blocks', 4),
    'stage1_backbone_lr_mult':  training_cfg.get('stage1_backbone_lr_mult', 0.1),
    'stage1_head_lr_mult':      training_cfg.get('stage1_head_lr_mult', 0.5),
    'stage2_head_lr_mult':      training_cfg.get('stage2_head_lr_mult', 0.4),
}
if CONFIG['canonical_penalty_ramp_end_epoch'] is None:
    CONFIG['canonical_penalty_ramp_end_epoch'] = CONFIG['unfreeze_stage2_epoch']

# ================== PROFILE BOOST ==================
if CONFIG['training_profile'] == 'ultra_solid':
    # Si possible, pousser le backbone en ViT-L pour meilleure séparation fine.
    if CONFIG['backbone'] == 'dinov2_vitb14':
        CONFIG['backbone'] = 'dinov2_vitl14'
    if 'vitl14' in CONFIG['backbone']:
        CONFIG['embed_dim'] = 1024
        CONFIG['total_blocks'] = 24
        # 518 + ViT-L est coûteux; 384 est plus robuste côté mémoire/temps.
        if CONFIG['image_size'] > 384:
            CONFIG['image_size'] = 384
        if CONFIG['batch_size'] > 12:
            CONFIG['batch_size'] = 12
        CONFIG['grad_accum_steps'] = max(CONFIG['grad_accum_steps'], 2)

    CONFIG['num_epochs'] = max(CONFIG['num_epochs'], 80)
    CONFIG['patience'] = max(CONFIG['patience'], 12)
    CONFIG['tta_checkpoint_every'] = min(CONFIG['tta_checkpoint_every'], 3)
    CONFIG['finetune_tail_epochs'] = max(CONFIG['finetune_tail_epochs'], 10)
    CONFIG['finetune_lr_mult'] = min(CONFIG['finetune_lr_mult'], 0.05)
    CONFIG['mix_schedule_epoch_1'] = min(CONFIG['mix_schedule_epoch_1'], 15)
    CONFIG['mix_schedule_epoch_2'] = min(CONFIG['mix_schedule_epoch_2'], 35)
    CONFIG['hard_mining_boost_cap'] = min(CONFIG['hard_mining_boost_cap'], 3.0)
    CONFIG['canonical_penalty_max'] = min(CONFIG['canonical_penalty_max'], 0.08)
elif CONFIG['training_profile'] == 'cost_stable':
    # Réglage coût réduit, stabilité correcte
    CONFIG['fine_tune_mode'] = 'partial'
    CONFIG['backbone'] = 'dinov2_vits14'
    CONFIG['embed_dim'] = 384  # For ViT-S
    CONFIG['total_blocks'] = 12
    # DINOv2 patch=14 => taille image doit être multiple de 14
    target_img = min(CONFIG['image_size'], 336)
    CONFIG['image_size'] = max(14, (target_img // 14) * 14)
    CONFIG['batch_size'] = min(CONFIG['batch_size'], 16)  # Further reduced for cost
    CONFIG['num_epochs'] = min(CONFIG['num_epochs'], 20)  # Reduced epochs for cost
    CONFIG['patience'] = min(CONFIG['patience'], 8)
    CONFIG['tta_checkpoint_every'] = max(CONFIG['tta_checkpoint_every'], 10)
    CONFIG['temperature_scaling'] = False
    CONFIG['grad_accum_steps'] = max(1, min(CONFIG['grad_accum_steps'], 2))

_seed_all(CONFIG['seed'])
print(f"✅ Config | profile={CONFIG['training_profile']} | {NUM_CLASSES} classes | "
      f"backbone={CONFIG['backbone']} | img={CONFIG['image_size']} | "
      f"batch={CONFIG['batch_size']} | accum={CONFIG['grad_accum_steps']} | seed={CONFIG['seed']}")


def get_canonical_penalty_weight(epoch: int) -> float:
    """w : 0 → canonical_penalty_max sur la rampe (jusqu'à ramp_end_epoch)."""
    cap = CONFIG['canonical_penalty_max']
    end = max(1, CONFIG['canonical_penalty_ramp_end_epoch'])
    t = min(1.0, float(epoch) / float(end))
    return t * cap


def get_mix_rare_fraction(epoch: int) -> float:
    """0.7 → 0.4 → 0.2 ; tail / finetune = 0 (géré par force_identity)."""
    tail = max(0, CONFIG['num_epochs'] - CONFIG['finetune_tail_epochs'])
    if epoch >= tail:
        return 0.0
    e1 = CONFIG['mix_schedule_epoch_1']
    e2 = CONFIG['mix_schedule_epoch_2']
    if epoch < e1:
        return 0.7
    if epoch < e2:
        return 0.4
    return 0.2


def log_learning_rates(optimizer, global_epoch: int):
    bb = optimizer.param_groups[0]['lr']
    hd = optimizer.param_groups[1]['lr']
    print(f"  📈 global_epoch={global_epoch} | lr_backbone={bb:.2e} | lr_head={hd:.2e}")


def _worker_init_fn(worker_id: int):
    s = CONFIG['seed'] + worker_id
    np.random.seed(s)
    random.seed(s)


# ============================================================================
# CELL 4 : AUGMENTATIONS
# ============================================================================
IMG_SIZE  = CONFIG['image_size']
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]

def _rrc(size, scale):
    """
    Compat Albumentations v1/v2:
    - v1: RandomResizedCrop(height=..., width=..., ...)
    - v2: RandomResizedCrop(size=(h, w), ...)
    """
    try:
        return A.RandomResizedCrop(size=(size, size), scale=scale)
    except TypeError:
        return A.RandomResizedCrop(height=size, width=size, scale=scale)


def _coarse_dropout():
    """Compat CoarseDropout v1/v2."""
    try:
        return A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(8, 32),
            hole_width_range=(8, 32),
            p=0.3,
        )
    except TypeError:
        return A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3)


transform_core = A.Compose([
    _rrc(IMG_SIZE, (0.7, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, p=0.4),
    A.RandomGamma(p=0.2),
    A.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ToTensorV2(),
])
transform_extended = A.Compose([
    _rrc(IMG_SIZE, (0.5, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),
    A.Rotate(limit=30, p=0.5),
    A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05, p=0.5),
    A.OneOf([A.GaussianBlur(p=0.4), A.MotionBlur(p=0.3)], p=0.3),
    A.GaussNoise(p=0.2),
    A.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ToTensorV2(),
])
transform_rare = A.Compose([
    _rrc(IMG_SIZE, (0.4, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.Rotate(limit=45, p=0.7),
    A.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.35, hue=0.1, p=0.7),
    A.Affine(rotate=(-30, 30), translate_percent=(0.15, 0.15), scale=(0.85, 1.15), p=0.5),
    A.OneOf([A.GaussianBlur(p=0.3), A.MotionBlur(p=0.3)], p=0.4),
    A.GaussNoise(p=0.3),
    _coarse_dropout(),
    A.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ToTensorV2(),
])
val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ToTensorV2(),
])
print("✅ Transforms adaptatifs CORE / EXTENDED / RARE")


# ============================================================================
# CELL 5 : DATASET
# ============================================================================
class AgriDataset(Dataset):
    def __init__(self, meta_list, active_class_set=None, is_train=True):
        self.is_train = is_train
        self.light_aug = False  # V5 : True = val_transform pendant train (fine-tune tail)
        if active_class_set:
            meta_list = [x for x in meta_list if x['class'] in active_class_set]

        self.paths = []; self.labels = []
        self.crop_labels = []; self.cat_labels = []
        self.canonical_labels = []   # label canonique (pour CleanLabelLoss)
        self.groups = []

        for item in meta_list:
            cls = item['class']
            lbl = item['label']
            self.paths.append(item['path'])
            self.labels.append(lbl)
            self.crop_labels.append(item.get('crop_label', -1))
            self.cat_labels.append(item.get('category_label', -1))
            # Canonical = classe fusionnée si alias, sinon soi-même
            self.canonical_labels.append(canonical_idx.get(lbl, lbl))
            if cls in rare_classes:
                self.groups.append('rare')
            elif cls in extended_classes:
                self.groups.append('extended')
            else:
                self.groups.append('core')

        cnt = Counter(self.groups)
        print(f"  Dataset: {len(self.paths):,} imgs | "
              f"core={cnt['core']} extended={cnt['extended']} rare={cnt['rare']}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])
        img = (np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
               if img is None
               else cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        grp = self.groups[idx]
        if self.is_train and not self.light_aug:
            aug = (transform_rare     if grp == 'rare'
                   else transform_extended if grp == 'extended'
                   else transform_core)(image=img)
        else:
            aug = val_transform(image=img)

        return (aug['image'],
                self.labels[idx],
                self.crop_labels[idx],
                self.cat_labels[idx],
                self.canonical_labels[idx],
                grp)


# ============================================================================
# CELL 6 : HARD MINER (amélioration 3)
# ============================================================================
class HardMiner:
    """
    Maintient un taux d'erreur EMA par classe.
    Fournit des poids de sampling boostés pour les classes mal classées.
    Mis à jour après chaque epoch de validation.

    Formule : sample_weight[cls] = base_weight[cls] * boost_factor[cls]
    boost_factor = 1 + (error_rate * hard_mining_boost)
    """
    def __init__(self, num_classes: int, base_weights: dict,
                 class_to_idx: dict, idx_to_class: dict,
                 ema_alpha: float = 0.4, boost: float = 3.0):
        self.num_classes  = num_classes
        self.idx_to_class = idx_to_class
        self.ema_alpha    = ema_alpha
        self.boost        = boost

        # Taux d'erreur EMA (initialisé à 0.5 = incertitude totale)
        self.error_rates = np.full(num_classes, 0.5, dtype=np.float32)

        # Poids de base (log-based)
        self.base_w = np.ones(num_classes, dtype=np.float32)
        for cls, w in base_weights.items():
            if cls in class_to_idx:
                self.base_w[class_to_idx[cls]] = float(w)

    def update(self, preds: np.ndarray, labels: np.ndarray):
        """Mise à jour EMA des taux d'erreur par classe."""
        for cls_idx in range(self.num_classes):
            mask = labels == cls_idx
            if not mask.any():
                continue
            err = float((preds[mask] != cls_idx).mean())
            self.error_rates[cls_idx] = (
                self.ema_alpha * err
                + (1 - self.ema_alpha) * self.error_rates[cls_idx]
            )

    def get_sample_weights(self, dataset_labels: list) -> torch.DoubleTensor:
        """Calcule le poids final pour chaque sample du dataset."""
        boost_factor = 1.0 + self.error_rates * self.boost
        boost_factor = np.clip(boost_factor, 1.0, CONFIG['hard_mining_boost_cap'])
        sample_w = [
            float(self.base_w[lbl]) * float(boost_factor[lbl])
            for lbl in dataset_labels
        ]
        return torch.DoubleTensor(sample_w)

    def get_hardest_classes(self, n=10) -> list:
        """Retourne les n classes avec le taux d'erreur le plus élevé."""
        top_idx = np.argsort(self.error_rates)[::-1][:n]
        return [(self.idx_to_class.get(int(i), str(i)),
                 float(self.error_rates[i]))
                for i in top_idx]


# ============================================================================
# CELL 7 : MIXUP + CUTMIX (amélioration 2)
# ============================================================================
def rand_bbox(H: int, W: int, lam: float):
    """Calcule une bounding box aléatoire pour CutMix."""
    cut_rat = math.sqrt(1.0 - lam)
    cut_h = int(H * cut_rat)
    cut_w = int(W * cut_rat)
    cx = random.randint(0, W)
    cy = random.randint(0, H)
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)
    return x1, y1, x2, y2


def apply_cutmix(images, labels, alpha=1.0):
    """CutMix standard sur tout le batch."""
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(images.size(0))
    H, W = images.shape[2], images.shape[3]
    x1, y1, x2, y2 = rand_bbox(H, W, lam)
    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]
    # Ajuste lambda en fonction de la surface réelle coupée
    lam = 1.0 - (x2 - x1) * (y2 - y1) / (H * W)
    return mixed, labels, labels[idx], lam


def apply_mixup(images, labels, alpha=0.3):
    """MixUp standard sur tout le batch."""
    lam = max(np.random.beta(alpha, alpha), 0.5)
    idx = torch.randperm(images.size(0))
    mixed = lam * images + (1 - lam) * images[idx]
    return mixed, labels, labels[idx], lam


def smart_augment(images, labels, crop_lbl, cat_lbl, canonical_lbl, groups,
                  mixup_alpha, cutmix_alpha, rare_mix_prob=0.7,
                  force_identity=False):
    """
    MixUp ou CutMix. V5 : avec prob. rare_mix_prob, focus RARE (comme avant) ;
    sinon mix sur tout le batch pour éviter le biais « perturbé = rare ».
    """
    if force_identity:
        return (images, labels, labels,
                crop_lbl, crop_lbl,
                cat_lbl, cat_lbl,
                canonical_lbl, canonical_lbl,
                1.0, 'none')

    strategy = random.choice(['mixup', 'cutmix'])
    rare_only = random.random() < rare_mix_prob

    if rare_only:
        rare_mask = torch.tensor([g == 'rare' for g in groups], dtype=torch.bool)
        if rare_mask.sum() < 2:
            return (images, labels, labels,
                    crop_lbl, crop_lbl,
                    cat_lbl, cat_lbl,
                    canonical_lbl, canonical_lbl,
                    1.0, strategy)
        # Appliquer seulement sur les RARE, garder les autres intacts
        rare_imgs   = images[rare_mask]
        rare_labels = labels[rare_mask]
        if strategy == 'cutmix':
            aug_imgs, la, lb, lam = apply_cutmix(rare_imgs, rare_labels, cutmix_alpha)
        else:
            aug_imgs, la, lb, lam = apply_mixup(rare_imgs, rare_labels, mixup_alpha)
        mixed = images.clone()
        mixed[rare_mask] = aug_imgs
        la_full = labels.clone()
        lb_full = labels.clone()
        la_full[rare_mask] = la
        lb_full[rare_mask] = lb
        return (mixed, la_full, lb_full,
                crop_lbl, crop_lbl,
                cat_lbl, cat_lbl,
                canonical_lbl, canonical_lbl,
                lam, strategy)
    else:
        if strategy == 'cutmix':
            mixed, la, lb, lam = apply_cutmix(images, labels, cutmix_alpha)
        else:
            mixed, la, lb, lam = apply_mixup(images, labels, mixup_alpha)
        idx = torch.randperm(images.size(0))
        return (mixed, la, lb,
                crop_lbl, crop_lbl[idx],
                cat_lbl, cat_lbl[idx],
                canonical_lbl, canonical_lbl[idx],
                lam, strategy)


# ============================================================================
# CELL 8 : LOSS FUNCTIONS
# ============================================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, logits, targets):
        log_p = F.log_softmax(logits, dim=1)
        p_t   = log_p.exp().gather(1, targets.unsqueeze(1)).squeeze(1)
        focal = (1.0 - p_t) ** self.gamma
        ce    = F.nll_loss(log_p, targets, weight=self.weight, reduction='none')
        return (focal * ce).mean()


class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight    = weight

    def forward(self, logits, targets):
        n     = logits.size(1)
        log_p = F.log_softmax(logits, dim=1)
        with torch.no_grad():
            smooth = torch.full_like(logits, self.smoothing / n)
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        loss = -(smooth * log_p)
        if self.weight is not None:
            loss = loss * self.weight.unsqueeze(0)
        return loss.sum(dim=1).mean()


class ComboLoss(nn.Module):
    """0.7 × FocalLoss + 0.3 × LabelSmoothingCE"""
    def __init__(self, focal_gamma, weight, label_smoothing,
                 focal_weight=0.7, smooth_weight=0.3):
        super().__init__()
        self.focal  = FocalLoss(gamma=focal_gamma, weight=weight)
        self.smooth = LabelSmoothingCE(smoothing=label_smoothing, weight=weight)
        self.alpha  = focal_weight
        self.beta   = smooth_weight

    def forward(self, logits, targets):
        return self.alpha * self.focal(logits, targets) \
             + self.beta  * self.smooth(logits, targets)


class CleanLabelLoss(nn.Module):
    """
    V5 — Dual loss batch-wide (contrainte globale sur l'espace latent) :

        L = (1 - w) * L(exact) + w * L(canonical)

    `l_canon` est toujours calculée sur tout le batch (targets = canoniques).
    Pour les classes non-alias, exact == canon → les deux termes cohérents.
    """
    def __init__(self, base_loss: nn.Module, penalty_weight: float = 0.10):
        super().__init__()
        self.base_loss      = base_loss
        self.penalty_weight = penalty_weight

    def forward(self, logits, targets, canonical_targets=None,
                penalty_weight=None):
        w = self.penalty_weight if penalty_weight is None else penalty_weight
        l_exact = self.base_loss(logits, targets)
        if canonical_targets is None or w == 0:
            return l_exact
        l_canon = self.base_loss(logits, canonical_targets)
        return (1 - w) * l_exact + w * l_canon


# ── Instanciation ─────────────────────────────────────────────────────────
w_tensor = torch.ones(NUM_CLASSES)
for cls, wval in class_weights.items():
    if cls in class_to_idx:
        w_tensor[class_to_idx[cls]] = float(wval)
w_tensor = w_tensor.to(DEVICE)

combo_loss = ComboLoss(
    focal_gamma     = CONFIG['focal_gamma'],
    weight          = w_tensor,
    label_smoothing = CONFIG['label_smoothing'],
    focal_weight    = CONFIG['focal_weight'],
    smooth_weight   = CONFIG['smooth_weight'],
).to(DEVICE)

criterion_main = CleanLabelLoss(
    base_loss      = combo_loss,
    penalty_weight = CONFIG['canonical_penalty_max'],
).to(DEVICE)

criterion_aux = nn.CrossEntropyLoss(label_smoothing=0.05).to(DEVICE)

print(f"✅ CleanLabelLoss dual batch-wide | w_canon max={CONFIG['canonical_penalty_max']} (rampe 0→max)")


# ============================================================================
# CELL 9 : ARCHITECTURE — DINOv2 PROGRESSIVE UNFREEZING (amélioration 1)
# ============================================================================
class DINOv2Multitask(nn.Module):
    """
    DINOv2 avec dégel progressif par blocs (V5.1 LR : stage1 bb×0.1, stage2 bb×0.08).

      Stage 0 → backbone gelé, têtes actives
      Stage 1 → last 4 blocs + norm
      Stage 2 → backbone entier
    """
    def __init__(self, backbone_name, num_classes, num_crops,
                 num_categories, embed_dim, total_blocks=12):
        super().__init__()
        print(f"  📥 Chargement {backbone_name}...")
        self.backbone = torch.hub.load(
            'facebookresearch/dinov2', backbone_name, pretrained=True)
        if hasattr(self.backbone, 'head'):
            self.backbone.head = nn.Identity()

        self.embed_dim    = embed_dim
        self.total_blocks = total_blocks

        # Geler tout le backbone au départ
        self._freeze_all_backbone()

        # Bottleneck partagé
        self.shared_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        # Tête principale
        self.head_main = nn.Sequential(
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )
        # Têtes auxiliaires
        self.head_crop     = nn.Sequential(
            nn.Linear(512, 128), nn.GELU(), nn.Linear(128, num_crops))
        self.head_category = nn.Sequential(
            nn.Linear(512, 64), nn.GELU(), nn.Linear(64, num_categories))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        total  = sum(p.numel() for p in self.parameters())
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        print(f"  ✅ {total/1e6:.1f}M params | {frozen/1e6:.1f}M gelés (stage 0)")

    def _freeze_all_backbone(self):
        """Stage 0 : geler tout le backbone."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_last_n_blocks(self, n: int):
        """
        Stage 1 : dégeler les n derniers blocs transformer + norm finale.
        Laisse patch_embed, pos_embed, cls_token, premiers blocs gelés.
        """
        blocks = getattr(self.backbone, 'blocks', None)
        if blocks is None:
            # fallback : dégeler tout
            self.unfreeze_full()
            return
        start = max(0, len(blocks) - n)
        for i, blk in enumerate(blocks):
            if i >= start:
                for p in blk.parameters():
                    p.requires_grad = True
        # Dégeler la norm finale si présente
        for name, mod in self.backbone.named_modules():
            if 'norm' in name.lower() and isinstance(mod, (nn.LayerNorm, nn.BatchNorm1d)):
                for p in mod.parameters():
                    p.requires_grad = True
        frozen = sum(p.numel() for p in self.backbone.parameters() if not p.requires_grad)
        free   = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        print(f"  🔓 Stage 1 — last {n} blocs dégelés | "
              f"libre={free/1e6:.1f}M gelé={frozen/1e6:.1f}M")

    def unfreeze_full(self):
        """Stage 2 : dégeler tout le backbone."""
        for p in self.backbone.parameters():
            p.requires_grad = True
        total = sum(p.numel() for p in self.backbone.parameters())
        print(f"  🔓 Stage 2 — backbone entier dégelé ({total/1e6:.1f}M)")

    def forward(self, x):
        feat = self.backbone(x)
        if feat.dim() == 3:
            feat = feat[:, 0]
        shared = self.shared_proj(feat)
        return {
            'main':     self.head_main(shared),
            'crop':     self.head_crop(shared),
            'category': self.head_category(shared),
        }

    def get_features(self, x):
        with torch.no_grad():
            feat = self.backbone(x)
            if feat.dim() == 3:
                feat = feat[:, 0]
        return F.normalize(feat, p=2, dim=1)


model = DINOv2Multitask(
    backbone_name  = CONFIG['backbone'],
    num_classes    = CONFIG['num_classes'],
    num_crops      = CONFIG['num_crops'],
    num_categories = CONFIG['num_categories'],
    embed_dim      = CONFIG['embed_dim'],
    total_blocks   = CONFIG['total_blocks'],
).to(DEVICE)


# ============================================================================
# CELL 10 : HARD MINER + SAMPLER BUILDER
# ============================================================================
hard_miner = HardMiner(
    num_classes  = NUM_CLASSES,
    base_weights = class_weights,
    class_to_idx = class_to_idx,
    idx_to_class = idx_to_class,
    ema_alpha    = CONFIG['hard_mining_ema_alpha'],
    boost        = CONFIG['hard_mining_boost'],
)


def build_train_meta_with_core_replay(train_mt, phase_key, active_set):
    """
    V5 — En phase_2 / phase_3, mélange des samples CORE (replay) pour limiter le forgetting.
    """
    phase_items = [x for x in train_mt if x['class'] in active_set]
    if phase_key not in ('phase_2', 'phase_3'):
        return phase_items
    frac = CONFIG['core_replay_fraction']
    core_pool = [x for x in train_mt if x['class'] in core_classes]
    if not core_pool or not phase_items:
        return phase_items
    n_core = int(len(phase_items) * frac / max(1e-6, (1.0 - frac)))
    n_core = min(n_core, len(core_pool))
    if n_core <= 0:
        return phase_items
    core_replay = random.sample(core_pool, n_core)
    return phase_items + core_replay


def loader_class_filter(phase_key, active_set):
    """Classes autorisées dans le Dataset quand le meta inclut du replay CORE."""
    if phase_key in ('phase_2', 'phase_3'):
        return active_set | core_classes
    return active_set


def build_loader(meta_list, active_class_set, is_train, batch_size,
                 dataset_ref=None, use_hard_miner=False):
    """
    Construit un DataLoader.
    Si use_hard_miner=True et dataset_ref fourni → utilise les poids du miner.
    """
    ds = AgriDataset(meta_list, active_class_set, is_train)

    if is_train:
        if use_hard_miner and dataset_ref is not None:
            weights = hard_miner.get_sample_weights(ds.labels)
        else:
            # Poids de base (log) uniquement
            weights = torch.DoubleTensor([
                class_weights.get(idx_to_class.get(lbl, ''), 1.0)
                for lbl in ds.labels
            ])
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                          num_workers=CONFIG['num_workers'],
                          worker_init_fn=_worker_init_fn if CONFIG['num_workers'] > 0 else None,
                          pin_memory=True, drop_last=True), ds
    else:
        return DataLoader(ds, batch_size=batch_size, shuffle=False,
                          num_workers=CONFIG['num_workers'],
                          worker_init_fn=_worker_init_fn if CONFIG['num_workers'] > 0 else None,
                          pin_memory=True), ds


# ============================================================================
# CELL 11 : EMA
# ============================================================================
class ModelEMA:
    def __init__(self, model, decay=0.9999):
        self.ema   = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self.decay = decay

    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.ema[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def state_dict(self):
        return self.ema

    def load_state_dict(self, state_dict):
        """Restaure les poids EMA depuis un checkpoint (éval cohérente avec le meilleur run)."""
        self.ema = {k: v.clone().detach() for k, v in state_dict.items()}

    def apply(self, model):
        model.load_state_dict(self.ema)


ema = ModelEMA(model, decay=CONFIG['ema_decay'])


# ============================================================================
# CELL 12 : OPTIMIZER & SCHEDULER
# ============================================================================
def build_optimizer_and_scheduler(model, lr_backbone, lr_head,
                                   warmup_ep, total_ep):
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(),     'lr': lr_backbone},
        {'params': model.shared_proj.parameters(),  'lr': lr_head},
        {'params': model.head_main.parameters(),    'lr': lr_head},
        {'params': model.head_crop.parameters(),    'lr': lr_head},
        {'params': model.head_category.parameters(),'lr': lr_head},
    ], weight_decay=CONFIG['weight_decay'])

    def lam_bb(ep):
        if ep < warmup_ep:
            # Evite LR=0 à la toute première epoch (ep=0)
            return (ep + 1) / max(1, warmup_ep)
        p = (ep - warmup_ep) / max(1, total_ep - warmup_ep)
        return 0.5 * (1 + math.cos(math.pi * p))

    def lam_h(ep):
        w = min(2, warmup_ep)
        if ep < w:
            # Evite LR=0 à la toute première epoch (ep=0)
            return (ep + 1) / max(1, w)
        p = (ep - w) / max(1, total_ep - w)
        return 0.5 * (1 + math.cos(math.pi * p))

    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(optimizer,
                         lr_lambda=[lam_bb, lam_h, lam_h, lam_h, lam_h])
    return optimizer, scheduler


optimizer, scheduler = build_optimizer_and_scheduler(
    model,
    CONFIG['lr_backbone'], CONFIG['lr_head'],
    CONFIG['warmup_epochs'], CONFIG['num_epochs'],
)
scaler = GradScaler(enabled=(DEVICE.type == 'cuda'))
print("✅ Optimizer + Scheduler")


# ============================================================================
# CELL 13 : MÉTRIQUES AVANCÉES (amélioration 5)
# ============================================================================
def compute_advanced_metrics(preds, labels, class_hierarchy,
                              class_to_idx, idx_to_class):
    """
    Calcule accuracy par :
      - pattern visuel (spot / blight / rot / rust / mildew / virus...)
      - organe végétal (leaf / fruit / stem / root...)
      - crop (tomato / wheat / grape...)

    Retourne un dict de dicts {group_value: {accuracy, n_samples, correct}}
    """
    preds  = np.array(preds)
    labels = np.array(labels)

    results = {
        'per_pattern': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'per_organ':   defaultdict(lambda: {'correct': 0, 'total': 0}),
        'per_crop':    defaultdict(lambda: {'correct': 0, 'total': 0}),
    }

    for p, l in zip(preds, labels):
        cls_name = idx_to_class.get(int(l), '')
        hier     = class_hierarchy.get(cls_name, {})
        pattern  = hier.get('pattern', 'unknown')
        organ    = hier.get('organ',   'unknown')
        crop     = hier.get('crop',    'unknown')
        correct  = int(p == l)

        results['per_pattern'][pattern]['correct'] += correct
        results['per_pattern'][pattern]['total']   += 1
        results['per_organ'][organ]['correct']     += correct
        results['per_organ'][organ]['total']       += 1
        results['per_crop'][crop]['correct']       += correct
        results['per_crop'][crop]['total']         += 1

    # Calculer accuracy
    for group_key, group_data in results.items():
        for val, d in group_data.items():
            d['accuracy'] = round(d['correct'] / max(d['total'], 1), 4)

    return {k: dict(v) for k, v in results.items()}


def topk_acc(logits, targets, k=5):
    with torch.no_grad():
        _, pred = logits.topk(k, dim=1)
        t1 = pred[:, 0].eq(targets).float().mean().item()
        t5 = pred.eq(targets.unsqueeze(1)).any(1).float().mean().item()
    return t1, t5


def forward_main_logits_tta(model, images):
    """
    V5 — TTA : moyenne des logits (orig, flip H, crop centre ~90 %, resize léger).
    Retourne (logits_main_moyennés, sortie complète du 1er forward pour têtes auxiliaires).
    4 forwards au total.
    """
    wcfg = CONFIG.get('tta_weights', {})
    w_orig = float(wcfg.get('orig', 0.25))
    w_flip = float(wcfg.get('flip', 0.25))
    w_crop = float(wcfg.get('crop', 0.25))
    w_scale = float(wcfg.get('scale', 0.25))
    w_sum = max(1e-8, (w_orig + w_flip + w_crop + w_scale))

    outs = []
    o0 = model(images)
    outs.append((w_orig, o0['main']))
    outs.append((w_flip, model(torch.flip(images, dims=[3]))['main']))
    B, C, H, W = images.shape
    ch, cw = max(1, int(H * 0.05)), max(1, int(W * 0.05))
    cropped = images[:, :, ch:H - ch, cw:W - cw]
    cropped = F.interpolate(cropped, size=(H, W), mode='bilinear', align_corners=False)
    outs.append((w_crop, model(cropped)['main']))
    scaled = F.interpolate(images, scale_factor=0.92, mode='bilinear', align_corners=False)
    scaled = F.interpolate(scaled, size=(H, W), mode='bilinear', align_corners=False)
    outs.append((w_scale, model(scaled)['main']))
    main_logits = sum(w * lg for w, lg in outs) / w_sum
    return main_logits, o0


# ============================================================================
# CELL 14 : TRAIN EPOCH
# ============================================================================
def train_epoch(model, loader, optimizer, scaler, device, epoch, finetune_tail=False,
                mix_rare_fraction=0.7, canonical_penalty=0.0,
                start_batch_idx=0, global_step_start=0,
                save_resume_callback=None):
    model.train()
    total_loss = 0
    top1_list  = []
    aug_counts = Counter()
    seen_batches = 0
    global_step = global_step_start

    accum_steps = max(1, int(CONFIG.get('grad_accum_steps', 1)))
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(tqdm(loader, desc=f"Train E{epoch+1}")):
        if batch_idx < start_batch_idx:
            continue
        images, labels, crop_lbl, cat_lbl, canonical_lbl, groups = batch
        images       = images.to(device, non_blocking=True)
        labels       = labels.to(device, non_blocking=True)
        crop_lbl     = torch.tensor(crop_lbl,     dtype=torch.long).to(device)
        cat_lbl      = torch.tensor(cat_lbl,      dtype=torch.long).to(device)
        canonical_lbl= torch.tensor(canonical_lbl,dtype=torch.long).to(device)

        use_mix = mix_rare_fraction > 0 and not finetune_tail
        (images, la, lb, cra, crb, cata, catb,
         cana, canb, lam, strategy) = smart_augment(
            images, labels, crop_lbl, cat_lbl, canonical_lbl, groups,
            CONFIG['mixup_alpha'], CONFIG['cutmix_alpha'],
            rare_mix_prob=mix_rare_fraction if use_mix else 0.0,
            force_identity=(finetune_tail or not use_mix),
        )
        aug_counts[strategy] += 1

        with autocast(enabled=scaler.is_enabled()):
            out = model(images)

            # Loss principale (CleanLabelLoss wrapping ComboLoss)
            l_main = (
                lam * criterion_main(out['main'], la, cana,
                                     penalty_weight=canonical_penalty)
                + (1 - lam) * criterion_main(out['main'], lb, canb,
                                            penalty_weight=canonical_penalty))

            # Pertes auxiliaires
            vc = cra >= 0; vt = cata >= 0
            l_crop = (criterion_aux(out['crop'][vc], cra[vc])
                      if vc.any() else torch.tensor(0., device=device))
            l_cat  = (criterion_aux(out['category'][vt], cata[vt])
                      if vt.any() else torch.tensor(0., device=device))

            loss = (LOSS_WEIGHTS['main']     * l_main
                    + LOSS_WEIGHTS['crop']     * l_crop
                    + LOSS_WEIGHTS['category'] * l_cat)
            loss = loss / accum_steps

        scaler.scale(loss).backward()
        should_step = ((seen_batches + 1) % accum_steps == 0)
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            ema.update(model)

        total_loss += loss.item() * accum_steps
        t1, _ = topk_acc(out['main'].detach(), la)
        top1_list.append(t1)
        seen_batches += 1
        global_step += 1

        if (save_resume_callback is not None
                and CONFIG.get('intra_epoch_resume', True)
                and global_step % CONFIG['resume_save_every_steps'] == 0):
            # En cas de coupure, reprendre au batch suivant.
            save_resume_callback(resume_epoch=epoch,
                                 resume_batch_idx=batch_idx + 1,
                                 global_step=global_step)

    # Flush dernier mini-batch accumulé si besoin.
    if seen_batches > 0 and (seen_batches % accum_steps != 0):
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        ema.update(model)

    mean_loss = (total_loss / max(1, seen_batches))
    mean_top1 = float(np.mean(top1_list)) if top1_list else 0.0
    return mean_loss, mean_top1, dict(aug_counts), global_step


# ============================================================================
# CELL 15 : VALIDATE (métriques avancées)
# ============================================================================
@torch.no_grad()
def validate(model, loader, device, use_ema=True, tta=False,
             canonical_penalty=None):
    orig = None
    if use_ema:
        orig = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema.state_dict())
    model.eval()

    total_loss = 0
    top1_l, top5_l, crop_l, cat_l = [], [], [], []
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="Val" + ("+TTA" if tta else "")):
        images, labels, crop_lbl, cat_lbl, canonical_lbl, groups = batch
        images   = images.to(device)
        labels   = labels.to(device)
        crop_lbl = torch.tensor(crop_lbl, dtype=torch.long).to(device)
        cat_lbl  = torch.tensor(cat_lbl,  dtype=torch.long).to(device)
        canonical_lbl = torch.tensor(canonical_lbl, dtype=torch.long).to(device)

        if tta:
            main_logits, o0 = forward_main_logits_tta(model, images)
            out = {'main': main_logits, 'crop': o0['crop'], 'category': o0['category']}
        else:
            out = model(images)
        w = (CONFIG['canonical_penalty_max'] if canonical_penalty is None
             else canonical_penalty)
        loss = criterion_main(out['main'], labels, canonical_lbl,
                              penalty_weight=w)
        total_loss += loss.item()

        t1, t5 = topk_acc(out['main'], labels)
        top1_l.append(t1); top5_l.append(t5)

        vc = crop_lbl >= 0; vt = cat_lbl >= 0
        if vc.any():
            crop_l.append(out['crop'][vc].argmax(1)
                          .eq(crop_lbl[vc]).float().mean().item())
        if vt.any():
            cat_l.append(out['category'][vt].argmax(1)
                         .eq(cat_lbl[vt]).float().mean().item())

        all_preds.extend(out['main'].argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    if orig:
        model.load_state_dict(orig)

    rare_idx  = set(class_to_idx[c] for c in rare_classes if c in class_to_idx)
    rare_mask = np.array([l in rare_idx for l in all_labels])
    rare_acc  = float(np.mean(np.array(all_preds)[rare_mask]
                              == np.array(all_labels)[rare_mask])) \
                if rare_mask.any() else float('nan')

    # AMÉLIORATION 5 — métriques avancées
    adv = compute_advanced_metrics(
        all_preds, all_labels, class_hierarchy, class_to_idx, idx_to_class)

    return {
        'loss':         total_loss / len(loader),
        'top1':         float(np.mean(top1_l)),
        'top5':         float(np.mean(top5_l)),
        'rare_acc':     rare_acc,
        'crop_acc':     float(np.mean(crop_l))  if crop_l else float('nan'),
        'cat_acc':      float(np.mean(cat_l))   if cat_l  else float('nan'),
        'per_pattern':  adv['per_pattern'],
        'per_organ':    adv['per_organ'],
        'per_crop':     adv['per_crop'],
        'preds':        all_preds,
        'labels':       all_labels,
    }


@torch.no_grad()
def collect_logits_labels(model, loader, device, use_ema=True, tta=True):
    """Collecte logits + labels pour calibration température."""
    orig = None
    if use_ema:
        orig = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema.state_dict())
    model.eval()
    logits_all, labels_all = [], []
    for batch in tqdm(loader, desc="Collect logits"):
        images, labels, *_ = batch
        images = images.to(device)
        labels = torch.as_tensor(labels, dtype=torch.long, device=device)
        if tta:
            lg, _ = forward_main_logits_tta(model, images)
        else:
            lg = model(images)['main']
        logits_all.append(lg.detach().cpu())
        labels_all.append(labels.detach().cpu())
    if orig:
        model.load_state_dict(orig)
    return torch.cat(logits_all, dim=0), torch.cat(labels_all, dim=0)


def fit_temperature(logits_cpu, labels_cpu, max_iter=50):
    """Apprend une température scalaire T qui minimise la NLL sur val."""
    if logits_cpu.numel() == 0:
        return 1.0
    device = DEVICE
    logits = logits_cpu.to(device)
    labels = labels_cpu.to(device)
    log_t = torch.zeros(1, device=device, requires_grad=True)  # T=1 au départ
    nll = nn.CrossEntropyLoss()
    opt = optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        t = torch.exp(log_t).clamp(0.5, 10.0)
        loss = nll(logits / t, labels)
        loss.backward()
        return loss

    opt.step(closure)
    t = float(torch.exp(log_t).clamp(0.5, 10.0).item())
    return t


@torch.no_grad()
def eval_ensemble_mean_logits(checkpoint_paths, val_loader, device):
    """
    Moyenne des logits des poids EMA de plusieurs entraînements (seeds différents).
    Remplir CONFIG['ensemble_checkpoints'] avec les chemins `best_model_s*.pt` après chaque run.
    """
    paths = [Path(p) for p in checkpoint_paths if p]
    paths = [p for p in paths if p.is_file()]
    if len(paths) < 2:
        print("  ℹ️ Ensemble : au moins 2 checkpoints valides dans CONFIG['ensemble_checkpoints'].")
        return None
    logits_sum = None
    labels_ref = None
    n_ok = 0
    used_t = []
    for p in paths:
        ck = torch.load(p, map_location=device)
        if 'ema_state_dict' not in ck:
            print(f"  ⚠️ Pas d'EMA dans {p}")
            continue
        ema.load_state_dict(ck['ema_state_dict'])
        ema.apply(model)
        model.eval()
        batch_logits, batch_labels = [], []
        temp = float(ck.get('temperature', 1.0))
        used_t.append(temp)
        for batch in tqdm(val_loader, desc=f"Ens {p.stem}"):
            images, labels, *_ = batch
            images = images.to(device)
            labels = torch.as_tensor(labels, dtype=torch.long, device=device)
            lg, _ = forward_main_logits_tta(model, images)
            lg = (lg / max(1e-6, temp)).cpu()
            batch_logits.append(lg)
            batch_labels.append(labels.cpu())
        L = torch.cat(batch_logits, dim=0)
        y = torch.cat(batch_labels, dim=0)
        logits_sum = L if logits_sum is None else logits_sum + L
        labels_ref = y
        n_ok += 1
    if n_ok < 2 or logits_sum is None:
        return None
    mean_l = logits_sum / float(n_ok)
    acc = float((mean_l.argmax(1) == labels_ref).float().mean())
    print(f"  🔀 Ensemble moyenne logits+TTA ({n_ok} modèles, T moyen={np.mean(used_t):.3f}) | top-1 ≈ {acc:.4f}")
    return acc


# ============================================================================
# CELL 16 : BOUCLE PRINCIPALE
# ============================================================================
print("\n" + "="*70)
print("🚀 DINOv2 V5 — Unfreeze + Mix/CutMix + Hard mining + dual CleanLabel + CORE replay + TTA")
print(f"🎯 Fine-tune mode: {CONFIG['fine_tune_mode']} | profile: {CONFIG['training_profile']}")
print("="*70)

PHASE_SCHEDULE = [
    (0,  9,  'phase_1', False),
    (10, 39, 'phase_2', True),
    (40, CONFIG['num_epochs'] - 1, 'phase_3', True),
]

def get_phase(ep):
    for s, e, k, sm in PHASE_SCHEDULE:
        if s <= ep <= e:
            return k, sm
    return 'phase_3', True


train_mt = load_json(META_DIR / 'train_multitask.json')
val_mt   = load_json(META_DIR / 'val_multitask.json')

history       = defaultdict(list)
best_top1     = 0.0
patience_ctr  = 0
current_phase = None
train_loader  = val_loader = train_ds = None
unfreeze_done = {'stage1': False, 'stage2': False}
start_epoch   = 0
start_batch_idx = 0
global_step = 0
resume_state_path = CKPT_DIR / CONFIG['resume_checkpoint_name']

# Ajuster la stratégie de fine-tuning dès le départ.
if CONFIG['fine_tune_mode'] == 'feature_extract':
    model._freeze_all_backbone()
    unfreeze_done = {'stage1': True, 'stage2': True}
elif CONFIG['fine_tune_mode'] == 'partial':
    # stage2 (full unfreeze) désactivé en mode partiel
    unfreeze_done['stage2'] = True

if CONFIG.get('auto_resume', True) and resume_state_path.exists():
    rs = torch.load(resume_state_path, map_location=DEVICE)
    model.load_state_dict(rs['model_state_dict'])
    if 'ema_state_dict' in rs:
        ema.load_state_dict(rs['ema_state_dict'])
    if 'optimizer_state_dict' in rs:
        optimizer.load_state_dict(rs['optimizer_state_dict'])
    if 'scheduler_state_dict' in rs:
        scheduler.load_state_dict(rs['scheduler_state_dict'])
    if scaler.is_enabled() and rs.get('scaler_state_dict') is not None:
        scaler.load_state_dict(rs['scaler_state_dict'])

    hist_loaded = rs.get('history', {})
    history = defaultdict(list, {k: list(v) for k, v in hist_loaded.items()})
    best_top1 = float(rs.get('best_top1', 0.0))
    patience_ctr = int(rs.get('patience_ctr', 0))
    global_step = int(rs.get('global_step', 0))
    if 'resume_epoch' in rs:
        start_epoch = int(rs.get('resume_epoch', 0))
        start_batch_idx = int(rs.get('resume_batch_idx', 0))
    else:
        # compat ancien format: reprise à l'epoch suivante
        start_epoch = int(rs.get('epoch', -1)) + 1
        start_batch_idx = 0
    current_phase = rs.get('current_phase', None)
    unfreeze_done = rs.get('unfreeze_done', unfreeze_done)
    print(f"♻️ Reprise auto depuis epoch {start_epoch}, batch {start_batch_idx} | "
          f"best_top1={best_top1:.4f} | global_step={global_step}")

for epoch in range(start_epoch, CONFIG['num_epochs']):
    phase_key, use_sampler = get_phase(epoch)

    # ── AMÉLIORATION 1 : progressive unfreezing ───────────────────────────
    if (CONFIG['fine_tune_mode'] in ('partial', 'full')
            and not unfreeze_done['stage1']
            and epoch >= CONFIG['unfreeze_stage1_epoch']):
        model.unfreeze_last_n_blocks(CONFIG['stage1_unfreeze_blocks'])
        # V5 : stage1 = adaptation délicate du backbone (évite destruction des features)
        optimizer, scheduler = build_optimizer_and_scheduler(
            model,
            CONFIG['lr_backbone'] * CONFIG['stage1_backbone_lr_mult'],
            CONFIG['lr_head'] * CONFIG['stage1_head_lr_mult'],
            CONFIG['warmup_epochs'], CONFIG['num_epochs'],
        )
        unfreeze_done['stage1'] = True

    if (CONFIG['fine_tune_mode'] == 'full'
            and not unfreeze_done['stage2']
            and epoch >= CONFIG['unfreeze_stage2_epoch']):
        model.unfreeze_full()
        optimizer, scheduler = build_optimizer_and_scheduler(
            model,
            CONFIG['lr_backbone'] * CONFIG['lr_backbone_stage2_mult'],
            CONFIG['lr_head'] * CONFIG['stage2_head_lr_mult'],
            CONFIG['warmup_epochs'], CONFIG['num_epochs'],
        )
        unfreeze_done['stage2'] = True

    # ── Changement de phase → reconstruire loaders ───────────────────────
    if phase_key != current_phase or train_loader is None or val_loader is None:
        current_phase = phase_key
        ph = phase_groups[phase_key]
        active_set = set(item['class'] for item in ph['classes'])
        meta_train = build_train_meta_with_core_replay(train_mt, phase_key, active_set)
        lcf = loader_class_filter(phase_key, active_set)
        print(f"\n{'='*60}")
        print(f"📌 PHASE {phase_key.upper()} | {ph['description']}")
        print(f"   {len(active_set)} classes | train rows={len(meta_train):,} | "
              f"Sampler={'✅' if use_sampler else '❌'}")
        print(f"{'='*60}")
        (train_loader, train_ds) = build_loader(
            meta_train, lcf, is_train=True,
            batch_size=CONFIG['batch_size'],
            dataset_ref=train_ds,
            use_hard_miner=(use_sampler and epoch >= CONFIG['hard_mining_start_epoch']),
        )
        (val_loader, _) = build_loader(
            val_mt, None, is_train=False, batch_size=CONFIG['batch_size'])

    # ── AMÉLIORATION 3 : mise à jour hard miner (V5 : chaque epoch si freq=1) ─
    elif (use_sampler
          and epoch >= CONFIG['hard_mining_start_epoch']
          and epoch % CONFIG['hard_mining_update_freq'] == 0
          and len(history['val_top1']) > 0):
        ph = phase_groups[phase_key]
        active_set = set(item['class'] for item in ph['classes'])
        meta_train = build_train_meta_with_core_replay(train_mt, phase_key, active_set)
        lcf = loader_class_filter(phase_key, active_set)
        (train_loader, train_ds) = build_loader(
            meta_train, lcf, is_train=True, batch_size=CONFIG['batch_size'],
            dataset_ref=train_ds, use_hard_miner=True,
        )
        hardest = hard_miner.get_hardest_classes(5)
        print(f"  ⛏️  Hard mining mis à jour | Top-5 classes dures :")
        for cls_name, err in hardest:
            print(f"     {cls_name:<45s} err={err:.3f}")

    tail_start = max(0, CONFIG['num_epochs'] - CONFIG['finetune_tail_epochs'])
    if epoch >= tail_start and train_ds is not None:
        train_ds.light_aug = True
    if tail_start > 0 and epoch == tail_start:
        for g in optimizer.param_groups:
            g['lr'] *= CONFIG['finetune_lr_mult']
        print(f"\n  🎯 Fine-tune final (epochs {tail_start+1}–{CONFIG['num_epochs']}) : "
              f"Mix/CutMix off, aug légère, LR × {CONFIG['finetune_lr_mult']}")

    canon_w = get_canonical_penalty_weight(epoch)
    mix_r   = get_mix_rare_fraction(epoch)

    print(f"\n📌 Epoch {epoch+1}/{CONFIG['num_epochs']} [{phase_key}]  |  "
          f"w_canon={canon_w:.3f}  mix_rare={mix_r:.2f}")
    log_learning_rates(optimizer, epoch)

    def save_resume_state(resume_epoch, resume_batch_idx, global_step):
        torch.save({
            'epoch': epoch,
            'resume_epoch': int(resume_epoch),
            'resume_batch_idx': int(resume_batch_idx),
            'global_step': int(global_step),
            'model_state_dict': model.state_dict(),
            'ema_state_dict': ema.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict() if scaler.is_enabled() else None,
            'history': {k: list(v) for k, v in history.items()},
            'best_top1': best_top1,
            'patience_ctr': patience_ctr,
            'current_phase': current_phase,
            'unfreeze_done': unfreeze_done,
        }, resume_state_path)

    epoch_start_batch_idx = start_batch_idx if epoch == start_epoch else 0
    train_loss, train_top1, aug_counts, global_step = train_epoch(
        model, train_loader, optimizer, scaler, DEVICE, epoch,
        finetune_tail=(epoch >= tail_start),
        mix_rare_fraction=mix_r,
        canonical_penalty=canon_w,
        start_batch_idx=epoch_start_batch_idx,
        global_step_start=global_step,
        save_resume_callback=save_resume_state,
    )
    start_batch_idx = 0
    val_m = validate(model, val_loader, DEVICE, use_ema=True, tta=False,
                     canonical_penalty=canon_w)

    val_tta = None
    if (epoch + 1) % CONFIG['tta_checkpoint_every'] == 0:
        val_tta = validate(model, val_loader, DEVICE, use_ema=True, tta=True,
                           canonical_penalty=canon_w)
        selection_metric = val_tta['top1']
        sel_name = 'TTA'
    else:
        selection_metric = val_m['top1']
        sel_name = 'EMA'

    # Mettre à jour hard miner (métrique rapide, cohérent avec le miner)
    hard_miner.update(np.array(val_m['preds']), np.array(val_m['labels']))

    # Scheduler actif avant le tail; optionnel dans le tail.
    if tail_start > 0 and epoch < tail_start:
        scheduler.step()
    elif CONFIG['tail_scheduler_enabled'] and epoch >= tail_start:
        scheduler.step()

    print(f"  Train | loss={train_loss:.4f}  top1={train_top1:.4f}  "
          f"aug={aug_counts}")
    print(f"  Val   | loss={val_m['loss']:.4f}  top1={val_m['top1']:.4f}  "
          f"top5={val_m['top5']:.4f}  rare={val_m['rare_acc']:.4f}  (EMA, no TTA)")
    if val_tta is not None:
        print(f"  Val† | top1={val_tta['top1']:.4f}  top5={val_tta['top5']:.4f}  "
              f"(EMA+TTA) ← sélection checkpoint")
    print(f"  📊 Sélection cette epoch : {sel_name}  metric={selection_metric:.4f}")

    # Afficher top-3 patterns et organes
    pat_sorted = sorted(val_m['per_pattern'].items(),
                        key=lambda x: x[1]['accuracy'])
    org_sorted = sorted(val_m['per_organ'].items(),
                        key=lambda x: x[1]['accuracy'])
    print(f"  Pattern  worst: {pat_sorted[0][0]}={pat_sorted[0][1]['accuracy']:.3f} "
          f"| best: {pat_sorted[-1][0]}={pat_sorted[-1][1]['accuracy']:.3f}")
    print(f"  Organ    worst: {org_sorted[0][0]}={org_sorted[0][1]['accuracy']:.3f} "
          f"| best: {org_sorted[-1][0]}={org_sorted[-1][1]['accuracy']:.3f}")

    # Logging
    history['train_loss'].append(train_loss)
    history['train_top1'].append(train_top1)
    history['lr_backbone'].append(float(optimizer.param_groups[0]['lr']))
    history['lr_head'].append(float(optimizer.param_groups[1]['lr']))
    history['canonical_w'].append(float(canon_w))
    history['mix_rare_sched'].append(float(mix_r))
    history['val_top1_tta'].append(
        float(val_tta['top1']) if val_tta is not None else float('nan'))
    for k in ['loss', 'top1', 'top5', 'rare_acc', 'crop_acc', 'cat_acc']:
        v = val_m[k]
        history[f'val_{k}'].append(
            float(v) if not (isinstance(v, float) and math.isnan(v)) else 0.0)
    # Accuracy par pattern (logguer pour graphiques)
    for pat, d in val_m['per_pattern'].items():
        history[f'pat_{pat}'].append(d['accuracy'])

    # Checkpoint (métrique d'élection = EMA ou EMA+TTA selon l'epoch)
    if selection_metric > best_top1:
        best_top1    = selection_metric
        patience_ctr = 0
        ckpt_payload = {
            'epoch':            epoch,
            'phase':            phase_key,
            'model_state_dict': model.state_dict(),
            'ema_state_dict':   ema.state_dict(),
            'val_top1':         val_m['top1'],
            'val_top5':         val_m['top5'],
            'selection_metric': selection_metric,
            'selection_name':   sel_name,
            'val_top1_tta':     val_tta['top1'] if val_tta else None,
            'rare_acc':         val_m['rare_acc'],
            'per_pattern':      val_m['per_pattern'],
            'per_organ':        val_m['per_organ'],
            'class_to_idx':     class_to_idx,
            'config':           CONFIG,
            'seed':             CONFIG['seed'],
            'temperature':      1.0,
        }
        torch.save(ckpt_payload, CKPT_DIR / 'best_model.pt')
        torch.save(ckpt_payload, CKPT_DIR / f"best_model_s{CONFIG['seed']}.pt")
        print(f"  ✅ Meilleur modèle ({sel_name} top1={best_top1:.4f})")
    else:
        patience_ctr += 1
        if patience_ctr >= CONFIG['patience']:
            print("  ⏹️  Early stopping")
            break

    if (epoch + 1) % 10 == 0:
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()},
                   CKPT_DIR / f'ckpt_e{epoch+1}.pt')

    # Checkpoint de reprise (sauvé à chaque epoch)
    torch.save({
        'epoch': epoch,
        'resume_epoch': epoch + 1,
        'resume_batch_idx': 0,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'ema_state_dict': ema.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler.is_enabled() else None,
        'history': {k: list(v) for k, v in history.items()},
        'best_top1': best_top1,
        'patience_ctr': patience_ctr,
        'current_phase': current_phase,
        'unfreeze_done': unfreeze_done,
    }, resume_state_path)

print(f"\n✅ Entraînement terminé | meilleure top-1 = {best_top1:.4f}")


# ============================================================================
# CELL 17 : LOGS & GRAPHIQUES AVANCÉS
# ============================================================================
def _json_safe_float(x):
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return float(x)

log_path = LOG_DIR / 'training_logs.json'
with open(log_path, 'w') as f:
    json.dump({k: [_json_safe_float(v) for v in vals]
               for k, vals in history.items()}, f, indent=2)

# Repères de phase
phase_changes = [i for i in range(1, len(history.get('val_top1', [])))
                 if (i < len(history.get('val_top1', [])) and
                     i < len(history.get('val_top1', [])))]

fig = plt.figure(figsize=(20, 14))
fig.suptitle('DINOv2 V5 — Unfreeze + Mix/CutMix + Hard mining + dual CleanLabel + CORE replay',
             fontsize=13, weight='bold')

# Layout : 3 lignes × 3 colonnes
axes = fig.subplots(3, 3)

def vlines(ax):
    for ep in [CONFIG['unfreeze_stage1_epoch'], CONFIG['unfreeze_stage2_epoch']]:
        ax.axvline(ep, color='navy', linestyle=':', alpha=0.6, linewidth=1.2)

# Row 0
axes[0,0].plot(history['train_loss'], color='steelblue', label='Train')
axes[0,0].plot(history['val_loss'],   color='tomato',    label='Val')
axes[0,0].set_title('CleanLabel ComboLoss'); axes[0,0].legend(); axes[0,0].grid(alpha=.3)
vlines(axes[0,0])

axes[0,1].plot(history['train_top1'], color='steelblue', label='Train')
axes[0,1].plot(history['val_top1'],   color='tomato',    label='Val')
axes[0,1].set_title('Top-1 Accuracy'); axes[0,1].legend(); axes[0,1].grid(alpha=.3)
vlines(axes[0,1])

axes[0,2].plot(history['val_top5'], color='mediumseagreen')
axes[0,2].set_title('Val Top-5'); axes[0,2].grid(alpha=.3)
vlines(axes[0,2])

# Row 1 — métriques avancées
axes[1,0].plot(history['val_rare_acc'], color='darkorange')
axes[1,0].set_title('RARE Classes Accuracy'); axes[1,0].grid(alpha=.3)
vlines(axes[1,0])

# Per-pattern (courbes colorées)
pattern_colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3',
                  '#ff7f00','#a65628','#f781bf','#999999',
                  '#66c2a5','#fc8d62','#8da0cb','#e78ac3']
for i, (pat, col) in enumerate(zip(
        sorted(k[4:] for k in history if k.startswith('pat_')),
        pattern_colors)):
    key = f'pat_{pat}'
    if key in history and history[key]:
        axes[1,1].plot(history[key], label=pat, color=col, linewidth=1)
axes[1,1].set_title('Per-Pattern Accuracy'); axes[1,1].legend(fontsize=6)
axes[1,1].grid(alpha=.3)
vlines(axes[1,1])

axes[1,2].plot(history['val_crop_acc'], label='Crop',     color='purple')
axes[1,2].plot(history['val_cat_acc'],  label='Category', color='goldenrod')
axes[1,2].set_title('Têtes auxiliaires'); axes[1,2].legend(); axes[1,2].grid(alpha=.3)
vlines(axes[1,2])

# Row 2 — analyse finale (EMA du meilleur checkpoint + TTA)
ckpt = torch.load(CKPT_DIR / 'best_model.pt', map_location=DEVICE)
if 'ema_state_dict' in ckpt:
    ema.load_state_dict(ckpt['ema_state_dict'])
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
temperature = float(ckpt.get('temperature', 1.0))
if CONFIG.get('temperature_scaling', True):
    logits_cpu, labels_cpu = collect_logits_labels(
        model, val_loader, DEVICE, use_ema=True, tta=True)
    temperature = fit_temperature(logits_cpu, labels_cpu, max_iter=50)
    ckpt['temperature'] = temperature
    torch.save(ckpt, CKPT_DIR / 'best_model.pt')
    print(f"🌡️ Temperature scaling calibrée: T={temperature:.4f}")
final = validate(model, val_loader, DEVICE, use_ema=True, tta=True,
                 canonical_penalty=CONFIG['canonical_penalty_max'])
if temperature != 1.0:
    all_logits, all_labels = collect_logits_labels(
        model, val_loader, DEVICE, use_ema=True, tta=True)
    logits_t = (all_logits / temperature)
    preds_t = logits_t.argmax(1).numpy()
    labels_t = all_labels.numpy()
    final['top1'] = float((preds_t == labels_t).mean())
    top5_idx = torch.topk(logits_t, k=min(5, logits_t.shape[1]), dim=1).indices
    hit5 = top5_idx.eq(all_labels.unsqueeze(1)).any(1).float().mean().item()
    final['top5'] = float(hit5)

# Per-organ bar chart
org_names = list(final['per_organ'].keys())
org_acc   = [final['per_organ'][o]['accuracy'] for o in org_names]
org_n     = [final['per_organ'][o]['total']    for o in org_names]
axes[2,0].barh(org_names, org_acc, color='steelblue', alpha=0.8)
axes[2,0].set_xlim(0, 1.0)
axes[2,0].set_title('Accuracy par Organe (final)')
axes[2,0].grid(axis='x', alpha=.3)
for i, (a, n) in enumerate(zip(org_acc, org_n)):
    axes[2,0].text(a + 0.01, i, f'{a:.3f} (n={n})', va='center', fontsize=7)

# Per-pattern bar chart
pat_names = list(final['per_pattern'].keys())
pat_acc   = [final['per_pattern'][p]['accuracy'] for p in pat_names]
pat_n     = [final['per_pattern'][p]['total']    for p in pat_names]
bar_colors = ['#d73027' if a < 0.7 else '#fdae61' if a < 0.85 else '#1a9850'
              for a in pat_acc]
axes[2,1].barh(pat_names, pat_acc, color=bar_colors, alpha=0.85)
axes[2,1].set_xlim(0, 1.0)
axes[2,1].set_title('Accuracy par Pattern visuel (final)')
axes[2,1].grid(axis='x', alpha=.3)
for i, (a, n) in enumerate(zip(pat_acc, pat_n)):
    axes[2,1].text(a + 0.01, i, f'{a:.3f} (n={n})', va='center', fontsize=7)

# Résumé + hard mining
axes[2,2].axis('off')
axes[2,2].text(.02, .96, '✅ DINOv2 V5 (EMA+TTA)', fontsize=11, weight='bold')
axes[2,2].text(.02, .84, f'Top-1    : {final["top1"]:.4f}', fontsize=10)
axes[2,2].text(.02, .74, f'Top-5    : {final["top5"]:.4f}', fontsize=10)
axes[2,2].text(.02, .64, f'RARE acc : {final["rare_acc"]:.4f}', fontsize=10)
axes[2,2].text(.02, .52, 'Améliorations actives :', fontsize=9, weight='bold')
axes[2,2].text(.02, .43, '✅ Progressive unfreeze (stage1/2)', fontsize=8)
axes[2,2].text(.02, .34, '✅ Mix/CutMix 70/30 + fine-tune tail', fontsize=8)
axes[2,2].text(.02, .25, '✅ Hard mining (EMA, freq=1)', fontsize=8)
axes[2,2].text(.02, .16, '✅ CleanLabel dual batch-wide', fontsize=8)
axes[2,2].text(.02, .07, '✅ CORE replay ph2/3 + per-group metrics', fontsize=8)
hardest = hard_miner.get_hardest_classes(3)
axes[2,2].text(.02, -.02, 'Classes les + dures :', fontsize=8, weight='bold')
for j, (cls_name, err) in enumerate(hardest):
    axes[2,2].text(.02, -.10 - j * 0.09,
                   f'  {cls_name[:30]} ({err:.2f})', fontsize=7)

plt.tight_layout()
plt.savefig(LOG_DIR / 'training_metrics_v5.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"✅ Graphiques : {LOG_DIR / 'training_metrics_v5.png'}")


# ============================================================================
# CELL 18 : ÉVALUATION FINALE COMPLÈTE
# ============================================================================
print(f"\n📊 RÉSULTATS FINAUX (évaluation EMA + TTA)")
print(f"   Top-1    : {final['top1']:.4f}")
print(f"   Top-5    : {final['top5']:.4f}")
print(f"   RARE acc : {final['rare_acc']:.4f}")

print("\n   Per-pattern accuracy :")
for pat, d in sorted(final['per_pattern'].items(), key=lambda x: x[1]['accuracy']):
    bar = '█' * int(d['accuracy'] * 20)
    print(f"     {pat:<12s} {bar:<20s} {d['accuracy']:.4f}  (n={d['total']})")

print("\n   Per-organ accuracy :")
for org, d in sorted(final['per_organ'].items(), key=lambda x: x[1]['accuracy']):
    bar = '█' * int(d['accuracy'] * 20)
    print(f"     {org:<10s} {bar:<20s} {d['accuracy']:.4f}  (n={d['total']})")

# Top 5 crops les plus difficiles
print("\n   5 crops les plus difficiles :")
for crop, d in sorted(final['per_crop'].items(),
                      key=lambda x: x[1]['accuracy'])[:5]:
    print(f"     {crop:<20s} {d['accuracy']:.4f}  (n={d['total']})")

# Per-class + sauvegarde
all_p = np.array(final['preds']); all_l = np.array(final['labels'])
per_cls = {}
for i in range(NUM_CLASSES):
    mask = all_l == i
    if not mask.any():
        continue
    cls_name = idx_to_class[i]
    hier     = class_hierarchy.get(cls_name, {})
    per_cls[cls_name] = {
        'accuracy': round(float((all_p[mask] == i).mean()), 4),
        'n_val':    int(mask.sum()),
        'level':    class_report.get(cls_name, {}).get('level', '?'),
        'organ':    hier.get('organ',   '?'),
        'pattern':  hier.get('pattern', '?'),
        'crop':     hier.get('crop',    '?'),
    }

with open(LOG_DIR / 'per_class_accuracy.json', 'w') as f:
    json.dump(dict(sorted(per_cls.items(), key=lambda x: x[1]['accuracy'])),
              f, indent=2)

# Sauvegarder métriques avancées séparément
with open(LOG_DIR / 'advanced_metrics.json', 'w') as f:
    json.dump({
        'per_pattern': final['per_pattern'],
        'per_organ':   final['per_organ'],
        'per_crop':    {k: v for k, v in sorted(
            final['per_crop'].items(), key=lambda x: x[1]['accuracy'])[:30]},
    }, f, indent=2)

# Matrice de confusion top 30
top30   = np.argsort(np.bincount(all_l, minlength=NUM_CLASSES))[-30:][::-1]
mask30  = np.isin(all_l, top30)
remap   = {orig: new for new, orig in enumerate(top30)}
sl = all_l[mask30]; sp = all_p[mask30]
sl_r = np.array([remap[x] for x in sl])
sp_r = np.array([remap.get(x, -1) for x in sp])
v = sp_r >= 0
cm = confusion_matrix(sl_r[v], sp_r[v], labels=list(range(30)))
names30 = [idx_to_class[i][:22] for i in top30]
fig, ax = plt.subplots(figsize=(22, 18))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=names30, yticklabels=names30,
            ax=ax, linewidths=.3, cbar_kws={'shrink': .5})
ax.set_title('Matrice de confusion — Top 30 classes', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=7)
plt.yticks(fontsize=7)
plt.tight_layout()
plt.savefig(LOG_DIR / 'confusion_matrix_top30.png', dpi=150, bbox_inches='tight')
plt.show()

# Export final
torch.save({
    'model_state_dict': model.state_dict(),
    'ema_state_dict':   ema.state_dict(),
    'class_to_idx':     class_to_idx,
    'idx_to_class':     idx_to_class,
    'config':           CONFIG,
    'val_metrics': {k: v for k, v in final.items()
                    if k not in ('preds', 'labels',
                                 'per_pattern', 'per_organ', 'per_crop')},
    'per_pattern': final['per_pattern'],
    'per_organ':   final['per_organ'],
    'timestamp':   datetime.now().isoformat(),
}, OUT_DIR / 'model_dinov2_v5.pth')

if CONFIG.get('ensemble_checkpoints') and len(CONFIG['ensemble_checkpoints']) >= 2:
    eval_ensemble_mean_logits(CONFIG['ensemble_checkpoints'], val_loader, DEVICE)

print(f"""
{'='*70}
✅ PIPELINE V5 TERMINÉ
{'='*70}
📊 Top-1={final['top1']:.4f} | Top-5={final['top5']:.4f} | RARE={final['rare_acc']:.4f}
   (métriques finales : poids EMA + TTA)

🔧 V5 — récapitulatif :
  ✅ 1. Unfreeze LR           stage1 bb×0.1 / head×0.5 → stage2 bb×{CONFIG['lr_backbone_stage2_mult']} / head×0.4
  ✅ 2. Mix/CutMix            décroissant 0.7→0.4→0.2 puis tail ; w_canon rampe 0→{CONFIG['canonical_penalty_max']}
  ✅ 3. Hard mining + sélection checkpoint tous les {CONFIG['tta_checkpoint_every']} ep (TTA) sinon EMA rapide
  ✅ 4. CleanLabel + label_smoothing={CONFIG['label_smoothing']}
  ✅ 5. CORE replay           phase_2 & phase_3 (~{int(CONFIG['core_replay_fraction']*100)} %)
  ✅ 6. Fine-tune tail        {CONFIG['finetune_tail_epochs']} derniers epochs | LR×{CONFIG['finetune_lr_mult']}
  ✅ 7. Logs LR / global_epoch | seed={CONFIG['seed']} | ensemble optionnel (CONFIG['ensemble_checkpoints'])
  💡 ViT-L : backbone dinov2_vitl14 + total_blocks=24 si VRAM OK

📁 Outputs :
  {OUT_DIR}/model_dinov2_v5.pth
  {CKPT_DIR}/best_model.pt
  {CKPT_DIR}/best_model_s{CONFIG['seed']}.pt
  {LOG_DIR}/per_class_accuracy.json
  {LOG_DIR}/advanced_metrics.json
  {LOG_DIR}/training_metrics_v5.png
  {LOG_DIR}/confusion_matrix_top30.png
""")
