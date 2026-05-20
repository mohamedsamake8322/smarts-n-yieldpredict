# ============================================================================
# 🌿 DINOv2 — EXTRACTION OFFLINE + CLASSIFIEUR LÉGER (AMÉLIORÉ)
# ============================================================================
"""
Principe : DINOv2 ne s'entraîne PAS. On l'utilise comme extracteur de features.

PHASE 1 — Extraction (une seule fois, ~20 min sur T4)
  → DINOv2 fait un forward pass sur toutes les images
  → Les vecteurs (embeddings) sont sauvegardés sur Drive
  → Ne coûte rien à répéter : on recharge juste les fichiers .npy

PHASE 2 — Classifieur (très rapide, CPU ou petite GPU)
  → On entraîne un MLP puissant sur les vecteurs sauvegardés
  → Pas de backprop dans DINOv2 → 10x moins de VRAM, 20x plus rapide
  → Epochs de quelques secondes seulement

Pourquoi ça marche si bien :
  DINOv2 a appris à reconnaître textures, formes, patterns sur 142M d'images
  → les features qu'il produit sont déjà riches pour les maladies de plantes
  → le MLP n'a qu'à apprendre la séparation entre classes

PRÉ-REQUIS :
  ✅ Plantdataset_metadata/class_report.json
  ✅ Plantdataset_metadata/class_groups.json
  ✅ Plantdataset_metadata/train.json   [{"path": "...", "class": "..."}, ...]
  ✅ Plantdataset_metadata/val.json

AMÉLIORATIONS AJOUTÉES :
  ✅ Checkpointing périodique toutes les N étapes
  ✅ Reprise d'entraînement au batch précis
  ✅ Sauvegarde complète : model, EMA, optimizer, scheduler, scaler, history
  ✅ Robustesse pour datasets complexes
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
for pkg in ["torch torchvision", "opencv-python",
            "albumentations", "scikit-learn", "tqdm", "seaborn"]:
    subprocess.run(f"pip install -q {pkg}", shell=True)
print("✅ Dépendances installées")

# ============================================================================
# CELL 3 : IMPORTS & CONFIG
# ============================================================================
import os, json, math, random, warnings, time
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast

import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import albumentations as A
from albumentations.pytorch import ToTensorV2
warnings.filterwarnings('ignore')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"   GPU  : {torch.cuda.get_device_name(0)}")
    print(f"   VRAM : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    torch.backends.cudnn.benchmark = True

# ── Chemins ───────────────────────────────────────────────────────────────
META_DIR  = Path('/content/drive/MyDrive/Plantdataset_metadata')
FEAT_DIR  = Path('/content/drive/MyDrive/dinov2_features')   # embeddings cachés
OUT_DIR   = Path('/content/drive/MyDrive/models_dinov2_hybrid')
CKPT_DIR  = Path('/content/drive/MyDrive/checkpoints_hybrid')
LOG_DIR   = Path('/content/drive/MyDrive/logs_hybrid')
for d in [FEAT_DIR, OUT_DIR, CKPT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ===========================================================================
# ⚙️  CONFIGURATION — SEUL ENDROIT À MODIFIER
# ===========================================================================
CONFIG = {
    # ── Backbone DINOv2 ────────────────────────────────────────────────────
    # 'dinov2_vits14' → 384 dims, très rapide,  ~1 GB VRAM
    # 'dinov2_vitb14' → 768 dims, recommandé,   ~2 GB VRAM   ← BON ÉQUILIBRE
    # 'dinov2_vitl14' → 1024 dims, plus riche,  ~4 GB VRAM
    'backbone':       'dinov2_vitb14',
    'embed_dim':      768,          # vits14=384 / vitb14=768 / vitl14=1024
    'image_size':     224,          # 224 suffit, économise VRAM extraction

    # ── Extraction ────────────────────────────────────────────────────────
    'extract_batch':  128,          # grand batch OK (pas de gradient)
    'num_workers':    4,
    'force_extract':  False,        # True = réextraire même si fichiers existent

    # ── Classifieur MLP ──────────────────────────────────────────────────
    'hidden_dims':    [1024, 512, 256],  # couches cachées
    'dropout':        0.35,
    'batch_norm':     True,             # stabilise l'entraînement

    # ── Training classifieur ─────────────────────────────────────────────
    'num_epochs':     150,          # rapide → on peut aller loin
    'batch_size':     512,          # très grand OK (données en RAM)
    'lr':             3e-4,
    'weight_decay':   1e-4,
    'warmup_epochs':  5,
    'label_smoothing':0.1,
    'focal_gamma':    2.0,
    'mixup_alpha':    0.3,
    'patience':       15,           # plus patient car epochs rapides

    # ── Checkpointing & Resume ───────────────────────────────────────────
    'resume_save_every_steps': 500,  # sauvegarde toutes les N étapes
    'resume': False,                 # True pour reprendre depuis checkpoint
    'resume_epoch': 0,               # epoch de reprise (auto-détecté)
    'resume_batch_idx': 0,           # batch de reprise (auto-détecté)
    'global_step': 0,                # étape globale (auto-détecté)

    # ── TTA & Calibration ───────────────────────────────────────────────
    'tta_every_epochs': 20,          # appliquer TTA toutes les N epochs (réduit)
    'calibrate_epochs': 50,          # calibrer le modèle toutes les N epochs (réduit)
    'temperature_init': 1.0,         # température initiale pour calibration

    # ── Optimisations de coût ───────────────────────────────────────────
    'val_every_epochs': 2,           # validation toutes les N epochs (au lieu de chaque)
    'tta_views': 2,                  # nombre de vues TTA (réduit de 3 à 2)
    'grad_accum_steps': 4,           # accumulation de gradients (simule batchs plus grands)
    'compile_model': True,           # utiliser torch.compile si disponible
    'val_batch_size_multiplier': 2,  # batch size validation (plus grand pour vitesse)

    'seed': 42,
}
# ===========================================================================

random.seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
torch.manual_seed(CONFIG['seed'])

def load_json(p):
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)

print(f"\n⚙️  Backbone : {CONFIG['backbone']}  ({CONFIG['embed_dim']}D)")
print(f"   MLP : {CONFIG['hidden_dims']}  |  epochs : {CONFIG['num_epochs']}")
print(f"   Checkpointing : every {CONFIG['resume_save_every_steps']} steps")
print(f"   Optimisations : grad_accum={CONFIG['grad_accum_steps']}x, val_every={CONFIG['val_every_epochs']}ep, compile={CONFIG['compile_model']}")
print(f"   TTA : every {CONFIG['tta_every_epochs']}ep ({CONFIG['tta_views']} vues), Calibration : every {CONFIG['calibrate_epochs']}ep")


# ============================================================================
# CELL 4 : MÉTADONNÉES
# ============================================================================
print("\n📂 Chargement des métadonnées...")

class_report  = load_json(META_DIR / 'class_report.json')
class_groups  = load_json(META_DIR / 'class_groups.json')
train_raw     = load_json(META_DIR / 'train.json')
val_raw       = load_json(META_DIR / 'val.json')

all_classes  = sorted(class_report.keys())
class_to_idx = {cls: i for i, cls in enumerate(all_classes)}
idx_to_class = {i: cls for cls, i in class_to_idx.items()}
NUM_CLASSES  = len(class_to_idx)
CONFIG['num_classes'] = NUM_CLASSES

core_classes     = set(class_groups.get('CORE',     []))
extended_classes = set(class_groups.get('EXTENDED', []))
rare_classes     = set(class_groups.get('RARE',     []))

print(f"✅ {NUM_CLASSES} classes | "
      f"CORE={len(core_classes)} EXTENDED={len(extended_classes)} RARE={len(rare_classes)}")
print(f"   Train={len(train_raw):,} | Val={len(val_raw):,}")

# Class weights log-based
train_counts = Counter(item['class'] for item in train_raw)
total_train  = sum(train_counts.values())
W_MIN, W_MAX = 0.5, 8.0
raw_w = {
    cls: max(W_MIN, min(W_MAX, math.log(total_train / max(c, 1))))
    for cls, c in train_counts.items()
}
mean_w = sum(raw_w.values()) / len(raw_w)
class_weights = {k: round(v / mean_w, 6) for k, v in raw_w.items()}

w_tensor = torch.ones(NUM_CLASSES)
for cls, w in class_weights.items():
    if cls in class_to_idx:
        w_tensor[class_to_idx[cls]] = float(w)
w_tensor = w_tensor.to(DEVICE)


# ============================================================================
# CELL 5 : DATASET POUR L'EXTRACTION (pas d'augmentation — features stables)
# ============================================================================
IMG  = CONFIG['image_size']
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# Augmentation TTA (Test-Time Augmentation) légère pour enrichir les features
# On extrait 3 vues de chaque image et on moyenne les embeddings → +2-3% acc
tta_transforms = [
    # Vue 1 : resize standard
    A.Compose([A.Resize(IMG, IMG),
               A.Normalize(mean=MEAN, std=STD), ToTensorV2()]),
    # Vue 2 : flip horizontal
    A.Compose([A.Resize(IMG, IMG), A.HorizontalFlip(p=1.0),
               A.Normalize(mean=MEAN, std=STD), ToTensorV2()]),
    # Vue 3 : crop légèrement zoomé
    A.Compose([A.RandomResizedCrop(size=(IMG, IMG), scale=(0.85, 1.0)),
               A.Normalize(mean=MEAN, std=STD), ToTensorV2()]),
]
# Vue unique pour validation (stable, reproductible)
val_transform = A.Compose([
    A.Resize(IMG, IMG), A.Normalize(mean=MEAN, std=STD), ToTensorV2()])


class ExtractionDataset(Dataset):
    """Dataset minimal pour l'extraction — lit les images, applique un transform."""
    def __init__(self, items, transform):
        self.items     = [(it['path'], class_to_idx.get(it['class'], -1))
                          for it in items if it['class'] in class_to_idx]
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = cv2.imread(path)
        img = (np.zeros((IMG, IMG, 3), dtype=np.uint8)
               if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return self.transform(image=img)['image'], label, path


# ============================================================================
# CELL 6 : CHARGEMENT DINOv2 (backbone gelé)
# ============================================================================
print(f"\n📥 Chargement {CONFIG['backbone']} (backbone gelé)...")

backbone = torch.hub.load(
    'facebookresearch/dinov2',
    CONFIG['backbone'],
    pretrained=True,
)
if hasattr(backbone, 'head'):
    backbone.head = nn.Identity()

# Geler tout le backbone — on ne l'entraîne JAMAIS
for p in backbone.parameters():
    p.requires_grad = False
backbone = backbone.to(DEVICE).eval()

total_bb = sum(p.numel() for p in backbone.parameters())
print(f"✅ Backbone chargé et gelé ({total_bb/1e6:.1f}M params — 0 entraînable)")


# ============================================================================
# CELL 7 : EXTRACTION DES FEATURES (PHASE 1)
# ============================================================================
def extract_features(items, split_name, use_tta=False):
    """
    Extrait les features DINOv2 pour un split.
    Si use_tta=True, extrait 3 vues et moyenne les embeddings.
    Sauvegarde : features_{split}.npy  +  labels_{split}.npy
    """
    feat_path  = FEAT_DIR / f'features_{split_name}.npy'
    label_path = FEAT_DIR / f'labels_{split_name}.npy'
    path_file  = FEAT_DIR / f'paths_{split_name}.json'

    # Si déjà extrait et force_extract=False → recharger
    if feat_path.exists() and label_path.exists() and not CONFIG['force_extract']:
        print(f"  ♻️  {split_name} : chargement depuis le cache...")
        feats  = np.load(feat_path)
        labels = np.load(label_path)
        print(f"     {feats.shape[0]:,} vecteurs {feats.shape[1]}D chargés")
        return feats, labels

    # Extraction
    transforms_to_use = tta_transforms if use_tta else [val_transform]
    n_views = len(transforms_to_use)
    all_feats  = []
    all_labels = []
    all_paths  = []

    print(f"  🔄 Extraction {split_name} ({n_views} vue(s) par image)...")
    t0 = time.time()

    for view_idx, transform in enumerate(transforms_to_use):
        ds     = ExtractionDataset(items, transform)
        loader = DataLoader(ds, batch_size=CONFIG['extract_batch'],
                            shuffle=False, num_workers=CONFIG['num_workers'],
                            pin_memory=True)
        view_feats = []

        with torch.no_grad():
            for images, labels, paths in tqdm(loader,
                                               desc=f"    Vue {view_idx+1}/{n_views}"):
                images = images.to(DEVICE, non_blocking=True)
                feat   = backbone(images)
                if feat.dim() == 3:
                    feat = feat[:, 0]         # CLS token
                feat = F.normalize(feat, p=2, dim=1)   # L2-normalize
                view_feats.append(feat.cpu().numpy())
                if view_idx == 0:
                    all_labels.extend(labels.numpy())
                    all_paths.extend(paths)

        all_feats.append(np.vstack(view_feats))

    # Moyenne des vues TTA
    feats  = np.mean(all_feats, axis=0).astype(np.float32)
    labels = np.array(all_labels, dtype=np.int32)

    # Re-normaliser après la moyenne
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    feats = feats / np.maximum(norms, 1e-12)

    # Sauvegarder
    np.save(feat_path, feats)
    np.save(label_path, labels)
    with open(path_file, 'w') as f:
        json.dump(all_paths, f)

    elapsed = time.time() - t0
    print(f"     ✅ {feats.shape[0]:,} vecteurs {feats.shape[1]}D "
          f"extraits en {elapsed/60:.1f} min → {feat_path.name}")
    return feats, labels


print("\n" + "="*60)
print("🔬 PHASE 1 : EXTRACTION DES FEATURES DINOv2")
print("="*60)
print("  (cette phase ne se fait qu'une seule fois)")

# Train : 3 vues TTA pour enrichir les features d'entraînement
train_feats, train_labels = extract_features(train_raw, 'train', use_tta=True)
# Val : 1 vue stable pour évaluation reproductible
val_feats,   val_labels   = extract_features(val_raw,   'val',   use_tta=False)

print(f"\n✅ Extraction terminée")
print(f"   Train : {train_feats.shape}  Val : {val_feats.shape}")

# Libérer le backbone de la VRAM — plus besoin jusqu'à l'inférence
del backbone
torch.cuda.empty_cache()
print("   🗑️  Backbone retiré de la VRAM")


# ============================================================================
# CELL 8 : TENSORDATASET EN RAM (ultra-rapide pour l'entraînement)
# ============================================================================
print("\n📦 Chargement des features en RAM...")

# Convertir en tenseurs Float32
X_train = torch.from_numpy(train_feats).float()
y_train = torch.from_numpy(train_labels).long()
X_val   = torch.from_numpy(val_feats).float()
y_val   = torch.from_numpy(val_labels).long()

print(f"   RAM utilisée : ~{(X_train.nbytes + X_val.nbytes)/1e6:.0f} MB")

# WeightedSampler basé sur class_weights_log
sample_w = torch.DoubleTensor([
    class_weights.get(idx_to_class.get(int(lbl), ''), 1.0)
    for lbl in y_train
])
sampler = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=CONFIG['batch_size'],
    sampler=sampler,
    num_workers=0,       # données déjà en RAM → pas de workers nécessaires
    pin_memory=False,
)
val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=CONFIG['batch_size'] * CONFIG['val_batch_size_multiplier'],
    shuffle=False, num_workers=0,
)
print("✅ DataLoaders features prêts (epochs instantanées)")


# ============================================================================
# CELL 9 : CLASSIFIEUR MLP PUISSANT
# ============================================================================
class PowerfulMLP(nn.Module):
    """
    MLP profond entraîné sur les features DINOv2.

    Architecture : BatchNorm → [FC → BN → GELU → Dropout] × N → FC(classes)

    Pourquoi si profond :
      Les features DINOv2 sont génériques (entraîné sur ImageNet-style)
      Le MLP doit apprendre la transformation vers ton espace de maladies
      de plantes → il faut assez de capacité.
    """
    def __init__(self, in_dim, hidden_dims, num_classes,
                 dropout=0.35, use_bn=True):
        super().__init__()
        layers = []

        # Normalisation d'entrée
        layers.append(nn.BatchNorm1d(in_dim))

        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev = h

        # Couche de sortie
        layers.append(nn.Linear(prev, num_classes))

        self.net = nn.Sequential(*layers)

        # Initialisation
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        total = sum(p.numel() for p in self.parameters())
        print(f"  ✅ MLP : {in_dim}D → {' → '.join(map(str, hidden_dims))} "
              f"→ {num_classes}  ({total/1e6:.2f}M params)")

    def forward(self, x):
        return self.net(x)


print("\n🏗️  Construction du classifieur MLP...")
classifier = PowerfulMLP(
    in_dim      = CONFIG['embed_dim'],
    hidden_dims = CONFIG['hidden_dims'],
    num_classes = NUM_CLASSES,
    dropout     = CONFIG['dropout'],
    use_bn      = CONFIG['batch_norm'],
).to(DEVICE)

# Compiler le modèle pour accélérer (si PyTorch 2.0+)
if CONFIG['compile_model'] and hasattr(torch, 'compile'):
    print("⚡ Compilation du modèle avec torch.compile...")
    classifier = torch.compile(classifier)
else:
    print("⚠️  torch.compile non disponible, modèle non compilé")


# ============================================================================
# CELL 10 : LOSS — Combo FocalLoss + LabelSmoothing
# ============================================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma; self.weight = weight

    def forward(self, logits, targets):
        log_p = F.log_softmax(logits, dim=1)
        p_t   = log_p.exp().gather(1, targets.unsqueeze(1)).squeeze(1)
        focal = (1.0 - p_t) ** self.gamma
        ce    = F.nll_loss(log_p, targets, weight=self.weight, reduction='none')
        return (focal * ce).mean()


class LabelSmoothCE(nn.Module):
    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing; self.weight = weight

    def forward(self, logits, targets):
        n = logits.size(1)
        log_p = F.log_softmax(logits, dim=1)
        with torch.no_grad():
            s = torch.full_like(logits, self.smoothing / n)
            s.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        loss = -(s * log_p)
        if self.weight is not None:
            loss = loss * self.weight.unsqueeze(0)
        return loss.sum(1).mean()


class ComboLoss(nn.Module):
    def __init__(self, gamma, weight, smoothing, alpha=0.6):
        super().__init__()
        self.focal  = FocalLoss(gamma=gamma, weight=weight)
        self.smooth = LabelSmoothCE(smoothing=smoothing, weight=weight)
        self.alpha  = alpha

    def forward(self, logits, targets):
        return (self.alpha * self.focal(logits, targets)
              + (1 - self.alpha) * self.smooth(logits, targets))


criterion = ComboLoss(
    gamma     = CONFIG['focal_gamma'],
    weight    = w_tensor,
    smoothing = CONFIG['label_smoothing'],
).to(DEVICE)

print(f"✅ ComboLoss 0.6×Focal + 0.4×LabelSmooth | class_weights log-based")


# ============================================================================
# CELL 11 : MIXUP SUR LES FEATURES
# ============================================================================
def feature_mixup(feats, labels, alpha):
    """MixUp directement sur les vecteurs de features (plus stable que sur images)."""
    if alpha <= 0:
        return feats, labels, labels, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(feats.size(0), device=feats.device)
    return lam * feats + (1 - lam) * feats[idx], labels, labels[idx], lam


# ============================================================================
# CELL 12 : EMA
# ============================================================================
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema   = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self.decay = decay

    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.ema[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def state_dict(self):
        return self.ema

    def apply_to(self, model):
        model.load_state_dict(self.ema)


ema = ModelEMA(classifier)

# ============================================================================
# CELL 13 : OPTIMIZER & SCHEDULER
# ============================================================================
optimizer = optim.AdamW(classifier.parameters(),
                        lr=CONFIG['lr'],
                        weight_decay=CONFIG['weight_decay'])

def build_scheduler(optimizer, warmup, total):
    from torch.optim.lr_scheduler import LambdaLR
    def lam(ep):
        if ep < warmup:
            return ep / max(1, warmup)
        p = (ep - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))
    return LambdaLR(optimizer, lr_lambda=lam)

scheduler = build_scheduler(optimizer, CONFIG['warmup_epochs'], CONFIG['num_epochs'])
scaler    = GradScaler(enabled=(DEVICE.type == 'cuda'))
print(f"✅ AdamW lr={CONFIG['lr']:.1e} | cosine warmup {CONFIG['warmup_epochs']} ep")


# ============================================================================
# CELL 14 : CHECKPOINTING FUNCTIONS
# ============================================================================
def save_checkpoint(epoch, batch_idx, global_step, model, ema, optimizer, scheduler, scaler, history, path):
    """Sauvegarde complète du checkpoint pour reprise."""
    torch.save({
        'epoch': epoch,
        'batch_idx': batch_idx,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'ema_state_dict': ema.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'history': history,
        'class_to_idx': class_to_idx,
        'idx_to_class': idx_to_class,
        'config': CONFIG,
        'timestamp': datetime.now().isoformat(),
    }, path)
    print(f"💾 Checkpoint sauvegardé : {path}")


def load_checkpoint(path, model, ema, optimizer, scheduler, scaler):
    """Charge un checkpoint pour reprise."""
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    ema.ema = ckpt['ema_state_dict']
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    scaler.load_state_dict(ckpt['scaler_state_dict'])
    history = ckpt.get('history', defaultdict(list))
    epoch = ckpt['epoch']
    batch_idx = ckpt['batch_idx']
    global_step = ckpt['global_step']
    print(f"🔄 Checkpoint chargé : epoch {epoch}, batch {batch_idx}, step {global_step}")
    return epoch, batch_idx, global_step, history


# ============================================================================
# CELL 15 : TTA (TEST-TIME AUGMENTATION) FUNCTIONS
# ============================================================================
def apply_tta_to_features(feats, model, n_views=None):
    """Applique TTA sur les features en créant des vues augmentées."""
    if n_views is None:
        n_views = CONFIG['tta_views']
    augmented_logits = []

    # Vue originale
    with torch.no_grad():
        logits = model(feats)
        augmented_logits.append(logits)

    # Vues augmentées (bruit gaussien léger sur les features)
    for _ in range(n_views - 1):
        noise = torch.randn_like(feats) * 0.1  # bruit léger
        augmented_feats = feats + noise
        with torch.no_grad():
            logits = model(augmented_feats)
            augmented_logits.append(logits)

    # Moyenne des logits
    avg_logits = torch.mean(torch.stack(augmented_logits), dim=0)
    return avg_logits


# ============================================================================
# CELL 16 : CALIBRATION (TEMPERATURE SCALING)
# ============================================================================
class TemperatureScaler(nn.Module):
    """Module pour calibrer les logits avec une température."""
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, logits):
        return logits / self.temperature

    def calibrate(self, model, val_loader, num_epochs=5):
        """Calibre la température sur le validation set."""
        print("🔥 Calibration de la température...")
        self.train()
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval_loss():
            total_loss = 0.0
            n_samples = 0
            with torch.no_grad():
                for feats, labels in val_loader:
                    feats = feats.to(DEVICE)
                    labels = labels.to(DEVICE)
                    logits = model(feats)
                    calibrated_logits = self(logits)
                    loss = F.cross_entropy(calibrated_logits, labels)
                    total_loss += loss.item() * feats.size(0)
                    n_samples += feats.size(0)
            return total_loss / n_samples

        def closure():
            optimizer.zero_grad()
            loss = eval_loss()
            loss.backward()
            return loss

        for _ in range(num_epochs):
            optimizer.step(closure)

        self.eval()
        print(f"✅ Température calibrée : {self.temperature.item():.3f}")
        return self.temperature.item()


# Initialiser le scaler de température
temp_scaler = TemperatureScaler(CONFIG['temperature_init'])


# ============================================================================
# CELL 17 : TRAIN / VALIDATE
# ============================================================================
def topk_acc(logits, targets, k=(1, 5)):
    with torch.no_grad():
        maxk = max(k)
        _, pred = logits.topk(maxk, dim=1)
        correct = pred.eq(targets.unsqueeze(1))
        return {f'top{ki}': correct[:, :ki].any(1).float().mean().item()
                for ki in k}


def train_epoch(model, loader, optimizer, scaler, epoch, start_batch=0, global_step=0):
    model.train()
    total_loss, top1_list = 0.0, []

    # Reprendre au batch précis
    data_iter = iter(loader)
    for _ in range(start_batch):
        next(data_iter)

    accum_steps = CONFIG['grad_accum_steps']
    optimizer.zero_grad()

    for batch_idx, (feats, labels) in enumerate(data_iter, start=start_batch):
        feats  = feats.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        # MixUp sur les features (pas sur les images — beaucoup plus stable)
        feats, la, lb, lam = feature_mixup(feats, labels, CONFIG['mixup_alpha'])

        # Accumulation de gradients
        with autocast(enabled=scaler.is_enabled()):
            logits = model(feats)
            loss   = lam * criterion(logits, la) + (1 - lam) * criterion(logits, lb)
            loss = loss / accum_steps  # normaliser par nombre d'accumulations

        scaler.scale(loss).backward()

        # Accumuler les gradients
        if (batch_idx + 1) % accum_steps == 0:
            scaler.unscale(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps  # remettre à l'échelle
        top1_list.append(topk_acc(logits.detach(), la)['top1'])

        global_step += 1

        # Sauvegarde périodique
        if global_step % CONFIG['resume_save_every_steps'] == 0:
            save_checkpoint(epoch, batch_idx + 1, global_step, model, ema, optimizer, scheduler, scaler, history, CKPT_DIR / f'checkpoint_step_{global_step}.pt')

    return total_loss / len(loader), float(np.mean(top1_list)), global_step


@torch.no_grad()
def validate(model, loader, use_ema=True, use_tta=False, use_calibration=False):
    orig = None
    if use_ema:
        orig = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema.state_dict())
    model.eval()

    total_loss = 0.0
    top1_l, top5_l = [], []
    all_preds, all_labels = [], []

    for feats, labels in loader:
        feats  = feats.to(DEVICE)
        labels = labels.to(DEVICE)

        if use_tta:
            logits = apply_tta_to_features(feats, model, n_views=CONFIG['tta_views'])
        else:
            logits = model(feats)

        if use_calibration:
            logits = temp_scaler(logits)

        total_loss += criterion(logits, labels).item()
        m = topk_acc(logits, labels, k=(1, 5))
        top1_l.append(m['top1']); top5_l.append(m['top5'])
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    if orig:
        model.load_state_dict(orig)

    rare_idx  = {class_to_idx[c] for c in rare_classes if c in class_to_idx}
    rare_mask = np.array([l in rare_idx for l in all_labels])
    rare_acc  = (float(np.mean(np.array(all_preds)[rare_mask]
                               == np.array(all_labels)[rare_mask]))
                 if rare_mask.any() else float('nan'))

    return {
        'loss':     total_loss / len(loader),
        'top1':     float(np.mean(top1_l)),
        'top5':     float(np.mean(top5_l)),
        'rare_acc': rare_acc,
        'preds':    all_preds,
        'labels':   all_labels,
    }


# ============================================================================
# CELL 16 : BOUCLE D'ENTRAÎNEMENT DU CLASSIFIEUR (PHASE 2)
# ============================================================================
print("\n" + "="*60)
print("🚀 PHASE 2 : ENTRAÎNEMENT DU CLASSIFIEUR MLP")
print("="*60)
print(f"   {len(X_train):,} vecteurs × {CONFIG['embed_dim']}D")
print(f"   Epochs rapides (~quelques secondes chacune)")
print(f"   Checkpointing every {CONFIG['resume_save_every_steps']} steps")

history      = defaultdict(list)
best_top1    = 0.0
patience_ctr = 0
t_start      = time.time()

# Charger checkpoint si reprise
if CONFIG['resume']:
    ckpt_path = CKPT_DIR / 'latest_checkpoint.pt'
    if ckpt_path.exists():
        CONFIG['resume_epoch'], CONFIG['resume_batch_idx'], CONFIG['global_step'], history = load_checkpoint(
            ckpt_path, classifier, ema, optimizer, scheduler, scaler)
    else:
        print("⚠️  Aucun checkpoint trouvé pour reprise, démarrage depuis zéro")

global_step = CONFIG['global_step']

for epoch in range(CONFIG['resume_epoch'], CONFIG['num_epochs']):
    train_loss, train_top1, global_step = train_epoch(
        classifier, train_loader, optimizer, scaler, epoch, CONFIG['resume_batch_idx'], global_step)

    # Validation seulement toutes les N epochs pour réduire le coût
    val_every = CONFIG['val_every_epochs']
    if epoch % val_every == 0 or epoch == CONFIG['num_epochs'] - 1:
        val_m = validate(classifier, val_loader)

        # TTA périodique
        use_tta = (epoch + 1) % CONFIG['tta_every_epochs'] == 0
        if use_tta:
            print(f"  🎭 Appliquant TTA à l'epoch {epoch+1}...")
            val_m_tta = validate(classifier, val_loader, use_tta=True)
            val_m = val_m_tta  # utiliser les résultats TTA pour les métriques

        # Calibration périodique
        use_calibration = (epoch + 1) % CONFIG['calibrate_epochs'] == 0
        if use_calibration:
            print(f"  🔥 Calibrant le modèle à l'epoch {epoch+1}...")
            temp_scaler.calibrate(classifier, val_loader)
            val_m_cal = validate(classifier, val_loader, use_calibration=True)
            val_m = val_m_cal  # utiliser les résultats calibrés

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_top1'].append(train_top1)
        history['val_loss'].append(val_m['loss'])
        history['val_top1'].append(val_m['top1'])
        history['val_top5'].append(val_m['top5'])
        history['val_rare'].append(
            val_m['rare_acc'] if not math.isnan(val_m['rare_acc']) else 0.0)

        # Afficher toutes les 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed = time.time() - t_start
            lr_now  = optimizer.param_groups[0]['lr']
            tta_status = " + TTA" if use_tta else ""
            cal_status = " + CAL" if use_calibration else ""
            print(f"  E{epoch+1:>3}/{CONFIG['num_epochs']} | "
                  f"train={train_top1:.4f}  val={val_m['top1']:.4f}  "
                  f"top5={val_m['top5']:.4f}  rare={val_m['rare_acc']:.4f}  "
                  f"lr={lr_now:.1e}{tta_status}{cal_status}  [{elapsed/60:.1f}min]")

        if val_m['top1'] > best_top1:
            best_top1    = val_m['top1']
            patience_ctr = 0
            torch.save({
                'epoch':            epoch,
                'model_state_dict': classifier.state_dict(),
                'ema_state_dict':   ema.state_dict(),
                'val_top1':         val_m['top1'],
                'val_top5':         val_m['top5'],
                'rare_acc':         val_m['rare_acc'],
                'class_to_idx':     class_to_idx,
                'idx_to_class':     idx_to_class,
                'config':           CONFIG,
            }, CKPT_DIR / 'best_classifier.pt')
        else:
            patience_ctr += 1
            if patience_ctr >= CONFIG['patience']:
                print(f"\n  ⏹️  Early stopping à epoch {epoch+1}")
                break
    else:
        # Pas de validation cette epoch - juste scheduler step
        scheduler.step()
        history['train_loss'].append(train_loss)
        history['train_top1'].append(train_top1)
        # Remplir avec des valeurs par défaut pour les métriques de validation
        history['val_loss'].append(history['val_loss'][-1] if history['val_loss'] else 0.0)
        history['val_top1'].append(history['val_top1'][-1] if history['val_top1'] else 0.0)
        history['val_top5'].append(history['val_top5'][-1] if history['val_top5'] else 0.0)
        history['val_rare'].append(history['val_rare'][-1] if history['val_rare'] else 0.0)

    # Sauvegarde du dernier checkpoint pour reprise
    save_checkpoint(epoch + 1, 0, global_step, classifier, ema, optimizer, scheduler, scaler, history, CKPT_DIR / 'latest_checkpoint.pt')

    # Reset batch_idx après epoch complète
    CONFIG['resume_batch_idx'] = 0

total_time = time.time() - t_start
print(f"\n✅ Classifieur entraîné en {total_time/60:.1f} min | "
      f"meilleure val top-1 = {best_top1:.4f}")


# ============================================================================
# CELL 17 : GRAPHIQUES
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle(f"DINOv2 Hybrid — {CONFIG['backbone']} + MLP{CONFIG['hidden_dims']}",
             weight='bold')

axes[0].plot(history['train_loss'], color='steelblue', label='Train')
axes[0].plot(history['val_loss'],   color='tomato',    label='Val')
axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(alpha=.3)

axes[1].plot(history['train_top1'],    color='steelblue', label='Train Top1')
axes[1].plot(history['val_top1'],      color='tomato',    label='Val Top1')
axes[1].plot(history['val_top5'],      color='seagreen',  label='Val Top5', linestyle='--')
axes[1].axhline(0.95, color='gold', linestyle=':', linewidth=1.5, label='95%')
axes[1].set_title('Accuracy'); axes[1].legend(); axes[1].grid(alpha=.3)

axes[2].plot(history['val_rare'], color='darkorange')
axes[2].set_title('RARE Classes Accuracy'); axes[2].grid(alpha=.3)

plt.tight_layout()
plt.savefig(LOG_DIR / 'training_curves.png', dpi=150, bbox_inches='tight')
plt.show()

with open(LOG_DIR / 'training_logs.json', 'w') as f:
    json.dump({k: [float(v) for v in vals]
               for k, vals in history.items()}, f, indent=2)


# ============================================================================
# CELL 18 : ÉVALUATION FINALE
# ============================================================================
print("\n🔄 Chargement meilleur checkpoint...")
ckpt = torch.load(CKPT_DIR / 'best_classifier.pt')
classifier.load_state_dict(ckpt['model_state_dict'])
classifier.eval()

# Évaluation finale avec TTA et calibration
print("📊 Évaluation finale avec TTA + Calibration...")
final = validate(classifier, val_loader, use_ema=False, use_tta=True, use_calibration=True)
all_p = np.array(final['preds'])
all_l = np.array(final['labels'])

print(f"\n📊 RÉSULTATS FINAUX (avec TTA + Calibration)")
print(f"   Top-1    : {final['top1']:.4f}  ({final['top1']*100:.2f}%)")
print(f"   Top-5    : {final['top5']:.4f}")
print(f"   RARE acc : {final['rare_acc']:.4f}")

# Per-class accuracy
per_cls = {}
for i in range(NUM_CLASSES):
    mask = all_l == i
    if not mask.any():
        continue
    cls_name = idx_to_class[i]
    per_cls[cls_name] = {
        'accuracy': round(float((all_p[mask] == i).mean()), 4),
        'n_val':    int(mask.sum()),
        'level':    class_report.get(cls_name, {}).get('level', '?'),
    }

per_cls_sorted = dict(sorted(per_cls.items(), key=lambda x: x[1]['accuracy']))
with open(LOG_DIR / 'per_class_accuracy.json', 'w') as f:
    json.dump(per_cls_sorted, f, indent=2)

print("\n⚠️  10 classes les plus difficiles :")
for cls_name, info in list(per_cls_sorted.items())[:10]:
    bar = '█' * int(info['accuracy'] * 20) or '▏'
    print(f"  [{info['level']:<8s}] {cls_name:<40s} {bar:<20s} {info['accuracy']:.4f}")

# Matrice de confusion Top 30
top30 = np.argsort(np.bincount(all_l, minlength=NUM_CLASSES))[-30:][::-1]
m30   = np.isin(all_l, top30)
remap = {orig: new for new, orig in enumerate(top30)}
sl    = all_l[m30]; sp = all_p[m30]
sl_r  = np.array([remap[x]         for x in sl])
sp_r  = np.array([remap.get(x, -1) for x in sp])
ok    = sp_r >= 0
cm    = confusion_matrix(sl_r[ok], sp_r[ok], labels=list(range(30)))
names = [idx_to_class[i][:20] for i in top30]

fig, ax = plt.subplots(figsize=(20, 16))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=names, yticklabels=names,
            ax=ax, linewidths=.3, cbar_kws={'shrink': .5})
ax.set_title('Matrice de confusion — Top 30 classes', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=7)
plt.yticks(fontsize=7)
plt.tight_layout()
plt.savefig(LOG_DIR / 'confusion_matrix_top30.png', dpi=150, bbox_inches='tight')
plt.show()


# ============================================================================
# CELL 19 : EXPORT FINAL (backbone + classifieur réunis)
# ============================================================================
torch.save({
    # Classifieur MLP (léger ~quelques MB)
    'classifier_state_dict': classifier.state_dict(),
    'ema_state_dict':        ema.state_dict(),
    # Calibration
    'temperature_scaler':    temp_scaler.state_dict(),
    # Info pour recharger
    'class_to_idx':  class_to_idx,
    'idx_to_class':  idx_to_class,
    'config':        CONFIG,
    'val_top1':      final['top1'],
    'val_top5':      final['top5'],
    'rare_acc':      final['rare_acc'],
    'timestamp':     datetime.now().isoformat(),
    # Note : le backbone DINOv2 est rechargé via torch.hub à l'inférence
    # (pas sauvegardé ici car ~330 MB — utiliser force_extract=False)
}, OUT_DIR / 'hybrid_classifier.pth')

print(f"""
{'='*60}
✅ PIPELINE HYBRIDE TERMINÉ
{'='*60}
🔬 Phase 1 : Extraction features DINOv2 (one-shot)
   → Vecteurs sauvegardés dans {FEAT_DIR}
   → Force_extract=False → rechargement instantané les prochaines fois

🧠 Phase 2 : Classifieur MLP entraîné en {total_time/60:.1f} min
   → Top-1 : {final['top1']:.4f}  ({final['top1']*100:.2f}%)
   → Top-5 : {final['top5']:.4f}
   → RARE  : {final['rare_acc']:.4f}

📁 Fichiers :
   {OUT_DIR}/hybrid_classifier.pth   (classifieur seul, léger)
   {CKPT_DIR}/best_classifier.pt
   {CKPT_DIR}/latest_checkpoint.pt   (pour reprise)
   {CKPT_DIR}/checkpoint_step_*.pt   (sauvegardes périodiques)
   {FEAT_DIR}/features_train.npy     (réutilisable, ne pas supprimer)
   {FEAT_DIR}/features_val.npy
   {LOG_DIR}/training_curves.png
   {LOG_DIR}/per_class_accuracy.json

💡 POUR L'INFÉRENCE sur une nouvelle image :
   1. Charger DINOv2 backbone (gelé)
   2. Extraire le vecteur : feat = backbone(image)[:, 0]
   3. Normaliser L2 : feat = F.normalize(feat, p=2, dim=-1)
   4. Appliquer TTA (optionnel) : feat_tta = apply_tta_to_features(feat, classifier)
   5. Classifier : logits = classifier(feat_tta)
   6. Calibrer : logits = temp_scaler(logits)
   7. Résultat : pred = logits.argmax() → idx_to_class[pred]

💡 POUR AMÉLIORER ENCORE :
   CONFIG['backbone'] = 'dinov2_vitl14'  → +1024D, plus riche
   CONFIG['hidden_dims'] = [2048, 1024, 512, 256]  → MLP plus profond
   CONFIG['num_epochs'] = 200  → laisser converger
   force_extract = True uniquement si tu changes de backbone

💡 POUR RÉDUIRE ENCORE LE COÛT :
   CONFIG['grad_accum_steps'] = 8  # accumulation plus forte
   CONFIG['val_every_epochs'] = 5  # validation moins fréquente
   CONFIG['batch_size'] = 256  # batchs plus petits
   CONFIG['tta_every_epochs'] = 50  # TTA moins fréquent
   CONFIG['calibrate_epochs'] = 100  # calibration moins fréquente

💡 OPTIMISATIONS ACTIVES :
   - Accumulation de gradients (simule batchs plus grands)
   - Validation périodique (pas à chaque epoch)
   - torch.compile pour accélération GPU
   - TTA et calibration moins fréquents
   - Batch size validation plus grand

💡 POUR CHARGER LE MODÈLE CALIBRÉ À L'INFÉRENCE :
   ckpt = torch.load('hybrid_classifier.pth')
   classifier.load_state_dict(ckpt['classifier_state_dict'])
   temp_scaler.load_state_dict(ckpt['temperature_scaler'])
   classifier.eval(); temp_scaler.eval()
""")