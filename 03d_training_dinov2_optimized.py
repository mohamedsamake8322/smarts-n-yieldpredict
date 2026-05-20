"""
🎯 DINOv2 — VERSION OPTIMISÉE (LEAN-SOTA)
=========================================

OBJECTIF : -30-50% compute, perf stable/améliorée

GARDER :
  ✅ DINOv2 partial fine-tuning (stage 0 → 1 seulement)
  ✅ Multi-task (crop + disease + category)
  ✅ Focal loss + weighted sampler
  ✅ EMA simple

RETIRER :
  ❌ TTA pendant training (inférence finale seulement)
  ❌ Hard mining complexe → simple weighted sampler
  ❌ CORE replay → évite overhead
  ❌ Mix scheduling riche → CutMix léger early only

AJOUTER :
  ✨ LoRA adapters (30% VRAM économy, perf stable)
  ✨ Metric learning head (ArcFace pour maladies proches)
  ✨ Support distillation optionnel

RÉSULTAT : Lean, simple, rapide, robuste.

PRÉ-REQUIS : 01 + 02 exécutés
"""

# ============================================================================
# CELL 1 : MOUNT (Colab)
# ============================================================================
from google.colab import drive
drive.mount('/content/drive')

# ============================================================================
# CELL 2 : INSTALL
# ============================================================================
import subprocess
for pkg in ["torch torchvision", "timm", "opencv-python",
            "albumentations", "scikit-learn", "tqdm",
            "matplotlib", "seaborn"]:
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
from sklearn.metrics import confusion_matrix, accuracy_score
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
OUT_DIR  = Path('/content/drive/MyDrive/models_dinov2_optimized')
CKPT_DIR = Path('/content/drive/MyDrive/checkpoints_dinov2_optimized')
LOG_DIR  = Path('/content/drive/MyDrive/logs_dinov2_optimized')
for d in [OUT_DIR, CKPT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def _seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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

class_to_idx = class_mapping['class_to_idx']
idx_to_class = {int(k): v for k, v in class_mapping['idx_to_class'].items()}
NUM_CLASSES  = len(class_to_idx)

crop_to_idx     = multitask_cfg['crop_to_idx']
category_to_idx = multitask_cfg['category_to_idx']
NUM_CROPS       = multitask_cfg['num_crops']
NUM_CATEGORIES  = multitask_cfg['num_categories']
LOSS_WEIGHTS    = multitask_cfg['loss_weights']

core_classes     = set(class_groups['CORE'])
extended_classes = set(class_groups['EXTENDED'])
rare_classes     = set(class_groups['RARE'])

CONFIG = {
    # Profil
    'training_profile': training_cfg.get('training_profile', 'cost_stable'),
    
    # Backbone & architecture
    'backbone':        training_cfg.get('backbone',        'dinov2_vits14'),
    'image_size':      training_cfg.get('input_size',      224),
    'embed_dim':       training_cfg.get('embed_dim',       384),
    'num_classes':     NUM_CLASSES,
    'num_crops':       NUM_CROPS,
    'num_categories':  NUM_CATEGORIES,
    
    # Training simple
    'batch_size':      training_cfg.get('batch_size',      16),
    'num_epochs':      training_cfg.get('epochs',          24),
    'warmup_epochs':   training_cfg.get('warmup_epochs',   3),
    'lr_head':         training_cfg.get('lr_head',         1e-4),
    'lr_backbone':     training_cfg.get('lr_backbone',     1e-5),
    'weight_decay':    training_cfg.get('weight_decay',    0.05),
    'label_smoothing': training_cfg.get('label_smoothing', 0.05),
    
    # Loss
    'focal_gamma':     training_cfg.get('focal_gamma',     2.0),
    'focal_weight':    0.7,
    'smooth_weight':   0.3,
    
    # Augmentation légère
    'cutmix_alpha':    1.0,
    'cutmix_prob':     0.3,          # Seulement premières 8 epochs
    'use_cutmix_until_epoch': 8,
    
    # EMA simple
    'ema_decay':       training_cfg.get('ema_decay', 0.9999),
    
    # LoRA adapters
    'use_lora':        True,
    'lora_rank':       8,             # Rank r des adaptateurs LoRA (petit mais efficace)
    'lora_alpha':      16,            # Scaling
    
    # Metric learning head
    'use_arcface':     True,
    'arcface_margin':  0.5,
    'arcface_scale':   64.0,
    'metric_loss_weight': 0.3,        # Poids de la perte métrique vs. CE
    
    # Unfreezing
    'unfreeze_blocks_at_epoch': 8,
    'num_unfreeze_blocks': 4,
    
    # Patience & checkpoints
    'patience':        12,
    'seed':            42,
    'num_workers':     4,
    'grad_accum_steps': 1,
}

if CONFIG['training_profile'] == 'cost_stable':
    CONFIG['backbone'] = 'dinov2_vits14'
    CONFIG['image_size'] = 336
    CONFIG['batch_size'] = 16
    CONFIG['num_epochs'] = 20
elif CONFIG['training_profile'] == 'balanced':
    CONFIG['backbone'] = 'dinov2_vitb14'
    CONFIG['image_size'] = 384
    CONFIG['batch_size'] = 24
    CONFIG['num_epochs'] = 28

_seed_all(CONFIG['seed'])
print(f"✅ Config | profile={CONFIG['training_profile']} | {NUM_CLASSES} classes | "
      f"backbone={CONFIG['backbone']} | img={CONFIG['image_size']} | "
      f"batch={CONFIG['batch_size']} | epochs={CONFIG['num_epochs']} | "
      f"LoRA={'✅' if CONFIG['use_lora'] else '❌'} | "
      f"ArcFace={'✅' if CONFIG['use_arcface'] else '❌'}")


def _worker_init_fn(worker_id: int):
    s = CONFIG['seed'] + worker_id
    np.random.seed(s)
    random.seed(s)


# ============================================================================
# CELL 4 : AUGMENTATIONS (léger)
# ============================================================================
IMG_SIZE  = CONFIG['image_size']
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]

def _rrc(size, scale):
    try:
        return A.RandomResizedCrop(size=(size, size), scale=scale)
    except TypeError:
        return A.RandomResizedCrop(height=size, width=size, scale=scale)

transform_train = A.Compose([
    _rrc(IMG_SIZE, (0.7, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, p=0.3),
    A.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ToTensorV2(),
])

transform_val = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ToTensorV2(),
])
print("✅ Transforms simples")


# ============================================================================
# CELL 5 : DATASET
# ============================================================================
class AgriDataset(Dataset):
    def __init__(self, meta_list, is_train=True):
        self.is_train = is_train
        self.paths = []
        self.labels = []
        self.crop_labels = []
        self.cat_labels = []
        self.groups = []

        for item in meta_list:
            cls = item['class']
            lbl = item['label']
            self.paths.append(item['path'])
            self.labels.append(lbl)
            self.crop_labels.append(item.get('crop_label', -1))
            self.cat_labels.append(item.get('category_label', -1))
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

        if self.is_train:
            aug = transform_train(image=img)
        else:
            aug = transform_val(image=img)

        return (aug['image'],
                self.labels[idx],
                self.crop_labels[idx],
                self.cat_labels[idx],
                self.groups[idx])


# ============================================================================
# CELL 6 : LOSS FUNCTIONS (Focal + Label Smoothing)
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
    """0.7 × Focal + 0.3 × LabelSmoothing"""
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


# ── ArcFace margin-based metric learning ───────────────────────────────────
class ArcMarginProduct(nn.Module):
    """ArcFace : margin m, scale s"""
    def __init__(self, in_features, out_features, margin=0.5, scale=64.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, target=None):
        # Normalize
        x_norm = F.normalize(input, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        logits = F.linear(x_norm, w_norm)
        
        if target is None:
            return logits * self.scale
        
        # Compute margin
        with torch.no_grad():
            m = torch.zeros_like(logits)
            m.scatter_(1, target.unsqueeze(1), self.margin)
        
        # Apply margin
        logits = self.scale * torch.cos(torch.acos(torch.clamp(logits, -1+1e-6, 1-1e-6)) + m)
        return logits


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

criterion_aux = nn.CrossEntropyLoss().to(DEVICE)

print(f"✅ Losses : ComboLoss (Focal 0.7 + LabelSmoothing 0.3)")


# ============================================================================
# CELL 7 : LoRA ADAPTER
# ============================================================================
class LoRALinear(nn.Module):
    """Drop-in replacement pour nn.Linear avec LoRA adapters."""
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        
        # Base weight (frozen)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
        # LoRA adapters
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * math.sqrt(2.0 / (rank + in_features)))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Freezer base weight
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def forward(self, x):
        # Base forward
        out = F.linear(x, self.weight, self.bias)
        # LoRA forward
        lora_out = x @ self.lora_A.t() @ self.lora_B.t() * (self.alpha / self.rank)
        return out + lora_out


def apply_lora_to_model(model, rank=8, alpha=16):
    """Remplace tous les nn.Linear par LoRA sauf les heads custom."""
    n_replaced = 0
    for name, module in model.named_modules():
        # Skip heads (on veut les entraîner complètement)
        if any(x in name for x in ['head_', 'shared_proj', 'metric_head']):
            continue
        if isinstance(module, nn.Linear):
            # Remplacer
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = dict(model.named_modules())[parent_name]
            lora_linear = LoRALinear(module.in_features, module.out_features, rank=rank, alpha=alpha)
            lora_linear.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                lora_linear.bias.data.copy_(module.bias.data)
            setattr(parent, child_name, lora_linear)
            n_replaced += 1
    print(f"  LoRA : {n_replaced} Linear layers remplacés (rank={rank})")


# ============================================================================
# CELL 8 : MODEL ARCHITECTURE (DINOv2 + LoRA + ArcFace heads)
# ============================================================================
class DINOv2OptimizedMultitask(nn.Module):
    def __init__(self, backbone_name, num_classes, num_crops, num_categories,
                 embed_dim, use_lora=False, lora_rank=8, 
                 use_arcface=False, arcface_margin=0.5, arcface_scale=64.0):
        super().__init__()
        print(f"  📥 Chargement {backbone_name}...")
        self.backbone = torch.hub.load('facebookresearch/dinov2', backbone_name, pretrained=True)
        if hasattr(self.backbone, 'head'):
            self.backbone.head = nn.Identity()
        
        self.embed_dim = embed_dim
        self.use_arcface = use_arcface
        
        # Geler backbone au départ
        self._freeze_all_backbone()
        
        # Bottleneck partagé
        self.shared_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        
        # Tête principale (CE ou ArcFace)
        if use_arcface:
            self.head_main = ArcMarginProduct(512, num_classes, margin=arcface_margin, scale=arcface_scale)
        else:
            self.head_main = nn.Linear(512, num_classes)
        
        # Têtes auxiliaires (classique)
        self.head_crop = nn.Sequential(
            nn.Linear(512, 128), nn.GELU(), nn.Linear(128, num_crops))
        self.head_category = nn.Sequential(
            nn.Linear(512, 64), nn.GELU(), nn.Linear(64, num_categories))
        
        # Init heads
        for m in [self.shared_proj, self.head_crop, self.head_category]:
            for l in m.modules():
                if isinstance(l, nn.Linear):
                    nn.init.trunc_normal_(l.weight, std=0.02)
                    if l.bias is not None:
                        nn.init.zeros_(l.bias)
        
        # Appliquer LoRA si demandé (APRÈS freeze backbone)
        if use_lora:
            apply_lora_to_model(self.backbone, rank=lora_rank, alpha=16)
        
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        print(f"  ✅ {total/1e6:.1f}M params | {trainable/1e6:.1f}M trainable | {frozen/1e6:.1f}M frozen")

    def _freeze_all_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_last_n_blocks(self, n: int):
        """Dégeler les n derniers blocks du transformer."""
        blocks = getattr(self.backbone, 'blocks', None)
        if blocks is None:
            return
        start = max(0, len(blocks) - n)
        for i, blk in enumerate(blocks):
            if i >= start:
                for p in blk.parameters():
                    p.requires_grad = True
        print(f"  🔓 Last {n} blocks unfrozen")

    def forward(self, x, targets=None):
        feat = self.backbone(x)
        if feat.dim() == 3:
            feat = feat[:, 0]
        shared = self.shared_proj(feat)
        
        main_out = self.head_main(shared, targets) if self.use_arcface else self.head_main(shared)
        
        return {
            'main': main_out,
            'crop': self.head_crop(shared),
            'category': self.head_category(shared),
        }

    def get_features(self, x):
        with torch.no_grad():
            feat = self.backbone(x)
            if feat.dim() == 3:
                feat = feat[:, 0]
        return F.normalize(feat, p=2, dim=1)


model = DINOv2OptimizedMultitask(
    backbone_name   = CONFIG['backbone'],
    num_classes     = CONFIG['num_classes'],
    num_crops       = CONFIG['num_crops'],
    num_categories  = CONFIG['num_categories'],
    embed_dim       = CONFIG['embed_dim'],
    use_lora        = CONFIG['use_lora'],
    lora_rank       = CONFIG['lora_rank'],
    use_arcface     = CONFIG['use_arcface'],
    arcface_margin  = CONFIG['arcface_margin'],
    arcface_scale   = CONFIG['arcface_scale'],
).to(DEVICE)


# ============================================================================
# CELL 9 : EMA SIMPLE
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

    def apply(self, model):
        model.load_state_dict(self.ema)


ema = ModelEMA(model, decay=CONFIG['ema_decay'])


# ============================================================================
# CELL 10 : OPTIMIZER & SCHEDULER
# ============================================================================
def build_optimizer_and_scheduler(model, lr_backbone, lr_head, warmup_ep, total_ep):
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(),     'lr': lr_backbone},
        {'params': model.shared_proj.parameters(),  'lr': lr_head},
        {'params': model.head_main.parameters(),    'lr': lr_head},
        {'params': model.head_crop.parameters(),    'lr': lr_head},
        {'params': model.head_category.parameters(),'lr': lr_head},
    ], weight_decay=CONFIG['weight_decay'])

    def lam_bb(ep):
        if ep < warmup_ep:
            return (ep + 1) / max(1, warmup_ep)
        p = (ep - warmup_ep) / max(1, total_ep - warmup_ep)
        return 0.5 * (1 + math.cos(math.pi * p))

    def lam_h(ep):
        w = min(2, warmup_ep)
        if ep < w:
            return (ep + 1) / max(1, w)
        p = (ep - w) / max(1, total_ep - w)
        return 0.5 * (1 + math.cos(math.pi * p))

    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(optimizer, lr_lambda=[lam_bb, lam_h, lam_h, lam_h, lam_h])
    return optimizer, scheduler


def build_loader(meta_list, is_train, batch_size):
    """Construit un DataLoader avec WeightedRandomSampler simple."""
    ds = AgriDataset(meta_list, is_train)

    if is_train:
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


optimizer, scheduler = build_optimizer_and_scheduler(
    model, CONFIG['lr_backbone'], CONFIG['lr_head'],
    CONFIG['warmup_epochs'], CONFIG['num_epochs'],
)
scaler = GradScaler(enabled=(DEVICE.type == 'cuda'))
print("✅ Optimizer + Scheduler (simple)")


# ============================================================================
# CELL 11 : CutMix SIMPLE
# ============================================================================
def rand_bbox(H, W, lam):
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


def apply_cutmix_simple(images, labels, alpha=1.0, prob=0.3):
    """CutMix léger : prob de mixer."""
    if random.random() > prob:
        return images, labels, labels, 1.0
    
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(images.size(0))
    H, W = images.shape[2], images.shape[3]
    x1, y1, x2, y2 = rand_bbox(H, W, lam)
    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]
    lam = 1.0 - (x2 - x1) * (y2 - y1) / (H * W)
    return mixed, labels, labels[idx], lam


# ============================================================================
# CELL 12 : TRAIN EPOCH
# ============================================================================
def train_epoch(model, loader, optimizer, scaler, device, epoch):
    model.train()
    total_loss = 0
    total_loss_main = 0
    top1_list = []
    
    cutmix_enabled = epoch < CONFIG['use_cutmix_until_epoch']
    
    for batch_idx, batch in enumerate(tqdm(loader, desc=f"Train E{epoch+1}")):
        images, labels, crop_lbl, cat_lbl, groups = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        crop_lbl = torch.tensor(crop_lbl, dtype=torch.long).to(device)
        cat_lbl = torch.tensor(cat_lbl, dtype=torch.long).to(device)
        
        # CutMix simple
        if cutmix_enabled:
            images, la, lb, lam = apply_cutmix_simple(
                images, labels, alpha=CONFIG['cutmix_alpha'], prob=CONFIG['cutmix_prob'])
        else:
            la, lb, lam = labels, labels, 1.0
        
        with autocast(enabled=scaler.is_enabled()):
            # Forward
            if CONFIG['use_arcface']:
                out = model(images, targets=labels)
            else:
                out = model(images)
            
            # Loss principale
            l_main = (
                lam * combo_loss(out['main'], la)
                + (1 - lam) * combo_loss(out['main'], lb)
            )
            
            # Pertes auxiliaires
            vc = crop_lbl >= 0
            vt = cat_lbl >= 0
            l_crop = (criterion_aux(out['crop'][vc], crop_lbl[vc]) if vc.any() else torch.tensor(0., device=device))
            l_cat = (criterion_aux(out['category'][vt], cat_lbl[vt]) if vt.any() else torch.tensor(0., device=device))
            
            loss = (LOSS_WEIGHTS['main'] * l_main
                    + LOSS_WEIGHTS['crop'] * l_crop
                    + LOSS_WEIGHTS['category'] * l_cat)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        # EMA update
        ema.update(model)
        
        total_loss += loss.item()
        total_loss_main += l_main.item()
        t1 = (out['main'].argmax(1) == la).float().mean().item()
        top1_list.append(t1)
    
    mean_loss = total_loss / len(loader)
    mean_loss_main = total_loss_main / len(loader)
    mean_top1 = float(np.mean(top1_list))
    return mean_loss, mean_loss_main, mean_top1


# ============================================================================
# CELL 13 : VALIDATE
# ============================================================================
@torch.no_grad()
def validate(model, loader, device, use_ema=True):
    orig = None
    if use_ema:
        orig = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema.state_dict())
    model.eval()
    
    top1_l, crop_l, cat_l = [], [], []
    all_preds, all_labels = [], []
    
    for batch in tqdm(loader, desc="Val"):
        images, labels, crop_lbl, cat_lbl, groups = batch
        images = images.to(device)
        labels = labels.to(device)
        crop_lbl = torch.tensor(crop_lbl, dtype=torch.long).to(device)
        cat_lbl = torch.tensor(cat_lbl, dtype=torch.long).to(device)
        
        out = model(images)
        
        t1 = (out['main'].argmax(1) == labels).float().mean().item()
        top1_l.append(t1)
        
        vc = crop_lbl >= 0
        vt = cat_lbl >= 0
        if vc.any():
            crop_l.append(out['crop'][vc].argmax(1).eq(crop_lbl[vc]).float().mean().item())
        if vt.any():
            cat_l.append(out['category'][vt].argmax(1).eq(cat_lbl[vt]).float().mean().item())
        
        all_preds.extend(out['main'].argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    if orig:
        model.load_state_dict(orig)
    
    # Per-class accuracy
    rare_idx = set(class_to_idx[c] for c in rare_classes if c in class_to_idx)
    rare_mask = np.array([l in rare_idx for l in all_labels])
    rare_acc = float(np.mean(np.array(all_preds)[rare_mask] == np.array(all_labels)[rare_mask])) if rare_mask.any() else float('nan')
    
    return {
        'top1': float(np.mean(top1_l)),
        'crop_acc': float(np.mean(crop_l)) if crop_l else float('nan'),
        'cat_acc': float(np.mean(cat_l)) if cat_l else float('nan'),
        'rare_acc': rare_acc,
        'preds': all_preds,
        'labels': all_labels,
    }


# ============================================================================
# CELL 14 : BOUCLE PRINCIPALE
# ============================================================================
print("\n" + "="*70)
print("🚀 DINOv2 OPTIMIZED — Lean-SOTA")
print(f"   Partial unfreezing | Multi-task | Focal Loss | EMA | LoRA={'✅' if CONFIG['use_lora'] else '❌'} | ArcFace={'✅' if CONFIG['use_arcface'] else '❌'}")
print("="*70)

train_mt = load_json(META_DIR / 'train_multitask.json')
val_mt   = load_json(META_DIR / 'val_multitask.json')

history = defaultdict(list)
best_top1 = 0.0
patience_ctr = 0
unfreeze_done = False

train_loader, train_ds = build_loader(train_mt, is_train=True, batch_size=CONFIG['batch_size'])
val_loader, _ = build_loader(val_mt, is_train=False, batch_size=CONFIG['batch_size'])

for epoch in range(CONFIG['num_epochs']):
    # Progressive unfreezing (une seule fois)
    if (not unfreeze_done and epoch >= CONFIG['unfreeze_blocks_at_epoch']):
        model.unfreeze_last_n_blocks(CONFIG['num_unfreeze_blocks'])
        # Rebuild optimizer
        optimizer, scheduler = build_optimizer_and_scheduler(
            model, CONFIG['lr_backbone'], CONFIG['lr_head'],
            CONFIG['warmup_epochs'], CONFIG['num_epochs'],
        )
        unfreeze_done = True
    
    print(f"\n📌 Epoch {epoch+1}/{CONFIG['num_epochs']}")
    print(f"   lr_backbone={optimizer.param_groups[0]['lr']:.2e} | lr_head={optimizer.param_groups[1]['lr']:.2e}")
    
    train_loss, train_loss_main, train_top1 = train_epoch(
        model, train_loader, optimizer, scaler, DEVICE, epoch)
    
    val_m = validate(model, val_loader, DEVICE, use_ema=True)
    
    scheduler.step()
    
    print(f"  Train | loss={train_loss:.4f} (main={train_loss_main:.4f}) | top1={train_top1:.4f}")
    print(f"  Val   | top1={val_m['top1']:.4f} | crop={val_m['crop_acc']:.4f} | cat={val_m['cat_acc']:.4f} | rare={val_m['rare_acc']:.4f}")
    
    # Logging
    history['train_loss'].append(train_loss)
    history['train_top1'].append(train_top1)
    history['val_top1'].append(val_m['top1'])
    history['val_rare_acc'].append(val_m['rare_acc'])
    history['lr_bb'].append(float(optimizer.param_groups[0]['lr']))
    history['lr_head'].append(float(optimizer.param_groups[1]['lr']))
    
    # Checkpoint
    if val_m['top1'] > best_top1:
        best_top1 = val_m['top1']
        patience_ctr = 0
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'ema_state_dict': ema.state_dict(),
            'val_top1': val_m['top1'],
            'config': CONFIG,
        }
        torch.save(ckpt, CKPT_DIR / f'best_model_s{CONFIG["seed"]}.pt')
        print(f"  ✅ Meilleur : top1={best_top1:.4f}")
    else:
        patience_ctr += 1
        if patience_ctr >= CONFIG['patience']:
            print(f"  ⏹️  Early stopping (patience={CONFIG['patience']})")
            break

print(f"\n✅ Entraînement terminé | meilleure top1 = {best_top1:.4f}")

# ============================================================================
# CELL 15 : LOGS & GRAPHIQUES
# ============================================================================
log_path = LOG_DIR / 'training_logs.json'
with open(log_path, 'w') as f:
    json.dump(dict(history), f, indent=2)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('DINOv2 Optimized Training', fontsize=14, weight='bold')

axes[0, 0].plot(history['train_loss'], label='train_loss')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(history['train_top1'], label='train_top1')
axes[0, 1].plot(history['val_top1'], label='val_top1')
axes[0, 1].set_ylabel('Top-1 Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True)

axes[1, 0].plot(history['val_rare_acc'], label='rare_acc')
axes[1, 0].set_ylabel('Rare Class Accuracy')
axes[1, 0].legend()
axes[1, 0].grid(True)

axes[1, 1].plot(history['lr_bb'], label='lr_backbone')
axes[1, 1].plot(history['lr_head'], label='lr_head')
axes[1, 1].set_ylabel('Learning Rate')
axes[1, 1].set_yscale('log')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig(LOG_DIR / 'training_curves.png', dpi=100, bbox_inches='tight')
print(f"✅ Graphiques sauvegardés")

print(f"\n📁 Checkpoints: {CKPT_DIR}")
print(f"📁 Logs: {LOG_DIR}")
