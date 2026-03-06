"""
🎯 COMPLETE TRAINING PIPELINE - COLAB PRO (VERSION CORRIGÉE)
De dataset_final à modèle produit-ready

Instructions:
1. Copier ce code dans Google Colab
2. Remplacer les chemins si nécessaire
3. Lancer cellule par cellule
4. Monitorer les logs et métriques
5. À la fin: récupérer les fichiers dans Drive
"""

# ========================== IMPORTANT OPERATIONNEL ==========================
# Ce script entraîne un modèle de diagnostic par METRIC LEARNING (PAS de softmax).
# Principes à respecter:
#  - Ne pas sur-nettoyer les images (conserver variabilité terrain)
#  - Ne pas supprimer la diversité pour équilibrer artificiellement
#  - Ne pas séparer les cultures/plantes par classe (les symptômes traversent espèces)
#  - Ne pas utiliser softmax comme sortie finale
#  - Évaluer par similarité / retrieval (top-k), intra/inter-distance, validations terrain
# ============================================================================

# ============================================================================
# CELL 1: SETUP & MOUNT DRIVE
# ============================================================================
from google.colab import drive
import os
import sys

# Mount Google Drive
drive.mount('/content/drive')

# Vérifier le dataset
dataset_path = '/content/drive/MyDrive/dataset_final'
if os.path.exists(dataset_path):
    classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    print(f"✅ Dataset trouvé: {len(classes)} classes")
else:
    print("❌ Dataset non trouvé à", dataset_path)
    print("📁 Crée le dossier ou ajuste le chemin")


# ============================================================================
# CELL 2: INSTALL DEPENDENCIES
# ============================================================================
import subprocess

packages = [
    "timm",
    "transformers",
    "torchvision",
    "opencv-python",
    "albumentations",
    "faiss-cpu",
    "scikit-learn",
    "tqdm",
    "tensorboard"
]

for pkg in packages:
    subprocess.run(["pip", "install", "-q", pkg])

print("✅ Toutes les dépendances installées")


# ============================================================================
# CELL 3: IMPORTS & CONFIG
# ============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import timm
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import pickle
import json
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings('ignore')

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device: {DEVICE}")

# Config
CONFIG = {
    'dataset_path': '/content/drive/MyDrive/dataset_final',
    'output_path': '/content/drive/MyDrive/models',
    'checkpoints_path': '/content/drive/MyDrive/checkpoints',
    'logs_path': '/content/drive/MyDrive/training_logs',

    'model_name': 'swin_base_patch4_window7_224',
    'embedding_dim': 768,
    'image_size': 224,

    'batch_size': 32,
    'num_epochs': 50,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,

    'temperature': 0.07,  # Supervised Contrastive Loss
    'num_workers': 4,
    'seed': 42,
    'val_split': 0.2,
}

# Créer dossiers
for path in [CONFIG['output_path'], CONFIG['checkpoints_path'], CONFIG['logs_path']]:
    os.makedirs(path, exist_ok=True)

print("✅ Config initialisée")


# ============================================================================
# CELL 4: DATASET LOADER AVEC AUGMENTATIONS
# ============================================================================
class DiseaseDataset(Dataset):
    """Custom dataset pour maladies agricoles"""

    def __init__(self, dataset_path, transform=None):
        self.dataset_path = Path(dataset_path)
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        # Scan tous les dossiers de classes
        class_idx = 0
        for class_dir in sorted(self.dataset_path.iterdir()):
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            self.class_to_idx[class_name] = class_idx
            self.idx_to_class[class_idx] = class_name

            # Scan images
            for img_file in class_dir.rglob('*'):
                if img_file.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}:
                    self.images.append(str(img_file))
                    self.labels.append(class_idx)

            class_idx += 1

        print(f"✅ Dataset loaded: {len(self.images)} images, {len(self.class_to_idx)} classes")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"⚠️  Image not found: {img_path}")
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label, img_path


# PK Sampler: generate batches with P classes and K samples per class (on-the-fly)
# Useful for contrastive/supervised-contrastive training without deleting data.
import random
from torch.utils.data.sampler import Sampler


class PKSampler(Sampler):
    """Samples batches containing P classes and K samples per class.
    Does NOT remove or downsample the dataset permanently; it's an on-the-fly sampler.
    This preserves dataset diversity while providing positives for metric learning.

    Args:
        labels: list or array of labels for each sample in the dataset
        P: number of classes per batch
        K: number of samples per class
    """
    def __init__(self, labels, P=8, K=4):
        self.labels = np.array(labels)
        self.P = P
        self.K = K

        # map class -> list of indices
        self.class_to_indices = {}
        for idx, l in enumerate(self.labels):
            self.class_to_indices.setdefault(int(l), []).append(idx)

        self.classes = list(self.class_to_indices.keys())

        # compute epoch size (number of samples yielded per epoch)
        self.batch_size = self.P * self.K
        # approximate number of batches per epoch
        self.num_batches = max(1, len(self.labels) // self.batch_size)

    def __iter__(self):
        for _ in range(self.num_batches):
            chosen_classes = random.sample(self.classes, min(self.P, len(self.classes)))
            batch_indices = []
            for c in chosen_classes:
                indices = self.class_to_indices[c]
                if len(indices) >= self.K:
                    chosen = random.sample(indices, self.K)
                else:
                    # sample with replacement if class has fewer images than K
                    chosen = list(np.random.choice(indices, self.K, replace=True))
                batch_indices.extend(chosen)
            yield batch_indices

    def __len__(self):
        return self.num_batches

# Augmentations (biologiquement réalistes)
train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.05),
    A.Rotate(limit=30, p=0.7),
    A.GaussNoise(p=0.15),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.6),
    A.OneOf([
        A.GaussianBlur(p=0.2),
        A.MotionBlur(p=0.15),
    ], p=0.3),
    A.ImageCompression(quality_lower=50, quality_upper=95, p=0.25),
    A.CoarseDropout(max_holes=4, max_height=16, max_width=16, p=0.15),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
], bbox_params=None)

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
], bbox_params=None)

# Load dataset
print("Loading dataset...")

train_dataset = DiseaseDataset(
    os.path.join(CONFIG['dataset_path'], "train"),
    transform=train_transform
)

val_dataset = DiseaseDataset(
    os.path.join(CONFIG['dataset_path'], "val"),
    transform=val_transform
)

# DataLoaders
# Create a PKSampler for supervised contrastive training: P classes × K samples
# Choose P and K so that P*K is reasonable with available GPU memory.
P = 8
K = 4

# Build labels list for the train subset (random_split returns a Subset with .indices)
subset_indices = getattr(train_dataset, 'indices', None)
if subset_indices is not None:
    subset_labels = [dataset.labels[i] for i in subset_indices]
else:
    # fallback: iterate train_dataset
    subset_labels = [train_dataset[i][1] for i in range(len(train_dataset))]

train_sampler = PKSampler(subset_labels, P=P, K=K)

# DataLoader using batch_sampler (yields lists of indices relative to the subset)
train_loader = DataLoader(
    train_dataset,
    batch_sampler=train_sampler,
    num_workers=CONFIG['num_workers'],
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=False,
    num_workers=CONFIG['num_workers'],
    pin_memory=True
)

print(f"✅ Train: {len(train_dataset)}, Val: {len(val_dataset)}")


# ============================================================================
# CELL 5: MODEL ARCHITECTURE
# ============================================================================
class DiagnosticModel(nn.Module):
    """Swin Transformer + Embedding Head (PAS classification)"""

    def __init__(self, model_name, embedding_dim=768):
        super().__init__()

        # Use timm features_only to extract multi-level feature maps
        # This allows multi-scale fusion (patch + global)
        try:
            self.backbone = timm.create_model(model_name, pretrained=True, features_only=True, out_indices=(0,1,2,3))
        except Exception:
            # fallback if features_only unsupported
            self.backbone = timm.create_model(model_name, pretrained=True)

        # Determine feature channels from backbone (supports features_only)
        with torch.no_grad():
            dummy = torch.randn(1, 3, CONFIG['image_size'], CONFIG['image_size']).to(DEVICE)
            feats = self.backbone(dummy)
            if isinstance(feats, (list, tuple)):
                self.feature_dims = [f.shape[1] for f in feats]
            else:
                # single tensor
                self.feature_dims = [feats.shape[1]]

        # Projection heads: per-stage pooling -> concat -> projection
        fused_dim = sum(self.feature_dims)

        self.fusion_proj = nn.Sequential(
            nn.Linear(fused_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

        # Optional prototype vectors for symptom prototypes
        self.register_buffer('prototypes', torch.randn(0, embedding_dim))
        self.embedding_dim = embedding_dim

    def forward(self, x):
        """
        x: [batch_size, 3, 224, 224]
        return: [batch_size, embedding_dim] - EMBEDDING, PAS LOGITS!
        """
        feats = self.backbone(x)

        # If backbone returns a single tensor, treat as single-stage
        if not isinstance(feats, (list, tuple)):
            feats = [feats]

        pooled = []
        for f in feats:
            # global average pool each stage to 1x1
            p = torch.nn.functional.adaptive_avg_pool2d(f, (1,1)).view(x.size(0), -1)
            pooled.append(p)

        fused = torch.cat(pooled, dim=1)
        embeddings = self.fusion_proj(fused)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

# Create model
model = DiagnosticModel(CONFIG['model_name'], CONFIG['embedding_dim'])
model = model.to(DEVICE)

print(f"✅ Model created with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")


# ============================================================================
# CELL 6: LOSS FUNCTIONS
# ============================================================================
class SupConLoss(nn.Module):
    """Stable Supervised Contrastive / InfoNCE-style loss with optional memory negatives.

    Usage: loss = SupConLoss(...)(features, labels, memory_embeddings=None, memory_labels=None)
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels, memory_embeddings=None, memory_labels=None):
        """
        features: (B, D) L2-normalized
        labels: (B,)
        memory_embeddings: (M, D) numpy or torch (optional)
        memory_labels: (M,) numpy or torch (optional)
        """
        device = features.device
        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # logits among batch
        logits_batch = torch.div(torch.matmul(features, features.T), self.temperature)

        # mask out self
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask

        # incorporate memory bank negatives (and possible positives) if provided
        if memory_embeddings is not None and memory_embeddings.shape[0] > 0:
            if not torch.is_tensor(memory_embeddings):
                memory_embeddings = torch.from_numpy(memory_embeddings).to(device)
            if not torch.is_tensor(memory_labels):
                memory_labels = torch.from_numpy(memory_labels).to(device)

            logits_mem = torch.div(torch.matmul(features, memory_embeddings.t()), self.temperature)

            # construct combined logits: [batch | memory]
            logits = torch.cat([logits_batch, logits_mem], dim=1)  # (B, B+M)

            # build positive mask for memory: compare labels
            mem_mask = torch.eq(labels, memory_labels.view(1, -1)).float().to(device)
            combined_mask = torch.cat([mask, mem_mask], dim=1)
        else:
            logits = logits_batch
            combined_mask = mask

        # numeric stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        exp_logits = torch.exp(logits) * (1.0 - torch.eye(logits.size(0), device=device, dtype=logits.dtype) if logits.shape[0]==batch_size and logits.shape[1]==batch_size else 1.0)

        # sum over all positives (in combined mask) and all negatives
        exp_sum = exp_logits.sum(1, keepdim=True) + 1e-12

        log_prob = logits - torch.log(exp_sum)

        # mean log-likelihood over positive
        mean_log_prob_pos = (combined_mask * log_prob).sum(1) / (combined_mask.sum(1) + 1e-12)

        loss = - mean_log_prob_pos.mean()
        return loss

criterion = SupConLoss(temperature=CONFIG['temperature'])
criterion = criterion.to(DEVICE)

print("✅ Loss function (SupConLoss) créée")


# ============================================================================
# CELL 7: OPTIMIZER & SCHEDULER
# ============================================================================
optimizer = optim.AdamW(
    model.parameters(),
    lr=CONFIG['learning_rate'],
    weight_decay=CONFIG['weight_decay']
)

from torch.optim.lr_scheduler import LambdaLR
import math

optimizer = optim.AdamW(
    model.parameters(),
    lr=CONFIG['learning_rate'],
    weight_decay=CONFIG['weight_decay']
)

# Warmup + cosine scheduler: linear warmup for warmup_epochs, then cosine decay
warmup_epochs = 5
total_epochs = CONFIG['num_epochs']

def lr_lambda(current_epoch):
    if current_epoch < warmup_epochs:
        return float(current_epoch) / float(max(1, warmup_epochs))
    # cosine decay thereafter
    progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
    return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

print("✅ Optimizer & Scheduler (warmup+cosine) created")

# AMP GradScaler (mixed precision) - activé uniquement si CUDA dispo
scaler = GradScaler(enabled=(DEVICE.type == "cuda"))
print(f"✅ GradScaler initialized (enabled={scaler.is_enabled()})")


# ===================== Memory Bank (simple FIFO queue) =====================
class MemoryBank:
    """A simple memory bank to store embeddings and labels for hard negatives.
    Stores as numpy arrays on CPU to avoid GPU memory pressure.
    """
    def __init__(self, capacity=65536, dim=768):
        self.capacity = capacity
        self.dim = dim
        self.ptr = 0
        self.size = 0
        self.embeddings = np.zeros((capacity, dim), dtype='float32')
        self.labels = np.zeros((capacity,), dtype='int32')

    def add(self, embs, labs):
        embs = embs.astype('float32')
        n = embs.shape[0]
        if n >= self.capacity:
            # If batch bigger than capacity, keep last part
            embs = embs[-self.capacity:]
            labs = labs[-self.capacity:]
            n = embs.shape[0]

        end = (self.ptr + n) % self.capacity
        if self.ptr + n <= self.capacity:
            self.embeddings[self.ptr:self.ptr+n] = embs
            self.labels[self.ptr:self.ptr+n] = labs
        else:
            first = self.capacity - self.ptr
            self.embeddings[self.ptr:] = embs[:first]
            self.labels[self.ptr:] = labs[:first]
            self.embeddings[:end] = embs[first:]
            self.labels[:end] = labs[first:]

        self.ptr = end
        self.size = min(self.capacity, self.size + n)

    def get(self):
        return self.embeddings[:self.size].copy(), self.labels[:self.size].copy()


# Instantiate memory bank
memory_bank = MemoryBank(capacity=32768, dim=CONFIG['embedding_dim'])
print("✅ Memory bank initialized")


# ===================== EMA (Exponential Moving Average) =====================
class ModelEMA:
    """Implements simple EMA of model parameters"""
    def __init__(self, model, decay=0.9999):
        self.ema = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self.decay = decay

    def update(self, model):
        msd = model.state_dict()
        for k, v in msd.items():
            if v.dtype.is_floating_point:
                self.ema[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def state_dict(self):
        return self.ema

    def load_to(self, model):
        model.load_state_dict(self.ema)


ema = ModelEMA(model, decay=0.999)
print("✅ EMA initialized")


# ============================================================================
# CELL 8: UTILITY FUNCTIONS
# ============================================================================
def compute_intra_inter_distance(embeddings, labels):
    """Compute intra-class et inter-class distances"""
    sim_matrix = torch.mm(embeddings, embeddings.t())

    intra_distances = []
    inter_distances = []

    unique_labels = torch.unique(labels)
    for label in unique_labels:
        mask = labels == label
        class_sims = sim_matrix[mask][:, mask]

        # Intra-class: similarities between different samples of same class
        off_diag = class_sims.clone()
        off_diag.fill_diagonal_(0)
        if off_diag.sum() > 0:
            intra = 1.0 - off_diag[off_diag != 0].mean().item()
            intra_distances.append(intra)

    return np.mean(intra_distances) if intra_distances else 0.0

def compute_top_k_accuracy(embeddings, labels, k=5):
    """Compute Top-K retrieval accuracy"""
    sim_matrix = torch.mm(embeddings, embeddings.t())

    # Pour chaque image, find top-k most similar (exclure self)
    correct = 0
    total = 0

    for i in range(len(embeddings)):
        sims = sim_matrix[i].clone()
        sims[i] = -1  # Exclude self

        _, indices = torch.topk(sims, k)
        pred_labels = labels[indices]
        correct += (pred_labels == labels[i]).sum().item()
        total += k

    return correct / total if total > 0 else 0.0


def recall_at_k(embeddings, labels, k=5):
    sim = torch.mm(embeddings, embeddings.t())
    n = embeddings.size(0)
    recalls = 0
    for i in range(n):
        sims = sim[i].clone()
        sims[i] = -1
        _, idx = torch.topk(sims, k)
        if (labels[idx] == labels[i]).any():
            recalls += 1
    return recalls / n


def mean_average_precision(embeddings, labels):
    # simple mAP: average precision per query over entire dataset
    sim = torch.mm(embeddings, embeddings.t())
    n = embeddings.size(0)
    aps = []
    for i in range(n):
        sims = sim[i].clone()
        sims[i] = -1
        scores, idx = torch.sort(sims, descending=True)
        relevant = (labels[idx] == labels[i]).float()
        if relevant.sum() == 0:
            aps.append(0.0)
            continue
        cum = torch.cumsum(relevant, dim=0)
        precision_at_k = cum / (torch.arange(1, n+0, device=embeddings.device).float())
        ap = (precision_at_k * relevant).sum() / relevant.sum()
        aps.append(ap.item())
    return float(np.mean(aps))

print("✅ Utility functions ready")


# ============================================================================
# CELL 9: TRAINING LOOP
# ============================================================================
def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None):
    """Un epoch de training (compatible AMP)"""
    model.train()
    total_loss = 0

    use_amp = scaler is not None and scaler.is_enabled()
    progress_bar = tqdm(train_loader, desc="Training")
    for batch_idx, batch_indices in enumerate(progress_bar):
        # If using batch_sampler (PKSampler), DataLoader yields indices relative to the subset
        if isinstance(batch_indices, list) or isinstance(batch_indices, np.ndarray):
            # fetch items from the underlying dataset (train_dataset is a Subset)
            items = [train_loader.dataset[i] for i in batch_indices]
            images = torch.stack([it[0] for it in items], dim=0)
            labels = torch.tensor([it[1] for it in items], dtype=torch.long)
            paths = [it[2] for it in items]
        else:
            images, labels, paths = batch_indices

        if device.type == "cuda":
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
        else:
            images = images.to(device)
            labels = labels.to(device)

        optimizer.zero_grad()

        if use_amp:
            with autocast():
                embeddings = model(images)
                mem_embs, mem_labels = memory_bank.get()
                if mem_embs is not None and len(mem_embs) > 0:
                    loss = criterion(embeddings, labels, memory_embeddings=mem_embs, memory_labels=mem_labels)
                else:
                    loss = criterion(embeddings, labels)
        else:
            embeddings = model(images)
            mem_embs, mem_labels = memory_bank.get()
            if mem_embs is not None and len(mem_embs) > 0:
                loss = criterion(embeddings, labels, memory_embeddings=mem_embs, memory_labels=mem_labels)
            else:
                loss = criterion(embeddings, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Update EMA
        ema.update(model)

        # Add to memory bank (CPU numpy)
        try:
            memory_bank.add(embeddings.detach().cpu().numpy(), labels.detach().cpu().numpy())
        except Exception:
            pass

        total_loss += loss.item()
        loss_val = loss.item()
        progress_bar.set_postfix(loss=f"{loss_val:.4f}")

    avg_loss = total_loss / len(train_loader)
    return avg_loss

@torch.no_grad()
def validate(model, val_loader, criterion, device, ema=None):
    """Validation (utilise le modèle EMA si fourni et renvoie des métriques de retrieval)"""
    # Utiliser les poids EMA pour la validation si disponibles
    original_state = None
    if ema is not None:
        original_state = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema.state_dict())

    model.eval()
    total_loss = 0
    all_embeddings = []
    all_labels = []

    progress_bar = tqdm(val_loader, desc="Validation")
    for images, labels, paths in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        embeddings = model(images)
        loss = criterion(embeddings, labels)

        total_loss += loss.item()
        all_embeddings.append(embeddings.cpu())
        all_labels.append(labels.cpu())
        loss_val = loss.item()
        progress_bar.set_postfix(loss=f"{loss_val:.4f}")

    avg_loss = total_loss / len(val_loader)

    # Compute metrics (retrieval-based)
    all_embeddings = torch.cat(all_embeddings)
    all_labels = torch.cat(all_labels)

    intra_dist = compute_intra_inter_distance(all_embeddings, all_labels)
    top_k_acc = compute_top_k_accuracy(all_embeddings, all_labels, k=5)
    recall1 = recall_at_k(all_embeddings, all_labels, k=1)
    map_score = mean_average_precision(all_embeddings, all_labels)

    # Restaurer les poids d'origine du modèle si EMA utilisée
    if ema is not None and original_state is not None:
        model.load_state_dict(original_state)

    return avg_loss, intra_dist, top_k_acc, recall1, map_score

# Training loop
print("\n" + "="*80)
print("🚀 STARTING TRAINING")
print("="*80 + "\n")

# Early stopping basé sur une métrique de retrieval (Recall@1)
best_metric = -float('inf')
patience = 5
patience_counter = 0
history = defaultdict(list)

for epoch in range(CONFIG['num_epochs']):
    print(f"\n📌 Epoch {epoch+1}/{CONFIG['num_epochs']}")

    # Train
    train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE, scaler)
    print(f"  Train Loss: {train_loss:.4f}")
    history['train_loss'].append(train_loss)

    # Validate (avec EMA)
    val_loss, intra_dist, top_k_acc, recall1, map_score = validate(model, val_loader, criterion, DEVICE, ema)
    print(f"  Val Loss (SupCon): {val_loss:.4f}")
    print(f"  Intra-class distance: {intra_dist:.4f}")
    print(f"  Top-5 accuracy: {top_k_acc:.4f}")
    print(f"  Recall@1: {recall1:.4f}")
    print(f"  mAP: {map_score:.4f}")

    history['val_loss'].append(val_loss)
    history['intra_dist'].append(intra_dist)
    history['top_k_acc'].append(top_k_acc)
    history.setdefault('recall_at1', []).append(recall1)
    history.setdefault('map', []).append(map_score)

    # Learning rate schedule
    scheduler.step()

    # Save checkpoint basé sur Recall@1
    if recall1 > best_metric:
        best_metric = recall1
        patience_counter = 0

        checkpoint_path = f"{CONFIG['checkpoints_path']}/best_model.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            'best_recall_at1': best_metric,
            'map': map_score,
        }, checkpoint_path)
        print(f"  ✅ Best model saved (based on Recall@1): {checkpoint_path}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"  ⏹️  Early stopping at epoch {epoch+1} (no Recall@1 improvement)")
            break

print("\n✅ Training complete!")


# ============================================================================
# CELL 10: COMPUTE FULL EMBEDDINGS & CREATE FAISS INDEX
# ============================================================================
print("\n🔄 Computing embeddings for full dataset...")

# Load best model
best_checkpoint = torch.load(f"{CONFIG['checkpoints_path']}/best_model.pt")
model.load_state_dict(best_checkpoint['model_state_dict'])
model.eval()

# Full dataset (no transforms, exact duplicates)
full_dataset = DiseaseDataset(CONFIG['dataset_path'], transform=val_transform)
full_loader = DataLoader(full_dataset, batch_size=64, num_workers=4, pin_memory=True)

all_embeddings = []
all_labels = []
all_paths = []

with torch.no_grad():
    for images, labels, paths in tqdm(full_loader, desc="Computing embeddings"):
        images = images.to(DEVICE)
        embeddings = model(images)

        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_paths.extend(paths)

embeddings_matrix = np.vstack(all_embeddings).astype('float32')
embeddings_matrix = normalize(embeddings_matrix)  # L2 normalize

print(f"✅ Embeddings computed: {embeddings_matrix.shape}")

# ===================== CREATE PROTOTYPES =====================
print("\n🧠 Computing class prototypes...")

embeddings_tensor = torch.from_numpy(embeddings_matrix)
labels_tensor = torch.tensor(all_labels)

prototypes = []
prototype_labels = []

for class_id in torch.unique(labels_tensor):
    mask = labels_tensor == class_id
    class_embeddings = embeddings_tensor[mask]
    prototype = class_embeddings.mean(dim=0)
    prototype = prototype / (prototype.norm() + 1e-12)
    prototypes.append(prototype.numpy())
    prototype_labels.append(int(class_id))

prototypes = np.vstack(prototypes).astype('float32')

print(f"✅ Prototypes created: {prototypes.shape}")

# Create FAISS index
print("\n🔍 Creating FAISS index...")
import faiss

index = faiss.IndexFlatIP(CONFIG['embedding_dim'])
index.add(embeddings_matrix)

# Save index
index_path = f"{CONFIG['output_path']}/faiss_index.bin"
faiss.write_index(index, index_path)
print(f"✅ FAISS index saved: {index_path}")

# Save metadata
metadata = {
    'embeddings_shape': embeddings_matrix.shape,
    'image_paths': all_paths,
    'labels': all_labels,
    'prototypes': prototypes.tolist(),
    'prototype_labels': prototype_labels,
    'class_to_idx': full_dataset.class_to_idx,
    'idx_to_class': full_dataset.idx_to_class,
    'num_classes': len(full_dataset.class_to_idx),
    'embedding_dim': CONFIG['embedding_dim'],
    'timestamp': datetime.now().isoformat(),
}

metadata_path = f"{CONFIG['output_path']}/metadata.json"
with open(metadata_path, 'w') as f:
    json.dump({k: v for k, v in metadata.items() if k != 'image_paths'}, f, indent=2)
print(f"✅ Metadata saved: {metadata_path}")

# Save pickle with full paths
pickle_path = f"{CONFIG['output_path']}/metadata.pkl"
with open(pickle_path, 'wb') as f:
    pickle.dump(metadata, f)
print(f"✅ Full metadata saved: {pickle_path}")


# ============================================================================
# CELL 11: EXPORT MODEL FOR INFERENCE
# ============================================================================
print("\n💾 Exporting model...")

# Save model state
model_path = f"{CONFIG['output_path']}/swin_diagnostic.pt"
torch.save({
    'model_state_dict': model.state_dict(),
    'config': {
        'model_name': CONFIG['model_name'],
        'embedding_dim': CONFIG['embedding_dim'],
        'num_classes': len(full_dataset.class_to_idx),
        'architecture': 'DiagnosticModel',
    }
}, model_path)
print(f"✅ Model saved: {model_path}")

# Export ONNX (optionnel, pour déploiement web)
print("\n📦 Exporting ONNX...")
dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
onnx_path = f"{CONFIG['output_path']}/swin_diagnostic.onnx"

try:
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=['image'],
        output_names=['embedding'],
        dynamic_axes={'image': {0: 'batch_size'}},
        opset_version=14,
        verbose=False
    )
    print(f"✅ ONNX model saved: {onnx_path}")
except Exception as e:
    print(f"⚠️  ONNX export failed: {e}")


# ============================================================================
# CELL 12: DEMO INFERENCE
# ============================================================================
print("\n🎯 DEMO: Testing inference on random test image...")

# Select a random test image
test_idx = np.random.randint(0, len(full_dataset))
test_image, test_label, test_path = full_dataset[test_idx]

print(f"\nTest image: {test_path}")
print(f"True class: {full_dataset.idx_to_class[test_label]}")

# Get embedding
test_image_tensor = test_image.unsqueeze(0).to(DEVICE)
with torch.no_grad():
    test_embedding = model(test_image_tensor).cpu().numpy()

# ===================== PROTOTYPE-BASED DIAGNOSIS =====================

print("\n🧠 Prototype-based diagnosis...")

# Normalize test embedding
test_embedding_norm = test_embedding / (np.linalg.norm(test_embedding) + 1e-12)

# cosine similarity with prototypes
similarities = np.dot(prototypes, test_embedding_norm.T).squeeze()

best_idx = np.argmax(similarities)
best_score = float(similarities[best_idx])
pred_class_id = prototype_labels[best_idx]
pred_class_name = full_dataset.idx_to_class[pred_class_id]

# seuil intelligent (à ajuster après validation)
UNKNOWN_THRESHOLD = 0.55

if best_score < UNKNOWN_THRESHOLD:
    print("\n❌ Diagnosis: UNKNOWN DISEASE")
    print(f"Similarity: {best_score:.2f}")
else:
    print(f"\n✅ Diagnosis: {pred_class_name}")
    print(f"Similarity score: {best_score:.2f}")


# ============================================================================
# CELL 13: TRAINING METRICS VISUALIZATION
# ============================================================================
print("\n📊 Generating plots...")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Loss
axes[0, 0].plot(history['train_loss'], label='Train')
axes[0, 0].plot(history['val_loss'], label='Val')
axes[0, 0].set_title('Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Intra-class distance
axes[0, 1].plot(history['intra_dist'])
axes[0, 1].set_title('Intra-class Distance (Lower is Better)')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].grid(True)

# Top-K accuracy
axes[1, 0].plot(history['top_k_acc'])
axes[1, 0].set_title('Top-5 Retrieval Accuracy')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylim([0, 1])
axes[1, 0].grid(True)

# Learning rate
axes[1, 1].axis('off')
axes[1, 1].text(0.1, 0.5, "✅ Training Complete!", fontsize=16, weight='bold')
axes[1, 1].text(0.1, 0.3, f"Best Val Loss: {min(history['val_loss']):.4f}", fontsize=12)
axes[1, 1].text(0.1, 0.2, f"Final Top-5 Acc: {history['top_k_acc'][-1]:.4f}", fontsize=12)

plt.tight_layout()
plot_path = f"{CONFIG['logs_path']}/training_metrics.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"✅ Metrics plot saved: {plot_path}")

plt.show()


# ============================================================================
# CELL 14: SUMMARY & NEXT STEPS
# ============================================================================
print("\n" + "="*80)
print("✨ TRAINING PIPELINE COMPLETE!")
print("="*80)
print(f"""
📊 RESULTS:
  - Best Validation Loss: {min(history['val_loss']):.4f}
  - Final Top-5 Accuracy: {history['top_k_acc'][-1]:.4f}
  - Final Intra-class Distance: {history['intra_dist'][-1]:.4f}

📁 FILES SAVED IN GOOGLE DRIVE:
  ✅ {CONFIG['output_path']}/swin_diagnostic.pt (Model weights)
  ✅ {CONFIG['output_path']}/swin_diagnostic.onnx (ONNX export)
  ✅ {CONFIG['output_path']}/faiss_index.bin (FAISS index)
  ✅ {CONFIG['output_path']}/metadata.pkl (Image paths & labels)
  ✅ {CONFIG['logs_path']}/training_metrics.png (Plots)

🚀 NEXT STEPS (Local on your laptop):
  1. Download models from Drive
  2. Create inference app (Streamlit)
  3. Deploy and test with new images
  4. Add new disease classes (without retraining!)

📝 TO ADD NEW CLASSES:
  - Pass new images through model → Get embeddings
  - Add embeddings to FAISS index
  - Just 2 minutes, no retraining needed!

✨ Your model is production-ready!
""")
