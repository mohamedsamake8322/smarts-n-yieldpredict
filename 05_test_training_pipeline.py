"""
LOCAL TEST SCRIPT - Pipeline Verification Before Colab

Charge le VRAI dataset depuis C:\smarts-n-yieldpredict.git\dataset_final
Teste la pipeline complete sans GPU requis
Utilise un petit subset pour verification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import timm
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import os
import sys
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("[LOCAL TEST] Pipeline Verification Before Colab")
print("="*80)

# ========================== STEP 1: SETUP ==========================
print("\n[STEP 1] Setup...")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[OK] Device: {DEVICE}")

DATASET_PATH = r"C:\smarts-n-yieldpredict.git\dataset_final"
if not os.path.exists(DATASET_PATH):
    print(f"[ERROR] Dataset not found at {DATASET_PATH}")
    sys.exit(1)

# Count classes
classes = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
print(f"[OK] Dataset: {len(classes)} classes found")

# Count images
total_images = 0
for class_dir in classes:
    class_path = os.path.join(DATASET_PATH, class_dir)
    images = len([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    total_images += images

print(f"[OK] Total images: {total_images}")

# ========================== STEP 2: LOAD REAL DATASET (SUBSET) ==========================
print("\n[STEP 2] Creating test dataset loader from REAL data...")

class RealDiseaseDataset(Dataset):
    """Load real dataset"""
    def __init__(self, dataset_path, max_classes=5, max_samples_per_class=10):
        self.dataset_path = Path(dataset_path)
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        class_idx = 0
        for class_dir in sorted(self.dataset_path.iterdir())[:max_classes]:
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            self.class_to_idx[class_name] = class_idx
            self.idx_to_class[class_idx] = class_name

            # Load only max_samples_per_class images
            img_count = 0
            for img_file in class_dir.glob('*'):
                if img_file.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}:
                    self.images.append(str(img_file))
                    self.labels.append(class_idx)
                    img_count += 1
                    if img_count >= max_samples_per_class:
                        break

            class_idx += 1

        print(f"  [OK] Loaded {len(self.images)} images, {len(self.class_to_idx)} classes")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        image = cv2.imread(img_path)
        if image is None:
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32) / 255.0  # Make sure it's float32!

        # Normalize
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std

        image = torch.from_numpy(image).permute(2, 0, 1).float()  # Explicit float()

        return image, label, img_path

print("  Creating minimal test dataset (5 classes, 10 images each)...")
try:
    dataset = RealDiseaseDataset(DATASET_PATH, max_classes=5, max_samples_per_class=20)
    print(f"[OK] Loaded {len(dataset)} images, {len(dataset.class_to_idx)} classes")
except Exception as e:
    print(f"[ERROR] Failed to load dataset: {e}")
    sys.exit(1)

# Create DataLoader
try:
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    print("[OK] DataLoader created: {} batches".format(len(train_loader)))
except Exception as e:
    print(f"[ERROR] Failed to create DataLoader: {e}")
    sys.exit(1)

# ========================== STEP 3: CREATE MODEL ==========================
print("\n[STEP 3] Creating model...")

class DiagnosticModel(nn.Module):
    def __init__(self, embedding_dim=768, pretrained=False):
        super().__init__()
        try:
            # Use pretrained=False for local testing (faster)
            # Will use pretrained=True on Colab
            self.backbone = timm.create_model('swin_base_patch4_window7_224', pretrained=pretrained, num_classes=0)
            # num_classes=0 removes the classification head, outputs features
            backbone_dim = self.backbone.num_features  # Get actual output dimension
        except Exception as e:
            print(f"[ERROR] Failed to load Swin Transformer: {e}")
            raise

        # Embedding head - adapt to actual backbone output dimension
        self.embedding_head = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.embedding_head(features)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

try:
    model = DiagnosticModel(embedding_dim=768, pretrained=False)  # No pretrained for local test
    model = model.to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[OK] Model created: {num_params:.1f}M parameters")
except Exception as e:
    print(f"[ERROR] Failed to create model: {e}")
    sys.exit(1)

# ========================== STEP 4: CREATE LOSS FUNCTION ==========================
print("\n[STEP 4] Creating loss function...")

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        batch_size = features.shape[0]

        sim = torch.mm(features, features.t()) / self.temperature
        mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask = mask.fill_diagonal_(False)

        neg_mask = ~mask
        exp_sim = torch.exp(sim)

        pos_sim = torch.masked_select(exp_sim, mask).view(batch_size, -1).sum(dim=1)
        neg_sim = torch.masked_select(exp_sim, neg_mask).view(batch_size, -1).sum(dim=1)

        loss = -torch.log(pos_sim / (pos_sim + neg_sim)).mean()
        return loss

try:
    criterion = SupConLoss(temperature=0.07)
    criterion = criterion.to(DEVICE)
    print("[OK] Loss function created")
except Exception as e:
    print(f"[ERROR] Failed to create loss function: {e}")
    sys.exit(1)

# ========================== STEP 5: CREATE OPTIMIZER ==========================
print("\n[STEP 5] Creating optimizer...")

try:
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-7)
    print("[OK] Optimizer created")
except Exception as e:
    print(f"[ERROR] Failed to create optimizer: {e}")
    sys.exit(1)

# ========================== STEP 6: TEST TRAINING LOOP ==========================
print("\n[STEP 6] Running test training loop (2 epochs)...")
print("="*80)

try:
    for epoch in range(2):
        model.train()
        total_loss = 0.0
        batch_count = 0

        for batch_idx, (images, labels, paths) in enumerate(train_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward pass
            embeddings = model(images)

            # Compute loss
            loss = criterion(embeddings, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            print(f"Epoch {epoch+1}/2, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / batch_count
        print(f"  [OK] Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")

    print("="*80)
    print("[OK] Training loop test passed!")

except Exception as e:
    print(f"[ERROR] Training loop failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ========================== STEP 7: TEST INFERENCE ==========================
print("\n[STEP 7] Testing inference...")

try:
    model.eval()
    with torch.no_grad():
        for images, labels, paths in train_loader:
            images = images.to(DEVICE)
            embeddings = model(images)
            print(f"[OK] Inference successful: embeddings shape {embeddings.shape}")
            break

except Exception as e:
    print(f"[ERROR] Inference failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ========================== FINAL SUMMARY ==========================
print("\n" + "="*80)
print("SUCCESS! All tests passed!")
print("="*80)
print(f"""
Summary:
  - Device: {DEVICE}
  - Dataset: {len(dataset)} images, {len(dataset.class_to_idx)} classes
  - Model: {num_params:.1f}M parameters (Swin Transformer)
  - Loss function: Supervised Contrastive Loss
  - Training: 2 epochs completed successfully
  - Inference: Working correctly

You can now safely launch Colab Pro training!

Next steps:
  1. Copy 02_training_colab_complete.py to Google Colab
  2. Run the training (will take ~65 hours)
  3. Download models when done

Let's go!
""")
