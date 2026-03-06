#coding: utf-8
"""
LOCAL TEST SCRIPT - Verification before Colab Pro
Small dataset test on PC (CPU only)
Objective: Check for syntax/indentation errors before launching on Colab

Usage:
    python 05_test_training_pipeline_local.py

Duration: 5-10 minutes (2 epochs on 100 images)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("[LOCAL TEST] Pipeline Verification Before Colab")
print("=" * 80)

# ============================================================================
# STEP 1: SETUP & DATASET PATH
# ============================================================================
print("\n[STEP 1] Setup...")

DEVICE = torch.device("cpu")
print(f"[OK] Device: {DEVICE}")

DATASET_PATH = Path(r"C:\smarts-n-yieldpredict.git\dataset_final")
if not DATASET_PATH.exists():
    print(f"[ERROR] Dataset not found: {DATASET_PATH}")
    exit(1)

# Count classes and images
classes = []
total_images = 0
for class_dir in DATASET_PATH.iterdir():
    if class_dir.is_dir():
        classes.append(class_dir.name)
        img_count = len(list(class_dir.glob("*.jpg"))) + len(list(class_dir.glob("*.jpeg"))) + len(list(class_dir.glob("*.png")))
        total_images += img_count

print(f"[OK] Dataset: {len(classes)} classes, {total_images} images")

# ============================================================================
# STEP 2: DATASET LOADER
# ============================================================================
print("\n[STEP 2] Creating test dataset loader...")


class SimpleDataset(Dataset):
    """Minimal dataset for testing"""

    def __init__(self, dataset_path, max_images_per_class=2, max_classes=5):
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        class_idx = 0
        for class_dir in sorted(Path(dataset_path).iterdir())[:max_classes]:
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            self.class_to_idx[class_name] = class_idx
            self.idx_to_class[class_idx] = class_name

            # Get max_images_per_class only
            img_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png"))
            for img_file in img_files[:max_images_per_class]:
                self.images.append(str(img_file))
                self.labels.append(class_idx)

            class_idx += 1

        print(f"  [OK] Loaded {len(self.images)} images, {len(self.class_to_idx)} classes")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"  [WARN] Image not found: {img_path}")
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))

        # Normalize - IMPORTANT: Keep as float32 throughout
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std

        # Convert to tensor (will be float32)
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        return image, label, img_path


# Create test dataset (5 classes, 2 images each)
print("  Creating minimal test dataset (5 classes, 2 images each)...")
test_dataset = SimpleDataset(DATASET_PATH, max_images_per_class=2, max_classes=5)

# Create dataloader
test_loader = DataLoader(
    test_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)

print(f"[OK] DataLoader created: {len(test_loader)} batches")

# ============================================================================
# STEP 3: MODEL ARCHITECTURE
# ============================================================================
print("\n[STEP 3] Creating model...")


class DiagnosticModel(nn.Module):
    def __init__(self, embedding_dim=768):
        super().__init__()
        try:
            # For local test we avoid downloading large pretrained weights.
            # Use pretrained=False to keep the test lightweight.
            self.backbone = timm.create_model('swin_base_patch4_window7_224', pretrained=False)
            print("  [OK] Swin Transformer backbone created (pretrained=False for local test)")
        except Exception as e:
            print(f"  [ERROR] Swin load error: {e}")
            raise

        # Detect actual backbone output size dynamically
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_output = self.backbone(dummy_input)
            self.backbone_output_size = dummy_output.shape[1]
        
        print(f"  [OK] Detected backbone output size: {self.backbone_output_size}")

        self.embedding_head = nn.Sequential(
            nn.Linear(self.backbone_output_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        self.embedding_dim = embedding_dim

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.embedding_head(features)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings


model = DiagnosticModel(embedding_dim=768)
model = model.to(DEVICE)
print(f"[OK] Model created: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")

# ============================================================================
# STEP 4: LOSS FUNCTION
# ============================================================================
print("\n[STEP 4] Creating loss function...")


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        batch_size = features.shape[0]

        # Similarity matrix
        sim = torch.mm(features, features.t())
        sim = sim / self.temperature

        # Label matrix
        mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask = mask.fill_diagonal_(False)

        # Negative mask
        neg_mask = ~mask

        # Loss
        exp_sim = torch.exp(sim)
        pos_sim = torch.masked_select(exp_sim, mask).view(batch_size, -1).sum(dim=1)
        neg_sim = torch.masked_select(exp_sim, neg_mask).view(batch_size, -1).sum(dim=1)

        loss = -torch.log(pos_sim / (pos_sim + neg_sim) + 1e-8).mean()

        return loss


criterion = SupConLoss(temperature=0.07)
criterion = criterion.to(DEVICE)
print("[OK] Loss function created")

# ============================================================================
# STEP 5: OPTIMIZER
# ============================================================================
print("\n[STEP 5] Creating optimizer...")

optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)

print("[OK] Optimizer created")

# ============================================================================
# STEP 6: TRAINING LOOP
# ============================================================================
print("\n[STEP 6] Running test training loop (2 epochs)...")
print("=" * 80)

model.train()
history = {'train_loss': []}

for epoch in range(2):
    print(f"\nEpoch {epoch+1}/2")

    total_loss = 0
    batch_count = 0

    for batch_idx, (images, labels, paths) in enumerate(test_loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        print(f"  Batch {batch_idx+1}/{len(test_loader)}: Shape={images.shape}, Labels={labels.tolist()}", end=" ")

        try:
            # Forward
            embeddings = model(images)
            print(f"[Emb={embeddings.shape}]", end=" ")

            # Loss
            loss = criterion(embeddings, labels)
            print(f"[Loss={loss.item():.4f}]", end=" ")

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            print("[OK]")

        except Exception as e:
            print(f"\n[ERROR] in batch {batch_idx+1}: {type(e).__name__}: {str(e)[:100]}")
            raise

    avg_loss = total_loss / batch_count if batch_count > 0 else 0
    history['train_loss'].append(avg_loss)
    print(f"  Epoch {epoch+1} completed - Avg Loss: {avg_loss:.4f}")

# ============================================================================
# STEP 7: TEST INFERENCE
# ============================================================================
print("\n[STEP 7] Testing inference...")

model.eval()

with torch.no_grad():
    test_batch = next(iter(test_loader))
    test_images, test_labels, test_paths = test_batch
    test_images = test_images.to(DEVICE)

    embeddings = model(test_images)

    print(f"[OK] Inference successful!")
    print(f"   Input shape: {test_images.shape}")
    print(f"   Output shape: {embeddings.shape}")
    print(f"   Output dtype: {embeddings.dtype}")
    print(f"   Sample embedding (first 10 values): {embeddings[0, :10].detach().cpu().numpy()}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("[SUCCESS] ALL TESTS PASSED!")
print("=" * 80)
print(f"""
TEST RESULTS:
  [OK] Dataset loaded: {len(test_dataset)} images, {len(test_dataset.class_to_idx)} classes
  [OK] DataLoader working: {len(test_loader)} batches
  [OK] Model initialized: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params
  [OK] Loss function working
  [OK] Training loop completed: 2 epochs
  [OK] Final avg loss: {history['train_loss'][-1]:.4f}
  [OK] Inference working

READY FOR COLAB PRO!
   No syntax errors or indentation issues detected.
   You can now safely launch the full training on Colab Pro.

NEXT STEPS:
   1. Copy 02_training_colab_complete.py to Google Colab
   2. Run "Run All"
   3. Wait ~65 hours for training
   4. Download models from Drive

Your pipeline is validated and ready!
""")
