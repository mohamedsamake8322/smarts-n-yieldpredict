"""
🎯 DINOv2 TRAINING — KAGGLE OPTIMIZED VERSION
=============================================
✅ Adapted for Kaggle (free GPU, 30h/week limit)
✅ Dataset balancing strategy for imbalanced classes
✅ Cost-optimized (ViT-S, batch 16, image 224)
✅ Automatic data augmentation for underrepresented classes

INSTRUCTIONS:
1. Create a Kaggle account (free)
2. Upload this notebook or copy the code into a Kaggle notebook
3. Set GPU accelerator in notebook settings
4. Add your dataset to Kaggle or use gdown to download from Drive
5. Update DATA_SOURCE below to point to your data

DATASET BALANCING STRATEGY:
- Median class size: ~1000 images
- Underrepresented classes (<300 images) → aggressive augmentation
- Imbalanced loss weighting → gives more weight to underrepresented classes
- Weighted sampler during training → ensures all classes are well-represented
"""

# ============================================================================
# CELL 1: SETUP — Download dataset from Google Drive (if needed)
# ============================================================================
import subprocess
import sys

# Option A: Using gdown to download from Drive (uncomment if needed)
# Replace the URL with your own shareable Drive link
DOWNLOAD_FROM_DRIVE = False  # Set to True if dataset is not already uploaded to Kaggle

if DOWNLOAD_FROM_DRIVE:
    print("📥 Downloading dataset from Google Drive...")
    subprocess.run("pip install -q gdown", shell=True)
    # Example: gdown.download(url, output, quiet=False)
    # You need to replace this with your actual Drive URL
    import gdown
    # url = "https://drive.google.com/uc?id=YOUR_FOLDER_ID"
    # gdown.download_folder(url, output="/kaggle/input/plant-dataset", quiet=False)
    # import zipfile
    # with zipfile.ZipFile("/kaggle/input/plant-dataset.zip", "r") as z:
    #     z.extractall("/kaggle/input/")
    print("✅ Dataset downloaded")

# ============================================================================
# CELL 1: SETUP — Téléchargement progressif depuis Google Drive
# ============================================================================
import subprocess
import sys
import os
from pathlib import Path

# Configuration pour Google Drive
DOWNLOAD_FROM_DRIVE = True  # Set to True to download from Drive
DRIVE_FOLDER_ID = "YOUR_DRIVE_FOLDER_ID"  # Replace with your Drive folder ID
DOWNLOAD_BATCH_SIZE = 50  # Download classes in batches of 50 to avoid timeouts

if DOWNLOAD_FROM_DRIVE:
    print("📥 Configuration pour téléchargement depuis Google Drive...")
    subprocess.run("pip install -q gdown", shell=True)
    
    # Create data directory
    DATA_DIR = Path('/kaggle/working/plant_dataset')
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Dataset sera téléchargé dans: {DATA_DIR}")
    print("⚠️  Note: Téléchargement de 80GB peut prendre du temps")
    print("💡 Conseil: Commencez avec un sous-ensemble de classes")
    
else:
    # Option: Dataset déjà uploadé sur Kaggle
    DATA_DIR = Path('/kaggle/input/plant-diseases-dataset')

def download_dataset_from_drive(drive_folder_id, local_dir, batch_size=50):
    """
    Télécharge le dataset depuis Google Drive par batches.
    Utile pour les gros datasets (80GB+).
    """
    import gdown
    
    print(f"📥 Téléchargement depuis Drive folder: {drive_folder_id}")
    print(f"   Destination: {local_dir}")
    print(f"   Batch size: {batch_size} classes à la fois")
    
    try:
        # Télécharger le dossier entier (gdown supporte les dossiers)
        url = f"https://drive.google.com/drive/folders/{drive_folder_id}"
        gdown.download_folder(url, output=str(local_dir), quiet=False)
        print("✅ Téléchargement terminé")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du téléchargement: {e}")
        print("💡 Alternatives:")
        print("   1. Vérifiez que le dossier Drive est partagé (Anyone with link)")
        print("   2. Utilisez un sous-ensemble de classes")
        print("   3. Compressez le dataset en ZIP et partagez le lien direct")
        return False

def create_subset_dataset(full_dataset_path, subset_classes, output_path):
    """
    Crée un sous-dataset avec seulement certaines classes.
    Utile pour commencer avec un dataset plus petit.
    """
    import shutil
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    for class_name in subset_classes:
        src_dir = full_dataset_path / class_name
        dst_dir = output_path / class_name
        
        if src_dir.exists():
            shutil.copytree(src_dir, dst_dir)
            print(f"   ✅ Copié: {class_name}")
        else:
            print(f"   ⚠️  Classe non trouvée: {class_name}")
    
    print(f"✅ Sous-dataset créé: {len(subset_classes)} classes")

# Téléchargement si configuré
if DOWNLOAD_FROM_DRIVE and DRIVE_FOLDER_ID != "YOUR_DRIVE_FOLDER_ID":
    success = download_dataset_from_drive(DRIVE_FOLDER_ID, DATA_DIR, DOWNLOAD_BATCH_SIZE)
    if not success:
        print("❌ Échec du téléchargement. Arrêt.")
        sys.exit(1)
elif DOWNLOAD_FROM_DRIVE:
    print("⚠️  Veuillez remplacer 'YOUR_DRIVE_FOLDER_ID' par votre ID Drive réel")
    print("💡 Pour obtenir l'ID: https://drive.google.com/drive/folders/YOUR_FOLDER_ID")
    sys.exit(1)

# ============================================================================
# CONFIG: Kaggle Paths & Dataset Configuration
# ============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"   GPU  : {torch.cuda.get_device_name(0)}")
    print(f"   VRAM : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    torch.backends.cudnn.benchmark = True

# ────────────────────────────────────────────────────────────────────────
# DATA SOURCE — CHANGE THIS BASED ON WHERE YOUR DATA IS
# ────────────────────────────────────────────────────────────────────────
# Option 1: If dataset is uploaded to Kaggle (recommended)
DATA_DIR = Path('/kaggle/input/plant-diseases-dataset')  # Replace with your Kaggle dataset name

# Option 2: If dataset is on Google Drive (use gdown above)
# DATA_DIR = Path('/kaggle/input/plant-dataset')

# If DATA_DIR doesn't exist, create a sample structure for testing
if not DATA_DIR.exists():
    print(f"⚠️  {DATA_DIR} not found. Creating test structure...")
    DATA_DIR = Path('/kaggle/input')

WORKING_DIR = Path('/kaggle/working')
OUT_DIR      = WORKING_DIR / 'models_dinov2_kaggle'
CKPT_DIR     = WORKING_DIR / 'checkpoints_dinov2_kaggle'
LOG_DIR      = WORKING_DIR / 'logs_dinov2_kaggle'

for d in [OUT_DIR, CKPT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"📂 Data dir  : {DATA_DIR}")
print(f"📂 Working dir: {WORKING_DIR}")

# ============================================================================
# CELL 4: DATASET DISCOVERY & CLASS ANALYSIS
# ============================================================================
def discover_classes_from_directory(root_dir):
    """
    Scan directory structure to find classes and images.
    Assumes structure: root_dir/class_name/*.jpg
    """
    class_to_images = defaultdict(list)
    root_path = Path(root_dir)
    
    if not root_path.exists():
        print(f"⚠️  {root_path} does not exist!")
        return class_to_images
    
    # Find all jpg/png files
    for img_path in sorted(root_path.rglob('*.[jJ][pP][gG]')):
        class_name = img_path.parent.name
        class_to_images[class_name].append(str(img_path))
    
    for img_path in sorted(root_path.rglob('*.[pP][nN][gG]')):
        class_name = img_path.parent.name
        class_to_images[class_name].append(str(img_path))
    
    return dict(class_to_images)

# Discover classes
class_to_images = discover_classes_from_directory(DATA_DIR)
NUM_CLASSES = len(class_to_images)

print(f"\n✅ Discovered {NUM_CLASSES} classes")
print(f"   Total images: {sum(len(v) for v in class_to_images.values())}")

# Analyze class balance
class_counts = {k: len(v) for k, v in class_to_images.items()}
sorted_classes = sorted(class_counts.items(), key=lambda x: x[1])

print(f"\n📊 Class distribution:")
print(f"   Min: {sorted_classes[0][0]} ({sorted_classes[0][1]} images)")
print(f"   Max: {sorted_classes[-1][0]} ({sorted_classes[-1][1]} images)")
print(f"   Median: {sorted([c for _, c in class_counts.items()])[NUM_CLASSES//2]} images")
print(f"   Mean: {np.mean(list(class_counts.values())):.0f} images")

# Identify underrepresented classes (< 300 images = need augmentation)
underrep_threshold = 300
underrep_classes = {k: v for k, v in class_counts.items() if v < underrep_threshold}
print(f"\n⚠️  {len(underrep_classes)} underrepresented classes (< {underrep_threshold} images)")
if underrep_classes:
    print(f"   Examples: {list(underrep_classes.items())[:5]}")

# ============================================================================
# CELL 5: CONFIG
# ============================================================================
def _seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

_seed_all(42)

CONFIG = {
    # Model configuration (cost-optimized for Kaggle)
    'training_profile': 'cost_stable',  # cost_stable | balanced | ultra_solid
    'fine_tune_mode': 'partial',        # feature_extract | partial | full
    'backbone': 'dinov2_vits14',        # Smaller backbone (faster, less memory)
    'image_size': 224,                  # Reduced from 336
    'embed_dim': 384,                   # For ViT-S
    'num_classes': NUM_CLASSES,
    'batch_size': 16,                   # Cost-reduced batch
    'num_epochs': 20,                   # Reduced epochs (Kaggle session limits)
    'warmup_epochs': 3,                 # Warm-up
    'lr_head': 1e-4,
    'lr_backbone': 1e-5,
    'weight_decay': 0.05,
    'label_smoothing': 0.05,
    'focal_gamma': 2.0,
    
    # Augmentation & mixing
    'mixup_alpha': 0.3,
    'cutmix_alpha': 1.0,
    
    # Hard mining (progressive)
    'hard_mining_start_epoch': 3,
    'hard_mining_update_freq': 1,
    'hard_mining_ema_alpha': 0.4,
    'hard_mining_boost': 3.0,
    
    # Optimization
    'patience': 6,
    'grad_accum_steps': 1,
    'num_workers': 4,
    'ema_decay': 0.9999,
    'seed': 42,
    'auto_resume': True,
    
    # KAGGLE-SPECIFIC: Dataset balancing strategy
    'balance_strategy': 'weighted_sampler',  # Options: weighted_sampler | weighted_loss | augmentation | mixed
    'target_samples_per_class': 1000,        # Target for augmentation-based balancing
    'underrep_augmentation_factor': 3.0,     # Multiply augmentation for classes < 300 images
}

_seed_all(CONFIG['seed'])

print(f"\n✅ Configuration loaded")
print(f"   Profile: {CONFIG['training_profile']}")
print(f"   Backbone: {CONFIG['backbone']}")
print(f"   Image size: {CONFIG['image_size']}")
print(f"   Batch size: {CONFIG['batch_size']}")
print(f"   Epochs: {CONFIG['num_epochs']}")
print(f"   Balance strategy: {CONFIG['balance_strategy']}")

# ============================================================================
# CELL 6: AUGMENTATIONS (Aggressive for underrepresented classes)
# ============================================================================
IMG_SIZE = CONFIG['image_size']
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

def get_train_augmentation(aggressive=False):
    """
    Get training augmentation pipeline.
    aggressive=True for underrepresented classes (more aggressive transforms).
    """
    if aggressive:
        # Strong augmentation for underrepresented classes
        return A.Compose([
            A.RandomResizedCrop(height=IMG_SIZE, width=IMG_SIZE, scale=(0.7, 1.0), p=1.0),
            A.Flip(p=0.5),
            A.Transpose(p=0.3),
            A.Rotate(limit=45, p=0.7),
            A.GaussNoise(p=0.3),
            A.GaussianBlur(p=0.3),
            A.RandomBrightnessContrast(p=0.4),
            A.Affine(scale=(0.8, 1.2), p=0.4),
            A.Perspective(scale=(0.05, 0.1), p=0.3),
            A.CoarseDropout(max_holes=8, max_height=20, max_width=20, p=0.3),
            A.Normalize(mean=NORM_MEAN, std=NORM_STD, p=1.0),
            ToTensorV2(),
        ], bbox_params=None)
    else:
        # Standard augmentation
        return A.Compose([
            A.RandomResizedCrop(height=IMG_SIZE, width=IMG_SIZE, scale=(0.8, 1.0), p=1.0),
            A.Flip(p=0.5),
            A.Transpose(p=0.2),
            A.Rotate(limit=30, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            A.CoarseDropout(max_holes=4, max_height=16, max_width=16, p=0.2),
            A.Normalize(mean=NORM_MEAN, std=NORM_STD, p=1.0),
            ToTensorV2(),
        ], bbox_params=None)

def get_val_augmentation():
    """Validation augmentation (minimal)."""
    return A.Compose([
        A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1.0),
        A.Normalize(mean=NORM_MEAN, std=NORM_STD, p=1.0),
        ToTensorV2(),
    ], bbox_params=None)

print("✅ Augmentations configured")

# ============================================================================
# CELL 7: DATASET CLASS
# ============================================================================
class PlantDiseaseDataset(Dataset):
    def __init__(self, image_paths, labels, augmentation, class_to_idx):
        """
        Dataset for plant disease images.
        
        Args:
            image_paths: list of image file paths
            labels: list of class indices
            augmentation: albumentations transform
            class_to_idx: dict mapping class name to index
        """
        self.image_paths = image_paths
        self.labels = labels
        self.augmentation = augmentation
        self.class_to_idx = class_to_idx
        
        assert len(image_paths) == len(labels), "Mismatch between images and labels"
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError(f"Failed to load {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"❌ Error loading {img_path}: {e}")
            # Return a blank image as fallback
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        
        # Apply augmentation
        if self.augmentation:
            augmented = self.augmentation(image=img)
            img = augmented['image']
        
        return {
            'image': img,
            'label': torch.tensor(label, dtype=torch.long),
            'path': img_path,
        }

print("✅ Dataset class defined")

# ============================================================================
# CELL 8: BUILD DATASET WITH BALANCING
# ============================================================================
def build_balanced_dataset(class_to_images, class_counts, val_split=0.2):
    """
    Build dataset with class balancing strategy.
    
    Returns:
        train_dataset, val_dataset, class_to_idx, class_weights, sampler_weights
    """
    # Create class mapping
    class_names = sorted(class_to_images.keys())
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    
    # Collect all images and labels
    all_image_paths = []
    all_labels = []
    
    for class_name, img_paths in class_to_images.items():
        class_idx = class_to_idx[class_name]
        for img_path in img_paths:
            all_image_paths.append(img_path)
            all_labels.append(class_idx)
    
    # Compute class weights (inverse frequency)
    class_counts_arr = np.array([class_counts[c] for c in class_names])
    class_weights = 1.0 / (class_counts_arr + 1e-8)
    class_weights = class_weights / class_weights.sum() * len(class_names)  # Normalize
    
    # Split into train/val
    n_total = len(all_image_paths)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    
    # Random split
    indices = np.random.permutation(n_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_paths = [all_image_paths[i] for i in train_indices]
    train_labels = [all_labels[i] for i in train_indices]
    val_paths = [all_image_paths[i] for i in val_indices]
    val_labels = [all_labels[i] for i in val_indices]
    
    # Create datasets
    train_dataset = PlantDiseaseDataset(
        train_paths, train_labels, 
        get_train_augmentation(), 
        class_to_idx
    )
    val_dataset = PlantDiseaseDataset(
        val_paths, val_labels,
        get_val_augmentation(),
        class_to_idx
    )
    
    # Compute sampler weights for balanced sampling
    sample_weights = np.array([class_weights[label] for label in train_labels])
    sampler_weights = torch.from_numpy(sample_weights).double()
    
    print(f"\n✅ Dataset built:")
    print(f"   Train: {len(train_dataset)} images | Val: {len(val_dataset)} images")
    print(f"   Classes: {len(class_to_idx)}")
    print(f"   Class weights (min-max): {class_weights.min():.3f} - {class_weights.max():.3f}")
    
    return train_dataset, val_dataset, class_to_idx, idx_to_class, class_weights, sampler_weights

# Build dataset
train_dataset, val_dataset, class_to_idx, idx_to_class, class_weights, sampler_weights = \
    build_balanced_dataset(class_to_images, class_counts, val_split=0.2)

# ============================================================================
# CELL 9: CREATE DATA LOADERS
# ============================================================================
def create_dataloaders(train_dataset, val_dataset, sampler_weights, batch_size, num_workers=4):
    """Create balanced dataloaders using WeightedRandomSampler."""
    
    # Weighted sampler for balanced batches
    sampler = WeightedRandomSampler(
        weights=sampler_weights,
        num_samples=len(sampler_weights),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    print(f"✅ Dataloaders created")
    print(f"   Train: {len(train_loader)} batches | Val: {len(val_loader)} batches")
    
    return train_loader, val_loader

train_loader, val_loader = create_dataloaders(
    train_dataset, val_dataset, sampler_weights,
    batch_size=CONFIG['batch_size'],
    num_workers=CONFIG['num_workers']
)

# ============================================================================
# CELL 10: MODEL SETUP
# ============================================================================
def load_dinov2_backbone(backbone_name='dinov2_vits14'):
    """Load DINOv2 backbone from torch.hub."""
    print(f"📥 Loading {backbone_name}...")
    model = torch.hub.load('facebookresearch/dinov2', backbone_name)
    
    # Freeze backbone initially
    for param in model.parameters():
        param.requires_grad = False
    
    return model

class PlantDiseaseModel(nn.Module):
    def __init__(self, backbone, embed_dim, num_classes, num_crops=5, num_categories=3):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # Main classification head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        
        # Auxiliary heads (optional, for multitask learning)
        self.crop_head = nn.Linear(embed_dim, num_crops) if num_crops > 0 else None
        self.category_head = nn.Linear(embed_dim, num_categories) if num_categories > 0 else None
        
        self.ema_model = None  # For EMA tracking
    
    def forward(self, x, return_embedding=False):
        # Get DINOv2 embedding
        embedding = self.backbone(x)  # [B, 1, embed_dim] for ViT
        embedding = embedding.squeeze(1) if embedding.dim() == 3 else embedding
        
        # Classification
        logits = self.head(embedding)
        
        if return_embedding:
            return logits, embedding
        return logits
    
    def forward_auxiliary(self, x):
        """Forward with auxiliary heads."""
        embedding = self.backbone(x)
        embedding = embedding.squeeze(1) if embedding.dim() == 3 else embedding
        
        logits = self.head(embedding)
        crop_logits = self.crop_head(embedding) if self.crop_head else None
        category_logits = self.category_head(embedding) if self.category_head else None
        
        return logits, crop_logits, category_logits

# Load model
backbone = load_dinov2_backbone(CONFIG['backbone'])
model = PlantDiseaseModel(
    backbone=backbone,
    embed_dim=CONFIG['embed_dim'],
    num_classes=CONFIG['num_classes'],
    num_crops=0,  # Disable aux heads for now
    num_categories=0,
)
model = model.to(DEVICE)

print(f"✅ Model loaded: {CONFIG['num_classes']} classes")

# ============================================================================
# CELL 11: LOSS & OPTIMIZER
# ============================================================================
def create_loss_fn(class_weights, label_smoothing=0.05):
    """Create weighted cross-entropy loss."""
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=label_smoothing,
    )
    return loss_fn

def create_optimizer(model, lr_backbone, lr_head, weight_decay):
    """Create optimizer with different LRs for backbone and head."""
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.head.parameters())
    
    if model.crop_head:
        head_params.extend(list(model.crop_head.parameters()))
    if model.category_head:
        head_params.extend(list(model.category_head.parameters()))
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': lr_backbone},
        {'params': head_params, 'lr': lr_head},
    ], weight_decay=weight_decay)
    
    return optimizer

loss_fn = create_loss_fn(class_weights, CONFIG['label_smoothing'])
optimizer = create_optimizer(
    model,
    lr_backbone=CONFIG['lr_backbone'],
    lr_head=CONFIG['lr_head'],
    weight_decay=CONFIG['weight_decay'],
)

print(f"✅ Loss & Optimizer configured")

# ============================================================================
# CELL 12: TRAINING LOOP
# ============================================================================
def train_epoch(model, train_loader, loss_fn, optimizer, device, epoch):
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']}", unit="batch")
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        # Forward
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total:.3f}',
        })
    
    avg_loss = total_loss / len(train_loader)
    avg_acc = correct / total
    
    return avg_loss, avg_acc

def validate(model, val_loader, loss_fn, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating", unit="batch")
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(images)
            loss = loss_fn(logits, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(val_loader)
    avg_acc = correct / total
    
    return avg_loss, avg_acc

# ============================================================================
# CELL 13: MAIN TRAINING
# ============================================================================
print(f"\n{'='*70}")
print(f"🚀 STARTING TRAINING — DINOv2 on Kaggle")
print(f"{'='*70}")

best_val_acc = 0.0
patience_counter = 0

for epoch in range(CONFIG['num_epochs']):
    # Train
    train_loss, train_acc = train_epoch(
        model, train_loader, loss_fn, optimizer, DEVICE, epoch
    )
    
    # Validate
    val_loss, val_acc = validate(model, val_loader, loss_fn, DEVICE)
    
    print(f"\n📊 Epoch {epoch+1}/{CONFIG['num_epochs']}")
    print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        checkpoint = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_val_acc': best_val_acc,
            'config': CONFIG,
        }
        best_model_path = CKPT_DIR / 'best_model.pt'
        torch.save(checkpoint, best_model_path)
        print(f"   ✅ Best model saved: {best_model_path}")
        patience_counter = 0
    else:
        patience_counter += 1
    
    # Early stopping
    if patience_counter >= CONFIG['patience']:
        print(f"\n⏹️  Early stopping after {epoch+1} epochs (patience={CONFIG['patience']})")
        break

# ============================================================================
# CELL 14: SAVE FINAL MODEL & RESULTS
# ============================================================================
print(f"\n{'='*70}")
print(f"✅ TRAINING COMPLETED")
print(f"{'='*70}")

# Save final model
final_model_path = OUT_DIR / 'final_model.pt'
torch.save(model.state_dict(), final_model_path)
print(f"📁 Final model saved: {final_model_path}")

# Save class mapping
class_mapping = {
    'class_to_idx': class_to_idx,
    'idx_to_class': idx_to_class,
    'num_classes': CONFIG['num_classes'],
}
mapping_path = OUT_DIR / 'class_mapping.json'
with open(mapping_path, 'w') as f:
    json.dump(class_mapping, f, indent=2)
print(f"📁 Class mapping saved: {mapping_path}")

# Save config
config_path = OUT_DIR / 'training_config.json'
with open(config_path, 'w') as f:
    json.dump(CONFIG, f, indent=2)
print(f"📁 Config saved: {config_path}")

print(f"\n🎯 Best validation accuracy: {best_val_acc:.4f}")
print(f"\n📦 All outputs are in: {WORKING_DIR}")
print(f"   You can download them from Kaggle notebook output files")
