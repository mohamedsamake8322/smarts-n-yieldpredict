import os
import json
import pickle
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Sequence, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import timm


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainingConfig:
    dataset_path: str
    output_path: str
    checkpoints_path: str
    logs_path: str

    model_name: str
    embedding_dim: int = 768
    image_size: int = 224

    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4

    temperature: float = 0.07  # SupCon temperature
    # Colab peut freezer avec trop de workers -> 2 par defaut
    num_workers: int = 2
    seed: int = 42

    experiment_name: str = "experiment"

    # Sampling (PKSampler)
    P: int = 8
    K: int = 4

    # Regularisation / imbalance handling
    use_class_weighting: bool = True
    class_weight_power: float = 0.5  # weight ~ 1 / (count ** power)

    # MixUp/CutMix (appliqué uniquement "within-class" pour garder des labels durs)
    mix_prob: float = 0.0
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0

    # Memory / EMA
    memory_bank_capacity: int = 32768
    ema_decay: float = 0.999

    # Monitoring avancé (pour éviter les explosions en O(n^2))
    max_val_eval_samples: int = 2500
    compute_confusion_matrix: bool = True

    # Visualisation "attention" (publication)
    export_attention_viz: bool = False
    attention_viz_samples: int = 12

    # Force ou non les augmentations lourdes (RandomResizedCrop + bruit + dropout fort)
    strong_augmentation: bool = True

    # Reprise: si True et last_checkpoint.pt existe, on reprend a l'epoch suivant
    resume_from_checkpoint: bool = True


def get_default_config(
    model_name: str = "swin_base_patch4_window7_224",
    embedding_dim: int = 768,
    num_epochs: int = 50,
    experiment_name: str = "swin_base",
) -> TrainingConfig:
    """
    Default config.

    - Sur Colab + Drive monté, on utilise par défaut:
        /content/drive/MyDrive/dataset_final
        /content/drive/MyDrive/outputs/<experiment_name>/
    - Sinon, on retombe sur:
        ./dataset_final
        ./outputs/<experiment_name>/

    Dans tous les cas, DATASET_PATH et OUTPUT_ROOT peuvent surcharger.
    """
    # Détection simple de l'environnement Colab + Drive
    default_dataset = "./dataset_final"
    default_outputs = "./outputs"
    if os.path.isdir("/content/drive/MyDrive"):
        default_dataset = "/content/drive/MyDrive/dataset_final"
        default_outputs = "/content/drive/MyDrive/outputs"

    dataset_root = os.environ.get("DATASET_PATH", default_dataset)
    base_output = Path(os.environ.get("OUTPUT_ROOT", default_outputs)) / experiment_name

    return TrainingConfig(
        dataset_path=str(dataset_root),
        output_path=str(base_output / "models"),
        checkpoints_path=str(base_output / "checkpoints"),
        logs_path=str(base_output / "logs"),
        model_name=model_name,
        embedding_dim=embedding_dim,
        num_epochs=num_epochs,
        experiment_name=experiment_name,
    )


class DiseaseDataset(Dataset):
    """
    Dataset pour structure:

    dataset_root/
        train/
            class_A/
            class_B/
        val/
            class_A/
            class_B/
        test/
            ...

    On peut choisir un ou plusieurs splits.
    """

    def __init__(
        self,
        root: str,
        splits: Sequence[str],
        transform=None,
        image_size: int = 224,
    ):
        self.root = Path(root)
        self.splits = splits
        self.transform = transform
        self.image_size = int(image_size)

        self.images: List[str] = []
        self.labels: List[int] = []
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        class_idx = 0
        for split in self.splits:
            split_dir = self.root / split
            if not split_dir.exists():
                # On ignore simplement les splits absents (robustesse)
                continue

            for class_dir in sorted(split_dir.iterdir()):
                if not class_dir.is_dir():
                    continue

                class_name = class_dir.name
                if class_name not in self.class_to_idx:
                    self.class_to_idx[class_name] = class_idx
                    self.idx_to_class[class_idx] = class_name
                    class_idx += 1

                label = self.class_to_idx[class_name]

                for img_file in class_dir.rglob("*"):
                    if img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                        self.images.append(str(img_file))
                        self.labels.append(label)

        if len(self.images) == 0:
            raise RuntimeError(
                f"Aucune image trouvée dans {self.root} pour les splits {self.splits}"
            )

        print(
            f"✅ Dataset loaded from {self.root} (splits={self.splits}): "
            f"{len(self.images)} images, {len(self.class_to_idx)} classes"
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        image = cv2.imread(img_path)
        if image is None:
            # Robustesse: on log et on renvoie un patch noir
            print(f"⚠️  Image not found or unreadable: {img_path}")
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label, img_path


class PKSampler(Sampler):
    """
    P classes × K images par batch.
    On travaille sur les labels du dataset complet (pas Subset).
    """

    def __init__(self, labels: Sequence[int], P: int = 8, K: int = 4):
        self.labels = np.array(labels)
        self.P = P
        self.K = K

        self.class_to_indices: Dict[int, List[int]] = {}
        for idx, l in enumerate(self.labels):
            self.class_to_indices.setdefault(int(l), []).append(idx)

        self.classes = list(self.class_to_indices.keys())
        self.batch_size = self.P * self.K
        self.num_batches = max(1, len(self.labels) // self.batch_size)

    def __iter__(self):
        rng = np.random.default_rng()
        for _ in range(self.num_batches):
            chosen_classes = rng.choice(
                self.classes, size=min(self.P, len(self.classes)), replace=False
            )
            batch_indices: List[int] = []
            for c in chosen_classes:
                indices = self.class_to_indices[c]
                if len(indices) >= self.K:
                    chosen = rng.choice(indices, size=self.K, replace=False)
                else:
                    chosen = rng.choice(indices, size=self.K, replace=True)
                batch_indices.extend(chosen.tolist())
            yield batch_indices

    def __len__(self):
        return self.num_batches


def build_transforms(image_size: int = 224, strong: bool = True):
    """
    Deux régimes d'augmentations:
    - strong=True  : pipeline agressif (pour Swin, metric learning, prod)
    - strong=False : pipeline plus simple (baseline ResNet scientifique)
    """
    if strong:
        train_transform = A.Compose(
            [
                # Albumentations >=2.0: RandomResizedCrop attend size=(H, W)
                A.RandomResizedCrop(
                    size=(image_size, image_size),
                    scale=(0.6, 1.0),
                    ratio=(0.75, 1.33),
                    p=1.0,
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.05),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.12,
                    rotate_limit=15,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.7,
                ),
                A.GaussNoise(p=0.15),
                A.ColorJitter(
                    brightness=0.25,
                    contrast=0.25,
                    saturation=0.25,
                    hue=0.05,
                    p=0.7,
                ),
                A.OneOf(
                    [
                        A.GaussianBlur(p=0.2),
                        A.MotionBlur(p=0.15),
                    ],
                    p=0.3,
                ),
                A.ImageCompression(
                    quality_lower=50, quality_upper=95, p=0.25
                ),
                A.CoarseDropout(
                    max_holes=8, max_height=24, max_width=24, p=0.25
                ),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Baseline plus "propre": peu d'augmentations lourdes
        train_transform = A.Compose(
            [
                A.Resize(image_size, image_size),
                A.RandomResizedCrop(
                    size=(image_size, image_size),
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1),
                    p=0.8,
                ),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.15,
                    hue=0.03,
                    p=0.5,
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )

    val_transform = A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )

    return train_transform, val_transform


class DiagnosticModel(nn.Module):
    """
    Backbone timm + tête d'embedding L2-normalisée.
    """

    def __init__(self, model_name: str, embedding_dim: int = 768, image_size: int = 224):
        super().__init__()

        try:
            self.backbone = timm.create_model(
                model_name,
                pretrained=True,
                features_only=True,
                out_indices=(0, 1, 2, 3),
            )
        except Exception:
            self.backbone = timm.create_model(model_name, pretrained=True)

        dummy = torch.randn(1, 3, image_size, image_size)
        with torch.no_grad():
            feats = self.backbone(dummy)
            if isinstance(feats, (list, tuple)):
                self.feature_dims = [f.shape[1] for f in feats]
            else:
                self.feature_dims = [feats.shape[1]]

        fused_dim = sum(self.feature_dims)
        self.fusion_proj = nn.Sequential(
            nn.Linear(fused_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        if not isinstance(feats, (list, tuple)):
            feats = [feats]

        pooled = []
        for f in feats:
            p = torch.nn.functional.adaptive_avg_pool2d(f, (1, 1)).view(
                x.size(0), -1
            )
            pooled.append(p)

        fused = torch.cat(pooled, dim=1)
        embeddings = self.fusion_proj(fused)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings


class SupConLoss(nn.Module):
    """
    Supervised contrastive / InfoNCE-style loss.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        memory_embeddings: Optional[np.ndarray] = None,
        memory_labels: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        device = features.device
        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        logits_batch = torch.div(
            torch.matmul(features, features.T), self.temperature
        )

        logits_mask = torch.ones_like(mask) - torch.eye(
            batch_size, device=device
        )
        mask = mask * logits_mask

        if memory_embeddings is not None and len(memory_embeddings) > 0:
            if not torch.is_tensor(memory_embeddings):
                memory_embeddings = torch.from_numpy(memory_embeddings).to(
                    device
                )
            if not torch.is_tensor(memory_labels):
                memory_labels = torch.from_numpy(memory_labels).to(device)

            logits_mem = torch.div(
                torch.matmul(features, memory_embeddings.t()), self.temperature
            )

            logits = torch.cat([logits_batch, logits_mem], dim=1)

            mem_mask = torch.eq(
                labels, memory_labels.view(1, -1)
            ).float().to(device)
            combined_mask = torch.cat([mask, mem_mask], dim=1)
        else:
            logits = logits_batch
            combined_mask = mask

        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        if logits.shape[0] == batch_size and logits.shape[1] == batch_size:
            eye = torch.eye(
                logits.size(0), device=device, dtype=logits.dtype
            )
            exp_logits = torch.exp(logits) * (1.0 - eye)
        else:
            exp_logits = torch.exp(logits)

        exp_sum = exp_logits.sum(1, keepdim=True) + 1e-12
        log_prob = logits - torch.log(exp_sum)

        mean_log_prob_pos = (
            combined_mask * log_prob
        ).sum(1) / (combined_mask.sum(1) + 1e-12)

        loss = -mean_log_prob_pos.mean()
        return loss


class MemoryBank:
    """
    Simple FIFO memory bank sur CPU (numpy).
    """

    def __init__(self, capacity: int = 65536, dim: int = 768):
        self.capacity = capacity
        self.dim = dim
        self.ptr = 0
        self.size = 0
        self.embeddings = np.zeros((capacity, dim), dtype="float32")
        self.labels = np.zeros((capacity,), dtype="int32")

    def add(self, embs: np.ndarray, labs: np.ndarray) -> None:
        embs = embs.astype("float32")
        labs = labs.astype("int32")
        n = embs.shape[0]
        if n >= self.capacity:
            embs = embs[-self.capacity :]
            labs = labs[-self.capacity :]
            n = embs.shape[0]

        end = (self.ptr + n) % self.capacity
        if self.ptr + n <= self.capacity:
            self.embeddings[self.ptr : self.ptr + n] = embs
            self.labels[self.ptr : self.ptr + n] = labs
        else:
            first = self.capacity - self.ptr
            self.embeddings[self.ptr :] = embs[:first]
            self.labels[self.ptr :] = labs[:first]
            self.embeddings[:end] = embs[first:]
            self.labels[:end] = labs[first:]

        self.ptr = end
        self.size = min(self.capacity, self.size + n)

    def get(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.embeddings[: self.size].copy(), self.labels[: self.size].copy()


class ModelEMA:
    """
    EMA des poids du modèle.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.ema = {
            k: v.clone().detach() for k, v in model.state_dict().items()
        }
        self.decay = decay

    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, v in msd.items():
            if v.dtype.is_floating_point:
                self.ema[k].mul_(self.decay).add_(
                    v.detach(), alpha=1.0 - self.decay
                )

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.ema


def compute_intra_inter_distance(
    embeddings: torch.Tensor, labels: torch.Tensor
) -> float:
    sim_matrix = torch.mm(embeddings, embeddings.t())

    intra_distances: List[float] = []
    unique_labels = torch.unique(labels)
    for label in unique_labels:
        mask = labels == label
        class_sims = sim_matrix[mask][:, mask]

        off_diag = class_sims.clone()
        off_diag.fill_diagonal_(0)
        if off_diag.sum() > 0:
            intra = 1.0 - off_diag[off_diag != 0].mean().item()
            intra_distances.append(intra)

    return float(np.mean(intra_distances)) if intra_distances else 0.0


def compute_top_k_accuracy(
    embeddings: torch.Tensor, labels: torch.Tensor, k: int = 5
) -> float:
    sim_matrix = torch.mm(embeddings, embeddings.t())
    correct = 0
    total = 0

    for i in range(len(embeddings)):
        sims = sim_matrix[i].clone()
        sims[i] = -1

        _, indices = torch.topk(sims, k)
        pred_labels = labels[indices]
        correct += (pred_labels == labels[i]).sum().item()
        total += k

    return correct / total if total > 0 else 0.0


def recall_at_k(
    embeddings: torch.Tensor, labels: torch.Tensor, k: int = 5
) -> float:
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


def mean_average_precision(
    embeddings: torch.Tensor, labels: torch.Tensor
) -> float:
    sim = torch.mm(embeddings, embeddings.t())
    n = embeddings.size(0)
    aps: List[float] = []
    for i in range(n):
        sims = sim[i].clone()
        sims[i] = -1
        scores, idx = torch.sort(sims, descending=True)
        relevant = (labels[idx] == labels[i]).float()
        if relevant.sum() == 0:
            aps.append(0.0)
            continue
        cum = torch.cumsum(relevant, dim=0)
        precision_at_k = cum / (
            torch.arange(1, n + 1, device=embeddings.device).float()
        )
        ap = (precision_at_k * relevant).sum() / relevant.sum()
        aps.append(ap.item())
    return float(np.mean(aps)) if aps else 0.0


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: SupConLoss,
    optimizer: optim.Optimizer,
    memory_bank: MemoryBank,
    ema: ModelEMA,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
) -> float:
    model.train()
    total_loss = 0.0
    use_amp = scaler is not None and scaler.is_enabled()

    # Avec batch_sampler, DataLoader retourne deja (images, labels, paths)
    progress_bar = tqdm(train_loader, desc="Training")
    for images, labels, paths in progress_bar:
        if device.type == "cuda":
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
        else:
            images = images.to(device)
            labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast():
                embeddings = model(images)
                mem_embs, mem_labels = memory_bank.get()
                if len(mem_embs) > 0:
                    loss = criterion(
                        embeddings,
                        labels,
                        memory_embeddings=mem_embs,
                        memory_labels=mem_labels,
                    )
                else:
                    loss = criterion(embeddings, labels)
        else:
            embeddings = model(images)
            mem_embs, mem_labels = memory_bank.get()
            if len(mem_embs) > 0:
                loss = criterion(
                    embeddings,
                    labels,
                    memory_embeddings=mem_embs,
                    memory_labels=mem_labels,
                )
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

        ema.update(model)

        try:
            memory_bank.add(
                embeddings.detach().cpu().numpy(),
                labels.detach().cpu().numpy(),
            )
        except Exception:
            pass

        loss_val = loss.item()
        total_loss += loss_val
        progress_bar.set_postfix(loss=f"{loss_val:.4f}")

    return total_loss / max(1, len(train_loader))


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: SupConLoss,
    device: torch.device,
    ema: Optional[ModelEMA] = None,
) -> Dict[str, float]:
    original_state = None
    if ema is not None:
        original_state = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema.state_dict())

    model.eval()
    total_loss = 0.0
    all_embeddings: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    progress_bar = tqdm(val_loader, desc="Validation")
    for images, labels, paths in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        embeddings = model(images)
        loss = criterion(embeddings, labels)

        total_loss += loss.item()
        all_embeddings.append(embeddings.cpu())
        all_labels.append(labels.cpu())
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    metrics: Dict[str, float] = {}
    metrics["val_loss"] = total_loss / max(1, len(val_loader))

    # Metriques sur GPU pour accelerer (evite 269s/it sur CPU)
    all_embeddings_t = torch.cat(all_embeddings).to(device)
    all_labels_t = torch.cat(all_labels).to(device)

    metrics["intra_dist"] = compute_intra_inter_distance(
        all_embeddings_t, all_labels_t
    )
    metrics["top5_acc"] = compute_top_k_accuracy(
        all_embeddings_t, all_labels_t, k=5
    )
    metrics["recall_at1"] = recall_at_k(all_embeddings_t, all_labels_t, k=1)
    metrics["map"] = mean_average_precision(all_embeddings_t, all_labels_t)

    # Accuracy computed as Recall@1 (nearest neighbor classification accuracy)
    # We also store correct/total for clear reporting
    total = all_labels_t.size(0)
    if total > 0:
        # Vectorized nearest-neighbor prediction (excluding self-match)
        sim = torch.mm(all_embeddings_t, all_embeddings_t.T)
        sim.fill_diagonal_(-1)
        top1 = torch.topk(sim, 1, dim=1).indices.squeeze(1)
        correct = (all_labels_t[top1] == all_labels_t).sum().item()
        metrics["accuracy"] = correct / total
        metrics["accuracy_correct"] = int(correct)
        metrics["accuracy_total"] = int(total)
    else:
        metrics["accuracy"] = 0.0
        metrics["accuracy_correct"] = 0
        metrics["accuracy_total"] = 0

    if ema is not None and original_state is not None:
        model.load_state_dict(original_state)

    return metrics


def compute_full_embeddings_and_prototypes(
    model: nn.Module,
    dataset: DiseaseDataset,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
) -> Dict[str, Any]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    all_embeddings: List[np.ndarray] = []
    all_labels: List[int] = []
    all_paths: List[str] = []

    model.eval()
    with torch.no_grad():
        for images, labels, paths in tqdm(loader, desc="Computing embeddings"):
            images = images.to(device)
            emb = model(images)
            all_embeddings.append(emb.cpu().numpy())
            all_labels.extend(labels.numpy().tolist())
            all_paths.extend(list(paths))

    embeddings_matrix = np.vstack(all_embeddings).astype("float32")
    embeddings_matrix = normalize(embeddings_matrix)

    print(f"✅ Embeddings computed: {embeddings_matrix.shape}")

    # Prototypes
    print("\n🧠 Computing class prototypes...")
    embeddings_tensor = torch.from_numpy(embeddings_matrix)
    labels_tensor = torch.tensor(all_labels)

    prototypes_list: List[np.ndarray] = []
    prototype_labels: List[int] = []

    for class_id in torch.unique(labels_tensor):
        mask = labels_tensor == class_id
        class_embeddings = embeddings_tensor[mask]
        prototype = class_embeddings.mean(dim=0)
        prototype = prototype / (prototype.norm() + 1e-12)
        prototypes_list.append(prototype.numpy())
        prototype_labels.append(int(class_id))

    prototypes = np.vstack(prototypes_list).astype("float32")
    print(f"✅ Prototypes created: {prototypes.shape}")

    return {
        "embeddings_matrix": embeddings_matrix,
        "labels": all_labels,
        "paths": all_paths,
        "prototypes": prototypes,
        "prototype_labels": prototype_labels,
    }


def safe_import_faiss():
    try:
        import faiss  # type: ignore

        return faiss
    except Exception as e:
        print(f"⚠️  FAISS import failed, index will not be created: {e}")
        return None


def run_training(config: TrainingConfig) -> None:
    """
    Lance l'entraînement complet pour un config donné.
    Robuste aux erreurs fréquentes (FAISS absent, etc.).
    """
    print(f"\n🧩 CONFIGURATION\n{json.dumps(asdict(config), indent=2)}")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Seeds pour reproductibilité
    import random

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Dossiers
    for path in [
        config.output_path,
        config.checkpoints_path,
        config.logs_path,
    ]:
        Path(path).mkdir(parents=True, exist_ok=True)

    train_transform, val_transform = build_transforms(
        config.image_size, strong=config.strong_augmentation
    )

    # Datasets
    print("\n📁 Loading datasets...")
    dataset_root = config.dataset_path
    train_dataset = DiseaseDataset(
        dataset_root,
        splits=("train",),
        transform=train_transform,
        image_size=config.image_size,
    )
    val_dataset = DiseaseDataset(
        dataset_root,
        splits=("val",),
        transform=val_transform,
        image_size=config.image_size,
    )

    num_classes = len(train_dataset.class_to_idx)
    print(f"✅ Detected {num_classes} classes in TRAIN split")

    # DataLoaders & PKSampler
    P, K = config.P, config.K
    pk_sampler = PKSampler(train_dataset.labels, P=P, K=K)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=pk_sampler,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    print(
        f"✅ Train: {len(train_dataset)} images, Val: {len(val_dataset)} images, "
        f"batch_size(P*K)={P*K}"
    )

    # Modèle
    print("\n🧠 Creating model...")
    model = DiagnosticModel(
        config.model_name,
        embedding_dim=config.embedding_dim,
        image_size=config.image_size,
    ).to(DEVICE)
    print(
        f"✅ Model {config.model_name} created "
        f"({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)"
    )

    criterion = SupConLoss(temperature=config.temperature).to(DEVICE)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    from torch.optim.lr_scheduler import LambdaLR
    import math

    warmup_epochs = 5

    def lr_lambda(current_epoch: int):
        if current_epoch < warmup_epochs:
            return float(current_epoch) / float(max(1, warmup_epochs))
        progress = float(current_epoch - warmup_epochs) / float(
            max(1, config.num_epochs - warmup_epochs)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    print("✅ Optimizer & Scheduler (warmup+cosine) created")

    memory_bank = MemoryBank(
        capacity=config.memory_bank_capacity, dim=config.embedding_dim
    )
    ema = ModelEMA(model, decay=config.ema_decay)
    scaler = GradScaler(enabled=(DEVICE.type == "cuda"))
    print(f"✅ EMA & GradScaler initialized (AMP={scaler.is_enabled()})")

    # Reprise depuis last_checkpoint si demande et fichier present
    start_epoch = 0
    best_metric = -float("inf")
    best_accuracy = 0.0
    patience = 5
    patience_counter = 0
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "intra_dist": [],
        "top_k_acc": [],
        "recall_at1": [],
        "accuracy": [],
        "map": [],
    }

    last_ckpt_path = Path(config.checkpoints_path) / "last_checkpoint.pt"
    if (
        getattr(config, "resume_from_checkpoint", True)
        and last_ckpt_path.exists()
    ):
        try:
            ckpt = torch.load(last_ckpt_path, map_location=DEVICE)
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_metric = ckpt.get("best_metric", -float("inf"))
            patience_counter = ckpt.get("patience_counter", 0)
            history = ckpt.get("history", history)
            ema = ModelEMA(model, decay=config.ema_decay)
            print(
                f"\n✅ REPRISE depuis epoch {start_epoch} "
                f"(best Recall@1={best_metric:.4f})\n"
            )
        except Exception as e:
            print(f"\n⚠️  Resume failed ({e}), demarrage depuis epoch 0\n")
            start_epoch = 0

    print("\n" + "=" * 80)
    print(f"🚀 STARTING TRAINING: {config.experiment_name}")
    print("=" * 80 + "\n")

    try:
        for epoch in range(start_epoch, config.num_epochs):
            print(f"\n📌 Epoch {epoch + 1}/{config.num_epochs}")

            train_loss = train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                memory_bank,
                ema,
                DEVICE,
                scaler,
            )
            print(f"  Train Loss: {train_loss:.4f}")
            history["train_loss"].append(train_loss)

            metrics = validate(model, val_loader, criterion, DEVICE, ema)
            print(f"  Val Loss (SupCon): {metrics['val_loss']:.4f}")
            print(f"  Intra-class distance: {metrics['intra_dist']:.4f}")
            print(f"  Top-5 accuracy: {metrics['top5_acc']:.4f}")
            print(f"  Recall@1: {metrics['recall_at1']:.4f}")
            if "accuracy" in metrics:
                print(
                    f"  Validation accuracy: {metrics['accuracy']:.4f} "
                    f"({metrics.get('accuracy_correct', 0)}/{metrics.get('accuracy_total', 0)})"
                )
            print(f"  mAP: {metrics['map']:.4f}")

            history["val_loss"].append(metrics["val_loss"])
            history["intra_dist"].append(metrics["intra_dist"])
            history["top_k_acc"].append(metrics["top5_acc"])
            history["recall_at1"].append(metrics["recall_at1"])
            history["accuracy"].append(metrics.get("accuracy", 0.0))
            history["map"].append(metrics["map"])

            scheduler.step()

            current_metric = metrics["recall_at1"]
            if current_metric > best_metric:
                best_metric = current_metric
                best_accuracy = metrics.get("accuracy", best_accuracy)
                patience_counter = 0

                checkpoint_path = (
                    Path(config.checkpoints_path) / "best_model.pt"
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "metrics": metrics,
                        "config": asdict(config),
                    },
                    checkpoint_path,
                )
                print(
                    f"  ✅ Best model saved (Recall@1={best_metric:.4f}, accuracy={best_accuracy:.4f}): "
                    f"{checkpoint_path}"
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(
                        f"  ⏹️  Early stopping at epoch {epoch + 1} "
                        "(no Recall@1 improvement)"
                    )
                    break

            # Sauvegarder last_checkpoint pour reprise (chaque epoch)
            Path(config.checkpoints_path).mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_metric": best_metric,
                    "patience_counter": patience_counter,
                    "history": history,
                },
                last_ckpt_path,
            )
    except RuntimeError as e:
        print(f"\n❌ RuntimeError during training: {e}")
        print("   Try reducing batch size or image size if this is OOM.")
        return

    print("\n✅ Training complete!")

    # Charger meilleur modèle
    best_ckpt_path = Path(config.checkpoints_path) / "best_model.pt"
    if best_ckpt_path.exists():
        best_checkpoint = torch.load(best_ckpt_path, map_location=DEVICE)
        model.load_state_dict(best_checkpoint["model_state_dict"])
        print(f"✅ Best model re-loaded from {best_ckpt_path}")
    else:
        print("⚠️  Best checkpoint not found, using last model state.")

    # Embeddings + prototypes sur train+val+test
    print("\n🔄 Computing embeddings / prototypes on full dataset...")
    full_dataset = DiseaseDataset(
        dataset_root, splits=("train", "val", "test"), transform=val_transform
    )
    emb_info = compute_full_embeddings_and_prototypes(
        model,
        full_dataset,
        DEVICE,
        batch_size=max(32, config.batch_size),
        num_workers=config.num_workers,
    )

    embeddings_matrix = emb_info["embeddings_matrix"]
    all_labels = emb_info["labels"]
    all_paths = emb_info["paths"]
    prototypes = emb_info["prototypes"]
    prototype_labels = emb_info["prototype_labels"]

    # FAISS
    faiss = safe_import_faiss()
    index_path = None
    if faiss is not None:
        print("\n🔍 Creating FAISS index...")
        index = faiss.IndexFlatIP(config.embedding_dim)
        index.add(embeddings_matrix)
        index_path = Path(config.output_path) / "faiss_index.bin"
        Path(config.output_path).mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_path))
        print(f"✅ FAISS index saved: {index_path}")

    # Metadata
    metadata = {
        "embeddings_shape": embeddings_matrix.shape,
        "labels": all_labels,
        "image_paths": all_paths,
        "prototypes": prototypes.tolist(),
        "prototype_labels": prototype_labels,
        "class_to_idx": full_dataset.class_to_idx,
        "idx_to_class": full_dataset.idx_to_class,
        "num_classes": len(full_dataset.class_to_idx),
        "embedding_dim": config.embedding_dim,
        "experiment_name": config.experiment_name,
        "timestamp": datetime.now().isoformat(),
        "history": history,
        "best_recall_at1": best_metric,
        "best_accuracy": best_accuracy,
        "faiss_index_path": str(index_path) if index_path is not None else None,
    }

    output_path = Path(config.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    metadata_json = output_path / "metadata.json"
    with open(metadata_json, "w") as f:
        json.dump(
            {
                k: v
                for k, v in metadata.items()
                if k not in ("image_paths",)
            },
            f,
            indent=2,
        )
    print(f"✅ Metadata (light) saved: {metadata_json}")

    metadata_pkl = output_path / "metadata.pkl"
    with open(metadata_pkl, "wb") as f:
        pickle.dump(metadata, f)
    print(f"✅ Full metadata (with paths) saved: {metadata_pkl}")

    # Sauvegarde du modèle
    model_path = output_path / "senedisease_macro_f1.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "class_to_idx": full_dataset.class_to_idx,
            "idx_to_class": full_dataset.idx_to_class,
        },
        model_path,
    )
    print(f"✅ Model weights saved: {model_path}")


__all__ = ["TrainingConfig", "get_default_config", "run_training"]

