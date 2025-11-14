# train_and_infer.py
r"""
Script modulaire pour entraîner / évaluer / prédire un modèle de classification
(détecter une maladie depuis une image). Conçu pour être évolutif (scale-up).
Usage:
  - Entraîner:
      python train_and_infer.py --mode train --data-file "C:\Users\moham\Music\plantdataset\multimodal_dataset_fr.jsonl"
  - Prédire:
      python train_and_infer.py --mode predict --image "C:\Users\moham\Pictures\New folder\img.jpg" --model-path out/best.pth
"""

import os
import json
import time
import random
import argparse
from collections import Counter, defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import torchvision.models as tvmodels

from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

# -------------------------
# Dataset
# -------------------------
class MultimodalImageDataset(Dataset):
    """
    Lit le JSONL généré (image_path, label, split, text) et renvoie l'image transformée + label_idx.
    image_path dans JSONL est relatif au param root_dir (ou absolu).
    """
    def __init__(self, jsonl_file, split="train", root_dir=r"C:\Users\moham\Music\plantdataset",
                 image_size=224, verbose=True):
        self.root_dir = root_dir
        self.split = split
        self.samples = []
        self.verbose = verbose

        if verbose:
            print(f"📂 Loading JSONL: {jsonl_file}  (split={split})  root_dir={root_dir}")

        with open(jsonl_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                try:
                    j = json.loads(line.strip())
                except Exception as e:
                    if verbose:
                        print(f"⚠️ JSON parse error line {i}: {e}")
                    continue
                if j.get("split", "train") == split:
                    self.samples.append(j)

        if verbose:
            print(f"✅ Found {len(self.samples)} samples for split '{split}'")

        # Construire la liste de transforms sans lambda (problème de pickling sous Windows)
        transforms_list = [
            T.Resize((image_size, image_size))
        ]
        if split == "train":
            transforms_list.append(T.RandomHorizontalFlip(p=0.5))
        transforms_list.extend([
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        self.transform = T.Compose(transforms_list)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_rel = sample["image_path"]
        # join only if not absolute
        if os.path.isabs(img_rel):
            img_path = img_rel
        else:
            img_path = os.path.join(self.root_dir, img_rel.replace("/", os.sep))
        if not os.path.exists(img_path):
            # fallback: print and return a black image
            print(f"⚠️ Image not found: {img_path} (index {idx}) - returning zero image")
            img = Image.new("RGB", (224,224), (0,0,0))
        else:
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"⚠️ Error opening {img_path}: {e} - returning zero image")
                img = Image.new("RGB", (224,224), (0,0,0))

        image = self.transform(img)

        label = sample.get("label", "unknown")
        text = sample.get("text", "")

        return {"image": image, "label": label, "text": text, "image_path": img_rel}

# -------------------------
# Model
# -------------------------
class ImageClassifier(nn.Module):
    """
    Simple wrapper: backbone -> projection -> classification head
    - backbone: resnet18/resnet50 (pretrained)
    - proj_dim: embedding dimension before classification
    """
    def __init__(self, backbone_name="resnet18", num_classes=2, proj_dim=512, pretrained=True, freeze_backbone=False):
        super().__init__()
        backbone_name = backbone_name.lower()
        if backbone_name == "resnet18":
            # Support both old (pretrained=...) and new (weights=...) torchvision APIs
            try:
                weights = tvmodels.ResNet18_Weights.DEFAULT if pretrained else None
                backbone = tvmodels.resnet18(weights=weights)
            except Exception:
                backbone = tvmodels.resnet18(pretrained=pretrained)
            feat_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif backbone_name == "resnet50":
            try:
                weights = tvmodels.ResNet50_Weights.DEFAULT if pretrained else None
                backbone = tvmodels.resnet50(weights=weights)
            except Exception:
                backbone = tvmodels.resnet50(pretrained=pretrained)
            feat_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
        else:
            raise ValueError("backbone must be resnet18 or resnet50")

        self.backbone = backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.proj = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.classifier = nn.Linear(proj_dim, num_classes)

    def forward(self, images):
        feats = self.backbone(images)         # [B, feat_dim]
        emb = self.proj(feats)               # [B, proj_dim]
        logits = self.classifier(emb)        # [B, num_classes]
        return logits, emb

# -------------------------
# Utils: label mapping, class weights
# -------------------------
def build_label_mapping(dataset):
    labels = [s["label"] for s in dataset.samples]
    unique = sorted(list(set(labels)))
    label2idx = {l:i for i,l in enumerate(unique)}
    idx2label = {i:l for l,i in label2idx.items()}
    return label2idx, idx2label

def compute_class_weights(dataset, label2idx):
    counts = Counter([s["label"] for s in dataset.samples])
    labels = list(label2idx.keys())
    freq = np.array([counts.get(lbl,0) for lbl in labels], dtype=np.float32)
    # avoid zero
    freq = np.where(freq==0, 1.0, freq)
    inv = 1.0 / freq
    weights = inv / np.sum(inv) * len(labels)
    return torch.tensor(weights, dtype=torch.float32)

# -------------------------
# Training / Evaluation
# -------------------------
def train_epoch(model, loader, criterion, optimizer, device, scaler=None, accumulate_steps=1, print_every=50):
    model.train()
    running_loss = 0.0
    running_preds = []
    running_labels = []
    optimizer.zero_grad()
    pbar = tqdm(enumerate(loader), total=len(loader), desc="train", leave=False)
    for step, batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        labels_raw = batch["label"]
        # labels should be indices outside; we'll assume labels_idx provided by caller via dataset mapping
        labels = torch.tensor([label_map[l] for l in labels_raw], dtype=torch.long, device=device)

        # Utiliser la nouvelle API autocast si disponible
        if scaler is not None and device.type == "cuda":
            try:
                with torch.amp.autocast(device_type="cuda"):
                    logits, _ = model(images)
                    loss = criterion(logits, labels) / accumulate_steps
            except (AttributeError, TypeError):
                # Fallback vers l'ancienne API
                with torch.cuda.amp.autocast():
                    logits, _ = model(images)
                    loss = criterion(logits, labels) / accumulate_steps
        else:
            logits, _ = model(images)
            loss = criterion(logits, labels) / accumulate_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step+1) % accumulate_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * accumulate_steps
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        running_preds.extend(preds.tolist())
        running_labels.extend(labels.detach().cpu().numpy().tolist())

        if (step+1) % print_every == 0:
            acc = accuracy_score(running_labels, running_preds)
            pbar.set_postfix({"loss": f"{running_loss/(step+1):.4f}", "acc": f"{acc:.4f}"})

    avg_loss = running_loss / len(loader)
    acc = accuracy_score(running_labels, running_preds)
    return avg_loss, acc

def eval_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    preds_all = []
    labels_all = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            labels_raw = batch["label"]
            labels = torch.tensor([label_map[l] for l in labels_raw], dtype=torch.long, device=device)
            logits, _ = model(images)
            loss = criterion(logits, labels)
            running_loss += loss.item()
            preds = logits.argmax(dim=1).cpu().numpy()
            preds_all.extend(preds.tolist())
            labels_all.extend(labels.detach().cpu().numpy().tolist())

    avg_loss = running_loss / len(loader)
    acc = accuracy_score(labels_all, preds_all)
    f1_macro = f1_score(labels_all, preds_all, average="macro")
    prec, rec, f1, _ = precision_recall_fscore_support(labels_all, preds_all, average=None, zero_division=0)
    cm = confusion_matrix(labels_all, preds_all)
    return {"loss": avg_loss, "acc": acc, "f1_macro": f1_macro, "per_class": {"precision":prec.tolist(), "recall":rec.tolist(), "f1":f1.tolist()}, "cm": cm, "y_true": labels_all, "y_pred": preds_all}

# -------------------------
# Checkpoint helpers
# -------------------------
def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    return checkpoint

# -------------------------
# Main train/predict
# -------------------------
def main(args):
    global label_map  # used inside train_epoch/eval_model
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    print(f"Using device: {device}")

    if args.mode == "train":
        # datasets
        train_ds = MultimodalImageDataset(args.data_file, split="train", root_dir=args.root_dir,
                                          image_size=args.image_size, verbose=True)
        val_ds = MultimodalImageDataset(args.data_file, split="val", root_dir=args.root_dir,
                                        image_size=args.image_size, verbose=True)

        # label mapping
        label2idx, idx2label = build_label_mapping(train_ds)
        # ensure labels from val included
        for s in val_ds.samples:
            if s["label"] not in label2idx:
                label2idx[s["label"]] = len(label2idx)
                idx2label[label2idx[s["label"]]] = s["label"]

        label_map = label2idx  # for inner functions

        # save mapping
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "label2idx.json"), "w", encoding="utf-8") as f:
            json.dump(label2idx, f, indent=2, ensure_ascii=False)
        with open(os.path.join(args.output_dir, "idx2label.json"), "w", encoding="utf-8") as f:
            json.dump(idx2label, f, indent=2, ensure_ascii=False)
        print(f"Labels ({len(label2idx)}): {list(label2idx.keys())}")

        # dataloaders (désactiver multiprocessing par défaut sous Windows pour éviter pickle issues)
        pin_mem = (device.type == "cuda")
        effective_num_workers = args.num_workers if args.use_multiprocessing else 0
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=effective_num_workers, pin_memory=pin_mem, persistent_workers=False)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=effective_num_workers, pin_memory=pin_mem, persistent_workers=False)

        # model
        model = ImageClassifier(backbone_name=args.backbone, num_classes=len(label2idx),
                                proj_dim=args.proj_dim, pretrained=not args.no_pretrained,
                                freeze_backbone=args.freeze_backbone).to(device)
        print(model)

        # class weights & loss
        class_weights = compute_class_weights(train_ds, label2idx).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights if args.use_class_weights else None)

        # optimizer & scheduler
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1,args.epochs))

        # Initialiser GradScaler selon la version de PyTorch (fallback si nécessaire)
        scaler = None
        if args.amp and device.type == "cuda":
            try:
                # Essayer l'API torch.amp (PyTorch récents)
                from torch import amp as torch_amp
                try:
                    scaler = torch_amp.GradScaler(device_type="cuda")
                except TypeError:
                    scaler = torch_amp.GradScaler()
            except Exception:
                # Fallback vers l'ancienne API
                try:
                    scaler = torch.cuda.amp.GradScaler()
                except Exception:
                    scaler = None

        best_val_acc = 0.0
        best_path = os.path.join(args.output_dir, "best.pth")
        start_epoch = 0

        # resume?
        if args.resume and os.path.exists(args.resume):
            cp = load_checkpoint(args.resume, device)
            model.load_state_dict(cp["model_state"])
            optimizer.load_state_dict(cp["opt_state"])
            start_epoch = cp.get("epoch", 0) + 1
            best_val_acc = cp.get("best_val_acc", 0.0)
            print(f"Resumed from {args.resume}, starting epoch {start_epoch}")

        print("🚀 Start training loop")
        for epoch in range(start_epoch, args.epochs):
            t0 = time.time()
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer,
                                                device, scaler=scaler, accumulate_steps=args.accumulate_steps,
                                                print_every=args.print_every)
            metrics = eval_model(model, val_loader, criterion, device)
            val_loss = metrics["loss"]
            val_acc = metrics["acc"]
            print(f"Epoch {epoch+1}/{args.epochs}  time={(time.time()-t0):.1f}s  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  val_loss={val_loss:.4f} val_acc={val_acc:.4f} f1={metrics['f1_macro']:.4f}")

            # scheduler step
            scheduler.step()

            # save checkpoint
            state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "opt_state": optimizer.state_dict(),
                "best_val_acc": best_val_acc
            }
            cp_path = os.path.join(args.output_dir, f"ckpt_epoch{epoch+1}.pth")
            save_checkpoint(state, cp_path)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(state, best_path)
                print(f"✨ New best model saved to {best_path} (val_acc={best_val_acc:.4f})")

            # simple early stopping by accuracy threshold (optional)
            if args.stop_if_val_acc >= 0 and val_acc >= args.stop_if_val_acc:
                print(f"Stopping early because val_acc >= {args.stop_if_val_acc}")
                break

        print("Training finished. Best val_acc:", best_val_acc)
        print("Model + label mapping saved in", args.output_dir)

    elif args.mode == "predict":
        # Load model and label mapping
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(args.model_path)
        label2idx = json.load(open(os.path.join(args.output_dir, "label2idx.json"), "r", encoding="utf-8"))
        idx2label = {int(k):v for k,v in json.load(open(os.path.join(args.output_dir, "idx2label.json"), "r", encoding="utf-8")).items()}
        label_map = label2idx

        num_classes = len(label2idx)
        model = ImageClassifier(backbone_name=args.backbone, num_classes=num_classes,
                                proj_dim=args.proj_dim, pretrained=False).to(device)
        ckpt = load_checkpoint(args.model_path, device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        # load mapping JSON for descriptions (maladies_enrichies.json)
        if args.mapping_json and os.path.exists(args.mapping_json):
            mapping = json.load(open(args.mapping_json, "r", encoding="utf-8"))
        else:
            mapping = {}

        # load image & preprocess
        transform = T.Compose([T.Resize((args.image_size,args.image_size)), T.ToTensor(),
                               T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        img_path = args.image
        if not os.path.isabs(img_path):
            img_path = os.path.join(args.root_dir, img_path)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"⚠️ Impossible d'ouvrir l'image: {img_path} -> {e}. Utilisation d'une image noire.")
            img = Image.new("RGB", (args.image_size, args.image_size), (0,0,0))
        x = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits, emb = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            topk = int(min(5, len(probs)))
            top_idx = probs.argsort()[::-1][:topk]
            print("🔎 Predictions (top probabilities):")
            for i in top_idx:
                cls_name = idx2label[i]
                p = probs[i]
                info = mapping.get(cls_name, {})
                # build short description
                desc = ""
                if isinstance(info, dict):
                    # try french description
                    desc = info.get("description", {})
                    if isinstance(desc, dict):
                        desc = desc.get("fr", next(iter(desc.values()), ""))
                    sympt = info.get("symptômes", "")
                    if isinstance(sympt, dict):
                        sympt = sympt.get("fr", "")
                    treatment = info.get("traitement", "")
                    if isinstance(treatment, dict):
                        treatment = treatment.get("fr", "")
                    print(f"- {cls_name}  p={p:.3f}")
                    if desc:
                        print(f"   Description: {desc}")
                    if sympt:
                        print(f"   Symptômes: {sympt}")
                    if treatment:
                        print(f"   Traitement: {treatment}")
                else:
                    print(f"- {cls_name}  p={p:.3f}")
        print("Done.")

    else:
        raise ValueError("mode must be train or predict")

# -------------------------
# Argument parsing
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train","predict"])
    parser.add_argument("--data-file", type=str, default=r"C:\Users\moham\Music\plantdataset\multimodal_dataset_fr.jsonl")
    parser.add_argument("--root-dir", dest="root_dir", type=str, default=r"C:\Users\moham\Music\plantdataset")
    parser.add_argument("--output-dir", dest="output_dir", type=str, default="out")
    parser.add_argument("--mapping-json", type=str, default=r"C:\Users\moham\Music\plantdataset\maladies_enrichies.json")
    # model / training params
    parser.add_argument("--backbone", type=str, default="resnet18", help="resnet18 or resnet50")
    parser.add_argument("--proj-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument("--accumulate-steps", type=int, default=1)
    parser.add_argument("--amp", action="store_true", help="use mixed precision (if GPU available)")
    parser.add_argument("--resume", type=str, default="", help="path to checkpoint to resume")
    parser.add_argument("--model-path", type=str, default="out/best.pth", help="used in predict mode")
    parser.add_argument("--image", type=str, default="", help="image path for predict mode")
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--stop-if-val-acc", type=float, default=-1.0, help="stop early when reaching this val acc (>=0).")
    parser.add_argument("--resume-best", action="store_true")
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--use-multiprocessing", action="store_true")
    args = parser.parse_args()

    # convenience aliases
    args.mapping_json = args.mapping_json
    args.model_path = args.model_path
    main(args)
