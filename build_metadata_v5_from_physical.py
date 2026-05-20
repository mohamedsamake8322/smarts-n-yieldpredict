"""
SMART METADATA BUILDER V5.1 (COLAB-FIRST)
=========================================
Build full V5.1 metadata directly from physical Plantdataset folders.

This script generates:
  - class_report.json
  - class_groups.json
  - class_mapping.json
  - class_hierarchy.json
  - phase_groups.json
  - multitask_config.json
  - training_config.json
  - class_weights_log.json
  - clean_label_map.json
  - train_multitask.json
  - val_multitask.json
  - test_multitask.json
  - train.json / val.json / test.json (compat)
  - dataset_summary.json
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ============================================================================
# CONFIG (Colab-style)
# ============================================================================
FORCE_REMOUNT_DRIVE = True
DATASET_DIR = Path("/content/drive/MyDrive/Plantdataset")
META_DIR = Path("/content/drive/MyDrive/Plantdataset_metadata")

SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
CORE_THRESHOLD = 500
RARE_THRESHOLD = 100


def save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def maybe_mount_drive(force_remount: bool = True) -> None:
    """Mount Google Drive when running in Colab; noop elsewhere."""
    try:
        from google.colab import drive  # type: ignore
    except Exception:
        print("ℹ️ google.colab non détecté (exécution locale).")
        return
    drive.mount("/content/drive", force_remount=force_remount)
    print("✅ Google Drive monté.")


def list_images(folder: Path) -> List[Path]:
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTS]


def tokenize(name: str) -> List[str]:
    s = name.replace("-", "_").replace(" ", "_")
    return [t for t in s.split("_") if t]


def infer_organ(tokens_lower: List[str]) -> str:
    organ_map = {
        "leaf": "leaf",
        "leaves": "leaf",
        "foliage": "leaf",
        "fruit": "fruit",
        "stem": "stem",
        "root": "root",
        "flower": "flower",
        "seed": "seed",
    }
    for t in tokens_lower:
        if t in organ_map:
            return organ_map[t]
    return "unknown"


def infer_pattern(tokens_lower: List[str]) -> str:
    pattern_keywords = {
        "healthy": "healthy",
        "rust": "rust",
        "blight": "blight",
        "spot": "spot",
        "scab": "scab",
        "mildew": "mildew",
        "rot": "rot",
        "wilt": "wilt",
        "mosaic": "mosaic",
        "virus": "virus",
        "bacterial": "bacterial",
        "fungal": "fungal",
        "deficiency": "deficiency",
        "chlorosis": "chlorosis",
    }
    for t in tokens_lower:
        if t in pattern_keywords:
            return pattern_keywords[t]
    return "other"


def infer_category(tokens_lower: List[str]) -> str:
    if "healthy" in tokens_lower:
        return "healthy"
    return "disease"


def infer_crop(tokens: List[str]) -> str:
    if not tokens:
        return "unknown"
    return tokens[0].lower()


def stable_split_counts(n: int, train_ratio: float, val_ratio: float) -> Tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    if n >= 3:
        if n_val == 0:
            n_val = 1
            n_train = max(1, n_train - 1)
        if n_test == 0:
            n_test = 1
            n_train = max(1, n_train - 1)
    n_train = max(1, n_train) if n >= 1 else 0
    total = n_train + n_val + n_test
    if total != n:
        n_test += n - total
    return n_train, n_val, n_test


def make_class_weights_log(class_counts: Dict[str, int]) -> Dict[str, float]:
    if not class_counts:
        return {}
    max_count = max(class_counts.values())
    raw = {}
    for cls, count in class_counts.items():
        count = max(1, int(count))
        raw[cls] = math.log1p(max_count / count)
    mean_w = sum(raw.values()) / max(1, len(raw))
    return {k: round(v / mean_w, 6) for k, v in raw.items()}


def make_clean_label_map(classes: List[str]) -> Dict[str, str]:
    """
    Heuristic alias map:
      - <crop>_Healthy_Leaf -> <crop>_Healthy (if exists)
      - <crop>_Healthy_Leaves -> <crop>_Healthy (if exists)
    """
    cls_set = set(classes)
    out = {}
    for c in classes:
        tokens = tokenize(c)
        tokens_l = [t.lower() for t in tokens]
        if len(tokens) >= 3 and "healthy" in tokens_l:
            if tokens_l[-1] in {"leaf", "leaves"}:
                base = [t for t in tokens if t.lower() not in {"leaf", "leaves"}]
                cand = "_".join(base)
                if cand in cls_set and cand != c:
                    out[c] = cand
    return out


def main() -> None:
    print("🔧 SMART METADATA BUILDER V5.1 (COLAB-FIRST)")
    maybe_mount_drive(force_remount=FORCE_REMOUNT_DRIVE)
    random.seed(SEED)
    ratio_sum = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"TRAIN_RATIO + VAL_RATIO + TEST_RATIO must be 1.0, got {ratio_sum}")

    dataset_dir = Path(DATASET_DIR)
    meta_dir = Path(META_DIR)
    meta_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 DATASET_DIR: {dataset_dir}")
    print(f"📁 META_DIR   : {meta_dir}")

    # Quick write test (same spirit as your analyzer)
    test_file = meta_dir / "test_write.json"
    save_json({"status": "ok"}, test_file)
    print(f"🧪 Test write: {'OK' if test_file.exists() else 'FAILED'}")

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    class_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()], key=lambda p: p.name.lower())
    if not class_dirs:
        raise RuntimeError(f"No class folders found in: {dataset_dir}")

    # 1) Index physical images
    dataset_index = []
    class_counts: Dict[str, int] = {}
    for d in class_dirs:
        cls = d.name
        imgs = list_images(d)
        class_counts[cls] = len(imgs)
        for img in imgs:
            dataset_index.append({"path": str(img), "class": cls})

    # 2) Group classes
    class_groups = {"CORE": [], "EXTENDED": [], "RARE": []}
    for cls, count in class_counts.items():
        if count >= CORE_THRESHOLD:
            class_groups["CORE"].append(cls)
        elif count >= RARE_THRESHOLD:
            class_groups["EXTENDED"].append(cls)
        else:
            class_groups["RARE"].append(cls)

    # 3) Build class report
    class_report = {}
    for cls, count in class_counts.items():
        if cls in class_groups["CORE"]:
            level = "CORE"
        elif cls in class_groups["EXTENDED"]:
            level = "EXTENDED"
        else:
            level = "RARE"
        class_report[cls] = {"count": int(count), "level": level}

    # 4) Mapping
    classes_sorted = sorted(class_counts.keys())
    class_to_idx = {c: i for i, c in enumerate(classes_sorted)}
    idx_to_class = {str(i): c for c, i in class_to_idx.items()}
    class_mapping = {"class_to_idx": class_to_idx, "idx_to_class": idx_to_class}

    # 5) Hierarchy + multitask label spaces
    class_hierarchy = {}
    crop_set = set()
    category_set = set()
    for cls in classes_sorted:
        tokens = tokenize(cls)
        t_l = [t.lower() for t in tokens]
        crop = infer_crop(tokens)
        organ = infer_organ(t_l)
        pattern = infer_pattern(t_l)
        category = infer_category(t_l)
        class_hierarchy[cls] = {
            "crop": crop,
            "organ": organ,
            "pattern": pattern,
            "category": category,
        }
        crop_set.add(crop)
        category_set.add(category)

    crop_to_idx = {c: i for i, c in enumerate(sorted(crop_set))}
    category_to_idx = {c: i for i, c in enumerate(sorted(category_set))}

    multitask_config = {
        "crop_to_idx": crop_to_idx,
        "category_to_idx": category_to_idx,
        "num_crops": len(crop_to_idx),
        "num_categories": len(category_to_idx),
        "loss_weights": {"main": 1.0, "crop": 0.2, "category": 0.15},
    }

    # 6) Split stratified
    by_class = defaultdict(list)
    for item in dataset_index:
        by_class[item["class"]].append(item)

    train_raw, val_raw, test_raw = [], [], []
    for cls, items in by_class.items():
        random.shuffle(items)
        n = len(items)
        n_train, n_val, n_test = stable_split_counts(n, TRAIN_RATIO, VAL_RATIO)
        train_raw.extend(items[:n_train])
        val_raw.extend(items[n_train:n_train + n_val])
        test_raw.extend(items[n_train + n_val:n_train + n_val + n_test])

    # 7) Convert to multitask records
    def to_multitask(items: List[dict]) -> List[dict]:
        out = []
        for it in items:
            cls = it["class"]
            hier = class_hierarchy.get(cls, {})
            out.append({
                "path": it["path"],
                "class": cls,
                "label": class_to_idx[cls],
                "crop": hier.get("crop", "unknown"),
                "organ": hier.get("organ", "unknown"),
                "pattern": hier.get("pattern", "other"),
                "category": hier.get("category", "disease"),
                "crop_label": crop_to_idx.get(hier.get("crop", "unknown"), -1),
                "category_label": category_to_idx.get(hier.get("category", "disease"), -1),
            })
        return out

    train_mt = to_multitask(train_raw)
    val_mt = to_multitask(val_raw)
    test_mt = to_multitask(test_raw)

    # 8) Class weights + clean alias map
    class_weights_log = make_class_weights_log(class_counts)
    clean_label_map = make_clean_label_map(classes_sorted)

    # 9) Phase groups expected by training script
    def as_phase_classes(names: List[str]) -> List[dict]:
        return [{"class": c, "count": int(class_counts.get(c, 0))} for c in sorted(names)]

    phase_groups = {
        "phase_1": {
            "description": "Core classes only",
            "classes": as_phase_classes(class_groups["CORE"]),
        },
        "phase_2": {
            "description": "Core + Extended",
            "classes": as_phase_classes(class_groups["CORE"] + class_groups["EXTENDED"]),
        },
        "phase_3": {
            "description": "All classes (Core + Extended + Rare)",
            "classes": as_phase_classes(classes_sorted),
        },
    }

    training_config = {
        "backbone": "dinov2_vitb14",
        "input_size": 518,
        "embed_dim": 768,
        "batch_size": 32,
        "epochs": 60,
        "warmup_epochs": 5,
        "lr_head": 1e-4,
        "lr_backbone": 1e-5,
        "weight_decay": 0.05,
        "label_smoothing": 0.05,
        "focal_gamma": 2.0,
    }

    summary = {
        "dataset_dir": str(dataset_dir),
        "total_images": len(dataset_index),
        "total_classes": len(class_counts),
        "split": {"train": len(train_raw), "val": len(val_raw), "test": len(test_raw)},
        "group_counts": {
            "core": len(class_groups["CORE"]),
            "extended": len(class_groups["EXTENDED"]),
            "rare": len(class_groups["RARE"]),
        },
        "num_crops": len(crop_to_idx),
        "num_categories": len(category_to_idx),
        "clean_alias_pairs": len(clean_label_map),
        "seed": SEED,
    }

    # 10) Save everything
    save_json(class_report, meta_dir / "class_report.json")
    save_json(class_groups, meta_dir / "class_groups.json")
    save_json(class_mapping, meta_dir / "class_mapping.json")
    save_json(class_hierarchy, meta_dir / "class_hierarchy.json")
    save_json(phase_groups, meta_dir / "phase_groups.json")
    save_json(multitask_config, meta_dir / "multitask_config.json")
    save_json(training_config, meta_dir / "training_config.json")
    save_json(class_weights_log, meta_dir / "class_weights_log.json")
    save_json(clean_label_map, meta_dir / "clean_label_map.json")
    save_json(train_mt, meta_dir / "train_multitask.json")
    save_json(val_mt, meta_dir / "val_multitask.json")
    save_json(test_mt, meta_dir / "test_multitask.json")

    # Compatibility files
    save_json(train_raw, meta_dir / "train.json")
    save_json(val_raw, meta_dir / "val.json")
    save_json(test_raw, meta_dir / "test.json")
    save_json(summary, meta_dir / "dataset_summary.json")

    # Quick write test
    save_json({"status": "ok"}, meta_dir / "test_write.json")

    print("\n===== METADATA V5.1 BUILT =====")
    print(f"META_DIR: {meta_dir}")
    print(json.dumps(summary, indent=2))
    print("\nFiles:")
    for p in sorted(meta_dir.iterdir(), key=lambda x: x.name.lower()):
        print(f" - {p.name}")


if __name__ == "__main__":
    main()
