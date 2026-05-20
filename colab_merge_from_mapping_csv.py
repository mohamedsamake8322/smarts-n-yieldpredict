#!/usr/bin/env python3
"""
Fusion securisee de classes depuis Boua_extracted vers Plantdataset a partir d'un CSV.

Usage Colab:
  !python /content/drive/MyDrive/colab_merge_from_mapping_csv.py \
      --csv /content/drive/MyDrive/cross_dataset_merge_candidates_v1.csv \
      --source /content/drive/MyDrive/Boua_extracted \
      --target /content/drive/MyDrive/Plantdataset \
      --dry-run

Puis en reel:
  !python /content/drive/MyDrive/colab_merge_from_mapping_csv.py \
      --csv /content/drive/MyDrive/cross_dataset_merge_candidates_v1.csv \
      --source /content/drive/MyDrive/Boua_extracted \
      --target /content/drive/MyDrive/Plantdataset
"""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple


ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


@dataclass(frozen=True)
class MappingRow:
    dataset1_class: str
    dataset2_class: str
    canonical_name: str
    match_type: str


def list_image_files(folder: Path) -> List[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    files: List[Path] = []
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS:
            files.append(p)
    return files


def normalize_label(name: str) -> str:
    return "_".join(name.strip().split())


def load_mapping_rows(csv_path: Path) -> List[MappingRow]:
    rows: List[MappingRow] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"dataset1_class", "dataset2_class", "canonical_name", "match_type"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Colonnes manquantes dans CSV: {sorted(missing)}")

        for raw in reader:
            row = MappingRow(
                dataset1_class=normalize_label(raw["dataset1_class"]),
                dataset2_class=normalize_label(raw["dataset2_class"]),
                canonical_name=normalize_label(raw["canonical_name"]),
                match_type=normalize_label(raw["match_type"]).upper(),
            )
            rows.append(row)
    return rows


def build_move_plan(
    rows: List[MappingRow],
    include_review: bool,
) -> Tuple[Dict[str, str], List[str]]:
    allowed_types = {"EXACT", "ALIAS"}
    if include_review:
        allowed_types.add("REVIEW")

    class_to_canonical: Dict[str, str] = {}
    skipped: List[str] = []

    for row in rows:
        if row.match_type not in allowed_types:
            skipped.append(
                f"SKIP match_type={row.match_type}: {row.dataset1_class} / {row.dataset2_class}"
            )
            continue

        candidates = [row.dataset1_class, row.dataset2_class]
        for cls in candidates:
            previous = class_to_canonical.get(cls)
            if previous is None:
                class_to_canonical[cls] = row.canonical_name
            elif previous != row.canonical_name:
                raise ValueError(
                    "Ambiguite mapping: "
                    f"la classe '{cls}' pointe vers '{previous}' et '{row.canonical_name}'."
                )

    return class_to_canonical, skipped


def preflight(
    source_root: Path,
    target_root: Path,
    class_to_canonical: Dict[str, str],
) -> Dict[str, object]:
    source_classes = sorted([p.name for p in source_root.iterdir() if p.is_dir()])
    source_set = set(source_classes)

    mapped_existing = sorted([c for c in class_to_canonical if c in source_set])
    mapped_missing = sorted([c for c in class_to_canonical if c not in source_set])

    canonical_targets: Set[str] = set(class_to_canonical.values())
    existing_target_classes = sorted(
        [c for c in canonical_targets if (target_root / c).exists()]
    )

    return {
        "source_total_classes": len(source_classes),
        "mapped_existing_classes": len(mapped_existing),
        "mapped_missing_classes": len(mapped_missing),
        "existing_target_canonical_classes": len(existing_target_classes),
        "mapped_existing_list": mapped_existing,
        "mapped_missing_list": mapped_missing,
        "existing_target_canonical_list": existing_target_classes,
    }


def safe_move_file(src_file: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_file = dst_dir / src_file.name
    if not dst_file.exists():
        shutil.move(str(src_file), str(dst_file))
        return dst_file

    stem = src_file.stem
    suffix = src_file.suffix
    idx = 1
    while True:
        candidate = dst_dir / f"{stem}__dup{idx}{suffix}"
        if not candidate.exists():
            shutil.move(str(src_file), str(candidate))
            return candidate
        idx += 1


def execute_merge(
    source_root: Path,
    target_root: Path,
    class_to_canonical: Dict[str, str],
    report_path: Path,
    dry_run: bool,
) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    target_root.mkdir(parents=True, exist_ok=True)

    moved_files = 0
    moved_classes = 0
    skipped_classes = 0
    report_rows: List[Dict[str, str]] = []

    source_classes = sorted([p for p in source_root.iterdir() if p.is_dir()])
    for src_class_dir in source_classes:
        src_class = src_class_dir.name
        canonical = class_to_canonical.get(src_class)
        if canonical is None:
            skipped_classes += 1
            report_rows.append(
                {
                    "timestamp": now,
                    "source_class": src_class,
                    "canonical_name": "",
                    "status": "SKIPPED_NOT_IN_MAPPING",
                    "images_moved": "0",
                    "note": "Classe non presente dans le CSV filtre",
                }
            )
            continue

        images = list_image_files(src_class_dir)
        image_count = len(images)
        dst_dir = target_root / canonical

        if dry_run:
            report_rows.append(
                {
                    "timestamp": now,
                    "source_class": src_class,
                    "canonical_name": canonical,
                    "status": "DRY_RUN_READY",
                    "images_moved": str(image_count),
                    "note": f"Destination: {dst_dir}",
                }
            )
            moved_classes += 1
            moved_files += image_count
            continue

        actual_moved = 0
        for src_file in images:
            safe_move_file(src_file, dst_dir)
            actual_moved += 1

        remaining_images = list_image_files(src_class_dir)
        if len(remaining_images) == 0:
            archive_root = source_root / "_merged_empty_dirs"
            archive_root.mkdir(parents=True, exist_ok=True)
            archived_dir = archive_root / src_class
            if archived_dir.exists():
                shutil.rmtree(archived_dir)
            shutil.move(str(src_class_dir), str(archived_dir))
            status = "MERGED_AND_ARCHIVED_EMPTY_DIR"
        else:
            status = "MERGED_PARTIAL_NON_IMAGE_LEFT"

        report_rows.append(
            {
                "timestamp": now,
                "source_class": src_class,
                "canonical_name": canonical,
                "status": status,
                "images_moved": str(actual_moved),
                "note": f"Destination: {dst_dir}",
            }
        )
        moved_classes += 1
        moved_files += actual_moved

    with report_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "source_class",
                "canonical_name",
                "status",
                "images_moved",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(report_rows)

    print("\n=================== RAPPORT EXECUTION ===================")
    print(f"Mode               : {'DRY-RUN' if dry_run else 'REEL'}")
    print(f"Classes traitees   : {moved_classes}")
    print(f"Classes ignorees   : {skipped_classes}")
    print(f"Images deplacees   : {moved_files}")
    print(f"Rapport CSV        : {report_path}")
    print("=========================================================\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fusion securisee de classes par CSV (Colab-friendly)."
    )
    parser.add_argument(
        "--csv",
        default="/content/drive/MyDrive/cross_dataset_merge_candidates_v1.csv",
        help="Chemin du CSV mapping.",
    )
    parser.add_argument(
        "--source",
        default="/content/drive/MyDrive/Boua_extracted",
        help="Dossier source contenant les classes a deplacer.",
    )
    parser.add_argument(
        "--target",
        default="/content/drive/MyDrive/Plantdataset",
        help="Dossier destination des classes canoniques.",
    )
    parser.add_argument(
        "--include-review",
        action="store_true",
        help="Inclure les lignes CSV marquees REVIEW (deconseille sans verification).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulation sans deplacement reel.",
    )
    parser.add_argument(
        "--report",
        default="merge_execution_report.csv",
        help="Nom du rapport CSV genere dans le dossier source.",
    )
    # En notebook (Colab/Jupyter), des arguments systeme (ex: -f kernel.json)
    # peuvent etre injectes automatiquement. On les ignore pour eviter un crash.
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[INFO] Arguments ignores (environnement notebook): {unknown}")
    return args


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv)
    source_root = Path(args.source)
    target_root = Path(args.target)
    report_path = source_root / args.report

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV introuvable: {csv_path}")
    if not source_root.exists():
        raise FileNotFoundError(f"Source introuvable: {source_root}")
    if not source_root.is_dir():
        raise NotADirectoryError(f"Source invalide (pas un dossier): {source_root}")

    print("\n=================== CONFIGURATION ===================")
    print(f"CSV               : {csv_path}")
    print(f"Source            : {source_root}")
    print(f"Destination       : {target_root}")
    print(f"include REVIEW    : {args.include_review}")
    print(f"Mode              : {'DRY-RUN' if args.dry_run else 'REEL'}")
    print("=====================================================\n")

    rows = load_mapping_rows(csv_path)
    class_to_canonical, skipped_rows = build_move_plan(rows, args.include_review)

    print(f"Lignes CSV lues                     : {len(rows)}")
    print(f"Classes mappees uniques             : {len(class_to_canonical)}")
    print(f"Lignes ignorees (type non autorise) : {len(skipped_rows)}")
    if skipped_rows:
        print("Exemples ignorees:")
        for line in skipped_rows[:5]:
            print(f"  - {line}")

    summary = preflight(source_root, target_root, class_to_canonical)
    print("\n=================== PREFLIGHT ===================")
    print(f"Classes source totales              : {summary['source_total_classes']}")
    print(f"Classes source presentes et mappees : {summary['mapped_existing_classes']}")
    print(f"Classes mappees absentes en source  : {summary['mapped_missing_classes']}")
    print(
        f"Classes canoniques deja presentes   : "
        f"{summary['existing_target_canonical_classes']}"
    )
    print("=================================================\n")

    execute_merge(
        source_root=source_root,
        target_root=target_root,
        class_to_canonical=class_to_canonical,
        report_path=report_path,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
