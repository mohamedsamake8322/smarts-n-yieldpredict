from pathlib import Path
import shutil
import argparse

CLASS_MAP = {
    "ALS": "Class 0 Alternaria Leaf Spot",
    "CLS": "Class 1 - Cercospora leaf spot",
    "DM": "Class 2- Downy Mildew",
    "HLT": "Class 3 - Healthy",
    "H": "Class 3 - Healthy",
    "LCV": "Class 4 - Leaf Curly Virus",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}


def find_prefix(filename: str) -> str | None:
    stem = Path(filename).stem
    # Exemple: ALS_189_JPG.rf.3309fdb03ae0c6855921ce85440fcffa
    parts = stem.split("_")
    if parts:
        prefix = parts[0].upper()
        if prefix in CLASS_MAP:
            return prefix
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Repartir des images dans des dossiers de classe selon leur préfixe de nom."
    )
    parser.add_argument(
        "source_dir",
        help="Dossier source contenant les images (ex: C:\\Downloads\\OKRA_DISEASE_IDENTIFICATION.v1i.yolov8\\train\\images)",
    )
    parser.add_argument(
        "dest_root",
        nargs="?",
        default=None,
        help="Dossier racine de destination. Si omitted, le tri se fait dans source_dir."
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copier les fichiers au lieu de les déplacer.",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir).expanduser().resolve()
    if args.dest_root:
        dest_root = Path(args.dest_root).expanduser().resolve()
    else:
        dest_root = source_dir

    if not source_dir.exists() or not source_dir.is_dir():
        raise SystemExit(f"Le dossier source n'existe pas ou n'est pas un dossier: {source_dir}")

    dest_root.mkdir(parents=True, exist_ok=True)

    moved = 0
    skipped = 0
    unknown = []

    for path in sorted(source_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTS:
            continue

        prefix = find_prefix(path.name)
        if prefix is None:
            unknown.append(path.name)
            skipped += 1
            continue

        class_name = CLASS_MAP[prefix]
        dest_dir = dest_root / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / path.name

        if args.copy:
            shutil.copy2(path, dest_path)
        else:
            shutil.move(str(path), str(dest_path))

        moved += 1

    print(f"Images triées : {moved}")
    if skipped:
        print(f"Images ignorées (préfixe non reconnu) : {skipped}")
        for name in unknown[:20]:
            print(f"  - {name}")
        if len(unknown) > 20:
            print(f"  ... et {len(unknown) - 20} autres")


if __name__ == "__main__":
    main()
