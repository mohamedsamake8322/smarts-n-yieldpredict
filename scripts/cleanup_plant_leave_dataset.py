import shutil
from pathlib import Path
from PIL import Image

# Mapping minimal pour correspondre aux classes existantes
# Ajuster si nécessaire pour d'autres variantes de noms
RENAME_MAP = {
    'Pepper_Early_Blight': 'Pepper_Early_Blight',
    'Tomato_Mosaic': 'Tomato_Mosaic',
    'Pepper_Late_Blight': 'Pepper_Late_Blight',
    'Generic_Mites': 'Generic_Mites',
    'Corn_Northern_Leaf_Blight': 'Corn_Northern_Leaf_Blight',
    'Generic_Beetle': 'Generic_Beetle',
    'Cassava_Mosaic_Disease': 'Cassava_Mosaic_Disease',
    'Generic_Bollworm': 'Generic_Bollworm',
    'Generic_Stem_Borer': 'Generic_Stem_Borer',
    'Tomato_Fusarium': 'Tomato_Fusarium',
    # Ajouter d'autres variantes si nécessaire
    'Tomato_Early_Blight': 'Tomato_Early_Blight',
    'Tomato_Septoria': 'Tomato_Septoria',
    'Pepper_Septoria': 'Pepper_Septoria',
    'Pepper_Cearospora': 'Pepper_Cercospora',
}

# Classes à garantir
TARGET_CLASSES = [
    'Pepper_Early_Blight', 'Tomato_Mosaic', 'Pepper_Late_Blight', 'Generic_Mites',
    'Corn_Northern_Leaf_Blight', 'Generic_Beetle', 'Cassava_Mosaic_Disease',
    'Generic_Bollworm', 'Generic_Stem_Borer', 'Tomato_Fusarium'
]

# Helpers

def normalize_name(name: str) -> str:
    s = name.strip().replace(' ', '_').replace('-', '_')
    s = s.replace('__', '_')
    return RENAME_MAP.get(s, s)


def image_quality(path: Path):
    try:
        img = Image.open(path)
        w, h = img.size
        return w*h
    except Exception:
        return 0


def limit_class_size(class_dir: Path, max_images=1000):
    imgs = [p for p in class_dir.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png')]
    if len(imgs) <= max_images:
        return 0
    scored = [(image_quality(p), p) for p in imgs]
    scored.sort(key=lambda x: x[0], reverse=True)
    to_delete = [p for _, p in scored[max_images:]]
    for p in to_delete:
        p.unlink()
    return len(to_delete)


def process_dataset(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(exist_ok=True)
    stats = {'moved':0,'renamed':0,'merged':0,'limited':0}

    for class_path in src_dir.iterdir():
        if not class_path.is_dir():
            continue
        class_name = normalize_name(class_path.name)
        target_dir = dst_dir / class_name
        target_dir.mkdir(exist_ok=True)

        for img in class_path.glob('*'):
            if not img.is_file():
                continue
            if img.suffix.lower() not in ('.jpg','.jpeg','.png'):
                continue

            dest = target_dir / img.name
            if dest.exists():
                # éviter collisions noms
                dest = target_dir / f"{img.stem}_{img.stat().st_size}{img.suffix}"

            shutil.move(str(img), str(dest))
            stats['moved'] += 1

        if class_name != class_path.name:
            stats['renamed'] += 1

    # limiter chaque classe
    for target in dst_dir.iterdir():
        if target.is_dir():
            removed = limit_class_size(target, 1000)
            if removed:
                stats['limited'] += removed

    return stats


def main():
    src = Path(r"C:\smarts-n-yieldpredict.git\Plant_leave_diseases_dataset_with_augmentation")
    dst = Path(r"C:\smarts-n-yieldpredict.git\Plant_leave_diseases_dataset_with_augmentation_cleaned")

    if not src.exists():
        raise FileNotFoundError(f"Source directory not found: {src}")
    if dst.exists():
        print(f"Destination exists, fusion en cours: {dst}")

    stats = process_dataset(src, dst)

    # ajouter trainsfer des classes manquantes si besoin depuis le dataset previous
    # pour enrichir, on peut fusionner classes similaires du dataset nettoyé précédent

    print('=== Rapport ===')
    print(f"moved: {stats['moved']}")
    print(f"renamed: {stats['renamed']}")
    print(f"limited removed: {stats['limited']}")


if __name__ == '__main__':
    main()
