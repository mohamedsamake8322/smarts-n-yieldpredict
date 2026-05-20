import argparse
import os
from pathlib import Path
import shutil


def read_class_map(classes_txt_path: Path):
    if not classes_txt_path.exists():
        return {}
    class_map = {}
    with classes_txt_path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if not parts:
                continue
            idx = parts[0].strip()
            label = parts[1].strip() if len(parts) > 1 else idx
            if idx.isdigit():
                class_map[int(idx)] = label
    return class_map


def parse_split_file(split_file_path: Path):
    rows = []
    with split_file_path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            parts = text.split()
            if len(parts) < 2:
                continue
            filename, class_id = parts[0], parts[1]
            if not class_id.lstrip('-').isdigit():
                continue
            rows.append((filename, int(class_id)))
    return rows


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description='Organize IP102 images in folder structure by split/class')
    parser.add_argument('--ip102-root', type=Path, default=Path(r'C:\smarts-n-yieldpredict.git\Camera Roll\ip102_v1.1'), help='Chemin vers le dossier ip102_v1.1')
    parser.add_argument('--out-root', type=Path, default=Path(r'C:\smarts-n-yieldpredict.git\Camera Roll\ip102_v1.1\organized'), help='Dossier de sortie organisé')
    parser.add_argument('--jpegimages-dir', type=Path, default=Path(r'C:\smarts-n-yieldpredict.git\Camera Roll\JPEGImages'), help='Chemin alternatif pour les images si non trouvées dans ip102_root/images')
    parser.add_argument('--classes-txt', type=Path, default=Path(r'C:\smarts-n-yieldpredict.git\Camera Roll\classes.txt'), help='Fichier classes.txt pour noms humains des labels')
    parser.add_argument('--split-files', nargs='+', default=['train.txt', 'val.txt', 'test.txt'], help='Fichiers split à traiter')
    parser.add_argument('--auto-scan-root', action='store_true', help='Scanner automatiquement *.txt dans parent de ip102-root en plus des splits explicites')
    parser.add_argument('--dry-run', action='store_true', help='Afficher les actions sans copier')
    parser.add_argument('--move', action='store_true', help='Déplacer les fichiers au lieu de copier')
    args = parser.parse_args()

    images_dir = args.ip102_root / 'images'
    jpeg_dir = args.jpegimages_dir

    class_map = read_class_map(args.classes_txt)

    if not images_dir.exists() or not images_dir.is_dir():
        print(f"Attention : dossier d'images introuvable dans ip102_root : {images_dir}")
    if not jpeg_dir.exists() or not jpeg_dir.is_dir():
        print(f"Attention : dossier JPEGImages introuvable : {jpeg_dir}")

    # Traitement des fichiers splits demandés + fallback vers parent (Camera Roll)
    parent_dir = args.ip102_root.parent
    split_candidates = []
    for split in args.split_files:
        path = args.ip102_root / split
        if not path.exists():
            path = parent_dir / split
        if path.exists():
            split_candidates.append((split, path))
        else:
            print(f"Fichier split introuvable (ignoré) : {split} (ni dans ip102_root ni dans parent)")

    if args.auto_scan_root:
        for txtf in parent_dir.glob('*.txt'):
            if txtf.name.lower() in ('classes.txt', 'dataset.txt'):
                continue
            if any(txtf.name == existing for existing, _ in split_candidates):
                continue
            split_candidates.append((txtf.name, txtf))

    if not split_candidates:
        print("Aucun split trouvé. Vérifiez les chemins et les fichiers train/val/test.")
        return

    for split_name, split_path in split_candidates:
        target_split_dir = args.out_root / split_name.replace('.txt', '')
        ensure_dir(target_split_dir)

        rows = parse_split_file(split_path)
        print(f"Traitement split {split_name}: {len(rows)} lignes (source: {split_path})")

        for filename, class_id in rows:
            # On accepte un chemin relatif ou simple, plusieurs répertoires source
            candidates = []
            if os.path.isabs(filename):
                candidates.append(Path(filename))
            else:
                candidates.extend([
                    images_dir / filename,
                    jpeg_dir / filename,
                    parent_dir / filename,
                    args.ip102_root / filename,
                ])

            src = next((p for p in candidates if p.exists() and p.is_file()), None)
            if src is None:
                print(f" [!] image introuvable : {filename} (vérifier ip102_root/images, JPEGImages, etc.)")
                continue

            class_label = class_map.get(class_id + 1, class_map.get(class_id, str(class_id)))
            class_label_safe = ''.join(c for c in class_label if c.isalnum() or c in ' _-').strip() or str(class_id)
            dst_dir = target_split_dir / class_label_safe
            ensure_dir(dst_dir)
            dst_file = dst_dir / src.name

            action = 'Déplacer' if args.move else 'Copier'
            print(f" {action} {src} -> {dst_file}")
            if not args.dry_run:
                if args.move:
                    shutil.move(str(src), str(dst_file))
                else:
                    shutil.copy2(str(src), str(dst_file))

    print('Terminé.')


if __name__ == '__main__':
    main()
