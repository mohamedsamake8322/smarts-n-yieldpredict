import os
import shutil
import pandas as pd
from pathlib import Path

# Chemins utilisateurs
SOURCE_ROOT = Path(r'C:\Downloads\Plantdataset')
TARGET_ROOT = Path(r'C:\Downloads\Plantdataset_organized')
CLASS_DISTRIBUTION_XLSX = SOURCE_ROOT / 'class_distribution (1).xlsx'


def slugify(s: str):
    return (s.strip().lower().replace(' ', '_').replace('-', '_').replace('/', '_').replace('(', '').replace(')', '').replace('__', '_'))


def load_class_distribution(xls_path: Path):
    df = pd.read_excel(xls_path, sheet_name=0)
    df.columns = [c.strip() for c in df.columns]

    rows = []
    current_crop = None
    current_primary = None

    for _, r in df.iterrows():
        crop = r.get('Crop') if not pd.isna(r.get('Crop')) else None
        primary = r.get('Primary Class') if not pd.isna(r.get('Primary Class')) else None
        secondary_abbr = r.get('Secondary Class Abbrv.') if not pd.isna(r.get('Secondary Class Abbrv.')) else None
        secondary_full = r.get('Secondary Class Full Form') if not pd.isna(r.get('Secondary Class Full Form')) else None
        sample = r.get('Sample Size') if not pd.isna(r.get('Sample Size')) else None

        if crop:
            current_crop = str(crop).strip()

        if primary:
            current_primary = str(primary).strip()

        if current_crop is None or current_primary is None:
            continue

        if secondary_abbr is None and current_primary and current_primary.lower() == 'healthy':
            secondary_abbr = 'healthy'
            secondary_full = 'Healthy'

        if secondary_abbr is None:
            continue

        # 57 classes attendus
        rows.append({
            'crop': current_crop,
            'primary': current_primary,
            'secondary_abbr': str(secondary_abbr).strip(),
            'secondary_full': str(secondary_full).strip() if secondary_full else str(secondary_abbr).strip(),
            'sample_size': int(sample) if sample is not None else None,
            'source_folder': f"{slugify(current_crop)}__{str(secondary_abbr).strip().lower()}"
        })

    return pd.DataFrame(rows)


def organize_dataset():
    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(f"Source root not found: {SOURCE_ROOT}")

    if not CLASS_DISTRIBUTION_XLSX.exists():
        raise FileNotFoundError(f"Excel class distribution not found: {CLASS_DISTRIBUTION_XLSX}")

    mapping = load_class_distribution(CLASS_DISTRIBUTION_XLSX)
    mapping = mapping.drop_duplicates(subset=['source_folder'])

    # Inspect what folders exist
    src_folders = sorted([d.name for d in SOURCE_ROOT.iterdir() if d.is_dir() and d.name.startswith('part_')])
    print('Part folders:', src_folders)
    print('Class mapping rows:', len(mapping))

    # Prebuild a dictionary pour lookup
    class_map = {row.source_folder: row for row in mapping.itertuples(index=False)}

    # Preparations target
    if TARGET_ROOT.exists():
        print('Target already exists. Contents may be overwritten in this run.')
    TARGET_ROOT.mkdir(parents=True, exist_ok=True)

    manifest = []

    for part in sorted(SOURCE_ROOT.iterdir()):
        if not part.is_dir() or not part.name.startswith('part_'):
            continue

        for class_dir in sorted(part.iterdir()):
            if not class_dir.is_dir():
                continue

            folder_key = class_dir.name.lower()
            if folder_key not in class_map:
                print(f'⚠️  Observed class folder not in mapping: {folder_key}')
                continue

            row = class_map[folder_key]
            crop = slugify(row.crop)
            primary = slugify(row.primary)
            sec_abbr = slugify(row.secondary_abbr)
            sec_full = slugify(row.secondary_full)

            target_folder = TARGET_ROOT / crop / primary / sec_full
            target_folder.mkdir(parents=True, exist_ok=True)

            # Copy images
            count = 0
            for img in sorted(class_dir.iterdir()):
                if img.is_file() and img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff']:
                    dst = target_folder / img.name
                    # if duplicate filename across parts, suffix Automatically
                    if dst.exists():
                        stem, ext = os.path.splitext(img.name)
                        i = 1
                        while True:
                            dst2 = target_folder / f"{stem}_{i}{ext}"
                            if not dst2.exists():
                                dst = dst2
                                break
                            i += 1
                    shutil.copy2(img, dst)
                    manifest.append({
                        'src': str(img),
                        'dst': str(dst),
                        'crop': row.crop,
                        'primary': row.primary,
                        'secondary_abbr': row.secondary_abbr,
                        'secondary_full': row.secondary_full,
                        'target_folder': str(target_folder)
                    })
                    count += 1

            print(f'Copied {count:4d} images {class_dir.name} => {target_folder}')

    # Save manifest
    manifest_path = TARGET_ROOT / 'organize_manifest.csv'
    pd.DataFrame(manifest).to_csv(manifest_path, index=False, encoding='utf-8-sig')
    print(f'Manifest saved: {manifest_path}')

    summary = mapping.groupby(['crop', 'primary', 'secondary_abbr', 'secondary_full']).size().reset_index(name='classes')
    summary_path = TARGET_ROOT / 'class_summary.csv'
    summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f'Summary saved: {summary_path}')

    print('Organisation terminée avec succès')


if __name__ == '__main__':
    organize_dataset()
