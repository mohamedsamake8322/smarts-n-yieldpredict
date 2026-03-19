"""
Scaffold BLIP2 disease JSON translations folders.

Usage:
  python scripts/scaffold_blip2_i18n.py --langs fr en tr sw ha ar zh ff bm wo

This copies all JSON files from the source folder (default: BLIP2)
into `BLIP2_i18n/<lang>/` so you can translate the values while keeping
the exact same filenames and JSON schema.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="+", required=True, help="Language codes (e.g. fr en ar sw ...)")
    parser.add_argument("--source-dir", default="BLIP2", help="Source folder containing original disease JSONs")
    parser.add_argument("--dest-root", default="BLIP2_i18n", help="Destination root folder for i18n JSONs")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing translated files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_dir = Path(args.source_dir)
    dest_root = Path(args.dest_root)

    if not source_dir.exists():
        raise SystemExit(f"Source directory not found: {source_dir}")

    json_files = sorted(source_dir.glob("*.json"))
    if not json_files:
        raise SystemExit(f"No JSON files found in source directory: {source_dir}")

    for lang in args.langs:
        lang_dir = dest_root / lang
        lang_dir.mkdir(parents=True, exist_ok=True)

        for json_path in json_files:
            dest_path = lang_dir / json_path.name
            if dest_path.exists() and not args.overwrite:
                continue
            shutil.copy2(json_path, dest_path)

        print(f"[OK] Scaffolded language: {lang_dir}")

    print("[DONE] BLIP2_i18n scaffolding complete.")


if __name__ == "__main__":
    main()

