"""
Validate BLIP2 i18n JSON schema.

Checks that every translated JSON has the same top-level keys as the source.
Optionally checks that every target file exists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="+", required=True, help="Language codes to validate (e.g. fr en ar ...)")
    parser.add_argument("--source-dir", default="BLIP2", help="Source folder containing original disease JSONs")
    parser.add_argument("--dest-root", default="BLIP2_i18n", help="Destination i18n root folder")
    parser.add_argument("--strict-files", action="store_true", help="Fail if any translated file is missing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_dir = Path(args.source_dir)
    dest_root = Path(args.dest_root)

    if not source_dir.exists():
        raise SystemExit(f"Source directory not found: {source_dir}")

    base_files = sorted(source_dir.glob("*.json"))
    if not base_files:
        raise SystemExit(f"No JSON files found in source directory: {source_dir}")

    base_keys_by_file: dict[str, set[str]] = {}
    for base_path in base_files:
        with base_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        base_keys_by_file[base_path.name] = set(data.keys())

    ok = True

    for lang in args.langs:
        lang_dir = dest_root / lang
        if not lang_dir.exists():
            print(f"[SKIP] Missing language folder: {lang_dir}")
            ok = False
            continue

        lang_ok = True
        for base_path in base_files:
            target_path = lang_dir / base_path.name
            if not target_path.exists():
                if args.strict_files:
                    print(f"[FAIL] {lang}: missing file {target_path.name}")
                    lang_ok = False
                continue

            with target_path.open("r", encoding="utf-8") as f:
                translated = json.load(f)

            expected_keys = base_keys_by_file[base_path.name]
            actual_keys = set(translated.keys())
            if actual_keys != expected_keys:
                print(f"[FAIL] {lang}: schema mismatch for {base_path.name}")
                print(f"        expected keys: {sorted(expected_keys)}")
                print(f"        actual keys:   {sorted(actual_keys)}")
                lang_ok = False

        if lang_ok:
            print(f"[OK] {lang}: schema looks consistent")
        else:
            ok = False

    if not ok:
        raise SystemExit(1)

    print("[DONE] BLIP2_i18n schema validation complete.")


if __name__ == "__main__":
    main()

