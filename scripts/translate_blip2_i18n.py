"""
Automatic translation of BLIP2 disease JSON files.

This script translates English JSON files from BLIP2/ to target languages
using Helsinki-NLP Opus-MT models.

Usage:
  python scripts/translate_blip2_i18n.py --langs tr fr sw ha ar zh

Requirements:
  pip install transformers torch
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List
import time

import torch
from transformers import pipeline


# Mapping of language codes to Helsinki-NLP Opus-MT models
OPUS_MT_MODELS = {
    "tr": "Helsinki-NLP/opus-mt-en-tr",
    "fr": "Helsinki-NLP/opus-mt-en-fr",
    "sw": "Helsinki-NLP/opus-mt-en-swc",
    "ha": "Helsinki-NLP/opus-mt-en-ha",
    "ar": "Helsinki-NLP/opus-mt-en-ar",
    "zh": "Helsinki-NLP/opus-mt-en-zh",
    "ff": "Helsinki-NLP/opus-mt-en-ff",
    "bm": "Helsinki-NLP/opus-mt-en-bm",
    "wo": "Helsinki-NLP/opus-mt-en-wo",
}

# Mapping of language codes to MBART language codes (legacy)
LANG_CODES = {
    "tr": "tr_TR",  # Turkish
    "fr": "fr_XX",  # French
    "sw": "sw_XX",  # Swahili
    "ha": "ha_XX",  # Hausa
    "ar": "ar_AR",  # Arabic
    "zh": "zh_CN",  # Chinese
    "ff": "ff_XX",  # Pulaar (might not be supported)
    "bm": "bm_XX",  # Bambara (might not be supported)
    "wo": "wo_XX",  # Wolof (might not be supported)
}

MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"

RETRY_DELAY_BASE = 2
MAX_RETRIES = 6


def get_hf_auth_token() -> str:
    """Read Hugging Face token from environment variables."""
    return os.environ.get("HF_HUB_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN") or ""


def assert_torch_version():
    """Ensure torch is compatible with recent transformers vulnerability policy."""
    try:
        original = torch.__version__
        major, minor, *rest = [int(x) for x in original.split(".") if x.isdigit()]
    except Exception:
        return

    if (major, minor) < (2, 6):
        print("\n❌ Incompatible torch version detected:", torch.__version__)
        print("⚠️  transformers requires torch>=2.6 for safe model loading due CVE-2025-32434.")
        print("   Upgrade command: pip install -U 'torch>=2.6' transformers sentencepiece")
        sys.exit(1)


def load_translation_pipeline(model_name: str, use_gpu: bool, src_lang: str = None, tgt_lang: str = None):
    """Load translation pipeline with retry and optional fallback via token."""
    device = 0 if use_gpu else -1
    token = get_hf_auth_token()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            kwargs = {"model": model_name, "device": device}
            if token:
                kwargs["use_auth_token"] = token

            if model_name.startswith("facebook/m2m100") or model_name.startswith("facebook/mbart"):
                if src_lang:
                    kwargs["src_lang"] = src_lang
                if tgt_lang:
                    kwargs["tgt_lang"] = tgt_lang

            translator = pipeline("translation", **kwargs)
            return translator
        except Exception as e:
            wait = RETRY_DELAY_BASE ** min(attempt, 4)
            print(f"  ⚠️  Attempt {attempt}/{MAX_RETRIES} failed for {model_name} (src={src_lang} tgt={tgt_lang}): {e}")
            if attempt == MAX_RETRIES:
                raise
            print(f"  ⏳ Retrying in {wait}s... (HUGGINGFACE_TOKEN set: {'yes' if bool(token) else 'no'})")
            time.sleep(wait)

    raise RuntimeError(f"Failed to load model pipeline for {model_name}")


def translate_text(text: str, translator, tgt_lang: str) -> str:
    """Translate a single text string using Opus-MT model."""
    if not text or not isinstance(text, str):
        return text
    try:
        result = translator(text, max_length=512)
        return result[0]['translation_text']
    except Exception as e:
        print(f"  ⚠️  Translation error for '{text[:50]}...': {e}")
        return text  # Fallback to original


def translate_recursive(data: Any, translator, lang_code: str, fields_to_translate: List[str]) -> Any:
    """Recursively translate text fields in JSON data."""
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            if key in fields_to_translate and isinstance(value, str):
                result[key] = translate_text(value, translator, lang_code)
            elif key == "susceptibility" and isinstance(value, dict):
                # Special handling for susceptibility dict
                result[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, list):
                        result[key][sub_key] = [translate_text(item, translator, lang_code) if isinstance(item, str) else item for item in sub_value]
                    else:
                        result[key][sub_key] = sub_value
            else:
                result[key] = translate_recursive(value, translator, lang_code, fields_to_translate)
        return result
    elif isinstance(data, list):
        return [translate_recursive(item, translator, lang_code, fields_to_translate) for item in data]
    else:
        return data


def translate_json_file(source_path: Path, target_path: Path, translator, lang_code: str):
    """Translate a single JSON file."""
    print(f"  📄 Processing: {source_path.name}...", end=" ", flush=True)
    
    try:
        with open(source_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Fields to translate (text content)
        fields_to_translate = [
            "description",
            "symptoms_and_damage",
            "disease_cycle_and_spread",
            "favorable_conditions",
            "pathogen_characteristics",
            "monitoring",
            "management",
            "prevention",
            "hosts",  # Plant names might be kept in English, but translate if needed
        ]

        translated_data = translate_recursive(data, translator, lang_code, fields_to_translate)

        # Ensure target directory exists
        target_path.parent.mkdir(parents=True, exist_ok=True)

        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(translated_data, f, indent=2, ensure_ascii=False)

        disease_name = translated_data.get("disease", "Unknown")
        print(f"✅ [{disease_name}]")
    except Exception as e:
        print(f"❌ ERROR: {e}")


def main():
    import time
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="+", required=True, help="Language codes to translate to")
    parser.add_argument("--source-dir", default="BLIP2", help="Source directory with English JSONs")
    parser.add_argument("--target-root", default="BLIP2_i18n", help="Target root directory for translations")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    target_root = Path(args.target_root)

    assert_torch_version()

    print("\n" + "="*70)
    print("🌍 BLIP2 MULTILINGUAL TRANSLATION SCRIPT")
    print("="*70)
    print(f"✓ Source directory:  {source_dir.absolute()}")
    print(f"✓ Target root:       {target_root.absolute()}")
    print(f"✓ Languages:         {', '.join(args.langs)}")
    print("="*70 + "\n")

    if not source_dir.exists():
        raise SystemExit(f"❌ Source directory not found: {source_dir}")

    json_files = sorted(source_dir.glob("*.json"))
    if not json_files:
        raise SystemExit(f"❌ No JSON files found in {source_dir}")

    total_translated = 0
    total_skipped = 0

    for lang in args.langs:
        lang_display = f"{lang.upper()}"

        # prefer Opus-MT per-language model, fallback to m2m100 if not available
        if lang in OPUS_MT_MODELS:
            model_name = OPUS_MT_MODELS[lang]
            src_lang, tgt_lang = None, None
        else:
            print(f"⚠️  Warning: No Opus model defined for '{lang}'. Using fallback m2m100.")
            model_name = "facebook/m2m100_418M"
            src_lang = "en"
            tgt_lang = lang

        print(f"\n{'─'*70}")
        print(f"🔄 Translating to {lang_display} (model: {model_name})")
        print(f"{'─'*70}")

        # Load translator for this language
        try:
            print(f"  ⏳ Loading translator...", end=" ", flush=True)
            load_start = time.time()
            translator = load_translation_pipeline(model_name, torch.cuda.is_available(), src_lang=src_lang, tgt_lang=tgt_lang)
            load_time = time.time() - load_start
            print(f"✅ ({load_time:.1f}s)")
        except Exception as e:
            print(f"❌ Failed to load translator for {lang}: {e}")
            print(f"  ⏭️  Skipping {lang} due to model loading error")
            continue

        lang_dir = target_root / lang
        lang_dir.mkdir(parents=True, exist_ok=True)

        lang_translated = 0
        lang_skipped = 0
        start_lang_time = time.time()

        for idx, json_path in enumerate(json_files, 1):
            target_path = lang_dir / json_path.name
            if target_path.exists():
                print(f"  ⏭️  [{idx:3d}/{len(json_files)}] SKIP (exists): {json_path.name}")
                lang_skipped += 1
                total_skipped += 1
                continue

            translate_json_file(json_path, target_path, translator, lang)
            lang_translated += 1
            total_translated += 1

        elapsed = time.time() - start_lang_time
        print(f"\n  📈 Summary for {lang_display}:")
        print(f"     • Translated: {lang_translated:3d} files")
        print(f"     • Skipped:    {lang_skipped:3d} files (already exist)")
        print(f"     • Time:       {elapsed:.1f}s")
        if lang_translated > 0:
            print(f"     • Speed:      {lang_translated / elapsed:.1f} files/sec")

        # Clean up memory after each language to avoid OOM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        del translator

    print("\n" + "="*70)
    print(f"✅ TRANSLATION COMPLETE")
    print("="*70)
    print(f"📊 Total translated: {total_translated} files")
    print(f"⏭️  Total skipped:    {total_skipped} files")
    print(f"📁 Output location:  {target_root.absolute()}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()