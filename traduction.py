"""
Robust multilingual translation of BLIP2 JSON files.

✔ Automatic detection of all text fields in JSON
✔ Uses OPUS MT models where available
✔ Falls back to M2M100 for missing languages
✔ Retry system included
✔ Works for African + selected international languages

Usage:
python translate_blip2_i18n.py --source-dir BLIP2 --target-root BLIP2_i18n
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------------
# Language models configuration
# -----------------------------
# OPUS MT models (preferred)
OPUS_MT_MODELS = {
    "fr": "Helsinki-NLP/opus-mt-en-fr",
    "es": "Helsinki-NLP/opus-mt-en-es",
    "de": "Helsinki-NLP/opus-mt-en-de",
    "ar": "Helsinki-NLP/opus-mt-en-ar",
    "zh": "Helsinki-NLP/opus-mt-en-zh",
    "ru": "Helsinki-NLP/opus-mt-en-ru",
    "tr": "Helsinki-NLP/opus-mt-en-tr",
    "sw": "Helsinki-NLP/opus-mt-en-sw",
    "ha": "Helsinki-NLP/opus-mt-en-ha",
    "ig": "Helsinki-NLP/opus-mt-en-ig",
    "am": "Helsinki-NLP/opus-mt-en-am",
}

# Fallback multilingual model
FALLBACK_MODEL = "facebook/m2m100_418M"

RETRY_DELAY_BASE = 2
MAX_RETRIES = 5

# -----------------------------
# Utility functions
# -----------------------------

def assert_torch_version():
    try:
        version = torch.__version__
        major, minor = map(int, version.split(".")[:2])
        if (major, minor) < (2, 6):
            print(f"\n⚠️ Warning: torch {version} < 2.6 (safe to continue)\n")
    except:
        pass


def load_translation_pipeline(tgt_lang: str, use_gpu: bool = False):
    """
    Load the translation model pipeline for the given target language.
    Tries OPUS MT first, fallback to M2M100 if necessary.
    """
    device = 0 if use_gpu else -1

    model_name = OPUS_MT_MODELS.get(tgt_lang, FALLBACK_MODEL)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            translator = pipeline(
                "translation",
                model=model_name,
                device=device,
                src_lang="en" if model_name != FALLBACK_MODEL else None,
                tgt_lang=tgt_lang if model_name != FALLBACK_MODEL else None,
            )
            print(f"⚡ Using {('OPUS' if model_name != FALLBACK_MODEL else 'M2M100')} model for {tgt_lang}")
            return translator

        except Exception as e:
            wait = RETRY_DELAY_BASE ** attempt
            print(f"⚠️ Attempt {attempt}/{MAX_RETRIES} failed for {tgt_lang}: {e}")
            if attempt == MAX_RETRIES:
                raise
            time.sleep(wait)


def translate_text(text: str, translator) -> str:
    if not isinstance(text, str) or not text.strip():
        return text
    try:
        result = translator(text, max_length=512)
        return result[0]["translation_text"]
    except Exception as e:
        print(f"⚠️ Translation error: {text[:40]}... -> {e}")
        return text


def translate_recursive(data: Any, translator) -> Any:
    """
    Recursively translate all string fields in JSON.
    Works for dicts and lists.
    """
    if isinstance(data, dict):
        return {k: translate_recursive(v, translator) for k, v in data.items()}
    elif isinstance(data, list):
        return [translate_recursive(i, translator) for i in data]
    elif isinstance(data, str):
        return translate_text(data, translator)
    else:
        return data


def translate_json(source: Path, target: Path, translator):
    print(f"📄 {source.name}...", end=" ")
    try:
        with open(source, "r", encoding="utf-8") as f:
            data = json.load(f)

        result = translate_recursive(data, translator)

        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print("✅")
    except Exception as e:
        print(f"❌ {e}")


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="+", default=list(OPUS_MT_MODELS.keys()))
    parser.add_argument("--source-dir", default="BLIP2")
    parser.add_argument("--target-root", default="BLIP2_i18n")

    args = parser.parse_args()

    assert_torch_version()

    source_dir = Path(args.source_dir)
    target_root = Path(args.target_root)

    if not source_dir.exists():
        sys.exit(f"❌ Source folder not found: {source_dir}")

    files = list(source_dir.glob("*.json"))
    if not files:
        sys.exit("❌ No JSON files found")

    print("\n🌍 MULTILINGUAL TRANSLATION START\n")

    for lang in args.langs:
        print(f"\n🔄 Language: {lang.upper()}")

        try:
            translator = load_translation_pipeline(lang, torch.cuda.is_available())
        except Exception as e:
            print(f"❌ Model load failed for {lang}: {e}")
            continue

        out_dir = target_root / lang
        for file in files:
            target = out_dir / file.name
            if target.exists():
                print(f"⏭️ SKIP {file.name}")
                continue
            translate_json(file, target, translator)

        del translator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n✅ DONE\n")


if __name__ == "__main__":
    main()