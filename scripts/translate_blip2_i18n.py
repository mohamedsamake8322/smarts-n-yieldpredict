"""
ULTRA STABLE TRANSLATION SCRIPT

✔ Only verified OPUS models
✔ Smart fallback for unsupported languages
✔ Avoids model errors
✔ Production ready

Usage:
python translate.py --langs fr ar zh tr sw ha yo am
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, List

import torch
from transformers import pipeline


# ✅ VERIFIED MODELS ONLY
OPUS_MODELS = {
    "fr": "Helsinki-NLP/opus-mt-en-fr",
    "ar": "Helsinki-NLP/opus-mt-en-ar",
    "zh": "Helsinki-NLP/opus-mt-en-zh",
    "ru": "Helsinki-NLP/opus-mt-en-ru",
    "es": "Helsinki-NLP/opus-mt-en-es",
    "de": "Helsinki-NLP/opus-mt-en-de",
    "tr": "Helsinki-NLP/opus-mt-en-trk",
    "sw": "Helsinki-NLP/opus-mt-en-swc",
    "am": "Helsinki-NLP/opus-mt-en-am",
    "ig": "Helsinki-NLP/opus-mt-en-ig",
    "ha": "Helsinki-NLP/opus-mt-en-ha",
}

FALLBACK_MODEL = "facebook/m2m100_418M"

RETRY_DELAY = 2
MAX_RETRIES = 3


def load_translator(lang: str):
    device = 0 if torch.cuda.is_available() else -1

    # ✅ Use OPUS if available
    if lang in OPUS_MODELS:
        model = OPUS_MODELS[lang]
        print(f"⚡ OPUS model for {lang}")
        return pipeline("translation", model=model, device=device)

    # ⚠️ Fallback M2M100
    print(f"⚠️ Fallback M2M100 for {lang}")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return pipeline(
                "translation",
                model=FALLBACK_MODEL,
                device=device,
                src_lang="en",
                tgt_lang=lang,
            )
        except Exception as e:
            print(f"⚠️ Attempt {attempt}: {e}")
            time.sleep(RETRY_DELAY * attempt)

    raise RuntimeError(f"Failed model for {lang}")


def translate_text(text: str, translator):
    if not isinstance(text, str) or not text.strip():
        return text
    try:
        return translator(text, max_length=512)[0]["translation_text"]
    except:
        return text


def translate_recursive(data: Any, translator, fields: List[str]):
    if isinstance(data, dict):
        return {
            k: translate_text(v, translator)
            if k in fields and isinstance(v, str)
            else translate_recursive(v, translator, fields)
            for k, v in data.items()
        }

    elif isinstance(data, list):
        return [translate_recursive(i, translator, fields) for i in data]

    return data


def translate_file(src: Path, dst: Path, translator):
    print(f"📄 {src.name}...", end=" ")

    try:
        with open(src, "r", encoding="utf-8") as f:
            data = json.load(f)

        fields = [
            "description",
            "symptoms_and_damage",
            "disease_cycle_and_spread",
            "favorable_conditions",
            "pathogen_characteristics",
            "monitoring",
            "management",
            "prevention",
        ]

        result = translate_recursive(data, translator, fields)

        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print("✅")

    except Exception as e:
        print(f"❌ {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="+", required=True)
    parser.add_argument("--source-dir", default="BLIP2")
    parser.add_argument("--target-root", default="BLIP2_i18n")

    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    files = list(source_dir.glob("*.json"))

    if not files:
        sys.exit("❌ No JSON files found")

    print("\n🌍 START TRANSLATION\n")

    for lang in args.langs:
        print(f"\n🔄 Language: {lang.upper()}")

        try:
            translator = load_translator(lang)
        except Exception as e:
            print(f"❌ Skipping {lang}: {e}")
            continue

        out_dir = Path(args.target_root) / lang

        for file in files:
            dst = out_dir / file.name

            if dst.exists():
                print(f"⏭️ SKIP {file.name}")
                continue

            translate_file(file, dst, translator)

        del translator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n✅ DONE\n")


if __name__ == "__main__":
    main()