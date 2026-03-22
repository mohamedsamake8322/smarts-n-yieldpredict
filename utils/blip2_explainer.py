"""BLIP-2 explanation helper.

This module provides utilities to:
- Load disease knowledge from JSON files (symptoms, causes, management).
- Build a structured prompt for BLIP-2.
- Load BLIP-2 (Salesforce/blip2-flan-t5-*) and generate a short explanation.

The idea is that your existing Swin-based classifier identifies a disease class,
then we load the corresponding JSON information and feed it to a VLM to produce
an explanation in natural language.

The BLIP-2 model is **NOT** used for classification, only for text generation
based on the structured prompt.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
import torch


def _normalize_label(label: str) -> str:
    """Normalize a disease label / filename to support fuzzy matching."""
    s = label.lower().strip()
    # Fix common filename artifacts like "... ,json.json" or "... ,json"
    # by removing any embedded/ending "json" token left after extension stripping.
    s = s.replace(".json", "")
    s = s.replace(",json", "")
    s = s.replace("json,", "")
    # Replace common separators with spaces
    s = re.sub(r"[\-_]+", " ", s)
    # Remove common punctuation
    s = re.sub(r"[\.,()\[\]{}'\"]", "", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s)
    # If "json" remained as a trailing token, remove it
    s = re.sub(r"\bjson\b", "", s).strip()
    return s


@lru_cache(maxsize=8)
def _build_disease_index(library_dir: Path) -> Dict[str, Path]:
    """Build a mapping of normalized names -> JSON file path."""
    library_dir = Path(library_dir)
    mapping: Dict[str, Path] = {}

    if not library_dir.exists():
        return mapping

    for json_path in sorted(library_dir.glob("*.json")):
        key = _normalize_label(json_path.stem)
        mapping[key] = json_path
    return mapping


def _find_best_json_path(
    predicted_label: str,
    library_dir: Path = Path("BLIP2"),
    allow_fuzzy: bool = False,
) -> Optional[Path]:
    """Find the best matching JSON file for a predicted disease label.

    By default, this uses an exact normalized match (safe). If allow_fuzzy=True,
    it will fall back to looser matching heuristics.
    """
    if not predicted_label:
        return None

    idx = _normalize_label(predicted_label)
    mapping = _build_disease_index(library_dir)

    # Exact (normalized) match – this is the safest strategy.
    if idx in mapping:
        return mapping[idx]

    if not allow_fuzzy:
        return None

    # Fuzzy fallback: try to match by prefix
    for key, path in mapping.items():
        if idx.startswith(key) or key.startswith(idx):
            return path

    # Fuzzy fallback: try words overlap
    idx_words = set(idx.split())
    best_path = None
    best_score = 0
    for key, path in mapping.items():
        score = len(idx_words & set(key.split()))
        if score > best_score:
            best_score = score
            best_path = path

    return best_path if best_score > 0 else None


def load_disease_info(
    predicted_label: str,
    library_dir: Path = Path("BLIP2"),
    allow_fuzzy: bool = False,
    language_code: str = "en",
) -> Dict[str, Any]:
    """Load the disease JSON info for a predicted class.

    If no matching JSON is found, returns a minimal dictionary.
    """
    # Si un dossier de traductions existe, on le tente en priorité, puis on retombe
    # sur le dossier original (BLIP2 ou BLIP2_normalized) si le fichier n'existe pas.
    translations_root = None
    try:
        # Prefer explicit config if available
        from config import BLIP2_I18N_DIR

        translations_root = Path(BLIP2_I18N_DIR)
    except Exception:
        translations_root = Path(os.environ.get("BLIP2_I18N_ROOT", "BLIP2_i18n"))
    dirs_to_try: List[Path] = []
    language_code = (language_code or "en").lower()

    if language_code and language_code not in {"en", ""}:
        candidate = translations_root / language_code
        if candidate.exists() and candidate.is_dir():
            dirs_to_try.append(candidate)

    # Support bonus fallback directories in priority order
    dirs_to_try.append(Path("BLIP2_normalized"))
    dirs_to_try.append(Path(library_dir))

    json_path: Optional[Path] = None
    for lib in dirs_to_try:
        json_path = _find_best_json_path(predicted_label, lib, allow_fuzzy=allow_fuzzy)
        if json_path is not None and json_path.exists():
            break

    if json_path is None or not json_path.exists():
        return {
            "disease": predicted_label,
            "symptoms": [],
            "cause": "",
            "management": [],
            "prevention": [],
            "hosts": [],
            "scientific_name": "",
            "description": "",
        }

    with open(json_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            # Expose source path for debugging (affiche la langue utilisée)
            data["_source_file"] = str(json_path)
        except Exception:
            return {
                "disease": predicted_label,
                "symptoms": [],
                "cause": "",
                "management": [],
                "prevention": [],
                "hosts": [],
                "scientific_name": "",
                "description": "",
            }

    def _to_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(x).strip() for x in value if str(x).strip()]
        if isinstance(value, str):
            s = value.strip()
            if not s:
                return []
            # Heuristique simple: beaucoup de JSON normalisés sont des phrases séparées par des espaces
            # ou des sections "cultural:", "chemical:", etc. On coupe d'abord par "  " / "\n" / ";"
            # puis on garde des items non vides.
            parts = re.split(r"[\n;]+", s)
            out: List[str] = []
            for p in parts:
                p = re.sub(r"\s+", " ", p).strip()
                if p:
                    out.append(p)
            return out
        return [str(value).strip()] if str(value).strip() else []

    # Supporte 2+ schémas:
    # - "BLIP2_normalized" (format normalisé): name / causal_agent / symptoms / management / prevention / hosts
    # - format "dossier BLIPD" (Plantix-like): disease / scientific_name / symptoms_and_damage / disease_cycle_and_spread /
    #   favorable_conditions / pathogen_characteristics / monitoring / management / hosts / ...
    disease_name = data.get("name") or data.get("disease") or data.get("disease_name") or predicted_label
    cause = data.get("causal_agent") or data.get("cause") or data.get("pathogen") or ""
    # Symptoms: peut être soit "symptoms" soit "symptoms_and_damage"
    symptoms = data.get("symptoms", None)
    if symptoms is None:
        symptoms = data.get("symptoms_and_damage", []) or []

    management = data.get("management", []) or data.get("cultural_control", []) or []
    if data.get("chemical_control"):
        # Combine cultural and chemical control
        chemical = _to_list(data.get("chemical_control", []))
        management.extend(chemical)
    prevention = data.get("prevention", []) or []

    disease_cycle_and_spread = data.get("disease_cycle_and_spread", []) or []
    favorable_conditions = data.get("favorable_conditions", []) or []
    pathogen_characteristics = data.get("pathogen_characteristics", []) or []
    monitoring = data.get("monitoring", []) or []

    pathogen_type = data.get("pathogen_type", "") or data.get("type", "") or ""

    susceptibility_raw = data.get("susceptibility", {}) or {}
    susceptibility: Dict[str, List[str]] = {}
    if isinstance(susceptibility_raw, dict):
        for k, v in susceptibility_raw.items():
            susceptibility[str(k)] = _to_list(v)

    # Handle description - create synthetic one if missing
    description = data.get("description", "")
    if not description or not isinstance(description, str) or not description.strip():
        # Create synthetic description from available data
        desc_parts = []
        if pathogen_type:
            desc_parts.append(f"{pathogen_type} disease")
        if cause:
            desc_parts.append(f"caused by {cause}")
        if symptoms and len(symptoms) > 0:
            symptom_text = symptoms[0] if len(symptoms) == 1 else f"{symptoms[0]} and other symptoms"
            desc_parts.append(f"characterized by {symptom_text.lower()}")
        if desc_parts:
            description = " ".join(desc_parts) + "."
        else:
            description = f"Disease affecting plants with symptoms including {', '.join(symptoms[:2]) if symptoms else 'various plant issues'}."

    # Extract hosts from various formats - try normalized first, then fallback to original
    hosts = []
    if data.get("hosts") and len(data["hosts"]) > 0:
        hosts = _to_list(data["hosts"])
    else:
        # If normalized file has empty hosts, try to get from original BLIP2 file
        original_path = _find_best_json_path(predicted_label, Path("BLIP2"), allow_fuzzy=allow_fuzzy)
        if original_path and original_path.exists() and str(json_path) != str(original_path):
            try:
                with open(original_path, "r", encoding="utf-8") as f:
                    original_data = json.load(f)
                    if original_data.get("host_plants"):
                        hosts = _to_list(original_data["host_plants"])
                    elif original_data.get("hosts"):
                        hosts = _to_list(original_data["hosts"])
            except Exception:
                pass

    # Ensure required fields exist (⚠️ on ignore volontairement "sources")
    return {
        "disease": str(disease_name),
        "scientific_name": str(data.get("scientific_name", "") or ""),
        "description": str(description),
        "pathogen_type": str(pathogen_type),
        "hosts": hosts,
        "symptoms": _to_list(symptoms),
        "cause": str(cause),
        "management": _to_list(management),
        "prevention": _to_list(prevention),
        "disease_cycle_and_spread": _to_list(disease_cycle_and_spread),
        "favorable_conditions": _to_list(favorable_conditions),
        "pathogen_characteristics": _to_list(pathogen_characteristics),
        "monitoring": _to_list(monitoring),
        "susceptibility": susceptibility,
    }


def build_blip_prompt(disease_data: Dict[str, Any], language_code: str = "en") -> str:
    """Build a structured prompt for BLIP-2 from disease JSON data."""
    symptoms = disease_data.get("symptoms") or []
    management = disease_data.get("management") or []

    # Éviter les f-string avec `'\n'` dans l'expression `{...}` (bug SyntaxError
    # sur certaines versions Python / environnements).
    symptom_lines = "\n".join([f"- {s}" for s in symptoms]).strip()
    management_lines = "\n".join([f"- {s}" for s in management]).strip()
    # Keep the prompt deterministic: only the language instruction changes.
    lang_instruction_map = {
        "fr": "Respond in French.",
        "en": "Respond in English.",
        "tr": "Respond in Turkish.",
        "sw": "Respond in Swahili.",
        "ha": "Respond in Hausa.",
        "ar": "Respond in Arabic.",
        "zh": "Respond in Chinese.",
    }
    lang_instruction = lang_instruction_map.get(language_code.lower(), lang_instruction_map["en"])

    prompt = f"""
You are an agricultural plant pathology expert.
{lang_instruction}

Disease: {disease_data.get('disease', '')}

Symptoms:
{symptom_lines}

Cause:
{disease_data.get('cause', '')}

Management:
{management_lines}

Explain this disease in a simple way for farmers.
""".strip()

    return prompt


class BLIP2Explainer:
    """Helper for BLIP-2 inference.

    This class caches the model and processor to avoid reloading them on every call.
    """

    def __init__(self, model_name: str = "Salesforce/blip2-flan-t5-base", device: Optional[torch.device] = None):
        self.model_name = model_name
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self._model: Optional[Any] = None
        self._processor: Optional[Any] = None

    def _load_model(self):
        from transformers import Blip2ForConditionalGeneration, Blip2Processor

        self._processor = Blip2Processor.from_pretrained(self.model_name)
        self._model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        ).to(self.device)

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def processor(self):
        if self._processor is None:
            self._load_model()
        return self._processor

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 120,
        num_beams: int = 1,
        min_length: int = 10,
    ) -> str:
        """Generate an explanation for the given image + prompt."""
        inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            min_length=min_length,
            early_stopping=True,
        )

        output = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        return output


def generate_explanation_for_image(
    image: Image.Image,
    predicted_label: str,
    model_name: str = "Salesforce/blip2-flan-t5-base",
    library_dir: Path = Path("BLIP2"),
    language_code: str = "en",
) -> str:
    """High level helper: given an image and a predicted label, returns a BLIP-2 explanation."""
    disease_data = load_disease_info(
        predicted_label,
        library_dir=library_dir,
        allow_fuzzy=False,
        language_code=language_code,
    )
    prompt = build_blip_prompt(disease_data, language_code=language_code)
    explainer = BLIP2Explainer(model_name=model_name)
    return explainer.generate(image, prompt)
