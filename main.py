#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
 STORYBOOK IA: generador inteligente de libros de cuentos con IA generativa
==============================================================================
Todo el código fuente de la aplicación vive en este único archivo:

   1. IMPORTACIONES
   2. CONFIGURACIÓN GENERAL
   3. CONFIGURACIÓN DE MODELOS
   4. GÉNEROS Y PALETAS
   5. UTILIDADES DE TEXTO
   6. GENERACIÓN DE TEXTO .......... Qwen3 (Hugging Face Transformers)
   7. GENERACIÓN DE IMÁGENES ....... SD v1.4 / SDXL / FLUX.1-schnell (Diffusers)
   8. GENERACIÓN DE AUDIO .......... Piper TTS en español (+ AudioLDM opcional)
   9. CONTROL DE ESTADOS ........... estados de cada recurso y persistencia
  10. CREACIÓN DE PORTADA Y ESCENAS  construcción progresiva, en orden
  11. SERVIDOR WEB (API)
  12. INTERFAZ GRÁFICA ............. HTML + CSS del libro
  13. ANIMACIÓN DE PÁGINAS Y NAVEGACIÓN (JavaScript que corre en el navegador)
  14. EJECUCIÓN PRINCIPAL

Uso:
    python main.py                       inicia la aplicación (http://127.0.0.1:8000)
    python main.py --descargar-modelos   descarga una vez los pesos de Hugging Face

Todos los modelos se ejecutan LOCALMENTE. Hugging Face solo se usa para
descargar los pesos la primera vez; no se usa ninguna API de inferencia remota.
"""

# ============================================================================
# 1. IMPORTACIONES
# ============================================================================
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import queue
import random
import re
import secrets
import threading
import time
import traceback
import wave
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field, field_validator

# Las librerías pesadas (torch, transformers, diffusers, piper) se importan
# dentro de las funciones que las usan: la aplicación arranca en segundos y
# los modelos se cargan solo cuando hacen falta.

log = logging.getLogger("storybook")

# ============================================================================
# 2. CONFIGURACIÓN GENERAL
# ============================================================================
ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")

NUM_SCENES = 5                      # el libro tiene SIEMPRE 5 escenas + portada
PIPER_REPO = "rhasspy/piper-voices"
AMBIENT_MODEL_DEFAULT = "cvssp/audioldm-s-full-v2"


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _env_int(name: str) -> int | None:
    value = _env(name)
    return int(value) if value is not None else None


def _env_float(name: str, default: float) -> float:
    value = _env(name)
    return float(value) if value is not None else default


def _env_bool(name: str, default: bool) -> bool:
    value = _env(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "si", "sí", "on"}


# ============================================================================
# 3. CONFIGURACIÓN DE MODELOS
# ============================================================================
# Solo los modelos del documento "Resumen de modelos Storybook IA Generativa".
TEXT_MODELS = {
    "Qwen/Qwen3-0.6B": {
        # Modelo híbrido: se usa en modo "no thinking" con los parámetros
        # recomendados por su ficha para ese modo.
        "enable_thinking": False,
        "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
    },
    "Qwen/Qwen3-4B-Instruct-2507": {
        # Modelo solo-instrucción (no genera bloques <think>).
        "enable_thinking": None,
        "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
    },
}

IMAGE_MODELS = {
    "CompVis/stable-diffusion-v1-4": {
        "family": "sd", "width": 512, "height": 512, "cover_width": 512, "cover_height": 704,
        "steps": 30, "guidance": 7.5, "negative": True, "max_tokens": 77,
    },
    "stabilityai/stable-diffusion-xl-base-1.0": {
        "family": "sdxl", "width": 1024, "height": 1024, "cover_width": 896, "cover_height": 1152,
        "steps": 30, "guidance": 7.0, "negative": True, "max_tokens": 77,
    },
    "black-forest-labs/FLUX.1-schnell": {
        # Modelo destilado: 4 pasos y guidance 0 (recomendación oficial).
        "family": "flux", "width": 1024, "height": 1024, "cover_width": 896, "cover_height": 1152,
        "steps": 4, "guidance": 0.0, "negative": False, "max_tokens": 256,
    },
}

# "Combinaciones orientativas" del documento de modelos.
PROFILES = {
    "accesible": ("Qwen/Qwen3-0.6B", "CompVis/stable-diffusion-v1-4"),
    "equilibrada": ("Qwen/Qwen3-4B-Instruct-2507", "stabilityai/stable-diffusion-xl-base-1.0"),
    "avanzada": ("Qwen/Qwen3-4B-Instruct-2507", "black-forest-labs/FLUX.1-schnell"),
}


def gpu_memory_gib() -> float | None:
    """Memoria total de la GPU CUDA en GiB (None si no hay GPU CUDA)."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1024**3
    except Exception:  # noqa: BLE001
        pass
    return None


def pick_default_profile(device: str, vram_gib: float | None) -> str:
    """Perfil por defecto si STORYBOOK_PROFILE está vacío. Qwen3-4B + SDXL
    necesitan una GPU de unos 12 GB; con menos (p. ej. RTX 4050 de 6 GB) se usa
    el perfil accesible, que sí cabe en la GPU."""
    if device == "cuda" and vram_gib is not None and vram_gib >= 11.5:
        return "equilibrada"
    return "accesible"


def detect_device() -> str:
    forced = _env("DEVICE")
    if forced:
        return forced
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Settings:
    device: str
    profile: str
    text_model: str
    image_model: str
    image_params: dict
    text_params: dict
    piper_voice: str
    piper_length_scale: float
    enable_ambient: bool
    ambient_model: str
    ambient_seconds: float
    ambient_steps: int
    cpu_offload: bool
    low_memory: bool
    image_seed_mode: str
    vram_gib: float | None
    max_pending_jobs: int
    abandon_timeout: float
    data_dir: Path
    models_dir: Path
    host: str
    port: int

    def public_info(self) -> dict:
        """Información segura para el navegador (nunca incluye tokens)."""
        return {
            "device": self.device, "profile": self.profile,
            "text_model": self.text_model, "image_model": self.image_model,
            "voice": f"Piper · {self.piper_voice}",
            "ambient_enabled": self.enable_ambient,
            "ambient_model": self.ambient_model if self.enable_ambient else None,
            "vram_gib": self.vram_gib,
        }


def load_settings() -> Settings:
    device = detect_device()
    vram = gpu_memory_gib() if device == "cuda" else None
    profile = _env("STORYBOOK_PROFILE") or pick_default_profile(device, vram)
    if profile not in PROFILES:
        raise ValueError(f"STORYBOOK_PROFILE inválido: {profile}. Opciones: {', '.join(PROFILES)}")
    text_model, image_model = PROFILES[profile]
    text_model = _env("TEXT_MODEL", text_model)
    image_model = _env("IMAGE_MODEL", image_model)
    if text_model not in TEXT_MODELS:
        raise ValueError(f"TEXT_MODEL no permitido: {text_model}. Opciones: {', '.join(TEXT_MODELS)}")
    if image_model not in IMAGE_MODELS:
        raise ValueError(f"IMAGE_MODEL no permitido: {image_model}. Opciones: {', '.join(IMAGE_MODELS)}")

    image_params = dict(IMAGE_MODELS[image_model])
    for key, env_name in (("width", "IMAGE_WIDTH"), ("height", "IMAGE_HEIGHT"),
                          ("cover_width", "COVER_WIDTH"), ("cover_height", "COVER_HEIGHT"),
                          ("steps", "IMAGE_STEPS")):
        override = _env_int(env_name)
        if override is not None:
            image_params[key] = override
    image_params["guidance"] = _env_float("IMAGE_GUIDANCE", image_params["guidance"])
    text_params = dict(TEXT_MODELS[text_model])
    text_params["temperature"] = _env_float("TEXT_TEMPERATURE", text_params["temperature"])

    low_memory = _env_bool("LOW_MEMORY", False)
    data_dir = Path(_env("DATA_DIR", str(ROOT_DIR / "data")))
    data_dir.mkdir(parents=True, exist_ok=True)
    return Settings(
        device=device, profile=profile, text_model=text_model, image_model=image_model,
        image_params=image_params, text_params=text_params,
        piper_voice=_env("PIPER_VOICE", "es_MX-ald-medium"),
        piper_length_scale=_env_float("PIPER_LENGTH_SCALE", 1.05),
        enable_ambient=_env_bool("ENABLE_AMBIENT_SOUND", False),
        ambient_model=_env("AMBIENT_MODEL", AMBIENT_MODEL_DEFAULT),
        ambient_seconds=_env_float("AMBIENT_SECONDS", 10.0),
        ambient_steps=_env_int("AMBIENT_STEPS") or 25,
        cpu_offload=_env_bool("CPU_OFFLOAD", False),
        low_memory=low_memory,
        # "incremental" (semilla + nº de escena): cada escena tiene su propia composición;
        # la continuidad la dan los rasgos de los personajes y el estilo, idénticos en
        # todos los prompts. Con "fixed" las 5 imágenes salían casi iguales.
        image_seed_mode=_env("IMAGE_SEED_MODE", "incremental"),
        vram_gib=round(vram, 1) if vram else None,
        max_pending_jobs=_env_int("MAX_PENDING_JOBS") or 3,
        abandon_timeout=_env_float("ABANDON_TIMEOUT", 180.0),
        data_dir=data_dir,
        models_dir=Path(_env("MODELS_DIR", str(ROOT_DIR / "models"))),
        host=_env("HOST", "127.0.0.1"),
        port=_env_int("PORT") or 8000,
    )


def free_memory():
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class Cancelled(Exception):
    """El usuario detuvo la generación (o nadie la está observando)."""


# ============================================================================
# 4. GÉNEROS Y PALETAS
# ============================================================================
# narrative: guía de estilo (español) para el modelo de texto.
# art_style / palette_hint: estilo visual (inglés) para el modelo de imagen.
# Los colores de la interfaz de cada género están en el CSS (sección 12).
GENRES: dict[str, dict] = {
    "romance": {"label": "Romance", "emoji": "💕",
                "narrative": "tono cálido, emotivo y tierno; enfatiza los sentimientos y las miradas",
                "art_style": "soft romantic storybook illustration, dreamy watercolor, warm golden light",
                "palette_hint": "pastel pink, soft red and cream tones"},
    "terror": {"label": "Terror", "emoji": "🕯️",
               "narrative": "atmósfera inquietante y oscura, tensión creciente, sin violencia explícita",
               "art_style": "dark gothic storybook illustration, eerie shadows, moody cinematic lighting",
               "palette_hint": "black, grey and dark red tones"},
    "fantasia": {"label": "Fantasía", "emoji": "🧚",
                 "narrative": "maravilla, magia y descubrimiento; descripciones luminosas y evocadoras",
                 "art_style": "magical fantasy storybook illustration, glowing sparkles, ethereal light",
                 "palette_hint": "lilac, violet and magical blue tones"},
    "ciencia_ficcion": {"label": "Ciencia ficción", "emoji": "🚀",
                        "narrative": "imaginación tecnológica y futurista, asombro científico, ritmo ágil",
                        "art_style": "futuristic sci-fi concept art illustration, neon glow, sleek technology",
                        "palette_hint": "electric blue and violet tones"},
    "aventura": {"label": "Aventura", "emoji": "🧭",
                 "narrative": "ritmo dinámico, exploración, valentía y descubrimiento de lugares",
                 "art_style": "adventure storybook illustration, painterly, vivid natural landscapes",
                 "palette_hint": "green, brown and beige natural tones"},
    "accion": {"label": "Acción", "emoji": "⚡",
               "narrative": "ritmo trepidante, frases cortas, movimiento constante y desafíos",
               "art_style": "dynamic action illustration, bold composition, motion, dramatic lighting",
               "palette_hint": "orange, black and steel tones"},
    "misterio": {"label": "Misterio", "emoji": "🔍",
                 "narrative": "pistas, preguntas y revelaciones graduales; tono intrigante",
                 "art_style": "mysterious noir storybook illustration, fog, lamplight, deep shadows",
                 "palette_hint": "dark blue, grey and black tones with golden accents"},
    "suspenso": {"label": "Suspenso", "emoji": "⏳",
                 "narrative": "tensión sostenida, incertidumbre y giros inesperados",
                 "art_style": "suspenseful cinematic illustration, tense atmosphere, high contrast",
                 "palette_hint": "deep teal, charcoal and amber tones"},
    "drama": {"label": "Drama", "emoji": "🎭",
              "narrative": "conflictos emocionales profundos, introspección y decisiones difíciles",
              "art_style": "emotional painterly illustration, expressive faces, soft dramatic light",
              "palette_hint": "burgundy, slate and muted gold tones"},
    "comedia": {"label": "Comedia", "emoji": "😂",
                "narrative": "humor amable, situaciones absurdas y diálogos ingeniosos",
                "art_style": "cheerful cartoon storybook illustration, playful expressions, bright colors",
                "palette_hint": "sunny yellow, orange and turquoise tones"},
    "infantil": {"label": "Cuentos infantiles", "emoji": "🧸",
                 "narrative": "lenguaje sencillo y tierno para niños, mensaje positivo, tono amable",
                 "art_style": "cute children's picture book illustration, soft rounded shapes, gentle",
                 "palette_hint": "soft pastel colors"},
    "mitologia": {"label": "Mitología", "emoji": "🏛️",
                  "narrative": "tono épico y solemne, dioses, héroes y criaturas legendarias",
                  "art_style": "epic mythological illustration, classical painting style, majestic",
                  "palette_hint": "gold, marble white and deep blue tones"},
    "fabula": {"label": "Fábula", "emoji": "🦊",
               "narrative": "animales con rasgos humanos y una moraleja clara al final",
               "art_style": "classic fable illustration, anthropomorphic animals, gouache, storybook",
               "palette_hint": "warm earthy greens, ochre and soft browns"},
    "historico": {"label": "Histórico", "emoji": "📜",
                  "narrative": "ambientación de época verosímil, detalles históricos y tono sobrio",
                  "art_style": "historical oil painting illustration, period accurate costumes, detailed",
                  "palette_hint": "sepia, parchment and dark brown tones"},
}


def get_genre(key: str) -> dict:
    if key not in GENRES:
        raise KeyError(key)
    return GENRES[key]


def public_genres() -> list[dict]:
    return [{"key": k, "label": v["label"], "emoji": v["emoji"]} for k, v in GENRES.items()]


# ============================================================================
# 5. UTILIDADES DE TEXTO
# ============================================================================
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
# Qwen a veces mezcla caracteres CJK; se eliminan del texto en español.
_CJK_RE = re.compile(r"[\u3000-\u303f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uff00-\uffef]+")
_SPANISH_WORDS = {
    "el", "la", "los", "las", "de", "del", "y", "que", "en", "un", "una", "con", "por",
    "para", "su", "sus", "se", "al", "lo", "como", "más", "pero", "era", "fue", "había",
    "es", "no", "sin", "sobre", "entre", "cuando", "muy", "también", "hacia", "desde",
}
_ENGLISH_WORDS = {
    "the", "and", "of", "to", "in", "was", "were", "with", "her", "his", "she", "he",
    "they", "that", "this", "is", "it", "for", "on", "at", "from", "had", "but", "not",
}


def strip_think(text: str) -> str:
    text = _THINK_RE.sub("", text)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.replace("<think>", "").strip()


def remove_cjk(text: str) -> str:
    return _CJK_RE.sub("", text)


def extract_json(text: str) -> dict:
    """Extrae el primer objeto JSON válido de la respuesta del modelo."""
    text = strip_think(text)
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    start = text.find("{")
    if start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start:i + 1])
                    break
    for raw in candidates:
        for attempt in (raw, re.sub(r",\s*([}\]])", r"\1", raw)):  # tolera comas finales
            try:
                value = json.loads(attempt)
                if isinstance(value, dict):
                    return value
            except json.JSONDecodeError:
                continue
    raise ValueError("La respuesta del modelo no contiene un JSON válido.")


def _word_lists(text: str) -> tuple[int, int]:
    words = re.findall(r"[a-záéíóúñü]+", text.lower())
    return (sum(1 for w in words if w in _SPANISH_WORDS), sum(1 for w in words if w in _ENGLISH_WORDS))


def spanish_score(text: str) -> float:
    es, en = _word_lists(text)
    return es / (es + en) if es + en else 0.0


def looks_english(text: str) -> bool:
    es, en = _word_lists(text)
    return en > es


def is_spanish(text: str, min_score: float = 0.75) -> bool:
    return spanish_score(text) >= min_score and not _CJK_RE.search(text)


def clean_scene_text(text: str, scene_title: str = "") -> str:
    """Normaliza el texto narrativo de una escena (quita encabezados, CJK, etc.)."""
    text = remove_cjk(strip_think(text))
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            lines.append("")
            continue
        if re.match(r"^(#+\s|\*\*.+\*\*$|escena\s*\d+\s*[:.\-–]?\s*$)", line, re.IGNORECASE):
            continue
        line = re.sub(r"^escena\s*\d+\s*[:.\-–]\s*", "", line, flags=re.IGNORECASE)
        if scene_title and line.strip("*#: ").lower() == scene_title.strip().lower():
            continue
        lines.append(line.replace("**", "").replace("__", ""))
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip().strip('"“”').strip()


_STOP_ES = _SPANISH_WORDS | {"este", "esta", "eso", "esa", "pero", "porque", "todo", "todos", "estaba",
                              "estaban", "habia", "tenía", "tenia", "donde", "mientras", "luego", "aquel"}


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-záéíóúñü]+", text.lower()) if len(w) > 3 and w not in _STOP_ES}


def text_similarity(a: str, b: str) -> float:
    """Parecido entre dos textos (Jaccard de secuencias de 3 palabras, 0..1).
    Detecta escenas copiadas o casi copiadas de otra."""
    def shingles(t: str) -> set[tuple[str, ...]]:
        w = re.findall(r"\w+", t.lower())
        return {tuple(w[i:i + 3]) for i in range(max(0, len(w) - 2))}
    sa, sb = shingles(a), shingles(b)
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0


def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text))


def truncate_words(text: str, max_words: int) -> str:
    """Recorta al final de la última oración completa dentro del límite."""
    words = text.split()
    if len(words) <= max_words:
        return text
    cut = " ".join(words[:max_words])
    last = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"), cut.rfind("…"))
    return cut[: last + 1] if last > len(cut) * 0.5 else cut + "…"


def clean_short(text: Any, limit: int = 300) -> str:
    text = remove_cjk(strip_think(str(text or ""))).replace("\n", " ").strip().strip('"')
    return re.sub(r"\s+", " ", text)[:limit].strip()


# ============================================================================
# 6. GENERACIÓN DE TEXTO (Qwen3 local con Transformers)
# ============================================================================
class LocalLLM:
    """Modelo de lenguaje causal ejecutado localmente.

    chat(..., on_text=fn) transmite el texto token a token con
    transformers.TextIteratorStreamer (streaming REAL). should_stop() permite
    interrumpir la generación entre tokens (StoppingCriteria).
    """

    def __init__(self, model_id: str, params: dict, device: str):
        self.model_id, self.params, self.device = model_id, params, device
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        log.info("Cargando modelo de texto %s en %s…", self.model_id, self.device)
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=dtype)
        self._model.to(self.device)
        self._model.eval()

    def unload(self):
        with self._lock:
            if self._model is not None:
                self._model = self._tokenizer = None
                free_memory()
                log.info("Modelo de texto descargado de memoria.")

    def chat(self, messages: list[dict], max_new_tokens: int = 512, temperature: float | None = None,
             on_text: Callable[[str], None] | None = None,
             should_stop: Callable[[], bool] | None = None) -> str:
        with self._lock:
            self._load()
            import torch
            from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

            template_kwargs = {"add_generation_prompt": True, "tokenize": False}
            if self.params.get("enable_thinking") is not None:
                template_kwargs["enable_thinking"] = self.params["enable_thinking"]
            prompt = self._tokenizer.apply_chat_template(messages, **template_kwargs)
            inputs = self._tokenizer([prompt], return_tensors="pt").to(self.device)
            temp = self.params["temperature"] if temperature is None else temperature
            gen_kwargs = dict(**inputs, max_new_tokens=max_new_tokens, do_sample=temp > 0,
                              temperature=max(temp, 1e-5), top_p=self.params["top_p"],
                              top_k=self.params["top_k"], min_p=self.params["min_p"],
                              # 1.1 (antes 1.05): la ficha de Qwen3 recomienda penalizar la
                              # repetición si el modelo repite frases; transformers no tiene
                              # presence_penalty, su equivalente es repetition_penalty.
                              repetition_penalty=1.1, pad_token_id=self._tokenizer.eos_token_id)
            if should_stop is not None:
                class _Stop(StoppingCriteria):
                    def __call__(self, input_ids, scores, **kwargs):
                        return torch.full((input_ids.shape[0],), bool(should_stop()),
                                          dtype=torch.bool, device=input_ids.device)
                gen_kwargs["stopping_criteria"] = StoppingCriteriaList([_Stop()])

            if on_text is None:
                with torch.inference_mode():
                    output = self._model.generate(**gen_kwargs)
                text = self._tokenizer.decode(output[0][inputs["input_ids"].shape[1]:],
                                              skip_special_tokens=True)
            else:
                streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
                gen_kwargs["streamer"] = streamer
                errors: list[Exception] = []

                def run():
                    try:
                        with torch.inference_mode():  # el modo de inferencia es por hilo
                            self._model.generate(**gen_kwargs)
                    except Exception as exc:  # noqa: BLE001
                        errors.append(exc)
                        streamer.on_finalized_text("", stream_end=True)

                worker = threading.Thread(target=run, name="llm-stream", daemon=True)
                worker.start()
                text = ""
                for chunk in streamer:
                    text += chunk
                    on_text(text)
                worker.join()
                if errors:
                    raise errors[0]
            if should_stop is not None and should_stop():
                raise Cancelled()
            return text


SCENE_ROLES = [
    "Inicio: presenta a los personajes, el lugar y la situación inicial.",
    "Desarrollo: aparece el conflicto o el desafío principal.",
    "Desarrollo: el conflicto se complica y los personajes toman decisiones.",
    "Clímax: el momento de mayor tensión o emoción de la historia.",
    "Desenlace: resuelve el conflicto y cierra la historia de forma satisfactoria.",
]

PLAN_SYSTEM = ("Eres un autor profesional de libros ilustrados y director de arte. "
               "Respondes únicamente con JSON válido, sin texto adicional.")

PLAN_TEMPLATE = """Planifica un cuento ilustrado del género «{genre}» ({narrative}).

Idea del usuario (respétala y desarróllala creativamente):
\"\"\"{description}\"\"\"
{title_line}
Reglas:
- Exactamente 5 escenas: inicio, desarrollo, desarrollo, clímax y desenlace.
- "title", "summary" y los títulos de escena: en ESPAÑOL.
- "visual", "appearance", "setting", "cover" y "sound": en INGLÉS, frases cortas y concretas para un ilustrador.
- "appearance" describe rasgos físicos FIJOS y ropa de cada personaje (edad, cabello, piel, ropa, colores) para que sean idénticos en todas las ilustraciones.
- Máximo 3 personajes.

Devuelve SOLO este JSON:
{{
  "title": "título en español",
  "setting": "main location in English",
  "characters": [
    {{"name": "nombre", "appearance": "physical traits and clothing in English"}}
  ],
  "cover": "cover illustration idea in English",
  "scenes": [
    {{"title": "título de la escena en español", "summary": "qué ocurre, en español", "visual": "what the illustration shows, in English", "characters": ["nombre"], "sound": "ambient sound in English"}}
  ]
}}"""

# Plan de respaldo: si el modelo pequeño no logra un JSON válido, se le pide un
# formato mucho más simple (5 líneas numeradas) para no dejar el libro detenido.
FALLBACK_TEMPLATE = """Idea del cuento ({genre}): «{description}»

Escribe EXACTAMENTE 5 líneas numeradas, en español, una por escena (inicio, desarrollo, desarrollo, clímax y desenlace), con este formato:
1. Título de la escena: qué ocurre
2. Título de la escena: qué ocurre
3. Título de la escena: qué ocurre
4. Título de la escena: qué ocurre
5. Título de la escena: qué ocurre"""

APPEARANCE_TEMPLATE = """Idea del cuento: «{description}»
Describe in ONE short line, in English, the physical appearance and clothing of the main character (age, hair, clothes, colors). Only the description."""

SCENE_SYSTEM = ("Eres un escritor de cuentos en español. Escribes SIEMPRE y ÚNICAMENTE en "
                "español correcto, con buena ortografía y puntuación. Nunca usas otro idioma.")

# Prompt de escena pensado también para modelos pequeños: solo lo ya ocurrido
# (resúmenes) y lo que debe ocurrir AHORA. No se pega el texto anterior completo,
# porque un modelo pequeño tiende a copiarlo.
SCENE_TEMPLATE = """Cuento: «{title}» (género {genre}; estilo: {narrative}).
Personajes: {characters}.
{story_so_far}
AHORA escribe SOLO la escena {n} de 5: «{scene_title}».
En esta escena ocurre: {scene_summary}
Función de la escena: {role}

Reglas:
- Entre 90 y 150 palabras, en prosa narrativa, en español.
- Cuenta hechos NUEVOS de esta escena; no repitas lo que ya se contó en escenas anteriores.
- Mantén los nombres y rasgos de los personajes.
- No escribas títulos, encabezados, listas ni notas: solo el texto.{ending}{extra}"""

# Descripción de la ilustración a partir del texto REAL de la escena (en inglés,
# porque los codificadores de texto de Stable Diffusion / FLUX entienden inglés).
ILLUSTRATION_TEMPLATE = """Read this passage from a story (written in Spanish) and describe ONE illustration for it.
Write in ENGLISH, maximum 25 words, one line: who appears, what they are doing right now, and where.
Describe the most important visible moment of THIS passage. Only the description, no names of artists.

Characters: {characters}
Passage:
\"\"\"{text}\"\"\"

Illustration description in English:"""

COVER_TEMPLATE = """Describe in ENGLISH, maximum 25 words, one line, a book cover illustration for this story.
Title: {title}
Story: {summary}
Only the description:"""


class StoryGenerationError(RuntimeError):
    pass


class StoryWriter:
    """Historia en dos etapas:
    1. PLAN: título, biblia visual de personajes y exactamente 5 escenas.
    2. ESCENAS: cada escena se redacta por separado con el plan completo y el
       texto de la escena anterior (continuidad narrativa).
    """

    def __init__(self, llm, max_attempts: int = 3):
        self.llm = llm
        self.max_attempts = max_attempts

    # ------------------------------------------------------------------ plan
    def create_plan(self, description: str, genre_key: str, user_title: str = "",
                    should_stop=None) -> dict:
        genre = get_genre(genre_key)
        title_line = f"Título obligatorio del cuento: «{user_title}» (úsalo tal cual).\n" if user_title else ""
        prompt = PLAN_TEMPLATE.format(genre=genre["label"], narrative=genre["narrative"],
                                      description=description, title_line=title_line)
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            raw = self.llm.chat([{"role": "system", "content": PLAN_SYSTEM},
                                 {"role": "user", "content": prompt}],
                                max_new_tokens=1400, temperature=None if attempt == 1 else 0.5,
                                should_stop=should_stop)
            try:
                plan = self._normalize_plan(extract_json(raw), user_title, description)
                log.info("Plan generado en el intento %d", attempt)
                return plan
            except (ValueError, KeyError, TypeError) as exc:
                last_error = exc
                log.warning("Plan inválido (intento %d): %s", attempt, exc)
        log.warning("Se usa el plan de respaldo (líneas numeradas) tras: %s", last_error)
        try:
            return self._fallback_plan(description, genre_key, user_title, should_stop)
        except (ValueError, KeyError) as exc:
            raise StoryGenerationError(f"No se pudo planificar la historia: {exc}") from exc

    def _fallback_plan(self, description: str, genre_key: str, user_title: str, should_stop) -> dict:
        genre = get_genre(genre_key)
        raw = self.llm.chat([{"role": "system", "content": SCENE_SYSTEM},
                             {"role": "user", "content": FALLBACK_TEMPLATE.format(
                                 genre=genre["label"], description=description)}],
                            max_new_tokens=500, should_stop=should_stop)
        found: dict[int, str] = {}
        for num, line in re.findall(r"^\s*\**\s*([1-5])\s*[.)\-:]\s*(.+)$", strip_think(raw), re.MULTILINE):
            found.setdefault(int(num), clean_short(line, 400))
        if len(found) < NUM_SCENES:
            raise ValueError("el modelo no devolvió 5 escenas")
        appearance = clean_short(self.llm.chat(
            [{"role": "user", "content": APPEARANCE_TEMPLATE.format(description=description)}],
            max_new_tokens=80, should_stop=should_stop), 200)
        scenes = []
        for i in range(1, NUM_SCENES + 1):
            title, _, summary = found[i].partition(":")
            summary = summary.strip() or title
            scenes.append({"title": title.strip(" *"), "summary": summary, "visual": summary,
                           "characters": ["protagonista"], "sound": ""})
        data = {"title": user_title or f"Un cuento de {genre['label'].lower()}",
                "setting": "", "cover": description,
                "characters": [{"name": "protagonista", "appearance": appearance}], "scenes": scenes}
        return self._normalize_plan(data, user_title, description)

    def _normalize_plan(self, data: dict, user_title: str, description: str) -> dict:
        scenes = data.get("scenes")
        if not isinstance(scenes, list) or len(scenes) < NUM_SCENES:
            raise ValueError(f"se esperaban 5 escenas y llegaron {len(scenes) if isinstance(scenes, list) else 0}")
        scenes = scenes[:NUM_SCENES]  # nunca más de 5
        characters = [{"name": clean_short(ch["name"], 60),
                       "appearance": clean_short(ch.get("appearance") or ch.get("visual") or "", 220)}
                      for ch in (data.get("characters") or []) if isinstance(ch, dict) and ch.get("name")][:3]
        title = user_title.strip() or clean_short(data.get("title"), 90)
        if not title:
            raise ValueError("el plan no tiene título")
        norm_scenes = []
        for i, sc in enumerate(scenes, start=1):
            if not isinstance(sc, dict):
                raise ValueError(f"la escena {i} no es un objeto")
            summary = clean_short(sc.get("summary"), 400)
            if not summary:
                raise ValueError(f"la escena {i} no tiene resumen")
            scene_title = clean_short(sc.get("title"), 80)
            if not scene_title or looks_english(scene_title):
                scene_title = f"Escena {i}"
            names = sc.get("characters") if isinstance(sc.get("characters"), list) else []
            norm_scenes.append({
                "index": i, "title": scene_title, "summary": summary,
                "visual": clean_short(sc.get("visual"), 300) or summary,
                "characters": [clean_short(n, 60) for n in names if isinstance(n, str)],
                "sound": clean_short(sc.get("sound"), 160),
            })
        for i in range(len(norm_scenes)):
            for j in range(i + 1, len(norm_scenes)):
                a, b = content_words(norm_scenes[i]["summary"]), content_words(norm_scenes[j]["summary"])
                if a and b and len(a & b) / len(a | b) > 0.6:
                    raise ValueError(f"las escenas {i + 1} y {j + 1} del plan son casi iguales")
        return {"title": title, "setting": clean_short(data.get("setting"), 220), "characters": characters,
                "cover": clean_short(data.get("cover"), 300) or norm_scenes[0]["visual"],
                "scenes": norm_scenes, "description": description}

    # --------------------------------------------------------------- escenas
    def write_scene(self, plan: dict, genre_key: str, n: int, previous_texts: list[str] | None = None,
                    on_draft=None, should_stop=None) -> str:
        """Redacta la escena n. previous_texts son los textos ya escritos de las
        escenas anteriores: se usan para la continuidad y para RECHAZAR una escena
        que repita a otra. on_draft(texto_parcial, intento) recibe el streaming."""
        genre = get_genre(genre_key)
        scene = plan["scenes"][n - 1]
        previous_texts = [t for t in (previous_texts or []) if t]
        characters = ", ".join(c["name"] for c in plan["characters"]) or "los protagonistas"
        if n > 1:
            done = "\n".join(f"- Escena {s['index']}: {s['summary']}" for s in plan["scenes"][:n - 1])
            last = ""
            if previous_texts:
                sentences = re.split(r"(?<=[.!?…])\s+", previous_texts[-1].strip())
                last = f"\nAsí terminó la escena anterior: «{' '.join(sentences[-2:])[-300:]}»"
            story_so_far = f"Lo que ya pasó en el cuento:\n{done}{last}\n"
        else:
            story_so_far = f"Es el comienzo del cuento. Idea general: {plan.get('description', '')[:300]}\n"
        if n == NUM_SCENES:
            ending = "\n- Es la última escena: cierra la historia con un final claro." + (
                " Termina con una moraleja breve." if genre_key == "fabula" else "")
        else:
            ending = "\n- No termines la historia todavía."
        base = dict(title=plan["title"], genre=genre["label"], narrative=genre["narrative"], characters=characters,
                    story_so_far=story_so_far, n=n, scene_title=scene["title"], scene_summary=scene["summary"],
                    role=SCENE_ROLES[n - 1], ending=ending)
        extra = ""
        candidates: list[tuple[float, str]] = []
        for attempt in range(1, self.max_attempts + 2):
            prompt = SCENE_TEMPLATE.format(**base, extra=extra)
            stream = None
            if on_draft is not None:
                on_draft("", attempt)
                stream = (lambda raw, a=attempt: on_draft(
                    truncate_words(clean_scene_text(raw, scene["title"]), 190), a))
            raw = self.llm.chat([{"role": "system", "content": SCENE_SYSTEM},
                                 {"role": "user", "content": prompt}],
                                max_new_tokens=420, on_text=stream, should_stop=should_stop,
                                temperature=None if attempt == 1 else 0.9)
            text = truncate_words(clean_scene_text(raw, scene["title"]), 190)
            if on_draft is not None:
                on_draft(text, attempt)
            repeated = max((text_similarity(text, t) for t in previous_texts), default=0.0)
            if word_count(text) >= 40 and is_spanish(text) and repeated < 0.3:
                return text
            log.warning("Escena %d rechazada (intento %d): %d palabras, español=%s, parecido con otra escena=%.2f",
                        n, attempt, word_count(text), is_spanish(text), repeated)
            if word_count(text) >= 25 and is_spanish(text, 0.6) and repeated < 0.3:
                candidates.append((-len(content_words(text) & content_words(scene["summary"])), text))
            if repeated >= 0.3:
                extra = ("\n\nIMPORTANTE: tu respuesta anterior repetía lo que ya se contó. Escribe una escena "
                         f"DIFERENTE, con hechos nuevos: {scene['summary']}")
        if candidates:
            return remove_cjk(sorted(candidates)[0][1])
        raise StoryGenerationError(
            f"No se pudo redactar la escena {n}: el modelo repetía otra escena o no escribía en español. "
            "Pulsa «Reintentar texto»; con GPU de 12 GB o más, el perfil 'equilibrada' es más fiable.")

    # ------------------------------------------------ descripción de imágenes
    def _english_line(self, prompt: str, should_stop=None) -> str | None:
        """Una línea descriptiva en inglés (o None si el modelo no la produce)."""
        for attempt in (1, 2):
            raw = self.llm.chat([{"role": "user", "content": prompt}], max_new_tokens=80,
                                temperature=0.3 if attempt == 1 else 0.6, should_stop=should_stop)
            line = clean_short(strip_think(raw).split("\n")[0], 260).strip(" .:-*")
            line = re.sub(r"^(illustration( description)?|description)\s*:\s*", "", line, flags=re.IGNORECASE)
            if len(line.split()) >= 4 and not is_spanish(line, 0.5):
                return line
        return None

    def describe_scene(self, plan: dict, n: int, text: str, should_stop=None) -> str:
        """Descripción visual (inglés) del momento que cuenta el texto REAL de la
        escena n. Si el modelo no la produce, se usa la del plan."""
        names = [c["name"] for c in scene_characters(plan, plan["scenes"][n - 1])] or ["the main character"]
        line = self._english_line(ILLUSTRATION_TEMPLATE.format(characters=", ".join(names), text=text[:900]),
                                  should_stop)
        fallback = plan["scenes"][n - 1].get("visual") or ""
        return line or fallback

    def describe_cover(self, plan: dict, should_stop=None) -> str:
        if plan.get("cover") and not is_spanish(plan["cover"], 0.5):
            return plan["cover"]
        summary = " ".join(s["summary"] for s in plan["scenes"][:2])[:500]
        return self._english_line(COVER_TEMPLATE.format(title=plan["title"], summary=summary), should_stop) \
            or plan.get("cover", "")


# ============================================================================
# 7. GENERACIÓN DE IMÁGENES (Diffusers local)
# ============================================================================
NEGATIVE_PROMPT = ("text, letters, words, watermark, signature, logo, blurry, lowres, deformed, "
                   "disfigured, extra limbs, extra fingers, bad anatomy, duplicate, cropped, "
                   "photo, nsfw, gore")

# Proyección lineal aproximada de los latentes (4 canales) a RGB para mostrar
# cómo "se revela" la imagen real en cada paso, sin decodificar con el VAE.
# Coeficientes de uso común en herramientas abiertas (p. ej. ComfyUI).
LATENT_RGB = {
    "sd": ([[0.3512, 0.2297, 0.3227], [0.3250, 0.4974, 0.2350],
            [-0.2829, 0.1762, 0.2721], [-0.2120, -0.2616, -0.7177]], [0.0, 0.0, 0.0]),
    "sdxl": ([[0.3651, 0.4232, 0.4341], [-0.2533, -0.0042, 0.1068],
              [0.1076, 0.1111, -0.0362], [-0.3165, -0.2492, -0.2188]], [0.1084, -0.0175, -0.0011]),
}


def scene_characters(plan: dict, scene: dict | None) -> list[dict]:
    if scene is None:
        return plan["characters"]
    names = {n.lower() for n in scene.get("characters", [])}
    chosen = [c for c in plan["characters"] if c["name"].lower() in names]
    return chosen or plan["characters"][:1]  # sin datos, se usa el protagonista


def build_prompt_parts(plan: dict, genre_key: str, scene: dict | None) -> list[tuple[int, str]]:
    """(prioridad, texto) en orden natural. Los rasgos de personaje y el estilo
    son textualmente idénticos en todas las escenas: continuidad visual."""
    genre = get_genre(genre_key)
    parts: list[tuple[int, str]] = [(0, f"book cover illustration, {plan['cover']}" if scene is None
                                     else f"storybook illustration of {scene['visual']}")]
    for i, ch in enumerate(scene_characters(plan, scene)):
        if ch.get("appearance"):  # CLIP no conoce los nombres propios: solo los rasgos
            parts.append((1 + i, ch["appearance"]))
    parts.append((1, genre["art_style"]))
    parts.append((4, genre["palette_hint"]))
    if plan.get("setting"):
        parts.append((5, plan["setting"]))
    parts.append((6, "consistent character design, highly detailed, no text"))
    return parts


def fit_prompt(parts: list[tuple[int, str]], count_tokens, max_tokens: int) -> str:
    """Agrega fragmentos por prioridad mientras quepan (CLIP trunca a 77 tokens)."""
    selected: set[int] = set()
    for idx in sorted(range(len(parts)), key=lambda i: (parts[i][0], i)):
        trial = selected | {idx}
        if count_tokens(", ".join(parts[i][1] for i in sorted(trial))) <= max_tokens or not selected:
            selected = trial
    return ", ".join(parts[i][1] for i in sorted(selected))


def build_prompt(plan: dict, genre_key: str, scene: dict | None, count_tokens, max_tokens: int) -> str:
    return fit_prompt(build_prompt_parts(plan, genre_key, scene), count_tokens, max_tokens)


def latents_to_preview(latents, family: str) -> bytes | None:
    """Miniatura JPEG aproximada de los latentes del paso actual (FLUX: no)."""
    if family not in LATENT_RGB:
        return None
    import torch
    from PIL import Image

    factors, bias = LATENT_RGB[family]
    lat = latents[0].detach().float().cpu()
    if lat.shape[0] != 4:
        return None
    rgb = torch.einsum("chw,cr->hwr", lat, torch.tensor(factors)) + torch.tensor(bias)
    rgb = ((rgb + 1.0) / 2.0).clamp(0, 1).mul(255).byte().numpy()
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=70)
    return buf.getvalue()


class ImageGenerator:
    """SD v1.4 / SDXL / FLUX.1-schnell con AutoPipelineForText2Image."""

    def __init__(self, model_id: str, params: dict, device: str, cpu_offload: bool = False,
                 seed_mode: str = "incremental"):
        self.model_id, self.params, self.device = model_id, params, device
        self.cpu_offload, self.seed_mode = cpu_offload, seed_mode
        self._pipe = None
        self._lock = threading.Lock()

    def _load(self):
        if self._pipe is not None:
            return
        import torch
        from diffusers import AutoPipelineForText2Image

        family = self.params["family"]
        dtype = (torch.bfloat16 if family == "flux" else torch.float16) if self.device == "cuda" else torch.float32
        kwargs: dict = {"torch_dtype": dtype}
        if family in ("sd", "sdxl") and dtype == torch.float16:
            kwargs["variant"] = "fp16"
        if family == "sd":
            # El safety checker de SD v1.4 da falsos positivos (imágenes negras) en
            # ilustraciones; el negative prompt ya excluye contenido explícito.
            kwargs["safety_checker"] = None
            kwargs["requires_safety_checker"] = False
        log.info("Cargando modelo de imagen %s (%s) en %s…", self.model_id, dtype, self.device)
        try:
            pipe = AutoPipelineForText2Image.from_pretrained(self.model_id, **kwargs)
        except (OSError, ValueError) as exc:
            if "variant" not in kwargs:
                raise
            log.warning("Sin pesos fp16 (%s); se cargan los pesos estándar.", exc)
            kwargs.pop("variant")
            pipe = AutoPipelineForText2Image.from_pretrained(self.model_id, **kwargs)
        if self.cpu_offload and self.device == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self.device)
        if hasattr(pipe, "enable_attention_slicing") and self.device != "cuda":
            pipe.enable_attention_slicing()
        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe

    def unload(self):
        with self._lock:
            if self._pipe is not None:
                self._pipe = None
                free_memory()
                log.info("Modelo de imagen descargado de memoria.")

    def count_tokens(self, text: str) -> int:
        tokenizer = None
        if self._pipe is not None:
            tokenizer = getattr(self._pipe, "tokenizer_2" if self.params["family"] == "flux" else "tokenizer", None)
        if tokenizer is None:  # estimación conservadora si aún no hay pipeline
            return int(len(text.split()) * 1.4) + 2
        return len(tokenizer(text, truncation=False)["input_ids"])

    def generate(self, plan: dict, genre_key: str, scene: dict | None, seed: int, out_path: Path,
                 on_progress=None, should_stop=None) -> dict:
        """scene=None genera la portada. on_progress(paso, total, preview_jpeg|None)
        se invoca en cada paso real de difusión; should_stop() la interrumpe."""
        with self._lock:
            self._load()
            import torch

            prompt = build_prompt(plan, genre_key, scene, self.count_tokens, self.params["max_tokens"])
            index = 0 if scene is None else scene["index"]
            final_seed = seed if self.seed_mode == "fixed" else seed + index
            is_cover = scene is None
            width = self.params["cover_width" if is_cover else "width"]
            height = self.params["cover_height" if is_cover else "height"]
            family, total = self.params["family"], self.params["steps"]
            every = max(1, total // 12)

            def on_step_end(pipe, step, timestep, callback_kwargs):
                if should_stop is not None and should_stop():
                    pipe._interrupt = True
                if on_progress is not None:
                    preview = None
                    if step % every == 0 or step == total - 1:
                        try:
                            preview = latents_to_preview(callback_kwargs["latents"], family)
                        except Exception:  # noqa: BLE001 - la vista previa es opcional
                            preview = None
                    on_progress(step + 1, total, preview)
                return callback_kwargs

            kwargs = dict(prompt=prompt, width=width, height=height, num_inference_steps=total,
                          guidance_scale=self.params["guidance"],
                          generator=torch.Generator(device="cpu").manual_seed(final_seed),
                          callback_on_step_end=on_step_end,
                          callback_on_step_end_tensor_inputs=["latents"])
            if family == "flux":
                kwargs["max_sequence_length"] = self.params["max_tokens"]
            elif self.params.get("negative"):
                kwargs["negative_prompt"] = NEGATIVE_PROMPT
            log.info("Generando imagen %s (seed=%d): %s", "portada" if is_cover else index, final_seed, prompt)
            image = self._pipe(**kwargs).images[0]
            if should_stop is not None and should_stop():
                raise Cancelled()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            image.convert("RGB").save(out_path, format="JPEG", quality=90)
            return {"prompt": prompt, "seed": final_seed, "width": width, "height": height,
                    "steps": total, "guidance": self.params["guidance"]}


# ============================================================================
# 8. GENERACIÓN DE AUDIO (Piper TTS en español + ambiente opcional)
# ============================================================================
_VOICE_RE = re.compile(r"^(es_[A-Z]{2})-([a-z0-9_]+)-(x_low|low|medium|high)$")


def voice_files(voice: str) -> tuple[str, str]:
    """Rutas en rhasspy/piper-voices de una voz en español (p. ej. es_MX-ald-medium)."""
    match = _VOICE_RE.match(voice)
    if not match:
        raise ValueError(f"PIPER_VOICE inválida: {voice}. Debe ser una voz en español, "
                         "por ejemplo es_MX-ald-medium o es_ES-davefx-medium.")
    lang_code, name, quality = match.groups()
    base = f"es/{lang_code}/{name}/{quality}/{voice}.onnx"
    return base, base + ".json"


def prepare_for_speech(text: str) -> str:
    """Normalización tipográfica; NO cambia las palabras del texto."""
    text = text.replace("—", ", ").replace("–", ", ").replace("…", "...")
    text = re.sub(r"[«»“”\"]", "", text)
    return re.sub(r"\s+", " ", text).strip()


class PiperNarrator:
    """Narración local (ONNX). El audio contiene solo la voz: sin música ni efectos."""

    def __init__(self, voice: str, models_dir: Path, length_scale: float = 1.0):
        self.voice_name = voice
        self.models_dir = models_dir / "piper"
        self.length_scale = length_scale
        self._voice = None
        self._lock = threading.Lock()

    def load(self):
        if self._voice is not None:
            return
        from huggingface_hub import hf_hub_download
        from piper import PiperVoice

        onnx_file, json_file = voice_files(self.voice_name)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        model_path = hf_hub_download(PIPER_REPO, onnx_file, local_dir=self.models_dir)
        config_path = hf_hub_download(PIPER_REPO, json_file, local_dir=self.models_dir)
        log.info("Cargando voz Piper %s", self.voice_name)
        self._voice = PiperVoice.load(model_path, config_path=config_path)

    def synthesize(self, text: str, out_path: Path) -> dict:
        with self._lock:
            self.load()
            from piper import SynthesisConfig

            out_path.parent.mkdir(parents=True, exist_ok=True)
            config = SynthesisConfig(length_scale=self.length_scale, normalize_audio=True)
            with wave.open(str(out_path), "wb") as wav_file:
                self._voice.synthesize_wav(prepare_for_speech(text), wav_file, syn_config=config)
            with wave.open(str(out_path), "rb") as wav_file:
                duration = wav_file.getnframes() / float(wav_file.getframerate())
            if duration < 0.5:
                raise RuntimeError("Piper generó un audio vacío.")
            return {"voice": self.voice_name, "duration": round(duration, 2)}


class AmbientGenerator:
    """Ambiente sonoro OPCIONAL con AudioLDM (ENABLE_AMBIENT_SOUND=true).
    Desactivado por defecto: la especificación prohíbe sonidos ambientales;
    el PDF del curso los pide. Se guarda en una pista separada de la voz.
    Licencia CC BY-NC-SA 4.0 (uso no comercial)."""

    NEGATIVE = "music, melody, singing, speech, voice, talking, low quality"

    def __init__(self, model_id: str, device: str, seconds: float = 10.0, steps: int = 25):
        self.model_id, self.device, self.seconds, self.steps = model_id, device, seconds, steps
        self._pipe = None
        self._lock = threading.Lock()

    def _load(self):
        if self._pipe is not None:
            return
        import torch
        from diffusers import AudioLDMPipeline

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._pipe = AudioLDMPipeline.from_pretrained(self.model_id, torch_dtype=dtype).to(self.device)
        self._pipe.set_progress_bar_config(disable=True)

    def unload(self):
        with self._lock:
            self._pipe = None
            free_memory()

    def generate(self, description: str, seed: int, out_path: Path) -> dict:
        with self._lock:
            self._load()
            import numpy as np
            import torch

            prompt = f"ambient soundscape, {description}, no music, no voices"
            audio = self._pipe(prompt, negative_prompt=self.NEGATIVE, num_inference_steps=self.steps,
                               audio_length_in_s=self.seconds,
                               generator=torch.Generator(device="cpu").manual_seed(seed)).audios[0]
            peak = float(np.max(np.abs(audio))) or 1.0
            pcm = (audio / peak * 0.8 * 32767).astype(np.int16)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(out_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(pcm.tobytes())
            return {"prompt": prompt, "seconds": self.seconds}


# ============================================================================
# 9. CONTROL DE ESTADOS
# ============================================================================
# Cada recurso tiene su propio estado:  pending → running → done | error
#   portada:   book["cover"]
#   escena N:  scene["writing"] (texto), scene["image"], scene["audio"], scene["ambient"]
# Los archivos se nombran con el índice de su escena (scene_N.jpg, scene_N.wav),
# así el texto, la imagen y el audio de la escena N nunca se mezclan con otra.
# Estado del libro: queued → running → completed | completed_with_errors,
# además de failed (falló el plan), cancelled (detenido; se puede reanudar) y retrying.
ASSET_KEYS = ("writing", "image", "audio", "ambient")
API_KIND = {"text": "writing", "image": "image", "audio": "audio", "ambient": "ambient"}
BUILDING = ("queued", "running")
_ID_RE = re.compile(r"^[a-f0-9]{12}$")


class BookNotFound(KeyError):
    pass


def new_asset(enabled: bool = True) -> dict | None:
    if not enabled:
        return None
    return {"status": "pending", "url": None, "error": None, "meta": None, "version": 0}


def iter_assets(book: dict):
    if book.get("cover"):
        yield book["cover"]
    for scene in book.get("scenes", []):
        for key in ASSET_KEYS:
            if scene.get(key):
                yield scene[key]


def get_asset(book: dict, kind: str, index: int) -> dict:
    """kind: cover | text | image | audio | ambient."""
    if kind == "cover":
        return book["cover"]
    key = API_KIND.get(kind)
    if key is None or not 1 <= index <= len(book["scenes"]):
        raise KeyError(f"{kind}:{index}")
    asset = book["scenes"][index - 1].get(key)
    if asset is None:
        raise KeyError(f"{kind}:{index}")
    return asset


def file_for(kind: str, index: int) -> str:
    if kind == "cover":
        return "cover.jpg"
    return {"image": f"scene_{index}.jpg", "audio": f"scene_{index}.wav",
            "ambient": f"scene_{index}_ambient.wav"}[kind]


def count_progress(book: dict) -> int:
    done = 1 if book.get("planning", {}).get("status") in ("done", "error") else 0
    return done + sum(1 for a in iter_assets(book) if a["status"] in ("done", "error"))


def is_out_of_memory(exc: BaseException) -> bool:
    """Falta de memoria de GPU (CUDA) o de RAM (CPU)."""
    text = str(exc).lower()
    return (isinstance(exc, MemoryError) or exc.__class__.__name__ == "OutOfMemoryError"
            or "out of memory" in text or "not enough memory" in text or "defaultcpuallocator" in text)


def friendly_error(exc: Exception) -> str:
    """Mensajes de error claros para el usuario."""
    if isinstance(exc, ModuleNotFoundError):
        return (f"Falta la dependencia '{exc.name}'. Instala los requisitos "
                "(pip install -r requirements.txt) y PyTorch.")
    text = str(exc) or exc.__class__.__name__
    lowered = text.lower()
    if is_out_of_memory(exc):
        return ("Memoria insuficiente para el modelo, incluso tras liberar memoria y reintentar. "
                "Usa STORYBOOK_PROFILE=accesible, cierra juegos u otros programas que usen la GPU, "
                "o activa LOW_MEMORY=true / CPU_OFFLOAD=true en el .env.")
    if "gated" in lowered or "401" in lowered or "403" in lowered:
        return ("El modelo requiere aceptar su licencia en Hugging Face y configurar HF_TOKEN "
                "en el archivo .env del servidor.")
    if any(k in lowered for k in ("offline", "connection", "couldn't connect", "can't load",
                                   "does not appear to have", "not found on the hub", "localentrynotfound")):
        return ("No se encontraron los pesos del modelo en la caché local. Ejecuta "
                "«python main.py --descargar-modelos» con conexión a Internet.")
    return text[:300]


class BookStore:
    """Persistencia en data/books/<id>/book.json (escritura atómica) + recursos."""

    def __init__(self, data_dir: Path):
        self.root = data_dir / "books"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._cache: dict[str, dict] = {}

    def book_dir(self, book_id: str) -> Path:
        if not _ID_RE.match(book_id):
            raise BookNotFound(book_id)
        return self.root / book_id

    def create(self, book: dict) -> dict:
        with self._lock:
            self.book_dir(book["id"]).mkdir(parents=True, exist_ok=True)
            self._write(book)
            return deepcopy(book)

    def get(self, book_id: str) -> dict:
        with self._lock:
            if book_id not in self._cache:
                path = self.book_dir(book_id) / "book.json"
                if not path.exists():
                    raise BookNotFound(book_id)
                self._cache[book_id] = json.loads(path.read_text(encoding="utf-8"))
            return deepcopy(self._cache[book_id])

    def update(self, book_id: str, mutate) -> dict:
        with self._lock:
            book = self.get(book_id)
            mutate(book)
            book["updated_at"] = time.time()
            self._write(book)
            return deepcopy(book)

    def list(self, limit: int = 20) -> list[dict]:
        books = []
        for path in self.root.glob("*/book.json"):
            try:
                books.append(self.get(path.parent.name))
            except (BookNotFound, json.JSONDecodeError):
                continue
        books.sort(key=lambda b: b.get("created_at", 0), reverse=True)
        return books[:limit]

    def recover_interrupted(self):
        """Tras reiniciar el servidor, los libros a medias quedan "detenidos" y se
        pueden reanudar conservando lo ya generado."""
        for book in self.list(limit=10_000):
            if book["status"] in ("queued", "running", "retrying"):
                def mutate(b):
                    for asset in iter_assets(b):
                        if asset["status"] == "running":
                            asset["status"] = "pending"
                    if b.get("planning", {}).get("status") == "running":
                        b["planning"]["status"] = "pending"
                    b.update(status="cancelled", building_page=None,
                             error="La generación se interrumpió al reiniciar el servidor.")
                self.update(book["id"], mutate)

    def _write(self, book: dict):
        self._cache[book["id"]] = deepcopy(book)
        path = self.book_dir(book["id"]) / "book.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(book, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)


# ============================================================================
# 10. CREACIÓN DE PORTADA Y ESCENAS (construcción progresiva)
# ============================================================================
# Orden de construcción = orden en que el usuario lo ve:
#   1. Plan (Qwen3): título, biblia visual de personajes, 5 escenas.
#   2. PORTADA: ilustración (pasos reales de difusión + vista previa) y título.
#   3. ESCENA 1..5, una tras otra:
#        a. texto en streaming real;
#        b. ilustración y narración en paralelo (la imagen depende del plan, la
#           voz del texto ya terminado);
#        c. [opcional] ambiente.
#      La escena N+1 no empieza hasta que la escena N está completa o con error.
#   4. Revisión: reintentos pedidos durante la construcción, en orden.
# El navegador consulta GET /api/books/{id} cada ~0,5 s: recibe el estado de cada
# recurso, el texto que se está escribiendo y la vista previa de la imagen.
@dataclass
class Services:
    writer: StoryWriter
    images: Any
    narrator: Any
    ambient: Any | None = None
    low_memory: bool = False
    models: dict | None = None


class QueueFull(RuntimeError):
    pass


class BookBuilder:
    def __init__(self, store: BookStore, services: Services, max_pending: int = 3,
                 abandon_timeout: float = 180.0):
        self.store = store
        self.services = services
        self.max_pending = max_pending
        self.abandon_timeout = abandon_timeout  # 0 = nunca cancelar por abandono
        self._queue: queue.Queue = queue.Queue()
        self._pending = 0
        self._pending_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._job_lock = threading.Lock()   # libro en construcción / reintentos
        self._active_build: str | None = None
        self._cancel: set[str] = set()
        self._last_seen: dict[str, float] = {}
        # Datos "en vivo" (solo memoria): texto que se está escribiendo y vista previa.
        self._live: dict[str, dict] = {}
        self._live_lock = threading.Lock()
        threading.Thread(target=self._worker, name="storybook-worker", daemon=True).start()

    # ---------------------------------------------------- estado compartido
    def _update(self, book_id: str, mutate) -> dict:
        with self._state_lock:
            def wrapped(b):
                mutate(b)
                b["rev"] = b.get("rev", 0) + 1       # revisión creciente
                b["progress"]["done"] = count_progress(b)
            return self.store.update(book_id, wrapped)

    def snapshot(self, book_id: str) -> dict:
        book = self.store.get(book_id)
        with self._live_lock:
            live = self._live.get(book_id) or {"drafts": {}, "previews": {}}
            book["live"] = {"drafts": dict(live["drafts"]), "previews": dict(live["previews"])}
        return book

    def _live_for(self, book_id: str) -> dict:
        return self._live.setdefault(book_id, {"drafts": {}, "previews": {}})

    def _draft(self, book_id: str, index: int, text: str, attempt: int):
        with self._live_lock:
            self._live_for(book_id)["drafts"][str(index)] = {"text": text, "attempt": attempt}

    def _preview(self, book_id: str, kind: str, index: int, step: int, total: int, jpeg: bytes | None):
        key = f"{kind}:{index}"
        with self._live_lock:
            previews = self._live_for(book_id)["previews"]
            image = ("data:image/jpeg;base64," + base64.b64encode(jpeg).decode()) if jpeg \
                else previews.get(key, {}).get("image")
            previews[key] = {"kind": kind, "index": index, "step": step, "total": total, "image": image}

    def _clear_live(self, book_id: str, kind: str, index: int):
        with self._live_lock:
            live = self._live.get(book_id)
            if live:
                if kind == "text":
                    live["drafts"].pop(str(index), None)
                else:
                    live["previews"].pop(f"{kind}:{index}", None)

    def _drop_live(self, book_id: str):
        with self._live_lock:
            self._live.pop(book_id, None)

    def _step(self, book_id: str, text: str, page: int | None = None):
        def mutate(b):
            b["progress"]["step"] = text
            if page is not None:
                b["building_page"] = page
        self._update(book_id, mutate)

    # ------------------------------------------------- cancelación / abandono
    def touch(self, book_id: str):
        """Alguien está mirando el libro (cada consulta del navegador)."""
        self._last_seen[book_id] = time.monotonic()

    def _should_stop(self, book_id: str):
        def stop() -> bool:
            if book_id not in self._cancel and self.abandon_timeout:
                seen = self._last_seen.get(book_id, time.monotonic())
                if time.monotonic() - seen > self.abandon_timeout:
                    log.info("Nadie observa el libro %s: se detiene la generación.", book_id)
                    self._cancel.add(book_id)
            return book_id in self._cancel
        return stop

    def _check_cancel(self, book_id: str):
        if self._should_stop(book_id)():
            raise Cancelled()

    # -------------------------------------------------------- API pública
    def submit_retry_failed(self, book_id: str) -> dict:
        """Marca como pendientes TODOS los recursos con error y los regenera en orden."""
        with self._job_lock:
            book = self.store.get(book_id)
            if book["status"] == "failed" or not book.get("plan"):
                raise ValueError("La historia no se planificó; usa «Volver a intentar».")
            if book["status"] == "cancelled":
                raise ValueError("La generación está detenida: pulsa «Reanudar» para continuar.")
            if book["status"] == "retrying":
                raise ValueError("Ya se están regenerando recursos.")
            failed = [a for a in iter_assets(book) if a["status"] == "error"]
            if not failed:
                raise ValueError("No hay recursos con error.")
            building = self._active_build == book_id or book["status"] in BUILDING

            def mutate(b):
                if b["cover"]["status"] == "error":
                    b["cover"].update(status="pending", error=None)
                for sc in b["scenes"]:
                    text_failed = sc["writing"]["status"] == "error"
                    for key in ASSET_KEYS:
                        a = sc.get(key)
                        if a and (a["status"] == "error" or (key == "audio" and text_failed)):
                            a.update(status="pending", error=None)
                if not building:
                    b["status"] = "retrying"
                    b["progress"]["step"] = "Reintentando los recursos con error…"
            book = self._update(book_id, mutate)
            if not building:
                self._reserve(force=True)
                self._queue.put(("retry_all", book_id, None, None))
            return book

    def _reserve(self, force: bool = False):
        with self._pending_lock:
            if not force and self._pending >= self.max_pending:
                raise QueueFull("Hay demasiados libros en cola. Intenta de nuevo en unos minutos.")
            self._pending += 1

    def submit_book(self, title: str, genre: str, description: str) -> dict:
        get_genre(genre)
        self._reserve()
        ambient_on = self.services.ambient is not None
        per_scene = 3 + (1 if ambient_on else 0)
        book = {
            "id": secrets.token_hex(6), "created_at": time.time(), "rev": 0,
            "status": "queued", "error": None,
            "request": {"title": title, "genre": genre, "description": description},
            "title": title or None, "genre": genre, "seed": random.randint(0, 2**31 - 1),
            "models": self.services.models or {},
            "progress": {"done": 0, "total": 2 + NUM_SCENES * per_scene, "step": "En cola…"},
            "building_page": 0, "planning": new_asset(), "plan": None,
            "cover": new_asset(), "scenes": [], "ambient_enabled": ambient_on,
        }
        self.store.create(book)
        self.touch(book["id"])
        self._queue.put(("build", book["id"], None, None))
        return book

    def submit_resume(self, book_id: str) -> dict:
        book = self.store.get(book_id)
        if book["status"] not in ("cancelled", "failed"):
            raise ValueError("Solo se puede reanudar un libro detenido o fallido.")
        self._reserve()
        self.touch(book_id)

        def mutate(b):
            if b["planning"]["status"] == "error":
                b["planning"].update(status="pending", error=None)
            b.update(status="queued", error=None, building_page=0 if not b["plan"] else b.get("building_page"))
            b["progress"]["step"] = "En cola para reanudar…"
        book = self._update(book_id, mutate)
        self._queue.put(("build", book_id, None, None))
        return book

    def cancel(self, book_id: str) -> dict:
        book = self.store.get(book_id)
        if book["status"] not in BUILDING:
            return book
        self._cancel.add(book_id)
        if book["status"] == "queued":
            return self._update(book_id, lambda b: b.update(
                status="cancelled", building_page=None, error="Generación detenida por el usuario."))
        return book

    def submit_retry(self, book_id: str, kind: str, index: int) -> dict:
        with self._job_lock:
            book = self.store.get(book_id)
            if book["status"] == "failed" or not book.get("plan"):
                raise ValueError("La historia no se planificó; usa «Volver a intentar».")
            if book["status"] == "cancelled":
                raise ValueError("La generación está detenida: pulsa «Reanudar» para continuar.")
            asset = get_asset(book, kind, index)
            if asset["status"] in ("pending", "running"):
                raise ValueError("Ese recurso ya está en proceso.")
            if kind == "audio" and not book["scenes"][index - 1].get("text"):
                raise ValueError("Primero hay que generar el texto de la escena.")
            building = self._active_build == book_id or book["status"] in BUILDING

            def mutate(b):
                get_asset(b, kind, index).update(status="pending", error=None)
                if kind == "text":  # la narración debe coincidir con el nuevo texto
                    b["scenes"][index - 1]["audio"].update(status="pending", error=None)
                if not building:
                    b["status"] = "retrying"
            book = self._update(book_id, mutate)
            if not building:  # durante la construcción, la propia construcción lo atiende
                self._reserve(force=True)
                self._queue.put(("retry", book_id, kind, index))
            return book

    # ------------------------------------------------------------- worker
    def _worker(self):
        while True:
            job, book_id, kind, index = self._queue.get()
            try:
                if job == "build":
                    self.build_book(book_id)
                elif job == "retry_all":
                    try:
                        self._sweep(book_id)
                    except Cancelled:
                        pass
                    self._finalize(book_id)
                else:
                    self._retry_job(book_id, kind, index)
            except Exception:  # noqa: BLE001 - el hilo nunca debe morir
                log.error("Error inesperado en %s/%s:\n%s", job, book_id, traceback.format_exc())
            finally:
                with self._pending_lock:
                    self._pending -= 1
                self._queue.task_done()

    def _retry_job(self, book_id: str, kind: str, index: int):
        try:
            self.generate_asset(book_id, kind, index)
            if kind == "text":
                self.generate_asset(book_id, "audio", index)
        except Cancelled:
            pass
        self._finalize(book_id)

    def build_book(self, book_id: str):
        """Construye el libro completo en orden: plan → portada → escenas 1..5."""
        book = self.store.get(book_id)
        if book["status"] == "cancelled":  # detenido mientras esperaba en la cola
            self._cancel.discard(book_id)
            return
        with self._job_lock:
            self._active_build = book_id
        self._update(book_id, lambda b: b.update(status="running", error=None))
        try:
            if not self.store.get(book_id)["plan"] and not self.create_plan(book_id):
                return
            self.create_cover(book_id)
            for n in range(1, NUM_SCENES + 1):
                self._check_cancel(book_id)
                self._sweep(book_id, before=n)
                self.create_scene(book_id, n)
            # Reintentos pendientes y cierre, bajo el mismo candado que acepta reintentos.
            while True:
                self._check_cancel(book_id)
                with self._job_lock:
                    if not self._pending_assets(self.store.get(book_id)):
                        self._active_build = None
                        self._finalize(book_id)
                        break
                self._sweep(book_id)
        except Cancelled:
            log.info("Libro %s detenido.", book_id)

            def mutate(b):
                for a in iter_assets(b):
                    if a["status"] == "running":
                        a["status"] = "pending"
                if b["planning"]["status"] == "running":
                    b["planning"]["status"] = "pending"
                b.update(status="cancelled", building_page=None,
                         error="Generación detenida. Lo ya creado se conserva; puedes reanudarla.")
                b["progress"]["step"] = "Generación detenida"
            self._update(book_id, mutate)
            self._drop_live(book_id)
        finally:
            self._cancel.discard(book_id)
            with self._job_lock:
                if self._active_build == book_id:
                    self._active_build = None

    def create_plan(self, book_id: str) -> bool:
        s = self.services
        req = self.store.get(book_id)["request"]

        def start(b):
            b["planning"]["status"] = "running"
            b["building_page"] = 0
            b["progress"]["step"] = "Analizando la idea y diseñando personajes…"
        self._update(book_id, start)
        self._unload_for("text")
        try:
            plan = s.writer.create_plan(req["description"], req["genre"], req["title"],
                                        should_stop=self._should_stop(book_id))
        except Cancelled:
            raise
        except Exception as exc:  # noqa: BLE001
            log.error("Fallo al planificar:\n%s", traceback.format_exc())
            message = f"No se pudo planificar la historia: {friendly_error(exc)}"

            def failed(b):
                b["planning"].update(status="error", error=message)
                b.update(status="failed", error=message, building_page=0)
            self._update(book_id, failed)
            return False
        ambient_on = s.ambient is not None

        def set_plan(b):
            b["plan"] = plan
            b["title"] = plan["title"]
            b["planning"].update(status="done", error=None)
            b["scenes"] = [{
                "index": sc["index"], "id": f"scene_{sc['index']}", "title": sc["title"], "text": None,
                "writing": new_asset(), "image": new_asset(), "audio": new_asset(),
                "ambient": new_asset(ambient_on),
            } for sc in plan["scenes"]]
        self._update(book_id, set_plan)
        return True

    def create_cover(self, book_id: str):
        """CREACIÓN DE PORTADA: ilustración independiente de las 5 escenas."""
        book = self.store.get(book_id)
        if book["cover"]["status"] == "pending":
            self._step(book_id, "Ilustrando la portada…", page=0)
            try:
                cover = self.services.writer.describe_cover(book["plan"], should_stop=self._should_stop(book_id))
                self._update(book_id, lambda b: b["plan"].update(cover=cover))
            except Cancelled:
                raise
            except Exception:  # noqa: BLE001 - se usa la descripción del plan
                log.warning("No se pudo describir la portada:\n%s", traceback.format_exc())
            self.generate_asset(book_id, "cover", 0)

    def create_scene(self, book_id: str, n: int):
        """CREACIÓN DE ESCENA n: texto (+ descripción visual del texto) →
        (ilustración ∥ narración) → [ambiente]."""
        s = self.services
        self._step(book_id, f"Escribiendo la escena {n}…", page=n)
        scene = self.store.get(book_id)["scenes"][n - 1]
        if scene["writing"]["status"] == "pending":
            self.generate_asset(book_id, "text", n)
        scene = self.store.get(book_id)["scenes"][n - 1]
        # La ilustración se basa en el texto ya escrito; la narración también.
        # Ambas se generan a la vez.
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix=f"escena{n}") as pool:
            futures = []
            if scene["image"]["status"] == "pending":
                self._step(book_id, f"Ilustrando y narrando la escena {n}…")
                futures.append(pool.submit(self.generate_asset, book_id, "image", n))
            if scene["audio"]["status"] == "pending":
                futures.append(pool.submit(self.generate_asset, book_id, "audio", n))
            for f in futures:
                f.result()  # propaga Cancelled
        if s.ambient is not None and self.store.get(book_id)["scenes"][n - 1]["ambient"]["status"] == "pending":
            self._step(book_id, f"Creando el ambiente sonoro de la escena {n}…")
            self.generate_asset(book_id, "ambient", n)

    @staticmethod
    def _pending_assets(book: dict) -> list[tuple[str, int]]:
        found = [("cover", 0)] if book["cover"]["status"] == "pending" else []
        for sc in book["scenes"]:
            for kind, key in (("text", "writing"), ("image", "image"), ("audio", "audio"), ("ambient", "ambient")):
                if sc.get(key) and sc[key]["status"] == "pending":
                    found.append((kind, sc["index"]))
        return found

    def _sweep(self, book_id: str, before: int | None = None):
        """Genera recursos pendientes (reintentos) de páginas anteriores."""
        for kind, index in self._pending_assets(self.store.get(book_id)):
            if before is not None and index >= before:
                continue
            self._check_cancel(book_id)
            what = {"cover": "la portada", "text": "el texto", "image": "la imagen", "audio": "el audio",
                    "ambient": "el ambiente"}[kind]
            self._step(book_id, f"Reintentando {what}" + (f" de la escena {index}…" if index else "…"))
            self.generate_asset(book_id, kind, index)

    def _recover_from_oom(self, kind: str):
        """Activa el modo de memoria baja para el resto de la sesión, descarga
        los modelos que no se necesitan para `kind` y vacía la caché de la GPU."""
        s = self.services
        if not s.low_memory:
            log.warning("Se activa el modo de memoria baja (LOW_MEMORY) para el resto de la sesión.")
        s.low_memory = True
        self._unload_for(kind)
        free_memory()

    def _unload_for(self, kind: str):
        """Con LOW_MEMORY, solo un modelo pesado en memoria a la vez."""
        s = self.services
        if not s.low_memory:
            return
        if kind == "text":
            for heavy in (s.images, s.ambient):
                if heavy is not None and hasattr(heavy, "unload"):
                    heavy.unload()
        elif kind in ("cover", "image", "ambient") and hasattr(s.writer.llm, "unload"):
            s.writer.llm.unload()
            if kind == "ambient" and hasattr(s.images, "unload"):
                s.images.unload()

    def generate_asset(self, book_id: str, kind: str, index: int):
        """Genera UN recurso. Los errores quedan como estado "error" del recurso
        (salvo Cancelled). El resultado se asigna siempre al (libro, tipo, índice)
        que lo lanzó y solo si su número de intento sigue vigente."""
        s = self.services
        book = self.store.get(book_id)
        plan, genre = book["plan"], book["genre"]
        attempt = secrets.token_hex(4)
        self._update(book_id, lambda b: get_asset(b, kind, index).update(status="running", error=None,
                                                                          attempt=attempt))
        stop = self._should_stop(book_id)
        self._unload_for(kind)

        def work() -> tuple[dict, str | None]:
            if kind == "text":
                previous = [sc["text"] for sc in book["scenes"][:index - 1] if sc.get("text")]
                scene_text = s.writer.write_scene(plan, genre, index, previous,
                                                  on_draft=lambda t, a: self._draft(book_id, index, t, a),
                                                  should_stop=stop)
                # La ilustración se describe a partir de lo que dice ESTE texto.
                # Los personajes de la imagen son los que se nombran en ESTE texto.
                named = [c["name"] for c in plan["characters"]
                         if c["name"] and re.search(rf"\b{re.escape(c['name'].lower())}\b", scene_text.lower())]
                if named:
                    plan["scenes"][index - 1]["characters"] = named
                visual = s.writer.describe_scene(plan, index, scene_text, should_stop=stop)
                return {"words": len(scene_text.split()), "visual": visual, "characters": named}, scene_text
            out_path = self.store.book_dir(book_id) / file_for(kind, index)
            if kind in ("cover", "image"):
                scene = None if kind == "cover" else plan["scenes"][index - 1]
                return s.images.generate(
                    plan, genre, scene, book["seed"], out_path,
                    on_progress=lambda st, tot, jpg: self._preview(book_id, kind, index, st, tot, jpg),
                    should_stop=stop), None
            if kind == "audio":
                current_text = book["scenes"][index - 1]["text"]
                if not current_text:
                    raise RuntimeError("La narración necesita el texto de la escena.")
                return s.narrator.synthesize(current_text, out_path), None
            sc = plan["scenes"][index - 1]
            return s.ambient.generate(sc.get("sound") or sc["visual"], book["seed"] + index, out_path), None

        try:
            try:
                meta, text = work()
            except Cancelled:
                raise
            except Exception as exc:  # noqa: BLE001
                if not is_out_of_memory(exc):
                    raise
                # Falta de memoria: se libera la memoria (un solo modelo pesado a la
                # vez a partir de ahora) y se reintenta UNA vez, para que un fallo
                # no arrastre a todos los recursos siguientes.
                log.warning("Memoria insuficiente al generar %s %s; se libera memoria y se reintenta.",
                            kind, index)
                self._recover_from_oom(kind)
                meta, text = work()
            if stop():
                raise Cancelled()
        except Cancelled:
            self._clear_live(book_id, kind, index)
            raise
        except Exception as exc:  # noqa: BLE001
            log.error("Fallo al generar %s %s:\n%s", kind, index, traceback.format_exc())
            message = friendly_error(exc)

            def failed(b):
                a = get_asset(b, kind, index)
                if a.get("attempt") == attempt:
                    a.update(status="error", error=message)
                    if kind == "text":  # sin texto no puede haber narración
                        b["scenes"][index - 1]["audio"].update(
                            status="error", error="Se necesita el texto de la escena para narrarla.")
            self._update(book_id, failed)
            self._clear_live(book_id, kind, index)
            return

        def done(b):
            a = get_asset(b, kind, index)
            if a.get("attempt") != attempt:  # resultado obsoleto: se descarta
                return
            a["version"] = a.get("version", 0) + 1
            a.update(status="done", error=None, meta=meta)
            if kind == "text":
                b["scenes"][index - 1]["text"] = text
                b["scenes"][index - 1]["visual"] = meta.get("visual")
                if meta.get("visual"):
                    b["plan"]["scenes"][index - 1]["visual"] = meta["visual"]
                if meta.get("characters"):
                    b["plan"]["scenes"][index - 1]["characters"] = meta["characters"]
            else:
                a["url"] = f"/media/{book_id}/{file_for(kind, index)}?v={a['version']}"
        self._update(book_id, done)
        self._clear_live(book_id, kind, index)

    def _finalize(self, book_id: str):
        def mutate(b):
            assets = list(iter_assets(b))
            if any(a["status"] in ("pending", "running") for a in assets):
                return
            failed = sum(1 for a in assets if a["status"] == "error")
            b["status"] = "completed_with_errors" if failed else "completed"
            b["building_page"] = None
            b["error"] = None
            b["progress"]["step"] = (f"Libro listo con {failed} recurso(s) por reintentar"
                                     if failed else "¡Tu libro está listo!")
        self._update(book_id, mutate)
        self._drop_live(book_id)

    def wait_idle(self):
        self._queue.join()


# ============================================================================
# 11. SERVIDOR WEB (API)
# ============================================================================
_MEDIA_RE = re.compile(r"^(cover|scene_[1-5](_ambient)?)\.(jpg|wav)$")


class BookRequest(BaseModel):
    title: str = Field(default="", max_length=90)
    genre: str
    description: str = Field(min_length=10, max_length=1500)

    @field_validator("title", "description")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()

    @field_validator("description")
    @classmethod
    def description_not_blank(cls, value: str) -> str:
        if len(value) < 10:
            raise ValueError("Describe tu idea con al menos 10 caracteres.")
        return value

    @field_validator("genre")
    @classmethod
    def valid_genre(cls, value: str) -> str:
        if value not in GENRES:
            raise ValueError("Selecciona un género válido.")
        return value


class RetryRequest(BaseModel):
    kind: str = Field(pattern="^(cover|text|image|audio|ambient)$")
    index: int = Field(default=0, ge=0, le=5)


def build_services(settings: Settings) -> Services:
    """Servicios reales; los modelos se cargan de forma diferida."""
    ambient = (AmbientGenerator(settings.ambient_model, settings.device, settings.ambient_seconds,
                                settings.ambient_steps) if settings.enable_ambient else None)
    return Services(
        writer=StoryWriter(LocalLLM(settings.text_model, settings.text_params, settings.device)),
        images=ImageGenerator(settings.image_model, settings.image_params, settings.device,
                              settings.cpu_offload, settings.image_seed_mode),
        narrator=PiperNarrator(settings.piper_voice, settings.models_dir, settings.piper_length_scale),
        ambient=ambient, low_memory=settings.low_memory,
        models=settings.public_info(),
    )


def create_app(settings: Settings | None = None, services: Services | None = None) -> FastAPI:
    settings = settings or load_settings()
    services = services or build_services(settings)
    store = BookStore(settings.data_dir)
    store.recover_interrupted()
    builder = BookBuilder(store, services, settings.max_pending_jobs, settings.abandon_timeout)

    app = FastAPI(title="Storybook IA", version="2.0.0")
    app.state.builder = builder
    app.state.store = store

    def not_found(exc: Exception):
        return HTTPException(status_code=404, detail="Libro no encontrado.")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse(page_html(), headers={"Cache-Control": "no-cache"})

    @app.get("/api/health")
    def health():
        return {"status": "ok", "models": services.models or settings.public_info()}

    @app.get("/api/genres")
    def genres():
        return public_genres()

    @app.post("/api/books", status_code=202)
    def create_book(req: BookRequest):
        try:
            book = builder.submit_book(req.title, req.genre, req.description)
        except QueueFull as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        return {"id": book["id"], "status": book["status"]}

    @app.get("/api/books")
    def list_books():
        return [{"id": b["id"], "title": b.get("title") or b["request"].get("title"), "genre": b["genre"],
                 "status": b["status"], "created_at": b["created_at"]} for b in store.list()]

    @app.get("/api/books/{book_id}")
    def get_book(book_id: str):
        """Estado completo + datos en vivo. El navegador lo consulta cada ~0,5 s."""
        try:
            book = builder.snapshot(book_id)
        except BookNotFound as exc:
            raise not_found(exc) from exc
        builder.touch(book_id)
        return book

    @app.post("/api/books/{book_id}/cancel", status_code=202)
    def cancel_book(book_id: str):
        try:
            book = builder.cancel(book_id)
        except BookNotFound as exc:
            raise not_found(exc) from exc
        return {"id": book["id"], "status": book["status"]}

    @app.post("/api/books/{book_id}/resume", status_code=202)
    def resume_book(book_id: str):
        try:
            book = builder.submit_resume(book_id)
        except BookNotFound as exc:
            raise not_found(exc) from exc
        except QueueFull as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"id": book["id"], "status": book["status"]}

    @app.post("/api/books/{book_id}/retry-failed", status_code=202)
    def retry_failed(book_id: str):
        try:
            book = builder.submit_retry_failed(book_id)
        except BookNotFound as exc:
            raise not_found(exc) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"id": book["id"], "status": book["status"]}

    @app.post("/api/books/{book_id}/retry", status_code=202)
    def retry_asset(book_id: str, req: RetryRequest):
        index = 0 if req.kind == "cover" else req.index
        if req.kind != "cover" and index < 1:
            raise HTTPException(status_code=422, detail="Indica la escena (1-5).")
        try:
            book = builder.submit_retry(book_id, req.kind, index)
        except BookNotFound as exc:
            raise not_found(exc) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Recurso no encontrado.") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"id": book["id"], "status": book["status"]}

    @app.get("/media/{book_id}/{filename}")
    def media(book_id: str, filename: str):
        if not _MEDIA_RE.match(filename):
            raise HTTPException(status_code=404)
        try:
            path = store.book_dir(book_id) / filename
        except BookNotFound as exc:
            raise HTTPException(status_code=404) from exc
        if not path.is_file():
            raise HTTPException(status_code=404)
        return FileResponse(path, media_type="image/jpeg" if filename.endswith(".jpg") else "audio/wav",
                            headers={"Cache-Control": "no-cache"})

    return app


# ============================================================================
# 12. INTERFAZ GRÁFICA (HTML + CSS)
# ============================================================================
FRONTEND_CSS = r"""
/* =========================================================================
   Identidad visual por género.
   Cada tema redefine las mismas variables; todo el libro (portada, fondos,
   páginas, adornos y controles) las consume, así el diseño es coherente.
   Se aplica con data-genre="<clave>" en <body> y en las tarjetas del selector.
   ========================================================================= */
:root,
[data-genre] {
  --bg-1: #f3ede4;
  --bg-2: #e3d6c3;
  --paper: #fffaf2;
  --paper-2: #f3e9d8;
  --paper-edge: #d9c7a8;
  --ink: #3b2f25;
  --ink-soft: #6f5f50;
  --accent: #9c5b2e;
  --accent-2: #d8a35d;
  --accent-ink: #ffffff;
  --cover-shade: rgba(30, 18, 8, 0.65);
  --cover-title: #fff7e6;
  --on-bg: var(--cover-title);
  --on-bg-shadow: 0 1px 3px rgba(0,0,0,.45);
  --glow: rgba(216, 163, 93, 0.45);
  --font-title: "Palatino Linotype", "Book Antiqua", Palatino, Georgia, serif;
  --font-body: "Iowan Old Style", "Palatino Linotype", Georgia, "Times New Roman", serif;
  --title-transform: none;
  --title-spacing: 0.01em;
  --title-style: normal;
  --ornament: "❦";
  --pattern: radial-gradient(circle at 20% 20%, rgba(255,255,255,.35) 0 2px, transparent 3px);
  --pattern-size: 26px 26px;
}

[data-genre="romance"] {
  --on-bg: var(--ink); --on-bg-shadow: 0 1px 0 rgba(255,255,255,.6);
  --bg-1: #fde8ee; --bg-2: #f5c3cf;
  --paper: #fff8f4; --paper-2: #fcebe6; --paper-edge: #efc1c8;
  --ink: #5b2633; --ink-soft: #8f5b66;
  --accent: #d4506d; --accent-2: #f3a3b5; --accent-ink: #fff;
  --cover-shade: rgba(110, 24, 48, 0.55); --cover-title: #fff3f5;
  --glow: rgba(243, 163, 181, 0.6);
  --font-title: "Palatino Linotype", "Book Antiqua", Palatino, Georgia, serif;
  --title-style: italic; --ornament: "♥";
  --pattern: radial-gradient(circle at 50% 50%, rgba(212,80,109,.12) 0 3px, transparent 4px);
}

[data-genre="terror"] {
  --bg-1: #1a1517; --bg-2: #050405;
  --paper: #e9e2d6; --paper-2: #d4cabb; --paper-edge: #8b7f70;
  --ink: #1f1416; --ink-soft: #5a4446;
  --accent: #8e1b1b; --accent-2: #c23b2c; --accent-ink: #fbeaea;
  --cover-shade: rgba(8, 0, 0, 0.78); --cover-title: #e8d6d6;
  --glow: rgba(194, 59, 44, 0.45);
  --font-title: Copperplate, "Copperplate Gothic Light", "Trajan Pro", Georgia, serif;
  --title-transform: uppercase; --title-spacing: 0.12em; --ornament: "✠";
  --pattern: repeating-linear-gradient(115deg, rgba(0,0,0,.08) 0 2px, transparent 2px 14px);
  --pattern-size: auto;
}

[data-genre="fantasia"] {
  --bg-1: #e8dcff; --bg-2: #6d55c9;
  --paper: #fbf8ff; --paper-2: #efe7ff; --paper-edge: #c8b6f2;
  --ink: #2e2153; --ink-soft: #6a5a95;
  --accent: #7b4dd6; --accent-2: #5fa8ff; --accent-ink: #fff;
  --cover-shade: rgba(36, 14, 92, 0.6); --cover-title: #fff9e3;
  --glow: rgba(160, 130, 255, 0.65);
  --font-title: "Trajan Pro", Optima, "Palatino Linotype", Palatino, serif;
  --title-spacing: 0.06em; --ornament: "✦";
  --pattern: radial-gradient(circle at 30% 30%, rgba(255,255,255,.7) 0 1px, transparent 2px),
             radial-gradient(circle at 70% 60%, rgba(160,130,255,.35) 0 1.5px, transparent 2.5px);
  --pattern-size: 34px 34px;
}

[data-genre="ciencia_ficcion"] {
  --bg-1: #0c1b3d; --bg-2: #05060f;
  --paper: #eef4ff; --paper-2: #dfe8fb; --paper-edge: #7d8fc9;
  --ink: #0f1d3d; --ink-soft: #48598a;
  --accent: #1f6fff; --accent-2: #9b5cff; --accent-ink: #fff;
  --cover-shade: rgba(4, 8, 32, 0.7); --cover-title: #e6f3ff;
  --glow: rgba(31, 111, 255, 0.6);
  --font-title: Eurostile, "Bank Gothic", "Segoe UI", system-ui, sans-serif;
  --font-body: "Segoe UI", system-ui, -apple-system, Roboto, sans-serif;
  --title-transform: uppercase; --title-spacing: 0.16em; --ornament: "◈";
  --pattern: linear-gradient(rgba(31,111,255,.08) 1px, transparent 1px),
             linear-gradient(90deg, rgba(31,111,255,.08) 1px, transparent 1px);
  --pattern-size: 22px 22px;
}

[data-genre="aventura"] {
  --bg-1: #e9e2c8; --bg-2: #6f8a4e;
  --paper: #fbf6e6; --paper-2: #efe5c8; --paper-edge: #c3ad7d;
  --ink: #33301c; --ink-soft: #6a6342;
  --accent: #3f7a3a; --accent-2: #a8743f; --accent-ink: #fff;
  --cover-shade: rgba(26, 34, 14, 0.62); --cover-title: #fff7dd;
  --glow: rgba(168, 116, 63, 0.45);
  --font-title: Rockwell, "Roboto Slab", "Courier New", Georgia, serif;
  --title-spacing: 0.04em; --ornament: "✧";
  --pattern: repeating-radial-gradient(circle at 0 0, transparent 0 18px, rgba(63,122,58,.07) 18px 19px);
  --pattern-size: auto;
}

[data-genre="accion"] {
  --bg-1: #2a2a2e; --bg-2: #101012;
  --paper: #f7f4ef; --paper-2: #e7e1d8; --paper-edge: #9b9387;
  --ink: #1d1b1a; --ink-soft: #57504a;
  --accent: #ff6a13; --accent-2: #ffb703; --accent-ink: #1b1007;
  --cover-shade: rgba(12, 8, 4, 0.7); --cover-title: #fff;
  --glow: rgba(255, 106, 19, 0.55);
  --font-title: Impact, "Haettenschweiler", "Arial Narrow Bold", sans-serif;
  --title-transform: uppercase; --title-spacing: 0.05em; --ornament: "⚡";
  --pattern: repeating-linear-gradient(135deg, rgba(255,106,19,.07) 0 6px, transparent 6px 18px);
  --pattern-size: auto;
}

[data-genre="misterio"] {
  --bg-1: #1c2738; --bg-2: #0a0f18;
  --paper: #f1eee6; --paper-2: #e0dbcf; --paper-edge: #8f8a7b;
  --ink: #18202c; --ink-soft: #4d5868;
  --accent: #c9a24a; --accent-2: #2d4a73; --accent-ink: #121820;
  --cover-shade: rgba(6, 10, 20, 0.72); --cover-title: #f3dfa6;
  --glow: rgba(201, 162, 74, 0.5);
  --font-title: Georgia, "Times New Roman", serif;
  --title-transform: uppercase; --title-spacing: 0.14em; --ornament: "❖";
  --pattern: radial-gradient(circle at 50% 50%, rgba(201,162,74,.12) 0 1px, transparent 2px);
}

[data-genre="suspenso"] {
  --bg-1: #1f3a3d; --bg-2: #0b1416;
  --paper: #f2f0ea; --paper-2: #dfe0d8; --paper-edge: #84908a;
  --ink: #142023; --ink-soft: #4a5b5e;
  --accent: #e0902f; --accent-2: #2c6e6a; --accent-ink: #1a1206;
  --cover-shade: rgba(4, 14, 16, 0.72); --cover-title: #fbe8c8;
  --glow: rgba(224, 144, 47, 0.45);
  --font-title: Georgia, "Times New Roman", serif;
  --title-spacing: 0.08em; --ornament: "◆";
  --pattern: repeating-linear-gradient(0deg, rgba(0,0,0,.05) 0 1px, transparent 1px 9px);
  --pattern-size: auto;
}

[data-genre="drama"] {
  --bg-1: #4a1f2b; --bg-2: #1e1418;
  --paper: #faf5ef; --paper-2: #eee3d8; --paper-edge: #b59f8c;
  --ink: #2e1c20; --ink-soft: #6c5358;
  --accent: #8c2f45; --accent-2: #c8a15a; --accent-ink: #fff;
  --cover-shade: rgba(40, 10, 18, 0.66); --cover-title: #fbeed8;
  --glow: rgba(200, 161, 90, 0.45);
  --font-title: "Palatino Linotype", "Book Antiqua", Palatino, serif;
  --title-style: italic; --ornament: "❧";
}

[data-genre="comedia"] {
  --on-bg: var(--ink); --on-bg-shadow: 0 1px 0 rgba(255,255,255,.6);
  --bg-1: #fff4c2; --bg-2: #ffc94d;
  --paper: #fffdf3; --paper-2: #fff1c9; --paper-edge: #f0c86a;
  --ink: #3a2a10; --ink-soft: #7a6236;
  --accent: #ff7a1a; --accent-2: #1fb8b0; --accent-ink: #fff;
  --cover-shade: rgba(70, 40, 0, 0.45); --cover-title: #fffbe8;
  --glow: rgba(31, 184, 176, 0.5);
  --font-title: "Chalkboard SE", "Comic Sans MS", "Marker Felt", "Trebuchet MS", sans-serif;
  --font-body: "Trebuchet MS", "Segoe UI", system-ui, sans-serif;
  --title-spacing: 0.02em; --ornament: "✺";
  --pattern: radial-gradient(circle at 50% 50%, rgba(255,122,26,.16) 0 4px, transparent 5px);
  --pattern-size: 30px 30px;
}

[data-genre="infantil"] {
  --on-bg: var(--ink); --on-bg-shadow: 0 1px 0 rgba(255,255,255,.6);
  --bg-1: #e4f6ff; --bg-2: #ffd9ec;
  --paper: #fffdf8; --paper-2: #fff1f6; --paper-edge: #f5c6dc;
  --ink: #3b3355; --ink-soft: #776d96;
  --accent: #ff8fb8; --accent-2: #7fd3c4; --accent-ink: #fff;
  --cover-shade: rgba(60, 40, 110, 0.38); --cover-title: #ffffff;
  --glow: rgba(255, 143, 184, 0.6);
  --font-title: "Comic Sans MS", "Chalkboard SE", "Comic Neue", "Trebuchet MS", sans-serif;
  --font-body: "Comic Sans MS", "Chalkboard SE", "Trebuchet MS", sans-serif;
  --ornament: "★";
  --pattern: radial-gradient(circle at 25% 25%, rgba(127,211,196,.3) 0 5px, transparent 6px),
             radial-gradient(circle at 75% 75%, rgba(255,143,184,.25) 0 4px, transparent 5px);
  --pattern-size: 44px 44px;
}

[data-genre="mitologia"] {
  --bg-1: #e6dcc0; --bg-2: #1d3a6b;
  --paper: #fdfaf1; --paper-2: #f1e9d2; --paper-edge: #c9b27a;
  --ink: #1f2a44; --ink-soft: #55607d;
  --accent: #b88a1e; --accent-2: #1d4f91; --accent-ink: #fff;
  --cover-shade: rgba(12, 22, 48, 0.62); --cover-title: #f7e3a3;
  --glow: rgba(184, 138, 30, 0.5);
  --font-title: "Trajan Pro", Optima, "Times New Roman", serif;
  --title-transform: uppercase; --title-spacing: 0.18em; --ornament: "☀";
  --pattern: repeating-linear-gradient(90deg, rgba(184,138,30,.08) 0 2px, transparent 2px 24px);
  --pattern-size: auto;
}

[data-genre="fabula"] {
  --bg-1: #eef0d6; --bg-2: #a9b37a;
  --paper: #fdf9ec; --paper-2: #f2ebd3; --paper-edge: #c9b98a;
  --ink: #3a3420; --ink-soft: #6f6644;
  --accent: #c17a2c; --accent-2: #6d8a3b; --accent-ink: #fff;
  --cover-shade: rgba(40, 34, 10, 0.55); --cover-title: #fff8e2;
  --glow: rgba(193, 122, 44, 0.45);
  --font-title: "Palatino Linotype", "Book Antiqua", Palatino, Georgia, serif;
  --ornament: "❀";
}

[data-genre="historico"] {
  --bg-1: #d9c7a3; --bg-2: #5b4127;
  --paper: #f6ecd6; --paper-2: #eadbb9; --paper-edge: #b39563;
  --ink: #3b2a18; --ink-soft: #6d5639;
  --accent: #7a4b22; --accent-2: #a8864a; --accent-ink: #fff;
  --cover-shade: rgba(40, 24, 8, 0.62); --cover-title: #f8ecd0;
  --glow: rgba(168, 134, 74, 0.45);
  --font-title: Garamond, "EB Garamond", "Times New Roman", serif;
  --title-spacing: 0.05em; --ornament: "⚜";
  --pattern: radial-gradient(ellipse at 50% 50%, rgba(122,75,34,.08) 0 40%, transparent 70%);
  --pattern-size: 120px 90px;
}

/* ===================== Base, formulario y progreso ===================== */
*, *::before, *::after { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0;
  min-height: 100vh;
  min-height: 100dvh;
  font-family: var(--font-body);
  color: var(--ink);
  background: var(--bg-2);
  transition: color .5s ease;
  overflow-x: hidden;
}
.backdrop {
  position: fixed; inset: 0; z-index: -1;
  background: linear-gradient(160deg, var(--bg-1), var(--bg-2));
}
.backdrop::before { content: ""; position: absolute; inset: 0; background: var(--pattern); background-size: var(--pattern-size); opacity: .8; }
[hidden] { display: none !important; }
button, input, textarea { font: inherit; color: inherit; }
:focus-visible { outline: 3px solid var(--accent); outline-offset: 2px; }

.screen { width: min(980px, 100%); margin: 0 auto; padding: 32px 16px 48px; }
.card {
  background: color-mix(in srgb, var(--paper) 94%, transparent);
  border: 1px solid var(--paper-edge);
  border-radius: 20px;
  box-shadow: 0 18px 50px rgba(0,0,0,.18);
}

/* ---------- Cabecera ---------- */
.hero { text-align: center; margin: 8px auto 28px; max-width: 720px; }
.eyebrow { text-transform: uppercase; letter-spacing: .18em; font-size: .78rem; color: var(--accent); font-weight: 700; margin: 0 0 8px; }
.hero h1 {
  font-family: var(--font-title); font-style: var(--title-style); text-transform: var(--title-transform);
  letter-spacing: var(--title-spacing);
  font-size: clamp(2rem, 5vw, 3.2rem); margin: 0 0 12px; line-height: 1.1;
  color: var(--on-bg); text-shadow: var(--on-bg-shadow), 0 0 18px var(--glow);
}
body[data-genre=""] .hero h1 { color: #3b2f25; text-shadow: none; }
.lead { margin: 0; font-size: 1.08rem; line-height: 1.55; color: var(--on-bg); opacity: .92; text-shadow: var(--on-bg-shadow); }
body[data-genre=""] .lead { color: #5a4a3c; text-shadow: none; }

/* ---------- Formulario ---------- */
.form { padding: clamp(18px, 4vw, 34px); display: grid; gap: 22px; }
.field { display: grid; gap: 8px; border: 0; padding: 0; margin: 0; min-width: 0; }
.field > label, .field > legend { font-weight: 700; font-size: 1.02rem; padding: 0; margin-bottom: 8px; }
.optional { font-weight: 400; color: var(--ink-soft); font-size: .88rem; }
input[type="text"], textarea {
  width: 100%; border: 1.5px solid var(--paper-edge); border-radius: 12px;
  background: var(--paper); padding: 12px 14px; font-size: 1rem; line-height: 1.5;
  transition: border-color .2s, box-shadow .2s;
}
textarea { resize: vertical; min-height: 120px; }
input[type="text"]:focus, textarea:focus { outline: none; border-color: var(--accent); box-shadow: 0 0 0 4px var(--glow); }
.field-foot { display: flex; justify-content: space-between; gap: 12px; align-items: center; }
.counter { margin-left: auto; font-size: .82rem; color: var(--ink-soft); }
.field-error, .form-error { color: #b3261e; margin: 0; font-size: .9rem; font-weight: 600; }
.form-error { background: #fdecea; border: 1px solid #f5c2bd; padding: 10px 14px; border-radius: 10px; }

.genre-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(132px, 1fr)); gap: 10px; }
.genre-card {
  position: relative; display: flex; flex-direction: column; align-items: flex-start; gap: 6px;
  padding: 12px; min-height: 92px; border-radius: 14px; cursor: pointer;
  background: linear-gradient(150deg, var(--bg-1), var(--bg-2));
  border: 2px solid transparent; color: var(--cover-title);
  box-shadow: 0 4px 12px rgba(0,0,0,.12);
  transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
  overflow: hidden; text-align: left;
}
.genre-card::after { content: ""; position: absolute; inset: 0; background: var(--pattern); background-size: var(--pattern-size); opacity: .7; pointer-events: none; }
.genre-card input { position: absolute; opacity: 0; pointer-events: none; }
.genre-card .emoji { font-size: 1.6rem; line-height: 1; filter: drop-shadow(0 2px 3px rgba(0,0,0,.25)); }
.genre-card .name { font-family: var(--font-title); font-style: var(--title-style); font-weight: 700; font-size: .98rem; text-shadow: 0 1px 3px rgba(0,0,0,.45); z-index: 1; }
.genre-card .swatches { display: flex; gap: 4px; margin-top: auto; z-index: 1; }
.genre-card .swatches i { width: 14px; height: 14px; border-radius: 50%; border: 1.5px solid rgba(255,255,255,.7); }
.genre-card:hover { transform: translateY(-2px); box-shadow: 0 10px 20px rgba(0,0,0,.18); }
.genre-card:has(input:checked) { border-color: var(--accent); box-shadow: 0 0 0 4px var(--glow), 0 10px 22px rgba(0,0,0,.2); transform: translateY(-2px); }
.genre-card:has(input:checked)::before { content: "✓"; position: absolute; top: 8px; right: 10px; z-index: 2; background: var(--accent); color: var(--accent-ink); width: 22px; height: 22px; border-radius: 50%; display: grid; place-items: center; font-size: .8rem; font-weight: 800; }
.genre-card:has(input:focus-visible) { outline: 3px solid var(--accent); outline-offset: 2px; }

/* ---------- Botones ---------- */
.btn {
  display: inline-flex; align-items: center; justify-content: center; gap: 8px;
  border: 1.5px solid var(--paper-edge); background: var(--paper); color: var(--ink);
  padding: 10px 18px; border-radius: 999px; cursor: pointer; font-weight: 700;
  transition: transform .15s, box-shadow .15s, background .2s, opacity .2s;
}
.btn:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 6px 16px rgba(0,0,0,.15); }
.btn:disabled { opacity: .55; cursor: not-allowed; }
.btn-primary { background: linear-gradient(135deg, var(--accent), color-mix(in srgb, var(--accent) 70%, var(--accent-2))); color: var(--accent-ink); border-color: transparent; }
.btn-lg { padding: 14px 26px; font-size: 1.08rem; }
.btn-ghost { background: color-mix(in srgb, var(--paper) 70%, transparent); }
.btn-sm { padding: 6px 12px; font-size: .86rem; }
.row { display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; }

/* ---------- Libros recientes ---------- */
.recent { margin-top: 26px; }
.recent h2 { font-family: var(--font-title); color: var(--on-bg); text-shadow: var(--on-bg-shadow); font-size: 1.2rem; }
body[data-genre=""] .recent h2 { color: var(--ink); text-shadow: none; }
.recent-list { list-style: none; padding: 0; margin: 0; display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 10px; }
.recent-list a {
  display: block; padding: 12px 14px; border-radius: 12px; text-decoration: none; color: var(--cover-title);
  background: linear-gradient(150deg, var(--bg-1), var(--bg-2)); border: 1px solid rgba(255,255,255,.25);
  box-shadow: 0 4px 12px rgba(0,0,0,.12);
}
.recent-list strong { display: block; font-family: var(--font-title); text-shadow: 0 1px 3px rgba(0,0,0,.4); }
.recent-list small { opacity: .9; text-shadow: 0 1px 2px rgba(0,0,0,.4); }

.models { margin-top: 28px; text-align: center; font-size: .82rem; color: var(--on-bg); opacity: .85; text-shadow: var(--on-bg-shadow); }
body[data-genre=""] .models { color: var(--ink-soft); text-shadow: none; }

@keyframes spin { to { transform: rotate(360deg); } }

@media (max-width: 600px) {
  .screen { padding: 20px 16px 36px; }
  .genre-grid { grid-template-columns: repeat(2, 1fr); }
  .genre-card { min-height: 84px; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation-duration: .01ms !important; transition-duration: .01ms !important; }
}

/* ========================= Libro interactivo ========================= */
.screen-book { width: 100%; max-width: 1240px; padding-top: 14px; }

.book-topbar { display: flex; align-items: center; gap: 14px; margin-bottom: 12px; }
.book-topbar-title {
  margin: 0; font-family: var(--font-title); font-style: var(--title-style);
  text-transform: var(--title-transform); letter-spacing: var(--title-spacing);
  color: var(--on-bg); text-shadow: var(--on-bg-shadow);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: 1.05rem;
}
.banner {
  margin: 0 auto 12px; max-width: 900px; padding: 10px 16px; border-radius: 12px;
  background: #fff4e5; border: 1px solid #f3c98b; color: #6b3d00; font-weight: 600; text-align: center;
}

.book-wrap { display: flex; justify-content: center; }
.book {
  --book-w: min(1120px, calc(100vw - 32px), calc((100dvh - 260px) * 1.5));
  position: relative; width: var(--book-w); aspect-ratio: 3 / 2; min-height: 340px;
  border-radius: 10px;
  filter: drop-shadow(0 26px 30px rgba(0,0,0,.35));
}
/* grosor de las hojas */
.book::before, .book::after {
  content: ""; position: absolute; left: 1.5%; right: 1.5%; height: 10px; bottom: -6px; z-index: -1;
  border-radius: 0 0 8px 8px; background: repeating-linear-gradient(0deg, var(--paper-edge) 0 1px, var(--paper-2) 1px 3px);
}
.book::after { left: 3%; right: 3%; bottom: -11px; opacity: .7; }

.book-stage { position: absolute; inset: 0; perspective: 2400px; }
.view-full { position: absolute; top: 0; bottom: 0; left: 0; width: 100%; }
.layer { position: absolute; inset: 0; }

/* ----------- doble página ----------- */
.spread { position: absolute; inset: 0; display: grid; grid-template-columns: 1fr 1fr; border-radius: 10px; overflow: hidden; }
.spread::after {
  content: ""; position: absolute; top: 0; bottom: 0; left: calc(50% - 30px); width: 60px; pointer-events: none; z-index: 3;
  background: linear-gradient(90deg, transparent, rgba(0,0,0,.18) 45%, rgba(0,0,0,.28) 50%, rgba(0,0,0,.18) 55%, transparent);
}
.page { position: relative; height: 100%; overflow: hidden; background: linear-gradient(180deg, var(--paper), var(--paper-2)); color: var(--ink); }
.page::before { content: ""; position: absolute; inset: 0; pointer-events: none; background: var(--pattern); background-size: var(--pattern-size); opacity: .35; }
.page-left { box-shadow: inset -18px 0 30px -18px rgba(0,0,0,.25); }
.page-right { box-shadow: inset 18px 0 30px -18px rgba(0,0,0,.25); }

.half { position: absolute; top: 0; bottom: 0; width: 50%; overflow: hidden; }
.half-left { left: 0; }
.half-right { left: 50%; }
.half .view-full { width: 200%; }
.half-right .view-full { left: -100%; }

/* ----------- hoja que se voltea ----------- */
.leaf { position: absolute; top: 0; bottom: 0; left: 50%; width: 50%; transform-origin: left center; transform-style: preserve-3d; z-index: 5; will-change: transform; }
.face { position: absolute; inset: 0; overflow: hidden; backface-visibility: hidden; -webkit-backface-visibility: hidden; }
.face .view-full { width: 200%; }
.face.front .view-full { left: -100%; }
.face.back { transform: rotateY(180deg); }
.face.back .view-full { left: 0; }
.shade { position: absolute; inset: 0; pointer-events: none; opacity: 0; z-index: 4; }
.face.front .shade { background: linear-gradient(90deg, rgba(0,0,0,.45), rgba(0,0,0,.05) 70%); }
.face.back .shade { background: linear-gradient(270deg, rgba(0,0,0,.45), rgba(0,0,0,.05) 70%); }

/* ----------- portada ----------- */
.endpaper {
  display: grid; place-items: center; text-align: center; padding: 10%;
  background: linear-gradient(160deg, var(--bg-1), var(--bg-2)); color: var(--cover-title);
}
.endpaper .emblem { font-size: clamp(2rem, 6vw, 4rem); text-shadow: 0 0 22px var(--glow); }
.endpaper p { margin: 6px 0; text-shadow: 0 1px 3px rgba(0,0,0,.4); }
.endpaper .genre-name { font-family: var(--font-title); font-style: var(--title-style); text-transform: var(--title-transform); letter-spacing: var(--title-spacing); font-size: clamp(1rem, 2.2vw, 1.5rem); }
.endpaper .small { font-size: .85rem; opacity: .85; }
.start-reading { margin-top: 14px; }

.cover { background: #222; color: var(--cover-title); }
.cover img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; }
.cover::after {
  content: ""; position: absolute; inset: 0; pointer-events: none;
  background: linear-gradient(180deg, var(--cover-shade) 0%, transparent 38%, transparent 58%, var(--cover-shade) 100%);
}
.cover-frame { position: absolute; inset: 4%; border: 2px solid color-mix(in srgb, var(--cover-title) 65%, transparent); border-radius: 6px; z-index: 2; pointer-events: none; box-shadow: inset 0 0 0 5px color-mix(in srgb, var(--accent) 35%, transparent); }
.cover-top, .cover-bottom { position: absolute; left: 8%; right: 8%; z-index: 3; text-align: center; }
.cover-top { top: 8%; }
.cover-bottom { bottom: 8%; }
.cover-title {
  margin: 0; font-family: var(--font-title); font-style: var(--title-style); text-transform: var(--title-transform);
  letter-spacing: var(--title-spacing); font-weight: 700; line-height: 1.08;
  font-size: clamp(1.6rem, 3.6vw, 3rem); color: var(--cover-title);
  text-shadow: 0 2px 4px rgba(0,0,0,.6), 0 0 26px var(--glow);
  overflow-wrap: anywhere;
}
.cover-ornament { display: block; font-size: 1.2rem; margin: 8px 0; color: var(--accent-2); text-shadow: 0 0 10px var(--glow); }
.cover-genre { display: inline-block; padding: 4px 14px; border-radius: 999px; background: var(--accent); color: var(--accent-ink); font-weight: 700; font-size: .82rem; letter-spacing: .08em; text-transform: uppercase; box-shadow: 0 2px 10px rgba(0,0,0,.3); }
.cover-sub { margin: 8px 0 0; font-size: .82rem; opacity: .9; text-shadow: 0 1px 3px rgba(0,0,0,.6); }
.cover.no-image { background: linear-gradient(160deg, var(--accent), var(--bg-2)); }

/* ----------- página de ilustración ----------- */
.page-illustration { display: flex; flex-direction: column; padding: 7% 7% 6%; }
.illustration { position: relative; flex: 1; min-height: 0; border-radius: 6px; overflow: hidden; background: var(--paper-2); box-shadow: 0 8px 20px rgba(0,0,0,.22), 0 0 0 6px var(--paper), 0 0 0 7px var(--paper-edge); }
.illustration img { width: 100%; height: 100%; object-fit: cover; display: block; animation: fadein .6s ease; }
@keyframes fadein { from { opacity: 0; } to { opacity: 1; } }
.scene-badge { position: absolute; top: 10px; left: 10px; z-index: 2; background: var(--accent); color: var(--accent-ink); font-weight: 800; border-radius: 999px; padding: 3px 11px; font-size: .8rem; box-shadow: 0 2px 6px rgba(0,0,0,.3); }
.asset-error, .asset-loading { position: absolute; inset: 0; display: grid; place-content: center; gap: 10px; text-align: center; padding: 18px; color: var(--ink); }
.asset-error { background: repeating-linear-gradient(45deg, var(--paper-2) 0 12px, var(--paper) 12px 24px); }
.asset-error p { margin: 0; font-weight: 600; font-size: .92rem; }
.asset-error .err-detail { font-weight: 400; font-size: .8rem; color: var(--ink-soft); }
.asset-loading .spinner { width: 34px; height: 34px; margin: 0 auto; border-radius: 50%; border: 3px solid var(--paper-edge); border-top-color: var(--accent); animation: spin 1s linear infinite; }
.page-number { text-align: center; font-size: .8rem; color: var(--ink-soft); margin-top: 10px; letter-spacing: .1em; }

/* ----------- página de texto ----------- */
.page-text { display: flex; flex-direction: column; height: 100%; padding: 7% 9% 5%; }
.page-text .scroll { flex: 1; overflow-y: auto; min-height: 0; padding-right: 4px; scrollbar-width: thin; }
.scene-kicker { margin: 0; text-transform: uppercase; letter-spacing: .16em; font-size: .74rem; font-weight: 800; color: var(--accent); }
.scene-title {
  margin: 6px 0 4px; font-family: var(--font-title); font-style: var(--title-style); text-transform: var(--title-transform);
  letter-spacing: var(--title-spacing); font-size: clamp(1.2rem, 2.1vw, 1.75rem); line-height: 1.15; color: var(--ink);
}
.divider { text-align: center; color: var(--accent); margin: 6px 0 12px; letter-spacing: .6em; }
.divider::before { content: var(--ornament); }
.scene-text p { margin: 0 0 .8em; line-height: 1.68; font-size: clamp(.95rem, 1.25vw, 1.08rem); text-align: justify; hyphens: auto; }
.scene-text p:first-child::first-letter {
  float: left; font-family: var(--font-title); font-size: 3.1em; line-height: .82; padding: 4px 8px 0 0; color: var(--accent); font-weight: 700;
}
.the-end { text-align: center; font-family: var(--font-title); font-style: var(--title-style); color: var(--accent); font-size: 1.3rem; margin: 6px 0 0; }

/* ----------- vista de una sola página (móvil) ----------- */
.single { position: absolute; inset: 0; border-radius: 10px; overflow: hidden; }
.single .page { position: absolute; inset: 0; }
.single .page-mobile { display: flex; flex-direction: column; padding: 0; }
.page-mobile .illustration { flex: none; height: 42%; aspect-ratio: 1 / 1; width: auto; margin: 14px auto 6px; border-radius: 6px; max-width: calc(100% - 28px); }
.book-topbar .btn { white-space: nowrap; }
.page-mobile .page-text { flex: 1; min-height: 0; padding: 12px 18px 10px; }
.leaf.single-leaf { left: 0; width: 100%; }
.leaf.single-leaf .face .view-full { left: 0; width: 100%; }
.leaf.single-leaf .face.back { background: linear-gradient(90deg, var(--paper-2), var(--paper)); }

/* ----------- barra de lectura ----------- */
.reader-bar { display: flex; align-items: center; justify-content: center; gap: 18px; margin: 22px auto 10px; }
.nav-btn {
  width: 54px; height: 54px; border-radius: 50%; border: 0; cursor: pointer; font-size: 2rem; line-height: 1;
  background: var(--accent); color: var(--accent-ink); box-shadow: 0 6px 16px rgba(0,0,0,.25), 0 0 0 4px color-mix(in srgb, var(--paper) 50%, transparent);
  transition: transform .15s, opacity .2s; display: grid; place-items: center; padding-bottom: 4px;
}
.nav-btn:hover:not(:disabled) { transform: scale(1.07); }
.nav-btn:disabled { opacity: .35; cursor: not-allowed; }
.page-indicator { min-width: 150px; text-align: center; color: var(--on-bg); text-shadow: var(--on-bg-shadow); font-weight: 700; }
.dots { display: flex; gap: 7px; justify-content: center; margin-top: 7px; }
.dots button { width: 11px; height: 11px; padding: 0; border-radius: 50%; border: 2px solid var(--on-bg); background: transparent; cursor: pointer; opacity: .75; }
.dots button.cover-dot { border-radius: 3px; }
.dots button[aria-current="true"] { background: var(--accent-2); border-color: var(--accent-2); opacity: 1; transform: scale(1.2); }

.audio-bar {
  max-width: 900px; margin: 0 auto; padding: 12px 16px; display: flex; flex-wrap: wrap; gap: 12px 22px; align-items: center; justify-content: space-between;
  background: color-mix(in srgb, var(--paper) 92%, transparent); border: 1px solid var(--paper-edge); border-radius: 16px; box-shadow: 0 10px 26px rgba(0,0,0,.18);
}
.audio-main { display: flex; align-items: center; gap: 10px; flex: 1 1 340px; min-width: 0; }
.icon-btn { width: 42px; height: 42px; flex: none; border-radius: 50%; border: 1.5px solid var(--paper-edge); background: var(--paper); cursor: pointer; font-size: 1.1rem; color: var(--ink); }
.icon-btn:hover:not(:disabled) { border-color: var(--accent); color: var(--accent); }
.icon-btn:disabled { opacity: .4; cursor: not-allowed; }
.audio-progress { flex: 1; height: 6px; border-radius: 999px; background: var(--paper-2); overflow: hidden; min-width: 50px; }
.audio-progress div { height: 100%; width: 0; background: var(--accent); }
.audio-status { font-size: .86rem; color: var(--ink-soft); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 46%; }
.audio-status .btn { margin-left: 6px; }
.audio-settings { display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
.slider, .toggle { display: flex; align-items: center; gap: 8px; font-size: .86rem; color: var(--ink-soft); cursor: pointer; }
.slider input { width: 110px; accent-color: var(--accent); }
.toggle input { accent-color: var(--accent); width: 16px; height: 16px; }

.unlock { position: fixed; inset: 0; z-index: 50; display: grid; place-items: center; background: rgba(0,0,0,.45); backdrop-filter: blur(3px); padding: 16px; }
.unlock-card { padding: 26px; text-align: center; max-width: 380px; }
.unlock-card p { margin: 0 0 16px; line-height: 1.5; }

/* ----------- responsive ----------- */
@media (max-width: 820px) {
  .book { --book-w: 100%; aspect-ratio: auto; height: max(460px, calc(100dvh - 300px)); max-width: 560px; margin: 0 auto; }
  .book-stage { overflow: hidden; border-radius: 10px; }
  .book-wrap { display: block; }
  .reader-bar { gap: 12px; margin-top: 16px; }
  .nav-btn { width: 48px; height: 48px; }
  .audio-status { max-width: 100%; flex-basis: 100%; order: 3; }
  .audio-main { flex-wrap: wrap; }
  .cover-title { font-size: clamp(1.7rem, 8vw, 2.6rem); }
}
@media (max-width: 420px) {
  .slider input { width: 84px; }
  .audio-settings { gap: 10px; }
}

/* ===================== Construcción progresiva en vivo ===================== */
.status-pill { font-family: var(--font-body); font-style: normal; text-transform: none; letter-spacing: 0; font-size: .82rem; opacity: .9; }
.pill-running .status-pill, .status-pill.pill-running, .status-pill.pill-queued { animation: breathe 2s ease-in-out infinite; }
@keyframes breathe { 50% { opacity: .45; } }

.build-bar {
  position: relative; overflow: hidden;
  max-width: 900px; margin: 0 auto 12px; padding: 7px 12px 9px; border-radius: 12px;
  background: color-mix(in srgb, var(--paper) 90%, transparent); border: 1px solid var(--paper-edge);
  box-shadow: 0 6px 18px rgba(0,0,0,.14); display: grid; grid-template-columns: 1fr auto; align-items: center; gap: 10px;
}
.build-row { display: flex; align-items: center; gap: 10px; font-size: .9rem; min-width: 0; }
.build-dot { width: 10px; height: 10px; border-radius: 50%; background: var(--accent); flex: none; box-shadow: 0 0 0 0 var(--glow); animation: pulse 1.6s infinite; }
.build-bar[data-state="cancelled"] .build-dot, .build-bar[data-state="failed"] .build-dot { animation: none; background: #b3261e; }
@keyframes pulse { 0% { box-shadow: 0 0 0 0 var(--glow); } 70% { box-shadow: 0 0 0 9px transparent; } 100% { box-shadow: 0 0 0 0 transparent; } }
.build-step { flex: 1; min-width: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: var(--ink); font-weight: 600; }
.build-meta { color: var(--ink-soft); font-size: .82rem; white-space: nowrap; }
.build-progress { position: absolute; left: 0; right: 0; bottom: 0; height: 3px; background: var(--paper-2); }
.build-progress div { height: 100%; width: 0; border-radius: inherit; background: linear-gradient(90deg, var(--accent), var(--accent-2)); transition: width .6s ease; }
.build-actions { display: flex; gap: 6px; justify-content: flex-end; flex-wrap: wrap; }
.build-actions .btn-sm { padding: 4px 11px; }
.screen-book:has(.build-bar:not([hidden])) .book { --book-w: min(1120px, calc(100vw - 32px), calc((100dvh - 310px) * 1.5)); }
@media (max-width: 600px) { .build-bar { grid-template-columns: 1fr; gap: 6px; } .build-meta { display: none; } }

/* chips de estado por recurso */
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 2px 0 4px; }
.chip { display: inline-flex; align-items: center; gap: 5px; font-size: .7rem; padding: 2px 8px; border-radius: 999px; border: 1px solid var(--paper-edge); color: var(--ink-soft); background: color-mix(in srgb, var(--paper) 80%, transparent); }
.chip b { font-weight: 700; }
.chip i { width: 7px; height: 7px; border-radius: 50%; background: var(--paper-edge); }
.chip-running { color: var(--ink); border-color: var(--accent); }
.chip-running i { background: var(--accent); animation: breathe 1s infinite; }
.chip-done i { background: #3a9d5d; }
.chip-done { color: var(--ink); }
.chip-error { color: #b3261e; border-color: #e7a59f; }
.chip-error i { background: #b3261e; }

/* espacio reservado de la ilustración */
.asset-placeholder, .asset-building { position: absolute; inset: 0; display: grid; place-content: center; text-align: center; gap: 8px; color: var(--ink-soft); }
.asset-placeholder { border: 2px dashed var(--paper-edge); border-radius: 6px; margin: 10px; }
.asset-placeholder span { font-size: 2.2rem; opacity: .45; }
.asset-placeholder p { margin: 0; font-size: .9rem; }
.asset-building { background: linear-gradient(110deg, var(--paper-2) 30%, color-mix(in srgb, var(--accent-2) 25%, var(--paper)) 50%, var(--paper-2) 70%); background-size: 250% 100%; animation: sweep 2.2s linear infinite; }
@keyframes sweep { from { background-position: 120% 0; } to { background-position: -120% 0; } }
.live-preview { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; filter: blur(6px) saturate(.85); transform: scale(1.06); opacity: .85; transition: opacity .4s; image-rendering: auto; }
.building-overlay { position: relative; z-index: 2; display: grid; gap: 8px; justify-items: center; padding: 14px 18px; margin: auto; border-radius: 14px; max-width: 80%;
  background: color-mix(in srgb, var(--paper) 78%, transparent); backdrop-filter: blur(2px); color: var(--ink); box-shadow: 0 6px 18px rgba(0,0,0,.15); }
.building-overlay p { margin: 0; font-weight: 700; }
.building-overlay small { color: var(--ink-soft); font-size: .78rem; }
.brush { font-size: 1.8rem; animation: paint 1.6s ease-in-out infinite; }
@keyframes paint { 0%, 100% { transform: rotate(-12deg) translateX(-4px); } 50% { transform: rotate(14deg) translateX(4px); } }
.mini-progress { width: 140px; height: 5px; border-radius: 999px; background: var(--paper-2); overflow: hidden; }
.mini-progress div { width: 0; height: 100%; background: var(--accent); transition: width .3s; }
/* la ilustración terminada "se revela" desde la vista previa */
.develop { animation: develop 1.4s ease-out both; }
@keyframes develop { from { filter: blur(14px) saturate(.6); opacity: .35; transform: scale(1.05); } to { filter: none; opacity: 1; transform: none; } }

/* portada en construcción */
.cover-slot { position: absolute; inset: 0; }
.cover-slot > .page { position: absolute; inset: 0; }
.cover-building { display: grid; place-items: center; background: linear-gradient(160deg, color-mix(in srgb, var(--bg-1) 55%, var(--paper)), color-mix(in srgb, var(--bg-2) 55%, var(--paper))); }
.cover-building::before { content: ""; position: absolute; inset: 0; background: linear-gradient(110deg, transparent 30%, rgba(255,255,255,.35) 50%, transparent 70%); background-size: 250% 100%; animation: sweep 2.6s linear infinite; }
.cover-building .building-overlay { max-width: 78%; }
.cb-title { font-family: var(--font-title); font-style: var(--title-style); }
.cover-reveal .cover-title, .cover-reveal .cover-ornament, .cover-reveal .cover-sub, .cover-reveal .cover-genre { animation: rise .9s .5s ease-out both; }
.cover-reveal .cover-ornament { animation-delay: .8s; }
.cover-reveal .cover-sub { animation-delay: 1s; }
@keyframes rise { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }

/* texto en construcción */
.text-placeholder { display: grid; gap: 10px; padding-top: 6px; }
.text-placeholder i { height: 10px; border-radius: 6px; background: var(--paper-2); }
.text-placeholder i.short { width: 60%; }
.text-placeholder p { margin: 6px 0 0; color: var(--ink-soft); font-size: .88rem; text-align: center; }
.typing { color: var(--ink-soft); font-size: .9rem; display: flex; align-items: center; gap: 4px; margin: 4px 0; }
.typing span { width: 6px; height: 6px; border-radius: 50%; background: var(--accent); animation: typing 1.2s infinite; }
.typing span:nth-child(2) { animation-delay: .2s; }
.typing span:nth-child(3) { animation-delay: .4s; margin-right: 6px; }
@keyframes typing { 0%, 80%, 100% { opacity: .25; transform: translateY(0); } 40% { opacity: 1; transform: translateY(-3px); } }
.caret { display: inline-block; width: 2px; height: 1.05em; margin-left: 2px; vertical-align: -2px; background: var(--accent); animation: blink 1s steps(1) infinite; }
@keyframes blink { 50% { opacity: 0; } }
.scene-text.streaming p:first-child::first-letter { color: var(--accent); }
.rewrite-note { font-size: .8rem; color: var(--ink-soft); background: var(--paper-2); padding: 6px 10px; border-radius: 8px; margin: 0 0 8px; }
.asset-error.inline { position: static; padding: 14px; border-radius: 8px; }

.endpaper-status { display: inline-flex; align-items: center; gap: 8px; margin-top: 14px !important; padding: 6px 14px; border-radius: 999px; background: rgba(0,0,0,.18); font-weight: 600; }
.endpaper-status i { width: 9px; height: 9px; border-radius: 50%; background: #fff; }
.endpaper-status.state-running i { animation: breathe 1s infinite; }
.endpaper-status.state-done i { background: #6fdc8c; }
.endpaper-status.state-error i { background: #ff7b72; }

.page-sub { display: block; font-weight: 400; font-size: .78rem; opacity: .9; margin-top: 2px; }
.dots button:disabled { opacity: .3; cursor: not-allowed; }
.dots button.dot-running { border-color: var(--accent-2); animation: breathe 1.2s infinite; }
.dots button.dot-done { background: color-mix(in srgb, var(--on-bg) 55%, transparent); }
.dots button.dot-error { border-color: #ff7b72; background: #ff7b72; }
.mini-spin { display: inline-block; width: 12px; height: 12px; border-radius: 50%; border: 2px solid var(--paper-edge); border-top-color: var(--accent); animation: spin 1s linear infinite; vertical-align: -2px; }

/* ---- avance automático: la navegación manual se habilita al terminar ---- */
.conn-warning { max-width: 900px; margin: 0 auto 10px; padding: 8px 14px; border-radius: 10px; background: #fdecea; border: 1px solid #f5c2bd; color: #6b1a14; font-weight: 600; text-align: center; }

"""

FRONTEND_HTML = r"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Storybook IA</title>
  <meta name="description" content="Crea libros de cuentos ilustrados y narrados con inteligencia artificial generativa local." />
  <link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Ctext y='.9em' font-size='90'%3E📖%3C/text%3E%3C/svg%3E" />
  <style>
/*__CSS__*/
  </style>
</head>
<body data-genre="">
  <div class="backdrop" aria-hidden="true"></div>

  <!-- ============ 1. CREACIÓN ============ -->
  <main id="screen-create" class="screen" aria-labelledby="create-title">
    <header class="hero">
      <p class="eyebrow">IA generativa · ejecución local</p>
      <h1 id="create-title">Crea tu libro de cuentos</h1>
      <p class="lead">Escribe una idea, elige un género y la inteligencia artificial escribirá, ilustrará y narrará un libro de cinco escenas.</p>
    </header>

    <form id="story-form" class="card form" novalidate>
      <div class="field">
        <label for="title">Título del cuento <span class="optional">(opcional: si lo dejas vacío, la IA propone uno)</span></label>
        <input id="title" name="title" type="text" maxlength="90" placeholder="Ej.: Caperucita Roja" autocomplete="off" />
      </div>

      <fieldset class="field">
        <legend>Género literario</legend>
        <div id="genre-grid" class="genre-grid" role="radiogroup" aria-label="Género literario"></div>
        <p class="field-error" id="genre-error" hidden>Selecciona un género.</p>
      </fieldset>

      <div class="field">
        <label for="description">¿De qué trata tu historia?</label>
        <textarea id="description" name="description" rows="5" maxlength="1500"
          placeholder="Ej.: Una niña que utiliza una caperucita roja atraviesa un bosque para visitar a su abuela y se encuentra con un lobo."></textarea>
        <div class="field-foot">
          <p class="field-error" id="description-error" hidden>Describe tu idea con al menos 10 caracteres.</p>
          <span class="counter" id="description-counter">0 / 1500</span>
        </div>
      </div>

      <p class="form-error" id="form-error" role="alert" hidden></p>
      <button class="btn btn-primary btn-lg" id="generate-btn" type="submit">
        <span aria-hidden="true">✨</span> Generar libro
      </button>
    </form>

    <section id="recent" class="recent" hidden>
      <h2>Tus libros</h2>
      <ul id="recent-list" class="recent-list"></ul>
    </section>

    <footer class="models" id="models-info"></footer>
  </main>

  <!-- ============ 2. LIBRO (se construye a la vista) ============ -->
  <main id="screen-book" class="screen screen-book" hidden>
    <div class="book-topbar">
      <button class="btn btn-ghost" id="new-book-btn" type="button">← Crear otro libro</button>
      <p class="book-topbar-title"><span id="topbar-title"></span> <span class="status-pill" id="status-pill" hidden></span></p>
    </div>

    <!-- Indicador discreto de la construcción: nunca sustituye al libro -->
    <section class="build-bar" id="build-bar" aria-live="polite" hidden>
      <div class="build-row">
        <span class="build-dot" aria-hidden="true"></span>
        <span class="build-step" id="build-step">Preparando el libro…</span>
        <span class="build-meta"><span id="build-percent">0 %</span> · <span id="build-elapsed">0:00</span></span>
      </div>
      <div class="build-progress" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0" id="build-progress"><div id="build-fill"></div></div>
      <div class="build-actions">
        <button class="btn btn-sm" id="cancel-btn" type="button">■ Detener</button>
        <button class="btn btn-primary btn-sm" id="resume-btn" type="button" hidden>▶ Reanudar la generación</button>
      </div>
    </section>
    <div class="conn-warning" id="conn-warning" role="alert" hidden></div>
    <div class="banner" id="book-banner" role="status" hidden></div>

    <div class="book-wrap">
      <div class="book" id="book" aria-live="polite">
        <div class="book-stage" id="book-stage"></div>
      </div>
    </div>

    <nav class="reader-bar" aria-label="Navegación del libro">
      <button class="nav-btn" id="prev-btn" type="button" aria-label="Página anterior">‹</button>
      <div class="page-indicator">
        <span id="page-label">Portada</span>
        <small id="page-sub" class="page-sub"></small>
        <div class="dots" id="page-dots"></div>
      </div>
      <button class="nav-btn" id="next-btn" type="button" aria-label="Página siguiente">›</button>
    </nav>

    <section class="audio-bar" id="audio-bar" aria-label="Narración">
      <div class="audio-main">
        <button class="icon-btn" id="play-btn" type="button" aria-label="Pausar">⏸</button>
        <button class="icon-btn" id="replay-btn" type="button" aria-label="Reproducir desde el inicio">↺</button>
        <div class="audio-progress" aria-hidden="true"><div id="audio-progress-fill"></div></div>
        <span class="audio-status" id="audio-status">La portada no tiene narración</span>
      </div>
      <div class="audio-settings">
        <label class="slider"><span>🔊 Voz</span>
          <input type="range" id="volume" min="0" max="1" step="0.05" value="0.9" aria-label="Volumen de la narración" /></label>
        <label class="slider" id="ambient-control" hidden><span>🌫️ Ambiente</span>
          <input type="range" id="ambient-volume" min="0" max="1" step="0.05" value="0.25" aria-label="Volumen del ambiente" /></label>
        <label class="toggle"><input type="checkbox" id="autoplay" checked /> <span>Narración automática</span></label>
        <label class="toggle"><input type="checkbox" id="flip-sound" checked /> <span>Sonido de página</span></label>
      </div>
    </section>

    <div class="unlock" id="audio-unlock" hidden>
      <div class="card unlock-card">
        <p>Tu navegador necesita un toque para permitir la narración automática.</p>
        <button class="btn btn-primary btn-lg" id="unlock-btn" type="button">▶ Activar narración</button>
      </div>
    </div>
  </main>

  <script>
/*__JS__*/
</script>
</body>
</html>
"""

# ============================================================================
# 13. ANIMACIÓN DE PÁGINAS Y NAVEGACIÓN (JavaScript que corre en el navegador)
# ============================================================================
# Secciones: 13.1 API · 13.2 sonido de página · 13.3 narración · 13.4 estados de
# página · 13.5 render del libro · 13.6 animación de páginas · 13.8 consulta del
# estado · 13.9 avance automático · 13.10 navegación manual · 13.11 audio ·
# 13.12 indicadores · 13.13 formulario y rutas.
FRONTEND_JS = r"""
(() => {
"use strict";

// ---------------------------------------------------------------------------
// 13.1 UTILIDADES Y CLIENTE DE LA API (solo habla con este mismo servidor)
// ---------------------------------------------------------------------------
const NUM_SCENES = 5;
const HOLD_MS = 1300;        // tiempo que se ve una página terminada antes de pasarla
const POLL_ACTIVE_MS = 500;  // consulta del estado mientras se genera
const POLL_IDLE_MS = 4000;   // consulta del estado con el libro terminado
const FLIP_MS = 820;
const BUILDING = new Set(["queued", "running"]);
const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const isBuilding = (b) => !!b && BUILDING.has(b.status);
const fmtTime = (ms) => { const s = Math.max(0, Math.floor(ms / 1000)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; };

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(path, { headers: { "Content-Type": "application/json" }, cache: "no-store", ...options });
  } catch {
    throw new Error("No se pudo conectar con el servidor local. ¿Está en ejecución?");
  }
  let data = null;
  try { data = await response.json(); } catch { /* sin cuerpo */ }
  if (!response.ok) {
    const d = data && data.detail;
    throw new Error(typeof d === "string" ? d : Array.isArray(d) ? d.map((x) => (x.msg || "").replace(/^Value error, /, "")).join(" ") : `Error ${response.status}`);
  }
  return data;
}
const api = {
  health: () => request("/api/health"),
  genres: () => request("/api/genres"),
  listBooks: () => request("/api/books"),
  getBook: (id) => request(`/api/books/${id}`),
  createBook: (p) => request("/api/books", { method: "POST", body: JSON.stringify(p) }),
  cancel: (id) => request(`/api/books/${id}/cancel`, { method: "POST" }),
  resume: (id) => request(`/api/books/${id}/resume`, { method: "POST" }),
  retryFailed: (id) => request(`/api/books/${id}/retry-failed`, { method: "POST" }),
  retry: (id, kind, index) => request(`/api/books/${id}/retry`, { method: "POST", body: JSON.stringify({ kind, index }) }),
};

// ---------------------------------------------------------------------------
// 13.2 EFECTO SONORO DE PASAR PÁGINA (Web Audio API, sin archivos externos)
// ---------------------------------------------------------------------------
const pageFlipSound = (() => {
  let ctx = null, noise = null, enabled = true;
  const context = () => {
    if (!ctx) { const AC = window.AudioContext || window.webkitAudioContext; if (!AC) return null; ctx = new AC(); }
    return ctx;
  };
  const getNoise = (c) => {
    if (noise) return noise;
    const len = Math.floor(c.sampleRate * 0.6);
    noise = c.createBuffer(1, len, c.sampleRate);
    const d = noise.getChannelData(0);
    let last = 0;
    for (let i = 0; i < len; i++) { const w = Math.random() * 2 - 1; last = (last + 0.04 * w) / 1.04; d[i] = last * 3.2 + w * 0.25; }
    return noise;
  };
  return {
    unlock() { const c = context(); if (c && c.state === "suspended") c.resume(); },
    setEnabled(v) { enabled = v; },
    play() {
      if (!enabled) return;
      const c = context(); if (!c) return;
      if (c.state === "suspended") c.resume();
      const now = c.currentTime;
      const src = c.createBufferSource(); src.buffer = getNoise(c); src.playbackRate.value = 0.9 + Math.random() * 0.2;
      const band = c.createBiquadFilter(); band.type = "bandpass"; band.Q.value = 0.8;
      band.frequency.setValueAtTime(2600, now); band.frequency.exponentialRampToValueAtTime(900, now + 0.42);
      const high = c.createBiquadFilter(); high.type = "highpass"; high.frequency.value = 350;
      const g = c.createGain();
      g.gain.setValueAtTime(0.0001, now); g.gain.exponentialRampToValueAtTime(0.16, now + 0.06);
      g.gain.exponentialRampToValueAtTime(0.06, now + 0.25); g.gain.exponentialRampToValueAtTime(0.0001, now + 0.5);
      src.connect(band).connect(high).connect(g).connect(c.destination); src.start(now); src.stop(now + 0.55);
      const thump = c.createOscillator(); thump.type = "sine";
      thump.frequency.setValueAtTime(140, now + 0.42); thump.frequency.exponentialRampToValueAtTime(60, now + 0.55);
      const tg = c.createGain();
      tg.gain.setValueAtTime(0.0001, now + 0.42); tg.gain.exponentialRampToValueAtTime(0.05, now + 0.45); tg.gain.exponentialRampToValueAtTime(0.0001, now + 0.6);
      thump.connect(tg).connect(c.destination); thump.start(now + 0.42); thump.stop(now + 0.62);
    },
  };
})();

// ---------------------------------------------------------------------------
// 13.3 NARRACIÓN: un solo <audio> para que nunca suenen dos narraciones a la vez
// ---------------------------------------------------------------------------
const SILENT_WAV = "data:audio/wav;base64,UklGRiYAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YQIAAAAAAA==";
class Narration {
  constructor({ onState, onProgress, onBlocked }) {
    this.voice = new Audio(); this.voice.preload = "auto";
    this.ambient = new Audio(); this.ambient.loop = true; this.ambient.volume = 0.25;
    Object.assign(this, { onState, onProgress, onBlocked });
    this.unlocked = false; this.token = 0; this.currentUrl = null; this.ambientUrl = null;
    const v = this.voice;
    v.addEventListener("playing", () => this.onState("playing"));
    v.addEventListener("pause", () => { if (!v.ended && this.currentUrl) this.onState("paused"); });
    v.addEventListener("ended", () => { this.onState("ended"); this.onProgress(1); });
    v.addEventListener("timeupdate", () => { if (v.duration) this.onProgress(v.currentTime / v.duration); });
    v.addEventListener("error", () => { if (this.currentUrl) this.onState("error"); });
  }
  unlock() {
    if (this.unlocked) return;
    this.unlocked = true;
    const probe = new Audio(SILENT_WAV); probe.volume = 0;
    probe.play().catch(() => { this.unlocked = false; });
  }
  stop() { this.token++; this.currentUrl = null; this.ambientUrl = null; this.voice.pause(); this.ambient.pause(); this.onProgress(0); }
  play(url, ambientUrl = null) {
    this.stop();
    if (!url) { this.onState("none"); return; }
    this.currentUrl = url; this.voice.src = url; this.voice.currentTime = 0;
    this.onState("loading"); this._start(this.token);
    if (ambientUrl) { this.ambientUrl = ambientUrl; this.ambient.src = ambientUrl; this.ambient.play().catch(() => {}); }
  }
  _start(token) {
    const p = this.voice.play(); if (!p) return;
    p.catch((err) => {
      if (token !== this.token) return;
      if (err && err.name === "NotAllowedError") { this.onState("blocked"); this.onBlocked(); }
      else if (err && err.name !== "AbortError") this.onState("error");
    });
  }
  resumeAfterUnlock() { this.unlocked = false; this.unlock(); if (this.currentUrl) this._start(this.token); if (this.ambientUrl) this.ambient.play().catch(() => {}); }
  togglePause() {
    if (!this.currentUrl) return;
    if (this.voice.paused) { if (this.voice.ended) this.voice.currentTime = 0; this._start(this.token); if (this.ambientUrl) this.ambient.play().catch(() => {}); }
    else { this.voice.pause(); this.ambient.pause(); }
  }
  replay() { if (!this.currentUrl) return; this.voice.currentTime = 0; this._start(this.token); }
  setVolume(v) { this.voice.volume = v; }
  setAmbientVolume(v) { this.ambient.volume = v; }
}

// ---------------------------------------------------------------------------
// 13.4 CONTROL DE ESTADOS DE CADA PÁGINA
// ---------------------------------------------------------------------------
const PENDING = { status: "pending" };
const busy = (a) => a && (a.status === "pending" || a.status === "running");

/** Estado de la página n (0 = portada): pending | running | done | error. */
function pageState(book, n) {
  if (n === 0) {
    const plan = book.planning || PENDING;
    if (plan.status === "error") return { key: "error", label: "Error al planificar" };
    if (plan.status !== "done") return { key: plan.status === "running" ? "running" : "pending", label: "Diseñando la historia" };
    const c = book.cover || PENDING;
    if (c.status === "done") return { key: "done", label: "Portada completada" };
    if (c.status === "error") return { key: "error", label: "Error en la portada" };
    return { key: c.status === "running" ? "running" : "pending", label: c.status === "running" ? "Generando imagen" : "Pendiente" };
  }
  const s = book.scenes && book.scenes[n - 1];
  if (!s) return { key: "pending", label: "Pendiente" };
  const assets = [s.writing, s.image, s.audio, s.ambient].filter(Boolean);
  if (s.writing.status === "running") return { key: "running", label: "Generando texto" };
  if (s.image.status === "running") return { key: "running", label: s.audio.status === "running" ? "Generando imagen y audio" : "Generando imagen" };
  if (s.audio.status === "running") return { key: "running", label: "Generando audio" };
  if (s.ambient && s.ambient.status === "running") return { key: "running", label: "Generando ambiente" };
  if (assets.every((a) => a.status === "done")) return { key: "done", label: "Completada" };
  if (assets.some((a) => a.status === "error") && !assets.some(busy)) return { key: "error", label: "Error de generación" };
  return { key: s.writing.status === "pending" ? "pending" : "running", label: s.writing.status === "pending" ? "Pendiente" : "En proceso" };
}
const pageFinished = (book, n) => ["done", "error"].includes(pageState(book, n).key);
/** La página n ya tiene contenido real (o se está creando). */
function pageHasContent(book, n) {
  if (n === 0) return true;
  const s = book.scenes && book.scenes[n - 1];
  return !!s && s.writing.status !== "pending";
}

// ---------------------------------------------------------------------------
// 13.5 RENDER DEL LIBRO
// La vista se divide en regiones (data-region). Con cada estado nuevo solo se
// reemplazan las regiones que cambiaron; el texto que se escribe y la vista
// previa de la imagen se aplican directamente (applyLive), sin reiniciar animaciones.
// ---------------------------------------------------------------------------
const paragraphs = (t) => String(t || "").split(/\n+/).map((p) => p.trim()).filter(Boolean).map((p) => `<p>${esc(p)}</p>`).join("");
const retryButton = (kind, index, label) => `<button class="btn btn-primary btn-sm" type="button" data-retry="${kind}" data-index="${index}">↻ ${label}</button>`;

function chip(label, asset, runningLabel) {
  const s = (asset && asset.status) || "pending";
  const text = { pending: "pendiente", running: runningLabel, done: "completado", error: "error" }[s];
  return `<span class="chip chip-${s}" title="${esc(label)}: ${text}"><i aria-hidden="true"></i>${esc(label)}<b>${text}</b></span>`;
}
const chipsRegion = (s) => `<div class="chips" data-region="chips-${s.index}">${chip("Texto", s.writing, "generando")}${chip("Imagen", s.image, "generando")}${chip("Audio", s.audio, "generando")}${s.ambient ? chip("Ambiente", s.ambient, "generando") : ""}</div>`;

function illustrationRegion(s, fresh) {
  const a = s.image || PENDING, key = `image:${s.index}`;
  let inner;
  if (a.status === "done") inner = `<img class="${fresh ? "develop" : ""}" src="${esc(a.url)}" alt="Ilustración de la escena ${s.index}: ${esc(s.title)}" draggable="false" />`;
  else if (a.status === "error") inner = `<div class="asset-error"><p>No se pudo generar la ilustración.</p><p class="err-detail">${esc(a.error)}</p>${retryButton("image", s.index, "Reintentar ilustración")}</div>`;
  else if (a.status === "running") inner = `<div class="asset-building"><img class="live-preview" data-preview-img="${key}" alt="" hidden />
      <div class="building-overlay"><div class="brush" aria-hidden="true">🖌️</div><p>Ilustrando esta escena…</p>
      <div class="mini-progress"><div data-preview-bar="${key}"></div></div><small data-preview-step="${key}">Preparando el modelo de imagen…</small></div></div>`;
  else inner = `<div class="asset-placeholder"><span aria-hidden="true">🖼️</span><p>La ilustración aparecerá aquí</p></div>`;
  return `<div class="illustration state-${a.status}" data-region="illus-${s.index}"><span class="scene-badge">${s.index} / ${NUM_SCENES}</span>${inner}</div>`;
}

function textRegion(s) {
  const w = s.writing || PENDING;
  let body;
  if (w.status === "done") body = `<div class="scene-text">${paragraphs(s.text)}</div>${s.index === NUM_SCENES ? '<p class="the-end">Fin</p>' : ""}`;
  else if (w.status === "running") body = `<p class="rewrite-note" data-draft-note="${s.index}" hidden></p><div class="scene-text streaming" data-draft="${s.index}"></div>
      <p class="typing" data-draft-typing="${s.index}"><span></span><span></span><span></span> Escribiendo la historia…</p>`;
  else if (w.status === "error") body = `<div class="asset-error inline"><p>No se pudo escribir esta escena.</p><p class="err-detail">${esc(w.error)}</p>${retryButton("text", s.index, "Reintentar texto")}</div>`;
  else body = `<div class="text-placeholder"><i></i><i></i><i></i><i></i><i class="short"></i><p>El texto aparecerá aquí</p></div>`;
  return `<div class="text-body" data-region="text-${s.index}">${body}</div>`;
}

const textPage = (s) => `<div class="page-text"><div class="scroll">
    <p class="scene-kicker">Escena ${s.index} de ${NUM_SCENES}</p>
    <h2 class="scene-title" data-region="title-${s.index}">${esc(s.title || `Escena ${s.index}`)}</h2>
    ${chipsRegion(s)}<div class="divider" aria-hidden="true"></div>${textRegion(s)}
  </div><p class="page-number">— ${s.index * 2} —</p></div>`;

function coverRegion(book, genreLabel, fresh) {
  const plan = book.planning || PENDING, a = book.cover || PENDING;
  const title = book.title || (book.request && book.request.title) || "";
  let content;
  if (a.status === "done" || a.status === "error") {
    const img = a.status === "done" ? `<img class="${fresh ? "develop" : ""}" src="${esc(a.url)}" alt="Portada de ${esc(title)}" draggable="false" />` : "";
    const error = a.status === "error" ? `<p class="cover-sub">No se pudo generar la ilustración de portada.</p>${retryButton("cover", 0, "Reintentar portada")}` : "";
    content = `<div class="page cover ${a.status === "done" ? "" : "no-image"} ${fresh ? "cover-reveal" : ""}">${img}<div class="cover-frame"></div>
      <div class="cover-top"><span class="cover-genre">${esc(genreLabel)}</span></div>
      <div class="cover-bottom"><h1 class="cover-title">${esc(title)}</h1><span class="cover-ornament" aria-hidden="true">✦ ✦ ✦</span>
      <p class="cover-sub">Un libro ilustrado y narrado con IA</p>${error}</div></div>`;
  } else {
    let status;
    if (plan.status === "error") status = `<p class="cb-title">No se pudo planificar la historia</p><p class="err-detail">${esc(plan.error || book.error)}</p>
        <button class="btn btn-primary btn-sm" type="button" data-action="resume">↻ Volver a intentar</button>`;
    else if (plan.status !== "done") status = `<div class="brush" aria-hidden="true">✍️</div><p class="cb-title">Diseñando la historia y los personajes…</p>
        <small>${book.status === "queued" ? "Esperando turno para generar" : "El modelo de lenguaje analiza tu idea"}</small>`;
    else if (a.status === "running") status = `<div class="brush" aria-hidden="true">🖌️</div><p class="cb-title">Ilustrando la portada…</p>
        <div class="mini-progress"><div data-preview-bar="cover:0"></div></div><small data-preview-step="cover:0">Preparando el modelo de imagen…</small>`;
    else status = `<p class="cb-title">La portada aparecerá aquí</p>`;
    content = `<div class="page cover cover-building"><img class="live-preview" data-preview-img="cover:0" alt="" hidden /><div class="cover-frame"></div>
      <div class="cover-top"><span class="cover-genre">${esc(genreLabel)}</span></div><div class="building-overlay">${status}</div></div>`;
  }
  return `<div class="cover-slot" data-region="cover">${content}</div>`;
}

function endpaperRegion(book, genre, auto) {
  const st = pageState(book, 0);
  const action = !auto && pageFinished(book, 0) && book.scenes && book.scenes.length
    ? `<button class="btn btn-primary start-reading" type="button" data-action="next">Comenzar a leer ›</button>`
    : `<p class="endpaper-status state-${st.key}"><i aria-hidden="true"></i>${esc(st.label)}</p>`;
  return `<div class="page page-left endpaper" data-region="endpaper"><div><div class="emblem" aria-hidden="true">${esc(genre.emoji || "📖")}</div>
    <p class="genre-name">${esc(genre.label)}</p><p class="small">${NUM_SCENES} escenas · ilustraciones y narración en español</p>${action}</div></div>`;
}

class BookReader {
  constructor(stage, { onRetry, onAction }) {
    Object.assign(this, { stage, onRetry, onAction });
    this.genres = []; this.book = null; this.auto = false;
    this.live = { drafts: {}, previews: {} };
    this.fresh = new Set(); this.index = 0; this.animating = false;
    this.sources = new WeakMap();
    this.mobileQuery = window.matchMedia("(max-width: 820px)");
    this.reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
    stage.addEventListener("click", (e) => {
      const r = e.target.closest("[data-retry]");
      if (r && !r.disabled) { r.disabled = true; this.onRetry(r.dataset.retry, Number(r.dataset.index)); return; }
      const a = e.target.closest("[data-action]");
      if (a) this.onAction(a.dataset.action);
    });
    this.mobileQuery.addEventListener("change", () => this.render(true));
  }
  get mobile() { return this.mobileQuery.matches; }
  genre() { return this.genres.find((g) => g.key === this.book.genre) || { label: this.book.genre, emoji: "📖" }; }
  scene(n) { return (this.book.scenes && this.book.scenes[n - 1]) || { index: n, title: null, text: null, writing: PENDING, image: PENDING, audio: PENDING }; }

  setBook(book, auto) {
    const prev = this.book;
    if (prev && prev.id === book.id) {
      if (prev.cover && prev.cover.status !== "done" && book.cover && book.cover.status === "done") this.markFresh("cover");
      (book.scenes || []).forEach((s, i) => {
        const p = prev.scenes && prev.scenes[i];
        if (p && p.image.status !== "done" && s.image.status === "done") this.markFresh(`image:${s.index}`);
      });
    }
    this.live = book.live ? { drafts: { ...book.live.drafts }, previews: { ...book.live.previews } } : { drafts: {}, previews: {} };
    this.book = book; this.auto = auto;
    if (!this.animating) this.render();
  }
  markFresh(key) { this.fresh.add(key); setTimeout(() => this.fresh.delete(key), 2500); }

  applyDraft(n) {
    const box = this.stage.querySelector(`[data-draft="${n}"]`);
    if (!box) return;
    const d = this.live.drafts[String(n)], text = (d && d.text) || "";
    const scroller = box.closest(".scroll");
    const atBottom = scroller && scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 40;
    if (box.dataset.text !== text) {
      box.dataset.text = text;
      box.innerHTML = text ? paragraphs(text) : "";
      if (text && box.lastElementChild) box.lastElementChild.insertAdjacentHTML("beforeend", '<span class="caret" aria-hidden="true"></span>');
    }
    const typing = this.stage.querySelector(`[data-draft-typing="${n}"]`);
    if (typing) typing.hidden = !!text;
    const note = this.stage.querySelector(`[data-draft-note="${n}"]`);
    if (note) {
      note.hidden = !(d && d.attempt > 1);
      note.textContent = d && d.attempt > 1 ? `Reescribiendo la escena para asegurar un texto correcto en español (intento ${d.attempt})…` : "";
    }
    if (scroller && atBottom) scroller.scrollTop = scroller.scrollHeight;
  }
  applyPreview(key) {
    const p = this.live.previews[key];
    if (!p) return;
    this.stage.querySelectorAll(`[data-preview-img="${key}"]`).forEach((img) => { if (p.image && img.getAttribute("src") !== p.image) img.src = p.image; img.hidden = !p.image; });
    this.stage.querySelectorAll(`[data-preview-bar="${key}"]`).forEach((bar) => { bar.style.width = `${Math.round((p.step / p.total) * 100)}%`; });
    this.stage.querySelectorAll(`[data-preview-step="${key}"]`).forEach((el) => { el.textContent = `Paso ${p.step} de ${p.total} de difusión`; });
  }
  applyLive() {
    for (let n = 1; n <= NUM_SCENES; n++) this.applyDraft(n);
    Object.keys(this.live.previews).forEach((k) => this.applyPreview(k));
  }

  viewHTML(index) {
    const book = this.book, genre = this.genre();
    if (this.mobile) {
      if (index === 0) return `<div class="single">${coverRegion(book, genre.label, this.fresh.has("cover"))}</div>`;
      const s = this.scene(index);
      return `<div class="single"><div class="page page-mobile">${illustrationRegion(s, this.fresh.has(`image:${index}`))}${textPage(s)}</div></div>`;
    }
    if (index === 0) return `<div class="spread">${endpaperRegion(book, genre, this.auto)}<div class="page page-right">${coverRegion(book, genre.label, this.fresh.has("cover"))}</div></div>`;
    const s = this.scene(index);
    return `<div class="spread"><div class="page page-left page-illustration">${illustrationRegion(s, this.fresh.has(`image:${index}`))}
      <p class="page-number">— ${index * 2 - 1} —</p></div><div class="page page-right">${textPage(s)}</div></div>`;
  }

  /** Dibuja la página actual reemplazando solo las regiones que cambiaron. */
  render(force = false) {
    if (!this.book) return;
    const viewKey = `${this.mobile ? "m" : "d"}:${this.index}`;
    const html = this.viewHTML(this.index);
    const layer = this.stage.querySelector(":scope > .layer");
    if (force || !layer || layer.dataset.view !== viewKey || this.stage.children.length !== 1) {
      this.stage.innerHTML = `<div class="layer" data-view="${viewKey}" data-page="${this.index}">${html}</div>`;
      this.stage.querySelectorAll("[data-region]").forEach((el) => this.sources.set(el, el.outerHTML));
    } else {
      const tpl = document.createElement("template");
      tpl.innerHTML = html;
      tpl.content.querySelectorAll("[data-region]").forEach((fresh) => {
        const current = layer.querySelector(`[data-region="${fresh.dataset.region}"]`);
        const source = fresh.outerHTML;
        if (!current || this.sources.get(current) === source) return;
        const scroller = current.closest(".scroll"), top = scroller && scroller.scrollTop;
        current.replaceWith(fresh);
        this.sources.set(fresh, source);
        if (scroller) scroller.scrollTop = top;
      });
    }
    this.applyLive();
  }

  // -------------------------------------------------------------------------
  // 13.6 ANIMACIÓN DE PÁGINAS (hoja que se voltea sobre el lomo)
  // -------------------------------------------------------------------------
  async flipTo(target) {
    if (this.animating || target === this.index || target < 0 || target > NUM_SCENES) return false;
    const forward = target > this.index, from = this.index;
    this.index = target;
    if (this.reducedMotion.matches) { this.render(true); return true; }
    this.animating = true;
    const current = this.viewHTML(from), next = this.viewHTML(target);
    const full = (h) => `<div class="view-full">${h}</div>`;
    let html;
    if (this.mobile) {
      html = `<div class="layer">${forward ? next : current}</div>
        <div class="leaf single-leaf"><div class="face front">${full(forward ? current : next)}<div class="shade"></div></div><div class="face back"><div class="shade"></div></div></div>`;
    } else {
      // La hoja derecha gira sobre el lomo y en su reverso muestra la página izquierda siguiente.
      html = `<div class="layer">${next}</div><div class="half half-${forward ? "left" : "right"}">${full(current)}</div>
        <div class="leaf"><div class="face front">${full(forward ? current : next)}<div class="shade"></div></div>
        <div class="face back">${full(forward ? next : current)}<div class="shade"></div></div></div>`;
    }
    this.stage.innerHTML = html;
    this.applyLive();
    const leaf = this.stage.querySelector(".leaf");
    const frames = forward ? [{ transform: "rotateY(0deg)" }, { transform: "rotateY(-180deg)" }] : [{ transform: "rotateY(-180deg)" }, { transform: "rotateY(0deg)" }];
    const opts = { duration: FLIP_MS, easing: "cubic-bezier(.45,.05,.3,1)", fill: "forwards" };
    const anims = [leaf.animate(frames, opts)];
    this.stage.querySelectorAll(".shade").forEach((el) => anims.push(el.animate([{ opacity: 0 }, { opacity: 0.9 }, { opacity: 0 }], opts)));
    // Plazo de seguridad: aunque el navegador no complete la animación (pestaña en
    // segundo plano), el avance continúa.
    await Promise.race([Promise.all(anims.map((a) => a.finished)).catch(() => {}), sleep(FLIP_MS + 600)]);
    this.animating = false;
    this.render(true);
    return true;
  }
}

// ---------------------------------------------------------------------------
// 13.7 ESTADO GLOBAL DE LA INTERFAZ
// ---------------------------------------------------------------------------
const screens = { create: $("screen-create"), book: $("screen-book") };
const state = {
  genres: [], genre: "", book: null, rev: -1, bookId: null, pollToken: 0, pollWake: null,
  auto: false,        // avance automático: la vista acompaña la construcción
  completedAt: {},    // página -> instante en que se vio terminada
  advanceTimer: null, armed: null, narrationState: "none", lastRequest: null, tick: null,
};

const narration = new Narration({
  onState: (s) => { state.narrationState = s; renderAudioBar(); },
  onProgress: (p) => { $("audio-progress-fill").style.width = `${Math.round(p * 100)}%`; },
  onBlocked: () => { $("audio-unlock").hidden = false; },
});
const reader = new BookReader($("book-stage"), {
  onRetry: (kind, index) => retryAsset(kind, index),
  onAction: (a) => { if (a === "next") goTo(reader.index + 1); if (a === "resume") resumeBook(); },
});
const unlockAudio = () => { narration.unlock(); pageFlipSound.unlock(); };
function show(name) { for (const [k, el] of Object.entries(screens)) el.hidden = k !== name; window.scrollTo({ top: 0 }); }
const setTheme = (g) => { document.body.dataset.genre = g || ""; };

// ---------------------------------------------------------------------------
// 13.8 CONSULTA DEL ESTADO (polling): un único canal, funciona detrás de
// cualquier proxy y no deja conexiones abiertas. Cada respuesta incluye el
// estado de cada recurso, el texto que se está escribiendo y la vista previa.
// ---------------------------------------------------------------------------
async function pollLoop(id) {
  const token = ++state.pollToken;
  while (token === state.pollToken) {
    try {
      const book = await api.getBook(id);
      if (token !== state.pollToken) return;
      $("conn-warning").hidden = true;
      onBook(book);
    } catch (err) {
      if (token !== state.pollToken) return;
      $("conn-warning").textContent = `${err.message} Reintentando…`;
      $("conn-warning").hidden = false;
    }
    const active = state.book && (isBuilding(state.book) || state.book.status === "retrying");
    await new Promise((resolve) => { state.pollWake = resolve; setTimeout(resolve, active ? POLL_ACTIVE_MS : POLL_IDLE_MS); });
  }
}
const pollNow = () => { if (state.pollWake) state.pollWake(); };
const stopPolling = () => { state.pollToken++; pollNow(); };

function onBook(book) {
  const prev = state.book;
  if (prev && prev.id && prev.id !== book.id) return;              // otro libro
  if (prev && prev.id === book.id && book.rev < state.rev) return; // respuesta vieja
  state.book = book;
  state.rev = book.rev;
  for (let n = 0; n <= NUM_SCENES; n++) {
    if (pageFinished(book, n)) { if (!state.completedAt[n]) state.completedAt[n] = Date.now(); }
    else delete state.completedAt[n];
  }
  if (isBuilding(book) && !state.auto && prev && prev.id === book.id && !isBuilding(prev)) state.auto = true; // reanudado
  reader.setBook(book, state.auto);
  renderChrome(prev);
  narrationOnUpdate(prev, book);
  autoAdvance();
}

// ---------------------------------------------------------------------------
// 13.9 AVANCE AUTOMÁTICO DURANTE LA GENERACIÓN
// La vista pasa de la página i a la i+1 SOLO cuando:
//   1. el servidor reporta la página i terminada (todos sus recursos done/error),
//   2. se ha mostrado terminada durante HOLD_MS, y
//   3. la página i+1 ya tiene contenido real (el servidor empezó a crearla).
// Avanza de una en una (portada → 1 → 2 → 3 → 4 → 5): nunca salta páginas.
// ---------------------------------------------------------------------------
function autoAdvance() {
  clearTimeout(state.advanceTimer);
  const book = state.book;
  if (!book || !state.auto || reader.animating) return;
  const i = reader.index;
  if (i >= NUM_SCENES || !pageFinished(book, i) || !pageHasContent(book, i + 1)) {
    if (!isBuilding(book) && (i >= NUM_SCENES || !pageHasContent(book, i + 1))) endAutoMode();
    return;
  }
  const wait = HOLD_MS - (Date.now() - (state.completedAt[i] || Date.now()));
  if (wait > 0) { state.advanceTimer = setTimeout(autoAdvance, wait); return; }
  flip(i + 1, false);
}

function endAutoMode() {
  if (!state.auto) return;
  state.auto = false;
  reader.setBook(state.book, false);
  renderChrome(state.book);
}

function flip(target, manual) {
  narration.stop();
  pageFlipSound.play();
  const done = reader.flipTo(target);
  renderNav();
  if (manual) playCurrent();
  renderAudioBar();
  done.then(() => { renderNav(); renderAudioBar(); autoAdvance(); });
}

// ---------------------------------------------------------------------------
// 13.10 NAVEGACIÓN MANUAL (habilitada cuando no hay generación automática)
// ---------------------------------------------------------------------------
function reachable(n) {
  const book = state.book;
  if (!book || n < 0 || n > NUM_SCENES) return false;
  return n === 0 || pageHasContent(book, n);
}
function goTo(target) {
  if (!state.book || state.auto || reader.animating || target === reader.index || !reachable(target)) return;
  unlockAudio();
  state.armed = target;
  flip(target, true);
}

// ---------------------------------------------------------------------------
// 13.11 AUDIO DE CADA ESCENA
// ---------------------------------------------------------------------------
const currentScene = () => (reader.index > 0 && state.book && state.book.scenes ? state.book.scenes[reader.index - 1] || null : null);
function playCurrent(force = false) {
  const scene = currentScene();
  if (!scene || (!force && (state.armed !== reader.index || !$("autoplay").checked))) {
    narration.stop();
    state.narrationState = scene ? (scene.audio.status === "done" ? "ready" : "waiting") : "none";
    renderAudioBar();
    return;
  }
  const amb = scene.ambient && scene.ambient.status === "done" ? scene.ambient.url : null;
  if (scene.audio.status === "done") narration.play(scene.audio.url, amb);
  else { narration.stop(); state.narrationState = scene.audio.status === "error" ? "error" : "waiting"; renderAudioBar(); }
}
function narrationOnUpdate(prev, book) {
  if (!prev || prev.id !== book.id || reader.index === 0 || state.auto) return;
  const i = reader.index - 1;
  const before = prev.scenes && prev.scenes[i] && prev.scenes[i].audio, after = book.scenes && book.scenes[i] && book.scenes[i].audio;
  // El audio estaba pendiente (p. ej. reintento) y quedó listo en la página activa.
  if (after && after.status === "done" && (!before || before.url !== after.url) && state.armed === reader.index && $("autoplay").checked) playCurrent();
}

// ---------------------------------------------------------------------------
// 13.12 BARRAS, INDICADORES Y CONTROLES
// ---------------------------------------------------------------------------
function renderChrome(prev) {
  const book = state.book;
  if (!book) return;
  $("topbar-title").textContent = book.title || (book.request && book.request.title) || "Tu libro";
  document.title = `${book.title || "Tu libro"} · Storybook IA`;
  const pills = { queued: "En cola", running: "Generándose", cancelled: "Detenido", failed: "Error" };
  const pill = $("status-pill");
  pill.hidden = !pills[book.status];
  pill.textContent = pills[book.status] ? `(${pills[book.status]})` : "";
  pill.className = `status-pill pill-${book.status}`;
  $("ambient-control").hidden = !book.ambient_enabled;
  renderBanner(prev); renderBuildBar(); renderNav(); renderAudioBar();
}

function renderBuildBar() {
  const book = state.book;
  if (!book) return;
  const building = isBuilding(book), stopped = book.status === "cancelled" || book.status === "failed";
  const bar = $("build-bar");
  bar.hidden = !building && !stopped && !state.auto;
  if (bar.hidden) return;
  bar.dataset.state = book.status;
  const pct = Math.round((book.progress.done / book.progress.total) * 100);
  let step = stopped ? (book.error || "Generación detenida") : book.progress.step;
  if (!building && !stopped) step = "Mostrando las últimas páginas creadas…";
  const running = book.cover && book.cover.status === "running" ? "cover:0"
    : ((book.scenes || []).find((s) => s.image.status === "running") || {}).index;
  const p = running && reader.live.previews[typeof running === "string" ? running : `image:${running}`];
  if (building && p) step += ` · paso ${p.step}/${p.total}`;
  $("build-step").textContent = step;
  $("build-percent").textContent = `${pct} %`;
  $("build-fill").style.width = `${pct}%`;
  $("build-progress").setAttribute("aria-valuenow", pct);
  $("build-elapsed").textContent = fmtTime(Date.now() - book.created_at * 1000);
  $("cancel-btn").hidden = !building || !book.id;
  $("resume-btn").hidden = !stopped || !book.plan && book.status !== "cancelled";
}

function renderBanner(prev) {
  const book = state.book, banner = $("book-banner");
  const errors = [book.cover, ...(book.scenes || []).flatMap((s) => [s.writing, s.image, s.audio, s.ambient])].filter((a) => a && a.status === "error").length;
  const finished = ["completed", "completed_with_errors"].includes(book.status);
  if (book.status === "retrying") banner.innerHTML = "Regenerando el recurso solicitado… el resto del libro sigue disponible.";
  else if (finished && errors) {
    banner.innerHTML = `Algunos recursos no se pudieron generar (${errors}). Puedes reintentarlos uno a uno desde su página o todos a la vez, sin perder el resto del libro.
      <button class="btn btn-primary btn-sm" type="button" id="retry-all">↻ Reintentar todo lo que falló</button>`;
    $("retry-all").addEventListener("click", retryAllFailed);
  }
  else if (book.status === "completed" && !state.auto && (banner.dataset.kind === "done" || (prev && isBuilding(prev)) || state.justFinished)) {
    state.justFinished = false;
    banner.dataset.kind = "done";
    banner.innerHTML = `✓ ¡Tu libro está terminado! Ya puedes recorrerlo libremente. <button class="btn btn-sm" type="button" id="read-start">Leer desde la portada</button>`;
    $("read-start").addEventListener("click", () => { banner.hidden = true; banner.dataset.kind = ""; goTo(0); });
  } else if (isBuilding(book) && errors) banner.innerHTML = `Hay ${errors} recurso(s) con error. La creación continúa; podrás reintentarlos en su página.`;
  else { if (finished && state.auto) state.justFinished = true; banner.hidden = true; return; }
  banner.hidden = false;
}

function renderNav() {
  const book = state.book;
  if (!book) return;
  const i = reader.index, auto = state.auto;
  $("prev-btn").disabled = auto || i === 0;
  $("next-btn").disabled = auto || !reachable(i + 1);
  $("next-btn").title = auto ? "El libro avanza solo mientras se crea" : reachable(i + 1) ? "" : "Esta página todavía no se ha creado";
  $("page-label").textContent = i === 0 ? "Portada" : `Escena ${i} de ${NUM_SCENES}`;
  const st = pageState(book, i);
  const parts = [`Página ${i + 1} de ${NUM_SCENES + 1}`];
  if (auto) parts.push(st.key === "done" ? "Completada" : st.key === "error" ? st.label : "Creación en progreso");
  else if (st.key !== "done") parts.push(st.label);
  $("page-sub").textContent = parts.join(" · ");
  $("page-dots").innerHTML = Array.from({ length: NUM_SCENES + 1 }, (_, k) => {
    const s = pageState(book, k), label = `${k === 0 ? "Portada" : "Escena " + k}: ${s.label}`;
    return `<button type="button" class="dot-${s.key} ${k === 0 ? "cover-dot" : ""}" data-page="${k}" title="${label}" aria-label="${label}" ${k === i ? 'aria-current="true"' : ""} ${!auto && reachable(k) ? "" : "disabled"}></button>`;
  }).join("");
}

function renderAudioBar() {
  const scene = currentScene(), s = state.narrationState;
  const available = !!scene && scene.audio.status === "done";
  $("play-btn").disabled = !available || s === "blocked";
  $("replay-btn").disabled = !available;
  const playing = s === "playing" || s === "loading";
  $("play-btn").textContent = playing ? "⏸" : "▶";
  $("play-btn").setAttribute("aria-label", playing ? "Pausar" : "Reproducir");
  const status = $("audio-status");
  if (!scene) { status.textContent = reader.index === 0 ? "La portada no tiene narración" : "La narración se preparará al terminar el texto"; return; }
  if (scene.audio.status === "error" || s === "error") {
    status.innerHTML = `No se pudo generar la narración. <button class="btn btn-sm" id="retry-audio" type="button">↻ Reintentar</button>`;
    $("retry-audio").addEventListener("click", (e) => { e.target.disabled = true; retryAsset("audio", scene.index); });
    return;
  }
  if (scene.audio.status !== "done") {
    status.innerHTML = `<span class="mini-spin" aria-hidden="true"></span> ${scene.writing.status === "done" ? "Generando audio…" : "La narración se preparará al terminar el texto"}`;
    return;
  }
  status.textContent = { playing: "Narrando…", loading: "Cargando narración…", paused: "En pausa", ended: "Narración terminada", blocked: "Toca «Activar narración»" }[s]
    || (state.auto ? "Audio listo · pulsa ▶ para escucharlo" : "Narración lista");
}

async function retryAsset(kind, index) {
  try { await api.retry(state.book.id, kind, index); pollNow(); }
  catch (err) { $("book-banner").textContent = err.message; $("book-banner").hidden = false; reader.render(true); }
}
async function retryAllFailed() {
  const btn = $("retry-all");
  if (btn) btn.disabled = true;
  try { await api.retryFailed(state.book.id); pollNow(); }
  catch (err) { $("book-banner").textContent = err.message; $("book-banner").hidden = false; }
}
async function cancelBook() {
  if (!state.book || !state.book.id) return;
  $("cancel-btn").disabled = true;
  try { await api.cancel(state.book.id); pollNow(); } finally { $("cancel-btn").disabled = false; }
}
async function resumeBook() {
  if (!state.book || !state.book.id) return;
  try { await api.resume(state.book.id); state.auto = true; state.armed = null; narration.stop(); pollNow(); }
  catch (err) { $("book-banner").textContent = err.message; $("book-banner").hidden = false; }
}

// ---------------------------------------------------------------------------
// 13.13 FORMULARIO DE CREACIÓN, LIBROS RECIENTES Y RUTAS
// ---------------------------------------------------------------------------
function renderGenres() {
  $("genre-grid").innerHTML = state.genres.map((g) => `<label class="genre-card" data-genre="${g.key}">
      <input type="radio" name="genre" value="${g.key}" /><span class="emoji" aria-hidden="true">${g.emoji}</span><span class="name">${g.label}</span>
      <span class="swatches" aria-hidden="true"><i style="background:var(--bg-1)"></i><i style="background:var(--accent)"></i><i style="background:var(--accent-2)"></i><i style="background:var(--paper)"></i></span></label>`).join("");
}
$("genre-grid").addEventListener("change", (e) => { state.genre = e.target.value; setTheme(state.genre); $("genre-error").hidden = true; });

function validateForm() {
  const ok = $("description").value.trim().length >= 10;
  $("genre-error").hidden = !!state.genre;
  $("description-error").hidden = ok;
  return !!state.genre && ok;
}

/** Libro vacío que se dibuja AL INSTANTE, antes de que responda el servidor. */
const skeletonBook = (p) => ({
  id: null, rev: -1, status: "queued", title: p.title || null, genre: p.genre, request: p, created_at: Date.now() / 1000,
  building_page: 0, planning: { status: "pending" }, cover: { status: "pending" }, scenes: [], ambient_enabled: false,
  progress: { done: 0, total: 2 + NUM_SCENES * 3, step: "Preparando el libro…" },
});

function showBook(book, auto) {
  narration.stop();
  clearTimeout(state.advanceTimer);
  Object.assign(state, { book: null, rev: -1, completedAt: {}, armed: null, auto, justFinished: false });
  $("book-banner").dataset.kind = "";
  setTheme(book.genre);
  show("book");
  reader.genres = state.genres;
  reader.book = null;
  reader.index = 0;
  if (auto && isBuilding(book) && book.building_page) {
    // Libro reabierto a mitad de la construcción: se muestra la última página terminada.
    let i = 0;
    while (i < NUM_SCENES && pageFinished(book, i) && pageHasContent(book, i + 1)) i++;
    reader.index = i;
  }
  onBook(book);
  clearInterval(state.tick);
  state.tick = setInterval(renderBuildBar, 1000);
}

async function startCreation(payload) {
  $("form-error").hidden = true;
  stopPolling();
  state.bookId = null;
  showBook(skeletonBook(payload), true);
  try {
    const { id } = await api.createBook(payload);
    state.bookId = id;
    history.replaceState(null, "", `#/libro/${id}`);
    pollLoop(id);
  } catch (err) {
    state.book = null;
    show("create");
    setTheme(state.genre);
    $("form-error").textContent = err.message;
    $("form-error").hidden = false;
  }
}

$("story-form").addEventListener("submit", (e) => {
  e.preventDefault();
  unlockAudio(); // el clic en "Generar" cuenta como interacción para el audio
  if (!validateForm()) return;
  startCreation({ title: $("title").value.trim(), genre: state.genre, description: $("description").value.trim() });
});
$("description").addEventListener("input", (e) => {
  $("description-counter").textContent = `${e.target.value.length} / 1500`;
  if (e.target.value.trim().length >= 10) $("description-error").hidden = true;
});

async function loadRecent() {
  try {
    const books = (await api.listBooks()).filter((b) => b.status !== "failed").slice(0, 6);
    $("recent").hidden = books.length === 0;
    const label = { queued: "Generándose…", running: "Generándose…", completed: "Listo", cancelled: "Detenido", retrying: "Listo" };
    $("recent-list").innerHTML = books.map((b) => {
      const g = state.genres.find((x) => x.key === b.genre);
      return `<li><a href="#/libro/${b.id}" data-genre="${b.genre}"><strong>${esc(b.title || "Sin título")}</strong>
        <small>${g ? g.emoji + " " + g.label : b.genre} · ${label[b.status] || "Con recursos por reintentar"}</small></a></li>`;
    }).join("");
  } catch { $("recent").hidden = true; }
}

async function openBook(id) {
  if (state.bookId === id && state.book && !screens.book.hidden) return;
  stopPolling();
  try {
    const book = await api.getBook(id);
    state.bookId = id;
    showBook(book, isBuilding(book));
    pollLoop(id);
  } catch (err) {
    location.hash = "";
    $("form-error").textContent = err.message;
    $("form-error").hidden = false;
  }
}

function route() {
  const m = location.hash.match(/^#\/libro\/([a-f0-9]{12})$/);
  if (m) { openBook(m[1]); return; }
  stopPolling();
  clearInterval(state.tick);
  clearTimeout(state.advanceTimer);
  narration.stop();
  Object.assign(state, { book: null, bookId: null, auto: false });
  document.title = "Storybook IA";
  setTheme(state.genre);
  show("create");
  loadRecent();
}
window.addEventListener("hashchange", route);

// Controles
$("prev-btn").addEventListener("click", () => goTo(reader.index - 1));
$("next-btn").addEventListener("click", () => goTo(reader.index + 1));
$("page-dots").addEventListener("click", (e) => { const b = e.target.closest("[data-page]"); if (b && !b.disabled) goTo(Number(b.dataset.page)); });
$("play-btn").addEventListener("click", () => { unlockAudio(); if (!narration.currentUrl) { state.armed = reader.index; playCurrent(true); } else narration.togglePause(); });
$("replay-btn").addEventListener("click", () => { unlockAudio(); if (!narration.currentUrl) { state.armed = reader.index; playCurrent(true); } else narration.replay(); });
$("volume").addEventListener("input", (e) => narration.setVolume(Number(e.target.value)));
$("ambient-volume").addEventListener("input", (e) => narration.setAmbientVolume(Number(e.target.value)));
$("flip-sound").addEventListener("change", (e) => pageFlipSound.setEnabled(e.target.checked));
$("unlock-btn").addEventListener("click", () => { $("audio-unlock").hidden = true; pageFlipSound.unlock(); narration.resumeAfterUnlock(); });
$("cancel-btn").addEventListener("click", cancelBook);
$("resume-btn").addEventListener("click", resumeBook);
$("new-book-btn").addEventListener("click", async () => {
  if (isBuilding(state.book) && state.book.id) {
    if (!confirm("El libro todavía se está generando. ¿Quieres detener la generación y salir?")) return;
    await cancelBook().catch(() => {});
  }
  location.hash = "";
  route();
});
narration.setVolume(Number($("volume").value));
document.addEventListener("keydown", (e) => {
  if (screens.book.hidden || e.target.matches("input, textarea")) return;
  if (e.key === "ArrowRight") goTo(reader.index + 1);
  else if (e.key === "ArrowLeft") goTo(reader.index - 1);
  else if (e.key === " " && !e.target.closest("button")) { e.preventDefault(); $("play-btn").click(); }
});
let touchX = null;
$("book").addEventListener("touchstart", (e) => { touchX = e.touches[0].clientX; }, { passive: true });
$("book").addEventListener("touchend", (e) => {
  if (touchX === null) return;
  const dx = e.changedTouches[0].clientX - touchX; touchX = null;
  if (Math.abs(dx) > 60) goTo(reader.index + (dx < 0 ? 1 : -1));
});

// ---------------------------------------------------------------------------
// 13.14 INICIO
// ---------------------------------------------------------------------------
(async function init() {
  try { state.genres = await api.genres(); renderGenres(); }
  catch (err) { $("form-error").textContent = err.message; $("form-error").hidden = false; }
  api.health().then(({ models }) => {
    $("models-info").textContent = `Texto: ${models.text_model} · Imagen: ${models.image_model} · Voz: ${models.voice}` +
      (models.ambient_enabled ? ` · Ambiente: ${models.ambient_model}` : "") + ` · Dispositivo: ${models.device.toUpperCase()}` +
      (models.vram_gib ? ` (${models.vram_gib} GB)` : "");
  }).catch(() => {});
  route();
})();
})();

"""


def page_html() -> str:
    """Página única de la aplicación con el CSS y el JavaScript incluidos."""
    return FRONTEND_HTML.replace("/*__CSS__*/", FRONTEND_CSS).replace("/*__JS__*/", FRONTEND_JS)


# ============================================================================
# 14. EJECUCIÓN PRINCIPAL
# ============================================================================
def download_models(all_profiles: bool = False):
    """Descarga los pesos una sola vez (requiere Internet). Después se puede
    trabajar sin conexión con HF_HUB_OFFLINE=1 en el archivo .env."""
    from diffusers import DiffusionPipeline
    from huggingface_hub import snapshot_download

    settings = load_settings()
    pairs = set(PROFILES.values()) if all_profiles else {(settings.text_model, settings.image_model)}
    for text_model in sorted({t for t, _ in pairs}):
        print(f"→ Texto: {text_model}")
        snapshot_download(text_model, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])
    for image_model in sorted({i for _, i in pairs}):
        print(f"→ Imagen: {image_model}")
        variant = {"variant": "fp16"} if settings.device == "cuda" and IMAGE_MODELS[image_model]["family"] in ("sd", "sdxl") else {}
        try:
            DiffusionPipeline.download(image_model, **variant)
        except (OSError, ValueError):
            if not variant:
                raise
            DiffusionPipeline.download(image_model)
    print(f"→ Voz Piper: {settings.piper_voice}")
    PiperNarrator(settings.piper_voice, settings.models_dir).load()
    if settings.enable_ambient:
        from diffusers import AudioLDMPipeline

        print(f"→ Ambiente (opcional): {settings.ambient_model}")
        AudioLDMPipeline.download(settings.ambient_model)
    print("\nListo. Para trabajar sin conexión añade HF_HUB_OFFLINE=1 a tu archivo .env.")


def main():
    parser = argparse.ArgumentParser(description="Storybook IA: libros de cuentos con IA generativa local.")
    parser.add_argument("--descargar-modelos", action="store_true", help="descarga los pesos de los modelos y termina")
    parser.add_argument("--todos-los-perfiles", action="store_true", help="con --descargar-modelos: los tres perfiles")
    parser.add_argument("--host", help="dirección de escucha (por defecto HOST del .env o 127.0.0.1)")
    parser.add_argument("--port", type=int, help="puerto (por defecto PORT del .env o 8000)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.descargar_modelos:
        download_models(args.todos_los_perfiles)
        return
    import uvicorn

    settings = load_settings()
    host, port = args.host or settings.host, args.port or settings.port
    log.info("Configuración: %s", settings.public_info())
    print(f"\n  Storybook IA → http://{host}:{port}\n")
    uvicorn.run(create_app(settings), host=host, port=port)


if __name__ == "__main__":
    main()
