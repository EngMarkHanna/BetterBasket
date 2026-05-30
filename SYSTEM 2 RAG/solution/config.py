"""Credential + configuration loading for SYSTEM 2 RAG.

Two surfaces:
  - OpenAI public API (embeddings only) - key read from .env at repo root.
  - Azure OpenAI deployment (LLM judge calls) - reused from openai_creds.yaml.

Keep the two surfaces separate so we can swap embedding providers without
touching the LLM client and vice versa.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[2]
DOTENV_PATH = REPO_ROOT / ".env"
AZURE_CREDS_PATH = REPO_ROOT / "openai_creds.yaml"


def _ensure_env_loaded() -> None:
    if DOTENV_PATH.exists():
        load_dotenv(DOTENV_PATH, override=False)


# Native output dim per supported model. Hard-coded so we never silently
# request a wrong-shape vector. Add new entries here when adopting a model.
_MODEL_NATIVE_DIM: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


def openai_embedding_config() -> dict:
    """Returns the OpenAI public-API config used for embeddings.

    Raises a clear error if the key isn't set so we fail fast.

    Note: the embedding dimension is fixed by the chosen model (no
    OPENAI_EMBEDDING_DIM override). The OpenAI `dimensions=` parameter
    is supported only on `-3-small`/`-3-large` and we don't use it for
    System 2 - we always take the native dim. If you want a smaller
    vector, change the model and update _MODEL_NATIVE_DIM.
    """
    _ensure_env_loaded()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            f"OPENAI_API_KEY is empty. Paste your key into {DOTENV_PATH} and re-run."
        )
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    if model not in _MODEL_NATIVE_DIM:
        raise RuntimeError(
            f"Unknown embedding model {model!r}. Add to _MODEL_NATIVE_DIM in config.py."
        )
    return {
        "api_key": api_key,
        "model": model,
        "dim": _MODEL_NATIVE_DIM[model],
        "batch": int(os.getenv("OPENAI_EMBEDDING_BATCH", "100")),
    }


def azure_llm_config() -> dict:
    """Returns the Azure OpenAI config used for LLM judge calls."""
    if not AZURE_CREDS_PATH.exists():
        raise RuntimeError(f"Azure creds file missing: {AZURE_CREDS_PATH}")
    blob = yaml.safe_load(AZURE_CREDS_PATH.read_text())
    creds = blob["openai"]
    return {
        "endpoint": creds["endpoint"],
        "api_key": creds["api_key"],
        "deployment": creds["deployment_name"],
    }
