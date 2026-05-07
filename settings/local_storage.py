import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
VECTOR_DB_DIR = DATA_DIR / "vector_db"
MODEL_CACHE_DIR = DATA_DIR / "model_cache"
APP_CACHE_DIR = DATA_DIR / "cache"


def configure_local_storage():
    """Force model/tool caches to stay inside the project directory."""
    for path in [DATA_DIR, VECTOR_DB_DIR, MODEL_CACHE_DIR, APP_CACHE_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("XDG_CACHE_HOME", str(APP_CACHE_DIR))
    os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR / "hf"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(MODEL_CACHE_DIR / "hf" / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(MODEL_CACHE_DIR / "transformers"))
    os.environ.setdefault(
        "SENTENCE_TRANSFORMERS_HOME", str(MODEL_CACHE_DIR / "sentence_transformers")
    )
    os.environ.setdefault("TORCH_HOME", str(MODEL_CACHE_DIR / "torch"))
