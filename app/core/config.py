import os
import spacy
from dataclasses import dataclass
from dotenv import load_dotenv
from google import genai
from pinecone import Pinecone

load_dotenv()

# ── Google Embedding client ───────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is not set in .env file")
google_client = genai.Client(api_key=GOOGLE_API_KEY)

# ── Pinecone client ───────────────────────────────────────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is not set in .env file")
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index("semantic-cache")

# ── Spacy setup ───────────────────────────────────────────────────────────────
nlp = spacy.load("en_core_web_sm")


# ── LLM Config dataclass ─────────────────────────────────────────────────────
@dataclass(frozen=True)
class LLMConfig:
    """
    Holds one LLM provider's settings (base_url, api_key, model_name).

    Used by the RouteLLM system to define two tiers:
        small_model_config  — cheap/fast  (easy tasks)
        large_model_config  — expensive   (hard tasks)

    Each tier can point to a different provider or the same one.
    """
    base_url: str
    api_key: str
    model_name: str


def _load_model_config(prefix: str) -> LLMConfig:
    """Load a LLMConfig from env vars with the given prefix (SMALL_MODEL / LARGE_MODEL)."""
    base_url = os.getenv(f"{prefix}_BASE_URL")
    api_key = os.getenv(f"{prefix}_API_KEY")
    model_name = os.getenv(f"{prefix}_NAME")

    if not all([base_url, api_key, model_name]):
        raise RuntimeError(
            f"{prefix} config is incomplete. "
            f"Set {prefix}_BASE_URL, {prefix}_API_KEY, and {prefix}_NAME in .env."
        )

    return LLMConfig(base_url=base_url, api_key=api_key, model_name=model_name)


# ── Two-tier model configs (used by RouteLLM) ────────────────────────────────
small_model_config = _load_model_config("SMALL_MODEL")
large_model_config = _load_model_config("LARGE_MODEL")

print(f"✅ Small model: {small_model_config.base_url} | {small_model_config.model_name}")
print(f"✅ Large model: {large_model_config.base_url} | {large_model_config.model_name}")