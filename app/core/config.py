import os
import spacy
from dataclasses import dataclass
from dotenv import load_dotenv
from groq import Groq
from google import genai
from pinecone import Pinecone

load_dotenv()

# ── Groq client (kept for legacy use) ────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print(f"DEBUG KEY LOADED: '{GROQ_API_KEY}'")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set in .env file")
groq_client = Groq(api_key=GROQ_API_KEY)

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


# ── Universal LLM config (loaded from .env) ───────────────────────────────────
@dataclass(frozen=True)
class LLMConfig:
    """
    Holds the active LLM provider settings read from .env.

    To switch provider, change these three values in .env:
        LLM_BASE_URL    — OpenAI-compatible API endpoint
        LLM_API_KEY     — Provider API key
        LLM_MODEL_NAME  — Model identifier string

    Supported providers (anything OpenAI-compatible):
        Groq       → https://api.groq.com/openai/v1
        OpenAI     → https://api.openai.com/v1
        Mistral    → https://api.mistral.ai/v1
        Together   → https://api.together.xyz/v1
        Ollama     → http://localhost:11434/v1
        Perplexity → https://api.perplexity.ai
        Anyscale   → https://api.endpoints.anyscale.com/v1
    """
    base_url: str
    api_key: str
    model_name: str


_llm_base_url = os.getenv("LLM_BASE_URL")
_llm_api_key = os.getenv("LLM_API_KEY")
_llm_model_name = os.getenv("LLM_MODEL_NAME")

if not all([_llm_base_url, _llm_api_key, _llm_model_name]):
    raise RuntimeError(
        "LLM provider config is incomplete. "
        "Set LLM_BASE_URL, LLM_API_KEY, and LLM_MODEL_NAME in your .env file."
    )

# Singleton: import this anywhere to access the active LLM provider settings
llm_config = LLMConfig(
    base_url=_llm_base_url,
    api_key=_llm_api_key,
    model_name=_llm_model_name,
)

print(f"✅ LLM provider: {llm_config.base_url} | model: {llm_config.model_name}")