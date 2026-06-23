import os
import spacy
from dotenv import load_dotenv
from groq import Groq
from google import genai
from pinecone import Pinecone

load_dotenv()

# ── Groq client ──────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print(f"DEBUG KEY LOADED: '{GROQ_API_KEY}'")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set in .env file")
groq_client = Groq(api_key=GROQ_API_KEY)

# ── Google Embedding client ───────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is not set in .env file")
google_client = genai.Client(api_key=GOOGLE_API_KEY)

# ── Pinecone client ───────────────────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is not set in .env file")
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index("semantic-cache")

# ── Spacy setup ───────────────────────────────────────────
nlp = spacy.load("en_core_web_sm")