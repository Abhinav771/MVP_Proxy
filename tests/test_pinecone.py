import os
import asyncio
from pinecone import Pinecone
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# ── Clients ──────────────────────────────────────────────
google_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("semantic-cache")

# ── Helpers ──────────────────────────────────────────────
async def embed(text: str) -> list[float]:
    result = await asyncio.to_thread(
        google_client.models.embed_content,
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(
            task_type="SEMANTIC_SIMILARITY",
            output_dimensionality=768
        )
    )
    return result.embeddings[0].values

async def store_embedding(id: str, vector: list[float], metadata: dict):
    await asyncio.to_thread(
        index.upsert,
        vectors=[{"id": id, "values": vector, "metadata": metadata}]
    )

async def query_similar(vector: list[float], top_k: int = 1):
    result = await asyncio.to_thread(
        index.query,
        vector=vector,
        top_k=top_k,
        include_metadata=True
    )
    return result.matches

# ── Test ─────────────────────────────────────────────────
async def main():
    SIMILARITY_THRESHOLD = 0.85

    # Step 1 — Store "What is Python?"
    print("📥 Storing: 'What is Python?'")
    original_prompt = "What is Python?"
    original_vector = await embed(original_prompt)
    await store_embedding(
        id="python-001",
        vector=original_vector,
        metadata={
            "prompt": original_prompt,
            "response": "Python is a high-level programming language known for simplicity."
        }
    )
    print("✅ Stored!\n")

    # Step 2 — Query with "Explain Python"
    query_prompt = "Explain Python"
    print(f"🔍 Querying with: '{query_prompt}'")
    query_vector = await embed(query_prompt)
    matches = await query_similar(query_vector)

    if not matches:
        print("❌ No matches found")
        return

    match = matches[0]
    print(f"\n📊 Result:")
    print(f"   Matched prompt : {match.metadata['prompt']}")
    print(f"   Score          : {match.score:.4f}")
    print(f"   Threshold      : {SIMILARITY_THRESHOLD}")

    if match.score >= SIMILARITY_THRESHOLD:
        print(f"\n🟢 CACHE HIT! Score {match.score:.4f} >= {SIMILARITY_THRESHOLD}")
        print(f"   Cached response: {match.metadata['response']}")
        print(f"   ✅ Would skip Groq API call!")
    else:
        print(f"\n🔴 CACHE MISS. Score {match.score:.4f} < {SIMILARITY_THRESHOLD}")
        print(f"   Would call Groq API...")

asyncio.run(main())