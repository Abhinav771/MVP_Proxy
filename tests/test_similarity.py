import os
import asyncio
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

google_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

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

def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = sum(x ** 2 for x in a) ** 0.5
    magnitude_b = sum(x ** 2 for x in b) ** 0.5
    return dot_product / (magnitude_a * magnitude_b)

async def main():
    texts = [
        "hello",
        "hi there",
        "what is the capital of France?",   # unrelated → should be far
    ]

    print("Generating embeddings...\n")
    vectors = {}
    for text in texts:
        vectors[text] = await embed(text)
        print(f"✅ '{text}'")
        print(f"   Length  : {len(vectors[text])}")
        print(f"   First 3 : {[round(v, 4) for v in vectors[text][:3]]}\n")

    print("=" * 50)
    print("SIMILARITY SCORES (1.0 = identical, 0.0 = unrelated)")
    print("=" * 50)

    pairs = [
        ("hello", "hi there"),                          # should be HIGH
        ("hello", "what is the capital of France?"),    # should be LOW
        ("hi there", "what is the capital of France?"), # should be LOW
    ]

    for a, b in pairs:
        score = cosine_similarity(vectors[a], vectors[b])
        bar = "🟢" if score > 0.8 else "🔴"
        print(f"{bar} '{a}' vs '{b}'")
        print(f"   Score: {score:.4f}\n")

asyncio.run(main())