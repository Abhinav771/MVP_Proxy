import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# New client style
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Embed two similar texts
texts = ["hello", "hi there"]

for text in texts:
    result = client.models.embed_content(
        model="gemini-embedding-001",          # text-only model
        contents=text,
        config=types.EmbedContentConfig(
            task_type="SEMANTIC_SIMILARITY",   # perfect for caching use case
            output_dimensionality=768          # smaller = faster, still accurate
        )
    )
    vector = result.embeddings[0].values
    print(f"Text     : '{text}'")
    print(f"Length   : {len(vector)}")         # should be 768
    print(f"First 5  : {vector[:5]}")
    print()