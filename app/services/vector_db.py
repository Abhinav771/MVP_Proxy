import asyncio
from google.genai import types
from app.core.config import google_client, pinecone_index

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
    """Store a vector with metadata in Pinecone"""
    await asyncio.to_thread(
        pinecone_index.upsert,
        vectors=[{
            "id": id,           
            "values": vector,   
            "metadata": metadata  
        }]
    )

async def query_similar(vector: list[float], top_k: int = 1):
    """Find the most similar vector in Pinecone"""
    result = await asyncio.to_thread(
        pinecone_index.query,
        vector=vector,
        top_k=top_k,
        include_metadata=True   
    )
    return result.matches