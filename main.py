import os
import asyncio  
import spacy
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv
from groq import Groq, APIConnectionError, APIStatusError, APITimeoutError
import redis.asyncio as aioredis
from google import genai
from google.genai import types
from pinecone import Pinecone

load_dotenv()

# ── Redis client (set during lifespan) ──────────────────
redis_client = None
# ── Rate Limiter ─────────────────────────────────────────
MAX_REQUESTS_PER_MINUTE = 10  # adjust as needed
SIMILARITY_THRESHOLD = 0.85 

async def rate_limit(ip: str):
    key = f"rate:{ip}"
    
    count = await redis_client.incr(key)   # atomically increment; creates key if missing
    
    if count == 1:
        # First request in this window → start the 60s expiry clock
        await redis_client.expire(key, 60)
    
    if count > MAX_REQUESTS_PER_MINUTE:
        # How many seconds until the window resets?
        ttl = await redis_client.ttl(key)
        raise HTTPException(
            status_code=429,
            detail=f"Too many requests. Try again in {ttl}s."
        )

# ── Lifespan: runs on startup and shutdown ───────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    global redis_client
    redis_client = aioredis.from_url(
        "redis://localhost:6379",
        encoding="utf-8",
        decode_responses=True
    )
    print("✅ Connected to Redis")
    
    yield  # server runs here
    
    # SHUTDOWN
    await redis_client.aclose()
    print("🔴 Redis connection closed")


app = FastAPI(lifespan=lifespan)

# ── Groq client ──────────────────────────────────────────
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY is not set in .env file")

client = Groq(api_key=api_key)

# ── Google Embedding client ───────────────────────────────
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise RuntimeError("GOOGLE_API_KEY is not set in .env file")
google_client = genai.Client(api_key=google_api_key)

# ── Pinecone client ───────────────────────────────────────
pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    raise RuntimeError("PINECONE_API_KEY is not set in .env file")

pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("semantic-cache")


nlp = spacy.load("en_core_web_sm")


VOLATILE_KEYWORDS = {
    "now", "today", "tonight", "currently", "current",
    "live", "latest", "right now", "at the moment",
    "breaking", "trending", "real-time", "realtime",
    "this week", "this month", "yesterday", "tomorrow",
    "price", "stock", "weather", "score", "news"
}

VOLATILE_ENTITIES = {"DATE", "TIME"}

def is_volatile(prompt: str) -> bool:
    prompt_lower = prompt.lower()

    if any(keyword in prompt_lower for keyword in VOLATILE_KEYWORDS):
        print(f"⚡ VOLATILE (keyword) | prompt: '{prompt}'")
        return True

    doc = nlp(prompt)
    for ent in doc.ents:
        if ent.label_ in VOLATILE_ENTITIES:
            print(f"⚡ VOLATILE (NER: {ent.label_}={ent.text}) | prompt: '{prompt}'")
            return True

    return False



# ── Embedding helper ──────────────────────────────────────
async def embed(text: str) -> list[float]:
    result = await asyncio.to_thread(          # runs sync SDK in async context
        google_client.models.embed_content,
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(
            task_type="SEMANTIC_SIMILARITY",
            output_dimensionality=768
        )
    )
    return result.embeddings[0].values


# ── Pinecone helpers ──────────────────────────────────────
async def store_embedding(id: str, vector: list[float], metadata: dict):
    """Store a vector with metadata in Pinecone"""
    await asyncio.to_thread(
        index.upsert,
        vectors=[{
            "id": id,           # unique ID for this vector
            "values": vector,   # the 768-dimensional vector
            "metadata": metadata  # extra data (prompt, response, etc.)
        }]
    )

async def query_similar(vector: list[float], top_k: int = 1):
    """Find the most similar vector in Pinecone"""
    result = await asyncio.to_thread(
        index.query,
        vector=vector,
        top_k=top_k,
        include_metadata=True   # return metadata alongside the match
    )
    return result.matches        # list of matches with score + metadata

class ChatRequest(BaseModel):
    prompt: str

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Prompt cannot be empty")
        return v


@app.get("/health")
async def check_health():
    try:
        await redis_client.ping()
        redis_status = "ok"
    except Exception:
        redis_status = "unavailable"

    # 👇 temporary embed test
    vector = await embed("hello")
    print(f"Embed test → length: {len(vector)}, first value: {vector[0]:.4f}")

    return {"status": "ok", "redis": redis_status}


@app.post("/chat")
async def chat_endpoint(request: ChatRequest, req: Request):
    await rate_limit(req.client.host)

    # ── Step 1: Check volatility FIRST ───────────────────
    if is_volatile(request.prompt):
        print(f"⚡ VOLATILE | skipping cache | prompt: '{request.prompt}'")

        async def volatile_stream():
            try:
                stream = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": request.prompt}],
                    stream=True,
                    timeout=30
                )
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content

            except APITimeoutError:
                yield "Error: Request timed out."
            except APIStatusError as e:
                if e.status_code == 401:
                    yield "Error: Invalid API key."
                elif e.status_code == 429:
                    yield "Error: Rate limit hit. Please wait."
                elif e.status_code == 500:
                    yield "Error: Groq server error."
                else:
                    yield f"Error: {e.status_code} - {e.message}"
            except APIConnectionError:
                yield "Error: Could not connect to Groq."
            except Exception as e:
                yield f"Unexpected error: {str(e)}"

        return StreamingResponse(volatile_stream(), media_type="text/plain")

    # ── Step 2: Non-volatile → check cache ───────────────
    prompt_vector = await embed(request.prompt)
    matches = await query_similar(prompt_vector, top_k=1)

    is_cache_hit = (
        len(matches) > 0 and
        matches[0].score >= SIMILARITY_THRESHOLD
    )

    if is_cache_hit:
        cached = matches[0].metadata
        print(f"🟢 CACHE HIT  | score: {matches[0].score:.4f} | matched: '{cached['prompt']}'")
    else:
        print(f"🔴 CACHE MISS | prompt: '{request.prompt}'")

    async def stream_generator():
        try:
            if is_cache_hit:
                cached = matches[0].metadata
                synthesis_prompt = f"""A user previously asked: "{cached['prompt']}"
And received this answer: "{cached['response']}"

Now the user is asking: "{request.prompt}"

Using the above as context, give a fresh and direct answer to the new question."""

                stream = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": synthesis_prompt}],
                    stream=True,
                    timeout=30
                )
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content

            else:
                stream = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": request.prompt}],
                    stream=True,
                    timeout=30
                )
                full_response = ""
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        yield content

                if full_response:
                    vector_id = str(abs(hash(request.prompt)))[:16]
                    await store_embedding(
                        id=vector_id,
                        vector=prompt_vector,
                        metadata={
                            "prompt": request.prompt,
                            "response": full_response[:1000]
                        }
                    )
                    print(f"💾 Stored in Pinecone | id: {vector_id} | prompt: '{request.prompt}'")

        except APITimeoutError:
            yield "Error: Request timed out."
        except APIStatusError as e:
            if e.status_code == 401:
                yield "Error: Invalid API key."
            elif e.status_code == 429:
                yield "Error: Rate limit hit. Please wait."
            elif e.status_code == 500:
                yield "Error: Groq server error."
            else:
                yield f"Error: {e.status_code} - {e.message}"
        except APIConnectionError:
            yield "Error: Could not connect to Groq."
        except Exception as e:
            yield f"Unexpected error: {str(e)}"

    return StreamingResponse(stream_generator(), media_type="text/plain")