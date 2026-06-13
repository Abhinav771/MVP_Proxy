import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv
from groq import Groq, APIConnectionError, APIStatusError, APITimeoutError
import redis.asyncio as aioredis

load_dotenv()

# ── Redis client (set during lifespan) ──────────────────
redis_client = None
# ── Rate Limiter ─────────────────────────────────────────
MAX_REQUESTS_PER_MINUTE = 10  # adjust as needed

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
    # Also check Redis is alive
    try:
        await redis_client.ping()
        redis_status = "ok"
    except Exception:
        redis_status = "unavailable"
    return {"status": "ok", "redis": redis_status}


@app.post("/chat")
async def chat_endpoint(request: ChatRequest,req: Request):
    await rate_limit(req.client.host)
    async def stream_generator():
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

    return StreamingResponse(stream_generator(), media_type="text/plain")