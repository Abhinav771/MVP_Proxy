from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from groq import APIConnectionError, APIStatusError, APITimeoutError

from app.models.schemas import ChatRequest
from app.core import redis_db
from app.core.config import groq_client
from app.services.rate_limiter import rate_limit
from app.services.nlp_service import is_volatile
from app.services.vector_db import embed, query_similar, store_embedding

router = APIRouter()
SIMILARITY_THRESHOLD = 0.85

@router.get("/health")
async def check_health():
    try:
        await redis_db.redis_client.ping()
        redis_status = "ok"
    except Exception:
        redis_status = "unavailable"

    # 👇 temporary embed test
    vector = await embed("hello")
    print(f"Embed test → length: {len(vector)}, first value: {vector[0]:.4f}")

    return {"status": "ok", "redis": redis_status}


@router.post("/chat")
async def chat_endpoint(request: ChatRequest, req: Request):
    await rate_limit(req.client.host)

    # ── Step 1: Check volatility FIRST ───────────────────
    if is_volatile(request.prompt):
        print(f"⚡ VOLATILE | skipping cache | prompt: '{request.prompt}'")

        async def volatile_stream():
            try:
                stream = groq_client.chat.completions.create(
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

                stream = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": synthesis_prompt}],
                    stream=True,
                    timeout=30
                )
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content

            else:
                stream = groq_client.chat.completions.create(
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