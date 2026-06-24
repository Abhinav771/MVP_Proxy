from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.models.schemas import ChatRequest
from app.core import redis_db
from app.core.config import llm_config
from app.services.rate_limiter import rate_limit
from app.services.nlp_service import is_volatile
from app.services.vector_db import embed, query_similar, store_embedding
from app.services.llm_client import async_stream_llm

router = APIRouter()
SIMILARITY_THRESHOLD = 0.85


@router.get("/health")
async def check_health():
    try:
        await redis_db.redis_client.ping()
        redis_status = "ok"
    except Exception:
        redis_status = "unavailable"

    # Embed sanity check
    vector = await embed("hello")
    print(f"Embed test → length: {len(vector)}, first value: {vector[0]:.4f}")

    return {
        "status": "ok",
        "redis": redis_status,
        "llm_provider": llm_config.base_url,
        "llm_model": llm_config.model_name,
    }


@router.post("/chat")
async def chat_endpoint(request: ChatRequest, req: Request):
    await rate_limit(req.client.host)

    # ── Step 1: Check volatility FIRST ───────────────────────────────────────
    if is_volatile(request.prompt):
        print(f"⚡ VOLATILE | skipping cache | prompt: '{request.prompt}'")

        async def volatile_stream():
            async for token in async_stream_llm(
                base_url=llm_config.base_url,
                api_key=llm_config.api_key,
                model_name=llm_config.model_name,
                prompt=request.prompt,
            ):
                yield token

        return StreamingResponse(volatile_stream(), media_type="text/plain")

    # ── Step 2: Non-volatile → check semantic cache ───────────────────────────
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
        if is_cache_hit:
            cached = matches[0].metadata
            synthesis_prompt = (
                f'A user previously asked: "{cached["prompt"]}"\n'
                f'And received this answer: "{cached["response"]}"\n\n'
                f'Now the user is asking: "{request.prompt}"\n\n'
                f"Using the above as context, give a fresh and direct answer to the new question."
            )
            async for token in async_stream_llm(
                base_url=llm_config.base_url,
                api_key=llm_config.api_key,
                model_name=llm_config.model_name,
                prompt=synthesis_prompt,
            ):
                yield token

        else:
            # Cache miss: stream tokens live while collecting the full response
            collector = []
            async for token in async_stream_llm(
                base_url=llm_config.base_url,
                api_key=llm_config.api_key,
                model_name=llm_config.model_name,
                prompt=request.prompt,
                collector=collector,   # filled once the stream ends
            ):
                yield token

            # Store in Pinecone for future cache hits (collector[0] = full text)
            full_response = collector[0] if collector else ""
            if full_response and not full_response.startswith("Error:"):
                vector_id = str(abs(hash(request.prompt)))[:16]
                await store_embedding(
                    id=vector_id,
                    vector=prompt_vector,
                    metadata={
                        "prompt": request.prompt,
                        "response": full_response[:1000],
                    },
                )
                print(f"💾 Stored in Pinecone | id: {vector_id} | prompt: '{request.prompt}'")

    return StreamingResponse(stream_generator(), media_type="text/plain")