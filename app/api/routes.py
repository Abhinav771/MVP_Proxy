from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.models.schemas import ChatRequest, SetLimitRequest
from app.core import redis_db
from app.core.config import small_model_config, large_model_config
from app.services.rate_limiter import rate_limit
from app.services.nlp_service import is_volatile
from app.services.vector_db import embed, query_similar, store_embedding
from app.services.llm_client import async_stream_llm
from app.services.router import get_model_config, is_low_quality
from app.services.token_budget import (
    count_tokens, check_budget, check_budget_boolean,
    consume_budget, set_custom_limit, get_current_usage, get_all_users_usage
)

router = APIRouter()
SIMILARITY_THRESHOLD = 0.93

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
        "small_model": small_model_config.model_name,
        "large_model": large_model_config.model_name,
    }

@router.post("/admin/set-limit")
async def admin_set_limit(request: SetLimitRequest):
    if request.model_type not in ["small", "large"]:
        raise HTTPException(status_code=400, detail="model_type must be 'small' or 'large'")
    await set_custom_limit(request.ip, request.model_type, request.limit)
    return {"status": "ok", "message": f"Limit set to {request.limit} for {request.model_type} model on {request.ip}"}

@router.get("/admin/usage/{ip}")
async def admin_get_usage(ip: str):
    return await get_current_usage(ip)

@router.get("/admin/users")
async def admin_get_all_users():
    return {"users": await get_all_users_usage()}

@router.post("/chat")
async def chat_endpoint(request: ChatRequest, req: Request):
    client_ip = req.client.host
    await rate_limit(client_ip)

    # ── Route the prompt to the right model ──────────────────────────────────
    config, decision = get_model_config(request.prompt)
    routed_model = "large" if decision["route"] == "large" else "small"
    needs_quality_check = decision["route"] == "small_then_escalate"

    # ── Token Budget Check ───────────────────────────────────────────────────
    prompt_tokens = count_tokens(request.prompt)
    await check_budget(client_ip, routed_model, prompt_tokens)
    await consume_budget(client_ip, routed_model, prompt_tokens)

    print(
        f"🧠 ROUTED → {decision['route']} ({routed_model}) | "
        f"model: {config.model_name} | "
        f"confidence: {decision['confidence']:.2f} | "
        f"reason: {decision['reason']}"
    )

    # ── Step 1: Check volatility FIRST ───────────────────────────────────────
    if is_volatile(request.prompt):
        print(f"⚡ VOLATILE | skipping cache | prompt: '{request.prompt}'")

        async def volatile_stream():
            collector = []
            async for token in async_stream_llm(
                base_url=config.base_url,
                api_key=config.api_key,
                model_name=config.model_name,
                prompt=request.prompt,
                collector=collector,
            ):
                yield token

            # Escalate if small_then_escalate and quality is poor
            if needs_quality_check and collector:
                is_bad, reason = is_low_quality(collector[0])
                if is_bad:
                    has_large_budget = await check_budget_boolean(client_ip, "large", prompt_tokens)
                    if not has_large_budget:
                        yield f"\n\n---\n[Error: Tried to escalate to large model due to low quality ('{reason}'), but daily large model token limit is exceeded.]"
                    else:
                        print(f"⬆️ ESCALATING → large | reason: {reason} | model: {large_model_config.model_name}")
                        await consume_budget(client_ip, "large", prompt_tokens)
                        yield "\n\n---\n🔄 Improving answer with a more powerful model...\n\n"
                        large_collector = []
                        async for token in async_stream_llm(
                            base_url=large_model_config.base_url,
                            api_key=large_model_config.api_key,
                            model_name=large_model_config.model_name,
                            prompt=request.prompt,
                            collector=large_collector,
                        ):
                            yield token
                        if large_collector and large_collector[0]:
                            await consume_budget(client_ip, "large", count_tokens(large_collector[0]))
                        return
                else:
                    print(f"✅ QUALITY CHECK PASSED | small model response accepted ({len(collector[0])} chars)")

            if collector and collector[0]:
                await consume_budget(client_ip, routed_model, count_tokens(collector[0]))

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
            # Re-check budget using synthesis prompt tokens instead
            synth_prompt_tokens = count_tokens(synthesis_prompt)
            # We already consumed request.prompt tokens, so let's consume the diff
            diff = max(0, synth_prompt_tokens - prompt_tokens)
            await consume_budget(client_ip, routed_model, diff)

            collector = []
            async for token in async_stream_llm(
                base_url=config.base_url,
                api_key=config.api_key,
                model_name=config.model_name,
                prompt=synthesis_prompt,
                collector=collector,
            ):
                yield token

            # Escalate if small_then_escalate and quality is poor
            if needs_quality_check and collector:
                is_bad, reason = is_low_quality(collector[0])
                if is_bad:
                    has_large_budget = await check_budget_boolean(client_ip, "large", synth_prompt_tokens)
                    if not has_large_budget:
                        yield f"\n\n---\n[Error: Tried to escalate to large model due to low quality ('{reason}'), but daily large model token limit is exceeded.]"
                    else:
                        print(f"⬆️ ESCALATING → large | reason: {reason} | model: {large_model_config.model_name}")
                        await consume_budget(client_ip, "large", synth_prompt_tokens)
                        yield "\n\n---\n🔄 Improving answer with a more powerful model...\n\n"
                        large_collector = []
                        async for token in async_stream_llm(
                            base_url=large_model_config.base_url,
                            api_key=large_model_config.api_key,
                            model_name=large_model_config.model_name,
                            prompt=synthesis_prompt,
                            collector=large_collector,
                        ):
                            yield token
                        if large_collector and large_collector[0]:
                            await consume_budget(client_ip, "large", count_tokens(large_collector[0]))
                        return
                else:
                    print(f"✅ QUALITY CHECK PASSED | small model response accepted ({len(collector[0])} chars)")

            if collector and collector[0]:
                await consume_budget(client_ip, routed_model, count_tokens(collector[0]))

        else:
            # Cache miss: stream tokens live while collecting the full response
            collector = []
            async for token in async_stream_llm(
                base_url=config.base_url,
                api_key=config.api_key,
                model_name=config.model_name,
                prompt=request.prompt,
                collector=collector,
            ):
                yield token

            full_response = collector[0] if collector else ""
            used_large_model = False

            # Escalate if small_then_escalate and quality is poor
            if needs_quality_check and full_response:
                is_bad, reason = is_low_quality(full_response)
                if is_bad:
                    has_large_budget = await check_budget_boolean(client_ip, "large", prompt_tokens)
                    if not has_large_budget:
                        yield f"\n\n---\n[Error: Tried to escalate to large model due to low quality ('{reason}'), but daily large model token limit is exceeded.]"
                    else:
                        print(f"⬆️ ESCALATING → large | reason: {reason} | model: {large_model_config.model_name}")
                        await consume_budget(client_ip, "large", prompt_tokens)
                        yield "\n\n---\n🔄 Improving answer with a more powerful model...\n\n"
                        large_collector = []
                        async for token in async_stream_llm(
                            base_url=large_model_config.base_url,
                            api_key=large_model_config.api_key,
                            model_name=large_model_config.model_name,
                            prompt=request.prompt,
                            collector=large_collector,
                        ):
                            yield token
                        
                        full_response = large_collector[0] if large_collector else full_response
                        used_large_model = True
                else:
                    print(f"✅ QUALITY CHECK PASSED | small model response accepted ({len(full_response)} chars)")
            
            # Consume budget for output tokens
            if full_response:
                consumed_model = "large" if used_large_model else routed_model
                await consume_budget(client_ip, consumed_model, count_tokens(full_response))

            # Store in Pinecone for future cache hits
            if full_response and not full_response.startswith("Error:") and not "[Error:" in full_response:
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