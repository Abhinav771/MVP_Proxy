from datetime import datetime, timezone
import json
from app.core import redis_db
from app.core.config import small_model_config, large_model_config
from app.services.token_budget import get_all_users_usage

async def record_request():
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    await redis_db.redis_client.incr(f"telemetry:requests:{date_str}")

async def record_volatile(prompt: str):
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    await redis_db.redis_client.incr(f"telemetry:volatile:{date_str}")
    # Keep last 10 volatile prompts
    await redis_db.redis_client.lpush("telemetry:recent_volatile", prompt)
    await redis_db.redis_client.ltrim("telemetry:recent_volatile", 0, 9)

async def record_cache_hit(prompt: str, score: float):
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    await redis_db.redis_client.incr(f"telemetry:cache_hits:{date_str}")
    await redis_db.redis_client.incrbyfloat(f"telemetry:similarity_sum:{date_str}", score)
    await redis_db.redis_client.zincrby("telemetry:top_cached_prompts", 1, prompt)

async def record_cache_miss():
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    await redis_db.redis_client.incr(f"telemetry:cache_misses:{date_str}")

async def record_escalation(reason: str):
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    await redis_db.redis_client.incr(f"telemetry:escalations:{date_str}")
    await redis_db.redis_client.lpush("telemetry:recent_escalations", reason)
    await redis_db.redis_client.ltrim("telemetry:recent_escalations", 0, 9)

async def record_route(model_type: str, method: str):
    """
    model_type: "small" or "large"
    method: "rule" or "code" or "ml"
    """
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    await redis_db.redis_client.incr(f"telemetry:route:{model_type}:{date_str}")
    await redis_db.redis_client.incr(f"telemetry:route_method:{method}:{date_str}")

async def get_dashboard_metrics() -> dict:
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    r = redis_db.redis_client
    
    # Fetch all simple counters
    reqs = await r.get(f"telemetry:requests:{date_str}")
    hits = await r.get(f"telemetry:cache_hits:{date_str}")
    misses = await r.get(f"telemetry:cache_misses:{date_str}")
    vols = await r.get(f"telemetry:volatile:{date_str}")
    escs = await r.get(f"telemetry:escalations:{date_str}")
    route_small = await r.get(f"telemetry:route:small:{date_str}")
    route_large = await r.get(f"telemetry:route:large:{date_str}")
    meth_rule = await r.get(f"telemetry:route_method:rule:{date_str}")
    meth_code = await r.get(f"telemetry:route_method:code:{date_str}")
    meth_ml = await r.get(f"telemetry:route_method:ml:{date_str}")
    sim_sum = await r.get(f"telemetry:similarity_sum:{date_str}")
    
    hits_val = int(hits) if hits else 0
    misses_val = int(misses) if misses else 0
    sim_val = float(sim_sum) if sim_sum else 0.0
    
    avg_sim = (sim_val / hits_val) if hits_val > 0 else 0.0

    recent_vols_bytes = await r.lrange("telemetry:recent_volatile", 0, 9)
    recent_vols = [v.decode("utf-8") if isinstance(v, bytes) else v for v in recent_vols_bytes] if recent_vols_bytes else []
    
    recent_escs_bytes = await r.lrange("telemetry:recent_escalations", 0, 9)
    recent_escs = [e.decode("utf-8") if isinstance(e, bytes) else e for e in recent_escs_bytes] if recent_escs_bytes else []

    top_prompts_tuples = await r.zrevrange("telemetry:top_cached_prompts", 0, 4, withscores=True)
    top_prompts = [{"prompt": p.decode("utf-8") if isinstance(p, bytes) else p, "count": int(s)} for p, s in top_prompts_tuples]

    # Exact token savings calculation
    all_users = await get_all_users_usage()
    global_small_tokens = sum(u["small_model"]["used"] for u in all_users)
    global_large_tokens = sum(u["large_model"]["used"] for u in all_users)
    global_actual_cost = sum(u.get("actual_cost", 0.0) for u in all_users)
    
    s_cost = small_model_config.cost_per_1m
    l_cost = large_model_config.cost_per_1m
    
    rs_val = int(route_small) if route_small else 0
    
    cost_without_proxy = (global_small_tokens + global_large_tokens) / 1_000_000 * l_cost
    total_savings = max(0, cost_without_proxy - global_actual_cost)
    
    return {
        "date": date_str,
        "total_requests": int(reqs) if reqs else 0,
        "cache_hits": hits_val,
        "cache_misses": misses_val,
        "volatile_requests": int(vols) if vols else 0,
        "escalations": int(escs) if escs else 0,
        "route_small": rs_val,
        "route_large": int(route_large) if route_large else 0,
        "route_methods": {
            "rule": int(meth_rule) if meth_rule else 0,
            "code": int(meth_code) if meth_code else 0,
            "ml": int(meth_ml) if meth_ml else 0
        },
        "avg_similarity": avg_sim,
        "recent_volatile": recent_vols,
        "recent_escalations": recent_escs,
        "top_cached_prompts": top_prompts,
        "estimated_savings": total_savings,
        "actual_cost": global_actual_cost,
        "global_small_tokens": global_small_tokens,
        "global_large_tokens": global_large_tokens
    }
