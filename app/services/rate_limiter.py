from fastapi import HTTPException
from app.core import redis_db

MAX_REQUESTS_PER_MINUTE = 10

async def rate_limit(ip: str):
    key = f"rate:{ip}"
    
    # atomically increment; creates key if missing
    count = await redis_db.redis_client.incr(key)   
    
    if count == 1:
        # First request in this window → start the 60s expiry clock
        await redis_db.redis_client.expire(key, 60)
    
    if count > MAX_REQUESTS_PER_MINUTE:
        # How many seconds until the window resets?
        ttl = await redis_db.redis_client.ttl(key)
        raise HTTPException(
            status_code=429,
            detail=f"Too many requests. Try again in {ttl}s."
        )