import tiktoken
from datetime import datetime, timezone
from fastapi import HTTPException
from app.core import redis_db

# Default daily limits
DEFAULT_SMALL_LIMIT = 1_000_000
DEFAULT_LARGE_LIMIT = 100_000

# Initialize the tokenizer
# cl100k_base is fast and serves as a good proxy for most models including Llama.
encoder = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """Return the number of tokens in a string."""
    if not text:
        return 0
    return len(encoder.encode(text))

def get_budget_key(ip: str, model_type: str, date_str: str = None) -> str:
    """Generate the daily token usage key for a given IP and model type."""
    date_str = date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"tokens:{ip}:{model_type}:{date_str}"

def get_limit_key(ip: str, model_type: str, date_str: str = None) -> str:
    """Generate the daily custom limit key for a given IP and model type."""
    date_str = date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"limit:{ip}:{model_type}:{date_str}"

async def get_user_limit(ip: str, model_type: str) -> int:
    """
    Fetch the custom limit for the user from Redis if it exists.
    Otherwise, return the default for that model type.
    """
    key = get_limit_key(ip, model_type)
    custom_limit = await redis_db.redis_client.get(key)
    if custom_limit:
        return int(custom_limit)
    return DEFAULT_LARGE_LIMIT if model_type == "large" else DEFAULT_SMALL_LIMIT

async def check_budget(ip: str, model_type: str, required_tokens: int = 0) -> None:
    """
    Check if the user has enough token budget left for the day.
    Raises HTTPException 429 if the budget is exceeded.
    """
    key = get_budget_key(ip, model_type)
    
    current_usage = await redis_db.redis_client.get(key)
    current_usage = int(current_usage) if current_usage else 0
    
    daily_limit = await get_user_limit(ip, model_type)
    
    if current_usage + required_tokens > daily_limit:
        raise HTTPException(
            status_code=429,
            detail=f"Your daily token limit for {model_type} model is over."
        )

async def check_budget_boolean(ip: str, model_type: str, required_tokens: int = 0) -> bool:
    """Non-raising version of check_budget. Returns True if OK, False if exceeded."""
    key = get_budget_key(ip, model_type)
    
    current_usage = await redis_db.redis_client.get(key)
    current_usage = int(current_usage) if current_usage else 0
    
    daily_limit = await get_user_limit(ip, model_type)
    
    return current_usage + required_tokens <= daily_limit

async def consume_budget(ip: str, model_type: str, used_tokens: int) -> None:
    """
    Increment the user's daily token usage by `used_tokens`
    and ensure the key expires in 24 hours.
    """
    if used_tokens <= 0:
        return
        
    key = get_budget_key(ip, model_type)
    
    # Increment usage
    new_usage = await redis_db.redis_client.incrby(key, used_tokens)
    
    # Set expiration if it's the first time setting this key
    # 86400 seconds = 24 hours
    if new_usage == used_tokens:
        await redis_db.redis_client.expire(key, 86400)

async def set_custom_limit(ip: str, model_type: str, new_limit: int) -> None:
    """Admin function: Set a custom token limit for a specific IP and model type for today only."""
    if new_limit < 0:
        raise ValueError("Limit must be positive")
    key = get_limit_key(ip, model_type)
    await redis_db.redis_client.set(key, new_limit, ex=86400)

async def get_current_usage(ip: str, date_str: str = None) -> dict:
    """Admin function: Get the daily usage and limit for a specific IP (both models)."""
    small_usage = await redis_db.redis_client.get(get_budget_key(ip, "small", date_str))
    large_usage = await redis_db.redis_client.get(get_budget_key(ip, "large", date_str))
    
    small_limit = await get_user_limit(ip, "small") # Assuming limit applies broadly
    large_limit = await get_user_limit(ip, "large")
    
    s_used = int(small_usage) if small_usage else 0
    l_used = int(large_usage) if large_usage else 0
    
    from app.core.config import small_model_config, large_model_config
    s_cost = small_model_config.cost_per_1m
    l_cost = large_model_config.cost_per_1m
    actual_cost = (s_used / 1_000_000 * s_cost) + (l_used / 1_000_000 * l_cost)
    
    return {
        "ip": ip,
        "small_model": {
            "used": s_used,
            "limit": small_limit,
            "remaining": small_limit - s_used
        },
        "large_model": {
            "used": l_used,
            "limit": large_limit,
            "remaining": large_limit - l_used
        },
        "actual_cost": actual_cost
    }

async def get_all_users_usage(date_str: str = None) -> list[dict]:
    """Admin function: Get usage for all IPs that have used tokens on the given date."""
    date_str = date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    pattern = f"tokens:*:{date_str}"
    
    keys = await redis_db.redis_client.keys(pattern)
    
    ips = set()
    for key in keys:
        key_str = key if isinstance(key, str) else key.decode("utf-8")
        parts = key_str.split(":")
        # Format is tokens:{ip}:{model_type}:{date} -> so len is 4
        if len(parts) >= 4:
            ips.add(parts[1])
            
    users = []
    for ip in ips:
        users.append(await get_current_usage(ip, date_str))
            
    return users
