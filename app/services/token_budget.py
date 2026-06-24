import tiktoken
from datetime import datetime, timezone
from fastapi import HTTPException
from app.core import redis_db

# Default daily limit for a user (employee)
DEFAULT_DAILY_LIMIT = 1_000_000

# Initialize the tokenizer
# cl100k_base is fast and serves as a good proxy for most models including Llama.
encoder = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """Return the number of tokens in a string."""
    if not text:
        return 0
    return len(encoder.encode(text))

def get_budget_key(ip: str) -> str:
    """Generate the daily token usage key for a given IP."""
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"tokens:{ip}:{date_str}"

def get_limit_key(ip: str) -> str:
    """Generate the daily custom limit key for a given IP."""
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"limit:{ip}:{date_str}"

async def get_user_limit(ip: str) -> int:
    """
    Fetch the custom limit for the user from Redis if it exists.
    Otherwise, return the DEFAULT_DAILY_LIMIT.
    """
    key = get_limit_key(ip)
    custom_limit = await redis_db.redis_client.get(key)
    if custom_limit:
        return int(custom_limit)
    return DEFAULT_DAILY_LIMIT

async def check_budget(ip: str, required_tokens: int = 0) -> None:
    """
    Check if the user has enough token budget left for the day.
    Raises HTTPException 429 if the budget is exceeded.
    """
    key = get_budget_key(ip)
    
    current_usage = await redis_db.redis_client.get(key)
    current_usage = int(current_usage) if current_usage else 0
    
    daily_limit = await get_user_limit(ip)
    
    if current_usage + required_tokens > daily_limit:
        raise HTTPException(
            status_code=429,
            detail="Your daily token limit is over."
        )

async def consume_budget(ip: str, used_tokens: int) -> None:
    """
    Increment the user's daily token usage by `used_tokens`
    and ensure the key expires in 24 hours.
    """
    if used_tokens <= 0:
        return
        
    key = get_budget_key(ip)
    
    # Increment usage
    new_usage = await redis_db.redis_client.incrby(key, used_tokens)
    
    # Set expiration if it's the first time setting this key
    # 86400 seconds = 24 hours
    if new_usage == used_tokens:
        await redis_db.redis_client.expire(key, 86400)

async def set_custom_limit(ip: str, new_limit: int) -> None:
    """Admin function: Set a custom token limit for a specific IP for today only."""
    if new_limit < 0:
        raise ValueError("Limit must be positive")
    key = get_limit_key(ip)
    await redis_db.redis_client.set(key, new_limit, ex=86400)

async def get_current_usage(ip: str) -> dict:
    """Admin function: Get the current daily usage and limit for a specific IP."""
    key = get_budget_key(ip)
    usage = await redis_db.redis_client.get(key)
    limit = await get_user_limit(ip)
    return {
        "ip": ip,
        "tokens_used_today": int(usage) if usage else 0,
        "daily_limit": limit,
        "remaining": limit - (int(usage) if usage else 0)
    }

async def get_all_users_usage() -> list[dict]:
    """Admin function: Get usage for all IPs that have used tokens today."""
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    pattern = f"tokens:*:{date_str}"
    
    keys = await redis_db.redis_client.keys(pattern)
    
    users = []
    for key in keys:
        key_str = key if isinstance(key, str) else key.decode("utf-8")
        parts = key_str.split(":")
        if len(parts) >= 3:
            # Reconstruct the IP (in case IP is IPv6, though splitting by ':' makes IPv6 tricky. 
            # We assume IPv4 for MVP, or just extract the substring between 'tokens:' and ':{date_str}')
            # A safer way to extract the IP:
            prefix = "tokens:"
            suffix = f":{date_str}"
            if key_str.startswith(prefix) and key_str.endswith(suffix):
                ip = key_str[len(prefix):-len(suffix)]
                usage_data = await get_current_usage(ip)
                users.append(usage_data)
            
    return users
