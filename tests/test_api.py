import requests
import time
import concurrent.futures

BASE_URL = "http://localhost:8000"

def print_header(title: str):
    print(f"\n{'-'*50}\n🚀 TESTING: {title}\n{'-'*50}")

def stream_chat(prompt: str):
    """Helper function to stream responses from the /chat endpoint."""
    print(f"User: {prompt}\nAI: ", end="", flush=True)
    
    response = requests.post(
        f"{BASE_URL}/chat",
        json={"prompt": prompt},
        stream=True
    )
    
    if response.status_code == 200:
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                print(chunk, end="", flush=True)
        print("\n")
    else:
        print(f"\n[ERROR] Status Code: {response.status_code}")
        print(response.json())
    
    return response.status_code

def test_health():
    print_header("Health Check")
    try:
        response = requests.get(f"{BASE_URL}/health")
        data = response.json()
        print(f"Status Code: {response.status_code}")
        print(f"Response: {data}")
        
        if data.get("redis") != "ok":
            print("⚠️ WARNING: Redis is not connected properly.")
    except Exception as e:
        print(f"❌ Failed to connect to server: {e}")

def test_volatility():
    print_header("Volatility Detection (Should bypass cache)")
    # Using keywords "today" and "weather"
    prompt = "What is the weather like today?"
    stream_chat(prompt)
    print("👉 Check your server logs! You should see: ⚡ VOLATILE")

def test_semantic_cache():
    print_header("Semantic Caching (Miss -> Hit)")
    
    # 1. First request (Should Miss)
    prompt_1 = "Explain quantum computing in one simple sentence."
    print("Request 1 (Expecting Cache Miss)...")
    stream_chat(prompt_1)
    print("👉 Check your server logs! You should see: 🔴 CACHE MISS")
    print("👉 Check your server logs! You should see: 💾 Stored in Pinecone")
    
    # Wait for Pinecone to index the new vector
    print("\nWaiting 5 seconds for Pinecone to index...")
    time.sleep(5)
    
    # 2. Second request (Should Hit)
    prompt_2 = "Tell me about quantum computing briefly."
    print("Request 2 (Expecting Cache Hit)...")
    stream_chat(prompt_2)
    print("👉 Check your server logs! You should see: 🟢 CACHE HIT")

def fire_request(i):
    """Fires a quick request and returns the status code."""
    response = requests.post(
        f"{BASE_URL}/chat",
        json={"prompt": f"Test prompt {i}"},
        stream=False # We don't care about reading the stream, just the status
    )
    return response.status_code

def test_rate_limiter():
    print_header("Rate Limiter (Max 10 requests / min)")
    print("Firing 12 rapid concurrent requests...")
    
    # Fire 12 requests simultaneously to trigger the Redis rate limit
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        results = list(executor.map(fire_request, range(12)))
    
    # Tally up the status codes
    successes = results.count(200)
    rate_limits = results.count(429)
    
    print(f"✅ Successful requests (200 OK): {successes}")
    print(f"🚫 Rate limited requests (429 Too Many Requests): {rate_limits}")
    
    if rate_limits > 0:
        print("🎉 Rate limiter is working perfectly!")
    else:
        print("⚠️ Rate limiter did not catch the excess requests. Check your Redis connection and MAX_REQUESTS_PER_MINUTE variable.")

if __name__ == "__main__":
    print("Make sure your FastAPI server is running (uvicorn app.main:app) before proceeding.")
    time.sleep(2)
    
    test_health()
    test_volatility()
    test_semantic_cache()
    test_rate_limiter()
    
    print("\n✅ All tests finished!")