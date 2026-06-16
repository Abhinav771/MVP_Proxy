import httpx
import asyncio

URL = "http://localhost:8000/chat"
TOTAL_REQUESTS = 15  # more than our limit of 10

async def send_request(client, i):
    try:
        response = await client.post(
            URL,
            json={"prompt": "say hi in one word"},
            timeout=30
        )
        if response.status_code == 429:
            print(f"Request {i:02d} → ❌ 429 BLOCKED — {response.json()['detail']}")
        else:
            # read first 40 chars of streamed response
            print(f"Request {i:02d} → ✅ 200 OK — {response.text[:40]}")
    except Exception as e:
        print(f"Request {i:02d} → 💥 Error: {e}")

async def main():
    async with httpx.AsyncClient() as client:
        tasks = [send_request(client, i+1) for i in range(TOTAL_REQUESTS)]
        await asyncio.gather(*tasks)  # fires all at once

if __name__ == "__main__":
    asyncio.run(main())