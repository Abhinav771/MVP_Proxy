import httpx
import asyncio

async def test(prompt: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/chat",
            json={"prompt": prompt},
            timeout=30
        )
        print(f"Prompt   : {prompt}")
        print(f"Response : {response.text[:80]}...")
        print()

async def main():
    await test("What is the weather today?")   # ⚡ volatile
    await test("Explain JavaScript to me")     # 🟢 cache hit
    await test("What is FastAPI?")             # 🔴 cache miss

asyncio.run(main())