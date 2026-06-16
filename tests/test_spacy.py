import spacy
import httpx
import asyncio
nlp = spacy.load("en_core_web_sm")

VOLATILE_KEYWORDS = {
    "now", "today", "tonight", "currently", "current",
    "live", "latest", "right now", "at the moment",
    "breaking", "trending", "real-time", "realtime",
    "this week", "this month", "yesterday", "tomorrow",
    "price", "stock", "weather", "score", "news"
}

VOLATILE_ENTITIES = {"DATE", "TIME"}



def is_volatile(prompt: str) -> bool:
    prompt_lower = prompt.lower()
    if any(keyword in prompt_lower for keyword in VOLATILE_KEYWORDS):
        return True
    doc = nlp(prompt)
    for ent in doc.ents:
        if ent.label_ in VOLATILE_ENTITIES:
            return True
    return False

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