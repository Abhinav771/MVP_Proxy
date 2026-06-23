from app.core.config import nlp

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
        print(f"⚡ VOLATILE (keyword) | prompt: '{prompt}'")
        return True

    doc = nlp(prompt)
    for ent in doc.ents:
        if ent.label_ in VOLATILE_ENTITIES:
            print(f"⚡ VOLATILE (NER: {ent.label_}={ent.text}) | prompt: '{prompt}'")
            return True

    return False