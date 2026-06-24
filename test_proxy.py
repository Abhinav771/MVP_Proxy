import requests
import time

URL = "http://127.0.0.1:8000/chat"

prompts = [
    # 1. Easy prompt (Should route to small)
    "What is the capital of France?",
    
    # 2. Easy prompt (Cache hit test - exact same meaning)
    "Can you tell me the capital of France?",
    
    # 3. Volatile prompt (Should skip cache)
    "What is the exact time right now?",
    
    # 4. Volatile prompt (Should skip cache)
    "What is the weather like today?",
    
    # 5. Coding prompt (Should route to large due to code keyword or ML)
    "Write a Python script to reverse a linked list.",
    
    # 6. Coding prompt (Cache hit for large model)
    "How do I reverse a linked list in Python?",
    
    # 7. Math prompt (Might route to large depending on ML router)
    "Calculate the square root of 144.",
    
    # 8. Another easy prompt to pump up small model usage
    "Name three colors of the rainbow.",
    
    # 9. Complex reasoning (Should route to large)
    "Explain quantum entanglement to a 5-year-old using an analogy about apples.",
    
    # 10. Cache hit for complex reasoning
    "Use an apple analogy to explain quantum entanglement to a child."
]

import random

mock_ips = [
    "192.168.1.100",
    "10.0.0.5",
    "172.16.254.1",
    "203.0.113.42",
    "198.51.100.7"
]

print("🚀 Starting Proxy Test Script with Simulated IPs\n")

for i, prompt in enumerate(prompts, 1):
    simulated_ip = random.choice(mock_ips)
    print(f"[{i}/10] Sending prompt from IP {simulated_ip}: '{prompt}'")
    
    payload = {"prompt": prompt}
    headers = {"X-Forwarded-For": simulated_ip}
    
    try:
        start_time = time.time()
        # Stream=False here so it waits and captures the whole response text for printing
        response = requests.post(URL, json=payload, headers=headers)
        
        if response.status_code == 429:
            print("❌ Error 429: Rate limit or Daily Token Budget exceeded!\n")
        else:
            answer = response.text
            elapsed = time.time() - start_time
            # Print a snippet of the answer
            print(f"✅ Response ({elapsed:.2f}s): {answer.strip()[:100]}...\n")
            
    except Exception as e:
        print(f"❌ Connection Error: {e}\n")
    
    # Sleep to avoid hitting local rate limits instantly
    time.sleep(2)

print("🎉 Test complete! Check your React Dashboard to see the new metrics!")
