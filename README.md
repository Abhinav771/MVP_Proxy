# 🚀 MVP Proxy: Intelligent LLM API Gateway

MVP Proxy is a high-performance API Gateway designed to sit in front of your Large Language Models (LLMs). It drastically reduces API costs and latency through intelligent request routing and semantic caching, while providing a beautiful React dashboard to monitor your exact financial savings in real-time.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![React](https://img.shields.io/badge/React-18.2+-61DAFB.svg)

---

## ✨ Core Features

1. **🧠 RouteLLM (Model Routing)**
   Automatically analyzes incoming prompts and routes complex queries to a powerful "Large Model" and simpler queries to a cheaper/faster "Small Model". Drastically reduces costs without sacrificing response quality.
   
2. **📦 Semantic Caching**
   Instead of exact string matching, the proxy uses **Pinecone Vector Database** and **Spacy** to identify *semantically similar* prompts (e.g., "What is the capital of France?" vs "Tell me the French capital"). Cache hits bypass the expensive Large Model entirely.

3. **⏱️ Volatility Checking**
   Smart enough to recognize time-sensitive or volatile prompts (e.g., "What is the weather today?" or "What time is it?"). Volatile prompts automatically bypass the cache to ensure the user gets real-time, accurate data.

4. **💰 Token Budgets & Rate Limiting**
   Tracks exact token usage per user IP address via **Redis**. Administrators can set strict daily token budgets for both the Small and Large models on a per-user basis.

5. **📊 Interactive React Dashboard**
   A beautiful, real-time analytics dashboard built with Vite and Chart.js.
   - **Financial Tracking:** Calculates exact actual costs vs estimated savings down to the fraction of a cent.
   - **Historical Charts:** Interactive stacked bar charts showing your cost savings over a 7-day rolling window.
   - **Active Users:** See exactly which IPs are consuming your token budget in real-time.

---

## 🏗️ System Architecture

- **Backend:** FastAPI (Python), Redis (Token Tracking & Telemetry), Pinecone (Vector Search), Tiktoken (Token Counting).
- **Frontend:** React, Vite, Chart.js, Lucide React (Icons), Vanilla CSS (Custom Design System).

---

## ⚙️ Installation & Setup

### 1. Backend Setup (FastAPI)

Clone the repository and navigate to the project root:

```bash
cd MVP_Proxy
```

Create a virtual environment and install dependencies:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

pip install fastapi uvicorn redis pinecone-client spacy tiktoken python-dotenv httpx
python -m spacy download en_core_web_sm
```

### 2. Environment Variables

Create a `.env` file in the root directory and populate it with your API keys:

```env
# Vector DB & Embeddings
PINECONE_API_KEY=your_pinecone_key
GOOGLE_API_KEY=your_google_gemini_key

# Small Model (Cheap/Fast)
SMALL_MODEL_BASE_URL=https://api.groq.com/openai/v1
SMALL_MODEL_API_KEY=your_groq_key
SMALL_MODEL_NAME=llama-3.1-8b-instant
SMALL_MODEL_COST_PER_1M=0.05

# Large Model (Expensive/Powerful)
LARGE_MODEL_BASE_URL=https://api.groq.com/openai/v1
LARGE_MODEL_API_KEY=your_groq_key
LARGE_MODEL_NAME=llama-3.3-70b-versatile
LARGE_MODEL_COST_PER_1M=0.50
```

### 3. Frontend Setup (React Dashboard)

Open a new terminal and navigate to the dashboard folder:

```bash
cd dashboard
npm install
```

---

## 🚀 Running the Application

**Start the FastAPI Proxy (Terminal 1):**
```bash
venv\Scripts\uvicorn app.main:app --reload
```
*The proxy will be available at `http://127.0.0.1:8000/chat`.*

**Start the React Dashboard (Terminal 2):**
```bash
cd dashboard
npm run dev
```
*The dashboard will be available at `http://localhost:5173`.*

---

## 🧪 Testing

A robust test script is included to help you simulate traffic and watch the dashboard charts populate.

```bash
venv\Scripts\python test_proxy.py
```

The script automatically spoofs different IP addresses and sends a mix of easy, complex, volatile, and heavily cached prompts to ensure all routing and caching logic is functioning properly.

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
