# 🧠 AI RAG Assistant (ML Tutor)

An advanced Retrieval-Augmented Generation (RAG) system that answers machine learning questions using:

- 📚 PDF textbooks
- 🌐 Web search (fallback)
- 🧠 LLM reasoning (LangGraph pipeline)

---

## 🚀 Features

- Multi-document RAG pipeline
- Automatic relevance evaluation
- Web search fallback (Tavily)
- Context refinement using LLM
- FastAPI backend
- Simple frontend UI

---

## 🛠️ Technology Stack

 Backend Framework: FastAPI, Uvicorn

AI & Orchestration: LangChain, LangGraph

Vector Database: FAISS (CPU)

Models: OpenAI (via GitHub Models API)

Tools: Tavily Search API

Frontend: Streamlit

---

## ⚙️ Setup

```bash
git clone https://github.com/jaish/ai-rag-app.git
cd ai-rag-app
pip install -r requirements.txt