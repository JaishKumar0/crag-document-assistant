# 🧠 Corrective RAG (CRAG) Document Assistant

An advanced **self-correcting Retrieval-Augmented Generation (RAG) system** that reduces hallucinations by validating retrieved context and dynamically deciding when to search the web.

---

## 🚀 Features

* 🔁 **Self-Correcting Pipeline** using LangGraph
* ✍️ **Query Rewriting** for better retrieval
* 📊 **Document Relevance Grading**
* 🌐 **Web Search Fallback (Tavily API)**
* 🧠 **Multi-step Reasoning with LLMs**
* ⚡ FastAPI backend (ready for deployment)

---

## 🏗️ Architecture

```text
User Query → Rewrite → Retrieve → Grade Docs
        ↓                    ↓
     Good Docs           Bad Docs
        ↓                    ↓
    Generate Answer     Web Search → Generate
```

---

## 🛠️ Tech Stack

* **LLM Frameworks:** LangChain, LangGraph
* **Vector DB:** ChromaDB
* **APIs:** Tavily
* **Backend:** FastAPI
* **Language:** Python

---

## 📦 Installation

```bash
git clone https://github.com/JaishKumar0/crag-document-assistant
cd crag-document-assistant
pip install -r requirements.txt
```

---

## ▶️ Run Locally

```bash
uvicorn main:app --reload
```

---

## 📊 Results

* Reduced hallucination cases by ~25–30% (manual evaluation)
* Improved response relevance using dynamic routing

---

## 📌 Future Improvements

* Add UI (React / Streamlit)
* Deploy on Render
* Add memory support

---

## 👤 Author

Jaish Kumar
LinkedIn: https://www.linkedin.com/in/jaish-kumar-2b420425b
