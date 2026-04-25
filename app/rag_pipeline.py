import re
import operator
import sqlite3
from typing import TypedDict, List, Annotated
from pydantic import BaseModel

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from app.config import GITHUB_TOKEN, GITHUB_BASE_URL
from app.vector_store import create_or_load_vs
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

# -----------------------------
# Configuration & LLM
# -----------------------------
# Using GitHub Models to share quota with embeddings and avoid versioning errors
llm = ChatOpenAI(
    model="gpt-4o-mini",  
    api_key=GITHUB_TOKEN,
    base_url=GITHUB_BASE_URL,
    max_retries=5
)

tavily = TavilySearchResults(max_results=5)

UPPER_TH = 0.7
LOWER_TH = 0.3

# -----------------------------
# State
# -----------------------------
class State(TypedDict, total=False):
    question: str
    file_id: str
    
    docs: List[Document]
    good_docs: List[Document]
    verdict: str
    reason: str
    strips: List[str]
    kept_strips: List[str]
    refined_context: str
    web_query: str
    web_docs: List[Document]
    answer: str
    messages: Annotated[List[str], operator.add]

# -----------------------------
# Nodes & Chains
# -----------------------------

def retrieve_node(state: State):
    q = state['question']
    file_id = state['file_id']
    
    # Load the vector store dynamically for the specific file
    vs = create_or_load_vs(file_id)
    retriever = vs.as_retriever(search_kwargs={"k": 4})
    
    result = retriever.invoke(q) 
    return {'docs': result}

# 1. Evaluate Docs Node
class evalDocsScore(BaseModel):
    score: float
    reason: str

doc_eval_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a strict retrieval evaluator for RAG.\n"
     "You will be given ONE retrieved chunk and a question.\n"
     "Return a relevance score in [0.0, 1.0].\n"
     "- 1.0: chunk alone is sufficient to answer fully/mostly\n"
     "- 0.0: chunk is irrelevant\n"
     "Be conservative with high scores.\n"
     "Also return a short reason.\n"),
    ("human", "Question: {question}\n\nChunk:\n{chunk}"),
])

doc_eval_chain = doc_eval_prompt | llm.with_structured_output(evalDocsScore)

def eval_docs_node(state: State):
    q = state['question']
    scores: List[float] = []
    good: List[Document] = []

    for doc in state.get('docs', []):
        out = doc_eval_chain.invoke({'question': q, 'chunk': doc.page_content})
        scores.append(out.score)
        if out.score > LOWER_TH:
            good.append(doc)

    if any(score > UPPER_TH for score in scores):
        return {'good_docs': good, 'verdict': 'CORRECT', 'reason': f'chunk score > {UPPER_TH}'}      
    
    if len(scores) > 0 and all(score < LOWER_TH for score in scores):
        return {'good_docs': [], 'verdict': 'INCORRECT', 'reason': f'all retrieval chunks score < {LOWER_TH}'}
        
    return {'good_docs': good, 'verdict': 'AMBIGUOUS', 'reason': f'No chunks score > {UPPER_TH}, but not all score < {LOWER_TH}.'}

# 2. Refine Node
def refine_node(state: State):
    verdict = state.get("verdict")
    
    if verdict == "CORRECT":
        docs_to_use = state.get("good_docs", [])
    elif verdict == "INCORRECT":
        docs_to_use = state.get("web_docs", [])
    else:  # AMBIGUOUS
        docs_to_use = state.get("good_docs", []) + state.get("web_docs", [])

    # FIX: Merging documents directly instead of looping sentence-by-sentence to avoid 429 Rate Limits
    refined_context = "\n\n".join(d.page_content for d in docs_to_use).strip()

    return {"refined_context": refined_context}

# 3. Rewrite & Web Search Nodes
class WebQuery(BaseModel):
    query: str

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "Rewrite the user question into a web search query composed of keywords.\n"
     "Rules:\n"
     "- Keep it short (6–14 words).\n"
     "- If the question implies recency, add a constraint like (last 30 days).\n"
     "- Do NOT answer the question."),
    ("human", "Question: {question}"),
])

rewrite_chain = rewrite_prompt | llm.with_structured_output(WebQuery)

def rewrite_query_node(state: State):
    out = rewrite_chain.invoke({"question": state["question"]})
    return {"web_query": out.query}

def web_search_node(state: State):
    q = state.get("web_query") or state["question"]
    results = tavily.invoke({"query": q})
    web_docs: List[Document] = []
    
    for r in results or []:
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "") or r.get("snippet", "")
        text = f"TITLE: {title}\nURL: {url}\nCONTENT:\n{content}"
        web_docs.append(Document(page_content=text, metadata={"url": url, "title": title}))
        
    return {"web_docs": web_docs}

# 4. Generate Node
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a helpful ML tutor. Answer ONLY using the provided context.\n"
     "If the context is empty or insufficient, say: 'I don't know.'\n"
     "Use context + history."),
    ("human", "History:\n{history}\n\nQuestion: {question}\n\nContext:\n{context}")
])

def generate_node(state: State):
    history = "\n".join(state.get("messages", [])[-5:]) # Pull last 5 turns from memory
    
    out = (answer_prompt | llm).invoke({
        "question": state["question"], 
        "context": state.get("refined_context", ""),
        "history": history
    })
    
    return {
        "answer": out.content,
        "messages": [
            f"User: {state['question']}",
            f"AI: {out.content}"
        ]
    }

# -----------------------------
# Graph Routing Logic
# -----------------------------
def route_after_eval(state: State) -> str:
    if state.get("verdict") == "CORRECT":
        return "refine"
    else:
        return "rewrite_query"

# -----------------------------
# Graph Build
# -----------------------------
g = StateGraph(State)

g.add_node("retrieve", retrieve_node)
g.add_node("eval_each_doc", eval_docs_node)
g.add_node("rewrite_query", rewrite_query_node)
g.add_node("web_search", web_search_node)
g.add_node("refine", refine_node)
g.add_node("generate", generate_node)

g.add_edge(START, "retrieve")
g.add_edge("retrieve", "eval_each_doc")

g.add_conditional_edges(
    "eval_each_doc",
    route_after_eval,
    {
        "refine": "refine",
        "rewrite_query": "rewrite_query",
    },
)

g.add_edge("rewrite_query", "web_search")
g.add_edge("web_search", "refine")
g.add_edge("refine", "generate")
g.add_edge("generate", END)

# Persistent Memory Initialization setup using sqlite3 connection
conn = sqlite3.connect("memory.db", check_same_thread=False)
memory = SqliteSaver(conn)

app = g.compile(checkpointer=memory)