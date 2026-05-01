import os, shutil
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from app.file_loader import load_document
from app.config import GITHUB_TOKEN, GITHUB_BASE_URL

VECTOR_DIR = "vectorstores"
os.makedirs(VECTOR_DIR, exist_ok=True)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=GITHUB_TOKEN,
    base_url=GITHUB_BASE_URL,
    chunk_size=100
)

def get_vs_path(file_id):
    return f"{VECTOR_DIR}/{file_id}"

def create_or_load_vs(file_id, file_path=None):
    path = get_vs_path(file_id)

    if os.path.exists(path):
        return FAISS.load_local(
            path,
            embeddings,
            allow_dangerous_deserialization=True
        )

   
    if file_path is None:
        raise FileNotFoundError(f"No vector store found for file_id='{file_id}' and no file_path provided.")

    docs = load_document(file_path)

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150
    ).split_documents(docs)

    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(path)

    return vs

def delete_vs(file_id):
    path = get_vs_path(file_id)
    if os.path.exists(path):
        shutil.rmtree(path)
