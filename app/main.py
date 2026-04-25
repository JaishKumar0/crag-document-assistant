from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import os, shutil, uuid
from app.rag_pipeline1 import app as rag_app
from app.config import GITHUB_TOKEN, GITHUB_BASE_URL   
from app.vector_store import create_or_load_vs, delete_vs
#from app.rag_pipeline import app as rag_app

api = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@api.post("/upload")
async def upload(file: UploadFile = File(...)):
    allowed = {".pdf", ".txt", ".docx"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}. Allowed: {allowed}")

    file_id = str(uuid.uuid4())
    file_path = f"{UPLOAD_DIR}/{file_id}_{file.filename}"

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    create_or_load_vs(file_id, file_path)

    return {"file_id": file_id}


class Query(BaseModel):
    question: str
    file_id: str
    session_id: str


@api.post("/ask")
def ask(q: Query):
    try:
        
        vs = create_or_load_vs(q.file_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"file_id '{q.file_id}' not found. Upload the file first.")

   
    result = rag_app.invoke(
        {
            "question": q.question,
            "file_id": q.file_id  
        },
        config={"configurable": {"thread_id": q.session_id}}
    )

    return {"answer": result["answer"]}




@api.delete("/delete/{file_id}")
def delete(file_id: str):
    delete_vs(file_id)
    return {"message": "Deleted"}
