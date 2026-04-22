from rag import answer_question
from ingest import run_ingest
from config import chroma_client
from fastapi import FastAPI, UploadFile, HTTPException, File
from recipe.pipeline import generate_recipe
from pydantic import BaseModel 
from pathlib import Path 
from typing import Optional, Dict, Any 
from recipe.assemble import pipeline
import chromadb
import shutil
import hashlib

BASE_DIR = Path(__file__).resolve().parent.parent
chroma_client = chromadb.PersistentClient(BASE_DIR / "data" / "vectorstore")

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

class RecipeGenerateRequest(BaseModel):
    message: str
    overrides: Optional[Dict[str, Any]] = None 
    # debug: bool = False



@app.get("/")
def root():
    return {"Hello":"World"}

# Ask gerneal Question that is in RAG (PDF)
@app.post("/question")
def question(req: QuestionRequest):
    q = req.question
    results = answer_question(q)
    return results


# For User upload (PDF)
@app.post("/upload")
def upload(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Please upload a PDF")

    save_dir = BASE_DIR / "data" / "raw"
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / file.filename
    with save_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    doc_id = hashlib.sha256(save_path.read_bytes()).hexdigest()

    pages, chunks = run_ingest(chroma_client, save_path, doc_id)

    if chunks == 0: 
        status = "uploaded_no_text"
    else:
        status = "uploaded_and_ingested"

    return {"status": status, 
            "filename": file.filename, 
            "doc_id": doc_id,
            "stats": {
                "pages":pages,
                "chunks": chunks
            }}


# Receipe 
@app.post("/recipes/generate")
def genereate(req: RecipeGenerateRequest):
    return pipeline(
        req.message, 
        req.overrides)

