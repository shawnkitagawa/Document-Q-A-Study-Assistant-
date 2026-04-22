from pathlib import Path 
from pypdf import PdfReader
from config import OPEN_API_KEY, LLM_MODEL, get_collection, doc_exists
import chromadb
from utils import clean_text , chunk_text

BASE_DIR = Path(__file__).resolve().parent.parent

chroma_client = chromadb.PersistentClient(BASE_DIR / "data" / "vectorstore")


# Add the chunk data into the vector DB
def add_data(collection, ids, documents, metadatas):
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )

# Extract into chunks from the PDF, and write into Chroma in batches
def extract_chunks_from_pdf(pdf_file, collection, doc_id, flush_n=20):
    if doc_exists(collection, doc_id):
        print(f"Already ingested: {doc_id}")
        return

    ids, documents, metadatas = [], [], []
    count = 0
    total_chunks = 0 

    reader = PdfReader(str(pdf_file))
    for page_index, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        cleanText = clean_text(text)
        if not cleanText.strip():
            continue
        chunks = chunk_text(cleanText)
        total_chunks += len(chunks)

        for chunk_index, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_p{page_index+1}_c{chunk_index}"
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({
                "doc_id": doc_id,
                "source": pdf_file.name,
                "book": pdf_file.stem,
                "page": page_index + 1,
                "chunk": chunk_index
            })
            count += 1

            if count >= flush_n:
                add_data(collection, ids, documents, metadatas)
                ids.clear(); documents.clear(); metadatas.clear()
                count = 0

    if ids:
        add_data(collection, ids, documents, metadatas)

    return len(reader.pages), total_chunks




def run_ingest(chroma_client, pdf_file, doc_id):
    collection = get_collection(chroma_client)
    pages, chunks = extract_chunks_from_pdf(pdf_file, collection, doc_id, flush_n=20)

    return pages, chunks


