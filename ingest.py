'''
Load PDFs

Extract text

Split text into chunks

Convert chunks into embeddings

Save embeddings + metadata into a vector database

'''
from pathlib import Path 
from pypdf import PdfReader
from config import OPEN_API_KEY, LLM_MODEL, get_collection
import chromadb
from utils import clean_text , chunk_text
# import os
# global variable 
chroma_client = chromadb.PersistentClient(path="./data/vectorstore")

# Grab the PDF 
def ingest_pdf(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    return pdf_reader

# Extract into chunks from the PDF
def extract_chunks_from_pdf(pdf_reader, pdf_path ):
    # Divide PDF into chuns of text 200-300 words
    text = ""
    ids = []
    documents = []
    metadata = []
    current_id = ""
    # Loop every pages 
    for i in range(len(pdf_reader.pages)):
        text = pdf_reader.pages[i].extract_text() or ""

        cleanText = clean_text(text)
        chunks = chunk_text(cleanText)
        for j in range(len(chunks)):
            metadata.append({"file": str(pdf_path), "page":i, "chunk":j})

            current_id = f"id{i}_c{j}"
            ids.append(current_id)
            documents.append(chunks[j])

    return ids, documents, metadata


# Add the chunk data into the vector DB 
def add_data(collection, ids, documents, metadata):
    #2. add the data
    # Must provide unique string IDs for the documents
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadata
    )


# 1. Create/ Get the collection 
collection = get_collection(chroma_client)
pdf_path = "Document-Q-A-Study-Assistant-/data/raw/sample_lecture_1.pdf"

#2. Grab the data (PDF)
pdf_data = ingest_pdf(pdf_path)

#3.  break data into chunks 
ids, documents , metadata = extract_chunks_from_pdf(pdf_data, pdf_path)

#4. add data to the collection 
add_data(collection, ids, documents, metadata)





