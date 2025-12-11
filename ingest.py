'''
Load PDFs

Extract text

Split text into chunks

Convert chunks into embeddings

Save embeddings + metadata into a vector database

'''

from pathlib import Path 
from pypdf import PdfReader
from openai import OpenAI 
from config import OPEN_API_KEY, LLM_MODEL, get_collection
import chromadb
from utils import clean_text , chunk_text
import os

chroma_client = chromadb.Client()

#-------------------------------------------------------------

# Grab the PDF 
pdf_path = (
    "Document-Q-A-Study-Assistant-/data/raw/sample_lecture_1.pdf"
)
pdf_reader = PdfReader(pdf_path)

#-------------------------------------------------------------

# Divide PDF into chuns of text 200-300 words
text = ""
for i in range(len(pdf_reader.pages)):
    text += pdf_reader.pages[i].extract_text()

cleanText = clean_text(text)
chunks = chunk_text(cleanText)

#---------------------------------------------------------------



# vector database
connection = get_collection(chroma_client)
print(connection)