import os 
from pathlib import Path 
from dotenv import load_dotenv
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import chromadb
from datetime import datetime

load_dotenv()

OPEN_API_KEY = os.getenv("OPEN_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = 'gpt-4o-mini'

BASE_DIR = Path(__file__).resolve().parent.parent
chroma_client = chromadb.PersistentClient(BASE_DIR / "data" / "vectorstore")
# ---------Chroma for database------------

# For calling colleciton
def get_collection(client):
    return client.get_or_create_collection(
        embedding_function = OpenAIEmbeddingFunction
        (
            api_key=OPEN_API_KEY,
            model_name=EMBEDDING_MODEL
        ),
        name="Way_to_Master_Chef",
        metadata=
        {
            "domain": "Culinary_Science",
            "corpus": "cookbooks",
            "description": "RAG corpus built from food science and cooking books",
            "created": str(datetime.now())
        },
        )

def doc_exists(collection, doc_id: str) -> bool:
    result = collection.get(
        where={"doc_id": doc_id}
    )
    return len(result.get("ids", [])) > 0
