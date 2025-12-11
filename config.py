import os 
from dotenv import load_dotenv
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import chromadb
from datetime import datetime

load_dotenv()
chroma_client = chromadb.Client()

OPEN_API_KEY = os.getenv("OPEN_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = 'gpt-4o-mini'


# ---------Chroma for database------------

# For calling colleciton
def get_collection(client):
    return client.get_or_create_collection(
        embedding_function = OpenAIEmbeddingFunction
        (
            api_key=OPEN_API_KEY,
            model_name=EMBEDDING_MODEL
        ),
        name="Document_Q_A_Study_Assistant",
        metadata=
        {
            "source": "study",
            "topic": "lecture_notes",
            "description": "Personality type about INFP",
            "created": str(datetime.now())
        },
        )
