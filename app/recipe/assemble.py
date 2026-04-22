# LLM only presentation layer 
from .normalize import normalize_request
from .decision_keys import normalize_vars, build_queue, decision_key
from .technique import save_technique_plan
from config import get_collection
from pathlib import Path 
import chromadb



BASE_DIR = Path(__file__).resolve().parent.parent.parent
chroma_client = chromadb.PersistentClient(BASE_DIR / "data" / "vectorstore")
collection = get_collection(chroma_client)

# overrides is an extra instructions from the user to add what the user types 
def pipeline(question, overrides):
    
    '''
    CORE RESPONSIBILITY

    Given a single user message, decide:

    Whether the message is a recipe-related request

    Whether the intent is clear enough to proceed

    What explicit constraints, preferences, and requirements are present

    You are extracting signals, not completing missing data.
    
    '''
    # The LLM creates a normalize request but if overrides information is inserted then priortize the overrides
    normalize_requests = normalize_request(overrides, question)

    print(normalize_requests)

    # cleaning the normalize request any None value, empty string are dropped 
    clean_normalize = normalize_vars(normalize_requests)

    print(clean_normalize)
    # Based on the cleaned normalize request we pick the most optimal questions to ask pure LLM

    # What is stable question LLM only for?
    #  - It is incharge of understanding the user's intent and priorites
    # Look at the normalize intent and figures out the best approach logically not fact checking 

    # What is Rag (Retrieval + LLM) for?
    # In charge of checking facts and grouding in real sources 


    # add stable question to the queue based on the normalize request 
    queue = build_queue(normalize_requests)

    print(queue)

    # Output a list of dictionaries

    '''
    {
    "decision_key": key,
    "stable_questions": stable_questions,
    "rag_queries": rag_queries
        }
    '''
    steps = decision_key(queue, clean_normalize)

    print(steps)

    plan = save_technique_plan(steps, clean_normalize,collection)

    print(plan)

    return plan 


