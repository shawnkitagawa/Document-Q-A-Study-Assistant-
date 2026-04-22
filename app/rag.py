import chromadb
from config import get_collection
from pathlib import Path 
from openai import OpenAI 
from config import OPEN_API_KEY, LLM_MODEL

BASE_DIR = Path(__file__).resolve().parent.parent
chroma_client = chromadb.PersistentClient(BASE_DIR / "data" / "vectorstore")
collection = get_collection(chroma_client)

def query(text, collection):
    return collection.query(
        query_texts = [text],
        n_results = 4
    )


# 2. build_context() to make it communicateable using the query for the LLM's end 
def build_context(doc):
    context = ""
    chunks = doc["documents"][0]

    for i in range(len(chunks)):
        context += f"[Source {i+1}]\n"
        context += chunks[i]
        context += "\n\n"
    return context

def generate(documents, text, system_prompt):
    client = OpenAI()


    if documents: 
        user_prompt = f"""

        Context:
        {documents}


        Question:
        {text}

        """
    else:
        user_prompt = f"""
        Question:
        {text}
        """
    

    response = client.chat.completions.create(
        model = LLM_MODEL,
        messages = [
            {"role":"system", "content": system_prompt},
            {"role":"user", "content":user_prompt}
        ]
    )

    # print("\n\n---------------------\n\n")
    # print("ANSWER FROM OPENAI")

    return response.choices[0].message.content
    # print(response.choices)

    
def answer_question(question):
    # query the text and the data from the database to calcualte the difference based on the vector 
    results = query(question, collection)

    # search the sources that matched the closest vector distance
    documents = build_context(results)

    # generate an answer based on the context
    system_prompt = """
        You are a document-based question answering assistant.

    Rules:
    Use the provided context to answer the question.
    You may synthesize information across multiple passages.
    If the context is related but does not give a direct answer,
    infer the best possible answer from the evidence.
    If the context is completely unrelated, say "I do not know".

    After answering, include 1–3 short quotes (max 25 words each) from the retrieved context that best support the answer.


    """

    answer = generate(documents, question, system_prompt)

    docs = results["documents"][0]
    metadata = results["metadatas"][0]

    sources = []
    for i in range(len(docs)):
        sources.append({"text":docs[i], "metadata":metadata[i]})

    return {
        "answer": answer, 
        "sources": sources
    }

