import chromadb
from config import get_collection
from openai import OpenAI 
from config import OPEN_API_KEY, LLM_MODEL

chroma_client = chromadb.PersistentClient(path="./data/vectorstore")
collection = get_collection(chroma_client)

def query(text, collection):
    return collection.query(
        query_texts = [text],
        n_results = 2
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

def generate(documents, text):
    client = OpenAI()

    system_prompt = """
        You are a document-based question answering assistant.

    Rules:
    - Use ONLY the information provided in the Context to answer the question.
    - Do NOT use outside knowledge or make assumptions.
    - If the answer is not clearly present in the Context, say:
    "I do not know based on the provided context."
    - Be clear, concise, and factual.


    """

    user_prompt = f"""

    Context:
    {documents}


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
    answer = generate(documents, question)

    docs = results["documents"][0]
    metadata = results["metadatas"][0]

    sources = []
    for i in range(len(docs)):
        sources.append({"text":docs[i], "metadata":metadata[i]})

    return {
        "answer": answer, 
        "sources": sources
    }

