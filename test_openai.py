from openai import OpenAI 
from config import OPEN_API_KEY, LLM_MODEL 
import chromadb

chroma_client = chromadb.Client()

# collection store all data like embedding, documents and additional metdata
collection = chroma_client.create_collection(name="my_collection")


# Must provide unique string IDs for the documents
collection.add(
    ids=["id1", "id2"],
    documents=[
        "This is a document about pineapple",
        "This is a document about oranges"
    ]
)


results = collection.query(
    query_texts=["This is a query document about hawaii"], # Chroma will embed this for you
    n_results=2 # how many results to return
)
print(results)



# client = OpenAI(api_key=OPEN_API_KEY)
# # client = OpenAI()

# response = client.responses.create(
#     model=LLM_MODEL,
#     input="write a personality traits of INFP girls in Galcia Spain easy way and tell me what means when she suddenly dissapear"
# )


# print(response.output_text)