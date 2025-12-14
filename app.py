'''
Provide the user interface — the actual "assistant" people interact with.


'''
from rag import answer_question

def get_question():
    question = input("Ask any question related to the PDF documents")
    return question 




def main():
    question = get_question()
    results = answer_question(question)

    print("\nANSWER:\n")
    print(results["answer"])

    print("\nSource:\n")
    for source in results["sources"]:
        print(source["metadata"])



# call question 



if __name__ == "__main__":
    main()
