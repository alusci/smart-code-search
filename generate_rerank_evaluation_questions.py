import json
from generate_evaluation_questions import generate_rag_answers
from utils.vectorstore import get_vectorstore

NUM_NEIGHBORS = 5
NUM_TOP_K = 20

def main():
    # Load evaluation questions and answers
    with open("data/evaluation_questions.json", "r") as fp:
        questions = json.load(fp)

    vectorstore = get_vectorstore()
    if vectorstore is None:
        print("Error: No vectorstore found. Please run document_indexer.py first.")
        return

    generate_rag_answers(
        questions, vectorstore, k=NUM_NEIGHBORS, top_k=NUM_TOP_K, rerank=True
    )

    # Save questions for later evaluation
    with open("data/rerank_evaluation_questions.json", "w") as f:
        json.dump(questions, f, indent=2)
        
    print("Rerank evaluation questions saved to rerank_evaluation_questions.json")

    

if __name__ == "__main__":
    main()