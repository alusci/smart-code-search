import random
import json
import os
from utils.vectorstore import get_vectorstore
from utils.models import initialize_embeddings, initialize_llm
from utils.qa_chain import qa_search
from tqdm import tqdm

NUM_SAMPLES = 10
NUM_NEIGHBORS = 5

def sample_text_chunks(vectorstore, num_samples=5, neighbors=5):
    """
    Sample random text chunks from the vectorstore along with their nearest neighbors.
    
    Args:
        vectorstore: The FAISS vectorstore
        num_samples: Number of sample groups to retrieve
        neighbors: Number of neighbors to retrieve for each sample
        
    Returns:
        list: List of dictionaries containing sample text and its neighbors
    """
    print(f"Sampling {num_samples} random text chunks with {neighbors} neighbors each...")
    
    # Get all document ids - handle both numerical and string ids
    try:
        # Try to get the document IDs more safely
        doc_ids = list(vectorstore.docstore._dict.keys())
        
        if not doc_ids:
            raise ValueError("No documents found in vectorstore")
            
        print(f"Found {len(doc_ids)} documents in vectorstore")
        
        if len(doc_ids) < num_samples:
            print(f"Warning: Requested {num_samples} samples, but only {len(doc_ids)} documents available")
            num_samples = len(doc_ids)
    
        # Sample random document ids
        sample_ids = random.sample(doc_ids, num_samples)
        
        samples = []
        for i, doc_id in enumerate(sample_ids):
            try:
                # Get the sample document
                sample_doc = vectorstore.docstore._dict[doc_id]
                sample_text = sample_doc.page_content
                sample_path = sample_doc.metadata.get('source', 'Unknown')
                
                print(f"Processing sample {i+1}/{num_samples}: {os.path.basename(sample_path)}")
                
                # Get embeddings for the sample
                embeddings = initialize_embeddings()
                sample_embedding = embeddings.embed_documents([sample_text])[0]
                
                # Find nearest neighbors
                neighbor_docs = vectorstore.similarity_search_by_vector(
                    sample_embedding, 
                    k=neighbors+1  # +1 because it will find itself
                )
                
                # Remove the sample itself from neighbors (if present)
                neighbor_docs = [doc for doc in neighbor_docs if doc.page_content != sample_text][:neighbors]
                
                # Create a sample entry with text and metadata
                sample_entry = {
                    'id': i,
                    'sample_text': sample_text,
                    'sample_path': sample_path,
                    'neighbors': [
                        {
                            'text': doc.page_content,
                            'path': doc.metadata.get('source', 'Unknown')
                        } for doc in neighbor_docs
                    ]
                }
                
                samples.append(sample_entry)
                
            except Exception as e:
                print(f"Error processing document ID {doc_id}: {e}")
                continue
                
        return samples
        
    except Exception as e:
        print(f"Error accessing vectorstore docstore: {e}")
        raise

def generate_evaluation_questions(samples, num_questions=2):
    """
    Generate evaluation questions based on the sampled text chunks and their neighbors.
    
    Args:
        samples: List of sample text chunks with neighbors
        num_questions: Number of questions to generate per sample
        
    Returns:
        list: List of dictionaries containing evaluation questions
    """
    
    llm = initialize_llm()
    questions = []
    
    for sample in tqdm(samples, desc="Generating questions"):
        
        # Define context for the LLM
        context = "\n\n".join([
            sample["neighbors"][i]['text'] for i in range(len(sample["neighbors"]))
        ])
        
        # Construct prompt for the LLM to generate questions
        prompt = f"""
        Based on the following code, generate {num_questions} specific questions that could be answered using this code and its context:
        
        ```
        {context}
        ```
        
        Use the formula: Con you find the answer to this question in the code?
        Question must be specific to the code and its context (do not extrapolate).

        Return only the questions, one per line, without numbering or any other text.
        Focus on questions that require understanding the functionality, purpose, or implementation details.
        """
        
        # Generate questions
        try:
            response = llm.invoke(prompt)
            generated_questions = [q.strip() for q in response.content.strip().split('\n') if q.strip()]
            
            # Limit to requested number of questions
            generated_questions = generated_questions[:num_questions]
            
            # Store both the sample and its neighbors for evaluation
            for q in generated_questions:
                questions.append({
                    'sample_id': sample['id'],
                    'question': q,
                    'source_path': sample['sample_path'],
                    'context': sample['neighbors']  # Include neighbors for context
                })
                
        except Exception as e:
            print(f"Error generating questions for sample {sample['id']}: {e}")
    
    return questions


def generate_evaluation_answers(questions):
    """
    Generate answers for the evaluation questions using the RAG pipeline.
    
    Args:
        questions: List of evaluation questions
        
    Returns:
        list: List of dictionaries containing questions and their generated answers
    """
    
    llm = initialize_llm()
    answers = []
    
    for question in tqdm(questions, desc="Generating answers"):
        
        context = "\n\n".join([
            question["context"][i]['text'] for i in range(len(question["context"]))
        ])
        
        prompt = f"""
        Based on the following code, answer the question:
        
        Question: {question['question']}
        
        Context: {context}
        
        Answer:
        """

        # Generate answer
        try:
            response = llm.invoke(prompt)
            answer_text = response.content.strip()
            
            # Store the question and its generated answer
            question["answer"] = answer_text
            
        except Exception as e:
            print(f"Error generating answer for question '{question['question']}': {e}")
    
    return answers


def generate_rag_answers(questions, vectorstore, k=5, top_k=20, rerank=False):
    """
    Generate answers for the evaluation questions using the RAG pipeline.
    
    Args:
        questions: List of evaluation questions
        vectorstore: The desired vectorstore
        k: Number of neighbors to retrieve
        top_k: Number of documents to retrieve for re-ranking
        rerank: Whether to use re-ranking for the answers
        
    Returns:
        list: List of dictionaries containing questions and their generated answers
    """
    
    # Placeholder for RAG answer generation logic
    # This function should be implemented to use the RAG pipeline
    
    for question in tqdm(questions, desc="Generating RAG answers"):
        # Placeholder for RAG answer generation
        answer, sources = qa_search(
            query=question["question"],
            vectorstore=vectorstore,
            k=k,
            top_k=top_k,
            return_formatted=False,
            rerank=rerank,
        )
        question["rag_answer"] = answer
        question["rag_context"] = [doc.page_content for doc in sources]


def main():
    """Main function to run the evaluation preparation"""
    # Get vectorstore
    vectorstore = get_vectorstore()
    
    if vectorstore is None:
        print("Error: No vectorstore found. Please run document_indexer.py first.")
        return
    
    try:
        # Sample text chunks with neighbors
        samples = sample_text_chunks(vectorstore, num_samples=NUM_SAMPLES, neighbors=NUM_NEIGHBORS)
        print(f"Successfully sampled {len(samples)} text chunks with neighbors")
        
        # Generate evaluation questions
        questions = generate_evaluation_questions(samples, num_questions=2)
        print(f"Generated {len(questions)} evaluation questions")

        # Generate answers for the questions
        generate_evaluation_answers(questions)
        print(f"Generated answers for {len(questions)} questions")

        # Generate RAG answers for the questions
        generate_rag_answers(questions, vectorstore, k=NUM_NEIGHBORS)
        print(f"Generated RAG answers for {len(questions)} questions")
        
        # Save questions for later evaluation
        os.makedirs("data", exist_ok=True)
        with open("data/evaluation_questions.json", "w") as f:
            json.dump(questions, f, indent=2)
        
        print("Evaluation questions saved to evaluation_questions.json")
        
        # Print sample questions
        print("\nSample questions:")
        for i, q in enumerate(questions[:5]):
            print(f"{i+1}. {q['question']}")
            print(f"   Source: {os.path.basename(q['source_path'])}")
            print()
        
    except Exception as e:
        print(f"Error during sampling: {e}")

if __name__ == "__main__":
    main()