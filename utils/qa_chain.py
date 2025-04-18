import os
from .response_formatter import format_response

def qa_search(query, vectorstore):
    """
    Answer questions about the codebase using the vectorstore and LLM.
    
    Args:
        query (str): The question to answer
        vectorstore: The vectorstore containing the indexed documents
        
    Returns:
        str: The answer to the question with sources
    """
    from langchain.chains import RetrievalQA
    from utils.models import initialize_llm
    
    # Initialize LLM
    llm = initialize_llm()
    
    # Create QA chain with sources
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        ),
        return_source_documents=True
    )
    
    # Run the chain
    try:
        result = qa_chain.invoke(query)
        answer = result["result"]
        source_documents = result["source_documents"]
        
        # Format the response using common formatter
        return format_response(answer, source_documents)
        
    except Exception as e:
        return f"Error querying the model: {str(e)}"

def search_code(query, search_type="qa", k=5, vectorstore=None):
    """
    Search the code repository using either QA or similarity search
    
    Args:
        query (str): The search query
        search_type (str): 'qa' for question answering or 'similarity' for code retrieval
        k (int): Number of results to return for similarity search
        vectorstore: Optional vectorstore instance (will load if not provided)
    
    Returns:
        str: Formatted search results
    """
    # Get vectorstore if not provided
    if vectorstore is None:
        from utils.vectorstore import get_vectorstore
        vectorstore = get_vectorstore()
    
    if vectorstore is None:
        return "Error: No vectorstore found. Please index your code repository first."
    
    # Process based on search type
    if search_type == "qa":
        # Use QA search for answering questions
        return qa_search(query, vectorstore)
    else:
        # Use similarity search for retrieving relevant code snippets
        try:
            results = vectorstore.similarity_search(query, k=k)
            # Format the similarity search results using the same formatter
            return format_response(f"Results for: {query}", results)
        except Exception as e:
            return f"Error during search: {str(e)}"

