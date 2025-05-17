import os
from .response_formatter import format_response
from .reranker import create_mmr_retriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def qa_search(query, vectorstore, k=5, top_k=50, return_formatted=True, rerank=False):
    """
    Answer questions about the codebase using the vectorstore and LLM.
    
    Args:
        query (str): The question to answer
        vectorstore: The vectorstore containing the indexed documents
        k (int): Number of documents to retrieve
        top_k (int): Number of documents to retrieve for re-ranking
        return_formatted (bool): Whether to return the formatted response or (response, sources) tuple
        rerank (bool): Whether to use re-ranking for the retrieved documents
        
    Returns:
        str: The answer to the question with sources
    """
    from utils.models import initialize_llm
    
    # Initialize LLM
    llm = initialize_llm()
    
    # Create retriever
    if rerank:
        # Use MMR retriever if rerank is True
        retriever = create_mmr_retriever(vectorstore, k=k, top_k=top_k)
    else:
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
    
    # Define prompt template    
    template = """You are a helpful assistant that provides accurate information based on the given context.
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer the question based only on the provided context.
    If the context is incomplete or vague, do your best to respond using the available information. 
    If the question appears incomplete (e.g. starts with “Can you find”, “Do you”, or cuts off unexpectedly), politely inform the user that the question seems incomplete and ask them to rephrase. 
    If a full answer isn’t possible, clearly state what’s missing and politely ask the user for clarification. 
    Do not invent or assume facts that are not explicitly present in the context. 
    Keep responses brief and focused (no more than 500 words). 
    Use bullet points and code samples whenever appropriate to improve clarity. 
    Avoid repetition and long paragraphs. 
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create document chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create retrieval chain
    retrieval_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | document_chain
        | StrOutputParser()
    )
    
    # Run the chain
    try:
        # Get answer
        answer = retrieval_chain.invoke(query)
        
        # Get sources (need to run retriever separately to get docs)
        source_documents = retriever.invoke(query)
        
        # Format the response using common formatter
        if return_formatted:
            return format_response(answer, source_documents)
        else:
            return answer, source_documents
        
    except Exception as e:
        return f"Error querying the model: {str(e)}"

def search_code(query, search_type="qa", k=5, rerank=False, vectorstore=None):
    """
    Search the code repository using either QA or similarity search
    This function serves as a wrapper to handle both types of searches.
    No top_k parameter is needed here as it is handled in the default value for qa_search function.
    
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
        return qa_search(query, vectorstore, k=k, rerank=rerank)
    else:
        # Use similarity search for retrieving relevant code snippets
        try:
            results = vectorstore.similarity_search(query, k=k)
            # Format the similarity search results using the same formatter
            return format_response(f"Results for: {query}", results)
        except Exception as e:
            return f"Error during search: {str(e)}"

