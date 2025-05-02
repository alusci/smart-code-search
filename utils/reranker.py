def create_mmr_retriever(vectorstore, k=5, top_k=20, lambda_mult=0.8):
    """
    Create a retriever with optional re-ranking capability.
    
    Args:
        vectorstore: The base vector store to use for initial retrieval
        use_reranking (bool): Whether to use re-ranking or just vanilla retrieval
        k (int): Number of documents to keep after re-ranking
        top_k (int): Number of documents to retrieve in the initial pass
        lambda_mult (float): The balance between relevance and diversity in MMR
                             (0-1, lower values favor diversity more)
                             Default is 0.5 for equal balance.
        
    Returns:
        A retriever with optional re-ranking capability
    """
    # If re-ranking is requested, use MMR retriever
    
    try:
        mmr_retriever = vectorstore.as_retriever(
            search_type="mmr",  # Maximal Marginal Relevance
            search_kwargs={
                "k": k, 
                "fetch_k": top_k,   # Fetch more candidates initially
                "lambda_mult": lambda_mult  # Equal balance between relevance and diversity (0-1)
                                    # Lower values favor diversity more
            }
        )
        print("Using MMR retriever")
        return mmr_retriever
        
    except Exception as e:
        print(f"Warning: Could not create MMR retriever: {e}")
        print("Falling back to vanilla similarity search")
    
    # Default to vanilla similarity search
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )