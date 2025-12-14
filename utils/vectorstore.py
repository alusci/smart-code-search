from langchain_community.vectorstores import FAISS
from .models import initialize_embeddings
import os
import time
from tqdm import tqdm


def get_vectorstore(documents: list = None):
    """Get or create a vector store instance"""
    
    vectorstore = None
    embeddings = initialize_embeddings()
        
    # Check if index directory exists
    if os.path.exists("./faiss_index"):
        try:
            # Try to load existing index
            vectorstore = FAISS.load_local(
                "./faiss_index", 
                embeddings,
                allow_dangerous_deserialization=True # Don't do this in production!
            )
        except RuntimeError as e:
            print(f"Could not load existing index: {e}")
    
    return vectorstore


def init_vectorstore(documents, batch_size=100) -> bool:
    """
    Init FAISS vector store with batched processing for better performance
    
    Args:
        documents: List of documents to embed
        batch_size: Number of documents to process in each batch
        
    Returns:
        FAISS vectorstore
    """
    
    embeddings = initialize_embeddings()
    
    if not documents:
        print("Warning: No documents to index")
        return None
    
    start_time = time.time()
    print(f"Starting vectorstore creation with {len(documents)} documents")
        
    # Process first batch to initialize the vectorstore
    first_batch = documents[:min(batch_size, len(documents))]
    print(f"Initializing with first batch of {len(first_batch)} documents...")
    vectorstore = FAISS.from_documents(
        first_batch, 
        embeddings
    )
    
    # Process remaining documents in batches
    remaining = documents[batch_size:] if batch_size < len(documents) else []
    
    if remaining:
        add_document_batches(documents, vectorstore, batch_size=batch_size)
    
    # Calculate and display total processing time
    total_time = time.time() - start_time
    docs_per_second = len(documents) / total_time if total_time > 0 else 0
    print(f"Completed vectorstore creation in {total_time:.1f}s ({docs_per_second:.1f} docs/s)")
    
    return vectorstore


def add_document_batches(documents, vectorstore, batch_size=100):

    # Calculate total batches for the progress bar
    total_batches = (len(documents) + batch_size - 1) // batch_size
    
    print(f"Processing {len(documents)} documents in {total_batches} batches:")
    
    # Create progress bar
    pbar = tqdm(total=len(documents), unit="docs", desc="Embedding")
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        batch_start = time.time()
        
        # Update progress bar description with current batch info
        pbar.set_description(f"Batch {i//batch_size + 1}/{total_batches}")
        
        # Add documents to vectorstore
        try:
            vectorstore.add_documents(batch)
        except Exception as e:
            # In a real production scenario you should not catch such a generic exception
            print(e)
        
        # Update progress bar
        pbar.update(len(batch))
        
        # Calculate and display batch processing time
        batch_time = time.time() - batch_start
        docs_per_second = len(batch) / batch_time if batch_time > 0 else 0
        pbar.set_postfix({"docs/s": f"{docs_per_second:.1f}"})
    
    pbar.close()


def save_vectorstore(vectorstore):
    """Save the vector store to disk"""
    if vectorstore is not None:
        # Create directory if it doesn't exist
        os.makedirs("./faiss_index", exist_ok=True)
        vectorstore.save_local("./faiss_index")
