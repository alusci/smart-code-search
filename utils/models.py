from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from .config import EMBEDDING_MODEL, LLM_MODEL

def initialize_embeddings():
    """Initialize the embedding model"""
    
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,  # Remember to pull the model from Ollama!
        base_url="http://localhost:11434"
    )

def initialize_llm():
    """Initialize the LLM"""

    return ChatOllama(
        model=LLM_MODEL,  # Specify the OLLAMA LLM model you want to use
        temperature=0.0,
        base_url="http://localhost:11434"
    )

