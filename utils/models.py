from .config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL, OPENAI_MODEL
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

def initialize_embeddings():
    """
    Initialize the embedding model.
    """
    
    return OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        model=OPENAI_EMBEDDING_MODEL,
        disallowed_special=()
    )

def initialize_llm():
    """
    Initialize the LLM model.
    """
    
    llm = ChatOpenAI(
        model_name=OPENAI_MODEL,
        temperature=0.0
    )
    
    return llm


