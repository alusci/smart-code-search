from .config import OPENAI_API_KEY, EMBEDDING_MODEL
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI

def initialize_embeddings():
    """
    Initialize the embedding model.
    """
    
    return OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        model=EMBEDDING_MODEL,
        disallowed_special=()
    )

def initialize_llm():
    """
    Initialize the LLM model.
    """
    
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.0
    )
    
    return llm


