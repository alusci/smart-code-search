import os
import textwrap
from typing import List
from langchain_core.documents import Document

def format_response(answer, source_documents):
    """
    Format a response with answer and source documents.
    
    Args:
        answer (str): The text answer or heading
        source_documents (list): List of source documents
        
    Returns:
        str: Formatted response with answer and sources
    """
    formatted_response = f"## Answer\n\n{answer}\n\n## Sources\n\n"
    
    # Add source documents
    for i, doc in enumerate(source_documents):
        file_path = doc.metadata.get("source", "Unknown source")
        rel_path = os.path.basename(file_path)
        
        formatted_response += f"### Source {i+1}: {rel_path}\n"
        formatted_response += f"**Path:** `{file_path}`\n\n"
        
        # Detect language for syntax highlighting based on file extension
        extension = os.path.splitext(file_path)[1].lower()
        language = "python"  # default
        
        if extension in ['.js', '.jsx', '.ts', '.tsx']:
            language = "javascript"
        elif extension in ['.html', '.htm']:
            language = "html"
        elif extension in ['.css']:
            language = "css"
        elif extension in ['.json']:
            language = "json"
        elif extension in ['.md', '.markdown']:
            language = "markdown"
        
        formatted_response += f"```{language}\n"
        formatted_response += textwrap.dedent(doc.page_content)
        formatted_response += "\n```\n\n"
    
    return formatted_response
