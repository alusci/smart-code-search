import os
import git
from typing import List
from .vectorstore import get_vectorstore, init_vectorstore

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, UnstructuredFileLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.notebook import NotebookLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.schema import Document


def get_changed_files(repo_url: str, repo_path: str, get_all_files: bool = False) -> List[str]:
    """
    Get a list of changed files in the repository.

    Args:
        repo_url (str): The URL of the repository.
        repo_path (str): The local path to the repository.
        get_all_files (bool): If True, return all files in the repository.

    Returns:
        list: A list of changed files.
    """

    # Clone the repository if it doesn't exist
    if not os.path.exists(repo_path):
        print(f"Cloning repository {repo_path} from {repo_url}...")
        repo = git.Repo.clone_from(repo_url, repo_path)
    else:
        repo = git.Repo(repo_path)

    # Ensure we're on the main branch and fetch the latest changes
    # Try 'main' first, fall back to 'master' if main doesn't exist
    try:
        repo.git.checkout("main")
        default_branch = "main"
    except git.GitCommandError:
        repo.git.checkout("master")
        default_branch = "master"
        
    repo.git.pull()

    # Get all files if requested
    if get_all_files:
        print("Returning all files in the repository...")
        return [f"{repo_path}/{v.path}" for k,v in repo.index.entries.items()]
    
    # Get the latest two commits on the branch
    commits = list(repo.iter_commits(default_branch, max_count=2))
    
    # If there's only one commit, compare with empty tree
    if len(commits) == 1:
        latest_commit = commits[0]
        changed_files = [f"{repo_path}/{item.a_path}" for item in latest_commit.diff(git.NULL_TREE)]
    else:
        # Calculate the diff between the latest commit and the previous commit
        latest_commit, previous_commit = commits[0], commits[1]
        changed_files = [f"{repo_path}/{item.a_path}" for item in previous_commit.diff(latest_commit)]
    
    return changed_files


def load_and_split_documents(file_paths: List[str], chunk_size=1000, chunk_overlap=100)-> List[Document]:
    """
    Load and split specific documents from a list of file paths.
    
    Args:
        file_paths (list): List of file paths to load documents from
        chunk_size (int): Size of each text chunk
        chunk_overlap (int): Overlap between chunks
        
    Returns:
        list: List of document chunks
    """
    
    documents = []
    
    # Process each file individually
    failed, skipped = 0 ,0

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist, skipping.")
            continue
            
        try:
            # Choose loader based on file extension
            file_extension = os.path.splitext(file_path)[1].lower()
            
            # Text files and config files
            if file_extension in ['.txt', '.md', '.py', '.js', '.html', '.css', '.java', '.c', '.cpp', '.ts', 
                                 '.yaml', '.yml', '.toml', '.json', '.xml', '.ini', '.config', '.conf']:
                loader = TextLoader(file_path)
            # Data files
            elif file_extension == '.csv':
                loader = CSVLoader(file_path)
            # Notebooks
            elif file_extension == '.ipynb':
                loader = NotebookLoader(file_path, include_outputs=True, max_output_length=20)
            # PDFs
            elif file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            else:
                # Skip unknown file types
                skipped += 1
                continue
                
            # Load the document
            file_docs = loader.load()
            documents.extend(file_docs)
            
        except Exception as e:
            failed += 1
            print(f"Error loading {file_path}: {e}")

    processed = len(file_paths) - failed - skipped
    print(f"Processed {processed} files, failed to load {failed} files, skipped {skipped} files.")
    
    # Split documents
    if documents:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return text_splitter.split_documents(documents)
    else:
        return []
    

def index_documents(documents):
    """Index documents in the vector store"""
    
    vectorstore = get_vectorstore()
    if vectorstore:
        vectorstore.add_documents(documents)
    else:
        vectorstore = init_vectorstore(documents)
    return vectorstore 

