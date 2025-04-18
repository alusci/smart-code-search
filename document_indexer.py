from utils.document_processor import get_changed_files, load_and_split_documents, index_documents
from utils import config
from utils.vectorstore import save_vectorstore

def main():
    """
    Main function to process git repository documents.
    """
    try:
        # Get changed files
        changed_files = get_changed_files(
            config.REPO_URL, 
            config.REPO_PATH, 
            config.GET_ALL_FILES
        )
        
        print(f"Found {len(changed_files)} changed files")
        
        # Process documents
        documents = load_and_split_documents(
            changed_files,
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        
        print(f"Split into {len(documents)} document chunks")

        # Index documents
        vectorstore = index_documents(documents)
        save_vectorstore(vectorstore)
        print(f"Successfully indexed {len(documents)} document chunks")

    except Exception as e:
        print(f"Error: {e}")
        return []

if __name__ == "__main__":
    main()

