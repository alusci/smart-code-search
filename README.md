# Smart Code Search

Smart Code Search for Git—Enhances code discovery with Retrieval-Augmented Generation (RAG) integration.

This tool allows you to search through any GitHub repository using natural language, enabling you to:
- Ask questions about the codebase
- Find relevant code snippets based on functionality
- Understand how different parts of the code work together

## Features

- **Natural Language Search**: Ask questions in plain English about any part of your codebase
- **Code Retrieval**: Find specific code snippets based on functionality
- **Source-backed Answers**: All answers include the source code they're based on
- **Simple Web Interface**: Easy-to-use Gradio web app
- **Git Integration**: Works with any GitHub repository

## Installation

### Prerequisites

- Python 3.11 or higher
- Git

### Setup

1. Create a conda environment:
   ```bash
   conda create -n codesearch python=3.11
   conda activate codesearch
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Create a `.env` file in the root directory of your project and add the following configuration:

```markdown
# Git repository information
GIT_REPO_URL=https://github.com/username/repository.git
GIT_REPO_PATH=./repository

# Indexing settings
ALL_FILES=True # Set this parameter to False if you want to index just the latest changes
CHUNK_SIZE=1000
CHUNK_OVERLAP=100

# OpenAI API settings
OPENAI_API_KEY=your-api-key-here
EMBEDDING_MODEL=text-embedding-ada-002
```

## Usage

### 1. Choose a GitHub Repository

Update the `GIT_REPO_URL` and `GIT_REPO_PATH` in your `.env` file to point to the repository you want to search.

### 2. Index the Repository

Run the document indexer to clone, process, and index the repository:

```bash
python document_indexer.py
```

This will:

- Clone the repository if it doesn't exist locally
- Pull the latest changes if it does exist
- Process and index the files
- Create a vector store of the code

### 3. Launch the Web App

Start the web interface:

```bash
python app.py
```

The terminal will display a URL (typically http://127.0.0.1:7860) that you can open in your browser.

### 4. Interact with the App

The web interface offers two main modes:

#### Question Answering

Ask questions about the codebase like:

- "How does the document indexing work?"
- "What is the purpose of the vectorstore module?"
- "How are documents processed and split?"

#### Code Retrieval

Search for specific code snippets using queries like:

- "function to load documents"
- "error handling in vectorstore"
- "batched processing implementation"

## How It Works

1. **Document Processing**: The code repository is split into manageable chunks
2. **Vectorization**: These chunks are converted to vector embeddings using OpenAI's embedding model
3. **Indexing**: A FAISS vector index is created for efficient similarity search
4. **Retrieval**: When you ask a question, relevant code chunks are retrieved
5. **Generation**: For questions, an LLM uses the retrieved code to generate an answer

## Customization

- Adjust chunking settings in the `.env` file
- Modify the number of results in the web UI
- Choose between different search types

## Troubleshooting

- **No vectorstore found**: Run `document_indexer.py` to create the index
- **API errors**: Verify your OpenAI API key is correct in the `.env` file
- **Empty results**: Try reformulating your query or indexing more files

## License

Apache License 2.0

For more information or to contribute, please open an issue or pull request.