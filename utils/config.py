"""
Configuration module for the application.
Loads and validates parameters from environment variables on import.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Required parameters
REPO_URL = os.getenv("GIT_REPO_URL")
REPO_PATH = os.getenv("GIT_REPO_PATH")

# Optional parameters with defaults
GET_ALL_FILES = os.getenv("ALL_FILES", "False").lower() in ("true", "1", "yes")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("OPENAI_MODEL", "llama3.2:1b")

# Validate required parameters
if not REPO_URL:
    raise ValueError("GIT_REPO_URL is not set in the .env file")
if not REPO_PATH:
    raise ValueError("GIT_REPO_PATH is not set in the .env file")
