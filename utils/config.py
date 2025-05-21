import os
from pathlib import Path
from haystack.utils import Secret


class Settings:
    # API settings
    API_TITLE = "Document-Grounded Conversational Chatbot"
    API_DESCRIPTION = "A conversational chatbot that answers questions based on document content"

    # File paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    SCHEMA_DIR = BASE_DIR / "schemas"
    SCHEMA_PATH = SCHEMA_DIR / "intent_slot_schema.json"
    QA_LOG_FILE = BASE_DIR / "qa_log.txt"

    # Document store settings
    DOCUMENT_STORE_TYPE = "in_memory"  # Can be changed to other types

    # Model settings
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    OPENROUTER_API_KEY = Secret.from_token(os.getenv(
        "OPENROUTER_API_KEY",
        "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"
    ))

    # Intent processor model
    INTENT_MODEL = "qwen/qwen-2.5-7b-instruct"
    INTENT_MAX_TOKENS = 200
    INTENT_TEMPERATURE = 0.2

    # RAG model
    RAG_MODEL = "qwen/qwen-2.5-7b-instruct"
    RAG_MAX_TOKENS = 400
    RAG_TEMPERATURE = 0.5

    # Embedding model
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    # Document processing
    SPLIT_BY = "sentence"
    SPLIT_LENGTH = 6
    SPLIT_OVERLAP = 1

    # Retrieval settings
    RETRIEVER_TOP_K = 3


settings = Settings()