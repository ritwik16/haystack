import json
import os
from pathlib import Path

from utils.config import settings
from utils.logging import logger


def create_default_schema():
    """Create the default intent schema if it doesn't exist"""
    schema_dir = settings.SCHEMA_DIR
    schema_path = settings.SCHEMA_PATH

    # Create schema directory if it doesn't exist
    schema_dir.mkdir(parents=True, exist_ok=True)

    # Check if schema file exists, if not create default
    if not schema_path.exists():
        default_schema = {
            "intents": [
                {
                    "name": "document_query",
                    "description": "Questions about document content",
                    "slots": [
                        {
                            "name": "topic",
                            "description": "The topic the user is asking about",
                            "is_required": True
                        },
                        {
                            "name": "section",
                            "description": "The section of the document to focus on",
                            "is_required": False
                        }
                    ],
                    "examples": [
                        "What does the document say about health?",
                        "Tell me about the topic of climate in the document",
                        "I want to know what information is provided about AI"
                    ]
                },
                {
                    "name": "find_definition",
                    "description": "Looking for a definition of a term in the document",
                    "slots": [
                        {
                            "name": "term",
                            "description": "The term to find a definition for",
                            "is_required": True
                        }
                    ],
                    "examples": [
                        "What is the definition of blockchain in the document?",
                        "How does the document define AI?",
                        "What does the term climate change mean according to the document?"
                    ]
                },
                {
                    "name": "document_summary",
                    "description": "Requesting a summary of the document or a section",
                    "slots": [
                        {
                            "name": "section",
                            "description": "The section to summarize (optional)",
                            "is_required": False
                        }
                    ],
                    "examples": [
                        "Can you summarize the document?",
                        "Give me a summary of the section on education",
                        "Provide a brief overview of what the document is about"
                    ]
                },
                {
                    "name": "document_metadata",
                    "description": "Questions about document metadata like author, date, title",
                    "slots": [
                        {
                            "name": "metadata_type",
                            "description": "The type of metadata requested (author, date, title, etc.)",
                            "is_required": False
                        }
                    ],
                    "examples": [
                        "Who wrote this document?",
                        "When was this document published?",
                        "What's the title of this document?"
                    ]
                },
                {
                    "name": "out_of_scope",
                    "description": "Questions unrelated to the document",
                    "slots": [],
                    "examples": [
                        "What's the weather like today?",
                        "Tell me a joke",
                        "How do I make pancakes?"
                    ]
                }
            ],
            "responses": {
                "out_of_scope": "I'm designed to answer questions only about the content in the loaded document. I can't help with queries outside that scope.",
                "fallback_not_in_document": "I'm sorry, but I don't see information about that in the document. I can only answer questions based on the document content.",
                "no_documents_loaded": "There are no documents loaded yet. Please upload a document first.",
                "welcome": "Hello! I'm a document chatbot. I can answer questions about documents you upload. How can I help you today?"
            }
        }

        # Write default schema to file
        with open(schema_path, 'w') as f:
            json.dump(default_schema, f, indent=2)

        logger.info(f"Created default intent schema at {schema_path}")
    else:
        logger.info(f"Intent schema already exists at {schema_path}")

    return schema_path


def ensure_directory_structure():
    """Ensure all required directories exist"""
    # Create logs directory
    logs_dir = settings.BASE_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Create temp directory for file uploads
    temp_dir = settings.BASE_DIR / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Directory structure verified")


def initialize_app():
    """Run all initialization steps"""
    try:
        ensure_directory_structure()
        create_default_schema()
        logger.info("Application initialization complete")
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        raise