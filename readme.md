# Document-Grounded Conversational Chatbot

A modular FastAPI application that creates a conversational chatbot which answers questions based on document content.

## Features

- Intent and slot detection for user queries
- Document upload and indexing
- Retrieval-Augmented Generation (RAG) for document Q&A
- Intent-based response generation
- API endpoints for querying, document upload, and document info


### Prerequisites

- Python 3.8+
- Required packages (install using `pip install -r requirements.txt`)

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

```bash
python scripts/run.py
```

Options:
- `--host`: Host to run the API on (default: 0.0.0.0)
- `--port`: Port to run the API on (default: 8000)
- `--reload`: Enable hot reload for development

## API Endpoints

### Query Endpoint

```
POST /api/query
```

Query the chatbot with a question about uploaded documents.

**Request Body:**
```json
{
  "query": "What does the document say about AI?"
}
```

**Response:**
```json
{
  "query": "What does the document say about AI?",
  "intent": "document_query",
  "slots": {
    "topic": "AI"
  },
  "is_out_of_scope": false,
  "response": "The document explains that AI (Artificial Intelligence) is...",
  "confidence": 0.92,
  "documents_used": ["AI is a branch of computer science...", "...]
}
```

### Upload Document

```
POST /api/upload
```

Upload a document to be indexed and queried.

**Request Body:**
- Form data with file field

**Response:**
```json
{
  "message": "Document 'sample.txt' uploaded and indexed successfully",
  "document_count": 1,
  "success": true
}
```

### Get Documents Info

```
GET /api/documents
```

Get information about indexed documents.

**Response:**
```json
{
  "document_count": 2,
  "document_names": ["sample.txt", "document2.txt"]
}
```

## Architecture

The application uses a modular architecture with the following key components:

1. **Intent Processing**: Analyzes user queries to determine intent and extract slots
2. **RAG Service**: Manages document storage, indexing, and retrieval
3. **Response Generation**: Creates responses based on intent and retrieved documents
4. **API Layer**: Provides endpoints for interaction with the chatbot

## Configuration

Edit `core/config.py` to customize application settings:

- API settings
- File paths
- Model parameters
- Document processing settings
- Retrieval settings

## License

[MIT License](LICENSE)