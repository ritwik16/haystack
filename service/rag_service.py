from pathlib import Path
from typing import List, Dict, Optional
from haystack import Pipeline, component
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.converters import TextFileToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import PromptBuilder
from haystack.dataclasses import ChatMessage, Document
from utils.config import settings
from utils.logging import logger


# Custom component to convert a string prompt to List[ChatMessage]
@component
class ChatMessageConverter:
    @component.output_types(messages=List[ChatMessage])
    def run(self, prompt: str):
        return {"messages": [ChatMessage.from_user(prompt)]}


class RAGService:
    """Service for RAG (Retrieval Augmented Generation) operations"""

    def __init__(self):
        """Initialize RAG components and pipelines"""
        # Initialize document store
        self.document_store = InMemoryDocumentStore()

        # Define prompt template with required variables
        self.prompt_template = """
        You are a helpful assistant. Based on the following documents, answer the question concisely and accurately.

        Documents:
        {% for doc in documents %}
        - {{ doc.content }}
        {% endfor %}

        Question: {{ question }}
        Answer:
        """

        # Create indexing and QA pipelines
        self.indexing_pipeline = self._create_indexing_pipeline()
        self.qa_pipeline = self._create_qa_pipeline()

        logger.info("RAG service initialized successfully")

    def _create_indexing_pipeline(self) -> Pipeline:
        """Create the document indexing pipeline"""
        indexing_pipeline = Pipeline()
        indexing_pipeline.add_component("cleaner", DocumentCleaner())
        indexing_pipeline.add_component("splitter", DocumentSplitter(
            split_by=settings.SPLIT_BY,
            split_length=settings.SPLIT_LENGTH,
            split_overlap=settings.SPLIT_OVERLAP
        ))
        indexing_pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder(
            model=settings.EMBEDDING_MODEL
        ))
        indexing_pipeline.add_component("writer", DocumentWriter(document_store=self.document_store))

        # Connect components
        indexing_pipeline.connect("cleaner", "splitter")
        indexing_pipeline.connect("splitter", "embedder")
        indexing_pipeline.connect("embedder", "writer")

        return indexing_pipeline

    def _create_qa_pipeline(self) -> Pipeline:
        """Create the question-answering pipeline"""
        qa_pipeline = Pipeline()
        qa_pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder(
            model=settings.EMBEDDING_MODEL
        ))
        qa_pipeline.add_component("retriever", InMemoryEmbeddingRetriever(
            document_store=self.document_store,
            top_k=settings.RETRIEVER_TOP_K
        ))
        qa_pipeline.add_component("prompt_builder", PromptBuilder(
            template=self.prompt_template,
            required_variables=["documents", "question"]
        ))
        qa_pipeline.add_component("message_converter", ChatMessageConverter())
        qa_pipeline.add_component("generator", OpenAIChatGenerator(
            api_key=settings.OPENROUTER_API_KEY,
            api_base_url=settings.OPENROUTER_BASE_URL,
            model=settings.RAG_MODEL,
            generation_kwargs={
                "max_tokens": settings.RAG_MAX_TOKENS,
                "temperature": settings.RAG_TEMPERATURE
            }
        ))

        # Connect components
        qa_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        qa_pipeline.connect("retriever.documents", "prompt_builder.documents")
        qa_pipeline.connect("prompt_builder.prompt", "message_converter.prompt")
        qa_pipeline.connect("message_converter.messages", "generator.messages")

        return qa_pipeline

    def index_document(self, file_path: Path) -> Dict:
        """Index a document from a file path"""
        try:
            converter = TextFileToDocument()
            docs = converter.run(sources=[file_path])
            self.indexing_pipeline.run({"cleaner": {"documents": docs["documents"]}})

            doc_count = self.document_store.count_documents()
            logger.info(f"Document '{file_path}' indexed successfully. Total documents: {doc_count}")

            return {
                "success": True,
                "document_count": doc_count,
                "message": f"Document '{file_path.name}' indexed successfully"
            }
        except Exception as e:
            logger.error(f"Error indexing document '{file_path}': {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def retrieve_documents(self, query: str) -> List[Document]:
        """Retrieve relevant documents for a query"""
        try:
            return self.document_store.filter_documents()
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []

    def get_answer(self, query: str, enhanced_query: Optional[str] = None) -> Dict:
        """Get answer for a query using the QA pipeline"""
        try:
            # Use enhanced query if provided
            retrieval_query = enhanced_query if enhanced_query else query

            # Run QA pipeline
            result = self.qa_pipeline.run(
                {
                    "text_embedder": {"text": retrieval_query},
                    "prompt_builder": {"question": query}
                },
                include_outputs_from=["retriever", "generator"]
            )

            return {
                "success": True,
                "documents": result["retriever"]["documents"],
                "answer": result["generator"]["replies"][0]
            }
        except Exception as e:
            logger.error(f"Error getting answer for query '{query}': {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "documents": [],
                "answer": "An error occurred while generating the answer. Please try again."
            }

    def get_document_info(self) -> Dict:
        """Get information about indexed documents"""
        try:
            docs = self.document_store.filter_documents()
            return {
                "document_count": self.document_store.count_documents(),
                "document_names": list(set(doc.meta.get("file_path", "unknown") for doc in docs))
            }
        except Exception as e:
            logger.error(f"Error getting document info: {str(e)}")
            return {
                "document_count": 0,
                "document_names": [],
                "error": str(e)
            }


# Singleton instance
rag_service = RAGService()