import os
from pathlib import Path
from typing import Dict, List, Any

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from fastapi.responses import JSONResponse
from app.api.models import QueryRequest, QueryResponse, DocumentInfo, UploadResponse
from utils.config import settings
from utils.logging import logger, log_qa_to_file
from service.intent_processor import intent_service
from service.rag_service import rag_service
from service.response_generator import ResponseGeneratorService

# Initialize router
router = APIRouter(prefix="/api", tags=["Document Chatbot"])

# Initialize response generator
response_generator_service = ResponseGeneratorService(settings.SCHEMA_PATH)


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest) -> Dict[str, Any]:

    try:
        query = request.query
        logger.info(f"Processing query: '{query}'")

        # Step 1: Process intent and slots
        intent_result = intent_service.process_intent(query=query)
        intent = intent_result["intent"]
        slots = intent_result["slots"]
        is_out_of_scope = intent_result["is_out_of_scope"]
        missing_required_slots = intent_result["missing_required_slots"]
        confidence = intent_result["confidence"]

        # Step 2: Enhance retrieval query with slot values
        retrieval_query = query
        if "topic" in slots:
            retrieval_query = f"{query} {slots['topic']}"
        if "term" in slots:
            retrieval_query = f"{query} {slots['term']}"
        if "section" in slots:
            retrieval_query = f"{query} {slots['section']}"

        # Step 3: Retrieve relevant documents
        documents = []
        if not is_out_of_scope:
            qa_result = rag_service.get_answer(query=query, enhanced_query=retrieval_query)
            if qa_result["success"]:
                documents = qa_result["documents"]

        # Step 4: Generate response
        response_result = response_generator_service.generate_response(
            query=query,
            documents=documents,
            intent=intent,
            slots=slots,
            is_out_of_scope=is_out_of_scope,
            missing_required_slots=missing_required_slots,
            confidence=confidence
        )

        # Step 5: Log question and answer to file
        log_qa_to_file(query, response_result["response"])

        # Return the response
        return {
            "query": query,
            "intent": intent,
            "slots": slots,
            "is_out_of_scope": is_out_of_scope,
            "response": response_result["response"],
            "confidence": confidence,
            "documents_used": [doc.content for doc in documents]
        }
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> Dict[str, Any]:

    try:
        if not file.filename.endswith(".txt"):
            raise HTTPException(status_code=400, detail="Only .txt files are supported")

        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)

        index_result = rag_service.index_document(Path(temp_file_path))
        os.remove(temp_file_path)

        if not index_result["success"]:
            raise HTTPException(status_code=500,
                                detail=f"Error indexing document: {index_result.get('error', 'Unknown error')}")

        return {
            "message": f"Document '{file.filename}' uploaded and indexed successfully",
            "document_count": index_result["document_count"],
            "success": True
        }
    except HTTPException as he:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise he
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Error processing upload: {str(e)}")


@router.get("/documents", response_model=DocumentInfo)
async def get_documents() -> Dict[str, Any]:

    try:
        doc_info = rag_service.get_document_info()
        return {
            "document_count": doc_info["document_count"],
            "document_names": doc_info["document_names"]
        }
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving documents: {str(e)}")