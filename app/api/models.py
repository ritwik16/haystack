from typing import Dict, List, Optional
from pydantic import BaseModel


class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    query: str


class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    query: str
    intent: str
    slots: Dict[str, str]
    is_out_of_scope: bool
    response: str
    confidence: float
    documents_used: List[str]


class DocumentInfo(BaseModel):
    """Response model for document info endpoint"""
    document_count: int
    document_names: List[str]


class UploadResponse(BaseModel):
    """Response model for document upload endpoint"""
    message: str
    document_count: int
    success: bool