"""
FastAPI REST API for RAG Document Processing Utility

Provides a web-based interface for document upload, processing, and results retrieval.
"""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import Config
from processor import DocumentProcessor
from multimodal import get_multimodal_processor
from metadata_enhancement import get_metadata_enhancer

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Document Processing Utility API",
    description="API for processing documents for RAG applications",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global configuration and processor
config = Config()
document_processor = DocumentProcessor(config)
multimodal_processor = get_multimodal_processor(config)
metadata_enhancer = get_metadata_enhancer(config)

# Processing status storage
processing_status = {}


# Pydantic models for API requests/responses
class ProcessingRequest(BaseModel):
    """Request model for document processing."""
    document_path: str
    chunking_strategy: Optional[str] = "hybrid"
    metadata_level: Optional[str] = "advanced"
    enable_multimodal: Optional[bool] = True


class ProcessingResponse(BaseModel):
    """Response model for document processing."""
    success: bool
    document_id: str
    processing_time: float
    chunks_count: int
    quality_score: float
    message: str
    status_url: str


class StatusResponse(BaseModel):
    """Response model for processing status."""
    document_id: str
    status: str  # "pending", "processing", "completed", "failed"
    progress: float  # 0.0 to 1.0
    message: str
    result: Optional[Dict[str, Any]] = None


class BatchProcessingRequest(BaseModel):
    """Request model for batch document processing."""
    document_paths: List[str]
    chunking_strategy: Optional[str] = "hybrid"
    metadata_level: Optional[str] = "advanced"
    enable_multimodal: Optional[bool] = True


class BatchProcessingResponse(BaseModel):
    """Response model for batch processing."""
    success: bool
    batch_id: str
    total_documents: int
    processing_time: float
    results: List[Dict[str, Any]]
    message: str


# API endpoints
@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API documentation."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG Document Processing Utility API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .endpoint { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
            .method { font-weight: bold; color: #007bff; }
            .url { font-family: monospace; background: #f8f9fa; padding: 5px; }
        </style>
    </head>
    <body>
        <h1>RAG Document Processing Utility API</h1>
        <p>Welcome to the RAG Document Processing Utility API. This API provides endpoints for processing documents for RAG applications.</p>
        
        <h2>Available Endpoints</h2>
        
        <div class="endpoint">
            <div class="method">POST</div>
            <div class="url">/process-document</div>
            <p>Process a single document and return results.</p>
        </div>
        
        <div class="endpoint">
            <div class="method">POST</div>
            <div class="url">/process-batch</div>
            <p>Process multiple documents in batch mode.</p>
        </div>
        
        <div class="endpoint">
            <div class="method">GET</div>
            <div class="url">/status/{document_id}</div>
            <p>Get processing status for a document.</p>
        </div>
        
        <div class="endpoint">
            <div class="method">GET</div>
            <div class="url">/health</div>
            <p>Health check endpoint.</p>
        </div>
        
        <h2>Interactive API Documentation</h2>
        <p>For detailed API documentation and testing, visit:</p>
        <ul>
            <li><a href="/docs">Swagger UI</a></li>
            <li><a href="/redoc">ReDoc</a></li>
        </ul>
    </body>
    </html>
    """


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "RAG Document Processing Utility API",
        "version": "0.1.0",
        "timestamp": time.time()
    }


@app.post("/process-document", response_model=ProcessingResponse)
async def process_document(request: ProcessingRequest, background_tasks: BackgroundTasks):
    """Process a single document."""
    try:
        # Validate document path
        if not os.path.exists(request.document_path):
            raise HTTPException(status_code=400, detail="Document file not found")
        
        # Generate document ID
        document_id = f"doc_{int(time.time())}_{hash(request.document_path)}"
        
        # Store initial status
        processing_status[document_id] = {
            "status": "pending",
            "progress": 0.0,
            "message": "Processing queued",
            "result": None
        }
        
        # Start background processing
        background_tasks.add_task(
            process_document_background,
            document_id,
            request.document_path,
            request.chunking_strategy,
            request.metadata_level,
            request.enable_multimodal
        )
        
        return ProcessingResponse(
            success=True,
            document_id=document_id,
            processing_time=0.0,
            chunks_count=0,
            quality_score=0.0,
            message="Document processing started",
            status_url=f"/status/{document_id}"
        )
    
    except Exception as e:
        logger.error(f"Error starting document processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process-batch", response_model=BatchProcessingResponse)
async def process_batch(request: BatchProcessingRequest, background_tasks: BackgroundTasks):
    """Process multiple documents in batch mode."""
    try:
        # Validate document paths
        for doc_path in request.document_paths:
            if not os.path.exists(doc_path):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Document file not found: {doc_path}"
                )
        
        # Generate batch ID
        batch_id = f"batch_{int(time.time())}"
        
        # Start background batch processing
        background_tasks.add_task(
            process_batch_background,
            batch_id,
            request.document_paths,
            request.chunking_strategy,
            request.metadata_level,
            request.enable_multimodal
        )
        
        return BatchProcessingResponse(
            success=True,
            batch_id=batch_id,
            total_documents=len(request.document_paths),
            processing_time=0.0,
            results=[],
            message="Batch processing started"
        )
    
    except Exception as e:
        logger.error(f"Error starting batch processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/{document_id}", response_model=StatusResponse)
async def get_processing_status(document_id: str):
    """Get processing status for a document."""
    if document_id not in processing_status:
        raise HTTPException(status_code=404, detail="Document ID not found")
    
    status = processing_status[document_id]
    
    return StatusResponse(
        document_id=document_id,
        status=status["status"],
        progress=status["progress"],
        message=status["message"],
        result=status["result"]
    )


@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document file for processing."""
    try:
        # Create upload directory if it doesn't exist
        upload_dir = Path("temp/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded file
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Return file information
        return {
            "success": True,
            "filename": file.filename,
            "file_path": str(file_path),
            "size": len(content),
            "message": "File uploaded successfully"
        }
    
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
async def list_processed_documents():
    """List all processed documents."""
    try:
        documents = []
        for doc_id, status in processing_status.items():
            if status["status"] == "completed":
                documents.append({
                    "document_id": doc_id,
                    "status": status["status"],
                    "result": status["result"]
                })
        
        return {
            "success": True,
            "total_documents": len(documents),
            "documents": documents
        }
    
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Background processing functions
async def process_document_background(document_id: str, document_path: str, 
                                    chunking_strategy: str, metadata_level: str, 
                                    enable_multimodal: bool):
    """Background task for processing a single document."""
    try:
        # Update status to processing
        processing_status[document_id]["status"] = "processing"
        processing_status[document_id]["progress"] = 0.1
        processing_status[document_id]["message"] = "Starting document processing"
        
        # Process document
        result = document_processor.process_document(document_path)
        
        # Update progress
        processing_status[document_id]["progress"] = 0.5
        processing_status[document_id]["message"] = "Document processed, enhancing metadata"
        
        # Enhance metadata if enabled
        if enable_multimodal:
            # Process multimodal content
            multimodal_result = multimodal_processor.process_content(
                document_path, 
                Path(document_path).suffix[1:], 
                result
            )
            
            # Update progress
            processing_status[document_id]["progress"] = 0.8
            processing_status[document_id]["message"] = "Multimodal content processed"
        
        # Update final status
        processing_status[document_id]["status"] = "completed"
        processing_status[document_id]["progress"] = 1.0
        processing_status[document_id]["message"] = "Processing completed successfully"
        processing_status[document_id]["result"] = {
            "success": result.success,
            "chunks_count": len(result.chunks),
            "quality_score": result.quality_score,
            "processing_time": result.processing_time,
            "metadata": result.metadata,
            "multimodal_result": multimodal_result.to_dict() if enable_multimodal else None
        }
        
        logger.info(f"Document {document_id} processed successfully")
    
    except Exception as e:
        logger.error(f"Error processing document {document_id}: {e}")
        processing_status[document_id]["status"] = "failed"
        processing_status[document_id]["progress"] = 0.0
        processing_status[document_id]["message"] = f"Processing failed: {str(e)}"
        processing_status[document_id]["result"] = {"error": str(e)}


async def process_batch_background(batch_id: str, document_paths: List[str],
                                 chunking_strategy: str, metadata_level: str,
                                 enable_multimodal: bool):
    """Background task for processing multiple documents."""
    try:
        results = []
        total_docs = len(document_paths)
        
        for i, doc_path in enumerate(document_paths):
            try:
                # Process each document
                result = document_processor.process_document(doc_path)
                
                # Enhance metadata if enabled
                if enable_multimodal:
                    multimodal_result = multimodal_processor.process_content(
                        doc_path,
                        Path(doc_path).suffix[1:],
                        result
                    )
                
                results.append({
                    "document_path": doc_path,
                    "success": result.success,
                    "chunks_count": len(result.chunks),
                    "quality_score": result.quality_score,
                    "processing_time": result.processing_time
                })
                
                logger.info(f"Batch {batch_id}: Document {i+1}/{total_docs} processed")
                
            except Exception as e:
                logger.error(f"Error processing document in batch: {e}")
                results.append({
                    "document_path": doc_path,
                    "success": False,
                    "error": str(e)
                })
        
        # Store batch results
        processing_status[batch_id] = {
            "status": "completed",
            "progress": 1.0,
            "message": "Batch processing completed",
            "result": {
                "batch_id": batch_id,
                "total_documents": total_docs,
                "successful": len([r for r in results if r["success"]]),
                "failed": len([r for r in results if not r["success"]]),
                "results": results
            }
        }
        
        logger.info(f"Batch {batch_id} processing completed")
    
    except Exception as e:
        logger.error(f"Error in batch processing {batch_id}: {e}")
        processing_status[batch_id] = {
            "status": "failed",
            "progress": 0.0,
            "message": f"Batch processing failed: {str(e)}",
            "result": {"error": str(e)}
        }


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={"error": "Resource not found", "detail": str(exc)}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors."""
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    
    # Run the API server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
