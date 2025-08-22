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

from .config import Config
from .processor import DocumentProcessor
from .multimodal import get_multimodal_processor
from .metadata_enhancement import get_metadata_enhancer

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
    export_mkdocs: Optional[bool] = True  # New field for MkDocs export


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
    export_mkdocs: Optional[bool] = True  # New field for MkDocs export


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
            request.enable_multimodal,
            request.export_mkdocs  # Pass MkDocs export flag
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
            request.enable_multimodal,
            request.export_mkdocs  # Pass MkDocs export flag
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
        # This would typically query a database
        # For now, return mock data
        documents = [
            {
                "document_id": "doc_123",
                "filename": "example.pdf",
                "status": "processed",
                "chunks_count": 5,
                "quality_score": 0.85,
                "processing_time": 2.5,
                "mkdocs_export": {
                    "success": True,
                    "pages_created": 5,
                    "output_directory": "output/mkdocs"
                }
            }
        ]
        
        return {
            "success": True,
            "total_documents": len(documents),
            "documents": documents
        }
    
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/mkdocs/build")
async def build_mkdocs_site():
    """Build the MkDocs site from processed documents."""
    try:
        # This would trigger a MkDocs build
        # For now, return success response
        return {
            "success": True,
            "message": "MkDocs site build started",
            "output_directory": "output/mkdocs",
            "site_url": "http://localhost:8000"
        }
    
    except Exception as e:
        logger.error(f"Error building MkDocs site: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mkdocs/status")
async def get_mkdocs_status():
    """Get the status of MkDocs export and build."""
    try:
        # Check if MkDocs output exists
        mkdocs_dir = Path("output/mkdocs")
        docs_dir = mkdocs_dir / "docs" if mkdocs_dir.exists() else None
        
        if docs_dir and docs_dir.exists():
            # Count markdown files
            md_files = list(docs_dir.rglob("*.md"))
            total_pages = len(md_files)
            
            # Check for config file
            config_exists = (mkdocs_dir / "mkdocs.yml").exists()
            
            return {
                "success": True,
                "mkdocs_exported": True,
                "total_pages": total_pages,
                "config_exists": config_exists,
                "output_directory": str(mkdocs_dir),
                "docs_directory": str(docs_dir),
                "can_build": config_exists and total_pages > 0
            }
        else:
            return {
                "success": True,
                "mkdocs_exported": False,
                "total_pages": 0,
                "config_exists": False,
                "output_directory": str(mkdocs_dir) if mkdocs_dir.exists() else "Not created",
                "docs_directory": "Not created",
                "can_build": False
            }
    
    except Exception as e:
        logger.error(f"Error getting MkDocs status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mkdocs/navigation")
async def get_mkdocs_navigation():
    """Get the current MkDocs navigation structure."""
    try:
        nav_file = Path("output/mkdocs/docs/_navigation.md")
        
        if nav_file.exists():
            with open(nav_file, 'r', encoding='utf-8') as f:
                navigation_content = f.read()
            
            return {
                "success": True,
                "navigation_exists": True,
                "navigation_content": navigation_content,
                "navigation_file": str(nav_file)
            }
        else:
            return {
                "success": True,
                "navigation_exists": False,
                "navigation_content": "",
                "navigation_file": str(nav_file)
            }
    
    except Exception as e:
        logger.error(f"Error getting MkDocs navigation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Background processing functions
async def process_document_background(document_id: str, document_path: str, 
                                    chunking_strategy: str, metadata_level: str, 
                                    enable_multimodal: bool, export_mkdocs: bool = True):
    """Background task for processing a single document."""
    try:
        # Update status to processing
        processing_status[document_id]["status"] = "processing"
        processing_status[document_id]["progress"] = 0.1
        processing_status[document_id]["message"] = "Starting document processing"
        
        # Process document with MkDocs export if enabled
        if export_mkdocs:
            result = document_processor.process_document_with_mkdocs(document_path, export_mkdocs=True)
        else:
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
            "multimodal_result": multimodal_result.to_dict() if enable_multimodal else None,
            "mkdocs_export": result.metadata.get('mkdocs_export', {})  # Include MkDocs export info
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
                                 enable_multimodal: bool, export_mkdocs: bool = True):
    """Background task for processing multiple documents."""
    
    try:
        results = []
        total_docs = len(document_paths)
        
        for i, doc_path in enumerate(document_paths):
            try:
                # Process each document with MkDocs export if enabled
                if export_mkdocs:
                    result = document_processor.process_document_with_mkdocs(doc_path, export_mkdocs=True)
                else:
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
                    "processing_time": result.processing_time,
                    "mkdocs_export": result.metadata.get('mkdocs_export', {})  # Include MkDocs export info
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
        processing_status[batch_id]["status"] = "completed"
        processing_status[batch_id]["progress"] = 1.0
        processing_status[batch_id]["message"] = "Batch processing completed"
        processing_status[batch_id]["result"] = {
            "total_documents": total_docs,
            "successful_documents": len([r for r in results if r["success"]]),
            "failed_documents": len([r for r in results if not r["success"]]),
            "results": results
        }
        
        logger.info(f"Batch {batch_id} processing completed successfully")
    
    except Exception as e:
        logger.error(f"Error in batch processing {batch_id}: {e}")
        processing_status[batch_id]["status"] = "failed"
        processing_status[batch_id]["progress"] = 0.0
        processing_status[batch_id]["message"] = f"Batch processing failed: {str(e)}"
        processing_status[batch_id]["result"] = {"error": str(e)}


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
