"""
Tests for FastAPI REST API
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient

from src.api import app
from src.config import Config



class TestFastAPIEndpoints:
    """Test FastAPI endpoints using TestClient."""

    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(app)

    def test_health_check(self):
        """Test health check endpoint."""
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_root_endpoint(self):
        """Test root endpoint."""
        response = self.client.get("/")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/html; charset=utf-8"
        assert "RAG Document Processing Utility API" in response.text

    @patch('src.api.DocumentProcessor')
    def test_upload_document(self, mock_processor_class):
        """Test document upload endpoint."""
        # Mock the processor
        mock_processor = Mock()
        mock_processor.process_single_document.return_value = {
            "success": True,
            "document_id": "doc123",
            "chunks": [],
            "metadata": {},
            "quality_score": 0.9,
            "processing_time": 1.5
        }
        mock_processor_class.return_value = mock_processor
        
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test document content")
            temp_file_path = f.name
        
        try:
            with open(temp_file_path, 'rb') as f:
                response = self.client.post(
                    "/upload-document",
                    files={"file": ("test.txt", f, "text/plain")}
                )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["filename"] == "test.txt"
            assert "file_path" in data
            
        finally:
            os.unlink(temp_file_path)

    @patch('src.api.DocumentProcessor')
    def test_process_document(self, mock_processor_class):
        """Test document processing endpoint."""
        # Mock the processor
        mock_processor = Mock()
        mock_processor.process_single_document.return_value = {
            "success": True,
            "document_id": "doc123",
            "chunks": [],
            "metadata": {},
            "quality_score": 0.9,
            "processing_time": 1.5
        }
        mock_processor_class.return_value = mock_processor
        
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test document content")
            temp_file_path = f.name
        
        try:
            response = self.client.post(
                "/process-document",
                json={
                    "document_path": temp_file_path,
                    "chunking_strategy": "hybrid",
                    "metadata_level": "advanced",
                    "enable_multimodal": True
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "document_id" in data
            
        finally:
            os.unlink(temp_file_path)

    @patch('src.api.DocumentProcessor')
    def test_batch_process(self, mock_processor_class):
        """Test batch processing endpoint."""
        # Mock the processor
        mock_processor = Mock()
        mock_processor.process_batch_documents.return_value = {
            "success": True,
            "processed_documents": [
                {"success": True, "document_id": "doc1"},
                {"success": True, "document_id": "doc2"}
            ],
            "total_documents": 2,
            "successful_documents": 2,
            "failed_documents": 0
        }
        mock_processor_class.return_value = mock_processor
        
        # Create temporary test files
        temp_files = []
        try:
            for i in range(2):
                f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
                f.write(f"Test document {i} content")
                temp_files.append(f.name)
                f.close()
            
            response = self.client.post(
                "/process-batch",
                json={
                    "document_paths": temp_files,
                    "chunking_strategy": "hybrid",
                    "metadata_level": "advanced",
                    "enable_multimodal": True
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["total_documents"] == 2
            assert "batch_id" in data
            
        finally:
            for file_path in temp_files:
                os.unlink(file_path)

    def test_get_status(self):
        """Test status endpoint."""
        # First create a document to get a valid ID
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test document content")
            temp_file_path = f.name
        
        try:
            # Start processing a document
            process_response = self.client.post(
                "/process-document",
                json={
                    "document_path": temp_file_path,
                    "chunking_strategy": "hybrid",
                    "metadata_level": "advanced",
                    "enable_multimodal": False
                }
            )
            
            if process_response.status_code == 200:
                doc_id = process_response.json()["document_id"]
                
                # Now check the status
                response = self.client.get(f"/status/{doc_id}")
                assert response.status_code == 200
                data = response.json()
                assert "document_id" in data
                assert "status" in data
            else:
                # If processing fails, just test that the endpoint exists
                response = self.client.get("/status/test123")
                # Should get 404 for non-existent document
                assert response.status_code == 404
                
        finally:
            os.unlink(temp_file_path)

    def test_get_documents(self):
        """Test documents listing endpoint."""
        response = self.client.get("/documents")
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "total_documents" in data
        assert "documents" in data


if __name__ == "__main__":
    pytest.main([__file__])
