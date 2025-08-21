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
        data = response.json()
        assert "message" in data
        assert "version" in data

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
                    "/upload",
                    files={"file": ("test.txt", f, "text/plain")}
                )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["document_id"] == "doc123"
            
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
            with open(temp_file_path, 'rb') as f:
                response = self.client.post(
                    "/process",
                    files={"file": ("test.txt", f, "text/plain")}
                )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["document_id"] == "doc123"
            
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
            
            files = []
            for file_path in temp_files:
                with open(file_path, 'rb') as f:
                    files.append(("files", (os.path.basename(file_path), f, "text/plain")))
            
            response = self.client.post("/batch-process", files=files)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["total_documents"] == 2
            assert data["successful_documents"] == 2
            
        finally:
            for file_path in temp_files:
                os.unlink(file_path)

    def test_get_status(self):
        """Test status endpoint."""
        response = self.client.get("/status")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "uptime" in data

    def test_get_metrics(self):
        """Test metrics endpoint."""
        response = self.client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "metrics" in data


if __name__ == "__main__":
    pytest.main([__file__])
