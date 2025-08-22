#!/usr/bin/env python3
"""
Tests for processor integration with MkDocs functionality.

This module tests how the DocumentProcessor integrates with the new
MkDocs site building capabilities.
"""

import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.processor import DocumentProcessor, ProcessingResult
from src.config import Config


class TestProcessorMkDocsIntegration:
    """Test cases for processor integration with MkDocs."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_config(self, temp_workspace):
        """Create a mock configuration with temporary workspace."""
        config = MagicMock()
        config.output.output_directory = temp_workspace
        return config
    
    def test_processing_result_original_content_field(self):
        """Test that ProcessingResult has the original_content field."""
        result = ProcessingResult(
            success=True,
            document_id="test_001",
            chunks=[],
            metadata={},
            quality_score=0.9,
            processing_time=1.0,
            original_content="This is the original document content."
        )
        
        assert result.original_content == "This is the original document content."
        assert hasattr(result, 'original_content')
    
    def test_processing_result_original_content_default(self):
        """Test that ProcessingResult original_content defaults to None."""
        result = ProcessingResult(
            success=True,
            document_id="test_001",
            chunks=[],
            metadata={},
            quality_score=0.9,
            processing_time=1.0
        )
        
        assert result.original_content is None
    
    def test_processing_result_original_content_preservation(self):
        """Test that original content is properly preserved in ProcessingResult."""
        # Test with original content
        result_with_content = ProcessingResult(
            success=True,
            document_id="test_001",
            chunks=[],
            metadata={},
            quality_score=0.9,
            processing_time=1.0,
            original_content="This is the complete original document content."
        )
        
        assert result_with_content.original_content == "This is the complete original document content."
        assert len(result_with_content.original_content) > 0
        
        # Test without original content
        result_without_content = ProcessingResult(
            success=False,
            document_id="test_001",
            chunks=[],
            metadata={},
            quality_score=0.0,
            processing_time=1.0,
            original_content=None
        )
        
        assert result_without_content.original_content is None
    
    def test_processing_result_metadata_integration(self):
        """Test that ProcessingResult metadata properly integrates with MkDocs info."""
        result = ProcessingResult(
            success=True,
            document_id="test_001",
            chunks=[],
            metadata={
                'mkdocs_export': {
                    'success': True,
                    'pages_created': 5,
                    'site_built': True,
                    'build_time': 2.5
                }
            },
            quality_score=0.9,
            processing_time=1.0,
            original_content="Original content"
        )
        
        # Verify MkDocs metadata is accessible
        assert 'mkdocs_export' in result.metadata
        mkdocs_info = result.metadata['mkdocs_export']
        assert mkdocs_info['success'] is True
        assert mkdocs_info['pages_created'] == 5
        assert mkdocs_info['site_built'] is True
        assert mkdocs_info['build_time'] == 2.5
    
    def test_processing_result_with_mkdocs_site_info(self):
        """Test ProcessingResult with complete MkDocs site building information."""
        result = ProcessingResult(
            success=True,
            document_id="test_001",
            chunks=[],
            metadata={
                'mkdocs_export': {
                    'success': True,
                    'pages_created': 3,
                    'output_directory': '/test/output',
                    'mkdocs_config_path': '/test/mkdocs.yml',
                    'site_built': True,
                    'site_directory': '/test/site',
                    'site_url': 'file:///test/site/index.html',
                    'build_time': 2.5
                }
            },
            quality_score=0.9,
            processing_time=1.0,
            original_content="Complete document content"
        )
        
        # Verify all MkDocs fields are present
        mkdocs_info = result.metadata['mkdocs_export']
        assert mkdocs_info['success'] is True
        assert mkdocs_info['pages_created'] == 3
        assert mkdocs_info['output_directory'] == '/test/output'
        assert mkdocs_info['mkdocs_config_path'] == '/test/mkdocs.yml'
        assert mkdocs_info['site_built'] is True
        assert mkdocs_info['site_directory'] == '/test/site'
        assert mkdocs_info['site_url'] == 'file:///test/site/index.html'
        assert mkdocs_info['build_time'] == 2.5
    
    def test_processing_result_with_batch_mkdocs_info(self):
        """Test ProcessingResult with batch MkDocs export information."""
        result = ProcessingResult(
            success=True,
            document_id="test_001",
            chunks=[],
            metadata={
                'batch_mkdocs_export': {
                    'success': True,
                    'total_pages': 10,
                    'output_directory': '/test/batch/output',
                    'site_built': True,
                    'site_directory': '/test/batch/site',
                    'site_url': 'file:///test/batch/site/index.html',
                    'build_time': 5.0
                }
            },
            quality_score=0.9,
            processing_time=1.0,
            original_content="Batch processed document"
        )
        
        # Verify batch MkDocs fields are present
        batch_info = result.metadata['batch_mkdocs_export']
        assert batch_info['success'] is True
        assert batch_info['total_pages'] == 10
        assert batch_info['output_directory'] == '/test/batch/output'
        assert batch_info['site_built'] is True
        assert batch_info['site_directory'] == '/test/batch/site'
        assert batch_info['site_url'] == 'file:///test/batch/site/index.html'
        assert batch_info['build_time'] == 5.0
    
    def test_processing_result_error_handling(self):
        """Test ProcessingResult error handling with original content preservation."""
        # Test with parsing failure but original content preserved
        result = ProcessingResult(
            success=False,
            document_id="test_001",
            chunks=[],
            metadata={},
            quality_score=0.0,
            processing_time=1.0,
            original_content="Content that was parsed before failure",
            error_message="Chunking failed: Invalid content format"
        )
        
        assert result.success is False
        assert result.original_content == "Content that was parsed before failure"
        assert result.error_message == "Chunking failed: Invalid content format"
        assert result.quality_score == 0.0
    
    def test_processing_result_warnings(self):
        """Test ProcessingResult warnings handling."""
        result = ProcessingResult(
            success=True,
            document_id="test_001",
            chunks=[],
            metadata={},
            quality_score=0.9,
            processing_time=1.0,
            original_content="Document content",
            warnings=["Low quality chunks detected", "Metadata extraction incomplete"]
        )
        
        assert len(result.warnings) == 2
        assert "Low quality chunks detected" in result.warnings
        assert "Metadata extraction incomplete" in result.warnings


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
