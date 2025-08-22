#!/usr/bin/env python3
"""
Tests for MkDocs exporter functionality.

This module tests the MkDocs exporter's ability to:
1. Export documents with original content preservation
2. Build static HTML sites
3. Handle various document types and configurations
"""

import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.mkdocs_exporter import (
    MkDocsExporter, 
    MkDocsPage, 
    MkDocsSection, 
    MkDocsExportResult,
    get_mkdocs_exporter
)
from src.chunkers import DocumentChunk
from src.config import Config


class TestMkDocsExporter:
    """Test cases for MkDocsExporter class."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary output directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_config(self, temp_output_dir):
        """Create a mock configuration with temporary output directory."""
        config = MagicMock()
        config.output.output_directory = temp_output_dir
        return config
    
    @pytest.fixture
    def exporter(self, mock_config):
        """Create an MkDocsExporter instance for testing."""
        return MkDocsExporter(mock_config)
    
    @pytest.fixture
    def sample_chunks(self):
        """Create sample document chunks for testing."""
        return [
            DocumentChunk(
                chunk_id="chunk_001",
                content="This is the first chunk of the document.",
                chunk_type="paragraph",
                chunk_index=0,
                quality_score=0.9,
                metadata={"word_count": 10}
            ),
            DocumentChunk(
                chunk_id="chunk_002",
                content="This is the second chunk of the document.",
                chunk_type="paragraph",
                chunk_index=1,
                quality_score=0.85,
                metadata={"word_count": 10}
            )
        ]
    
    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata for testing."""
        return {
            "title": "Test Document",
            "quality_score": 0.875,
            "timestamp": "2024-01-01T00:00:00Z",
            "source_filename": "test_document.txt"
        }
    
    @pytest.fixture
    def original_content(self):
        """Create sample original content for testing."""
        return "This is the complete test document. It contains multiple sentences and should be preserved in its original form."

    def test_exporter_initialization(self, exporter, mock_config):
        """Test that the exporter initializes correctly."""
        assert exporter.config == mock_config
        assert exporter.output_dir == Path(mock_config.output.output_directory) / "mkdocs"
        assert exporter.docs_dir == exporter.output_dir / "docs"
        assert exporter.logger is not None

    def test_mkdocs_page_creation(self):
        """Test MkDocsPage dataclass creation."""
        page = MkDocsPage(
            title="Test Page",
            content="Test content",
            file_path="test/path.md",
            metadata={"key": "value"},
            order=1
        )
        
        assert page.title == "Test Page"
        assert page.content == "Test content"
        assert page.file_path == "test/path.md"
        assert page.metadata["key"] == "value"
        assert page.order == 1

    def test_mkdocs_section_creation(self, sample_chunks):
        """Test MkDocsSection dataclass creation."""
        pages = [
            MkDocsPage(
                title=f"Page {i}",
                content=f"Content {i}",
                file_path=f"page_{i}.md",
                metadata={},
                order=i
            )
            for i in range(3)
        ]
        
        section = MkDocsSection(
            name="test_section",
            title="Test Section",
            pages=pages,
            order=0
        )
        
        assert section.name == "test_section"
        assert section.title == "Test Section"
        assert len(section.pages) == 3
        assert section.order == 0

    def test_mkdocs_export_result_creation(self):
        """Test MkDocsExportResult dataclass creation."""
        result = MkDocsExportResult(
            success=True,
            pages_created=5,
            sections_created=1,
            output_directory="/test/output",
            mkdocs_config_path="/test/mkdocs.yml",
            navigation_file_path="/test/navigation.md",
            site_built=True,
            site_directory="/test/site",
            site_url="file:///test/site/index.html",
            build_time=2.5
        )
        
        assert result.success is True
        assert result.pages_created == 5
        assert result.sections_created == 1
        assert result.site_built is True
        assert result.build_time == 2.5
        assert result.errors == []
        assert result.warnings == []

    def test_generate_section_name(self, exporter):
        """Test section name generation from filename."""
        # Test basic filename
        assert exporter._generate_section_name("document.txt") == "document"
        
        # Test filename with spaces
        assert exporter._generate_section_name("My Document.pdf") == "my_document"
        
        # Test filename with special characters
        assert exporter._generate_section_name("doc@#$%^&*()_+.txt") == "doc"
        
        # Test filename with multiple dots
        assert exporter._generate_section_name("file.name.txt") == "file_name"

    def test_generate_section_title(self, exporter):
        """Test section title generation."""
        metadata = {"title": "Custom Title"}
        
        # Test with metadata title
        assert exporter._generate_section_title("document.txt", metadata) == "Custom Title"
        
        # Test fallback to filename
        assert exporter._generate_section_title("my_document.txt", {}) == "My Document"
        
        # Test with underscores and dashes
        assert exporter._generate_section_title("my-document_file.txt", {}) == "My Document File"

    def test_extract_title_from_chunk(self, exporter):
        """Test title extraction from chunk content."""
        # Test with markdown header
        content = "# Main Title\n\nSome content here."
        assert exporter._extract_title_from_chunk(content, 0) == "Main Title"
        
        # Test with first sentence
        content = "This is the first sentence. This is the second sentence."
        assert exporter._extract_title_from_chunk(content, 1) == "This is the first sentence"
        
        # Test with long sentence (should be truncated)
        long_content = "This is a very long sentence that should be truncated to fit within the title length limits and not be too long for display purposes."
        title = exporter._extract_title_from_chunk(long_content, 2)
        assert len(title) <= 60
        assert title.endswith("...")
        
        # Test fallback
        content = ""
        assert exporter._extract_title_from_chunk(content, 3) == "Chunk 4"

    def test_format_chunk_content(self, exporter):
        """Test chunk content formatting."""
        content = "Test chunk content"
        metadata = {
            "chunk_type": "paragraph",
            "quality_score": 0.85,
            "word_count": 3,
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
        formatted = exporter._format_chunk_content(content, metadata)
        
        assert "---" in formatted
        assert "title:" in formatted
        assert "chunk_type: paragraph" in formatted
        assert "quality_score: 0.850" in formatted
        assert "word_count: 3" in formatted
        assert "Test chunk content" in formatted

    def test_create_original_page(self, exporter, sample_metadata, original_content):
        """Test original page creation."""
        page = exporter._create_original_page(
            document_id="test_doc_001",
            metadata=sample_metadata,
            original_content=original_content,
            source_filename="test_document.txt"
        )
        
        assert page.title == "📄 Test Document (Complete Document)"
        assert page.file_path == "test_doc_001/original_document.md"
        assert page.order == -1
        assert page.metadata["document_type"] == "original_complete"
        assert page.metadata["word_count"] == 18
        assert page.metadata["character_count"] == 112
        assert "Complete Document" in page.content

    def test_format_original_content(self, exporter, sample_metadata, original_content):
        """Test original content formatting."""
        formatted = exporter._format_original_content(
            original_content, 
            sample_metadata, 
            "test_document.txt"
        )
        
        assert "---" in formatted
        assert "title:" in formatted
        assert "document_type: \"original_complete\"" in formatted
        assert "word_count: 18" in formatted
        assert "character_count: 112" in formatted
        assert "Complete Document" in formatted
        assert "Document Summary" in formatted
        assert "RAGPrep Document Processing Utility" in formatted

    def test_convert_chunks_to_pages(self, exporter, sample_chunks, sample_metadata):
        """Test conversion of chunks to pages."""
        pages = exporter._convert_chunks_to_pages(
            sample_chunks, 
            "test_doc_001", 
            sample_metadata
        )
        
        assert len(pages) == 2
        
        # Check first page
        assert pages[0].title == "This is the first chunk of the document"
        assert pages[0].file_path == "test_doc_001/chunk_000.md"
        assert pages[0].order == 0
        assert pages[0].metadata["chunk_id"] == "chunk_001"
        assert pages[0].metadata["chunk_type"] == "paragraph"
        assert pages[0].metadata["quality_score"] == 0.9
        
        # Check second page
        assert pages[1].title == "This is the second chunk of the document"
        assert pages[1].file_path == "test_doc_001/chunk_001.md"
        assert pages[1].order == 1

    def test_find_mkdocs_command(self, exporter):
        """Test MkDocs command detection."""
        # Mock successful command execution
        with patch('subprocess.run') as mock_subprocess:
            mock_subprocess.return_value.returncode = 0
            cmd = exporter._find_mkdocs_command()
            assert cmd == "mkdocs"

    def test_build_mkdocs_site_success(self, exporter):
        """Test successful MkDocs site building."""
        # Mock command detection
        with patch.object(exporter, '_find_mkdocs_command', return_value="mkdocs"):
            with patch('subprocess.run') as mock_run:
                mock_run.return_value.returncode = 0
                mock_run.return_value.stderr = ""
                
                result = exporter._build_mkdocs_site()
                
                assert result['success'] is True
                assert 'site_directory' in result
                assert 'site_url' in result
                assert result['build_time'] >= 0  # Can be 0 for very fast operations
                assert len(result['errors']) == 0

    def test_build_mkdocs_site_failure(self, exporter):
        """Test failed MkDocs site building."""
        # Mock command detection
        with patch.object(exporter, '_find_mkdocs_command', return_value="mkdocs"):
            with patch('subprocess.run') as mock_run:
                mock_run.return_value.returncode = 1
                mock_run.return_value.stderr = "Build failed: configuration error"
                
                result = exporter._build_mkdocs_site()
                
                assert result['success'] is False
                assert len(result['errors']) > 0
                assert "Build failed: configuration error" in result['errors'][0]

    def test_export_document_without_site_building(self, exporter, sample_chunks, sample_metadata, original_content):
        """Test document export without site building."""
        result = exporter.export_document(
            document_id="test_doc_001",
            chunks=sample_chunks,
            metadata=sample_metadata,
            source_filename="test_document.txt",
            original_content=original_content,
            build_site=False
        )
        
        assert result.success is True
        assert result.pages_created == 3  # 2 chunks + 1 original
        assert result.sections_created == 1
        assert result.site_built is False
        assert result.site_directory == ""
        assert result.site_url == ""

    @patch.object(MkDocsExporter, '_build_mkdocs_site')
    def test_export_document_with_site_building(self, mock_build, exporter, sample_chunks, sample_metadata, original_content):
        """Test document export with site building."""
        # Mock successful site building
        mock_build.return_value = {
            'success': True,
            'site_directory': '/test/site',
            'site_url': 'file:///test/site/index.html',
            'build_time': 2.5,
            'errors': []
        }
        
        result = exporter.export_document(
            document_id="test_doc_001",
            chunks=sample_chunks,
            metadata=sample_metadata,
            source_filename="test_document.txt",
            original_content=original_content,
            build_site=True
        )
        
        assert result.success is True
        assert result.site_built is True
        assert result.site_directory == "/test/site"
        assert result.site_url == "file:///test/site/index.html"
        assert result.build_time == 2.5

    def test_export_document_error_handling(self, exporter):
        """Test error handling in document export."""
        # Test with invalid chunks (should return error result)
        result = exporter.export_document(
            document_id="test_doc_001",
            chunks="invalid_chunks",  # This will cause an error
            metadata={},
            source_filename="test_document.txt"
        )
        
        # Should return error result instead of raising exception
        assert result.success is False
        assert len(result.errors) > 0
        assert "has no attribute 'content'" in result.errors[0]

    def test_write_pages_to_disk(self, exporter, temp_output_dir):
        """Test writing pages to disk."""
        # Create a test section
        pages = [
            MkDocsPage(
                title="Original Document",
                content="Original content",
                file_path="test_doc/original_document.md",
                metadata={"document_type": "original_complete"},
                order=-1
            ),
            MkDocsPage(
                title="Chunk 1",
                content="Chunk content",
                file_path="test_doc/chunk_000.md",
                metadata={},
                order=0
            )
        ]
        
        section = MkDocsSection(
            name="test_doc",
            title="Test Document",
            pages=pages,
            order=0
        )
        
        # Write pages to disk
        exporter._write_pages_to_disk(section)
        
        # Check that files were created
        site_dir = exporter.docs_dir / "test_doc"
        assert (site_dir / "original_document.md").exists()
        assert (site_dir / "chunk_000.md").exists()

    def test_update_navigation(self, exporter, temp_output_dir):
        """Test navigation update."""
        # Create a test section
        pages = [
            MkDocsPage(
                title="Original Document",
                content="Original content",
                file_path="test_doc/original_document.md",
                metadata={"document_type": "original_complete"},
                order=-1
            ),
            MkDocsPage(
                title="Chunk 1",
                content="Chunk content",
                file_path="test_doc/chunk_000.md",
                metadata={},
                order=0
            )
        ]
        
        section = MkDocsSection(
            name="test_doc",
            title="Test Document",
            pages=pages,
            order=0
        )
        
        # Update navigation
        exporter._update_navigation(section)
        
        # Check that navigation file was created
        nav_file = exporter.docs_dir / "_navigation.md"
        assert nav_file.exists()
        
        # Check navigation content
        nav_content = nav_file.read_text()
        assert "## Test Document" in nav_content
        assert "Original Document" in nav_content
        assert "Chunk 1" in nav_content

    def test_generate_mkdocs_config(self, exporter, temp_output_dir):
        """Test MkDocs configuration generation."""
        exporter._generate_mkdocs_config()
        
        config_file = exporter.output_dir / "mkdocs.yml"
        assert config_file.exists()
        
        config_content = config_file.read_text()
        assert "site_name: RAG Document Processing Utility" in config_content
        assert "theme:" in config_content
        assert "name: material" in config_content
        assert "plugins:" in config_content

    def test_generate_index_page(self, exporter, temp_output_dir):
        """Test index page generation."""
        exporter._generate_index_page()
        
        index_file = exporter.docs_dir / "index.md"
        assert index_file.exists()
        
        index_content = index_file.read_text()
        assert "# RAG Document Processing Utility" in index_content
        assert "Welcome to the processed documents" in index_content
        assert "Navigation" in index_content

    def test_export_batch(self, exporter, sample_chunks, sample_metadata, original_content):
        """Test batch document export."""
        documents = [
            {
                'document_id': 'doc_001',
                'chunks': sample_chunks,
                'metadata': sample_metadata,
                'source_filename': 'test_document.txt',
                'original_content': original_content
            }
        ]
        
        # Mock site building
        with patch.object(exporter, '_build_mkdocs_site') as mock_build:
            mock_build.return_value = {
                'success': True,
                'site_directory': '/test/site',
                'site_url': 'file:///test/site/index.html',
                'build_time': 2.5,
                'errors': []
            }
            
            result = exporter.export_batch(documents, build_site=True)
            
            assert result.success is True
            assert result.pages_created == 3  # 2 chunks + 1 original
            assert result.sections_created == 1
            assert result.site_built is True

    def test_get_mkdocs_exporter(self):
        """Test factory function for getting MkDocs exporter."""
        exporter = get_mkdocs_exporter()
        assert isinstance(exporter, MkDocsExporter)
        assert exporter.config is not None


class TestMkDocsExporterIntegration:
    """Integration tests for MkDocs exporter."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for integration testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_full_export_workflow(self, temp_workspace):
        """Test the complete export workflow."""
        # Create a temporary config
        config = Config()
        config.output.output_directory = temp_workspace
        
        exporter = MkDocsExporter(config)
        
        # Create test data
        chunks = [
            DocumentChunk(
                chunk_id="chunk_001",
                content="First chunk content",
                chunk_type="paragraph",
                chunk_index=0,
                quality_score=0.9,
                metadata={"word_count": 3}
            )
        ]
        
        metadata = {
            "title": "Integration Test Document",
            "quality_score": 0.9,
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
        original_content = "This is the complete document content for integration testing."
        
        # Export document
        result = exporter.export_document(
            document_id="integration_test",
            chunks=chunks,
            metadata=metadata,
            source_filename="integration_test.txt",
            original_content=original_content,
            build_site=False  # Skip site building for faster test
        )
        
        # Verify results
        assert result.success is True
        assert result.pages_created == 2  # 1 chunk + 1 original
        assert result.sections_created == 1
        
        # Check file structure
        docs_dir = exporter.docs_dir
        assert docs_dir.exists()
        
        # Check that markdown files were created
        test_doc_dir = docs_dir / "integration_test"
        assert test_doc_dir.exists()
        assert (test_doc_dir / "original_document.md").exists()
        assert (test_doc_dir / "chunk_000.md").exists()
        
        # Check navigation
        nav_file = docs_dir / "_navigation.md"
        assert nav_file.exists()
        
        # Check configuration
        config_file = exporter.output_dir / "mkdocs.yml"
        assert config_file.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
