"""
Tests for document chunkers
"""
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from chunkers import (
    DocumentChunker, FixedSizeChunker, StructuralChunker, SemanticChunker,
    HybridChunker, DocumentChunkerFactory, DocumentChunk, ChunkingResult
)
from parsers import ParsedContent
from config import Config


class TestDocumentChunk:
    """Test DocumentChunk dataclass."""
    
    def test_document_chunk_creation(self):
        """Test creating DocumentChunk instances."""
        chunk = DocumentChunk(
            chunk_id="test_chunk_001",
            content="Test content",
            metadata={"test": "data"},
            chunk_type="test",
            chunk_index=0
        )
        
        assert chunk.chunk_id == "test_chunk_001"
        assert chunk.content == "Test content"
        assert chunk.metadata["test"] == "data"
        assert chunk.chunk_type == "test"
        assert chunk.chunk_index == 0
        assert chunk.quality_score == 0.0
        assert chunk.child_chunk_ids == []
        assert chunk.relationships == []
    
    def test_document_chunk_defaults(self):
        """Test DocumentChunk default values."""
        chunk = DocumentChunk(
            chunk_id="test_chunk_002",
            content="Test content",
            metadata={},
            chunk_type="test",
            chunk_index=1
        )
        
        assert chunk.parent_chunk_id is None
        assert chunk.child_chunk_ids == []
        assert chunk.relationships == []


class TestChunkingResult:
    """Test ChunkingResult dataclass."""
    
    def test_chunking_result_creation(self):
        """Test creating ChunkingResult instances."""
        result = ChunkingResult(
            success=True,
            chunks=[],
            chunking_strategy="test",
            total_chunks=0,
            processing_time=1.5,
            metadata={"test": "data"},
            errors=[],
            warnings=[]
        )
        
        assert result.success is True
        assert result.chunking_strategy == "test"
        assert result.total_chunks == 0
        assert result.processing_time == 1.5
        assert result.metadata["test"] == "data"


class TestDocumentChunker:
    """Test base document chunker functionality."""
    
    def test_abstract_base_class(self):
        """Test that DocumentChunker is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            DocumentChunker(Mock(spec=Config))
    
    def test_chunk_id_generation(self):
        """Test chunk ID generation."""
        # Create a concrete chunker class for testing
        class TestChunker(DocumentChunker):
            def chunk_document(self, parsed_content: ParsedContent) -> ChunkingResult:
                return ChunkingResult(
                    success=True,
                    chunks=[],
                    chunking_strategy="test",
                    total_chunks=0,
                    processing_time=0.0,
                    metadata={},
                    errors=[],
                    warnings=[]
                )
        
        mock_config = Mock(spec=Config)
        chunker = TestChunker(mock_config)
        
        chunk_id = chunker._generate_chunk_id("test content", 0)
        assert chunk_id.startswith("chunk_0000_")
        assert len(chunk_id) > 10
    
    def test_chunk_metadata_extraction(self):
        """Test chunk metadata extraction."""
        class TestChunker(DocumentChunker):
            def chunk_document(self, parsed_content: ParsedContent) -> ChunkingResult:
                return ChunkingResult(
                    success=True,
                    chunks=[],
                    chunking_strategy="test",
                    total_chunks=0,
                    processing_time=0.0,
                    metadata={},
                    errors=[],
                    warnings=[]
                )
        
        mock_config = Mock(spec=Config)
        chunker = TestChunker(mock_config)
        
        metadata = chunker._extract_chunk_metadata("Test content\nSecond line", 0, "test")
        
        assert metadata["chunk_index"] == 0
        assert metadata["chunk_type"] == "test"
        assert metadata["content_length"] == 22
        assert metadata["word_count"] == 4
        assert metadata["sentence_count"] == 1
        assert metadata["paragraph_count"] == 2
        assert metadata["chunker"] == "TestChunker"
        assert "timestamp" in metadata
    
    def test_chunk_quality_assessment(self):
        """Test chunk quality assessment."""
        class TestChunker(DocumentChunker):
            def chunk_document(self, parsed_content: ParsedContent) -> ChunkingResult:
                return ChunkingResult(
                    success=True,
                    chunks=[],
                    chunking_strategy="test",
                    total_chunks=0,
                    processing_time=0.0,
                    metadata={},
                    errors=[],
                    warnings=[]
                )
        
        mock_config = Mock(spec=Config)
        chunker = TestChunker(mock_config)
        
        # Test empty content
        quality = chunker._assess_chunk_quality("")
        assert quality == 0.0
        
        # Test good content
        quality = chunker._assess_chunk_quality("This is a good chunk with multiple words and a complete sentence.")
        assert 0.0 <= quality <= 1.0


class TestFixedSizeChunker:
    """Test fixed-size chunking strategy."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = Mock(spec=Config)
        self.mock_chunking_config = Mock()
        self.mock_chunking_config.max_chunk_size = 100
        self.mock_chunking_config.min_chunk_size = 20
        self.mock_chunking_config.overlap_size = 10
        self.mock_config.get_chunking_config.return_value = self.mock_chunking_config
        
        self.chunker = FixedSizeChunker(self.mock_config)
    
    def test_chunk_document_empty_content(self):
        """Test chunking empty content."""
        mock_parsed_content = Mock(spec=ParsedContent)
        mock_parsed_content.text_content = ""
        
        result = self.chunker.chunk_document(mock_parsed_content)
        
        assert result.success is False
        assert result.chunks == []
        assert "Empty document content" in result.errors
    
    def test_chunk_document_small_content(self):
        """Test chunking content smaller than max chunk size."""
        mock_parsed_content = Mock(spec=ParsedContent)
        mock_parsed_content.text_content = "This is a small document that should fit in one chunk."
        
        result = self.chunker.chunk_document(mock_parsed_content)
        
        assert result.success is True
        assert len(result.chunks) == 1
        assert result.chunks[0].content == mock_parsed_content.text_content
        assert result.chunks[0].chunk_type == "fixed_size"
    
    def test_chunk_document_large_content(self):
        """Test chunking content larger than max chunk size."""
        mock_parsed_content = Mock(spec=ParsedContent)
        # Create content larger than max chunk size
        mock_parsed_content.text_content = "This is a sentence. " * 20  # ~400 characters
        
        result = self.chunker.chunk_document(mock_parsed_content)
        
        assert result.success is True
        assert len(result.chunks) > 1
        assert all(chunk.chunk_type == "fixed_size" for chunk in result.chunks)
        
        # Check that chunks respect size limits
        for chunk in result.chunks:
            assert len(chunk.content) <= self.mock_chunking_config.max_chunk_size
            assert len(chunk.content) >= self.mock_chunking_config.min_chunk_size
    
    def test_sentence_boundary_detection(self):
        """Test sentence boundary detection."""
        mock_parsed_content = Mock(spec=ParsedContent)
        mock_parsed_content.text_content = "First sentence. Second sentence. Third sentence."
        
        result = self.chunker.chunk_document(mock_parsed_content)
        
        assert result.success is True
        # Should respect sentence boundaries when possible


class TestStructuralChunker:
    """Test structural chunking strategy."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = Mock(spec=Config)
        self.mock_chunking_config = Mock()
        self.mock_chunking_config.max_chunk_size = 1000
        self.mock_chunking_config.min_chunk_size = 50
        self.mock_config.get_chunking_config.return_value = self.mock_chunking_config
        
        self.chunker = StructuralChunker(self.mock_config)
    
    def test_chunk_by_pages(self):
        """Test chunking by pages."""
        mock_parsed_content = Mock(spec=ParsedContent)
        mock_parsed_content.text_content = "Page content"
        mock_parsed_content.structured_content = {
            "pages": [
                {
                    "text_blocks": [
                        {
                            "lines": [
                                {
                                    "spans": [{"text": "Page 1 content"}]
                                }
                            ]
                        }
                    ]
                },
                {
                    "text_blocks": [
                        {
                            "lines": [
                                {
                                    "spans": [{"text": "Page 2 content"}]
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        
        result = self.chunker.chunk_document(mock_parsed_content)
        
        assert result.success is True
        assert len(result.chunks) == 2
        assert result.chunks[0].chunk_type == "page"
        assert result.chunks[1].chunk_type == "page"
        assert "Page 1 content" in result.chunks[0].content
        assert "Page 2 content" in result.chunks[1].content
    
    def test_chunk_by_paragraphs(self):
        """Test chunking by paragraphs."""
        mock_parsed_content = Mock(spec=ParsedContent)
        mock_parsed_content.text_content = "Paragraph content"
        mock_parsed_content.structured_content = {
            "paragraphs": [
                {"text": "First paragraph", "style": "Normal"},
                {"text": "Second paragraph", "style": "Heading1"}
            ]
        }
        
        result = self.chunker.chunk_document(mock_parsed_content)
        
        assert result.success is True
        assert len(result.chunks) == 2
        assert result.chunks[0].chunk_type == "paragraph"
        assert result.chunks[1].chunk_type == "paragraph"
        assert result.chunks[0].metadata["style"] == "Normal"
        assert result.chunks[1].metadata["style"] == "Heading1"
    
    def test_chunk_by_headings(self):
        """Test chunking by headings."""
        mock_parsed_content = Mock(spec=ParsedContent)
        mock_parsed_content.text_content = "Heading content"
        mock_parsed_content.structured_content = {
            "headings": [
                {"text": "Main Heading", "level": 1},
                {"text": "Sub Heading", "level": 2}
            ]
        }
        
        result = self.chunker.chunk_document(mock_parsed_content)
        
        assert result.success is True
        # Should create sections based on headings
    
    def test_fallback_to_fixed_size(self):
        """Test fallback to fixed-size chunking when no structural elements."""
        mock_parsed_content = Mock(spec=ParsedContent)
        mock_parsed_content.text_content = "Simple text content without structure"
        mock_parsed_content.structured_content = {}
        
        with patch('chunkers.FixedSizeChunker') as mock_fixed_chunker:
            mock_result = Mock(spec=ChunkingResult)
            mock_result.success = True
            mock_result.chunks = []
            mock_fixed_chunker.return_value.chunk_document.return_value = mock_result
            
            result = self.chunker.chunk_document(mock_parsed_content)
            
            # Should call fixed-size chunker
            mock_fixed_chunker.return_value.chunk_document.assert_called_once()


class TestSemanticChunker:
    """Test semantic chunking strategy."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = Mock(spec=Config)
        self.mock_chunking_config = Mock()
        self.mock_chunking_config.max_chunk_size = 200
        self.mock_chunking_config.min_chunk_size = 50
        self.mock_chunking_config.semantic_threshold = 0.7
        self.mock_config.get_chunking_config.return_value = self.mock_chunking_config
        
        self.chunker = SemanticChunker(self.mock_config)
    
    def test_chunk_by_semantic_boundaries(self):
        """Test semantic boundary detection."""
        mock_parsed_content = Mock(spec=ParsedContent)
        mock_parsed_content.text_content = "First paragraph with content.\n\nSecond paragraph with different content.\n\nThird paragraph."
        
        result = self.chunker.chunk_document(mock_parsed_content)
        
        assert result.success is True
        assert len(result.chunks) > 0
        assert all(chunk.chunk_type == "semantic" for chunk in result.chunks)


class TestHybridChunker:
    """Test hybrid chunking strategy."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = Mock(spec=Config)
        self.mock_chunking_config = Mock()
        self.mock_chunking_config.max_chunk_size = 1000
        self.mock_chunking_config.min_chunk_size = 50
        self.mock_config.get_chunking_config.return_value = self.mock_chunking_config
        
        self.chunker = HybridChunker(self.mock_config)
    
    def test_hybrid_strategy_structural_first(self):
        """Test hybrid strategy using structural chunking first."""
        mock_parsed_content = Mock(spec=ParsedContent)
        mock_parsed_content.text_content = "Structured content"
        
        # Mock structural chunker to return high-quality chunks
        with patch.object(self.chunker.structural_chunker, 'chunk_document') as mock_structural:
            mock_result = Mock(spec=ChunkingResult)
            mock_result.success = True
            mock_result.chunks = [
                Mock(spec=DocumentChunk, quality_score=0.9),
                Mock(spec=DocumentChunk, quality_score=0.8)
            ]
            mock_result.warnings = []
            mock_structural.return_value = mock_result
            
            result = self.chunker.chunk_document(mock_parsed_content)
            
            assert result.success is True
            assert result.chunking_strategy == "hybrid_structural"
            assert len(result.chunks) == 2
    
    def test_hybrid_strategy_semantic_fallback(self):
        """Test hybrid strategy falling back to semantic chunking."""
        mock_parsed_content = Mock(spec=ParsedContent)
        mock_parsed_content.text_content = "Content for chunking"
        
        # Mock structural chunker to return low-quality chunks
        with patch.object(self.chunker.structural_chunker, 'chunk_document') as mock_structural:
            mock_structural_result = Mock(spec=ChunkingResult)
            mock_structural_result.success = True
            mock_structural_result.chunks = [
                Mock(spec=DocumentChunk, quality_score=0.5),
                Mock(spec=DocumentChunk, quality_score=0.4)
            ]
            mock_structural.return_value = mock_structural_result
            
            # Mock semantic chunker to return medium-quality chunks
            with patch.object(self.chunker.semantic_chunker, 'chunk_document') as mock_semantic:
                mock_semantic_result = Mock(spec=ChunkingResult)
                mock_semantic_result.success = True
                mock_semantic_result.chunks = [
                    Mock(spec=DocumentChunk, quality_score=0.7),
                    Mock(spec=DocumentChunk, quality_score=0.6)
                ]
                mock_semantic_result.warnings = []
                mock_semantic.return_value = mock_semantic_result
                
                result = self.chunker.chunk_document(mock_parsed_content)
                
                assert result.success is True
                assert result.chunking_strategy == "hybrid_semantic"
                assert len(result.chunks) == 2
    
    def test_hybrid_strategy_fixed_fallback(self):
        """Test hybrid strategy falling back to fixed-size chunking."""
        mock_parsed_content = Mock(spec=ParsedContent)
        mock_parsed_content.text_content = "Content for chunking"
        
        # Mock both structural and semantic chunkers to fail
        with patch.object(self.chunker.structural_chunker, 'chunk_document') as mock_structural:
            mock_structural.return_value = Mock(spec=ChunkingResult, success=False)
            
            with patch.object(self.chunker.semantic_chunker, 'chunk_document') as mock_semantic:
                mock_semantic.return_value = Mock(spec=ChunkingResult, success=False)
                
                # Mock fixed chunker to succeed
                with patch.object(self.chunker.fixed_chunker, 'chunk_document') as mock_fixed:
                    mock_fixed_result = Mock(spec=ChunkingResult)
                    mock_fixed_result.success = True
                    mock_fixed_result.chunks = [Mock(spec=DocumentChunk)]
                    mock_fixed_result.errors = []
                    mock_fixed_result.warnings = []
                    mock_fixed.return_value = mock_fixed_result
                    
                    result = self.chunker.chunk_document(mock_parsed_content)
                    
                    assert result.success is True
                    assert result.chunking_strategy == "hybrid_fixed"
                    assert len(result.chunks) == 1


class TestDocumentChunkerFactory:
    """Test document chunker factory."""
    
    def test_create_fixed_chunker(self):
        """Test creating fixed-size chunker."""
        mock_config = Mock(spec=Config)
        
        chunker = DocumentChunkerFactory.create_chunker("fixed", mock_config)
        
        assert isinstance(chunker, FixedSizeChunker)
    
    def test_create_structural_chunker(self):
        """Test creating structural chunker."""
        mock_config = Mock(spec=Config)
        
        chunker = DocumentChunkerFactory.create_chunker("structural", mock_config)
        
        assert isinstance(chunker, StructuralChunker)
    
    def test_create_semantic_chunker(self):
        """Test creating semantic chunker."""
        mock_config = Mock(spec=Config)
        
        chunker = DocumentChunkerFactory.create_chunker("semantic", mock_config)
        
        assert isinstance(chunker, SemanticChunker)
    
    def test_create_hybrid_chunker(self):
        """Test creating hybrid chunker."""
        mock_config = Mock(spec=Config)
        
        chunker = DocumentChunkerFactory.create_chunker("hybrid", mock_config)
        
        assert isinstance(chunker, HybridChunker)
    
    def test_create_unknown_strategy(self):
        """Test creating chunker with unknown strategy."""
        mock_config = Mock(spec=Config)
        
        chunker = DocumentChunkerFactory.create_chunker("unknown", mock_config)
        
        # Should fall back to hybrid
        assert isinstance(chunker, HybridChunker)


# Test convenience function
def test_get_document_chunker():
    """Test getting document chunker instance."""
    chunker = get_document_chunker("fixed")
    assert isinstance(chunker, FixedSizeChunker)
    assert chunker.config is not None


if __name__ == "__main__":
    pytest.main([__file__])
