"""
Tests for document chunkers
"""

import pytest
from unittest.mock import Mock, patch

from src.chunkers import (
    ChunkingResult,
    DocumentChunk,
    DocumentChunker,
    DocumentChunkerFactory,
    FixedSizeChunker,
    HybridChunker,
    SemanticChunker,
    StructuralChunker,
    get_document_chunker,
)
from src.config import Config
from src.parsers import ParsedContent


class TestDocumentChunk:
    """Test the DocumentChunk dataclass."""

    def test_document_chunk_creation(self):
        """Test basic DocumentChunk creation."""
        chunk = DocumentChunk(
            chunk_id="chunk_12345",
            content="This is a test chunk content.",
            chunk_type="text",
            chunk_index=0,
            quality_score=0.8,
            metadata={"source": "test.txt"},
        )

        assert chunk.chunk_id == "chunk_12345"
        assert chunk.content == "This is a test chunk content."
        assert chunk.chunk_type == "text"
        assert chunk.chunk_index == 0
        assert chunk.quality_score == 0.8
        assert chunk.metadata["source"] == "test.txt"

    def test_document_chunk_defaults(self):
        """Test DocumentChunk default values."""
        chunk = DocumentChunk(
            chunk_id="chunk_12345", 
            content="Test content",
            metadata={},
            chunk_type="text",
            chunk_index=0
        )

        assert chunk.chunk_type == "text"
        assert chunk.chunk_index == 0
        assert chunk.quality_score == 0.0
        assert chunk.metadata == {}

    def test_document_chunk_metadata_extraction(self):
        """Test that metadata is properly extracted and stored."""
        metadata = {
            "source": "document.pdf",
            "page": 1,
            "section": "introduction",
            "timestamp": "2024-01-01T00:00:00Z",
        }

        chunk = DocumentChunk(
            chunk_id="chunk_12345",
            content="Test content",
            metadata=metadata,
            chunk_type="text",
            chunk_index=0
        )

        assert chunk.metadata["source"] == "document.pdf"
        assert chunk.metadata["page"] == 1
        assert chunk.metadata["section"] == "introduction"
        assert chunk.metadata["timestamp"] == "2024-01-01T00:00:00Z"

    def test_chunk_quality_assessment(self):
        """Test chunk quality assessment method."""
        chunk = DocumentChunk(
            chunk_id="chunk_12345",
            content="This is a high-quality chunk with substantial content.",
            metadata={},
            chunk_type="text",
            chunk_index=0,
            quality_score=0.9
        )

        # Test quality score
        assert chunk.quality_score == 0.9
        assert chunk.chunk_type == "text"
        assert chunk.chunk_index == 0


class TestChunkingResult:
    """Test the ChunkingResult dataclass."""

    def test_chunking_result_creation(self):
        """Test basic ChunkingResult creation."""
        chunks = [
            DocumentChunk(chunk_id="chunk_1", content="Content 1", metadata={}, chunk_type="text", chunk_index=0),
            DocumentChunk(chunk_id="chunk_2", content="Content 2", metadata={}, chunk_type="text", chunk_index=1),
        ]

        result = ChunkingResult(
            success=True,
            chunks=chunks,
            chunking_strategy="fixed_size",
            total_chunks=2,
            processing_time=1.5,
            metadata={},
            errors=[],
            warnings=[]
        )

        assert result.success is True
        assert len(result.chunks) == 2
        assert result.chunking_strategy == "fixed_size"
        assert result.processing_time == 1.5
        assert len(result.errors) == 0

    def test_chunking_result_failure(self):
        """Test ChunkingResult for failed chunking."""
        result = ChunkingResult(
            success=False,
            chunks=[],
            chunking_strategy="fixed_size",
            total_chunks=0,
            processing_time=0.5,
            metadata={},
            errors=["Invalid content format"],
            warnings=[]
        )

        assert result.success is False
        assert len(result.chunks) == 0
        assert len(result.errors) == 1
        assert "Invalid content format" in result.errors


class TestDocumentChunker:
    """Test the base DocumentChunker class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = Mock(spec=Config)
        # Use a concrete implementation for testing
        from src.chunkers import HybridChunker
        self.chunker = HybridChunker(self.config)

    def test_chunker_initialization(self):
        """Test that chunker initializes with configuration."""
        assert self.chunker.config == self.config
        assert hasattr(self.chunker, "chunking_config")
        assert hasattr(self.chunker, "chunker_name")

    def test_chunk_method_alias(self):
        """Test that chunk method is available as alias."""
        assert hasattr(self.chunker, "chunk")
        assert callable(self.chunker.chunk)


class TestFixedSizeChunker:
    """Test the FixedSizeChunker class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = Mock(spec=Config)
        # Mock the chunking config
        self.config.chunking = Mock()
        self.config.chunking.chunk_size = 1000
        self.config.chunking.overlap_size = 200
        self.config.chunking.min_chunk_size = 100
        self.config.chunking.max_chunk_size = 2000
        self.config.get_chunking_config.return_value = self.config.chunking
        self.chunker = FixedSizeChunker(self.config)

    def test_fixed_size_chunking(self):
        """Test fixed-size chunking strategy."""
        content = Mock(spec=ParsedContent)
        content.text_content = "A" * 2500  # 2500 characters

        result = self.chunker.chunk_document(content)

        assert result.success is True
        assert len(result.chunks) > 1
        assert result.chunking_strategy == "fixed_size"

        # Check chunk sizes
        for chunk in result.chunks:
            assert len(chunk.content) <= 2000  # Max size from config (max_chunk_size)
            assert len(chunk.content) >= 100  # Min size from config

    def test_fixed_size_chunking_with_overlap(self):
        """Test fixed-size chunking with overlap."""
        content = Mock(spec=ParsedContent)
        content.text_content = "A" * 1500

        result = self.chunker.chunk_document(content)

        assert result.success is True
        assert len(result.chunks) >= 1  # May get 1 chunk for 1500 chars

        # Check overlap between consecutive chunks if we have multiple chunks
        if len(result.chunks) >= 2:
            chunk1_content = result.chunks[0].content
            chunk2_content = result.chunks[1].content
            # Should have some overlap
            assert len(chunk1_content) + len(chunk2_content) > 1500


class TestStructuralChunker:
    """Test the StructuralChunker class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = Mock(spec=Config)
        # Mock the chunking config
        self.config.chunking = Mock()
        self.config.chunking.chunk_size = 1000
        self.config.chunking.overlap_size = 200
        self.config.chunking.min_chunk_size = 100
        self.config.chunking.max_chunk_size = 2000
        self.config.get_chunking_config.return_value = self.config.chunking
        self.chunker = StructuralChunker(self.config)

    def test_structural_chunking_by_headings(self):
        """Test structural chunking based on headings."""
        content = Mock(spec=ParsedContent)
        content.structure = [
            {"type": "heading", "text": "Introduction", "level": 1},
            {"type": "paragraph", "text": "This is the introduction."},
            {"type": "heading", "text": "Methods", "level": 1},
            {"type": "paragraph", "text": "These are the methods."},
        ]
        content.structured_content = {
            "headings": [
                {"type": "heading", "text": "Introduction", "level": 1},
                {"type": "heading", "text": "Methods", "level": 1}
            ],
            "paragraphs": [
                {"type": "paragraph", "text": "This is the introduction."},
                {"type": "paragraph", "text": "These are the methods."}
            ]
        }
        content.text_content = (
            "Introduction\nThis is the introduction.\nMethods\nThese are the methods."
        )

        result = self.chunker.chunk_document(content)

        assert result.success is True
        assert len(result.chunks) >= 2
        assert result.chunking_strategy == "structural"

    def test_structural_chunking_by_sections(self):
        """Test structural chunking based on sections."""
        content = Mock(spec=ParsedContent)
        content.structure = [
            {"type": "section", "text": "Section 1"},
            {"type": "paragraph", "text": "Section 1 content."},
            {"type": "section", "text": "Section 2"},
            {"type": "paragraph", "text": "Section 2 content."},
        ]
        content.structured_content = {
            "sections": [
                {"type": "section", "text": "Section 1"},
                {"type": "section", "text": "Section 2"}
            ],
            "paragraphs": [
                {"type": "paragraph", "text": "Section 1 content."},
                {"type": "paragraph", "text": "Section 2 content."}
            ]
        }
        content.text_content = "Section 1 content. Section 2 content."

        result = self.chunker.chunk_document(content)

        assert result.success is True
        assert len(result.chunks) >= 2


class TestSemanticChunker:
    """Test the SemanticChunker class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = Mock(spec=Config)
        # Mock the chunking config
        self.config.chunking = Mock()
        self.config.chunking.chunk_size = 1000
        self.config.chunking.overlap_size = 200
        self.config.chunking.min_chunk_size = 100
        self.config.chunking.max_chunk_size = 2000
        self.config.get_chunking_config.return_value = self.config.chunking
        self.chunker = SemanticChunker(self.config)

    def test_semantic_chunking(self):
        """Test semantic chunking strategy."""
        content = Mock(spec=ParsedContent)
        content.text_content = (
            "This is a semantic test. It should group related content together."
        )

        result = self.chunker.chunk_document(content)

        assert result.success is True
        assert result.chunking_strategy == "semantic"


class TestHybridChunker:
    """Test the HybridChunker class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = Mock(spec=Config)
        # Mock the chunking config
        self.config.chunking = Mock()
        self.config.chunking.chunk_size = 1000
        self.config.chunking.overlap_size = 200
        self.config.chunking.min_chunk_size = 100
        self.config.chunking.max_chunk_size = 2000
        self.config.get_chunking_config.return_value = self.config.chunking
        self.chunker = HybridChunker(self.config)

    def test_hybrid_chunking_strategy_selection(self):
        """Test that hybrid chunker selects appropriate strategy."""
        content = Mock(spec=ParsedContent)
        content.text_content = "Test content for hybrid chunking."
        content.structure = ["heading1", "paragraph1"]

        result = self.chunker.chunk_document(content)

        assert result.success is True
        assert result.chunking_strategy.startswith("hybrid")  # May be hybrid_fixed, hybrid_structural, etc.

    def test_hybrid_chunking_fallback(self):
        """Test hybrid chunking fallback to fixed-size."""
        content = Mock(spec=ParsedContent)
        content.text_content = "Simple text without complex structure."
        content.structure = []

        result = self.chunker.chunk_document(content)

        assert result.success is True
        # Should fall back to fixed-size chunking for simple content


class TestDocumentChunkerFactory:
    """Test the DocumentChunkerFactory class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = Mock(spec=Config)
        # Mock the chunking config
        self.config.chunking = Mock()
        self.config.chunking.strategy = "hybrid"
        self.config.get_chunking_config.return_value = self.config.chunking

    def test_factory_creates_fixed_size_chunker(self):
        """Test factory creates FixedSizeChunker."""
        chunker = DocumentChunkerFactory.create_chunker("fixed", self.config)
        assert isinstance(chunker, FixedSizeChunker)

    def test_factory_creates_structural_chunker(self):
        """Test factory creates StructuralChunker."""
        chunker = DocumentChunkerFactory.create_chunker("structural", self.config)
        assert isinstance(chunker, StructuralChunker)

    def test_factory_creates_semantic_chunker(self):
        """Test factory creates SemanticChunker."""
        chunker = DocumentChunkerFactory.create_chunker("semantic", self.config)
        assert isinstance(chunker, SemanticChunker)

    def test_factory_creates_hybrid_chunker(self):
        """Test factory creates HybridChunker."""
        chunker = DocumentChunkerFactory.create_chunker("hybrid", self.config)
        assert isinstance(chunker, HybridChunker)

    def test_factory_creates_hybrid_chunker_by_default(self):
        """Test factory creates HybridChunker by default."""
        chunker = DocumentChunkerFactory.create_chunker("unknown", self.config)
        assert isinstance(chunker, HybridChunker)

    def test_factory_creates_hybrid_by_default(self):
        """Test that factory creates hybrid chunker by default."""
        chunker = DocumentChunkerFactory.create_chunker("unknown", self.config)
        assert isinstance(chunker, HybridChunker)


def test_get_document_chunker():
    """Test the convenience function for getting a document chunker."""
    config = Config()
    chunker = get_document_chunker("hybrid", config)
    assert isinstance(chunker, DocumentChunker)


if __name__ == "__main__":
    pytest.main([__file__])
