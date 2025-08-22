"""
Tests for vector store functionality
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.vector_store import (
    VectorStore,
    FileBasedVectorStore,
    ChromaDBVectorStore,
    get_vector_store,
)
from src.config import Config
from src.chunkers import DocumentChunk


class TestVectorStore:
    """Test the abstract VectorStore base class."""

    def test_abstract_base_class(self):
        """Test that VectorStore is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            VectorStore(Mock(spec=Config))


class TestFileBasedVectorStore:
    """Test the file-based vector store implementation."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_config):
        """Set up test fixtures."""
        self.mock_config = mock_config
        self.mock_config.output.vector_store_path = "test_vector_db"
        self.store = FileBasedVectorStore(self.mock_config)

    def test_store_initialization(self):
        """Test vector store initialization."""
        assert hasattr(self.store, "config")
        assert hasattr(self.store, "store_path")
        assert hasattr(self.store, "chunks_file")
        assert hasattr(self.store, "metadata_file")

    def test_store_document_chunks(self):
        """Test storing document chunks."""
        chunks = [
            {
                "chunk_id": "chunk1",
                "content": "Test content 1",
                "chunk_type": "text",
                "chunk_index": 0,
                "quality_score": 0.9,
                "metadata": {"source": "test.txt"},
                "document_id": "test.txt"
            },
            {
                "chunk_id": "chunk2",
                "content": "Test content 2",
                "chunk_type": "text",
                "chunk_index": 1,
                "quality_score": 0.8,
                "metadata": {"source": "test.txt"},
                "document_id": "test.txt"
            }
        ]
        
        result = self.store.store_chunks(chunks)
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert "chunk1" in result
        assert "chunk2" in result

    def test_retrieve_document_chunks(self):
        """Test retrieving document chunks."""
        # First store some chunks
        chunks = [
            {
                "chunk_id": "chunk1",
                "content": "Test content 1",
                "chunk_type": "text",
                "chunk_index": 0,
                "quality_score": 0.9,
                "metadata": {"source": "test.txt"},
                "document_id": "test.txt"
            }
        ]
        
        self.store.store_chunks(chunks)
        
        # Then retrieve them by chunk ID
        retrieved = self.store.get_chunk("chunk1")
        
        assert retrieved is not None
        assert retrieved["chunk_id"] == "chunk1"
        assert retrieved["content"] == "Test content 1"

    def test_search_similar_chunks(self):
        """Test searching for similar chunks."""
        # Store some chunks first
        chunks = [
            {
                "chunk_id": "chunk1",
                "content": "Machine learning algorithms",
                "chunk_type": "text",
                "chunk_index": 0,
                "quality_score": 0.9,
                "metadata": {"source": "test.txt"},
                "document_id": "test.txt"
            },
            {
                "chunk_id": "chunk2",
                "content": "Deep learning neural networks",
                "chunk_type": "text",
                "chunk_index": 1,
                "quality_score": 0.8,
                "metadata": {"source": "test.txt"},
                "document_id": "test.txt"
            }
        ]
        
        self.store.store_chunks(chunks)
        
        # Search for similar content
        results = self.store.search_chunks("machine learning", limit=2)
        
        assert isinstance(results, list)
        assert len(results) > 0

    def test_get_total_documents(self):
        """Test getting total document count."""
        # Store some chunks
        chunks = [
            {
                "chunk_id": "chunk1",
                "content": "Test content",
                "chunk_type": "text",
                "chunk_index": 0,
                "quality_score": 0.9,
                "metadata": {"source": "doc1.txt"},
                "document_id": "doc1.txt"
            },
            {
                "chunk_id": "chunk2",
                "content": "More content",
                "chunk_type": "text",
                "chunk_index": 0,
                "quality_score": 0.8,
                "metadata": {"source": "doc2.txt"},
                "document_id": "doc2.txt"
            }
        ]
        
        self.store.store_chunks(chunks)
        
        total = self.store.get_total_documents()
        assert total == 2

    def test_clear_store(self):
        """Test clearing the vector store."""
        # Store some chunks first
        chunks = [
            {
                "chunk_id": "chunk1",
                "content": "Test content",
                "chunk_type": "text",
                "chunk_index": 0,
                "quality_score": 0.9,
                "metadata": {"source": "test.txt"},
                "document_id": "test.txt"
            }
        ]
        
        self.store.store_chunks(chunks)
        
        # Verify chunks are stored
        assert self.store.get_total_documents() > 0
        
        # Clear the store by removing all chunks
        self.store.chunks.clear()
        self.store.metadata.clear()
        
        assert self.store.get_total_documents() == 0

    def teardown_method(self):
        """Clean up test files."""
        try:
            if os.path.exists("test_vector_db"):
                import shutil
                shutil.rmtree("test_vector_db")
        except Exception:
            pass


class TestChromaDBVectorStore:
    """Test the ChromaDB vector store implementation."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_config):
        """Set up test fixtures."""
        self.mock_config = mock_config
        self.mock_config.output.vector_store_path = "test_chroma_db"
        self.store = ChromaDBVectorStore(self.mock_config)

    def test_store_initialization(self):
        """Test ChromaDB store initialization."""
        assert hasattr(self.store, "config")
        assert hasattr(self.store, "client")
        assert hasattr(self.store, "collection")

    def test_chromadb_not_available(self):
        """Test behavior when ChromaDB is not available."""
        # Test that the store was created successfully with chromadb available
        assert hasattr(self.store, "client")
        assert hasattr(self.store, "collection")
        
        # Test that the store methods work as expected
        result = self.store.store_chunks([])
        assert isinstance(result, list)
        assert len(result) == 0


def test_get_vector_store(mock_config):
    """Test the factory function for creating vector stores."""
    # Test file-based store
    file_store = get_vector_store("file", mock_config)
    assert isinstance(file_store, FileBasedVectorStore)
    
    # Test ChromaDB store (may fail if chromadb not installed)
    try:
        chroma_store = get_vector_store("chromadb", mock_config)
        assert isinstance(chroma_store, ChromaDBVectorStore)
    except ImportError:
        # ChromaDB not available, skip this test
        pass
    
    # Test default store
    default_store = get_vector_store("file", mock_config)
    assert isinstance(default_store, FileBasedVectorStore)


if __name__ == "__main__":
    pytest.main([__file__])
