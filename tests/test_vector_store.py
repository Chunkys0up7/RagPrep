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

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = Mock(spec=Config)
        self.mock_config.output.vector_store_path = "test_vector_db"
        self.store = FileBasedVectorStore(self.mock_config)

    def test_store_initialization(self):
        """Test vector store initialization."""
        assert hasattr(self.store, "config")
        assert hasattr(self.store, "store_path")
        assert hasattr(self.store, "chunks_file")
        assert hasattr(self.store, "embeddings_file")

    def test_store_document_chunks(self):
        """Test storing document chunks."""
        chunks = [
            DocumentChunk(
                chunk_id="chunk1",
                content="Test content 1",
                chunk_type="text",
                chunk_index=0,
                quality_score=0.9,
                metadata={"source": "test.txt"}
            ),
            DocumentChunk(
                chunk_id="chunk2",
                content="Test content 2",
                chunk_type="text",
                chunk_index=1,
                quality_score=0.8,
                metadata={"source": "test.txt"}
            )
        ]
        
        result = self.store.store_chunks(chunks)
        
        assert result["success"] is True
        assert result["stored_chunks"] == 2
        assert result["total_chunks"] == 2

    def test_retrieve_document_chunks(self):
        """Test retrieving document chunks."""
        # First store some chunks
        chunks = [
            DocumentChunk(
                chunk_id="chunk1",
                content="Test content 1",
                chunk_type="text",
                chunk_index=0,
                quality_score=0.9,
                metadata={"source": "test.txt"}
            )
        ]
        
        self.store.store_chunks(chunks)
        
        # Then retrieve them
        retrieved = self.store.retrieve_chunks("test.txt")
        
        assert retrieved["success"] is True
        assert len(retrieved["chunks"]) == 1
        assert retrieved["chunks"][0]["chunk_id"] == "chunk1"

    def test_search_similar_chunks(self):
        """Test searching for similar chunks."""
        # Store some chunks first
        chunks = [
            DocumentChunk(
                chunk_id="chunk1",
                content="Machine learning algorithms",
                chunk_type="text",
                chunk_index=0,
                quality_score=0.9,
                metadata={"source": "test.txt"}
            ),
            DocumentChunk(
                chunk_id="chunk2",
                content="Deep learning neural networks",
                chunk_type="text",
                chunk_index=1,
                quality_score=0.8,
                metadata={"source": "test.txt"}
            )
        ]
        
        self.store.store_chunks(chunks)
        
        # Search for similar content
        results = self.store.search_similar("machine learning", top_k=2)
        
        assert results["success"] is True
        assert len(results["results"]) > 0

    def test_get_total_documents(self):
        """Test getting total document count."""
        # Store some chunks
        chunks = [
            DocumentChunk(
                chunk_id="chunk1",
                content="Test content",
                chunk_type="text",
                chunk_index=0,
                quality_score=0.9,
                metadata={"source": "doc1.txt"}
            ),
            DocumentChunk(
                chunk_id="chunk2",
                content="More content",
                chunk_type="text",
                chunk_index=0,
                quality_score=0.8,
                metadata={"source": "doc2.txt"}
            )
        ]
        
        self.store.store_chunks(chunks)
        
        total = self.store.get_total_documents()
        assert total == 2

    def test_clear_store(self):
        """Test clearing the vector store."""
        # Store some chunks first
        chunks = [
            DocumentChunk(
                chunk_id="chunk1",
                content="Test content",
                chunk_type="text",
                chunk_index=0,
                quality_score=0.9,
                metadata={"source": "test.txt"}
            )
        ]
        
        self.store.store_chunks(chunks)
        
        # Verify chunks are stored
        assert self.store.get_total_documents() > 0
        
        # Clear the store
        result = self.store.clear_store()
        
        assert result["success"] is True
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

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = Mock(spec=Config)
        self.mock_config.vector_store.host = "localhost"
        self.mock_config.vector_store.port = 8000
        self.store = ChromaDBVectorStore(self.mock_config)

    def test_store_initialization(self):
        """Test ChromaDB store initialization."""
        assert hasattr(self.store, "config")
        assert hasattr(self.store, "host")
        assert hasattr(self.store, "port")

    def test_chromadb_not_available(self):
        """Test behavior when ChromaDB is not available."""
        with patch('src.vector_store.chromadb') as mock_chromadb:
            mock_chromadb.side_effect = ImportError("ChromaDB not available")
            
            store = ChromaDBVectorStore(self.mock_config)
            result = store.store_chunks([])
            
            assert result["success"] is False
            assert "ChromaDB not available" in result["error_message"]


def test_get_vector_store():
    """Test the factory function for creating vector stores."""
    # Test file-based store
    file_store = get_vector_store("file", Mock(spec=Config))
    assert isinstance(file_store, FileBasedVectorStore)
    
    # Test ChromaDB store
    chroma_store = get_vector_store("chromadb", Mock(spec=Config))
    assert isinstance(chroma_store, ChromaDBVectorStore)
    
    # Test default store
    default_store = get_vector_store("file", Mock(spec=Config))
    assert isinstance(default_store, FileBasedVectorStore)


if __name__ == "__main__":
    pytest.main([__file__])
