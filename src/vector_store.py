"""
Vector store implementations for storing and retrieving document chunks.

This module provides concrete implementations of the VectorStore abstract base class
for different vector database backends.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import json
import logging
import pickle
from pathlib import Path

from config import Config


class VectorStore(ABC):
    """Abstract base class for vector storage implementations."""

    def __init__(self, config: Config):
        """Initialize the vector store with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def store_chunks(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Store document chunks and return their IDs."""
        pass

    @abstractmethod
    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a chunk by ID."""
        pass

    @abstractmethod
    def search_chunks(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for chunks similar to the query."""
        pass

    @abstractmethod
    def get_document_ids(self) -> List[str]:
        """Get all document IDs in the store."""
        pass

    @abstractmethod
    def get_total_chunks(self) -> int:
        """Get total number of chunks in the store."""
        pass

    @abstractmethod
    def close(self):
        """Close the vector store connection."""
        pass


class FileBasedVectorStore(VectorStore):
    """Simple file-based vector store for development and testing."""

    def __init__(self, config: Config):
        """Initialize the file-based vector store."""
        super().__init__(config)
        self.store_path = Path(config.output.vector_store_path or "vector_db")
        self.store_path.mkdir(exist_ok=True)

        self.chunks_file = self.store_path / "chunks.json"
        self.metadata_file = self.store_path / "metadata.json"

        # Load existing data
        self.chunks = self._load_chunks()
        self.metadata = self._load_metadata()

        self.logger.info(f"File-based vector store initialized at {self.store_path}")

    def store_chunks(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Store chunks in the file-based store."""
        chunk_ids = []

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id", f"chunk_{len(self.chunks)}")
            self.chunks[chunk_id] = chunk
            chunk_ids.append(chunk_id)

            # Update metadata
            doc_id = chunk.get("document_id", "unknown")
            if doc_id not in self.metadata:
                self.metadata[doc_id] = {
                    "chunk_ids": [],
                    "total_chunks": 0,
                    "created_at": chunk.get("created_at", "unknown"),
                }

            if chunk_id not in self.metadata[doc_id]["chunk_ids"]:
                self.metadata[doc_id]["chunk_ids"].append(chunk_id)
                self.metadata[doc_id]["total_chunks"] += 1

        # Save to disk
        self._save_chunks()
        self._save_metadata()

        self.logger.info(f"Stored {len(chunk_ids)} chunks")
        return chunk_ids

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a chunk by ID."""
        return self.chunks.get(chunk_id)

    def search_chunks(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Simple text-based search (not semantic)."""
        # This is a basic implementation - in production, you'd use embeddings
        results = []
        query_lower = query.lower()

        for chunk_id, chunk in self.chunks.items():
            content = chunk.get("content", "").lower()
            if query_lower in content:
                results.append(chunk)
                if len(results) >= limit:
                    break

        return results

    def get_document_ids(self) -> List[str]:
        """Get all document IDs in the store."""
        return list(self.metadata.keys())

    def get_total_chunks(self) -> int:
        """Get total number of chunks in the store."""
        return len(self.chunks)

    def get_total_documents(self) -> int:
        """Get total number of documents in the store."""
        return len(self.metadata)

    def close(self):
        """Close the vector store (save data to disk)."""
        self._save_chunks()
        self._save_metadata()
        self.logger.info("File-based vector store closed")

    def _load_chunks(self) -> Dict[str, Any]:
        """Load chunks from disk."""
        if self.chunks_file.exists():
            try:
                with open(self.chunks_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load chunks: {e}")
        return {}

    def _save_chunks(self):
        """Save chunks to disk."""
        try:
            with open(self.chunks_file, "w", encoding="utf-8") as f:
                json.dump(self.chunks, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Could not save chunks: {e}")

    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load metadata: {e}")
        return {}

    def _save_metadata(self):
        """Save metadata to disk."""
        try:
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Could not save metadata: {e}")


class ChromaDBVectorStore(VectorStore):
    """ChromaDB vector store implementation."""

    def __init__(self, config: Config):
        """Initialize the ChromaDB vector store."""
        super().__init__(config)
        try:
            import chromadb

            self.client = chromadb.PersistentClient(
                path=str(self.config.output.vector_store_path or "chroma_db")
            )
            self.collection = self.client.get_or_create_collection("document_chunks")
            self.logger.info("ChromaDB vector store initialized")
        except ImportError:
            self.logger.error(
                "ChromaDB not available. Install with: pip install chromadb"
            )
            raise

    def store_chunks(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Store chunks in ChromaDB."""
        # Implementation would go here
        # For now, return placeholder IDs
        return [f"chroma_chunk_{i}" for i in range(len(chunks))]

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a chunk by ID from ChromaDB."""
        # Implementation would go here
        return None

    def search_chunks(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for chunks in ChromaDB."""
        # Implementation would go here
        return []

    def get_document_ids(self) -> List[str]:
        """Get all document IDs from ChromaDB."""
        # Implementation would go here
        return []

    def get_total_chunks(self) -> int:
        """Get total number of chunks in ChromaDB."""
        # Implementation would go here
        return 0

    def close(self):
        """Close the ChromaDB connection."""
        # ChromaDB handles its own connections
        pass


# Factory function for creating vector stores
def get_vector_store(store_type: str, config: Config) -> VectorStore:
    """Factory function to create vector store instances."""
    if store_type == "file":
        return FileBasedVectorStore(config)
    elif store_type == "chromadb":
        return ChromaDBVectorStore(config)
    else:
        raise ValueError(f"Unknown vector store type: {store_type}")
