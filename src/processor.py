"""
Document processing pipeline for RAG applications.

This module provides the main DocumentProcessor class that orchestrates
the complete document processing workflow.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from chunkers import DocumentChunk, get_document_chunker
from config import Config
from metadata_extractors import get_metadata_extractor
from parsers import ParsedContent, get_document_parser
from quality_assessment import get_quality_assessment_system
from security import SecurityManager


@dataclass
class ProcessingResult:
    """Result of document processing operation."""

    success: bool
    document_id: str
    chunks: List[DocumentChunk]
    metadata: Dict[str, Any]
    quality_score: float
    processing_time: float
    error_message: Optional[str] = None
    warnings: List[str] = None

    def __post_init__(self):
        """Initialize warnings list if not provided."""
        if self.warnings is None:
            self.warnings = []


class DocumentProcessor:
    """Main document processing orchestrator."""

    def __init__(self, config: Optional[Config] = None):
        """Initialize the document processor."""
        self.config = config or Config()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize components using factory functions
        self.parser = get_document_parser(self.config)
        self.chunker = get_document_chunker(self.config)
        self.metadata_extractor = get_metadata_extractor(self.config)
        self.quality_system = get_quality_assessment_system(self.config)
        self.security_manager = SecurityManager(self.config)

        # Initialize vector store
        try:
            from vector_store import get_vector_store

            self.vector_store = get_vector_store("file", self.config)
            self.logger.info("Vector store initialized successfully")
        except Exception as e:
            self.logger.warning(f"Vector store initialization failed: {e}")
            self.vector_store = None

        self.logger.info("Document processor initialized successfully")

    def process_document(self, document_path: str) -> ProcessingResult:
        """Process a single document through the complete pipeline."""
        start_time = time.time()
        document_id = self._generate_document_id(document_path)

        try:
            # Security check
            if not self.security_manager.is_file_safe_for_processing(
                Path(document_path)
            ):
                return ProcessingResult(
                    success=False,
                    document_id=document_id,
                    chunks=[],
                    metadata={},
                    quality_score=0.0,
                    processing_time=time.time() - start_time,
                    error_message="File failed security validation",
                )

            # 1. Parse document
            self.logger.info(f"Parsing document: {document_path}")
            parse_result = self.parser.parse(document_path)

            if not parse_result.success:
                return ProcessingResult(
                    success=False,
                    document_id=document_id,
                    chunks=[],
                    metadata={},
                    quality_score=0.0,
                    processing_time=time.time() - start_time,
                    error_message=f"Parsing failed: {parse_result.error_message}",
                )

            # 2. Chunk document
            self.logger.info("Chunking document content")
            chunk_result = self.chunker.chunk(parse_result.content)

            if not chunk_result.success:
                return ProcessingResult(
                    success=False,
                    document_id=document_id,
                    chunks=[],
                    metadata=parse_result.content.metadata,
                    quality_score=0.0,
                    processing_time=time.time() - start_time,
                    error_message=f"Chunking failed: {chunk_result.error_message}",
                )

            # 3. Extract metadata
            self.logger.info("Extracting metadata")
            metadata_result = self.metadata_extractor.extract(
                parse_result.content, chunk_result.chunks
            )

            # Use extracted metadata or fall back to basic metadata
            if not metadata_result.success:
                self.logger.warning(
                    f"Metadata extraction failed: {metadata_result.error_message}"
                )
                # Continue with basic metadata if extraction fails
                metadata = parse_result.content.metadata
            else:
                metadata = metadata_result.metadata

            # 4. Assess quality
            self.logger.info("Assessing document quality")
            quality_result = self.quality_system.assess_quality(
                parse_result.content, chunk_result.chunks, metadata_result
            )

            # 5. Store in vector database (if available)
            chunk_ids = []
            if self.vector_store:
                try:
                    self.logger.info("Storing chunks in vector database")
                    # Convert DocumentChunk objects to dictionaries
                    chunk_dicts = []
                    for chunk in chunk_result.chunks:
                        chunk_dict = {
                            "chunk_id": chunk.chunk_id,
                            "document_id": document_id,
                            "content": chunk.content,
                            "chunk_type": chunk.chunk_type,
                            "quality_score": chunk.quality_score,
                            "metadata": chunk.metadata,
                            "created_at": str(
                                chunk.metadata.get("timestamp", "unknown")
                            ),
                        }
                        chunk_dicts.append(chunk_dict)

                    chunk_ids = self.vector_store.store_chunks(chunk_dicts)
                    self.logger.info(
                        f"Stored {len(chunk_ids)} chunks in vector database"
                    )
                except NotImplementedError:
                    self.logger.warning("Vector storage not yet implemented")
                except Exception as e:
                    self.logger.error(f"Vector storage failed: {e}")
                    # Continue processing even if vector storage fails

            # Calculate overall quality score
            overall_quality = quality_result.overall_score

            return ProcessingResult(
                success=True,
                document_id=document_id,
                chunks=chunk_result.chunks,
                metadata=metadata,
                quality_score=overall_quality,
                processing_time=time.time() - start_time,
                warnings=chunk_result.errors,
            )

        except Exception as e:
            self.logger.error(f"Document processing failed: {e}")
            return ProcessingResult(
                success=False,
                document_id=document_id,
                chunks=[],
                metadata={},
                quality_score=0.0,
                processing_time=time.time() - start_time,
                error_message=str(e),
            )

    def process_batch(self, document_paths: List[str]) -> List[ProcessingResult]:
        """Process multiple documents in batch."""
        results = []
        total_docs = len(document_paths)

        for i, doc_path in enumerate(document_paths, 1):
            self.logger.info(f"Processing document {i}/{total_docs}: {doc_path}")
            result = self.process_document(doc_path)
            results.append(result)

            if not result.success:
                self.logger.warning(f"Document {doc_path} failed processing")

        return results

    def _generate_document_id(self, document_path: str) -> str:
        """Generate a unique document ID."""
        import hashlib

        # Combine path and timestamp for uniqueness
        unique_string = f"{document_path}_{time.time()}"
        return hashlib.md5(unique_string.encode()).hexdigest()[:16]

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        # This would collect statistics from all components
        return {
            "total_documents_processed": 0,  # Would be tracked
            "success_rate": 0.0,  # Would be calculated
            "average_processing_time": 0.0,  # Would be calculated
            "vector_store_status": "available" if self.vector_store else "unavailable",
        }
