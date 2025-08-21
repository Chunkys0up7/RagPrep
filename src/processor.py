"""
Main document processor that orchestrates the entire document processing pipeline.

This module provides the central DocumentProcessor class that coordinates parsing,
chunking, metadata extraction, quality assessment, and vector storage.
"""

import logging
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from config import Config
from parsers import get_document_parser, ParsedContent
from chunkers import get_document_chunker, DocumentChunk
from metadata_extractors import get_metadata_extractor
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


class DocumentProcessor:
    """Main orchestrator for document processing pipeline."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the document processor with configuration."""
        self.config = Config(config_path)
        self.logger = logging.getLogger(__name__)
        
        # Initialize pipeline components using factories/concrete classes
        self.parser = get_document_parser(config_path)
        self.chunker = get_document_chunker(
            self.config.chunking.strategy, 
            config_path
        )
        self.metadata_extractor = get_metadata_extractor(
            self.config.metadata.extraction_level, 
            config_path
        )
        self.quality_system = get_quality_assessment_system(config_path)
        
        # Initialize security manager
        self.security_manager = SecurityManager(self.config)
        
        # Initialize vector store
        try:
            from vector_store import get_vector_store
            self.vector_store = get_vector_store("file", self.config)  # Default to file-based store
            self.logger.info("Vector store initialized successfully")
        except Exception as e:
            self.logger.warning(f"Vector store initialization failed: {e}")
            self.vector_store = None
        
        self.logger.info("Document processor initialized successfully")
    
    def process_document(self, document_path: str) -> ProcessingResult:
        """
        Process a single document through the complete pipeline.
        
        Args:
            document_path: Path to the document to process
            
        Returns:
            ProcessingResult with processing outcomes
        """
        start_time = time.time()
        document_id = self._generate_document_id(document_path)
        
        try:
            # Security validation
            if not self.security_manager.is_file_safe_for_processing(Path(document_path)):
                raise SecurityException("Document failed security checks")
            
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
                    error_message=f"Parsing failed: {parse_result.error_message}"
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
                    error_message=f"Chunking failed: {chunk_result.error_message}"
                )
            
            # 3. Extract metadata
            self.logger.info("Extracting metadata")
            metadata_result = self.metadata_extractor.extract(
                parse_result.content,
                chunk_result.chunks
            )
            
            if not metadata_result.success:
                self.logger.warning(f"Metadata extraction failed: {metadata_result.error_message}")
                # Continue with basic metadata if extraction fails
                metadata = parse_result.content.metadata
            else:
                metadata = metadata_result.metadata
            
            # 4. Assess quality
            self.logger.info("Assessing document quality")
            quality_result = self.quality_system.assess_quality(
                parse_result.content,
                chunk_result.chunks,
                metadata_result
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
                            'chunk_id': chunk.chunk_id,
                            'document_id': document_id,
                            'content': chunk.content,
                            'chunk_type': chunk.chunk_type,
                            'quality_score': chunk.quality_score,
                            'metadata': chunk.metadata,
                            'created_at': str(chunk.metadata.get('timestamp', 'unknown'))
                        }
                        chunk_dicts.append(chunk_dict)
                    
                    chunk_ids = self.vector_store.store_chunks(chunk_dicts)
                except NotImplementedError:
                    self.logger.warning("Vector storage not yet implemented")
                except Exception as e:
                    self.logger.error(f"Vector storage failed: {e}")
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                success=True,
                document_id=document_id,
                chunks=chunk_result.chunks,
                metadata=metadata,
                quality_score=quality_result.overall_score,
                processing_time=processing_time,
                warnings=quality_result.recommendations
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Document processing failed: {e}")
            
            return ProcessingResult(
                success=False,
                document_id=document_id,
                chunks=[],
                metadata={},
                quality_score=0.0,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    def process_batch(self, document_paths: List[str]) -> List[ProcessingResult]:
        """
        Process multiple documents in batch.
        
        Args:
            document_paths: List of document paths to process
            
        Returns:
            List of ProcessingResult objects
        """
        results = []
        
        for i, doc_path in enumerate(document_paths):
            self.logger.info(f"Processing document {i+1}/{len(document_paths)}: {doc_path}")
            result = self.process_document(doc_path)
            results.append(result)
            
            if not result.success:
                self.logger.warning(f"Document {doc_path} failed processing")
        
        return results
    
    def _generate_document_id(self, document_path: str) -> str:
        """Generate a unique document ID."""
        import hashlib
        import time
        
        # Combine path and timestamp for uniqueness
        unique_string = f"{document_path}_{time.time()}"
        return hashlib.md5(unique_string.encode()).hexdigest()[:16]
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics and metrics."""
        if self.vector_store:
            try:
                return {
                    "total_documents": self.vector_store.get_total_documents(),
                    "total_chunks": self.vector_store.get_total_chunks(),
                    "document_ids": self.vector_store.get_document_ids()
                }
            except NotImplementedError:
                return {"vector_store": "Not implemented"}
            except Exception as e:
                return {"vector_store_error": str(e)}
        else:
            return {"vector_store": "Not available"}
    
    def close(self):
        """Clean up resources."""
        if self.vector_store:
            try:
                self.vector_store.close()
            except NotImplementedError:
                pass
            except Exception as e:
                self.logger.error(f"Error closing vector store: {e}")


class SecurityException(Exception):
    """Exception raised when security checks fail."""
    pass
