"""
Main Document Processor

This module contains the core DocumentProcessor class that orchestrates the entire
document processing pipeline from parsing to vector storage.
"""

import logging
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .parsers import DocumentParser
from .chunkers import DocumentChunker
from .metadata_extractors import LLMMetadataExtractor
from .quality_assessor import QualityAssessor
from .vector_store import VectorStore
from .config import Config


@dataclass
class ProcessingResult:
    """Result of document processing operation."""
    document_id: str
    chunks: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    quality_metrics: Dict[str, float]
    processing_time: float
    memory_usage: float
    errors: List[str]
    warnings: List[str]


class DocumentProcessor:
    """
    Main document processor that orchestrates the entire processing pipeline.
    
    This class coordinates document parsing, chunking, metadata extraction,
    quality assessment, and vector storage in a multi-stage pipeline.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the document processor.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        self.config = Config(config_path)
        self.logger = logging.getLogger(__name__)
        
        # Initialize pipeline components
        self.parser = DocumentParser(self.config)
        self.chunker = DocumentChunker(self.config)
        self.metadata_extractor = LLMMetadataExtractor(self.config)
        self.quality_assessor = QualityAssessor(self.config)
        self.vector_store = VectorStore(self.config)
        
        self.logger.info("Document processor initialized successfully")
    
    def process_document(self, document_path: str) -> ProcessingResult:
        """
        Process a single document through the complete pipeline.
        
        Args:
            document_path: Path to the document to process
            
        Returns:
            ProcessingResult containing all processing information
        """
        start_time = time.time()
        document_id = str(uuid.uuid4())
        
        try:
            self.logger.info(f"Starting processing of document: {document_path}")
            
            # Stage 1: Document parsing
            parsed_content = self._parse_document(document_path)
            
            # Stage 2: Content chunking
            chunks = self._chunk_content(parsed_content)
            
            # Stage 3: Metadata extraction
            enhanced_chunks = self._extract_metadata(chunks, document_id)
            
            # Stage 4: Quality assessment
            quality_metrics = self._assess_quality(parsed_content, enhanced_chunks)
            
            # Stage 5: Vector storage
            stored_chunks = self._store_chunks(enhanced_chunks)
            
            # Stage 6: Generate processing summary
            processing_time = time.time() - start_time
            result = self._create_result(
                document_id, enhanced_chunks, quality_metrics, 
                processing_time, document_path
            )
            
            self.logger.info(f"Document processing completed successfully in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing document {document_path}: {str(e)}")
            raise
    
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
            try:
                self.logger.info(f"Processing document {i+1}/{len(document_paths)}: {doc_path}")
                result = self.process_document(doc_path)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to process {doc_path}: {str(e)}")
                # Continue with next document
                continue
        
        return results
    
    def _parse_document(self, document_path: str) -> Dict[str, Any]:
        """Parse the document using appropriate parser."""
        return self.parser.parse(document_path)
    
    def _chunk_content(self, parsed_content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk the parsed content using intelligent chunking strategies."""
        return self.chunker.chunk(parsed_content)
    
    def _extract_metadata(self, chunks: List[Dict[str, Any]], document_id: str) -> List[Dict[str, Any]]:
        """Extract metadata for each chunk using LLM enhancement."""
        enhanced_chunks = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{document_id}_chunk_{i:04d}"
            enhanced_chunk = self.metadata_extractor.extract_metadata(
                chunk, chunk_id, document_id
            )
            enhanced_chunks.append(enhanced_chunk)
        
        return enhanced_chunks
    
    def _assess_quality(self, original_content: Dict[str, Any], 
                       processed_chunks: List[Dict[str, Any]]) -> Dict[str, float]:
        """Assess the quality of processing results."""
        return self.quality_assessor.assess(original_content, processed_chunks)
    
    def _store_chunks(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Store chunks in vector database."""
        return self.vector_store.store_chunks(chunks)
    
    def _create_result(self, document_id: str, chunks: List[Dict[str, Any]],
                      quality_metrics: Dict[str, float], processing_time: float,
                      document_path: str) -> ProcessingResult:
        """Create the final processing result."""
        return ProcessingResult(
            document_id=document_id,
            chunks=chunks,
            metadata={
                "source_path": document_path,
                "total_chunks": len(chunks),
                "chunk_types": self._get_chunk_type_distribution(chunks)
            },
            quality_metrics=quality_metrics,
            processing_time=processing_time,
            memory_usage=self._get_memory_usage(),
            errors=[],
            warnings=[]
        )
    
    def _get_chunk_type_distribution(self, chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get distribution of chunk types."""
        distribution = {}
        for chunk in chunks:
            chunk_type = chunk.get("chunk_type", "unknown")
            distribution[chunk_type] = distribution.get(chunk_type, 0) + 1
        return distribution
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0.0
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics and performance metrics."""
        return {
            "total_documents_processed": len(self.vector_store.get_document_ids()),
            "total_chunks_stored": self.vector_store.get_total_chunks(),
            "average_processing_time": self._get_average_processing_time(),
            "memory_usage": self._get_memory_usage(),
            "quality_metrics": self.quality_assessor.get_overall_metrics()
        }
    
    def _get_average_processing_time(self) -> float:
        """Get average processing time per document."""
        # This would typically be stored and retrieved from a database
        # For now, return a placeholder
        return 0.0
    
    def cleanup(self):
        """Clean up resources and close connections."""
        self.vector_store.close()
        self.logger.info("Document processor cleanup completed")
