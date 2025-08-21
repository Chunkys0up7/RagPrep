"""
RAG Document Processing Utility

A comprehensive tool for processing documents for RAG (Retrieval-Augmented Generation) applications.
Provides intelligent parsing, chunking, metadata extraction, and quality assessment.
"""

from .config import Config, get_config, reload_config
from .parsers import (
    DocumentParser, PDFParser, DOCXParser, TextParser, HTMLParser,
    CascadingDocumentParser, ParsedContent, ParserResult, get_document_parser
)
from .chunkers import (
    DocumentChunker, FixedSizeChunker, StructuralChunker, SemanticChunker,
    HybridChunker, DocumentChunkerFactory, DocumentChunk, ChunkingResult,
    get_document_chunker
)
from .metadata_extractors import (
    MetadataExtractor, BasicMetadataExtractor, LLMMetadataExtractor,
    MetadataExtractorFactory, Entity, Topic, Relationship, Summary,
    ExtractionResult, get_metadata_extractor
)
from .quality_assessment import (
    QualityMetric, QualityReport, PerformanceMetrics,
    QualityAssessor, ContentCompletenessAssessor, StructureIntegrityAssessor,
    MetadataAccuracyAssessor, PerformanceMonitor, QualityAssessmentSystem,
    get_quality_assessment_system
)
from .security import (
    SecurityCheck, FileSecurityProfile, FileValidator, FileSanitizer,
    ContentAnalyzer, SecurityManager
)

__version__ = "0.1.0"
__author__ = "RAGPrep Team"
__description__ = "RAG Document Processing Utility"
__url__ = "https://github.com/Chunkys0up7/RagPrep"

# Core classes and functions
__all__ = [
    # Configuration
    "Config", "get_config", "reload_config",

    # Document Parsing
    "DocumentParser", "PDFParser", "DOCXParser", "TextParser", "HTMLParser",
    "CascadingDocumentParser", "ParsedContent", "ParserResult", "get_document_parser",

    # Document Chunking
    "DocumentChunker", "FixedSizeChunker", "StructuralChunker", "SemanticChunker",
    "HybridChunker", "DocumentChunkerFactory", "DocumentChunk", "ChunkingResult",
    "get_document_chunker",

    # Metadata Extraction
    "MetadataExtractor", "BasicMetadataExtractor", "LLMMetadataExtractor",
    "MetadataExtractorFactory", "Entity", "Topic", "Relationship", "Summary",
    "ExtractionResult", "get_metadata_extractor",

    # Quality Assessment
    "QualityMetric", "QualityReport", "PerformanceMetrics",
    "QualityAssessor", "ContentCompletenessAssessor", "StructureIntegrityAssessor",
    "MetadataAccuracyAssessor", "PerformanceMonitor", "QualityAssessmentSystem",
    "get_quality_assessment_system",

    # Security
    "SecurityCheck", "FileSecurityProfile", "FileValidator", "FileSanitizer",
    "ContentAnalyzer", "SecurityManager",
]
