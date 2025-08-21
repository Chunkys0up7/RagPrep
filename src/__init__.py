"""
RAG Document Processing Utility

Provides intelligent parsing, chunking, metadata extraction, and quality
assessment.
"""

from .chunkers import (
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
from .config import Config, get_config, reload_config
from .metadata_extractors import (
    BasicMetadataExtractor,
    Entity,
    ExtractionResult,
    LLMMetadataExtractor,
    MetadataExtractor,
    MetadataExtractorFactory,
    Relationship,
    Summary,
    Topic,
    get_metadata_extractor,
)
from .parsers import (
    CascadingDocumentParser,
    DocumentParser,
    DOCXParser,
    HTMLParser,
    ParsedContent,
    ParserResult,
    PDFParser,
    TextParser,
    get_document_parser,
)
from .quality_assessment import (
    ContentCompletenessAssessor,
    MetadataAccuracyAssessor,
    PerformanceMetrics,
    PerformanceMonitor,
    QualityAssessmentSystem,
    QualityAssessor,
    QualityMetric,
    QualityReport,
    StructureIntegrityAssessor,
    get_quality_assessment_system,
)
from .security import (
    ContentAnalyzer,
    FileSanitizer,
    FileSecurityProfile,
    FileValidator,
    SecurityCheck,
    SecurityManager,
)
from .vector_store import (
    ChromaDBVectorStore,
    FileBasedVectorStore,
    VectorStore,
    get_vector_store,
)

__version__ = "0.1.0"
__author__ = "RAGPrep Team"
__description__ = "RAG Document Processing Utility"
__url__ = "https://github.com/Chunkys0up7/RagPrep"

# Core classes and functions
__all__ = [
    # Configuration
    "Config",
    "get_config",
    "reload_config",
    # Document Parsing
    "DocumentParser",
    "PDFParser",
    "DOCXParser",
    "TextParser",
    "HTMLParser",
    "CascadingDocumentParser",
    "ParsedContent",
    "ParserResult",
    "get_document_parser",
    # Document Chunking
    "DocumentChunker",
    "FixedSizeChunker",
    "StructuralChunker",
    "SemanticChunker",
    "HybridChunker",
    "DocumentChunkerFactory",
    "DocumentChunk",
    "ChunkingResult",
    "get_document_chunker",
    # Metadata Extraction
    "MetadataExtractor",
    "BasicMetadataExtractor",
    "LLMMetadataExtractor",
    "MetadataExtractorFactory",
    "Entity",
    "Topic",
    "Relationship",
    "Summary",
    "ExtractionResult",
    "get_metadata_extractor",
    # Quality Assessment
    "QualityMetric",
    "QualityReport",
    "PerformanceMetrics",
    "QualityAssessor",
    "ContentCompletenessAssessor",
    "StructureIntegrityAssessor",
    "MetadataAccuracyAssessor",
    "PerformanceMonitor",
    "QualityAssessmentSystem",
    "get_quality_assessment_system",
    # Security
    "SecurityCheck",
    "FileSecurityProfile",
    "FileValidator",
    "FileSanitizer",
    "ContentAnalyzer",
    "SecurityManager",
    # Vector Storage
    "VectorStore",
    "FileBasedVectorStore",
    "ChromaDBVectorStore",
    "get_vector_store",
]
