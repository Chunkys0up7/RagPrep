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
from .mkdocs_exporter import (
    MkDocsExporter,
    MkDocsExportResult,
    MkDocsPage,
    MkDocsSection,
    get_mkdocs_exporter,
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
    
    # Document processing
    "DocumentProcessor",
    "ProcessingResult",
    
    # Parsers
    "DocumentParser",
    "CascadingDocumentParser",
    "PDFParser",
    "DOCXParser",
    "HTMLParser",
    "TextParser",
    "ParsedContent",
    "ParserResult",
    "get_document_parser",
    
    # Chunkers
    "DocumentChunker",
    "FixedSizeChunker",
    "StructuralChunker",
    "SemanticChunker",
    "HybridChunker",
    "DocumentChunk",
    "ChunkingResult",
    "get_document_chunker",
    
    # Metadata extraction
    "MetadataExtractor",
    "BasicMetadataExtractor",
    "LLMMetadataExtractor",
    "Entity",
    "Topic",
    "Relationship",
    "Summary",
    "ExtractionResult",
    "get_metadata_extractor",
    
    # Quality assessment
    "QualityAssessmentSystem",
    "QualityAssessor",
    "QualityMetric",
    "QualityReport",
    "PerformanceMetrics",
    "PerformanceMonitor",
    "get_quality_assessment_system",
    
    # Security
    "SecurityManager",
    "FileValidator",
    "ContentAnalyzer",
    "FileSanitizer",
    "SecurityCheck",
    "FileSecurityProfile",
    
    # Vector storage
    "VectorStore",
    "FileBasedVectorStore",
    "ChromaDBVectorStore",
    "get_vector_store",
    
    # MkDocs export
    "MkDocsExporter",
    "MkDocsExportResult",
    "MkDocsPage",
    "MkDocsSection",
    "get_mkdocs_exporter",
]
