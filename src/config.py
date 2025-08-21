"""
Configuration Management

This module provides comprehensive configuration management for the RAG Document Processing Utility.
It handles loading from YAML files, environment variables, and provides validation and type safety.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field, validator, root_validator
from pydantic_settings import BaseSettings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParserConfig(BaseModel):
    """Configuration for document parsers."""
    parsers: List[str] = Field(default_factory=list, description="List of parser names to use")
    priority: int = Field(default=1, description="Priority order for this parser")
    chunking_strategy: str = Field(default="semantic", description="Chunking strategy for this format")
    metadata_extraction: str = Field(default="basic", description="Metadata extraction level")
    
    @validator('chunking_strategy')
    def validate_chunking_strategy(cls, v):
        valid_strategies = ['semantic', 'fixed', 'structural', 'hybrid']
        if v not in valid_strategies:
            raise ValueError(f'chunking_strategy must be one of {valid_strategies}')
        return v
    
    @validator('metadata_extraction')
    def validate_metadata_extraction(cls, v):
        valid_levels = ['basic', 'enhanced', 'llm_powered']
        if v not in valid_levels:
            raise ValueError(f'metadata_extraction must be one of {valid_levels}')
        return v


class ChunkingConfig(BaseModel):
    """Configuration for document chunking."""
    strategy: str = Field(default="hybrid", description="Primary chunking strategy")
    max_chunk_size: int = Field(default=1000, description="Maximum chunk size in characters")
    min_chunk_size: int = Field(default=100, description="Minimum chunk size in characters")
    overlap_size: int = Field(default=200, description="Overlap size between chunks")
    semantic_threshold: float = Field(default=0.7, description="Semantic similarity threshold")
    
    @validator('max_chunk_size')
    def validate_max_chunk_size(cls, v):
        if v < 100:
            raise ValueError('max_chunk_size must be at least 100')
        return v
    
    @validator('min_chunk_size')
    def validate_min_chunk_size(cls, v):
        if v < 50:
            raise ValueError('min_chunk_size must be at least 50')
        return v
    
    @validator('overlap_size')
    def validate_overlap_size(cls, v):
        if v < 0:
            raise ValueError('overlap_size must be non-negative')
        return v
    
    @validator('semantic_threshold')
    def validate_semantic_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('semantic_threshold must be between 0.0 and 1.0')
        return v


class MetadataConfig(BaseModel):
    """Metadata extraction configuration."""
    extraction_level: str = Field(default="enhanced", description="Metadata extraction level")
    llm_provider: str = Field(default="openai", description="LLM provider for enhanced extraction")
    llm_model: str = Field(default="gpt-3.5-turbo", description="LLM model to use for extraction")
    llm_temperature: float = Field(default=0.1, description="Temperature for LLM generation")
    entity_recognition: bool = Field(default=True, description="Enable entity recognition")
    topic_extraction: bool = Field(default=True, description="Enable topic extraction")
    relationship_extraction: bool = Field(default=True, description="Enable relationship extraction")
    summarization: bool = Field(default=True, description="Enable document summarization")
    
    @validator('extraction_level')
    def validate_extraction_level(cls, v):
        if v not in ['basic', 'enhanced']:
            raise ValueError('extraction_level must be either "basic" or "enhanced"')
        return v
    
    @validator('llm_provider')
    def validate_llm_provider(cls, v):
        if v not in ['openai', 'anthropic', 'local']:
            raise ValueError('llm_provider must be one of: openai, anthropic, local')
        return v
    
    @validator('llm_temperature')
    def validate_llm_temperature(cls, v):
        if not 0.0 <= v <= 2.0:
            raise ValueError('llm_temperature must be between 0.0 and 2.0')
        return v


class MultimodalConfig(BaseModel):
    """Configuration for multimodal content processing."""
    image_processing: bool = Field(default=True, description="Enable image processing")
    table_extraction: bool = Field(default=True, description="Enable table extraction")
    math_processing: bool = Field(default=True, description="Enable mathematical content processing")
    ocr_engine: str = Field(default="tesseract", description="OCR engine for image processing")
    table_parser: str = Field(default="camelot", description="Table parsing library")
    
    @validator('ocr_engine')
    def validate_ocr_engine(cls, v):
        valid_engines = ['tesseract', 'easyocr', 'paddleocr']
        if v not in valid_engines:
            raise ValueError(f'ocr_engine must be one of {valid_engines}')
        return v
    
    @validator('table_parser')
    def validate_table_parser(cls, v):
        valid_parsers = ['camelot', 'tabula', 'pdfplumber']
        if v not in valid_parsers:
            raise ValueError(f'table_parser must be one of {valid_parsers}')
        return v


class VectorDBConfig(BaseModel):
    """Configuration for vector database integration."""
    provider: str = Field(default="chromadb", description="Vector database provider")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="Embedding model")
    index_type: str = Field(default="hnsw", description="Index type for vector search")
    similarity_metric: str = Field(default="cosine", description="Similarity metric for search")
    
    @validator('provider')
    def validate_provider(cls, v):
        valid_providers = ['chromadb', 'pinecone', 'weaviate', 'faiss']
        if v not in valid_providers:
            raise ValueError(f'provider must be one of {valid_providers}')
        return v
    
    @validator('similarity_metric')
    def validate_similarity_metric(cls, v):
        valid_metrics = ['cosine', 'euclidean', 'dot_product']
        if v not in valid_metrics:
            raise ValueError(f'similarity_metric must be one of {valid_metrics}')
        return v


class QualityConfig(BaseModel):
    """Configuration for quality assurance."""
    enable_validation: bool = Field(default=True, description="Enable quality validation")
    content_completeness: bool = Field(default=True, description="Check content completeness")
    structure_integrity: bool = Field(default=True, description="Check structure integrity")
    metadata_accuracy: bool = Field(default=True, description="Check metadata accuracy")
    embedding_quality: bool = Field(default=True, description="Check embedding quality")
    performance_monitoring: bool = Field(default=True, description="Enable performance monitoring")


class PerformanceConfig(BaseModel):
    """Configuration for performance settings."""
    max_workers: int = Field(default=4, description="Maximum number of worker processes")
    batch_size: int = Field(default=10, description="Batch size for processing")
    memory_limit: str = Field(default="2GB", description="Memory limit for processing")
    timeout: int = Field(default=300, description="Timeout for operations in seconds")
    
    @validator('max_workers')
    def validate_max_workers(cls, v):
        if v < 1:
            raise ValueError('max_workers must be at least 1')
        return v
    
    @validator('batch_size')
    def validate_batch_size(cls, v):
        if v < 1:
            raise ValueError('batch_size must be at least 1')
        return v


class OutputConfig(BaseModel):
    """Output configuration settings."""
    output_directory: str = Field(default="output", description="Main output directory")
    chunks_directory: str = Field(default="output/chunks", description="Directory for document chunks")
    metadata_directory: str = Field(default="output/metadata", description="Directory for metadata files")
    embeddings_directory: str = Field(default="output/embeddings", description="Directory for embedding files")
    vector_store_path: str = Field(default="vector_db", description="Path for vector store data")
    enable_compression: bool = Field(default=True, description="Enable output compression")
    compression_format: str = Field(default="gzip", description="Compression format")
    enable_backup: bool = Field(default=True, description="Enable output backup")
    backup_retention_days: int = Field(default=30, ge=1, le=365, description="Backup retention period in days")


class LoggingConfig(BaseModel):
    """Configuration for logging."""
    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="Log format")
    file: Optional[str] = Field(default=None, description="Log file path")
    max_size: str = Field(default="10MB", description="Maximum log file size")
    backup_count: int = Field(default=5, description="Number of backup log files")
    
    @validator('level')
    def validate_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'level must be one of {valid_levels}')
        return v.upper()


class SecurityConfig(BaseModel):
    """Security configuration settings."""
    max_file_size_mb: int = Field(default=100, ge=1, le=1000, description="Maximum file size in MB")
    max_filename_length: int = Field(default=255, ge=1, le=1000, description="Maximum filename length")
    allowed_file_extensions: List[str] = Field(
        default=[".pdf", ".docx", ".txt", ".html", ".md", ".rtf"],
        description="Allowed file extensions"
    )
    allowed_mime_types: List[str] = Field(
        default=[
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain",
            "text/html",
            "text/markdown",
            "application/rtf"
        ],
        description="Allowed MIME types"
    )
    enable_content_scanning: bool = Field(default=True, description="Enable content scanning for threats")
    enable_file_validation: bool = Field(default=True, description="Enable file validation")
    enable_filename_sanitization: bool = Field(default=True, description="Enable filename sanitization")
    threat_level_threshold: str = Field(default="medium", description="Minimum threat level to block files")


class DocumentProcessingConfig(BaseModel):
    """Configuration for document processing."""
    supported_formats: Dict[str, ParserConfig] = Field(
        default_factory=lambda: {
            "pdf": ParserConfig(
                parsers=["marker", "pymupdf", "unstructured"],
                priority=1,
                chunking_strategy="semantic",
                metadata_extraction="enhanced"
            ),
            "docx": ParserConfig(
                parsers=["python-docx", "unstructured"],
                priority=2,
                chunking_strategy="structural",
                metadata_extraction="basic"
            ),
            "txt": ParserConfig(
                parsers=["unstructured"],
                priority=3,
                chunking_strategy="fixed",
                metadata_extraction="basic"
            )
        },
        description="Configuration for supported document formats"
    )
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    metadata: MetadataConfig = Field(default_factory=MetadataConfig)
    multimodal: MultimodalConfig = Field(default_factory=MultimodalConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


class Config(BaseSettings):
    """Main configuration class for the RAG Document Processing Utility."""
    
    # Core settings
    app_name: str = Field(default="RAG Document Processing Utility", description="Application name")
    version: str = Field(default="0.1.0", description="Application version")
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # Document processing settings
    document_processing: DocumentProcessingConfig = Field(
        default_factory=DocumentProcessingConfig,
        description="Document processing configuration"
    )
    
    # Parser settings
    parsers: ParserConfig = Field(
        default_factory=ParserConfig,
        description="Document parser configuration"
    )
    
    # Chunking settings
    chunking: ChunkingConfig = Field(
        default_factory=ChunkingConfig,
        description="Document chunking configuration"
    )
    
    # Metadata extraction settings
    metadata: MetadataConfig = Field(
        default_factory=MetadataConfig,
        description="Metadata extraction configuration"
    )
    
    # Multimodal processing settings
    multimodal: MultimodalConfig = Field(
        default_factory=MultimodalConfig,
        description="Multimodal processing configuration"
    )
    
    # Quality assessment settings
    quality: QualityConfig = Field(
        default_factory=QualityConfig,
        description="Quality assessment configuration"
    )
    
    # Performance settings
    performance: PerformanceConfig = Field(
        default_factory=PerformanceConfig,
        description="Performance configuration"
    )
    
    # Output settings
    output: OutputConfig = Field(
        default_factory=OutputConfig,
        description="Output configuration"
    )
    
    # Logging settings
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig,
        description="Logging configuration"
    )
    
    # Security settings
    security: SecurityConfig = Field(
        default_factory=SecurityConfig,
        description="Security configuration"
    )
    
    # Configuration file path (not loaded from env)
    config_path: Optional[str] = Field(default=None, exclude=True)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        """Initialize configuration with optional config file path."""
        super().__init__(**kwargs)
        
        if config_path:
            self.config_path = config_path
        
        # Load configuration from file if available
        if self.config_path and Path(self.config_path).exists():
            self._load_yaml_config()
        
        # Setup logging
        self._setup_logging()
    
    def _load_yaml_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                yaml_data = yaml.safe_load(file)
                
            if yaml_data:
                # Update document processing configuration
                if 'document_processing' in yaml_data:
                    self.document_processing = DocumentProcessingConfig(**yaml_data['document_processing'])
                    
                logger.info(f"Configuration loaded from {self.config_path}")
                
        except Exception as e:
            logger.warning(f"Failed to load YAML configuration: {e}")
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_config = self.document_processing.logging
        
        # Configure logging level
        logging.getLogger().setLevel(getattr(logging, log_config.level))
        
        # Configure formatter
        formatter = logging.Formatter(log_config.format)
        
        # Configure console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logging.getLogger().addHandler(console_handler)
        
        # Configure file handler if specified
        if log_config.file:
            file_handler = logging.FileHandler(log_config.file)
            file_handler.setFormatter(formatter)
            logging.getLogger().addHandler(file_handler)
    
    def is_format_supported(self, format_name: str) -> bool:
        """Check if a document format is supported."""
        return format_name.lower() in self.document_processing.supported_formats
    
    def get_parser_config(self, format_name: str) -> Optional[ParserConfig]:
        """Get parser configuration for a specific format."""
        format_name = format_name.lower()
        if format_name in self.document_processing.supported_formats:
            return self.document_processing.supported_formats[format_name]
        return None
    
    def get_chunking_config(self) -> ChunkingConfig:
        """Get chunking configuration."""
        return self.document_processing.chunking
    
    def get_metadata_config(self) -> MetadataConfig:
        """Get metadata extraction configuration."""
        return self.document_processing.metadata
    
    def get_multimodal_config(self) -> MultimodalConfig:
        """Get multimodal processing configuration."""
        return self.document_processing.multimodal
    
    def get_quality_config(self) -> QualityConfig:
        """Get quality assurance configuration."""
        return self.document_processing.quality
    
    def get_performance_config(self) -> PerformanceConfig:
        """Get performance configuration."""
        return self.document_processing.performance
    
    def get_output_config(self) -> OutputConfig:
        """Get output configuration."""
        return self.document_processing.output
    
    def get_vector_db_config(self) -> VectorDBConfig:
        """Get vector database configuration."""
        # This would be loaded from a separate config section
        # For now, return default configuration
        return VectorDBConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "openai_api_key": self.openai_api_key,
            "pinecone_api_key": self.pinecone_api_key,
            "weaviate_url": self.weaviate_url,
            "chroma_host": self.chroma_host,
            "chroma_port": self.chroma_port,
            "document_processing": self.document_processing.dict()
        }
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        try:
            # Validate document processing configuration
            self.document_processing.validate()
            
            # Check for required API keys based on configuration
            if (self.document_processing.metadata.extraction_level == "llm_powered" and 
                not self.openai_api_key):
                errors.append("OpenAI API key required for LLM-powered metadata extraction")
            
            # Validate output directories
            output_config = self.document_processing.output
            for dir_path in [output_config.output_dir, output_config.chunk_dir, 
                           output_config.metadata_dir, output_config.embedding_dir]:
                if not os.path.exists(dir_path):
                    try:
                        os.makedirs(dir_path, exist_ok=True)
                    except Exception as e:
                        errors.append(f"Failed to create directory {dir_path}: {e}")
            
        except Exception as e:
            errors.append(f"Configuration validation failed: {e}")
        
        return errors


# Global configuration instance
_config_instance: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """Get global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance


def reload_config(config_path: Optional[str] = None) -> Config:
    """Reload configuration from file."""
    global _config_instance
    _config_instance = Config(config_path)
    return _config_instance
