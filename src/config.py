"""
Configuration Management

This module handles loading and managing configuration for the RAG document processing utility.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class DocumentProcessingConfig(BaseModel):
    """Configuration for document processing settings."""
    supported_formats: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    chunking: Dict[str, Any] = Field(default_factory=dict)
    metadata_extraction: Dict[str, Any] = Field(default_factory=dict)


class MultimodalConfig(BaseModel):
    """Configuration for multimodal processing."""
    enabled: bool = True
    tables: Dict[str, Any] = Field(default_factory=dict)
    images: Dict[str, Any] = Field(default_factory=dict)
    math_content: Dict[str, Any] = Field(default_factory=dict)


class VectorDBConfig(BaseModel):
    """Configuration for vector database settings."""
    type: str = "chromadb"
    config: Dict[str, Any] = Field(default_factory=dict)
    indexing: Dict[str, Any] = Field(default_factory=dict)


class QualityAssuranceConfig(BaseModel):
    """Configuration for quality assurance."""
    enabled: bool = True
    thresholds: Dict[str, float] = Field(default_factory=dict)
    validation: Dict[str, bool] = Field(default_factory=dict)
    continuous_improvement: Dict[str, bool] = Field(default_factory=dict)


class PerformanceConfig(BaseModel):
    """Configuration for performance settings."""
    batch_size: int = 10
    max_workers: int = 4
    caching: Dict[str, Any] = Field(default_factory=dict)
    memory: Dict[str, Any] = Field(default_factory=dict)
    monitoring: Dict[str, bool] = Field(default_factory=dict)


class OutputConfig(BaseModel):
    """Configuration for output settings."""
    formats: Dict[str, list] = Field(default_factory=dict)
    directories: Dict[str, str] = Field(default_factory=dict)
    naming: Dict[str, str] = Field(default_factory=dict)


class LoggingConfig(BaseModel):
    """Configuration for logging settings."""
    level: str = "INFO"
    format: str = "structured"
    output: Dict[str, Any] = Field(default_factory=dict)
    structured: Dict[str, bool] = Field(default_factory=dict)
    performance: Dict[str, bool] = Field(default_factory=dict)


class Config(BaseSettings):
    """Main configuration class for the RAG document processing utility."""
    
    # Environment variables
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    pinecone_api_key: Optional[str] = Field(default=None, env="PINECONE_API_KEY")
    pinecone_environment: Optional[str] = Field(default=None, env="PINECONE_ENVIRONMENT")
    weaviate_api_key: Optional[str] = Field(default=None, env="WEAVIATE_API_KEY")
    weaviate_url: Optional[str] = Field(default=None, env="WEAVIATE_URL")
    
    # Document processing configuration
    document_processing: DocumentProcessingConfig = Field(default_factory=DocumentProcessingConfig)
    multimodal_processing: MultimodalConfig = Field(default_factory=MultimodalConfig)
    vector_database: VectorDBConfig = Field(default_factory=VectorDBConfig)
    quality_assurance: QualityAssuranceConfig = Field(default_factory=QualityAssuranceConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # Default values
    default_chunk_size: int = Field(default=1000, env="DEFAULT_CHUNK_SIZE")
    default_chunk_overlap: int = Field(default=150, env="DEFAULT_CHUNK_OVERLAP")
    max_document_size_mb: int = Field(default=100, env="MAX_DOCUMENT_SIZE_MB")
    supported_formats: str = Field(default="pdf,docx,html,txt,md", env="SUPPORTED_FORMATS")
    
    # LLM configuration
    default_llm_model: str = Field(default="gpt-4", env="DEFAULT_LLM_MODEL")
    default_embedding_model: str = Field(default="text-embedding-ada-002", env="DEFAULT_EMBEDDING_MODEL")
    max_tokens: int = Field(default=4000, env="MAX_TOKENS")
    temperature: float = Field(default=0.1, env="TEMPERATURE")
    
    # Vector database configuration
    vector_db_type: str = Field(default="chromadb", env="VECTOR_DB_TYPE")
    chroma_persist_directory: str = Field(default="./vector_db", env="CHROMA_PERSIST_DIRECTORY")
    embedding_dimension: int = Field(default=1536, env="EMBEDDING_DIMENSION")
    
    # Quality assurance settings
    min_content_accuracy: float = Field(default=0.95, env="MIN_CONTENT_ACCURACY")
    min_structure_preservation: float = Field(default=0.90, env="MIN_STRUCTURE_PRESERVATION")
    min_metadata_quality: float = Field(default=0.85, env="MIN_METADATA_QUALITY")
    max_processing_time_seconds: int = Field(default=30, env="MAX_PROCESSING_TIME_SECONDS")
    max_memory_usage_mb: int = Field(default=2048, env="MAX_MEMORY_USAGE_MB")
    
    # Processing pipeline configuration
    enable_multimodal_processing: bool = Field(default=True, env="ENABLE_MULTIMODAL_PROCESSING")
    enable_table_extraction: bool = Field(default=True, env="ENABLE_TABLE_EXTRACTION")
    enable_image_processing: bool = Field(default=True, env="ENABLE_IMAGE_PROCESSING")
    enable_math_content_processing: bool = Field(default=True, env="ENABLE_MATH_CONTENT_PROCESSING")
    
    # Logging configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="./logs/rag_prep.log", env="LOG_FILE")
    enable_structured_logging: bool = Field(default=True, env="ENABLE_STRUCTURED_LOGGING")
    
    # Development settings
    debug_mode: bool = Field(default=False, env="DEBUG_MODE")
    enable_profiling: bool = Field(default=False, env="ENABLE_PROFILING")
    save_intermediate_results: bool = Field(default=True, env="SAVE_INTERMEDIATE_RESULTS")
    test_mode: bool = Field(default=False, env="TEST_MODE")
    
    # Output configuration
    output_format: str = Field(default="json", env="OUTPUT_FORMAT")
    save_chunks_to_disk: bool = Field(default=True, env="SAVE_CHUNKS_TO_DISK")
    chunks_output_dir: str = Field(default="./output/chunks", env="CHUNKS_OUTPUT_DIR")
    metadata_output_dir: str = Field(default="./output/metadata", env="METADATA_OUTPUT_DIR")
    
    # Performance tuning
    batch_size: int = Field(default=10, env="BATCH_SIZE")
    max_workers: int = Field(default=4, env="MAX_WORKERS")
    enable_caching: bool = Field(default=True, env="ENABLE_CACHING")
    cache_ttl_hours: int = Field(default=24, env="CACHE_TTL_HOURS")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        """Initialize configuration with optional YAML file."""
        super().__init__(**kwargs)
        
        if config_path:
            self._load_yaml_config(config_path)
        
        self._setup_defaults()
        self._validate_config()
    
    def _load_yaml_config(self, config_path: str):
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
                self._update_from_dict(yaml_config)
        except Exception as e:
            raise ValueError(f"Failed to load configuration from {config_path}: {e}")
    
    def _update_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                if isinstance(value, dict) and hasattr(getattr(self, key), '__dict__'):
                    # Update nested objects
                    current = getattr(self, key)
                    for k, v in value.items():
                        if hasattr(current, k):
                            setattr(current, k, v)
                else:
                    setattr(self, key, value)
    
    def _setup_defaults(self):
        """Set up default configuration values."""
        # Ensure output directories exist
        os.makedirs(self.chunks_output_dir, exist_ok=True)
        os.makedirs(self.metadata_output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        # Set up logging
        if self.enable_structured_logging:
            self.logging.format = "structured"
    
    def _validate_config(self):
        """Validate configuration values."""
        if self.default_chunk_size <= 0:
            raise ValueError("default_chunk_size must be positive")
        
        if self.default_chunk_overlap < 0:
            raise ValueError("default_chunk_overlap must be non-negative")
        
        if self.default_chunk_overlap >= self.default_chunk_size:
            raise ValueError("default_chunk_overlap must be less than default_chunk_size")
        
        if not (0 <= self.temperature <= 2):
            raise ValueError("temperature must be between 0 and 2")
        
        if self.min_content_accuracy < 0 or self.min_content_accuracy > 1:
            raise ValueError("min_content_accuracy must be between 0 and 1")
    
    def get_parser_config(self, file_extension: str) -> Dict[str, Any]:
        """Get parser configuration for specific file type."""
        extension = file_extension.lower().lstrip('.')
        return self.document_processing.supported_formats.get(extension, {})
    
    def get_chunking_config(self, strategy: str) -> Dict[str, Any]:
        """Get chunking configuration for specific strategy."""
        return self.document_processing.chunking.strategies.get(strategy, {})
    
    def get_metadata_config(self, document_type: str) -> Dict[str, Any]:
        """Get metadata extraction configuration for document type."""
        return self.document_processing.metadata_extraction.extraction_schemas.get(
            document_type, 
            self.document_processing.metadata_extraction.extraction_schemas.get("general", {})
        )
    
    def is_format_supported(self, file_extension: str) -> bool:
        """Check if file format is supported."""
        extension = file_extension.lower().lstrip('.')
        return extension in self.document_processing.supported_formats
    
    def get_supported_formats(self) -> list:
        """Get list of supported file formats."""
        return list(self.document_processing.supported_formats.keys())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "document_processing": self.document_processing.dict(),
            "multimodal_processing": self.multimodal_processing.dict(),
            "vector_database": self.vector_database.dict(),
            "quality_assurance": self.quality_assurance.dict(),
            "performance": self.performance.dict(),
            "output": self.output.dict(),
            "logging": self.logging.dict(),
            "default_chunk_size": self.default_chunk_size,
            "default_chunk_overlap": self.default_chunk_overlap,
            "max_document_size_mb": self.max_document_size_mb,
            "supported_formats": self.supported_formats,
            "default_llm_model": self.default_llm_model,
            "default_embedding_model": self.default_embedding_model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "vector_db_type": self.vector_db_type,
            "chroma_persist_directory": self.chroma_persist_directory,
            "embedding_dimension": self.embedding_dimension,
            "min_content_accuracy": self.min_content_accuracy,
            "min_structure_preservation": self.min_structure_preservation,
            "min_metadata_quality": self.min_metadata_quality,
            "max_processing_time_seconds": self.max_processing_time_seconds,
            "max_memory_usage_mb": self.max_memory_usage_mb,
            "enable_multimodal_processing": self.enable_multimodal_processing,
            "enable_table_extraction": self.enable_table_extraction,
            "enable_image_processing": self.enable_image_processing,
            "enable_math_content_processing": self.enable_math_content_processing,
            "log_level": self.log_level,
            "log_file": self.log_file,
            "enable_structured_logging": self.enable_structured_logging,
            "debug_mode": self.debug_mode,
            "enable_profiling": self.enable_profiling,
            "save_intermediate_results": self.save_intermediate_results,
            "test_mode": self.test_mode,
            "output_format": self.output_format,
            "save_chunks_to_disk": self.save_chunks_to_disk,
            "chunks_output_dir": self.chunks_output_dir,
            "metadata_output_dir": self.metadata_output_dir,
            "batch_size": self.batch_size,
            "max_workers": self.max_workers,
            "enable_caching": self.enable_caching,
            "cache_ttl_hours": self.cache_ttl_hours
        }
