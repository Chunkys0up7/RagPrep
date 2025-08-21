"""
Configuration management for RAG Document Processing Utility.

It handles loading from YAML files, environment variables, and provides
validation and type safety.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

# Configure logging
logger = logging.getLogger(__name__)


class ParserConfig(BaseModel):
    """Configuration for document parsing."""

    supported_formats: List[str] = Field(
        default=[".pdf", ".docx", ".txt", ".html"],
        description="Supported document formats",
    )
    max_file_size_mb: int = Field(default=100, description="Maximum file size in MB")
    enable_fallback: bool = Field(
        default=True, description="Enable fallback parsing strategies"
    )
    timeout_seconds: int = Field(default=300, description="Parsing timeout in seconds")


class ChunkingConfig(BaseModel):
    """Configuration for document chunking."""

    strategy: str = Field(default="hybrid", description="Chunking strategy to use")
    chunk_size: int = Field(default=1000, description="Target chunk size in characters")
    overlap_size: int = Field(
        default=200, description="Overlap between chunks in characters"
    )
    min_chunk_size: int = Field(
        default=100, description="Minimum chunk size in characters"
    )
    max_chunk_size: int = Field(
        default=2000, description="Maximum chunk size in characters"
    )


class MetadataConfig(BaseModel):
    """Configuration for metadata extraction."""

    extraction_level: str = Field(
        default="advanced", description="Metadata extraction level"
    )
    enable_llm: bool = Field(default=True, description="Enable LLM-powered extraction")
    llm_model: str = Field(default="gpt-3.5-turbo", description="LLM model to use")
    llm_temperature: float = Field(default=0.1, description="LLM temperature setting")
    max_entities: int = Field(default=50, description="Maximum entities to extract")
    max_topics: int = Field(default=20, description="Maximum topics to extract")


class MultimodalConfig(BaseModel):
    """Configuration for multimodal content processing."""

    enable_image_processing: bool = Field(
        default=True, description="Enable image processing"
    )
    enable_table_extraction: bool = Field(
        default=True, description="Enable table extraction"
    )
    enable_math_extraction: bool = Field(
        default=True, description="Enable math expression extraction"
    )
    image_quality_threshold: float = Field(
        default=0.7, description="Minimum image quality score"
    )


class QualityConfig(BaseModel):
    """Configuration for quality assessment."""

    enable_quality_assessment: bool = Field(
        default=True, description="Enable quality assessment"
    )
    quality_threshold: float = Field(
        default=0.7, description="Minimum quality score threshold"
    )
    enable_performance_monitoring: bool = Field(
        default=True, description="Enable performance monitoring"
    )
    metrics_retention_days: int = Field(
        default=30, description="Days to retain performance metrics"
    )


class PerformanceConfig(BaseModel):
    """Configuration for performance settings."""

    max_concurrent_processes: int = Field(
        default=4, description="Maximum concurrent processes"
    )
    memory_limit_gb: int = Field(default=8, description="Memory limit in GB")
    enable_caching: bool = Field(default=True, description="Enable result caching")
    cache_ttl_hours: int = Field(default=24, description="Cache TTL in hours")


class OutputConfig(BaseModel):
    """Configuration for output settings."""

    output_directory: str = Field(
        default="./output", description="Output directory path"
    )
    enable_compression: bool = Field(
        default=False, description="Enable output compression"
    )
    output_format: str = Field(default="json", description="Output format")
    vector_store_path: str = Field(
        default="./vector_db", description="Vector store path"
    )


class LoggingConfig(BaseModel):
    """Configuration for logging."""

    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[str] = Field(default=None, description="Log file path")
    enable_console_logging: bool = Field(
        default=True, description="Enable console logging"
    )
    log_rotation: str = Field(default="daily", description="Log rotation policy")


class SecurityConfig(BaseModel):
    """Configuration for security settings."""

    enable_file_validation: bool = Field(
        default=True, description="Enable file validation"
    )
    max_file_size_mb: int = Field(default=100, description="Maximum file size in MB")
    allowed_file_extensions: set = Field(
        default={".pdf", ".docx", ".txt", ".html"},
        description="Allowed file extensions",
    )
    allowed_mime_types: set = Field(
        default={
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain",
            "text/html",
        },
        description="Allowed MIME types",
    )
    max_filename_length: int = Field(default=255, description="Maximum filename length")
    enable_content_analysis: bool = Field(
        default=True, description="Enable content security analysis"
    )


class Config(BaseModel):
    """Main configuration class for the RAG Document Processing Utility."""

    app_name: str = Field(
        default="RAG Document Processing Utility",
        description="Application name",
    )
    version: str = Field(default="0.1.0", description="Application version")
    debug: bool = Field(default=False, description="Enable debug mode")
    config_path: Optional[str] = Field(default=None, exclude=True)

    # Configuration sections
    parser: ParserConfig = Field(
        default_factory=ParserConfig,
        description="Parser configuration",
    )
    chunking: ChunkingConfig = Field(
        default_factory=ChunkingConfig,
        description="Chunking configuration",
    )
    metadata: MetadataConfig = Field(
        default_factory=MetadataConfig,
        description="Metadata configuration",
    )
    multimodal: MultimodalConfig = Field(
        default_factory=MultimodalConfig,
        description="Multimodal configuration",
    )
    quality: QualityConfig = Field(
        default_factory=QualityConfig,
        description="Quality configuration",
    )
    performance: PerformanceConfig = Field(
        default_factory=PerformanceConfig,
        description="Performance configuration",
    )
    output: OutputConfig = Field(
        default_factory=OutputConfig,
        description="Output configuration",
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig,
        description="Logging configuration",
    )
    security: SecurityConfig = Field(
        default_factory=SecurityConfig,
        description="Security configuration",
    )

    @field_validator("config_path", mode="before")
    @classmethod
    def set_config_path(cls, v):
        """Set config_path if not provided."""
        if v is None:
            return "./config/config.yaml"
        return v

    @model_validator(mode="after")
    def validate_config(self):
        """Validate configuration values."""
        # Ensure output directory exists
        output_dir = self.output.output_directory
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Validate file size limits
        parser_max_size = self.parser.max_file_size_mb
        security_max_size = self.security.max_file_size_mb
        if parser_max_size != security_max_size:
            logger.warning(
                "Parser and security max file sizes differ. "
                "Using the more restrictive value."
            )
            min_size = min(parser_max_size, security_max_size)
            self.parser.max_file_size_mb = min_size
            self.security.max_file_size_mb = min_size

        return self

    def get_parser_config(self) -> ParserConfig:
        """Get parser configuration."""
        return self.parser

    def get_chunking_config(self) -> ChunkingConfig:
        """Get chunking configuration."""
        return self.chunking

    def get_metadata_config(self) -> MetadataConfig:
        """Get metadata configuration."""
        return self.metadata

    def get_multimodal_config(self) -> MultimodalConfig:
        """Get multimodal configuration."""
        return self.multimodal

    def get_quality_config(self) -> QualityConfig:
        """Get quality configuration."""
        return self.quality

    def get_performance_config(self) -> PerformanceConfig:
        """Get performance configuration."""
        return self.performance

    def get_output_config(self) -> OutputConfig:
        """Get output configuration."""
        return self.output

    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration."""
        return self.logging

    def get_security_config(self) -> SecurityConfig:
        """Get security configuration."""
        return self.security

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()

    def save_to_file(self, file_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = self.to_dict()
        with open(file_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    @classmethod
    def from_file(cls, file_path: str) -> "Config":
        """Load configuration from YAML file."""
        if not os.path.exists(file_path):
            logger.warning(f"Config file {file_path} not found, using defaults")
            return cls()

        try:
            with open(file_path, "r") as f:
                config_data = yaml.safe_load(f)
            return cls(**config_data)
        except Exception as e:
            logger.error(f"Error loading config from {file_path}: {e}")
            logger.info("Using default configuration")
            return cls()

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        # This would be implemented to read from environment variables
        # For now, return default config
        return cls()


# Global configuration instance
_config_instance: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance


def reload_config(config_path: Optional[str] = None) -> Config:
    """Reload configuration from file."""
    global _config_instance
    if config_path:
        _config_instance = Config.from_file(config_path)
    else:
        _config_instance = Config()
    return _config_instance
