"""
Tests for configuration management
"""

import os
# Add src to path for imports
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import (ChunkingConfig, Config, LoggingConfig, MetadataConfig,
                    MultimodalConfig, OutputConfig, ParserConfig,
                    PerformanceConfig, QualityConfig)


class TestConfig:
    """Test configuration loading and validation."""

    def test_default_config(self):
        """Test default configuration values."""
        config = Config()

        # Test default document processing configuration
        assert config.document_processing.chunking.strategy == "hybrid"
        assert config.document_processing.chunking.max_chunk_size == 1000
        assert config.document_processing.chunking.min_chunk_size == 100
        assert config.document_processing.chunking.overlap_size == 200

        # Test default metadata configuration
        assert config.document_processing.metadata.extraction_level == "enhanced"
        assert config.document_processing.metadata.entity_recognition is True
        assert config.document_processing.metadata.topic_extraction is True

        # Test default multimodal configuration
        assert config.document_processing.multimodal.image_processing is True
        assert config.document_processing.multimodal.table_extraction is True
        assert config.document_processing.multimodal.ocr_engine == "tesseract"

    def test_environment_variables(self):
        """Test environment variable overrides."""
        # Set environment variables
        os.environ["OPENAI_API_KEY"] = "test_key_123"
        os.environ["CHROMA_HOST"] = "test_host"
        os.environ["CHROMA_PORT"] = "9000"

        config = Config()

        assert config.openai_api_key == "test_key_123"
        assert config.chroma_host == "test_host"
        assert config.chroma_port == 9000

        # Clean up
        del os.environ["OPENAI_API_KEY"]
        del os.environ["CHROMA_HOST"]
        del os.environ["CHROMA_PORT"]

    def test_yaml_config_loading(self):
        """Test loading configuration from YAML file."""
        # Create temporary YAML file
        yaml_content = """
document_processing:
  chunking:
    strategy: "semantic"
    max_chunk_size: 1500
    min_chunk_size: 200
  metadata:
    extraction_level: "llm_powered"
    entity_recognition: false
  multimodal:
    image_processing: false
    table_parser: "tabula"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_yaml_path = f.name

        try:
            config = Config(config_path=temp_yaml_path)

            # Test that YAML values override defaults
            assert config.document_processing.chunking.strategy == "semantic"
            assert config.document_processing.chunking.max_chunk_size == 1500
            assert config.document_processing.chunking.min_chunk_size == 200
            assert config.document_processing.metadata.extraction_level == "llm_powered"
            assert config.document_processing.metadata.entity_recognition is False
            assert config.document_processing.multimodal.image_processing is False
            assert config.document_processing.multimodal.table_parser == "tabula"

        finally:
            os.unlink(temp_yaml_path)

    def test_config_validation(self):
        """Test configuration validation."""
        config = Config()

        # Test valid configuration
        errors = config.validate_config()
        assert len(errors) == 0

        # Test invalid chunking configuration
        config.document_processing.chunking.max_chunk_size = 50  # Below minimum
        errors = config.validate_config()
        assert len(errors) > 0

        # Reset to valid value
        config.document_processing.chunking.max_chunk_size = 1000

    def test_format_support(self):
        """Test document format support checking."""
        config = Config()

        assert config.is_format_supported("pdf") is True
        assert config.is_format_supported("docx") is True
        assert config.is_format_supported("txt") is True
        assert config.is_format_supported("html") is True
        assert config.is_format_supported("md") is True
        assert config.is_format_supported("unknown") is False

    def test_parser_config_retrieval(self):
        """Test parser configuration retrieval."""
        config = Config()

        pdf_config = config.get_parser_config("pdf")
        assert pdf_config is not None
        assert pdf_config.parsers == ["marker", "pymupdf", "unstructured"]
        assert pdf_config.priority == 1
        assert pdf_config.chunking_strategy == "semantic"
        assert pdf_config.metadata_extraction == "enhanced"

        # Test case insensitivity
        pdf_config_upper = config.get_parser_config("PDF")
        assert pdf_config_upper is not None

        # Test unsupported format
        unknown_config = config.get_parser_config("unknown")
        assert unknown_config is None

    def test_chunking_config_retrieval(self):
        """Test chunking configuration retrieval."""
        config = Config()

        chunking_config = config.get_chunking_config()
        assert chunking_config.strategy == "hybrid"
        assert chunking_config.max_chunk_size == 1000
        assert chunking_config.min_chunk_size == 100
        assert chunking_config.overlap_size == 200
        assert chunking_config.semantic_threshold == 0.7

    def test_metadata_config_retrieval(self):
        """Test metadata configuration retrieval."""
        config = Config()

        metadata_config = config.get_metadata_config()
        assert metadata_config.extraction_level == "enhanced"
        assert metadata_config.llm_provider == "openai"
        assert metadata_config.entity_recognition is True
        assert metadata_config.topic_extraction is True
        assert metadata_config.relationship_extraction is True
        assert metadata_config.summarization is True

    def test_multimodal_config_retrieval(self):
        """Test multimodal configuration retrieval."""
        config = Config()

        multimodal_config = config.get_multimodal_config()
        assert multimodal_config.image_processing is True
        assert multimodal_config.table_extraction is True
        assert multimodal_config.math_processing is True
        assert multimodal_config.ocr_engine == "tesseract"
        assert multimodal_config.table_parser == "camelot"

    def test_quality_config_retrieval(self):
        """Test quality configuration retrieval."""
        config = Config()

        quality_config = config.get_quality_config()
        assert quality_config.enable_validation is True
        assert quality_config.content_completeness is True
        assert quality_config.structure_integrity is True
        assert quality_config.metadata_accuracy is True
        assert quality_config.embedding_quality is True
        assert quality_config.performance_monitoring is True

    def test_performance_config_retrieval(self):
        """Test performance configuration retrieval."""
        config = Config()

        performance_config = config.get_performance_config()
        assert performance_config.max_workers == 4
        assert performance_config.batch_size == 10
        assert performance_config.memory_limit == "2GB"
        assert performance_config.timeout == 300

    def test_output_config_retrieval(self):
        """Test output configuration retrieval."""
        config = Config()

        output_config = config.get_output_config()
        assert output_config.output_dir == "output"
        assert output_config.chunk_dir == "chunks"
        assert output_config.metadata_dir == "metadata"
        assert output_config.embedding_dir == "embeddings"
        assert output_config.temp_dir == "temp"
        assert output_config.preserve_structure is True

    def test_config_to_dict(self):
        """Test configuration serialization to dictionary."""
        config = Config()

        config_dict = config.to_dict()

        assert "openai_api_key" in config_dict
        assert "document_processing" in config_dict
        assert "chunking" in config_dict["document_processing"]
        assert "metadata" in config_dict["document_processing"]
        assert "multimodal" in config_dict["document_processing"]

    def test_parser_config_validation(self):
        """Test parser configuration validation."""
        # Test valid chunking strategy
        valid_config = ParserConfig(
            parsers=["test_parser"],
            priority=1,
            chunking_strategy="semantic",
            metadata_extraction="basic",
        )
        assert valid_config.chunking_strategy == "semantic"

        # Test invalid chunking strategy
        with pytest.raises(ValueError, match="chunking_strategy must be one of"):
            ParserConfig(
                parsers=["test_parser"],
                priority=1,
                chunking_strategy="invalid",
                metadata_extraction="basic",
            )

        # Test invalid metadata extraction level
        with pytest.raises(ValueError, match="metadata_extraction must be one of"):
            ParserConfig(
                parsers=["test_parser"],
                priority=1,
                chunking_strategy="semantic",
                metadata_extraction="invalid",
            )

    def test_chunking_config_validation(self):
        """Test chunking configuration validation."""
        # Test valid configuration
        valid_config = ChunkingConfig(
            strategy="hybrid",
            max_chunk_size=1000,
            min_chunk_size=100,
            overlap_size=200,
            semantic_threshold=0.7,
        )
        assert valid_config.max_chunk_size == 1000

        # Test invalid max chunk size
        with pytest.raises(ValueError, match="max_chunk_size must be at least 100"):
            ChunkingConfig(max_chunk_size=50)

        # Test invalid min chunk size
        with pytest.raises(ValueError, match="min_chunk_size must be at least 50"):
            ChunkingConfig(min_chunk_size=25)

        # Test invalid overlap size
        with pytest.raises(ValueError, match="overlap_size must be non-negative"):
            ChunkingConfig(overlap_size=-10)

        # Test invalid semantic threshold
        with pytest.raises(
            ValueError, match="semantic_threshold must be between 0.0 and 1.0"
        ):
            ChunkingConfig(semantic_threshold=1.5)

    def test_metadata_config_validation(self):
        """Test metadata configuration validation."""
        # Test valid configuration
        valid_config = MetadataConfig(
            extraction_level="enhanced",
            llm_provider="openai",
            entity_recognition=True,
            topic_extraction=True,
        )
        assert valid_config.extraction_level == "enhanced"

        # Test invalid extraction level
        with pytest.raises(ValueError, match="extraction_level must be one of"):
            MetadataConfig(extraction_level="invalid")

    def test_multimodal_config_validation(self):
        """Test multimodal configuration validation."""
        # Test valid configuration
        valid_config = MultimodalConfig(
            image_processing=True,
            table_extraction=True,
            ocr_engine="tesseract",
            table_parser="camelot",
        )
        assert valid_config.ocr_engine == "tesseract"

        # Test invalid OCR engine
        with pytest.raises(ValueError, match="ocr_engine must be one of"):
            MultimodalConfig(ocr_engine="invalid")

        # Test invalid table parser
        with pytest.raises(ValueError, match="table_parser must be one of"):
            MultimodalConfig(table_parser="invalid")

    def test_global_config_functions(self):
        """Test global configuration functions."""
        from config import get_config, reload_config

        # Test get_config
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2  # Same instance

        # Test reload_config
        config3 = reload_config()
        assert config3 is not config1  # New instance


if __name__ == "__main__":
    pytest.main([__file__])
