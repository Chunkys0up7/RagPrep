"""
Test configuration and fixtures for RAGPrep tests.
"""

import pytest
from unittest.mock import Mock, MagicMock
from src.config import Config


@pytest.fixture
def mock_config():
    """Create a properly configured mock Config object."""
    config = Mock(spec=Config)
    
    # Mock nested configuration sections
    config.parser = Mock()
    config.parser.supported_formats = [".pdf", ".docx", ".txt", ".html"]
    config.parser.max_file_size_mb = 100
    config.parser.enable_fallback = True
    config.parser.timeout_seconds = 300
    
    config.chunking = Mock()
    config.chunking.strategy = "hybrid"
    config.chunking.chunk_size = 1000
    config.chunking.overlap_size = 200
    config.chunking.min_chunk_size = 100
    config.chunking.max_chunk_size = 2000
    
    config.metadata = Mock()
    config.metadata.extraction_level = "advanced"
    config.metadata.enable_llm = True
    config.metadata.llm_model = "gpt-3.5-turbo"
    config.metadata.llm_temperature = 0.1
    config.metadata.max_entities = 50
    config.metadata.max_topics = 20
    
    config.multimodal = Mock()
    config.multimodal.enable_image_processing = True
    config.multimodal.enable_table_extraction = True
    config.multimodal.enable_math_extraction = True
    config.multimodal.image_quality_threshold = 0.7
    
    config.quality = Mock()
    config.quality.enable_quality_assessment = True
    config.quality.quality_threshold = 0.7
    config.quality.enable_performance_monitoring = True
    config.quality.metrics_retention_days = 30
    
    config.performance = Mock()
    config.performance.max_concurrent_processes = 4
    config.performance.memory_limit_gb = 8
    config.performance.enable_caching = True
    config.performance.cache_ttl_hours = 24
    
    config.output = Mock()
    config.output.output_directory = "./output"
    config.output.enable_compression = False
    config.output.output_format = "json"
    config.output.vector_store_path = "./vector_db"
    
    config.logging = Mock()
    config.logging.log_level = "INFO"
    config.logging.log_file = None
    config.logging.enable_console_logging = True
    config.logging.log_rotation = "daily"
    
    config.security = Mock()
    config.security.enable_file_validation = True
    config.security.max_file_size_mb = 100
    config.security.allowed_file_extensions = {".pdf", ".docx", ".txt", ".html"}
    config.security.allowed_mime_types = {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
        "text/html",
    }
    config.security.max_filename_length = 255
    config.security.enable_content_analysis = True
    
    # Mock method calls
    config.get_parser_config.return_value = config.parser
    config.get_chunking_config.return_value = config.chunking
    config.get_metadata_config.return_value = config.metadata
    config.get_multimodal_config.return_value = config.multimodal
    config.get_quality_config.return_value = config.quality
    config.get_performance_config.return_value = config.performance
    config.get_output_config.return_value = config.output
    config.get_logging_config.return_value = config.logging
    config.get_security_config.return_value = config.security
    
    return config


@pytest.fixture
def mock_parsed_content():
    """Create a mock ParsedContent object."""
    content = Mock()
    content.text_content = "This is sample text content for testing."
    content.structure = [
        {"type": "heading", "text": "Test Heading", "level": 1},
        {"type": "paragraph", "text": "This is a test paragraph."},
    ]
    content.metadata = {"source": "test.txt", "format": "text"}
    content.raw_content = "This is sample text content for testing."
    return content


@pytest.fixture
def mock_document_chunk():
    """Create a mock DocumentChunk object."""
    chunk = Mock()
    chunk.chunk_id = "chunk_12345"
    chunk.content = "This is a test chunk content."
    chunk.metadata = {"source": "test.txt", "chunk_index": 0}
    chunk.chunk_type = "text"
    chunk.chunk_index = 0
    chunk.quality_score = 0.8
    return chunk
