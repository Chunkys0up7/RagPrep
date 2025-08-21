"""
Tests for configuration management
"""

import pytest
import tempfile
import os
from pathlib import Path

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import Config


class TestConfig:
    """Test configuration loading and validation."""
    
    def test_default_config(self):
        """Test that default configuration loads correctly."""
        config = Config()
        
        # Check basic settings
        assert config.default_chunk_size == 1000
        assert config.default_chunk_overlap == 150
        assert config.max_document_size_mb == 100
        assert config.vector_db_type == "chromadb"
        
        # Check that output directories are created
        assert os.path.exists(config.chunks_output_dir)
        assert os.path.exists(config.metadata_output_dir)
    
    def test_environment_variables(self):
        """Test environment variable loading."""
        # Set test environment variables
        os.environ["DEFAULT_CHUNK_SIZE"] = "2000"
        os.environ["DEFAULT_CHUNK_OVERLAP"] = "300"
        
        config = Config()
        
        assert config.default_chunk_size == 2000
        assert config.default_chunk_overlap == 300
        
        # Clean up
        del os.environ["DEFAULT_CHUNK_SIZE"]
        del os.environ["DEFAULT_CHUNK_OVERLAP"]
    
    def test_yaml_config_loading(self):
        """Test YAML configuration file loading."""
        # Create temporary YAML config
        yaml_content = """
document_processing:
  supported_formats:
    pdf:
      parsers: [pymupdf, marker]
      priority: 1
      chunking_strategy: semantic
      metadata_extraction: enhanced
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name
        
        try:
            config = Config(temp_path)
            
            # Check that YAML config was loaded
            pdf_config = config.get_parser_config('.pdf')
            assert 'pymupdf' in pdf_config.get('parsers', [])
            assert pdf_config.get('chunking_strategy') == 'semantic'
            
        finally:
            os.unlink(temp_path)
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid chunk size
        with pytest.raises(ValueError, match="default_chunk_size must be positive"):
            Config(default_chunk_size=0)
        
        # Test invalid chunk overlap
        with pytest.raises(ValueError, match="default_chunk_overlap must be non-negative"):
            Config(default_chunk_overlap=-1)
        
        # Test chunk overlap >= chunk size
        with pytest.raises(ValueError, match="default_chunk_overlap must be less than default_chunk_size"):
            Config(default_chunk_size=1000, default_chunk_overlap=1000)
        
        # Test invalid temperature
        with pytest.raises(ValueError, match="temperature must be between 0 and 2"):
            Config(temperature=3.0)
        
        # Test invalid accuracy
        with pytest.raises(ValueError, match="min_content_accuracy must be between 0 and 1"):
            Config(min_content_accuracy=1.5)
    
    def test_format_support(self):
        """Test file format support checking."""
        config = Config()
        
        # Test supported formats
        assert config.is_format_supported('.pdf')
        assert config.is_format_supported('.docx')
        assert config.is_format_supported('.html')
        
        # Test unsupported formats
        assert not config.is_format_supported('.xyz')
        assert not config.is_format_supported('.unknown')
    
    def test_parser_config_retrieval(self):
        """Test parser configuration retrieval."""
        config = Config()
        
        # Test getting parser config for supported format
        pdf_config = config.get_parser_config('.pdf')
        assert isinstance(pdf_config, dict)
        
        # Test getting parser config for unsupported format
        unknown_config = config.get_parser_config('.xyz')
        assert unknown_config == {}
    
    def test_chunking_config_retrieval(self):
        """Test chunking configuration retrieval."""
        config = Config()
        
        # Test getting chunking config for strategy
        semantic_config = config.get_chunking_config('semantic')
        assert isinstance(semantic_config, dict)
        
        # Test getting chunking config for unknown strategy
        unknown_config = config.get_chunking_config('unknown')
        assert unknown_config == {}
    
    def test_metadata_config_retrieval(self):
        """Test metadata configuration retrieval."""
        config = Config()
        
        # Test getting metadata config for document type
        academic_config = config.get_metadata_config('academic_paper')
        assert isinstance(academic_config, dict)
        
        # Test fallback to general config
        general_config = config.get_metadata_config('unknown_type')
        assert isinstance(general_config, dict)
    
    def test_config_to_dict(self):
        """Test configuration serialization to dictionary."""
        config = Config()
        config_dict = config.to_dict()
        
        # Check that all expected keys are present
        expected_keys = [
            'default_chunk_size', 'default_chunk_overlap', 'max_document_size_mb',
            'vector_db_type', 'min_content_accuracy', 'min_structure_preservation',
            'min_metadata_quality', 'max_processing_time_seconds', 'max_memory_usage_mb'
        ]
        
        for key in expected_keys:
            assert key in config_dict
        
        # Check that values match
        assert config_dict['default_chunk_size'] == config.default_chunk_size
        assert config_dict['vector_db_type'] == config.vector_db_type


if __name__ == "__main__":
    pytest.main([__file__])
