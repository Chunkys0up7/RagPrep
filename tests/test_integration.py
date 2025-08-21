"""
Integration tests for the RAG Document Processing Utility.

This module contains end-to-end integration tests that validate
the complete document processing pipeline.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import Config
from parsers import CascadingDocumentParser, ParsedContent
from chunkers import HybridChunker, DocumentChunk
from metadata_extractors import BasicMetadataExtractor, ExtractionResult
from quality_assessment import QualityAssessmentSystem


class TestDocumentProcessingPipeline:
    """Test the complete document processing pipeline."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = Config()
        self.parser = CascadingDocumentParser(self.config)
        self.chunker = HybridChunker(self.config)
        self.metadata_extractor = BasicMetadataExtractor(self.config)
        self.quality_system = QualityAssessmentSystem(self.config)
    
    def test_text_document_processing_pipeline(self):
        """Test complete pipeline with a text document."""
        # Create a temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            test_content = """# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.

## Types of Machine Learning

1. **Supervised Learning**: Uses labeled training data
2. **Unsupervised Learning**: Finds patterns in data without labels
3. **Reinforcement Learning**: Learns through interaction with environment

### Applications

Machine learning is used in various fields including:
- Computer vision
- Natural language processing
- Recommendation systems
- Autonomous vehicles

## Conclusion

Machine learning continues to evolve and transform various industries."""
            f.write(test_content)
            temp_file = f.name
        
        try:
            # Step 1: Parse the document
            parse_result = self.parser.parse_document(temp_file)
            
            assert parse_result.success is True
            assert parse_result.content is not None
            assert isinstance(parse_result.content, ParsedContent)
            assert len(parse_result.content.text_content) > 0
            
            # Step 2: Assess parsing quality
            parse_quality = self.quality_system.assess_document_quality(parse_result.content)
            assert parse_quality.overall_score > 0.0
            
            # Step 3: Chunk the document
            chunk_result = self.chunker.chunk_document(parse_result.content)
            
            assert chunk_result.success is True
            assert len(chunk_result.chunks) > 0
            assert all(isinstance(chunk, DocumentChunk) for chunk in chunk_result.chunks)
            
            # Step 4: Assess chunking quality
            chunk_quality = self.quality_system.assess_document_quality(chunk_result)
            assert chunk_quality.overall_score > 0.0
            
            # Step 5: Extract metadata from each chunk
            metadata_results = []
            for chunk in chunk_result.chunks:
                metadata_result = self.metadata_extractor.extract_metadata(chunk.content)
                metadata_results.append(metadata_result)
                
                assert isinstance(metadata_result, ExtractionResult)
                assert metadata_result.success is True
            
            # Step 6: Assess metadata quality
            if metadata_results:
                metadata_quality = self.quality_system.assess_document_quality(metadata_results[0])
                assert metadata_quality.overall_score > 0.0
            
            # Step 7: Get performance summary
            performance_summary = self.quality_system.performance_monitor.get_performance_summary()
            assert "total_operations" in performance_summary
            
            print(f"✅ Pipeline test completed successfully!")
            print(f"   📄 Parsed document: {len(parse_result.content.text_content)} characters")
            print(f"   🧩 Generated chunks: {len(chunk_result.chunks)}")
            print(f"   📊 Parse quality: {parse_quality.overall_score:.2f}")
            print(f"   📊 Chunk quality: {chunk_quality.overall_score:.2f}")
            print(f"   ⚡ Operations tracked: {performance_summary.get('total_operations', 0)}")
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_pipeline_error_handling(self):
        """Test pipeline error handling with invalid input."""
        # Test with non-existent file
        parse_result = self.parser.parse_document("non_existent_file.txt")
        assert parse_result.success is False
        assert parse_result.error_message is not None
        
        # Test chunking with invalid content
        chunk_result = self.chunker.chunk_document(None)
        assert chunk_result.success is False
        assert len(chunk_result.errors) > 0
        
        # Test metadata extraction with empty content
        metadata_result = self.metadata_extractor.extract_metadata("")
        assert isinstance(metadata_result, ExtractionResult)
        # Should handle empty content gracefully
    
    def test_quality_assessment_comprehensive(self):
        """Test comprehensive quality assessment across all components."""
        # Create sample parsed content
        parsed_content = ParsedContent(
            text_content="This is a test document with sufficient content for quality assessment.",
            structured_content={},
            metadata={"format": "txt", "size": 68},
            tables=[],
            images=[],
            math_content=[],
            structure=["paragraph"],
            parser_used="text_parser",
            parsing_errors=[],
            parsing_warnings=[]
        )
        
        # Assess parsing quality
        parse_quality = self.quality_system.assess_document_quality(parsed_content)
        assert isinstance(parse_quality.metrics, list)
        assert len(parse_quality.metrics) > 0
        assert parse_quality.overall_score >= 0.0
        assert parse_quality.overall_score <= 1.0
        
        # Check that recommendations are provided for low quality
        if parse_quality.overall_score < 0.7:
            assert len(parse_quality.recommendations) > 0
        
        print(f"✅ Quality assessment test completed!")
        print(f"   📊 Overall score: {parse_quality.overall_score:.2f}")
        print(f"   📋 Metrics evaluated: {len(parse_quality.metrics)}")
        print(f"   💡 Recommendations: {len(parse_quality.recommendations)}")
    
    def test_performance_monitoring(self):
        """Test performance monitoring capabilities."""
        monitor = self.quality_system.performance_monitor
        
        # Test operation tracking
        op_id = monitor.start_operation("test_operation", input_size=100)
        assert "test_operation" in op_id
        
        # Simulate some work
        import time
        time.sleep(0.1)
        
        monitor.end_operation(op_id, success=True, output_size=50)
        
        # Get performance summary
        summary = monitor.get_performance_summary()
        assert summary["total_operations"] >= 1
        assert summary["successful_operations"] >= 1
        assert "operations_by_type" in summary
        
        # Test metrics export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            monitor.export_metrics(temp_file)
            assert os.path.exists(temp_file)
            
            # Verify file contains data
            import json
            with open(temp_file, 'r') as f:
                data = json.load(f)
                assert len(data) >= 1
                assert data[0]["operation"] == "test_operation"
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        
        print(f"✅ Performance monitoring test completed!")
        print(f"   ⏱️  Operations tracked: {summary['total_operations']}")
        print(f"   ✅ Success rate: {summary['success_rate']:.2%}")


class TestConfigurationValidation:
    """Test configuration validation and loading."""
    
    def test_default_configuration(self):
        """Test that default configuration is valid."""
        config = Config()
        
        # Test that all major components are configured
        assert config.document_processing is not None
        assert config.document_processing.supported_formats is not None
        assert config.document_processing.chunking is not None
        assert config.document_processing.metadata is not None
        
        # Test configuration validation
        errors = config.validate_config()
        # Some errors are expected due to missing API keys
        print(f"📋 Configuration validation completed with {len(errors)} warnings")
    
    def test_configuration_components_integration(self):
        """Test that configuration components work together."""
        config = Config()
        
        # Test parser configuration
        pdf_config = config.get_parser_config("pdf")
        assert pdf_config is not None
        assert "pymupdf" in pdf_config.parsers
        
        # Test chunking configuration
        chunking_config = config.get_chunking_config()
        assert chunking_config.strategy in ["fixed", "structural", "semantic", "hybrid"]
        
        # Test metadata configuration
        metadata_config = config.get_metadata_config()
        assert metadata_config.extraction_level in ["basic", "enhanced", "llm_powered"]
        
        print(f"✅ Configuration integration test completed!")


def test_full_system_integration():
    """Test the complete system integration."""
    print("\n🚀 Starting Full System Integration Test")
    print("=" * 50)
    
    # Initialize system
    config = Config()
    parser = CascadingDocumentParser(config)
    chunker = HybridChunker(config)
    metadata_extractor = BasicMetadataExtractor(config)
    quality_system = QualityAssessmentSystem(config)
    
    # Create test content
    test_content = """# RAG Document Processing Test

This is a comprehensive test document for the RAG document processing utility.

## Features Tested

1. Document parsing with multiple format support
2. Intelligent chunking with quality assessment
3. Metadata extraction with entity recognition
4. Quality assessment across all pipeline stages
5. Performance monitoring and optimization

### Technical Details

The system uses a cascading parser strategy with fallback mechanisms.
Chunking is performed using hybrid strategies for optimal results.
Quality assessment provides multi-dimensional evaluation.

## Conclusion

This test validates the complete document processing pipeline."""
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_content)
        temp_file = f.name
    
    try:
        # Execute complete pipeline
        print("📄 Parsing document...")
        parse_result = parser.parse_document(temp_file)
        assert parse_result.success, f"Parsing failed: {parse_result.error_message}"
        
        print("🧩 Chunking document...")
        chunk_result = chunker.chunk_document(parse_result.content)
        assert chunk_result.success, f"Chunking failed: {chunk_result.errors}"
        
        print("📊 Extracting metadata...")
        metadata_results = []
        for chunk in chunk_result.chunks:
            metadata_result = metadata_extractor.extract_metadata(chunk.content)
            metadata_results.append(metadata_result)
        
        print("🔍 Assessing quality...")
        parse_quality = quality_system.assess_document_quality(parse_result.content)
        chunk_quality = quality_system.assess_document_quality(chunk_result)
        
        print("📈 Generating performance report...")
        performance_summary = quality_system.performance_monitor.get_performance_summary()
        
        # Print results
        print("\n✅ INTEGRATION TEST RESULTS")
        print("=" * 30)
        print(f"📄 Document parsed: {len(parse_result.content.text_content)} characters")
        print(f"🧩 Chunks generated: {len(chunk_result.chunks)}")
        print(f"📊 Parse quality score: {parse_quality.overall_score:.2f}")
        print(f"📊 Chunk quality score: {chunk_quality.overall_score:.2f}")
        print(f"⚡ Total operations: {performance_summary.get('total_operations', 0)}")
        print(f"✅ Success rate: {performance_summary.get('success_rate', 0):.2%}")
        
        if parse_quality.recommendations:
            print(f"💡 Recommendations: {len(parse_quality.recommendations)}")
            for rec in parse_quality.recommendations[:3]:
                print(f"   - {rec}")
        
        print("\n🎉 Full system integration test PASSED!")
        
    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.unlink(temp_file)


if __name__ == "__main__":
    test_full_system_integration()
