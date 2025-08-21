#!/usr/bin/env python3
"""
Comprehensive Test Runner for RAG Document Processing Utility

This script runs a comprehensive test suite to validate the entire
document processing pipeline and generates a detailed test report.
"""

import sys
import time
import json
import tempfile
import os
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from config import Config
from parsers import CascadingDocumentParser, ParsedContent
from chunkers import HybridChunker, DocumentChunk
from metadata_extractors import BasicMetadataExtractor, ExtractionResult
from quality_assessment import QualityAssessmentSystem


class TestReport:
    """Test report generator and manager."""
    
    def __init__(self):
        self.start_time = time.time()
        self.tests = []
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.warnings = []
        
    def add_test(self, name: str, passed: bool, duration: float, details: dict = None):
        """Add a test result to the report."""
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
            
        self.tests.append({
            "name": name,
            "passed": passed,
            "duration": duration,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        })
    
    def add_warning(self, message: str):
        """Add a warning to the report."""
        self.warnings.append({
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
    
    def generate_report(self) -> dict:
        """Generate a comprehensive test report."""
        total_duration = time.time() - self.start_time
        
        return {
            "test_summary": {
                "total_tests": self.total_tests,
                "passed_tests": self.passed_tests,
                "failed_tests": self.failed_tests,
                "success_rate": self.passed_tests / self.total_tests if self.total_tests > 0 else 0.0,
                "total_duration": total_duration,
                "average_test_duration": total_duration / self.total_tests if self.total_tests > 0 else 0.0
            },
            "test_results": self.tests,
            "warnings": self.warnings,
            "system_info": {
                "python_version": sys.version,
                "platform": sys.platform,
                "timestamp": datetime.now().isoformat()
            }
        }


class ComprehensiveTestSuite:
    """Comprehensive test suite for the RAG document processing utility."""
    
    def __init__(self):
        self.report = TestReport()
        self.config = None
        self.parser = None
        self.chunker = None
        self.metadata_extractor = None
        self.quality_system = None
    
    def setup_components(self):
        """Set up all system components."""
        try:
            self.config = Config()
            self.parser = CascadingDocumentParser(self.config)
            self.chunker = HybridChunker(self.config)
            self.metadata_extractor = BasicMetadataExtractor(self.config)
            self.quality_system = QualityAssessmentSystem(self.config)
            return True
        except Exception as e:
            self.report.add_warning(f"Failed to initialize components: {e}")
            return False
    
    def test_configuration_system(self):
        """Test configuration system loading and validation."""
        start_time = time.time()
        test_name = "Configuration System Test"
        
        try:
            # Test basic configuration loading
            config = Config()
            
            # Test configuration validation
            errors = config.validate_config()
            
            # Check if we have the expected configuration structure
            assert hasattr(config, 'document_processing'), "Document processing config missing"
            assert hasattr(config, 'parsers'), "Parsers config missing"
            assert hasattr(config, 'chunking'), "Chunking config missing"
            assert hasattr(config, 'metadata'), "Metadata config missing"
            assert hasattr(config, 'quality'), "Quality config missing"
            assert hasattr(config, 'security'), "Security config missing"
            
            # Validate that security configuration is properly loaded
            assert config.security.max_file_size_mb > 0, "Invalid file size limit"
            assert len(config.security.allowed_file_extensions) > 0, "No allowed file extensions"
            assert len(config.security.allowed_mime_types) > 0, "No allowed MIME types"
            
            # Test configuration reloading
            reloaded_config = reload_config()
            assert reloaded_config is not None, "Config reload failed"
            
            duration = time.time() - start_time
            self.report.add_test(test_name, True, duration, {
                "validation_errors": len(errors),
                "config_sections": len([attr for attr in dir(config) if not attr.startswith('_')]),
                "security_enabled": config.security.enable_file_validation
            })
            
        except Exception as e:
            duration = time.time() - start_time
            self.report.add_test(test_name, False, duration, {"error": str(e)})
    
    def test_document_parsing(self):
        """Test document parsing functionality."""
        start_time = time.time()
        test_name = "Document Parsing Test"
        
        try:
            # Create test documents
            test_documents = [
                ("txt", "# Test Document\n\nThis is a test document for parsing.\n\n## Section 1\nContent here."),
                ("md", "# Markdown Test\n\n**Bold text** and *italic text*.\n\n- List item 1\n- List item 2"),
                ("html", "<html><body><h1>HTML Test</h1><p>This is HTML content.</p></body></html>")
            ]
            
            parsed_count = 0
            total_content_length = 0
            
            for file_ext, content in test_documents:
                with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{file_ext}', delete=False) as f:
                    f.write(content)
                    temp_file = f.name
                
                try:
                    result = self.parser.parse_document(temp_file)
                    if result.success:
                        parsed_count += 1
                        total_content_length += len(result.content.text_content)
                finally:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
            
            duration = time.time() - start_time
            self.report.add_test(test_name, parsed_count > 0, duration, {
                "parsed_documents": parsed_count,
                "total_documents": len(test_documents),
                "total_content_length": total_content_length
            })
            
        except Exception as e:
            duration = time.time() - start_time
            self.report.add_test(test_name, False, duration, {"error": str(e)})
    
    def test_document_chunking(self):
        """Test document chunking functionality."""
        start_time = time.time()
        test_name = "Document Chunking Test"
        
        try:
            # Create test content
            test_content = ParsedContent(
                text_content="""# Large Document Test

This is a comprehensive test document that should be chunked into multiple parts.

## Section 1: Introduction

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.

## Section 2: Details

Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

## Section 3: Conclusion

Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo.""",
                structured_content={},
                metadata={"format": "txt", "size": 800},
                tables=[],
                images=[],
                math_content=[],
                structure=["heading", "paragraph", "heading", "paragraph", "heading", "paragraph"],
                parser_used="text_parser",
                parsing_errors=[],
                parsing_warnings=[]
            )
            
            # Test chunking
            result = self.chunker.chunk_document(test_content)
            
            chunk_count = len(result.chunks) if result.chunks else 0
            avg_chunk_size = sum(len(chunk.content) for chunk in result.chunks) / chunk_count if chunk_count > 0 else 0
            
            duration = time.time() - start_time
            self.report.add_test(test_name, result.success, duration, {
                "chunk_count": chunk_count,
                "average_chunk_size": avg_chunk_size,
                "chunking_strategy": result.chunking_strategy
            })
            
        except Exception as e:
            duration = time.time() - start_time
            self.report.add_test(test_name, False, duration, {"error": str(e)})
    
    def test_metadata_extraction(self):
        """Test metadata extraction functionality."""
        start_time = time.time()
        test_name = "Metadata Extraction Test"
        
        try:
            test_texts = [
                "John Smith works at Microsoft. He lives in Seattle, Washington.",
                "The meeting is scheduled for January 15, 2024 at 3:00 PM PST.",
                "Contact us at support@example.com or visit our website at https://example.com",
                "Machine learning and artificial intelligence are transforming technology."
            ]
            
            total_entities = 0
            total_topics = 0
            successful_extractions = 0
            
            for text in test_texts:
                result = self.metadata_extractor.extract_metadata(text)
                if result.success:
                    successful_extractions += 1
                    total_entities += len(result.entities) if result.entities else 0
                    total_topics += len(result.topics) if result.topics else 0
            
            duration = time.time() - start_time
            self.report.add_test(test_name, successful_extractions > 0, duration, {
                "successful_extractions": successful_extractions,
                "total_tests": len(test_texts),
                "total_entities": total_entities,
                "total_topics": total_topics
            })
            
        except Exception as e:
            duration = time.time() - start_time
            self.report.add_test(test_name, False, duration, {"error": str(e)})
    
    def test_quality_assessment(self):
        """Test quality assessment functionality."""
        start_time = time.time()
        test_name = "Quality Assessment Test"
        
        try:
            # Test with sample content
            test_content = ParsedContent(
                text_content="This is a test document for quality assessment.",
                structured_content={},
                metadata={"format": "txt", "size": 47},
                tables=[],
                images=[],
                math_content=[],
                structure=["paragraph"],
                parser_used="text_parser",
                parsing_errors=[],
                parsing_warnings=[]
            )
            
            # Assess quality
            quality_report = self.quality_system.assess_document_quality(test_content)
            
            metrics_count = len(quality_report.metrics)
            overall_score = quality_report.overall_score
            recommendations_count = len(quality_report.recommendations)
            
            duration = time.time() - start_time
            self.report.add_test(test_name, overall_score >= 0.0, duration, {
                "overall_score": overall_score,
                "metrics_count": metrics_count,
                "recommendations_count": recommendations_count,
                "passed_quality_check": quality_report.passed
            })
            
        except Exception as e:
            duration = time.time() - start_time
            self.report.add_test(test_name, False, duration, {"error": str(e)})
    
    def test_performance_monitoring(self):
        """Test performance monitoring functionality."""
        start_time = time.time()
        test_name = "Performance Monitoring Test"
        
        try:
            monitor = self.quality_system.performance_monitor
            
            # Test operation tracking
            op_id = monitor.start_operation("test_operation", input_size=100)
            time.sleep(0.1)  # Simulate work
            monitor.end_operation(op_id, success=True, output_size=50)
            
            # Get performance summary
            summary = monitor.get_performance_summary()
            
            # Test metrics export
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                temp_file = f.name
            
            try:
                monitor.export_metrics(temp_file)
                export_successful = os.path.exists(temp_file)
            finally:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            
            duration = time.time() - start_time
            self.report.add_test(test_name, export_successful, duration, {
                "total_operations": summary.get("total_operations", 0),
                "success_rate": summary.get("success_rate", 0.0),
                "metrics_exported": export_successful
            })
            
        except Exception as e:
            duration = time.time() - start_time
            self.report.add_test(test_name, False, duration, {"error": str(e)})
    
    def test_end_to_end_pipeline(self):
        """Test the complete end-to-end pipeline."""
        start_time = time.time()
        test_name = "End-to-End Pipeline Test"
        
        try:
            # Create comprehensive test document
            test_content = """# RAG Document Processing Test

This is a comprehensive test document for the RAG document processing utility.

## Introduction

The RAG (Retrieval-Augmented Generation) approach combines information retrieval with language generation to create more informed and accurate responses.

## Key Components

1. **Document Parser**: Extracts text and structure from various document formats
2. **Chunker**: Splits documents into manageable, semantically coherent pieces
3. **Metadata Extractor**: Identifies entities, topics, and relationships
4. **Quality Assessor**: Evaluates processing quality and provides recommendations

## Technical Implementation

The system uses a modular architecture with:
- Cascading parser strategy for robust document handling
- Hybrid chunking with quality-based strategy selection
- LLM-powered metadata extraction with fallback mechanisms
- Comprehensive quality assessment and performance monitoring

## Example Data

Contact: support@example.com
Website: https://example.com
Date: January 15, 2024
Location: Seattle, Washington

## Conclusion

This system provides a complete pipeline for preparing documents for RAG applications."""
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(test_content)
                temp_file = f.name
            
            pipeline_success = True
            pipeline_details = {}
            
            try:
                # Step 1: Parse
                parse_result = self.parser.parse_document(temp_file)
                pipeline_details["parse_success"] = parse_result.success
                pipeline_details["content_length"] = len(parse_result.content.text_content) if parse_result.success else 0
                
                if not parse_result.success:
                    pipeline_success = False
                
                # Step 2: Chunk
                if parse_result.success:
                    chunk_result = self.chunker.chunk_document(parse_result.content)
                    pipeline_details["chunk_success"] = chunk_result.success
                    pipeline_details["chunk_count"] = len(chunk_result.chunks) if chunk_result.success else 0
                    
                    if not chunk_result.success:
                        pipeline_success = False
                
                # Step 3: Extract metadata
                if parse_result.success and chunk_result.success:
                    metadata_results = []
                    for chunk in chunk_result.chunks:
                        metadata_result = self.metadata_extractor.extract_metadata(chunk.content)
                        metadata_results.append(metadata_result)
                    
                    successful_extractions = sum(1 for r in metadata_results if r.success)
                    pipeline_details["metadata_success"] = successful_extractions > 0
                    pipeline_details["metadata_extractions"] = successful_extractions
                
                # Step 4: Assess quality
                if parse_result.success:
                    quality_report = self.quality_system.assess_document_quality(parse_result.content)
                    pipeline_details["quality_score"] = quality_report.overall_score
                    pipeline_details["quality_passed"] = quality_report.passed
                
                # Step 5: Performance summary
                performance_summary = self.quality_system.performance_monitor.get_performance_summary()
                pipeline_details["total_operations"] = performance_summary.get("total_operations", 0)
                
            finally:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            
            duration = time.time() - start_time
            self.report.add_test(test_name, pipeline_success, duration, pipeline_details)
            
        except Exception as e:
            duration = time.time() - start_time
            self.report.add_test(test_name, False, duration, {"error": str(e)})
    
    def run_all_tests(self):
        """Run all tests in the comprehensive test suite."""
        print("🚀 Starting Comprehensive RAG Document Processing Test Suite")
        print("=" * 60)
        
        # Initialize components
        print("🔧 Initializing system components...")
        if not self.setup_components():
            print("❌ Failed to initialize components. Aborting tests.")
            return self.report.generate_report()
        
        # Run all tests
        tests = [
            ("Configuration System", self.test_configuration_system),
            ("Document Parsing", self.test_document_parsing),
            ("Document Chunking", self.test_document_chunking),
            ("Metadata Extraction", self.test_metadata_extraction),
            ("Quality Assessment", self.test_quality_assessment),
            ("Performance Monitoring", self.test_performance_monitoring),
            ("End-to-End Pipeline", self.test_end_to_end_pipeline)
        ]
        
        for test_name, test_func in tests:
            print(f"\n📋 Running {test_name}...")
            test_func()
            
            # Get the latest test result
            latest_test = self.report.tests[-1] if self.report.tests else None
            if latest_test:
                status = "✅ PASSED" if latest_test["passed"] else "❌ FAILED"
                duration = latest_test["duration"]
                print(f"   {status} ({duration:.3f}s)")
                
                # Print key details
                details = latest_test.get("details", {})
                for key, value in details.items():
                    if key != "error":
                        print(f"   📊 {key}: {value}")
                
                if not latest_test["passed"] and "error" in details:
                    print(f"   ⚠️  Error: {details['error']}")
        
        return self.report.generate_report()


def main():
    """Main test runner function."""
    # Run comprehensive test suite
    test_suite = ComprehensiveTestSuite()
    report = test_suite.run_all_tests()
    
    # Print final summary
    print("\n" + "=" * 60)
    print("📊 FINAL TEST SUMMARY")
    print("=" * 60)
    
    summary = report["test_summary"]
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed_tests']} ✅")
    print(f"Failed: {summary['failed_tests']} ❌")
    print(f"Success Rate: {summary['success_rate']:.1%}")
    print(f"Total Duration: {summary['total_duration']:.2f}s")
    print(f"Average Test Duration: {summary['average_test_duration']:.3f}s")
    
    if report["warnings"]:
        print(f"\nWarnings: {len(report['warnings'])}")
        for warning in report["warnings"]:
            print(f"  ⚠️  {warning['message']}")
    
    # Save detailed report
    report_file = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    # Overall result
    if summary['failed_tests'] == 0:
        print("\n🎉 ALL TESTS PASSED! The RAG Document Processing Utility is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {summary['failed_tests']} TESTS FAILED. Please review the issues above.")
        return 1


if __name__ == "__main__":
    exit(main())
