"""
Minimal test suite for CI/CD pipeline.
Focuses on basic functionality that should always work.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))


def test_config_creation():
    """Test that configuration can be created."""
    from config import Config

    config = Config()
    assert config.app_name == "RAG Document Processing Utility"
    assert config.version == "0.1.0"
    assert config.parser.supported_formats == [".pdf", ".docx", ".txt", ".html"]
    assert config.chunking.strategy == "hybrid"
    assert config.metadata.extraction_level == "advanced"
    assert config.security.enable_file_validation is True


def test_basic_dataclasses():
    """Test that basic dataclasses can be instantiated."""
    from chunkers import DocumentChunk
    from metadata_extractors import Entity
    from parsers import ParsedContent
    from quality_assessment import QualityMetric
    from security import SecurityCheck

    # Test ParsedContent
    content = ParsedContent(
        text_content="Test content",
        structured_content={},
        metadata={"format": "txt"},
        tables=[],
        images=[],
        math_content=[],
        structure=["heading1", "paragraph1"],
        parser_used="text",
        parsing_errors=[],
        parsing_warnings=[],
    )
    assert content.text_content == "Test content"

    # Test DocumentChunk
    chunk = DocumentChunk(
        chunk_id="chunk_123",
        content="Test chunk",
        chunk_type="text",
        chunk_index=0,
        quality_score=0.8,
        metadata={"source": "test.txt"},
    )
    assert chunk.chunk_id == "chunk_123"

    # Test Entity
    entity = Entity(
        text="Test Entity",
        entity_type="PERSON",
        confidence=0.9,
        metadata={"source": "test"},
    )
    assert entity.text == "Test Entity"

    # Test QualityMetric
    metric = QualityMetric(name="test_metric", score=0.8, weight=1.0, threshold=0.7)
    assert metric.name == "test_metric"

    # Test SecurityCheck
    check = SecurityCheck(
        check_name="test_check",
        passed=True,
        details="Test security check",
        threat_level="low",
    )
    assert check.check_name == "test_check"


def test_factory_functions():
    """Test that factory functions can be called."""
    from chunkers import get_document_chunker
    from config import get_config
    from metadata_extractors import get_metadata_extractor
    from parsers import get_document_parser
    from quality_assessment import get_quality_assessment_system
    from vector_store import get_vector_store

    # Test get_config
    config = get_config()
    assert isinstance(config, type(get_config()))

    # Test get_document_parser
    parser = get_document_parser()
    assert parser is not None

    # Test get_document_chunker
    chunker = get_document_chunker()
    assert chunker is not None

    # Test get_metadata_extractor
    extractor = get_metadata_extractor()
    assert extractor is not None

    # Test get_quality_assessment_system
    quality_system = get_quality_assessment_system()
    assert quality_system is not None

    # Test get_vector_store
    vector_store = get_vector_store("file", config)
    assert vector_store is not None


def test_security_basic():
    """Test basic security functionality."""
    from config import Config
    from security import SecurityManager

    config = Config()
    security_manager = SecurityManager(config)

    # Test basic initialization
    assert security_manager.validator is not None
    assert security_manager.sanitizer is not None
    assert security_manager.analyzer is not None


def test_quality_assessment_basic():
    """Test basic quality assessment functionality."""
    from config import Config
    from quality_assessment import QualityAssessmentSystem

    config = Config()
    quality_system = QualityAssessmentSystem(config)

    # Test basic initialization
    assert quality_system.content_assessor is not None
    assert quality_system.structure_assessor is not None
    assert quality_system.metadata_assessor is not None
    assert quality_system.performance_monitor is not None


if __name__ == "__main__":
    # Run all tests
    tests = [
        test_config_creation,
        test_basic_dataclasses,
        test_factory_functions,
        test_security_basic,
        test_quality_assessment_basic,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            test()
            passed += 1
            print(f"✅ {test.__name__} passed")
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")

    print(f"\n📊 Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All minimal tests passed!")
        exit(0)
    else:
        print("❌ Some tests failed")
        exit(1)
