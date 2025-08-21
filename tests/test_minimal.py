"""
Minimal test suite for CI/CD pipeline.
Focuses on basic functionality that should always work.
"""


def test_config_creation():
    """Test that configuration can be created."""
    from src.config import Config

    config = Config()
    assert config.app_name == "RAG Document Processing Utility"
    assert config.version == "0.1.0"
    assert config.parser.supported_formats == [".pdf", ".docx", ".txt", ".html"]
    assert config.chunking.strategy == "hybrid"
    assert config.metadata.extraction_level == "enhanced"
    assert config.security.enable_file_validation is True


def test_basic_dataclasses():
    """Test that basic dataclasses can be instantiated."""
    from src.chunkers import DocumentChunk
    from src.metadata_extractors import Entity
    from src.parsers import ParsedContent
    from src.quality_assessment import QualityMetric
    from src.security import SecurityManager

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

    # Test SecurityManager
    # Note: SecurityManager requires config, so we'll just test the import
    assert SecurityManager is not None


def test_factory_functions():
    """Test that factory functions can be called."""
    from src.chunkers import get_document_chunker
    from src.config import get_config
    from src.metadata_extractors import get_metadata_extractor
    from src.parsers import get_document_parser
    from src.quality_assessment import get_quality_assessment_system
    from src.vector_store import get_vector_store

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
    from src.config import Config
    from src.security import SecurityManager

    config = Config()
    security_manager = SecurityManager(config)

    # Test basic initialization
    assert security_manager.validator is not None
    assert security_manager.sanitizer is not None
    assert security_manager.analyzer is not None


def test_quality_assessment_basic():
    """Test basic quality assessment functionality."""
    from src.config import Config
    from src.quality_assessment import QualityAssessmentSystem

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
