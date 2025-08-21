#!/usr/bin/env python3
"""
Simple test runner to check basic functionality.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_basic_imports():
    """Test that basic imports work."""
    try:
        from config import Config
        print("✅ Config import works")
        
        from parsers import ParsedContent
        print("✅ ParsedContent import works")
        
        from chunkers import DocumentChunk
        print("✅ DocumentChunk import works")
        
        from metadata_extractors import Entity
        print("✅ Entity import works")
        
        from quality_assessment import QualityMetric
        print("✅ QualityMetric import works")
        
        from security import SecurityCheck
        print("✅ SecurityCheck import works")
        
        from vector_store import VectorStore
        print("✅ VectorStore import works")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_basic_config():
    """Test basic configuration creation."""
    try:
        from config import Config
        
        config = Config()
        print("✅ Config creation works")
        
        # Test basic access
        assert config.app_name == "RAG Document Processing Utility"
        assert config.version == "0.1.0"
        assert config.parser.supported_formats == [".pdf", ".docx", ".txt", ".html"]
        assert config.chunking.strategy == "hybrid"
        assert config.metadata.extraction_level == "advanced"
        
        print("✅ Config validation works")
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_basic_dataclasses():
    """Test basic dataclass creation."""
    try:
        from parsers import ParsedContent
        from chunkers import DocumentChunk
        from metadata_extractors import Entity
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
            parsing_warnings=[]
        )
        assert content.text_content == "Test content"
        print("✅ ParsedContent creation works")
        
        # Test DocumentChunk
        chunk = DocumentChunk(
            chunk_id="chunk_123",
            content="Test chunk",
            chunk_type="text",
            chunk_index=0,
            quality_score=0.8,
            metadata={"source": "test.txt"}
        )
        assert chunk.chunk_id == "chunk_123"
        print("✅ DocumentChunk creation works")
        
        # Test Entity
        entity = Entity(
            text="Test Entity",
            entity_type="PERSON",
            confidence=0.9,
            metadata={"source": "test"}
        )
        assert entity.text == "Test Entity"
        print("✅ Entity creation works")
        
        # Test QualityMetric
        metric = QualityMetric(
            name="test_metric",
            score=0.8,
            weight=1.0,
            threshold=0.7
        )
        assert metric.name == "test_metric"
        print("✅ QualityMetric creation works")
        
        # Test SecurityCheck
        check = SecurityCheck(
            check_name="test_check",
            passed=True,
            details="Test security check",
            threat_level="low"
        )
        assert check.check_name == "test_check"
        print("✅ SecurityCheck creation works")
        
        return True
    except Exception as e:
        print(f"❌ Dataclass test failed: {e}")
        return False

def main():
    """Run all basic tests."""
    print("🧪 Running basic functionality tests...")
    print()
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Basic Config", test_basic_config),
        ("Basic Dataclasses", test_basic_dataclasses),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Testing: {test_name}")
        if test_func():
            passed += 1
        print()
    
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
