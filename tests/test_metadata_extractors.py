"""
Tests for metadata extractors
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.metadata_extractors import (
    Entity,
    Topic,
    Relationship,
    Summary,
    ExtractionResult,
    MetadataExtractor,
    BasicMetadataExtractor,
    LLMMetadataExtractor,
    MetadataExtractorFactory,
    get_metadata_extractor,
)
from src.config import Config
from src.parsers import ParsedContent
from src.chunkers import DocumentChunk


class TestEntity:
    """Test Entity dataclass."""

    def test_entity_creation(self):
        """Test creating Entity instances."""
        entity = Entity(
            text="John Doe",
            entity_type="person",
            confidence=0.9,
            start_position=10,
            end_position=18,
        )

        assert entity.text == "John Doe"
        assert entity.entity_type == "person"
        assert entity.confidence == 0.9
        assert entity.start_position == 10
        assert entity.end_position == 18
        assert entity.metadata == {}

    def test_entity_defaults(self):
        """Test Entity default values."""
        entity = Entity(text="Test Entity", entity_type="organization", confidence=0.8)

        assert entity.start_position is None
        assert entity.end_position is None
        assert entity.metadata == {}


class TestTopic:
    """Test Topic dataclass."""

    def test_topic_creation(self):
        """Test creating Topic instances."""
        topic = Topic(
            name="Artificial Intelligence",
            confidence=0.85,
            keywords=["AI", "machine learning", "neural networks"],
            description="Study of intelligent systems",
            category="technology",
        )

        assert topic.name == "Artificial Intelligence"
        assert topic.confidence == 0.85
        assert topic.keywords == ["AI", "machine learning", "neural networks"]
        assert topic.description == "Study of intelligent systems"
        assert topic.category == "technology"


class TestRelationship:
    """Test Relationship dataclass."""

    def test_relationship_creation(self):
        """Test creating Relationship instances."""
        relationship = Relationship(
            source_entity="John Doe",
            target_entity="Company Inc",
            relationship_type="employee_of",
            confidence=0.9,
            context="John Doe works at Company Inc",
        )

        assert relationship.source_entity == "John Doe"
        assert relationship.target_entity == "Company Inc"
        assert relationship.relationship_type == "employee_of"
        assert relationship.confidence == 0.9
        assert relationship.context == "John Doe works at Company Inc"
        assert relationship.metadata == {}

    def test_relationship_defaults(self):
        """Test Relationship default values."""
        relationship = Relationship(
            source_entity="Entity1",
            target_entity="Entity2",
            relationship_type="related_to",
            confidence=0.7,
        )

        assert relationship.context is None
        assert relationship.metadata == {}


class TestSummary:
    """Test Summary dataclass."""

    def test_summary_creation(self):
        """Test creating Summary instances."""
        summary = Summary(
            summary_type="extractive",
            content="This is a summary of the document.",
            length=35,
            confidence=0.8,
            key_points=["Point 1", "Point 2"],
        )

        assert summary.summary_type == "extractive"
        assert summary.content == "This is a summary of the document."
        assert summary.length == 35
        assert summary.confidence == 0.8
        assert summary.key_points == ["Point 1", "Point 2"]
        assert summary.metadata == {}

    def test_summary_defaults(self):
        """Test Summary default values."""
        summary = Summary(
            summary_type="generated",
            content="Test summary",
            length=12,
            confidence=0.9,
            key_points=[],
        )

        assert summary.metadata == {}


class TestExtractionResult:
    """Test ExtractionResult dataclass."""

    def test_extraction_result_creation(self):
        """Test creating ExtractionResult instances."""
        result = ExtractionResult(
            success=True,
            entities=[],
            topics=[],
            relationships=[],
            summaries=[],
            extraction_strategy="test",
            processing_time=1.5,
            metadata={"test": "data"},
            errors=[],
            warnings=[],
        )

        assert result.success is True
        assert result.extraction_strategy == "test"
        assert result.processing_time == 1.5
        assert result.metadata["test"] == "data"


class TestMetadataExtractor:
    """Test base metadata extractor functionality."""

    def test_abstract_base_class(self):
        """Test that MetadataExtractor is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            MetadataExtractor(Mock(spec=Config))

    def test_validate_content_string(self):
        """Test content validation with string input."""

        class TestExtractor(MetadataExtractor):
            def extract_metadata(self, content):
                return ExtractionResult(
                    success=True,
                    entities=[],
                    topics=[],
                    relationships=[],
                    summaries=[],
                    extraction_strategy="test",
                    processing_time=0.0,
                    metadata={},
                    errors=[],
                    warnings=[],
                )

        mock_config = Mock(spec=Config)
        extractor = TestExtractor(mock_config)

        content = extractor._validate_content("Test content")
        assert content == "Test content"

    def test_validate_content_parsed_content(self):
        """Test content validation with ParsedContent input."""

        class TestExtractor(MetadataExtractor):
            def extract_metadata(self, content):
                return ExtractionResult(
                    success=True,
                    entities=[],
                    topics=[],
                    relationships=[],
                    summaries=[],
                    extraction_strategy="test",
                    processing_time=0.0,
                    metadata={},
                    errors=[],
                    warnings=[],
                )

        mock_config = Mock(spec=Config)
        extractor = TestExtractor(mock_config)

        mock_parsed_content = Mock(spec=ParsedContent)
        mock_parsed_content.text_content = "Test content"

        content = extractor._validate_content(mock_parsed_content)
        assert content == "Test content"

    def test_validate_content_document_chunk(self):
        """Test content validation with DocumentChunk input."""

        class TestExtractor(MetadataExtractor):
            def extract_metadata(self, content):
                return ExtractionResult(
                    success=True,
                    entities=[],
                    topics=[],
                    relationships=[],
                    summaries=[],
                    extraction_strategy="test",
                    processing_time=0.0,
                    metadata={},
                    errors=[],
                    warnings=[],
                )

        mock_config = Mock(spec=Config)
        extractor = TestExtractor(mock_config)

        mock_chunk = Mock(spec=DocumentChunk)
        mock_chunk.content = "Test content"

        content = extractor._validate_content(mock_chunk)
        assert content == "Test content"

    def test_validate_content_invalid_type(self):
        """Test content validation with invalid input type."""

        class TestExtractor(MetadataExtractor):
            def extract_metadata(self, content):
                return ExtractionResult(
                    success=True,
                    entities=[],
                    topics=[],
                    relationships=[],
                    summaries=[],
                    extraction_strategy="test",
                    processing_time=0.0,
                    metadata={},
                    errors=[],
                    warnings=[],
                )

        mock_config = Mock(spec=Config)
        extractor = TestExtractor(mock_config)

        with pytest.raises(ValueError, match="Unsupported content type"):
            extractor._validate_content(123)

    def test_assess_extraction_quality(self):
        """Test extraction quality assessment."""

        class TestExtractor(MetadataExtractor):
            def extract_metadata(self, content):
                return ExtractionResult(
                    success=True,
                    entities=[],
                    topics=[],
                    relationships=[],
                    summaries=[],
                    extraction_strategy="test",
                    processing_time=0.0,
                    metadata={},
                    errors=[],
                    warnings=[],
                )

        mock_config = Mock(spec=Config)
        extractor = TestExtractor(mock_config)

        # Test empty extraction
        quality = extractor._assess_extraction_quality([], [], [])
        assert quality == 0.0

        # Test with entities only
        entities = [Entity("John", "person", 0.9)]
        quality = extractor._assess_extraction_quality(entities, [], [])
        assert 0.0 <= quality <= 1.0

        # Test with all types
        topics = [Topic("AI", 0.8, ["AI", "ML"])]
        relationships = [Relationship("John", "Company", "works_at", 0.7)]
        quality = extractor._assess_extraction_quality(entities, topics, relationships)
        assert 0.0 <= quality <= 1.0


class TestBasicMetadataExtractor:
    """Test basic metadata extractor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = Mock(spec=Config)
        self.mock_metadata_config = Mock()
        self.mock_metadata_config.extraction_level = "basic"
        self.mock_metadata_config.entity_recognition = True
        self.mock_metadata_config.topic_extraction = True
        self.mock_metadata_config.relationship_extraction = True
        self.mock_metadata_config.summarization = True
        self.mock_config.get_metadata_config.return_value = self.mock_metadata_config

        self.extractor = BasicMetadataExtractor(self.mock_config)

    def test_extract_metadata_empty_content(self):
        """Test extraction with empty content."""
        result = self.extractor.extract_metadata("")

        assert result.success is False
        assert "Empty content" in result.errors
        assert result.entities == []
        assert result.topics == []
        assert result.relationships == []
        assert result.summaries == []

    def test_extract_basic_entities(self):
        """Test basic entity extraction."""
        text = "Contact John Doe at john.doe@email.com or visit https://example.com"

        entities = self.extractor._extract_basic_entities(text)

        # Should extract email and URL
        email_entities = [e for e in entities if e.entity_type == "email"]
        url_entities = [e for e in entities if e.entity_type == "url"]

        assert len(email_entities) == 1
        assert email_entities[0].text == "john.doe@email.com"
        assert email_entities[0].confidence == 0.9

        assert len(url_entities) == 1
        assert url_entities[0].text == "https://example.com"
        assert url_entities[0].confidence == 0.9

    def test_extract_basic_topics(self):
        """Test basic topic extraction."""
        text = "This document discusses software development and business strategy. We analyze market trends and technology adoption."

        topics = self.extractor._extract_basic_topics(text)

        # Should identify technology and business topics
        tech_topic = next((t for t in topics if t.name == "technology"), None)
        business_topic = next((t for t in topics if t.name == "business"), None)

        assert tech_topic is not None
        assert "software" in tech_topic.keywords
        assert "technology" in tech_topic.keywords

        assert business_topic is not None
        assert "business" in business_topic.keywords
        assert "market" in business_topic.keywords

    def test_extract_basic_relationships(self):
        """Test basic relationship extraction."""
        text = "John works at Company Inc. Jane also works at Company Inc."

        entities = [
            Entity("john.doe@email.com", "email", 0.9),
            Entity("https://example.com", "url", 0.9),
        ]

        relationships = self.extractor._extract_basic_relationships(text, entities)

        # Should find co-occurrence relationships
        assert len(relationships) > 0
        assert all(r.relationship_type == "co_occurrence" for r in relationships)

    def test_generate_basic_summary(self):
        """Test basic summary generation."""
        text = "First sentence. Second sentence. Third sentence."

        summaries = self.extractor._generate_basic_summary(text)

        assert len(summaries) == 1
        assert summaries[0].summary_type == "extractive"
        assert "First sentence" in summaries[0].content
        assert summaries[0].confidence == 0.7


class TestLLMMetadataExtractor:
    """Test LLM-powered metadata extractor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = Mock(spec=Config)
        self.mock_config.openai_api_key = "test_key_123"
        self.mock_metadata_config = Mock()
        self.mock_metadata_config.extraction_level = "llm_powered"
        self.mock_config.get_metadata_config.return_value = self.mock_metadata_config

        self.extractor = LLMMetadataExtractor(self.mock_config)

    def test_extract_metadata_without_api_key(self):
        """Test extraction without OpenAI API key."""
        self.extractor.openai_api_key = None

        result = self.extractor.extract_metadata("Test content")

        # Should fall back to basic extraction
        assert result.extraction_strategy == "basic"

    @patch("metadata_extractors.openai")
    def test_extract_entities_with_llm(self, mock_openai):
        """Test LLM entity extraction."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps(
            [
                {
                    "text": "John Doe",
                    "entity_type": "person",
                    "confidence": 0.9,
                    "start_position": 0,
                    "end_position": 8,
                }
            ]
        )
        mock_openai.ChatCompletion.create.return_value = mock_response

        entities = self.extractor._extract_entities_with_llm("John Doe is a developer")

        assert len(entities) == 1
        assert entities[0].text == "John Doe"
        assert entities[0].entity_type == "person"
        assert entities[0].confidence == 0.9

    @patch("metadata_extractors.openai")
    def test_extract_topics_with_llm(self, mock_openai):
        """Test LLM topic extraction."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps(
            [
                {
                    "name": "Technology",
                    "confidence": 0.85,
                    "keywords": ["AI", "ML"],
                    "description": "Technology topics",
                    "category": "tech",
                }
            ]
        )
        mock_openai.ChatCompletion.create.return_value = mock_response

        topics = self.extractor._extract_topics_with_llm(
            "AI and machine learning are important"
        )

        assert len(topics) == 1
        assert topics[0].name == "Technology"
        assert topics[0].confidence == 0.85
        assert "AI" in topics[0].keywords

    @patch("metadata_extractors.openai")
    def test_extract_relationships_with_llm(self, mock_openai):
        """Test LLM relationship extraction."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps(
            [
                {
                    "source_entity": "John Doe",
                    "target_entity": "Company Inc",
                    "relationship_type": "employee_of",
                    "confidence": 0.9,
                    "context": "John Doe works at Company Inc",
                }
            ]
        )
        mock_openai.ChatCompletion.create.return_value = mock_response

        entities = [Entity("John Doe", "person", 0.9)]
        relationships = self.extractor._extract_relationships_with_llm(
            "John Doe works at Company Inc", entities
        )

        assert len(relationships) == 1
        assert relationships[0].source_entity == "John Doe"
        assert relationships[0].target_entity == "Company Inc"
        assert relationships[0].relationship_type == "employee_of"

    @patch("metadata_extractors.openai")
    def test_generate_summary_with_llm(self, mock_openai):
        """Test LLM summary generation."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "summary": "This document discusses AI and machine learning.",
                "key_points": ["AI", "ML", "technology"],
                "confidence": 0.9,
            }
        )
        mock_openai.ChatCompletion.create.return_value = mock_response

        summaries = self.extractor._generate_summary_with_llm(
            "AI and machine learning are important technologies"
        )

        assert len(summaries) == 1
        assert summaries[0].summary_type == "llm_generated"
        assert "AI and machine learning" in summaries[0].content
        assert summaries[0].confidence == 0.9

    def test_llm_extraction_error_handling(self):
        """Test LLM extraction error handling."""
        # Test with invalid JSON response
        with patch("metadata_extractors.openai.ChatCompletion.create") as mock_create:
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Invalid JSON"
            mock_create.return_value = mock_response

            entities = self.extractor._extract_entities_with_llm("Test content")
            assert entities == []

    def test_llm_extraction_import_error(self):
        """Test LLM extraction when OpenAI library is not available."""
        with patch.dict("sys.modules", {"openai": None}):
            entities = self.extractor._extract_entities_with_llm("Test content")
            assert entities == []


class TestMetadataExtractorFactory:
    """Test metadata extractor factory."""

    def test_create_basic_extractor(self):
        """Test creating basic extractor."""
        mock_config = Mock(spec=Config)

        extractor = MetadataExtractorFactory.create_extractor("basic", mock_config)

        assert isinstance(extractor, BasicMetadataExtractor)

    def test_create_enhanced_extractor(self):
        """Test creating enhanced extractor."""
        mock_config = Mock(spec=Config)

        extractor = MetadataExtractorFactory.create_extractor("enhanced", mock_config)

        assert isinstance(extractor, BasicMetadataExtractor)

    def test_create_llm_extractor(self):
        """Test creating LLM extractor."""
        mock_config = Mock(spec=Config)

        extractor = MetadataExtractorFactory.create_extractor(
            "llm_powered", mock_config
        )

        assert isinstance(extractor, LLMMetadataExtractor)

    def test_create_unknown_extractor(self):
        """Test creating extractor with unknown level."""
        mock_config = Mock(spec=Config)

        extractor = MetadataExtractorFactory.create_extractor("unknown", mock_config)

        # Should fall back to basic
        assert isinstance(extractor, BasicMetadataExtractor)


# Test convenience function
def test_get_metadata_extractor():
    """Test getting metadata extractor instance."""
    extractor = get_metadata_extractor("basic")
    assert isinstance(extractor, BasicMetadataExtractor)
    assert extractor.config is not None


if __name__ == "__main__":
    pytest.main([__file__])
