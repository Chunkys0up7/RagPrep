"""
Metadata Extractors

This module implements LLM-powered metadata extraction with entity recognition,
topic extraction, relationship mapping, and content summarization.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import json
import logging
import re
import time
from pathlib import Path

from chunkers import DocumentChunk
from config import Config, get_config
from parsers import ParsedContent

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Represents an extracted entity."""

    text: str
    entity_type: str
    confidence: float
    start_position: Optional[int] = None
    end_position: Optional[int] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Topic:
    """Represents an extracted topic."""

    name: str
    confidence: float
    keywords: List[str]
    description: Optional[str] = None
    category: Optional[str] = None


@dataclass
class Relationship:
    """Represents a relationship between entities."""

    source_entity: str
    target_entity: str
    relationship_type: str
    confidence: float
    context: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Summary:
    """Represents a content summary."""

    summary_type: str
    content: str
    length: int
    confidence: float
    key_points: List[str]
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ExtractionResult:
    """Result of metadata extraction operation."""

    success: bool
    entities: List[Entity]
    topics: List[Topic]
    relationships: List[Relationship]
    summaries: List[Summary]
    extraction_strategy: str
    processing_time: float
    metadata: Dict[str, Any]
    errors: List[str]
    warnings: List[str]


class MetadataExtractor(ABC):
    """Abstract base class for metadata extractors."""

    def __init__(self, config: Config):
        """Initialize extractor with configuration."""
        self.config = config
        self.metadata_config = config.get_metadata_config()
        self.extractor_name: str = self.__class__.__name__

    def extract(
        self,
        content: Union[str, ParsedContent, DocumentChunk],
        chunks: List[DocumentChunk],
    ) -> ExtractionResult:
        """Extract metadata from content (alias for extract_metadata)."""
        return self.extract_metadata(content)

    @abstractmethod
    def extract_metadata(
        self, content: Union[str, ParsedContent, DocumentChunk]
    ) -> ExtractionResult:
        """Extract metadata from content."""
        pass

    def _validate_content(
        self, content: Union[str, ParsedContent, DocumentChunk]
    ) -> str:
        """Validate and extract text content."""
        if isinstance(content, str):
            return content
        elif isinstance(content, ParsedContent):
            return content.text_content
        elif isinstance(content, DocumentChunk):
            return content.content
        else:
            raise ValueError(f"Unsupported content type: {type(content)}")

    def _assess_extraction_quality(
        self,
        entities: List[Entity],
        topics: List[Topic],
        relationships: List[Relationship],
    ) -> float:
        """Assess the quality of metadata extraction."""
        if not entities and not topics and not relationships:
            return 0.0

        # Entity quality score
        entity_score = 0.0
        if entities:
            avg_confidence = sum(e.confidence for e in entities) / len(entities)
            entity_score = avg_confidence * 0.4

        # Topic quality score
        topic_score = 0.0
        if topics:
            avg_confidence = sum(t.confidence for t in topics) / len(topics)
            topic_score = avg_confidence * 0.3

        # Relationship quality score
        relationship_score = 0.0
        if relationships:
            avg_confidence = sum(r.confidence for r in relationships) / len(
                relationships
            )
            relationship_score = avg_confidence * 0.3

        return round(entity_score + topic_score + relationship_score, 3)


class BasicMetadataExtractor(MetadataExtractor):
    """Basic metadata extractor using rule-based approaches."""

    def __init__(self, config: Config):
        """Initialize basic extractor."""
        super().__init__(config)
        self.extractor_name = "BasicMetadataExtractor"

    def extract_metadata(
        self, content: Union[str, ParsedContent, DocumentChunk]
    ) -> ExtractionResult:
        """Extract basic metadata using rule-based approaches."""
        start_time = time.time()
        errors = []
        warnings = []

        try:
            text_content = self._validate_content(content)

            if not text_content.strip():
                return ExtractionResult(
                    success=False,
                    entities=[],
                    topics=[],
                    relationships=[],
                    summaries=[],
                    extraction_strategy="basic",
                    processing_time=time.time() - start_time,
                    metadata={},
                    errors=["Empty content"],
                    warnings=[],
                )

            # Extract entities using basic patterns
            entities = self._extract_basic_entities(text_content)

            # Extract topics using keyword analysis
            topics = self._extract_basic_topics(text_content)

            # Extract basic relationships
            relationships = self._extract_basic_relationships(text_content, entities)

            # Generate basic summary
            summaries = self._generate_basic_summary(text_content)

            # Assess quality
            quality_score = self._assess_extraction_quality(
                entities, topics, relationships
            )

            return ExtractionResult(
                success=True,
                entities=entities,
                topics=topics,
                relationships=relationships,
                summaries=summaries,
                extraction_strategy="basic",
                processing_time=time.time() - start_time,
                metadata={
                    "quality_score": quality_score,
                    "content_length": len(text_content),
                    "word_count": len(text_content.split()),
                },
                errors=errors,
                warnings=warnings,
            )

        except Exception as e:
            error_msg = f"Error in basic metadata extraction: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)

            return ExtractionResult(
                success=False,
                entities=[],
                topics=[],
                relationships=[],
                summaries=[],
                extraction_strategy="basic",
                processing_time=time.time() - start_time,
                metadata={},
                errors=errors,
                warnings=warnings,
            )

    def _extract_basic_entities(self, text: str) -> List[Entity]:
        """Extract basic entities using pattern matching."""
        entities = []

        # Extract email addresses
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        for match in re.finditer(email_pattern, text):
            entities.append(
                Entity(
                    text=match.group(),
                    entity_type="email",
                    confidence=0.9,
                    start_position=match.start(),
                    end_position=match.end(),
                )
            )

        # Extract URLs
        url_pattern = r"https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?"
        for match in re.finditer(url_pattern, text):
            entities.append(
                Entity(
                    text=match.group(),
                    entity_type="url",
                    confidence=0.9,
                    start_position=match.start(),
                    end_position=match.end(),
                )
            )

        # Extract dates (basic patterns)
        date_patterns = [
            r"\b\d{1,2}/\d{1,2}/\d{4}\b",  # MM/DD/YYYY
            r"\b\d{4}-\d{1,2}-\d{1,2}\b",  # YYYY-MM-DD
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b",  # Month DD, YYYY
        ]

        for pattern in date_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(
                    Entity(
                        text=match.group(),
                        entity_type="date",
                        confidence=0.8,
                        start_position=match.start(),
                        end_position=match.end(),
                    )
                )

        # Extract numbers (potential quantities, measurements)
        number_pattern = r"\b\d+(?:\.\d+)?(?:[KMB]|million|billion)?\b"
        for match in re.finditer(number_pattern, text, re.IGNORECASE):
            entities.append(
                Entity(
                    text=match.group(),
                    entity_type="number",
                    confidence=0.7,
                    start_position=match.start(),
                    end_position=match.end(),
                )
            )

        return entities

    def _extract_basic_topics(self, text: str) -> List[Topic]:
        """Extract basic topics using keyword frequency analysis."""
        topics = []

        # Common topic keywords
        topic_keywords = {
            "technology": [
                "software",
                "hardware",
                "computer",
                "system",
                "technology",
                "digital",
                "data",
            ],
            "business": [
                "business",
                "company",
                "market",
                "industry",
                "management",
                "strategy",
                "finance",
            ],
            "science": [
                "research",
                "study",
                "analysis",
                "experiment",
                "scientific",
                "method",
                "theory",
            ],
            "health": [
                "health",
                "medical",
                "treatment",
                "patient",
                "disease",
                "medicine",
                "clinical",
            ],
            "education": [
                "education",
                "learning",
                "teaching",
                "student",
                "school",
                "course",
                "training",
            ],
        }

        text_lower = text.lower()
        words = re.findall(r"\b\w+\b", text_lower)
        word_freq = {}

        for word in words:
            if len(word) > 3:  # Skip short words
                word_freq[word] = word_freq.get(word, 0) + 1

        # Find topics based on keyword frequency
        for topic_name, keywords in topic_keywords.items():
            topic_score = 0
            found_keywords = []

            for keyword in keywords:
                if keyword in word_freq:
                    topic_score += word_freq[keyword]
                    found_keywords.append(keyword)

            if topic_score > 0:
                confidence = min(0.9, topic_score / 10.0)  # Normalize confidence
                topics.append(
                    Topic(
                        name=topic_name,
                        confidence=confidence,
                        keywords=found_keywords,
                        description=f"Topic identified based on keywords: {', '.join(found_keywords)}",
                    )
                )

        return topics

    def _extract_basic_relationships(
        self, text: str, entities: List[Entity]
    ) -> List[Relationship]:
        """Extract basic relationships between entities."""
        relationships = []

        # Simple co-occurrence relationships
        entity_texts = [e.text for e in entities if e.entity_type in ["email", "url"]]

        for i, entity1 in enumerate(entity_texts):
            for entity2 in entity_texts[i + 1 :]:
                # Check if entities appear in the same sentence
                sentences = re.split(r"[.!?]+", text)
                for sentence in sentences:
                    if entity1 in sentence and entity2 in sentence:
                        relationships.append(
                            Relationship(
                                source_entity=entity1,
                                target_entity=entity2,
                                relationship_type="co_occurrence",
                                confidence=0.6,
                                context=sentence.strip(),
                            )
                        )
                        break

        return relationships

    def _generate_basic_summary(self, text: str) -> List[Summary]:
        """Generate basic summary using extractive approach."""
        summaries = []

        # Simple extractive summary (first few sentences)
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if sentences:
            # Take first 2-3 sentences as summary
            summary_sentences = sentences[: min(3, len(sentences))]
            summary_text = ". ".join(summary_sentences) + "."

            summaries.append(
                Summary(
                    summary_type="extractive",
                    content=summary_text,
                    length=len(summary_text),
                    confidence=0.7,
                    key_points=summary_sentences,
                )
            )

        return summaries


class LLMMetadataExtractor(MetadataExtractor):
    """LLM-powered metadata extractor using OpenAI API."""

    def __init__(self, config: Config):
        """Initialize the LLM metadata extractor."""
        super().__init__(config)
        self.extractor_name = "LLMMetadataExtractor"
        self.openai_api_key = config.openai_api_key
        self.llm_model = config.metadata.llm_model
        self.llm_temperature = config.metadata.llm_temperature

        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required for LLM metadata extraction")

        # Initialize OpenAI client
        import openai

        openai.api_key = self.openai_api_key

    def extract_metadata(
        self, content: Union[str, ParsedContent, DocumentChunk]
    ) -> ExtractionResult:
        """Extract metadata using LLM."""
        start_time = time.time()
        errors = []
        warnings = []

        try:
            text_content = self._validate_content(content)

            if not text_content.strip():
                return ExtractionResult(
                    success=False,
                    entities=[],
                    topics=[],
                    relationships=[],
                    summaries=[],
                    extraction_strategy="llm",
                    processing_time=time.time() - start_time,
                    metadata={},
                    errors=["Empty content"],
                    warnings=[],
                )

            if not self.openai_api_key:
                warnings.append(
                    "OpenAI API key not available, falling back to basic extraction"
                )
                basic_extractor = BasicMetadataExtractor(self.config)
                return basic_extractor.extract_metadata(content)

            # Extract entities using LLM
            entities = self._extract_entities_with_llm(text_content)

            # Extract topics using LLM
            topics = self._extract_topics_with_llm(text_content)

            # Extract relationships using LLM
            relationships = self._extract_relationships_with_llm(text_content, entities)

            # Generate summary using LLM
            summaries = self._generate_summary_with_llm(text_content)

            # Assess quality
            quality_score = self._assess_extraction_quality(
                entities, topics, relationships
            )

            return ExtractionResult(
                success=True,
                entities=entities,
                topics=topics,
                relationships=relationships,
                summaries=summaries,
                extraction_strategy="llm",
                processing_time=time.time() - start_time,
                metadata={
                    "quality_score": quality_score,
                    "content_length": len(text_content),
                    "word_count": len(text_content.split()),
                    "llm_provider": "openai",
                },
                errors=errors,
                warnings=warnings,
            )

        except Exception as e:
            error_msg = f"Error in LLM metadata extraction: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)

            return ExtractionResult(
                success=False,
                entities=[],
                topics=[],
                relationships=[],
                summaries=[],
                extraction_strategy="llm",
                processing_time=time.time() - start_time,
                metadata={},
                errors=errors,
                warnings=warnings,
            )

    def _extract_entities_with_llm(self, text: str) -> List[Entity]:
        """Extract entities using OpenAI API."""
        try:
            import openai

            # Set API key
            openai.api_key = self.openai_api_key

            # Prepare prompt for entity extraction
            prompt = f"""
            Extract named entities from the following text. Return the results as a JSON array with the following structure:
            [
                {{
                    "text": "entity text",
                    "entity_type": "person|organization|location|date|time|money|percent|product|event",
                    "confidence": 0.0-1.0,
                    "start_position": character_position,
                    "end_position": character_position
                }}
            ]
            
            Text: {text[:4000]}  # Limit text length for API
            
            Focus on the most important and relevant entities. Only return valid JSON.
            """

            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise entity extraction assistant. Return only valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1000,
                temperature=self.llm_temperature,
            )

            # Parse response
            response_text = response.choices[0].message.content.strip()

            try:
                entities_data = json.loads(response_text)
                entities = []

                for entity_data in entities_data:
                    if isinstance(entity_data, dict):
                        entity = Entity(
                            text=entity_data.get("text", ""),
                            entity_type=entity_data.get("entity_type", "unknown"),
                            confidence=float(entity_data.get("confidence", 0.8)),
                            start_position=entity_data.get("start_position"),
                            end_position=entity_data.get("end_position"),
                        )
                        entities.append(entity)

                return entities

            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM response as JSON")
                return []

        except ImportError:
            logger.warning("OpenAI library not available")
            return []
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return []

    def _extract_topics_with_llm(self, text: str) -> List[Topic]:
        """Extract topics using OpenAI API."""
        try:
            import openai

            # Set API key
            openai.api_key = self.openai_api_key

            # Prepare prompt for topic extraction
            prompt = f"""
            Extract main topics from the following text. Return the results as a JSON array with the following structure:
            [
                {{
                    "name": "topic name",
                    "confidence": 0.0-1.0,
                    "keywords": ["keyword1", "keyword2"],
                    "description": "brief description",
                    "category": "general category"
                }}
            ]
            
            Text: {text[:4000]}
            
            Identify 3-5 main topics. Only return valid JSON.
            """

            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise topic extraction assistant. Return only valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1000,
                temperature=self.llm_temperature,
            )

            # Parse response
            response_text = response.choices[0].message.content.strip()

            try:
                topics_data = json.loads(response_text)
                topics = []

                for topic_data in topics_data:
                    if isinstance(topic_data, dict):
                        topic = Topic(
                            name=topic_data.get("name", ""),
                            confidence=float(topic_data.get("confidence", 0.8)),
                            keywords=topic_data.get("keywords", []),
                            description=topic_data.get("description"),
                            category=topic_data.get("category"),
                        )
                        topics.append(topic)

                return topics

            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM response as JSON")
                return []

        except ImportError:
            logger.warning("OpenAI library not available")
            return []
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return []

    def _extract_relationships_with_llm(
        self, text: str, entities: List[Entity]
    ) -> List[Relationship]:
        """Extract relationships using OpenAI API."""
        try:
            import openai

            # Set API key
            openai.api_key = self.openai_api_key

            # Prepare entity list for relationship extraction
            entity_list = [
                f"{e.text} ({e.entity_type})" for e in entities[:10]
            ]  # Limit entities

            prompt = f"""
            Extract relationships between entities from the following text. Return the results as a JSON array with the following structure:
            [
                {{
                    "source_entity": "entity1",
                    "target_entity": "entity2",
                    "relationship_type": "type of relationship",
                    "confidence": 0.0-1.0,
                    "context": "sentence or context where relationship appears"
                }}
            ]
            
            Text: {text[:4000]}
            Entities: {', '.join(entity_list)}
            
            Focus on meaningful relationships. Only return valid JSON.
            """

            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise relationship extraction assistant. Return only valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1000,
                temperature=self.llm_temperature,
            )

            # Parse response
            response_text = response.choices[0].message.content.strip()

            try:
                relationships_data = json.loads(response_text)
                relationships = []

                for rel_data in relationships_data:
                    if isinstance(rel_data, dict):
                        relationship = Relationship(
                            source_entity=rel_data.get("source_entity", ""),
                            target_entity=rel_data.get("target_entity", ""),
                            relationship_type=rel_data.get(
                                "relationship_type", "unknown"
                            ),
                            confidence=float(rel_data.get("confidence", 0.8)),
                            context=rel_data.get("context"),
                        )
                        relationships.append(relationship)

                return relationships

            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM response as JSON")
                return []

        except ImportError:
            logger.warning("OpenAI library not available")
            return []
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return []

    def _generate_summary_with_llm(self, text: str) -> List[Summary]:
        """Generate summary using OpenAI API."""
        try:
            import openai

            # Set API key
            openai.api_key = self.openai_api_key

            # Prepare prompt for summarization
            prompt = f"""
            Generate a concise summary of the following text. Return the results as a JSON object with the following structure:
            {{
                "summary": "summary text",
                "key_points": ["point1", "point2", "point3"],
                "confidence": 0.0-1.0
            }}
            
            Text: {text[:4000]}
            
            Create a clear, informative summary in 2-3 sentences. Only return valid JSON.
            """

            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise summarization assistant. Return only valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=500,
                temperature=self.llm_temperature,
            )

            # Parse response
            response_text = response.choices[0].message.content.strip()

            try:
                summary_data = json.loads(response_text)

                if isinstance(summary_data, dict):
                    summary = Summary(
                        summary_type="llm_generated",
                        content=summary_data.get("summary", ""),
                        length=len(summary_data.get("summary", "")),
                        confidence=float(summary_data.get("confidence", 0.8)),
                        key_points=summary_data.get("key_points", []),
                    )
                    return [summary]

                return []

            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM response as JSON")
                return []

        except ImportError:
            logger.warning("OpenAI library not available")
            return []
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return []


class MetadataExtractorFactory:
    """Factory for creating metadata extractors."""

    @staticmethod
    def create_extractor(extraction_level: str, config: Config) -> MetadataExtractor:
        """Create an extractor based on the specified extraction level."""
        extraction_level = extraction_level.lower()

        if extraction_level == "basic":
            return BasicMetadataExtractor(config)
        elif extraction_level == "enhanced":
            return BasicMetadataExtractor(config)  # Enhanced basic extraction
        elif extraction_level == "llm_powered":
            return LLMMetadataExtractor(config)
        else:
            logger.warning(
                f"Unknown extraction level '{extraction_level}', falling back to basic"
            )
            return BasicMetadataExtractor(config)


# Convenience function to get extractor instance
def get_metadata_extractor(
    extraction_level: str = "enhanced", config: Optional[Config] = None
) -> MetadataExtractor:
    """Get a configured metadata extractor instance."""
    if config is None:
        config = get_config()
    return MetadataExtractorFactory.create_extractor(extraction_level, config)
