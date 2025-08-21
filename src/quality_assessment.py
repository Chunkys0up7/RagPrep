from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import hashlib
import json
import logging
import statistics
import time
from datetime import datetime
from pathlib import Path

from chunkers import ChunkingResult, DocumentChunk
from config import Config, get_config
from metadata_extractors import ExtractionResult
from parsers import ParsedContent, ParserResult

logger = logging.getLogger(__name__)


@dataclass
class QualityMetric:
    """Individual quality metric with score and details."""

    name: str
    score: float  # 0.0 to 1.0
    weight: float = 1.0
    details: Dict[str, Any] = None
    threshold: float = 0.7
    passed: bool = True

    def __post_init__(self):
        if self.details is None:
            self.details = {}
        self.passed = self.score >= self.threshold


@dataclass
class QualityReport:
    """Comprehensive quality assessment report."""

    document_id: str
    timestamp: datetime
    overall_score: float
    metrics: List[QualityMetric]
    passed: bool
    recommendations: List[str]
    processing_time: float
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.recommendations is None:
            self.recommendations = []

        # Calculate overall score as weighted average
        total_weight = sum(metric.weight for metric in self.metrics)
        if total_weight > 0:
            self.overall_score = (
                sum(metric.score * metric.weight for metric in self.metrics)
                / total_weight
            )
        else:
            self.overall_score = 0.0

        # Determine if overall quality passed
        self.passed = all(metric.passed for metric in self.metrics)


@dataclass
class PerformanceMetrics:
    """Performance monitoring metrics."""

    operation: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    input_size: Optional[int] = None
    output_size: Optional[int] = None
    success: bool = True
    error_message: Optional[str] = None

    def __post_init__(self):
        if self.end_time is not None and self.duration is None:
            self.duration = self.end_time - self.start_time


class QualityAssessor(ABC):
    """Abstract base class for quality assessment strategies."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def assess_quality(self, content: Any) -> QualityReport:
        """Assess the quality of processed content."""
        pass

    def _generate_document_id(self, content: Any) -> str:
        """Generate a unique document ID based on content."""
        if hasattr(content, "text_content"):
            content_str = str(content.text_content)
        elif hasattr(content, "content"):
            content_str = str(content.content)
        else:
            content_str = str(content)

        return hashlib.md5(content_str.encode()).hexdigest()[:16]


class ContentCompletenessAssessor(QualityAssessor):
    """Assesses content completeness and coverage."""

    def assess_quality(
        self, content: Union[ParsedContent, DocumentChunk, ChunkingResult]
    ) -> QualityReport:
        """Assess content completeness."""
        start_time = time.time()
        metrics = []

        if isinstance(content, ParsedContent):
            metrics.extend(self._assess_parsed_content(content))
        elif isinstance(content, DocumentChunk):
            metrics.extend(self._assess_chunk_content(content))
        elif isinstance(content, ChunkingResult):
            metrics.extend(self._assess_chunking_result(content))
        else:
            metrics.append(
                QualityMetric(
                    name="content_type_support",
                    score=0.0,
                    details={"error": "Unsupported content type"},
                    threshold=0.0,
                )
            )

        processing_time = time.time() - start_time
        document_id = self._generate_document_id(content)

        return QualityReport(
            document_id=document_id,
            timestamp=datetime.now(),
            overall_score=0.0,  # Will be calculated in __post_init__
            metrics=metrics,
            passed=False,  # Will be calculated in __post_init__
            recommendations=[],
            processing_time=processing_time,
        )

    def _assess_parsed_content(self, content: ParsedContent) -> List[QualityMetric]:
        """Assess parsed content completeness."""
        metrics = []

        # Text content completeness
        if content.text_content:
            text_length = len(content.text_content.strip())
            if text_length > 0:
                text_score = min(
                    1.0, text_length / 1000
                )  # Normalize to 1.0 at 1000 chars
                metrics.append(
                    QualityMetric(
                        name="text_completeness",
                        score=text_score,
                        weight=2.0,
                        details={"text_length": text_length, "has_content": True},
                    )
                )
            else:
                metrics.append(
                    QualityMetric(
                        name="text_completeness",
                        score=0.0,
                        details={"text_length": 0, "has_content": False},
                    )
                )
        else:
            metrics.append(
                QualityMetric(
                    name="text_completeness",
                    score=0.0,
                    details={"text_length": 0, "has_content": False},
                )
            )

        # Structure completeness
        if content.structure:
            structure_score = min(
                1.0, len(content.structure) / 10
            )  # Normalize to 1.0 at 10 elements
            metrics.append(
                QualityMetric(
                    name="structure_completeness",
                    score=structure_score,
                    weight=1.5,
                    details={"structure_elements": len(content.structure)},
                )
            )
        else:
            metrics.append(
                QualityMetric(
                    name="structure_completeness",
                    score=0.0,
                    details={"structure_elements": 0},
                )
            )

        # Metadata completeness
        if content.metadata:
            metadata_score = min(
                1.0, len(content.metadata) / 5
            )  # Normalize to 1.0 at 5 metadata items
            metrics.append(
                QualityMetric(
                    name="metadata_completeness",
                    score=metadata_score,
                    weight=1.0,
                    details={"metadata_items": len(content.metadata)},
                )
            )
        else:
            metrics.append(
                QualityMetric(
                    name="metadata_completeness",
                    score=0.0,
                    details={"metadata_items": 0},
                )
            )

        # Error assessment
        if content.parsing_errors:
            error_score = max(
                0.0, 1.0 - len(content.parsing_errors) * 0.2
            )  # Reduce score for each error
            metrics.append(
                QualityMetric(
                    name="error_freedom",
                    score=error_score,
                    weight=1.5,
                    details={
                        "error_count": len(content.parsing_errors),
                        "errors": content.parsing_errors,
                    },
                )
            )
        else:
            metrics.append(
                QualityMetric(
                    name="error_freedom", score=1.0, details={"error_count": 0}
                )
            )

        return metrics

    def _assess_chunk_content(self, content: DocumentChunk) -> List[QualityMetric]:
        """Assess chunk content completeness."""
        metrics = []

        # Content length assessment
        if content.content:
            content_length = len(content.content.strip())
            if content_length > 0:
                # Score based on optimal chunk size (100-1000 chars)
                if 100 <= content_length <= 1000:
                    length_score = 1.0
                elif content_length < 100:
                    length_score = content_length / 100
                else:
                    length_score = max(0.5, 1.0 - (content_length - 1000) / 1000)

                metrics.append(
                    QualityMetric(
                        name="chunk_length_optimality",
                        score=length_score,
                        weight=1.5,
                        details={
                            "content_length": content_length,
                            "optimal_range": "100-1000",
                        },
                    )
                )
            else:
                metrics.append(
                    QualityMetric(
                        name="chunk_length_optimality",
                        score=0.0,
                        details={"content_length": 0},
                    )
                )
        else:
            metrics.append(
                QualityMetric(
                    name="chunk_length_optimality",
                    score=0.0,
                    details={"content_length": 0},
                )
            )

        # Metadata completeness
        if content.metadata:
            metadata_score = min(
                1.0, len(content.metadata) / 3
            )  # Normalize to 1.0 at 3 metadata items
            metrics.append(
                QualityMetric(
                    name="chunk_metadata_completeness",
                    score=metadata_score,
                    weight=1.0,
                    details={"metadata_items": len(content.metadata)},
                )
            )
        else:
            metrics.append(
                QualityMetric(
                    name="chunk_metadata_completeness",
                    score=0.0,
                    details={"metadata_items": 0},
                )
            )

        # Quality score assessment
        if hasattr(content, "quality_score") and content.quality_score is not None:
            metrics.append(
                QualityMetric(
                    name="chunk_quality_score",
                    score=content.quality_score,
                    weight=1.0,
                    details={"quality_score": content.quality_score},
                )
            )
        else:
            metrics.append(
                QualityMetric(
                    name="chunk_quality_score",
                    score=0.5,  # Default score for unknown quality
                    weight=1.0,
                    details={"quality_score": "unknown"},
                )
            )

        return metrics

    def _assess_chunking_result(self, content: ChunkingResult) -> List[QualityMetric]:
        """Assess chunking result completeness."""
        metrics = []

        # Success assessment
        success_score = 1.0 if content.success else 0.0
        metrics.append(
            QualityMetric(
                name="chunking_success",
                score=success_score,
                weight=2.0,
                details={"success": content.success},
            )
        )

        # Chunk count assessment
        if content.chunks:
            chunk_count = len(content.chunks)
            if chunk_count > 0:
                # Score based on reasonable chunk count (not too few, not too many)
                if 1 <= chunk_count <= 50:
                    count_score = 1.0
                elif chunk_count < 1:
                    count_score = 0.0
                else:
                    count_score = max(0.3, 1.0 - (chunk_count - 50) / 100)

                metrics.append(
                    QualityMetric(
                        name="chunk_count_optimality",
                        score=count_score,
                        weight=1.5,
                        details={"chunk_count": chunk_count, "optimal_range": "1-50"},
                    )
                )
            else:
                metrics.append(
                    QualityMetric(
                        name="chunk_count_optimality",
                        score=0.0,
                        details={"chunk_count": 0},
                    )
                )
        else:
            metrics.append(
                QualityMetric(
                    name="chunk_count_optimality", score=0.0, details={"chunk_count": 0}
                )
            )

        # Processing time assessment
        if content.processing_time:
            # Score based on reasonable processing time (under 30 seconds)
            if content.processing_time <= 30:
                time_score = 1.0
            else:
                time_score = max(0.1, 1.0 - (content.processing_time - 30) / 300)

            metrics.append(
                QualityMetric(
                    name="processing_efficiency",
                    score=time_score,
                    weight=1.0,
                    details={
                        "processing_time": content.processing_time,
                        "optimal_threshold": 30,
                    },
                )
            )

        # Error assessment
        if content.errors:
            error_score = max(
                0.0, 1.0 - len(content.errors) * 0.3
            )  # Reduce score for each error
            metrics.append(
                QualityMetric(
                    name="error_freedom",
                    score=error_score,
                    weight=1.5,
                    details={
                        "error_count": len(content.errors),
                        "errors": content.errors,
                    },
                )
            )
        else:
            metrics.append(
                QualityMetric(
                    name="error_freedom", score=1.0, details={"error_count": 0}
                )
            )

        return metrics


class StructureIntegrityAssessor(QualityAssessor):
    """Assesses structural integrity and consistency."""

    def assess_quality(
        self, content: Union[ParsedContent, DocumentChunk, ChunkingResult]
    ) -> QualityReport:
        """Assess structural integrity."""
        start_time = time.time()
        metrics = []

        if isinstance(content, ParsedContent):
            metrics.extend(self._assess_parsed_structure(content))
        elif isinstance(content, DocumentChunk):
            metrics.extend(self._assess_chunk_structure(content))
        elif isinstance(content, ChunkingResult):
            metrics.extend(self._assess_chunking_structure(content))
        else:
            metrics.append(
                QualityMetric(
                    name="structure_type_support",
                    score=0.0,
                    details={"error": "Unsupported content type"},
                    threshold=0.0,
                )
            )

        processing_time = time.time() - start_time
        document_id = self._generate_document_id(content)

        return QualityReport(
            document_id=document_id,
            timestamp=datetime.now(),
            overall_score=0.0,
            metrics=metrics,
            passed=False,
            recommendations=[],
            processing_time=processing_time,
        )

    def _assess_parsed_structure(self, content: ParsedContent) -> List[QualityMetric]:
        """Assess parsed content structure integrity."""
        metrics = []

        # Structure consistency
        if content.structure:
            structure_items = len(content.structure)
            if structure_items > 0:
                # Check for structural patterns
                has_headings = any(
                    "heading" in str(item).lower() for item in content.structure
                )
                has_paragraphs = any(
                    "paragraph" in str(item).lower() for item in content.structure
                )
                has_sections = any(
                    "section" in str(item).lower() for item in content.structure
                )

                structure_variety = (
                    sum([has_headings, has_paragraphs, has_sections]) / 3
                )
                metrics.append(
                    QualityMetric(
                        name="structure_variety",
                        score=structure_variety,
                        weight=1.5,
                        details={
                            "has_headings": has_headings,
                            "has_paragraphs": has_paragraphs,
                            "has_sections": has_sections,
                            "structure_items": structure_items,
                        },
                    )
                )
            else:
                metrics.append(
                    QualityMetric(
                        name="structure_variety",
                        score=0.0,
                        details={"structure_items": 0},
                    )
                )
        else:
            metrics.append(
                QualityMetric(
                    name="structure_variety", score=0.0, details={"structure_items": 0}
                )
            )

        # Content organization
        if content.text_content:
            # Check for logical organization (headings, paragraphs, etc.)
            lines = content.text_content.split("\n")
            non_empty_lines = [line.strip() for line in lines if line.strip()]

            if len(non_empty_lines) > 1:
                # Check for heading patterns
                heading_patterns = sum(
                    1
                    for line in non_empty_lines
                    if line.isupper() or line.startswith("#")
                )
                organization_score = min(1.0, heading_patterns / len(non_empty_lines))

                metrics.append(
                    QualityMetric(
                        name="content_organization",
                        score=organization_score,
                        weight=1.0,
                        details={
                            "total_lines": len(non_empty_lines),
                            "heading_patterns": heading_patterns,
                            "organization_ratio": organization_score,
                        },
                    )
                )
            else:
                metrics.append(
                    QualityMetric(
                        name="content_organization",
                        score=0.5,
                        weight=1.0,
                        details={"total_lines": len(non_empty_lines)},
                    )
                )
        else:
            metrics.append(
                QualityMetric(
                    name="content_organization",
                    score=0.0,
                    weight=1.0,
                    details={"total_lines": 0},
                )
            )

        return metrics

    def _assess_chunk_structure(self, content: DocumentChunk) -> List[QualityMetric]:
        """Assess chunk structure integrity."""
        metrics = []

        # Chunk ID consistency
        if content.chunk_id:
            id_validity = 1.0 if len(content.chunk_id) >= 8 else 0.5
            metrics.append(
                QualityMetric(
                    name="chunk_id_validity",
                    score=id_validity,
                    weight=0.5,
                    details={"chunk_id_length": len(content.chunk_id)},
                )
            )
        else:
            metrics.append(
                QualityMetric(
                    name="chunk_id_validity",
                    score=0.0,
                    weight=0.5,
                    details={"chunk_id_length": 0},
                )
            )

        # Chunk type consistency
        if content.chunk_type:
            type_validity = (
                1.0
                if content.chunk_type in ["text", "table", "image", "mixed"]
                else 0.5
            )
            metrics.append(
                QualityMetric(
                    name="chunk_type_consistency",
                    score=type_validity,
                    weight=1.0,
                    details={"chunk_type": content.chunk_type},
                )
            )
        else:
            metrics.append(
                QualityMetric(
                    name="chunk_type_consistency",
                    score=0.0,
                    weight=1.0,
                    details={"chunk_type": "unknown"},
                )
            )

        # Index consistency
        if content.chunk_index is not None:
            index_validity = 1.0 if content.chunk_index >= 0 else 0.0
            metrics.append(
                QualityMetric(
                    name="chunk_index_validity",
                    score=index_validity,
                    weight=0.5,
                    details={"chunk_index": content.chunk_index},
                )
            )
        else:
            metrics.append(
                QualityMetric(
                    name="chunk_index_validity",
                    score=0.0,
                    weight=0.5,
                    details={"chunk_index": "unknown"},
                )
            )

        return metrics

    def _assess_chunking_structure(
        self, content: ChunkingResult
    ) -> List[QualityMetric]:
        """Assess chunking result structure integrity."""
        metrics = []

        # Strategy consistency
        if content.chunking_strategy:
            strategy_validity = (
                1.0
                if content.chunking_strategy
                in ["fixed", "structural", "semantic", "hybrid"]
                else 0.5
            )
            metrics.append(
                QualityMetric(
                    name="strategy_consistency",
                    score=strategy_validity,
                    weight=1.0,
                    details={"chunking_strategy": content.chunking_strategy},
                )
            )
        else:
            metrics.append(
                QualityMetric(
                    name="strategy_consistency",
                    score=0.0,
                    weight=1.0,
                    details={"chunking_strategy": "unknown"},
                )
            )

        # Chunk ordering
        if content.chunks and len(content.chunks) > 1:
            # Check if chunks are properly ordered by index
            indices = [
                chunk.chunk_index
                for chunk in content.chunks
                if chunk.chunk_index is not None
            ]
            if indices:
                sorted_indices = sorted(indices)
                ordering_score = 1.0 if indices == sorted_indices else 0.5
                metrics.append(
                    QualityMetric(
                        name="chunk_ordering",
                        score=ordering_score,
                        weight=1.0,
                        details={
                            "indices": indices,
                            "sorted_indices": sorted_indices,
                            "properly_ordered": indices == sorted_indices,
                        },
                    )
                )
            else:
                metrics.append(
                    QualityMetric(
                        name="chunk_ordering",
                        score=0.5,
                        weight=1.0,
                        details={"indices": "none"},
                    )
                )
        else:
            metrics.append(
                QualityMetric(
                    name="chunk_ordering",
                    score=1.0,  # Single chunk or no chunks
                    weight=1.0,
                    details={
                        "chunk_count": len(content.chunks) if content.chunks else 0
                    },
                )
            )

        return metrics


class MetadataAccuracyAssessor(QualityAssessor):
    """Assesses metadata accuracy and consistency."""

    def assess_quality(
        self, content: Union[ExtractionResult, ParsedContent, DocumentChunk]
    ) -> QualityReport:
        """Assess metadata accuracy."""
        start_time = time.time()
        metrics = []

        if isinstance(content, ExtractionResult):
            metrics.extend(self._assess_extraction_result(content))
        elif isinstance(content, ParsedContent):
            metrics.extend(self._assess_parsed_metadata(content))
        elif isinstance(content, DocumentChunk):
            metrics.extend(self._assess_chunk_metadata(content))
        else:
            metrics.append(
                QualityMetric(
                    name="metadata_type_support",
                    score=0.0,
                    details={"error": "Unsupported content type"},
                    threshold=0.0,
                )
            )

        processing_time = time.time() - start_time
        document_id = self._generate_document_id(content)

        return QualityReport(
            document_id=document_id,
            timestamp=datetime.now(),
            overall_score=0.0,
            metrics=metrics,
            passed=False,
            recommendations=[],
            processing_time=processing_time,
        )

    def _assess_extraction_result(
        self, content: ExtractionResult
    ) -> List[QualityMetric]:
        """Assess extraction result metadata accuracy."""
        metrics = []

        # Success assessment
        success_score = 1.0 if content.success else 0.0
        metrics.append(
            QualityMetric(
                name="extraction_success",
                score=success_score,
                weight=2.0,
                details={"success": content.success},
            )
        )

        # Entity extraction quality
        if content.entities:
            entity_count = len(content.entities)
            # Score based on reasonable entity count (not too few, not too many)
            if 1 <= entity_count <= 100:
                entity_score = 1.0
            elif entity_count < 1:
                entity_score = 0.0
            else:
                entity_score = max(0.3, 1.0 - (entity_count - 100) / 1000)

            metrics.append(
                QualityMetric(
                    name="entity_extraction_quality",
                    score=entity_score,
                    weight=1.5,
                    details={"entity_count": entity_count, "optimal_range": "1-100"},
                )
            )
        else:
            metrics.append(
                QualityMetric(
                    name="entity_extraction_quality",
                    score=0.5,  # Neutral score for no entities
                    weight=1.5,
                    details={"entity_count": 0},
                )
            )

        # Topic extraction quality
        if content.topics:
            topic_count = len(content.topics)
            if 1 <= topic_count <= 20:
                topic_score = 1.0
            elif topic_count < 1:
                topic_score = 0.0
            else:
                topic_score = max(0.3, 1.0 - (topic_count - 20) / 100)

            metrics.append(
                QualityMetric(
                    name="topic_extraction_quality",
                    score=topic_score,
                    weight=1.0,
                    details={"topic_count": topic_count, "optimal_range": "1-20"},
                )
            )
        else:
            metrics.append(
                QualityMetric(
                    name="topic_extraction_quality",
                    score=0.5,
                    weight=1.0,
                    details={"topic_count": 0},
                )
            )

        # Processing time assessment
        if content.processing_time:
            if content.processing_time <= 60:  # 1 minute threshold
                time_score = 1.0
            else:
                time_score = max(0.1, 1.0 - (content.processing_time - 60) / 600)

            metrics.append(
                QualityMetric(
                    name="extraction_efficiency",
                    score=time_score,
                    weight=1.0,
                    details={
                        "processing_time": content.processing_time,
                        "optimal_threshold": 60,
                    },
                )
            )

        return metrics

    def _assess_parsed_metadata(self, content: ParsedContent) -> List[QualityMetric]:
        """Assess parsed content metadata accuracy."""
        metrics = []

        # Metadata completeness
        if content.metadata:
            metadata_count = len(content.metadata)
            if metadata_count > 0:
                # Check for essential metadata fields
                essential_fields = ["format", "size", "created_date", "modified_date"]
                present_fields = sum(
                    1 for field in essential_fields if field in content.metadata
                )
                completeness_score = present_fields / len(essential_fields)

                metrics.append(
                    QualityMetric(
                        name="metadata_completeness",
                        score=completeness_score,
                        weight=1.5,
                        details={
                            "metadata_count": metadata_count,
                            "essential_fields": essential_fields,
                            "present_fields": present_fields,
                        },
                    )
                )
            else:
                metrics.append(
                    QualityMetric(
                        name="metadata_completeness",
                        score=0.0,
                        weight=1.5,
                        details={"metadata_count": 0},
                    )
                )
        else:
            metrics.append(
                QualityMetric(
                    name="metadata_completeness",
                    score=0.0,
                    weight=1.5,
                    details={"metadata_count": 0},
                )
            )

        # Parser information
        if content.parser_used:
            parser_score = (
                1.0
                if content.parser_used
                in ["marker", "pymupdf", "unstructured", "python-docx", "beautifulsoup"]
                else 0.5
            )
            metrics.append(
                QualityMetric(
                    name="parser_validity",
                    score=parser_score,
                    weight=1.0,
                    details={"parser_used": content.parser_used},
                )
            )
        else:
            metrics.append(
                QualityMetric(
                    name="parser_validity",
                    score=0.0,
                    weight=1.0,
                    details={"parser_used": "unknown"},
                )
            )

        return metrics

    def _assess_chunk_metadata(self, content: DocumentChunk) -> List[QualityMetric]:
        """Assess chunk metadata accuracy."""
        metrics = []

        # Metadata consistency
        if content.metadata:
            metadata_count = len(content.metadata)
            if metadata_count > 0:
                # Check for chunk-specific metadata
                chunk_fields = ["chunk_type", "chunk_index", "parent_chunk_id"]
                present_fields = sum(
                    1 for field in chunk_fields if field in content.metadata
                )
                consistency_score = present_fields / len(chunk_fields)

                metrics.append(
                    QualityMetric(
                        name="chunk_metadata_consistency",
                        score=consistency_score,
                        weight=1.0,
                        details={
                            "metadata_count": metadata_count,
                            "chunk_fields": chunk_fields,
                            "present_fields": present_fields,
                        },
                    )
                )
            else:
                metrics.append(
                    QualityMetric(
                        name="chunk_metadata_consistency",
                        score=0.0,
                        weight=1.0,
                        details={"metadata_count": 0},
                    )
                )
        else:
            metrics.append(
                QualityMetric(
                    name="chunk_metadata_consistency",
                    score=0.0,
                    weight=1.0,
                    details={"metadata_count": 0},
                )
            )

        # Quality score validation
        if hasattr(content, "quality_score") and content.quality_score is not None:
            if 0.0 <= content.quality_score <= 1.0:
                score_validity = 1.0
            else:
                score_validity = 0.0

            metrics.append(
                QualityMetric(
                    name="quality_score_validity",
                    score=score_validity,
                    weight=0.5,
                    details={
                        "quality_score": content.quality_score,
                        "valid_range": "0.0-1.0",
                    },
                )
            )
        else:
            metrics.append(
                QualityMetric(
                    name="quality_score_validity",
                    score=0.0,
                    weight=0.5,
                    details={"quality_score": "missing"},
                )
            )

        return metrics


class PerformanceMonitor:
    """Monitors and tracks performance metrics."""

    def __init__(self, config: Config):
        self.config = config
        self.metrics: List[PerformanceMetrics] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def start_operation(self, operation: str, input_size: Optional[int] = None) -> str:
        """Start monitoring an operation and return operation ID."""
        operation_id = f"{operation}_{int(time.time() * 1000)}"

        metric = PerformanceMetrics(
            operation=operation, start_time=time.time(), input_size=input_size
        )

        self.metrics.append(metric)
        return operation_id

    def end_operation(
        self,
        operation_id: str,
        success: bool = True,
        error_message: Optional[str] = None,
        output_size: Optional[int] = None,
    ) -> None:
        """End monitoring an operation."""
        found_metric = False
        for metric in reversed(self.metrics):
            if metric.operation in operation_id and not hasattr(metric, "end_time"):
                metric.end_time = time.time()
                metric.success = success
                metric.error_message = error_message
                metric.output_size = output_size
                found_metric = True
                break

        if not found_metric:
            self.logger.warning(
                f"Operation ID '{operation_id}' not found or already ended. Cannot log end time."
            )

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of performance metrics."""
        if not self.metrics:
            return {"message": "No performance metrics available"}

        successful_ops = [m for m in self.metrics if m.success]
        failed_ops = [m for m in self.metrics if not m.success]

        # Calculate durations only for completed operations
        completed_metrics = [
            m for m in self.metrics if hasattr(m, "duration") and m.duration is not None
        ]

        summary = {
            "total_operations": len(self.metrics),
            "successful_operations": len(successful_ops),
            "failed_operations": len(failed_ops),
            "success_rate": (
                len(successful_ops) / len(self.metrics) if self.metrics else 0.0
            ),
            "average_duration": (
                statistics.mean([m.duration for m in completed_metrics])
                if completed_metrics
                else 0.0
            ),
            "total_duration": sum([m.duration for m in completed_metrics]),
            "operations_by_type": {},
        }

        # Group operations by type
        for metric in self.metrics:
            op_type = metric.operation
            if op_type not in summary["operations_by_type"]:
                summary["operations_by_type"][op_type] = {
                    "count": 0,
                    "success_count": 0,
                    "total_duration": 0.0,
                    "average_duration": 0.0,
                }

            summary["operations_by_type"][op_type]["count"] += 1
            if metric.success:
                summary["operations_by_type"][op_type]["success_count"] += 1
            if hasattr(metric, "duration") and metric.duration is not None:
                summary["operations_by_type"][op_type][
                    "total_duration"
                ] += metric.duration

        # Calculate averages for each operation type
        for op_type in summary["operations_by_type"]:
            op_data = summary["operations_by_type"][op_type]
            if op_data["count"] > 0:
                op_data["average_duration"] = (
                    op_data["total_duration"] / op_data["count"]
                )

        return summary

    def export_metrics(self, file_path: str) -> None:
        """Export performance metrics to a JSON file."""
        try:
            metrics_data = []
            for metric in self.metrics:
                metric_dict = {
                    "operation": metric.operation,
                    "start_time": metric.start_time,
                    "end_time": getattr(metric, "end_time", None),
                    "duration": getattr(metric, "duration", None),
                    "success": metric.success,
                    "error_message": metric.error_message,
                    "input_size": metric.input_size,
                    "output_size": metric.output_size,
                }
                metrics_data.append(metric_dict)

            with open(file_path, "w") as f:
                json.dump(metrics_data, f, indent=2, default=str)

            self.logger.info(f"Performance metrics exported to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to export performance metrics: {e}")


class QualityAssessmentSystem:
    """Main quality assessment system that orchestrates all assessors."""

    def __init__(self, config: Config):
        self.config = config
        self.content_assessor = ContentCompletenessAssessor(config)
        self.structure_assessor = StructureIntegrityAssessor(config)
        self.metadata_assessor = MetadataAccuracyAssessor(config)
        self.performance_monitor = PerformanceMonitor(config)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def assess_quality(
        self, content: Any, chunks: List[DocumentChunk], metadata_result: Any
    ) -> QualityReport:
        """Perform comprehensive quality assessment on document content (alias for assess_document_quality)."""
        return self.assess_document_quality(content)

    def assess_document_quality(self, content: Any) -> QualityReport:
        """Perform comprehensive quality assessment on document content."""
        operation_id = self.performance_monitor.start_operation("quality_assessment")

        try:
            # Run all quality assessments
            content_report = self.content_assessor.assess_quality(content)
            structure_report = self.structure_assessor.assess_quality(content)
            metadata_report = self.metadata_assessor.assess_quality(content)

            # Combine metrics from all assessors
            all_metrics = []
            all_metrics.extend(content_report.metrics)
            all_metrics.extend(structure_report.metrics)
            all_metrics.extend(metadata_report.metrics)

            # Generate recommendations
            recommendations = self._generate_recommendations(all_metrics)

            # Create comprehensive report
            comprehensive_report = QualityReport(
                document_id=content_report.document_id,
                timestamp=datetime.now(),
                overall_score=0.0,  # Will be calculated in __post_init__
                metrics=all_metrics,
                passed=False,  # Will be calculated in __post_init__
                recommendations=recommendations,
                processing_time=time.time() - content_report.timestamp.timestamp(),
                metadata={
                    "assessors_used": ["content", "structure", "metadata"],
                    "total_metrics": len(all_metrics),
                },
            )

            self.performance_monitor.end_operation(operation_id, success=True)
            return comprehensive_report

        except Exception as e:
            self.logger.error(f"Quality assessment failed: {e}")
            self.performance_monitor.end_operation(
                operation_id, success=False, error_message=str(e)
            )
            raise

    def _generate_recommendations(self, metrics: List[QualityMetric]) -> List[str]:
        """Generate recommendations based on quality metrics."""
        recommendations = []

        for metric in metrics:
            if not metric.passed:
                if metric.name == "text_completeness":
                    recommendations.append(
                        "Consider using a different parser or document format for better text extraction"
                    )
                elif metric.name == "structure_completeness":
                    recommendations.append(
                        "Document structure could be improved with better formatting or parsing"
                    )
                elif metric.name == "chunk_length_optimality":
                    recommendations.append(
                        "Adjust chunking parameters for optimal chunk sizes"
                    )
                elif metric.name == "error_freedom":
                    recommendations.append(
                        "Review and fix parsing errors to improve document quality"
                    )
                elif metric.name == "extraction_success":
                    recommendations.append(
                        "Check extraction configuration and input document quality"
                    )
                elif metric.name == "processing_efficiency":
                    recommendations.append(
                        "Consider optimizing processing parameters or using more efficient strategies"
                    )

        # Add general recommendations if overall quality is low
        failed_metrics = [m for m in metrics if not m.passed]
        if len(failed_metrics) > len(metrics) * 0.3:  # More than 30% failed
            recommendations.append(
                "Overall document quality is low. Consider preprocessing or using different source documents"
            )

        return recommendations

    def get_quality_summary(self) -> Dict[str, Any]:
        """Get a summary of quality assessment results."""
        return {
            "quality_assessors": {
                "content_completeness": "ContentCompletenessAssessor",
                "structure_integrity": "StructureIntegrityAssessor",
                "metadata_accuracy": "MetadataAccuracyAssessor",
            },
            "performance_monitoring": "PerformanceMonitor",
            "total_metrics_tracked": len(self.performance_monitor.metrics),
        }


def get_quality_assessment_system(
    config: Optional[Config] = None,
) -> QualityAssessmentSystem:
    """Convenience function to get a quality assessment system instance."""
    if config is None:
        config = get_config()
    return QualityAssessmentSystem(config)
