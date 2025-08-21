"""
Tests for quality assessment system
"""

import pytest
import time
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.quality_assessment import (
    QualityMetric,
    QualityReport,
    PerformanceMetrics,
    QualityAssessor,
    ContentCompletenessAssessor,
    StructureIntegrityAssessor,
    MetadataAccuracyAssessor,
    PerformanceMonitor,
    QualityAssessmentSystem,
    get_quality_assessment_system,
)
from src.config import Config
from src.parsers import ParsedContent, ParserResult
from src.chunkers import ChunkingResult, DocumentChunk
from src.metadata_extractors import ExtractionResult, Entity, Topic


class TestQualityMetric:
    """Test the QualityMetric dataclass."""

    def test_quality_metric_creation(self):
        """Test basic QualityMetric creation."""
        metric = QualityMetric(name="test_metric", score=0.8, weight=1.5, threshold=0.7)

        assert metric.name == "test_metric"
        assert metric.score == 0.8
        assert metric.weight == 1.5
        assert metric.threshold == 0.7
        assert metric.passed is True
        assert metric.details == {}

    def test_quality_metric_passed_calculation(self):
        """Test that passed is calculated correctly based on score and threshold."""
        # Score above threshold
        metric1 = QualityMetric(name="test1", score=0.9, threshold=0.7)
        assert metric1.passed is True

        # Score at threshold
        metric2 = QualityMetric(name="test2", score=0.7, threshold=0.7)
        assert metric2.passed is True

        # Score below threshold
        metric3 = QualityMetric(name="test3", score=0.6, threshold=0.7)
        assert metric3.passed is False

    def test_quality_metric_defaults(self):
        """Test QualityMetric default values."""
        metric = QualityMetric(name="test", score=0.5)

        assert metric.weight == 1.0
        assert metric.threshold == 0.7
        assert metric.details == {}
        assert metric.passed is False


class TestQualityReport:
    """Test the QualityReport dataclass."""

    def test_quality_report_creation(self):
        """Test basic QualityReport creation."""
        metrics = [
            QualityMetric(name="metric1", score=0.8, weight=1.0),
            QualityMetric(name="metric2", score=0.9, weight=2.0),
        ]

        report = QualityReport(
            document_id="doc123",
            timestamp=datetime.now(),
            overall_score=0.0,  # Will be calculated
            metrics=metrics,
            passed=False,  # Will be calculated
            recommendations=["rec1", "rec2"],
            processing_time=1.5,
        )

        assert report.document_id == "doc123"
        assert len(report.metrics) == 2
        assert len(report.recommendations) == 2
        assert report.processing_time == 1.5
        assert report.metadata == {}

    def test_quality_report_overall_score_calculation(self):
        """Test that overall score is calculated as weighted average."""
        metrics = [
            QualityMetric(name="metric1", score=0.8, weight=1.0),
            QualityMetric(name="metric2", score=0.9, weight=2.0),
        ]

        report = QualityReport(
            document_id="doc123",
            timestamp=datetime.now(),
            overall_score=0.0,
            metrics=metrics,
            passed=False,
            recommendations=[],
            processing_time=1.0,
        )

        # Expected: (0.8 * 1.0 + 0.9 * 2.0) / (1.0 + 2.0) = 0.867
        expected_score = (0.8 * 1.0 + 0.9 * 2.0) / (1.0 + 2.0)
        assert abs(report.overall_score - expected_score) < 0.001

    def test_quality_report_passed_calculation(self):
        """Test that passed is calculated based on all metrics passing."""
        # All metrics pass
        metrics1 = [
            QualityMetric(name="metric1", score=0.8, threshold=0.7),
            QualityMetric(name="metric2", score=0.9, threshold=0.7),
        ]

        report1 = QualityReport(
            document_id="doc123",
            timestamp=datetime.now(),
            overall_score=0.0,
            metrics=metrics1,
            passed=False,
            recommendations=[],
            processing_time=1.0,
        )
        assert report1.passed is True

        # Some metrics fail
        metrics2 = [
            QualityMetric(name="metric1", score=0.8, threshold=0.7),
            QualityMetric(name="metric2", score=0.6, threshold=0.7),
        ]

        report2 = QualityReport(
            document_id="doc123",
            timestamp=datetime.now(),
            overall_score=0.0,
            metrics=metrics2,
            passed=False,
            recommendations=[],
            processing_time=1.0,
        )
        assert report2.passed is False


class TestPerformanceMetrics:
    """Test the PerformanceMetrics dataclass."""

    def test_performance_metrics_creation(self):
        """Test basic PerformanceMetrics creation."""
        start_time = time.time()
        end_time = start_time + 1.5

        metric = PerformanceMetrics(
            operation="test_op",
            start_time=start_time,
            end_time=end_time,
            input_size=1000,
            output_size=500,
        )

        assert metric.operation == "test_op"
        assert metric.start_time == start_time
        assert metric.end_time == end_time
        assert metric.duration == 1.5
        assert metric.input_size == 1000
        assert metric.output_size == 500
        assert metric.success is True
        assert metric.error_message is None

    def test_performance_metrics_duration_calculation(self):
        """Test that duration is calculated correctly."""
        start_time = time.time()
        end_time = start_time + 2.5

        metric = PerformanceMetrics(
            operation="test_op", start_time=start_time, end_time=end_time
        )

        assert abs(metric.duration - 2.5) < 0.001


class TestContentCompletenessAssessor:
    """Test the ContentCompletenessAssessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = Mock(spec=Config)
        self.assessor = ContentCompletenessAssessor(self.config)

    def test_assess_parsed_content_text_completeness(self):
        """Test assessment of parsed content text completeness."""
        # Create mock parsed content with text
        content = Mock(spec=ParsedContent)
        content.text_content = (
            "This is a sample text content with some length to test completeness."
        )
        content.structure = ["heading1", "paragraph1", "paragraph2"]
        content.metadata = {"format": "pdf", "size": 1024}
        content.parsing_errors = []

        report = self.assessor.assess_quality(content)

        # Check that metrics were generated
        assert len(report.metrics) > 0

        # Find text completeness metric
        text_metric = next(
            (m for m in report.metrics if m.name == "text_completeness"), None
        )
        assert text_metric is not None
        assert text_metric.score > 0.0
        assert text_metric.details["has_content"] is True

    def test_assess_parsed_content_empty_text(self):
        """Test assessment of parsed content with empty text."""
        content = Mock(spec=ParsedContent)
        content.text_content = ""
        content.structure = []
        content.metadata = {}
        content.parsing_errors = []

        report = self.assessor.assess_quality(content)

        text_metric = next(
            (m for m in report.metrics if m.name == "text_completeness"), None
        )
        assert text_metric is not None
        assert text_metric.score == 0.0
        assert text_metric.details["has_content"] is False

    def test_assess_chunk_content_length_optimality(self):
        """Test assessment of chunk content length optimality."""
        # Create mock chunk with optimal length
        chunk = Mock(spec=DocumentChunk)
        chunk.content = (
            "This is a chunk with optimal length between 100 and 1000 characters."
        )
        chunk.metadata = {"chunk_type": "text", "chunk_index": 0}
        chunk.quality_score = 0.8

        report = self.assessor.assess_quality(chunk)

        length_metric = next(
            (m for m in report.metrics if m.name == "chunk_length_optimality"), None
        )
        assert length_metric is not None
        assert length_metric.score > 0.0
        assert "optimal_range" in length_metric.details

    def test_assess_chunking_result_success(self):
        """Test assessment of chunking result success."""
        # Create mock chunking result
        chunks = [Mock(spec=DocumentChunk) for _ in range(5)]
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i

        result = Mock(spec=ChunkingResult)
        result.success = True
        result.chunks = chunks
        result.processing_time = 15.0
        result.errors = []

        report = self.assessor.assess_quality(result)

        success_metric = next(
            (m for m in report.metrics if m.name == "chunking_success"), None
        )
        assert success_metric is not None
        assert success_metric.score == 1.0
        assert success_metric.details["success"] is True


class TestStructureIntegrityAssessor:
    """Test the StructureIntegrityAssessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = Mock(spec=Config)
        self.assessor = StructureIntegrityAssessor(self.config)

    def test_assess_parsed_structure_variety(self):
        """Test assessment of parsed content structure variety."""
        content = Mock(spec=ParsedContent)
        content.structure = ["heading1", "paragraph1", "section1", "heading2"]
        content.text_content = "Heading 1\nThis is a paragraph.\nSection 1\nHEADING 2"

        report = self.assessor.assess_quality(content)

        structure_metric = next(
            (m for m in report.metrics if m.name == "structure_variety"), None
        )
        assert structure_metric is not None
        assert structure_metric.score > 0.0
        assert structure_metric.details["structure_items"] == 4

    def test_assess_chunk_structure_consistency(self):
        """Test assessment of chunk structure consistency."""
        chunk = Mock(spec=DocumentChunk)
        chunk.chunk_id = "chunk_12345"
        chunk.chunk_type = "text"
        chunk.chunk_index = 0

        report = self.assessor.assess_quality(chunk)

        id_metric = next(
            (m for m in report.metrics if m.name == "chunk_id_validity"), None
        )
        type_metric = next(
            (m for m in report.metrics if m.name == "chunk_type_consistency"), None
        )
        index_metric = next(
            (m for m in report.metrics if m.name == "chunk_index_validity"), None
        )

        assert id_metric is not None
        assert type_metric is not None
        assert index_metric is not None
        assert id_metric.score == 1.0  # ID length >= 8
        assert type_metric.score == 1.0  # Valid chunk type
        assert index_metric.score == 1.0  # Valid index

    def test_assess_chunking_structure_ordering(self):
        """Test assessment of chunking result structure ordering."""
        chunks = []
        for i in range(3):
            chunk = Mock(spec=DocumentChunk)
            chunk.chunk_index = i
            chunks.append(chunk)

        result = Mock(spec=ChunkingResult)
        result.chunks = chunks
        result.chunking_strategy = "hybrid"

        report = self.assessor.assess_quality(result)

        ordering_metric = next(
            (m for m in report.metrics if m.name == "chunk_ordering"), None
        )
        strategy_metric = next(
            (m for m in report.metrics if m.name == "strategy_consistency"), None
        )

        assert ordering_metric is not None
        assert strategy_metric is not None
        assert ordering_metric.score == 1.0  # Properly ordered
        assert strategy_metric.score == 1.0  # Valid strategy


class TestMetadataAccuracyAssessor:
    """Test the MetadataAccuracyAssessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = Mock(spec=Config)
        self.assessor = MetadataAccuracyAssessor(self.config)

    def test_assess_extraction_result_success(self):
        """Test assessment of extraction result success."""
        # Create mock entities and topics
        entities = [Mock(spec=Entity) for _ in range(5)]
        topics = [Mock(spec=Topic) for _ in range(3)]

        result = Mock(spec=ExtractionResult)
        result.success = True
        result.entities = entities
        result.topics = topics
        result.processing_time = 45.0

        report = self.assessor.assess_quality(result)

        success_metric = next(
            (m for m in report.metrics if m.name == "extraction_success"), None
        )
        entity_metric = next(
            (m for m in report.metrics if m.name == "entity_extraction_quality"), None
        )
        topic_metric = next(
            (m for m in report.metrics if m.name == "topic_extraction_quality"), None
        )

        assert success_metric is not None
        assert entity_metric is not None
        assert topic_metric is not None
        assert success_metric.score == 1.0
        assert entity_metric.score == 1.0  # 5 entities in optimal range
        assert topic_metric.score == 1.0  # 3 topics in optimal range

    def test_assess_parsed_metadata_completeness(self):
        """Test assessment of parsed content metadata completeness."""
        content = Mock(spec=ParsedContent)
        content.metadata = {
            "format": "pdf",
            "size": 2048,
            "created_date": "2024-01-01",
            "modified_date": "2024-01-02",
        }
        content.parser_used = "pymupdf"

        report = self.assessor.assess_quality(content)

        completeness_metric = next(
            (m for m in report.metrics if m.name == "metadata_completeness"), None
        )
        parser_metric = next(
            (m for m in report.metrics if m.name == "parser_validity"), None
        )

        assert completeness_metric is not None
        assert parser_metric is not None
        assert completeness_metric.score == 1.0  # All essential fields present
        assert parser_metric.score == 1.0  # Valid parser

    def test_assess_chunk_metadata_consistency(self):
        """Test assessment of chunk metadata consistency."""
        chunk = Mock(spec=DocumentChunk)
        chunk.metadata = {
            "chunk_type": "text",
            "chunk_index": 0,
            "parent_chunk_id": "parent_123",
        }
        chunk.quality_score = 0.85

        report = self.assessor.assess_quality(chunk)

        consistency_metric = next(
            (m for m in report.metrics if m.name == "chunk_metadata_consistency"), None
        )
        score_metric = next(
            (m for m in report.metrics if m.name == "quality_score_validity"), None
        )

        assert consistency_metric is not None
        assert score_metric is not None
        assert consistency_metric.score == 1.0  # All chunk fields present
        assert score_metric.score == 1.0  # Valid quality score


class TestPerformanceMonitor:
    """Test the PerformanceMonitor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = Mock(spec=Config)
        self.monitor = PerformanceMonitor(self.config)

    def test_start_operation(self):
        """Test starting an operation."""
        operation_id = self.monitor.start_operation("test_op", input_size=1000)

        assert "test_op" in operation_id
        assert len(self.monitor.metrics) == 1

        metric = self.monitor.metrics[0]
        assert metric.operation == "test_op"
        assert metric.input_size == 1000
        assert metric.start_time > 0
        assert not hasattr(metric, "end_time")

    def test_end_operation(self):
        """Test ending an operation."""
        operation_id = self.monitor.start_operation("test_op")
        time.sleep(0.1)  # Small delay to ensure different timestamps

        self.monitor.end_operation(operation_id, success=True, output_size=500)

        metric = self.monitor.metrics[0]
        assert hasattr(metric, "end_time")
        assert metric.success is True
        assert metric.output_size == 500
        assert metric.duration > 0

    def test_get_performance_summary(self):
        """Test getting performance summary."""
        # Start and end a few operations
        op1 = self.monitor.start_operation("op1")
        time.sleep(0.1)
        self.monitor.end_operation(op1, success=True)

        op2 = self.monitor.start_operation("op2")
        time.sleep(0.1)
        self.monitor.end_operation(op2, success=False, error_message="Test error")

        summary = self.monitor.get_performance_summary()

        assert summary["total_operations"] == 2
        assert summary["successful_operations"] == 1
        assert summary["failed_operations"] == 1
        assert summary["success_rate"] == 0.5
        assert "operations_by_type" in summary

    def test_export_metrics(self):
        """Test exporting metrics to JSON file."""
        # Create some test metrics
        op_id = self.monitor.start_operation("test_op")
        self.monitor.end_operation(op_id, success=True)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_file = f.name

        try:
            self.monitor.export_metrics(temp_file)

            # Verify file was created and contains data
            assert os.path.exists(temp_file)
            with open(temp_file, "r") as f:
                data = json.load(f)
                assert len(data) == 1
                assert data[0]["operation"] == "test_op"
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class TestQualityAssessmentSystem:
    """Test the QualityAssessmentSystem class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = Mock(spec=Config)
        self.system = QualityAssessmentSystem(self.config)

    def test_system_initialization(self):
        """Test that the system initializes with all assessors."""
        assert hasattr(self.system, "content_assessor")
        assert hasattr(self.system, "structure_assessor")
        assert hasattr(self.system, "metadata_assessor")
        assert hasattr(self.system, "performance_monitor")

        assert isinstance(self.system.content_assessor, ContentCompletenessAssessor)
        assert isinstance(self.system.structure_assessor, StructureIntegrityAssessor)
        assert isinstance(self.system.metadata_assessor, MetadataAccuracyAssessor)
        assert isinstance(self.system.performance_monitor, PerformanceMonitor)

    def test_assess_document_quality_parsed_content(self):
        """Test comprehensive quality assessment of parsed content."""
        # Create mock parsed content
        content = Mock(spec=ParsedContent)
        content.text_content = "Sample text content for testing."
        content.structure = ["heading1", "paragraph1"]
        content.metadata = {"format": "pdf"}
        content.parsing_errors = []
        content.parser_used = "pymupdf"

        report = self.system.assess_document_quality(content)

        assert report.document_id is not None
        assert len(report.metrics) > 0
        assert len(report.recommendations) >= 0
        assert report.processing_time > 0
        assert "assessors_used" in report.metadata

    def test_generate_recommendations(self):
        """Test recommendation generation based on failed metrics."""
        # Create metrics with some failures
        metrics = [
            QualityMetric(name="text_completeness", score=0.3, threshold=0.7),
            QualityMetric(name="structure_completeness", score=0.8, threshold=0.7),
            QualityMetric(name="error_freedom", score=0.4, threshold=0.7),
        ]

        recommendations = self.system._generate_recommendations(metrics)

        assert len(recommendations) > 0
        assert any("parser" in rec.lower() for rec in recommendations)
        assert any("error" in rec.lower() for rec in recommendations)

    def test_get_quality_summary(self):
        """Test getting quality assessment system summary."""
        summary = self.system.get_quality_summary()

        assert "quality_assessors" in summary
        assert "performance_monitoring" in summary
        assert "total_metrics_tracked" in summary

        assessors = summary["quality_assessors"]
        assert "content_completeness" in assessors
        assert "structure_integrity" in assessors
        assert "metadata_accuracy" in assessors


def test_get_quality_assessment_system():
    """Test the convenience function for getting a quality assessment system."""
    system = get_quality_assessment_system()

    assert isinstance(system, QualityAssessmentSystem)
    assert hasattr(system, "content_assessor")
    assert hasattr(system, "structure_assessor")
    assert hasattr(system, "metadata_assessor")
    assert hasattr(system, "performance_monitor")
