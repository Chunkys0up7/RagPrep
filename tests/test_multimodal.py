"""
Tests for multimodal content processing
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.multimodal import (
    MultimodalProcessor,
    ImageProcessor,
    TableProcessor,
    ChartDetector,
    MathProcessor,
    get_multimodal_processor,
)
from src.config import Config


class TestImageProcessor:
    """Test image processing functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = Mock(spec=Config)
        self.mock_config.multimodal.image_processing = True
        self.mock_config.multimodal.ocr_engine = "tesseract"
        self.processor = ImageProcessor(self.mock_config)

    @patch('src.multimodal.cv2.imread')
    @patch('src.multimodal.pytesseract.image_to_string')
    def test_extract_text_from_image(self, mock_tesseract, mock_cv2):
        """Test OCR text extraction from images."""
        # Mock OpenCV
        mock_cv2.return_value = Mock()
        
        # Mock Tesseract
        mock_tesseract.return_value = "Extracted text from image"
        
        # Create a temporary image file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_image_path = f.name
        
        try:
            result = self.processor.extract_text(temp_image_path)
            
            assert result["success"] is True
            assert "Extracted text from image" in result["text"]
            assert result["confidence"] > 0
            
        finally:
            os.unlink(temp_image_path)

    def test_image_processing_disabled(self):
        """Test behavior when image processing is disabled."""
        self.mock_config.multimodal.image_processing = False
        processor = ImageProcessor(self.mock_config)
        
        result = processor.extract_text("test.png")
        assert result["success"] is False
        assert "disabled" in result["error_message"].lower()


class TestTableProcessor:
    """Test table processing functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = Mock(spec=Config)
        self.mock_config.multimodal.table_extraction = True
        self.mock_config.multimodal.table_parser = "camelot"
        self.processor = TableProcessor(self.mock_config)

    @patch('src.multimodal.camelot.read_pdf')
    def test_extract_tables_from_pdf(self, mock_camelot):
        """Test table extraction from PDF files."""
        # Mock Camelot
        mock_table = Mock()
        mock_table.df = [["Header1", "Header2"], ["Data1", "Data2"]]
        mock_camelot.return_value = [mock_table]
        
        result = self.processor.extract_tables("test.pdf")
        
        assert result["success"] is True
        assert len(result["tables"]) == 1
        assert result["tables"][0]["data"] == [["Header1", "Header2"], ["Data1", "Data2"]]

    def test_table_extraction_disabled(self):
        """Test behavior when table extraction is disabled."""
        self.mock_config.multimodal.table_extraction = False
        processor = TableProcessor(self.mock_config)
        
        result = processor.extract_tables("test.pdf")
        assert result["success"] is False
        assert "disabled" in result["error_message"].lower()


class TestChartDetector:
    """Test chart detection functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = Mock(spec=Config)
        self.mock_config.multimodal.chart_detection = True
        self.detector = ChartDetector(self.mock_config)

    @patch('src.multimodal.cv2.imread')
    def test_detect_charts_in_image(self, mock_cv2):
        """Test chart detection in images."""
        # Mock OpenCV
        mock_cv2.return_value = Mock()
        
        result = self.detector.detect_charts("test.png")
        
        assert result["success"] is True
        assert "charts" in result
        assert isinstance(result["charts"], list)

    def test_chart_detection_disabled(self):
        """Test behavior when chart detection is disabled."""
        self.mock_config.multimodal.chart_detection = False
        detector = ChartDetector(self.mock_config)
        
        result = detector.detect_charts("test.png")
        assert result["success"] is False
        assert "disabled" in result["error_message"].lower()


class TestMathProcessor:
    """Test mathematical content processing."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = Mock(spec=Config)
        self.mock_config.multimodal.math_processing = True
        self.processor = MathProcessor(self.mock_config)

    def test_extract_math_content(self):
        """Test mathematical content extraction."""
        content = "The equation x^2 + y^2 = z^2 represents the Pythagorean theorem."
        
        result = self.processor.extract_math_content(content)
        
        assert result["success"] is True
        assert "math_expressions" in result
        assert isinstance(result["math_expressions"], list)

    def test_math_processing_disabled(self):
        """Test behavior when math processing is disabled."""
        self.mock_config.multimodal.math_processing = False
        processor = MathProcessor(self.mock_config)
        
        result = processor.extract_math_content("test content")
        assert result["success"] is False
        assert "disabled" in result["error_message"].lower()


class TestMultimodalProcessor:
    """Test the main multimodal processor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = Mock(spec=Config)
        self.mock_config.multimodal.image_processing = True
        self.mock_config.multimodal.table_extraction = True
        self.mock_config.multimodal.chart_detection = True
        self.mock_config.multimodal.math_processing = True
        self.processor = MultimodalProcessor(self.mock_config)

    def test_processor_initialization(self):
        """Test multimodal processor initialization."""
        assert hasattr(self.processor, "image_processor")
        assert hasattr(self.processor, "table_extractor")
        assert hasattr(self.processor, "chart_detector")
        assert hasattr(self.processor, "math_processor")

    def test_process_multimodal_content(self):
        """Test processing of multimodal content."""
        content = {
            "text": "Sample text with math: x^2 + y^2 = z^2",
            "images": ["image1.png"],
            "tables": [],
            "charts": []
        }
        
        result = self.processor.process_content(content)
        
        assert result["success"] is True
        assert "processed_content" in result
        assert "extracted_features" in result


def test_get_multimodal_processor():
    """Test the factory function for creating multimodal processors."""
    processor = get_multimodal_processor()
    assert isinstance(processor, MultimodalProcessor)


if __name__ == "__main__":
    pytest.main([__file__])
