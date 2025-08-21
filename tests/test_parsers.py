"""
Tests for document parsers
"""

import os
import pytest

# Add src to path for imports
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import Config
from parsers import (
    CascadingDocumentParser,
    DocumentParser,
    DOCXParser,
    HTMLParser,
    ParsedContent,
    ParserResult,
    PDFParser,
    TextParser,
    get_document_parser,
)


class TestDocumentParser:
    """Test base document parser functionality."""

    def test_abstract_base_class(self):
        """Test that DocumentParser is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            DocumentParser(Mock(spec=Config))

    def test_get_metadata(self):
        """Test metadata extraction from file."""
        # Create a mock config
        mock_config = Mock(spec=Config)

        # Create a concrete parser class for testing
        class TestParser(DocumentParser):
            def can_parse(self, file_path: str) -> bool:
                return True

            def parse(self, file_path: str) -> ParserResult:
                return ParserResult(
                    success=True,
                    content=None,
                    error_message=None,
                    parser_name=self.parser_name,
                    processing_time=0.0,
                    metadata={},
                )

        parser = TestParser(mock_config)

        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test content")
            temp_file_path = f.name

        try:
            metadata = parser.get_metadata(temp_file_path)

            assert "filename" in metadata
            assert "file_extension" in metadata
            assert "file_size" in metadata
            assert "parser" in metadata

            assert metadata["filename"] == os.path.basename(temp_file_path)
            assert metadata["file_extension"] == ".txt"
            assert metadata["file_size"] > 0
            assert metadata["parser"] == "TestParser"

        finally:
            os.unlink(temp_file_path)


class TestTextParser:
    """Test text file parser."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = Mock(spec=Config)
        self.parser = TextParser(self.mock_config)

    def test_can_parse(self):
        """Test file type detection."""
        assert self.parser.can_parse("test.txt") is True
        assert self.parser.can_parse("test.md") is True
        assert self.parser.can_parse("test.csv") is True
        assert self.parser.can_parse("test.log") is True
        assert self.parser.can_parse("test.pdf") is False
        assert self.parser.can_parse("test.docx") is False

    def test_parse_text_file(self):
        """Test parsing a simple text file."""
        # Create a temporary text file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Line 1\nLine 2\n\nLine 4")
            temp_file_path = f.name

        try:
            result = self.parser.parse(temp_file_path)

            assert result.success is True
            assert result.content is not None
            assert result.error_message is None
            assert result.parser_name == "TextParser"
            assert result.processing_time > 0

            content = result.content
            assert "Line 1" in content.text_content
            assert "Line 2" in content.text_content
            assert "Line 4" in content.text_content

            assert content.structure["lines"] == 4
            assert content.structure["paragraphs"] == 3
            assert content.metadata["line_count"] == 4
            assert content.metadata["word_count"] == 6
            assert content.metadata["character_count"] == 20

        finally:
            os.unlink(temp_file_path)

    def test_parse_nonexistent_file(self):
        """Test parsing a file that doesn't exist."""
        result = self.parser.parse("nonexistent.txt")

        assert result.success is False
        assert result.content is None
        assert result.error_message is not None
        assert "Error parsing text file" in result.error_message

    def test_parse_unsupported_format(self):
        """Test parsing an unsupported file format."""
        result = self.parser.parse("test.pdf")

        assert result.success is False
        assert result.content is None
        assert result.error_message == "Cannot parse this file type"


class TestHTMLParser:
    """Test HTML parser."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = Mock(spec=Config)
        self.parser = HTMLParser(self.mock_config)

    def test_can_parse(self):
        """Test file type detection."""
        assert self.parser.can_parse("test.html") is True
        assert self.parser.can_parse("test.htm") is True
        assert self.parser.can_parse("test.txt") is False
        assert self.parser.can_parse("test.pdf") is False

    @patch("parsers.BeautifulSoup")
    def test_parse_html_file(self, mock_bs4):
        """Test parsing an HTML file."""
        # Mock BeautifulSoup
        mock_soup = Mock()
        mock_soup.title.string = "Test Title"
        mock_soup.get_text.return_value = "Test content"
        mock_soup.find_all.return_value = []
        mock_bs4.return_value = mock_soup

        # Create a temporary HTML file
        html_content = """
        <html>
            <head><title>Test Title</title></head>
            <body>
                <h1>Heading 1</h1>
                <p>Paragraph 1</p>
                <p>Paragraph 2</p>
            </body>
        </html>
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            f.write(html_content)
            temp_file_path = f.name

        try:
            result = self.parser.parse(temp_file_path)

            assert result.success is True
            assert result.content is not None
            assert result.error_message is None
            assert result.parser_name == "HTMLParser"

            content = result.content
            assert "Test content" in content.text_content
            assert content.metadata["title"] == "Test Title"

        finally:
            os.unlink(temp_file_path)

    def test_parse_without_beautifulsoup(self):
        """Test parser behavior when BeautifulSoup is not available."""
        # Temporarily remove BeautifulSoup availability
        original_bs4_available = self.parser.bs4_available
        self.parser.bs4_available = False

        try:
            result = self.parser.parse("test.html")
            assert result.success is False
            assert result.error_message == "Cannot parse this file type"
        finally:
            self.parser.bs4_available = original_bs4_available


class TestPDFParser:
    """Test PDF parser."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = Mock(spec=Config)
        self.parser = PDFParser(self.mock_config)

    def test_can_parse(self):
        """Test file type detection."""
        assert self.parser.can_parse("test.pdf") == self.parser.fitz_available
        assert self.parser.can_parse("test.txt") is False
        assert self.parser.can_parse("test.docx") is False

    @patch("parsers.fitz")
    def test_parse_pdf_file(self, mock_fitz):
        """Test parsing a PDF file."""
        # Mock PyMuPDF
        mock_doc = Mock()
        mock_page = Mock()
        mock_page.get_text.return_value = "Page content"
        mock_page.get_text_dict.return_value = {"blocks": []}
        mock_page.get_images.return_value = []
        mock_page.annots.return_value = []

        mock_doc.__len__.return_value = 1
        mock_doc.load_page.return_value = mock_page
        mock_doc.metadata = {"Title": "Test PDF"}
        mock_doc.close = Mock()

        mock_fitz.open.return_value = mock_doc

        # Create a temporary PDF file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdf", delete=False) as f:
            f.write("fake pdf content")
            temp_file_path = f.name

        try:
            result = self.parser.parse(temp_file_path)

            assert result.success is True
            assert result.content is not None
            assert result.error_message is None
            assert result.parser_name == "PDFParser"

            content = result.content
            assert "Page content" in content.text_content
            assert content.structure["pages"] == 1
            assert content.structure["total_pages"] == 1

        finally:
            os.unlink(temp_file_path)

    def test_parse_without_pymupdf(self):
        """Test parser behavior when PyMuPDF is not available."""
        # Temporarily remove PyMuPDF availability
        original_fitz_available = self.parser.fitz_available
        self.parser.fitz_available = False

        try:
            result = self.parser.parse("test.pdf")
            assert result.success is False
            assert result.error_message == "Cannot parse this file type"
        finally:
            self.parser.fitz_available = original_fitz_available


class TestDOCXParser:
    """Test DOCX parser."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = Mock(spec=Config)
        self.parser = DOCXParser(self.mock_config)

    def test_can_parse(self):
        """Test file type detection."""
        assert self.parser.can_parse("test.docx") == self.parser.docx_available
        assert self.parser.can_parse("test.doc") == self.parser.docx_available
        assert self.parser.can_parse("test.txt") is False
        assert self.parser.can_parse("test.pdf") is False

    @patch("parsers.Document")
    def test_parse_docx_file(self, mock_document):
        """Test parsing a DOCX file."""
        # Mock python-docx
        mock_doc = Mock()
        mock_para = Mock()
        mock_para.text = "Test paragraph"
        mock_para.style.name = "Normal"
        mock_para.runs = [Mock(text="Test paragraph")]

        mock_table = Mock()
        mock_row = Mock()
        mock_cell = Mock()
        mock_cell.text = "Cell content"
        mock_row.cells = [mock_cell]
        mock_table.rows = [mock_row]

        mock_props = Mock()
        mock_props.title = "Test Document"
        mock_props.author = "Test Author"

        mock_doc.paragraphs = [mock_para]
        mock_doc.tables = [mock_table]
        mock_doc.core_properties = mock_props

        mock_document.return_value = mock_doc

        # Create a temporary DOCX file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".docx", delete=False) as f:
            f.write("fake docx content")
            temp_file_path = f.name

        try:
            result = self.parser.parse(temp_file_path)

            assert result.success is True
            assert result.content is not None
            assert result.error_message is None
            assert result.parser_name == "DOCXParser"

            content = result.content
            assert "Test paragraph" in content.text_content
            assert content.structure["paragraphs"] == 1
            assert content.structure["tables"] == 1
            assert content.metadata["title"] == "Test Document"
            assert content.metadata["author"] == "Test Author"

        finally:
            os.unlink(temp_file_path)

    def test_parse_without_python_docx(self):
        """Test parser behavior when python-docx is not available."""
        # Temporarily remove python-docx availability
        original_docx_available = self.parser.docx_available
        self.parser.docx_available = False

        try:
            result = self.parser.parse("test.docx")
            assert result.success is False
            assert result.error_message == "Cannot parse this file type"
        finally:
            self.parser.docx_available = original_docx_available


class TestCascadingDocumentParser:
    """Test cascading document parser."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = Mock(spec=Config)
        self.parser = CascadingDocumentParser(self.mock_config)

    def test_initialization(self):
        """Test parser initialization."""
        assert len(self.parser.parsers) > 0
        assert all(hasattr(p, "parser_name") for p in self.parser.parsers)

    def test_get_available_parsers(self):
        """Test getting available parser names."""
        parser_names = self.parser.get_available_parsers()
        assert isinstance(parser_names, list)
        assert len(parser_names) > 0
        assert all(isinstance(name, str) for name in parser_names)

    def test_get_supported_formats(self):
        """Test getting supported file formats."""
        formats = self.parser.get_supported_formats()
        assert isinstance(formats, list)
        assert len(formats) > 0
        assert all(isinstance(fmt, str) for fmt in formats)

    def test_parse_nonexistent_file(self):
        """Test parsing a file that doesn't exist."""
        result = self.parser.parse_document("nonexistent.pdf")

        assert result.success is False
        assert result.content is None
        assert result.error_message is not None
        assert "File not found" in result.error_message

    def test_parse_unsupported_format(self):
        """Test parsing an unsupported file format."""
        # Create a temporary file with unsupported extension
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            f.write("test content")
            temp_file_path = f.name

        try:
            result = self.parser.parse_document(temp_file_path)

            assert result.success is False
            assert result.content is None
            assert result.error_message is not None
            assert "No suitable parser found" in result.error_message

        finally:
            os.unlink(temp_file_path)

    def test_parse_batch(self):
        """Test batch parsing."""
        # Create temporary files for testing
        temp_files = []
        try:
            for i in range(3):
                f = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
                f.write(f"Content {i}")
                f.close()
                temp_files.append(f.name)

            results = self.parser.parse_batch(temp_files)

            assert len(results) == 3
            assert all(isinstance(result, ParserResult) for result in results)

        finally:
            for temp_file in temp_files:
                os.unlink(temp_file)

    @patch("parsers.PDFParser.can_parse")
    @patch("parsers.PDFParser.parse")
    def test_cascading_strategy(self, mock_parse, mock_can_parse):
        """Test cascading parser strategy."""
        # Mock PDF parser to fail
        mock_can_parse.return_value = True
        mock_parse.return_value = ParserResult(
            success=False,
            content=None,
            error_message="PDF parsing failed",
            parser_name="PDFParser",
            processing_time=0.0,
            metadata={},
        )

        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdf", delete=False) as f:
            f.write("test content")
            temp_file_path = f.name

        try:
            result = self.parser.parse_document(temp_file_path)

            # Should fail since PDF parser failed and no other parser can handle PDF
            assert result.success is False
            assert "All parsers failed" in result.error_message

        finally:
            os.unlink(temp_file_path)


class TestParsedContent:
    """Test ParsedContent dataclass."""

    def test_parsed_content_creation(self):
        """Test creating ParsedContent instances."""
        content = ParsedContent(
            text_content="Test content",
            structured_content={"test": "data"},
            metadata={"key": "value"},
            tables=[],
            images=[],
            math_content=[],
            structure={"count": 1},
            parser_used="TestParser",
            parsing_errors=[],
            parsing_warnings=[],
        )

        assert content.text_content == "Test content"
        assert content.structured_content["test"] == "data"
        assert content.metadata["key"] == "value"
        assert content.parser_used == "TestParser"
        assert content.structure["count"] == 1


class TestParserResult:
    """Test ParserResult dataclass."""

    def test_parser_result_creation(self):
        """Test creating ParserResult instances."""
        result = ParserResult(
            success=True,
            content=Mock(spec=ParsedContent),
            error_message=None,
            parser_name="TestParser",
            processing_time=1.5,
            metadata={"key": "value"},
        )

        assert result.success is True
        assert result.content is not None
        assert result.error_message is None
        assert result.parser_name == "TestParser"
        assert result.processing_time == 1.5
        assert result.metadata["key"] == "value"


# Test convenience function
def test_get_document_parser():
    """Test getting document parser instance."""
    parser = get_document_parser()
    assert isinstance(parser, CascadingDocumentParser)
    assert parser.config is not None


if __name__ == "__main__":
    pytest.main([__file__])
