"""
Multimodal Content Processing Module

Handles advanced content types including images, tables, charts, and mathematical content
for enhanced RAG document preparation.
"""

import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class ImageContent:
    """Represents extracted image content."""
    
    image_id: str
    file_path: str
    content_type: str  # "ocr_text", "chart", "diagram", "photo"
    extracted_text: str
    confidence: float
    metadata: Dict[str, Any]
    bounding_box: Optional[Tuple[int, int, int, int]] = None


@dataclass
class TableContent:
    """Represents extracted table content."""
    
    table_id: str
    content: str
    structure: List[List[str]]
    headers: List[str]
    data_rows: List[List[str]]
    metadata: Dict[str, Any]
    quality_score: float


@dataclass
class MathContent:
    """Represents extracted mathematical content."""
    
    math_id: str
    content: str
    latex_form: str
    math_type: str  # "equation", "formula", "expression"
    complexity: float
    metadata: Dict[str, Any]


@dataclass
class MultimodalResult:
    """Result of multimodal content processing."""
    
    success: bool
    images: List[ImageContent]
    tables: List[TableContent]
    math_content: List[MathContent]
    processing_time: float
    errors: List[str]
    warnings: List[str]


class ImageProcessor(ABC):
    """Abstract base class for image processing."""
    
    @abstractmethod
    def process_image(self, image_path: str) -> ImageContent:
        """Process an image and extract content."""
        pass
    
    @abstractmethod
    def supports_format(self, format: str) -> bool:
        """Check if this processor supports the given image format."""
        pass


class OCRProcessor(ImageProcessor):
    """OCR-based image text extraction."""
    
    def __init__(self):
        self.ocr_engine = self._get_ocr_engine()
    
    def _get_ocr_engine(self):
        """Get the best available OCR engine."""
        try:
            import pytesseract
            return "tesseract"
        except ImportError:
            try:
                import easyocr
                return "easyocr"
            except ImportError:
                logger.warning("No OCR engine available")
                return None
    
    def process_image(self, image_path: str) -> ImageContent:
        """Extract text from image using OCR."""
        if not self.ocr_engine:
            raise RuntimeError("No OCR engine available")
        
        try:
            if self.ocr_engine == "tesseract":
                return self._process_with_tesseract(image_path)
            elif self.ocr_engine == "easyocr":
                return self._process_with_easyocr(image_path)
        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            raise
    
    def _process_with_tesseract(self, image_path: str) -> ImageContent:
        """Process image with Tesseract OCR."""
        import pytesseract
        
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        
        return ImageContent(
            image_id=f"img_{hash(image_path)}",
            file_path=image_path,
            content_type="ocr_text",
            extracted_text=text.strip(),
            confidence=0.8,  # Tesseract confidence
            metadata={"ocr_engine": "tesseract", "image_size": image.size}
        )
    
    def _process_with_easyocr(self, image_path: str) -> ImageContent:
        """Process image with EasyOCR."""
        import easyocr
        
        reader = easyocr.Reader(['en'])
        results = reader.readtext(image_path)
        
        text = " ".join([result[1] for result in results])
        confidence = np.mean([result[2] for result in results]) if results else 0.0
        
        return ImageContent(
            image_id=f"img_{hash(image_path)}",
            file_path=image_path,
            content_type="ocr_text",
            extracted_text=text.strip(),
            confidence=confidence,
            metadata={"ocr_engine": "easyocr", "detections": len(results)}
        )
    
    def supports_format(self, format: str) -> bool:
        """Check supported image formats."""
        supported = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'}
        return format.lower() in supported


class ChartDetector(ImageProcessor):
    """Detect and analyze charts and diagrams in images."""
    
    def __init__(self):
        self.chart_patterns = [
            r'chart|graph|plot|diagram|figure',
            r'bar|line|pie|scatter|histogram',
            r'axis|x-axis|y-axis|legend|title'
        ]
    
    def process_image(self, image_path: str) -> ImageContent:
        """Detect if image contains a chart and extract basic info."""
        try:
            image = Image.open(image_path)
            
            # Basic chart detection (simplified)
            is_chart = self._detect_chart_indicators(image)
            
            return ImageContent(
                image_id=f"chart_{hash(image_path)}",
                file_path=image_path,
                content_type="chart" if is_chart else "photo",
                extracted_text="Chart detected" if is_chart else "Image content",
                confidence=0.7 if is_chart else 0.3,
                metadata={
                    "image_size": image.size,
                    "mode": image.mode,
                    "is_chart": is_chart
                }
            )
        except Exception as e:
            logger.error(f"Chart detection failed: {e}")
            raise
    
    def _detect_chart_indicators(self, image: Image.Image) -> bool:
        """Detect chart indicators in image."""
        # Simplified chart detection
        # In production, this would use ML models
        return False  # Placeholder
    
    def supports_format(self, format: str) -> bool:
        """Check supported image formats."""
        supported = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
        return format.lower() in supported


class TableProcessor:
    """Process and extract table content."""
    
    def __init__(self):
        self.extractors = self._get_table_extractors()
    
    def _get_table_extractors(self) -> List[str]:
        """Get available table extraction methods."""
        extractors = []
        
        try:
            import camelot
            extractors.append("camelot")
        except ImportError:
            pass
        
        try:
            import tabula
            extractors.append("tabula")
        except ImportError:
            pass
        
        try:
            import pdfplumber
            extractors.append("pdfplumber")
        except ImportError:
            pass
        
        return extractors
    
    def extract_tables(self, document_path: str, content_type: str) -> List[TableContent]:
        """Extract tables from document."""
        if content_type == "pdf":
            return self._extract_from_pdf(document_path)
        elif content_type == "docx":
            return self._extract_from_docx(document_path)
        else:
            return []
    
    def _extract_from_pdf(self, pdf_path: str) -> List[TableContent]:
        """Extract tables from PDF."""
        tables = []
        
        if "pdfplumber" in self.extractors:
            tables.extend(self._extract_with_pdfplumber(pdf_path))
        
        if "camelot" in self.extractors:
            tables.extend(self._extract_with_camelot(pdf_path))
        
        return tables
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> List[TableContent]:
        """Extract tables using pdfplumber."""
        try:
            import pdfplumber
            
            tables = []
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    
                    for table_num, table in enumerate(page_tables):
                        if table and len(table) > 1:  # At least header + data
                            table_content = TableContent(
                                table_id=f"table_{page_num}_{table_num}",
                                content="\n".join([" | ".join(row) for row in table]),
                                structure=table,
                                headers=table[0] if table else [],
                                data_rows=table[1:] if len(table) > 1 else [],
                                metadata={
                                    "page": page_num + 1,
                                    "extractor": "pdfplumber",
                                    "rows": len(table),
                                    "columns": len(table[0]) if table else 0
                                },
                                quality_score=0.8
                            )
                            tables.append(table_content)
            
            return tables
        except Exception as e:
            logger.error(f"PDFPlumber table extraction failed: {e}")
            return []
    
    def _extract_with_camelot(self, pdf_path: str) -> List[TableContent]:
        """Extract tables using camelot-py."""
        try:
            import camelot
            
            tables = []
            table_list = camelot.read_pdf(pdf_path, pages='all')
            
            for table_num, table in enumerate(table_list):
                if table.df.shape[0] > 1:  # At least header + data
                    df = table.df
                    table_content = TableContent(
                        table_id=f"camelot_table_{table_num}",
                        content=df.to_string(index=False),
                        structure=df.values.tolist(),
                        headers=df.columns.tolist(),
                        data_rows=df.iloc[1:].values.tolist(),
                        metadata={
                            "extractor": "camelot",
                            "accuracy": table.accuracy,
                            "whitespace": table.whitespace,
                            "rows": df.shape[0],
                            "columns": df.shape[1]
                        },
                        quality_score=table.accuracy / 100.0
                    )
                    tables.append(table_content)
            
            return tables
        except Exception as e:
            logger.error(f"Camelot table extraction failed: {e}")
            return []
    
    def _extract_from_docx(self, docx_path: str) -> List[TableContent]:
        """Extract tables from DOCX."""
        try:
            from docx import Document
            
            tables = []
            doc = Document(docx_path)
            
            for table_num, table in enumerate(doc.tables):
                table_data = []
                for row in table.rows:
                    row_data = [cell.text.strip() for cell in row.cells]
                    table_data.append(row_data)
                
                if table_data and len(table_data) > 1:
                    table_content = TableContent(
                        table_id=f"docx_table_{table_num}",
                        content="\n".join([" | ".join(row) for row in table_data]),
                        structure=table_data,
                        headers=table_data[0] if table_data else [],
                        data_rows=table_data[1:] if len(table_data) > 1 else [],
                        metadata={
                            "extractor": "python-docx",
                            "rows": len(table_data),
                            "columns": len(table_data[0]) if table_data else 0
                        },
                        quality_score=0.9
                    )
                    tables.append(table_content)
            
            return tables
        except Exception as e:
            logger.error(f"DOCX table extraction failed: {e}")
            return []


class MathProcessor:
    """Process and extract mathematical content."""
    
    def __init__(self):
        self.math_patterns = [
            r'\$.*?\$',  # Inline math
            r'\\\[.*?\\\]',  # Display math
            r'\\begin\{.*?\}.*?\\end\{.*?\}',  # LaTeX environments
            r'[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*[^=]+',  # Equations
            r'[a-zA-Z_][a-zA-Z0-9_]*\s*[+\-*/]\s*[a-zA-Z0-9_]+',  # Expressions
        ]
    
    def extract_math_content(self, text: str) -> List[MathContent]:
        """Extract mathematical content from text."""
        math_content = []
        
        for pattern in self.math_patterns:
            matches = re.finditer(pattern, text, re.DOTALL)
            for match in matches:
                content = match.group(0)
                math_content.append(self._create_math_content(content, match.start()))
        
        return math_content
    
    def _create_math_content(self, content: str, position: int) -> MathContent:
        """Create MathContent object from extracted content."""
        # Determine math type
        if content.startswith('$') and content.endswith('$'):
            math_type = "inline_equation"
        elif content.startswith('\\[') and content.endswith('\\]'):
            math_type = "display_equation"
        elif '\\begin{' in content:
            math_type = "latex_environment"
        elif '=' in content:
            math_type = "equation"
        else:
            math_type = "expression"
        
        # Calculate complexity (simplified)
        complexity = self._calculate_complexity(content)
        
        return MathContent(
            math_id=f"math_{hash(content)}_{position}",
            content=content,
            latex_form=content,
            math_type=math_type,
            complexity=complexity,
            metadata={
                "position": position,
                "length": len(content),
                "has_functions": "\\" in content,
                "has_symbols": any(char in content for char in "αβγδεθλμπσφψω")
            }
        )
    
    def _calculate_complexity(self, content: str) -> float:
        """Calculate mathematical complexity score."""
        complexity = 0.0
        
        # Basic complexity factors
        if '\\' in content:  # LaTeX commands
            complexity += 0.3
        
        if any(char in content for char in "∫∑∏√∞"):
            complexity += 0.2
        
        if '=' in content:  # Equations
            complexity += 0.1
        
        if len(content) > 50:  # Long expressions
            complexity += 0.1
        
        return min(complexity, 1.0)


class MultimodalProcessor:
    """Main orchestrator for multimodal content processing."""
    
    def __init__(self, config=None):
        self.config = config
        self.image_processors = [
            OCRProcessor(),
            ChartDetector()
        ]
        self.table_processor = TableProcessor()
        self.math_processor = MathProcessor()
        self.logger = logging.getLogger(__name__)
    
    def process_content(self, document_path: str, content_type: str, 
                       parsed_content: Any) -> MultimodalResult:
        """Process multimodal content from document."""
        start_time = time.time()
        errors = []
        warnings = []
        
        try:
            # Process images
            images = self._process_images(document_path, content_type)
            
            # Process tables
            tables = self._process_tables(document_path, content_type)
            
            # Process math content
            math_content = self._process_math_content(parsed_content)
            
            processing_time = time.time() - start_time
            
            return MultimodalResult(
                success=True,
                images=images,
                tables=tables,
                math_content=math_content,
                processing_time=processing_time,
                errors=errors,
                warnings=warnings
            )
        
        except Exception as e:
            self.logger.error(f"Multimodal processing failed: {e}")
            errors.append(str(e))
            
            return MultimodalResult(
                success=False,
                images=[],
                tables=[],
                math_content=[],
                processing_time=time.time() - start_time,
                errors=errors,
                warnings=warnings
            )
    
    def _process_images(self, document_path: str, content_type: str) -> List[ImageContent]:
        """Process images in document."""
        images = []
        
        # For now, focus on embedded images in documents
        # In production, this would extract and process embedded images
        
        return images
    
    def _process_tables(self, document_path: str, content_type: str) -> List[TableContent]:
        """Process tables in document."""
        return self.table_processor.extract_tables(document_path, content_type)
    
    def _process_math_content(self, parsed_content: Any) -> List[MathContent]:
        """Process mathematical content."""
        if hasattr(parsed_content, 'text_content'):
            return self.math_processor.extract_math_content(parsed_content.text_content)
        return []


# Convenience function
def get_multimodal_processor(config=None) -> MultimodalProcessor:
    """Get a configured multimodal processor instance."""
    return MultimodalProcessor(config)
