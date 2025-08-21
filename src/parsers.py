"""
Document Parsers

This module contains parsers for different document formats including PDF, DOCX, HTML, etc.
"""

from typing import Dict, Any
from .config import Config


class DocumentParser:
    """Base class for document parsers."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def parse(self, document_path: str) -> Dict[str, Any]:
        """Parse a document and return structured content."""
        raise NotImplementedError("Subclasses must implement parse method")


# Placeholder implementations will be added in subsequent phases
