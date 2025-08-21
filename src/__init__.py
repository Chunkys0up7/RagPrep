"""
RAG Document Processing Utility

A sophisticated document processing utility that transforms various document formats
into optimized structures for RAG (Retrieval-Augmented Generation) applications.
"""

__version__ = "0.1.0"
__author__ = "RAGPrep Team"
__description__ = "Intelligent document processing for RAG applications"

from .processor import DocumentProcessor
from .parsers import DocumentParser
from .chunkers import DocumentChunker
from .metadata_extractors import LLMMetadataExtractor
from .quality_assessor import QualityAssessor
from .vector_store import VectorStore

__all__ = [
    "DocumentProcessor",
    "DocumentParser", 
    "DocumentChunker",
    "LLMMetadataExtractor",
    "QualityAssessor",
    "VectorStore"
]
