"""
Document Chunkers

This module implements intelligent document chunking with multiple strategies.
It provides semantic, structural, fixed-size, and hybrid chunking approaches.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import hashlib
import logging
import re
import time
from pathlib import Path

from .config import Config, get_config
from .parsers import ParsedContent

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a single document chunk."""

    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    chunk_type: str
    chunk_index: int
    parent_chunk_id: Optional[str] = None
    child_chunk_ids: List[str] = None
    relationships: List[str] = None
    quality_score: float = 0.0

    def __post_init__(self):
        """Initialize default values for lists."""
        if self.child_chunk_ids is None:
            self.child_chunk_ids = []
        if self.relationships is None:
            self.relationships = []


@dataclass
class ChunkingResult:
    """Result of document chunking operation."""

    success: bool
    chunks: List[DocumentChunk]
    chunking_strategy: str
    total_chunks: int
    processing_time: float
    metadata: Dict[str, Any]
    errors: List[str]
    warnings: List[str]


class DocumentChunker(ABC):
    """Abstract base class for document chunkers."""

    def __init__(self, config: Config):
        """Initialize chunker with configuration."""
        self.config = config
        self.chunking_config = config.get_chunking_config()
        self.chunker_name: str = self.__class__.__name__

    def chunk(self, parsed_content: ParsedContent) -> ChunkingResult:
        """Chunk the parsed document content (alias for chunk_document)."""
        return self.chunk_document(parsed_content)

    @abstractmethod
    def chunk_document(self, parsed_content: ParsedContent) -> ChunkingResult:
        """Chunk the parsed document content."""
        pass

    def _generate_chunk_id(self, content: str, index: int) -> str:
        """Generate a unique chunk ID."""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"chunk_{index:04d}_{content_hash}"

    def _extract_chunk_metadata(
        self, chunk_content: str, chunk_index: int, chunk_type: str
    ) -> Dict[str, Any]:
        """Extract metadata for a chunk."""
        return {
            "chunk_index": chunk_index,
            "chunk_type": chunk_type,
            "content_length": len(chunk_content),
            "word_count": len(chunk_content.split()),
            "sentence_count": len(re.split(r"[.!?]+", chunk_content)),
            "paragraph_count": len([p for p in chunk_content.split("\n") if p.strip()]),
            "chunker": self.chunker_name,
            "timestamp": time.time(),
        }

    def _assess_chunk_quality(self, chunk_content: str) -> float:
        """Assess the quality of a chunk."""
        if not chunk_content.strip():
            return 0.0

        # Basic quality metrics
        content_length = len(chunk_content.strip())
        word_count = len(chunk_content.split())
        sentence_count = len(re.split(r"[.!?]+", chunk_content))

        # Length-based score (prefer chunks within ideal range)
        ideal_length = self.chunking_config.max_chunk_size
        length_score = 1.0 - abs(content_length - ideal_length) / ideal_length
        length_score = max(0.0, min(1.0, length_score))

        # Completeness score (prefer complete sentences)
        if sentence_count > 0:
            last_sentence = chunk_content.strip().split(".")[-1]
            completeness_score = 1.0 if last_sentence.strip() else 0.8
        else:
            completeness_score = 0.6

        # Content richness score
        richness_score = min(1.0, word_count / 50.0)  # Prefer chunks with more words

        # Combined score
        quality_score = (
            length_score * 0.4 + completeness_score * 0.4 + richness_score * 0.2
        )

        return round(quality_score, 3)


class FixedSizeChunker(DocumentChunker):
    """Fixed-size chunking strategy."""

    def __init__(self, config: Config):
        """Initialize fixed-size chunker."""
        super().__init__(config)
        self.chunker_name = "FixedSizeChunker"

    def chunk_document(self, parsed_content: ParsedContent) -> ChunkingResult:
        """Chunk document using fixed-size strategy."""
        start_time = time.time()
        errors = []
        warnings = []

        try:
            text_content = parsed_content.text_content
            if not text_content.strip():
                return ChunkingResult(
                    success=False,
                    chunks=[],
                    chunking_strategy="fixed_size",
                    total_chunks=0,
                    processing_time=time.time() - start_time,
                    metadata={},
                    errors=["Empty document content"],
                    warnings=[],
                )

            # Get chunking parameters
            max_size = self.chunking_config.max_chunk_size
            min_size = self.chunking_config.min_chunk_size
            overlap_size = self.chunking_config.overlap_size

            chunks = []
            chunk_index = 0
            start_pos = 0

            while start_pos < len(text_content):
                # Determine chunk end position
                end_pos = start_pos + max_size

                # If this is not the last chunk, try to break at sentence boundary
                if end_pos < len(text_content):
                    # Look for sentence boundary within overlap region
                    overlap_start = max(start_pos, end_pos - overlap_size)
                    sentence_boundary = self._find_sentence_boundary(
                        text_content, overlap_start, end_pos
                    )
                    if sentence_boundary > start_pos:
                        end_pos = sentence_boundary

                # Extract chunk content
                chunk_content = text_content[start_pos:end_pos].strip()

                # Ensure minimum chunk size
                if len(chunk_content) < min_size and start_pos + min_size < len(
                    text_content
                ):
                    end_pos = start_pos + min_size
                    chunk_content = text_content[start_pos:end_pos].strip()

                if chunk_content:
                    # Generate chunk ID and metadata
                    chunk_id = self._generate_chunk_id(chunk_content, chunk_index)
                    metadata = self._extract_chunk_metadata(
                        chunk_content, chunk_index, "fixed_size"
                    )
                    metadata.update(
                        {
                            "start_position": start_pos,
                            "end_position": end_pos,
                            "overlap_size": overlap_size,
                        }
                    )

                    # Assess chunk quality
                    quality_score = self._assess_chunk_quality(chunk_content)

                    # Create chunk
                    chunk = DocumentChunk(
                        chunk_id=chunk_id,
                        content=chunk_content,
                        metadata=metadata,
                        chunk_type="fixed_size",
                        chunk_index=chunk_index,
                        quality_score=quality_score,
                    )

                    chunks.append(chunk)
                    chunk_index += 1

                # Move to next chunk position
                start_pos = end_pos

                # Add overlap for next chunk
                if start_pos < len(text_content):
                    start_pos = max(0, start_pos - overlap_size)

            # Add warnings if chunks are too small or too large
            for chunk in chunks:
                if len(chunk.content) < min_size:
                    warnings.append(
                        f"Chunk {chunk.chunk_id} is below minimum size ({len(chunk.content)} < {min_size})"
                    )
                elif len(chunk.content) > max_size * 1.2:  # Allow 20% tolerance
                    warnings.append(
                        f"Chunk {chunk.chunk_id} exceeds maximum size ({len(chunk.content)} > {max_size})"
                    )

            return ChunkingResult(
                success=True,
                chunks=chunks,
                chunking_strategy="fixed_size",
                total_chunks=len(chunks),
                processing_time=time.time() - start_time,
                metadata={
                    "max_chunk_size": max_size,
                    "min_chunk_size": min_size,
                    "overlap_size": overlap_size,
                    "total_content_length": len(text_content),
                },
                errors=errors,
                warnings=warnings,
            )

        except Exception as e:
            error_msg = f"Error in fixed-size chunking: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)

            return ChunkingResult(
                success=False,
                chunks=[],
                chunking_strategy="fixed_size",
                total_chunks=0,
                processing_time=time.time() - start_time,
                metadata={},
                errors=errors,
                warnings=warnings,
            )

    def _find_sentence_boundary(self, text: str, start: int, end: int) -> int:
        """Find the best sentence boundary within a range."""
        # Look for sentence endings in reverse order
        for i in range(end, start, -1):
            if i < len(text) and text[i] in ".!?":
                # Check if it's a real sentence ending (not an abbreviation)
                if i + 1 < len(text) and text[i + 1].isspace():
                    return i + 1

        # If no sentence boundary found, return the original end position
        return end


class StructuralChunker(DocumentChunker):
    """Structural chunking strategy based on document hierarchy."""

    def __init__(self, config: Config):
        """Initialize structural chunker."""
        super().__init__(config)
        self.chunker_name = "StructuralChunker"

    def chunk_document(self, parsed_content: ParsedContent) -> ChunkingResult:
        """Chunk document using structural strategy."""
        start_time = time.time()
        errors = []
        warnings = []

        try:
            text_content = parsed_content.text_content
            structured_content = parsed_content.structured_content

            if not text_content.strip():
                return ChunkingResult(
                    success=False,
                    chunks=[],
                    chunking_strategy="structural",
                    total_chunks=0,
                    processing_time=time.time() - start_time,
                    metadata={},
                    errors=["Empty document content"],
                    warnings=[],
                )

            chunks = []
            chunk_index = 0

            # Extract structural elements based on document type
            if "pages" in structured_content:
                # PDF-like structure
                chunks.extend(self._chunk_by_pages(structured_content, chunk_index))
                chunk_index = len(chunks)

            if "paragraphs" in structured_content:
                # DOCX-like structure
                chunks.extend(
                    self._chunk_by_paragraphs(structured_content, chunk_index)
                )
                chunk_index = len(chunks)

            if "headings" in structured_content:
                # HTML-like structure
                chunks.extend(self._chunk_by_headings(structured_content, chunk_index))
                chunk_index = len(chunks)

            # If no structural elements found, fall back to fixed-size chunking
            if not chunks:
                logger.warning(
                    "No structural elements found, falling back to fixed-size chunking"
                )
                fixed_chunker = FixedSizeChunker(self.config)
                return fixed_chunker.chunk_document(parsed_content)

            # Add warnings for chunks that are too small or too large
            min_size = self.chunking_config.min_chunk_size
            max_size = self.chunking_config.max_chunk_size

            for chunk in chunks:
                if len(chunk.content) < min_size:
                    warnings.append(
                        f"Chunk {chunk.chunk_id} is below minimum size ({len(chunk.content)} < {min_size})"
                    )
                elif (
                    len(chunk.content) > max_size * 1.5
                ):  # Allow 50% tolerance for structural chunks
                    warnings.append(
                        f"Chunk {chunk.chunk_id} exceeds maximum size ({len(chunk.content)} > {max_size})"
                    )

            return ChunkingResult(
                success=True,
                chunks=chunks,
                chunking_strategy="structural",
                total_chunks=len(chunks),
                processing_time=time.time() - start_time,
                metadata={
                    "structural_elements": list(structured_content.keys()),
                    "total_content_length": len(text_content),
                },
                errors=errors,
                warnings=warnings,
            )

        except Exception as e:
            error_msg = f"Error in structural chunking: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)

            return ChunkingResult(
                success=False,
                chunks=[],
                chunking_strategy="structural",
                total_chunks=0,
                processing_time=time.time() - start_time,
                metadata={},
                errors=errors,
                warnings=warnings,
            )

    def _chunk_by_pages(
        self, structured_content: Dict[str, Any], start_index: int
    ) -> List[DocumentChunk]:
        """Chunk document by pages."""
        chunks = []
        pages = structured_content.get("pages", [])

        for i, page in enumerate(pages):
            if isinstance(page, dict) and "text_blocks" in page:
                # Extract text from page blocks
                page_text = ""
                for block in page["text_blocks"]:
                    if "lines" in block:
                        for line in block["lines"]:
                            if "spans" in line:
                                for span in line["spans"]:
                                    page_text += span.get("text", "") + " "

                page_text = page_text.strip()
                if page_text:
                    chunk_id = self._generate_chunk_id(page_text, start_index + i)
                    metadata = self._extract_chunk_metadata(
                        page_text, start_index + i, "page"
                    )
                    metadata.update({"page_number": i + 1, "page_index": i})

                    quality_score = self._assess_chunk_quality(page_text)

                    chunk = DocumentChunk(
                        chunk_id=chunk_id,
                        content=page_text,
                        metadata=metadata,
                        chunk_type="page",
                        chunk_index=start_index + i,
                        quality_score=quality_score,
                    )

                    chunks.append(chunk)

        return chunks

    def _chunk_by_paragraphs(
        self, structured_content: Dict[str, Any], start_index: int
    ) -> List[DocumentChunk]:
        """Chunk document by paragraphs."""
        chunks = []
        paragraphs = structured_content.get("paragraphs", [])

        for i, para in enumerate(paragraphs):
            if isinstance(para, dict) and "text" in para:
                para_text = para["text"].strip()
                if para_text:
                    chunk_id = self._generate_chunk_id(para_text, start_index + i)
                    metadata = self._extract_chunk_metadata(
                        para_text, start_index + i, "paragraph"
                    )
                    metadata.update(
                        {"paragraph_index": i, "style": para.get("style", "Normal")}
                    )

                    quality_score = self._assess_chunk_quality(para_text)

                    chunk = DocumentChunk(
                        chunk_id=chunk_id,
                        content=para_text,
                        metadata=metadata,
                        chunk_type="paragraph",
                        chunk_index=start_index + i,
                        quality_score=quality_score,
                    )

                    chunks.append(chunk)

        return chunks

    def _chunk_by_headings(
        self, structured_content: Dict[str, Any], start_index: int
    ) -> List[DocumentChunk]:
        """Chunk content based on heading structure."""
        chunks = []
        headings = structured_content.get("headings", [])

        current_section = []
        current_heading = None

        for i, heading in enumerate(headings):
            if isinstance(heading, dict) and "text" in heading:
                # If we have accumulated content, create a chunk
                if current_section and current_heading:
                    section_text = "\n".join(current_section).strip()
                    if section_text:
                        chunk_id = self._generate_chunk_id(
                            section_text, start_index + len(chunks)
                        )
                        metadata = self._extract_chunk_metadata(
                            section_text, start_index + len(chunks), "section"
                        )
                        metadata.update(
                            {
                                "heading_text": current_heading["text"],
                                "heading_level": current_heading["level"],
                            }
                        )

                        quality_score = self._assess_chunk_quality(section_text)

                        chunk = DocumentChunk(
                            chunk_id=chunk_id,
                            content=section_text,
                            metadata=metadata,
                            chunk_type="section",
                            chunk_index=start_index + len(chunks),
                            quality_score=quality_score,
                        )

                        chunks.append(chunk)

                # Start new section
                current_heading = heading
                current_section = [
                    heading["text"]
                ]  # Extract text content from heading dict
            else:
                # Add content to current section (assuming it's already text or convertible)
                if current_section:
                    # If heading is not a dict, treat it as direct content
                    content_text = (
                        heading["text"]
                        if isinstance(heading, dict) and "text" in heading
                        else str(heading)
                    )
                    current_section.append(content_text)

        # Create chunk for the last section
        if current_section and current_heading:
            section_text = "\n".join(current_section).strip()
            if section_text:
                chunk_id = self._generate_chunk_id(
                    section_text, start_index + len(chunks)
                )
                metadata = self._extract_chunk_metadata(
                    section_text, start_index + len(chunks), "section"
                )
                metadata.update(
                    {
                        "heading_text": current_heading["text"],
                        "heading_level": current_heading["level"],
                    }
                )

                quality_score = self._assess_chunk_quality(section_text)

                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    content=section_text,
                    metadata=metadata,
                    chunk_type="section",
                    chunk_index=start_index + len(chunks),
                    quality_score=quality_score,
                )

                chunks.append(chunk)

        return chunks


class SemanticChunker(DocumentChunker):
    """Semantic chunking strategy using content similarity."""

    def __init__(self, config: Config):
        """Initialize semantic chunker."""
        super().__init__(config)
        self.chunker_name = "SemanticChunker"

    def chunk_document(self, parsed_content: ParsedContent) -> ChunkingResult:
        """Chunk document using semantic strategy."""
        start_time = time.time()
        errors = []
        warnings = []

        try:
            text_content = parsed_content.text_content
            if not text_content.strip():
                return ChunkingResult(
                    success=False,
                    chunks=[],
                    chunking_strategy="semantic",
                    total_chunks=0,
                    processing_time=time.time() - start_time,
                    metadata={},
                    errors=["Empty document content"],
                    warnings=[],
                )

            # For now, implement a simplified semantic chunking
            # In production, this would use embeddings and similarity detection
            chunks = self._chunk_by_semantic_boundaries(text_content)

            return ChunkingResult(
                success=True,
                chunks=chunks,
                chunking_strategy="semantic",
                total_chunks=len(chunks),
                processing_time=time.time() - start_time,
                metadata={
                    "total_content_length": len(text_content),
                },
                errors=errors,
                warnings=warnings,
            )

        except Exception as e:
            error_msg = f"Error in semantic chunking: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)

            return ChunkingResult(
                success=False,
                chunks=[],
                chunking_strategy="semantic",
                total_chunks=0,
                processing_time=time.time() - start_time,
                metadata={},
                errors=errors,
                warnings=warnings,
            )

    def _chunk_by_semantic_boundaries(self, text: str) -> List[DocumentChunk]:
        """Chunk text by semantic boundaries (simplified implementation)."""
        chunks = []

        # Split by paragraphs first
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

        current_chunk = []
        chunk_index = 0

        for para in paragraphs:
            current_chunk.append(para)

            # Check if current chunk is getting too large
            current_text = "\n".join(current_chunk)
            if len(current_text) > self.chunking_config.max_chunk_size:
                # Remove the last paragraph and create a chunk
                if len(current_chunk) > 1:
                    current_chunk.pop()
                    chunk_text = "\n".join(current_chunk).strip()

                    if chunk_text:
                        chunk_id = self._generate_chunk_id(chunk_text, chunk_index)
                        metadata = self._extract_chunk_metadata(
                            chunk_text, chunk_index, "semantic"
                        )
                        quality_score = self._assess_chunk_quality(chunk_text)

                        chunk = DocumentChunk(
                            chunk_id=chunk_id,
                            content=chunk_text,
                            metadata=metadata,
                            chunk_type="semantic",
                            chunk_index=chunk_index,
                            quality_score=quality_score,
                        )

                        chunks.append(chunk)
                        chunk_index += 1

                # Start new chunk with the current paragraph
                current_chunk = [para]

        # Create chunk for remaining content
        if current_chunk:
            chunk_text = "\n".join(current_chunk).strip()
            if chunk_text:
                chunk_id = self._generate_chunk_id(chunk_text, chunk_index)
                metadata = self._extract_chunk_metadata(
                    chunk_text, chunk_index, "semantic"
                )
                quality_score = self._assess_chunk_quality(chunk_text)

                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    content=chunk_text,
                    metadata=metadata,
                    chunk_type="semantic",
                    chunk_index=chunk_index,
                    quality_score=quality_score,
                )

                chunks.append(chunk)

        return chunks


class HybridChunker(DocumentChunker):
    """Hybrid chunking strategy combining multiple approaches."""

    def __init__(self, config: Config):
        """Initialize hybrid chunker."""
        super().__init__(config)
        self.chunker_name = "HybridChunker"

        # Initialize sub-chunkers
        self.structural_chunker = StructuralChunker(config)
        self.semantic_chunker = SemanticChunker(config)
        self.fixed_chunker = FixedSizeChunker(config)

    def chunk(self, parsed_content: ParsedContent) -> ChunkingResult:
        """Chunk document using hybrid strategy (alias for chunk_document)."""
        return self.chunk_document(parsed_content)

    def chunk_document(self, parsed_content: ParsedContent) -> ChunkingResult:
        """Chunk document using hybrid strategy."""
        start_time = time.time()
        errors = []
        warnings = []

        try:
            # Try structural chunking first
            structural_result = self.structural_chunker.chunk_document(parsed_content)

            if structural_result.success and len(structural_result.chunks) > 0:
                # Use structural chunks if they're good quality
                avg_quality = sum(
                    chunk.quality_score for chunk in structural_result.chunks
                ) / len(structural_result.chunks)

                if avg_quality >= 0.7:  # Good quality threshold
                    logger.info("Using structural chunking results")
                    return ChunkingResult(
                        success=True,
                        chunks=structural_result.chunks,
                        chunking_strategy="hybrid_structural",
                        total_chunks=len(structural_result.chunks),
                        processing_time=time.time() - start_time,
                        metadata=structural_result.metadata,
                        errors=errors,
                        warnings=warnings + structural_result.warnings,
                    )
                else:
                    logger.info(
                        "Structural chunks quality too low, trying semantic chunking"
                    )
                    warnings.append(
                        "Structural chunking quality below threshold, falling back to semantic"
                    )

            # Try semantic chunking
            semantic_result = self.semantic_chunker.chunk_document(parsed_content)

            if semantic_result.success and len(semantic_result.chunks) > 0:
                # Use semantic chunks if they're good quality
                avg_quality = sum(
                    chunk.quality_score for chunk in semantic_result.chunks
                ) / len(semantic_result.chunks)

                if avg_quality >= 0.6:  # Lower threshold for semantic chunks
                    logger.info("Using semantic chunking results")
                    return ChunkingResult(
                        success=True,
                        chunks=semantic_result.chunks,
                        chunking_strategy="hybrid_semantic",
                        total_chunks=len(semantic_result.chunks),
                        processing_time=time.time() - start_time,
                        metadata=semantic_result.metadata,
                        errors=errors,
                        warnings=warnings + semantic_result.warnings,
                    )
                else:
                    logger.info(
                        "Semantic chunks quality too low, using fixed-size chunking"
                    )
                    warnings.append(
                        "Semantic chunking quality below threshold, falling back to fixed-size"
                    )

            # Fall back to fixed-size chunking
            logger.info("Using fixed-size chunking as fallback")
            fixed_result = self.fixed_chunker.chunk_document(parsed_content)

            if fixed_result.success:
                return ChunkingResult(
                    success=True,
                    chunks=fixed_result.chunks,
                    chunking_strategy="hybrid_fixed",
                    total_chunks=len(fixed_result.chunks),
                    processing_time=time.time() - start_time,
                    metadata=fixed_result.metadata,
                    errors=errors + fixed_result.errors,
                    warnings=warnings + fixed_result.warnings,
                )
            else:
                errors.extend(fixed_result.errors)
                return ChunkingResult(
                    success=False,
                    chunks=[],
                    chunking_strategy="hybrid",
                    total_chunks=0,
                    processing_time=time.time() - start_time,
                    metadata={},
                    errors=errors,
                    warnings=warnings,
                )

        except Exception as e:
            error_msg = f"Error in hybrid chunking: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)

            return ChunkingResult(
                success=False,
                chunks=[],
                chunking_strategy="hybrid",
                total_chunks=0,
                processing_time=time.time() - start_time,
                metadata={},
                errors=errors,
                warnings=warnings,
            )


class DocumentChunkerFactory:
    """Factory for creating document chunkers."""

    @staticmethod
    def create_chunker(strategy: str, config: Config) -> DocumentChunker:
        """Create a chunker based on the specified strategy."""
        strategy = strategy.lower()

        if strategy == "fixed":
            return FixedSizeChunker(config)
        elif strategy == "structural":
            return StructuralChunker(config)
        elif strategy == "semantic":
            return SemanticChunker(config)
        elif strategy == "hybrid":
            return HybridChunker(config)
        else:
            logger.warning(
                f"Unknown chunking strategy '{strategy}', falling back to hybrid"
            )
            return HybridChunker(config)


# Convenience function to get chunker instance
def get_document_chunker(
    strategy: str = "hybrid", config: Optional[Config] = None
) -> DocumentChunker:
    """Get a configured document chunker instance."""
    if config is None:
        config = get_config()
    return DocumentChunkerFactory.create_chunker(strategy, config)
