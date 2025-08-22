"""
MkDocs Exporter for RAG Document Processing Utility

This module handles converting processed documents to markdown format
and integrating them into MkDocs documentation structure.
"""

import json
import logging
import os
import re
import subprocess
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .chunkers import DocumentChunk
from .config import Config
from .parsers import ParsedContent

logger = logging.getLogger(__name__)


@dataclass
class MkDocsPage:
    """Represents a single MkDocs page."""
    
    title: str
    content: str
    file_path: str
    metadata: Dict[str, Any]
    parent_section: Optional[str] = None
    order: int = 0


@dataclass
class MkDocsSection:
    """Represents a MkDocs section with pages."""
    
    name: str
    title: str
    pages: List[MkDocsPage]
    order: int = 0


@dataclass
class MkDocsExportResult:
    """Result of MkDocs export operation."""
    
    success: bool
    pages_created: int
    sections_created: int
    output_directory: str
    mkdocs_config_path: str
    navigation_file_path: str
    site_built: bool = False
    site_directory: str = ""
    site_url: str = ""
    build_time: float = 0.0
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        """Initialize lists if not provided."""
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class MkDocsExporter:
    """Exports processed documents to MkDocs format."""
    
    def __init__(self, config: Config):
        """Initialize the MkDocs exporter."""
        self.config = config
        self.output_dir = Path(config.output.output_directory) / "mkdocs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.docs_dir = self.output_dir / "docs"
        self.docs_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def export_document(self, 
                       document_id: str,
                       chunks: List[DocumentChunk],
                       metadata: Dict[str, Any],
                       source_filename: str,
                       original_content: Optional[str] = None,
                       build_site: bool = True) -> MkDocsExportResult:
        """Export a processed document to MkDocs format."""
        
        try:
            self.logger.info(f"Exporting document {document_id} to MkDocs format")
            
            # Create document section
            section_name = self._generate_section_name(source_filename)
            section_title = self._generate_section_title(source_filename, metadata)
            
            # Convert chunks to pages
            pages = self._convert_chunks_to_pages(chunks, document_id, metadata)
            
            # Add original document page if content is provided
            if original_content:
                original_page = self._create_original_page(
                    document_id, metadata, original_content, source_filename
                )
                pages.insert(0, original_page)  # Insert at beginning
                self.logger.info(f"Added original document page for {document_id}")
            
            # Create section
            section = MkDocsSection(
                name=section_name,
                title=section_title,
                pages=pages,
                order=0
            )
            
            # Write pages to disk
            self._write_pages_to_disk(section)
            
            # Update navigation
            self._update_navigation(section)
            
            # Generate MkDocs configuration
            self._generate_mkdocs_config()
            
            # Build the site if requested
            site_built = False
            site_directory = ""
            site_url = ""
            build_time = 0.0
            build_warnings = []
            
            if build_site:
                build_result = self._build_mkdocs_site()
                site_built = build_result['success']
                site_directory = build_result['site_directory']
                site_url = build_result['site_url']
                build_time = build_result['build_time']
                if not build_result['success']:
                    build_warnings.extend(build_result['errors'])
            
            return MkDocsExportResult(
                success=True,
                pages_created=len(pages),
                sections_created=1,
                output_directory=str(self.output_dir),
                mkdocs_config_path=str(self.output_dir / "mkdocs.yml"),
                navigation_file_path=str(self.docs_dir / "_navigation.md"),
                site_built=site_built,
                site_directory=site_directory,
                site_url=site_url,
                build_time=build_time,
                errors=[],
                warnings=build_warnings
            )
            
        except Exception as e:
            error_msg = f"Failed to export document {document_id}: {str(e)}"
            self.logger.error(error_msg)
            
            return MkDocsExportResult(
                success=False,
                pages_created=0,
                sections_created=0,
                output_directory=str(self.output_dir),
                mkdocs_config_path="",
                navigation_file_path="",
                site_built=False,
                site_directory="",
                site_url="",
                build_time=0.0,
                errors=[error_msg],
                warnings=[]
            )
    
    def _convert_chunks_to_pages(self, 
                                chunks: List[DocumentChunk],
                                document_id: str,
                                metadata: Dict[str, Any]) -> List[MkDocsPage]:
        """Convert document chunks to MkDocs pages."""
        
        pages = []
        
        for i, chunk in enumerate(chunks):
            # Generate page title from chunk content
            title = self._extract_title_from_chunk(chunk.content, i)
            
            # Clean and format content
            content = self._format_chunk_content(chunk.content, chunk.metadata)
            
            # Generate file path
            file_path = f"{document_id}/chunk_{i:03d}.md"
            
            # Create page
            page = MkDocsPage(
                title=title,
                content=content,
                file_path=file_path,
                metadata={
                    "chunk_id": chunk.chunk_id,
                    "chunk_type": chunk.chunk_type,
                    "quality_score": chunk.quality_score,
                    "source_document": document_id,
                    **chunk.metadata
                },
                order=i
            )
            
            pages.append(page)
        
        return pages
    
    def _create_original_page(self, 
                             document_id: str,
                             metadata: Dict[str, Any],
                             original_content: str,
                             source_filename: str) -> MkDocsPage:
        """Create a page containing the original, unchunked document."""
        
        # Generate title for original document
        title = self._generate_section_title(source_filename, metadata)
        original_title = f"📄 {title} (Complete Document)"
        
        # Format the original content
        content = self._format_original_content(original_content, metadata, source_filename)
        
        # Generate file path for original document
        file_path = f"{document_id}/original_document.md"
        
        # Create page
        page = MkDocsPage(
            title=original_title,
            content=content,
            file_path=file_path,
            metadata={
                "document_type": "original_complete",
                "source_document": document_id,
                "source_filename": source_filename,
                "word_count": len(original_content.split()),
                "character_count": len(original_content),
                "chunk_count": 0,  # Original is not chunked
                **metadata
            },
            order=-1  # Negative order to ensure it appears first
        )
        
        return page
    
    def _format_original_content(self, 
                                content: str, 
                                metadata: Dict[str, Any],
                                source_filename: str) -> str:
        """Format original document content for markdown."""
        
        # Add comprehensive metadata header
        metadata_header = f"""---
title: "{self._generate_section_title(source_filename, metadata)} (Complete Document)"
document_type: "original_complete"
source_filename: "{source_filename}"
word_count: {len(content.split())}
character_count: {len(content)}
chunk_count: 0
created_at: {metadata.get('timestamp', 'unknown')}
quality_score: {metadata.get('quality_score', 0.0):.3f}
---

# 📄 {self._generate_section_title(source_filename, metadata)} (Complete Document)

**Source File:** `{source_filename}`  
**Document Type:** Complete, Unchunked Version  
**Word Count:** {len(content.split()):,}  
**Character Count:** {len(content):,}  
**Processing Date:** {metadata.get('timestamp', 'unknown')}  
**Quality Score:** {metadata.get('quality_score', 0.0):.3f}

---

"""
        
        # Clean and format content
        cleaned_content = content.strip()
        
        # Add footer with chunk information
        footer = f"""

---

## 📋 Document Summary

This is the complete, original document that was processed through the RAGPrep pipeline. The document has been:

- ✅ **Parsed** and converted to markdown format
- ✅ **Enhanced** with metadata extraction
- ✅ **Quality assessed** and scored
- ✅ **Preserved** in its original, unchunked form

**Note:** This document is also available in chunked form for easier navigation and analysis. Each chunk represents a semantic segment of the content that can be used independently in RAG applications.

---
*Generated by RAGPrep Document Processing Utility*
"""
        
        return metadata_header + cleaned_content + footer
    
    def _extract_title_from_chunk(self, content: str, chunk_index: int) -> str:
        """Extract a meaningful title from chunk content."""
        
        # Look for markdown headers
        header_match = re.search(r'^#{1,3}\s+(.+)$', content, re.MULTILINE)
        if header_match:
            return header_match.group(1).strip()
        
        # Look for first sentence
        sentences = re.split(r'[.!?]+', content.strip())
        if sentences and sentences[0].strip():
            first_sentence = sentences[0].strip()
            # Limit length
            if len(first_sentence) > 60:
                first_sentence = first_sentence[:57] + "..."
            return first_sentence
        
        # Fallback
        return f"Chunk {chunk_index + 1}"
    
    def _format_chunk_content(self, content: str, metadata: Dict[str, Any]) -> str:
        """Format chunk content for markdown."""
        
        # Add metadata header
        metadata_header = f"""---
title: "{self._extract_title_from_chunk(content, 0)}"
chunk_type: {metadata.get('chunk_type', 'unknown')}
quality_score: {metadata.get('quality_score', 0.0):.3f}
word_count: {metadata.get('word_count', 0)}
created_at: {metadata.get('timestamp', 'unknown')}
---

"""
        
        # Clean content
        cleaned_content = content.strip()
        
        return metadata_header + cleaned_content
    
    def _generate_section_name(self, filename: str) -> str:
        """Generate a section name from filename."""
        # Remove extension and convert to lowercase
        name = Path(filename).stem.lower()
        # Replace spaces and special chars with underscores
        name = re.sub(r'[^a-z0-9]+', '_', name)
        # Remove leading/trailing underscores
        name = name.strip('_')
        return name
    
    def _generate_section_title(self, filename: str, metadata: Dict[str, Any]) -> str:
        """Generate a section title from filename and metadata."""
        # Try to get title from metadata first
        if 'title' in metadata:
            return metadata['title']
        
        # Fall back to filename
        name = Path(filename).stem
        # Convert underscores and dashes to spaces
        name = re.sub(r'[_-]', ' ', name)
        # Title case
        return name.title()
    
    def _write_pages_to_disk(self, section: MkDocsSection):
        """Write section pages to disk."""
        
        section_dir = self.docs_dir / section.name
        section_dir.mkdir(exist_ok=True)
        
        for page in section.pages:
            # Handle original document page differently
            if page.metadata.get('document_type') == 'original_complete':
                page_path = section_dir / "original_document.md"
            else:
                page_path = section_dir / f"chunk_{page.order:03d}.md"
            
            with open(page_path, 'w', encoding='utf-8') as f:
                f.write(page.content)
            
            self.logger.debug(f"Written page: {page_path}")
    
    def _update_navigation(self, section: MkDocsSection):
        """Update the navigation structure."""
        
        nav_file = self.docs_dir / "_navigation.md"
        
        # Read existing navigation
        existing_nav = ""
        if nav_file.exists():
            with open(nav_file, 'r', encoding='utf-8') as f:
                existing_nav = f.read()
        
        # Add new section
        section_nav = f"""
## {section.title}

"""
        
        # Add original document first if it exists
        original_page = next((p for p in section.pages if p.metadata.get('document_type') == 'original_complete'), None)
        if original_page:
            section_nav += f"- [{original_page.title}]({section.name}/original_document.md)\n"
        
        # Add chunked pages
        chunk_pages = [p for p in section.pages if p.metadata.get('document_type') != 'original_complete']
        for page in chunk_pages:
            section_nav += f"- [{page.title}]({section.name}/chunk_{page.order:03d}.md)\n"
        
        # Append to existing navigation
        updated_nav = existing_nav + section_nav
        
        # Write back
        with open(nav_file, 'w', encoding='utf-8') as f:
            f.write(updated_nav)
    
    def _generate_mkdocs_config(self):
        """Generate MkDocs configuration file."""
        
        config_content = f"""# MkDocs Configuration for RAG Document Processing Utility

site_name: RAG Document Processing Utility
site_description: Processed documents and chunks for RAG applications
site_author: RAGPrep

theme:
  name: material
  features:
    - navigation.tabs
    - navigation.sections
    - navigation.expand
    - search.highlight
    - search.share

nav:
  - Home: index.md
  - Navigation: _navigation.md

plugins:
  - search
  - mkdocstrings:
      default_handler: python
      handlers:
        python:
          paths: [../src]

markdown_extensions:
  - admonition
  - codehilite
  - footnotes
  - meta
  - toc:
      permalink: true
  - pymdownx.arithmatex
  - pymdownx.betterem
  - pymdownx.caret
  - pymdownx.details
  - pymdownx.emoji
  - pymdownx.inlinehilite
  - pymdownx.keys
  - pymdownx.magiclink
  - pymdownx.mark
  - pymdownx.smartsymbols
  - pymdownx.snippets
  - pymdownx.superfences
  - pymdownx.tabbed
  - pymdownx.tasklist
  - pymdownx.tilde

extra_css:
  - stylesheets/extra.css

extra_javascript:
  - javascripts/extra.js

repo_name: RAGPrep
repo_url: https://github.com/Chunkys0up7/RagPrep
edit_uri: edit/main/docs/

copyright: Copyright &copy; 2024 RAGPrep Team
"""
        
        config_path = self.output_dir / "mkdocs.yml"
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        self.logger.info(f"Generated MkDocs config: {config_path}")
    
    def export_batch(self, 
                    documents: List[Dict[str, Any]], 
                    build_site: bool = True) -> MkDocsExportResult:
        """Export multiple documents to MkDocs format."""
        
        total_pages = 0
        total_sections = 0
        errors = []
        warnings = []
        
        for doc in documents:
            try:
                result = self.export_document(
                    document_id=doc['document_id'],
                    chunks=doc['chunks'],
                    metadata=doc['metadata'],
                    source_filename=doc.get('source_filename', 'unknown'),
                    original_content=doc.get('original_content'),
                    build_site=False  # Don't build for individual docs in batch
                )
                
                if result.success:
                    total_pages += result.pages_created
                    total_sections += result.sections_created
                else:
                    errors.extend(result.errors)
                    
            except Exception as e:
                error_msg = f"Failed to export document {doc.get('document_id', 'unknown')}: {str(e)}"
                errors.append(error_msg)
        
        # Generate index page
        self._generate_index_page()
        
        # Build the site if requested and no errors
        site_built = False
        site_directory = ""
        site_url = ""
        build_time = 0.0
        build_warnings = []
        
        if build_site and len(errors) == 0:
            build_result = self._build_mkdocs_site()
            site_built = build_result['success']
            site_directory = build_result['site_directory']
            site_url = build_result['site_url']
            build_time = build_result['build_time']
            if not build_result['success']:
                build_warnings.extend(build_result['errors'])
        
        all_warnings = warnings + build_warnings
        
        return MkDocsExportResult(
            success=len(errors) == 0,
            pages_created=total_pages,
            sections_created=total_sections,
            output_directory=str(self.output_dir),
            mkdocs_config_path=str(self.output_dir / "mkdocs.yml"),
            navigation_file_path=str(self.docs_dir / "_navigation.md"),
            site_built=site_built,
            site_directory=site_directory,
            site_url=site_url,
            build_time=build_time,
            errors=errors,
            warnings=all_warnings
        )
    
    def _generate_index_page(self):
        """Generate the main index page."""
        
        index_content = """# RAG Document Processing Utility

Welcome to the processed documents from the RAG Document Processing Utility.

## Overview

This MkDocs site contains all documents that have been processed through the RAGPrep pipeline. Each document has been:

- Parsed and chunked into semantic pieces
- Enhanced with metadata extraction
- Quality assessed and scored
- Converted to markdown format
- Organized for easy navigation

## Navigation

See the [Navigation](_navigation.md) page for a complete list of all processed documents and chunks.

## Usage

These markdown files can be:

1. **Viewed directly** in this MkDocs site
2. **Exported** to other documentation systems
3. **Ingested** into RAG applications
4. **Used** for content analysis and research

## Processing Information

Each chunk includes metadata about:
- Content quality scores
- Word and character counts
- Processing timestamps
- Chunking strategy used
- Source document information

---

*Generated automatically by RAGPrep Document Processing Utility*
"""
        
        index_path = self.docs_dir / "index.md"
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(index_content)
        
        self.logger.info(f"Generated index page: {index_path}")
    
    def _build_mkdocs_site(self) -> Dict[str, Any]:
        """Build the MkDocs site and return build information."""
        import time
        
        start_time = time.time()
        build_result = {
            'success': False,
            'site_directory': '',
            'site_url': '',
            'build_time': 0.0,
            'errors': []
        }
        
        try:
            # Check if mkdocs command is available
            mkdocs_cmd = self._find_mkdocs_command()
            if not mkdocs_cmd:
                build_result['errors'].append("MkDocs command not found")
                return build_result
            
            # Set site directory
            site_dir = self.output_dir / "site"
            build_result['site_directory'] = str(site_dir)
            
            # Build the site
            self.logger.info("Building MkDocs site...")
            
            if mkdocs_cmd.startswith('python'):
                cmd = ['python', '-m', 'mkdocs', 'build', '--clean', '--site-dir', str(site_dir)]
            else:
                cmd = [mkdocs_cmd, 'build', '--clean', '--site-dir', str(site_dir)]
            
            result = subprocess.run(
                cmd,
                cwd=str(self.output_dir),
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            build_time = time.time() - start_time
            build_result['build_time'] = build_time
            
            if result.returncode == 0:
                build_result['success'] = True
                build_result['site_url'] = f"file://{site_dir.absolute()}/index.html"
                self.logger.info(f"MkDocs site built successfully in {build_time:.2f}s")
                self.logger.info(f"Site available at: {build_result['site_url']}")
            else:
                error_msg = f"MkDocs build failed: {result.stderr}"
                build_result['errors'].append(error_msg)
                self.logger.error(error_msg)
                
        except subprocess.TimeoutExpired:
            error_msg = "MkDocs build timed out after 5 minutes"
            build_result['errors'].append(error_msg)
            self.logger.error(error_msg)
        except Exception as e:
            error_msg = f"Error building MkDocs site: {str(e)}"
            build_result['errors'].append(error_msg)
            self.logger.error(error_msg)
        
        return build_result
    
    def _find_mkdocs_command(self) -> Optional[str]:
        """Find the mkdocs command in the system."""
        # Try different possible mkdocs commands
        possible_commands = ['mkdocs', 'mkdocs.exe', 'python -m mkdocs']
        
        for cmd in possible_commands:
            try:
                if cmd.startswith('python'):
                    # For python -m mkdocs, test differently
                    result = subprocess.run(
                        ['python', '-m', 'mkdocs', '--version'],
                        capture_output=True,
                        timeout=10
                    )
                    if result.returncode == 0:
                        return 'python -m mkdocs'
                else:
                    # Test if command exists
                    result = subprocess.run(
                        [cmd, '--version'],
                        capture_output=True,
                        timeout=10
                    )
                    if result.returncode == 0:
                        return cmd
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                continue
        
        return None
    
    def serve_mkdocs_site(self, port: int = 8000, host: str = "127.0.0.1") -> Dict[str, Any]:
        """Serve the MkDocs site locally."""
        serve_result = {
            'success': False,
            'url': f"http://{host}:{port}",
            'process': None,
            'errors': []
        }
        
        try:
            # Check if mkdocs command is available
            mkdocs_cmd = self._find_mkdocs_command()
            if not mkdocs_cmd:
                serve_result['errors'].append("MkDocs command not found")
                return serve_result
            
            # Check if site exists
            if not (self.output_dir / "mkdocs.yml").exists():
                serve_result['errors'].append("MkDocs configuration not found. Build site first.")
                return serve_result
            
            self.logger.info(f"Starting MkDocs server at {serve_result['url']}")
            
            if mkdocs_cmd.startswith('python'):
                cmd = ['python', '-m', 'mkdocs', 'serve', '--dev-addr', f"{host}:{port}"]
            else:
                cmd = [mkdocs_cmd, 'serve', '--dev-addr', f"{host}:{port}"]
            
            # Start the server process
            process = subprocess.Popen(
                cmd,
                cwd=str(self.output_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            serve_result['success'] = True
            serve_result['process'] = process
            self.logger.info(f"MkDocs server started successfully at {serve_result['url']}")
            
        except Exception as e:
            error_msg = f"Error starting MkDocs server: {str(e)}"
            serve_result['errors'].append(error_msg)
            self.logger.error(error_msg)
        
        return serve_result


def get_mkdocs_exporter(config: Optional[Config] = None) -> MkDocsExporter:
    """Get a configured MkDocs exporter instance."""
    if config is None:
        from .config import get_config
        config = get_config()
    return MkDocsExporter(config)
