#!/usr/bin/env python3
"""
Test script for original document functionality in MkDocs export

This script tests that the MkDocs exporter now creates both:
1. An original, unchunked version of each document
2. The chunked versions as before
"""

import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.processor import DocumentProcessor
from src.config import Config
from src.mkdocs_exporter import get_mkdocs_exporter


def test_original_document_export():
    """Test that original documents are exported alongside chunks."""
    
    print("🧪 Testing Original Document Export Functionality")
    print("=" * 60)
    
    # Initialize configuration and processor
    print("📋 Initializing components...")
    config = Config()
    processor = DocumentProcessor(config)
    
    # Check if we have a test document
    test_doc_path = "documents/test_document.txt"
    if not os.path.exists(test_doc_path):
        print(f"❌ Test document not found: {test_doc_path}")
        print("Please ensure you have a test document in the documents/ folder")
        return
    
    print(f"📄 Found test document: {test_doc_path}")
    
    # Process document with MkDocs export
    print("\n🔄 Processing document with MkDocs export...")
    start_time = time.time()
    
    try:
        result = processor.process_document_with_mkdocs(
            document_path=test_doc_path,
            export_mkdocs=True,
            build_site=True
        )
        
        processing_time = time.time() - start_time
        
        if result.success:
            print(f"✅ Document processed successfully in {processing_time:.2f} seconds")
            print(f"📊 Generated {len(result.chunks)} chunks")
            print(f"🎯 Quality score: {result.quality_score:.3f}")
            
            # Check if original content is preserved
            if result.original_content:
                print(f"📄 Original content preserved: {len(result.original_content)} characters")
                print(f"📝 Word count: {len(result.original_content.split())}")
            else:
                print("⚠️  No original content found in result")
            
            # Check MkDocs export status
            if 'mkdocs_export' in result.metadata:
                mkdocs_info = result.metadata['mkdocs_export']
                if mkdocs_info.get('success'):
                    print(f"📚 MkDocs export successful!")
                    print(f"   📄 Pages created: {mkdocs_info.get('pages_created', 0)}")
                    print(f"   📁 Output directory: {mkdocs_info.get('output_directory', 'N/A')}")
                    
                    # Show site building information
                    if mkdocs_info.get('site_built'):
                        print(f"   🌐 Static site built successfully in {mkdocs_info.get('build_time', 0):.2f}s")
                        print(f"   📁 Site directory: {mkdocs_info.get('site_directory', 'N/A')}")
                        print(f"   🔗 Site URL: {mkdocs_info.get('site_url', 'N/A')}")
                    else:
                        print(f"   ⚠️  Static site: Not built")
                    
                    # Verify that both original and chunked files were created
                    verify_mkdocs_output(mkdocs_info.get('output_directory', ''))
                else:
                    print(f"❌ MkDocs export failed: {mkdocs_info.get('errors', [])}")
            else:
                print("⚠️  No MkDocs export information found")
            
        else:
            print(f"❌ Document processing failed: {result.error_message}")
            return
            
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n🎉 Test completed!")


def verify_mkdocs_output(output_directory: str):
    """Verify that both original and chunked files were created."""
    
    if not output_directory:
        print("   ⚠️  No output directory specified")
        return
    
    print("\n🔍 Verifying MkDocs output structure...")
    
    docs_dir = Path(output_directory) / "docs"
    if not docs_dir.exists():
        print("   ❌ Docs directory not found")
        return
    
    # Find document directories
    doc_dirs = [d for d in docs_dir.iterdir() if d.is_dir() and d.name != "tmp"]
    
    for doc_dir in doc_dirs:
        print(f"   📁 Document directory: {doc_dir.name}")
        
        # Check for original document
        original_file = doc_dir / "original_document.md"
        if original_file.exists():
            print(f"      ✅ Original document: {original_file.name}")
            
            # Check content
            with open(original_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if "Complete Document" in content:
                    print(f"         📝 Contains 'Complete Document' marker")
                if "Document Type: Complete, Unchunked Version" in content:
                    print(f"         🏷️  Properly labeled as complete document")
        else:
            print(f"      ❌ Original document not found")
        
        # Check for chunked files
        chunk_files = list(doc_dir.glob("chunk_*.md"))
        if chunk_files:
            print(f"      📄 Chunked files: {len(chunk_files)} found")
            for chunk_file in chunk_files:
                print(f"         📄 {chunk_file.name}")
        else:
            print(f"      ⚠️  No chunked files found")
        
        # Check navigation
        nav_file = docs_dir / "_navigation.md"
        if nav_file.exists():
            with open(nav_file, 'r', encoding='utf-8') as f:
                nav_content = f.read()
                if "original_document.md" in nav_content:
                    print(f"      🧭 Original document included in navigation")
                else:
                    print(f"      ⚠️  Original document not in navigation")


def test_mkdocs_exporter_directly():
    """Test the MkDocs exporter directly with sample data."""
    
    print("\n🔧 Testing MkDocs Exporter Directly...")
    
    try:
        config = Config()
        exporter = get_mkdocs_exporter(config)
        
        # Create proper DocumentChunk objects
        from src.chunkers import DocumentChunk
        
        sample_chunks = [
            DocumentChunk(
                chunk_id="chunk_001",
                content="This is the first chunk of the document.",
                chunk_type="paragraph",
                chunk_index=0,
                quality_score=0.9,
                metadata={"word_count": 10}
            ),
            DocumentChunk(
                chunk_id="chunk_002", 
                content="This is the second chunk of the document.",
                chunk_type="paragraph",
                chunk_index=1,
                quality_score=0.85,
                metadata={"word_count": 10}
            )
        ]
        
        sample_metadata = {
            "title": "Test Document",
            "quality_score": 0.875,
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
        original_content = "This is the complete test document. It contains multiple sentences and should be preserved in its original form."
        
        # Test export
        result = exporter.export_document(
            document_id="test_doc_001",
            chunks=sample_chunks,
            metadata=sample_metadata,
            source_filename="test_document.txt",
            original_content=original_content,
            build_site=True
        )
        
        if result.success:
            print(f"✅ Direct export successful: {result.pages_created} pages")
            print(f"   📁 Output: {result.output_directory}")
            
            # Verify output
            verify_mkdocs_output(result.output_directory)
        else:
            print(f"❌ Direct export failed: {result.errors}")
            
    except Exception as e:
        print(f"❌ Error with direct exporter: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        # Test the main functionality
        test_original_document_export()
        
        # Test the exporter directly
        test_mkdocs_exporter_directly()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
