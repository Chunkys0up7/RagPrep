#!/usr/bin/env python3
"""
Demo script for MkDocs export functionality in RAGPrep

This script demonstrates how documents are automatically converted to markdown
and integrated into MkDocs as part of the processing pipeline.
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


def demo_mkdocs_export():
    """Demonstrate MkDocs export functionality."""
    
    print("🚀 RAGPrep MkDocs Export Demo")
    print("=" * 50)
    
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
                    print(f"   ⚙️  Config file: {mkdocs_info.get('mkdocs_config_path', 'N/A')}")
                    
                    # Show information about original document
                    if result.original_content:
                        print(f"   📄 Original document: Complete, unchunked version created")
                        print(f"   📄 Chunked versions: {len(result.chunks)} semantic chunks created")
                    else:
                        print(f"   ⚠️  Original document: Not available")
                    
                    # Show site building information
                    if mkdocs_info.get('site_built'):
                        print(f"   🌐 Static site built successfully in {mkdocs_info.get('build_time', 0):.2f}s")
                        print(f"   📁 Site directory: {mkdocs_info.get('site_directory', 'N/A')}")
                        print(f"   🔗 Site URL: {mkdocs_info.get('site_url', 'N/A')}")
                    else:
                        print(f"   ⚠️  Static site: Not built")
                else:
                    print(f"❌ MkDocs export failed: {mkdocs_info.get('errors', [])}")
            else:
                print("⚠️  No MkDocs export information found")
            
            # Show output structure
            print("\n📁 Generated MkDocs structure:")
            show_mkdocs_structure()
            
        else:
            print(f"❌ Document processing failed: {result.error_message}")
            return
            
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        return
    
    # Demonstrate MkDocs exporter directly
    print("\n🔧 Testing MkDocs exporter directly...")
    try:
        exporter = get_mkdocs_exporter(config)
        
        # Export the same document again
        mkdocs_result = exporter.export_document(
            document_id=result.document_id,
            chunks=result.chunks,
            metadata=result.metadata,
            source_filename=Path(test_doc_path).name,
            build_site=True
        )
        
        if mkdocs_result.success:
            print(f"✅ Direct export successful: {mkdocs_result.pages_created} pages")
        else:
            print(f"❌ Direct export failed: {mkdocs_result.errors}")
            
    except Exception as e:
        print(f"❌ Error with direct exporter: {e}")
    
    print("\n🎉 Demo completed!")
    print("\n📖 Next steps:")
    print("1. Check the generated MkDocs output in output/mkdocs/")
    print("2. Run 'mkdocs serve' in the output/mkdocs/ directory")
    print("3. View your processed documents as a beautiful documentation site")


def show_mkdocs_structure():
    """Show the generated MkDocs directory structure."""
    
    mkdocs_dir = Path("output/mkdocs")
    if not mkdocs_dir.exists():
        print("   📁 output/mkdocs/ (not created)")
        return
    
    print(f"   📁 {mkdocs_dir}/")
    
    # Show docs directory
    docs_dir = mkdocs_dir / "docs"
    if docs_dir.exists():
        print(f"   📁 {docs_dir}/")
        
        # Show markdown files
        for md_file in docs_dir.rglob("*.md"):
            relative_path = md_file.relative_to(docs_dir)
            print(f"   📄 {relative_path}")
    
    # Show config file
    config_file = mkdocs_dir / "mkdocs.yml"
    if config_file.exists():
        print(f"   ⚙️  mkdocs.yml")
    
    # Show navigation
    nav_file = docs_dir / "_navigation.md" if docs_dir.exists() else None
    if nav_file and nav_file.exists():
        print(f"   🧭 _navigation.md")


def demo_mkdocs_build():
    """Demonstrate building the MkDocs site."""
    
    print("\n🔨 Building MkDocs site...")
    
    mkdocs_dir = Path("output/mkdocs")
    if not mkdocs_dir.exists():
        print("❌ MkDocs output directory not found. Run the export demo first.")
        return
    
    # Check if we can build
    docs_dir = mkdocs_dir / "docs"
    config_file = mkdocs_dir / "mkdocs.yml"
    
    if not docs_dir.exists() or not config_file.exists():
        print("❌ Missing required files for MkDocs build")
        return
    
    # Count markdown files
    md_files = list(docs_dir.rglob("*.md"))
    if not md_files:
        print("❌ No markdown files found for build")
        return
    
    print(f"✅ Found {len(md_files)} markdown files")
    print(f"📁 Working directory: {mkdocs_dir.absolute()}")
    
    print("\n🚀 To build and serve your MkDocs site:")
    print(f"cd {mkdocs_dir.absolute()}")
    print("mkdocs serve")
    print("\n🌐 Your site will be available at: http://127.0.0.1:8000")


if __name__ == "__main__":
    try:
        # Run the main demo
        demo_mkdocs_export()
        
        # Show build instructions
        demo_mkdocs_build()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
