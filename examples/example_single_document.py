#!/usr/bin/env python3
"""
🎯 Example: Single Document Processing with MkDocs

This example demonstrates how to process a single document and generate
a beautiful MkDocs site with original document preservation.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.processor import DocumentProcessor
from src.config import Config


def main():
    print("🎯 Single Document MkDocs Example")
    print("=" * 40)
    
    # Initialize processor
    config = Config()
    processor = DocumentProcessor(config)
    
    # Check if test document exists
    test_doc = Path("../documents/test_document.txt")
    if not test_doc.exists():
        print(f"❌ Test document not found: {test_doc}")
        print("💡 Create a text file to process, or use an existing document")
        return
    
    print(f"📄 Processing document: {test_doc.name}")
    
    # Process document with MkDocs export and site building
    result = processor.process_document_with_mkdocs(
        document_path=str(test_doc),
        export_mkdocs=True,
        build_site=True  # This creates the static HTML site
    )
    
    if result.success:
        print("✅ Processing successful!")
        print(f"📊 Generated {len(result.chunks)} chunks")
        print(f"🎯 Quality score: {result.quality_score:.3f}")
        
        # Show original content information
        if result.original_content:
            word_count = len(result.original_content.split())
            char_count = len(result.original_content)
            print(f"📄 Original content: {word_count:,} words, {char_count:,} characters")
        
        # Show MkDocs export information
        if 'mkdocs_export' in result.metadata:
            mkdocs_info = result.metadata['mkdocs_export']
            
            if mkdocs_info.get('success'):
                print(f"\n📚 MkDocs Site Generated:")
                print(f"   📄 Pages created: {mkdocs_info.get('pages_created', 0)}")
                print(f"   📁 Output directory: {mkdocs_info.get('output_directory', 'N/A')}")
                
                if mkdocs_info.get('site_built'):
                    build_time = mkdocs_info.get('build_time', 0)
                    site_url = mkdocs_info.get('site_url', 'N/A')
                    site_dir = mkdocs_info.get('site_directory', 'N/A')
                    
                    print(f"   🌐 Static site built in {build_time:.2f}s")
                    print(f"   📁 Site directory: {site_dir}")
                    print(f"   🔗 Site URL: {site_url}")
                    
                    print(f"\n🎨 Site Features:")
                    print(f"   📄 Original document preserved (complete version)")
                    print(f"   🔍 {len(result.chunks)} semantic chunks for easy navigation")
                    print(f"   🔍 Full-text search across all content")
                    print(f"   📱 Mobile-friendly Material Design theme")
                    print(f"   📊 Quality assessment and metadata")
                    
                    # Offer to open site
                    print(f"\n💡 Next steps:")
                    print(f"   • Open {site_url} in your browser")
                    print(f"   • Or run: cd {mkdocs_info.get('output_directory', '')} && python -m mkdocs serve")
                    
                    try:
                        import webbrowser
                        choice = input(f"\n🌍 Open site in browser? (y/N): ").strip().lower()
                        if choice in ['y', 'yes']:
                            webbrowser.open(site_url)
                            print("✅ Site opened in browser!")
                    except:
                        pass
                        
            else:
                print("❌ MkDocs export failed")
                if mkdocs_info.get('errors'):
                    for error in mkdocs_info['errors']:
                        print(f"   Error: {error}")
    else:
        print(f"❌ Processing failed: {result.error_message}")
        if result.warnings:
            print("⚠️  Warnings:")
            for warning in result.warnings:
                print(f"   {warning}")


if __name__ == "__main__":
    main()
