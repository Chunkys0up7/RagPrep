#!/usr/bin/env python3
"""
Demo script for serving the MkDocs site locally

This script demonstrates how to build and serve the complete MkDocs site
with both original documents and chunked versions.
"""

import os
import sys
import time
import webbrowser
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.processor import DocumentProcessor
from src.config import Config
from src.mkdocs_exporter import get_mkdocs_exporter


def demo_mkdocs_site_building():
    """Demonstrate building and serving the complete MkDocs site."""
    
    print("🌐 RAGPrep MkDocs Site Building & Serving Demo")
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
    
    # Process document with full site building
    print("\n🔄 Processing document with full MkDocs site building...")
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
            
            # Check MkDocs export and site building status
            if 'mkdocs_export' in result.metadata:
                mkdocs_info = result.metadata['mkdocs_export']
                if mkdocs_info.get('success'):
                    print(f"\n📚 MkDocs export successful!")
                    print(f"   📄 Pages created: {mkdocs_info.get('pages_created', 0)}")
                    print(f"   📁 Output directory: {mkdocs_info.get('output_directory', 'N/A')}")
                    
                    # Show site building information
                    if mkdocs_info.get('site_built'):
                        print(f"\n🌐 Static site built successfully!")
                        print(f"   ⏱️  Build time: {mkdocs_info.get('build_time', 0):.2f}s")
                        print(f"   📁 Site directory: {mkdocs_info.get('site_directory', 'N/A')}")
                        print(f"   🔗 Site URL: {mkdocs_info.get('site_url', 'N/A')}")
                        
                        # Ask if user wants to open the site
                        open_site = input("\n🌐 Would you like to open the site in your browser? (y/n): ").lower().strip()
                        if open_site in ['y', 'yes']:
                            site_url = mkdocs_info.get('site_url', '')
                            if site_url:
                                print(f"🚀 Opening site in browser: {site_url}")
                                webbrowser.open(site_url)
                            else:
                                print("❌ No site URL available")
                        
                        # Demonstrate serving the site
                        serve_site = input("\n🖥️  Would you like to start a live development server? (y/n): ").lower().strip()
                        if serve_site in ['y', 'yes']:
                            start_development_server(mkdocs_info.get('output_directory', ''))
                        
                    else:
                        print(f"   ⚠️  Static site: Not built")
                        print(f"   ❌ Build errors: {mkdocs_info.get('warnings', [])}")
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
    
    print("\n🎉 Demo completed!")


def start_development_server(output_directory: str):
    """Start the MkDocs development server."""
    
    if not output_directory:
        print("❌ No output directory specified")
        return
    
    print(f"\n🖥️  Starting MkDocs development server...")
    print(f"📁 Working directory: {output_directory}")
    
    try:
        # Get the exporter to use its serving functionality
        config = Config()
        exporter = get_mkdocs_exporter(config)
        
        # Start the server
        serve_result = exporter.serve_mkdocs_site(port=8000, host="127.0.0.1")
        
        if serve_result['success']:
            print(f"✅ MkDocs server started successfully!")
            print(f"🌐 Server URL: {serve_result['url']}")
            print(f"📱 Access from any device on your network")
            print("\n📖 Features available:")
            print("   • 🔍 Full-text search across all documents")
            print("   • 📄 Original complete documents")
            print("   • 🔪 Semantic chunks for detailed analysis")
            print("   • 🎨 Beautiful Material Design interface")
            print("   • 📱 Mobile-responsive design")
            print("   • 🌙 Dark/light theme support")
            
            # Open in browser
            open_browser = input(f"\n🌐 Open {serve_result['url']} in browser? (y/n): ").lower().strip()
            if open_browser in ['y', 'yes']:
                webbrowser.open(serve_result['url'])
            
            print(f"\n⏹️  Press Ctrl+C to stop the server")
            
            # Keep the server running
            try:
                process = serve_result['process']
                process.wait()  # Wait for the process to complete
            except KeyboardInterrupt:
                print(f"\n⏹️  Stopping MkDocs server...")
                process.terminate()
                process.wait()
                print(f"✅ Server stopped successfully")
        else:
            print(f"❌ Failed to start server: {serve_result['errors']}")
            
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        import traceback
        traceback.print_exc()


def show_site_features():
    """Show the features of the generated MkDocs site."""
    
    print("\n🌟 Generated MkDocs Site Features")
    print("=" * 40)
    
    features = [
        "📄 **Complete Original Documents** - Full, unchunked versions preserved",
        "🔪 **Semantic Chunks** - Intelligent document segmentation", 
        "🔍 **Full-Text Search** - Find content across all documents instantly",
        "📱 **Mobile Responsive** - Perfect on desktop, tablet, and mobile",
        "🎨 **Material Design** - Beautiful, modern interface",
        "🌙 **Dark/Light Themes** - Comfortable viewing in any environment",
        "📊 **Document Metadata** - Quality scores, word counts, processing info",
        "🧭 **Smart Navigation** - Easy browsing between documents and chunks",
        "⚡ **Fast Loading** - Optimized static site generation",
        "🔗 **Direct Links** - Shareable URLs for specific documents and chunks",
        "📋 **Table of Contents** - Automatic TOC generation for long documents",
        "🏷️  **Document Labels** - Clear marking of original vs. chunked content"
    ]
    
    for feature in features:
        print(f"   {feature}")
    
    print(f"\n💡 **Use Cases:**")
    print(f"   • Research and document analysis")
    print(f"   • RAG application development") 
    print(f"   • Document quality assessment")
    print(f"   • Team collaboration on processed documents")
    print(f"   • Client presentation of processed content")


if __name__ == "__main__":
    try:
        # Show features first
        show_site_features()
        
        # Run the main demo
        demo_mkdocs_site_building()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
