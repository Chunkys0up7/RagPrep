#!/usr/bin/env python3
"""
🚀 RAGPrep Complete MkDocs Site Generation Demo

This comprehensive demo showcases the new MkDocs static site generation 
capabilities with original document preservation.

Features demonstrated:
✅ Original document preservation (complete, unchunked versions)
✅ Automatic HTML site building with Material Design theme
✅ Full-text search and navigation
✅ Batch processing with site building
✅ Site serving for local development
✅ Error handling and quality assessment

Run this demo to see the complete workflow in action!
"""

import os
import sys
import time
import webbrowser
import subprocess
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.processor import DocumentProcessor
from src.config import Config


def print_banner():
    """Print a welcome banner."""
    print("🚀 RAGPrep Complete MkDocs Site Generation Demo")
    print("=" * 60)
    print("🎯 This demo showcases:")
    print("   📄 Original document preservation")
    print("   🌐 Static HTML site generation")
    print("   🔍 Full-text search capabilities")
    print("   🎨 Beautiful Material Design theme")
    print("   📊 Comprehensive metadata")
    print("   🧭 Smart navigation")
    print("=" * 60)
    print()


def create_sample_documents():
    """Create sample documents for demonstration."""
    print("📁 Creating sample documents...")
    
    documents_dir = Path("demo_documents")
    documents_dir.mkdir(exist_ok=True)
    
    # Sample 1: Technical documentation
    tech_doc = documents_dir / "technical_overview.txt"
    tech_doc.write_text("""
# RAGPrep Technical Overview

## Introduction
RAGPrep is a comprehensive document processing utility designed for Retrieval-Augmented Generation (RAG) applications. It provides a complete pipeline for document ingestion, processing, and preparation.

## Core Components

### Document Parsing
- Multi-format support (TXT, HTML, PDF, DOCX)
- Robust error handling
- Metadata extraction

### Chunking Strategies
- Fixed-size chunking with configurable overlap
- Structural chunking based on document hierarchy
- Semantic chunking using content analysis
- Hybrid approach combining multiple strategies

### Quality Assessment
- Content completeness analysis
- Structure integrity validation
- Metadata accuracy assessment
- Performance monitoring

### Security Features
- File validation and sanitization
- Content analysis for malicious patterns
- Size and format restrictions
- Safe processing environment

## Advanced Features

### Vector Storage
- Multiple backend support
- Efficient similarity search
- Batch operations
- Metadata filtering

### MkDocs Integration
- Automatic site generation
- Original document preservation
- Static HTML output
- Material Design theme
- Full-text search

### Monitoring and Analytics
- Processing metrics
- Performance optimization
- Error tracking
- Quality reports

## Use Cases
1. Knowledge base creation
2. Documentation processing
3. Content analysis
4. Research document preparation
5. Educational material organization

## Getting Started
Install dependencies and run the processor on your documents to see the magic happen!
""")
    
    # Sample 2: Business document
    business_doc = documents_dir / "business_plan.txt"
    business_doc.write_text("""
# Business Plan: AI Document Processing Service

## Executive Summary
Our AI-powered document processing service transforms how organizations handle their knowledge assets. By leveraging advanced RAG technology, we enable instant document understanding and intelligent content retrieval.

## Market Analysis
The document processing market is experiencing rapid growth, driven by:
- Increasing digital document volumes
- Need for intelligent content search
- Compliance and governance requirements
- Remote work acceleration

## Product Features
- Automated document ingestion
- Multi-format processing capabilities
- Intelligent chunking algorithms
- Quality assessment metrics
- Real-time processing monitoring

## Target Market
- Enterprise organizations
- Legal firms
- Research institutions
- Educational institutions
- Government agencies

## Revenue Model
- SaaS subscription tiers
- Usage-based pricing
- Enterprise custom solutions
- API access fees

## Financial Projections
Year 1: $500K revenue target
Year 2: $2M revenue target
Year 3: $10M revenue target

## Risk Assessment
- Competition from established players
- Technology dependency risks
- Data privacy concerns
- Scaling challenges

## Implementation Timeline
Q1: MVP development
Q2: Beta testing and feedback
Q3: Market launch
Q4: Feature expansion and partnerships
""")
    
    # Sample 3: Research paper
    research_doc = documents_dir / "research_paper.txt"
    research_doc.write_text("""
# Advances in Document Processing for Knowledge Management

## Abstract
This paper presents novel approaches to document processing that enhance knowledge extraction and retrieval in large-scale information systems. We introduce hybrid chunking strategies that preserve semantic coherence while optimizing for retrieval performance.

## Introduction
The exponential growth of digital documents has created unprecedented challenges for knowledge management systems. Traditional approaches to document processing often fail to capture the nuanced relationships between different content segments, leading to suboptimal retrieval performance.

## Related Work
Previous research in document processing has focused primarily on:
- Rule-based chunking approaches (Smith et al., 2020)
- Statistical methods for content segmentation (Johnson et al., 2021)
- Neural approaches to document understanding (Chen et al., 2022)

## Methodology
Our approach combines multiple strategies:

### Hybrid Chunking Algorithm
1. Structural analysis using document hierarchy
2. Semantic coherence assessment using embedding similarity
3. Quality-based refinement using readability metrics
4. Performance optimization through size constraints

### Quality Assessment Framework
- Content completeness metrics
- Structural integrity validation
- Metadata accuracy assessment
- Processing performance analysis

## Experimental Results
We evaluated our approach on three datasets:
- Technical documentation corpus (1,000 documents)
- Academic paper collection (5,000 documents)
- Business document archive (2,500 documents)

Results show significant improvements:
- 35% better retrieval accuracy
- 50% faster processing time
- 25% higher user satisfaction

## Discussion
The hybrid approach demonstrates superior performance across diverse document types. The quality assessment framework provides valuable insights for continuous improvement.

## Limitations
- Computational complexity for very large documents
- Language-specific optimizations required
- Domain adaptation needs

## Future Work
- Multi-language support
- Real-time processing capabilities
- Advanced semantic understanding
- Integration with knowledge graphs

## Conclusion
Our hybrid document processing approach offers significant advantages for knowledge management applications. The combination of multiple chunking strategies with comprehensive quality assessment creates a robust foundation for intelligent document processing.

## References
1. Smith, J. et al. (2020). Rule-based document chunking for information retrieval.
2. Johnson, M. et al. (2021). Statistical methods in content segmentation.
3. Chen, L. et al. (2022). Neural document understanding architectures.
""")
    
    print(f"✅ Created {len(list(documents_dir.glob('*.txt')))} sample documents in {documents_dir}/")
    return documents_dir


def demonstrate_single_document_processing(processor, documents_dir):
    """Demonstrate processing a single document with MkDocs."""
    print("\n🔄 Single Document Processing Demo")
    print("-" * 40)
    
    # Process the technical document
    tech_doc = documents_dir / "technical_overview.txt"
    print(f"📄 Processing: {tech_doc.name}")
    
    start_time = time.time()
    result = processor.process_document_with_mkdocs(
        document_path=str(tech_doc),
        export_mkdocs=True,
        build_site=True
    )
    end_time = time.time()
    
    if result.success:
        print(f"✅ Processing completed in {end_time - start_time:.2f}s")
        print(f"📊 Generated {len(result.chunks)} chunks")
        print(f"🎯 Quality score: {result.quality_score:.3f}")
        
        if result.original_content:
            print(f"📄 Original content preserved: {len(result.original_content):,} characters")
            print(f"📝 Word count: {len(result.original_content.split()):,}")
        
        # Check MkDocs export
        if 'mkdocs_export' in result.metadata:
            mkdocs_info = result.metadata['mkdocs_export']
            if mkdocs_info.get('success'):
                print(f"📚 MkDocs export successful!")
                print(f"   📄 Pages created: {mkdocs_info.get('pages_created', 0)}")
                print(f"   📁 Output directory: {mkdocs_info.get('output_directory', 'N/A')}")
                
                if mkdocs_info.get('site_built'):
                    print(f"   🌐 Static site built in {mkdocs_info.get('build_time', 0):.2f}s")
                    print(f"   📁 Site directory: {mkdocs_info.get('site_directory', 'N/A')}")
                    print(f"   🔗 Site URL: {mkdocs_info.get('site_url', 'N/A')}")
                    return mkdocs_info.get('site_url')
    else:
        print(f"❌ Processing failed: {result.error_message}")
    
    return None


def demonstrate_batch_processing(processor, documents_dir):
    """Demonstrate batch processing with MkDocs."""
    print("\n🔄 Batch Processing Demo")
    print("-" * 40)
    
    # Get all sample documents
    doc_paths = list(documents_dir.glob("*.txt"))
    print(f"📄 Processing {len(doc_paths)} documents in batch...")
    
    start_time = time.time()
    results = processor.process_batch_with_mkdocs(
        document_paths=[str(p) for p in doc_paths],
        export_mkdocs=True,
        build_site=True
    )
    end_time = time.time()
    
    successful_results = [r for r in results if r.success]
    
    print(f"✅ Batch processing completed in {end_time - start_time:.2f}s")
    print(f"📊 Processed {len(successful_results)}/{len(results)} documents successfully")
    
    total_chunks = sum(len(r.chunks) for r in successful_results)
    avg_quality = sum(r.quality_score for r in successful_results) / len(successful_results) if successful_results else 0
    
    print(f"📄 Total chunks generated: {total_chunks}")
    print(f"🎯 Average quality score: {avg_quality:.3f}")
    
    # Check batch MkDocs export
    if successful_results and 'batch_mkdocs_export' in successful_results[0].metadata:
        batch_info = successful_results[0].metadata['batch_mkdocs_export']
        if batch_info.get('success'):
            print(f"📚 Batch MkDocs export successful!")
            print(f"   📄 Total pages created: {batch_info.get('total_pages', 0)}")
            print(f"   📁 Output directory: {batch_info.get('output_directory', 'N/A')}")
            
            if batch_info.get('site_built'):
                print(f"   🌐 Batch site built in {batch_info.get('build_time', 0):.2f}s")
                print(f"   📁 Site directory: {batch_info.get('site_directory', 'N/A')}")
                print(f"   🔗 Site URL: {batch_info.get('site_url', 'N/A')}")
                return batch_info.get('site_url')
    
    return None


def serve_mkdocs_site(site_url):
    """Offer to serve the MkDocs site locally."""
    if not site_url:
        return
    
    print("\n🌐 MkDocs Site Options")
    print("-" * 40)
    print("Choose an option:")
    print("1. Open site in browser")
    print("2. Start MkDocs dev server (live reload)")
    print("3. Continue without viewing")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            print("🌍 Opening site in browser...")
            webbrowser.open(site_url)
            print(f"✅ Site opened: {site_url}")
            
        elif choice == "2":
            print("🚀 Starting MkDocs development server...")
            mkdocs_dir = Path("output/mkdocs")
            if mkdocs_dir.exists():
                try:
                    # Start MkDocs dev server
                    process = subprocess.Popen(
                        ["python", "-m", "mkdocs", "serve"],
                        cwd=mkdocs_dir,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    print("✅ MkDocs server started!")
                    print("🌍 Site available at: http://localhost:8000")
                    print("💡 The server has live reload - changes will update automatically")
                    print("🛑 Press Ctrl+C to stop the server")
                    
                    # Wait a moment then open browser
                    time.sleep(2)
                    webbrowser.open("http://localhost:8000")
                    
                    # Wait for user to stop
                    input("\nPress Enter to stop the MkDocs server...")
                    process.terminate()
                    print("🛑 MkDocs server stopped.")
                    
                except FileNotFoundError:
                    print("❌ MkDocs command not found. Make sure MkDocs is installed.")
                    print("   Install with: pip install mkdocs mkdocs-material")
                except Exception as e:
                    print(f"❌ Error starting MkDocs server: {e}")
            else:
                print("❌ MkDocs directory not found")
                
        elif choice == "3":
            print("✅ Continuing without viewing site")
            
        else:
            print("❌ Invalid choice")
            
    except KeyboardInterrupt:
        print("\n🛑 Cancelled by user")


def show_site_features():
    """Show the features available in the generated site."""
    print("\n🎨 Generated Site Features")
    print("-" * 40)
    print("📄 Original Documents:")
    print("   • Complete, unchunked versions preserved")
    print("   • Rich metadata headers with processing info")
    print("   • Document summaries and statistics")
    print("   • Quality assessment scores")
    print()
    print("🔍 Chunked Documents:")
    print("   • Semantic chunks for easy navigation")
    print("   • Individual pages for each chunk")
    print("   • Quality scores and metadata")
    print("   • Relationships between chunks")
    print()
    print("🌐 Site Features:")
    print("   • Material Design theme")
    print("   • Full-text search across all content")
    print("   • Responsive design for all devices")
    print("   • Smart navigation with document hierarchy")
    print("   • Automatic table of contents")
    print("   • Code syntax highlighting")
    print("   • Mathematical formula support")
    print()
    print("📊 Metadata Available:")
    print("   • Processing timestamps")
    print("   • Quality assessment results")
    print("   • Content statistics (word count, etc.)")
    print("   • Chunk relationships")
    print("   • Performance metrics")


def cleanup_demo():
    """Clean up demo files."""
    print("\n🧹 Cleanup Options")
    print("-" * 40)
    print("Choose what to clean up:")
    print("1. Remove sample documents only")
    print("2. Remove sample documents and output")
    print("3. Keep everything")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            documents_dir = Path("demo_documents")
            if documents_dir.exists():
                import shutil
                shutil.rmtree(documents_dir)
                print("✅ Sample documents removed")
            
        elif choice == "2":
            import shutil
            documents_dir = Path("demo_documents")
            if documents_dir.exists():
                shutil.rmtree(documents_dir)
                print("✅ Sample documents removed")
            
            # Note: We might want to keep the output for further inspection
            print("💡 Output kept for inspection - remove manually if needed")
            
        elif choice == "3":
            print("✅ All files kept")
            
        else:
            print("❌ Invalid choice - keeping all files")
            
    except KeyboardInterrupt:
        print("\n✅ Keeping all files")


def main():
    """Run the comprehensive demo."""
    print_banner()
    
    try:
        # Initialize configuration and processor
        print("📋 Initializing RAGPrep processor...")
        config = Config()
        processor = DocumentProcessor(config)
        print("✅ Processor initialized successfully")
        
        # Create sample documents
        documents_dir = create_sample_documents()
        
        # Demonstrate single document processing
        site_url = demonstrate_single_document_processing(processor, documents_dir)
        
        # Demonstrate batch processing
        batch_site_url = demonstrate_batch_processing(processor, documents_dir)
        
        # Use the batch site URL if available, otherwise single site URL
        final_site_url = batch_site_url or site_url
        
        # Show site features
        show_site_features()
        
        # Offer to serve the site
        serve_mkdocs_site(final_site_url)
        
        print("\n🎉 Demo completed successfully!")
        print("\n💡 Next Steps:")
        print("   • Explore the generated MkDocs site")
        print("   • Try processing your own documents")
        print("   • Customize the MkDocs theme and configuration")
        print("   • Integrate with your RAG application")
        
        # Cleanup
        cleanup_demo()
        
    except KeyboardInterrupt:
        print("\n\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
