#!/usr/bin/env python3
"""
🎯 Example: Batch Document Processing with MkDocs

This example demonstrates how to process multiple documents and generate
a unified MkDocs site with all documents included.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.processor import DocumentProcessor
from src.config import Config


def create_sample_documents():
    """Create sample documents for batch processing."""
    examples_dir = Path(__file__).parent
    docs_dir = examples_dir / "sample_docs"
    docs_dir.mkdir(exist_ok=True)
    
    # Sample 1: API Documentation
    api_doc = docs_dir / "api_guide.txt"
    api_doc.write_text("""
# API Documentation

## Overview
This API provides endpoints for document processing and analysis.

## Authentication
All requests require an API key in the header:
```
Authorization: Bearer YOUR_API_KEY
```

## Endpoints

### POST /documents
Upload a document for processing.

**Parameters:**
- file: Document file (PDF, DOCX, TXT)
- options: Processing options (JSON)

**Response:**
```json
{
  "document_id": "uuid",
  "status": "processing",
  "estimated_time": 30
}
```

### GET /documents/{id}
Get document processing status and results.

**Response:**
```json
{
  "document_id": "uuid",
  "status": "completed",
  "chunks": [...],
  "metadata": {...},
  "quality_score": 0.85
}
```

## Error Handling
The API returns standard HTTP status codes:
- 200: Success
- 400: Bad Request
- 401: Unauthorized
- 429: Rate Limited
- 500: Server Error

## Rate Limits
- 100 requests per minute for standard users
- 1000 requests per minute for premium users
""")
    
    # Sample 2: User Guide
    user_guide = docs_dir / "user_guide.txt"
    user_guide.write_text("""
# User Guide

## Getting Started

Welcome to the document processing platform! This guide will help you get started quickly.

## Step 1: Account Setup
1. Sign up for an account
2. Verify your email address
3. Choose your subscription plan
4. Generate your API key

## Step 2: Upload Documents
You can upload documents in several ways:

### Web Interface
1. Go to the dashboard
2. Click "Upload Documents"
3. Select your files
4. Configure processing options
5. Click "Process"

### API Upload
Use the /documents endpoint to upload programmatically:
```bash
curl -X POST https://api.example.com/documents \
  -H "Authorization: Bearer YOUR_KEY" \
  -F "file=@document.pdf"
```

## Step 3: Review Results
Once processing is complete:
1. View the generated chunks
2. Review quality assessments
3. Download processed data
4. Access the generated website

## Tips for Best Results
- Use high-quality source documents
- Ensure text is readable (not scanned images)
- Choose appropriate chunking strategies
- Review quality scores and adjust settings

## Troubleshooting
Common issues and solutions:
- Upload failures: Check file format and size
- Poor quality scores: Review source document quality
- Processing timeouts: Large documents may take longer
""")
    
    # Sample 3: FAQ
    faq = docs_dir / "faq.txt"
    faq.write_text("""
# Frequently Asked Questions

## General Questions

### What file formats are supported?
We support:
- PDF documents
- Microsoft Word (DOCX)
- Plain text (TXT)
- HTML files
- Markdown (MD)

### What is the maximum file size?
- Standard users: 10MB per file
- Premium users: 100MB per file
- Enterprise users: 1GB per file

### How long does processing take?
Processing time depends on:
- Document size and complexity
- Selected processing options
- Current system load

Typical times:
- Small documents (< 1MB): 30 seconds
- Medium documents (1-10MB): 2-5 minutes
- Large documents (> 10MB): 10-30 minutes

## Technical Questions

### How does chunking work?
We use advanced algorithms to:
1. Analyze document structure
2. Identify semantic boundaries
3. Create coherent chunks
4. Maintain context relationships

### What quality metrics are calculated?
Quality assessment includes:
- Content completeness
- Structure integrity
- Metadata accuracy
- Processing performance

### Can I customize chunking strategies?
Yes! Options include:
- Fixed-size chunking
- Structural chunking
- Semantic chunking
- Hybrid approaches

## Billing Questions

### How is usage calculated?
Billing is based on:
- Number of documents processed
- Total processing time
- Storage used
- API calls made

### Can I upgrade or downgrade my plan?
Yes, you can change plans at any time. Changes take effect immediately.

### Do you offer refunds?
We offer full refunds within 30 days of purchase for any reason.
""")
    
    return docs_dir, [api_doc, user_guide, faq]


def main():
    print("🎯 Batch Document Processing Example")
    print("=" * 40)
    
    # Create sample documents
    print("📁 Creating sample documents...")
    docs_dir, doc_files = create_sample_documents()
    print(f"✅ Created {len(doc_files)} sample documents in {docs_dir}/")
    
    # Initialize processor
    config = Config()
    processor = DocumentProcessor(config)
    
    # Get document paths
    doc_paths = [str(doc) for doc in doc_files]
    
    print(f"\n🔄 Processing {len(doc_paths)} documents in batch...")
    print("📄 Documents:")
    for path in doc_paths:
        print(f"   • {Path(path).name}")
    
    # Start timer
    start_time = time.time()
    
    # Process documents in batch with MkDocs export
    results = processor.process_batch_with_mkdocs(
        document_paths=doc_paths,
        export_mkdocs=True,
        build_site=True  # Build unified site for all documents
    )
    
    # Calculate timing
    end_time = time.time()
    total_time = end_time - start_time
    
    # Analyze results
    successful_results = [r for r in results if r.success]
    failed_results = [r for r in results if not r.success]
    
    print(f"\n✅ Batch processing completed in {total_time:.2f}s")
    print(f"📊 Results: {len(successful_results)}/{len(results)} documents processed successfully")
    
    if failed_results:
        print(f"❌ Failed documents:")
        for result in failed_results:
            print(f"   • {result.document_id}: {result.error_message}")
    
    if successful_results:
        # Calculate aggregate statistics
        total_chunks = sum(len(r.chunks) for r in successful_results)
        avg_quality = sum(r.quality_score for r in successful_results) / len(successful_results)
        total_words = sum(len(r.original_content.split()) if r.original_content else 0 
                         for r in successful_results)
        
        print(f"\n📈 Aggregate Statistics:")
        print(f"   📄 Total chunks generated: {total_chunks:,}")
        print(f"   🎯 Average quality score: {avg_quality:.3f}")
        print(f"   📝 Total words processed: {total_words:,}")
        
        # Show batch MkDocs export information
        if 'batch_mkdocs_export' in successful_results[0].metadata:
            batch_info = successful_results[0].metadata['batch_mkdocs_export']
            
            if batch_info.get('success'):
                print(f"\n📚 Unified MkDocs Site Generated:")
                print(f"   📄 Total pages created: {batch_info.get('total_pages', 0)}")
                print(f"   📁 Output directory: {batch_info.get('output_directory', 'N/A')}")
                
                if batch_info.get('site_built'):
                    build_time = batch_info.get('build_time', 0)
                    site_url = batch_info.get('site_url', 'N/A')
                    site_dir = batch_info.get('site_directory', 'N/A')
                    
                    print(f"   🌐 Static site built in {build_time:.2f}s")
                    print(f"   📁 Site directory: {site_dir}")
                    print(f"   🔗 Site URL: {site_url}")
                    
                    print(f"\n🎨 Unified Site Features:")
                    print(f"   📄 All {len(successful_results)} documents preserved (original + chunked)")
                    print(f"   🔍 Unified search across all {total_chunks:,} chunks")
                    print(f"   📱 Single, cohesive documentation site")
                    print(f"   🧭 Smart navigation between related documents")
                    print(f"   📊 Quality assessment for all content")
                    
                    # Show document breakdown
                    print(f"\n📑 Document Breakdown:")
                    for result in successful_results:
                        word_count = len(result.original_content.split()) if result.original_content else 0
                        print(f"   • {result.document_id}: {len(result.chunks)} chunks, "
                              f"{word_count:,} words, quality {result.quality_score:.3f}")
                    
                    # Offer to open site
                    print(f"\n💡 Next steps:")
                    print(f"   • Open {site_url} in your browser")
                    print(f"   • Or run: cd {batch_info.get('output_directory', '')} && python -m mkdocs serve")
                    print(f"   • Search across all documents using the unified search")
                    print(f"   • Navigate between original documents and chunks")
                    
                    try:
                        import webbrowser
                        choice = input(f"\n🌍 Open unified site in browser? (y/N): ").strip().lower()
                        if choice in ['y', 'yes']:
                            webbrowser.open(site_url)
                            print("✅ Unified site opened in browser!")
                    except:
                        pass
                        
            else:
                print("❌ Batch MkDocs export failed")
                if batch_info.get('errors'):
                    for error in batch_info['errors']:
                        print(f"   Error: {error}")
    
    # Cleanup option
    print(f"\n🧹 Cleanup:")
    choice = input(f"Remove sample documents from {docs_dir}? (y/N): ").strip().lower()
    if choice in ['y', 'yes']:
        import shutil
        shutil.rmtree(docs_dir)
        print("✅ Sample documents removed")
    else:
        print(f"📁 Sample documents kept in {docs_dir}/")


if __name__ == "__main__":
    main()
