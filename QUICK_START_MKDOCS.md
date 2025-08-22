# 🚀 Quick Start: MkDocs Site Generation

Get started with RAGPrep's MkDocs static site generation in just a few minutes!

## 📦 Installation

Ensure you have the required dependencies:

```bash
pip install mkdocs mkdocs-material mkdocstrings mkdocstrings-python
```

## ⚡ 30-Second Demo

```python
# Import and initialize
from src.processor import DocumentProcessor
from src.config import Config

processor = DocumentProcessor(Config())

# Process any document and build a site
result = processor.process_document_with_mkdocs(
    document_path="your_document.txt",  # Any .txt, .pdf, .docx, .html
    export_mkdocs=True,
    build_site=True
)

# Check the result
if result.success:
    site_url = result.metadata['mkdocs_export']['site_url']
    print(f"🌐 Your site is ready: {site_url}")
```

## 🎯 What You Get

### ✅ **Original Document Preserved**
- Complete, unchunked version with beautiful formatting
- Rich metadata headers with processing information
- Quality assessment scores and statistics

### ✅ **Semantic Chunks**
- Individual pages for each document segment
- Smart navigation between related content
- Quality scores for each chunk

### ✅ **Professional Site**
- Material Design theme
- Full-text search across all content
- Mobile-friendly responsive design
- SEO-ready structure

## 🌐 View Your Site

### Option 1: Open Static Site
```python
import webbrowser
webbrowser.open(result.metadata['mkdocs_export']['site_url'])
```

### Option 2: Development Server
```bash
cd output/mkdocs
python -m mkdocs serve
# Visit http://localhost:8000
```

## 📁 Multiple Documents

Process multiple documents into a single site:

```python
# Batch processing
results = processor.process_batch_with_mkdocs(
    document_paths=[
        "doc1.txt",
        "doc2.pdf", 
        "doc3.docx"
    ],
    export_mkdocs=True,
    build_site=True
)

# All documents in one searchable site!
batch_info = results[0].metadata['batch_mkdocs_export']
print(f"📚 Site with {batch_info['total_pages']} pages ready!")
print(f"🌐 {batch_info['site_url']}")
```

## 🎨 Site Features

Your generated site includes:

- **📄 Original Documents**: Complete, unchunked versions
- **🔍 Chunked Content**: Semantic segments for easy browsing
- **🔍 Full-Text Search**: Find content instantly
- **📱 Mobile-Friendly**: Responsive design
- **🎨 Beautiful Theme**: Material Design
- **📊 Metadata**: Quality scores, processing info
- **🧭 Smart Navigation**: Hierarchical document organization

## 🎮 Interactive Demo

Run the comprehensive demo:

```bash
python demo_comprehensive.py
```

This demo will:
1. Create sample documents
2. Process them with MkDocs
3. Show you the generated site
4. Offer to start a development server

## 🚀 Deploy Your Site

The generated `output/mkdocs/site/` directory is ready for deployment to:
- GitHub Pages
- Netlify
- Vercel
- AWS S3
- Any static hosting service

## 🛠️ Customization

Edit `output/mkdocs/mkdocs.yml` to customize:
- Site name and description
- Theme colors and fonts
- Navigation structure
- Search configuration
- Additional plugins

## 📖 Full Documentation

For complete documentation, see:
- [MKDOCS_SITE_GENERATION.md](MKDOCS_SITE_GENERATION.md) - Complete guide
- [ORIGINAL_DOCUMENT_FEATURE.md](ORIGINAL_DOCUMENT_FEATURE.md) - Feature details

## 🎉 That's It!

You now have a beautiful, searchable documentation site generated from your documents. The original document content is preserved alongside intelligently chunked versions, all in a professional Material Design theme.

Happy documenting! 📚✨
