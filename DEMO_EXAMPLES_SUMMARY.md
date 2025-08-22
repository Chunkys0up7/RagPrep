# 🎮 Demo and Examples Summary

## 🚀 Available Demos and Examples

This directory contains comprehensive demos and examples showcasing RAGPrep's new MkDocs static site generation capabilities with original document preservation.

## 📁 Demo Scripts

### 1. **Comprehensive Demo** (`demo_comprehensive.py`)
**The complete, interactive demonstration of all features.**

```bash
python demo_comprehensive.py
```

**Features:**
- Creates sample documents automatically
- Demonstrates single and batch processing
- Shows original document preservation
- Builds beautiful MkDocs sites
- Offers to serve the site locally
- Interactive cleanup options

**Perfect for:** First-time users, showcasing capabilities

---

### 2. **Site Serving Demo** (`demo_mkdocs_serve.py`)
**Focused on building and serving MkDocs sites.**

```bash
python demo_mkdocs_serve.py
```

**Features:**
- Simple document processing
- Automatic site building
- Development server startup
- Browser integration

**Perfect for:** Quick site generation and testing

---

### 3. **Original Document Test** (`test_original_document.py`)
**Technical validation of original document preservation.**

```bash
python test_original_document.py
```

**Features:**
- Comprehensive functionality testing
- Output structure verification
- Direct MkDocs exporter testing
- Quality validation

**Perfect for:** Technical validation, debugging

## 📚 Example Scripts

### 1. **Single Document Example** (`examples/example_single_document.py`)
**Process one document and create a site.**

```python
# Simple single document processing
result = processor.process_document_with_mkdocs(
    document_path="document.txt",
    export_mkdocs=True,
    build_site=True
)
```

**Use case:** Individual document processing, API documentation

---

### 2. **Batch Processing Example** (`examples/example_batch_processing.py`)
**Process multiple documents into unified site.**

```python
# Batch processing with unified site
results = processor.process_batch_with_mkdocs(
    document_paths=["doc1.txt", "doc2.pdf", "doc3.docx"],
    export_mkdocs=True,
    build_site=True
)
```

**Use case:** Documentation collections, knowledge bases

---

### 3. **Custom Site Example** (`examples/example_custom_site.py`)
**Advanced customization with themes and assets.**

**Features:**
- Custom Material Design theme
- Custom CSS and JavaScript
- Enhanced navigation
- Quality score indicators
- Document type badges

**Use case:** Professional documentation sites, branding

## 🎯 Quick Start Guide

### 1. **Instant Demo** (30 seconds)
```bash
git clone <repo>
cd RagPrep
python demo_comprehensive.py
```

### 2. **Your Own Document** (1 minute)
```python
from src.processor import DocumentProcessor
from src.config import Config

processor = DocumentProcessor(Config())
result = processor.process_document_with_mkdocs("your_file.pdf", build_site=True)
print(f"Site: {result.metadata['mkdocs_export']['site_url']}")
```

### 3. **Custom Site** (5 minutes)
```bash
python examples/example_custom_site.py
```

## 🌐 What You Get

### 📄 **Original Documents**
- Complete, unchunked content preserved
- Rich metadata with processing information
- Quality assessment scores
- Beautiful Material Design formatting

### 🔍 **Chunked Content**
- Semantic segments for easy navigation
- Individual pages with metadata
- Quality scores per chunk
- Smart inter-chunk navigation

### 🎨 **Professional Sites**
- Material Design theme
- Full-text search across all content
- Mobile-responsive design
- Dark/light mode toggle
- SEO-friendly structure

### ⚡ **Developer Features**
- Live reload development server
- Custom CSS/JavaScript support
- Plugin ecosystem
- Static deployment ready

## 📊 Example Outputs

### Single Document Site
```
📚 Site Generated:
   📄 1 original document preserved
   🔍 4 semantic chunks created
   🔍 Full-text search enabled
   📱 Mobile-friendly design
   🌐 http://localhost:8000
```

### Batch Processing Site
```
📚 Unified Site Generated:
   📄 3 documents preserved (original + chunked)
   🔍 12 total chunks across all documents
   🔍 Unified search across all content
   🧭 Smart cross-document navigation
   📊 Aggregate quality statistics
```

### Custom Site Features
```
🎨 Custom Features:
   🎨 Purple Material Design theme
   🏷️ Document type badges
   📊 Interactive quality indicators
   🔍 Enhanced search with filtering
   📱 Advanced responsive design
   ⚡ Custom JavaScript interactions
```

## 🛠️ Technical Details

### Site Structure
```
output/mkdocs/
├── mkdocs.yml           # Configuration
├── site/                # Built HTML (deploy this)
│   ├── index.html
│   ├── search/
│   └── ...
└── docs/                # Source markdown
    ├── index.md
    ├── _navigation.md
    └── document_name/
        ├── original_document.md  # 📄 Complete
        ├── chunk_000.md         # 🔍 Chunks
        └── ...
```

### Metadata Available
- Processing timestamps
- Quality assessment results
- Content statistics (words, chars)
- Chunk relationships
- Performance metrics
- Build information

### Customization Options
- Theme colors and fonts
- Navigation structure
- Search configuration
- Plugin integration
- Custom CSS/JavaScript
- Asset management

## 🚀 Deployment Ready

The generated sites are ready for deployment to:
- **GitHub Pages**
- **Netlify** 
- **Vercel**
- **AWS S3**
- **Any static hosting**

Simply upload the `site/` directory!

## 💡 Pro Tips

1. **Start with the comprehensive demo** to see all features
2. **Use batch processing** for related documents
3. **Customize themes** for professional branding
4. **Enable analytics** for usage tracking
5. **Set up CI/CD** for automatic rebuilds
6. **Use the dev server** for live editing

## 🎉 Ready to Start?

Choose your path:
- **Quick demo:** `python demo_comprehensive.py`
- **Your documents:** `python examples/example_single_document.py`  
- **Professional site:** `python examples/example_custom_site.py`

Happy documenting! 📚✨
