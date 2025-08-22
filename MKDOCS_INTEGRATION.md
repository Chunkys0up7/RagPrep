# MkDocs Integration for RAGPrep

## 🎯 Overview

RAGPrep now includes **automatic MkDocs export** as part of the document processing pipeline. This means that every document you process is automatically:

1. **Converted to markdown format** with rich metadata
2. **Organized into a MkDocs site structure** 
3. **Ready for immediate viewing** as a beautiful documentation site
4. **Perfect for RAG ingestion** with clean, structured content

## 🚀 How It Works

### **Automatic Pipeline Integration**

When you process a document, the system now:

```
Document Input → Parse → Chunk → Extract Metadata → Export to MkDocs → Ready for RAG
```

### **What Gets Generated**

For each processed document, you get:

- **Individual markdown files** for each chunk
- **Rich metadata headers** with quality scores and processing info
- **Organized directory structure** by document
- **Navigation files** for easy browsing
- **MkDocs configuration** ready to serve

## 📁 Output Structure

```
output/mkdocs/
├── mkdocs.yml              # MkDocs configuration
├── docs/
│   ├── index.md            # Main index page
│   ├── _navigation.md      # Auto-generated navigation
│   └── {document_name}/    # Document-specific folders
│       ├── chunk_000.md    # Individual chunk files
│       ├── chunk_001.md
│       └── ...
```

## 🔧 Usage

### **Basic Processing with MkDocs Export**

```python
from src.processor import DocumentProcessor

# Initialize processor
processor = DocumentProcessor()

# Process document with automatic MkDocs export
result = processor.process_document_with_mkdocs(
    document_path="path/to/document.pdf",
    export_mkdocs=True  # This is the default
)

# Check export status
if result.metadata.get('mkdocs_export', {}).get('success'):
    print(f"MkDocs export successful!")
    print(f"Output directory: {result.metadata['mkdocs_export']['output_directory']}")
```

### **Batch Processing with MkDocs Export**

```python
# Process multiple documents
results = processor.process_batch_with_mkdocs(
    document_paths=["doc1.pdf", "doc2.docx", "doc3.txt"],
    export_mkdocs=True
)

# Each document gets its own section in MkDocs
for result in results:
    if result.success:
        print(f"Document exported: {result.metadata.get('mkdocs_export', {})}")
```

### **Direct MkDocs Exporter Usage**

```python
from src.mkdocs_exporter import get_mkdocs_exporter

exporter = get_mkdocs_exporter()

# Export existing processing results
mkdocs_result = exporter.export_document(
    document_id="doc_123",
    chunks=chunks,
    metadata=metadata,
    source_filename="document.pdf"
)

print(f"Created {mkdocs_result.pages_created} pages")
print(f"Output: {mkdocs_result.output_directory}")
```

## 🌐 Viewing Your MkDocs Site

### **Quick Start**

1. **Process a document** using the new methods
2. **Navigate to the output directory**:
   ```bash
   cd output/mkdocs
   ```
3. **Install MkDocs** (if not already installed):
   ```bash
   pip install mkdocs mkdocs-material
   ```
4. **Serve the site**:
   ```bash
   mkdocs serve
   ```
5. **Open your browser** to `http://127.0.0.1:8000`

### **Build for Production**

```bash
cd output/mkdocs
mkdocs build
```

This creates a `site/` directory with static HTML files ready for deployment.

## 📊 API Integration

### **New API Endpoints**

The FastAPI now includes MkDocs-specific endpoints:

```bash
# Process document with MkDocs export
POST /process-document
{
    "document_path": "path/to/doc.pdf",
    "export_mkdocs": true
}

# Check MkDocs status
GET /mkdocs/status

# Get navigation structure
GET /mkdocs/navigation

# Build MkDocs site
POST /mkdocs/build
```

### **Response Includes MkDocs Info**

```json
{
    "success": true,
    "document_id": "doc_123",
    "chunks_count": 5,
    "quality_score": 0.85,
    "mkdocs_export": {
        "success": true,
        "pages_created": 5,
        "output_directory": "output/mkdocs",
        "mkdocs_config_path": "output/mkdocs/mkdocs.yml"
    }
}
```

## 🎨 Customization

### **MkDocs Configuration**

The generated `mkdocs.yml` includes:

- **Material theme** with advanced features
- **Search functionality**
- **Code highlighting**
- **Navigation tabs and sections**
- **Responsive design**

### **Customizing the Export**

You can modify the `MkDocsExporter` class to:

- Change the theme
- Add custom CSS/JS
- Modify page templates
- Adjust navigation structure
- Add custom plugins

## 🔍 RAG Integration Benefits

### **Why This Helps with RAG**

1. **Clean Markdown**: Perfect format for RAG ingestion
2. **Structured Content**: Each chunk is a separate, focused document
3. **Rich Metadata**: Quality scores, processing info, and relationships
4. **Organized Structure**: Easy to navigate and understand content
5. **Searchable**: MkDocs provides excellent search capabilities

### **RAG Workflow**

```
1. Process documents → Get markdown + metadata
2. Ingest into RAG system → Clean, structured content
3. Use MkDocs site → Human-readable documentation
4. Query RAG system → Get relevant chunks with context
```

## 🧪 Testing the Integration

### **Run the Demo**

```bash
python demo_mkdocs.py
```

This will:
- Process a test document
- Export it to MkDocs format
- Show you the generated structure
- Provide instructions for viewing

### **Manual Testing**

```bash
# Process a document
python -c "
from src.processor import DocumentProcessor
processor = DocumentProcessor()
result = processor.process_document_with_mkdocs('documents/test_document.txt')
print(f'Success: {result.success}')
print(f'MkDocs export: {result.metadata.get(\"mkdocs_export\", {})}')
"

# Check output
ls -la output/mkdocs/
```

## 🚨 Troubleshooting

### **Common Issues**

1. **MkDocs not installed**: `pip install mkdocs mkdocs-material`
2. **Output directory not created**: Check file permissions
3. **Navigation not updating**: Ensure `_navigation.md` is writable
4. **Config file missing**: Check if exporter initialization failed

### **Debug Mode**

Enable debug logging to see detailed export information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🔮 Future Enhancements

### **Planned Features**

- **Template customization** for different document types
- **Automatic site building** after export
- **Git integration** for version control
- **Custom themes** and styling options
- **Batch export optimization** for large document sets

### **Integration Possibilities**

- **GitHub Pages** deployment
- **Netlify** integration
- **Custom RAG frameworks** integration
- **Document versioning** and history

## 📚 Additional Resources

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [RAGPrep Main Documentation](README.md)
- [API Reference](docs/api/)

---

**🎉 You now have a complete document processing pipeline that automatically creates beautiful, RAG-ready documentation sites!**
