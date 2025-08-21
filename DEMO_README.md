# 🚀 RAG Document Processing Utility - Demo Guide

This guide will walk you through setting up and running the RAG Document Processing Utility demo to validate all the critical bug fixes and showcase the system's capabilities.

## 📋 Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: At least 2GB RAM available
- **Disk Space**: At least 500MB free space

### Required Software
- Python 3.8+ with pip
- Git (for cloning the repository)

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Chunkys0up7/RagPrep.git
cd RagPrep
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**Note**: Some dependencies may take a while to install, especially PyTorch and transformers. This is normal.

### 3. Verify Installation
```bash
python -c "import pydantic, yaml, pathlib2; print('✅ Dependencies installed successfully')"
```

## 🧪 Test Data

The demo includes comprehensive test documents:

- **`documents/test_document.txt`** - Comprehensive text document with various content types
- **`documents/test_document.html`** - HTML document to test HTML parsing capabilities

These documents are designed to test:
- Text parsing and structure extraction
- Multiple chunking strategies (fixed-size, structural, semantic, hybrid)
- Metadata extraction capabilities
- Quality assessment accuracy
- Overall pipeline performance

## 🚀 Running the Demo

### Option 1: Automated Startup (Recommended)
```bash
python start_demo.py
```

This script will:
- ✅ Check Python version compatibility
- ✅ Verify all dependencies are installed
- ✅ Ensure test documents exist
- ✅ Create necessary directories
- ✅ Automatically start the demo

### Option 2: Manual Demo
```bash
python demo.py
```

## 📊 What the Demo Shows

### 1. Configuration System
- Pydantic-based configuration management
- Environment variable support
- Configuration validation
- Security settings

### 2. Security System
- File validation and sanitization
- Content threat analysis
- Security profile assessment
- Threat level classification

### 3. Document Processing Pipeline
- Document parsing (TXT, HTML)
- Intelligent chunking strategies
- Metadata extraction
- Quality assessment
- Vector storage

### 4. Batch Processing
- Multiple document processing
- Performance metrics
- Quality scoring
- Error handling

### 5. Quality Assessment
- Multi-dimensional quality metrics
- Performance monitoring
- Continuous improvement tracking

### 6. Vector Storage
- File-based storage implementation
- Document and chunk management
- Storage statistics

## 🔍 Expected Results

### Successful Demo Run
```
🎉 All demos completed successfully! The system is working correctly.

📊 Overall Results: 6/6 demos successful
   ✅ PASS: Configuration System
   ✅ PASS: Security System
   ✅ PASS: Document Processing Pipeline
   ✅ PASS: Batch Processing
   ✅ PASS: Quality Assessment
   ✅ PASS: Vector Storage
```

### Quality Metrics
- **Document Processing**: Should complete in < 30 seconds
- **Quality Score**: Should exceed 0.8 (80%)
- **Chunk Creation**: Should generate meaningful, well-structured chunks
- **Security**: Should pass all security checks without warnings

## 📁 Output Files

After running the demo, you'll find:

- **`demo.log`** - Detailed demo execution log
- **`vector_db/`** - Vector store data and metadata
- **`output/`** - Processed document chunks and metadata
- **`logs/`** - System logs and performance metrics

## 🐛 Troubleshooting

### Common Issues

#### 1. Import Errors
```
ModuleNotFoundError: No module named 'pydantic'
```
**Solution**: Install dependencies with `pip install -r requirements.txt`

#### 2. Python Version Issues
```
❌ Python 3.8 or higher is required
```
**Solution**: Upgrade to Python 3.8+ or use a virtual environment

#### 3. Missing Test Documents
```
❌ No test documents found in 'documents/' directory
```
**Solution**: Ensure the `documents/` directory exists with test files

#### 4. Permission Errors
```
PermissionError: [Errno 13] Permission denied
```
**Solution**: Run with appropriate permissions or check directory access

#### 5. Memory Issues
```
MemoryError: Unable to allocate array
```
**Solution**: Close other applications or increase system memory

### Debug Mode

For detailed debugging, set the log level to DEBUG:

```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

### Manual Testing

If the demo fails, you can test individual components:

```python
# Test configuration
from src.config import Config
config = Config()
print("Config loaded:", config.app_name)

# Test security
from src.security import SecurityManager
security = SecurityManager(config)
print("Security manager initialized")

# Test processor
from src.processor import DocumentProcessor
processor = DocumentProcessor()
print("Processor initialized")
```

## 🔧 Advanced Configuration

### Environment Variables
Create a `.env` file for custom configuration:

```bash
# .env
OPENAI_API_KEY=your_openai_key_here
DEBUG=true
LOG_LEVEL=DEBUG
```

### Custom Test Documents
Add your own test documents to the `documents/` directory:

- **Text files**: `.txt` extension
- **HTML files**: `.html` extension
- **Markdown files**: `.md` extension

## 📈 Performance Monitoring

The demo includes comprehensive performance monitoring:

- **Processing Time**: Per-document and batch processing times
- **Memory Usage**: Real-time memory consumption tracking
- **Quality Metrics**: Continuous quality assessment
- **Success Rates**: Operation success/failure tracking

## 🎯 Demo Validation Checklist

After running the demo, verify:

- [ ] All 6 demo sections complete successfully
- [ ] Test documents are processed without errors
- [ ] Quality scores exceed 0.8 (80%)
- [ ] Vector storage contains processed chunks
- [ ] Security checks pass without warnings
- [ ] Performance metrics are reasonable
- [ ] Output files are generated correctly

## 🚀 Next Steps

Once the demo runs successfully:

1. **Explore the Code**: Review the implementation in `src/` directory
2. **Customize Configuration**: Modify `config/config.yaml` for your needs
3. **Add Your Documents**: Process your own documents through the pipeline
4. **Extend Functionality**: Add new parsers, chunkers, or extractors
5. **Production Deployment**: Deploy to production with proper security measures

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review the `demo.log` file for detailed error information
3. Verify all prerequisites are met
4. Check the GitHub repository for updates and issues

## 🎉 Success!

Congratulations! You've successfully run the RAG Document Processing Utility demo. This validates that all critical bug fixes are working correctly and the system is ready for production use.

The demo demonstrates:
- ✅ Robust configuration management
- ✅ Comprehensive security features
- ✅ Intelligent document processing
- ✅ Quality assessment and monitoring
- ✅ Vector storage capabilities
- ✅ Batch processing efficiency

Your RAG Document Processing Utility is now ready to process real documents and integrate with your RAG applications!
