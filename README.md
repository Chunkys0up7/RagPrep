# RAG Document Processing Utility

A sophisticated document processing utility that transforms various document formats into optimized structures for RAG (Retrieval-Augmented Generation) applications. The system uses LLM-powered metadata extraction, intelligent chunking, and multimodal content handling.

## 🚀 Features

- **Multi-format Support**: PDF, DOCX, HTML, and more
- **Intelligent Chunking**: Semantic, structural, and fixed-size chunking strategies
- **LLM-Powered Metadata**: Advanced extraction using language models
- **Multimodal Processing**: Tables, images, and mathematical content handling
- **Quality Assurance**: Comprehensive validation at every processing stage
- **Vector Database Integration**: Multi-index strategy for efficient retrieval
- **Automated Pipeline**: GitHub Actions integration for continuous processing

## 🏗️ Architecture

The utility is built around a **multi-stage, intelligence-enhanced pipeline** that treats document processing as a sophisticated understanding task rather than simple text extraction.

### Core Components

1. **Document Parsing**: Cascading parser strategy with PyMuPDF, Marker, and Unstructured
2. **Intelligent Chunking**: Hybrid approach combining semantic, structural, and fixed-size methods
3. **Metadata Enhancement**: LLM-powered extraction of entities, topics, and relationships
4. **Multimodal Handling**: Advanced processing of tables, images, and mathematical content
5. **Vector Indexing**: Multi-index strategy for semantic search and metadata filtering
6. **Quality Assurance**: Multi-level validation and continuous improvement

## 📁 Project Structure

```
RAGPrep/
├── notebooks/           # Jupyter notebooks for development
├── src/                # Core source code
├── config/             # Configuration files and schemas
├── docs/               # MkDocs documentation
├── tests/              # Test suite
├── scripts/            # Automation scripts
├── requirements.txt    # Python dependencies
├── README.md          # Project overview
└── .github/           # GitHub Actions workflows
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd RAGPrep
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

5. **Launch Jupyter**
   ```bash
   jupyter lab
   ```

## 🚀 Quick Start

1. **Open the development notebook**
   - Navigate to `notebooks/01_document_analysis.ipynb`
   - Follow the step-by-step guide

2. **Process your first document**
   - Place a test document in the `documents/` folder
   - Run the processing pipeline
   - Review quality metrics and results

3. **Explore advanced features**
   - Try different chunking strategies
   - Experiment with metadata extraction
   - Test multimodal content processing

## 📚 Documentation

- **User Guide**: `docs/user-guide/`
- **API Reference**: `docs/api/`
- **Configuration**: `docs/configuration/`
- **Development**: `docs/development/`

## 🔧 Configuration

The utility uses JSON schemas and YAML configurations for different document types:

```yaml
document_types:
  academic_paper:
    parsers: [marker, pymupdf]
    chunking_strategy: semantic
    metadata_extraction: academic_enhanced
  technical_manual:
    parsers: [unstructured, pymupdf]
    chunking_strategy: structural
    metadata_extraction: technical_focused
```

## 🧪 Testing

Run the test suite:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest --cov=src tests/
```

## 📊 Quality Metrics

The system tracks several key performance indicators:

- **Content Extraction Accuracy**: >95% target
- **Structure Preservation**: >90% target
- **Metadata Quality**: >85% target
- **Processing Speed**: <30 seconds per document
- **Memory Efficiency**: <2GB peak memory

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/RAGPrep/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/RAGPrep/discussions)
- **Documentation**: [MkDocs Site](https://yourusername.github.io/RAGPrep/)

## 🗺️ Roadmap

- [ ] Phase 1: Foundation & Environment Setup (Weeks 1-2)
- [ ] Phase 2: Core Intelligence Implementation (Weeks 3-4)
- [ ] Phase 3: Advanced Processing Features (Weeks 5-6)
- [ ] Phase 4: Production Readiness (Weeks 7-8)

---

**Built with ❤️ for the RAG community**
