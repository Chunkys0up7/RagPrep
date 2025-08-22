# 🎨 RAG Document Processing Dashboard

A comprehensive, user-friendly web interface for the RAG Document Processing Utility that provides real-time visualization, document management, and processing control.

## ✨ Features

### 🏠 **Dashboard Overview**
- **Real-time Metrics**: View processing statistics, file counts, and system status
- **Quick Actions**: Easy access to upload, processing, and results pages
- **System Health**: Monitor overall system status and performance

### 📤 **Document Upload**
- **Multi-format Support**: Upload TXT, PDF, DOCX, HTML, and MD files
- **Batch Processing**: Process multiple documents simultaneously
- **Processing Options**: Configure chunk size, overlap, security scanning, and quality assessment
- **MkDocs Integration**: Automatically generate documentation sites

### ⚙️ **Processing Control**
- **Real-time Progress**: Visual progress bars and status updates
- **File Status Tracking**: Monitor individual file processing status
- **Processing Controls**: Pause, resume, and stop processing operations
- **Live Logs**: View processing logs and error messages

### 📊 **Results & Analytics**
- **Processing Results**: Detailed view of all processed documents
- **Performance Metrics**: Processing time, chunk counts, and success rates
- **Data Export**: Download results as CSV files
- **MkDocs Access**: Direct links to generated documentation sites

### 🔧 **Configuration Management**
- **Settings Panel**: View and modify system configuration
- **Processing Parameters**: Adjust chunking, security, and quality settings
- **Path Management**: Configure input, output, and temporary directories

## 🚀 Quick Start

### 1. **Install Dependencies**
```bash
pip install -r requirements-ui.txt
```

### 2. **Launch Dashboard**
```bash
python run_dashboard.py
```

### 3. **Access Dashboard**
The dashboard will automatically open in your browser at:
```
http://localhost:8501
```

## 📱 Dashboard Navigation

### **🏠 Dashboard (Home)**
- **Overview Cards**: Upload counts, processing status, output metrics
- **Quick Actions**: Start processing, upload documents, view results
- **System Status**: Overall health and performance indicators

### **📤 Upload Documents**
- **File Upload**: Drag & drop or browse for documents
- **Format Validation**: Automatic file type detection and validation
- **Processing Options**: Configure chunking, security, and export settings
- **Batch Configuration**: Set processing parameters for multiple files

### **⚙️ Processing**
- **Progress Tracking**: Real-time progress bars and status updates
- **File Status Table**: Individual file processing status
- **Processing Controls**: Pause, resume, and stop operations
- **Live Logs**: Real-time processing logs and messages

### **📊 Results**
- **Results Overview**: Processing statistics and success rates
- **Detailed Results**: Complete processing history and metadata
- **Data Export**: Download results in various formats
- **MkDocs Integration**: Access generated documentation sites

### **🔧 Settings**
- **Configuration Display**: Current system settings
- **Parameter Adjustment**: Modify processing parameters
- **Path Configuration**: Set input/output directories
- **System Preferences**: Log levels and performance settings

## 🎯 Use Cases

### **📚 Academic Research**
- Process research papers and academic documents
- Generate searchable knowledge bases
- Create documentation sites for research projects

### **🏢 Business Intelligence**
- Process business reports and documents
- Extract key information and insights
- Generate searchable corporate knowledge bases

### **📖 Content Management**
- Organize and categorize documents
- Generate documentation sites
- Create searchable content repositories

### **🔍 Information Retrieval**
- Build RAG systems for document search
- Generate embeddings for semantic search
- Create knowledge graphs from documents

## 🛠️ Technical Details

### **Frontend Framework**
- **Streamlit**: Modern, responsive web interface
- **Plotly**: Interactive charts and visualizations
- **Pandas**: Data manipulation and display

### **Backend Integration**
- **Document Processing**: Full integration with RAG processor
- **Real-time Updates**: Live status and progress tracking
- **File Management**: Secure file upload and processing
- **Configuration**: Dynamic settings management

### **Data Visualization**
- **Progress Bars**: Real-time processing status
- **Metrics Cards**: Key performance indicators
- **Status Tables**: Detailed file and processing information
- **Charts**: Processing analytics and trends

## 🔧 Configuration Options

### **Processing Parameters**
- **Chunk Size**: Number of characters per text chunk (100-2000)
- **Chunk Overlap**: Overlap between consecutive chunks (0-200)
- **Security Scanning**: Enable/disable security threat detection
- **Quality Assessment**: Enable/disable content quality analysis

### **Output Options**
- **MkDocs Export**: Generate documentation sites
- **Vector Storage**: Store processed chunks for retrieval
- **Metadata Extraction**: Extract and store document metadata
- **Format Conversion**: Convert to various output formats

### **System Settings**
- **Input/Output Paths**: Configure directory locations
- **Log Levels**: Set logging verbosity
- **Performance**: Optimize processing speed and memory usage
- **Security**: Configure security scanning parameters

## 📊 Performance Monitoring

### **Real-time Metrics**
- **Processing Speed**: Documents per minute
- **Memory Usage**: System resource consumption
- **Success Rates**: Processing success percentages
- **Error Tracking**: Failed operations and error messages

### **System Health**
- **Resource Usage**: CPU, memory, and disk utilization
- **Processing Queue**: Number of pending documents
- **Error Rates**: System error and failure rates
- **Performance Trends**: Historical performance data

## 🚨 Troubleshooting

### **Common Issues**

#### **Dashboard Won't Start**
```bash
# Check dependencies
pip install -r requirements-ui.txt

# Verify Python version
python --version  # Should be 3.8+

# Check file permissions
ls -la run_dashboard.py
```

#### **Upload Errors**
- Verify file formats are supported
- Check file size limits
- Ensure sufficient disk space
- Verify write permissions

#### **Processing Failures**
- Check log files for error messages
- Verify input file integrity
- Check system resources
- Review configuration settings

### **Performance Issues**
- Reduce chunk size for large documents
- Increase system memory allocation
- Optimize disk I/O performance
- Monitor system resource usage

## 🔒 Security Features

### **File Validation**
- **Format Checking**: Validate file types and extensions
- **Content Scanning**: Detect malicious content and scripts
- **Size Limits**: Prevent oversized file uploads
- **Path Traversal**: Block directory traversal attempts

### **Processing Security**
- **Sandboxed Processing**: Isolate document processing
- **Content Analysis**: Scan for security threats
- **Access Control**: Restrict file access permissions
- **Audit Logging**: Track all processing activities

## 📈 Future Enhancements

### **Planned Features**
- **Advanced Analytics**: Machine learning insights
- **Collaborative Processing**: Multi-user support
- **API Integration**: REST API for external access
- **Cloud Deployment**: AWS/Azure deployment options

### **User Experience**
- **Dark Mode**: Theme customization
- **Mobile Support**: Responsive mobile interface
- **Accessibility**: Screen reader and keyboard navigation
- **Internationalization**: Multi-language support

## 🤝 Contributing

### **Development Setup**
```bash
# Clone repository
git clone <repository-url>
cd RAGPrep

# Install development dependencies
pip install -r requirements-ui.txt
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Launch dashboard
python run_dashboard.py
```

### **Code Style**
- Follow PEP 8 Python style guidelines
- Use type hints for function parameters
- Add docstrings for all functions and classes
- Include unit tests for new features

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

### **Getting Help**
- **Documentation**: Check this README and code comments
- **Issues**: Report bugs and feature requests via GitHub issues
- **Discussions**: Join community discussions for help and ideas

### **Contact Information**
- **GitHub**: [Repository Issues](https://github.com/your-repo/issues)
- **Email**: [Support Email]
- **Documentation**: [Project Wiki]

---

**🎉 Enjoy using the RAG Document Processing Dashboard!**

Transform your documents into searchable knowledge with our powerful, user-friendly interface.
