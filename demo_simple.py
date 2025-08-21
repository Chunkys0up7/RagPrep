#!/usr/bin/env python3
"""
RAG Document Processing Utility - Simplified Demo

This is a simplified demo that works with minimal dependencies.
It demonstrates the core functionality without requiring heavy ML libraries.
"""

import os
import sys
import time
import logging
from pathlib import Path

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n--- {title} ---")

def check_python_version():
    """Check Python version compatibility."""
    print_section("Python Version Check")
    
    version = sys.version_info
    print(f"Python Version: {version.major}.{version.minor}.{version.micro}")
    
    if version < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    
    print("✅ Python version is compatible")
    return True

def check_dependencies():
    """Check if core dependencies are available."""
    print_section("Dependency Check")
    
    required_packages = [
        'pydantic', 'pydantic_settings', 'yaml', 'pathlib2'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'pydantic_settings':
                __import__('pydantic_settings')
            elif package == 'yaml':
                __import__('yaml')
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - Missing")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("   Install with: pip install -r requirements_minimal.txt")
        return False
    
    print("✅ All core dependencies are available")
    return True

def check_project_structure():
    """Check if project structure is correct."""
    print_section("Project Structure Check")
    
    required_dirs = ['src', 'config', 'documents', 'tests']
    required_files = ['src/__init__.py', 'config/config.yaml', 'documents/test_document.txt']
    
    # Check directories
    for directory in required_dirs:
        if Path(directory).exists():
            print(f"✅ Directory: {directory}/")
        else:
            print(f"❌ Directory: {directory}/ - Missing")
            return False
    
    # Check files
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ File: {file_path}")
        else:
            print(f"❌ File: {file_path} - Missing")
            return False
    
    print("✅ Project structure is correct")
    return True

def check_test_documents():
    """Check if test documents exist and are readable."""
    print_section("Test Documents Check")
    
    documents_dir = Path("documents")
    test_files = list(documents_dir.glob("*.txt")) + list(documents_dir.glob("*.html"))
    
    if not test_files:
        print("❌ No test documents found")
        return False
    
    print(f"✅ Found {len(test_files)} test documents:")
    for file in test_files:
        try:
            # Try to read the first few lines
            with open(file, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                print(f"   - {file.name}: {first_line[:50]}...")
        except Exception as e:
            print(f"   - {file.name}: Error reading file - {e}")
            return False
    
    return True

def demo_basic_functionality():
    """Demonstrate basic functionality without heavy dependencies."""
    print_section("Basic Functionality Demo")
    
    try:
        # Test configuration loading
        print("Testing configuration system...")
        from src.config import Config
        
        config = Config()
        print(f"✅ Configuration loaded: {config.app_name}")
        print(f"   Version: {config.version}")
        
        # Test security configuration (may be limited without full dependencies)
        try:
            security_enabled = config.security.enable_file_validation
            print(f"   Security enabled: {security_enabled}")
        except Exception as e:
            print(f"   Security config: Limited (missing dependencies)")
        
        # Test basic imports
        print("\nTesting core module imports...")
        from src.parsers import get_document_parser
        from src.chunkers import get_document_chunker
        from src.metadata_extractors import get_metadata_extractor
        
        print("✅ Core modules imported successfully")
        
        # Test document parser creation
        print("\nTesting document parser creation...")
        parser = get_document_parser()
        print(f"✅ Parser created: {type(parser).__name__}")
        
        # Test chunker creation
        print("\nTesting document chunker creation...")
        chunker = get_document_chunker("structural")
        print(f"✅ Chunker created: {type(chunker).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        logger.exception("Error in basic functionality test")
        return False

def create_sample_output():
    """Create sample output to demonstrate the system."""
    print_section("Creating Sample Output")
    
    try:
        # Create output directories
        output_dirs = ['output', 'output/chunks', 'output/metadata', 'vector_db']
        for dir_path in output_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {dir_path}")
        
        # Create sample output files
        sample_chunk = {
            "chunk_id": "sample_chunk_001",
            "content": "This is a sample document chunk demonstrating the RAG Document Processing Utility.",
            "chunk_type": "text",
            "quality_score": 0.95,
            "metadata": {
                "source": "test_document.txt",
                "chunk_number": 1,
                "word_count": 15
            }
        }
        
        import json
        with open('output/chunks/sample_chunk.json', 'w') as f:
            json.dump(sample_chunk, f, indent=2)
        print("✅ Created sample chunk file")
        
        # Create sample metadata
        sample_metadata = {
            "document_id": "test_doc_001",
            "total_chunks": 1,
            "processing_time": 2.5,
            "quality_score": 0.95,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open('output/metadata/sample_metadata.json', 'w') as f:
            json.dump(sample_metadata, f, indent=2)
        print("✅ Created sample metadata file")
        
        return True
        
    except Exception as e:
        print(f"❌ Sample output creation failed: {e}")
        return False

def main():
    """Main simplified demo function."""
    print_header("RAG Document Processing Utility - Simplified Demo")
    print("This demo works with minimal dependencies and shows core functionality.")
    
    # Run checks
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Project Structure", check_project_structure),
        ("Test Documents", check_test_documents),
    ]
    
    results = []
    
    for check_name, check_func in checks:
        try:
            success = check_func()
            results.append((check_name, success))
            
            if not success:
                print(f"\n❌ {check_name} check failed. Please fix the issues above.")
                break
                
        except Exception as e:
            print(f"❌ {check_name} check crashed: {e}")
            results.append((check_name, False))
            break
    
    # If all checks passed, run functionality demo
    if all(success for _, success in results):
        print("\n🎯 All checks passed! Testing basic functionality...")
        
        if demo_basic_functionality():
            print("\n🎯 Basic functionality test passed! Creating sample output...")
            
            if create_sample_output():
                print("\n🎉 Simplified demo completed successfully!")
                print("\n📁 Sample output created in:")
                print("   - output/chunks/sample_chunk.json")
                print("   - output/metadata/sample_metadata.json")
                print("   - vector_db/ (directory)")
            else:
                print("\n⚠️  Sample output creation failed")
        else:
            print("\n❌ Basic functionality test failed")
    
    # Summary
    print_header("Demo Summary")
    successful = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"📊 Check Results: {successful}/{total} checks passed")
    
    for check_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status}: {check_name}")
    
    if successful == total:
        print("\n🎉 The system is ready for basic functionality!")
        print("   To run the full demo, install all dependencies:")
        print("   pip install -r requirements.txt")
    else:
        print(f"\n⚠️  {total - successful} check(s) failed. Please fix the issues above.")
    
    print(f"\n📝 Check the output above for any issues that need to be resolved.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        logger.exception("Unexpected error in simplified demo")
