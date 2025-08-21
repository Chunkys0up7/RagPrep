#!/usr/bin/env python3
"""
RAG Document Processing Utility - Startup Script

This script checks prerequisites and starts the demo system.
"""

import os
import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True


def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'pydantic', 'pydantic-settings', 'pyyaml', 'pathlib2'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - Missing")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("   Install with: pip install -r requirements.txt")
        return False
    
    return True


def check_test_documents():
    """Check if test documents exist."""
    documents_dir = Path("documents")
    if not documents_dir.exists():
        print("❌ 'documents/' directory not found")
        return False
    
    test_files = list(documents_dir.glob("*.txt")) + list(documents_dir.glob("*.html"))
    if not test_files:
        print("❌ No test documents found in 'documents/' directory")
        return False
    
    print(f"✅ Found {len(test_files)} test documents:")
    for file in test_files:
        print(f"   - {file.name}")
    
    return True


def check_directories():
    """Check and create necessary directories."""
    directories = [
        "output",
        "output/chunks", 
        "output/metadata",
        "output/embeddings",
        "vector_db",
        "logs",
        "temp"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Directory ready: {directory}")


def install_dependencies():
    """Install dependencies if needed."""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def main():
    """Main startup function."""
    print("🚀 RAG Document Processing Utility - Startup Check")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check dependencies
    print("\n🔍 Checking dependencies...")
    if not check_dependencies():
        print("\n💡 Attempting to install dependencies...")
        if not install_dependencies():
            return False
        # Re-check after installation
        if not check_dependencies():
            return False
    
    # Check test documents
    print("\n📄 Checking test documents...")
    if not check_test_documents():
        return False
    
    # Setup directories
    print("\n📁 Setting up directories...")
    check_directories()
    
    print("\n✅ All checks passed! Starting demo...")
    print("=" * 50)
    
    # Start the demo
    try:
        subprocess.run([sys.executable, "demo.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Demo failed to start: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Startup failed. Please check the errors above.")
        sys.exit(1)
