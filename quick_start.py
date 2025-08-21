#!/usr/bin/env python3
"""
RAG Document Processing Utility - Quick Start Script

This script provides a simple way to get started with the RAG Document Processing Utility.
It will check prerequisites, install dependencies if needed, and run the demo.
"""

import os
import sys
import subprocess
import time
from pathlib import Path


def print_banner():
    """Print the startup banner."""
    print("=" * 70)
    print("🚀 RAG Document Processing Utility - Quick Start")
    print("=" * 70)
    print("This script will set up and run the demo automatically.")
    print("=" * 70)


def check_python():
    """Check Python version."""
    print("🔍 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print(f"❌ Python 3.8+ required. Current: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} - Compatible")
    return True


def check_pip():
    """Check if pip is available."""
    print("🔍 Checking pip availability...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, capture_output=True)
        print("✅ pip is available")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ pip not found")
        return False


def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    
    try:
        # Install core dependencies first
        core_deps = ["pydantic", "pydantic-settings", "pyyaml", "pathlib2"]
        for dep in core_deps:
            print(f"   Installing {dep}...")
            subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                          check=True, capture_output=True)
        
        # Install all dependencies from requirements.txt
        if Path("requirements.txt").exists():
            print("   Installing from requirements.txt...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                          check=True, capture_output=True)
            print("✅ All dependencies installed successfully")
        else:
            print("⚠️  requirements.txt not found, using core dependencies only")
            
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    required = ["pydantic", "pydantic_settings", "yaml", "pathlib2"]
    missing = []
    
    for dep in required:
        try:
            if dep == "pydantic_settings":
                __import__("pydantic_settings")
            elif dep == "yaml":
                __import__("yaml")
            else:
                __import__(dep)
            print(f"   ✅ {dep}")
        except ImportError:
            missing.append(dep)
            print(f"   ❌ {dep} - Missing")
    
    if missing:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
        return False
    
    print("✅ All required dependencies are available")
    return True


def setup_directories():
    """Create necessary directories."""
    print("📁 Setting up directories...")
    
    directories = [
        "output", "output/chunks", "output/metadata", "output/embeddings",
        "vector_db", "logs", "temp"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}")
    
    print("✅ All directories ready")


def check_test_documents():
    """Check if test documents exist."""
    print("📄 Checking test documents...")
    
    documents_dir = Path("documents")
    if not documents_dir.exists():
        print("   ❌ 'documents/' directory not found")
        return False
    
    test_files = list(documents_dir.glob("*.txt")) + list(documents_dir.glob("*.html"))
    if not test_files:
        print("   ❌ No test documents found")
        return False
    
    print(f"   ✅ Found {len(test_files)} test documents:")
    for file in test_files:
        print(f"      - {file.name}")
    
    return True


def run_demo():
    """Run the demo script."""
    print("🚀 Starting demo...")
    print("=" * 50)
    
    try:
        # Run the demo
        result = subprocess.run([sys.executable, "demo.py"], 
                              check=True, capture_output=False)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Demo failed with exit code: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
        return False


def main():
    """Main quick start function."""
    print_banner()
    
    # Step 1: Check Python
    if not check_python():
        print("\n❌ Python version check failed. Please upgrade Python.")
        return False
    
    # Step 2: Check pip
    if not check_pip():
        print("\n❌ pip not available. Please install pip.")
        return False
    
    # Step 3: Check dependencies
    if not check_dependencies():
        print("\n📦 Installing missing dependencies...")
        if not install_dependencies():
            print("❌ Failed to install dependencies. Please install manually:")
            print("   pip install -r requirements.txt")
            return False
        
        # Re-check after installation
        if not check_dependencies():
            print("❌ Dependencies still missing after installation")
            return False
    
    # Step 4: Setup directories
    setup_directories()
    
    # Step 5: Check test documents
    if not check_test_documents():
        print("\n❌ Test documents not found. Please ensure 'documents/' directory exists with test files.")
        return False
    
    # Step 6: Run demo
    print("\n🎯 All checks passed! Starting demo...")
    time.sleep(2)
    
    success = run_demo()
    
    if success:
        print("\n🎉 Demo completed successfully!")
        print("\n📁 Check the following directories for output:")
        print("   - vector_db/     - Vector store data")
        print("   - output/        - Processed documents")
        print("   - demo.log       - Demo execution log")
    else:
        print("\n❌ Demo failed. Check the output above for errors.")
    
    return success


if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Quick start interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
