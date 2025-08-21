#!/usr/bin/env python3
"""
Setup script for RAG Document Processing Utility

This script helps set up the development environment and install dependencies.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def create_directories():
    """Create necessary directories."""
    directories = [
        "logs",
        "output/chunks",
        "output/metadata",
        "output/embeddings",
        "temp",
        "vector_db"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")


def setup_virtual_environment():
    """Set up virtual environment."""
    if not Path("venv").exists():
        print("🔄 Creating virtual environment...")
        if run_command("python -m venv venv", "Creating virtual environment"):
            print("✅ Virtual environment created")
        else:
            return False
    else:
        print("✅ Virtual environment already exists")
    
    return True


def install_dependencies():
    """Install Python dependencies."""
    # Determine the correct pip command
    if os.name == 'nt':  # Windows
        pip_cmd = "venv\\Scripts\\pip"
    else:  # Unix/Linux/macOS
        pip_cmd = "venv/bin/pip"
    
    # Upgrade pip first
    if not run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install requirements
    if not run_command(f"{pip_cmd} install -r requirements.txt", "Installing dependencies"):
        return False
    
    return True


def setup_environment_file():
    """Set up environment configuration file."""
    env_file = Path(".env")
    env_example = Path("env.example")
    
    if not env_file.exists() and env_example.exists():
        shutil.copy(env_example, env_file)
        print("✅ Created .env file from template")
        print("⚠️  Please edit .env file with your API keys and configuration")
    elif env_file.exists():
        print("✅ .env file already exists")
    else:
        print("⚠️  No env.example file found, .env file not created")


def setup_git_hooks():
    """Set up Git hooks for code quality."""
    hooks_dir = Path(".git/hooks")
    if hooks_dir.exists():
        # Create pre-commit hook
        pre_commit_hook = hooks_dir / "pre-commit"
        hook_content = """#!/bin/sh
# Pre-commit hook for RAGPrep
echo "Running pre-commit checks..."

# Run tests
python -m pytest tests/ -v --tb=short

# Run linting (if available)
if command -v flake8 >/dev/null 2>&1; then
    flake8 src/ tests/
fi

echo "Pre-commit checks completed"
"""
        
        with open(pre_commit_hook, 'w') as f:
            f.write(hook_content)
        
        # Make executable
        os.chmod(pre_commit_hook, 0o755)
        print("✅ Git pre-commit hook created")
    else:
        print("⚠️  Not a Git repository, skipping Git hooks")


def run_tests():
    """Run basic tests to verify setup."""
    print("🧪 Running basic tests...")
    
    # Determine the correct python command
    if os.name == 'nt':  # Windows
        python_cmd = "venv\\Scripts\\python"
    else:  # Unix/Linux/macOS
        python_cmd = "venv/bin/python"
    
    if run_command(f"{python_cmd} -m pytest tests/test_config.py -v", "Running configuration tests"):
        print("✅ Basic tests passed")
        return True
    else:
        print("❌ Basic tests failed")
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up RAG Document Processing Utility")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Set up virtual environment
    if not setup_virtual_environment():
        print("❌ Failed to set up virtual environment")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Set up environment file
    setup_environment_file()
    
    # Set up Git hooks
    setup_git_hooks()
    
    # Run tests
    if not run_tests():
        print("❌ Setup verification failed")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Edit .env file with your API keys")
    print("2. Place test documents in the documents/ folder")
    print("3. Open notebooks/01_document_analysis.ipynb to get started")
    print("4. Run 'python -m pytest tests/' to run all tests")
    
    print("\n🔧 To activate the virtual environment:")
    if os.name == 'nt':  # Windows
        print("   venv\\Scripts\\activate")
    else:  # Unix/Linux/macOS
        print("   source venv/bin/activate")


if __name__ == "__main__":
    main()
