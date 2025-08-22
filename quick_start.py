#!/usr/bin/env python3
"""
RAG Document Processing UI - Quick Start
Easy launcher for all UI options
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Quick start launcher for RAG UI."""
    print("🎨 RAG Document Processing UI - Quick Start")
    print("=" * 50)
    
    # Check if streamlit is available
    try:
        import streamlit
        print("✅ Streamlit is available")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements-ui.txt"])
        print("✅ Dependencies installed")
    
    print("\n🚀 Choose your UI option:")
    print("1. 🎯 Full Dashboard (Complete RAG processing interface)")
    print("2. 🎪 Demo UI (Simplified demo with sample data)")
    print("3. 📚 MkDocs Site (Generated documentation)")
    print("4. 🔧 API Server (REST API endpoints)")
    print("5. 📖 Help & Documentation")
    print("6. ❌ Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == "1":
                launch_full_dashboard()
                break
            elif choice == "2":
                launch_demo_ui()
                break
            elif choice == "3":
                launch_mkdocs()
                break
            elif choice == "4":
                launch_api_server()
                break
            elif choice == "5":
                show_help()
                break
            elif choice == "6":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-6.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def launch_full_dashboard():
    """Launch the full RAG dashboard."""
    print("\n🚀 Launching Full RAG Dashboard...")
    
    dashboard_path = Path(__file__).parent / "src" / "ui" / "dashboard.py"
    
    if not dashboard_path.exists():
        print(f"❌ Dashboard not found at {dashboard_path}")
        print("💡 Try running the demo UI instead (option 2)")
        return
    
    print(f"📁 Dashboard found at: {dashboard_path}")
    print("🌐 Starting Streamlit server...")
    print("📱 The dashboard will open in your browser automatically")
    print("🔗 If it doesn't open, go to: http://localhost:8501")
    print("\n💡 Press Ctrl+C to stop the dashboard")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(dashboard_path),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")

def launch_demo_ui():
    """Launch the demo UI."""
    print("\n🎪 Launching Demo UI...")
    
    demo_path = Path(__file__).parent / "demo_ui.py"
    
    if not demo_path.exists():
        print(f"❌ Demo UI not found at {demo_path}")
        return
    
    print(f"📁 Demo UI found at: {demo_path}")
    print("🌐 Starting Streamlit server...")
    print("📱 The demo will open in your browser automatically")
    print("🔗 If it doesn't open, go to: http://localhost:8502")
    print("\n💡 Press Ctrl+C to stop the demo")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(demo_path),
            "--server.port", "8502",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
    except Exception as e:
        print(f"❌ Error launching demo: {e}")

def launch_mkdocs():
    """Launch MkDocs site."""
    print("\n📚 Launching MkDocs Site...")
    
    mkdocs_path = Path(__file__).parent / "output" / "mkdocs"
    
    if not mkdocs_path.exists():
        print(f"❌ MkDocs site not found at {mkdocs_path}")
        print("💡 Process some documents first to generate the site")
        return
    
    print(f"📁 MkDocs site found at: {mkdocs_path}")
    print("🌐 Starting MkDocs server...")
    print("📱 The documentation site will open in your browser")
    print("🔗 If it doesn't open, go to: http://localhost:8000")
    print("\n💡 Press Ctrl+C to stop the server")
    
    try:
        subprocess.run([
            sys.executable, "-m", "mkdocs", "serve",
            "-f", str(mkdocs_path / "mkdocs.yml"),
            "--dev-addr", "localhost:8000"
        ])
    except KeyboardInterrupt:
        print("\n👋 MkDocs server stopped by user")
    except Exception as e:
        print(f"❌ Error launching MkDocs: {e}")

def launch_api_server():
    """Launch the API server."""
    print("\n🔧 Launching API Server...")
    
    api_path = Path(__file__).parent / "src" / "api.py"
    
    if not api_path.exists():
        print(f"❌ API server not found at {api_path}")
        return
    
    print(f"📁 API server found at: {api_path}")
    print("🌐 Starting FastAPI server...")
    print("📱 The API will be available at: http://localhost:8000")
    print("📖 API documentation at: http://localhost:8000/docs")
    print("\n💡 Press Ctrl+C to stop the server")
    
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", "src.api:app",
            "--host", "localhost",
            "--port", "8000",
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n👋 API server stopped by user")
    except Exception as e:
        print(f"❌ Error launching API server: {e}")

def show_help():
    """Show help and documentation."""
    print("\n📖 RAG Document Processing UI - Help & Documentation")
    print("=" * 60)
    
    print("\n🎯 **What is this?**")
    print("A comprehensive web interface for processing documents and building RAG (Retrieval-Augmented Generation) systems.")
    
    print("\n🚀 **Quick Start Options:**")
    print("1. **Full Dashboard**: Complete interface with real processing capabilities")
    print("2. **Demo UI**: Simplified version with sample data for testing")
    print("3. **MkDocs Site**: Generated documentation from processed documents")
    print("4. **API Server**: REST API endpoints for programmatic access")
    
    print("\n📱 **Features:**")
    print("• Document upload and processing")
    print("• Real-time progress tracking")
    print("• Results visualization and analytics")
    print("• Configuration management")
    print("• MkDocs integration")
    
    print("\n🔧 **Requirements:**")
    print("• Python 3.8+")
    print("• Dependencies installed (pip install -r requirements-ui.txt)")
    print("• Streamlit framework")
    
    print("\n📚 **Documentation:**")
    print("• UI_README.md - Comprehensive usage guide")
    print("• UI_SUMMARY.md - Implementation summary")
    print("• README.md - Main project documentation")
    
    print("\n💡 **Tips:**")
    print("• Start with the Demo UI to explore features")
    print("• Use the Full Dashboard for actual document processing")
    print("• Check the MkDocs site for generated documentation")
    print("• Use the API for programmatic access")
    
    print("\n🆘 **Getting Help:**")
    print("• Check the documentation files")
    print("• Review error messages in the terminal")
    print("• Ensure all dependencies are installed")
    print("• Verify file paths and permissions")

if __name__ == "__main__":
    main()
