#!/usr/bin/env python3
"""
RAG Document Processing Dashboard Launcher
Run this script to start the Streamlit dashboard.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the RAG dashboard."""
    print("🚀 Launching RAG Document Processing Dashboard...")
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("✅ Streamlit is installed")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements-ui.txt"])
        print("✅ Dependencies installed")
    
    # Get the dashboard path
    dashboard_path = Path(__file__).parent / "src" / "ui" / "dashboard.py"
    
    if not dashboard_path.exists():
        print(f"❌ Dashboard not found at {dashboard_path}")
        return
    
    print(f"📁 Dashboard found at: {dashboard_path}")
    print("🌐 Starting Streamlit server...")
    print("📱 The dashboard will open in your browser automatically")
    print("🔗 If it doesn't open, go to: http://localhost:8501")
    print("\n💡 Press Ctrl+C to stop the dashboard")
    
    # Launch streamlit
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

if __name__ == "__main__":
    main()
