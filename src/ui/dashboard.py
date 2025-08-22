"""
RAG Document Processing Dashboard
A comprehensive UI for visualizing and managing document processing workflows.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import time
from datetime import datetime
import os
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.processor import DocumentProcessor
from src.config import Config
from src.mkdocs_exporter import MkDocsExporter


class RAGDashboard:
    """Main dashboard class for RAG document processing."""
    
    def __init__(self):
        """Initialize the dashboard."""
        st.set_page_config(
            page_title="RAG Document Processing Dashboard",
            page_icon="📚",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Initialize components
        self.config = self._load_config()
        self.processor = DocumentProcessor(self.config)
        self.mkdocs_exporter = MkDocsExporter(self.config)
        
        # Session state
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = []
        if 'processing_results' not in st.session_state:
            st.session_state.processing_results = []
        
        self._setup_sidebar()
        self._setup_main_content()
    
    def _load_config(self):
        """Load configuration with defaults."""
        try:
            return Config()
        except Exception:
            # Create minimal config for demo
            return Config(
                input_path="uploads",
                output_path="output",
                temp_path="temp",
                log_level="INFO"
            )
    
    def _setup_sidebar(self):
        """Setup the sidebar navigation and controls."""
        st.sidebar.title("📚 RAG Processor")
        st.sidebar.markdown("---")
        
        # Navigation
        page = st.sidebar.selectbox(
            "Navigation",
            ["🏠 Dashboard", "📤 Upload Documents", "⚙️ Processing", "📊 Results", "🔧 Settings"]
        )
        
        st.session_state.current_page = page
        
        # Quick stats
        st.sidebar.markdown("### Quick Stats")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Files", len(st.session_state.uploaded_files))
        with col2:
            st.metric("Processed", len(st.session_state.processing_results))
    
    def _setup_main_content(self):
        """Setup the main content area based on selected page."""
        page = st.session_state.current_page
        
        if page == "🏠 Dashboard":
            self._show_dashboard()
        elif page == "📤 Upload Documents":
            self._show_upload_page()
        elif page == "⚙️ Processing":
            self._show_processing_page()
        elif page == "📊 Results":
            self._show_results_page()
        elif page == "🔧 Settings":
            self._show_settings_page()
    
    def _show_dashboard(self):
        """Display the main dashboard."""
        st.title("🏠 RAG Document Processing Dashboard")
        st.markdown("Welcome to the RAG Document Processing Utility Dashboard")
        
        # Overview cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.info("📤 **Uploads**")
            st.metric("Total Files", len(st.session_state.uploaded_files))
        
        with col2:
            st.success("⚙️ **Processing**")
            st.metric("Completed", len([f for f in st.session_state.uploaded_files if f.get('status') == 'completed']))
        
        with col3:
            st.warning("📊 **Output**")
            st.metric("Chunks Generated", sum([r.get('chunks', 0) for r in st.session_state.processing_results]))
        
        with col4:
            st.info("🔍 **Vector Store**")
            st.metric("Stored Chunks", "N/A")
        
        # Quick actions
        st.markdown("### Quick Actions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚀 Start New Processing", type="primary"):
                st.session_state.current_page = "⚙️ Processing"
                st.rerun()
        
        with col2:
            if st.button("📤 Upload Documents"):
                st.session_state.current_page = "📤 Upload Documents"
                st.rerun()
        
        with col3:
            if st.button("📊 View Results"):
                st.session_state.current_page = "📊 Results"
                st.rerun()
    
    def _show_upload_page(self):
        """Display the document upload page."""
        st.title("📤 Upload Documents")
        st.markdown("Upload documents to be processed by the RAG system")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Choose documents to upload",
            type=['txt', 'pdf', 'docx', 'html', 'md'],
            accept_multiple_files=True,
            help="Supported formats: TXT, PDF, DOCX, HTML, MD"
        )
        
        if uploaded_files:
            st.markdown("### Uploaded Files")
            
            # Display file info
            file_data = []
            for file in uploaded_files:
                file_info = {
                    'name': file.name,
                    'size': f"{file.size / 1024:.1f} KB",
                    'type': file.type or 'Unknown',
                    'status': 'pending'
                }
                file_data.append(file_info)
                
                # Save file info to session state
                if file.name not in [f['name'] for f in st.session_state.uploaded_files]:
                    st.session_state.uploaded_files.append(file_info)
            
            # Display as table
            df = pd.DataFrame(file_data)
            st.dataframe(df, use_container_width=True)
            
            # Processing options
            st.markdown("### Processing Options")
            
            col1, col2 = st.columns(2)
            with col1:
                chunk_size = st.slider("Chunk Size", 100, 2000, 500, help="Number of characters per chunk")
                overlap = st.slider("Chunk Overlap", 0, 200, 50, help="Overlap between chunks")
            
            with col2:
                export_mkdocs = st.checkbox("Export to MkDocs", value=True, help="Generate MkDocs documentation site")
                enable_security = st.checkbox("Enable Security Scanning", value=True, help="Scan for security threats")
                enable_quality = st.checkbox("Enable Quality Assessment", value=True, help="Assess content quality")
            
            # Start processing button
            if st.button("🚀 Start Processing", type="primary"):
                self._start_processing(uploaded_files, {
                    'chunk_size': chunk_size,
                    'overlap': overlap,
                    'export_mkdocs': export_mkdocs,
                    'enable_security': enable_security,
                    'enable_quality': enable_quality
                })
    
    def _show_processing_page(self):
        """Display the processing page with real-time updates."""
        st.title("⚙️ Document Processing")
        
        if not st.session_state.uploaded_files:
            st.warning("No files uploaded. Please upload documents first.")
            return
        
        # Processing status
        st.markdown("### Processing Status")
        
        # Progress tracking
        total_files = len(st.session_state.uploaded_files)
        completed_files = len([f for f in st.session_state.uploaded_files if f.get('status') == 'completed'])
        progress = completed_files / total_files if total_files > 0 else 0
        
        st.progress(progress)
        st.markdown(f"**Progress:** {completed_files}/{total_files} files completed ({progress:.1%})")
        
        # File status table
        st.markdown("### File Status")
        status_df = pd.DataFrame(st.session_state.uploaded_files)
        st.dataframe(status_df, use_container_width=True)
    
    def _show_results_page(self):
        """Display the processing results page."""
        st.title("📊 Processing Results")
        
        if not st.session_state.processing_results:
            st.info("No processing results available. Process some documents first.")
            return
        
        # Results overview
        st.markdown("### Results Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            total_processed = len(st.session_state.processing_results)
            st.metric("Total Processed", total_processed)
        
        with col2:
            successful = len([r for r in st.session_state.processing_results if r.get('status') == 'completed'])
            st.metric("Successful", successful)
        
        with col3:
            failed = len([r for r in st.session_state.processing_results if r.get('status') == 'failed'])
            st.metric("Failed", failed)
        
        # Results table
        st.markdown("### Detailed Results")
        results_df = pd.DataFrame(st.session_state.processing_results)
        st.dataframe(results_df, use_container_width=True)
    
    def _show_settings_page(self):
        """Display the settings configuration page."""
        st.title("🔧 Settings & Configuration")
        
        st.markdown("### Current Configuration")
        
        # Display current config
        config_data = {
            'Input Path': self.config.input.path,
            'Output Path': self.config.output.path,
            'Temp Path': self.config.temp.path,
            'Log Level': self.config.logging.level
        }
        
        for key, value in config_data.items():
            st.markdown(f"**{key}:** {value}")
    
    def _start_processing(self, uploaded_files, options):
        """Start processing the uploaded files."""
        st.info("🚀 Starting document processing...")
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            for i, file in enumerate(uploaded_files):
                # Update status
                status_text.text(f"Processing {file.name}...")
                
                # Simulate processing steps
                time.sleep(0.5)  # Simulate processing time
                
                # Update progress
                progress = (i + 1) / len(uploaded_files)
                progress_bar.progress(progress)
                
                # Update file status
                for f in st.session_state.uploaded_files:
                    if f['name'] == file.name:
                        f['status'] = 'completed'
                        break
                
                # Add to results
                result = {
                    'filename': file.name,
                    'status': 'completed',
                    'processing_time': 0.5,
                    'chunks': 5,  # Simulated
                    'markdown_generated': options['export_mkdocs'],
                    'message': 'Successfully processed'
                }
                
                st.session_state.processing_results.append(result)
            
            progress_bar.progress(1.0)
            status_text.text("✅ Processing completed!")
            
            st.success(f"Successfully processed {len(uploaded_files)} documents!")
            
            # Auto-navigate to results
            st.session_state.current_page = "📊 Results"
            st.rerun()
            
        except Exception as e:
            st.error(f"Error during processing: {e}")
            progress_bar.progress(0)
            status_text.text("❌ Processing failed!")


def main():
    """Main function to run the dashboard."""
    dashboard = RAGDashboard()


if __name__ == "__main__":
    main()
