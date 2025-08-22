#!/usr/bin/env python3
"""
RAG Document Processing UI Demo
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

def main():
    st.set_page_config(page_title="RAG Dashboard Demo", page_icon="📚", layout="wide")
    
    st.title("🎨 RAG Document Processing Dashboard Demo")
    
    # Sidebar
    st.sidebar.title("📚 RAG Processor")
    page = st.sidebar.selectbox("Navigation", ["🏠 Dashboard", "📤 Upload", "⚙️ Processing", "📊 Results"])
    
    # Demo data
    demo_files = [
        {'name': 'research_paper.pdf', 'size': '2.3 MB', 'status': 'completed'},
        {'name': 'business_report.docx', 'size': '1.8 MB', 'status': 'completed'},
        {'name': 'technical_spec.html', 'size': '456 KB', 'status': 'processing'},
        {'name': 'user_manual.txt', 'size': '789 KB', 'status': 'pending'}
    ]
    
    demo_results = [
        {'filename': 'research_paper.pdf', 'status': 'completed', 'chunks': 45, 'time': 12.5},
        {'filename': 'business_report.docx', 'status': 'completed', 'chunks': 32, 'time': 8.2},
        {'filename': 'technical_spec.html', 'status': 'processing', 'chunks': 18, 'time': 5.1},
        {'filename': 'user_manual.txt', 'status': 'pending', 'chunks': 0, 'time': 0.0}
    ]
    
    if page == "🏠 Dashboard":
        show_dashboard(demo_files, demo_results)
    elif page == "📤 Upload":
        show_upload()
    elif page == "⚙️ Processing":
        show_processing(demo_files)
    elif page == "📊 Results":
        show_results(demo_results)

def show_dashboard(files, results):
    st.markdown("## 🏠 Dashboard Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.info("📤 **Uploads**")
        st.metric("Total Files", len(files))
    
    with col2:
        st.success("⚙️ **Processing**")
        completed = len([f for f in files if f.get('status') == 'completed'])
        st.metric("Completed", completed)
    
    with col3:
        st.warning("📊 **Output**")
        total_chunks = sum([r.get('chunks', 0) for r in results])
        st.metric("Chunks Generated", total_chunks)
    
    with col4:
        st.info("🔍 **Vector Store**")
        st.metric("Stored Chunks", 95)
    
    # Quick actions
    st.markdown("### Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Start Processing", type="primary"):
            st.success("Processing started!")
    
    with col2:
        if st.button("📤 Upload Documents"):
            st.info("Navigate to Upload page")
    
    with col3:
        if st.button("📊 View Results"):
            st.info("Navigate to Results page")

def show_upload():
    st.markdown("## 📤 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Choose documents to upload",
        type=['txt', 'pdf', 'docx', 'html', 'md'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.markdown("### Uploaded Files")
        file_data = []
        for file in uploaded_files:
            file_info = {
                'name': file.name,
                'size': f"{file.size / 1024:.1f} KB",
                'type': file.type or 'Unknown',
                'status': 'pending'
            }
            file_data.append(file_info)
        
        df = pd.DataFrame(file_data)
        st.dataframe(df, use_container_width=True)
        
        if st.button("🚀 Start Processing", type="primary"):
            st.success(f"Processing started for {len(uploaded_files)} documents!")

def show_processing(files):
    st.markdown("## ⚙️ Document Processing")
    
    # Progress tracking
    total_files = len(files)
    completed_files = len([f for f in files if f.get('status') == 'completed'])
    progress = completed_files / total_files if total_files > 0 else 0
    
    st.progress(progress)
    st.markdown(f"**Progress:** {completed_files}/{total_files} files completed ({progress:.1%})")
    
    # File status table
    st.markdown("### File Status")
    status_df = pd.DataFrame(files)
    st.dataframe(status_df, use_container_width=True)

def show_results(results):
    st.markdown("## 📊 Processing Results")
    
    # Results overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Processed", len(results))
    
    with col2:
        successful = len([r for r in results if r.get('status') == 'completed'])
        st.metric("Successful", successful)
    
    with col3:
        failed = len([r for r in results if r.get('status') == 'failed'])
        st.metric("Failed", failed)
    
    # Results table
    st.markdown("### Detailed Results")
    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True)
    
    # Status distribution chart
    if len(results) > 1:
        status_counts = results_df['status'].value_counts()
        fig = px.pie(values=status_counts.values, names=status_counts.index, title="Processing Status")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
