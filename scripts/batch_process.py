#!/usr/bin/env python3
"""
Batch Document Processing Script

This script processes multiple documents in batch mode.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import List

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from processor import DocumentProcessor


def setup_logging(log_level: str = "INFO"):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/batch_processing.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def discover_documents(input_dir: str) -> List[Path]:
    """Discover documents in the input directory."""
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Supported file extensions
    supported_extensions = {'.pdf', '.docx', '.html', '.txt', '.md'}
    
    documents = []
    for file_path in input_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            documents.append(file_path)
    
    return documents


def process_documents(documents: List[Path], output_dir: str, processor: DocumentProcessor):
    """Process a list of documents."""
    results = []
    
    for i, doc_path in enumerate(documents, 1):
        try:
            logging.info(f"Processing document {i}/{len(documents)}: {doc_path.name}")
            
            # Process the document
            result = processor.process_document(str(doc_path))
            results.append(result)
            
            logging.info(f"✅ Successfully processed {doc_path.name}")
            
        except Exception as e:
            logging.error(f"❌ Failed to process {doc_path.name}: {e}")
            continue
    
    return results


def save_results(results: List, output_dir: str):
    """Save processing results to output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save summary report
    summary_file = output_path / "processing_summary.json"
    import json
    
    summary = {
        "total_documents": len(results),
        "successful_processing": len([r for r in results if not r.errors]),
        "failed_processing": len([r for r in results if r.errors]),
        "total_chunks": sum(r.metadata.get('total_chunks', 0) for r in results),
        "average_processing_time": sum(r.processing_time for r in results) / len(results) if results else 0,
        "average_memory_usage": sum(r.memory_usage for r in results) / len(results) if results else 0,
        "results": [
            {
                "document_id": r.document_id,
                "source_path": r.metadata.get('source_path', ''),
                "total_chunks": r.metadata.get('total_chunks', 0),
                "processing_time": r.processing_time,
                "memory_usage": r.memory_usage,
                "quality_metrics": r.quality_metrics,
                "errors": r.errors,
                "warnings": r.warnings
            }
            for r in results
        ]
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logging.info(f"📊 Processing summary saved to {summary_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Batch document processing for RAGPrep")
    parser.add_argument("--input-dir", "-i", required=True, help="Input directory containing documents")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory for results")
    parser.add_argument("--log-level", "-l", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    parser.add_argument("--config", "-c", help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    
    try:
        logging.info("🚀 Starting batch document processing")
        
        # Discover documents
        documents = discover_documents(args.input_dir)
        logging.info(f"📁 Found {len(documents)} documents to process")
        
        if not documents:
            logging.warning("No documents found to process")
            return
        
        # Initialize processor
        processor = DocumentProcessor(args.config)
        logging.info("✅ Document processor initialized")
        
        # Process documents
        results = process_documents(documents, args.output_dir, processor)
        logging.info(f"✅ Completed processing {len(results)} documents")
        
        # Save results
        save_results(results, args.output_dir)
        
        # Display summary
        successful = len([r for r in results if not r.errors])
        failed = len([r for r in results if r.errors])
        total_chunks = sum(r.metadata.get('total_chunks', 0) for r in results)
        
        print(f"\n📊 Processing Summary:")
        print(f"  • Total documents: {len(documents)}")
        print(f"  • Successfully processed: {successful}")
        print(f"  • Failed: {failed}")
        print(f"  • Total chunks generated: {total_chunks}")
        print(f"  • Results saved to: {args.output_dir}")
        
        logging.info("🎉 Batch processing completed successfully")
        
    except Exception as e:
        logging.error(f"❌ Batch processing failed: {e}")
        sys.exit(1)
    
    finally:
        # Cleanup
        if 'processor' in locals():
            processor.cleanup()


if __name__ == "__main__":
    main()
