#!/usr/bin/env python3
"""
RAG Document Processing Utility - Demo Script

This script demonstrates the complete functionality of the RAG Document Processing Utility,
including document parsing, chunking, metadata extraction, quality assessment, and vector storage.
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

# Import modules - handle both package and direct imports
try:
    # Try relative package imports first
    import src
    from src.processor import DocumentProcessor, ProcessingResult
    from src.config import Config
    from src.security import SecurityManager
except ImportError:
    # Fallback to direct imports when src is in path
    from processor import DocumentProcessor, ProcessingResult  # type: ignore
    from config import Config  # type: ignore
    from security import SecurityManager  # type: ignore


def setup_logging():
    """Setup logging for the demo."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('demo.log')
        ]
    )


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n--- {title} ---")


def demo_configuration():
    """Demonstrate configuration system."""
    print_section("Configuration System")
    
    try:
        config = Config()
        print("✅ Configuration loaded successfully")
        print(f"   App Name: {config.app_name}")
        print(f"   Version: {config.version}")
        print(f"   Security: File validation enabled: {config.security.enable_file_validation}")
        print(f"   Chunking Strategy: {config.chunking.strategy}")
        print(f"   Metadata Level: {config.metadata.extraction_level}")
        print(f"   LLM Model: {config.metadata.llm_model}")
        print(f"   LLM Temperature: {config.metadata.llm_temperature}")
        
        # Validate configuration
        errors = config.validate_config()
        if errors:
            print(f"⚠️  Configuration warnings: {len(errors)}")
            for error in errors[:3]:  # Show first 3 errors
                print(f"   - {error}")
        else:
            print("✅ Configuration validation passed")
            
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False
    
    return True


def demo_security_system():
    """Demonstrate security system."""
    print_section("Security System")
    
    try:
        config = Config()
        security_manager = SecurityManager(config)
        
        # Test with a safe file
        test_file = Path("documents/test_document.txt")
        if test_file.exists():
            profile = security_manager.assess_file_security(test_file)
            print(f"✅ Security assessment completed")
            print(f"   File: {test_file.name}")
            print(f"   Safe: {profile.is_safe}")
            print(f"   Threat Level: {profile.overall_threat_level}")
            print(f"   File Size: {profile.file_size} bytes")
            print(f"   MIME Type: {profile.mime_type}")
            
            if profile.warnings:
                print(f"   Warnings: {len(profile.warnings)}")
                for warning in profile.warnings:
                    print(f"     - {warning}")
        else:
            print("⚠️  Test document not found for security testing")
            
    except Exception as e:
        print(f"❌ Security system error: {e}")
        return False
    
    return True


def demo_document_processing():
    """Demonstrate document processing pipeline."""
    print_section("Document Processing Pipeline")
    
    try:
        # Initialize processor
        processor = DocumentProcessor()
        print("✅ Document processor initialized")
        
        # Process test document
        test_file = "documents/test_document.txt"
        if not os.path.exists(test_file):
            print(f"❌ Test file not found: {test_file}")
            return False
        
        print(f"📄 Processing document: {test_file}")
        start_time = time.time()
        
        result = processor.process_document(test_file)
        
        processing_time = time.time() - start_time
        
        if result.success:
            print("✅ Document processing completed successfully")
            print(f"   Document ID: {result.document_id}")
            print(f"   Chunks Created: {len(result.chunks)}")
            print(f"   Quality Score: {result.quality_score:.2f}")
            print(f"   Processing Time: {processing_time:.2f}s")
            print(f"   Metadata Keys: {list(result.metadata.keys())}")
            
            # Show chunk details
            if result.chunks:
                print(f"\n   Chunk Details:")
                for i, chunk in enumerate(result.chunks[:3]):  # Show first 3 chunks
                    print(f"     Chunk {i+1}: {chunk.chunk_type} (Quality: {chunk.quality_score:.2f})")
                    print(f"       Content Preview: {chunk.content[:100]}...")
                    if chunk.metadata:
                        print(f"       Metadata: {list(chunk.metadata.keys())}")
            
            # Show warnings if any
            if result.warnings:
                print(f"\n   Warnings: {len(result.warnings)}")
                for warning in result.warnings[:3]:  # Show first 3 warnings
                    print(f"     - {warning}")
            
            # Get processing stats
            stats = processor.get_processing_stats()
            print(f"\n   Processing Stats: {stats}")
            
        else:
            print(f"❌ Document processing failed: {result.error_message}")
            return False
        
        # Clean up
        processor.close()
        
    except Exception as e:
        print(f"❌ Document processing error: {e}")
        return False
    
    return True


def demo_batch_processing():
    """Demonstrate batch processing capabilities."""
    print_section("Batch Processing")
    
    try:
        processor = DocumentProcessor()
        
        # Find all test documents
        test_dir = Path("documents")
        test_files = list(test_dir.glob("*.txt")) + list(test_dir.glob("*.html"))
        
        if not test_files:
            print("⚠️  No test documents found for batch processing")
            return True
        
        print(f"📚 Processing {len(test_files)} documents in batch")
        
        file_paths = [str(f) for f in test_files]
        start_time = time.time()
        
        results = processor.process_batch(file_paths)
        
        batch_time = time.time() - start_time
        
        successful = sum(1 for r in results if r.success)
        total_quality = sum(r.quality_score for r in results if r.success)
        avg_quality = total_quality / successful if successful > 0 else 0
        
        print(f"✅ Batch processing completed")
        print(f"   Total Documents: {len(results)}")
        print(f"   Successful: {successful}")
        print(f"   Failed: {len(results) - successful}")
        print(f"   Average Quality: {avg_quality:.2f}")
        print(f"   Total Time: {batch_time:.2f}s")
        print(f"   Average Time per Document: {batch_time/len(results):.2f}s")
        
        # Show individual results
        for i, result in enumerate(results):
            status = "✅" if result.success else "❌"
            print(f"   {status} {Path(result.document_id).name}: Quality {result.quality_score:.2f}")
        
        processor.close()
        
    except Exception as e:
        print(f"❌ Batch processing error: {e}")
        return False
    
    return True


def demo_quality_assessment():
    """Demonstrate quality assessment system."""
    print_section("Quality Assessment System")
    
    try:
        try:
            from src.quality_assessment import get_quality_assessment_system
        except ImportError:
            from quality_assessment import get_quality_assessment_system  # type: ignore
        
        quality_system = get_quality_assessment_system()
        print("✅ Quality assessment system initialized")
        
        # Get performance summary
        if hasattr(quality_system, 'performance_monitor'):
            summary = quality_system.performance_monitor.get_performance_summary()
            print(f"📊 Performance Summary:")
            print(f"   Total Operations: {summary.get('total_operations', 0)}")
            print(f"   Success Rate: {summary.get('success_rate', 0):.2%}")
            print(f"   Average Duration: {summary.get('average_duration', 0):.2f}s")
            print(f"   Operations by Type: {summary.get('operations_by_type', {})}")
        
    except Exception as e:
        print(f"❌ Quality assessment error: {e}")
        return False
    
    return True


def demo_vector_storage():
    """Demonstrate vector storage capabilities."""
    print_section("Vector Storage")
    
    try:
        try:
            from src.vector_store import get_vector_store
        except ImportError:
            from vector_store import get_vector_store  # type: ignore
        
        config = Config()
        vector_store = get_vector_store("file", config)
        print("✅ Vector store initialized")
        
        # Get storage statistics
        stats = {
            "total_documents": vector_store.get_total_documents(),
            "total_chunks": vector_store.get_total_chunks(),
            "document_ids": vector_store.get_document_ids()
        }
        
        print(f"📊 Storage Statistics:")
        print(f"   Total Documents: {stats['total_documents']}")
        print(f"   Total Chunks: {stats['total_chunks']}")
        print(f"   Document IDs: {len(stats['document_ids'])}")
        
        if stats['document_ids']:
            print(f"   Sample Document IDs: {stats['document_ids'][:3]}")
        
        vector_store.close()
        
    except Exception as e:
        print(f"❌ Vector storage error: {e}")
        return False
    
    return True


def main():
    """Main demo function."""
    print_header("RAG Document Processing Utility - Demo")
    print("This demo showcases the complete functionality of the system.")
    print("Make sure you have test documents in the 'documents/' directory.")
    
    # Setup logging
    setup_logging()
    
    # Run demos
    demos = [
        ("Configuration System", demo_configuration),
        ("Security System", demo_security_system),
        ("Document Processing Pipeline", demo_document_processing),
        ("Batch Processing", demo_batch_processing),
        ("Quality Assessment", demo_quality_assessment),
        ("Vector Storage", demo_vector_storage),
    ]
    
    results = []
    
    for demo_name, demo_func in demos:
        try:
            print_header(f"Demo: {demo_name}")
            success = demo_func()
            results.append((demo_name, success))
            
            if success:
                print(f"✅ {demo_name} demo completed successfully")
            else:
                print(f"❌ {demo_name} demo failed")
                
        except Exception as e:
            print(f"❌ {demo_name} demo crashed: {e}")
            results.append((demo_name, False))
        
        # Small delay between demos
        time.sleep(1)
    
    # Summary
    print_header("Demo Summary")
    successful = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"📊 Overall Results: {successful}/{total} demos successful")
    
    for demo_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status}: {demo_name}")
    
    if successful == total:
        print("\n🎉 All demos completed successfully! The system is working correctly.")
    else:
        print(f"\n⚠️  {total - successful} demo(s) failed. Check the logs for details.")
    
    print(f"\n📝 Demo log saved to: demo.log")
    print(f"📁 Vector store data saved to: vector_db/")
    print(f"📁 Output files saved to: output/")


if __name__ == "__main__":
    main()
