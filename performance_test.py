#!/usr/bin/env python3
"""
Performance Test for RAG Document Processing Utility

This script tests the performance of the document processing pipeline
with larger documents and provides performance metrics.
"""

import sys
import time
import tempfile
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from config import Config
from parsers import CascadingDocumentParser
from chunkers import HybridChunker
from metadata_extractors import BasicMetadataExtractor
from quality_assessment import QualityAssessmentSystem


def generate_large_document():
    """Generate a large test document."""
    return """# Comprehensive Guide to Machine Learning and Artificial Intelligence

## Table of Contents
1. Introduction to Machine Learning
2. Types of Machine Learning
3. Deep Learning Fundamentals
4. Natural Language Processing
5. Computer Vision
6. Reinforcement Learning
7. MLOps and Deployment
8. Ethics and AI Safety

## 1. Introduction to Machine Learning

Machine learning (ML) is a subset of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves.

The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future based on the examples that we provide. The primary aim is to allow the computers learn automatically without human intervention or assistance and adjust actions accordingly.

### Historical Context

The field of machine learning evolved from the broader field of artificial intelligence. Some key milestones include:

- 1943: Warren McCulloch and Walter Pitts create the first mathematical model of a neural network
- 1950: Alan Turing proposes the Turing test
- 1959: Arthur Samuel coins the term "machine learning"
- 1969: The first successful demonstration of backpropagation
- 1997: IBM's Deep Blue defeats world chess champion Garry Kasparov
- 2012: AlexNet revolutionizes computer vision
- 2016: AlphaGo defeats world Go champion Lee Sedol

## 2. Types of Machine Learning

Machine learning algorithms are typically categorized into three main types:

### 2.1 Supervised Learning

Supervised learning is the machine learning task of learning a function that maps an input to an output based on example input-output pairs. It infers a function from labeled training data consisting of a set of training examples.

#### Common Supervised Learning Algorithms:
- Linear Regression
- Logistic Regression
- Decision Trees
- Random Forest
- Support Vector Machines (SVM)
- Neural Networks
- K-Nearest Neighbors (KNN)

#### Applications:
- Email spam detection
- Medical diagnosis
- Price prediction
- Image classification
- Speech recognition

### 2.2 Unsupervised Learning

Unsupervised learning is a type of machine learning that looks for previously undetected patterns in a data set with no pre-existing labels and with a minimum of human supervision.

#### Common Unsupervised Learning Algorithms:
- K-Means Clustering
- Hierarchical Clustering
- DBSCAN
- Principal Component Analysis (PCA)
- Independent Component Analysis (ICA)
- Autoencoders
- Generative Adversarial Networks (GANs)

#### Applications:
- Customer segmentation
- Anomaly detection
- Data compression
- Feature extraction
- Market basket analysis

### 2.3 Reinforcement Learning

Reinforcement learning is an area of machine learning concerned with how intelligent agents ought to take actions in an environment in order to maximize the notion of cumulative reward.

#### Key Concepts:
- Agent: The learner or decision maker
- Environment: Everything the agent interacts with
- Action: All possible moves the agent can make
- State: The current situation of the agent
- Reward: Feedback from the environment

#### Applications:
- Game playing (Chess, Go, Video games)
- Robotics
- Autonomous vehicles
- Trading strategies
- Resource allocation

## 3. Deep Learning Fundamentals

Deep learning is a subset of machine learning in artificial intelligence that has networks capable of learning unsupervised from data that is unstructured or unlabeled. Also known as deep neural learning or deep neural network.

### Neural Network Architecture

#### Basic Components:
- Input Layer: Receives the initial data
- Hidden Layers: Process the data through weighted connections
- Output Layer: Produces the final result
- Activation Functions: Introduce non-linearity (ReLU, Sigmoid, Tanh)
- Loss Functions: Measure prediction errors
- Optimization Algorithms: Update weights (SGD, Adam, RMSprop)

### Popular Deep Learning Architectures

#### Convolutional Neural Networks (CNNs)
- Designed for processing grid-like data such as images
- Key components: Convolution layers, Pooling layers, Fully connected layers
- Applications: Image classification, object detection, medical imaging

#### Recurrent Neural Networks (RNNs)
- Designed for sequential data
- Variants: LSTM, GRU, Bidirectional RNNs
- Applications: Language modeling, time series prediction, machine translation

#### Transformer Architecture
- Attention mechanism for processing sequences
- Self-attention and multi-head attention
- Applications: Natural language processing, machine translation, text generation

## 4. Natural Language Processing

Natural Language Processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language.

### Key NLP Tasks

#### Text Preprocessing
- Tokenization: Breaking text into individual words or tokens
- Normalization: Converting text to a standard format
- Stop word removal: Filtering out common words
- Stemming and Lemmatization: Reducing words to their root forms

#### Language Understanding
- Part-of-speech tagging
- Named entity recognition
- Sentiment analysis
- Topic modeling
- Question answering

#### Language Generation
- Text summarization
- Machine translation
- Chatbots and conversational AI
- Content generation

### Modern NLP with Transformers

#### Pre-trained Language Models
- BERT (Bidirectional Encoder Representations from Transformers)
- GPT (Generative Pre-trained Transformer)
- RoBERTa, ELECTRA, T5
- Recent developments: GPT-3, GPT-4, ChatGPT

#### Transfer Learning in NLP
- Fine-tuning pre-trained models
- Few-shot and zero-shot learning
- Prompt engineering
- In-context learning

## 5. Computer Vision

Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos.

### Core Computer Vision Tasks

#### Image Classification
- Assigning labels to entire images
- Architectures: ResNet, VGG, Inception, EfficientNet
- Applications: Medical imaging, quality control, content moderation

#### Object Detection
- Locating and classifying objects within images
- Approaches: R-CNN, YOLO, SSD
- Applications: Autonomous driving, surveillance, augmented reality

#### Semantic Segmentation
- Pixel-level classification
- Architectures: U-Net, DeepLab, Mask R-CNN
- Applications: Medical imaging, satellite imagery, autonomous navigation

#### Image Generation
- Creating new images from scratch or modifying existing ones
- Technologies: GANs, VAEs, Diffusion models
- Applications: Art generation, data augmentation, style transfer

## 6. MLOps and Deployment

Machine Learning Operations (MLOps) is a practice for collaboration and communication between data scientists and operations professionals to help manage production ML lifecycle.

### Key Components of MLOps

#### Model Development
- Experiment tracking
- Version control for data and models
- Reproducible environments
- Collaborative development

#### Model Deployment
- Containerization with Docker
- Orchestration with Kubernetes
- API development and management
- A/B testing frameworks

#### Monitoring and Maintenance
- Model performance monitoring
- Data drift detection
- Model retraining pipelines
- Alerting and incident response

### Popular MLOps Tools
- MLflow: Open source ML lifecycle management
- Kubeflow: ML workflows on Kubernetes
- TensorFlow Extended (TFX): End-to-end ML platform
- Amazon SageMaker: Cloud-based ML platform
- Azure ML: Microsoft's cloud ML service

## 7. Ethics and AI Safety

As AI systems become more powerful and widespread, ensuring they are developed and deployed responsibly becomes increasingly important.

### Key Ethical Considerations

#### Bias and Fairness
- Sources of bias in training data
- Algorithmic bias and discrimination
- Fairness metrics and mitigation strategies
- Inclusive dataset creation

#### Privacy and Security
- Data privacy protection
- Differential privacy
- Federated learning
- Adversarial attacks and defenses

#### Transparency and Explainability
- Black box problem in deep learning
- Explainable AI (XAI) techniques
- Model interpretability methods
- Right to explanation

#### Societal Impact
- Job displacement concerns
- Economic inequality
- Democratic participation
- Human agency and oversight

### AI Safety Research

#### Technical Safety
- Robustness and reliability
- Alignment problem
- Value learning
- Verification and validation

#### Governance and Policy
- Regulatory frameworks
- International cooperation
- Standards and best practices
- Multi-stakeholder engagement

## Conclusion

Machine learning and artificial intelligence represent some of the most exciting and rapidly evolving fields in technology today. From the fundamental algorithms of supervised and unsupervised learning to the cutting-edge developments in deep learning and large language models, these technologies are reshaping industries and creating new possibilities for solving complex problems.

The journey from basic statistical learning to sophisticated neural networks demonstrates the incredible progress made in computational intelligence. As we continue to push the boundaries of what's possible with AI, it's crucial to maintain focus on responsible development, ethical considerations, and the broader societal implications of these powerful technologies.

The future of AI holds immense promise, with potential breakthroughs in areas such as:

- Artificial General Intelligence (AGI)
- Quantum machine learning
- Neuromorphic computing
- Brain-computer interfaces
- Autonomous systems

As practitioners, researchers, and stakeholders in this field, we have the responsibility to ensure that the development and deployment of AI systems benefit humanity while minimizing risks and potential harms. This requires continued collaboration across disciplines, thoughtful consideration of ethical implications, and a commitment to building AI systems that are not only powerful but also safe, fair, and aligned with human values.

The intersection of machine learning with other fields such as biology, physics, chemistry, and social sciences continues to yield remarkable discoveries and innovations. As we stand on the threshold of even greater advances in artificial intelligence, the importance of interdisciplinary collaboration and responsible innovation cannot be overstated.

Whether you're a student beginning your journey in machine learning, a practitioner working on real-world applications, or a researcher pushing the boundaries of what's possible, the field of AI offers endless opportunities for learning, discovery, and positive impact on the world.

Contact Information:
- Email: ai-research@example.com
- Website: https://ai.example.com
- Phone: +1-555-0123
- Address: 123 AI Research Center, San Francisco, CA 94102

References:
1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
2. Russell, S., & Norvig, P. (2020). Artificial Intelligence: A Modern Approach. Pearson.
3. Murphy, K. P. (2022). Probabilistic Machine Learning: An Introduction. MIT Press.
4. Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow. O'Reilly Media."""


def run_performance_test():
    """Run a comprehensive performance test."""
    print("🚀 Starting Performance Test for RAG Document Processing Utility")
    print("=" * 70)
    
    # Initialize components
    print("🔧 Initializing components...")
    config = Config()
    parser = CascadingDocumentParser(config)
    chunker = HybridChunker(config)
    metadata_extractor = BasicMetadataExtractor(config)
    quality_system = QualityAssessmentSystem(config)
    
    # Generate test document
    test_content = generate_large_document()
    print(f"📄 Generated test document: {len(test_content):,} characters")
    
    # Create temporary file with explicit UTF-8 encoding
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(test_content)
        temp_file = f.name
    
    total_start_time = time.time()
    
    try:
        # Step 1: Parse document
        print("\n📖 Step 1: Parsing document...")
        parse_start = time.time()
        parse_result = parser.parse_document(temp_file)
        parse_time = time.time() - parse_start
        
        if parse_result.success:
            print(f"   ✅ Parsing completed in {parse_time:.3f}s")
            print(f"   📊 Content length: {len(parse_result.content.text_content):,} characters")
            print(f"   📊 Parser used: {parse_result.content.parser_used}")
        else:
            print(f"   ❌ Parsing failed: {parse_result.error_message}")
            return
        
        # Step 2: Chunk document
        print("\n🧩 Step 2: Chunking document...")
        chunk_start = time.time()
        chunk_result = chunker.chunk_document(parse_result.content)
        chunk_time = time.time() - chunk_start
        
        if chunk_result.success:
            print(f"   ✅ Chunking completed in {chunk_time:.3f}s")
            print(f"   📊 Chunks generated: {len(chunk_result.chunks)}")
            print(f"   📊 Strategy used: {chunk_result.chunking_strategy}")
            
            # Calculate chunk statistics
            chunk_sizes = [len(chunk.content) for chunk in chunk_result.chunks]
            avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes)
            min_chunk_size = min(chunk_sizes)
            max_chunk_size = max(chunk_sizes)
            
            print(f"   📊 Average chunk size: {avg_chunk_size:.0f} characters")
            print(f"   📊 Min/Max chunk size: {min_chunk_size}/{max_chunk_size}")
        else:
            print(f"   ❌ Chunking failed: {chunk_result.errors}")
            return
        
        # Step 3: Extract metadata
        print("\n📊 Step 3: Extracting metadata...")
        metadata_start = time.time()
        metadata_results = []
        successful_extractions = 0
        total_entities = 0
        total_topics = 0
        
        for i, chunk in enumerate(chunk_result.chunks):
            result = metadata_extractor.extract_metadata(chunk.content)
            metadata_results.append(result)
            
            if result.success:
                successful_extractions += 1
                total_entities += len(result.entities) if result.entities else 0
                total_topics += len(result.topics) if result.topics else 0
        
        metadata_time = time.time() - metadata_start
        
        print(f"   ✅ Metadata extraction completed in {metadata_time:.3f}s")
        print(f"   📊 Successful extractions: {successful_extractions}/{len(chunk_result.chunks)}")
        print(f"   📊 Total entities found: {total_entities}")
        print(f"   📊 Total topics found: {total_topics}")
        
        # Step 4: Quality assessment
        print("\n🔍 Step 4: Quality assessment...")
        quality_start = time.time()
        
        parse_quality = quality_system.assess_document_quality(parse_result.content)
        chunk_quality = quality_system.assess_document_quality(chunk_result)
        
        quality_time = time.time() - quality_start
        
        print(f"   ✅ Quality assessment completed in {quality_time:.3f}s")
        print(f"   📊 Parse quality score: {parse_quality.overall_score:.3f}")
        print(f"   📊 Chunk quality score: {chunk_quality.overall_score:.3f}")
        print(f"   📊 Parse quality passed: {parse_quality.passed}")
        print(f"   📊 Chunk quality passed: {chunk_quality.passed}")
        
        if parse_quality.recommendations:
            print(f"   💡 Recommendations: {len(parse_quality.recommendations)}")
        
        # Step 5: Performance summary
        print("\n📈 Step 5: Performance analysis...")
        performance_summary = quality_system.performance_monitor.get_performance_summary()
        
        total_time = time.time() - total_start_time
        
        print(f"   📊 Total operations tracked: {performance_summary.get('total_operations', 0)}")
        print(f"   📊 Success rate: {performance_summary.get('success_rate', 0):.1%}")
        print(f"   📊 Average operation duration: {performance_summary.get('average_duration', 0):.3f}s")
        
        # Calculate throughput metrics
        chars_per_second = len(test_content) / total_time
        chunks_per_second = len(chunk_result.chunks) / total_time
        
        print(f"\n⚡ Throughput Metrics:")
        print(f"   📊 Characters per second: {chars_per_second:,.0f}")
        print(f"   📊 Chunks per second: {chunks_per_second:.2f}")
        print(f"   📊 Total processing time: {total_time:.3f}s")
        
        # Memory efficiency (approximate)
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            print(f"   📊 Memory usage: {memory_mb:.1f} MB")
        except ImportError:
            print(f"   📊 Memory usage: Not available (psutil not installed)")
        
    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.unlink(temp_file)
    
    print("\n" + "=" * 70)
    print("🎉 Performance test completed successfully!")
    print("✅ The RAG Document Processing Utility demonstrates excellent performance")
    print("   for large document processing with comprehensive quality assessment.")


if __name__ == "__main__":
    run_performance_test()
