"""
Tests for metadata enhancement features
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.metadata_enhancement import (
    MetadataEnhancer,
    CrossDocumentAnalyzer,
    SemanticClusterer,
    KnowledgeGraphBuilder,
    get_metadata_enhancer,
)
from src.config import Config
from src.parsers import ParsedContent
from src.chunkers import DocumentChunk


class TestCrossDocumentAnalyzer:
    """Test cross-document relationship analysis."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = Mock(spec=Config)
        self.mock_config.metadata.enhancement.cross_document_analysis = True
        self.analyzer = CrossDocumentAnalyzer(self.mock_config)

    def test_find_document_relationships(self):
        """Test finding relationships between documents."""
        documents = [
            {"id": "doc1", "content": "Machine learning algorithms", "metadata": {"topics": ["AI", "ML"]}},
            {"id": "doc2", "content": "Deep learning neural networks", "metadata": {"topics": ["AI", "DL"]}},
            {"id": "doc3", "content": "Data preprocessing techniques", "metadata": {"topics": ["Data", "ML"]}}
        ]
        
        relationships = self.analyzer.find_relationships(documents)
        
        assert isinstance(relationships, list)
        assert len(relationships) > 0
        assert all("source" in rel and "target" in rel for rel in relationships)

    def test_cross_document_analysis_disabled(self):
        """Test behavior when cross-document analysis is disabled."""
        self.mock_config.metadata.enhancement.cross_document_analysis = False
        analyzer = CrossDocumentAnalyzer(self.mock_config)
        
        relationships = analyzer.find_relationships([])
        assert relationships == []


class TestSemanticClusterer:
    """Test semantic clustering functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = Mock(spec=Config)
        self.mock_config.metadata.enhancement.semantic_clustering = True
        self.mock_config.metadata.enhancement.clustering_threshold = 0.7
        self.clusterer = SemanticClusterer(self.mock_config)

    @patch('src.metadata_enhancement.sentence_transformers.SentenceTransformer')
    def test_cluster_documents_by_semantics(self, mock_transformer):
        """Test semantic clustering of documents."""
        # Mock the sentence transformer
        mock_model = Mock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.9, 0.8, 0.7]]
        mock_transformer.return_value = mock_model
        
        documents = [
            {"id": "doc1", "content": "Machine learning introduction"},
            {"id": "doc2", "content": "Introduction to ML"},
            {"id": "doc3", "content": "Cooking recipes"}
        ]
        
        clusters = self.clusterer.cluster_documents(documents)
        
        assert isinstance(clusters, list)
        assert len(clusters) > 0
        assert all("documents" in cluster for cluster in clusters)

    def test_semantic_clustering_disabled(self):
        """Test behavior when semantic clustering is disabled."""
        self.mock_config.metadata.enhancement.semantic_clustering = False
        clusterer = SemanticClusterer(self.mock_config)
        
        clusters = clusterer.cluster_documents([])
        assert clusters == []


class TestKnowledgeGraphBuilder:
    """Test knowledge graph construction."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = Mock(spec=Config)
        self.mock_config.metadata.enhancement.knowledge_graph = True
        self.builder = KnowledgeGraphBuilder(self.mock_config)

    def test_build_knowledge_graph(self):
        """Test knowledge graph construction from documents."""
        documents = [
            {"id": "doc1", "entities": [{"text": "Python", "type": "PROGRAMMING_LANGUAGE"}]},
            {"id": "doc2", "entities": [{"text": "Machine Learning", "type": "FIELD"}]},
            {"id": "doc3", "entities": [{"text": "Python", "type": "PROGRAMMING_LANGUAGE"}]}
        ]
        
        graph = self.builder.build_graph(documents)
        
        assert isinstance(graph, dict)
        assert "nodes" in graph
        assert "edges" in graph
        assert len(graph["nodes"]) > 0

    def test_knowledge_graph_disabled(self):
        """Test behavior when knowledge graph is disabled."""
        self.mock_config.metadata.enhancement.knowledge_graph = False
        builder = KnowledgeGraphBuilder(self.mock_config)
        
        graph = builder.build_graph([])
        assert graph == {"nodes": [], "edges": []}


class TestMetadataEnhancer:
    """Test the main metadata enhancer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = Mock(spec=Config)
        self.mock_config.metadata.enhancement.cross_document_analysis = True
        self.mock_config.metadata.enhancement.semantic_clustering = True
        self.mock_config.metadata.enhancement.knowledge_graph = True
        self.enhancer = MetadataEnhancer(self.mock_config)

    def test_enhancer_initialization(self):
        """Test metadata enhancer initialization."""
        assert hasattr(self.enhancer, "cross_document_analyzer")
        assert hasattr(self.enhancer, "semantic_clusterer")
        assert hasattr(self.enhancer, "knowledge_graph_builder")

    def test_enhance_document_metadata(self):
        """Test enhancing metadata for a single document."""
        document = {
            "id": "doc1",
            "content": "Sample content",
            "metadata": {"topics": ["AI"]}
        }
        
        enhanced = self.enhancer.enhance_document(document)
        
        assert isinstance(enhanced, dict)
        assert "enhanced_metadata" in enhanced
        assert enhanced["success"] is True

    def test_enhance_document_collection(self):
        """Test enhancing metadata for a collection of documents."""
        documents = [
            {"id": "doc1", "content": "Content 1", "metadata": {}},
            {"id": "doc2", "content": "Content 2", "metadata": {}},
            {"id": "doc3", "content": "Content 3", "metadata": {}}
        ]
        
        enhanced = self.enhancer.enhance_collection(documents)
        
        assert isinstance(enhanced, dict)
        assert "enhanced_documents" in enhanced
        assert enhanced["success"] is True
        assert len(enhanced["enhanced_documents"]) == 3

    def test_enhancement_with_disabled_features(self):
        """Test enhancement when some features are disabled."""
        self.mock_config.metadata.enhancement.cross_document_analysis = False
        self.mock_config.metadata.enhancement.semantic_clustering = False
        self.mock_config.metadata.enhancement.knowledge_graph = False
        
        enhancer = MetadataEnhancer(self.mock_config)
        
        documents = [{"id": "doc1", "content": "Content", "metadata": {}}]
        enhanced = enhancer.enhance_collection(documents)
        
        assert enhanced["success"] is True
        assert "enhanced_documents" in enhanced


def test_get_metadata_enhancer():
    """Test the factory function for creating metadata enhancers."""
    enhancer = get_metadata_enhancer()
    assert isinstance(enhancer, MetadataEnhancer)


if __name__ == "__main__":
    pytest.main([__file__])
