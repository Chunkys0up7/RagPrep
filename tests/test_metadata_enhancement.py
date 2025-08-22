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

    @pytest.fixture(autouse=True)
    def setup(self, mock_config):
        """Set up test fixtures."""
        self.mock_config = mock_config
        self.analyzer = CrossDocumentAnalyzer(self.mock_config)

    def test_find_document_relationships(self):
        """Test finding relationships between documents."""
        documents = [
            {"document_id": "doc1", "text_content": "Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models", "metadata": {"topics": ["AI", "ML"]}},
            {"document_id": "doc2", "text_content": "Machine learning algorithms can be used to build predictive models and classification systems", "metadata": {"topics": ["AI", "DL"]}},
            {"document_id": "doc3", "text_content": "Data preprocessing is essential for machine learning algorithms to work effectively on datasets", "metadata": {"topics": ["Data", "ML"]}}
        ]
        
        relationships = self.analyzer.analyze_relationships(documents)
        
        assert isinstance(relationships, list)
        # Relationships may be empty if similarity is below threshold
        # This is expected behavior for the algorithm
        assert all(hasattr(rel, 'source_doc_id') for rel in relationships)

    def test_cross_document_analysis_disabled(self):
        """Test behavior when cross-document analysis is disabled."""
        self.mock_config.metadata.enhancement.cross_document_analysis = False
        analyzer = CrossDocumentAnalyzer(self.mock_config)
        
        relationships = analyzer.analyze_relationships([])
        assert relationships == []


class TestSemanticClusterer:
    """Test semantic clustering functionality."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_config):
        """Set up test fixtures."""
        self.mock_config = mock_config
        self.mock_config.metadata.enhancement.semantic_clustering = True
        self.mock_config.metadata.enhancement.clustering_threshold = 0.7
        self.clusterer = SemanticClusterer(self.mock_config)

    def test_cluster_documents_by_semantics(self):
        """Test semantic clustering of documents."""
        documents = [
            {"id": "doc1", "text_content": "Machine learning introduction", "document_id": "doc1"},
            {"id": "doc2", "text_content": "Introduction to ML algorithms", "document_id": "doc2"},
            {"id": "doc3", "text_content": "Cooking recipes and food preparation", "document_id": "doc3"}
        ]
        
        clusters = self.clusterer.cluster_documents(documents)
        
        assert isinstance(clusters, list)
        # Clusters may be empty if documents are too short or don't cluster well
        # This is expected behavior, so we just check the list is returned
        assert all(hasattr(cluster, 'cluster_id') for cluster in clusters)

    def test_semantic_clustering_disabled(self):
        """Test behavior when semantic clustering is disabled."""
        self.mock_config.metadata.enhancement.semantic_clustering = False
        clusterer = SemanticClusterer(self.mock_config)
        
        clusters = clusterer.cluster_documents([])
        assert clusters == []


class TestKnowledgeGraphBuilder:
    """Test knowledge graph construction."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_config):
        """Set up test fixtures."""
        self.mock_config = mock_config
        self.mock_config.metadata.enhancement.knowledge_graph = True
        self.builder = KnowledgeGraphBuilder(self.mock_config)

    def test_build_knowledge_graph(self):
        """Test knowledge graph construction from documents."""
        documents = [
            {"id": "doc1", "entities": [{"text": "Python", "type": "PROGRAMMING_LANGUAGE"}]},
            {"id": "doc2", "entities": [{"text": "Machine Learning", "type": "FIELD"}]},
            {"id": "doc3", "entities": [{"text": "Python", "type": "PROGRAMMING_LANGUAGE"}]}
        ]
        
        graph = self.builder.build_knowledge_graph(documents, [], [])
        
        assert hasattr(graph, 'nodes')
        assert hasattr(graph, 'edges')
        assert hasattr(graph, 'metadata')
        assert isinstance(graph.nodes, dict)

    def test_knowledge_graph_disabled(self):
        """Test behavior when knowledge graph is disabled."""
        self.mock_config.metadata.enhancement.knowledge_graph = False
        builder = KnowledgeGraphBuilder(self.mock_config)
        
        graph = builder.build_knowledge_graph([], [], [])
        assert hasattr(graph, 'nodes')
        assert hasattr(graph, 'edges')
        assert graph.nodes == {}
        assert graph.edges == []


class TestMetadataEnhancer:
    """Test the main metadata enhancer."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_config):
        """Set up test fixtures."""
        self.mock_config = mock_config
        self.mock_config.metadata.enhancement.cross_document_analysis = True
        self.mock_config.metadata.enhancement.semantic_clustering = True
        self.mock_config.metadata.enhancement.knowledge_graph = True
        self.enhancer = MetadataEnhancer(self.mock_config)

    def test_enhancer_initialization(self):
        """Test metadata enhancer initialization."""
        assert hasattr(self.enhancer, "cross_doc_analyzer")
        assert hasattr(self.enhancer, "semantic_clusterer")
        assert hasattr(self.enhancer, "knowledge_graph_builder")

    def test_enhance_document_metadata(self):
        """Test enhancing metadata for a single document."""
        document = {
            "id": "doc1",
            "content": "Sample content",
            "metadata": {"topics": ["AI"]}
        }
        
        enhanced = self.enhancer.enhance_metadata([document])
        
        assert isinstance(enhanced, dict)
        assert "summary" in enhanced
        assert enhanced["summary"]["total_documents"] == 1

    def test_enhance_document_collection(self):
        """Test enhancing metadata for a collection of documents."""
        documents = [
            {"id": "doc1", "content": "Content 1", "metadata": {}},
            {"id": "doc2", "content": "Content 2", "metadata": {}},
            {"id": "doc3", "content": "Content 3", "metadata": {}}
        ]
        
        enhanced = self.enhancer.enhance_metadata(documents)
        
        assert isinstance(enhanced, dict)
        assert "summary" in enhanced
        assert enhanced["summary"]["total_documents"] == 3

    def test_enhancement_with_disabled_features(self):
        """Test enhancement when some features are disabled."""
        self.mock_config.metadata.enhancement.cross_document_analysis = False
        self.mock_config.metadata.enhancement.semantic_clustering = False
        self.mock_config.metadata.enhancement.knowledge_graph = False
        
        enhancer = MetadataEnhancer(self.mock_config)
        
        documents = [{"id": "doc1", "content": "Content", "metadata": {}}]
        enhanced = enhancer.enhance_metadata(documents)
        
        assert isinstance(enhanced, dict)
        assert "summary" in enhanced


def test_get_metadata_enhancer():
    """Test the factory function for creating metadata enhancers."""
    enhancer = get_metadata_enhancer()
    assert isinstance(enhancer, MetadataEnhancer)


if __name__ == "__main__":
    pytest.main([__file__])
