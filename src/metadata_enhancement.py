"""
Advanced Metadata Enhancement Module

Provides sophisticated metadata enhancement capabilities including cross-document
relationship mapping, semantic similarity clustering, and knowledge graph construction.
"""

import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class DocumentRelationship:
    """Represents a relationship between documents."""
    
    source_doc_id: str
    target_doc_id: str
    relationship_type: str  # "similar", "related", "references", "cites"
    strength: float  # 0.0 to 1.0
    evidence: List[str]  # List of evidence for the relationship
    metadata: Dict[str, Any]


@dataclass
class SemanticCluster:
    """Represents a cluster of semantically similar documents."""
    
    cluster_id: str
    document_ids: List[str]
    centroid_vector: np.ndarray
    keywords: List[str]
    topic: str
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class KnowledgeGraph:
    """Represents a knowledge graph of document relationships."""
    
    nodes: Dict[str, Dict[str, Any]]  # Document nodes
    edges: List[Tuple[str, str, Dict[str, Any]]]  # Document relationships
    clusters: List[SemanticCluster]
    metadata: Dict[str, Any]


class CrossDocumentAnalyzer:
    """Analyzes relationships between multiple documents."""
    
    def __init__(self, config=None):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def analyze_relationships(self, documents: List[Dict[str, Any]]) -> List[DocumentRelationship]:
        """Analyze relationships between documents."""
        relationships = []
        
        if len(documents) < 2:
            return relationships
        
        # Extract text content for analysis
        doc_texts = []
        doc_ids = []
        
        for doc in documents:
            if 'text_content' in doc:
                doc_texts.append(doc['text_content'])
                doc_ids.append(doc.get('document_id', str(hash(doc['text_content']))))
        
        if len(doc_texts) < 2:
            return relationships
        
        # Vectorize documents
        try:
            vectors = self.vectorizer.fit_transform(doc_texts)
            similarity_matrix = cosine_similarity(vectors)
            
            # Find relationships based on similarity
            for i in range(len(doc_ids)):
                for j in range(i + 1, len(doc_ids)):
                    similarity = similarity_matrix[i][j]
                    
                    if similarity > 0.3:  # Threshold for relationship
                        relationship_type = self._determine_relationship_type(similarity)
                        evidence = self._generate_relationship_evidence(
                            doc_texts[i], doc_texts[j], similarity
                        )
                        
                        relationship = DocumentRelationship(
                            source_doc_id=doc_ids[i],
                            target_doc_id=doc_ids[j],
                            relationship_type=relationship_type,
                            strength=similarity,
                            evidence=evidence,
                            metadata={
                                "similarity_score": similarity,
                                "analysis_method": "tfidf_cosine"
                            }
                        )
                        relationships.append(relationship)
        
        except Exception as e:
            self.logger.error(f"Relationship analysis failed: {e}")
        
        return relationships
    
    def _determine_relationship_type(self, similarity: float) -> str:
        """Determine the type of relationship based on similarity score."""
        if similarity > 0.8:
            return "very_similar"
        elif similarity > 0.6:
            return "similar"
        elif similarity > 0.4:
            return "related"
        else:
            return "weakly_related"
    
    def _generate_relationship_evidence(self, text1: str, text2: str, similarity: float) -> List[str]:
        """Generate evidence for the relationship."""
        evidence = []
        
        # Find common keywords
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        common_words = words1.intersection(words2)
        
        if common_words:
            evidence.append(f"Common keywords: {', '.join(list(common_words)[:5])}")
        
        evidence.append(f"Similarity score: {similarity:.3f}")
        
        return evidence


class SemanticClusterer:
    """Clusters documents based on semantic similarity."""
    
    def __init__(self, config=None):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def cluster_documents(self, documents: List[Dict[str, Any]], 
                         eps: float = 0.3, min_samples: int = 2) -> List[SemanticCluster]:
        """Cluster documents based on semantic similarity."""
        clusters = []
        
        if len(documents) < 2:
            return clusters
        
        # Extract text content
        doc_texts = []
        doc_ids = []
        
        for doc in documents:
            if 'text_content' in doc:
                doc_texts.append(doc['text_content'])
                doc_ids.append(doc.get('document_id', str(hash(doc['text_content']))))
        
        if len(doc_texts) < 2:
            return clusters
        
        try:
            # Vectorize documents
            vectors = self.vectorizer.fit_transform(doc_texts)
            
            # Perform clustering
            clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
            cluster_labels = clustering.fit_predict(vectors)
            
            # Group documents by cluster
            cluster_groups = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                if label >= 0:  # Skip noise points
                    cluster_groups[label].append((doc_ids[i], vectors[i]))
            
            # Create cluster objects
            for cluster_id, group in cluster_groups.items():
                doc_ids_in_cluster = [doc_id for doc_id, _ in group]
                cluster_vectors = [vector for _, vector in group]
                
                # Calculate centroid
                centroid = np.mean(cluster_vectors, axis=0)
                
                # Extract keywords
                keywords = self._extract_cluster_keywords(doc_texts, doc_ids_in_cluster)
                
                # Determine topic
                topic = self._determine_cluster_topic(keywords, doc_texts, doc_ids_in_cluster)
                
                cluster = SemanticCluster(
                    cluster_id=f"cluster_{cluster_id}",
                    document_ids=doc_ids_in_cluster,
                    centroid_vector=centroid,
                    keywords=keywords,
                    topic=topic,
                    confidence=len(group) / len(doc_texts),
                    metadata={
                        "cluster_size": len(group),
                        "clustering_method": "dbscan",
                        "eps": eps,
                        "min_samples": min_samples
                    }
                )
                clusters.append(cluster)
        
        except Exception as e:
            self.logger.error(f"Semantic clustering failed: {e}")
        
        return clusters
    
    def _extract_cluster_keywords(self, all_texts: List[str], 
                                 cluster_doc_ids: List[str]) -> List[str]:
        """Extract keywords for a cluster."""
        # This is a simplified keyword extraction
        # In production, you'd use more sophisticated methods
        keywords = []
        
        # Find common words across cluster documents
        word_freq = defaultdict(int)
        for doc_id in cluster_doc_ids:
            # Find the corresponding text
            for text in all_texts:
                if str(hash(text)) in doc_id or doc_id in str(hash(text)):
                    words = re.findall(r'\b\w+\b', text.lower())
                    for word in words:
                        if len(word) > 3:  # Skip short words
                            word_freq[word] += 1
        
        # Get top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        keywords = [word for word, _ in sorted_words[:10]]
        
        return keywords
    
    def _determine_cluster_topic(self, keywords: List[str], 
                                all_texts: List[str], 
                                cluster_doc_ids: List[str]) -> str:
        """Determine the main topic of a cluster."""
        if not keywords:
            return "General"
        
        # Use the most frequent keyword as topic
        return keywords[0].title() if keywords else "General"


class KnowledgeGraphBuilder:
    """Builds a knowledge graph from document relationships and clusters."""
    
    def __init__(self, config=None):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def build_knowledge_graph(self, documents: List[Dict[str, Any]], 
                             relationships: List[DocumentRelationship],
                             clusters: List[SemanticCluster]) -> KnowledgeGraph:
        """Build a knowledge graph from documents and their relationships."""
        
        # Create document nodes
        nodes = {}
        for doc in documents:
            doc_id = doc.get('document_id', str(hash(doc.get('text_content', ''))))
            nodes[doc_id] = {
                'id': doc_id,
                'title': doc.get('title', 'Untitled'),
                'type': doc.get('type', 'document'),
                'metadata': doc.get('metadata', {}),
                'cluster_id': self._find_cluster_id(doc_id, clusters)
            }
        
        # Create edges from relationships
        edges = []
        for rel in relationships:
            edge_data = {
                'type': rel.relationship_type,
                'strength': rel.strength,
                'evidence': rel.evidence,
                'metadata': rel.metadata
            }
            edges.append((rel.source_doc_id, rel.target_doc_id, edge_data))
        
        # Create knowledge graph
        knowledge_graph = KnowledgeGraph(
            nodes=nodes,
            edges=edges,
            clusters=clusters,
            metadata={
                'total_documents': len(documents),
                'total_relationships': len(relationships),
                'total_clusters': len(clusters),
                'graph_type': 'document_relationship'
            }
        )
        
        return knowledge_graph
    
    def _find_cluster_id(self, doc_id: str, clusters: List[SemanticCluster]) -> Optional[str]:
        """Find which cluster a document belongs to."""
        for cluster in clusters:
            if doc_id in cluster.document_ids:
                return cluster.cluster_id
        return None


class MetadataEnhancer:
    """Main orchestrator for advanced metadata enhancement."""
    
    def __init__(self, config=None):
        self.config = config
        self.cross_doc_analyzer = CrossDocumentAnalyzer(config)
        self.semantic_clusterer = SemanticClusterer(config)
        self.knowledge_graph_builder = KnowledgeGraphBuilder(config)
        self.logger = logging.getLogger(__name__)
    
    def enhance_metadata(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Enhance metadata for a collection of documents."""
        start_time = time.time()
        
        try:
            # Analyze cross-document relationships
            relationships = self.cross_doc_analyzer.analyze_relationships(documents)
            
            # Perform semantic clustering
            clusters = self.semantic_clusterer.cluster_documents(documents)
            
            # Build knowledge graph
            knowledge_graph = self.knowledge_graph_builder.build_knowledge_graph(
                documents, relationships, clusters
            )
            
            # Generate enhanced metadata
            enhanced_metadata = {
                'relationships': [self._relationship_to_dict(rel) for rel in relationships],
                'clusters': [self._cluster_to_dict(cluster) for cluster in clusters],
                'knowledge_graph': self._knowledge_graph_to_dict(knowledge_graph),
                'summary': {
                    'total_documents': len(documents),
                    'total_relationships': len(relationships),
                    'total_clusters': len(clusters),
                    'processing_time': time.time() - start_time
                }
            }
            
            return enhanced_metadata
        
        except Exception as e:
            self.logger.error(f"Metadata enhancement failed: {e}")
            return {
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def _relationship_to_dict(self, rel: DocumentRelationship) -> Dict[str, Any]:
        """Convert DocumentRelationship to dictionary."""
        return {
            'source_doc_id': rel.source_doc_id,
            'target_doc_id': rel.target_doc_id,
            'relationship_type': rel.relationship_type,
            'strength': rel.strength,
            'evidence': rel.evidence,
            'metadata': rel.metadata
        }
    
    def _cluster_to_dict(self, cluster: SemanticCluster) -> Dict[str, Any]:
        """Convert SemanticCluster to dictionary."""
        return {
            'cluster_id': cluster.cluster_id,
            'document_ids': cluster.document_ids,
            'keywords': cluster.keywords,
            'topic': cluster.topic,
            'confidence': cluster.confidence,
            'metadata': cluster.metadata
        }
    
    def _knowledge_graph_to_dict(self, kg: KnowledgeGraph) -> Dict[str, Any]:
        """Convert KnowledgeGraph to dictionary."""
        return {
            'nodes': kg.nodes,
            'edges': kg.edges,
            'clusters': [self._cluster_to_dict(c) for c in kg.clusters],
            'metadata': kg.metadata
        }


# Convenience function
def get_metadata_enhancer(config=None) -> MetadataEnhancer:
    """Get a configured metadata enhancer instance."""
    return MetadataEnhancer(config)
