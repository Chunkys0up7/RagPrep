"""
Monitoring and Performance Optimization Module

Provides comprehensive monitoring, performance tracking, and optimization capabilities
for production deployment of the RAG Document Processing Utility.
"""

import asyncio
import json
import logging
import os
import psutil
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, deque

import numpy as np
from prometheus_client import Counter, Gauge, Histogram, start_http_server

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System resource metrics."""
    
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_usage_percent: float
    disk_used_gb: float
    disk_free_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    process_count: int


@dataclass
class PerformanceMetrics:
    """Application performance metrics."""
    
    timestamp: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    requests_per_second: float
    error_rate: float
    active_connections: int


@dataclass
class ProcessingMetrics:
    """Document processing metrics."""
    
    timestamp: float
    documents_processed: int
    total_processing_time: float
    average_processing_time: float
    chunks_generated: int
    quality_scores: List[float]
    error_count: int
    success_rate: float


class MetricsCollector:
    """Collects and manages system and application metrics."""
    
    def __init__(self, config=None):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize Prometheus metrics
        self._init_prometheus_metrics()
        
        # Metrics storage
        self.system_metrics_history = deque(maxlen=1000)
        self.performance_metrics_history = deque(maxlen=1000)
        self.processing_metrics_history = deque(maxlen=1000)
        
        # Performance tracking
        self.request_times = deque(maxlen=1000)
        self.processing_times = deque(maxlen=1000)
        self.error_counts = defaultdict(int)
        
        # Start metrics collection
        self.collection_active = True
        self.collection_interval = 30  # seconds
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        try:
            # Counters
            self.documents_processed_total = Counter(
                'rag_prep_documents_processed_total',
                'Total number of documents processed'
            )
            self.processing_errors_total = Counter(
                'rag_prep_processing_errors_total',
                'Total number of processing errors'
            )
            self.api_requests_total = Counter(
                'rag_prep_api_requests_total',
                'Total number of API requests'
            )
            
            # Gauges
            self.active_connections = Gauge(
                'rag_prep_active_connections',
                'Number of active connections'
            )
            self.processing_queue_size = Gauge(
                'rag_prep_processing_queue_size',
                'Size of processing queue'
            )
            self.system_cpu_usage = Gauge(
                'rag_prep_system_cpu_usage',
                'System CPU usage percentage'
            )
            self.system_memory_usage = Gauge(
                'rag_prep_system_memory_usage',
                'System memory usage percentage'
            )
            
            # Histograms
            self.processing_duration = Histogram(
                'rag_prep_processing_duration_seconds',
                'Document processing duration in seconds'
            )
            self.api_response_time = Histogram(
                'rag_prep_api_response_time_seconds',
                'API response time in seconds'
            )
            
            logger.info("Prometheus metrics initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize Prometheus metrics: {e}")
    
    def start_metrics_server(self, port: int = 8001):
        """Start Prometheus metrics server."""
        try:
            start_http_server(port)
            logger.info(f"Prometheus metrics server started on port {port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024**3)
            memory_available_gb = memory.available / (1024**3)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_usage_percent = disk.percent
            disk_used_gb = disk.used / (1024**3)
            disk_free_gb = disk.free / (1024**3)
            
            # Network metrics
            network = psutil.net_io_counters()
            network_bytes_sent = network.bytes_sent
            network_bytes_recv = network.bytes_recv
            
            # Process count
            process_count = len(psutil.pids())
            
            metrics = SystemMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_gb=memory_used_gb,
                memory_available_gb=memory_available_gb,
                disk_usage_percent=disk_usage_percent,
                disk_used_gb=disk_used_gb,
                disk_free_gb=disk_free_gb,
                network_bytes_sent=network_bytes_sent,
                network_bytes_recv=network_bytes_recv,
                process_count=process_count
            )
            
            # Update Prometheus metrics
            self.system_cpu_usage.set(cpu_percent)
            self.system_memory_usage.set(memory_percent)
            
            # Store in history
            self.system_metrics_history.append(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            return None
    
    def collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        try:
            # Calculate metrics from stored data
            total_requests = len(self.request_times)
            successful_requests = total_requests - sum(self.error_counts.values())
            failed_requests = sum(self.error_counts.values())
            
            if total_requests > 0:
                average_response_time = np.mean(list(self.request_times))
                error_rate = failed_requests / total_requests
            else:
                average_response_time = 0.0
                error_rate = 0.0
            
            # Calculate requests per second (last minute)
            current_time = time.time()
            recent_requests = sum(1 for t in self.request_times 
                                if current_time - t < 60)
            requests_per_second = recent_requests / 60.0
            
            metrics = PerformanceMetrics(
                timestamp=current_time,
                total_requests=total_requests,
                successful_requests=successful_requests,
                failed_requests=failed_requests,
                average_response_time=average_response_time,
                requests_per_second=requests_per_second,
                error_rate=error_rate,
                active_connections=0  # Would be tracked separately
            )
            
            # Store in history
            self.performance_metrics_history.append(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting performance metrics: {e}")
            return None
    
    def collect_processing_metrics(self) -> ProcessingMetrics:
        """Collect current processing metrics."""
        try:
            # Calculate metrics from stored data
            total_processing_time = sum(self.processing_times)
            documents_processed = len(self.processing_times)
            
            if documents_processed > 0:
                average_processing_time = total_processing_time / documents_processed
                success_rate = 1.0 - (sum(self.error_counts.values()) / documents_processed)
            else:
                average_processing_time = 0.0
                success_rate = 1.0
            
            metrics = ProcessingMetrics(
                timestamp=time.time(),
                documents_processed=documents_processed,
                total_processing_time=total_processing_time,
                average_processing_time=average_processing_time,
                chunks_generated=0,  # Would be tracked separately
                quality_scores=[],  # Would be tracked separately
                error_count=sum(self.error_counts.values()),
                success_rate=success_rate
            )
            
            # Store in history
            self.processing_metrics_history.append(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting processing metrics: {e}")
            return None
    
    def record_request_time(self, duration: float):
        """Record API request duration."""
        self.request_times.append(duration)
        self.api_response_time.observe(duration)
        self.api_requests_total.inc()
    
    def record_processing_time(self, duration: float):
        """Record document processing duration."""
        self.processing_times.append(duration)
        self.processing_duration.observe(duration)
        self.documents_processed_total.inc()
    
    def record_error(self, error_type: str):
        """Record an error occurrence."""
        self.error_counts[error_type] += 1
        self.processing_errors_total.inc()
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of all collected metrics."""
        try:
            # Get latest metrics
            latest_system = self.system_metrics_history[-1] if self.system_metrics_history else None
            latest_performance = self.performance_metrics_history[-1] if self.performance_metrics_history else None
            latest_processing = self.processing_metrics_history[-1] if self.processing_metrics_history else None
            
            # Calculate trends
            system_trends = self._calculate_system_trends()
            performance_trends = self._calculate_performance_trends()
            
            summary = {
                "timestamp": time.time(),
                "system": {
                    "current": asdict(latest_system) if latest_system else None,
                    "trends": system_trends
                },
                "performance": {
                    "current": asdict(latest_performance) if latest_performance else None,
                    "trends": performance_trends
                },
                "processing": {
                    "current": asdict(latest_processing) if latest_processing else None
                },
                "history": {
                    "system_metrics_count": len(self.system_metrics_history),
                    "performance_metrics_count": len(self.performance_metrics_history),
                    "processing_metrics_count": len(self.processing_metrics_history)
                }
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating metrics summary: {e}")
            return {"error": str(e)}
    
    def _calculate_system_trends(self) -> Dict[str, Any]:
        """Calculate system metrics trends."""
        if len(self.system_metrics_history) < 2:
            return {}
        
        try:
            recent_metrics = list(self.system_metrics_history)[-10:]  # Last 10 measurements
            
            cpu_trend = np.polyfit([m.timestamp for m in recent_metrics], 
                                 [m.cpu_percent for m in recent_metrics], 1)[0]
            memory_trend = np.polyfit([m.timestamp for m in recent_metrics], 
                                    [m.memory_percent for m in recent_metrics], 1)[0]
            
            return {
                "cpu_trend": cpu_trend,
                "memory_trend": memory_trend,
                "trend_direction": {
                    "cpu": "increasing" if cpu_trend > 0 else "decreasing",
                    "memory": "increasing" if memory_trend > 0 else "decreasing"
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating system trends: {e}")
            return {}
    
    def _calculate_performance_trends(self) -> Dict[str, Any]:
        """Calculate performance metrics trends."""
        if len(self.performance_metrics_history) < 2:
            return {}
        
        try:
            recent_metrics = list(self.performance_metrics_history)[-10:]  # Last 10 measurements
            
            response_time_trend = np.polyfit([m.timestamp for m in recent_metrics], 
                                          [m.average_response_time for m in recent_metrics], 1)[0]
            
            return {
                "response_time_trend": response_time_trend,
                "trend_direction": {
                    "response_time": "increasing" if response_time_trend > 0 else "decreasing"
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating performance trends: {e}")
            return {}
    
    def save_metrics_to_file(self, file_path: str):
        """Save collected metrics to a JSON file."""
        try:
            summary = self.get_metrics_summary()
            
            with open(file_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            self.logger.info(f"Metrics saved to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving metrics to file: {e}")
    
    def start_continuous_collection(self):
        """Start continuous metrics collection."""
        async def collect_loop():
            while self.collection_active:
                try:
                    self.collect_system_metrics()
                    self.collect_performance_metrics()
                    self.collect_processing_metrics()
                    
                    await asyncio.sleep(self.collection_interval)
                    
                except Exception as e:
                    self.logger.error(f"Error in metrics collection loop: {e}")
                    await asyncio.sleep(self.collection_interval)
        
        # Start the collection loop
        asyncio.create_task(collect_loop())
        self.logger.info("Continuous metrics collection started")
    
    def stop_continuous_collection(self):
        """Stop continuous metrics collection."""
        self.collection_active = False
        self.logger.info("Continuous metrics collection stopped")


class PerformanceOptimizer:
    """Provides performance optimization recommendations and actions."""
    
    def __init__(self, config=None):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.optimization_history = []
    
    def analyze_performance(self, metrics_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance metrics and provide optimization recommendations."""
        recommendations = []
        warnings = []
        
        try:
            system_metrics = metrics_summary.get("system", {}).get("current")
            performance_metrics = metrics_summary.get("performance", {}).get("current")
            
            if system_metrics:
                # CPU analysis
                if system_metrics.get("cpu_percent", 0) > 80:
                    recommendations.append({
                        "type": "cpu_optimization",
                        "priority": "high",
                        "message": "CPU usage is high (>80%). Consider scaling horizontally or optimizing processing algorithms.",
                        "actions": [
                            "Increase worker processes",
                            "Implement request queuing",
                            "Optimize document processing algorithms"
                        ]
                    })
                
                # Memory analysis
                if system_metrics.get("memory_percent", 0) > 85:
                    recommendations.append({
                        "type": "memory_optimization",
                        "priority": "high",
                        "message": "Memory usage is high (>85%). Consider increasing memory or implementing caching.",
                        "actions": [
                            "Increase container memory limits",
                            "Implement Redis caching",
                            "Optimize data structures"
                        ]
                    })
                
                # Disk analysis
                if system_metrics.get("disk_usage_percent", 0) > 90:
                    warnings.append({
                        "type": "disk_warning",
                        "priority": "critical",
                        "message": "Disk usage is critical (>90%). Immediate action required.",
                        "actions": [
                            "Clean up temporary files",
                            "Increase disk space",
                            "Implement log rotation"
                        ]
                    })
            
            if performance_metrics:
                # Response time analysis
                if performance_metrics.get("average_response_time", 0) > 5.0:
                    recommendations.append({
                        "type": "response_time_optimization",
                        "priority": "medium",
                        "message": "Average response time is high (>5s). Consider performance tuning.",
                        "actions": [
                            "Implement caching",
                            "Optimize database queries",
                            "Use async processing"
                        ]
                    })
                
                # Error rate analysis
                if performance_metrics.get("error_rate", 0) > 0.05:
                    warnings.append({
                        "type": "error_rate_warning",
                        "priority": "high",
                        "message": "Error rate is high (>5%). Investigate system stability.",
                        "actions": [
                            "Check error logs",
                            "Verify external dependencies",
                            "Implement circuit breakers"
                        ]
                    })
            
            analysis_result = {
                "timestamp": time.time(),
                "recommendations": recommendations,
                "warnings": warnings,
                "summary": {
                    "total_recommendations": len(recommendations),
                    "total_warnings": len(warnings),
                    "critical_issues": len([w for w in warnings if w["priority"] == "critical"]),
                    "high_priority": len([r for r in recommendations if r["priority"] == "high"])
                }
            }
            
            # Store analysis result
            self.optimization_history.append(analysis_result)
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error analyzing performance: {e}")
            return {"error": str(e)}
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization analysis history."""
        return self.optimization_history


# Convenience functions
def get_metrics_collector(config=None) -> MetricsCollector:
    """Get a configured metrics collector instance."""
    return MetricsCollector(config)


def get_performance_optimizer(config=None) -> PerformanceOptimizer:
    """Get a configured performance optimizer instance."""
    return PerformanceOptimizer(config)
