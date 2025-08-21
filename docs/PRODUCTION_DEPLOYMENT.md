# Production Deployment Guide

This guide provides comprehensive instructions for deploying the RAG Document Processing Utility in production environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Monitoring & Observability](#monitoring--observability)
6. [Scaling & Performance](#scaling--performance)
7. [Security Considerations](#security-considerations)
8. [Backup & Recovery](#backup--recovery)
9. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **CPU**: Minimum 4 cores, recommended 8+ cores
- **Memory**: Minimum 8GB RAM, recommended 16GB+ RAM
- **Storage**: Minimum 50GB available disk space
- **Network**: Stable internet connection for external API calls

### Software Requirements

- **Operating System**: Linux (Ubuntu 20.04+ recommended), macOS, or Windows with WSL2
- **Docker**: Version 20.10+ with Docker Compose
- **Python**: Version 3.11+ (for development/debugging)
- **Git**: For version control and deployment

### External Dependencies

- **OpenAI API Key**: For LLM-powered metadata extraction
- **Vector Database**: ChromaDB, Pinecone, or Weaviate
- **Monitoring**: Prometheus, Grafana (optional)

## Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Chunkys0up7/RagPrep.git
cd RagPrep
```

### 2. Environment Variables

Create a `.env` file in the project root:

```bash
# Required
ENVIRONMENT=production
OPENAI_API_KEY=your_openai_api_key_here

# Optional
POSTGRES_PASSWORD=secure_password_here
REDIS_PASSWORD=secure_redis_password
LOG_LEVEL=INFO
METRICS_PORT=8001
API_PORT=8000
```

### 3. Configuration Files

Ensure your configuration files are properly set:

```bash
# Check configuration
python -c "from src.config import Config; print(Config().model_dump_json(indent=2))"
```

## Docker Deployment

### 1. Quick Start with Docker Compose

```bash
# Navigate to docker directory
cd docker

# Build and start services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f rag-prep-api
```

### 2. Production Docker Compose

For production, use the full docker-compose.yml:

```bash
# Start all services
docker-compose -f docker-compose.yml up -d

# Services include:
# - rag-prep-api: Main API service
# - rag-prep-worker: Background processing worker
# - redis: Caching and job queuing
# - postgres: Persistent storage
# - nginx: Reverse proxy (optional)
```

### 3. Health Checks

Verify deployment health:

```bash
# API health check
curl http://localhost:8000/health

# Metrics endpoint
curl http://localhost:8001/metrics

# Service status
docker-compose ps
```

### 4. Scaling Services

Scale worker processes for better performance:

```bash
# Scale worker service
docker-compose up -d --scale rag-prep-worker=3

# Check scaled services
docker-compose ps
```

## Kubernetes Deployment

### 1. Prerequisites

- Kubernetes cluster (1.20+)
- kubectl configured
- Helm 3.0+

### 2. Deploy with Helm

```bash
# Add Helm repository (if using custom charts)
helm repo add ragprep https://charts.ragprep.com

# Install the application
helm install ragprep ./helm-charts/ragprep \
  --namespace ragprep \
  --create-namespace \
  --set environment=production \
  --set openai.apiKey=your_api_key
```

### 3. Kubernetes Manifests

For manual deployment, use the provided manifests:

```bash
# Create namespace
kubectl create namespace ragprep

# Apply manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
```

## Monitoring & Observability

### 1. Prometheus Metrics

The application exposes Prometheus metrics on port 8001:

```bash
# View metrics
curl http://localhost:8001/metrics

# Key metrics:
# - rag_prep_documents_processed_total
# - rag_prep_processing_duration_seconds
# - rag_prep_api_response_time_seconds
# - rag_prep_system_cpu_usage
# - rag_prep_system_memory_usage
```

### 2. Grafana Dashboard

Import the provided Grafana dashboard:

```bash
# Dashboard JSON is available in:
# monitoring/grafana-dashboard.json
```

### 3. Log Aggregation

Configure centralized logging:

```bash
# View application logs
docker-compose logs -f rag-prep-api

# Log format: JSON structured logging
# Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### 4. Health Monitoring

Set up health checks and alerts:

```bash
# Health check endpoint
GET /health

# Response format:
{
  "status": "healthy",
  "service": "RAG Document Processing Utility API",
  "version": "0.1.0",
  "timestamp": 1640995200.0
}
```

## Scaling & Performance

### 1. Horizontal Scaling

Scale API and worker services:

```bash
# Scale API replicas
kubectl scale deployment ragprep-api --replicas=3

# Scale worker replicas
kubectl scale deployment ragprep-worker --replicas=5
```

### 2. Resource Limits

Configure resource limits in Kubernetes:

```yaml
resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "2Gi"
    cpu: "1000m"
```

### 3. Performance Tuning

Optimize performance settings:

```yaml
# config/config.yaml
performance:
  max_concurrent_processes: 8
  memory_limit_gb: 16
  enable_caching: true
  cache_ttl_hours: 24
```

### 4. Load Balancing

Configure load balancing:

```bash
# Nginx configuration
# docker/nginx.conf

# Kubernetes Ingress
# k8s/ingress.yaml
```

## Security Considerations

### 1. Network Security

```bash
# Restrict network access
docker-compose up -d --scale nginx=1

# Use internal networks
docker network create ragprep-internal
```

### 2. Authentication & Authorization

```bash
# API key authentication
curl -H "X-API-Key: your_api_key" \
     http://localhost:8000/process-document

# JWT tokens (if implemented)
curl -H "Authorization: Bearer <jwt_token>" \
     http://localhost:8000/api/v1/documents
```

### 3. Data Encryption

```bash
# Enable TLS/SSL
# docker/ssl/ directory for certificates

# Database encryption
# PostgreSQL with SSL enabled
```

### 4. Security Scanning

```bash
# Run security scans
docker run --rm -v $(pwd):/app aquasec/trivy fs /app

# Check for vulnerabilities
safety check
bandit -r src/
```

## Backup & Recovery

### 1. Data Backup

```bash
# Backup vector database
docker exec ragprep-postgres pg_dump -U ragprep ragprep > backup.sql

# Backup configuration
cp -r config/ backup/config-$(date +%Y%m%d)/

# Backup processed documents
tar -czf backup/documents-$(date +%Y%m%d).tar.gz output/
```

### 2. Recovery Procedures

```bash
# Restore database
docker exec -i ragprep-postgres psql -U ragprep ragprep < backup.sql

# Restore configuration
cp -r backup/config-$(date +%Y%m%d)/ config/

# Restore documents
tar -xzf backup/documents-$(date +%Y%m%d).tar.gz
```

### 3. Disaster Recovery

```bash
# Full system recovery
./scripts/deploy.py --rollback

# Data recovery
./scripts/recover_data.py --backup-date 20241201
```

## Troubleshooting

### 1. Common Issues

#### Service Won't Start

```bash
# Check logs
docker-compose logs rag-prep-api

# Check resource usage
docker stats

# Verify configuration
python -c "from src.config import Config; Config()"
```

#### High Memory Usage

```bash
# Check memory usage
docker stats

# Optimize memory settings
# config/config.yaml -> performance.memory_limit_gb
```

#### Slow Processing

```bash
# Check CPU usage
docker stats

# Scale workers
docker-compose up -d --scale rag-prep-worker=3

# Check processing metrics
curl http://localhost:8001/metrics
```

### 2. Debug Mode

Enable debug logging:

```bash
# Set log level
export LOG_LEVEL=DEBUG

# Restart services
docker-compose restart rag-prep-api
```

### 3. Performance Profiling

```bash
# Run performance tests
python performance_test.py

# Monitor system resources
python -m src.monitoring
```

### 4. Support Resources

- **Documentation**: [README.md](../README.md)
- **Issues**: [GitHub Issues](https://github.com/Chunkys0up7/RagPrep/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Chunkys0up7/RagPrep/discussions)

## Maintenance

### 1. Regular Updates

```bash
# Update application
git pull origin master
docker-compose build --no-cache
docker-compose up -d

# Update dependencies
pip install -r requirements.txt --upgrade
```

### 2. Health Monitoring

```bash
# Daily health check
curl -f http://localhost:8000/health

# Weekly performance review
python -m src.monitoring --generate-report
```

### 3. Log Rotation

```bash
# Configure log rotation
# logs/ directory with rotation policies

# Clean old logs
find logs/ -name "*.log" -mtime +30 -delete
```

## Conclusion

This production deployment guide covers the essential aspects of deploying and maintaining the RAG Document Processing Utility in production environments. For additional support or questions, please refer to the project documentation or create an issue on GitHub.

---

**Last Updated**: December 2024  
**Version**: 0.1.0  
**Maintainer**: RAGPrep Team
