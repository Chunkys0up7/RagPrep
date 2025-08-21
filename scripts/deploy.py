#!/usr/bin/env python3
"""
Production Deployment Script for RAG Document Processing Utility

This script automates the deployment process for production environments.
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ProductionDeployer:
    """Handles production deployment of the RAG Document Processing Utility."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.project_root = Path(__file__).parent.parent
        self.docker_dir = self.project_root / "docker"
        
        # Deployment configuration
        self.deployment_config = {
            "environment": os.getenv("ENVIRONMENT", "production"),
            "docker_compose_file": self.docker_dir / "docker-compose.yml",
            "health_check_url": "http://localhost:8000/health",
            "metrics_url": "http://localhost:8001/metrics",
            "deployment_timeout": 300,  # 5 minutes
        }
    
    def deploy(self, force: bool = False, skip_tests: bool = False) -> bool:
        """Main deployment method."""
        logger.info("🚀 Starting production deployment...")
        
        try:
            # Pre-deployment checks
            if not self._pre_deployment_checks():
                logger.error("❌ Pre-deployment checks failed")
                return False
            
            # Run tests (unless skipped)
            if not skip_tests and not self._run_tests():
                logger.error("❌ Tests failed")
                return False
            
            # Build Docker images
            if not self._build_images():
                logger.error("❌ Docker image build failed")
                return False
            
            # Deploy services
            if not self._deploy_services():
                logger.error("❌ Service deployment failed")
                return False
            
            # Health checks
            if not self._wait_for_health_checks():
                logger.error("❌ Health checks failed")
                return False
            
            # Post-deployment verification
            if not self._post_deployment_verification():
                logger.error("❌ Post-deployment verification failed")
                return False
            
            logger.info("✅ Production deployment completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Deployment failed: {e}")
            return False
    
    def _pre_deployment_checks(self) -> bool:
        """Run pre-deployment checks."""
        logger.info("🔍 Running pre-deployment checks...")
        
        checks = [
            ("Docker installed", self._check_docker_installed),
            ("Docker Compose installed", self._check_docker_compose_installed),
            ("Docker daemon running", self._check_docker_daemon),
            ("Required files exist", self._check_required_files),
            ("Environment variables", self._check_environment_variables),
        ]
        
        for check_name, check_func in checks:
            try:
                if not check_func():
                    logger.error(f"❌ {check_name} check failed")
                    return False
                logger.info(f"✅ {check_name} check passed")
            except Exception as e:
                logger.error(f"❌ {check_name} check failed: {e}")
                return False
        
        return True
    
    def _check_docker_installed(self) -> bool:
        """Check if Docker is installed."""
        try:
            result = subprocess.run(
                ["docker", "--version"], 
                capture_output=True, 
                text=True
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def _check_docker_compose_installed(self) -> bool:
        """Check if Docker Compose is installed."""
        try:
            result = subprocess.run(
                ["docker-compose", "--version"], 
                capture_output=True, 
                text=True
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def _check_docker_daemon(self) -> bool:
        """Check if Docker daemon is running."""
        try:
            result = subprocess.run(
                ["docker", "info"], 
                capture_output=True, 
                text=True
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def _check_required_files(self) -> bool:
        """Check if required files exist."""
        required_files = [
            self.docker_dir / "Dockerfile",
            self.docker_dir / "docker-compose.yml",
            self.project_root / "requirements.txt",
        ]
        
        for file_path in required_files:
            if not file_path.exists():
                logger.error(f"Required file not found: {file_path}")
                return False
        
        return True
    
    def _check_environment_variables(self) -> bool:
        """Check if required environment variables are set."""
        required_vars = ["ENVIRONMENT"]
        optional_vars = ["OPENAI_API_KEY", "POSTGRES_PASSWORD"]
        
        # Check required variables
        for var in required_vars:
            if not os.getenv(var):
                logger.error(f"Required environment variable not set: {var}")
                return False
        
        # Log optional variables
        for var in optional_vars:
            if not os.getenv(var):
                logger.warning(f"Optional environment variable not set: {var}")
        
        return True
    
    def _run_tests(self) -> bool:
        """Run the test suite."""
        logger.info("🧪 Running test suite...")
        
        try:
            # Run basic tests
            result = subprocess.run(
                [sys.executable, "run_tests_simple.py"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Tests failed: {result.stderr}")
                return False
            
            logger.info("✅ Tests passed")
            return True
            
        except Exception as e:
            logger.error(f"Error running tests: {e}")
            return False
    
    def _build_images(self) -> bool:
        """Build Docker images."""
        logger.info("🔨 Building Docker images...")
        
        try:
            # Build the main application image
            result = subprocess.run(
                ["docker-compose", "-f", str(self.deployment_config["docker_compose_file"]), "build"],
                cwd=self.docker_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Docker build failed: {result.stderr}")
                return False
            
            logger.info("✅ Docker images built successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error building Docker images: {e}")
            return False
    
    def _deploy_services(self) -> bool:
        """Deploy services using Docker Compose."""
        logger.info("🚀 Deploying services...")
        
        try:
            # Start services
            result = subprocess.run(
                ["docker-compose", "-f", str(self.deployment_config["docker_compose_file"]), "up", "-d"],
                cwd=self.docker_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Service deployment failed: {result.stderr}")
                return False
            
            logger.info("✅ Services deployed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error deploying services: {e}")
            return False
    
    def _wait_for_health_checks(self) -> bool:
        """Wait for health checks to pass."""
        logger.info("🏥 Waiting for health checks...")
        
        start_time = time.time()
        timeout = self.deployment_config["deployment_timeout"]
        
        while time.time() - start_time < timeout:
            try:
                # Check API health
                result = subprocess.run(
                    ["curl", "-f", self.deployment_config["health_check_url"]],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    logger.info("✅ Health checks passed")
                    return True
                
                logger.info("⏳ Waiting for services to be ready...")
                time.sleep(10)
                
            except subprocess.TimeoutExpired:
                logger.warning("Health check timeout, retrying...")
                continue
            except Exception as e:
                logger.warning(f"Health check error: {e}")
                time.sleep(10)
        
        logger.error("❌ Health checks timed out")
        return False
    
    def _post_deployment_verification(self) -> bool:
        """Run post-deployment verification."""
        logger.info("🔍 Running post-deployment verification...")
        
        try:
            # Check service status
            result = subprocess.run(
                ["docker-compose", "-f", str(self.deployment_config["docker_compose_file"]), "ps"],
                cwd=self.docker_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Service status check failed: {result.stderr}")
                return False
            
            # Check if all services are running
            if "Up" not in result.stdout:
                logger.error("Not all services are running")
                return False
            
            logger.info("✅ Post-deployment verification passed")
            return True
            
        except Exception as e:
            logger.error(f"Error in post-deployment verification: {e}")
            return False
    
    def rollback(self) -> bool:
        """Rollback to previous deployment."""
        logger.info("🔄 Rolling back deployment...")
        
        try:
            # Stop and remove services
            result = subprocess.run(
                ["docker-compose", "-f", str(self.deployment_config["docker-compose_file"]), "down"],
                cwd=self.docker_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Rollback failed: {result.stderr}")
                return False
            
            logger.info("✅ Rollback completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during rollback: {e}")
            return False
    
    def get_deployment_status(self) -> Dict[str, any]:
        """Get current deployment status."""
        try:
            result = subprocess.run(
                ["docker-compose", "-f", str(self.deployment_config["docker-compose_file"]), "ps"],
                cwd=self.docker_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                return {
                    "status": "running",
                    "services": result.stdout,
                    "timestamp": time.time()
                }
            else:
                return {
                    "status": "stopped",
                    "error": result.stderr,
                    "timestamp": time.time()
                }
                
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": time.time()
            }


def main():
    """Main deployment script entry point."""
    parser = argparse.ArgumentParser(description="Production Deployment Script")
    parser.add_argument("--force", action="store_true", help="Force deployment without confirmation")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running tests before deployment")
    parser.add_argument("--rollback", action="store_true", help="Rollback to previous deployment")
    parser.add_argument("--status", action="store_true", help="Show deployment status")
    parser.add_argument("--config", help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # Create deployer
    deployer = ProductionDeployer(args.config)
    
    try:
        if args.rollback:
            success = deployer.rollback()
            sys.exit(0 if success else 1)
        elif args.status:
            status = deployer.get_deployment_status()
            print(f"Deployment Status: {status['status']}")
            if 'services' in status:
                print(status['services'])
            sys.exit(0)
        else:
            # Confirm deployment (unless forced)
            if not args.force:
                response = input("Are you sure you want to deploy to production? (yes/no): ")
                if response.lower() != "yes":
                    logger.info("Deployment cancelled")
                    sys.exit(0)
            
            # Run deployment
            success = deployer.deploy(force=args.force, skip_tests=args.skip_tests)
            sys.exit(0 if success else 1)
            
    except KeyboardInterrupt:
        logger.info("Deployment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Deployment script error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
