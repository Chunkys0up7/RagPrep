"""
Security module for RAG Document Processing Utility.

This module provides comprehensive security measures for handling untrusted input files,
including validation, sanitization, and security checks.
"""

import os
import hashlib
import mimetypes
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
import magic
import yara
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SecurityCheck:
    """Result of a security check operation."""
    passed: bool
    threat_level: str  # 'low', 'medium', 'high', 'critical'
    details: str
    recommendations: List[str]


@dataclass
class FileSecurityProfile:
    """Comprehensive security profile for a file."""
    file_path: Path
    file_hash: str
    mime_type: str
    file_size: int
    security_checks: List[SecurityCheck]
    overall_threat_level: str
    is_safe: bool
    warnings: List[str]


class FileValidator:
    """Validates file types, sizes, and basic security properties."""
    
    def __init__(self, config):
        self.config = config
        self.max_file_size = config.security.max_file_size_mb * 1024 * 1024
        self.allowed_extensions = set(config.security.allowed_file_extensions)
        self.allowed_mime_types = set(config.security.allowed_mime_types)
        
    def validate_file(self, file_path: Path) -> Tuple[bool, List[str]]:
        """
        Validate a file for basic security properties.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check if file exists
        if not file_path.exists():
            errors.append(f"File does not exist: {file_path}")
            return False, errors
            
        # Check file size
        try:
            file_size = file_path.stat().st_size
            if file_size > self.max_file_size:
                errors.append(f"File size {file_size} exceeds maximum allowed size {self.max_file_size}")
        except OSError as e:
            errors.append(f"Cannot access file size: {e}")
            
        # Check file extension
        file_ext = file_path.suffix.lower()
        if file_ext not in self.allowed_extensions:
            errors.append(f"File extension '{file_ext}' is not allowed")
            
        # Check MIME type
        try:
            mime_type, _ = mimetypes.guess_type(str(file_path))
            if mime_type and mime_type not in self.allowed_mime_types:
                errors.append(f"MIME type '{mime_type}' is not allowed")
        except Exception as e:
            errors.append(f"Could not determine MIME type: {e}")
            
        return len(errors) == 0, errors


class FileSanitizer:
    """Sanitizes file names and content for safe processing."""
    
    def __init__(self, config):
        self.config = config
        
    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize a filename to prevent path traversal and other attacks.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Remove path separators and dangerous characters
        dangerous_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        sanitized = filename
        
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '_')
            
        # Remove null bytes and control characters
        sanitized = ''.join(char for char in sanitized if ord(char) >= 32)
        
        # Limit length
        max_length = self.config.security.max_filename_length
        if len(sanitized) > max_length:
            name, ext = os.path.splitext(sanitized)
            sanitized = name[:max_length-len(ext)] + ext
            
        return sanitized or "unnamed_file"
        
    def create_safe_output_path(self, input_path: Path, output_dir: Path) -> Path:
        """
        Create a safe output path for processed files.
        
        Args:
            input_path: Original input file path
            output_dir: Output directory
            
        Returns:
            Safe output path
        """
        safe_filename = self.sanitize_filename(input_path.name)
        return output_dir / safe_filename


class ContentAnalyzer:
    """Analyzes file content for potential security threats."""
    
    def __init__(self, config):
        self.config = config
        self.suspicious_patterns = self._load_suspicious_patterns()
        
    def _load_suspicious_patterns(self) -> Dict[str, str]:
        """Load YARA rules for detecting suspicious content patterns."""
        patterns = {
            "executable_content": """
                rule executable_content {
                    strings:
                        $mz_header = "MZ"
                        $pe_header = "PE"
                    condition:
                        $mz_header at 0 and $pe_header at 0x3C
                }
            """,
            "script_content": """
                rule script_content {
                    strings:
                        $php_tag = "<?php"
                        $js_code = "javascript:"
                        $vb_code = "vbscript:"
                    condition:
                        any of them
                }
            """,
            "macro_content": """
                rule macro_content {
                    strings:
                        $vba_start = "Sub "
                        $vba_function = "Function "
                        $vba_macro = "Macro"
                    condition:
                        any of them
                }
            """
        }
        return patterns
        
    def analyze_content(self, file_path: Path) -> List[SecurityCheck]:
        """
        Analyze file content for security threats.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            List of security check results
        """
        checks = []
        
        try:
            # Read file content (first 1MB for performance)
            with open(file_path, 'rb') as f:
                content = f.read(1024 * 1024)
                
            # Check for executable content
            if content.startswith(b'MZ'):
                checks.append(SecurityCheck(
                    passed=False,
                    threat_level='high',
                    details="File appears to contain executable content (MZ header detected)",
                    recommendations=["Reject file", "Scan with antivirus software"]
                ))
                
            # Check for script content
            content_str = content.decode('utf-8', errors='ignore')
            if any(pattern in content_str.lower() for pattern in ['<?php', 'javascript:', 'vbscript:']):
                checks.append(SecurityCheck(
                    passed=False,
                    threat_level='medium',
                    details="File contains script-like content",
                    recommendations=["Review content manually", "Validate script safety"]
                ))
                
            # Check for macro content
            if any(pattern in content_str.lower() for pattern in ['sub ', 'function ', 'macro']):
                checks.append(SecurityCheck(
                    passed=False,
                    threat_level='medium',
                    details="File contains macro-like content",
                    recommendations=["Review content manually", "Validate macro safety"]
                ))
                
        except Exception as e:
            checks.append(SecurityCheck(
                passed=False,
                threat_level='low',
                details=f"Could not analyze file content: {e}",
                recommendations=["Review file manually", "Check file permissions"]
            ))
            
        return checks


class SecurityManager:
    """Main security manager that orchestrates all security checks."""
    
    def __init__(self, config):
        self.config = config
        self.validator = FileValidator(config)
        self.sanitizer = FileSanitizer(config)
        self.analyzer = ContentAnalyzer(config)
        
    def assess_file_security(self, file_path: Path) -> FileSecurityProfile:
        """
        Perform comprehensive security assessment of a file.
        
        Args:
            file_path: Path to the file to assess
            
        Returns:
            Comprehensive security profile
        """
        # Basic validation
        is_valid, validation_errors = self.validator.validate_file(file_path)
        
        # File information
        try:
            file_size = file_path.stat().st_size
            with open(file_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Error getting file info: {e}")
            file_size = 0
            file_hash = "unknown"
            
        # MIME type
        try:
            mime_type, _ = mimetypes.guess_type(str(file_path))
            mime_type = mime_type or "unknown"
        except Exception:
            mime_type = "unknown"
            
        # Content analysis
        content_checks = self.analyzer.analyze_content(file_path)
        
        # Combine all checks
        all_checks = []
        if not is_valid:
            all_checks.append(SecurityCheck(
                passed=False,
                threat_level='high',
                details=f"File validation failed: {', '.join(validation_errors)}",
                recommendations=["Fix validation issues", "Review file properties"]
            ))
        all_checks.extend(content_checks)
        
        # Determine overall threat level
        threat_levels = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
        max_threat = max([threat_levels[check.threat_level] for check in all_checks], default=0)
        overall_threat = [k for k, v in threat_levels.items() if v == max_threat][0]
        
        # Determine if file is safe
        is_safe = all_checks and all(check.passed for check in all_checks)
        
        # Collect warnings
        warnings = []
        if file_size > self.config.security.max_file_size_mb * 1024 * 1024 * 0.8:  # 80% of max
            warnings.append("File size approaching maximum limit")
        if not is_safe:
            warnings.append("File failed security checks")
            
        return FileSecurityProfile(
            file_path=file_path,
            file_hash=file_hash,
            mime_type=mime_type,
            file_size=file_size,
            security_checks=all_checks,
            overall_threat_level=overall_threat,
            is_safe=is_safe,
            warnings=warnings
        )
        
    def is_file_safe_for_processing(self, file_path: Path) -> bool:
        """
        Quick check if a file is safe for processing.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if file is safe, False otherwise
        """
        profile = self.assess_file_security(file_path)
        return profile.is_safe
        
    def get_security_summary(self, files: List[Path]) -> Dict[str, any]:
        """
        Get security summary for multiple files.
        
        Args:
            files: List of file paths to assess
            
        Returns:
            Security summary statistics
        """
        profiles = [self.assess_file_security(f) for f in files]
        
        total_files = len(profiles)
        safe_files = sum(1 for p in profiles if p.is_safe)
        threat_levels = [p.overall_threat_level for p in profiles]
        
        return {
            'total_files': total_files,
            'safe_files': safe_files,
            'unsafe_files': total_files - safe_files,
            'threat_level_distribution': {
                'low': threat_levels.count('low'),
                'medium': threat_levels.count('medium'),
                'high': threat_levels.count('high'),
                'critical': threat_levels.count('critical')
            },
            'safety_rate': safe_files / total_files if total_files > 0 else 0
        }
