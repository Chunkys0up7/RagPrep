"""
Security Module for RAG Document Processing Utility

This module provides comprehensive file validation, sanitization, and content analysis
for security purposes.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import hashlib
import logging
import mimetypes
import re
from pathlib import Path

# Try to import magic, but provide fallback for Windows
try:
    import magic

    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
    logging.warning("python-magic not available, using fallback MIME type detection")

from .config import Config


@dataclass
class SecurityCheck:
    """Result of a security check."""

    check_name: str
    passed: bool
    details: str
    threat_level: str = "low"


@dataclass
class FileSecurityProfile:
    """Security profile for a file."""

    file_path: Path
    is_safe: bool
    overall_threat_level: str
    file_size: int
    mime_type: str
    file_extension: str
    security_checks: List[SecurityCheck]
    warnings: List[str]
    recommendations: List[str]


class FileValidator:
    """Validates files for security and safety."""

    def __init__(self, config: Config):
        """Initialize file validator."""
        self.config = config
        self.logger = logging.getLogger(__name__)

    def validate_file(self, file_path: Path) -> List[SecurityCheck]:
        """Perform comprehensive file validation."""
        checks = []

        # Check file existence
        if not file_path.exists():
            checks.append(
                SecurityCheck(
                    check_name="File Existence",
                    passed=False,
                    details="File does not exist",
                    threat_level="high",
                )
            )
            return checks

        # Check file size
        try:
            file_size = file_path.stat().st_size
            max_size = self.config.security.max_file_size_mb * 1024 * 1024

            if file_size > max_size:
                checks.append(
                    SecurityCheck(
                        check_name="File Size",
                        passed=False,
                        details=f"File size {file_size} exceeds limit {max_size}",
                        threat_level="medium",
                    )
                )
            else:
                checks.append(
                    SecurityCheck(
                        check_name="File Size",
                        passed=True,
                        details=f"File size {file_size} within limits",
                        threat_level="low",
                    )
                )
        except Exception as e:
            checks.append(
                SecurityCheck(
                    check_name="File Size",
                    passed=False,
                    details=f"Could not check file size: {e}",
                    threat_level="medium",
                )
            )

        # Check file extension
        file_ext = file_path.suffix.lower()
        if file_ext not in self.config.security.allowed_file_extensions:
            checks.append(
                SecurityCheck(
                    check_name="File Extension",
                    passed=False,
                    details=f"Extension {file_ext} not allowed",
                    threat_level="high",
                )
            )
        else:
            checks.append(
                SecurityCheck(
                    check_name="File Extension",
                    passed=True,
                    details=f"Extension {file_ext} is allowed",
                    threat_level="low",
                )
            )

        # Check MIME type
        try:
            if MAGIC_AVAILABLE:
                # Use python-magic for accurate MIME detection
                mime_type = magic.from_file(str(file_path), mime=True)
            else:
                # Fallback to mimetypes module
                mime_type, _ = mimetypes.guess_type(str(file_path))
                if mime_type is None:
                    mime_type = "application/octet-stream"

            if mime_type not in self.config.security.allowed_mime_types:
                checks.append(
                    SecurityCheck(
                        check_name="MIME Type",
                        passed=False,
                        details=f"MIME type {mime_type} not allowed",
                        threat_level="high",
                    )
                )
            else:
                checks.append(
                    SecurityCheck(
                        check_name="MIME Type",
                        passed=True,
                        details=f"MIME type {mime_type} is allowed",
                        threat_level="low",
                    )
                )
        except Exception as e:
            checks.append(
                SecurityCheck(
                    check_name="MIME Type",
                    passed=False,
                    details=f"Could not determine MIME type: {e}",
                    threat_level="medium",
                )
            )

        return checks


class FileSanitizer:
    """Sanitizes filenames and paths for security."""

    def __init__(self, config: Config):
        """Initialize file sanitizer."""
        self.config = config
        self.logger = logging.getLogger(__name__)

    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename by removing dangerous characters."""
        # Remove or replace dangerous characters
        dangerous_chars = ["<", ">", ":", '"', "|", "?", "*", "\\", "/"]
        sanitized = filename

        for char in dangerous_chars:
            sanitized = sanitized.replace(char, "_")

        # Remove control characters
        sanitized = "".join(char for char in sanitized if ord(char) >= 32)

        # Limit length
        if len(sanitized) > 255:
            name, ext = (
                sanitized.rsplit(".", 1) if "." in sanitized else (sanitized, "")
            )
            sanitized = name[: 255 - len(ext) - 1] + ("." + ext if ext else "")

        return sanitized

    def sanitize_path(self, file_path: Path) -> Path:
        """Sanitize file path for security."""
        # Resolve to absolute path
        try:
            resolved_path = file_path.resolve()
        except Exception:
            resolved_path = file_path.absolute()

        # Check if path is within allowed directories
        allowed_dirs = [
            Path.cwd(),
            Path.cwd() / "documents",
            Path.cwd() / "output",
            Path.cwd() / "temp",
        ]

        for allowed_dir in allowed_dirs:
            try:
                if resolved_path.is_relative_to(allowed_dir):
                    return resolved_path
            except ValueError:
                continue

        # If not in allowed directories, restrict to current working directory
        self.logger.warning(
            f"Path {file_path} not in allowed directories, restricting to CWD"
        )
        return Path.cwd() / file_path.name


class ContentAnalyzer:
    """Analyzes file content for security threats."""

    def __init__(self, config: Config):
        """Initialize content analyzer."""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Patterns for detecting potentially dangerous content
        self.dangerous_patterns = {
            "executable": [
                rb"MZ",  # Windows executable
                rb"\x7fELF",  # Linux executable
                rb"\xfe\xed\xfa\xce",  # macOS executable
                rb"#!/",  # Shebang
            ],
            "script": [
                rb"<script",  # HTML script tags
                rb"javascript:",  # JavaScript protocol
                rb"vbscript:",  # VBScript protocol
                rb"<?php",  # PHP opening tag
                rb"<?=",  # PHP short echo tag
                rb"<?",  # PHP short tag
                rb"<%",  # ASP tags
                rb"<%=",  # ASP echo tags
            ],
            "macro": [
                rb"VBA",  # VBA macros
                rb"Sub ",  # VBA subroutines
                rb"Function ",  # VBA functions
            ],
        }

    def analyze_content(self, file_path: Path) -> List[SecurityCheck]:
        """Analyze file content for security threats."""
        checks = []

        try:
            # Read first few KB for pattern matching
            with open(file_path, "rb") as f:
                content = f.read(8192)  # Read first 8KB

            for threat_type, patterns in self.dangerous_patterns.items():
                for pattern in patterns:
                    if pattern in content:
                        checks.append(
                            SecurityCheck(
                                check_name=f"Content Analysis - {threat_type.title()}",
                                passed=False,
                                details=f"Detected {threat_type} pattern in content",
                                threat_level="high",
                            )
                        )
                        break
                else:
                    checks.append(
                        SecurityCheck(
                            check_name=f"Content Analysis - {threat_type.title()}",
                            passed=True,
                            details=f"No {threat_type} patterns detected",
                            threat_level="low",
                        )
                    )

            # Check for suspicious file headers
            if content.startswith(b"PK"):
                # ZIP file - check for executable content
                checks.append(
                    SecurityCheck(
                        check_name="Archive Analysis",
                        passed=True,
                        details="ZIP archive detected, contents should be validated",
                        threat_level="medium",
                    )
                )

        except Exception as e:
            checks.append(
                SecurityCheck(
                    check_name="Content Analysis",
                    passed=False,
                    details=f"Could not analyze content: {e}",
                    threat_level="medium",
                )
            )

        return checks


class SecurityManager:
    """Main security manager that coordinates all security operations."""

    def __init__(self, config: Config):
        """Initialize security manager."""
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.validator = FileValidator(config)
        self.sanitizer = FileSanitizer(config)
        self.analyzer = ContentAnalyzer(config)

    def assess_file_security(self, file_path: Path) -> FileSecurityProfile:
        """Perform comprehensive security assessment of a file."""
        self.logger.info(f"Assessing security of file: {file_path}")

        # Perform all security checks
        validation_checks = self.validator.validate_file(file_path)
        content_checks = self.analyzer.analyze_content(file_path)

        all_checks = validation_checks + content_checks

        # Determine overall threat level
        threat_levels = {"low": 0, "medium": 0, "high": 0}
        for check in all_checks:
            threat_levels[check.threat_level] += 1

        if threat_levels["high"] > 0:
            overall_threat = "high"
        elif threat_levels["medium"] > 0:
            overall_threat = "medium"
        else:
            overall_threat = "low"

        # Determine if file is safe
        is_safe = all(check.passed for check in all_checks)

        # Get file metadata
        try:
            file_size = file_path.stat().st_size
            file_ext = file_path.suffix.lower()

            if MAGIC_AVAILABLE:
                mime_type = magic.from_file(str(file_path), mime=True)
            else:
                mime_type, _ = mimetypes.guess_type(str(file_path))
                if mime_type is None:
                    mime_type = "application/octet-stream"
        except Exception as e:
            self.logger.error(f"Error getting file metadata: {e}")
            file_size = 0
            mime_type = "unknown"
            file_ext = ""

        # Generate warnings and recommendations
        warnings = []
        recommendations = []

        for check in all_checks:
            if not check.passed:
                warnings.append(f"{check.check_name}: {check.details}")

                if check.threat_level == "high":
                    recommendations.append("File should not be processed")
                elif check.threat_level == "medium":
                    recommendations.append("File should be reviewed before processing")

        if not warnings:
            recommendations.append("File is safe for processing")

        return FileSecurityProfile(
            file_path=file_path,
            is_safe=is_safe,
            overall_threat_level=overall_threat,
            file_size=file_size,
            mime_type=mime_type,
            file_extension=file_ext,
            security_checks=all_checks,
            warnings=warnings,
            recommendations=recommendations,
        )

    def is_file_safe_for_processing(self, file_path: Path) -> bool:
        """Quick check if file is safe for processing."""
        try:
            profile = self.assess_file_security(file_path)
            return profile.is_safe
        except Exception as e:
            self.logger.error(f"Error checking file safety: {e}")
            return False

    def sanitize_file_path(self, file_path: Path) -> Path:
        """Sanitize file path for safe processing."""
        return self.sanitizer.sanitize_path(file_path)


def get_security_manager(config: Optional[Config] = None) -> SecurityManager:
    """Factory function to create a SecurityManager instance."""
    if config is None:
        from .config import Config
        config = Config()
    return SecurityManager(config)
