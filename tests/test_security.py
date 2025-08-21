"""
Unit tests for the security module.

Tests file validation, sanitization, content analysis, and security management.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import Config
from security import (ContentAnalyzer, FileSanitizer, FileSecurityProfile,
                      FileValidator, SecurityCheck, SecurityManager)


class TestSecurityCheck:
    """Test SecurityCheck dataclass."""

    def test_security_check_creation(self):
        """Test creating a SecurityCheck instance."""
        check = SecurityCheck(
            passed=True,
            threat_level="low",
            details="Test check",
            recommendations=["Test recommendation"],
        )

        assert check.passed is True
        assert check.threat_level == "low"
        assert check.details == "Test check"
        assert check.recommendations == ["Test recommendation"]


class TestFileSecurityProfile:
    """Test FileSecurityProfile dataclass."""

    def test_file_security_profile_creation(self):
        """Test creating a FileSecurityProfile instance."""
        profile = FileSecurityProfile(
            file_path=Path("test.pdf"),
            file_hash="abc123",
            mime_type="application/pdf",
            file_size=1024,
            security_checks=[],
            overall_threat_level="low",
            is_safe=True,
            warnings=[],
        )

        assert profile.file_path == Path("test.pdf")
        assert profile.file_hash == "abc123"
        assert profile.mime_type == "application/pdf"
        assert profile.file_size == 1024
        assert profile.overall_threat_level == "low"
        assert profile.is_safe is True


class TestFileValidator:
    """Test FileValidator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = Mock()
        self.config.security.max_file_size_mb = 100
        self.config.security.allowed_file_extensions = {".pdf", ".txt"}
        self.config.security.allowed_mime_types = {"application/pdf", "text/plain"}

        self.validator = FileValidator(self.config)

    def test_validate_file_exists(self):
        """Test file existence validation."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            temp_path = Path(f.name)

        try:
            is_valid, errors = self.validator.validate_file(temp_path)
            assert is_valid is True
            assert len(errors) == 0
        finally:
            os.unlink(temp_path)

    def test_validate_file_not_exists(self):
        """Test validation of non-existent file."""
        temp_path = Path("/nonexistent/file.pdf")
        is_valid, errors = self.validator.validate_file(temp_path)

        assert is_valid is False
        assert len(errors) == 1
        assert "does not exist" in errors[0]

    def test_validate_file_size(self):
        """Test file size validation."""
        # Create a large file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"x" * (101 * 1024 * 1024))  # 101MB
            temp_path = Path(f.name)

        try:
            is_valid, errors = self.validator.validate_file(temp_path)
            assert is_valid is False
            assert len(errors) == 1
            assert "exceeds maximum allowed size" in errors[0]
        finally:
            os.unlink(temp_path)

    def test_validate_file_extension(self):
        """Test file extension validation."""
        with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as f:
            temp_path = Path(f.name)

        try:
            is_valid, errors = self.validator.validate_file(temp_path)
            assert is_valid is False
            assert len(errors) == 1
            assert "is not allowed" in errors[0]
        finally:
            os.unlink(temp_path)

    def test_validate_file_mime_type(self):
        """Test MIME type validation."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Mock mimetypes.guess_type to return an invalid MIME type
            with patch(
                "security.mimetypes.guess_type", return_value=("application/exe", None)
            ):
                is_valid, errors = self.validator.validate_file(temp_path)
                assert is_valid is False
                assert len(errors) == 1
                assert "is not allowed" in errors[0]
        finally:
            os.unlink(temp_path)


class TestFileSanitizer:
    """Test FileSanitizer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = Mock()
        self.config.security.max_filename_length = 255
        self.sanitizer = FileSanitizer(self.config)

    def test_sanitize_filename_dangerous_chars(self):
        """Test sanitization of dangerous characters."""
        dangerous_filename = "file<name>with*chars?|"
        sanitized = self.sanitizer.sanitize_filename(dangerous_filename)

        assert "<" not in sanitized
        assert ">" not in sanitized
        assert "*" not in sanitized
        assert "?" not in sanitized
        assert "|" not in sanitized
        assert "_" in sanitized

    def test_sanitize_filename_path_separators(self):
        """Test sanitization of path separators."""
        filename_with_path = "path/to/file.pdf"
        sanitized = self.sanitizer.sanitize_filename(filename_with_path)

        assert "/" not in sanitized
        assert "\\" not in sanitized
        assert "_" in sanitized

    def test_sanitize_filename_length_limit(self):
        """Test filename length limiting."""
        long_filename = "a" * 300 + ".pdf"
        sanitized = self.sanitizer.sanitize_filename(long_filename)

        assert len(sanitized) <= 255
        assert sanitized.endswith(".pdf")

    def test_sanitize_filename_empty_result(self):
        """Test handling of filename that becomes empty after sanitization."""
        dangerous_filename = "///"
        sanitized = self.sanitizer.sanitize_filename(dangerous_filename)

        assert sanitized == "unnamed_file"

    def test_create_safe_output_path(self):
        """Test creation of safe output path."""
        input_path = Path("input/file<name>.pdf")
        output_dir = Path("output")

        safe_path = self.sanitizer.create_safe_output_path(input_path, output_dir)

        assert safe_path.parent == output_dir
        assert "<" not in safe_path.name
        assert ">" not in safe_path.name
        assert safe_path.name.endswith(".pdf")


class TestContentAnalyzer:
    """Test ContentAnalyzer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = Mock()
        self.analyzer = ContentAnalyzer(self.config)

    def test_analyze_content_executable(self):
        """Test detection of executable content."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"MZ" + b"x" * 100)  # MZ header
            temp_path = Path(f.name)

        try:
            checks = self.analyzer.analyze_content(temp_path)

            assert len(checks) == 1
            assert checks[0].passed is False
            assert checks[0].threat_level == "high"
            assert "executable content" in checks[0].details.lower()
        finally:
            os.unlink(temp_path)

    def test_analyze_content_script(self):
        """Test detection of script content."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"<?php echo 'hello'; ?>")
            temp_path = Path(f.name)

        try:
            checks = self.analyzer.analyze_content(temp_path)

            assert len(checks) == 1
            assert checks[0].passed is False
            assert checks[0].threat_level == "medium"
            assert "script-like content" in checks[0].details.lower()
        finally:
            os.unlink(temp_path)

    def test_analyze_content_macro(self):
        """Test detection of macro content."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"Sub TestMacro()\nEnd Sub")
            temp_path = Path(f.name)

        try:
            checks = self.analyzer.analyze_content(temp_path)

            assert len(checks) == 1
            assert checks[0].passed is False
            assert checks[0].threat_level == "medium"
            assert "macro-like content" in checks[0].details.lower()
        finally:
            os.unlink(temp_path)

    def test_analyze_content_safe(self):
        """Test analysis of safe content."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"This is a safe text document.")
            temp_path = Path(f.name)

        try:
            checks = self.analyzer.analyze_content(temp_path)

            # Should pass all checks
            assert all(check.passed for check in checks)
        finally:
            os.unlink(temp_path)

    def test_analyze_content_read_error(self):
        """Test handling of file read errors."""
        # Mock open to raise an exception
        with patch("builtins.open", side_effect=PermissionError("Access denied")):
            temp_path = Path("nonexistent.txt")
            checks = self.analyzer.analyze_content(temp_path)

            assert len(checks) == 1
            assert checks[0].passed is False
            assert checks[0].threat_level == "low"
            assert "could not analyze" in checks[0].details.lower()


class TestSecurityManager:
    """Test SecurityManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = Mock()
        self.config.security.max_file_size_mb = 100
        self.config.security.allowed_file_extensions = {".pdf", ".txt"}
        self.config.security.allowed_mime_types = {"application/pdf", "text/plain"}

        self.security_manager = SecurityManager(self.config)

    def test_assess_file_security_safe(self):
        """Test security assessment of a safe file."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"This is a safe text file.")
            temp_path = Path(f.name)

        try:
            profile = self.security_manager.assess_file_security(temp_path)

            assert profile.file_path == temp_path
            assert profile.is_safe is True
            assert profile.overall_threat_level == "low"
            assert len(profile.warnings) == 0
        finally:
            os.unlink(temp_path)

    def test_assess_file_security_unsafe(self):
        """Test security assessment of an unsafe file."""
        with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as f:
            f.write(b"MZ" + b"x" * 100)  # Executable content
            temp_path = Path(f.name)

        try:
            profile = self.security_manager.assess_file_security(temp_path)

            assert profile.is_safe is False
            assert profile.overall_threat_level in ["high", "critical"]
            assert len(profile.warnings) > 0
        finally:
            os.unlink(temp_path)

    def test_is_file_safe_for_processing(self):
        """Test quick safety check."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"Safe content")
            temp_path = Path(f.name)

        try:
            is_safe = self.security_manager.is_file_safe_for_processing(temp_path)
            assert is_safe is True
        finally:
            os.unlink(temp_path)

    def test_get_security_summary(self):
        """Test security summary for multiple files."""
        # Create test files
        files = []
        try:
            # Safe file
            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
                f.write(b"Safe content")
                files.append(Path(f.name))

            # Unsafe file
            with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as f:
                f.write(b"MZ" + b"x" * 100)
                files.append(Path(f.name))

            summary = self.security_manager.get_security_summary(files)

            assert summary["total_files"] == 2
            assert summary["safe_files"] == 1
            assert summary["unsafe_files"] == 1
            assert summary["safety_rate"] == 0.5
            assert "threat_level_distribution" in summary

        finally:
            for file_path in files:
                if file_path.exists():
                    os.unlink(file_path)

    def test_assess_file_security_with_validation_failure(self):
        """Test security assessment when file validation fails."""
        # Create a file with disallowed extension
        with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as f:
            f.write(b"Safe content but wrong extension")
            temp_path = Path(f.name)

        try:
            profile = self.security_manager.assess_file_security(temp_path)

            assert profile.is_safe is False
            assert profile.overall_threat_level == "high"
            assert any(
                "validation failed" in check.details
                for check in profile.security_checks
            )
        finally:
            os.unlink(temp_path)


class TestSecurityIntegration:
    """Test integration between security components."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = Mock()
        self.config.security.max_file_size_mb = 100
        self.config.security.allowed_file_extensions = {".pdf", ".txt"}
        self.config.security.allowed_mime_types = {"application/pdf", "text/plain"}
        self.config.security.max_filename_length = 255

        self.security_manager = SecurityManager(self.config)

    def test_end_to_end_security_workflow(self):
        """Test complete security workflow from validation to assessment."""
        # Create a file with suspicious content
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"<?php system('rm -rf /'); ?>")  # Dangerous script
            temp_path = Path(f.name)

        try:
            # Test validation
            validator = self.security_manager.validator
            is_valid, errors = validator.validate_file(temp_path)
            assert is_valid is True  # Extension and size are valid

            # Test sanitization
            sanitizer = self.security_manager.sanitizer
            safe_name = sanitizer.sanitize_filename(temp_path.name)
            assert safe_name == temp_path.name  # Should be unchanged for safe name

            # Test content analysis
            analyzer = self.security_manager.analyzer
            checks = analyzer.analyze_content(temp_path)
            assert len(checks) == 1
            assert checks[0].threat_level == "medium"

            # Test full security assessment
            profile = self.security_manager.assess_file_security(temp_path)
            assert profile.is_safe is False
            assert profile.overall_threat_level == "medium"

        finally:
            os.unlink(temp_path)

    def test_security_manager_component_access(self):
        """Test that security manager properly initializes all components."""
        assert hasattr(self.security_manager, "validator")
        assert hasattr(self.security_manager, "sanitizer")
        assert hasattr(self.security_manager, "analyzer")

        assert isinstance(self.security_manager.validator, FileValidator)
        assert isinstance(self.security_manager.sanitizer, FileSanitizer)
        assert isinstance(self.security_manager.analyzer, ContentAnalyzer)
