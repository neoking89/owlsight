import sys
import pytest

sys.path.append("src")
from owlsight.app.default_functions import is_url

def test_is_url():
    # Arrange
    test_cases = [
        # Valid URLs
        ("https://claude.ai/new", True),
        ("http://example.com", True),
        ("ftp://ftp.example.com/file.txt", True),
        ("https://sub.domain.example.com/path?query=123#fragment", True),
        ("http://localhost:8000", True),
        ("http://127.0.0.1", True),
        ("https://123.123.123.123", True),
        ("https://example.com:8080", True),
        ("https://example.com/path/to/page", True),

        # Invalid URLs
        ("www.google.nl", False),  # Missing protocol
        ("htp://missing-t.com", False),  # Invalid protocol
        ("http:/missing-slash.com", False),  # Malformed protocol
        ("www.google", False),  # Missing top-level domain
        ("https://", False),  # Incomplete URL
        ("ftp://", False),  # Incomplete URL with only protocol
        ("http://?", False),  # Missing domain
        ("//example.com", False),  # Missing protocol
        ("example", False),  # Not a URL
        ("https://.com", False),  # Missing domain name
        ("https://example..com", False),  # Invalid domain with double dot
    ]

    # Act & Assert
    for url, expected in test_cases:
        result = is_url(url)
        assert result == expected, f"Test failed for URL: {url}. Expected: {expected}, Got: {result}"

if __name__ == "__main__":
    pytest.main([__file__])