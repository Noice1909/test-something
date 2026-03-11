"""Tests for sensitive data masking in logging."""

from __future__ import annotations

from src.logging_config import _mask_sensitive


class TestMaskSensitive:

    def test_masks_openai_api_key(self):
        text = "Using key sk-abc12345678901234567890"
        assert "sk-abc" not in _mask_sensitive(text)
        assert "***REDACTED***" in _mask_sensitive(text)

    def test_masks_password_field(self):
        text = 'password=super_secret_123'
        result = _mask_sensitive(text)
        assert "super_secret_123" not in result

    def test_masks_uri_credentials(self):
        text = "Connecting to neo4j://admin:s3cret@localhost:7687"
        result = _mask_sensitive(text)
        assert "s3cret" not in result

    def test_masks_bearer_token(self):
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig"
        result = _mask_sensitive(text)
        assert "eyJhbG" not in result

    def test_no_false_positive_on_normal_text(self):
        text = "Query returned 42 rows for Application nodes"
        assert _mask_sensitive(text) == text

    def test_masks_json_password(self):
        text = '{"password": "my_secret", "user": "admin"}'
        result = _mask_sensitive(text)
        assert "my_secret" not in result
