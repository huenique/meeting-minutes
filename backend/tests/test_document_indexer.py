"""Tests for the document indexer module."""

import os
import sys
import tempfile

import pytest

# Add the app directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from document_indexer import (
    SUPPORTED_EXTENSIONS,
    DocumentMetadata,
    _compute_file_hash,
    _extract_text_from_file,
    _extract_text_from_html,
    chunk_text,
)


class TestChunkText:
    """Tests for the chunk_text function."""

    def test_empty_text(self):
        assert chunk_text("") == []

    def test_whitespace_only(self):
        assert chunk_text("   \n  ") == []

    def test_single_short_paragraph(self):
        text = "This is a short paragraph."
        result = chunk_text(text, chunk_size=500)
        assert len(result) == 1
        assert result[0] == text

    def test_multiple_paragraphs_fit_in_one_chunk(self):
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        result = chunk_text(text, chunk_size=500)
        assert len(result) == 1

    def test_paragraphs_split_across_chunks(self):
        text = "A" * 300 + "\n\n" + "B" * 300
        result = chunk_text(text, chunk_size=350, overlap=50)
        assert len(result) >= 2

    def test_chunk_size_validation(self):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            chunk_text("hello", chunk_size=0)

    def test_overlap_validation(self):
        with pytest.raises(ValueError, match="overlap must be non-negative"):
            chunk_text("hello", overlap=-1)

    def test_overlap_exceeds_chunk_size(self):
        # Should not raise, overlap gets adjusted
        result = chunk_text("Hello world. This is a test.", chunk_size=10, overlap=15)
        assert isinstance(result, list)

    def test_long_text_produces_multiple_chunks(self):
        text = " ".join(["word"] * 1000)
        result = chunk_text(text, chunk_size=100, overlap=10)
        assert len(result) > 1


class TestExtractTextFromFile:
    """Tests for file text extraction."""

    def test_extract_text_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("Hello, world!")
            f.flush()
            text = _extract_text_from_file(f.name)
            assert text == "Hello, world!"
        os.unlink(f.name)

    def test_extract_markdown_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as f:
            f.write("# Heading\n\nParagraph content.")
            f.flush()
            text = _extract_text_from_file(f.name)
            assert "Heading" in text
            assert "Paragraph" in text
        os.unlink(f.name)

    def test_unsupported_extension(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xyz", delete=False
        ) as f:
            f.write("test")
            f.flush()
            with pytest.raises(ValueError, match="Unsupported file type"):
                _extract_text_from_file(f.name)
        os.unlink(f.name)


class TestExtractTextFromHtml:
    """Tests for HTML text extraction."""

    def test_strips_html_tags(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False
        ) as f:
            f.write("<html><body><p>Hello <b>world</b></p></body></html>")
            f.flush()
            text = _extract_text_from_html(f.name)
            assert "Hello" in text
            assert "world" in text
            assert "<" not in text
        os.unlink(f.name)


class TestComputeFileHash:
    """Tests for file hash computation."""

    def test_consistent_hash(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("test content")
            f.flush()
            hash1 = _compute_file_hash(f.name)
            hash2 = _compute_file_hash(f.name)
            assert hash1 == hash2
        os.unlink(f.name)

    def test_different_content_different_hash(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f1:
            f1.write("content A")
            f1.flush()
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as f2:
                f2.write("content B")
                f2.flush()
                assert _compute_file_hash(f1.name) != _compute_file_hash(
                    f2.name
                )
        os.unlink(f1.name)
        os.unlink(f2.name)


class TestSupportedExtensions:
    """Tests for supported file extensions."""

    def test_common_extensions_supported(self):
        assert ".md" in SUPPORTED_EXTENSIONS
        assert ".txt" in SUPPORTED_EXTENSIONS
        assert ".pdf" in SUPPORTED_EXTENSIONS
        assert ".csv" in SUPPORTED_EXTENSIONS
        assert ".json" in SUPPORTED_EXTENSIONS

    def test_unsupported_extensions(self):
        assert ".exe" not in SUPPORTED_EXTENSIONS
        assert ".mp3" not in SUPPORTED_EXTENSIONS
        assert ".docx" not in SUPPORTED_EXTENSIONS
