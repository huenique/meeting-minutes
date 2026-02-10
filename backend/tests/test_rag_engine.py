"""Tests for the RAG engine module."""

import os
import sys
import tempfile

import pytest

# Add the app directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from rag_engine import _build_rag_prompt, RAGAnswer, RAGContext, QuestionWithAnswer, read_files_as_context
from question_detector import DetectedQuestion


class TestBuildRAGPrompt:
    """Tests for the RAG prompt builder."""

    def test_prompt_with_all_context(self):
        prompt = _build_rag_prompt(
            question="What is the deadline?",
            meeting_context="We discussed the Q2 roadmap.",
            knowledge_context="[Source 1: roadmap.md]\nQ2 deadline is June 30.",
        )
        assert "What is the deadline?" in prompt
        assert "Q2 roadmap" in prompt
        assert "June 30" in prompt
        assert "Meeting Context" in prompt
        assert "Knowledge Base" in prompt

    def test_prompt_without_meeting_context(self):
        prompt = _build_rag_prompt(
            question="What is the deadline?",
            meeting_context="",
            knowledge_context="[Source 1: roadmap.md]\nQ2 deadline is June 30.",
        )
        assert "What is the deadline?" in prompt
        assert "Knowledge Base" in prompt
        assert "Meeting Context" not in prompt

    def test_prompt_without_knowledge_context(self):
        prompt = _build_rag_prompt(
            question="What is the deadline?",
            meeting_context="We discussed the Q2 roadmap.",
            knowledge_context="",
        )
        assert "What is the deadline?" in prompt
        assert "Meeting Context" in prompt
        assert "Knowledge Base" not in prompt

    def test_prompt_with_no_context(self):
        prompt = _build_rag_prompt(
            question="What is the deadline?",
            meeting_context="",
            knowledge_context="",
        )
        assert "What is the deadline?" in prompt
        assert "meeting assistant" in prompt.lower()

    def test_prompt_with_file_context(self):
        prompt = _build_rag_prompt(
            question="What is the budget?",
            meeting_context="",
            knowledge_context="",
            file_context="[File 1: budget.md]\nThe Q2 budget is $50,000.",
        )
        assert "What is the budget?" in prompt
        assert "Attached File Context" in prompt
        assert "$50,000" in prompt
        assert "Knowledge Base" not in prompt

    def test_prompt_with_both_knowledge_and_file_context(self):
        prompt = _build_rag_prompt(
            question="What is the plan?",
            meeting_context="We discussed Q2.",
            knowledge_context="[Source 1: plan.md]\nDeliver feature X.",
            file_context="[File 1: roadmap.md]\nQ2 priorities.",
        )
        assert "Knowledge Base" in prompt
        assert "Attached File Context" in prompt
        assert "Meeting Context" in prompt

    def test_prompt_without_file_context(self):
        prompt = _build_rag_prompt(
            question="What is the deadline?",
            meeting_context="context",
            knowledge_context="knowledge",
            file_context="",
        )
        assert "Attached File Context" not in prompt


class TestReadFilesAsContext:
    """Tests for the read_files_as_context function."""

    def test_read_text_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("Hello from context file!")
            f.flush()
            results = read_files_as_context([f.name])
            assert len(results) == 1
            assert results[0]["text"] == "Hello from context file!"
            assert results[0]["filename"] == os.path.basename(f.name)
        os.unlink(f.name)

    def test_read_markdown_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as f:
            f.write("# Context\n\nThis is context.")
            f.flush()
            results = read_files_as_context([f.name])
            assert len(results) == 1
            assert "Context" in results[0]["text"]
        os.unlink(f.name)

    def test_skip_nonexistent_file(self):
        results = read_files_as_context(["/nonexistent/file.txt"])
        assert len(results) == 0

    def test_skip_unsupported_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xyz", delete=False
        ) as f:
            f.write("test")
            f.flush()
            results = read_files_as_context([f.name])
            assert len(results) == 0
        os.unlink(f.name)

    def test_read_multiple_files(self):
        files = []
        for i in range(3):
            f = tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            )
            f.write(f"Content {i}")
            f.flush()
            f.close()
            files.append(f.name)

        results = read_files_as_context(files)
        assert len(results) == 3
        for i, r in enumerate(results):
            assert f"Content {i}" in r["text"]

        for fp in files:
            os.unlink(fp)

    def test_empty_file_list(self):
        results = read_files_as_context([])
        assert results == []

    def test_skip_empty_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("")
            f.flush()
            results = read_files_as_context([f.name])
            assert len(results) == 0
        os.unlink(f.name)


class TestRAGModels:
    """Tests for RAG Pydantic models."""

    def test_rag_answer_creation(self):
        answer = RAGAnswer(
            question="What is the plan?",
            answer="The plan is to deliver by Friday.",
            sources=[{"filename": "plan.md", "text_preview": "...", "distance": 0.1}],
            confidence=0.8,
        )
        assert answer.question == "What is the plan?"
        assert answer.confidence == 0.8
        assert len(answer.sources) == 1

    def test_rag_context_creation(self):
        context = RAGContext(
            question="test?",
            meeting_context="meeting text",
            retrieved_chunks=[],
        )
        assert context.question == "test?"

    def test_question_with_answer(self):
        q = DetectedQuestion(
            text="What is the plan?",
            start_index=0,
            end_index=17,
            confidence=0.9,
        )
        qwa = QuestionWithAnswer(question=q, answer=None)
        assert qwa.question.text == "What is the plan?"
        assert qwa.answer is None
