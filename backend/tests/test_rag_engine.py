"""Tests for the RAG engine module."""

import os
import sys
import tempfile

import pytest

# Add the app directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from rag_engine import _build_rag_prompt, RAGAnswer, RAGContext, QuestionWithAnswer
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
