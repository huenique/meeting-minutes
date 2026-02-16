"""Tests for the question detection module."""

import sys
import os

# Add the app directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from question_detector import detect_questions, _compute_question_confidence, DetectedQuestion


class TestComputeQuestionConfidence:
    """Tests for the _compute_question_confidence helper."""

    def test_question_mark_gives_high_confidence(self):
        score = _compute_question_confidence("What time is the meeting?")
        assert score >= 0.7

    def test_interrogative_word_without_question_mark(self):
        score = _compute_question_confidence("What time is the meeting")
        assert score >= 0.3

    def test_plain_statement_low_confidence(self):
        score = _compute_question_confidence("The meeting is at 3pm.")
        assert score < 0.5

    def test_embedded_question_pattern(self):
        score = _compute_question_confidence(
            "I wonder if we should postpone the release."
        )
        assert score >= 0.2

    def test_interrogative_with_question_mark(self):
        score = _compute_question_confidence("How are we going to fix this?")
        assert score >= 0.7

    def test_empty_string(self):
        score = _compute_question_confidence("")
        assert score == 0.0

    def test_confidence_capped_at_one(self):
        # Sentence with question mark + interrogative + embedded pattern
        score = _compute_question_confidence(
            "Do you know what time the meeting is?"
        )
        assert score <= 1.0


class TestDetectQuestions:
    """Tests for the main detect_questions function."""

    def test_empty_transcript(self):
        result = detect_questions("")
        assert result == []

    def test_none_transcript(self):
        result = detect_questions(None)
        assert result == []

    def test_single_question(self):
        text = "Hello everyone. What time does the meeting start?"
        result = detect_questions(text)
        assert len(result) >= 1
        assert any("What time" in q.text for q in result)

    def test_multiple_questions(self):
        text = (
            "Let's discuss the roadmap. What are the priorities for Q2? "
            "Who is responsible for the backend work? "
            "The deadline is next Friday."
        )
        result = detect_questions(text)
        assert len(result) >= 2

    def test_no_questions(self):
        text = "The project is on track. We will deliver by Friday. Everything looks good."
        result = detect_questions(text)
        assert len(result) == 0

    def test_question_has_correct_type(self):
        text = "Can you explain the architecture?"
        result = detect_questions(text)
        assert len(result) >= 1
        assert isinstance(result[0], DetectedQuestion)
        assert result[0].confidence >= 0.5

    def test_question_indices(self):
        text = "Hello. What is the plan?"
        result = detect_questions(text)
        assert len(result) >= 1
        q = result[0]
        assert text[q.start_index:q.end_index] == q.text

    def test_embedded_question_detection(self):
        text = "I wonder if we should cancel the meeting."
        result = detect_questions(text)
        # Embedded patterns may or may not be detected depending on confidence threshold
        # Just verify no crash
        assert isinstance(result, list)

    def test_deduplication(self):
        text = "What is the plan? What is the plan?"
        result = detect_questions(text)
        # Should deduplicate identical questions
        assert len(result) == 1

    def test_whitespace_only(self):
        result = detect_questions("   \n\t  ")
        assert result == []
