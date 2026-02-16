"""
Question Detection Module for Meetily.

Detects questions in meeting transcripts using heuristic patterns and
optional LLM-based classification. Designed to work as a post-processing
step after transcription.
"""

import re
import logging
from typing import List
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class DetectedQuestion(BaseModel):
    """Represents a detected question from the transcript."""
    text: str
    start_index: int
    end_index: int
    confidence: float  # 0.0 to 1.0


# Interrogative words/phrases that commonly start questions
_INTERROGATIVE_STARTERS = re.compile(
    r"^\s*(?:who|what|when|where|why|how|which|whom|whose|"
    r"is|are|was|were|do|does|did|can|could|will|would|"
    r"shall|should|may|might|have|has|had|isn't|aren't|"
    r"wasn't|weren't|don't|doesn't|didn't|can't|couldn't|"
    r"won't|wouldn't|shouldn't|haven't|hasn't|hadn't)\b",
    re.IGNORECASE,
)

# Pattern for embedded question markers (e.g., "I wonder if...", "Can anyone tell me...")
_EMBEDDED_QUESTION_PATTERNS = re.compile(
    r"(?:I\s+wonder|do\s+you\s+know|can\s+(?:you|anyone|somebody)\s+(?:tell|explain|clarify)|"
    r"does\s+anyone\s+know|any\s+idea|any\s+thoughts\s+on|what\s+do\s+you\s+think)",
    re.IGNORECASE,
)


def detect_questions(transcript_text: str) -> List[DetectedQuestion]:
    """
    Detect questions in a transcript using heuristic analysis.

    Uses a combination of:
    - Question mark detection
    - Interrogative word/phrase matching
    - Embedded question pattern recognition

    Args:
        transcript_text: The transcript text to analyze.

    Returns:
        A list of DetectedQuestion objects.
    """
    if not transcript_text or not transcript_text.strip():
        return []

    questions: List[DetectedQuestion] = []
    seen_texts: set = set()

    # Split into sentences (handles ?, !, . and newlines)
    sentences = re.split(r'(?<=[.!?\n])\s+', transcript_text)

    current_pos = 0
    for sentence in sentences:
        sentence_stripped = sentence.strip()
        if not sentence_stripped:
            current_pos += len(sentence)
            continue

        start_index = transcript_text.find(sentence_stripped, current_pos)
        if start_index == -1:
            start_index = current_pos
        end_index = start_index + len(sentence_stripped)

        confidence = _compute_question_confidence(sentence_stripped)

        if confidence >= 0.5:
            normalized = sentence_stripped.lower()
            if normalized not in seen_texts:
                seen_texts.add(normalized)
                questions.append(
                    DetectedQuestion(
                        text=sentence_stripped,
                        start_index=start_index,
                        end_index=end_index,
                        confidence=confidence,
                    )
                )

        current_pos = end_index

    logger.info(f"Detected {len(questions)} questions in transcript of length {len(transcript_text)}")
    return questions


def _compute_question_confidence(sentence: str) -> float:
    """
    Compute the confidence score that a sentence is a question.

    Returns a float between 0.0 and 1.0.
    """
    score = 0.0

    # Strong signal: ends with a question mark
    if sentence.rstrip().endswith("?"):
        score += 0.7

    # Moderate signal: starts with an interrogative word
    if _INTERROGATIVE_STARTERS.match(sentence):
        score += 0.3

    # Weaker signal: contains embedded question patterns
    if _EMBEDDED_QUESTION_PATTERNS.search(sentence):
        score += 0.2

    # Cap at 1.0
    return min(score, 1.0)
