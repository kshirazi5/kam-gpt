"""Logic for choosing the best experience-based response.

This module contains the keyword definitions for each experience that Kamran
likes to highlight as well as the helper utilities that score incoming
questions.  The selection logic is intentionally simple so it can run inside
Streamlit without depending on an LLM.
"""
from __future__ import annotations

import dataclasses
import re
from typing import List, Sequence

_WORD_RE = re.compile(r"\b\w+\b")


@dataclasses.dataclass(frozen=True)
class Experience:
    """Representation of a portfolio story that can be surfaced to a user."""

    name: str
    keywords: Sequence[str]
    answer: str
    follow_up_questions: Sequence[str]


def _tokenize(text: str) -> List[str]:
    """Tokenize *text* into lowercase word tokens.

    The helper makes the behaviour explicit and testable instead of relying on
    Python's ``split`` which struggles with punctuation.  ``_select_experience``
    and ``generate_engineer_response`` both depend on the function so questions
    are scored with whole-word matching.
    """

    return [token.lower() for token in _WORD_RE.findall(text)]


def _phrase_in_tokens(tokens: Sequence[str], phrase: str) -> bool:
    """Return ``True`` when *phrase* (a word or multi-word phrase) is present.

    Multi-word phrases are matched by requiring contiguous tokens, making it
    possible to distinguish phrases like ``"return on investment"`` from the
    word ``"return"`` on its own.
    """

    phrase_tokens = _tokenize(phrase)
    if not phrase_tokens:
        return False

    if len(phrase_tokens) == 1:
        return phrase_tokens[0] in tokens

    window = len(phrase_tokens)
    for index in range(len(tokens) - window + 1):
        if list(tokens[index : index + window]) == phrase_tokens:
            return True
    return False


EXPERIENCES: Sequence[Experience] = (
    Experience(
        name="data_quality",
        keywords=(
            "data quality",
            "reliability",
            "android",
            "experiments",
            "slo",
            "metrics",
            "crash",
        ),
        answer=(
            "On the Android data-quality team I rebuilt the reliability pipeline so "
            "that experimentation data could be trusted. We introduced better "
            "validation, created SLO dashboards, and reduced crash loops by 37% "
            "without blocking feature velocity."
        ),
        follow_up_questions=(
            "How did you measure the impact on reliability?",
            "What tooling supported the data-quality checks?",
        ),
    ),
    Experience(
        name="roi_story",
        keywords=(
            "roi",
            "return",
            "investment",
            "executive",
            "business",
            "sales",
        ),
        answer=(
            "I led a go-to-market analytics project that connected product usage "
            "to revenue outcomes. By pairing cohort data with sales motion we "
            "outlined a clear ROI narrative for executives deciding between "
            "competing investments."
        ),
        follow_up_questions=(
            "Which metrics resonated most with leadership?",
            "How did the work influence roadmap priorities?",
        ),
    ),
    Experience(
        name="ml_infra",
        keywords=("machine", "learning", "pipeline", "ml", "inference"),
        answer=(
            "I also operated a machine learning pipeline that supported near-real "
            "time inference. That included tightening feedback loops between data "
            "labelling, training, and deployment so product engineers could ship "
            "personalised features confidently."
        ),
        follow_up_questions=(
            "What guardrails kept the models stable?",
            "How did you collaborate with product teams?",
        ),
    ),
)

DEFAULT_EXPERIENCE = EXPERIENCES[0]


def _select_experience(question: str) -> Experience:
    """Return the best matching :class:`Experience` for *question*.

    The scoring logic uses token comparisons rather than raw substring checks to
    prevent words like ``"roi"`` from being accidentally matched inside
    ``"android"``.  The experience with the highest keyword hit count is chosen;
    when no keywords match we fall back to :data:`DEFAULT_EXPERIENCE`.
    """

    tokens = _tokenize(question)
    best = DEFAULT_EXPERIENCE
    best_score = 0

    for experience in EXPERIENCES:
        score = sum(1 for keyword in experience.keywords if _phrase_in_tokens(tokens, keyword))
        if score > best_score:
            best = experience
            best_score = score

    return best


def generate_engineer_response(question: str) -> dict:
    """Generate the structured response for a portfolio question.

    The function exposes the matched experience name so downstream components can
    tailor styling or analytics without having to re-run the selection logic.
    """

    experience = _select_experience(question)
    return {
        "experience": experience.name,
        "answer": experience.answer,
        "follow_up_questions": list(experience.follow_up_questions),
    }


__all__ = [
    "Experience",
    "EXPERIENCES",
    "DEFAULT_EXPERIENCE",
    "_select_experience",
    "generate_engineer_response",
]
