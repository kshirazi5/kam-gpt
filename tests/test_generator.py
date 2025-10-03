import pytest

from kam_gpt.generator import (
    DEFAULT_EXPERIENCE,
    _phrase_in_tokens,
    _select_experience,
    _tokenize,
    generate_engineer_response,
)


def test_android_reliability_selects_data_quality():
    question = "Tell me about Android reliability work"
    response = generate_engineer_response(question)

    assert response["experience"] == "data_quality"


def test_roi_question_selects_roi_story():
    question = "What's the ROI on your last project?"
    response = generate_engineer_response(question)

    assert response["experience"] == "roi_story"


def test_multiword_phrase_matching():
    question = "Walk me through the return on investment study"
    response = generate_engineer_response(question)

    assert response["experience"] == "roi_story"

    # Ensure that looking for "return on investment" does not spill into
    # unrelated words like "android".
    response = generate_engineer_response("Android reliability was the focus")
    assert response["experience"] == "data_quality"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("ROI?", ["roi"]),
        ("Android reliability", ["android", "reliability"]),
        ("Return-on-investment", ["return", "on", "investment"]),
    ],
)
def test_tokenize_handles_punctuation(text, expected):
    assert _tokenize(text) == expected


@pytest.mark.parametrize(
    "tokens, phrase, expected",
    [
        (["android", "reliability"], "android", True),
        (["return", "on", "investment"], "return on investment", True),
        (["return", "reliability", "investment"], "return on investment", False),
    ],
)
def test_phrase_in_tokens(tokens, phrase, expected):
    assert _phrase_in_tokens(tokens, phrase) is expected


def test_select_experience_defaults_when_no_keywords_match():
    experience = _select_experience("Tell me about hobbies outside work")

    assert experience is DEFAULT_EXPERIENCE
