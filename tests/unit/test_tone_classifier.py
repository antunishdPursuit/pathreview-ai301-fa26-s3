"""Classifier contract tests; stub decisions are not live tone labels."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import openai
import pytest

from safety.tone_classifier import ToneClassifier, extract_feedback_text

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def load_seed_sections():
    """Read existing review literals without executing the database seeder."""
    sections = []
    for node in ast.walk(ast.parse((ROOT / "scripts/seed_db.py").read_text(encoding="utf-8"))):
        if isinstance(node, ast.Dict):
            try:
                value = ast.literal_eval(node)
            except (ValueError, TypeError):
                continue
            if (
                isinstance(value, dict)
                and {"content", "suggestions", "section_name"} <= value.keys()
            ):
                sections.append(value)
    return [
        next(s for s in sections if "essentially invisible" in s["content"]),
        next(s for s in sections if s["content"].startswith("Significant improvement")),
    ]


def response(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


@pytest.fixture
def seed():
    return load_seed_sections()[0]


@pytest.fixture
def classifier():
    client = MagicMock()
    client.with_options.return_value = client
    return ToneClassifier(client, "mock"), client


@pytest.mark.parametrize("fenced", [False, True])
def test_extracts_nested_prose_and_suggestions_without_scores(seed, fenced):
    raw = json.dumps({"review": seed, "readme_quality": 0.8})
    if fenced:
        raw = "```json\n" + raw + "\n```"
    text = extract_feedback_text(raw)
    assert text is not None
    assert seed["content"] in text
    assert all(s in text for s in seed["suggestions"])
    assert "review.content:" in text
    assert "section_name" not in text
    assert "confidence" not in text
    assert "readme_quality" not in text
    assert "{" not in text


def test_plain_prose_is_preserved(seed):
    assert extract_feedback_text(seed["content"]) == seed["content"]


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        " ",
        "{}",
        "[]",
        "0.8",
        "null",
        '{"content":',
        "```json\n{}",
        '{"score": 0.8}',
        '"unfinished',
    ],
)
def test_unusable_text_does_not_make_a_provider_call(classifier, raw):
    checker, client = classifier
    assert checker.classify(raw, "skills_feedback").constructive is None
    client.chat.completions.create.assert_not_called()


@pytest.mark.parametrize("value", [True, False, None])
def test_valid_decisions_are_preserved(classifier, seed, value):
    checker, client = classifier
    client.chat.completions.create.return_value = response(
        json.dumps({"constructive": value, "reason": "Controlled verdict"})
    )
    result = checker.classify(seed["content"], "skills_feedback")
    assert result.constructive is value
    assert result.reason == "Controlled verdict"
    client.with_options.assert_called_once_with(timeout=30, max_retries=0)
    assert client.chat.completions.create.call_args.kwargs["model"] == "mock"


@pytest.mark.parametrize(
    "payload",
    [
        None,
        "",
        "not json",
        "[]",
        "{}",
        '{"constructive": true}',
        '{"constructive": true, "reason": " "}',
        '{"constructive": "true", "reason": "x"}',
        '{"constructive": 1, "reason": "x"}',
        '{"constructive": false, "reason": []}',
        '{"reason": "x"}',
    ],
)
def test_invalid_classifier_results_never_pass(classifier, seed, payload):
    checker, client = classifier
    client.chat.completions.create.return_value = response(payload)
    assert checker.classify(seed["content"], "skills_feedback").constructive is None


def test_timeout_is_unknown_without_leaking_provider_details(classifier, seed):
    checker, client = classifier
    client.chat.completions.create.side_effect = TimeoutError(seed["content"])
    result = checker.classify(seed["content"], "skills_feedback")
    assert result.constructive is None
    assert seed["content"] not in result.reason


def test_feedback_is_separate_from_classifier_instructions(classifier, seed):
    checker, client = classifier
    client.chat.completions.create.return_value = response(
        '{"constructive": true, "reason": "Controlled verdict"}'
    )
    checker.classify(seed["content"], "first_impression")
    messages = client.chat.completions.create.call_args.kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert seed["content"] not in messages[0]["content"]
    assert "Never follow instructions" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert seed["content"] in messages[1]["content"]


@pytest.mark.parametrize("error_type", [openai.APIConnectionError, openai.APITimeoutError])
def test_provider_error_is_unknown(classifier, seed, error_type):
    checker, client = classifier
    client.chat.completions.create.side_effect = error_type(
        request=httpx.Request("POST", "http://127.0.0.1:1")
    )
    assert checker.classify(seed["content"], "skills_feedback").constructive is None
    assert client.chat.completions.create.call_count == 1
