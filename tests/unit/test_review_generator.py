"""Tone-check integration tests using existing repository data."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag.generator.review_generator import ReviewConfig, ReviewGenerator
from tests.unit.test_tone_classifier import load_seed_sections, response

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
NAMES = [
    "skills_feedback",
    "projects_feedback",
    "presentation_feedback",
    "gaps_feedback",
    "first_impression",
]


def decision(value):
    return response(json.dumps({"constructive": value, "reason": "Controlled test verdict"}))


@pytest.fixture
def setup_review():
    profile = json.loads(
        (ROOT / "tests/fixtures/sample_profiles/basic_profile.json").read_text(encoding="utf-8")
    )
    profile["projects"] = profile["repos"]
    chunks = [
        {"text": r["readme_content"], "metadata": {"source_id": r["name"]}}
        for r in profile["repos"]
    ]
    client = MagicMock()
    client.with_options.return_value = client
    with patch("rag.generator.review_generator.openai.OpenAI", return_value=client):
        generator = ReviewGenerator(ReviewConfig("unused", "http://127.0.0.1:1", "mock"))
    return generator, client, profile, chunks, load_seed_sections()


def test_pass_checks_before_parsing_and_preserves_output(setup_review):
    generator, client, profile, chunks, seeds = setup_review
    raw = json.dumps({"skills_feedback": seeds[0]})
    client.chat.completions.create.side_effect = [response(raw), decision(True)]
    result = generator.generate_section("skills_feedback", chunks, profile)
    assert client.chat.completions.create.call_count == 2
    assert result.content == json.dumps(seeds[0])
    assert result.suggestions == seeds[0]["suggestions"]
    check_message = client.chat.completions.create.call_args_list[1].kwargs["messages"][1][
        "content"
    ]
    assert seeds[0]["content"] in check_message
    assert all(s in check_message for s in seeds[0]["suggestions"])
    assert "Sources:" not in check_message
    assert all(
        c.kwargs == {"timeout": 30, "max_retries": 0} for c in client.with_options.call_args_list
    )


def test_failed_response_is_replaced_only_after_passing_check(setup_review):
    generator, client, profile, chunks, seeds = setup_review
    first, second = (json.dumps({"skills_feedback": s}) for s in seeds)
    client.chat.completions.create.side_effect = [
        response(first),
        decision(False),
        response(second),
        decision(True),
    ]
    result = generator.generate_section("skills_feedback", chunks, profile)
    assert client.chat.completions.create.call_count == 4
    assert result.content == json.dumps(seeds[1])
    assert seeds[0]["content"] not in result.content
    retry_prompt = client.chat.completions.create.call_args_list[2].kwargs["messages"][1]["content"]
    first_prompt = client.chat.completions.create.call_args_list[0].kwargs["messages"][1]["content"]
    assert retry_prompt.startswith(first_prompt)
    assert "constructive" in retry_prompt


@pytest.mark.parametrize("verdict", [False, None])
def test_exhaustion_returns_unavailable_without_failed_text(setup_review, verdict):
    generator, client, profile, chunks, seeds = setup_review
    raw = json.dumps({"skills_feedback": seeds[0]})
    client.chat.completions.create.side_effect = [response(raw), decision(verdict)] * 2
    result = generator.generate_section("skills_feedback", chunks, profile)
    assert client.chat.completions.create.call_count == 4
    assert result.section_name == "skills_feedback"
    assert result.confidence == 0.0
    assert result.suggestions == []
    assert "unavailable" in result.content.lower()
    assert seeds[0]["content"] not in result.content


def test_full_review_checks_all_five_sections_before_citations(setup_review):
    generator, client, profile, chunks, seeds = setup_review
    client.chat.completions.create.side_effect = [
        item for name in NAMES for item in (response(json.dumps({name: seeds[0]})), decision(True))
    ]
    results = generator.generate_full_review(profile, chunks)
    assert [s.section_name for s in results] == NAMES
    assert client.chat.completions.create.call_count == 10
    citation = "\nSources: " + ", ".join(sorted(c["metadata"]["source_id"] for c in chunks))
    assert all(s.content == json.dumps(seeds[0]) + citation for s in results)
    assert all(s.suggestions == seeds[0]["suggestions"] for s in results)
    for call in client.chat.completions.create.call_args_list[1::2]:
        assert "Sources:" not in call.kwargs["messages"][1]["content"]


def test_full_review_exhaustion_is_bounded_and_continues(setup_review):
    generator, client, profile, chunks, seeds = setup_review
    client.chat.completions.create.side_effect = [
        item
        for name in NAMES
        for _ in range(2)
        for item in (response(json.dumps({name: seeds[0]})), decision(False))
    ]
    results = generator.generate_full_review(profile, chunks)
    assert [s.section_name for s in results] == NAMES
    assert client.chat.completions.create.call_count == 20
    assert all(s.confidence == 0.0 and s.suggestions == [] for s in results)
    assert all(seeds[0]["content"] not in s.content for s in results)


def test_generation_timeout_is_bounded_and_private(setup_review, capsys):
    generator, client, profile, chunks, seeds = setup_review
    client.chat.completions.create.side_effect = TimeoutError(seeds[0]["content"])
    result = generator.generate_section("skills_feedback", chunks, profile)
    assert client.chat.completions.create.call_count == 2
    assert seeds[0]["content"] not in capsys.readouterr().out
    assert result.confidence == 0.0
    assert seeds[0]["content"] not in result.content


def test_later_json_fields_are_checked_before_first_field_selection(setup_review):
    generator, client, profile, chunks, seeds = setup_review
    raw = json.dumps({"readme_quality": 0.8, "presentation_suggestions": seeds[0]["suggestions"]})
    client.chat.completions.create.side_effect = [response(raw), decision(True)]
    result = generator.generate_section("presentation_feedback", chunks, profile)
    assert client.chat.completions.create.call_count == 2
    check_text = client.chat.completions.create.call_args_list[1].kwargs["messages"][1]["content"]
    assert all(s in check_text for s in seeds[0]["suggestions"])
    # Preserve the current parser result; repairing it is outside issue 27.
    assert result.content == "0.8"


@pytest.mark.parametrize("failure", [response("not json"), TimeoutError("provider unavailable")])
def test_invalid_check_or_timeout_exhausts_without_returning_text(setup_review, failure):
    generator, client, profile, chunks, seeds = setup_review
    raw = json.dumps({"skills_feedback": seeds[0]})
    client.chat.completions.create.side_effect = [response(raw), failure] * 2
    result = generator.generate_section("skills_feedback", chunks, profile)
    assert client.chat.completions.create.call_count == 4
    assert result.confidence == 0.0
    assert seeds[0]["content"] not in result.content


def test_exhausted_section_does_not_block_later_accepted_sections(setup_review):
    generator, client, profile, chunks, seeds = setup_review
    client.chat.completions.create.side_effect = [
        response(json.dumps({NAMES[0]: seeds[0]})),
        decision(False),
        response(json.dumps({NAMES[0]: seeds[0]})),
        decision(False),
    ] + [
        item
        for name in NAMES[1:]
        for item in (response(json.dumps({name: seeds[1]})), decision(True))
    ]
    results = generator.generate_full_review(profile, chunks)
    assert [s.section_name for s in results] == NAMES
    assert results[0].confidence == 0.0
    assert all(s.confidence > 0 for s in results[1:])
    assert client.chat.completions.create.call_count == 12
