import pytest

from searchkernel.search.expansion import hypothetical_answer_expander, synonym_expander


def test_synonym_expander_appends_known_synonyms() -> None:
    expand = synonym_expander({"car": ["auto", "vehicle"]})
    assert expand("find a car") == "find a car auto vehicle"


def test_synonym_expander_expands_every_matching_term() -> None:
    expand = synonym_expander({"car": ["auto"], "dealer": ["seller"]})
    assert expand("car dealer") == "car dealer auto seller"


def test_synonym_expander_unmatched_query_returned_unchanged() -> None:
    expand = synonym_expander({"car": ["auto"]})
    assert expand("bicycle shop") == "bicycle shop"


def test_synonym_expander_deterministic_term_order_across_repeated_runs() -> None:
    expand = synonym_expander({"car": ["vehicle", "auto"], "dealer": ["seller"]})
    results = [expand("car dealer") for _ in range(5)]
    assert len(set(results)) == 1
    assert results[0] == "car dealer vehicle auto seller"


def test_synonym_expander_matching_is_case_insensitive() -> None:
    expand = synonym_expander({"car": ["auto"]})
    assert expand("Find a CAR") == "Find a CAR auto"


def test_synonym_expander_preserves_synonym_casing() -> None:
    expand = synonym_expander({"car": ["Automobile"]})
    assert expand("car") == "car Automobile"


def test_synonym_expander_deduplicates_synonyms_across_terms() -> None:
    expand = synonym_expander({"car": ["vehicle"], "auto": ["vehicle"]})
    assert expand("car auto") == "car auto vehicle"


def test_synonym_expander_skips_synonym_identical_to_matched_token() -> None:
    expand = synonym_expander({"car": ["car", "auto"]})
    assert expand("car") == "car auto"


def test_synonym_expander_caps_total_appended_terms() -> None:
    many_synonyms = [f"syn{i}" for i in range(50)]
    expand = synonym_expander({"car": many_synonyms})
    expanded = expand("car")
    appended = expanded.split()[1:]
    assert len(appended) == 20
    assert appended == many_synonyms[:20]


def test_synonym_expander_rejects_empty_synonyms() -> None:
    with pytest.raises(ValueError, match="synonyms"):
        synonym_expander({})


def test_hypothetical_answer_expander_returns_the_models_text() -> None:
    expand = hypothetical_answer_expander(lambda _prompt: "a hypothetical answer")
    assert expand("what is searchkernel?") == "a hypothetical answer"


def test_hypothetical_answer_expander_passes_the_query_into_the_prompt() -> None:
    captured: list[str] = []

    def complete(prompt: str) -> str:
        captured.append(prompt)
        return "answer"

    expand = hypothetical_answer_expander(complete)
    expand("what is searchkernel?")

    assert "what is searchkernel?" in captured[0]


def test_hypothetical_answer_expander_truncates_at_max_chars() -> None:
    expand = hypothetical_answer_expander(lambda _prompt: "x" * 100, max_chars=10)
    assert expand("query") == "x" * 10


def test_hypothetical_answer_expander_propagates_complete_failures() -> None:
    def complete(_prompt: str) -> str:
        raise RuntimeError("model unavailable")

    expand = hypothetical_answer_expander(complete)
    with pytest.raises(RuntimeError, match="model unavailable"):
        expand("query")


def test_hypothetical_answer_expander_rejects_non_positive_max_chars() -> None:
    with pytest.raises(ValueError, match="max_chars"):
        hypothetical_answer_expander(lambda _prompt: "text", max_chars=0)
