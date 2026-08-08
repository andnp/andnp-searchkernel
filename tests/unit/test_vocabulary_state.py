from searchkernel.indices.vocabulary_state import VocabularyLifecycleState


def test_missing_state_reconstructs_stale_state_from_authoritative_terms() -> None:
    state = VocabularyLifecycleState.from_dict(
        None,
        has_authoritative_terms=True,
        has_materialized_terms=False,
    )

    assert state.status == "stale"
    assert state.authoritative_revision == 1
    assert state.materialized_revision == 0
    assert state.needs_catch_up() is True


def test_build_remains_stale_when_authoritative_terms_change_mid_build() -> None:
    state = VocabularyLifecycleState()
    state.mark_authoritative_mutation()
    build_revision = state.begin_build()
    state.mark_authoritative_mutation()

    state.finish_build(build_revision=build_revision, caught_up=True)

    assert state.status == "stale"
    assert state.ready_for_query_expansion() is False
    assert state.materialized_revision == 1

    state.finish_build(build_revision=state.authoritative_revision, caught_up=True)

    assert state.status == "ready"
    assert state.ready_for_query_expansion() is True


def test_failed_build_is_retryable_and_preserves_error() -> None:
    state = VocabularyLifecycleState()
    state.mark_authoritative_mutation()
    state.mark_failed("vocabulary index unavailable")

    assert state.status == "failed"
    assert state.last_error == "vocabulary index unavailable"
    assert state.needs_catch_up() is True
    assert state.ready_for_query_expansion() is False
