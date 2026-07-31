from searchkernel.ports import CandidateFilterSupport


class CandidateFilteringAdapter:
    supports_candidate_filtering = True


class BasicAdapter:
    pass


def test_candidate_filter_support_is_opt_in() -> None:
    assert isinstance(CandidateFilteringAdapter(), CandidateFilterSupport)
    assert not isinstance(BasicAdapter(), CandidateFilterSupport)
