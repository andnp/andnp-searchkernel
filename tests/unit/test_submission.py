from typing import Literal

import pytest

from searchkernel.indexing.submission import (
    TaskBatchSubmissionResult,
    TaskSubmissionResult,
)


@pytest.mark.parametrize(
    ("status", "accepted", "retry", "available", "enqueued"),
    [
        ("enqueued", True, False, True, True),
        ("already_pending", True, False, True, False),
        ("backpressured", False, True, True, False),
        ("unavailable", False, False, False, False),
    ],
)
def test_task_submission_result_exposes_queue_outcome(
    status: Literal["enqueued", "already_pending", "backpressured", "unavailable"],
    accepted: bool,
    retry: bool,
    available: bool,
    enqueued: bool,
) -> None:
    result = TaskSubmissionResult(status)

    assert result.accepted_by_queue is accepted
    assert result.should_retry_later is retry
    assert result.queue_available is available
    assert result.enqueued is enqueued


def test_batch_submission_result_distinguishes_backpressure_from_unavailability() -> None:
    backpressured = TaskBatchSubmissionResult(
        queue_available=True,
        requested_unique_count=3,
        enqueued_count=1,
        already_pending_count=1,
        backpressured_items=("doc-3",),
    )
    unavailable = TaskBatchSubmissionResult(
        queue_available=False,
        requested_unique_count=3,
        enqueued_count=0,
    )

    assert backpressured.backpressured_count == 1
    assert backpressured.should_retry_later is True
    assert backpressured.all_represented is False
    assert unavailable.should_retry_later is False
    assert unavailable.all_represented is False


def test_batch_submission_is_represented_when_enqueued_and_pending_cover_request() -> None:
    result = TaskBatchSubmissionResult(
        queue_available=True,
        requested_unique_count=3,
        enqueued_count=2,
        already_pending_count=1,
    )

    assert result.all_represented is True
