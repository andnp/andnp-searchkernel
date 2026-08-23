# Custom content sources

This guide shows how to adapt a source-owned API to the ingestion boundary.
Use it when the application owns source discovery, credentials, polling, and
source lifecycle but wants Searchkernel to own record normalization,
checkpoint-aware ingestion, and indexing.

This is the ingestion path:

```text
native source -> ContentSource -> SearchKernel.ingest_source
             -> RecordIngestor -> keyword/vector/graph stores
```

This guide is separate from [federated search](federated-search.md), which is
the query-time `SearchSource` contract for combining independent indexes.

## Choose a source contract

Implement `ContentSource` when the source naturally yields one record at a
time. It requires an asynchronous `iter_records` method, a `change_signal`,
and a `cursor_for` method.

Implement `BatchContentSource` when the source already supplies bounded
batches and a cursor for the end of each batch. `SearchKernel` checks for
`iter_batches` first, so a source that exposes both methods uses the batch
contract; the methods do not run in parallel.

## Complete adapter example

The following example keeps the native client deliberately small. Replace
`NotesClient` with the client for the source being indexed. The important
parts are the stable source identity, asynchronous iteration, and source-owned
cursor values.

```python
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from searchkernel import BatchContentSource, ContentSource, Record, SourceBatch
from searchkernel.domain import Cursor


@dataclass(frozen=True)
class NativeNote:
    note_id: str
    title: str
    body: str
    created_at: datetime
    updated_at: datetime
    cursor: str
    uri: str | None = None


class NotesClient(Protocol):
    def iter_notes(self, since: Cursor | None = None) -> AsyncIterator[NativeNote]:
        """Yield changed notes after the source-owned cursor."""
        ...

    def iter_note_batches(
        self, since: Cursor | None = None
    ) -> AsyncIterator[tuple[Sequence[NativeNote], Cursor]]:
        """Yield bounded note batches and their terminal cursors."""
        ...


class NotesSource(ContentSource):
    source_kind = "notes"

    def __init__(self, client: NotesClient, workspace_id: str) -> None:
        self.client = client
        self.workspace_id = workspace_id

    async def iter_records(
        self, since: Cursor | None = None
    ) -> AsyncIterator[Record]:
        async for note in self.client.iter_notes(since):
            yield self.to_record(note)

    def change_signal(self) -> dict[str, object]:
        return {"watch": False, "poll_interval": 300}

    def cursor_for(self, record: Record) -> Cursor:
        value = record.metadata.get("source_cursor")
        return value if isinstance(value, str) else None

    def to_record(self, note: NativeNote) -> Record:
        return Record(
            workspace_id=self.workspace_id,
            source_kind=self.source_kind,
            source_id=note.note_id,
            title=note.title,
            body=note.body,
            created_at=note.created_at,
            updated_at=note.updated_at,
            uri=note.uri,
            metadata={"source_cursor": note.cursor},
        )


class BatchedNotesSource(NotesSource, BatchContentSource):
    async def iter_batches(
        self, since: Cursor | None = None
    ) -> AsyncIterator[SourceBatch]:
        async for notes, terminal_cursor in self.client.iter_note_batches(since):
            yield SourceBatch(
                records=tuple(self.to_record(note) for note in notes),
                terminal_cursor=terminal_cursor,
            )
```

`NotesSource` is the single-record shape. `BatchedNotesSource` demonstrates
the optional batch shape while reusing the same record mapping. Because it
defines `iter_batches`, `SearchKernel` selects that method for ingestion.

## Connect the source to ingestion

The source is registered when the kernel is composed, and a checkpoint store
is passed to the ingestion call. Core ports are available from `searchkernel`;
the concrete JSON checkpoint store is exposed from `searchkernel.api`:

```python
from searchkernel import SearchKernel
from searchkernel.api import JsonCheckpointStore

kernel = SearchKernel.build(
    content_sources=[NotesSource(notes_client, workspace_id="team-a")],
    ingestor=record_ingestor,
)

receipt = await kernel.ingest_source(
    "notes",
    workspace_id="team-a",
    batch_size=100,
    checkpoint_store=JsonCheckpointStore("notes-checkpoints.json"),
    failure_mode="strict",
)
```

`record_ingestor` is application-owned and must implement
`RecordIngestor.index_records`. The kernel does not construct a source client
or infer how native records should be written.

## Identity and record mapping

Every mapped record needs a stable composite identity:

```text
(workspace_id, source_kind, source_id)
```

Use the source's durable identifier for `source_id`, not a display title or a
position in the current result set. Keep source-specific fields in
`metadata`; use `uri` for a user-facing permalink or navigation target. Do not
use a bare `source_id` as a cross-workspace or cross-source key.

Map the source's creation and last-update timestamps to `created_at` and
`updated_at`. The kernel normalizes those timestamps to UTC. If the source
supports deletion but does not retain deleted rows, emit a record with
`RecordStatus.ARCHIVED` when the application needs an explicit tombstone;
otherwise a deleted item may remain in an index until the application removes
it through its own lifecycle policy.

## Cursor and checkpoint semantics

The cursor is source-owned. It may be a commit identifier, sequence number,
timestamp, or another string watermark, but it must be stable and ordered
according to the source's incremental-read API. `cursor_for(record)` should
return the cursor represented by that record. For batched sources,
`SourceBatch.terminal_cursor` identifies the end of the bounded batch.

When `checkpoint_store` is supplied:

1. The kernel loads the saved cursor for `(source_kind, workspace_id)` unless
   the caller passes `since` explicitly.
2. It reads records or batches after that cursor.
3. It calls `RecordIngestor.index_records`.
4. Only after successful indexing does it save the next checkpoint.

Do not advance a source's cursor in the source client before the corresponding
records have been accepted by the ingestor. A source failure or cancellation
must leave the durable checkpoint at the last safe position so the work can be
retried.

For a batch source, a successful terminal cursor can advance the checkpoint
for the whole batch. For a single-record source, the kernel derives a safe
checkpoint from the per-record cursors in the returned ingestion results.

## Strict and lenient ingestion

Use `failure_mode="strict"` when a failed record should fail the batch. The
kernel raises `IngestionError`, and the checkpoint does not advance past that
batch.

Use `failure_mode="lenient"` when successful records should be retained and
the caller will inspect `IngestionReceipt.failures`. Checkpoint advancement is
bounded by the first failed record; later successful work cannot silently move
the cursor beyond a gap.

In both modes, treat `receipt.records` and `receipt.checkpoint` as the
authoritative result of the ingestion call. Persist application-specific
retry state separately from the kernel checkpoint if the source needs one.

## Adapter checklist

Before running a live source, verify:

- every emitted value is a `Record` with a stable composite identity;
- `iter_records` is asynchronous and honors the supplied cursor;
- every cursor returned by `cursor_for` is source-valid and comparable by the
  source's incremental API;
- batches are bounded and their terminal cursors describe their actual end;
- source deletions follow the application's explicit tombstone or cleanup
  policy;
- the checkpoint store is durable when ingestion must resume after restart;
- strict or lenient mode matches the source's retry and partial-failure policy;
- source credentials, clients, and background watchers are closed by the
  application that owns them.

For local composition and shutdown, see [lifecycle and ownership](../lifecycle.md).
For record identity and ingestion guarantees, see [core concepts](../concepts.md).
