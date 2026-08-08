# Lifecycle and ownership

`DatabaseManager.close()` is terminal. It is idempotent, releases every
connection owned by the manager, and makes subsequent `get_connection()` and
`checkpoint()` calls fail with `RuntimeError`.

`LocalRecordBackend` owns the SQLite database it creates from `db_path` or
with its in-memory default. Its `close()` releases that database. When a
database manager is injected, the backend borrows it and does not close it;
the caller that created the manager remains responsible for its lifetime.

`build_local_record_kernel()` returns a composition that owns its local
backend. Call `close()` or use it as a context manager when the composition is
no longer needed. Closing the composition does not close an embedding
provider, reranker, or other externally supplied resource. Such resources
must be closed explicitly by the code that owns them.

After shutdown, callers must stop using the composition and its stores.
Concurrent callers may finish work already holding a connection, but a
manager that has observed shutdown never creates or returns another one.
