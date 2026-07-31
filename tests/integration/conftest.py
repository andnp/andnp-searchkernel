"""Integration test fixtures with xdist worker isolation.

pytest.mark.xdist_group (used elsewhere in this suite for "serial" tests) only
has an effect under `--dist loadgroup` -- it is a no-op under this repo's
default `--dist worksteal` (see tests/integration/test_pgvector_index.py's
docstring). So pgvector integration tests cannot rely on grouping to avoid
running concurrently with other workers against the shared Postgres
container; they need real per-worker isolation instead.
"""

import os
import socket
import time
import uuid

import pytest


def _free_localhost_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _container_logs(container) -> str:
    return container.logs(tail=100).decode("utf-8", errors="replace")


def _wait_for_postgres_ready(container, timeout: float = 60.0) -> None:
    from docker.errors import APIError

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        container.reload()
        if container.status in {"dead", "exited"}:
            raise RuntimeError(
                "pgvector container exited before becoming ready:\n"
                f"{_container_logs(container)}"
            )

        try:
            result = container.exec_run(
                ["pg_isready", "-h", "127.0.0.1", "-p", "5432", "-U", "postgres"]
            )
        except APIError:
            result = None

        if result is not None and result.exit_code == 0:
            return
        time.sleep(0.5)

    raise TimeoutError(
        f"pgvector container did not become ready within {timeout:.0f} seconds:\n"
        f"{_container_logs(container)}"
    )


@pytest.fixture(scope="session", autouse=True)
def pgvector_test_database(request):
    """Provide SEARCHKERNEL_PG_DSN through Docker when not supplied."""
    existing_dsn = os.environ.get("SEARCHKERNEL_PG_DSN")
    if existing_dsn:
        yield
        return

    docker = pytest.importorskip(
        "docker", reason="Docker SDK is required for pgvector integration tests"
    )
    from docker.errors import DockerException, ImageNotFound, NotFound

    try:
        client = docker.from_env()
    except DockerException as exc:
        pytest.skip(f"Docker is unavailable for pgvector integration tests: {exc}")

    worker_id = getattr(request.config, "workerinput", {}).get("workerid", "master")
    container = None
    try:
        image_name = "pgvector/pgvector:pg17"
        try:
            image = client.images.get(image_name)
        except ImageNotFound:
            image = client.images.pull("pgvector/pgvector", "pg17")

        port = _free_localhost_port()
        container_name = f"searchkernel-pgvector-{worker_id}-{uuid.uuid4().hex[:8]}"
        try:
            container = client.containers.run(
                image,
                detach=True,
                environment={
                    "POSTGRES_DB": "searchkernel",
                    "POSTGRES_PASSWORD": "searchkernel",
                    "POSTGRES_USER": "postgres",
                },
                name=container_name,
                ports={"5432/tcp": port},
                tmpfs={"/var/lib/postgresql/data": "rw,size=512m"},
            )
        except DockerException as exc:
            pytest.skip(f"Could not start pgvector container: {exc}")

        _wait_for_postgres_ready(container)
        os.environ["SEARCHKERNEL_PG_DSN"] = (
            f"postgresql://postgres:searchkernel@127.0.0.1:{port}/searchkernel"
        )
        yield
    finally:
        if existing_dsn is None:
            os.environ.pop("SEARCHKERNEL_PG_DSN", None)
        else:
            os.environ["SEARCHKERNEL_PG_DSN"] = existing_dsn
        if container is not None:
            try:
                container.remove(force=True)
            except NotFound:
                pass
        client.close()


def pg_worker_schema(config) -> str:
    """A stable, valid Postgres schema name unique to this xdist worker.

    Returns e.g. "pgtest_gw0", or "pgtest_master" when not running under xdist.
    """
    worker_id = "master"
    if hasattr(config, "workerinput"):
        worker_id = config.workerinput.get("workerid", "master")
    return f"pgtest_{worker_id}"


def pg_dsn_for_schema(base_dsn: str, schema: str) -> str:
    """Return base_dsn with a libpq `options` param pinning search_path to schema.

    Table DDL/DML in this suite is never schema-qualified, so pinning
    search_path is sufficient to give each xdist worker its own private set
    of `records`/`vector_tables`/`graph_edges`/`cache_store`/`index_epoch`
    tables and per-model vector tables, without touching production code.
    """
    from urllib.parse import quote

    options = quote(f"-c search_path={schema},public", safe="")
    separator = "&" if "?" in base_dsn else "?"
    return f"{base_dsn}{separator}options={options}"
