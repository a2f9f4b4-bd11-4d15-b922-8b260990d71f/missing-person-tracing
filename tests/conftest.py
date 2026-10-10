import os

import psycopg
import pytest

from tracer.store import Store

TEST_DB = "tracer_test"


def _test_dsn() -> str:
    return os.environ["TRACER_DSN"].rsplit("/", 1)[0] + f"/{TEST_DB}"


@pytest.fixture(scope="session", autouse=True)
def _ensure_test_database():
    """Tests run against their own database. Sharing the dev one means a
    fixture's TRUNCATE turns every clip on disk into a retention orphan."""
    with psycopg.connect(os.environ["TRACER_DSN"], autocommit=True) as conn:
        found = conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (TEST_DB,)).fetchone()
        if not found:
            conn.execute(f"CREATE DATABASE {TEST_DB}")


@pytest.fixture
def store() -> Store:
    s = Store(_test_dsn())
    s.ensure_schema()
    s.reset()
    yield s
    s.reset()
    s.close()
