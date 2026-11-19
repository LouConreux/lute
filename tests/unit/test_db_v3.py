import multiprocessing
import os
import shutil
import signal
import sqlite3
import subprocess
import sys

import pytest

import lute.io._db.v3._sqlite_versioning as versioning
from lute.io._db.common_sqlite import DatabaseError, get_user_version
from lute.io._db.v3._sqlite import create_tables
from lute.io._db.v3._sqlite_versioning import open_db
from lute.io.db import import_function

from test_db_v2 import BASELINE_LATEST


def test_create_tables_v3(tmp_path):
    con = sqlite3.connect(str(tmp_path / "lute.db"))
    create_tables(con)

    res = con.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in res.fetchall()]
    assert "executions" in tables
    assert "tasks" in tables
    assert "parameters" in tables
    assert "results" in tables
    con.close()


@pytest.mark.parametrize("for_run, valid_only, param, expected", BASELINE_LATEST)
def test_read_latest_db_entry_v3(v3_db_dir, for_run, valid_only, param, expected):
    from lute.io._db.v3.api import read_latest_db_entry

    value = read_latest_db_entry(
        v3_db_dir,
        "TestWriteOutput",
        param,
        valid_only=valid_only,
        for_run=for_run,
    )
    assert value == expected


def test_read_latest_db_entry_v3_filters_task(v3_db_dir):
    from lute.io._db.v3.api import read_latest_db_entry

    assert read_latest_db_entry(v3_db_dir, "Test", "str_var", for_run=36) == "s36"
    assert read_latest_db_entry(v3_db_dir, "Test", "str_var", for_run=37) is None


def test_import_function_v3():
    func = import_function("read_latest_db_entry", 3)
    assert func.__module__ == "lute.io._db.v3.api"


def test_env_var_selects_v3():
    """`LUTE_DB_SPEC_VERSION` is read at import, so check it in a fresh process."""
    code = (
        "from lute.io.db import LUTE_DB_SPEC_VERSION, import_function;"
        "print(LUTE_DB_SPEC_VERSION,"
        " import_function('update_analysis_db', LUTE_DB_SPEC_VERSION).__module__)"
    )
    env = dict(os.environ, LUTE_DB_SPEC_VERSION="3")
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert out[-2:] == ["3", "lute.io._db.v3.api"]


@pytest.mark.parametrize("version", ["v2", "v3"])
def test_api_import_does_not_load_backend(version):
    """The API layer only imports the backend inside its functions."""
    code = (
        "import sys;"
        f"import lute.io._db.{version}.api;"
        "print(sorted(m for m in sys.modules"
        f" if m.startswith('lute.io._db.{version}._')))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    assert out[-1] == "[]"


# Versioning and migrations (step 2)


def _user_version(db_path):
    con = sqlite3.connect(db_path)
    try:
        return get_user_version(con)
    finally:
        con.close()


def _tables(db_path):
    con = sqlite3.connect(db_path)
    try:
        res = con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return {r[0] for r in res.fetchall()}
    finally:
        con.close()


def test_open_db_new_file_is_latest(tmp_path):
    db_path = str(tmp_path / "lute.db")
    con = open_db(db_path)
    n_triggers = con.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='trigger'"
    ).fetchone()[0]
    n_base_schema = con.execute("SELECT COUNT(*) FROM base_schema").fetchone()[0]
    con.close()

    assert _user_version(db_path) == versioning.latest_user_version() == 300
    assert {"executions", "config", "parameters", "results"} <= _tables(db_path)
    assert n_triggers == 6
    assert n_base_schema == 2


def test_open_db_twice_is_noop(tmp_path):
    db_path = str(tmp_path / "lute.db")
    open_db(db_path).close()
    open_db(db_path).close()
    assert _user_version(db_path) == 300
    assert not os.path.exists(f"{db_path}.bak-v300")


def test_fixture_written_with_v3_is_latest(v3_db_dir):
    assert _user_version(os.path.join(v3_db_dir, "lute.db")) == 300


def test_migration_applies_with_backup(tmp_path, v3_db_dir, monkeypatch):
    db_path = str(tmp_path / "lute.db")
    shutil.copy(os.path.join(v3_db_dir, "lute.db"), db_path)

    def add_table(con):
        con.execute("CREATE TABLE new_table (id INTEGER PRIMARY KEY)")

    monkeypatch.setattr(
        versioning, "MIGRATIONS", versioning.MIGRATIONS + [(300, 301, add_table)]
    )
    open_db(db_path).close()

    assert _user_version(db_path) == 301
    assert "new_table" in _tables(db_path)
    backup = f"{db_path}.bak-v300"
    assert _user_version(backup) == 300
    assert "new_table" not in _tables(backup)


def test_failed_migration_leaves_file_intact(tmp_path, v3_db_dir, monkeypatch):
    db_path = str(tmp_path / "lute.db")
    shutil.copy(os.path.join(v3_db_dir, "lute.db"), db_path)

    def broken(con):
        con.execute("CREATE TABLE half_done (id INTEGER PRIMARY KEY)")
        con.execute("UPDATE results SET valid_flag = 0")
        raise RuntimeError("migration failed")

    monkeypatch.setattr(
        versioning, "MIGRATIONS", versioning.MIGRATIONS + [(300, 301, broken)]
    )
    with pytest.raises(RuntimeError):
        open_db(db_path)

    assert _user_version(db_path) == 300
    assert "half_done" not in _tables(db_path)
    con = sqlite3.connect(db_path)
    n_valid = con.execute("SELECT SUM(valid_flag) FROM results").fetchone()[0]
    con.close()
    assert n_valid == 4  # 5 executions, 1 failed


def _killed_migration(con):
    con.execute("UPDATE results SET valid_flag = 0")
    # A small cache makes SQLite write pages to the file before the commit
    con.execute("PRAGMA cache_size = -64")
    con.execute("CREATE TABLE big (x TEXT)")
    rows = (("y" * 1000,) for _ in range(5000))
    con.executemany("INSERT INTO big VALUES (?)", rows)
    os.kill(os.getpid(), signal.SIGKILL)  # The job is killed mid-migration


def test_killed_migration_leaves_file_intact(tmp_path, v3_db_dir, monkeypatch):
    """The journal on disk lets the next connection undo a killed migration."""
    db_path = str(tmp_path / "lute.db")
    shutil.copy(os.path.join(v3_db_dir, "lute.db"), db_path)
    monkeypatch.setattr(
        versioning,
        "MIGRATIONS",
        versioning.MIGRATIONS + [(300, 301, _killed_migration)],
    )
    ctx = multiprocessing.get_context("fork")
    proc = ctx.Process(target=open_db, args=(db_path,))
    proc.start()
    proc.join(timeout=120)
    assert proc.exitcode == -signal.SIGKILL

    con = sqlite3.connect(db_path)  # Rolls back from lute.db-journal
    try:
        assert con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        n_valid = con.execute("SELECT SUM(valid_flag) FROM results").fetchone()[0]
    finally:
        con.close()
    assert n_valid == 4
    assert _user_version(db_path) == 300
    assert "big" not in _tables(db_path)


def test_missing_migration_raises(tmp_path, monkeypatch):
    db_path = str(tmp_path / "lute.db")
    open_db(db_path).close()
    monkeypatch.setattr(
        versioning,
        "MIGRATIONS",
        versioning.MIGRATIONS + [(301, 302, lambda con: None)],
    )
    with pytest.raises(DatabaseError, match="No migration"):
        open_db(db_path)
    assert _user_version(db_path) == 300


def _open_after_barrier(db_path, barrier, results):
    barrier.wait()
    try:
        open_db(db_path).close()
        results.put("ok")
    except Exception as err:  # Report to the parent instead of dying silently
        results.put(repr(err))


def test_concurrent_openers_migrate_once(tmp_path):
    db_path = str(tmp_path / "lute.db")
    n_procs = 6
    ctx = multiprocessing.get_context("fork")
    barrier = ctx.Barrier(n_procs)
    results = ctx.Queue()
    procs = [
        ctx.Process(target=_open_after_barrier, args=(db_path, barrier, results))
        for _ in range(n_procs)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=120)

    assert [results.get(timeout=5) for _ in procs] == ["ok"] * n_procs
    assert _user_version(db_path) == 300
    con = sqlite3.connect(db_path)
    assert con.execute("SELECT COUNT(*) FROM base_schema").fetchone()[0] == 2
    con.close()


def test_opener_racing_a_migration(tmp_path, monkeypatch):
    """Another process migrates right after this one read the old version.

    This opener reads 0 (new file), then another one creates the tables and
    commits before this one checks anything. It must not mistake the result for
    a v1/v2 file (version 0 with tables), and must find it already migrated.
    """
    db_path = str(tmp_path / "lute.db")
    real_get_user_version = versioning.get_user_version
    raced = []

    def stale_then_migrated(con):
        user_version = real_get_user_version(con)
        if not raced:
            raced.append(True)
            open_db(db_path).close()  # The other process migrates and commits
        return user_version

    monkeypatch.setattr(versioning, "get_user_version", stale_then_migrated)
    con = open_db(db_path)
    con.close()

    assert raced
    assert _user_version(db_path) == 300


def test_v3_refuses_v2_file(v2_db_dir):
    db_path = os.path.join(v2_db_dir, "lute.db")
    with pytest.raises(DatabaseError, match="LUTE_DB_SPEC_VERSION=2"):
        open_db(db_path)
    assert _user_version(db_path) == 0


def test_v3_api_refuses_v2_file(v2_db_dir):
    from lute.io._db.v3.api import read_latest_db_entry

    with pytest.raises(DatabaseError):
        read_latest_db_entry(v2_db_dir, "TestWriteOutput", "outfile_name")


def test_v3_refuses_newer_file(tmp_path):
    db_path = str(tmp_path / "lute.db")
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA user_version = 999")
    con.close()
    with pytest.raises(DatabaseError, match="newer"):
        open_db(db_path)


def test_v2_refuses_v3_file(v3_db_dir):
    from lute.io._db.v2.api import read_latest_db_entry

    with pytest.raises(DatabaseError, match="LUTE_DB_SPEC_VERSION=3"):
        read_latest_db_entry(v3_db_dir, "TestWriteOutput", "outfile_name")


# Atomic writes: one execution is written entirely or not at all


def _described(work_dir, run=36):
    from lute.io.models.base import AnalysisHeader
    from lute.io.models.tests import TestWriteOutputParameters
    from lute.tasks.dataclasses import DescribedAnalysis, TaskResult, TaskStatus

    params = TestWriteOutputParameters(
        lute_config=AnalysisHeader(experiment="mfxtest", run=run, work_dir=work_dir)
    )
    # `add_execution` deletes these from the Config class after each write
    params.Config.version_specifier = None
    params.Config.task_version = None
    return DescribedAnalysis(
        task_result=TaskResult(
            task_name="TestWriteOutput",
            task_status=TaskStatus.COMPLETED,
            summary="",
            payload="",
        ),
        task_parameters=params,
        task_env={"LUTE_TEST": "1"},
        executor_name="Executor",
        poll_interval=0.05,
        communicator_desc=["PipeCommunicator: pipe"],
    )


def _row_counts(db_path):
    con = sqlite3.connect(db_path)
    try:
        tables = [
            r[0]
            for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
                " AND name NOT LIKE 'sqlite_%'"
            )
        ]
        count = 'SELECT COUNT(*) FROM "{}"'
        return {t: con.execute(count.format(t)).fetchone()[0] for t in tables}
    finally:
        con.close()


def _fail_env_vars(monkeypatch, error):
    """Make the last step of an execution write (environment variables) fail."""
    import lute.io._db.v3._sqlite as backend

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(backend, "_add_env_vars", fail)


def test_failed_write_leaves_db_unchanged(tmp_path, monkeypatch):
    from lute.io._db.v3 import api

    db_path = str(tmp_path / "lute.db")
    open_db(db_path).close()
    before = _row_counts(db_path)

    # `executions`, `parameters`, `config`... are written before it fails
    _fail_env_vars(monkeypatch, sqlite3.OperationalError("database is locked"))
    api.record_analysis_db(_described(str(tmp_path)))  # Logged, not raised

    assert _row_counts(db_path) == before
    assert before["executions"] == 0


def test_unexpected_error_rolls_back_and_raises(tmp_path, monkeypatch):
    from lute.io._db.v3 import api

    db_path = str(tmp_path / "lute.db")
    open_db(db_path).close()
    before = _row_counts(db_path)

    _fail_env_vars(monkeypatch, RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        api.record_analysis_db(_described(str(tmp_path)))

    assert _row_counts(db_path) == before


def test_failed_update_keeps_placeholder(tmp_path, monkeypatch):
    from lute.io._db.v3 import api

    db_path = str(tmp_path / "lute.db")
    desc = _described(str(tmp_path))
    row_ids = api.record_parameters_db(desc.task_parameters)  # Task side
    assert row_ids is not None
    before = _row_counts(db_path)

    _fail_env_vars(monkeypatch, sqlite3.OperationalError("database is locked"))
    api.update_analysis_db(desc, row_ids)  # Executor side

    assert _row_counts(db_path) == before
    assert before["executions"] == 0


def test_successful_write_after_failure(tmp_path, monkeypatch):
    """A failed write does not leave the connection or the file locked."""
    from lute.io._db.v3 import api

    _fail_env_vars(monkeypatch, sqlite3.OperationalError("database is locked"))
    api.record_analysis_db(_described(str(tmp_path)))
    monkeypatch.undo()

    api.record_analysis_db(_described(str(tmp_path)))
    counts = _row_counts(str(tmp_path / "lute.db"))
    assert counts["executions"] == 1
    assert counts["environment"] == 1
