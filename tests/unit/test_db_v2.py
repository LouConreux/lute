import sqlite3
import pytest
import os

# import json
# from unittest.mock import MagicMock
# from lute.io._db.v2.api import (
#     record_analysis_db,
#     record_parameters_db,
#     update_analysis_db,
# )
from lute.io._db.v2._sqlite import create_tables

# from lute.tasks.dataclasses import DescribedAnalysis, TaskResult, TaskStatus
# from lute.io.parameters import TaskParameters, RowIds


# Complicated to setup TaskParameters... Do some basic mocking for now...
class MockLuteConfig:
    def __init__(self, work_dir):
        self.work_dir = work_dir


class MockTaskParameters:
    def __init__(self, work_dir):
        self.lute_config = MockLuteConfig(work_dir)

    def dict(self):
        return {"param1": "val1"}

    def schema(self):
        return {"properties": {"param1": {"type": "string"}}}


@pytest.fixture
def temp_db_dir(tmp_path):
    d = tmp_path / "db"
    d.mkdir()
    return str(d)


def test_create_tables_v2(temp_db_dir):
    db_path = os.path.join(temp_db_dir, "lute.db")
    con = sqlite3.connect(db_path)
    create_tables(con)

    # Check if key tables exist
    res = con.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in res.fetchall()]
    assert "executions" in tables
    assert "tasks" in tables
    assert "parameters" in tables
    assert "results" in tables
    con.close()


# Baseline answers of `read_latest_db_entry` on the `make_lute_db` fixture.
# (for_run, valid_only, param, expected). Later DB versions must give the same.
BASELINE_LATEST = [
    (36, True, "outfile_name", "r36_b.txt"),
    (36, False, "outfile_name", "r36_b.txt"),
    ("36", True, "outfile_name", "r36_b.txt"),
    (37, True, "outfile_name", "r37_a.txt"),
    (37, False, "outfile_name", "r37_fail.txt"),
    (None, True, "outfile_name", "r37_a.txt"),
    (None, False, "outfile_name", "r37_fail.txt"),
    (36, True, "result.payload", "r36_b.txt"),
    (37, False, "result.payload", "r37_fail.txt"),
    (37, True, "result.valid_flag", 1),
    (37, False, "result.valid_flag", 0),
    (38, True, "outfile_name", None),
]


@pytest.mark.parametrize("for_run, valid_only, param, expected", BASELINE_LATEST)
def test_read_latest_db_entry_v2(v2_db_dir, for_run, valid_only, param, expected):
    from lute.io._db.v2.api import read_latest_db_entry

    value = read_latest_db_entry(
        v2_db_dir,
        "TestWriteOutput",
        param,
        valid_only=valid_only,
        for_run=for_run,
    )
    assert value == expected


def test_read_latest_db_entry_v2_filters_task(v2_db_dir):
    from lute.io._db.v2.api import read_latest_db_entry

    assert read_latest_db_entry(v2_db_dir, "Test", "str_var", for_run=36) == "s36"
    assert read_latest_db_entry(v2_db_dir, "Test", "str_var", for_run=37) is None


def test_read_latest_db_entry_v2_run_not_normalised(v2_db_dir):
    """v2 stores the run as text, so "0036" does not match run 36 (fixed in v3)."""
    from lute.io._db.v2.api import read_latest_db_entry

    value = read_latest_db_entry(
        v2_db_dir, "TestWriteOutput", "outfile_name", for_run="0036"
    )
    assert value is None


def test_fixture_counts_v2(v2_db_dir):
    con = sqlite3.connect(os.path.join(v2_db_dir, "lute.db"))
    n_exec = con.execute("SELECT COUNT(*) FROM executions").fetchone()[0]
    runs = con.execute("SELECT DISTINCT run FROM config ORDER BY run").fetchall()
    con.close()
    assert n_exec == 5
    assert runs == [("36",), ("37",)]
