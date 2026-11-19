"""Shared fixtures for the database unit tests."""

import time
from types import ModuleType

import pytest

from lute.io.models.base import AnalysisHeader, TaskParameters
from lute.io.models.tests import TestParameters, TestWriteOutputParameters
from lute.tasks.dataclasses import DescribedAnalysis, TaskResult, TaskStatus

FIXTURE_EXPERIMENT: str = "mfxtest"


def _record(
    api: ModuleType,
    work_dir: str,
    params_cls: type,
    run: int,
    payload: str,
    status: TaskStatus = TaskStatus.COMPLETED,
    **kwargs,
) -> None:
    """Write one execution through the database API, as the Executor would."""
    params: TaskParameters = params_cls(
        lute_config=AnalysisHeader(
            experiment=FIXTURE_EXPERIMENT, run=run, work_dir=work_dir
        ),
        **kwargs,
    )
    # `add_execution` deletes these from the Config class after each write. The
    # Executor writes once per process, but the fixture writes several times.
    params.Config.version_specifier = None
    params.Config.task_version = None
    desc: DescribedAnalysis = DescribedAnalysis(
        task_result=TaskResult(
            task_name=params_cls.__name__[: -len("Parameters")],
            task_status=status,
            summary="",
            payload=payload,
        ),
        task_parameters=params,
        task_env={"LUTE_FIXTURE": "1"},
        executor_name="Executor",
        poll_interval=0.05,
        communicator_desc=["PipeCommunicator: pipe"],
    )
    api.record_analysis_db(desc)
    # v2 orders "latest" by a 1 second resolution timestamp with no tie-break.
    time.sleep(1.1)


def make_lute_db(db_dir: str, api: ModuleType) -> str:
    """Populate `db_dir/lute.db` with a small, known set of executions.

    - Run 36: `TestWriteOutput` twice (r36_a.txt then r36_b.txt), and one `Test`.
    - Run 37: `TestWriteOutput` once (r37_a.txt), then once FAILED (r37_fail.txt).

    Args:
        db_dir (str): Directory in which `lute.db` is created.

        api (ModuleType): The database API module to write with, e.g.
            `lute.io._db.v2.api`.

    Returns:
        db_dir (str): The same directory, for convenience.
    """
    with pytest.MonkeyPatch.context() as mp:
        # AnalysisHeader sets LUTE_WORK_DIR as a side effect of validation.
        mp.delenv("LUTE_WORK_DIR", raising=False)
        write: type = TestWriteOutputParameters
        _record(api, db_dir, write, 36, "r36_a.txt", outfile_name="r36_a.txt")
        _record(api, db_dir, write, 36, "r36_b.txt", outfile_name="r36_b.txt")
        _record(
            api, db_dir, TestParameters, 36, "test36", str_var="s36", compound_var={}
        )
        _record(api, db_dir, write, 37, "r37_a.txt", outfile_name="r37_a.txt")
        _record(
            api,
            db_dir,
            write,
            37,
            "r37_fail.txt",
            status=TaskStatus.FAILED,
            outfile_name="r37_fail.txt",
        )
    return db_dir


@pytest.fixture(scope="session")
def v2_db_dir(tmp_path_factory) -> str:
    """Directory holding the fixture `lute.db` written with DB v2."""
    import lute.io._db.v2.api as v2_api

    return make_lute_db(str(tmp_path_factory.mktemp("v2_db")), v2_api)


@pytest.fixture(scope="session")
def v3_db_dir(tmp_path_factory) -> str:
    """Directory holding the fixture `lute.db` written with DB v3."""
    import lute.io._db.v3.api as v3_api

    return make_lute_db(str(tmp_path_factory.mktemp("v3_db")), v3_api)
