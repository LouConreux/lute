"""Tools for working with the LUTE parameter and configuration database.

The current implementation relies on a sqlite backend database. In the future
this may change - therefore relatively few high-level API function calls are
intended to be public. These abstract away the details of the database interface
and work exclusively on LUTE objects.

Functions:
    record_analysis_db(cfg: DescribedAnalysis) -> None: Writes the configuration
        to the backend database.

    read_latest_db_entry(db_dir: str, task_name: str, param: str) -> Any: Retrieve
        the most recent entry from a database for a specific Task.

Exceptions:
    DatabaseError: Generic exception raised for LUTE database errors.
"""

__all__ = [
    "record_analysis_db",
    "record_parameters_db",
    "read_latest_db_entry",
]
__author__ = "Gabriel Dorlhiac"

import json
import logging
import os
import stat
from typing import Any, Dict, List, Optional, Tuple, Union

from lute.execution.logging import get_logger
from lute.io.parameters import RowIds, TaskParameters
from lute.tasks.dataclasses import DescribedAnalysis

if __debug__:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

logger: logging.Logger = get_logger(__name__, is_task=False)


def record_analysis_db(cfg: DescribedAnalysis) -> None:
    """Write an DescribedAnalysis object to the database.

    The DescribedAnalysis object is maintained by the Executor and contains all
    information necessary to fully describe a single `Task` execution. The
    contained fields are split across multiple tables within the database as
    some of the information can be shared across multiple Tasks. Refer to
    `docs/design/database.md` for more information on the database specification.

    This function is meant to be called by the Executor at the end of Task
    execution, assuming the Task has not previously entered partial data into the
    database. See `record_parameters_db` and `update_task_entry_db` for how to
    handle the case where the Task and Executor both store some information.

    Args:
        cfg (DescribedAnalysis): The DescribedAnalysis completed by the Executor
            after Task completion.
    """
    import sqlite3
    from lute.io._db.common_sqlite import write_transaction
    from ._sqlite import add_execution
    from ._sqlite_versioning import open_db

    from lute.io.models.base import TaskParameters

    assert isinstance(cfg.task_parameters, TaskParameters)
    try:
        assert hasattr(cfg.task_parameters, "lute_config")
        work_dir: str = cfg.task_parameters.lute_config.work_dir
    except AttributeError:
        logger.error(
            (
                "Unable to access TaskParameters object. Likely wasn't created. "
                "Cannot store result."
            )
        )
        return
    assert hasattr(cfg.task_parameters, "lute_config")
    del cfg.task_parameters.lute_config.work_dir

    db_path: str = f"{work_dir}/lute.db"
    con: sqlite3.Connection = open_db(db_path)
    try:
        # One transaction: the execution is written entirely or not at all
        with write_transaction(con):
            add_execution(con=con, cfg=cfg)
    except sqlite3.OperationalError as err:
        logger.error(f"Database storage error: {err}")
    finally:
        con.close()

    db_stat: os.stat_result = os.stat(db_path)
    db_perms: str = oct(stat.S_IMODE(db_stat.st_mode))
    if db_perms != "0o664":
        try:
            os.chmod(db_path, 0o664)
        except Exception:
            logger.warning("Cannot setup permissions on database!")


def record_parameters_db(params: TaskParameters) -> Optional[RowIds]:
    """Write all tables that are possible using only the TaskParameters.

    This function is intended to be called at the Task layer, prior to completing
    execution. It allows saving state into the database.

    The entries are then intended to be updated by the Executor upon completion
    with all information which was excluded.

    Args:
        params (TaskParameters): The TaskParameters object - fully validated.
    """
    import sqlite3
    from lute.io._db.common_sqlite import write_transaction
    from ._sqlite import add_placeholder_execution
    from ._sqlite_versioning import open_db

    from lute.io.models.base import TaskParameters

    assert isinstance(params, TaskParameters)
    try:
        assert hasattr(params, "lute_config")
        work_dir: str = params.lute_config.work_dir
    except AttributeError:
        logger.error(
            (
                "Unable to access TaskParameters object. Likely wasn't created. "
                "Cannot store result."
            )
        )
        return None
    assert hasattr(params, "lute_config")
    del params.lute_config.work_dir

    db_path: str = f"{work_dir}/lute.db"
    con: sqlite3.Connection = open_db(db_path)
    row_ids: Optional[RowIds]
    try:
        with write_transaction(con):
            row_ids = add_placeholder_execution(con=con, params=params)
    except sqlite3.OperationalError as err:
        logger.error(f"Database storage error: {err}")
        row_ids = None
    finally:
        con.close()
    try:
        os.chmod(db_path, 0o664)
    except Exception:
        logger.error("Cannot setup permissions on database!")
    params.lute_config.work_dir = work_dir
    return row_ids


def update_analysis_db(cfg: DescribedAnalysis, row_ids: RowIds) -> None:
    import sqlite3
    from lute.io._db.common_sqlite import write_transaction
    from ._sqlite import update_execution
    from ._sqlite_versioning import open_db

    assert cfg.task_parameters is not None
    db_dir: str = cfg.task_parameters.lute_config.work_dir
    del cfg.task_parameters.lute_config.work_dir
    db_path: str = f"{db_dir}/lute.db"
    con: sqlite3.Connection = open_db(db_path)
    try:
        with write_transaction(con):
            update_execution(con=con, cfg=cfg, row_ids=row_ids)
    except sqlite3.OperationalError as err:
        logger.error(f"Database storage error: {err}")
    finally:
        con.close()
    try:
        os.chmod(db_path, 0o664)
    except Exception:
        logger.error("Cannot setup permissions on database!")


def read_latest_db_entry(
    db_dir: str,
    task_name: str,
    param: str,
    valid_only: bool = True,
    for_run: Optional[Union[str, int]] = os.getenv("RUN"),
) -> Optional[Any]:
    """Read most recent value entered into the database for a Task parameter.

    (Will be updated for schema compliance as well as Task name.)

    Args:
        db_dir (str): Database location.

        task_name (str): The name of the Task to check the database for.

        param (str): The parameter name for the Task that we want to retrieve.

        valid_only (bool): Whether to consider only valid results or not. E.g.
            An input file may be useful even if the Task result is invalid
            (Failed). Default = True.

        for_run (Optional[str | int]): Only consider latest entries from the
            specific experiment run provided.

    Returns:
        val (Any): The most recently entered value for `param` of `task_name`
            that can be found in the database. Returns None if nothing found.
    """
    import sqlite3
    from ._sqlite import select_param_from_db
    from ._sqlite_versioning import open_db

    db_path: str = f"{db_dir}/lute.db"
    con: sqlite3.Connection = open_db(db_path)
    with con:
        try:
            cond: Dict[str, str] = {}
            if valid_only:
                cond["valid_flag"] = "1"

            if for_run is not None:
                cond["run"] = str(for_run)

            new_param: str = param
            is_result: bool = False
            if "result." in param:
                new_param = param.split(".")[1]
                is_result = True

            return select_param_from_db(
                con=con,
                task_name=task_name,
                param_name=new_param,
                condition=cond,
                is_result=is_result,
            )
        except sqlite3.OperationalError as err:
            logger.error(f"Cannot retrieve value {param} due to: {err}")
            return None


def get_executions_summary(
    db_dir: str,
) -> List[Tuple[int, str, str, str, str, str, int]]:
    """Return some summary fields of all executions recorded.

    Args:
        db_dir (str): Database location.

    Returns:
        rows (List[Tuple[int, str, str, str, str, str, int]]): Returns a list
            of rows consisting of tuples with the following entries:
            (
                executions.id,
                executions.timestamp,
                tasks.name,
                results.summary,
                results.payload,
                results.summary,
                results.valid_flag,
            ).
            An example of how to manipulate this data is in `utilities/src/dbview.py`
    """
    import sqlite3
    from ._sqlite import executions_summary
    from ._sqlite_versioning import open_db

    db_path: str = f"{db_dir}/lute.db"
    con: sqlite3.Connection = open_db(db_path)
    with con:
        return executions_summary(con=con)


def get_task_parameters_summary(
    db_dir: str, task_name: str
) -> List[Tuple[int, str, int, str, str]]:
    """Return parameters for a specific task ordered by execution.

    Args:
        db_dir (str): Database location.

        task_name (str): Name of the Task to retrieve parameters for.

    Returns:
        rows (List[Tuple[int, str, str, str, str, str, int]]): Returns a list
            of rows consisting of tuples with the following entries:
            (
                executions.id,
                executions.timestamp,
                results.valid_flag,
                parameters.name,
                parameters.value,
            ).
            An example of how to manipulate this data is in `utilities/src/dbview.py`
    """
    import sqlite3
    from ._sqlite import task_parameters_summary
    from ._sqlite_versioning import open_db

    db_path: str = f"{db_dir}/lute.db"
    con: sqlite3.Connection = open_db(db_path)
    with con:
        return task_parameters_summary(con=con, task_name=task_name)


def get_task_parameters_defn_and_params(
    db_dir: str,
    row_ids: RowIds,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return a TaskParameters definition, and the param/value pairs.

    Args:
        db_dir (str): Database location.

        row_ids (RowIds): The ids in the various tables to reconstruct the
            TaskParameters definition and the various parameters.

    Returns:
        definition (Dict[str, Any]): The TaskParameters definition (JSON schema).

        param_values (Dict[str, Any]): The parameter: value dictionary.
    """
    import sqlite3
    from ._sqlite import get_task_parameters_definition_and_params
    from ._sqlite_versioning import open_db

    db_path: str = f"{db_dir}/lute.db"
    con: sqlite3.Connection = open_db(db_path)
    with con:
        rows: List[Tuple[str, str, str]] = get_task_parameters_definition_and_params(
            con=con, row_ids=row_ids
        )
        definition: Dict[str, Any] = {}
        param_values: Dict[str, Any] = {}
        for row in rows:
            if not definition:
                definition = json.loads(row[0])
            param_values[row[1]] = json.loads(row[2])

    return definition, param_values
