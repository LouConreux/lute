"""Backend SQLite database utilites.

Functions should be used only by the higher-level database module.

This contains the implementation for the v0.3 database specification.

None of these functions commit: the caller owns the transaction (see
`common_sqlite.write_transaction`), so that one execution is written entirely
or not at all.
"""

from typing import List

__all__: List[str] = []
__author__ = "Gabriel Dorlhiac"

import json
import logging
import sqlite3
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, overload

from lute.io._db.common_sqlite import (
    does_table_exist,
    execute_in_transaction,
    DatabaseError,
)

if TYPE_CHECKING:
    from lute.io.models.base import AnalysisHeader, TaskParameters
else:
    from lute.io.parameters import AnalysisHeader, TaskParameters
from lute.io.parameters import LUTE_PARAMETER_FIELD_ATTRS, RowIds
from lute.tasks.dataclasses import (
    BaseSchema,
    DescribedAnalysis,
    TaskResult,
    TaskStatus,
    VersionSpecifier,
)

if __debug__:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

logger: logging.Logger = logging.getLogger(__name__)

# Should be used on every connection
# The rollback journal is kept on disk and synced (DELETE, FULL), so a process
# killed, or a node failing, during a write leaves a journal which the next
# connection uses to roll back. With MEMORY/OFF, part of the uncommitted
# transaction could stay in the file.
PRAGMAS: str = """
PRAGMA journal_mode = DELETE;
PRAGMA synchronous = FULL;
PRAGMA foreign_keys = ON;
PRAGMA ignore_check_constraints = OFF;
PRAGMA auto_vacuum = NONE;
PRAGMA secure_delete = OFF;
"""


def _create_executions_table(con: sqlite3.Connection) -> None:
    """Setup the `executions` table.

    The `executions` table holds pointers to all tables to describe the execution
    each time a `Task` is run. A new row is added each time LUTE is run.

    This table DOES NOT contain constraints and breaks full normalization by allowing
    redundant entries.

    Args:
        con (sqlite3.Connection): A connection to the database.
    """
    query: str = """
    CREATE TABLE IF NOT EXISTS executions (
        id integer PRIMARY KEY,
        task_id INTEGER,
        task_version_id INTEGER,
        parameter_type_id INTEGER,
        executor_id INTEGER,
        config_id INTEGER,
        result_id INTEGER,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (task_id) REFERENCES tasks(id),
        FOREIGN KEY (task_version_id) REFERENCES task_version (id),
        FOREIGN KEY (parameter_type_id) REFERENCES parameter_types (id)
        FOREIGN KEY (executor_id) REFERENCES executors(id),
        FOREIGN KEY (config_id) REFERENCES config(id),
        FOREIGN KEY (result_id) REFERENCES results(id)
    );
    """
    con.execute(query)


def _create_config_table(con: sqlite3.Connection) -> None:
    """Setup the `config` table.

    The `config` table holds information from the AnalysisHeader which is often
    shared between many executions of many `Task`s.

    This table contains constraints:
    - The combination of all columns
      (title, experiment, run, date, lute_version, task_timeout) must be unique.
      Multiple executions can reference the same row of this table.

    Args:
        con (sqlite3.Connection): A connection to the database.
    """
    query: str = """
    CREATE TABLE IF NOT EXISTS config (
        id integer PRIMARY KEY,
        title TEXT,
        experiment TEXT,
        run TEXT,
        date TEXT,
        lute_version TEXT,
        task_timeout real,
        UNIQUE(title, experiment, run, date, lute_version, task_timeout)
    );
    """
    con.execute(query)


def _create_tasks_table(con: sqlite3.Connection) -> None:
    """Setup the `tasks` table.

    The `tasks` table holds the names of `Task`s.

    This table contains constraints:
    - The column `name` must be unique.

    Args:
        con (sqlite3.Connection): A connection to the database.
    """
    query: str = """
    CREATE TABLE IF NOT EXISTS tasks (
        id integer PRIMARY KEY,
        name TEXT UNIQUE
    );
    """
    con.execute(query)


def _create_parameters_table(con: sqlite3.Connection) -> None:
    """Setup the `parameters` table.

    The `parameters` table holds parameter/value combinations. It also contains
    an `execution_id` which references the specific row in the `executions` table
    which the name/value pair corresponds to. Additional Field metadata attributes
    are stored in the `param_meta` table, the rows of which are referenced by the
    `meta_id` column.

    This table DOES NOT contain constraints and breaks full normalization by allowing
    redundant entries.

    Args:
        con (sqlite3.Connection): A connection to the database.
    """
    query: str = """
    CREATE TABLE IF NOT EXISTS parameters (
        id integer PRIMARY KEY,
        execution_id INTEGER,
        meta_id INTEGER,
        name TEXT,
        value TEXT,
        FOREIGN KEY (execution_id) REFERENCES executions (id),
        FOREIGN KEY (meta_id) REFERENCES param_meta (id)
    );
    """
    con.execute(query)


def _create_parameter_types_table(con: sqlite3.Connection) -> None:
    """Setup the `parameter_types` table.

    The `parameter_types` table holds the schema information of the TaskParameters
    object associated to a specific execution in the `executions` table.

    This table contains constraints:
    - The `definition` column must be unique. The `type_name` column is allowed to
      be repeated to account for the possibility (although small) that a TaskParameters
      object is updated over the lifetime of the database.
    - The `definition` is required to be NOT NULL. All entries in the table must
      contain a definition.

    Note: Despite parameters themselves possibly having a complex type definitions,
          they do not reference entries in this table. The overall TaskParameters
          object schema contains the definitions of all parameters, so the individual
          parameters do not need to also reference this table.

    Args:
        con (sqlite3.Connection): A connection to the database.
    """
    query: str = """
    CREATE TABLE IF NOT EXISTS parameter_types (
        id integer PRIMARY KEY,
        type_name TEXT,
        definition TEXT NOT NULL UNIQUE
    );
    """
    con.execute(query)


def _create_param_meta_table(con: sqlite3.Connection) -> None:
    """Setup the `param_meta` table.

    The `param_meta` table holds meta-data attributes for the Field definitions
    of various parameters (that in turn comprise a TaskParameters object). These
    allow reconstruction of run-time objects post-validation within LUTE without
    re-validating a config file, or even having access to pydantic.

    This table contains constraints:
    - The combination of all columns
      (rename_param, flag_type, description, is_result) must be unique.
      Multiple parameters can reference the same row of this table.

    Note: The entries of this table may be NULL since parameters need not define
          these attributes.

    Args:
        con (sqlite3.Connection): A connection to the database.
    """
    query: str = """
    CREATE TABLE IF NOT EXISTS param_meta (
        id integer PRIMARY KEY,
        rename_param TEXT NULL,
        description TEXT NULL,
        flag_type TEXT NULL,
        is_result INTEGER NULL,
        UNIQUE(rename_param, description, flag_type, is_result)
    );
    """
    con.execute(query)


def _create_executors_table(con: sqlite3.Connection) -> None:
    """Setup the `executors` table.

    The `executors` table holds information about the `Executor`s used for various
    executions.

    This table contains constraints:
    - The combination of all columns (name, poll_interval, comm) must be unique.
      Multiple executions can reference the same row of this table.
    - The `comm` entry must be a BITWISE OR of the ids in the `communicators`
      table. This constraint is setup by the separate `_setup_triggers_and_indices`
      as TRIGGERs.

    Args:
        con (sqlite3.Connection): A connection to the database.
    """
    query: str = """
    CREATE TABLE IF NOT EXISTS executors (
        id integer PRIMARY KEY,
        name TEXT,
        poll_interval INTEGER,
        comm INTEGER,
        UNIQUE(name, poll_interval, comm)
    );
    """
    con.execute(query)


def _create_communicators_table(con: sqlite3.Connection) -> None:
    """Setup the `communicators` table.

    The `communicators` table holds information about the possible types of
    `Communicator`s that may be used by `Executors` (and `Task`s) for IPC.

    This table contains constraints:
    - The (name, description) combination must be unique. The name is allowed to
      be duplicate, because the same Communicator may have slightly different
      implementations which are reflected in the description column.
    - The `id` entry must be a power of 2 as the `schema` table uses a bitwise
      OR to indicate implementation of multiple `base_schema`.

    Note: Because the `id` must be a power of two, it must be inserted manually.

    Args:
        con (sqlite3.Connection): A connection to the database.
    """
    # id is a power of two
    query: str = """
    CREATE TABLE IF NOT EXISTS communicators (
        id integer PRIMARY KEY,
        name TEXT,
        description TEXT,
        CHECK (id == 0 OR (id > 0 AND (id & (id - 1)) = 0)),
        UNIQUE(name, description)
    );
    """
    con.execute(query)


def _create_environment_table(con: sqlite3.Connection) -> None:
    """Setup the `environment` table.

    The `environment` table holds the environment variable/value pairs associated
    to an execution. Each environment variable is entered into this table.

    This table DOES NOT contain constraints and breaks full normalization by allowing
    redundant entries.

    A unique index is setup using (execution_id, name) in the function
    `_setup_triggers_and_indices`.

    Args:
        con (sqlite3.Connection): A connection to the database.
    """
    query: str = """
    CREATE TABLE IF NOT EXISTS environment (
        id integer PRIMARY KEY,
        execution_id INTEGER,
        name TEXT,
        value TEXT,
        FOREIGN KEY (execution_id) REFERENCES executions (id)
    );
    """
    con.execute(query)


def _create_results_table(con: sqlite3.Connection) -> None:
    """Setup the `results` table.

    The `results` table holds information about the result of each execution.

    This table contains constraints:
    - The combination of all columns (schema_id, payload, summary, status, valid_flag)
      must be unique. Multiple executions can reference the same row of this table.

    Args:
        con (sqlite3.Connection): A connection to the database.
    """
    query: str = """
    CREATE TABLE IF NOT EXISTS results (
        id integer PRIMARY KEY,
        schema_id INTEGER,
        payload TEXT,
        summary TEXT,
        status INTEGER,
        valid_flag INTEGER,
        FOREIGN KEY (schema_id) REFERENCES schema (id)
        FOREIGN KEY (schema_id) REFERENCES schema (id),
        UNIQUE(schema_id, payload, summary, status, valid_flag)
    );
    """
    con.execute(query)


def _create_schema_table(con: sqlite3.Connection) -> None:
    """Setup the `schema` table.

    The `schema` table holds information about the `impl_schemas` that a Task
    may implement as part of its result.

    This table contains constraints:
    - The `schema` column must be unique. Multiple executions may reference the
      same entry in this table.
    - The `schema` entry must be a BITWISE OR of the ids in the `base_schema`
      table. This constraint is setup by the separate `_setup_triggers_and_indices`
      as TRIGGERs.

    Args:
        con (sqlite3.Connection): A connection to the database.
    """
    query: str = """
    CREATE TABLE IF NOT EXISTS schema (
        id INTEGER PRIMARY KEY,
        schema INTEGER UNIQUE
    );
    """
    con.execute(query)


def _create_base_schema_table(con: sqlite3.Connection) -> None:
    """Setup the `base_schema` table.

    The `base_schema` table holds information about the atomic schema that can
    be implemented as `impl_schema` by a `Task`. The total `schema` may be a
    combination of various entries in this table.

    This table contains constraints:
    - The name (and id) must be unique.
    - The `id` entry must be a power of 2 as the `schema` table uses a bitwise
      OR to indicate implementation of multiple `base_schema`.

    This function will also insert all the base_schema already defined in
    `lute.tasks.dataclasses.BaseSchema`.

    Note: Because the `id` must be a power of two, it must be inserted manually.

    Args:
        con (sqlite3.Connection): A connection to the database.
    """
    create_query: str = """
    CREATE TABLE IF NOT EXISTS base_schema (
        id INTEGER PRIMARY KEY,
        name TEXT UNIQUE,
        CHECK (id == 0 OR (id > 0 AND (id & (id - 1)) = 0))
    );
    """

    # In addition to creation, we will add base_schema that we already know about
    insert_query: str = "INSERT OR IGNORE INTO base_schema (id, name) VALUES (?, ?)"

    con.execute(create_query)
    for bs in BaseSchema:
        con.execute(insert_query, (bs.value, bs.name))


def _create_task_version_table(con: sqlite3.Connection) -> None:
    """Setup the `task_version` table.

    The `task_version` table holds information about the specific version of the
    underlying `Task` code that was run for that specific `Task`.

    This table contains constraints:
    - The `version_specifier` entry must be a BITWISE OR of the ids in the
      `version_specifiers` table. This constraint is setup by the separate
      `_setup_triggers_and_indices` as TRIGGERs.
    - The `version_info` is required to be NOT NULL. This field contains a JSON blob
      representing the version data, as defined by the `version_specifiers` field.

    Args:
        con (sqlite3.Connection): A connection to the database.
    """
    query: str = """
    CREATE TABLE IF NOT EXISTS task_version (
        id INTEGER PRIMARY KEY,
        version_specifier INTEGER,
        version_info TEXT NOT NULL UNIQUE
    );
    """
    con.execute(query)


def _create_version_specifiers_table(con: sqlite3.Connection) -> None:
    """Setup the `version_specifiers` table.

    The `version_specifiers` table holds information about how to understand the
    version information for an execution. The enumerators defined here define a
    single concrete access mechanism that is used for interpretation. E.g.,
    a combination of commit sha256 and a diff may be used for a versioning a
    Task that is run from a cloned git repository.

    This table contains constraints:
    - The name (and id) must be unique.
    - The `id` entry must be a power of 2 as the `schema` table uses a bitwise
      OR to indicate implementation of multiple `base_schema`.

    This function will also insert all the base_schema already defined in
    `lute.tasks.dataclasses.BaseSchema`.

    Note: Because the `id` must be a power of two, it must be inserted manually.

    Args:
        con (sqlite3.Connection): A connection to the database.
    """
    create_query: str = """
    CREATE TABLE IF NOT EXISTS version_specifiers (
        id INTEGER PRIMARY KEY,
        name TEXT UNIQUE,
        CHECK (id == 0 OR (id > 0 AND (id & (id - 1)) = 0))
    );
    """

    # In addition to creation, we will add version specifiers that we already know about
    insert_query: str = (
        "INSERT OR IGNORE INTO version_specifiers (id, name) VALUES (?, ?)"
    )

    con.execute(create_query)
    for vs in VersionSpecifier:
        con.execute(insert_query, (vs.value, vs.name))


def _setup_triggers_and_indices(con: sqlite3.Connection) -> None:
    """Setup the trigger constraints and unique indices.

    This function sets up some additional triggers to implement constraints:
    - Enforcing that the `executors.comm` column is a BITWISE OR of the entries
      in the `communicators.id` column.
    - Enforcing that the `schema.schema` column is a BITWISE OR of the entries
      in the `base_schema.id` column.

    It also sets up a unique index on the `environment` table:
    - index = (execution_id, name)

    Args:
        con (sqlite3.Connection): A connection to the database.
    """
    query: str = """
    CREATE UNIQUE INDEX IF NOT EXISTS environment_index_0 ON environment (execution_id, name);

    CREATE TRIGGER IF NOT EXISTS check_schema_update
    BEFORE UPDATE ON schema
    BEGIN
        SELECT
        CASE
            WHEN NEW.schema > (SELECT SUM(id) FROM base_schema)
            THEN RAISE(ABORT, 'schema value exceeds total base_schema sum')
        END;
    END;

    CREATE TRIGGER IF NOT EXISTS check_schema_insert
    BEFORE INSERT ON schema
    BEGIN
        SELECT
        CASE
            WHEN NEW.schema > (SELECT SUM(id) FROM base_schema)
            THEN RAISE(ABORT, 'schema value exceeds total base_schema sum')
        END;
    END;

    CREATE TRIGGER IF NOT EXISTS check_version_update
    BEFORE UPDATE ON task_version
    BEGIN
        SELECT
        CASE
            WHEN NEW.version_specifier > (SELECT SUM(id) FROM version_specifiers)
            THEN RAISE(ABORT, 'version_specifier value exceeds total version_specifiers sum')
        END;
    END;

    CREATE TRIGGER IF NOT EXISTS check_version_insert
    BEFORE INSERT ON task_version
    BEGIN
        SELECT
        CASE
            WHEN NEW.version_specifier > (SELECT SUM(id) FROM version_specifiers)
            THEN RAISE(ABORT, 'version_specifier value exceeds total version_specifiers sum')
        END;
    END;

    CREATE TRIGGER IF NOT EXISTS check_executors_update
    BEFORE UPDATE ON executors
    BEGIN
        SELECT
        CASE
            WHEN NEW.comm > (SELECT SUM(id) FROM communicators)
            THEN RAISE(ABORT, 'comm value exceeds total communicators sum')
        END;
    END;

    CREATE TRIGGER IF NOT EXISTS check_executors_insert
    BEFORE INSERT ON executors
    BEGIN
        SELECT
        CASE
            WHEN NEW.comm > (SELECT SUM(id) FROM communicators)
            THEN RAISE(ABORT, 'comm value exceeds total communicators sum')
        END;
    END;
    """
    execute_in_transaction(con, query)


def create_tables(con: sqlite3.Connection) -> None:
    """Setup the full database.

    See individual table function calls for more information

    This function does not commit. It is run as a migration by `open_db`, which
    owns the transaction, so none of the table creation functions may use
    `with con:` or `con.executescript` (both commit).

    Args:
        con (sqlite3.Connection): A connection to the database.
    """
    if does_table_exist(con, "executions"):
        return None
    else:
        _create_tasks_table(con)
        _create_parameter_types_table(con)

        _create_version_specifiers_table(con)
        _create_task_version_table(con)

        _create_communicators_table(con)
        _create_executors_table(con)
        _create_config_table(con)

        _create_base_schema_table(con)
        _create_schema_table(con)
        _create_results_table(con)

        _create_executions_table(con)

        _create_param_meta_table(con)
        _create_parameters_table(con)

        _create_environment_table(con)

        _setup_triggers_and_indices(con)


def _insert_maybe_ignore_return_id(
    con: sqlite3.Connection, table: str, entries: Dict[str, Any], ignore: bool = False
) -> int:
    """Insert entries into a table.

    This function will try to insert a new row into the requested table. It may
    ignore (depending on the value of the `ignore` argument) errors due to table
    constraints.

    After insertion (or silent failure if `ignore == True`), the function will then
    query to find the `id` matching the newly inserted row, or the pre-existing
    row that matches the entries provided.

    Args:
        con (sqlite3.Connection): A connection to the database.

        table (str): The database table to insert into.

        entries (dict[str, Any]): A dictionary consisting of `COLUMN_NAME: VALUE`
            key/value pairs to be inserted into `table`.

        ignore (bool): If `True`, ignore errors arising from database constraints.
            Use this option if you frequently want to try to insert new rows into
            tables that have uniqueness constraints which will result in failure.

    Returns:
        id (int): The id of the row matching `entries`. This will be either newly
            inserted, or pre-existing.
    """
    keys = list(entries.keys())
    keys_clause: str = f'({",".join(keys)})'
    values_clause: str = f'({",".join(f":{key}" for key in keys)})'

    match_clause: str = " AND ".join(
        f'"{key}" = :{key}' if val is not None else f'"{key}" IS NULL'
        for key, val in entries.items()
    )
    insert_query: str
    if ignore:
        insert_query = (
            f'INSERT OR IGNORE INTO "{table}" {keys_clause} VALUES {values_clause}'
        )
    else:
        insert_query = f'INSERT INTO "{table}" {keys_clause} VALUES {values_clause}'

    select_query: str = (
        f'SELECT id FROM "{table}" WHERE {match_clause} ORDER BY id DESC LIMIT 1'
    )

    # Insert or ignore
    con.execute(insert_query, entries)
    # Get the row
    cur: sqlite3.Cursor = con.execute(select_query, entries)
    row: Optional[Tuple[int]] = cur.fetchone()

    if row:
        if table == "executions":
            logger.debug(f"Selecting {table} row: {row[0]}")
        return row[0]
    else:
        raise DatabaseError(f"Failed to insert or find a row for {table}.")


def _insert_pow2_id_return_id(
    con: sqlite3.Connection, table: str, entries: Dict[str, Any]
) -> int:
    """Insert entries into a table with a power of 2 constraint on the id.

    This function will see if the entries already exist in the table. If not it
    will calculate the next power of 2 id, and insert them.

    Args:
        con (sqlite3.Connection): A connection to the database.

        table (str): The database table to insert into.

        entries (dict[str, Any]): A dictionary consisting of `COLUMN_NAME: VALUE`
            key/value pairs to be inserted into `table`.

    Returns:
        id (int): The id of the row matching `entries`. This will be either newly
            inserted, or pre-existing.
    """
    keys: List[str] = list(entries.keys())

    match_clause: str = " AND ".join(f'"{key}" = :{key}' for key in keys)
    select_query: str = f'SELECT id FROM "{table}" WHERE {match_clause}'

    select_max_id_query: str = f"SELECT MAX(id) FROM {table}"

    # Try to get the row
    cur: sqlite3.Cursor = con.execute(select_query, entries)
    row: Optional[Tuple[int]] = cur.fetchone()
    if row:
        # Entry already exists, so just return it
        return row[0]
    else:
        # Otherwise, get last row_id and enter new row with next power of 2 id
        cur = con.execute(select_max_id_query)
        row = cur.fetchone()
        if row:
            max_id: Optional[int] = row[0]
            new_id: int
            if max_id is None:
                # This is the very first entry
                new_id = 1
            else:
                new_id = max_id * 2
            keys = list(entries.keys())
            # Add id to keys/entries
            keys.append("id")
            entries["id"] = new_id
            # Reconstruct query
            keys_clause: str = f'({",".join(keys)})'
            values_clause: str = f'({",".join(f":{key}" for key in keys)})'
            insert_query: str = (
                f'INSERT INTO "{table}" {keys_clause} VALUES {values_clause}'
            )
            con.execute(insert_query, entries)
            # Assuming successful, we don't need to query again since we specified id
            # Will throw error if there is some other issue
            return new_id
        else:
            raise DatabaseError(
                f"Failed to insert or find a row (with ID = power of 2) for {table}."
            )


def _update_row(
    con: sqlite3.Connection, table: str, entries: Dict[str, Any], row_id: int
) -> None:
    """Update entries in a row of a table, given the row_id.

    Args:
        con (sqlite3.Connection): A connection to the database.

        table (str): The database table to insert into.

        entries (dict[str, Any]): A dictionary consisting of `COLUMN_NAME: VALUE`
            key/value pairs to be inserted into `table`.

        row_id (int): Row to update.
    """
    set_clause: str = ", ".join(
        f'"{key}" = :{key}' if val is not None else f'"{key}" IS NULL'
        for key, val in entries.items()
    )
    where_clause: str = "id = :id"
    update_query: str = f'UPDATE "{table}" SET {set_clause} WHERE {where_clause}'
    entries["id"] = row_id

    con.execute(update_query, entries)


@overload
def _add_parameters(
    con: sqlite3.Connection,
    params: TaskParameters,
    execution_id: int,
    return_ids: bool = False,
) -> None: ...


@overload
def _add_parameters(
    con: sqlite3.Connection,
    params: TaskParameters,
    execution_id=None,
    return_ids: bool = True,
) -> List[int]: ...


def _add_parameters(
    con: sqlite3.Connection,
    params: TaskParameters,
    execution_id: Optional[int] = None,
    return_ids: bool = False,
) -> Optional[List[int]]:
    """Add all parameters into the `parameters` table for `execution_id`.

    Args:
        con (sqlite3.Connection): A connection to the database.

        params (TaskParameters): The TaskParameters object used for the execution.

        execution_id (Optional[int]): The row id in the `executions` table associated
            to this set of parameters. It may be None if preparing a placeholder.

        return_ids (bool): If True, return the set of ids for the rows inserted.
    """
    schema: Dict[str, Any] = params.schema()
    param_dict: Dict[str, Any] = params.dict()
    row_ids: List[int] = []
    # Note that `Config` is added to the schema, but doesn't appear in params.dict()
    # This is intentional: It can be retrieved via schema validation, but as an
    # internal set of options it doesn't reflect a "true" parameter
    for param in param_dict:
        try:
            props: Dict[str, Any] = schema["properties"][param]
        except KeyError:
            # It may not be defined at top level if nested
            for _, defn in schema["definitions"].items():
                if param in defn["properties"]:
                    props = defn["properties"][param]
                    break
            else:
                logger.critical(
                    f"Unable to parse parameter {param} properly! Metadata will be incorrect!"
                )
                props = {}
        param_meta_entries: Dict[str, Any] = {
            key: props[key] if key in props else None
            for key in LUTE_PARAMETER_FIELD_ATTRS
        }
        param_meta_id: int = _insert_maybe_ignore_return_id(
            con=con, table="param_meta", entries=param_meta_entries, ignore=True
        )
        raw_val: Any = param_dict[param]
        json_val: str
        # Type checking may not work because of the parameters hack to support different
        # environments -- so check the class name... not great...
        if raw_val.__class__.__name__ == "TemplateParameters":
            if hasattr(raw_val.params, "dict"):
                json_val = json.dumps(raw_val.params.dict(), default=list)
            else:
                json_val = json.dumps(raw_val.params, default=list)
        elif hasattr(raw_val, "dict"):
            json_val = json.dumps(raw_val.dict(), default=list)
        else:
            json_val = json.dumps(raw_val, default=list)
        param_entries: Dict[str, Any] = {
            "execution_id": execution_id,
            "meta_id": param_meta_id,
            "name": param,
            "value": json_val,
        }
        # We should allow redundant entries, so don't ignore.
        # If we get a constraint-based error, something has gone wrong.
        param_id: int = _insert_maybe_ignore_return_id(
            con=con, table="parameters", entries=param_entries, ignore=False
        )
        if return_ids:
            row_ids.append(param_id)

    if return_ids:
        return row_ids
    return None


def _update_parameters(
    con: sqlite3.Connection, param_ids: List[int], execution_id: int
) -> None:
    for param_id in param_ids:
        param_entries: Dict[str, Any] = {
            "execution_id": execution_id,
        }
        _update_row(con, table="parameters", entries=param_entries, row_id=param_id)


def _add_env_vars(
    con: sqlite3.Connection, env_vars: Dict[str, str], execution_id: int
) -> None:
    """Add all environment variables into the `environment` table for `execution_id`.

    Args:
        con (sqlite3.Connection): A connection to the database.

        env_vars (Dict[str, str]): Dictionary of environment variable/value pairs.

        execution_id (int): The row id in the `executions` table associated to this
            set of parameters.
    """
    selected_env_vars: Dict[str, str] = {
        key: env_vars[key]
        for key in env_vars
        if ("LUTE_" in key and "_TENV_" not in key) or "SLURM_" in key
    }
    for key in selected_env_vars:
        entries: Dict[str, Any] = {
            "execution_id": execution_id,
            "name": key,
            "value": selected_env_vars[key],
        }
        # We should allow redundant entries, so don't ignore.
        # If we get a constraint-based error, something has gone wrong.
        # We don't care about the returned id in this case.
        _insert_maybe_ignore_return_id(
            con=con, table="environment", entries=entries, ignore=False
        )

    return None


def _add_executor_and_communicators(
    con: sqlite3.Connection, cfg: DescribedAnalysis
) -> int:
    entries: Dict[str, Any]
    # Need to do communicators
    full_comm_id: int = 0
    for comm in cfg.communicator_desc:
        # communicator_desc is a list[str]
        # each entry is in format: `name: description`
        name: str
        desc: str
        name, desc = comm.split(":")
        desc = desc[1:]  # Remove space
        # id is required, but it will be added to entries by the function being called
        # so we don't include it here.
        entries = {
            # "id": 123,
            "name": name,
            "description": desc,
        }
        full_comm_id += _insert_pow2_id_return_id(
            con=con, table="communicators", entries=entries
        )

    # Need to update to pass name
    entries = {
        "name": cfg.executor_name,
        "poll_interval": cfg.poll_interval,
        "comm": full_comm_id,
    }
    executor_id: int = _insert_maybe_ignore_return_id(
        con=con, table="executors", entries=entries, ignore=True
    )
    return executor_id


def _add_schema(con: sqlite3.Connection, impl_schemas: Optional[str]) -> int:
    # Setup schema first
    # For now we assume all base_schema were inserted during creation -
    # see _create_base_schema_table
    # Can extend later to add new schema as needed. Here we just calculate
    # the appropriate bitwise OR to add to the actual `schema` table
    combined_schema_val: int = 0
    if impl_schemas is not None:
        for bs_str in impl_schemas.split(";"):
            if bs_str in BaseSchema.__members__:
                combined_schema_val += BaseSchema.__members__[bs_str].value

    entries: Dict[str, int] = {"schema": combined_schema_val}
    schema_id: int = _insert_maybe_ignore_return_id(
        con=con, table="schema", entries=entries, ignore=True
    )
    return schema_id


def _add_version(
    con: sqlite3.Connection,
    version_specifier: Optional[int] = None,
    task_version: Optional[str] = None,
) -> int:
    # Setup schema first
    # For now we assume all version_specifiers were inserted during creation -
    # see _create_version_specifiers_table
    # We can still check that the version specifier matches possible values,
    # it must be a bitwise OR of the possible individual specifier enumerators
    # TODO: In the future, could support inserting new kinds of specifiers on the
    #       fly, but probably want a way of attaching descriptive information to the
    #       possibilities.
    enum_val: int = int(list(VersionSpecifier.__members__.values())[-1])
    enum_val_sum: int = 0
    while enum_val:
        enum_val_sum += enum_val
        enum_val >>= 1
    if version_specifier is not None and version_specifier > enum_val_sum:
        # Want hard error? Or just warning?
        logger.warning(
            f"Combination version specifier value of {version_specifier} is invalid! "
            f"The max expected value (combo of all known enumerators) is {enum_val_sum}. "
            "We will continue. But the version information will be unusable upon retrieval!"
        )

    entries: Dict[str, Any] = {
        "version_specifier": version_specifier if version_specifier else 0,
        "version_info": task_version if task_version else "",
    }
    task_version_id: int = _insert_maybe_ignore_return_id(
        con=con, table="task_version", entries=entries, ignore=True
    )
    return task_version_id


def add_execution(con: sqlite3.Connection, cfg: DescribedAnalysis) -> None:
    """Write an DescribedAnalysis object to the database.

    This will unpack all the values from the object and distribute the entries
    across all the various tables.

    Args:
        con (sqlite3.Connection): A connection to the database.

        cfg (DescribedAnalysis): The DescribedAnalysis completed by the Executor
            after Task completion.
    """
    if cfg.task_parameters is None:
        return
    entries: Dict[str, Any] = {}
    # Task
    entries["name"] = cfg.task_result.task_name
    task_id: int = _insert_maybe_ignore_return_id(
        con=con, table="tasks", entries=entries, ignore=True
    )
    del cfg.task_result.task_name
    entries.clear()

    entries.clear()

    # Config
    lute_config: AnalysisHeader = cfg.task_parameters.lute_config
    del cfg.task_parameters.lute_config
    config_id: int = _insert_maybe_ignore_return_id(
        con=con, table="config", entries=lute_config.dict(), ignore=True
    )

    # Task Version information
    version_specifier: Optional[int] = cfg.task_parameters.Config.version_specifier
    version_info: Optional[str] = cfg.task_parameters.Config.task_version

    # Some Tasks rely on tasklets to setup the code-base
    # In those cases, the version recording at parameter validation would not work
    # since the Executor would not have run the tasklet yet
    if version_specifier is not None and version_specifier > 0:
        if version_info is None or version_info == "" or version_info == "{}":
            from lute.io.version_utils import record_version

            location: Optional[str] = getattr(
                cfg.task_parameters.Config, "version_location", None
            )
            diff_args: Optional[List[str]] = getattr(
                cfg.task_parameters.Config, "version_diff_args", None
            )

            new_version_info: Dict[str, str] = record_version(
                version_specifier=version_specifier,
                location=location,
                diff_args=diff_args,
            )

            # Store where version information was taken from
            new_version_info["version-location"] = json.dumps(location)
            new_version_info["version-diff-args"] = json.dumps(diff_args)

            if new_version_info:
                version_info = json.dumps(new_version_info)

    task_version_id: int = _add_version(
        con=con, version_specifier=version_specifier, task_version=version_info
    )
    del cfg.task_parameters.Config.task_version
    del cfg.task_parameters.Config.version_specifier

    schema_id: int = _add_schema(con=con, impl_schemas=cfg.task_result.impl_schemas)

    # Prepare flag to indicate whether the task entry is valid or not
    # By default we say it is assuming proper completion
    valid_flag: int = 1 if cfg.task_result.task_status == TaskStatus.COMPLETED else 0

    # Results
    task_result: TaskResult = cfg.task_result
    entries = {
        "schema_id": schema_id,
        "payload": task_result.payload,
        "summary": task_result.summary,
        "status": str(task_result.task_status),
        "valid_flag": valid_flag,
    }
    result_id: int = _insert_maybe_ignore_return_id(
        con=con, table="results", entries=entries, ignore=True
    )
    entries.clear()

    executor_id: int = _add_executor_and_communicators(con=con, cfg=cfg)

    # Include the parameter type definition.
    ## Have sets in the schema so we will convert those with `default=list`
    entries = {
        "type_name": cfg.task_parameters.__class__.__name__,
        "definition": json.dumps(cfg.task_parameters.schema(), default=list),
    }
    parameter_type_id: int = _insert_maybe_ignore_return_id(
        con=con, table="parameter_types", entries=entries, ignore=True
    )
    entries.clear()

    entries = {
        "task_id": task_id,
        "task_version_id": task_version_id,
        "parameter_type_id": parameter_type_id,
        "executor_id": executor_id,
        "config_id": config_id,
        "result_id": result_id,
    }
    execution_id: int = _insert_maybe_ignore_return_id(
        con=con, table="executions", entries=entries, ignore=False
    )

    # Add individual param/values and their param_meta
    # NOTE: `param_meta` is probably redundant at the moment, since this may all
    #       be included in the parameter_types.definition column.
    #       We will maintain the table in case it is needed however.
    _add_parameters(con=con, params=cfg.task_parameters, execution_id=execution_id)

    # Add all environment variable/values
    env: Dict[str, str] = cfg.task_env
    _add_env_vars(con=con, env_vars=env, execution_id=execution_id)
    del cfg.task_env

    return None


def update_execution(
    con: sqlite3.Connection, cfg: DescribedAnalysis, row_ids: RowIds
) -> None:
    """Update all entries associated with an Execution.

    This is intended to be called after completion of Task execution, when the
    Task has previously populated a subset of required entries.

    Args:
        con (sqlite3.Connection): A connection to the database.

        cfg (DescribedAnalysis): The DescribedAnalysis completed by the Executor
            after Task completion.

        row_ids (RowIds): A dictionary containing the entries previously added by the
            Task.
    """
    if cfg.task_parameters is None:
        return
    entries: Dict[str, Any] = {}

    executor_id: int = _add_executor_and_communicators(con=con, cfg=cfg)

    # Prepare flag to indicate whether the task entry is valid or not
    # By default we say it is assuming proper completion
    valid_flag: int = 1 if cfg.task_result.task_status == TaskStatus.COMPLETED else 0

    schema_id: int = _add_schema(con=con, impl_schemas=cfg.task_result.impl_schemas)

    # Results
    task_result: TaskResult = cfg.task_result
    entries = {
        "schema_id": schema_id,
        "payload": task_result.payload,
        "summary": task_result.summary,
        "status": str(task_result.task_status),
        "valid_flag": valid_flag,
    }
    result_id: int = _insert_maybe_ignore_return_id(
        con=con, table="results", entries=entries, ignore=True
    )
    entries.clear()

    # Some Tasks rely on tasklets to setup the code-base
    # In those cases, the version recording at parameter validation would not work
    # since the Executor would not have run the tasklet yet
    version_specifier: Optional[int] = cfg.task_parameters.Config.version_specifier
    task_version: Optional[str] = cfg.task_parameters.Config.task_version
    task_version_id: int = row_ids["task_version_id"]

    if version_specifier is not None and version_specifier > 0:
        if task_version is None or task_version == "" or task_version == "{}":
            from lute.io.version_utils import record_version

            location: Optional[str] = getattr(
                cfg.task_parameters.Config, "version_location", None
            )
            diff_args: Optional[List[str]] = getattr(
                cfg.task_parameters.Config, "version_diff_args", None
            )

            new_version_info: Dict[str, str] = record_version(
                version_specifier=version_specifier,
                location=location,
                diff_args=diff_args,
            )

            # Store where version information was taken from
            new_version_info["version-location"] = json.dumps(location)
            new_version_info["version-diff-args"] = json.dumps(diff_args)

            if new_version_info:
                task_version_id = _add_version(
                    con=con,
                    version_specifier=version_specifier,
                    task_version=json.dumps(new_version_info),
                )

    entries = {
        "task_id": row_ids["task_id"],
        "task_version_id": task_version_id,
        "parameter_type_id": row_ids["parameter_type_id"],
        "executor_id": executor_id,
        "config_id": row_ids["config_id"],
        "result_id": result_id,  # row_ids["result_id"],
    }
    execution_id: int = _insert_maybe_ignore_return_id(
        con=con, table="executions", entries=entries, ignore=False
    )

    _update_parameters(
        con=con, param_ids=row_ids["parameter_ids"], execution_id=execution_id
    )

    # Add all environment variable/values
    env: Dict[str, str] = cfg.task_env
    _add_env_vars(con=con, env_vars=env, execution_id=execution_id)
    del cfg.task_env

    return None


def add_placeholder_execution(
    con: sqlite3.Connection, params: TaskParameters
) -> Optional[RowIds]:
    """Write a TaskParameters object to the database.

    This will unpack all the values from the object and distribute the entries
    across all the various tables.

    This is intended to be a placeholder, written by the Task layer. The execution
    layer should then call `update_execution` to fill in the rest of the information.

    Args:
        con (sqlite3.Connection): A connection to the database.

        params (TaskParameters): The TaskParameters object.

    Returns:
        row_ids (RowIds): The set of row ids for all entries that were inserted.
    """
    if params is None:
        return None
    entries: Dict[str, Any] = {}
    # Task
    entries["name"] = params.__class__.__name__[:-10]
    task_id: int = _insert_maybe_ignore_return_id(
        con=con, table="tasks", entries=entries, ignore=True
    )
    entries.clear()

    # Config
    lute_config: AnalysisHeader = params.lute_config
    config_id: int = _insert_maybe_ignore_return_id(
        con=con, table="config", entries=lute_config.dict(), ignore=True
    )

    # Task Version information
    version_specifier: Optional[int] = params.Config.version_specifier
    version_info: Optional[str] = params.Config.task_version
    task_version_id: int = _add_version(
        con=con, version_specifier=version_specifier, task_version=version_info
    )

    # Include the parameter type definition.
    ## Have sets in the schema so we will convert those with `default=list`
    entries = {
        "type_name": params.__class__.__name__,
        "definition": json.dumps(params.schema(), default=list),
    }
    parameter_type_id: int = _insert_maybe_ignore_return_id(
        con=con, table="parameter_types", entries=entries, ignore=True
    )
    entries.clear()

    # Add individual param/values and their param_meta
    # NOTE: `param_meta` is probably redundant at the moment, since this may all
    #       be included in the parameter_types.definition column.
    #       We will maintain the table in case it is needed however.
    parameter_ids: List[int] = _add_parameters(con=con, params=params, return_ids=True)

    row_ids: RowIds = {
        "task_id": task_id,
        "task_version_id": task_version_id,
        "parameter_type_id": parameter_type_id,
        "config_id": config_id,
        "parameter_ids": parameter_ids,
    }

    return row_ids


def select_param_from_db(
    con: sqlite3.Connection,
    task_name: str,
    param_name: str,
    condition: Dict[str, str],
    is_result: bool = False,
) -> Optional[Any]:
    """Retrieve a specific value for a parameter subject to conditions.

    If multiple parameter/value pairs match the provided conditions, this function
    will return the latest entry.

    Args:
        con (sqlite3.Connection): A connection to the database.

        task_name (str): The `Task` of interest.

        param_name (str): The parameter for that `Task`. Or the result field, if
            `is_result` is True.

        condition (Dict[str, str]): A dictionary of conditions. Currently supports:
            - valid_flag: 1/0 # Only include "valid" results.
            - run: XYZ # Only look at entries from this run. Otherwise, take the latest.

        is_result (bool): If True, select the field from the results table instead
            of the parameters table. E.g. `param_name = payload, is_result=True`
            selects `payload` from `results`.

    Returns:
        value (Optional[Any]): The retrieved value from the `parameters` table or
            None if nothing is found (or potentially if the value is None). Values
            are stored serialized as json. This function deserializes and returns
            the Python object.
    """
    safe_condition: Dict[str, str] = dict(condition)

    safe_condition["param"] = param_name
    safe_condition["task"] = task_name

    formatted_conditions: List[str] = []
    for key in safe_condition.keys():
        if key == "valid_flag":
            formatted_conditions.append("r.valid_flag = :valid_flag")
        elif key == "run":
            formatted_conditions.append("c.run = :run")
        elif key == "param":
            if not is_result:
                formatted_conditions.append("p.name = :param")
        elif key == "task":
            formatted_conditions.append("t.name = :task")

    where_clause: str = (
        f"WHERE {' AND '.join(formatted_conditions)}" if formatted_conditions else ""
    )

    join_query: str
    if is_result:
        result_keys: Tuple[str, ...] = (
            "schema_id",
            "payload",
            "summary",
            "status",
            "valid_flag",
        )
        if param_name not in result_keys:
            raise DatabaseError(f"Cannot search results table for {param_name}!")

        join_query = f"""
        SELECT r.{param_name}
        FROM results r
        JOIN executions e ON e.result_id = r.id
        JOIN config c ON e.config_id = c.id
        JOIN tasks t ON e.task_id = t.id
        {where_clause}
        ORDER BY e.timestamp DESC
        LIMIT 1
        """
    else:
        join_query = f"""
        SELECT p.value
        FROM parameters p
        JOIN executions e ON p.execution_id = e.id
        JOIN config c ON e.config_id = c.id
        JOIN results r ON e.result_id = r.id
        JOIN tasks t ON e.task_id = t.id
        {where_clause}
        ORDER BY e.timestamp DESC
        LIMIT 1
        """

    cur: sqlite3.Cursor = con.execute(join_query, safe_condition)
    row: Any = cur.fetchone()

    if not row:
        return None

    try:
        return json.loads(row[0])
    except (TypeError, json.JSONDecodeError):
        # For straight strings just return. Hopefully only strings
        return row[0]


def executions_summary(
    con: sqlite3.Connection,
) -> List[Tuple[int, str, str, str, str, str, int]]:
    """Return some summary fields of all executions recorded.

    Args:
        con (sqlite3.Connection): A connection to the database.

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
    join_query: str = """
        SELECT e.id, e.timestamp, t.name, r.summary, r.payload, r.status, r.valid_flag
        FROM parameters p
        JOIN executions e ON p.execution_id = e.id
        JOIN config c ON e.config_id = c.id
        JOIN results r ON e.result_id = r.id
        JOIN tasks t ON e.task_id = t.id
        ORDER BY e.timestamp ASC
    """
    with con:
        cur: sqlite3.Cursor = con.execute(join_query)
        rows: List[Tuple[int, str, str, str, str, str, int]] = cur.fetchall()
        return rows


def task_parameters_summary(
    con: sqlite3.Connection, task_name: str
) -> List[Tuple[int, str, int, str, str]]:
    """Return parameters for a specific task ordered by execution.

    Args:
        con (sqlite3.Connection): A connection to the database.

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
    join_query: str = """
        SELECT e.id, e.timestamp, r.valid_flag, p.name, p.value
        FROM parameters p
        JOIN executions e ON p.execution_id = e.id
        JOIN config c ON e.config_id = c.id
        JOIN results r ON e.result_id = r.id
        JOIN tasks t ON e.task_id = t.id
        WHERE t.name = ?
        ORDER BY e.timestamp ASC
    """
    with con:
        cur: sqlite3.Cursor = con.execute(join_query, (task_name,))
        rows: List[Tuple[int, str, int, str, str]] = cur.fetchall()
        return rows


def get_task_parameters_definition_and_params(
    con: sqlite3.Connection, row_ids: RowIds
) -> List[Tuple[str, str, str]]:
    """Return parameters and their definition, given row IDs.

    Args:
        con (sqlite3.Connection): A connection to the database.

        row_ids (RowIds): The selection of RowIds to find the requested definition
            and parameter names/values.

    Returns:
        rows (List[Tuple[str, str, str]]): Returns a list
            of rows consisting of tuples with the following entries:
            (
                parameter_types.definition,
                parameters.name,
                parameters.value,
            ).
            NOTE: The paremeter_types.definition will be repeated in each row, but
                  is identical.
    """

    parameters_id_in_clause: str = ", ".join("?" for _ in row_ids["parameter_ids"])
    parameters_id_in_clause = f"({parameters_id_in_clause})"

    join_query: str = f"""
        SELECT types.definition, p.name, p.value
        FROM parameters p
        JOIN parameter_types types ON types.id = ?
        WHERE p.id IN {parameters_id_in_clause}
    """
    with con:
        substitutions: List[int] = [row_ids["parameter_type_id"]]
        substitutions.extend(row_ids["parameter_ids"])
        cur: sqlite3.Cursor = con.execute(join_query, substitutions)
        rows: List[Tuple[str, str, str]] = cur.fetchall()
        return rows
