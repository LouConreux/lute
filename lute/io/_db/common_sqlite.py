"""SQLite utility functions that are independent of API version."""

from typing import List

__all__: List[str] = []
__author__ = "Gabriel Dorlhiac"

import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple


class DatabaseError(Exception):
    """General LUTE database error."""

    ...


def does_table_exist(con: sqlite3.Connection, table_name: str) -> bool:
    """Check whether a table exists.

    Args:
        con (sqlite3.Connection): Database connection.

        table_name (str): The table to check for.

    Returns:
        does_exist (bool): Whether the table exists.
    """
    res: sqlite3.Cursor = con.execute(
        f"SELECT name FROM sqlite_master WHERE name='{table_name}'"
    )
    if res.fetchone() is None:
        return False
    else:
        return True


def get_tables(con: sqlite3.Connection) -> List[str]:
    """Retrieve a list of all tables in a database.

    Args:
        con (sqlite3.Connection): Database connection.

    Returns:
        tables (List[str]): A list of database tables.
    """
    # sql: str = "SELECT name FROM sqlite_schema"
    sql: str = (
        "SELECT name FROM sqlite_schema "
        "WHERE type = 'table' "
        "AND name NOT LIKE 'sqlite_%'"
    )
    with con:
        res: sqlite3.Cursor = con.execute(sql)

    tables: List[str] = [table[0] for table in res.fetchall()]
    return tables


def get_table_cols(con: sqlite3.Connection, table_name: str) -> Dict[str, str]:
    """Retrieve the columns currently present in a table.

    Args:
        con (sqlite3.Connection): Database connection.

        table_name (str): The table's name.

    Returns:
        cols (Dict[str, str]): A dictionary of column names and types.
    """
    res: sqlite3.Cursor = con.execute(f"PRAGMA table_info({table_name})")
    # Retrieves: list = [col_id, col_name, col_type, -, default_val, -]
    table_info: List[Tuple[int, str, str, int, str, int]] = res.fetchall()

    cols: Dict[str, str] = {col[1]: col[2] for col in table_info}
    return cols


def get_all_rows_for_table(
    con: sqlite3.Connection, table_name: str
) -> List[Tuple[Any, ...]]:
    """Return all rows for a requested table.

    Args:
        con (sqlite3.Connection): Database connection.

        table_name (str): The table's name.

    Returns:
        rows (List[Tuple[Any, ...]]): ALL rows for a table.
    """
    sql: str = f'SELECT * FROM "{table_name}"'
    with con:
        res: sqlite3.Cursor = con.execute(sql)

    rows: List[Tuple[Any, ...]] = res.fetchall()
    return rows


def compare_cols(
    cols1: Dict[str, str], cols2: Dict[str, str]
) -> Optional[Dict[str, str]]:
    """Compare whether two sets of columns are identical.

    The comparison is unidirectional - This function tests for columns present
    in `cols2` which are not present in `cols1`, but NOT vice versa. Switch the
    order of the arguments in order to retrieve the other comparison.

    Args:
        cols1 (Dict[str, str]): Dictionary of first set of column names and
            types.

        cols2 (Dict[str, str]): Dictionary of second set of column names and
            types.

    Returns:
        diff (Dict[str, str] | None): Any columns present in `cols2` which
            are not present in `cols1`. If `cols2` has no entries which are
            not present in `cols1`, returns `None`.
    """
    diff: Dict[str, str] = {}

    for col_name in cols2.keys():
        if col_name not in cols1.keys():
            diff[col_name] = cols2[col_name]

    return diff if diff else None


# `PRAGMA user_version` of a database created by the v3 specification. Any later
# change within v3 increments it (301, 302, ...). v1/v2 never set it (0).
V3_BASE_USER_VERSION: int = 300


def get_user_version(con: sqlite3.Connection) -> int:
    """Retrieve the schema version stored in the database file.

    Args:
        con (sqlite3.Connection): Database connection.

    Returns:
        user_version (int): The value of `PRAGMA user_version`. 0 if never set.
    """
    return con.execute("PRAGMA user_version").fetchone()[0]


def set_user_version(con: sqlite3.Connection, user_version: int) -> None:
    """Set the schema version stored in the database file.

    This is transactional: it is undone if the enclosing transaction is rolled
    back.

    Args:
        con (sqlite3.Connection): Database connection.

        user_version (int): The new value of `PRAGMA user_version`.
    """
    # PRAGMA does not accept bound parameters
    con.execute(f"PRAGMA user_version = {int(user_version)}")


def execute_in_transaction(con: sqlite3.Connection, script: str) -> None:
    """Execute a multi-statement SQL script without committing.

    `sqlite3.Connection.executescript` always commits any pending transaction
    first, so it cannot be used inside a migration. This splits the script into
    complete statements (TRIGGER bodies included) and executes them one by one.

    Args:
        con (sqlite3.Connection): Database connection.

        script (str): The SQL statements to execute.

    Raises:
        DatabaseError: If the script ends with an incomplete statement.
    """
    statement: str = ""
    for line in script.splitlines(keepends=True):
        statement += line
        if sqlite3.complete_statement(statement):
            con.execute(statement)
            statement = ""
    if statement.strip():
        raise DatabaseError(f"Incomplete SQL statement: {statement.strip()}")


@contextmanager
def write_transaction(con: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Run a block of writes as one transaction.

    The write lock is taken at the start (`BEGIN IMMEDIATE`), so concurrent
    writers wait for each other instead of interleaving. The block is committed
    if it completes, and rolled back entirely on any exception, which is then
    re-raised.

    Args:
        con (sqlite3.Connection): Database connection, not in a transaction.

    Yields:
        con (sqlite3.Connection): The same connection.
    """
    con.execute("BEGIN IMMEDIATE")
    try:
        yield con
    except BaseException:
        con.rollback()
        raise
    con.commit()
