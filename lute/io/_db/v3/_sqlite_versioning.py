"""Schema versioning and migrations for the v3 database specification (SQLite).

This is part of the SQLite backend: like `_sqlite`, it is only imported inside
the functions of `api`, which stays independent of the backend.

The schema version of a database file is stored in `PRAGMA user_version`. v3
files start at `V3_BASE_USER_VERSION` (300) and every later schema change within
v3 is a migration which increments it. v1/v2 files never set it, so they read 0.

Functions:
    open_db(db_path: str) -> sqlite3.Connection: Open a database, creating it or
        applying any missing migrations first.
"""

from typing import List

__all__: List[str] = ["open_db"]

import logging
import os
import sqlite3
from typing import Callable, Tuple

from lute.io._db.common_sqlite import (
    DatabaseError,
    V3_BASE_USER_VERSION,
    get_user_version,
    set_user_version,
)

from ._sqlite import PRAGMAS, create_tables

logger: logging.Logger = logging.getLogger(__name__)

Migration = Tuple[int, int, Callable[[sqlite3.Connection], None]]

# (from user_version, to user_version, function). Applied in order. A migration
# function must not commit: no `with con:` and no `con.executescript`.
MIGRATIONS: List[Migration] = [
    (0, V3_BASE_USER_VERSION, create_tables),
]


def latest_user_version() -> int:
    """Return the `user_version` of a database with every migration applied."""
    return MIGRATIONS[-1][1]


def _has_tables(con: sqlite3.Connection) -> bool:
    res: sqlite3.Cursor = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' "
        "AND name NOT LIKE 'sqlite_%' LIMIT 1"
    )
    return res.fetchone() is not None


def _check_version(con: sqlite3.Connection, user_version: int, db_path: str) -> None:
    """Raise if the database cannot be used or migrated by this version of LUTE."""
    if user_version > latest_user_version():
        raise DatabaseError(
            f"{db_path} has schema version {user_version}, newer than this LUTE "
            f"supports ({latest_user_version()}). Update LUTE to use it."
        )
    if user_version == 0 and _has_tables(con):
        raise DatabaseError(
            f"{db_path} was written with database specification v1/v2 and cannot "
            "be used with v3. Move it aside, or set LUTE_DB_SPEC_VERSION=2."
        )
    if 0 < user_version < V3_BASE_USER_VERSION:
        raise DatabaseError(
            f"{db_path} has an unknown schema version ({user_version})."
        )


def _backup(db_path: str, user_version: int) -> None:
    """Copy the database before migrating it. Keeps an existing backup.

    Must be called while holding the write lock but before writing anything.
    SQLite refuses to back up from a connection with an open write transaction
    (`backup()` would retry forever), so this reads through a second connection,
    which the write lock still allows.
    """
    backup_path: str = f"{db_path}.bak-v{user_version}"
    if os.path.exists(backup_path):
        logger.info(f"Backup {backup_path} already exists, keeping it.")
        return
    src: sqlite3.Connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    dest: sqlite3.Connection = sqlite3.connect(backup_path)
    try:
        src.backup(dest)
    finally:
        dest.close()
        src.close()
    logger.info(f"Backed up {db_path} to {backup_path} before migrating.")


def _migrate(con: sqlite3.Connection, db_path: str) -> None:
    """Apply missing migrations in one transaction, holding the write lock."""
    con.isolation_level = None  # Manage the transaction explicitly
    try:
        # Takes the write lock now, so concurrent openers wait here, then find
        # the database already migrated.
        con.execute("BEGIN IMMEDIATE")
        user_version: int = get_user_version(con)
        _check_version(con, user_version, db_path)
        if 0 < user_version < latest_user_version():
            _backup(db_path, user_version)
        while user_version < latest_user_version():
            migration: Migration
            for migration in MIGRATIONS:
                if migration[0] == user_version:
                    break
            else:
                raise DatabaseError(
                    f"No migration from schema version {user_version} for {db_path}."
                )
            _, to_version, migrate = migration
            logger.debug(f"Migrating {db_path}: {user_version} -> {to_version}")
            migrate(con)
            set_user_version(con, to_version)
            user_version = to_version
        con.execute("COMMIT")
    except BaseException:
        if con.in_transaction:
            con.execute("ROLLBACK")
        raise
    finally:
        con.isolation_level = ""


def open_db(db_path: str, timeout: float = 60.0) -> sqlite3.Connection:
    """Open a v3 database, creating it or migrating it to the latest schema.

    If the database is already at the latest schema version, no write lock is
    taken, so users with read-only access can still open it.

    Args:
        db_path (str): Path to the `lute.db` file. Created if it does not exist.

        timeout (float): Seconds to wait for another process holding the lock.

    Returns:
        con (sqlite3.Connection): Connection with the LUTE PRAGMAS applied.

    Raises:
        DatabaseError: If the file was written by v1/v2 or by a newer LUTE, or
            if a migration failed. A failed migration leaves the file unchanged,
            including when the process is killed: the next connection rolls
            the migration back from the journal on disk.
    """
    con: sqlite3.Connection = sqlite3.connect(db_path, timeout=timeout)
    try:
        con.executescript(PRAGMAS)
        # Fast path, without the lock: the version only increases, so a file
        # read at the latest version stays at it. Anything else is checked by
        # `_migrate` under the write lock, as another process may be migrating
        # the file right now (checking here could mix a stale version with its
        # newly created tables).
        if get_user_version(con) != latest_user_version():
            _migrate(con, db_path)
    except BaseException:
        con.close()
        raise
    return con
