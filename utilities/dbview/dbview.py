import argparse
import json
import os
import sqlite3
from typing import Any, Dict, List, Set, Tuple
from collections import defaultdict

from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.reactive import Reactive
from textual.widgets import Footer, Header, DataTable, TabbedContent, TabPane

from lute.io.db import (
    common_sqlite,
    get_executions_summary,
    get_task_parameters_summary,
)
from lute.io.db import LUTE_DB_SPEC_VERSION as api_version


class DBView(App):
    """DBView - A LUTE database inspector.

    This TUI application facilitates inspection of tables recorded in a LUTE
    database. Databases are opened in read-only mode - no modifications of data
    or records are permitted through this utility.
    """

    BINDINGS = [
        ("d", "toggle_dark", "Toggle dark mode"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self, *args, dbpath: str, summarize_v2: bool, **kwargs) -> None:
        """Open a view to a LUTE database.

        Args:
            dbpath (str): Path to a SQLite LUTE database.
        """
        super().__init__(*args, **kwargs)
        self._dbpath: str = dbpath
        self._summarize_v2: bool = summarize_v2
        self._con: sqlite3.Connection = sqlite3.Connection(self._dbpath)
        self._tables: List[str]
        if api_version == 1 or not self._summarize_v2:
            self._tables = common_sqlite.get_tables(self._con)
        else:
            # Will be updated dynamically
            self._tables = ["Executions"]

    def compose(self) -> ComposeResult:
        """Compose our UI."""
        yield Header()
        with TabbedContent():
            for table in self._tables:
                with TabPane(table):
                    with VerticalScroll(id=f"view_{table}"):
                        if api_version == 1 or not self._summarize_v2:
                            yield self.pull_table_data_v1(DataTable(id=f"data_{table}"))
                        elif api_version in (2, 3):
                            yield self.pull_table_data_v2(DataTable(id=f"data_{table}"))
        yield Footer()

    def pull_table_data_v1(self, table: DataTable) -> DataTable:
        """Query database for all rows in a table and add to display.

        This method is for database specification v0.1.
        """
        table_name: str = table.id[5:] if table.id is not None else "<UNKNOWN>"
        table.add_columns(*common_sqlite.get_table_cols(self._con, table_name))
        rows: List[Tuple[Any, ...]] = common_sqlite.get_all_rows_for_table(
            self._con, table_name
        )
        for row in rows:
            table.add_row(*row)

        return table

    def action_toggle_dark(self) -> None:
        self.dark: Reactive[bool] = not self.dark

    def pull_table_data_v2(self, table: DataTable) -> DataTable:
        """Query database for all rows in a table and add to display.

        This method is for database specification v0.2.
        """
        table_name: str = table.id[5:] if table.id is not None else "<UNKNOWN>"
        if table_name == "Executions":
            return self._executions_summary(table)
        else:  # Per task
            return self._task_param_summary(table, table_name)

    def _executions_summary(self, table: DataTable) -> DataTable:
        rows: List[Tuple[int, str, str, str, str, str, int]] = get_executions_summary(
            os.path.dirname(self._dbpath)
        )
        executions: defaultdict = defaultdict(dict)

        for (
            execution_id,
            timestamp,
            task_name,
            summary,
            payload,
            status,
            valid_flag,
        ) in rows:
            row = executions[execution_id]
            row["Execution ID"] = execution_id
            row["Timestamp"] = timestamp
            row["Task"] = task_name
            if task_name not in self._tables:
                self._tables.append(task_name)
            row["Summary"] = summary
            row["Payload"] = payload
            row["Status"] = status
            row["Valid?"] = True if valid_flag == 1 else False

        table_rows: List[Dict] = list(executions.values())
        cols_set: Set[str] = set()
        for row in table_rows:
            cols_set.update(row.keys())

        base_cols: List[str] = [
            "Execution ID",
            "Timestamp",
            "Task",
            "Summary",
            "Payload",
            "Status",
            "Valid?",
        ]

        table.add_columns(*base_cols)
        rows_list: List = []
        for row in table_rows:
            rows_list.append([row.get(col) for col in base_cols])
        for row in rows_list:
            table.add_row(*row)

        return table

    def _task_param_summary(self, table: DataTable, task: str) -> DataTable:
        rows: List[Tuple[int, str, int, str, str]] = get_task_parameters_summary(
            os.path.dirname(self._dbpath), task_name=task
        )
        executions: defaultdict = defaultdict(dict)

        for execution_id, timestamp, valid_flag, param_name, param_value in rows:
            row = executions[execution_id]
            row["Execution ID"] = execution_id
            row["Timestamp"] = timestamp
            row["Valid?"] = True if valid_flag == 1 else False
            row[param_name] = json.loads(param_value)

        table_rows: List[Dict] = list(executions.values())
        cols_set: Set[str] = set()
        for row in table_rows:
            cols_set.update(row.keys())

        base_cols = ["Execution ID", "Timestamp", "Valid?"]
        param_cols: List[str] = sorted(c for c in cols_set if c not in base_cols)
        cols: List[str] = base_cols + param_cols

        table.add_columns(*cols)
        rows_list: List = []
        for row in table_rows:
            rows_list.append([row.get(col) for col in cols])
        for row in rows_list:
            table.add_row(*row)

        return table


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="DBView",
        description="LUTE database inspection utility. Read-Only.",
        epilog="Refer to https://github.com/slac-lcls/lute for more information.",
    )
    parser.add_argument("-p", "--path", type=str, help="Path to SQLite database.")
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Pivot and reorganize to provide a summary. Only applies to db spec v0.2.",
    )
    args: argparse.Namespace = parser.parse_args()
    app: DBView = DBView(dbpath=args.path, summarize_v2=args.summarize)
    app.run()


if __name__ == "__main__":
    main()
