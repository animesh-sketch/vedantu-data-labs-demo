"""Read-only SQL validation and execution on DuckDB (in-memory, data registered as a view)."""
from __future__ import annotations

import re

import duckdb
import pandas as pd

TABLE = "data"
FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|create|alter|attach|detach|copy|export|import|install|load|pragma|set|reset|"
    r"call|checkpoint|vacuum|grant|revoke|truncate|merge|use|begin|commit|rollback|execute|prepare)\b", re.I)
FORBIDDEN_FUNCS = re.compile(
    r"\b(read_csv\w*|read_parquet|read_json\w*|read_text|read_blob|parquet_scan|csv_scan|glob|getenv|httpfs|"
    r"sqlite_scan|postgres_scan|mysql_scan|query_table|query)\s*\(", re.I)
MAX_ROWS = 10_000


class UnsafeSQLError(ValueError):
    pass


def _strip_literals(sql: str) -> str:
    return re.sub(r"'(?:[^']|'')*'", "''", sql)


def validate(sql: str) -> str:
    if not sql or not sql.strip():
        raise UnsafeSQLError("Empty query.")
    s = sql.strip().rstrip(";").strip()
    bare = _strip_literals(s)
    if ";" in bare:
        raise UnsafeSQLError("Only a single statement is allowed.")
    if "--" in bare or "/*" in bare:
        raise UnsafeSQLError("SQL comments are not allowed.")
    if not re.match(r"^\s*(select|with)\b", bare, re.I):
        raise UnsafeSQLError("Only SELECT / WITH queries are allowed.")
    if m := FORBIDDEN.search(bare):
        raise UnsafeSQLError(f"Keyword '{m.group(1).upper()}' is not allowed (read-only analytics only).")
    if m := FORBIDDEN_FUNCS.search(bare):
        raise UnsafeSQLError(f"Function '{m.group(1)}' is not allowed (no file/network/system access).")
    if re.search(r"\bfrom\s+'|\bjoin\s+'", bare, re.I) or re.search(r"\w\.(csv|tsv|parquet|json|db|duckdb|xlsx)\b", bare, re.I):
        raise UnsafeSQLError("Direct file access is not allowed; query the table `data`.")
    return s


def run(sql: str, df: pd.DataFrame) -> pd.DataFrame:
    """Validate then execute against an isolated in-memory DuckDB with external access disabled."""
    safe = validate(sql)
    con = duckdb.connect(":memory:")
    try:
        con.register("_src", df)
        con.execute(f"CREATE VIEW {TABLE} AS SELECT * FROM _src")
        con.execute("SET enable_external_access = false")
        con.execute("SET lock_configuration = true")
        return con.execute(f"SELECT * FROM ({safe}) AS q LIMIT {MAX_ROWS}").df()
    finally:
        con.close()
