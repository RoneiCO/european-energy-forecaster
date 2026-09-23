from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def get_connection() -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection with views over the raw Parquet lake (prices, load_data, generation)."""
    con = duckdb.connect()
    for name in ("prices", "load", "generation"):
        view_name = "load_data" if name == "load" else name
        glob_path = str(RAW_DATA_DIR / name / "*.parquet")
        con.execute(
            f"CREATE VIEW {view_name} AS "
            f"SELECT DISTINCT * FROM read_parquet('{glob_path}', union_by_name = true)"
        )
    return con
