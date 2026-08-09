"""Land raw extracted files (parquet) into DuckDB as raw.* tables."""
import duckdb

from config import DUCKDB_PATH, RAW_DIR

RAW_TABLES = {
    "raw_edgar_facts": "edgar_facts.parquet",
    "raw_yfinance_prices": "yfinance_prices.parquet",
    "raw_yfinance_valuation_snapshot": "yfinance_valuation_snapshot.parquet",
    "raw_fred_series": "fred_series.parquet",
}


def main():
    con = duckdb.connect(str(DUCKDB_PATH))
    con.execute("CREATE SCHEMA IF NOT EXISTS raw")

    for table, filename in RAW_TABLES.items():
        path = RAW_DIR / filename
        if not path.exists():
            print(f"  [skip] {filename} not found, run its extractor first")
            continue
        con.execute(
            f"CREATE OR REPLACE TABLE raw.{table} AS SELECT * FROM read_parquet('{path.as_posix()}')"
        )
        count = con.execute(f"SELECT COUNT(*) FROM raw.{table}").fetchone()[0]
        print(f"Loaded raw.{table}: {count} rows")

    con.close()


if __name__ == "__main__":
    main()
