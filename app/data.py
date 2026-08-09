"""DuckDB read access + cached queries for the Streamlit app."""
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

DUCKDB_PATH = Path(__file__).resolve().parent.parent / "data" / "warehouse.duckdb"

TICKER_ORDER = ["MSFT", "AAPL", "GOOGL", "AMZN", "META"]


def get_connection() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(DUCKDB_PATH), read_only=True)


@st.cache_data(ttl=3600)
def load_financials_quarterly() -> pd.DataFrame:
    con = get_connection()
    df = con.execute("select * from main.fct_financials_quarterly order by ticker, period_end").fetchdf()
    con.close()
    return df


@st.cache_data(ttl=3600)
def load_valuation_daily() -> pd.DataFrame:
    con = get_connection()
    tables = con.execute("select table_name from information_schema.tables where table_name = 'fct_valuation_daily'").fetchdf()
    if tables.empty:
        con.close()
        return pd.DataFrame()
    df = con.execute("select * from main.fct_valuation_daily order by ticker, price_date").fetchdf()
    con.close()
    return df


@st.cache_data(ttl=3600)
def load_company_names() -> dict:
    return {
        "MSFT": "Microsoft Corporation",
        "AAPL": "Apple Inc.",
        "GOOGL": "Alphabet Inc.",
        "AMZN": "Amazon.com, Inc.",
        "META": "Meta Platforms, Inc.",
    }
