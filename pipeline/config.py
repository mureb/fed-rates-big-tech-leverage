"""Shared configuration: company universe, date range, file paths."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
DUCKDB_PATH = DATA_DIR / "warehouse.duckdb"

START_DATE = "2021-01-01"

# SEC EDGAR CIKs are 10-digit, zero-padded.
COMPANIES = {
    "MSFT": {"name": "Microsoft Corporation", "cik": "0000789019"},
    "AAPL": {"name": "Apple Inc.", "cik": "0000320193"},
    "GOOGL": {"name": "Alphabet Inc.", "cik": "0001652044"},
    "AMZN": {"name": "Amazon.com, Inc.", "cik": "0001018724"},
    "META": {"name": "Meta Platforms, Inc.", "cik": "0001326801"},
}

TICKERS = list(COMPANIES.keys())

# SEC requires a descriptive User-Agent with contact info on every request.
SEC_USER_AGENT = "Data Engineering Portfolio Project leonardomureb@gmail.com"

for d in (RAW_DIR,):
    d.mkdir(parents=True, exist_ok=True)
