"""Pull Fed Funds Rate (and related macro series) from the FRED API."""
import os

import pandas as pd
import requests
from dotenv import load_dotenv

from config import RAW_DIR, START_DATE

load_dotenv()

FRED_API_KEY = os.environ.get("FRED_API_KEY")
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# FEDFUNDS: Effective Federal Funds Rate (monthly, percent)
# DGS10: 10-Year Treasury Constant Maturity Rate, useful context for valuation multiples
SERIES = ["FEDFUNDS", "DGS10"]


def fetch_series(series_id: str) -> pd.DataFrame:
    if not FRED_API_KEY:
        raise RuntimeError(
            "FRED_API_KEY is not set. Get a free key at "
            "https://fred.stlouisfed.org/docs/api/api_key.html and add it to .env"
        )
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": START_DATE,
    }
    resp = requests.get(FRED_BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    obs = resp.json()["observations"]
    df = pd.DataFrame(obs)[["date", "value"]]
    df["series_id"] = series_id
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def main():
    frames = [fetch_series(s) for s in SERIES]
    out = pd.concat(frames, ignore_index=True)
    out_path = RAW_DIR / "fred_series.parquet"
    out.to_parquet(out_path, index=False)
    print(f"Wrote {len(out)} rows to {out_path}")


if __name__ == "__main__":
    main()
