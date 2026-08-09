"""Pull daily prices and valuation snapshot data from Yahoo Finance via yfinance."""
import pandas as pd
import yfinance as yf

from config import RAW_DIR, START_DATE, TICKERS


def fetch_prices() -> pd.DataFrame:
    # Downloaded one ticker at a time: yfinance's shared sqlite cache can silently
    # drop a ticker's data ("database is locked") when multiple tickers are batched.
    frames = []
    for ticker in TICKERS:
        df = yf.download(ticker, start=START_DATE, auto_adjust=True, progress=False)
        df = df.reset_index()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df[["Date", "Close", "Volume"]].copy()
        df.columns = ["date", "close", "volume"]
        df["ticker"] = ticker
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def fetch_valuation_snapshot() -> pd.DataFrame:
    """Current-point-in-time valuation multiples (yfinance does not expose history for these)."""
    rows = []
    for ticker in TICKERS:
        info = yf.Ticker(ticker).info
        rows.append(
            {
                "ticker": ticker,
                "market_cap": info.get("marketCap"),
                "enterprise_value": info.get("enterpriseValue"),
                "trailing_pe": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "ev_to_ebitda": info.get("enterpriseToEbitda"),
                "shares_outstanding": info.get("sharesOutstanding"),
                "total_debt": info.get("totalDebt"),
                "total_cash": info.get("totalCash"),
            }
        )
    return pd.DataFrame(rows)


def main():
    prices = fetch_prices()
    prices_path = RAW_DIR / "yfinance_prices.parquet"
    prices.to_parquet(prices_path, index=False)
    print(f"Wrote {len(prices)} rows to {prices_path}")

    snapshot = fetch_valuation_snapshot()
    snapshot_path = RAW_DIR / "yfinance_valuation_snapshot.parquet"
    snapshot.to_parquet(snapshot_path, index=False)
    print(f"Wrote {len(snapshot)} rows to {snapshot_path}")


if __name__ == "__main__":
    main()
