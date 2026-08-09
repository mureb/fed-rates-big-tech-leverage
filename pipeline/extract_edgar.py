"""Pull balance sheet / income statement facts from SEC EDGAR's XBRL company-concept API.

Companies don't always tag the same economic concept with the same us-gaap element
(e.g. some use `Revenues`, others `RevenueFromContractWithCustomerExcludingAssessedTax`).
For each concept we try a list of known aliases and keep the first one that returns data.
"""
import time

import pandas as pd
import requests

from config import COMPANIES, RAW_DIR, START_DATE, SEC_USER_AGENT

BASE_URL = "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{tag}.json"

# concept name -> ordered list of candidate XBRL tags to try
CONCEPTS = {
    "assets": ["Assets"],
    "assets_current": ["AssetsCurrent"],
    "liabilities": ["Liabilities"],
    "liabilities_current": ["LiabilitiesCurrent"],
    "stockholders_equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "long_term_debt": [
        "LongTermDebtNoncurrent",
        "LongTermDebt",
    ],
    "short_term_debt": [
        "LongTermDebtCurrent",
        "ShortTermBorrowings",
        "DebtCurrent",
    ],
    "cash_and_equivalents": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss"],
    "revenues": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
    ],
    "depreciation_amortization": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
        "DepreciationAndAmortization",
        "DepreciationAmortizationAndImpairment",
        "Depreciation",
    ],
}


def fetch_concept(cik: str, tag: str) -> pd.DataFrame | None:
    url = BASE_URL.format(cik=cik, tag=tag)
    resp = requests.get(url, headers={"User-Agent": SEC_USER_AGENT}, timeout=30)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    payload = resp.json()
    usd_facts = payload.get("units", {}).get("USD", [])
    if not usd_facts:
        return None
    df = pd.DataFrame(usd_facts)
    df = df[df["end"] >= START_DATE]
    if df.empty:
        return None
    df["tag"] = tag
    return df


def fetch_company(ticker: str, cik: str) -> pd.DataFrame:
    rows = []
    for concept, tag_candidates in CONCEPTS.items():
        for tag in tag_candidates:
            df = fetch_concept(cik, tag)
            time.sleep(0.15)  # be polite to SEC's rate limits (10 req/sec max)
            if df is not None:
                df["concept"] = concept
                df["ticker"] = ticker
                rows.append(df)
                break  # first working alias wins
        else:
            print(f"  [warn] {ticker}: no data found for concept '{concept}'")
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    keep_cols = ["ticker", "concept", "tag", "form", "fy", "fp", "start", "end", "filed", "val"]
    return out[[c for c in keep_cols if c in out.columns]]


def main():
    frames = []
    for ticker, meta in COMPANIES.items():
        print(f"Fetching EDGAR facts for {ticker} ({meta['name']})...")
        frames.append(fetch_company(ticker, meta["cik"]))
    out = pd.concat(frames, ignore_index=True)
    out_path = RAW_DIR / "edgar_facts.parquet"
    out.to_parquet(out_path, index=False)
    print(f"Wrote {len(out)} rows to {out_path}")


if __name__ == "__main__":
    main()
