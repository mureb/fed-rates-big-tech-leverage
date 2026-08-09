"""Quantify the relationship between the Fed funds rate and Big Tech valuation.

Regressing valuation *levels* on the rate *level* would show a strong relationship purely
because both series trend over 2021-2026 -- a classic spurious-regression trap for two
non-stationary time series. Instead this module regresses month-over-month *changes*: does a
change in the Fed funds rate coincide with a change in market cap / EV-EBITDA that same month.
"""
import pandas as pd
import statsmodels.api as sm
import streamlit as st


@st.cache_data(ttl=3600)
def monthly_changes(valuation: pd.DataFrame) -> pd.DataFrame:
    if valuation.empty:
        return pd.DataFrame()

    frames = []
    for ticker, grp in valuation.groupby("ticker"):
        monthly = (
            grp.dropna(subset=["fed_funds_rate"])
            .set_index("price_date")
            .resample("MS")
            .last()[["market_cap_approx", "ev_to_ebitda_approx", "fed_funds_rate"]]
        )
        monthly["ticker"] = ticker
        monthly["market_cap_pct_change"] = monthly["market_cap_approx"].pct_change() * 100
        monthly["ev_ebitda_pct_change"] = monthly["ev_to_ebitda_approx"].pct_change() * 100
        monthly["rate_change_pp"] = monthly["fed_funds_rate"].diff()
        frames.append(monthly.reset_index())

    out = pd.concat(frames, ignore_index=True)
    return out.dropna(subset=["rate_change_pp"])


def _fit_ols(df: pd.DataFrame, x_col: str, y_col: str) -> dict:
    d = df[[x_col, y_col]].dropna()
    if len(d) < 6:
        return {"n": len(d), "beta": None, "r_squared": None, "p_value": None}
    x = sm.add_constant(d[x_col])
    model = sm.OLS(d[y_col], x).fit()
    return {
        "n": int(model.nobs),
        "beta": model.params[x_col],
        "r_squared": model.rsquared,
        "p_value": model.pvalues[x_col],
    }


@st.cache_data(ttl=3600)
def per_ticker_stats(valuation: pd.DataFrame) -> pd.DataFrame:
    changes = monthly_changes(valuation)
    if changes.empty:
        return pd.DataFrame()
    rows = []
    for ticker, grp in changes.groupby("ticker"):
        mc = _fit_ols(grp, "rate_change_pp", "market_cap_pct_change")
        ev = _fit_ols(grp, "rate_change_pp", "ev_ebitda_pct_change")
        rows.append({
            "ticker": ticker,
            "months": mc["n"],
            "market_cap_beta": mc["beta"],
            "market_cap_r2": mc["r_squared"],
            "market_cap_pvalue": mc["p_value"],
            "ev_ebitda_beta": ev["beta"],
            "ev_ebitda_r2": ev["r_squared"],
            "ev_ebitda_pvalue": ev["p_value"],
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600)
def pooled_stats(valuation: pd.DataFrame) -> dict:
    changes = monthly_changes(valuation)
    if changes.empty:
        return {}
    return {
        "market_cap": _fit_ols(changes, "rate_change_pp", "market_cap_pct_change"),
        "ev_ebitda": _fit_ols(changes, "rate_change_pp", "ev_ebitda_pct_change"),
        "n_companies": changes["ticker"].nunique(),
    }


@st.cache_data(ttl=3600)
def quarterly_leverage_changes(financials: pd.DataFrame, valuation: pd.DataFrame) -> pd.DataFrame:
    """Quarter-over-quarter change in leverage/liquidity ratios vs. the change in the Fed funds
    rate as of each quarter-end (nearest available daily observation, no look-ahead)."""
    if financials.empty or valuation.empty:
        return pd.DataFrame()

    fed = (
        valuation[["price_date", "fed_funds_rate"]]
        .dropna()
        .drop_duplicates()
        .sort_values("price_date")
    )

    frames = []
    for ticker, grp in financials.sort_values("period_end").groupby("ticker"):
        grp = grp[["period_end", "debt_to_equity", "current_ratio"]].copy()
        grp = pd.merge_asof(grp, fed, left_on="period_end", right_on="price_date", direction="backward")
        grp["ticker"] = ticker
        grp["de_change"] = grp["debt_to_equity"].diff()
        grp["cr_change"] = grp["current_ratio"].diff()
        grp["rate_change_pp"] = grp["fed_funds_rate"].diff()
        frames.append(grp)

    out = pd.concat(frames, ignore_index=True)
    return out.dropna(subset=["rate_change_pp"])


@st.cache_data(ttl=3600)
def leverage_pooled_stats(financials: pd.DataFrame, valuation: pd.DataFrame) -> dict:
    changes = quarterly_leverage_changes(financials, valuation)
    if changes.empty:
        return {}
    return {
        "debt_to_equity": _fit_ols(changes, "rate_change_pp", "de_change"),
        "current_ratio": _fit_ols(changes, "rate_change_pp", "cr_change"),
        "n_companies": changes["ticker"].nunique(),
    }


@st.cache_data(ttl=3600)
def leverage_per_ticker_stats(financials: pd.DataFrame, valuation: pd.DataFrame) -> pd.DataFrame:
    changes = quarterly_leverage_changes(financials, valuation)
    if changes.empty:
        return pd.DataFrame()
    rows = []
    for ticker, grp in changes.groupby("ticker"):
        de = _fit_ols(grp, "rate_change_pp", "de_change")
        cr = _fit_ols(grp, "rate_change_pp", "cr_change")
        rows.append({
            "ticker": ticker,
            "quarters": de["n"],
            "debt_to_equity_beta": de["beta"],
            "debt_to_equity_r2": de["r_squared"],
            "debt_to_equity_pvalue": de["p_value"],
            "current_ratio_beta": cr["beta"],
            "current_ratio_r2": cr["r_squared"],
            "current_ratio_pvalue": cr["p_value"],
        })
    return pd.DataFrame(rows)


def context_summary(financials: pd.DataFrame, valuation: pd.DataFrame) -> str:
    """Compact plain-text summary for the chat's system prompt -- trend + regression results
    covering the *full* history (including the 2022-2023 hiking cycle itself), not just the
    latest few quarters."""
    if valuation.empty:
        return "(no valuation/Fed rate data available)"

    lines = []
    fed = valuation[["price_date", "fed_funds_rate"]].dropna()
    lines.append(
        f"Fed funds rate: {fed['fed_funds_rate'].min():.2f}% (low) to "
        f"{fed['fed_funds_rate'].max():.2f}% (peak) over {fed['price_date'].min().date()} to "
        f"{fed['price_date'].max().date()}; most recent value "
        f"{fed.sort_values('price_date')['fed_funds_rate'].iloc[-1]:.2f}%."
    )

    for ticker, grp in valuation.groupby("ticker"):
        grp = grp.dropna(subset=["market_cap_approx"]).sort_values("price_date")
        if grp.empty:
            continue
        first, last = grp.iloc[0], grp.iloc[-1]
        pct_change = (last["market_cap_approx"] / first["market_cap_approx"] - 1) * 100
        lines.append(
            f"{ticker}: approx. market cap changed {pct_change:+.0f}% from "
            f"{first['price_date'].date()} to {last['price_date'].date()}."
        )

    if not financials.empty:
        lines.append("")
        lines.append("Leverage/liquidity trend across the FULL history, including the hiking cycle itself:")
        for ticker, grp in financials.sort_values("period_end").groupby("ticker"):
            grp = grp.dropna(subset=["debt_to_equity"])
            if grp.empty:
                continue
            peak_row = grp.loc[grp["debt_to_equity"].idxmax()]
            first, last = grp.iloc[0], grp.iloc[-1]
            lines.append(
                f"{ticker}: debt/equity was {first['debt_to_equity']:.2f} as of "
                f"{first['period_end'].date()}, peaked at {peak_row['debt_to_equity']:.2f} "
                f"({peak_row['period_end'].date()}), now {last['debt_to_equity']:.2f} "
                f"({last['period_end'].date()}). Current ratio: {first['current_ratio']:.2f} -> "
                f"{last['current_ratio']:.2f} over the same window."
            )

    pooled = pooled_stats(valuation)
    if pooled:
        mc, ev = pooled.get("market_cap", {}), pooled.get("ev_ebitda", {})
        lines.append("")
        if mc.get("beta") is not None:
            lines.append(
                "Regression (month-over-month changes, pooled across all 5 companies, "
                f"n={mc['n']} company-months): a 1 percentage point change in the Fed funds rate "
                f"is associated with a {mc['beta']:+.2f} percentage point change in market cap "
                f"that same month (R^2={mc['r_squared']:.3f}, p={mc['p_value']:.3f})."
            )
        if ev.get("beta") is not None:
            lines.append(
                "Same regression against EV/EBITDA: "
                f"{ev['beta']:+.2f} pp change in EV/EBITDA per 1pp rate change "
                f"(R^2={ev['r_squared']:.3f}, p={ev['p_value']:.3f})."
            )

    lev_pooled = leverage_pooled_stats(financials, valuation) if not financials.empty else {}
    if lev_pooled:
        de, cr = lev_pooled.get("debt_to_equity", {}), lev_pooled.get("current_ratio", {})
        if de.get("beta") is not None:
            lines.append(
                "Regression (quarter-over-quarter changes, pooled, n="
                f"{de['n']} company-quarters): a 1 percentage point change in the Fed funds rate "
                f"is associated with a {de['beta']:+.3f} change in debt-to-equity that quarter "
                f"(R^2={de['r_squared']:.3f}, p={de['p_value']:.3f})."
            )
        if cr.get("beta") is not None:
            lines.append(
                "Same regression against the current ratio: "
                f"{cr['beta']:+.3f} change per 1pp rate change "
                f"(R^2={cr['r_squared']:.3f}, p={cr['p_value']:.3f})."
            )

    if pooled or lev_pooled:
        lines.append(
            "Note: all regressions above use changes, not levels, specifically to avoid the "
            "spurious correlation two trending series would otherwise show. Small samples "
            "(n per company ~55 months / ~20 quarters) -- read R^2/p-values as indicative, not "
            "definitive, and don't imply causation from correlation alone."
        )

    per_ticker = per_ticker_stats(valuation)
    if not per_ticker.empty:
        lines.append("Per-company beta (market cap %-change per 1pp rate change): " + ", ".join(
            f"{r.ticker}={r.market_cap_beta:+.2f}" for r in per_ticker.itertuples()
            if r.market_cap_beta is not None
        ))

    lev_per_ticker = leverage_per_ticker_stats(financials, valuation) if not financials.empty else pd.DataFrame()
    if not lev_per_ticker.empty:
        lines.append("Per-company beta (debt-to-equity change per 1pp rate change): " + ", ".join(
            f"{r.ticker}={r.debt_to_equity_beta:+.3f}" for r in lev_per_ticker.itertuples()
            if r.debt_to_equity_beta is not None
        ))

    return "\n".join(lines)
