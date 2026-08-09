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


def context_summary(valuation: pd.DataFrame) -> str:
    """Compact plain-text summary for the chat's system prompt -- trend + regression results,
    not raw daily rows."""
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

    pooled = pooled_stats(valuation)
    if pooled:
        mc, ev = pooled.get("market_cap", {}), pooled.get("ev_ebitda", {})
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
        lines.append(
            "Note: these are contemporaneous month-over-month regressions on a small monthly "
            "sample (n per company ~55) -- read R^2/p-values as indicative, not definitive, and "
            "don't imply causation from correlation alone."
        )

    per_ticker = per_ticker_stats(valuation)
    if not per_ticker.empty:
        lines.append("Per-company beta (market cap %-change per 1pp rate change): " + ", ".join(
            f"{r.ticker}={r.market_cap_beta:+.2f}" for r in per_ticker.itertuples()
            if r.market_cap_beta is not None
        ))

    return "\n".join(lines)
