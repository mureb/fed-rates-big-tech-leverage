import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import chat
import stats as stats_module
from data import load_company_names, load_financials_quarterly, load_valuation_daily

TICKER_ORDER = ["MSFT", "AAPL", "GOOGL", "AMZN", "META"]
PALETTE = {
    "MSFT": "#2a78d6",
    "AAPL": "#eb6834",
    "GOOGL": "#1baf7a",
    "AMZN": "#eda100",
    "META": "#e87ba4",
}

st.set_page_config(page_title="Fed Rates vs Big Tech Leverage", layout="wide")


def styled(fig):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend_title_text="",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def main():
    st.title("How did Fed rate hikes since 2022 affect Big Tech balance sheet leverage and valuation?")
    st.caption("MSFT · AAPL · GOOGL · AMZN · META — 2021 to present. Data: SEC EDGAR, FRED, Yahoo Finance. Pipeline: Python → DuckDB → dbt.")

    company_names = load_company_names()
    financials = load_financials_quarterly()
    valuation = load_valuation_daily()

    tickers = st.sidebar.multiselect("Companies", TICKER_ORDER, default=TICKER_ORDER)
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**About this project**\n\n"
        "A scoped, end-to-end data engineering portfolio project: extraction (Python) → "
        "warehouse (DuckDB) → transformation + tests (dbt) → dashboard + AI chat (Streamlit + Claude API)."
    )

    tab_dashboard, tab_chat = st.tabs(["Dashboard", "Ask the Data"])

    with tab_dashboard:
        render_dashboard(financials, valuation, tickers, company_names)

    with tab_chat:
        render_chat(financials, valuation)


def render_dashboard(financials: pd.DataFrame, valuation: pd.DataFrame, tickers: list, company_names: dict):
    fin = financials[financials["ticker"].isin(tickers)]

    st.subheader("Latest quarter snapshot")
    # Balance sheet and income statement quarters can be a filing or two out of
    # sync (e.g. a 10-Q's balance sheet posts before that quarter's income
    # statement is fully tagged), so pull revenue/EBITDA from the latest
    # quarter that actually has them rather than the latest quarter overall.
    fin_sorted = fin.sort_values("period_end")
    latest_balance = fin_sorted.groupby("ticker").tail(1).set_index("ticker")
    latest_income = (
        fin_sorted.dropna(subset=["revenue"]).groupby("ticker").tail(1).set_index("ticker")
    )
    latest = latest_balance.copy()
    for col in ["revenue", "ebitda_ttm"]:
        latest[col] = latest_income[col].reindex(latest.index)
    latest = latest.reindex([t for t in TICKER_ORDER if t in tickers])
    display_cols = {
        "period_end": "As of",
        "current_ratio": "Current Ratio",
        "debt_to_equity": "Debt / Equity",
        "liabilities_to_assets": "Liabilities / Assets",
        "revenue": "Revenue (qtr)",
        "ebitda_ttm": "EBITDA (TTM)",
    }
    st.dataframe(
        latest[list(display_cols.keys())].rename(columns=display_cols),
        use_container_width=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(
            fin, x="period_end", y="debt_to_equity", color="ticker",
            color_discrete_map=PALETTE, category_orders={"ticker": TICKER_ORDER},
            title="Debt-to-Equity Ratio", markers=True,
        )
        st.plotly_chart(styled(fig), use_container_width=True)
    with col2:
        fig = px.line(
            fin, x="period_end", y="current_ratio", color="ticker",
            color_discrete_map=PALETTE, category_orders={"ticker": TICKER_ORDER},
            title="Current Ratio", markers=True,
        )
        st.plotly_chart(styled(fig), use_container_width=True)

    if valuation.empty:
        st.info(
            "Fed funds rate and daily valuation trend charts need FRED data. "
            "Run `pipeline/extract_fred.py` (requires a free FRED API key) and "
            "`dbt run` to populate `fct_valuation_daily`."
        )
    else:
        val = valuation[valuation["ticker"].isin(tickers)].copy()

        st.subheader("Fed Funds Rate")
        fed = val[["price_date", "fed_funds_rate"]].dropna().drop_duplicates()
        fig = px.line(fed, x="price_date", y="fed_funds_rate", title="Effective Fed Funds Rate (%)")
        fig.update_traces(line_color=PALETTE["MSFT"])
        st.plotly_chart(styled(fig), use_container_width=True)

        st.subheader("Approx. Market Cap (indexed to 100 at first observation)")
        val["market_cap_indexed"] = val.groupby("ticker")["market_cap_approx"].transform(
            lambda s: s / s.iloc[0] * 100
        )
        fig = px.line(
            val, x="price_date", y="market_cap_indexed", color="ticker",
            color_discrete_map=PALETTE, category_orders={"ticker": TICKER_ORDER},
            title="Indexed Market Cap",
        )
        st.plotly_chart(styled(fig), use_container_width=True)

        st.subheader("Approx. EV / EBITDA (TTM)")
        ev = val.dropna(subset=["ev_to_ebitda_approx"])
        fig = px.line(
            ev, x="price_date", y="ev_to_ebitda_approx", color="ticker",
            color_discrete_map=PALETTE, category_orders={"ticker": TICKER_ORDER},
            title="EV / EBITDA (approx., TTM basis)",
        )
        st.plotly_chart(styled(fig), use_container_width=True)

        render_statistical_analysis(val)

    with st.expander("View underlying quarterly data"):
        st.dataframe(fin, use_container_width=True)


def render_statistical_analysis(val: pd.DataFrame):
    st.subheader("Statistical Analysis: Fed Funds Rate vs. Valuation")
    st.caption(
        "Regressing valuation *levels* on the rate *level* would show a strong relationship "
        "just because both series trend over 2021-2026 -- a classic spurious-regression trap "
        "for two non-stationary series. This regresses month-over-month **changes** instead: "
        "does a change in the Fed funds rate coincide with a change in valuation that same month."
    )

    changes = stats_module.monthly_changes(val)
    pooled = stats_module.pooled_stats(val)
    per_ticker = stats_module.per_ticker_stats(val)

    if changes.empty or not pooled:
        st.info("Not enough overlapping months of data to fit a regression for the selected companies.")
        return

    mc, ev = pooled.get("market_cap", {}), pooled.get("ev_ebitda", {})
    col1, col2 = st.columns(2)
    with col1:
        if mc.get("beta") is not None:
            st.metric(
                "Market cap sensitivity to Fed rate",
                f"{mc['beta']:+.2f} pp per 1pp rate move",
                help=f"Pooled OLS across all selected companies. R² = {mc['r_squared']:.3f}, "
                     f"p = {mc['p_value']:.3f}, n = {mc['n']} company-months.",
            )
    with col2:
        if ev.get("beta") is not None:
            st.metric(
                "EV/EBITDA sensitivity to Fed rate",
                f"{ev['beta']:+.2f} pp per 1pp rate move",
                help=f"Pooled OLS across all selected companies. R² = {ev['r_squared']:.3f}, "
                     f"p = {ev['p_value']:.3f}, n = {ev['n']} company-months.",
            )

    fig = px.scatter(
        changes, x="rate_change_pp", y="market_cap_pct_change", color="ticker",
        color_discrete_map=PALETTE, category_orders={"ticker": TICKER_ORDER},
        trendline="ols", trendline_scope="overall", trendline_color_override="#0b0b0b",
        labels={
            "rate_change_pp": "Δ Fed Funds Rate (pp, month-over-month)",
            "market_cap_pct_change": "Δ Market Cap (%, month-over-month)",
        },
        title="Monthly Change in Market Cap vs. Monthly Change in Fed Funds Rate",
    )
    st.plotly_chart(styled(fig), use_container_width=True)

    if not per_ticker.empty:
        st.dataframe(
            per_ticker.rename(columns={
                "ticker": "Ticker", "months": "Months",
                "market_cap_beta": "Mkt Cap β (pp/1pp)", "market_cap_r2": "Mkt Cap R²",
                "market_cap_pvalue": "Mkt Cap p-value",
                "ev_ebitda_beta": "EV/EBITDA β (pp/1pp)", "ev_ebitda_r2": "EV/EBITDA R²",
                "ev_ebitda_pvalue": "EV/EBITDA p-value",
            }).set_index("Ticker"),
            use_container_width=True,
        )
        st.caption(
            "β is the average percentage-point change in the metric per 1 percentage point "
            "change in the Fed funds rate that month. Small monthly sample sizes (~55 per "
            "company) mean these are indicative, not definitive -- and correlation isn't causation."
        )


def render_chat(financials: pd.DataFrame, valuation: pd.DataFrame):
    st.subheader("Ask the data")
    st.caption("Answers are grounded only in the curated dataset -- most recent 4 quarters per company, plus the Fed-rate/valuation regression above.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("e.g. Which company had the biggest jump in debt-to-equity since 2022?")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        context = chat.build_context(financials)
        rate_context = stats_module.context_summary(valuation)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = chat.ask(question, context, rate_context, st.session_state.chat_history[:-1])
            st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
