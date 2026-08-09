"""Simple RAG-lite chat: stuff the curated latest-quarter dataset into the system
prompt (small enough not to need a vector store) and let Claude answer questions
about it."""
import os

import anthropic
import pandas as pd
import streamlit as st

MODEL = "claude-opus-5"

SYSTEM_PROMPT = """You are a financial data analyst assistant embedded in a public portfolio \
dashboard about how Fed rate hikes since 2022 affected Big Tech balance sheet leverage and \
valuation (MSFT, AAPL, GOOGL, AMZN, META).

You answer questions using ONLY the curated dataset and statistics provided below. If the \
answer isn't in the data, say so plainly rather than guessing. Cite specific numbers and quarters \
when relevant, including regression coefficients, R^2, and p-values where they're the most direct \
answer to a question about how the Fed rate relates to leverage or valuation. Keep answers \
concise -- a few sentences unless the user asks for detail.

=== Fed funds rate vs. leverage/valuation: trend and regression summary (covers the full history, \
including the 2022-2023 hiking cycle) ===
{rate_context}

=== Full quarterly balance sheet / income statement history per company, 2021-present (USD unless noted) ===
{context}
"""


def _get_api_key() -> str | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return None


def build_context(financials: pd.DataFrame) -> str:
    """Full quarterly history, not just the latest few quarters -- the chat needs to see the
    2022-2023 hiking cycle itself, not just where leverage/liquidity landed afterward. The
    dataset is small (~20 quarters x 5 tickers) so this comfortably fits in the system prompt
    without needing a vector store."""
    if financials.empty:
        return "(no data available)"
    full_history = financials.sort_values(["ticker", "period_end"])
    cols = [
        "ticker", "period_end", "revenue", "net_income", "ebitda", "ebitda_ttm",
        "current_ratio", "debt_to_equity", "liabilities_to_assets",
        "total_assets", "total_debt", "stockholders_equity",
    ]
    cols = [c for c in cols if c in full_history.columns]
    return full_history[cols].to_csv(index=False)


def ask(question: str, context: str, rate_context: str, history: list[dict]) -> str:
    api_key = _get_api_key()
    if not api_key:
        return (
            "No ANTHROPIC_API_KEY configured. Add it to `.env` locally or to "
            "Streamlit Cloud secrets to enable the chat."
        )

    client = anthropic.Anthropic(api_key=api_key)
    messages = history + [{"role": "user", "content": question}]

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT.format(context=context, rate_context=rate_context),
        output_config={"effort": "medium"},
        messages=messages,
    )
    if response.stop_reason == "refusal":
        return "The request was declined by safety filters. Try rephrasing."
    return next((b.text for b in response.content if b.type == "text"), "")
