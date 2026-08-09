select
    ticker,
    market_cap,
    enterprise_value,
    trailing_pe,
    forward_pe,
    ev_to_ebitda as ev_to_ebitda_asof_today,
    shares_outstanding,
    total_debt,
    total_cash
from {{ source('raw', 'raw_yfinance_valuation_snapshot') }}
