-- Daily valuation trend overlaid with the Fed funds rate. Market cap and EV are
-- approximated using current shares outstanding (yfinance does not expose historical
-- share counts on the free tier) applied to historical close price -- a documented
-- simplification, not a restatement of actual historical share count.
-- ASOF joins ensure each price date only sees financial facts that had already been
-- publicly filed as of that date (no look-ahead bias).
with prices as (
    select ticker, price_date, close_price
    from {{ ref('stg_yfinance_prices') }}
),

snapshot as (
    select ticker, shares_outstanding, total_debt as current_total_debt, total_cash as current_total_cash
    from {{ ref('stg_yfinance_valuation_snapshot') }}
),

financials as (
    select ticker, period_end, filed_date, ebitda_ttm, total_debt as reported_total_debt, cash_and_equivalents
    from {{ ref('fct_financials_quarterly') }}
    where ebitda_ttm is not null
),

fed_funds as (
    select obs_date, value as fed_funds_rate
    from {{ ref('stg_fred_series') }}
    where series_id = 'FEDFUNDS'
),

priced as (
    select
        p.ticker,
        p.price_date,
        p.close_price,
        p.close_price * s.shares_outstanding as market_cap_approx
    from prices p
    join snapshot s on p.ticker = s.ticker
),

with_financials as (
    select
        pr.*,
        f.period_end as latest_filed_period_end,
        f.ebitda_ttm,
        f.reported_total_debt,
        f.cash_and_equivalents
    from priced pr
    asof left join financials f
        on pr.ticker = f.ticker and pr.price_date >= f.filed_date
)

select
    wf.*,
    market_cap_approx + coalesce(reported_total_debt, 0) - coalesce(cash_and_equivalents, 0) as enterprise_value_approx,
    round(
        (market_cap_approx + coalesce(reported_total_debt, 0) - coalesce(cash_and_equivalents, 0))
        / nullif(ebitda_ttm, 0),
        2
    ) as ev_to_ebitda_approx,
    ff.fed_funds_rate
from with_financials wf
asof left join fed_funds ff on wf.price_date >= ff.obs_date
order by ticker, price_date
