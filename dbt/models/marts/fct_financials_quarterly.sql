-- One row per ticker/quarter combining balance sheet + income statement facts
-- and the core solvency/leverage ratios.
select
    b.ticker,
    b.period_end,
    greatest(b.filed_date, i.filed_date) as filed_date,
    b.total_assets,
    b.current_assets,
    b.total_liabilities,
    b.current_liabilities,
    b.stockholders_equity,
    b.total_debt,
    b.cash_and_equivalents,
    round(b.current_assets / nullif(b.current_liabilities, 0), 2) as current_ratio,
    round(b.total_debt / nullif(b.stockholders_equity, 0), 2) as debt_to_equity,
    round(b.total_liabilities / nullif(b.total_assets, 0), 2) as liabilities_to_assets,
    i.revenue,
    i.operating_income,
    i.net_income,
    i.ebitda,
    case when i.trailing_quarters_available = 4 then i.ebitda_ttm end as ebitda_ttm
from {{ ref('fct_balance_sheet_quarterly') }} b
left join {{ ref('fct_income_statement_quarterly') }} i
    on b.ticker = i.ticker and b.period_end = i.period_end
order by b.ticker, b.period_end
