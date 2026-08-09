-- Balance sheet items are "instant" XBRL facts (no start date), one row per
-- ticker + as-of date. `filed_date` records when the filing became public so
-- downstream models can join without look-ahead bias.
with instant_facts as (
    select ticker, period_end, concept, val, filed
    from {{ ref('stg_edgar_facts') }}
    where period_start is null
),

pivoted as (
    select
        ticker,
        period_end,
        max(filed) as filed_date,
        max(case when concept = 'assets' then val end) as total_assets,
        max(case when concept = 'assets_current' then val end) as current_assets,
        max(case when concept = 'liabilities' then val end) as reported_total_liabilities,
        max(case when concept = 'liabilities_current' then val end) as current_liabilities,
        max(case when concept = 'stockholders_equity' then val end) as stockholders_equity,
        max(case when concept = 'long_term_debt' then val end) as long_term_debt,
        max(case when concept = 'short_term_debt' then val end) as short_term_debt,
        max(case when concept = 'cash_and_equivalents' then val end) as cash_and_equivalents
    from instant_facts
    group by ticker, period_end
)

select
    ticker,
    period_end,
    filed_date,
    total_assets,
    current_assets,
    -- AMZN doesn't tag total Liabilities directly; derive it from the accounting identity.
    coalesce(reported_total_liabilities, total_assets - stockholders_equity) as total_liabilities,
    current_liabilities,
    stockholders_equity,
    coalesce(long_term_debt, 0) + coalesce(short_term_debt, 0) as total_debt,
    cash_and_equivalents
from pivoted
where total_assets is not null
order by ticker, period_end
