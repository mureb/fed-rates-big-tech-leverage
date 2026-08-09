-- Income statement items are "duration" XBRL facts. Companies tag Q1-Q3 as discrete
-- 3-month durations, but Q4 is never filed on its own -- it's only implied by the
-- annual (10-K, ~365-day) total. This model derives Q4 = FY total - (Q1+Q2+Q3).
--
-- Note: SEC's own fy/fp fields describe the *filing* that reported a fact, not the
-- fact's intrinsic fiscal period -- after deduping to the most-recently-filed version
-- of each period, those labels can point at the wrong fiscal year. So fiscal-year
-- buckets are derived here purely from date-range membership against the annual
-- period end dates, not from fy/fp text.
with quarterly_facts as (
    select ticker, period_start, period_end, concept, val, filed
    from {{ ref('stg_edgar_facts') }}
    where period_length_days between 80 and 100
),

quarterly_direct as (
    select
        ticker,
        period_start,
        period_end,
        max(filed) as filed_date,
        max(case when concept = 'revenues' then val end) as revenue,
        max(case when concept = 'operating_income' then val end) as operating_income,
        max(case when concept = 'net_income' then val end) as net_income,
        max(case when concept = 'depreciation_amortization' then val end) as depreciation_amortization
    from quarterly_facts
    group by ticker, period_start, period_end
),

annual_facts as (
    select ticker, period_end, concept, val, filed
    from {{ ref('stg_edgar_facts') }}
    where period_length_days between 350 and 380
),

annual as (
    select
        ticker,
        period_end as fy_end,
        max(filed) as fy_filed,
        max(case when concept = 'revenues' then val end) as revenue_fy,
        max(case when concept = 'operating_income' then val end) as operating_income_fy,
        max(case when concept = 'net_income' then val end) as net_income_fy,
        max(case when concept = 'depreciation_amortization' then val end) as da_fy
    from annual_facts
    group by ticker, period_end
),

fy_buckets as (
    select
        ticker,
        fy_end,
        fy_filed,
        revenue_fy,
        operating_income_fy,
        net_income_fy,
        da_fy,
        lag(fy_end) over (partition by ticker order by fy_end) as prior_fy_end
    from annual
),

quarter_bucketed as (
    select
        q.ticker,
        q.period_start,
        q.period_end,
        q.filed_date,
        q.revenue,
        q.operating_income,
        q.net_income,
        q.depreciation_amortization,
        b.fy_end
    from quarterly_direct q
    join fy_buckets b
        on q.ticker = b.ticker
        and q.period_end <= b.fy_end
        and (b.prior_fy_end is null or q.period_end > b.prior_fy_end)
),

bucket_agg as (
    select
        ticker,
        fy_end,
        count(*) as n_quarters,
        max(period_end) as q3_period_end,
        sum(revenue) as revenue_q1q3,
        sum(operating_income) as operating_income_q1q3,
        sum(net_income) as net_income_q1q3,
        sum(depreciation_amortization) as da_q1q3
    from quarter_bucketed
    group by ticker, fy_end
),

q4_derived as (
    select
        a.ticker,
        b.q3_period_end as period_start,
        a.fy_end as period_end,
        a.fy_filed as filed_date,
        a.revenue_fy - b.revenue_q1q3 as revenue,
        a.operating_income_fy - b.operating_income_q1q3 as operating_income,
        a.net_income_fy - b.net_income_q1q3 as net_income,
        a.da_fy - b.da_q1q3 as depreciation_amortization,
        a.fy_end
    from annual a
    join bucket_agg b on a.ticker = b.ticker and a.fy_end = b.fy_end
    where b.n_quarters = 3
),

all_quarters as (
    select ticker, period_start, period_end, filed_date, revenue, operating_income, net_income, depreciation_amortization, fy_end
    from quarter_bucketed
    union all
    select ticker, period_start, period_end, filed_date, revenue, operating_income, net_income, depreciation_amortization, fy_end
    from q4_derived
),

with_ebitda as (
    select
        *,
        operating_income + coalesce(depreciation_amortization, 0) as ebitda,
        row_number() over (partition by ticker, fy_end order by period_end) as fiscal_quarter
    from all_quarters
    where operating_income is not null
)

select
    ticker,
    fy_end as fiscal_year_end,
    'Q' || fiscal_quarter as fiscal_quarter,
    period_start,
    period_end,
    filed_date,
    revenue,
    operating_income,
    net_income,
    depreciation_amortization,
    ebitda,
    sum(ebitda) over (
        partition by ticker order by period_end
        rows between 3 preceding and current row
    ) as ebitda_ttm,
    count(*) over (
        partition by ticker order by period_end
        rows between 3 preceding and current row
    ) as trailing_quarters_available
from with_ebitda
order by ticker, period_end
