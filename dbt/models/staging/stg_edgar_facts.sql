-- SEC XBRL facts get restated across multiple filings (a 10-K reports prior-year
-- comparatives that a 10-Q already reported). Keep only the most recently filed
-- version of each (ticker, concept, period).
with source as (
    select * from {{ source('raw', 'raw_edgar_facts') }}
    where form in ('10-Q', '10-K')
),

deduped as (
    select
        ticker,
        concept,
        tag,
        form,
        fy,
        fp,
        "start" as period_start,
        "end" as period_end,
        filed,
        val,
        row_number() over (
            partition by ticker, concept, coalesce("start", ''), "end"
            order by filed desc
        ) as rn
    from source
)

select
    ticker,
    concept,
    tag,
    form,
    fy,
    fp,
    period_start::date as period_start,
    period_end::date as period_end,
    filed::date as filed,
    val,
    case
        when period_start is null then null
        else date_diff('day', period_start::date, period_end::date)
    end as period_length_days
from deduped
where rn = 1
