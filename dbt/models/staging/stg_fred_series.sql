select
    series_id,
    "date"::date as obs_date,
    value
from {{ source('raw', 'raw_fred_series') }}
where value is not null
