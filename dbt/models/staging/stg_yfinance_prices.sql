select
    ticker,
    "date"::date as price_date,
    close as close_price,
    volume
from {{ source('raw', 'raw_yfinance_prices') }}
where close is not null
