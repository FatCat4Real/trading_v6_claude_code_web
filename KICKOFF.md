I want to test lots of indicator on @data/set100_stocks_2020_2025.parquet. The simulation is simple, for example, if I want to test "MACD" strategy
1. apply MACD on stock data, each ticker. So you have to group by ticker.
2. We use LONG ONLY strategy, so no shorting.
3. Mark which day has the buy/sell signal.
4. Since im on daily timeframe, we will take action (buy/sell) on the next bar at open price.
5. The buy position continues until there is a sell.
6. If the first signal in dataset (for a given ticker) is sell, ignore it till we find the first buy since we are LONG ONLY.
7. If there is an overlapping in signals (eg. buy and sell on the same bar), alert me with the ticker name, date, and strategy/indicator name.
8. Once we can label which date is to buy and sell stocks, match them by joining them to create a dataframe with this structure:
    ENTRY_DATE, TICKER, ENTRY_PRICE, ENTRY_BAR (bar number at entry), EXIT_DATE, EXIT_PRICE, EXIT_BAR (bar number at exit), DATE_DIFF, BAR_DIFF, PRICE_DIFF_PCT (selling price vs buying price for pnl calulation)
9. We ignore fees and slipage for now.
10. Once you get the dataframe in item 8., calculate the following metrics: 
    n_trades
    , win_rate
    , avg_return (pct)
    , avg_win (pct)
    , avg_loss (pct)
    , win_loss_ratio
    , profit_factor
    , expectancy
    , avg_holding_days
    , avg_holding_bars
    , max_drawdown (pct)
    , total_return (pct)
    , sharpe
    , return (pct) for each year
since we don't use initial capital in this simulation, most metrics are concluded using percentage

11. I want to test this on many many indicators so I can see which indicator works best, test most popular indicators available on TradingView first (i believe there are the official ones which can be more than 20+), then test other indicators you see fit, please reference the source of each indicator too like "TradingView", "ta-lib", etc.
12. I want the result in tabular format with strategy/indicator name as columns and the calculated metrics from item 10. as index.