import numpy as np
import pandas as pd


def generate_trades(ticker_df, signals):
    """Convert signals to trades for a single ticker.
    signals: Series with 1=buy, -1=sell, 0=nothing.
    Executes on NEXT bar at OPEN price. LONG ONLY.
    """
    ticker = ticker_df['TICKER'].iloc[0]
    dates = ticker_df['DATE'].values
    opens = ticker_df['OPEN'].values
    n_bars = len(ticker_df)

    # extract non-zero signals
    sig_mask = signals.values != 0
    sig_values = signals.values[sig_mask]
    sig_indices = np.where(sig_mask)[0]

    if len(sig_values) == 0:
        return []

    # remove consecutive duplicate signals (keep alternating)
    if len(sig_values) > 1:
        change = np.concatenate([[True], sig_values[1:] != sig_values[:-1]])
        sig_values = sig_values[change]
        sig_indices = sig_indices[change]

    # LONG ONLY: first signal must be buy
    if sig_values[0] == -1:
        sig_values = sig_values[1:]
        sig_indices = sig_indices[1:]

    # if last signal is buy with no sell, drop it
    if len(sig_values) > 0 and sig_values[-1] == 1:
        sig_values = sig_values[:-1]
        sig_indices = sig_indices[:-1]

    n_trades = len(sig_values) // 2
    if n_trades == 0:
        return []

    trades = []
    for i in range(n_trades):
        buy_bar = sig_indices[i * 2]
        sell_bar = sig_indices[i * 2 + 1]

        entry_bar = buy_bar + 1   # execute on next bar
        exit_bar = sell_bar + 1

        if entry_bar >= n_bars or exit_bar >= n_bars:
            continue

        entry_price = opens[entry_bar]
        exit_price = opens[exit_bar]

        if entry_price <= 0:
            continue

        pnl = (exit_price - entry_price) / entry_price * 100
        date_diff = int((dates[exit_bar] - dates[entry_bar]) / np.timedelta64(1, 'D'))

        trades.append({
            'ENTRY_DATE': dates[entry_bar],
            'TICKER': ticker,
            'ENTRY_PRICE': entry_price,
            'ENTRY_BAR': int(entry_bar),
            'EXIT_DATE': dates[exit_bar],
            'EXIT_PRICE': exit_price,
            'EXIT_BAR': int(exit_bar),
            'DATE_DIFF': date_diff,
            'BAR_DIFF': int(exit_bar - entry_bar),
            'PRICE_DIFF_PCT': round(pnl, 4),
        })

    return trades


def calc_metrics(trades_df, all_years):
    """Calculate performance metrics from trades dataframe."""
    if trades_df.empty:
        return {}

    trades_df = trades_df.copy()
    trades_df['ENTRY_DATE'] = pd.to_datetime(trades_df['ENTRY_DATE'])
    trades_df['EXIT_DATE'] = pd.to_datetime(trades_df['EXIT_DATE'])

    returns = trades_df['PRICE_DIFF_PCT']
    n = len(returns)
    wins = returns[returns > 0]
    losses = returns[returns <= 0]

    n_trades = n
    win_rate = len(wins) / n * 100
    avg_return = returns.mean()
    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = losses.mean() if len(losses) > 0 else 0.0
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    wr = len(wins) / n
    expectancy = wr * avg_win + (1 - wr) * avg_loss

    avg_holding_days = trades_df['DATE_DIFF'].mean()
    avg_holding_bars = trades_df['BAR_DIFF'].mean()

    # max drawdown from cumulative sum equity curve (arithmetic, not compounded)
    # compounding doesn't make sense for multi-ticker independent trades
    sorted_trades = trades_df.sort_values('EXIT_DATE')
    cum_returns = sorted_trades['PRICE_DIFF_PCT'].cumsum()
    peak = cum_returns.cummax()
    drawdown = cum_returns - peak
    max_dd = drawdown.min()

    # total return (sum of all pct returns)
    total_return = returns.sum()

    # sharpe (annualized from trade returns)
    date_range = (trades_df['ENTRY_DATE'].max() - trades_df['ENTRY_DATE'].min()).days
    years = max(date_range / 365.25, 1)
    if returns.std() > 0:
        trades_per_year = n / years
        sharpe = (returns.mean() / returns.std()) * np.sqrt(trades_per_year)
    else:
        sharpe = 0.0

    metrics = {
        'n_trades': n_trades,
        'win_rate': round(win_rate, 2),
        'avg_return': round(avg_return, 2),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'win_loss_ratio': round(win_loss_ratio, 2),
        'profit_factor': round(profit_factor, 2),
        'expectancy': round(expectancy, 2),
        'avg_holding_days': round(avg_holding_days, 1),
        'avg_holding_bars': round(avg_holding_bars, 1),
        'max_drawdown': round(max_dd, 2),
        'total_return': round(total_return, 2),
        'sharpe': round(sharpe, 2),
    }

    # yearly returns
    for year in all_years:
        yr_trades = trades_df[trades_df['ENTRY_DATE'].dt.year == year]
        if len(yr_trades) > 0:
            metrics[f'return_{year}'] = round(yr_trades['PRICE_DIFF_PCT'].sum(), 2)
        else:
            metrics[f'return_{year}'] = 0.0

    return metrics


def run_strategy(data, indicator_fn, strategy_name):
    """Run a strategy across all tickers. Returns (trades_df, overlaps)."""
    all_trades = []
    overlaps = []

    for ticker, gdf in data.groupby('TICKER'):
        gdf = gdf.sort_values('DATE').reset_index(drop=True)

        try:
            buy_mask, sell_mask = indicator_fn(gdf)
            buy_mask = buy_mask.fillna(False).astype(bool)
            sell_mask = sell_mask.fillna(False).astype(bool)
        except Exception as e:
            continue

        # check for overlapping signals
        overlap = buy_mask & sell_mask
        if overlap.any():
            for idx in gdf.index[overlap]:
                overlaps.append(f"  {ticker} | {gdf.loc[idx, 'DATE']} | {strategy_name}")

        # build signal series
        signals = pd.Series(0, index=gdf.index)
        signals[buy_mask] = 1
        signals[sell_mask] = -1
        signals[overlap] = 0  # neutralize overlapping signals

        trades = generate_trades(gdf, signals)
        all_trades.extend(trades)

    trades_df = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()
    return trades_df, overlaps
