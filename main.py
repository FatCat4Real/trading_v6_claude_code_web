import pandas as pd
from indicators import INDICATORS
from backtest import run_strategy, calc_metrics


def main():
    print("Loading data...")
    data = pd.read_parquet('data/set100_stocks_2020_2025.parquet')
    print(f"Loaded {len(data):,} rows, {data['TICKER'].nunique()} tickers")
    print(f"Date range: {data['DATE'].min().date()} to {data['DATE'].max().date()}")

    all_years = sorted(data['DATE'].dt.year.unique())
    print(f"Years: {all_years}\n")

    results = {}
    all_overlaps = []

    for name, (fn, source) in INDICATORS.items():
        label = f"{name} [{source}]"
        print(f"  Running {name}...", end=" ", flush=True)

        try:
            trades_df, overlaps = run_strategy(data, fn, name)
            all_overlaps.extend(overlaps)

            if not trades_df.empty:
                metrics = calc_metrics(trades_df, all_years)
                results[label] = metrics
                print(f"{metrics['n_trades']} trades, "
                      f"win_rate={metrics['win_rate']}%, "
                      f"total_return={metrics['total_return']}%")
            else:
                print("No trades generated")
        except Exception as e:
            print(f"ERROR: {e}")

    # overlap warnings
    if all_overlaps:
        print(f"\n{'='*60}")
        print(f"OVERLAPPING SIGNALS ({len(all_overlaps)} found)")
        print(f"{'='*60}")
        for o in all_overlaps[:20]:  # show first 20
            print(o)
        if len(all_overlaps) > 20:
            print(f"  ... and {len(all_overlaps) - 20} more")

    # build results table (indicators as columns, metrics as index)
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}\n")

    results_df = pd.DataFrame(results)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_colwidth', 25)
    pd.set_option('display.float_format', '{:.2f}'.format)

    print(results_df.to_string())

    # save to CSV
    results_df.to_csv('results.csv')
    print("\n\nResults saved to results.csv")


if __name__ == '__main__':
    main()
