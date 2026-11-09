"""
Data quality check for the symbols used by multi_backtest.py.
  python check_data.py                      # 1h, 730d
Prints per-symbol: bars, date span, bars/day, duplicates, flat bars, big jumps, gaps.
"""
import argparse
import pandas as pd
import yfinance as yf
from multi_backtest import SYMBOLS


def check(sym, interval, period):
    df = yf.download(sym, interval=interval, period=period, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty:
        return f"{sym:<15} NO DATA"
    n = len(df)
    days = df.index.normalize().nunique()
    per_day = df.groupby(df.index.normalize()).size()
    dup = int(df.index.duplicated().sum())
    nan = int(df[["Open", "High", "Low", "Close"]].isna().any(axis=1).sum())
    flat = (df["High"] == df["Low"]).mean() * 100
    jump = (df["Close"].pct_change().abs() * 100).max()
    gap = df.index.to_series().diff().dt.days.max()
    flags = []
    if per_day.median() > 8 or per_day.max() > 12: flags.append("odd bars/day")
    if dup: flags.append("duplicates")
    if nan: flags.append("NaN rows")
    if flat > 2: flags.append("many flat bars")
    if jump > 12: flags.append("huge jump")
    if gap and gap > 5: flags.append("long gap")
    return (f"{sym:<15} bars {n:>5} | days {days:>3} | {str(df.index[0].date())} to {str(df.index[-1].date())} | "
            f"bars/day med {per_day.median():.0f} max {per_day.max()} | dup {dup} | flat {flat:.1f}% | "
            f"maxJump {jump:.1f}% | maxGap {gap}d | {', '.join(flags) or 'OK'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--period", default="730d")
    a = ap.parse_args()
    for s in SYMBOLS:
        try:
            print(check(s, a.interval, a.period))
        except Exception as e:
            print(f"{s:<15} ERROR {e}")
