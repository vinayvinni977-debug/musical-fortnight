"""
Backtest for paper_bot.py (same signal, same sizing, same stops) with costs
and an in-sample / out-of-sample split.

Usage:
  python backtest.py                                   # Nifty, 1h, 2 years
  python backtest.py --symbol RELIANCE.NS --interval 1h --period 730d --cost 0.05
  python backtest.py --interval 1d --period 10y

--cost = % of trade value charged PER SIDE (brokerage + STT + slippage).
Intraday index/stock trades are often ~0.03-0.10% per side; check your broker.

Yahoo limits: 15m = last 60 days, 1h = last 730 days, 1d = many years.
"""
import argparse
import itertools
import numpy as np
import pandas as pd
import yfinance as yf
import paper_bot as pb

CFG = pb.CFG


def load(symbol, interval, period):
    df = yf.download(symbol, interval=interval, period=period,
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna().copy()
    df["RSI"] = pb.rsi(df["Close"], CFG["rsi_len"])
    df["ATR"] = pb.atr(df, CFG["atr_len"])
    return df.dropna()


def run(df, start, end, k, sl_atr, rr, cost_pct, capital=10000.0, trend=False):
    """Simulate bars [start, end). Returns (trades, equity_curve)."""
    CFG["pivot_k"] = k
    cost = cost_pct / 100
    eq, pos = capital, None
    trades, curve = [], []
    warm = CFG["lookback"] + 2 * k + 2
    hi, lo, op, cl, at = (df[c].values for c in ("High", "Low", "Open", "Close", "ATR"))
    ema = df["Close"].ewm(span=200, adjust=False).mean().values

    def close(price):
        nonlocal eq, pos
        sign = 1 if pos["side"] == "long" else -1
        gross = sign * (price - pos["entry"]) * pos["qty"]
        fees = cost * pos["qty"] * (pos["entry"] + price)
        pnl = gross - fees
        eq += pnl
        trades.append((pnl, pnl / pos["risk"]))
        pos = None

    for i in range(max(start, warm), end):
        if pos:
            if pos["side"] == "long":
                if lo[i] <= pos["sl"]:
                    close(min(pos["sl"], op[i]))      # gap through stop fills at open
                elif hi[i] >= pos["tp"]:
                    close(pos["tp"])
            else:
                if hi[i] >= pos["sl"]:
                    close(max(pos["sl"], op[i]))
                elif lo[i] <= pos["tp"]:
                    close(pos["tp"])
        if not pos:
            side = pb.signal(df, end=i + 1)
            if side and trend:  # only trade with the 200-EMA trend
                if i < 200 or (side == "long") != (cl[i] > ema[i]):
                    side = None
            if side:
                price, dist = cl[i], sl_atr * at[i]
                risk_amt = eq * CFG["risk_pct"] / 100
                qty = min(risk_amt / dist, eq / price)
                sign = 1 if side == "long" else -1
                pos = dict(side=side, entry=price, qty=qty, risk=risk_amt,
                           sl=price - sign * dist, tp=price + sign * dist * rr)
        curve.append(eq)
    return trades, curve


def stats(trades, curve, capital=10000.0):
    if not trades:
        return dict(trades=0, win=0, exp_R=0, pf=0, net=0, maxdd=0)
    pnl = np.array([t[0] for t in trades])
    r = np.array([t[1] for t in trades])
    wins, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    c = np.array(curve)
    dd = ((np.maximum.accumulate(c) - c) / np.maximum.accumulate(c)).max() * 100
    return dict(trades=len(pnl), win=(pnl > 0).mean() * 100, exp_R=r.mean(),
                pf=(wins / losses if losses else float("inf")),
                net=(c[-1] / capital - 1) * 100, maxdd=dd)


def show(label, s):
    print(f"{label:<22} trades {s['trades']:>4} | win {s['win']:5.1f}% | "
          f"exp {s['exp_R']:+.2f}R | PF {s['pf']:.2f} | "
          f"net {s['net']:+6.1f}% | maxDD {s['maxdd']:5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=CFG["symbol"])
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--period", default="730d")
    ap.add_argument("--cost", type=float, default=0.05, help="%% per side")
    ap.add_argument("--split", type=float, default=0.7, help="in-sample fraction")
    ap.add_argument("--min-trades", type=int, default=30)
    a = ap.parse_args()

    df = load(a.symbol, a.interval, a.period)
    n = len(df)
    cut = int(n * a.split)
    print(f"{a.symbol} {a.interval}: {n} bars | in-sample {cut} | out-of-sample {n - cut}")
    print(f"cost {a.cost}% per side\n")

    d = (CFG["pivot_k"], CFG["sl_atr"], CFG["rr"])
    t, c = run(df, 0, cut, *d, a.cost)
    show("DEFAULT in-sample", stats(t, c))
    t, c = run(df, cut, n, *d, a.cost)
    show("DEFAULT out-of-sample", stats(t, c))

    print("\nGrid search on IN-SAMPLE only (pivot_k, sl_atr, rr)...")
    best = None
    for k, sl, rr in itertools.product([2, 3, 4], [1.0, 1.5, 2.0], [1.5, 2.0, 3.0]):
        t, c = run(df, 0, cut, k, sl, rr, a.cost)
        s = stats(t, c)
        if s["trades"] >= a.min_trades and (best is None or s["exp_R"] > best[0]["exp_R"]):
            best = (s, (k, sl, rr))
    if not best:
        print("No setting reached the minimum trade count. Use more data or a lower interval.")
        return
    show(f"BEST in-sample {best[1]}", best[0])
    t, c = run(df, cut, n, *best[1], a.cost)
    so = stats(t, c)
    show("same params, OOS", so)

    print("\nHow to read this:")
    print("- Only the OUT-OF-SAMPLE lines count. In-sample results are tuned and flattering.")
    print("- Want: OOS expectancy > 0, PF > 1.2, 100+ trades, drawdown you could live with.")
    print("- If OOS is negative or far worse than in-sample, the 'edge' was overfit. Don't trade it.")
    if so["trades"] < 100:
        print(f"- Only {so['trades']} OOS trades: too few to trust either way.")


if __name__ == "__main__":
    main()
