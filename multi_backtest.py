"""
Pooled multi-symbol backtest: SAME fixed rules on ~22 liquid symbols, no tuning.
Pools every trade so the sample is big enough to mean something.
Compares: baseline vs 200-EMA trend filter, in-sample vs out-of-sample.

  python multi_backtest.py                       # 1h, 730d, 0.05% cost per side
  python multi_backtest.py --cost 0.10
"""
import argparse
import numpy as np
import backtest as bt
import paper_bot as pb

SYMBOLS = ["^NSEI", "^NSEBANK", "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS",
           "ICICIBANK.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS", "AXISBANK.NS",
           "KOTAKBANK.NS", "HINDUNILVR.NS", "MARUTI.NS", "SUNPHARMA.NS", "TATASTEEL.NS",
           "BAJFINANCE.NS", "ASIANPAINT.NS", "NTPC.NS", "ONGC.NS", "WIPRO.NS"]


def pooled_stats(rs, pnls):
    n = len(rs)
    if n == 0:
        return None
    rs, pnls = np.array(rs), np.array(pnls)
    g, l = pnls[pnls > 0].sum(), -pnls[pnls < 0].sum()
    se = rs.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    return dict(n=n, win=(pnls > 0).mean() * 100, exp=rs.mean(), pf=(g / l if l else float("inf")),
                lo=rs.mean() - 1.96 * se, hi=rs.mean() + 1.96 * se)


def line(label, s):
    if not s:
        return f"{label:<26} no trades"
    return (f"{label:<26} trades {s['n']:>4} | win {s['win']:4.1f}% | exp {s['exp']:+.2f}R "
            f"(95% range {s['lo']:+.2f} to {s['hi']:+.2f}) | PF {s['pf']:.2f}")


def evaluate(dfs, cost, split, trend, d):
    out = {"IS": ([], []), "OOS": ([], [])}
    for df in dfs.values():
        n = len(df)
        cut = int(n * split)
        for name, (a, b) in (("IS", (0, cut)), ("OOS", (cut, n))):
            t, _ = bt.run(df, a, b, *d, cost, trend=trend)
            out[name][0].extend(x[1] for x in t)
            out[name][1].extend(x[0] for x in t)
    return {k: pooled_stats(*v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--period", default="730d")
    ap.add_argument("--cost", type=float, default=0.05)
    ap.add_argument("--split", type=float, default=0.7)
    a = ap.parse_args()

    dfs = {}
    for sym in SYMBOLS:
        try:
            df = bt.load(sym, a.interval, a.period)
            if len(df) > 500:
                dfs[sym] = df
        except Exception as e:
            print("skip", sym, "-", e)
    print(f"{len(dfs)} symbols loaded | {a.interval} {a.period} | cost {a.cost}% per side")
    print("Fixed default rules, nothing tuned per symbol.\n")

    d = (pb.CFG["pivot_k"], pb.CFG["sl_atr"], pb.CFG["rr"])
    res = {}
    for label, trend in (("BASELINE", False), ("WITH 200-EMA TREND", True)):
        r = evaluate(dfs, a.cost, a.split, trend, d)
        res[label] = r
        print(line(f"{label} in-sample", r["IS"]))
        print(line(f"{label} OUT-of-sample", r["OOS"]))
        print()

    print("How to read this:")
    print("- Only OUT-of-sample lines count. Want 100+ trades, PF > 1.2, and the whole")
    print("  95% range above 0. If the range includes 0, the edge is not proven.")
    print("- If you test more variants, expect one to look good by luck. Prefer the")
    print("  simplest version, and re-test it on fresh data before trusting it.")
    best = max(res.values(), key=lambda r: (r["OOS"] or {"exp": -9})["exp"])["OOS"]
    if not best or best["n"] < 100 or best["lo"] <= 0 or best["pf"] < 1.2:
        print("\nVERDICT: no proven edge. Do not trade this with real money.")
    else:
        print("\nVERDICT: passes the basic bar. Next step is a 2-3 month paper run, not real money.")


if __name__ == "__main__":
    main()
