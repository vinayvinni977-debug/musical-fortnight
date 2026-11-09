"""
RSI Divergence PAPER trading bot (no real money, no broker orders).
Install:  pip install yfinance pandas numpy
Run:      python paper_bot.py
Logs trades to trades.csv, keeps state in state.json (safe to restart).
Go live only after 2-3 months of paper results you trust.
"""
import csv, json, os, time
import numpy as np
import pandas as pd
import yfinance as yf

CFG = dict(
    symbol="^NSEI",          # Nifty 50 index; change to any Yahoo symbol
    interval="15m", period="5d",
    capital=10000.0,         # paper capital (use only money you can lose)
    risk_pct=1.0,            # risk per trade, % of equity
    max_daily_loss_pct=3.0,  # halt for the day
    max_month_loss_pct=8.0,  # halt for the month
    rsi_len=14, atr_len=14,
    pivot_k=3, lookback=60,  # pivot width, bars to search for prior pivot
    sl_atr=1.5, rr=2.0,      # stop = 1.5 ATR, target = 2R
    poll_sec=60,
    log="trades.csv", state="state.json",
)


# ---------- indicators ----------
def rsi(close, n):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def atr(df, n):
    pc = df["Close"].shift()
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - pc).abs(),
                    (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def is_pivot(series, i, k, low=True):
    w = series.iloc[i - k:i + k + 1]
    return series.iloc[i] == (w.min() if low else w.max())


def signal(df, end=None):
    """Bullish/bearish RSI divergence on the pivot that just got confirmed."""
    k, lb = CFG["pivot_k"], CFG["lookback"]
    n = len(df) if end is None else end
    p2 = n - 1 - k
    if p2 - lb < k:
        return None
    r = df["RSI"]
    for low, side in ((True, "long"), (False, "short")):
        px = df["Low"] if low else df["High"]
        if not is_pivot(px, p2, k, low):
            continue
        for p1 in range(p2 - 2 * k, max(k, p2 - lb), -1):
            if is_pivot(px, p1, k, low):
                if low and px.iloc[p2] < px.iloc[p1] and r.iloc[p2] > r.iloc[p1] and r.iloc[p1] < 40:
                    return side
                if not low and px.iloc[p2] > px.iloc[p1] and r.iloc[p2] < r.iloc[p1] and r.iloc[p1] > 60:
                    return side
                break
    return None


# ---------- state & logging ----------
def load_state():
    if os.path.exists(CFG["state"]):
        return json.load(open(CFG["state"]))
    return dict(equity=CFG["capital"], pos=None, last_bar=None, day={}, month={})


def save_state(s):
    json.dump(s, open(CFG["state"], "w"), indent=2)


def log_trade(row):
    new = not os.path.exists(CFG["log"])
    with open(CFG["log"], "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "side", "entry", "exit", "qty", "pnl", "reason", "equity"])
        w.writerow(row)


def note(ts, kind, text):
    line = f"{ts} {kind:<6} {text}"
    print(line)
    with open("decisions.log", "a") as f:
        f.write(line + "\n")


def halted(s, ts):
    d, m = s["day"].get(str(ts.date()), 0.0), s["month"].get(ts.strftime("%Y-%m"), 0.0)
    return (d <= -CFG["capital"] * CFG["max_daily_loss_pct"] / 100 or
            m <= -CFG["capital"] * CFG["max_month_loss_pct"] / 100)


# ---------- trading logic ----------
def close_pos(s, ts, price, reason):
    p = s["pos"]
    sign = 1 if p["side"] == "long" else -1
    pnl = sign * (price - p["entry"]) * p["qty"]
    s["equity"] += pnl
    dk, mk = str(ts.date()), ts.strftime("%Y-%m")
    s["day"][dk] = s["day"].get(dk, 0.0) + pnl
    s["month"][mk] = s["month"].get(mk, 0.0) + pnl
    log_trade([str(ts), p["side"], p["entry"], round(price, 2), round(p["qty"], 4),
               round(pnl, 2), reason, round(s["equity"], 2)])
    note(ts, "EXIT", f"{p['side']} @ {price:.2f} pnl {pnl:+.2f} | reason: {reason} | equity {s['equity']:.2f}")
    s["pos"] = None


def on_bar(s, df):
    bar, ts = df.iloc[-1], df.index[-1]
    p = s["pos"]
    if p:  # stop checked first (conservative)
        if p["side"] == "long":
            if bar["Low"] <= p["sl"]:
                close_pos(s, ts, p["sl"], "stop")
            elif bar["High"] >= p["tp"]:
                close_pos(s, ts, p["tp"], "target")
        else:
            if bar["High"] >= p["sl"]:
                close_pos(s, ts, p["sl"], "stop")
            elif bar["Low"] <= p["tp"]:
                close_pos(s, ts, p["tp"], "target")
    if s["pos"]:
        return
    side = signal(df)
    if not side:
        return
    if halted(s, ts):
        note(ts, "SKIP", f"{side} signal ignored | reason: daily/monthly loss limit hit")
        return
    price, a = bar["Close"], df["ATR"].iloc[-1]
    stop_dist = CFG["sl_atr"] * a
    qty = s["equity"] * CFG["risk_pct"] / 100 / stop_dist
    qty = min(qty, s["equity"] / price)  # no leverage
    sign = 1 if side == "long" else -1
    s["pos"] = dict(side=side, entry=float(price), qty=float(qty),
                    sl=float(price - sign * stop_dist),
                    tp=float(price + sign * stop_dist * CFG["rr"]))
    note(ts, "ENTER", f"{side} @ {price:.2f} | size {qty * price / s['equity'] * 100:.0f}% eq | "
         f"sl {s['pos']['sl']:.2f} tp {s['pos']['tp']:.2f} | reason: RSI divergence confirmed")


def fetch():
    df = yf.download(CFG["symbol"], interval=CFG["interval"], period=CFG["period"],
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna().iloc[:-1]  # drop the still-forming candle
    df["RSI"] = rsi(df["Close"], CFG["rsi_len"])
    df["ATR"] = atr(df, CFG["atr_len"])
    return df.dropna()


def main():
    import sys
    s = load_state()
    if "--once" in sys.argv:  # single check, for GitHub Actions / cron
        df = fetch()
        ts = str(df.index[-1])
        if ts != s["last_bar"]:
            s["last_bar"] = ts
            on_bar(s, df)
            save_state(s)
        else:
            print("no new candle")
        return
    print("Paper bot running. Ctrl+C to stop.")
    while True:
        try:
            df = fetch()
            ts = str(df.index[-1])
            if ts != s["last_bar"]:
                s["last_bar"] = ts
                on_bar(s, df)
                save_state(s)
        except Exception as e:
            print("error:", e)
        time.sleep(CFG["poll_sec"])


if __name__ == "__main__":
    main()
