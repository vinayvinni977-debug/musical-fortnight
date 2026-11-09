"""
Terminal dashboard for paper_bot.py  (run in a second terminal/tab)
  pip install rich
  python dashboard.py          # live, refreshes every 2s
  python dashboard.py --once   # print one frame and exit
Reads state.json, trades.csv and decisions.log written by the bot.
"""
import csv, json, os, sys, time
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

START = 10000.0  # keep equal to CFG["capital"] in paper_bot.py


def read_state():
    if os.path.exists("state.json"):
        return json.load(open("state.json"))
    return dict(equity=START, pos=None, last_bar=None, day={}, month={})


def read_trades():
    if not os.path.exists("trades.csv"):
        return []
    return list(csv.DictReader(open("trades.csv")))


def tail(path, n):
    if not os.path.exists(path):
        return []
    return open(path).read().splitlines()[-n:]


def render():
    s, tr = read_state(), read_trades()
    wins = sum(1 for t in tr if float(t["pnl"]) > 0)
    losses = len(tr) - wins
    eq = s["equity"]
    chg = (eq / START - 1) * 100
    col = "green" if chg >= 0 else "red"

    head = Table.grid(expand=True)
    head.add_column(ratio=2)
    for _ in range(4):
        head.add_column(justify="right", min_width=10)
    head.add_row(
        Text("PAPER BOT TERMINAL", style="bold cyan"),
        Text(f"${eq:,.2f}\nbalance", style=f"bold {col}"),
        Text(f"{len(tr)}\ntrades"),
        Text(f"{wins}\nwins", style="green"),
        Text(f"{losses}\nlosses", style="red"),
    )

    agents = Table.grid(padding=(0, 1))
    agents.add_row("SCANNER ", "reads latest closed candle")
    agents.add_row("SIGNAL  ", "RSI divergence on confirmed pivot")
    agents.add_row("SIZER   ", "1% equity risk, no leverage")
    agents.add_row("GUARD   ", "daily 3% / monthly 8% loss halt")
    agents.add_row("EXECUTOR", "paper fills, stop checked first")
    agents.add_row("LOGGER  ", "every decision has a reason")

    p = s["pos"]
    if p:
        pos_txt = (f"{p['side'].upper()}  entry {p['entry']:.2f}\n"
                   f"stop {p['sl']:.2f}   target {p['tp']:.2f}\n"
                   f"qty {p['qty']:.3f}")
    else:
        pos_txt = "flat - waiting for a signal"

    day = sum(v for k, v in s["day"].items() if k == max(s["day"], default=""))
    stats = (f"equity vs start  {chg:+.2f}%\n"
             f"today P&L        {day:+.2f}\n"
             f"last bar         {s['last_bar']}")

    log = Text("\n".join(tail("decisions.log", 14)) or "no decisions yet")

    lay = Layout()
    lay.split_column(Layout(Panel(head), size=6), Layout(name="mid", size=10), Layout(name="log"))
    lay["mid"].split_row(Layout(Panel(agents, title="AGENTS")),
                         Layout(Panel(pos_txt, title="OPEN POSITION")),
                         Layout(Panel(stats, title="EQUITY")))
    lay["log"].update(Panel(log, title="LOG (newest last)"))
    return lay


if __name__ == "__main__":
    if "--once" in sys.argv:
        Console().print(render())
    else:
        with Live(render(), refresh_per_second=1, screen=True) as live:
            while True:
                live.update(render())
                time.sleep(2)
