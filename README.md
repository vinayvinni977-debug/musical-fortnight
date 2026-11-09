# RSI Divergence Paper Bot

Paper-trading only. No broker connection, no real orders, no API keys.

- `paper_bot.py` - the bot (`python paper_bot.py` loops; `--once` checks one candle)
- `backtest.py` - backtest with costs and out-of-sample split
- `dashboard.py` - live terminal view of state, trades and decision log
- `.github/workflows/paper-bot.yml` - runs the bot every 15 min in NSE hours
- `.github/workflows/backtest.yml` - manual backtest, results shown in the run summary

State is saved to `state.json`, `trades.csv`, `decisions.log` and committed
back by the workflow. Keep this repo PRIVATE.

Not financial advice. Backtest first; judge only out-of-sample results.
