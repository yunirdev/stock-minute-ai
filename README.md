# stock-minute-ai

**Real-time and historical US stock minute-bar data pipeline with DuckDB storage and an interactive Streamlit candlestick dashboard powered by [Alpaca Markets](https://alpaca.markets/).**

---

## Overview

`stock-minute-ai` is a lightweight, self-contained market-data toolkit that lets you:

- **Backfill** years of daily and minute OHLCV bars for any set of US stocks via the Alpaca historical data API.
- **Stream** live minute bars over a WebSocket connection and persist them on the fly.
- **Explore** your local data through an interactive Streamlit dashboard featuring candlestick charts, volume overlays, and selectable timeframes (1 m, 5 m, 30 m, 1 h, 1 d).

All data is stored in a single [DuckDB](https://duckdb.org/) file, which means zero infrastructure — no separate database server required.

---

## Architecture

```
Alpaca Markets API
      │
      ├─ Historical (REST)  ──► ingest/history.py  ──► DuckDB (minute_bars, daily_bars)
      │
      └─ Real-time (WS)     ──► ingest/alpaca_stream.py ──► DuckDB (minute_bars)
                                                                  │
                                                            app/ui.py (Streamlit)
```

---

## Requirements

- Python ≥ 3.13
- An [Alpaca Markets](https://alpaca.markets/) account with an API key and secret (free tier works with the `iex` data feed)

---

## Setup

1. **Clone the repository and install dependencies** (using [uv](https://github.com/astral-sh/uv)):

   ```bash
   git clone https://github.com/yunirdev/stock-minute-ai.git
   cd stock-minute-ai
   uv sync
   ```

2. **Create a `.env` file** in the project root:

   ```dotenv
   ALPACA_API_KEY=your_key_here
   ALPACA_API_SECRET=your_secret_here

   # Data feed: iex (free), sip (paid), or delayed_sip
   ALPACA_DATA_FEED=iex

   # Comma-separated list of symbols to track
   SYMBOLS=AAPL,MSFT,NVDA

   # Historical backfill start date
   HISTORY_START=2020-01-01

   # Number of days per API request chunk (reduce if you hit rate limits)
   HISTORY_CHUNK_DAYS=20

   # Path to the DuckDB database file
   DB_PATH=market.duckdb
   ```

---

## Usage

### 1. Backfill historical data

Downloads daily and minute bars from `HISTORY_START` through today and stores them in DuckDB:

```bash
python -m ingest.history
```

### 2. Launch the interactive dashboard

Starts a Streamlit app that streams live bars and visualises both historical and real-time data:

```bash
streamlit run app/ui.py
```

Open the URL shown in your terminal (usually `http://localhost:8501`).

### 3. Verify the database

Quick sanity-check to confirm the date ranges stored in DuckDB:

```bash
python check.py
```

---

## Project Structure

```
stock-minute-ai/
├── app/
│   └── ui.py               # Streamlit dashboard (candlestick charts, live streaming)
├── ingest/
│   ├── alpaca_stream.py    # WebSocket streamer — writes live minute bars to DuckDB
│   └── history.py          # Historical backfill — daily & minute bars via REST API
├── check.py                # Prints the date range stored in DuckDB
├── fix_source.py           # One-time migration: backfills the `source` column
├── pyproject.toml          # Project metadata and dependencies
└── .env                    # API credentials and runtime configuration (not committed)
```

---

## License

This project is provided as-is for educational and personal use.
