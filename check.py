import os
from pathlib import Path

import duckdb
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
db_path = os.getenv("DB_PATH", "market.duckdb")

con = duckdb.connect(db_path)

daily = con.execute(
    "select min(dt), max(dt) from daily_bars"
).fetchone()

minute = con.execute(
    "select min(ts), max(ts) from minute_bars"
).fetchone()

print(f"DB      : {db_path}")
print("Daily range :", daily[0], "→", daily[1])
print("Minute range:", minute[0], "→", minute[1])
