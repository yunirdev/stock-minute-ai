import duckdb

con = duckdb.connect("market.duckdb")

daily = con.execute(
    "select min(dt), max(dt) from daily_bars"
).fetchone()

minute = con.execute(
    "select min(ts), max(ts) from minute_bars"
).fetchone()

print("Daily range :", daily[0], "→", daily[1])
print("Minute range:", minute[0], "→", minute[1])