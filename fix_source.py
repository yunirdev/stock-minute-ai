import duckdb

con = duckdb.connect("data.duckdb")
con.execute("UPDATE minute_bars SET source='alpaca' WHERE source IS NULL")
con.close()

print("source fixed")