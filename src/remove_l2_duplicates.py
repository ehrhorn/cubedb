from datetime import datetime
from pathlib import Path
import pickle
import sqlite3

duplicate_file = Path().home() / "duplicates.pkl"
db_file = Path().home() / "data" / "local_meta.db"
print(f"{datetime.now()}: Opening duplicates file...")
with open(duplicate_file, "rb") as f:
    duplicate_events = pickle.load(f)

duplicate_events = ", ".join([str(event_no) for event_no in duplicate_events])
query = f"delete from meta where event_no in ({duplicate_events})"
with sqlite3.connect(str(db_file)) as con:
    cur = con.cursor()
    print(f"{datetime.now()}: Executing deletion query...")
    cur.execute(query)
print(f"{datetime.now()}: Done.")
