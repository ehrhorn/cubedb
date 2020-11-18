from datetime import datetime
from pathlib import Path
from cubedb.save_random_events import random_events_saver

datasets_root = Path().home() / "work" / "datasets"
datasets = [f for f in datasets_root.iterdir() if f.is_dir()]
print(f"{datetime.now()}: Looking through datasets")
for dataset in datasets:
    powershovel_db = dataset / "meta" / "powershovel_events.db"
    if not powershovel_db.is_file():
        print(
            f"""
{datetime.now()}: {dataset.name} does not have Powershovel events.
Creating..."""
        )
        random_events_saver(dataset.name)
    else:
        print(
            f"""
{datetime.now()}: {dataset.name} has Powershovel events.
Skipping"""
        )