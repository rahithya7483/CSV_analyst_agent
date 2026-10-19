"""Generate the synthetic sales dataset used for evaluation (fixed seed -> reproducible).

Run from the project root:  python scripts/generate_sample_sales.py
"""
import csv
import random
from datetime import date, timedelta
from pathlib import Path

random.seed(42)

REGIONS = ["North", "South", "East", "West"]
CATEGORIES = {"Laptop": (55000, 90000), "Phone": (15000, 60000), "Tablet": (20000, 45000),
              "Monitor": (8000, 25000), "Headphones": (1500, 9000)}
SEGMENTS = ["Retail", "Corporate", "Education"]
SEGMENT_WEIGHTS = [0.5, 0.3, 0.2]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
ROWS = 240

out = Path(__file__).resolve().parent.parent / "data" / "sales.csv"
start = date(2024, 1, 1)
with out.open("w", newline="") as fh:
    writer = csv.writer(fh, lineterminator="\n")
    writer.writerow(["order_id", "order_date", "month", "region", "category",
                     "segment", "units", "unit_price", "revenue"])
    for i in range(1, ROWS + 1):
        d = start + timedelta(days=random.randint(0, 365))
        category = random.choice(list(CATEGORIES))
        low, high = CATEGORIES[category]
        unit_price = round(random.randint(low, high), -2)
        units = random.randint(1, 30)
        writer.writerow([f"ORD-{1000 + i}", d.isoformat(), MONTHS[d.month - 1],
                         random.choice(REGIONS), category,
                         random.choices(SEGMENTS, SEGMENT_WEIGHTS)[0],
                         units, unit_price, units * unit_price])
print(f"wrote {ROWS} rows to {out}")
