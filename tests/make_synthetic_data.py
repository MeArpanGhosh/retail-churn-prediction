"""Create a SYNTHETIC file shaped like UCI Online Retail II, for smoke-testing the pipeline.

This is NOT real data. Never report results obtained from it.
Usage:  python tests/make_synthetic_data.py data/raw/synthetic_test.xlsx
"""
import sys
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
START = pd.Timestamp("2009-12-01")
END_DAY = 739                                   # 2011-12-09
N_CUST, N_PRODUCTS = 2500, 600

rows, inv_no = [], 500000
prices = rng.lognormal(0.9, 0.6, N_PRODUCTS).round(2) + 0.2
for cid in range(12000, 12000 + N_CUST):
    join = int(rng.uniform(0, 620))
    rate = rng.lognormal(-3.3, 0.6)               # orders per day while active
    life = rng.exponential(260)                   # days until the customer stops buying
    day = join
    while day < min(join + life, END_DAY):
        inv_no += 1
        n_lines = 1 + rng.poisson(5)
        ts = START + pd.Timedelta(days=int(day), seconds=int(rng.integers(28800, 64800)))
        for sc in rng.choice(N_PRODUCTS, n_lines, replace=False):
            rows.append((str(inv_no), str(10000 + sc), f"ITEM {sc}", 1 + rng.poisson(5),
                         ts, prices[sc], float(cid), "United Kingdom"))
        if rng.random() < 0.04:                   # a credit note a few days later
            inv_no += 1
            rows.append((f"C{inv_no}", str(10000 + sc), f"ITEM {sc}", -2,
                         ts + pd.Timedelta(days=3), prices[sc], float(cid), "United Kingdom"))
        day += 1 + rng.exponential(1 / rate)

df = pd.DataFrame(rows, columns=["Invoice", "StockCode", "Description", "Quantity",
                                 "InvoiceDate", "Price", "Customer ID", "Country"])
# add the kinds of mess found in the real file
df.loc[df.sample(frac=0.03, random_state=1).index, "Customer ID"] = np.nan
df.loc[df.sample(frac=0.005, random_state=2).index, "Price"] = 0.0
fees = df.sample(frac=0.004, random_state=3).copy()
fees["StockCode"] = "POST"
df = pd.concat([df, fees], ignore_index=True).sort_values("InvoiceDate").reset_index(drop=True)

split = pd.Timestamp("2010-12-09 23:59:59")
sheet1 = df[df["InvoiceDate"] <= split]
sheet2 = df[df["InvoiceDate"] >= pd.Timestamp("2010-12-01")]   # overlapping week, as in the real file
out = sys.argv[1] if len(sys.argv) > 1 else "synthetic_test.xlsx"
with pd.ExcelWriter(out) as xw:
    sheet1.to_excel(xw, sheet_name="Year 2009-2010", index=False)
    sheet2.to_excel(xw, sheet_name="Year 2010-2011", index=False)
print(f"wrote {out}: {len(sheet1):,} + {len(sheet2):,} rows")
