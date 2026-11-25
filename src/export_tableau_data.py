"""Export compact, Tableau-Public-ready CSVs from the retail churn SQLite database.

Builds three aggregated datasets in outputs/tableau/ (without duplicating the 794K raw lines):
  1. sales_orders.csv    — invoice-level purchases (36.6K rows) for exact KPIs, YoY monthly
                           trends, and top countries with Date & Country cross-filtering.
  2. sales_products.csv  — Month x Country x Product revenue (117K rows) with canonical product
                           descriptions so Date & Country filters update Top 10 Products too.
  3. scored_customers.csv — test-snapshot customers (2,775 rows) with XGBoost churn risk score,
                           risk decile, RFM quintiles/segments, last order date, and total spend.

Usage:
  python src/export_tableau_data.py
  python src/export_tableau_data.py --data data/raw/online_retail_II.xlsx
"""
import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.churn_pipeline import (  # noqa: E402
    DB_PATH,
    ORIGIN,
    TEST_CUTOFF,
    TRAIN_CUTOFFS,
    assign_risk_deciles,
    build_db,
    clean,
    fit_and_score,
    load_raw,
    snapshot_features,
)

DEFAULT_RAW = ROOT / "data" / "raw" / "online_retail_II.xlsx"
DEFAULT_OUT = ROOT / "outputs" / "tableau"

SALES_ORDERS_SQL = """
SELECT invoice,
       date(invoice_date)                                         AS order_date,
       strftime('%Y-%m-01', invoice_date)                         AS order_month,
       CAST(strftime('%Y', invoice_date) AS INTEGER)              AS year,
       CAST(strftime('%m', invoice_date) AS INTEGER)              AS month_num,
       customer_id,
       country,
       ROUND(SUM(line_amount), 2)                                 AS order_value,
       COUNT(DISTINCT stock_code)                                 AS n_items,
       SUM(quantity)                                              AS total_units
FROM transactions
WHERE is_return = 0
GROUP BY invoice, date(invoice_date), strftime('%Y-%m-01', invoice_date), customer_id, country
ORDER BY order_date, invoice;
"""

CANONICAL_DESC_SQL = """
WITH desc_counts AS (
    SELECT stock_code,
           TRIM(description) AS description,
           COUNT(*)          AS cnt
    FROM transactions
    WHERE is_return = 0
      AND description IS NOT NULL
      AND TRIM(description) != ''
    GROUP BY stock_code, TRIM(description)
),
ranked_desc AS (
    SELECT stock_code,
           description,
           ROW_NUMBER() OVER (
               PARTITION BY stock_code
               ORDER BY cnt DESC, description ASC
           ) AS rn
    FROM desc_counts
)
SELECT stock_code, description
FROM ranked_desc
WHERE rn = 1;
"""

SALES_PRODUCTS_SQL = """
SELECT strftime('%Y-%m-01', invoice_date)            AS order_month,
       CAST(strftime('%Y', invoice_date) AS INTEGER) AS year,
       CAST(strftime('%m', invoice_date) AS INTEGER) AS month_num,
       country,
       stock_code,
       ROUND(SUM(line_amount), 2)                    AS revenue,
       SUM(quantity)                                 AS units_sold,
       COUNT(DISTINCT invoice)                       AS order_count
FROM transactions
WHERE is_return = 0
GROUP BY strftime('%Y-%m-01', invoice_date), country, stock_code
ORDER BY order_month, country, revenue DESC;
"""

CUSTOMER_COUNTRY_SQL = """
WITH latest_tx AS (
    SELECT customer_id,
           country,
           ROW_NUMBER() OVER (
               PARTITION BY customer_id
               ORDER BY day_num DESC, invoice DESC
           ) AS rn
    FROM transactions
    WHERE is_return = 0
      AND day_num < :cutoff
)
SELECT customer_id, country
FROM latest_tx
WHERE rn = 1;
"""

MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}


def assign_rfm_segment(r: int, f: int, m: int) -> str:
    """Map RFM quintile scores (5 = best) to mutually-exclusive business segments."""
    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    if r >= 3 and f >= 4:
        return "Loyal Customers"
    if r >= 4 and f in (2, 3):
        return "Potential Loyalists"
    if r >= 4 and f == 1:
        return "Recent Customers"
    if r == 3 and f <= 3:
        return "Need Attention"
    if r <= 2 and f >= 4:
        return "Can't Lose Them"
    if r == 2 and f <= 3:
        return "At Risk"
    return "Hibernating"


def assign_risk_tier(decile: int) -> str:
    """Group risk deciles (1 = highest risk) into three retention priority tiers."""
    if decile <= 2:
        return "High Risk (Deciles 1-2)"
    if decile <= 7:
        return "Medium Risk (Deciles 3-7)"
    return "Low Risk (Deciles 8-10)"


def open_or_build_db(db_path: Path, data_path: Path) -> sqlite3.Connection:
    """Open an existing SQLite DB with descriptions, or rebuild it from raw data."""
    if db_path.exists():
        con = sqlite3.connect(db_path)
        cols = {row[1] for row in con.execute("PRAGMA table_info(transactions)")}
        if {"description", "country"}.issubset(cols):
            return con
        con.close()
    if not data_path.exists():
        raise FileNotFoundError(
            f"Database {db_path} missing required columns or absent, and raw file {data_path} not found."
        )
    print(f"Building {db_path} from {data_path}...")
    raw = load_raw(data_path)
    tx, _ = clean(raw)
    return build_db(tx)


def export_sales_tables(con: sqlite3.Connection, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Export invoice-grain sales_orders.csv and month-country-product sales_products.csv."""
    orders = pd.read_sql_query(SALES_ORDERS_SQL, con)
    orders["month_name"] = orders["month_num"].map(MONTH_NAMES)
    orders["month_status"] = orders["order_month"].map(
        lambda m: "Partial (Dec 1-9)" if m == "2011-12-01" else "Complete"
    )
    orders = orders[
        [
            "invoice", "order_date", "order_month", "year", "month_num", "month_name",
            "month_status", "customer_id", "country", "order_value", "n_items", "total_units",
        ]
    ]
    orders.to_csv(out_dir / "sales_orders.csv", index=False)

    products = pd.read_sql_query(SALES_PRODUCTS_SQL, con)
    desc_map = pd.read_sql_query(CANONICAL_DESC_SQL, con)
    products = products.merge(desc_map, on="stock_code", how="left")
    products["description"] = products["description"].fillna(products["stock_code"])
    products["product_label"] = products["description"] + " (" + products["stock_code"] + ")"
    products = products[
        [
            "order_month", "year", "month_num", "country", "stock_code",
            "description", "product_label", "revenue", "units_sold", "order_count",
        ]
    ]
    products.to_csv(out_dir / "sales_products.csv", index=False)
    return orders, products


def export_scored_customers(con: sqlite3.Connection, out_dir: Path) -> pd.DataFrame:
    """Train models on earlier snapshots and export scored test-snapshot customers."""
    train = pd.concat([snapshot_features(con, c) for c in TRAIN_CUTOFFS], ignore_index=True)
    test = snapshot_features(con, TEST_CUTOFF)
    _, _, gbm_name, scores = fit_and_score(train, test)

    cutoff_day = int((pd.Timestamp(TEST_CUTOFF) - ORIGIN).days)
    countries = pd.read_sql_query(
        CUSTOMER_COUNTRY_SQL, con, params={"cutoff": cutoff_day}
    )
    scored = test.merge(countries, on="customer_id", how="left")
    scored["country"] = scored["country"].fillna("Unknown")
    scored["snapshot_date"] = TEST_CUTOFF
    scored["predicted_risk"] = scores[gbm_name]
    scored["risk_decile"] = assign_risk_deciles(scored["predicted_risk"].to_numpy())
    scored["risk_tier"] = scored["risk_decile"].map(assign_risk_tier)
    scored["total_spend"] = scored["total_spend"].round(2)
    scored["avg_order_value"] = scored["raw_avg_order_value"].round(2)

    scored = scored.sort_values(
        by=["predicted_risk", "total_spend", "customer_id"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    scored["risk_rank"] = scored.index + 1
    scored["is_top50_at_risk"] = scored["risk_rank"] <= 50

    scored["rfm_code"] = (
        scored["r_score"].astype(str)
        + scored["f_score"].astype(str)
        + scored["m_score"].astype(str)
    )
    scored["rfm_score_sum"] = scored["r_score"] + scored["f_score"] + scored["m_score"]
    scored["rfm_segment"] = [
        assign_rfm_segment(int(r), int(f), int(m))
        for r, f, m in zip(scored["r_score"], scored["f_score"], scored["m_score"])
    ]

    cols = [
        "customer_id", "country", "snapshot_date", "risk_rank", "is_top50_at_risk",
        "predicted_risk", "risk_decile", "risk_tier", "churned", "last_order_date",
        "recency_days", "n_orders", "total_spend", "avg_order_value",
        "r_score", "f_score", "m_score", "rfm_code", "rfm_score_sum", "rfm_segment",
    ]
    scored = scored[cols]
    scored.to_csv(out_dir / "scored_customers.csv", index=False)
    return scored


def main():
    ap = argparse.ArgumentParser(description="Export aggregated CSVs for Tableau Public")
    ap.add_argument("--db", type=Path, default=DB_PATH, help="path to SQLite DB")
    ap.add_argument("--data", type=Path, default=DEFAULT_RAW, help="fallback raw Excel/CSV path")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="output directory for CSVs")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    con = open_or_build_db(args.db, args.data)
    try:
        orders, products = export_sales_tables(con, args.out_dir)
        scored = export_scored_customers(con, args.out_dir)
    finally:
        con.close()

    print(f"Exported Tableau CSVs to {args.out_dir}:")
    print(f"  sales_orders.csv:     {len(orders):,} rows | total revenue £{orders['order_value'].sum():,.2f}")
    print(f"  sales_products.csv:   {len(products):,} rows | total revenue £{products['revenue'].sum():,.2f}")
    print(f"  scored_customers.csv: {len(scored):,} rows | top 50 min risk = {scored.head(50)['predicted_risk'].min():.3f}")


if __name__ == "__main__":
    main()
