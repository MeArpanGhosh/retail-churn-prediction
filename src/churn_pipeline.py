"""Customer churn prediction for an online retailer (UCI Online Retail II).

Data source: https://archive.ics.uci.edu/dataset/502/online+retail+ii
Citation: Chen, D. (2012). Online Retail II [Dataset]. UCI ML Repository. CC BY 4.0.

End-to-end pipeline that:
  1. Loads and cleans raw transactions (every removal is logged for auditability).
  2. Loads them into SQLite and builds customer features + churn label with SQL
     (CTEs and window functions — see sql/features.sql) for several snapshot dates.
  3. Trains on earlier snapshots, tests on a LATER snapshot (time-based split, no leakage).
  4. Compares a recency-only baseline, logistic regression, and XGBoost.
  5. Writes metrics, tables, figures, and a results summary to outputs/.

Run:
  python src/churn_pipeline.py --data data/raw/online_retail_II.xlsx
"""
import argparse
import json
import sqlite3
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:  # lets the script run without xgboost, but the project should use it
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_XGB = False

ROOT = Path(__file__).resolve().parents[1]
SQL_PATH = ROOT / "sql" / "features.sql"
OUT = ROOT / "outputs"
DB_PATH = ROOT / "data" / "processed" / "retail.db"

# --------------------------------------------------------------------------- #
# Configuration                                                                #
# --------------------------------------------------------------------------- #
ORIGIN = pd.Timestamp("2009-12-01")          # day 0 of the dataset
HORIZON = 90                                  # churn = no purchase in the next 90 days
ACTIVE_WINDOW = 180                           # score customers who bought in the last 180 days
TRAIN_CUTOFFS = ["2010-06-09", "2010-09-09", "2010-12-09", "2011-03-09"]
TEST_CUTOFF = "2011-09-09"                    # label window ends 2011-12-08, inside the data
SEED = 42
TOP_FRACTION = 0.20                           # share of customers a campaign could contact

COLMAP = {"invoice": "invoice", "invoiceno": "invoice", "stockcode": "stock_code",
          "description": "description", "quantity": "quantity",
          "invoicedate": "invoice_date", "price": "price", "unitprice": "price",
          "customer id": "customer_id", "customerid": "customer_id", "country": "country"}

MONEY_COLS = ["monetary", "avg_order_value", "spend_last_90d", "spend_prev_90d",
              "last_order_value"]
FEATURES = ["recency_days", "tenure_days", "n_orders", "monetary", "avg_order_value",
            "avg_items", "avg_gap_days", "max_gap_days", "orders_last_90d",
            "orders_prev_90d", "spend_last_90d", "spend_prev_90d", "last_order_value",
            "return_invoices", "r_score", "f_score", "m_score",
            "spend_trend", "recency_vs_gap", "is_single_order", "return_rate"]


# --------------------------------------------------------------------------- #
# 1. Data Loading & Cleaning                                                   #
# --------------------------------------------------------------------------- #
def load_raw(path: Path) -> pd.DataFrame:
    """Load raw transaction data from Excel or CSV."""
    if path.suffix.lower() in (".xlsx", ".xls"):
        sheets = pd.read_excel(path, sheet_name=None)      # the real file has two sheets
        df = pd.concat(sheets.values(), ignore_index=True)
    else:
        df = pd.read_csv(path, encoding="latin-1")
    df = df.rename(columns=lambda c: COLMAP.get(str(c).strip().lower(), c))
    missing = {"invoice", "stock_code", "quantity", "invoice_date", "price",
               "customer_id"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}. Found: {list(df.columns)}")
    return df


def clean(df: pd.DataFrame):
    """Clean transactions, returning the cleaned frame and a data-quality audit log."""
    log = [("rows in raw file", len(df))]

    def step(name, mask_keep):
        nonlocal df
        removed = int((~mask_keep).sum())
        df = df[mask_keep].copy()
        log.append((name, removed))

    step("exact duplicate rows (the two sheets overlap in Dec 2010)", ~df.duplicated())
    step("missing customer id", df["customer_id"].notna())
    df["customer_id"] = df["customer_id"].astype("int64")
    df["invoice"] = df["invoice"].astype(str)
    df["stock_code"] = df["stock_code"].astype(str)
    step("non-product stock codes (postage, fees, adjustments)",
         df["stock_code"].str.match(r"^\d{5}"))
    step("price <= 0", df["price"] > 0)
    df["is_return"] = df["invoice"].str.upper().str.startswith("C").astype(int)
    step("non-return rows with quantity <= 0", (df["is_return"] == 1) | (df["quantity"] > 0))

    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["day_num"] = (df["invoice_date"].dt.normalize() - ORIGIN).dt.days.astype(int)
    df["line_amount"] = df["quantity"] * df["price"]
    log.append(("rows after cleaning", len(df)))
    cols = ["invoice", "stock_code", "quantity", "price", "customer_id", "invoice_date",
            "day_num", "is_return", "line_amount"]
    if "country" in df.columns:
        cols.append("country")
    out = df[cols].copy()
    out["invoice_date"] = out["invoice_date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return out, pd.DataFrame(log, columns=["step", "rows"])


# --------------------------------------------------------------------------- #
# 2. SQL Feature Engineering                                                   #
# --------------------------------------------------------------------------- #
def build_db(df: pd.DataFrame) -> sqlite3.Connection:
    """Load cleaned transactions into SQLite for SQL-based feature engineering."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    df.to_sql("transactions", con, if_exists="replace", index=False)
    con.execute("CREATE INDEX IF NOT EXISTS ix_cust_day ON transactions (customer_id, day_num)")
    con.commit()
    return con


def snapshot_features(con, cutoff: str) -> pd.DataFrame:
    """Generate customer-level features for a single snapshot date using SQL."""
    cutoff_day = int((pd.Timestamp(cutoff) - ORIGIN).days)
    sql = SQL_PATH.read_text()
    snap = pd.read_sql_query(sql, con, params={"cutoff": cutoff_day, "horizon": HORIZON,
                                               "active": ACTIVE_WINDOW})
    snap["cutoff"] = cutoff
    return add_python_features(snap)


def add_python_features(d: pd.DataFrame) -> pd.DataFrame:
    """Add derived features that are more naturally expressed in Python."""
    d = d.copy()
    d["spend_trend"] = np.log((d["spend_last_90d"].clip(lower=0) + 1)
                              / (d["spend_prev_90d"].clip(lower=0) + 1))
    gap = d["avg_gap_days"].where(d["avg_gap_days"] > 0)
    d["recency_vs_gap"] = d["recency_days"] / gap            # >1 means "overdue" vs own rhythm
    d["is_single_order"] = (d["n_orders"] == 1).astype(int)
    d["return_rate"] = d["return_invoices"] / d["n_orders"]
    for c in MONEY_COLS:                                      # tame heavy right tails
        d[c] = np.log1p(d[c].clip(lower=0))
    return d


# --------------------------------------------------------------------------- #
# 3. Model Training & Evaluation                                              #
# --------------------------------------------------------------------------- #
def make_models():
    """Initialize baseline models with conservative hyperparameters."""
    logreg = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                           LogisticRegression(max_iter=2000, random_state=SEED))
    if HAS_XGB:
        gbm = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                            random_state=SEED, n_jobs=-1)
    else:
        warnings.warn("xgboost not installed - using sklearn HistGradientBoosting instead.")
        gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4,
                                             random_state=SEED)
    return logreg, gbm


def evaluate(y, p, top_fraction=TOP_FRACTION):
    """Compute classification metrics including business-relevant lift statistics."""
    n = len(y)
    k = int(np.ceil(top_fraction * n))
    order = np.argsort(-p)
    base = float(np.mean(y))
    prec_k = float(np.mean(y[order][:k]))
    return {"roc_auc": float(roc_auc_score(y, p)),
            "pr_auc": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, np.clip(p, 0, 1))),
            "precision_at_top20pct": prec_k,
            "recall_at_top20pct": float(np.sum(y[order][:k]) / np.sum(y)),
            "lift_at_top20pct": prec_k / base if base > 0 else float("nan")}


def decile_table(y, p):
    """Build a risk decile table for business stakeholder communication."""
    t = pd.DataFrame({"y": y, "p": p}).sort_values("p", ascending=False).reset_index(drop=True)
    t["decile"] = pd.qcut(t.index, 10, labels=range(1, 11))
    g = t.groupby("decile", observed=True).agg(customers=("y", "size"),
                                               churners=("y", "sum"),
                                               churn_rate=("y", "mean"),
                                               mean_predicted=("p", "mean")).reset_index()
    g["cum_share_of_churners"] = g["churners"].cumsum() / g["churners"].sum()
    return g


# --------------------------------------------------------------------------- #
# 4. Output Generation                                                         #
# --------------------------------------------------------------------------- #
def make_figures(y_test, scores, deciles, base_rate, perm):
    """Generate evaluation figures: ROC/PR curves, decile chart, feature importance."""
    OUT.mkdir(exist_ok=True)

    # ROC and PR curves
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, p in scores.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        ax[0].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_test, p):.3f})")
        pr, rc, _ = precision_recall_curve(y_test, p)
        ax[1].plot(rc, pr, label=f"{name} (AP {average_precision_score(y_test, p):.3f})")
    ax[0].plot([0, 1], [0, 1], "k:", lw=1)
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (test snapshot)")
    ax[1].axhline(base_rate, color="k", ls=":", lw=1)
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall (test snapshot)")
    ax[0].legend(fontsize=8)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "roc_pr_curves.png", dpi=150)
    plt.close(fig)

    # Decile churn rate chart
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(deciles["decile"].astype(int), deciles["churn_rate"], color="#2f6db5")
    ax.axhline(base_rate, color="k", ls="--", lw=1, label=f"overall churn rate {base_rate:.2f}")
    ax.set(xlabel="Risk decile (1 = highest predicted risk)", ylabel="Actual churn rate",
           title="Do high-risk deciles really churn more?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "decile_churn_rate.png", dpi=150)
    plt.close(fig)

    # Feature importance chart
    top = perm.sort_values("importance_mean", ascending=True).tail(10)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#2f6db5")
    ax.set(xlabel="Drop in ROC-AUC when shuffled", title="Top features (permutation importance)")
    fig.tight_layout()
    fig.savefig(OUT / "feature_importance.png", dpi=150)
    plt.close(fig)


def write_summary(metrics, dq_log, sizes, deciles, model_name):
    """Generate a markdown summary report with model results and business insights."""
    best = metrics[model_name]
    top2 = deciles.head(2)
    capture = float(top2["churners"].sum() / deciles["churners"].sum())
    lines = ["# Results summary (generated by src/churn_pipeline.py)", ""]
    if not HAS_XGB:
        lines += ["> NOTE: xgboost was not installed; the gradient-boosting model below is the "
                  "scikit-learn fallback, not XGBoost.", ""]
    lines += [f"Test snapshot: customers active in the 180 days before {TEST_CUTOFF}; "
              f"churn = no purchase in the following {HORIZON} days.", "",
              "| Customers | Churn rate |", "|---|---|",
              f"| Train ({len(TRAIN_CUTOFFS)} snapshots): {sizes['train_n']:,} rows "
              f"| {sizes['train_rate']:.1%} |",
              f"| Test: {sizes['test_n']:,} customers | {sizes['test_rate']:.1%} |", "",
              "## Model comparison (test snapshot)", "",
              "| Model | ROC-AUC | PR-AUC | Brier | Precision@top20% | Recall@top20% | Lift@top20% |",
              "|---|---|---|---|---|---|---|"]
    for name, m in metrics.items():
        brier = f"{m['brier']:.3f}" if "brier" in m else "n/a"
        lines.append(f"| {name} | {m['roc_auc']:.3f} | {m['pr_auc']:.3f} | {brier} | "
                     f"{m['precision_at_top20pct']:.3f} | {m['recall_at_top20pct']:.3f} | "
                     f"{m['lift_at_top20pct']:.2f} |")
    max_lift = 1 / sizes["test_rate"] if sizes["test_rate"] > 0 else float("nan")
    lines += ["", f"Lift is capped by the base rate: with a {sizes['test_rate']:.0%} churn rate the "
              f"best possible lift at the top 20% is {max_lift:.2f}x, so read lift together with "
              "precision and recall.", "", "## Business reading", "",
              f"- Contacting the {TOP_FRACTION:.0%} of customers with the highest predicted risk "
              f"reaches {best['recall_at_top20pct']:.0%} of customers who actually lapse, "
              f"{best['lift_at_top20pct']:.2f}x better than contacting customers at random.",
              f"- The two highest-risk deciles contain {capture:.0%} of all churners.",
              "- Suggested use: send a retention offer to the highest-risk deciles first, and "
              "compare repeat purchases against a held-out group to measure whether the "
              "offer, not just the model, causes the improvement.", "",
              "## Data-quality log", "", "| Step | Rows |", "|---|---|"]
    lines += [f"| {r.step} | {r.rows:,} |" for r in dq_log.itertuples()]
    (OUT / "results_summary.md").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# Main Entry Point                                                             #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Customer churn prediction pipeline")
    ap.add_argument("--data", required=True, help="path to online_retail_II.xlsx (or a CSV)")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    # Step 1: Load and clean
    print("=" * 60)
    print("STEP 1: Loading and cleaning raw transactions")
    print("=" * 60)
    raw = load_raw(Path(args.data))
    tx, dq_log = clean(raw)
    dq_log.to_csv(OUT / "data_quality_log.csv", index=False)
    print(dq_log.to_string(index=False))

    # Step 2: Build features with SQL
    print("\n" + "=" * 60)
    print("STEP 2: Building customer features with SQL")
    print("=" * 60)
    con = build_db(tx)
    train = pd.concat([snapshot_features(con, c) for c in TRAIN_CUTOFFS], ignore_index=True)
    test = snapshot_features(con, TEST_CUTOFF)
    con.close()
    print(f"Train rows: {len(train):,} (churn rate: {train['churned'].mean():.1%})")
    print(f"Test customers: {len(test):,} (churn rate: {test['churned'].mean():.1%})")

    # Step 3: Train models
    print("\n" + "=" * 60)
    print("STEP 3: Training models")
    print("=" * 60)
    X_tr, y_tr = train[FEATURES], train["churned"].to_numpy()
    X_te, y_te = test[FEATURES], test["churned"].to_numpy()
    logreg, gbm = make_models()
    gbm_name = "XGBoost" if HAS_XGB else "GradientBoosting (fallback)"

    logreg.fit(X_tr, y_tr)
    print("  ✓ Logistic Regression trained")
    gbm.fit(X_tr, y_tr)
    print(f"  ✓ {gbm_name} trained")

    scores = {"Baseline: recency only": test["recency_days"].to_numpy().astype(float),
              "Logistic regression": logreg.predict_proba(X_te)[:, 1],
              gbm_name: gbm.predict_proba(X_te)[:, 1]}

    # Step 4: Evaluate
    print("\n" + "=" * 60)
    print("STEP 4: Evaluating models")
    print("=" * 60)
    metrics = {}
    for name, p in scores.items():
        metrics[name] = evaluate(y_te, p)
        if name.startswith("Baseline"):          # a raw heuristic score has no probability
            metrics[name].pop("brier")
        print(f"  {name}: ROC-AUC = {metrics[name]['roc_auc']:.3f}, "
              f"Lift@top20% = {metrics[name]['lift_at_top20pct']:.2f}x")

    deciles = decile_table(y_te, scores[gbm_name])
    deciles.to_csv(OUT / "decile_table.csv", index=False)

    pi = permutation_importance(gbm, X_te, y_te, scoring="roc_auc", n_repeats=5,
                                random_state=SEED)
    perm = pd.DataFrame({"feature": FEATURES, "importance_mean": pi.importances_mean,
                         "importance_std": pi.importances_std})
    perm.sort_values("importance_mean", ascending=False).to_csv(
        OUT / "feature_importance.csv", index=False)

    # Step 5: Generate outputs
    print("\n" + "=" * 60)
    print("STEP 5: Generating outputs")
    print("=" * 60)
    make_figures(y_te, scores, deciles, float(y_te.mean()), perm)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    sizes = {"train_n": len(train), "train_rate": float(y_tr.mean()),
             "test_n": len(test), "test_rate": float(y_te.mean())}
    write_summary(metrics, dq_log, sizes, deciles, gbm_name)
    print("  ✓ metrics.json")
    print("  ✓ feature_importance.csv")
    print("  ✓ decile_table.csv")
    print("  ✓ roc_pr_curves.png")
    print("  ✓ decile_churn_rate.png")
    print("  ✓ feature_importance.png")
    print("  ✓ results_summary.md")

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print((OUT / "results_summary.md").read_text())


if __name__ == "__main__":
    main()
