-- =============================================================================
-- Customer-level features and churn label for ONE snapshot date.
-- =============================================================================
-- This query demonstrates:
--   - CTEs (Common Table Expressions) for layered feature computation
--   - Window functions: ROW_NUMBER, LAG, NTILE
--   - Parameterized queries for flexible snapshot-based feature generation
--
-- Bound parameters (from Python):
--   :cutoff   snapshot day number. Only data BEFORE this day is used for features.
--   :horizon  length of the outcome window in days (label looks at [cutoff, cutoff+horizon)).
--   :active   customers must have bought within this many days before the cutoff to be scored.
-- =============================================================================

-- Step 1: Aggregate purchase invoices (excluding returns)
WITH purchases AS (
    SELECT customer_id,
           invoice,
           day_num,
           SUM(line_amount)            AS order_value,
           COUNT(DISTINCT stock_code)  AS n_items
    FROM transactions
    WHERE is_return = 0
      AND day_num < :cutoff
    GROUP BY customer_id, invoice, day_num
),

-- Step 2: Order sequence and previous order date per customer (using window functions)
ordered AS (
    SELECT p.*,
           ROW_NUMBER() OVER (PARTITION BY customer_id
                              ORDER BY day_num DESC, invoice DESC) AS rn_latest,
           LAG(day_num)    OVER (PARTITION BY customer_id
                              ORDER BY day_num, invoice)           AS prev_day
    FROM purchases p
),

-- Step 3: Inter-purchase gap statistics
gaps AS (
    SELECT customer_id,
           AVG(day_num - prev_day) AS avg_gap_days,
           MAX(day_num - prev_day) AS max_gap_days
    FROM ordered
    WHERE prev_day IS NOT NULL
    GROUP BY customer_id
),

-- Step 4: RFM-style aggregates and recent-activity window comparisons
agg AS (
    SELECT customer_id,
           COUNT(*)           AS n_orders,
           SUM(order_value)   AS monetary,
           AVG(order_value)   AS avg_order_value,
           AVG(n_items)       AS avg_items,
           MIN(day_num)       AS first_day,
           MAX(day_num)       AS last_day,
           -- Recent 90-day window
           SUM(CASE WHEN day_num >= :cutoff - 90 THEN 1 ELSE 0 END)                 AS orders_last_90d,
           SUM(CASE WHEN day_num >= :cutoff - 90 THEN order_value ELSE 0 END)       AS spend_last_90d,
           -- Previous 90-day window (90-180 days before cutoff)
           SUM(CASE WHEN day_num >= :cutoff - 180
                     AND day_num <  :cutoff - 90 THEN 1 ELSE 0 END)                 AS orders_prev_90d,
           SUM(CASE WHEN day_num >= :cutoff - 180
                     AND day_num <  :cutoff - 90 THEN order_value ELSE 0 END)       AS spend_prev_90d
    FROM ordered
    GROUP BY customer_id
),

-- Step 5: Most recent order value
latest AS (
    SELECT customer_id, order_value AS last_order_value
    FROM ordered
    WHERE rn_latest = 1
),

-- Step 6: Return behavior
returns AS (
    SELECT customer_id, COUNT(DISTINCT invoice) AS return_invoices
    FROM transactions
    WHERE is_return = 1
      AND day_num < :cutoff
    GROUP BY customer_id
),

-- Step 7: Filter to "active" customers (bought within :active days of cutoff)
active AS (
    SELECT a.*,
           :cutoff - a.last_day  AS recency_days,
           :cutoff - a.first_day AS tenure_days
    FROM agg a
    WHERE a.last_day >= :cutoff - :active
),

-- Step 8: RFM quintile scores using NTILE window function
scored AS (
    SELECT act.*,
           NTILE(5) OVER (ORDER BY recency_days DESC) AS r_score,   -- 5 = bought most recently
           NTILE(5) OVER (ORDER BY n_orders)          AS f_score,
           NTILE(5) OVER (ORDER BY monetary)          AS m_score
    FROM active act
),

-- Step 9: Future purchases (for label generation — strictly after cutoff)
future AS (
    SELECT DISTINCT customer_id
    FROM transactions
    WHERE is_return = 0
      AND day_num >= :cutoff
      AND day_num <  :cutoff + :horizon
)

-- Final output: one row per active customer with features and churn label
SELECT s.customer_id,
       s.last_day,
       s.recency_days,
       s.tenure_days,
       s.n_orders,
       s.monetary,
       s.avg_order_value,
       s.avg_items,
       g.avg_gap_days,
       g.max_gap_days,
       s.orders_last_90d,
       s.orders_prev_90d,
       s.spend_last_90d,
       s.spend_prev_90d,
       l.last_order_value,
       COALESCE(r.return_invoices, 0) AS return_invoices,
       s.r_score, s.f_score, s.m_score,
       CASE WHEN f.customer_id IS NULL THEN 1 ELSE 0 END AS churned   -- 1 = no purchase in the horizon
FROM scored s
LEFT JOIN gaps    g ON g.customer_id = s.customer_id
LEFT JOIN latest  l ON l.customer_id = s.customer_id
LEFT JOIN returns r ON r.customer_id = s.customer_id
LEFT JOIN future  f ON f.customer_id = s.customer_id
ORDER BY s.customer_id;
