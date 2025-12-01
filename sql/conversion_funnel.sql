WITH funnel AS (
  SELECT
    user_id,
    experiment_variant,
    MAX(CASE WHEN event_type = 'recommendation_view' THEN 1 ELSE 0 END) as viewed,
    MAX(CASE WHEN event_type = 'recommendation_click' THEN 1 ELSE 0 END) as clicked,
    MAX(CASE WHEN event_type = 'add_to_cart' THEN 1 ELSE 0 END) as added_to_cart,
    MAX(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) as purchased
  FROM events
  WHERE experiment_name = 'recommendation_v2'
  GROUP BY 1, 2
)
SELECT
  experiment_variant,
  COUNT(*) as total_users,
  SUM(viewed) as viewed_recs,
  SUM(clicked) as clicked_recs,
  SUM(added_to_cart) as added_to_cart,
  SUM(purchased) as purchased,
  ROUND(100.0 * SUM(clicked) / NULLIF(SUM(viewed), 0), 2) as ctr_pct,
  ROUND(100.0 * SUM(purchased) / COUNT(*), 2) as conversion_rate_pct
FROM funnel
GROUP BY 1;