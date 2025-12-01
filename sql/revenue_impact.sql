SELECT
  experiment_variant,
  user_segment, -- e.g., 'new_user', 'returning', 'power_user'
  COUNT(DISTINCT user_id) as users,
  SUM(revenue) as total_revenue,
  ROUND(SUM(revenue) / COUNT(DISTINCT user_id), 2) as revenue_per_user
FROM user_purchases
WHERE experiment_name = 'recommendation_v2'
GROUP BY 1, 2
ORDER BY 1, 2;