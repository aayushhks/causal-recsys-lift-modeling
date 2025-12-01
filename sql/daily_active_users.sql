SELECT
  DATE(timestamp) as date,
  experiment_variant,
  COUNT(DISTINCT user_id) as dau,
  COUNT(*) as total_sessions
FROM user_sessions
WHERE experiment_name = 'recommendation_v2'
  AND date BETWEEN '2025-01-01' AND '2025-01-14'
GROUP BY 1, 2
ORDER BY 1, 2;