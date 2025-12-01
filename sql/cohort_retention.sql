SELECT
  DATE_TRUNC('week', first_session_date) as cohort_week,
  experiment_variant,
  DATEDIFF(day, first_session_date, activity_date) as days_since_first,
  COUNT(DISTINCT user_id) as retained_users
FROM (
  SELECT
    user_id,
    experiment_variant,
    MIN(DATE(timestamp)) as first_session_date
  FROM user_sessions
  GROUP BY 1, 2
) first_sessions
JOIN user_sessions activity
  ON first_sessions.user_id = activity.user_id
WHERE days_since_first IN (1, 7, 14, 30)
GROUP BY 1, 2, 3
ORDER BY 1, 2, 3;