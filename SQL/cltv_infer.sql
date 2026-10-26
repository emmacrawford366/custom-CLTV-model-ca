/*
-- Customer Lifetime Value (CLTV) Analysis by Cumulative Month with Advertising Spend --
-- INFERENCE VERSION: Predicts change from most recent month to next month --
-- Dynamically identifies the most recent month in the data --
-- Enhanced with clicks, impressions, average order value, distinct campaigns, and monthly frequency --

------- Customization Instructions -------
1) TIME WINDOW: Set to 12 months (1 year) in the AMC query editor
2) OPTIONAL UPDATE: Modify tracked_item filter if analyzing specific ASINs
3) OPTIONAL UPDATE: Adjust marketplace_id if analyzing specific regions
4) OPTIONAL UPDATE: Update campaign_id filters if analyzing specific campaigns
5) NOTE: If field names differ (e.g., 'click' instead of 'clicks', 'impression' instead of 'impressions'),
   update the column references in monthly_dsp_spend and monthly_sa_spend CTEs accordingly
6) NOTE: DATEADD syntax is for Redshift. For PostgreSQL, replace DATEADD('month', 1, date) with: date + INTERVAL '1 month'
   For BigQuery, use: DATE_ADD(date, INTERVAL 1 MONTH)
*/
WITH
  monthly_purchases AS (
    -- Aggregate purchases by customer and month
    SELECT
      user_id,
      DATE_TRUNC ('month', event_date_utc) AS purchase_month,
      SUM(CAST(total_product_sales AS DECIMAL(18, 2))) AS monthly_sales,
      SUM(total_units_sold) AS monthly_units,
      COUNT(DISTINCT conversion_id) AS monthly_orders,
      -- Calculate average order value
      CASE 
        WHEN COUNT(DISTINCT conversion_id) > 0 
        THEN SUM(CAST(total_product_sales AS DECIMAL(18, 2))) / COUNT(DISTINCT conversion_id)
        ELSE 0
      END AS monthly_avg_order_value
    FROM
      conversions_all_for_audiences
    WHERE
      user_id IS NOT NULL
      AND total_product_sales > 0
      AND event_category = 'purchase'
    GROUP BY
      1,
      2
  ),
  monthly_dsp_spend AS (
    -- Aggregate DSP advertising spend, clicks, impressions, and campaigns by customer and month
    SELECT
      user_id,
      DATE_TRUNC ('month', impression_date_utc) AS spend_month,
      SUM(total_cost) / 100000 AS monthly_dsp_spend,
      sum(clicks) AS monthly_dsp_clicks,
      SUM(impressions) AS monthly_dsp_impressions,
      COUNT(DISTINCT campaign_id) AS monthly_dsp_campaigns
    FROM
      (
      select 
      user_id,
      impression_date_utc,
      campaign_id,
      suM(total_cost) as total_cost,
      0 as clicks,
      sum(impressions) as impressions
      from dsp_impressions_for_audiences
      group by 1,2,3
      
      UNION ALL
      
      select 
      user_id,
      impression_date_utc,
      campaign_id,
      0 as total_cost,
      sum(clicks) as clicks,
      0 as impressions
      from dsp_clicks_for_audiences
      group by 1,2,3
      
      )
    WHERE
      user_id IS NOT NULL
      AND total_cost > 0
    GROUP BY
      1,
      2
  ),
  monthly_sa_spend AS (
    -- Aggregate Sponsored Ads spend, clicks, impressions, and campaigns by customer and month
    SELECT
      user_id,
      DATE_TRUNC ('month', event_date_utc) AS spend_month,
      SUM(spend) / 1000000 AS monthly_sa_spend,
      SUM(clicks) AS monthly_sa_clicks,
      SUM(impressions) AS monthly_sa_impressions,
      COUNT(DISTINCT campaign) AS monthly_sa_campaigns
    FROM
      sponsored_ads_traffic_for_audiences
    WHERE
      user_id IS NOT NULL
      AND spend > 0
    GROUP BY
      1,
      2
  ),
  campaign_tracking AS (
    -- Track distinct campaigns by user and month for accurate cumulative counting
    SELECT
      user_id,
      DATE_TRUNC ('month', impression_date_utc) AS activity_month,
      campaign,
      'DSP' AS source
    FROM
      dsp_impressions_for_audiences
    WHERE
      user_id IS NOT NULL
      AND campaign IS NOT NULL
    UNION ALL
    SELECT
      user_id,
      DATE_TRUNC ('month', event_date_utc) AS activity_month,
      campaign,
      'SA' AS source
    FROM
      sponsored_ads_traffic_for_audiences
    WHERE
      user_id IS NOT NULL
      AND campaign IS NOT NULL
  ),
  monthly_combined AS (
    -- Combine purchases and spend data by customer and month
    SELECT
      COALESCE(mp.user_id, ds.user_id, sa.user_id) AS user_id,
      COALESCE(mp.purchase_month, ds.spend_month, sa.spend_month) AS activity_month,
      COALESCE(mp.monthly_sales, 0) AS monthly_sales,
      COALESCE(mp.monthly_units, 0) AS monthly_units,
      COALESCE(mp.monthly_orders, 0) AS monthly_orders,
      COALESCE(mp.monthly_avg_order_value, 0) AS monthly_avg_order_value,
      COALESCE(ds.monthly_dsp_spend, 0) + COALESCE(sa.monthly_sa_spend, 0) AS monthly_spend,
      COALESCE(ds.monthly_dsp_clicks, 0) + COALESCE(sa.monthly_sa_clicks, 0) AS monthly_clicks,
      COALESCE(ds.monthly_dsp_impressions, 0) + COALESCE(sa.monthly_sa_impressions, 0) AS monthly_impressions,
      COALESCE(ds.monthly_dsp_campaigns, 0) + COALESCE(sa.monthly_sa_campaigns, 0) AS monthly_campaigns
    FROM
      monthly_purchases mp
      FULL OUTER JOIN monthly_dsp_spend ds ON mp.user_id = ds.user_id
      AND mp.purchase_month = ds.spend_month
      FULL OUTER JOIN monthly_sa_spend sa ON COALESCE(mp.user_id, ds.user_id) = sa.user_id
      AND COALESCE(mp.purchase_month, ds.spend_month) = sa.spend_month
  ),
  cumulative_calculations AS (
    -- Calculate cumulative metrics using self-join approach
    SELECT
      mc1.user_id,
      mc1.activity_month,
      mc1.monthly_sales,
      mc1.monthly_units,
      mc1.monthly_orders,
      mc1.monthly_spend,
      mc1.monthly_clicks,
      mc1.monthly_impressions,
      mc1.monthly_campaigns,
      mc1.monthly_avg_order_value,
      SUM(mc2.monthly_sales) AS cumulative_sales,
      SUM(mc2.monthly_spend) AS cumulative_spend,
      SUM(mc2.monthly_sales) - SUM(mc2.monthly_spend) AS cumulative_cltv,
      SUM(mc2.monthly_units) AS cumulative_units,
      SUM(mc2.monthly_orders) AS cumulative_orders,
      SUM(mc2.monthly_clicks) AS cumulative_clicks,
      SUM(mc2.monthly_impressions) AS cumulative_impressions,
      COUNT(DISTINCT mc2.activity_month) AS months_with_activity,
      -- Calculate cumulative average order value
      CASE 
        WHEN SUM(mc2.monthly_orders) > 0 
        THEN SUM(mc2.monthly_sales) / SUM(mc2.monthly_orders)
        ELSE 0
      END AS cumulative_avg_order_value
    FROM
      monthly_combined mc1
      INNER JOIN monthly_combined mc2 ON mc1.user_id = mc2.user_id
      AND mc2.activity_month <= mc1.activity_month
    GROUP BY
      mc1.user_id,
      mc1.activity_month,
      mc1.monthly_sales,
      mc1.monthly_units,
      mc1.monthly_orders,
      mc1.monthly_spend,
      mc1.monthly_clicks,
      mc1.monthly_impressions,
      mc1.monthly_campaigns,
      mc1.monthly_avg_order_value
  ),
  cumulative_campaigns AS (
    -- Calculate cumulative distinct campaigns per user up to each month
    SELECT
      ct1.user_id,
      ct1.activity_month,
      COUNT(DISTINCT ct2.campaign) AS cumulative_campaigns
    FROM
      campaign_tracking ct1
      INNER JOIN campaign_tracking ct2 ON ct1.user_id = ct2.user_id
      AND ct2.activity_month <= ct1.activity_month
    GROUP BY
      ct1.user_id,
      ct1.activity_month
  ),
  customer_monthly_metrics AS (
    -- Add recency ranking to identify most recent month and join cumulative campaigns
    SELECT
      ccm.user_id,
      ccm.activity_month,
      ccm.monthly_sales,
      ccm.monthly_units,
      ccm.monthly_orders,
      ccm.monthly_spend,
      ccm.monthly_clicks,
      ccm.monthly_impressions,
      ccm.monthly_campaigns,
      ccm.monthly_avg_order_value,
      ccm.cumulative_sales,
      ccm.cumulative_spend,
      ccm.cumulative_cltv,
      ccm.cumulative_units,
      ccm.cumulative_orders,
      ccm.cumulative_clicks,
      ccm.cumulative_impressions,
      ccm.months_with_activity,
      ccm.cumulative_avg_order_value,
      COALESCE(cc.cumulative_campaigns, 0) AS cumulative_campaigns,
      -- Calculate monthly frequency (impressions in this specific month)
      ccm.monthly_impressions AS monthly_frequency,
      ROW_NUMBER() OVER (
        PARTITION BY
          ccm.user_id
        ORDER BY
          ccm.activity_month DESC
      ) AS recency_rank
    FROM
      cumulative_calculations ccm
      LEFT JOIN cumulative_campaigns cc ON ccm.user_id = cc.user_id
      AND ccm.activity_month = cc.activity_month
  ),
  most_recent_month AS (
    -- Dynamically find the most recent month in the data
    -- This allows the query to work with any time period without hardcoded dates
    SELECT MAX(activity_month) AS max_month
    FROM customer_monthly_metrics
  ),
  most_recent_metrics AS (
    -- Get metrics for the most recent month and next month (if available) for CLTV prediction
    SELECT
      cmm.user_id,
      -- Most recent month metrics (through most recent month)
      MAX(CASE WHEN cmm.activity_month <= mrm.max_month THEN cmm.cumulative_cltv END) as october_cltv,
      MAX(CASE WHEN cmm.activity_month <= mrm.max_month THEN cmm.cumulative_sales END) as sales_through_october,
      MAX(CASE WHEN cmm.activity_month <= mrm.max_month THEN cmm.cumulative_spend END) as spend_through_october,
      MAX(CASE WHEN cmm.activity_month <= mrm.max_month THEN cmm.cumulative_clicks END) as clicks_through_october,
      MAX(CASE WHEN cmm.activity_month <= mrm.max_month THEN cmm.cumulative_impressions END) as impressions_through_october,
      MAX(CASE WHEN cmm.activity_month <= mrm.max_month THEN cmm.cumulative_avg_order_value END) as aov_through_october,
      MAX(CASE WHEN cmm.activity_month <= mrm.max_month THEN cmm.cumulative_campaigns END) as campaigns_through_october,
      -- Most recent month monthly frequency (impressions in most recent month only)
      SUM(CASE WHEN cmm.activity_month = mrm.max_month THEN cmm.monthly_impressions ELSE 0 END) as october_monthly_frequency,
      -- Next month metrics (through next month, cumulative up to and including next month)
      -- If next month data doesn't exist yet, these will be NULL (for inference)
      -- NOTE: DATEADD syntax is for Redshift. For PostgreSQL, use: mrm.max_month + INTERVAL '1 month'
      -- For BigQuery, use: DATE_ADD(mrm.max_month, INTERVAL 1 MONTH)
      MAX(CASE WHEN cmm.activity_month <= '2025-12-01' THEN cmm.cumulative_cltv END) as november_cltv,
      MAX(CASE WHEN cmm.activity_month <= '2025-12-01'  THEN cmm.cumulative_sales END) as sales_through_november,
      MAX(CASE WHEN cmm.activity_month <= '2025-12-01'  THEN cmm.cumulative_spend END) as spend_through_november,
      MAX(CASE WHEN cmm.activity_month <= '2025-12-01' THEN cmm.cumulative_clicks END) as clicks_through_november,
      MAX(CASE WHEN cmm.activity_month <= '2025-12-01'  THEN cmm.cumulative_impressions END) as impressions_through_november,
      MAX(CASE WHEN cmm.activity_month <= '2025-12-01' THEN cmm.cumulative_avg_order_value END) as aov_through_november,
      MAX(CASE WHEN cmm.activity_month <= '2025-12-01'  THEN cmm.cumulative_campaigns END) as campaigns_through_november,
      -- Next month monthly frequency (impressions in next month only)
      SUM(CASE WHEN cmm.activity_month between '2025-11-01' and '2025-12-01'  THEN cmm.monthly_impressions ELSE 0 END) as november_monthly_frequency,
      -- Activity counts
      COUNT(DISTINCT CASE WHEN cmm.monthly_sales > 0 THEN cmm.activity_month END) as months_active_sales,
      COUNT(DISTINCT CASE WHEN cmm.monthly_spend > 0 THEN cmm.activity_month END) as months_active_spend
    FROM customer_monthly_metrics cmm
    CROSS JOIN most_recent_month mrm
    GROUP BY cmm.user_id
  )

-- Final output for inference dataset
-- Features through most recent month (october_* columns) will be populated
-- Next month metrics (november_* columns) will be NULL if next month data doesn't exist yet
-- Target variable (pct_change_cltv_oct_to_nov) will be NULL for inference (predicted by model)
SELECT
  onm.user_id,
  onm.months_active_sales,
  onm.months_active_spend,
  -- Most recent month metrics (features for prediction)
  onm.sales_through_october,
  onm.spend_through_october,
  onm.clicks_through_october,
  onm.impressions_through_october,
  onm.aov_through_october,
  onm.campaigns_through_october,
  onm.october_monthly_frequency,
  -- Next month metrics (will be NULL for inference when next month data doesn't exist)
  onm.sales_through_november,
  onm.spend_through_november,
  onm.clicks_through_november,
  onm.impressions_through_november,
  onm.aov_through_november,
  onm.campaigns_through_november,
  onm.november_monthly_frequency
FROM most_recent_metrics onm
WHERE onm.user_id IS NOT NULL
  -- Filter to only include users who have activity through or before the most recent month
  -- This ensures we have valid features for prediction
  AND onm.october_cltv IS NOT NULL;

