-- ==============================================================================
-- INSTAMART CUSTOMER INTELLIGENCE & SUPPORT COPILOT
-- PART A: SQL ANALYSIS QUERIES
-- Database: data/instamart.db
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- Q1: WHERE and HAVING
-- High-volume categories with completed orders exceeding 1,000 orders
-- ------------------------------------------------------------------------------
-- name: q1_where_and_having
SELECT 
    category,
    COUNT(order_id) AS completed_orders,
    ROUND(SUM(order_amount), 2) AS total_revenue_inr,
    ROUND(AVG(order_amount), 2) AS avg_order_value_inr
FROM orders
WHERE status = 'completed'
GROUP BY category
HAVING COUNT(order_id) > 1000
ORDER BY total_revenue_inr DESC;


-- ------------------------------------------------------------------------------
-- Q2: Three-table JOIN for city and product-category revenue
-- Combines customers, orders, and products to evaluate cross-segment revenue
-- ------------------------------------------------------------------------------
-- name: q2_three_table_join
SELECT 
    c.city,
    p.category,
    COUNT(o.order_id) AS total_orders,
    ROUND(SUM(o.order_amount), 2) AS total_revenue_inr,
    ROUND(AVG(o.order_amount), 2) AS avg_order_value_inr
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city ASC, total_revenue_inr DESC;


-- ------------------------------------------------------------------------------
-- Q3: Scalar subquery for products priced strictly above average price
-- Identifies premium product offerings across categories
-- ------------------------------------------------------------------------------
-- name: q3_scalar_subquery
SELECT 
    product_id,
    product_name,
    category,
    price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC;


-- ------------------------------------------------------------------------------
-- Q4: LEFT JOIN for customers without completed orders
-- Identifies registered customers who have never completed an order
-- ------------------------------------------------------------------------------
-- name: q4_left_join
SELECT 
    c.customer_id,
    c.city,
    c.gender,
    c.age,
    c.signup_date
FROM customers c
LEFT JOIN orders o 
    ON c.customer_id = o.customer_id 
    AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id ASC;


-- ------------------------------------------------------------------------------
-- Q5: CTE for above-average customer spending
-- Aggregates customer lifetime spending and filters for above-mean spenders
-- ------------------------------------------------------------------------------
-- name: q5_cte_above_average_spend
WITH customer_spending AS (
    SELECT 
        c.customer_id,
        c.city,
        COUNT(o.order_id) AS total_completed_orders,
        ROUND(SUM(o.order_amount), 2) AS total_spend_inr
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.customer_id, c.city
)
SELECT 
    customer_id,
    city,
    total_completed_orders,
    total_spend_inr
FROM customer_spending
WHERE total_spend_inr > (SELECT AVG(total_spend_inr) FROM customer_spending)
ORDER BY total_spend_inr DESC;


-- ------------------------------------------------------------------------------
-- Q6: RANK, DENSE_RANK and ROW_NUMBER by city
-- Ranks customer lifetime spending within each individual city
-- ------------------------------------------------------------------------------
-- name: q6_window_ranking_by_city
WITH city_customer_spend AS (
    SELECT 
        c.city,
        c.customer_id,
        ROUND(SUM(o.order_amount), 2) AS total_spend_inr
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.city, c.customer_id
)
SELECT 
    city,
    customer_id,
    total_spend_inr,
    ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend_inr DESC) AS row_num,
    RANK() OVER (PARTITION BY city ORDER BY total_spend_inr DESC) AS rank_val,
    DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend_inr DESC) AS dense_rank_val
FROM city_customer_spend
ORDER BY city ASC, rank_val ASC;


-- ------------------------------------------------------------------------------
-- Q7: Daily running revenue using a window function
-- Cumulative revenue timeline across chronological order dates
-- ------------------------------------------------------------------------------
-- name: q7_daily_running_revenue
WITH daily_revenue AS (
    SELECT 
        SUBSTR(order_ts, 1, 10) AS order_date,
        COUNT(order_id) AS daily_orders,
        ROUND(SUM(order_amount), 2) AS daily_revenue_inr
    FROM orders
    WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
)
SELECT 
    order_date,
    daily_orders,
    daily_revenue_inr,
    ROUND(SUM(daily_revenue_inr) OVER (ORDER BY order_date ASC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), 2) AS running_total_revenue_inr
FROM daily_revenue
ORDER BY order_date ASC;
