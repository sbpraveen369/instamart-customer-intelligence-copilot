-- ==============================================================================
-- INSTAMART CUSTOMER INTELLIGENCE & SUPPORT COPILOT
-- PART A: SQL ANALYSIS QUERIES
-- Database: data/instamart.db
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- Q1: WHERE then HAVING
-- Keep only completed orders with order_amount > 500 (WHERE), group by customer,
-- and keep customers whose total of those orders exceeds 8000 (HAVING).
-- Reference: 30 customers.
-- ------------------------------------------------------------------------------
-- name: q1_where_and_having
SELECT 
    customer_id,
    ROUND(SUM(order_amount), 2) AS total_amount
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total_amount DESC;


-- ------------------------------------------------------------------------------
-- Q2: Three-Table JOIN
-- Revenue by city and product category (join orders, customers, products).
-- Reference: 30 rows; revenues sum to INR 2,998,668.80; largest cell is Bengaluru / Electronics at INR 233,909.45.
-- ------------------------------------------------------------------------------
-- name: q2_three_table_join
SELECT 
    c.city,
    p.category,
    ROUND(SUM(o.order_amount), 2) AS revenue
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city ASC, revenue DESC;


-- ------------------------------------------------------------------------------
-- Q3: Scalar Subquery
-- Products priced above the average catalogue price.
-- Reference: 11 products.
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
-- Q4: LEFT JOIN
-- Customers who have no completed order.
-- Reference: 25 customers.
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
-- Q5: Common Table Expression (CTE)
-- One CTE computes each customer's total completed spend and a second computes
-- the average of those totals; the main query returns customers whose total is above average.
-- Reference: 550 customers; average INR 2,552.06.
-- ------------------------------------------------------------------------------
-- name: q5_cte_above_average_spend
WITH customer_spending AS (
    SELECT 
        customer_id,
        SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
average_spending AS (
    SELECT 
        AVG(total_spend) AS mean_spend
    FROM customer_spending
)
SELECT 
    cs.customer_id,
    ROUND(cs.total_spend, 2) AS total_spend_inr
FROM customer_spending cs, average_spending av
WHERE cs.total_spend > av.mean_spend
ORDER BY cs.total_spend DESC;


-- ------------------------------------------------------------------------------
-- Q6: Ranking Window Functions
-- For each customer with completed spend, compute total spend and then
-- RANK(), DENSE_RANK() and ROW_NUMBER() using OVER (PARTITION BY city ORDER BY total_spend DESC).
-- Reference: top per city (Bengaluru C0082, Chennai C0930, Delhi C0291, Hyderabad C0743,
-- Mumbai C0693, Pune C0426); exactly 8 rows have RANK() != ROW_NUMBER().
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
ORDER BY city ASC, total_spend_inr DESC;


-- ------------------------------------------------------------------------------
-- Q7: Running Total Window Function
-- Daily completed revenue with a cumulative revenue column computed as SUM(revenue) OVER (ORDER BY day).
-- Reference: 183 daily rows; last cumulative value is INR 2,998,668.80.
-- ------------------------------------------------------------------------------
-- name: q7_daily_running_revenue
WITH daily_revenue AS (
    SELECT 
        SUBSTR(order_ts, 1, 10) AS day,
        ROUND(SUM(order_amount), 2) AS daily_revenue_inr
    FROM orders
    WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
)
SELECT 
    day,
    daily_revenue_inr,
    ROUND(SUM(daily_revenue_inr) OVER (ORDER BY day ASC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), 2) AS running_total_revenue_inr
FROM daily_revenue
ORDER BY day ASC;
