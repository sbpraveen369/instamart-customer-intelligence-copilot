"""
src/part_a/sql_analysis.py
Step 9 — SQL Analysis

Executes all seven queries defined in sql/queries.sql using pandas.read_sql_query.
Displays summaries and outputs to outputs/sql_results.txt.
"""

import os
import re
import sqlite3
import pandas as pd

DB_PATH   = "data/instamart.db"
SQL_PATH  = "sql/queries.sql"
OUT_PATH  = "outputs/sql_results.txt"


def parse_queries(sql_file_path):
    with open(sql_file_path, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = r"-- name:\s*(\w+)\s*\n(.*?)(?=(?:-- name:|\Z))"
    matches = re.findall(pattern, content, flags=re.DOTALL)
    queries = {}
    for name, sql in matches:
        queries[name] = sql.strip()
    return queries


def run_sql_analysis():
    assert os.path.exists(DB_PATH), f"Database not found at {DB_PATH}"
    assert os.path.exists(SQL_PATH), f"SQL file not found at {SQL_PATH}"

    con = sqlite3.connect(DB_PATH)
    queries = parse_queries(SQL_PATH)
    assert len(queries) == 7, f"Expected 7 queries, found {len(queries)}"

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    report_lines = [
        "=" * 80,
        "INSTAMART CUSTOMER INTELLIGENCE & SUPPORT COPILOT — SQL ANALYSIS REPORT",
        "=" * 80,
        f"Database: {DB_PATH}",
        f"Queries executed: {len(queries)}\n"
    ]

    query_labels = {
        "q1_where_and_having": "Q1: WHERE then HAVING (Completed orders > 500, customer total > 8000)",
        "q2_three_table_join": "Q2: Three-Table JOIN (City & Product Category Completed Revenue)",
        "q3_scalar_subquery": "Q3: Scalar Subquery (Products Priced Above Catalogue Average)",
        "q4_left_join": "Q4: LEFT JOIN (Customers Without Completed Orders)",
        "q5_cte_above_average_spend": "Q5: Common Table Expression (Customers with Above-Average Spend)",
        "q6_window_ranking_by_city": "Q6: Ranking Window Functions (RANK, DENSE_RANK, ROW_NUMBER by City)",
        "q7_daily_running_revenue": "Q7: Running Total Window Function (Daily & Cumulative Revenue)"
    }

    results = {}

    for qname, qtitle in query_labels.items():
        sql = queries[qname]
        df = pd.read_sql_query(sql, con)
        results[qname] = df

        header = f"\n{'='*80}\n{qtitle}\n{'='*80}"
        print(header)
        report_lines.append(header)
        report_lines.append(f"SQL Query:\n{sql}\n")
        report_lines.append(f"Row count returned: {len(df)}")

        if len(df) <= 15:
            table_str = df.to_string(index=False)
        else:
            table_str = (
                df.head(10).to_string(index=False)
                + f"\n... [{len(df) - 15} rows omitted] ...\n"
                + df.tail(5).to_string(index=False)
            )

        print(table_str)
        report_lines.append(f"Output:\n{table_str}\n")

    con.close()

    # Save full report
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\nAll 7 SQL queries executed successfully. Report saved to: {OUT_PATH}")
    return results


if __name__ == "__main__":
    run_sql_analysis()
