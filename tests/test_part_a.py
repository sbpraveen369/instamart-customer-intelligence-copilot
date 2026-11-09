"""
tests/test_part_a.py
Comprehensive test suite for Part A Data Foundation
"""

import os
import sqlite3
import unittest
import pandas as pd

CLEAN_DIR = "data/clean"
DB_PATH   = "data/instamart.db"


class TestPartADataFoundation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cust = pd.read_csv(f"{CLEAN_DIR}/customers.csv")
        cls.prod = pd.read_csv(f"{CLEAN_DIR}/products.csv")
        cls.orders = pd.read_csv(f"{CLEAN_DIR}/orders.csv")

    def test_clean_customers_count(self):
        self.assertEqual(len(self.cust), 1200)

    def test_clean_customers_city_distribution(self):
        counts = self.cust["city"].value_counts().to_dict()
        expected = {
            "Pune": 227,
            "Delhi": 210,
            "Bengaluru": 197,
            "Chennai": 191,
            "Hyderabad": 191,
            "Mumbai": 184
        }
        for city, exp_cnt in expected.items():
            self.assertEqual(counts.get(city), exp_cnt, f"Mismatch for {city}")

    def test_clean_customers_gender_distribution(self):
        counts = self.cust["gender"].value_counts().to_dict()
        self.assertEqual(counts.get("female"), 610)
        self.assertEqual(counts.get("male"), 560)
        self.assertEqual(counts.get("unknown"), 30)

    def test_clean_customers_no_missing_ages(self):
        self.assertEqual(self.cust["age"].isna().sum(), 0)

    def test_clean_orders_count(self):
        self.assertEqual(len(self.orders), 12309)

    def test_clean_orders_status_distribution(self):
        counts = self.orders["status"].value_counts().to_dict()
        self.assertEqual(counts.get("completed"), 11110)
        self.assertEqual(counts.get("cancelled"), 1199)
        self.assertNotIn("pending", counts)

    def test_clean_orders_no_missing_amounts(self):
        self.assertEqual(self.orders["order_amount"].isna().sum(), 0)

    def test_clean_orders_delivery_minutes(self):
        comp = self.orders[self.orders["status"] == "completed"]
        canc = self.orders[self.orders["status"] == "cancelled"]
        self.assertEqual(comp["delivery_minutes"].isna().sum(), 0)
        self.assertEqual(canc["delivery_minutes"].isna().sum(), 1199)

    def test_completed_revenue_exact(self):
        comp = self.orders[self.orders["status"] == "completed"]
        total_rev = round(float(comp["order_amount"].sum()), 2)
        self.assertEqual(total_rev, 2998668.80)

    def test_clean_products(self):
        self.assertEqual(len(self.prod), 40)
        self.assertEqual(self.prod["price_inr"].isna().sum(), 0)
        self.assertTrue((self.prod["price_inr"] > 0).all())

    def test_sqlite_database_tables(self):
        self.assertTrue(os.path.exists(DB_PATH))
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        c_cnt = cur.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        p_cnt = cur.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        o_cnt = cur.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        self.assertEqual(c_cnt, 1200)
        self.assertEqual(p_cnt, 40)
        self.assertEqual(o_cnt, 12309)
        con.close()


if __name__ == "__main__":
    unittest.main()
