"""
tests/test_part_a.py
Automated test suite for Part A Data Foundation
"""

import os
import sqlite3
import unittest
import pandas as pd

RAW_DIR   = "data/raw"
CLEAN_DIR = "data/clean"
DB_PATH   = "data/instamart.db"


class TestPartADataFoundation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.raw_orders = pd.read_csv(f"{RAW_DIR}/orders.csv")
        cls.raw_cust   = pd.read_csv(f"{RAW_DIR}/customers.csv")
        cls.clean_cust = pd.read_csv(f"{CLEAN_DIR}/customers.csv")
        cls.clean_prod = pd.read_csv(f"{CLEAN_DIR}/products.csv")
        cls.clean_orders = pd.read_csv(f"{CLEAN_DIR}/orders.csv")

    def test_raw_orders_metrics(self):
        self.assertEqual(len(self.raw_orders), 13084)
        self.assertEqual(self.raw_orders.duplicated().sum(), 119)
        self.assertEqual((self.raw_orders["status"] == "pending").sum(), 656)
        self.assertEqual(self.raw_orders["order_id"].nunique(), 12309)
        oa_num = pd.to_numeric(self.raw_orders["order_amount"], errors="coerce")
        self.assertEqual(oa_num.isna().sum(), 120)
        dm_num = pd.to_numeric(self.raw_orders["delivery_minutes"], errors="coerce")
        self.assertEqual((dm_num < 0).sum(), 106)
        self.assertEqual(self.raw_orders["delivery_minutes"].isna().sum(), 2541)

    def test_raw_customers_metrics(self):
        self.assertEqual(self.raw_cust["age"].isna().sum(), 61)
        age_num = pd.to_numeric(self.raw_cust["age"], errors="coerce")
        self.assertEqual(((age_num < 10) | (age_num > 100)).sum(), 6)
        self.assertEqual(self.raw_cust["city"].nunique(), 24)
        self.assertEqual(self.raw_cust["gender"].dropna().nunique(), 10)
        self.assertEqual(self.raw_cust["gender"].isna().sum(), 30)

    def test_clean_customers(self):
        self.assertEqual(len(self.clean_cust), 1200)
        exp_cities = {"Pune": 227, "Delhi": 210, "Bengaluru": 197, "Chennai": 191, "Hyderabad": 191, "Mumbai": 184}
        city_counts = self.clean_cust["city"].value_counts().to_dict()
        self.assertEqual(city_counts, exp_cities)
        exp_genders = {"female": 610, "male": 560, "unknown": 30}
        self.assertEqual(self.clean_cust["gender"].value_counts().to_dict(), exp_genders)
        self.assertEqual(self.clean_cust["age"].isna().sum(), 0)
        self.assertTrue(self.clean_cust["age"].between(18, 60).all())

    def test_clean_orders(self):
        self.assertEqual(len(self.clean_orders), 12309)
        status_cnt = self.clean_orders["status"].value_counts().to_dict()
        self.assertEqual(status_cnt.get("completed"), 11110)
        self.assertEqual(status_cnt.get("cancelled"), 1199)
        self.assertNotIn("pending", status_cnt)
        self.assertEqual(self.clean_orders["order_amount"].isna().sum(), 0)

        comp = self.clean_orders[self.clean_orders["status"] == "completed"]
        canc = self.clean_orders[self.clean_orders["status"] == "cancelled"]
        self.assertEqual(comp["delivery_minutes"].isna().sum(), 0)
        self.assertEqual(canc["delivery_minutes"].isna().sum(), 1199)
        self.assertEqual(round(float(comp["order_amount"].sum()), 2), 2998668.80)

    def test_clean_products(self):
        self.assertEqual(len(self.clean_prod), 40)
        self.assertEqual(self.clean_prod["product_id"].isna().sum(), 0)
        self.assertTrue((self.clean_prod["price_inr"] > 0).all())

    def test_database_tables(self):
        self.assertTrue(os.path.exists(DB_PATH))
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        self.assertEqual(cur.execute("SELECT COUNT(*) FROM customers").fetchone()[0], 1200)
        self.assertEqual(cur.execute("SELECT COUNT(*) FROM products").fetchone()[0], 40)
        self.assertEqual(cur.execute("SELECT COUNT(*) FROM orders").fetchone()[0], 12309)
        con.close()


if __name__ == "__main__":
    unittest.main()
