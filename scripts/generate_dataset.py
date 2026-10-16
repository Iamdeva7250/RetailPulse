import random
from datetime import datetime

import numpy as np
import pandas as pd

from src.config import RAW_DATA_DIR


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


NUM_CUSTOMERS = 3000
NUM_PRODUCTS = 100
NUM_TRANSACTIONS = 50000

START_DATE = "2024-01-01"
END_DATE = "2025-12-31"


# ============================================================
# PRODUCT INFORMATION
# ============================================================

PRODUCT_CATEGORIES = {
    "Electronics": [
        "Wireless Headphones",
        "Smartphone",
        "Laptop",
        "Smart Watch",
        "Bluetooth Speaker",
        "Tablet",
        "Power Bank",
        "USB Cable",
        "Keyboard",
        "Mouse",
    ],
    "Clothing": [
        "T-Shirt",
        "Jeans",
        "Jacket",
        "Sneakers",
        "Hoodie",
        "Shirt",
        "Dress",
        "Shorts",
        "Sweater",
        "Cap",
    ],
    "Home & Kitchen": [
        "Coffee Maker",
        "Blender",
        "Cookware Set",
        "Dinner Set",
        "Water Bottle",
        "Storage Box",
        "Bedsheet",
        "Pillow",
        "Lamp",
        "Vacuum Cleaner",
    ],
    "Beauty": [
        "Face Wash",
        "Moisturizer",
        "Shampoo",
        "Perfume",
        "Lipstick",
        "Face Cream",
        "Sunscreen",
        "Hair Oil",
        "Body Lotion",
        "Serum",
    ],
    "Sports": [
        "Yoga Mat",
        "Running Shoes",
        "Cricket Bat",
        "Football",
        "Tennis Racket",
        "Dumbbells",
        "Gym Bag",
        "Skipping Rope",
        "Cycling Helmet",
        "Sports Bottle",
    ],
}


PAYMENT_METHODS = [
    "Credit Card",
    "Debit Card",
    "UPI",
    "Cash",
    "Net Banking",
]


REGIONS = [
    "North",
    "South",
    "East",
    "West",
    "Central",
]


GENDERS = [
    "Male",
    "Female",
    "Other",
]


# ============================================================
# PRODUCT GENERATION
# ============================================================

def generate_products():
    products = []

    product_counter = 1

    for category, product_names in PRODUCT_CATEGORIES.items():

        for product_name in product_names:

            for _ in range(2):

                product_id = f"P{product_counter:04d}"

                base_price = round(
                    np.random.uniform(200, 80000),
                    2,
                )

                stock_quantity = int(
                    np.random.randint(20, 500)
                )

                supplier_lead_time = int(
                    np.random.randint(2, 15)
                )

                reorder_level = int(
                    np.random.randint(10, 100)
                )

                products.append(
                    {
                        "product_id": product_id,
                        "product_name": product_name,
                        "category": category,
                        "base_price": base_price,
                        "stock_quantity": stock_quantity,
                        "supplier_lead_time": supplier_lead_time,
                        "reorder_level": reorder_level,
                    }
                )

                product_counter += 1

    return pd.DataFrame(products)


# ============================================================
# CUSTOMER GENERATION
# ============================================================

def generate_customers():

    customers = []

    for i in range(1, NUM_CUSTOMERS + 1):

        customer_id = f"C{i:05d}"

        age = int(
            np.clip(
                np.random.normal(35, 12),
                18,
                75,
            )
        )

        gender = random.choice(GENDERS)

        region = random.choice(REGIONS)

        customers.append(
            {
                "customer_id": customer_id,
                "customer_age": age,
                "customer_gender": gender,
                "region": region,
            }
        )

    return pd.DataFrame(customers)


# ============================================================
# TRANSACTION GENERATION
# ============================================================

def generate_transactions(
    customers,
    products,
):

    transaction_dates = pd.date_range(
        start=START_DATE,
        end=END_DATE,
        freq="D",
    )

    customer_ids = customers["customer_id"].tolist()

    product_ids = products["product_id"].tolist()

    records = []

    for i in range(1, NUM_TRANSACTIONS + 1):

        customer_id = random.choice(
            customer_ids
        )

        product_id = random.choice(
            product_ids
        )

        customer = customers[
            customers["customer_id"] == customer_id
        ].iloc[0]

        product = products[
            products["product_id"] == product_id
        ].iloc[0]

        transaction_date = random.choice(
            transaction_dates
        )

        quantity = int(
            np.random.choice(
                [1, 2, 3, 4, 5],
                p=[
                    0.50,
                    0.25,
                    0.12,
                    0.08,
                    0.05,
                ],
            )
        )

        unit_price = float(
            product["base_price"]
        )

        discount = float(
            np.random.choice(
                [0, 5, 10, 15, 20],
                p=[
                    0.35,
                    0.25,
                    0.20,
                    0.15,
                    0.05,
                ],
            )
        )

        gross_amount = (
            quantity * unit_price
        )

        sales_amount = (
            gross_amount
            * (1 - discount / 100)
        )

        sales_amount = round(
            sales_amount,
            2,
        )

        payment_method = random.choice(
            PAYMENT_METHODS
        )

        records.append(
            {
                "transaction_id": f"T{i:07d}",

                "customer_id": customer_id,

                "product_id": product_id,

                "product_name": product[
                    "product_name"
                ],

                "category": product[
                    "category"
                ],

                "transaction_date": transaction_date,

                "quantity": quantity,

                "unit_price": round(
                    unit_price,
                    2,
                ),

                "discount": discount,

                "sales_amount": sales_amount,

                "payment_method": payment_method,

                "region": customer[
                    "region"
                ],

                "customer_age": customer[
                    "customer_age"
                ],

                "customer_gender": customer[
                    "customer_gender"
                ],

                "stock_quantity": product[
                    "stock_quantity"
                ],

                "supplier_lead_time": product[
                    "supplier_lead_time"
                ],

                "reorder_level": product[
                    "reorder_level"
                ],
            }
        )

    return pd.DataFrame(records)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("RetailPulse Dataset Generator")
    print("=" * 60)

    print("\nGenerating products...")

    products = generate_products()

    print(
        f"Products generated: {len(products)}"
    )

    print("\nGenerating customers...")

    customers = generate_customers()

    print(
        f"Customers generated: {len(customers)}"
    )

    print("\nGenerating transactions...")

    sales = generate_transactions(
        customers,
        products,
    )

    print(
        f"Transactions generated: {len(sales)}"
    )

    # --------------------------------------------------------
    # Sort data
    # --------------------------------------------------------

    sales = sales.sort_values(
        "transaction_date"
    )

    # --------------------------------------------------------
    # Reset index
    # --------------------------------------------------------

    sales = sales.reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Convert date to string for CSV
    # --------------------------------------------------------

    sales["transaction_date"] = (
        pd.to_datetime(
            sales["transaction_date"]
        ).dt.strftime(
            "%Y-%m-%d"
        )
    )

    # --------------------------------------------------------
    # Save dataset
    # --------------------------------------------------------

    output_path = (
        RAW_DATA_DIR / "sales.csv"
    )

    sales.to_csv(
        output_path,
        index=False,
    )

    print("\nDataset saved successfully!")

    print(
        f"Location: {output_path}"
    )

    print("\nDataset shape:")

    print(
        sales.shape
    )

    print("\nFirst 5 rows:")

    print(
        sales.head()
    )

    print("\nColumns:")

    for column in sales.columns:
        print(
            f" - {column}"
        )

    print("\nMissing values:")

    print(
        sales.isnull().sum()
    )

    print("\nDataset generation complete.")


if __name__ == "__main__":
    main()