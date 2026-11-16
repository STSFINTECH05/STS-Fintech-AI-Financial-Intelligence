import os
import sys
from datetime import datetime, date
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import psycopg2.extensions

# Register NumPy adapters with psycopg2 for defense-in-depth
try:
    psycopg2.extensions.register_adapter(np.int64, lambda val: psycopg2.extensions.AsIs(int(val)))
    psycopg2.extensions.register_adapter(np.int32, lambda val: psycopg2.extensions.AsIs(int(val)))
    psycopg2.extensions.register_adapter(np.float64, lambda val: psycopg2.extensions.AsIs(float(val)))
    psycopg2.extensions.register_adapter(np.float32, lambda val: psycopg2.extensions.AsIs(float(val)))
    psycopg2.extensions.register_adapter(np.bool_, lambda val: psycopg2.extensions.AsIs(bool(val)))
except Exception:
    pass

DATABASE_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "sts_financial_ai",
    "user": "stsfintech",
    "password": ""
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")


def connect_database():
    print("Connecting to PostgreSQL (sts_financial_ai)...")
    connection = psycopg2.connect(
        host=DATABASE_CONFIG["host"],
        port=DATABASE_CONFIG["port"],
        database=DATABASE_CONFIG["database"],
        user=DATABASE_CONFIG["user"],
        password=DATABASE_CONFIG["password"]
    )
    print("✅ PostgreSQL connection successful")
    return connection


def sanitize_value(val):
    """
    Safely converts NumPy scalars, pandas timestamps, and missing values
    into pure Python native types (int, float, bool, str, None) for psycopg2.
    Prevents PostgreSQL syntax errors caused by NumPy 2.0+ scalar repr formatting.
    """
    if val is None:
        return None

    # Check for pandas/numpy missing values (NaN, NaT, pd.NA)
    if pd.isna(val):
        return None

    # Handle pandas Timestamp or datetime objects
    if isinstance(val, (pd.Timestamp, datetime)):
        return val.to_pydatetime() if isinstance(val, pd.Timestamp) else val

    # Handle dates
    if isinstance(val, date):
        return val

    # Handle NumPy scalars using .item()
    if isinstance(val, (np.generic,)):
        val_item = val.item()
        # In case item() is float NaN or Inf
        if isinstance(val_item, float) and (np.isnan(val_item) or np.isinf(val_item)):
            return None
        return val_item

    # Handle standard Python floats that are NaN or Inf
    if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
        return None

    return val


def get_table_row_count(connection, table_name):
    """Returns the current row count of a table."""
    cursor = connection.cursor()
    cursor.execute(f"SELECT count(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    cursor.close()
    return count


def load_table(
    connection,
    csv_file,
    table_name,
    columns,
    primary_key=None,
    data_dir=DATA_DIR,
    skip_if_populated=True
):
    path = os.path.join(data_dir, csv_file)
    print(f"\nProcessing {csv_file} for table '{table_name}'...")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Source CSV not found: {path}")

    # Check existing records to prevent accidental duplication
    existing_count = get_table_row_count(connection, table_name)
    if skip_if_populated and existing_count > 0:
        print(f"ℹ️ Table '{table_name}' already contains {existing_count:,} records. Skipping reload to preserve existing data.")
        return existing_count

    df = pd.read_csv(path)
    print(f"Records found in CSV: {len(df):,}")

    values = []
    # Using itertuples preserves each column's specific dtype (int stays int, float stays float)
    for row in df[columns].itertuples(index=False):
        record = [sanitize_value(val) for val in row]
        values.append(tuple(record))

    cursor = connection.cursor()

    # Build query with ON CONFLICT if primary key specified for idempotent replay
    if primary_key:
        query = f"""
            INSERT INTO {table_name}
            ({', '.join(columns)})
            VALUES %s
            ON CONFLICT ({primary_key}) DO NOTHING
        """
    else:
        query = f"""
            INSERT INTO {table_name}
            ({', '.join(columns)})
            VALUES %s
        """

    execute_values(
        cursor,
        query,
        values,
        page_size=1000
    )

    connection.commit()
    cursor.close()

    new_count = get_table_row_count(connection, table_name)
    inserted = new_count - existing_count
    print(f"✅ Ingestion successful for '{table_name}': {inserted:,} new records inserted (Total rows: {new_count:,})")
    return new_count


def load_budgets_only(connection=None):
    """
    Dedicated function to ingest only the budgets dataset safely.
    Validates table count before and after, preserves rollback on error.
    """
    should_close = False
    if connection is None:
        connection = connect_database()
        should_close = True

    try:
        print("\n======================================")
        print("BUDGET DATA INGESTION (REPAIR)")
        print("======================================")

        count = load_table(
            connection,
            "budgets.csv",
            "budgets",
            [
                "budget_id",
                "department_id",
                "budget_year",
                "budget_month",
                "allocated_amount",
                "actual_amount",
                "remaining_amount",
                "utilization_percentage"
            ],
            primary_key="budget_id",
            skip_if_populated=False
        )

        return count

    except Exception as error:
        connection.rollback()
        print(f"\n❌ Ingestion error during budgets loading: {error}")
        raise error
    finally:
        if should_close:
            connection.close()
            print("PostgreSQL connection closed.")


def main(table_filter=None):
    connection = connect_database()

    try:
        if table_filter == "budgets":
            load_budgets_only(connection)
            return

        # -------------------------
        # Departments
        # -------------------------
        load_table(
            connection,
            "departments.csv",
            "departments",
            [
                "department_id",
                "department_code",
                "department_name",
                "manager_name",
                "location",
                "annual_budget"
            ],
            primary_key="department_id"
        )

        # -------------------------
        # Employees
        # -------------------------
        load_table(
            connection,
            "employees.csv",
            "employees",
            [
                "employee_id",
                "employee_name",
                "department_id",
                "designation",
                "employment_type",
                "joining_date",
                "location",
                "salary_grade",
                "status"
            ],
            primary_key="employee_id"
        )

        # -------------------------
        # Vendors
        # -------------------------
        load_table(
            connection,
            "vendors.csv",
            "vendors",
            [
                "vendor_id",
                "vendor_name",
                "vendor_category",
                "location",
                "contact_email",
                "risk_level",
                "payment_terms_days",
                "vendor_rating",
                "status"
            ],
            primary_key="vendor_id"
        )

        # -------------------------
        # Salaries
        # -------------------------
        load_table(
            connection,
            "salaries.csv",
            "salaries",
            [
                "salary_id",
                "employee_id",
                "salary_month",
                "basic_salary",
                "hra",
                "allowances",
                "bonus",
                "deductions",
                "gross_salary",
                "net_salary"
            ],
            primary_key="salary_id"
        )

        # -------------------------
        # Transactions
        # -------------------------
        load_table(
            connection,
            "transactions.csv",
            "transactions",
            [
                "transaction_id",
                "employee_id",
                "department_id",
                "vendor_id",
                "transaction_date",
                "transaction_type",
                "category",
                "description",
                "amount",
                "payment_method",
                "currency",
                "is_business_expense",
                "is_anomaly",
                "anomaly_score"
            ],
            primary_key="transaction_id"
        )

        # -------------------------
        # Bills
        # -------------------------
        load_table(
            connection,
            "bills.csv",
            "bills",
            [
                "bill_id",
                "vendor_id",
                "department_id",
                "bill_date",
                "due_date",
                "bill_amount",
                "tax_amount",
                "total_amount",
                "payment_status",
                "payment_date",
                "description"
            ],
            primary_key="bill_id"
        )

        # -------------------------
        # Invoices
        # -------------------------
        load_table(
            connection,
            "invoices.csv",
            "invoices",
            [
                "invoice_id",
                "vendor_id",
                "department_id",
                "invoice_date",
                "due_date",
                "subtotal",
                "tax_amount",
                "total_amount",
                "invoice_status",
                "payment_status",
                "is_duplicate",
                "duplicate_score"
            ],
            primary_key="invoice_id"
        )

        # -------------------------
        # Budgets
        # -------------------------
        load_table(
            connection,
            "budgets.csv",
            "budgets",
            [
                "budget_id",
                "department_id",
                "budget_year",
                "budget_month",
                "allocated_amount",
                "actual_amount",
                "remaining_amount",
                "utilization_percentage"
            ],
            primary_key="budget_id",
            skip_if_populated=False
        )

        print("\n======================================")
        print("DATABASE INGESTION COMPLETED")
        print("======================================")

    except Exception as error:
        connection.rollback()
        print("\n❌ DATABASE ERROR")
        print(error)
        raise error
    finally:
        connection.close()
        print("PostgreSQL connection closed.")


if __name__ == "__main__":
    target = sys.argv[1].replace("--table=", "").replace("--", "") if len(sys.argv) > 1 else None
    main(target)
