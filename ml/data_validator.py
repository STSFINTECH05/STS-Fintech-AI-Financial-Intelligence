import pandas as pd
import os

DATA_DIR = "data/raw"


def load_file(filename):
    path = os.path.join(DATA_DIR, filename)

    if not os.path.exists(path):
        print(f"❌ Missing file: {filename}")
        return None

    return pd.read_csv(path)


def check_missing_values(df, name):
    print(f"\n{name} - Missing Values")

    missing = df.isnull().sum()

    missing = missing[missing > 0]

    if len(missing) == 0:
        print("✅ No missing values")
    else:
        print(missing)


def check_duplicates(df, name, column):
    duplicates = df[column].duplicated().sum()

    print(
        f"{name} duplicate {column}: "
        f"{duplicates}"
    )

    if duplicates == 0:
        print("✅ No duplicate IDs")
    else:
        print("⚠️ Duplicate IDs detected")


def validate_transactions():

    df = load_file("transactions.csv")

    print("\n======================================")
    print("TRANSACTION VALIDATION")
    print("======================================")

    print("Rows:", len(df))

    print("Columns:", len(df.columns))

    check_missing_values(
        df,
        "Transactions"
    )

    check_duplicates(
        df,
        "Transactions",
        "transaction_id"
    )

    print("\nTransaction Amount Statistics")

    print(
        df["amount"].describe()
    )

    print("\nAnomaly Distribution")

    print(
        df["is_anomaly"].value_counts()
    )

    print("\nAnomaly Percentage")

    anomaly_percentage = (
        df["is_anomaly"].mean() * 100
    )

    print(
        f"{anomaly_percentage:.2f}%"
    )

    print("\nNegative Amounts")

    negative_amounts = (
        df["amount"] < 0
    ).sum()

    print(
        negative_amounts
    )

    if negative_amounts == 0:
        print("✅ No negative transactions")


def validate_employees():

    df = load_file("employees.csv")

    print("\n======================================")
    print("EMPLOYEE VALIDATION")
    print("======================================")

    print("Rows:", len(df))

    check_missing_values(
        df,
        "Employees"
    )

    check_duplicates(
        df,
        "Employees",
        "employee_id"
    )


def validate_vendors():

    df = load_file("vendors.csv")

    print("\n======================================")
    print("VENDOR VALIDATION")
    print("======================================")

    print("Rows:", len(df))

    check_missing_values(
        df,
        "Vendors"
    )

    check_duplicates(
        df,
        "Vendors",
        "vendor_id"
    )


def validate_invoices():

    df = load_file("invoices.csv")

    print("\n======================================")
    print("INVOICE VALIDATION")
    print("======================================")

    print("Rows:", len(df))

    check_missing_values(
        df,
        "Invoices"
    )

    check_duplicates(
        df,
        "Invoices",
        "invoice_id"
    )

    print("\nDuplicate Invoice Distribution")

    print(
        df["is_duplicate"].value_counts()
    )


def main():

    print("\n")
    print("==============================================")
    print("STS FINTECH AI - DATA QUALITY VALIDATION")
    print("==============================================")

    validate_employees()

    validate_vendors()

    validate_transactions()

    validate_invoices()

    print("\n==============================================")
    print("DATA VALIDATION COMPLETED")
    print("==============================================")


if __name__ == "__main__":
    main()