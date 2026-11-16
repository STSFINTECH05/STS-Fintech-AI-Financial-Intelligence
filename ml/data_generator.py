import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

fake = Faker("en_IN")
Faker.seed(SEED)

OUTPUT_DIR = "data/raw"

NUM_DEPARTMENTS = 50
NUM_EMPLOYEES = 10_000
NUM_VENDORS = 2_000

NUM_SALARIES = 60_000
NUM_TRANSACTIONS = 10_000
NUM_BILLS = 5_000
NUM_INVOICES = 5_000
NUM_BUDGETS = 5_000


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def random_date(start_date, end_date):
    delta = end_date - start_date
    random_days = random.randint(0, delta.days)
    return start_date + timedelta(days=random_days)


START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2026, 10, 31)


# ============================================================
# 1. DEPARTMENTS
# ============================================================

print("Generating departments...")

department_names = [
    "Information Technology",
    "Finance",
    "Human Resources",
    "Sales",
    "Marketing",
    "Procurement",
    "Operations",
    "Administration",
    "Research and Development",
    "Legal",
    "Customer Support",
    "Logistics",
    "Manufacturing",
    "Quality Assurance",
    "Engineering",
    "Security",
    "Business Development",
    "Supply Chain",
    "Audit",
    "Facilities"
]

departments = []

for i in range(1, NUM_DEPARTMENTS + 1):

    name = department_names[(i - 1) % len(department_names)]

    if i > len(department_names):
        name = f"{name} {((i - 1) // len(department_names)) + 1}"

    departments.append({
        "department_id": i,
        "department_code": f"DEPT{i:03d}",
        "department_name": name,
        "manager_name": fake.name(),
        "location": random.choice([
            "Chennai",
            "Bangalore",
            "Vellore",
            "Hyderabad",
            "Pune",
            "Mumbai"
        ]),
        "annual_budget": round(random.uniform(5_000_000, 100_000_000), 2)
    })

departments_df = pd.DataFrame(departments)


# ============================================================
# 2. EMPLOYEES
# ============================================================

print("Generating employees...")

designations = [
    "Software Engineer",
    "Senior Software Engineer",
    "Manager",
    "Senior Manager",
    "Analyst",
    "Senior Analyst",
    "Executive",
    "Consultant",
    "Team Lead",
    "Director"
]

employment_types = [
    "FULL_TIME",
    "FULL_TIME",
    "FULL_TIME",
    "CONTRACT",
    "INTERN"
]

employees = []

for i in range(1, NUM_EMPLOYEES + 1):

    employee_id = f"EMP{i:06d}"

    employees.append({
        "employee_id": employee_id,
        "employee_name": fake.name(),
        "department_id": random.randint(1, NUM_DEPARTMENTS),
        "designation": random.choice(designations),
        "employment_type": random.choice(employment_types),
        "joining_date": random_date(
            datetime(2015, 1, 1),
            datetime(2026, 1, 1)
        ).date(),
        "location": random.choice([
            "Chennai",
            "Bangalore",
            "Vellore",
            "Hyderabad",
            "Pune",
            "Mumbai"
        ]),
        "salary_grade": random.choice([
            "G1", "G2", "G3", "G4", "G5", "G6"
        ]),
        "status": random.choices(
            ["ACTIVE", "INACTIVE"],
            weights=[95, 5]
        )[0]
    })

employees_df = pd.DataFrame(employees)


# ============================================================
# 3. VENDORS
# ============================================================

print("Generating vendors...")

vendor_categories = [
    "IT Services",
    "Software",
    "Office Supplies",
    "Travel",
    "Transportation",
    "Consulting",
    "Cloud Services",
    "Maintenance",
    "Security",
    "Marketing",
    "Legal Services",
    "Professional Services",
    "Hardware",
    "Telecommunications",
    "Facilities"
]

vendors = []

for i in range(1, NUM_VENDORS + 1):

    vendors.append({
        "vendor_id": f"VEN{i:05d}",
        "vendor_name": fake.company(),
        "vendor_category": random.choice(vendor_categories),
        "location": random.choice([
            "Chennai",
            "Bangalore",
            "Mumbai",
            "Delhi",
            "Hyderabad",
            "Pune"
        ]),
        "contact_email": fake.company_email(),
        "risk_level": random.choices(
            ["LOW", "MEDIUM", "HIGH"],
            weights=[70, 25, 5]
        )[0],
        "payment_terms_days": random.choice([
            15, 30, 45, 60, 90
        ]),
        "vendor_rating": round(
            random.uniform(2.5, 5.0), 2
        ),
        "status": "ACTIVE"
    })

vendors_df = pd.DataFrame(vendors)


# ============================================================
# 4. SALARIES
# ============================================================

print("Generating salaries...")

salary_records = []

salary_months = pd.date_range(
    start="2025-01-01",
    end="2026-10-01",
    freq="MS"
)

salary_employee_ids = np.random.choice(
    employees_df["employee_id"],
    size=NUM_SALARIES,
    replace=True
)

for employee_id, month in zip(
    salary_employee_ids,
    np.random.choice(salary_months, NUM_SALARIES)
):

    basic = random.uniform(25_000, 180_000)

    hra = basic * random.uniform(0.20, 0.40)

    allowances = basic * random.uniform(0.05, 0.20)

    bonus = random.choice([
        0,
        0,
        0,
        basic * 0.10,
        basic * 0.20
    ])

    deductions = basic * random.uniform(0.05, 0.15)

    gross = basic + hra + allowances + bonus

    net = gross - deductions

    salary_records.append({
        "salary_id": len(salary_records) + 1,
        "employee_id": employee_id,
        "salary_month": pd.Timestamp(month).date(),
        "basic_salary": round(basic, 2),
        "hra": round(hra, 2),
        "allowances": round(allowances, 2),
        "bonus": round(bonus, 2),
        "deductions": round(deductions, 2),
        "gross_salary": round(gross, 2),
        "net_salary": round(net, 2)
    })

salaries_df = pd.DataFrame(salary_records)


# ============================================================
# 5. TRANSACTIONS
# ============================================================

print("Generating transactions...")

transaction_categories = {
    "IT": [
        "Software",
        "Cloud Services",
        "Hardware",
        "IT Services"
    ],
    "OPERATIONS": [
        "Maintenance",
        "Transportation",
        "Facilities"
    ],
    "GENERAL": [
        "Travel",
        "Meals",
        "Office Supplies",
        "Professional Services",
        "Marketing"
    ]
}

transactions = []

for i in range(1, NUM_TRANSACTIONS + 1):

    employee = employees_df.sample(1).iloc[0]
    vendor = vendors_df.sample(1).iloc[0]

    category_group = random.choice(
        list(transaction_categories.keys())
    )

    category = random.choice(
        transaction_categories[category_group]
    )

    amount = np.random.lognormal(
        mean = np.log(5_000),
        sigma = 1.0
    )

    amount = max(100, min(amount, 500_000))

    transaction_date = random_date(
        START_DATE,
        END_DATE
    )

    # --------------------------------------------------------
    # Inject anomalies
    # --------------------------------------------------------

    is_anomaly = False

    # Large unusual transaction
    if random.random() < 0.03:
        amount *= random.uniform(5, 20)
        is_anomaly = True

    # Weekend transaction
    if transaction_date.weekday() >= 5:
        if random.random() < 0.15:
            is_anomaly = True

    # High-risk vendor
    if vendor["risk_level"] == "HIGH":
        if random.random() < 0.25:
            is_anomaly = True

    anomaly_score = (
        random.uniform(0.75, 0.99)
        if is_anomaly
        else random.uniform(0.01, 0.30)
    )

    transactions.append({
        "transaction_id": i,
        "employee_id": employee["employee_id"],
        "department_id": employee["department_id"],
        "vendor_id": vendor["vendor_id"],
        "transaction_date": transaction_date,
        "transaction_type": random.choice([
            "EXPENSE",
            "PAYMENT",
            "REIMBURSEMENT"
        ]),
        "category": category,
        "description": f"{category} expense",
        "amount": round(amount, 2),
        "payment_method": random.choice([
            "BANK_TRANSFER",
            "CREDIT_CARD",
            "DEBIT_CARD",
            "UPI",
            "CASH"
        ]),
        "currency": "INR",
        "is_business_expense": True,
        "is_anomaly": is_anomaly,
        "anomaly_score": round(anomaly_score, 4)
    })

transactions_df = pd.DataFrame(transactions)


# ============================================================
# 6. BILLS
# ============================================================

print("Generating bills...")

bills = []

for i in range(1, NUM_BILLS + 1):

    vendor = vendors_df.sample(1).iloc[0]
    department_id = random.randint(1, NUM_DEPARTMENTS)

    bill_date = random_date(
        START_DATE,
        END_DATE
    )

    due_date = bill_date + timedelta(
        days=int(vendor["payment_terms_days"])
    )

    bill_amount = random.uniform(
        1_000,
        500_000
    )

    tax = bill_amount * 0.18

    bills.append({
        "bill_id": f"BILL{i:07d}",
        "vendor_id": vendor["vendor_id"],
        "department_id": department_id,
        "bill_date": bill_date.date(),
        "due_date": due_date.date(),
        "bill_amount": round(bill_amount, 2),
        "tax_amount": round(tax, 2),
        "total_amount": round(bill_amount + tax, 2),
        "payment_status": random.choice([
            "PAID",
            "PENDING",
            "OVERDUE"
        ]),
        "payment_date": None,
        "description": "Vendor bill"
    })

bills_df = pd.DataFrame(bills)


# ============================================================
# 7. INVOICES
# ============================================================

print("Generating invoices...")

invoices = []

for i in range(1, NUM_INVOICES + 1):

    vendor = vendors_df.sample(1).iloc[0]

    invoice_date = random_date(
        START_DATE,
        END_DATE
    )

    subtotal = random.uniform(
        2_000,
        1_000_000
    )

    tax = subtotal * 0.18

    invoices.append({
        "invoice_id": f"INV{i:08d}",
        "vendor_id": vendor["vendor_id"],
        "department_id": random.randint(
            1,
            NUM_DEPARTMENTS
        ),
        "invoice_date": invoice_date.date(),
        "due_date": (
            invoice_date +
            timedelta(days=int(vendor["payment_terms_days"]))
        ).date(),
        "subtotal": round(subtotal, 2),
        "tax_amount": round(tax, 2),
        "total_amount": round(subtotal + tax, 2),
        "invoice_status": random.choice([
            "APPROVED",
            "PENDING",
            "REJECTED"
        ]),
        "payment_status": random.choice([
            "PAID",
            "PENDING",
            "OVERDUE"
        ]),
        "is_duplicate": random.random() < 0.02,
        "duplicate_score": round(
            random.uniform(0.80, 0.99),
            4
        )
    })

invoices_df = pd.DataFrame(invoices)


# ============================================================
# 8. BUDGETS
# ============================================================

print("Generating budgets...")

budgets = []

for i in range(1, NUM_BUDGETS + 1):

    department_id = random.randint(
        1,
        NUM_DEPARTMENTS
    )

    year = random.choice([
        2025,
        2026
    ])

    month = random.randint(1, 12)

    allocated = random.uniform(
        500_000,
        10_000_000
    )

    actual = allocated * random.uniform(
        0.50,
        1.20
    )

    remaining = allocated - actual

    utilization = (
        actual / allocated
    ) * 100

    budgets.append({
        "budget_id": i,
        "department_id": department_id,
        "budget_year": year,
        "budget_month": month,
        "allocated_amount": round(
            allocated,
            2
        ),
        "actual_amount": round(
            actual,
            2
        ),
        "remaining_amount": round(
            remaining,
            2
        ),
        "utilization_percentage": round(
            utilization,
            2
        )
    })

budgets_df = pd.DataFrame(budgets)


# ============================================================
# SAVE DATA
# ============================================================

print("\nSaving datasets...")


datasets = {
    "departments.csv": departments_df,
    "employees.csv": employees_df,
    "salaries.csv": salaries_df,
    "vendors.csv": vendors_df,
    "transactions.csv": transactions_df,
    "bills.csv": bills_df,
    "invoices.csv": invoices_df,
    "budgets.csv": budgets_df
}


for filename, dataframe in datasets.items():

    path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    dataframe.to_csv(
        path,
        index=False
    )

    print(
        f"✓ {filename}: "
        f"{len(dataframe):,} records"
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n============================================")
print("DATA GENERATION COMPLETED")
print("============================================")

for filename, dataframe in datasets.items():

    print(
        f"{filename:<25} "
        f"{len(dataframe):>10,}"
    )

print("============================================")

print(
    "\nAnomalous transactions:",
    transactions_df["is_anomaly"].sum()
)

print(
    "Duplicate invoices:",
    invoices_df["is_duplicate"].sum()
)