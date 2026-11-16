# STS FINTECH – FINANCIAL DATASET V2 DATA DICTIONARY

## Overview
This document specifies the schema, data types, primary keys, foreign keys, and operational constraints for the Version 2 (V2) synthetic enterprise financial datasets located in `data/v2/raw/`.

---

## 1. `departments.csv`
Contains corporate department hierarchies, locations, and annual budget baselines.

| Column Name | Data Type | Key Type | Nullable | Description / Business Meaning |
| :--- | :--- | :---: | :---: | :--- |
| `department_id` | Integer | PK | No | Unique department identifier (1 to 50). |
| `department_code` | String | Unique | No | Alphanumeric department code (e.g., `DEPT001`). |
| `department_name` | String | — | No | Functional department name (e.g., `Core Engineering`). |
| `division` | String | — | No | High-level division (`Technology`, `Commercial`, `Finance & Admin`, etc.). |
| `manager_name` | String | — | No | Name of the department head / budget owner. |
| `location` | String | — | No | Operating regional office city. |
| `annual_budget_baseline` | Float | — | No | Baseline annual planned expenditure in INR. |

---

## 2. `employees.csv`
Contains employee directory, organizational mapping, and spending authorization limits.

| Column Name | Data Type | Key Type | Nullable | Description / Business Meaning |
| :--- | :--- | :---: | :---: | :--- |
| `employee_id` | String | PK | No | Unique employee identifier (e.g., `EMP000001` to `EMP005000`). |
| `employee_name` | String | — | No | Full legal name of the employee. |
| `department_id` | Integer | FK (`departments.department_id`) | No | Department assignment. |
| `designation` | String | — | No | Organizational role title. |
| `employment_type` | String | — | No | Contract type (`FULL_TIME`, `CONTRACT`, `INTERN`). |
| `monthly_base_salary`| Float | — | No | Monthly base compensation in INR. |
| `spending_approval_limit` | Float | — | No | Internal authorization limit for expense approval in INR. |
| `joining_date` | Date (`YYYY-MM-DD`) | — | No | Date of employment commencement. |
| `status` | String | — | No | Employment status (`ACTIVE`, `INACTIVE`). |

---

## 3. `vendors.csv`
Contains the commercial vendor and supplier directory.

| Column Name | Data Type | Key Type | Nullable | Description / Business Meaning |
| :--- | :--- | :---: | :---: | :--- |
| `vendor_id` | String | PK | No | Unique vendor identifier (e.g., `VEN00001` to `VEN02000`). |
| `vendor_name` | String | — | No | Registered corporate vendor entity name. |
| `vendor_category` | String | — | No | Commercial supplier category (e.g., `Cloud Computing Provider`). |
| `location` | String | — | No | Vendor corporate headquarters location. |
| `contact_email` | String | — | No | Operational accounts payable contact email. |
| `payment_terms_days`| Integer | — | No | Standard agreed net payment terms in days (`15`, `30`, `45`, `60`, `90`). |
| `vendor_rating` | Float | — | No | Historical vendor quality rating (1.5 to 5.0). |
| `is_preferred_vendor`| Boolean | — | No | Flag indicating approved preferred supplier status. |
| `status` | String | — | No | Vendor operational status (`ACTIVE`, `UNDER_REVIEW`). |

---

## 4. `transactions.csv`
The core enterprise financial ledger. 100% reconciled with monthly departmental budgets.

| Column Name | Data Type | Key Type | Nullable | Description / Business Meaning |
| :--- | :--- | :---: | :---: | :--- |
| `transaction_id` | Integer | PK | No | Unique ledger transaction identifier (1 to 60000). |
| `employee_id` | String | FK (`employees.employee_id`) | No | Employee who initiated or incurred the expense. |
| `department_id` | Integer | FK (`departments.department_id`) | No | Department bearing the financial cost. |
| `vendor_id` | String | FK (`vendors.vendor_id`) | No | Counterparty vendor receiving payment. |
| `transaction_date` | Date (`YYYY-MM-DD`) | — | No | Transaction date (`2024-01-01` to `2026-12-31`). |
| `transaction_type` | String | — | No | Ledger transaction type (`EXPENSE`, `PAYMENT`, `INVOICE_SETTLEMENT`). |
| `category` | String | — | No | **Prediction Target for Model A:** 14 financial expense categories. |
| `description` | String | — | No | Realistic transaction narrative **(zero category name leakage)**. |
| `amount` | Float | — | No | Transaction amount in INR (> 0). |
| `currency` | String | — | No | Transaction currency (100% `INR`). |
| `payment_method` | String | — | No | Payment method (`BANK_TRANSFER`, `CORPORATE_CREDIT_CARD`, `UPI`, etc.). |
| `is_business_expense`| Boolean | — | No | Corporate expense eligibility flag. |
| `is_anomaly` | Boolean | — | No | **Evaluation Label for Model B:** Ground-truth anomaly flag. |
| `anomaly_type` | String | — | No | Ground-truth anomaly scenario provenance (e.g. `UNUSUAL_LARGE_AMOUNT`). |

---

## 5. `budgets.csv`
Departmental monthly budget allocations and actuals. **Mathematically guaranteed: actual_amount = sum of transactions.**

| Column Name | Data Type | Key Type | Nullable | Description / Business Meaning |
| :--- | :--- | :---: | :---: | :--- |
| `budget_id` | Integer | PK | No | Unique budget allocation record identifier (1 to 1800). |
| `department_id` | Integer | FK (`departments.department_id`) | No | Budgeted department. |
| `budget_year` | Integer | — | No | Budget calendar year (`2024`, `2025`, `2026`). |
| `budget_month` | Integer | — | No | Budget month (`1` to `12`). |
| `allocated_amount` | Float | — | No | Total authorized budget allocation in INR. |
| `actual_amount` | Float | — | No | Total actual expenditure (**exact sum of matching transactions**). |
| `remaining_amount` | Float | — | No | `allocated_amount - actual_amount`. |
| `utilization_percentage`| Float | — | No | **Target for Model D:** `(actual_amount / allocated_amount) * 100`. |
| `allocation_lines_count`| Integer| — | No | Number of planned allocation lines for this period. |
| `is_overspent` | Boolean | — | No | Target classification flag (`actual_amount > allocated_amount`). |

---

## 6. `budget_snapshots.csv`
Historical intra-month prediction snapshots for Day 7, Day 14, and Day 21 of each monthly period.

| Column Name | Data Type | Key Type | Nullable | Description / Business Meaning |
| :--- | :--- | :---: | :---: | :--- |
| `snapshot_id` | Integer | PK | No | Unique snapshot identifier (1 to 5400). |
| `department_id` | Integer | FK (`departments.department_id`) | No | Target department. |
| `budget_year` | Integer | — | No | Budget year. |
| `budget_month` | Integer | — | No | Budget month. |
| `as_of_day` | Integer | — | No | Snapshot day of month (`7`, `14`, `21`). |
| `elapsed_month_pct` | Float | — | No | Percentage of month elapsed as of snapshot. |
| `allocated_amount` | Float | — | No | Monthly allocated budget available at snapshot time. |
| `month_to_date_actual`| Float | — | No | Spend accumulated strictly up to `as_of_day`. |
| `transaction_count_mtd`| Integer| — | No | Number of transactions accumulated strictly up to `as_of_day`. |
| `final_actual_amount`| Float | — | No | **Evaluation Target:** Full-month final expenditure. |
| `final_utilization_pct`| Float | — | No | **Evaluation Target:** Full-month final utilization %. |
| `is_overspent_final`| Boolean | — | No | **Evaluation Target:** Full-month overspending outcome. |

---

## 7. `invoices.csv`
Supplier invoices submitted for approval and payment.

| Column Name | Data Type | Key Type | Nullable | Description / Business Meaning |
| :--- | :--- | :---: | :---: | :--- |
| `invoice_id` | String | PK | No | Unique invoice identifier (`INV00000001` to `INV00010000`). |
| `vendor_id` | String | FK (`vendors.vendor_id`) | No | Invoicing supplier. |
| `department_id` | Integer | FK (`departments.department_id`) | No | Recipient department. |
| `invoice_date` | Date (`YYYY-MM-DD`) | — | No | Date invoice was issued. |
| `due_date` | Date (`YYYY-MM-DD`) | — | No | Contractual payment due date. |
| `subtotal` | Float | — | No | Net invoice amount before tax in INR. |
| `tax_amount` | Float | — | No | 18% Goods and Services Tax (GST) in INR. |
| `total_amount` | Float | — | No | Gross payable invoice amount in INR. |
| `invoice_status` | String | — | No | Approval status (`APPROVED`, `PENDING`, `REJECTED`, `DISPUTED`). |
| `is_duplicate` | Boolean | — | No | Flag indicating duplicate invoice submission. |
| `rejection_reason` | String | — | Yes | Operational rejection code (e.g., `PRICE_MISMATCH`, `DUPLICATE_SUBMISSION`). |

---

## 8. `bills.csv`
Accounts payable records modeling the full payment settlement lifecycle.

| Column Name | Data Type | Key Type | Nullable | Description / Business Meaning |
| :--- | :--- | :---: | :---: | :--- |
| `bill_id` | String | PK | No | Unique bill identifier (`BILL0000001` to `BILL0010000`). |
| `invoice_id` | String | FK (`invoices.invoice_id`) | No | Corresponding invoice reference. |
| `vendor_id` | String | FK (`vendors.vendor_id`) | No | Vendor counterparty. |
| `department_id` | Integer | FK (`departments.department_id`) | No | Paying department. |
| `bill_date` | Date (`YYYY-MM-DD`) | — | No | Bill entry date. |
| `due_date` | Date (`YYYY-MM-DD`) | — | No | Bill payment due date. |
| `bill_amount` | Float | — | No | Net bill amount in INR. |
| `tax_amount` | Float | — | No | Tax amount in INR. |
| `total_amount` | Float | — | No | Gross payable amount in INR. |
| `payment_status` | String | — | No | Settlement status (`PAID`, `PENDING`, `OVERDUE`). |
| `payment_date` | Date (`YYYY-MM-DD`) | — | Yes | Actual date payment was disbursed (**90% populated, modeled delays**). |
| `is_late_payment` | Boolean | — | No | Flag indicating payment was made after `due_date` or is overdue. |
| `description` | String | — | No | Accounts payable settlement description. |

---

## 9. `salaries.csv`
Monthly recurring payroll disbursements across departments and employees.

| Column Name | Data Type | Key Type | Nullable | Description / Business Meaning |
| :--- | :--- | :---: | :---: | :--- |
| `salary_id` | Integer | PK | No | Unique payroll record identifier. |
| `employee_id` | String | FK (`employees.employee_id`) | No | Employee recipient. |
| `department_id` | Integer | FK (`departments.department_id`) | No | Department bearing salary cost. |
| `salary_month` | Date (`YYYY-MM-DD`) | — | No | Pay period month start date. |
| `basic_salary` | Float | — | No | Base salary component in INR. |
| `hra` | Float | — | No | House Rent Allowance (40% of basic) in INR. |
| `allowances` | Float | — | No | Special and medical allowances in INR. |
| `bonus` | Float | — | No | Performance / festival bonus in INR. |
| `deductions` | Float | — | No | Statutory PF and tax deductions in INR. |
| `gross_salary` | Float | — | No | Gross compensation before deductions in INR. |
| `net_salary` | Float | — | No | Final disbursed compensation in INR. |
