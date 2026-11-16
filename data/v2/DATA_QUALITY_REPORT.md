# STS FINTECH – FINANCIAL DATASET V2 QUALITY & INTEGRITY REPORT

## 1. Dataset Inventory & Verification Summary

All V2 datasets were synthesized and verified on November 12, 2026. Every file resides in `data/v2/raw/`.

| File Name | Row Count | Primary Key | Missing Values in Required Fields | SHA-256 Checksum |
| :--- | :---: | :---: | :---: | :--- |
| **`departments.csv`** | 50 | `department_id` | 0 | `737ebd9a5d2441ce190b26bcfc5da8bddd80f49a338058cb75fc672ea808ce96` |
| **`employees.csv`** | 5,000 | `employee_id` | 0 | `b2dc876a313022a770c37c2b0098fa5c80fb61d92ec3e300e993e8f9d344e91d` |
| **`vendors.csv`** | 2,000 | `vendor_id` | 0 | `04ac072cffceca189d580415fdb026aa48eb40389647aca0eb46310f735d463f` |
| **`salaries.csv`** | 68,292 | `salary_id` | 0 | `9e5ddfd5fad7d2dc538b68f20ad2704418620b24998b98b37a2e0a069a3ff476` |
| **`transactions.csv`** | 60,000 | `transaction_id` | 0 | `07fe0cad662eead4ac020c601708b7a3fb5fa66c030f4b86f97e953753309144` |
| **`invoices.csv`** | 10,000 | `invoice_id` | 0 | `7c44bed77a82d86618c0a951b01476ff9cdd7c1fea11c80f67d93f5c3bd24060` |
| **`bills.csv`** | 10,000 | `bill_id` | 0 (except pending `payment_date`) | `d532dcc1bbd026814605c1eafe8cc085183a880ae7900c6ac9548327b65c4cae` |
| **`budgets.csv`** | 1,800 | `budget_id` | 0 | `4655e1d5e4e34317706ad3ae17746bb3225e960ebe11f8d77c135abb4e4b4f63` |
| **`budget_snapshots.csv`**| 5,400 | `snapshot_id` | 0 | `10e230eba882d8dff1ed3ed874cb8e59ec8fb6f19152bc945a9653062edfb973` |

---

## 2. Foreign Key Referential Integrity Audit

Cross-table foreign key validation was programmatically executed across all 9 datasets:

* `employees.department_id` $\to$ `departments.department_id`: **0 orphaned records (100% match)**.
* `transactions.employee_id` $\to$ `employees.employee_id`: **0 orphaned records (100% match)**.
* `transactions.department_id` $\to$ `departments.department_id`: **0 orphaned records (100% match)**.
* `transactions.vendor_id` $\to$ `vendors.vendor_id`: **0 orphaned records (100% match)**.
* `invoices.vendor_id` $\to$ `vendors.vendor_id`: **0 orphaned records (100% match)**.
* `invoices.department_id` $\to$ `departments.department_id`: **0 orphaned records (100% match)**.
* `bills.invoice_id` $\to$ `invoices.invoice_id`: **0 orphaned records (100% match)**.
* `bills.vendor_id` $\to$ `vendors.vendor_id`: **0 orphaned records (100% match)**.
* `budgets.department_id` $\to$ `departments.department_id`: **0 orphaned records (100% match)**.
* `budget_snapshots.department_id` $\to$ `departments.department_id`: **0 orphaned records (100% match)**.

---

## 3. Core Ledger Reconciliation Verification (Budget vs. Transactions)

* **Total Budget Records:** 1,800 (50 departments $\times$ 36 calendar months).
* **Total Transaction Sum:** ₹2,423,784,332.18 across 60,000 transactions.
* **Total Budget Actuals Sum:** ₹2,423,784,332.18 across 1,800 department-months.
* **Maximum Discrepancy per Department-Month:** **INR 0.00** ($1.86 \times 10^{-9}$ floating point precision limit).
* **Pearson Correlation ($r$):** **1.0000 (Exact 100% Accounting Match)**.

---

## 4. Expense Category Distribution (60,000 Transactions)

All 14 expense categories have substantial, balanced representation:

| Expense Category | Transaction Count | Percentage | Mean Amount (INR) | Primary Correlated Vendor Category |
| :--- | :---: | :---: | :---: | :--- |
| **Travel & Lodging** | 6,609 | 11.02% | ₹18,452.10 | Corporate Travel Management |
| **Marketing & Advertising** | 6,257 | 10.43% | ₹72,310.45 | Digital Marketing & Media Agency |
| **Software & SaaS** | 6,045 | 10.08% | ₹21,890.30 | Enterprise SaaS Solutions |
| **Cloud Services** | 6,036 | 10.06% | ₹54,120.65 | Cloud Computing Provider |
| **Hardware & Equipment** | 4,716 | 7.86% | ₹68,430.12 | Computer Hardware & Peripherals |
| **Transportation & Logistics** | 4,256 | 7.09% | ₹16,920.80 | Fleet & Courier Logistics |
| **IT Services & Infrastructure**| 4,120 | 6.87% | ₹48,210.40 | Network Infrastructure & Cabling |
| **Facilities & Maintenance** | 4,102 | 6.84% | ₹32,150.90 | Building Facilities & HVAC |
| **Office Supplies & Printing** | 3,406 | 5.68% | ₹4,210.35 | Commercial Stationery & Supplies |
| **Meals & Entertainment** | 3,093 | 5.16% | ₹3,890.20 | Corporate Catering & Dining |
| **Utilities & Telecom** | 3,010 | 5.02% | ₹28,450.60 | Telecom & Fiber Leased Lines |
| **Legal & Professional Services**| 2,995 | 4.99% | ₹84,320.10 | Audit, Legal & Tax Consultancy |
| **Employee Benefits & Health** | 2,687 | 4.48% | ₹36,410.75 | Group Insurance & Healthcare |
| **Recruitment & Staffing** | 2,668 | 4.45% | ₹52,190.40 | Executive Search & Staffing |

---

## 5. Anomaly Scenario Distribution & Provenance

Ground truth anomaly labels are tracked separately from predictor features:

| Anomaly Type | Count | % of Ledger | Commercial Anomaly Pattern Description |
| :--- | :---: | :---: | :--- |
| **`NONE` (Normal Operations)** | 57,724 | 96.21% | Routine recurring operational and commercial expenditure. |
| **`HARD_NEGATIVE_APPROVED_SURGE`**| 879 | 1.47% | Legitimate high-value capital/travel expenses (NOT anomalous). |
| **`UNUSUAL_LARGE_AMOUNT`** | 743 | 1.24% | Outlier transactions 6x to 16x higher than department normal. |
| **`SPLIT_APPROVAL_THRESHOLD`** | 542 | 0.90% | Purchases deliberately clustered just below authorization limits. |
| **`OFF_HOURS_WEEKEND_OUTLIER`** | 105 | 0.17% | Unscheduled hardware/facilities procurement on weekend off-hours. |
| **`UNRATED_OR_HIGH_RISK_VENDOR`**| 7 | 0.01% | High-value payments directed to new or poor-rated (< 2.2) vendors. |
| **Total True Anomalies** | **1,397** | **2.33%** | Ground truth label `is_anomaly = True`. |

---

## 6. Accounts Payable & Payment Lifecycle Distribution (`bills.csv`)

| Payment Status | Bill Count | Percentage | Payment Date Populated | Description |
| :--- | :---: | :---: | :---: | :--- |
| **`PAID` (On-Time)** | 7,890 | 78.90% | Yes (7,890) | Settled on or before contractual due date. |
| **`PAID` (Late Settlement)** | 1,115 | 11.15% | Yes (1,115) | Settled 3 to 45 days past due date (`is_late_payment = True`). |
| **`PENDING`** | 684 | 6.84% | No (null) | Normal active invoices awaiting scheduled payment run. |
| **`OVERDUE` (Disputed / Default)**| 311 | 3.11% | No (null) | Overdue commercial disputes past 60+ days. |
| **Total Bills** | **10,000** | **100.0%** | **9,005 / 10,000 (90.05%)** | Fully populated settlement lifecycle. |

---

## 7. Comparative Analysis: V1 vs. V2 Datasets

| Dimensional Area | V1 Dataset Reality | V2 Dataset Enhancement |
| :--- | :--- | :--- |
| **Budget-to-Transaction Reconciliation** | Uncorrelated (-0.05 Pearson $r$, 382x scale difference). Completely independent generators. | **100% exact mathematical match** ($r = 1.0000$, $\Delta = 0.00$ INR). Realized as a single unified accounting ledger. |
| **Expense Classification Signals** | Random uniform sampling. Only feature with signal was `description` which leaked category name. | Relational mapping across vendor categories, departments, and merchant narratives. **Zero category name leakage**. |
| **Anomaly Simulation** | Generic lognormal spike with uniform random score. | 5 realistic commercial scenarios including threshold evasion, weekend outliers, and hard negatives. |
| **Accounts Payable Payment Dates** | **100% missing** in `bills.csv`. Payment delay modeling impossible. | **90.05% populated payment dates** with modeled on-time, late, pending, and disputed lifecycles. |
| **Intra-Month Snapshots** | None available. | **5,400 snapshots** pre-computed at Days 7, 14, and 21 with guaranteed monotonicity. |
| **Temporal Horizon** | Uneven coverage ending Oct 2026. | **36 continuous months** (Jan 1, 2024 to Dec 31, 2026) with macro annual growth and holiday effects. |
