# STS FINTECH – FINANCIAL DATA GENERATION METHODOLOGY V2

## Executive Summary
Version 2 (V2) of the STS Fintech Synthetic Enterprise Financial Data Generator was engineered to resolve the statistical independence flaws identified in V1. 

V2 implements true relational coherence across corporate departments, employee spending authorizations, vendor supply categories, core ledger transactions, monthly budget allocations, and accounts payable settlement lifecycles.

---

## 1. Domain Modeling & Relational Coherence

### 1.1 Organizational Hierarchy
The enterprise consists of **50 distinct corporate departments** across 6 high-level divisions:
1. `Technology`: Core Engineering, Cloud Platform, Data Engineering, SRE, Cybersecurity, Mobile, Frontend, Backend, AI Research.
2. `Finance & Admin`: Corporate Finance, Accounting, Treasury, Tax, Internal Audit, FP&A.
3. `Human Resources`: Talent Acquisition, Employee Experience, L&D, Total Rewards, HR Operations.
4. `Commercial`: Enterprise Sales, Regional Commercial Sales (South, West, North), Business Development, Account Management, Solutions Engineering.
5. `Marketing`: Brand, Digital & Growth, Product Marketing, Creative Studio, Events & Sponsorships.
6. `Workplace Ops & Legal`: Facilities, Corporate Travel, Legal, Compliance, Risk Management, Customer Success.

Each department has an assigned annual baseline budget, operating city, and designated cost center manager.

### 1.2 Spending Authorization Hierarchy
Each of the **5,000 employees** is assigned a tiered spending authorization limit based on seniority:
* Associate: ₹25,000
* Senior Associate: ₹50,000
* Specialist / Team Lead: ₹100,000
* Manager: ₹250,000
* Senior Manager / Director: ₹1,000,000

This authorization threshold provides the foundational boundary for the *Split Approval Threshold* anomaly simulation.

---

## 2. Category Modeling Without Data Leakage (Use Case A)

### 2.1 The Problem in V1
In V1, transaction categories were sampled uniformly at random (`random.choice(categories)`), resulting in zero correlation with vendors or departments. Furthermore, V1 generated descriptions via `f"{category} expense"`, creating trivial string matching leakage. When that description was removed, Model A collapsed to 7.40% accuracy.

### 2.2 The V2 Relational Mapping
In V2, transactions are generated with realistic commercial probabilities:
1. **Departmental Purchasing Mandate:** Technology departments spend 85%+ on Cloud, SaaS, Hardware, and Infrastructure. Commercial departments spend 75%+ on Travel, Lodging, Marketing, and Entertainment. Facilities departments spend heavily on Maintenance, Utilities, and Office Supplies.
2. **Vendor Specialization:** Each of the 2,000 vendors is registered in a specific commercial category (e.g., `Cloud Computing Provider`, `Corporate Travel Management`). A transaction with a Cloud Computing Provider has an 90% probability of being categorized as `Cloud Services`, with a 10% probability of representing legitimate cross-category purchases (e.g., cloud advisory training, migration consulting).
3. **Realistic Narratives without Target Name:** Descriptions are drawn from specialized merchant narrative templates (e.g., `"Monthly EC2 on-demand compute and S3 storage consumption"` or `"Dedicated cloud database instance multi-AZ hosting fee"` for Cloud Services; `"Round-trip air ticket domestic client technical workshop"` for Travel). The literal category string (`Cloud Services`, `Travel & Lodging`) is strictly omitted.

---

## 3. Anomaly Simulation & Ground-Truth Provenance (Use Case B)

Anomalies represent **2.33%** of all transactions and are generated across 5 distinct commercial patterns:

1. **`UNUSUAL_LARGE_AMOUNT` (743 transactions):** Transactions exceeding 6x to 16x the typical department/role baseline.
2. **`SPLIT_APPROVAL_THRESHOLD` (542 transactions):** Purchases intentionally positioned just below employee authorization thresholds (e.g. ₹48,800 for a ₹50,000 limit) to evade secondary management approval.
3. **`OFF_HOURS_WEEKEND_OUTLIER` (105 transactions):** Capital expenditure or facility overhaul charges occurring on weekend midnight timestamps without prior scheduling.
4. **`UNRATED_OR_HIGH_RISK_VENDOR` (7 transactions):** High-value disbursements (> ₹100,000) directed to recently created, unvetted, low-rated suppliers (< 2.2 rating).
5. **`HARD_NEGATIVE_APPROVED_SURGE` (879 transactions):** Legitimate, high-value corporate events (e.g., annual server cluster renewal, approved executive summit lodging) that exhibit high amounts but are **flagged as normal (`is_anomaly = False`)**. This forces anomaly models to learn contextual multidimensional patterns rather than simplistic amount thresholding.

Ground truth is recorded separately in `is_anomaly` and `anomaly_type`. Neither column is ever exposed as an input feature during model training.

---

## 4. Time-Series Coherence & Seasonality (Use Case C)

The transaction timeline spans **36 continuous calendar months** (`2024-01-01` to `2026-12-31`, 1,096 days):
* **Weekly Seasonality:** Weekday transaction volumes (Mon–Fri) follow normal corporate operations (50–75 transactions/day), while weekend volumes drop to 8–20 transactions/day.
* **Month-End Surges:** Commercial accounts payable settlement peaks between the 25th and 30th of each month (1.30x base volume).
* **Macro Annual Growth:** A 8.0% year-over-year corporate expansion trend is applied.
* **Calendar Holiday Adjustments:** Indian national holidays (Republic Day, Independence Day, Gandhi Jayanti, Diwali week) exhibit reduced operational B2B activity.

---

## 5. 100% Exact Budget-to-Ledger Reconciliation (Use Case D)

### 5.1 Reconciliation Principle
Unlike V1 (which generated budgets independently with a -0.05 correlation and 382x scale error), V2 enforces exact mathematical ledger reconciliation:

$$\text{actual\_amount}_{d, y, m} = \sum_{t \in \text{transactions}_{d, y, m}} \text{amount}_t$$

Programmatic verification confirms the maximum absolute discrepancy between `budgets.csv` and `transactions.csv` across all 1,800 department-months is **$\approx 0.00$ INR ($< 10^{-8}$)**.

### 5.2 Realistic Budget Allocation Profiles
Allocated budgets are modeled based on corporate planning horizons:
* **60% Within Budget:** Planned with adequate operational buffer (utilization 70% to 90%).
* **25% Approaching Limit:** Tight operational allocation (utilization 88% to 99%).
* **15% Legitimate Overruns:** Unanticipated projects, emergency repairs, or expansion spikes (utilization 104% to 135%).

### 5.3 Intra-Month Snapshots
To support intra-month forecasting, `budget_snapshots.csv` pre-aggregates ledger actuals strictly up to **Day 7, Day 14, and Day 21** of each month. It is mathematically verified that:
$$\text{MTD}_{\text{Day 7}} \le \text{MTD}_{\text{Day 14}} \le \text{MTD}_{\text{Day 21}} \le \text{Final Full-Month Actual}$$

---

## 6. Accounts Payable Settlement Lifecycle (Use Case E)

In V1, `bills.csv` had 100% null `payment_date` values. V2 implements an end-to-end commercial settlement lifecycle across 10,000 invoices and 10,000 bills:
* **Invoice Duplication:** 2.5% of invoices represent duplicate submissions (flagged with `is_duplicate = True` and rejected under `rejection_reason = "DUPLICATE_SUBMISSION"`).
* **Payment Settlement Distribution:**
  * **On-Time Settlement (72.0%):** Disbursed within contractual payment terms (`payment_date <= due_date`).
  * **Late Payments (18.0%):** Disbursed 3 to 45 days past due date (`is_late_payment = True`).
  * **Pending Settlements (6.8%):** Normal active invoices awaiting payment run (`payment_date = null, payment_status = "PENDING"`).
  * **Overdue / Disputed (3.2%):** Commercial disputes or payment holds past 60+ days (`payment_date = null, payment_status = "OVERDUE"`).
