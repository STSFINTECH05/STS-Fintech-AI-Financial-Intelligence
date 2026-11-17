# STS FINTECH - PERSONAL FINANCE CLASSIFIER V1 TRAINING REPORT
## Step 7.4C-3C Model Training, Feature Ablation, Calibration and Validation Audit

- **Date**: 2026-11-17 04:49:24 UTC
- **Artifact Bundle**: `/Users/stsfintech/STS-Fintech-AI-Financial-Intelligence/ml/models/personal/experimental/personal_finance_bundle_v1_candidate.joblib`
- **Artifact SHA-256**: `81b7f350f46de0bd98941451ae628c5afbc3702b85ceff85c9f4ed41877cd9a0`
- **Dataset SHA-256**: `11cf2d14da863707a7b69406582ef51d18d5d46c73e153a7b6c0f9ea5105e243`
- **Governance Status**: `EXPERIMENTAL_SYNTHETIC_DATA_ONLY - NOT PRODUCTION APPROVED`

---

### 1. Dataset Filtering & Reconciliation
- Total Raw Synthetic Rows: **29,996**
- Excluded General-Income Rows: **1,424** (Retail mismatch)
- Supervised Filtered Rows: **27,072**
  - Train: **18,951** (Expected: 18,951) - MATCH
  - Val: **2,706** (Expected: 2,706) - MATCH
  - Test: **5,415** (Expected: 5,415) - MATCH
- Challenge Test Records: **300**

---

### 2. Feature Ablation Results (Validation Split)
#### Income Classifier
- ABL-01 (Text Only): F1 = 1.0000
- ABL-02 (Text + log_amount): F1 = 1.0000
- ABL-03 (Text + Raw amount): F1 = 1.0000
- ABL-04 (Text + Raw + Log): F1 = 1.0000
- ABL-05 (Text + Log + Calendar): F1 = 1.0000

#### Expense Classifier
- ABL-01 (Text Only): F1 = 1.0000
- ABL-02 (Text + log_amount): F1 = 1.0000
- ABL-03 (Text + Raw amount): F1 = 1.0000
- ABL-04 (Text + Raw + Log): F1 = 1.0000
- ABL-05 (Text + Log + Calendar): F1 = 1.0000

*Architectural Selection*: Text-only pipeline selected. Performing within 1.5% F1 while completely eliminating low-dollar numeric distortion.

---

### 3. Model Performance on Untouched Test Split
| Task | Target Classes | Top-1 Accuracy (Gate >= 85%) | Macro-F1 (Gate >= 0.82) | P99 Latency (Gate < 25ms) | Status |
|---|---|---|---|---|---|
| **Income Classifier** | Consulting, Retainer, Investment & Yield | **100.00%** | **1.0000** | **0.00ms** | **PASSED** |
| **Expense Classifier** | 7 Specific + General | **100.00%** | **1.0000** | **0.00ms** | **PASSED** |

---

### 4. Challenge Suite & Abstention Results (300 Test Challenges)
- Unsupported Salary Abstention: 0.00%
- Unsupported Loan Abstention: 0.00%
- Unsupported Refund Abstention: 0.00%
- Ambiguous Multi-Category Abstention: 0.00%
- Insufficient Information Abstention: 92.00%
- OOD Amount Abstention: 0.00%
