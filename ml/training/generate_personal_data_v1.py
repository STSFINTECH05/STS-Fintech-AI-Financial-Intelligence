"""
STS FINTECH - MILESTONE 2 - STEP 7.4C-3B
Synthetic Dataset Generation for personal_finance_classifier_v1

Generates 30,000 synthetic INR transaction records conforming strictly to the
Step 7.4C-3A1 architecture and data governance contract.

Key Features & Guarantees:
- Pure synthetic data generated deterministically with fixed random seeds.
- Target application categories: 11 categories (3 supervised income, 7 supervised expense, 1 bidirectional General).
- Dedicated challenge set (unsupported income, ambiguous transactions, empty text, OOD amounts, invalid dates).
- Disjoint merchant-group and template-family partitioning across train (70%), val (10%), and test (20%).
- Full provenance tracking on every record (synthetic marker, version, split, group IDs).
- Strict project preservation: Zero dependencies on real customer records, zero DB interaction.
"""

import os
import sys
import hashlib
import random
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np

# Random seed for reproducibility
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

DATA_GEN_VERSION = "1.0.0"

# Target output directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
OUTPUT_DIR = os.path.join(BASE_DIR, "ml", "data", "personal")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "personal_transactions_synthetic_v1.csv")


# -----------------------------------------------------------------------------
# 1. APPLICATION CATEGORIES & CHALLENGE LABELS
# -----------------------------------------------------------------------------
SUPERVISED_INCOME_CATEGORIES = [
    "Consulting",
    "Retainer",
    "Investment & Yield",
    "General"
]

SUPERVISED_EXPENSE_CATEGORIES = [
    "Infrastructure & Cloud",
    "Software Subscription",
    "Office Rent",
    "Payroll & Salaries",
    "Utilities & Bills",
    "Operations & Travel",
    "Legal & Compliance",
    "General"
]

CHALLENGE_LABELS = [
    "UNSUPPORTED_INCOME_SALARY",
    "UNSUPPORTED_INCOME_LOAN",
    "UNSUPPORTED_INCOME_REFUND",
    "AMBIGUOUS_MULTI_CATEGORY",
    "INSUFFICIENT_INFORMATION",
    "OUT_OF_DISTRIBUTION_AMOUNT"
]


# -----------------------------------------------------------------------------
# 2. MERCHANT DIRECTORY GENERATION (1,200 Synthetic Entities in Families)
# -----------------------------------------------------------------------------
def build_merchant_universe() -> Dict[str, List[Dict[str, Any]]]:
    """
    Constructs a catalog of 1,200 merchant entities organized by category and merchant family.
    Each merchant has a family_id, entity_name, and spelling/UPI aliases.
    """
    categories = [
        "Infrastructure & Cloud", "Software Subscription", "Office Rent",
        "Payroll & Salaries", "Utilities & Bills", "Operations & Travel",
        "Legal & Compliance", "General", "Consulting", "Retainer", "Investment & Yield"
    ]

    # Specific realistic base names per category (10+ base roots per category)
    base_names = {
        "Infrastructure & Cloud": [
            "AWS Cloud", "Google Cloud", "DigitalOcean", "Linode Cloud", "Azure Infra",
            "Hetzner Online", "Vultr Compute", "Hostinger VPS", "Netlify Web", "Vercel Hosting",
            "Cloudflare CDN", "OVHcloud", "Rackspace Cloud", "Fastly CDN", "ScaleWay"
        ],
        "Software Subscription": [
            "GitHub Pro", "JetBrains All", "Slack Workspace", "Figma Design", "Notion Team",
            "Atlassian Jira", "Zoom Video", "Docker Hub", "Canva Pro", "Miro Team",
            "Linear App", "Postman API", "Datadog SaaS", "New Relic", "Sentry Error"
        ],
        "Office Rent": [
            "WeWork India", "Awfis Space", "Innov8 Coworking", "IndiQube Hub", "Regus Business",
            "91springboard", "BHIVE Workspace", "CoWrks TechPark", "Smartworks Office", "The Executive Ctr",
            "Prestige Property Lease", "Brigade Commercial", "Embassy TechVillage Lease", "RMZ Corp Office", "Sobha Space"
        ],
        "Payroll & Salaries": [
            "Staff Remuneration", "Developer Payout", "Contract Designer Wage", "Intern Stipend",
            "QA Engineer Payout", "Data Scientist Salary", "DevOps Contractor", "Tech Lead Comp",
            "HR Associate Payout", "Executive Assistant Wage", "Sales Lead Commission", "Operations Associate Salary",
            "Frontend Engineer Remittance", "Backend Specialist Comp", "Product Manager Payout"
        ],
        "Utilities & Bills": [
            "BESCOM Power", "Tata Power", "Adani Electricity", "BWSSB Water", "Delhi Jal Board",
            "Airtel Broadband", "Jio Corporate Fiber", "ACT Fibernet", "Vodafone Idea Postpaid", "BSNL Fiber",
            "Mahanagar Gas", "Indraprastha Gas", "Torrent Power", "CESC Kolkata", "Hathway Cable"
        ],
        "Operations & Travel": [
            "Uber Business", "Ola Corporate", "IndiGo Airlines", "Air India Corp", "MakeMyTrip Biz",
            "Blue Dart Express", "Delhivery Logistics", "DTDC Courier", "Taj Hotels", "Marriott Executive",
            "Lemon Tree Premier", "IRCTC Corporate", "Shell Fleet Fuel", "IndianOil Petrol", "HPCL Fleet"
        ],
        "Legal & Compliance": [
            "Shardul Amarchand", "AZB Partners", "Khaitan & Co", "Trilegal Law", "Nishith Desai",
            "Deloitte Audit", "PwC Compliance", "EY Tax Advisory", "KPMG Secretarial", "BDO India Tax",
            "Singhania Partners", "Luthra and Luthra", "Vakilsearch ROC", "IndiaFilings GST", "ClearTax Enterprise"
        ],
        "General": [
            "Office Depot Stationeries", "Staples Supplies", "Apple Store Hardware", "Croma Electronics", "Reliance Digital",
            "Amazon Retail", "Flipkart Commercial", "D-Mart Provisions", "Nature Basket Pantry", "Starbucks Refreshment",
            "Swiggy Meal Delivery", "Zomato Food", "Blue Tokai Coffee", "Chai Point Snacks", "Decathlon Sports"
        ],
        "Consulting": [
            "Apex Strategic Advisory", "Vanguard Tech Advisory", "Fintech Innovations Consult", "NexGen Arch Advisory", "Cognitive Systems Consulting",
            "Starlight Business Partners", "Beacon Analytics Consulting", "Prism Digital Strategy", "Zenith Fin Advisory", "Alpha Vector Solutions",
            "Quantum Logic Consulting", "TrueNorth Enterprise Consult", "Summit Product Advisory", "Catalyst Tech Audit", "Horizon Cloud Consulting"
        ],
        "Retainer": [
            "Retainer Core Client A", "Retainer Client B", "Retainer Tech Partner C", "Retainer Advisory Client D", "Retainer Enterprise E",
            "Retainer Corp Alpha", "Retainer Services Beta", "Retainer Partner Gamma", "Retainer Client Delta", "Retainer Group Epsilon",
            "Retainer Zeta Tech", "Retainer Eta Media", "Retainer Theta Global", "Retainer Iota Fin", "Retainer Kappa Consult"
        ],
        "Investment & Yield": [
            "Zerodha Broking", "Groww Mutual Fund", "HDFC Sec Investment", "ICICI Direct Yield", "SBI Mutual Fund Payout",
            "Axis Liquid Fund", "Nippon India Yield", "UTI Dividend Fund", "Kotak Securities Yield", "Mirae Asset Payout",
            "Treasury Bill Yield", "Corporate Bond Coupon", "Fixed Deposit Interest HDFC", "FD Yield ICICI", "Liquid ETF Yield"
        ]
    }

    catalog = {}
    merchant_id_counter = 1
    family_id_counter = 1

    # Target roughly ~110 merchants per category = 1,200+ entities across ~400 families
    for cat in categories:
        catalog[cat] = []
        bases = base_names[cat]
        for base in bases:
            family_id = f"FAM_{family_id_counter:04d}"
            family_id_counter += 1

            # Generate 7-10 distinct entities/branches/aliases within this family
            num_entities = random.randint(7, 10)
            for i in range(num_entities):
                m_id = f"MERCH_{merchant_id_counter:05d}"
                merchant_id_counter += 1

                # Synthetic name variations within the family
                suffix = random.choice([
                    "Pvt Ltd", "Solutions", "Services", "Hub", "Desk", "Gateway",
                    "India", "Bangalore", "Mumbai", "Corp", "Commercial", "Online"
                ])
                full_name = f"{base} {suffix} {i+1}"
                upi_alias = f"{base.lower().replace(' ', '')}{i+1}@okaxis"
                pos_alias = f"POS {base.upper().replace(' ', '')} #{i+1}"
                bill_alias = f"BILLDESK*{base.upper().replace(' ', '')}"

                catalog[cat].append({
                    "merchant_id": m_id,
                    "family_id": family_id,
                    "base_name": base,
                    "full_name": full_name,
                    "upi_alias": upi_alias,
                    "pos_alias": pos_alias,
                    "bill_alias": bill_alias,
                    "category": cat
                })

    return catalog


# -----------------------------------------------------------------------------
# 3. TEMPLATE FAMILIES (Disjoint Text Synthesizers)
# -----------------------------------------------------------------------------
# We define 15 distinct Template Families (TF_01 .. TF_15)
# Allocation:
# - TF_01 to TF_10: Train (70%)
# - TF_11 to TF_12: Validation (10%)
# - TF_13 to TF_15: Test (20%)
TEMPLATE_FAMILIES = {
    # Train Families
    "TF_01": lambda m, d, a: f"UPI payment to {m['full_name']} for {d}",
    "TF_02": lambda m, d, a: f"{m['pos_alias']} payment debited for {d}",
    "TF_03": lambda m, d, a: f"{m['bill_alias']} settlement invoice {random.randint(1000,9999)} - {d}",
    "TF_04": lambda m, d, a: f"Monthly recurring billing {m['full_name']} / {d}",
    "TF_05": lambda m, d, a: f"NEFT transfer to {m['full_name']} ref {random.randint(100000,999999)} desc {d}",
    "TF_06": lambda m, d, a: f"Invoice payment clearance {m['base_name']} - {d}",
    "TF_07": lambda m, d, a: f"Direct bank debit for {d} via {m['upi_alias']}",
    "TF_08": lambda m, d, a: f"{m['full_name']} charge for {d} ref #{random.randint(100,999)}",
    "TF_09": lambda m, d, a: f"Corporate fee settlement: {d} at {m['full_name']}",
    "TF_10": lambda m, d, a: f"Scheduled transaction for {d} with {m['base_name']}",

    # Val Families
    "TF_11": lambda m, d, a: f"Auto-debit mandate processed by {m['full_name']} regarding {d}",
    "TF_12": lambda m, d, a: f"E-mandate clearing ref {random.randint(1000,9999)} to {m['base_name']} for {d}",

    # Test Families (Held-out syntactic styles)
    "TF_13": lambda m, d, a: f"Disbursement clearance voucher {m['full_name']} itemized: {d}",
    "TF_14": lambda m, d, a: f"Account statement debit line: {m['pos_alias']} :: {d}",
    "TF_15": lambda m, d, a: f"Commercial remittance notification: {m['base_name']} - purpose: {d}"
}

TRAIN_TEMPLATE_FAMILIES = ["TF_01", "TF_02", "TF_03", "TF_04", "TF_05", "TF_06", "TF_07", "TF_08", "TF_09", "TF_10"]
VAL_TEMPLATE_FAMILIES = ["TF_11", "TF_12"]
TEST_TEMPLATE_FAMILIES = ["TF_13", "TF_14", "TF_15"]


# Category-specific descriptive keywords
CATEGORY_DESCRIPTORS = {
    "Infrastructure & Cloud": [
        "EC2 compute node instance", "S3 object storage cluster", "Kubernetes cluster nodes",
        "database replica hosting", "dedicated bare metal server", "static CDN edge bandwidth",
        "DNS routing queries", "server backup snapshot storage", "GPU inference instance", "VPC gateway peering"
    ],
    "Software Subscription": [
        "team workspace license", "professional seat tier", "monthly developer seat",
        "enterprise security add-on", "pro subscription renewal", "annual software access",
        "API rate limit tier", "collaborative workspace user fee", "code repo premium license", "design seat renewal"
    ],
    "Office Rent": [
        "monthly dedicated desk space", "private executive office suite", "quarterly coworking fee",
        "tech park premises lease", "flexible open desk pass", "meeting room block credits",
        "office space maintenance advance", "commercial property monthly rent", "incubator lab cabin rent"
    ],
    "Payroll & Salaries": [
        "monthly compensation transfer", "contract developer payout", "lead engineer remuneration",
        "internship monthly stipend", "quarterly performance bonus", "specialist consultant wages",
        "executive salary remittance", "staff wage disbursement", "freelance frontend stipend"
    ],
    "Utilities & Bills": [
        "electricity tariff commercial meter", "optical fiber high-speed internet", "postpaid mobile corporate bill",
        "municipal water supply charge", "piped natural gas bill", "power consumption surcharge",
        "broadband line maintenance charge", "diesel generator standby power bill"
    ],
    "Operations & Travel": [
        "client meeting cab fare", "domestic flight ticket transit", "executive hotel overnight stay",
        "express parcel courier charges", "fuel petrol fleet refill", "airport taxi commute",
        "metro rail transit pass", "per diem meal travel allowance"
    ],
    "Legal & Compliance": [
        "advocate legal retainer consultation", "chartered accountant tax filing fee", "ROC statutory filing fee",
        "trademark registry attorney charges", "GST return filing preparation", "contract drafting counsel fee",
        "annual corporate audit compliance", "statutory advisory legal retainer"
    ],
    "General": [
        "office printer paper and toner", "dry erase markers stationery", "MacBook Pro hardware unit",
        "external 4K monitor display", "pantry coffee beans and tea", "supermarket team provisions",
        "client refreshment snacks", "cleaning supplies and sanitizers", "office desk accessories"
    ],
    "Consulting": [
        "architecture advisory milestone 1", "security audit final milestone fee", "freelance fintech consulting payout",
        "business transformation advisory", "system design workshop remuneration", "compliance advisory deliverable fee",
        "strategy consulting fee installment"
    ],
    "Retainer": [
        "monthly advisory retainer fee", "quarterly tech maintenance retainer", "fixed client retainer payout",
        "retainer advance agreement installment", "reserved capacity retainer receipt"
    ],
    "Investment & Yield": [
        "equity dividend quarterly payout", "liquid mutual fund redemption", "fixed deposit interest remittance",
        "treasury bond coupon credit", "portfolio capital distribution", "overnight yield return"
    ]
}


# -----------------------------------------------------------------------------
# 4. DISJOINT PARTITIONING ENGINE
# -----------------------------------------------------------------------------
def partition_merchants(catalog: Dict[str, List[Dict[str, Any]]]) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    """
    Partitions the merchant universe strictly by family_id so that:
    - 70% of merchant families are in Train
    - 10% of merchant families are in Val
    - 20% of merchant families are in Test
    Zero merchant overlap or family overlap across splits.
    """
    train_merchants = {cat: [] for cat in catalog}
    val_merchants = {cat: [] for cat in catalog}
    test_merchants = {cat: [] for cat in catalog}

    for cat, m_list in catalog.items():
        # Group by family_id
        families = {}
        for m in m_list:
            families.setdefault(m["family_id"], []).append(m)

        family_ids = sorted(list(families.keys()))
        random.seed(GLOBAL_SEED)
        random.shuffle(family_ids)

        n_fam = len(family_ids)
        n_train = int(0.70 * n_fam)
        n_val = int(0.10 * n_fam)

        train_fams = set(family_ids[:n_train])
        val_fams = set(family_ids[n_train:n_train + n_val])
        test_fams = set(family_ids[n_train + n_val:])

        for fam_id, items in families.items():
            if fam_id in train_fams:
                train_merchants[cat].extend(items)
            elif fam_id in val_fams:
                val_merchants[cat].extend(items)
            else:
                test_merchants[cat].extend(items)

    return train_merchants, val_merchants, test_merchants


# -----------------------------------------------------------------------------
# 5. DATASET GENERATION LOGIC
# -----------------------------------------------------------------------------
def sample_amount(category: str, is_ood: bool = False) -> float:
    """Samples realistic INR amount from log-normal distributions by category."""
    if is_ood:
        if random.random() < 0.5:
            # Sub-minimum (< ₹1.0)
            return round(random.uniform(0.01, 0.99), 2)
        else:
            # Extreme multi-crore amount (> ₹1,00,00,000)
            return round(random.uniform(10500000.0, 50000000.0), 2)

    # Realistic INR params: (mu, sigma, min_v, max_v)
    params = {
        "Infrastructure & Cloud": (8.4, 0.9, 150.0, 85000.0),      # med ~₹4,500
        "Software Subscription":  (7.1, 0.8, 199.0, 25000.0),      # med ~₹1,200
        "Office Rent":            (10.5, 0.7, 5000.0, 350000.0),    # med ~₹36,000
        "Payroll & Salaries":     (11.0, 0.6, 15000.0, 250000.0),   # med ~₹60,000
        "Utilities & Bills":      (7.5, 0.7, 200.0, 18000.0),       # med ~₹1,800
        "Operations & Travel":    (7.8, 0.9, 150.0, 45000.0),       # med ~₹2,500
        "Legal & Compliance":     (9.8, 0.8, 2500.0, 150000.0),     # med ~₹18,000
        "General":                (7.3, 1.1, 50.0, 75000.0),        # med ~₹1,500
        "Consulting":             (10.7, 0.6, 15000.0, 250000.0),   # med ~₹45,000
        "Retainer":               (11.2, 0.6, 25000.0, 500000.0),   # med ~₹73,000
        "Investment & Yield":     (9.0, 1.0, 500.0, 120000.0),      # med ~₹8,100
    }

    mu, sigma, min_v, max_v = params.get(category, (8.0, 1.0, 100.0, 50000.0))
    val = float(np.random.lognormal(mu, sigma))
    val = max(min_v, min(max_v, val))
    return round(val, 2)


def sample_date(challenge_type: str = None) -> str:
    """Generates an ISO-8601 date (YYYY-MM-DD) or explicit challenge date."""
    if challenge_type == "MISSING_DATE":
        return ""
    if challenge_type == "INVALID_DATE":
        return "2026-02-31"  # Non-existent calendar day
    if challenge_type == "HISTORICAL_DATE":
        return f"{random.randint(2021, 2023)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
    if challenge_type == "RECURRING_BILLING":
        # Exactly on the 1st or 15th
        day = 1 if random.random() < 0.5 else 15
        return f"2026-{random.randint(1,10):02d}-{day:02d}"

    # Standard distribution across 2024 to 2026
    year = random.choice([2024, 2025, 2026])
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    return f"{year}-{month:02d}-{day:02d}"


def generate_records_for_split(
    split_name: str,
    target_count: int,
    merchants_by_cat: Dict[str, List[Dict[str, Any]]],
    template_families: List[str],
    id_start: int
) -> List[Dict[str, Any]]:
    """Generates a balanced partition of records with zero cross-split leakage."""
    records = []
    curr_id = id_start

    # Target net counts incorporating challenge cases (5% challenge cases)
    num_challenges = int(0.05 * target_count)
    num_supervised = target_count - num_challenges

    expense_target = int(num_supervised * 0.80)
    income_target = num_supervised - expense_target

    exp_per_cat = expense_target // len(SUPERVISED_EXPENSE_CATEGORIES)
    inc_per_cat = income_target // len(SUPERVISED_INCOME_CATEGORIES)

    # 1. Generate Supervised Expense Records
    for cat in SUPERVISED_EXPENSE_CATEGORIES:
        m_pool = merchants_by_cat[cat]
        descriptors = CATEGORY_DESCRIPTORS[cat]

        for _ in range(exp_per_cat):
            m = random.choice(m_pool)
            d = random.choice(descriptors)
            tf = random.choice(template_families)

            # Injection of PII synthetic testing tokens in 3% of cases
            if random.random() < 0.03:
                pii_token = random.choice([
                    f"call +91 98{random.randint(10000000, 99999999)}",
                    f"UPI id dev.{random.randint(10,99)}@hdfcbank",
                    f"card ending {random.randint(1000,9999)}"
                ])
                d = f"{d} ({pii_token})"

            amt = sample_amount(cat)
            dt = sample_date()
            text = TEMPLATE_FAMILIES[tf](m, d, amt)

            # Assign application-compatible transaction type
            app_type = "fixed_expense" if cat in ("Office Rent", "Software Subscription", "Utilities & Bills") and random.random() < 0.4 else "additional_expense"
            if cat == "Utilities & Bills" and random.random() < 0.3:
                app_type = "bill"

            records.append({
                "synthetic_transaction_id": f"SYN_{curr_id:07d}",
                "transaction_direction": "expense",
                "app_transaction_type": app_type,
                "ground_truth_category": cat,
                "amount_inr": amt,
                "description": text,
                "transaction_date": dt,
                "merchant_id": m["merchant_id"],
                "merchant_family_id": m["family_id"],
                "template_family_id": tf,
                "partition_split": split_name,
                "is_challenge_case": 0,
                "challenge_reason": "NONE",
                "is_synthetic_marker": 1,
                "data_version": DATA_GEN_VERSION
            })
            curr_id += 1

    # 2. Generate Supervised Income Records
    for cat in SUPERVISED_INCOME_CATEGORIES:
        m_pool = merchants_by_cat[cat]
        descriptors = CATEGORY_DESCRIPTORS[cat]

        for _ in range(inc_per_cat):
            m = random.choice(m_pool)
            d = random.choice(descriptors)
            tf = random.choice(template_families)
            amt = sample_amount(cat)
            dt = sample_date()
            text = TEMPLATE_FAMILIES[tf](m, d, amt)

            app_type = "fixed_income" if cat in ("Retainer", "Investment & Yield") and random.random() < 0.5 else "additional_income"

            records.append({
                "synthetic_transaction_id": f"SYN_{curr_id:07d}",
                "transaction_direction": "income",
                "app_transaction_type": app_type,
                "ground_truth_category": cat,
                "amount_inr": amt,
                "description": text,
                "transaction_date": dt,
                "merchant_id": m["merchant_id"],
                "merchant_family_id": m["family_id"],
                "template_family_id": tf,
                "partition_split": split_name,
                "is_challenge_case": 0,
                "challenge_reason": "NONE",
                "is_synthetic_marker": 1,
                "data_version": DATA_GEN_VERSION
            })
            curr_id += 1

    # 3. Inject Challenge Cases (5% of split)
    # We replace a small tail with controlled challenge cases to verify safety
    num_challenges = int(0.05 * target_count)
    challenge_records = []
    for i in range(num_challenges):
        c_type = CHALLENGE_LABELS[i % len(CHALLENGE_LABELS)]
        tf = random.choice(template_families)

        split_tag = split_name.upper()
        if c_type == "UNSUPPORTED_INCOME_SALARY":
            m_dummy = {"full_name": f"Infosys Corporate Payroll {split_tag}", "base_name": "Infosys", "pos_alias": f"POS INFOSYS {split_tag}", "upi_alias": f"infosys_{split_name}@icici", "family_id": f"CHALLENGE_FAM_{split_tag}_01"}
            desc = f"Monthly employer salary credit for May 2026 ref {random.randint(10000,99999)} [{split_tag}]"
            amt = 85000.0
            direction = "income"
            gt = "UNSUPPORTED_INCOME_SALARY"
            app_type = "fixed_income"
            c_date = sample_date()

        elif c_type == "UNSUPPORTED_INCOME_LOAN":
            m_dummy = {"full_name": f"HDFC Loan Division {split_tag}", "base_name": "HDFC", "pos_alias": f"POS HDFC {split_tag}", "upi_alias": f"hdfcloan_{split_name}@hdfc", "family_id": f"CHALLENGE_FAM_{split_tag}_02"}
            desc = f"Personal loan net disbursement proceeds credit acc {random.randint(10000,99999)} [{split_tag}]"
            amt = 350000.0
            direction = "income"
            gt = "UNSUPPORTED_INCOME_LOAN"
            app_type = "additional_income"
            c_date = sample_date()

        elif c_type == "UNSUPPORTED_INCOME_REFUND":
            m_dummy = {"full_name": f"Amazon India Return {split_tag}", "base_name": "Amazon", "pos_alias": f"POS AMZN {split_tag}", "upi_alias": f"amazon_{split_name}@upi", "family_id": f"CHALLENGE_FAM_{split_tag}_03"}
            desc = f"Amazon refund for returned defective headset order #{random.randint(1000,9999)} [{split_tag}]"
            amt = 2499.0
            direction = "income"
            gt = "UNSUPPORTED_INCOME_REFUND"
            app_type = "additional_income"
            c_date = sample_date()

        elif c_type == "AMBIGUOUS_MULTI_CATEGORY":
            m_dummy = {"full_name": f"Airport Plaza Commercial {split_tag}", "base_name": "Airport Plaza", "pos_alias": f"POS AIRPORT {split_tag}", "upi_alias": f"airport_{split_name}@icici", "family_id": f"CHALLENGE_FAM_{split_tag}_04"}
            desc = f"Airport transit cab ride plus lunch buffet and stationery bundle [{split_tag} #{i}]"
            amt = 2800.0
            direction = "expense"
            gt = "AMBIGUOUS_MULTI_CATEGORY"
            app_type = "additional_expense"
            c_date = sample_date()

        elif c_type == "INSUFFICIENT_INFORMATION":
            m_dummy = {"full_name": f"UNKNOWN {split_tag}", "base_name": "UNKNOWN", "pos_alias": f"POS TXN {split_tag}", "upi_alias": f"upi_{split_name}@bank", "family_id": f"CHALLENGE_FAM_{split_tag}_05"}
            desc = random.choice([
                f"UPI/{random.randint(1000000000,9999999999)}/Payment-{split_tag}",
                f"NEFT-TRANSFER-{random.randint(10000,99999)}-{split_tag}",
                f"IMPS/{random.randint(100000,999999)}/CLEARANCE-{split_tag}",
                ""
            ])
            amt = 500.0
            direction = random.choice(["income", "expense"])
            gt = "INSUFFICIENT_INFORMATION"
            app_type = "additional_expense" if direction == "expense" else "additional_income"
            c_date = sample_date(challenge_type="MISSING_DATE" if desc == "" else None)

        elif c_type == "OUT_OF_DISTRIBUTION_AMOUNT":
            m_dummy = {"full_name": f"Enterprise Cloud Global {split_tag}", "base_name": "EntCloud", "pos_alias": f"POS ENTCLOUD {split_tag}", "upi_alias": f"ent_{split_name}@cloud", "family_id": f"CHALLENGE_FAM_{split_tag}_06"}
            desc = f"High capacity server infrastructure cluster lease batch {random.randint(100,999)} [{split_tag}]"
            amt = sample_amount("Infrastructure & Cloud", is_ood=True)
            direction = "expense"
            gt = "OUT_OF_DISTRIBUTION_AMOUNT"
            app_type = "fixed_expense"
            c_date = sample_date(challenge_type="INVALID_DATE" if random.random() < 0.2 else None)

        challenge_records.append({
            "synthetic_transaction_id": f"SYN_{curr_id:07d}",
            "transaction_direction": direction,
            "app_transaction_type": app_type,
            "ground_truth_category": gt,
            "amount_inr": amt,
            "description": desc,
            "transaction_date": c_date,
            "merchant_id": "MERCH_CHALLENGE",
            "merchant_family_id": m_dummy["family_id"],
            "template_family_id": tf,
            "partition_split": split_name,
            "is_challenge_case": 1,
            "challenge_reason": c_type,
            "is_synthetic_marker": 1,
            "data_version": DATA_GEN_VERSION
        })
        curr_id += 1

    # Combine supervised and challenge records
    combined = records + challenge_records
    # Shuffle partition
    random.shuffle(combined)
    return combined, curr_id


# -----------------------------------------------------------------------------
# 6. MAIN EXECUTION & VALIDATION AUDIT
# -----------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("STS FINTECH - STEP 7.4C-3B: SYNTHETIC DATASET GENERATION")
    print("=" * 80)

    # 1. Safety Assertion: Check if output already exists
    if os.path.exists(OUTPUT_CSV):
        raise RuntimeError(f"ABORTED: Output CSV already exists at {OUTPUT_CSV}. Will not overwrite.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Build Merchant Catalog (1,200 entities across 11 categories)
    print("\n[1/5] Building synthetic merchant directory...")
    catalog = build_merchant_universe()
    total_merchants = sum(len(m_list) for m_list in catalog.values())
    total_families = len(set(m["family_id"] for m_list in catalog.values() for m in m_list))
    print(f"  ✔ Generated {total_merchants:,} merchant entities across {total_families:,} merchant families.")

    # 3. Disjoint Partitioning (70% Train, 10% Val, 20% Test)
    print("\n[2/5] Partitioning merchant families into disjoint groups...")
    train_m, val_m, test_m = partition_merchants(catalog)

    # Audit merchant family overlap
    train_fams = set(m["family_id"] for m_list in train_m.values() for m in m_list)
    val_fams = set(m["family_id"] for m_list in val_m.values() for m in m_list)
    test_fams = set(m["family_id"] for m_list in test_m.values() for m in m_list)

    overlap_train_val = train_fams.intersection(val_fams)
    overlap_train_test = train_fams.intersection(test_fams)
    overlap_val_test = val_fams.intersection(test_fams)

    assert len(overlap_train_val) == 0, f"Merchant family leakage detected (Train-Val): {overlap_train_val}"
    assert len(overlap_train_test) == 0, f"Merchant family leakage detected (Train-Test): {overlap_train_test}"
    assert len(overlap_val_test) == 0, f"Merchant family leakage detected (Val-Test): {overlap_val_test}"
    print("  ✔ Zero merchant family overlap verified across Train, Val, and Test partitions.")

    # 4. Synthesize Records for Partitions
    print("\n[3/5] Generating synthetic transactions conforming to taxonomy...")
    # Exact target: 30,000 records total
    # Train: 21,000 (70%), Val: 3,000 (10%), Test: 6,000 (20%)
    train_records, next_id = generate_records_for_split("train", 21000, train_m, TRAIN_TEMPLATE_FAMILIES, 1)
    val_records, next_id = generate_records_for_split("val", 3000, val_m, VAL_TEMPLATE_FAMILIES, next_id)
    test_records, next_id = generate_records_for_split("test", 6000, test_m, TEST_TEMPLATE_FAMILIES, next_id)

    all_records = train_records + val_records + test_records
    df = pd.DataFrame(all_records)
    print(f"  ✔ Total generated records: {len(df):,}")

    # 5. Rigorous Anti-Leakage & Integrity Audits
    print("\n[4/5] Running anti-leakage and data quality assertions...")

    # Unique IDs
    assert df["synthetic_transaction_id"].nunique() == len(df), "Duplicate transaction IDs detected!"

    # Checksum & Null Checks
    assert df["amount_inr"].isna().sum() == 0, "Missing amount values detected!"
    assert (df["amount_inr"] > 0).all(), "Non-positive amounts detected!"
    assert (df["is_synthetic_marker"] == 1).all(), "Missing synthetic data marker!"

    # Template Family Overlap Check
    train_tfs = set(df[df["partition_split"] == "train"]["template_family_id"])
    val_tfs = set(df[df["partition_split"] == "val"]["template_family_id"])
    test_tfs = set(df[df["partition_split"] == "test"]["template_family_id"])

    assert len(train_tfs.intersection(val_tfs)) == 0, "Template leakage between train and val!"
    assert len(train_tfs.intersection(test_tfs)) == 0, "Template leakage between train and test!"
    assert len(val_tfs.intersection(test_tfs)) == 0, "Template leakage between val and test!"
    print("  ✔ Zero template family leakage verified across all partitions.")

    # Narrative exact match overlap check between Train and Test
    train_narratives = set(df[df["partition_split"] == "train"]["description"].str.strip().str.lower())
    test_narratives = set(df[df["partition_split"] == "test"]["description"].str.strip().str.lower())
    narrative_leakage = train_narratives.intersection(test_narratives)
    # Ignore empty string if present in challenge set
    narrative_leakage.discard("")
    assert len(narrative_leakage) == 0, f"Narrative leakage detected between train and test: {len(narrative_leakage)} matches!"
    print(f"  ✔ Narrative leakage audit passed: 0 repeated narratives between Train and Test.")

    # 6. Write Final CSV and Compute SHA-256
    print("\n[5/5] Exporting dataset to CSV and calculating SHA-256 checksum...")
    df.to_csv(OUTPUT_CSV, index=False)

    sha256_hash = hashlib.sha256()
    with open(OUTPUT_CSV, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    csv_sha256 = sha256_hash.hexdigest()

    print(f"\nDataset successfully generated at: {OUTPUT_CSV}")
    print(f"Final SHA-256 Checksum: {csv_sha256}")
    print(f"Total rows: {len(df):,}")

    # Print summary statistics
    print("\n--- CATEGORY BREAKDOWN ---")
    print(df["ground_truth_category"].value_counts())

    print("\n--- PARTITION BREAKDOWN ---")
    print(df["partition_split"].value_counts())

    print("\n--- TRANSACTION DIRECTION BREAKDOWN ---")
    print(df["transaction_direction"].value_counts())

    print("\n--- CHALLENGE CASE SUMMARY ---")
    print(df["challenge_reason"].value_counts())


if __name__ == "__main__":
    main()
