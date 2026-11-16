"""
STS Fintech – Milestone 2 – Step 7.3A: Enterprise Financial Data Generator V2
Generates realistic, relational synthetic financial datasets for machine learning.
Ensures:
- Realistic category relationships with vendor categories, departments, and merchant narratives.
- Zero category name leakage in transaction descriptions.
- Complex anomaly injection scenarios with separate ground truth provenance.
- Coherent time-series seasonality, holiday calendar effects, and growth trends.
- 100% mathematical reconciliation between transaction ledger actuals and departmental budgets.
- Pre-computed intra-month snapshots at Days 7, 14, and 21.
- Realistic vendor billing lifecycle with modeled payment dates, delays, disputes, and duplicates.
"""

import os
import json
import random
import hashlib
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd


# ==============================================================================
# CONFIGURATION
# ==============================================================================

class GeneratorConfig:
    """Configurable parameters for V2 financial dataset generation."""
    def __init__(
        self,
        seed: int = 42,
        start_date: str = "2024-01-01",
        end_date: str = "2026-12-31",
        num_departments: int = 50,
        num_employees: int = 5000,
        num_vendors: int = 2000,
        num_transactions: int = 60000,
        num_invoices: int = 10000,
        num_bills: int = 10000,
        output_dir: str = "data/v2/raw"
    ):
        self.seed = seed
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        self.num_departments = num_departments
        self.num_employees = num_employees
        self.num_vendors = num_vendors
        self.num_transactions = num_transactions
        self.num_invoices = num_invoices
        self.num_bills = num_bills
        self.output_dir = output_dir


# ==============================================================================
# DOMAIN DATA DICTIONARIES & TAXONOMY
# ==============================================================================

INDIAN_CITIES = [
    "Chennai", "Bangalore", "Hyderabad", "Mumbai", "Pune",
    "Delhi NCR", "Kolkata", "Ahmedabad", "Coimbatore", "Kochi"
]

FIRST_NAMES = [
    "Aarav", "Aditi", "Ajay", "Amit", "Ananya", "Anil", "Anjali", "Arjun",
    "Deepak", "Divya", "Gaurav", "Gayatri", "Harish", "Ishaan", "Karthik",
    "Kavita", "Krishna", "Lakshmi", "Manish", "Meera", "Manoj", "Neha",
    "Nikhil", "Pooja", "Pranav", "Priya", "Rahul", "Rajesh", "Ramesh",
    "Ritu", "Rohan", "Sandhya", "Sanjay", "Shreya", "Siddharth", "Sneha",
    "Suresh", "Swati", "Varun", "Venkatesh", "Vidya", "Vikram", "Vishal"
]

LAST_NAMES = [
    "Agarwal", "Balasubramanian", "Bose", "Chawla", "Deshmukh", "Gupta",
    "Iyer", "Jain", "Kapoor", "Kulkarni", "Kumar", "Mehta", "Menon",
    "Murthy", "Nair", "Patel", "Pillai", "Prasad", "Ranganathan", "Rao",
    "Reddy", "Saxena", "Sen", "Sharma", "Singh", "Srinivasan", "Sundaram",
    "Verma", "Venkataraman", "Yadav"
]

EXPENSE_CATEGORIES = [
    "Cloud Services",
    "Software & SaaS",
    "Hardware & Equipment",
    "IT Services & Infrastructure",
    "Travel & Lodging",
    "Transportation & Logistics",
    "Office Supplies & Printing",
    "Facilities & Maintenance",
    "Utilities & Telecommunications",
    "Legal & Professional Services",
    "Marketing & Advertising",
    "Meals & Entertainment",
    "Employee Benefits & Healthcare",
    "Recruitment & Staffing"
]

VENDOR_CATEGORIES = [
    "Cloud Computing Provider",
    "Enterprise SaaS Solutions",
    "Computer Hardware & Peripherals",
    "Network Infrastructure & Cabling",
    "Corporate Travel Management",
    "Fleet & Courier Logistics",
    "Commercial Stationery & Supplies",
    "Building Facilities & HVAC",
    "Telecom & Fiber Leased Lines",
    "Audit, Legal & Tax Consultancy",
    "Digital Marketing & Media Agency",
    "Corporate Catering & Dining",
    "Group Insurance & Healthcare",
    "Executive Search & Staffing"
]

# Mapping between Vendor Categories and Primary Expense Categories
VENDOR_CATEGORY_TO_EXPENSE = {
    "Cloud Computing Provider": ("Cloud Services", 0.90),
    "Enterprise SaaS Solutions": ("Software & SaaS", 0.88),
    "Computer Hardware & Peripherals": ("Hardware & Equipment", 0.85),
    "Network Infrastructure & Cabling": ("IT Services & Infrastructure", 0.85),
    "Corporate Travel Management": ("Travel & Lodging", 0.88),
    "Fleet & Courier Logistics": ("Transportation & Logistics", 0.86),
    "Commercial Stationery & Supplies": ("Office Supplies & Printing", 0.86),
    "Building Facilities & HVAC": ("Facilities & Maintenance", 0.88),
    "Telecom & Fiber Leased Lines": ("Utilities & Telecommunications", 0.88),
    "Audit, Legal & Tax Consultancy": ("Legal & Professional Services", 0.90),
    "Digital Marketing & Media Agency": ("Marketing & Advertising", 0.88),
    "Corporate Catering & Dining": ("Meals & Entertainment", 0.88),
    "Group Insurance & Healthcare": ("Employee Benefits & Healthcare", 0.92),
    "Executive Search & Staffing": ("Recruitment & Staffing", 0.90)
}

# Rich descriptions without category names (prevents simple string matching leakage)
REALISTIC_NARRATIVES = {
    "Cloud Services": [
        "Monthly EC2 on-demand compute and S3 storage consumption",
        "Kubernetes container cluster production host billing",
        "Dedicated cloud database instance multi-AZ hosting fee",
        "Content delivery network data transfer bandwidth charges",
        "Cloud object storage cold archive quarterly bill",
        "Virtual private cloud inter-region peering data transfer"
    ],
    "Software & SaaS": [
        "Annual enterprise CRM team seat license renewal",
        "Code repository organization plan user tier billing",
        "Real-time observability platform monthly host subscription",
        "Project tracking workspace premium seat allotment",
        "Video conferencing enterprise account annual licensing",
        "Collaborative UI design software quarterly license seats"
    ],
    "Hardware & Equipment": [
        "Developer workstation laptop batch procurement",
        "27-inch 4K dual monitor display units purchase",
        "Enterprise network rackmount switch and patch panels",
        "Ergonomic keyboard and wireless peripheral accessories",
        "High-performance memory upgrade modules for build servers",
        "Executive ultra-thin notebook replacement unit"
    ],
    "IT Services & Infrastructure": [
        "Tier-4 datacenter rack space co-location quarterly fee",
        "Structured fiber-optic cabling installation 4th floor",
        "Annual enterprise firewall hardware security maintenance",
        "Onsite IT field engineer scheduled workstation maintenance",
        "SSL/TLS wildcard digital certificate authority renewal",
        "Enterprise identity access gateway deployment consulting"
    ],
    "Travel & Lodging": [
        "Round-trip air ticket domestic client technical workshop",
        "Corporate guest house executive lodging 4 nights onsite",
        "Commercial hotel suite room accommodation regional audit",
        "Inter-city executive train ticket senior delegate visit",
        "Airport transit terminal day-use room accommodation",
        "Annual tech leadership summit delegate hotel reservation"
    ],
    "Transportation & Logistics": [
        "Monthly pooled corporate cab fleet employee commute",
        "Express priority documents inter-branch courier freight",
        "Regional client site local taxi voucher reimbursement",
        "Material transport heavy cargo logistics dispatch fee",
        "Airport executive car rental transfers for overseas client",
        "Emergency datacenter hard-drive courier delivery"
    ],
    "Office Supplies & Printing": [
        "A4 80gsm premium copier paper cartons bulk inventory",
        "Replacement high-yield black laser toner cartridges",
        "Whiteboard dry-erase markers and desktop stationery pack",
        "Corporate branded employee welcome onboarding stationery",
        "Thermal receipt roll boxes and packaging packing tape",
        "Heavy-duty presentation binder folders and document sleeves"
    ],
    "Facilities & Maintenance": [
        "Quarterly central HVAC cooling tower chiller plant AMC",
        "Commercial elevator scheduled lift inspection and service",
        "Corporate facility deep sanitization and pest treatment",
        "Plumbing fixtures repair and washroom fittings overhaul",
        "Electrical distribution panel circuit breaker maintenance",
        "Fire suppression extinguisher annual safety hydro-test"
    ],
    "Utilities & Telecommunications": [
        "Commercial state electricity distribution high-tension tariff",
        "1Gbps primary fiber leased line internet broadband bill",
        "Corporate postpaid corporate fleet pooled talk-time/data",
        "Municipal commercial property water supply utility bill",
        "Backup diesel generator fuel delivery for power outage",
        "SIP trunk enterprise telephony incoming DID channel bill"
    ],
    "Legal & Professional Services": [
        "Statutory internal audit retainer professional service fee",
        "Trademark intellectual property filing and legal advisory",
        "Cross-border transfer pricing statutory compliance counsel",
        "Company secretarial board governance annual filings",
        "Labor law compliance review and payroll policy audit",
        "Commercial contract dispute arbitration counsel retainer"
    ],
    "Marketing & Advertising": [
        "Search engine sponsored keyword pay-per-click campaign",
        "Professional social network lead generation sponsored post",
        "Annual international fintech trade expo exhibition booth",
        "Brand video production and commercial creative agency fee",
        "Digital billboard display outdoor hoardings regional campaign",
        "Customer case study collateral copywriting and layout agency"
    ],
    "Meals & Entertainment": [
        "Client project milestone dinner delegation hospitality",
        "Late-night overtime food allowance engineering launch sprint",
        "Quarterly all-hands departmental team lunch catering",
        "New joinee cohort orientation breakfast buffet spread",
        "Executive strategy offsite refreshments and beverage pantry",
        "Vendor relationship review executive working lunch"
    ],
    "Employee Benefits & Healthcare": [
        "Corporate group health insurance annual family floater premium",
        "Executive annual preventive comprehensive medical checkup",
        "Employee mental wellness subscription platform corporate access",
        "Workplace ergonomic assessment consultation sessions",
        "Corporate gym and fitness facility subsidized membership",
        "Accidental disability insurance policy group coverage"
    ],
    "Recruitment & Staffing": [
        "Online talent portal job posting database search license",
        "Executive search contingency placement success fee",
        "Pre-employment technical skills assessment portal credits",
        "Comprehensive candidate background verification screening",
        "Campus recruitment drive logistics and technical venue hiring",
        "Contract staffing agency managed payroll administration"
    ]
}

DEPARTMENT_SPECIFICATIONS = [
    ("Core Engineering", "Technology", 45_000_000),
    ("Cloud Platform Operations", "Technology", 55_000_000),
    ("Data Platform & Analytics", "Technology", 35_000_000),
    ("DevOps & SRE", "Technology", 30_000_000),
    ("Cybersecurity Operations", "Technology", 28_000_000),
    ("Quality Engineering", "Technology", 22_000_000),
    ("Mobile Application Dev", "Technology", 25_000_000),
    ("Frontend Web Platforms", "Technology", 24_000_000),
    ("Backend API Services", "Technology", 32_000_000),
    ("AI Research & Modeling", "Technology", 40_000_000),
    ("Corporate Finance", "Finance & Admin", 18_000_000),
    ("Financial Accounting", "Finance & Admin", 15_000_000),
    ("Treasury & Investments", "Finance & Admin", 20_000_000),
    ("Taxation & Regulatory", "Finance & Admin", 14_000_000),
    ("Internal Financial Audit", "Finance & Admin", 12_000_000),
    ("Financial Planning & Budgeting", "Finance & Admin", 16_000_000),
    ("Talent Acquisition", "Human Resources", 25_000_000),
    ("Employee Experience", "Human Resources", 18_000_000),
    ("Learning & Org Development", "Human Resources", 15_000_000),
    ("Total Rewards & Payroll", "Human Resources", 20_000_000),
    ("HR Operations & Systems", "Human Resources", 16_000_000),
    ("Enterprise Accounts Sales", "Commercial", 50_000_000),
    ("Commercial Sales South", "Commercial", 38_000_000),
    ("Commercial Sales West", "Commercial", 42_000_000),
    ("Commercial Sales North", "Commercial", 40_000_000),
    ("Global Business Development", "Commercial", 35_000_000),
    ("Client Account Management", "Commercial", 28_000_000),
    ("Pre-Sales Technical Solutions", "Commercial", 22_000_000),
    ("Brand & Communications", "Marketing", 30_000_000),
    ("Digital & Growth Marketing", "Marketing", 45_000_000),
    ("Product Marketing", "Marketing", 25_000_000),
    ("Content & Creative Studio", "Marketing", 18_000_000),
    ("Events & Sponsorships", "Marketing", 28_000_000),
    ("Strategic Sourcing", "Procurement & Supply", 20_000_000),
    ("Vendor Operations", "Procurement & Supply", 16_000_000),
    ("IT Asset Procurement", "Procurement & Supply", 35_000_000),
    ("Corporate Logistics", "Procurement & Supply", 18_000_000),
    ("Real Estate & Facilities", "Workplace Ops", 60_000_000),
    ("Office Administration", "Workplace Ops", 22_000_000),
    ("Physical Security Ops", "Workplace Ops", 18_000_000),
    ("Corporate Travel Desk", "Workplace Ops", 24_000_000),
    ("Legal & Corporate Secretarial", "Legal & Compliance", 22_000_000),
    ("Ethics & Regulatory Compliance", "Legal & Compliance", 16_000_000),
    ("Enterprise Risk Management", "Legal & Compliance", 15_000_000),
    ("Customer Success Tier 1", "Operations", 26_000_000),
    ("Customer Success Tier 2", "Operations", 24_000_000),
    ("Customer Implementation", "Operations", 28_000_000),
    ("Technical Support Desk", "Operations", 20_000_000),
    ("Product Architecture", "Product & Design", 25_000_000),
    ("UI/UX Product Design", "Product & Design", 18_000_000)
]


# ==============================================================================
# DATA GENERATOR ENGINE V2
# ==============================================================================

class EnterpriseDataGeneratorV2:
    """
    Synthesizes enterprise-grade relational financial datasets with verifiable
    accounting reconciliation, realistic category distributions, and rich billing lifecycles.
    """

    def __init__(self, config: GeneratorConfig):
        self.cfg = config
        random.seed(config.seed)
        np.random.seed(config.seed)

        os.makedirs(config.output_dir, exist_ok=True)

    def _random_person_name(self) -> str:
        return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"

    def _random_company_name(self, category: str) -> str:
        prefixes = [
            "Apex", "Zenith", "Tata", "Reliance", "Infosys", "Wipro", "HCL",
            "Mahindra", "Bajaj", "Adani", "Cognizant", "Larsen", "Godrej",
            "Cyient", "Persistent", "Mindtree", "Tech", "Global", "NextGen",
            "Prime", "Vanguard", "Starlight", "Nexus", "Horizon", "InnoWave"
        ]
        suffixes = {
            "Cloud Computing Provider": ["Cloud Services Pvt Ltd", "Datacenter Technologies", "Cloud Networks India"],
            "Enterprise SaaS Solutions": ["Software Systems Ltd", "SaaS Solutions", "Digital Workspace Tech"],
            "Computer Hardware & Peripherals": ["Infotech Hardware Ltd", "Micro Devices India", "Peripherals Corp"],
            "Network Infrastructure & Cabling": ["Networks & Cabling Ltd", "Fiber Communications", "Infra Systems"],
            "Corporate Travel Management": ["Corporate Travels India", "Business Voyages Ltd", "Holidays Corporate"],
            "Fleet & Courier Logistics": ["Logistics & Couriers Ltd", "Fleet Transport Services", "Express Cargo"],
            "Commercial Stationery & Supplies": ["Stationery & Paper Kraft", "Office Solutions Pvt Ltd", "Supplies Corp"],
            "Building Facilities & HVAC": ["Facility Management Ltd", "HVAC & Engineering Services", "Infra Care"],
            "Telecom & Fiber Leased Lines": ["Telecom Solutions", "Enterprise Fiber Communications", "Leased Line Networks"],
            "Audit, Legal & Tax Consultancy": ["Associates & Advisors LLP", "Legal Partners & Consultants", "Taxation Advisory"],
            "Digital Marketing & Media Agency": ["Media & Advertising Agency", "Digital Performance Labs", "Communications Ltd"],
            "Corporate Catering & Dining": ["Hospitality & Food Services", "Corporate Dining Solutions", "Catering Corp"],
            "Group Insurance & Healthcare": ["Health & Assurance Ltd", "Corporate Wellness Care", "Insurance Brokers"],
            "Executive Search & Staffing": ["Search Partners India", "Talent Solutions Ltd", "Staffing Consultants"]
        }
        sfx_list = suffixes.get(category, ["Enterprises Ltd", "Technologies India", "Services Pvt Ltd"])
        return f"{random.choice(prefixes)} {random.choice(sfx_list)}"

    # --------------------------------------------------------------------------
    # 1. DEPARTMENTS
    # --------------------------------------------------------------------------
    def generate_departments(self) -> pd.DataFrame:
        print("Generating departments...")
        depts = []
        for i in range(1, self.cfg.num_departments + 1):
            spec = DEPARTMENT_SPECIFICATIONS[(i - 1) % len(DEPARTMENT_SPECIFICATIONS)]
            dept_name = spec[0] if i <= len(DEPARTMENT_SPECIFICATIONS) else f"{spec[0]} {i // len(DEPARTMENT_SPECIFICATIONS) + 1}"
            division = spec[1]
            annual_budget = spec[2] * random.uniform(0.9, 1.15)

            depts.append({
                "department_id": i,
                "department_code": f"DEPT{i:03d}",
                "department_name": dept_name,
                "division": division,
                "manager_name": self._random_person_name(),
                "location": random.choice(INDIAN_CITIES),
                "annual_budget_baseline": round(annual_budget, 2)
            })

        df = pd.DataFrame(depts)
        df.to_csv(os.path.join(self.cfg.output_dir, "departments.csv"), index=False)
        return df

    # --------------------------------------------------------------------------
    # 2. EMPLOYEES
    # --------------------------------------------------------------------------
    def generate_employees(self, depts_df: pd.DataFrame) -> pd.DataFrame:
        print("Generating employees...")
        designations_by_level = [
            ("Associate", 25_000, 45_000, 25_000),
            ("Senior Associate", 45_000, 75_000, 50_000),
            ("Specialist / Lead", 75_000, 130_000, 100_000),
            ("Manager", 130_000, 220_000, 250_000),
            ("Senior Manager / Director", 220_000, 380_000, 1_000_000)
        ]

        emps = []
        dept_ids = depts_df["department_id"].tolist()

        for i in range(1, self.cfg.num_employees + 1):
            lvl = random.choices(designations_by_level, weights=[45, 30, 15, 7, 3])[0]
            salary = random.uniform(lvl[1], lvl[2])
            hire_offset = random.randint(0, 365 * 6)
            hire_date = date(2018, 1, 1) + timedelta(days=hire_offset)

            emps.append({
                "employee_id": f"EMP{i:06d}",
                "employee_name": self._random_person_name(),
                "department_id": random.choice(dept_ids),
                "designation": lvl[0],
                "employment_type": random.choices(["FULL_TIME", "CONTRACT", "INTERN"], weights=[88, 10, 2])[0],
                "monthly_base_salary": round(salary, 2),
                "spending_approval_limit": lvl[3],
                "joining_date": hire_date,
                "status": random.choices(["ACTIVE", "INACTIVE"], weights=[95, 5])[0]
            })

        df = pd.DataFrame(emps)
        df.to_csv(os.path.join(self.cfg.output_dir, "employees.csv"), index=False)
        return df

    # --------------------------------------------------------------------------
    # 3. VENDORS
    # --------------------------------------------------------------------------
    def generate_vendors(self) -> pd.DataFrame:
        print("Generating vendors...")
        vendors = []
        # Ensure at least 1 vendor per category
        for i, cat in enumerate(VENDOR_CATEGORIES, 1):
            rating = round(float(np.clip(random.gauss(3.8, 0.6), 1.5, 5.0)), 2)
            terms = random.choice([15, 30, 45, 60, 90])
            status = random.choices(["UNDER_REVIEW", "ACTIVE"], weights=[70, 30])[0] if rating < 2.5 else "ACTIVE"
            vendors.append({
                "vendor_id": f"VEN{i:05d}",
                "vendor_name": self._random_company_name(cat),
                "vendor_category": cat,
                "location": random.choice(INDIAN_CITIES),
                "contact_email": f"billing.inquiries@vendor{i:05d}.in",
                "payment_terms_days": terms,
                "vendor_rating": rating,
                "is_preferred_vendor": rating >= 4.2 and random.random() < 0.4,
                "status": status
            })

        for i in range(len(VENDOR_CATEGORIES) + 1, self.cfg.num_vendors + 1):
            cat = random.choice(VENDOR_CATEGORIES)
            rating = round(float(np.clip(random.gauss(3.8, 0.6), 1.5, 5.0)), 2)
            terms = random.choice([15, 30, 45, 60, 90])

            if rating < 2.5:
                status = random.choices(["UNDER_REVIEW", "ACTIVE"], weights=[70, 30])[0]
            else:
                status = "ACTIVE"

            vendors.append({
                "vendor_id": f"VEN{i:05d}",
                "vendor_name": self._random_company_name(cat),
                "vendor_category": cat,
                "location": random.choice(INDIAN_CITIES),
                "contact_email": f"billing.inquiries@vendor{i:05d}.in",
                "payment_terms_days": terms,
                "vendor_rating": rating,
                "is_preferred_vendor": rating >= 4.2 and random.random() < 0.4,
                "status": status
            })

        df = pd.DataFrame(vendors)
        df.to_csv(os.path.join(self.cfg.output_dir, "vendors.csv"), index=False)
        return df

    # --------------------------------------------------------------------------
    # 4. SALARIES (RECURRING PAYROLL)
    # --------------------------------------------------------------------------
    def generate_salaries(self, emps_df: pd.DataFrame) -> pd.DataFrame:
        print("Generating recurring payroll...")
        active_emps = emps_df[emps_df["status"] == "ACTIVE"].copy()
        salary_records = []
        curr_date = date(2024, 1, 1)
        end_date = date(2026, 12, 31)

        salary_months = []
        while curr_date <= end_date:
            salary_months.append(curr_date)
            # increment month
            if curr_date.month == 12:
                curr_date = date(curr_date.year + 1, 1, 1)
            else:
                curr_date = date(curr_date.year, curr_date.month + 1, 1)

        sal_id = 1
        for month_dt in salary_months:
            # Sample subset of active employees for realistic corporate sample
            sampled = active_emps.sample(frac=0.4, random_state=self.cfg.seed + month_dt.month)
            for _, emp in sampled.iterrows():
                base = emp["monthly_base_salary"]
                hra = base * 0.40
                allowances = base * 0.15
                bonus = base * 0.10 if (month_dt.month in [3, 12] and random.random() < 0.3) else 0.0
                deductions = base * 0.12
                gross = base + hra + allowances + bonus
                net = gross - deductions

                salary_records.append({
                    "salary_id": sal_id,
                    "employee_id": emp["employee_id"],
                    "department_id": emp["department_id"],
                    "salary_month": month_dt.isoformat(),
                    "basic_salary": round(base, 2),
                    "hra": round(hra, 2),
                    "allowances": round(allowances, 2),
                    "bonus": round(bonus, 2),
                    "deductions": round(deductions, 2),
                    "gross_salary": round(gross, 2),
                    "net_salary": round(net, 2)
                })
                sal_id += 1

        df = pd.DataFrame(salary_records)
        df.to_csv(os.path.join(self.cfg.output_dir, "salaries.csv"), index=False)
        return df

    # --------------------------------------------------------------------------
    # 5. TRANSACTIONS (RECONCILED CORE LEDGER)
    # --------------------------------------------------------------------------
    def generate_transactions(
        self,
        depts_df: pd.DataFrame,
        emps_df: pd.DataFrame,
        vendors_df: pd.DataFrame
    ) -> pd.DataFrame:
        print("Generating transactions (reconciled ledger foundation)...")

        # Create quick lookup mappings
        v_by_cat = {}
        for cat in VENDOR_CATEGORIES:
            v_by_cat[cat] = vendors_df[vendors_df["vendor_category"] == cat].to_dict("records")

        emps_by_dept = {}
        for dept_id in depts_df["department_id"]:
            emps_by_dept[dept_id] = emps_df[emps_df["department_id"] == dept_id].to_dict("records")

        # Indian public holidays (approximate calendar dates)
        holidays = {
            (1, 26), (8, 15), (10, 2), (11, 1), (12, 25)
        }

        start_dt = self.cfg.start_date
        total_days = (self.cfg.end_date - start_dt).days + 1

        tx_list = []
        tx_id = 1

        # Generate base transactions per day following macro seasonality
        for day_idx in range(total_days):
            current_day = start_dt + timedelta(days=day_idx)
            weekday = current_day.weekday()
            dom = current_day.day
            year_factor = 1.0 + (current_day.year - 2024) * 0.08  # 8% annual growth

            # Daily volume weighting
            if weekday < 5:
                vol_base = random.randint(50, 75)
                if (current_day.month, current_day.day) in holidays:
                    vol_base = int(vol_base * 0.35)
            else:
                vol_base = random.randint(8, 20)  # low weekend activity

            # Month-end vendor invoice surge (days 25 to 30)
            if dom >= 25:
                vol_base = int(vol_base * 1.30)

            daily_tx_count = int(vol_base * year_factor)

            for _ in range(daily_tx_count):
                if tx_id > self.cfg.num_transactions:
                    break

                # Pick department and representative employee
                dept_id = random.randint(1, self.cfg.num_departments)
                dept_emps = emps_by_dept[dept_id]
                emp = random.choice(dept_emps)

                # Select Vendor Category conditioned on Department function
                # (Tech depts buy more cloud/hardware; Commercial spends on travel/marketing)
                if dept_id <= 10:  # Technology
                    v_cat = random.choices(
                        VENDOR_CATEGORIES,
                        weights=[30, 25, 20, 15, 2, 1, 1, 1, 2, 1, 1, 1, 0, 0]
                    )[0]
                elif 22 <= dept_id <= 33:  # Commercial / Marketing
                    v_cat = random.choices(
                        VENDOR_CATEGORIES,
                        weights=[2, 8, 3, 1, 30, 10, 3, 1, 2, 3, 30, 5, 1, 1]
                    )[0]
                elif 38 <= dept_id <= 41:  # Workplace & Facilities
                    v_cat = random.choices(
                        VENDOR_CATEGORIES,
                        weights=[1, 2, 2, 5, 5, 15, 15, 35, 10, 2, 1, 2, 2, 3]
                    )[0]
                else:  # General Operations / HR / Finance / Legal
                    v_cat = random.choice(VENDOR_CATEGORIES)

                vendor = random.choice(v_by_cat[v_cat])

                # Determine Expense Category based on vendor mapping with legitimate cross-category noise
                primary_exp_cat, conf = VENDOR_CATEGORY_TO_EXPENSE[v_cat]
                if random.random() < conf:
                    exp_cat = primary_exp_cat
                else:
                    # Legitimate cross-category purchase (e.g. cloud vendor providing training/consulting)
                    exp_cat = random.choice(EXPENSE_CATEGORIES)

                # Realistic Amount Distribution by category
                if exp_cat in ["Hardware & Equipment", "Cloud Services"]:
                    base_amt = np.random.lognormal(mean=np.log(45_000), sigma=1.1)
                elif exp_cat in ["Legal & Professional Services", "Marketing & Advertising"]:
                    base_amt = np.random.lognormal(mean=np.log(60_000), sigma=1.2)
                elif exp_cat in ["Meals & Entertainment", "Office Supplies & Printing"]:
                    base_amt = np.random.lognormal(mean=np.log(3_500), sigma=0.8)
                else:
                    base_amt = np.random.lognormal(mean=np.log(15_000), sigma=0.9)

                base_amt = float(np.clip(base_amt, 150.0, 1_500_000.0))

                # Narrative from realistic catalog without leaking category name
                description = random.choice(REALISTIC_NARRATIVES[exp_cat])

                # Anomaly Injection Logic (ground-truth provenance tracked separately)
                is_anomaly = False
                anomaly_type = "NONE"

                # Scenario 1: Unusually large transaction (1.2% chance)
                if random.random() < 0.012:
                    base_amt *= random.uniform(6.0, 16.0)
                    is_anomaly = True
                    anomaly_type = "UNUSUAL_LARGE_AMOUNT"

                # Scenario 2: Split approval threshold (0.9% chance)
                # Employee purchases just under their authorization limit
                elif random.random() < 0.009:
                    limit = emp["spending_approval_limit"]
                    base_amt = float(limit - random.uniform(100.0, 1200.0))
                    is_anomaly = True
                    anomaly_type = "SPLIT_APPROVAL_THRESHOLD"

                # Scenario 3: Off-hours / weekend mismatch (0.8% chance)
                elif weekday >= 5 and exp_cat in ["Hardware & Equipment", "Facilities & Maintenance"] and random.random() < 0.15:
                    is_anomaly = True
                    anomaly_type = "OFF_HOURS_WEEKEND_OUTLIER"

                # Scenario 4: Unrated or poor vendor receiving high amount (0.6% chance)
                elif vendor["vendor_rating"] < 2.2 and base_amt > 100_000.0 and random.random() < 0.20:
                    is_anomaly = True
                    anomaly_type = "UNRATED_OR_HIGH_RISK_VENDOR"

                # Hard Negative: Legitimate large approved expense (e.g. annual server renewal)
                # large amount but NOT an anomaly
                elif random.random() < 0.015:
                    base_amt *= random.uniform(3.5, 6.0)
                    is_anomaly = False
                    anomaly_type = "HARD_NEGATIVE_APPROVED_SURGE"

                # Payment method conditioned on amount
                if base_amt > 200_000.0:
                    pay_method = "BANK_TRANSFER"
                elif base_amt > 20_000.0:
                    pay_method = random.choice(["BANK_TRANSFER", "CORPORATE_CREDIT_CARD"])
                else:
                    pay_method = random.choice(["CORPORATE_CREDIT_CARD", "UPI", "DIRECT_DEBIT", "CASH"])

                tx_list.append({
                    "transaction_id": tx_id,
                    "employee_id": emp["employee_id"],
                    "department_id": dept_id,
                    "vendor_id": vendor["vendor_id"],
                    "transaction_date": current_day.isoformat(),
                    "transaction_type": random.choice(["EXPENSE", "PAYMENT", "INVOICE_SETTLEMENT"]),
                    "category": exp_cat,  # Target
                    "description": description,  # Realistic narrative (leak-free)
                    "amount": round(base_amt, 2),
                    "currency": "INR",
                    "payment_method": pay_method,
                    "is_business_expense": True,
                    "is_anomaly": is_anomaly,  # Separate ground truth
                    "anomaly_type": anomaly_type  # Separate ground truth provenance
                })
                tx_id += 1

            if tx_id > self.cfg.num_transactions:
                break

        df = pd.DataFrame(tx_list)
        df.to_csv(os.path.join(self.cfg.output_dir, "transactions.csv"), index=False)
        return df

    # --------------------------------------------------------------------------
    # 6. BUDGETS & RECONCILIATION ENGINE (100% LEDGER MATCH)
    # --------------------------------------------------------------------------
    def generate_reconciled_budgets(self, tx_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        print("Generating 100% reconciled budgets and intra-month snapshots...")

        # Parse transaction dates
        tx = tx_df.copy()
        tx["date"] = pd.to_datetime(tx["transaction_date"])
        tx["year"] = tx["date"].dt.year
        tx["month"] = tx["date"].dt.month
        tx["day"] = tx["date"].dt.day

        # Compute ground truth departmental monthly actuals
        monthly_actuals = tx.groupby(["department_id", "year", "month"]).agg(
            actual_amount=("amount", "sum"),
            transaction_count=("transaction_id", "count")
        ).reset_index()

        budget_records = []
        snapshot_records = []

        budget_id = 1
        snapshot_id = 1

        for _, row in monthly_actuals.iterrows():
            d_id = int(row["department_id"])
            yr = int(row["year"])
            mo = int(row["month"])
            act = float(row["actual_amount"])
            tx_count = int(row["transaction_count"])

            # Determine realistic budget allocation planning policy:
            # - 60% within budget (utilization 70% to 90%)
            # - 25% approaching limit (utilization 88% to 99%)
            # - 15% overrun (utilization 104% to 135%)
            planning_mode = random.choices(["WITHIN", "APPROACHING", "OVERRUN"], weights=[60, 25, 15])[0]
            if planning_mode == "WITHIN":
                alloc = act / random.uniform(0.70, 0.90)
            elif planning_mode == "APPROACHING":
                alloc = act / random.uniform(0.88, 0.99)
            else:
                alloc = act / random.uniform(1.04, 1.35)

            alloc = round(alloc, 2)
            rem = round(alloc - act, 2)
            util = round((act / alloc) * 100.0, 2)

            budget_records.append({
                "budget_id": budget_id,
                "department_id": d_id,
                "budget_year": yr,
                "budget_month": mo,
                "allocated_amount": alloc,
                "actual_amount": round(act, 2),  # EXACT 100% sum of transactions
                "remaining_amount": rem,
                "utilization_percentage": util,
                "allocation_lines_count": random.randint(1, 5),
                "is_overspent": act > alloc
            })
            budget_id += 1

            # Compute intra-month snapshots at Days 7, 14, 21
            dept_tx_month = tx[(tx["department_id"] == d_id) & (tx["year"] == yr) & (tx["month"] == mo)]
            days_in_mo = 31 if mo in [1, 3, 5, 7, 8, 10, 12] else (28 if mo == 2 else 30)

            for snap_day in [7, 14, 21]:
                snap_tx = dept_tx_month[dept_tx_month["day"] <= snap_day]
                mtd_act = float(snap_tx["amount"].sum())
                mtd_cnt = int(len(snap_tx))
                elapsed_pct = round((snap_day / days_in_mo) * 100.0, 1)

                snapshot_records.append({
                    "snapshot_id": snapshot_id,
                    "department_id": d_id,
                    "budget_year": yr,
                    "budget_month": mo,
                    "as_of_day": snap_day,
                    "elapsed_month_pct": elapsed_pct,
                    "allocated_amount": alloc,
                    "month_to_date_actual": round(mtd_act, 2),
                    "transaction_count_mtd": mtd_cnt,
                    # Targets for evaluation only:
                    "final_actual_amount": round(act, 2),
                    "final_utilization_pct": util,
                    "is_overspent_final": act > alloc
                })
                snapshot_id += 1

        budgets_df = pd.DataFrame(budget_records)
        snapshots_df = pd.DataFrame(snapshot_records)

        budgets_df.to_csv(os.path.join(self.cfg.output_dir, "budgets.csv"), index=False)
        snapshots_df.to_csv(os.path.join(self.cfg.output_dir, "budget_snapshots.csv"), index=False)

        return budgets_df, snapshots_df

    # --------------------------------------------------------------------------
    # 7. INVOICES & BILLS (REALISTIC LIFECYCLE, PAYMENT DATES, DUPLICATES)
    # --------------------------------------------------------------------------
    def generate_invoices_and_bills(
        self,
        depts_df: pd.DataFrame,
        vendors_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        print("Generating invoices and bills with full payment lifecycle...")

        v_dict = vendors_df.set_index("vendor_id").to_dict("index")
        v_ids = vendors_df["vendor_id"].tolist()
        d_ids = depts_df["department_id"].tolist()

        invoices = []
        bills = []

        start_dt = self.cfg.start_date
        total_days = (self.cfg.end_date - start_dt).days + 1

        for i in range(1, self.cfg.num_invoices + 1):
            vid = random.choice(v_ids)
            did = random.choice(d_ids)
            v_info = v_dict[vid]
            terms = int(v_info["payment_terms_days"])

            # Invoice date
            inv_day_offset = random.randint(0, total_days - 45)
            inv_date = start_dt + timedelta(days=inv_day_offset)
            due_date = inv_date + timedelta(days=terms)

            subtotal = float(np.clip(np.random.lognormal(mean=np.log(45_000), sigma=1.2), 1_500.0, 1_200_000.0))
            tax = subtotal * 0.18
            total = subtotal + tax

            # Status and duplicates
            is_dup = random.random() < 0.025  # 2.5% realistic duplicates
            if is_dup:
                status = "REJECTED"
                rejection_reason = "DUPLICATE_SUBMISSION"
            else:
                status = random.choices(["APPROVED", "PENDING", "REJECTED", "DISPUTED"], weights=[78, 12, 7, 3])[0]
                rejection_reason = None
                if status == "REJECTED":
                    rejection_reason = random.choice(["PRICE_MISMATCH", "MISSING_PO_NUMBER", "DAMAGED_ITEMS", "UNAPPROVED_SERVICE"])

            invoices.append({
                "invoice_id": f"INV{i:08d}",
                "vendor_id": vid,
                "department_id": did,
                "invoice_date": inv_date.isoformat(),
                "due_date": due_date.isoformat(),
                "subtotal": round(subtotal, 2),
                "tax_amount": round(tax, 2),
                "total_amount": round(total, 2),
                "invoice_status": status,
                "is_duplicate": is_dup,
                "rejection_reason": rejection_reason
            })

            # Corresponding Accounts Payable Bill Lifecycle
            if i <= self.cfg.num_bills:
                # Payment Date simulation based on commercial behavior
                # 72% on-time, 18% late, 7% pending/disputed, 3% long overdue
                pay_behavior = random.choices(["ON_TIME", "LATE", "PENDING", "DEFAULT_OVERDUE"], weights=[72, 18, 7, 3])[0]

                if pay_behavior == "ON_TIME":
                    pay_delay = random.randint(0, terms)
                    pay_date = inv_date + timedelta(days=pay_delay)
                    pay_status = "PAID"
                    is_late = False
                elif pay_behavior == "LATE":
                    delay_days = random.randint(3, 45)
                    pay_date = due_date + timedelta(days=delay_days)
                    pay_status = "PAID"
                    is_late = True
                elif pay_behavior == "PENDING":
                    pay_date = None
                    pay_status = "PENDING"
                    is_late = False
                else:  # DEFAULT_OVERDUE
                    pay_date = None
                    pay_status = "OVERDUE"
                    is_late = True

                bills.append({
                    "bill_id": f"BILL{i:07d}",
                    "invoice_id": f"INV{i:08d}",
                    "vendor_id": vid,
                    "department_id": did,
                    "bill_date": inv_date.isoformat(),
                    "due_date": due_date.isoformat(),
                    "bill_amount": round(subtotal, 2),
                    "tax_amount": round(tax, 2),
                    "total_amount": round(total, 2),
                    "payment_status": pay_status,
                    "payment_date": pay_date.isoformat() if pay_date else None,
                    "is_late_payment": is_late,
                    "description": f"Accounts payable settlement for INV{i:08d}"
                })

        inv_df = pd.DataFrame(invoices)
        bills_df = pd.DataFrame(bills)

        inv_df.to_csv(os.path.join(self.cfg.output_dir, "invoices.csv"), index=False)
        bills_df.to_csv(os.path.join(self.cfg.output_dir, "bills.csv"), index=False)

        return inv_df, bills_df

    # --------------------------------------------------------------------------
    # 8. ORCHESTRATE ALL GENERATION & METADATA
    # --------------------------------------------------------------------------
    def run_all(self) -> Dict[str, Any]:
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        start_time = datetime.utcnow()
        print("=" * 80)
        print("STARTING ENTERPRISE FINANCIAL DATA GENERATOR V2")
        print(f"Output Directory: {self.cfg.output_dir}")
        print(f"Seed: {self.cfg.seed}")
        print("=" * 80)

        depts_df = self.generate_departments()
        emps_df = self.generate_employees(depts_df)
        vendors_df = self.generate_vendors()
        salaries_df = self.generate_salaries(emps_df)
        tx_df = self.generate_transactions(depts_df, emps_df, vendors_df)
        budgets_df, snapshots_df = self.generate_reconciled_budgets(tx_df)
        inv_df, bills_df = self.generate_invoices_and_bills(depts_df, vendors_df)

        elapsed = (datetime.utcnow() - start_time).total_seconds()
        print(f"\nGeneration completed in {elapsed:.2f} seconds.")

        # Compute SHA-256 Checksums
        datasets = {
            "departments.csv": depts_df,
            "employees.csv": emps_df,
            "vendors.csv": vendors_df,
            "salaries.csv": salaries_df,
            "transactions.csv": tx_df,
            "invoices.csv": inv_df,
            "bills.csv": bills_df,
            "budgets.csv": budgets_df,
            "budget_snapshots.csv": snapshots_df
        }

        checksums = {}
        row_counts = {}
        for fname in datasets.keys():
            fpath = os.path.join(self.cfg.output_dir, fname)
            sha = hashlib.sha256()
            with open(fpath, "rb") as fp:
                while chunk := fp.read(65536):
                    sha.update(chunk)
            checksums[fname] = sha.hexdigest()
            row_counts[fname] = len(datasets[fname])

        manifest = {
            "version": "2.0.0",
            "generation_timestamp": datetime.utcnow().isoformat(),
            "generator_script": "ml/data_generator_v2.py",
            "seed": self.cfg.seed,
            "date_range": [self.cfg.start_date.isoformat(), self.cfg.end_date.isoformat()],
            "target_scale": {
                "departments": len(depts_df),
                "employees": len(emps_df),
                "vendors": len(vendors_df),
                "transactions": len(tx_df),
                "invoices": len(inv_df),
                "bills": len(bills_df),
                "budgets": len(budgets_df),
                "budget_snapshots": len(snapshots_df),
                "salaries": len(salaries_df)
            },
            "checksums": checksums,
            "row_counts": row_counts
        }

        if self.cfg.output_dir == "data/v2/raw":
            manifest_path = "data/v2/GENERATOR_VERSION.json"
        else:
            manifest_path = os.path.join(self.cfg.output_dir, "GENERATOR_VERSION.json")

        with open(manifest_path, "w") as fp:
            json.dump(manifest, fp, indent=2)

        print(f"Persisted generator version manifest to {manifest_path}")
        return manifest


if __name__ == "__main__":
    cfg = GeneratorConfig()
    gen = EnterpriseDataGeneratorV2(cfg)
    gen.run_all()
