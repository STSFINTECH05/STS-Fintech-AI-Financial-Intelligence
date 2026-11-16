-- ============================================================
-- STS FINTECH AI FINANCIAL INTELLIGENCE
-- Standalone Financial Database
-- ============================================================

-- =========================
-- DEPARTMENTS
-- =========================

CREATE TABLE departments (
    department_id SERIAL PRIMARY KEY,
    department_code VARCHAR(20) UNIQUE NOT NULL,
    department_name VARCHAR(100) NOT NULL,
    manager_name VARCHAR(150),
    location VARCHAR(100),
    annual_budget NUMERIC(15,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- =========================
-- EMPLOYEES
-- =========================

CREATE TABLE employees (
    employee_id VARCHAR(20) PRIMARY KEY,
    employee_name VARCHAR(150) NOT NULL,
    department_id INTEGER REFERENCES departments(department_id),
    designation VARCHAR(100),
    employment_type VARCHAR(50),
    joining_date DATE,
    location VARCHAR(100),
    salary_grade VARCHAR(20),
    status VARCHAR(30) DEFAULT 'ACTIVE',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- =========================
-- SALARIES
-- =========================

CREATE TABLE salaries (
    salary_id BIGSERIAL PRIMARY KEY,
    employee_id VARCHAR(20) NOT NULL
        REFERENCES employees(employee_id),

    salary_month DATE NOT NULL,

    basic_salary NUMERIC(12,2),
    hra NUMERIC(12,2),
    allowances NUMERIC(12,2),
    bonus NUMERIC(12,2),
    deductions NUMERIC(12,2),

    gross_salary NUMERIC(12,2),
    net_salary NUMERIC(12,2),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- =========================
-- VENDORS
-- =========================

CREATE TABLE vendors (
    vendor_id VARCHAR(20) PRIMARY KEY,
    vendor_name VARCHAR(200) NOT NULL,
    vendor_category VARCHAR(100),
    location VARCHAR(100),
    contact_email VARCHAR(150),
    risk_level VARCHAR(30),
    payment_terms_days INTEGER,
    vendor_rating NUMERIC(3,2),
    status VARCHAR(30) DEFAULT 'ACTIVE',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- =========================
-- TRANSACTIONS
-- =========================

CREATE TABLE transactions (
    transaction_id BIGSERIAL PRIMARY KEY,

    employee_id VARCHAR(20)
        REFERENCES employees(employee_id),

    department_id INTEGER
        REFERENCES departments(department_id),

    vendor_id VARCHAR(20)
        REFERENCES vendors(vendor_id),

    transaction_date TIMESTAMP NOT NULL,

    transaction_type VARCHAR(30),
    category VARCHAR(100),

    description TEXT,

    amount NUMERIC(15,2) NOT NULL,

    payment_method VARCHAR(50),

    currency VARCHAR(10) DEFAULT 'INR',

    is_business_expense BOOLEAN DEFAULT TRUE,

    is_anomaly BOOLEAN DEFAULT FALSE,

    anomaly_score NUMERIC(6,4),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- =========================
-- BILLS
-- =========================

CREATE TABLE bills (
    bill_id VARCHAR(30) PRIMARY KEY,

    vendor_id VARCHAR(20)
        REFERENCES vendors(vendor_id),

    department_id INTEGER
        REFERENCES departments(department_id),

    bill_date DATE NOT NULL,

    due_date DATE,

    bill_amount NUMERIC(15,2),

    tax_amount NUMERIC(15,2),

    total_amount NUMERIC(15,2),

    payment_status VARCHAR(30),

    payment_date DATE,

    description TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- =========================
-- INVOICES
-- =========================

CREATE TABLE invoices (
    invoice_id VARCHAR(30) PRIMARY KEY,

    vendor_id VARCHAR(20)
        REFERENCES vendors(vendor_id),

    department_id INTEGER
        REFERENCES departments(department_id),

    invoice_date DATE,

    due_date DATE,

    subtotal NUMERIC(15,2),

    tax_amount NUMERIC(15,2),

    total_amount NUMERIC(15,2),

    invoice_status VARCHAR(30),

    payment_status VARCHAR(30),

    is_duplicate BOOLEAN DEFAULT FALSE,

    duplicate_score NUMERIC(6,4),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- =========================
-- BUDGETS
-- =========================

CREATE TABLE budgets (
    budget_id BIGSERIAL PRIMARY KEY,

    department_id INTEGER
        REFERENCES departments(department_id),

    budget_year INTEGER NOT NULL,

    budget_month INTEGER,

    allocated_amount NUMERIC(15,2),

    actual_amount NUMERIC(15,2),

    remaining_amount NUMERIC(15,2),

    utilization_percentage NUMERIC(7,2),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- ============================================================
-- INDEXES
-- ============================================================

CREATE INDEX idx_employee_department
ON employees(department_id);

CREATE INDEX idx_salary_employee
ON salaries(employee_id);

CREATE INDEX idx_salary_month
ON salaries(salary_month);

CREATE INDEX idx_transaction_employee
ON transactions(employee_id);

CREATE INDEX idx_transaction_department
ON transactions(department_id);

CREATE INDEX idx_transaction_vendor
ON transactions(vendor_id);

CREATE INDEX idx_transaction_date
ON transactions(transaction_date);

CREATE INDEX idx_transaction_category
ON transactions(category);

CREATE INDEX idx_bill_vendor
ON bills(vendor_id);

CREATE INDEX idx_invoice_vendor
ON invoices(vendor_id);

CREATE INDEX idx_budget_department
ON budgets(department_id);

CREATE INDEX idx_budget_year_month
ON budgets(budget_year, budget_month);