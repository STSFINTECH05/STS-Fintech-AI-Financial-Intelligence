"""
STS FINTECH - PERSONAL FINANCE CLASSIFIER SERVICE
Milestone 2 - Step 7.4C-3D (Experimental Microservice)

Provides isolated, thread-safe inference for personal finance transactions:
- Enforces cryptographic SHA-256 validation of the candidate bundle.
- Employs a custom NumpyUnpickler subclass that intercepts only __main__.pii_sanitizer_transform
  and resolves it to ml.training.train_personal_finance_v1.pii_sanitizer_transform without modifying sys.path.
- Preserves full joblib NumPy array reconstruction logic.
- Implements pre-vectorization context-aware PII sanitization (phone, UPI, card endings, accounts).
- Implements direction-aware safety guardrails (salary, loan, refund, ambiguity, OOD).
- Emits structured SUCCESS, ABSTAIN, or NOT_EVALUATED responses.
- Operates strictly text-only; does not store or log PII or raw transaction payloads.
"""

import os
import re
import hashlib
import logging
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from joblib.numpy_pickle import NumpyUnpickler

# Import transformer using existing package environment (zero sys.path mutations)
from ml.training.train_personal_finance_v1 import pii_sanitizer_transform

logger = logging.getLogger("sts_ml_service.personal_classifier")

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
BUNDLE_PATH = os.path.join(BASE_DIR, "ml", "models", "personal", "experimental", "personal_finance_bundle_v1_candidate.joblib")
EXPECTED_BUNDLE_SHA256 = "81b7f350f46de0bd98941451ae628c5afbc3702b85ceff85c9f4ed41877cd9a0"
MODEL_VERSION = "personal_finance_bundle_v1_candidate"

# Experimental Decision Thresholds (Internal Ranking Only)
RANKING_THRESHOLD = 0.65
TOP_TWO_MARGIN_THRESHOLD = 0.20

# OOD Amount Limits in INR
MIN_AMOUNT_INR = 1.0
MAX_AMOUNT_INR = 10000000.0  # 1 Crore INR


# -----------------------------------------------------------------------------
# Context-Aware PII Sanitization Patterns
# -----------------------------------------------------------------------------
PHONE_PATTERN = re.compile(r'(?:\+91[\-\s]?)?[6-9]\d{9}\b')
UPI_PATTERN = re.compile(r'\b[a-zA-Z0-9.\-_]{2,64}@[a-zA-Z]{2,64}\b')
ACCOUNT_PATTERN = re.compile(r'\b(?:acc(?:ount)?[\s#:\-]*)?\d{9,18}\b', re.IGNORECASE)

# Comprehensive context-aware Card pattern:
# Catches 'card ending 1234', 'card ending in 1234', 'card XXXX1234', 'card no. 1234', full 16-digit cards
CARD_PATTERN = re.compile(
    r'\b(?:card\s*(?:ending\s*(?:in\s*)?|no\.?\s*|number\s*)?|xxxx[\-\s]*)'
    r'(?:\d{4}[\-\s]?\d{4}[\-\s]?\d{4}[\-\s]?\d{4}|\d{4}|[xX*]{4,12}[\-\s]?\d{4})\b',
    re.IGNORECASE
)


def sanitize_input_text(text: str) -> str:
    """
    Context-aware PII sanitizer for inbound transaction narratives before model ingestion.
    Redacts phone numbers, UPI VPAs, full card numbers, and card-ending references.
    Preserves ordinary non-card 4-digit numbers like invoice years and calendar dates.
    """
    if not isinstance(text, str):
        return ""
    t = text
    # 1. Redact UPI VPAs
    t = UPI_PATTERN.sub("[UPI]", t)
    # 2. Redact Phone numbers
    t = PHONE_PATTERN.sub("[PHONE]", t)
    # 3. Redact Card references and endings
    t = CARD_PATTERN.sub("[CARD]", t)
    # 4. Redact Bank Account numbers
    t = ACCOUNT_PATTERN.sub("[ACCT]", t)
    # 5. Clean whitespace
    t = re.sub(r'\s+', ' ', t).strip()
    return t


# -----------------------------------------------------------------------------
# Joblib-Compatible Custom Unpickler (Zero sys.path Mutation)
# -----------------------------------------------------------------------------
class PersonalFinanceNumpyUnpickler(NumpyUnpickler):
    """
    Subclasses joblib.numpy_pickle.NumpyUnpickler to safely resolve serialized
    references to __main__.pii_sanitizer_transform without mutating sys.path,
    altering global interpreter state, or breaking joblib's NumPy persistence.
    """

    def find_class(self, module: str, name: str) -> Any:
        if module == "__main__" and name == "pii_sanitizer_transform":
            return pii_sanitizer_transform
        return super().find_class(module, name)


def safe_load_personal_bundle(bundle_path: str) -> Dict[str, Any]:
    """Loads candidate joblib artifact using PersonalFinanceNumpyUnpickler."""
    with open(bundle_path, "rb") as f:
        unpickler = PersonalFinanceNumpyUnpickler(bundle_path, f)
        return unpickler.load()


# -----------------------------------------------------------------------------
# Direction-Aware Regex Guardrails
# -----------------------------------------------------------------------------
# Incoming salary / payroll patterns
# Specifically targets salary, wages, stipend, bonus, payroll credit
SALARY_INCOME_REGEX = re.compile(
    r'(?i)\b(salary|payroll|stipend|wages?|monthly salary|payroll credit|salary credit|bonus payout)\b'
)

# Incoming loan / credit borrowing patterns
LOAN_INCOME_REGEX = re.compile(
    r'(?i)\b(personal loan|loan disbursement|disbursement|borrowed|lending|emi credit|loan credit|bajaj finance disbursement|kreditbee|moneyview)\b'
)

# Incoming refund / reversal / chargeback patterns
REFUND_INCOME_REGEX = re.compile(
    r'(?i)\b(refund|reversal|cashback|returned item|chargeback|merchant refund)\b'
)

# Insufficient information / generic banking strings
INSUFFICIENT_INFO_REGEX = re.compile(
    r'(?i)^(\s*|upi|payment|transfer|neft|imps|rtgs|debit|credit|misc|expenses?|bill|charge|txn|transaction|pos|atm|cash|cheque|fund transfer|to transfer|by transfer|imps transfer|upi transfer|neft transfer|\d+)$'
)

# Multi-category ambiguity patterns (detects explicit cross-category combinations)
# e.g. "AWS hosting and office furniture purchase", "flight tickets and laptop rent"
MULTI_CATEGORY_CONJUNCTIONS = re.compile(
    r'(?i)\b(and also|as well as|along with|plus purchase of|and purchase of|combined with)\b'
)

CATEGORY_KEYWORDS = {
    "infra": ["aws", "azure", "cloud", "hosting", "server", "vps", "digitalocean", "linode", "hetzner"],
    "software": ["software", "saas", "subscription", "license", "figma", "github", "jira", "slack", "zoom", "chatgpt"],
    "rent": ["rent", "office rent", "coworking", "wework", "deposit", "lease"],
    "payroll": ["salary", "payroll", "stipend", "wages", "contractor payment"],
    "utilities": ["electricity", "power", "water", "broadband", "internet", "wifi", "bescom", "airtel", "act fibernet"],
    "travel": ["flight", "hotel", "cab", "uber", "ola", "travel", "airline", "indigo", "train", "conveyance"],
    "legal": ["legal", "lawyer", "advocate", "compliance", "trademark", "retainer fee for legal", "audit fee", "ca fee"],
    "consulting": ["consulting", "consultation", "advisory", "technical consultation", "due diligence"],
    "investment": ["dividend", "interest", "fixed deposit", "fd interest", "mutual fund", "shares", "equity yield"]
}


def detect_multi_category_ambiguity(text: str) -> bool:
    """Detects narrative combining distinct categories via explicit conjunctions."""
    lower = text.lower()
    if not MULTI_CATEGORY_CONJUNCTIONS.search(lower):
        # Also check for presence of multiple distinct keyword clusters separated by ' and ' or ' & '
        if " and " in lower or " & " in lower or " + " in lower:
            matched_domains = set()
            for domain, kws in CATEGORY_KEYWORDS.items():
                if any(kw in lower for kw in kws):
                    matched_domains.add(domain)
            if len(matched_domains) >= 2:
                return True
        return False

    # If conjunction present, check if at least 2 distinct domain keywords are present
    matched_domains = set()
    for domain, kws in CATEGORY_KEYWORDS.items():
        if any(kw in lower for kw in kws):
            matched_domains.add(domain)
    return len(matched_domains) >= 2


class PersonalFinanceClassifierService:
    """
    Singleton service managing the lifecycle and inference of the personal finance bundle.
    Ensures safe single-load in-memory deserialization without affecting enterprise models.
    """

    _instance: Optional["PersonalFinanceClassifierService"] = None

    def __init__(self, bundle_path: str = BUNDLE_PATH):
        self.bundle_path = bundle_path
        self.is_loaded = False
        self.load_error: Optional[str] = None
        self.bundle_metadata: Dict[str, Any] = {}
        self.income_pipeline = None
        self.expense_pipeline = None
        self.income_classes: List[str] = []
        self.expense_classes: List[str] = []

        self._load_bundle_safe()

    @classmethod
    def get_instance(cls) -> "PersonalFinanceClassifierService":
        if cls._instance is None:
            cls._instance = PersonalFinanceClassifierService()
        return cls._instance

    def _load_bundle_safe(self) -> None:
        """Safely loads candidate bundle with cryptographic hash verification and custom unpickler."""
        try:
            if not os.path.exists(self.bundle_path):
                self.load_error = f"Artifact not found at {self.bundle_path}"
                logger.error(self.load_error)
                return

            # Cryptographic SHA-256 verification
            h = hashlib.sha256()
            with open(self.bundle_path, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            actual_sha = h.hexdigest()

            if actual_sha != EXPECTED_BUNDLE_SHA256:
                self.load_error = f"Artifact integrity failure: Expected {EXPECTED_BUNDLE_SHA256}, got {actual_sha}"
                logger.critical(self.load_error)
                return

            # Safe deserialization via PersonalFinanceNumpyUnpickler (Zero sys.path mutation)
            bundle = safe_load_personal_bundle(self.bundle_path)

            self.income_pipeline = bundle["income_pipeline"]
            self.expense_pipeline = bundle["expense_pipeline"]
            self.income_classes = bundle["income_classes"]
            self.expense_classes = bundle["expense_classes"]
            self.bundle_metadata = {
                "bundle_name": bundle.get("bundle_name"),
                "status": bundle.get("status"),
                "created_at": bundle.get("created_at"),
                "dataset_sha256": bundle.get("dataset_sha256"),
                "scikit_learn_version": bundle.get("scikit_learn_version"),
            }
            self.is_loaded = True
            logger.info("Successfully loaded personal finance candidate bundle (%s).", MODEL_VERSION)

        except Exception as e:
            self.is_loaded = False
            self.load_error = f"Fatal error deserializing candidate bundle: {str(e)}"
            logger.exception(self.load_error)

    def classify_transaction(
        self,
        transaction_direction: str,
        description: str,
        amount_inr: float,
        transaction_date: str
    ) -> Dict[str, Any]:
        """
        Executes direction-aware guardrails and model inference.
        Returns a dictionary adhering to PersonalClassifyResponse schema.
        """
        # 1. Model Availability Check
        if not self.is_loaded:
            return {
                "status": "NOT_EVALUATED",
                "category": None,
                "ranking_score": None,
                "score_semantics": "INTERNAL_RANKING_ONLY",
                "requires_manual_confirmation": True,
                "abstention_reason": "MODEL_UNAVAILABLE",
                "model_version": MODEL_VERSION
            }

        # 2. Context-Aware Inbound PII Sanitization & Clean
        desc_sanitized = sanitize_input_text(description)

        # 3. Insufficient Information Check
        if not desc_sanitized or INSUFFICIENT_INFO_REGEX.match(desc_sanitized):
            return {
                "status": "ABSTAIN",
                "category": None,
                "ranking_score": None,
                "score_semantics": "INTERNAL_RANKING_ONLY",
                "requires_manual_confirmation": True,
                "abstention_reason": "INSUFFICIENT_INFORMATION",
                "model_version": MODEL_VERSION
            }

        # 4. OOD Amount Boundary Check
        if amount_inr < MIN_AMOUNT_INR or amount_inr > MAX_AMOUNT_INR:
            return {
                "status": "ABSTAIN",
                "category": None,
                "ranking_score": None,
                "score_semantics": "INTERNAL_RANKING_ONLY",
                "requires_manual_confirmation": True,
                "abstention_reason": "OUT_OF_DISTRIBUTION_AMOUNT",
                "model_version": MODEL_VERSION
            }

        # 5. Multi-Category Ambiguity Check
        if detect_multi_category_ambiguity(desc_sanitized):
            return {
                "status": "ABSTAIN",
                "category": None,
                "ranking_score": None,
                "score_semantics": "INTERNAL_RANKING_ONLY",
                "requires_manual_confirmation": True,
                "abstention_reason": "AMBIGUOUS",
                "model_version": MODEL_VERSION
            }

        # 6. Direction-Aware Guardrails
        if transaction_direction == "income":
            # Guardrail A: Incoming Salary / Payroll
            if SALARY_INCOME_REGEX.search(desc_sanitized):
                return {
                    "status": "ABSTAIN",
                    "category": None,
                    "ranking_score": None,
                    "score_semantics": "INTERNAL_RANKING_ONLY",
                    "requires_manual_confirmation": True,
                    "abstention_reason": "UNSUPPORTED_INCOME_SALARY",
                    "model_version": MODEL_VERSION
                }

            # Guardrail B: Incoming Personal Loan Disbursement
            if LOAN_INCOME_REGEX.search(desc_sanitized):
                return {
                    "status": "ABSTAIN",
                    "category": None,
                    "ranking_score": None,
                    "score_semantics": "INTERNAL_RANKING_ONLY",
                    "requires_manual_confirmation": True,
                    "abstention_reason": "UNSUPPORTED_INCOME_LOAN",
                    "model_version": MODEL_VERSION
                }

            # Guardrail C: Incoming Refund / Reversal
            if REFUND_INCOME_REGEX.search(desc_sanitized):
                return {
                    "status": "ABSTAIN",
                    "category": None,
                    "ranking_score": None,
                    "score_semantics": "INTERNAL_RANKING_ONLY",
                    "requires_manual_confirmation": True,
                    "abstention_reason": "UNSUPPORTED_INCOME_REFUND",
                    "model_version": MODEL_VERSION
                }

            pipeline = self.income_pipeline

        elif transaction_direction == "expense":
            # Outgoing expenses do NOT reject legitimate Payroll, Loan EMI, or Refund charges
            pipeline = self.expense_pipeline

        else:
            return {
                "status": "NOT_EVALUATED",
                "category": None,
                "ranking_score": None,
                "score_semantics": "INTERNAL_RANKING_ONLY",
                "requires_manual_confirmation": True,
                "abstention_reason": "UNSUPPORTED_DIRECTION",
                "model_version": MODEL_VERSION
            }

        # 7. Execute Text-Only Inference
        try:
            probs = pipeline.predict_proba([desc_sanitized])[0]
            classes = pipeline.named_steps["classifier"].classes_

            best_idx = int(np.argmax(probs))
            top_category = str(classes[best_idx])
            top_score = float(probs[best_idx])

            # Calculate top-two margin
            sorted_probs = np.sort(probs)[::-1]
            top_two_delta = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 1.0

            # 8. Experimental Threshold Check
            if top_score < RANKING_THRESHOLD:
                return {
                    "status": "ABSTAIN",
                    "category": None,
                    "ranking_score": round(top_score, 4),
                    "score_semantics": "INTERNAL_RANKING_ONLY",
                    "requires_manual_confirmation": True,
                    "abstention_reason": "LOW_SCORE",
                    "model_version": MODEL_VERSION
                }

            if top_two_delta < TOP_TWO_MARGIN_THRESHOLD:
                return {
                    "status": "ABSTAIN",
                    "category": None,
                    "ranking_score": round(top_score, 4),
                    "score_semantics": "INTERNAL_RANKING_ONLY",
                    "requires_manual_confirmation": True,
                    "abstention_reason": "AMBIGUOUS",
                    "model_version": MODEL_VERSION
                }

            # 9. Successful Suggestion
            return {
                "status": "SUCCESS",
                "category": top_category,
                "ranking_score": round(top_score, 4),
                "score_semantics": "INTERNAL_RANKING_ONLY",
                "requires_manual_confirmation": True,
                "abstention_reason": None,
                "model_version": MODEL_VERSION
            }

        except Exception as e:
            logger.error("Error executing text-only inference for direction %s: %s", transaction_direction, str(e))
            return {
                "status": "NOT_EVALUATED",
                "category": None,
                "ranking_score": None,
                "score_semantics": "INTERNAL_RANKING_ONLY",
                "requires_manual_confirmation": True,
                "abstention_reason": "INFERENCE_ERROR",
                "model_version": MODEL_VERSION
            }
