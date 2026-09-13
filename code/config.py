"""Configuration, paths, constants, and schema definitions for Buy or Wait? financial agent."""

from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = REPO_ROOT / "code"
DATASET_DIR = REPO_ROOT / "dataset"
MEDIA_DIR = DATASET_DIR / "media" / "images"

REQUESTS_CSV = DATASET_DIR / "requests.csv"
SAMPLE_REQUESTS_CSV = DATASET_DIR / "sample_requests.csv"
FINANCIAL_PROFILES_CSV = DATASET_DIR / "financial_profiles.csv"
FINANCIAL_EVENTS_CSV = DATASET_DIR / "financial_events.csv"
EXCHANGE_RATES_CSV = DATASET_DIR / "exchange_rates.csv"
REQUEST_PAYMENT_OPTIONS_CSV = DATASET_DIR / "request_payment_options.csv"
MESSAGES_CSV = DATASET_DIR / "messages.csv"
IMAGES_CSV = DATASET_DIR / "images.csv"
OUTPUT_CSV = REPO_ROOT / "output.csv"
USAGE_REPORT_MD = CODE_DIR / "evaluation" / "usage_report.md"
ENV_FILE = REPO_ROOT / ".env"

# Auto-load .env file if present
import os
if ENV_FILE.exists():
    try:
        with open(ENV_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v
    except Exception:
        pass

# Output CSV Column Order
OUTPUT_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]

# Allowed Categorical Values
AFFORDABILITY_STATUSES = {
    "affordable_now",
    "affordable_with_plan",
    "affordable_later",
    "not_affordable",
}

RECOMMENDED_PAYMENT_METHODS = {
    "full_payment",
    "partial_payment",
    "installments",
    "wait",
    "not_recommended",
}

# Forecast horizon in days
FORECAST_HORIZON_DAYS = 90
