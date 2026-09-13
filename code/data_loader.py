"""Data loader and normalization utilities for Buy or Wait? financial agent."""

import csv
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from code.config import (
    EXCHANGE_RATES_CSV,
    FINANCIAL_EVENTS_CSV,
    FINANCIAL_PROFILES_CSV,
    IMAGES_CSV,
    MESSAGES_CSV,
    REQUEST_PAYMENT_OPTIONS_CSV,
    REQUESTS_CSV,
    SAMPLE_REQUESTS_CSV,
)
from code.models import (
    DecisionResult,
    EvaluationRequest,
    FinancialEvent,
    FinancialProfile,
    ImageEvidence,
    MessageEvidence,
    PaymentOption,
)


class ExchangeRateConverter:
    """Provides fixed, dated currency conversions from exchange_rates.csv."""

    def __init__(self, rates_file=EXCHANGE_RATES_CSV):
        # Key: (rate_date, from_currency, to_currency) -> rate
        self.rates: Dict[Tuple[str, str, str], float] = {}
        self.dates_for_pair: Dict[Tuple[str, str], List[str]] = {}

        if rates_file.exists():
            with open(rates_file, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    r_date = row["rate_date"].strip()
                    from_c = row["from_currency"].strip()
                    to_c = row["to_currency"].strip()
                    rate = float(row["rate"].strip())
                    self.rates[(r_date, from_c, to_c)] = rate

                    # Also store reverse rate if not already provided
                    rev_key = (r_date, to_c, from_c)
                    if rev_key not in self.rates and rate > 0:
                        self.rates[rev_key] = 1.0 / rate

                    pair = (from_c, to_c)
                    rev_pair = (to_c, from_c)
                    if pair not in self.dates_for_pair:
                        self.dates_for_pair[pair] = []
                    self.dates_for_pair[pair].append(r_date)
                    if rev_pair not in self.dates_for_pair:
                        self.dates_for_pair[rev_pair] = []
                    self.dates_for_pair[rev_pair].append(r_date)

            for pair in self.dates_for_pair:
                self.dates_for_pair[pair].sort()

    def convert(self, amount: float, from_currency: str, to_currency: str, date_str: str) -> float:
        """Converts amount to to_currency using rate for date_str."""
        if from_currency == to_currency or amount == 0:
            return amount

        # Try exact date match first
        key = (date_str, from_currency, to_currency)
        if key in self.rates:
            return amount * self.rates[key]

        # Try nearest prior date for the currency pair
        pair = (from_currency, to_currency)
        if pair in self.dates_for_pair and self.dates_for_pair[pair]:
            # Pick latest date <= date_str, or earliest available if all > date_str
            applicable_dates = [d for d in self.dates_for_pair[pair] if d <= date_str]
            chosen_date = applicable_dates[-1] if applicable_dates else self.dates_for_pair[pair][0]
            rate = self.rates.get((chosen_date, from_currency, to_currency))
            if rate is not None:
                return amount * rate

        # Direct 1:1 fallback if unknown
        return amount


class DataLoader:
    """Loads and indexes all dataset files."""

    def __init__(self):
        self.rate_converter = ExchangeRateConverter()
        self.profiles: Dict[str, FinancialProfile] = {}
        self.events_by_user: Dict[str, List[FinancialEvent]] = {}
        self.events_by_id: Dict[str, FinancialEvent] = {}
        self.payment_options_by_request: Dict[str, List[PaymentOption]] = {}
        self.messages_by_user: Dict[str, List[MessageEvidence]] = {}
        self.messages_by_request: Dict[str, List[MessageEvidence]] = {}
        self.images_by_event: Dict[str, ImageEvidence] = {}
        self.images_by_request: Dict[str, List[ImageEvidence]] = {}

        self._load_profiles()
        self._load_images()
        self._load_events()
        self._load_payment_options()
        self._load_messages()

    def _load_profiles(self):
        if not FINANCIAL_PROFILES_CSV.exists():
            return
        with open(FINANCIAL_PROFILES_CSV, mode="r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                uid = row["user_id"].strip()
                max_inst = row.get("max_installment_months", "").strip()
                max_inst_val = int(max_inst) if max_inst and max_inst.isdigit() else None

                priorities = [p.strip() for p in row.get("financial_priorities", "").split("|") if p.strip()]
                protect = set(p.strip() for p in row.get("expense_categories_to_protect", "").split("|") if p.strip())
                reduce_cat = set(p.strip() for p in row.get("expense_categories_user_is_willing_to_reduce", "").split("|") if p.strip())
                stop_cat = set(p.strip() for p in row.get("expense_categories_user_is_willing_to_stop", "").split("|") if p.strip())
                methods = set(p.strip() for p in row.get("payment_methods_user_will_consider", "").split("|") if p.strip())

                self.profiles[uid] = FinancialProfile(
                    user_id=uid,
                    home_currency=row["home_currency"].strip(),
                    current_available_balance=float(row["current_available_balance"]),
                    minimum_balance_to_keep=float(row["minimum_balance_to_keep"]),
                    financial_priorities=priorities,
                    expense_categories_to_protect=protect,
                    expense_categories_user_is_willing_to_reduce=reduce_cat,
                    expense_categories_user_is_willing_to_stop=stop_cat,
                    payment_methods_user_will_consider=methods,
                    max_installment_months=max_inst_val,
                )

    def _load_images(self):
        if not IMAGES_CSV.exists():
            return
        with open(IMAGES_CSV, mode="r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ev = ImageEvidence(
                    image_id=row["image_id"].strip(),
                    user_id=row["user_id"].strip(),
                    request_id=row.get("request_id", "").strip(),
                    related_event_id=row.get("related_event_id", "").strip(),
                )
                if ev.related_event_id:
                    self.images_by_event[ev.related_event_id] = ev
                if ev.request_id:
                    if ev.request_id not in self.images_by_request:
                        self.images_by_request[ev.request_id] = []
                    self.images_by_request[ev.request_id].append(ev)

    def _load_events(self):
        if not FINANCIAL_EVENTS_CSV.exists():
            return
        with open(FINANCIAL_EVENTS_CSV, mode="r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                amt_str = row.get("amount", "").strip()
                amt = float(amt_str) if amt_str else None

                min_amt_str = row.get("minimum_allowed_amount", "").strip()
                min_amt = float(min_amt_str) if min_amt_str else None

                ev = FinancialEvent(
                    event_id=row["event_id"].strip(),
                    user_id=row["user_id"].strip(),
                    event_type=row["event_type"].strip(),
                    description=row["description"].strip(),
                    category=row["category"].strip(),
                    direction=row["direction"].strip(),
                    amount=amt,
                    currency=row["currency"].strip(),
                    event_date=row["event_date"].strip(),
                    settlement_date=row["settlement_date"].strip(),
                    status=row["status"].strip(),
                    linked_event_id=row.get("linked_event_id", "").strip(),
                    flexibility=row.get("flexibility", "fixed").strip(),
                    minimum_allowed_amount=min_amt,
                )
                self.events_by_id[ev.event_id] = ev
                uid = ev.user_id
                if uid not in self.events_by_user:
                    self.events_by_user[uid] = []
                self.events_by_user[uid].append(ev)

    def _load_payment_options(self):
        if not REQUEST_PAYMENT_OPTIONS_CSV.exists():
            return
        with open(REQUEST_PAYMENT_OPTIONS_CSV, mode="r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                freq_str = row.get("payment_frequency_days", "").strip()
                freq = int(freq_str) if freq_str and freq_str.isdigit() else None
                fee = float(row.get("financing_fee", "0") or "0")
                tot = float(row.get("total_payable_amount", "0") or "0")

                opt = PaymentOption(
                    payment_option_id=row["payment_option_id"].strip(),
                    request_id=row["request_id"].strip(),
                    payment_method=row["payment_method"].strip(),
                    payment_amount=float(row["payment_amount"]),
                    number_of_payments=int(row["number_of_payments"]),
                    first_payment_date=row["first_payment_date"].strip(),
                    payment_frequency_days=freq,
                    financing_fee=fee,
                    total_payable_amount=tot,
                )
                rid = opt.request_id
                if rid not in self.payment_options_by_request:
                    self.payment_options_by_request[rid] = []
                self.payment_options_by_request[rid].append(opt)

    def _load_messages(self):
        if not MESSAGES_CSV.exists():
            return
        with open(MESSAGES_CSV, mode="r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                msg = MessageEvidence(
                    message_id=row["message_id"].strip(),
                    user_id=row["user_id"].strip(),
                    request_id=row.get("request_id", "").strip(),
                    related_event_id=row.get("related_event_id", "").strip(),
                    sent_at=row["sent_at"].strip(),
                    source_type=row["source_type"].strip(),
                    message_text=row["message_text"].strip(),
                )
                uid = msg.user_id
                if uid not in self.messages_by_user:
                    self.messages_by_user[uid] = []
                self.messages_by_user[uid].append(msg)
                if msg.request_id:
                    if msg.request_id not in self.messages_by_request:
                        self.messages_by_request[msg.request_id] = []
                    self.messages_by_request[msg.request_id].append(msg)

    def load_requests(self, filepath=REQUESTS_CSV) -> List[EvaluationRequest]:
        """Loads evaluation requests."""
        reqs = []
        with open(filepath, mode="r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                allows_part = row.get("allows_partial_payment", "").strip().lower() == "true"
                reqs.append(
                    EvaluationRequest(
                        request_id=row["request_id"].strip(),
                        user_id=row["user_id"].strip(),
                        request_date=row["request_date"].strip(),
                        request_type=row["request_type"].strip(),
                        requested_amount=float(row["requested_amount"]),
                        desired_completion_date=row["desired_completion_date"].strip(),
                        allows_partial_payment=allows_part,
                        request_text=row.get("request_text", "").strip(),
                    )
                )
        return reqs

    def load_sample_requests(self) -> List[Tuple[EvaluationRequest, DecisionResult]]:
        """Loads solved sample requests with ground truth."""
        pairs = []
        with open(SAMPLE_REQUESTS_CSV, mode="r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                allows_part = row.get("allows_partial_payment", "").strip().lower() == "true"
                req = EvaluationRequest(
                    request_id=row["request_id"].strip(),
                    user_id=row["user_id"].strip(),
                    request_date=row["request_date"].strip(),
                    request_type=row["request_type"].strip(),
                    requested_amount=float(row["requested_amount"]),
                    desired_completion_date=row["desired_completion_date"].strip(),
                    allows_partial_payment=allows_part,
                    request_text=row.get("request_text", "").strip(),
                )
                safe_amt = float(row["amount_safe_to_pay"])
                res = DecisionResult(
                    request_id=row["request_id"].strip(),
                    amount_safe_to_pay=safe_amt,
                    affordability_status=row["affordability_status"].strip(),
                    recommended_payment_method=row["recommended_payment_method"].strip(),
                    payment_plan=row["payment_plan"].strip(),
                    earliest_date_for_full_payment=row["earliest_date_for_full_payment"].strip(),
                    spending_changes_needed=row["spending_changes_needed"].strip(),
                    decision_explanation=row["decision_explanation"].strip(),
                )
                pairs.append((req, res))
        return pairs
