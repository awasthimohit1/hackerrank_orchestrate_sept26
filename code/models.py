"""Domain data models for Buy or Wait? financial agent."""

from dataclasses import dataclass, field
from typing import List, Optional, Set


@dataclass
class FinancialProfile:
    user_id: str
    home_currency: str
    current_available_balance: float
    minimum_balance_to_keep: float
    financial_priorities: List[str]
    expense_categories_to_protect: Set[str]
    expense_categories_user_is_willing_to_reduce: Set[str]
    expense_categories_user_is_willing_to_stop: Set[str]
    payment_methods_user_will_consider: Set[str]
    max_installment_months: Optional[int] = None


@dataclass
class FinancialEvent:
    event_id: str
    user_id: str
    event_type: str
    description: str
    category: str
    direction: str  # 'credit', 'debit', 'non_cash'
    amount: Optional[float]
    currency: str
    event_date: str
    settlement_date: str
    status: str  # 'settled', 'scheduled', 'pending', 'cancelled', 'failed', 'unrealized'
    linked_event_id: str = ""
    flexibility: str = "fixed"  # 'fixed', 'reducible', 'stoppable', 'reducible_or_stoppable'
    minimum_allowed_amount: Optional[float] = None


@dataclass
class PaymentOption:
    payment_option_id: str
    request_id: str
    payment_method: str  # 'full_payment', 'installments'
    payment_amount: float
    number_of_payments: int
    first_payment_date: str
    payment_frequency_days: Optional[int] = None
    financing_fee: float = 0.0
    total_payable_amount: float = 0.0


@dataclass
class MessageEvidence:
    message_id: str
    user_id: str
    request_id: str
    related_event_id: str
    sent_at: str
    source_type: str
    message_text: str


@dataclass
class ImageEvidence:
    image_id: str
    user_id: str
    request_id: str
    related_event_id: str


@dataclass
class EvaluationRequest:
    request_id: str
    user_id: str
    request_date: str
    request_type: str
    requested_amount: float
    desired_completion_date: str
    allows_partial_payment: bool
    request_text: str


@dataclass
class DecisionResult:
    request_id: str
    amount_safe_to_pay: float
    affordability_status: str
    recommended_payment_method: str
    payment_plan: str
    earliest_date_for_full_payment: str
    spending_changes_needed: str
    decision_explanation: str
