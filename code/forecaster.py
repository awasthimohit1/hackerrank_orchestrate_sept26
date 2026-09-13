"""90-day daily cashflow simulator for Buy or Wait? financial agent."""

from calendar import monthrange
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

from code.config import FORECAST_HORIZON_DAYS
from code.data_loader import DataLoader
from code.evidence_extractor import EvidenceExtractor
from code.models import EvaluationRequest, FinancialEvent, FinancialProfile


@dataclass
class RecurringStream:
    category: str
    description: str
    amount: float
    day_of_month: int
    frequency: str  # 'monthly', 'interval'
    interval_days: int
    flexibility: str  # 'fixed', 'stoppable', 'reducible', 'reducible_or_stoppable'
    minimum_allowed_amount: Optional[float]
    last_event_id: str
    last_settlement_date: str


class CashflowForecaster:
    """Reconstructs cash balances and simulates 90-day daily cashflow."""

    def __init__(self, data_loader: DataLoader, evidence_extractor: Optional[EvidenceExtractor] = None):
        self.loader = data_loader
        self.extractor = evidence_extractor or EvidenceExtractor()

    def _resolve_event_amount(self, event: FinancialEvent) -> float:
        """Returns the event amount in user's home currency, resolving missing amounts from images."""
        amt = event.amount
        if amt is None:
            img_ev = self.loader.images_by_event.get(event.event_id)
            if img_ev:
                extracted = self.extractor.extract_image_amount(img_ev.image_id)
                if extracted is not None:
                    amt = extracted
        if amt is None:
            amt = 0.0

        profile = self.loader.profiles.get(event.user_id)
        if profile and event.currency != profile.home_currency and amt > 0:
            amt = self.loader.rate_converter.convert(
                amt, event.currency, profile.home_currency, event.settlement_date
            )
        return amt

    def detect_recurring_streams(self, user_id: str, request_date: str) -> List[RecurringStream]:
        """Detects active recurring expense and income streams from history up to request_date."""
        events = self.loader.events_by_user.get(user_id, [])
        past_events = [
            e for e in events
            if e.settlement_date <= request_date
            and e.status in ("settled", "scheduled")
            and e.direction == "debit"
        ]

        variable_cats = {"groceries", "transport", "dining"}
        grouped_var = defaultdict(list)
        grouped_fixed = defaultdict(list)

        for e in past_events:
            if e.category in variable_cats:
                grouped_var[e.category].append(e)
            else:
                grouped_fixed[(e.category, e.description)].append(e)

        streams = []

        # Process fixed bills
        for (cat, desc), ev_list in grouped_fixed.items():
            if len(ev_list) < 2:
                continue
            ev_list.sort(key=lambda x: x.settlement_date)
            last_ev = ev_list[-1]
            last_dt = datetime.strptime(last_ev.settlement_date, "%Y-%m-%d")

            amt = self._resolve_event_amount(last_ev)
            min_amt = last_ev.minimum_allowed_amount
            if min_amt is not None and last_ev.currency != self.loader.profiles[user_id].home_currency:
                min_amt = self.loader.rate_converter.convert(
                    min_amt, last_ev.currency, self.loader.profiles[user_id].home_currency, last_ev.settlement_date
                )

            streams.append(
                RecurringStream(
                    category=cat,
                    description=desc,
                    amount=amt,
                    day_of_month=last_dt.day,
                    frequency="monthly",
                    interval_days=30,
                    flexibility=last_ev.flexibility,
                    minimum_allowed_amount=min_amt,
                    last_event_id=last_ev.event_id,
                    last_settlement_date=last_ev.settlement_date,
                )
            )

        # Process variable living expenses
        for cat, ev_list in grouped_var.items():
            if len(ev_list) < 2:
                continue
            ev_list.sort(key=lambda x: x.settlement_date)
            last_ev = ev_list[-1]

            intervals = []
            for i in range(1, len(ev_list)):
                d1 = datetime.strptime(ev_list[i - 1].settlement_date, "%Y-%m-%d")
                d2 = datetime.strptime(ev_list[i].settlement_date, "%Y-%m-%d")
                intervals.append((d2 - d1).days)

            avg_interval = sum(intervals) / len(intervals)
            if avg_interval <= 9:
                freq = "weekly"
            elif avg_interval <= 18:
                freq = "biweekly"
            else:
                freq = "monthly"

            avg_amt = sum(self._resolve_event_amount(e) for e in ev_list) / len(ev_list)

            min_amt = last_ev.minimum_allowed_amount
            if min_amt is not None and last_ev.currency != self.loader.profiles[user_id].home_currency:
                min_amt = self.loader.rate_converter.convert(
                    min_amt, last_ev.currency, self.loader.profiles[user_id].home_currency, last_ev.settlement_date
                )

            streams.append(
                RecurringStream(
                    category=cat,
                    description=last_ev.description,
                    amount=avg_amt,
                    day_of_month=datetime.strptime(last_ev.settlement_date, "%Y-%m-%d").day,
                    frequency=freq,
                    interval_days=30,
                    flexibility=last_ev.flexibility,
                    minimum_allowed_amount=min_amt,
                    last_event_id=last_ev.event_id,
                    last_settlement_date=last_ev.settlement_date,
                )
            )

        return streams

    def simulate_90_days(
        self,
        request: EvaluationRequest,
        spending_changes: Optional[List[Tuple[str, str, Optional[float]]]] = None,
        additional_payment_plan: Optional[List[Tuple[str, float]]] = None,
    ) -> Tuple[List[float], float, float]:
        """Simulates daily balance over 90 days.

        spending_changes: list of ('stop'|'reduce_to', event_id, new_amount)
        additional_payment_plan: list of (date_str, payment_amount)
        Returns: (daily_balances, min_balance_reached, min_headroom)
        """
        user_id = request.user_id
        profile = self.loader.profiles[user_id]
        start_date = datetime.strptime(request.request_date, "%Y-%m-%d")
        min_balance_to_keep = profile.minimum_balance_to_keep

        stopped_event_ids: Set[str] = set()
        reduced_amounts: Dict[str, float] = {}
        if spending_changes:
            for action, eid, val in spending_changes:
                if action == "stop":
                    stopped_event_ids.add(eid)
                elif action == "reduce_to" and val is not None:
                    reduced_amounts[eid] = val

        events = self.loader.events_by_user.get(user_id, [])
        end_date = start_date + timedelta(days=FORECAST_HORIZON_DAYS)

        daily_credits = [0.0] * (FORECAST_HORIZON_DAYS + 1)
        daily_debits = [0.0] * (FORECAST_HORIZON_DAYS + 1)

        salary_ended = False
        salary_adjustment_amount: Optional[float] = None
        salary_date_shift: Optional[str] = None
        rent_increase_percent: Optional[float] = None

        messages = self.loader.messages_by_user.get(user_id, []) + self.loader.messages_by_request.get(request.request_id, [])
        for msg in messages:
            fact = self.extractor.parse_message_financial_fact(msg)
            if fact["type"] == "income_ended":
                salary_ended = True
            elif fact["type"] == "salary_adjustment" and fact["amount"]:
                salary_adjustment_amount = float(fact["amount"])
            elif fact["type"] == "salary_date_shift" and fact["date"]:
                salary_date_shift = fact["date"]
            elif fact["type"] == "rent_increase" and fact["percent"]:
                rent_increase_percent = float(fact["percent"])

        explicit_debit_dates: Set[Tuple[str, str]] = set()
        explicit_salary_dates: Set[str] = set()

        # 1. Place all explicit events
        for ev in events:
            if ev.settlement_date < request.request_date or ev.settlement_date > end_date.strftime("%Y-%m-%d"):
                continue

            dt = datetime.strptime(ev.settlement_date, "%Y-%m-%d")
            day_idx = (dt - start_date).days

            if ev.status in ("failed", "cancelled", "unrealized") or ev.direction == "non_cash":
                continue

            if ev.status == "pending" and ev.direction == "credit":
                continue

            amt = self._resolve_event_amount(ev)

            if ev.event_id in stopped_event_ids:
                amt = 0.0
            elif ev.event_id in reduced_amounts:
                amt = reduced_amounts[ev.event_id]

            if ev.direction == "credit":
                if ev.category == "salary":
                    if salary_adjustment_amount is not None:
                        amt = salary_adjustment_amount
                    if salary_date_shift is not None and salary_date_shift >= request.request_date:
                        shift_dt = datetime.strptime(salary_date_shift, "%Y-%m-%d")
                        if 0 <= (shift_dt - start_date).days <= FORECAST_HORIZON_DAYS:
                            day_idx = (shift_dt - start_date).days
                    explicit_salary_dates.add(dt.strftime("%Y-%m-%d"))

                daily_credits[day_idx] += amt
            elif ev.direction == "debit":
                if ev.category == "rent" and rent_increase_percent is not None:
                    amt *= (1.0 + rent_increase_percent / 100.0)
                daily_debits[day_idx] += amt
                explicit_debit_dates.add((ev.settlement_date, ev.category))

        # 2. Project recurring streams
        streams = self.detect_recurring_streams(user_id, request.request_date)

        for stream in streams:
            amt = stream.amount
            if stream.last_event_id in stopped_event_ids:
                amt = 0.0
            elif stream.last_event_id in reduced_amounts:
                amt = reduced_amounts[stream.last_event_id]

            if stream.category == "rent" and rent_increase_percent is not None:
                amt *= (1.0 + rent_increase_percent / 100.0)

            if amt <= 0:
                continue

            if stream.frequency == "monthly":
                curr_dt = start_date
                while curr_dt <= end_date:
                    max_d = monthrange(curr_dt.year, curr_dt.month)[1]
                    target_day = min(stream.day_of_month, max_d)
                    try:
                        occ_dt = datetime(curr_dt.year, curr_dt.month, target_day)
                    except ValueError:
                        occ_dt = datetime(curr_dt.year, curr_dt.month, max_d)

                    if start_date <= occ_dt <= end_date:
                        d_str = occ_dt.strftime("%Y-%m-%d")
                        if (d_str, stream.category) not in explicit_debit_dates:
                            d_idx = (occ_dt - start_date).days
                            daily_debits[d_idx] += amt

                    if curr_dt.month == 12:
                        curr_dt = datetime(curr_dt.year + 1, 1, 1)
                    else:
                        curr_dt = datetime(curr_dt.year, curr_dt.month + 1, 1)

            elif stream.frequency == "weekly":
                last_dt = datetime.strptime(stream.last_settlement_date, "%Y-%m-%d")
                nxt = last_dt + timedelta(days=7)
                while nxt <= end_date:
                    if nxt >= start_date:
                        d_idx = (nxt - start_date).days
                        d_str = nxt.strftime("%Y-%m-%d")
                        if (d_str, stream.category) not in explicit_debit_dates:
                            daily_debits[d_idx] += amt
                    nxt += timedelta(days=7)

            elif stream.frequency == "biweekly":
                last_dt = datetime.strptime(stream.last_settlement_date, "%Y-%m-%d")
                nxt = last_dt + timedelta(days=14)
                while nxt <= end_date:
                    if nxt >= start_date:
                        d_idx = (nxt - start_date).days
                        d_str = nxt.strftime("%Y-%m-%d")
                        if (d_str, stream.category) not in explicit_debit_dates:
                            daily_debits[d_idx] += amt
                    nxt += timedelta(days=14)

        # 3. Project recurring monthly salary using mode day of regular payroll
        salary_events = [
            e for e in events
            if e.category == "salary"
            and e.direction == "credit"
            and e.status in ("settled", "scheduled")
        ]
        if salary_events:
            regular_salaries = [
                e for e in salary_events
                if "arrears" not in e.description.lower()
                and "prorated" not in e.description.lower()
            ]
            target_salaries = regular_salaries if regular_salaries else salary_events
            target_salaries.sort(key=lambda x: x.settlement_date)
            last_salary = target_salaries[-1]
            if "final" in last_salary.description.lower():
                salary_ended = True

            if not salary_ended:
                if salary_date_shift is not None:
                    sal_day = datetime.strptime(salary_date_shift, "%Y-%m-%d").day
                else:
                    days = [int(e.settlement_date[8:10]) for e in target_salaries]
                    sal_day = Counter(days).most_common(1)[0][0]

                # Most recent regular salary amount
                sal_amt = self._resolve_event_amount(target_salaries[-1])
                if salary_adjustment_amount is not None:
                    sal_amt = salary_adjustment_amount

                curr_dt = start_date
                while curr_dt <= end_date:
                    max_d = monthrange(curr_dt.year, curr_dt.month)[1]
                    target_d = min(sal_day, max_d)
                    occ_dt = datetime(curr_dt.year, curr_dt.month, target_d)
                    d_str = occ_dt.strftime("%Y-%m-%d")
                    if start_date <= occ_dt <= end_date:
                        if d_str not in explicit_salary_dates:
                            d_idx = (occ_dt - start_date).days
                            daily_credits[d_idx] += sal_amt
                            explicit_salary_dates.add(d_str)

                    if curr_dt.month == 12:
                        curr_dt = datetime(curr_dt.year + 1, 1, 1)
                    else:
                        curr_dt = datetime(curr_dt.year, curr_dt.month + 1, 1)

        # 4. Add additional payment plan
        if additional_payment_plan:
            for p_date, p_amt in additional_payment_plan:
                p_dt = datetime.strptime(p_date, "%Y-%m-%d")
                if start_date <= p_dt <= end_date:
                    p_idx = (p_dt - start_date).days
                    daily_debits[p_idx] += p_amt

        # 5. Run daily simulation
        current_bal = profile.current_available_balance
        daily_balances = []
        min_balance_reached = current_bal

        for day in range(0, FORECAST_HORIZON_DAYS + 1):
            current_bal = current_bal + daily_credits[day] - daily_debits[day]
            daily_balances.append(current_bal)
            if current_bal < min_balance_reached:
                min_balance_reached = current_bal

        min_headroom = min_balance_reached - min_balance_to_keep
        return daily_balances, min_balance_reached, min_headroom
