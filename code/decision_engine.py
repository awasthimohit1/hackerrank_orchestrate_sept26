"""Decision and plan generation engine for Buy or Wait? financial agent."""

from calendar import monthrange
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

from code.data_loader import DataLoader
from code.evidence_extractor import EvidenceExtractor
from code.forecaster import CashflowForecaster
from code.models import DecisionResult, EvaluationRequest, PaymentOption


def format_amount(amt: float) -> str:
    """Formats amount as integer if whole number, or with 2 decimal places."""
    if abs(amt - round(amt)) < 1e-4:
        return str(int(round(amt)))
    return f"{amt:.2f}"


@dataclass
class CandidatePlan:
    affordability_status: str
    recommended_payment_method: str
    payment_plan: str
    earliest_date_for_full_payment: str
    spending_changes_needed: str
    completes_by_deadline: bool
    spending_changes_count: int
    total_amount_paid: float
    first_payment_date: str
    num_payments: int
    payment_option_id: str
    explanation: str


class DecisionEngine:
    """Evaluates requests, explores candidate payment plans, and ranks them lexicographically."""

    def __init__(self, data_loader: DataLoader, forecaster: CashflowForecaster):
        self.loader = data_loader
        self.forecaster = forecaster

    def compute_earliest_date_for_full_payment(
        self, request: EvaluationRequest, amount_safe_to_pay: float
    ) -> str:
        """Finds the earliest date when paying requested_amount in full is safe."""
        if abs(amount_safe_to_pay - request.requested_amount) < 0.05:
            return request.request_date

        start_dt = datetime.strptime(request.request_date, "%Y-%m-%d")
        events = self.loader.events_by_user.get(request.user_id, [])

        # Find mode salary day
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
            days = [int(e.settlement_date[8:10]) for e in target_salaries]
            sal_day = Counter(days).most_common(1)[0][0]
        else:
            sal_day = 15

        # Check messages for salary date shifts
        messages = self.loader.messages_by_user.get(request.user_id, []) + self.loader.messages_by_request.get(request.request_id, [])
        shifted_date = None
        for msg in messages:
            fact = self.forecaster.extractor.parse_message_financial_fact(msg)
            if fact["type"] == "salary_date_shift" and fact["date"]:
                shifted_date = fact["date"]

        if shifted_date:
            sal_day = datetime.strptime(shifted_date, "%Y-%m-%d").day

        sal_dates = set()
        if shifted_date and shifted_date >= request.request_date:
            sal_dates.add(shifted_date)

        # Generate candidate salary dates for up to 4 months
        curr_dt = start_dt
        for _ in range(4):
            max_d = monthrange(curr_dt.year, curr_dt.month)[1]
            target_d = min(sal_day, max_d)
            cand_dt = datetime(curr_dt.year, curr_dt.month, target_d)
            cand_str = cand_dt.strftime("%Y-%m-%d")
            if cand_str >= request.request_date:
                sal_dates.add(cand_str)

            if curr_dt.month == 12:
                curr_dt = datetime(curr_dt.year + 1, 1, 1)
            else:
                curr_dt = datetime(curr_dt.year, curr_dt.month + 1, 1)

        sorted_dates = sorted(list(sal_dates))

        # Check if salary ended
        last_sal = target_salaries[-1]
        if "final" in last_sal.description.lower():
            return ""
        for msg in messages:
            if "ended" in msg.message_text.lower():
                return ""

        balances, _, _ = self.forecaster.simulate_90_days(request)
        min_b = self.loader.profiles[request.user_id].minimum_balance_to_keep

        # Test each salary date
        for i, d_str in enumerate(sorted_dates):
            dt = datetime.strptime(d_str, "%Y-%m-%d")
            d_idx = (dt - start_dt).days
            if d_idx > 90:
                break

            # Check window: if next paycheck is outside 90 days, check first 16 days after payment
            if i + 1 < len(sorted_dates):
                next_dt = datetime.strptime(sorted_dates[i + 1], "%Y-%m-%d")
                next_idx = (next_dt - start_dt).days
                if next_idx > 90:
                    next_idx = min(len(balances), d_idx + 16)
            else:
                next_idx = min(len(balances), d_idx + 16)

            sub = [balances[k] - request.requested_amount for k in range(d_idx, next_idx)]
            if min(sub) >= min_b - 0.01:
                return d_str

        return ""

    def evaluate_request(self, request: EvaluationRequest) -> DecisionResult:
        """Evaluates a single request and returns the winning DecisionResult."""
        user = self.loader.profiles[request.user_id]
        curr = user.home_currency
        req_amt = request.requested_amount

        # 1. Base simulation without changes
        balances, min_bal, headroom = self.forecaster.simulate_90_days(request=request)
        safe_amt = max(0.0, min(req_amt, headroom))

        # 2. Earliest date for full payment
        earliest_date = self.compute_earliest_date_for_full_payment(request, safe_amt)

        # 3. Generate candidate plans
        candidates: List[CandidatePlan] = []

        # Candidate A: Full Payment Today (No spending changes)
        if "full_payment" in user.payment_methods_user_will_consider:
            if abs(safe_amt - req_amt) < 0.05:
                p_plan = f"{request.request_date}:{format_amount(req_amt)}"
                explanation = (
                    f"Pay {curr} {format_amount(req_amt)} today. "
                    f"This leaves at least {curr} {format_amount(user.minimum_balance_to_keep)} available over the next 90 days."
                )
                candidates.append(
                    CandidatePlan(
                        affordability_status="affordable_now",
                        recommended_payment_method="full_payment",
                        payment_plan=p_plan,
                        earliest_date_for_full_payment=request.request_date,
                        spending_changes_needed="none",
                        completes_by_deadline=True,
                        spending_changes_count=0,
                        total_amount_paid=req_amt,
                        first_payment_date=request.request_date,
                        num_payments=1,
                        payment_option_id="",
                        explanation=explanation,
                    )
                )

        # Candidate B: Installment Options
        if "installments" in user.payment_methods_user_will_consider:
            options = self.loader.payment_options_by_request.get(request.request_id, [])
            for opt in options:
                if opt.payment_method != "installments":
                    continue

                if user.max_installment_months is not None and opt.number_of_payments > user.max_installment_months:
                    continue

                start_dt = datetime.strptime(opt.first_payment_date, "%Y-%m-%d")
                freq = opt.payment_frequency_days or 30
                inst_plan = []
                plan_strs = []
                for i in range(opt.number_of_payments):
                    p_dt = start_dt + timedelta(days=i * freq)
                    d_str = p_dt.strftime("%Y-%m-%d")
                    inst_plan.append((d_str, opt.payment_amount))
                    plan_strs.append(f"{d_str}:{format_amount(opt.payment_amount)}")

                last_payment_date = inst_plan[-1][0]
                completes_by_dl = last_payment_date <= request.desired_completion_date

                balances_inst, min_b_inst, headroom_inst = self.forecaster.simulate_90_days(
                    request=request, additional_payment_plan=inst_plan
                )

                if headroom_inst >= -0.01:
                    first_dt_obj = datetime.strptime(opt.first_payment_date, "%Y-%m-%d")
                    month_names = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
                    formatted_start = f"{first_dt_obj.day} {month_names[first_dt_obj.month - 1]} {first_dt_obj.year}"
                    explanation = (
                        f"Use {opt.number_of_payments} installments of {curr} {format_amount(opt.payment_amount)}, "
                        f"starting {formatted_start}. This leaves at least {curr} {format_amount(user.minimum_balance_to_keep)} available."
                    )
                    candidates.append(
                        CandidatePlan(
                            affordability_status="affordable_with_plan",
                            recommended_payment_method="installments",
                            payment_plan="|".join(plan_strs),
                            earliest_date_for_full_payment=earliest_date,
                            spending_changes_needed="none",
                            completes_by_deadline=completes_by_dl,
                            spending_changes_count=0,
                            total_amount_paid=opt.total_payable_amount or (opt.payment_amount * opt.number_of_payments),
                            first_payment_date=opt.first_payment_date,
                            num_payments=opt.number_of_payments,
                            payment_option_id=opt.payment_option_id,
                            explanation=explanation,
                        )
                    )

        # Candidate C: Partial Payment
        if (
            request.allows_partial_payment
            and "partial_payment" in user.payment_methods_user_will_consider
            and 0 < safe_amt < req_amt
            and earliest_date != ""
            and earliest_date <= request.desired_completion_date
        ):
            rem_amt = req_amt - safe_amt
            plan_str = f"{request.request_date}:{format_amount(safe_amt)}|{earliest_date}:{format_amount(rem_amt)}"
            earliest_dt = datetime.strptime(earliest_date, "%Y-%m-%d")
            month_names = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
            formatted_earliest = f"{earliest_dt.day} {month_names[earliest_dt.month - 1]} {earliest_dt.year}"
            explanation = (
                f"Pay {curr} {format_amount(safe_amt)} today and the remaining {curr} {format_amount(rem_amt)} on {formatted_earliest}. "
                f"This completes the full request and keeps the {curr} {format_amount(user.minimum_balance_to_keep)} minimum protected."
            )
            candidates.append(
                CandidatePlan(
                    affordability_status="affordable_with_plan",
                    recommended_payment_method="partial_payment",
                    payment_plan=plan_str,
                    earliest_date_for_full_payment=earliest_date,
                    spending_changes_needed="none",
                    completes_by_deadline=True,
                    spending_changes_count=0,
                    total_amount_paid=req_amt,
                    first_payment_date=request.request_date,
                    num_payments=2,
                    payment_option_id="",
                    explanation=explanation,
                )
            )

        # Candidate D: Wait
        if "full_payment" in user.payment_methods_user_will_consider and earliest_date != "":
            completes_by_dl = earliest_date <= request.desired_completion_date
            earliest_dt = datetime.strptime(earliest_date, "%Y-%m-%d")
            month_names = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
            formatted_earliest = f"{earliest_dt.day} {month_names[earliest_dt.month - 1]} {earliest_dt.year}"
            explanation = (
                f"Pay {curr} {format_amount(req_amt)} in full on {formatted_earliest}. "
                f"Paying earlier would take the balance below the {curr} {format_amount(user.minimum_balance_to_keep)} minimum."
            )
            candidates.append(
                CandidatePlan(
                    affordability_status="affordable_later",
                    recommended_payment_method="wait",
                    payment_plan=f"{earliest_date}:{format_amount(req_amt)}",
                    earliest_date_for_full_payment=earliest_date,
                    spending_changes_needed="none",
                    completes_by_deadline=completes_by_dl,
                    spending_changes_count=0,
                    total_amount_paid=req_amt,
                    first_payment_date=earliest_date,
                    num_payments=1,
                    payment_option_id="",
                    explanation=explanation,
                )
            )

        # Candidate E: Spending Changes (only if needed)
        # Only evaluate spending changes if no 0-change candidate completes by deadline
        has_zero_change_success = any(c.completes_by_deadline and c.spending_changes_count == 0 for c in candidates)
        if not has_zero_change_success and "full_payment" in user.payment_methods_user_will_consider and safe_amt < req_amt:
            streams = self.forecaster.detect_recurring_streams(request.user_id, request.request_date)
            stoppable = [
                s for s in streams
                if s.category in user.expense_categories_user_is_willing_to_stop
                and s.flexibility in ("stoppable", "reducible_or_stoppable")
            ]
            reducible = [
                s for s in streams
                if s.category in user.expense_categories_user_is_willing_to_reduce
                and s.flexibility in ("reducible", "reducible_or_stoppable")
                and s.minimum_allowed_amount is not None
                and s.minimum_allowed_amount < s.amount
            ]

            from itertools import combinations
            change_options = []
            for s in stoppable:
                change_options.append(("stop", s.last_event_id, None, s.description, s.amount))
            for s in reducible:
                change_options.append(("reduce_to", s.last_event_id, s.minimum_allowed_amount, s.description, s.amount - s.minimum_allowed_amount))

            best_change_plan = None
            for k in [1, 2, 3]:
                if best_change_plan:
                    break
                for combo in combinations(change_options, k):
                    eids = set(c[1] for c in combo)
                    if len(eids) < len(combo):
                        continue

                    changes = [(c[0], c[1], c[2]) for c in combo]
                    plan = [(request.request_date, req_amt)]
                    b_ch, min_b_ch, headroom_ch = self.forecaster.simulate_90_days(
                        request=request, spending_changes=changes, additional_payment_plan=plan
                    )
                    if headroom_ch >= -50.0:
                        best_change_plan = combo
                        break

            if best_change_plan:
                changes_str = "|".join(
                    f"stop:{c[1]}" if c[0] == "stop" else f"reduce_to:{c[1]}:{format_amount(c[2])}"
                    for c in best_change_plan
                )
                desc_parts = []
                for c in best_change_plan:
                    if c[0] == "stop":
                        desc_parts.append(f"Stop the {c[3].lower()}")
                    else:
                        desc_parts.append(f"reduce the {c[3].lower()} to {curr} {format_amount(c[2])}")
                change_desc = " and ".join(desc_parts)

                explanation = (
                    f"{change_desc}, then pay {curr} {format_amount(req_amt)} today. "
                    f"This leaves at least {curr} {format_amount(user.minimum_balance_to_keep)} available."
                )
                candidates.append(
                    CandidatePlan(
                        affordability_status="affordable_with_plan",
                        recommended_payment_method="full_payment",
                        payment_plan=f"{request.request_date}:{format_amount(req_amt)}",
                        earliest_date_for_full_payment=earliest_date,
                        spending_changes_needed=changes_str,
                        completes_by_deadline=True,
                        spending_changes_count=len(best_change_plan),
                        total_amount_paid=req_amt,
                        first_payment_date=request.request_date,
                        num_payments=1,
                        payment_option_id="",
                        explanation=explanation,
                    )
                )

        # 4. Filter and Rank candidates
        valid_candidates = [c for c in candidates if c.completes_by_deadline]
        if valid_candidates:
            def sort_key(c: CandidatePlan):
                return (
                    c.spending_changes_count,          # Criterion 1: 0 before 1 before 2
                    c.total_amount_paid,               # Criterion 2: Lower total amount
                    c.first_payment_date,              # Criterion 3: Earlier start date
                    c.num_payments,                    # Criterion 4: Fewer payments
                    c.payment_option_id or "zzzzz",    # Criterion 5: Lowest option ID
                )

            valid_candidates.sort(key=sort_key)
            best = valid_candidates[0]

            return DecisionResult(
                request_id=request.request_id,
                amount_safe_to_pay=round(safe_amt, 2),
                affordability_status=best.affordability_status,
                recommended_payment_method=best.recommended_payment_method,
                payment_plan=best.payment_plan,
                earliest_date_for_full_payment=best.earliest_date_for_full_payment,
                spending_changes_needed=best.spending_changes_needed,
                decision_explanation=best.explanation,
            )

        # 5. Fallback: Not Affordable / Not Recommended
        dl_dt = datetime.strptime(request.desired_completion_date, "%Y-%m-%d")
        month_names = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        formatted_dl = f"{dl_dt.day} {month_names[dl_dt.month - 1]} {dl_dt.year}"

        if safe_amt > 0 and earliest_date == "":
            explanation = (
                f"Do not proceed with the {curr} {format_amount(req_amt)} request. "
                f"Although {curr} {format_amount(safe_amt)} is available today, the full amount cannot be completed safely within 90 days."
            )
        else:
            explanation = (
                f"Do not make this payment by {formatted_dl}. "
                f"None of the available options keeps the {curr} {format_amount(user.minimum_balance_to_keep)} minimum protected."
            )

        return DecisionResult(
            request_id=request.request_id,
            amount_safe_to_pay=round(safe_amt, 2),
            affordability_status="not_affordable",
            recommended_payment_method="not_recommended",
            payment_plan="none",
            earliest_date_for_full_payment="",
            spending_changes_needed="none",
            decision_explanation=explanation,
        )
