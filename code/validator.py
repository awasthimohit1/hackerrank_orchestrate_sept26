"""Property-based invariant validation engine for Buy or Wait? agent.

Independently checks prediction outputs against all 14 domain rules and financial invariants.
"""

import csv
import os
import sys
from typing import Dict, List, Optional, Tuple

# Ensure repository root is on sys.path
sys.path.insert(0, os.getcwd())

from code.config import (
    AFFORDABILITY_STATUSES,
    OUTPUT_COLUMNS,
    RECOMMENDED_PAYMENT_METHODS,
)
from code.data_loader import DataLoader
from code.models import DecisionResult, EvaluationRequest, FinancialProfile, PaymentOption


class InvariantValidator:
    """Validates predictions against domain invariants, bounds, and consistency rules."""

    def __init__(self, data_loader: Optional[DataLoader] = None):
        self.loader = data_loader or DataLoader()

    def validate_single_result(
        self,
        result: DecisionResult,
        request: EvaluationRequest,
        profile: FinancialProfile,
        options: List[PaymentOption],
    ) -> List[str]:
        """Validates all 14 property-based invariants for a single prediction row."""
        errors: List[str] = []
        req_id = request.request_id

        # 1. Safe Amount Bounds: 0 <= amount_safe_to_pay <= requested_amount
        if result.amount_safe_to_pay < -1e-4:
            errors.append(f"[{req_id}] amount_safe_to_pay ({result.amount_safe_to_pay}) is negative")
        if result.amount_safe_to_pay > request.requested_amount + 0.05:
            errors.append(
                f"[{req_id}] amount_safe_to_pay ({result.amount_safe_to_pay}) exceeds requested_amount ({request.requested_amount})"
            )

        # 2. Categorical Enums
        if result.affordability_status not in AFFORDABILITY_STATUSES:
            errors.append(f"[{req_id}] Invalid affordability_status: '{result.affordability_status}'")
        if result.recommended_payment_method not in RECOMMENDED_PAYMENT_METHODS:
            errors.append(f"[{req_id}] Invalid recommended_payment_method: '{result.recommended_payment_method}'")

        # 3. User Payment Preferences
        method = result.recommended_payment_method
        if method == "full_payment":
            if "full_payment" not in profile.payment_methods_user_will_consider:
                errors.append(f"[{req_id}] Recommended full_payment, but user does not consider full_payment")
        elif method == "installments":
            if "installments" not in profile.payment_methods_user_will_consider:
                errors.append(f"[{req_id}] Recommended installments, but user does not consider installments")
        elif method == "partial_payment":
            if "partial_payment" not in profile.payment_methods_user_will_consider:
                errors.append(f"[{req_id}] Recommended partial_payment, but user does not consider partial_payment")
            if not request.allows_partial_payment:
                errors.append(f"[{req_id}] Recommended partial_payment, but request.allows_partial_payment is False")
        elif method == "wait":
            if "full_payment" not in profile.payment_methods_user_will_consider:
                errors.append(f"[{req_id}] Recommended wait, but user does not consider full_payment")

        # 4. Invariant: affordable_now
        if result.affordability_status == "affordable_now":
            if method != "full_payment":
                errors.append(f"[{req_id}] affordable_now must use full_payment, got '{method}'")
            if result.earliest_date_for_full_payment != request.request_date:
                errors.append(
                    f"[{req_id}] affordable_now earliest_date must equal request_date ({request.request_date}), got '{result.earliest_date_for_full_payment}'"
                )
            if result.spending_changes_needed != "none":
                errors.append(f"[{req_id}] affordable_now cannot have spending changes, got '{result.spending_changes_needed}'")

        # 5. Invariant: affordable_later
        elif result.affordability_status == "affordable_later":
            if method != "wait":
                errors.append(f"[{req_id}] affordable_later must use wait, got '{method}'")
            if not result.earliest_date_for_full_payment:
                errors.append(f"[{req_id}] affordable_later must specify earliest_date_for_full_payment")
            elif result.earliest_date_for_full_payment < request.request_date:
                errors.append(f"[{req_id}] earliest_date ({result.earliest_date_for_full_payment}) is before request_date ({request.request_date})")
            if result.spending_changes_needed != "none":
                errors.append(f"[{req_id}] affordable_later cannot require spending changes, got '{result.spending_changes_needed}'")

        # 6. Invariant: not_affordable
        elif result.affordability_status == "not_affordable":
            if method != "not_recommended":
                errors.append(f"[{req_id}] not_affordable must use not_recommended, got '{method}'")
            if result.payment_plan != "none":
                errors.append(f"[{req_id}] not_affordable must have payment_plan 'none', got '{result.payment_plan}'")
            if result.earliest_date_for_full_payment != "":
                errors.append(f"[{req_id}] not_affordable must have empty earliest_date, got '{result.earliest_date_for_full_payment}'")
            if result.spending_changes_needed != "none":
                errors.append(f"[{req_id}] not_affordable must have spending_changes_needed 'none', got '{result.spending_changes_needed}'")

        # 7. Invariant: affordable_with_plan
        elif result.affordability_status == "affordable_with_plan":
            if method not in ("full_payment", "partial_payment", "installments"):
                errors.append(f"[{req_id}] affordable_with_plan must use full_payment, partial_payment, or installments, got '{method}'")

        # 8. Partial Payment Invariants
        if method == "partial_payment":
            if not (0 < result.amount_safe_to_pay < request.requested_amount):
                errors.append(
                    f"[{req_id}] partial_payment requires 0 < safe ({result.amount_safe_to_pay}) < requested ({request.requested_amount})"
                )
            if not result.earliest_date_for_full_payment:
                errors.append(f"[{req_id}] partial_payment requires earliest_date_for_full_payment")
            elif result.earliest_date_for_full_payment > request.desired_completion_date:
                errors.append(
                    f"[{req_id}] partial_payment 2nd payment ({result.earliest_date_for_full_payment}) exceeds deadline ({request.desired_completion_date})"
                )

            # Plan must have exactly two payments
            parts = result.payment_plan.split("|")
            if len(parts) != 2:
                errors.append(f"[{req_id}] partial_payment must have exactly 2 installments, got {len(parts)} ({result.payment_plan})")
            else:
                try:
                    d1, a1_str = parts[0].split(":")
                    d2, a2_str = parts[1].split(":")
                    a1 = float(a1_str)
                    a2 = float(a2_str)
                    if d1 != request.request_date:
                        errors.append(f"[{req_id}] partial_payment payment 1 date ({d1}) != request_date ({request.request_date})")
                    if d2 != result.earliest_date_for_full_payment:
                        errors.append(f"[{req_id}] partial_payment payment 2 date ({d2}) != earliest_date ({result.earliest_date_for_full_payment})")
                    if abs((a1 + a2) - request.requested_amount) > 0.05:
                        errors.append(f"[{req_id}] partial_payment sum ({a1 + a2}) != requested_amount ({request.requested_amount})")
                except Exception as e:
                    errors.append(f"[{req_id}] Malformed partial_payment plan: {result.payment_plan} ({e})")

        # 9. Installment Plan Invariants
        if method == "installments":
            matching_option = None
            for opt in options:
                if opt.payment_method == "installments":
                    plan_parts = result.payment_plan.split("|")
                    if len(plan_parts) == opt.number_of_payments:
                        first_p_date = plan_parts[0].split(":")[0]
                        if first_p_date == opt.first_payment_date:
                            matching_option = opt
                            break

            if not matching_option:
                errors.append(f"[{req_id}] Installment plan '{result.payment_plan}' does not match any provider option in request_payment_options.csv")
            else:
                if profile.max_installment_months and matching_option.number_of_payments > profile.max_installment_months:
                    errors.append(
                        f"[{req_id}] Installment option exceeds user max_installment_months ({matching_option.number_of_payments} > {profile.max_installment_months})"
                    )
                # Check deadline
                last_p_date = result.payment_plan.split("|")[-1].split(":")[0]
                if last_p_date > request.desired_completion_date:
                    errors.append(
                        f"[{req_id}] Installment plan final date ({last_p_date}) exceeds desired_completion_date ({request.desired_completion_date})"
                    )

        # 10. Chronological Ordering of Payment Plan
        if result.payment_plan != "none":
            dates = []
            for part in result.payment_plan.split("|"):
                if ":" in part:
                    dates.append(part.split(":")[0])
            for i in range(1, len(dates)):
                if dates[i] < dates[i - 1]:
                    errors.append(f"[{req_id}] Payment plan is not in chronological order: {dates[i-1]} followed by {dates[i]}")

        # 11. Spending Changes Invariants
        if result.spending_changes_needed != "none":
            changes = result.spending_changes_needed.split("|")
            if len(changes) > 3:
                errors.append(f"[{req_id}] spending_changes_needed has {len(changes)} actions (maximum permitted is 3)")

            used_eids = set()
            user_events = {e.event_id: e for e in self.loader.events_by_user.get(request.user_id, [])}

            for ch in changes:
                parts = ch.split(":")
                action = parts[0]
                if action not in ("stop", "reduce_to"):
                    errors.append(f"[{req_id}] Invalid spending change action: '{action}'")
                    continue

                eid = parts[1]
                if eid in used_eids:
                    errors.append(f"[{req_id}] Duplicate event_id '{eid}' modified multiple times in spending changes")
                used_eids.add(eid)

                ev = user_events.get(eid)
                if not ev:
                    errors.append(f"[{req_id}] Spending change targets unknown event_id '{eid}' for user '{request.user_id}'")
                    continue

                cat = ev.category
                if cat in profile.expense_categories_to_protect:
                    errors.append(f"[{req_id}] Target event '{eid}' is in protected category '{cat}'")

                if action == "stop":
                    if cat not in profile.expense_categories_user_is_willing_to_stop:
                        errors.append(f"[{req_id}] User not willing to stop category '{cat}' for event '{eid}'")
                    if ev.flexibility not in ("stoppable", "reducible_or_stoppable"):
                        errors.append(f"[{req_id}] Event '{eid}' flexibility is '{ev.flexibility}' (not stoppable)")

                elif action == "reduce_to":
                    if cat not in profile.expense_categories_user_is_willing_to_reduce:
                        errors.append(f"[{req_id}] User not willing to reduce category '{cat}' for event '{eid}'")
                    if ev.flexibility not in ("reducible", "reducible_or_stoppable"):
                        errors.append(f"[{req_id}] Event '{eid}' flexibility is '{ev.flexibility}' (not reducible)")
                    if len(parts) < 3:
                        errors.append(f"[{req_id}] reduce_to missing amount: '{ch}'")
                    else:
                        red_amt = float(parts[2])
                        if ev.minimum_allowed_amount is not None and red_amt < ev.minimum_allowed_amount - 0.01:
                            errors.append(
                                f"[{req_id}] reduce_to amount ({red_amt}) below minimum_allowed_amount ({ev.minimum_allowed_amount})"
                            )

        # 12. Decision Explanation Invariant
        if not result.decision_explanation or len(result.decision_explanation.strip()) < 10:
            errors.append(f"[{req_id}] decision_explanation is missing or too short")

        return errors

    def validate_csv_file(self, csv_path: str) -> Tuple[bool, List[str]]:
        """Validates a complete CSV file against schema and domain invariants."""
        if not os.path.exists(csv_path):
            return False, [f"File not found: {csv_path}"]

        errors: List[str] = []
        rows: List[Dict[str, str]] = []

        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return False, ["CSV file is completely empty"]

            # Schema header check
            if os.path.basename(csv_path) == "output.csv":
                if header != OUTPUT_COLUMNS:
                    errors.append(
                        f"output.csv header mismatch.\nExpected: {OUTPUT_COLUMNS}\nFound:    {header}"
                    )
            else:
                missing = [c for c in OUTPUT_COLUMNS if c not in header]
                if missing:
                    errors.append(f"CSV missing required columns: {missing}")

            dict_reader = csv.DictReader(f, fieldnames=header)
            for r in dict_reader:
                rows.append(r)

        requests_dict = {r.request_id: r for r in self.loader.load_requests()}
        for r, _ in self.loader.load_sample_requests():
            if r.request_id not in requests_dict:
                requests_dict[r.request_id] = r

        for row in rows:
            req_id = row.get("request_id", "")
            req = requests_dict.get(req_id)

            if not req:
                errors.append(f"Unknown request_id: '{req_id}'")
                continue

            profile = self.loader.profiles[req.user_id]
            options = self.loader.payment_options_by_request.get(req_id, [])

            try:
                res = DecisionResult(
                    request_id=req_id,
                    amount_safe_to_pay=float(row.get("amount_safe_to_pay", 0)),
                    affordability_status=row.get("affordability_status", ""),
                    recommended_payment_method=row.get("recommended_payment_method", ""),
                    payment_plan=row.get("payment_plan", ""),
                    earliest_date_for_full_payment=row.get("earliest_date_for_full_payment", ""),
                    spending_changes_needed=row.get("spending_changes_needed", ""),
                    decision_explanation=row.get("decision_explanation", ""),
                )
            except Exception as e:
                errors.append(f"[{req_id}] Parse error: {e}")
                continue

            row_errors = self.validate_single_result(res, req, profile, options)
            errors.extend(row_errors)

        is_valid = len(errors) == 0
        return is_valid, errors


def run_validator_cli():
    """CLI runner for validation."""
    if len(sys.argv) < 2:
        print("Usage: python code/validator.py <path_to_csv>")
        sys.exit(1)

    target_csv = sys.argv[1]
    validator = InvariantValidator()
    print(f"\n=======================================================")
    print(f"   BUY OR WAIT? INVARIANT VALIDATION AUDIT")
    print(f"=======================================================")
    print(f"Target File: {target_csv}\n")

    valid, errors = validator.validate_csv_file(target_csv)
    if valid:
        print(">>> ALL 14 INVARIANTS PASSED PERFECTLY! ZERO ERRORS.")
        print("=======================================================\n")
        sys.exit(0)
    else:
        print(f">>> VALIDATION FAILED with {len(errors)} invariant violations:")
        for err in errors[:20]:
            print(f"  * {err}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors.")
        print("=======================================================\n")
        sys.exit(1)


if __name__ == "__main__":
    run_validator_cli()
