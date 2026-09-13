"""Evaluation script benchmarking Buy or Wait? agent against sample_requests.csv."""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.getcwd())

from code.data_loader import DataLoader
from code.decision_engine import DecisionEngine
from code.evidence_extractor import EvidenceExtractor
from code.forecaster import CashflowForecaster


def run_benchmark():
    loader = DataLoader()
    extractor = EvidenceExtractor()
    forecaster = CashflowForecaster(loader, extractor)
    engine = DecisionEngine(loader, forecaster)

    samples = loader.load_sample_requests()
    print(f"\n=======================================================")
    print(f"   HACKERRANK ORCHESTRATE: BENCHMARK EVALUATION")
    print(f"=======================================================")
    print(f"Total Ground-Truth Samples: {len(samples)}\n")

    correct_safe = 0
    correct_status = 0
    correct_method = 0
    correct_plan = 0
    correct_earliest = 0
    correct_changes = 0
    perfect_matches = 0

    diff_reports = []

    for req, gt in samples:
        pred = engine.evaluate_request(req)

        # Tolerance 0.05 for amount_safe_to_pay
        match_safe = abs(pred.amount_safe_to_pay - gt.amount_safe_to_pay) < 0.05
        match_status = pred.affordability_status == gt.affordability_status
        match_method = pred.recommended_payment_method == gt.recommended_payment_method
        match_plan = pred.payment_plan == gt.payment_plan
        match_earliest = pred.earliest_date_for_full_payment == gt.earliest_date_for_full_payment
        match_changes = pred.spending_changes_needed == gt.spending_changes_needed

        if match_safe:
            correct_safe += 1
        if match_status:
            correct_status += 1
        if match_method:
            correct_method += 1
        if match_plan:
            correct_plan += 1
        if match_earliest:
            correct_earliest += 1
        if match_changes:
            correct_changes += 1

        is_perfect = match_status and match_method and match_plan and match_earliest and match_changes
        if is_perfect:
            perfect_matches += 1
        else:
            diff_reports.append((req, gt, pred))

    total = len(samples)
    print("-------------------------------------------------------")
    print(f"{'Dimension':<32} | {'Matches':<8} | {'Accuracy':<8}")
    print("-------------------------------------------------------")
    print(f"{'amount_safe_to_pay':<32} | {correct_safe}/{total:<6} | {correct_safe/total*100:6.1f}%")
    print(f"{'affordability_status':<32} | {correct_status}/{total:<6} | {correct_status/total*100:6.1f}%")
    print(f"{'recommended_payment_method':<32} | {correct_method}/{total:<6} | {correct_method/total*100:6.1f}%")
    print(f"{'payment_plan':<32} | {correct_plan}/{total:<6} | {correct_plan/total*100:6.1f}%")
    print(f"{'earliest_date_for_full_payment':<32} | {correct_earliest}/{total:<6} | {correct_earliest/total*100:6.1f}%")
    print(f"{'spending_changes_needed':<32} | {correct_changes}/{total:<6} | {correct_changes/total*100:6.1f}%")
    print("-------------------------------------------------------")
    print(f"{'OVERALL PERFECT DECISIONS':<32} | {perfect_matches}/{total:<6} | {perfect_matches/total*100:6.1f}%")
    print("=======================================================\n")

    if diff_reports:
        print(f"DIAGNOSTIC TRACE FOR {len(diff_reports)} MISMATCHES:\n")
        for req, gt, pred in diff_reports[:10]:
            print(f"[{req.request_id}] User={req.user_id} Date={req.request_date} Amt={req.requested_amount}")
            print(f"  Status:   GT={gt.affordability_status:<20} Pred={pred.affordability_status}")
            print(f"  Method:   GT={gt.recommended_payment_method:<20} Pred={pred.recommended_payment_method}")
            print(f"  Earliest: GT={gt.earliest_date_for_full_payment:<20} Pred={pred.earliest_date_for_full_payment}")
            print(f"  Changes:  GT={gt.spending_changes_needed:<20} Pred={pred.spending_changes_needed}")
            print(f"  Plan GT:   {gt.payment_plan}")
            print(f"  Plan Pred: {pred.payment_plan}")
            print()


if __name__ == "__main__":
    run_benchmark()
