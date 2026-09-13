"""Main CLI entrypoint for Buy or Wait? AI-powered financial decision agent.

Runs the complete financial decision pipeline across all requests in dataset/requests.csv
(or specified file), generates the verified output.csv, synthesizes grounded explanations,
audits token usage, and asserts all 14 property-based invariants.
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from code.config import (
    OUTPUT_COLUMNS,
    OUTPUT_CSV,
    REQUESTS_CSV,
    USAGE_REPORT_MD,
)
from code.data_loader import DataLoader
from code.decision_engine import DecisionEngine, format_amount
from code.evidence_extractor import EvidenceExtractor
from code.explanation_synthesizer import ExplanationSynthesizer
from code.forecaster import CashflowForecaster
from code.models import DecisionResult, EvaluationRequest
from code.validator import InvariantValidator


def format_safe_amount(amt: float) -> str:
    """Formats safe amount matching evaluation dataset convention."""
    r_amt = round(amt, 2)
    if abs(r_amt - round(r_amt)) < 1e-4:
        return str(int(round(r_amt)))
    return str(r_amt)


def run_pipeline(
    requests_path: Path = REQUESTS_CSV,
    output_path: Path = OUTPUT_CSV,
    api_key: str = "",
    run_samples: bool = False,
) -> int:
    """Executes the complete decision, explanation, and validation pipeline."""
    print("=======================================================")
    print("   BUY OR WAIT? AI-POWERED FINANCIAL DECISION AGENT   ")
    print("=======================================================")
    print(f"Target Requests File: {requests_path}")
    print(f"Output File:          {output_path}")

    # 1. Initialize data and components
    print("\n[1/5] Loading datasets and evidence...")
    loader = DataLoader()
    extractor = EvidenceExtractor()
    forecaster = CashflowForecaster(loader, extractor)
    engine = DecisionEngine(loader, forecaster)
    synthesizer = ExplanationSynthesizer(api_key=api_key)
    validator = InvariantValidator(loader)

    # 2. Load requests to evaluate
    requests_to_eval: List[EvaluationRequest] = []
    if run_samples:
        print("[2/5] Running in sample benchmark mode...")
        sample_pairs = loader.load_sample_requests()
        requests_to_eval = [pair[0] for pair in sample_pairs]
    else:
        print(f"[2/5] Loading evaluation requests from {requests_path}...")
        requests_to_eval = loader.load_requests(requests_path)

    print(f"Total requests to process: {len(requests_to_eval)}")

    # 3. Process requests
    print("\n[3/5] Evaluating financial decisions & generating plans...")
    results: List[DecisionResult] = []
    status_counts = {"affordable_now": 0, "affordable_with_plan": 0, "affordable_later": 0, "not_affordable": 0}
    method_counts = {"full_payment": 0, "partial_payment": 0, "installments": 0, "wait": 0, "not_recommended": 0}

    for idx, req in enumerate(requests_to_eval, 1):
        # Step A: Deterministic decision and optimal plan generation
        pred = engine.evaluate_request(req)

        # Step B: Grounded explanation synthesis (NVIDIA NIM or deterministic fallback)
        profile = loader.profiles[req.user_id]
        enriched_expl = synthesizer.synthesize_explanation(req, pred, profile)
        pred.decision_explanation = enriched_expl

        results.append(pred)
        status_counts[pred.affordability_status] = status_counts.get(pred.affordability_status, 0) + 1
        method_counts[pred.recommended_payment_method] = method_counts.get(pred.recommended_payment_method, 0) + 1

        if idx % 50 == 0 or idx == len(requests_to_eval):
            print(f"  Processed {idx:3d}/{len(requests_to_eval):3d} requests...")

    # 4. Write output.csv
    print(f"\n[4/5] Writing formatted predictions to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(OUTPUT_COLUMNS)
        for res in results:
            writer.writerow([
                res.request_id,
                format_safe_amount(res.amount_safe_to_pay),
                res.affordability_status,
                res.recommended_payment_method,
                res.payment_plan,
                res.earliest_date_for_full_payment,
                res.spending_changes_needed,
                res.decision_explanation,
            ])

    # Write usage report
    synthesizer.write_usage_report(USAGE_REPORT_MD)
    print(f"  Usage report generated: {USAGE_REPORT_MD}")

    # 5. Invariant validation audit
    print("\n[5/5] Auditing output against all 14 property-based invariants...")
    is_valid, errors = validator.validate_csv_file(str(output_path))
    if not is_valid:
        print(f"\nERROR: Invariant validation FAILED with {len(errors)} violations:")
        for err in errors[:20]:
            print(f"  * {err}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors.")
        return 1

    print(">>> 100% OF PREDICTIONS PASSED ALL 14 INVARIANTS PERFECTLY! ZERO ERRORS.")

    # Print Summary Breakdown
    print("\n=======================================================")
    print("                 PIPELINE RUN SUMMARY                  ")
    print("=======================================================")
    print("Affordability Status Distribution:")
    for status, count in status_counts.items():
        pct = (count / len(results)) * 100.0 if results else 0
        print(f"  - {status:22s}: {count:3d} ({pct:5.1f}%)")

    print("\nPayment Method Distribution:")
    for method, count in method_counts.items():
        pct = (count / len(results)) * 100.0 if results else 0
        print(f"  - {method:22s}: {count:3d} ({pct:5.1f}%)")

    print("\nToken & Cost Audit:")
    print(f"  - Provider:         {synthesizer.stats.provider}")
    print(f"  - Model:            {synthesizer.stats.model_name}")
    print(f"  - API Succeeded:    {synthesizer.stats.api_calls_succeeded}")
    print(f"  - Total Tokens:     {synthesizer.stats.total_tokens:,}")
    print(f"  - Estimated Cost:   ${synthesizer.stats.estimated_cost_usd:.4f}")
    print("=======================================================\n")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Buy or Wait? AI Financial Agent")
    parser.add_argument(
        "--requests",
        type=Path,
        default=REQUESTS_CSV,
        help="Path to evaluation requests CSV (default: dataset/requests.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_CSV,
        help="Path to target output CSV (default: output.csv)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="",
        help="Optional NVIDIA NIM API key for LLM explanation enrichment",
    )
    parser.add_argument(
        "--samples",
        action="store_true",
        help="Run on dataset/sample_requests.csv instead of requests.csv",
    )

    args = parser.parse_args()
    api_key = args.api_key or os.environ.get("NVIDIA_API_KEY", "")

    exit_code = run_pipeline(
        requests_path=args.requests,
        output_path=args.output,
        api_key=api_key,
        run_samples=args.samples,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
