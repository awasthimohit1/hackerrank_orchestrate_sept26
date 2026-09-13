"""Test script to evaluate forecaster accuracy on sample requests."""

import os
import sys

sys.path.insert(0, os.getcwd())

from code.data_loader import DataLoader
from code.evidence_extractor import EvidenceExtractor
from code.forecaster import CashflowForecaster

loader = DataLoader()
extractor = EvidenceExtractor()
forecaster = CashflowForecaster(loader, extractor)

samples = loader.load_sample_requests()
print(f"Loaded {len(samples)} sample requests.\n")

matches = 0
for req, gt in samples:
    balances, min_bal, headroom = forecaster.simulate_90_days(request=req)
    predicted_safe = max(0.0, min(req.requested_amount, headroom))
    gt_safe = gt.amount_safe_to_pay

    diff = abs(predicted_safe - gt_safe)
    is_match = diff < 0.05
    if is_match:
        matches += 1
    print(f"{req.request_id} ({req.user_id}): GT_safe={gt_safe:.2f}, Pred_safe={predicted_safe:.2f}, diff={diff:.2f}, match={is_match}")

print(f"\nSafe Amount Matches: {matches}/{len(samples)} ({matches/len(samples)*100:.1f}%)")
