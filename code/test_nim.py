"""Test script to verify NVIDIA NIM API key connectivity and model response."""

import os
import sys

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from code.config import REPO_ROOT
from code.data_loader import DataLoader
from code.decision_engine import DecisionEngine
from code.evidence_extractor import EvidenceExtractor
from code.explanation_synthesizer import ExplanationSynthesizer
from code.forecaster import CashflowForecaster


def test_nim_connectivity():
    api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
    print("=======================================================")
    print("      NVIDIA NIM API KEY CONNECTIVITY TEST             ")
    print("=======================================================")

    if not api_key:
        print("[!] No NVIDIA_API_KEY found in environment or .env file.")
        print("    Please set it using one of the following methods:")
        print("    1. Create a .env file in the repo root:")
        print("       echo 'NVIDIA_API_KEY=nvapi-...' > .env")
        print("    2. Or export in your terminal:")
        print("       export NVIDIA_API_KEY='nvapi-...'")
        print("=======================================================")
        return 1

    masked_key = api_key[:7] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
    print(f"Detected API Key: {masked_key}")
    print("Testing call to NVIDIA NIM (deepseek-ai/deepseek-v4-pro-0813)...")

    loader = DataLoader()
    extractor = EvidenceExtractor()
    forecaster = CashflowForecaster(loader, extractor)
    engine = DecisionEngine(loader, forecaster)
    synthesizer = ExplanationSynthesizer(api_key=api_key)

    samples = loader.load_sample_requests()
    test_req, _ = samples[0]
    profile = loader.profiles[test_req.user_id]
    pred = engine.evaluate_request(test_req)

    print(f"\nEvaluating test sample: {test_req.request_id} ({test_req.request_type})")
    print(f"Base Deterministic Explanation:\n  -> {pred.decision_explanation}")

    enriched = synthesizer.synthesize_explanation(test_req, pred, profile)
    print(f"\nSynthesizer Output:\n  -> {enriched}")

    if synthesizer.stats.api_calls_succeeded > 0:
        print("\n>>> SUCCESS! NVIDIA NIM API key is working perfectly.")
        print(f"    - Prompt Tokens:     {synthesizer.stats.prompt_tokens}")
        print(f"    - Completion Tokens: {synthesizer.stats.completion_tokens}")
        print(f"    - Total Cost:        ${synthesizer.stats.estimated_cost_usd:.6f}")
        print("=======================================================")
        return 0
    else:
        print("\n[X] FAILED: API call did not succeed. Fell back to deterministic output.")
        if synthesizer.last_error:
            print(f"    Error detail: {synthesizer.last_error}")
        print("    Please check that the API key is valid and has access to deepseek-ai/deepseek-v4-pro-0813.")
        print("=======================================================")
        return 1


if __name__ == "__main__":
    sys.exit(test_nim_connectivity())
