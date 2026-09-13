# HackerRank Orchestrate — Model Token Usage and Cost Report

**Generated for:** Buy or Wait? Financial Decision Agent  
**Challenge:** HackerRank Orchestrate (September 2026)  

---

## 1. Executive Summary

This report documents model calls, token consumption, and financial expenditure for the final evaluation dataset run across all requests in `dataset/requests.csv`.

| Metric | Value |
| :--- | :--- |
| **Model Provider** | `NVIDIA NIM` |
| **Model Name** | `nvidia/nemotron-3-ultra-550b-a55b` |
| **Total Requests Evaluated** | `250` |
| **API Calls Succeeded** | `66` |
| **API Calls Failed / Skipped** | `184` |
| **Deterministic Fallback Explanations** | `184` |

---

## 2. Token Consumption

| Token Category | Count |
| :--- | :--- |
| **Total Input (Prompt) Tokens** | `18,193` |
| **Total Output (Completion) Tokens** | `19,265` |
| **Combined Total Tokens** | `37,458` |
| **Average Tokens per Request** | `149.83` |

---

## 3. Financial Cost Analysis

*Pricing basis: NVIDIA NIM benchmark rate ($0.14 / 1M prompt tokens, $0.28 / 1M completion tokens).*

| Cost Dimension | USD |
| :--- | :--- |
| **Total Run Cost** | `$0.007941` |
| **Average Cost per Request** | `$0.000032` |

---

## 4. Architecture & Security Compliance

- **Deterministic-First Foundation**: Financial decisions, daily cash flows, candidate search, debt reservations, and invariant enforcement are 100% deterministic and rule-based.
- **Zero Hallucination Guarantee**: Explanations strictly reference computed dates, currency symbols, and reserve thresholds from the financial engine.
- **Secret Hygiene**: No API keys, credentials, or personal tokens are stored in code or repository artifacts.
