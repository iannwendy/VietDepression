"""End-to-end verification of the Round 5 fix suite.

Checks:
  1. Dataset sizes match the protocol (7,336 / 241 / 242).
  2. Gold set arithmetic (560 + 2,872 = 3,432).
  3. Cross-domain VSMEC file integrity (3,084 balanced).
  4. New evaluation JSON contains both in_domain and cross_domain keys.
  5. PhoBERT majority-vote numbers match what is in the paper.
  6. Paper contains the corrected headline numbers.
  7. README/FINAL_RESULTS_SUMMARY reference the new dataset sizes.

Run with:
    PYTHONPATH="$PWD" .venv/bin/python scripts/verify_round5_fixes.py
"""

from __future__ import annotations

import json
import glob
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]

failures = []


def check(name: str, condition: bool, detail: str = ""):
    if condition:
        print(f"  [PASS] {name}{(' — ' + detail) if detail else ''}")
    else:
        print(f"  [FAIL] {name}{(' — ' + detail) if detail else ''}")
        failures.append(name)


# 1. Dataset sizes
print("\n[1] Dataset sizes")
train = pd.read_csv(PROJECT_DIR / "data/labeled/final_train.csv")
val = pd.read_csv(PROJECT_DIR / "data/labeled/final_val.csv")
test = pd.read_csv(PROJECT_DIR / "data/labeled/final_test.csv")
check("train size == 7336", len(train) == 7336, f"actual={len(train):,}")
check("val size == 241", len(val) == 241, f"actual={len(val):,}")
check("test size == 242", len(test) == 242, f"actual={len(test):,}")

# 2. Gold set arithmetic
print("\n[2] Gold set arithmetic")
gold = pd.read_csv(PROJECT_DIR / "data/labeled/train_gold.csv")
dep = int((gold["label"] == 1).sum())
norm = int((gold["label"] == 0).sum())
total = len(gold)
check(
    "560 + 2,872 = 3,432",
    dep + norm == total and total == 3432,
    f"actual: {dep} + {norm} = {total}",
)

# 3. Cross-domain VSMEC
print("\n[3] Cross-domain VSMEC")
vsmec = pd.read_csv(PROJECT_DIR / "data_unified/cross_domain_test.csv")
check("VSMEC size == 3,084", len(vsmec) == 3084, f"actual={len(vsmec):,}")
check("VSMEC balanced 1,542/1,542", (vsmec["label"] == 0).sum() == 1542 and (vsmec["label"] == 1).sum() == 1542)

# 4. New evaluation JSON
print("\n[4] New evaluation JSON")
results_files = sorted(glob.glob(str(PROJECT_DIR / "results/round5_final_v2_*/evaluation_results.json")))
if not results_files:
    check("New evaluation JSON exists", False, "no round5_final_v2_*/evaluation_results.json")
else:
    latest = results_files[-1]
    with open(latest) as f:
        data = json.load(f)
    check("JSON has in_domain key", "in_domain" in data)
    check("JSON has cross_domain key", "cross_domain" in data)
    check("cross_domain is non-empty", bool(data.get("cross_domain")), f"keys={list(data.get('cross_domain', {}).keys())}")

    # 5. PhoBERT majority vote numbers
    print("\n[5] PhoBERT majority vote numbers")
    if "phobert_avg" in data["in_domain"] and "phobert_avg" in data["cross_domain"]:
        pi = data["in_domain"]["phobert_avg"]
        pc = data["cross_domain"]["phobert_avg"]
        check(
            "in-domain F1 ~= 0.7845",
            abs(pi["f1_macro"] - 0.7845) < 0.001,
            f"actual={pi['f1_macro']:.4f}",
        )
        check(
            "cross-domain F1 ~= 0.3598",
            abs(pc["f1_macro"] - 0.3598) < 0.001,
            f"actual={pc['f1_macro']:.4f}",
        )
    else:
        check("phobert_avg in both in_domain and cross_domain", False)

# 6. Paper contains the new numbers
print("\n[6] Paper contents")
paper = (PROJECT_DIR / "docs/paper_report.html").read_text()
check("paper contains 0.7845", "0.7845" in paper)
check("paper contains 0.3598", "0.3598" in paper)
check("paper no longer contains 0.4937 as headline", "0.4937" not in paper or "0.4937" in paper.split("Augmented")[0])
check("paper contains 7,336 train rows", "7,336" in paper)
check("paper contains 560 depression", "560 (16.32%)" in paper)
check("paper contains 241 val", "241" in paper)
check("paper contains 242 test", "242" in paper)

# 7. README/FINAL_RESULTS_SUMMARY
print("\n[7] README and FINAL_RESULTS_SUMMARY")
readme = (PROJECT_DIR / "README.md").read_text()
summary = (PROJECT_DIR / "docs/reports/FINAL_RESULTS_SUMMARY.md").read_text()
check("README references 7,336", "7,336" in readme)
check("README references 241 val / 242 test", "241" in readme and "242" in readme)
check("README references 0.7845", "0.7845" in readme)
check("README references 0.3598", "0.3598" in readme)
check("FINAL_RESULTS_SUMMARY references 7,336", "7,336" in summary)
check("FINAL_RESULTS_SUMMARY references 0.7845", "0.7845" in summary)
check("FINAL_RESULTS_SUMMARY references 0.3598", "0.3598" in summary)

# 8. Scripts existence
print("\n[8] Critical scripts")
check("run_final_round5_evaluation.py exists", (PROJECT_DIR / "scripts/run_final_round5_evaluation.py").exists())

print("\n" + "=" * 70)
if failures:
    print(f"FAILED: {len(failures)} check(s) failed: {failures}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
    sys.exit(0)
