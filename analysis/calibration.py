#!/usr/bin/env python3
"""
calibration.py
==============
Summary of the calibration runs (Phase 0, 2-18 Dec 2025) recorded in
data/calibration_log.csv. Calibration set: 32 use cases (58 functional
processes, 190 CFP) from a product completed and manually measured in 2024,
disjoint from P1-P5. Each row is one configuration run on the whole set.

* Sentence Splitter rows (C01-C05): few-shot examples (16 = original study,
  30 = 16 + 14 company examples) x reasoning effort; the figure is the number
  of use cases whose functional-process count differs from the manual
  decomposition ("decomposition errors").
* COSMIC Analyzer rows (C06-C20): Splitter fixed at 30 examples / medium;
  examples 7/9/12/15 x reasoning effort low/medium/high, plus two max_tokens
  trials at 4,096 (C18-C19) and the confirmation run of the frozen
  configuration (C20). Metrics: data-movement ROUGE-L F1 and MAE (CFP per
  functional process) against the 2024 manual measurement, outputs truncated
  by max_tokens, and median end-to-end seconds per functional process.

The retained configuration is C04 + C13/C20: 30 / 12 examples, medium
reasoning effort, max_tokens 8,192 (frozen 19 Dec 2025). Only aggregate
scores are released; the calibration use cases are confidential.
"""
import os
import pandas as pd

here = os.path.dirname(os.path.abspath(__file__))
log = pd.read_csv(os.path.join(here, "..", "data", "calibration_log.csv"))
pd.set_option("display.width", 200)
s = log[log.component == "splitter"]
print("Sentence Splitter (decomposition errors over 32 use cases):")
print(s.pivot(index="splitter_examples", columns="reasoning_effort", values="decomposition_errors_uc")[["low", "medium", "high"]].to_string())
a = log[(log.component == "analyzer") & (log.max_tokens == 8192) & (log.run_id != "C20")]
print("\nCOSMIC Analyzer, max_tokens 8192 (DM ROUGE-L F1 / MAE CFP per FP / median s per FP):")
for m in ["dm_rougeL_f1", "mae_cfp_per_fp", "median_seconds"]:
    print(f"  {m}"); print(a.pivot(index="analyzer_examples", columns="reasoning_effort", values=m)[["low", "medium", "high"]].to_string())
t = log[log.max_tokens == 4096]
print("\nmax_tokens 4096 trials (truncated outputs):"); print(t[["run_id", "analyzer_examples", "reasoning_effort", "dm_rougeL_f1", "mae_cfp_per_fp", "truncated_outputs"]].to_string(index=False))
c = log[log.run_id.isin(["C13", "C20"])]
print("\nFrozen configuration (C13, confirmed by C20):"); print(c[["run_id", "date", "analyzer_examples", "reasoning_effort", "max_tokens", "dm_rougeL_f1", "mae_cfp_per_fp", "median_seconds"]].to_string(index=False))
