#!/usr/bin/env python3
"""
bootstrap_ci.py
===============
Point estimates, dispersion and 95% cluster-bootstrap confidence intervals for
the RQ1 accuracy metrics, exactly as defined in the paper's Study Design.

Definitions (mirroring Sec. VI)
------------------------------
* Unit of analysis: the MEASUREMENT (one functional process measured at one
  point in time; a process re-measured after a later change is a new
  measurement). Each measurement was executed N_RUNS = 10 times; the per-run
  scores are first averaged within the measurement.
* Point estimate: POOLED per-measurement mean (each measurement weighs 1).
* SD: sample standard deviation (n-1) over the per-measurement means.
* "Correctly recovered data movements": sum over measurements of the LCS length
  between reference and tool sequence (run-averaged), over the sum of reference
  data movements.
* MAE / MdAE: mean / median of |tool CFP - reference CFP| per measurement.
* Uncertainty: CLUSTER bootstrap. Functional processes of one project share
  domain and writing style, so the resampling unit is the PROJECT: draw 5
  projects with replacement (a project drawn twice contributes all its
  measurements twice), recompute the pooled statistic, repeat B = 10,000 times,
  take the 2.5th / 97.5th percentiles. PERCENTILE method throughout (no BCa).
  With 5 clusters there are only 126 distinct resamples, so intervals are
  coarse and the MdAE interval is degenerate; the paper says so.
* Sample: ALL 1,000 measurements (6,000 CFP), i.e. both reference-provenance
  groups (852 with an independent manual reference / 5,140 CFP, and 148 whose
  reference descends from the reviewed draft / 860 CFP), exactly as in Sec. VII-A.
  `--independent-only` restricts to the 852 as a sensitivity analysis.
* Supplementary descriptives reported in Sec. VII-A: (a) micro-averaged ROUGE-L
  recall = sum over measurements of the run-averaged LCS length, rounded to an
  integer, over the total reference data movements; (b) the data-movement
  ROUGE-L F1 with equal PROJECT weights; (c) exact-count rate = mean over the
  ten executions of the execution-level exact-match percentage.

Usage
-----
    python analysis/bootstrap_ci.py                       # all 1,000 measurements (paper)
    python analysis/bootstrap_ci.py --independent-only    # sensitivity: 852 with independent reference
    python analysis/bootstrap_ci.py --by-project
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

SEED = 20260629
B = 10_000

METRICS = {  # column (per-run) -> label ; TSE GPT-4 value for the compatibility check
    "te_rougeL_f1":  ("Triggering event  ROUGE-L F1", 1.000),
    "fu_rougeL_f1":  ("Functional user   ROUGE-L F1", 0.998),
    "dg_rougeL_f1":  ("Data group        ROUGE-L F1", 0.944),
    "ooi_rougeL_f1": ("Object of interest ROUGE-L F1", 0.845),
    "dm_rougeL_f1":  ("Data movement     ROUGE-L F1", 0.990),
    "dm_bleu2":      ("Data movement     BLEU-2", None),
    "fu_bertscore_f1": ("Functional user   BERTScore F1", None),
    "dg_bertscore_f1": ("Data group        BERTScore F1", None),
    "ooi_bertscore_f1": ("Object of interest BERTScore F1", None),
    "abs_err_cfp":   ("Data movement     MAE (CFP)", 0.038),
}


def per_measurement(df: pd.DataFrame) -> pd.DataFrame:
    """Average the 10 runs within each measurement."""
    keep = ["measurement_id", "project", "phase", "fp_id", "reference_type"]
    num = [c for c in df.columns if c not in keep + ["run", "spot_check", "operational_draft", "reference_corrected_in_phase1_review", "process_version"] and pd.api.types.is_numeric_dtype(df[c])]
    return df.groupby(keep, as_index=False)[num].mean()


def cluster_bootstrap(pm: pd.DataFrame, col: str, stat: str = "mean"):
    """Project-cluster bootstrap of the pooled statistic. Each resample draws the
    5 projects with replacement and concatenates all their measurements; a fresh
    generator seeded with SEED is used for every metric so each interval is
    reproducible on its own."""
    rng = np.random.default_rng(SEED)
    groups = [g[col].to_numpy(float) for _, g in pm.groupby("project")]
    k = len(groups)
    idx = rng.integers(0, k, size=(B, k))
    if stat == "mean":   # pooled mean = size-weighted mean of the drawn project means
        means = np.array([g.mean() for g in groups]); ns = np.array([g.size for g in groups])
        out = (means[idx] * ns[idx]).sum(1) / ns[idx].sum(1)
    else:                # median: needs the concatenated values
        out = np.array([np.median(np.concatenate([groups[i] for i in row])) for row in idx])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--input", default=os.path.join(here, "..", "data", "scores_per_measurement.csv"))
    ap.add_argument("--output", default=os.path.join(here, "..", "results", "rq1_confidence_intervals.csv"))
    ap.add_argument("--independent-only", action="store_true", help="restrict to measurements with an independent manual reference")
    ap.add_argument("--strict-independent", action="store_true", help="also drop the 11 Phase 0/1 references corrected in the Phase 1 discrepancy review")
    ap.add_argument("--post-review-only", action="store_true", help="the 148 measurements whose reference descends from the reviewed draft (point estimates; no CI reported in the paper)")
    ap.add_argument("--by-project", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if args.independent_only or args.strict_independent:
        df = df[df.reference_type == "independent"]
    if args.post_review_only:
        df = df[df.reference_type == "post-review"]
    if args.strict_independent and "reference_corrected_in_phase1_review" in df.columns:
        df = df[df.reference_corrected_in_phase1_review == 0]
    pm = per_measurement(df)
    print(f"sample: {len(pm)} measurements, {int(pm.ref_dm.sum())} reference data movements, "
          f"{pm.project.nunique()} projects, {df.run.nunique()} runs each "
          f"({'independent reference only' if args.independent_only else 'both reference-provenance groups'})")
    for rt, g in pm.groupby("reference_type"):
        print(f"  {rt:<12} {len(g):4d} measurements / {int(g.ref_dm.sum())} CFP")
    lcs_sum = pm.lcs_correct.sum()
    print(f"matched data movements (run-averaged LCS, summed, rounded): {round(lcs_sum)} / {int(pm.ref_dm.sum())} "
          f"-> micro-averaged ROUGE-L recall {100*lcs_sum/pm.ref_dm.sum():.1f}%")
    print(f"data-movement ROUGE-L F1: equal measurement weight {100*pm.dm_rougeL_f1.mean():.1f}%  |  "
          f"equal project weight {100*pm.groupby('project').dm_rougeL_f1.mean().mean():.1f}%")
    exact_by_run = df.groupby("run").abs_err_cfp.apply(lambda s: np.mean(s == 0))
    print(f"exact-count rate (mean of the {df.run.nunique()} execution-level percentages): {100*exact_by_run.mean():.0f}%   "
          f"MdAE = {np.median(pm.abs_err_cfp):.0f} (no interval: zero in every resample)")

    rows = []
    for col, (label, tse) in METRICS.items():
        if col not in pm.columns:
            continue
        v = pm[col].to_numpy(float)
        mean, sd = float(v.mean()), float(v.std(ddof=1))
        if np.allclose(v, v[0]):
            lo, hi = mean, mean
        else:
            lo, hi = cluster_bootstrap(pm, col)
        rows.append({"metric": label, "n": len(v), "mean": round(mean, 4), "sd": round(sd, 4),
                     "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
                     "tse_gpt4": tse if tse is not None else "",
                     "tse_in_ci": ("" if tse is None else ("yes" if lo <= tse <= hi else "NO")),
                     "method": "cluster percentile", "B": B, "seed": SEED})
    # median absolute error (degenerate CI with 5 clusters, reported for completeness)
    lo, hi = cluster_bootstrap(pm, "abs_err_cfp", stat="median")
    rows.append({"metric": "Data movement     MdAE (CFP)", "n": len(pm),
                 "mean": float(pm.abs_err_cfp.median()), "sd": "", "ci95_low": lo, "ci95_high": hi,
                 "tse_gpt4": 0, "tse_in_ci": "yes" if lo <= 0 <= hi else "NO",
                 "method": "cluster percentile", "B": B, "seed": SEED})

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out.to_csv(args.output, index=False)
    pd.set_option("display.width", 160)
    print("\n95% cluster-bootstrap CIs (B = 10,000, percentile, seed = {})".format(SEED))
    print(out.to_string(index=False))
    print(f"\nWritten: {os.path.relpath(args.output)}")

    if args.by_project:
        print("\nPer project (mean of per-measurement scores):")
        print(pm.groupby("project")[["dm_rougeL_f1", "dm_bleu2", "abs_err_cfp", "ooi_rougeL_f1"]]
              .mean().round(3).to_string())


if __name__ == "__main__":
    main()
