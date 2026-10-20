#!/usr/bin/env python3
"""
operational_draft.py
====================
Representativeness check: the ten executions used for the accuracy figures vs the ONE draft the
measurer actually received in Phase 2 (the operational execution). Phase 2 only (200 measurements),
same references as in the paper, no new executions, no confidence intervals.

The operational draft is identified by the column `operational_draft` (1 = the execution whose
output was reviewed by the measurer; it carries `run = 1` by construction of the file, and for the
52 spot-checked measurements its sequence equals `raw_draft_sequence` in spot_checks.csv), not by the
row order of the file.

Aggregation as in Sec. VI-D: per-measurement mean over executions, then pooled mean over
measurements (ROUGE-L F1, MAE); exact-count rate = mean over executions of the execution-level
exact-match percentage. For the single operational draft the two aggregations coincide. The check documents the closeness of the
aggregate results; it makes no claim about the statistical equivalence of the executions.
"""
import os, pandas as pd
here = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(here, "..", "data", "scores_per_measurement.csv"))
p2 = df[df.phase == 2]
assert set(p2.groupby("measurement_id").operational_draft.sum()) == {1}, "exactly one operational draft per measurement"
sp = pd.read_csv(os.path.join(here, "..", "data", "spot_checks.csv"))
op = p2[p2.operational_draft == 1].set_index("measurement_id")
assert all(op.loc[m, "tool_sequence"] == s for m, s in zip(sp.measurement_id, sp.raw_draft_sequence)), "spot-check raw drafts must be the operational drafts"

def summarise(d):
    pm = d.groupby("measurement_id")[["dm_rougeL_f1", "abs_err_cfp"]].mean()
    exact = d.groupby("run").abs_err_cfp.apply(lambda s: (s == 0).mean()).mean()
    return len(pm), pm.dm_rougeL_f1.mean(), pm.abs_err_cfp.mean(), exact
rows = [("Operational drafts actually reviewed", *summarise(p2[p2.operational_draft == 1])),
        ("All ten executions of the same measurements", *summarise(p2))]
out = pd.DataFrame(rows, columns=["outputs", "measurements", "dm_rougeL_f1", "mae_cfp", "exact_count_rate"])
os.makedirs(os.path.join(here, "..", "results"), exist_ok=True)
out.to_csv(os.path.join(here, "..", "results", "operational_draft_vs_ten_runs.csv"), index=False)
print("Phase 2, 200 measurements, same references as the paper:")
for _, r in out.iterrows():
    print(f"  {r.outputs:<46} n={r.measurements}  ROUGE-L F1 {100*r.dm_rougeL_f1:.1f}%  MAE {r.mae_cfp:.3f} CFP  exact-count {100*r.exact_count_rate:.1f}%")
print("Written: results/operational_draft_vs_ten_runs.csv")
