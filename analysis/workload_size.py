#!/usr/bin/env python3
"""
workload_size.py
================
Size of the measured requirements, for the maintenance products only (P2-P5): the Phase 1 manual
workload (200 measurements) vs the Phase 2 assisted workload (200 measurements). Size = number of
numbered steps of the original use case belonging to the functional process, INCLUDING steps without
data movements (`ref_steps`, one entry per step). One observation per measurement (operational
execution row); successive versions of the same process are distinct observations. P1 (Phase 0 manual,
no tool-first counterpart) is excluded. Descriptive only: median, Q1-Q3, min-max, per product and pooled.
"""
import os, pandas as pd
here = os.path.dirname(os.path.abspath(__file__))
sc = pd.read_csv(os.path.join(here, "..", "data", "scores_per_measurement.csv"))
r = sc[(sc.operational_draft == 1) & (sc.project != "P1")].copy()
r["n_steps"] = r.ref_steps.str.count(r"\|") + 1
r["workflow"] = r.phase.map({1: "Manual (Phase 1)", 2: "Assisted (Phase 2)"})
def desc(g):
    q = g.n_steps.quantile([.25, .5, .75])
    return pd.Series({"N": len(g), "median_steps": q[.5], "Q1": q[.25], "Q3": q[.75], "min": g.n_steps.min(), "max": g.n_steps.max(),
                      "mean": round(g.n_steps.mean(), 2), "total_steps": int(g.n_steps.sum()), "CFP": int(g.ref_dm.sum())})
rows = [desc(g).rename((p, w)) for (p, w), g in r.groupby(["project", "workflow"], sort=True)]
rows += [desc(g).rename(("P2-P5", w)) for w, g in r.groupby("workflow")]
out = pd.DataFrame(rows); out.index.names = ["project", "workflow"]
out = out.reset_index(); out["_o"] = out.project.map({"P2": 0, "P3": 1, "P4": 2, "P5": 3, "P2-P5": 4}) * 2 + out.workflow.map({"Manual (Phase 1)": 0, "Assisted (Phase 2)": 1})
out = out.sort_values("_o").drop(columns="_o")
for c in ["N", "min", "max", "total_steps"]: out[c] = out[c].astype(int)
out.to_csv(os.path.join(here, "..", "results", "workload_size_steps.csv"), index=False)
print("Original use-case steps per measurement (incl. steps without movements), maintenance products, one row per measurement:")
print(out.to_string(index=False))
print("Written: results/workload_size_steps.csv")
