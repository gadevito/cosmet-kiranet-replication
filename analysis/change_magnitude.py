#!/usr/bin/env python3
"""
change_magnitude.py
===================
How much had each measured requirement changed? Extracted by the company from its version
control (October 2026) for the 400 maintenance measurements (P2-P5, Phase 1 manual and Phase 2
assisted): for each functional process, the number of numbered use-case steps added, removed or
modified with respect to the previous version accepted in the product baseline
(`data/change_magnitude.csv`, column `steps_changed_vs_previous_version`). `case` = "nuovo"
(process absent from the previous version) or "non reperibile" (previous version not
retrievable): these rows have no count and are reported separately. share = changed steps /
steps of the measured version. One row per measurement; versions of the same process are distinct.
Descriptive only: it describes the two workloads, it does not test their equivalence.
"""
import os, numpy as np, pandas as pd
here = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(here, "..", "data", "change_magnitude.csv"))
d["workflow"] = d.phase.map({1: "Manual (Phase 1)", 2: "Assisted (Phase 2)"})
d["share"] = d.steps_changed_vs_previous_version / d.steps_measured_version
def desc(g):
    c = g[g.steps_changed_vs_previous_version.notna()]
    return pd.Series({"N_rows": len(g), "N_with_count": len(c), "new_process": int((g.case == "nuovo").sum()),
                      "not_retrievable": int((g.case == "non reperibile").sum()),
                      "median_changed_steps": c.steps_changed_vs_previous_version.median(), "Q1": c.steps_changed_vs_previous_version.quantile(.25),
                      "Q3": c.steps_changed_vs_previous_version.quantile(.75), "min": c.steps_changed_vs_previous_version.min(), "max": c.steps_changed_vs_previous_version.max(),
                      "mean_changed_steps": round(c.steps_changed_vs_previous_version.mean(), 2),
                      "median_share": round(c.share.median(), 3), "mean_share": round(c.share.mean(), 3), "zero_changed": int((c.steps_changed_vs_previous_version == 0).sum())})
by_phase = d.groupby("workflow").apply(desc, include_groups=False).loc[["Manual (Phase 1)", "Assisted (Phase 2)"]]
by_prod = d.groupby(["project", "workflow"]).apply(desc, include_groups=False)
p2 = d[d.phase == 2].copy(); p2["kind"] = np.where(p2.process_version > 1, "re-measurement (version 2-3)", "first measurement in the period")
by_kind = p2.groupby("kind").apply(desc, include_groups=False)
os.makedirs(os.path.join(here, "..", "results"), exist_ok=True)
pd.concat([by_phase.assign(level="phase"), by_prod.reset_index(level=0).assign(level="project"), by_kind.assign(level="phase2_kind")]).to_csv(
    os.path.join(here, "..", "results", "change_magnitude.csv"))
pd.set_option("display.width", 220)
print("Requirement change per measured process (steps added/removed/modified vs previous accepted version), maintenance products:")
print(by_phase.to_string()); print(); print(by_prod.to_string()); print(); print(by_kind.to_string())
print("\nWritten: results/change_magnitude.csv")
