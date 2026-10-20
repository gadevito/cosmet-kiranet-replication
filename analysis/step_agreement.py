#!/usr/bin/env python3
"""Step-level agreement (Sec. VII-A, supplementary).

Grid. Both analyses follow the numbered steps of the use case. CosMet's Splitter rewrites
a step into one or more sub-processes but keeps the step number of the source sentence, so
every movement in the draft carries the number of the original step; the manual analysis
sheet is organised by the same numbers. Steps are therefore compared by number, never by
alignment.

Within a step. The step's movements are an ORDERED list of types. A step's types are
"reproduced exactly" when the draft's list equals the reference's list, element by element,
in order (so ['R','R'] != ['R'], and ['E','X'] != ['X','E']). Matched movements (matched_dm)
use the per-step LCS of the two lists, which handles repeated types by multiplicity.

Labels. A movement's data group and object of interest are "reproduced" when
text_metrics.label_match holds, i.e. the two strings are equal after text_metrics.normalise
(NFKC, lower-case, every non-alphanumeric character replaced by a space, whitespace-tokenised,
accents kept). This is the same normalisation used for the ROUGE-L/BLEU-2 text scores; no other
transformation is applied and no semantic equivalence is credited. The joint criterion for a
step is implemented, position by position, in text_metrics.step_joint_reproduced.

Empty steps. A step with no data movement in the reference counts as reproduced when the
draft also assigns it no movement; this is reported both INCLUDING and EXCLUDING empty steps,
because including them raises the agreement.

Aggregation. For each execution, the agreement is the proportion of the measurement's steps
that are reproduced; the ten executions are averaged within the measurement; the reported
value is the mean of the 1,000 per-measurement values (equal weight per measurement, as for
every other RQ1 metric), with its SD and the 95% project-cluster percentile bootstrap CI
(B = 10,000, same procedure and seed as bootstrap_ci.py). The pooled proportion over all
steps is printed alongside for comparison.

Columns: step_type_agreement is recomputed here from ref_steps/tool_steps; step_joint_agreement
(type + DG + OOI) is released as a per-execution score because the labels are confidential."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sequence_metrics import parse_steps, step_type_agreement
from text_metrics import normalise, label_match, step_joint_reproduced  # the label procedure (documented there)
from bootstrap_ci import per_measurement, cluster_bootstrap
here = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(here, "..", "data", "scores_per_measurement.csv"))
R = df.ref_steps.map(parse_steps); C = df.tool_steps.map(parse_steps)
df["step_type_agreement"] = [step_type_agreement(r, c) for r, c in zip(R, C)]
ne = [[a == b for a, b in zip(r, c) if a] for r, c in zip(R, C)]
df["step_type_agreement_nonempty"] = [np.mean(x) if x else np.nan for x in ne]
n_empty_run = sum(1 for r in R for st in r if not st) / df.run.nunique()
empty_ok = sum(1 for r, c in zip(R, C) for a, b in zip(r, c) if not a and not b) / df.run.nunique()
pooled_all = sum(a == b for r, c in zip(R, C) for a, b in zip(r, c)) / sum(len(r) for r in R)
pooled_ne = sum(x for l in ne for x in l) / sum(len(l) for l in ne)
pm = per_measurement(df)
R1 = df[df.run == 1].ref_steps.map(parse_steps)
n_steps = sum(len(r) for r in R1); n_empty = sum(1 for r in R1 for st in r if not st)
print(f"steps in the reference analyses: {int(n_steps):,} for 6,000 data movements ({int(n_empty):,} steps without a data movement)")
print(f"empty reference steps reproduced as empty by the draft: {empty_ok/n_empty_run:.3f}")
for col, lab in [("step_type_agreement", "step type agreement, all steps"),
                 ("step_type_agreement_nonempty", "step type agreement, steps with movements only"),
                 ("step_joint_agreement", "step joint agreement (type + DG + OOI), all steps"),
                 ("step_joint_agreement_nonempty", "step joint agreement (type + DG + OOI), steps with movements only")]:
    v = pm[col].dropna(); lo, hi = cluster_bootstrap(pm.dropna(subset=[col]), col)
    print(f"{lab}: mean of per-measurement proportions {v.mean():.3f} (SD {v.std(ddof=1):.3f}; 95% CI [{lo:.3f}, {hi:.3f}])")
print(f"pooled over steps (for comparison): type agreement all steps {pooled_all:.3f}, steps with movements only {pooled_ne:.3f}")
print("(both joint columns are released per execution because the labels are confidential; the non-empty variant is the same criterion restricted to reference steps with movements)")
print("fraction of measurements with every step jointly correct in all executions:",
      f"{(df.groupby('measurement_id').step_joint_agreement.min() == 1).mean():.3f}")
pd.DataFrame({"metric": ["step_type_agreement", "step_type_agreement_nonempty", "step_joint_agreement", "step_joint_agreement_nonempty"],
              "mean": [pm.step_type_agreement.mean(), pm.step_type_agreement_nonempty.mean(), pm.step_joint_agreement.mean(), pm.step_joint_agreement_nonempty.mean()]}).to_csv(
    os.path.join(here, "..", "results", "step_agreement.csv"), index=False)
