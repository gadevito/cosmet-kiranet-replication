#!/usr/bin/env python3
"""Run-to-run repeatability of the raw tool output, independent of the reference:
for each measurement, do the 10 executions yield the same E/X/R/W sequence and the
same CFP count? (Sec. VII-A.)

What is verifiable here and what is not
---------------------------------------
* Verifiable from the released data: the figures below are recomputed from the per-execution
  sequences (`tool_sequence`, `tool_dm`, 10 rows per measurement), together with the spread of the
  varying CFP counts and the per-execution completion tokens / finish state (`analysis/timing.py`:
  no execution hit the output limit).
* Documented, not verifiable from the package: that all 10,000 executions ran under the frozen
  configuration (`environment/request_params.json`, `vllm_serve.sh`, `versions.txt`). The per-request
  server logs are not released; the package contains no per-execution record of the request parameters.
* The cause of the variation under greedy decoding (temperature 0) has not been identified; this script
  and the documentation do not attribute it to a specific source.
"""
import os, pandas as pd
here = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(here, "..", "data", "scores_per_measurement.csv"))
g = df.groupby("measurement_id")
nseq, ncfp = g.tool_sequence.nunique(), g.tool_dm.nunique()
n = len(nseq); same = int((nseq == 1).sum()); var = n - same
same_cfp = int(((nseq > 1) & (ncfp == 1)).sum())
dev = int(g.tool_sequence.apply(lambda s: len(s) - s.value_counts().iloc[0]).sum())
modal_eq_ref = int(g.apply(lambda d: d.tool_sequence.value_counts().index[0] == d.ref_sequence.iloc[0], include_groups=False)[nseq > 1].sum())
print(f"measurements with an identical data-movement sequence in all {df.run.nunique()} executions: {same}/{n} ({100*same/n:.1f}%)")
print(f"measurements whose sequence varied: {var} ({100*var/n:.1f}%); of these, CFP count unchanged in all executions: {same_cfp}, CFP count varied: {var-same_cfp}")
print(f"executions deviating from the measurement's modal sequence: {dev}/{len(df)} ({100*dev/len(df):.1f}%)")
print(f"varying measurements whose modal sequence equals the reference: {modal_eq_ref}/{var}")
print("varying measurements by project:", (nseq > 1).groupby(g.project.first()).sum().to_dict())
spread = (g.tool_dm.max() - g.tool_dm.min())[ncfp > 1]
print(f"CFP-varying measurements: distinct CFP values per measurement {ncfp[ncfp > 1].value_counts().sort_index().to_dict()}; "
      f"max-min spread {spread.value_counts().sort_index().to_dict()} (all executions within one CFP of the reference: {bool((df.abs_err_cfp <= 1).all())})")
print("configuration: documented in environment/ (frozen 19 Dec 2025); per-request server logs not released; cause of the variation not identified")
pd.DataFrame([{"identical_all_runs": same, "varied": var, "varied_same_cfp": same_cfp, "varied_cfp": var - same_cfp,
               "deviating_executions": dev, "varying_modal_equals_reference": modal_eq_ref}]).to_csv(
    os.path.join(here, "..", "results", "repeatability.csv"), index=False)
