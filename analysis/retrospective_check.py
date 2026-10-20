#!/usr/bin/env python3
"""
retrospective_check.py
======================
Retrospective independent check of the routine review (Phase 2), on 50 of the 148 measurements that
were not spot-checked during the study: the 25 whose movement-type sequence the review changed (A != B)
and 25 drawn at random (numpy default_rng, seed 20260629, choice without replacement over the candidate ids in
file order) from the 122 unchanged and retrievable ones (one of the 123 has no retrievable document version,
data/change_magnitude.csv). Sample: data/retrospective_check_sample.csv; the script re-runs the draw and checks it.

A checker who neither produced nor reviewed those measurements re-measures each process on the archived
document version, blind to the draft (A) and to the accepted sheet (B). The returned sheet is expected as
data/retrospective_check.csv with columns check_id, measurement_id, sequence (E/X/R/W string, step order;
empty when the version is not retrievable), checker.

Outcome per measurement, comparing A (operational draft, run 1), B (accepted after review) and C (checker):
  review-changed (25):  confirmed   B == C != A   the review corrected a draft error
                        reversed    A == C != B   the review introduced a discrepancy into a correct draft
                        partial     A != C, B != C  the draft was wrong and the review did not reach the checker's value
  unchanged (25):       residual    B != C        a discrepancy the review did not detect
                        agreed      B == C
The script reports the counts and the CFP differences; it does not interpret them.
"""
import os, sys, pandas as pd
here = os.path.dirname(os.path.abspath(__file__))
smp = pd.read_csv(os.path.join(here, "..", "data", "retrospective_check_sample.csv"))
path = os.path.join(here, "..", "data", "retrospective_check.csv")
print(f"Sample: {len(smp)} measurements ({int(smp.review_changed.sum())} review-changed, "
      f"{int((smp.review_changed == 0).sum())} unchanged), seed 20260629")
if not os.path.exists(path):
    print("data/retrospective_check.csv not present: the checker's sheet has not been returned yet; nothing computed.")
    sys.exit(0)
chk = pd.read_csv(path).set_index("measurement_id")
sc = pd.read_csv(os.path.join(here, "..", "data", "scores_per_measurement.csv"))
A = sc[(sc.phase == 2) & (sc.operational_draft == 1)].set_index("measurement_id")

# Reproduce the draw of the 25 unchanged measurements from the seed (Sec. VI-D: "the replication package
# provides the sampling seed"): candidates = Phase 2 measurements not spot-checked whose step-level
# movement-type sequence the review left unchanged (results/review_changes_phase2.csv, produced by
# review_changes.py) and whose document version is retrievable (data/change_magnitude.csv, case != "non reperibile").
rc_path = os.path.join(here, "..", "results", "review_changes_phase2.csv")
if os.path.exists(rc_path):
    import numpy as np
    rc = pd.read_csv(rc_path)
    cm = pd.read_csv(os.path.join(here, "..", "data", "change_magnitude.csv"))
    not_retrievable = set(cm[cm.case == "non reperibile"].measurement_id)
    cands = rc[(rc.AB_unchanged_types == 1) & (rc.spot_check == 0) & (~rc.measurement_id.isin(not_retrievable))].measurement_id.tolist()
    drawn = set(np.random.default_rng(20260629).choice(cands, 25, replace=False))
    in_file = set(smp[smp.review_changed == 0].measurement_id)
    print(f"Draw check: {len(cands)} candidates (unchanged, not spot-checked, retrievable); "
          f"numpy default_rng(20260629).choice(cands, 25, replace=False) {'reproduces' if drawn == in_file else 'DOES NOT reproduce'} "
          f"the 25 unchanged measurements of the sample file")
    assert set(smp[smp.review_changed == 1].measurement_id) == set(rc[rc.AB_unchanged_types == 0].measurement_id), \
        "the 25 review-changed measurements of the sample must be exactly those of review_changes_phase2.csv"
rows = []
for r in smp.itertuples(index=False):
    a, b = A.loc[r.measurement_id, "tool_sequence"], A.loc[r.measurement_id, "ref_sequence"]
    c = chk.loc[r.measurement_id, "sequence"] if r.measurement_id in chk.index else None
    c = None if (c is None or pd.isna(c) or str(c).strip() == "") else "".join(ch for ch in str(c).upper() if ch in "EXRW")
    if c is None: out = "not retrievable"
    elif r.review_changed: out = "confirmed" if b == c else ("reversed" if a == c else "partial")
    else: out = "agreed" if b == c else "residual"
    rows.append(dict(check_id=r.check_id, measurement_id=r.measurement_id, project=r.project, review_changed=r.review_changed,
                     draft=a, accepted=b, checker=c, outcome=out,
                     cfp_draft=len(a), cfp_accepted=len(b), cfp_checker=(len(c) if c else None)))
df = pd.DataFrame(rows)
res = os.path.join(here, "..", "results"); os.makedirs(res, exist_ok=True)
df.to_csv(os.path.join(res, "retrospective_check.csv"), index=False)
print("\nReview-changed measurements:"); print(df[df.review_changed == 1].outcome.value_counts().to_string())
print("\nUnchanged measurements:"); print(df[df.review_changed == 0].outcome.value_counts().to_string())
m = df[df.checker.notna()]
print(f"\nCFP: checker vs accepted differs in {(m.cfp_checker != m.cfp_accepted).sum()}/{len(m)}; "
      f"checker vs draft differs in {(m.cfp_checker != m.cfp_draft).sum()}/{len(m)}")
print("Written: results/retrospective_check.csv")
