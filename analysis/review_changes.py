#!/usr/bin/env python3
"""
review_changes.py
=================
What the routine review changed in Phase 2, at the level the released data allow: per use-case step,
the data-movement TYPES (E/X/R/W) and the CFP total. Three versions of each Phase 2 analysis:

  A  operational draft actually received by the measurer   -> scores_per_measurement.csv, operational_draft = 1
                                                               (for the 52 spot-checked measurements also spot_checks.raw_draft_sequence)
  B  analysis after the routine review, before any reconciliation with the independent spot-check
       148 non-checked measurements: B is the accepted analysis -> ref_steps / ref_sequence
        52 spot-checked measurements: B -> spot_checks.reviewed_sequence (kept distinct from C)
  C  accepted analysis after reconciliation (52 spot-checked measurements) -> spot_checks.manual_sequence
       (= the independent manual analysis; the 2 discrepancies were resolved in its favour)

Comparison A -> B per measurement (step-anchored where both sides are step-anchored, otherwise on the
movement-type sequence): unchanged / changed; among changed: movements added or removed (count differs
within a step), movements reclassified (same count, different type); CFP total changed (separate field).
Each measurement counts once per change type.

NOT available in the released data, hence not classified here: wording-only changes of data-group /
object-of-interest / functional-user labels and changes of the entity identified by a label (the labels
are confidential; only run-averaged similarity scores against the accepted analysis are released), and
changes of the triggering event (its score is 1.0 throughout). The full A/B/C analysis sheets exist in the
company's measurement repository (CosMet front end: draft; baseline: reviewed sheet; spot-check form:
checker's sheet) and a terminological-vs-substantive classification requires reading them.
"""
import os, pandas as pd
here = os.path.dirname(os.path.abspath(__file__))
sc = pd.read_csv(os.path.join(here, "..", "data", "scores_per_measurement.csv"))
sp = pd.read_csv(os.path.join(here, "..", "data", "spot_checks.csv")).set_index("measurement_id")
A = sc[(sc.phase == 2) & (sc.operational_draft == 1)].set_index("measurement_id")
steps = lambda s: [[] if st == "N" else st.split(",") for st in s.split("|")]

def diff_steps(a, b):
    added_removed = reclass = False
    for sa, sb in zip(a, b):
        if len(sa) != len(sb): added_removed = True
        elif sa != sb: reclass = True
    return added_removed, reclass

rows = []
for mid, r in A.iterrows():
    a_steps = steps(r.tool_steps); a_seq = r.tool_sequence; a_cfp = int(r.tool_dm)
    if r.spot_check == 1:
        b_seq = sp.loc[mid, "reviewed_sequence"]; b_cfp = int(sp.loc[mid, "reviewed_cfp"]); b_steps = None
        c_seq = sp.loc[mid, "manual_sequence"]; c_cfp = int(sp.loc[mid, "manual_cfp"])
    else:
        b_steps = steps(r.ref_steps); b_seq = r.ref_sequence; b_cfp = int(r.ref_dm); c_seq = c_cfp = None
    if b_steps is not None:
        ar, rc = diff_steps(a_steps, b_steps)
    else:   # sequence-level fallback for the 52 (reviewed analysis released as a sequence)
        ar = len(a_seq) != len(b_seq); rc = (not ar) and a_seq != b_seq
    acc_cfp = int(r.ref_dm)                       # accepted CFP = the reference used by the accuracy scripts
    rows.append(dict(measurement_id=mid, process_version=int(r.process_version), project=r.project, spot_check=int(r.spot_check),
                     draft_cfp=a_cfp, reviewed_cfp=b_cfp, accepted_cfp=acc_cfp,
                     review_changed_cfp=int(a_cfp != b_cfp), draft_matches_accepted_cfp=int(a_cfp == acc_cfp),
                     absolute_cfp_error=abs(a_cfp - acc_cfp),
                     AB_unchanged_types=int(a_seq == b_seq), AB_added_or_removed=int(ar), AB_reclassified=int(rc),
                     AB_cfp_changed=int(a_cfp != b_cfp),
                     BC_changed=("" if c_seq is None else int(b_seq != c_seq)),
                     BC_cfp_changed=("" if c_seq is None else int(b_cfp != c_cfp)),
                     A_equals_C=("" if c_seq is None else int(a_seq == c_seq))))
out = pd.DataFrame(rows); out.to_csv(os.path.join(here, "..", "results", "review_changes_phase2.csv"), index=False)
n = len(out); ch = out[out.AB_unchanged_types == 0]
# consistency with the operational-draft figures of Sec. VII-A: same selection (operational_draft = 1, Phase 2), same key
assert (out.reviewed_cfp == out.accepted_cfp).all(), "B and the accepted analysis must have the same CFP total in all 200 rows"
assert out.measurement_id.is_unique and len(out) == 200
print(f"CFP consistency: reviewed_cfp == accepted_cfp in {int((out.reviewed_cfp == out.accepted_cfp).sum())}/200; "
      f"review changed the CFP total in {int(out.review_changed_cfp.sum())}; exact-count rate of the operational drafts "
      f"{100*out.draft_matches_accepted_cfp.mean():.1f}%; MAE {out.absolute_cfp_error.mean():.3f} CFP (must equal operational_draft.py)")
print(f"Phase 2, {n} measurements, A (operational draft) -> B (after routine review), movement types per step and CFP:")
print(f"  step-level movement-type sequence unchanged: {int(out.AB_unchanged_types.sum())}   changed: {len(ch)}  (added/removed movements: {int(out.AB_added_or_removed.sum())}; reclassified: {int(out.AB_reclassified.sum())}; CFP total changed: {int(out.AB_cfp_changed.sum())})")
print(f"  by project (changed / n): " + ", ".join(f"{p} {int((g.AB_unchanged_types==0).sum())}/{len(g)}" for p, g in out.groupby("project")))
s = out[out.spot_check == 1]
print(f"  52 spot-checked: A->B changed {int((s.AB_unchanged_types==0).sum())}; B->C changed {int(pd.to_numeric(s.BC_changed).sum())} (CFP changed {int(pd.to_numeric(s.BC_cfp_changed).sum())}); "
      f"A == C in {int(pd.to_numeric(s.A_equals_C).sum())}")
both = s[(pd.to_numeric(s.BC_changed) == 1)]
print("  the 2 errors that survived review: present in A and unchanged by B (A -> B unchanged):",
      [(m, bool(out.set_index('measurement_id').loc[m, 'AB_unchanged_types'])) for m in both.measurement_id])
print("Perimeter: observed changes to movement types and CFP only. Measurements with an unchanged type sequence may still have required\n  changes to labels or other elements not in the released data; no claim of 'accepted without changes' or 'terminology-only' is made.")
print("Written: results/review_changes_phase2.csv")
