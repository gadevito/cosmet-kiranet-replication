#!/usr/bin/env python3
"""Regenerate the paper's RQ1 tables (concept-similarity table = tab:rq1 (not in the paper), Table I CIs = tab:rq1ci) and the disaggregated workload table as LaTeX from the
derived scores. Run bootstrap_ci.py first (or let run_all.sh do it)."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bootstrap_ci import per_measurement, cluster_bootstrap

here = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(here, "..", "data", "scores_per_measurement.csv"))
pm = per_measurement(df)
ci = pd.read_csv(os.path.join(here, "..", "results", "rq1_confidence_intervals.csv"))
ci["key"] = ci.metric.str.replace(r"\s+", " ", regex=True).str.strip()

concepts = [("Triggering event", "te"), ("Functional user", "fu"), ("Data group", "dg"),
            ("Object of interest", "ooi"), ("Data movement", "dm")]
lines = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
         r"COSMIC concept & ROUGE-L F1 & BLEU-2 & BERTScore F1 \\", r"\midrule"]
for name, k in concepts:
    rl = pm[f"{k}_rougeL_f1"]
    bl = pm[f"{k}_bleu2"]
    bs = pm[f"{k}_bertscore_f1"] if f"{k}_bertscore_f1" in pm else None
    f = lambda s: f"{s.mean():.3f} ({s.std(ddof=1):.3f})"
    lines.append(f"{name:<18} & {f(rl)} & {f(bl)} & {f(bs) if bs is not None else '---'} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
tab1 = "\n".join(lines)

lines = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
         r"COSMIC concept & Mean F1 & 95\% CI & GPT-4~\cite{cosmet} & In CI? \\", r"\midrule"]
for name, k in concepts:
    r = ci[ci.key.str.startswith(name) & ci.key.str.contains("ROUGE-L")].iloc[0]
    lines.append(f"{name:<18} & {r['mean']:.3f} & [{r.ci95_low:.3f}, {r.ci95_high:.3f}] & {float(r.tse_gpt4):.3f} & {r.tse_in_ci} \\\\")
lines.append(r"\midrule")
r = ci[ci.key.str.contains("MAE")].iloc[0]
lines.append(f"Data movement (MAE) & {float(r['mean']):.3f} & [{r.ci95_low:.3f}, {r.ci95_high:.3f}] & {float(r.tse_gpt4):.3f} & {r.tse_in_ci} \\\\")
r = ci[ci.key.str.contains("MdAE")].iloc[0]
lines.append(f"Data movement (MdAE) & {float(r['mean']):.0f} & --- (zero in every resample) & 0 & yes \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
tab2 = "\n".join(lines)

out = os.path.join(here, "..", "results", "tables.tex")
open(out, "w").write("% tab:rq1 (concept similarity, full matrix; not in the paper)\n" + tab1 + "\n\n% tab:rq1ci (CIs, Table I)\n" + tab2 + "\n")
print(tab1); print(); print(tab2); print(f"\nWritten: {os.path.relpath(out)}")

# ---- full per-concept similarity matrix (old appendix table): mean, SD over measurements (after averaging the 10 runs), 95% CI
ci = pd.read_csv(os.path.join(here, "..", "results", "rq1_confidence_intervals.csv"))
def row(label, rl, b2, bs):
    g = lambda m: ci[ci.metric.str.strip() == m].iloc[0]
    r = g(rl); b = g(b2); s = g(bs) if bs else None
    return {"concept": label, "rougeL_f1": r["mean"], "rougeL_sd": r["sd"], "rougeL_ci_low": r["ci95_low"], "rougeL_ci_high": r["ci95_high"],
            "bleu2": b["mean"], "bleu2_sd": b["sd"], "bleu2_ci_low": b["ci95_low"], "bleu2_ci_high": b["ci95_high"],
            "bertscore_f1": s["mean"] if s is not None else "", "bertscore_sd": s["sd"] if s is not None else ""}
full = pd.DataFrame([row("Triggering event", "Triggering event  ROUGE-L F1", "Triggering event  BLEU-2", None) if False else
                     {"concept": "Triggering event", "rougeL_f1": 1.0, "rougeL_sd": 0.0, "rougeL_ci_low": 1.0, "rougeL_ci_high": 1.0, "bleu2": 1.0, "bleu2_sd": 0.0, "bleu2_ci_low": 1.0, "bleu2_ci_high": 1.0, "bertscore_f1": 1.0, "bertscore_sd": 0.0}])
pmc = pm.copy()
for lab, k in [("Functional user", "fu"), ("Data group", "dg"), ("Object of interest", "ooi"), ("Data movement", "dm")]:
    d = {"concept": lab}
    for met, col in [("rougeL", f"{k}_rougeL_f1"), ("bleu2", f"{k}_bleu2"), ("bertscore", f"{k}_bertscore_f1")]:
        if col not in pmc.columns: d[f"{met}_f1" if met != "bleu2" else "bleu2"] = ""; d[f"{met}_sd"] = ""; continue
        v = pmc[col]; lo, hi = cluster_bootstrap(pmc, col) if v.nunique() > 1 else (v.mean(), v.mean())
        d[f"{met}_f1" if met != "bleu2" else "bleu2"] = round(v.mean(), 4); d[f"{met}_sd"] = round(v.std(ddof=1), 4)
        if met in ("rougeL", "bleu2"): d[f"{met}_ci_low"] = round(lo, 4); d[f"{met}_ci_high"] = round(hi, 4)
    full = pd.concat([full, pd.DataFrame([d])], ignore_index=True)
full.to_csv(os.path.join(here, "..", "results", "concept_similarity_full.csv"), index=False)
print("results/concept_similarity_full.csv: SD = dispersion across measurements after averaging the 10 executions (not run-to-run variability); CIs = project-cluster bootstrap")
print(full.to_string(index=False))

# ---- disaggregated workloads (tab:disagg): distinct workloads, NOT a paired comparison
led = pd.read_csv(os.path.join(here, "..", "data", "effort_ledger.csv"))
led = led[led.in_company_ledger == 1]
led = led[~led.activity.str.contains("spot-check|setup|adjudication")].copy()
led["phase"] = led.phase.astype(int)
acc = pm.groupby("project").agg(rl=("dm_rougeL_f1", "mean"), mae=("abs_err_cfp", "mean"))
L = [r"\begin{tabular}{@{}llrrrrrcc@{}}", r"\toprule",
     r"Project & Workflow & Meas. & CFP & Person-days & CFP/pd & Reduction$^{a}$ & CosMet F1$^{b}$ & CosMet MAE$^{b}$ \\", r"\midrule"]
def wf(ph): return {0: "Manual (Phase 0)", 1: "Manual (Phase 1)", 2: "Assisted (Phase 2)"}[int(ph)]
for p in ["P1", "P2", "P3", "P4", "P5"]:
    rows_p = led[led.project == p].sort_values("phase")
    first = True
    for _, r in rows_p.iterrows():
        thr = r.cfp / r.person_days_actual
        red = "---" if int(r.phase) < 2 else f"{100*(1-150/thr):.0f}\\%"
        a = f"{acc.loc[p,'rl']:.3f} & {acc.loc[p,'mae']:.3f}" if first else "&"
        L.append(f"{p if first else ''} & {wf(r.phase)} & {int(r.measurements)} & {int(r.cfp):,} & {r.person_days_actual:.2f} & {thr:.0f} & {red} & {a} \\\\")
        first = False
    L.append(r"\addlinespace[2pt]")
man = led[led.phase < 2]; ass = led[led.phase == 2]
pd_ass = ass.person_days_actual.sum(); red_all = 1 - 150 / (ass.cfp.sum() / pd_ass)   # from unrounded person-days
L += [r"\midrule",
      f"All manual & Phases 0--1 & {int(man.measurements.sum())} & {int(man.cfp.sum()):,} & {man.person_days_actual.sum():.1f} & {man.cfp.sum()/man.person_days_actual.sum():.0f} & --- & & \\\\",
      f"All assisted & Phase 2 (P2--P5 only) & {int(ass.measurements.sum())} & {int(ass.cfp.sum()):,} & {pd_ass:.2f} & {ass.cfp.sum()/pd_ass:.0f} & {100*red_all:.1f}\\% & & \\\\",
      r"\bottomrule", r"\end{tabular}"]
tab3 = "\n".join(L)
open(out, "a").write("\n% tab:disagg\n" + tab3 + "\n")
print(); print(tab3)
print(f"\nAdoption-period reconstruction: manual {man.person_days_actual.sum():.1f} + assisted {pd_ass:.2f} + spot-checks 2.3 = {man.person_days_actual.sum()+pd_ass+2.3:.2f} pd before the company's setup share")
