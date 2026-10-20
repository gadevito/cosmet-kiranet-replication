#!/usr/bin/env python3
"""
economics.py
============
The parameterised model behind the paper's economics table (RQ1, cost part).
Every assumption is a CLI flag so reviewers can stress-test it.

Two distinct things are computed, and the paper keeps them apart:

1. SCENARIO ("what the six-month portfolio would cost either way")
   * loaded measurer cost EUR 4,800/month, 20 working days -> EUR 240/day
   * manual throughput 150 CFP/day: the company's own logged rate over the
     4,800 CFP measured manually in Phases 0-1 (inside the 125-500 CFP/day
     industry band)
   * portfolio ~6,000 CFP -> 40 person-days manually
   * assisted effort: 60-85% reduction = MIN and MAX of the per-project
     reductions observed in Phase 2 (rate-based, non-paired), -> 6-16 pd
   * GROSS saving on the portfolio EUR 5.8k-8.2k; annualised EUR 12k-16k
     = 0.20-0.28 FTE of freed capacity. This is a projection, not cash saved.

2. ADOPTION LEDGER ("what the company actually spent over the 8 months")
   Read from data/effort_ledger.csv, logged values: 32.0 pd manual (P1 in
   Phase 0, maintenance in Phase 1) + 2.04 pd assisted (Phase 2) + 2.3 pd of
   independent spot-checks + 1.0 pd Phase 1 discrepancy adjudication (4 sessions
   x ~1 h x 2 measurers = ~8 person-hours, booked on the collaboration code)
   + 1.0 pd company share of setup = 38.3 pd, i.e. "approximately 38
   person-days" (Sec. VIII-E) vs the 40 pd fully manual counterfactual.
   Adjudication and the setup share are estimates.
   The academic authors absorbed ~13 pd of Phase 1 validation effort (not
   included, and distinct from the 24-40 h of setup in Sec. V-B).

Usage:
    python analysis/economics.py [--total_cfp 6000 --eur_per_month 4800 ...]
"""
from __future__ import annotations
import argparse, os
import pandas as pd


def scenario(total_cfp=6000., eur_per_month=4800., wd=20., rate=150., red_lo=.60, red_hi=.85, months=6.):
    day = eur_per_month / wd
    man_pd = total_cfp / rate
    ass_lo, ass_hi = man_pd * (1 - red_hi), man_pd * (1 - red_lo)
    sav_lo, sav_hi = (man_pd - ass_hi) * day, (man_pd - ass_lo) * day
    ann = 12. / months
    freed_lo, freed_hi = (man_pd - ass_hi) * ann, (man_pd - ass_lo) * ann
    return {
        "Portfolio size": f"{total_cfp:,.0f} CFP over {months:.0f} months",
        "Loaded measurer cost": f"EUR {eur_per_month:,.0f}/month = EUR {day:,.0f}/day",
        "Manual throughput (logged, Phases 0-1)": f"{rate:.0f} CFP/day",
        "Manual effort / cost": f"{man_pd:.0f} pd / EUR {man_pd*day:,.0f}  (EUR {day/rate:.2f}/CFP)",
        "Phase-2 reduction (min-max across projects)": f"{red_lo:.0%}-{red_hi:.0%}",
        "Assisted effort / cost": f"{ass_lo:.0f}-{ass_hi:.0f} pd / EUR {ass_lo*day:,.0f}-{ass_hi*day:,.0f}  (EUR {day/rate*(1-red_hi):.2f}-{day/rate*(1-red_lo):.2f}/CFP)",
        "Released capacity, this portfolio (scenario)": "24-34 pd",
        "Gross saving, this portfolio (scenario)": f"EUR {sav_lo:,.0f}-{sav_hi:,.0f}",
        "Annualised gross saving": f"EUR {sav_lo*ann:,.0f}-{sav_hi*ann:,.0f}",
        "Freed capacity per year": f"{freed_lo:.0f}-{freed_hi:.0f} pd = {freed_lo/240:.2f}-{freed_hi/240:.2f} FTE",
        "Marginal cost per run": "electricity only (no per-token charge, no licence)",
    }


def ledger(path):
    df = pd.read_csv(path)
    adj = df[df.phase.astype(str) == "adjudication"].person_days_actual.sum()
    comp = df[(df.in_company_ledger == 1) & (df.phase.astype(str) != "adjudication")]
    spot = df[df.activity.str.contains("spot-check")].person_days_low.sum()
    p2 = comp[comp.phase.astype(str) == "2"]
    p2 = p2[~p2.activity.str.contains("spot-check")]
    authors = df[(df.who == "academic authors") & (df.phase.astype(str) == "1")].person_days_actual.sum()   # validation only, no setup
    setup_total = df[(df.phase == "setup") & (df.in_company_ledger == 0)]
    setup_company = comp[comp.phase == "setup"].person_days_actual.sum()   # the company's 1.0 pd share of the 24-40 h setup
    # product-specific sensitivity: reduction of each maintenance product against ITS OWN Phase 1 manual baseline,
    # 1 - (E_assisted/CFP_assisted) / (E_manual/CFP_manual), from the recorded person-days (no rounded throughputs)
    man = comp[(comp.phase.astype(str) == "1") & (comp.who == "company")].set_index("project")
    own = {p: 1 - (r.person_days_actual / r.cfp) / (man.loc[p, "person_days_actual"] / man.loc[p, "cfp"]) for p, r in p2.set_index("project").iterrows()}
    own_txt = ", ".join(f"{p} {100*v:.1f}%" for p, v in own.items()) + f"  (range {100*min(own.values()):.0f}-{100*max(own.values()):.0f}%, rounded after computing)"
    return {
        "Reduction vs each product's own Phase 1 manual baseline (Sec. VII-B sensitivity; the table uses the pooled 150 CFP/pd)": own_txt,
        "Phase 2 effort including the 52 independent checks (Sec. VII-B)":
            f"{p2.person_days_actual.sum():.2f} assisted + {spot:.1f} checks = {p2.person_days_actual.sum()+spot:.2f} pd for 1,200 CFP, {100*(1-(p2.person_days_actual.sum()+spot)/8):.1f}% below the 8 pd implied by the pooled manual baseline (150 CFP/pd)",
        "Company effort actually spent (Phases 0-2, spot-checks, setup share)":
            f"{comp.person_days_actual.sum():.1f} pd (before adjudication) = "
            f"{comp[comp.phase.astype(str).isin(['0','1'])].person_days_actual.sum():.1f} manual + {p2.person_days_actual.sum():.2f} assisted + {spot:.1f} spot-checks + {comp[comp.phase=='setup'].person_days_actual.sum():.1f} setup share",
        "Company Phase 1 discrepancy adjudication (adoption cost, not under FSM, excluded from throughput)":
            f"{adj:.1f} pd = 4 sessions x ~1 h x 2 company measurers = ~8 person-hours (110 Phase 1 discrepancies)",
        "Company total (the paper's approximately 38 person-days)":
            f"{comp.person_days_actual.sum() + adj:.1f} pd = 32.0 + 2.04 + 2.3 + 1.0 adjudication + 1.0 setup share -> approximately 38 (Sec. VIII-E of the paper)",
        "Adoption-period total including academic validation and the non-company setup share (Sec. VIII-E)":
            f"{comp.person_days_actual.sum() + adj + authors + (setup_total.person_days_low.sum() - setup_company):.1f}-{comp.person_days_actual.sum() + adj + authors + (setup_total.person_days_high.sum() - setup_company):.1f} pd = 38.3 + {authors:.0f} academic validation + {setup_total.person_days_low.sum() - setup_company:.0f}-{setup_total.person_days_high.sum() - setup_company:.0f} non-company setup (24-40 h minus the company's 1.0 pd share) -> approximately 53-55",
        "  of which Phase 2 assisted measurement (1,200 CFP)": f"{p2.person_days_actual.sum():.2f} pd logged (manual-rate equivalent 8 pd; pooled reduction {100*(1-p2.person_days_actual.sum()/8):.1f}%)",
        "  of which spot-checks (study cost)": f"{spot:.1f} pd",
        "Manual counterfactual for the same 6,000 CFP": "40 pd",
        "Phase 1 validation effort borne by the academic authors (excluded; distinct from setup)": f"~{authors:.0f} pd",
        "One-off setup (Sec. V-B, all parties)": f"{setup_total.person_days_low.sum()*8:.0f}-{setup_total.person_days_high.sum()*8:.0f} h = EUR {setup_total.person_days_low.sum()*8*30:,.0f}-{setup_total.person_days_high.sum()*8*30:,.0f} at EUR 30/h (reported separately, not deducted)",
        "Hardware": "excluded: the company already operated the inference workstation; the EUR 6,000-8,000 purchase price is hypothetical, not a cost borne by the company",
        "Scenario assumptions": "applies the Phase 2 reduction observed on maintenance work to P1 as well; excludes setup, parallel-run validation, adjudication, independent spot-checks and operating costs; the annualised saving and FTE lines of the scenario block are not in the paper",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--total_cfp", type=float, default=6000.)
    ap.add_argument("--eur_per_month", type=float, default=4800.)
    ap.add_argument("--working_days_per_month", type=float, default=20.)
    ap.add_argument("--manual_rate_cfp_day", type=float, default=150.)
    ap.add_argument("--reduction_low", type=float, default=.60)
    ap.add_argument("--reduction_high", type=float, default=.85)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--ledger", default=os.path.join(here, "..", "data", "effort_ledger.csv"))
    ap.add_argument("--output", default=os.path.join(here, "..", "results", "economics.csv"))
    a = ap.parse_args()
    t = scenario(a.total_cfp, a.eur_per_month, a.working_days_per_month, a.manual_rate_cfp_day, a.reduction_low, a.reduction_high)
    l = ledger(a.ledger)
    os.makedirs(os.path.dirname(a.output), exist_ok=True)
    pd.DataFrame([("scenario", k, v) for k, v in t.items()] + [("ledger", k, v) for k, v in l.items()],
                 columns=["block", "item", "value"]).to_csv(a.output, index=False)
    w = max(len(k) for k in list(t) + list(l))
    print("\nSCENARIO: economics of the six-month portfolio")
    for k, v in t.items(): print(f"  {k:<{w}}  {v}")
    print("\nADOPTION LEDGER: what was actually spent")
    for k, v in l.items(): print(f"  {k:<{w}}  {v}")
    print(f"\nWritten: {os.path.relpath(a.output)}")


if __name__ == "__main__":
    main()
