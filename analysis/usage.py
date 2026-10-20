#!/usr/bin/env python3
"""
usage.py
========
Temporal distribution of the Phase 2 (tool-first) usage, from data/usage_log.csv.

The log has one row per Phase 2 measurement (200 rows) with the use-case
document VERSION it belongs to (release-specific: a document revised and
re-measured for the next release is a new version), the working day, the ISO
calendar week (Mon-Sun; a week straddling two months is counted in both months,
hence 5+5+4 = 14 month-level active weeks vs 13 overall), the measurer who ran
CosMet and reviewed the draft, and the working session (one assisted-measurement
timesheet entry: one measurer, one day, one product/release activity code). It answers "how intensely was
the tool used", which the calendar length of the adoption (Nov 2025 - Jun 2026)
does not: the assisted measurement effort of Phase 2 is 2.04 person-days in
total (data/effort_ledger.csv), unevenly distributed across the five measurers.

Outputs (results/usage_by_month.csv, results/usage_by_measurer.csv):
  * per calendar month: measurements, document versions, sessions, active measurers
  * per measurer: measurements, document versions, sessions, active weeks, first/last date
  * interaction frequency: measurements per session, sessions per active week
"""
from __future__ import annotations
import os
import pandas as pd

here = os.path.dirname(os.path.abspath(__file__))
log = pd.read_csv(os.path.join(here, "..", "data", "usage_log.csv"), parse_dates=["date"])
res = os.path.join(here, "..", "results"); os.makedirs(res, exist_ok=True)

print(f"Phase 2 usage log: {len(log)} measurements, {log.document_id.nunique()} document versions, "
      f"{log.session_id.nunique()} sessions, {log.week.nunique()} active weeks of "
      f"{len(pd.period_range(log.date.min(), log.date.max(), freq='W'))} "
      f"({log.date.min().date()} -> {log.date.max().date()}), {log.measurer.nunique()} measurers")

log["month"] = log.date.dt.strftime("%Y-%m")
by_month = log.groupby("month").agg(measurements=("measurement_id", "size"), document_versions=("document_id", "nunique"),
                                    sessions=("session_id", "nunique"), active_weeks=("week", "nunique"),
                                    measurers=("measurer", "nunique")).reset_index()
by_month.to_csv(os.path.join(res, "usage_by_month.csv"), index=False)
print("\nPer month"); print(by_month.to_string(index=False))

sess = log.groupby("session_id").agg(measurer=("measurer", "first"), week=("week", "first"), n=("measurement_id", "size"))
by_meas = log.groupby("measurer").agg(measurements=("measurement_id", "size"), document_versions=("document_id", "nunique"),
                                      sessions=("session_id", "nunique"), active_weeks=("week", "nunique"),
                                      first=("date", "min"), last=("date", "max")).reset_index()
by_meas["meas_per_session_median"] = by_meas.measurer.map(sess.groupby("measurer").n.median())
by_meas["sessions_per_active_week"] = (by_meas.sessions / by_meas.active_weeks).round(2)
by_meas["first"] = by_meas["first"].dt.date; by_meas["last"] = by_meas["last"].dt.date
by_meas.to_csv(os.path.join(res, "usage_by_measurer.csv"), index=False)
print("\nPer measurer"); print(by_meas.to_string(index=False))

spw = log.groupby("week").session_id.nunique()
print(f"\nMeasurements per session: median {sess.n.median():.0f}, IQR {sess.n.quantile(.25):.0f}-{sess.n.quantile(.75):.0f}, max {sess.n.max()}")
print(f"Sessions per active week: median {spw.median():.0f}, range {spw.min()}-{spw.max()}")
print("Per product/release (measurements, primary measurer):")
print(log.groupby(["project", "release"]).agg(measurements=("measurement_id", "size"), document_versions=("document_id", "nunique"),
      first=("date", "min"), last=("date", "max"), primary_measurer=("measurer", lambda s: s.mode()[0])).to_string())
if "session_person_days" in log.columns:
    sp = log.drop_duplicates("session_id").groupby("project").session_person_days.sum()
    print("Session timesheet entries summed per project (must equal the Phase 2 ledger rows):", {k: round(v, 2) for k, v in sp.items()}, "total", round(sp.sum(), 2))
sc = pd.read_csv(os.path.join(here, "..", "data", "scores_per_measurement.csv"))
r1 = sc[sc.run == 1].sort_values("measurement_id").copy(); r1["remeasured_in_period"] = r1.groupby("fp_id").cumcount() > 0
rem = r1[r1.phase == 2].groupby("project").remeasured_in_period.agg(["sum", "size"])
print("\nRe-measurements within Jan-Jun 2026 (process already measured once in the period): by phase",
      r1.groupby("phase").remeasured_in_period.sum().to_dict(), "; Phase 2 by project (n / measurements):",
      {p: f"{int(a)}/{int(b)} ({100*a/b:.0f}%)" for p, (a, b) in rem.iterrows()})
print(f"\nWritten: results/usage_by_month.csv, results/usage_by_measurer.csv")
