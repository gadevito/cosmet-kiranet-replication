#!/usr/bin/env python3
"""Automated processing time per functional process (Sec. VII-B), from the client-side
timings recorded for every execution: e2e_seconds = Splitter share + prefill +
Analyzer generation + parsing (warm server, cached few-shot prefix, one request at a
time; human review excluded). Decoding throughput = completion tokens (incl.
reasoning) / generation time, per Analyzer request."""
import os, numpy as np, pandas as pd
here = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(here, "..", "data", "scores_per_measurement.csv"))
lat = df.e2e_seconds; tok = df.analyzer_completion_tokens; thr = df.analyzer_decode_tok_s
q = np.percentile(lat, [10, 25, 50, 75, 90, 95, 99])
print(f"end-to-end seconds per functional process over {len(df):,} executions: "
      f"median {q[2]:.1f}, IQR {q[1]:.1f}-{q[3]:.1f}, P10-P90 {q[0]:.1f}-{q[4]:.1f}, P95 {q[5]:.1f}, P99 {q[6]:.1f}, max {lat.max():.1f}, mean {lat.mean():.2f}")
print(f"Analyzer completion tokens (reasoning + answer): median {tok.median():.0f}, P10-P90 {np.percentile(tok,10):.0f}-{np.percentile(tok,90):.0f}, max {tok.max()} (limit 8,192; no truncation)")
print(f"Analyzer decoding throughput: mean per request {thr.mean():.0f} tok/s; completion-weighted {tok.sum()/(tok/thr).sum():.0f} tok/s")
print("median seconds by project:", df.groupby("project").e2e_seconds.median().round(1).to_dict())
pd.DataFrame([{"n_exec": len(df), "median_s": q[2], "p10_s": q[0], "p90_s": q[4], "p99_s": q[6], "max_s": lat.max(),
               "mean_s": lat.mean(), "median_tokens": tok.median(), "max_tokens": int(tok.max()), "mean_tok_s": thr.mean()}]).to_csv(
    os.path.join(here, "..", "results", "timing.csv"), index=False)
