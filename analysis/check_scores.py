#!/usr/bin/env python3
"""Recompute ROUGE-L, BLEU-2 (on the movement-type sequence in step order), matched movements
(per-step), CFP and |err| from the stored step-anchored analyses (ref_steps / tool_steps) and verify
they equal the stored per-run scores; also checks ref_sequence/tool_sequence == movement sequences."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sequence_metrics import rouge_l, bleu_bigram, parse_steps, movement_sequence, cfp, matched_movements
here = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(here, "..", "data", "scores_per_measurement.csv"))
R = df.ref_steps.map(parse_steps); C = df.tool_steps.map(parse_steps)
rl = [rouge_l(movement_sequence(r), movement_sequence(c))["f1"] for r, c in zip(R, C)]
b2 = [bleu_bigram(movement_sequence(r), movement_sequence(c)) for r, c in zip(R, C)]
mm = [matched_movements(r, c) for r, c in zip(R, C)]
rc = [cfp(r) for r in R]; tc = [cfp(c) for c in C]
ok = (np.allclose(rl, df.dm_rougeL_f1, atol=1e-6) and np.allclose(b2, df.dm_bleu2, atol=1e-6)
      and (np.array(mm) == df.matched_dm).all() and (np.array(rc) == df.ref_dm).all() and (np.array(tc) == df.tool_dm).all()
      and (np.abs(np.array(tc) - np.array(rc)) == df.abs_err_cfp).all()
      and all(len(r) == len(c) for r, c in zip(R, C))
      and (df.ref_sequence == R.map(lambda r: "".join(movement_sequence(r)))).all()
      and (df.tool_sequence == C.map(lambda c: "".join(movement_sequence(c)))).all())
print(f"{len(df)} rows checked (same step grid on both sides, scores recomputed from steps): {'OK' if ok else 'MISMATCH'}")
sys.exit(0 if ok else 1)
