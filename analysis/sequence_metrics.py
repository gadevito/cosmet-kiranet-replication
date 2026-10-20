#!/usr/bin/env python3
"""
sequence_metrics.py
===================
Reference implementations of the two data-movement sequence metrics used in RQ1,
so that others can recompute per-functional-process scores on their own data
(the inputs to bootstrap_ci.py). Both treat a functional process as an ORDERED
sequence of data-movement labels drawn from {E, X, R, W}.

* ROUGE-L (precision, recall, F1): longest-common-subsequence overlap. Rewards
  movements recovered in the right order, tolerates a missed/inserted sub-process.
* BLEU with weights (0.5, 0.5): clipped unigram+bigram precision with the standard
  brevity penalty, i.e. nltk's sentence_bleu(weights=(0.5, 0.5)). The bigram term
  is zero unless consecutive movements appear in the reference order.
Both are order-sensitive, which is what the COSMIC data-movement flow requires.
ROUGE-1 / unigram-BLEU are also provided, for reference only.

BERTScore (used for the natural-language COSMIC elements, not for the E/X/R/W
labels) is intentionally NOT reimplemented here: it requires a pretrained model.
A thin optional wrapper is provided that calls `bert_score` if it is installed.

Run a self-test with:
    python sequence_metrics.py --selftest
"""
from __future__ import annotations

import argparse
import math
from collections import Counter
from typing import Sequence


def rouge1(reference: Sequence[str], candidate: Sequence[str]) -> dict:
    """Unigram ROUGE-1 precision/recall/F1 over label multisets."""
    ref, cand = Counter(reference), Counter(candidate)
    overlap = sum((ref & cand).values())
    p = overlap / len(candidate) if candidate else 0.0
    r = overlap / len(reference) if reference else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def bleu_unigram(reference: Sequence[str], candidate: Sequence[str]) -> float:
    """BLEU with weights (1,0,0,0): clipped unigram precision x brevity penalty."""
    c, r = len(candidate), len(reference)
    if c == 0:
        return 0.0
    ref, cand = Counter(reference), Counter(candidate)
    clipped = sum(min(cand[w], ref[w]) for w in cand)
    precision = clipped / c
    bp = 1.0 if c > r else math.exp(1 - r / c) if c > 0 else 0.0
    return bp * precision


def bleu_bigram(reference, candidate):
    """BLEU with w=(0.5,0.5): geometric mean of unigram and bigram precision,
    times the brevity penalty. ORDER-SENSITIVE (the bigram term vanishes unless
    consecutive labels match the reference order)."""
    import math as _m
    c, r = len(candidate), len(reference)
    if c == 0:
        return 0.0
    def _prec(n):
        cn = Counter(tuple(candidate[i:i+n]) for i in range(len(candidate)-n+1))
        rn = Counter(tuple(reference[i:i+n]) for i in range(len(reference)-n+1))
        tot = sum(cn.values())
        if tot == 0:
            return 0.0
        return sum(min(v, rn[g]) for g, v in cn.items()) / tot
    p1, p2 = _prec(1), _prec(2)
    if p1 == 0 or p2 == 0:
        return 0.0
    bp = 1.0 if c > r else _m.exp(1 - r / c)
    return bp * _m.exp(0.5 * _m.log(p1) + 0.5 * _m.log(p2))


def _lcs_len(a, b):
    m = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, x in enumerate(a, 1):
        for j, y in enumerate(b, 1):
            m[i][j] = m[i-1][j-1] + 1 if x == y else max(m[i-1][j], m[i][j-1])
    return m[-1][-1]


def rouge_l(reference, candidate) -> dict:
    """ROUGE-L (Lin 2004): precision/recall/F1 on the longest common subsequence.
    ORDER-SENSITIVE, tolerant of gaps (a missed or inserted sub-process)."""
    l = _lcs_len(list(reference), list(candidate))
    p = l / len(candidate) if candidate else 0.0
    r = l / len(reference) if reference else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def positional_accuracy(reference: Sequence[str], candidate: Sequence[str]) -> float:
    """Fraction of positions where the labels agree (order-SENSITIVE).
    Length mismatch counts as disagreement on the surplus positions."""
    n = max(len(reference), len(candidate))
    if n == 0:
        return 1.0
    hits = sum(1 for i in range(min(len(reference), len(candidate)))
               if reference[i] == candidate[i])
    return hits / n


def normalised_edit_similarity(reference: Sequence[str], candidate: Sequence[str]) -> float:
    """1 - (Levenshtein distance / max length): order-SENSITIVE, tolerant of
    insertions/deletions (e.g. a missed or extra sub-process)."""
    r, c = list(reference), list(candidate)
    if not r and not c:
        return 1.0
    prev = list(range(len(c) + 1))
    for i, a in enumerate(r, 1):
        cur = [i]
        for j, b in enumerate(c, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a != b)))
        prev = cur
    return 1.0 - prev[-1] / max(len(r), len(c))


def bertscore_f1(references, candidates, lang: str = "en"):
    """Optional BERTScore F1 wrapper (requires `bert_score` + a model download)."""
    try:
        from bert_score import score  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "BERTScore requires the 'bert-score' package and a pretrained model. "
            "Install with: pip install bert-score"
        ) from exc
    _, _, f1 = score(candidates, references, lang=lang, verbose=False)
    return [float(x) for x in f1]


def _selftest() -> None:
    # Reference flow [E, R, W, X]; candidate transposes R/E (same labels, wrong order)
    ref = ["E", "R", "W", "X"]
    perfect = ["E", "R", "W", "X"]
    transposed = ["R", "E", "X", "W"]
    missing = ["E", "R", "W"]

    print(f"{'case':14s}{'ROUGE-1':>9s}{'BLEU-1':>8s} | {'ROUGE-L':>8s}{'BLEU-2':>8s}")
    print("  " + "-" * 48)
    for name, cand in [("perfect", perfect), ("transposed", transposed),
                       ("reversed", list(reversed(ref))), ("missing-one", missing)]:
        print(f"  {name:12s}{rouge1(ref, cand)['f1']:9.3f}{bleu_unigram(ref, cand):8.3f} | "
              f"{rouge_l(ref, cand)['f1']:8.3f}{bleu_bigram(ref, cand):8.3f}")
    print("\nLEFT  (ROUGE-1, BLEU-1): multiset metrics -- blind to order.")
    print("RIGHT (ROUGE-L, BLEU-2): order-sensitive -- these are our primary metrics.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        ap.print_help()


# ---------------------------------------------------------------------------
# Step-anchored representation (used by the Kiranet evaluation)
# ---------------------------------------------------------------------------
# A functional-process analysis is an ordered list of the use case's numbered STEPS;
# each step holds the ordered list of its data-movement types (possibly empty).
# Serialised as "E|N|R,X|W": steps separated by '|', movements within a step by
# ',', and 'N' for a step with no data movement. The reference and the draft share
# the same step grid, so steps are aligned by position without any LCS alignment.

def parse_steps(s: str) -> list:
    return [[] if st == "N" else st.split(",") for st in str(s).split("|")]

def movement_sequence(steps: list) -> list:
    """The sequence scored by ROUGE-L / BLEU-2: the data-movement types in step order.
    A step without a data movement contributes no symbol (its 'N' serves the step
    alignment only), so the length of this sequence is the CFP count."""
    return [sym for st in steps for sym in st]

def flatten(steps: list) -> list:
    """Step-aligned display form: one 'N' per empty step (NOT the scored sequence)."""
    return [sym for st in steps for sym in (st if st else ["N"])]

def cfp(steps: list) -> int:
    """Functional size = number of data movements (non-N symbols)."""
    return sum(len(st) for st in steps)

def matched_movements(ref_steps: list, tool_steps: list) -> int:
    """Reference movements reproduced in the SAME step (per-step LCS of types);
    steps are aligned by position because both analyses follow the use case's steps."""
    return sum(_lcs_len(r, c) for r, c in zip(ref_steps, tool_steps))

def step_type_agreement(ref_steps: list, tool_steps: list) -> float:
    """Fraction of steps whose movement-type list is reproduced exactly."""
    return sum(r == c for r, c in zip(ref_steps, tool_steps)) / len(ref_steps)
