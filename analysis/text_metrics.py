#!/usr/bin/env python3
"""
text_metrics.py
===============
Exact procedure used to score the natural-language COSMIC elements (triggering
event, functional user, data group, object of interest) of one functional-process
analysis against its accepted reference. The confidential labels are not part
of the package; this module documents and implements the comparison so that it
can be re-applied to any analysis/reference pair.

Unit compared
-------------
One functional-process measurement. Both the manual analysis and the draft follow
the use case's numbered steps (the same grid as ref_steps/tool_steps). For each
element the analysis is reduced to the ORDERED LIST of that element's labels, one
per data movement, in step order (main scenario, then alternative and exception
flows in document order); a step without a data movement has empty label cells
and contributes nothing to the list. The triggering event is a single label per
functional process. Lists are NOT de-duplicated: a data group used by three
movements appears three times. The step grid also supports the step-level joint
agreement (type + DG + OOI per step) reported by step_agreement.py.

Sequence construction
---------------------
The list of labels is joined into one token sequence: each label is normalised
(Unicode NFKC, lower-cased, punctuation replaced by spaces, whitespace
collapsed; accents are kept because the labels are Italian) and split on
whitespace; the token lists are concatenated in row order with no boundary
token (a bigram may therefore straddle two adjacent labels; this is the plain
concatenation also used in the original study). A row that is present but
whose cell is empty or a dash contributes the single token "<none>" at that
row's position. A row that the analysis does not contain at all contributes
nothing: the sequence is simply shorter, because no explicit row alignment is
performed and there is no step index from which the position of a missing row
could be reconstructed. Alignment is implicit in the sequence, exactly as for
the data-movement types, so a missing row is penalised as a gap (ROUGE-L) or
as missing n-grams and brevity (BLEU-2), and an extra row as an insertion.

Metrics (same implementation as sequence_metrics.py; no external library)
-------------------------------------------------------------------------
* ROUGE-L F1: LCS-based precision/recall/F1 over the token sequences.
* BLEU-2: geometric mean of clipped unigram and bigram precision, weights
  (0.5, 0.5), with the standard brevity penalty, NO smoothing. If the
  reference has fewer than two tokens (a one-word triggering event), the
  bigram term is undefined and BLEU-2 falls back to unigram BLEU; this is the
  only special case, and it is why a one-word exact match scores 1.0 rather
  than 0.
* BERTScore F1: bert_score 0.3.13, model_type="bert-base-multilingual-cased",
  num_layers=9 (the package default for that model), idf=False,
  rescale_with_baseline=False, lang unset (model_type given explicitly),
  computed on the joined label strings (labels separated by "; ", no "<none>" token; an omitted element is scored against the empty string,
  which yields 0).

Consequence for interpretation: these scores measure the similarity of the
element's labels in sequence; they do not verify the association between each
movement and its data group beyond what the shared row order implies.
"""
from __future__ import annotations
import re
import unicodedata
from typing import Sequence

from sequence_metrics import rouge_l, bleu_bigram, bleu_unigram

NONE = "<none>"
_punct = re.compile(r"[^\w\s]", re.UNICODE)


def normalise(label: str | None) -> list[str]:
    if label is None:
        return [NONE]
    s = unicodedata.normalize("NFKC", str(label)).lower()
    s = _punct.sub(" ", s)
    toks = s.split()
    return toks if toks and toks != ["-"] else [NONE]


def label_match(ref_label: str | None, tool_label: str | None) -> bool:
    """Textual equality of two labels after `normalise` (the ONLY normalisation used
    anywhere in the analysis: NFKC, lower-case, every non-alphanumeric character
    replaced by a space, whitespace-tokenised; accents kept). No synonym list, no
    stemming, no semantic equivalence."""
    return normalise(ref_label) == normalise(tool_label)


def step_joint_reproduced(ref_step: Sequence[tuple], tool_step: Sequence[tuple]) -> bool:
    """Joint agreement for ONE use-case step. Each argument is the ordered list of the
    step's data movements as (type, data_group, object_of_interest) tuples; an empty
    list is a step without data movements.
    Reproduced iff: same number of movements; at every movement position the type is
    identical; and at every position label_match holds for the data group and for the
    object of interest. A reference-empty step is reproduced iff the draft step is empty."""
    if len(ref_step) != len(tool_step):
        return False
    return all(rt == tt and label_match(rd, td) and label_match(ro, to)
               for (rt, rd, ro), (tt, td, to) in zip(ref_step, tool_step))


def to_sequence(labels: Sequence[str | None]) -> list[str]:
    out: list[str] = []
    for lab in labels:
        out.extend(normalise(lab))
    return out


def score_element(ref_labels: Sequence[str | None], cand_labels: Sequence[str | None]) -> dict:
    r, c = to_sequence(ref_labels), to_sequence(cand_labels)
    b2 = bleu_bigram(r, c) if len(r) >= 2 else bleu_unigram(r, c)
    return {"rougeL_f1": rouge_l(r, c)["f1"], "bleu2": b2, "ref_tokens": len(r), "cand_tokens": len(c)}


def bertscore_strings(labels: Sequence[str | None]) -> str:
    return "; ".join(str(l).strip() for l in labels if l not in (None, "", "-"))


def bertscore_f1(ref_labels, cand_labels):
    """Requires bert-score==0.3.13 and the multilingual model."""
    from bert_score import score  # type: ignore
    r, c = bertscore_strings(ref_labels), bertscore_strings(cand_labels)
    if not c or not r:
        return 0.0
    _, _, f1 = score([c], [r], model_type="bert-base-multilingual-cased", num_layers=9,
                     idf=False, rescale_with_baseline=False, verbose=False)
    return float(f1[0])


if __name__ == "__main__":
    print("step joint (identical)     ", step_joint_reproduced([("E", "Paziente", "Paziente"), ("X", "Messaggio di errore", "Paziente")],
                                                              [("E", "paziente", "PAZIENTE"), ("X", "messaggio di errore", "paziente")]))
    print("step joint (synonym -> no) ", step_joint_reproduced([("R", "Prenotazione", "Prenotazione")], [("R", "Testata prenotazione", "Prenotazione")]))
    print("step joint (order matters) ", step_joint_reproduced([("E", "A", "A"), ("X", "B", "B")], [("X", "B", "B"), ("E", "A", "A")]))
    print("step joint (both empty)    ", step_joint_reproduced([], []))
    ref = ["Infermiere", "Infermiere", "Infermiere"]
    print("FU exact          ", score_element(ref, ref))
    print("FU generic actor  ", score_element(ref, ["Utente", "Utente", "Utente"]))
    print("TE one word exact ", score_element(["Login"], ["Login"]))
    print("DG synonym        ", score_element(["Prenotazione", "Testata prenotazione"], ["Prenotazione", "Prenotazione"]))
    print("DG missing row    ", score_element(["Paziente", "Reparto", "Referto"], ["Paziente", "Referto"]))
