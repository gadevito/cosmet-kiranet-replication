#!/usr/bin/env bash
# End-to-end reproduction of the paper's RQ1/RQ2 figures from the derived data in data/.
set -euo pipefail
cd "$(dirname "$0")"
echo "==> 0   calibration runs (Phase 0, Dec 2025): configurations tried and the retained one"
python3 analysis/calibration.py
echo; echo "==> 1/5 recompute the sequence metrics from the E/X/R/W sequences and check them against the stored scores"
python3 analysis/check_scores.py
echo; echo "==> 2/5 sequence-metric self-test"
python3 analysis/sequence_metrics.py --selftest
echo; echo "==> 2b  text-metric procedure self-test (labels are confidential; this shows the rules)"
(cd analysis && python3 text_metrics.py)
echo; echo "==> 3/5 RQ1 point estimates + cluster-bootstrap 95% CIs"
python3 analysis/bootstrap_ci.py --by-project
echo; echo "==> 3b  sensitivity: 852 measurements with an independent manual reference"
python3 analysis/bootstrap_ci.py --independent-only --output results/rq1_confidence_intervals_independent_only.csv
echo; echo "==> 3b' sensitivity: 841 measurements, excluding also the references corrected in the Phase 1 discrepancy review"
python3 analysis/bootstrap_ci.py --strict-independent --output results/rq1_confidence_intervals_strict_independent.csv
echo; echo "==> 3b'' post-review references only (148 measurements; point estimates)"
python3 analysis/bootstrap_ci.py --post-review-only --output results/rq1_point_estimates_post_review_only.csv
echo; echo "==> 3c  run-to-run repeatability of the raw output"
python3 analysis/repeatability.py
echo; echo "==> 3g  operational draft actually reviewed vs the ten executions (Phase 2, 200 measurements)"
python3 analysis/operational_draft.py
echo; echo "==> 3h  what the routine review changed (A draft -> B reviewed; B -> C for the 52 spot-checks)"
python3 analysis/review_changes.py
echo; echo "==> 3e  step-level agreement (type, and type + DG + OOI jointly)"
python3 analysis/step_agreement.py
echo; echo "==> 3d  automated processing time and decoding throughput"
python3 analysis/timing.py
echo; echo "==> 3f  Phase 2 usage intensity (sessions, documents, measurements per measurer, per month)"
python3 analysis/usage.py
echo; echo "==> 3i  size of the measured requirements (original use-case steps), maintenance products, manual vs assisted"
python3 analysis/workload_size.py
echo; echo "==> 3j  requirement change per measured process (company extraction from version control), manual vs assisted"
python3 analysis/change_magnitude.py

echo; echo "==> 3k  retrospective independent check of the routine review (A/B/C comparison, 50 measurements)"
python3 analysis/retrospective_check.py
echo; echo "==> 4/5 economics (scenario + adoption ledger)"
python3 analysis/economics.py
echo; echo "==> 5/5 LaTeX tables"
python3 analysis/make_tables.py
echo; echo "Done. See results/."
