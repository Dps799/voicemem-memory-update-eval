#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
mkdir -p reports/run_logs
"$PYTHON" -m unittest discover -s tests -v 2>&1 | tee reports/run_logs/tests.txt
"$PYTHON" src/run_phase_a.py 2>&1 | tee reports/run_logs/phase_a.txt
"$PYTHON" src/aggregate_metrics.py 2>&1 | tee reports/run_logs/aggregate.txt
"$PYTHON" src/run_phase_b.py 2>&1 | tee reports/run_logs/phase_b.txt
"$PYTHON" src/render_report.py
"$PYTHON" src/record_run.py | tee reports/run_logs/artifact_validation.txt
