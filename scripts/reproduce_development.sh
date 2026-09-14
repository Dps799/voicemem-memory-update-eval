#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
"$PYTHON" src/prepare_n1.py
"$PYTHON" -m unittest discover -s tests -v 2>&1 | tee reports/development_tests.txt
"$PYTHON" src/run_n1_dev.py --run-id "n1_dev_${RUN_TAG}"
"$PYTHON" src/analyze_n1_auto.py "reports/n1_dev/n1_dev_${RUN_TAG}"
"$PYTHON" src/run_n2_dev.py --run-id "n2_dev_${RUN_TAG}"
"$PYTHON" src/report_development.py --n1 "reports/n1_dev/n1_dev_${RUN_TAG}" --n2 "reports/n2_dev/n2_dev_${RUN_TAG}"
