#!/usr/bin/env bash
# CPU-only smoke test of the whole pipeline (samples -> prep -> select -> extract ->
# benchmark -> metrics -> stats -> figures) on a tiny synthetic world, through the real
# CLI, in a scratch directory; nothing in the repository is touched.  ~2 minutes.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pytest tests/test_end_to_end.py -q
