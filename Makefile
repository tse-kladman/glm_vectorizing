# One-command entry points.  Run from the repository root.
#   make smoke    CPU-only check of the WHOLE pipeline on a synthetic world (~2 min)
#   make test     the full unit test suite
#   make figures  redraw every figure from the committed tables (no GPU, no genomes)
PYTHON ?= python

.PHONY: smoke test figures

smoke:
	$(PYTHON) -m pytest tests/test_end_to_end.py -q

test:
	$(PYTHON) -m pytest tests -q

figures:
	$(PYTHON) run_pipeline.py figures --plot-only
