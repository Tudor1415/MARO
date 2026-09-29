# Reproduce the study.  `make test` takes seconds; the experiments take about 35 CPU-hours,
# spread over as many cores as WORKERS allows.
WORKERS ?= $(shell nproc)

test:
	python -m pytest -q

experiments:
	python -m experiments.run E0 E1 E2 E3 E3b E6 E6h --workers $(WORKERS)
	python -m experiments.run E4 E5x150 E5x250 E5mb --workers $(WORKERS)

analysis:
	python -m experiments.analyse
	python -m experiments.figures

.PHONY: test experiments analysis
