PYTHON ?= python
BIBTEX ?= $(shell command -v bibtex 2>/dev/null || command -v bibtex8 2>/dev/null)
.PHONY: test reproduce figures paper

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q

reproduce: test
	PYTHONPATH=src $(PYTHON) experiments/exact.py
	PYTHONPATH=src $(PYTHON) experiments/fragility.py
	PYTHONPATH=src $(PYTHON) experiments/trained_rnn.py --seeds 10 --steps 1000
	PYTHONPATH=src $(PYTHON) experiments/matched_size.py
	$(PYTHON) experiments/plots.py

figures:
	$(PYTHON) experiments/plots.py

paper: figures
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex
	cd paper && $(BIBTEX) main
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex
