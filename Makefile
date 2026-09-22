PY ?= python

.PHONY: all analysis figures paper clean

all: analysis paper

# everything: analyses, model, figures and tables (about five minutes on 8 cores)
analysis:
	$(PY) src/run_all.py

# only the figures and tables, from an existing cache
figures:
	$(PY) src/run_all.py figures

paper:
	cd paper && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
	cd paper && pdflatex -interaction=nonstopmode si.tex && pdflatex -interaction=nonstopmode si.tex

clean:
	rm -rf cache/*.pkl cache/*.json figures/*.png results/numbers.txt
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.synctex.gz
