PY ?= python

.PHONY: all analysis figures paper clean

all: analysis paper

# everything: analyses, model, figures and tables (about five minutes on 8 cores)
analysis:
	$(PY) src/run_all.py

# only the figures and tables, from an existing cache
figures:
	$(PY) src/run_all.py figures

# main, extended_data and si reference one another (xr-hyper): two rounds over the three
paper:
	cd paper && for d in extended_data si main extended_data si main; do pdflatex -interaction=nonstopmode $$d.tex >/dev/null; done
	cd paper && bibtex main >/dev/null; pdflatex -interaction=nonstopmode main.tex >/dev/null

clean:
	rm -rf cache/*.pkl cache/*.json figures/*.png results/numbers.txt
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.synctex.gz
