PY ?= python

.PHONY: all data analysis figures paper clean

all: analysis figures paper

data:
	$(PY) src/build_dataset.py

analysis: data
	$(PY) src/describe_population.py
	$(PY) src/timeline.py
	$(PY) src/analysis_pages.py
	$(PY) src/analysis_names.py
	$(PY) src/analysis_forms.py

figures:
	$(PY) src/figure1_setting.py
	$(PY) src/figure2_pages.py
	$(PY) src/figure3_names.py
	$(PY) src/figure4_forms.py
	cp figures/fig1a_schematic.pdf figures/figA_pages.pdf figures/figB_names.pdf figures/figC_forms.pdf paper/figs/

paper:
	cd paper && pdflatex -interaction=nonstopmode final.tex && pdflatex -interaction=nonstopmode final.tex

clean:
	rm -rf cache/*.pkl cache/*.json figures/*.pdf figures/*.png figures/*.svg results/numbers.txt
	rm -f paper/final.aux paper/final.log paper/final.out
