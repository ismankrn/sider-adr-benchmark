PY ?= ../code/.venv/bin/python
all: tables pdf
tables:
	$(PY) make_tables.py
pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
clean:
	latexmk -C; rm -f *.bbl *.run.xml
.PHONY: all tables pdf clean
