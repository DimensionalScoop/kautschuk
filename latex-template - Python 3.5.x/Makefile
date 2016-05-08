# Benennung der Files:
# Protokoll.tex
# PythonSkript.py
# curve_fit.py
# table.py
# meine-matplotlibrc
# header-matplotlib.tex
# ... (einstellen)

ifeq (,$(shell sh -c 'cygpath --version 2> /dev/null'))
  # Unix
  pwd := $$(pwd)
  translate = $1
else
  # Windows mit MSys2/Cygwin
  pwd := $$(cygpath -m "$$(pwd)")
  translate = $(shell echo '$1' | sed 's/:/;/g')
endif

all: build/Protokoll.pdf

build/Protokoll.pdf: FORCE $(addprefix build/, table.tex a.tex b.tex c.tex d.tex plot.pdf) | build messdaten     # hier Abhängigkeiten einstellen
	  TEXINPUTS="$(call translate,build:)" \
	  max_print_line=1048576 \
	latexmk \
	  --lualatex \
	  --output-directory=build \
	  --interaction=nonstopmode \
	  --halt-on-error \
		--synctex=1 \
	Protokoll.tex

build/plot.pdf: PythonSkript.py ./python_custom_scripts/curve_fit.py ./python_custom_scripts/table.py meine-matplotlibrc header-matplotlib.tex | build    # hier Abhängigkeiten einstellen
	TEXINPUTS="$(call translate,$(pwd):)" python PythonSkript.py

$(addprefix build/, table.tex a.tex b.tex c.tex d.tex): build/plot.pdf

messdaten: datenersteller.py
	python datenersteller.py

build:
	mkdir -p build

clean:
	rm -rf build
	rm -f data.txt

FORCE:

.PHONY: all clean
