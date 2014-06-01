all:
	latexmk -outdir=build -bibtex -shell-escape -pdf thesis.tex

preview:
	latexmk -outdir=build -bibtex -shell-escape -pdf -pvc thesis.tex


clean:
	latexmk -outdir=build -c
