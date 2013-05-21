.PHONY: main

main:
	runhaskell haskell.lhs; [ $$? -eq 0 ]
	sed -E "s/\begin\{code\}/\begin\{minted\}\{haskell\}/g" haskell.lhs | sed -E "s/\end\{code\}/\end\{minted\}/g" > haskell.tex
	pdflatex -shell-escape haskell.tex
	rm haskell.tex
	rm *.out *.nav *.log *.toc *.vrb *.aux *.snm

extract:
	ghc -E haskell.lhs -o haskell1.hs && cat -s haskell1.hs > haskell.hs && rm haskell1.hs 

clean:
	rm haskell.pdf
