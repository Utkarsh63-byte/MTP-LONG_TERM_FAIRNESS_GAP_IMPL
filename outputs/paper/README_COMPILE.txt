COMPILING THE PAPER
===================

Files:
  main.tex          the paper (self-contained, article class)
  references.bib    bibliography
  figures/          all figures referenced by main.tex
  MTP_Research_Paper.docx   Word version, identical content

Local compile:
  pdflatex main && bibtex main && pdflatex main && pdflatex main

No LaTeX installed? Upload this whole folder to Overleaf
(New Project -> Upload Project) and it compiles unchanged.

SWITCHING VENUE FORMAT
----------------------
Replace the \documentclass line at the top of main.tex:
  Springer LNCS (ECML PKDD, ECAI):  \documentclass{llncs}
  ACM (KDD, WWW, FAccT):            \documentclass[sigconf]{acmart}
  IEEE (ICDM):                      \documentclass[conference]{IEEEtran}
For llncs also replace \maketitle with the LNCS \institute/\author
block, and change \bibliographystyle{plain} to {splncs04}.

Statistics: 10 sections, 14 numbered equations, 9 tables, 4 figures, ~4444 words of prose.
