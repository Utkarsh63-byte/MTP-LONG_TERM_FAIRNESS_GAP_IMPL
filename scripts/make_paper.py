"""Render the research paper to DOCX and LaTeX from one content model.

Both outputs come from scripts/paper_content.py, so they cannot diverge.

Run:  .venv/bin/python -m scripts.make_paper
Produces:
  outputs/paper/MTP_Research_Paper.docx
  outputs/paper/main.tex          (self-contained, compiles with pdflatex)
  outputs/paper/references.bib
  outputs/paper/figures/*.png
  outputs/paper/README_COMPILE.txt
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from ltf.config import OUTPUT, ROOT
from scripts.paper_content import build, load_values

TITLE = ("Equal Earnings Are Not Equal Pay: Hours-Aware and Access-Aware "
         "Fairness for Driver Allocation in Ride-Hailing")
AUTHORS = "Utkarsh Pandey"
AFFIL = "Master's Thesis Project"

NAVY = RGBColor(0x10, 0x25, 0x40)
PAPER_DIR = OUTPUT / "paper"

# ---------------------------------------------------------------- math mapping
GREEK = {
    r"\rho": "\u03c1", r"\tau": "\u03c4", r"\lambda": "\u03bb", r"\omega": "\u03c9",
    r"\nu": "\u03bd", r"\sigma": "\u03c3", r"\gamma": "\u03b3", r"\mu": "\u03bc",
    r"\pi": "\u03c0", r"\Pi": "\u03a0", r"\Delta": "\u0394", r"\delta": "\u03b4",
    r"\alpha": "\u03b1", r"\beta": "\u03b2", r"\theta": "\u03b8", r"\ell": "\u2113",
}
OPS = {
    r"\le": "\u2264", r"\ge": "\u2265", r"\leq": "\u2264", r"\geq": "\u2265",
    r"\times": "\u00d7", r"\approx": "\u2248", r"\cdot": "\u00b7",
    r"\in": "\u2208", r"\neq": "\u2260", r"\to": "\u2192", r"\pm": "\u00b1",
    r"\sum": "\u03a3", r"\square": "\u220e", r"\emph": "", r"\textbf": "",
    r"\mathbb{E}": "E", r"\mathrm": "", r"\text": "", r"\,": " ", r"\;": " ",
    r"\big": "", r"\Big": "", r"\left": "", r"\right": "", r"\quad": "   ",
    r"\qquad": "     ", r"\\": " ", r"\min": "min", r"\max": "max",
    r"\partial": "\u2202", r"\lceil": "\u2308", r"\rceil": "\u2309",
    r"\frac": "", r"\underbrace": "", r"\;=\;": " = ",
}
SUB = str.maketrans("0123456789+-=()aeioxvbnpwrhtsjkl",
                    "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089"
                    "\u208a\u208b\u208c\u208d\u208e\u2090\u2091\u1d62\u2092\u2093"
                    "\u1d65\u1d66\u2099\u209a\u1d69\u1d63\u2095\u209c\u209b\u2c7c"
                    "\u2096\u2097")


def math_to_unicode(s: str) -> str:
    """Best-effort conversion of a LaTeX math fragment to readable Unicode."""
    s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\underbrace\{([^{}]*)\}_\{[^{}]*\}", r"\1", s)
    s = re.sub(r"\\bar\{([^{}]*)\}", lambda m: m.group(1) + "\u0304", s)
    s = re.sub(r"\\tilde\{([^{}]*)\}", lambda m: m.group(1) + "\u0303", s)
    s = re.sub(r"\\hat\{([^{}]*)\}", lambda m: m.group(1) + "\u0302", s)
    for k, v in sorted({**GREEK, **OPS}.items(), key=lambda kv: -len(kv[0])):
        s = s.replace(k, v)
    s = re.sub(r"\^\{([^{}]*)\}", r"^\1", s)
    s = re.sub(r"_\{([^{}]*)\}",
               lambda m: m.group(1).translate(SUB) if m.group(1).isalnum()
               else "_" + m.group(1), s)
    s = re.sub(r"_([a-zA-Z0-9])", lambda m: m.group(1).translate(SUB), s)
    s = s.replace("{", "").replace("}", "").replace("$", "")
    return re.sub(r"\s{2,}", " ", s).strip()


REFMAP: dict[str, str] = {}


def prose_to_plain(s: str) -> str:
    """Strip LaTeX markup from prose for the DOCX renderer."""
    s = re.sub(r"\\cite\{([^}]*)\}",
               lambda m: "[" + ", ".join(x.strip() for x in m.group(1).split(",")) + "]", s)
    # resolve cross-references to the numbers assigned in the pre-pass
    s = re.sub(r"(Table|Figure|Equation|Eq\.)~?\\ref\{([^}]*)\}",
               lambda m: f"{m.group(1)} {REFMAP.get(m.group(2), '?')}", s)
    s = re.sub(r"\\ref\{([^}]*)\}",
               lambda m: REFMAP.get(m.group(1), m.group(1).split(":")[-1]), s)
    s = s.replace("~", " ")
    s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\emph\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\\((.+?)\\\)", lambda m: math_to_unicode(m.group(1)), s,
               flags=re.S)
    s = s.replace("---", "\u2014").replace("--", "\u2013")
    s = s.replace("``", "\u201c").replace("''", "\u201d")
    s = s.replace(r"\%", "%").replace(r"\$", "$").replace(r"\&", "&")
    s = s.replace(r"\_", "_").replace(r"\\\"", "\u0308")
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    return re.sub(r"\s{2,}", " ", s).strip()


def latex_escape_prose(s: str) -> str:
    """Prose is authored in LaTeX already; only fix a stray ampersand."""
    return s


# ---------------------------------------------------------------- DOCX
def shade(cell, hx):
    tcPr = cell._tc.get_or_add_tcPr()
    el = OxmlElement("w:shd")
    el.set(qn("w:val"), "clear")
    el.set(qn("w:fill"), hx)
    tcPr.append(el)


def emit_docx(DOC, out: Path) -> None:
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(10)
    for s in doc.sections:
        s.top_margin = s.bottom_margin = Inches(0.9)
        s.left_margin = s.right_margin = Inches(0.9)

    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(TITLE); r.bold = True; r.font.size = Pt(17)
    r.font.color.rgb = NAVY
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(AUTHORS); r.font.size = Pt(11.5)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(AFFIL); r.font.size = Pt(9.5); r.italic = True
    doc.add_paragraph()

    sec_n, sub_n, eqn, tabn, fign, prop_n = 0, 0, 0, 0, 0, 0

    # pre-pass: assign numbers so cross-references resolve in the DOCX too
    REFMAP.clear()
    TAB_ORDER = ["tab:notation", "tab:main", "tab:iso", "tab:causes",
                 "tab:ablation", "tab:horizon", "tab:frontier", "tab:seeds",
                 "tab:repro"]
    FIG_ORDER = ["fig:congestion", "fig:bymethod", "fig:horizon", "fig:pareto"]
    for kind, payload in DOC:
        if kind == "eq" and payload[2]:
            eqn += 1
            REFMAP[payload[2]] = str(eqn)
        elif kind == "table":
            tabn += 1
            if tabn <= len(TAB_ORDER):
                REFMAP[TAB_ORDER[tabn - 1]] = str(tabn)
        elif kind == "fig":
            fign += 1
            if fign <= len(FIG_ORDER):
                REFMAP[FIG_ORDER[fign - 1]] = str(fign)
    eqn = tabn = fign = 0

    for kind, payload in DOC:
        if kind == "abstract":
            p = doc.add_paragraph(); r = p.add_run("Abstract. ")
            r.bold = True; r.font.size = Pt(9.5)
            r2 = p.add_run(prose_to_plain(payload)); r2.font.size = Pt(9.5)
            p.paragraph_format.space_after = Pt(6)
        elif kind == "keywords":
            p = doc.add_paragraph(); r = p.add_run("Keywords: ")
            r.bold = True; r.font.size = Pt(9.5)
            r2 = p.add_run(prose_to_plain(payload)); r2.font.size = Pt(9.5)
            r2.italic = True
            doc.add_paragraph()
        elif kind == "h1":
            sec_n += 1; sub_n = 0
            h = doc.add_heading(f"{sec_n}  {payload}", level=1)
            for rr in h.runs:
                rr.font.color.rgb = NAVY; rr.font.size = Pt(13)
        elif kind == "h2":
            sub_n += 1
            h = doc.add_heading(f"{sec_n}.{sub_n}  {payload}", level=2)
            for rr in h.runs:
                rr.font.color.rgb = NAVY; rr.font.size = Pt(11)
        elif kind == "p":
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            r = p.add_run(prose_to_plain(payload)); r.font.size = Pt(10)
            p.paragraph_format.space_after = Pt(5)
        elif kind == "pb":
            doc.add_page_break()
        elif kind in ("bul", "enum"):
            for i, it in enumerate(payload):
                p = doc.add_paragraph(
                    style="List Number" if kind == "enum" else "List Bullet")
                r = p.add_run(prose_to_plain(it)); r.font.size = Pt(9.5)
                p.paragraph_format.space_after = Pt(3)
        elif kind == "prop":
            title, body = payload
            prop_n += 1
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            r = p.add_run(f"Proposition {prop_n} ({prose_to_plain(title)}). ")
            r.bold = True; r.font.size = Pt(10)
            r2 = p.add_run(prose_to_plain(body))
            r2.italic = True; r2.font.size = Pt(10)
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.left_indent = Inches(0.15)
        elif kind == "proof":
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            r = p.add_run("Proof. "); r.italic = True; r.font.size = Pt(10)
            r2 = p.add_run(prose_to_plain(payload) + " \u220e")
            r2.font.size = Pt(10)
            p.paragraph_format.space_after = Pt(6)
            p.paragraph_format.left_indent = Inches(0.15)
        elif kind == "eq":
            latex, plain, lab = payload
            eqn += 1
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(plain if plain else math_to_unicode(latex))
            r.font.name = "Cambria Math"; r.font.size = Pt(10.5)
            if lab:
                r2 = p.add_run(f"     ({eqn})"); r2.font.size = Pt(9.5)
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(6)
        elif kind == "table":
            cap, header, rows, widths = payload
            tabn += 1
            p = doc.add_paragraph()
            r = p.add_run(f"Table {tabn}. "); r.bold = True; r.font.size = Pt(9)
            r2 = p.add_run(prose_to_plain(cap)); r2.font.size = Pt(9)
            p.paragraph_format.space_after = Pt(3)
            t = doc.add_table(rows=1, cols=len(header))
            t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
            for i, hh in enumerate(header):
                c = t.rows[0].cells[i]; c.text = ""
                rr = c.paragraphs[0].add_run(prose_to_plain(str(hh)))
                rr.bold = True; rr.font.size = Pt(7.5)
                shade(c, "DDE5EE")
            for row in rows:
                cells = t.add_row().cells
                for i, v in enumerate(row):
                    cells[i].text = ""
                    rr = cells[i].paragraphs[0].add_run(prose_to_plain(str(v)))
                    rr.font.size = Pt(7.5)
            if widths:
                for row in t.rows:
                    for i, w in enumerate(widths):
                        row.cells[i].width = Inches(w)
            doc.add_paragraph().paragraph_format.space_after = Pt(6)
        elif kind == "fig":
            path, cap, width = payload
            fign += 1
            src = ROOT / path
            if src.exists():
                p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.add_run().add_picture(str(src), width=Inches(width))
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(f"Figure {fign}. "); r.bold = True; r.font.size = Pt(9)
            r2 = p.add_run(prose_to_plain(cap)); r2.font.size = Pt(9)
            p.paragraph_format.space_after = Pt(8)
        elif kind == "bib":
            h = doc.add_heading("References", level=1)
            for rr in h.runs:
                rr.font.color.rgb = NAVY; rr.font.size = Pt(13)
            for i, (key, text) in enumerate(payload, 1):
                p = doc.add_paragraph()
                r = p.add_run(f"[{key}]  ")
                r.font.size = Pt(8.5); r.bold = True
                r2 = p.add_run(prose_to_plain(text)); r2.font.size = Pt(8.5)
                p.paragraph_format.space_after = Pt(2)
                p.paragraph_format.left_indent = Inches(0.35)
                p.paragraph_format.first_line_indent = Inches(-0.35)
    doc.save(out)


# ---------------------------------------------------------------- LaTeX
PREAMBLE = r"""% =====================================================================
%  """ + TITLE + r"""
%
%  Self-contained. Compile with:
%      pdflatex main && bibtex main && pdflatex main && pdflatex main
%
%  TO SWITCH VENUE FORMAT, replace the \documentclass line:
%    Springer LNCS (ECML PKDD, ECAI):  \documentclass{llncs}
%    ACM (KDD, WWW, FAccT):            \documentclass[sigconf]{acmart}
%    IEEE (ICDM):                      \documentclass[conference]{IEEEtran}
%    AAAI:                             \documentclass[letterpaper]{article} + aaai.sty
%  and delete the \titlepagestyle block below, which exists only for the
%  generic article class.
% =====================================================================
\documentclass[10pt,twocolumn]{article}

\usepackage[margin=0.75in]{geometry}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{multirow}
\usepackage{array}
\usepackage{caption}
\usepackage{subcaption}
\usepackage[hidelinks]{hyperref}
\usepackage{xcolor}
\usepackage{times}
\usepackage{balance}

\captionsetup{font=small,labelfont=bf}
\captionsetup[table]{skip=4pt}
\setlength{\tabcolsep}{3.5pt}
\renewcommand{\arraystretch}{1.05}

\newtheorem{proposition}{Proposition}

\newcommand{\Geo}{\mathrm{Geo}}
\newcommand{\Var}{\mathrm{Var}}
\newcommand{\E}{\mathbb{E}}
\newcommand{\occ}{\mathrm{occ}}
\newcommand{\opp}{\mathrm{opp}}
\newcommand{\up}{$\uparrow$}
\newcommand{\dn}{$\downarrow$}

\title{\textbf{""" + TITLE + r"""}}
\author{""" + AUTHORS + r"""\\ \small """ + AFFIL + r"""}
\date{}

\begin{document}
\maketitle
"""

TAIL = r"""
\balance
\bibliographystyle{plain}
\bibliography{references}

\end{document}
"""


def tex_table(cap, header, rows, widths, label):
    ncol = len(header)
    colspec = "l" + "r" * (ncol - 1)
    star = "*" if ncol >= 7 else ""
    env = f"table{star}"
    pos = "[t]" if star else "[t]"
    out = [f"\\begin{{{env}}}{pos}", "\\centering",
           f"\\caption{{{cap}}}", f"\\label{{{label}}}",
           "\\small" if ncol < 7 else "\\footnotesize",
           f"\\begin{{tabular}}{{{colspec}}}", "\\toprule"]
    out.append(" & ".join(str(h) for h in header) + " \\\\")
    out.append("\\midrule")
    for r in rows:
        out.append(" & ".join(str(x) for x in r) + " \\\\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append(f"\\end{{{env}}}")
    return "\n".join(out)


def emit_latex(DOC, out: Path, figdir: Path) -> None:
    L = [PREAMBLE]
    tab_i, fig_i = 0, 0
    tab_labels = {
        1: "tab:notation", 2: "tab:main", 3: "tab:iso", 4: "tab:causes",
        5: "tab:ablation", 6: "tab:horizon", 7: "tab:frontier", 8: "tab:seeds",
        9: "tab:repro",
    }
    fig_labels = {1: "fig:congestion", 2: "fig:bymethod", 3: "fig:horizon",
                  4: "fig:pareto"}
    for kind, payload in DOC:
        if kind == "abstract":
            L.append("\\begin{abstract}\n" + payload + "\n\\end{abstract}")
        elif kind == "keywords":
            L.append("\\noindent\\textbf{Keywords:} \\emph{" + payload + "}\n")
        elif kind == "h1":
            L.append("\\section{" + payload + "}")
        elif kind == "h2":
            L.append("\\subsection{" + payload + "}")
        elif kind == "h3":
            L.append("\\subsubsection{" + payload + "}")
        elif kind == "p":
            L.append(payload + "\n")
        elif kind == "pb":
            L.append("")
        elif kind == "bul":
            L.append("\\begin{itemize}\n\\itemsep2pt")
            for it in payload:
                L.append("  \\item " + it)
            L.append("\\end{itemize}")
        elif kind == "enum":
            L.append("\\begin{enumerate}\n\\itemsep2pt")
            for it in payload:
                L.append("  \\item " + it)
            L.append("\\end{enumerate}")
        elif kind == "prop":
            title, body = payload
            L.append("\\begin{proposition}[" + title + "]\n" + body
                     + "\n\\end{proposition}")
        elif kind == "proof":
            L.append("\\begin{proof}\n" + payload + "\n\\end{proof}")
        elif kind == "eq":
            latex, _plain, lab = payload
            if lab:
                L.append("\\begin{equation}\\label{" + lab + "}\n" + latex
                         + "\n\\end{equation}")
            else:
                L.append("\\begin{equation*}\n" + latex + "\n\\end{equation*}")
        elif kind == "table":
            cap, header, rows, widths = payload
            tab_i += 1
            L.append(tex_table(cap, header, rows, widths,
                               tab_labels.get(tab_i, f"tab:t{tab_i}")))
        elif kind == "fig":
            path, cap, width = payload
            fig_i += 1
            name = Path(path).name
            src = ROOT / path
            if src.exists():
                shutil.copy(src, figdir / name)
            star = "*"
            L.append(
                f"\\begin{{figure{star}}}[t]\n\\centering\n"
                f"\\includegraphics[width=0.92\\linewidth]{{figures/{name}}}\n"
                f"\\caption{{{cap}}}\n"
                f"\\label{{{fig_labels.get(fig_i, f'fig:f{fig_i}')}}}\n"
                f"\\end{{figure{star}}}")
        elif kind == "bib":
            bib = ["% Auto-generated. Entries are plain-text 'misc' so the file "
                   "compiles anywhere.", ""]
            for key, text in payload:
                bib.append("@misc{" + key + ",")
                bib.append("  note = {" + text + "}")
                bib.append("}")
                bib.append("")
            (out.parent / "references.bib").write_text("\n".join(bib))
    L.append(TAIL)
    out.write_text("\n".join(L))


# ---------------------------------------------------------------- main
def main() -> int:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    figdir = PAPER_DIR / "figures"
    figdir.mkdir(exist_ok=True)

    V = load_values()
    DOC = build(V)

    docx_out = PAPER_DIR / "MTP_Research_Paper.docx"
    tex_out = PAPER_DIR / "main.tex"
    emit_docx(DOC, docx_out)
    emit_latex(DOC, tex_out, figdir)

    n_eq = sum(1 for kind, p in DOC if kind == "eq")
    n_tab = sum(1 for kind, p in DOC if kind == "table")
    n_fig = sum(1 for kind, p in DOC if kind == "fig")
    n_sec = sum(1 for kind, p in DOC if kind == "h1")
    words = sum(len(prose_to_plain(p).split())
                for kind, p in DOC if kind in ("p", "abstract"))

    (PAPER_DIR / "README_COMPILE.txt").write_text(
        "COMPILING THE PAPER\n"
        "===================\n\n"
        "Files:\n"
        "  main.tex          the paper (self-contained, article class)\n"
        "  references.bib    bibliography\n"
        "  figures/          all figures referenced by main.tex\n"
        "  MTP_Research_Paper.docx   Word version, identical content\n\n"
        "Local compile:\n"
        "  pdflatex main && bibtex main && pdflatex main && pdflatex main\n\n"
        "No LaTeX installed? Upload this whole folder to Overleaf\n"
        "(New Project -> Upload Project) and it compiles unchanged.\n\n"
        "SWITCHING VENUE FORMAT\n"
        "----------------------\n"
        "Replace the \\documentclass line at the top of main.tex:\n"
        "  Springer LNCS (ECML PKDD, ECAI):  \\documentclass{llncs}\n"
        "  ACM (KDD, WWW, FAccT):            \\documentclass[sigconf]{acmart}\n"
        "  IEEE (ICDM):                      \\documentclass[conference]{IEEEtran}\n"
        "For llncs also replace \\maketitle with the LNCS \\institute/\\author\n"
        "block, and change \\bibliographystyle{plain} to {splncs04}.\n\n"
        f"Statistics: {n_sec} sections, {n_eq} numbered equations, {n_tab} tables, "
        f"{n_fig} figures, ~{words} words of prose.\n")

    print(f"wrote {docx_out}  ({docx_out.stat().st_size:,} bytes)")
    print(f"wrote {tex_out}  ({tex_out.stat().st_size:,} bytes)")
    print(f"wrote {PAPER_DIR/'references.bib'}")
    print(f"copied {len(list(figdir.glob('*.png')))} figures")
    print(f"\n{n_sec} sections | {n_eq} equations | {n_tab} tables | "
          f"{n_fig} figures | ~{words} words prose")
    return 0


if __name__ == "__main__":
    sys.exit(main())
