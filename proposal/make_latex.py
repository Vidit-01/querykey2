"""Generate research_proposal.tex from research_proposal.md and compile to PDF."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MD_PATH = ROOT / "research_proposal.md"
TEX_PATH = ROOT / "research_proposal.tex"
FIG_DIR = ROOT / "figures"
OUT_DIR = ROOT.parent / "output" / "pdf"
OUT_PDF = OUT_DIR / "research_proposal.pdf"
BUILD_DIR = ROOT / "latex_build"

PREAMBLE = r"""\documentclass[11pt,letterpaper]{article}

\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{microtype}
\usepackage{geometry}
\geometry{left=1.05in,right=1.05in,top=0.95in,bottom=1.0in}

\usepackage{amsmath,amssymb,amsthm,mathtools}
\usepackage{booktabs,tabularx,array}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{enumitem}
\usepackage{fancyhdr}
\usepackage[hidelinks]{hyperref}
\usepackage[nameinlink,capitalise,noabbrev]{cleveref}

\definecolor{ink}{HTML}{1C1917}
\definecolor{navy}{HTML}{1E3A4C}
\definecolor{accent}{HTML}{9A3412}
\definecolor{rule}{HTML}{D6C7B0}
\definecolor{soft}{HTML}{57534E}
\definecolor{boxbg}{HTML}{F4F1EA}
\definecolor{boxbd}{HTML}{B08968}
\definecolor{absbg}{HTML}{EEF3F6}
\definecolor{absbd}{HTML}{C5D0D8}

\hypersetup{colorlinks=true,linkcolor=navy,citecolor=navy,urlcolor=navy}

\newtheorem{lemma}{Lemma}[section]
\newtheorem{proposition}{Proposition}[section]
\theoremstyle{definition}
\newtheorem{remark}{Remark}[section]

\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\textcolor{navy}{The Geometry of Transformer Initialization}}
\fancyhead[R]{\small\textcolor{soft}{Research Proposal \textbullet\ v2.0}}
\fancyfoot[L]{\small\textcolor{soft}{Vidit Gupta}}
\fancyfoot[C]{\small\textcolor{soft}{\thepage}}
\fancyfoot[R]{\small\textcolor{soft}{Deep Learning Theory}}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0.4pt}
\setlength{\headheight}{14pt}

\setlist[itemize]{leftmargin=1.35em,itemsep=0.25em,topsep=0.35em}
\setlist[enumerate]{leftmargin=1.55em,itemsep=0.25em,topsep=0.35em}
\setlength{\parskip}{0.35em}
\setlength{\parindent}{0pt}

\newcommand{\qedbox}{\ensuremath{\square}}
\newenvironment{principle}{%
  \begin{quote}\small\color{ink}\textbf{Design principle.}\ %
}{%
  \end{quote}%
}

\newcommand{\FigEntropy}{figures/fig_entropy}
\newcommand{\FigDiversity}{figures/fig_diversity}
\newcommand{\FigPhase}{figures/fig_phase}

\title{\vspace{-1.2em}{\color{navy}\bfseries The Geometry of Transformer Initialization}\\[0.35em]
{\large\color{soft} Exact Query-Key Moment Identities and a Falsifiable Initialization Atlas}}
\author{\textbf{Vidit Gupta}\\[0.2em]
{\small Deep Learning Theory \textbullet\ Optimization \textbullet\ Transformer Architectures}\\[0.15em]
{\small Status: exact finite-width identities + preregistered experiments}}
\date{Research Proposal \textbullet\ v2.0 (mathematical audit revision)}

\begin{document}
\maketitle
\vspace{-1.5em}
"""

POSTAMBLE = r"""
\end{document}
"""


def escape_text(text: str) -> str:
    """Escape LaTeX special characters outside math."""
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciitilde{}"),
    ]
    out: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == "$":
            j = text.find("$", i + 1)
            if j == -1:
                out.append(text[i:])
                break
            out.append(text[i : j + 1])
            i = j + 1
            continue
        ch = text[i]
        replaced = False
        for old, new in replacements:
            if ch == old:
                out.append(new)
                replaced = True
                break
        if not replaced:
            out.append(ch)
        i += 1
    return "".join(out)


def inline_md(text: str) -> str:
    text = escape_text(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\\textit{\1}", text)
    text = re.sub(r"`([^`]+)`", r"\\texttt{\1}", text)
    return text


THM_RE = re.compile(
    r"^\*\*(Lemma|Proposition|Remark)\s+(\d+)\s*\(([^)]*)\)\.\*\*\s*(.*)$",
    re.DOTALL,
)


def split_table_row(line: str) -> list[str]:
    """Split a markdown table row on unescaped pipe characters."""
    body = line.strip().strip("|")
    cells: list[str] = []
    current: list[str] = []
    i = 0
    while i < len(body):
        ch = body[i]
        if ch == "|" and (i == 0 or body[i - 1] != "\\"):
            cells.append("".join(current).strip())
            current = []
            i += 1
            continue
        current.append(ch)
        i += 1
    cells.append("".join(current).strip())
    return cells


def convert_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    ncols = len(rows[0])
    if ncols == 2:
        colspec = "@{}p{0.22\\textwidth}p{0.73\\textwidth}@{}"
    elif ncols == 3:
        colspec = "@{}p{0.28\\textwidth}p{0.18\\textwidth}p{0.47\\textwidth}@{}"
    else:
        colspec = "@{}*" + str(ncols) + "{>{\\raggedright\\arraybackslash}p{0.9\\textwidth/" + str(ncols) + "}}@{}"
    lines = ["\\begin{tabular}{" + colspec + "}", "\\toprule"]
    for ridx, row in enumerate(rows):
        cells = " & ".join(inline_md(c) for c in row)
        lines.append(cells + " \\\\")
        if ridx == 0:
            lines.append("\\midrule")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def figure_block(path: str, caption: str, label: str, width: str = r"0.92\textwidth") -> str:
    return (
        f"\\begin{{figure}}[htbp]\n"
        f"\\centering\n"
        f"\\includegraphics[width={width}]{{{path}}}\n"
        f"\\caption{{{inline_md(caption)}}}\n"
        f"\\label{{{label}}}\n"
        f"\\end{{figure}}\n"
    )


def maybe_insert_figures(
    title: str,
    out: list[str],
    inserted_fig_entropy: bool,
    inserted_fig_diversity: bool,
    inserted_fig_phase: bool,
) -> tuple[bool, bool, bool]:
    if title.startswith("4.5") and not inserted_fig_entropy:
        out.append(
            figure_block(
                r"\FigEntropy",
                "Exact finite-width logit moments versus Monte Carlo simulation. "
                "Panels (a) and (b) test Proposition~1 across alignment; panel (c) "
                "resolves the finite-width variance difference between the orthogonal "
                "and independent channels.",
                "fig:entropy",
            )
        )
        inserted_fig_entropy = True
    if title.startswith("4.8") and not inserted_fig_diversity:
        out.append(
            figure_block(
                r"\FigDiversity",
                "The exact uncentered common-bias statistic $R_{12}$ from "
                "Proposition~5. Centering removes the ensemble mean, so this statistic "
                "is not a functional head-diversity guarantee.",
                "fig:diversity",
                r"0.78\textwidth",
            )
        )
        inserted_fig_diversity = True
    if title.startswith("6.") and not inserted_fig_phase:
        out.append(
            figure_block(
                r"\FigPhase",
                "Finite-$n$ unmasked i.i.d.-Gaussian screening surrogate in "
                "$(s,\\alpha)$ at fixed $n=64$, $m=16$, and $r=0.25$. Contours are "
                "operational guides, not trainability theorems; Experiment~1 replaces "
                "them with a replicated empirical atlas.",
                "fig:phase",
            )
        )
        inserted_fig_phase = True
    return inserted_fig_entropy, inserted_fig_diversity, inserted_fig_phase


def convert_md_to_latex(md: str) -> str:
    lines = md.replace("\r\n", "\n").split("\n")
    out: list[str] = []
    i = 0
    n = len(lines)
    in_abstract = False
    inserted_fig_entropy = False
    inserted_fig_diversity = False
    inserted_fig_phase = False

    def flush_paragraph(buf: list[str]) -> None:
        text = " ".join(x.strip() for x in buf if x.strip())
        if text:
            if text.startswith("> "):
                body = inline_md(text[2:].strip())
                if body.startswith("\\textbf{Design principle.}"):
                    out.append("\\begin{principle}")
                    out.append(body.replace("\\textbf{Design principle.}", "").strip())
                    out.append("\\end{principle}")
                else:
                    out.append("\\begin{quote}\\small " + body + "\\end{quote}")
            else:
                out.append(inline_md(text))
                out.append("")

    while i < n:
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("# ") and not stripped.startswith("## "):
            i += 1
            continue
        if stripped.startswith("## ") and not stripped.startswith("### "):
            title = stripped[3:].strip()
            if title == "Abstract":
                out.append("\\begin{center}")
                out.append("\\colorbox{absbg}{\\parbox{0.96\\textwidth}{\\vspace{0.4em}")
                out.append("{\\color{navy}\\textbf{\\small ABSTRACT}}\\\\[0.35em]")
                in_abstract = True
                i += 1
                continue
            if in_abstract:
                out.append("}}")
                out.append("\\end{center}")
                out.append("\\vspace{0.5em}")
                in_abstract = False
            sec = title.split(".", 1)[0] if re.match(r"^\d+\.", title) else title
            if title.startswith("Appendix"):
                out.append("\\appendix")
            out.append(f"\\section{{{inline_md(title)}}}")
            inserted_fig_entropy, inserted_fig_diversity, inserted_fig_phase = maybe_insert_figures(
                title, out, inserted_fig_entropy, inserted_fig_diversity, inserted_fig_phase
            )
            i += 1
            continue

        if stripped.startswith("### "):
            title = stripped[4:].strip()
            out.append(f"\\subsection{{{inline_md(title)}}}")
            inserted_fig_entropy, inserted_fig_diversity, inserted_fig_phase = maybe_insert_figures(
                title, out, inserted_fig_entropy, inserted_fig_diversity, inserted_fig_phase
            )
            i += 1
            continue

        if stripped == "---":
            out.append("\\vspace{0.35em}\\hrule\\vspace{0.35em}")
            i += 1
            continue

        if stripped.startswith("$$"):
            eq_lines = []
            if stripped.endswith("$$") and len(stripped) > 4:
                eq_lines.append(stripped[2:-2])
                i += 1
            else:
                i += 1
                while i < n and lines[i].strip() != "$$":
                    eq_lines.append(lines[i])
                    i += 1
                i += 1
            eq = "\n".join(eq_lines).strip()
            out.append("\\[")
            out.append(eq)
            out.append("\\]")
            continue

        if stripped.startswith("|") and i + 1 < n and re.match(r"^\s*\|?\s*-+", lines[i + 1]):
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                row = split_table_row(lines[i])
                if not all(re.match(r"^:?-+:?$", c.replace(" ", "")) for c in row):
                    rows.append(row)
                i += 1
            out.append(convert_table(rows))
            out.append("")
            continue

        if re.match(r"^[-*]\s+", stripped):
            items = []
            while i < n and re.match(r"^[-*]\s+", lines[i].strip()):
                items.append(re.sub(r"^[-*]\s+", "", lines[i].strip()))
                i += 1
            out.append("\\begin{itemize}")
            for it in items:
                out.append(f"  \\item {inline_md(it)}")
            out.append("\\end{itemize}")
            continue

        if re.match(r"^\d+\.\s+", stripped):
            items = []
            while i < n and re.match(r"^\d+\.\s+", lines[i].strip()):
                items.append(re.sub(r"^\d+\.\s+", "", lines[i].strip()))
                i += 1
            out.append("\\begin{enumerate}")
            for it in items:
                out.append(f"  \\item {inline_md(it)}")
            out.append("\\end{enumerate}")
            continue

        m = THM_RE.match(stripped)
        if m:
            kind, num, title, rest = m.groups()
            env = kind.lower()
            out.append(f"\\begin{{{env}}}[{inline_md(title)}]")
            if rest.strip():
                out.append(inline_md(rest.strip()))
            i += 1
            while i < n:
                nxt = lines[i].strip()
                if not nxt or nxt.startswith("#") or nxt == "---" or nxt.startswith("|") or nxt == "$$":
                    break
                if nxt.startswith("*Proof") or nxt.startswith("**Proof"):
                    break
                if THM_RE.match(nxt):
                    break
                if re.match(r"^[-*]\s+", nxt) or re.match(r"^\d+\.\s+", nxt):
                    break
                out.append(inline_md(nxt))
                i += 1
            out.append(f"\\end{{{env}}}")
            continue

        if stripped.startswith("*Proof") or stripped.startswith("**Proof"):
            body = re.sub(r"^\*+\s*Proof\.?\*+\s*", "", stripped)
            out.append("\\begin{proof}")
            if body:
                out.append(inline_md(body))
            i += 1
            while i < n:
                nxt = lines[i].strip()
                if not nxt or nxt.startswith("#") or nxt == "---" or THM_RE.match(nxt):
                    break
                if re.match(r"^[-*]\s+", nxt) or re.match(r"^\d+\.\s+", nxt):
                    break
                out.append(inline_md(nxt))
                i += 1
            out.append("\\end{proof}")
            continue

        if not stripped:
            i += 1
            continue

        buf = [stripped]
        i += 1
        while i < n:
            nxt = lines[i].strip()
            if (
                not nxt
                or nxt.startswith("#")
                or nxt == "---"
                or nxt.startswith("|")
                or nxt == "$$"
                or re.match(r"^[-*]\s+", nxt)
                or re.match(r"^\d+\.\s+", nxt)
                or THM_RE.match(nxt)
                or nxt.startswith("*Proof")
                or nxt.startswith("**Proof")
            ):
                break
            buf.append(nxt)
            i += 1
        flush_paragraph(buf)

    if in_abstract:
        out.append("}}")
        out.append("\\end{center}")

    if not inserted_fig_entropy:
        out.append(
            figure_block(
                r"\FigEntropy",
                "Exact finite-width logit moments versus Monte Carlo simulation.",
                "fig:entropy",
            )
        )
    if not inserted_fig_diversity:
        out.append(
            figure_block(
                r"\FigDiversity",
                "Uncentered common-bias statistic $R_{12}$ from Proposition~5.",
                "fig:diversity",
                r"0.78\textwidth",
            )
        )
    if not inserted_fig_phase:
        out.append(
            figure_block(
                r"\FigPhase",
                "Finite-$n$ unmasked Gaussian screening surrogate.",
                "fig:phase",
            )
        )

    refs_start = None
    for idx, line in enumerate(out):
        if line.startswith("\\section{References}"):
            refs_start = idx
            break
    if refs_start is not None:
        body = out[refs_start + 1 :]
        out = out[: refs_start + 1]
        out.append("\\begin{thebibliography}{20}")
        tail: list[str] = []
        for line in body:
            stripped_line = line.strip()
            if stripped_line.startswith("\\appendix") or stripped_line.startswith("\\section{Appendix"):
                tail.append(line)
                continue
            if tail:
                tail.append(line)
                continue
            m = re.match(r"^\[(\d+)\]\s*(.+)$", stripped_line)
            if m:
                num, cite = m.groups()
                out.append(f"\\bibitem{{ref{num}}} {cite}")
        out.append("\\end{thebibliography}")
        if tail:
            out.append("")
            out.extend(tail)

    return "\n".join(out)


def compile_latex() -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for aux in ("aux", "log", "out", "toc", "fls", "fdb_latexmk"):
        p = BUILD_DIR / f"research_proposal.{aux}"
        if p.exists():
            p.unlink()

    shutil.copy2(TEX_PATH, BUILD_DIR / "research_proposal.tex")
    fig_dst = BUILD_DIR / "figures"
    if fig_dst.exists():
        shutil.rmtree(fig_dst)
    shutil.copytree(FIG_DIR, fig_dst)

    cmd = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-output-directory",
        str(BUILD_DIR),
        "research_proposal.tex",
    ]
    for run in (1, 2):
        proc = subprocess.run(
            cmd,
            cwd=BUILD_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if proc.returncode != 0:
            log = BUILD_DIR / "research_proposal.log"
            tail = log.read_text(encoding="utf-8", errors="replace")[-8000:] if log.exists() else proc.stderr
            raise RuntimeError(f"pdflatex failed on run {run}:\n{tail}")

    built = BUILD_DIR / "research_proposal.pdf"
    if not built.exists():
        raise RuntimeError("pdflatex did not produce research_proposal.pdf")
    shutil.copy2(built, OUT_PDF)


def main() -> None:
    md = MD_PATH.read_text(encoding="utf-8")
    body = convert_md_to_latex(md)
    tex = PREAMBLE + body + POSTAMBLE
    TEX_PATH.write_text(tex, encoding="utf-8")
    print(f"wrote {TEX_PATH}")
    compile_latex()
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
