"""Combine research_proposal.md + unique .docx figures into a deduped PDF."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib.colors import Color, HexColor, white
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    CondPageBreak,
    Flowable,
    Frame,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent
MD_PATH = ROOT / "research_proposal.md"
FIG_DIR = ROOT / "figures"
TMP_EQ = ROOT.parent / "tmp" / "pdfs" / "eq"
OUT_DIR = ROOT.parent / "output" / "pdf"
OUT_PDF = OUT_DIR / "research_proposal.pdf"

INK = HexColor("#1c1917")
NAVY = HexColor("#1e3a4c")
ACCENT = HexColor("#9a3412")
RULE = HexColor("#d6c7b0")
MUTED = HexColor("#57534e")
SOFT = HexColor("#78716c")
ABSTRACT_BG = HexColor("#eef3f6")
ABSTRACT_BD = HexColor("#c5d0d8")
QUOTE_BG = HexColor("#f8f4ec")
QUOTE_BD = HexColor("#c4a574")
THM_BG = HexColor("#f4f1ea")
THM_BD = HexColor("#b08968")
TABLE_HEAD = HexColor("#1e3a4c")
TABLE_ALT = HexColor("#f4f1ea")
EQ_BG = HexColor("#f7f5f1")


def register_fonts() -> None:
    fonts = Path(r"C:\Windows\Fonts")
    pdfmetrics.registerFont(TTFont("Body", str(fonts / "cambria.ttc"), subfontIndex=0))
    pdfmetrics.registerFont(TTFont("Body-Bold", str(fonts / "cambriab.ttf")))
    pdfmetrics.registerFont(TTFont("Body-Italic", str(fonts / "cambriai.ttf")))
    pdfmetrics.registerFont(TTFont("Body-BoldItalic", str(fonts / "cambriaz.ttf")))
    pdfmetrics.registerFont(TTFont("Sans", str(fonts / "calibri.ttf")))
    pdfmetrics.registerFont(TTFont("Sans-Bold", str(fonts / "calibrib.ttf")))
    pdfmetrics.registerFont(TTFont("Sans-Italic", str(fonts / "calibrii.ttf")))
    pdfmetrics.registerFont(TTFont("Mono", str(fonts / "consola.ttf")))


def styles() -> dict[str, ParagraphStyle]:
    s = getSampleStyleSheet()
    out = {}
    out["title"] = ParagraphStyle(
        "TitleX",
        parent=s["Normal"],
        fontName="Sans-Bold",
        fontSize=20.5,
        leading=24.5,
        textColor=NAVY,
        alignment=TA_CENTER,
        spaceAfter=4,
    )
    out["subtitle"] = ParagraphStyle(
        "SubtitleX",
        parent=s["Normal"],
        fontName="Body-Italic",
        fontSize=11.2,
        leading=15,
        textColor=MUTED,
        alignment=TA_CENTER,
        spaceAfter=10,
    )
    out["meta"] = ParagraphStyle(
        "MetaX",
        parent=s["Normal"],
        fontName="Sans",
        fontSize=9.2,
        leading=13,
        textColor=SOFT,
        alignment=TA_CENTER,
        spaceAfter=2,
    )
    out["h1"] = ParagraphStyle(
        "H1X",
        parent=s["Normal"],
        fontName="Sans-Bold",
        fontSize=13.2,
        leading=16.5,
        textColor=NAVY,
        spaceBefore=14,
        spaceAfter=6,
        borderPadding=0,
        keepWithNext=True,
    )
    out["h2"] = ParagraphStyle(
        "H2X",
        parent=s["Normal"],
        fontName="Sans-Bold",
        fontSize=11.2,
        leading=14.5,
        textColor=HexColor("#3f4f5f"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True,
    )
    out["body"] = ParagraphStyle(
        "BodyX",
        parent=s["Normal"],
        fontName="Body",
        fontSize=10.3,
        leading=14.2,
        textColor=INK,
        alignment=TA_JUSTIFY,
        spaceAfter=7,
        firstLineIndent=0,
    )
    out["body_first"] = ParagraphStyle(
        "BodyFirst",
        parent=out["body"],
        spaceBefore=0,
    )
    out["abstract"] = ParagraphStyle(
        "AbsX",
        parent=out["body"],
        fontSize=9.8,
        leading=13.4,
        alignment=TA_JUSTIFY,
        spaceAfter=0,
    )
    out["quote"] = ParagraphStyle(
        "QuoteX",
        parent=out["body"],
        fontName="Body-Italic",
        fontSize=10.1,
        leading=13.8,
        textColor=NAVY,
        alignment=TA_LEFT,
        leftIndent=6,
        rightIndent=4,
        spaceAfter=0,
    )
    out["bullet"] = ParagraphStyle(
        "BulletX",
        parent=out["body"],
        leftIndent=14,
        firstLineIndent=0,
        spaceAfter=3.5,
        alignment=TA_LEFT,
        bulletIndent=0,
    )
    out["caption"] = ParagraphStyle(
        "CapX",
        parent=s["Normal"],
        fontName="Body-Italic",
        fontSize=8.6,
        leading=11.4,
        textColor=MUTED,
        alignment=TA_CENTER,
        spaceBefore=3,
        spaceAfter=10,
    )
    out["eq"] = ParagraphStyle(
        "EqX",
        parent=s["Normal"],
        fontName="Body-Italic",
        fontSize=10.2,
        leading=14,
        textColor=INK,
        alignment=TA_CENTER,
        spaceBefore=4,
        spaceAfter=4,
    )
    out["thm"] = ParagraphStyle(
        "ThmX",
        parent=out["body"],
        fontSize=10.1,
        leading=13.8,
        spaceAfter=0,
    )
    out["proof"] = ParagraphStyle(
        "ProofX",
        parent=out["body"],
        fontName="Body",
        fontSize=9.7,
        leading=13.2,
        textColor=HexColor("#3f3a36"),
        leftIndent=8,
        spaceAfter=6,
    )
    out["table"] = ParagraphStyle(
        "TblX",
        parent=s["Normal"],
        fontName="Body",
        fontSize=8.6,
        leading=11.2,
        textColor=INK,
    )
    out["table_h"] = ParagraphStyle(
        "TblHX",
        parent=s["Normal"],
        fontName="Sans-Bold",
        fontSize=8.4,
        leading=11,
        textColor=white,
    )
    out["ref"] = ParagraphStyle(
        "RefX",
        parent=out["body"],
        fontSize=9.2,
        leading=12.4,
        leftIndent=16,
        firstLineIndent=-16,
        spaceAfter=2,
        alignment=TA_LEFT,
    )
    out["footer"] = ParagraphStyle(
        "FootX",
        parent=s["Normal"],
        fontName="Sans",
        fontSize=8,
        textColor=SOFT,
    )
    return out


def ascii_dashes(s: str) -> str:
    return (
        s.replace("\u2011", "-")
        .replace("\u2010", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", " - ")
        .replace("\u2212", "-")
    )


SUB = str.maketrans(
    "0123456789+-=()aeioruvxknij",
    "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑᵢₒᵣᵤᵥₓₖₙᵢⱼ",
)
SUP = str.maketrans("0123456789+-=()n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ")


def _cmd(name: str, repl: str) -> tuple[str, str]:
    return rf"\\{name}\b", repl


LATEX_CMDS = [
    (r"\\mathbb\{([A-Za-z])\}", r"\1"),
    (r"\\mathbb\s*([A-Za-z])", r"\1"),
    (r"\\mathcal\{([A-Za-z])\}", r"\1"),
    (r"\\mathcal\s*([A-Za-z])", r"\1"),
    (r"\\mathrm\{erank\}", "erank"),
    (r"\\mathrm\{softmax\}", "softmax"),
    (r"\\mathrm\{tr\}", "tr"),
    (r"\\operatorname\{Var\}", "Var"),
    (r"\\operatorname\{tr\}", "tr"),
    (r"\\operatorname\{([^}]*)\}", r"\1"),
    (r"\\text\{([^}]*)\}", r"\1"),
    (r"\\textit\{([^}]*)\}", r"\1"),
    (r"\\textbf\{([^}]*)\}", r"\1"),
    (r"\\mathrm\{([^}]*)\}", r"\1"),
    (r"\\mathbf\{([^}]*)\}", r"\1"),
    (r"\\hat\{([^}]*)\}", r"\1"),
    (r"\\hat\s*([A-Za-z])", r"\1"),
    (r"\\bar\{([^}]*)\}", r"\1"),
    (r"\\bar\s*([A-Za-z])", r"\1"),
    (r"\\tilde\{([^}]*)\}", r"\1"),
    (r"\\tilde\s*([A-Za-z])", r"\1"),
    (r"\\sqrt\{([^}]*)\}", r"sqrt(\1)"),
    (r"\\sqrt\s*([A-Za-z0-9])", r"sqrt(\1)"),
    (r"\\t?frac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)"),
    (r"\\t?frac([0-9])([0-9A-Za-z])", r"(\1)/(\2)"),
    (r"\\blacksquare", "∎"),
    (r"\\square", "□"),
    (r"\\infty", "∞"),
    (r"\\subseteq", "⊆"),
    (r"\\subset", "⊂"),
    (r"\\notin", "∉"),
    (r"\\alpha", "α"),
    (r"\\beta", "β"),
    (r"\\gamma", "γ"),
    (r"\\phi", "φ"),
    (r"\\Pi", "Π"),
    (r"\\perp", "⊥"),
    (r"\\delta", "δ"),
    (r"\\Delta", "Δ"),
    (r"\\epsilon", "ε"),
    (r"\\varepsilon", "ε"),
    (r"\\kappa", "κ"),
    (r"\\lambda", "λ"),
    (r"\\mu", "μ"),
    (r"\\nu", "ν"),
    (r"\\rho", "ρ"),
    (r"\\sigma", "σ"),
    (r"\\ell", "ℓ"),
    (r"\\top", "T"),
    (r"\\cdot", "·"),
    (r"\\times", "×"),
    (r"\\otimes", "⊗"),
    (r"\\odot", "⊙"),
    (r"\\approx", "≈"),
    (r"\\simeq", "≃"),
    (r"\\sim", "~"),
    (r"\\propto", "∝"),
    (r"\\gtrsim", "≳"),
    (r"\\lesssim", "≲"),
    (r"\\gg", "≫"),
    (r"\\ll", "≪"),
    (r"\\geq", "≥"),
    (r"\\ge", "≥"),
    (r"\\leq", "≤"),
    (r"\\le", "≤"),
    (r"\\neq", "≠"),
    (r"\\equiv", "≡"),
    (r"\\in", "∈"),
    (r"\\cup", "∪"),
    (r"\\cap", "∩"),
    (r"\\wedge", "∧"),
    (r"\\vee", "∨"),
    (r"\\partial", "∂"),
    (r"\\nabla", "∇"),
    (r"\\sum", "Σ"),
    (r"\\prod", "Π"),
    (r"\\int", "∫"),
    (r"\\log", "log"),
    (r"\\cos", "cos"),
    (r"\\sin", "sin"),
    (r"\\exp", "exp"),
    (r"\\max", "max"),
    (r"\\min", "min"),
    (r"\\arg", "arg"),
    (r"\\rightarrow", "→"),
    (r"\\Rightarrow", "⇒"),
    (r"\\Leftrightarrow", "⇔"),
    (r"\\Longleftrightarrow", "⇔"),
    (r"\\mapsto", "↦"),
    (r"\\to", "→"),
    (r"\\langle", "⟨"),
    (r"\\rangle", "⟩"),
    (r"\\\|", "‖"),
    (r"\\ldots", "..."),
    (r"\\cdots", "···"),
    (r"\\dots", "..."),
    (r"\\forall", "∀"),
    (r"\\exists", "∃"),
    (r"\\emptyset", "∅"),
    (r"\\circ", "◦"),
    (r"\\ast", "*"),
    (r"\\star", "*"),
    (r"\\pm", "±"),
    (r"\\mp", "∓"),
    (r"\\colon", ":"),
    (r"\\mid", "|"),
    (r"\\middle", ""),
    (r"\\,\\", " "),
    (r"\\,", " "),
    (r"\\;", " "),
    (r"\\:", " "),
    (r"\\!", ""),
    (r"\\qquad", "   "),
    (r"\\quad", "  "),
    (r"\\bigg", ""),
    (r"\\Bigg", ""),
    (r"\\big", ""),
    (r"\\Big", ""),
    (r"\\left", ""),
    (r"\\right", ""),
    (r"\\displaystyle", ""),
    (r"\\textstyle", ""),
    (r"\\nolimits", ""),
    (r"\\limits", ""),
]


def latex_to_unicode(tex: str) -> str:
    t = tex.strip()
    t = t.replace(r"\!", "")
    # Remove sizing commands before shorter commands such as \le are matched.
    t = (
        t.replace(r"\left", "")
        .replace(r"\right", "")
        .replace(r"\middle", "")
        .replace(r"\bigg", "")
        .replace(r"\Bigg", "")
        .replace(r"\big", "")
        .replace(r"\Big", "")
    )
    for pat, repl in LATEX_CMDS:
        t = re.sub(pat, repl, t)
    # superscripts / subscripts of a single token
    def sub_fn(m: re.Match) -> str:
        body = m.group(1)
        if len(body) <= 3 and all(c in "0123456789+-=()aeioruvxknijq" for c in body):
            return body.translate(SUB)
        return "_" + body

    def sup_fn(m: re.Match) -> str:
        body = m.group(1)
        if body == "\\top" or body == "T":
            return "ᵀ"
        if len(body) <= 3 and all(c in "0123456789+-=()nT" for c in body):
            return body.translate(SUP)
        return "^" + body

    t = re.sub(r"\^\{([^}]+)\}", sup_fn, t)
    t = re.sub(r"_\{([^}]+)\}", sub_fn, t)
    t = re.sub(r"\^([A-Za-z0-9])", lambda m: m.group(1).translate(SUP) if m.group(1) in "0123456789n" else "^" + m.group(1), t)
    t = re.sub(r"_([A-Za-z0-9])", lambda m: m.group(1).translate(SUB) if m.group(1) in "0123456789aeioruvxknijq" else "_" + m.group(1), t)
    t = t.replace(r"\{", "{").replace(r"\}", "}")
    t = re.sub(r"\\[A-Za-z]+", "", t)
    t = t.replace("{", "").replace("}", "")
    t = t.replace("\\", "")
    t = re.sub(r"\s+", " ", t).strip()
    return ascii_dashes(t)


HARD_MATH = re.compile(
    r"\\(mathbb|mathcal|operatorname|Big|bigg|left|right|langle|rangle|dots|blacksquare|Vert)"
)


def prep_mathtext(tex: str) -> str:
    t = tex.strip()
    t = t.replace(r"\!", "")
    t = re.sub(r"\\rm\s+([A-Za-z]+)", r"\\mathrm{\1}", t)
    t = re.sub(r"\\mathbb\s*\{?([A-Za-z])\}?", r"\\mathrm{\1}", t)
    t = re.sub(r"\\mathcal\s*\{?([A-Za-z])\}?", r"\\mathcal{\1}", t)
    t = re.sub(r"\\sqrt\s*([A-Za-z0-9])", r"\\sqrt{\1}", t)
    t = t.replace(r"\operatorname{Var}", r"\mathrm{Var}")
    t = t.replace(r"\operatorname{tr}", r"\mathrm{tr}")
    t = re.sub(r"\\text\{([^}]*)\}", r"\\mathrm{\1}", t)
    t = re.sub(r"\\t?frac([0-9])([0-9A-Za-z])", r"\\frac{\1}{\2}", t)
    t = re.sub(
        r"\\t?frac\s*([A-Za-z0-9])\s*([A-Za-z0-9])",
        r"\\frac{\1}{\2}",
        t,
    )
    t = t.replace(r"\tfrac", r"\frac")
    t = t.replace(r"\|", r"|")
    t = t.replace(r"\middle", "")
    t = t.replace(r"\mid", r"|")
    t = t.replace(r"\langle", r"\langle ")
    t = t.replace(r"\rangle", r"\rangle ")
    t = t.replace(r"\big", "").replace(r"\Big", "")
    t = t.replace(r"\bigg", "").replace(r"\Bigg", "")
    t = t.replace(r"\left", "").replace(r"\right", "")
    t = t.replace(r"\displaystyle", "")
    t = t.replace(r"\nolimits", "")
    t = t.replace(r"\limits", "")
    t = t.replace(r"\qquad", r"\quad")
    t = t.replace(r"\dots", r"\ldots")
    t = t.replace(r"\blacksquare", r"")
    return t.rstrip(".")


def _mathtext_ok(mt: str) -> bool:
    try:
        from matplotlib import mathtext

        mathtext.MathTextParser("agg").parse(f"${mt}$")
        return True
    except Exception:
        return False


def render_equation(tex: str, width_px: int = 980) -> Path:
    TMP_EQ.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(tex.encode("utf-8")).hexdigest()[:16]
    path = TMP_EQ / f"eq_{key}.png"
    if path.exists():
        return path
    mt = re.sub(r"\s+", " ", prep_mathtext(tex)).strip()
    fig = plt.figure(figsize=(width_px / 140, 0.58), dpi=160)
    fig.patch.set_facecolor("white")
    if mt and _mathtext_ok(mt):
        fig.text(0.5, 0.5, f"${mt}$", ha="center", va="center", fontsize=12, color="#1c1917")
    else:
        fig.text(0.5, 0.5, latex_to_unicode(tex), ha="center", va="center", fontsize=11, color="#1c1917")
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close(fig)
    return path


def escape_xml(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def inline_md_to_xml(text: str) -> str:
    """Markdown inline (math, bold, italic) -> reportlab XML."""
    text = ascii_dashes(text)
    parts: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "$" and not text.startswith("$$", i):
            j = text.find("$", i + 1)
            if j != -1:
                math = latex_to_unicode(text[i + 1 : j])
                parts.append(f'<font name="Body-Italic">{escape_xml(math)}</font>')
                i = j + 1
                continue
        if text.startswith("**", i):
            j = text.find("**", i + 2)
            if j != -1:
                inner = inline_md_to_xml(text[i + 2 : j])
                parts.append(f"<b>{inner}</b>")
                i = j + 2
                continue
        if text[i] == "*" and (i + 1 < n and text[i + 1] != "*"):
            j = text.find("*", i + 1)
            if j != -1 and not text.startswith("**", j):
                inner = inline_md_to_xml(text[i + 1 : j])
                parts.append(f"<i>{inner}</i>")
                i = j + 1
                continue
        if text.startswith("`", i):
            j = text.find("`", i + 1)
            if j != -1:
                parts.append(f'<font name="Mono" size="8.6">{escape_xml(text[i + 1 : j])}</font>')
                i = j + 1
                continue
        # consume until next special
        nxt = n
        for tok in ("$", "**", "*", "`"):
            k = text.find(tok, i + 1)
            if k != -1:
                nxt = min(nxt, k)
        chunk = text[i:nxt]
        # leftover $... without close
        parts.append(escape_xml(chunk))
        i = nxt
    return "".join(parts)


class HRule(Flowable):
    def __init__(self, color=RULE, thickness=0.6, space=6):
        super().__init__()
        self.color = color
        self.thickness = thickness
        self.space = space
        self.height = space * 2 + thickness
        self.width = 0

    def wrap(self, aw, ah):
        self.width = aw
        return aw, self.height

    def draw(self):
        self.canv.setStrokeColor(self.color)
        self.canv.setLineWidth(self.thickness)
        y = self.space
        self.canv.line(0, y, self.width, y)


class AccentBar(Flowable):
    def __init__(self, color=ACCENT, height=3.2):
        super().__init__()
        self.color = color
        self._h = height
        self.width = 0

    def wrap(self, aw, ah):
        self.width = aw
        return aw, self._h + 4

    def draw(self):
        self.canv.setFillColor(self.color)
        w = min(72, self.width)
        self.canv.roundRect((self.width - w) / 2, 2, w, self._h, 1.2, fill=1, stroke=0)


def boxed(story_items, width, bg, border, pad=8, radius=4) -> Table:
    inner = story_items
    data = [[inner]]
    t = Table(data, colWidths=[width])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), bg),
                ("BOX", (0, 0), (-1, -1), 0.6, border),
                ("LEFTPADDING", (0, 0), (-1, -1), pad),
                ("RIGHTPADDING", (0, 0), (-1, -1), pad),
                ("TOPPADDING", (0, 0), (-1, -1), pad - 1),
                ("BOTTOMPADDING", (0, 0), (-1, -1), pad - 1),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return t


def eq_block(tex: str, max_w: float, sty: dict | None = None) -> Flowable:
    cleaned = latex_to_unicode(tex)
    eq_style = (sty or {}).get("eq") or ParagraphStyle(
        "eqtmp", fontName="Body-Italic", fontSize=10.2, leading=14, alignment=TA_CENTER, textColor=INK
    )
    inner: Flowable = Paragraph(escape_xml(cleaned), eq_style)
    try:
        path = render_equation(tex)
        img = Image(str(path))
        iw, ih = img.imageWidth, img.imageHeight
        scale = min(1.0, (max_w - 28) / max(iw, 1))
        img.drawWidth = iw * scale
        img.drawHeight = min(ih * scale, 1.2 * inch)
        img.hAlign = "CENTER"
        inner = img
    except Exception:
        inner = Paragraph(escape_xml(cleaned), eq_style)
    wrap = Table([[inner]], colWidths=[max_w])
    wrap.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), EQ_BG),
                ("BOX", (0, 0), (-1, -1), 0.4, RULE),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ]
        )
    )
    return wrap


def figure_block(path: Path, caption: str, max_w: float, sty) -> list:
    if not path.exists():
        return [Paragraph(inline_md_to_xml(f"*Missing figure:* {path.name}"), sty["caption"])]
    img = Image(str(path))
    iw, ih = img.imageWidth, img.imageHeight
    scale = min(1.0, max_w / iw)
    # cap height so a figure never eats a whole page
    max_h = 4.6 * inch
    if ih * scale > max_h:
        scale = max_h / ih
    img.drawWidth = iw * scale
    img.drawHeight = ih * scale
    img.hAlign = "CENTER"
    return [Spacer(1, 8), img, Paragraph(inline_md_to_xml(caption), sty["caption"])]


THM_PREFIXES = (
    "Definition ",
    "Lemma ",
    "Proposition ",
    "Theorem ",
    "Corollary ",
    "Remarks.",
    "Remark.",
    "Consequence.",
    "Interpretation.",
    "Design principle.",
)


def is_thm(text: str) -> bool:
    plain = re.sub(r"[*_]", "", text).lstrip()
    return plain.startswith(THM_PREFIXES)


def is_proof(text: str) -> bool:
    plain = text.lstrip()
    return plain.startswith("*Proof") or plain.startswith("**Proof") or plain.startswith("Proof.")


def parse_md(md: str) -> list[tuple]:
    """Return a list of typed blocks from markdown."""
    md = md.replace("\r\n", "\n")
    lines = md.split("\n")
    blocks: list[tuple] = []
    i = 0
    n = len(lines)

    def flush_para(buf: list[str]) -> None:
        text = " ".join(x.strip() for x in buf if x.strip())
        if text:
            blocks.append(("p", text))

    while i < n:
        line = lines[i]
        stripped = line.strip()

        if stripped == "$$":
            i += 1
            eq = []
            while i < n and lines[i].strip() != "$$":
                eq.append(lines[i])
                i += 1
            blocks.append(("eq", "\n".join(eq)))
            i += 1
            continue

        if stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) > 4:
            blocks.append(("eq", stripped[2:-2]))
            i += 1
            continue

        if stripped == "---":
            blocks.append(("hr", None))
            i += 1
            continue

        if stripped.startswith("#"):
            m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
            if m:
                blocks.append(("h", len(m.group(1)), m.group(2).strip()))
                i += 1
                continue

        if stripped.startswith("|") and i + 1 < n and re.match(r"^\s*\|?\s*-+", lines[i + 1]):
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                row = [
                    c.strip()
                    for c in re.split(
                        r"(?<!\\)\|", lines[i].strip().strip("|")
                    )
                ]
                if not all(
                    re.match(r"^:?-+:?$", c.replace(" ", ""))
                    for c in row
                ):
                    rows.append(row)
                i += 1
            blocks.append(("table", rows))
            continue

        if stripped.startswith(">"):
            q = []
            while i < n and lines[i].strip().startswith(">"):
                q.append(re.sub(r"^>\s?", "", lines[i].strip()))
                i += 1
            blocks.append(("quote", " ".join(q)))
            continue

        if re.match(r"^[-*]\s+", stripped):
            items = []
            while i < n and re.match(r"^[-*]\s+", lines[i].strip()):
                items.append(re.sub(r"^[-*]\s+", "", lines[i].strip()))
                i += 1
            blocks.append(("ul", items))
            continue

        if re.match(r"^\d+\.\s+", stripped):
            items = []
            while i < n and re.match(r"^\d+\.\s+", lines[i].strip()):
                items.append(
                    re.sub(r"^\d+\.\s+", "", lines[i].strip())
                )
                i += 1
            blocks.append(("ol", items))
            continue

        if not stripped:
            i += 1
            continue

        buf = [stripped]
        i += 1
        while i < n:
            nxt = lines[i].strip()
            if not nxt or nxt.startswith("#") or nxt == "---" or nxt.startswith("|") or nxt.startswith(">") or nxt == "$$" or re.match(r"^[-*]\s+", nxt):
                break
            buf.append(nxt)
            i += 1
        flush_para(buf)

    return blocks


def header_footer(canvas, doc):
    canvas.saveState()
    w, h = letter
    canvas.setFillColor(NAVY)
    canvas.rect(0, h - 28, w, 28, fill=1, stroke=0)
    canvas.setFillColor(ACCENT)
    canvas.rect(0, h - 31, w, 3, fill=1, stroke=0)
    canvas.setFillColor(white)
    canvas.setFont("Sans", 8)
    canvas.drawString(0.8 * inch, h - 19, "The Geometry of Transformer Initialization")
    canvas.drawRightString(w - 0.8 * inch, h - 19, "Research Proposal  ·  v2.0")
    canvas.setFillColor(RULE)
    canvas.rect(0, 0, w, 32, fill=1, stroke=0)
    canvas.setFillColor(NAVY)
    canvas.setFont("Sans", 8)
    canvas.drawString(0.8 * inch, 14, "Vidit Gupta")
    canvas.drawCentredString(w / 2, 14, f"{doc.page}")
    canvas.drawRightString(w - 0.8 * inch, 14, "Deep Learning Theory")
    canvas.restoreState()


def should_skip_block(kind, payload, heading_text: str | None) -> bool:
    """Drop appendix figure-caption list; figures are placed in-body."""
    if kind == "h":
        return False
    if heading_text and heading_text.strip().startswith("A2."):
        return True
    if kind == "p" and isinstance(payload, str) and payload.strip().startswith("- **Figure"):
        return True
    if kind == "ul" and heading_text and "Figure captions" in heading_text:
        return True
    return False


def build_story(blocks, sty, frame_w: float) -> list:
    story: list = []
    current_h = ""
    seen_title = False
    inserted = {"fig1": False, "fig2": False, "fig3": False}
    meta_lines: list[str] = []
    collecting_meta = False

    def maybe_insert_figures(next_heading: str) -> None:
        if next_heading.startswith("4.5") and not inserted["fig2"]:
            story.extend(
                figure_block(
                    FIG_DIR / "fig_entropy.png",
                    "Figure 1. Exact finite-width logit moments versus Monte Carlo simulation. "
                    "Panels (a) and (b) test Proposition 1 across alignment; panel (c) resolves the "
                    "finite-width variance difference between the orthogonal and independent channels.",
                    frame_w,
                    sty,
                )
            )
            inserted["fig2"] = True
        if next_heading.startswith("4.8") and not inserted["fig3"]:
            story.extend(
                figure_block(
                    FIG_DIR / "fig_diversity.png",
                    "Figure 2. The exact uncentered common-bias statistic R_12 from Proposition 5. "
                    "Centering removes the ensemble mean, so this statistic is not a functional "
                    "head-diversity guarantee.",
                    frame_w,
                    sty,
                )
            )
            inserted["fig3"] = True
        if next_heading.startswith("6.") and not inserted["fig1"]:
            story.extend(
                figure_block(
                    FIG_DIR / "fig_phase.png",
                    "Figure 3. Finite-n unmasked i.i.d.-Gaussian screening surrogate in (s, alpha) at fixed "
                    "n=64, m=16, and r=0.25. Contours are operational guides, not trainability "
                    "theorems; Experiment 1 replaces them with a replicated empirical atlas.",
                    frame_w,
                    sty,
                )
            )
            inserted["fig1"] = True

    i = 0
    while i < len(blocks):
        b = blocks[i]
        kind = b[0]

        if kind == "h":
            level, text = b[1], b[2]
            if level == 1 and not seen_title:
                # Title + subtitle + meta
                story.append(Spacer(1, 8))
                story.append(Paragraph(inline_md_to_xml(text), sty["title"]))
                seen_title = True
                collecting_meta = True
                i += 1
                continue
            if level == 2 and collecting_meta and not text.startswith("Abstract"):
                story.append(Paragraph(inline_md_to_xml(text), sty["subtitle"]))
                story.append(AccentBar())
                story.append(Spacer(1, 6))
                i += 1
                continue
            if collecting_meta and level >= 2 and text.startswith("Abstract"):
                collecting_meta = False
                if meta_lines:
                    story.append(Paragraph("<br/>".join(inline_md_to_xml(x) for x in meta_lines), sty["meta"]))
                    story.append(Spacer(1, 8))
                    meta_lines = []
                current_h = text
                label = Paragraph('<font name="Sans-Bold" color="#9a3412" size="8">ABSTRACT</font>', sty["meta"])
                i += 1
                abs_paras = []
                while i < len(blocks) and blocks[i][0] != "h":
                    bk = blocks[i]
                    if bk[0] == "hr":
                        i += 1
                        continue
                    if bk[0] == "eq":
                        abs_paras.append(eq_block(bk[1], frame_w - 28, sty))
                    elif bk[0] == "p":
                        abs_paras.append(Paragraph(inline_md_to_xml(bk[1]), sty["abstract"]))
                    i += 1
                inner = []
                inner.append(label)
                inner.append(Spacer(1, 4))
                inner.extend(abs_paras)
                story.append(boxed(inner, frame_w, ABSTRACT_BG, ABSTRACT_BD, pad=10))
                story.append(Spacer(1, 8))
                continue

            maybe_insert_figures(text)
            current_h = text
            if text.strip() == "Appendix":
                story.append(CondPageBreak(120))
            if text.startswith("Appendix A."):
                story.append(PageBreak())
            if text.strip() == "References":
                story.append(CondPageBreak(140))
            # Skip A2 heading; replace with a short pointer
            if text.startswith("A2."):
                i += 1
                while i < len(blocks) and blocks[i][0] in ("ul", "p", "hr"):
                    payload = blocks[i][1] if len(blocks[i]) > 1 else ""
                    blob = str(payload)
                    if "Figure" in blob or (blocks[i][0] == "ul" and any("Figure" in x for x in payload)):
                        i += 1
                        continue
                    break
                continue
            style = sty["h1"] if level <= 2 else sty["h2"]
            # markdown ### is level 3
            if level == 2 and not re.match(r"^\d", text) and text not in ("Abstract",):
                # document subtitle already handled; remaining ## are major sections
                style = sty["h1"]
            if level >= 3:
                style = sty["h2"]
            story.append(Paragraph(inline_md_to_xml(text), style))
            i += 1
            continue

        if collecting_meta and kind == "p":
            parts = re.split(r"(?=\*\*(?:Author|Field|Status):)", b[1])
            meta_lines.extend(p.strip() for p in parts if p.strip())
            i += 1
            continue

        if kind == "hr":
            if not collecting_meta:
                story.append(HRule())
            i += 1
            continue

        if kind == "eq":
            story.append(Spacer(1, 2))
            story.append(eq_block(b[1], frame_w, sty))
            story.append(Spacer(1, 4))
            i += 1
            continue

        if kind == "quote":
            q = Paragraph(inline_md_to_xml(b[1]), sty["quote"])
            story.append(boxed([q], frame_w, QUOTE_BG, QUOTE_BD, pad=9))
            story.append(Spacer(1, 8))
            i += 1
            continue

        if kind in ("ul", "ol"):
            if current_h.startswith("A2."):
                i += 1
                continue
            if current_h.startswith("A3."):
                joined = " &nbsp;·&nbsp; ".join(inline_md_to_xml(it) for it in b[1])
                story.append(Paragraph(joined, sty["body"]))
            else:
                for number, it in enumerate(b[1], start=1):
                    marker = (
                        f'<font color="#9a3412"><b>{number}.</b></font>'
                        if kind == "ol"
                        else '<font color="#9a3412"><b>•</b></font>'
                    )
                    story.append(
                        Paragraph(
                            f"{marker}  {inline_md_to_xml(it)}",
                            sty["bullet"],
                        )
                    )
                story.append(Spacer(1, 4))
            i += 1
            continue

        if kind == "table":
            rows = b[1]
            styled = []
            for ridx, row in enumerate(rows):
                st = sty["table_h"] if ridx == 0 else sty["table"]
                styled.append([Paragraph(inline_md_to_xml(c), st) for c in row])
            col_w = [1.15 * inch, frame_w - 1.15 * inch]
            if len(rows[0]) == 3:
                col_w = [frame_w * 0.28, frame_w * 0.36, frame_w * 0.36]
            elif len(rows[0]) != 2:
                col_w = [frame_w / len(rows[0])] * len(rows[0])
            tbl = Table(styled, colWidths=col_w, repeatRows=1)
            cmds = [
                ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEAD),
                ("TEXTCOLOR", (0, 0), (-1, 0), white),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("GRID", (0, 0), (-1, -1), 0.35, HexColor("#d4cdc3")),
                ("BOX", (0, 0), (-1, -1), 0.7, NAVY),
            ]
            for ridx in range(1, len(styled)):
                if ridx % 2 == 0:
                    cmds.append(("BACKGROUND", (0, ridx), (-1, ridx), TABLE_ALT))
            tbl.setStyle(TableStyle(cmds))
            story.append(Spacer(1, 4))
            story.append(tbl)
            story.append(Spacer(1, 8))
            i += 1
            continue

        if kind == "p":
            text = b[1]
            if current_h.startswith("References"):
                story.append(Paragraph(inline_md_to_xml(text), sty["ref"]))
                i += 1
                continue
            if is_proof(text):
                story.append(Paragraph(inline_md_to_xml(text), sty["proof"]))
                i += 1
                continue
            if is_thm(text):
                story.append(boxed([Paragraph(inline_md_to_xml(text), sty["thm"])], frame_w, THM_BG, THM_BD, pad=8))
                story.append(Spacer(1, 7))
                i += 1
                continue
            story.append(Paragraph(inline_md_to_xml(text), sty["body"]))
            i += 1
            continue

        i += 1

    # leftover figures if headings were missing
    if not inserted["fig2"]:
        maybe_insert_figures("4.5")
    if not inserted["fig3"]:
        maybe_insert_figures("4.8")
    if not inserted["fig1"]:
        maybe_insert_figures("6.")
    return story


def main() -> None:
    register_fonts()
    sty = styles()
    md = MD_PATH.read_text(encoding="utf-8")
    blocks = parse_md(md)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_EQ.mkdir(parents=True, exist_ok=True)

    page_w, page_h = letter
    left = right = 0.78 * inch
    top = 0.70 * inch
    bottom = 0.62 * inch
    frame_w = page_w - left - right

    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=letter,
        leftMargin=left,
        rightMargin=right,
        topMargin=top,
        bottomMargin=bottom,
        title="The Geometry of Transformer Initialization",
        author="Vidit Gupta",
        subject="Exact query-key moment identities and a falsifiable initialization atlas",
    )
    story = build_story(blocks, sty, frame_w)
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    print(f"wrote {OUT_PDF}  pages~check")


if __name__ == "__main__":
    main()
