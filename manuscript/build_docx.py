"""Build the Word manuscript for Molecular Informatics from main.tex (pandoc + python-docx post-processing)."""
import re, subprocess, sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent
tex = (HERE / "main.tex").read_text()
tex = re.sub(r"^\\(newcommand\{\\(ecfp|repourl|repodoi|fittable|todo)\}|input\{tables/numbers\}|newsavebox|usepackage\[[^\]]*\]\{lineno\}|linenumbers|onehalfspacing|setlength\{\\emergencystretch\}).*$\n", "", tex, flags=re.M)
aux = (HERE / "main.aux").read_text()
# --- resolve \ref / \eqref numbers from the LaTeX aux file
labels = dict(re.findall(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", aux))
def ref(m):
    key = m.group(2); num = labels.get(key, "??")
    num = re.sub(r"\\[a-zA-Z]+\s*", "", num).strip("{}")
    return f"({num})" if m.group(1) == "eqref" else num
tex = re.sub(r"\\(ref|eqref)\{([^}]+)\}", ref, tex)
# textual author-year style for \citet in a numeric journal: "Surname et al. [n]"
bib = (HERE / "references.bib").read_text()
def first_author(key):
    i = bib.find("{" + key + ","); j = bib.find("author={", i)
    if i < 0 or j < 0: return key
    k = j + len("author={"); depth = 1; field = ""
    while depth:
        c = bib[k]; depth += (c == "{") - (c == "}")
        if depth: field += c
        k += 1
    auth = [a.strip() for a in field.split(" and ")]
    def sur(a): return a.split(",")[0].strip() if "," in a else a.split()[-1]
    if len(auth) == 1: return sur(auth[0])
    if len(auth) == 2: return sur(auth[0]) + " and " + sur(auth[1])
    return sur(auth[0]) + " et al."
tex = re.sub(r"\\citet\{([^}]+)\}", lambda m: first_author(m.group(1)) + " \\citep{" + m.group(1) + "}", tex)
tex = re.sub(r"\\paragraph\{([^}]*)\}\s*", lambda m: "\\textbf{" + m.group(1) + "} ", tex)
tex = tex.replace("~", "\u00a0")  # non-breaking spaces in "Table 3", "Section 2"
# --- numbers macros
nums = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", (HERE / "tables/numbers.tex").read_text()))
for k, v in sorted(nums.items(), key=lambda kv: -len(kv[0])):
    tex = tex.replace("\\" + k + "{}", v); tex = re.sub(r"\\" + k + r"(?![A-Za-z])", v, tex)
tex = tex.replace(r"\ecfp{}", "ECFP4").replace(r"\ecfp", "ECFP4")
tex = tex.replace(r"\repourl{}", r"\url{https://github.com/ismankrn/sider-adr-benchmark}")
tex = tex.replace(r"\repodoi{}", r"\href{https://doi.org/10.5281/zenodo.23798100}{10.5281/zenodo.23798100}")
# --- simple inline math -> Unicode text (keeps Word tables compact); display equations are untouched
simple = {r"$>$": ">", r"$\pm$": "±", r"$<$": "<", r"$\le$": "≤", r"$\ge$": "≥", r"$\Delta$": "Δ", r"$\times$": "×",
          r"$p$": r"\emph{p}", r"$d$": r"\emph{d}", r"$n$": r"\emph{n}", r"$t$": r"\emph{t}", r"$k$": r"\emph{k}", r"$C$": r"\emph{C}",
          r"$n_\mathrm{train}$": r"\emph{n}\textsubscript{train}", r"$\pm1.5$": "±1.5", r"$10^{-3}$": "10⁻³", r"$10^{-5}$": "10⁻⁵",
          r"$10^{-4}$": "10⁻⁴", r"$10^{-2}$": "10⁻²", r"$5\times10^{-5}$": "5×10⁻⁵", r"$3\times10^{-5}$": "3×10⁻⁵",
          r"$8\times10^{-3}$": "8×10⁻³", r"$2\times10^{-4}$": "2×10⁻⁴", r"$C=1$": r"\emph{C} = 1", r"$C=0.1$": r"\emph{C} = 0.1",
          r"$n_-/n_+$": r"\emph{n}\textsubscript{−}/\emph{n}\textsubscript{+}", r"$1 \times 64$": "1 × 64", r"$2 \times 64$": "2 × 64", r"$3 \times 64$": "3 × 64"}
# --- inline the generated tables, drop size/fit wrappers
def inp(m):
    return (HERE / f"tables/{m.group(1)}.tex").read_text()
tex = re.sub(r"\\input\{tables/(\w+)\}", inp, tex)
FLAT = {
 "Model & \\multicolumn{3}{c}{At threshold 0.5} & \\multicolumn{2}{c}{MCC} & \\multicolumn{2}{c}{Balanced acc.} & \\multicolumn{2}{c}{F1} & \\multicolumn{2}{c}{Threshold} \\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}\\cmidrule(lr){11-12}\n & Precision & Recall & Specificity & 0.5 & tuned & 0.5 & tuned & 0.5 & tuned & mean & s.d. \\\\":
 "Model & Prec. (0.5) & Rec. (0.5) & Spec. (0.5) & MCC (0.5) & MCC (tuned) & BA (0.5) & BA (tuned) & F1 (0.5) & F1 (tuned) & Thr. mean & Thr. s.d. \\\\",
 "Optimiser & Best val.\\ AUROC & Test AUROC (strat.) & Test AUROC (scaffold) & Test MCC (strat.) & \\multicolumn{2}{c}{$\\Delta$AUROC vs.\\ random ($d$), $p$ / Holm $p$} \\\\\n\\cmidrule(lr){6-7} & & & & & stratified & scaffold \\\\":
 "Optimiser & Best val.\\ AUROC & Test AUROC (strat.) & Test AUROC (scaffold) & Test MCC (strat.) & $\\Delta$AUROC vs.\\ random ($d$), $p$ / Holm $p$, stratified & $\\Delta$AUROC vs.\\ random ($d$), $p$ / Holm $p$, scaffold \\\\",
 "Model & \\multicolumn{2}{c}{SIDER (250 to 1{,}077)} & \\multicolumn{2}{c}{Tox21 (250 to 5{,}087)} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n & Gain per doubling & Total gain & Gain per doubling & Total gain \\\\":
 "Model & SIDER gain per doubling (250 to 1{,}077) & SIDER total gain & Tox21 gain per doubling (250 to 5{,}087) & Tox21 total gain \\\\",
 "Dataset & $n_\\mathrm{train}$ & \\multicolumn{3}{c}{CL tuned (PSO) vs.\\ CL1 default} & \\multicolumn{3}{c}{RF-ECFP4 vs.\\ CL tuned (PSO)} \\\\\n\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\n & & $\\Delta$ & $p$ / Holm $p$ & $d$ & $\\Delta$ & $p$ / Holm $p$ & $d$ \\\\":
 "Dataset & $n_\\mathrm{train}$ & $\\Delta$ tuned vs.\\ default & $p$ / Holm $p$ & $d$ & $\\Delta$ RF vs.\\ tuned & $p$ / Holm $p$ & $d$ \\\\",
}
for k_, v_ in FLAT.items():
    assert k_ in tex, k_[:60]
    tex = tex.replace(k_, v_)
for k_, v_ in sorted(simple.items(), key=lambda kv: -len(kv[0])): tex = tex.replace(k_, v_)
# statistics in parentheses such as $p < 0.001$, $d = 1.86$, $p \le 0.002$ -> italic symbol + text
tex = re.sub(r"\$([pdt])\s*(<|>|=|\\le|\\ge)\s*(-?[0-9.]+)\$", lambda m: "\\emph{%s} %s %s" % (m.group(1), {"<":"<",">":">","=":"=","\\le":"≤","\\ge":"≥"}[m.group(2)], m.group(3).replace("-", "−")), tex)
tex = re.sub(r"\$([+-]?[0-9.]+)\$", lambda m: m.group(1).replace("-", "−"), tex)
tex = re.sub(r"\\quad\s*|\\hspace\{[^}]*\}", "", tex)
tex = re.sub(r"\\(small|scriptsize|footnotesize|centering)\s*", "", tex)
tex = re.sub(r"\\fittable\{", "{", tex)
tex = re.sub(r"\\setlength\{\\tabcolsep\}\{[^}]*\}|\\renewcommand\{\\arraystretch\}\{[^}]*\}", "", tex)
tex = re.sub(r"\\cmidrule\(lr\)\{[^}]*\}", "", tex)
# --- three-panel attribution figure -> one composed image
tex = re.sub(r"\\includegraphics\[width=\.32\\textwidth\]\{rank00_idx1030\}\\hfill\n\\includegraphics\[width=\.32\\textwidth\]\{rank06_idx450\}\\hfill\n\\includegraphics\[width=\.32\\textwidth\]\{rank11_idx137\}", r"\\includegraphics[width=\\textwidth]{attribution_panels}", tex)
# --- figures: use PNGs
tex = re.sub(r"\\includegraphics\[([^\]]*)\]\{([^}]+)\}", lambda m: f"\\includegraphics[{m.group(1)}]{{figures/{pathlib.Path(m.group(2)).name}.png}}" if not m.group(2).startswith("rank") else f"\\includegraphics[{m.group(1)}]{{../code/figures/attribution/{m.group(2)}.png}}", tex)
tex = re.sub(r"\\graphicspath\{[^\n]*\}\n", "", tex)
# --- drop LaTeX-only preamble lines
for pat in [r"\\newcommand\{\\(ecfp|repourl|repodoi)\}[^\n]*\n", r"\\input\{tables/numbers\}\n", r"\\newsavebox[^\n]*\n", r"\\usepackage\[[^\]]*\]\{lineno\}\n", r"\\linenumbers\n", r"\\onehalfspacing\n", r"\\setlength\{\\emergencystretch\}\{1em\}\n", r"\\newsavebox\{\\tablebox\}\n", r"\\newcommand\{\\fittable\}[^\n]*\n", r"\\newcommand\{\\todo\}[^\n]*\n"]:
    tex = re.sub(pat, "", tex)
(HERE / "build" ).mkdir(exist_ok=True)
(HERE / "build/main_docx.tex").write_text(tex)
csl = sys.argv[1] if len(sys.argv) > 1 else "angewandte-chemie.csl"
cmd = ["pandoc", "build/main_docx.tex", "-f", "latex", "-t", "docx", "--number-sections", "--citeproc", "--bibliography", "references.bib", "--csl", csl,
       "--resource-path", ".:figures:../code/figures/attribution", "-o", "build/manuscript_MolInf.docx"]
if (HERE / "reference.docx").exists(): cmd += ["--reference-doc", "reference.docx"]
print(" ".join(cmd)); subprocess.run(cmd, cwd=HERE, check=True)
# --- post-processing with python-docx
import docx
from docx.shared import Cm, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
d = docx.Document(HERE / "build/manuscript_MolInf.docx")
for s in d.styles:
    try:
        if s.type == 1: s.font.name = "Arial"
    except Exception: pass
d.styles["Normal"].font.size = Pt(10)
for p in d.paragraphs:
    for r in p.runs:
        if r._element.xpath(".//w:drawing"):
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for shp in r._element.xpath(".//wp:extent"):
                cx = int(shp.get("cx")); cy = int(shp.get("cy"))
                maxw = Cm(17.5)
                if cx > maxw:
                    shp.set("cy", str(int(cy * maxw / cx))); shp.set("cx", str(int(maxw)))
                    for ext in r._element.xpath(".//a:ext"):
                        ext.set("cx", str(int(maxw))); ext.set("cy", str(int(cy * maxw / cx)))
for t in d.tables:
    t.style = d.styles["Table Grid"] if "Table Grid" in [s.name for s in d.styles] else t.style
    for row in t.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                for r in p.runs: r.font.size = Pt(8)
# --- title page: rebuild the author block with superscript footnote letters and one line per affiliation
import copy
for p in d.paragraphs:
    if p.style.name == "Author":
        for r in list(p.runs): r._element.getparent().remove(r._element)
        for om in p._element.xpath(".//m:oMath") + p._element.xpath(".//m:oMathPara"): om.getparent().remove(om)
        def sup(par, txt): rr = par.add_run(txt); rr.font.superscript = True
        p.add_run("Isman Kurniawan"); sup(p, "[a],*"); p.add_run(", Angel Metanosa Afinda"); sup(p, "[b]"); p.add_run(", Aaz Muhammad Hafidz Azis"); sup(p, "[a]")
        aff = ["[a] School of Computing, Telkom University, Jl. Telekomunikasi No. 1, Bandung 40257, Indonesia",
               "[b] School of Electrical Engineering, Telkom University, Jl. Telekomunikasi No. 1, Bandung 40257, Indonesia",
               "* Corresponding author. E-mail: ismankrn@telkomuniversity.ac.id"]
        anchor = p
        for a in aff:
            np_ = copy.deepcopy(p._element); anchor._element.addnext(np_)
            from docx.text.paragraph import Paragraph
            q = Paragraph(np_, p._parent)
            for r in list(q.runs): r._element.getparent().remove(r._element)
            rr = q.add_run(a); rr.font.size = Pt(9); anchor = q
        break
# --- number the captions ("Table n." / "Figure n.") in order of appearance, label in bold
tn = fn = 0
for p in d.paragraphs:
    if p.style.name == "Table Caption":
        tn += 1; lab = f"Table {tn}. "
    elif p.style.name == "Image Caption":
        fn += 1; lab = f"Figure {fn}. "
    else:
        continue
    r = p.runs[0]._element if p.runs else None
    nr = p.add_run(lab); nr.bold = True
    if r is not None: r.addprevious(nr._element)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
# --- images placed inside table cells (multi-panel figure): limit each to one third of the text width
for t in d.tables:
    for row in t.rows:
        for cell in row.cells:
            for ext in cell._element.xpath(".//wp:extent"):
                cx = int(ext.get("cx")); cy = int(ext.get("cy")); maxw = Cm(5.6)
                if cx > maxw:
                    ext.set("cx", str(int(maxw))); ext.set("cy", str(int(cy * maxw / cx)))
            for ext in cell._element.xpath(".//a:ext"):
                cx = int(ext.get("cx")); cy = int(ext.get("cy")); maxw = Cm(5.6)
                if cx > maxw:
                    ext.set("cx", str(int(maxw))); ext.set("cy", str(int(cy * maxw / cx)))
# --- table rules: thin line above header, below header and below the last row; no vertical lines; table caption kept with table
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
def set_border(cell, edge, sz=4):
    tcPr = cell._element.get_or_add_tcPr(); b = tcPr.find(qn("w:tcBorders"))
    if b is None: b = OxmlElement("w:tcBorders"); tcPr.append(b)
    e = b.find(qn("w:" + edge))
    if e is None: e = OxmlElement("w:" + edge); b.append(e)
    e.set(qn("w:val"), "single"); e.set(qn("w:sz"), str(sz)); e.set(qn("w:color"), "000000"); e.set(qn("w:space"), "0")
for t in d.tables:
    # full text width, automatic column widths
    tblPr = t._element.tblPr
    for tag in ("w:tblW", "w:tblLayout"):
        for e in tblPr.findall(qn(tag)): tblPr.remove(e)
    w = OxmlElement("w:tblW"); w.set(qn("w:type"), "dxa"); w.set(qn("w:w"), "9070"); tblPr.append(w)
    ncol = max(len(r.cells) for r in t.rows)
    CH = 95; PAD = 260  # twips per character at 8 pt Arial, cell padding
    if ncol >= 10:  # very wide tables: 7 pt
        CH = 84; PAD = 200
        for row in t.rows:
            for cell in row.cells:
                for pp in cell.paragraphs:
                    for rr in pp.runs: rr.font.size = Pt(7)
    mins = [PAD + CH * 3] * ncol; des = [PAD + CH * 3] * ncol
    for r in t.rows:
        for i, c in enumerate(r.cells[:ncol]):
            words = c.text.split(); longest = max([len(w) for w in words] + [0])
            cap = 60 if i == 0 else 30
            mins[i] = max(mins[i], PAD + CH * min(longest, cap))
            des[i] = max(des[i], PAD + CH * min(len(c.text), cap))
    TOTAL = 9070  # twips, text width of an A4 page with 2.5 cm margins
    if sum(mins) >= TOTAL:
        dxa = [int(m * TOTAL / sum(mins)) for m in mins]
    else:
        extra = TOTAL - sum(mins); wts = [max(d - m, 0) for d, m in zip(des, mins)]
        dxa = [int(m + extra * (w / sum(wts) if sum(wts) else 1 / ncol)) for m, w in zip(mins, wts)]
    for e in tblPr.findall(qn("w:tblLayout")): tblPr.remove(e)
    lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); tblPr.append(lay)
    grid = t._element.find(qn("w:tblGrid"))
    if grid is not None:
        for gc in grid.findall(qn("w:gridCol")): grid.remove(gc)
        for wdt in dxa:
            gc = OxmlElement("w:gridCol"); gc.set(qn("w:w"), str(wdt)); grid.append(gc)
    for r in t.rows:
        for i, cell in enumerate(r.cells[:ncol]):
            tcPr = cell._element.get_or_add_tcPr()
            for e in tcPr.findall(qn("w:tcW")): tcPr.remove(e)
            cw = OxmlElement("w:tcW"); cw.set(qn("w:type"), "dxa"); cw.set(qn("w:w"), str(dxa[i])); tcPr.append(cw)
    # keep the header row with the first data row
    for r in t.rows[:2]:
        for c in r.cells:
            for pp in c.paragraphs: pp.paragraph_format.keep_with_next = True
    rows = t.rows
    if len(rows) < 2: continue
    for c in rows[0].cells: set_border(c, "top", 8); set_border(c, "bottom", 4)
    for c in rows[-1].cells: set_border(c, "bottom", 8)
for p in d.paragraphs:
    if p.style.name == "Table Caption": p.paragraph_format.keep_with_next = True
from ooxml_order import fix_document
print('schema-ordered elements:', fix_document(d))
d.save(HERE / "build/manuscript_MolInf.docx")
print(f"wrote build/manuscript_MolInf.docx with {tn} tables and {fn} figures numbered")
