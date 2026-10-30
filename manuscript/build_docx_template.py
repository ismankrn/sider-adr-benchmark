"""Pour the pandoc-generated manuscript (build/manuscript_MolInf.docx) into the Wiley
Molecular Informatics Full Paper template (template/2022_full_paper.docx).

Run build_docx.py first.  Output: build/manuscript_MolInf_template.docx
"""
import copy, io, pathlib, re, sys
import docx
from docx.shared import Pt, Cm, Emu
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH

HERE = pathlib.Path(__file__).resolve().parent
S = docx.Document(HERE / "build/manuscript_MolInf.docx")      # source (pandoc output)
T = docx.Document(HERE / "template/2022_full_paper.docx")     # Wiley template
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
R_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"

TEXT_W = 10092      # twips, two-column text width (21 cm - 2 x 1.6 cm)
COL_W = 4876        # twips, one column
CM = 567            # twips per cm

TITLE = "A controlled benchmark of SMILES-based recurrent networks, molecular fingerprints and pretrained chemical language models for adverse drug reaction prediction on SIDER"
AUTHORS = [("Isman Kurniawan", "[a]", True), ("Angel Metanosa Afinda", "[b]", False), ("Aaz Muhammad Hafidz Azis", "[a]", False)]
ADDRESSES = [("[a]", "School of Computing, Telkom University", "Jl. Telekomunikasi No. 1, Bandung 40257, Indonesia"),
             ("[b]", "School of Electrical Engineering, Telkom University", "Jl. Telekomunikasi No. 1, Bandung 40257, Indonesia")]
EMAIL = "*e-mail: ismankrn@telkomuniversity.ac.id"
RUNNING_AUTHOR = "I. Kurniawan et al."

STYLE_MAP = {"Heading1": "H1", "Heading2": "H2", "Heading3": "H2", "FirstParagraph": "P1withoutIndendation",
             "BodyText": "P1withIndendation", "TableCaption": "TableCaption", "ImageCaption": "FigureCaption",
             "Bibliography": "References", "CaptionedFigure": "Normal", "Compact": "P1withIndendation"}
BACK_HEADINGS = {"Author Contributions", "Acknowledgements", "Conflict of Interest", "Data Availability Statement", "Funding"}

def ptext(el):
    return "".join(t.text or "" for t in el.iter(qn("w:t")))

def pstyle(el):
    st = el.find(".//" + qn("w:pStyle"))
    return st.get(qn("w:val")) if st is not None else ""

def set_pstyle(el, sid):
    pPr = el.find(qn("w:pPr"))
    if pPr is None:
        pPr = OxmlElement("w:pPr"); el.insert(0, pPr)
    st = pPr.find(qn("w:pStyle"))
    if st is None:
        st = OxmlElement("w:pStyle"); pPr.insert(0, st)
    st.set(qn("w:val"), sid)

TWO_COL_SECTPR = None   # set below from the template
def make_sect_break(cols):
    """Empty paragraph carrying a continuous section break with `cols` columns (copied from the template)."""
    p = OxmlElement("w:p"); pPr = OxmlElement("w:pPr"); p.append(pPr)
    sp = copy.deepcopy(TWO_COL_SECTPR)
    for e in sp.findall(qn("w:headerReference")) + sp.findall(qn("w:footerReference")): sp.remove(e)
    if cols == 1:
        for e in sp.findall(qn("w:cols")): sp.remove(e)
    pPr.append(sp)
    # make the break paragraph as small as possible
    rPr = OxmlElement("w:rPr"); sz = OxmlElement("w:sz"); sz.set(qn("w:val"), "2"); rPr.append(sz); pPr.insert(0, rPr)
    spc = OxmlElement("w:spacing"); spc.set(qn("w:before"), "0"); spc.set(qn("w:after"), "0"); spc.set(qn("w:line"), "20"); spc.set(qn("w:lineRule"), "exact"); pPr.insert(0, spc)
    return p

# ------------------------------------------------------------------ 1. collect source body (everything after the Keywords paragraph)
src = list(S.element.body.iterchildren())
start = next(i for i, el in enumerate(src) if el.tag == qn("w:p") and ptext(el).startswith("Keywords:")) + 1
abstract_text = next(ptext(el) for el in src if el.tag == qn("w:p") and pstyle(el) == "Abstract")
keywords_text = next(ptext(el) for el in src if el.tag == qn("w:p") and ptext(el).startswith("Keywords:"))[len("Keywords:"):].strip()
content = [el for el in src[start:] if el.tag in (qn("w:p"), qn("w:tbl"))]

# ------------------------------------------------------------------ 2. template: title, authors, address frame, abstract table, header
tb = T.element.body
tels = list(tb.iterchildren())
TWO_COL_SECTPR = next(sp for sp in tb.iter(qn("w:sectPr")) if sp.find(qn("w:cols")) is not None and sp.find(qn("w:cols")).get(qn("w:num")) == "2")
FIRST_SECTPR = next(sp for sp in tb.iter(qn("w:sectPr")) if sp.find(qn("w:headerReference")) is not None)
def tfind(style=None, contains=None):
    for el in tels:
        if el.tag != qn("w:p"): continue
        if style and pstyle(el) != style: continue
        if contains and contains not in ptext(el): continue
        return el
    raise KeyError((style, contains))

def clear_runs(p):
    for ch in list(p):
        if ch.tag != qn("w:pPr"): p.remove(ch)

def add_run(p, text, sup=False, bold=False, size=None):
    r = OxmlElement("w:r"); rPr = OxmlElement("w:rPr")
    if bold: rPr.append(OxmlElement("w:b"))
    if sup: va = OxmlElement("w:vertAlign"); va.set(qn("w:val"), "superscript"); rPr.append(va)
    if size: sz = OxmlElement("w:sz"); sz.set(qn("w:val"), str(int(size * 2))); rPr.append(sz)
    r.append(rPr); t = OxmlElement("w:t"); t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve"); t.text = text; r.append(t); p.append(r)
    return r

# title
p_title = tfind("Title"); clear_runs(p_title); add_run(p_title, TITLE)
# authors
p_auth = tfind("Authors"); clear_runs(p_auth)
for i, (name, mark, corr) in enumerate(AUTHORS):
    sep = "" if i == 0 else (" and " if i == len(AUTHORS) - 1 else " ")
    add_run(p_auth, sep + name + ("," if i < len(AUTHORS) - 1 else ""))
    add_run(p_auth, mark + ("*" if corr else ""), sup=True)
# dedication: remove
p_ded = tfind("Dedication"); tb.remove(p_ded)
# address frame lives in the template's "1 Introduction" paragraph
p_intro_t = tfind("H1", "Introduction")
frame_run = next(r for r in p_intro_t.findall(qn("w:r")) if r.find(".//" + qn("w:txbxContent")) is not None)
txbx = frame_run.find(".//" + qn("w:txbxContent"))
addr_ps = txbx.findall(qn("w:p"))
# paragraph 0 -> [a], paragraph 1 -> [b] + e-mail, paragraph 2 (Supporting Information note) -> removed
for p, (mark, inst, addr) in zip(addr_ps[:2], ADDRESSES):
    clear_runs(p)
    r = OxmlElement("w:r"); r.append(OxmlElement("w:rPr"))
    def t_(txt):
        t = OxmlElement("w:t"); t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve"); t.text = txt; r.append(t)
    t_(mark); r.append(OxmlElement("w:tab")); t_(inst); r.append(OxmlElement("w:br")); t_(addr)
    if mark == "[a]":
        r.append(OxmlElement("w:br")); t_(EMAIL)
    p.append(r)
for p in addr_ps[2:]: txbx.remove(p)
frame_copy = copy.deepcopy(frame_run)
# abstract table
tbl_abs = next(el for el in tels if el.tag == qn("w:tbl") and "Abstract" in ptext(el))
cells = tbl_abs.findall(".//" + qn("w:tc"))
sents = re.split(r"(?<=\.)\s+", abstract_text.strip())
half = len(abstract_text) / 2; acc = 0; k = 0
for k, s_ in enumerate(sents):
    acc += len(s_) + 1
    if acc >= half: break
left, right = " ".join(sents[:k + 1]), " ".join(sents[k + 1:])
pL = cells[0].find(qn("w:p")); clear_runs(pL); add_run(pL, "Abstract:", bold=True); add_run(pL, " " + left)
pR = cells[1].find(qn("w:p")); clear_runs(pR); add_run(pR, right)
pK = cells[2].find(qn("w:p")); clear_runs(pK); add_run(pK, "Keywords:", bold=True); add_run(pK, " " + keywords_text.replace(";", ","))
# running header
for sec in T.sections:
    for hp in sec.header.paragraphs:
        for r in hp.runs:
            if "C. Author" in r.text: r.text = r.text.replace("C. Author et al.", RUNNING_AUTHOR)

# ------------------------------------------------------------------ 3. remove the template's dummy body (from "1 Introduction" up to History)
p_hist = tfind("History")
i0 = tels.index(p_intro_t); i1 = tels.index(p_hist)
for el in tels[i0:i1]: tb.remove(el)
insert_before = p_hist
# everything after History goes, and the final section becomes the two-column one with the logo header and page-number footer
for el in tels[i1 + 1:]:
    if el.tag != qn("w:sectPr"): tb.remove(el)
final = tb.find(qn("w:sectPr")); new_final = copy.deepcopy(TWO_COL_SECTPR)
for e in reversed(FIRST_SECTPR.findall(qn("w:headerReference")) + FIRST_SECTPR.findall(qn("w:footerReference"))): new_final.insert(0, copy.deepcopy(e))
tb.replace(final, new_final)

# ------------------------------------------------------------------ 4. helpers for copying content
def copy_images(el):
    """Re-embed every picture of `el` (copied from S) into T."""
    for blip in el.iter("{http://schemas.openxmlformats.org/drawingml/2006/main}blip"):
        rid = blip.get(R_NS + "embed")
        if rid is None or rid not in S.part.related_parts: continue
        blob = S.part.related_parts[rid].blob
        new_rid, _ = T.part.get_or_add_image(io.BytesIO(blob))
        blip.set(R_NS + "embed", new_rid)

def set_image_width(el, max_w_twips):
    max_emu = int(max_w_twips * 635)
    for ext in el.iter("{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}extent"):
        cx, cy = int(ext.get("cx")), int(ext.get("cy"))
        if cx > max_emu:
            ext.set("cx", str(max_emu)); ext.set("cy", str(int(cy * max_emu / cx)))
    for ext in el.iter("{http://schemas.openxmlformats.org/drawingml/2006/main}ext"):
        cx, cy = int(ext.get("cx")), int(ext.get("cy"))
        if cx > max_emu:
            ext.set("cx", str(max_emu)); ext.set("cy", str(int(cy * max_emu / cx)))

def unwrap_hyperlinks(el):
    for h in list(el.iter(qn("w:hyperlink"))):
        parent = h.getparent(); idx = parent.index(h)
        for ch in list(h): parent.insert(idx, ch); idx += 1
        parent.remove(h)
    for rs in list(el.iter(qn("w:rStyle"))):
        val = rs.get(qn("w:val"))
        rPr = rs.getparent()
        if val == "Hyperlink":
            rPr.remove(rs)
        elif val == "VerbatimChar":
            rPr.remove(rs)
            f = OxmlElement("w:rFonts")
            for a in ("w:ascii", "w:hAnsi", "w:cs"): f.set(qn(a), "Courier New")
            rPr.insert(0, f)
        elif val == "SectionNumber":
            rPr.remove(rs)
    for tab in list(el.iter(qn("w:tab"))):
        r = tab.getparent()
        if pstyle(el) in ("Heading1", "Heading2", "H1", "H2", "Bibliography", "References"):
            t = OxmlElement("w:t"); t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve"); t.text = " " if pstyle(el) in ("Heading1", "Heading2", "H1", "H2") else "\t"
            if t.text == "\t": continue
            r.replace(tab, t)

def table_min_width(tbl, ch=88, pad=330):
    rows = tbl.findall(qn("w:tr"))
    ncol = max(len(r.findall(qn("w:tc"))) for r in rows)
    mins = [pad + ch * 3] * ncol
    for r in rows:
        for i, c in enumerate(r.findall(qn("w:tc"))[:ncol]):
            words = ptext(c).split(); longest = max([len(w) for w in words] + [0])
            mins[i] = max(mins[i], pad + ch * min(longest, 60 if i == 0 else 30))
    return ncol, mins

def set_table_width(tbl, total, ch=88, pad=330):
    rows = tbl.findall(qn("w:tr")); ncol, mins = table_min_width(tbl, ch, pad)
    des = [pad + ch * 3] * ncol
    for r in rows:
        for i, c in enumerate(r.findall(qn("w:tc"))[:ncol]):
            des[i] = max(des[i], pad + ch * min(len(ptext(c)), 60 if i == 0 else 30))
    if sum(mins) >= total:
        dxa = [int(m * total / sum(mins)) for m in mins]
    else:
        extra = total - sum(mins); wts = [max(d - m, 0) for d, m in zip(des, mins)]
        dxa = [int(m + extra * (w / sum(wts) if sum(wts) else 1 / ncol)) for m, w in zip(mins, wts)]
    tblPr = tbl.find(qn("w:tblPr"))
    for tag in ("w:tblW", "w:tblLayout", "w:tblInd", "w:jc"):
        for e in tblPr.findall(qn(tag)): tblPr.remove(e)
    w = OxmlElement("w:tblW"); w.set(qn("w:type"), "dxa"); w.set(qn("w:w"), str(total)); tblPr.append(w)
    lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); tblPr.append(lay)
    grid = tbl.find(qn("w:tblGrid"))
    if grid is not None:
        for gc in grid.findall(qn("w:gridCol")): grid.remove(gc)
        for wd in dxa:
            gc = OxmlElement("w:gridCol"); gc.set(qn("w:w"), str(wd)); grid.append(gc)
    for r in rows:
        tcs = r.findall(qn("w:tc"))
        for i, c in enumerate(tcs[:ncol]):
            tcPr = c.find(qn("w:tcPr"))
            if tcPr is None: tcPr = OxmlElement("w:tcPr"); c.insert(0, tcPr)
            span = tcPr.find(qn("w:gridSpan")); n = int(span.get(qn("w:val"))) if span is not None else 1
            for e in tcPr.findall(qn("w:tcW")): tcPr.remove(e)
            cw = OxmlElement("w:tcW"); cw.set(qn("w:type"), "dxa"); cw.set(qn("w:w"), str(sum(dxa[i:i + n]))); tcPr.insert(0, cw)

def style_table(tbl, size):
    rows = tbl.findall(qn("w:tr"))
    for ri, r in enumerate(rows):
        for c in r.findall(qn("w:tc")):
            for p in c.findall(qn("w:p")):
                set_pstyle(p, "TableHead" if ri == 0 else "TableBody")
                pPr = p.find(qn("w:pPr"))
                for e in pPr.findall(qn("w:spacing")): pPr.remove(e)
                spc = OxmlElement("w:spacing"); spc.set(qn("w:before"), "10"); spc.set(qn("w:after"), "10"); pPr.append(spc)
                for rr in p.findall(qn("w:r")):
                    rPr = rr.find(qn("w:rPr"))
                    if rPr is None: rPr = OxmlElement("w:rPr"); rr.insert(0, rPr)
                    for e in rPr.findall(qn("w:sz")) + rPr.findall(qn("w:szCs")): rPr.remove(e)
                    sz = OxmlElement("w:sz"); sz.set(qn("w:val"), str(int(size * 2))); rPr.append(sz)

# ------------------------------------------------------------------ 4b. display equations rendered with pdflatex
import subprocess
from PIL import Image
EQ_DIR = HERE / "build/eq"; EQ_DIR.mkdir(parents=True, exist_ok=True)
tex_src = (HERE / "main.tex").read_text()
equations = re.findall(r"\\begin\{equation\}(.*?)\\end\{equation\}", tex_src, re.S)
EQ_IMAGES = []
for k, body in enumerate(equations):
    body = re.sub(r"\\label\{[^}]*\}", "", body).strip()
    texf = EQ_DIR / f"eq{k}.tex"
    texf.write_text("\\documentclass[preview,border=1pt]{standalone}\n\\usepackage{amsmath,amssymb}\n\\begin{document}\n$\\displaystyle " + body + "$\n\\end{document}\n")
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-output-directory", str(EQ_DIR), str(texf)], capture_output=True, check=True)
    subprocess.run(["pdftoppm", "-r", "600", "-png", "-singlefile", str(EQ_DIR / f"eq{k}.pdf"), str(EQ_DIR / f"eq{k}")], check=True)
    EQ_IMAGES.append(EQ_DIR / f"eq{k}.png")
eq_index = 0
def equation_paragraph(width_twips):
    """Paragraph holding the next display equation as a picture, scaled to at most width_twips."""
    global eq_index
    png = EQ_IMAGES[eq_index]; eq_index += 1
    w_px, h_px = Image.open(png).size
    nat_cm = w_px / 600 * 2.54 * 0.9          # 10 pt LaTeX scaled towards the 9 pt body text
    w_cm = min(nat_cm, width_twips / CM - 0.2)
    inline = T.part.new_pic_inline(str(png), Cm(w_cm), Cm(w_cm * h_px / w_px))
    pnew = OxmlElement("w:p"); pPr = OxmlElement("w:pPr"); pnew.append(pPr)
    jc = OxmlElement("w:jc"); jc.set(qn("w:val"), "center"); pPr.append(jc)
    spc = OxmlElement("w:spacing"); spc.set(qn("w:before"), "80"); spc.set(qn("w:after"), "80"); pPr.append(spc)
    r = OxmlElement("w:r"); drawing = OxmlElement("w:drawing"); drawing.append(inline); r.append(drawing); pnew.append(r)
    return pnew

# ------------------------------------------------------------------ 5. copy the content, grouping captions with their table/figure
blocks = []   # (kind, [elements]) kind in {"text", "table", "figure"}
i = 0
while i < len(content):
    el = content[i]
    if el.tag == qn("w:tbl"):
        grp = []
        if blocks and blocks[-1][0] == "text" and pstyle(blocks[-1][1][-1]) == "TableCaption":
            grp.append(blocks[-1][1].pop())
            if not blocks[-1][1]: blocks.pop()
        grp.append(el); blocks.append(("table", grp)); i += 1; continue
    if el.tag == qn("w:p") and pstyle(el) == "CaptionedFigure":
        grp = [el]
        if i + 1 < len(content) and pstyle(content[i + 1]) == "ImageCaption":
            grp.append(content[i + 1]); i += 1
        blocks.append(("figure", grp)); i += 1; continue
    if blocks and blocks[-1][0] == "text": blocks[-1][1].append(el)
    else: blocks.append(("text", [el]))
    i += 1

in_full = False
def switch(full):
    """Insert a section break so that what follows is single-column (full=True) or two-column."""
    global in_full
    if full == in_full: return
    tb.insert(tb.index(insert_before), make_sect_break(2 if full else 1))
    in_full = full

seen_refs = False
for kind, els in blocks:
    if kind == "table":
        tbl = els[-1]; ncol, mins = table_min_width(tbl)
        full = sum(mins) > COL_W or ncol > 5
        switch(full)
        set_table_width(tbl, TEXT_W if full else COL_W)
        style_table(tbl, 7 if (full and ncol >= 10) else 7.5)
    elif kind == "figure":
        switch(True)
        for el in els: copy_images(el); set_image_width(el, TEXT_W)
    else:
        switch(False)
    for el in els:
        new = copy.deepcopy(el) if el.getparent() is not None else el
        if new.tag == qn("w:p") and new.find(".//{http://schemas.openxmlformats.org/officeDocument/2006/math}oMathPara") is not None and eq_index < len(EQ_IMAGES):
            tb.insert(tb.index(insert_before), equation_paragraph(TEXT_W if in_full else COL_W)); continue
        if new.tag == qn("w:p"):
            sid = pstyle(new)
            txt = ptext(new)
            if sid == "Heading1" and txt.strip() in BACK_HEADINGS:
                set_pstyle(new, "HAcknowledgements")
            elif sid in STYLE_MAP:
                set_pstyle(new, STYLE_MAP[sid])
            if sid == "Bibliography":
                if not seen_refs:
                    h = OxmlElement("w:p"); set_pstyle(h, "H1"); add_run(h, "References"); tb.insert(tb.index(insert_before), h); seen_refs = True
                # "12.<tab>Author ..." -> "[12]<tab>Author ..."
                first_t = next((t for t in new.iter(qn("w:t")) if t.text and re.match(r"^\d+\.$", t.text.strip())), None)
                if first_t is not None: first_t.text = "[" + first_t.text.strip()[:-1] + "]"
            numPr = new.find(".//" + qn("w:numPr"))
            if numPr is not None:
                list_counter = globals().get("list_counter", 0) + 1; globals()["list_counter"] = list_counter
                numPr.getparent().remove(numPr); set_pstyle(new, "P1withoutIndendation")
                pPr = new.find(qn("w:pPr")); ind = OxmlElement("w:ind"); ind.set(qn("w:left"), "284"); ind.set(qn("w:hanging"), "284"); pPr.append(ind)
                r0 = OxmlElement("w:r"); r0.append(OxmlElement("w:rPr")); t0 = OxmlElement("w:t"); t0.set("{http://www.w3.org/XML/1998/namespace}space", "preserve"); t0.text = f"{list_counter}.\t"; r0.append(t0)
                pPr.addnext(r0)
            if pstyle(new) in ("H1", "H2", "HAcknowledgements"):
                pPr = new.find(qn("w:pPr")); pPr.append(OxmlElement("w:keepNext"))
            if sid == "CaptionedFigure":
                pPr = new.find(qn("w:pPr")); jc = OxmlElement("w:jc"); jc.set(qn("w:val"), "center"); pPr.append(jc)
            unwrap_hyperlinks(new)
        tb.insert(tb.index(insert_before), new)
switch(False)

# back-matter text paragraphs after HAcknowledgements headings -> Acknowledgements style
prev = ""
for el in tb.iterchildren():
    if el.tag != qn("w:p"): continue
    sid = pstyle(el)
    if sid in ("P1withoutIndendation", "P1withIndendation") and prev in ("HAcknowledgements", "Acknowledgements"):
        set_pstyle(el, "P1withoutIndendation")
        sid = "Acknowledgements"
    prev = sid

# address frame goes into the first heading ("1 Introduction"), as in the template
first_h1 = next(el for el in tb.iterchildren() if el.tag == qn("w:p") and pstyle(el) == "H1")
first_h1.append(frame_copy)
for t in tb.iter(qn("w:t")):
    if t.text and "underhttps://" in t.text: t.text = t.text.replace("underhttps://", "under https://")
assert eq_index == len(EQ_IMAGES), (eq_index, len(EQ_IMAGES))
out = HERE / "build/manuscript_MolInf_template.docx"
T.save(out)
print("wrote", out)
