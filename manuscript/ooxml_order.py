"""Reorder the children of OOXML property elements into the order required by the
ECMA-376 schema.  Microsoft Word refuses documents whose property children are out of
order, whereas LibreOffice accepts them, so every builder runs this before saving."""
from docx.oxml.ns import qn

ORDER = {
    "w:pPr": ["pStyle", "keepNext", "keepLines", "pageBreakBefore", "framePr", "widowControl", "numPr",
              "suppressLineNumbers", "pBdr", "shd", "tabs", "suppressAutoHyphens", "kinsoku", "wordWrap",
              "overflowPunct", "topLinePunct", "autoSpaceDE", "autoSpaceDN", "bidi", "adjustRightInd",
              "snapToGrid", "spacing", "ind", "contextualSpacing", "mirrorIndents", "suppressOverlap", "jc",
              "textDirection", "textAlignment", "textboxTightWrap", "outlineLvl", "divId", "cnfStyle", "rPr",
              "sectPr", "pPrChange"],
    "w:rPr": ["rStyle", "rFonts", "b", "bCs", "i", "iCs", "caps", "smallCaps", "strike", "dstrike", "outline",
              "shadow", "emboss", "imprint", "noProof", "snapToGrid", "vanish", "webHidden", "color", "spacing",
              "w", "kern", "position", "sz", "szCs", "highlight", "u", "effect", "bdr", "shd", "fitText",
              "vertAlign", "rtl", "cs", "em", "lang", "eastAsianLayout", "specVanish", "oMath"],
    "w:tblPr": ["tblStyle", "tblpPr", "tblOverlap", "bidiVisual", "tblStyleRowBandSize", "tblStyleColBandSize",
                "tblW", "jc", "tblCellSpacing", "tblInd", "tblBorders", "shd", "tblLayout", "tblCellMar", "tblLook",
                "tblCaption", "tblDescription"],
    "w:tcPr": ["cnfStyle", "tcW", "gridSpan", "hMerge", "vMerge", "tcBorders", "shd", "noWrap", "tcMar",
               "textDirection", "tcFitText", "vAlign", "hideMark"],
    "w:trPr": ["cnfStyle", "divId", "gridBefore", "gridAfter", "wBefore", "wAfter", "cantSplit", "trHeight",
               "tblHeader", "tblCellSpacing", "jc", "hidden"],
    "w:sectPr": ["headerReference", "footerReference", "footnotePr", "endnotePr", "type", "pgSz", "pgMar",
                 "paperSrc", "pgBorders", "lnNumType", "pgNumType", "cols", "formProt", "vAlign", "noEndnote",
                 "titlePg", "textDirection", "bidi", "rtlGutter", "docGrid", "printerSettings"],
}
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _fix(el, order):
    rank = {W + name: i for i, name in enumerate(order)}
    kids = list(el)
    # drop exact duplicates of single-valued properties (keep the last one written)
    seen = {}
    for k in kids:
        if k.tag in rank and k.tag not in (W + "headerReference", W + "footerReference"):
            if k.tag in seen:
                el.remove(seen[k.tag])
            seen[k.tag] = k
    kids = list(el)
    kids_sorted = sorted(kids, key=lambda k: (rank.get(k.tag, len(order)), kids.index(k)))
    if kids_sorted != kids:
        for k in kids: el.remove(k)
        for k in kids_sorted: el.append(k)


def fix_document(document):
    """Apply the schema order to the body and to every header and footer part."""
    roots = [document.element]
    # existing header/footer parts only (accessing section.header would create new ones)
    for rel in document.part.rels.values():
        if rel.reltype.endswith("/header") or rel.reltype.endswith("/footer"):
            roots.append(rel.target_part.element)
    n = 0
    for root in roots:
        for tag, order in ORDER.items():
            for el in root.iter(qn(tag)):
                _fix(el, order); n += 1
    return n
