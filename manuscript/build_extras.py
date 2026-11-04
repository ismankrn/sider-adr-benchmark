"""Cover letter and Table-of-Contents entry for the Molecular Informatics submission.
Output: build/cover_letter_MolInf.docx, build/TOC_entry_MolInf.docx"""
import datetime, pathlib
import docx
from docx.shared import Cm, Pt

HERE = pathlib.Path(__file__).resolve().parent
TITLE = ("A controlled benchmark of SMILES-based recurrent networks, molecular fingerprints and pretrained "
         "chemical language models for adverse drug reaction prediction on SIDER")
AUTHORS = "Isman Kurniawan*, Angel Metanosa Afinda, Aaz Muhammad Hafidz Azis, Aditya Wibawa Sakti"
CO_AUTHORS = "Angel Metanosa Afinda, Aaz Muhammad Hafidz Azis and Aditya Wibawa Sakti"


def new_doc():
    d = docx.Document()
    d.styles["Normal"].font.name = "Arial"
    d.styles["Normal"].font.size = Pt(10)
    return d


# ------------------------------------------------------------------ cover letter
from docx.enum.text import WD_ALIGN_PARAGRAPH as AL

d = new_doc()
sec = d.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
sec.left_margin = sec.right_margin = Cm(2.5); sec.top_margin = sec.bottom_margin = Cm(2.0)
d.styles["Normal"].font.size = Pt(10)
d.styles["Normal"].paragraph_format.space_after = Pt(8)
d.styles["Normal"].paragraph_format.line_spacing = 1.1

def P(text="", align=AL.JUSTIFY, bold=False, after=None, size=None):
    p = d.add_paragraph(); p.alignment = align
    if text:
        r = p.add_run(text); r.bold = bold
        if size: r.font.size = Pt(size)
    if after is not None: p.paragraph_format.space_after = Pt(after)
    return p

# sender block (from the signature of the letter), right-aligned, with a rule underneath
P("Isman Kurniawan", AL.RIGHT, bold=True, after=0)
for line in ("School of Computing, Telkom University", "Jl. Telekomunikasi No. 1, Bandung 40257, Indonesia",
             "ismankrn@telkomuniversity.ac.id"):
    P(line, AL.RIGHT, after=0, size=9.5)
rule = P(after=14)
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
pPr = rule._element.get_or_add_pPr(); bdr = OxmlElement("w:pBdr"); bot = OxmlElement("w:bottom")
for k, v in (("w:val", "single"), ("w:sz", "6"), ("w:space", "1"), ("w:color", "000000")): bot.set(qn(k), v)
bdr.append(bot); pPr.append(bdr)

P("Bandung, " + datetime.date.today().strftime("%d %B %Y").lstrip("0"), AL.LEFT, after=14)
P("Editors", AL.LEFT, after=0)
P("Molecular Informatics", AL.LEFT, after=14)
P("Dear Editors of Molecular Informatics,", AL.LEFT)
P(f"Please find enclosed our manuscript “{TITLE}”, which we submit for consideration as a Full Paper.")
P("Recurrent networks trained on SMILES strings, often tuned with "
  "nature-inspired metaheuristics, are widely applied to adverse drug reaction prediction on SIDER, yet the design "
  "choices behind them are rarely isolated and the evaluation practice in this literature frequently rests on random "
  "splits and threshold-dependent metrics. Our study varies one factor at a time under a single protocol, namely three "
  "input encodings, 15 recurrent architectures, five hyper-parameter optimisers with an identical budget and three "
  "search seeds, classical learners on fingerprints and descriptors, two fine-tuned chemical language models and a "
  "graph neural network, evaluated with stratified and scaffold cross-validation over three seeds and paired tests with "
  "Holm correction, on all 27 SIDER organ classes and on a Tox21 task.")
P("The encoding of the SMILES tokens is the dominant design choice, since integer-index inputs reduce "
  "the network to chance level whereas a learned embedding restores learning, while the architecture and the optimiser "
  "change the AUROC by less than the fold-to-fold variation. Only particle swarm optimisation improves on random search "
  "consistently on both splits. Fingerprint ensembles outperform every deep model on the primary task and on 26 of the "
  "27 organ classes, and paired learning curves locate the training-set size at which deep models catch up at about "
  "four to five thousand molecules. We also show that the F1 score and accuracy at a fixed threshold rank a trivial "
  "majority-class predictor above every learned model on this task.")
P("The novelty of the study is stated in the Abstract and Introduction. The "
  "models are compared against published MoleculeNet results on SIDER and validated on a second dataset (Tox21) and on "
  "scaffold splits that hold out unseen chemotypes. All code, standardised data and per-fold results are publicly "
  "available on GitHub (https://github.com/ismankrn/sider-adr-benchmark) and archived on Zenodo "
  "(DOI 10.5281/zenodo.23798100).")
P("The manuscript has not been published previously and is not under consideration elsewhere. All authors have "
  "approved the submission and declare no competing interests. The work was supported by a research grant from "
  "Telkom University, Indonesia.")
P("Suggested referees can be provided on request. We thank you for considering our manuscript.")
P("Yours sincerely,", AL.LEFT, after=30)
P("Isman Kurniawan (corresponding author)", AL.LEFT, bold=True, after=0)
for line in ("School of Computing, Telkom University, Jl. Telekomunikasi No. 1, Bandung 40257, Indonesia",
             "ismankrn@telkomuniversity.ac.id"):
    P(line, AL.LEFT, after=0)
P(f"on behalf of {CO_AUTHORS}", AL.LEFT, after=0).paragraph_format.space_before = Pt(8)
d.save(HERE / "build/cover_letter_MolInf.docx")

# ------------------------------------------------------------------ table-of-contents entry
d = new_doc()
d.add_paragraph().add_run("Table of Contents entry").bold = True
d.add_paragraph().add_run(TITLE).bold = True
d.add_paragraph(AUTHORS)
d.add_picture(str(HERE / "figures/toc_graphic.png"), width=Cm(5.0))
d.add_paragraph("On about one thousand SIDER molecules, how a SMILES string is encoded matters more than the recurrent "
                "architecture or the metaheuristic used to tune it, and fingerprint ensembles outperform recurrent "
                "networks, fine-tuned chemical language models and a graph network. Paired learning curves on Tox21 "
                "show that deep models catch up only once four to five thousand training molecules are available.")
d.save(HERE / "build/TOC_entry_MolInf.docx")
print("wrote cover letter and TOC entry")
