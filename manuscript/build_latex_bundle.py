"""Self-contained LaTeX archive of the main manuscript for journal submission.

Collects main.tex, references.bib, the compiled main.bbl, every \\input table and every
figure that main.tex uses, rewrites \\graphicspath to the flat figures/ folder, and writes
build/Kurniawan_MolInf_LaTeX.zip.  Run after `make` so that main.bbl is current."""
import pathlib, re, shutil, tempfile, zipfile

HERE = pathlib.Path(__file__).resolve().parent
SEARCH = [HERE / "figures", HERE.parent / "code/figures", HERE.parent / "code/figures/attribution"]
EXT = [".pdf", ".png", ".jpg"]          # pdflatex order of preference
OUT = HERE / "build/Kurniawan_MolInf_LaTeX.zip"

tex = (HERE / "main.tex").read_text()
tex = re.sub(r"\\graphicspath\{.*\}", r"\\graphicspath{{figures/}}", tex)

files = {"main.tex": None, "references.bib": HERE / "references.bib", "main.bbl": HERE / "main.bbl"}
for name in sorted(set(re.findall(r"\\input\{(tables/[^}]+)\}", tex))):
    files[name + ".tex"] = HERE / (name + ".tex")
for name in sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex))):
    src = next((d / (name + e) for e in EXT for d in SEARCH if (d / (name + e)).exists()), None)
    if src is None:
        raise FileNotFoundError(name)
    files["figures/" + src.name] = src

missing = [k for k, v in files.items() if v is not None and not v.exists()]
if missing:
    raise FileNotFoundError(missing)

OUT.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("main.tex", tex)
    for arc, src in files.items():
        if src is not None:
            z.write(src, arc)
print(f"wrote {OUT} with {len(files)} files")
for arc in sorted(files): print("  ", arc)
