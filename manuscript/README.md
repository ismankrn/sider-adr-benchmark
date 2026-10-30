# Manuscript scaffold

* `main.tex` — struktur IMRaD lengkap, teks draf Methods/Results sudah terisi dari hasil, semua bagian sudah bernarasi; placeholder yang tersisa hanya `\repourl` (URL repositori/DOI) di preamble `main.tex`.
* `make_tables.py` — membangkitkan semua tabel di `tables/` dan makro angka `tables/numbers.tex` langsung dari `../code/results/*.csv`. Jalankan ulang setiap kali hasil berubah: `make tables`.
* `references.bib` — referensi inti; entri bertanda VERIFY harus dicek ke sumber sebelum submit.
* Gambar diambil langsung dari `../code/figures/` lewat `\graphicspath`; versi final untuk publikasi taruh di `figures/` (prioritas lebih tinggi).
* Kompilasi: `make` (butuh TeX Live: pdflatex + bibtex + latexmk).

## Judul
Final (30 Okt 2026): "A controlled benchmark of SMILES-based recurrent networks, molecular fingerprints and pretrained chemical language models for adverse drug reaction prediction on SIDER" (bentuk frasa nominal formal). Alternatif yang tidak dipakai disimpan di bawah.
0. (sebelumnya) How much data do deep molecular models need? A controlled benchmark of fingerprints, SMILES recurrent networks, pretrained chemical language models and metaheuristic tuning for adverse drug reaction prediction
1. Representation beats tuning: a controlled benchmark of SMILES-based recurrent networks, molecular fingerprints, and metaheuristic hyper-parameter optimisers for adverse drug reaction prediction on SIDER
2. Do SMILES recurrent networks need metaheuristic tuning? A controlled benchmark for adverse drug reaction prediction
3. Input encoding, not hyper-parameter search, governs SMILES-LSTM performance in adverse drug reaction prediction: a 27-task SIDER benchmark

## Target jurnal (cek quartile terkini)
Journal of Cheminformatics; Molecular Informatics; Computers in Biology and Medicine; Journal of Molecular Graphics and Modelling; BMC Bioinformatics.
