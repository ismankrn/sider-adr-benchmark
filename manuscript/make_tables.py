"""Generate every LaTeX table in manuscript/tables/ directly from code/results/*.csv.
Run:  ../code/.venv/bin/python make_tables.py   (from the manuscript/ folder)
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "code" / "results"
OUT = HERE / "tables"; OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE.parent / "code" / "src"))
from adr.stats import paired_test, holm  # noqa: E402

def ms(x, nd=3):
    return f"{x.mean():.{nd}f} $\\pm$ {x.std(ddof=1):.{nd}f}" if len(x) > 1 else f"{x.mean():.{nd}f}"

fmtp = lambda v: "$<$0.001" if v < 0.001 else f"{v:.3f}"


def bold_best_rows(entries, cols, nd=3):
    """entries: ('__group__', label) or (label, {col: Series}). Highest mean per column in bold, missing -> '--'."""
    best = {}
    for e in entries:
        if e[0] == "__group__": continue
        for c in cols:
            v = e[1].get(c)
            if v is not None and len(v) > 0:
                best[c] = max(best.get(c, -1e9), round(float(v.mean()), nd))
    out = []
    for e in entries:
        if e[0] == "__group__":
            out.append(r"\multicolumn{%d}{l}{\emph{%s}} \\" % (len(cols) + 1, e[1])); continue
        cells = []
        for c in cols:
            v = e[1].get(c)
            if v is None or len(v) == 0:
                cells.append("--"); continue
            txt = ms(v, nd)
            if round(float(v.mean()), nd) == best[c]:
                txt = r"\textbf{" + txt + "}"
            cells.append(txt)
        out.append(f"{e[0]} & " + " & ".join(cells) + r" \\")
    return out

def esc(s):
    return str(s).replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")

def write(name, body):
    (OUT / name).write_text(body); print("wrote", OUT / name)

# ------------------------------------------------------------------ load
rep = json.load(open(RES / "data_report.json"))
bl = pd.read_csv(RES / "baselines.csv"); bl["name"] = bl["model"] + "[" + bl["features"].astype(str) + "]"
dl = pd.read_csv(RES / "dl_untuned.csv"); dl["name"] = "DL:" + dl["model"]
hf = pd.read_csv(RES / "hpo_final.csv"); hf["name"] = "DL-tuned:" + hf["method"]
hf_c = hf[hf.split == "scaffold"]; hf = hf[hf.split == "stratified"]
allr_tuned_all = pd.concat([hf, hf_c])
ht = pd.read_csv(RES / "hpo_trace.csv")
at = pd.read_csv(RES / "all_tasks.csv")
pr = pd.read_csv(RES / "pretrained_primary.csv"); pr["name"] = "pre:" + pr["model"]
pt = pd.read_csv(RES / "pretrained_tasks.csv")
at = pd.concat([at, pt[[c for c in pt.columns if c in at.columns]]], ignore_index=True)
allr = pd.concat([bl, dl, hf, hf_c, pr], ignore_index=True)
S = allr[allr.split == "stratified"]; C = allr[allr.split == "scaffold"]
task = bl["task"].iloc[0]
pos = rep["positives_per_task"]

# ------------------------------------------------------------------ Table: dataset (SIDER + Tox21)
rep_t = json.load(open(RES / "data_report_tox21.json"))
df_clean = pd.read_csv(HERE.parent / "code" / "data" / "processed" / "sider_clean.csv")
tx = pd.read_csv(HERE.parent / "code" / "data" / "processed" / "tox21_clean.csv")
n1 = int(df_clean[task].sum()); n0 = len(df_clean) - n1
lab = tx["NR-AhR"].dropna(); t1 = int(lab.sum()); t0 = len(lab) - t1
body = r"""\begin{tabular}{lrr}
\toprule
 & SIDER & Tox21 \\
\midrule
Raw entries (MoleculeNet release) & %d & %d \\
Valid after RDKit parsing & %d & %d \\
Unique molecules after standardisation and merging & %d & %d \\
Removed as macromolecules ($>$ %d tokens) & %d & %d \\
Final number of molecules & %d & %d \\
Number of binary tasks & %d & %d \\
\midrule
Task used for the primary / learning-curve analyses & Blood and lymphatic & NR-AhR \\
\quad Labelled molecules & %d & %d \\
\quad Positive / negative & %d / %d & %d / %d \\
\quad Positive rate & %.3f & %.3f \\
\quad F1 of the all-positive classifier & %.3f & %.3f \\
\bottomrule
\end{tabular}""" % (rep["n_raw"], rep_t["n_raw"], rep["n_valid"], rep_t["n_valid"], rep["n_after_dedup"], rep_t["n_after_dedup"],
                      rep["max_tokens"], rep["n_removed_too_long"], rep_t["n_removed_too_long"], rep["n_final"], rep_t["n_final"],
                      len(rep["tasks"]), len(rep_t["tasks"]), len(df_clean), len(lab), n1, n0, t1, t0, n1 / len(df_clean), t1 / len(lab),
                      2 * n1 / (len(df_clean) + n1), 2 * t1 / (len(lab) + t1))
write("tab_dataset.tex", body)

# ------------------------------------------------------------------ Table: encoding ablation (CL1, default hyper-parameters)
MCOLS = ["auroc_s", "auroc_c", "auprc", "mcc", "balanced_acc", "f1"]
HEAD = r"& AUROC (strat.) & AUROC (scaffold) & AUPRC & MCC & Bal.\ acc. & F1 \\"
def _entry(label, key):
    s_, c_ = S[S.name == key], C[C.name == key]
    return (label, {"auroc_s": s_.auroc, "auroc_c": c_.auroc, "auprc": s_.auprc, "mcc": s_.mcc, "balanced_acc": s_.balanced_acc, "f1": s_.f1})
enc_rows = [("DL:CL1-index", "Integer index"), ("DL:CL1-onehot", "One-hot"), ("DL:CL1-embedding", "Learned embedding")]
rows = [r"\begin{tabular}{lccccccc}", r"\toprule", "Encoding " + HEAD.replace(r"& F1 \\", r"& F1 & Epochs \\"), r"\midrule"]
_un = pd.read_csv(RES / "dl_untuned.csv"); _un = _un[(_un.split == "stratified") & (_un.scheme == "CL") & (_un.n_layers == 1)]
_ep = {"DL:CL1-index": _un[_un.input_mode == "index"].epochs, "DL:CL1-onehot": _un[_un.input_mode == "onehot"].epochs, "DL:CL1-embedding": _un[_un.input_mode == "embedding"].epochs}
_body = bold_best_rows([_entry(lab, key) for key, lab in enc_rows], MCOLS)
rows += [r[:-2].rstrip() + f" & {ms(_ep[key], 1)} \\\\" for r, (key, _) in zip(_body, enc_rows)] + [r"\bottomrule", r"\end{tabular}"]
write("tab_encoding.tex", "\n".join(rows))

# ------------------------------------------------------------------ Table: paired tests between encodings (both splits, one Holm family)
enc_tests = [(r"Learned embedding vs.\ integer index", "DL:CL1-embedding", "DL:CL1-index"),
             (r"One-hot vs.\ integer index", "DL:CL1-onehot", "DL:CL1-index"),
             (r"One-hot vs.\ learned embedding", "DL:CL1-onehot", "DL:CL1-embedding")]
res = []
for split_name, D_ in [("Stratified", S), ("Scaffold", C)]:
    for label, a, b in enc_tests:
        t = paired_test(D_, "name", a, b); res.append((split_name, label, t))
padj = holm([t["wilcoxon_p"] for _, _, t in res])
rows = [r"\begin{tabular}{llccccc}", r"\toprule", r"Split & Comparison (A vs.\ B) & AUROC A & AUROC B & $\Delta$ & Wilcoxon $p$ / Holm $p$ & Cohen's $d$ \\", r"\midrule"]
prev = None
for (sp, label, t), pa in zip(res, padj):
    if prev and sp != prev: rows.append(r"\midrule")
    rows.append(f"{sp if sp != prev else ''} & {label} & {t['mean_a']:.3f} & {t['mean_b']:.3f} & {t['diff']:+.3f} & {fmtp(t['wilcoxon_p'])} / {fmtp(pa)} & {t['cohen_d']:.2f} \\\\"); prev = sp
rows += [r"\bottomrule", r"\end{tabular}"]
write("tab_encstats.tex", "\n".join(rows))

# ------------------------------------------------------------------ Table: main benchmark (overview with the three best default and tuned recurrent models)
emb_configs = ["L1", "L2", "L3", "CL1", "CL2", "CL3", "L1D", "L2D", "L3D", "CL1D", "CL2D", "CL3D", "BiLSTM1", "GRU1", "CNN-GRU"]
best_default = S[S.name.isin([f"DL:{a}" for a in emb_configs])].groupby("name").auroc.mean().sort_values(ascending=False).index[:3].tolist()
best_tuned = S[S.name.str.startswith("DL-tuned:")].groupby("name").auroc.mean().sort_values(ascending=False).index[:3].tolist()
groups = [
    ("Trivial baselines", ["majority[-]", "random[-]"]),
    ("Fingerprint / descriptor models", ["logreg[ecfp4]", "svm[ecfp4]", "rf[ecfp4]", "xgb[ecfp4]", "rf[descriptors]", "xgb[descriptors]"]),
    ("Recurrent networks, default hyper-parameters (three best)", best_default),
    ("Recurrent networks, tuned (three best optimisers)", best_tuned),
    ("Pretrained chemical language models and GNN", ["pre:chemberta", "pre:molformer", "pre:chemprop"]),
]
pretty = {"majority[-]": "Majority class", "random[-]": "Random scores", "logreg[ecfp4]": "Logistic regression, ECFP4",
          "svm[ecfp4]": "SVM (RBF), ECFP4", "rf[ecfp4]": "Random forest, ECFP4", "xgb[ecfp4]": "XGBoost, ECFP4",
          "rf[descriptors]": "Random forest, RDKit descriptors", "xgb[descriptors]": "XGBoost, RDKit descriptors",
          "DL:CL1-index": "Integer index (original pipeline)", "DL:CL1-onehot": "One-hot", "DL:CL1-embedding": "Learned embedding",
          "DL:L1": "L1", "DL:L2": "L2", "DL:L3": "L3", "DL:CL1": "CL1", "DL:CL2": "CL2", "DL:CL3": "CL3", "DL:L1D": "L1D", "DL:L2D": "L2D", "DL:L3D": "L3D",
          "DL:CL1D": "CL1D", "DL:CL2D": "CL2D", "DL:CL3D": "CL3D", "DL:BiLSTM1": "BiLSTM", "DL:GRU1": "GRU", "DL:CNN-GRU": "CNN-GRU (SMILES2Vec replica)",
          "pre:chemberta": "ChemBERTa-2 (77M-MLM), fine-tuned", "pre:molformer": "MoLFormer-XL, fine-tuned", "pre:chemprop": "Chemprop D-MPNN",
          "DL-tuned:random": "CL tuned, random search", "DL-tuned:tpe": "CL tuned, TPE", "DL-tuned:ga": "CL tuned, GA", "DL-tuned:mbo": "CL tuned, MBO", "DL-tuned:pso": "CL tuned, PSO"}
entries = []
for g, names in groups:
    entries.append(("__group__", g))
    for n in names:
        if len(S[S.name == n]) == 0: continue
        entries.append(_entry(r"\quad " + pretty.get(n, esc(n)), n))
rows = [r"\begin{tabular}{lcccccc}", r"\toprule", "Model " + HEAD, r"\midrule"] + bold_best_rows(entries, MCOLS) + [r"\bottomrule", r"\end{tabular}"]
write("tab_main.tex", "\n".join(rows))

# ------------------------------------------------------------------ Table: 12 architectures + 3 variants
arch = ["L1", "L2", "L3", "CL1", "CL2", "CL3", "L1D", "L2D", "L3D", "CL1D", "CL2D", "CL3D"]
variants = [("BiLSTM1", "BiLSTM"), ("GRU1", "GRU"), ("CNN-GRU", "CNN-GRU")]
ACOLS = ["auroc_s", "mcc_s", "auroc_c", "mcc_c"]
def _aentry(label, key):
    s_, c_ = S[S.name == key], C[C.name == key]
    return (label, {"auroc_s": s_.auroc, "mcc_s": s_.mcc, "auroc_c": c_.auroc, "mcc_c": c_.mcc})
entries = [_aentry(a, f"DL:{a}") for a in arch] + [("__group__", "Variants")] + [_aentry(lab, f"DL:{key}") for key, lab in variants]
rows = [r"\begin{tabular}{lcccc}", r"\toprule", r"Configuration & AUROC (strat.) & MCC (strat.) & AUROC (scaffold) & MCC (scaffold) \\", r"\midrule"]
rows += bold_best_rows(entries, ACOLS) + [r"\bottomrule", r"\end{tabular}"]
write("tab_arch.tex", "\n".join(rows))

# ------------------------------------------------------------------ Table: architectural factor contrasts (stratified, paired on folds)
def _grp(models):
    return S[S.name.isin([f"DL:{m}" for m in models])].groupby(["seed", "fold"]).auroc.mean()
contrasts = [
    ("Convolutional front-end (CL, CLD) vs.\ none (L, LD)", ["CL1", "CL2", "CL3", "CL1D", "CL2D", "CL3D"], ["L1", "L2", "L3", "L1D", "L2D", "L3D"]),
    ("Dense head (LD, CLD) vs.\ none (L, CL)", ["L1D", "L2D", "L3D", "CL1D", "CL2D", "CL3D"], ["L1", "L2", "L3", "CL1", "CL2", "CL3"]),
    ("Two recurrent layers vs.\ one", ["L2", "CL2", "L2D", "CL2D"], ["L1", "CL1", "L1D", "CL1D"]),
    ("Three recurrent layers vs.\ one", ["L3", "CL3", "L3D", "CL3D"], ["L1", "CL1", "L1D", "CL1D"]),
    ("GRU vs.\ LSTM (one layer)", ["GRU1"], ["L1"]),
    ("Bidirectional vs.\ unidirectional LSTM (one layer)", ["BiLSTM1"], ["L1"]),
    ("CNN-GRU vs.\ CL1", ["CNN-GRU"], ["CL1"]),
]
res = []
for label, a, b in contrasts:
    x, y_ = _grp(a).align(_grp(b), join="inner"); dlt = x - y_
    w = stats.wilcoxon(x, y_, zero_method="zsplit").pvalue if np.any(dlt != 0) else 1.0
    res.append((label, x.mean(), y_.mean(), dlt.mean(), w, dlt.mean() / (dlt.std(ddof=1) or 1.0)))
padj = holm([r[4] for r in res])
rows = [r"\begin{tabular}{lccccc}", r"\toprule", r"Contrast (A vs.\ B) & AUROC A & AUROC B & $\Delta$ & Wilcoxon $p$ & Holm-adjusted $p$ \\", r"\midrule"]
for (label, ma, mb, d_, w, cd), pa in zip(res, padj):
    rows.append(f"{label} & {ma:.3f} & {mb:.3f} & {(d_ if abs(d_) >= 0.0005 else 0.0):+.3f} & {fmtp(w)} & {fmtp(pa)} \\\\")
rows += [r"\bottomrule", r"\end{tabular}"]
write("tab_archfactors.tex", "\n".join(rows))

# ------------------------------------------------------------------ Table: HPO benchmark
last = ht.sort_values("eval").groupby(["scheme", "method", "seed"]).tail(1).copy()
last["best_val"] = 1 - last["best_so_far"]
rows = [r"\begin{tabular}{lcccccccc}", r"\toprule",
        r"Optimiser & Best val.\ AUROC & Test AUROC (strat.) & Test AUROC (scaffold) & Test MCC (strat.) & \multicolumn{2}{c}{$\Delta$AUROC vs.\ random ($d$), $p$ / Holm $p$} \\",
        r"\cmidrule(lr){6-7} & & & & & stratified & scaffold \\", r"\midrule"]
hp = {m: paired_test(hf, "method", m, "random", keys=("hpo_seed", "seed", "fold")) for m in ["tpe", "ga", "pso", "mbo"]}
hpc = {m: paired_test(hf_c, "method", m, "random", keys=("hpo_seed", "seed", "fold")) for m in ["tpe", "ga", "pso", "mbo"]}
_h8 = holm([hp[m]["wilcoxon_p"] for m in hp] + [hpc[m]["wilcoxon_p"] for m in hpc])
hadj = dict(zip(hp, _h8[:4])); hadjc = dict(zip(hpc, _h8[4:]))
meth = ["random", "tpe", "ga", "pso", "mbo"]
vals = {m: (last[last.method == m].best_val, hf[hf.method == m].auroc, hf_c[hf_c.method == m].auroc, hf[hf.method == m].mcc) for m in meth}
bests = [max(round(vals[m][i].mean(), 3) for m in meth) for i in range(4)]
def _b(v, i):
    t = ms(v); return r"\textbf{" + t + "}" if round(v.mean(), 3) == bests[i] else t
for m in meth:
    l = last[last.method == m]; bv, ta, tc_, tm = vals[m]
    if m == "random":
        cs, cc = "--", "--"
    else:
        t = hp[m]; cs = f"{t['diff']:+.3f} ({t['cohen_d']:.2f}), {fmtp(t['wilcoxon_p'])} / {fmtp(hadj[m])}"
        t = hpc[m]; cc = f"{t['diff']:+.3f} ({t['cohen_d']:.2f}), {fmtp(t['wilcoxon_p'])} / {fmtp(hadjc[m])}"
    rows.append(f"{pretty['DL-tuned:'+m].replace('CL tuned, ', '').replace('random search', 'Random search')} & {_b(bv, 0)} & {_b(ta, 1)} & {_b(tc_, 2)} & {_b(tm, 3)} & {cs} & {cc} \\\\")
rows += [r"\bottomrule", r"\end{tabular}"]
write("tab_hpo.tex", "\n".join(rows))

# ------------------------------------------------------------------ Table: best configuration found by each optimiser (+ its validation and test AUROC)
rows = [r"\begin{tabular}{lcccccccccc}", r"\toprule",
        r"Optimiser & Layers & Units & BiRNN & Emb.\ dim & Filters & Kernel & Dropout & LR & Val.\ AUROC & Test AUROC \\", r"\midrule"]
for m in ["random", "tpe", "ga", "pso", "mbo"]:
    l = last[last.method == m]; best_seed = int(l.loc[l.best_val.idxmax(), "seed"])
    cfg = json.loads(hf[(hf.method == m) & (hf.hpo_seed == best_seed)]["config"].iloc[0])
    vbest = float(l.loc[l.seed == best_seed, "best_val"].iloc[0]); tbest = hf[(hf.method == m) & (hf.hpo_seed == best_seed)].auroc.mean()
    rows.append(f"{pretty['DL-tuned:'+m].replace('CL tuned, ', '').replace('random search', 'Random search')} & {len(cfg['rnn_units'])} & {cfg['rnn_units'][0]} & {'yes' if cfg['bidirectional'] else 'no'} & "
                f"{cfg['emb_dim']} & {cfg['conv_filters']} & {cfg['kernel_size']} & {cfg['dropout']:.2f} & {cfg['lr']:.1e} & {vbest:.3f} & {tbest:.3f} \\\\")
rows += [r"\bottomrule", r"\end{tabular}"]
write("tab_bestcfg.tex", "\n".join(rows))

# ------------------------------------------------------------------ Table: tuned configurations vs default configurations (stratified, one Holm family)
tun_tests = [(r"CL tuned (PSO) vs.\ CL1 default", "DL-tuned:pso", "DL:CL1-embedding"),
             (r"CL tuned (PSO) vs.\ CL2D default", "DL-tuned:pso", "DL:CL2D"),
             (r"CL tuned (MBO) vs.\ CL1 default", "DL-tuned:mbo", "DL:CL1-embedding"),
             (r"CL tuned (MBO) vs.\ CL2D default", "DL-tuned:mbo", "DL:CL2D")]
res = [(label, paired_test(S, "name", a, b)) for label, a, b in tun_tests]
padj = holm([t["wilcoxon_p"] for _, t in res])
rows = [r"\begin{tabular}{lccccc}", r"\toprule", r"Comparison (A vs.\ B) & AUROC A & AUROC B & $\Delta$ & Wilcoxon $p$ / Holm $p$ & Cohen's $d$ \\", r"\midrule"]
for (label, t), pa in zip(res, padj):
    rows.append(f"{label} & {t['mean_a']:.3f} & {t['mean_b']:.3f} & {t['diff']:+.3f} & {fmtp(t['wilcoxon_p'])} / {fmtp(pa)} & {t['cohen_d']:.2f} \\\\")
rows += [r"\bottomrule", r"\end{tabular}"]
write("tab_tunedstats.tex", "\n".join(rows))

# ------------------------------------------------------------------ Table: paired tests between model groups (both splits, one Holm family)
tests = [
    (r"XGBoost-ECFP4 vs.\ best default recurrent (CL1 one-hot)", "xgb[ecfp4]", "DL:CL1-onehot"),
    (r"XGBoost-ECFP4 vs.\ best tuned recurrent (PSO)", "xgb[ecfp4]", "DL-tuned:pso"),
    (r"XGBoost-ECFP4 vs.\ MoLFormer-XL", "xgb[ecfp4]", "pre:molformer"),
    (r"XGBoost-ECFP4 vs.\ ChemBERTa-2", "xgb[ecfp4]", "pre:chemberta"),
    (r"XGBoost-ECFP4 vs.\ Chemprop", "xgb[ecfp4]", "pre:chemprop"),
    (r"MoLFormer-XL vs.\ best tuned recurrent (PSO)", "pre:molformer", "DL-tuned:pso"),
]
res = []
for split_name, D_ in [("Stratified", S), ("Scaffold", C)]:
    for label, a, b in tests:
        res.append((split_name, label, paired_test(D_, "name", a, b)))
padj = holm([t["wilcoxon_p"] for _, _, t in res])
rows = [r"\begin{tabular}{lccccc}", r"\toprule", r"Comparison (A vs.\ B) & AUROC A & AUROC B & $\Delta$ & Wilcoxon $p$ / Holm $p$ & Cohen's $d$ \\", r"\midrule"]
prev = None
for (sp, label, t), pa in zip(res, padj):
    if sp != prev:
        if prev: rows.append(r"\midrule")
        rows.append(r"\multicolumn{6}{l}{\emph{%s split, deep models versus the classical reference}} \\" % sp); prev = sp
    if label.startswith("MoLFormer"):
        rows.append(r"\multicolumn{6}{l}{\emph{%s split, within deep models}} \\" % sp)
    rows.append(f"{label} & {t['mean_a']:.3f} & {t['mean_b']:.3f} & {t['diff']:+.3f} & {fmtp(t['wilcoxon_p'])} / {fmtp(pa)} & {t['cohen_d']:.2f} \\\\")
rows += [r"\bottomrule", r"\end{tabular}"]
write("tab_stats.tex", "\n".join(rows))

# ------------------------------------------------------------------ Table: threshold 0.5 vs validation-tuned threshold (primary task, stratified)
tuned_models = ["xgb[ecfp4]", "rf[ecfp4]", "DL:CL1-onehot", "DL:CL1-embedding", "DL-tuned:pso", "pre:molformer", "pre:chemberta", "pre:chemprop"]
rows = [r"\begin{tabular}{lccccccccccc}", r"\toprule",
        r"Model & \multicolumn{3}{c}{At threshold 0.5} & \multicolumn{2}{c}{MCC} & \multicolumn{2}{c}{Balanced acc.} & \multicolumn{2}{c}{F1} & \multicolumn{2}{c}{Threshold} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}",
        r" & Precision & Recall & Specificity & 0.5 & tuned & 0.5 & tuned & 0.5 & tuned & mean & s.d. \\", r"\midrule"]
tm_rows = [(n, S[S.name == n]) for n in tuned_models if len(S[S.name == n]) and not S[S.name == n]["mcc_tuned"].isna().all()]
TC = ["precision", "recall", "specificity", "mcc", "mcc_tuned", "balanced_acc", "balanced_acc_tuned", "f1", "f1_tuned"]
tbest = {c: max(round(g[c].mean(), 3) for _, g in tm_rows) for c in TC}
def _tb(g, c):
    t = f"{g[c].mean():.3f}"; return r"\textbf{" + t + "}" if round(g[c].mean(), 3) == tbest[c] else t
gm_ = S[S.name == "majority[-]"]
rows.append(f"Majority class & {gm_.precision.mean():.3f} & {gm_.recall.mean():.3f} & {gm_.specificity.mean():.3f} & {gm_.mcc.mean():.3f} & -- & {gm_.balanced_acc.mean():.3f} & -- & {gm_.f1.mean():.3f} & -- & -- & -- \\\\")
for n, g in tm_rows:
    rows.append(f"{pretty.get(n, esc(n))} & " + " & ".join(_tb(g, c) for c in TC) + f" & {g.threshold_tuned.mean():.2f} & {g.threshold_tuned.std():.2f} \\\\")
rows += [r"\bottomrule", r"\end{tabular}"]
write("tab_tuned.tex", "\n".join(rows))

# ------------------------------------------------------------------ Table: effect of the validation-tuned threshold on the test MCC (paired per fold), one Holm family
res = []
for n, g in tm_rows:
    d_ = g["mcc_tuned"].values - g["mcc"].values
    w = stats.wilcoxon(g["mcc_tuned"].values, g["mcc"].values, zero_method="zsplit").pvalue if np.any(d_ != 0) else 1.0
    res.append((pretty.get(n, esc(n)), g["mcc"].mean(), g["mcc_tuned"].mean(), d_.mean(), w, d_.mean() / (d_.std(ddof=1) or 1.0)))
padj = holm([r[4] for r in res])
rows = [r"\begin{tabular}{lccccc}", r"\toprule", r"Model & MCC at 0.5 & MCC at tuned threshold & $\Delta$ & Wilcoxon $p$ / Holm $p$ & Cohen's $d$ \\", r"\midrule"]
for (lab, m0, m1, dm, w, cd), pa in zip(res, padj):
    rows.append(f"{lab} & {m0:.3f} & {m1:.3f} & {dm:+.3f} & {fmtp(w)} / {fmtp(pa)} & {cd:.2f} \\\\")
rows += [r"\bottomrule", r"\end{tabular}"]
write("tab_thrstats.tex", "\n".join(rows))

# ------------------------------------------------------------------ Table: agreement between the model ranking under each metric and the AUROC ranking (Table main models)
main_models = [n for g_, names in groups for n in names if len(S[S.name == n])]
gmean = S[S.name.isin(main_models)].groupby("name")[["auroc", "auprc", "mcc", "balanced_acc", "accuracy", "f1"]].mean()
rows = [r"\begin{tabular}{lcc}", r"\toprule", r"Metric & Spearman correlation with the AUROC ranking & Rank of the majority-class predictor \\", r"\midrule"]
for col, lab in [("auprc", "AUPRC"), ("mcc", "MCC"), ("balanced_acc", "Balanced accuracy"), ("accuracy", "Accuracy"), ("f1", "F1 score")]:
    rho = stats.spearmanr(gmean["auroc"], gmean[col]).correlation
    rank = int(gmean[col].rank(ascending=False)["majority[-]"])
    rows.append(f"{lab} & {rho:.2f} & {rank} of {len(gmean)} \\\\")
rows += [r"\bottomrule", r"\end{tabular}"]
write("tab_metricrank.tex", "\n".join(rows))

# ------------------------------------------------------------------ Table: 27 tasks
piv = at.groupby(["task", "model"])["auroc"].agg(["mean", "std"])
rate = at.groupby("task")["pos_rate"].first()
TM = ["rf_ecfp4", "dl_single", "dl_multilabel", "chemberta_multitask", "molformer_multitask", "chemprop_multitask"]
wins = {m: 0 for m in TM}
rows = [r"\begin{tabular}{lccccccc}", r"\toprule", r"Organ class (SIDER) & Pos.\ rate & RF-ECFP4 & LSTM single & LSTM multi & ChemBERTa-2 & MoLFormer & Chemprop \\", r"\midrule"]
for t in sorted(rate.index):
    cells = []
    best = max(TM, key=lambda m: piv.loc[(t, m), "mean"]); wins[best] += 1
    for m in TM:
        v = f"{piv.loc[(t, m), 'mean']:.3f}"
        cells.append(f"\\textbf{{{v}}}" if m == best else v)
    rows.append(f"{esc(t)[:60]} & {rate[t]:.2f} & " + " & ".join(cells) + r" \\")
rows.append(r"\midrule")
mean = at.groupby("model")["auroc"].mean()
rows.append(f"Mean over {len(rate)} tasks & -- & " + " & ".join(f"{mean[m]:.3f}" for m in TM) + r" \\")
rows.append("Tasks won & -- & " + " & ".join(str(wins[m]) for m in TM) + r" \\")
rows += [r"\bottomrule", r"\end{tabular}"]
write("tab_tasks.tex", "\n".join(rows))


# ------------------------------------------------------------------ Table: recurrent schemes (parameter counts from the actual models)
import torch
from adr.models_dl import SeqClassifier, config_from_name
from adr.features import SmilesTokenizer
_tok = SmilesTokenizer("atom").fit(df_clean["smiles"]); _V = _tok.vocab_size
_base = dict(input_mode="embedding", emb_dim=64, rnn_type="lstm", rnn_units=[64], bidirectional=False, conv_filters=64, kernel_size=5, dense_units=64, dropout=0.2)
def _nparams(cfg):
    m = SeqClassifier(_V, 1, **cfg); return sum(p.numel() for p in m.parameters())
rows = [r"\begin{tabular}{llcccccr}", r"\toprule",
        r"Name & Scheme & Conv.\ filters & Cell & Layers $\times$ units & Bidir. & Dense units & Parameters \\", r"\midrule"]
for a in ["L1", "L2", "L3", "CL1", "CL2", "CL3", "L1D", "L2D", "L3D", "CL1D", "CL2D", "CL3D"]:
    c = config_from_name(a, _base); n = len(c["rnn_units"])
    rows.append(f"{a} & {c['scheme']} & {'64 (width 5)' if c['scheme'].startswith('C') else '--'} & LSTM & {n} $\\times$ 64 & no & {'64' if c['scheme'].endswith('D') else '--'} & {_nparams(c):,} \\\\")
rows.append(r"\midrule")
for name, c in [("BiLSTM", {**_base, "scheme": "L", "bidirectional": True}), ("GRU", {**_base, "scheme": "L", "rnn_type": "gru"}), ("CNN-GRU", {**_base, "scheme": "CL", "rnn_type": "gru"})]:
    rows.append(f"{name} & {c['scheme']} & {'64 (width 5)' if c['scheme'].startswith('C') else '--'} & {c['rnn_type'].upper()} & 1 $\\times$ 64 & {'yes' if c['bidirectional'] else 'no'} & -- & {_nparams(c):,} \\\\")
rows += [r"\bottomrule", r"\end{tabular}"]
write("tab_schemes.tex", "\n".join(rows))

# ------------------------------------------------------------------ Table: learning curves (both datasets)
LM = ["rf_ecfp4", "xgb_ecfp4", "lstm_emb", "cl_tuned_pso", "chemberta", "molformer", "chemprop"]
rows = [r"\begin{tabular}{llccccccccc}", r"\toprule",
        r"Dataset & $n_\mathrm{train}$ & RF-ECFP4 & XGB-ECFP4 & CL1 default & CL tuned (PSO) & ChemBERTa-2 & MoLFormer & Chemprop & Best FP $-$ best deep & $p$ \\", r"\midrule"]
lc_nums = {}
for ds, label in [("sider", "SIDER"), ("tox21", "Tox21")]:
    lc = pd.read_csv(RES / f"lc_{ds}.csv"); lc["n"] = lc["n"].astype(str)
    order = sorted(lc.n.unique(), key=lambda x: (x == "all", int(x) if x != "all" else 0))
    ntr = lc.groupby("n").n_train.first()
    for i, n in enumerate(order):
        sub = lc[lc.n == n].assign(name=lambda d: d.model); gm = sub.groupby("model").auroc.mean()
        a, b = gm[[m for m in LM[2:] if m in gm.index]].idxmax(), gm[LM[:2]].idxmax(); t = paired_test(sub, "name", a, b)
        pv = "$<$0.001" if t["wilcoxon_p"] < 0.001 else f"{t['wilcoxon_p']:.3f}"
        present = [m for m in LM if m in gm.index]
        cells = [(f"\\textbf{{{gm[m]:.3f}}}" if m == gm[present].idxmax() else f"{gm[m]:.3f}") if m in gm.index else "--" for m in LM]
        rows.append(f"{label if i == 0 else ''} & {int(ntr[n])} & " + " & ".join(cells) + f" & {gm[b]-gm[a]:+.3f} & {pv} \\\\")
        lc_nums[(ds, n)] = (gm[b] - gm[a], t["wilcoxon_p"], int(ntr[n]))
    if ds == "sider":
        rows.append(r"\midrule")
rows += [r"\bottomrule", r"\end{tabular}"]
write("tab_lc.tex", "\n".join(rows))

# ------------------------------------------------------------------ Table: learning-curve slopes (AUROC gain per doubling of the training set) and total gain
LCMODELS = [("rf_ecfp4", "Random forest, ECFP4"), ("xgb_ecfp4", "XGBoost, ECFP4"), ("lstm_emb", "CL1 default (LSTM, embedding)"), ("cl_tuned_pso", "CL tuned (PSO configuration)"), ("chemberta", "ChemBERTa-2"), ("molformer", "MoLFormer-XL"), ("chemprop", "Chemprop D-MPNN")]
rows = [r"\begin{tabular}{lcccc}", r"\toprule", r"Model & \multicolumn{2}{c}{SIDER (250 to 1{,}077)} & \multicolumn{2}{c}{Tox21 (250 to 5{,}087)} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}", r" & Gain per doubling & Total gain & Gain per doubling & Total gain \\", r"\midrule"]
slope_cells = {}
for ds in ["sider", "tox21"]:
    lc = pd.read_csv(RES / f"lc_{ds}.csv"); lc["n"] = lc.n.astype(str)
    order = sorted(lc.n.unique(), key=lambda x: (x == "all", int(x) if x != "all" else 0))
    ntr = lc.groupby("n").n_train.first().loc[order].values; m = lc.groupby(["n", "model"]).auroc.mean().unstack().loc[order]
    for key, _ in LCMODELS:
        if key not in m.columns or m[key].isna().any():
            slope_cells[(ds, key)] = (np.nan, np.nan); continue
        y = m[key].values; slope = np.polyfit(np.log2(ntr), y, 1)[0]; slope_cells[(ds, key)] = (slope, y[-1] - y[0])
for key, lab in LCMODELS:
    a, b = slope_cells[("sider", key)]; c_, d_ = slope_cells[("tox21", key)]
    f = lambda v: "--" if np.isnan(v) else f"{v:+.3f}"
    rows.append(f"{lab} & {f(a)} & {f(b)} & {f(c_)} & {f(d_)} \\\\")
rows += [r"\bottomrule", r"\end{tabular}"]
write("tab_lcslope.tex", "\n".join(rows))

# ------------------------------------------------------------------ Table: tuned vs default and RF vs tuned along the learning curve (one Holm family)
res = []
for ds, label in [("sider", "SIDER"), ("tox21", "Tox21")]:
    lc = pd.read_csv(RES / f"lc_{ds}.csv"); lc["n"] = lc.n.astype(str); lc["name"] = lc.model
    order = sorted(lc.n.unique(), key=lambda x: (x == "all", int(x) if x != "all" else 0)); ntr = lc.groupby("n").n_train.first()
    for n in order:
        sub = lc[lc.n == n]
        res.append((label, int(ntr[n]), "tuned vs.\ default", paired_test(sub, "name", "cl_tuned_pso", "lstm_emb")))
        res.append((label, int(ntr[n]), "RF vs.\ tuned", paired_test(sub, "name", "rf_ecfp4", "cl_tuned_pso")))
padj = holm([t["wilcoxon_p"] for *_, t in res])
rows = [r"\begin{tabular}{llcccccc}", r"\toprule",
        r"Dataset & $n_\mathrm{train}$ & \multicolumn{3}{c}{CL tuned (PSO) vs.\ CL1 default} & \multicolumn{3}{c}{RF-ECFP4 vs.\ CL tuned (PSO)} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}", r" & & $\Delta$ & $p$ / Holm $p$ & $d$ & $\Delta$ & $p$ / Holm $p$ & $d$ \\", r"\midrule"]
by = {}
for (label, n, kind, t), pa in zip(res, padj):
    by.setdefault((label, n), {})[kind] = (t, pa)
prev = None
for (label, n), d_ in by.items():
    if prev and label != prev: rows.append(r"\midrule")
    a, pa = d_["tuned vs.\ default"]; b, pb = d_["RF vs.\ tuned"]
    rows.append(f"{label if label != prev else ''} & {n} & {a['diff']:+.3f} & {fmtp(a['wilcoxon_p'])} / {fmtp(pa)} & {a['cohen_d']:.2f} & {b['diff']:+.3f} & {fmtp(b['wilcoxon_p'])} / {fmtp(pb)} & {b['cohen_d']:.2f} \\\\"); prev = label
rows += [r"\bottomrule", r"\end{tabular}"]
write("tab_lctuned.tex", "\n".join(rows))

# ------------------------------------------------------------------ Table: paired tests over the 27 organ classes (task x seed x fold), one Holm family
both_ = at.copy(); both_["name"] = both_["model"]
task_tests = [(r"RF-ECFP4 vs.\ LSTM single-task", "rf_ecfp4", "dl_single"), (r"RF-ECFP4 vs.\ LSTM multi-label", "rf_ecfp4", "dl_multilabel"),
              (r"RF-ECFP4 vs.\ ChemBERTa-2 multi-task", "rf_ecfp4", "chemberta_multitask"), (r"RF-ECFP4 vs.\ MoLFormer-XL multi-task", "rf_ecfp4", "molformer_multitask"),
              (r"RF-ECFP4 vs.\ Chemprop multi-task", "rf_ecfp4", "chemprop_multitask"), (r"MoLFormer-XL vs.\ Chemprop (multi-task)", "molformer_multitask", "chemprop_multitask"),
              (r"LSTM multi-label vs.\ LSTM single-task", "dl_multilabel", "dl_single")]
res = [(label, paired_test(both_, "name", a, b, keys=("task", "seed", "fold"))) for label, a, b in task_tests]
padj = holm([t["wilcoxon_p"] for _, t in res])
rows = [r"\begin{tabular}{lcccccc}", r"\toprule", r"Comparison (A vs.\ B) & AUROC A & AUROC B & $\Delta$ & Wilcoxon $p$ / Holm $p$ & Cohen's $d$ & Pairs \\", r"\midrule"]
for (label, t), pa in zip(res, padj):
    rows.append(f"{label} & {t['mean_a']:.3f} & {t['mean_b']:.3f} & {t['diff']:+.3f} & {fmtp(t['wilcoxon_p'])} / {fmtp(pa)} & {t['cohen_d']:.2f} & {t['n']} \\\\")
rows += [r"\bottomrule", r"\end{tabular}"]
write("tab_tasksstats.tex", "\n".join(rows))

# ------------------------------------------------------------------ Table: the 12 most confident positive predictions explained with Integrated Gradients
at = pd.read_csv(RES / "attributions_top.csv")
ATTR_NAMES = {1030: "Hydroxyurea", 1016: "Nitroxyl", 1211: "Deferasirox", 1221: "Nitrazepam", 954: "Raltegravir", 806: "Naproxen",
              450: "Propylthiouracil", 492: "Ceftaroline fosamil", 1230: "Fluorescein", 490: "Dorzolamide", 30: "Entecavir", 137: "Dabrafenib"}
rows = [r"\begin{tabular}{clccrlr}", r"\toprule",
        r"Rank & Compound & Heavy atoms & SIDER label & $\hat{p}$ & Token with the largest $|A_t|$ & $A_t$ \\", r"\midrule"]
from rdkit import Chem
for _, r in at.sort_values("rank").iterrows():
    tt = json.loads(r.top_tokens)[0]
    nha = Chem.MolFromSmiles(r.smiles).GetNumHeavyAtoms()
    rows.append(f"{int(r['rank'])+1} & {ATTR_NAMES.get(int(r['index']), r.smiles)} & {nha} & {int(r.label)} & {r.prob:.3f} & \\texttt{{{esc(tt[0])}}} & {tt[1]:+.2f} \\\\")
rows += [r"\bottomrule", r"\end{tabular}"]
write("tab_attrtop.tex", "\n".join(rows))

# ------------------------------------------------------------------ Table: mean attribution per token type over the 12 molecules (token types occurring at least three times)
ti = pd.read_csv(RES / "token_importance.csv")
ti = ti[ti["count"] >= 3].sort_values("mean_attr", ascending=False)
rows = [r"\begin{tabular}{lrr}", r"\toprule", r"Token & Occurrences & Mean attribution \\", r"\midrule"]
for _, r in ti.iterrows():
    rows.append(f"\\texttt{{{esc(r.token)}}} & {int(r['count'])} & {r.mean_attr:+.3f} \\\\")
rows += [r"\bottomrule", r"\end{tabular}"]
write("tab_tokenimp.tex", "\n".join(rows))

# ------------------------------------------------------------------ Table: positive rate of the primary task by presence of nitrogen and by molecule size (whole SIDER set)
sd = pd.read_csv(HERE.parent / "code" / "data" / "processed" / "sider_clean.csv")
_task = [c for c in sd.columns if "lood" in c][0]
_mols = sd.smiles.apply(Chem.MolFromSmiles)
_hasN = _mols.apply(lambda m: any(a.GetSymbol() == "N" for a in m.GetAtoms()))
_nha = _mols.apply(lambda m: m.GetNumHeavyAtoms())
rows = [r"\begin{tabular}{lrr}", r"\toprule", r"Subset of molecules & $n$ & Positive rate \\", r"\midrule"]
for lab, mask in [("At least one nitrogen atom", _hasN), ("No nitrogen atom", ~_hasN),
                  ("At most 10 heavy atoms", _nha <= 10), ("More than 10 heavy atoms", _nha > 10), ("All molecules", _nha > 0)]:
    rows.append(f"{lab} & {int(mask.sum())} & {sd[_task][mask].mean():.3f} \\\\")
rows += [r"\bottomrule", r"\end{tabular}"]
write("tab_attrbase.tex", "\n".join(rows))

# ------------------------------------------------------------------ numbers for the text
nums = {
    "xgb_auroc": S[S.name == "xgb[ecfp4]"].auroc.mean(), "rf_auroc": S[S.name == "rf[ecfp4]"].auroc.mean(),
    "index_auroc": S[S.name == "DL:CL1-index"].auroc.mean(), "emb_auroc": S[S.name == "DL:CL1-embedding"].auroc.mean(),
    "onehot_auroc": S[S.name == "DL:CL1-onehot"].auroc.mean(),
    "best_untuned": S[S.name.str.startswith("DL:")].groupby("name").auroc.mean().max(),
    "pso_auroc": hf[hf.method == "pso"].auroc.mean(), "mbo_auroc": hf[hf.method == "mbo"].auroc.mean(),
    "random_auroc": hf[hf.method == "random"].auroc.mean(), "majority_fone": S[S.name == "majority[-]"].f1.mean(),
    "rf_tasks_mean": mean["rf_ecfp4"], "dl_tasks_mean": mean["dl_single"], "rf_wins": wins["rf_ecfp4"], "n_tasks": len(rate),
    "n_dl_trainings": len(dl), "n_hpo_evals": len(ht),
    "pso_auroc_c": hf_c[hf_c.method == "pso"].auroc.mean(), "mbo_auroc_c": hf_c[hf_c.method == "mbo"].auroc.mean(), "random_auroc_c": hf_c[hf_c.method == "random"].auroc.mean(),
    "molformer_auroc": S[S.name == "pre:molformer"].auroc.mean(), "chemberta_auroc": S[S.name == "pre:chemberta"].auroc.mean(),
    "chemprop_auroc": S[S.name == "pre:chemprop"].auroc.mean(),
    "lc_gap_tox_small": lc_nums[("tox21", "250")][0], "lc_gap_tox_full": lc_nums[("tox21", "all")][0], "lc_p_tox_full": lc_nums[("tox21", "all")][1],
    "lc_gap_tox_fourk": lc_nums[("tox21", "4000")][0], "lc_p_tox_fourk": lc_nums[("tox21", "4000")][1], "lc_n_tox_full": lc_nums[("tox21", "all")][2],
    "lc_gap_sider_full": lc_nums[("sider", "all")][0], "lc_p_sider_full": lc_nums[("sider", "all")][1],
    "n_lc_trainings": int(pd.read_csv(RES / "lc_tox21.csv").shape[0] + pd.read_csv(RES / "lc_sider.csv").shape[0]),
}
macros = "\n".join(f"\\newcommand{{\\num{k.replace('_','')}}}{{{v:.3f}}}" if isinstance(v, float) else f"\\newcommand{{\\num{k.replace('_','')}}}{{{v}}}" for k, v in nums.items())
write("numbers.tex", "% auto-generated numbers used in the text\n" + macros + "\n")
