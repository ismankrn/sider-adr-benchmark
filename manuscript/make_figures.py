"""Publication figures for the manuscript, generated from ../code/results. Run: ../code/.venv/bin/python make_figures.py"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; RES = HERE.parent / "code" / "results"; OUT = HERE / "figures"; OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})

bl = pd.read_csv(RES / "baselines.csv"); bl["name"] = bl.model + "[" + bl.features.astype(str) + "]"
dl = pd.read_csv(RES / "dl_untuned.csv"); dl["name"] = "DL:" + dl.model
hf = pd.read_csv(RES / "hpo_final.csv"); hf["name"] = "DL-tuned:" + hf.method
pr = pd.read_csv(RES / "pretrained_primary.csv"); pr["name"] = "pre:" + pr.model
allr = pd.concat([bl, dl, hf, pr], ignore_index=True)

# rows (bottom -> top order will be reversed for display), with group colours
ROWS = [
    ("majority[-]", "Majority class", "baseline"), ("random[-]", "Random scores", "baseline"),
    ("logreg[ecfp4]", "Logistic regression, ECFP4", "classical"), ("svm[ecfp4]", "SVM, ECFP4", "classical"),
    ("rf[descriptors]", "Random forest, descriptors", "classical"), ("xgb[descriptors]", "XGBoost, descriptors", "classical"),
    ("rf[ecfp4]", "Random forest, ECFP4", "classical"), ("xgb[ecfp4]", "XGBoost, ECFP4", "classical"),
    ("DL:CL1-index", "CL1, integer index", "recurrent"), ("DL:CL1-embedding", "CL1, learned embedding", "recurrent"), ("DL:CL1-onehot", "CL1, one-hot", "recurrent"),
    ("DL:L3D", "L3D", "recurrent"), ("DL:CL3D", "CL3D", "recurrent"), ("DL:CL2D", "CL2D", "recurrent"), ("DL:BiLSTM1", "BiLSTM", "recurrent"), ("DL:GRU1", "GRU", "recurrent"), ("DL:CNN-GRU", "CNN-GRU", "recurrent"),
    ("DL-tuned:mbo", "CL tuned, MBO", "tuned"), ("DL-tuned:ga", "CL tuned, GA", "tuned"), ("DL-tuned:pso", "CL tuned, PSO", "tuned"),
    ("pre:chemprop", "Chemprop D-MPNN", "pretrained"), ("pre:chemberta", "ChemBERTa-2", "pretrained"), ("pre:molformer", "MoLFormer-XL", "pretrained"),
]
COL = {"baseline": "#9e9e9e", "classical": "#1f77b4", "recurrent": "#2ca02c", "tuned": "#ff7f0e", "pretrained": "#9467bd"}

fig, axes = plt.subplots(1, 2, figsize=(7.0, 5.2), sharey=True)
labels_all = [label for _, label, _ in ROWS]
for ax, split in zip(axes, ["stratified", "scaffold"]):
    d = allr[allr.split == split]
    data, pos, colors = [], [], []
    for i, (key, label, grp) in enumerate(ROWS, start=1):
        v = d[d.name == key].auroc.dropna().values
        if len(v) == 0:
            continue                                  # keep the row position empty (e.g. tuned models have no scaffold runs)
        data.append(v); pos.append(i); colors.append(COL[grp])
    bp = ax.boxplot(data, positions=pos, orientation="horizontal", widths=0.6, patch_artist=True, showfliers=True,
                    flierprops=dict(marker="o", markersize=2, markerfacecolor="k", markeredgecolor="none", alpha=.6),
                    medianprops=dict(color="k", lw=1.0), whiskerprops=dict(lw=.7), capprops=dict(lw=.7), boxprops=dict(lw=.7))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(.45); patch.set_edgecolor(c)
    ax.axvline(0.5, ls="--", lw=.8, c="k"); ax.set_xlim(0.42, 0.82); ax.set_xlabel("Test AUROC")
    ax.set_title(f"{split.capitalize()} split", fontsize=9); ax.grid(axis="x", lw=.4, alpha=.5)
    ax.set_yticks(range(1, len(ROWS) + 1)); ax.set_yticklabels(labels_all); ax.set_ylim(0.3, len(ROWS) + 0.7)
# group separators
for ax in axes:
    for y in [2.5, 8.5, 17.5, 20.5]:
        ax.axhline(y, lw=.5, c="#bbbbbb")
from matplotlib.patches import Patch
LEGEND = [("baseline", "Trivial baselines"), ("classical", "Classical learners"), ("recurrent", "Recurrent networks"),
          ("tuned", "Tuned recurrent networks"), ("pretrained", "Pretrained and graph models")]
handles = [Patch(facecolor=COL[k], edgecolor=COL[k], alpha=.6, label=lab) for k, lab in LEGEND]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7.5, handlelength=1.4, columnspacing=1.5, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(w_pad=1.0, rect=(0, 0.075, 1, 1))
fig.savefig(OUT / "box_primary.pdf"); fig.savefig(OUT / "box_primary.png", dpi=300); plt.close(fig)
print("wrote", OUT / "box_primary.pdf")

# ------------------------------------------------------------------ optimiser benchmark: convergence + distribution of all evaluations
ht = pd.read_csv(RES / "hpo_trace.csv"); ht["best_val"] = 1 - ht["best_so_far"]
ORDER = [("random", "Random search", "#7f7f7f"), ("tpe", "TPE", "#9467bd"), ("ga", "GA", "#2ca02c"), ("pso", "PSO", "#d62728"), ("mbo", "MBO", "#ff7f0e")]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw=dict(width_ratios=[1.6, 1]))
for key, lab, col in ORDER:
    g = ht[ht.method == key].groupby("eval")["best_val"].agg(["mean", "std"])
    ax1.plot(g.index, g["mean"], color=col, lw=1.3, label=lab)
    ax1.fill_between(g.index, g["mean"] - g["std"].fillna(0), g["mean"] + g["std"].fillna(0), color=col, alpha=.12, lw=0)
ax1.set_xlabel("Model trainings"); ax1.set_ylabel("Best validation AUROC so far"); ax1.set_xlim(1, 100); ax1.set_ylim(0.55, 0.73)
ax1.grid(lw=.4, alpha=.5); ax1.legend(frameon=False, fontsize=7, loc="lower right"); ax1.set_title("(a) Convergence", fontsize=9)
data = [ht[ht.method == k]["val_auroc"].values for k, _, _ in ORDER]
bp = ax2.boxplot(data, orientation="vertical", widths=0.6, patch_artist=True, showfliers=True,
                 flierprops=dict(marker="o", markersize=1.5, markerfacecolor="k", markeredgecolor="none", alpha=.5),
                 medianprops=dict(color="k", lw=1.0), whiskerprops=dict(lw=.7), capprops=dict(lw=.7), boxprops=dict(lw=.7))
for patch, (_, _, col) in zip(bp["boxes"], ORDER):
    patch.set_facecolor(col); patch.set_alpha(.45); patch.set_edgecolor(col)
ax2.set_xticks(range(1, 6)); ax2.set_xticklabels([lab for _, lab, _ in ORDER], rotation=30, ha="right")
ax2.set_ylabel("Validation AUROC of each evaluation"); ax2.axhline(0.5, ls="--", lw=.7, c="k"); ax2.grid(axis="y", lw=.4, alpha=.5)
ax2.set_title("(b) All 300 evaluations", fontsize=9)
fig.tight_layout(w_pad=1.5); fig.savefig(OUT / "hpo_benchmark.pdf"); fig.savefig(OUT / "hpo_benchmark.png", dpi=300); plt.close(fig)
print("wrote", OUT / "hpo_benchmark.pdf")

# ------------------------------------------------------------------ training curves of the three encodings (CL1, fold 0, three seeds)
tc_path = RES / "training_curves.csv"
if tc_path.exists():
    tc = pd.read_csv(tc_path)
    ENC = [("index", "Integer index", "#d62728"), ("onehot", "One-hot", "#1f77b4"), ("embedding", "Learned embedding", "#2ca02c")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    for ax, (key, lab, col) in zip([None] * 0, []): pass
    for key, lab, col in ENC:
        d = tc[tc.encoding == key]
        for seed, g in d.groupby("seed"):
            axes[0].plot(g.epoch + 1, g.train_loss, color=col, lw=1.0, alpha=.85, label=lab if seed == d.seed.min() else None)
            axes[1].plot(g.epoch + 1, g.val_loss, color=col, lw=1.0, alpha=.85, label=lab if seed == d.seed.min() else None)
    for ax, title in zip(axes, ["(a) Training loss", "(b) Validation loss"]):
        ax.axhline(0.506, ls="--", lw=.7, c="k"); ax.set_xlabel("Epoch"); ax.set_title(title, fontsize=9); ax.grid(lw=.4, alpha=.5)
    axes[0].set_ylabel("Class-weighted binary cross-entropy"); axes[1].legend(frameon=False, fontsize=7.5)
    fig.tight_layout(w_pad=1.2); fig.savefig(OUT / "training_curves.pdf"); fig.savefig(OUT / "training_curves.png", dpi=300); plt.close(fig)
    print("wrote", OUT / "training_curves.pdf")

# ------------------------------------------------------------------ learning curves (publication style, both datasets)
LCM = [("rf_ecfp4", "Random forest, ECFP4", "#1f77b4", "o"), ("xgb_ecfp4", "XGBoost, ECFP4", "#17becf", "s"),
       ("lstm_emb", "CL1 default (LSTM, embedding)", "#2ca02c", "^"), ("cl_tuned_pso", "CL tuned (PSO configuration)", "#ff7f0e", "<"), ("chemberta", "ChemBERTa-2", "#d62728", "v"),
       ("molformer", "MoLFormer-XL", "#9467bd", "D"), ("chemprop", "Chemprop D-MPNN", "#8c564b", "P")]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
for ax, ds, title in zip(axes, ["sider", "tox21"], ["(a) SIDER, Blood and lymphatic system disorders", "(b) Tox21, NR-AhR"]):
    lc = pd.read_csv(RES / f"lc_{ds}.csv"); lc["n"] = lc.n.astype(str)
    order = sorted(lc.n.unique(), key=lambda x: (x == "all", int(x) if x != "all" else 0))
    ntr = lc.groupby("n").n_train.first().loc[order].values
    g = lc.groupby(["n", "model"]).auroc.agg(["mean", "std", "count"]).unstack()
    for key, lab, col, mk in LCM:
        if key not in g["mean"].columns: continue
        mu = g["mean"][key].loc[order].values; se = (g["std"][key] / np.sqrt(g["count"][key])).loc[order].values
        ax.errorbar(ntr, mu, yerr=se, color=col, marker=mk, ms=3.5, lw=1.1, capsize=2, label=lab)
    ax.set_xscale("log"); ax.set_xticks(ntr[:-1]); ax.set_xticklabels([str(int(n)) for n in ntr[:-1]], fontsize=7)   # the full fold (last point) is unlabeled to avoid overlap
    ax.minorticks_off(); ax.set_xlabel("Training molecules"); ax.set_title(title, fontsize=8.5); ax.grid(lw=.4, alpha=.5)
axes[0].set_ylabel("Test AUROC (mean ± s.e., 15 folds)"); axes[0].set_ylim(0.54, 0.75); axes[1].set_ylim(0.68, 0.91)
axes[1].legend(frameon=False, fontsize=6.8, loc="lower right")
fig.tight_layout(w_pad=1.2); fig.savefig(OUT / "learning_curves.pdf"); fig.savefig(OUT / "learning_curves.png", dpi=300); plt.close(fig)
print("wrote", OUT / "learning_curves.pdf")


# ------------------------------------------------------------------ 27 organ classes: dot plot sorted by RF AUROC
at = pd.read_csv(RES / "all_tasks.csv"); pt = pd.read_csv(RES / "pretrained_tasks.csv")
both = pd.concat([at, pt[[c for c in pt.columns if c in at.columns]]], ignore_index=True)
TM = [("rf_ecfp4", "Random forest, ECFP4", "#1f77b4", "o"), ("dl_single", "LSTM single-task", "#2ca02c", "^"), ("dl_multilabel", "LSTM multi-label", "#98df8a", "v"),
      ("chemberta_multitask", "ChemBERTa-2 multi-task", "#d62728", "s"), ("molformer_multitask", "MoLFormer-XL multi-task", "#9467bd", "D"), ("chemprop_multitask", "Chemprop multi-task", "#8c564b", "P")]
piv = both.groupby(["task", "model"]).auroc.agg(["mean", "std", "count"]).unstack()
order = piv["mean"]["rf_ecfp4"].sort_values().index.tolist()
rate = both.groupby("task").pos_rate.first().loc[order]
fig, ax = plt.subplots(figsize=(7.0, 6.4))
ypos = np.arange(len(order))
for key, lab, col, mk in TM:
    mu = piv["mean"][key].loc[order].values; se = (piv["std"][key] / np.sqrt(piv["count"][key])).loc[order].values
    ax.errorbar(mu, ypos, xerr=se, fmt=mk, color=col, ms=4, lw=.8, capsize=1.5, label=lab, alpha=.9)
ax.set_yticks(ypos); ax.set_yticklabels([f"{(t if len(t) <= 44 else t[:41].rstrip() + chr(8230))} ({r:.2f})" for t, r in zip(order, rate.values)], fontsize=7)
ax.axvline(0.5, ls="--", lw=.8, c="k"); ax.set_xlabel("Test AUROC (mean ± s.e., 15 folds)"); ax.grid(axis="x", lw=.4, alpha=.5)
ax.set_xlim(0.44, 0.82); ax.legend(frameon=False, fontsize=7, loc="lower right"); ax.set_ylim(-0.7, len(order) - 0.3)
fig.tight_layout(); fig.savefig(OUT / "all_tasks.pdf"); fig.savefig(OUT / "all_tasks.png", dpi=300); plt.close(fig)
print("wrote", OUT / "all_tasks.pdf")
