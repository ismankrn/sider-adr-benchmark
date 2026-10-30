"""Step 7: tables (mean ± std), Wilcoxon tests, and figures from all result CSVs."""
from _common import *
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from adr.stats import summarize, paired_test, to_markdown
R, F = resolve("results"), resolve("figures"); F.mkdir(exist_ok=True)
md = ["# Hasil ringkasan\n"]
def load(name):
    p = R / name
    return pd.read_csv(p) if p.exists() else None

bl, dl, hf, ht, at = load("baselines.csv"), load("dl_untuned.csv"), load("hpo_final.csv"), load("hpo_trace.csv"), load("all_tasks.csv")
pr, pt = load("pretrained_primary.csv"), load("pretrained_tasks.csv")
if pt is not None and at is not None:
    at = pd.concat([at, pt[[c for c in pt.columns if c in at.columns]]], ignore_index=True)
frames = []
if bl is not None:
    bl["name"] = bl["model"] + "[" + bl["features"].astype(str) + "]"; frames.append(bl)
if dl is not None:
    dl["name"] = "DL:" + dl["model"]; frames.append(dl)
if hf is not None:
    hf["name"] = "DL-tuned:" + hf["scheme"] + "/" + hf["method"]; frames.append(hf)
    hf = hf[hf.split == "stratified"]
if pr is not None:
    pr["name"] = "pretrained:" + pr["model"]; frames.append(pr)
if frames:
    allr = pd.concat(frames, ignore_index=True)
    for split, g in allr.groupby("split"):
        tab = summarize(g, ["name"]).sort_values("auroc", ascending=False)
        md += [f"\n## Tabel utama ({split} split, mean ± std over seeds x folds)\n", to_markdown(tab)]
        plt.figure(figsize=(10, max(4, 0.3 * tab.shape[0])))
        sns.boxplot(data=g, y="name", x="auroc", order=tab["name"]); plt.axvline(0.5, ls="--", c="k")
        plt.title(f"AUROC per model ({split})"); plt.tight_layout(); plt.savefig(F / f"auroc_boxplot_{split}.png", dpi=150); plt.close()
    # statistical tests
    tests = []
    s = allr[allr["split"] == "stratified"]
    names = s["name"].unique()
    if "DL:CL1-embedding" in names and "DL:CL1-index" in names:
        tests.append(paired_test(s, "name", "DL:CL1-embedding", "DL:CL1-index"))
    best_bl = s[s["name"].str.contains(r"\[")].groupby("name")["auroc"].mean().idxmax() if bl is not None else None
    for n in names:
        if n.startswith("pretrained") and best_bl:
            tests.append(paired_test(s, "name", n, best_bl))
        if n.startswith("DL-tuned") and best_bl:
            tests.append(paired_test(s, "name", n, best_bl))
        if n.startswith("DL-tuned") and "DL:CL1" in names:
            tests.append(paired_test(s, "name", n, "DL:CL1"))
    if tests:
        md += ["\n## Uji statistik (Wilcoxon signed-rank, paired on seed x fold, AUROC)\n", to_markdown(pd.DataFrame(tests).round(4))]
if ht is not None:
    md += ["\n## HPO: best validation AUROC per method (mean ± std over HPO seeds)\n"]
    last = ht.sort_values("eval").groupby(["scheme", "method", "seed"]).tail(1)
    last["best_val_auroc"] = 1 - last["best_so_far"]
    md.append(to_markdown(summarize(last, ["scheme", "method"], ["best_val_auroc", "elapsed"])))
    plt.figure(figsize=(7, 4))
    for (sc, me), g in ht.groupby(["scheme", "method"]):
        curve = g.groupby("eval")["best_so_far"].agg(["mean", "std"])
        plt.plot(curve.index, 1 - curve["mean"], label=f"{sc}/{me}")
        plt.fill_between(curve.index, 1 - curve["mean"] - curve["std"].fillna(0), 1 - curve["mean"] + curve["std"].fillna(0), alpha=.15)
    plt.xlabel("model evaluations"); plt.ylabel("best validation AUROC so far"); plt.legend(); plt.tight_layout()
    plt.savefig(F / "hpo_convergence.png", dpi=150); plt.close()
    if hf is not None:
        md += ["\n## HPO: CV test performance of the tuned configs\n", to_markdown(summarize(hf, ["scheme", "method"]))]
        for m in ["mbo", "ga", "pso", "tpe"]:
            if m in hf["method"].unique() and "random" in hf["method"].unique():
                md.append(str(paired_test(hf, "method", m, "random", keys=("hpo_seed", "seed", "fold"))))
if at is not None:
    md += ["\n## Semua task SIDER (AUROC mean over seeds x folds)\n"]
    piv = at.groupby(["task", "model"])["auroc"].mean().unstack().round(3)
    piv["pos_rate"] = at.groupby("task")["pos_rate"].first().round(3)
    md.append(to_markdown(piv.reset_index()))
    plt.figure(figsize=(9, 8)); sns.barplot(data=at, y="task", x="auroc", hue="model", errorbar="sd")
    plt.axvline(0.5, ls="--", c="k"); plt.tight_layout(); plt.savefig(F / "all_tasks_auroc.png", dpi=150); plt.close()
    md.append("\nMean over tasks:\n" + to_markdown(at.groupby("model")[["auroc", "auprc", "mcc"]].mean().round(3).reset_index()))
for ds in ["sider", "tox21"]:
    lc = load(f"lc_{ds}.csv")
    if lc is None:
        continue
    lc["n"] = lc["n"].astype(str)
    order = sorted(lc.n.unique(), key=lambda x: (x == "all", int(x) if x != "all" else 0))
    ntr = lc.groupby("n").n_train.first().loc[order]
    g = lc.groupby(["n", "model"]).auroc.agg(["mean", "std"]).unstack()
    tab = g["mean"].loc[order].round(3); tab.insert(0, "n_train", ntr)
    md += [f"\n## Learning curve {ds} (AUROC mean over seeds x folds)\n", to_markdown(tab.reset_index())]
    fpm = ["xgb_ecfp4", "rf_ecfp4"]; dpm = [m for m in ["lstm_emb", "chemberta", "chemprop", "molformer"] if m in g["mean"].columns]
    gap_rows = []
    for n in order:
        sub = lc[lc.n == n].assign(name=lambda d: d.model); gm = sub.groupby("model").auroc.mean()
        a, b = gm[dpm].idxmax(), gm[fpm].idxmax(); t = paired_test(sub, "name", a, b)
        gap_rows.append(dict(n=n, n_train=int(ntr[n]), best_fingerprint=b, auroc_fp=round(gm[b], 3), best_deep=a, auroc_deep=round(gm[a], 3), gap=round(gm[b] - gm[a], 3), wilcoxon_p=round(t["wilcoxon_p"], 4)))
    md += [f"\nGap fingerprint vs deep ({ds}):\n", to_markdown(pd.DataFrame(gap_rows))]
    plt.figure(figsize=(6.5, 4.2))
    for m in ["rf_ecfp4", "xgb_ecfp4", "lstm_emb", "chemberta", "molformer", "chemprop"]:
        if m not in g["mean"].columns:
            continue
        mu, sd = g["mean"][m].loc[order].values, g["std"][m].loc[order].fillna(0).values
        plt.errorbar(ntr.values, mu, yerr=sd / np.sqrt(15), marker="o", capsize=2, label=m)
    plt.xscale("log"); plt.xlabel("training molecules"); plt.ylabel("test AUROC (mean ± s.e., 15 folds)")
    plt.title(f"Learning curve: {ds.upper()}"); plt.grid(alpha=.3); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(F / f"lc_{ds}.png", dpi=150); plt.close()
(R / "summary.md").write_text("\n".join(md)); print("\n".join(md)); print("\nfigures ->", F)
