"""Step 2: classical baselines (LogReg/RF/XGB/SVM on ECFP4 + RDKit descriptors) + trivial baselines."""
from _common import *
from adr.baselines import fit_predict
cfg = load_config(); df, task = load_processed(cfg)
smiles, y = df["smiles"].tolist(), df[task].values
feats = {}
if "ecfp4" in cfg["baselines"]["features"]:
    feats["ecfp4"] = Fe.ecfp(smiles, 2, 2048)
if "descriptors" in cfg["baselines"]["features"]:
    feats["descriptors"], _ = Fe.rdkit_descriptors(smiles)
out = "results/baselines.csv"
for split, seed, fold, tr, va, te in iter_folds(cfg, df, task):
    rows = []
    base = dict(task=task, split=split, seed=seed, fold=fold)
    rows.append({**base, "model": "majority", "features": "-", **M.majority_baseline(y[tr], y[te])})
    rows.append({**base, "model": "random", "features": "-", **M.random_baseline(y[te], seed)})
    trva = np.concatenate([tr, va])      # classical models: no early stopping -> use train+val
    for fname, X in feats.items():
        for mname in cfg["baselines"]["models"]:
            _, p_va = fit_predict(mname, X[tr], y[tr], X[va], seed)
            _, p_te = fit_predict(mname, X[trva], y[trva], X[te], seed)
            m = eval_with_threshold(y[va], p_va, y[te], p_te, cfg["eval"]["threshold_criterion"])
            rows.append({**base, "model": mname, "features": fname, **m})
            print(f"{split} s{seed} f{fold} {mname:7s} {fname:11s} AUROC={m['auroc']:.3f} MCC={m['mcc']:.3f}")
    append_rows(out, rows)
print("saved ->", out)
