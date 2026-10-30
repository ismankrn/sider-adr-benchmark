"""Step 5: generalisation across all 27 SIDER organ classes. Single-task RF/DL per task and one
multi-label DL model trained jointly on all tasks. Stratified CV only (seeds from config)."""
import argparse, json
from _common import *
from adr.baselines import fit_predict
from adr.models_dl import train_model, predict_proba
ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); args = ap.parse_args()
cfg = load_config(); df = pd.read_csv(resolve(cfg["data"]["processed_path"]))
tasks = [c for c in df.columns if c not in ("smiles", "smiles_raw", "length")]
if cfg["multitask"]["tasks"] != "all":
    tasks = [find_task(tasks, q) for q in cfg["multitask"]["tasks"]]
if args.quick:
    tasks = tasks[:3]
tok = Fe.SmilesTokenizer(cfg["data"]["token_level"]).fit(df["smiles"])
X = tok.encode(df["smiles"]); L = tok.lengths(X); Y = df[tasks].values.astype(float)
E = Fe.ecfp(df["smiles"].tolist())
base = dict(cfg["dl"]["base"]); dev = cfg["dl"]["device"]
p = resolve(cfg["multitask"]["dl_cfg_from"])
if p.exists():
    base.update(json.load(open(p))["config"]); print("DL config from", p)
if args.quick:
    base.update(max_epochs=3, patience=2)
models = cfg["multitask"]["models"]; crit = cfg["eval"]["threshold_criterion"]
seeds = cfg["eval"]["seeds"][:1] if args.quick else cfg["eval"]["seeds"]
out = "results/all_tasks.csv"
for seed in seeds:
    # one shared fold assignment per seed (stratified on the main task) so tasks are comparable
    all_cols = [c for c in df.columns if c not in ("smiles", "smiles_raw", "length")]
    main = find_task(all_cols, cfg["data"]["task"]) if cfg["data"]["task"] else all_cols[0]
    folds = D.stratified_folds(df[main].values.astype(int), cfg["eval"]["n_folds"], seed)
    for fold, (tr_all, te) in enumerate(folds):
        if args.quick and fold > 0:
            continue
        rows = []
        tr, va = D.train_val_split(tr_all, df[main].values.astype(int), cfg["eval"]["val_fraction"], seed)
        if "dl_multilabel" in models:
            model, info = train_model(X[tr], L[tr], Y[tr], X[va], L[va], Y[va], tok.vocab_size, base, seed, dev)
            P_va, P_te = predict_proba(model, X[va], L[va]), predict_proba(model, X[te], L[te])
        for ti, t in enumerate(tasks):
            y = Y[:, ti].astype(int); b = dict(task=t, seed=seed, fold=fold, pos_rate=float(y.mean()))
            if len(np.unique(y[te])) < 2 or y[tr].sum() < 5:
                continue
            if "majority" in models:
                rows.append({**b, "model": "majority", **M.majority_baseline(y[tr_all], y[te])})
            if "rf_ecfp4" in models:
                _, pv = fit_predict("rf", E[tr], y[tr], E[va], seed); _, pt = fit_predict("rf", E[tr_all], y[tr_all], E[te], seed)
                rows.append({**b, "model": "rf_ecfp4", **eval_with_threshold(y[va], pv, y[te], pt, crit)})
            if "dl_single" in models:
                m1, _ = train_model(X[tr], L[tr], y[tr], X[va], L[va], y[va], tok.vocab_size, base, seed, dev)
                pv, pt = predict_proba(m1, X[va], L[va])[:, 0], predict_proba(m1, X[te], L[te])[:, 0]
                rows.append({**b, "model": "dl_single", **eval_with_threshold(y[va], pv, y[te], pt, crit)})
            if "dl_multilabel" in models:
                rows.append({**b, "model": "dl_multilabel", **eval_with_threshold(y[va], P_va[:, ti], y[te], P_te[:, ti], crit)})
            print(f"s{seed} f{fold} {t[:40]:40s} " + " ".join(f"{r['model']}={r['auroc']:.3f}" for r in rows if r["task"] == t))
        append_rows(out, rows)
print("saved ->", out)
