"""Step 8: learning curves (test AUROC vs. training-set size) for fingerprint models and the embedding-LSTM,
on SIDER and Tox21. Shares subsamples with script 09 through adr.lc so curves are paired."""
import argparse
from _common import *
from adr.lc import lc_plan
from adr.baselines import fit_predict
from adr.models_dl import train_model, predict_proba, config_from_name
ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="tox21"); ap.add_argument("--models", nargs="*")
ap.add_argument("--quick", action="store_true"); args = ap.parse_args()
cfg = load_config(); df, task = load_processed(cfg, args.dataset); lc = cfg["lc"]
models = args.models or lc["models"]; sizes, seeds = lc["sizes"], lc["seeds"]
if args.quick:
    sizes, seeds = [100, 300], seeds[:1]
smiles = df["smiles"].tolist(); y = df[task].values
E = Fe.ecfp(smiles) if any(m in models for m in ("xgb", "rf")) else None
tok = Fe.SmilesTokenizer(cfg["data"]["token_level"]).fit(df["smiles"]); X = tok.encode(df["smiles"]); L = tok.lengths(X)
base = config_from_name("CL1", dict(cfg["dl"]["base"])); dev = cfg["dl"]["device"] if cfg["dl"]["device"] != "cpu" else "auto"
if args.quick:
    base.update(max_epochs=3, patience=2)
out = f"results/lc_{args.dataset}.csv"; crit = cfg["eval"]["threshold_criterion"]
for p in lc_plan(df, task, sizes, seeds, lc["n_folds"], cfg["eval"]["val_fraction"]):
    if args.quick and p["fold"] > 0:
        continue
    tr, va, te = p["tr"], p["va"], p["te"]; trva = np.concatenate([tr, va]); rows = []
    b = dict(dataset=args.dataset, task=task, seed=p["seed"], fold=p["fold"], n=p["n"], n_train=p["n_train"])
    rows.append({**b, "model": "majority", **M.majority_baseline(y[trva].astype(int), y[te].astype(int))})
    for m in ("xgb", "rf"):
        if m in models:
            _, pv = fit_predict(m, E[tr], y[tr].astype(int), E[va], p["seed"]); _, pt = fit_predict(m, E[trva], y[trva].astype(int), E[te], p["seed"])
            rows.append({**b, "model": f"{m}_ecfp4", **eval_with_threshold(y[va], pv, y[te], pt, crit)})
    if "lstm" in models:
        mdl, info = train_model(X[tr], L[tr], y[tr], X[va], L[va], y[va], tok.vocab_size, base, p["seed"], dev)
        pv, pt = predict_proba(mdl, X[va], L[va])[:, 0], predict_proba(mdl, X[te], L[te])[:, 0]
        rows.append({**b, "model": "lstm_emb", "train_time": info["train_time"], **eval_with_threshold(y[va], pv, y[te], pt, crit)})
    append_rows(out, rows)
    print(f"{args.dataset} s{p['seed']} f{p['fold']} n={p['n']:>5} " + " ".join(f"{r['model']}={r['auroc']:.3f}" for r in rows))
print("saved ->", out)
