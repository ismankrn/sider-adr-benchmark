"""Step 4b: re-evaluate every configuration selected in the optimiser benchmark (results/hpo_final.csv)
on the SCAFFOLD split with the same protocol (5 folds x 3 seeds), appending rows with split='scaffold'."""
import argparse, json
from _common import *
from adr.models_dl import train_model, predict_proba
ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); ap.add_argument("--methods", nargs="*"); args = ap.parse_args()
cfg = load_config(); df, task = load_processed(cfg)
tok = Fe.SmilesTokenizer(cfg["data"]["token_level"]).fit(df["smiles"])
X = tok.encode(df["smiles"]); L = tok.lengths(X); y = df[task].values
dev = cfg["dl"]["device"]; crit = cfg["eval"]["threshold_criterion"]
hf = pd.read_csv(resolve("results/hpo_final.csv"))
runs = hf[hf.split == "stratified"].drop_duplicates(["scheme", "method", "hpo_seed"])
if args.methods:
    runs = runs[runs.method.isin(args.methods)]
done = hf[hf.split == "scaffold"][["scheme", "method", "hpo_seed", "seed", "fold"]].apply(tuple, axis=1).tolist() if "scaffold" in hf.split.values else []
out = "results/hpo_final.csv"
for _, r in runs.iterrows():
    best_cfg = json.loads(r.config)
    if args.quick:
        best_cfg = {**best_cfg, "max_epochs": 3, "patience": 2}
    rows = []
    for split, s2, fold, tr2, va2, te2 in iter_folds(cfg, df, task):
        if split != "scaffold" or (r.scheme, r.method, r.hpo_seed, s2, fold) in done or (args.quick and (fold > 0 or s2 > 0)):
            continue
        model, info = train_model(X[tr2], L[tr2], y[tr2], X[va2], L[va2], y[va2], tok.vocab_size, best_cfg, s2, dev)
        p_va, p_te = predict_proba(model, X[va2], L[va2])[:, 0], predict_proba(model, X[te2], L[te2])[:, 0]
        m = eval_with_threshold(y[va2], p_va, y[te2], p_te, crit)
        rows.append(dict(task=task, scheme=r.scheme, method=r.method, hpo_seed=int(r.hpo_seed), split=split, seed=s2, fold=fold,
                         budget=int(r.budget), search_time=r.search_time, config=r.config, **m))
    if rows:
        append_rows(out, rows)
        print(f"[{r.scheme}/{r.method}/seed{int(r.hpo_seed)}] scaffold CV AUROC = {np.mean([x['auroc'] for x in rows]):.4f} over {len(rows)} folds")
print("saved ->", out)
