"""Step 4: HPO benchmark. Each method (random / TPE / MBO / GA / PSO) gets the SAME budget of model
trainings on the SAME dev split. The best config of each run is then re-evaluated with k-fold CV."""
import argparse, json
from _common import *
from adr.models_dl import train_model, predict_proba
from adr.hpo import Objective, run_hpo
ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true")
ap.add_argument("--methods", nargs="*"); args = ap.parse_args()
cfg = load_config(); df, task = load_processed(cfg); h = cfg["hpo"]
tok = Fe.SmilesTokenizer(cfg["data"]["token_level"]).fit(df["smiles"])
X = tok.encode(df["smiles"]); L = tok.lengths(X); y = df[task].values
base = dict(cfg["dl"]["base"]); dev = cfg["dl"]["device"]
budget, methods, seeds = h["budget"], args.methods or h["methods"], h["seeds"]
if args.quick:
    base.update(max_epochs=3, patience=2); budget = 6; seeds = seeds[:1]

# dev split for the search: fold 0 of a stratified split (test part is NEVER seen by the search)
folds = D.stratified_folds(y, cfg["eval"]["n_folds"], h["hpo_split_seed"])
dev_tr, dev_te = folds[0]
tr, va = D.train_val_split(dev_tr, y, cfg["eval"]["val_fraction"], h["hpo_split_seed"])
data = (X[tr], L[tr], y[tr], X[va], L[va], y[va])

best_overall = (np.inf, None, None)
for scheme in h["schemes"]:
    for method in methods:
        for seed in seeds:
            set_seed(seed)
            obj = Objective(data, tok.vocab_size, scheme, base, seed=seed, device=dev)
            fit, best_cfg = run_hpo(method, obj, budget, seed, population_size=h["population_size"], **h.get("mbo", {}))
            log = pd.DataFrame(obj.log); log.insert(0, "scheme", scheme); log.insert(0, "seed", seed); log.insert(0, "method", method)
            append_rows("results/hpo_trace.csv", log.to_dict("records"))
            print(f"[{scheme}/{method}/seed{seed}] best val AUROC={1-fit:.4f} after {len(log)} evals, {log['elapsed'].iloc[-1]:.0f}s")
            # --- final unbiased evaluation of the found configuration: k-fold CV x seeds
            rows = []
            for split, s2, fold, tr2, va2, te2 in iter_folds(cfg, df, task):
                if split != "stratified" or (args.quick and (fold > 0 or s2 > 0)):
                    continue
                model, info = train_model(X[tr2], L[tr2], y[tr2], X[va2], L[va2], y[va2], tok.vocab_size, best_cfg, s2, dev)
                p_va, p_te = predict_proba(model, X[va2], L[va2])[:, 0], predict_proba(model, X[te2], L[te2])[:, 0]
                m = eval_with_threshold(y[va2], p_va, y[te2], p_te, cfg["eval"]["threshold_criterion"])
                rows.append(dict(task=task, scheme=scheme, method=method, hpo_seed=seed, split=split, seed=s2, fold=fold,
                                 budget=budget, search_time=log["elapsed"].iloc[-1], config=json.dumps(best_cfg), **m))
            append_rows("results/hpo_final.csv", rows)
            mean_auroc = np.mean([r["auroc"] for r in rows])
            print(f"   -> CV test AUROC of tuned {scheme}: {mean_auroc:.4f}")
            if fit < best_overall[0]:
                best_overall = (fit, best_cfg, f"{scheme}/{method}/seed{seed}")
dump_json({"source": best_overall[2], "val_auroc": 1 - best_overall[0], "config": best_overall[1]}, "results/hpo_best_config.json")
print("saved -> results/hpo_trace.csv, results/hpo_final.csv, results/hpo_best_config.json")
