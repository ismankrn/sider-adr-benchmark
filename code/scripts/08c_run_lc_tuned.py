"""Step 8c: add the PSO-selected CL configuration (best search seed by validation AUROC) as a fixed model
to the learning curves of SIDER and Tox21 (same paired subsamples as scripts 08/09) -> rows 'cl_tuned_pso' in lc_<dataset>.csv"""
import argparse, json
from _common import *
from adr.lc import lc_plan
from adr.models_dl import train_model, predict_proba
ap = argparse.ArgumentParser(); ap.add_argument("--datasets", nargs="*", default=["sider", "tox21"]); ap.add_argument("--method", default="pso"); ap.add_argument("--quick", action="store_true"); args = ap.parse_args()
cfg = load_config(); lc = cfg["lc"]; crit = cfg["eval"]["threshold_criterion"]; dev = cfg["dl"]["device"]
ht = pd.read_csv(resolve("results/hpo_trace.csv")); hf = pd.read_csv(resolve("results/hpo_final.csv")); hf = hf[hf.split == "stratified"]
t = ht[ht.method == args.method]; best_seed = int(t.loc[t.best_so_far.idxmin(), "seed"])
best_cfg = json.loads(hf[(hf.method == args.method) & (hf.hpo_seed == best_seed)].config.iloc[0])
print(f"using {args.method} configuration from search seed {best_seed}: {best_cfg}")
if args.quick:
    best_cfg = {**best_cfg, "max_epochs": 3, "patience": 2}
for ds in args.datasets:
    df, task = load_processed(cfg, ds)
    tok = Fe.SmilesTokenizer(cfg["data"]["token_level"]).fit(df["smiles"]); X = tok.encode(df["smiles"]); L = tok.lengths(X); y = df[task].values
    sizes, seeds = lc["sizes"], lc["seeds"]
    if args.quick:
        sizes, seeds = [100, 300], seeds[:1]
    out = f"results/lc_{ds}.csv"
    for p in lc_plan(df, task, sizes, seeds, lc["n_folds"], cfg["eval"]["val_fraction"]):
        if args.quick and p["fold"] > 0:
            continue
        tr, va, te = p["tr"], p["va"], p["te"]
        mdl, info = train_model(X[tr], L[tr], y[tr], X[va], L[va], y[va], tok.vocab_size, best_cfg, p["seed"], dev)
        pv, pt = predict_proba(mdl, X[va], L[va])[:, 0], predict_proba(mdl, X[te], L[te])[:, 0]
        row = dict(dataset=ds, task=task, seed=p["seed"], fold=p["fold"], n=p["n"], n_train=p["n_train"], model=f"cl_tuned_{args.method}",
                   train_time=info["train_time"], **eval_with_threshold(y[va], pv, y[te], pt, crit))
        append_rows(out, [row]); print(f"{ds} s{p['seed']} f{p['fold']} n={p['n']:>5} cl_tuned_{args.method}={row['auroc']:.3f}")
print("done")
