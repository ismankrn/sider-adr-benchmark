"""Step 9: pretrained chemical language models (ChemBERTa-2, MoLFormer) and Chemprop D-MPNN.
  --mode primary   : primary task, full CV protocol (stratified + scaffold x seeds)  -> results/pretrained_primary.csv
  --mode multitask : one multi-task model over all task columns, stratified CV       -> results/pretrained_tasks.csv
  --mode lc        : learning curve on the dataset's LC task (paired with script 08)  -> results/lc_<dataset>.csv
"""
import argparse, json
from _common import *
from adr.lc import lc_plan
from adr.pretrained import finetune_lm, predict_lm, chemprop_fit_predict
ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="sider"); ap.add_argument("--mode", default="primary", choices=["primary", "multitask", "lc"])
ap.add_argument("--models", nargs="*", default=["chemberta", "chemprop"]); ap.add_argument("--quick", action="store_true"); args = ap.parse_args()
cfg = load_config(); df, task = load_processed(cfg, args.dataset); P = cfg["pretrained"]; crit = cfg["eval"]["threshold_criterion"]
tasks = task_columns(df); smiles = np.asarray(df["smiles"]); Y = df[tasks].values.astype(float)
quick = dict(max_epochs=2, patience=1) if args.quick else {}
qp = dict(epochs=5, patience=2) if args.quick else {}


def run_model(name, tr, va, te, targets):
    """Return (P_va, P_te) with shape (n, len(targets))."""
    ti = [tasks.index(t) for t in targets]
    if name in ("chemberta", "molformer"):
        c = {**P[name], **quick}
        mdl, tok, info = finetune_lm(c["hf_id"], smiles[tr], Y[np.ix_(tr, ti)], smiles[va], Y[np.ix_(va, ti)], c["max_len"], c["lr"],
                                     c["batch_size"], c["max_epochs"], c["patience"], seed=0, amp=c.get("amp", True))
        return predict_lm(mdl, tok, smiles[va], c["max_len"], amp=c.get("amp", True)), predict_lm(mdl, tok, smiles[te], c["max_len"], amp=c.get("amp", True)), info
    if name == "chemprop":
        c = {**P["chemprop"], **qp}
        return chemprop_fit_predict(smiles[tr], Y[np.ix_(tr, ti)], smiles[va], Y[np.ix_(va, ti)], smiles[te], targets,
                                    c["epochs"], c["patience"], c["hidden"], c["depth"])
    raise ValueError(name)


def score(rows_base, name, targets, Pva, Pte, va, te, info):
    rows = []
    for k, t in enumerate(targets):
        ti = tasks.index(t); yva, yte = Y[va, ti], Y[te, ti]; mva, mte = ~np.isnan(yva), ~np.isnan(yte)
        if mte.sum() == 0 or len(np.unique(yte[mte])) < 2:
            continue
        m = eval_with_threshold(yva[mva].astype(int), Pva[mva, k], yte[mte].astype(int), Pte[mte, k], crit)
        rows.append({**rows_base, "task": t, "model": name, "pos_rate": float(np.nanmean(Y[:, ti])), "train_time": info.get("train_time"), **m})
    return rows


if args.mode == "primary":
    out = "results/pretrained_primary.csv"
    for split, seed, fold, tr, va, te in iter_folds(cfg, df.assign(**{task: df[task].fillna(0)}), task):
        if args.quick and (fold > 0 or seed > 0):
            continue
        lab = lambda ix: ix[~np.isnan(Y[ix, tasks.index(task)])]; tr, va, te = lab(tr), lab(va), lab(te)
        for name in args.models:
            Pva, Pte, info = run_model(name, tr, va, te, [task])
            rows = score(dict(dataset=args.dataset, split=split, seed=seed, fold=fold), name, [task], Pva, Pte, va, te, info)
            append_rows(out, rows); print(f"{split} s{seed} f{fold} {name:10s} AUROC={rows[0]['auroc']:.3f} MCC={rows[0]['mcc']:.3f} t={info['train_time']:.0f}s")

elif args.mode == "multitask":
    out = "results/pretrained_tasks.csv"; y0 = df[task].fillna(0).astype(int).values
    for seed in (cfg["eval"]["seeds"][:1] if args.quick else cfg["eval"]["seeds"]):
        for fold, (tr_all, te) in enumerate(D.stratified_folds(y0, cfg["eval"]["n_folds"], seed)):
            if args.quick and fold > 0:
                continue
            tr, va = D.train_val_split(tr_all, y0, cfg["eval"]["val_fraction"], seed)
            for name in args.models:
                Pva, Pte, info = run_model(name, tr, va, te, tasks)
                rows = score(dict(dataset=args.dataset, seed=seed, fold=fold), name + "_multitask", tasks, Pva, Pte, va, te, info)
                append_rows(out, rows); print(f"s{seed} f{fold} {name:10s} mean AUROC over {len(rows)} tasks = {np.mean([r['auroc'] for r in rows]):.3f} t={info['train_time']:.0f}s")

else:  # learning curve
    lc = cfg["lc"]; out = f"results/lc_{args.dataset}.csv"; sizes, seeds = lc["sizes"], lc["seeds"]
    if args.quick:
        sizes, seeds = [100, 300], seeds[:1]
    for p in lc_plan(df, task, sizes, seeds, lc["n_folds"], cfg["eval"]["val_fraction"]):
        if args.quick and p["fold"] > 0:
            continue
        for name in args.models:
            Pva, Pte, info = run_model(name, p["tr"], p["va"], p["te"], [task])
            rows = score(dict(dataset=args.dataset, seed=p["seed"], fold=p["fold"], n=p["n"], n_train=p["n_train"]), name, [task], Pva, Pte, p["va"], p["te"], info)
            append_rows(out, rows); print(f"{args.dataset} s{p['seed']} f{p['fold']} n={p['n']:>5} {name}={rows[0]['auroc']:.3f} t={info['train_time']:.0f}s")
print("saved ->", out)
