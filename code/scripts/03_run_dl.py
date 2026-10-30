"""Step 3: sequence models. (a) representation ablation index/onehot/embedding, (b) 12 untuned
architectures L1..CL3D, (c) extra variants (BiLSTM, GRU, CNN-GRU). Full CV protocol."""
import argparse
from _common import *
from adr.models_dl import train_model, predict_proba, config_from_name
ap = argparse.ArgumentParser(); ap.add_argument("--part", default="all", choices=["all", "ablation", "arch", "extra"])
ap.add_argument("--quick", action="store_true"); args = ap.parse_args()
cfg = load_config(); df, task = load_processed(cfg)
tok = Fe.SmilesTokenizer(cfg["data"]["token_level"]).fit(df["smiles"])
X = tok.encode(df["smiles"]); L = tok.lengths(X); y = df[task].values
base = dict(cfg["dl"]["base"]); dev = cfg["dl"]["device"]
if args.quick:
    base.update(max_epochs=3, patience=2)

runs = []   # (group, name, cfg)
if args.part in ("all", "ablation"):
    for mode in cfg["dl"]["input_modes"]:
        runs.append(("ablation", f"CL1-{mode}", config_from_name("CL1", {**base, "input_mode": mode})))
if args.part in ("all", "arch"):
    for a in cfg["dl"]["architectures"]:
        runs.append(("arch", a, config_from_name(a, base)))
if args.part in ("all", "extra"):
    for v in cfg["dl"]["extra_variants"]:
        v = dict(v); name = v.pop("name"); runs.append(("extra", name, {**base, **v}))

out = "results/dl_untuned.csv"
for split, seed, fold, tr, va, te in iter_folds(cfg, df, task):
    if args.quick and (fold > 0 or seed > 0):
        continue
    rows = []
    for group, name, c in runs:
        model, info = train_model(X[tr], L[tr], y[tr], X[va], L[va], y[va], tok.vocab_size, c, seed, dev)
        p_va, p_te = predict_proba(model, X[va], L[va])[:, 0], predict_proba(model, X[te], L[te])[:, 0]
        m = eval_with_threshold(y[va], p_va, y[te], p_te, cfg["eval"]["threshold_criterion"])
        rows.append(dict(task=task, split=split, seed=seed, fold=fold, group=group, model=name,
                         input_mode=c["input_mode"], scheme=c["scheme"], n_layers=len(c["rnn_units"]),
                         epochs=info["epochs"], train_time=info["train_time"], val_loss=info["best_val_loss"], **m))
        print(f"{split} s{seed} f{fold} {name:16s} AUROC={m['auroc']:.3f} AUPRC={m['auprc']:.3f} MCC={m['mcc']:.3f} "
              f"F1={m['f1']:.3f} valloss={info['best_val_loss']:.3f} ep={info['epochs']}")
    append_rows(out, rows)
print("saved ->", out)
