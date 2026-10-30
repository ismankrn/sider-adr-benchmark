"""Record per-epoch training and validation loss of the CL1 network under the three input encodings
(fold 0 of the stratified split, three seeds) -> results/training_curves.csv"""
import argparse
from _common import *
from adr.models_dl import train_model, config_from_name
ap = argparse.ArgumentParser(); ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2]); ap.add_argument("--quick", action="store_true"); args = ap.parse_args()
cfg = load_config(); df, task = load_processed(cfg)
tok = Fe.SmilesTokenizer(cfg["data"]["token_level"]).fit(df["smiles"]); X = tok.encode(df["smiles"]); L = tok.lengths(X); y = df[task].values
base = dict(cfg["dl"]["base"]); dev = cfg["dl"]["device"]
if args.quick:
    base.update(max_epochs=3, patience=2)
rows = []
for seed in args.seeds:
    tr_all, te = D.stratified_folds(y, cfg["eval"]["n_folds"], seed)[0]
    tr, va = D.train_val_split(tr_all, y, cfg["eval"]["val_fraction"], seed)
    for mode in cfg["dl"]["input_modes"]:
        c = config_from_name("CL1", {**base, "input_mode": mode})
        _, info = train_model(X[tr], L[tr], y[tr], X[va], L[va], y[va], tok.vocab_size, c, seed, dev)
        for h in info["history"]:
            rows.append(dict(encoding=mode, seed=seed, **h))
        print(f"seed {seed} {mode:10s} epochs={info['epochs']} best_val={info['best_val_loss']:.4f}")
pd.DataFrame(rows).to_csv(resolve("results/training_curves.csv"), index=False); print("saved -> results/training_curves.csv")
