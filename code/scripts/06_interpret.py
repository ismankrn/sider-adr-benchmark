"""Step 6: train the best configuration on fold 0 and explain test predictions (Integrated Gradients)."""
import argparse, json
from _common import *
from adr.models_dl import train_model, predict_proba
from adr.interpret import integrated_gradients, draw_attribution
from adr.features import tokenize
ap = argparse.ArgumentParser(); ap.add_argument("--top", type=int, default=10); ap.add_argument("--quick", action="store_true")
args = ap.parse_args()
cfg = load_config(); df, task = load_processed(cfg)
tok = Fe.SmilesTokenizer(cfg["data"]["token_level"]).fit(df["smiles"])
X = tok.encode(df["smiles"]); L = tok.lengths(X); y = df[task].values
base = dict(cfg["dl"]["base"]); p = resolve("results/hpo_best_config.json")
if p.exists():
    base.update(json.load(open(p))["config"])
if args.quick:
    base.update(max_epochs=3, patience=2)
tr_all, te = D.stratified_folds(y, cfg["eval"]["n_folds"], 0)[0]
tr, va = D.train_val_split(tr_all, y, cfg["eval"]["val_fraction"], 0)
model, _ = train_model(X[tr], L[tr], y[tr], X[va], L[va], y[va], tok.vocab_size, base, 0, cfg["dl"]["device"])
pt = predict_proba(model, X[te], L[te])[:, 0]
order = te[np.argsort(-pt)]            # most confident positives first
figdir = resolve("figures/attribution"); figdir.mkdir(parents=True, exist_ok=True)
rows, token_imp = [], {}
for rank, i in enumerate(order[: args.top]):
    toks = tokenize(df["smiles"][i], cfg["data"]["token_level"])
    attr = integrated_gradients(model, X[i], L[i])
    for t, a in zip(toks, attr):
        token_imp.setdefault(t, []).append(float(a))
    draw_attribution(df["smiles"][i], toks, attr, figdir / f"rank{rank:02d}_idx{i}.png")
    rows.append(dict(rank=rank, index=int(i), smiles=df["smiles"][i], label=int(y[i]), prob=float(pt[list(te).index(i)]),
                     top_tokens=json.dumps(sorted(zip(toks, map(float, attr)), key=lambda z: -abs(z[1]))[:8])))
pd.DataFrame(rows).to_csv(resolve("results/attributions_top.csv"), index=False)
imp = pd.DataFrame([(t, np.mean(v), len(v)) for t, v in token_imp.items()], columns=["token", "mean_attr", "count"]).sort_values("mean_attr")
imp.to_csv(resolve("results/token_importance.csv"), index=False)
print(imp.tail(10)); print("figures ->", figdir)
