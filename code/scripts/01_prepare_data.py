"""Step 1: download SIDER (MoleculeNet), standardise SMILES, de-duplicate, report class balance."""
import argparse
from _common import *
ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="sider"); args = ap.parse_args()
cfg = load_config()
d = D.dataset_cfg(cfg, args.dataset)
raw = D.load_raw(D.download_sider(d["raw_url"], d["raw_path"]))
df, rep = D.prepare_dataset(raw, d["keep_largest_fragment"], d["neutralize"])
from adr.features import tokenize
ntok = df["smiles"].map(lambda s: len(tokenize(s, d["token_level"])))
cut = d.get("max_tokens")
if cut:
    rep["n_removed_too_long"] = int((ntok > cut).sum()); rep["max_tokens"] = cut
    df = df[ntok <= cut].reset_index(drop=True); rep["n_final"] = len(df)
    rep["max_len"] = int(df["length"].max())
out = resolve(d["processed_path"]); out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
dump_json(rep, f"results/data_report{'' if args.dataset == 'sider' else '_' + args.dataset}.json")
task = find_task(rep["tasks"], d["task"])
lab = df[task].dropna(); n1 = int(lab.sum()); n0 = len(lab) - n1
print(f"raw={rep['n_raw']} valid={rep['n_valid']} unique={rep['n_after_dedup']} max_len={rep['max_len']}")
print(f"task='{task}': labelled={len(lab)} class1={n1} class0={n0} (pos rate {n1/len(lab):.3f})")
print(f"Trivial all-positive F1 = {2*n1/(len(lab)+n1):.4f}  <- every model must beat this")
