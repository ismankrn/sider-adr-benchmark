import sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)
import numpy as np, pandas as pd
from adr.utils import load_config, resolve, set_seed, find_task, append_rows, dump_json
from adr import data as D, features as Fe, metrics as M


def load_processed(cfg, dataset="sider"):
    d = D.dataset_cfg(cfg, dataset)
    df = pd.read_csv(resolve(d["processed_path"]))
    task = find_task([c for c in df.columns if c not in ("smiles", "smiles_raw", "length")], d["task"])
    return df, task


def task_columns(df):
    return [c for c in df.columns if c not in ("smiles", "smiles_raw", "length")]


def iter_folds(cfg, df, task):
    """Yield (split, seed, fold, train_idx, val_idx, test_idx) for the full evaluation protocol."""
    y = df[task].values
    for split in cfg["eval"]["splits"]:
        for seed in cfg["eval"]["seeds"]:
            for fold, (tr, te) in enumerate(D.make_folds(df, task, split, cfg["eval"]["n_folds"], seed)):
                tr, va = D.train_val_split(tr, y, cfg["eval"]["val_fraction"], seed)
                yield split, seed, fold, tr, va, te


def eval_with_threshold(y_va, p_va, y_te, p_te, criterion):
    thr = M.best_threshold(y_va, p_va, criterion)
    m05 = M.compute_metrics(y_te, p_te, 0.5)
    mt = M.compute_metrics(y_te, p_te, thr)
    return {**m05, **{f"{k}_tuned": v for k, v in mt.items() if k in M.METRIC_COLS}, "threshold_tuned": thr}
