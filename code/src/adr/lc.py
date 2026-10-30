"""Learning-curve protocol shared by scripts 08 and 09 so that every model sees identical subsamples.

For a task: keep only labelled molecules -> stratified K-fold (seed) -> for each fold and size n:
  train_sub = stratified subsample of the training fold (seed, n)   [n >= fold size -> whole fold, tag 'all']
  val       = 15 % of train_sub (early stopping / threshold),  test = the full held-out fold.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


def labelled_index(df, task):
    return np.where(df[task].notna().values)[0]


def lc_plan(df, task, sizes, seeds, n_folds=5, val_fraction=0.15):
    """Yield dicts {seed, fold, n, n_label, tr, va, te} with absolute row indices into df."""
    idx = labelled_index(df, task); y = df[task].values[idx].astype(int)
    for seed in seeds:
        skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
        for fold, (tr_rel, te_rel) in enumerate(skf.split(np.zeros(len(idx)), y)):
            done_all = False
            for n in sorted(sizes):
                if n >= len(tr_rel):
                    if done_all:
                        continue
                    sub, label, done_all = tr_rel, "all", True
                else:
                    sub, _ = train_test_split(tr_rel, train_size=n, random_state=seed + 1000 * fold, stratify=y[tr_rel])
                    label = str(n)
                tr, va = train_test_split(sub, test_size=val_fraction, random_state=seed, stratify=y[sub])
                yield dict(seed=seed, fold=fold, n=label, n_train=len(sub), tr=idx[tr], va=idx[va], te=idx[te_rel])
