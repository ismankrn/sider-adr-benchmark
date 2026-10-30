"""Aggregation, statistical tests and table/figure export."""
import numpy as np
import pandas as pd
from scipy import stats
from .metrics import METRIC_COLS


def summarize(df: pd.DataFrame, by, metrics=METRIC_COLS) -> pd.DataFrame:
    g = df.groupby(by)[metrics]
    mean, std = g.mean(), g.std()
    out = mean.copy()
    for m in metrics:
        out[m] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(mean[m], std[m].fillna(0))]
    out["n"] = g.size()
    return out.reset_index()


def paired_test(df, by, a, b, metric="auroc", keys=("seed", "fold")):
    """Wilcoxon signed-rank (and paired t) between two levels of `by`, paired on seed+fold."""
    keys = [k for k in keys if k in df.columns]
    # average any replicates that share the same pairing keys (e.g. several HPO seeds per CV fold)
    x = df[df[by] == a].groupby(keys)[metric].mean()
    y = df[df[by] == b].groupby(keys)[metric].mean()
    common = x.index.intersection(y.index)
    x, y = x.loc[common].values, y.loc[common].values
    if len(x) < 3:
        return {"a": a, "b": b, "n": len(x)}
    w = stats.wilcoxon(x, y, zero_method="zsplit") if np.any(x != y) else None
    t = stats.ttest_rel(x, y)
    return {"a": a, "b": b, "metric": metric, "n": len(x), "mean_a": x.mean(), "mean_b": y.mean(),
            "diff": (x - y).mean(), "wilcoxon_p": w.pvalue if w else 1.0, "ttest_p": t.pvalue,
            "cohen_d": (x - y).mean() / ((x - y).std(ddof=1) or 1.0)}


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(lines)


def holm(pvals):
    """Holm step-down adjusted p-values (family-wise error control), same order as input."""
    p = np.asarray(pvals, float); m = len(p); order = np.argsort(p)
    adj = np.empty(m); running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i]); adj[i] = min(1.0, running)
    return adj
