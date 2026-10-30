import os, random, json, time
from pathlib import Path
import numpy as np
import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path=None) -> dict:
    path = Path(path or os.environ.get("ADR_CONFIG") or project_root() / "config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else project_root() / p


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def find_task(columns, query: str) -> str:
    """Case-insensitive substring match of a SIDER task column."""
    q = query.lower()
    hits = [c for c in columns if q in c.lower()]
    if not hits:
        raise KeyError(f"Task '{query}' not found. Available: {list(columns)}")
    return hits[0]


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter(); return self
    def __exit__(self, *a):
        self.elapsed = time.perf_counter() - self.t0


def append_rows(csv_path, rows: list):
    """Append list-of-dict rows to a CSV (creates header if new)."""
    import pandas as pd
    csv_path = resolve(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if csv_path.exists():
        cols = list(pd.read_csv(csv_path, nrows=0).columns)
        if set(df.columns) <= set(cols):
            df = df.reindex(columns=cols)                       # same schema -> plain append
        else:                                                   # new columns -> rewrite with the union
            old = pd.read_csv(csv_path)
            pd.concat([old, df], ignore_index=True).to_csv(csv_path, index=False); return
        df.to_csv(csv_path, mode="a", index=False, header=False)
    else:
        df.to_csv(csv_path, index=False)


def dump_json(obj, path):
    path = resolve(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
