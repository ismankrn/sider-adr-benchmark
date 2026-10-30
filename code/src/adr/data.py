"""Dataset loading, SMILES standardisation, and splitting (stratified k-fold + scaffold)."""
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.model_selection import StratifiedKFold, train_test_split

from .utils import resolve

RDLogger.DisableLog("rdApp.*")
_LARGEST = rdMolStandardize.LargestFragmentChooser()
_UNCHARGER = rdMolStandardize.Uncharger()


def download_sider(url: str, dest) -> Path:
    """Download the MoleculeNet SIDER csv.gz (1,427 drugs x 27 organ-class tasks)."""
    import urllib.request
    dest = resolve(dest); dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"Downloading {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)
    return dest


def load_raw(path) -> pd.DataFrame:
    path = resolve(path)
    df = pd.read_csv(path)
    smi_col = [c for c in df.columns if c.lower() == "smiles"][0]
    df = df.rename(columns={smi_col: "smiles"})
    keep = ["smiles"] + [c for c in df.columns if c != "smiles" and pd.api.types.is_numeric_dtype(df[c])]
    return df[keep]          # drops id columns such as Tox21's mol_id


def standardize_smiles(smi: str, keep_largest_fragment=True, neutralize=True):
    """Canonical SMILES after optional salt stripping + neutralisation. Returns None if invalid."""
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    try:
        if keep_largest_fragment:
            mol = _LARGEST.choose(mol)
        if neutralize:
            mol = _UNCHARGER.uncharge(mol)
        smi_out = Chem.MolToSmiles(mol, canonical=True)
        # the uncharger can emit strings RDKit itself cannot re-parse (e.g. metal hydrides) -> treat as invalid
        return smi_out if Chem.MolFromSmiles(smi_out) is not None else None
    except Exception:
        return None


def prepare_dataset(raw_df: pd.DataFrame, keep_largest_fragment=True, neutralize=True,
                    drop_duplicates=True) -> tuple[pd.DataFrame, dict]:
    """Return cleaned dataframe (smiles + task columns) and a preprocessing report."""
    tasks = [c for c in raw_df.columns if c != "smiles"]
    df = raw_df.copy()
    df["smiles_raw"] = df["smiles"]
    df["smiles"] = df["smiles_raw"].map(lambda s: standardize_smiles(s, keep_largest_fragment, neutralize))
    n_raw = len(df)
    df = df.dropna(subset=["smiles"])
    n_valid = len(df)
    if drop_duplicates:
        # if duplicates disagree on labels, keep the majority vote via mean>=0.5
        agg = {t: "mean" for t in tasks}
        agg["smiles_raw"] = "first"
        df = df.groupby("smiles", as_index=False).agg(agg)
        for t in tasks:      # keep NaN (unlabelled) as NaN; majority vote otherwise
            df[t] = df[t].where(df[t].isna(), (df[t] >= 0.5).astype(float))
    n_final = len(df)
    df["length"] = df["smiles"].str.len()
    report = {"n_raw": n_raw, "n_valid": n_valid, "n_after_dedup": n_final,
              "max_len": int(df["length"].max()), "tasks": tasks,
              "positives_per_task": {t: int(df[t].sum()) for t in tasks},
              "labelled_per_task": {t: int(df[t].notna().sum()) for t in tasks}}
    return df[["smiles", "smiles_raw", "length"] + tasks], report


# ---------------------------------------------------------------- splits
def stratified_folds(y, n_splits=5, seed=0):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


def train_val_split(train_idx, y, val_fraction=0.15, seed=0):
    tr, va = train_test_split(train_idx, test_size=val_fraction, random_state=seed,
                              stratify=y[train_idx])
    return np.asarray(tr), np.asarray(va)


def murcko_scaffold(smi: str) -> str:
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smi, includeChirality=False)
    except Exception:
        return ""


def scaffold_split(smiles, frac_train=0.8, frac_valid=0.1, seed=None):
    """Bemis-Murcko scaffold split. seed=None -> deterministic MoleculeNet-style (largest scaffolds
    to train); seed=int -> randomised scaffold assignment (for repeated runs)."""
    groups = defaultdict(list)
    for i, s in enumerate(smiles):
        groups[murcko_scaffold(s)].append(i)
    sets = list(groups.values())
    if seed is None:
        sets.sort(key=lambda g: (len(g), g[0]), reverse=True)
    else:
        rng = np.random.RandomState(seed); rng.shuffle(sets)
    n = len(smiles); n_tr, n_va = frac_train * n, (frac_train + frac_valid) * n
    tr, va, te = [], [], []
    for g in sets:
        if len(tr) + len(g) <= n_tr:
            tr += g
        elif len(tr) + len(va) + len(g) <= n_va:
            va += g
        else:
            te += g
    return np.array(tr), np.array(va), np.array(te)


def scaffold_folds(smiles, n_splits=5, seed=0):
    """K-fold over scaffold groups (GroupKFold with shuffled groups) -> (train_idx, test_idx) list."""
    from sklearn.model_selection import GroupKFold
    scaf = np.array([murcko_scaffold(s) for s in smiles])
    uniq = np.unique(scaf)
    rng = np.random.RandomState(seed)
    remap = dict(zip(uniq, rng.permutation(len(uniq))))
    g = np.array([remap[s] for s in scaf])
    return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(smiles)), groups=g))


def make_folds(df, task, split="stratified", n_splits=5, seed=0):
    y = df[task].values
    if split == "stratified":
        return stratified_folds(y, n_splits, seed)
    if split == "scaffold":
        return scaffold_folds(df["smiles"].tolist(), n_splits, seed)
    raise ValueError(split)


def dataset_cfg(cfg: dict, name: str = "sider") -> dict:
    """Return {raw_url, raw_path, processed_path, task, ...} for a dataset name."""
    if name == "sider":
        return cfg["data"]
    d = dict(cfg["data"]); d.update(cfg["datasets"][name]); return d
